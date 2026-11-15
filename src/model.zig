const std = @import("std");
const metal = @import("metal.zig");
const config = @import("config.zig");
const weights = @import("weights.zig");
const resident = @import("resident.zig");
const Weight = weights.Weight;
const Buffer = metal.Buffer;

const MLP = struct { gate: Weight, up: Weight, down: Weight };
const Layer = struct {
    input_norm: Weight,
    post_norm: Weight,
    q: Weight,
    k: Weight,
    v: Weight,
    o: Weight,
    q_norm: Weight,
    k_norm: Weight,
    sink: Weight,
    dense: ?MLP = null,
    experts: []MLP = &.{},
    shared: ?MLP = null,
    router: ?Weight = null,
    bias: ?Buffer = null,
    descriptors: Buffer = undefined,
    qkv_descriptors: Buffer = undefined,
    keys: Buffer,
    values: Buffer,
    capacity: u32,
};
const Scratch = struct {
    x: Buffer,
    norm: Buffer,
    q: Buffer,
    k: Buffer,
    v: Buffer,
    attention: Buffer,
    projection: Buffer,
    activation: Buffer,
    router: Buffer,
    route_ids: Buffer,
    route_weights: Buffer,
    route_error: Buffer,
    logits: Buffer,
};

/// Eager, single-sequence FP32-activation baseline; weights stay in mapped unified memory.
/// Sliding layers have bounded ring caches; full-attention layers retain the requested context.
pub const Model = struct {
    arena: *std.heap.ArenaAllocator,
    ctx: metal.Context,
    config_map: metal.Mapping,
    parsed: std.json.Parsed(config.Config),
    cfg: config.Config,
    store: weights.Store,
    buffers: std.ArrayList(Buffer) = .empty,
    layers: []Layer = &.{},
    embedding: Weight = undefined,
    final_norm: Weight = undefined,
    head: Weight = undefined,
    scratch: Scratch = undefined,
    last_execution: metal.Stats = .{ .submissions = 0, .dispatches = 0, .barriers = 0 },
    context: u32,
    position: u32 = 0,
    failed: bool = false,
    resident_execution: ?resident.Execution = null,
    staged_row_tile: u32 = 1,

    pub fn init(gpa: std.mem.Allocator, ctx: metal.Context, directory: []const u8, context: u32) !Model {
        const arena = try gpa.create(std.heap.ArenaAllocator);
        arena.* = std.heap.ArenaAllocator.init(gpa);
        errdefer {
            arena.deinit();
            gpa.destroy(arena);
        }
        const a = arena.allocator();
        const path = try std.fs.path.join(a, &.{ directory, "config.json" });
        const mapping = try metal.Mapping.open(a, path);
        errdefer mapping.deinit();
        const parsed = try config.parse(a, mapping.bytes);
        errdefer parsed.deinit();
        const cfg = parsed.value;
        if (context == 0 or context > cfg.max_position_embeddings) return error.InvalidContextLength;
        // The reference attention kernel keeps at most eight FP32 components per SIMD lane.
        if (cfg.head_dim > 256 or cfg.head_dim % 2 != 0) return error.UnsupportedHeadDimension;
        var self: Model = .{
            .arena = arena,
            .ctx = ctx,
            .config_map = mapping,
            .parsed = parsed,
            .cfg = cfg,
            .store = try weights.Store.init(a, ctx, directory, if (cfg.quantization) |q| q.group_size else 0),
            .context = context,
        };
        errdefer {
            for (self.buffers.items) |b| b.deinit();
            self.store.deinit();
        }
        self.embedding = try self.store.get("model.embed_tokens.weight", &.{ cfg.vocab_size, cfg.hidden_size });
        self.final_norm = try self.store.get("model.norm.weight", &.{cfg.hidden_size});
        self.head = try self.store.get("lm_head.weight", &.{ cfg.vocab_size, cfg.hidden_size });
        self.layers = try a.alloc(Layer, cfg.num_hidden_layers);
        for (self.layers, 0..) |*layer, i| {
            const cap = if (cfg.sliding(i)) @min(context, cfg.sliding_window) else context;
            const cache_elements = try std.math.mul(usize, cap, @as(usize, cfg.num_key_value_heads) * cfg.head_dim);
            layer.* = .{
                .input_norm = try self.layerWeight(i, "input_layernorm.weight", &.{cfg.hidden_size}),
                .post_norm = try self.layerWeight(i, "post_attention_layernorm.weight", &.{cfg.hidden_size}),
                .q = try self.layerWeight(i, "self_attn.q_proj.weight", &.{ cfg.num_attention_heads * cfg.head_dim, cfg.hidden_size }),
                .k = try self.layerWeight(i, "self_attn.k_proj.weight", &.{ cfg.num_key_value_heads * cfg.head_dim, cfg.hidden_size }),
                .v = try self.layerWeight(i, "self_attn.v_proj.weight", &.{ cfg.num_key_value_heads * cfg.head_dim, cfg.hidden_size }),
                .o = try self.layerWeight(i, "self_attn.o_proj.weight", &.{ cfg.hidden_size, cfg.num_attention_heads * cfg.head_dim }),
                .q_norm = try self.layerWeight(i, "self_attn.q_norm.weight", &.{cfg.head_dim}),
                .k_norm = try self.layerWeight(i, "self_attn.k_norm.weight", &.{cfg.head_dim}),
                .sink = try self.layerWeight(i, "self_attn.softmax_offset", &.{cfg.num_attention_heads}),
                .keys = try self.alloc(cache_elements),
                .values = try self.alloc(cache_elements),
                .capacity = cap,
            };
            if (cfg.dense(i)) {
                layer.dense = try self.loadMLP(i, "mlp", cfg.intermediate_size);
            } else {
                layer.router = try self.layerWeight(i, "mlp.gate.weight", &.{ cfg.num_experts, cfg.hidden_size });
                var name_buf: [192]u8 = undefined;
                layer.bias = try self.alloc(cfg.num_experts);
                @memset(layer.bias.?.floats(), 0);
                const bias_name = try std.fmt.bufPrint(&name_buf, "model.layers.{d}.mlp.gate.expert_bias", .{i});
                if (try self.store.find(bias_name)) |bias| {
                    if (!bias.tensor.dtype.isFloat() or !std.mem.eql(usize, bias.tensor.shape, &.{cfg.num_experts})) return error.WeightShapeMismatch;
                    for (layer.bias.?.floats(), 0..) |*value, j| value.* = bias.scalar(j);
                }
                layer.experts = try a.alloc(MLP, cfg.num_experts);
                const stacked_name = try std.fmt.bufPrint(&name_buf, "model.layers.{d}.mlp.switch_mlp.gate_proj.weight", .{i});
                if (try self.store.find(stacked_name) != null) {
                    const stacked = try self.loadStackedMLP(i);
                    for (layer.experts, 0..) |*expert, j| expert.* = .{ .gate = stacked.gate.expert(j), .up = stacked.up.expert(j), .down = stacked.down.expert(j) };
                } else {
                    for (layer.experts, 0..) |*expert, j| {
                        const prefix = try std.fmt.bufPrint(&name_buf, "mlp.experts.{d}", .{j});
                        expert.* = try self.loadMLP(i, prefix, cfg.moe_intermediate_size);
                    }
                }
                if (cfg.shared_expert_intermediate_size > 0)
                    layer.shared = try self.loadMLP(i, "mlp.shared_expert", cfg.shared_expert_intermediate_size);
            }
            layer.qkv_descriptors = try self.alloc(3 * @sizeOf(metal.Matrix) / 4);
            @memcpy(layer.qkv_descriptors.structs(metal.Matrix), &[_]metal.Matrix{ layer.q.matrix(), layer.k.matrix(), layer.v.matrix() });
            const count: usize = if (layer.dense != null) 1 else @as(usize, cfg.num_experts) + 1;
            layer.descriptors = try self.alloc(count * @sizeOf(metal.MLP) / @sizeOf(f32));
            const descriptors = layer.descriptors.structs(metal.MLP);
            if (layer.dense) |dense| {
                descriptors[0] = describe(dense);
            } else {
                for (layer.experts, 0..) |expert, j| descriptors[j] = describe(expert);
                descriptors[cfg.num_experts] = describe(layer.shared.?);
            }
        }
        const hidden = cfg.hidden_size;
        const qsize = cfg.num_attention_heads * cfg.head_dim;
        const kvsize = cfg.num_key_value_heads * cfg.head_dim;
        const intermediate = @max(@as(usize, cfg.intermediate_size), (@as(usize, cfg.num_experts_per_tok) + 1) * @max(cfg.moe_intermediate_size, cfg.shared_expert_intermediate_size));
        self.scratch = .{
            .x = try self.alloc(hidden),
            .norm = try self.alloc(hidden),
            .q = try self.alloc(qsize),
            .k = try self.alloc(kvsize),
            .v = try self.alloc(kvsize),
            .attention = try self.alloc(qsize),
            .projection = try self.alloc(hidden),
            .activation = try self.alloc(intermediate),
            .router = try self.alloc(cfg.num_experts),
            .route_ids = try self.alloc(cfg.num_experts_per_tok),
            .route_weights = try self.alloc(cfg.num_experts_per_tok),
            .route_error = try self.alloc(1),
            .logits = try self.alloc(cfg.vocab_size),
        };
        return self;
    }
    pub fn deinit(self: *Model) void {
        self.ctx.finish() catch {};
        for (self.buffers.items) |b| b.deinit();
        self.store.deinit();
        self.parsed.deinit();
        self.config_map.deinit();
        const gpa = self.arena.child_allocator;
        self.arena.deinit();
        gpa.destroy(self.arena);
    }
    fn alloc(self: *Model, count: usize) !Buffer {
        const buf = try self.ctx.alloc(count);
        errdefer buf.deinit();
        try self.buffers.append(self.arena.allocator(), buf);
        return buf;
    }
    fn layerWeight(self: *const Model, layer: usize, suffix: []const u8, shape: []const usize) !Weight {
        var buffer: [256]u8 = undefined;
        const name = try std.fmt.bufPrint(&buffer, "model.layers.{d}.{s}", .{ layer, suffix });
        return self.store.get(name, shape);
    }
    fn loadMLP(self: *const Model, layer: usize, prefix: []const u8, size: u32) !MLP {
        var buf: [192]u8 = undefined;
        const gate = try self.layerWeight(layer, try std.fmt.bufPrint(&buf, "{s}.gate_proj.weight", .{prefix}), &.{ size, self.cfg.hidden_size });
        const up = try self.layerWeight(layer, try std.fmt.bufPrint(&buf, "{s}.up_proj.weight", .{prefix}), &.{ size, self.cfg.hidden_size });
        const down = try self.layerWeight(layer, try std.fmt.bufPrint(&buf, "{s}.down_proj.weight", .{prefix}), &.{ self.cfg.hidden_size, size });
        return .{ .gate = gate, .up = up, .down = down };
    }
    fn loadStackedMLP(self: *const Model, layer: usize) !MLP {
        const c = self.cfg;
        return .{
            .gate = try self.layerWeight(layer, "mlp.switch_mlp.gate_proj.weight", &.{ c.num_experts, c.moe_intermediate_size, c.hidden_size }),
            .up = try self.layerWeight(layer, "mlp.switch_mlp.up_proj.weight", &.{ c.num_experts, c.moe_intermediate_size, c.hidden_size }),
            .down = try self.layerWeight(layer, "mlp.switch_mlp.down_proj.weight", &.{ c.num_experts, c.hidden_size, c.moe_intermediate_size }),
        };
    }
    fn matvec(self: *const Model, w: Weight, x: Buffer, out: Buffer) !void {
        var p = w.params();
        p.m = @intCast(w.tensor.shape[0]);
        p.n = @intCast(w.tensor.shape[1]);
        if (w.quant) |q| {
            const fast = q.bits == 4 and q.group_size == 64 and p.n % 64 == 0 and p.offset % 4 == 0;
            const tile = if (fast) self.staged_row_tile else 1;
            const kernel: [:0]const u8 = if (!fast) "qmatvec" else switch (tile) {
                2 => "staged_qmatvec_t2",
                4 => "staged_qmatvec_t4",
                8 => "staged_qmatvec_t8",
                else => "qmatvec4_g64",
            };
            try self.ctx.dispatch(kernel, &.{ w.buffer, q.scales, q.biases, x, out }, p, ((@as(usize, p.m) + tile - 1) / tile) * 32, 128);
        } else {
            const split = p.m <= 256 and p.n >= 512 and p.n <= resident.certified.dense_max_cols and p.dtype <= 2;
            const kernel: [:0]const u8 = if (split) switch (p.dtype) {
                0 => "dense_matvec_split_f32",
                1 => "dense_matvec_split_f16",
                2 => "dense_matvec_split_bf16",
                else => unreachable,
            } else switch (p.dtype) {
                0 => "dense_matvec_f32",
                1 => "dense_matvec_f16",
                2 => "dense_matvec_bf16",
                else => "matvec",
            };
            const group: usize = if (split) resident.certified.threadgroup_size else 128;
            const row_threads: usize = if (split) resident.certified.threadgroup_size else 32;
            try self.ctx.dispatch(kernel, &.{ w.buffer, x, out }, p, @as(usize, p.m) * row_threads, group);
        }
    }
    fn qkv(self: *const Model, layer: Layer) !void {
        const s = self.scratch;
        const matrices = layer.qkv_descriptors.structs(metal.Matrix);
        // Only combine the established INT4/g64 fast path. Other checkpoint
        // formats retain their existing dtype-specialized projections.
        if (matrices[0].reserved != 1 or matrices[1].reserved != 1 or matrices[2].reserved != 1) {
            try self.matvec(layer.q, s.norm, s.q);
            try self.matvec(layer.k, s.norm, s.k);
            try self.matvec(layer.v, s.norm, s.v);
            return;
        }
        const tile = self.staged_row_tile;
        var tiles: usize = 0;
        for (matrices) |matrix| tiles += (@as(usize, matrix.rows) + tile - 1) / tile;
        const kernel: [:0]const u8 = switch (tile) {
            2 => "staged_qkv_t2",
            4 => "staged_qkv_t4",
            8 => "staged_qkv_t8",
            else => "staged_qkv_t1",
        };
        try self.ctx.dispatch(kernel, &.{ layer.qkv_descriptors, s.norm, s.q, s.k, s.v }, .{}, tiles * 32, 128);
    }
    fn norm(self: *const Model, w: Weight, x: Buffer, out: Buffer, heads: u32, dim: u32) !void {
        var p = w.params();
        p.m = heads;
        p.n = dim;
        p.eps = @floatCast(self.cfg.rms_norm_eps);
        if (heads == 1 and dim >= 512 and dim <= 8192) {
            try self.ctx.dispatch("rmsnorm_parallel", &.{ w.buffer, x, out }, p, 256, 256);
        } else {
            try self.ctx.dispatch("rmsnorm", &.{ w.buffer, x, out }, p, @as(usize, heads) * 32, 32);
        }
    }
    fn add(self: *const Model, x: Buffer, y: Buffer, scale: f32) !void {
        try self.ctx.dispatch("add_scaled", &.{ x, y }, .{ .n = self.cfg.hidden_size, .scale = scale }, self.cfg.hidden_size, 128);
    }
    fn describe(mlp: MLP) metal.MLP {
        return .{ .gate = mlp.gate.matrix(), .up = mlp.up.matrix(), .down = mlp.down.matrix() };
    }
    fn fusedMLP(self: *const Model, layer: Layer, routed: u32, shared_index: u32, stride: u32) !void {
        const s = self.scratch;
        const p: metal.Params = .{ .n = routed, .m = shared_index, .k = stride, .dim = self.cfg.hidden_size };
        const tile = self.staged_row_tile;
        const gate: [:0]const u8 = switch (tile) {
            2 => "staged_gate_up_t2",
            4 => "staged_gate_up_t4",
            8 => "staged_gate_up_t8",
            else => "fused_gate_up",
        };
        const down: [:0]const u8 = switch (tile) {
            2 => "staged_down_t2",
            4 => "staged_down_t4",
            8 => "staged_down_t8",
            else => "fused_down_reduce",
        };
        try self.ctx.dispatch(gate, &.{ layer.descriptors, s.norm, s.route_ids, s.activation }, p, (@as(usize, routed) + 1) * ((@as(usize, stride) + tile - 1) / tile) * 32, 128);
        try self.ctx.dispatch(down, &.{ layer.descriptors, s.activation, s.route_ids, s.route_weights, s.x }, p, ((@as(usize, self.cfg.hidden_size) + tile - 1) / tile) * 32, 128);
    }
    /// Opt-in resident execution. The Lean-certified program is consumed directly
    /// by one kernel; GPU residency and floating-point behavior are tested, not proved.
    pub fn enableResident(self: *Model, workers: u32) !void {
        if (workers == 0 or workers > resident.certified.max_workers) return error.InvalidResidentWorkers;
        if (self.position != 0 or self.failed) return error.ResidentModeNeedsFreshSequence;
        if (self.resident_execution != null) return error.ResidentAlreadyEnabled;
        const c = self.cfg;
        const code = resident.certified.program(c.num_hidden_layers) orelse return error.UnsupportedResidentLayers;
        if (self.context > 4096) return error.UnsupportedResidentContext;
        if (c.hidden_size > 8192 or c.num_attention_heads > 64 or c.num_key_value_heads > 64 or
            c.num_experts > 1024 or c.num_experts_per_tok > 32 or c.vocab_size > 1048576 or
            @max(c.intermediate_size, @max(c.moe_intermediate_size, c.shared_expert_intermediate_size)) > 16384)
            return error.UnsupportedResidentDimensions;
        _ = std.math.mul(u32, self.context, c.num_key_value_heads * c.head_dim) catch return error.UnsupportedResidentDimensions;
        const layer_buffer = try self.alloc(@as(usize, c.num_hidden_layers) * @sizeOf(resident.Layer) / 4);
        for (self.layers, layer_buffer.structs(resident.Layer), 0..) |l, *r, i| {
            r.* = .{
                .input_norm = resident.vector(l.input_norm),
                .post_norm = resident.vector(l.post_norm),
                .q = l.q.matrix(),
                .k = l.k.matrix(),
                .v = l.v.matrix(),
                .o = l.o.matrix(),
                .q_norm = resident.vector(l.q_norm),
                .k_norm = resident.vector(l.k_norm),
                .sink = resident.vector(l.sink),
                .router = if (l.router) |w| w.matrix() else std.mem.zeroes(metal.Matrix),
                .experts = l.descriptors.address(),
                .bias = if (l.bias) |b| b.address() else 0,
                .keys = l.keys.address(),
                .values = l.values.address(),
                .capacity = l.capacity,
                .window = if (c.sliding(i)) c.sliding_window else 0,
                .rope = @intFromBool(c.rope(i)),
                .routed = if (l.dense != null) 0 else c.num_experts_per_tok,
                .shared_index = if (l.dense != null) 0 else c.num_experts,
                .stride = if (l.dense != null) c.intermediate_size else @max(c.moe_intermediate_size, c.shared_expert_intermediate_size),
            };
        }
        const program = try self.alloc(code.len);
        @memcpy(program.structs(u32), code);
        const descriptor = try self.alloc(@sizeOf(resident.Descriptor) / 4);
        const s = self.scratch;
        descriptor.structs(resident.Descriptor)[0] = .{
            .embedding = self.embedding.matrix(),
            .final_norm = resident.vector(self.final_norm),
            .head = self.head.matrix(),
            .layers = layer_buffer.address(),
            .x = s.x.address(),
            .norm = s.norm.address(),
            .q = s.q.address(),
            .k = s.k.address(),
            .v = s.v.address(),
            .attention = s.attention.address(),
            .projection = s.projection.address(),
            .activation = s.activation.address(),
            .router = s.router.address(),
            .route_ids = s.route_ids.address(),
            .route_weights = s.route_weights.address(),
            .logits = s.logits.address(),
            .hidden = c.hidden_size,
            .heads = c.num_attention_heads,
            .kv_heads = c.num_key_value_heads,
            .dim = c.head_dim,
            .layer_count = c.num_hidden_layers,
            .position = 0,
            .token = 0,
            .workers = workers,
            .eps = @floatCast(c.rms_norm_eps),
            .theta = @floatCast(c.rope_theta),
            .row_tile = 1,
        };
        self.resident_execution = .{ .descriptor = descriptor, .layers = layer_buffer, .program = program, .control = try self.alloc(@sizeOf(resident.Control) / 4), .workers = workers };
    }
    pub fn setStagedRowTile(self: *Model, tile: u32) !void {
        if (!resident.certified.validRowTile(tile)) return error.InvalidStagedRowTile;
        if (self.position != 0 or self.failed) return error.ResidentModeNeedsFreshSequence;
        if (self.resident_execution != null) return error.StagedTuningRequiresStaged;
        self.staged_row_tile = tile;
    }
    pub fn setResidentRowTile(self: *Model, tile: u32) !void {
        if (!resident.certified.validRowTile(tile)) return error.InvalidResidentRowTile;
        if (self.position != 0 or self.failed) return error.ResidentModeNeedsFreshSequence;
        const execution = self.resident_execution orelse return error.ResidentTuningRequiresResident;
        execution.descriptor.structs(resident.Descriptor)[0].row_tile = tile;
    }
    fn forwardResident(self: *Model, execution: resident.Execution, token: u32) ![]const f32 {
        const d = &execution.descriptor.structs(resident.Descriptor)[0];
        if (!resident.certified.validRowTile(d.row_tile)) return error.InvalidResidentRowTile;
        const kernel: [:0]const u8 = switch (d.row_tile) {
            1 => "resident_decode_t1",
            2 => "resident_decode_t2",
            4 => "resident_decode_t4",
            8 => "resident_decode_t8",
            else => return error.InvalidResidentRowTile,
        };
        d.position = self.position;
        d.token = token;
        @memset(execution.control.structs(u32), 0);
        const before = self.ctx.stats();
        try self.store.resident(self.ctx);
        for (self.buffers.items) |buffer| try self.ctx.residentWrite(buffer);
        try self.ctx.dispatch(kernel, &.{ execution.descriptor, execution.program, execution.control }, .{ .n = @intCast(execution.program.len / 4) }, @as(usize, execution.workers) * resident.certified.threadgroup_size, resident.certified.threadgroup_size);
        try self.ctx.finish();
        const after = self.ctx.stats();
        self.last_execution = .{ .submissions = after.submissions - before.submissions, .dispatches = after.dispatches - before.dispatches, .barriers = after.barriers - before.barriers, .gpu_seconds = after.gpu_seconds - before.gpu_seconds };
        const control = execution.control.structs(resident.Control)[0];
        if (control.status == 2) return error.NonFiniteLogits;
        if (control.status != 0 or control.completed != 1 or control.phase != execution.program.len / 4) {
            std.debug.print("Resident execution rejected: status={d}, completed={d}, phase={d}/{d}\n", .{ control.status, control.completed, control.phase, execution.program.len / 4 });
            return error.ResidentSynchronizationFailed;
        }
        self.position += 1;
        return self.scratch.logits.floats();
    }
    pub fn reset(self: *Model) !void {
        try self.ctx.finish();
        self.position = 0;
        self.failed = false;
        // No memset required: every cache entry is written before it can be read.
    }
    pub fn forward(self: *Model, token: u32) ![]const f32 {
        if (self.failed) return error.ModelNeedsReset;
        if (token >= self.cfg.vocab_size) return error.InvalidToken;
        if (self.position >= self.context) return error.ContextExhausted;
        errdefer self.failed = true;
        if (self.resident_execution) |execution| return self.forwardResident(execution, token);
        const c = self.cfg;
        const s = self.scratch;
        const before = self.ctx.stats();
        s.route_error.structs(u32)[0] = 0;
        try self.store.resident(self.ctx);
        var ep = self.embedding.params();
        ep.n = c.hidden_size;
        ep.pos = token;
        if (self.embedding.quant) |q| {
            try self.ctx.dispatch("qembedding", &.{ self.embedding.buffer, q.scales, q.biases, s.x }, ep, c.hidden_size, 128);
        } else {
            try self.ctx.dispatch("embedding", &.{ self.embedding.buffer, s.x }, ep, c.hidden_size, 128);
        }
        for (self.layers, 0..) |layer, i| {
            try self.norm(layer.input_norm, s.x, s.norm, 1, c.hidden_size);
            try self.qkv(layer);
            var qp = layer.q_norm.params();
            qp.scale_offset = layer.k_norm.tensor.offset;
            qp.scale_dtype = @backingInt(layer.k_norm.tensor.dtype);
            qp.heads = c.num_attention_heads;
            qp.kv_heads = c.num_key_value_heads;
            qp.dim = c.head_dim;
            qp.pos = self.position;
            qp.capacity = layer.capacity;
            qp.eps = @floatCast(c.rms_norm_eps);
            qp.theta = @floatCast(c.rope_theta);
            qp.bits = @intFromBool(c.rope(i));
            try self.ctx.dispatch("qk_rope_cache_fused", &.{ s.q, s.k, s.v, layer.q_norm.buffer, layer.k_norm.buffer, layer.keys, layer.values }, qp, @as(usize, c.num_attention_heads + c.num_key_value_heads) * 32, 32);
            var ap = layer.sink.params();
            ap.n = c.num_key_value_heads * c.head_dim;
            ap.pos = self.position;
            ap.capacity = layer.capacity;
            ap.heads = c.num_attention_heads;
            ap.kv_heads = c.num_key_value_heads;
            ap.dim = c.head_dim;
            ap.window = if (c.sliding(i)) c.sliding_window else 0;
            ap.scale = 1 / @sqrt(@as(f32, @floatFromInt(c.head_dim)));
            const parallel_attention = self.position >= 4;
            const attention_group: usize = if (parallel_attention) 256 else 32;
            try self.ctx.dispatch(if (parallel_attention) "attention_parallel" else "attention", &.{ s.q, layer.keys, layer.values, layer.sink.buffer, s.attention }, ap, @as(usize, c.num_attention_heads) * attention_group, attention_group);
            try self.matvec(layer.o, s.attention, s.projection);
            if (c.hidden_size <= 8192) {
                var np = layer.post_norm.params();
                np.n = c.hidden_size;
                np.eps = @floatCast(c.rms_norm_eps);
                try self.ctx.dispatch("residual_norm", &.{ layer.post_norm.buffer, s.x, s.projection, s.norm }, np, 256, 256);
            } else {
                try self.add(s.x, s.projection, 1);
                try self.norm(layer.post_norm, s.x, s.norm, 1, c.hidden_size);
            }
            if (layer.dense != null) {
                try self.fusedMLP(layer, 0, 0, c.intermediate_size);
            } else {
                try self.matvec(layer.router.?, s.norm, s.router);
                const route_kernel: [:0]const u8 = if (c.num_experts <= resident.certified.cached_router_experts) "route_topk_cached" else "route_topk";
                try self.ctx.dispatch(route_kernel, &.{ s.router, layer.bias.?, s.route_ids, s.route_weights, s.route_error }, .{ .n = c.num_experts_per_tok, .m = c.num_experts }, 32, 32);
                try self.fusedMLP(layer, c.num_experts_per_tok, c.num_experts, @max(c.moe_intermediate_size, c.shared_expert_intermediate_size));
            }
        }
        try self.norm(self.final_norm, s.x, s.norm, 1, c.hidden_size);
        try self.matvec(self.head, s.norm, s.logits);
        try self.ctx.finish();
        const after = self.ctx.stats();
        self.last_execution = .{ .submissions = after.submissions - before.submissions, .dispatches = after.dispatches - before.dispatches, .barriers = after.barriers - before.barriers, .gpu_seconds = after.gpu_seconds - before.gpu_seconds };
        if (s.route_error.structs(u32)[0] != 0) return error.NonFiniteLogits;
        self.position += 1;
        return s.logits.floats();
    }
};

test "resident specialized kernels reject mismatched descriptor tiles before barriers" {
    const ctx = try metal.Context.init();
    defer ctx.deinit();
    const descriptor = try ctx.alloc(@sizeOf(resident.Descriptor) / 4);
    defer descriptor.deinit();
    const program = try ctx.alloc(1);
    defer program.deinit();
    const control = try ctx.alloc(@sizeOf(resident.Control) / 4);
    defer control.deinit();
    program.structs(u32)[0] = @backingInt(resident.certified.Op.embedding);
    const kernels = [_][:0]const u8{ "resident_decode_t1", "resident_decode_t2", "resident_decode_t4", "resident_decode_t8" };
    const mismatches = [_]u32{ 2, 4, 8, 1 };
    for (kernels, mismatches) |kernel, mismatch| {
        for ([_]u32{ mismatch, 0 }) |tile| {
            descriptor.structs(resident.Descriptor)[0] = std.mem.zeroes(resident.Descriptor);
            descriptor.structs(resident.Descriptor)[0].row_tile = tile;
            descriptor.structs(resident.Descriptor)[0].workers = 4;
            @memset(control.structs(u32), 0);
            try ctx.dispatch(kernel, &.{ descriptor, program, control }, .{ .n = 1 }, 4 * resident.certified.threadgroup_size, resident.certified.threadgroup_size);
            try ctx.finish();
            const result = control.structs(resident.Control)[0];
            try std.testing.expectEqual(@as(u32, 3), result.status);
            try std.testing.expectEqual(@as(u32, 0), result.arrivals);
            try std.testing.expectEqual(@as(u32, 0), result.phase);
            try std.testing.expectEqual(@as(u32, 0), result.completed);
        }
    }
}
