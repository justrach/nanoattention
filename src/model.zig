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
    kv_bf16: bool = false,
};
const Scratch = struct {
    x: Buffer,
    norm: Buffer,
    q: Buffer,
    k: Buffer,
    v: Buffer,
    attention: Buffer,
    attention_scores: Buffer,
    projection: Buffer,
    activation: Buffer,
    down_outputs: ?Buffer = null,
    router: Buffer,
    route_ids: Buffer,
    route_weights: Buffer,
    route_error: Buffer,
    logits: Buffer,
};

/// Device-pointer ABI shared with token_batch.metal. Each token owns disjoint
/// activations; only immutable weights and causal KV are shared.
const TokenScratch = extern struct {
    x: u64,
    norm: u64,
    q: u64,
    k: u64,
    v: u64,
    attention: u64,
    projection: u64,
    activation: u64,
    router: u64,
    route_ids: u64,
    route_weights: u64,
    route_error: u64,
    logits: u64,
};
const Batch = struct {
    scratch: []Scratch,
    descriptors: Buffer,
    projections: []Buffer,
    head: Buffer,
    states: Buffer,
    greedy: Buffer,
    undo: []u8,
    undo_offsets: []usize,
    previous_logits: []f32,
    start: u32 = 0,
    count: usize = 0,
};
pub const max_batch_tokens = 32;

/// Eager, single-sequence FP32-activation baseline; weights stay in mapped unified memory.
/// Sliding layers have bounded ring caches; full-attention layers retain the requested context.
pub const Model = struct {
    pub const max_speculative_tokens = max_batch_tokens;
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
    batch: ?Batch = null,
    speculative_tokens: u32 = 0,
    batch_shared_attention: bool = true,
    batch_query_group: u32 = 2,
    batch_weight_reuse: bool = false,
    batch_mlp_weight_reuse: bool = false,
    last_execution: metal.Stats = .{ .submissions = 0, .dispatches = 0, .barriers = 0 },
    context: u32,
    kv_bf16: bool = false,
    bf16_blocks: u32 = 0,
    bf16_vector_loads: bool = false,
    position: u32 = 0,
    failed: bool = false,
    resident_execution: ?resident.Execution = null,
    staged_row_tile: u32 = 1,
    head_row_tile: u32 = 0,
    qkv_row_tile: u32 = 0,
    gate_row_tile: u32 = 0,
    projection_row_tile: u32 = 0,
    packed_router: bool = false,
    fixed_loops: bool = false,
    affine_loads: bool = false,
    fixed_attention: bool = false,
    // 0: measured M3 Ultra policy (two heads/SIMD at >=16K); 1: reference.
    gqa_reuse: u32 = 0,
    gqa_blocks: u32 = 1,
    gqa_prefetch: u32 = 4,
    wide_attention: bool = false,
    ring_attention: bool = false,
    prepared_inputs: bool = false,
    prepared_fused_norm: bool = false,
    prepared_fused_gate: bool = false,
    quant_prepared: ?Buffer = null,
    activation_prepared: ?Buffer = null,
    parallel_down: bool = false,
    down_row_tile: u32 = 0,
    attention_chunk: u32 = 0,
    mlx_attention: bool = false,
    trace_position: ?u32 = null,

    pub fn init(gpa: std.mem.Allocator, ctx: metal.Context, directory: []const u8, context: u32) !Model {
        return initWithKv(gpa, ctx, directory, context, false);
    }
    /// Experimental storage-only BF16 for global KV; activations and sliding KV
    /// remain FP32. Callers must enable matched staged attention before inference.
    pub fn initWithKv(gpa: std.mem.Allocator, ctx: metal.Context, directory: []const u8, context: u32, kv_bf16: bool) !Model {
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
        if (kv_bf16 and (cfg.head_dim != 128 or cfg.num_attention_heads / cfg.num_key_value_heads != 8 or !ctx.isM3Ultra())) return error.UnsupportedBf16KvShape;
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
            .kv_bf16 = kv_bf16,
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
            const layer_bf16 = kv_bf16 and !cfg.sliding(i);
            // alloc() counts four-byte words. D128 guarantees an even element
            // count, so BF16 uses exactly two bytes/element without padding.
            const cache_words = cache_elements / @as(usize, if (layer_bf16) 2 else 1);
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
                .keys = try self.alloc(cache_words),
                .values = try self.alloc(cache_words),
                .capacity = cap,
                .kv_bf16 = layer_bf16,
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
            .attention_scores = try self.alloc(if (self.splitAttentionSupported())
                @as(usize, cfg.num_attention_heads) * @max(context, resident.certified.mlxLargeDeviceBlocks(context) * resident.certified.mlx_attention_state_stride)
            else
                1),
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
        // ICB commands retain direct buffers; release those bindings BEFORE
        // freeing any no-copy checkpoint mapping or model-owned scratch.
        self.ctx.clearReplay() catch {};
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
    /// Candidate-only scratch and descriptor validation, outside decode timing.
    pub fn prepareQuantInputs(self: *Model) !void {
        if (self.quant_prepared != null) return;
        if (!self.mlx_attention or self.resident_execution != null or self.batch != null) return error.PreparedInputsRequireOrdinaryMatched;
        if (self.head.matrix().reserved != 1) return error.UnsupportedPreparedWeights;
        for (self.layers) |layer| {
            for ([_]Weight{ layer.q, layer.k, layer.v, layer.o }) |w|
                if (w.matrix().reserved != 1) return error.UnsupportedPreparedWeights;
            for (layer.descriptors.structs(metal.MLP)) |mlp| {
                if (mlp.gate.rows % 16 != 0) return error.UnsupportedPreparedWeights;
                for ([_]metal.Matrix{ mlp.gate, mlp.up, mlp.down }) |matrix|
                    if (matrix.reserved != 1) return error.UnsupportedPreparedWeights;
            }
        }
        const vector = @max(@as(usize, self.cfg.hidden_size), @as(usize, self.cfg.num_attention_heads) * self.cfg.head_dim);
        self.quant_prepared = try self.alloc(try std.math.add(usize, vector, vector / 16));
        const activation = self.scratch.activation.len / @sizeOf(f32);
        self.activation_prepared = try self.alloc(try std.math.add(usize, activation, activation / 16));
    }
    fn prepareQuant(self: *const Model, raw: Buffer, prepared: Buffer, columns: u32, vectors: u32) !void {
        if (columns == 0 or columns % 64 != 0) return error.InvalidPreparedGeometry;
        const raw_words = try std.math.mul(usize, columns, vectors);
        const words = try std.math.add(usize, raw_words, raw_words / 16);
        if (raw_words > raw.len / 4 or words > prepared.len / 4) return error.PreparedScratchTooSmall;
        try self.ctx.dispatch("prepare_quant_input", &.{ raw, prepared }, .{ .n = columns, .m = vectors }, raw_words / 16, 128);
    }
    fn matvec(self: *const Model, w: Weight, x: Buffer, out: Buffer) !void {
        return self.matvecTiled(w, x, out, self.staged_row_tile);
    }
    fn matvecTiled(self: *const Model, w: Weight, x: Buffer, out: Buffer, row_tile: u32) !void {
        var p = w.params();
        p.m = @intCast(w.tensor.shape[0]);
        p.n = @intCast(w.tensor.shape[1]);
        if (w.quant) |q| {
            const fast = q.bits == 4 and q.group_size == 64 and p.n % 64 == 0 and p.offset % 4 == 0;
            const tile = if (fast) row_tile else 1;
            var input = x;
            if (fast and self.prepared_inputs) {
                input = self.quant_prepared orelse return error.PreparedInputsNotPrepared;
                if (!self.prepared_fused_norm or x.handle != self.scratch.norm.handle)
                    try self.prepareQuant(x, input, p.n, 1);
            }
            const kernel: [:0]const u8 = if (!fast) "qmatvec" else if (self.prepared_inputs) switch (tile) {
                2 => "staged_qmatvec_prepared_t2",
                4 => "staged_qmatvec_prepared_t4",
                8 => "staged_qmatvec_prepared_t8",
                else => "staged_qmatvec_prepared_t1",
            } else if (self.affine_loads) switch (tile) {
                2 => "staged_qmatvec_affine_t2",
                4 => "staged_qmatvec_affine_t4",
                8 => "staged_qmatvec_affine_t8",
                else => "staged_qmatvec_affine_t1",
            } else if (self.fixed_loops) switch (tile) {
                2 => "staged_qmatvec_fixed_t2",
                4 => "staged_qmatvec_fixed_t4",
                8 => "staged_qmatvec_fixed_t8",
                else => "staged_qmatvec_fixed_t1",
            } else switch (tile) {
                2 => "staged_qmatvec_t2",
                4 => "staged_qmatvec_t4",
                8 => "staged_qmatvec_t8",
                else => "qmatvec4_g64",
            };
            try self.ctx.dispatch(kernel, &.{ w.buffer, q.scales, q.biases, input, out }, p, ((@as(usize, p.m) + tile - 1) / tile) * 32, 128);
        } else {
            const packed_router = self.packed_router and self.mlx_attention and (p.dtype == 0 or p.dtype == 2) and p.m == 256 and p.n == 2048;
            const split = p.m <= 256 and p.n >= 512 and p.n <= resident.certified.dense_max_cols and p.dtype <= 2;
            const kernel: [:0]const u8 = if (packed_router) (if (p.dtype == 2) "dense_matvec_router_bf16" else "dense_matvec_router_f32") else if (split) switch (p.dtype) {
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
            const group: usize = if (packed_router) 128 else if (split) resident.certified.threadgroup_size else 128;
            const row_threads: usize = if (split and !packed_router) resident.certified.threadgroup_size else 32;
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
        const tile = if (self.qkv_row_tile == 0) self.staged_row_tile else self.qkv_row_tile;
        var tiles: usize = 0;
        for (matrices) |matrix| tiles += (@as(usize, matrix.rows) + tile - 1) / tile;
        var input = s.norm;
        if (self.prepared_inputs) {
            input = self.quant_prepared orelse return error.PreparedInputsNotPrepared;
            if (!self.prepared_fused_norm) try self.prepareQuant(s.norm, input, self.cfg.hidden_size, 1);
        }
        const kernel: [:0]const u8 = if (self.prepared_inputs) switch (tile) {
            2 => "staged_qkv_prepared_t2",
            4 => "staged_qkv_prepared_t4",
            8 => "staged_qkv_prepared_t8",
            else => "staged_qkv_prepared_t1",
        } else if (self.affine_loads) switch (tile) {
            2 => "staged_qkv_affine_t2",
            4 => "staged_qkv_affine_t4",
            8 => "staged_qkv_affine_t8",
            else => "staged_qkv_affine_t1",
        } else if (self.fixed_loops) switch (tile) {
            2 => "staged_qkv_fixed_t2",
            4 => "staged_qkv_fixed_t4",
            8 => "staged_qkv_fixed_t8",
            else => "staged_qkv_fixed_t1",
        } else switch (tile) {
            2 => "staged_qkv_t2",
            4 => "staged_qkv_t4",
            8 => "staged_qkv_t8",
            else => "staged_qkv_t1",
        };
        try self.ctx.dispatch(kernel, &.{ layer.qkv_descriptors, input, s.q, s.k, s.v }, .{}, tiles * 32, 128);
    }
    fn norm(self: *const Model, w: Weight, x: Buffer, out: Buffer, heads: u32, dim: u32) !void {
        var p = w.params();
        p.m = heads;
        p.n = dim;
        p.eps = @floatCast(self.cfg.rms_norm_eps);
        if (self.mlx_attention and heads == 1 and dim == resident.certified.aligned_norm_dim) {
            if (self.prepared_inputs and self.prepared_fused_norm) {
                const prepared = self.quant_prepared orelse return error.PreparedInputsNotPrepared;
                try self.ctx.dispatch("rmsnorm_aligned_prepared", &.{ w.buffer, x, out, prepared }, p, resident.certified.aligned_norm_threads, resident.certified.aligned_norm_threads);
            } else try self.ctx.dispatch("rmsnorm_aligned", &.{ w.buffer, x, out }, p, resident.certified.aligned_norm_threads, resident.certified.aligned_norm_threads);
        } else if (heads == 1 and dim >= 512 and dim <= 8192) {
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
    /// Allocate candidate-only scratch outside inference timing. Default serving
    /// retains its original fused-down path and does not allocate this buffer.
    pub fn prepareParallelDown(self: *Model) !void {
        if (self.scratch.down_outputs == null)
            self.scratch.down_outputs = try self.alloc(try std.math.mul(usize, @as(usize, self.cfg.num_experts_per_tok) + 1, self.cfg.hidden_size));
    }
    fn fusedMLP(self: *const Model, layer: Layer, routed: u32, shared_index: u32, stride: u32) !void {
        const s = self.scratch;
        const p: metal.Params = .{ .n = routed, .m = shared_index, .k = stride, .dim = self.cfg.hidden_size };
        const tile = if (self.gate_row_tile == 0) self.staged_row_tile else self.gate_row_tile;
        const gate: [:0]const u8 = if (self.prepared_inputs and self.prepared_fused_gate) switch (tile) {
            2 => "staged_gate_up_prepared_fused_t2",
            4 => "staged_gate_up_prepared_fused_t4",
            8 => "staged_gate_up_prepared_fused_t8",
            else => "staged_gate_up_prepared_fused_t1",
        } else if (self.prepared_inputs) switch (tile) {
            2 => "staged_gate_up_prepared_t2",
            4 => "staged_gate_up_prepared_t4",
            8 => "staged_gate_up_prepared_t8",
            else => "staged_gate_up_prepared_t1",
        } else if (self.affine_loads) switch (tile) {
            2 => "staged_gate_up_affine_t2",
            4 => "staged_gate_up_affine_t4",
            8 => "staged_gate_up_affine_t8",
            else => "staged_gate_up_affine_t1",
        } else if (self.fixed_loops) switch (tile) {
            2 => "staged_gate_up_fixed_t2",
            4 => "staged_gate_up_fixed_t4",
            8 => "staged_gate_up_fixed_t8",
            else => "staged_gate_up_fixed_t1",
        } else switch (tile) {
            2 => "staged_gate_up_t2",
            4 => "staged_gate_up_t4",
            8 => "staged_gate_up_t8",
            else => "fused_gate_up",
        };
        const down_tile = if (self.down_row_tile == 0) self.staged_row_tile else self.down_row_tile;
        const down: [:0]const u8 = if (self.prepared_inputs) switch (down_tile) {
            2 => "staged_down_prepared_t2",
            4 => "staged_down_prepared_t4",
            8 => "staged_down_prepared_t8",
            else => "staged_down_prepared_t1",
        } else if (self.affine_loads) switch (down_tile) {
            2 => "staged_down_affine_t2",
            4 => "staged_down_affine_t4",
            8 => "staged_down_affine_t8",
            else => "staged_down_affine_t1",
        } else if (self.fixed_loops) switch (down_tile) {
            2 => "staged_down_fixed_t2",
            4 => "staged_down_fixed_t4",
            8 => "staged_down_fixed_t8",
            else => "staged_down_fixed_t1",
        } else switch (down_tile) {
            2 => "staged_down_t2",
            4 => "staged_down_t4",
            8 => "staged_down_t8",
            else => "fused_down_reduce",
        };
        var input = s.norm;
        if (self.prepared_inputs) {
            input = self.quant_prepared orelse return error.PreparedInputsNotPrepared;
            if (!self.prepared_fused_norm) try self.prepareQuant(s.norm, input, self.cfg.hidden_size, 1);
        }
        const threads = (@as(usize, routed) + 1) * ((@as(usize, stride) + tile - 1) / tile) * 32;
        if (self.prepared_inputs and self.prepared_fused_gate) {
            const prepared = self.activation_prepared orelse return error.PreparedInputsNotPrepared;
            try self.ctx.dispatch(gate, &.{ layer.descriptors, input, s.route_ids, s.activation, prepared }, p, threads, 512 / tile);
        } else try self.ctx.dispatch(gate, &.{ layer.descriptors, input, s.route_ids, s.activation }, p, threads, 128);
        var activation = s.activation;
        if (self.prepared_inputs) {
            activation = self.activation_prepared orelse return error.PreparedInputsNotPrepared;
            if (!self.prepared_fused_gate) try self.prepareQuant(s.activation, activation, stride, routed + 1);
        }
        if (self.parallel_down and self.mlx_attention and routed != 0) {
            const parallel: [:0]const u8 = switch (down_tile) {
                2 => "staged_down_parallel_t2",
                4 => "staged_down_parallel_t4",
                8 => "staged_down_parallel_t8",
                else => "staged_down_parallel_t1",
            };
            const temporary = s.down_outputs orelse return error.ParallelDownNotPrepared;
            try self.ctx.dispatch(parallel, &.{ layer.descriptors, s.activation, s.route_ids, s.route_weights, temporary }, p, (@as(usize, routed) + 1) * ((@as(usize, self.cfg.hidden_size) + down_tile - 1) / down_tile) * 32, 128);
            try self.ctx.dispatch("staged_down_merge", &.{ temporary, s.route_weights, s.x }, p, self.cfg.hidden_size, 128);
        } else {
            try self.ctx.dispatch(down, &.{ layer.descriptors, activation, s.route_ids, s.route_weights, s.x }, p, ((@as(usize, self.cfg.hidden_size) + down_tile - 1) / down_tile) * 32, 128);
        }
    }
    /// Opt-in resident execution. The Lean-certified program is consumed directly
    /// by one kernel; GPU residency and floating-point behavior are tested, not proved.
    pub fn enableResident(self: *Model, workers: u32) !void {
        if (self.kv_bf16) return error.Bf16KvRequiresMatchedStaged;
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
    fn splitAttentionSupported(self: *const Model) bool {
        return self.context <= resident.certified.attention_max_context and
            self.cfg.num_attention_heads <= resident.certified.attention_max_heads and
            self.cfg.num_key_value_heads <= resident.certified.attention_max_heads and
            self.cfg.head_dim <= resident.certified.attention_max_dim;
    }
    /// Tuning may exceed the automatic schedule's scratch allocation. Validate
    /// and grow it before any prefix inference, never after encoding a producer.
    pub fn setBf16Blocks(self: *Model, blocks: u32) !void {
        if (blocks == 0) {
            self.bf16_blocks = 0;
            return;
        }
        if (!self.kv_bf16) return error.Bf16KvRequiresMatchedStaged;
        if (blocks != 128 and blocks != 512 and blocks != 1024) return error.InvalidBf16Blocks;
        if (self.position != 0 or self.failed) return error.Bf16TuningRequiresFreshSequence;
        if (self.batch != null and blocks != self.bf16_blocks) return error.Bf16TuningRequiresBeforeBatch;
        const words = try std.math.mul(usize, self.cfg.num_attention_heads, @as(usize, blocks) * resident.certified.mlx_attention_state_stride);
        if (words > self.scratch.attention_scores.len / @sizeOf(f32))
            self.scratch.attention_scores = try self.alloc(words);
        self.bf16_blocks = blocks;
    }
    pub fn enableMlxAttention(self: *Model) !void {
        if (self.cfg.head_dim != resident.certified.mlx_attention_dim or !self.splitAttentionSupported() or self.cfg.num_attention_heads / self.cfg.num_key_value_heads > resident.certified.mlx_attention_max_gqa) return error.UnsupportedAttentionShape;
        if (self.context >= 16384 and (!self.ctx.isM3Ultra() or self.cfg.num_attention_heads / self.cfg.num_key_value_heads != 8)) return error.UnsupportedLongContextAttentionDevice;
        if (self.position != 0 or self.failed or self.resident_execution != null or self.attention_chunk != 0) return error.ConflictingAttentionMode;
        self.mlx_attention = true;
    }
    pub fn setAttentionChunk(self: *Model, chunk: u32) !void {
        if (chunk != 0 and !resident.certified.validAttentionChunk(chunk)) return error.InvalidAttentionChunk;
        if (self.position != 0 or self.failed) return error.AttentionTuningNeedsFreshSequence;
        if (self.resident_execution != null) return error.AttentionTuningRequiresStaged;
        self.attention_chunk = chunk;
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
        if (self.batch) |*b| b.count = 0;
        // No memset required: every cache entry is written before it can be read.
    }
    fn trace(self: *Model, layer: usize, stage: []const u8, buffer: Buffer, count: usize) !void {
        if (self.trace_position == null or self.trace_position.? != self.position) return;
        try self.ctx.finish();
        if (std.mem.eql(u8, stage, "route_ids")) {
            try @import("diagnostics.zig").emit(self.position, layer, stage, buffer.structs(u32)[0..count]);
        } else {
            try @import("diagnostics.zig").emit(self.position, layer, stage, buffer.floats()[0..count]);
        }
        // A diagnostic completion starts a new encoder; GPU-address weights
        // must be declared resident again before subsequent indirect accesses.
        try self.store.resident(self.ctx);
    }
    /// Append every prefix token to KV, but materialize vocabulary logits only
    /// for the final token. The resident plan remains unchanged and falls back
    /// to its full forward; staged execution omits only the terminal norm/head.
    pub fn prefill(self: *Model, tokens: []const u32) ![]const f32 {
        if (self.failed) return error.ModelNeedsReset;
        if (tokens.len == 0) return error.EmptyPrompt;
        for (tokens) |token| if (token >= self.cfg.vocab_size) return error.InvalidToken;
        if (@as(u64, self.position) + tokens.len > self.context) return error.ContextExhausted;
        var logits: []const f32 = &.{};
        for (tokens, 0..) |token, i| logits = try self.forwardImpl(token, i + 1 == tokens.len);
        return logits;
    }
    pub fn forward(self: *Model, token: u32) ![]const f32 {
        return self.forwardImpl(token, true);
    }
    /// Prepare a bounded multi-token target verifier. Allocation and kernel
    /// compilation warmup belong outside timing; ordinary forward is unchanged.
    pub fn prepareBatch(self: *Model) !void {
        if (self.prepared_inputs) return error.PreparedInputsRequireOrdinaryMatched;
        if (self.batch != null) return;
        if (!self.mlx_attention or self.resident_execution != null or self.trace_position != null)
            return error.BatchRequiresMatchedStaged;
        if (self.head.matrix().reserved != 1) return error.UnsupportedBatchWeights;
        for (self.layers) |layer| {
            if (layer.o.matrix().reserved != 1) return error.UnsupportedBatchWeights;
            for (layer.qkv_descriptors.structs(metal.Matrix)) |m|
                if (m.reserved != 1) return error.UnsupportedBatchWeights;
            if (layer.capacity < max_batch_tokens) return error.UnsupportedBatchCapacity;
        }
        const a = self.arena.allocator();
        const lanes = try a.alloc(Scratch, max_batch_tokens);
        const fields = [_][]const u8{ "x", "norm", "q", "k", "v", "attention", "projection", "activation", "router", "route_ids", "route_weights", "route_error", "logits" };
        lanes[0] = self.scratch;
        for (lanes[1..]) |*lane| {
            lane.* = self.scratch;
            inline for (fields ++ .{"attention_scores"}) |field|
                @field(lane, field) = try self.alloc(@field(self.scratch, field).len / 4);
            lane.down_outputs = null;
        }
        const descriptors = try self.alloc(max_batch_tokens * @sizeOf(TokenScratch) / 4);
        for (lanes, descriptors.structs(TokenScratch)) |lane, *d| {
            inline for (fields) |field|
                @field(d, field) = @field(lane, field).address();
        }
        const projections = try a.alloc(Buffer, self.layers.len);
        for (self.layers, projections) |layer, *d| {
            d.* = try self.alloc(@sizeOf(metal.Matrix) / 4);
            d.structs(metal.Matrix)[0] = layer.o.matrix();
        }
        const head = try self.alloc(@sizeOf(metal.Matrix) / 4);
        head.structs(metal.Matrix)[0] = self.head.matrix();
        const offsets = try a.alloc(usize, self.layers.len);
        var bytes: usize = 0;
        for (self.layers, offsets, 0..) |layer, *offset, i| {
            offset.* = bytes;
            if (self.cfg.sliding(i)) bytes += max_batch_tokens * (layer.keys.len + layer.values.len) / layer.capacity;
        }
        self.batch = .{ .scratch = lanes, .descriptors = descriptors, .projections = projections, .head = head, .greedy = try self.alloc(max_batch_tokens * 2), .states = try self.alloc(max_batch_tokens * @as(usize, self.cfg.num_attention_heads) * @max(self.bf16_blocks, resident.certified.mlxLargeDeviceBlocks(self.context)) * resident.certified.mlx_attention_state_stride), .undo = try a.alloc(u8, bytes), .undo_offsets = offsets, .previous_logits = try a.alloc(f32, self.cfg.vocab_size) };
    }
    /// True layer-major multi-token execution. Projection/MLP dispatches cover
    /// all token rows together. Cache writes and attention stay token-ordered
    /// within each layer, so a full sliding ring never exposes future writes.
    /// Returned rows predict the token AFTER each input. Call finishBatch once
    /// to commit an accepted prefix (or zero); never expose unaccepted tokens.
    pub fn forwardBatch(self: *Model, tokens: []const u32) ![]const Scratch {
        if (self.prepared_inputs) return error.PreparedInputsRequireOrdinaryMatched;
        if (self.failed) return error.ModelNeedsReset;
        if (!self.mlx_attention or self.resident_execution != null or self.trace_position != null) return error.BatchRequiresMatchedStaged;
        if (self.batch_query_group != 2 and self.batch_query_group != 4 and self.batch_query_group != 8) return error.InvalidBatchQueryGroup;
        if (self.bf16_blocks != 0 and (!self.kv_bf16 or (self.bf16_blocks != 128 and self.bf16_blocks != 512 and self.bf16_blocks != 1024))) return error.InvalidBf16Blocks;
        if (tokens.len == 0 or tokens.len > max_batch_tokens) return error.InvalidBatchLength;
        for (tokens) |token| if (token >= self.cfg.vocab_size) return error.InvalidToken;
        if (@as(u64, self.position) + tokens.len > self.context) return error.ContextExhausted;
        for ([_]u32{ self.staged_row_tile, self.head_row_tile, self.qkv_row_tile, self.gate_row_tile, self.projection_row_tile, self.down_row_tile }) |tile|
            if (tile != 0 and !resident.certified.validRowTile(tile)) return error.InvalidStagedRowTile;
        if (self.staged_row_tile == 0) return error.InvalidStagedRowTile;
        try self.prepareBatch();
        const batch = &self.batch.?;
        if (batch.count != 0) return error.BatchPendingCommit;
        const state_words = @as(usize, self.cfg.num_attention_heads) * @max(self.bf16_blocks, resident.certified.mlxLargeDeviceBlocks(self.context)) * resident.certified.mlx_attention_state_stride;
        if (state_words > self.scratch.attention_scores.len / 4 or max_batch_tokens * state_words > batch.states.len / 4) return error.Bf16ScratchTooSmall;
        const saved = self.scratch;
        const start = self.position;
        defer self.scratch = saved;
        defer self.position = start;
        try self.ctx.finish();
        @memcpy(batch.previous_logits, saved.logits.floats());
        batch.start = start;
        batch.count = tokens.len;
        for (self.layers, 0..) |layer, i| {
            if (!self.cfg.sliding(i)) continue;
            const ks = layer.keys.len / layer.capacity;
            const vs = layer.values.len / layer.capacity;
            for (0..tokens.len) |j| {
                const slot = (@as(usize, start) + j) % layer.capacity;
                const offset = batch.undo_offsets[i] + j * (ks + vs);
                @memcpy(batch.undo[offset..][0..ks], layer.keys.structs(u8)[slot * ks ..][0..ks]);
                @memcpy(batch.undo[offset + ks ..][0..vs], layer.values.structs(u8)[slot * vs ..][0..vs]);
            }
        }
        errdefer self.failed = true;
        const c = self.cfg;
        const token_groups = if (self.batch_weight_reuse) (tokens.len + 1) / 2 else tokens.len;
        const before = self.ctx.stats();
        try self.store.resident(self.ctx);
        // TokenScratch contains device addresses, not directly bound resources.
        // Only active token scratch is indirect. Cache/descriptor/temporary
        // buffers are directly bound, and immutable weights are declared above.
        // Do not make unused draft lanes or every layer allocation resident.
        const active_fields = [_][]const u8{ "x", "norm", "q", "k", "v", "attention", "attention_scores", "projection", "activation", "router", "route_ids", "route_weights", "route_error", "logits" };
        for (batch.scratch[0..tokens.len]) |lane| {
            inline for (active_fields) |field| try self.ctx.residentWrite(@field(lane, field));
        }
        for (batch.scratch[0..tokens.len], tokens) |lane, token| {
            lane.route_error.structs(u32)[0] = 0;
            var ep = self.embedding.params();
            ep.n = c.hidden_size;
            ep.pos = token;
            if (self.embedding.quant) |q| {
                try self.ctx.dispatch("qembedding", &.{ self.embedding.buffer, q.scales, q.biases, lane.x }, ep, c.hidden_size, 128);
            } else {
                try self.ctx.dispatch("embedding", &.{ self.embedding.buffer, lane.x }, ep, c.hidden_size, 128);
            }
        }
        for (self.layers, 0..) |layer, i| {
            for (batch.scratch[0..tokens.len]) |lane|
                try self.norm(layer.input_norm, lane.x, lane.norm, 1, c.hidden_size);
            const qt = if (self.qkv_row_tile == 0) self.staged_row_tile else self.qkv_row_tile;
            var tiles: usize = 0;
            for (layer.qkv_descriptors.structs(metal.Matrix)) |m| tiles += (@as(usize, m.rows) + qt - 1) / qt;
            try self.ctx.dispatch(if (self.batch_weight_reuse) "token_batch_qkv_weight2" else "token_batch_qkv", &.{ layer.qkv_descriptors, batch.descriptors }, .{ .m = @intCast(tokens.len), .n = qt }, token_groups * tiles * 32, 128);
            const first_blocks = if (layer.kv_bf16 and self.bf16_blocks != 0) self.bf16_blocks else resident.certified.mlxLargeDeviceBlocks(start + 1);
            const last_blocks = if (layer.kv_bf16 and self.bf16_blocks != 0) self.bf16_blocks else resident.certified.mlxLargeDeviceBlocks(start + @as(u32, @intCast(tokens.len)));
            const shared_attention = self.batch_shared_attention and !c.sliding(i) and start >= 1023 and
                c.num_attention_heads == 32 and c.num_key_value_heads == 4 and c.head_dim == 128 and first_blocks == last_blocks;
            for (batch.scratch[0..tokens.len], 0..) |lane, j| {
                self.scratch = lane;
                self.position = start + @as(u32, @intCast(j));
                if (shared_attention) try self.encodeQkCache(layer, i) else try self.encodeAttention(layer, i);
            }
            if (shared_attention) {
                var ap = layer.sink.params();
                ap.m = @intCast(tokens.len);
                ap.pos = start;
                ap.capacity = layer.capacity;
                ap.heads = c.num_attention_heads;
                ap.kv_heads = c.num_key_value_heads;
                ap.dim = c.head_dim;
                ap.k = first_blocks;
                ap.bits = @intFromBool(layer.kv_bf16);
                ap.scale = 1 / @sqrt(@as(f32, @floatFromInt(c.head_dim)));
                const query_group = self.batch_query_group;
                const split: [:0]const u8 = switch (query_group) {
                    4 => "token_batch_attention_split_q4",
                    8 => "token_batch_attention_split_q8",
                    else => "token_batch_attention_split",
                };
                try self.ctx.dispatch(split, &.{ batch.descriptors, layer.keys, layer.values, layer.sink.buffer, batch.states }, ap, ((tokens.len + query_group - 1) / query_group) * c.num_key_value_heads * @as(usize, ap.k) * 128, 128);
                try self.ctx.dispatch("token_batch_attention_merge", &.{ batch.states, batch.descriptors }, ap, tokens.len * c.num_attention_heads * 1024, 1024);
            }
            const ot = if (self.projection_row_tile == 0) self.staged_row_tile else self.projection_row_tile;
            try self.ctx.dispatch(if (self.batch_weight_reuse) "token_batch_matvec_weight2" else "token_batch_matvec", &.{ batch.projections[i], batch.descriptors }, .{ .m = @intCast(tokens.len), .n = ot, .bits = 0 }, token_groups * ((@as(usize, c.hidden_size) + ot - 1) / ot) * 32, 128);
            for (batch.scratch[0..tokens.len], 0..) |lane, j| {
                self.scratch = lane;
                self.position = start + @as(u32, @intCast(j));
                try self.encodePostNorm(layer, i);
                if (layer.router) |router| {
                    try self.matvec(router, lane.norm, lane.router);
                    const route_kernel: [:0]const u8 = if (c.num_experts <= resident.certified.cached_router_experts) "route_topk_cached" else "route_topk";
                    try self.ctx.dispatch(route_kernel, &.{ lane.router, layer.bias.?, lane.route_ids, lane.route_weights, lane.route_error }, .{ .n = c.num_experts_per_tok, .m = c.num_experts }, 32, 32);
                }
            }
            const routed: u32 = if (layer.dense != null) 0 else c.num_experts_per_tok;
            const shared: u32 = if (layer.dense != null) 0 else c.num_experts;
            const stride: u32 = if (layer.dense != null) c.intermediate_size else @max(c.moe_intermediate_size, c.shared_expert_intermediate_size);
            const gt = if (self.gate_row_tile == 0) self.staged_row_tile else self.gate_row_tile;
            const dt = if (self.down_row_tile == 0) self.staged_row_tile else self.down_row_tile;
            const mlp_groups = if (self.batch_mlp_weight_reuse) (tokens.len + 1) / 2 else tokens.len;
            var mp: metal.Params = .{ .n = routed, .m = shared, .k = stride, .dim = c.hidden_size, .heads = @intCast(tokens.len), .bits = gt };
            try self.ctx.dispatch(if (self.batch_mlp_weight_reuse) "token_batch_gate_up_weight2" else "token_batch_gate_up", &.{ layer.descriptors, batch.descriptors }, mp, mlp_groups * (@as(usize, routed) + 1) * ((@as(usize, stride) + gt - 1) / gt) * 32, 128);
            mp.bits = dt;
            try self.ctx.dispatch(if (self.batch_mlp_weight_reuse) "token_batch_down_weight2" else "token_batch_down", &.{ layer.descriptors, batch.descriptors }, mp, mlp_groups * ((@as(usize, c.hidden_size) + dt - 1) / dt) * 32, 128);
        }
        for (batch.scratch[0..tokens.len]) |lane|
            try self.norm(self.final_norm, lane.x, lane.norm, 1, c.hidden_size);
        const ht = if (self.head_row_tile == 0) self.staged_row_tile else self.head_row_tile;
        try self.ctx.dispatch(if (self.batch_weight_reuse) "token_batch_matvec_weight2" else "token_batch_matvec", &.{ batch.head, batch.descriptors }, .{ .m = @intCast(tokens.len), .n = ht, .bits = 1 }, token_groups * ((@as(usize, c.vocab_size) + ht - 1) / ht) * 32, 128);
        try self.ctx.dispatch("token_batch_greedy", &.{ batch.descriptors, batch.greedy }, .{ .m = @intCast(tokens.len), .n = c.vocab_size }, tokens.len * 256, 256);
        try self.ctx.finish();
        const after = self.ctx.stats();
        self.last_execution = .{ .submissions = after.submissions - before.submissions, .dispatches = after.dispatches - before.dispatches, .barriers = after.barriers - before.barriers, .gpu_seconds = after.gpu_seconds - before.gpu_seconds };
        const predictions = batch.greedy.structs(u32);
        for (batch.scratch[0..tokens.len], 0..) |lane, j| {
            if (lane.route_error.structs(u32)[0] != 0 or predictions[2 * j + 1] != 0) return error.NonFiniteLogits;
            if (predictions[2 * j] >= c.vocab_size) return error.InvalidGreedyToken;
        }
        return batch.scratch[0..tokens.len];
    }
    pub fn batchPrediction(self: *const Model, index: usize) !u32 {
        if (self.failed) return error.ModelNeedsReset;
        const batch = self.batch orelse return error.NoPendingBatch;
        if (index >= batch.count) return error.InvalidBatchPrediction;
        return batch.greedy.structs(u32)[2 * index];
    }
    pub fn finishBatch(self: *Model, accepted: usize) ![]const f32 {
        const batch = if (self.batch) |*b| b else return error.NoPendingBatch;
        if (batch.count == 0 or accepted > batch.count) return error.InvalidBatchCommit;
        if (self.failed) return error.ModelNeedsReset;
        try self.ctx.finish();
        for (self.layers, 0..) |layer, i| {
            if (!self.cfg.sliding(i)) continue;
            const ks = layer.keys.len / layer.capacity;
            const vs = layer.values.len / layer.capacity;
            for (accepted..batch.count) |j| {
                const slot = (@as(usize, batch.start) + j) % layer.capacity;
                const offset = batch.undo_offsets[i] + j * (ks + vs);
                @memcpy(layer.keys.structs(u8)[slot * ks ..][0..ks], batch.undo[offset..][0..ks]);
                @memcpy(layer.values.structs(u8)[slot * vs ..][0..vs], batch.undo[offset + ks ..][0..vs]);
            }
        }
        const logits = if (accepted == 0) batch.previous_logits else batch.scratch[accepted - 1].logits.floats();
        if (logits.ptr != self.scratch.logits.floats().ptr) @memcpy(self.scratch.logits.floats(), logits);
        self.position = batch.start + @as(u32, @intCast(accepted));
        batch.count = 0;
        return self.scratch.logits.floats();
    }
    fn encodeInput(self: *Model, layer: Layer, i: usize) !void {
        const c = self.cfg;
        const s = self.scratch;
        try self.trace(i, "input", s.x, c.hidden_size);
        try self.norm(layer.input_norm, s.x, s.norm, 1, c.hidden_size);
        try self.trace(i, "input_norm", s.norm, c.hidden_size);
        try self.qkv(layer);
        try self.trace(i, "q_raw", s.q, c.num_attention_heads * c.head_dim);
        try self.trace(i, "k_raw", s.k, c.num_key_value_heads * c.head_dim);
        try self.trace(i, "v_raw", s.v, c.num_key_value_heads * c.head_dim);
    }
    fn encodeQkCache(self: *Model, layer: Layer, i: usize) !void {
        const c = self.cfg;
        const s = self.scratch;
        var qp = layer.q_norm.params();
        qp.scale_offset = layer.k_norm.tensor.offset;
        qp.scale_dtype = @backingInt(layer.k_norm.tensor.dtype);
        qp.heads = c.num_attention_heads;
        qp.kv_heads = c.num_key_value_heads;
        qp.dim = c.head_dim;
        qp.pos = self.position;
        qp.capacity = layer.capacity;
        qp.eps = @floatCast(c.rms_norm_eps);
        qp.theta = if (self.mlx_attention) @log2(@as(f32, @floatCast(c.rope_theta))) else @floatCast(c.rope_theta);
        qp.bits = @intFromBool(c.rope(i));
        try self.ctx.dispatch(if (layer.kv_bf16) "qk_rope_cache_aligned_bf16" else if (self.mlx_attention and c.head_dim == resident.certified.aligned_head_dim) "qk_rope_cache_aligned" else "qk_rope_cache_fused", &.{ s.q, s.k, s.v, layer.q_norm.buffer, layer.k_norm.buffer, layer.keys, layer.values }, qp, @as(usize, c.num_attention_heads + c.num_key_value_heads) * 32, 32);
        try self.trace(i, "q", s.q, c.num_attention_heads * c.head_dim);
        try self.trace(i, "k", s.k, c.num_key_value_heads * c.head_dim);
    }
    fn encodeAttention(self: *Model, layer: Layer, i: usize) !void {
        try self.encodeQkCache(layer, i);
        const c = self.cfg;
        const s = self.scratch;
        var ap = layer.sink.params();
        ap.n = c.num_key_value_heads * c.head_dim;
        ap.pos = self.position;
        ap.capacity = layer.capacity;
        ap.heads = c.num_attention_heads;
        ap.kv_heads = c.num_key_value_heads;
        ap.dim = c.head_dim;
        ap.window = if (c.sliding(i)) c.sliding_window else 0;
        ap.scale = 1 / @sqrt(@as(f32, @floatFromInt(c.head_dim)));
        const length = if (ap.window == 0) self.position + 1 else @min(self.position + 1, ap.window);
        if (self.mlx_attention and c.head_dim == 128 and c.num_attention_heads / c.num_key_value_heads <= 32) {
            if (length >= 1024) {
                // MLX's applegpu_g15d GQA8 schedule changes at 16K and 64K.
                ap.k = if (layer.kv_bf16 and self.bf16_blocks != 0) self.bf16_blocks else resident.certified.mlxLargeDeviceBlocks(length);
                const linear = ap.window == 0 and ap.heads / ap.kv_heads == 8 and ap.pos < 262144 and ap.pos + 1 <= ap.capacity and (ap.k == 128 or ap.k == 512 or ap.k == 1024);
                // enableMlxAttention already restricts >=16K contexts to M3
                // Ultra/GQA8. Short/sliding prefixes retain the original path.
                const requested_reuse = if (self.gqa_reuse == 0 and length >= 16384) @as(u32, 2) else self.gqa_reuse;
                const reuse = if (linear and ap.dtype <= 2 and ap.offset % @as(u64, if (ap.dtype == 0) 4 else 2) == 0 and (requested_reuse == 2 or requested_reuse == 4)) requested_reuse else 1;
                const effective_reuse: u32 = if (layer.kv_bf16) 2 else reuse;
                const threads: usize = resident.certified.mlx_attention_lanes * (c.num_attention_heads / c.num_key_value_heads) / effective_reuse;
                const blocks: usize = if (layer.kv_bf16) 1 else if ((reuse == 2 and (self.gqa_blocks == 2 or self.gqa_blocks == 4)) or (reuse == 4 and self.gqa_blocks == 4)) self.gqa_blocks else 1;
                const kernel: [:0]const u8 = if (layer.kv_bf16 and self.bf16_vector_loads) "attention_mlx_split_gqa2_bf16_vector" else if (layer.kv_bf16) "attention_mlx_split_gqa2_bf16" else if (reuse == 2 and blocks == 2) "attention_mlx_split_gqa2_b2" else if (reuse == 2 and blocks == 4) "attention_mlx_split_gqa2_b4" else if (reuse == 4 and blocks == 4) "attention_mlx_split_gqa4_b4" else if (reuse == 2 and self.gqa_prefetch == 1) "attention_mlx_split_gqa2_tk1" else if (reuse == 2 and self.gqa_prefetch == 2) "attention_mlx_split_gqa2_tk2" else if (reuse == 2) "attention_mlx_split_gqa2" else if (reuse == 4 and self.gqa_prefetch == 1) "attention_mlx_split_gqa4_tk1" else if (reuse == 4 and self.gqa_prefetch == 2) "attention_mlx_split_gqa4_tk2" else if (reuse == 4) "attention_mlx_split_gqa4" else if (self.fixed_attention and linear) "attention_mlx_split_fixed" else "attention_mlx_split";
                try self.ctx.dispatch(kernel, &.{ s.q, layer.keys, layer.values, layer.sink.buffer, s.attention_scores }, ap, @as(usize, c.num_key_value_heads) * ap.k * threads, threads * blocks);
                try self.ctx.dispatch(if (self.wide_attention and !layer.kv_bf16) "attention_mlx_merge_wide" else "attention_mlx_merge", &.{ s.attention_scores, s.attention }, ap, @as(usize, c.num_attention_heads) * (resident.certified.mlx_attention_lanes * resident.certified.mlx_attention_merge_simds), resident.certified.mlx_attention_lanes * resident.certified.mlx_attention_merge_simds);
            } else {
                try self.ctx.dispatch(if (layer.kv_bf16) "attention_mlx_single_bf16" else if (self.ring_attention and ap.capacity == 513 and ap.window == 513 and ap.pos >= 512) "attention_mlx_single_ring513" else if (self.wide_attention) "attention_mlx_single_wide" else "attention_mlx_single", &.{ s.q, layer.keys, layer.values, layer.sink.buffer, s.attention }, ap, @as(usize, c.num_attention_heads) * (resident.certified.mlx_attention_lanes * resident.certified.mlx_attention_single_simds), resident.certified.mlx_attention_lanes * resident.certified.mlx_attention_single_simds);
            }
        } else if (self.attention_chunk != 0 and length >= 512 and self.splitAttentionSupported()) {
            ap.bits = self.attention_chunk;
            ap.k = (length + ap.bits - 1) / ap.bits;
            const threads = resident.certified.attention_simds * resident.certified.attention_lanes;
            try self.ctx.dispatch("attention_scores", &.{ s.q, layer.keys, s.attention_scores }, ap, @as(usize, c.num_attention_heads) * ap.k * threads, threads);
            try self.ctx.dispatch("attention_ordered_values", &.{ s.attention_scores, layer.values, layer.sink.buffer, s.attention }, ap, @as(usize, c.num_attention_heads) * threads, threads);
        } else {
            const parallel_attention = self.position >= 4;
            const attention_group: usize = if (parallel_attention) 256 else 32;
            try self.ctx.dispatch(if (parallel_attention) "attention_parallel" else "attention", &.{ s.q, layer.keys, layer.values, layer.sink.buffer, s.attention }, ap, @as(usize, c.num_attention_heads) * attention_group, attention_group);
        }
        try self.trace(i, "attention", s.attention, c.num_attention_heads * c.head_dim);
    }
    fn encodePostNorm(self: *Model, layer: Layer, i: usize) !void {
        const c = self.cfg;
        const s = self.scratch;
        if (c.hidden_size <= 8192) {
            var np = layer.post_norm.params();
            np.n = c.hidden_size;
            np.eps = @floatCast(c.rms_norm_eps);
            const aligned = self.mlx_attention and c.hidden_size == resident.certified.aligned_norm_dim;
            const norm_threads: usize = if (aligned) resident.certified.aligned_norm_threads else 256;
            if (aligned and self.prepared_inputs and self.prepared_fused_norm) {
                const prepared = self.quant_prepared orelse return error.PreparedInputsNotPrepared;
                try self.ctx.dispatch("residual_norm_aligned_prepared", &.{ layer.post_norm.buffer, s.x, s.projection, s.norm, prepared }, np, norm_threads, norm_threads);
            } else try self.ctx.dispatch(if (aligned) "residual_norm_aligned" else "residual_norm", &.{ layer.post_norm.buffer, s.x, s.projection, s.norm }, np, norm_threads, norm_threads);
        } else {
            try self.add(s.x, s.projection, 1);
            try self.norm(layer.post_norm, s.x, s.norm, 1, c.hidden_size);
        }
        try self.trace(i, "residual", s.x, c.hidden_size);
        try self.trace(i, "post_norm", s.norm, c.hidden_size);
    }
    fn encodeMlp(self: *Model, layer: Layer, i: usize) !void {
        const c = self.cfg;
        const s = self.scratch;
        if (layer.dense != null) {
            try self.fusedMLP(layer, 0, 0, c.intermediate_size);
        } else {
            try self.matvec(layer.router.?, s.norm, s.router);
            try self.trace(i, "router", s.router, c.num_experts);
            const route_kernel: [:0]const u8 = if (c.num_experts <= resident.certified.cached_router_experts) "route_topk_cached" else "route_topk";
            try self.ctx.dispatch(route_kernel, &.{ s.router, layer.bias.?, s.route_ids, s.route_weights, s.route_error }, .{ .n = c.num_experts_per_tok, .m = c.num_experts }, 32, 32);
            try self.trace(i, "route_ids", s.route_ids, c.num_experts_per_tok);
            try self.trace(i, "route_weights", s.route_weights, c.num_experts_per_tok);
            try self.fusedMLP(layer, c.num_experts_per_tok, c.num_experts, @max(c.moe_intermediate_size, c.shared_expert_intermediate_size));
        }
        try self.trace(i, "output", s.x, c.hidden_size);
    }
    fn forwardImpl(self: *Model, token: u32, emit_logits: bool) ![]const f32 {
        if (self.batch) |b| if (b.count != 0) return error.BatchPendingCommit;
        if (self.kv_bf16 and !self.mlx_attention) return error.Bf16KvRequiresMatchedStaged;
        if (self.bf16_blocks != 0 and (!self.kv_bf16 or (self.bf16_blocks != 128 and self.bf16_blocks != 512 and self.bf16_blocks != 1024))) return error.InvalidBf16Blocks;
        if (self.bf16_blocks != 0) {
            const words = try std.math.mul(usize, self.cfg.num_attention_heads, @as(usize, self.bf16_blocks) * resident.certified.mlx_attention_state_stride);
            if (words > self.scratch.attention_scores.len / @sizeOf(f32)) return error.Bf16ScratchTooSmall;
        }
        for ([_]u32{ self.head_row_tile, self.qkv_row_tile, self.gate_row_tile, self.projection_row_tile, self.down_row_tile }) |tile|
            if (tile != 0 and !resident.certified.validRowTile(tile)) return error.InvalidStagedRowTile;
        if (self.prepared_fused_norm and (!self.prepared_inputs or self.cfg.hidden_size != 2048)) return error.InvalidPreparedNorm;
        if (self.prepared_fused_gate and !self.prepared_inputs) return error.InvalidPreparedGate;
        if (self.prepared_inputs and (self.quant_prepared == null or self.activation_prepared == null or !self.mlx_attention or self.parallel_down or self.resident_execution != null or self.batch != null)) return error.InvalidPreparedExecution;
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
        try self.trace(0, "embedding", s.x, c.hidden_size);
        for (self.layers, 0..) |layer, i| {
            try self.encodeInput(layer, i);
            try self.encodeAttention(layer, i);
            try self.matvecTiled(layer.o, s.attention, s.projection, if (self.projection_row_tile == 0) self.staged_row_tile else self.projection_row_tile);
            try self.trace(i, "projection", s.projection, c.hidden_size);
            try self.encodePostNorm(layer, i);
            try self.encodeMlp(layer, i);
        }
        if (emit_logits) {
            try self.norm(self.final_norm, s.x, s.norm, 1, c.hidden_size);
            try self.matvecTiled(self.head, s.norm, s.logits, if (self.head_row_tile == 0) self.staged_row_tile else self.head_row_tile);
        }
        try self.ctx.finish();
        const after = self.ctx.stats();
        self.last_execution = .{ .submissions = after.submissions - before.submissions, .dispatches = after.dispatches - before.dispatches, .barriers = after.barriers - before.barriers, .gpu_seconds = after.gpu_seconds - before.gpu_seconds };
        if (s.route_error.structs(u32)[0] != 0) return error.NonFiniteLogits;
        self.position += 1;
        return if (emit_logits) s.logits.floats() else &.{};
    }
};

test "prepared single-token input cache rejects speculative batch APIs before allocation" {
    var model: Model = undefined;
    model.prepared_inputs = true;
    try std.testing.expectError(error.PreparedInputsRequireOrdinaryMatched, model.prepareBatch());
    try std.testing.expectError(error.PreparedInputsRequireOrdinaryMatched, model.forwardBatch(&.{0}));
}

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
