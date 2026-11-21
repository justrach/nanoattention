//! Native text graph for the pinned Inkling-2 Mini checkpoint.
//! The language-model path intentionally excludes vision until it has its own
//! validated tensor/operator implementation.
const std = @import("std");
const ir = @import("ir.zig");
const packed_weights = @import("packed.zig");

pub const LayerKind = enum { local, global };

pub const Config = struct {
    hidden: u32,
    layers: []LayerKind,
    heads: u32,
    kv_heads: u32,
    head_dim: u32,
    rel_dim: u32,
    rel_extent: u32,
    window: u32,
    conv_taps: u32,
    eps: f32,
    vocab: u32,
    unpadded_vocab: u32,
    eos: []u32,
    max_context: u32,
    intermediate: u32,
    experts: u32,
    top_k: u32,
    route_scale: f32,
    gate_softcap: f32,
    up_softcap: f32,
    final_softcap: f32,
    block_layers: u32,
    loop_start: u32,
    loop_end: u32,
    loop_passes: u32,
    gate_rank: u32,
    qk_clip: f32,
    rel_clip: f32,
    log_scaling_floor: u32,
    log_scaling_alpha: f32,

    pub fn logicalVisits(self: Config) u32 {
        return self.loop_start + (self.loop_end - self.loop_start) * self.loop_passes + (@as(u32, @intCast(self.layers.len)) - self.loop_end);
    }
};

pub const VisitStage = enum { prelude, loop, coda };
pub const Visit = struct {
    logical_index: u32,
    physical_layer: u32,
    weight_layer: u32,
    state_slot: u32,
    pass: u32,
    stage: VisitStage,
};

/// Expand the physical layer list into logical visits. Weights are keyed by
/// physical layer; recurrent KV/convolution state is keyed by logical visit.
pub fn visitSchedule(a: std.mem.Allocator, c: Config) ![]Visit {
    const schedule = try a.alloc(Visit, c.logicalVisits());
    var index: u32 = 0;
    for (0..c.loop_start) |layer| {
        schedule[index] = .{ .logical_index = index, .physical_layer = @intCast(layer), .weight_layer = @intCast(layer), .state_slot = index, .pass = 0, .stage = .prelude };
        index += 1;
    }
    for (0..c.loop_passes) |pass| {
        for (c.loop_start..c.loop_end) |layer| {
            schedule[index] = .{ .logical_index = index, .physical_layer = @intCast(layer), .weight_layer = @intCast(layer), .state_slot = index, .pass = @intCast(pass), .stage = .loop };
            index += 1;
        }
    }
    for (c.loop_end..c.layers.len) |layer| {
        schedule[index] = .{ .logical_index = index, .physical_layer = @intCast(layer), .weight_layer = @intCast(layer), .state_slot = index, .pass = 0, .stage = .coda };
        index += 1;
    }
    if (index != schedule.len) return error.InvalidLoopGeometry;
    return schedule;
}

fn object(value: ?std.json.Value) !std.json.ObjectMap {
    const v = value orelse return error.MissingConfigField;
    return switch (v) {
        .object => |o| o,
        else => error.InvalidConfigObject,
    };
}
fn integer(value: ?std.json.Value, default: ?u32) !u32 {
    const v = value orelse return default orelse error.MissingConfigField;
    return switch (v) {
        .integer => |n| std.math.cast(u32, n) orelse error.InvalidConfigInteger,
        else => error.InvalidConfigInteger,
    };
}
fn number(value: ?std.json.Value, default: ?f64) !f64 {
    const v = value orelse return default orelse error.MissingConfigField;
    return switch (v) {
        .integer => |n| @floatFromInt(n),
        .float => |n| n,
        else => error.InvalidConfigNumber,
    };
}
fn string(value: ?std.json.Value) ![]const u8 {
    const v = value orelse return error.MissingConfigField;
    return switch (v) {
        .string => |s| s,
        else => error.InvalidConfigString,
    };
}
fn expectString(o: std.json.ObjectMap, key: []const u8, expected: []const u8) !void {
    if (!std.mem.eql(u8, try string(o.get(key)), expected)) return error.UnsupportedConfigOption;
}
fn eosIds(a: std.mem.Allocator, value: ?std.json.Value) ![]u32 {
    const v = value orelse return error.MissingConfigField;
    switch (v) {
        .integer => |n| return try a.dupe(u32, &.{std.math.cast(u32, n) orelse return error.InvalidConfigInteger}),
        .array => |arr| {
            const ids = try a.alloc(u32, arr.items.len);
            for (arr.items, ids) |item, *id| id.* = switch (item) {
                .integer => |n| std.math.cast(u32, n) orelse return error.InvalidConfigInteger,
                else => return error.InvalidConfigInteger,
            };
            return ids;
        },
        else => return error.InvalidConfigInteger,
    }
}

/// Parse only the released Inkling2Loop text configuration and reject options
/// whose semantics are not represented by the current native backend.
fn expectBool(o: std.json.ObjectMap, key: []const u8, expected: bool) !void {
    const value = o.get(key) orelse return error.MissingConfigField;
    if (value != .bool or value.bool != expected) return error.UnsupportedConfigOption;
}

/// Parse only the released Inkling2Loop text configuration and reject options
/// whose semantics are not represented by the current native backend.
pub fn parse(a: std.mem.Allocator, bytes: []const u8) !Config {
    const parsed = try std.json.parseFromSliceLeaky(std.json.Value, a, bytes, .{});
    const root = try object(parsed);
    if (!std.mem.eql(u8, try string(root.get("model_type")), "inkling2_loop")) return error.NotInkling2Loop;
    try expectString(root, "dtype", "bfloat16");
    const text = try object(root.get("text_config"));
    try expectString(text, "model_type", "inkling2_loop_text");
    try expectString(text, "torch_dtype", "bfloat16");

    const hidden = try integer(text.get("hidden_size"), null);
    const n_layers = try integer(text.get("num_hidden_layers"), null);
    const heads = try integer(text.get("num_attention_heads"), null);
    const kv_heads = try integer(text.get("num_key_value_heads"), null);
    const head_dim = try integer(text.get("head_dim"), null);
    const rel_dim = try integer(text.get("d_rel"), null);
    const rel_extent = try integer(text.get("rel_extent"), null);
    const window = try integer(text.get("sliding_window_size"), null);
    const conv_taps = try integer(text.get("sconv_kernel_size"), null);
    const vocab = try integer(text.get("vocab_size"), null);
    const unpadded_vocab = try integer(text.get("unpadded_vocab_size"), null);
    const max_context = try integer(text.get("model_max_length"), null);
    const intermediate = try integer(text.get("intermediate_size"), null);
    const experts = try integer(text.get("n_routed_experts"), null);
    const top_k = try integer(text.get("num_experts_per_tok"), null);
    const block_layers = try integer(text.get("n_layers_per_attn_res_block"), null);
    const loop_cfg = try object(text.get("loop_config"));
    const loop_start = try integer(loop_cfg.get("core_start"), null);
    const loop_end = try integer(loop_cfg.get("core_end"), null);
    const loop_passes = try integer(loop_cfg.get("num_passes"), null);
    const gate_rank = try integer(loop_cfg.get("gate_rank"), null);
    try expectBool(loop_cfg, "is_varloop", false);

    try expectBool(text, "q_bias", false);
    try expectBool(text, "o_bias", false);
    try expectBool(text, "use_value_norm_no_gamma", true);
    try expectBool(text, "attn_res_normalize_values", true);
    try expectBool(text, "use_gate_bias", true);
    if (try integer(text.get("n_shared_experts"), null) != 0 or
        try integer(text.get("swa_head_dim"), null) != head_dim or
        try integer(text.get("swa_num_attention_heads"), null) != heads or
        try integer(text.get("swa_num_key_value_heads"), null) != kv_heads)
        return error.UnsupportedConfigOption;

    const local_value = text.get("local_layer_ids") orelse return error.MissingConfigField;
    if (local_value != .array) return error.InvalidLayerTypes;
    const layers = try a.alloc(LayerKind, n_layers);
    const seen = try a.alloc(bool, n_layers);
    @memset(layers, .global);
    @memset(seen, false);
    for (local_value.array.items) |value| {
        const layer: u32 = switch (value) {
            .integer => |n| std.math.cast(u32, n) orelse return error.InvalidConfigInteger,
            else => return error.InvalidLayerTypes,
        };
        if (layer >= n_layers or seen[layer]) return error.InvalidLayerTypes;
        seen[layer] = true;
        layers[layer] = .local;
    }

    const c: Config = .{
        .hidden = hidden,
        .layers = layers,
        .heads = heads,
        .kv_heads = kv_heads,
        .head_dim = head_dim,
        .rel_dim = rel_dim,
        .rel_extent = rel_extent,
        .window = window,
        .conv_taps = conv_taps,
        .eps = @floatCast(try number(text.get("rms_norm_eps"), null)),
        .vocab = vocab,
        .unpadded_vocab = unpadded_vocab,
        .eos = try eosIds(a, root.get("eos_token_id")),
        .max_context = max_context,
        .intermediate = intermediate,
        .experts = experts,
        .top_k = top_k,
        .route_scale = @floatCast(try number(text.get("route_scale"), null)),
        .gate_softcap = @floatCast(try number(text.get("mlp_gate_softcap"), null)),
        .up_softcap = @floatCast(try number(text.get("mlp_up_softcap"), null)),
        .final_softcap = @floatCast(try number(text.get("final_logit_softcapping"), null)),
        .block_layers = block_layers,
        .loop_start = loop_start,
        .loop_end = loop_end,
        .loop_passes = loop_passes,
        .gate_rank = gate_rank,
        .qk_clip = @floatCast(try number(text.get("qk_l2_clip_threshold"), null)),
        .rel_clip = @floatCast(try number(text.get("rel_l2_clip_threshold"), null)),
        .log_scaling_floor = try integer(text.get("log_scaling_n_floor"), null),
        .log_scaling_alpha = @floatCast(try number(text.get("log_scaling_alpha"), null)),
    };
    if (hidden == 0 or n_layers == 0 or heads == 0 or kv_heads == 0 or head_dim == 0 or
        heads % kv_heads != 0 or rel_dim == 0 or rel_extent == 0 or window == 0 or conv_taps == 0 or
        vocab == 0 or unpadded_vocab == 0 or unpadded_vocab > vocab or max_context == 0 or
        intermediate == 0 or experts == 0 or top_k == 0 or top_k > experts or block_layers == 0 or
        loop_passes == 0 or gate_rank == 0 or loop_start == 0 or loop_start >= loop_end or loop_end > n_layers or
        loop_start % block_layers != 0 or loop_end % block_layers != 0 or (loop_end - loop_start) % block_layers != 0 or
        !(c.eps > 0) or !(c.route_scale > 0) or !(c.gate_softcap > 0) or !(c.up_softcap > 0) or !(c.final_softcap > 0) or
        !(c.qk_clip > 0) or !(c.rel_clip > 0) or !(c.log_scaling_alpha >= 0)) return error.InvalidConfig;
    if (head_dim != 128 or hidden != 2304 or n_layers != 24 or heads != 24 or kv_heads != 8 or rel_dim != 16 or
        rel_extent != 1024 or window != 1024 or conv_taps != 4 or vocab != 201024 or unpadded_vocab != 200058 or
        max_context != 1048576 or intermediate != 1152 or block_layers != 6 or loop_start != 6 or loop_end != 18 or
        loop_passes != 3 or experts != 128 or top_k != 8 or gate_rank != 64 or
        c.eps != @as(f32, 0.000001) or c.route_scale != 8 or c.gate_softcap != 4 or c.up_softcap != 25 or c.final_softcap != 30 or
        c.qk_clip != @as(f32, 22.627416997969522) or c.rel_clip != 8 or c.log_scaling_floor != 8192 or c.log_scaling_alpha != @as(f32, 0.1))
        return error.UnsupportedConfigOption;
    for (c.eos) |id| if (id >= vocab) return error.InvalidConfig;
    if (local_value.array.items.len != n_layers - n_layers / block_layers) return error.UnsupportedLayerLayout;
    for (layers, 0..) |kind, i| {
        const expected_global = (i + 1) % block_layers == 0;
        if ((kind == .global) != expected_global) return error.UnsupportedLayerLayout;
    }
    return c;
}

const test_config =
    \\{"model_type":"inkling2_loop","dtype":"bfloat16","eos_token_id":200006,
    \\"text_config":{"model_type":"inkling2_loop_text","torch_dtype":"bfloat16",
    \\"model_max_length":1048576,
    \\"hidden_size":2304,"num_hidden_layers":24,"vocab_size":201024,
    \\"num_attention_heads":24,"num_key_value_heads":8,"head_dim":128,"d_rel":16,
    \\"rel_extent":1024,"q_bias":false,"o_bias":false,"log_scaling_n_floor":8192,
    \\"log_scaling_alpha":0.1,"rms_norm_eps":0.000001,"use_value_norm_no_gamma":true,
    \\"qk_l2_clip_threshold":22.627416997969522,"rel_l2_clip_threshold":8,
    \\"local_layer_ids":[0,1,2,3,4,6,7,8,9,10,12,13,14,15,16,18,19,20,21,22],
    \\"sconv_kernel_size":4,"unpadded_vocab_size":200058,"final_logit_softcapping":30,
    \\"swa_head_dim":128,"swa_num_attention_heads":24,"swa_num_key_value_heads":8,
    \\"sliding_window_size":1024,"n_layers_per_attn_res_block":6,
    \\"attn_res_normalize_values":true,"n_routed_experts":128,"num_experts_per_tok":8,
    \\"n_shared_experts":0,"intermediate_size":1152,"mlp_gate_softcap":4,
    \\"mlp_up_softcap":25,"route_scale":8,"use_gate_bias":true,
    \\"loop_config":{"core_start":6,"core_end":18,"num_passes":3,"gate_rank":64,"is_varloop":false}}}
;

test "Inkling2 Mini loop geometry is pinned to 48 visits" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const c = try parse(arena.allocator(), test_config);
    try std.testing.expectEqual(@as(u32, 48), c.logicalVisits());
    try std.testing.expectEqual(@as(usize, 24), c.layers.len);
    try std.testing.expectEqual(@as(u32, 200058), c.unpadded_vocab);
}

test "visit schedule shares physical weights and separates logical state" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const c = try parse(arena.allocator(), test_config);
    const schedule = try visitSchedule(arena.allocator(), c);
    try std.testing.expectEqual(@as(usize, 48), schedule.len);
    for (schedule, 0..) |visit, i| {
        try std.testing.expectEqual(@as(u32, @intCast(i)), visit.logical_index);
        try std.testing.expectEqual(visit.logical_index, visit.state_slot);
    }
    const layer_six_first = schedule[6];
    const layer_six_second = schedule[18];
    const layer_six_third = schedule[30];
    try std.testing.expectEqual(@as(u32, 6), layer_six_first.physical_layer);
    try std.testing.expectEqual(layer_six_first.weight_layer, layer_six_second.weight_layer);
    try std.testing.expectEqual(layer_six_first.weight_layer, layer_six_third.weight_layer);
    try std.testing.expect(layer_six_first.state_slot != layer_six_second.state_slot);
    try std.testing.expect(layer_six_second.state_slot != layer_six_third.state_slot);
}

test "Inkling2 rejects unsupported loop geometry" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const malformed = std.mem.replaceOwned(u8, arena.allocator(), test_config, "\"num_passes\":3", "\"num_passes\":2") catch unreachable;
    try std.testing.expectError(error.UnsupportedConfigOption, parse(arena.allocator(), malformed));
}

// Checkpoint names and math pinned to the released safetensors header and the
// Inkling 2 Mini MLX reference (jundot/mlx-lm 0c4c99192230f).
pub fn build(gpa: std.mem.Allocator, bytes: []const u8) !ir.Program {
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    return lower(gpa, try parse(arena.allocator(), bytes));
}

const Lowering = struct {
    b: *ir.Builder,
    c: Config,

    fn scratch(self: Lowering, name: []const u8, width: u32) !ir.View {
        return self.b.view(try self.b.state(.{ .name = name, .kind = .scratch, .width = width }));
    }
    fn weight(self: Lowering, name: []const u8, shape: []const usize) !ir.WeightId {
        return self.b.weight(try self.b.print("model.llm.{s}", .{name}), shape);
    }
    fn layerWeight(self: Lowering, layer: u32, name: []const u8, shape: []const usize) !ir.WeightId {
        return self.weight(try self.b.print("layers.{d}.{s}", .{ layer, name }), shape);
    }
    fn slice(v: ir.View, index: u32, width: u32) ir.View {
        return .{ .state = v.state, .col = v.col + index * width, .width = width };
    }
    fn transform(self: Lowering, x: ir.View, out: ir.View, t: ir.RowTransform, rounding: ir.Round) !void {
        try self.b.op(.{ .row_transform = .{ .x = x, .out = out, .transform = t, .round = rounding } });
    }
    fn rounded(self: Lowering, x: ir.View, rounding: ir.Round) !ir.View {
        if (rounding != .none) try self.transform(x, x, .copy, rounding);
        return x;
    }
    fn norm(self: Lowering, x: ir.View, w: ?ir.WeightId, group: u32, rounding: ir.Round) !ir.View {
        const out = try self.scratch("norm", x.width);
        try self.b.op(.{ .rmsnorm = .{ .x = x, .weight = w, .out = out, .group = group, .eps = self.c.eps } });
        return self.rounded(out, rounding);
    }
    fn normBf16Input(self: Lowering, x: ir.View, w: ir.WeightId) !ir.View {
        const out = try self.scratch("norm", x.width);
        try self.b.op(.{ .rmsnorm = .{ .x = x, .weight = w, .out = out, .group = x.width, .eps = self.c.eps, .round_normalized = .bf16 } });
        return self.rounded(out, .bf16);
    }
    fn dense(self: Lowering, x: ir.View, w: ir.WeightId, n: u32, rounding: ir.Round) !ir.View {
        const out = try self.scratch("linear", n);
        try self.b.op(.{ .linear = .{ .w = w, .x = x, .out = out, .input_bf16 = true } });
        return self.rounded(out, rounding);
    }
    fn layerDense(self: Lowering, layer: u32, x: ir.View, name: []const u8, n: u32, rounding: ir.Round) !ir.View {
        return self.dense(x, try self.layerWeight(layer, name, &.{ n, x.width }), n, rounding);
    }
    fn binary(self: Lowering, kind: ir.BinaryKind, x: ir.View, y: ir.View, y_act: ir.Act, rounding: ir.Round) !ir.View {
        const out = try self.scratch("binary", x.width);
        try self.b.op(.{ .binary = .{ .kind = kind, .a = x, .b = y, .b_act = y_act, .out = out } });
        return self.rounded(out, rounding);
    }
    fn concat(self: Lowering, x: ir.View, y: ir.View) !ir.View {
        const out = try self.scratch("concat", x.width + y.width);
        try self.b.op(.{ .concat = .{ .a = x, .b = y, .out = out } });
        return out;
    }
    fn depth(self: Lowering, bank: ir.View, partial: ir.View, query: ir.WeightId, index: u32) !ir.View {
        const out = try self.scratch("depth", partial.width);
        const scale = 1 / @sqrt(@as(f32, @floatFromInt(partial.width)));
        try self.b.op(.{ .depth_attention = .{ .bank = bank, .partial = partial, .query = query, .query_index = index, .out = out, .scale = scale } });
        return out;
    }
    fn cache(self: Lowering, v: Visit, name: []const u8, width: u32, horizon: u32) !ir.StateId {
        return self.b.state(.{ .name = try self.b.print("visit.{d}.{s}", .{ v.state_slot, name }), .width = width, .horizon = horizon, .kind = if (horizon > 0) .ring else .sequence });
    }
    fn conv(self: Lowering, v: Visit, x: ir.View, name: []const u8, dynamic_input: ?ir.View) !ir.View {
        const c = self.c;
        const state = try self.cache(v, name, x.width, c.conv_taps - 1);
        try self.b.op(.{ .store = .{ .src = x, .dst = state } });
        const out = try self.scratch("sconv", x.width);
        if (dynamic_input) |d| {
            const sw = try self.layerWeight(v.weight_layer, try self.b.print("{s}.static_w", .{name}), &.{ c.conv_taps, x.width });
            const taps = try self.layerDense(v.weight_layer, d, try self.b.print("{s}.weight_proj", .{name}), c.kv_heads * c.conv_taps, .bf16);
            try self.b.op(.{ .inkling_dynamic_conv = .{ .src = state, .static_weight = sw, .dynamic = taps, .residual = x, .out = out, .heads = c.kv_heads, .taps = c.conv_taps, .round = .bf16 } });
        } else {
            const w = try self.layerWeight(v.weight_layer, try self.b.print("{s}.weight", .{name}), &.{ x.width, 1, c.conv_taps });
            try self.b.op(.{ .inkling_short_conv = .{ .src = state, .weight = w, .residual = x, .out = out, .taps = c.conv_taps, .round = .bf16 } });
        }
        return out;
    }
    fn clip(self: Lowering, x: ir.View, group: u32, threshold: f32, w: ?ir.WeightId) !void {
        try self.b.op(.{ .l2_clip = .{ .x = x, .out = x, .group = group, .threshold = threshold, .scale = w } });
        _ = try self.rounded(x, .bf16);
    }
    fn attention(self: Lowering, v: Visit, inputs: [3]ir.View) !ir.View {
        const c = self.c;
        const layer = v.weight_layer;
        const global = c.layers[layer] == .global;
        const q_dim = c.heads * c.head_dim;
        const kv_dim = c.kv_heads * c.head_dim;
        const q = try self.layerDense(layer, inputs[0], "attn.wq_du.weight", q_dim, .bf16);
        const k = try self.layerDense(layer, inputs[1], "attn.wk_dv.weight", kv_dim, .bf16);
        const value = try self.layerDense(layer, inputs[2], "attn.wv_dv.weight", kv_dim, .bf16);
        const r = try self.layerDense(layer, inputs[0], "attn.wr_du.weight", c.heads * c.rel_dim, .bf16);
        const gate = try self.layerDense(layer, inputs[0], "attn.gate_attn.weight", q_dim, .bf16);
        const sink = try self.layerDense(layer, inputs[0], "attn.attn_sink_proj.weight", c.heads, .bf16);
        const ck = try self.conv(v, k, "attn.k_sconv", if (global) inputs[1] else null);
        const cv = try self.conv(v, value, "attn.v_sconv", if (global) inputs[2] else null);
        const nv = try self.norm(cv, null, c.head_dim, .bf16);
        try self.clip(q, c.head_dim, c.qk_clip, try self.layerWeight(layer, "attn.q_l2_clip.scale", &.{ c.heads, c.head_dim }));
        try self.clip(ck, c.head_dim, c.qk_clip, null);
        try self.clip(r, c.rel_dim, c.rel_clip, try self.layerWeight(layer, "attn.rel_l2_clip.scale", &.{ c.heads, c.rel_dim }));
        if (global) {
            const tau: ir.RowTransform = .{ .log_position_scale = .{ .floor = c.log_scaling_floor, .alpha = c.log_scaling_alpha } };
            try self.transform(q, q, tau, .bf16);
            try self.transform(r, r, tau, .bf16);
        }
        const extent = if (global) c.rel_extent else c.window;
        const bias = try self.scratch("relative_bias", c.heads * extent);
        const rel = try self.layerWeight(layer, "attn.rel_logits_proj.proj", &.{ c.rel_dim, extent });
        try self.b.op(.{ .grouped_project = .{ .x = r, .out = bias, .weight = rel, .groups = c.heads, .round = .bf16, .causal_prefix = true } });
        const window = if (global) 0 else c.window;
        const keys = try self.cache(v, "keys", kv_dim, window -| 1);
        const values = try self.cache(v, "values", kv_dim, window -| 1);
        try self.b.op(.{ .store = .{ .src = ck, .dst = keys } });
        try self.b.op(.{ .store = .{ .src = nv, .dst = values } });
        const out = try self.scratch("attended", q_dim);
        try self.b.op(.{ .relative_attention = .{ .q = q, .bias = bias, .sink = sink, .keys = keys, .values = values, .out = out, .heads = c.heads, .kv_heads = c.kv_heads, .dim = c.head_dim, .relative_extent = extent, .window = window, .scale = 1 / @sqrt(@as(f32, @floatFromInt(c.head_dim))) } });
        const gated = try self.binary(.mul, try self.rounded(out, .bf16), gate, .sigmoid, .bf16);
        return self.layerDense(layer, gated, "attn.wo_ud.weight", c.hidden, .bf16);
    }
    const Bank = struct { blocks: ir.View, last: ir.View };
    fn message(self: Lowering, v: Visit, input: ir.View, bank: Bank, stream: usize) !ir.View {
        const c = self.c;
        const local = v.physical_layer - c.loop_start;
        const key_value = stream == 1 or stream == 2;
        const w = try self.weight(if (key_value) "middle_cycle_cross_pass_kv_queries" else "middle_cycle_cross_pass_queries", &.{ 2 * (c.loop_end - c.loop_start), c.hidden });
        const msg = try self.depth(bank.blocks, bank.last, w, 2 * local + @as(u32, @intFromBool(stream >= 2)));
        const stem = [_][]const u8{ "attn", "k", "v", "mlp" };
        const cat = try self.concat(try self.norm(input, null, c.hidden, .none), try self.norm(msg, null, c.hidden, .none));
        const d = try self.layerDense(v.weight_layer, try self.rounded(cat, .bf16), try self.b.print("controllers.{s}.down_weight", .{stem[stream]}), c.gate_rank, .bf16);
        const f = try self.layerDense(v.weight_layer, d, try self.b.print("controllers.{s}.forget_weight", .{stem[stream]}), c.hidden, .none);
        const g = try self.layerDense(v.weight_layer, d, try self.b.print("controllers.{s}.gain_weight", .{stem[stream]}), c.hidden, .none);
        const out = try self.scratch("message_merge", c.hidden);
        try self.b.op(.{ .inkling_message_merge = .{ .hidden = input, .message = msg, .gates = try self.concat(f, g), .out = out } });
        return out;
    }
    fn visit(self: Lowering, v: Visit, hidden: ir.View, blocks: ir.View, count: *u32, prior: ?Bank) !ir.View {
        const c = self.c;
        const h = c.hidden;
        const layer = v.weight_layer;
        const global = c.layers[layer] == .global;
        const partial = try self.norm(hidden, null, h, .none);
        var inputs: [3]ir.View = @splat(hidden);
        if (layer > 0) {
            const name = try self.b.print("attn_res.attn_queries.{d}", .{layer - 1});
            const query = try self.weight(name, if (global) &.{ 3, h } else &.{h});
            const bank = slice(blocks, 0, count.* * h);
            if (global) {
                for (&inputs, 0..) |*x, i| x.* = try self.depth(bank, partial, query, @intCast(i));
            } else inputs = @splat(try self.depth(bank, partial, query, 0));
        }
        const norms = [_][]const u8{ "q_input_norm.weight", "k_input_norm.weight", "v_input_norm.weight" };
        if (global or prior != null) {
            for (&inputs, 0..) |*x, i| {
                if (prior) |bank| x.* = try self.message(v, x.*, bank, i);
                x.* = try self.norm(x.*, try self.layerWeight(layer, if (global) norms[i] else "attn_norm.weight", &.{h}), h, .bf16);
            }
        } else {
            const w = try self.layerWeight(layer, "attn_norm.weight", &.{h});
            inputs = @splat(if (layer == 0) try self.normBf16Input(inputs[0], w) else try self.norm(inputs[0], w, h, .bf16));
        }
        const attn = try self.conv(v, try self.attention(v, inputs), "attn_sconv", null);
        const residual = if (layer % c.block_layers == 0) blk: {
            try self.transform(partial, slice(blocks, count.*, h), .copy, .none);
            count.* += 1;
            break :blk attn;
        } else try self.binary(.add, hidden, attn, .identity, .none);
        const mp = try self.norm(residual, null, h, .none);
        const mq = try self.weight(try self.b.print("attn_res.mlp_queries.{d}", .{layer}), &.{h});
        var m = try self.depth(slice(blocks, 0, count.* * h), mp, mq, 0);
        if (prior) |bank| m = try self.message(v, m, bank, 3);
        m = try self.norm(m, try self.layerWeight(layer, "mlp_norm.weight", &.{h}), h, .bf16);
        const route_logits = try self.layerDense(layer, m, "mlp.gate.weight", c.experts, .none);
        const correction = try self.layerWeight(layer, "mlp.gate.bias", &.{c.experts});
        const routes = try self.scratch("routes", 2 * c.top_k);
        try self.b.op(.{ .route_topk = .{ .logits = route_logits, .correction = correction, .out = routes, .experts = c.experts, .top_k = c.top_k, .eps = 1e-20, .scale = c.route_scale } });
        const experts = try packed_weights.declareLayer(self.b, layer, c.experts, h, c.intermediate);
        const up = try self.scratch("experts_up", c.top_k * c.intermediate);
        const down = try self.scratch("experts_down", h);
        try self.b.op(.{ .expert_gate_up = .{ .x = m, .routes = routes, .out = up, .bank = experts.gate_up, .experts = c.experts, .top_k = c.top_k, .gate_cap = c.gate_softcap, .up_cap = c.up_softcap, .round = .bf16, .input_bf16 = true } });
        try self.b.op(.{ .expert_down = .{ .x = up, .routes = routes, .out = down, .bank = experts.down, .experts = c.experts, .top_k = c.top_k, .round = .bf16, .input_bf16 = true } });
        const ff = try self.conv(v, down, "mlp_sconv", null);
        return self.binary(.add, residual, ff, .identity, .bf16);
    }
};

fn lower(gpa: std.mem.Allocator, c: Config) !ir.Program {
    var b = ir.Builder.init(gpa);
    errdefer b.deinit();
    const l: Lowering = .{ .b = &b, .c = c };
    const h = c.hidden;
    const tokens = b.view(try b.state(.{ .name = "tokens", .kind = .tokens, .width = 1 }));
    _ = try b.state(.{ .name = "position", .kind = .position, .width = 1 });
    const embed = try l.scratch("embedding", h);
    try b.op(.{ .embed = .{ .table = try l.weight("embed.weight", &.{ c.vocab, h }), .tokens = tokens, .out = embed } });
    var hidden = try l.normBf16Input(embed, try l.weight("embed_norm.weight", &.{h}));
    try b.trace(hidden);
    const blocks = try l.scratch("attention_blocks", @as(u32, @intCast(c.layers.len)) / c.block_layers * h);
    const prelude_blocks = c.loop_start / c.block_layers;
    const message_blocks = (c.loop_end - c.loop_start) / c.block_layers - 1;
    const saved = try l.scratch("prior_pass_blocks", message_blocks * h);
    var count: u32 = 0;
    var prelude: ir.View = undefined;
    var prior: ?Lowering.Bank = null;
    for (try visitSchedule(b.allocator(), c)) |v| {
        if (v.stage == .loop and v.physical_layer == c.loop_start) {
            if (v.pass == 0) {
                prelude = hidden;
            } else {
                // Drop the copy of the prelude ending stored at the first core
                // boundary, then preserve the previous pass before reusing its bank.
                const start = Lowering.slice(blocks, prelude_blocks + 1, h);
                const bank: ir.View = .{ .state = start.state, .col = start.col, .width = saved.width };
                try l.transform(bank, saved, .copy, .none);
                prior = .{ .blocks = saved, .last = try l.norm(hidden, null, h, .none) };
                hidden = prelude;
                count = prelude_blocks;
            }
        }
        hidden = try l.visit(v, hidden, blocks, &count, if (v.stage == .loop) prior else null);
        try b.trace(hidden);
    }
    const head = b.mark();
    const partial = try l.norm(hidden, null, h, .none);
    const final = try l.depth(Lowering.slice(blocks, 0, count * h), partial, try l.weight("attn_res.output_query", &.{h}), 0);
    const normed = try l.norm(final, try l.weight("norm.weight", &.{h}), h, .bf16);
    const raw = try l.dense(normed, try l.weight("unembed.weight", &.{ c.vocab, h }), c.vocab, .bf16);
    const logits = try l.scratch("usable_logits", c.unpadded_vocab);
    try l.transform(raw, logits, .{ .softcap = c.final_softcap }, .none);
    const next = try l.scratch("next", 1);
    try b.op(.{ .argmax = .{ .x = logits, .out = next } });
    return b.finish(.{ .model_type = "inkling2_loop", .vocab = c.unpadded_vocab, .eos = c.eos, .max_context = c.max_context }, head, tokens, logits, next);
}

test "lowering reuses all packed physical layers and gives each visit its own state" {
    var p = try build(std.testing.allocator, test_config);
    defer p.deinit();
    try std.testing.expectEqual(@as(usize, 49), p.traces.len);
    try std.testing.expectEqual(@as(u32, 200058), p.logits.width);
    var experts: [48]ir.WeightId = undefined;
    var slots: [65536]bool = @splat(false);
    var routed: usize = 0;
    var caches: usize = 0;
    var controllers: usize = 0;
    for (p.ops, 0..) |op, i| switch (op) {
        .expert_gate_up => |e| {
            experts[routed] = e.bank.payload;
            routed += 1;
        },
        .store => |s| {
            try std.testing.expect(!slots[s.dst]);
            slots[s.dst] = true;
            caches += 1;
        },
        .inkling_message_merge => controllers += 1,
        .argmax => try std.testing.expect(i >= p.head_start),
        else => {},
    };
    try std.testing.expectEqual(@as(usize, 48), routed);
    try std.testing.expectEqual(@as(usize, 48 * 6), caches);
    try std.testing.expectEqual(@as(usize, 2 * 12 * 4), controllers);
    try std.testing.expectEqualSlices(ir.WeightId, experts[6..18], experts[18..30]);
    try std.testing.expectEqualSlices(ir.WeightId, experts[6..18], experts[30..42]);
    try std.testing.expect(experts[5] != experts[6] and experts[42] != experts[6]);
}

extern "c" fn mkdtemp([*:0]u8) ?[*:0]u8;
extern "c" fn unlink([*:0]const u8) c_int;
extern "c" fn rmdir([*:0]const u8) c_int;

fn writeFixture(a: std.mem.Allocator, p: *const ir.Program, path: []const u8) !void {
    // A real safetensors file exercises name/shape binding as well as Metal.
    var header: std.ArrayList(u8) = .empty;
    var data: std.ArrayList(u8) = .empty;
    try header.append(a, '{');
    for (p.weights, 0..) |w, i| {
        const dtype: []const u8 = if (w.raw_dtype) |t| switch (t) {
            .u8 => "U8",
            .f8_e4m3 => "F8_E4M3",
            .f32 => "F32",
        } else if (std.mem.endsWith(u8, w.name, "gate.bias") or std.mem.indexOf(u8, w.name, "queries") != null or std.mem.endsWith(u8, w.name, "output_query")) "F32" else "BF16";
        const offset = data.items.len;
        var count: usize = 1;
        for (w.shape) |dim| count *= dim;
        for (0..count) |j| {
            const varied = @as(f32, @floatFromInt(@as(i32, @intCast((j * 11 + j / 32 + i * 17) % 19)) - 9)) / 128;
            const unit_weight = std.mem.endsWith(u8, w.name, "norm.weight") or std.mem.endsWith(u8, w.name, "clip.scale");
            const v: f32 = if (unit_weight) 1 + varied else varied;
            if (std.mem.eql(u8, dtype, "F32")) {
                var bytes: [4]u8 = undefined;
                std.mem.writeInt(u32, &bytes, @bitCast(if (w.raw_dtype != null) @as(f32, 1) else v), .little);
                try data.appendSlice(a, &bytes);
            } else if (std.mem.eql(u8, dtype, "BF16")) {
                var bytes: [2]u8 = undefined;
                std.mem.writeInt(u16, &bytes, @truncate(@as(u32, @bitCast(v)) >> 16), .little);
                try data.appendSlice(a, &bytes);
            } else {
                const scales = std.mem.endsWith(u8, w.name, "weight_scale");
                const values = [_]u8{ 0x18, 0x1a, 0x9c, 0x00, 0x21, 0xa0, 0x02 };
                try data.append(a, if (scales) (if (w.raw_dtype.? == .u8) @as(u8, 125) else 0x18) else values[(j + j / 32 + i) % values.len]);
            }
        }
        try header.appendSlice(a, try std.fmt.allocPrint(a, "{s}\"{s}\":{{\"dtype\":\"{s}\",\"shape\":[", .{ if (i == 0) "" else ",", w.name, dtype }));
        for (w.shape, 0..) |dim, j| try header.appendSlice(a, try std.fmt.allocPrint(a, "{s}{d}", .{ if (j == 0) "" else ",", dim }));
        try header.appendSlice(a, try std.fmt.allocPrint(a, "],\"data_offsets\":[{d},{d}]}}", .{ offset, data.items.len }));
    }
    try header.append(a, '}');
    while (header.items.len % 8 != 0) try header.append(a, ' ');
    var prefix: [8]u8 = undefined;
    std.mem.writeInt(u64, &prefix, header.items.len, .little);
    try @import("io.zig").writeFile(a, path, try std.mem.concat(a, u8, &.{ &prefix, header.items, data.items }));
}

test "native Inkling fixture matches sequential, batched and continued replay past the window" {
    const gpa = std.testing.allocator;
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var c = try parse(a, test_config);
    c.hidden = 128;
    c.intermediate = 128;
    c.heads = 2;
    c.kv_heads = 1;
    c.head_dim = 8;
    c.rel_dim = 4;
    c.rel_extent = 4;
    c.window = 4;
    c.log_scaling_floor = 3;
    c.vocab = 16;
    c.unpadded_vocab = 14;
    c.eos = try a.dupe(u32, &.{13});
    c.gate_rank = 4;
    c.max_context = 16;
    var p = try lower(gpa, c);
    defer p.deinit();
    var tmp = "/tmp/nano-inkling-XXXXXX".*;
    const dir = std.mem.span(mkdtemp(&tmp) orelse return error.TempDirectoryFailed);
    defer _ = rmdir(dir);
    const path = try std.fmt.allocPrintSentinel(a, "{s}/model.safetensors", .{dir}, 0);
    defer _ = unlink(path);
    try writeFixture(a, &p, path);
    const ctx = try @import("gpu.zig").open();
    defer ctx.deinit();
    var store = try @import("weights").Store.init(a, ctx, dir, 0);
    defer store.deinit();
    const Engine = @import("exec.zig").Engine;
    var seq = try Engine.init(gpa, &p, ctx, &store, .{ .batch = 8 }, 16);
    defer seq.deinit();
    var batch = try Engine.init(gpa, &p, ctx, &store, .{ .batch = 8 }, 16);
    defer batch.deinit();
    var fresh = try Engine.init(gpa, &p, ctx, &store, .{ .batch = 8 }, 16);
    defer fresh.deinit();
    const tokens = [_]u32{ 1, 4, 11, 2, 3, 6, 9, 1 };
    try batch.forward(tokens[0..6], 0);
    for (tokens[0..6], 0..) |t, row| {
        try seq.forward(&.{t}, 0);
        const expected = try seq.logits(0);
        for (expected) |value| try std.testing.expect(std.math.isFinite(value));
        try std.testing.expectEqualSlices(f32, expected, try batch.logits(@intCast(row)));
    }
    try std.testing.expect(!std.mem.eql(f32, try batch.logits(0), try batch.logits(5)));
    try batch.forward(tokens[6..], 0);
    try fresh.forward(&tokens, 6);
    try std.testing.expectEqualSlices(f32, try fresh.logits(6), try batch.logits(0));
    try std.testing.expectEqualSlices(f32, try fresh.logits(7), try batch.logits(1));

    // Enable the exact tiling/cast path with a partial head tile, and the
    // expert-major path over both packed formats and all visit state.
    var tuned = try Engine.init(gpa, &p, ctx, &store, .{
        .batch = 8,
        .expert_prefill = true,
        .bf16_linear_tile = true,
        .fuse_bf16_cast = true,
    }, 16);
    defer tuned.deinit();
    try tuned.forward(&tokens, 6);
    try std.testing.expectEqualSlices(f32, try fresh.logits(6), try tuned.logits(6));
    try std.testing.expectEqualSlices(f32, try fresh.logits(7), try tuned.logits(7));

    // The wider tile stages both original packed formats with a different
    // memory layout, but should have the same matrix accumulation order.
    var wide = try Engine.init(gpa, &p, ctx, &store, .{
        .batch = 8,
        .expert_prefill = true,
        .expert_wide_k = true,
        .bf16_linear_tile = true,
        .fuse_bf16_cast = true,
    }, 16);
    defer wide.deinit();
    try wide.forward(&tokens, 6);
    try std.testing.expectEqualSlices(f32, try tuned.logits(6), try wide.logits(6));
    try std.testing.expectEqualSlices(f32, try tuned.logits(7), try wide.logits(7));

    // Test narrower NVFP4 staging and real untraced concurrent dispatches.
    wide.reset();
    try wide.forward(&tokens, 0);
    var concurrent = try Engine.init(gpa, &p, ctx, &store, .{
        .batch = 8,
        .expert_prefill = true,
        .expert_wide_k = true,
        .expert_bf16_tiles = true,
        .bf16_linear_tile = true,
        .bf16_prefill_tile = 8,
        .expert_simd = true,
        .expert_gate_pair = true,
        .fuse_bf16_cast = true,
        .concurrent_scratch = true,
    }, 16);
    defer concurrent.deinit();
    const before = ctx.stats();
    try concurrent.forward(&tokens, 0);
    for (0..tokens.len) |row| {
        try std.testing.expectEqualSlices(f32, try wide.logits(@intCast(row)), try concurrent.logits(@intCast(row)));
    }
    const after = ctx.stats();
    try std.testing.expect(after.dispatches - before.dispatches > after.barriers - before.barriers + 1);

    // The fixture has exact BF16 weights after unpacking and unit outer
    // scales, so the smaller matrix tiles must preserve the wide path here.
    concurrent.knobs.expert_mlx_scale = true;
    concurrent.reset();
    try concurrent.forward(&tokens, 0);
    for (0..tokens.len) |row| {
        try std.testing.expectEqualSlices(f32, try wide.logits(@intCast(row)), try concurrent.logits(@intCast(row)));
    }
}
