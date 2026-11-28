//! Native text graph for the pinned Inkling-2 Mini checkpoint.
//! Vision encoding lives in `inkling_vision.zig` and splices into the post-embed_norm hidden state.
const std = @import("std");
const chat = @import("chat.zig");
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

pub const test_config =
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
pub const LoopCdMode = enum { off, hidden, logits, logits_adaptive };
pub const NoPeMode = enum { off, global, all };

pub const Options = struct {
    loop_passes: ?u32 = null,
    loopcd: LoopCdMode = .off,
    loopcd_omega: f32 = 1.0,
    loopcd_weak_pass: u32 = 1,
    nope: NoPeMode = .off,
};

pub fn parseLoopCdMode(value: []const u8) ?LoopCdMode {
    if (std.mem.eql(u8, value, "off")) return .off;
    if (std.mem.eql(u8, value, "hidden")) return .hidden;
    if (std.mem.eql(u8, value, "logits")) return .logits;
    if (std.mem.eql(u8, value, "logits-adaptive") or std.mem.eql(u8, value, "logits_adaptive")) return .logits_adaptive;
    return null;
}

pub fn parseNoPeMode(value: []const u8) ?NoPeMode {
    if (std.mem.eql(u8, value, "off")) return .off;
    if (std.mem.eql(u8, value, "global")) return .global;
    if (std.mem.eql(u8, value, "all")) return .all;
    return null;
}

pub fn build(gpa: std.mem.Allocator, bytes: []const u8) !ir.Program {
    return buildWithOptions(gpa, bytes, .{});
}

pub fn buildWithOptions(gpa: std.mem.Allocator, bytes: []const u8, options: Options) !ir.Program {
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    var c = try parse(arena.allocator(), bytes);
    if (options.loop_passes) |lp| {
        if (lp == 0 or lp > c.loop_passes) return error.InvalidLoopPasses;
        c.loop_passes = lp;
    }
    if (!std.math.isFinite(options.loopcd_omega) or options.loopcd_omega < 0) return error.InvalidLoopCdOptions;
    if (options.loopcd != .off) {
        if (c.loop_passes < 2) return error.InvalidLoopCdOptions;
        if (options.loopcd_weak_pass == 0 or options.loopcd_weak_pass >= c.loop_passes) return error.InvalidLoopCdOptions;
    } else if (options.loopcd_weak_pass == 0) return error.InvalidLoopCdOptions;
    return lowerWithOptions(gpa, c, options);
}

const Lowering = struct {
    b: *ir.Builder,
    c: Config,
    nope: NoPeMode = .off,

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
    /// Two row vectors side by side, each written where it is read: `out` is one
    /// state and its halves are the producers' outputs, so nothing is copied.
    fn pair(self: Lowering, width: u32) ![2]ir.View {
        const out = try self.scratch("pair", 2 * width);
        return .{ slice(out, 0, width), slice(out, 1, width) };
    }
    fn joined(halves: [2]ir.View) ir.View {
        return .{ .state = halves[0].state, .col = halves[0].col, .width = halves[0].width + halves[1].width };
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
        const nope = self.nope == .all or (self.nope == .global and global);
        const q_dim = c.heads * c.head_dim;
        const kv_dim = c.kv_heads * c.head_dim;
        const q = try self.layerDense(layer, inputs[0], "attn.wq_du.weight", q_dim, .bf16);
        const r = if (nope) null else try self.layerDense(layer, inputs[0], "attn.wr_du.weight", c.heads * c.rel_dim, .bf16);
        const gate = try self.layerDense(layer, inputs[0], "attn.gate_attn.weight", q_dim, .bf16);
        const sink = try self.layerDense(layer, inputs[0], "attn.attn_sink_proj.weight", c.heads, .bf16);
        const k = try self.layerDense(layer, inputs[1], "attn.wk_dv.weight", kv_dim, .bf16);
        const value = try self.layerDense(layer, inputs[2], "attn.wv_dv.weight", kv_dim, .bf16);
        const ck = try self.conv(v, k, "attn.k_sconv", if (global) inputs[1] else null);
        const cv = try self.conv(v, value, "attn.v_sconv", if (global) inputs[2] else null);
        const nv = try self.norm(cv, null, c.head_dim, .bf16);
        try self.clip(q, c.head_dim, c.qk_clip, try self.layerWeight(layer, "attn.q_l2_clip.scale", &.{ c.heads, c.head_dim }));
        try self.clip(ck, c.head_dim, c.qk_clip, null);
        if (r) |rv| try self.clip(rv, c.rel_dim, c.rel_clip, try self.layerWeight(layer, "attn.rel_l2_clip.scale", &.{ c.heads, c.rel_dim }));
        if (global) {
            const tau: ir.RowTransform = .{ .log_position_scale = .{ .floor = c.log_scaling_floor, .alpha = c.log_scaling_alpha } };
            try self.transform(q, q, tau, .bf16);
            if (r) |rv| try self.transform(rv, rv, tau, .bf16);
        }
        const window = if (global) 0 else c.window;
        var bias: ir.View = sink;
        var extent: u32 = 0;
        if (r) |rv| {
            extent = if (global) c.rel_extent else c.window;
            bias = try self.scratch("relative_bias", c.heads * extent);
            const rel = try self.layerWeight(layer, "attn.rel_logits_proj.proj", &.{ c.rel_dim, extent });
            try self.b.op(.{ .grouped_project = .{ .x = rv, .out = bias, .weight = rel, .groups = c.heads, .round = .bf16, .causal_prefix = true } });
        }
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
        // [norm(input), norm(message)] rounded to BF16, then [forget, gain]: each half
        // is normalized, rounded or projected in place in its pair.
        const cat = try self.pair(c.hidden);
        for ([_]ir.View{ input, msg }, cat) |x, half| {
            try self.b.op(.{ .rmsnorm = .{ .x = x, .weight = null, .out = half, .group = c.hidden, .eps = c.eps } });
            _ = try self.rounded(half, .bf16);
        }
        const d = try self.layerDense(v.weight_layer, joined(cat), try self.b.print("controllers.{s}.down_weight", .{stem[stream]}), c.gate_rank, .bf16);
        const gates = try self.pair(c.hidden);
        for ([_][]const u8{ "forget_weight", "gain_weight" }, gates) |name, half| {
            const gate_w = try self.layerWeight(v.weight_layer, try self.b.print("controllers.{s}.{s}", .{ stem[stream], name }), &.{ c.hidden, d.width });
            try self.b.op(.{ .linear = .{ .w = gate_w, .x = d, .out = half, .input_bf16 = true } });
        }
        const out = try self.scratch("message_merge", c.hidden);
        try self.b.op(.{ .inkling_message_merge = .{ .hidden = input, .message = msg, .gates = joined(gates), .out = out } });
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

pub fn lower(gpa: std.mem.Allocator, c: Config) !ir.Program {
    return lowerWithOptions(gpa, c, .{});
}

fn lowerWithOptions(gpa: std.mem.Allocator, c: Config, options: Options) !ir.Program {
    var b = ir.Builder.init(gpa);
    errdefer b.deinit();
    const l: Lowering = .{ .b = &b, .c = c, .nope = options.nope };
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
    var weak_hidden: ?ir.View = null;
    var weak_blocks: ?ir.View = null;
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
        if (v.stage == .loop and v.physical_layer == c.loop_end - 1) {
            if (options.loopcd != .off and v.pass + 1 == options.loopcd_weak_pass) {
                const wh = try l.scratch("loopcd_weak_hidden", h);
                try l.transform(hidden, wh, .copy, .none);
                weak_hidden = wh;
                if (options.loopcd == .logits or options.loopcd == .logits_adaptive) {
                    const wb = try l.scratch("loopcd_weak_blocks", count * h);
                    try l.transform(Lowering.slice(blocks, 0, count * h), wb, .copy, .none);
                    weak_blocks = wb;
                }
            }
            if (options.loopcd == .hidden and v.pass + 1 == c.loop_passes) {
                const contrasted = try l.scratch("loopcd_hidden", h);
                try b.op(.{ .loopcd_contrast = .{
                    .strong = hidden,
                    .weak = weak_hidden.?,
                    .out = contrasted,
                    .omega = options.loopcd_omega,
                    .adaptive = false,
                    .round = .bf16,
                } });
                hidden = contrasted;
            }
        }
    }
    const output_query = try l.weight("attn_res.output_query", &.{h});
    const norm_w = try l.weight("norm.weight", &.{h});
    const unembed_w = try l.weight("unembed.weight", &.{ c.vocab, h });
    const head = b.mark();
    const partial = try l.norm(hidden, null, h, .none);
    const final = try l.depth(Lowering.slice(blocks, 0, count * h), partial, output_query, 0);
    const normed = try l.norm(final, norm_w, h, .bf16);
    try b.setDraftFinalNorm(normed);
    var weak_normed_opt: ?ir.View = null;
    if (options.loopcd == .logits or options.loopcd == .logits_adaptive) {
        const weak_partial = try l.norm(weak_hidden.?, null, h, .none);
        const weak_final = try l.depth(weak_blocks.?, weak_partial, output_query, 0);
        weak_normed_opt = try l.norm(weak_final, norm_w, h, .bf16);
    }
    const raw = try l.dense(normed, unembed_w, c.vocab, .bf16);
    const logits = try l.scratch("usable_logits", c.unpadded_vocab);
    try l.transform(raw, logits, .{ .softcap = c.final_softcap }, .none);
    if (weak_normed_opt) |weak_normed| {
        try b.op(.{ .linear = .{ .w = unembed_w, .x = weak_normed, .out = raw, .input_bf16 = true } });
        _ = try l.rounded(raw, .bf16);
        const weak_logits = try l.scratch("weak_logits", c.unpadded_vocab);
        try l.transform(raw, weak_logits, .{ .softcap = c.final_softcap }, .none);
        try b.op(.{ .loopcd_contrast = .{
            .strong = logits,
            .weak = weak_logits,
            .out = logits,
            .omega = options.loopcd_omega,
            .adaptive = options.loopcd == .logits_adaptive,
            .round = .none,
        } });
    }
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

pub fn writeFixture(a: std.mem.Allocator, p: *const ir.Program, path: []const u8) !void {
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

    // Row liveness: the last visit's attention output and feed-forward run on the
    // rows its convolution caches and the head read. A chunk without logits, a
    // chunk with logits on its last row and a single row must each leave the
    // logits the untrimmed engine computes, reference kernels and tuned ones.
    const prompt = [_]u32{ 1, 4, 11, 2, 3, 6, 9, 1, 5, 7, 2, 12, 8, 3, 10, 6 };
    for ([_]@import("knobs.zig").Knobs{ .{ .batch = 8 }, concurrent.knobs }) |base| {
        var trimmed_knobs = base;
        trimmed_knobs.tail_trim = 1;
        trimmed_knobs.scratch_reuse = true;
        var whole = try Engine.init(gpa, &p, ctx, &store, base, 16);
        defer whole.deinit();
        var trimmed = try Engine.init(gpa, &p, ctx, &store, trimmed_knobs, 16);
        defer trimmed.deinit();
        const start = ctx.stats();
        try whole.forward(prompt[0..8], 8);
        const middle = ctx.stats();
        try trimmed.forward(prompt[0..8], 8);
        const end = ctx.stats();
        // The last visit's final residual add has no reader in a chunk without logits.
        try std.testing.expect(end.dispatches - middle.dispatches < middle.dispatches - start.dispatches);
        try whole.forward(prompt[8..15], 6);
        try trimmed.forward(prompt[8..15], 6);
        try std.testing.expectEqualSlices(f32, try whole.logits(6), try trimmed.logits(6));
        try whole.forward(prompt[15..], 0);
        try trimmed.forward(prompt[15..], 0);
        try std.testing.expectEqualSlices(f32, try whole.logits(0), try trimmed.logits(0));
    }

    // The head tap changes nothing the engine computes and records one block a visit and forward.
    {
        var plain = try Engine.init(gpa, &p, ctx, &store, .{ .batch = 8 }, 16);
        defer plain.deinit();
        var tapped = try Engine.init(gpa, &p, ctx, &store, .{ .batch = 8 }, 16);
        defer tapped.deinit();
        var tap: @import("exec.zig").Tap = .{ .gpa = gpa, .attention = true, .context = 4 };
        defer tap.deinit();
        tapped.tap = &tap;
        try plain.forward(tokens[0..3], 0);
        try tapped.forward(tokens[0..3], 0);
        for (0..3) |row| try std.testing.expectEqualSlices(f32, try plain.logits(@intCast(row)), try tapped.logits(@intCast(row)));
        try plain.forward(tokens[3..4], 0);
        try tapped.forward(tokens[3..4], 0);
        try std.testing.expectEqualSlices(f32, try plain.logits(0), try tapped.logits(0));
        try std.testing.expectEqual(@as(u32, 2), tap.forwards);
        try std.testing.expectEqual(@as(u32, 2), tap.heads);
        try std.testing.expectEqual(@as(usize, 48), tap.windows.items.len);
        var global: usize = 0;
        for (tap.windows.items) |w| {
            try std.testing.expect(w == 0 or w == 4);
            global += @intFromBool(w == 0);
        }
        try std.testing.expect(global > 0 and global < 48);
        try std.testing.expectEqual(@as(usize, 48 * 2 * (3 + 1)), tap.norms.items.len);
        var lit: usize = 0;
        for (tap.norms.items) |x| {
            try std.testing.expect(std.math.isFinite(x) and x >= 0);
            lit += @intFromBool(x > 0);
        }
        try std.testing.expect(lit > tap.norms.items.len / 2);
        try std.testing.expectEqual(@as(usize, 48 * (3 + 1) * 2 * tap.top_k), tap.routes.items.len);
        // Attention weights: one row of `context` per visit and row; each row with its sink sums to
        // one, nothing beyond the row's position or outside a sliding window, and the host's
        // weighted sum of values is the engine's output.
        try std.testing.expectEqual(@as(usize, 48 * (3 + 1) * 4), tap.weights.items.len);
        try std.testing.expectEqual(@as(usize, 48 * (3 + 1)), tap.sinks.items.len);
        try std.testing.expect(tap.worst < 1e-3);
        var block: usize = 0;
        for (0..2) |forward| {
            const rows: usize = if (forward == 0) 3 else 1;
            const base: usize = if (forward == 0) 0 else 48 * 3;
            for (0..48) |visit| for (0..rows) |row| {
                const at = (base + visit * rows + row);
                const position = row + (if (forward == 0) @as(usize, 0) else 3);
                var total: f64 = tap.sinks.items[at];
                for (tap.weights.items[at * 4 ..][0..4], 0..) |w, key| {
                    total += w;
                    if (key > position) try std.testing.expectEqual(@as(f32, 0), w);
                    const window = tap.windows.items[visit];
                    if (window != 0 and key + window <= position) try std.testing.expectEqual(@as(f32, 0), w);
                }
                try std.testing.expectApproxEqAbs(@as(f64, 1), total, 1e-4);
                block += 1;
            };
        }
        try std.testing.expectEqual(@as(usize, 48 * 4), block);
        var at: usize = 0;
        while (at < tap.routes.items.len) : (at += 2 * tap.top_k) {
            for (tap.routes.items[at..][0..tap.top_k]) |id| try std.testing.expect(id >= 0 and id < @as(f32, @floatFromInt(c.experts)));
        }
    }
}

test "inkling plan Lean bindings match geometry and scheduling helpers" {
    const ip = @import("inkling_plan.zig");
    try std.testing.expectEqual(@as(u32, 2304), ip.hidden);
    try std.testing.expectEqual(@as(u32, 48), ip.logical_visits);
    try std.testing.expectEqual(@as(u32, 576), ip.depth_reg_threads);
    try std.testing.expectEqual(@as(u32, 256), ip.splitk_threads);
    try std.testing.expectEqual(@as(u32, 64), ip.expert_block_n64);
    try std.testing.expectEqual(@as(u32, 64), ip.argmax_groups);

    const counts = [_]u32{ 3072, 1024, 1024, 384, 3072, 24 };
    const p1 = counts[0];
    const p2 = p1 + counts[1];
    const p3 = p2 + counts[2];
    const p4 = p3 + counts[3];
    const p5 = p4 + counts[4];
    const prefixes = [_]u32{ 0, p1, p2, p3, p4, p5 };
    for (0..prefixes[5] + counts[5]) |tile_usize| {
        const tile: u32 = @intCast(tile_usize);
        const idx = ip.multiProjIndex(tile, p1, p2, p3, p4, p5);
        const loc = ip.multiProjLocalTile(tile, p1, p2, p3, p4, p5);
        try std.testing.expect(idx < ip.max_multi_proj);
        try std.testing.expect(loc < counts[idx]);
        try std.testing.expectEqual(tile, prefixes[idx] + loc);
    }
    try std.testing.expectEqual(@as(u32, 9), ip.ringBack(2, 3, 10));
    try std.testing.expectEqual(@as(u32, 32), ip.farBandEnd(1060, 1024, 16));
    try std.testing.expectEqual(@as(u32, 0), ip.farBandEnd(1030, 1024, 16));
    try std.testing.expectEqual(@as(u32, 3126), ip.argmaxSpan(200058));
}

test "Inkling fixture d2_gate, decode_argmax and concurrent_scratch match reference bit-for-bit including forced redos and reset" {
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
    var tmp = "/tmp/nano-inkling-d2-XXXXXX".*;
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

    const seq = [_]u32{ 1, 4, 11, 2, 3, 6, 9, 1 };
    const V = c.unpadded_vocab;
    var exp_next1: [5]u32 = undefined;
    var exp_logits1: [5 * 14]f32 = undefined;
    var exp_next2: [12]u32 = undefined;
    var exp_last_logits2: [14]f32 = undefined;
    {
        var ref = try Engine.init(gpa, &p, ctx, &store, .{ .batch = 8 }, 12);
        defer ref.deinit();
        try ref.forward(seq[0..4], 3);
        exp_next1[0] = try ref.next(3);
        @memcpy(exp_logits1[0..V], try ref.logits(3));
        for (seq[4..], 1..) |t, k| {
            try ref.forward(&.{t}, 0);
            exp_next1[k] = try ref.next(0);
            @memcpy(exp_logits1[k * V ..][0..V], try ref.logits(0));
        }
        ref.reset();
        for (0..12) |k| {
            try ref.forward(&.{seq[k % seq.len]}, 0);
            exp_next2[k] = try ref.next(0);
        }
        @memcpy(&exp_last_logits2, try ref.logits(0));
    }

    // Separate sequential runs avoid sharing pending command-buffer queue slots between engines.
    {
        var gated = try Engine.init(gpa, &p, ctx, &store, .{
            .batch = 8,
            .concurrent_scratch = true,
            .scratch_reuse = true,
            .decode_argmax = 1,
            .d2_gate = true,
            .conv_ring = true,
        }, 12);
        defer gated.deinit();
        try gated.forward(seq[0..4], 3);
        try std.testing.expectEqual(exp_next1[0], try gated.next(3));
        try std.testing.expectEqualSlices(f32, exp_logits1[0..V], try gated.logits(3));
        for (seq[4..], 1..) |t, k| {
            try gated.forward(&.{t}, 0);
            try std.testing.expectEqual(exp_next1[k], try gated.next(0));
            try std.testing.expectEqualSlices(f32, exp_logits1[k * V ..][0..V], try gated.logits(0));
        }
        try std.testing.expectEqual(@as(u64, 0), gated.d2_redos);
        gated.reset();
        for (0..12) |k| {
            try gated.forward(&.{seq[k % seq.len]}, 0);
            try std.testing.expectEqual(exp_next2[k], try gated.next(0));
        }
        try std.testing.expectEqualSlices(f32, &exp_last_logits2, try gated.logits(0));
        try gated.d2Settle();
    }

    {
        var redos = try Engine.init(gpa, &p, ctx, &store, .{
            .batch = 8,
            .concurrent_scratch = true,
            .scratch_reuse = true,
            .decode_argmax = 1,
            .d2_gate = true,
            .d2_gate_spins = 1,
            .conv_ring = true,
        }, 12);
        defer redos.deinit();
        try redos.forward(seq[0..4], 3);
        try std.testing.expectEqual(exp_next1[0], try redos.next(3));
        for (seq[4..], 1..) |t, k| {
            try redos.forward(&.{t}, 0);
            try std.testing.expectEqual(exp_next1[k], try redos.next(0));
        }
        redos.reset();
        for (0..12) |k| {
            try redos.forward(&.{seq[k % seq.len]}, 0);
            try std.testing.expectEqual(exp_next2[k], try redos.next(0));
        }
        // At pos = 11 == context - 1, no ahead buffer is queued, so arena logits stay clean.
        try std.testing.expectEqualSlices(f32, &exp_last_logits2, try redos.logits(0));
        try std.testing.expect(redos.d2_redos > 0);
    }
}

test "Inkling fixture Step 3B BF16 multi-proj, controller-merge, head-softcap, wide and splitk" {
    const gpa = std.testing.allocator;
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var c = try parse(a, test_config);
    c.hidden = 256; // divisible by 256 so bf16_decode_wide triggers
    c.intermediate = 128;
    c.heads = 4; // q/r/gate/sink all multiples of 4
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
    var tmp = "/tmp/nano-inkling-3b-XXXXXX".*;
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

    const seq = [_]u32{ 1, 4, 11, 2, 3, 6 };
    const V = c.unpadded_vocab;
    var base_logits: [2 * 14]f32 = undefined;
    var base = try Engine.init(gpa, &p, ctx, &store, .{
        .batch = 8,
        .bf16_linear_tile = true,
        .bf16_decode_vec = true,
        .fuse_bf16_cast = true,
        .scratch_reuse = true,
    }, 16);
    defer base.deinit();
    try base.forward(seq[0..4], 3);
    try base.forward(seq[4..5], 0);
    @memcpy(base_logits[0..V], try base.logits(0));
    try base.forward(seq[5..6], 0);
    @memcpy(base_logits[V .. 2 * V], try base.logits(0));

    // Exact-math fusions (controller_merge, bf16_projections, head_softcap) must be bit-identical to bf16_decode_vec base!
    var fused_k = base.knobs;
    fused_k.fuse_controller_merge = true;
    fused_k.fuse_bf16_projections = true;
    fused_k.fuse_head_softcap = true;
    fused_k.d2_gate = true;
    fused_k.decode_argmax = 1;
    var fused = try Engine.init(gpa, &p, ctx, &store, fused_k, 16);
    defer fused.deinit();
    // Verify expected fusions fired in the plan
    var n_cm: usize = 0;
    var n_mp: usize = 0;
    var n_hs: usize = 0;
    for (fused.steps) |s| switch (s) {
        .controller_merge => n_cm += 1,
        .bf16_projections => n_mp += 1,
        .head_softcap => n_hs += 1,
        else => {},
    };
    try std.testing.expect(n_cm > 0 and n_mp > 0 and n_hs == 1);
    try fused.forward(seq[0..4], 3);
    try fused.forward(seq[4..5], 0);
    try std.testing.expectEqualSlices(f32, base_logits[0..V], try fused.logits(0));
    try fused.forward(seq[5..6], 0);
    try std.testing.expectEqualSlices(f32, base_logits[V .. 2 * V], try fused.logits(0));
    try fused.d2Settle();

    // Reassociating decode variants (bf16_decode_wide + bf16_decode_splitk) must stay within tight float tolerance.
    var re_k = fused_k;
    re_k.bf16_decode_wide = true;
    re_k.bf16_decode_splitk = true;
    var re = try Engine.init(gpa, &p, ctx, &store, re_k, 16);
    defer re.deinit();
    try re.forward(seq[0..4], 3);
    try re.forward(seq[4..5], 0);
    for (base_logits[0..V], try re.logits(0)) |ev, av| try std.testing.expectApproxEqAbs(ev, av, 1e-4);
    try re.forward(seq[5..6], 0);
    for (base_logits[V .. 2 * V], try re.logits(0)) |ev, av| try std.testing.expectApproxEqAbs(ev, av, 1e-4);
    try re.d2Settle();
}

test "Inkling Step 3C exact store_pack_bf16, fuse_inkling_conv, relative_bias_decode and fused depth/norm/clip" {
    const gpa = std.testing.allocator;
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var c = try parse(a, test_config);
    c.hidden = 256;
    c.intermediate = 128;
    c.heads = 4;
    c.kv_heads = 1;
    c.head_dim = 8;
    c.rel_dim = 16; // triggers relative_bias_decode (dim == 16)
    c.rel_extent = 8; // m % 8 == 0
    c.window = 8;
    c.log_scaling_floor = 3;
    c.vocab = 16;
    c.unpadded_vocab = 14;
    c.eos = try a.dupe(u32, &.{13});
    c.gate_rank = 4;
    c.max_context = 16;
    var p = try lower(gpa, c);
    defer p.deinit();
    var tmp = "/tmp/nano-inkling-3c-XXXXXX".*;
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

    const seq = [_]u32{ 1, 4, 11, 2, 3, 6 };
    const V = c.unpadded_vocab;
    var base_logits: [2 * 14]f32 = undefined;
    var base = try Engine.init(gpa, &p, ctx, &store, .{
        .batch = 8,
        .bf16_linear_tile = true,
        .bf16_decode_vec = true,
        .fuse_bf16_cast = true,
        .conv_ring = true,
        .relative_split = true,
        .scratch_reuse = true,
    }, 16);
    defer base.deinit();
    try base.forward(seq[0..4], 3);
    try base.forward(seq[4..5], 0);
    @memcpy(base_logits[0..V], try base.logits(0));
    try base.forward(seq[5..6], 0);
    @memcpy(base_logits[V .. 2 * V], try base.logits(0));

    // Exact-math 3C additions (fuse_inkling_conv, relative_bias_decode, add_norm? Wait add_norm/rmsnorm/depth reassociate slightly).
    // First check pure exact-math additions on top of base: fuse_inkling_conv + relative_bias_decode!
    var exact_k = base.knobs;
    exact_k.fuse_inkling_conv = true;
    exact_k.relative_bias_decode = true;
    var exact_e = try Engine.init(gpa, &p, ctx, &store, exact_k, 16);
    defer exact_e.deinit();
    var n_conv: usize = 0;
    for (exact_e.steps) |s| if (s == .inkling_conv) { n_conv += 1; };
    try std.testing.expect(n_conv > 0);
    try exact_e.forward(seq[0..4], 3);
    try exact_e.forward(seq[4..5], 0);
    try std.testing.expectEqualSlices(f32, base_logits[0..V], try exact_e.logits(0));
    try exact_e.forward(seq[5..6], 0);
    try std.testing.expectEqualSlices(f32, base_logits[V .. 2 * V], try exact_e.logits(0));

    // Next check all Step 3C optimizations together (depth_attention_fast, fuse_depth_norm, rmsnorm_fast,
    // fuse_add_norm, fuse_inkling_conv, relative_bias_decode, d2_gate, decode_argmax).
    var full_k = exact_k;
    full_k.depth_attention_fast = true;
    full_k.fuse_depth_norm = true;
    full_k.rmsnorm_fast = true;
    full_k.fuse_add_norm = true;
    full_k.fuse_controller_merge = true;
    full_k.fuse_bf16_projections = true;
    full_k.fuse_head_softcap = true;
    full_k.d2_gate = true;
    full_k.decode_argmax = 1;
    var full = try Engine.init(gpa, &p, ctx, &store, full_k, 16);
    defer full.deinit();
    var n_cdn: usize = 0;
    var n_dn: usize = 0;
    var n_an: usize = 0;
    for (full.steps) |s| switch (s) {
        .controller_depth_norms => n_cdn += 1,
        .depth_norm => n_dn += 1,
        .add_norm => n_an += 1,
        else => {},
    };
    try std.testing.expect(n_cdn > 0 and n_dn > 0 and n_an > 0);
    try full.forward(seq[0..4], 3);
    try full.forward(seq[4..5], 0);
    for (base_logits[0..V], try full.logits(0)) |ev, av| try std.testing.expectApproxEqAbs(ev, av, 1e-4);
    try full.forward(seq[5..6], 0);
    for (base_logits[V .. 2 * V], try full.logits(0)) |ev, av| try std.testing.expectApproxEqAbs(ev, av, 1e-4);
    try full.d2Settle();
}

test "Inkling Step 3C 2304-wide GPU kernels match reference arithmetic within tolerance" {
    const gpu = @import("gpu.zig");
    const ip = @import("inkling_plan.zig");
    const ctx = try gpu.open();
    defer ctx.deinit();
    var prng = std.Random.DefaultPrng.init(31415);
    const r = prng.random();
    const n: u32 = ip.hidden; // 2304
    const m_bank: u32 = 6;
    const heads = m_bank + 1;
    // Arena layout: bank(6*2304), partial(2304), inp(2304), q_out(2304), cat0(2304), cat1(2304), ref_tmp(2304), ref_out(2304)
    const arena_buf = try ctx.alloc(20 * n);
    defer arena_buf.deinit();
    const qbuf = try ctx.alloc(n);
    defer qbuf.deinit();
    const wbuf = try ctx.alloc(n);
    defer wbuf.deinit();
    const af = arena_buf.floats();
    for (af[0 .. (m_bank + 3) * n]) |*v| v.* = (r.float(f32) - 0.5) * 0.5;
    for (qbuf.floats()) |*v| v.* = (r.float(f32) - 0.5) * 0.5;
    for (wbuf.floats()) |*v| v.* = 1.0 + (r.float(f32) - 0.5) * 0.2;

    const gpa = std.testing.allocator;
    const off_bank: u32 = 0;
    const off_partial: u32 = m_bank * n;
    const off_inp: u32 = (m_bank + 1) * n;
    const off_out1: u32 = (m_bank + 2) * n;
    const off_out2: u32 = (m_bank + 3) * n;
    const off_cat0: u32 = (m_bank + 4) * n;
    const off_ref1: u32 = (m_bank + 8) * n;
    const off_ref2: u32 = (m_bank + 9) * n;
    const off_ref_cat0: u32 = (m_bank + 10) * n;

    // 1. Compare mk_depth_attention_reg vs mk_depth_attention
    var p: gpu.Params = .{ .rows = 1, .x = off_bank, .xs = m_bank * n, .y = off_partial, .ys = n, .z = off_ref1, .zs = n, .n = n, .m = m_bank, .heads = heads, .scale = 1.0 / @sqrt(@as(f32, @floatFromInt(n))) };
    try gpu.dispatch(ctx, "mk_depth_attention", &.{ arena_buf, qbuf }, &p, 256, 256);
    p.z = off_out1;
    try gpu.dispatch(ctx, "mk_depth_attention_reg", &.{ arena_buf, qbuf }, &p, ip.depth_reg_threads, ip.depth_reg_threads);
    try ctx.finish();
    for (af[off_ref1..][0..n], af[off_out1..][0..n]) |ev, av| try std.testing.expectApproxEqAbs(ev, av, 1e-5);

    // 2. Compare mk_depth_norm_2304 vs mk_depth_attention + mk_rmsnorm(weight + round_out bf16)
    var pn: gpu.Params = .{ .rows = 1, .x = off_ref1, .xs = n, .y = off_ref2, .ys = n, .n = n, .heads = 1, .eps = 1e-5, .flags = gpu.flag_has_weight | gpu.flag_round_out };
    try gpu.dispatch(ctx, "mk_rmsnorm", &.{ wbuf, arena_buf }, &pn, ip.depth_reg_threads, ip.depth_reg_threads);
    var pdn = p;
    pdn.z = off_out2;
    pdn.eps = 1e-5;
    pdn.flags = gpu.flag_has_weight | gpu.flag_round_out;
    try gpu.dispatch(ctx, "mk_depth_norm_2304", &.{ arena_buf, qbuf, wbuf }, &pdn, ip.depth_reg_threads, ip.depth_reg_threads);
    try ctx.finish();
    for (af[off_ref2..][0..n], af[off_out2..][0..n]) |ev, av| try std.testing.expectApproxEqAbs(ev, av, 1e-4);

    // 3. Compare mk_controller_depth_norms_2304 vs mk_depth_attention + 2 x mk_rmsnorm(null, round_out bf16)
    var pn1: gpu.Params = .{ .rows = 1, .x = off_inp, .xs = n, .y = off_ref_cat0, .ys = 2 * n, .n = n, .heads = 1, .eps = 1e-5, .flags = gpu.flag_round_out };
    var pn2: gpu.Params = .{ .rows = 1, .x = off_ref1, .xs = n, .y = off_ref_cat0 + n, .ys = 2 * n, .n = n, .heads = 1, .eps = 1e-5, .flags = gpu.flag_round_out };
    try gpu.dispatch(ctx, "mk_rmsnorm", &.{ arena_buf, arena_buf }, &pn1, ip.depth_reg_threads, ip.depth_reg_threads);
    try gpu.dispatch(ctx, "mk_rmsnorm", &.{ arena_buf, arena_buf }, &pn2, ip.depth_reg_threads, ip.depth_reg_threads);
    var pcdn: gpu.Params = .{
        .rows = 1,
        .x = off_bank,
        .xs = m_bank * n,
        .y = off_partial,
        .ys = n,
        .z = off_out2,
        .zs = n,
        .every = off_inp,
        .pad_k = n,
        .rot = off_cat0,
        .kind = 2 * n,
        .n = n,
        .m = m_bank,
        .heads = heads,
        .eps = 1e-5,
        .scale = p.scale,
        .flags = gpu.flag_round_out,
    };
    try gpu.dispatch(ctx, "mk_controller_depth_norms_2304", &.{ arena_buf, qbuf }, &pcdn, ip.depth_reg_threads, ip.depth_reg_threads);
    try ctx.finish();
    _ = gpa;
    for (af[off_ref1..][0..n], af[off_out2..][0..n]) |ev, av| try std.testing.expectApproxEqAbs(ev, av, 1e-5);
    for (af[off_ref_cat0..][0 .. 2 * n], af[off_cat0..][0 .. 2 * n]) |ev, av| try std.testing.expectApproxEqAbs(ev, av, 1e-4);

    // 4. Compare mk_store_pack_bf16 vs mk_store + mk_pack_bf16
    const cap: u32 = 20;
    const f_ref = try ctx.alloc(cap * n);
    defer f_ref.deinit();
    const f_new = try ctx.alloc(cap * n);
    defer f_new.deinit();
    const h_ref = try ctx.alloc((cap + 16) * n);
    defer h_ref.deinit();
    const h_new = try ctx.alloc((cap + 16) * n);
    defer h_new.deinit();
    @memset(f_ref.floats(), 0);
    @memset(f_new.floats(), 0);
    @memset(h_ref.floats(), 0);
    @memset(h_new.floats(), 0);
    var ps: gpu.Params = .{ .rows = 1, .pos = 19, .x = off_inp, .xs = n, .n = n, .capacity = cap };
    try gpu.dispatch(ctx, "mk_store", &.{ arena_buf, f_ref }, &ps, n, 256);
    try gpu.dispatch(ctx, "mk_pack_bf16", &.{ arena_buf, h_ref }, &ps, n, 256);
    ps.pos = 21; // wraps to slot 1 < 16 mirror pad too
    var ps2 = ps;
    try gpu.dispatch(ctx, "mk_store", &.{ arena_buf, f_ref }, &ps2, n, 256);
    try gpu.dispatch(ctx, "mk_pack_bf16", &.{ arena_buf, h_ref }, &ps2, n, 256);
    ps.pos = 19;
    try gpu.dispatch(ctx, "mk_store_pack_bf16", &.{ arena_buf, f_new, h_new }, &ps, n, 256);
    try gpu.dispatch(ctx, "mk_store_pack_bf16", &.{ arena_buf, f_new, h_new }, &ps2, n, 256);
    try ctx.finish();
    try std.testing.expectEqualSlices(u32, std.mem.bytesAsSlice(u32, std.mem.sliceAsBytes(f_ref.floats())), std.mem.bytesAsSlice(u32, std.mem.sliceAsBytes(f_new.floats())));
    try std.testing.expectEqualSlices(u32, std.mem.bytesAsSlice(u32, std.mem.sliceAsBytes(h_ref.floats())), std.mem.bytesAsSlice(u32, std.mem.sliceAsBytes(h_new.floats())));
}

test "Inkling Step 3D NVFP4 sub-SIMD and MXFP8 vectorized decode experts match dot8 within tolerance" {
    const gpa = std.testing.allocator;
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var c = try parse(a, test_config);
    c.hidden = 256; // divisible by 256 so mk_expert_gate_up_nv16 & mx_vec trigger
    c.intermediate = 128; // divisible by 128 so mk_expert_down_nv8 & mx_vec trigger
    c.heads = 4;
    c.kv_heads = 1;
    c.head_dim = 8;
    c.rel_dim = 16;
    c.rel_extent = 8;
    c.window = 8;
    c.log_scaling_floor = 3;
    c.vocab = 16;
    c.unpadded_vocab = 14;
    c.eos = try a.dupe(u32, &.{13});
    c.gate_rank = 4;
    c.max_context = 16;
    var p = try lower(gpa, c);
    defer p.deinit();
    var tmp = "/tmp/nano-inkling-3d-XXXXXX".*;
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

    const seq = [_]u32{ 1, 4, 11, 2, 3, 6 };
    const V = c.unpadded_vocab;
    var ref_logits: [2 * 14]f32 = undefined;
    var base = try Engine.init(gpa, &p, ctx, &store, .{
        .batch = 8,
        .expert_simd = true,
        .expert_wide_dot = true,
        .expert_gate_pair = true,
        .bf16_linear_tile = true,
        .bf16_decode_vec = true,
        .fuse_bf16_cast = true,
        .scratch_reuse = true,
    }, 16);
    defer base.deinit();
    try base.forward(seq[0..4], 3);
    try base.forward(seq[4..5], 0);
    @memcpy(ref_logits[0..V], try base.logits(0));
    try base.forward(seq[5..6], 0);
    @memcpy(ref_logits[V .. 2 * V], try base.logits(0));

    var fast_k = base.knobs;
    fast_k.expert_nvfp4_fast = true;
    var fast = try Engine.init(gpa, &p, ctx, &store, fast_k, 16);
    defer fast.deinit();
    try fast.forward(seq[0..4], 3);
    try fast.forward(seq[4..5], 0);
    for (ref_logits[0..V], try fast.logits(0)) |ev, av| try std.testing.expectApproxEqAbs(ev, av, 1e-4);
    try fast.forward(seq[5..6], 0);
    for (ref_logits[V .. 2 * V], try fast.logits(0)) |ev, av| try std.testing.expectApproxEqAbs(ev, av, 1e-4);
}

test "Inkling Step 3E relative_long_fast and fuse_relative_gate match reference on global and wrapped ring layers" {
    const gpa = std.testing.allocator;
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    const gpu = @import("gpu.zig");
    const ctx = try gpu.open();
    defer ctx.deinit();

    const heads: u32 = 6;
    const kv_heads: u32 = 2;
    const dim: u32 = 128;
    const extent: u32 = 16;
    const kv_w: u32 = kv_heads * dim;
    const q_w: u32 = heads * dim;

    var prng = std.Random.DefaultPrng.init(31);
    const rnd = prng.random();
    const bf16f = struct {
        fn cast(v: f32) f32 {
            const bits: u32 = @bitCast(v);
            return @bitCast((bits + 0x7fff + ((bits >> 16) & 1)) & 0xffff0000);
        }
    }.cast;

    for ([_]struct { keys: u32, window: u32, pos: u32 }{
        .{ .keys = 40, .window = 0, .pos = 39 }, // global linear slot
        .{ .keys = 24, .window = 16, .pos = 23 }, // wrapped sliding window (capacity=20)
    }) |cfg| {
        const cap: u32 = if (cfg.window != 0) 20 else cfg.keys;
        const n_arena = q_w + heads * extent + heads + q_w + 3 * q_w;
        const mem = try ctx.alloc(n_arena);
        defer mem.deinit();
        const f = mem.floats();
        const q_off: u32 = 0;
        const b_off: u32 = q_w;
        const s_off: u32 = b_off + heads * extent;
        const g_off: u32 = s_off + heads;
        const ref_out: u32 = g_off + q_w;
        const fast_out: u32 = ref_out + q_w;
        const gated_out: u32 = fast_out + q_w;
        for (f[q_off..][0..q_w]) |*x| x.* = bf16f(0.25 * rnd.floatNorm(f32));
        for (f[b_off..][0 .. heads * extent]) |*x| x.* = bf16f(0.2 * rnd.floatNorm(f32));
        for (f[s_off..][0..heads]) |*x| x.* = bf16f(0.2 * rnd.floatNorm(f32));
        for (f[g_off..][0..q_w]) |*x| x.* = bf16f(0.5 * rnd.floatNorm(f32));

        const k32 = try ctx.alloc(@as(usize, cap) * kv_w);
        defer k32.deinit();
        const v32 = try ctx.alloc(@as(usize, cap) * kv_w);
        defer v32.deinit();
        const k16 = try ctx.alloc((@as(usize, cap + 16) * kv_w + 1) / 2);
        defer k16.deinit();
        const v16 = try ctx.alloc((@as(usize, cap + 16) * kv_w + 1) / 2);
        defer v16.deinit();
        for ([_][2]@TypeOf(k32){ .{ k32, k16 }, .{ v32, v16 } }) |pair| {
            const half = std.mem.bytesAsSlice(u16, std.mem.sliceAsBytes(pair[1].floats()));
            for (pair[0].floats(), half[0 .. @as(usize, cap) * kv_w]) |*dst, *hdst| {
                dst.* = bf16f(0.25 * rnd.floatNorm(f32));
                hdst.* = @intCast(@as(u32, @bitCast(dst.*)) >> 16);
            }
            @memcpy(half[@as(usize, cap) * kv_w ..][0 .. 16 * kv_w], half[0 .. 16 * kv_w]);
        }
        const parts = try ctx.alloc(@as(usize, heads) * 8 * 130);
        defer parts.deinit();

        var p: gpu.Params = .{
            .rows = 1,
            .pos = cfg.pos,
            .x = q_off,
            .xs = q_w,
            .y = ref_out,
            .ys = q_w,
            .z = b_off,
            .zs = heads * extent,
            .s = s_off,
            .every = heads,
            .heads = heads,
            .kv_heads = kv_heads,
            .dim = dim,
            .m = extent,
            .window = cfg.window,
            .capacity = cap,
            .taps = 5,
            .scale = 1 / @sqrt(@as(f32, dim)),
        };
        try gpu.dispatch(ctx, "mk_relative_attention", &.{ mem, k32, v32 }, &p, heads * 32, 32);

        var p_fast = p;
        p_fast.y = fast_out;
        try gpu.dispatch(ctx, "mk_relative_long_keys_fast", &.{ mem, k16, v16, parts }, &p_fast, @as(usize, kv_heads) * 5 * 256, 256);
        try gpu.dispatch(ctx, "mk_relative_long_merge_fast", &.{ mem, parts }, &p_fast, @as(usize, heads) * 128, 128);

        var p_gated = p_fast;
        p_gated.y = gated_out;
        p_gated.z = g_off;
        p_gated.zs = q_w;
        p_gated.flags = gpu.flag_round_bf16 | gpu.flag_round_out;
        try gpu.dispatch(ctx, "mk_relative_long_merge_gated", &.{ mem, parts }, &p_gated, @as(usize, heads) * 128, 128);
        try ctx.finish();

        for (0..q_w) |idx| {
            const rv = f[ref_out + idx];
            const fv = f[fast_out + idx];
            try std.testing.expectApproxEqAbs(rv, fv, 1e-4);
            const expected_gated = bf16f(bf16f(fv) * (1.0 / (1.0 + @exp(-f[g_off + idx]))));
            try std.testing.expectApproxEqAbs(expected_gated, f[gated_out + idx], 1e-4);
        }
    }

    // End-to-end Inkling fixture check (tests head=8 fallback to standard path & planner matching on real geometry).
    var c = try parse(a, test_config);
    c.hidden = 128;
    c.intermediate = 128;
    c.heads = 6;
    c.kv_heads = 2;
    c.head_dim = 128;
    c.rel_dim = 16;
    c.rel_extent = 16;
    c.window = 16;
    c.log_scaling_floor = 3;
    c.vocab = 16;
    c.unpadded_vocab = 14;
    c.eos = try a.dupe(u32, &.{13});
    c.gate_rank = 4;
    c.max_context = 32;
    var prog = try lower(gpa, c);
    defer prog.deinit();
    var tmp = "/tmp/nano-inkling-3e-XXXXXX".*;
    const dir = std.mem.span(mkdtemp(&tmp) orelse return error.TempDirectoryFailed);
    defer _ = rmdir(dir);
    const path = try std.fmt.allocPrintSentinel(a, "{s}/model.safetensors", .{dir}, 0);
    defer _ = unlink(path);
    try writeFixture(a, &prog, path);
    var store = try @import("weights").Store.init(a, ctx, dir, 0);
    defer store.deinit();
    const Engine = @import("exec.zig").Engine;

    var base = try Engine.init(gpa, &prog, ctx, &store, .{
        .batch = 8,
        .relative_stripes = true,
        .scratch_reuse = true,
    }, 32);
    defer base.deinit();
    var fast_unfused_k = base.knobs;
    fast_unfused_k.relative_long_fast = true;
    var fast_unfused = try Engine.init(gpa, &prog, ctx, &store, fast_unfused_k, 32);
    defer fast_unfused.deinit();
    var fast_k = base.knobs;
    fast_k.relative_long_fast = true;
    fast_k.fuse_relative_gate = true;
    var fast = try Engine.init(gpa, &prog, ctx, &store, fast_k, 32);
    defer fast.deinit();

    const tokens = [_]u32{ 1, 4, 11, 2, 3, 6, 9, 1 };
    try base.forward(tokens[0..6], 5);
    try fast_unfused.forward(tokens[0..6], 5);
    try fast.forward(tokens[0..6], 5);
    try std.testing.expectEqualSlices(f32, try base.logits(5), try fast_unfused.logits(5));
    try std.testing.expectEqualSlices(f32, try base.logits(5), try fast.logits(5));
    for (tokens[6..]) |t| {
        try base.forward(&.{t}, 0);
        try fast_unfused.forward(&.{t}, 0);
        try fast.forward(&.{t}, 0);
        // Fusing relative_gate must match relative_long_fast bit-for-bit (both round in-place BF16).
        try std.testing.expectEqualSlices(f32, try fast_unfused.logits(0), try fast.logits(0));
        // Over 48 recurrent visits, fast::exp2 (~3e-7) can flip 1 BF16 rounding tie (2^-7 = 0.0078).
        for (try base.logits(0), try fast.logits(0)) |ev, av| try std.testing.expectApproxEqAbs(ev, av, 0.01);
    }
}

test "Inkling Step 2 relative_prefill_fast matches mk_relative_attention across global far-band, wrapped sliding window, and MK_BIAS16" {
    const gpu = @import("gpu.zig");
    const ctx = try gpu.open();
    defer ctx.deinit();

    var prng = std.Random.DefaultPrng.init(2086);
    const rnd = prng.random();
    const bf16f = struct {
        fn cast(v: f32) f32 {
            const bits: u32 = @bitCast(v);
            return @bitCast((bits + 0x7fff + ((bits >> 16) & 1)) & 0xffff0000);
        }
    }.cast;

    const heads: u32 = 6;
    const kv_heads: u32 = 2;
    const dim: u32 = 128;
    const rel_ext: u32 = 64;
    const Case = struct { pos: u32, rows: u32, window: u32, capacity: u32, bias16: bool };
    const cases = [_]Case{
        .{ .pos = 0, .rows = 530, .window = 0, .capacity = 560, .bias16 = false }, // global pos=0 + far-band + tail rows
        .{ .pos = 256, .rows = 70, .window = 0, .capacity = 400, .bias16 = false }, // global pos>0 (far_end > 0 even at q0=0)
        .{ .pos = 48, .rows = 40, .window = 64, .capacity = 64, .bias16 = false }, // wrapped sliding window + mirror pad
        .{ .pos = 160, .rows = 96, .window = 0, .capacity = 300, .bias16 = true }, // MK_BIAS16 packed bias
    };

    for (cases) |c| {
        const kv_total = c.pos + c.rows;
        const k32 = try ctx.alloc(c.capacity * kv_heads * dim);
        defer k32.deinit();
        const v32 = try ctx.alloc(c.capacity * kv_heads * dim);
        defer v32.deinit();
        const k16 = try ctx.alloc((c.capacity + 16) * kv_heads * dim / 2);
        defer k16.deinit();
        const v16 = try ctx.alloc((c.capacity + 16) * kv_heads * dim / 2);
        defer v16.deinit();
        @memset(k32.floats(), 0);
        @memset(v32.floats(), 0);
        @memset(k16.floats(), 0);
        @memset(v16.floats(), 0);

        const k16u = std.mem.bytesAsSlice(u16, std.mem.sliceAsBytes(k16.floats()));
        const v16u = std.mem.bytesAsSlice(u16, std.mem.sliceAsBytes(v16.floats()));
        for (0..@min(kv_total, c.capacity)) |idx| {
            // Fill slots chronologically around the ring so both history and current rows exist.
            const slot: usize = idx;
            for (0..kv_heads * dim) |d| {
                const kv = bf16f(0.25 * rnd.floatNorm(f32));
                const vv = bf16f(0.25 * rnd.floatNorm(f32));
                k32.floats()[slot * kv_heads * dim + d] = kv;
                v32.floats()[slot * kv_heads * dim + d] = vv;
                const kh: u16 = @intCast(@as(u32, @bitCast(kv)) >> 16);
                const vh: u16 = @intCast(@as(u32, @bitCast(vv)) >> 16);
                k16u[slot * kv_heads * dim + d] = kh;
                v16u[slot * kv_heads * dim + d] = vh;
                if (slot < 16) {
                    k16u[(c.capacity + slot) * kv_heads * dim + d] = kh;
                    v16u[(c.capacity + slot) * kv_heads * dim + d] = vh;
                }
            }
        }
        // When wrapped window writes past capacity (c.pos + c.rows > c.capacity), fill wrapped prefix slots as latest positions.
        if (kv_total > c.capacity) {
            for (c.capacity..kv_total) |abs_p| {
                const slot = abs_p % c.capacity;
                for (0..kv_heads * dim) |d| {
                    const kv = bf16f(0.25 * rnd.floatNorm(f32));
                    const vv = bf16f(0.25 * rnd.floatNorm(f32));
                    k32.floats()[slot * kv_heads * dim + d] = kv;
                    v32.floats()[slot * kv_heads * dim + d] = vv;
                    const kh: u16 = @intCast(@as(u32, @bitCast(kv)) >> 16);
                    const vh: u16 = @intCast(@as(u32, @bitCast(vv)) >> 16);
                    k16u[slot * kv_heads * dim + d] = kh;
                    v16u[slot * kv_heads * dim + d] = vh;
                    if (slot < 16) {
                        k16u[(c.capacity + slot) * kv_heads * dim + d] = kh;
                        v16u[(c.capacity + slot) * kv_heads * dim + d] = vh;
                    }
                }
            }
        }

        const q_words = c.rows * heads * dim;
        const bias_words = c.rows * heads * rel_ext;
        const sink_words = c.rows * heads;
        const arena_buf = try ctx.alloc(q_words + bias_words + bias_words + sink_words + q_words * 2);
        defer arena_buf.deinit();
        const af = arena_buf.floats();
        const q_off: u32 = 0;
        const bias_off: u32 = q_words;
        const bias16_off: u32 = bias_off + bias_words;
        const sink_off: u32 = bias16_off + bias_words;
        const ref_out_off: u32 = sink_off + sink_words;
        const fast_out_off: u32 = ref_out_off + q_words;

        for (af[q_off .. q_off + q_words]) |*x| x.* = bf16f(0.3 * rnd.floatNorm(f32));
        for (0..c.rows) |r| {
            const row_u16 = std.mem.bytesAsSlice(u16, std.mem.sliceAsBytes(af[bias16_off + r * heads * rel_ext ..][0 .. heads * rel_ext]));
            for (0 .. heads * rel_ext) |i| {
                const bv = bf16f(0.15 * rnd.floatNorm(f32));
                af[bias_off + r * heads * rel_ext + i] = bv;
                row_u16[i] = @intCast(@as(u32, @bitCast(bv)) >> 16);
            }
        }
        for (af[sink_off .. sink_off + sink_words]) |*x| x.* = bf16f(rnd.floatNorm(f32));

        var p_ref: gpu.Params = .{
            .rows = c.rows,
            .row0 = 0,
            .pos = c.pos,
            .x = q_off,
            .xs = heads * dim,
            .y = ref_out_off,
            .ys = heads * dim,
            .z = bias_off,
            .zs = heads * rel_ext,
            .s = sink_off,
            .every = heads,
            .heads = heads,
            .kv_heads = kv_heads,
            .dim = dim,
            .m = rel_ext,
            .window = c.window,
            .scale = 1.0 / @sqrt(@as(f32, dim)),
            .capacity = c.capacity,
        };
        try gpu.dispatch(ctx, "mk_relative_attention", &.{ arena_buf, k32, v32 }, &p_ref, @as(usize, c.rows) * heads * 32, 32);

        var p_fast = p_ref;
        p_fast.y = fast_out_off;
        if (c.bias16) {
            p_fast.z = bias16_off;
            p_fast.flags = gpu.flag_bias16;
        }
        try gpu.dispatch(ctx, "mk_relative_prefill_fast", &.{ arena_buf, k16, v16 }, &p_fast, @as(usize, (c.rows + 31) / 32) * heads * 128, 128);
        try ctx.finish();

        for (af[ref_out_off..][0..q_words], af[fast_out_off..][0..q_words]) |ev, av| {
            try std.testing.expectApproxEqAbs(ev, av, 5e-4);
        }
    }
}

test "Inkling Step 1 expert_block_m64n64 + mk_pack_rows_bf16 match mk_expert_block bit-for-bit across NVFP4/MXFP8, up/down, and M=8/16/32/64 tails" {
    const gpu = @import("gpu.zig");
    const ctx = try gpu.open();
    defer ctx.deinit();

    var prng = std.Random.DefaultPrng.init(6464);
    const rnd = prng.random();
    const bf16f = struct {
        fn cast(v: f32) f32 {
            const bits: u32 = @bitCast(v);
            return @bitCast((bits + 0x7fff + ((bits >> 16) & 1)) & 0xffff0000);
        }
    }.cast;

    const experts: u32 = 5;
    const rows: u32 = 256; // exercises parts = 4 striping on m64n64
    const top_k: u32 = 2;
    const k_dim: u32 = 128;
    const n_dim: u32 = 128;

    const lists_buf = try ctx.alloc(experts * (rows * top_k + 1));
    defer lists_buf.deinit();
    const list_words = lists_buf.structs(u32);
    @memset(list_words, 0);
    // e0=69 (M=8 tail), e1=77 (M=16 tail), e2=87 (M=32 tail), e3=105 (M=64 tail), e4=174 (2*64+46 -> striped parts=4 + M=64 tail)
    const counts = [_]u32{ 69, 77, 87, 105, 174 };
    var route_idx: u32 = 0;
    for (counts, 0..) |cnt, e| {
        const base = e * (rows * top_k + 1);
        list_words[base] = cnt;
        for (0..cnt) |j| {
            list_words[base + 1 + j] = route_idx;
            route_idx += 1;
        }
    }

    const arena_buf = try ctx.alloc(rows * top_k * k_dim + rows * top_k * n_dim * 2);
    defer arena_buf.deinit();
    const af = arena_buf.floats();
    const x_off: u32 = 0;
    const y_ref_off: u32 = rows * top_k * k_dim;
    const y_new_off: u32 = y_ref_off + rows * top_k * n_dim;
    for (af[0..y_ref_off]) |*v| v.* = bf16f(0.4 * rnd.floatNorm(f32));

    const partial_ref = try ctx.alloc(@max(rows * top_k * n_dim, rows * k_dim / 2));
    defer partial_ref.deinit();
    const partial_new = try ctx.alloc(@max(rows * top_k * n_dim, rows * k_dim / 2));
    defer partial_new.deinit();

    const nv_w = try ctx.alloc(experts * 2 * n_dim * k_dim / 8);
    defer nv_w.deinit();
    for (nv_w.structs(u32)) |*v| v.* = rnd.int(u32);
    const nv_s = try ctx.alloc(experts * 2 * n_dim * k_dim / 64);
    defer nv_s.deinit();
    for (std.mem.sliceAsBytes(nv_s.floats())) |*b| b.* = @intCast(rnd.intRangeAtMost(u8, 0x28, 0x48));
    const amax_buf = try ctx.alloc(experts);
    defer amax_buf.deinit();
    for (amax_buf.floats()) |*v| v.* = 0.5 + rnd.float(f32);
    const denom_buf = try ctx.alloc(1);
    defer denom_buf.deinit();
    denom_buf.floats()[0] = 6.0;

    const mx_w = try ctx.alloc(experts * 2 * n_dim * k_dim / 4);
    defer mx_w.deinit();
    for (std.mem.sliceAsBytes(mx_w.floats())) |*b| {
        const exp: u8 = @intCast(rnd.intRangeAtMost(u8, 4, 9));
        const mant: u8 = @intCast(rnd.intRangeAtMost(u8, 0, 7));
        const sign: u8 = @as(u8, @intCast(rnd.intRangeAtMost(u8, 0, 1))) << 7;
        b.* = sign | (exp << 3) | mant;
    }
    const mx_s = try ctx.alloc(experts * 2 * n_dim * k_dim / 128);
    defer mx_s.deinit();
    for (std.mem.sliceAsBytes(mx_s.floats())) |*b| b.* = @intCast(rnd.intRangeAtMost(u8, 124, 129));

    for ([_]bool{ true, false }) |is_nv| {
        for ([_]bool{ false, true }) |is_down| {
            @memset(af[y_ref_off .. y_new_off + rows * top_k * n_dim], 0);
            @memset(partial_ref.floats(), 0);
            @memset(partial_new.floats(), 0);
            const proj_rows: u32 = if (is_down) n_dim else 2 * n_dim;
            var p: gpu.ExpertParams = .{
                .rows = rows,
                .row0 = 0,
                .x = x_off,
                .xs = top_k * k_dim,
                .y = y_ref_off,
                .ys = top_k * n_dim,
                .n = n_dim,
                .k = k_dim,
                .experts = experts,
                .top_k = top_k,
                .nvfp4 = @intFromBool(is_nv),
                .round_bf16 = 1,
                .gate_cap = 16.0,
                .up_cap = 16.0,
            };
            const wbuf = if (is_nv) nv_w else mx_w;
            const sbuf = if (is_nv) nv_s else mx_s;
            const ref_k: [:0]const u8 = if (is_nv)
                (if (is_down) "mk_expert_block_down_nv" else "mk_expert_block_up_nv")
            else
                (if (is_down) "mk_expert_block_down_mx" else "mk_expert_block_up_mx");
            const new_k: [:0]const u8 = if (is_nv)
                (if (is_down) "mk_expert_block_down_nv64_m64" else "mk_expert_block_up_nv64_m64")
            else
                (if (is_down) "mk_expert_block_down_mx64_m64" else "mk_expert_block_up_mx64_m64");

            try gpu.dispatch(ctx, ref_k, &.{ arena_buf, wbuf, sbuf, amax_buf, denom_buf, lists_buf, partial_ref }, &p, @as(usize, experts) * (proj_rows / 32) * 128 * 4, 128);
            p.y = y_new_off;
            if (!is_down) {
                try gpu.dispatch(ctx, "mk_pack_rows_bf16", &.{ arena_buf, partial_new }, &p, @as(usize, rows) * (k_dim / 4), 256);
            }
            try gpu.dispatch(ctx, new_k, &.{ arena_buf, wbuf, sbuf, amax_buf, denom_buf, lists_buf, partial_new }, &p, @as(usize, experts) * (proj_rows / 64) * 128 * 4, 128);
            try ctx.finish();

            if (is_down) {
                const len = rows * top_k * n_dim;
                try std.testing.expectEqualSlices(u32, std.mem.bytesAsSlice(u32, std.mem.sliceAsBytes(partial_ref.floats()[0..len])), std.mem.bytesAsSlice(u32, std.mem.sliceAsBytes(partial_new.floats()[0..len])));
            } else {
                const len = rows * top_k * n_dim;
                try std.testing.expectEqualSlices(u32, std.mem.bytesAsSlice(u32, std.mem.sliceAsBytes(af[y_ref_off..][0..len])), std.mem.bytesAsSlice(u32, std.mem.sliceAsBytes(af[y_new_off..][0..len])));
            }
        }
    }
}

test "Inkling Qwegn Wave 1 expert_fp_lut, expert_compact_tiles and expert_bf16_io match reference" {
    const gpa = std.testing.allocator;
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var c = try parse(a, test_config);
    c.hidden = 256;
    c.intermediate = 128;
    c.heads = 4;
    c.kv_heads = 1;
    c.head_dim = 8;
    c.rel_dim = 16;
    c.rel_extent = 8;
    c.window = 8;
    c.log_scaling_floor = 3;
    c.vocab = 16;
    c.unpadded_vocab = 14;
    c.eos = try a.dupe(u32, &.{13});
    c.gate_rank = 4;
    c.max_context = 32;
    var p = try lower(gpa, c);
    defer p.deinit();
    var tmp = "/tmp/nano-inkling-wave1-XXXXXX".*;
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

    const seq = [_]u32{ 1, 4, 11, 2, 3, 6, 8, 9, 5, 7, 10, 12, 1, 2, 3, 4, 5, 6 };
    const V = c.unpadded_vocab;
    var ref_logits: [2 * 14]f32 = undefined;
    var base = try Engine.init(gpa, &p, ctx, &store, .{
        .batch = 16,
        .expert_prefill = true,
        .expert_mlx_scale = true,
        .expert_lists_split = true,
        .expert_block_m64n64 = true,
        .expert_nvfp4_fast = true,
        .expert_simd = true,
        .expert_wide_dot = true,
        .expert_gate_pair = true,
        .bf16_linear_tile = true,
        .bf16_decode_vec = true,
        .fuse_bf16_cast = true,
        .scratch_reuse = true,
    }, 32);
    defer base.deinit();
    try base.forward(seq[0..16], 15);
    try base.forward(seq[16..17], 0);
    @memcpy(ref_logits[0..V], try base.logits(0));
    try base.forward(seq[17..18], 0);
    @memcpy(ref_logits[V .. 2 * V], try base.logits(0));

    for ([_]struct { lut: bool, ct: bool, bf16: bool }{
        .{ .lut = true, .ct = false, .bf16 = false },
        .{ .lut = false, .ct = true, .bf16 = false },
        .{ .lut = false, .ct = false, .bf16 = true },
        .{ .lut = true, .ct = true, .bf16 = true },
    }) |cfg| {
        var k = base.knobs;
        k.expert_fp_lut = cfg.lut;
        k.expert_compact_tiles = cfg.ct;
        k.expert_bf16_io = cfg.bf16;
        var eng = try Engine.init(gpa, &p, ctx, &store, k, 32);
        defer eng.deinit();
        try eng.forward(seq[0..16], 15);
        try eng.forward(seq[16..17], 0);
        for (ref_logits[0..V], try eng.logits(0)) |ev, av| try std.testing.expectApproxEqAbs(ev, av, 1e-4);
        try eng.forward(seq[17..18], 0);
        for (ref_logits[V .. 2 * V], try eng.logits(0)) |ev, av| try std.testing.expectApproxEqAbs(ev, av, 1e-4);
    }
}

test "Inkling Qwegn Wave 1 standalone compact tiles + LUT + BF16 combine match ref bit-for-bit across full/tail tiles" {
    const gpu = @import("gpu.zig");
    const inkling_plan = @import("inkling_plan.zig");
    const ctx = try gpu.open();
    defer ctx.deinit();

    const rows: u32 = 130;
    const experts: u32 = 16;
    const top_k: u32 = 8;
    const k_dim: u32 = 128;
    const n_dim: u32 = 128;
    const pairs = rows * top_k;
    var prng = std.Random.DefaultPrng.init(99);
    const rnd = prng.random();
    const bf16f = struct {
        fn cast(v: f32) f32 {
            const bits: u32 = @bitCast(v);
            return @bitCast((bits + 0x7fff + ((bits >> 16) & 1)) & 0xffff0000);
        }
    }.cast;

    const x_words: u32 = rows * top_k * k_dim;
    const routes_words: u32 = rows * top_k * 2;
    const y_ref_off: u32 = x_words + routes_words;
    const y_new_off: u32 = y_ref_off + rows * top_k * n_dim;
    const y_new_up_off: u32 = y_new_off + rows * n_dim;
    const y_new_dn_off: u32 = y_new_up_off + rows * top_k * n_dim;
    const arena_buf = try ctx.alloc(y_new_dn_off + rows * n_dim);
    defer arena_buf.deinit();
    const af = arena_buf.floats();
    const routes_off: u32 = x_words;
    for (af[0..x_words]) |*v| v.* = bf16f(0.25 * rnd.floatNorm(f32));
    for (0..rows) |r| {
        for (0..top_k) |s_i| {
            // Mix empty/tiny/medium/multi-64-tile experts across routes.
            const exp_id: u32 = if (s_i == 0) 0 else @intCast((r * 3 + s_i) % (experts - 1));
            af[routes_off + r * top_k * 2 + s_i] = @floatFromInt(exp_id);
            af[routes_off + r * top_k * 2 + top_k + s_i] = bf16f(0.125 * @as(f32, @floatFromInt(s_i + 1)));
        }
    }

    const list_words = @max(@as(usize, experts) * (pairs + 33), @as(usize, experts) + 2 + pairs + 2 * inkling_plan.moeMaxTiles(experts, pairs));
    const lists_buf = try ctx.alloc(list_words);
    defer lists_buf.deinit();
    const compact_buf = try ctx.alloc(list_words);
    defer compact_buf.deinit();
    const partial_ref = try ctx.alloc(@as(usize, pairs) * n_dim);
    defer partial_ref.deinit();
    const partial_new = try ctx.alloc(@as(usize, pairs) * n_dim);
    defer partial_new.deinit();

    const nv_w = try ctx.alloc(experts * 2 * n_dim * k_dim / 8);
    defer nv_w.deinit();
    for (nv_w.structs(u32)) |*v| v.* = rnd.int(u32);
    const nv_s = try ctx.alloc(experts * 2 * n_dim * k_dim / 64);
    defer nv_s.deinit();
    for (std.mem.sliceAsBytes(nv_s.floats())) |*b| b.* = @intCast(rnd.intRangeAtMost(u8, 0x28, 0x48));
    const amax_buf = try ctx.alloc(experts);
    defer amax_buf.deinit();
    for (amax_buf.floats()) |*v| v.* = 0.5 + rnd.float(f32);
    const denom_buf = try ctx.alloc(1);
    defer denom_buf.deinit();
    denom_buf.floats()[0] = 6.0;

    const mx_w = try ctx.alloc(experts * 2 * n_dim * k_dim / 4);
    defer mx_w.deinit();
    for (std.mem.sliceAsBytes(mx_w.floats())) |*b| {
        const exp: u8 = @intCast(rnd.intRangeAtMost(u8, 4, 9));
        const mant: u8 = @intCast(rnd.intRangeAtMost(u8, 0, 7));
        const sign: u8 = @as(u8, @intCast(rnd.intRangeAtMost(u8, 0, 1))) << 7;
        b.* = sign | (exp << 3) | mant;
    }
    const mx_s = try ctx.alloc(experts * 2 * n_dim * k_dim / 128);
    defer mx_s.deinit();
    for (std.mem.sliceAsBytes(mx_s.floats())) |*b| b.* = @intCast(rnd.intRangeAtMost(u8, 124, 129));

    for ([_]bool{ true, false }) |is_nv| {
        const wbuf = if (is_nv) nv_w else mx_w;
        const sbuf = if (is_nv) nv_s else mx_s;
        @memset(af[y_ref_off .. y_new_off + rows * top_k * n_dim], 0);
        @memset(partial_ref.floats(), 0);
        @memset(partial_new.floats(), 0);

        // Reference: non-compact up -> down -> combine_mm
        var p_up: gpu.ExpertParams = .{
            .rows = rows, .row0 = 0, .x = 0, .xs = top_k * k_dim,
            .routes = routes_off, .rs = top_k * 2, .y = y_ref_off, .ys = top_k * n_dim,
            .n = n_dim, .k = k_dim, .experts = experts, .top_k = top_k,
            .nvfp4 = @intFromBool(is_nv), .round_bf16 = 1, .gate_cap = 16.0, .up_cap = 16.0,
        };
        try gpu.dispatch(ctx, "mk_expert_row_lists_simd", &.{ arena_buf, lists_buf }, &p_up, @as(usize, experts) * 32, 128);
        try gpu.dispatch(ctx, "mk_pack_rows_bf16", &.{ arena_buf, partial_ref }, &p_up, @as(usize, rows) * (k_dim / 4), 256);
        try gpu.dispatch(ctx, if (is_nv) "mk_expert_block_up_nv64_m64" else "mk_expert_block_up_mx64_m64", &.{ arena_buf, wbuf, sbuf, amax_buf, denom_buf, lists_buf, partial_ref }, &p_up, @as(usize, experts) * (2 * n_dim / 64) * 128 * 2, 128);

        var p_dn = p_up;
        p_dn.x = y_ref_off;
        p_dn.xs = top_k * n_dim;
        p_dn.y = y_new_off; // temporary ref combined output slot; down output is n_dim wide per row
        p_dn.ys = n_dim;
        try gpu.dispatch(ctx, if (is_nv) "mk_expert_block_down_nv64_m64" else "mk_expert_block_down_mx64_m64", &.{ arena_buf, wbuf, sbuf, amax_buf, denom_buf, lists_buf, partial_ref }, &p_dn, @as(usize, experts) * (n_dim / 64) * 128 * 2, 128);
        try gpu.dispatch(ctx, "mk_expert_combine_mm", &.{ arena_buf, partial_ref }, &p_dn, @as(usize, rows) * n_dim, 256);

        // New: compact tiles + LUT + BF16 I/O up -> down -> combine_mm_bf16
        const max_tiles = inkling_plan.moeMaxTiles(experts, pairs);
        p_up.y = y_new_up_off;
        try gpu.dispatch(ctx, "mk_expert_group_tiles", &.{ arena_buf, compact_buf }, &p_up, 256, 256);
        try gpu.dispatch(ctx, "mk_pack_rows_bf16", &.{ arena_buf, partial_new }, &p_up, @as(usize, rows) * (k_dim / 4), 256);
        try gpu.dispatch(ctx, if (is_nv) "mk_expert_block_up_nv64_m64_ct_lut_bf16" else "mk_expert_block_up_mx64_m64_ct_lut_bf16", &.{ arena_buf, wbuf, sbuf, amax_buf, denom_buf, compact_buf, partial_new }, &p_up, @as(usize, max_tiles) * (2 * n_dim / 64) * 128, 128);

        p_dn.x = y_new_up_off;
        p_dn.y = y_new_dn_off;
        try gpu.dispatch(ctx, if (is_nv) "mk_expert_block_down_nv64_m64_ct_lut_bf16" else "mk_expert_block_down_mx64_m64_ct_lut_bf16", &.{ arena_buf, wbuf, sbuf, amax_buf, denom_buf, compact_buf, partial_new }, &p_dn, @as(usize, max_tiles) * (n_dim / 64) * 128, 128);
        try gpu.dispatch(ctx, "mk_expert_combine_mm_bf16", &.{ arena_buf, partial_new }, &p_dn, @as(usize, rows) * (n_dim / 4), 256);
        try ctx.finish();

        const out_len = rows * n_dim;
        try std.testing.expectEqualSlices(u32, std.mem.bytesAsSlice(u32, std.mem.sliceAsBytes(af[y_new_off..][0..out_len])), std.mem.bytesAsSlice(u32, std.mem.sliceAsBytes(af[y_new_dn_off..][0..out_len])));
    }
}

test "Inkling Qwegn Wave 2 fuse_qkvr_prep matches reference on sliding and global layers for prefill and decode" {
    const gpa = std.testing.allocator;
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var c = try parse(a, test_config);
    c.hidden = 128;
    c.intermediate = 64;
    c.heads = 3;
    c.kv_heads = 1;
    c.head_dim = 128;
    c.rel_dim = 16;
    c.rel_extent = 16;
    c.window = 16;
    c.log_scaling_floor = 3;
    c.vocab = 16;
    c.unpadded_vocab = 14;
    c.eos = try a.dupe(u32, &.{13});
    c.gate_rank = 4;
    c.max_context = 32;
    var p = try lower(gpa, c);
    defer p.deinit();
    var tmp = "/tmp/nano-inkling-wave2-XXXXXX".*;
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

    const seq = [_]u32{ 1, 4, 11, 2, 3, 6, 8, 9, 5, 7, 10, 12, 1, 2, 3, 4, 5, 6 };
    const V = c.unpadded_vocab;
    var ref_logits: [2 * 14]f32 = undefined;
    var base = try Engine.init(gpa, &p, ctx, &store, .{
        .batch = 16,
        .relative_stripes = true,
        .relative_tile_prefill = true,
        .relative_bias16 = true,
        .relative_bias_d8 = true,
        .relative_bias_decode = true,
        .relative_long_fast = true,
        .relative_prefill_fast = true,
        .store_pack_bf16 = true,
        .conv_ring = true,
        .fuse_inkling_conv = true,
        .fuse_bf16_cast = true,
        .rmsnorm_fast = true,
        .scratch_reuse = true,
        .concurrent_scratch = true,
    }, 32);
    defer base.deinit();
    try base.forward(seq[0..16], 15);
    try base.forward(seq[16..17], 0);
    @memcpy(ref_logits[0..V], try base.logits(0));
    try base.forward(seq[17..18], 0);
    @memcpy(ref_logits[V .. 2 * V], try base.logits(0));

    var qkvr_k = base.knobs;
    qkvr_k.fuse_qkvr_prep = true;
    var qkvr = try Engine.init(gpa, &p, ctx, &store, qkvr_k, 32);
    defer qkvr.deinit();
    var sconv_steps: usize = 0;
    var dyn_steps: usize = 0;
    for (qkvr.steps) |st| switch (st) {
        .qkvr_sconv_prep => sconv_steps += 1,
        .qkvr_dyn_prep => dyn_steps += 1,
        else => {},
    };
    try std.testing.expect(sconv_steps > 0 and dyn_steps > 0);
    try qkvr.forward(seq[0..16], 15);
    try qkvr.forward(seq[16..17], 0);
    for (ref_logits[0..V], try qkvr.logits(0)) |ev, av| try std.testing.expectApproxEqAbs(ev, av, 1e-5);
    try qkvr.forward(seq[17..18], 0);
    for (ref_logits[V .. 2 * V], try qkvr.logits(0)) |ev, av| try std.testing.expectApproxEqAbs(ev, av, 1e-5);
}

test "Inkling Qwegn Wave 3 relative_inline_bias matches reference across d2_gate" {
    const gpa = std.testing.allocator;
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var c = try parse(a, test_config);
    c.hidden = 128;
    c.intermediate = 64;
    c.heads = 3;
    c.kv_heads = 1;
    c.head_dim = 128;
    c.rel_dim = 16;
    c.rel_extent = 16;
    c.window = 16;
    c.log_scaling_floor = 3;
    c.vocab = 16;
    c.unpadded_vocab = 14;
    c.eos = try a.dupe(u32, &.{13});
    c.gate_rank = 4;
    c.max_context = 512;
    var p = try lower(gpa, c);
    defer p.deinit();
    var tmp = "/tmp/nano-inkling-wave3-XXXXXX".*;
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

    var seq: [264]u32 = undefined;
    for (&seq, 0..) |*tok, idx| tok.* = @intCast((idx * 7 + 1) % 14);
    const V = c.unpadded_vocab;
    var ref_logits: [2 * 14]f32 = undefined;
    var base = try Engine.init(gpa, &p, ctx, &store, .{
        .batch = 260,
        .relative_stripes = true,
        .relative_tile_prefill = true,
        .relative_bias16 = true,
        .relative_bias_d8 = true,
        .relative_bias_decode = true,
        .relative_long_fast = true,
        .relative_prefill_fast = true,
        .fuse_relative_gate = true,
        .store_pack_bf16 = true,
        .conv_ring = true,
        .fuse_inkling_conv = true,
        .fuse_bf16_cast = true,
        .rmsnorm_fast = true,
        .scratch_reuse = true,
        .concurrent_scratch = true,
        .d2_gate = true,
    }, 512);
    defer base.deinit();
    try base.forward(seq[0..260], 259);
    try base.forward(seq[260..261], 0);
    @memcpy(ref_logits[0..V], try base.logits(0));
    try base.forward(seq[261..262], 0);
    @memcpy(ref_logits[V .. 2 * V], try base.logits(0));

    for ([_]bool{ false, true }) |fuse_qkvr| {
        var k = base.knobs;
        k.relative_inline_bias = true;
        k.fuse_qkvr_prep = fuse_qkvr;
        var eng = try Engine.init(gpa, &p, ctx, &store, k, 512);
        defer eng.deinit();
        var ib_cnt: usize = 0;
        for (eng.inline_bias_ok.?) |ok| ib_cnt += @intFromBool(ok);
        try std.testing.expect(ib_cnt > 0);
        try eng.forward(seq[0..260], 259);
        try eng.forward(seq[260..261], 0);
        for (ref_logits[0..V], try eng.logits(0)) |ev, av| try std.testing.expectApproxEqAbs(ev, av, 1e-4);
        try eng.forward(seq[261..262], 0);
        for (ref_logits[V .. 2 * V], try eng.logits(0)) |ev, av| try std.testing.expectApproxEqAbs(ev, av, 1e-4);
    }
}

test "Inkling Qwegn Wave 4 fuse_merge_norm matches reference" {
    const gpa = std.testing.allocator;
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var c = try parse(a, test_config);
    c.hidden = 128;
    c.intermediate = 64;
    c.heads = 4;
    c.kv_heads = 1;
    c.head_dim = 8;
    c.rel_dim = 16;
    c.rel_extent = 8;
    c.window = 8;
    c.log_scaling_floor = 3;
    c.vocab = 16;
    c.unpadded_vocab = 14;
    c.eos = try a.dupe(u32, &.{13});
    c.gate_rank = 4;
    c.max_context = 32;
    var p = try lower(gpa, c);
    defer p.deinit();
    var tmp = "/tmp/nano-inkling-wave4-XXXXXX".*;
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

    const seq = [_]u32{ 1, 4, 11, 2, 3, 6, 8, 9, 5, 7, 10, 12, 1, 2, 3, 4, 5, 6 };
    const V = c.unpadded_vocab;
    var ref_logits: [2 * 14]f32 = undefined;
    var base = try Engine.init(gpa, &p, ctx, &store, .{
        .batch = 16,
        .bf16_linear_tile = true,
        .bf16_prefill_tile = 8,
        .bf16_prefill_matrix = true,
        .bf16_matrix_small = true,
        .bf16_decode_vec = true,
        .fuse_bf16_cast = true,
        .fuse_controller_merge = true,
        .fuse_depth_norm = true,
        .rmsnorm_fast = true,
        .scratch_reuse = true,
        .concurrent_scratch = true,
    }, 32);
    defer base.deinit();
    try base.forward(seq[0..16], 15);
    try base.forward(seq[16..17], 0);
    @memcpy(ref_logits[0..V], try base.logits(0));
    try base.forward(seq[17..18], 0);
    @memcpy(ref_logits[V .. 2 * V], try base.logits(0));

    var w4_knobs = base.knobs;
    w4_knobs.fuse_merge_norm = true;
    var w4 = try Engine.init(gpa, &p, ctx, &store, w4_knobs, 32);
    defer w4.deinit();
    var mn_steps: usize = 0;
    for (w4.steps) |st| switch (st) {
        .controller_merge_norm => mn_steps += 1,
        else => {},
    };
    try std.testing.expect(mn_steps > 0);
    try w4.forward(seq[0..16], 15);
    try w4.forward(seq[16..17], 0);
    for (ref_logits[0..V], try w4.logits(0)) |ev, av| try std.testing.expectApproxEqAbs(ev, av, 1e-5);
    try w4.forward(seq[17..18], 0);
    for (ref_logits[V .. 2 * V], try w4.logits(0)) |ev, av| try std.testing.expectApproxEqAbs(ev, av, 1e-5);
}

test "Inkling2 loop passes and LoopCD lowering validation" {
    const frontend = @import("frontend.zig");
    const gpa = std.testing.allocator;
    var p1 = try buildWithOptions(gpa, test_config, .{ .loop_passes = 1 });
    defer p1.deinit();
    try std.testing.expectEqual(@as(usize, 25), p1.traces.len);

    var p2 = try buildWithOptions(gpa, test_config, .{ .loop_passes = 2 });
    defer p2.deinit();
    try std.testing.expectEqual(@as(usize, 37), p2.traces.len);

    var ph = try buildWithOptions(gpa, test_config, .{ .loop_passes = 2, .loopcd = .hidden, .loopcd_omega = 0.5, .loopcd_weak_pass = 1 });
    defer ph.deinit();
    var hidden_cd: usize = 0;
    for (ph.ops, 0..) |op, i| if (op == .loopcd_contrast) {
        try std.testing.expect(i < ph.head_start);
        try std.testing.expect(!op.loopcd_contrast.adaptive);
        hidden_cd += 1;
    };
    try std.testing.expectEqual(@as(usize, 1), hidden_cd);

    var pl = try buildWithOptions(gpa, test_config, .{ .loopcd = .logits_adaptive, .loopcd_omega = 1.0, .loopcd_weak_pass = 2 });
    defer pl.deinit();
    var logit_cd: usize = 0;
    for (pl.ops, 0..) |op, i| if (op == .loopcd_contrast) {
        try std.testing.expect(i >= pl.head_start);
        try std.testing.expect(op.loopcd_contrast.adaptive);
        logit_cd += 1;
    };
    try std.testing.expectEqual(@as(usize, 1), logit_cd);

    try std.testing.expectError(error.InvalidLoopPasses, buildWithOptions(gpa, test_config, .{ .loop_passes = 0 }));
    try std.testing.expectError(error.InvalidLoopPasses, buildWithOptions(gpa, test_config, .{ .loop_passes = 4 }));
    try std.testing.expectError(error.InvalidLoopCdOptions, buildWithOptions(gpa, test_config, .{ .loop_passes = 1, .loopcd = .logits }));
    try std.testing.expectError(error.InvalidLoopCdOptions, buildWithOptions(gpa, test_config, .{ .loop_passes = 2, .loopcd = .logits, .loopcd_weak_pass = 2 }));
    try std.testing.expectError(error.InvalidLoopCdOptions, buildWithOptions(gpa, test_config, .{ .loopcd = .hidden, .loopcd_weak_pass = 0 }));
    try std.testing.expectError(error.InvalidLoopCdOptions, buildWithOptions(gpa, test_config, .{ .loopcd = .hidden, .loopcd_omega = -0.1 }));

    try std.testing.expectError(error.UnsupportedLoopOptions, frontend.build(gpa, "lfm3", "{}", .{ .context = undefined, .has = undefined }, .{ .loop_passes = 2 }));
    try std.testing.expectError(error.UnsupportedLoopOptions, frontend.build(gpa, "lfm2", "{}", .{ .context = undefined, .has = undefined }, .{ .loopcd = .logits }));
}

test "Inkling2 LoopCD hidden, logits, adaptive, omega=0 parity, and tuned knobs" {
    const gpa = std.testing.allocator;
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var c = try parse(a, test_config);
    c.hidden = 128;
    c.intermediate = 64;
    c.heads = 4;
    c.kv_heads = 1;
    c.head_dim = 8;
    c.rel_dim = 16;
    c.rel_extent = 8;
    c.window = 8;
    c.log_scaling_floor = 3;
    c.vocab = 16;
    c.unpadded_vocab = 14;
    c.eos = try a.dupe(u32, &.{13});
    c.gate_rank = 4;
    c.max_context = 32;

    var base_p = try lower(gpa, c);
    defer base_p.deinit();
    var tmp = "/tmp/nano-inkling-loopcd-XXXXXX".*;
    const dir = std.mem.span(mkdtemp(&tmp) orelse return error.TempDirectoryFailed);
    defer _ = rmdir(dir);
    const path = try std.fmt.allocPrintSentinel(a, "{s}/model.safetensors", .{dir}, 0);
    defer _ = unlink(path);
    try writeFixture(a, &base_p, path);
    const ctx = try @import("gpu.zig").open();
    defer ctx.deinit();
    var store = try @import("weights").Store.init(a, ctx, dir, 0);
    defer store.deinit();
    const Engine = @import("exec.zig").Engine;

    const seq = [_]u32{ 1, 4, 11, 2, 3, 6, 8, 9 };
    const V = c.unpadded_vocab;
    var base_eng = try Engine.init(gpa, &base_p, ctx, &store, .{ .batch = 8 }, 32);
    defer base_eng.deinit();
    try base_eng.forward(seq[0..6], 5);
    try base_eng.forward(seq[6..7], 0);
    const base_logits = try a.dupe(f32, try base_eng.logits(0));

    // Check omega=0 bit-exactness across all 3 LoopCD modes and non-zero contrast + batched==sequential + tuned knobs.
    for ([_]LoopCdMode{ .hidden, .logits, .logits_adaptive }) |mode| {
        var zero_p = try lowerWithOptions(gpa, c, .{ .loopcd = mode, .loopcd_omega = 0.0, .loopcd_weak_pass = 1 });
        defer zero_p.deinit();
        var zero_eng = try Engine.init(gpa, &zero_p, ctx, &store, .{ .batch = 8 }, 32);
        defer zero_eng.deinit();
        try zero_eng.forward(seq[0..6], 5);
        try zero_eng.forward(seq[6..7], 0);
        try std.testing.expectEqualSlices(f32, base_logits, try zero_eng.logits(0));

        var act_p = try lowerWithOptions(gpa, c, .{ .loopcd = mode, .loopcd_omega = 0.8, .loopcd_weak_pass = 1 });
        defer act_p.deinit();
        var ref_eng = try Engine.init(gpa, &act_p, ctx, &store, .{ .batch = 8 }, 32);
        defer ref_eng.deinit();
        try ref_eng.forward(seq[0..6], 5);
        try ref_eng.forward(seq[6..7], 0);
        const ref_out = try a.dupe(f32, try ref_eng.logits(0));
        var diff: f32 = 0;
        for (base_logits, ref_out) |bv, rv| diff = @max(diff, @abs(bv - rv));
        try std.testing.expect(diff > 1e-4);

        var seq_eng = try Engine.init(gpa, &act_p, ctx, &store, .{ .batch = 8 }, 32);
        defer seq_eng.deinit();
        for (seq[0..7]) |tok| try seq_eng.forward(&.{tok}, 0);
        try std.testing.expectEqualSlices(f32, ref_out, try seq_eng.logits(0));

        try seq_eng.forward(seq[7..8], 0);
        const seq_out = try a.dupe(f32, try seq_eng.logits(0));

        {
            var tuned_eng = try Engine.init(gpa, &act_p, ctx, &store, .{
                .batch = 8,
                .bf16_linear_tile = true,
                .bf16_prefill_tile = 8,
                .bf16_matrix_small = true,
                .bf16_decode_vec = true,
                .fuse_bf16_cast = true,
                .fuse_bf16_projections = true,
                .fuse_controller_merge = true,
                .fuse_merge_norm = true,
                .fuse_depth_norm = true,
                .fuse_head_softcap = true,
                .rmsnorm_fast = true,
                .scratch_reuse = true,
                .concurrent_scratch = true,
                .tail_trim = 4,
                .d2_gate = true,
            }, 32);
            defer tuned_eng.deinit();
            try tuned_eng.forward(seq[0..6], 5);
            try tuned_eng.forward(seq[6..7], 0);
            try tuned_eng.forward(seq[7..8], 0);
            for (seq_out[0..V], try tuned_eng.logits(0)) |ev, av| {
                try std.testing.expect(std.math.isFinite(av));
                try std.testing.expectApproxEqAbs(ev, av, 2e-2);
            }
        }
    }

    // Check loop_passes=2 + LoopCD.
    var c2 = c;
    c2.loop_passes = 2;
    var p2_cd = try lowerWithOptions(gpa, c2, .{ .loopcd = .logits_adaptive, .loopcd_omega = 0.6, .loopcd_weak_pass = 1 });
    defer p2_cd.deinit();
    var e2 = try Engine.init(gpa, &p2_cd, ctx, &store, .{ .batch = 8 }, 32);
    defer e2.deinit();
    try e2.forward(seq[0..7], 6);
    for (try e2.logits(6)) |v| try std.testing.expect(std.math.isFinite(v));
}

test "Inkling2 LSA dense bypass, sparse selection, tuned knobs, and combined LoopCD" {
    const gpa = std.testing.allocator;
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var c = try parse(a, test_config);
    c.hidden = 128;
    c.intermediate = 64;
    c.heads = 3;
    c.kv_heads = 1;
    c.head_dim = 128;
    c.rel_dim = 16;
    c.rel_extent = 16;
    c.window = 16;
    c.log_scaling_floor = 3;
    c.vocab = 16;
    c.unpadded_vocab = 14;
    c.eos = try a.dupe(u32, &.{13});
    c.gate_rank = 4;
    c.max_context = 64;

    var p = try lowerWithOptions(gpa, c, .{});
    defer p.deinit();
    var tmp = "/tmp/nano-inkling-lsa-XXXXXX".*;
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

    var seq: [44]u32 = undefined;
    for (&seq, 0..) |*t, i| t.* = @intCast((i * 5 + 1) % 13 + 1);

    // 1. Dense bypass: total_blocks <= 1 + 2 + 3 = 6 blocks (< 28 tokens): bit-identical at seq[0..20].
    var ref = try Engine.init(gpa, &p, ctx, &store, .{ .batch = 32 }, 64);
    defer ref.deinit();
    var lsa_ref = try Engine.init(gpa, &p, ctx, &store, .{
        .batch = 32,
        .lsa = true,
        .lsa_budget = 8,
        .lsa_window = 12,
        .lsa_coarse_threshold = 0,
    }, 64);
    defer lsa_ref.deinit();
    try ref.forward(seq[0..20], 19);
    try lsa_ref.forward(seq[0..20], 19);
    try std.testing.expectEqualSlices(f32, try ref.logits(19), try lsa_ref.logits(19));

    // 2. Past dense bypass (32..43 tokens):
    //    (a) FP32 ref LSA (`row_step=1`) vs striped fast BF16 decode + inline bias + gated merge + qkvr (`relative_tile_prefill=false`, same `row_step=1`),
    //    (b) tiled BF16 MMA prefill baseline (`relative_tile_prefill=true`, `row_step=32`) vs full tuned Qwegn+LSA stack (`tail_trim` + `d2_gate`).
    try lsa_ref.forward(seq[20..40], 19);
    try lsa_ref.forward(seq[40..41], 0);
    try lsa_ref.forward(seq[41..42], 0);
    const lsa_ref_out = try a.dupe(f32, try lsa_ref.logits(0));
    for (lsa_ref_out) |v| try std.testing.expect(std.math.isFinite(v));

    {
        var lsa_fast_base = try Engine.init(gpa, &p, ctx, &store, .{
            .batch = 32,
            .relative_stripes = true,
            .relative_long_fast = true,
            .store_pack_bf16 = true,
            .conv_ring = true,
            .fuse_inkling_conv = true,
            .fuse_bf16_cast = true,
            .rmsnorm_fast = true,
            .scratch_reuse = true,
            .lsa = true,
            .lsa_budget = 8,
            .lsa_window = 12,
            .lsa_coarse_threshold = 0,
        }, 64);
        defer lsa_fast_base.deinit();
        try lsa_fast_base.forward(seq[0..20], 19);
        try lsa_fast_base.forward(seq[20..40], 19);
        try lsa_fast_base.forward(seq[40..41], 0);
        try lsa_fast_base.forward(seq[41..42], 0);
        const fast_base_out = try a.dupe(f32, try lsa_fast_base.logits(0));
        for (lsa_ref_out, fast_base_out) |rv, fv| {
            try std.testing.expect(std.math.isFinite(fv));
            try std.testing.expectApproxEqAbs(rv, fv, 5e-2);
        }

        var lsa_fast_r1 = try Engine.init(gpa, &p, ctx, &store, .{
            .batch = 32,
            .relative_stripes = true,
            .relative_long_fast = true,
            .relative_inline_bias = true,
            .fuse_relative_gate = true,
            .fuse_qkvr_prep = true,
            .store_pack_bf16 = true,
            .conv_ring = true,
            .fuse_inkling_conv = true,
            .fuse_bf16_cast = true,
            .rmsnorm_fast = true,
            .scratch_reuse = true,
            .lsa = true,
            .lsa_budget = 8,
            .lsa_window = 12,
            .lsa_coarse_threshold = 0,
        }, 64);
        defer lsa_fast_r1.deinit();
        try lsa_fast_r1.forward(seq[0..20], 19);
        try lsa_fast_r1.forward(seq[20..40], 19);
        try lsa_fast_r1.forward(seq[40..41], 0);
        try lsa_fast_r1.forward(seq[41..42], 0);
        for (fast_base_out, try lsa_fast_r1.logits(0)) |bv, fv| {
            try std.testing.expect(std.math.isFinite(fv));
            try std.testing.expectApproxEqAbs(bv, fv, 1e-4);
        }
    }

    var prefill_base_out: [14]f32 = undefined;
    {
        var lsa_prefill_base = try Engine.init(gpa, &p, ctx, &store, .{
            .batch = 32,
            .relative_stripes = true,
            .relative_tile_prefill = true,
            .relative_bias16 = true,
            .relative_bias_d8 = true,
            .relative_bias_decode = true,
            .relative_long_fast = true,
            .relative_prefill_fast = true,
            .store_pack_bf16 = true,
            .conv_ring = true,
            .fuse_inkling_conv = true,
            .fuse_bf16_cast = true,
            .rmsnorm_fast = true,
            .scratch_reuse = true,
            .lsa = true,
            .lsa_budget = 8,
            .lsa_window = 12,
            .lsa_coarse_threshold = 0,
        }, 64);
        defer lsa_prefill_base.deinit();
        try lsa_prefill_base.forward(seq[0..20], 19);
        try lsa_prefill_base.forward(seq[20..40], 19);
        try lsa_prefill_base.forward(seq[40..41], 0);
        try lsa_prefill_base.forward(seq[41..42], 0);
        @memcpy(&prefill_base_out, try lsa_prefill_base.logits(0));
    }

    {
        var lsa_tuned = try Engine.init(gpa, &p, ctx, &store, .{
            .batch = 32,
            .relative_stripes = true,
            .relative_tile_prefill = true,
            .relative_bias16 = true,
            .relative_bias_d8 = true,
            .relative_bias_decode = true,
            .relative_long_fast = true,
            .relative_prefill_fast = true,
            .relative_inline_bias = true,
            .fuse_relative_gate = true,
            .fuse_qkvr_prep = true,
            .store_pack_bf16 = true,
            .conv_ring = true,
            .fuse_inkling_conv = true,
            .fuse_bf16_cast = true,
            .rmsnorm_fast = true,
            .scratch_reuse = true,
            .concurrent_scratch = true,
            .tail_trim = 8,
            .d2_gate = true,
            .lsa = true,
            .lsa_budget = 8,
            .lsa_window = 12,
            .lsa_coarse_threshold = 0,
        }, 64);
        defer lsa_tuned.deinit();
        try lsa_tuned.forward(seq[0..20], 19);
        try lsa_tuned.forward(seq[20..40], 19); // exercises mk_relative_lsa_prefill (rows >= 8) past bypass
        try lsa_tuned.forward(seq[40..41], 0); // exercises mk_relative_lsa_decode_keys_inline_fast + gated merge
        try lsa_tuned.forward(seq[41..42], 0);
        for (&prefill_base_out, try lsa_tuned.logits(0)) |bv, tv| {
            try std.testing.expect(std.math.isFinite(tv));
            try std.testing.expectApproxEqAbs(bv, tv, 1e-3);
        }
    }

    // 3. Combined LSA + LoopCD adaptive + reduced passes.
    var p_combo = try lowerWithOptions(gpa, c, .{ .loop_passes = 2, .loopcd = .logits_adaptive, .loopcd_omega = 0.7, .loopcd_weak_pass = 1 });
    defer p_combo.deinit();
    var eng_combo = try Engine.init(gpa, &p_combo, ctx, &store, .{
        .batch = 32,
        .relative_stripes = true,
        .relative_long_fast = true,
        .relative_tile_prefill = true,
        .fuse_qkvr_prep = true,
        .fuse_relative_gate = true,
        .scratch_reuse = true,
        .lsa = true,
        .lsa_budget = 8,
        .lsa_window = 12,
        .lsa_coarse_threshold = 0,
    }, 64);
    defer eng_combo.deinit();
    try eng_combo.forward(seq[0..20], 20);
    try eng_combo.forward(seq[20..40], 19);
    try eng_combo.forward(seq[40..41], 0);
    for (try eng_combo.logits(0)) |v| try std.testing.expect(std.math.isFinite(v));
}

test "LoopCD logits_adaptive packs at batch=4096 and fuses head_softcap_loopcd" {
    const plan = @import("plan.zig");
    const exec = @import("exec.zig");
    const gpa = std.testing.allocator;
    var pl = try buildWithOptions(gpa, test_config, .{ .loopcd = .logits_adaptive, .loopcd_omega = 1.0, .loopcd_weak_pass = 1 });
    defer pl.deinit();

    const offsets = try gpa.alloc(u32, pl.states.len);
    defer gpa.free(offsets);
    const total = try exec.packScratch(gpa, &pl, 4096, 4200, offsets);
    try std.testing.expect(total <= std.math.maxInt(u32));

    const info = try gpa.alloc(plan.WeightInfo, pl.weights.len);
    defer gpa.free(info);
    @memset(info, .{ .q4 = false, .buffer = 1, .dense = true });
    const steps = try plan.build(gpa, &pl, .{ .head_softcap = true }, info);
    defer gpa.free(steps);
    var head_softcaps: usize = 0;
    var head_softcap_loopcd: usize = 0;
    for (steps) |s| switch (s) {
        .head_softcap => head_softcaps += 1,
        .head_softcap_loopcd => head_softcap_loopcd += 1,
        else => {},
    };
    try std.testing.expectEqual(@as(usize, 0), head_softcaps);
    try std.testing.expectEqual(@as(usize, 1), head_softcap_loopcd);
}

test "Inkling2 NoPE global and all lowering, reference vs tuned parity, and LSA" {
    const frontend = @import("frontend.zig");
    const plan = @import("plan.zig");
    const Engine = @import("exec.zig").Engine;
    const gpa = std.testing.allocator;
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();

    // Non-Inkling models fail closed when --nope != off.
    try std.testing.expectError(error.UnsupportedLoopOptions, frontend.build(gpa, "lfm2", "{}", .{ .context = undefined, .has = undefined }, .{ .nope = .global }));

    var c = try parse(a, test_config);
    c.hidden = 128;
    c.intermediate = 64;
    c.heads = 3;
    c.kv_heads = 1;
    c.head_dim = 128;
    c.rel_dim = 16;
    c.rel_extent = 16;
    c.window = 16;
    c.log_scaling_floor = 3;
    c.vocab = 16;
    c.unpadded_vocab = 14;
    c.eos = try a.dupe(u32, &.{13});
    c.gate_rank = 4;
    c.max_context = 64;

    var p_off = try lowerWithOptions(gpa, c, .{ .nope = .off });
    defer p_off.deinit();
    var p_global = try lowerWithOptions(gpa, c, .{ .nope = .global });
    defer p_global.deinit();
    var p_all = try lowerWithOptions(gpa, c, .{ .nope = .all });
    defer p_all.deinit();
    try std.testing.expect(p_global.ops.len < p_off.ops.len);
    try std.testing.expect(p_all.ops.len < p_global.ops.len);
    try std.testing.expect(p_global.states.len < p_off.states.len);

    // Planner uses qkv_sconv_prep / qkv_dyn_prep on NoPE layers.
    const info_all = try a.alloc(plan.WeightInfo, p_all.weights.len);
    @memset(info_all, .{ .q4 = false, .buffer = 1, .dense = true });
    const steps_all = try plan.build(a, &p_all, .{ .qkvr_prep = true }, info_all);
    var sconv_no_r: usize = 0;
    var dyn_no_r: usize = 0;
    for (steps_all) |st| switch (st) {
        .qkv_sconv_prep => sconv_no_r += 1,
        .qkv_dyn_prep => dyn_no_r += 1,
        else => {},
    };
    try std.testing.expect(sconv_no_r > 0 and dyn_no_r > 0);

    // Write full checkpoint (including rel_logits_proj) from p_off, then evaluate .global and .all.
    var tmp = "/tmp/nano-inkling-nope-XXXXXX".*;
    const dir = std.mem.span(mkdtemp(&tmp) orelse return error.TempDirectoryFailed);
    defer _ = rmdir(dir);
    const path = try std.fmt.allocPrintSentinel(a, "{s}/model.safetensors", .{dir}, 0);
    defer _ = unlink(path);
    try writeFixture(a, &p_off, path);
    const ctx = try @import("gpu.zig").open();
    defer ctx.deinit();
    var store = try @import("weights").Store.init(a, ctx, dir, 0);
    defer store.deinit();

    var seq: [44]u32 = undefined;
    for (&seq, 0..) |*t, i| t.* = @intCast((i * 5 + 1) % 13 + 1);

    for ([_]*const @import("ir.zig").Program{ &p_global, &p_all }) |prog| {
        var ref = try Engine.init(gpa, prog, ctx, &store, .{ .batch = 32 }, 64);
        defer ref.deinit();
        var fast_base = try Engine.init(gpa, prog, ctx, &store, .{
            .batch = 32,
            .relative_stripes = true,
            .relative_long_fast = true,
            .relative_inline_bias = true,
            .fuse_relative_gate = true,
            .fuse_qkvr_prep = true,
            .store_pack_bf16 = true,
            .conv_ring = true,
            .fuse_inkling_conv = true,
            .fuse_bf16_cast = true,
            .rmsnorm_fast = true,
            .scratch_reuse = true,
        }, 64);
        defer fast_base.deinit();
        var prefill_base = try Engine.init(gpa, prog, ctx, &store, .{
            .batch = 32,
            .relative_stripes = true,
            .relative_tile_prefill = true,
            .relative_long_fast = true,
            .relative_prefill_fast = true,
            .store_pack_bf16 = true,
            .fuse_bf16_cast = true,
            .rmsnorm_fast = true,
            .scratch_reuse = true,
        }, 64);
        defer prefill_base.deinit();
        var tuned = try Engine.init(gpa, prog, ctx, &store, .{
            .batch = 32,
            .relative_stripes = true,
            .relative_tile_prefill = true,
            .relative_bias16 = true,
            .relative_bias_d8 = true,
            .relative_bias_decode = true,
            .relative_long_fast = true,
            .relative_prefill_fast = true,
            .relative_inline_bias = true,
            .fuse_relative_gate = true,
            .fuse_qkvr_prep = true,
            .store_pack_bf16 = true,
            .conv_ring = true,
            .fuse_inkling_conv = true,
            .fuse_bf16_cast = true,
            .rmsnorm_fast = true,
            .scratch_reuse = true,
            .concurrent_scratch = true,
            .tail_trim = 8,
            .d2_gate = true,
        }, 64);
        defer tuned.deinit();

        try ref.forward(seq[0..20], 19);
        try ref.forward(seq[20..40], 19);
        try ref.forward(seq[40..41], 0);
        try ref.forward(seq[41..42], 0);
        const ref_logits = try a.dupe(f32, try ref.logits(0));

        try fast_base.forward(seq[0..20], 19);
        try fast_base.forward(seq[20..40], 19);
        try fast_base.forward(seq[40..41], 0);
        try fast_base.forward(seq[41..42], 0);
        for (ref_logits, try fast_base.logits(0)) |rv, fv| {
            try std.testing.expect(std.math.isFinite(fv));
            try std.testing.expectApproxEqAbs(rv, fv, 5e-2);
        }

        try prefill_base.forward(seq[0..20], 19);
        try prefill_base.forward(seq[20..40], 19);
        try prefill_base.forward(seq[40..41], 0);
        try prefill_base.forward(seq[41..42], 0);
        const pbase_logits = try a.dupe(f32, try prefill_base.logits(0));
        for (pbase_logits) |pv| try std.testing.expect(std.math.isFinite(pv));

        try tuned.forward(seq[0..20], 19);
        try tuned.forward(seq[20..40], 19);
        try tuned.forward(seq[40..41], 0);
        try tuned.forward(seq[41..42], 0);
        for (pbase_logits, try tuned.logits(0)) |pv, tv| {
            try std.testing.expect(std.math.isFinite(tv));
            try std.testing.expectApproxEqAbs(pv, tv, 1e-3);
        }

        var lsa_prefill_base = try Engine.init(gpa, prog, ctx, &store, .{
            .batch = 32,
            .relative_stripes = true,
            .relative_tile_prefill = true,
            .relative_long_fast = true,
            .relative_prefill_fast = true,
            .store_pack_bf16 = true,
            .fuse_bf16_cast = true,
            .rmsnorm_fast = true,
            .scratch_reuse = true,
            .lsa = true,
            .lsa_budget = 8,
            .lsa_window = 12,
            .lsa_coarse_threshold = 0,
        }, 64);
        defer lsa_prefill_base.deinit();
        var lsa_tuned = try Engine.init(gpa, prog, ctx, &store, .{
            .batch = 32,
            .relative_stripes = true,
            .relative_tile_prefill = true,
            .relative_bias16 = true,
            .relative_bias_d8 = true,
            .relative_bias_decode = true,
            .relative_long_fast = true,
            .relative_prefill_fast = true,
            .relative_inline_bias = true,
            .fuse_relative_gate = true,
            .fuse_qkvr_prep = true,
            .store_pack_bf16 = true,
            .conv_ring = true,
            .fuse_inkling_conv = true,
            .fuse_bf16_cast = true,
            .rmsnorm_fast = true,
            .scratch_reuse = true,
            .concurrent_scratch = true,
            .tail_trim = 8,
            .d2_gate = true,
            .lsa = true,
            .lsa_budget = 8,
            .lsa_window = 12,
            .lsa_coarse_threshold = 0,
        }, 64);
        defer lsa_tuned.deinit();
        try lsa_prefill_base.forward(seq[0..20], 19);
        try lsa_prefill_base.forward(seq[20..40], 19);
        try lsa_prefill_base.forward(seq[40..41], 0);
        try lsa_prefill_base.forward(seq[41..42], 0);
        const lsa_pbase_out = try a.dupe(f32, try lsa_prefill_base.logits(0));
        try lsa_tuned.forward(seq[0..20], 19);
        try lsa_tuned.forward(seq[20..40], 19);
        try lsa_tuned.forward(seq[40..41], 0);
        try lsa_tuned.forward(seq[41..42], 0);
        for (lsa_pbase_out, try lsa_tuned.logits(0)) |pv, tv| {
            try std.testing.expect(std.math.isFinite(tv));
            try std.testing.expectApproxEqAbs(pv, tv, 1e-3);
        }
    }
}

test "Engine.Capture and Engine.rewind preserve exact logits and traces" {
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
    c.max_context = 32;
    var p = try lower(gpa, c);
    defer p.deinit();
    try std.testing.expect(p.draft_final_norm != null);

    var tmp = "/tmp/nano-inkling-cap-XXXXXX".*;
    const dir = std.mem.span(mkdtemp(&tmp) orelse return error.TempDirectoryFailed);
    defer _ = rmdir(dir);
    const path = try std.fmt.allocPrintSentinel(a, "{s}/model.safetensors", .{dir}, 0);
    defer _ = unlink(path);
    try writeFixture(a, &p, path);

    const ctx = try @import("gpu.zig").open();
    defer ctx.deinit();
    var store = try @import("weights").Store.init(a, ctx, dir, 0);
    defer store.deinit();
    const exec_mod = @import("exec.zig");
    const Engine = exec_mod.Engine;

    const knobs: @import("knobs.zig").Knobs = .{
        .batch = 16,
        .bf16_linear_tile = true,
        .bf16_prefill_tile = 8,
        .fuse_bf16_cast = true,
        .tail_trim = 2,
        .scratch_reuse = true,
        .concurrent_scratch = true,
    };

    // Reference: uninterrupted execution of prefix[0..8] + kept[8..10] + next[10..12]
    const tokens = [_]u32{ 1, 4, 11, 2, 3, 6, 8, 9, 5, 7, 10, 12 };
    var ref = try Engine.init(gpa, &p, ctx, &store, knobs, 32);
    defer ref.deinit();
    try ref.forward(tokens[0..10], 9);
    var ref_trace = try a.alloc(f32, p.traces.len * 10 * c.hidden);
    var ref2 = try Engine.init(gpa, &p, ctx, &store, .{ .batch = 16 }, 32);
    defer ref2.deinit();
    try ref2.forwardTraced(tokens[0..10], 0, .{ .data = ref_trace, .width = c.hidden });

    // Speculative capture on tail rows [4..10) with out_first = 9 (so head trims unembed but computes draft_final_norm on rows 4..9!).
    var spec = try Engine.init(gpa, &p, ctx, &store, knobs, 32);
    defer spec.deinit();
    const cap_indices = [_]u32{ 1, 5 };
    const cap_buf = try ctx.alloc(6 * 3 * c.hidden);
    defer cap_buf.deinit();
    var cap: exec_mod.Capture = .{
        .trace_indices = &cap_indices,
        .include_final_norm = true,
        .row0 = 4,
        .rows = 6,
        .dest = cap_buf,
        .width = c.hidden,
        .stride = 3 * c.hidden,
    };
    spec.capture = &cap;
    try spec.forward(tokens[0..10], 9);
    spec.capture = null;
    try std.testing.expectEqualSlices(f32, try ref.logits(9), try spec.logits(9));
    // Verify captured trace[1] and trace[5] match exact reference traces for rows 4..9.
    const cf = cap_buf.floats();
    for (0..6) |r| {
        const ref_row = 4 + r;
        const t1_ref = ref_trace[(1 * 10 + ref_row) * c.hidden ..][0..c.hidden];
        const t5_ref = ref_trace[(5 * 10 + ref_row) * c.hidden ..][0..c.hidden];
        for (0..c.hidden) |d| {
            try std.testing.expectApproxEqAbs(t1_ref[d], cf[r * 3 * c.hidden + d], 1e-4);
            try std.testing.expectApproxEqAbs(t5_ref[d], cf[r * 3 * c.hidden + c.hidden + d], 1e-4);
        }
    }

    // Now test rewind: feed 4 candidate tokens [10, 0, 1, 2], keep first 1 (`10`), rewind to 11, then feed `12`.
    try ref.forward(tokens[10..12], 0);
    const cand = [_]u32{ tokens[10], 0, 1, 2 };
    try spec.forward(&cand, 0);
    try spec.rewind(11);
    try std.testing.expectEqual(@as(u32, 11), spec.position);
    // Logits for kept token `cand[0]` remain intact after rewind.
    try std.testing.expectEqualSlices(f32, try ref.logits(0), try spec.logits(0));
    try spec.forward(tokens[11..12], 0);
    try std.testing.expectEqualSlices(f32, try ref.logits(1), try spec.logits(0));

    // Verify fast verify T=5 & T=7 paths (mk_bf16_linear_2xR_vec, mk_expert_block_*_m8_ct, mk_relative_prefill_stripes).
    var fast_k = knobs;
    fast_k.bf16_decode_vec = true;
    fast_k.bf16_decode_splitk = true;
    fast_k.expert_mlx_scale = true;
    fast_k.expert_block_m64n64 = true;
    fast_k.expert_fp_lut = true;
    fast_k.expert_compact_tiles = true;
    fast_k.expert_bf16_io = true;
    fast_k.bf16_prefill_matrix = true;
    fast_k.bf16_matrix_small = true;
    fast_k.relative_tile_prefill = true;
    fast_k.relative_stripes = true;
    var fast_ref = try Engine.init(gpa, &p, ctx, &store, .{ .batch = 16 }, 32);
    defer fast_ref.deinit();
    var fast_eng = try Engine.init(gpa, &p, ctx, &store, fast_k, 32);
    defer fast_eng.deinit();
    try fast_ref.forward(tokens[0..5], 0);
    try fast_eng.forward(tokens[0..5], 0);
    for (0..5) |r| {
        for (try fast_ref.logits(@intCast(r)), try fast_eng.logits(@intCast(r))) |ev, av| {
            try std.testing.expectApproxEqAbs(ev, av, 0.05);
        }
    }
    try fast_ref.forward(tokens[5..12], 0);
    try fast_eng.forward(tokens[5..12], 0);
    for (0..7) |r| {
        for (try fast_ref.logits(@intCast(r)), try fast_eng.logits(@intCast(r))) |ev, av| {
            try std.testing.expectApproxEqAbs(ev, av, 0.05);
        }
    }
}

test "Inkling2 multi-row LSA rewind, LoopCD 1xR fusion, and cross-loop LSA reuse" {
    const gpa = std.testing.allocator;
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var c = try parse(a, test_config);
    c.hidden = 128;
    c.intermediate = 64;
    c.heads = 3;
    c.kv_heads = 1;
    c.head_dim = 128;
    c.rel_dim = 16;
    c.rel_extent = 16;
    c.window = 16;
    c.log_scaling_floor = 3;
    c.vocab = 16;
    c.unpadded_vocab = 14;
    c.eos = try a.dupe(u32, &.{13});
    c.gate_rank = 4;
    c.max_context = 64;

    var p = try lower(gpa, c);
    defer p.deinit();
    var tmp = "/tmp/nano-inkling-mix-XXXXXX".*;
    const dir = std.mem.span(mkdtemp(&tmp) orelse return error.TempDirectoryFailed);
    defer _ = rmdir(dir);
    const path = try std.fmt.allocPrintSentinel(a, "{s}/model.safetensors", .{dir}, 0);
    defer _ = unlink(path);
    try writeFixture(a, &p, path);
    const ctx = try @import("gpu.zig").open();
    defer ctx.deinit();
    var store = try @import("weights").Store.init(a, ctx, dir, 0);
    defer store.deinit();
    const exec_mod = @import("exec.zig");
    const Engine = exec_mod.Engine;

    var seq: [44]u32 = undefined;
    for (&seq, 0..) |*t, i| t.* = @intCast((i * 7 + 3) % 13 + 1);

    // 1. Multi-row LSA (rows=8) + inline_bias + relative_gate + rewind across 4-token / 16-token boundaries.
    var lsa_ref = try Engine.init(gpa, &p, ctx, &store, .{
        .batch = 32,
        .lsa = true,
        .lsa_budget = 8,
        .lsa_window = 12,
        .lsa_coarse_threshold = 16,
    }, 64);
    defer lsa_ref.deinit();
    const lsa_fast_knobs: @import("knobs.zig").Knobs = .{
        .batch = 32,
        .relative_stripes = true,
        .relative_long_fast = true,
        .relative_inline_bias = true,
        .fuse_relative_gate = true,
        .fuse_qkvr_prep = true,
        .store_pack_bf16 = true,
        .conv_ring = true,
        .fuse_inkling_conv = true,
        .fuse_bf16_cast = true,
        .rmsnorm_fast = true,
        .scratch_reuse = true,
        .lsa = true,
        .lsa_budget = 8,
        .lsa_window = 12,
        .lsa_coarse_threshold = 16,
    };
    var lsa_fast = try Engine.init(gpa, &p, ctx, &store, lsa_fast_knobs, 64);
    defer lsa_fast.deinit();
    var lsa_bf16_ref_knobs = lsa_fast_knobs;
    lsa_bf16_ref_knobs.relative_inline_bias = false;
    lsa_bf16_ref_knobs.fuse_relative_gate = false;
    lsa_bf16_ref_knobs.fuse_qkvr_prep = false;
    var lsa_bf16_ref = try Engine.init(gpa, &p, ctx, &store, lsa_bf16_ref_knobs, 64);
    defer lsa_bf16_ref.deinit();
    var lsa_fast_seq = try Engine.init(gpa, &p, ctx, &store, lsa_fast_knobs, 64);
    defer lsa_fast_seq.deinit();

    try lsa_ref.forward(seq[0..32], 31);
    try lsa_bf16_ref.forward(seq[0..32], 31);
    try lsa_fast.forward(seq[0..32], 31);
    try lsa_fast_seq.forward(seq[0..32], 31);
    try lsa_ref.forward(seq[32..40], 0);
    try lsa_bf16_ref.forward(seq[32..40], 0);
    try lsa_fast.forward(seq[32..40], 0);
    const ref_r3_pre = try a.dupe(f32, try lsa_ref.logits(3));
    const seq_step_logits = try a.alloc([]const f32, 8);
    for (0..8) |r| {
        try lsa_fast_seq.forward(seq[32 + r .. 33 + r], 0);
        seq_step_logits[r] = try a.dupe(f32, try lsa_fast_seq.logits(0));
    }
    for (0..8) |r| {
        const fl = try lsa_fast.logits(@intCast(r));
        const brl = try lsa_bf16_ref.logits(@intCast(r));
        for (brl, fl, seq_step_logits[r]) |bv, fv, sv| {
            try std.testing.expect(std.math.isFinite(fv));
            try std.testing.expectApproxEqAbs(sv, fv, 1e-4);
            try std.testing.expectApproxEqAbs(bv, fv, 1e-3);
        }
    }
    try lsa_fast.rewind(35);
    try lsa_fast.forward(seq[35..36], 0);
    try lsa_bf16_ref.rewind(35);
    try lsa_bf16_ref.forward(seq[35..36], 0);
    try lsa_ref.rewind(35);
    try lsa_ref.forward(seq[35..36], 0);
    for (ref_r3_pre, try lsa_ref.logits(0), try lsa_bf16_ref.logits(0), try lsa_fast.logits(0), seq_step_logits[3]) |rpre, rpost, bv, fv, sv| {
        try std.testing.expect(std.math.isFinite(fv));
        try std.testing.expectApproxEqAbs(rpre, rpost, 1e-5);
        try std.testing.expectApproxEqAbs(sv, fv, 1e-4);
        try std.testing.expectApproxEqAbs(bv, fv, 1e-3);
    }

    // 2. Cross-loop LSA block reuse preserves close finite logits with strictly fewer dispatches (including d2_gate).
    var eng_no_reuse = try Engine.init(gpa, &p, ctx, &store, .{
        .batch = 32,
        .relative_stripes = true,
        .relative_long_fast = true,
        .relative_inline_bias = true,
        .fuse_relative_gate = true,
        .fuse_qkvr_prep = true,
        .store_pack_bf16 = true,
        .fuse_bf16_cast = true,
        .rmsnorm_fast = true,
        .scratch_reuse = true,
        .d2_gate = true,
        .lsa = true,
        .lsa_budget = 8,
        .lsa_window = 12,
        .lsa_coarse_threshold = 16,
        .lsa_cross_loop_reuse = false,
    }, 64);
    defer eng_no_reuse.deinit();
    var eng_reuse = try Engine.init(gpa, &p, ctx, &store, .{
        .batch = 32,
        .relative_stripes = true,
        .relative_long_fast = true,
        .relative_inline_bias = true,
        .fuse_relative_gate = true,
        .fuse_qkvr_prep = true,
        .store_pack_bf16 = true,
        .fuse_bf16_cast = true,
        .rmsnorm_fast = true,
        .scratch_reuse = true,
        .d2_gate = true,
        .lsa = true,
        .lsa_budget = 8,
        .lsa_window = 12,
        .lsa_coarse_threshold = 16,
        .lsa_cross_loop_reuse = true,
    }, 64);
    defer eng_reuse.deinit();

    var disp_no: u64 = 0;
    var disp_yes: u64 = 0;
    try eng_no_reuse.forward(seq[0..32], 31);
    disp_no += eng_no_reuse.last.dispatches;
    try eng_no_reuse.forward(seq[32..40], 0);
    disp_no += eng_no_reuse.last.dispatches;
    try eng_no_reuse.forward(seq[40..41], 0);
    disp_no += eng_no_reuse.last.dispatches;
    try eng_no_reuse.forward(seq[41..42], 0);
    disp_no += eng_no_reuse.last.dispatches;
    const no_reuse_logits = try a.dupe(f32, try eng_no_reuse.logits(0));
    try eng_no_reuse.d2Settle();

    try eng_reuse.forward(seq[0..32], 31);
    disp_yes += eng_reuse.last.dispatches;
    try eng_reuse.forward(seq[32..40], 0);
    disp_yes += eng_reuse.last.dispatches;
    try eng_reuse.forward(seq[40..41], 0);
    disp_yes += eng_reuse.last.dispatches;
    try eng_reuse.forward(seq[41..42], 0);
    disp_yes += eng_reuse.last.dispatches;

    const reuse_logits = try a.dupe(f32, try eng_reuse.logits(0));
    try eng_reuse.d2Settle();
    try std.testing.expect(disp_yes < disp_no);
    for (no_reuse_logits, reuse_logits) |nv, yv| {
        try std.testing.expect(std.math.isFinite(yv));
        try std.testing.expectApproxEqAbs(nv, yv, 0.25);
    }

    var eng_reuse_sync = try Engine.init(gpa, &p, ctx, &store, .{
        .batch = 32,
        .relative_stripes = true,
        .relative_long_fast = true,
        .relative_inline_bias = true,
        .fuse_relative_gate = true,
        .fuse_qkvr_prep = true,
        .store_pack_bf16 = true,
        .fuse_bf16_cast = true,
        .rmsnorm_fast = true,
        .scratch_reuse = true,
        .d2_gate = false,
        .lsa = true,
        .lsa_budget = 8,
        .lsa_window = 12,
        .lsa_coarse_threshold = 16,
        .lsa_cross_loop_reuse = true,
    }, 64);
    defer eng_reuse_sync.deinit();
    try eng_reuse_sync.forward(seq[0..32], 31);
    try eng_reuse_sync.forward(seq[32..40], 0);
    try eng_reuse_sync.forward(seq[40..41], 0);
    try eng_reuse_sync.forward(seq[41..42], 0);
    for (try eng_reuse_sync.logits(0), reuse_logits) |sv, rv| {
        try std.testing.expectApproxEqAbs(sv, rv, 1e-4);
    }

    // 3. Multi-row LoopCD (rows=5, rows=8) + 2-pass Capture parity across hidden/logits/logits_adaptive.
    var c2 = c;
    c2.loop_passes = 2;
    inline for (.{ LoopCdMode.hidden, LoopCdMode.logits, LoopCdMode.logits_adaptive }) |mode| {
        var p_cd = try lowerWithOptions(gpa, c2, .{ .loop_passes = 2, .loopcd = mode, .loopcd_omega = 0.7, .loopcd_weak_pass = 1 });
        defer p_cd.deinit();
        try std.testing.expectEqual(@as(usize, 37), p_cd.traces.len);
        var tmp_cd = "/tmp/nano-inkling-cd-XXXXXX".*;
        const dir_cd = std.mem.span(mkdtemp(&tmp_cd) orelse return error.TempDirectoryFailed);
        defer _ = rmdir(dir_cd);
        const path_cd = try std.fmt.allocPrintSentinel(a, "{s}/model.safetensors", .{dir_cd}, 0);
        defer _ = unlink(path_cd);
        try writeFixture(a, &p_cd, path_cd);
        var store_cd = try @import("weights").Store.init(a, ctx, dir_cd, 0);
        defer store_cd.deinit();

        var cd_ref = try Engine.init(gpa, &p_cd, ctx, &store_cd, .{ .batch = 16 }, 64);
        defer cd_ref.deinit();
        var cd_fast = try Engine.init(gpa, &p_cd, ctx, &store_cd, .{
            .batch = 16,
            .bf16_linear_tile = true,
            .bf16_prefill_tile = 8,
            .bf16_decode_vec = true,
            .bf16_decode_wide = true,
            .fuse_bf16_cast = true,
            .tail_trim = 2,
            .scratch_reuse = true,
            .concurrent_scratch = true,
        }, 64);
        defer cd_fast.deinit();

        const cap_ids = [_]u32{ 5, 17, 29 };
        const cap_buf = try ctx.alloc(8 * 4 * c.hidden);
        defer cap_buf.deinit();
        var cap5: exec_mod.Capture = .{
            .trace_indices = &cap_ids,
            .include_final_norm = true,
            .row0 = 0,
            .rows = 5,
            .dest = cap_buf,
            .width = c.hidden,
            .stride = 4 * c.hidden,
        };
        cd_fast.capture = &cap5;
        try cd_ref.forward(seq[0..5], 0);
        try cd_fast.forward(seq[0..5], 0);
        cd_fast.capture = null;
        for (0..5) |r| {
            for (try cd_ref.logits(@intCast(r)), try cd_fast.logits(@intCast(r))) |ev, av| {
                try std.testing.expect(std.math.isFinite(av));
                try std.testing.expectApproxEqAbs(ev, av, 1e-3);
            }
        }

        var cap8: exec_mod.Capture = .{
            .trace_indices = &cap_ids,
            .include_final_norm = true,
            .row0 = 0,
            .rows = 8,
            .dest = cap_buf,
            .width = c.hidden,
            .stride = 4 * c.hidden,
        };
        cd_fast.capture = &cap8;
        try cd_ref.forward(seq[5..13], 0);
        try cd_fast.forward(seq[5..13], 0);
        cd_fast.capture = null;
        for (0..8) |r| {
            for (try cd_ref.logits(@intCast(r)), try cd_fast.logits(@intCast(r))) |ev, av| {
                try std.testing.expect(std.math.isFinite(av));
                try std.testing.expectApproxEqAbs(ev, av, 1e-3);
            }
        }
    }
}

pub const chat_template: chat.Template = .{
    .render = renderChat,
    .next_turn = nextTurn,
    .think_open = "<|content_thinking|>",
    .think_close = "<|end_message|>",
    .starts_in_think = false,
};

fn emitEffort(out: *std.ArrayList(u8), a: std.mem.Allocator, thinking: bool) !void {
    try out.print(a, "<|message_system|><|content_text|>Thinking effort level: {s}<|end_message|>", .{
        if (thinking) "0.9" else "0",
    });
}

fn renderChat(a: std.mem.Allocator, messages: []const chat.Message, options: chat.Options) ![]u8 {
    var out: std.ArrayList(u8) = .empty;
    var effort_emitted = false;
    for (messages) |m| {
        if (!effort_emitted and m.role != .system) {
            try emitEffort(&out, a, options.thinking);
            effort_emitted = true;
        }
        switch (m.role) {
            .system => try out.print(a, "<|message_system|><|content_text|>{s}<|end_message|>", .{m.content}),
            .user => try out.print(a, "<|message_user|><|content_text|>{s}<|end_message|>", .{m.content}),
            .assistant => try out.print(a, "<|message_model|><|content_text|>{s}<|end_message|><|content_model_end_sampling|>", .{m.content}),
        }
    }
    if (!effort_emitted) {
        try emitEffort(&out, a, options.thinking);
    }
    return out.items;
}

fn nextTurn(a: std.mem.Allocator, closed: bool, user: []const u8, _: chat.Options) ![]u8 {
    return std.fmt.allocPrint(a, "{s}<|message_user|><|content_text|>{s}<|end_message|>", .{
        if (closed) "" else "<|end_message|><|content_model_end_sampling|>",
        user,
    });
}

/// Render an Inkling 2 user turn with one or more images (matching chat_template.jinja):
/// each image emits `<|message_user|><|content_image|><|unused_200054|>...<|end_message|>`
/// with `image_token_counts[i]` placeholder tokens, followed by the text part (if non-empty).
pub fn renderImageTurn(
    a: std.mem.Allocator,
    system: ?[]const u8,
    user: []const u8,
    image_token_counts: []const u32,
    thinking: bool,
) ![]u8 {
    var out: std.ArrayList(u8) = .empty;
    if (system) |s| try out.print(a, "<|message_system|><|content_text|>{s}<|end_message|>", .{s});
    try emitEffort(&out, a, thinking);
    for (image_token_counts) |count| {
        try out.appendSlice(a, "<|message_user|><|content_image|>");
        for (0..count) |_| try out.appendSlice(a, "<|unused_200054|>");
        try out.appendSlice(a, "<|end_message|>");
    }
    if (user.len > 0 or image_token_counts.len == 0) {
        try out.print(a, "<|message_user|><|content_text|>{s}<|end_message|>", .{user});
    }
    return out.items;
}

pub fn renderImageNextTurn(
    a: std.mem.Allocator,
    closed: bool,
    user: []const u8,
    image_token_counts: []const u32,
) ![]u8 {
    var out: std.ArrayList(u8) = .empty;
    if (!closed) try out.appendSlice(a, "<|end_message|><|content_model_end_sampling|>");
    for (image_token_counts) |count| {
        try out.appendSlice(a, "<|message_user|><|content_image|>");
        for (0..count) |_| try out.appendSlice(a, "<|unused_200054|>");
        try out.appendSlice(a, "<|end_message|>");
    }
    if (user.len > 0 or image_token_counts.len == 0) {
        try out.print(a, "<|message_user|><|content_text|>{s}<|end_message|>", .{user});
    }
    return out.items;
}

test "Inkling chat template matches HuggingFace text-only chat_template.jinja" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();

    const s_off = try renderChat(a, &.{
        .{ .role = .system, .content = "Be brief." },
        .{ .role = .user, .content = "hi" },
    }, .{ .thinking = false });
    try std.testing.expectEqualStrings(
        "<|message_system|><|content_text|>Be brief.<|end_message|><|message_system|><|content_text|>Thinking effort level: 0<|end_message|><|message_user|><|content_text|>hi<|end_message|>",
        s_off,
    );

    const s_on = try renderChat(a, &.{
        .{ .role = .system, .content = "Be brief." },
        .{ .role = .user, .content = "hi" },
    }, .{ .thinking = true });
    try std.testing.expectEqualStrings(
        "<|message_system|><|content_text|>Be brief.<|end_message|><|message_system|><|content_text|>Thinking effort level: 0.9<|end_message|><|message_user|><|content_text|>hi<|end_message|>",
        s_on,
    );

    const one = try renderChat(a, &.{.{ .role = .user, .content = "hi" }}, .{ .thinking = false });
    const whole = try renderChat(a, &.{
        .{ .role = .user, .content = "hi" },
        .{ .role = .assistant, .content = "Hello." },
        .{ .role = .user, .content = "more" },
    }, .{ .thinking = false });
    const next_closed = try nextTurn(a, true, "more", .{});
    try std.testing.expectEqualStrings(whole, try std.mem.concat(a, u8, &.{
        one,
        "<|message_model|><|content_text|>Hello.<|end_message|><|content_model_end_sampling|>",
        next_closed,
    }));
    const next_open = try nextTurn(a, false, "more", .{});
    try std.testing.expectEqualStrings(whole, try std.mem.concat(a, u8, &.{
        one,
        "<|message_model|><|content_text|>Hello.",
        next_open,
    }));

    const img_turn = try renderImageTurn(a, "Sys.", "Describe", &.{ 2, 1 }, false);
    try std.testing.expectEqualStrings(
        "<|message_system|><|content_text|>Sys.<|end_message|><|message_system|><|content_text|>Thinking effort level: 0<|end_message|><|message_user|><|content_image|><|unused_200054|><|unused_200054|><|end_message|><|message_user|><|content_image|><|unused_200054|><|end_message|><|message_user|><|content_text|>Describe<|end_message|>",
        img_turn,
    );
    const img_next = try renderImageNextTurn(a, false, "Again", &.{3});
    try std.testing.expectEqualStrings(
        "<|end_message|><|content_model_end_sampling|><|message_user|><|content_image|><|unused_200054|><|unused_200054|><|unused_200054|><|end_message|><|message_user|><|content_text|>Again<|end_message|>",
        img_next,
    );
}
