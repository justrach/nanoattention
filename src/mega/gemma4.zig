//! Gemma 4 (`gemma4` / `gemma4_text` / `gemma4_unified` / `gemma4_unified_text`)
//! dense text front end for the megakernel IR (text-only path; ignores vision/audio weights).
const std = @import("std");
const ir = @import("ir.zig");
const liquid = @import("lfm3.zig");
const plan = @import("plan.zig");

pub const Checkpoint = liquid.Checkpoint;

pub const LayerType = enum {
    sliding_attention,
    full_attention,
};

pub const RopeSpec = struct {
    theta: f64,
    partial_rotary_factor: f64,
    factor: f64,
};

pub const Config = struct {
    model_type: []const u8,
    hidden: u32,
    intermediate: u32,
    layers: []const LayerType,
    heads: u32,
    sliding_kv_heads: u32,
    full_kv_heads: u32,
    sliding_head_dim: u32,
    full_head_dim: u32,
    k_eq_v_full: bool,
    sliding_window: u32,
    eps: f32,
    sliding_rope: RopeSpec,
    full_rope: RopeSpec,
    final_logit_softcapping: ?f32,
    tie_word_embeddings: bool,
    vocab: u32,
    eos: []const u32,
    max_context: u32,
};

const RopeParamsRaw = struct {
    rope_theta: ?f64 = null,
    partial_rotary_factor: ?f64 = null,
    factor: ?f64 = null,
    rope_type: ?[]const u8 = null,
};

const RopeParametersByLayer = struct {
    sliding_attention: ?RopeParamsRaw = null,
    full_attention: ?RopeParamsRaw = null,
    rope_theta: ?f64 = null,
    partial_rotary_factor: ?f64 = null,
    factor: ?f64 = null,
    rope_type: ?[]const u8 = null,
};

const TextRaw = struct {
    model_type: []const u8 = "gemma4_text",
    hidden_size: u32,
    intermediate_size: u32,
    num_hidden_layers: u32,
    layer_types: ?[]const LayerType = null,
    sliding_window_pattern: ?u32 = null,
    num_attention_heads: u32,
    num_key_value_heads: u32,
    num_global_key_value_heads: ?u32 = null,
    head_dim: u32,
    global_head_dim: ?u32 = null,
    attention_k_eq_v: bool = false,
    attention_bias: bool = false,
    sliding_window: u32 = 1024,
    rms_norm_eps: f32 = 1e-6,
    hidden_activation: ?[]const u8 = null,
    final_logit_softcapping: ?f32 = null,
    rope_theta: ?f64 = null,
    rope_traditional: bool = false,
    rope_parameters: ?RopeParametersByLayer = null,
    enable_moe_block: bool = false,
    hidden_size_per_layer_input: u32 = 0,
    num_kv_shared_layers: u32 = 0,
    use_double_wide_mlp: bool = false,
    tie_word_embeddings: ?bool = null,
    vocab_size: u32,
    max_position_embeddings: u32,
    eos_token_id: ?std.json.Value = null,
};

fn positive(x: f64) bool {
    return std.math.isFinite(x) and x > 0;
}

fn isGemma4ModelType(mt: []const u8) bool {
    return std.mem.eql(u8, mt, "gemma4") or
        std.mem.eql(u8, mt, "gemma4_text") or
        std.mem.eql(u8, mt, "gemma4_unified") or
        std.mem.eql(u8, mt, "gemma4_unified_text");
}

fn tokenId(value: std.json.Value, vocab: u32) !u32 {
    if (value != .integer or value.integer < 0 or value.integer >= vocab) return error.InvalidEos;
    return @intCast(value.integer);
}

fn appendEos(a: std.mem.Allocator, out: *std.ArrayList(u32), value: std.json.Value, vocab: u32) !void {
    switch (value) {
        .integer => {
            const id = try tokenId(value, vocab);
            if (std.mem.indexOfScalar(u32, out.items, id) == null) try out.append(a, id);
        },
        .array => |arr| for (arr.items) |elem| {
            const id = try tokenId(elem, vocab);
            if (std.mem.indexOfScalar(u32, out.items, id) == null) try out.append(a, id);
        },
        else => return error.InvalidEos,
    }
}

fn resolveRopeSpec(raw_layer: ?RopeParamsRaw, raw_root: ?RopeParametersByLayer, default_theta: f64, flat_theta: ?f64) !RopeSpec {
    const rp: RopeParamsRaw = raw_layer orelse if (raw_root) |rr| .{
        .rope_theta = rr.rope_theta,
        .partial_rotary_factor = rr.partial_rotary_factor,
        .factor = rr.factor,
        .rope_type = rr.rope_type,
    } else .{};
    const theta = rp.rope_theta orelse flat_theta orelse default_theta;
    const partial = rp.partial_rotary_factor orelse 1.0;
    const factor = rp.factor orelse 1.0;
    if (!positive(theta) or !positive(partial) or partial > 1.0 or !positive(factor)) return error.InvalidConfig;
    const rt = rp.rope_type orelse "default";
    if (std.mem.eql(u8, rt, "default")) {
        if (partial != 1.0 or factor != 1.0) return error.UnsupportedRopeType;
    } else if (!std.mem.eql(u8, rt, "proportional")) {
        return error.UnsupportedRopeType;
    }
    return .{ .theta = theta, .partial_rotary_factor = partial, .factor = factor };
}

pub fn parse(a: std.mem.Allocator, bytes: []const u8) !Config {
    const root = try std.json.parseFromSliceLeaky(std.json.Value, a, bytes, .{});
    if (root != .object) return error.InvalidConfig;
    const obj = root.object;
    const outer_mt = switch (obj.get("model_type") orelse return error.MissingConfigField) {
        .string => |s| s,
        else => return error.InvalidConfig,
    };
    if (!isGemma4ModelType(outer_mt)) return error.NotGemma4;

    var outer_eos: ?std.json.Value = obj.get("eos_token_id");
    if (outer_eos == null) if (obj.get("generation_config")) |gc| {
        if (gc == .object) outer_eos = gc.object.get("eos_token_id");
    };
    var outer_tie: ?bool = null;
    if (obj.get("tie_word_embeddings")) |tv| if (tv == .bool) {
        outer_tie = tv.bool;
    };

    const text_val: std.json.Value = if (obj.get("text_config")) |tc| blk: {
        if (tc != .object) return error.InvalidConfig;
        break :blk tc;
    } else root;

    const r = try std.json.parseFromValueLeaky(TextRaw, a, text_val, .{ .ignore_unknown_fields = true });
    if (!isGemma4ModelType(r.model_type)) return error.NotGemma4;
    if (r.attention_bias) return error.UnsupportedAttentionBias;
    if (r.enable_moe_block) return error.UnsupportedMoeBlock;
    if (r.hidden_size_per_layer_input != 0) return error.UnsupportedPerLayerInput;
    if (r.num_kv_shared_layers != 0) return error.UnsupportedKvSharing;
    if (r.use_double_wide_mlp) return error.UnsupportedDoubleWideMlp;
    if (r.rope_traditional) return error.UnsupportedRopeTraditional;
    if (r.hidden_activation) |act| {
        if (!std.mem.eql(u8, act, "gelu_pytorch_tanh")) return error.UnsupportedActivation;
    }
    if (r.final_logit_softcapping) |cap| {
        if (!std.math.isFinite(cap) or !(cap > 0)) return error.InvalidConfig;
    }

    const h = r.hidden_size;
    const heads = r.num_attention_heads;
    if (h == 0 or heads == 0 or r.intermediate_size == 0 or r.num_hidden_layers == 0 or
        r.vocab_size == 0 or r.max_position_embeddings == 0 or !positive(r.rms_norm_eps))
    {
        return error.InvalidConfig;
    }

    const layers: []const LayerType = if (r.layer_types) |lt| lt else blk: {
        const pattern = r.sliding_window_pattern orelse 6;
        if (pattern == 0) return error.InvalidConfig;
        const arr = try a.alloc(LayerType, r.num_hidden_layers);
        for (arr, 0..) |*slot, idx| {
            slot.* = if ((idx + 1) % pattern == 0) .full_attention else .sliding_attention;
        }
        break :blk arr;
    };
    if (layers.len != r.num_hidden_layers) return error.InvalidConfig;

    var has_sliding = false;
    var has_full = false;
    for (layers) |lt| switch (lt) {
        .sliding_attention => has_sliding = true,
        .full_attention => has_full = true,
    };

    const sliding_kv = r.num_key_value_heads;
    const full_kv = if (r.attention_k_eq_v and r.num_global_key_value_heads != null)
        r.num_global_key_value_heads.?
    else
        r.num_key_value_heads;
    const sliding_dim = r.head_dim;
    const full_dim = if (r.global_head_dim) |gh| (if (gh != 0) gh else r.head_dim) else r.head_dim;

    if (has_sliding) {
        if (sliding_kv == 0 or heads % sliding_kv != 0 or sliding_dim == 0 or sliding_dim > 512 or sliding_dim % 2 != 0 or r.sliding_window == 0)
            return error.InvalidConfig;
    }
    if (has_full) {
        if (full_kv == 0 or heads % full_kv != 0 or full_dim == 0 or full_dim > 512 or full_dim % 2 != 0)
            return error.InvalidConfig;
    }

    const sliding_rope = try resolveRopeSpec(
        if (r.rope_parameters) |rp| rp.sliding_attention else null,
        r.rope_parameters,
        10000.0,
        r.rope_theta,
    );
    const full_rope = try resolveRopeSpec(
        if (r.rope_parameters) |rp| rp.full_attention else null,
        r.rope_parameters,
        1000000.0,
        r.rope_theta,
    );

    var eos: std.ArrayList(u32) = .empty;
    if (outer_eos) |e| try appendEos(a, &eos, e, r.vocab_size);
    if (r.eos_token_id) |e| try appendEos(a, &eos, e, r.vocab_size);

    return .{
        .model_type = outer_mt,
        .hidden = h,
        .intermediate = r.intermediate_size,
        .layers = layers,
        .heads = heads,
        .sliding_kv_heads = sliding_kv,
        .full_kv_heads = full_kv,
        .sliding_head_dim = sliding_dim,
        .full_head_dim = full_dim,
        .k_eq_v_full = r.attention_k_eq_v,
        .sliding_window = r.sliding_window,
        .eps = r.rms_norm_eps,
        .sliding_rope = sliding_rope,
        .full_rope = full_rope,
        .final_logit_softcapping = r.final_logit_softcapping,
        .tie_word_embeddings = r.tie_word_embeddings orelse outer_tie orelse true,
        .vocab = r.vocab_size,
        .eos = eos.items,
        .max_context = r.max_position_embeddings,
    };
}

/// Proportional RoPE inverse frequencies matching `mlx_lm.models.rope_utils.ProportionalRoPE`.
/// Pairs past `floor(partial_rotary_factor * dim / 2)` carry frequency `0.0` (unrotated).
pub fn proportionalRopeFrequencies(a: std.mem.Allocator, dim: u32, theta: f64, partial_rotary_factor: f64, factor: f64) ![]const f32 {
    const half = dim / 2;
    const rotated_pairs: u32 = @intFromFloat(@floor(partial_rotary_factor * @as(f64, @floatFromInt(dim)) / 2.0));
    const out = try a.alloc(f32, half);
    for (0..half) |j| {
        if (j < rotated_pairs) {
            const exp = @as(f64, @floatFromInt(2 * j)) / @as(f64, @floatFromInt(dim));
            out[j] = @floatCast(1.0 / (factor * std.math.pow(f64, theta, exp)));
        } else {
            out[j] = 0.0;
        }
    }
    return out;
}

fn hasWeight(checkpoint: Checkpoint, name: []const u8) bool {
    return checkpoint.has(checkpoint.context, name);
}

fn weightPrefix(a: std.mem.Allocator, checkpoint: Checkpoint) ![]const u8 {
    for ([_][]const u8{ "language_model.model.", "model.language_model.", "model.", "" }) |prefix| {
        const probe = try std.fmt.allocPrint(a, "{s}embed_tokens.weight", .{prefix});
        if (hasWeight(checkpoint, probe)) return prefix;
    }
    return "model.";
}

fn lmHeadWeightName(checkpoint: Checkpoint, prefix: []const u8) []const u8 {
    if (std.mem.eql(u8, prefix, "language_model.model.")) {
        if (hasWeight(checkpoint, "language_model.lm_head.weight") or !hasWeight(checkpoint, "lm_head.weight"))
            return "language_model.lm_head.weight";
        return "lm_head.weight";
    }
    if (std.mem.eql(u8, prefix, "model.language_model.")) {
        if (hasWeight(checkpoint, "lm_head.weight")) return "lm_head.weight";
        if (hasWeight(checkpoint, "model.lm_head.weight")) return "model.lm_head.weight";
        if (hasWeight(checkpoint, "language_model.lm_head.weight")) return "language_model.lm_head.weight";
        return "lm_head.weight";
    }
    return "lm_head.weight";
}

/// Lower a Gemma 4 dense checkpoint into a validated megakernel IR program.
pub fn build(gpa: std.mem.Allocator, config_bytes: []const u8, checkpoint: Checkpoint) !ir.Program {
    var b = ir.Builder.init(gpa);
    errdefer b.deinit();
    const a = b.allocator();
    const c = try parse(a, config_bytes);
    const prefix = try weightPrefix(a, checkpoint);

    const h = c.hidden;
    const inter = c.intermediate;

    var has_sliding = false;
    var has_full = false;
    for (c.layers) |lt| switch (lt) {
        .sliding_attention => has_sliding = true,
        .full_attention => has_full = true,
    };

    const freqs_sliding: ir.TableId = if (has_sliding)
        try b.table(try proportionalRopeFrequencies(a, c.sliding_head_dim, c.sliding_rope.theta, c.sliding_rope.partial_rotary_factor, c.sliding_rope.factor))
    else
        0;
    const freqs_full: ir.TableId = if (has_full)
        try b.table(try proportionalRopeFrequencies(a, c.full_head_dim, c.full_rope.theta, c.full_rope.partial_rotary_factor, c.full_rope.factor))
    else
        0;

    const tokens = try b.state(.{ .name = "tokens", .kind = .tokens, .width = 1 });
    _ = try b.state(.{ .name = "position", .kind = .position, .width = 1 });
    const x = try b.state(.{ .name = "hidden", .kind = .scratch, .width = h });
    const normed = try b.state(.{ .name = "norm", .kind = .scratch, .width = h });
    const norm_tmp = try b.state(.{ .name = "norm_tmp", .kind = .scratch, .width = h });
    const branch_out = try b.state(.{ .name = "branch_out", .kind = .scratch, .width = h });

    const ffn_gate = try b.state(.{ .name = "ffn_gate", .kind = .scratch, .width = inter });
    const ffn_up = try b.state(.{ .name = "ffn_up", .kind = .scratch, .width = inter });
    const ffn_act = try b.state(.{ .name = "ffn_act", .kind = .scratch, .width = inter });
    const mlp_out = try b.state(.{ .name = "mlp_out", .kind = .scratch, .width = h });

    const s_q_dim = c.heads * c.sliding_head_dim;
    const s_kv_dim = c.sliding_kv_heads * c.sliding_head_dim;
    const f_q_dim = c.heads * c.full_head_dim;
    const f_kv_dim = c.full_kv_heads * c.full_head_dim;

    var q_s: ir.StateId = undefined;
    var k_s: ir.StateId = undefined;
    var v_s: ir.StateId = undefined;
    var att_s: ir.StateId = undefined;
    if (has_sliding) {
        q_s = try b.state(.{ .name = "q_sliding", .kind = .scratch, .width = s_q_dim });
        k_s = try b.state(.{ .name = "k_sliding", .kind = .scratch, .width = s_kv_dim });
        v_s = try b.state(.{ .name = "v_sliding", .kind = .scratch, .width = s_kv_dim });
        att_s = try b.state(.{ .name = "att_sliding", .kind = .scratch, .width = s_q_dim });
    }

    var q_f: ir.StateId = undefined;
    var k_f: ir.StateId = undefined;
    var v_f: ir.StateId = undefined;
    var att_f: ir.StateId = undefined;
    if (has_full) {
        if (has_sliding and f_q_dim == s_q_dim and f_kv_dim == s_kv_dim) {
            q_f = q_s;
            k_f = k_s;
            v_f = v_s;
            att_f = att_s;
        } else {
            q_f = try b.state(.{ .name = "q_full", .kind = .scratch, .width = f_q_dim });
            k_f = try b.state(.{ .name = "k_full", .kind = .scratch, .width = f_kv_dim });
            v_f = try b.state(.{ .name = "v_full", .kind = .scratch, .width = f_kv_dim });
            att_f = try b.state(.{ .name = "att_full", .kind = .scratch, .width = f_q_dim });
        }
    }

    const logits = try b.state(.{ .name = "logits", .kind = .scratch, .width = c.vocab });
    const next = try b.state(.{ .name = "next", .kind = .scratch, .width = 1 });

    const embed_w = try b.weight(try b.print("{s}embed_tokens.weight", .{prefix}), &.{ c.vocab, h });
    try b.op(.{ .embed = .{ .table = embed_w, .tokens = b.view(tokens), .out = b.view(x) } });
    const embed_scale: f32 = @floatCast(@sqrt(@as(f64, @floatFromInt(h))));
    try b.op(.{ .row_transform = .{ .x = b.view(x), .out = b.view(x), .transform = .{ .scale = embed_scale } } });
    try b.trace(b.view(x));

    for (c.layers, 0..) |layer_type, idx| {
        const layer: u32 = @intCast(idx);
        const sliding = layer_type == .sliding_attention;
        const dim = if (sliding) c.sliding_head_dim else c.full_head_dim;
        const kv_heads = if (sliding) c.sliding_kv_heads else c.full_kv_heads;
        const q_dim = c.heads * dim;
        const kv_dim = kv_heads * dim;
        const q = if (sliding) q_s else q_f;
        const k = if (sliding) k_s else k_f;
        const v = if (sliding) v_s else v_f;
        const att = if (sliding) att_s else att_f;
        const use_k_eq_v = !sliding and c.k_eq_v_full;

        const in_norm_w = try b.weight(try b.print("{s}layers.{d}.input_layernorm.weight", .{ prefix, layer }), &.{h});
        try b.op(.{ .rmsnorm = .{ .x = b.view(x), .weight = in_norm_w, .out = b.view(normed), .group = h, .eps = c.eps } });

        const w_q = try b.weight(try b.print("{s}layers.{d}.self_attn.q_proj.weight", .{ prefix, layer }), &.{ q_dim, h });
        const w_k = try b.weight(try b.print("{s}layers.{d}.self_attn.k_proj.weight", .{ prefix, layer }), &.{ kv_dim, h });
        try b.op(.{ .linear = .{ .w = w_q, .x = b.view(normed), .out = b.view(q) } });
        try b.op(.{ .linear = .{ .w = w_k, .x = b.view(normed), .out = b.view(k) } });
        if (!use_k_eq_v) {
            const w_v = try b.weight(try b.print("{s}layers.{d}.self_attn.v_proj.weight", .{ prefix, layer }), &.{ kv_dim, h });
            try b.op(.{ .linear = .{ .w = w_v, .x = b.view(normed), .out = b.view(v) } });
        }

        // Read raw keys into scale-free v_norm before k_norm overwrites k in place.
        const v_raw = if (use_k_eq_v) k else v;
        try b.op(.{ .rmsnorm = .{ .x = b.view(v_raw), .weight = null, .out = b.view(v), .group = dim, .eps = c.eps } });

        const q_norm_w = try b.weight(try b.print("{s}layers.{d}.self_attn.q_norm.weight", .{ prefix, layer }), &.{dim});
        const k_norm_w = try b.weight(try b.print("{s}layers.{d}.self_attn.k_norm.weight", .{ prefix, layer }), &.{dim});
        const table = if (sliding) freqs_sliding else freqs_full;
        const kind: ir.StateKind = if (sliding) .ring else .sequence;
        const horizon: u32 = if (sliding) @max(c.sliding_window - 1, 1) else 0;
        const kc = try b.state(.{ .name = try b.print("k{d}", .{layer}), .kind = kind, .width = kv_dim, .horizon = horizon });
        const vc = try b.state(.{ .name = try b.print("v{d}", .{layer}), .kind = kind, .width = kv_dim, .horizon = horizon });

        try b.op(.{ .rmsnorm = .{ .x = b.view(q), .weight = q_norm_w, .out = b.view(q), .group = dim, .eps = c.eps } });
        try b.op(.{ .rmsnorm = .{ .x = b.view(k), .weight = k_norm_w, .out = b.view(k), .group = dim, .eps = c.eps } });
        try b.op(.{ .rope = .{ .x = b.view(q), .heads = c.heads, .dim = dim, .rot = dim, .freqs = table } });
        try b.op(.{ .rope = .{ .x = b.view(k), .heads = kv_heads, .dim = dim, .rot = dim, .freqs = table } });
        try b.op(.{ .store = .{ .src = b.view(k), .dst = kc } });
        try b.op(.{ .store = .{ .src = b.view(v), .dst = vc } });

        try b.op(.{ .attention = .{
            .q = b.view(q),
            .keys = kc,
            .values = vc,
            .out = b.view(att),
            .heads = c.heads,
            .kv_heads = kv_heads,
            .dim = dim,
            .window = if (sliding) c.sliding_window else 0,
            .scale = 1.0,
        } });

        const w_o = try b.weight(try b.print("{s}layers.{d}.self_attn.o_proj.weight", .{ prefix, layer }), &.{ h, q_dim });
        try b.op(.{ .linear = .{ .w = w_o, .x = b.view(att), .out = b.view(branch_out) } });

        const post_attn_norm_w = try b.weight(try b.print("{s}layers.{d}.post_attention_layernorm.weight", .{ prefix, layer }), &.{h});
        const pre_ffn_norm_w = try b.weight(try b.print("{s}layers.{d}.pre_feedforward_layernorm.weight", .{ prefix, layer }), &.{h});
        try b.op(.{ .rmsnorm = .{ .x = b.view(branch_out), .weight = post_attn_norm_w, .out = b.view(norm_tmp), .group = h, .eps = c.eps } });
        try b.op(.{ .binary = .{ .kind = .add, .a = b.view(x), .b = b.view(norm_tmp), .out = b.view(x) } });
        try b.op(.{ .rmsnorm = .{ .x = b.view(x), .weight = pre_ffn_norm_w, .out = b.view(normed), .group = h, .eps = c.eps } });

        const w_gate = try b.weight(try b.print("{s}layers.{d}.mlp.gate_proj.weight", .{ prefix, layer }), &.{ inter, h });
        const w_up = try b.weight(try b.print("{s}layers.{d}.mlp.up_proj.weight", .{ prefix, layer }), &.{ inter, h });
        const w_down = try b.weight(try b.print("{s}layers.{d}.mlp.down_proj.weight", .{ prefix, layer }), &.{ h, inter });
        try b.op(.{ .linear = .{ .w = w_gate, .x = b.view(normed), .out = b.view(ffn_gate) } });
        try b.op(.{ .linear = .{ .w = w_up, .x = b.view(normed), .out = b.view(ffn_up) } });
        try b.op(.{ .binary = .{ .kind = .mul, .a = b.view(ffn_gate), .b = b.view(ffn_up), .a_act = .gelu_tanh, .out = b.view(ffn_act) } });
        try b.op(.{ .linear = .{ .w = w_down, .x = b.view(ffn_act), .out = b.view(mlp_out) } });

        const post_ffn_norm_w = try b.weight(try b.print("{s}layers.{d}.post_feedforward_layernorm.weight", .{ prefix, layer }), &.{h});
        try b.op(.{ .rmsnorm = .{ .x = b.view(mlp_out), .weight = post_ffn_norm_w, .out = b.view(norm_tmp), .group = h, .eps = c.eps } });
        try b.op(.{ .binary = .{ .kind = .add, .a = b.view(x), .b = b.view(norm_tmp), .out = b.view(x) } });

        const scalar_name = try b.print("{s}layers.{d}.layer_scalar", .{ prefix, layer });
        if (hasWeight(checkpoint, scalar_name)) {
            const scalar_w = try b.weight(scalar_name, &.{1});
            try b.op(.{ .row_transform = .{ .x = b.view(x), .out = b.view(x), .transform = .{ .weight_scale = scalar_w } } });
        }
        try b.trace(b.view(x));
    }

    const head = b.mark();
    const final_norm_w = try b.weight(try b.print("{s}norm.weight", .{prefix}), &.{h});
    try b.op(.{ .rmsnorm = .{ .x = b.view(x), .weight = final_norm_w, .out = b.view(normed), .group = h, .eps = c.eps } });
    const lm_head_w = if (c.tie_word_embeddings)
        embed_w
    else
        try b.weight(lmHeadWeightName(checkpoint, prefix), &.{ c.vocab, h });
    try b.op(.{ .linear = .{ .w = lm_head_w, .x = b.view(normed), .out = b.view(logits) } });
    if (c.final_logit_softcapping) |cap| {
        try b.op(.{ .row_transform = .{ .x = b.view(logits), .out = b.view(logits), .transform = .{ .softcap = cap } } });
    }
    try b.op(.{ .argmax = .{ .x = b.view(logits), .out = b.view(next) } });

    return b.finish(.{
        .model_type = c.model_type,
        .vocab = c.vocab,
        .eos = c.eos,
        .max_context = c.max_context,
    }, head, b.view(tokens), b.view(logits), b.view(next));
}

fn hasUnifiedPrefix(_: *const anyopaque, name: []const u8) bool {
    return std.mem.eql(u8, name, "language_model.model.embed_tokens.weight") or
        std.mem.endsWith(u8, name, ".layer_scalar");
}

fn noScalarPrefix(_: *const anyopaque, name: []const u8) bool {
    return std.mem.eql(u8, name, "model.embed_tokens.weight");
}

test "Gemma 4 parses 12B config wrapper, synthesizes layer_types, and fails closed on unsupported variants" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();

    const cfg_12b =
        \\{"model_type":"gemma4_unified","eos_token_id":[1,106,50],"tie_word_embeddings":true,
        \\"text_config":{"model_type":"gemma4_unified_text","hidden_size":3840,"intermediate_size":15360,
        \\"num_hidden_layers":48,"num_attention_heads":16,"num_key_value_heads":8,"num_global_key_value_heads":1,
        \\"head_dim":256,"global_head_dim":512,"attention_k_eq_v":true,"attention_bias":false,"rms_norm_eps":1e-6,
        \\"sliding_window":1024,"hidden_activation":"gelu_pytorch_tanh","final_logit_softcapping":30.0,
        \\"max_position_embeddings":262144,"vocab_size":262144,"eos_token_id":1,"tie_word_embeddings":true,
        \\"enable_moe_block":false,"hidden_size_per_layer_input":0,"num_kv_shared_layers":0,"use_double_wide_mlp":false,
        \\"rope_parameters":{"full_attention":{"partial_rotary_factor":0.25,"rope_theta":1000000.0,"rope_type":"proportional"},
        \\"sliding_attention":{"rope_theta":10000.0,"rope_type":"default"}}}}
    ;
    const parsed = try parse(a, cfg_12b);
    try std.testing.expectEqualStrings("gemma4_unified", parsed.model_type);
    try std.testing.expectEqual(@as(u32, 3840), parsed.hidden);
    try std.testing.expectEqual(@as(u32, 48), @as(u32, @intCast(parsed.layers.len)));
    try std.testing.expectEqual(LayerType.sliding_attention, parsed.layers[0]);
    try std.testing.expectEqual(LayerType.sliding_attention, parsed.layers[4]);
    try std.testing.expectEqual(LayerType.full_attention, parsed.layers[5]);
    try std.testing.expectEqual(LayerType.full_attention, parsed.layers[47]);
    try std.testing.expectEqualSlices(u32, &.{ 1, 106, 50 }, parsed.eos);
    try std.testing.expectEqual(@as(u32, 1), parsed.full_kv_heads);
    try std.testing.expectEqual(@as(u32, 8), parsed.sliding_kv_heads);
    try std.testing.expectEqual(@as(u32, 512), parsed.full_head_dim);
    try std.testing.expectEqual(@as(f32, 30.0), parsed.final_logit_softcapping.?);

    const freqs = try proportionalRopeFrequencies(a, 512, 1000000.0, 0.25, 1.0);
    try std.testing.expectEqual(@as(usize, 256), freqs.len);
    try std.testing.expectApproxEqRel(@as(f32, 1.0), freqs[0], 1e-6);
    try std.testing.expect(freqs[63] > 0.0);
    for (freqs[64..]) |f| try std.testing.expectEqual(@as(f32, 0.0), f);

    const bad_cases = [_][]const u8{
        \\{"model_type":"gemma4_text","hidden_size":64,"intermediate_size":128,"num_hidden_layers":2,"num_attention_heads":4,"num_key_value_heads":2,"head_dim":32,"vocab_size":64,"max_position_embeddings":128,"attention_bias":true}
        ,
        \\{"model_type":"gemma4_text","hidden_size":64,"intermediate_size":128,"num_hidden_layers":2,"num_attention_heads":4,"num_key_value_heads":2,"head_dim":32,"vocab_size":64,"max_position_embeddings":128,"enable_moe_block":true}
        ,
        \\{"model_type":"gemma4_text","hidden_size":64,"intermediate_size":128,"num_hidden_layers":2,"num_attention_heads":4,"num_key_value_heads":2,"head_dim":32,"vocab_size":64,"max_position_embeddings":128,"hidden_size_per_layer_input":32}
        ,
        \\{"model_type":"gemma4_text","hidden_size":64,"intermediate_size":128,"num_hidden_layers":2,"num_attention_heads":4,"num_key_value_heads":2,"head_dim":32,"vocab_size":64,"max_position_embeddings":128,"num_kv_shared_layers":1}
        ,
        \\{"model_type":"gemma4_text","hidden_size":64,"intermediate_size":128,"num_hidden_layers":2,"num_attention_heads":4,"num_key_value_heads":2,"head_dim":32,"vocab_size":64,"max_position_embeddings":128,"use_double_wide_mlp":true}
        ,
        \\{"model_type":"gemma4_text","hidden_size":64,"intermediate_size":128,"num_hidden_layers":2,"num_attention_heads":4,"num_key_value_heads":2,"head_dim":32,"vocab_size":64,"max_position_embeddings":128,"rope_traditional":true}
        ,
        \\{"model_type":"gemma4_text","hidden_size":64,"intermediate_size":128,"num_hidden_layers":2,"num_attention_heads":4,"num_key_value_heads":2,"head_dim":32,"vocab_size":64,"max_position_embeddings":128,"hidden_activation":"silu"}
        ,
        \\{"model_type":"gemma4_text","hidden_size":64,"intermediate_size":128,"num_hidden_layers":2,"num_attention_heads":4,"num_key_value_heads":2,"head_dim":32,"vocab_size":64,"max_position_embeddings":128,"rope_parameters":{"full_attention":{"rope_type":"yarn","rope_theta":1000000.0}}}
        ,
    };
    for (bad_cases) |bad| {
        try std.testing.expect(std.meta.isError(parse(a, bad)));
    }
}

test "Gemma 4 builds valid IR with sliding + full k_eq_v layers and matches planner fusions" {
    const cfg =
        \\{"model_type":"gemma4_unified","eos_token_id":[1,106],"tie_word_embeddings":true,
        \\"text_config":{"model_type":"gemma4_unified_text","hidden_size":128,"intermediate_size":256,
        \\"num_hidden_layers":2,"layer_types":["sliding_attention","full_attention"],
        \\"num_attention_heads":4,"num_key_value_heads":2,"num_global_key_value_heads":1,
        \\"head_dim":64,"global_head_dim":512,"attention_k_eq_v":true,"rms_norm_eps":1e-6,
        \\"sliding_window":8,"hidden_activation":"gelu_pytorch_tanh","final_logit_softcapping":30.0,
        \\"max_position_embeddings":512,"vocab_size":256,
        \\"rope_parameters":{"full_attention":{"partial_rotary_factor":0.25,"rope_theta":1000000.0,"rope_type":"proportional"},
        \\"sliding_attention":{"rope_theta":10000.0,"rope_type":"default"}}}}
    ;
    var prog = try build(std.testing.allocator, cfg, .{ .context = undefined, .has = hasUnifiedPrefix });
    defer prog.deinit();

    try std.testing.expectEqual(@as(usize, 3), prog.traces.len);
    var saw_v_norm = false;
    var saw_layer_scalar = false;
    var saw_softcap = false;
    var saw_gelu_tanh = false;
    for (prog.ops) |op| switch (op) {
        .rmsnorm => |n| if (n.weight == null) {
            saw_v_norm = true;
        },
        .row_transform => |rt| switch (rt.transform) {
            .weight_scale => saw_layer_scalar = true,
            .softcap => saw_softcap = true,
            else => {},
        },
        .binary => |b| if (b.a_act == .gelu_tanh) {
            saw_gelu_tanh = true;
        },
        else => {},
    };
    try std.testing.expect(saw_v_norm);
    try std.testing.expect(saw_layer_scalar);
    try std.testing.expect(saw_softcap);
    try std.testing.expect(saw_gelu_tanh);

    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const info = try a.alloc(plan.WeightInfo, prog.weights.len);
    for (prog.weights, info) |w, *wi| {
        wi.* = if (w.shape.len == 2) .{ .q4 = true, .group = 64, .buffer = 1 } else .{ .q4 = false, .dense = true, .buffer = 1 };
    }
    const steps = try plan.build(a, &prog, .{ .projections = true, .glu = true, .norms = true, .attn_prep = true }, info);
    var projs: usize = 0;
    var glus: usize = 0;
    var norms: usize = 0;
    var preps: usize = 0;
    for (steps) |s| switch (s) {
        .projections => projs += 1,
        .glu => glus += 1,
        .norms => norms += 1,
        .attn_prep => preps += 1,
        else => {},
    };
    try std.testing.expectEqual(@as(usize, 2), projs);
    try std.testing.expectEqual(@as(usize, 2), glus);
    try std.testing.expectEqual(@as(usize, 2), norms);
    try std.testing.expectEqual(@as(usize, 2), preps);

    // Also build without layer_scalar to verify untied / scalar-optional path.
    var prog_no_scalar = try build(std.testing.allocator, cfg, .{ .context = undefined, .has = noScalarPrefix });
    defer prog_no_scalar.deinit();
    try std.testing.expect(prog_no_scalar.ops.len + 2 == prog.ops.len);
}
