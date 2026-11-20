//! LFM3 front end: config.json -> Program. The only file that knows LFM3.
//! Layer = sandwich-normed operator (gated short conv or gated GQA attention)
//! plus a sandwich-normed SwiGLU MLP; tied, embedding-normed output head.
const std = @import("std");
const ir = @import("ir.zig");

pub const Checkpoint = struct {
    context: *const anyopaque,
    has: *const fn (*const anyopaque, []const u8) bool,
};

const LayerType = enum { conv, full_attention, sliding_attention };

pub const Config = struct {
    hidden: u32,
    layers: []LayerType,
    heads: u32,
    kv_heads: u32,
    head_dim: u32,
    intermediate: u32,
    conv_taps: u32,
    eps: f32,
    vocab: u32,
    eos: []u32,
    window: u32,
    max_context: u32,
    full_theta: f64,
    full_partial: f64,
    sliding_theta: f64,
    sliding_partial: f64,
};

fn int(v: ?std.json.Value) !u32 {
    const value = v orelse return error.MissingConfigField;
    return switch (value) {
        .integer => |n| std.math.cast(u32, n) orelse error.InvalidConfigInteger,
        else => error.InvalidConfigInteger,
    };
}
fn float(v: ?std.json.Value) !f64 {
    const value = v orelse return error.MissingConfigField;
    return switch (value) {
        .integer => |n| @floatFromInt(n),
        .float => |f| f,
        else => error.InvalidConfigNumber,
    };
}
fn boolean(v: ?std.json.Value, default: bool) !bool {
    const value = v orelse return default;
    return switch (value) {
        .bool => |b| b,
        else => error.InvalidConfigBoolean,
    };
}

/// Fails closed on features this front end does not implement (conv bias,
/// untied heads, non-sandwich norms, RoPE scaling types).
pub fn parse(a: std.mem.Allocator, bytes: []const u8) !Config {
    const parsed = try std.json.parseFromSliceLeaky(std.json.Value, a, bytes, .{});
    if (parsed != .object) return error.InvalidConfig;
    const o = parsed.object;
    const model_type = o.get("model_type") orelse return error.MissingConfigField;
    if (model_type != .string or !std.mem.eql(u8, model_type.string, "lfm3")) return error.NotLfm3;
    if (try boolean(o.get("conv_bias"), false)) return error.UnsupportedConvBias;
    if (!try boolean(o.get("tie_word_embeddings"), true)) return error.UnsupportedUntiedHead;
    if (!try boolean(o.get("use_sandwich_norm"), true)) return error.UnsupportedNormLayout;
    const types = o.get("layer_types") orelse return error.MissingConfigField;
    if (types != .array) return error.InvalidConfig;
    const n_layers = try int(o.get("num_hidden_layers"));
    if (types.array.items.len != n_layers or n_layers == 0) return error.InvalidLayerTypes;
    const layers = try a.alloc(LayerType, n_layers);
    for (types.array.items, layers) |t, *l| {
        if (t != .string) return error.InvalidLayerTypes;
        l.* = std.meta.stringToEnum(LayerType, t.string) orelse return error.UnsupportedLayerType;
    }
    var eos: std.ArrayList(u32) = .empty;
    if (o.get("eos_token_id")) |e| switch (e) {
        .integer => try eos.append(a, try int(e)),
        .array => |arr| for (arr.items) |item| try eos.append(a, try int(item)),
        else => return error.InvalidConfig,
    };
    const rope = o.get("rope_parameters") orelse return error.MissingConfigField;
    if (rope != .object) return error.InvalidConfig;
    var thetas: [2]f64 = .{ 0, 0 };
    var partials: [2]f64 = .{ 1, 1 };
    for ([_][]const u8{ "full_attention", "sliding_attention" }, 0..) |key, i| {
        const r = rope.object.get(key) orelse continue;
        if (r != .object) return error.InvalidConfig;
        if (r.object.get("rope_type")) |t| if (t != .string or !std.mem.eql(u8, t.string, "default")) return error.UnsupportedRopeType;
        thetas[i] = try float(r.object.get("rope_theta"));
        if (r.object.get("partial_rotary_factor")) |f| partials[i] = try float(f);
    }
    const c: Config = .{
        .hidden = try int(o.get("hidden_size")),
        .layers = layers,
        .heads = try int(o.get("num_attention_heads")),
        .kv_heads = try int(o.get("num_key_value_heads")),
        .head_dim = try int(o.get("head_dim")),
        .intermediate = try int(o.get("intermediate_size")),
        .conv_taps = try int(o.get("conv_L_cache")),
        .eps = @floatCast(try float(o.get("norm_eps"))),
        .vocab = try int(o.get("vocab_size")),
        .eos = eos.items,
        .window = if (o.get("sliding_window")) |w| try int(w) else 0,
        .max_context = try int(o.get("max_position_embeddings")),
        .full_theta = thetas[0],
        .full_partial = partials[0],
        .sliding_theta = thetas[1],
        .sliding_partial = partials[1],
    };
    for (c.layers) |l| {
        if (l == .full_attention and !(c.full_theta > 0)) return error.MissingRope;
        if (l == .sliding_attention and (!(c.sliding_theta > 0) or c.window == 0)) return error.MissingRope;
    }
    if (c.hidden == 0 or c.heads == 0 or c.kv_heads == 0 or c.heads % c.kv_heads != 0 or c.head_dim % 2 != 0 or c.conv_taps < 2 or c.vocab == 0) return error.InvalidConfig;
    return c;
}

/// HF zero-tail partial RoPE: the first int(factor * dim) / 2 pairs use the
/// full-dimension progression theta^(-2i/dim); the rest have frequency 0.
/// Float32, as torch computes it: inv = 1 / f32(theta ** f32(f32(2i) / dim)).
pub fn ropeFrequencies(a: std.mem.Allocator, dim: u32, theta: f64, partial: f64) ![]f32 {
    const half = dim / 2;
    const rotated: u32 = @intFromFloat(@floor(partial * @as(f64, @floatFromInt(dim))));
    const angles = @min(half, rotated / 2);
    const out = try a.alloc(f32, half);
    for (out, 0..) |*f, i| {
        if (i >= angles) {
            f.* = 0;
            continue;
        }
        const exponent: f32 = @as(f32, @floatFromInt(2 * i)) / @as(f32, @floatFromInt(dim));
        const power: f32 = @floatCast(std.math.pow(f64, theta, @as(f64, exponent)));
        f.* = 1.0 / power;
    }
    return out;
}

pub const Options = struct {
    /// Number of positions a sliding query sees, counting itself (HF mask convention).
    sliding_visible: ?u32 = null,
    /// Round the embedding lookup to bf16, as an MLX dequantize into bf16 does.
    embed_round: ir.Round = .none,
};

pub fn build(gpa: std.mem.Allocator, config_bytes: []const u8, checkpoint: Checkpoint, options: Options) !ir.Program {
    var b = ir.Builder.init(gpa);
    errdefer b.deinit();
    const a = b.allocator();
    const c = try parse(a, config_bytes);
    const prefix = for ([_][]const u8{ "language_model.model.", "model.", "" }) |p| {
        const probe = try b.print("{s}embed_tokens.weight", .{p});
        if (checkpoint.has(checkpoint.context, probe)) break p;
    } else return error.MissingEmbedding;
    const H = c.hidden;
    const qw = c.heads * c.head_dim;
    const kvw = c.kv_heads * c.head_dim;

    const tokens = try b.state(.{ .name = "tokens", .kind = .tokens, .width = 1 });
    _ = try b.state(.{ .name = "position", .kind = .position, .width = 1 });
    const x = try b.state(.{ .name = "x", .kind = .scratch, .width = H });
    const n = try b.state(.{ .name = "norm", .kind = .scratch, .width = H });
    const t = try b.state(.{ .name = "post", .kind = .scratch, .width = H });
    const o = try b.state(.{ .name = "operator", .kind = .scratch, .width = H });
    const bcx = try b.state(.{ .name = "bcx", .kind = .scratch, .width = 3 * H });
    const u = try b.state(.{ .name = "bx", .kind = .scratch, .width = H });
    const cv = try b.state(.{ .name = "conv", .kind = .scratch, .width = H });
    const y = try b.state(.{ .name = "gated", .kind = .scratch, .width = H });
    const q = try b.state(.{ .name = "q", .kind = .scratch, .width = qw });
    const k = try b.state(.{ .name = "k", .kind = .scratch, .width = kvw });
    const v = try b.state(.{ .name = "v", .kind = .scratch, .width = kvw });
    const g = try b.state(.{ .name = "gate", .kind = .scratch, .width = c.heads });
    const att = try b.state(.{ .name = "attention", .kind = .scratch, .width = qw });
    const ag = try b.state(.{ .name = "attention_gated", .kind = .scratch, .width = qw });
    const f1 = try b.state(.{ .name = "w1", .kind = .scratch, .width = c.intermediate });
    const f3 = try b.state(.{ .name = "w3", .kind = .scratch, .width = c.intermediate });
    const fa = try b.state(.{ .name = "swiglu", .kind = .scratch, .width = c.intermediate });
    const f = try b.state(.{ .name = "ffn", .kind = .scratch, .width = H });
    const logits = try b.state(.{ .name = "logits", .kind = .scratch, .width = c.vocab });
    const next = try b.state(.{ .name = "next", .kind = .scratch, .width = 1 });

    const freqs_full = try b.table(try ropeFrequencies(a, c.head_dim, c.full_theta, c.full_partial));
    const freqs_sliding = try b.table(try ropeFrequencies(a, c.head_dim, c.sliding_theta, c.sliding_partial));
    const visible = options.sliding_visible orelse c.window;

    const embed = try b.weight(try b.print("{s}embed_tokens.weight", .{prefix}), &.{ c.vocab, H });
    try b.op(.{ .embed = .{ .table = embed, .tokens = b.view(tokens), .out = b.view(x), .round = options.embed_round } });
    try b.trace(b.view(x));
    const scale: f32 = @floatCast(1.0 / @sqrt(@as(f64, @floatFromInt(c.head_dim))));
    for (c.layers, 0..) |layer, i| {
        const l = try b.print("{s}layers.{d}.", .{ prefix, i });
        const norm = struct {
            fn get(bb: *ir.Builder, pre: []const u8, name: []const u8, size: u32) !ir.WeightId {
                return bb.weight(try bb.print("{s}{s}.weight", .{ pre, name }), &.{size});
            }
        }.get;
        const lin = struct {
            fn get(bb: *ir.Builder, pre: []const u8, name: []const u8, rows: u32, cols: u32) !ir.WeightId {
                return bb.weight(try bb.print("{s}{s}.weight", .{ pre, name }), &.{ rows, cols });
            }
        }.get;
        try b.op(.{ .rmsnorm = .{ .x = b.view(x), .weight = try norm(&b, l, "operator_norm", H), .out = b.view(n), .group = H, .eps = c.eps } });
        switch (layer) {
            .conv => {
                const ring = try b.state(.{ .name = try b.print("conv{d}", .{i}), .kind = .ring, .width = H, .horizon = c.conv_taps - 1 });
                try b.op(.{ .linear = .{ .w = try lin(&b, l, "conv.in_proj", 3 * H, H), .x = b.view(n), .out = b.view(bcx) } });
                // in_proj output is split in order into B, C, X.
                try b.op(.{ .binary = .{ .kind = .mul, .a = .{ .state = bcx, .col = 0, .width = H }, .b = .{ .state = bcx, .col = 2 * H, .width = H }, .out = b.view(u) } });
                try b.op(.{ .store = .{ .src = b.view(u), .dst = ring } });
                const cw = try b.weight(try b.print("{s}conv.conv.weight", .{l}), &.{ H, c.conv_taps, 1 });
                try b.op(.{ .conv = .{ .src = ring, .weight = cw, .out = b.view(cv), .taps = c.conv_taps } });
                try b.op(.{ .binary = .{ .kind = .mul, .a = .{ .state = bcx, .col = H, .width = H }, .b = b.view(cv), .out = b.view(y) } });
                try b.op(.{ .linear = .{ .w = try lin(&b, l, "conv.out_proj", H, H), .x = b.view(y), .out = b.view(o) } });
            },
            .full_attention, .sliding_attention => {
                const sliding = layer == .sliding_attention;
                const kind: ir.StateKind = if (sliding) .ring else .sequence;
                const horizon: u32 = if (sliding) visible - 1 else 0;
                const kc = try b.state(.{ .name = try b.print("keys{d}", .{i}), .kind = kind, .width = kvw, .horizon = @max(horizon, 1) });
                const vc = try b.state(.{ .name = try b.print("values{d}", .{i}), .kind = kind, .width = kvw, .horizon = @max(horizon, 1) });
                try b.op(.{ .linear = .{ .w = try lin(&b, l, "self_attn.q_proj", qw, H), .x = b.view(n), .out = b.view(q) } });
                try b.op(.{ .linear = .{ .w = try lin(&b, l, "self_attn.k_proj", kvw, H), .x = b.view(n), .out = b.view(k) } });
                try b.op(.{ .linear = .{ .w = try lin(&b, l, "self_attn.v_proj", kvw, H), .x = b.view(n), .out = b.view(v) } });
                try b.op(.{ .linear = .{ .w = try lin(&b, l, "self_attn.gate_proj", c.heads, H), .x = b.view(n), .out = b.view(g) } });
                try b.op(.{ .rmsnorm = .{ .x = b.view(q), .weight = try norm(&b, l, "self_attn.q_norm", c.head_dim), .out = b.view(q), .group = c.head_dim, .eps = c.eps } });
                try b.op(.{ .rmsnorm = .{ .x = b.view(k), .weight = try norm(&b, l, "self_attn.k_norm", c.head_dim), .out = b.view(k), .group = c.head_dim, .eps = c.eps } });
                const table = if (sliding) freqs_sliding else freqs_full;
                try b.op(.{ .rope = .{ .x = b.view(q), .heads = c.heads, .dim = c.head_dim, .rot = c.head_dim, .freqs = table } });
                try b.op(.{ .rope = .{ .x = b.view(k), .heads = c.kv_heads, .dim = c.head_dim, .rot = c.head_dim, .freqs = table } });
                try b.op(.{ .store = .{ .src = b.view(k), .dst = kc } });
                try b.op(.{ .store = .{ .src = b.view(v), .dst = vc } });
                try b.op(.{ .attention = .{ .q = b.view(q), .keys = kc, .values = vc, .out = b.view(att), .heads = c.heads, .kv_heads = c.kv_heads, .dim = c.head_dim, .window = if (sliding) visible else 0, .scale = scale } });
                try b.op(.{ .binary = .{ .kind = .mul, .a = b.view(att), .b = b.view(g), .out = b.view(ag), .b_act = .sigmoid, .b_every = c.head_dim } });
                try b.op(.{ .linear = .{ .w = try lin(&b, l, "self_attn.o_proj", H, qw), .x = b.view(ag), .out = b.view(o) } });
            },
        }
        try b.op(.{ .rmsnorm = .{ .x = b.view(o), .weight = try norm(&b, l, "post_operator_norm", H), .out = b.view(t), .group = H, .eps = c.eps } });
        try b.op(.{ .binary = .{ .kind = .add, .a = b.view(x), .b = b.view(t), .out = b.view(x) } });
        try b.op(.{ .rmsnorm = .{ .x = b.view(x), .weight = try norm(&b, l, "ffn_norm", H), .out = b.view(n), .group = H, .eps = c.eps } });
        try b.op(.{ .linear = .{ .w = try lin(&b, l, "feed_forward.w1", c.intermediate, H), .x = b.view(n), .out = b.view(f1) } });
        try b.op(.{ .linear = .{ .w = try lin(&b, l, "feed_forward.w3", c.intermediate, H), .x = b.view(n), .out = b.view(f3) } });
        try b.op(.{ .binary = .{ .kind = .mul, .a = b.view(f1), .b = b.view(f3), .out = b.view(fa), .a_act = .silu } });
        try b.op(.{ .linear = .{ .w = try lin(&b, l, "feed_forward.w2", H, c.intermediate), .x = b.view(fa), .out = b.view(f) } });
        try b.op(.{ .rmsnorm = .{ .x = b.view(f), .weight = try norm(&b, l, "post_ffn_norm", H), .out = b.view(t), .group = H, .eps = c.eps } });
        try b.op(.{ .binary = .{ .kind = .add, .a = b.view(x), .b = b.view(t), .out = b.view(x) } });
        try b.trace(b.view(x));
    }
    const head = b.mark();
    try b.op(.{ .rmsnorm = .{ .x = b.view(x), .weight = try b.weight(try b.print("{s}embedding_norm.weight", .{prefix}), &.{H}), .out = b.view(n), .group = H, .eps = c.eps } });
    try b.op(.{ .linear = .{ .w = embed, .x = b.view(n), .out = b.view(logits) } });
    try b.op(.{ .argmax = .{ .x = b.view(logits), .out = b.view(next) } });
    return b.finish(.{ .model_type = "lfm3", .vocab = c.vocab, .eos = c.eos, .max_context = c.max_context }, head, b.view(tokens), b.view(logits), b.view(next));
}

const test_config =
    \\{"model_type":"lfm3","hidden_size":64,"num_hidden_layers":3,"layer_types":["conv","full_attention","sliding_attention"],
    \\"num_attention_heads":4,"num_key_value_heads":2,"head_dim":32,"intermediate_size":96,"conv_L_cache":3,"norm_eps":1e-5,
    \\"vocab_size":40,"eos_token_id":[7],"sliding_window":4,"max_position_embeddings":128,"tie_word_embeddings":true,
    \\"rope_parameters":{"full_attention":{"partial_rotary_factor":0.5,"rope_theta":1000000.0,"rope_type":"default"},
    \\"sliding_attention":{"rope_theta":10000,"rope_type":"default"}}}
;
fn alwaysModelPrefix(_: *const anyopaque, name: []const u8) bool {
    return std.mem.startsWith(u8, name, "model.");
}

test "LFM3 front end emits states, shapes and op counts from config alone" {
    var p = try build(std.testing.allocator, test_config, .{ .context = undefined, .has = alwaysModelPrefix }, .{});
    defer p.deinit();
    // embed + per layer (input norm + conv 6 / attention 13 + shared 9) + head 3.
    try std.testing.expectEqual(@as(usize, 1 + (1 + 6 + 9) + (1 + 13 + 9) * 2 + 3), p.ops.len);
    try std.testing.expectEqual(@as(usize, 4), p.traces.len);
    var rings: u32 = 0;
    var sequences: u32 = 0;
    for (p.states) |s| switch (s.kind) {
        .ring => rings += 1,
        .sequence => sequences += 1,
        else => {},
    };
    try std.testing.expectEqual(@as(u32, 3), rings); // conv window + sliding K and V
    try std.testing.expectEqual(@as(u32, 2), sequences);
    for (p.states) |s| if (std.mem.eql(u8, s.name, "conv0")) try std.testing.expectEqual(@as(u32, 2), s.horizon);
    for (p.states) |s| if (std.mem.eql(u8, s.name, "keys2")) try std.testing.expectEqual(@as(u32, 3), s.horizon);
    try std.testing.expectEqualStrings("model.embed_tokens.weight", p.weights[0].name);
    try std.testing.expectEqualSlices(u32, &.{7}, p.meta.eos);
}

test "zero-tail partial RoPE table" {
    const f = try ropeFrequencies(std.testing.allocator, 16, 1e6, 0.5);
    defer std.testing.allocator.free(f);
    try std.testing.expectEqual(@as(usize, 8), f.len);
    try std.testing.expectEqual(@as(f32, 1), f[0]);
    try std.testing.expect(f[3] > 0 and f[3] < f[2]);
    for (f[4..]) |value| try std.testing.expectEqual(@as(f32, 0), value);
}

test "LFM3 config fails closed on unsupported features" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    try std.testing.expectError(error.NotLfm3, parse(a, "{\"model_type\":\"llama\"}"));
    try std.testing.expectError(error.UnsupportedConvBias, parse(a, "{\"model_type\":\"lfm3\",\"conv_bias\":true}"));
    try std.testing.expectError(error.UnsupportedUntiedHead, parse(a, "{\"model_type\":\"lfm3\",\"tie_word_embeddings\":false}"));
}

// ---- chat template (cli) ----
// The text-only path of the checkpoint's chat_template.jinja, written directly
// (no Jinja interpreter); tools/nano_tokenizer_check.py template compares it
// with the real template. The system message is trimmed and written only if
// non-empty; user (and later system) content is trimmed; an assistant turn
// before the last user turn becomes "<think></think>" + its text after the last
// "</think>" (trimmed, if there was one); a later one keeps its text and gets
// "<think></think>" unless it contains "</think>". The generation prompt ends
// in "<think>" with thinking on, "<think></think>" with it off.
const chat = @import("chat.zig");

pub const chat_template: chat.Template = .{
    .render = renderChat,
    .next_turn = nextTurn,
    .think_open = "<think>",
    .think_close = "</think>",
};

fn renderChat(a: std.mem.Allocator, messages: []const chat.Message, options: chat.Options) ![]u8 {
    var out: std.ArrayList(u8) = .empty;
    try out.appendSlice(a, "<|startoftext|>");
    var msgs = messages;
    if (msgs.len > 0 and msgs[0].role == .system) {
        const system = chat.trim(msgs[0].content);
        if (system.len > 0) try out.print(a, "<|im_start|>system\n{s}<|im_end|>\n", .{system});
        msgs = msgs[1..];
    }
    var last_user: ?usize = null;
    for (msgs, 0..) |m, i| if (m.role == .user) {
        last_user = i;
    };
    for (msgs, 0..) |m, i| {
        try out.print(a, "<|im_start|>{s}\n", .{@tagName(m.role)});
        if (m.role != .assistant) {
            try out.print(a, "{s}<|im_end|>\n", .{chat.trim(m.content)});
            continue;
        }
        const keep_thinking = last_user == null or i > last_user.?;
        if (keep_thinking) {
            if (std.mem.indexOf(u8, m.content, "</think>") == null) try out.appendSlice(a, "<think></think>");
            try out.appendSlice(a, m.content);
        } else {
            var content = m.content;
            if (std.mem.lastIndexOf(u8, content, "</think>")) |p| content = chat.trim(content[p + "</think>".len ..]);
            try out.print(a, "<think></think>{s}", .{content});
        }
        try out.appendSlice(a, "<|im_end|>\n");
    }
    if (options.generate) try out.print(a, "<|im_start|>assistant\n{s}", .{if (options.thinking) "<think>" else "<think></think>"});
    return out.items;
}

fn nextTurn(a: std.mem.Allocator, closed: bool, user: []const u8, options: chat.Options) ![]u8 {
    return std.fmt.allocPrint(a, "{s}\n<|im_start|>user\n{s}<|im_end|>\n<|im_start|>assistant\n{s}", .{
        if (closed) "" else "<|im_end|>", chat.trim(user), if (options.thinking) "<think>" else "<think></think>",
    });
}

test "LFM3 chat template: one turn, system, history, continuation" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const one = try renderChat(a, &.{.{ .role = .user, .content = " hi " }}, .{});
    try std.testing.expectEqualStrings("<|startoftext|><|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n<think></think>", one);
    const sys = try renderChat(a, &.{ .{ .role = .system, .content = "  S\n" }, .{ .role = .user, .content = "u" } }, .{ .thinking = true });
    try std.testing.expectEqualStrings("<|startoftext|><|im_start|>system\nS<|im_end|>\n<|im_start|>user\nu<|im_end|>\n<|im_start|>assistant\n<think>", sys);
    // A reply in state plus the next turn's text is the template's render of the longer conversation.
    const whole = try renderChat(a, &.{ .{ .role = .user, .content = " hi " }, .{ .role = .assistant, .content = "Hello." }, .{ .role = .user, .content = "more" } }, .{});
    const next = try nextTurn(a, true, "more", .{});
    try std.testing.expectEqualStrings(whole, try std.mem.concat(a, u8, &.{ one, "Hello.<|im_end|>", next }));
    const cut = try nextTurn(a, false, "more", .{});
    try std.testing.expectEqualStrings(whole, try std.mem.concat(a, u8, &.{ one, "Hello.", cut }));
    const hidden = try renderChat(a, &.{ .{ .role = .user, .content = "q" }, .{ .role = .assistant, .content = "<think>x</think>  y " }, .{ .role = .user, .content = "r" } }, .{ .generate = false });
    try std.testing.expect(std.mem.indexOf(u8, hidden, "<|im_start|>assistant\n<think></think>y<|im_end|>") != null);
}
