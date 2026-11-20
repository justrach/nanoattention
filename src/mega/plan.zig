//! Schedules: the op list grouped into launch steps. The reference schedule is
//! one step per op. Fusion knobs let a peephole pass replace short op
//! sequences by one fused step when the pattern, the weight formats and the
//! liveness of the intermediates allow it. Patterns are structural (op kinds,
//! views, activations); nothing here knows a model.
const std = @import("std");
const ir = @import("ir.zig");

pub const Fusion = struct {
    /// Consecutive linears that share one input (and weight buffer) in one launch.
    projections: bool = false,
    /// linear, linear, binary(act_a(a) (+|*) act_b(b)) with dead a, b: one launch.
    glu: bool = false,
    /// rmsnorm(y -> t), add(r, t -> r), rmsnorm(r -> out) with dead t: one launch.
    norms: bool = false,
    /// mul(a, b -> u), store(u -> ring), conv(ring -> v), mul(c, v -> y): one launch.
    conv: bool = false,
    /// attention(-> a), mul(a, act(g)) with dead a: one launch.
    gate: bool = false,
    // ---- decode search ----
    /// rmsnorm(q), rmsnorm(k) per head in place, rope(q), rope(k), store(k), store(v): one launch.
    attn_prep: bool = false,

    pub fn any(self: Fusion) bool {
        return self.projections or self.glu or self.norms or self.conv or self.gate or self.attn_prep;
    }
};

pub const Step = union(enum) {
    op: u32,
    projections: struct { first: u32, count: u32 },
    glu: u32,
    norms: u32,
    conv: u32,
    gate: u32,
    attn_prep: u32,

    pub fn first(self: Step) u32 {
        return switch (self) {
            .op, .glu, .norms, .conv, .gate, .attn_prep => |i| i,
            .projections => |m| m.first,
        };
    }
    pub fn len(self: Step) u32 {
        return switch (self) {
            .op => 1,
            .projections => |m| m.count,
            .glu, .norms => 3,
            .conv => 4,
            .gate => 2,
            .attn_prep => 6,
        };
    }
};

/// What the planner needs to know about a bound weight.
pub const WeightInfo = struct {
    /// Served by the affine 4-bit reference kernel (mk_linear_q4_*).
    q4: bool,
    /// Identity of the Metal buffer holding weight, scales and biases (0 = several).
    buffer: usize,
    /// Affine group size (0 for dense weights).
    group: u32 = 0,
};

fn overlaps(a: ir.View, b: ir.View) bool {
    return a.state == b.state and a.col < b.col + b.width and b.col < a.col + a.width;
}
fn covers(a: ir.View, b: ir.View) bool {
    return a.state == b.state and a.col <= b.col and a.col + a.width >= b.col + b.width;
}

/// Views an op reads and writes (scratch/tokens only; caches are persistent).
fn access(op: ir.Op, reads: *[3]?ir.View, writes: *?ir.View) void {
    reads.* = .{ null, null, null };
    writes.* = null;
    switch (op) {
        .embed => |e| {
            reads[0] = e.tokens;
            writes.* = e.out;
        },
        .rmsnorm => |n| {
            reads[0] = n.x;
            writes.* = n.out;
        },
        .linear => |l| {
            reads[0] = l.x;
            writes.* = l.out;
        },
        .binary => |b| {
            reads[0] = b.a;
            reads[1] = b.b;
            writes.* = b.out;
        },
        .store => |s| reads[0] = s.src,
        .conv => |c| writes.* = c.out,
        .rope => |r| {
            reads[0] = r.x;
            writes.* = r.x;
        },
        .attention => |a| {
            reads[0] = a.q;
            writes.* = a.out;
        },
        .argmax => |m| {
            reads[0] = m.x;
            writes.* = m.out;
        },
    }
}

/// True when the value op `writer` stores in `v` is read by no op after
/// `last_reader` (it is overwritten first, or never read), and is not a host
/// output or a trace.
pub fn deadAfter(p: *const ir.Program, v: ir.View, last_reader: u32) bool {
    if (overlaps(v, p.logits) or overlaps(v, p.next)) return false;
    for (p.traces) |t| if (t.view.state == v.state) return false;
    var i = last_reader + 1;
    while (i < p.ops.len) : (i += 1) {
        var reads: [3]?ir.View = undefined;
        var writes: ?ir.View = null;
        access(p.ops[i], &reads, &writes);
        for (reads) |r| if (r) |rv| if (overlaps(rv, v)) return false;
        if (writes) |w| if (covers(w, v)) return true;
    }
    return true;
}

fn sameSide(p: *const ir.Program, first: u32, count: u32) bool {
    if (first + count > p.ops.len) return false;
    return (first >= p.head_start) == (first + count - 1 >= p.head_start);
}
fn sameView(a: ir.View, b: ir.View) bool {
    return a.state == b.state and a.col == b.col and a.width == b.width;
}

fn matchNorms(p: *const ir.Program, info: []const WeightInfo, i: u32) bool {
    if (!sameSide(p, i, 3)) return false;
    const n1 = switch (p.ops[i]) {
        .rmsnorm => |n| n,
        else => return false,
    };
    const add = switch (p.ops[i + 1]) {
        .binary => |b| b,
        else => return false,
    };
    const n2 = switch (p.ops[i + 2]) {
        .rmsnorm => |n| n,
        else => return false,
    };
    if (n1.weight == null or n2.weight == null) return false;
    if (info[n1.weight.?].buffer == 0 or info[n1.weight.?].buffer != info[n2.weight.?].buffer) return false;
    if (n1.group != n1.x.width or n2.group != n2.x.width or n1.group != n2.group or n1.eps != n2.eps) return false;
    if (add.kind != .add or add.a_act != .identity or add.b_act != .identity or add.b_every != 1) return false;
    const t = n1.out;
    const r = if (sameView(add.b, t)) add.a else if (sameView(add.a, t)) add.b else return false;
    if (!sameView(add.out, r) or !sameView(n2.x, r) or overlaps(n2.out, r) or overlaps(n1.x, r) or overlaps(t, r) or overlaps(n2.out, n1.x)) return false;
    return deadAfter(p, t, i + 1);
}

fn linearOf(op: ir.Op) ?@FieldType(ir.Op, "linear") {
    return switch (op) {
        .linear => |l| l,
        else => null,
    };
}

fn matchGlu(p: *const ir.Program, info: []const WeightInfo, i: u32) bool {
    if (!sameSide(p, i, 3)) return false;
    const l0 = linearOf(p.ops[i]) orelse return false;
    const l1 = linearOf(p.ops[i + 1]) orelse return false;
    const b = switch (p.ops[i + 2]) {
        .binary => |b| b,
        else => return false,
    };
    if (!info[l0.w].q4 or !info[l1.w].q4 or info[l0.w].buffer == 0 or info[l0.w].buffer != info[l1.w].buffer or info[l0.w].group != info[l1.w].group) return false;
    if (!sameView(l0.x, l1.x) or l0.out.width != l1.out.width or overlaps(l0.out, l1.out) or overlaps(l0.out, l1.x)) return false;
    if (!sameView(b.a, l0.out) or !sameView(b.b, l1.out) or b.b_every != 1 or overlaps(b.out, l0.x)) return false;
    return deadAfter(p, l0.out, i + 2) and deadAfter(p, l1.out, i + 2);
}

fn matchProjections(p: *const ir.Program, info: []const WeightInfo, i: u32) u32 {
    const l0 = linearOf(p.ops[i]) orelse return 0;
    if (!info[l0.w].q4 or info[l0.w].buffer == 0) return 0;
    var n: u32 = 1;
    while (n < 4 and i + n < p.ops.len and sameSide(p, i, n + 1)) : (n += 1) {
        const l = linearOf(p.ops[i + n]) orelse break;
        if (!sameView(l.x, l0.x) or !info[l.w].q4 or info[l.w].buffer != info[l0.w].buffer or info[l.w].group != info[l0.w].group) break;
        // Outputs must not feed a later member or each other.
        var clash = overlaps(l.out, l0.x);
        for (0..n) |k| clash = clash or overlaps(linearOf(p.ops[i + k]).?.out, l.out);
        if (clash) break;
    }
    return if (n >= 2) n else 0;
}

fn matchConv(p: *const ir.Program, i: u32) bool {
    if (!sameSide(p, i, 4)) return false;
    const m0 = switch (p.ops[i]) {
        .binary => |b| b,
        else => return false,
    };
    const st = switch (p.ops[i + 1]) {
        .store => |s| s,
        else => return false,
    };
    const cv = switch (p.ops[i + 2]) {
        .conv => |c| c,
        else => return false,
    };
    const m1 = switch (p.ops[i + 3]) {
        .binary => |b| b,
        else => return false,
    };
    if (m0.kind != .mul or m0.b_every != 1 or m1.kind != .mul or m1.b_every != 1) return false;
    if (!sameView(st.src, m0.out) or st.dst != cv.src or p.states[cv.src].kind != .ring) return false;
    const conv_left = sameView(m1.a, cv.out);
    if (!conv_left and !sameView(m1.b, cv.out)) return false;
    const other = if (conv_left) m1.b else m1.a;
    // The per-channel thread reads a, b, other and writes out at its own channel only.
    if (overlaps(m1.out, m0.a) or overlaps(m1.out, m0.b) or overlaps(m1.out, other) and !sameView(m1.out, other)) return false;
    if (m0.out.width != cv.out.width or other.width != cv.out.width) return false;
    return deadAfter(p, m0.out, i + 1) and deadAfter(p, cv.out, i + 3);
}

fn matchGate(p: *const ir.Program, i: u32) bool {
    if (!sameSide(p, i, 2)) return false;
    const a = switch (p.ops[i]) {
        .attention => |a| a,
        else => return false,
    };
    const b = switch (p.ops[i + 1]) {
        .binary => |b| b,
        else => return false,
    };
    if (b.kind != .mul or b.a_act != .identity or !sameView(b.a, a.out) or b.b_every != a.dim or b.b.width != a.heads) return false;
    if (overlaps(b.out, a.q) or overlaps(b.b, a.out)) return false;
    return deadAfter(p, a.out, i + 1);
}

// ---- decode search ----

/// rmsnorm(q in place), rmsnorm(k in place) per head of `dim`, rope(q), rope(k)
/// with one table, store(k), store(v) into caches of one shape; q, k, v disjoint.
fn matchAttnPrep(p: *const ir.Program, i: u32) bool {
    if (!sameSide(p, i, 6)) return false;
    const n1 = switch (p.ops[i]) {
        .rmsnorm => |n| n,
        else => return false,
    };
    const n2 = switch (p.ops[i + 1]) {
        .rmsnorm => |n| n,
        else => return false,
    };
    const r1 = switch (p.ops[i + 2]) {
        .rope => |r| r,
        else => return false,
    };
    const r2 = switch (p.ops[i + 3]) {
        .rope => |r| r,
        else => return false,
    };
    const s1 = switch (p.ops[i + 4]) {
        .store => |st| st,
        else => return false,
    };
    const s2 = switch (p.ops[i + 5]) {
        .store => |st| st,
        else => return false,
    };
    if (n1.weight == null or n2.weight == null or !sameView(n1.x, n1.out) or !sameView(n2.x, n2.out)) return false;
    if (n1.eps != n2.eps or n1.group != r1.dim or n2.group != r1.dim) return false;
    if (!sameView(r1.x, n1.x) or !sameView(r2.x, n2.x) or r2.dim != r1.dim or r2.rot != r1.rot or r2.freqs != r1.freqs or r2.fast_trig != r1.fast_trig) return false;
    if (r1.rot > r1.dim or r1.x.width != r1.heads * r1.dim or r2.x.width != r2.heads * r2.dim) return false;
    if (!sameView(s1.src, r2.x) or s2.src.width != s1.src.width or s1.dst == s2.dst) return false;
    const k1 = p.states[s1.dst];
    const k2 = p.states[s2.dst];
    if (k1.kind != k2.kind or k1.horizon != k2.horizon or k1.width != s1.src.width or k2.width != s2.src.width) return false;
    return !overlaps(n1.x, n2.x) and !overlaps(s2.src, n1.x) and !overlaps(s2.src, n2.x);
}

/// The schedule for a fusion selection. Steps cover every op exactly once, in order.
pub fn build(a: std.mem.Allocator, p: *const ir.Program, fusion: Fusion, info: []const WeightInfo) ![]Step {
    var steps: std.ArrayList(Step) = .empty;
    var i: u32 = 0;
    while (i < p.ops.len) {
        const step: Step = blk: {
            if (fusion.norms and matchNorms(p, info, i)) break :blk .{ .norms = i };
            if (fusion.glu and matchGlu(p, info, i)) break :blk .{ .glu = i };
            if (fusion.conv and matchConv(p, i)) break :blk .{ .conv = i };
            if (fusion.gate and matchGate(p, i)) break :blk .{ .gate = i };
            if (fusion.attn_prep and matchAttnPrep(p, i)) break :blk .{ .attn_prep = i };
            if (fusion.projections) {
                const n = matchProjections(p, info, i);
                if (n >= 2) break :blk .{ .projections = .{ .first = i, .count = n } };
            }
            break :blk .{ .op = i };
        };
        try steps.append(a, step);
        i += step.len();
    }
    return steps.toOwnedSlice(a);
}

fn noWeights(_: *const anyopaque, _: []const u8) bool {
    return true;
}

test "reference schedule is one step per op; fusions cover every op once" {
    const lfm3 = @import("lfm3.zig");
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const config =
        \\{"model_type":"lfm3","hidden_size":64,"num_hidden_layers":3,"layer_types":["conv","full_attention","sliding_attention"],
        \\"num_attention_heads":4,"num_key_value_heads":2,"head_dim":32,"intermediate_size":96,"conv_L_cache":3,"norm_eps":1e-5,
        \\"vocab_size":40,"eos_token_id":[7],"sliding_window":4,"max_position_embeddings":128,
        \\"rope_parameters":{"full_attention":{"partial_rotary_factor":0.5,"rope_theta":1000000.0},"sliding_attention":{"rope_theta":10000}}}
    ;
    var p = try lfm3.build(std.testing.allocator, config, .{ .context = undefined, .has = noWeights }, .{});
    defer p.deinit();
    const info = try a.alloc(WeightInfo, p.weights.len);
    @memset(info, .{ .q4 = true, .buffer = 1 });
    const reference = try build(a, &p, .{}, info);
    try std.testing.expectEqual(p.ops.len, reference.len);
    const fused = try build(a, &p, .{ .projections = true, .glu = true, .norms = true, .conv = true, .gate = true }, info);
    var next: u32 = 0;
    var counts: [6]u32 = @splat(0);
    for (fused) |s| {
        try std.testing.expectEqual(next, s.first());
        next += s.len();
        counts[@backingInt(std.meta.activeTag(s))] += 1;
    }
    try std.testing.expectEqual(@as(u32, @intCast(p.ops.len)), next);
    try std.testing.expectEqual(@as(u32, 2), counts[1]); // q/k/v/gate projections, 2 attention layers
    try std.testing.expectEqual(@as(u32, 3), counts[2]); // one GLU per layer
    // Per layer: post-operator norm+add+ffn norm; post-ffn norm+add+next operator norm
    // except the last layer, whose next norm is a head op (different rows).
    try std.testing.expectEqual(@as(u32, 5), counts[3]);
    try std.testing.expectEqual(@as(u32, 1), counts[4]);
    try std.testing.expectEqual(@as(u32, 2), counts[5]);
    // Not 4-bit or split buffers: no multi-matrix fusion.
    @memset(info, .{ .q4 = false, .buffer = 1 });
    const generic = try build(a, &p, .{ .projections = true, .glu = true }, info);
    try std.testing.expectEqual(p.ops.len, generic.len);
}

test "liveness: a view read later is live, an overwritten one is dead" {
    var b = ir.Builder.init(std.testing.allocator);
    defer b.deinit();
    const tok = try b.state(.{ .name = "tokens", .kind = .tokens, .width = 1 });
    _ = try b.state(.{ .name = "position", .kind = .position, .width = 1 });
    const x = try b.state(.{ .name = "x", .kind = .scratch, .width = 4 });
    const y = try b.state(.{ .name = "y", .kind = .scratch, .width = 4 });
    try b.op(.{ .binary = .{ .kind = .add, .a = b.view(x), .b = b.view(x), .out = b.view(y) } });
    try b.op(.{ .binary = .{ .kind = .add, .a = b.view(y), .b = b.view(x), .out = b.view(x) } });
    try b.op(.{ .binary = .{ .kind = .add, .a = b.view(x), .b = b.view(x), .out = b.view(y) } });
    try b.op(.{ .binary = .{ .kind = .add, .a = b.view(y), .b = b.view(y), .out = b.view(x) } });
    const p: ir.Program = .{ .arena = b.arena, .meta = .{ .model_type = "t", .vocab = 1, .eos = &.{}, .max_context = 1 }, .weights = &.{}, .states = b.states.items, .tables = &.{}, .ops = b.ops.items, .head_start = 4, .traces = &.{}, .tokens = b.view(tok), .logits = .{ .state = tok, .width = 1 }, .next = .{ .state = tok, .width = 1 } };
    try std.testing.expect(deadAfter(&p, b.view(y), 1)); // op 2 overwrites y before any read
    try std.testing.expect(!deadAfter(&p, b.view(x), 1)); // op 2 reads x
    try std.testing.expect(deadAfter(&p, b.view(x), 3));
}

test "decode search: attention prep fusion covers each attention layer's six ops once" {
    const lfm3 = @import("lfm3.zig");
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const config =
        \\{"model_type":"lfm3","hidden_size":64,"num_hidden_layers":3,"layer_types":["conv","full_attention","sliding_attention"],
        \\"num_attention_heads":4,"num_key_value_heads":2,"head_dim":32,"intermediate_size":96,"conv_L_cache":3,"norm_eps":1e-5,
        \\"vocab_size":40,"eos_token_id":[7],"sliding_window":4,"max_position_embeddings":128,
        \\"rope_parameters":{"full_attention":{"partial_rotary_factor":0.5,"rope_theta":1000000.0},"sliding_attention":{"rope_theta":10000}}}
    ;
    var p = try lfm3.build(std.testing.allocator, config, .{ .context = undefined, .has = noWeights }, .{});
    defer p.deinit();
    const info = try a.alloc(WeightInfo, p.weights.len);
    @memset(info, .{ .q4 = true, .buffer = 1 });
    const steps = try build(a, &p, .{ .projections = true, .glu = true, .norms = true, .conv = true, .gate = true, .attn_prep = true }, info);
    var next: u32 = 0;
    var preps: u32 = 0;
    for (steps) |s| {
        try std.testing.expectEqual(next, s.first());
        next += s.len();
        if (s == .attn_prep) preps += 1;
    }
    try std.testing.expectEqual(@as(u32, @intCast(p.ops.len)), next);
    try std.testing.expectEqual(@as(u32, 2), preps);
}
