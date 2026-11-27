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
    /// add(a, b -> r), rmsnorm(r -> out), leaving r materialized.
    add_norm: bool = false,
    /// mul(a, b -> u), store(u -> ring), conv(ring -> v), mul(c, v -> y): one launch.
    conv: bool = false,
    /// attention(-> a), mul(a, act(g)) with dead a: one launch.
    gate: bool = false,
    // ---- decode search ----
    /// rmsnorm(q), rmsnorm(k) per head in place, rope(q), rope(k), store(k), store(v): one launch.
    attn_prep: bool = false,
    bf16_cast: bool = false,
    controller_merge: bool = false,
    bf16_projections: bool = false,
    head_softcap: bool = false,
    depth_norm: bool = false,
    inkling_conv: bool = false,
    relative_gate: bool = false,
    qkvr_prep: bool = false,
    merge_norm: bool = false,

    pub fn any(self: Fusion) bool {
        return self.projections or self.glu or self.norms or self.add_norm or self.conv or self.gate or self.attn_prep or self.bf16_cast or self.controller_merge or self.bf16_projections or self.head_softcap or self.depth_norm or self.inkling_conv or self.relative_gate or self.qkvr_prep or self.merge_norm;
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
    add_norm: struct { first: u32, ops_len: u32, round_add: bool, round_out: bool },
    bf16_cast: u32,
    controller_merge: u32,
    bf16_projections: struct { first: u32, count: u32, ops_len: u32, round_mask: u32 },
    head_softcap: u32,
    head_softcap_loopcd: u32,
    controller_depth_norms: u32,
    depth_norm: struct { first: u32, ops_len: u32, round_out: bool },
    inkling_conv: u32,
    relative_gate: u32,
    qkvr_sconv_prep: u32,
    qkvr_dyn_prep: u32,
    qkv_sconv_prep: u32,
    qkv_dyn_prep: u32,
    controller_merge_norm: struct { first: u32, ops_len: u32, round_out: bool },

    pub fn first(self: Step) u32 {
        return switch (self) {
            .op, .glu, .norms, .conv, .gate, .attn_prep, .bf16_cast, .controller_merge, .head_softcap, .head_softcap_loopcd, .controller_depth_norms, .inkling_conv, .relative_gate, .qkvr_sconv_prep, .qkvr_dyn_prep, .qkv_sconv_prep, .qkv_dyn_prep => |i| i,
            .add_norm => |m| m.first,
            .projections => |m| m.first,
            .bf16_projections => |m| m.first,
            .depth_norm => |m| m.first,
            .controller_merge_norm => |m| m.first,
        };
    }
    pub fn len(self: Step) u32 {
        return switch (self) {
            .op => 1,
            .projections => |m| m.count,
            .glu, .norms, .controller_merge, .head_softcap => 3,
            .head_softcap_loopcd => 7,
            .conv, .relative_gate => 4,
            .gate, .bf16_cast, .inkling_conv => 2,
            .attn_prep => 6,
            .add_norm => |m| m.ops_len,
            .bf16_projections => |m| m.ops_len,
            .controller_depth_norms => 5,
            .depth_norm => |m| m.ops_len,
            .qkvr_sconv_prep => 15,
            .qkvr_dyn_prep => 13,
            .qkv_sconv_prep => 12,
            .qkv_dyn_prep => 9,
            .controller_merge_norm => |m| m.ops_len,
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
    dense: bool = false,
};

fn overlaps(a: ir.View, b: ir.View) bool {
    return a.state == b.state and a.col < b.col + b.width and b.col < a.col + a.width;
}
fn covers(a: ir.View, b: ir.View) bool {
    return a.state == b.state and a.col <= b.col and a.col + a.width >= b.col + b.width;
}

/// Views an op reads and writes (scratch/tokens only; caches are persistent).
pub fn access(op: ir.Op, reads: *[3]?ir.View, writes: *?ir.View) void {
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
        .row_transform => |t| {
            reads[0] = t.x;
            writes.* = t.out;
        },
        .grouped_project => |p| {
            reads[0] = p.x;
            writes.* = p.out;
        },
        .concat => |c| {
            reads[0] = c.a;
            reads[1] = c.b;
            writes.* = c.out;
        },
        .activate => |a| {
            reads[0] = a.x;
            writes.* = a.out;
        },
        .softcapped_swiglu => |s| {
            reads[0] = s.gate;
            reads[1] = s.up;
            writes.* = s.out;
        },
        .depth_attention => |d| {
            reads[0] = d.bank;
            reads[1] = d.partial;
            writes.* = d.out;
        },
        .store => |s| reads[0] = s.src,
        .conv => |c| writes.* = c.out,
        .inkling_short_conv => |c| {
            reads[0] = c.residual;
            writes.* = c.out;
        },
        .inkling_dynamic_conv => |c| {
            reads[0] = c.dynamic;
            reads[1] = c.residual;
            writes.* = c.out;
        },
        .inkling_message_merge => |m| {
            reads[0] = m.hidden;
            reads[1] = m.message;
            reads[2] = m.gates;
            writes.* = m.out;
        },
        .rope => |r| {
            reads[0] = r.x;
            writes.* = r.x;
        },
        .attention => |a| {
            reads[0] = a.q;
            writes.* = a.out;
        },
        .relative_attention => |a| {
            reads[0] = a.q;
            reads[1] = a.bias;
            reads[2] = a.sink;
            writes.* = a.out;
        },
        .l2_clip => |c| {
            reads[0] = c.x;
            writes.* = c.out;
        },
        .route_topk => |r| {
            reads[0] = r.logits;
            writes.* = r.out;
        },
        .expert_gate_up => |e| {
            reads[0] = e.x;
            reads[1] = e.routes;
            writes.* = e.out;
        },
        .expert_down => |e| {
            reads[0] = e.x;
            reads[1] = e.routes;
            writes.* = e.out;
        },
        .select_halves => |s| {
            reads[0] = s.x;
            writes.* = s.out;
        },
        .gated_delta_net => |g| {
            reads[0] = g.qkv;
            reads[1] = g.a;
            reads[2] = g.b;
            writes.* = g.out;
        },
        .argmax => |m| {
            reads[0] = m.x;
            writes.* = m.out;
        },
        .loopcd_contrast => |m| {
            reads[0] = m.strong;
            reads[1] = m.weak;
            writes.* = m.out;
        },
    }
}

const LaunchAccess = struct {
    reads: [6]?ir.View = @splat(null),
    writes: [6]?ir.View = @splat(null),

    fn overlapsWrites(a: LaunchAccess, b: LaunchAccess) bool {
        for (a.writes) |wa| if (wa) |av| {
            for (b.writes) |wb| if (wb) |bv| if (overlaps(av, bv)) return true;
            for (b.reads) |rb| if (rb) |bv| if (overlaps(av, bv)) return true;
        };
        for (b.writes) |wb| if (wb) |bv| {
            for (a.reads) |ra| if (ra) |av| if (overlaps(av, bv)) return true;
        };
        return false;
    }
};

fn cacheView(p: *const ir.Program, id: ir.StateId) ir.View {
    return .{ .state = id, .width = p.states[id].width };
}

fn launchAccess(step: Step, p: *const ir.Program, info: []const WeightInfo) ?LaunchAccess {
    switch (step) {
        .op, .bf16_cast => |i| {
            const op = p.ops[i];
            switch (op) {
                .linear => |l| if (!info[l.w].dense) return null,
                .embed, .rmsnorm, .binary, .row_transform, .grouped_project, .concat, .activate,
                .softcapped_swiglu, .depth_attention, .inkling_message_merge, .loopcd_contrast, .l2_clip, .route_topk,
                .select_halves, .gated_delta_net,
                .store, .conv, .inkling_short_conv, .inkling_dynamic_conv => {},
                else => return null,
            }
            var reads: [3]?ir.View = undefined;
            var write: ?ir.View = null;
            access(op, &reads, &write);
            var out: LaunchAccess = .{};
            out.reads[0] = reads[0];
            out.reads[1] = reads[1];
            out.reads[2] = reads[2];
            // Persistent caches occupy distinct buffers. Conservatively mark a
            // whole cache as read or written, even when only one position is used.
            switch (op) {
                .store => |o| write = cacheView(p, o.dst),
                .gated_delta_net => |o| {
                    out.reads[3] = cacheView(p, o.state);
                    out.writes[1] = cacheView(p, o.state);
                },
                .conv => |o| out.reads[2] = cacheView(p, o.src),
                .inkling_short_conv => |o| out.reads[2] = cacheView(p, o.src),
                .inkling_dynamic_conv => |o| out.reads[2] = cacheView(p, o.src),
                else => {},
            }
            out.writes[0] = write orelse return null;
            return out;
        },
        .inkling_conv => |i| {
            const c = p.ops[i + 1].inkling_short_conv;
            const ring = cacheView(p, c.src);
            var out: LaunchAccess = .{};
            out.reads[0] = c.residual;
            out.reads[1] = ring;
            out.writes[0] = c.out;
            out.writes[1] = ring;
            return out;
        },
        .add_norm => |m| {
            const b = p.ops[m.first].binary;
            const norm_idx = m.first + 1 + @intFromBool(m.round_add);
            const n = p.ops[norm_idx].rmsnorm;
            var out: LaunchAccess = .{};
            out.reads[0] = b.a;
            out.reads[1] = b.b;
            out.writes[0] = b.out;
            out.writes[1] = n.out;
            return out;
        },
        .depth_norm => |m| {
            const d = p.ops[m.first].depth_attention;
            const n = p.ops[m.first + 1].rmsnorm;
            var out: LaunchAccess = .{};
            out.reads[0] = d.bank;
            out.reads[1] = d.partial;
            out.writes[0] = d.out;
            out.writes[1] = n.out;
            return out;
        },
        .controller_depth_norms => |i| {
            const d = p.ops[i].depth_attention;
            const n1 = p.ops[i + 1].rmsnorm;
            const n2 = p.ops[i + 3].rmsnorm;
            var out: LaunchAccess = .{};
            out.reads[0] = d.bank;
            out.reads[1] = d.partial;
            out.reads[2] = n1.x;
            out.writes[0] = d.out;
            out.writes[1] = n1.out;
            out.writes[2] = n2.out;
            return out;
        },
        .controller_merge => |i| {
            const l0 = p.ops[i].linear;
            const m = p.ops[i + 2].inkling_message_merge;
            var out: LaunchAccess = .{};
            out.reads[0] = l0.x;
            out.reads[1] = m.hidden;
            out.reads[2] = m.message;
            out.writes[0] = m.gates;
            out.writes[1] = m.out;
            return out;
        },
        .controller_merge_norm => |cmn| {
            const l0 = p.ops[cmn.first].linear;
            const m = p.ops[cmn.first + 2].inkling_message_merge;
            const n = p.ops[cmn.first + 3].rmsnorm;
            var out: LaunchAccess = .{};
            out.reads[0] = l0.x;
            out.reads[1] = m.hidden;
            out.reads[2] = m.message;
            out.writes[0] = m.gates;
            out.writes[1] = m.out;
            out.writes[2] = n.out;
            return out;
        },
        .bf16_projections => |m| {
            const l0 = p.ops[m.first].linear;
            var out: LaunchAccess = .{};
            out.reads[0] = l0.x;
            var cursor = m.first;
            for (0..m.count) |k| {
                out.writes[k] = p.ops[cursor].linear.out;
                const rounded = ((m.round_mask >> @intCast(k)) & 1) != 0;
                cursor += if (rounded) @as(u32, 2) else 1;
            }
            return out;
        },
        else => return null,
    }
}

/// Only omit a barrier when every dispatch since the last one is a known
/// single-launch operation with no RAW, WAW, or WAR hazard and within 36 ops
/// (< packScratch reuse_margin = 40).
pub fn independentLaunches(gpa: std.mem.Allocator, steps: []const Step, p: *const ir.Program, info: []const WeightInfo) ![]bool {
    const independent = try gpa.alloc(bool, steps.len);
    var active: [16]LaunchAccess = undefined;
    var count: usize = 0;
    var run_start_op: u32 = 0;
    for (steps, independent) |step, *can_overlap| {
        can_overlap.* = false;
        const now = launchAccess(step, p, info) orelse {
            count = 0;
            run_start_op = step.first() + step.len();
            continue;
        };
        const end_op = step.first() + step.len();
        if (count != 0 and count < active.len and end_op - run_start_op <= 36) {
            can_overlap.* = true;
            for (active[0..count]) |prev| {
                if (now.overlapsWrites(prev)) {
                    can_overlap.* = false;
                    break;
                }
            }
        }
        if (!can_overlap.*) {
            count = 0;
            run_start_op = step.first();
        }
        active[count] = now;
        count += 1;
    }
    return independent;
}

/// True when the value op `writer` stores in `v` is read by no op after
/// `last_reader` (it is overwritten first, or never read), and is not a host
/// output or a trace.
pub fn deadAfter(p: *const ir.Program, v: ir.View, last_reader: u32) bool {
    if (overlaps(v, p.logits) or overlaps(v, p.next)) return false;
    for (p.traces) |t| if (t.view.state == v.state) return false;
    if (p.draft_final_norm) |dfn| if (dfn.view.state == v.state) return false;
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
    if (n1.round_normalized != .none or n2.round_normalized != .none) return false;
    if (info[n1.weight.?].buffer == 0 or info[n1.weight.?].buffer != info[n2.weight.?].buffer) return false;
    if (n1.group != n1.x.width or n2.group != n2.x.width or n1.group != n2.group or n1.eps != n2.eps) return false;
    if (add.kind != .add or add.a_act != .identity or add.b_act != .identity or add.b_every != 1) return false;
    const t = n1.out;
    const r = if (sameView(add.b, t)) add.a else if (sameView(add.a, t)) add.b else return false;
    if (!sameView(add.out, r) or !sameView(n2.x, r) or overlaps(n2.out, r) or overlaps(n1.x, r) or overlaps(t, r) or overlaps(n2.out, n1.x)) return false;
    return deadAfter(p, t, i + 1);
}

fn matchAddNorm(p: *const ir.Program, i: u32) ?@FieldType(Step, "add_norm") {
    if (!sameSide(p, i, 2)) return null;
    const add = switch (p.ops[i]) {
        .binary => |b| b,
        else => return null,
    };
    if (add.kind != .add or add.a_act != .identity or add.b_act != .identity or add.b_every != 1) return null;
    var cursor: u32 = i + 1;
    var round_add = false;
    if (cursor < p.ops.len and sameSide(p, i, 3) and p.ops[cursor] == .row_transform) {
        const t = p.ops[cursor].row_transform;
        if (t.transform == .copy and t.round == .bf16 and sameView(add.out, t.x) and sameView(add.out, t.out)) {
            round_add = true;
            cursor += 1;
        }
    }
    if (cursor >= p.ops.len or !sameSide(p, i, cursor - i + 1)) return null;
    const norm = switch (p.ops[cursor]) {
        .rmsnorm => |n| n,
        else => return null,
    };
    const w = norm.group;
    if (w == 0 or (w != 2048 and (w % 4 != 0 or w > 4096)) or !sameView(norm.x, add.out)) return null;
    if (norm.weight == null and w == 2048) return null;
    if (add.a.width != w or add.b.width != w or add.out.width != w) return null;
    if (overlaps(add.a, add.out) and !sameView(add.a, add.out)) return null;
    if (overlaps(add.b, add.out) and !sameView(add.b, add.out)) return null;
    if (overlaps(norm.out, add.out) or overlaps(norm.out, add.a) or overlaps(norm.out, add.b)) return null;
    var round_out = false;
    if (cursor + 1 < p.ops.len and sameSide(p, i, cursor - i + 2) and p.ops[cursor + 1] == .row_transform) {
        const t = p.ops[cursor + 1].row_transform;
        if (t.transform == .copy and t.round == .bf16 and sameView(norm.out, t.x) and sameView(norm.out, t.out)) {
            round_out = true;
            cursor += 1;
        }
    }
    const last_op = cursor;
    for (p.traces) |t| {
        if (round_add and t.after_op == i) return null;
        if (t.after_op >= i and t.after_op < last_op and overlaps(t.view, norm.out)) return null;
        if (t.after_op == last_op and overlaps(t.view, norm.out)) return null;
    }
    return .{ .first = i, .ops_len = last_op - i + 1, .round_add = round_add, .round_out = round_out };
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
    if (@backingInt(m0.a_act) >= 4 or @backingInt(m0.b_act) >= 4 or @backingInt(m1.a_act) >= 4 or @backingInt(m1.b_act) >= 4) return false;
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
    if (n1.round_normalized != .none or n2.round_normalized != .none) return false;
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
fn matchBf16Cast(p: *const ir.Program, info: []const WeightInfo, i: u32) bool {
    if (!sameSide(p, i, 2)) return false;
    const out = switch (p.ops[i]) {
        .rmsnorm => |n| n.out,
        .linear => |l| if (info[l.w].dense) l.out else return false,
        .binary => |b| b.out,
        .l2_clip => |c| c.out,
        else => return false,
    };
    const cast = switch (p.ops[i + 1]) {
        .row_transform => |t| t,
        else => return false,
    };
    if (cast.transform != .copy or cast.round != .bf16 or !sameView(out, cast.x) or !sameView(out, cast.out)) return false;
    for (p.traces) |trace| if (trace.after_op == i) return false;
    return true;
}

fn hasTraceBetween(p: *const ir.Program, first: u32, last_exclusive: u32) bool {
    for (p.traces) |t| if (t.after_op >= first and t.after_op < last_exclusive) return true;
    return false;
}

fn matchControllerMergeOps(p: *const ir.Program, info: []const WeightInfo, i: u32) ?struct {
    l0: @FieldType(ir.Op, "linear"),
    l1: @FieldType(ir.Op, "linear"),
    m: @FieldType(ir.Op, "inkling_message_merge"),
} {
    const l0 = linearOf(p.ops[i]) orelse return null;
    const l1 = linearOf(p.ops[i + 1]) orelse return null;
    const m = switch (p.ops[i + 2]) {
        .inkling_message_merge => |v| v,
        else => return null,
    };
    if (info[l0.w].q4 or !info[l0.w].dense or info[l1.w].q4 or !info[l1.w].dense) return null;
    if (!l0.input_bf16 or !l1.input_bf16 or !sameView(l0.x, l1.x)) return null;
    if (l0.out.state != l1.out.state or l0.out.state != m.gates.state) return null;
    if (l0.out.width != l1.out.width or l0.out.width != m.out.width or m.gates.width != 2 * l0.out.width) return null;
    if (l0.out.col != m.gates.col or l1.out.col != l0.out.col + l0.out.width) return null;
    if (overlaps(l0.out, l0.x) or overlaps(l1.out, l0.x) or overlaps(m.out, m.hidden) or overlaps(m.out, m.message) or overlaps(m.out, l0.x)) return null;
    if (!deadAfter(p, m.gates, i + 2)) return null;
    return .{ .l0 = l0, .l1 = l1, .m = m };
}

fn matchControllerMerge(p: *const ir.Program, info: []const WeightInfo, i: u32) bool {
    if (!sameSide(p, i, 3) or hasTraceBetween(p, i, i + 2)) return false;
    return matchControllerMergeOps(p, info, i) != null;
}

fn matchControllerMergeNorm(p: *const ir.Program, info: []const WeightInfo, i: u32) ?@FieldType(Step, "controller_merge_norm") {
    if (!sameSide(p, i, 4)) return null;
    const cm = matchControllerMergeOps(p, info, i) orelse return null;
    const norm = switch (p.ops[i + 3]) {
        .rmsnorm => |v| v,
        else => return null,
    };
    if (!sameView(cm.m.out, norm.x) or norm.x.width != norm.group) return null;
    if (overlaps(norm.out, cm.m.gates) or overlaps(norm.out, cm.m.hidden) or overlaps(norm.out, cm.m.message) or overlaps(norm.out, cm.l0.x)) return null;
    var ops_len: u32 = 4;
    var round_out = false;
    if (i + 4 < p.ops.len and sameSide(p, i, 5)) {
        if (p.ops[i + 4] == .row_transform) {
            const t = p.ops[i + 4].row_transform;
            if (t.transform == .copy and t.round == .bf16 and sameView(norm.out, t.x) and sameView(norm.out, t.out)) {
                ops_len = 5;
                round_out = true;
            }
        }
    }
    if (hasTraceBetween(p, i, i + ops_len - 1)) return null;
    if (!deadAfter(p, cm.m.out, i + 3)) return null;
    return .{ .first = i, .ops_len = ops_len, .round_out = round_out };
}

fn matchBf16Projections(p: *const ir.Program, info: []const WeightInfo, i: u32) ?@FieldType(Step, "bf16_projections") {
    const l0 = linearOf(p.ops[i]) orelse return null;
    if (info[l0.w].q4 or !info[l0.w].dense or info[l0.w].buffer == 0 or !l0.input_bf16) return null;
    if (l0.out.width % 4 != 0 or l0.x.width % 128 != 0) return null;
    var outs: [6]ir.View = undefined;
    var count: u32 = 0;
    var cursor: u32 = i;
    var round_mask: u32 = 0;
    while (count < 6 and cursor < p.ops.len and sameSide(p, i, cursor - i + 1)) {
        const l = linearOf(p.ops[cursor]) orelse break;
        if (info[l.w].q4 or !info[l.w].dense or info[l.w].buffer != info[l0.w].buffer or !l.input_bf16) break;
        if (!sameView(l.x, l0.x) or l.out.width % 4 != 0) break;
        var clash = overlaps(l.out, l0.x);
        for (outs[0..count]) |prev| clash = clash or overlaps(prev, l.out);
        if (clash) break;
        var step_len: u32 = 1;
        if (cursor + 1 < p.ops.len and sameSide(p, i, cursor - i + 2) and !hasTraceBetween(p, cursor, cursor + 1)) {
            if (p.ops[cursor + 1] == .row_transform) {
                const t = p.ops[cursor + 1].row_transform;
                if (t.transform == .copy and t.round == .bf16 and sameView(l.out, t.x) and sameView(l.out, t.out)) {
                    round_mask |= @as(u32, 1) << @intCast(count);
                    step_len = 2;
                }
            }
        }
        outs[count] = l.out;
        count += 1;
        cursor += step_len;
    }
    if (count < 2) return null;
    return .{ .first = i, .count = count, .ops_len = cursor - i, .round_mask = round_mask };
}

fn matchHeadSoftcap(p: *const ir.Program, info: []const WeightInfo, i: u32) bool {
    if (!sameSide(p, i, 3) or hasTraceBetween(p, i, i + 2)) return false;
    const l = linearOf(p.ops[i]) orelse return false;
    if (info[l.w].q4 or !info[l.w].dense or !l.input_bf16) return false;
    const cast = switch (p.ops[i + 1]) {
        .row_transform => |t| t,
        else => return false,
    };
    const cap = switch (p.ops[i + 2]) {
        .row_transform => |t| t,
        else => return false,
    };
    if (cast.transform != .copy or cast.round != .bf16 or !sameView(l.out, cast.x) or !sameView(l.out, cast.out)) return false;
    if (cap.transform != .softcap or cap.round != .none) return false;
    if (!sameView(cap.x, l.out) or cap.out.width > l.out.width) return false;
    if (overlaps(cap.out, l.out)) return false;
    return deadAfter(p, l.out, i + 2);
}

fn matchHeadSoftcapLoopCd(p: *const ir.Program, info: []const WeightInfo, i: u32) bool {
    if (!sameSide(p, i, 7) or hasTraceBetween(p, i, i + 6)) return false;
    if (!matchHeadSoftcap(p, info, i) or !matchHeadSoftcap(p, info, i + 3)) return false;
    const l0 = p.ops[i].linear;
    const cap0 = p.ops[i + 2].row_transform;
    const l1 = p.ops[i + 3].linear;
    const cap1 = p.ops[i + 5].row_transform;
    const cd = switch (p.ops[i + 6]) {
        .loopcd_contrast => |v| v,
        else => return false,
    };
    if (l0.w != l1.w or l0.x.width != l1.x.width) return false;
    if (cap0.transform.softcap != cap1.transform.softcap or cap0.out.width != cap1.out.width) return false;
    if (overlaps(cap0.out, cap1.out) or overlaps(cap0.out, l0.x) or overlaps(cap0.out, l1.x) or
        overlaps(cap1.out, l0.x) or overlaps(cap1.out, l1.x)) return false;
    if (!sameView(cd.strong, cap0.out) or !sameView(cd.weak, cap1.out) or !sameView(cd.out, cap0.out)) return false;
    if (cd.round != .none) return false;
    return deadAfter(p, cap1.out, i + 6);
}

fn matchControllerDepthNorms(p: *const ir.Program, i: u32) bool {
    if (!sameSide(p, i, 5) or hasTraceBetween(p, i, i + 4)) return false;
    const d = switch (p.ops[i]) {
        .depth_attention => |v| v,
        else => return false,
    };
    const n1 = switch (p.ops[i + 1]) {
        .rmsnorm => |v| v,
        else => return false,
    };
    const c1 = switch (p.ops[i + 2]) {
        .row_transform => |v| v,
        else => return false,
    };
    const n2 = switch (p.ops[i + 3]) {
        .rmsnorm => |v| v,
        else => return false,
    };
    const c2 = switch (p.ops[i + 4]) {
        .row_transform => |v| v,
        else => return false,
    };
    if (n1.weight != null or n2.weight != null or n1.round_normalized != .none or n2.round_normalized != .none) return false;
    if (n1.eps != n2.eps or n1.group != d.partial.width or n2.group != d.partial.width or n1.x.width != n1.group or n2.x.width != n2.group) return false;
    if (!sameView(n2.x, d.out) or sameView(n1.x, d.out)) return false;
    if (c1.transform != .copy or c1.round != .bf16 or !sameView(n1.out, c1.x) or !sameView(n1.out, c1.out)) return false;
    if (c2.transform != .copy or c2.round != .bf16 or !sameView(n2.out, c2.x) or !sameView(n2.out, c2.out)) return false;
    if (n1.out.state != n2.out.state or n2.out.col != n1.out.col + n1.out.width) return false;
    for ([_]ir.View{ d.out, n1.out, n2.out }) |out| {
        if (overlaps(out, d.bank) or overlaps(out, d.partial) or overlaps(out, n1.x)) return false;
    }
    return !overlaps(d.out, n1.out) and !overlaps(d.out, n2.out);
}

fn matchDepthNorm(p: *const ir.Program, i: u32) ?@FieldType(Step, "depth_norm") {
    if (!sameSide(p, i, 2)) return null;
    const d = switch (p.ops[i]) {
        .depth_attention => |v| v,
        else => return null,
    };
    const n = switch (p.ops[i + 1]) {
        .rmsnorm => |v| v,
        else => return null,
    };
    if (n.group != d.partial.width or n.x.width != n.group or !sameView(n.x, d.out)) return null;
    if (overlaps(n.out, d.bank) or overlaps(n.out, d.partial) or overlaps(n.out, d.out)) return null;
    var ops_len: u32 = 2;
    var round_out = false;
    if (i + 2 < p.ops.len and sameSide(p, i, 3) and p.ops[i + 2] == .row_transform) {
        const t = p.ops[i + 2].row_transform;
        if (t.transform == .copy and t.round == .bf16 and sameView(n.out, t.x) and sameView(n.out, t.out)) {
            ops_len = 3;
            round_out = true;
        }
    }
    if (hasTraceBetween(p, i, i + ops_len - 1)) return null;
    if (!deadAfter(p, d.out, i + 1)) return null;
    return .{ .first = i, .ops_len = ops_len, .round_out = round_out };
}

fn matchInklingConv(p: *const ir.Program, i: u32) bool {
    if (!sameSide(p, i, 2) or hasTraceBetween(p, i, i + 1)) return false;
    const st = switch (p.ops[i]) {
        .store => |v| v,
        else => return false,
    };
    const cv = switch (p.ops[i + 1]) {
        .inkling_short_conv => |v| v,
        else => return false,
    };
    if (st.dst != cv.src or p.states[cv.src].kind != .ring) return false;
    if (!sameView(st.src, cv.residual) or overlaps(cv.out, st.src)) return false;
    return true;
}

fn matchRelativeGate(p: *const ir.Program, i: u32) bool {
    if (!sameSide(p, i, 4) or hasTraceBetween(p, i, i + 3)) return false;
    const a = switch (p.ops[i]) {
        .relative_attention => |v| v,
        else => return false,
    };
    const c1 = switch (p.ops[i + 1]) {
        .row_transform => |v| v,
        else => return false,
    };
    const b = switch (p.ops[i + 2]) {
        .binary => |v| v,
        else => return false,
    };
    const c2 = switch (p.ops[i + 3]) {
        .row_transform => |v| v,
        else => return false,
    };
    if (c1.transform != .copy or c1.round != .bf16 or !sameView(c1.x, a.out) or !sameView(c1.out, a.out)) return false;
    if (b.kind != .mul or b.a_act != .identity or b.b_act != .sigmoid or b.b_every != 1) return false;
    if (!sameView(b.a, a.out) or b.out.width != a.out.width or overlaps(b.out, b.b) or overlaps(b.out, a.out)) return false;
    if (c2.transform != .copy or c2.round != .bf16 or !sameView(c2.x, b.out) or !sameView(c2.out, b.out)) return false;
    return deadAfter(p, a.out, i + 2);
}

fn viewsDisjoint(vs: []const ir.View) bool {
    for (vs, 0..) |a, idx| for (vs[idx + 1 ..]) |b| if (overlaps(a, b)) return false;
    return true;
}

fn isCopyBf16(op: ir.Op, x: ir.View, out: ir.View) bool {
    const t = switch (op) {
        .row_transform => |v| v,
        else => return false,
    };
    return t.transform == .copy and t.round == .bf16 and sameView(t.x, x) and sameView(t.out, out);
}

fn matchQkvTail(p: *const ir.Program, i: u32, cv: ir.View, ck: ir.View, dim: u32, has_r: bool) ?struct {
    nv: ir.View,
    q: ir.View,
    r: ?ir.View,
    has_log: bool,
} {
    const n_v = switch (p.ops[i]) {
        .rmsnorm => |v| v,
        else => return null,
    };
    if (n_v.weight != null or n_v.round_normalized != .none or !sameView(n_v.x, cv) or n_v.group != dim or cv.width % dim != 0) return null;
    if (!isCopyBf16(p.ops[i + 1], n_v.out, n_v.out)) return null;
    const cl_q = switch (p.ops[i + 2]) {
        .l2_clip => |v| v,
        else => return null,
    };
    if (cl_q.scale == null or !sameView(cl_q.out, cl_q.x) or cl_q.group != dim or cl_q.x.width % dim != 0) return null;
    if (!isCopyBf16(p.ops[i + 3], cl_q.x, cl_q.x)) return null;
    const cl_k = switch (p.ops[i + 4]) {
        .l2_clip => |v| v,
        else => return null,
    };
    if (cl_k.scale != null or !sameView(cl_k.x, ck) or !sameView(cl_k.out, ck) or cl_k.group != dim or ck.width != cv.width) return null;
    if (!isCopyBf16(p.ops[i + 5], ck, ck)) return null;
    var r_view: ?ir.View = null;
    var off: u32 = i + 6;
    if (has_r) {
        const cl_r = switch (p.ops[i + 6]) {
            .l2_clip => |v| v,
            else => return null,
        };
        if (cl_r.scale == null or !sameView(cl_r.out, cl_r.x) or cl_r.group == 0 or cl_r.x.width % cl_r.group != 0) return null;
        if (!isCopyBf16(p.ops[i + 7], cl_r.x, cl_r.x)) return null;
        if (cl_q.x.width / dim != cl_r.x.width / cl_r.group) return null;
        r_view = cl_r.x;
        off = i + 8;
    }
    var has_log = false;
    if (p.ops[off] == .row_transform and p.ops[off].row_transform.transform == .log_position_scale) {
        const tq = p.ops[off].row_transform;
        if (tq.round != .bf16 or !sameView(tq.x, cl_q.x) or !sameView(tq.out, cl_q.x)) return null;
        if (r_view) |rv| {
            if (off + 4 >= p.ops.len) return null;
            const tr = switch (p.ops[off + 1]) {
                .row_transform => |v| v,
                else => return null,
            };
            if (tr.transform != .log_position_scale or tr.round != .bf16 or !sameView(tr.x, rv) or !sameView(tr.out, rv)) return null;
            if (!std.meta.eql(tq.transform.log_position_scale, tr.transform.log_position_scale)) return null;
            off += 2;
        } else {
            if (off + 2 >= p.ops.len) return null;
            off += 1;
        }
        has_log = true;
    }
    if (r_view) |rv| {
        const gp = switch (p.ops[off]) {
            .grouped_project => |v| v,
            else => return null,
        };
        const sk = switch (p.ops[off + 1]) {
            .store => |v| v,
            else => return null,
        };
        const sv = switch (p.ops[off + 2]) {
            .store => |v| v,
            else => return null,
        };
        const cl_r_group = p.ops[i + 6].l2_clip.group;
        if (!sameView(gp.x, rv) or gp.groups != rv.width / cl_r_group) return null;
        if (!sameView(sk.src, ck) or !sameView(sv.src, n_v.out) or sk.dst == sv.dst) return null;
        if (!p.states[sk.dst].persistent() or !p.states[sv.dst].persistent()) return null;
        if (!viewsDisjoint(&.{ cl_q.x, rv, ck, cv, n_v.out, gp.out })) return null;
        if (!deadAfter(p, ck, off + 2) or !deadAfter(p, n_v.out, off + 2)) return null;
    } else {
        const sk = switch (p.ops[off]) {
            .store => |v| v,
            else => return null,
        };
        const sv = switch (p.ops[off + 1]) {
            .store => |v| v,
            else => return null,
        };
        if (!sameView(sk.src, ck) or !sameView(sv.src, n_v.out) or sk.dst == sv.dst) return null;
        if (!p.states[sk.dst].persistent() or !p.states[sv.dst].persistent()) return null;
        if (!viewsDisjoint(&.{ cl_q.x, ck, cv, n_v.out })) return null;
        if (!deadAfter(p, ck, off + 1) or !deadAfter(p, n_v.out, off + 1)) return null;
    }
    return .{ .nv = n_v.out, .q = cl_q.x, .r = r_view, .has_log = has_log };
}

fn matchQkvrSconvPrepImpl(p: *const ir.Program, i: u32, comptime has_r: bool) bool {
    const ops_len: u32 = if (has_r) 15 else 12;
    if (!sameSide(p, i, ops_len) or hasTraceBetween(p, i, i + ops_len - 1)) return false;
    const sk = switch (p.ops[i]) {
        .store => |v| v,
        else => return false,
    };
    const ck = switch (p.ops[i + 1]) {
        .inkling_short_conv => |v| v,
        else => return false,
    };
    const sv = switch (p.ops[i + 2]) {
        .store => |v| v,
        else => return false,
    };
    const cv = switch (p.ops[i + 3]) {
        .inkling_short_conv => |v| v,
        else => return false,
    };
    if (p.states[sk.dst].kind != .ring or p.states[sv.dst].kind != .ring or sk.dst == sv.dst) return false;
    if (ck.src != sk.dst or cv.src != sv.dst or !sameView(sk.src, ck.residual) or !sameView(sv.src, cv.residual)) return false;
    if (ck.round != .bf16 or cv.round != .bf16 or ck.taps != cv.taps or ck.out.width != cv.out.width) return false;
    const dim = switch (p.ops[i + 4]) {
        .rmsnorm => |v| v.group,
        else => return false,
    };
    const tail = matchQkvTail(p, i + 4, cv.out, ck.out, dim, has_r) orelse return false;
    if (tail.has_log) return false;
    if (has_r) {
        const gp = p.ops[i + 12].grouped_project;
        if (!viewsDisjoint(&.{ tail.q, tail.r.?, sk.src, sv.src, ck.out, cv.out, tail.nv, gp.out })) return false;
    } else {
        if (!viewsDisjoint(&.{ tail.q, sk.src, sv.src, ck.out, cv.out, tail.nv })) return false;
    }
    return deadAfter(p, cv.out, i + ops_len - 1);
}

fn matchQkvrDynPrepImpl(p: *const ir.Program, i: u32, comptime has_r: bool) bool {
    const ops_len: u32 = if (has_r) 13 else 9;
    if (!sameSide(p, i, ops_len) or hasTraceBetween(p, i, i + ops_len - 1)) return false;
    const nv = switch (p.ops[i]) {
        .rmsnorm => |v| v,
        else => return false,
    };
    const cl_k = switch (p.ops[i + 4]) {
        .l2_clip => |v| v,
        else => return false,
    };
    const tail = matchQkvTail(p, i, nv.x, cl_k.x, nv.group, has_r) orelse return false;
    return tail.has_log;
}

pub fn build(a: std.mem.Allocator, p: *const ir.Program, fusion: Fusion, info: []const WeightInfo) ![]Step {
    var steps: std.ArrayList(Step) = .empty;
    var i: u32 = 0;
    while (i < p.ops.len) {
        const step: Step = blk: {
            if (fusion.depth_norm) {
                if (matchControllerDepthNorms(p, i)) break :blk .{ .controller_depth_norms = i };
                if (matchDepthNorm(p, i)) |m| break :blk .{ .depth_norm = m };
            }
            if (fusion.merge_norm) if (matchControllerMergeNorm(p, info, i)) |m| break :blk .{ .controller_merge_norm = m };
            if (fusion.controller_merge and matchControllerMerge(p, info, i)) break :blk .{ .controller_merge = i };
            if (fusion.head_softcap) {
                if (matchHeadSoftcapLoopCd(p, info, i)) break :blk .{ .head_softcap_loopcd = i };
                if (matchHeadSoftcap(p, info, i)) break :blk .{ .head_softcap = i };
            }
            if (fusion.bf16_projections) if (matchBf16Projections(p, info, i)) |m| break :blk .{ .bf16_projections = m };
            if (fusion.add_norm) if (matchAddNorm(p, i)) |m| break :blk .{ .add_norm = m };
            if (fusion.qkvr_prep) {
                if (matchQkvrSconvPrepImpl(p, i, true)) break :blk .{ .qkvr_sconv_prep = i };
                if (matchQkvrDynPrepImpl(p, i, true)) break :blk .{ .qkvr_dyn_prep = i };
                if (matchQkvrSconvPrepImpl(p, i, false)) break :blk .{ .qkv_sconv_prep = i };
                if (matchQkvrDynPrepImpl(p, i, false)) break :blk .{ .qkv_dyn_prep = i };
            }
            if (fusion.inkling_conv and matchInklingConv(p, i)) break :blk .{ .inkling_conv = i };
            if (fusion.relative_gate and matchRelativeGate(p, i)) break :blk .{ .relative_gate = i };
            if (fusion.bf16_cast and matchBf16Cast(p, info, i)) break :blk .{ .bf16_cast = i };
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
    var counts: [@typeInfo(Step).@"union".field_names.len]u32 = @splat(0);
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

test "extended independentLaunches handles fused steps and enforces op_span <= 36 cutoff" {
    var b = ir.Builder.init(std.testing.allocator);
    defer b.deinit();
    const tok = try b.state(.{ .name = "tokens", .kind = .tokens, .width = 1 });
    _ = try b.state(.{ .name = "position", .kind = .position, .width = 1 });
    const ring1 = try b.state(.{ .name = "r1", .kind = .ring, .width = 4, .horizon = 3 });
    const ring2 = try b.state(.{ .name = "r2", .kind = .ring, .width = 4, .horizon = 3 });
    const a1 = try b.state(.{ .name = "a1", .kind = .scratch, .width = 4 });
    const a2 = try b.state(.{ .name = "a2", .kind = .scratch, .width = 4 });
    const o1 = try b.state(.{ .name = "o1", .kind = .scratch, .width = 4 });
    const o2 = try b.state(.{ .name = "o2", .kind = .scratch, .width = 4 });
    const cw1 = try b.weight("cw1", &.{ 4, 1, 4 });
    const cw2 = try b.weight("cw2", &.{ 4, 1, 4 });
    // Two independent inkling_conv steps (4 ops total) should overlap.
    try b.op(.{ .store = .{ .src = b.view(a1), .dst = ring1 } });
    try b.op(.{ .inkling_short_conv = .{ .src = ring1, .weight = cw1, .residual = b.view(a1), .out = b.view(o1), .taps = 4, .round = .bf16 } });
    try b.op(.{ .store = .{ .src = b.view(a2), .dst = ring2 } });
    try b.op(.{ .inkling_short_conv = .{ .src = ring2, .weight = cw2, .residual = b.view(a2), .out = b.view(o2), .taps = 4, .round = .bf16 } });
    // Another conv touching ring1 conflicts on the persistent ring cache.
    const a3 = try b.state(.{ .name = "a3", .kind = .scratch, .width = 4 });
    const o3 = try b.state(.{ .name = "o3", .kind = .scratch, .width = 4 });
    try b.op(.{ .store = .{ .src = b.view(a3), .dst = ring1 } });
    try b.op(.{ .inkling_short_conv = .{ .src = ring1, .weight = cw1, .residual = b.view(a3), .out = b.view(o3), .taps = 4, .round = .bf16 } });
    // Add 14 independent 3-op add_norm (binary + bf16_cast + rmsnorm) steps so op_span crosses 36 before count == 16.
    var s_in: [14]ir.StateId = undefined;
    var s_out: [14]ir.StateId = undefined;
    for (0..14) |k| {
        s_in[k] = try b.state(.{ .name = try b.print("in{d}", .{k}), .kind = .scratch, .width = 4 });
        const s_mid = try b.state(.{ .name = try b.print("mid{d}", .{k}), .kind = .scratch, .width = 4 });
        s_out[k] = try b.state(.{ .name = try b.print("out{d}", .{k}), .kind = .scratch, .width = 4 });
        try b.op(.{ .binary = .{ .kind = .add, .a = b.view(s_in[k]), .b = b.view(s_in[k]), .out = b.view(s_mid) } });
        try b.op(.{ .row_transform = .{ .x = b.view(s_mid), .out = b.view(s_mid), .transform = .copy, .round = .bf16 } });
        try b.op(.{ .rmsnorm = .{ .x = b.view(s_mid), .weight = null, .out = b.view(s_out[k]), .group = 4, .eps = 1e-5 } });
    }
    const p: ir.Program = .{ .arena = b.arena, .meta = .{ .model_type = "t", .vocab = 1, .eos = &.{}, .max_context = 1 }, .weights = b.weights.items, .states = b.states.items, .tables = &.{}, .ops = b.ops.items, .head_start = @intCast(b.ops.items.len), .traces = &.{}, .tokens = b.view(tok), .logits = .{ .state = tok, .width = 1 }, .next = .{ .state = tok, .width = 1 } };
    const info = try std.testing.allocator.alloc(WeightInfo, p.weights.len);
    defer std.testing.allocator.free(info);
    @memset(info, .{ .q4 = false, .buffer = 1 });
    const steps = try build(std.testing.allocator, &p, .{ .inkling_conv = true, .add_norm = true }, info);
    defer std.testing.allocator.free(steps);
    try std.testing.expectEqual(@as(usize, 17), steps.len);
    const indep = try independentLaunches(std.testing.allocator, steps, &p, info);
    defer std.testing.allocator.free(indep);
    try std.testing.expect(!indep[0]);
    try std.testing.expect(indep[1]); // r1 and r2 are disjoint
    try std.testing.expect(!indep[2]); // conflicts on r1 (run_start_op = 4)
    for (3..14) |idx| try std.testing.expect(indep[idx]); // ops 4..39 (span <= 36, count <= 12)
    try std.testing.expect(!indep[14]); // op span 4..43 = 39 > 36 forces a barrier even with count == 12 < 16!
}
