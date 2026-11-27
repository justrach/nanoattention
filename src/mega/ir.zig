//! Model-agnostic program IR. A model is a weight table, a state table and a
//! flat op list with explicit shapes. Nothing here knows which model it runs.
//! Every op processes `rows` consecutive token rows (T = 1 when decoding);
//! row r of a batch is absolute position `position + r`.
const std = @import("std");

pub const StateId = u16;
pub const WeightId = u16;
pub const TableId = u16;

/// All state is 32-bit words. The kind decides addressing and lifetime.
pub const StateKind = enum(u8) {
    /// Host-written u32 token ids, one word per batch row.
    tokens,
    /// Per-batch activations: `batch` rows of `width` words, undefined between forwards.
    scratch,
    /// Persistent rows addressed by absolute position (full-attention KV): capacity = context.
    sequence,
    /// Persistent rows addressed by position mod capacity (sliding KV, conv windows).
    /// Readers look back at most `horizon` positions before the row being computed,
    /// so capacity = horizon + batch keeps batch writes off rows an earlier row reads.
    ring,
    /// Persistent fixed-size recurrent state (one snapshot per layer, zeroed on reset).
    recurrent,
    /// The host-held number of cached positions. Exactly one per program.
    position,
};

pub const State = struct {
    name: []const u8,
    kind: StateKind,
    width: u32,
    horizon: u32 = 0,
    /// Scratch only: rows to allocate when not every batch row needs one (0 = batch).
    rows: u32 = 0,

    pub fn persistent(self: State) bool {
        return self.kind == .sequence or self.kind == .ring or self.kind == .recurrent;
    }
    /// Rows allocated for this state under a given batch and context.
    pub fn capacity(self: State, batch: u32, context: u32) u32 {
        return switch (self.kind) {
            .tokens => batch,
            .scratch => if (self.rows != 0) @min(self.rows, batch) else batch,
            .sequence => context,
            .ring => @min(self.horizon + batch, context),
            .recurrent => 1,
            .position => 1,
        };
    }
    /// Rows that hold information after `position` tokens: the snapshot extent.
    pub fn validRows(self: State, batch: u32, context: u32, position: u32) u32 {
        return switch (self.kind) {
            .sequence, .ring => @min(position, self.capacity(batch, context)),
            .recurrent => if (position > 0) 1 else 0,
            .position => 1,
            else => 0,
        };
    }
};

pub const RawWeightDType = enum(u8) { u8, f8_e4m3, f32 };

pub const PackedProjection = struct {
    payload: WeightId,
    scales: WeightId,
    /// Both present for NVFP4; both absent for MXFP8.
    amax: ?WeightId = null,
    denominator: ?WeightId = null,
};

pub const Weight = struct {
    name: []const u8,
    /// Logical shape: quantized matrices are [rows, cols] of dequantized values.
    shape: []const usize,
    /// Non-null only for explicitly packed formats; shape is then physical storage.
    raw_dtype: ?RawWeightDType = null,
};

/// A column window of a state's rows. Row stride is always the state's width.
pub const View = struct {
    state: StateId,
    col: u32 = 0,
    width: u32,
};

pub const Act = enum(u8) { identity = 0, silu = 1, sigmoid = 2, tanh = 3, gelu_tanh = 4 };
pub const BinaryKind = enum(u8) { add = 0, mul = 1, sub = 2 };
/// Optional output rounding, used where a reference materialises a narrower dtype.
pub const Round = enum(u8) { none = 0, bf16 = 1 };
pub const RowTransform = union(enum) {
    copy,
    softcap: f32,
    log_position_scale: struct { floor: u32, alpha: f32 },
    scale: f32,
    weight_scale: WeightId,
};

pub const Op = union(enum) {
    /// out[r] = table[tokens[r]] (dense or affine-quantized table).
    embed: struct { table: WeightId, tokens: View, out: View, round: Round = .none },
    /// RMSNorm over consecutive groups of `group` words: whole vector or per head.
    rmsnorm: struct { x: View, weight: ?WeightId, out: View, group: u32, eps: f32, round_normalized: Round = .none },
    /// out[r] = W x[r]; W is [out.width, x.width], dense or affine-quantized.
    linear: struct { w: WeightId, x: View, out: View, input_bf16: bool = false },
    /// out = act_a(a) (+|*|-) act_b(b[i / b_every]).
    binary: struct { kind: BinaryKind, a: View, b: View, out: View, a_act: Act = .identity, b_act: Act = .identity, b_every: u32 = 1 },
    /// Concatenate two row views into one row.
    concat: struct { a: View, b: View, out: View },
    /// Apply a pointwise activation to a row view.
    activate: struct { kind: Act, x: View, out: View },
    /// Copy, optionally narrow to a prefix, or apply a pointwise transform.
    row_transform: struct { x: View, out: View, transform: RowTransform = .copy, round: Round = .none },
    /// Apply the same dense [input, output] matrix to each contiguous group.
    /// For a causal distance table, only distances <= the absolute row position are defined.
    grouped_project: struct { x: View, out: View, weight: WeightId, groups: u32, round: Round = .none, causal_prefix: bool = false },
    /// Inkling softcapped SwiGLU with independent gate/up caps.
    softcapped_swiglu: struct { gate: View, up: View, out: View, gate_cap: f32, up_cap: f32, round: Round = .none },
    /// Scale each group by an optional per-element scale, then L2-clip it.
    l2_clip: struct { x: View, out: View, group: u32, threshold: f32, scale: ?WeightId = null },
    /// Softmax attention over bank rows plus the current partial block.
    depth_attention: struct { bank: View, partial: View, query: WeightId, query_index: u32 = 0, out: View, scale: f32 },
    /// Write each row to its position's slot of a sequence/ring state.
    store: struct { src: View, dst: StateId },
    /// Depthwise causal conv over a ring of past inputs; weight is [channels, taps].
    /// Positions before 0 read zeros. The ring must already hold this batch's rows.
    conv: struct { src: StateId, weight: WeightId, out: View, taps: u32 },
    /// Inkling short convolution: (depthwise causal conv + residual) / sqrt(taps + 1).
    inkling_short_conv: struct { src: StateId, weight: WeightId, residual: View, out: View, taps: u32, round: Round = .none },
    /// Inkling dynamic short convolution: per-head projected taps + per-channel static taps.
    inkling_dynamic_conv: struct { src: StateId, static_weight: WeightId, dynamic: View, residual: View, out: View, heads: u32, taps: u32, round: Round = .none },
    /// Merge loop-controller gates: (1 - tanh(forget)) * hidden + tanh(gain) * message.
    inkling_message_merge: struct { hidden: View, message: View, gates: View, out: View, round: Round = .none },
    /// Rotate-half RoPE on pairs (i, i + rot/2), i < rot/2, of each head, with a
    /// per-pair frequency table (zero = unrotated, so partial rotation is a table).
    rope: struct { x: View, heads: u32, dim: u32, rot: u32, freqs: TableId, fast_trig: bool = false },
    /// Grouped-query causal attention reading cached K/V of positions
    /// [max(0, p - window + 1), p] (window 0 = unbounded) for row position p.
    attention: struct { q: View, keys: StateId, values: StateId, out: View, heads: u32, kv_heads: u32, dim: u32, window: u32, scale: f32 },
    /// Relative-position attention with one denominator-only sink logit per head.
    relative_attention: struct { q: View, bias: View, sink: View, keys: StateId, values: StateId, out: View, heads: u32, kv_heads: u32, dim: u32, relative_extent: u32, window: u32, scale: f32 },
    /// Route top_k experts by corrected sigmoid score, normalize uncorrected scores.
    /// Output stores expert IDs followed by route weights in one scratch row.
    route_topk: struct { logits: View, correction: WeightId, out: View, experts: u32, top_k: u32, eps: f32, scale: f32 },
    /// Gate/up rows alternate within each packed expert. Output is top_k
    /// consecutive vectors containing the activated intermediate per route.
    expert_gate_up: struct { x: View, routes: View, out: View, bank: PackedProjection, experts: u32, top_k: u32, gate_cap: f32, up_cap: f32, round: Round, input_bf16: bool = false },
    /// Project the selected experts and combine in route order in float32.
    expert_down: struct { x: View, routes: View, out: View, bank: PackedProjection, experts: u32, top_k: u32, round: Round, input_bf16: bool = false },
    /// Split interleaved `[groups, 2, width]` rows into two contiguous halves
    /// `[first_halves, second_halves]`.
    select_halves: struct { x: View, out: View, groups: u32 },
    /// Gated delta rule linear attention over conv+silu QKV (`[q | k | v]`),
    /// updating a persistent `[v_heads, v_head_dim, k_head_dim]` recurrent state.
    gated_delta_net: struct {
        qkv: View,
        a: View,
        b: View,
        a_log: WeightId,
        dt_bias: WeightId,
        state: StateId,
        out: View,
        k_heads: u32,
        v_heads: u32,
        k_head_dim: u32,
        v_head_dim: u32,
        eps: f32 = 1e-6,
    },
    /// out[r] = lowest index of the maximum of x[r]; all-u32 sentinel on non-finite input.
    argmax: struct { x: View, out: View },
    /// LoopCD off-the-shelf loop contrastive decoding: out = strong + w * (strong - weak).
    /// When `adaptive` is true, computes w = omega * (1 - (p1 - p2)) from the top-2 softmax
    /// probabilities of `strong` over the row; otherwise uses fixed weight `omega`.
    loopcd_contrast: struct { strong: View, weak: View, out: View, omega: f32, adaptive: bool = false, round: Round = .none },

    pub fn name(self: Op) []const u8 {
        return @tagName(std.meta.activeTag(self));
    }
};

/// The residual stream after a layer, for debugging and golden comparison.
pub const Trace = struct { after_op: u32, view: View };

pub const Meta = struct {
    model_type: []const u8,
    vocab: u32,
    eos: []const u32,
    max_context: u32,
};

pub const Program = struct {
    arena: std.heap.ArenaAllocator,
    meta: Meta,
    weights: []const Weight,
    states: []const State,
    tables: []const []const f32,
    ops: []const Op,
    /// Ops at index >= head_start run only on the rows whose logits are wanted.
    head_start: u32,
    /// Residual-stream checkpoints (embedding output, then each layer's output).
    traces: []const Trace,
    /// Normalized hidden state immediately before unembed (for draft models).
    draft_final_norm: ?Trace = null,
    tokens: View,
    logits: View,
    next: View,

    pub fn deinit(self: *Program) void {
        self.arena.deinit();
    }
    pub fn state(self: *const Program, id: StateId) State {
        return self.states[id];
    }
    pub fn positionState(self: *const Program) ?StateId {
        for (self.states, 0..) |s, i| if (s.kind == .position) return @intCast(i);
        return null;
    }

    /// Structural validation: ids in range, views inside their states, shapes agree.
    /// `weightShape` is checked separately against the checkpoint by the binder.
    pub fn validate(self: *const Program) !void {
        var positions: u32 = 0;
        for (self.states) |s| {
            if (s.width == 0) return error.InvalidStateWidth;
            if (s.kind == .ring and s.horizon == 0) return error.InvalidRingHorizon;
            if (s.kind == .position) positions += 1;
        }
        if (positions != 1) return error.InvalidPositionState;
        if (self.head_start > self.ops.len) return error.InvalidHeadStart;
        try self.checkView(self.tokens, .tokens);
        try self.checkView(self.logits, .scratch);
        try self.checkView(self.next, .scratch);
        for (self.traces) |t| {
            if (t.after_op >= self.ops.len) return error.InvalidTrace;
            try self.checkView(t.view, .scratch);
        }
        if (self.draft_final_norm) |dfn| {
            if (dfn.after_op >= self.ops.len) return error.InvalidTrace;
            try self.checkView(dfn.view, .scratch);
        }
        for (self.ops) |op| try self.checkOp(op);
    }
    fn checkView(self: *const Program, v: View, kind: StateKind) !void {
        if (v.state >= self.states.len) return error.InvalidState;
        const s = self.states[v.state];
        if (s.kind != kind) return error.InvalidViewKind;
        if (v.width == 0 or @as(u64, v.col) + v.width > s.width) return error.ViewOutOfBounds;
    }
    fn checkCache(self: *const Program, id: StateId) !State {
        if (id >= self.states.len) return error.InvalidState;
        const s = self.states[id];
        if (!s.persistent()) return error.InvalidCacheKind;
        return s;
    }
    fn weight(self: *const Program, id: WeightId) !Weight {
        if (id >= self.weights.len) return error.InvalidWeight;
        return self.weights[id];
    }
    fn checkOp(self: *const Program, op: Op) !void {
        switch (op) {
            .embed => |e| {
                try self.checkView(e.tokens, .tokens);
                try self.checkView(e.out, .scratch);
                const w = try self.weight(e.table);
                if (w.shape.len != 2 or w.shape[1] != e.out.width or e.tokens.width != 1) return error.ShapeMismatch;
            },
            .rmsnorm => |n| {
                try self.checkView(n.x, .scratch);
                try self.checkView(n.out, .scratch);
                if (n.x.width != n.out.width or n.group == 0 or n.x.width % n.group != 0) return error.ShapeMismatch;
                if (n.weight) |id| {
                    const w = try self.weight(id);
                    if (w.shape.len != 1 or w.shape[0] != n.group) return error.ShapeMismatch;
                }
                if (!(n.eps > 0)) return error.InvalidEpsilon;
            },
            .linear => |l| {
                try self.checkView(l.x, .scratch);
                try self.checkView(l.out, .scratch);
                const w = try self.weight(l.w);
                if (w.shape.len != 2 or w.shape[0] != l.out.width or w.shape[1] != l.x.width) return error.ShapeMismatch;
                if (l.x.state == l.out.state) return error.LinearAliasing;
            },
            .binary => |b| {
                try self.checkView(b.a, .scratch);
                try self.checkView(b.b, .scratch);
                try self.checkView(b.out, .scratch);
                if (b.b_every == 0 or b.a.width != b.out.width or b.a.width != b.b.width * b.b_every) return error.ShapeMismatch;
            },
            .row_transform => |t| {
                try self.checkView(t.x, .scratch);
                try self.checkView(t.out, .scratch);
                if (t.out.width > t.x.width) return error.ShapeMismatch;
                if (t.x.state == t.out.state and t.x.col != t.out.col) return error.ActivationAliasing;
                switch (t.transform) {
                    .copy => {},
                    .softcap => |cap| if (!std.math.isFinite(cap) or !(cap > 0)) return error.ShapeMismatch,
                    .log_position_scale => |p| if (p.floor == 0 or !std.math.isFinite(p.alpha) or p.alpha < 0) return error.ShapeMismatch,
                    .scale => |s| if (!std.math.isFinite(s)) return error.InvalidScale,
                    .weight_scale => |id| {
                        const w = try self.weight(id);
                        if (w.raw_dtype != null or w.shape.len != 1 or w.shape[0] != 1) return error.ShapeMismatch;
                    },
                }
            },
            .grouped_project => |p| {
                try self.checkView(p.x, .scratch);
                try self.checkView(p.out, .scratch);
                const w = try self.weight(p.weight);
                if (p.groups == 0 or p.x.width % p.groups != 0 or p.out.width % p.groups != 0 or
                    w.shape.len != 2 or w.shape[0] != p.x.width / p.groups or w.shape[1] != p.out.width / p.groups) return error.ShapeMismatch;
                if (p.x.state == p.out.state) return error.ActivationAliasing;
            },
            .concat => |c| {
                try self.checkView(c.a, .scratch);
                try self.checkView(c.b, .scratch);
                try self.checkView(c.out, .scratch);
                if (@as(u64, c.a.width) + c.b.width != c.out.width) return error.ShapeMismatch;
                if (c.out.state == c.a.state or c.out.state == c.b.state) return error.ConcatAliasing;
            },
            .activate => |a| {
                try self.checkView(a.x, .scratch);
                try self.checkView(a.out, .scratch);
                if (a.x.width != a.out.width) return error.ShapeMismatch;
                if (a.x.state == a.out.state) return error.ActivationAliasing;
            },
            .softcapped_swiglu => |s| {
                try self.checkView(s.gate, .scratch);
                try self.checkView(s.up, .scratch);
                try self.checkView(s.out, .scratch);
                if (s.gate.width == 0 or s.gate.width != s.up.width or s.gate.width != s.out.width or
                    !(s.gate_cap > 0) or !(s.up_cap > 0)) return error.ShapeMismatch;
                if (s.out.state == s.gate.state or s.out.state == s.up.state) return error.SoftcappedSwiGLUAliasing;
            },
            .depth_attention => |d| {
                try self.checkView(d.bank, .scratch);
                try self.checkView(d.partial, .scratch);
                try self.checkView(d.out, .scratch);
                const q = try self.weight(d.query);
                if (d.partial.width != d.out.width or d.bank.width % d.partial.width != 0 or
                    d.bank.width / d.partial.width > 7 or !(d.scale > 0)) return error.ShapeMismatch;
                if (q.shape.len == 0 or q.shape[q.shape.len - 1] != d.partial.width) return error.ShapeMismatch;
                var query_count: usize = 1;
                for (q.shape[0 .. q.shape.len - 1]) |dim| query_count = std.math.mul(usize, query_count, dim) catch return error.ShapeMismatch;
                if (d.query_index >= query_count) return error.ShapeMismatch;
                if (d.out.state == d.bank.state or d.out.state == d.partial.state) return error.DepthAttentionAliasing;
            },
            .store => |s| {
                try self.checkView(s.src, .scratch);
                const d = try self.checkCache(s.dst);
                if (d.width != s.src.width) return error.ShapeMismatch;
            },
            .conv => |c| {
                const s = try self.checkCache(c.src);
                try self.checkView(c.out, .scratch);
                const w = try self.weight(c.weight);
                if (c.taps == 0 or s.width != c.out.width or w.shape.len < 2 or w.shape[0] != s.width or w.shape[1] != c.taps) return error.ShapeMismatch;
                if (w.shape.len == 3 and w.shape[2] != 1) return error.ShapeMismatch;
                if (s.kind == .ring and s.horizon < c.taps - 1) return error.RingTooShort;
            },
            .inkling_short_conv => |c| {
                const s = try self.checkCache(c.src);
                try self.checkView(c.residual, .scratch);
                try self.checkView(c.out, .scratch);
                const w = try self.weight(c.weight);
                if (c.taps == 0 or s.kind != .ring or s.width != c.out.width or c.residual.width != c.out.width or
                    (w.shape.len != 2 and w.shape.len != 3) or w.shape[0] != s.width) return error.ShapeMismatch;
                if (w.shape.len == 2) {
                    if (w.shape[1] != c.taps) return error.ShapeMismatch;
                } else if (!((w.shape[1] == 1 and w.shape[2] == c.taps) or
                    (w.shape[1] == c.taps and w.shape[2] == 1))) return error.ShapeMismatch;
                if (s.horizon < c.taps - 1) return error.RingTooShort;
                if (c.residual.state == c.out.state) return error.InklingConvAliasing;
            },
            .inkling_dynamic_conv => |c| {
                const s = try self.checkCache(c.src);
                try self.checkView(c.dynamic, .scratch);
                try self.checkView(c.residual, .scratch);
                try self.checkView(c.out, .scratch);
                const w = try self.weight(c.static_weight);
                if (c.taps == 0 or c.heads == 0 or s.kind != .ring or s.width != c.out.width or c.residual.width != c.out.width or
                    c.out.width % c.heads != 0 or c.dynamic.width != c.heads * c.taps or w.shape.len != 2 or
                    w.shape[0] != c.taps or w.shape[1] != c.out.width) return error.ShapeMismatch;
                if (s.horizon < c.taps - 1) return error.RingTooShort;
                if (c.residual.state == c.out.state or c.dynamic.state == c.out.state) return error.InklingConvAliasing;
            },
            .inkling_message_merge => |m| {
                try self.checkView(m.hidden, .scratch);
                try self.checkView(m.message, .scratch);
                try self.checkView(m.gates, .scratch);
                try self.checkView(m.out, .scratch);
                if (m.hidden.width == 0 or m.message.width != m.hidden.width or m.out.width != m.hidden.width or m.gates.width != 2 * m.hidden.width)
                    return error.ShapeMismatch;
                if (m.out.state == m.hidden.state or m.out.state == m.message.state or m.out.state == m.gates.state)
                    return error.MessageMergeAliasing;
            },
            .rope => |r| {
                try self.checkView(r.x, .scratch);
                if (r.freqs >= self.tables.len) return error.InvalidTable;
                if (r.dim == 0 or r.rot == 0 or r.rot % 2 != 0 or r.rot > r.dim or r.x.width != r.heads * r.dim or self.tables[r.freqs].len != r.rot / 2) return error.ShapeMismatch;
            },
            .attention => |a| {
                try self.checkView(a.q, .scratch);
                try self.checkView(a.out, .scratch);
                const k = try self.checkCache(a.keys);
                const v = try self.checkCache(a.values);
                if (a.kv_heads == 0 or a.heads % a.kv_heads != 0 or a.dim == 0 or a.dim > 512) return error.ShapeMismatch;
                if (a.q.width != a.heads * a.dim or a.out.width != a.q.width or k.width != a.kv_heads * a.dim or v.width != k.width or k.kind != v.kind) return error.ShapeMismatch;
                if (k.kind == .ring and (a.window == 0 or k.horizon + 1 < a.window or v.horizon != k.horizon)) return error.RingTooShort;
                if (a.q.state == a.out.state) return error.AttentionAliasing;
            },
            .relative_attention => |a| {
                try self.checkView(a.q, .scratch);
                try self.checkView(a.bias, .scratch);
                try self.checkView(a.sink, .scratch);
                try self.checkView(a.out, .scratch);
                const k = try self.checkCache(a.keys);
                const v = try self.checkCache(a.values);
                if (a.kv_heads == 0 or a.heads % a.kv_heads != 0 or a.dim == 0 or a.dim > 256) return error.ShapeMismatch;
                if (a.q.width != a.heads * a.dim or a.out.width != a.q.width or
                    (a.relative_extent != 0 and a.bias.width != a.heads * a.relative_extent) or a.sink.width != a.heads or
                    k.width != a.kv_heads * a.dim or v.width != k.width or k.kind != v.kind or !(a.scale > 0)) return error.ShapeMismatch;
                if (k.kind == .ring and (a.window == 0 or k.horizon + 1 < a.window or v.horizon != k.horizon)) return error.RingTooShort;
                if (a.out.state == a.q.state or a.out.state == a.bias.state or a.out.state == a.sink.state) return error.AttentionAliasing;
            },
            .l2_clip => |c| {
                try self.checkView(c.x, .scratch);
                try self.checkView(c.out, .scratch);
                if (c.group == 0 or c.x.width != c.out.width or c.x.width % c.group != 0 or !(c.threshold > 0)) return error.ShapeMismatch;
                if (c.scale) |id| {
                    const scale = try self.weight(id);
                    if (scale.shape.len != 2 or scale.shape[0] != c.x.width / c.group or scale.shape[1] != c.group) return error.ShapeMismatch;
                }
                if (c.x.state == c.out.state and c.x.col != c.out.col) return error.L2ClipAliasing;
            },
            .route_topk => |r| {
                try self.checkView(r.logits, .scratch);
                try self.checkView(r.out, .scratch);
                const correction = try self.weight(r.correction);
                if (r.experts != 128 or r.top_k == 0 or r.top_k > 32 or r.top_k > r.experts or
                    r.logits.width != r.experts or r.out.width != 2 * r.top_k or
                    correction.shape.len != 1 or correction.shape[0] != r.experts or
                    !(r.eps > 0) or !(r.scale > 0)) return error.ShapeMismatch;
                if (r.logits.state == r.out.state) return error.RouterAliasing;
            },
            .expert_gate_up => |e| {
                try self.checkExpertViews(e.x, e.routes, e.out, e.experts, e.top_k);
                if (e.out.width % e.top_k != 0 or !std.math.isFinite(e.gate_cap) or
                    !std.math.isFinite(e.up_cap) or !(e.gate_cap > 0) or !(e.up_cap > 0)) return error.ShapeMismatch;
                try self.checkPackedProjection(e.bank, e.experts, 2 * @as(usize, e.out.width / e.top_k), e.x.width);
            },
            .expert_down => |e| {
                try self.checkExpertViews(e.x, e.routes, e.out, e.experts, e.top_k);
                if (e.x.width % e.top_k != 0) return error.ShapeMismatch;
                try self.checkPackedProjection(e.bank, e.experts, e.out.width, e.x.width / e.top_k);
            },
            .select_halves => |s| {
                try self.checkView(s.x, .scratch);
                try self.checkView(s.out, .scratch);
                if (s.groups == 0 or s.x.width != s.out.width or s.x.width % (2 * s.groups) != 0) return error.ShapeMismatch;
                if (s.x.state == s.out.state and s.x.col < s.out.col + s.out.width and s.out.col < s.x.col + s.x.width) return error.ShapeMismatch;
            },
            .gated_delta_net => |g| {
                try self.checkView(g.qkv, .scratch);
                try self.checkView(g.a, .scratch);
                try self.checkView(g.b, .scratch);
                try self.checkView(g.out, .scratch);
                const st = try self.checkCache(g.state);
                if (st.kind != .recurrent) return error.InvalidCacheKind;
                if (g.k_heads == 0 or g.v_heads == 0 or g.k_head_dim == 0 or g.v_head_dim == 0 or
                    g.v_heads % g.k_heads != 0 or g.k_head_dim > 256 or
                    !std.math.isFinite(g.eps) or g.eps <= 0) return error.ShapeMismatch;
                const key_dim = @as(u64, g.k_heads) * g.k_head_dim;
                const val_dim = @as(u64, g.v_heads) * g.v_head_dim;
                const state_words = @as(u64, g.v_heads) * g.v_head_dim * g.k_head_dim;
                if (g.qkv.width != 2 * key_dim + val_dim or g.a.width != g.v_heads or g.b.width != g.v_heads or
                    g.out.width != val_dim or st.width != state_words) return error.ShapeMismatch;
                const a_log = try self.weight(g.a_log);
                const dt_bias = try self.weight(g.dt_bias);
                if (a_log.raw_dtype != null or dt_bias.raw_dtype != null or
                    !std.mem.eql(usize, a_log.shape, &.{g.v_heads}) or
                    !std.mem.eql(usize, dt_bias.shape, &.{g.v_heads})) return error.ShapeMismatch;
            },
            .argmax => |a| {
                try self.checkView(a.x, .scratch);
                try self.checkView(a.out, .scratch);
                if (a.out.width != 1) return error.ShapeMismatch;
            },
            .loopcd_contrast => |l| {
                try self.checkView(l.strong, .scratch);
                try self.checkView(l.weak, .scratch);
                try self.checkView(l.out, .scratch);
                if (l.strong.width == 0 or l.strong.width != l.weak.width or l.strong.width != l.out.width) return error.ShapeMismatch;
                if (!std.math.isFinite(l.omega) or l.omega < 0) return error.InvalidScale;
                const bad_strong = l.out.state == l.strong.state and l.out.col != l.strong.col;
                const bad_weak = l.out.state == l.weak.state and l.out.col != l.weak.col;
                if (bad_strong or bad_weak) return error.LoopCdContrastAliasing;
            },
        }
    }
    fn checkExpertViews(self: Program, x: View, routes: View, out: View, experts: u32, top_k: u32) !void {
        try self.checkView(x, .scratch);
        try self.checkView(routes, .scratch);
        try self.checkView(out, .scratch);
        if (top_k == 0 or top_k > 32 or top_k > experts or experts == 0 or routes.width != 2 * top_k) return error.ShapeMismatch;
        if (out.state == x.state or out.state == routes.state) return error.ExpertAliasing;
    }

    fn checkPackedTensor(self: Program, id: WeightId, dtype: RawWeightDType, shape: []const usize) !void {
        const w = try self.weight(id);
        if (w.raw_dtype != dtype or !std.mem.eql(usize, w.shape, shape)) return error.InvalidPackedProjection;
    }

    fn checkPackedProjection(self: Program, bank: PackedProjection, experts: u32, n: usize, k: u32) !void {
        if (bank.amax) |amax| {
            if (k % 16 != 0) return error.InvalidPackedProjection;
            try self.checkPackedTensor(bank.payload, .u8, &.{ experts, n, k / 2 });
            try self.checkPackedTensor(bank.scales, .f8_e4m3, &.{ experts, n, k / 16 });
            try self.checkPackedTensor(amax, .f32, &.{experts});
            try self.checkPackedTensor(bank.denominator orelse return error.InvalidPackedProjection, .f32, &.{1});
        } else {
            if (bank.denominator != null or k % 32 != 0) return error.InvalidPackedProjection;
            try self.checkPackedTensor(bank.payload, .f8_e4m3, &.{ experts, n, k });
            try self.checkPackedTensor(bank.scales, .u8, &.{ experts, n, k / 32 });
        }
    }
};

/// Incremental construction. All storage lives in the program's arena.
pub const Builder = struct {
    arena: std.heap.ArenaAllocator,
    weights: std.ArrayList(Weight) = .empty,
    states: std.ArrayList(State) = .empty,
    tables: std.ArrayList([]const f32) = .empty,
    ops: std.ArrayList(Op) = .empty,
    traces: std.ArrayList(Trace) = .empty,
    draft_final_norm: ?Trace = null,

    pub fn init(gpa: std.mem.Allocator) Builder {
        return .{ .arena = std.heap.ArenaAllocator.init(gpa) };
    }
    pub fn deinit(self: *Builder) void {
        self.arena.deinit();
    }
    pub fn allocator(self: *Builder) std.mem.Allocator {
        return self.arena.allocator();
    }
    pub fn print(self: *Builder, comptime fmt: []const u8, args: anytype) ![]const u8 {
        return std.fmt.allocPrint(self.allocator(), fmt, args);
    }
    pub fn weight(self: *Builder, name: []const u8, shape: []const usize) !WeightId {
        return self.addWeight(name, shape, null);
    }
    pub fn rawWeight(self: *Builder, name: []const u8, shape: []const usize, dtype: RawWeightDType) !WeightId {
        return self.addWeight(name, shape, dtype);
    }
    fn addWeight(self: *Builder, name: []const u8, shape: []const usize, raw_dtype: ?RawWeightDType) !WeightId {
        for (self.weights.items, 0..) |w, i| if (std.mem.eql(u8, w.name, name)) {
            if (!std.mem.eql(usize, w.shape, shape)) return error.WeightShapeConflict;
            if (w.raw_dtype != raw_dtype) return error.WeightStorageConflict;
            return @intCast(i);
        };
        const a = self.allocator();
        try self.weights.append(a, .{ .name = try a.dupe(u8, name), .shape = try a.dupe(usize, shape), .raw_dtype = raw_dtype });
        return @intCast(self.weights.items.len - 1);
    }
    pub fn state(self: *Builder, s: State) !StateId {
        const a = self.allocator();
        var copy = s;
        copy.name = try a.dupe(u8, s.name);
        try self.states.append(a, copy);
        return @intCast(self.states.items.len - 1);
    }
    /// A whole-state view.
    pub fn view(self: *Builder, id: StateId) View {
        return .{ .state = id, .width = self.states.items[id].width };
    }
    pub fn table(self: *Builder, values: []const f32) !TableId {
        const a = self.allocator();
        try self.tables.append(a, try a.dupe(f32, values));
        return @intCast(self.tables.items.len - 1);
    }
    pub fn op(self: *Builder, o: Op) !void {
        try self.ops.append(self.allocator(), o);
    }
    pub fn trace(self: *Builder, v: View) !void {
        if (self.ops.items.len == 0) return error.InvalidTrace;
        try self.traces.append(self.allocator(), .{ .after_op = @intCast(self.ops.items.len - 1), .view = v });
    }
    pub fn setDraftFinalNorm(self: *Builder, v: View) !void {
        if (self.ops.items.len == 0) return error.InvalidTrace;
        self.draft_final_norm = .{ .after_op = @intCast(self.ops.items.len - 1), .view = v };
    }
    pub fn mark(self: *Builder) u32 {
        return @intCast(self.ops.items.len);
    }
    /// Consumes the builder on success; on a validation error the builder keeps its storage.
    pub fn finish(self: *Builder, meta: Meta, head_start: u32, tokens: View, logits: View, next: View) !Program {
        const a = self.allocator();
        const owned: Meta = .{ .model_type = try a.dupe(u8, meta.model_type), .vocab = meta.vocab, .eos = try a.dupe(u32, meta.eos), .max_context = meta.max_context };
        const program: Program = .{
            .arena = self.arena,
            .meta = owned,
            .weights = self.weights.items,
            .states = self.states.items,
            .tables = self.tables.items,
            .ops = self.ops.items,
            .head_start = head_start,
            .traces = self.traces.items,
            .draft_final_norm = self.draft_final_norm,
            .tokens = tokens,
            .logits = logits,
            .next = next,
        };
        try program.validate();
        self.* = undefined;
        return program;
    }
};

test "state capacities and snapshot extents by kind" {
    const ring: State = .{ .name = "r", .kind = .ring, .width = 4, .horizon = 6 };
    try std.testing.expectEqual(@as(u32, 6 + 8), ring.capacity(8, 1000));
    try std.testing.expectEqual(@as(u32, 5), ring.validRows(8, 1000, 5));
    try std.testing.expectEqual(@as(u32, 14), ring.validRows(8, 1000, 99));
    try std.testing.expectEqual(@as(u32, 10), ring.capacity(8, 10));
    const seq: State = .{ .name = "s", .kind = .sequence, .width = 4 };
    try std.testing.expectEqual(@as(u32, 100), seq.capacity(8, 100));
    try std.testing.expectEqual(@as(u32, 37), seq.validRows(8, 100, 37));
    const scratch: State = .{ .name = "x", .kind = .scratch, .width = 4, .rows = 1 };
    try std.testing.expectEqual(@as(u32, 1), scratch.capacity(8, 100));
    try std.testing.expectEqual(@as(u32, 0), scratch.validRows(8, 100, 50));
}

fn tinyProgram(gpa: std.mem.Allocator, bad_linear: bool) !Program {
    var b = Builder.init(gpa);
    errdefer b.deinit();
    const tok = try b.state(.{ .name = "tokens", .kind = .tokens, .width = 1 });
    _ = try b.state(.{ .name = "position", .kind = .position, .width = 1 });
    const x = try b.state(.{ .name = "x", .kind = .scratch, .width = 8 });
    const y = try b.state(.{ .name = "y", .kind = .scratch, .width = 16 });
    const logits = try b.state(.{ .name = "logits", .kind = .scratch, .width = 32 });
    const next = try b.state(.{ .name = "next", .kind = .scratch, .width = 1 });
    const kc = try b.state(.{ .name = "k", .kind = .ring, .width = 8, .horizon = 3 });
    const emb = try b.weight("emb", &.{ 32, 8 });
    const w = try b.weight("w", &.{ 16, if (bad_linear) 7 else 8 });
    const nw = try b.weight("norm", &.{8});
    try b.op(.{ .embed = .{ .table = emb, .tokens = b.view(tok), .out = b.view(x) } });
    try b.op(.{ .rmsnorm = .{ .x = b.view(x), .weight = nw, .out = b.view(x), .group = 8, .eps = 1e-5 } });
    try b.trace(b.view(x));
    try b.op(.{ .store = .{ .src = b.view(x), .dst = kc } });
    try b.op(.{ .linear = .{ .w = w, .x = b.view(x), .out = b.view(y) } });
    try b.op(.{ .binary = .{ .kind = .mul, .a = .{ .state = y, .width = 8 }, .b = .{ .state = y, .col = 8, .width = 8 }, .out = b.view(x), .a_act = .silu } });
    const head = b.mark();
    try b.op(.{ .linear = .{ .w = emb, .x = b.view(x), .out = b.view(logits) } });
    try b.op(.{ .argmax = .{ .x = b.view(logits), .out = b.view(next) } });
    return b.finish(.{ .model_type = "tiny", .vocab = 32, .eos = &.{0}, .max_context = 64 }, head, b.view(tok), b.view(logits), b.view(next));
}

test "builder emits a validated program and deduplicates weights" {
    var p = try tinyProgram(std.testing.allocator, false);
    defer p.deinit();
    try std.testing.expectEqual(@as(usize, 3), p.weights.len);
    try std.testing.expectEqual(@as(usize, 7), p.ops.len);
    try std.testing.expectEqual(@as(u32, 5), p.head_start);
    try std.testing.expectEqual(@as(u32, 1), p.traces[0].after_op);
    try std.testing.expectEqualStrings("argmax", p.ops[6].name());
    try std.testing.expectEqual(@as(?StateId, 1), p.positionState());
}

test "validation rejects shape mismatches, bad views and missing position" {
    try std.testing.expectError(error.ShapeMismatch, tinyProgram(std.testing.allocator, true));
    var b = Builder.init(std.testing.allocator);
    defer b.deinit();
    const tok = try b.state(.{ .name = "tokens", .kind = .tokens, .width = 1 });
    const x = try b.state(.{ .name = "x", .kind = .scratch, .width = 8 });
    try std.testing.expectError(error.InvalidPositionState, b.finish(.{ .model_type = "t", .vocab = 1, .eos = &.{}, .max_context = 1 }, 0, b.view(tok), b.view(x), b.view(x)));
    var c = Builder.init(std.testing.allocator);
    defer c.deinit();
    const t2 = try c.state(.{ .name = "tokens", .kind = .tokens, .width = 1 });
    _ = try c.state(.{ .name = "position", .kind = .position, .width = 1 });
    const x2 = try c.state(.{ .name = "x", .kind = .scratch, .width = 8 });
    try c.op(.{ .binary = .{ .kind = .add, .a = .{ .state = x2, .col = 4, .width = 8 }, .b = c.view(x2), .out = c.view(x2) } });
    try std.testing.expectError(error.ViewOutOfBounds, c.finish(.{ .model_type = "t", .vocab = 8, .eos = &.{}, .max_context = 4 }, 1, c.view(t2), c.view(x2), .{ .state = x2, .width = 1 }));
}


fn depthAttentionProgram(gpa: std.mem.Allocator, bank_width: u32, query_index: u32) !Program {
    var b = Builder.init(gpa);
    errdefer b.deinit();
    const tokens = try b.state(.{ .name = "tokens", .kind = .tokens, .width = 1 });
    _ = try b.state(.{ .name = "position", .kind = .position, .width = 1 });
    const bank = try b.state(.{ .name = "bank", .kind = .scratch, .width = bank_width });
    const partial = try b.state(.{ .name = "partial", .kind = .scratch, .width = 4 });
    const out = try b.state(.{ .name = "out", .kind = .scratch, .width = 4 });
    const logits = try b.state(.{ .name = "logits", .kind = .scratch, .width = 1 });
    const next = try b.state(.{ .name = "next", .kind = .scratch, .width = 1 });
    const query = try b.weight("depth_query", &.{ 2, 4 });
    try b.op(.{ .depth_attention = .{ .bank = b.view(bank), .partial = b.view(partial), .query = query, .query_index = query_index, .out = b.view(out), .scale = 0.5 } });
    return b.finish(.{ .model_type = "tiny_depth", .vocab = 1, .eos = &.{}, .max_context = 1 }, b.mark(), b.view(tokens), b.view(logits), b.view(next));
}

test "depth attention validates packed bank geometry and query index" {
    var p = try depthAttentionProgram(std.testing.allocator, 8, 1);
    defer p.deinit();
    try std.testing.expectError(error.ShapeMismatch, depthAttentionProgram(std.testing.allocator, 7, 1));
    try std.testing.expectError(error.ShapeMismatch, depthAttentionProgram(std.testing.allocator, 8, 2));
}


fn relativeAttentionProgram(gpa: std.mem.Allocator, bad_bias: bool) !Program {
    var b = Builder.init(gpa);
    errdefer b.deinit();
    const tokens = try b.state(.{ .name = "tokens", .kind = .tokens, .width = 1 });
    _ = try b.state(.{ .name = "position", .kind = .position, .width = 1 });
    const q = try b.state(.{ .name = "q", .kind = .scratch, .width = 8 });
    const bias = try b.state(.{ .name = "bias", .kind = .scratch, .width = if (bad_bias) 5 else 6 });
    const sink = try b.state(.{ .name = "sink", .kind = .scratch, .width = 2 });
    const out = try b.state(.{ .name = "out", .kind = .scratch, .width = 8 });
    const logits = try b.state(.{ .name = "logits", .kind = .scratch, .width = 1 });
    const next = try b.state(.{ .name = "next", .kind = .scratch, .width = 1 });
    const keys = try b.state(.{ .name = "keys", .kind = .ring, .width = 4, .horizon = 1 });
    const values = try b.state(.{ .name = "values", .kind = .ring, .width = 4, .horizon = 1 });
    try b.op(.{ .relative_attention = .{ .q = b.view(q), .bias = b.view(bias), .sink = b.view(sink), .keys = keys, .values = values, .out = b.view(out), .heads = 2, .kv_heads = 1, .dim = 4, .relative_extent = 3, .window = 2, .scale = 0.5 } });
    return b.finish(.{ .model_type = "tiny_relative", .vocab = 1, .eos = &.{}, .max_context = 8 }, b.mark(), b.view(tokens), b.view(logits), b.view(next));
}

test "relative attention validates bias shape and ring horizon" {
    var p = try relativeAttentionProgram(std.testing.allocator, false);
    defer p.deinit();
    try std.testing.expectError(error.ShapeMismatch, relativeAttentionProgram(std.testing.allocator, true));
}


fn routeTopKProgram(gpa: std.mem.Allocator, bad_width: bool) !Program {
    var b = Builder.init(gpa);
    errdefer b.deinit();
    const tokens = try b.state(.{ .name = "tokens", .kind = .tokens, .width = 1 });
    _ = try b.state(.{ .name = "position", .kind = .position, .width = 1 });
    const logits = try b.state(.{ .name = "logits", .kind = .scratch, .width = 128 });
    const out = try b.state(.{ .name = "routes", .kind = .scratch, .width = if (bad_width) 5 else 6 });
    const final_logits = try b.state(.{ .name = "final_logits", .kind = .scratch, .width = 1 });
    const next = try b.state(.{ .name = "next", .kind = .scratch, .width = 1 });
    const bias = try b.weight("router_bias", &.{128});
    try b.op(.{ .route_topk = .{ .logits = b.view(logits), .correction = bias, .out = b.view(out), .experts = 128, .top_k = 3, .eps = 1e-20, .scale = 8 } });
    return b.finish(.{ .model_type = "tiny_route", .vocab = 1, .eos = &.{}, .max_context = 1 }, b.mark(), b.view(tokens), b.view(final_logits), b.view(next));
}

test "route top-k validates packed output width" {
    var p = try routeTopKProgram(std.testing.allocator, false);
    defer p.deinit();
    try std.testing.expectError(error.ShapeMismatch, routeTopKProgram(std.testing.allocator, true));
}


fn l2ClipProgram(gpa: std.mem.Allocator, bad_group: bool, bad_scale: bool) !Program {
    var b = Builder.init(gpa);
    errdefer b.deinit();
    const tokens = try b.state(.{ .name = "tokens", .kind = .tokens, .width = 1 });
    _ = try b.state(.{ .name = "position", .kind = .position, .width = 1 });
    const x = try b.state(.{ .name = "x", .kind = .scratch, .width = 8 });
    const out = try b.state(.{ .name = "out", .kind = .scratch, .width = 8 });
    const logits = try b.state(.{ .name = "logits", .kind = .scratch, .width = 1 });
    const next = try b.state(.{ .name = "next", .kind = .scratch, .width = 1 });
    const scale = try b.weight("clip.scale", if (bad_scale) &.{ 2, 3 } else &.{ 2, 4 });
    try b.op(.{ .l2_clip = .{ .x = b.view(x), .out = b.view(out), .group = if (bad_group) 3 else 4, .threshold = 2, .scale = scale } });
    return b.finish(.{ .model_type = "tiny_clip", .vocab = 1, .eos = &.{}, .max_context = 1 }, b.mark(), b.view(tokens), b.view(logits), b.view(next));
}

test "L2 clipping validates per-group geometry" {
    var p = try l2ClipProgram(std.testing.allocator, false, false);
    defer p.deinit();
    try std.testing.expectError(error.ShapeMismatch, l2ClipProgram(std.testing.allocator, true, false));
    try std.testing.expectError(error.ShapeMismatch, l2ClipProgram(std.testing.allocator, false, true));
}


test "Inkling short convolution validates ring and tensor geometry" {
    var b = Builder.init(std.testing.allocator);
    errdefer b.deinit();
    const tokens = try b.state(.{ .name = "tokens", .kind = .tokens, .width = 1 });
    _ = try b.state(.{ .name = "position", .kind = .position, .width = 1 });
    const residual = try b.state(.{ .name = "residual", .kind = .scratch, .width = 4 });
    const out = try b.state(.{ .name = "out", .kind = .scratch, .width = 4 });
    const logits = try b.state(.{ .name = "logits", .kind = .scratch, .width = 1 });
    const next = try b.state(.{ .name = "next", .kind = .scratch, .width = 1 });
    const ring = try b.state(.{ .name = "conv_state", .kind = .ring, .width = 4, .horizon = 2 });
    const weight = try b.weight("conv_weight", &.{ 4, 3 });
    try b.op(.{ .inkling_short_conv = .{ .src = ring, .weight = weight, .residual = b.view(residual), .out = b.view(out), .taps = 3 } });
    var p = try b.finish(.{ .model_type = "tiny_inkling_conv", .vocab = 1, .eos = &.{}, .max_context = 8 }, b.mark(), b.view(tokens), b.view(logits), b.view(next));
    defer p.deinit();

    var bad = Builder.init(std.testing.allocator);
    defer bad.deinit();
    const bt = try bad.state(.{ .name = "tokens", .kind = .tokens, .width = 1 });
    _ = try bad.state(.{ .name = "position", .kind = .position, .width = 1 });
    const br = try bad.state(.{ .name = "residual", .kind = .scratch, .width = 4 });
    const bo = try bad.state(.{ .name = "out", .kind = .scratch, .width = 4 });
    const bl = try bad.state(.{ .name = "logits", .kind = .scratch, .width = 1 });
    const bn = try bad.state(.{ .name = "next", .kind = .scratch, .width = 1 });
    const short_ring = try bad.state(.{ .name = "conv_state", .kind = .ring, .width = 4, .horizon = 1 });
    const bw = try bad.weight("conv_weight", &.{ 4, 3 });
    try bad.op(.{ .inkling_short_conv = .{ .src = short_ring, .weight = bw, .residual = bad.view(br), .out = bad.view(bo), .taps = 3 } });
    try std.testing.expectError(error.RingTooShort, bad.finish(.{ .model_type = "tiny_inkling_conv", .vocab = 1, .eos = &.{}, .max_context = 8 }, bad.mark(), bad.view(bt), bad.view(bl), bad.view(bn)));
}


fn dynamicConvProgram(gpa: std.mem.Allocator, bad_dynamic: bool) !Program {
    var b = Builder.init(gpa);
    errdefer b.deinit();
    const tokens = try b.state(.{ .name = "tokens", .kind = .tokens, .width = 1 });
    _ = try b.state(.{ .name = "position", .kind = .position, .width = 1 });
    const dynamic = try b.state(.{ .name = "dynamic", .kind = .scratch, .width = if (bad_dynamic) 5 else 6 });
    const residual = try b.state(.{ .name = "residual", .kind = .scratch, .width = 4 });
    const out = try b.state(.{ .name = "out", .kind = .scratch, .width = 4 });
    const logits = try b.state(.{ .name = "logits", .kind = .scratch, .width = 1 });
    const next = try b.state(.{ .name = "next", .kind = .scratch, .width = 1 });
    const ring = try b.state(.{ .name = "conv_state", .kind = .ring, .width = 4, .horizon = 2 });
    const weight = try b.weight("static_weight", &.{ 3, 4 });
    try b.op(.{ .inkling_dynamic_conv = .{ .src = ring, .static_weight = weight, .dynamic = b.view(dynamic), .residual = b.view(residual), .out = b.view(out), .heads = 2, .taps = 3 } });
    return b.finish(.{ .model_type = "tiny_inkling_dynamic_conv", .vocab = 1, .eos = &.{}, .max_context = 8 }, b.mark(), b.view(tokens), b.view(logits), b.view(next));
}

test "Inkling dynamic convolution validates projected filter shape" {
    var p = try dynamicConvProgram(std.testing.allocator, false);
    defer p.deinit();
    try std.testing.expectError(error.ShapeMismatch, dynamicConvProgram(std.testing.allocator, true));
}


fn messageMergeProgram(gpa: std.mem.Allocator, bad_gates: bool) !Program {
    var b = Builder.init(gpa);
    errdefer b.deinit();
    const tokens = try b.state(.{ .name = "tokens", .kind = .tokens, .width = 1 });
    _ = try b.state(.{ .name = "position", .kind = .position, .width = 1 });
    const hidden = try b.state(.{ .name = "hidden", .kind = .scratch, .width = 4 });
    const message = try b.state(.{ .name = "message", .kind = .scratch, .width = 4 });
    const gates = try b.state(.{ .name = "gates", .kind = .scratch, .width = if (bad_gates) 7 else 8 });
    const out = try b.state(.{ .name = "out", .kind = .scratch, .width = 4 });
    const logits = try b.state(.{ .name = "logits", .kind = .scratch, .width = 1 });
    const next = try b.state(.{ .name = "next", .kind = .scratch, .width = 1 });
    try b.op(.{ .inkling_message_merge = .{ .hidden = b.view(hidden), .message = b.view(message), .gates = b.view(gates), .out = b.view(out) } });
    return b.finish(.{ .model_type = "tiny_inkling_message", .vocab = 1, .eos = &.{}, .max_context = 1 }, b.mark(), b.view(tokens), b.view(logits), b.view(next));
}

test "Inkling message merge validates the packed gate width" {
    var p = try messageMergeProgram(std.testing.allocator, false);
    defer p.deinit();
    try std.testing.expectError(error.ShapeMismatch, messageMergeProgram(std.testing.allocator, true));
}

test "raw weights preserve physical shape and reject storage conflicts" {
    var b = Builder.init(std.testing.allocator);
    defer b.deinit();
    const id = try b.rawWeight("experts.w13", &.{ 128, 2304, 1152 }, .u8);
    try std.testing.expectEqual(RawWeightDType.u8, b.weights.items[id].raw_dtype.?);
    try std.testing.expectError(error.WeightStorageConflict, b.weight("experts.w13", &.{ 128, 2304, 1152 }));
}
