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
        return self.kind == .sequence or self.kind == .ring;
    }
    /// Rows allocated for this state under a given batch and context.
    pub fn capacity(self: State, batch: u32, context: u32) u32 {
        return switch (self.kind) {
            .tokens => batch,
            .scratch => if (self.rows != 0) @min(self.rows, batch) else batch,
            .sequence => context,
            .ring => @min(self.horizon + batch, context),
            .position => 1,
        };
    }
    /// Rows that hold information after `position` tokens: the snapshot extent.
    pub fn validRows(self: State, batch: u32, context: u32, position: u32) u32 {
        return switch (self.kind) {
            .sequence, .ring => @min(position, self.capacity(batch, context)),
            .position => 1,
            else => 0,
        };
    }
};

pub const Weight = struct {
    name: []const u8,
    /// Logical shape: quantized matrices are [rows, cols] of dequantized values.
    shape: []const usize,
};

/// A column window of a state's rows. Row stride is always the state's width.
pub const View = struct {
    state: StateId,
    col: u32 = 0,
    width: u32,
};

pub const Act = enum(u8) { identity = 0, silu = 1, sigmoid = 2 };
pub const BinaryKind = enum(u8) { add = 0, mul = 1 };
/// Optional output rounding, used where a reference materialises a narrower dtype.
pub const Round = enum(u8) { none = 0, bf16 = 1 };

pub const Op = union(enum) {
    /// out[r] = table[tokens[r]] (dense or affine-quantized table).
    embed: struct { table: WeightId, tokens: View, out: View, round: Round = .none },
    /// RMSNorm over consecutive groups of `group` words: whole vector or per head.
    rmsnorm: struct { x: View, weight: ?WeightId, out: View, group: u32, eps: f32 },
    /// out[r] = W x[r]; W is [out.width, x.width], dense or affine-quantized.
    linear: struct { w: WeightId, x: View, out: View },
    /// out = act_a(a) (+|*) act_b(b[i / b_every]).
    binary: struct { kind: BinaryKind, a: View, b: View, out: View, a_act: Act = .identity, b_act: Act = .identity, b_every: u32 = 1 },
    /// Write each row to its position's slot of a sequence/ring state.
    store: struct { src: View, dst: StateId },
    /// Depthwise causal conv over a ring of past inputs; weight is [channels, taps].
    /// Positions before 0 read zeros. The ring must already hold this batch's rows.
    conv: struct { src: StateId, weight: WeightId, out: View, taps: u32 },
    /// Rotate-half RoPE on pairs (i, i + rot/2), i < rot/2, of each head, with a
    /// per-pair frequency table (zero = unrotated, so partial rotation is a table).
    rope: struct { x: View, heads: u32, dim: u32, rot: u32, freqs: TableId, fast_trig: bool = false },
    /// Grouped-query causal attention reading cached K/V of positions
    /// [max(0, p - window + 1), p] (window 0 = unbounded) for row position p.
    attention: struct { q: View, keys: StateId, values: StateId, out: View, heads: u32, kv_heads: u32, dim: u32, window: u32, scale: f32 },
    /// out[r] = lowest index of the maximum of x[r]; all-u32 sentinel on non-finite input.
    argmax: struct { x: View, out: View },

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
                if (a.kv_heads == 0 or a.heads % a.kv_heads != 0 or a.dim == 0 or a.dim > 256) return error.ShapeMismatch;
                if (a.q.width != a.heads * a.dim or a.out.width != a.q.width or k.width != a.kv_heads * a.dim or v.width != k.width or k.kind != v.kind) return error.ShapeMismatch;
                if (k.kind == .ring and (a.window == 0 or k.horizon + 1 < a.window or v.horizon != k.horizon)) return error.RingTooShort;
                if (a.q.state == a.out.state) return error.AttentionAliasing;
            },
            .argmax => |a| {
                try self.checkView(a.x, .scratch);
                try self.checkView(a.out, .scratch);
                if (a.out.width != 1) return error.ShapeMismatch;
            },
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
        for (self.weights.items, 0..) |w, i| if (std.mem.eql(u8, w.name, name)) {
            if (!std.mem.eql(usize, w.shape, shape)) return error.WeightShapeConflict;
            return @intCast(i);
        };
        const a = self.allocator();
        try self.weights.append(a, .{ .name = try a.dupe(u8, name), .shape = try a.dupe(usize, shape) });
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
