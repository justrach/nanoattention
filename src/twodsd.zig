//! Two-dimensional speculative decoding (2DSD) for looped language models.
//!
//! CPU scheduler matching Kim, Chavan, Seo, Hu, Lee, Kang, Mahoney, Keutzer,
//! Hooper and Gholami, *2DSD: Two-Dimensional Speculative Decoding for Looped
//! Language Models* (2026), paper/2DSD.pdf in https://github.com/SqueezeAILab/2DSD
//! (announced https://x.com/minseokim0525/status/2123533353823662556).
//!
//! Drafts along tokens (block size `k`) and along depth (early-loop review at
//! `d1`, optional promotion/fork at `d2`, lossless verify at loop `R`). Last-loop
//! KV sharing (`shared_last`) is accounted here; Metal overlapping decode is not
//! wired. No GPU, checkpoint, or default-path behaviour.
const std = @import("std");
const history_draft = @import("speculative.zig");

pub const max_block: u32 = 16;
pub const max_loops: u32 = 8;
pub const tiny_vocab: u32 = 97;
pub const tiny_R: u32 = 4;

pub const Stage = enum { stage1, stage2a, stage2b };
pub const KvPolicy = enum { full, shared_last };
pub const DrafterKind = enum { none, oracle, oracle_corrupt, history };
pub const TinyKind = enum { growing, promotion, fork, nested, wide, markov };

pub const Options = struct {
    k: u32 = 1,
    stage: Stage = .stage2b,
    d1: u32 = 1,
    d2: u32 = 2,
    temperature: f32 = 0,
    top_p: f32 = 1,
    proposal_top_p: f32 = 1,
    seed: u64 = 0,
    kv_policy: KvPolicy = .full,
    kv_window: u32 = 0,
    drafter: DrafterKind = .none,
    record_trace: bool = false,

    pub fn sampling(self: Options) bool {
        return self.temperature > 0;
    }
};

pub fn parseStage(s: []const u8) ?Stage {
    if (std.mem.eql(u8, s, "stage1")) return .stage1;
    if (std.mem.eql(u8, s, "stage2a")) return .stage2a;
    if (std.mem.eql(u8, s, "stage2b")) return .stage2b;
    return null;
}

pub fn parseKvPolicy(s: []const u8) ?KvPolicy {
    if (std.mem.eql(u8, s, "full")) return .full;
    if (std.mem.eql(u8, s, "shared_last")) return .shared_last;
    return null;
}

pub fn parseEnable(s: []const u8) ?bool {
    if (std.mem.eql(u8, s, "0")) return false;
    if (std.mem.eql(u8, s, "1")) return true;
    return null;
}

/// Opt-in CLI state. Defaults leave generate/session/run/chat unchanged.
pub const Cli = struct {
    enabled: bool = false,
    k: u32 = 1,
    stage: Stage = .stage2b,
    d1: u32 = 1,
    d2: u32 = 2,
    kv_policy: KvPolicy = .full,
    kv_window: u32 = 0,

    pub fn anyTuning(self: Cli) bool {
        return self.k != 1 or self.stage != .stage2b or self.d1 != 1 or self.d2 != 2 or
            self.kv_policy != .full or self.kv_window != 0;
    }

    /// Fail closed: Flint/LFM are not looped; Inkling looping is sequential per token.
    pub fn apply(self: Cli, model_type: []const u8) !void {
        if (!self.enabled) {
            if (self.anyTuning()) return error.TwoDsdTuningRequiresEnable;
            return;
        }
        try enableOrReject(model_type);
    }
};

pub fn loopedFamily(model_type: []const u8) bool {
    return std.mem.eql(u8, model_type, "inkling2_loop");
}

/// Opt-in Metal generate/session path. Flint/LFM are not looped; Inkling is
/// looped but the megakernel still runs every pass of one token before the next.
pub fn enableOrReject(model_type: []const u8) error{ TwoDsdRequiresLoopedModel, TwoDsdOverlappingDecodeNotWired }!void {
    if (!loopedFamily(model_type)) return error.TwoDsdRequiresLoopedModel;
    return error.TwoDsdOverlappingDecodeNotWired;
}

pub fn validate(opts: Options, R: u32) !void {
    if (opts.k < 1 or opts.k > max_block) return error.InvalidBlockSize;
    if (R < 1 or R > max_loops) return error.InvalidLoopCount;
    if (opts.d1 < 1 or opts.d1 > R) return error.InvalidDepth;
    if (opts.stage != .stage1 and !(opts.d1 < opts.d2 and opts.d2 < R)) return error.InvalidStage2Depths;
    if (opts.temperature < 0 or !std.math.isFinite(opts.temperature)) return error.InvalidTemperature;
    if (!(opts.top_p > 0 and opts.top_p <= 1) or !(opts.proposal_top_p > 0 and opts.proposal_top_p <= 1))
        return error.InvalidTopP;
    if ((opts.top_p < 1 or opts.proposal_top_p < 1) and opts.temperature == 0) return error.TopPRequiresSampling;
    if (opts.k > 1 and opts.drafter == .none) return error.DrafterRequired;
    if (opts.k == 1 and opts.drafter != .none) return error.DrafterRequiresBlock;
    try checkTreeWindow(R, opts.k, opts.kv_policy, opts.kv_window);
}

/// 2DSD keeps up to R * k tokens in flight; all must fit in the shared-last window.
pub fn checkTreeWindow(R: u32, k: u32, policy: KvPolicy, window: u32) !void {
    if (policy != .shared_last) return;
    if (window < 1) return error.InvalidWindow;
    if (@as(u64, R) * k > window) return error.WindowTooSmall;
}

/// History-KV bytes of `layers` attention layers over `seq` committed positions.
/// `kv_width` is elements per position (K and V together). `elem` is bytes/element.
/// `shared_last` stores one last-loop history plus an in-flight window per loop.
pub fn historyKvBytes(layers: u32, R: u32, seq: u32, kv_width: u32, elem: u32, policy: KvPolicy, window: u32) u64 {
    const row: u64 = @as(u64, layers) * seq * kv_width * elem;
    return switch (policy) {
        .full => row * R,
        .shared_last => row + @as(u64, layers) * window * kv_width * elem * R,
    };
}

/// Inkling 2 Mini loop-block visits: 12 physical layers × R passes. Prelude/coda
/// (12 layers) are not looped and are unchanged by last-loop sharing.
pub const inkling_loop_layers: u32 = 12;
pub const inkling_R: u32 = 3;

pub fn inklingLoopKvBytes(seq: u32, kv_width: u32, elem: u32, R: u32, policy: KvPolicy, window: u32) u64 {
    return historyKvBytes(inkling_loop_layers, R, seq, kv_width, elem, policy, window);
}

pub fn oracleNext(tokens: []const u32, causal: bool) u32 {
    return oracleNextR(tokens, causal, tiny_R);
}

pub fn oracleNextR(tokens: []const u32, causal: bool, loops: u32) u32 {
    var values: [512]u32 = undefined;
    const n = tokens.len;
    std.debug.assert(n > 0 and n <= values.len);
    @memcpy(values[0..n], tokens);
    if (causal) {
        var tmp: [512]u32 = undefined;
        for (0..loops) |loop| {
            var prefix_sum: u32 = 0;
            for (values[0..n], tmp[0..n]) |value, *out| {
                out.* = (3 * value + 2 * prefix_sum + @as(u32, @intCast(loop)) + 1) % tiny_vocab;
                prefix_sum += value;
            }
            @memcpy(values[0..n], tmp[0..n]);
        }
    }
    return (values[n - 1] + 1) % tiny_vocab;
}

pub fn arGreedy(prompt: []const u32, count: usize, causal: bool, eos: []const u32, out: []u32) usize {
    var sequence: [1024]u32 = undefined;
    std.debug.assert(prompt.len + count <= sequence.len);
    @memcpy(sequence[0..prompt.len], prompt);
    var len = prompt.len;
    var n: usize = 0;
    while (n < count) : (n += 1) {
        const token = oracleNext(sequence[0..len], causal);
        out[n] = token;
        sequence[len] = token;
        len += 1;
        if (std.mem.indexOfScalar(u32, eos, token) != null) return n + 1;
    }
    return n;
}

const Role = enum { main, frozen, fork };
const Origin = enum {
    initial,
    raw_draft,
    d1_reject,
    d1_accepted,
    d1_frontier,
    promotion,
    promotion_reject,
    promotion_accepted,
    promotion_frontier,
    d2_fork,
    bonus,
};

fn originName(o: Origin) []const u8 {
    return switch (o) {
        .initial => "initial",
        .raw_draft => "raw_draft",
        .d1_reject => "d1_reject",
        .d1_accepted => "d1_accepted",
        .d1_frontier => "d1_frontier",
        .promotion => "promotion",
        .promotion_reject => "promotion_reject",
        .promotion_accepted => "promotion_accepted",
        .promotion_frontier => "promotion_frontier",
        .d2_fork => "d2_fork",
        .bonus => "bonus",
    };
}

fn roleName(r: Role) []const u8 {
    return switch (r) {
        .main => "main",
        .frozen => "frozen",
        .fork => "fork",
    };
}

const Child = struct { token: u32, id: u32 };

const Node = struct {
    id: u32,
    parent: ?u32,
    token: u32,
    position: u32,
    depth: u32 = 0,
    hidden: f64 = 0,
    features: f64 = 0,
    role: Role = .main,
    origin: Origin = .initial,
    growing: bool = true,
    ever_frozen: bool = false,
    promoted: bool = false,
    registered: bool = false,
    proposal_depth: u32 = 0,
    readouts: [max_loops + 1]u32 = @splat(0),
    has_readout: [max_loops + 1]bool = @splat(false),
    loop_features: [max_loops + 1]f64 = @splat(0),
    has_loop_feature: [max_loops + 1]bool = @splat(false),
    children: [8]Child = @splat(.{ .token = 0, .id = 0 }),
    child_count: u8 = 0,
    frozen_suffix: [max_block]u32 = @splat(0),
    frozen_len: u8 = 0,
    dists: [max_loops + 1][97]f32 = @splat(@splat(0)),
    has_dist: [max_loops + 1]bool = @splat(false),
    proposal_row: [97]f32 = @splat(0),
    has_proposal: bool = false,

    fn childId(self: Node, token: u32) ?u32 {
        for (self.children[0..self.child_count]) |c| if (c.token == token) return c.id;
        return null;
    }
    fn addChild(self: *Node, token: u32, id: u32) void {
        std.debug.assert(self.child_count < self.children.len);
        self.children[self.child_count] = .{ .token = token, .id = id };
        self.child_count += 1;
    }
    fn removeChild(self: *Node, token: u32) void {
        var i: u8 = 0;
        while (i < self.child_count) : (i += 1) {
            if (self.children[i].token == token) {
                self.children[i] = self.children[self.child_count - 1];
                self.child_count -= 1;
                return;
            }
        }
    }
};

const RunnerNode = struct {
    parent: ?u32,
    token: u32,
    depth: u32 = 0,
    inputs: [max_loops]f64 = @splat(0),
    has_input: [max_loops]bool = @splat(false),
};

pub const TinyRunner = struct {
    gpa: std.mem.Allocator,
    R: u32 = tiny_R,
    vocab: u32 = tiny_vocab,
    max_len: u32 = 512,
    max_nodes: u32 = 8,
    causal: bool = true,
    kind: TinyKind = .growing,
    prefix_tokens: std.ArrayList(u32) = .empty,
    prefix_inputs: [max_loops]std.ArrayList(f64) = @splat(.empty),
    nodes: std.AutoHashMapUnmanaged(u32, RunnerNode) = .empty,
    keys: std.AutoHashMapUnmanaged(u64, u32) = .empty,
    markov: ?*[5][5][5]f32 = null, // [depth 1..=4][token][next] for V=5
    calls: u32 = 0,
    cells: u32 = 0,
    registered_nodes: u32 = 0,
    committed_nodes: u32 = 0,
    released_nodes: u32 = 0,
    release_roots: std.ArrayList(u32) = .empty,

    pub fn init(gpa: std.mem.Allocator, max_nodes: u32, causal: bool, kind: TinyKind) TinyRunner {
        return .{ .gpa = gpa, .max_nodes = max_nodes, .causal = causal, .kind = kind };
    }

    pub fn deinit(self: *TinyRunner) void {
        self.reset();
        self.prefix_tokens.deinit(self.gpa);
        for (&self.prefix_inputs) |*row| row.deinit(self.gpa);
        self.nodes.deinit(self.gpa);
        self.keys.deinit(self.gpa);
        self.release_roots.deinit(self.gpa);
    }

    pub fn prefixLength(self: *const TinyRunner) u32 {
        return @intCast(self.prefix_tokens.items.len);
    }

    pub fn reset(self: *TinyRunner) void {
        self.prefix_tokens.clearRetainingCapacity();
        for (&self.prefix_inputs) |*row| row.clearRetainingCapacity();
        self.nodes.clearRetainingCapacity();
        self.keys.clearRetainingCapacity();
        self.release_roots.clearRetainingCapacity();
        self.calls = 0;
        self.cells = 0;
        self.registered_nodes = 0;
        self.committed_nodes = 0;
        self.released_nodes = 0;
    }

    fn keyOf(parent: ?u32, token: u32) u64 {
        const p: u64 = if (parent) |id| id else std.math.maxInt(u32);
        return (p << 32) | token;
    }

    pub fn ensureCapacity(self: *TinyRunner, required: u32) void {
        while (required > self.max_nodes) self.max_nodes *= 2;
    }

    pub fn embed(self: *const TinyRunner, token: u32) f64 {
        _ = self;
        return @floatFromInt(token);
    }

    pub fn prefill(self: *TinyRunner, tokens: []const u32) !f64 {
        self.reset();
        try self.prefix_tokens.appendSlice(self.gpa, tokens);
        var values: [512]u32 = undefined;
        std.debug.assert(tokens.len <= values.len);
        @memcpy(values[0..tokens.len], tokens);
        const n = tokens.len;
        for (0..self.R) |loop| {
            try self.prefix_inputs[loop].ensureTotalCapacity(self.gpa, n);
            self.prefix_inputs[loop].clearRetainingCapacity();
            for (values[0..n]) |v| self.prefix_inputs[loop].appendAssumeCapacity(@floatFromInt(v));
            if (self.causal) {
                var tmp: [512]u32 = undefined;
                var prefix_sum: u32 = 0;
                for (values[0..n], tmp[0..n]) |value, *out| {
                    out.* = (3 * value + 2 * prefix_sum + @as(u32, @intCast(loop)) + 1) % self.vocab;
                    prefix_sum += value;
                }
                @memcpy(values[0..n], tmp[0..n]);
            }
        }
        if (self.kind == .growing) return @floatFromInt(values[n - 1]);
        // Controlled / Markov: readout from the last prompt token at loop R.
        return @as(f64, @floatFromInt(tokens[n - 1])) + 4000;
    }

    pub fn logitsArgmax(self: *const TinyRunner, hidden: f64) u32 {
        return self.logitsBest(hidden);
    }

    fn logitsBest(self: *const TinyRunner, hidden: f64) u32 {
        if (self.kind == .growing) {
            const v: i64 = @intFromFloat(@round(hidden));
            return @intCast(@mod(v + 1, @as(i64, @intCast(self.vocab))));
        }
        const raw: i64 = @intFromFloat(@round(hidden));
        const token: u32 = @intCast(@mod(raw, 1000));
        const depth: u32 = @intCast(@divTrunc(raw, 1000));
        if (self.kind == .markov) {
            const chain = self.markov.?;
            var best: u32 = 0;
            var best_p: f32 = -1;
            const d: usize = @intCast(depth);
            const t: usize = @intCast(token);
            for (0..5) |next| {
                const p = chain[d][t][next];
                if (p > best_p) {
                    best_p = p;
                    best = @intCast(next);
                }
            }
            return best;
        }
        var pred: u32 = (token + 1) % self.vocab;
        if (self.kind == .wide) {
            if (depth == 2) pred = (token + 7) % self.vocab;
            return pred;
        }
        if (token == 12 and depth == 1) pred = 50;
        if (self.kind == .fork and token == 12 and depth >= 2) pred = 60;
        if (self.kind == .nested and token == 13 and depth == 2) pred = 70;
        return pred;
    }

    fn softmaxRow(self: *const TinyRunner, hidden: f64, out: []f32, temperature: f32, top_p: f32) void {
        const V = out.len;
        if (self.kind == .markov) {
            const raw: i64 = @intFromFloat(@round(hidden));
            const token: u32 = @intCast(@mod(raw, 1000));
            const depth: u32 = @intCast(@divTrunc(raw, 1000));
            const chain = self.markov.?;
            const d: usize = @intCast(depth);
            const t: usize = @intCast(token);
            var max_logit: f32 = -std.math.inf(f32);
            for (0..V) |i| {
                const p = chain[d][t][i];
                const logit = if (p > 0) @log(p) / temperature else -1e9;
                out[i] = logit;
                if (logit > max_logit) max_logit = logit;
            }
            var sum: f32 = 0;
            for (out) |*x| {
                x.* = @exp(x.* - max_logit);
                sum += x.*;
            }
            for (out) |*x| x.* /= sum;
            applyTopP(out, top_p);
            return;
        }
        @memset(out, 0);
        const pred = self.logitsBest(hidden);
        if (pred < V) {
            out[pred] = 1;
        }
    }

    pub fn register(self: *TinyRunner, node_id: u32, parent_id: ?u32, token: u32) !void {
        const k = keyOf(parent_id, token);
        if (self.nodes.getPtr(node_id)) |node| {
            if (node.parent != parent_id or node.token != token)
                return error.ReplacedExistingNode;
            return;
        }
        if (self.keys.contains(k)) return error.UnmergedCandidate;
        if (self.nodes.count() >= self.max_nodes) return error.NodeBudget;
        if (parent_id) |p| if (!self.nodes.contains(p)) return error.LostAncestor;
        try self.nodes.put(self.gpa, node_id, .{ .parent = parent_id, .token = token });
        try self.keys.put(self.gpa, k, node_id);
        self.registered_nodes += 1;
    }

    fn path(self: *const TinyRunner, node_id: u32, buf: []u32) []const u32 {
        var tmp: [64]u32 = undefined;
        var n: usize = 0;
        var cur: ?u32 = node_id;
        while (cur) |id| {
            tmp[n] = id;
            n += 1;
            cur = self.nodes.get(id).?.parent;
        }
        var i: usize = 0;
        while (i < n) : (i += 1) buf[i] = tmp[n - 1 - i];
        return buf[0..n];
    }

    pub fn advance(self: *TinyRunner, ids: []const u32, loops: []const u32, hiddens: []const f64, out_hidden: []f64) !void {
        if (self.calls > 2000) return error.StarvedVerification;
        var seen: std.AutoHashMapUnmanaged(u32, void) = .empty;
        defer seen.deinit(self.gpa);
        for (ids) |id| {
            if (seen.contains(id)) return error.DoubleAdvance;
            try seen.put(self.gpa, id, {});
        }
        var unfinished: u32 = 0;
        var it = self.nodes.iterator();
        while (it.next()) |kv| {
            if (kv.value_ptr.depth < self.R) unfinished += 1;
        }
        if (unfinished != ids.len) return error.MissedUnfinishedNode;
        for (ids, loops, hiddens, out_hidden) |id, loop, hidden, *out| {
            const node = self.nodes.getPtr(id).?;
            if (node.depth != loop) return error.WrongLoop;
            node.inputs[loop] = hidden;
            node.has_input[loop] = true;
            var ancestors: [64]u32 = undefined;
            const pth = self.path(id, &ancestors);
            var prefix_sum: f64 = 0;
            for (self.prefix_inputs[loop].items) |v| prefix_sum += v;
            for (pth[0 .. pth.len - 1]) |anc| {
                const a = self.nodes.get(anc).?;
                if (!a.has_input[loop]) return error.MissingAncestorKv;
                prefix_sum += a.inputs[loop];
            }
            var y = hidden;
            if (self.causal) {
                const loop_i: f64 = @floatFromInt(loop);
                const iv: i64 = @intFromFloat(@round(3 * hidden + 2 * prefix_sum + loop_i + 1));
                y = @floatFromInt(@mod(iv, @as(i64, @intCast(self.vocab))));
            }
            if (self.kind != .growing) {
                y = @as(f64, @floatFromInt(node.token)) + @as(f64, @floatFromInt(loop + 1)) * 1000;
            }
            out.* = y;
            node.depth += 1;
        }
        self.calls += 1;
        self.cells += @intCast(ids.len);
    }

    pub fn release(self: *TinyRunner, ids: []const u32) !void {
        var removed: std.AutoHashMapUnmanaged(u32, void) = .empty;
        defer removed.deinit(self.gpa);
        for (ids) |id| try removed.put(self.gpa, id, {});
        var it = self.nodes.iterator();
        while (it.next()) |kv| {
            if (removed.contains(kv.key_ptr.*)) continue;
            var ancestors: [64]u32 = undefined;
            const pth = self.path(kv.key_ptr.*, &ancestors);
            for (pth[0 .. pth.len - 1]) |anc| {
                if (removed.contains(anc)) return error.StaleAncestry;
            }
        }
        for (ids) |id| {
            const node = self.nodes.get(id).?;
            if (node.parent) |p| {
                if (!removed.contains(p)) try self.release_roots.append(self.gpa, p);
            }
        }
        for (ids) |id| {
            const node = self.nodes.get(id).?;
            _ = self.keys.remove(keyOf(node.parent, node.token));
            _ = self.nodes.remove(id);
        }
        self.released_nodes += @intCast(ids.len);
    }

    pub fn commit(self: *TinyRunner, ids: []const u32) !void {
        if (ids.len == 0) return;
        for (ids, 0..) |id, i| {
            const node = self.nodes.get(id).?;
            const expected: ?u32 = if (i == 0) null else ids[i - 1];
            if (node.parent != expected or node.depth != self.R) return error.InvalidCommitPath;
        }
        for (ids) |id| {
            const node = self.nodes.get(id).?;
            try self.prefix_tokens.append(self.gpa, node.token);
            for (0..self.R) |loop| {
                try self.prefix_inputs[loop].append(self.gpa, node.inputs[loop]);
            }
        }
        var kept: std.AutoHashMapUnmanaged(u32, void) = .empty;
        defer kept.deinit(self.gpa);
        for (ids) |id| try kept.put(self.gpa, id, {});
        var drop: std.ArrayList(u32) = .empty;
        defer drop.deinit(self.gpa);
        var it = self.nodes.iterator();
        while (it.next()) |kv| try drop.append(self.gpa, kv.key_ptr.*);
        for (drop.items) |id| {
            if (kept.contains(id)) {
                _ = self.nodes.remove(id);
            }
        }
        self.keys.clearRetainingCapacity();
        var it2 = self.nodes.iterator();
        while (it2.next()) |kv| {
            if (kv.value_ptr.parent) |p| {
                if (kept.contains(p)) kv.value_ptr.parent = null;
            }
            try self.keys.put(self.gpa, keyOf(kv.value_ptr.parent, kv.value_ptr.token), kv.key_ptr.*);
        }
        self.committed_nodes += @intCast(ids.len);
    }
};

fn applyTopP(row: []f32, top_p: f32) void {
    if (top_p >= 1) return;
    const V = row.len;
    var order: [32]u32 = undefined;
    std.debug.assert(V <= order.len);
    for (0..V) |i| order[i] = @intCast(i);
    std.mem.sort(u32, order[0..V], row, struct {
        fn lt(probs: []f32, a: u32, b: u32) bool {
            if (probs[a] == probs[b]) return a < b;
            return probs[a] > probs[b];
        }
    }.lt);
    var mass: f32 = 0;
    var keep: usize = 0;
    while (keep < V) : (keep += 1) {
        mass += row[order[keep]];
        if (mass >= top_p) {
            keep += 1;
            break;
        }
    }
    var rest: [32]f32 = @splat(0);
    var sum: f32 = 0;
    for (0..keep) |i| {
        rest[order[i]] = row[order[i]];
        sum += rest[order[i]];
    }
    if (sum > 0) {
        for (0..V) |i| row[i] = rest[i] / sum;
    }
}

pub const CommitSources = struct {
    main: u32 = 0,
    frozen: u32 = 0,
    fork: u32 = 0,
    bonus: u32 = 0,
    pub fn total(self: CommitSources) u32 {
        return self.main + self.frozen + self.fork + self.bonus;
    }
};

pub const Stats = struct {
    calls: u32 = 0,
    cells: u32 = 0,
    max_live_nodes: u32 = 0,
    committed_decode_tokens: u32 = 0,
    accepted_candidates: u32 = 0,
    bonus_tokens: u32 = 0,
    stage1_blocks: u32 = 0,
    stage1_progress_tokens: u32 = 0,
    stage2_reviews: u32 = 0,
    stage2_disagreements: u32 = 0,
    frozen_created: u32 = 0,
    frozen_tokens: u32 = 0,
    promotions: u32 = 0,
    promotion_reused_tokens: u32 = 0,
    promoted_heads_committed: u32 = 0,
    previously_frozen_tokens_committed: u32 = 0,
    forks: u32 = 0,
    pruned_nodes: u32 = 0,
    drafter_batches: u32 = 0,
    draft_slots: u32 = 0,
    prefix_rows: u32 = 0,
    tree_rows_bound: u32 = 0,
    drafter_prefix_rows: u32 = 0,
    attention_history_pairs: u64 = 0,
    attention_tree_pairs: u64 = 0,
    commit_sources: CommitSources = .{},
};

pub const TraceKind = enum { anchor, freeze, stage1, promotion, commit };
pub const TraceEvent = struct {
    call: u32,
    kind: TraceKind,
    token: u32 = 0,
    tokens: []u32 = &.{},
    position: u32 = 0,
    depth: u32 = 0,
    cause: []const u8 = "",
    source: []const u8 = "",
    progress: u32 = 0,
};

pub const Result = struct {
    output: []u32,
    stats: Stats,
    trace: []TraceEvent = &.{},
    prefill_ns: u64 = 0,
    decode_ns: u64 = 0,
};

const DraftEvent = struct {
    head: u32,
    parent_position: u32,
    parent_features: f64,
    depth: u32,
    cause: Origin,
};

const Block = struct {
    ids: [max_block]u32 = @splat(0),
    len: u8 = 0,
};

pub const Executor = struct {
    gpa: std.mem.Allocator,
    runner: *TinyRunner,
    opts: Options,
    nodes: std.AutoHashMapUnmanaged(u32, Node) = .empty,
    blocks: std.AutoHashMapUnmanaged(u32, Block) = .empty,
    events: std.ArrayList(DraftEvent) = .empty,
    prefix_tokens: std.ArrayList(u32) = .empty,
    output: std.ArrayList(u32) = .empty,
    trace: std.ArrayList(TraceEvent) = .empty,
    freeze_tokens: std.ArrayList(u32) = .empty,
    next_id: u32 = 0,
    cursor: u32 = 0,
    stats: Stats = .{},
    max_new: u32 = 0,
    output_limit: u32 = 0,
    eos: []const u32 = &.{},
    rng: std.Random.DefaultPrng = undefined,

    pub fn init(gpa: std.mem.Allocator, runner: *TinyRunner, opts: Options) !Executor {
        try validate(opts, runner.R);
        return .{ .gpa = gpa, .runner = runner, .opts = opts, .rng = .init(opts.seed) };
    }

    pub fn deinit(self: *Executor) void {
        self.nodes.deinit(self.gpa);
        self.blocks.deinit(self.gpa);
        self.events.deinit(self.gpa);
        self.prefix_tokens.deinit(self.gpa);
        self.output.deinit(self.gpa);
        self.trace.deinit(self.gpa);
        self.freeze_tokens.deinit(self.gpa);
    }

    fn resetCandidates(self: *Executor) void {
        self.nodes.clearRetainingCapacity();
        self.blocks.clearRetainingCapacity();
        self.events.clearRetainingCapacity();
        self.next_id = 0;
        self.stats = .{};
        self.trace.clearRetainingCapacity();
        self.freeze_tokens.clearRetainingCapacity();
    }

    fn nodePtr(self: *Executor, id: u32) *Node {
        return self.nodes.getPtr(id).?;
    }

    fn readoutDepths(self: *const Executor, d: u32) bool {
        const R = self.runner.R;
        if (d == R or d == self.opts.d1) return true;
        if (self.opts.stage != .stage1 and d == self.opts.d2) return true;
        return false;
    }

    fn child(self: *Executor, parent: ?u32, token: u32, role: Role, origin: Origin, proposal_depth: u32) !u32 {
        if (parent) |p| {
            if (self.nodePtr(p).childId(token)) |existing| return existing;
        }
        const position: u32 = if (parent) |p| self.nodePtr(p).position + 1 else self.runner.prefixLength();
        const id = self.next_id;
        self.next_id += 1;
        const n = Node{
            .id = id,
            .parent = parent,
            .token = token,
            .position = position,
            .role = role,
            .origin = origin,
            .proposal_depth = proposal_depth,
        };
        if (parent) |p| self.nodePtr(p).addChild(token, id);
        try self.nodes.put(self.gpa, id, n);
        return id;
    }

    fn descendants(self: *Executor, id: u32, out: *std.ArrayList(u32)) !void {
        try out.append(self.gpa, id);
        var i: usize = out.items.len - 1;
        while (i < out.items.len) : (i += 1) {
            const n = self.nodePtr(out.items[i]);
            for (n.children[0..n.child_count]) |c| try out.append(self.gpa, c.id);
        }
    }

    fn drop(self: *Executor, removed: []const u32) !void {
        var registered: std.ArrayList(u32) = .empty;
        defer registered.deinit(self.gpa);
        var set: std.AutoHashMapUnmanaged(u32, void) = .empty;
        defer set.deinit(self.gpa);
        for (removed) |id| {
            try set.put(self.gpa, id, {});
            if (self.nodes.get(id)) |n| {
                if (n.registered) try registered.append(self.gpa, id);
            }
        }
        if (registered.items.len > 0) try self.runner.release(registered.items);
        for (removed) |id| {
            if (self.nodes.get(id)) |n| {
                if (n.parent) |p| {
                    if (self.nodes.getPtr(p)) |parent| parent.removeChild(n.token);
                }
            }
        }
        for (removed) |id| _ = self.nodes.remove(id);
        self.stats.pruned_nodes += @intCast(removed.len);
    }

    fn queue(self: *Executor, parent: u32, head: u32, depth: u32, cause: Origin) !void {
        const h = self.nodePtr(head);
        if (h.position >= self.output_limit) return;
        try self.events.append(self.gpa, .{
            .head = head,
            .parent_position = self.nodePtr(parent).position,
            .parent_features = self.nodePtr(parent).features,
            .depth = depth,
            .cause = cause,
        });
        try self.addTrace(.{
            .call = self.stats.calls,
            .kind = .anchor,
            .token = h.token,
            .position = h.position,
            .depth = depth,
            .cause = originName(cause),
        });
    }

    fn newAnchor(self: *Executor, parent: u32, depth: u32, cause: Origin, role: Role) !u32 {
        const p = self.nodePtr(parent);
        if (!p.has_readout[depth]) return error.MissingReadout;
        const token = p.readouts[depth];
        if (p.childId(token) != null) return error.DuplicateAnchor;
        const head = try self.child(parent, token, role, cause, depth);
        try self.queue(parent, head, depth, cause);
        return head;
    }

    fn freeze(self: *Executor, suffix: []const u32) !void {
        const head = self.nodePtr(suffix[0]);
        head.role = .frozen;
        head.frozen_len = @intCast(suffix.len);
        for (suffix, 0..) |id, i| {
            head.frozen_suffix[i] = id;
            const n = self.nodePtr(id);
            n.growing = false;
            n.ever_frozen = true;
        }
        self.stats.frozen_created += 1;
        self.stats.frozen_tokens += @intCast(suffix.len);
        const start = self.freeze_tokens.items.len;
        for (suffix) |id| try self.freeze_tokens.append(self.gpa, self.nodePtr(id).token);
        try self.addTrace(.{
            .call = self.stats.calls,
            .kind = .freeze,
            .tokens = self.freeze_tokens.items[start..],
            .position = head.position,
        });
    }

    fn scan(self: *Executor, head_id: u32, suffix: []const u32, depth: u32, cause: Origin) !u32 {
        var parent_id = head_id;
        var passed: u32 = 0;
        self.nodePtr(head_id).growing = true;
        for (suffix, 0..) |child_id, index| {
            const ch = self.nodePtr(child_id);
            const parent = self.nodePtr(parent_id);
            if (ch.parent != parent_id or ch.depth < depth) return error.BrokenSuffix;
            if (!parent.has_readout[depth] or parent.readouts[depth] != ch.token) {
                try self.freeze(suffix[index..]);
                const reject: Origin = switch (cause) {
                    .promotion, .promotion_accepted, .promotion_frontier => .promotion_reject,
                    else => .d1_reject,
                };
                _ = try self.newAnchor(parent_id, depth, reject, .main);
                return passed;
            }
            ch.growing = true;
            ch.proposal_depth = depth;
            ch.origin = switch (cause) {
                .promotion, .promotion_accepted, .promotion_frontier => .promotion_accepted,
                else => .d1_accepted,
            };
            parent_id = child_id;
            passed += 1;
        }
        if (self.nodePtr(parent_id).position + 1 < self.output_limit) {
            const frontier: Origin = switch (cause) {
                .promotion, .promotion_accepted, .promotion_frontier => .promotion_frontier,
                else => .d1_frontier,
            };
            _ = try self.newAnchor(parent_id, depth, frontier, .main);
        }
        return passed;
    }

    fn addTrace(self: *Executor, ev: TraceEvent) !void {
        if (self.opts.record_trace) try self.trace.append(self.gpa, ev);
    }

    fn findRole(self: *Executor, parent: u32, role: Role) ?u32 {
        const p = self.nodePtr(parent);
        for (p.children[0..p.child_count]) |c| {
            if (self.nodePtr(c.id).role == role) return c.id;
        }
        return null;
    }

    fn stage1(self: *Executor) !void {
        if (self.opts.d1 == self.runner.R) return;
        var heads: std.ArrayList(u32) = .empty;
        defer heads.deinit(self.gpa);
        var it = self.blocks.iterator();
        while (it.next()) |kv| try heads.append(self.gpa, kv.key_ptr.*);
        for (heads.items) |head_id| {
            const block = self.blocks.get(head_id) orelse continue;
            const head = self.nodes.get(head_id) orelse {
                _ = self.blocks.remove(head_id);
                continue;
            };
            if (head.depth < self.opts.d1) continue;
            _ = self.blocks.remove(head_id);
            if (!head.growing) continue;
            if (head.depth != self.opts.d1) return error.MissedD1Review;
            const position = head.position;
            const passed = try self.scan(head_id, block.ids[1..block.len], self.opts.d1, .d1_accepted);
            self.stats.stage1_blocks += 1;
            self.stats.stage1_progress_tokens += 1 + passed;
            try self.addTrace(.{ .call = self.stats.calls, .kind = .stage1, .position = position, .progress = 1 + passed });
        }
    }

    fn stage2(self: *Executor, advanced: []const u32) !void {
        if (self.opts.stage == .stage1) return;
        var handled: std.AutoHashMapUnmanaged(u32, void) = .empty;
        defer handled.deinit(self.gpa);
        for (advanced) |parent_id| {
            if (!self.nodes.contains(parent_id)) continue;
            const parent_snap = self.nodes.get(parent_id).?;
            if (parent_snap.depth != self.opts.d2 or !parent_snap.growing) continue;
            if (handled.contains(parent_id)) continue;
            const main = self.findRole(parent_id, .main) orelse continue;
            const frozen = self.findRole(parent_id, .frozen);
            self.stats.stage2_reviews += 1;
            if (!parent_snap.has_readout[self.opts.d2]) return error.MissingReadout;
            const predicted = parent_snap.readouts[self.opts.d2];
            if (predicted == self.nodePtr(main).token) continue;
            self.stats.stage2_disagreements += 1;
            if (frozen) |fid| {
                if (predicted == self.nodePtr(fid).token) {
                    const fr = self.nodes.get(fid).?;
                    if (fr.frozen_len == 0 or fr.frozen_suffix[0] != fid) return error.PromotionHasNoSuffix;
                    var suffix: [max_block]u32 = undefined;
                    const slen = fr.frozen_len;
                    @memcpy(suffix[0..slen], fr.frozen_suffix[0..slen]);
                    for (suffix[0..slen]) |id| {
                        if (self.nodePtr(id).depth != self.opts.d2) return error.PromotionNotAtD2;
                        try handled.put(self.gpa, id, {});
                    }
                    const pos = fr.position;
                    self.nodePtr(fid).promoted = true;
                    self.nodePtr(fid).frozen_len = 0;
                    self.stats.promotions += 1;
                    const passed = try self.scan(fid, suffix[1..slen], self.opts.d2, .promotion);
                    self.stats.promotion_reused_tokens += 1 + passed;
                    try self.addTrace(.{
                        .call = self.stats.calls,
                        .kind = .promotion,
                        .position = pos,
                        .progress = 1 + passed,
                        .cause = "promotion",
                    });
                    continue;
                }
            }
            if (self.opts.stage == .stage2b and parent_snap.position + 1 < self.output_limit) {
                if (self.nodePtr(parent_id).childId(predicted) != null) return error.DuplicateD2Event;
                _ = try self.newAnchor(parent_id, self.opts.d2, .d2_fork, .fork);
                self.stats.forks += 1;
            }
        }
    }

    fn verify(self: *Executor) !bool {
        var accepted: std.ArrayList(u32) = .empty;
        defer accepted.deinit(self.gpa);
        var removed: std.ArrayList(u32) = .empty;
        defer removed.deinit(self.gpa);
        var stopped = false;
        const R = self.runner.R;
        while (self.nodePtr(self.cursor).depth == R) {
            const current = self.cursor;
            const cur = self.nodePtr(current);
            if (!cur.has_readout[R]) return error.MissingReadout;
            const token = cur.readouts[R];
            var source: []const u8 = "bonus";
            const chosen = if (cur.childId(token)) |cid| blk: {
                const ch = self.nodePtr(cid);
                source = roleName(ch.role);
                if (ch.role == .main and ch.ever_frozen and ch.proposal_depth == 0) source = "frozen";
                self.stats.accepted_candidates += 1;
                if (ch.promoted) self.stats.promoted_heads_committed += 1;
                if (ch.ever_frozen) self.stats.previously_frozen_tokens_committed += 1;
                break :blk cid;
            } else blk: {
                const id = try self.child(current, token, .main, .bonus, R);
                self.stats.bonus_tokens += 1;
                source = "bonus";
                break :blk id;
            };
            for (self.nodePtr(current).children[0..self.nodePtr(current).child_count]) |c| {
                if (c.id != chosen) try self.descendants(c.id, &removed);
            }
            try accepted.append(self.gpa, current);
            self.cursor = chosen;
            try self.output.append(self.gpa, self.nodePtr(chosen).token);
            self.stats.committed_decode_tokens += 1;
            if (std.mem.eql(u8, source, "main")) self.stats.commit_sources.main += 1;
            if (std.mem.eql(u8, source, "frozen")) self.stats.commit_sources.frozen += 1;
            if (std.mem.eql(u8, source, "fork")) self.stats.commit_sources.fork += 1;
            if (std.mem.eql(u8, source, "bonus")) self.stats.commit_sources.bonus += 1;
            try self.addTrace(.{
                .call = self.stats.calls,
                .kind = .commit,
                .token = self.nodePtr(chosen).token,
                .position = self.nodePtr(chosen).position,
                .source = source,
                .depth = self.nodePtr(chosen).depth,
                .cause = originName(self.nodePtr(chosen).origin),
            });
            if (std.mem.indexOfScalar(u32, self.eos, self.nodePtr(chosen).token) != null or
                self.output.items.len >= self.max_new)
            {
                stopped = true;
                break;
            }
            if (std.mem.eql(u8, source, "bonus")) {
                try self.queue(current, chosen, R, .bonus);
            }
        }
        if (accepted.items.len > 0) {
            try self.drop(removed.items);
            try self.runner.commit(accepted.items);
            for (accepted.items) |id| try self.prefix_tokens.append(self.gpa, self.nodePtr(id).token);
            self.nodePtr(self.cursor).parent = null;
            for (accepted.items) |id| _ = self.nodes.remove(id);
            var stale: std.ArrayList(u32) = .empty;
            defer stale.deinit(self.gpa);
            var it = self.blocks.iterator();
            while (it.next()) |kv| {
                if (!self.nodes.contains(kv.key_ptr.*)) try stale.append(self.gpa, kv.key_ptr.*);
            }
            for (stale.items) |id| _ = self.blocks.remove(id);
        }
        return stopped;
    }

    fn pathTokens(self: *Executor, head: u32, buf: []u32) []u32 {
        var tmp: [128]u32 = undefined;
        var n: usize = 0;
        var cur: ?u32 = head;
        while (cur) |id| {
            tmp[n] = self.nodePtr(id).token;
            n += 1;
            cur = self.nodePtr(id).parent;
        }
        var i: usize = 0;
        while (i < n) : (i += 1) buf[i] = tmp[n - 1 - i];
        return buf[0..n];
    }

    fn draftEvents(self: *Executor) !void {
        var pending = self.events;
        self.events = .empty;
        defer pending.deinit(self.gpa);
        var seen: std.AutoHashMapUnmanaged(u32, void) = .empty;
        defer seen.deinit(self.gpa);
        var plans: std.ArrayList(u32) = .empty;
        defer plans.deinit(self.gpa);
        var widths: std.ArrayList(u32) = .empty;
        defer widths.deinit(self.gpa);
        for (pending.items) |event| {
            const head = self.nodes.getPtr(event.head) orelse continue;
            if (!head.growing) continue;
            if (seen.contains(head.id) or head.child_count != 0) return error.DoubleDraft;
            try seen.put(self.gpa, head.id, {});
            const width = @min(self.opts.k, self.output_limit - head.position);
            var block: Block = .{};
            block.ids[0] = head.id;
            block.len = 1;
            try self.blocks.put(self.gpa, head.id, block);
            if (width <= 1) continue;
            try plans.append(self.gpa, head.id);
            try widths.append(self.gpa, width);
        }
        if (plans.items.len > 0) {
            for (plans.items, widths.items) |head_id, width| {
                var path_buf: [128]u32 = undefined;
                const path = self.pathTokens(head_id, &path_buf);
                var seq: [1024]u32 = undefined;
                @memcpy(seq[0..self.prefix_tokens.items.len], self.prefix_tokens.items);
                @memcpy(seq[self.prefix_tokens.items.len..][0..path.len], path);
                const seq_len = self.prefix_tokens.items.len + path.len;
                var slots: [max_block]u32 = undefined;
                const n = width - 1;
                switch (self.opts.drafter) {
                    .none => unreachable,
                    .oracle, .oracle_corrupt => {
                        const corrupt = self.opts.drafter == .oracle_corrupt;
                        var working: [1024]u32 = undefined;
                        @memcpy(working[0..seq_len], seq[0..seq_len]);
                        var wlen = seq_len;
                        var sum: u32 = 0;
                        for (working[0..wlen]) |t| sum += t;
                        for (0..n) |i| {
                            const token = oracleNext(working[0..wlen], self.runner.causal);
                            const used = if (corrupt and (sum + @as(u32, @intCast(i))) % 3 == 0)
                                (token + 7) % tiny_vocab
                            else
                                token;
                            slots[i] = used;
                            working[wlen] = used;
                            sum += used;
                            wlen += 1;
                        }
                    },
                    .history => {
                        var hist_out: [max_block]u32 = undefined;
                        const got = history_draft.propose(seq[0..seq_len], hist_out[0..n]);
                        var i: usize = 0;
                        while (i < n) : (i += 1) {
                            slots[i] = if (i < got) hist_out[i] else oracleNext(seq[0..seq_len], self.runner.causal);
                        }
                    },
                }
                var parent = head_id;
                var block = self.blocks.getPtr(head_id).?;
                for (slots[0..n]) |token| {
                    parent = try self.child(parent, token, .main, .raw_draft, 0);
                    block.ids[block.len] = parent;
                    block.len += 1;
                }
                self.stats.draft_slots += n;
            }
            self.stats.drafter_batches += 1;
            self.stats.drafter_prefix_rows += self.runner.prefixLength();
        }
        try self.registerPending();
    }

    fn registerPending(self: *Executor) !void {
        self.runner.ensureCapacity(@intCast(self.nodes.count()));
        var pending: std.ArrayList(u32) = .empty;
        defer pending.deinit(self.gpa);
        var it = self.nodes.iterator();
        while (it.next()) |kv| {
            if (!kv.value_ptr.registered) try pending.append(self.gpa, kv.key_ptr.*);
        }
        std.mem.sort(u32, pending.items, self, struct {
            fn lt(ex: *Executor, a: u32, b: u32) bool {
                const na = ex.nodePtr(a);
                const nb = ex.nodePtr(b);
                if (na.position != nb.position) return na.position < nb.position;
                return a < b;
            }
        }.lt);
        for (pending.items) |id| {
            const n = self.nodePtr(id);
            try self.runner.register(id, n.parent, n.token);
            n.registered = true;
        }
        self.stats.max_live_nodes = @max(self.stats.max_live_nodes, @as(u32, @intCast(self.nodes.count())));
    }

    fn collectActive(self: *Executor, buf: *std.ArrayList(u32)) !void {
        buf.clearRetainingCapacity();
        const R = self.runner.R;
        var it = self.nodes.iterator();
        while (it.next()) |kv| {
            if (kv.value_ptr.depth < R) try buf.append(self.gpa, kv.key_ptr.*);
        }
        std.mem.sort(u32, buf.items, self, struct {
            fn lt(ex: *Executor, a: u32, b: u32) bool {
                const na = ex.nodePtr(a);
                const nb = ex.nodePtr(b);
                if (na.position != nb.position) return na.position < nb.position;
                return a < b;
            }
        }.lt);
    }

    pub fn generate(self: *Executor, a: std.mem.Allocator, prompt: []const u32, max_new: u32, eos: []const u32) !Result {
        if (prompt.len == 0 or max_new < 1 or prompt.len + max_new > self.runner.max_len)
            return error.InvalidGenerate;
        self.resetCandidates();
        self.runner.reset();
        self.prefix_tokens.clearRetainingCapacity();
        self.output.clearRetainingCapacity();
        self.max_new = max_new;
        self.eos = eos;
        self.output_limit = @intCast(prompt.len + max_new);
        self.rng = .init(self.opts.seed);

        const last_hidden = try self.runner.prefill(prompt);
        try self.prefix_tokens.appendSlice(self.gpa, prompt);
        const first = self.runner.logitsArgmax(last_hidden);
        try self.output.append(self.gpa, first);
        self.cursor = try self.child(null, first, .main, .initial, self.runner.R);
        var stopped = std.mem.indexOfScalar(u32, eos, first) != null or max_new == 1;
        if (!stopped) {
            const parent_pos: u32 = @intCast(prompt.len - 1);
            try self.events.append(self.gpa, .{
                .head = self.cursor,
                .parent_position = parent_pos,
                .parent_features = last_hidden,
                .depth = self.runner.R,
                .cause = .initial,
            });
            // The initial parent is the last prompt token; features are the prefill readout.
            try self.draftEvents();
        }

        var active: std.ArrayList(u32) = .empty;
        defer active.deinit(self.gpa);
        const R = self.runner.R;
        while (!stopped) {
            try self.collectActive(&active);
            if (active.items.len == 0) return error.EmptyTree;
            const prefix_length = self.runner.prefixLength();
            const cells: u32 = @intCast(active.items.len);
            self.stats.calls += 1;
            self.stats.cells += cells;
            var loop_set: u32 = 0;
            for (active.items) |id| {
                const d = self.nodePtr(id).depth;
                loop_set |= @as(u32, 1) << @intCast(d);
            }
            var loops_hit: u32 = 0;
            var b: u32 = loop_set;
            while (b != 0) : (b &= b - 1) loops_hit += 1;
            self.stats.prefix_rows += prefix_length;
            self.stats.tree_rows_bound += @as(u32, @intCast(self.nodes.count())) * loops_hit;
            self.stats.attention_history_pairs += @as(u64, cells) * prefix_length;
            var tree_pairs: u64 = 0;
            for (active.items) |id| {
                tree_pairs += self.nodePtr(id).position;
            }
            self.stats.attention_tree_pairs += tree_pairs - @as(u64, cells) * (prefix_length -| 1);

            const n_active = active.items.len;
            const ids = try self.gpa.alloc(u32, n_active);
            defer self.gpa.free(ids);
            const loops = try self.gpa.alloc(u32, n_active);
            defer self.gpa.free(loops);
            const hiddens = try self.gpa.alloc(f64, n_active);
            defer self.gpa.free(hiddens);
            const outs = try self.gpa.alloc(f64, n_active);
            defer self.gpa.free(outs);
            for (active.items, 0..) |id, i| {
                const n = self.nodePtr(id);
                if (n.depth == 0) n.hidden = self.runner.embed(n.token);
                ids[i] = id;
                loops[i] = n.depth;
                hiddens[i] = n.hidden;
            }
            try self.runner.advance(ids, loops, hiddens, outs);
            for (active.items, 0..) |id, i| {
                const n = self.nodePtr(id);
                n.hidden = outs[i];
                n.features = outs[i];
                n.depth += 1;
                if (n.depth != R and (n.depth == self.opts.d1 or (self.opts.stage != .stage1 and n.depth == self.opts.d2))) {
                    n.loop_features[n.depth] = n.features;
                    n.has_loop_feature[n.depth] = true;
                }
                if (self.readoutDepths(n.depth)) {
                    n.readouts[n.depth] = self.runner.logitsArgmax(n.hidden);
                    n.has_readout[n.depth] = true;
                    if (self.opts.sampling()) {
                        const top_p = if (n.depth == R) self.opts.top_p else self.opts.proposal_top_p;
                        const row = n.dists[n.depth][0..self.runner.vocab];
                        self.runner.softmaxRow(n.hidden, row, self.opts.temperature, top_p);
                        n.has_dist[n.depth] = true;
                    }
                }
            }
            const before = self.output.items.len;
            if (self.opts.sampling()) {
                stopped = try self.verifySampled(a);
                if (!stopped) {
                    try self.stage2Sampled(active.items, a);
                    try self.stage1Sampled(a);
                    try self.draftEvents();
                }
            } else {
                stopped = try self.verify();
                if (!stopped) {
                    try self.stage2(active.items);
                    try self.stage1();
                    try self.draftEvents();
                }
            }
            _ = before;
        }
        const out = try a.dupe(u32, self.output.items);
        const tr = try a.dupe(TraceEvent, self.trace.items);
        for (tr) |*ev| {
            if (ev.tokens.len > 0) ev.tokens = try a.dupe(u32, ev.tokens);
        }
        const stats = self.stats;
        self.resetCandidates();
        return .{
            .output = out,
            .stats = stats,
            .trace = tr,
        };
    }

    fn rand01(self: *Executor) f64 {
        return self.rng.random().float(f64);
    }

    fn exp1(self: *Executor) f64 {
        var u = self.rand01();
        while (u == 0) u = self.rand01();
        return -@log(u);
    }

    fn draw(self: *Executor, weights: []const f32) u32 {
        var best: u32 = 0;
        var best_s: f64 = -std.math.inf(f64);
        for (weights, 0..) |w, i| {
            if (w <= 0) continue;
            const s = @as(f64, w) / self.exp1();
            if (s > best_s) {
                best_s = s;
                best = @intCast(i);
            }
        }
        return best;
    }

    fn sampleAnchors(self: *Executor, parents: []const u32, causes: []const Origin, excluded: []const ?u32, depth: u32, role: Role) !void {
        for (parents, causes, excluded) |parent, cause, excl| {
            const p = self.nodePtr(parent);
            if (!p.has_dist[depth]) return error.MissingReadout;
            const dist = p.dists[depth][0..self.runner.vocab];
            var tmp: [97]f32 = undefined;
            std.debug.assert(dist.len <= tmp.len);
            @memcpy(tmp[0..dist.len], dist);
            if (excl) |t| tmp[t] = 0;
            const token = self.draw(tmp[0..dist.len]);
            if (p.childId(token) != null) return error.DuplicateAnchor;
            const head = try self.child(parent, token, role, cause, depth);
            try self.queue(parent, head, depth, cause);
        }
    }

    fn reviewSampled(self: *Executor, a: std.mem.Allocator, head_id: u32, suffix: []const u32, depth: u32, cause: Origin) !u32 {
        _ = a;
        var parent_id = head_id;
        var passed: u32 = 0;
        self.nodePtr(head_id).growing = true;
        for (suffix, 0..) |child_id, index| {
            const ch = self.nodePtr(child_id);
            const parent = self.nodePtr(parent_id);
            if (ch.parent != parent_id or ch.depth < depth) return error.BrokenSuffix;
            if (!parent.has_dist[depth]) return error.MissingReadout;
            const dist = parent.dists[depth][0..self.runner.vocab];
            const p = dist[ch.token];
            if (self.rand01() >= p) {
                try self.freeze(suffix[index..]);
                const reject: Origin = switch (cause) {
                    .promotion, .promotion_accepted => .promotion_reject,
                    else => .d1_reject,
                };
                try self.sampleAnchors(&.{parent_id}, &.{reject}, &.{ch.token}, depth, .main);
                return passed;
            }
            ch.growing = true;
            ch.proposal_depth = depth;
            ch.origin = switch (cause) {
                .promotion, .promotion_accepted => .promotion_accepted,
                else => .d1_accepted,
            };
            parent_id = child_id;
            passed += 1;
        }
        if (self.nodePtr(parent_id).position + 1 < self.output_limit) {
            const frontier: Origin = switch (cause) {
                .promotion, .promotion_accepted => .promotion_frontier,
                else => .d1_frontier,
            };
            try self.sampleAnchors(&.{parent_id}, &.{frontier}, &.{null}, depth, .main);
        }
        return passed;
    }

    fn stage1Sampled(self: *Executor, a: std.mem.Allocator) !void {
        if (self.opts.d1 == self.runner.R) return;
        var heads: std.ArrayList(u32) = .empty;
        defer heads.deinit(self.gpa);
        var it = self.blocks.iterator();
        while (it.next()) |kv| try heads.append(self.gpa, kv.key_ptr.*);
        for (heads.items) |head_id| {
            const block = self.blocks.get(head_id) orelse continue;
            const head = self.nodes.getPtr(head_id) orelse {
                _ = self.blocks.remove(head_id);
                continue;
            };
            if (head.depth < self.opts.d1) continue;
            _ = self.blocks.remove(head_id);
            if (!head.growing) continue;
            if (head.depth != self.opts.d1) return error.MissedD1Review;
            const passed = try self.reviewSampled(a, head_id, block.ids[1..block.len], self.opts.d1, .d1_accepted);
            self.stats.stage1_blocks += 1;
            self.stats.stage1_progress_tokens += 1 + passed;
        }
    }

    fn stage2Sampled(self: *Executor, advanced: []const u32, a: std.mem.Allocator) !void {
        if (self.opts.stage == .stage1) return;
        for (advanced) |parent_id| {
            if (!self.nodes.contains(parent_id)) continue;
            const parent = self.nodePtr(parent_id);
            if (parent.depth != self.opts.d2 or !parent.growing) continue;
            const main = self.findRole(parent_id, .main) orelse continue;
            const frozen = self.findRole(parent_id, .frozen);
            self.stats.stage2_reviews += 1;
            if (!parent.has_dist[self.opts.d2]) return error.MissingReadout;
            const dist = parent.dists[self.opts.d2][0..self.runner.vocab];
            var mode: u32 = 0;
            var best: f32 = -1;
            for (dist, 0..) |p, i| {
                if (p > best) {
                    best = p;
                    mode = @intCast(i);
                }
            }
            const promote = frozen != null and mode == self.nodePtr(frozen.?).token;
            const fork = !promote and mode != self.nodePtr(main).token;
            if (promote) {
                const fid = frozen.?;
                const fr = self.nodePtr(fid);
                if (fr.frozen_len == 0 or fr.frozen_suffix[0] != fid) return error.PromotionHasNoSuffix;
                var suffix: [max_block]u32 = undefined;
                const slen = fr.frozen_len;
                @memcpy(suffix[0..slen], fr.frozen_suffix[0..slen]);
                fr.promoted = true;
                fr.frozen_len = 0;
                self.stats.promotions += 1;
                self.stats.stage2_disagreements += 1;
                const passed = try self.reviewSampled(a, fid, suffix[1..slen], self.opts.d2, .promotion);
                self.stats.promotion_reused_tokens += 1 + passed;
            } else if (fork) {
                self.stats.stage2_disagreements += 1;
                const parent_pos = self.nodePtr(parent_id).position;
                if (self.opts.stage == .stage2b and parent_pos + 1 < self.output_limit) {
                    if (!self.nodePtr(parent_id).has_dist[self.opts.d1]) return error.MissingReadout;
                    const d1 = self.nodePtr(parent_id).dists[self.opts.d1][0..self.runner.vocab];
                    var residual: [97]f32 = undefined;
                    std.debug.assert(dist.len <= residual.len);
                    var any = false;
                    for (dist, d1, 0..) |p2, p1, i| {
                        residual[i] = @max(p2 - p1, 0);
                        if (i == self.nodePtr(main).token) residual[i] = 0;
                        if (frozen) |fid| {
                            if (i == self.nodePtr(fid).token) residual[i] = 0;
                        }
                        if (residual[i] > 0) any = true;
                    }
                    if (!any) continue;
                    var sum: f32 = 0;
                    for (residual[0..dist.len]) |x| sum += x;
                    if (sum <= 0) continue;
                    for (residual[0..dist.len]) |*x| x.* /= sum;
                    const token = self.draw(residual[0..dist.len]);
                    if (parent.childId(token) != null) return error.DuplicateD2Event;
                    const head = try self.child(parent_id, token, .fork, .d2_fork, self.opts.d2);
                    const hn = self.nodePtr(head);
                    @memcpy(hn.proposal_row[0..dist.len], residual[0..dist.len]);
                    hn.has_proposal = true;
                    try self.queue(parent_id, head, self.opts.d2, .d2_fork);
                    self.stats.forks += 1;
                }
            }
        }
    }

    fn verifySampled(self: *Executor, a: std.mem.Allocator) !bool {
        _ = a;
        var accepted: std.ArrayList(u32) = .empty;
        defer accepted.deinit(self.gpa);
        var removed: std.ArrayList(u32) = .empty;
        defer removed.deinit(self.gpa);
        var stopped = false;
        const R = self.runner.R;
        const V = self.runner.vocab;
        while (self.nodePtr(self.cursor).depth == R) {
            const current = self.cursor;
            const cur = self.nodePtr(current);
            if (!cur.has_dist[R]) return error.MissingReadout;
            const original = cur.dists[R][0..self.runner.vocab];
            var target_buf: [97]f32 = undefined;
            std.debug.assert(original.len <= target_buf.len);
            @memcpy(target_buf[0..original.len], original);
            var target = target_buf[0..original.len];
            var kids: [8]u32 = undefined;
            var kn: u8 = 0;
            for (cur.children[0..cur.child_count]) |c| {
                kids[kn] = c.id;
                kn += 1;
            }
            std.mem.sort(u32, kids[0..kn], self, struct {
                fn lt(ex: *Executor, x: u32, y: u32) bool {
                    const rx = @intFromEnum(ex.nodePtr(x).role);
                    const ry = @intFromEnum(ex.nodePtr(y).role);
                    return rx < ry;
                }
            }.lt);
            var chosen: ?u32 = null;
            var source: []const u8 = "bonus";
            for (kids[0..kn]) |cid| {
                const ch = self.nodePtr(cid);
                const proposal: ?[]const f32 = if (ch.has_proposal) ch.proposal_row[0..V] else null;
                const t = target[ch.token];
                const ratio: f64 = if (proposal) |g| blk: {
                    if (g[ch.token] <= 0) return error.ZeroProposal;
                    break :blk t / g[ch.token];
                } else t;
                if (self.rand01() < ratio) {
                    chosen = cid;
                    source = roleName(ch.role);
                    if (ch.role == .main and ch.ever_frozen and ch.proposal_depth == 0) source = "frozen";
                    self.stats.accepted_candidates += 1;
                    if (ch.promoted) self.stats.promoted_heads_committed += 1;
                    if (ch.ever_frozen) self.stats.previously_frozen_tokens_committed += 1;
                    break;
                }
                if (proposal) |g| {
                    var sum: f32 = 0;
                    for (target, g, 0..) |tv, gv, i| {
                        target[i] = @max(tv - gv, 0);
                        sum += target[i];
                    }
                    if (sum > 0) {
                        for (target) |*x| x.* /= sum;
                    } else @memcpy(target, original);
                } else {
                    target[ch.token] = 0;
                    var sum: f32 = 0;
                    for (target) |x| sum += x;
                    if (sum > 0) {
                        for (target) |*x| x.* /= sum;
                    } else @memcpy(target, original);
                }
            }
            const pick = chosen orelse blk: {
                const token = self.draw(target);
                const id = try self.child(current, token, .main, .bonus, R);
                self.stats.bonus_tokens += 1;
                source = "bonus";
                break :blk id;
            };
            for (self.nodePtr(current).children[0..self.nodePtr(current).child_count]) |c| {
                if (c.id != pick) try self.descendants(c.id, &removed);
            }
            try accepted.append(self.gpa, current);
            self.cursor = pick;
            try self.output.append(self.gpa, self.nodePtr(pick).token);
            self.stats.committed_decode_tokens += 1;
            if (std.mem.eql(u8, source, "main")) self.stats.commit_sources.main += 1;
            if (std.mem.eql(u8, source, "frozen")) self.stats.commit_sources.frozen += 1;
            if (std.mem.eql(u8, source, "fork")) self.stats.commit_sources.fork += 1;
            if (std.mem.eql(u8, source, "bonus")) self.stats.commit_sources.bonus += 1;
            if (std.mem.indexOfScalar(u32, self.eos, self.nodePtr(pick).token) != null or
                self.output.items.len >= self.max_new)
            {
                stopped = true;
                break;
            }
            if (std.mem.eql(u8, source, "bonus")) try self.queue(current, pick, R, .bonus);
        }
        if (accepted.items.len > 0) {
            try self.drop(removed.items);
            try self.runner.commit(accepted.items);
            for (accepted.items) |id| try self.prefix_tokens.append(self.gpa, self.nodePtr(id).token);
            self.nodePtr(self.cursor).parent = null;
            for (accepted.items) |id| _ = self.nodes.remove(id);
            var stale: std.ArrayList(u32) = .empty;
            defer stale.deinit(self.gpa);
            var it = self.blocks.iterator();
            while (it.next()) |kv| {
                if (!self.nodes.contains(kv.key_ptr.*)) try stale.append(self.gpa, kv.key_ptr.*);
            }
            for (stale.items) |id| _ = self.blocks.remove(id);
        }
        return stopped;
    }
};

fn runCase(a: std.mem.Allocator, runner: *TinyRunner, opts: Options, prompt: []const u32, count: u32, eos: []const u32) !Result {
    var ex = try Executor.init(a, runner, opts);
    defer ex.deinit();
    return ex.generate(a, prompt, count, eos);
}

test "tree window rejects R*k > W and accepts equality" {
    try checkTreeWindow(4, 16, .shared_last, 64);
    try checkTreeWindow(4, 16, .full, 0);
    try std.testing.expectError(error.WindowTooSmall, checkTreeWindow(4, 16, .shared_last, 32));
    try std.testing.expectError(error.InvalidWindow, checkTreeWindow(4, 1, .shared_last, 0));
    try checkTreeWindow(2, 8, .shared_last, 16);
    try std.testing.expectError(error.WindowTooSmall, checkTreeWindow(2, 9, .shared_last, 16));
}

test "shared-last history KV is one last-loop copy plus the in-flight window" {
    const layers: u32 = 12;
    const seq: u32 = 2048;
    const width: u32 = 8 * 128 * 2;
    const elem: u32 = 2;
    const full = historyKvBytes(layers, 3, seq, width, elem, .full, 0);
    const shared = historyKvBytes(layers, 3, seq, width, elem, .shared_last, 64);
    try std.testing.expectEqual(full / 3, historyKvBytes(layers, 1, seq, width, elem, .full, 0));
    try std.testing.expect(shared < full);
    const history = @as(u64, layers) * seq * width * elem;
    const window = @as(u64, layers) * 64 * width * elem * 3;
    try std.testing.expectEqual(history + window, shared);
    const ink_full = inklingLoopKvBytes(seq, width, elem, inkling_R, .full, 0);
    const ink_shared = inklingLoopKvBytes(seq, width, elem, inkling_R, .shared_last, 64);
    try std.testing.expectEqual(full, ink_full);
    try std.testing.expect(ink_shared < ink_full);
    try std.testing.expectEqual(@as(u64, 1_207_959_552), inklingLoopKvBytes(4096, 2048, 4, inkling_R, .full, 0));
    try std.testing.expectEqual(@as(u64, 421_527_552), inklingLoopKvBytes(4096, 2048, 4, inkling_R, .shared_last, 64));
}

test "enableOrReject is fail-closed for Flint and unwired Inkling Metal" {
    try std.testing.expectError(error.TwoDsdRequiresLoopedModel, enableOrReject("flint"));
    try std.testing.expectError(error.TwoDsdRequiresLoopedModel, enableOrReject("lfm3"));
    try std.testing.expectError(error.TwoDsdOverlappingDecodeNotWired, enableOrReject("inkling2_loop"));
    try std.testing.expect(loopedFamily("inkling2_loop"));
    try std.testing.expect(!loopedFamily("flint"));
}

test "Cli.apply is off by default and fail-closed when enabled" {
    var c: Cli = .{};
    try c.apply("flint");
    try c.apply("inkling2_loop");
    try std.testing.expectEqual(false, parseEnable("0").?);
    try std.testing.expectEqual(true, parseEnable("1").?);
    try std.testing.expect(parseEnable("yes") == null);
    c.k = 4;
    try std.testing.expectError(error.TwoDsdTuningRequiresEnable, c.apply("flint"));
    c = .{ .enabled = true };
    try std.testing.expectError(error.TwoDsdRequiresLoopedModel, c.apply("flint"));
    try std.testing.expectError(error.TwoDsdRequiresLoopedModel, c.apply("lfm3"));
    try std.testing.expectError(error.TwoDsdOverlappingDecodeNotWired, c.apply("inkling2_loop"));
}

test "all stages and widths match exact causal AR" {
    const a = std.testing.allocator;
    const prompt = [_]u32{ 3, 4, 5 };
    var expect_buf: [37]u32 = undefined;
    const expect_n = arGreedy(&prompt, 37, true, &.{}, &expect_buf);
    for ([_]Stage{ .stage1, .stage2a, .stage2b }) |stage| {
        for ([_]u32{ 1, 4, 8, 16 }) |k| {
            for ([_]bool{ false, true }) |corrupt| {
                var runner = TinyRunner.init(a, 8, true, .growing);
                defer runner.deinit();
                const drafter: DrafterKind = if (k == 1) .none else if (corrupt) .oracle_corrupt else .oracle;
                const result = try runCase(a, &runner, .{
                    .k = k,
                    .stage = stage,
                    .d1 = 1,
                    .d2 = 2,
                    .drafter = drafter,
                }, &prompt, 37, &.{});
                defer a.free(result.output);
                defer a.free(result.trace);
                try std.testing.expectEqualSlices(u32, expect_buf[0..expect_n], result.output);
                try std.testing.expectEqual(@as(u32, 36), result.stats.commit_sources.total());
            }
        }
    }
}

test "LoopSpec d1 and DFlash boundary call counts" {
    const a = std.testing.allocator;
    const cases = [_]struct { k: u32, d1: u32, calls: u32 }{
        .{ .k = 1, .d1 = 1, .calls = 67 },
        .{ .k = 4, .d1 = 1, .calls = 19 },
        .{ .k = 4, .d1 = 4, .calls = 64 },
    };
    for (cases) |c| {
        var runner = TinyRunner.init(a, 8, false, .growing);
        defer runner.deinit();
        const result = try runCase(a, &runner, .{
            .k = c.k,
            .stage = .stage1,
            .d1 = c.d1,
            .drafter = if (c.k > 1) .oracle else .none,
        }, &.{10}, 65, &.{});
        defer a.free(result.output);
        defer a.free(result.trace);
        var expect_buf: [65]u32 = undefined;
        const n = arGreedy(&.{10}, 65, false, &.{}, &expect_buf);
        try std.testing.expectEqualSlices(u32, expect_buf[0..n], result.output);
        try std.testing.expectEqual(c.calls, result.stats.calls);
        if (c.k == 1) try std.testing.expectEqual(@as(u32, 0), result.stats.drafter_batches);
    }
}

test "frozen suffix is retained until final verification" {
    const a = std.testing.allocator;
    var runner = TinyRunner.init(a, 8, false, .promotion);
    defer runner.deinit();
    const result = try runCase(a, &runner, .{
        .k = 4,
        .stage = .stage1,
        .d1 = 1,
        .drafter = .oracle,
        .record_trace = true,
    }, &.{10}, 20, &.{});
    defer a.free(result.output);
    defer {
        for (result.trace) |ev| if (ev.tokens.len > 0) a.free(ev.tokens);
        a.free(result.trace);
    }
    var expect_buf: [20]u32 = undefined;
    for (0..20) |i| expect_buf[i] = @intCast(11 + i);
    try std.testing.expectEqualSlices(u32, &expect_buf, result.output);
    try std.testing.expect(result.stats.frozen_created > 0);
    try std.testing.expect(result.stats.commit_sources.frozen > 0);
    try std.testing.expectEqual(@as(u32, 0), result.stats.promotions);
    var saw_freeze = false;
    for (result.trace) |ev| {
        if (ev.kind == .freeze) {
            try std.testing.expectEqualSlices(u32, &.{ 13, 14 }, ev.tokens);
            saw_freeze = true;
            break;
        }
    }
    try std.testing.expect(saw_freeze);
}

test "promotion reuses the frozen suffix and preserves AR output" {
    const a = std.testing.allocator;
    var runner = TinyRunner.init(a, 8, false, .promotion);
    defer runner.deinit();
    const result = try runCase(a, &runner, .{
        .k = 4,
        .stage = .stage2a,
        .d1 = 1,
        .d2 = 2,
        .drafter = .oracle,
        .record_trace = true,
    }, &.{10}, 20, &.{});
    defer a.free(result.output);
    defer {
        for (result.trace) |ev| if (ev.tokens.len > 0) a.free(ev.tokens);
        a.free(result.trace);
    }
    var expect_buf: [20]u32 = undefined;
    for (0..20) |i| expect_buf[i] = @intCast(11 + i);
    try std.testing.expectEqualSlices(u32, &expect_buf, result.output);
    try std.testing.expect(result.stats.promotions > 0);
    try std.testing.expect(result.stats.promoted_heads_committed > 0);
    try std.testing.expectEqual(@as(u32, 0), result.stats.forks);
    var saw_commit13 = false;
    for (result.trace) |ev| {
        if (ev.kind == .commit and ev.token == 13) {
            try std.testing.expectEqualStrings("frozen", ev.source);
            try std.testing.expectEqual(@as(u32, 4), ev.depth);
            saw_commit13 = true;
        }
    }
    try std.testing.expect(saw_commit13);
}

test "d2 fork is separate from promotion and preserves target output" {
    const a = std.testing.allocator;
    var times: [3]u32 = undefined;
    for ([_]Stage{ .stage1, .stage2a, .stage2b }, 0..) |stage, i| {
        var runner = TinyRunner.init(a, 8, false, .fork);
        defer runner.deinit();
        const result = try runCase(a, &runner, .{
            .k = 4,
            .stage = stage,
            .d1 = 1,
            .d2 = 2,
            .drafter = .oracle,
            .record_trace = true,
        }, &.{10}, 20, &.{});
        defer a.free(result.output);
        defer {
            for (result.trace) |ev| if (ev.tokens.len > 0) a.free(ev.tokens);
            a.free(result.trace);
        }
        var expect_buf: [20]u32 = undefined;
        expect_buf[0] = 11;
        expect_buf[1] = 12;
        for (2..20) |j| expect_buf[j] = @intCast(58 + j); // 60, 61, ...
        try std.testing.expectEqualSlices(u32, &expect_buf, result.output);
        try std.testing.expectEqual(@as(u32, 0), result.stats.promotions);
        times[i] = result.stats.calls;
        if (stage == .stage2b) {
            try std.testing.expect(result.stats.commit_sources.fork > 0);
        } else {
            try std.testing.expectEqual(@as(u32, 0), result.stats.forks);
        }
    }
    try std.testing.expect(times[2] < times[1]);
}

test "no hidden 64-node pruning: capacity grows and output matches AR" {
    const a = std.testing.allocator;
    var runner = TinyRunner.init(a, 8, false, .wide);
    defer runner.deinit();
    const result = try runCase(a, &runner, .{
        .k = 16,
        .stage = .stage2b,
        .d1 = 1,
        .d2 = 2,
        .drafter = .oracle,
    }, &.{10}, 80, &.{});
    defer a.free(result.output);
    defer a.free(result.trace);
    var expect_buf: [80]u32 = undefined;
    const n = arGreedy(&.{10}, 80, false, &.{}, &expect_buf);
    try std.testing.expectEqualSlices(u32, expect_buf[0..n], result.output);
    try std.testing.expect(result.stats.max_live_nodes > 64);
    try std.testing.expect(runner.max_nodes > 64);
}

test "nested promotion rejection freezes again without a second d2 event" {
    const a = std.testing.allocator;
    var runner = TinyRunner.init(a, 8, false, .nested);
    defer runner.deinit();
    const result = try runCase(a, &runner, .{
        .k = 4,
        .stage = .stage2b,
        .d1 = 1,
        .d2 = 2,
        .drafter = .oracle,
        .record_trace = true,
    }, &.{10}, 20, &.{});
    defer a.free(result.output);
    defer {
        for (result.trace) |ev| if (ev.tokens.len > 0) a.free(ev.tokens);
        a.free(result.trace);
    }
    var expect_buf: [20]u32 = undefined;
    for (0..20) |i| expect_buf[i] = @intCast(11 + i);
    try std.testing.expectEqualSlices(u32, &expect_buf, result.output);
    var nested: u32 = 0;
    for (result.trace) |ev| {
        if (ev.kind == .anchor and std.mem.eql(u8, ev.cause, "promotion_reject")) {
            try std.testing.expectEqual(@as(u32, 70), ev.token);
            nested += 1;
        }
        if (ev.kind == .commit and ev.token == 14) {
            try std.testing.expectEqualStrings("frozen", ev.source);
            try std.testing.expectEqual(@as(u32, 4), ev.depth);
        }
    }
    try std.testing.expectEqual(@as(u32, 1), nested);
    try std.testing.expectEqual(@as(u32, 0), result.stats.forks);
}

test "EOS, partial blocks and executor reuse match AR" {
    const a = std.testing.allocator;
    var runner = TinyRunner.init(a, 8, false, .growing);
    defer runner.deinit();
    var ex = try Executor.init(a, &runner, .{ .k = 16, .stage = .stage2b, .d1 = 1, .d2 = 2, .drafter = .oracle });
    defer ex.deinit();
    const cases = [_]struct { count: u32, eos: []const u32 }{
        .{ .count = 1, .eos = &.{} },
        .{ .count = 2, .eos = &.{} },
        .{ .count = 3, .eos = &.{} },
        .{ .count = 30, .eos = &.{11} },
        .{ .count = 30, .eos = &.{13} },
        .{ .count = 40, .eos = &.{} },
    };
    for (cases) |c| {
        const result = try ex.generate(a, &.{10}, c.count, c.eos);
        defer a.free(result.output);
        defer a.free(result.trace);
        var expect_buf: [40]u32 = undefined;
        const n = arGreedy(&.{10}, c.count, false, c.eos, &expect_buf);
        try std.testing.expectEqualSlices(u32, expect_buf[0..n], result.output);
        try std.testing.expectEqual(@as(u32, 0), @as(u32, @intCast(ex.nodes.count())));
        try std.testing.expectEqual(@as(usize, 0), ex.events.items.len);
    }
}

test "history drafter is lossless against AR" {
    const a = std.testing.allocator;
    var runner = TinyRunner.init(a, 8, true, .growing);
    defer runner.deinit();
    const prompt = [_]u32{ 1, 2, 1, 2, 1 };
    const result = try runCase(a, &runner, .{
        .k = 4,
        .stage = .stage2b,
        .d1 = 1,
        .d2 = 2,
        .drafter = .history,
    }, &prompt, 24, &.{});
    defer a.free(result.output);
    defer a.free(result.trace);
    var expect_buf: [24]u32 = undefined;
    const n = arGreedy(&prompt, 24, true, &.{}, &expect_buf);
    try std.testing.expectEqualSlices(u32, expect_buf[0..n], result.output);
}

test "rejection sampling accepts with min(1, t/g) and residual renormalizes" {
    var rng = std.Random.DefaultPrng.init(1);
    const target = [_]f32{ 0.1, 0.7, 0.2 };
    const g_onehot = [_]f32{ 0, 1, 0 };
    // One-hot draft of token 1: accept iff u < t[1] = 0.7.
    var accepts: u32 = 0;
    const N: u32 = 4000;
    for (0..N) |_| {
        if (rng.random().float(f32) < target[1] / g_onehot[1]) accepts += 1;
    }
    const rate = @as(f32, @floatFromInt(accepts)) / @as(f32, @floatFromInt(N));
    try std.testing.expect(@abs(rate - 0.7) < 0.05);
    // Residual t ⊖ g for a rejected one-hot is t without that token.
    var rest = target;
    rest[1] = 0;
    var sum: f32 = 0;
    for (rest) |x| sum += x;
    for (&rest) |*x| x.* /= sum;
    try std.testing.expectApproxEqAbs(@as(f32, 0.1 / 0.3), rest[0], 1e-6);
    try std.testing.expectEqual(@as(f32, 0), rest[1]);
}

test "k=1 sampling on a tiny Markov looped model follows loop-R" {
    const a = std.testing.allocator;
    var chain: [5][5][5]f32 = @splat(@splat(@splat(0)));
    // Depth 4 (served): token t -> (t+1) mod 5 with probability 0.8, else uniform.
    for (1..5) |d| {
        for (0..5) |t| {
            for (0..5) |n| chain[d][t][n] = 0.05;
            chain[d][t][(t + 1) % 5] = 0.8;
        }
    }
    // Shallow d=1 is noisier.
    for (0..5) |t| {
        for (0..5) |n| chain[1][t][n] = 0.2;
    }
    var counts: [5]u32 = @splat(0);
    const runs: u32 = 400;
    for (0..runs) |seed| {
        var runner = TinyRunner.init(a, 8, false, .markov);
        runner.markov = &chain;
        runner.vocab = 5;
        defer runner.deinit();
        const result = try runCase(a, &runner, .{
            .k = 1,
            .stage = .stage1,
            .d1 = 1,
            .temperature = 1,
            .seed = seed,
        }, &.{0}, 1, &.{});
        defer a.free(result.output);
        defer a.free(result.trace);
        try std.testing.expectEqual(@as(usize, 1), result.output.len);
        counts[result.output[0]] += 1;
    }
    // Prefill of [0] at loop R (depth 4) predicts mostly token 1.
    try std.testing.expect(counts[1] > counts[0]);
    try std.testing.expect(counts[1] > runs / 2);
}

test "validate rejects illegal stage/k/top-p combinations" {
    try std.testing.expectError(error.InvalidBlockSize, validate(.{ .k = 0 }, 4));
    try std.testing.expectError(error.InvalidBlockSize, validate(.{ .k = 17 }, 4));
    try std.testing.expectError(error.DrafterRequired, validate(.{ .k = 4, .drafter = .none }, 4));
    try std.testing.expectError(error.InvalidStage2Depths, validate(.{ .k = 1, .stage = .stage2b, .d1 = 1, .d2 = 1 }, 4));
    try std.testing.expectError(error.TopPRequiresSampling, validate(.{ .k = 1, .top_p = 0.9 }, 4));
    try validate(.{ .k = 1, .stage = .stage1, .d1 = 4 }, 4);
    try validate(.{ .k = 16, .stage = .stage2b, .d1 = 1, .d2 = 2, .drafter = .oracle, .kv_policy = .shared_last, .kv_window = 64 }, 4);
}
