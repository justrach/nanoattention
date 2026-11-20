//! Host sampler over the logits read back from the GPU, in the order Hugging
//! Face's generate applies them: repetition penalty (over every token in the
//! sequence so far, prompt included), then temperature, top-k, top-p, then a
//! draw from the softmax with a seeded generator. Temperature 0 with no
//! penalty is greedy, which the CLI serves from the engine's GPU argmax and
//! never through this file.
const std = @import("std");
const Allocator = std.mem.Allocator;

pub const Params = struct {
    temperature: f32 = 0,
    /// 0 = off.
    top_k: u32 = 0,
    /// 1 = off.
    top_p: f32 = 1,
    /// 1 = off. Logits of seen tokens: positive divided by it, negative multiplied.
    repetition_penalty: f32 = 1,
    seed: u64 = 0,

    pub fn greedy(self: Params) bool {
        return self.temperature == 0 and self.repetition_penalty == 1;
    }

    pub fn validate(self: Params) !void {
        if (!(self.temperature >= 0) or !std.math.isFinite(self.temperature)) return error.InvalidTemperature;
        if (!(self.top_p > 0 and self.top_p <= 1)) return error.InvalidTopP;
        if (!(self.repetition_penalty > 0) or !std.math.isFinite(self.repetition_penalty)) return error.InvalidRepetitionPenalty;
    }

    /// Fields of a generation_config.json (do_sample false means greedy).
    pub fn fromGenerationConfig(a: Allocator, bytes: []const u8) !Params {
        const v = try std.json.parseFromSliceLeaky(std.json.Value, a, bytes, .{});
        if (v != .object) return error.InvalidGenerationConfig;
        const o = v.object;
        var p: Params = .{};
        const do_sample = if (o.get("do_sample")) |d| d == .bool and d.bool else false;
        if (do_sample) {
            p.temperature = if (o.get("temperature")) |t| try number(t) else 1;
            if (o.get("top_k")) |t| p.top_k = @intFromFloat(try number(t));
            if (o.get("top_p")) |t| p.top_p = try number(t);
        }
        if (o.get("repetition_penalty")) |t| p.repetition_penalty = try number(t);
        try p.validate();
        return p;
    }
};

fn number(v: std.json.Value) !f32 {
    return switch (v) {
        .integer => |n| @floatFromInt(n),
        .float => |f| @floatCast(f),
        else => error.InvalidGenerationConfig,
    };
}

const Candidate = struct { id: u32, logit: f32 };

pub const Sampler = struct {
    params: Params,
    rng: std.Random.DefaultPrng,
    seen: std.DynamicBitSetUnmanaged,
    scratch: []f32,
    candidates: std.ArrayList(Candidate) = .empty,

    pub fn init(gpa: Allocator, params: Params, vocab: u32) !Sampler {
        try params.validate();
        const scratch = try gpa.alloc(f32, vocab);
        errdefer gpa.free(scratch);
        return .{ .params = params, .rng = .init(params.seed), .seen = try .initEmpty(gpa, vocab), .scratch = scratch };
    }
    pub fn deinit(self: *Sampler, gpa: Allocator) void {
        self.seen.deinit(gpa);
        gpa.free(self.scratch);
        self.candidates.deinit(gpa);
    }
    /// Tokens entering the sequence (prompt and generated), for the penalty.
    pub fn observe(self: *Sampler, ids: []const u32) void {
        for (ids) |id| if (id < self.seen.bit_length) self.seen.set(id);
    }
    pub fn reset(self: *Sampler) void {
        self.seen.unsetAll();
    }

    pub fn sample(self: *Sampler, gpa: Allocator, logits: []const f32) !u32 {
        const p = self.params;
        const x = self.scratch[0..logits.len];
        @memcpy(x, logits);
        for (x) |v| if (!std.math.isFinite(v)) return error.NonFiniteLogits;
        if (p.repetition_penalty != 1) {
            var it = self.seen.iterator(.{});
            while (it.next()) |id| {
                if (id >= x.len) break;
                x[id] = if (x[id] < 0) x[id] * p.repetition_penalty else x[id] / p.repetition_penalty;
            }
        }
        if (p.temperature == 0) return argmax(x);
        for (x) |*v| v.* /= p.temperature;
        // Candidates: the top k (ties at the k-th value kept, as HF's top-k does), or all.
        const c = &self.candidates;
        c.clearRetainingCapacity();
        if (p.top_k > 0 and p.top_k < x.len) {
            const kth = try kthLargest(gpa, x, p.top_k);
            for (x, 0..) |v, i| if (v >= kth) try c.append(gpa, .{ .id = @intCast(i), .logit = v });
        } else {
            try c.ensureTotalCapacity(gpa, x.len);
            for (x, 0..) |v, i| c.appendAssumeCapacity(.{ .id = @intCast(i), .logit = v });
        }
        var max: f32 = -std.math.inf(f32);
        for (c.items) |e| max = @max(max, e.logit);
        var total: f64 = 0;
        for (c.items) |*e| {
            e.logit = @exp(e.logit - max);
            total += e.logit;
        }
        if (p.top_p < 1) {
            // HF: in ascending order, drop tokens while the cumulative probability is <= 1 - top_p; keep at least one.
            // Tokens below a cutoff whose total mass is itself <= 1 - top_p come first in that order and are all
            // dropped, so only the rest is sorted (the whole vocabulary otherwise: ~10 ms a token).
            var below: f64 = 0;
            var cutoff: f32 = 1.0 / 1024.0;
            while (cutoff > 1e-30) : (cutoff /= 1024.0) {
                below = 0;
                for (c.items) |e| if (e.logit < cutoff) {
                    below += e.logit;
                };
                if (below / total <= 1 - p.top_p) break;
            } else {
                cutoff = 0;
                below = 0;
            }
            if (cutoff > 0) {
                var n: usize = 0;
                for (c.items) |e| if (e.logit >= cutoff) {
                    c.items[n] = e;
                    n += 1;
                };
                c.items.len = n;
            }
            std.sort.pdq(Candidate, c.items, {}, struct {
                fn lt(_: void, l: Candidate, r: Candidate) bool {
                    return l.logit < r.logit or (l.logit == r.logit and l.id > r.id);
                }
            }.lt);
            var cum: f64 = below / total;
            var drop: usize = 0;
            while (drop + 1 < c.items.len) : (drop += 1) {
                cum += c.items[drop].logit / total;
                if (cum > 1 - p.top_p) break;
            }
            var kept: f64 = 0;
            for (c.items[drop..]) |e| kept += e.logit;
            std.mem.copyForwards(Candidate, c.items[0 .. c.items.len - drop], c.items[drop..]);
            c.items.len -= drop;
            total = kept;
        }
        var r = self.rng.random().float(f64) * total;
        for (c.items) |e| {
            r -= e.logit;
            if (r < 0) return e.id;
        }
        return c.items[c.items.len - 1].id;
    }
};

/// Lowest index of the maximum.
pub fn argmax(x: []const f32) u32 {
    var best: u32 = 0;
    for (x, 0..) |v, i| if (v > x[best]) {
        best = @intCast(i);
    };
    return best;
}

/// The k-th largest value, through a size-k min-heap.
fn kthLargest(gpa: Allocator, x: []const f32, k: u32) !f32 {
    var h: std.ArrayList(f32) = .empty;
    defer h.deinit(gpa);
    try h.ensureTotalCapacity(gpa, k);
    for (x) |v| {
        if (h.items.len < k) {
            h.appendAssumeCapacity(v);
            var i = h.items.len - 1;
            while (i > 0) {
                const parent = (i - 1) / 2;
                if (h.items[parent] <= h.items[i]) break;
                std.mem.swap(f32, &h.items[parent], &h.items[i]);
                i = parent;
            }
        } else if (v > h.items[0]) {
            h.items[0] = v;
            var i: usize = 0;
            while (true) {
                const l = 2 * i + 1;
                if (l >= k) break;
                var m = l;
                if (l + 1 < k and h.items[l + 1] < h.items[l]) m = l + 1;
                if (h.items[i] <= h.items[m]) break;
                std.mem.swap(f32, &h.items[i], &h.items[m]);
                i = m;
            }
        }
    }
    return h.items[0];
}

test "sampler: penalty, top-k, top-p, seeded determinism" {
    const gpa = std.testing.allocator;
    var s = try Sampler.init(gpa, .{ .temperature = 0, .repetition_penalty = 2 }, 4);
    defer s.deinit(gpa);
    s.observe(&.{0});
    // 3.0 / 2 = 1.5 < 2.0: the penalty moves the argmax.
    try std.testing.expectEqual(@as(u32, 1), try s.sample(gpa, &.{ 3.0, 2.0, -1.0, 0.0 }));
    var k1 = try Sampler.init(gpa, .{ .temperature = 1, .top_k = 1, .seed = 7 }, 5);
    defer k1.deinit(gpa);
    for (0..20) |_| try std.testing.expectEqual(@as(u32, 3), try k1.sample(gpa, &.{ 0, 1, 2, 9, 3 }));
    var p = try Sampler.init(gpa, .{ .temperature = 1, .top_p = 0.5, .seed = 1 }, 3);
    defer p.deinit(gpa);
    // Probabilities about 0.84 / 0.11 / 0.04: top-p 0.5 keeps only the first.
    for (0..20) |_| try std.testing.expectEqual(@as(u32, 0), try p.sample(gpa, &.{ 4, 2, 1 }));
    // Many tiny tokens below the cutoff: still only the first survives top-p 0.5.
    var wide = try Sampler.init(gpa, .{ .temperature = 1, .top_p = 0.5, .seed = 2 }, 3000);
    defer wide.deinit(gpa);
    var many: [3000]f32 = @splat(-20);
    many[1234] = 10;
    many[7] = 9;
    for (0..20) |_| try std.testing.expectEqual(@as(u32, 1234), try wide.sample(gpa, &many));
    var a = try Sampler.init(gpa, .{ .temperature = 1, .seed = 42 }, 8);
    defer a.deinit(gpa);
    var b = try Sampler.init(gpa, .{ .temperature = 1, .seed = 42 }, 8);
    defer b.deinit(gpa);
    const logits = [_]f32{ 0.1, 0.5, 0.2, 0.9, 0.3, 0.0, 0.4, 0.8 };
    for (0..50) |_| try std.testing.expectEqual(try a.sample(gpa, &logits), try b.sample(gpa, &logits));
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const g = try Params.fromGenerationConfig(arena.allocator(), "{\"do_sample\":true,\"temperature\":0.1,\"top_k\":50,\"repetition_penalty\":1.1}");
    try std.testing.expectEqual(@as(u32, 50), g.top_k);
    try std.testing.expectApproxEqAbs(@as(f32, 0.1), g.temperature, 1e-7);
    try std.testing.expect(!g.greedy());
    try std.testing.expect((try Params.fromGenerationConfig(arena.allocator(), "{\"do_sample\":false}")).greedy());
}
