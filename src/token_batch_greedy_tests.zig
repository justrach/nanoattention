//! Isolated, headless GPU parity gates. Not imported by production code.
//! Compile-only: zig test nanoattention/src/token_batch_greedy_tests.zig -fno-emit-bin
//! Running requires metal.m linkage and explicit scheduling by the coordinator.
const std = @import("std");
const metal = @import("metal.zig");
const sampling = @import("sampling.zig");
extern fn fm_open([*:0]const u8) ?*anyopaque;
extern fn fm_error() [*:0]const u8;

// Independent mirror avoids importing unrelated token-batch GPU test suites.
const TokenScratch = extern struct {
    x: u64 = 0,
    norm: u64 = 0,
    q: u64 = 0,
    k: u64 = 0,
    v: u64 = 0,
    attention: u64 = 0,
    projection: u64 = 0,
    activation: u64 = 0,
    router: u64 = 0,
    route_ids: u64 = 0,
    route_weights: u64 = 0,
    route_error: u64 = 0,
    logits: u64 = 0,
};
comptime {
    const fields = [_][]const u8{ "x", "norm", "q", "k", "v", "attention", "projection", "activation", "router", "route_ids", "route_weights", "route_error", "logits" };
    if (@sizeOf(TokenScratch) != 104 or @alignOf(TokenScratch) != 8)
        @compileError("TokenScratch must be thirteen GPU addresses, size 104, align 8");
    for (fields, 0..) |field, i| {
        if (@offsetOf(TokenScratch, field) != 8 * i)
            @compileError("TokenScratch field offset mismatch");
    }
}
const max_vocab = 150144;
const batches = [_]usize{ 1, 2, 3, 4, 5, 6, 7, 8, 16, 31, 32 };
const guards = 8;
const poison: u32 = 0xdeadbeef;
// Nonfinite guards also detect accidental inclusion of prefix/tail padding.
const guard: u32 = 0x7fc12345;
const vocabs = [_]usize{ 1, 31, 32, 33, 257, max_vocab };
const nonfinite = [_]u32{ 0x7fc00001, 0xffc00001, 0x7f800001, 0xff800001, 0x7f800000, 0xff800000 };

fn context() !metal.Context {
    const source = @embedFile("kernels.metal") ++ "\n" ++
        @embedFile("resident_plan.metal") ++ "\n" ++
        @embedFile("row_tiling.metal") ++ "\n" ++
        @embedFile("token_batch.metal") ++ "\n" ++
        @embedFile("token_batch_greedy.metal");
    return .{ .handle = fm_open(source) orelse {
        std.debug.print("Metal batch greedy: {s}\n", .{std.mem.span(fm_error())});
        return error.MetalFailure;
    } };
}

const Fixture = struct {
    ctx: metal.Context,
    rows: [32]metal.Buffer,
    scratch: metal.Buffer,
    results: metal.Buffer,

    fn init(ctx: metal.Context) !Fixture {
        var rows: [32]metal.Buffer = undefined;
        var initialized: usize = 0;
        errdefer for (rows[0..initialized]) |row| row.deinit();
        for (&rows) |*row| {
            row.* = try ctx.alloc(max_vocab + 2 * guards);
            initialized += 1;
        }
        const scratch = try ctx.alloc(32 * @sizeOf(TokenScratch) / 4);
        errdefer scratch.deinit();
        const results = try ctx.alloc(64 + guards);
        for (rows, 0..) |row, t|
            scratch.structs(TokenScratch)[t] = .{ .logits = row.address() + 4 * guards };
        return .{ .ctx = ctx, .rows = rows, .scratch = scratch, .results = results };
    }
    fn deinit(self: Fixture) void {
        // Complete any submitted work even if an assertion or dispatch failed.
        self.ctx.finish() catch {};
        for (self.rows) |row| row.deinit();
        self.scratch.deinit();
        self.results.deinit();
    }
    fn logits(self: Fixture, token: usize, n: usize) []f32 {
        return self.rows[token].floats()[guards..][0..n];
    }
    fn reset(self: Fixture, n: usize) void {
        for (self.rows, 0..) |row, t| {
            @memset(row.structs(u32), guard);
            @memset(self.logits(t, n), -7);
            self.logits(t, n)[0] = 10;
        }
        @memset(self.results.structs(u32), poison);
    }
    fn dispatch(self: Fixture, p: metal.Params, groups: usize, threads: usize) !void {
        // Logits are indirect GPU-address resources; buffers 0/1 are direct.
        for (self.rows) |row| try self.ctx.resident(row);
        const before = self.ctx.stats();
        try self.ctx.dispatch("token_batch_greedy", &.{ self.scratch, self.results }, p, groups * threads, threads);
        try self.ctx.finish();
        const after = self.ctx.stats();
        try std.testing.expectEqual(@as(u64, 1), after.dispatches - before.dispatches);
        try std.testing.expectEqual(@as(u64, 1), after.submissions - before.submissions);
    }
    fn check(self: Fixture, batch: usize, n: usize) !void {
        try self.dispatch(.{ .m = @intCast(batch), .n = @intCast(n) }, batch, 256);
        for (0..batch) |t| {
            const expected = sampling.greedy(self.logits(t, n));
            if (expected) |id| {
                try std.testing.expectEqual(@as(u32, 0), self.results.structs(u32)[2 * t + 1]);
                try std.testing.expectEqual(id, self.results.structs(u32)[2 * t]);
            } else |err| {
                try std.testing.expectEqual(error.NonFiniteLogits, err);
                // The ID is deliberately not consumed on invalid rows.
                try std.testing.expectEqual(@as(u32, 1), self.results.structs(u32)[2 * t + 1]);
            }
        }
        for (self.results.structs(u32)[2 * batch ..]) |word|
            try std.testing.expectEqual(poison, word);
        for (self.rows) |row| {
            for (row.structs(u32)[0..guards]) |word| try std.testing.expectEqual(guard, word);
            for (row.structs(u32)[guards + n ..]) |word| try std.testing.expectEqual(guard, word);
        }
    }
};

const Pattern = enum { mixed, negative, positive, floor, all_zero, zero_ties, first, last, ties, random_bits };
fn fill(values: []f32, token: usize, pattern: Pattern) void {
    var state: u32 = @as(u32, @intCast(token)) + 12345;
    for (values, 0..) |*value, i| {
        const magnitude: f32 = @floatFromInt((i * 17 + token * 43) % 1009 + 1);
        state = state *% 1664525 +% 1013904223;
        var bits = state;
        if (bits & 0x7f800000 == 0x7f800000) bits ^= 0x00800000;
        value.* = switch (pattern) {
            .mixed => magnitude - 500,
            .negative => -magnitude,
            .positive => magnitude,
            .floor => -std.math.floatMax(f32),
            .all_zero => @bitCast(@as(u32, if ((i + token) % 2 == 0) 0x80000000 else 0)),
            .zero_ties, .first, .last, .ties => -7,
            .random_bits => @bitCast(bits),
        };
    }
    switch (pattern) {
        .first => values[0] = std.math.floatMax(f32),
        .last => values[values.len - 1] = std.math.floatMax(f32),
        .zero_ties, .ties => {
            // Distinct SIMD lanes, SIMD groups, and per-thread stride segments.
            const candidates = [_]usize{ 1, 31, 32, 33, 255, 256, 511, 512, values.len / 2, values.len - 1 };
            for (candidates, 0..) |i, j| {
                if (i < values.len) values[i] = if (pattern == .ties) 9 else @bitCast(@as(u32, if ((j + token) % 2 == 0) 0x80000000 else 0));
            }
        },
        else => {},
    }
}

test "batch greedy exact CPU parity through batch 32, vocabulary tails, extrema and ties" {
    const ctx = try context();
    defer ctx.deinit();
    const f = try Fixture.init(ctx);
    defer f.deinit();
    for (batches) |batch| {
        for (vocabs) |n| {
            for (std.enums.values(Pattern)) |pattern| {
                f.reset(n);
                for (0..batch) |t| fill(f.logits(t, n), t, pattern);
                try f.check(batch, n);
            }
            // A later index in a lower SIMD lane must NOT beat an earlier ID.
            const pairs = [_][2]usize{ .{ 31, 32 }, .{ 255, 256 }, .{ 32, 256 }, .{ 257, 512 }, .{ 511, n - 1 } };
            for (pairs) |pair| {
                if (pair[0] >= n or pair[1] >= n) continue;
                f.reset(n);
                for (0..batch) |t| {
                    const values = f.logits(t, n);
                    @memset(values, -7);
                    values[pair[0]] = 20;
                    values[pair[1]] = 20;
                }
                try f.check(batch, n);
            }
        }
    }
}

test "batch greedy rejects nonfinite headers and tails for every batch and vocabulary" {
    const ctx = try context();
    defer ctx.deinit();
    const f = try Fixture.init(ctx);
    defer f.deinit();
    for (batches) |batch| {
        for (vocabs) |n| {
            for (nonfinite) |bits| {
                for (0..3) |location| {
                    f.reset(n);
                    // Mix valid and invalid rows: flags may not leak between
                    // tokens and an invalid rejected row may not hide a peer.
                    for (0..batch) |t| {
                        if (t % 2 == 1 and location != 2) continue;
                        const index = switch (location) {
                            0 => 0,
                            1 => n - 1,
                            else => (t * 37 + 1) % n,
                        };
                        f.logits(t, n)[index] = @bitCast(bits);
                    }
                    try f.check(batch, n);
                }
            }
        }
    }
    // Every physical lane in all eight SIMD groups, including the long-vocab
    // tail's final stride iteration. One bad logit/row; all other logits finite,
    // with an earlier finite winner at index 0 whenever the bad index is > 0.
    for ([_]usize{ 257, max_vocab }) |n| {
        for ([_]usize{ 0, n - 256 }) |start| {
            for (nonfinite) |bits| {
                for (0..32) |block| {
                    f.reset(n);
                    for (0..8) |t|
                        f.logits(t, n)[start + block * 8 + t] = @bitCast(bits);
                    try f.check(8, n);
                }
            }
        }
    }
}

test "batch greedy uniform invalid parameters and geometry leave poisoned output untouched" {
    const ctx = try context();
    defer ctx.deinit();
    const f = try Fixture.init(ctx);
    defer f.deinit();
    const Case = struct { batch: u32 = 2, vocab: u32 = 33, groups: usize = 2, threads: usize = 256 };
    const cases = [_]Case{
        .{ .batch = 0 },    .{ .batch = 33 },    .{ .batch = std.math.maxInt(u32) },
        .{ .vocab = 0 },    .{ .groups = 1 },    .{ .groups = 3 },
        .{ .threads = 32 }, .{ .threads = 128 }, .{ .threads = 512 },
    };
    for (cases) |case| {
        f.reset(33);
        // Keep valid finite logits so a missing uniform guard would write a
        // result instead of being accidentally hidden by a null-pointer guard.
        try f.dispatch(.{ .m = case.batch, .n = case.vocab }, case.groups, case.threads);
        for (f.results.structs(u32)) |word| try std.testing.expectEqual(poison, word);
    }
}
