const std = @import("std");
const metal = @import("metal.zig");

// Integration: append attention_mlx_wide.metal AFTER attention_mlx.metal in
// Context.initMatched(), and import this file from the production test root.
// Admission is bitwise against the original kernels in matched math mode, not
// against a tolerance/CPU softmax or a separately compiled test-only library.
// Headless Metal only: no checkpoint, server, UI, or subprocess is started.
const dim: usize = 128;
const guard_words: usize = 16;
const guard_bits: u32 = 0x4b123456;
const original_poison: u32 = 0x7fc12345;
const wide_poison: u32 = 0x7fc54321;

const SingleCase = struct {
    tokens: u32,
    pos: ?u32 = null,
    capacity: u32 = 0,
    window: u32 = 0,
    heads: u32 = 2,
    kv_heads: u32 = 1,
    dtype: u32 = 0,
    sink_offset: u64 = 12,
    dominant_sink: bool = false,
};

fn poison(buffer: metal.Buffer, payload: usize, bits: u32) void {
    @memset(buffer.structs(u32)[0..payload], bits);
    @memset(buffer.structs(u32)[payload..], guard_bits);
}

fn expectGuard(buffer: metal.Buffer, payload: usize) !void {
    try std.testing.expectEqual(payload + guard_words, buffer.structs(u32).len);
    for (buffer.structs(u32)[payload..]) |bits|
        try std.testing.expectEqual(guard_bits, bits);
}

fn expectBitwise(original: metal.Buffer, wide: metal.Buffer, payload: usize, label: []const u8, p: metal.Params) !void {
    try expectGuard(original, payload);
    try expectGuard(wide, payload);
    // Separate NaN payloads ensure that identical missed writes cannot pass.
    // Check all head/dimension words, including signed-zero bits. No tolerance.
    for (original.structs(u32)[0..payload], wide.structs(u32)[0..payload], 0..) |wanted, actual, i| {
        if (wanted != actual or !std.math.isFinite(@as(f32, @bitCast(actual)))) {
            std.debug.print("wide MLX {s}: pos={d}, window={d}, capacity={d}, heads={d}, B={d}, dtype={d}, word={d}, original=0x{x}, wide=0x{x}\n", .{
                label, p.pos, p.window, p.capacity, p.heads, p.k, p.dtype, i, wanted, actual,
            });
            try std.testing.expectEqual(wanted, actual);
            try std.testing.expect(std.math.isFinite(@as(f32, @bitCast(actual))));
        }
    }
}

fn writeSinks(sink: metal.Buffer, p: metal.Params, dominant: bool) void {
    // Wrong byte offsets/dtypes encounter NaNs. FP16/BF16 offsets intentionally
    // are not FP32-aligned. All learned sink logits are finite and nonzero.
    @memset(sink.structs(u8), 0xff);
    const offset: usize = @intCast(p.offset);
    for (0..p.heads) |head| {
        const value: f32 = if (dominant)
            12 + @as(f32, @floatFromInt(head)) * 0.25
        else
            -0.75 + @as(f32, @floatFromInt(head)) * 0.375;
        switch (p.dtype) {
            0 => sink.floats()[offset / 4 + head] = value,
            1 => sink.structs(u16)[offset / 2 + head] = @bitCast(@as(f16, @floatCast(value))),
            2 => sink.structs(u16)[offset / 2 + head] = @truncate(@as(u32, @bitCast(value)) >> 16),
            else => unreachable,
        }
    }
}

fn runSingle(ctx: metal.Context, case: SingleCase) !void {
    const p: metal.Params = .{
        .offset = case.sink_offset,
        .dtype = case.dtype,
        .pos = case.pos orelse (case.tokens - 1),
        .capacity = if (case.capacity == 0) case.tokens + 7 else case.capacity,
        .heads = case.heads,
        .kv_heads = case.kv_heads,
        .dim = dim,
        .window = case.window,
        .scale = if (case.dominant_sink) 0.01 else 0.13,
    };
    const end: usize = @as(usize, p.pos) + 1;
    const begin = if (p.window == 0) 0 else end -| p.window;
    try std.testing.expectEqual(@as(usize, case.tokens), end - begin);
    try std.testing.expect(case.tokens <= p.capacity);
    try std.testing.expect(p.heads >= 1 and p.heads <= 2);
    const count = @as(usize, p.heads) * dim;
    const cache_count = @as(usize, p.capacity) * p.kv_heads * dim;
    const sink_element_bytes: usize = if (p.dtype == 0) 4 else 2;
    const sink_words = (@as(usize, @intCast(p.offset)) + p.heads * sink_element_bytes + 3) / 4;
    const q = try ctx.alloc(count + guard_words);
    defer q.deinit();
    const keys = try ctx.alloc(cache_count + guard_words);
    defer keys.deinit();
    const values = try ctx.alloc(cache_count + guard_words);
    defer values.deinit();
    const sink = try ctx.alloc(sink_words + guard_words);
    defer sink.deinit();
    const original = try ctx.alloc(count + guard_words);
    defer original.deinit();
    const wide = try ctx.alloc(count + guard_words);
    defer wide.deinit();
    // Drain encoded work before any referenced buffer is freed on failure.
    defer ctx.finish() catch {};

    poison(q, count, original_poison);
    poison(keys, cache_count, original_poison);
    poison(values, cache_count, original_poison);
    poison(original, count, original_poison);
    poison(wide, count, wide_poison);
    writeSinks(sink, p, case.dominant_sink);
    @memset(sink.structs(u32)[sink_words..], guard_bits);
    const sink_snapshot = try std.testing.allocator.dupe(u32, sink.structs(u32));
    defer std.testing.allocator.free(sink_snapshot);
    for (q.floats()[0..count], 0..) |*value, i|
        value.* = @as(f32, @floatFromInt((i * 19 + i / dim * 7) % 257)) / 128 - 1;
    // Absolute-position data distinguishes wraps; only the live ring prefix is
    // initialized, leaving unused slots poisoned. Full exact-sized rings are
    // intentionally consumed in physical-slot order by BOTH production kernels.
    for (begin..end) |token| {
        for (0..p.kv_heads) |head| {
            for (0..dim) |d| {
                const index = (token % p.capacity * p.kv_heads + head) * dim + d;
                keys.floats()[index] = @as(f32, @floatFromInt((token * 17 + token / 31 * 13 + head * 43 + d * 7) % 257)) / 128 - 1;
                values.floats()[index] = @as(f32, @floatFromInt((token * 29 + token / 17 * 11 + head * 19 + d * 37) % 257)) / 256 - 0.25;
            }
        }
    }
    try ctx.dispatch("attention_mlx_single", &.{ q, keys, values, sink, original }, p, @as(usize, p.heads) * 1024, 1024);
    try ctx.dispatch("attention_mlx_single_wide", &.{ q, keys, values, sink, wide }, p, @as(usize, p.heads) * 1024, 1024);
    try ctx.finish();
    try expectBitwise(original, wide, count, "single", p);
    if (p.capacity == 513 and p.window == 513 and p.pos >= 512) {
        poison(wide, count, wide_poison);
        try ctx.dispatch("attention_mlx_single_ring513", &.{ q, keys, values, sink, wide }, p, @as(usize, p.heads) * 1024, 1024);
        try ctx.finish();
        try expectBitwise(original, wide, count, "physical ring513", p);
    }
    try expectGuard(q, count);
    try expectGuard(keys, cache_count);
    try expectGuard(values, cache_count);
    try expectGuard(sink, sink_words);
    try std.testing.expectEqualSlices(u32, sink_snapshot, sink.structs(u32));
}

const StateCase = enum { varied, tied_maxima, zero_denominator };

fn runMerge(ctx: metal.Context, blocks: u32, heads: u32, case: StateCase) !void {
    const p: metal.Params = .{ .dim = dim, .heads = heads, .kv_heads = 1, .capacity = 1, .k = blocks };
    const count = @as(usize, heads) * dim;
    const state_count = @as(usize, heads) * blocks * 130;
    const states = try ctx.alloc(state_count + guard_words);
    defer states.deinit();
    const original = try ctx.alloc(count + guard_words);
    defer original.deinit();
    const wide = try ctx.alloc(count + guard_words);
    defer wide.deinit();
    defer ctx.finish() catch {};
    poison(states, state_count, original_poison);
    poison(original, count, original_poison);
    poison(wide, count, wide_poison);
    for (0..heads) |head| {
        for (0..blocks) |block| {
            const record = states.floats()[(head * blocks + block) * 130 ..][0..130];
            for (record[0..dim], 0..) |*value, d| {
                // Raw, unnormalized producer-like accumulators with cancellation
                // across lanes, SIMDs, components, blocks, and query heads.
                value.* = @as(f32, @floatFromInt((block * 37 + block / 32 * 11 + head * 53 + d * 19) % 1021)) / 64 - 7.75;
            }
            record[128] = switch (case) {
                .varied => @as(f32, @floatFromInt((block * 29 + block / 32 * 7 + head * 13) % 97)) / 8 - 6,
                .tied_maxima, .zero_denominator => @as(f32, @floatFromInt(head)) * 0.5 - 0.75,
            };
            record[129] = switch (case) {
                .varied => 1 + @as(f32, @floatFromInt((block * 17 + head * 5) % 61)) / 8,
                .tied_maxima => if (block % 7 == 0) 0 else 0.25 + @as(f32, @floatFromInt((block * 13 + head) % 31)) / 16,
                // Synthetic finite states also exercise the denom==0 branch;
                // raw output is deliberately nonzero in this defensive case.
                .zero_denominator => 0,
            };
        }
    }
    for (states.floats()[0..state_count]) |value|
        try std.testing.expect(std.math.isFinite(value));
    const state_snapshot = try std.testing.allocator.dupe(u32, states.structs(u32));
    defer std.testing.allocator.free(state_snapshot);
    // Deliberately bypass split: merge parity must cover every certified B
    // without depending on a producer to happen to generate informative states.
    try ctx.dispatch("attention_mlx_merge", &.{ states, original }, p, @as(usize, heads) * 1024, 1024);
    try ctx.dispatch("attention_mlx_merge_wide", &.{ states, wide }, p, @as(usize, heads) * 1024, 1024);
    try ctx.finish();
    try expectBitwise(original, wide, count, @tagName(case), p);
    try expectGuard(states, state_count);
    try std.testing.expectEqualSlices(u32, state_snapshot, states.structs(u32));
}

const InvalidCase = struct { p: metal.Params, group: usize = 1024, merge_only: bool = false };

fn expectUntouched(buffer: metal.Buffer, payload: usize, bits: u32) !void {
    try expectGuard(buffer, payload);
    for (buffer.structs(u32)[0..payload]) |actual|
        try std.testing.expectEqual(bits, actual);
}

fn runInvalid(ctx: metal.Context, case: InvalidCase) !void {
    // All tested shapes have at most two heads. Keep otherwise valid finite
    // inputs so an accidentally missing early exit is visible as output writes.
    const count = 2 * dim;
    const q = try ctx.alloc(count);
    defer q.deinit();
    const keys = try ctx.alloc(count);
    defer keys.deinit();
    const values = try ctx.alloc(count);
    defer values.deinit();
    const sink = try ctx.alloc(2);
    defer sink.deinit();
    const states = try ctx.alloc(2 * 256 * 130);
    defer states.deinit();
    const original = try ctx.alloc(count + guard_words);
    defer original.deinit();
    const wide = try ctx.alloc(count + guard_words);
    defer wide.deinit();
    defer ctx.finish() catch {};
    @memset(q.floats(), 0.125);
    @memset(keys.floats(), 0.25);
    @memset(values.floats(), 0.5);
    @memset(sink.floats(), -0.75);
    @memset(states.floats(), 0.25);
    // Always dispatch at least one group, including the zero-head rejection.
    const threads = @max(@as(usize, case.p.heads), 1) * case.group;
    if (!case.merge_only) {
        poison(original, count, original_poison);
        poison(wide, count, wide_poison);
        try ctx.dispatch("attention_mlx_single", &.{ q, keys, values, sink, original }, case.p, threads, case.group);
        try ctx.dispatch("attention_mlx_single_wide", &.{ q, keys, values, sink, wide }, case.p, threads, case.group);
        try ctx.finish();
        try expectUntouched(original, count, original_poison);
        try expectUntouched(wide, count, wide_poison);
    }
    poison(original, count, original_poison);
    poison(wide, count, wide_poison);
    try ctx.dispatch("attention_mlx_merge", &.{ states, original }, case.p, threads, case.group);
    try ctx.dispatch("attention_mlx_merge_wide", &.{ states, wide }, case.p, threads, case.group);
    try ctx.finish();
    try expectUntouched(original, count, original_poison);
    try expectUntouched(wide, count, wide_poison);
}

test "wide MLX single preserves every original output bit at SIMD boundaries and sliding rings" {
    const ctx = try metal.Context.initMatched(true);
    defer ctx.deinit();
    for ([_]SingleCase{
        .{ .tokens = 1, .heads = 1, .dominant_sink = true },
        .{ .tokens = 31, .dtype = 1, .sink_offset = 6 },
        .{ .tokens = 32, .heads = 1, .dtype = 2, .sink_offset = 10 },
        .{ .tokens = 33, .kv_heads = 2 },
        .{ .tokens = 513, .dtype = 2, .sink_offset = 10 },
        .{ .tokens = 1023, .dtype = 1, .sink_offset = 6 },
        // Exact-size 513-slot physical-order ring: first wrap and multiple
        // nonaligned wraps. All these positions exceed the window.
        .{ .tokens = 513, .pos = 513, .capacity = 513, .window = 513 },
        .{ .tokens = 513, .pos = 1025, .capacity = 513, .window = 513, .heads = 1, .dtype = 1, .sink_offset = 6 },
        .{ .tokens = 513, .pos = 8390, .capacity = 513, .window = 513, .dtype = 2, .sink_offset = 10, .dominant_sink = true },
        // Capacity exceeds the window: chronological start is nonzero and the
        // live range crosses the physical ring boundary; unused slots are NaN.
        .{ .tokens = 257, .pos = 8390, .capacity = 513, .window = 257 },
    }) |case| try runSingle(ctx, case);
}

test "wide MLX merge preserves every original output bit for finite B64 B128 B512 B1024 states" {
    const ctx = try metal.Context.initMatched(true);
    defer ctx.deinit();
    for ([_]u32{ 64, 128, 512, 1024 }) |blocks| {
        try runMerge(ctx, blocks, 2, .varied);
        try runMerge(ctx, blocks, 1, .tied_maxima);
        try runMerge(ctx, blocks, 2, .zero_denominator);
    }
}

test "wide MLX unsupported dimensions threadgroups and metadata leave outputs untouched" {
    const ctx = try metal.Context.initMatched(true);
    defer ctx.deinit();
    const valid: metal.Params = .{ .dim = dim, .heads = 2, .kv_heads = 1, .capacity = 1, .k = 64 };
    for ([_]u32{ 0, 64, 127, 129 }) |invalid_dim| {
        var p = valid;
        p.dim = invalid_dim;
        try runInvalid(ctx, .{ .p = p });
    }
    for ([_]usize{ 32, 256, 512 }) |group|
        try runInvalid(ctx, .{ .p = valid, .group = group });
    var p = valid;
    p.capacity = 0;
    try runInvalid(ctx, .{ .p = p });
    p = valid;
    p.heads = 0;
    try runInvalid(ctx, .{ .p = p });
    p = valid;
    p.kv_heads = 0;
    try runInvalid(ctx, .{ .p = p });
    p = valid;
    p.kv_heads = 3; // Nonintegral GQA must reject before reading cache data.
    try runInvalid(ctx, .{ .p = p });
    for ([_]u32{ 0, 32, 65, 256 }) |blocks| {
        p = valid;
        p.k = blocks;
        try runInvalid(ctx, .{ .p = p, .merge_only = true });
    }
}
