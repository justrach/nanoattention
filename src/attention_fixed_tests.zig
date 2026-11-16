const std = @import("std");
const metal = @import("metal.zig");

// Integration prerequisites: Context.initMatched() must concatenate
// attention_mlx_fixed.metal AFTER attention_mlx.metal, and the test root must
// import this file. Use the production ABI/compiler settings, not a test-only
// Metal library. Pure headless Metal; no model, server, GUI, or subprocess.
const dim: u32 = 128;
const kv_heads: u32 = 4;
const heads: u32 = kv_heads * 8;
const group: usize = 32 * 8;
const guard_words: usize = 16;
const guard_bits: u32 = 0x4b123456;
const baseline_poison: u32 = 0x7fc12345;
const candidate_poison: u32 = 0x7fc54321;

const Case = struct {
    tokens: u32,
    dtype: u32 = 0,
    sink_offset: u64 = 12,
    extra_capacity: u32 = 7,
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

fn expectBitwise(baseline: metal.Buffer, candidate: metal.Buffer, payload: usize, label: []const u8, p: metal.Params) !void {
    try expectGuard(baseline, payload);
    try expectGuard(candidate, payload);
    // Check every producer word, including all raw output dimensions, maxima,
    // denominators and empty stripes. Different NaN poisons catch missing writes
    // even if BOTH implementations miss the same logical output. No tolerance.
    for (baseline.structs(u32)[0..payload], candidate.structs(u32)[0..payload], 0..) |wanted, actual, i| {
        if (wanted != actual or !std.math.isFinite(@as(f32, @bitCast(actual)))) {
            std.debug.print("fixed attention {s}: N={d}, B={d}, dtype={d}, word={d}, original=0x{x}, fixed=0x{x}\n", .{
                label, p.pos + 1, p.k, p.dtype, i, wanted, actual,
            });
            try std.testing.expectEqual(wanted, actual);
            try std.testing.expect(std.math.isFinite(@as(f32, @bitCast(actual))));
        }
    }
}

fn writeSink(sink: metal.Buffer, p: metal.Params, dominant: bool) void {
    // Distinct per-head learned sinks with nonzero byte offsets verify the
    // existing sink ABI, including offsets unaligned for FP32 in 16-bit cases.
    @memset(sink.structs(u8), 0xff);
    const offset: usize = @intCast(p.offset);
    for (0..heads) |head| {
        const value: f32 = if (dominant)
            12 + @as(f32, @floatFromInt(head)) * 0.125
        else
            -0.75 + @as(f32, @floatFromInt(head)) * 0.0625;
        switch (p.dtype) {
            0 => sink.floats()[offset / 4 + head] = value,
            1 => sink.structs(u16)[offset / 2 + head] = @bitCast(@as(f16, @floatCast(value))),
            2 => sink.structs(u16)[offset / 2 + head] = @truncate(@as(u32, @bitCast(value)) >> 16),
            else => unreachable,
        }
    }
}

fn runCase(ctx: metal.Context, case: Case) !void {
    var p: metal.Params = .{
        .offset = case.sink_offset,
        .dtype = case.dtype,
        .pos = case.tokens - 1,
        .capacity = case.tokens + case.extra_capacity,
        .heads = heads,
        .kv_heads = kv_heads,
        .dim = dim,
        .window = 0,
        .scale = if (case.dominant_sink) 0.01 else 0.13,
    };
    const query_count: usize = heads * dim;
    const cache_count = @as(usize, p.capacity) * kv_heads * dim;
    const sink_bytes: usize = if (p.dtype == 0) 4 else 2;
    const sink_words = (@as(usize, @intCast(p.offset)) + heads * sink_bytes + 3) / 4;
    const q = try ctx.alloc(query_count + guard_words);
    defer q.deinit();
    const keys = try ctx.alloc(cache_count + guard_words);
    defer keys.deinit();
    const values = try ctx.alloc(cache_count + guard_words);
    defer values.deinit();
    const sink = try ctx.alloc(sink_words + guard_words);
    defer sink.deinit();
    const original_out = try ctx.alloc(query_count + guard_words);
    defer original_out.deinit();
    const fixed_out = try ctx.alloc(query_count + guard_words);
    defer fixed_out.deinit();
    defer ctx.finish() catch {};

    poison(q, query_count, baseline_poison);
    poison(keys, cache_count, baseline_poison);
    poison(values, cache_count, baseline_poison);
    writeSink(sink, p, case.dominant_sink);
    @memset(sink.structs(u32)[sink_words..], guard_bits);
    const sink_snapshot = try std.testing.allocator.dupe(u32, sink.structs(u32));
    defer std.testing.allocator.free(sink_snapshot);
    for (q.floats()[0..query_count], 0..) |*v, i|
        v.* = @as(f32, @floatFromInt((i * 19 + i / dim * 7) % 257)) / 128 - 1;
    // Only active tokens are initialized. Unused capacity stays NaN, exposing
    // accidental reads past the linear prefix as well as incorrect tail bounds.
    for (0..case.tokens) |t| {
        for (0..kv_heads) |head| {
            for (0..dim) |d| {
                const index = (t * kv_heads + head) * dim + d;
                keys.floats()[index] = @as(f32, @floatFromInt((t * 17 + t / 31 * 13 + head * 43 + d * 7) % 257)) / 128 - 1;
                values.floats()[index] = @as(f32, @floatFromInt((t * 29 + t / 17 * 11 + head * 19 + d * 37) % 257)) / 256 - 0.25;
            }
        }
    }

    for ([_]u32{ 128, 512, 1024 }) |blocks| {
        p.k = blocks; // Identical B for original and candidate; never retile.
        const state_count = @as(usize, heads) * blocks * 130;
        const original_states = try ctx.alloc(state_count + guard_words);
        defer original_states.deinit();
        const fixed_states = try ctx.alloc(state_count + guard_words);
        defer fixed_states.deinit();
        // Drain encoded work before freeing any buffers on a dispatch failure.
        defer ctx.finish() catch {};
        poison(original_states, state_count, baseline_poison);
        poison(fixed_states, state_count, candidate_poison);
        poison(original_out, query_count, baseline_poison);
        poison(fixed_out, query_count, candidate_poison);

        const threads = @as(usize, kv_heads) * blocks * group;
        // Exactly one producer and one unchanged merge per implementation per
        // shape. No warmup, benchmark loop, model load, or CPU softmax needed:
        // this test specifically gates bitwise parity with the existing kernel.
        try ctx.dispatch("attention_mlx_split", &.{ q, keys, values, sink, original_states }, p, threads, group);
        try ctx.dispatch("attention_mlx_merge", &.{ original_states, original_out }, p, @as(usize, heads) * 1024, 1024);
        try ctx.dispatch("attention_mlx_split_fixed", &.{ q, keys, values, sink, fixed_states }, p, threads, group);
        try ctx.dispatch("attention_mlx_merge", &.{ fixed_states, fixed_out }, p, @as(usize, heads) * 1024, 1024);
        try ctx.finish();

        try expectBitwise(original_states, fixed_states, state_count, "producer states", p);
        try expectBitwise(original_out, fixed_out, query_count, "merged output", p);
        try expectGuard(q, query_count);
        try expectGuard(keys, cache_count);
        try expectGuard(values, cache_count);
        try expectGuard(sink, sink_words);
        try std.testing.expectEqualSlices(u32, sink_snapshot, sink.structs(u32));
    }
}

test "fixed MLX GQA8 KV4 D128 B128/B512/B1024 preserves producer and merge bits at 1K/16K/32K and tails" {
    // Matched production math mode, unchanged for BOTH implementations. Context
    // ownership is local and always closed; this test never starts a server.
    const ctx = try metal.Context.initMatched(true);
    defer ctx.deinit();
    for ([_]Case{
        .{ .tokens = 1, .dominant_sink = true }, // Empty stripes and sink once.
        .{ .tokens = 1024 },
        .{ .tokens = 1025, .dtype = 1, .sink_offset = 6 },
        .{ .tokens = 4097, .dtype = 2, .sink_offset = 10, .dominant_sink = true }, // TK4 then tail for every B.
        .{ .tokens = 16384, .extra_capacity = 0 }, // Exact-sized linear cache.
        .{ .tokens = 32768, .dtype = 2, .sink_offset = 10 },
        .{ .tokens = 32769, .dtype = 1, .sink_offset = 6 }, // Long TK4 plus scalar tail.
    }) |case| try runCase(ctx, case);
}
