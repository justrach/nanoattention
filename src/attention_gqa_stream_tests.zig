const std = @import("std");
const metal = @import("metal.zig");

// Integration prerequisites (owned by the root): append attention_gqa_stream.metal
// AFTER attention_gqa_reuse.metal in Context.initMatched(), then import this
// file from the test root. Both use the SAME production library/math flags.
// Compare every scratch and merged output bit to the ordinary TK4 GQA2, not
// to a CPU softmax or a differently partitioned producer. Pure headless Metal;
// no server/UI. These are parity gates, not benchmarks or a bit-parity claim.
// One KV head/eight queries keeps max K/V allocation near 32 MiB total. Fill
// shared buffers directly and never construct a large CPU reference/copy.
const dim: u32 = 128;
const kv_heads: u32 = 1;
const heads: u32 = kv_heads * 8;
const guard_words: usize = 16;
const guard_bits: u32 = 0x4b123456;
const original_poison: u32 = 0x7fc12345;
const candidate_poison: u32 = 0x7fc54321;
const blocks_list = [_]u32{ 128, 512, 1024 };
const variants = [_]struct { kernel: [:0]const u8, group: usize }{
    .{ .kernel = "attention_mlx_split_gqa2_tk1", .group = 128 },
    .{ .kernel = "attention_mlx_split_gqa2_tk2", .group = 128 },
    .{ .kernel = "attention_mlx_split_gqa4_tk1", .group = 64 },
    .{ .kernel = "attention_mlx_split_gqa4_tk2", .group = 64 },
};
const group_size: usize = 128;

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

fn expectBitwise(original: metal.Buffer, candidate: metal.Buffer, payload: usize, label: []const u8, kernel: []const u8, p: metal.Params) !void {
    try expectGuard(original, payload);
    try expectGuard(candidate, payload);
    // Every word, not a sampled/tolerance comparison: raw_o[128], max, sum in
    // EVERY stripe/head, including empty stripes; then every merged output.
    // Distinct poisons catch unwritten records even if both producers omit one.
    for (original.structs(u32)[0..payload], candidate.structs(u32)[0..payload], 0..) |wanted, actual, i| {
        if (wanted != actual or !std.math.isFinite(@as(f32, @bitCast(actual)))) {
            std.debug.print("GQA stream {s} {s}: N={d}, B={d}, dtype={d}, word={d}, original=0x{x}, candidate=0x{x}\n", .{
                kernel, label, p.pos + 1, p.k, p.dtype, i, wanted, actual,
            });
            try std.testing.expectEqual(wanted, actual);
            try std.testing.expect(std.math.isFinite(@as(f32, @bitCast(actual))));
        }
    }
}

fn writeSink(sink: metal.Buffer, p: metal.Params, dominant: bool) void {
    @memset(sink.structs(u8), 0xff);
    const offset: usize = @intCast(p.offset);
    for (0..heads) |head| {
        // Negative case is negative for EVERY head, not a mixture of signs.
        // Dominant case overwhelms bounded token scores at scale=0.01 and
        // makes missing/duplicated sinks or normalized producer scratch visible.
        const value: f32 = if (dominant)
            12 + @as(f32, @floatFromInt(head)) * 0.125
        else
            -2 - @as(f32, @floatFromInt(head)) * 0.0625;
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
    const candidate_out = try ctx.alloc(query_count + guard_words);
    defer candidate_out.deinit();
    // On dispatch failure, drain work BEFORE releasing any referenced buffers.
    defer ctx.finish() catch {};

    poison(q, query_count, original_poison);
    poison(keys, cache_count, original_poison);
    poison(values, cache_count, original_poison);
    writeSink(sink, p, case.dominant_sink);
    @memset(sink.structs(u32)[sink_words..], guard_bits);
    const sink_snapshot = try std.testing.allocator.dupe(u32, sink.structs(u32));
    defer std.testing.allocator.free(sink_snapshot);
    for (q.floats()[0..query_count], 0..) |*v, i|
        v.* = @as(f32, @floatFromInt((i * 19 + i / dim * 7) % 257)) / 128 - 1;
    // Unused linear capacity stays poisoned: all prefetch/tail loads are bounded by
    // the logical prefix, not the allocation. Different heads and dimensions
    // expose incorrect sharing/head mapping even though K/V are shared in GQA.
    for (0..case.tokens) |t| {
        for (0..kv_heads) |head| {
            for (0..dim) |d| {
                const index = (t * kv_heads + head) * dim + d;
                keys.floats()[index] = @as(f32, @floatFromInt((t * 17 + t / 31 * 13 + head * 43 + d * 7) % 257)) / 128 - 1;
                values.floats()[index] = @as(f32, @floatFromInt((t * 29 + t / 17 * 11 + head * 19 + d * 37) % 257)) / 256 - 0.25;
            }
        }
    }

    for (blocks_list) |blocks| {
        p.k = blocks; // Same stripe count/partition, never retile for a candidate.
        const state_count = @as(usize, heads) * blocks * 130;
        const original_states = try ctx.alloc(state_count + guard_words);
        defer original_states.deinit();
        const candidate_states = try ctx.alloc(state_count + guard_words);
        defer candidate_states.deinit();
        defer ctx.finish() catch {};
        poison(original_states, state_count, original_poison);
        poison(original_out, query_count, original_poison);
        const groups = @as(usize, kv_heads) * blocks;
        try ctx.dispatch("attention_mlx_split_gqa2", &.{ q, keys, values, sink, original_states }, p, groups * group_size, group_size);
        try ctx.dispatch("attention_mlx_merge", &.{ original_states, original_out }, p, @as(usize, heads) * 1024, 1024);
        try ctx.finish();
        for (variants) |variant| {
            const kernel = variant.kernel;
            poison(candidate_states, state_count, candidate_poison);
            poison(candidate_out, query_count, candidate_poison);
            try ctx.dispatch(kernel, &.{ q, keys, values, sink, candidate_states }, p, groups * variant.group, variant.group);
            // Unchanged original merge, encoded after its producer in the same
            // submission, using the exact same P and head-major state layout.
            try ctx.dispatch("attention_mlx_merge", &.{ candidate_states, candidate_out }, p, @as(usize, heads) * 1024, 1024);
            try ctx.finish();
            try expectBitwise(original_states, candidate_states, state_count, "producer states", kernel, p);
            try expectBitwise(original_out, candidate_out, query_count, "merged output", kernel, p);
            try expectGuard(q, query_count);
            try expectGuard(keys, cache_count);
            try expectGuard(values, cache_count);
            try expectGuard(sink, sink_words);
            try std.testing.expectEqualSlices(u32, sink_snapshot, sink.structs(u32));
            const active_count = @as(usize, case.tokens) * kv_heads * dim;
            for (keys.structs(u32)[active_count..cache_count]) |bits|
                try std.testing.expectEqual(original_poison, bits);
            for (values.structs(u32)[active_count..cache_count]) |bits|
                try std.testing.expectEqual(original_poison, bits);
        }
    }
}

test "GQA2/GQA4 TK1/TK2 match every TK4 scratch/output bit for N513/N16385/N32769 and B128/B512/B1024" {
    // Preserve production matched math settings for baseline and candidates.
    // The context is owned here and always closed; no server is started.
    const ctx = try metal.Context.initMatched(true);
    defer ctx.deinit();
    for ([_]Case{
        .{ .tokens = 1, .dominant_sink = true }, // Empty stripes; one sink/head.
        .{ .tokens = 513 }, // B1024 empty stripes, B512 single-token tail.
        .{ .tokens = 1024, .extra_capacity = 0 }, // Exact chunk/stripe boundary.
        .{ .tokens = 16385, .dtype = 1, .sink_offset = 6, .dominant_sink = true },
        .{ .tokens = 32769, .dtype = 2, .sink_offset = 10 }, // Long stripes + tail.
    }) |case| try runCase(ctx, case);
}

const Rejection = enum {
    dim,
    zero_heads,
    zero_kv_heads,
    nonintegral_gqa,
    other_gqa,
    sliding,
    zero_capacity,
    nonresident,
    too_long,
    position_overflow,
    unsupported_blocks,
    unsupported_dtype,
    unaligned_sink,
    wrong_group,
};

test "GQA2/GQA4 TK1/TK2 unsupported parameters/groups uniformly leave scratch untouched" {
    const ctx = try metal.Context.initMatched(true);
    defer ctx.deinit();
    const query_count: usize = heads * dim;
    const cache_count: usize = kv_heads * dim;
    const state_count: usize = heads * 128 * 130;
    const q = try ctx.alloc(query_count + guard_words);
    defer q.deinit();
    const keys = try ctx.alloc(cache_count + guard_words);
    defer keys.deinit();
    const values = try ctx.alloc(cache_count + guard_words);
    defer values.deinit();
    const sink = try ctx.alloc(heads + guard_words);
    defer sink.deinit();
    const states = try ctx.alloc(state_count + guard_words);
    defer states.deinit();
    defer ctx.finish() catch {};
    poison(q, query_count, original_poison);
    poison(keys, cache_count, original_poison);
    poison(values, cache_count, original_poison);
    poison(sink, heads, original_poison);
    for (variants) |variant| {
        const kernel = variant.kernel;
        for (std.enums.values(Rejection)) |reason| {
            var p: metal.Params = .{ .heads = heads, .kv_heads = kv_heads, .dim = dim, .capacity = 1, .k = 128 };
            var group = variant.group;
            switch (reason) {
                .dim => p.dim = 64,
                .zero_heads => p.heads = 0,
                .zero_kv_heads => p.kv_heads = 0,
                .nonintegral_gqa => p.kv_heads = 3,
                .other_gqa => p.heads = kv_heads * 4,
                .sliding => p.window = 1,
                .zero_capacity => p.capacity = 0,
                .nonresident => p.pos = 1,
                .too_long => {
                    p.pos = 262144;
                    p.capacity = 262145;
                },
                .position_overflow => p.pos = std.math.maxInt(u32),
                .unsupported_blocks => p.k = 64,
                .unsupported_dtype => p.dtype = 3,
                .unaligned_sink => p.offset = 2,
                .wrong_group => group = variant.group / 2,
            }
            poison(states, state_count, candidate_poison);
            // No merge: rejection is intentionally NOT a fallback. The host
            // must route to a supported producer or reject without merging.
            try ctx.dispatch(kernel, &.{ q, keys, values, sink, states }, p, @as(usize, kv_heads) * 128 * group, group);
            try ctx.finish();
            for (states.structs(u32)[0..state_count], 0..) |bits, i| {
                if (bits != candidate_poison)
                    std.debug.print("GQA stream {s} rejection {s} wrote scratch word {d}: 0x{x}\n", .{ kernel, @tagName(reason), i, bits });
                try std.testing.expectEqual(candidate_poison, bits);
            }
            try expectGuard(states, state_count);
            try expectGuard(q, query_count);
            try expectGuard(keys, cache_count);
            try expectGuard(values, cache_count);
            try expectGuard(sink, heads);
        }
    }
}
