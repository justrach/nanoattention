const std = @import("std");
const metal = @import("metal.zig");

// Context.init() uses the production concatenated Metal source. The caller must
// embed attention_mlx.metal after kernels.metal; do not compile a test-only ABI.
const tolerance: f32 = 1e-5;
const guard_words: usize = 16;
const guard_bits: u32 = 0x4b123456;
const dim: u32 = 128;
const heads: u32 = 16;
const kv_heads: u32 = 2;
const gqa: usize = heads / kv_heads; // Eight query heads per KV head, two KV groups.

const Case = struct {
    tokens: u32,
    pos: ?u32 = null,
    capacity: u32 = 0,
    window: u32 = 0,
    dtype: u32 = 0,
    sink_offset: u64 = 12,
    dominant_sink: bool = false,
    scale: f32 = 0.13,
};

// Adapted from attention_split_tests.zig: bounded, exactly representable binary
// fractions, with absolute positions (not ring slots) determining K and V.
fn keyAt(t: usize, head: usize, d: usize) f32 {
    return @as(f32, @floatFromInt((t * 17 + t / 31 * 13 + head * 43 + d * 7) % 257)) / 128 - 1;
}

fn valueAt(t: usize, head: usize, d: usize) f32 {
    // A nonzero mean makes duplicate/missing learned sinks observable even at
    // long contexts; every value remains in [-0.25, 0.75].
    return @as(f32, @floatFromInt((t * 29 + t / 17 * 11 + head * 19 + d * 37) % 257)) / 256 - 0.25;
}

fn writeSink(buffer: metal.Buffer, p: metal.Params, head: usize, dominant: bool) f32 {
    const value: f32 = if (dominant)
        10 + @as(f32, @floatFromInt(head)) * 0.25
    else
        -0.7 + @as(f32, @floatFromInt(head)) * 0.4;
    const offset: usize = @intCast(p.offset);
    switch (p.dtype) {
        0 => {
            buffer.structs(f32)[offset / 4 + head] = value;
            return value;
        },
        1 => {
            const rounded: f16 = @floatCast(value);
            buffer.structs(u16)[offset / 2 + head] = @bitCast(rounded);
            return @floatCast(rounded);
        },
        2 => {
            const bits: u32 = @bitCast(value);
            const rounded: u16 = @truncate(bits >> 16);
            buffer.structs(u16)[offset / 2 + head] = rounded;
            return @bitCast(@as(u32, rounded) << 16);
        },
        else => unreachable,
    }
}

// Independent materialized FP64 softmax: no online recurrence, no producer
// scratch, and no reads from the cache under test. Include the decoded learned
// sink as an explicit extra logit with V=0, exactly once for every query head.
fn reference(p: metal.Params, q: []const f32, sinks: []const f32, out: []f32) !void {
    const end: usize = @as(usize, p.pos) + 1;
    const begin = if (p.window == 0) 0 else end -| p.window;
    const n = end - begin;
    const probabilities = try std.testing.allocator.alloc(f64, n + 1);
    defer std.testing.allocator.free(probabilities);
    for (0..p.heads) |head| {
        const kv_head = head / (p.heads / p.kv_heads);
        probabilities[n] = sinks[head];
        var maximum: f64 = probabilities[n];
        for (begin..end, 0..) |t, j| {
            var dot: f64 = 0;
            for (0..p.dim) |d|
                dot += @as(f64, q[head * p.dim + d]) * @as(f64, keyAt(t, kv_head, d));
            probabilities[j] = dot * @as(f64, p.scale);
            maximum = @max(maximum, probabilities[j]);
        }
        var denominator: f64 = 0;
        for (probabilities) |*probability| {
            probability.* = @exp(probability.* - maximum);
            denominator += probability.*;
        }
        for (0..p.dim) |d| {
            var numerator: f64 = 0;
            for (begin..end, 0..) |t, j|
                numerator += probabilities[j] * @as(f64, valueAt(t, kv_head, d));
            out[head * p.dim + d] = @floatCast(numerator / denominator);
        }
    }
}

fn poisonAndGuard(buffer: metal.Buffer, payload: usize) void {
    @memset(buffer.floats()[0..payload], std.math.nan(f32));
    @memset(buffer.structs(u32)[payload..], guard_bits);
}

fn expectGuard(buffer: metal.Buffer, payload: usize) !void {
    // Enforce the exact logical output/scratch count, not just a comparison of
    // whatever prefix happened to be written. The ABI starts at buffer index 0.
    try std.testing.expectEqual(payload + guard_words, buffer.structs(u32).len);
    for (buffer.structs(u32)[payload..]) |bits|
        try std.testing.expectEqual(guard_bits, bits);
}

fn expectOutput(expected: []const f32, out: metal.Buffer) !void {
    try std.testing.expectEqual(@as(usize, heads) * dim, expected.len);
    try expectGuard(out, expected.len);
    for (expected, out.floats()[0..expected.len], 0..) |wanted, actual, i| {
        if (!std.math.isFinite(actual) or @abs(wanted - actual) > tolerance)
            std.debug.print("MLX attention output {d}/{d}: expected={d}, actual={d}\n", .{ i, expected.len, wanted, actual });
        try std.testing.expect(std.math.isFinite(actual));
        try std.testing.expectApproxEqAbs(wanted, actual, tolerance);
    }
}

fn expectStates(states: metal.Buffer, p: metal.Params, tokens: usize, sinks: []const f32, dominant: bool) !void {
    try expectGuard(states, @as(usize, p.heads) * p.k * (p.dim + 2));
    for (0..p.heads) |head| {
        for (0..p.k) |block| {
            const base = (head * p.k + block) * (p.dim + 2);
            const record = states.floats()[base .. base + p.dim + 2];
            // Empty stripes must overwrite poison too. MLX uses finite -FLT_MAX
            // and denominator zero, not a positive denominator for every stripe.
            for (record) |v| try std.testing.expect(std.math.isFinite(v));
            const maximum = record[p.dim];
            const denominator = record[p.dim + 1];
            if (block >= tokens and block != 0) {
                try std.testing.expectEqual(-std.math.floatMax(f32), maximum);
                try std.testing.expectEqual(@as(f32, 0), denominator);
                for (record[0..p.dim]) |v|
                    try std.testing.expectEqual(@as(f32, 0), v);
            } else {
                try std.testing.expect(denominator > 0);
                const stripe_tokens = if (block < tokens) 1 + (tokens - 1 - block) / p.k else 0;
                const logit_count = stripe_tokens + @as(usize, if (block == 0) 1 else 0);
                try std.testing.expect(denominator <= @as(f32, @floatFromInt(logit_count)) + tolerance);
                if (dominant) {
                    if (block == 0) {
                        try std.testing.expectApproxEqAbs(sinks[head], maximum, tolerance);
                    } else {
                        // The sink exceeds every bounded token score at 0.01
                        // scale, exposing accidental inclusion in later stripes.
                        try std.testing.expect(maximum <= @as(f32, @floatFromInt(p.dim)) * p.scale);
                    }
                }
            }
        }
    }
}

fn runSplit(ctx: metal.Context, base: metal.Params, blocks: u32, tokens: usize, q: metal.Buffer, keys: metal.Buffer, values: metal.Buffer, sink: metal.Buffer, sinks: []const f32, out: metal.Buffer, expected: []const f32, dominant: bool) !void {
    var p = base;
    p.k = blocks; // MLX B is a stripe count, NOT a contiguous token chunk size.
    const state_count = @as(usize, p.heads) * p.k * (p.dim + 2);
    const states = try ctx.alloc(state_count + guard_words);
    defer states.deinit();
    // Drain previously encoded work before releasing buffers on dispatch failure.
    defer ctx.finish() catch {};
    poisonAndGuard(states, state_count);
    poisonAndGuard(out, expected.len);
    // Production ABI: q0, keys1, values2, sink3, states4, P8; merge states0,
    // out1, P8. Two KV groups, B groups/KV, eight SIMDs (GQA8) per producer.
    // Keep producer and consumer in one submission to exercise their barrier.
    try ctx.dispatch("attention_mlx_split", &.{ q, keys, values, sink, states }, p, @as(usize, p.kv_heads) * p.k * (32 * gqa), 32 * gqa);
    try ctx.dispatch("attention_mlx_merge", &.{ states, out }, p, @as(usize, p.heads) * 1024, 1024);
    try ctx.finish();
    try expectStates(states, p, tokens, sinks, dominant);
    try expectOutput(expected, out);
}

fn runCase(ctx: metal.Context, case: Case) !void {
    const p: metal.Params = .{
        .offset = case.sink_offset,
        .dtype = case.dtype,
        .pos = case.pos orelse (case.tokens - 1),
        .capacity = if (case.capacity == 0) case.tokens + 7 else case.capacity,
        .heads = heads,
        .kv_heads = kv_heads,
        .dim = dim,
        .window = case.window,
        .scale = case.scale,
    };
    const end: usize = @as(usize, p.pos) + 1;
    const begin = if (p.window == 0) 0 else end -| p.window;
    try std.testing.expectEqual(@as(usize, case.tokens), end - begin);
    try std.testing.expect(case.tokens > 0 and case.tokens <= 32768);
    try std.testing.expect(case.tokens <= p.capacity);
    const count = @as(usize, p.heads) * p.dim;
    const cache_count = @as(usize, p.capacity) * p.kv_heads * p.dim;
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
    const out = try ctx.alloc(count + guard_words);
    defer out.deinit();
    defer ctx.finish() catch {};
    const expected = try std.testing.allocator.alloc(f32, count);
    defer std.testing.allocator.free(expected);
    var sinks: [heads]f32 = undefined;

    poisonAndGuard(q, count);
    poisonAndGuard(keys, cache_count);
    poisonAndGuard(values, cache_count);
    poisonAndGuard(out, count);
    // Wrong offset/dtype reads encounter poison. The 16-bit cases deliberately
    // start at offsets 6/10, which are not aligned to a float32 sink element.
    @memset(sink.structs(u8), 0xff);
    @memset(sink.structs(u32)[sink_words..], guard_bits);
    for (&sinks, 0..) |*s, head| s.* = writeSink(sink, p, head, case.dominant_sink);
    const sink_snapshot = try std.testing.allocator.dupe(u32, sink.structs(u32));
    defer std.testing.allocator.free(sink_snapshot);
    for (q.floats()[0..count], 0..) |*v, i|
        v.* = @as(f32, @floatFromInt((i * 19 + i / p.dim * 7) % 257)) / 128 - 1;
    // Only active absolute positions are materialized; unused ring slots and
    // cache guard words must never participate in attention, even after wraps.
    for (begin..end) |t| {
        for (0..p.kv_heads) |head| {
            for (0..p.dim) |d| {
                const index = (t % p.capacity) * p.kv_heads * p.dim + head * p.dim + d;
                keys.floats()[index] = keyAt(t, head, d);
                values.floats()[index] = valueAt(t, head, d);
            }
        }
    }
    try reference(p, q.floats()[0..count], &sinks, expected);

    try ctx.dispatch("attention_mlx_single", &.{ q, keys, values, sink, out }, p, @as(usize, p.heads) * 1024, 1024);
    try ctx.finish();
    expectOutput(expected, out) catch |err| {
        std.debug.print("MLX single-pass (1024 threads/head)\n", .{});
        return err;
    };
    if (p.window != 0 and p.capacity == p.window and p.pos + 1 > p.window) {
        // MLX returns a full rotating cache in physical-slot order. Relabeling
        // this same buffer as an unwrapped full prefix must preserve every bit,
        // not merely stay close to the permutation-invariant FP64 softmax.
        const physical = try std.testing.allocator.dupe(f32, out.floats()[0..count]);
        defer std.testing.allocator.free(physical);
        var unwrapped = p;
        unwrapped.pos = p.capacity - 1;
        try ctx.dispatch("attention_mlx_single", &.{ q, keys, values, sink, out }, unwrapped, @as(usize, p.heads) * 1024, 1024);
        try ctx.finish();
        try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(physical), std.mem.sliceAsBytes(out.floats()[0..count]));
    }
    for (@import("resident_plan.zig").mlx_attention_blocks) |blocks| {
        runSplit(ctx, p, blocks, case.tokens, q, keys, values, sink, &sinks, out, expected, case.dominant_sink) catch |err| {
            std.debug.print("MLX split B={d} (256 threads/group), merge (1024 threads/head)\n", .{blocks});
            return err;
        };
        // Check read-only buffers after each variant, not just the final one.
        try expectGuard(q, count);
        try expectGuard(keys, cache_count);
        try expectGuard(values, cache_count);
        try expectGuard(sink, sink_words);
        try std.testing.expectEqualSlices(u32, sink_snapshot, sink.structs(u32));
    }
}

fn runCases(cases: []const Case) !void {
    // Pure headless Metal: no model checkpoint, UI, server, or external process.
    const ctx = try metal.Context.init();
    defer ctx.deinit();
    for (cases, 0..) |case, i| {
        runCase(ctx, case) catch |err| {
            std.debug.print("MLX attention case {d}: tokens={d}, pos={d}, capacity={d}, window={d}, dtype={d}, sink_offset={d}, dominant_sink={}\n", .{
                i,                                                          case.tokens,        case.pos orelse (case.tokens - 1),
                if (case.capacity == 0) case.tokens + 7 else case.capacity, case.window,        case.dtype,
                case.sink_offset,                                           case.dominant_sink,
            });
            return err;
        };
    }
}

test "MLX GQA8 D128 single and B64/B128 split attention match materialized FP64 through 8192 tokens" {
    // Six bounded lengths cover idle SIMDs/stripes, partial stripes, the TK=4
    // prefetch loop and its tail, and long-context lane-strided merge reduction.
    try runCases(&.{
        .{ .tokens = 1 },
        .{ .tokens = 31, .dtype = 1, .sink_offset = 6 },
        .{ .tokens = 65, .dtype = 2, .sink_offset = 10 },
        .{ .tokens = 1024 },
        .{ .tokens = 4096, .dtype = 1, .sink_offset = 6 },
        .{ .tokens = 8192, .dtype = 2, .sink_offset = 10 },
    });
}

test "MLX attention certified stripes match FP64 at 16K and 32K" {
    try runCases(&.{
        .{ .tokens = 16384 },
        .{ .tokens = 32768, .dtype = 2, .sink_offset = 10 },
    });
}

test "MLX attention preserves nonaligned sliding rings, sink byte offsets, output counts and guards" {
    // Start=8391-window is neither a stripe nor a ring boundary. Capacity may
    // exceed the active window or be exactly full, after many prior ring wraps.
    // Dominant sinks expose replication per SIMD/stripe and normalized scratch.
    try runCases(&.{
        .{ .tokens = 1, .pos = 8390, .capacity = 17, .window = 1, .dtype = 1, .sink_offset = 6, .dominant_sink = true, .scale = 0.01 },
        .{ .tokens = 31, .pos = 8390, .capacity = 47, .window = 31, .dtype = 2, .sink_offset = 10 },
        .{ .tokens = 65, .pos = 8390, .capacity = 97, .window = 65, .dtype = 1, .sink_offset = 6, .dominant_sink = true, .scale = 0.01 },
        .{ .tokens = 513, .pos = 8390, .capacity = 513, .window = 513, .dtype = 2, .sink_offset = 10, .dominant_sink = true, .scale = 0.01 },
        // A window larger than the prefix must saturate start at zero.
        .{ .tokens = 31, .window = 65, .dtype = 1, .sink_offset = 6 },
        .{ .tokens = 4096, .pos = 16390, .capacity = 4099, .window = 4096, .dominant_sink = true, .scale = 0.01 },
    });
}
