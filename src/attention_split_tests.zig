const std = @import("std");
const metal = @import("metal.zig");

const tolerance: f32 = 1e-5; // Keep the existing attention absolute tolerance.
const guard_words = 16;
const guard_bits: u32 = 0x4b123456;

const Case = struct {
    tokens: u32,
    chunk: u32,
    dim: u32,
    pos: ?u32 = null,
    capacity: u32 = 0,
    window: u32 = 0,
    heads: u32 = 4,
    kv_heads: u32 = 2,
    dtype: u32 = 0,
    sink_offset: u64 = 12,
    dominant_sink: bool = false,
    scale: f32 = 0.13,
};

// All Q/K/V fixture values are in [-1, 1], with exactly representable binary
// fractions. Absolute positions, not physical ring slots, determine K and V.
// No growing value magnitude can hide behind a relaxed absolute tolerance.
fn keyAt(t: usize, head: usize, d: usize) f32 {
    return @as(f32, @floatFromInt((t * 17 + t / 31 * 13 + head * 43 + d * 7) % 257)) / 128 - 1;
}

fn valueAt(t: usize, head: usize, d: usize) f32 {
    // A nonzero mean keeps duplicate/missing sink errors visible at 1e-5 even
    // for long contexts; values still stay bounded in [-0.25, 0.75].
    return @as(f32, @floatFromInt((t * 29 + t / 17 * 11 + head * 19 + d * 37) % 257)) / 256 - 0.25;
}

// Return the decoded value so the reference uses the same representable sink,
// but never reads the producer's output or implements its online recurrence.
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

// Independent, explicitly materialized two-pass FP64 softmax. The scratch is
// dynamically sized to the actual context (including 8192), not the 64-token
// bound of attention_parallel_tests. The sink is one extra logit with zero V.
fn reference(p: metal.Params, q: []const f32, sinks: []const f32, out: []f32) !void {
    const end: usize = @as(usize, p.pos) + 1;
    const begin = if (p.window == 0) 0 else end -| p.window;
    const probabilities = try std.testing.allocator.alloc(f64, end - begin);
    defer std.testing.allocator.free(probabilities);
    for (0..p.heads) |head| {
        const kv_head = head / (p.heads / p.kv_heads);
        var maximum: f64 = sinks[head];
        for (begin..end, 0..) |t, j| {
            var dot: f64 = 0;
            for (0..p.dim) |d|
                dot += @as(f64, q[head * p.dim + d]) * @as(f64, keyAt(t, kv_head, d));
            probabilities[j] = dot * @as(f64, p.scale);
            maximum = @max(maximum, probabilities[j]);
        }
        var denominator = @exp(@as(f64, sinks[head]) - maximum);
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
    for (buffer.structs(u32)[payload..]) |bits|
        try std.testing.expectEqual(guard_bits, bits);
}

fn runCase(ctx: metal.Context, case: Case) !void {
    const pos = case.pos orelse (case.tokens - 1);
    const p: metal.Params = .{
        .offset = case.sink_offset,
        .dtype = case.dtype,
        .pos = pos,
        .capacity = if (case.capacity == 0) case.tokens + 7 else case.capacity,
        .heads = case.heads,
        .kv_heads = case.kv_heads,
        .dim = case.dim,
        .window = case.window,
        .scale = case.scale,
        .k = (case.tokens + case.chunk - 1) / case.chunk,
        .bits = case.chunk,
    };
    const end: usize = @as(usize, p.pos) + 1;
    const begin = if (p.window == 0) 0 else end -| p.window;
    try std.testing.expectEqual(@as(usize, case.tokens), end - begin);
    try std.testing.expect(case.tokens <= p.capacity);
    try std.testing.expect(p.heads % p.kv_heads == 0);
    const count: usize = @as(usize, p.heads) * p.dim;
    const state_count: usize = @as(usize, p.heads) * p.k * (p.dim + 2);
    const q = try ctx.alloc(count);
    defer q.deinit();
    const keys = try ctx.alloc(@as(usize, p.capacity) * p.kv_heads * p.dim);
    defer keys.deinit();
    const values = try ctx.alloc(@as(usize, p.capacity) * p.kv_heads * p.dim);
    defer values.deinit();
    const sink_element_bytes: usize = if (p.dtype == 0) 4 else 2;
    const sink_bytes = @as(usize, @intCast(p.offset)) + p.heads * sink_element_bytes;
    const sink = try ctx.alloc((sink_bytes + 3) / 4 + guard_words);
    defer sink.deinit();
    const states = try ctx.alloc(state_count + guard_words);
    defer states.deinit();
    const out = try ctx.alloc(count + guard_words);
    defer out.deinit();
    // If a dispatch fails, drain any earlier encoded work before freeing buffers.
    defer ctx.finish() catch {};
    const expected = try std.testing.allocator.alloc(f32, count);
    defer std.testing.allocator.free(expected);
    const sinks = try std.testing.allocator.alloc(f32, p.heads);
    defer std.testing.allocator.free(sinks);

    @memset(sink.structs(u8), 0xff); // Wrong offset/dtype reads encounter NaNs.
    for (sinks, 0..) |*s, head| s.* = writeSink(sink, p, head, case.dominant_sink);
    for (q.floats(), 0..) |*v, i|
        v.* = @as(f32, @floatFromInt((i * 19 + i / p.dim * 7) % 257)) / 128 - 1;
    @memset(keys.floats(), std.math.nan(f32));
    @memset(values.floats(), std.math.nan(f32));
    // Populate only the active absolute positions. Unused cache slots remain
    // poisoned, and the nonaligned ring case represents many prior wraps.
    for (begin..end) |t| {
        for (0..p.kv_heads) |head| {
            for (0..p.dim) |d| {
                const index = (t % p.capacity) * p.kv_heads * p.dim + head * p.dim + d;
                keys.floats()[index] = keyAt(t, head, d);
                values.floats()[index] = valueAt(t, head, d);
            }
        }
    }
    poisonAndGuard(states, state_count);
    poisonAndGuard(out, count);
    try reference(p, q.floats(), sinks, expected);

    // Exact production ABI: q0, keys1, values2, sink bytes3, states4, P8;
    // then states0, out1, P8. One producer group per (head, chunk), eight
    // SIMDs/group, followed by exactly one SIMD/group per head for the merge.
    try ctx.dispatch("attention_split", &.{ q, keys, values, sink, states }, p, @as(usize, p.heads) * p.k * 256, 256);
    try ctx.dispatch("attention_merge", &.{ states, out }, p, @as(usize, p.heads) * 32, 32);
    try ctx.finish();

    try expectGuard(states, state_count);
    try expectGuard(out, count);
    // Every unnormalized u[dim], maximum and denominator must have replaced its
    // poison at (head * chunks + chunk) * (dim + 2), including short chunks.
    for (0..p.heads) |head| {
        for (0..p.k) |chunk| {
            const base = (head * p.k + chunk) * (p.dim + 2);
            for (states.floats()[base .. base + p.dim + 2]) |v|
                try std.testing.expect(std.math.isFinite(v));
            try std.testing.expect(states.floats()[base + p.dim + 1] > 0);
            if (case.dominant_sink) {
                if (chunk == 0) {
                    try std.testing.expectApproxEqAbs(sinks[head], states.floats()[base + p.dim], tolerance);
                } else {
                    // With scale=0.01, no bounded Q/K token can reach the
                    // sink logit. A sink in any later chunk is observable.
                    try std.testing.expect(states.floats()[base + p.dim] <= @as(f32, @floatFromInt(p.dim)) * p.scale);
                }
            }
        }
    }
    for (expected, out.floats()[0..count]) |ref, actual| {
        try std.testing.expect(std.math.isFinite(actual));
        try std.testing.expectApproxEqAbs(ref, actual, tolerance);
    }
    // Production ordered split must preserve the previous attention reduction,
    // not merely stay locally close to FP64 (MoE routing amplifies tiny drift).
    const score_count = @as(usize, p.heads) * (end - begin);
    const scores = try ctx.alloc(score_count + guard_words);
    defer scores.deinit();
    const ordered = try ctx.alloc(count + guard_words);
    defer ordered.deinit();
    const legacy = try ctx.alloc(count + guard_words);
    defer legacy.deinit();
    poisonAndGuard(scores, score_count);
    poisonAndGuard(ordered, count);
    poisonAndGuard(legacy, count);
    try ctx.dispatch("attention_scores", &.{ q, keys, scores }, p, @as(usize, p.heads) * p.k * 256, 256);
    try ctx.dispatch("attention_ordered_values", &.{ scores, values, sink, ordered }, p, @as(usize, p.heads) * 256, 256);
    try ctx.dispatch("attention_parallel", &.{ q, keys, values, sink, legacy }, p, @as(usize, p.heads) * 256, 256);
    try ctx.finish();
    try expectGuard(scores, score_count);
    try expectGuard(ordered, count);
    try expectGuard(legacy, count);
    try std.testing.expectEqualSlices(f32, legacy.floats()[0..count], ordered.floats()[0..count]);
    for (expected, ordered.floats()[0..count]) |ref, actual|
        try std.testing.expectApproxEqAbs(ref, actual, tolerance);
}

test "split attention and merge match materialized FP64 softmax through 8k contexts" {
    const ctx = try metal.Context.init();
    defer ctx.deinit();
    // Boundary-focused cases rather than a Cartesian product. Neighboring
    // lengths exercise full/partial chunks, idle SIMDs, and dimension tails.
    const cases = [_]Case{
        .{ .tokens = 1, .chunk = 128, .dim = 4, .heads = 2, .kv_heads = 1, .scale = 1.7 },
        .{ .tokens = 127, .chunk = 128, .dim = 128 },
        .{ .tokens = 128, .chunk = 128, .dim = 130 },
        .{ .tokens = 129, .chunk = 128, .dim = 256 },
        .{ .tokens = 255, .chunk = 256, .dim = 4, .scale = 1.7 },
        .{ .tokens = 256, .chunk = 256, .dim = 128 },
        .{ .tokens = 257, .chunk = 256, .dim = 130 },
        .{ .tokens = 511, .chunk = 512, .dim = 256 },
        .{ .tokens = 512, .chunk = 512, .dim = 130 },
        .{ .tokens = 513, .chunk = 512, .dim = 128 },
        .{ .tokens = 1024, .chunk = 1024, .dim = 256 },
        .{ .tokens = 4096, .chunk = 256, .dim = 128, .dtype = 1, .sink_offset = 6 },
        // 64 chunks exercises the merge's lane-strided maximum loop, not just
        // the common <=32-chunk path. All chunk sizes also see a long context.
        .{ .tokens = 8192, .chunk = 128, .dim = 130, .dtype = 2, .sink_offset = 10 },
        .{ .tokens = 8192, .chunk = 256, .dim = 4, .scale = 1.7 },
        .{ .tokens = 8192, .chunk = 512, .dim = 128, .dtype = 1, .sink_offset = 6 },
        .{ .tokens = 8192, .chunk = 1024, .dim = 256 },
        // start = 8391 - 513 = 7878: neither a chunk boundary nor a ring
        // boundary. The window includes the current token at absolute 8390.
        .{ .tokens = 513, .chunk = 128, .dim = 130, .pos = 8390, .capacity = 513, .window = 513, .dtype = 1, .sink_offset = 6 },
        .{ .tokens = 513, .chunk = 256, .dim = 256, .pos = 8390, .capacity = 513, .window = 513, .dtype = 2, .sink_offset = 10 },
        .{ .tokens = 513, .chunk = 512, .dim = 128, .pos = 8390, .capacity = 513, .window = 513 },
        .{ .tokens = 513, .chunk = 1024, .dim = 4, .pos = 8390, .capacity = 513, .window = 513, .heads = 2, .kv_heads = 2, .scale = 1.7 },
        // A dominant sink exposes accidentally normalized chunk states or a
        // sink included per SIMD/per chunk instead of only chunk0/sg0.
        .{ .tokens = 129, .chunk = 128, .dim = 4, .dominant_sink = true, .scale = 0.01 },
        .{ .tokens = 257, .chunk = 256, .dim = 130, .dtype = 1, .sink_offset = 6, .dominant_sink = true, .scale = 0.01 },
        .{ .tokens = 513, .chunk = 512, .dim = 128, .dtype = 2, .sink_offset = 10, .dominant_sink = true, .scale = 0.01 },
        .{ .tokens = 4096, .chunk = 1024, .dim = 256, .dominant_sink = true, .scale = 0.01 },
    };
    for (cases, 0..) |case, i| {
        runCase(ctx, case) catch |err| {
            std.debug.print("split attention case {d}: tokens={d}, chunk={d}, dim={d}, dtype={d}, pos={d}, window={d}\n", .{
                i, case.tokens, case.chunk, case.dim, case.dtype, case.pos orelse (case.tokens - 1), case.window,
            });
            return err;
        };
    }
}
