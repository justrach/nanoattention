const std = @import("std");
const metal = @import("metal.zig");

const Case = struct {
    dim: u32,
    pos: u32,
    capacity: u32,
    window: u32 = 0,
    heads: u32 = 4,
    kv_heads: u32 = 2,
    dtype: u32 = 0,
    scale: f32,
};

fn keyAt(t: usize, head: usize, d: usize) f32 {
    return @as(f32, @floatFromInt((t * 7 + head * 11 + d * 3) % 29)) / 19 - 0.7;
}

fn valueAt(t: usize, head: usize, d: usize) f32 {
    // Absolute position changes values even after a cache slot is overwritten.
    return @as(f32, @floatFromInt(t)) / 33 +
        @as(f32, @floatFromInt((head * 13 + d * 7 + t * 3) % 23)) / 17 - 0.8;
}

fn writeSink(buffer: metal.Buffer, p: metal.Params, head: usize) f32 {
    const value: f32 = @as(f32, @floatFromInt(head)) * 0.7 - 0.9;
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
            const bf16: u16 = @truncate(bits >> 16);
            buffer.structs(u16)[offset / 2 + head] = bf16;
            return @bitCast(@as(u32, bf16) << 16);
        },
        else => unreachable,
    }
}

// Independent two-pass, explicitly materialized FP64 softmax, not a copy of
// either online kernel. Keys/values come from absolute fixture positions, not
// cache slots. The learned sink is a logit with zero value, included only once.
fn reference(p: metal.Params, q: []const f32, sinks: []const f32, out: []f32) void {
    const end: usize = @as(usize, p.pos) + 1;
    const begin = if (p.window == 0) 0 else end -| p.window;
    std.debug.assert(end - begin <= 64);
    for (0..p.heads) |head| {
        const kv_head = head / (p.heads / p.kv_heads);
        var scores: [64]f64 = undefined;
        var maximum: f64 = sinks[head];
        for (begin..end, 0..) |t, j| {
            var dot: f64 = 0;
            for (0..p.dim) |d|
                dot += @as(f64, q[head * p.dim + d]) * @as(f64, keyAt(t, kv_head, d));
            scores[j] = dot * @as(f64, p.scale);
            maximum = @max(maximum, scores[j]);
        }
        var denominator = @exp(@as(f64, sinks[head]) - maximum);
        for (scores[0 .. end - begin]) |*score| {
            score.* = @exp(score.* - maximum);
            denominator += score.*;
        }
        for (0..p.dim) |d| {
            var numerator: f64 = 0;
            for (begin..end, 0..) |t, j|
                numerator += scores[j] * @as(f64, valueAt(t, kv_head, d));
            out[head * p.dim + d] = @floatCast(numerator / denominator);
        }
    }
}

test "parallel attention matches explicit CPU softmax and existing attention tolerance" {
    const ctx = try metal.Context.init();
    defer ctx.deinit();
    const cases = [_]Case{
        // Seven idle SIMD groups: only sg0 has a token and the learned sink.
        .{ .dim = 4, .pos = 0, .capacity = 1, .heads = 2, .kv_heads = 1, .scale = 1.7 },
        .{ .dim = 4, .pos = 7, .capacity = 8, .scale = 1.7 },
        .{ .dim = 128, .pos = 31, .capacity = 32, .scale = 0.13 },
        .{ .dim = 256, .pos = 63, .capacity = 64, .scale = 0.037 },
        // Nonzero starts, idle partitions, non-power-of-two ring wrap, dtype
        // decoding at a byte offset, and windows including the current token.
        .{ .dim = 128, .pos = 7, .capacity = 5, .window = 3, .dtype = 1, .scale = 0.23 },
        .{ .dim = 256, .pos = 63, .capacity = 17, .window = 13, .dtype = 2, .scale = 0.071 },
        .{ .dim = 4, .pos = 31, .capacity = 3, .window = 1, .dtype = 2, .scale = 0.63 },
        .{ .dim = 128, .pos = 63, .capacity = 8, .window = 8, .heads = 2, .kv_heads = 2, .scale = 0.11 },
    };
    for (cases) |case| {
        const p: metal.Params = .{
            .offset = 8,
            .dtype = case.dtype,
            .pos = case.pos,
            .capacity = case.capacity,
            .heads = case.heads,
            .kv_heads = case.kv_heads,
            .dim = case.dim,
            .window = case.window,
            .scale = case.scale,
        };
        const q = try ctx.alloc(p.heads * p.dim);
        defer q.deinit();
        const keys = try ctx.alloc(p.capacity * p.kv_heads * p.dim);
        defer keys.deinit();
        const values = try ctx.alloc(p.capacity * p.kv_heads * p.dim);
        defer values.deinit();
        const sink = try ctx.alloc(2 + p.heads);
        defer sink.deinit();
        const scalar = try ctx.alloc(p.heads * p.dim);
        defer scalar.deinit();
        const parallel = try ctx.alloc(p.heads * p.dim);
        defer parallel.deinit();
        const expected = try std.testing.allocator.alloc(f32, p.heads * p.dim);
        defer std.testing.allocator.free(expected);
        var sinks: [4]f32 = undefined;
        @memset(sink.structs(u8), 0x7f); // Wrong byte-offset reads cannot look like valid sinks.
        for (0..p.heads) |head| sinks[head] = writeSink(sink, p, head);
        for (q.floats(), 0..) |*v, i|
            v.* = @as(f32, @floatFromInt((i * 5 + i / p.dim * 3) % 23)) / 17 - 0.6;
        @memset(keys.floats(), std.math.nan(f32));
        @memset(values.floats(), std.math.nan(f32));
        for (0..@as(usize, p.pos) + 1) |t| {
            for (0..p.kv_heads) |head| {
                for (0..p.dim) |d| {
                    const index = (t % p.capacity) * p.kv_heads * p.dim + head * p.dim + d;
                    keys.floats()[index] = keyAt(t, head, d);
                    values.floats()[index] = valueAt(t, head, d);
                }
            }
        }
        @memset(scalar.floats(), std.math.nan(f32));
        @memset(parallel.floats(), std.math.nan(f32));
        reference(p, q.floats(), sinks[0..p.heads], expected);
        try ctx.dispatch("attention", &.{ q, keys, values, sink, scalar }, p, p.heads * 32, 32);
        try ctx.dispatch("attention_parallel", &.{ q, keys, values, sink, parallel }, p, p.heads * 256, 256);
        try ctx.finish();
        for (expected, scalar.floats(), parallel.floats()) |ref, old, actual| {
            // Unchanged tolerance from the existing attention test, not a
            // relaxed threshold to accommodate reassociation.
            try std.testing.expect(std.math.isFinite(actual));
            try std.testing.expectApproxEqAbs(ref, old, 1e-5);
            try std.testing.expectApproxEqAbs(ref, actual, 1e-5);
            try std.testing.expectApproxEqAbs(old, actual, 1e-5);
        }
    }
}
