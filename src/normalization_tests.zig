const std = @import("std");
const metal = @import("metal.zig");

const tolerance: f32 = 1e-5;
const sentinel: f32 = -1234.5;

// Store exactly representable weights, including nonzero byte offsets, and return
// the values actually represented by each checkpoint dtype to the CPU reference.
fn normWeights(buffer: metal.Buffer, offset: usize, dtype: u32, reference: []f32) void {
    @memset(buffer.structs(u8), 0xa5);
    for (reference, 0..) |*value, i| {
        value.* = @as(f32, @floatFromInt(i % 13 + 1)) / 16;
        switch (dtype) {
            0 => buffer.floats()[offset / 4 + i] = value.*,
            1 => buffer.structs(u16)[offset / 2 + i] = @bitCast(@as(f16, @floatCast(value.*))),
            2 => buffer.structs(u16)[offset / 2 + i] = @intCast(@as(u32, @bitCast(value.*)) >> 16),
            else => unreachable,
        }
    }
}

fn fillInput(values: []f32, seed: usize) void {
    for (values, 0..) |*value, i| value.* = @as(f32, @floatFromInt((i * 7 + seed) % 31)) / 16 - 0.75;
}

fn cpuNorm(input: []const f32, weights: []const f32, eps: f32, output: []f32) void {
    // A higher-precision reference avoids conflating CPU serial-sum error with
    // either GPU reduction tree; shader arithmetic remains entirely FP32.
    var square_sum: f64 = 0;
    for (input) |value| square_sum += @as(f64, value) * value;
    const inv: f32 = @floatCast(1 / @sqrt(square_sum / @as(f64, @floatFromInt(input.len)) + eps));
    for (input, weights, output) |value, weight, *out| out.* = value * inv * weight;
}

fn expectClose(expected: []const f32, actual: []const f32) !void {
    try std.testing.expectEqual(expected.len, actual.len);
    for (expected, actual) |ref, value| try std.testing.expectApproxEqAbs(ref, value, tolerance);
}

test "parallel RMSNorm covers 2048, odd tail 513, 8192, checkpoint dtypes and in-place output" {
    const ctx = try metal.Context.init();
    defer ctx.deinit();
    for ([_]usize{ 2048, 513, 8192 }) |n| {
        const w = try ctx.alloc(n + 4);
        defer w.deinit();
        const x = try ctx.alloc(n + 4);
        defer x.deinit();
        const out = try ctx.alloc(n + 4);
        defer out.deinit();
        const weights = try std.testing.allocator.alloc(f32, n);
        defer std.testing.allocator.free(weights);
        const input = try std.testing.allocator.alloc(f32, n);
        defer std.testing.allocator.free(input);
        const expected = try std.testing.allocator.alloc(f32, n);
        defer std.testing.allocator.free(expected);
        fillInput(input, 3);
        for (0..3) |dtype| {
            normWeights(w, 12, @intCast(dtype), weights);
            cpuNorm(input, weights, 1e-6, expected);
            for ([_]bool{ false, true }) |in_place| {
                @memset(x.floats(), sentinel);
                @memset(out.floats(), sentinel);
                @memcpy(x.floats()[0..n], input);
                const destination = if (in_place) x else out;
                try ctx.dispatch("rmsnorm_parallel", &.{ w, x, destination }, .{ .n = @intCast(n), .m = 1, .offset = 12, .dtype = @intCast(dtype) }, 256, 256);
                try ctx.finish();
                try expectClose(expected, destination.floats()[0..n]);
                for (destination.floats()[n..]) |value| try std.testing.expectEqual(sentinel, value);
                for (x.floats()[n..]) |value| try std.testing.expectEqual(sentinel, value);
                if (!in_place) try std.testing.expectEqualSlices(f32, input, x.floats()[0..n]);
            }
        }
    }
}

test "fused residual RMSNorm preserves the in-place residual and matches CPU with odd tails" {
    const ctx = try metal.Context.init();
    defer ctx.deinit();
    for ([_]usize{ 2048, 513 }) |n| {
        const w = try ctx.alloc(n + 4);
        defer w.deinit();
        const x = try ctx.alloc(n + 4);
        defer x.deinit();
        const projection = try ctx.alloc(n + 4);
        defer projection.deinit();
        const out = try ctx.alloc(n + 4);
        defer out.deinit();
        const weights = try std.testing.allocator.alloc(f32, n);
        defer std.testing.allocator.free(weights);
        const residual = try std.testing.allocator.alloc(f32, n);
        defer std.testing.allocator.free(residual);
        const expected = try std.testing.allocator.alloc(f32, n);
        defer std.testing.allocator.free(expected);
        @memset(projection.floats(), sentinel);
        fillInput(projection.floats()[0..n], 19);
        for (0..3) |dtype| {
            @memset(x.floats(), sentinel);
            @memset(out.floats(), sentinel);
            fillInput(x.floats()[0..n], 5);
            for (residual, x.floats()[0..n], projection.floats()[0..n]) |*ref, a, b| ref.* = a + b;
            normWeights(w, 8, @intCast(dtype), weights);
            cpuNorm(residual, weights, 3e-5, expected);
            try ctx.dispatch("residual_norm", &.{ w, x, projection, out }, .{ .n = @intCast(n), .m = 1, .offset = 8, .dtype = @intCast(dtype), .eps = 3e-5 }, 256, 256);
            try ctx.finish();
            try std.testing.expectEqualSlices(f32, residual, x.floats()[0..n]);
            try expectClose(expected, out.floats()[0..n]);
            for (x.floats()[n..]) |value| try std.testing.expectEqual(sentinel, value);
            for (out.floats()[n..]) |value| try std.testing.expectEqual(sentinel, value);
            // Projection is read-only, including its tail.
            for (projection.floats()[0..n], 0..) |value, i| {
                const ref = @as(f32, @floatFromInt((i * 7 + 19) % 31)) / 16 - 0.75;
                try std.testing.expectEqual(ref, value);
            }
            for (projection.floats()[n..]) |value| try std.testing.expectEqual(sentinel, value);
        }
    }
}

fn cpuHeads(input: []const f32, weights: []const f32, pos: u32, rope: bool, output: []f32) void {
    const dim = weights.len;
    const half = dim / 2;
    for (0..input.len / dim) |head| {
        const row = output[head * dim ..][0..dim];
        cpuNorm(input[head * dim ..][0..dim], weights, 1e-6, row);
        if (rope) for (0..half) |j| {
            const angle = @as(f32, @floatFromInt(pos)) * std.math.pow(f32, 10000, -2 * @as(f32, @floatFromInt(j)) / @as(f32, @floatFromInt(dim)));
            const a = row[j];
            const b = row[j + half];
            row[j] = a * @cos(angle) - b * @sin(angle);
            row[j + half] = b * @cos(angle) + a * @sin(angle);
        };
    }
}

test "fused QK norm optional split-half RoPE and KV cache match CPU across wrap and head tails" {
    const ctx = try metal.Context.init();
    defer ctx.deinit();
    const heads = 3;
    const kv_heads = 2;
    const capacity = 3;
    // 66 exercises partial SIMD pairs, 128 the model head size, 256 the upper bound.
    for ([_]usize{ 2, 66, 128, 256 }) |dim| {
        const qn = heads * dim;
        const kn = kv_heads * dim;
        const q = try ctx.alloc(qn + 4);
        defer q.deinit();
        const k = try ctx.alloc(kn + 4);
        defer k.deinit();
        const v = try ctx.alloc(kn + 4);
        defer v.deinit();
        const qw = try ctx.alloc(dim + 4);
        defer qw.deinit();
        const kw = try ctx.alloc(dim + 4);
        defer kw.deinit();
        const keys = try ctx.alloc(capacity * kn + 4);
        defer keys.deinit();
        const values = try ctx.alloc(capacity * kn + 4);
        defer values.deinit();
        var qweights: [256]f32 = undefined;
        var kweights: [256]f32 = undefined;
        var qref: [heads * 256]f32 = undefined;
        var kref: [kv_heads * 256]f32 = undefined;
        var keyref: [capacity * kv_heads * 256 + 4]f32 = undefined;
        var valueref: [capacity * kv_heads * 256 + 4]f32 = undefined;
        for (0..3) |qdtype| {
            const kdtype = (qdtype + 1) % 3;
            normWeights(qw, 12, @intCast(qdtype), qweights[0..dim]);
            normWeights(kw, 8, @intCast(kdtype), kweights[0..dim]);
            for ([_]bool{ false, true }) |rope| {
                @memset(keys.floats(), sentinel);
                @memset(values.floats(), sentinel);
                @memset(keyref[0 .. capacity * kn + 4], sentinel);
                @memset(valueref[0 .. capacity * kn + 4], sentinel);
                // Absolute position 4 overwrites slot 1; untouched slots stay exact.
                for ([_]u32{ 0, 1, 4 }) |pos| {
                    @memset(q.floats(), sentinel);
                    @memset(k.floats(), sentinel);
                    @memset(v.floats(), sentinel);
                    fillInput(q.floats()[0..qn], 3 + pos);
                    fillInput(k.floats()[0..kn], 9 + pos);
                    fillInput(v.floats()[0..kn], 17 + pos);
                    cpuHeads(q.floats()[0..qn], qweights[0..dim], pos, rope, qref[0..qn]);
                    cpuHeads(k.floats()[0..kn], kweights[0..dim], pos, rope, kref[0..kn]);
                    const slot = @as(usize, pos % capacity) * kn;
                    @memcpy(keyref[slot..][0..kn], kref[0..kn]);
                    @memcpy(valueref[slot..][0..kn], v.floats()[0..kn]);
                    const p: metal.Params = .{
                        .offset = 12,
                        .scale_offset = 8,
                        .dtype = @intCast(qdtype),
                        .scale_dtype = @intCast(kdtype),
                        .heads = heads,
                        .kv_heads = kv_heads,
                        .dim = @intCast(dim),
                        .pos = pos,
                        .capacity = capacity,
                        .bits = @intFromBool(rope),
                    };
                    // Deliberately pad to eight SIMD heads to exercise head guards.
                    const group: usize = if (qdtype == 1) 128 else 32;
                    try ctx.dispatch("qk_rope_cache_fused", &.{ q, k, v, qw, kw, keys, values }, p, 8 * 32, group);
                    try ctx.finish();
                    try expectClose(qref[0..qn], q.floats()[0..qn]);
                    try expectClose(kref[0..kn], k.floats()[0..kn]);
                    try expectClose(keyref[0 .. capacity * kn + 4], keys.floats());
                    try std.testing.expectEqualSlices(f32, valueref[0 .. capacity * kn + 4], values.floats());
                    for (q.floats()[qn..]) |value| try std.testing.expectEqual(sentinel, value);
                    for (k.floats()[kn..]) |value| try std.testing.expectEqual(sentinel, value);
                    for (v.floats()[0..kn], 0..) |value, i| {
                        const ref = @as(f32, @floatFromInt((i * 7 + 17 + pos) % 31)) / 16 - 0.75;
                        try std.testing.expectEqual(ref, value);
                    }
                    for (v.floats()[kn..]) |value| try std.testing.expectEqual(sentinel, value);
                }
            }
        }
    }
}
