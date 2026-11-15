const std = @import("std");
const metal = @import("metal.zig");

const tolerance: f32 = 1e-5;
const sentinel: f32 = -1234.5;
const hidden = 2048;
const dim = 128;
extern fn fm_open([*:0]const u8) ?*anyopaque;

// Isolated candidate library: no changes to production Context.init or selection.
fn context() !metal.Context {
    const source = @embedFile("kernels.metal") ++ "\n" ++ @embedFile("resident_plan.metal") ++ "\n" ++ @embedFile("normalization.metal") ++ "\n" ++ @embedFile("normalization_aligned.metal");
    return .{ .handle = fm_open(source) orelse return error.MetalFailure };
}

fn normWeights(buffer: metal.Buffer, offset: usize, dtype: u32, reference: []f32) void {
    @memset(buffer.structs(u8), 0xa5);
    for (reference, 0..) |*value, i| {
        value.* = @as(f32, @floatFromInt(i % 13 + 1)) / 17;
        switch (dtype) {
            0 => buffer.floats()[offset / 4 + i] = value.*,
            1 => {
                const rounded: f16 = @floatCast(value.*);
                buffer.structs(u16)[offset / 2 + i] = @bitCast(rounded);
                value.* = @floatCast(rounded);
            },
            2 => {
                const bits: u16 = @intCast(@as(u32, @bitCast(value.*)) >> 16);
                buffer.structs(u16)[offset / 2 + i] = bits;
                value.* = @bitCast(@as(u32, bits) << 16);
            },
            else => unreachable,
        }
    }
}

fn fillInput(values: []f32, seed: usize) void {
    for (values, 0..) |*value, i| value.* = @as(f32, @floatFromInt((i * 7 + seed) % 31)) / 29 - 0.75;
}

fn cpuNorm(input: []const f32, weights: []const f32, eps: f32, output: []f32) void {
    // Independent FP64 accumulation, not an emulation of either GPU tree.
    var square_sum: f64 = 0;
    for (input) |value| square_sum += @as(f64, value) * value;
    const inv: f32 = @floatCast(1 / @sqrt(square_sum / @as(f64, @floatFromInt(input.len)) + eps));
    for (input, weights, output) |value, weight, *out| out.* = value * inv * weight;
}

fn expectClose(expected: []const f32, actual: []const f32) !void {
    try std.testing.expectEqual(expected.len, actual.len);
    for (expected, actual) |ref, value| try std.testing.expectApproxEqAbs(ref, value, tolerance);
}

fn expectGuard(values: []const f32) !void {
    for (values) |value| try std.testing.expectEqual(sentinel, value);
}

test "aligned 2048 RMSNorm FP64 reference, weight byte offsets, dtypes and in-place output" {
    const ctx = try context();
    defer ctx.deinit();
    const w = try ctx.alloc(hidden + 4);
    defer w.deinit();
    const x = try ctx.alloc(hidden + 4);
    defer x.deinit();
    const out = try ctx.alloc(hidden + 4);
    defer out.deinit();
    var weights: [hidden]f32 = undefined;
    var input: [hidden]f32 = undefined;
    var expected: [hidden]f32 = undefined;
    fillInput(&input, 3);
    for (0..3) |dtype| {
        const offset: usize = if (dtype == 0) 12 else 2;
        normWeights(w, offset, @intCast(dtype), &weights);
        cpuNorm(&input, &weights, 1e-6, &expected);
        for ([_]bool{ false, true }) |in_place| {
            @memset(x.floats(), sentinel);
            @memset(out.floats(), sentinel);
            @memcpy(x.floats()[0..hidden], &input);
            const destination = if (in_place) x else out;
            try ctx.dispatch("rmsnorm_aligned", &.{ w, x, destination }, .{ .n = hidden, .m = 1, .offset = offset, .dtype = @intCast(dtype) }, 512, 512);
            try ctx.finish();
            try expectClose(&expected, destination.floats()[0..hidden]);
            try expectGuard(destination.floats()[hidden..]);
            try expectGuard(x.floats()[hidden..]);
            if (!in_place) try std.testing.expectEqualSlices(f32, &input, x.floats()[0..hidden]);
            if (in_place) try expectGuard(out.floats());
        }
    }
}

test "aligned 2048 residual RMSNorm FP64 reference, updated residual and read-only projection" {
    const ctx = try context();
    defer ctx.deinit();
    const w = try ctx.alloc(hidden + 4);
    defer w.deinit();
    const x = try ctx.alloc(hidden + 4);
    defer x.deinit();
    const projection = try ctx.alloc(hidden + 4);
    defer projection.deinit();
    const out = try ctx.alloc(hidden + 4);
    defer out.deinit();
    var weights: [hidden]f32 = undefined;
    var residual: [hidden]f32 = undefined;
    var projected: [hidden]f32 = undefined;
    var expected: [hidden]f32 = undefined;
    fillInput(&projected, 19);
    @memset(projection.floats(), sentinel);
    @memcpy(projection.floats()[0..hidden], &projected);
    for (0..3) |dtype| {
        const offset: usize = if (dtype == 0) 8 else 6;
        @memset(x.floats(), sentinel);
        @memset(out.floats(), sentinel);
        fillInput(x.floats()[0..hidden], 5);
        // The residual update is FP32, as in the model; its norm reference is FP64.
        for (&residual, x.floats()[0..hidden], &projected) |*ref, a, b| ref.* = a + b;
        normWeights(w, offset, @intCast(dtype), &weights);
        cpuNorm(&residual, &weights, 3e-5, &expected);
        try ctx.dispatch("residual_norm_aligned", &.{ w, x, projection, out }, .{ .n = hidden, .m = 1, .offset = offset, .dtype = @intCast(dtype), .eps = 3e-5 }, 512, 512);
        try ctx.finish();
        try std.testing.expectEqualSlices(f32, &residual, x.floats()[0..hidden]);
        try expectClose(&expected, out.floats()[0..hidden]);
        try std.testing.expectEqualSlices(f32, &projected, projection.floats()[0..hidden]);
        try expectGuard(x.floats()[hidden..]);
        try expectGuard(out.floats()[hidden..]);
        try expectGuard(projection.floats()[hidden..]);
    }
}

fn cpuHeads(input: []const f32, weights: []const f32, pos: u32, rope: bool, output: []f32) void {
    const half = dim / 2;
    for (0..input.len / dim) |head| {
        const row = output[head * dim ..][0..dim];
        cpuNorm(input[head * dim ..][0..dim], weights, 1e-6, row);
        if (rope) for (0..half) |j| {
            // Independent mathematical split-half RoPE reference at bounded positions.
            const angle = @as(f32, @floatFromInt(pos)) * std.math.pow(f32, 10000, -2 * @as(f32, @floatFromInt(j)) / @as(f32, dim));
            const a = row[j];
            const b = row[j + half];
            row[j] = a * @cos(angle) - b * @sin(angle);
            row[j + half] = b * @cos(angle) + a * @sin(angle);
        };
    }
}

test "aligned QK128 CPU reference, optional RoPE, KV ring wrap and padded head tails" {
    const ctx = try context();
    defer ctx.deinit();
    const heads = 3;
    const kv_heads = 2;
    const capacity = 3;
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
    var qweights: [dim]f32 = undefined;
    var kweights: [dim]f32 = undefined;
    var qref: [qn]f32 = undefined;
    var kref: [kn]f32 = undefined;
    var vref: [kn]f32 = undefined;
    var keyref: [capacity * kn + 4]f32 = undefined;
    var valueref: [capacity * kn + 4]f32 = undefined;
    for (0..3) |qdtype| {
        const kdtype = (qdtype + 1) % 3;
        const qoffset: usize = if (qdtype == 0) 12 else 2;
        const koffset: usize = if (kdtype == 0) 8 else 6;
        normWeights(qw, qoffset, @intCast(qdtype), &qweights);
        normWeights(kw, koffset, @intCast(kdtype), &kweights);
        for ([_]bool{ false, true }) |rope| {
            @memset(keys.floats(), sentinel);
            @memset(values.floats(), sentinel);
            @memset(&keyref, sentinel);
            @memset(&valueref, sentinel);
            // Small absolute positions: 4 wraps to slot 1; slot 2 stays untouched.
            for ([_]u32{ 0, 1, 4 }) |pos| {
                @memset(q.floats(), sentinel);
                @memset(k.floats(), sentinel);
                @memset(v.floats(), sentinel);
                fillInput(q.floats()[0..qn], 3 + pos);
                fillInput(k.floats()[0..kn], 9 + pos);
                fillInput(&vref, 17 + pos);
                @memcpy(v.floats()[0..kn], &vref);
                cpuHeads(q.floats()[0..qn], &qweights, pos, rope, &qref);
                cpuHeads(k.floats()[0..kn], &kweights, pos, rope, &kref);
                const slot = @as(usize, pos % capacity) * kn;
                @memcpy(keyref[slot..][0..kn], &kref);
                @memcpy(valueref[slot..][0..kn], &vref);
                const p: metal.Params = .{
                    .offset = qoffset,
                    .scale_offset = koffset,
                    .dtype = @intCast(qdtype),
                    .scale_dtype = @intCast(kdtype),
                    .heads = heads,
                    .kv_heads = kv_heads,
                    .dim = dim,
                    .pos = pos,
                    .capacity = capacity,
                    .bits = @intFromBool(rope),
                    .theta = @log2(@as(f32, 10000)),
                };
                // Eight SIMD heads for five real heads, in both supported groupings.
                const group: usize = if (qdtype == 1) 128 else 32;
                try ctx.dispatch("qk_rope_cache_aligned", &.{ q, k, v, qw, kw, keys, values }, p, 8 * 32, group);
                try ctx.finish();
                try expectClose(&qref, q.floats()[0..qn]);
                try expectClose(&kref, k.floats()[0..kn]);
                try expectClose(&keyref, keys.floats());
                try std.testing.expectEqualSlices(f32, &valueref, values.floats());
                // Untouched ring slots and tails must remain exact, not approximate.
                try expectGuard(keys.floats()[2 * kn ..]);
                try expectGuard(values.floats()[2 * kn ..]);
                try expectGuard(q.floats()[qn..]);
                try expectGuard(k.floats()[kn..]);
                try expectGuard(v.floats()[kn..]);
                try std.testing.expectEqualSlices(f32, &vref, v.floats()[0..kn]);
                try std.testing.expectEqualSlices(f32, k.floats()[0..kn], keys.floats()[slot..][0..kn]);
            }
        }
    }
}
