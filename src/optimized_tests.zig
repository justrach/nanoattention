const std = @import("std");
const metal = @import("metal.zig");

test "packed INT4 group-64 vector matvec matches scalar affine reference with offsets" {
    try checkPacked(false);
}

test "MLX-grouped packed INT4 matvec matches scalar affine reference" {
    try checkPacked(true);
}

fn checkPacked(matched: bool) !void {
    const ctx = try metal.Context.initMatched(matched);
    defer ctx.deinit();
    const storage = try ctx.alloc(33); // 4-byte offset plus two 128-value INT4 rows.
    defer storage.deinit();
    const scales = try ctx.alloc(5);
    defer scales.deinit();
    const biases = try ctx.alloc(6);
    defer biases.deinit();
    const x = try ctx.alloc(128);
    defer x.deinit();
    const generic = try ctx.alloc(2);
    defer generic.deinit();
    const vector = try ctx.alloc(2);
    defer vector.deinit();
    @memset(storage.structs(u8), 0);
    @memcpy(scales.floats(), &[_]f32{ 0, 0.125, 0.25, 0.375, 0.5 });
    @memcpy(biases.floats(), &[_]f32{ 0, 0, -0.25, 0.125, 0.5, -1 });
    for (x.floats(), 0..) |*v, i| v.* = @as(f32, @floatFromInt(i % 11)) / 4 - 1;
    var expected: [2]f32 = .{ 0, 0 };
    for (0..256) |i| {
        const code: u8 = @intCast((i * 7 + 3) % 16);
        storage.structs(u8)[4 + i / 2] |= code << @as(u3, @intCast((i % 2) * 4));
        expected[i / 128] += (@as(f32, @floatFromInt(code)) * scales.floats()[1 + i / 64] + biases.floats()[2 + i / 64]) * x.floats()[i % 128];
    }
    const p: metal.Params = .{ .offset = 4, .scale_offset = 4, .bias_offset = 8, .n = 128, .m = 2, .k = 64, .bits = 4, .dtype = 3 };
    try ctx.dispatch("qmatvec", &.{ storage, scales, biases, x, generic }, p, 64, 32);
    try ctx.dispatch("qmatvec4_g64", &.{ storage, scales, biases, x, vector }, p, 64, 32);
    try ctx.finish();
    for (expected, generic.floats(), vector.floats()) |ref, scalar, fast| {
        try std.testing.expectApproxEqAbs(ref, scalar, 2e-4);
        try std.testing.expectApproxEqAbs(ref, fast, 2e-4);
    }
}
