const std = @import("std");
const metal = @import("metal.zig");
const plan = @import("resident_plan.zig");

test "certified row tiles preserve quantized and floating outputs including partial tails" {
    const ctx = try metal.Context.init();
    defer ctx.deinit();
    const storage = try ctx.alloc(81); // 4-byte offset + 5 * 128 / 2 packed bytes.
    defer storage.deinit();
    const floats = try ctx.alloc(5 * 128);
    defer floats.deinit();
    const scales = try ctx.alloc(11);
    defer scales.deinit();
    const biases = try ctx.alloc(12);
    defer biases.deinit();
    const x = try ctx.alloc(128);
    defer x.deinit();
    const descriptor = try ctx.alloc(3 * @sizeOf(metal.Matrix) / 4);
    defer descriptor.deinit();
    const y = try ctx.alloc(5);
    defer y.deinit();
    const key = try ctx.alloc(3);
    defer key.deinit();
    const value = try ctx.alloc(1);
    defer value.deinit();
    @memset(storage.structs(u8), 0);
    for (x.floats(), 0..) |*v, i| v.* = @as(f32, @floatFromInt(i % 11)) / 7 - 0.4;
    for (scales.floats(), 0..) |*v, i| v.* = @as(f32, @floatFromInt(i + 1)) / 16;
    for (biases.floats(), 0..) |*v, i| v.* = @as(f32, @floatFromInt(i % 5)) / 8 - 0.25;
    var quantized_reference: [5]f32 = @splat(0);
    var float_reference: [5]f32 = @splat(0);
    for (0..5 * 128) |i| {
        const code: u8 = @intCast((i * 7 + 3) % 16);
        storage.structs(u8)[4 + i / 2] |= code << @as(u3, @intCast(i % 2 * 4));
        const q = @as(f32, @floatFromInt(code)) * scales.floats()[1 + i / 64] + biases.floats()[2 + i / 64];
        quantized_reference[i / 128] += q * x.floats()[i % 128];
        const f = @as(f32, @floatFromInt(i % 19)) / 7 - 1.1;
        floats.floats()[i] = f;
        float_reference[i / 128] += f * x.floats()[i % 128];
    }
    const matrices = [_]metal.Matrix{
        .{ .weight = storage.address() + 4, .scales = scales.address() + 4, .biases = biases.address() + 8, .rows = 5, .cols = 128, .dtype = 3, .bits = 4, .group_size = 64, .reserved = 1 },
        .{ .weight = floats.address(), .rows = 5, .cols = 128, .dtype = 0 },
    };
    for (matrices, 0..) |matrix, kind| {
        @memset(descriptor.structs(metal.Matrix), matrix);
        descriptor.structs(metal.Matrix)[1].rows = 3;
        descriptor.structs(metal.Matrix)[2].rows = 1;
        for (plan.row_tiles) |tile| {
            @memset(y.floats(), std.math.nan(f32));
            try ctx.resident(storage);
            try ctx.resident(floats);
            try ctx.resident(scales);
            try ctx.resident(biases);
            try ctx.dispatch("row_tile_probe", &.{ descriptor, x, y }, .{ .n = tile }, ((5 + tile - 1) / tile) * 32, 32);
            try ctx.finish();
            const expected = if (kind == 0) quantized_reference else float_reference;
            for (expected, y.floats()) |ref, actual| try std.testing.expectApproxEqAbs(ref, actual, 2e-4);
            if (kind == 0) {
                const kernel: [:0]const u8 = switch (tile) {
                    1 => "staged_qmatvec_t1",
                    2 => "staged_qmatvec_t2",
                    4 => "staged_qmatvec_t4",
                    8 => "staged_qmatvec_t8",
                    else => unreachable,
                };
                @memset(y.floats(), std.math.nan(f32));
                try ctx.dispatch(kernel, &.{ storage, scales, biases, x, y }, .{ .offset = 4, .scale_offset = 4, .bias_offset = 8, .n = 128, .m = 5, .k = 64, .bits = 4, .dtype = 3 }, ((5 + tile - 1) / tile) * 32, 32);
                try ctx.finish();
                for (expected, y.floats()) |ref, actual| try std.testing.expectApproxEqAbs(ref, actual, 2e-4);
            }
            const qkv_kernel: [:0]const u8 = switch (tile) {
                1 => "staged_qkv_t1",
                2 => "staged_qkv_t2",
                4 => "staged_qkv_t4",
                8 => "staged_qkv_t8",
                else => unreachable,
            };
            @memset(y.floats(), std.math.nan(f32));
            @memset(key.floats(), std.math.nan(f32));
            @memset(value.floats(), std.math.nan(f32));
            try ctx.resident(storage);
            try ctx.resident(floats);
            try ctx.resident(scales);
            try ctx.resident(biases);
            const tiles = (5 + tile - 1) / tile + (3 + tile - 1) / tile + 1;
            try ctx.dispatch(qkv_kernel, &.{ descriptor, x, y, key, value }, .{}, tiles * 32, 32);
            try ctx.finish();
            for (expected, y.floats()) |ref, actual| try std.testing.expectApproxEqAbs(ref, actual, 2e-4);
            for (expected[0..3], key.floats()) |ref, actual| try std.testing.expectApproxEqAbs(ref, actual, 2e-4);
            try std.testing.expectApproxEqAbs(expected[0], value.floats()[0], 2e-4);
        }
    }
}
