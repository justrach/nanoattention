const std = @import("std");
const metal = @import("metal.zig");

test "packed BF16 router matches matched split bits with byte offsets and overlaunch guards" {
    const ctx = try metal.Context.initMatched(true);
    defer ctx.deinit();
    const input = try ctx.alloc(2048 + 16);
    defer input.deinit();
    const original = try ctx.alloc(256 + 16);
    defer original.deinit();
    const candidate = try ctx.alloc(256 + 16);
    defer candidate.deinit();
    for (input.floats()[0..2048], 0..) |*value, i|
        value.* = @as(f32, @floatFromInt((i * 17) % 71)) / 37 - 0.9;
    @memset(input.structs(u32)[2048..], 0x4b123456);
    for ([_]usize{ 2, 6 }) |offset| {
        const w = try ctx.alloc((offset + 256 * 2048 * 2 + 64) / 4);
        defer w.deinit();
        defer ctx.finish() catch {};
        @memset(w.structs(u16), 0x5a6d);
        var state: u32 = 0x12345678;
        for (0..256 * 2048) |i| {
            state = state *% 1664525 +% 1013904223;
            const value = @as(f32, @floatFromInt(state % 151)) / 79 - 0.95;
            w.structs(u16)[offset / 2 + i] = @truncate(@as(u32, @bitCast(value)) >> 16);
        }
        @memset(original.structs(u32), 0x7fc12345);
        @memset(candidate.structs(u32), 0x7fc54321);
        const p: metal.Params = .{ .m = 256, .n = 2048, .dtype = 2, .offset = offset };
        try ctx.dispatch("dense_matvec_split_bf16", &.{ w, input, original }, p, 256 * 256, 256);
        try ctx.dispatch("dense_matvec_router_bf16", &.{ w, input, candidate }, p, (256 + 8) * 32, 128);
        try ctx.finish();
        try std.testing.expectEqualSlices(u32, original.structs(u32)[0..256], candidate.structs(u32)[0..256]);
        for (candidate.floats()[0..256]) |value| try std.testing.expect(std.math.isFinite(value));
        for (candidate.structs(u32)[256..]) |value| try std.testing.expectEqual(@as(u32, 0x7fc54321), value);
        for (input.structs(u32)[2048..]) |value| try std.testing.expectEqual(@as(u32, 0x4b123456), value);
        for (w.structs(u16)[0 .. offset / 2]) |value| try std.testing.expectEqual(@as(u16, 0x5a6d), value);
        for (w.structs(u16)[offset / 2 + 256 * 2048 ..]) |value| try std.testing.expectEqual(@as(u16, 0x5a6d), value);
        // A shape outside the host-selected profile must not partially write.
        @memset(candidate.structs(u32), 0x7fc54321);
        try ctx.dispatch("dense_matvec_router_bf16", &.{ w, input, candidate }, .{ .m = 255, .n = 2048, .dtype = 2, .offset = offset }, p.m * 32, 128);
        try ctx.finish();
        for (candidate.structs(u32)) |value| try std.testing.expectEqual(@as(u32, 0x7fc54321), value);
    }
}
