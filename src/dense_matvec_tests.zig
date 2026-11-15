const std = @import("std");
const metal = @import("metal.zig");

const sentinel: f32 = -1234.5;
const byte_sentinel: u8 = 0xa5;
const kernels = [_][:0]const u8{ "dense_matvec_f32", "dense_matvec_f16", "dense_matvec_bf16" };
const split_kernels = [_][:0]const u8{ "dense_matvec_split_f32", "dense_matvec_split_f16", "dense_matvec_split_bf16" };

// Replace each source value with the value actually stored in that dtype, so
// quantization error is never charged to the GPU dot product. BF16 truncation
// matches kernels.metal's weight() decoder and checkpoint representation.
fn packWeights(bytes: []u8, offset: usize, dtype: u32, values: []f32) void {
    @memset(bytes, byte_sentinel);
    for (values, 0..) |*value, i| {
        switch (dtype) {
            0 => std.mem.writeInt(u32, bytes[offset + i * 4 ..][0..4], @bitCast(value.*), .little),
            1 => {
                const rounded: f16 = @floatCast(value.*);
                std.mem.writeInt(u16, bytes[offset + i * 2 ..][0..2], @bitCast(rounded), .little);
                value.* = @floatCast(rounded);
            },
            2 => {
                const bits: u16 = @intCast(@as(u32, @bitCast(value.*)) >> 16);
                std.mem.writeInt(u16, bytes[offset + i * 2 ..][0..2], bits, .little);
                value.* = @bitCast(@as(u32, bits) << 16);
            },
            else => unreachable,
        }
    }
}

fn fillWeights(values: []f32) void {
    for (values, 0..) |*value, i| value.* = @as(f32, @floatFromInt((i * 17 + i / 7) % 71)) / 37 - 0.9;
}

fn fillInput(values: []f32) void {
    for (values, 0..) |*value, i| value.* = @as(f32, @floatFromInt((i * 7 + 3) % 31)) / 29 - 0.45;
}

fn cpuMatvec(weights: []const f32, input: []const f32, output: []f32) void {
    for (output, 0..) |*out, row| {
        // Higher precision avoids comparing GPU SIMD reduction with CPU serial
        // FP32 summation error. Shader accumulation itself stays FP32.
        var sum: f64 = 0;
        for (input, 0..) |value, col| sum += @as(f64, weights[row * input.len + col]) * value;
        out.* = @floatCast(sum);
    }
}

fn expectClose(expected: []const f32, actual: []const f32, tolerance: f32) !void {
    try std.testing.expectEqual(expected.len, actual.len);
    for (expected, actual) |ref, value| try std.testing.expectApproxEqAbs(ref, value, tolerance);
}

test "dense CPU weight packing preserves byte offsets and represented dtype values" {
    const source = [_]f32{ 1.00390625, -2.0078125, 0.333251953125 };
    for (0..3) |dtype| {
        var bytes: [32]u8 = undefined;
        var values = source;
        const offset: usize = if (dtype == 0) 12 else 2;
        const width: usize = if (dtype == 0) 4 else 2;
        packWeights(&bytes, offset, @intCast(dtype), &values);
        for (bytes[0..offset]) |byte| try std.testing.expectEqual(byte_sentinel, byte);
        for (bytes[offset + source.len * width ..]) |byte| try std.testing.expectEqual(byte_sentinel, byte);
        const expected = if (dtype == 2) [_]f32{ 1, -2, 0.33203125 } else source;
        try std.testing.expectEqualSlices(f32, &expected, &values);
        for (values, 0..) |value, i| {
            const decoded: f32 = switch (dtype) {
                0 => @bitCast(std.mem.readInt(u32, bytes[offset + i * 4 ..][0..4], .little)),
                1 => @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, bytes[offset + i * 2 ..][0..2], .little)))),
                2 => @bitCast(@as(u32, std.mem.readInt(u16, bytes[offset + i * 2 ..][0..2], .little)) << 16),
                else => unreachable,
            };
            try std.testing.expectEqual(value, decoded);
        }
    }
    var output: [2]f32 = undefined;
    cpuMatvec(&.{ 1, -2, 3, -4, 5, -6 }, &.{ 0.5, 2, -1 }, &output);
    try std.testing.expectEqualSlices(f32, &.{ -6.5, 14 }, &output);
}

test "dense dtype-specialized matvec matches CPU and original with row tails and packed half offsets" {
    // Root integration must append dense_matvec.metal to Context.init's source
    // and import this test module. This test intentionally uses the normal ABI.
    const ctx = try metal.Context.init();
    defer ctx.deinit();
    const rows = 5;
    const guard = 4;
    // 33 catches packed-half odd row starts; 128 checks the unrolled block;
    // 129 combines an unrolled block and tail; 2048 is the router input width.
    for ([_]usize{ 0, 1, 31, 32, 33, 127, 128, 129, 255, 256, 257, 2048 }) |n| {
        const w = try ctx.alloc(rows * n + guard);
        defer w.deinit();
        const x = try ctx.alloc(n + guard);
        defer x.deinit();
        const original = try ctx.alloc(rows + guard);
        defer original.deinit();
        const out = try ctx.alloc(rows + guard);
        defer out.deinit();
        const weights = try std.testing.allocator.alloc(f32, rows * n);
        defer std.testing.allocator.free(weights);
        const input = try std.testing.allocator.alloc(f32, n);
        defer std.testing.allocator.free(input);
        var expected: [rows]f32 = undefined;
        fillInput(input);
        @memset(x.floats(), sentinel);
        @memcpy(x.floats()[0..n], input);
        for (kernels, 0..) |kernel, dtype| {
            // Half offsets deliberately are not float/half2 aligned. Odd n
            // additionally alternates 2-byte row alignment for F16 and BF16.
            const offset: usize = if (dtype == 0) 12 else 2;
            const width: usize = if (dtype == 0) 4 else 2;
            fillWeights(weights);
            packWeights(w.structs(u8), offset, @intCast(dtype), weights);
            cpuMatvec(weights, input, &expected);
            const p: metal.Params = .{ .offset = offset, .n = @intCast(n), .m = rows, .dtype = @intCast(dtype) };
            @memset(original.floats(), sentinel);
            try ctx.dispatch("matvec", &.{ w, x, original }, p, rows * 32, 32);
            try ctx.finish();
            try expectClose(&expected, original.floats()[0..rows], 2e-4);
            var specialized_p = p;
            // Kernel selection, not the legacy runtime dtype field, must
            // determine decoding. Poison it to expose accidental fallback.
            specialized_p.dtype = 99;
            for ([_]usize{ 32, 64, 128 }) |group| {
                @memset(out.floats(), sentinel);
                // Pad to eight rows: excess SIMD groups must leave guards alone.
                try ctx.dispatch(kernel, &.{ w, x, out }, specialized_p, 8 * 32, group);
                try ctx.finish();
                try expectClose(&expected, out.floats()[0..rows], 2e-4);
                try expectClose(original.floats()[0..rows], out.floats()[0..rows], 1e-5);
                for (out.floats()[rows..]) |value| try std.testing.expectEqual(sentinel, value);
                for (original.floats()[rows..]) |value| try std.testing.expectEqual(sentinel, value);
                try std.testing.expectEqualSlices(f32, input, x.floats()[0..n]);
                for (x.floats()[n..]) |value| try std.testing.expectEqual(sentinel, value);
                for (w.structs(u8)[0..offset]) |byte| try std.testing.expectEqual(byte_sentinel, byte);
                for (w.structs(u8)[offset + rows * n * width ..]) |byte| try std.testing.expectEqual(byte_sentinel, byte);
            }
        }
    }
}

test "dense split matvec matches CPU and original kernels with fixed 256-thread groups" {
    const ctx = try metal.Context.init();
    defer ctx.deinit();
    const rows = 5;
    const guard = 4;
    // Retain every existing width, including empty input and packed-half odd
    // rows, and cover both sides of the intended 512-column dispatch threshold.
    for ([_]usize{ 0, 1, 31, 32, 33, 127, 128, 129, 255, 256, 257, 511, 512, 513, 2048 }) |n| {
        const w = try ctx.alloc(rows * n + guard);
        defer w.deinit();
        const x = try ctx.alloc(n + guard);
        defer x.deinit();
        const original = try ctx.alloc(rows + guard);
        defer original.deinit();
        const dense = try ctx.alloc(rows + guard);
        defer dense.deinit();
        const out = try ctx.alloc(rows + guard);
        defer out.deinit();
        const weights = try std.testing.allocator.alloc(f32, rows * n);
        defer std.testing.allocator.free(weights);
        const input = try std.testing.allocator.alloc(f32, n);
        defer std.testing.allocator.free(input);
        var expected: [rows]f32 = undefined;
        fillInput(input);
        @memset(x.floats(), sentinel);
        @memcpy(x.floats()[0..n], input);
        for (split_kernels, 0..) |kernel, dtype| {
            const offset: usize = if (dtype == 0) 12 else 2;
            const width: usize = if (dtype == 0) 4 else 2;
            fillWeights(weights);
            packWeights(w.structs(u8), offset, @intCast(dtype), weights);
            cpuMatvec(weights, input, &expected);
            const p: metal.Params = .{ .offset = offset, .n = @intCast(n), .m = rows, .dtype = @intCast(dtype) };
            @memset(original.floats(), sentinel);
            @memset(dense.floats(), sentinel);
            @memset(out.floats(), sentinel);
            try ctx.dispatch("matvec", &.{ w, x, original }, p, rows * 32, 32);
            var specialized_p = p;
            specialized_p.dtype = 99;
            try ctx.dispatch(kernels[dtype], &.{ w, x, dense }, specialized_p, 8 * 32, 32);
            // Eight whole threadgroups produce five rows and three guarded
            // excess rows. Never substitute the old one-SIMD-per-row grid.
            try ctx.dispatch(kernel, &.{ w, x, out }, specialized_p, 8 * 256, 256);
            try ctx.finish();
            try expectClose(&expected, original.floats()[0..rows], 2e-4);
            try expectClose(&expected, dense.floats()[0..rows], 2e-4);
            try expectClose(&expected, out.floats()[0..rows], 2e-4);
            try expectClose(original.floats()[0..rows], out.floats()[0..rows], 1e-5);
            try expectClose(dense.floats()[0..rows], out.floats()[0..rows], 1e-5);
            for (original.floats()[rows..]) |value| try std.testing.expectEqual(sentinel, value);
            for (dense.floats()[rows..]) |value| try std.testing.expectEqual(sentinel, value);
            for (out.floats()[rows..]) |value| try std.testing.expectEqual(sentinel, value);
            try std.testing.expectEqualSlices(f32, input, x.floats()[0..n]);
            for (x.floats()[n..]) |value| try std.testing.expectEqual(sentinel, value);
            for (w.structs(u8)[0..offset]) |byte| try std.testing.expectEqual(byte_sentinel, byte);
            for (w.structs(u8)[offset + rows * n * width ..]) |byte| try std.testing.expectEqual(byte_sentinel, byte);
        }
    }
}
