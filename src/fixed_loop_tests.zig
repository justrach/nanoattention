const std = @import("std");
const metal = @import("metal.zig");
const plan = @import("resident_plan.zig");

// Wiring: append row_tiling_fixed.metal after row_tiling.metal in Context's
// embedded source and import this file from main.zig's test block. Both are
// deliberately left to the integrating caller; no private Metal bridge here.
const rows = 5;
const guards = 64;
const sentinel: u32 = 0x7fc12345;
const Kind = enum { int4_g64, int4_generic_g32, fp32, fp16, bf16 };
const kernels = struct {
    const qmatvec = [_][:0]const u8{ "staged_qmatvec_t1", "staged_qmatvec_t2", "staged_qmatvec_t4", "staged_qmatvec_t8" };
    const qmatvec_fixed = [_][:0]const u8{ "staged_qmatvec_fixed_t1", "staged_qmatvec_fixed_t2", "staged_qmatvec_fixed_t4", "staged_qmatvec_fixed_t8" };
    const qkv = [_][:0]const u8{ "staged_qkv_t1", "staged_qkv_t2", "staged_qkv_t4", "staged_qkv_t8" };
    const qkv_fixed = [_][:0]const u8{ "staged_qkv_fixed_t1", "staged_qkv_fixed_t2", "staged_qkv_fixed_t4", "staged_qkv_fixed_t8" };
    const gate_up = [_][:0]const u8{ "staged_gate_up_t1", "staged_gate_up_t2", "staged_gate_up_t4", "staged_gate_up_t8" };
    const gate_up_fixed = [_][:0]const u8{ "staged_gate_up_fixed_t1", "staged_gate_up_fixed_t2", "staged_gate_up_fixed_t4", "staged_gate_up_fixed_t8" };
    const down = [_][:0]const u8{ "staged_down_t1", "staged_down_t2", "staged_down_t4", "staged_down_t8" };
    const down_fixed = [_][:0]const u8{ "staged_down_fixed_t1", "staged_down_fixed_t2", "staged_down_fixed_t4", "staged_down_fixed_t8" };
};

test "fixed column loops preserve all output bits in matched MLX math" {
    try checkFixed(true);
}

test "fixed column loops preserve legacy output bits and residual ordering" {
    try checkFixed(false);
}

fn checkFixed(matched: bool) !void {
    const ctx = try metal.Context.initMatched(matched);
    defer ctx.deinit();
    // All supported widths, plus two generic-loop fallbacks (including a
    // non-uniform number of 16-element segments per lane at K=1088).
    for ([_]u32{ 512, 1024, 2048, 4096, 6144, 128, 1088 }) |cols| {
        for ([_]u32{ 0, 1, 2 }) |affine_dtype| {
            const fixture = try Fixture.init(ctx, cols, .int4_g64, affine_dtype);
            defer fixture.deinit();
            try checkMatvecQkv(ctx, fixture, true);
            try checkRoutes(ctx, fixture, false);
        }
    }
    // A supported K must not select the fixed INT4 path for other dtypes or
    // quantization groups. K=128 also tests those fallback combinations.
    for ([_]u32{ 512, 128 }) |cols| {
        for ([_]Kind{ .int4_generic_g32, .fp32, .fp16, .bf16 }) |kind| {
            const fixture = try Fixture.init(ctx, cols, kind, 0);
            defer fixture.deinit();
            try checkMatvecQkv(ctx, fixture, false);
            try checkRoutes(ctx, fixture, false);
        }
    }
    // Different up widths force the original separate-dot fallback while gate
    // and down remain supported shapes. Covers selected and shared experts.
    const fixture = try Fixture.init(ctx, 512, .int4_g64, 0);
    defer fixture.deinit();
    try checkRoutes(ctx, fixture, true);
}

fn writeFloat(buffer: metal.Buffer, index: usize, dtype: u32, value: f32) void {
    switch (dtype) {
        0 => buffer.floats()[index] = value,
        1 => {
            const rounded: f16 = @floatCast(value);
            buffer.structs(u16)[index] = @bitCast(rounded);
        },
        2 => {
            const bits: u32 = @bitCast(value);
            buffer.structs(u16)[index] = @truncate(bits >> 16);
        },
        else => unreachable,
    }
}

const Fixture = struct {
    cols: u32,
    affine_dtype: u32,
    quantized: metal.Buffer,
    floating: metal.Buffer,
    scales: metal.Buffer,
    biases: metal.Buffer,
    x: metal.Buffer,
    matrices: [3]metal.Matrix,

    fn init(ctx: metal.Context, cols: u32, kind: Kind, affine_dtype: u32) !Fixture {
        const elements: usize = rows * cols;
        const is_quantized = kind == .int4_g64 or kind == .int4_generic_g32;
        const group_size: u32 = if (kind == .int4_generic_g32) 32 else 64;
        const groups = elements / group_size;
        const quantized = try ctx.alloc(1 + 3 * elements / 8);
        errdefer quantized.deinit();
        const floating = try ctx.alloc(2 + 3 * elements);
        errdefer floating.deinit();
        // FP32-sized slots accommodate FP16/BF16 as well; all base offsets are
        // nonzero, independently checking quantized/scale/bias byte addressing.
        const scales = try ctx.alloc(1 + 3 * groups);
        errdefer scales.deinit();
        const biases = try ctx.alloc(2 + 3 * groups);
        errdefer biases.deinit();
        const x = try ctx.alloc(cols);
        errdefer x.deinit();
        @memset(quantized.structs(u8), 0);
        @memset(floating.structs(u32), sentinel);
        @memset(scales.structs(u32), sentinel);
        @memset(biases.structs(u32), sentinel);
        for (x.floats(), 0..) |*value, col|
            value.* = @as(f32, @floatFromInt((col * 5 + col / 19) % 47)) / 29 - 0.8;
        const dtype: u32 = switch (kind) {
            .int4_g64, .int4_generic_g32 => 3,
            .fp32 => 0,
            .fp16 => 1,
            .bf16 => 2,
        };
        var matrices: [3]metal.Matrix = undefined;
        for (0..3) |slot| {
            for (0..elements) |i| {
                const index = slot * elements + i;
                const code: u8 = @intCast((i * 7 + slot * 11 + i / cols) % 16);
                quantized.structs(u8)[4 + index / 2] |= code << @as(u3, @intCast(index % 2 * 4));
                if (!is_quantized) {
                    const value = @as(f32, @floatFromInt((i * 13 + slot * 17 + i / cols) % 53)) / 137 - 0.2;
                    const base = 8 + slot * elements * 4;
                    writeFloat(floating, base / @as(usize, if (dtype == 0) 4 else 2) + i, dtype, value);
                }
            }
            for (0..groups) |g| {
                const element_bytes: usize = if (affine_dtype == 0) 4 else 2;
                const scale_base = 4 + slot * groups * 4;
                const bias_base = 8 + slot * groups * 4;
                writeFloat(scales, scale_base / element_bytes + g, affine_dtype, @as(f32, @floatFromInt((g * 3 + slot * 5) % 17 + 1)) / 2048);
                writeFloat(biases, bias_base / element_bytes + g, affine_dtype, @as(f32, @floatFromInt((g + slot * 3) % 19)) / 512 - 0.015625);
            }
            matrices[slot] = .{
                .weight = if (is_quantized) quantized.address() + 4 + slot * elements / 2 else floating.address() + 8 + slot * elements * 4,
                .scales = scales.address() + 4 + slot * groups * 4,
                .biases = biases.address() + 8 + slot * groups * 4,
                .rows = rows,
                .cols = cols,
                .dtype = dtype,
                .bits = if (is_quantized) 4 else 0,
                .group_size = if (is_quantized) group_size else 0,
                .scale_dtype = affine_dtype,
                .bias_dtype = affine_dtype,
                .reserved = if (kind == .int4_g64) 1 else 0,
            };
        }
        return .{ .cols = cols, .affine_dtype = affine_dtype, .quantized = quantized, .floating = floating, .scales = scales, .biases = biases, .x = x, .matrices = matrices };
    }

    fn deinit(self: Fixture) void {
        self.x.deinit();
        self.biases.deinit();
        self.scales.deinit();
        self.floating.deinit();
        self.quantized.deinit();
    }

    fn resident(self: Fixture, ctx: metal.Context) !void {
        try ctx.resident(self.quantized);
        try ctx.resident(self.floating);
        try ctx.resident(self.scales);
        try ctx.resident(self.biases);
    }
};

fn poison(buffer: metal.Buffer) void {
    @memset(buffer.structs(u32), sentinel);
}

fn expectOutputs(reference: metal.Buffer, actual: metal.Buffer, count: usize) !void {
    for (reference.floats()[0..count]) |value| try std.testing.expect(std.math.isFinite(value));
    // Include every guard/padding word, not just computed outputs.
    try std.testing.expectEqualSlices(u32, reference.structs(u32), actual.structs(u32));
    for (actual.structs(u32)[count..]) |bits| try std.testing.expectEqual(sentinel, bits);
}

fn checkMatvecQkv(ctx: metal.Context, fixture: Fixture, qmatvec: bool) !void {
    const descriptor = try ctx.alloc(3 * @sizeOf(metal.Matrix) / 4);
    defer descriptor.deinit();
    @memcpy(descriptor.structs(metal.Matrix), &fixture.matrices);
    descriptor.structs(metal.Matrix)[1].rows = 3;
    descriptor.structs(metal.Matrix)[2].rows = 1;
    var reference: [3]metal.Buffer = undefined;
    var actual: [3]metal.Buffer = undefined;
    // Allocate paired buffers separately so an unexpected overwrite is visible
    // even for the one-row V projection and every tile's partial final row.
    var allocated: usize = 0;
    defer for (0..allocated) |i| {
        reference[i].deinit();
        actual[i].deinit();
    };
    for (0..3) |i| {
        reference[i] = try ctx.alloc(rows + guards);
        actual[i] = ctx.alloc(rows + guards) catch |err| {
            reference[i].deinit();
            return err;
        };
        allocated += 1;
    }
    for (plan.row_tiles, 0..) |tile, ti| {
        if (qmatvec) {
            poison(reference[0]);
            poison(actual[0]);
            const p: metal.Params = .{ .offset = 4, .scale_offset = 4, .bias_offset = 8, .n = fixture.cols, .m = rows, .k = 64, .bits = 4, .dtype = 3, .scale_dtype = fixture.affine_dtype, .bias_dtype = fixture.affine_dtype };
            const threads = ((rows + tile - 1) / tile + 1) * 32;
            try ctx.dispatch(kernels.qmatvec[ti], &.{ fixture.quantized, fixture.scales, fixture.biases, fixture.x, reference[0] }, p, threads, 32);
            try ctx.dispatch(kernels.qmatvec_fixed[ti], &.{ fixture.quantized, fixture.scales, fixture.biases, fixture.x, actual[0] }, p, threads, 32);
            try ctx.finish();
            try expectOutputs(reference[0], actual[0], rows);
        }
        for (reference) |buffer| poison(buffer);
        for (actual) |buffer| poison(buffer);
        const tiles = (rows + tile - 1) / tile + (3 + tile - 1) / tile + 1;
        try fixture.resident(ctx);
        try ctx.dispatch(kernels.qkv[ti], &.{ descriptor, fixture.x, reference[0], reference[1], reference[2] }, .{}, (tiles + 1) * 32, 32);
        try ctx.dispatch(kernels.qkv_fixed[ti], &.{ descriptor, fixture.x, actual[0], actual[1], actual[2] }, .{}, (tiles + 1) * 32, 32);
        try ctx.finish();
        for ([_]usize{ 5, 3, 1 }, 0..) |count, i| try expectOutputs(reference[i], actual[i], count);
    }
}

fn checkRoutes(ctx: metal.Context, fixture: Fixture, mismatched_up: bool) !void {
    const experts = try ctx.alloc(3 * @sizeOf(metal.MLP) / 4);
    defer experts.deinit();
    for (experts.structs(metal.MLP), 0..) |*expert, i| {
        var up = fixture.matrices[(i + 1) % 3];
        if (mismatched_up) up.cols = 128;
        expert.* = .{ .gate = fixture.matrices[i], .up = up, .down = fixture.matrices[(i + 2) % 3] };
    }
    const ids = try ctx.alloc(2);
    defer ids.deinit();
    @memcpy(ids.structs(u32), &[_]u32{ 2, 0 });
    const weights = try ctx.alloc(3);
    defer weights.deinit();
    @memcpy(weights.floats(), &[_]f32{ 0.17, 0.31, std.math.nan(f32) });
    const gate_stride = rows + 16;
    const down_stride = fixture.cols + 16;
    const gate_reference = try ctx.alloc(3 * gate_stride + guards);
    defer gate_reference.deinit();
    const gate_actual = try ctx.alloc(3 * gate_stride + guards);
    defer gate_actual.deinit();
    const activation = try ctx.alloc(3 * down_stride);
    defer activation.deinit();
    poison(activation);
    for (0..3) |route| {
        for (0..fixture.cols) |col|
            activation.floats()[route * down_stride + col] = @as(f32, @floatFromInt((col * 3 + route * 11) % 41)) / 31 - 0.6;
    }
    const down_reference = try ctx.alloc(rows + guards);
    defer down_reference.deinit();
    const down_actual = try ctx.alloc(rows + guards);
    defer down_actual.deinit();
    // Shared-only and nontrivial selected expert order; shared weight is always
    // implicit 1.0, with a poisoned trailing explicit weight to catch misuse.
    for ([_]u32{ 0, 2 }) |routes| {
        for (plan.row_tiles, 0..) |tile, ti| {
            poison(gate_reference);
            poison(gate_actual);
            const gate_p: metal.Params = .{ .n = routes, .m = 1, .k = gate_stride };
            const gate_threads = ((routes + 1) * ((gate_stride + tile - 1) / tile) + 1) * 32;
            try fixture.resident(ctx);
            try ctx.dispatch(kernels.gate_up[ti], &.{ experts, fixture.x, ids, gate_reference }, gate_p, gate_threads, 32);
            try ctx.dispatch(kernels.gate_up_fixed[ti], &.{ experts, fixture.x, ids, gate_actual }, gate_p, gate_threads, 32);
            try ctx.finish();
            try std.testing.expectEqualSlices(u32, gate_reference.structs(u32), gate_actual.structs(u32));
            for (gate_actual.structs(u32), 0..) |bits, i| {
                if (i < (routes + 1) * gate_stride and i % gate_stride < rows) {
                    try std.testing.expect(std.math.isFinite(gate_actual.floats()[i]));
                } else try std.testing.expectEqual(sentinel, bits);
            }
            poison(down_reference);
            poison(down_actual);
            for (0..rows) |row| {
                const value = @as(f32, @floatFromInt(row * 7 % 31)) / 11 - 1.3;
                down_reference.floats()[row] = value;
                down_actual.floats()[row] = value;
            }
            const down_p: metal.Params = .{ .n = routes, .m = 1, .k = down_stride, .dim = rows };
            const down_threads = ((rows + tile - 1) / tile + 1) * 32;
            try fixture.resident(ctx);
            try ctx.dispatch(kernels.down[ti], &.{ experts, activation, ids, weights, down_reference }, down_p, down_threads, 32);
            try ctx.dispatch(kernels.down_fixed[ti], &.{ experts, activation, ids, weights, down_actual }, down_p, down_threads, 32);
            try ctx.finish();
            try expectOutputs(down_reference, down_actual, rows);
        }
    }
}
