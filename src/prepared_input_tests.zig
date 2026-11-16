const std = @import("std");
const metal = @import("metal.zig");
const plan = @import("resident_plan.zig");

// Integration: append row_tiling_prepared.metal once after the affine source in
// Context.initMatched, then import this file from main.zig's test block. These
// tests use the production Context/ABI and are admission tests, not a tolerance
// check: any output-bit difference rejects the candidate.
const rows = 5;
const guards = 64;
const sentinel: u32 = 0x7fc12345;
const Affine = struct { scale: u32, bias: u32 };
const kernels = struct {
    const qmatvec = [_][:0]const u8{ "staged_qmatvec_t1", "staged_qmatvec_t2", "staged_qmatvec_t4", "staged_qmatvec_t8" };
    const qmatvec_prepared = [_][:0]const u8{ "staged_qmatvec_prepared_t1", "staged_qmatvec_prepared_t2", "staged_qmatvec_prepared_t4", "staged_qmatvec_prepared_t8" };
    const qkv = [_][:0]const u8{ "staged_qkv_t1", "staged_qkv_t2", "staged_qkv_t4", "staged_qkv_t8" };
    const qkv_prepared = [_][:0]const u8{ "staged_qkv_prepared_t1", "staged_qkv_prepared_t2", "staged_qkv_prepared_t4", "staged_qkv_prepared_t8" };
    const gate_up = [_][:0]const u8{ "staged_gate_up_t1", "staged_gate_up_t2", "staged_gate_up_t4", "staged_gate_up_t8" };
    const gate_up_prepared = [_][:0]const u8{ "staged_gate_up_prepared_t1", "staged_gate_up_prepared_t2", "staged_gate_up_prepared_t4", "staged_gate_up_prepared_t8" };
    const down = [_][:0]const u8{ "staged_down_t1", "staged_down_t2", "staged_down_t4", "staged_down_t8" };
    const down_prepared = [_][:0]const u8{ "staged_down_prepared_t1", "staged_down_prepared_t2", "staged_down_prepared_t4", "staged_down_prepared_t8" };
};

fn poison(buffer: metal.Buffer) void {
    @memset(buffer.structs(u32), sentinel);
}
fn expectPoison(buffer: metal.Buffer) !void {
    for (buffer.structs(u32)) |bits| try std.testing.expectEqual(sentinel, bits);
}
fn expectOutputs(reference: metal.Buffer, actual: metal.Buffer, count: usize) !void {
    for (reference.floats()[0..count]) |value| try std.testing.expect(std.math.isFinite(value));
    try std.testing.expectEqualSlices(u32, reference.structs(u32), actual.structs(u32));
    for (actual.structs(u32)[count..]) |bits| try std.testing.expectEqual(sentinel, bits);
}
fn expectBits(expected: f32, actual: f32) !void {
    try std.testing.expectEqual(@as(u32, @bitCast(expected)), @as(u32, @bitCast(actual)));
}
fn fillRaw(buffer: metal.Buffer) void {
    for (buffer.floats(), 0..) |*value, col|
        value.* = @as(f32, @floatFromInt((col * 5 + col / 19) % 47)) / 29 - 0.8;
}

// Explicit scalar FP32 order, including the left-associated quartet addition
// before input_sum += quartet. Do not replace this with a sum/reduction helper.
fn expectPrepared(raw: metal.Buffer, prepared: metal.Buffer, cols: u32, vectors: u32) !void {
    @setFloatMode(.strict);
    const stride: usize = cols + cols / 16;
    for (0..vectors) |vector| {
        for (0..cols / 16) |segment| {
            var input_sum: f32 = 0;
            var j: usize = 0;
            while (j < 16) : (j += 4) {
                const base = vector * cols + segment * 16 + j;
                const a = raw.floats()[base];
                const b = raw.floats()[base + 1];
                const c = raw.floats()[base + 2];
                const d = raw.floats()[base + 3];
                input_sum += ((a + b) + c) + d;
                const out = vector * stride + segment * 16 + j;
                try expectBits(a, prepared.floats()[out]);
                try expectBits(b / @as(f32, 16), prepared.floats()[out + 1]);
                try expectBits(c / @as(f32, 256), prepared.floats()[out + 2]);
                try expectBits(d / @as(f32, 4096), prepared.floats()[out + 3]);
            }
            try expectBits(input_sum, prepared.floats()[vector * stride + cols + segment]);
        }
    }
    for (prepared.structs(u32)[vectors * stride ..]) |bits| try std.testing.expectEqual(sentinel, bits);
}
fn prepare(ctx: metal.Context, raw: metal.Buffer, cols: u32, vectors: u32) !metal.Buffer {
    const out = try ctx.alloc(@as(usize, vectors) * (cols + cols / 16) + guards);
    errdefer out.deinit();
    poison(out);
    // Overlaunch exercises the segment bound as well as vector stride.
    const threads = @as(usize, vectors) * (cols / 16) + 32;
    try ctx.dispatch("prepare_quant_input", &.{ raw, out }, .{ .n = cols, .m = vectors }, threads, 32);
    try ctx.finish();
    try expectPrepared(raw, out, cols, vectors);
    return out;
}

test "prepared input caches exactly the scalar quartet order and scaled bits" {
    const ctx = try metal.Context.initMatched(true);
    defer ctx.deinit();
    for ([_]u32{ 64, 512, 1024, 2048, 1088 }) |cols| {
        const raw = try ctx.alloc(3 * cols);
        defer raw.deinit();
        fillRaw(raw);
        for (0..3) |vector| {
            // Cancellation distinguishes a quartet sum from sequentially
            // accumulating every element into input_sum. Normal exponents and
            // signed zeros also exercise materialized division/store bits.
            const values = [_]f32{ 16777216, 1, -16777216, 0.5, -0.0, 0.0, -0.0, 0.0, 0x1.000002p-30, -0x1.fffffep+20, 0x1.000002p+12, -0x1.000002p-12, 1, -2, 3, -4 };
            @memcpy(raw.floats()[vector * cols ..][0..16], &values);
        }
        const prepared = try prepare(ctx, raw, cols, 3);
        defer prepared.deinit();
    }
}

test "prepared INT4 g64 matvec QKV gate-up and down require exact native bits" {
    const ctx = try metal.Context.initMatched(true);
    defer ctx.deinit();
    for ([_]u32{ 512, 1024, 2048, 64, 1088 }) |cols| {
        for ([_]Affine{ .{ .scale = 0, .bias = 0 }, .{ .scale = 1, .bias = 1 }, .{ .scale = 2, .bias = 2 }, .{ .scale = 2, .bias = 0 }, .{ .scale = 1, .bias = 2 } }) |affine| {
            var fixture = try Fixture.init(ctx, cols, affine);
            defer fixture.deinit();
            const prepared = try prepare(ctx, fixture.x, cols, 1);
            defer prepared.deinit();
            for ([_]u32{ 3, 4 }) |storage_dtype| {
                for (&fixture.matrices) |*matrix| matrix.dtype = storage_dtype;
                try checkMatvecQkv(ctx, fixture, prepared);
                try checkRoutes(ctx, fixture, prepared);
            }
        }
    }
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
    affine: Affine,
    quantized: metal.Buffer,
    scales: metal.Buffer,
    biases: metal.Buffer,
    x: metal.Buffer,
    matrices: [3]metal.Matrix,

    fn init(ctx: metal.Context, cols: u32, affine: Affine) !Fixture {
        const elements: usize = rows * cols;
        const groups = elements / 64;
        const quantized = try ctx.alloc(1 + 3 * elements / 8);
        errdefer quantized.deinit();
        const scales = try ctx.alloc(1 + 3 * groups);
        errdefer scales.deinit();
        const biases = try ctx.alloc(2 + 3 * groups);
        errdefer biases.deinit();
        const x = try ctx.alloc(cols);
        errdefer x.deinit();
        @memset(quantized.structs(u8), 0);
        poison(scales);
        poison(biases);
        fillRaw(x);
        var matrices: [3]metal.Matrix = undefined;
        for (0..3) |slot| {
            for (0..elements) |i| {
                const index = slot * elements + i;
                const code: u8 = @intCast((i * 7 + slot * 11 + i / cols) % 16);
                quantized.structs(u8)[4 + index / 2] |= code << @as(u3, @intCast(index % 2 * 4));
            }
            // Nonzero byte offsets, independently sized FP32/FP16/BF16 loads.
            const scale_base = 4 + slot * groups * 4;
            const bias_base = 8 + slot * groups * 4;
            for (0..groups) |g| {
                writeFloat(scales, scale_base / @as(usize, if (affine.scale == 0) 4 else 2) + g, affine.scale, @as(f32, @floatFromInt((g * 3 + slot * 5) % 17 + 1)) / 2048);
                writeFloat(biases, bias_base / @as(usize, if (affine.bias == 0) 4 else 2) + g, affine.bias, @as(f32, @floatFromInt((g + slot * 3) % 19)) / 512 - 0.015625);
            }
            matrices[slot] = .{
                .weight = quantized.address() + 4 + slot * elements / 2,
                .scales = scales.address() + scale_base,
                .biases = biases.address() + bias_base,
                .rows = rows,
                .cols = cols,
                .dtype = 3,
                .bits = 4,
                .group_size = 64,
                .scale_dtype = affine.scale,
                .bias_dtype = affine.bias,
                .reserved = 1,
            };
        }
        return .{ .cols = cols, .affine = affine, .quantized = quantized, .scales = scales, .biases = biases, .x = x, .matrices = matrices };
    }
    fn deinit(self: Fixture) void {
        self.x.deinit();
        self.biases.deinit();
        self.scales.deinit();
        self.quantized.deinit();
    }
    fn resident(self: Fixture, ctx: metal.Context) !void {
        try ctx.resident(self.quantized);
        try ctx.resident(self.scales);
        try ctx.resident(self.biases);
    }
    fn matvecParams(self: Fixture) metal.Params {
        return .{ .offset = 4, .scale_offset = 4, .bias_offset = 8, .n = self.cols, .m = rows, .k = 64, .bits = 4, .dtype = self.matrices[0].dtype, .scale_dtype = self.affine.scale, .bias_dtype = self.affine.bias };
    }
};

fn checkMatvecQkv(ctx: metal.Context, fixture: Fixture, prepared: metal.Buffer) !void {
    const descriptor = try ctx.alloc(3 * @sizeOf(metal.Matrix) / 4);
    defer descriptor.deinit();
    @memcpy(descriptor.structs(metal.Matrix), &fixture.matrices);
    descriptor.structs(metal.Matrix)[1].rows = 3;
    descriptor.structs(metal.Matrix)[2].rows = 1;
    var reference: [3]metal.Buffer = undefined;
    var actual: [3]metal.Buffer = undefined;
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
        poison(reference[0]);
        poison(actual[0]);
        const threads = ((rows + tile - 1) / tile + 1) * 32;
        const p = fixture.matvecParams();
        try ctx.dispatch(kernels.qmatvec[ti], &.{ fixture.quantized, fixture.scales, fixture.biases, fixture.x, reference[0] }, p, threads, 32);
        try ctx.dispatch(kernels.qmatvec_prepared[ti], &.{ fixture.quantized, fixture.scales, fixture.biases, prepared, actual[0] }, p, threads, 32);
        try ctx.finish();
        try expectOutputs(reference[0], actual[0], rows);
        for (reference) |buffer| poison(buffer);
        for (actual) |buffer| poison(buffer);
        const tiles = (rows + tile - 1) / tile + (3 + tile - 1) / tile + 1;
        try fixture.resident(ctx);
        try ctx.dispatch(kernels.qkv[ti], &.{ descriptor, fixture.x, reference[0], reference[1], reference[2] }, .{}, (tiles + 1) * 32, 32);
        try ctx.dispatch(kernels.qkv_prepared[ti], &.{ descriptor, prepared, actual[0], actual[1], actual[2] }, .{}, (tiles + 1) * 32, 32);
        try ctx.finish();
        for ([_]usize{ 5, 3, 1 }, 0..) |count, i| try expectOutputs(reference[i], actual[i], count);
    }
}

fn makeExperts(ctx: metal.Context, fixture: Fixture) !metal.Buffer {
    const experts = try ctx.alloc(3 * @sizeOf(metal.MLP) / 4);
    for (experts.structs(metal.MLP), 0..) |*expert, i|
        expert.* = .{ .gate = fixture.matrices[i], .up = fixture.matrices[(i + 1) % 3], .down = fixture.matrices[(i + 2) % 3] };
    return experts;
}
fn checkRoutes(ctx: metal.Context, fixture: Fixture, prepared_x: metal.Buffer) !void {
    const experts = try makeExperts(ctx, fixture);
    defer experts.deinit();
    // Down widths need not equal one another or the raw activation stride.
    // The underlying allocation is large enough for each smaller descriptor;
    // native/prepared see the same packed rows and affine groups.
    experts.structs(metal.MLP)[1].down.cols = if (fixture.cols > 64) fixture.cols - 64 else 64;
    experts.structs(metal.MLP)[2].down.cols = @max(64, (fixture.cols / 128) * 64);
    const down_stride = fixture.cols + 64;
    const ids = try ctx.alloc(2);
    defer ids.deinit();
    @memcpy(ids.structs(u32), &[_]u32{ 2, 0 });
    const weights = try ctx.alloc(3);
    defer weights.deinit();
    @memcpy(weights.floats(), &[_]f32{ 0.17, 0.31, std.math.nan(f32) });
    const gate_stride = rows + 16;
    const gate_reference = try ctx.alloc(3 * gate_stride + guards);
    defer gate_reference.deinit();
    const gate_actual = try ctx.alloc(3 * gate_stride + guards);
    defer gate_actual.deinit();
    const activation = try ctx.alloc(3 * down_stride);
    defer activation.deinit();
    fillRaw(activation);
    for (0..3) |route| {
        // Deliberately stale finite padding beyond even the largest down.
        // Shorter downs also ignore the nonzero raw values before this tail.
        for (fixture.cols..down_stride) |col|
            activation.floats()[route * down_stride + col] = 0x1.abcdep+60;
    }
    const prepared_activation = try prepare(ctx, activation, down_stride, 3);
    defer prepared_activation.deinit();
    const down_reference = try ctx.alloc(rows + guards);
    defer down_reference.deinit();
    const down_actual = try ctx.alloc(rows + guards);
    defer down_actual.deinit();
    // Shared-only and nontrivial selected-expert order. Shared weight must be
    // implicit 1.0, never the poisoned trailing explicit mixture weight.
    for ([_]u32{ 0, 2 }) |routes| {
        for (plan.row_tiles, 0..) |tile, ti| {
            poison(gate_reference);
            poison(gate_actual);
            const gate_p: metal.Params = .{ .n = routes, .m = 1, .k = gate_stride };
            const gate_threads = ((routes + 1) * ((gate_stride + tile - 1) / tile) + 1) * 32;
            try fixture.resident(ctx);
            try ctx.dispatch(kernels.gate_up[ti], &.{ experts, fixture.x, ids, gate_reference }, gate_p, gate_threads, 32);
            try ctx.dispatch(kernels.gate_up_prepared[ti], &.{ experts, prepared_x, ids, gate_actual }, gate_p, gate_threads, 32);
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
            try ctx.dispatch(kernels.down_prepared[ti], &.{ experts, prepared_activation, ids, weights, down_actual }, down_p, down_threads, 32);
            try ctx.finish();
            try expectOutputs(down_reference, down_actual, rows);
        }
    }
}

const Invalid = enum { unreserved, bits, group, dtype, columns, empty, scale_dtype, bias_dtype };
fn invalidate(matrix: *metal.Matrix, reason: Invalid) void {
    switch (reason) {
        .unreserved => matrix.reserved = 0,
        .bits => matrix.bits = 8,
        .group => matrix.group_size = 32,
        .dtype => matrix.dtype = 2,
        .columns => matrix.cols = 63,
        .empty => matrix.cols = 0,
        .scale_dtype => matrix.scale_dtype = 3,
        .bias_dtype => matrix.bias_dtype = 3,
    }
    // A raw-helper fallback would dereference these null indirect resources.
    matrix.weight = 0;
    matrix.scales = 0;
    matrix.biases = 0;
}

test "prepared entrypoints reject unsupported descriptors without touching outputs" {
    const ctx = try metal.Context.initMatched(true);
    defer ctx.deinit();
    const fixture = try Fixture.init(ctx, 512, .{ .scale = 2, .bias = 2 });
    defer fixture.deinit();
    const prepared = try prepare(ctx, fixture.x, fixture.cols, 1);
    defer prepared.deinit();
    const descriptors = try ctx.alloc(3 * @sizeOf(metal.Matrix) / 4);
    defer descriptors.deinit();
    const experts = try makeExperts(ctx, fixture);
    defer experts.deinit();
    const ids = try ctx.alloc(1);
    defer ids.deinit();
    ids.structs(u32)[0] = 0;
    const weights = try ctx.alloc(1);
    defer weights.deinit();
    weights.floats()[0] = 0.3;
    const output = try ctx.alloc(3 * rows + guards);
    defer output.deinit();
    poison(output);
    for (plan.row_tiles, 0..) |tile, ti| {
        const threads = ((rows + tile - 1) / tile + 1) * 32;
        for (std.enums.values(Invalid)) |reason| {
            @memcpy(descriptors.structs(metal.Matrix), &fixture.matrices);
            invalidate(&descriptors.structs(metal.Matrix)[1], reason);
            try ctx.dispatch(kernels.qkv_prepared[ti], &.{ descriptors, prepared, output, output, output }, .{}, 3 * threads, 32);
            try ctx.finish();
            try expectPoison(output);
            experts.structs(metal.MLP)[0] = .{ .gate = fixture.matrices[0], .up = fixture.matrices[1], .down = fixture.matrices[2] };
            invalidate(&experts.structs(metal.MLP)[0].up, reason);
            try ctx.dispatch(kernels.gate_up_prepared[ti], &.{ experts, prepared, ids, output }, .{ .n = 0, .m = 0, .k = rows }, threads, 32);
            try ctx.finish();
            try expectPoison(output);
            // Invalid second/shared route must reject before reading even the
            // valid first route's activation or modifying any residual bits.
            experts.structs(metal.MLP)[0].down = fixture.matrices[0];
            experts.structs(metal.MLP)[1].down = fixture.matrices[1];
            invalidate(&experts.structs(metal.MLP)[1].down, reason);
            try fixture.resident(ctx);
            try ctx.dispatch(kernels.down_prepared[ti], &.{ experts, prepared, ids, weights, output }, .{ .n = 1, .m = 1, .k = 512, .dim = rows }, threads, 32);
            try ctx.finish();
            try expectPoison(output);
        }
        for (0..7) |reason| {
            var p = fixture.matvecParams();
            switch (reason) {
                0 => p.n = 63,
                1 => p.n = 0,
                2 => p.k = 32,
                3 => p.bits = 8,
                4 => p.dtype = 2,
                5 => p.scale_dtype = 3,
                6 => p.bias_dtype = 3,
                else => unreachable,
            }
            try ctx.dispatch(kernels.qmatvec_prepared[ti], &.{ fixture.quantized, fixture.scales, fixture.biases, prepared, output }, p, threads, 32);
            try ctx.finish();
            try expectPoison(output);
        }
    }
    for ([_]metal.Params{ .{ .n = 0, .m = 1 }, .{ .n = 63, .m = 1 }, .{ .n = 512, .m = 0 } }) |p| {
        try ctx.dispatch("prepare_quant_input", &.{ fixture.x, output }, p, 32, 32);
        try ctx.finish();
        try expectPoison(output);
    }
}
