const std = @import("std");
const metal = @import("metal.zig");
const plan = @import("resident_plan.zig");

// Root integration: embed row_tiling_prepared_fused.metal after the prepared
// helpers and import this file in main.zig. These are exact-bit admission tests,
// not tolerance checks; all GPU execution belongs to the root verifier.
const guards = 64;
const sentinel: u32 = 0x7fc12345;
const Affine = struct { scale: u32, bias: u32 };
const original = [_][:0]const u8{ "staged_gate_up_prepared_t1", "staged_gate_up_prepared_t2", "staged_gate_up_prepared_t4", "staged_gate_up_prepared_t8" };
const fused = [_][:0]const u8{ "staged_gate_up_prepared_fused_t1", "staged_gate_up_prepared_fused_t2", "staged_gate_up_prepared_fused_t4", "staged_gate_up_prepared_fused_t8" };

fn poison(buffer: metal.Buffer) void {
    @memset(buffer.structs(u32), sentinel);
}
fn expectPoison(buffer: metal.Buffer) !void {
    for (buffer.structs(u32)) |bits| try std.testing.expectEqual(sentinel, bits);
}
fn expectFiniteBits(reference: metal.Buffer, actual: metal.Buffer, index: usize) !void {
    try std.testing.expect(std.math.isFinite(reference.floats()[index]));
    try std.testing.expect(std.math.isFinite(actual.floats()[index]));
    try std.testing.expectEqual(reference.structs(u32)[index], actual.structs(u32)[index]);
}
fn writeFloat(buffer: metal.Buffer, byte_offset: usize, dtype: u32, value: f32) void {
    switch (dtype) {
        0 => buffer.floats()[byte_offset / 4] = value,
        1 => {
            const rounded: f16 = @floatCast(value);
            buffer.structs(u16)[byte_offset / 2] = @bitCast(rounded);
        },
        2 => {
            const bits: u32 = @bitCast(value);
            buffer.structs(u16)[byte_offset / 2] = @truncate(bits >> 16);
        },
        else => unreachable,
    }
}

const Fixture = struct {
    rows: u32,
    cols: u32,
    quantized: metal.Buffer,
    scales: metal.Buffer,
    biases: metal.Buffer,
    norm: metal.Buffer,
    prepared_norm: metal.Buffer,
    experts: metal.Buffer,
    ids: metal.Buffer,

    fn init(ctx: metal.Context, rows: u32, cols: u32, affine: Affine) !Fixture {
        const elements: usize = @as(usize, rows) * cols;
        const groups = elements / 64;
        const quantized = try ctx.alloc(1 + 6 * elements / 8);
        errdefer quantized.deinit();
        const scales = try ctx.alloc(1 + 6 * groups);
        errdefer scales.deinit();
        const biases = try ctx.alloc(2 + 6 * groups);
        errdefer biases.deinit();
        const norm = try ctx.alloc(cols);
        errdefer norm.deinit();
        const prepared_norm = try ctx.alloc(cols + cols / 16);
        errdefer prepared_norm.deinit();
        const experts = try ctx.alloc(3 * @sizeOf(metal.MLP) / 4);
        errdefer experts.deinit();
        const ids = try ctx.alloc(2);
        errdefer ids.deinit();
        @memcpy(ids.structs(u32), &[_]u32{ 2, 0 });
        @memset(quantized.structs(u8), 0);
        poison(scales);
        poison(biases);
        for (norm.floats(), 0..) |*value, col|
            value.* = @as(f32, @floatFromInt((col * 5 + col / 19) % 47)) / 29 - 0.8;
        // Unlike a zero-only fixture, cancellation and signs can distinguish
        // arithmetic reassociation. Values remain finite through gate/up.
        @memcpy(norm.floats()[0..16], &[_]f32{ 16777216, 1, -16777216, 0.5, -0.0, 0.0, -0.0, 0.0, 0x1.000002p-30, -0x1.fffffep+20, 0x1.000002p+12, -0x1.000002p-12, 1, -2, 3, -4 });
        var matrices: [6]metal.Matrix = undefined;
        for (0..6) |slot| {
            for (0..elements) |i| {
                const index = slot * elements + i;
                const code: u8 = @intCast((i * 7 + slot * 11 + i / cols) % 16);
                quantized.structs(u8)[4 + index / 2] |= code << @as(u3, @intCast(index % 2 * 4));
            }
            // Separate projections, nonzero byte offsets and independently
            // sized FP32/FP16/BF16 affine resources, like prepared_input_tests.
            const scale_base = 4 + slot * groups * 4;
            const bias_base = 8 + slot * groups * 4;
            for (0..groups) |g| {
                writeFloat(scales, scale_base + g * @as(usize, if (affine.scale == 0) 4 else 2), affine.scale, @as(f32, @floatFromInt((g * 3 + slot * 5) % 17 + 1)) / 2048);
                writeFloat(biases, bias_base + g * @as(usize, if (affine.bias == 0) 4 else 2), affine.bias, @as(f32, @floatFromInt((g + slot * 3) % 19)) / 512 - 0.015625);
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
        for (experts.structs(metal.MLP), 0..) |*expert, i| {
            expert.* = .{ .gate = matrices[2 * i], .up = matrices[2 * i + 1], .down = matrices[0] };
            // Shared expert has fewer real rows in the 128-row fixture. Both
            // 64 and 128 real widths coexist within one padded route launch.
            if (rows == 128 and i == 1) {
                expert.gate.rows = 64;
                expert.up.rows = 64;
            }
        }
        try ctx.dispatch("prepare_quant_input", &.{ norm, prepared_norm }, .{ .n = cols, .m = 1 }, cols / 16, 32);
        try ctx.finish();
        return .{ .rows = rows, .cols = cols, .quantized = quantized, .scales = scales, .biases = biases, .norm = norm, .prepared_norm = prepared_norm, .experts = experts, .ids = ids };
    }
    fn deinit(self: Fixture) void {
        self.ids.deinit();
        self.experts.deinit();
        self.prepared_norm.deinit();
        self.norm.deinit();
        self.biases.deinit();
        self.scales.deinit();
        self.quantized.deinit();
    }
    fn resident(self: Fixture, ctx: metal.Context) !void {
        try ctx.resident(self.quantized);
        try ctx.resident(self.scales);
        try ctx.resident(self.biases);
    }
};

fn checkRoutes(ctx: metal.Context, fixture: Fixture, routes: u32, raw_stride: u32) !void {
    const vectors = routes + 1;
    const prepared_stride = raw_stride + raw_stride / 16;
    const raw_reference = try ctx.alloc(vectors * raw_stride + guards);
    defer raw_reference.deinit();
    const raw_actual = try ctx.alloc(vectors * raw_stride + guards);
    defer raw_actual.deinit();
    const prepared_reference = try ctx.alloc(vectors * prepared_stride + guards);
    defer prepared_reference.deinit();
    const prepared_actual = try ctx.alloc(vectors * prepared_stride + guards);
    defer prepared_actual.deinit();
    for (plan.row_tiles, 0..) |tile, ti| {
        poison(raw_reference);
        poison(raw_actual);
        poison(prepared_reference);
        poison(prepared_actual);
        // The baseline prepare reads padding: give it finite stale values.
        // Fused raw/prepared padding instead stays poisoned and must never be
        // written, including when a complete padded 16-row group is launched.
        for (raw_reference.floats()[0 .. vectors * raw_stride], 0..) |*value, i|
            value.* = @as(f32, @floatFromInt(i % 29 + 1)) * 0x1.abcdep+20;
        const p: metal.Params = .{ .n = routes, .m = 1, .k = raw_stride };
        try fixture.resident(ctx);
        try ctx.dispatch(original[ti], &.{ fixture.experts, fixture.prepared_norm, fixture.ids, raw_reference }, p, (vectors * (raw_stride / tile) + 1) * 32, 32);
        try ctx.dispatch("prepare_quant_input", &.{ raw_reference, prepared_reference }, .{ .n = raw_stride, .m = vectors }, vectors * (raw_stride / 16) + 32, 32);
        const threads = 512 / tile;
        try ctx.dispatch(fused[ti], &.{ fixture.experts, fixture.prepared_norm, fixture.ids, raw_actual, prepared_actual }, p, (vectors * (raw_stride / 16) + 1) * threads, threads);
        try ctx.finish();
        for (0..vectors) |route| {
            const expert = if (route < routes) fixture.ids.structs(u32)[route] else 1;
            const real_rows = fixture.experts.structs(metal.MLP)[expert].gate.rows;
            for (0..raw_stride) |row| {
                const raw_index = route * raw_stride + row;
                const prepared_index = route * prepared_stride + row;
                if (row < real_rows) {
                    try expectFiniteBits(raw_reference, raw_actual, raw_index);
                    try expectFiniteBits(prepared_reference, prepared_actual, prepared_index);
                } else {
                    try std.testing.expectEqual(sentinel, raw_actual.structs(u32)[raw_index]);
                    try std.testing.expectEqual(sentinel, prepared_actual.structs(u32)[prepared_index]);
                }
            }
            for (0..raw_stride / 16) |segment| {
                const index = route * prepared_stride + raw_stride + segment;
                if (segment < real_rows / 16) {
                    try expectFiniteBits(prepared_reference, prepared_actual, index);
                } else try std.testing.expectEqual(sentinel, prepared_actual.structs(u32)[index]);
            }
        }
        for ([_]metal.Buffer{ raw_reference, raw_actual }) |buffer|
            for (buffer.structs(u32)[vectors * raw_stride ..]) |bits| try std.testing.expectEqual(sentinel, bits);
        for ([_]metal.Buffer{ prepared_reference, prepared_actual }) |buffer|
            for (buffer.structs(u32)[vectors * prepared_stride ..]) |bits| try std.testing.expectEqual(sentinel, bits);
    }
}

test "fused prepared gate-up preserves every real raw and prepared bit" {
    const ctx = try metal.Context.initMatched(true);
    defer ctx.deinit();
    for ([_]struct { rows: u32, cols: u32 }{ .{ .rows = 64, .cols = 512 }, .{ .rows = 128, .cols = 2048 } }) |shape| {
        for ([_]u32{ 0, 1, 2 }) |scale_dtype| {
            for ([_]u32{ 0, 1, 2 }) |bias_dtype| {
                const fixture = try Fixture.init(ctx, shape.rows, shape.cols, .{ .scale = scale_dtype, .bias = bias_dtype });
                defer fixture.deinit();
                for ([_]u32{ 3, 4 }) |storage_dtype| {
                    for (fixture.experts.structs(metal.MLP)) |*expert| {
                        expert.gate.dtype = storage_dtype;
                        expert.up.dtype = storage_dtype;
                    }
                    for ([_]u32{ 0, 2 }) |routes| {
                        for ([_]u32{ shape.rows + 64, shape.rows + 128 }) |raw_stride|
                            try checkRoutes(ctx, fixture, routes, raw_stride);
                    }
                }
            }
        }
    }
}

const Invalid = enum {
    unreserved,
    bits,
    group,
    dtype,
    columns,
    empty_columns,
    empty_rows,
    scale_dtype,
    bias_dtype,
    unequal_columns,
    unequal_rows,
    rows_exceed_stride,
    partial_segment,
};
fn invalidate(expert: *metal.MLP, reason: Invalid, invalidate_gate: bool) void {
    const matrix = if (invalidate_gate) &expert.gate else &expert.up;
    switch (reason) {
        .unreserved => matrix.reserved = 0,
        .bits => matrix.bits = 8,
        .group => matrix.group_size = 32,
        .dtype => matrix.dtype = 2,
        .columns => matrix.cols = 63,
        .empty_columns => matrix.cols = 0,
        .empty_rows => matrix.rows = 0,
        .scale_dtype => matrix.scale_dtype = 3,
        .bias_dtype => matrix.bias_dtype = 3,
        .unequal_columns => matrix.cols = 1024,
        .unequal_rows => matrix.rows = 32,
        .rows_exceed_stride => {
            expert.gate.rows = 256;
            expert.up.rows = 256;
        },
        .partial_segment => {
            expert.gate.rows = 63;
            expert.up.rows = 63;
        },
    }
    // Any attempted raw fallback or read before uniform validation dereferences
    // null indirect resources. The descriptor buffers themselves stay valid.
    expert.gate.weight = 0;
    expert.gate.scales = 0;
    expert.gate.biases = 0;
    expert.up.weight = 0;
    expert.up.scales = 0;
    expert.up.biases = 0;
}

test "fused prepared gate-up rejects invalid descriptors and group sizes uniformly" {
    const ctx = try metal.Context.initMatched(true);
    defer ctx.deinit();
    const fixture = try Fixture.init(ctx, 64, 512, .{ .scale = 2, .bias = 1 });
    defer fixture.deinit();
    const valid = fixture.experts.structs(metal.MLP)[1];
    const raw = try ctx.alloc(128 + guards);
    defer raw.deinit();
    const prepared = try ctx.alloc(128 + 128 / 16 + guards);
    defer prepared.deinit();
    poison(raw);
    poison(prepared);
    // A one-float input ensures no valid activation read is required on reject.
    const unread = try ctx.alloc(1);
    defer unread.deinit();
    poison(unread);
    for (plan.row_tiles, 0..) |tile, ti| {
        const threads = 512 / tile;
        for (std.enums.values(Invalid)) |reason| {
            for ([_]bool{ false, true }) |gate| {
                fixture.experts.structs(metal.MLP)[1] = valid;
                invalidate(&fixture.experts.structs(metal.MLP)[1], reason, gate);
                try ctx.dispatch(fused[ti], &.{ fixture.experts, unread, fixture.ids, raw, prepared }, .{ .n = 0, .m = 1, .k = 128 }, 8 * threads, threads);
                try ctx.finish();
                try expectPoison(raw);
                try expectPoison(prepared);
            }
        }
        // Geometry and raw-stride guards must precede even descriptor reads.
        // The tiny unread buffer stands in for experts and ids as well as norm.
        for ([_]usize{ threads / 2, threads - 1 }) |bad_group| {
            try ctx.dispatch(fused[ti], &.{ unread, unread, unread, raw, prepared }, .{ .n = 0, .m = 0, .k = 128 }, 2 * bad_group, bad_group);
            try ctx.finish();
            try expectPoison(raw);
            try expectPoison(prepared);
        }
        for ([_]u32{ 0, 63, 65, 127 }) |bad_stride| {
            try ctx.dispatch(fused[ti], &.{ unread, unread, unread, raw, prepared }, .{ .n = 0, .m = 0, .k = bad_stride }, threads, threads);
            try ctx.finish();
            try expectPoison(raw);
            try expectPoison(prepared);
        }
        // Partial final group: all complete and partial groups must reject the
        // non-multiple grid before reaching the shared-memory barrier.
        try ctx.dispatch(fused[ti], &.{ unread, unread, unread, raw, prepared }, .{ .n = 0, .m = 0, .k = 128 }, threads + 1, threads);
        try ctx.finish();
        try expectPoison(raw);
        try expectPoison(prepared);
    }
}
