const std = @import("std");
const metal = @import("metal.zig");

// Root integration appends normalization_prepared.metal after both the aligned
// normalization and prepared-input sources, then imports this test module.
// Use the production library and math mode, not a separately compiled fixture.
const hidden = 2048;
const prepared_len = hidden + hidden / 16;
const guards = 32;
const sentinel: u32 = 0x7fc12345;

fn poison(buffer: metal.Buffer) void {
    @memset(buffer.structs(u32), sentinel);
}

fn expectGuard(bits: []const u32) !void {
    for (bits) |value| try std.testing.expectEqual(sentinel, value);
}

fn expectBits(reference: metal.Buffer, actual: metal.Buffer, count: usize) !void {
    // Integer comparisons include signed zero and every prepared segment sum;
    // approximate float equality is not an admission criterion here.
    try std.testing.expectEqualSlices(u32, reference.structs(u32), actual.structs(u32));
    try expectGuard(reference.structs(u32)[count..]);
    try expectGuard(actual.structs(u32)[count..]);
    for (actual.floats()[0..count]) |value| try std.testing.expect(std.math.isFinite(value));
}

fn expectInput(reference: []const f32, actual: metal.Buffer) !void {
    for (reference, actual.structs(u32)[0..reference.len]) |value, bits|
        try std.testing.expectEqual(@as(u32, @bitCast(value)), bits);
    try expectGuard(actual.structs(u32)[reference.len..]);
}

fn normWeights(buffer: metal.Buffer, dtype: u32, offset: usize) void {
    @memset(buffer.structs(u8), 0xa5);
    for (0..hidden) |i| {
        const value: f32 = @as(f32, @floatFromInt(i % 29 + 1)) / 19 * (if (i % 3 == 0) @as(f32, -1) else 1);
        switch (dtype) {
            0 => buffer.floats()[offset / 4 + i] = value,
            1 => {
                const rounded: f16 = @floatCast(value);
                buffer.structs(u16)[offset / 2 + i] = @bitCast(rounded);
            },
            2 => buffer.structs(u16)[offset / 2 + i] = @intCast(@as(u32, @bitCast(value)) >> 16),
            else => unreachable,
        }
    }
}

fn fillInput(values: []f32, pattern: usize, seed: usize) void {
    const cancellation = [_]f32{ 16777216, 1, -16777216, 0.5, -0.0, 0.0, -0.0, 0.0, 0x1.000002p-30, -0x1.fffffep+20, 0x1.000002p+12, -0x1.000002p-12, 1, -2, 3, -4 };
    for (values, 0..) |*value, i| {
        value.* = switch (pattern) {
            0 => @as(f32, @floatFromInt((i * 7 + seed) % 61)) / 29 - 1.125,
            // Cancellation, varying exponents and signed zeros span every
            // segment and SIMD group rather than only the first quartet.
            1 => cancellation[(i + seed) % cancellation.len],
            2 => if ((i + seed) % 2 == 0) @as(f32, -0.0) else 0.0,
            else => unreachable,
        };
    }
}

fn resetInput(buffer: metal.Buffer, values: []const f32) void {
    poison(buffer);
    @memcpy(buffer.floats()[0..values.len], values);
}

fn prepareReference(ctx: metal.Context, raw: metal.Buffer, prepared: metal.Buffer) !void {
    // Production prepare performs four left-associated quartet additions per
    // segment, with the exact /16, /256 and /4096 materialized output bits.
    try ctx.dispatch("prepare_quant_input", &.{ raw, prepared }, .{ .n = hidden, .m = 1 }, hidden / 16, 32);
}

test "prepared aligned RMSNorm matches original raw and all prepared bits, dtypes offsets and aliasing" {
    const ctx = try metal.Context.initMatched(true);
    defer ctx.deinit();
    const w = try ctx.alloc(hidden + guards);
    defer w.deinit();
    const x = try ctx.alloc(hidden + guards);
    defer x.deinit();
    const reference_x = try ctx.alloc(hidden + guards);
    defer reference_x.deinit();
    const raw = try ctx.alloc(hidden + guards);
    defer raw.deinit();
    const reference_raw = try ctx.alloc(hidden + guards);
    defer reference_raw.deinit();
    const prepared = try ctx.alloc(prepared_len + guards);
    defer prepared.deinit();
    const reference_prepared = try ctx.alloc(prepared_len + guards);
    defer reference_prepared.deinit();
    var input: [hidden]f32 = undefined;
    for (0..3) |dtype| {
        const offset: usize = if (dtype == 0) 12 else 6;
        normWeights(w, @intCast(dtype), offset);
        for (0..3) |pattern| {
            fillInput(&input, pattern, 0);
            for ([_]bool{ false, true }) |in_place| {
                resetInput(x, &input);
                resetInput(reference_x, &input);
                poison(raw);
                poison(reference_raw);
                poison(prepared);
                poison(reference_prepared);
                const destination = if (in_place) x else raw;
                const reference_destination = if (in_place) reference_x else reference_raw;
                const p: metal.Params = .{ .n = hidden, .m = 1, .offset = offset, .dtype = @intCast(dtype), .eps = if (pattern == 1) 3e-5 else 1e-6 };
                try ctx.dispatch("rmsnorm_aligned", &.{ w, reference_x, reference_destination }, p, 512, 512);
                try prepareReference(ctx, reference_destination, reference_prepared);
                try ctx.dispatch("rmsnorm_aligned_prepared", &.{ w, x, destination, prepared }, p, 512, 512);
                try ctx.finish();
                try expectBits(reference_destination, destination, hidden);
                try expectBits(reference_prepared, prepared, prepared_len);
                if (in_place) {
                    try expectGuard(raw.structs(u32));
                    try expectGuard(reference_raw.structs(u32));
                } else {
                    try expectInput(&input, x);
                    try expectInput(&input, reference_x);
                }
            }
        }
    }
}

test "prepared aligned residual RMSNorm matches original raw updated residual and every prepared bit" {
    const ctx = try metal.Context.initMatched(true);
    defer ctx.deinit();
    const w = try ctx.alloc(hidden + guards);
    defer w.deinit();
    const residual = try ctx.alloc(hidden + guards);
    defer residual.deinit();
    const reference_residual = try ctx.alloc(hidden + guards);
    defer reference_residual.deinit();
    const projection = try ctx.alloc(hidden + guards);
    defer projection.deinit();
    const raw = try ctx.alloc(hidden + guards);
    defer raw.deinit();
    const reference_raw = try ctx.alloc(hidden + guards);
    defer reference_raw.deinit();
    const prepared = try ctx.alloc(prepared_len + guards);
    defer prepared.deinit();
    const reference_prepared = try ctx.alloc(prepared_len + guards);
    defer reference_prepared.deinit();
    var input: [hidden]f32 = undefined;
    var projected: [hidden]f32 = undefined;
    var updated: [hidden]f32 = undefined;
    for (0..3) |dtype| {
        const offset: usize = if (dtype == 0) 8 else 2;
        normWeights(w, @intCast(dtype), offset);
        for (0..3) |pattern| {
            fillInput(&input, pattern, 0);
            fillInput(&projected, pattern, 5);
            for (&updated, input, projected) |*value, a, b| value.* = a + b;
            resetInput(residual, &input);
            resetInput(reference_residual, &input);
            resetInput(projection, &projected);
            poison(raw);
            poison(reference_raw);
            poison(prepared);
            poison(reference_prepared);
            const p: metal.Params = .{ .n = hidden, .m = 1, .offset = offset, .dtype = @intCast(dtype), .eps = if (pattern == 1) 3e-5 else 1e-6 };
            try ctx.dispatch("residual_norm_aligned", &.{ w, reference_residual, projection, reference_raw }, p, 512, 512);
            try prepareReference(ctx, reference_raw, reference_prepared);
            try ctx.dispatch("residual_norm_aligned_prepared", &.{ w, residual, projection, raw, prepared }, p, 512, 512);
            try ctx.finish();
            try expectBits(reference_raw, raw, hidden);
            try expectBits(reference_residual, residual, hidden);
            try expectInput(&updated, residual);
            try expectBits(reference_prepared, prepared, prepared_len);
            try expectInput(&projected, projection);
        }
    }
}

test "prepared aligned normalization rejects unsupported shapes without touching either output or residual" {
    const ctx = try metal.Context.initMatched(true);
    defer ctx.deinit();
    const w = try ctx.alloc(hidden + guards);
    defer w.deinit();
    const x = try ctx.alloc(hidden + guards);
    defer x.deinit();
    const projection = try ctx.alloc(hidden + guards);
    defer projection.deinit();
    const raw = try ctx.alloc(hidden + guards);
    defer raw.deinit();
    const prepared = try ctx.alloc(prepared_len + guards);
    defer prepared.deinit();
    var input: [hidden]f32 = undefined;
    var projected: [hidden]f32 = undefined;
    fillInput(&input, 0, 3);
    fillInput(&projected, 0, 19);
    normWeights(w, 0, 12);
    const Shape = struct { n: u32, group: usize };
    for ([_]Shape{
        .{ .n = 0, .group = 512 },
        .{ .n = 1024, .group = 512 },
        .{ .n = 2047, .group = 512 },
        .{ .n = 2049, .group = 512 },
        .{ .n = hidden, .group = 32 },
        .{ .n = hidden, .group = 256 },
    }) |shape| {
        for ([_]bool{ false, true }) |residual_norm| {
            resetInput(x, &input);
            resetInput(projection, &projected);
            poison(raw);
            poison(prepared);
            const p: metal.Params = .{ .n = shape.n, .m = 1, .dtype = 0, .offset = 12 };
            if (residual_norm) {
                try ctx.dispatch("residual_norm_aligned_prepared", &.{ w, x, projection, raw, prepared }, p, shape.group, shape.group);
            } else {
                try ctx.dispatch("rmsnorm_aligned_prepared", &.{ w, x, raw, prepared }, p, shape.group, shape.group);
            }
            try ctx.finish();
            try expectGuard(raw.structs(u32));
            try expectGuard(prepared.structs(u32));
            try expectInput(&input, x);
            try expectInput(&projected, projection);
        }
    }
}
