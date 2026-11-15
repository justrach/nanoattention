const std = @import("std");
const metal = @import("metal.zig");
const plan = @import("resident_plan.zig");

const Kind = enum { int4_g64, fp32, fp16, bf16, mixed };
const guard_words = 128;
const sentinel: u32 = 0x7fc12345;
const expert_count: u32 = 8;
const shared_expert: u32 = 7;
const route_ids = [_]u32{ 5, 1, 4, 0, 3, 2 };

const Case = struct { rows: u32, cols: u32, routes: u32 };

test "parallel down is bitwise equal to staged down in matched MLX context" {
    try checkParallelDown(true);
}

test "parallel down preserves legacy residual-first bit patterns" {
    try checkParallelDown(false);
}

fn checkParallelDown(matched: bool) !void {
    const ctx = try metal.Context.initMatched(matched);
    defer ctx.deinit();
    const cases = [_]Case{
        // Partial rows for tiles 2/4/8, several selected experts, plus shared.
        .{ .rows = 5, .cols = 128, .routes = 3 },
        // Each lane accumulates multiple segments; activation stride includes
        // poisoned padding rather than being inferred from the matrix columns.
        .{ .rows = 17, .cols = 1088, .routes = 6 },
        // Shared-only routing, and a merge tail beyond one 128-thread group.
        .{ .rows = 129, .cols = 64, .routes = 0 },
    };
    for (cases) |case| {
        for ([_]Kind{ .int4_g64, .fp32, .fp16, .bf16, .mixed }) |kind|
            try checkCase(ctx, case, kind);
    }
}

fn expertKind(kind: Kind, expert: usize) Kind {
    return if (kind == .mixed) switch (expert % 4) {
        0 => .int4_g64,
        1 => .fp32,
        2 => .fp16,
        else => .bf16,
    } else kind;
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

fn checkCase(ctx: metal.Context, case: Case, kind: Kind) !void {
    const matrix_elements: usize = @as(usize, case.rows) * case.cols;
    const matrix_groups = matrix_elements / 64;
    const stride = case.cols + 16;
    const temporary_words = (case.routes + 1) * case.rows;
    const quantized = try ctx.alloc(1 + expert_count * matrix_elements / 8);
    defer quantized.deinit();
    // Give every expert a full FP32-sized slot, including mixed-dtype cases.
    const floating = try ctx.alloc(2 + expert_count * matrix_elements);
    defer floating.deinit();
    const scales = try ctx.alloc(1 + expert_count * matrix_groups);
    defer scales.deinit();
    const biases = try ctx.alloc(2 + expert_count * matrix_groups);
    defer biases.deinit();
    const experts = try ctx.alloc(expert_count * @sizeOf(metal.MLP) / 4);
    defer experts.deinit();
    const activation = try ctx.alloc((case.routes + 1) * stride);
    defer activation.deinit();
    const ids = try ctx.alloc(route_ids.len + 1);
    defer ids.deinit();
    const weights = try ctx.alloc(route_ids.len + 1);
    defer weights.deinit();
    const reference = try ctx.alloc(case.rows + guard_words);
    defer reference.deinit();
    const actual = try ctx.alloc(case.rows + guard_words);
    defer actual.deinit();
    const temporary = try ctx.alloc(temporary_words + guard_words);
    defer temporary.deinit();

    @memset(quantized.structs(u8), 0);
    @memset(floating.structs(u32), sentinel);
    @memset(scales.structs(u32), sentinel);
    @memset(biases.structs(u32), sentinel);
    for (0..expert_count) |expert| {
        for (0..matrix_groups) |g| {
            scales.floats()[1 + expert * matrix_groups + g] =
                @as(f32, @floatFromInt((g * 3 + expert * 5) % 17 + 1)) / 256;
            biases.floats()[2 + expert * matrix_groups + g] =
                @as(f32, @floatFromInt((g + expert * 3) % 19)) / 64 - 0.125;
        }
        const ek = expertKind(kind, expert);
        const dtype: u32 = switch (ek) {
            .int4_g64 => 3,
            .fp32 => 0,
            .fp16 => 1,
            .bf16 => 2,
            .mixed => unreachable,
        };
        for (0..matrix_elements) |i| {
            const index = expert * matrix_elements + i;
            const code: u8 = @intCast((i * 7 + expert * 11 + i / case.cols) % 16);
            quantized.structs(u8)[4 + index / 2] |= code << @as(u3, @intCast(index % 2 * 4));
            if (ek != .int4_g64) {
                const value = @as(f32, @floatFromInt((i * 13 + expert * 17 + i / case.cols) % 53)) / 37 - 0.7;
                const offset: usize = if (dtype == 0) 2 else 4;
                const slot_elements = matrix_elements * @as(usize, if (dtype == 0) 1 else 2);
                writeFloat(floating, offset + expert * slot_elements + i, dtype, value);
            }
        }
        const down: metal.Matrix = .{
            .weight = if (ek == .int4_g64)
                quantized.address() + 4 + expert * matrix_elements / 2
            else
                floating.address() + 8 + expert * matrix_elements * 4,
            .scales = scales.address() + 4 + expert * matrix_groups * 4,
            .biases = biases.address() + 8 + expert * matrix_groups * 4,
            .rows = case.rows,
            .cols = case.cols,
            .dtype = dtype,
            .bits = if (ek == .int4_g64) 4 else 0,
            .group_size = if (ek == .int4_g64) 64 else 0,
            .reserved = if (ek == .int4_g64) 1 else 0,
        };
        // Gate/up are deliberately irrelevant; only .down may be consumed.
        experts.structs(metal.MLP)[expert] = .{ .gate = down, .up = down, .down = down };
    }
    @memset(activation.structs(u32), sentinel);
    for (0..case.routes + 1) |route| {
        for (0..case.cols) |col|
            activation.floats()[route * stride + col] =
                @as(f32, @floatFromInt((col * 5 + route * 19) % 47)) / 29 - 0.8;
    }
    @memset(ids.structs(u32), 0xffffffff);
    @memcpy(ids.structs(u32)[0..case.routes], route_ids[0..case.routes]);
    @memset(weights.structs(u32), sentinel);
    // Non-unit, unequal weights make premature weighting observable. Poison
    // weights[n] to ensure the shared expert always uses the implicit 1.0.
    const route_weights = [_]f32{ 0.17, 0.31, 0.07, 0.23, 0.13, 0.09 };
    @memcpy(weights.floats()[0..case.routes], route_weights[0..case.routes]);
    const p: metal.Params = .{ .n = case.routes, .m = shared_expert, .k = stride, .dim = case.rows };

    for (plan.row_tiles) |tile| {
        const baseline: [:0]const u8 = switch (tile) {
            1 => "staged_down_t1",
            2 => "staged_down_t2",
            4 => "staged_down_t4",
            8 => "staged_down_t8",
            else => unreachable,
        };
        const producer: [:0]const u8 = switch (tile) {
            1 => "staged_down_parallel_t1",
            2 => "staged_down_parallel_t2",
            4 => "staged_down_parallel_t4",
            8 => "staged_down_parallel_t8",
            else => unreachable,
        };
        const row_tiles = (case.rows + tile - 1) / tile;
        initResidual(reference, case.rows);
        try makeResident(ctx, quantized, floating, scales, biases);
        try ctx.dispatch(baseline, &.{ experts, activation, ids, weights, reference }, p, row_tiles * 32, 128);
        try ctx.finish();
        for (reference.floats()[0..case.rows]) |value| try std.testing.expect(std.math.isFinite(value));
        try expectGuard(reference, case.rows);

        // Test both the exact launch contract and deliberate extra SIMD groups
        // / scalar threads. Guards must survive route and output-row overrun.
        for ([_]u32{ 0, 128 }) |extra| {
            initResidual(actual, case.rows);
            @memset(temporary.structs(u32), sentinel);
            try makeResident(ctx, quantized, floating, scales, biases);
            const before = ctx.stats();
            try ctx.dispatch(producer, &.{ experts, activation, ids, weights, temporary }, p, (case.routes + 1) * row_tiles * 32 + extra, 128);
            try ctx.dispatch("staged_down_merge", &.{ temporary, weights, actual }, p, case.rows + extra, 128);
            try ctx.finish();
            const after = ctx.stats();
            try std.testing.expectEqual(@as(u64, 1), after.submissions - before.submissions);
            try std.testing.expectEqual(@as(u64, 2), after.dispatches - before.dispatches);
            try std.testing.expectEqual(@as(u64, 1), after.barriers - before.barriers);
            // Compare every u32, including the untouched residual suffix.
            try std.testing.expectEqualSlices(u32, reference.structs(u32), actual.structs(u32));
            for (temporary.floats()[0..temporary_words]) |value| try std.testing.expect(std.math.isFinite(value));
            try expectGuard(temporary, temporary_words);
        }
    }
}

fn makeResident(ctx: metal.Context, quantized: metal.Buffer, floating: metal.Buffer, scales: metal.Buffer, biases: metal.Buffer) !void {
    // Matrix descriptors contain indirect GPU addresses, unlike direct bindings.
    try ctx.resident(quantized);
    try ctx.resident(floating);
    try ctx.resident(scales);
    try ctx.resident(biases);
}

fn initResidual(buffer: metal.Buffer, rows: usize) void {
    @memset(buffer.structs(u32), sentinel);
    for (buffer.floats()[0..rows], 0..) |*value, row|
        value.* = @as(f32, @floatFromInt((row * 7) % 31)) / 11 - 1.3;
}

fn expectGuard(buffer: metal.Buffer, begin: usize) !void {
    for (buffer.structs(u32)[begin..]) |bits| try std.testing.expectEqual(sentinel, bits);
}
