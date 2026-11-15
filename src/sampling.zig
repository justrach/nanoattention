const std = @import("std");

pub const Route = struct { id: u32, weight: f32 };

/// Bias influences expert selection only; mixture weights use unbiased sigmoid scores.
pub fn route(logits: []const f32, bias: ?[]const f32, selected: []Route) !void {
    if (selected.len == 0 or selected.len > logits.len) return error.InvalidTopK;
    if (bias) |b| if (b.len != logits.len) return error.InvalidRouterBias;
    for (logits) |x| if (!std.math.isFinite(x)) return error.NonFiniteLogits;
    if (bias) |b| for (b) |x| {
        if (!std.math.isFinite(x)) return error.NonFiniteLogits;
    };
    var total: f32 = 0;
    for (selected, 0..) |*out, k| {
        var best_score: f32 = -std.math.inf(f32);
        var best: u32 = 0;
        for (logits, 0..) |logit, i| {
            var used = false;
            for (selected[0..k]) |prev| if (prev.id == i) {
                used = true;
                break;
            };
            if (used) continue;
            const score = sigmoid(logit) + if (bias) |b| b[i] else @as(f32, 0);
            if (score > best_score) {
                best_score = score;
                best = @intCast(i);
            }
        }
        out.* = .{ .id = best, .weight = sigmoid(logits[best]) };
        total += out.weight;
    }
    for (selected) |*r| r.weight /= total + 1e-20;
}
fn sigmoid(x: f32) f32 {
    return 1 / (1 + @exp(-x));
}

pub fn greedy(logits: []const f32) !u32 {
    if (logits.len == 0) return error.EmptyLogits;

    const width = 16;
    const Values = @Vector(width, f32);
    const Indices = @Vector(width, u32);
    const offsets: Indices = .{ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 };
    var lane_values: Values = @splat(-std.math.inf(f32));
    var lane_indices: Indices = @splat(std.math.maxInt(u32));
    var invalid: @Vector(width, bool) = @splat(false);
    var i: usize = 0;
    while (logits.len - i >= width) : (i += width) {
        const values: Values = logits[i..][0..width].*;
        // Inspect every exponent, including losing lanes, without a branch or
        // horizontal reduction in the hot loop. No floating-point mode changes.
        const bits: Indices = @bitCast(values);
        const exponent_mask: Indices = @splat(0x7f800000);
        invalid |= (bits & exponent_mask) == exponent_mask;
        const better = values > lane_values;
        lane_values = @select(f32, better, values, lane_values);
        lane_indices = @select(u32, better, offsets + @as(Indices, @splat(@intCast(i))), lane_indices);
    }
    if (@reduce(.Or, invalid)) return error.NonFiniteLogits;

    var best_value: f32 = -std.math.inf(f32);
    var best: u32 = 0;
    if (i != 0) {
        best_value = @reduce(.Max, lane_values);
        // Lane order is not index order after multiple blocks. Select the
        // earliest absolute index among all equal maxima (including +/-0).
        best = @reduce(.Min, @select(u32, lane_values == @as(Values, @splat(best_value)), lane_indices, @as(Indices, @splat(std.math.maxInt(u32)))));
    }
    for (logits[i..], i..) |value, index| {
        if (!std.math.isFinite(value)) return error.NonFiniteLogits;
        if (value > best_value) {
            best_value = value;
            best = @intCast(index);
        }
    }
    return best;
}

// Deliberately retain the original scalar implementation only as a test oracle.
fn greedyReference(logits: []const f32) !u32 {
    if (logits.len == 0) return error.EmptyLogits;
    var best: u32 = 0;
    for (logits, 0..) |value, i| {
        if (!std.math.isFinite(value)) return error.NonFiniteLogits;
        if (value > logits[best]) best = @intCast(i);
    }
    return best;
}

test "greedy SIMD agrees with scalar across body and tail boundaries" {
    try std.testing.expectError(error.EmptyLogits, greedy(&.{}));
    var values: [97]f32 = undefined;
    var state: u32 = 0x12345678;
    for (1..values.len + 1) |len| {
        for (values[0..len]) |*value| {
            state = state *% 1664525 +% 1013904223;
            // Exercise the full finite exponent/sign range, not just integers.
            var bits = state;
            if (bits & 0x7f800000 == 0x7f800000) bits ^= 0x00800000;
            value.* = @bitCast(bits);
        }
        try std.testing.expectEqual(try greedyReference(values[0..len]), try greedy(values[0..len]));
        // Every location must be selectable, including every tail position.
        for (0..len) |winner| {
            @memset(values[0..len], -100);
            values[winner] = -1;
            try std.testing.expectEqual(@as(u32, @intCast(winner)), try greedy(values[0..len]));
        }
    }
}

test "greedy SIMD rejects every nonfinite body and tail lane" {
    var values: [65]f32 = undefined;
    const invalid = [_]f32{ std.math.nan(f32), -std.math.nan(f32), std.math.inf(f32), -std.math.inf(f32), @bitCast(@as(u32, 0x7f800001)), @bitCast(@as(u32, 0xff800001)) };
    for (invalid) |value| {
        for (1..values.len + 1) |len| {
            for (0..len) |index| {
                @memset(values[0..len], 10);
                values[index] = value;
                try std.testing.expectError(error.NonFiniteLogits, greedy(values[0..len]));
            }
        }
    }
}

test "greedy SIMD keeps earliest ties signed zeros and extreme finite values" {
    var values: [49]f32 = undefined;
    @memset(&values, -7);
    try std.testing.expectEqual(@as(u32, 0), try greedy(&values));
    // The earlier maximum has a higher lane number than later equal maxima.
    values[15] = -1;
    values[16] = -1;
    values[48] = -1;
    try std.testing.expectEqual(@as(u32, 15), try greedy(&values));
    values[0] = -0.0;
    values[16] = 0.0;
    values[48] = 0.0;
    try std.testing.expectEqual(@as(u32, 0), try greedy(&values));
    values[0] = 0.0;
    values[16] = -0.0;
    try std.testing.expectEqual(@as(u32, 0), try greedy(&values));
    @memset(&values, -std.math.floatMax(f32));
    try std.testing.expectEqual(@as(u32, 0), try greedy(&values));
    values[31] = std.math.floatMax(f32);
    values[32] = std.math.floatMax(f32);
    try std.testing.expectEqual(@as(u32, 31), try greedy(&values));
    @memset(&values, -std.math.floatTrueMin(f32));
    values[32] = std.math.floatTrueMin(f32);
    try std.testing.expectEqual(@as(u32, 32), try greedy(&values));
}

test "greedy SIMD preserves full vocabulary indices" {
    var values: [151936]f32 = @splat(-20);
    values[131071] = 5;
    values[131072] = 5;
    values[values.len - 1] = 5;
    try std.testing.expectEqual(try greedyReference(&values), try greedy(&values));
    try std.testing.expectEqual(@as(u32, 131071), try greedy(&values));
    values[values.len - 1] = 6;
    try std.testing.expectEqual(@as(u32, values.len - 1), try greedy(&values));
}

test "sigmoid top-k uses bias only for selection and renormalizes" {
    var routes: [2]Route = undefined;
    try route(&.{ 0, 2, -2 }, &.{ 0, 0, 2 }, &routes);
    try std.testing.expectEqual(@as(u32, 2), routes[0].id);
    try std.testing.expectEqual(@as(u32, 1), routes[1].id);
    try std.testing.expectApproxEqAbs(@as(f32, 1), routes[0].weight + routes[1].weight, 1e-6);
    try std.testing.expectApproxEqAbs(sigmoid(-2) / (sigmoid(-2) + sigmoid(2)), routes[0].weight, 1e-6);
    try std.testing.expectError(error.NonFiniteLogits, route(&.{std.math.nan(f32)}, null, routes[0..1]));
}
