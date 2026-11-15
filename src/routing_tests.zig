const std = @import("std");
const metal = @import("metal.zig");
const sampling = @import("sampling.zig");
const plan = @import("resident_plan.zig");

test "register-cached routing preserves CPU and legacy selection, ties, weights and finite checks" {
    const ctx = try metal.Context.init();
    defer ctx.deinit();
    const logits = try ctx.alloc(plan.cached_router_experts);
    defer logits.deinit();
    const bias = try ctx.alloc(plan.cached_router_experts);
    defer bias.deinit();
    const ids = try ctx.alloc(256);
    defer ids.deinit();
    const weights = try ctx.alloc(256);
    defer weights.deinit();
    const legacy_ids = try ctx.alloc(256);
    defer legacy_ids.deinit();
    const legacy_weights = try ctx.alloc(256);
    defer legacy_weights.deinit();
    const flag = try ctx.alloc(1);
    defer flag.deinit();
    const Case = struct { count: u32, k: u32, ties: bool = false };
    for ([_]Case{ .{ .count = 4, .k = 2 }, .{ .count = 33, .k = 6 }, .{ .count = 256, .k = 6 }, .{ .count = 256, .k = 6, .ties = true }, .{ .count = 256, .k = 256, .ties = true } }) |case| {
        for (logits.floats(), bias.floats(), 0..) |*l, *b, i| {
            l.* = if (case.ties) 0 else @as(f32, @floatFromInt(i % 29)) / 7 - 2;
            b.* = if (case.ties) 0 else @as(f32, @floatFromInt((i * 13) % 17)) / 23;
        }
        var expected: [256]sampling.Route = undefined;
        try sampling.route(logits.floats()[0..case.count], bias.floats()[0..case.count], expected[0..case.k]);
        flag.structs(u32)[0] = 0;
        const p: metal.Params = .{ .m = case.count, .n = case.k };
        try ctx.dispatch("route_topk", &.{ logits, bias, legacy_ids, legacy_weights, flag }, p, 32, 32);
        try ctx.dispatch("route_topk_cached", &.{ logits, bias, ids, weights, flag }, p, 32, 32);
        try ctx.finish();
        try std.testing.expectEqual(@as(u32, 0), flag.structs(u32)[0]);
        for (expected[0..case.k], ids.structs(u32)[0..case.k], weights.floats()[0..case.k], legacy_ids.structs(u32)[0..case.k], legacy_weights.floats()[0..case.k]) |ref, id, w, old_id, old_w| {
            try std.testing.expectEqual(ref.id, id);
            try std.testing.expectEqual(old_id, id);
            try std.testing.expectApproxEqAbs(ref.weight, w, 1e-6);
            try std.testing.expectEqual(old_w, w);
        }
    }
    for ([_]bool{ false, true }) |bad_bias| {
        @memset(logits.floats(), 0);
        @memset(bias.floats(), 0);
        if (bad_bias) bias.floats()[255] = std.math.inf(f32) else logits.floats()[255] = std.math.nan(f32);
        flag.structs(u32)[0] = 0;
        try ctx.dispatch("route_topk_cached", &.{ logits, bias, ids, weights, flag }, .{ .m = 256, .n = 6 }, 32, 32);
        try ctx.finish();
        try std.testing.expectEqual(@as(u32, 1), flag.structs(u32)[0]);
    }
    flag.structs(u32)[0] = 0;
    try ctx.dispatch("route_topk_cached", &.{ logits, bias, ids, weights, flag }, .{ .m = 257, .n = 6 }, 32, 32);
    try ctx.finish();
    try std.testing.expectEqual(@as(u32, 1), flag.structs(u32)[0]);
}
