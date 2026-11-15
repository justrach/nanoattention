const std = @import("std");
const plan = @import("plan");

test "row tiles expose precisely the certified admissible launch parameters" {
    try std.testing.expectEqualSlices(u32, &.{ 1, 2, 4, 8 }, &plan.row_tiles);
    for (0..257) |tile| {
        const expected = tile == 1 or tile == 2 or tile == 4 or tile == 8;
        try std.testing.expectEqual(expected, plan.validRowTile(@intCast(tile)));
    }
    try std.testing.expect(!plan.validRowTile(std.math.maxInt(u32)));
}

test "independent QKV intervals decode every local tile including empty projections" {
    for (0..10) |q_count| {
        for (0..10) |k_count| {
            for (0..10) |v_count| {
                const q: u32 = @intCast(q_count);
                const k: u32 = @intCast(k_count);
                const counts = [_]u32{ q, k, @intCast(v_count) };
                var prefix: u32 = 0;
                for (counts, 0..) |count, projection| {
                    for (0..count) |local| {
                        const global = prefix + @as(u32, @intCast(local));
                        try std.testing.expectEqual(@as(u32, @intCast(projection)), plan.stagedQkvProjection(global, q, k));
                        try std.testing.expectEqual(@as(u32, @intCast(local)), plan.stagedQkvLocalTile(global, q, k));
                    }
                    prefix += count;
                }
            }
        }
    }
}

test "QKV decoding preserves independently rounded partial row tiles" {
    for (plan.row_tiles) |width| {
        const rows = [_]u32{ 17, 3, 9 };
        var counts: [3]u32 = undefined;
        for (rows, 0..) |count, p| counts[p] = (count + width - 1) / width;
        const total = counts[0] + counts[1] + counts[2];
        var visits: [17 + 3 + 9]u32 = @splat(0);
        for (0..total) |global| {
            const p = plan.stagedQkvProjection(@intCast(global), counts[0], counts[1]);
            const local = plan.stagedQkvLocalTile(@intCast(global), counts[0], counts[1]);
            const offset: u32 = if (p == 0) 0 else if (p == 1) rows[0] else rows[0] + rows[1];
            for (0..width) |j| {
                const row = local * width + @as(u32, @intCast(j));
                if (row < rows[p]) visits[offset + row] += 1;
            }
        }
        for (visits) |count| try std.testing.expectEqual(@as(u32, 1), count);
    }
}

test "QKV helpers respect representable prefix sums at u32 boundary" {
    const max = std.math.maxInt(u32);
    const cases = [_][3]u32{
        .{ max, 0, 0 },     .{ 0, max, 0 },     .{ 0, 0, max },
        .{ max - 2, 1, 1 }, .{ 1, max - 2, 1 }, .{ 1, 1, max - 2 },
    };
    for (cases) |counts| {
        var prefix: u32 = 0;
        for (counts, 0..) |count, p| {
            if (count != 0) {
                for ([_]u32{ 0, count - 1 }) |local| {
                    const global = prefix + local;
                    try std.testing.expectEqual(@as(u32, @intCast(p)), plan.stagedQkvProjection(global, counts[0], counts[1]));
                    try std.testing.expectEqual(local, plan.stagedQkvLocalTile(global, counts[0], counts[1]));
                }
            }
            prefix += count;
        }
    }
}

test "dense split geometry covers guarded columns exactly once without u32 wrap" {
    try std.testing.expectEqual(@as(u32, 8), plan.dense_simds);
    try std.testing.expectEqual(@as(u32, 8192), plan.dense_max_cols);
    try std.testing.expectEqual(plan.threadgroup_size, plan.dense_simds * plan.router_lanes);
    const widths = [_]u32{ 0, 1, 31, 32, 33, 255, 256, 257, 8191, plan.dense_max_cols };
    for (widths) |n| {
        var visits: [plan.dense_max_cols]u8 = @splat(0);
        for (0..plan.dense_simds) |simd| {
            for (0..plan.router_lanes) |lane| {
                var col: u32 = @intCast(simd * plan.router_lanes + lane);
                try std.testing.expect(col < plan.threadgroup_size);
                var iteration: u32 = 0;
                while (col < n) {
                    try std.testing.expect(iteration < plan.dense_max_cols / plan.threadgroup_size);
                    try std.testing.expectEqual(@as(u32, @intCast(simd)), (col % plan.threadgroup_size) / plan.router_lanes);
                    try std.testing.expectEqual(@as(u32, @intCast(lane)), col % plan.router_lanes);
                    try std.testing.expectEqual(iteration, col / plan.threadgroup_size);
                    visits[col] += 1;
                    const advanced = @addWithOverflow(col, plan.threadgroup_size);
                    try std.testing.expectEqual(@as(u1, 0), advanced[1]);
                    col = advanced[0];
                    iteration += 1;
                }
            }
        }
        for (visits, 0..) |count, col| {
            try std.testing.expectEqual(@as(u8, if (col < n) 1 else 0), count);
        }
    }
}

test "all certified arrays expose the complete ordered schedule" {
    try std.testing.expect(plan.program(0) == null);
    try std.testing.expect(plan.program(50) == null);
    try std.testing.expect(plan.program(std.math.maxInt(u32)) == null);
    try std.testing.expectEqual(@as(u32, 256), plan.threadgroup_size);
    try std.testing.expectEqual(@as(u32, 32), plan.max_workers);
    try std.testing.expectEqual(@as(u32, 10000000), plan.spin_limit);
    try std.testing.expectEqual(@as(u32, 32), plan.router_lanes);
    try std.testing.expectEqual(@as(u32, 8), plan.router_slots);
    try std.testing.expectEqual(@as(u32, 256), plan.cached_router_experts);
    for (1..50) |count| {
        const instructions = plan.program(@intCast(count)).?;
        try std.testing.expectEqual(count * 10 + 3, instructions.len);
        try std.testing.expectEqual(@backingInt(plan.Op.embedding), instructions[0]);
        for (0..count) |layer| {
            for (2..12) |opcode| {
                const instruction = instructions[layer * 10 + opcode - 1];
                try std.testing.expectEqual(@as(u32, @intCast(layer)), instruction / 256);
                try std.testing.expectEqual(@as(u32, @intCast(opcode)), instruction % 256);
            }
        }
        try std.testing.expectEqual(@backingInt(plan.Op.final_norm), instructions[instructions.len - 2]);
        try std.testing.expectEqual(@backingInt(plan.Op.head), instructions[instructions.len - 1]);
    }
}
