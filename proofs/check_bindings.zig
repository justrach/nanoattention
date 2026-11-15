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

test "attention bindings admit precisely the finite chunk family and geometry" {
    try std.testing.expectEqualSlices(u32, &.{ 128, 256, 512, 1024 }, &plan.attention_chunks);
    for (0..2049) |c| {
        const expected = c == 128 or c == 256 or c == 512 or c == 1024;
        try std.testing.expectEqual(expected, plan.validAttentionChunk(@intCast(c)));
    }
    try std.testing.expect(!plan.validAttentionChunk(std.math.maxInt(u32)));
    try std.testing.expectEqual(@as(u32, 262144), plan.attention_max_context);
    try std.testing.expectEqual(@as(u32, 64), plan.attention_max_heads);
    try std.testing.expectEqual(@as(u32, 256), plan.attention_max_dim);
    try std.testing.expectEqual(@as(u32, 8), plan.attention_simds);
    try std.testing.expectEqual(@as(u32, 32), plan.attention_lanes);
    try std.testing.expectEqual(plan.threadgroup_size, plan.attention_simds * plan.attention_lanes);
}

test "attention chunk producers cover relative tokens once including partial chunks and sliding starts" {
    const lengths = [_]u32{ 0, 1, 7, 8, 9, 31, 32, 33, 127, 128, 129, 255, 256, 257, 511, 512, 513, 1023, 1024, 1025, 262143, 262144 };
    for (plan.attention_chunks) |width| {
        for (lengths) |n| {
            const chunks = (n + width - 1) / width;
            try std.testing.expect(chunks <= 2048);
            try std.testing.expectEqual(n == 0, chunks == 0);
            const start = plan.attention_max_context - n;
            var visits: [plan.attention_max_context]u8 = @splat(0);
            var sinks: u32 = 0;
            for (0..chunks) |chunk_index| {
                const chunk: u32 = @intCast(chunk_index);
                const begin = start + width * chunk;
                try std.testing.expect(begin + width <= plan.attention_max_context + 1024);
                for (0..plan.attention_simds) |simd_index| {
                    const simd: u32 = @intCast(simd_index);
                    if (chunk == 0 and simd == 0) sinks += 1;
                    var iteration: u32 = 0;
                    while (simd + plan.attention_simds * iteration < width) : (iteration += 1) {
                        const relative = width * chunk + simd + plan.attention_simds * iteration;
                        const advance = @addWithOverflow(start + relative, plan.attention_simds);
                        try std.testing.expectEqual(@as(u1, 0), advance[1]);
                        if (relative >= n) continue;
                        try std.testing.expect(start + relative < plan.attention_max_context);
                        try std.testing.expectEqual(chunk, relative / width);
                        try std.testing.expectEqual(simd, (relative % width) % plan.attention_simds);
                        try std.testing.expectEqual(iteration, (relative % width) / plan.attention_simds);
                        visits[relative] += 1;
                    }
                }
            }
            try std.testing.expectEqual(@as(u32, if (n == 0) 0 else 1), sinks);
            for (visits, 0..) |count, r| {
                try std.testing.expectEqual(@as(u8, if (r < n) 1 else 0), count);
            }
        }
    }
}

test "attention head-major state stride and maximal cache element indices fit u32" {
    const max_chunks = (plan.attention_max_context + plan.attention_chunks[0] - 1) / plan.attention_chunks[0];
    try std.testing.expectEqual(@as(u32, 2048), max_chunks);
    const state_limit: u32 = max_chunks * plan.attention_max_heads * (plan.attention_max_dim + 2);
    try std.testing.expectEqual(@as(u32, 33816576), state_limit);
    for ([_]u32{ 1, 2, 2047, max_chunks }) |chunks| {
        for ([_]u32{ 1, plan.attention_max_heads }) |heads| {
            for ([_]u32{ 0, 1, 255, plan.attention_max_dim }) |dim| {
                const stride = dim + 2;
                const allocation = heads * chunks * stride;
                try std.testing.expect(allocation <= state_limit);
                for (0..heads) |head_index| {
                    const head: u32 = @intCast(head_index);
                    for (0..chunks) |chunk_index| {
                        const chunk: u32 = @intCast(chunk_index);
                        const flat = head * chunks + chunk;
                        try std.testing.expectEqual(head, flat / chunks);
                        try std.testing.expectEqual(chunk, flat % chunks);
                        for ([_]u32{ 0, stride - 1 }) |field| {
                            try std.testing.expect(flat * stride + field < allocation);
                        }
                    }
                }
            }
        }
    }
    // Element indexing only: the maximal float buffer's byte offset needs u64.
    const max_cache: u64 = @as(u64, plan.attention_max_context) * plan.attention_max_heads * plan.attention_max_dim;
    try std.testing.expectEqual(@as(u64, 1) << 32, max_cache);
    for ([_]u32{ 1, 63, plan.attention_max_heads }) |heads| {
        for ([_]u32{ 1, 255, plan.attention_max_dim }) |dim| {
            const pos = plan.attention_max_context - 1;
            const offset: u32 = (pos * heads + (heads - 1)) * dim + (dim - 1);
            const expected = @as(u64, plan.attention_max_context) * heads * dim - 1;
            try std.testing.expectEqual(expected, @as(u64, offset));
        }
    }
}

test "MLX candidate exports exact geometry and adjacent-four component ownership" {
    try std.testing.expectEqual(@as(u32, 128), plan.mlx_attention_dim);
    try std.testing.expectEqual(@as(u32, 32), plan.mlx_attention_lanes);
    try std.testing.expectEqual(@as(u32, 4), plan.mlx_attention_elements_per_lane);
    try std.testing.expectEqual(@as(u32, 32), plan.mlx_attention_single_simds);
    try std.testing.expectEqual(@as(u32, 32), plan.mlx_attention_merge_simds);
    try std.testing.expectEqual(@as(u32, 32), plan.mlx_attention_max_gqa);
    try std.testing.expectEqual(@as(u32, 64), plan.mlx_attention_max_heads);
    try std.testing.expectEqual(@as(u32, 262144), plan.mlx_attention_max_context);
    try std.testing.expectEqual(@as(u32, 1024), plan.mlx_attention_max_blocks);
    try std.testing.expectEqual(@as(u32, 8519680), plan.mlx_attention_max_state_elements);
    try std.testing.expectEqual(plan.mlx_attention_max_state_elements, plan.mlx_attention_max_heads * plan.mlx_attention_max_blocks * plan.mlx_attention_state_stride);
    try std.testing.expectEqual(@as(u32, 130), plan.mlx_attention_state_stride);
    try std.testing.expectEqual(@as(u32, 128), plan.mlx_attention_max_field);
    try std.testing.expectEqual(@as(u32, 129), plan.mlx_attention_sum_field);
    try std.testing.expectEqual(@as(u32, 1024), plan.mlx_attention_single_simds * plan.mlx_attention_lanes);
    try std.testing.expectEqual(@as(u32, 1024), plan.mlx_attention_merge_simds * plan.mlx_attention_lanes);
    var visits: [plan.mlx_attention_dim]u8 = @splat(0);
    for (0..plan.mlx_attention_lanes) |lane| {
        for (0..plan.mlx_attention_elements_per_lane) |j| {
            const d = lane * plan.mlx_attention_elements_per_lane + j;
            try std.testing.expect(d < plan.mlx_attention_max_field);
            try std.testing.expectEqual(lane, d / plan.mlx_attention_elements_per_lane);
            try std.testing.expectEqual(j, d % plan.mlx_attention_elements_per_lane);
            visits[d] += 1;
        }
    }
    for (visits) |count| try std.testing.expectEqual(@as(u8, 1), count);
    try std.testing.expectEqualSlices(u32, &.{ 64, 128, 512, 1024 }, &plan.mlx_attention_blocks);
    for (0..2049) |blocks| {
        const expected = blocks == 64 or blocks == 128 or blocks == 512 or blocks == 1024;
        try std.testing.expectEqual(expected, plan.validMlxAttentionBlocks(@intCast(blocks)));
    }
    try std.testing.expect(!plan.validMlxAttentionBlocks(std.math.maxInt(u32)));
}

test "Flint GQA8 apple_g15d host policy has exact boundaries and only admissible results" {
    const cases = [_][2]u32{
        .{ 0, 128 }, .{ 8192, 128 }, .{ 16383, 128 }, .{ 16384, 512 },
        .{ 16385, 512 }, .{ 65535, 512 }, .{ 65536, 1024 }, .{ 65537, 1024 },
        .{ plan.mlx_attention_max_context, 1024 }, .{ std.math.maxInt(u32), 1024 },
    };
    for (cases) |case| try std.testing.expectEqual(case[1], plan.mlxLargeDeviceBlocks(case[0]));
    for (0..plan.mlx_attention_max_context + 1) |length| {
        const n: u32 = @intCast(length);
        const expected: u32 = if (n < 16384) 128 else if (n < 65536) 512 else 1024;
        const blocks = plan.mlxLargeDeviceBlocks(n);
        try std.testing.expectEqual(expected, blocks);
        try std.testing.expect(plan.validMlxAttentionBlocks(blocks));
    }
}

test "MLX stripe loops cover tokens exactly once including TK4 tails and empty stripes" {
    const lengths = [_]u32{ 0, 1, 31, 32, 33, 63, 64, 65, 127, 128, 129, 255, 256, 257, 511, 512, 513, 1023, 1024, 1025, 2047, 2048, 2049, 4095, 4096, 4097, 8191, 8192, 16383, 16384, 16385, 65535, 65536, 65537, 262143, 262144 };
    for (plan.mlx_attention_blocks) |blocks| {
        for (lengths) |n| {
            var visits: [plan.mlx_attention_max_context]u8 = @splat(0);
            var sinks: u32 = 0;
            for (0..blocks) |block_index| {
                const block: u32 = @intCast(block_index);
                if (block == 0) sinks += 1;
                try std.testing.expectEqual(@as(u1, 0), @addWithOverflow(block, 3 * blocks)[1]);
                try std.testing.expectEqual(@as(u1, 0), @addWithOverflow(block, 4 * blocks)[1]);
                var i = block;
                while (i + 3 * blocks < n) : (i += 4 * blocks) {
                    for (0..4) |t| {
                        const token = i + @as(u32, @intCast(t)) * blocks;
                        try std.testing.expectEqual(block, token % blocks);
                        visits[token] += 1;
                    }
                    try std.testing.expectEqual(@as(u1, 0), @addWithOverflow(i, 4 * blocks)[1]);
                }
                while (i < n) : (i += blocks) {
                    try std.testing.expectEqual(block, i % blocks);
                    try std.testing.expectEqual(i, block + (i / blocks) * blocks);
                    try std.testing.expectEqual(@as(u1, 0), @addWithOverflow(i, blocks)[1]);
                    visits[i] += 1;
                }
            }
            // Unlike contiguous-chunk producers, all B stripes exist even at n=0.
            try std.testing.expectEqual(@as(u32, 1), sinks);
            for (visits, 0..) |count, t| try std.testing.expectEqual(@as(u8, if (t < n) 1 else 0), count);
        }
    }
}

test "MLX packed state fields and merge block coordinates are disjoint and fit u32" {
    // The maximum state now exceeds the usual native stack budget; keep test bookkeeping on the heap.
    const visits = try std.testing.allocator.alloc(u8, plan.mlx_attention_max_state_elements);
    defer std.testing.allocator.free(visits);
    for (plan.mlx_attention_blocks) |blocks| {
        const allocation = plan.mlx_attention_max_heads * blocks * plan.mlx_attention_state_stride;
        try std.testing.expect(allocation <= plan.mlx_attention_max_state_elements);
        @memset(visits, 0);
        for (0..plan.mlx_attention_max_heads) |head| {
            for (0..blocks) |block| {
                const record = head * blocks + block;
                try std.testing.expectEqual(head, record / blocks);
                try std.testing.expectEqual(block, record % blocks);
                try std.testing.expect(block % plan.mlx_attention_merge_simds < plan.mlx_attention_merge_simds);
                try std.testing.expect(block / plan.mlx_attention_merge_simds < blocks / plan.mlx_attention_merge_simds);
                for (0..plan.mlx_attention_state_stride) |field| {
                    const index: u32 = @intCast(record * plan.mlx_attention_state_stride + field);
                    try std.testing.expect(index < allocation);
                    visits[index] += 1;
                }
            }
        }
        for (visits, 0..) |count, index| try std.testing.expectEqual(@as(u8, if (index < allocation) 1 else 0), count);
    }
}

test "MLX producer GQA launch owns every query head once" {
    for (1..plan.mlx_attention_max_gqa + 1) |gqa| {
        try std.testing.expect(gqa * plan.mlx_attention_lanes <= 1024);
        for (1..plan.mlx_attention_max_heads / gqa + 1) |kvheads| {
            var visits: [plan.mlx_attention_max_heads]u8 = @splat(0);
            const heads = kvheads * gqa;
            for (0..kvheads) |kvhead| {
                for (0..gqa) |simd| {
                    const head = kvhead * gqa + simd;
                    try std.testing.expectEqual(kvhead, head / gqa);
                    try std.testing.expectEqual(simd, head % gqa);
                    visits[head] += 1;
                }
            }
            for (visits, 0..) |count, head| try std.testing.expectEqual(@as(u8, if (head < heads) 1 else 0), count);
        }
    }
}

test "aligned norm and head integer ownership cover exactly four components per thread" {
    try std.testing.expectEqual(@as(u32, 2048), plan.aligned_norm_dim);
    try std.testing.expectEqual(@as(u32, 512), plan.aligned_norm_threads);
    try std.testing.expectEqual(@as(u32, 4), plan.aligned_norm_elements_per_thread);
    try std.testing.expectEqual(@as(u32, 128), plan.aligned_head_dim);
    try std.testing.expectEqual(@as(u32, 32), plan.aligned_head_threads);
    try std.testing.expectEqual(@as(u32, 4), plan.aligned_head_elements_per_thread);
    const geometries = [_][3]u32{
        .{ plan.aligned_norm_dim, plan.aligned_norm_threads, plan.aligned_norm_elements_per_thread },
        .{ plan.aligned_head_dim, plan.aligned_head_threads, plan.aligned_head_elements_per_thread },
    };
    for (geometries) |geometry| {
        const dim, const threads, const width = geometry;
        try std.testing.expectEqual(dim, threads * width);
        var visits: [2048]u8 = @splat(0);
        for (0..threads) |thread| {
            for (0..width) |j| {
                const d = thread * width + j;
                try std.testing.expectEqual(thread, d / width);
                try std.testing.expectEqual(j, d % width);
                visits[d] += 1;
            }
        }
        for (visits, 0..) |count, d| try std.testing.expectEqual(@as(u8, if (d < dim) 1 else 0), count);
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
