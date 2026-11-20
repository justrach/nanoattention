//! decode climb 2: headless tests of the decode2.metal kernels against the
//! kernels they replace, bit for bit, on random and adversarial inputs.
const std = @import("std");
const gpu = @import("gpu.zig");
const metal = @import("weights").metal;
const testing = std.testing;

fn bitsEqual(expected: []const f32, actual: []const f32) !void {
    try testing.expectEqualSlices(u32, std.mem.bytesAsSlice(u32, std.mem.sliceAsBytes(expected)), std.mem.bytesAsSlice(u32, std.mem.sliceAsBytes(actual)));
}

/// Signed zeros, subnormals, infinities, NaN, extremes and ordinary values.
fn adversarial(r: std.Random, i: usize) f32 {
    const specials = [_]f32{ 0.0, -0.0, std.math.floatTrueMin(f32), -std.math.floatTrueMin(f32), 1e-40, -3e-39, std.math.inf(f32), -std.math.inf(f32), std.math.nan(f32), std.math.floatMax(f32), -std.math.floatMax(f32), 1.0, -1.0 };
    if (i % 3 == 0) return specials[r.uintLessThan(usize, specials.len)];
    return (r.float(f32) - 0.5) * std.math.pow(f32, 2, @floatFromInt(r.intRangeAtMost(i32, -20, 20)));
}

test "decode climb 2: gated embedding equals mk_embed bit for bit; a stale generation gives NaN and the flag" {
    const ctx = try gpu.open();
    defer ctx.deinit();
    var prng = std.Random.DefaultPrng.init(2020);
    const r = prng.random();
    const vocab = 300;
    const n = 192;
    // One table buffer, 2 bytes off a word boundary like the checkpoint: dense
    // f32 / f16 / bf16 rows, or 4- and 8-bit codes with bf16 or f32 scales and biases.
    const bytes_len = 2 + vocab * n * 4 + 2 * vocab * n * 4 + 64;
    const tbuf = try ctx.alloc(bytes_len / 4 + 1);
    defer tbuf.deinit();
    const tb = std.mem.sliceAsBytes(tbuf.floats());
    const arena = try ctx.alloc(4 * n);
    defer arena.deinit();
    const gate = try ctx.alloc(16);
    defer gate.deinit();
    const Case = struct { bits: u32, dtype: u32, sdtype: u32, group: u32, round: bool, off: u64 };
    const cases = [_]Case{
        .{ .bits = 0, .dtype = 0, .sdtype = 0, .group = 0, .round = false, .off = 4 },
        .{ .bits = 0, .dtype = 1, .sdtype = 0, .group = 0, .round = false, .off = 2 },
        .{ .bits = 0, .dtype = 2, .sdtype = 0, .group = 0, .round = true, .off = 2 },
        .{ .bits = 4, .dtype = 0, .sdtype = 2, .group = 64, .round = false, .off = 2 },
        .{ .bits = 4, .dtype = 0, .sdtype = 2, .group = 32, .round = true, .off = 2 },
        .{ .bits = 4, .dtype = 0, .sdtype = 0, .group = 64, .round = false, .off = 4 },
        .{ .bits = 8, .dtype = 0, .sdtype = 2, .group = 32, .round = false, .off = 2 },
    };
    for (cases, 0..) |c, ci| {
        // Random bytes everywhere: codes, and dense values or scale/bias words that
        // include every class of special value.
        r.bytes(tb);
        const elem: u64 = if (c.bits != 0) 0 else if (c.dtype == 0) 4 else 2;
        const code_bytes: u64 = if (c.bits != 0) vocab * n * c.bits / 8 else vocab * n * elem;
        const sb_off = (c.off + code_bytes + 3) / 4 * 4 + 2 * @as(u64, @intFromBool(c.sdtype != 0));
        const groups: u64 = if (c.group != 0) vocab * n / c.group else 0;
        const ssz: u64 = if (c.sdtype == 0) 4 else 2;
        const bb_off = sb_off + groups * ssz + 4;
        const fill = struct {
            fn f(bytes: []u8, off: u64, count: u64, dt: u32, rr: std.Random) void {
                for (0..count) |i| {
                    const v = adversarial(rr, i);
                    const at = off + i * (if (dt == 0) @as(u64, 4) else 2);
                    switch (dt) {
                        0 => std.mem.writeInt(u32, bytes[at..][0..4], @bitCast(v), .little),
                        1 => std.mem.writeInt(u16, bytes[at..][0..2], @bitCast(@as(f16, @floatCast(v))), .little),
                        else => std.mem.writeInt(u16, bytes[at..][0..2], @intCast(@as(u32, @bitCast(v)) >> 16), .little),
                    }
                }
            }
        }.f;
        if (c.bits == 0) fill(tb, c.off, vocab * n, c.dtype, r) else {
            fill(tb, sb_off, groups, c.sdtype, r);
            fill(tb, bb_off, groups, c.sdtype, r);
        }
        for (0..12) |k| {
            const token: u32 = switch (k) {
                0 => 0,
                1 => vocab - 1,
                else => r.uintLessThan(u32, vocab),
            };
            const gen: u32 = @intCast((ci * 37 + k * 11) & 0xfff);
            arena.structs(u32)[0] = token;
            var p: gpu.Params = .{ .rows = 1, .x = 0, .xs = 1, .y = 4, .ys = n, .n = n, .w = c.off, .dtype = c.dtype, .bits = c.bits, .group = c.group, .s = sb_off, .b = bb_off, .sdtype = c.sdtype, .bdtype = c.sdtype };
            if (c.round) p.flags |= gpu.flag_round_bf16;
            try gpu.dispatch(ctx, "mk_embed", &.{ tbuf, tbuf, tbuf, arena }, &p, n, 64);
            try ctx.finish();
            const expected = try testing.allocator.dupe(f32, arena.floats()[4..][0..n]);
            defer testing.allocator.free(expected);
            @memset(arena.floats()[4..][0..n], 12345);
            @memset(gate.structs(u32), 0);
            gate.structs(u32)[0] = gen << 20 | token;
            p.every = gen;
            try gpu.dispatch(ctx, "mk_d2_embed_gated", &.{ tbuf, tbuf, tbuf, arena, gate }, &p, n, 64);
            try ctx.finish();
            try bitsEqual(expected, arena.floats()[4..][0..n]);
            try testing.expectEqual(@as(u32, 0), gate.structs(u32)[1]);
            // Stale generation: NaN everywhere, flag set.
            p.every = (gen + 1) & 0xfff;
            try gpu.dispatch(ctx, "mk_d2_embed_gated", &.{ tbuf, tbuf, tbuf, arena, gate }, &p, n, 64);
            try ctx.finish();
            for (arena.floats()[4..][0..n]) |v| try testing.expect(std.math.isNan(v));
            try testing.expectEqual(@as(u32, 1), gate.structs(u32)[1]);
        }
    }
}

test "decode climb 2: the gate passes on its generation and gives up after its bound otherwise" {
    const ctx = try gpu.open();
    defer ctx.deinit();
    const gate = try ctx.alloc(16);
    defer gate.deinit();
    for ([_][:0]const u8{"mk_d2_gate_packed"}) |kernel| {
        @memset(gate.structs(u32), 0);
        gate.structs(u32)[0] = 5 << 20 | 1234;
        var p: gpu.Params = .{ .every = 5, .n = 1000 };
        try gpu.dispatch(ctx, kernel, &.{gate}, &p, 32, 32);
        try ctx.finish();
        try testing.expectEqual(@as(u32, 0), gate.structs(u32)[2]);
        try testing.expectEqual(@as(u32, 5 << 20 | 1234), gate.structs(u32)[0]);
        p.every = 6;
        try gpu.dispatch(ctx, kernel, &.{gate}, &p, 32, 32);
        try ctx.finish();
        try testing.expectEqual(@as(u32, 1000), gate.structs(u32)[2]);
        try testing.expectEqual(@as(u32, 5 << 20 | 1234), gate.structs(u32)[0]);
    }
}

test "decode climb 2: register-resident sandwich norm equals mk_norm_add_norm bit for bit, adversarial inputs" {
    const ctx = try gpu.open();
    defer ctx.deinit();
    var prng = std.Random.DefaultPrng.init(4242);
    const r = prng.random();
    // (n, weight dtype, kind): LFM3's 2048, ragged widths, the 4096 limit, both operand orders.
    const cases = [_]struct { n: u32, dt: u32, kind: u32 }{
        .{ .n = 2048, .dt = 2, .kind = 0 }, .{ .n = 2048, .dt = 2, .kind = 1 }, .{ .n = 2040, .dt = 0, .kind = 0 },
        .{ .n = 100, .dt = 1, .kind = 1 },  .{ .n = 4096, .dt = 2, .kind = 0 }, .{ .n = 33, .dt = 0, .kind = 1 },
        .{ .n = 4, .dt = 2, .kind = 0 },
    };
    const rows = 3;
    for (cases, 0..) |c, ci| {
        const n = c.n;
        const wbuf = try ctx.alloc(2 * n + 4);
        defer wbuf.deinit();
        const wb = std.mem.sliceAsBytes(wbuf.floats());
        const esz: usize = if (c.dt == 0) 4 else 2;
        const w0off: usize = 2 * @as(usize, @intFromBool(c.dt != 0)); // 2 mod 4 for 16-bit weights
        const w1off = w0off + n * esz;
        for (0..2 * n) |i| {
            // Mostly ordinary weights, some specials (but finite-most rows stay informative).
            const v: f32 = if (i % 97 == 5) adversarial(r, 0) else 1 + 0.2 * r.floatNorm(f32);
            const at = w0off + i * esz;
            switch (c.dt) {
                0 => std.mem.writeInt(u32, wb[at..][0..4], @bitCast(v), .little),
                1 => std.mem.writeInt(u16, wb[at..][0..2], @bitCast(@as(f16, @floatCast(v))), .little),
                else => std.mem.writeInt(u16, wb[at..][0..2], @intCast(@as(u32, @bitCast(v)) >> 16), .little),
            }
        }
        // arena: y | r | out, rows x n each. Row 0 ordinary, row 1 adversarial
        // (signed zeros, subnormals, inf, NaN, extremes), row 2 heavy cancellation.
        var init = try testing.allocator.alloc(f32, 2 * rows * n);
        defer testing.allocator.free(init);
        for (0..n) |i| {
            init[i] = r.floatNorm(f32);
            init[rows * n + i] = r.floatNorm(f32) * 3;
            init[n + i] = adversarial(r, i);
            init[rows * n + n + i] = adversarial(r, i + 1);
            const big = 1e30 * (r.float(f32) + 0.5);
            init[2 * n + i] = if (i % 2 == 0) big else -big;
            init[rows * n + 2 * n + i] = if (i % 2 == 0) -1e30 else 1e30;
        }
        var outs: [2][]f32 = undefined;
        for ([_][:0]const u8{ "mk_norm_add_norm", "mk_d2_norm_add_norm" }, 0..) |kernel, k| {
            const buf = try ctx.alloc(3 * rows * n);
            defer buf.deinit();
            @memcpy(buf.floats()[0 .. 2 * rows * n], init);
            @memset(buf.floats()[2 * rows * n ..], 7);
            const p: gpu.Params = .{ .rows = rows, .x = 0, .xs = n, .y = rows * n, .ys = n, .z = 2 * rows * n, .zs = n, .n = n, .w = w0off, .dtype = c.dt, .s = w1off, .sdtype = c.dt, .eps = 1e-5, .kind = c.kind };
            const tg = gpu.normThreads(n);
            try gpu.dispatch(ctx, kernel, &.{ wbuf, buf }, &p, rows * tg, tg);
            try ctx.finish();
            outs[k] = try testing.allocator.dupe(f32, buf.floats()[rows * n ..]);
        }
        defer for (outs) |o| testing.allocator.free(o);
        bitsEqual(outs[0], outs[1]) catch |e| {
            std.debug.print("case {d} (n {d}) differs\n", .{ ci, n });
            return e;
        };
    }
}
