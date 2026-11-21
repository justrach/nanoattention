//! Headless Metal tests: each megakernel against a plain CPU implementation,
//! and bit identity of batched rows against one-row dispatches.
const std = @import("std");
const gpu = @import("gpu.zig");
const metal = @import("weights").metal;
const testing = std.testing;

fn rng(seed: u64) std.Random.DefaultPrng {
    return std.Random.DefaultPrng.init(seed);
}
fn bf16(v: f32) u16 {
    const bits: u32 = @bitCast(v);
    return @intCast((bits + 0x7fff + ((bits >> 16) & 1)) >> 16);
}
fn fromBf16(h: u16) f32 {
    return @bitCast(@as(u32, h) << 16);
}
fn close(expected: f64, actual: f32, rel: f64) !void {
    const err = @abs(expected - @as(f64, actual));
    if (err > rel * (1 + @abs(expected))) {
        std.debug.print("expected {d} got {d}\n", .{ expected, actual });
        return error.TestExpectedApproxEq;
    }
}

const Quant = struct { w: metal.Buffer, s: metal.Buffer, b: metal.Buffer, dense: []f32 };

/// Affine 4-bit [m, n] with bf16 scales/biases; `dense` holds decoded values.
fn quantize(ctx: metal.Context, a: std.mem.Allocator, r: *std.Random.DefaultPrng, m: usize, n: usize, group: usize) !Quant {
    const w = try ctx.alloc(m * n / 8);
    const s = try ctx.alloc((m * n / group + 1) / 2);
    const b = try ctx.alloc((m * n / group + 1) / 2);
    const dense = try a.alloc(f32, m * n);
    const words = w.structs(u32);
    @memset(words, 0);
    const scales = std.mem.bytesAsSlice(u16, std.mem.sliceAsBytes(s.floats()));
    const biases = std.mem.bytesAsSlice(u16, std.mem.sliceAsBytes(b.floats()));
    for (0..m * n / group) |g| {
        scales[g] = bf16(0.01 + 0.05 * r.random().float(f32));
        biases[g] = bf16(-0.3 + 0.1 * r.random().float(f32));
    }
    for (0..m * n) |i| {
        const code: u32 = r.random().uintLessThan(u32, 16);
        words[i / 8] |= code << @intCast((i % 8) * 4);
        dense[i] = @as(f32, @floatFromInt(code)) * fromBf16(scales[i / group]) + fromBf16(biases[i / group]);
    }
    return .{ .w = w, .s = s, .b = b, .dense = dense };
}

test "4-bit linear matches CPU and every row/token tile is bit-identical to one-row dispatch" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const ctx = try gpu.open();
    defer ctx.deinit();
    var r = rng(11);
    for ([_][3]usize{ .{ 13, 128, 64 }, .{ 9, 2048, 64 }, .{ 7, 96, 32 } }) |shape| {
        const m = shape[0];
        const n = shape[1];
        const q = try quantize(ctx, a, &r, m, n, shape[2]);
        defer {
            q.w.deinit();
            q.s.deinit();
            q.b.deinit();
        }
        const T = 5;
        const arena_buf = try ctx.alloc(T * n + T * m);
        defer arena_buf.deinit();
        const f = arena_buf.floats();
        for (f[0 .. T * n]) |*v| v.* = r.random().floatNorm(f32);
        var p: gpu.Params = .{ .rows = 1, .x = 0, .xs = @intCast(n), .y = @intCast(T * n), .ys = @intCast(m), .n = @intCast(n), .m = @intCast(m), .bits = 4, .group = @intCast(shape[2]), .sdtype = 2, .bdtype = 2 };
        const reference = try a.alloc(f32, T * m);
        for (0..T) |t| {
            p.row0 = @intCast(t);
            try gpu.dispatch(ctx, "mk_linear_q4_r1_t1", &.{ q.w, q.s, q.b, arena_buf }, &p, m * 32, 32);
        }
        try ctx.finish();
        @memcpy(reference, f[T * n ..][0 .. T * m]);
        for (0..T) |t| for (0..m) |row| {
            var sum: f64 = 0;
            for (0..n) |c| sum += @as(f64, q.dense[row * n + c]) * f[t * n + c];
            try close(sum, reference[t * m + row], 1e-4);
        };
        for ([_]u32{ 1, 2, 4, 8 }) |R| for ([_]u32{ 1, 2, 4 }) |TT| {
            @memset(f[T * n ..][0 .. T * m], 0);
            p.row0 = 0;
            p.rows = T;
            const name = try std.fmt.allocPrintSentinel(a, "mk_linear_q4_r{d}_t{d}", .{ R, TT }, 0);
            const simds = ((m + R - 1) / R) * ((T + TT - 1) / TT);
            try gpu.dispatch(ctx, name, &.{ q.w, q.s, q.b, arena_buf }, &p, simds * 32, 64);
            try ctx.finish();
            try testing.expectEqualSlices(u32, std.mem.bytesAsSlice(u32, std.mem.sliceAsBytes(reference)), std.mem.bytesAsSlice(u32, std.mem.sliceAsBytes(f[T * n ..][0 .. T * m])));
        };
    }
}

test "generic linear: dense bf16 and affine 8-bit" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const ctx = try gpu.open();
    defer ctx.deinit();
    var r = rng(5);
    const m = 6;
    const n = 40;
    const T = 3;
    const w = try ctx.alloc(m * n / 2);
    defer w.deinit();
    const halves = std.mem.bytesAsSlice(u16, std.mem.sliceAsBytes(w.floats()));
    const dense = try a.alloc(f32, m * n);
    for (dense, 0..) |*d, i| {
        halves[i] = bf16(r.random().floatNorm(f32));
        d.* = fromBf16(halves[i]);
    }
    const buf = try ctx.alloc(T * n + T * m);
    defer buf.deinit();
    const f = buf.floats();
    for (f[0 .. T * n]) |*v| v.* = r.random().floatNorm(f32);
    var p: gpu.Params = .{ .rows = T, .xs = n, .y = T * n, .ys = m, .n = n, .m = m, .dtype = 2 };
    try gpu.dispatch(ctx, "mk_linear_generic", &.{ w, w, w, buf }, &p, m * T * 32, 64);
    try ctx.finish();
    for (0..T) |t| for (0..m) |row| {
        var sum: f64 = 0;
        for (0..n) |c| sum += @as(f64, dense[row * n + c]) * f[t * n + c];
        try close(sum, f[T * n + t * m + row], 1e-5);
    };
    // 8-bit affine, group 8, f32 scales.
    const q = try ctx.alloc(m * n / 4);
    defer q.deinit();
    const sb = try ctx.alloc(m * n / 8 * 2);
    defer sb.deinit();
    const bytes = std.mem.sliceAsBytes(q.floats());
    const sc = sb.floats();
    for (0..m * n / 8) |g| {
        sc[g] = 0.01 + 0.01 * r.random().float(f32);
        sc[m * n / 8 + g] = -0.5;
    }
    for (0..m * n) |i| {
        bytes[i] = r.random().int(u8);
        dense[i] = @as(f32, @floatFromInt(bytes[i])) * sc[i / 8] + sc[m * n / 8 + i / 8];
    }
    p = .{ .rows = T, .xs = n, .y = T * n, .ys = m, .n = n, .m = m, .bits = 8, .group = 8, .sdtype = 0, .bdtype = 0, .b = m * n / 8 * 4 };
    try gpu.dispatch(ctx, "mk_linear_generic", &.{ q, sb, sb, buf }, &p, m * T * 32, 64);
    try ctx.finish();
    for (0..T) |t| for (0..m) |row| {
        var sum: f64 = 0;
        for (0..n) |c| sum += @as(f64, dense[row * n + c]) * f[t * n + c];
        try close(sum, f[T * n + t * m + row], 1e-5);
    };
}

test "RMSNorm per head and whole vector, in place, with bf16 weight" {
    const ctx = try gpu.open();
    defer ctx.deinit();
    var r = rng(7);
    for ([_][2]u32{ .{ 128, 4 }, .{ 2048, 1 }, .{ 40, 3 } }) |shape| {
        const n = shape[0];
        const heads = shape[1];
        const T = 3;
        const width = n * heads;
        const w = try ctx.alloc(n / 2 + 1);
        defer w.deinit();
        const halves = std.mem.bytesAsSlice(u16, std.mem.sliceAsBytes(w.floats()));
        for (0..n) |i| halves[i] = bf16(1 + 0.1 * r.random().floatNorm(f32));
        const buf = try ctx.alloc(T * width);
        defer buf.deinit();
        const f = buf.floats();
        for (f) |*v| v.* = r.random().floatNorm(f32) * 3;
        var copy: [3 * 2048 * 4]f32 = undefined;
        @memcpy(copy[0 .. T * width], f);
        const p: gpu.Params = .{ .rows = T, .xs = width, .ys = width, .n = n, .heads = heads, .dtype = 2, .flags = gpu.flag_has_weight, .eps = 1e-5 };
        const tg = gpu.normThreads(n);
        try gpu.dispatch(ctx, "mk_rmsnorm", &.{ w, buf }, &p, T * heads * tg, tg);
        try ctx.finish();
        for (0..T * heads) |g| {
            var ss: f64 = 0;
            for (0..n) |i| ss += @as(f64, copy[g * n + i]) * copy[g * n + i];
            const inv = 1 / @sqrt(ss / @as(f64, @floatFromInt(n)) + 1e-5);
            for (0..n) |i| try close(copy[g * n + i] * inv * fromBf16(halves[i]), f[g * n + i], 1e-5);
        }
    }
}

test "binary ops with activations and broadcast gate" {
    const ctx = try gpu.open();
    defer ctx.deinit();
    const buf = try ctx.alloc(64);
    defer buf.deinit();
    const f = buf.floats();
    for (0..8) |i| f[i] = @as(f32, @floatFromInt(i)) - 3.5;
    for (0..2) |i| f[16 + i] = @as(f32, @floatFromInt(i)) * 2 - 1;
    var p: gpu.Params = .{ .rows = 1, .x = 0, .y = 16, .z = 32, .n = 8, .every = 4, .kind = 1, .act_b = 2 };
    try gpu.dispatch(ctx, "mk_binary", &.{buf}, &p, 8, 8);
    p = .{ .rows = 1, .x = 0, .y = 0, .z = 48, .n = 8, .every = 1, .kind = 0, .act_a = 1 };
    try gpu.dispatch(ctx, "mk_binary", &.{buf}, &p, 8, 8);
    p = .{ .rows = 1, .x = 0, .y = 16, .z = 56, .n = 8, .every = 4, .kind = 2, .act_a = 3 };
    try gpu.dispatch(ctx, "mk_binary", &.{buf}, &p, 8, 8);
    try ctx.finish();
    for (0..8) |i| {
        const x: f64 = @as(f64, @floatFromInt(i)) - 3.5;
        const g: f64 = @as(f64, @floatFromInt(i / 4)) * 2 - 1;
        try close(x / (1 + @exp(-g)), f[32 + i], 1e-6);
        try close(x / (1 + @exp(-x)) + x, f[48 + i], 1e-6);
        const y: f64 = @as(f64, @floatFromInt(i / 4)) * 2 - 1;
        try close((2.0 / (1.0 + @exp(-2.0 * x)) - 1.0) - y, f[56 + i], 1e-6);
    }
}

test "concat and tanh activation match CPU reference" {
    const ctx = try gpu.open();
    defer ctx.deinit();
    const buf = try ctx.alloc(64);
    defer buf.deinit();
    const f = buf.floats();
    @memcpy(f[0..8], &[_]f32{ 1, 2, 3, 4, 5, 6, 7, 8 });
    @memcpy(f[8..12], &[_]f32{ 9, 10, 11, 12 });
    var p: gpu.Params = .{ .rows = 2, .x = 0, .xs = 4, .y = 8, .ys = 2, .z = 16, .zs = 6, .n = 4, .m = 2 };
    try gpu.dispatch(ctx, "mk_concat", &.{buf}, &p, 12, 8);
    @memcpy(f[32..37], &[_]f32{ -2, -0.5, 0, 0.5, 2 });
    p = .{ .rows = 1, .x = 32, .xs = 5, .z = 40, .zs = 5, .n = 5, .act_a = 3 };
    try gpu.dispatch(ctx, "mk_activate", &.{buf}, &p, 5, 8);
    try ctx.finish();
    try std.testing.expectEqualSlices(f32, &.{ 1, 2, 3, 4, 9, 10 }, f[16..22]);
    try std.testing.expectEqualSlices(f32, &.{ 5, 6, 7, 8, 11, 12 }, f[22..28]);
    for (f[32..37], f[40..45]) |x, y| try close((2.0 / (1.0 + @exp(-2.0 * x)) - 1.0), y, 1e-6);
}

test "Inkling softcapped SwiGLU matches the reference formula" {
    const ctx = try gpu.open();
    defer ctx.deinit();
    const buf = try ctx.alloc(32);
    defer buf.deinit();
    const f = buf.floats();
    const gates = [_]f32{ -5, 1, 12 };
    const ups = [_]f32{ -3, 10, 0.2 };
    @memcpy(f[0..3], &gates);
    @memcpy(f[8..11], &ups);
    const p: gpu.Params = .{ .rows = 1, .x = 0, .xs = 3, .y = 8, .ys = 3, .z = 16, .zs = 3, .n = 3, .eps = 4, .scale = 25 };
    try gpu.dispatch(ctx, "mk_inkling_softcapped_swiglu", &.{buf}, &p, 3, 8);
    try ctx.finish();
    for (gates, ups, f[16..19]) |gate, up, actual| {
        const gt = 2.0 / (1.0 + @exp(-2.0 * @as(f64, gate / 4.0))) - 1.0;
        const ut = 2.0 / (1.0 + @exp(-2.0 * @as(f64, up / 25.0))) - 1.0;
        const capped_gate = gt * 4.0;
        const capped_up = ut * 25.0;
        const sigmoid = 1.0 / (1.0 + @exp(-@as(f64, gate)));
        try close(capped_gate * sigmoid * capped_up, actual, 1e-6);
    }
}

test "depth attention matches CPU softmax over bank and partial state" {
    const ctx = try gpu.open();
    defer ctx.deinit();
    const arena = try ctx.alloc(64);
    defer arena.deinit();
    const qbuf = try ctx.alloc(4);
    defer qbuf.deinit();
    const f = arena.floats();
    @memcpy(f[0..16], &[_]f32{ 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1 });
    @memcpy(f[16..24], &[_]f32{ 1, 1, 0, 0, 0, 0, 1, 1 });
    @memcpy(qbuf.floats()[0..4], &[_]f32{ 1, 1, 0, 0 });
    const p: gpu.Params = .{ .rows = 2, .x = 0, .xs = 8, .y = 16, .ys = 4, .z = 24, .zs = 4, .n = 4, .m = 2, .heads = 3, .dtype = 0, .scale = 0.5 };
    try gpu.dispatch(ctx, "mk_depth_attention", &.{ arena, qbuf }, &p, 2 * 256, 256);
    try ctx.finish();
    for (0..2) |row| {
        var logits: [3]f64 = undefined;
        var maximum: f64 = -std.math.inf(f64);
        for (0..3) |candidate| {
            var dot: f64 = 0;
            for (0..4) |d| {
                const state: f32 = if (candidate < 2)
                    f[row * 8 + candidate * 4 + d]
                else
                    f[16 + row * 4 + d];
                dot += @as(f64, state) * qbuf.floats()[d];
            }
            logits[candidate] = dot * 0.5;
            maximum = @max(maximum, logits[candidate]);
        }
        var denominator: f64 = 0;
        for (&logits) |*value| {
            value.* = @exp(value.* - maximum);
            denominator += value.*;
        }
        for (0..4) |d| {
            var expected: f64 = 0;
            for (0..3) |candidate| {
                const state: f32 = if (candidate < 2)
                    f[row * 8 + candidate * 4 + d]
                else
                    f[16 + row * 4 + d];
                expected += logits[candidate] / denominator * state;
            }
            try close(expected, f[24 + row * 4 + d], 2e-6);
        }
    }
}

test "ring store and causal conv: CPU match, zero history, batch equals one-at-a-time" {
    const ctx = try gpu.open();
    defer ctx.deinit();
    const C = 5;
    const taps = 4;
    const T = 6;
    const capacity = taps - 1 + T;
    var r = rng(3);
    const w = try ctx.alloc(C * taps);
    defer w.deinit();
    for (w.floats()) |*v| v.* = r.random().floatNorm(f32);
    var inputs: [20 * C]f32 = undefined;
    for (&inputs) |*v| v.* = r.random().floatNorm(f32);
    var outs: [2][20 * C]f32 = undefined;
    for ([_]u32{ 1, T }, 0..) |feed, mode| {
        const ring = try ctx.alloc(capacity * C);
        defer ring.deinit();
        @memset(ring.floats(), std.math.nan(f32)); // never read before written
        const buf = try ctx.alloc(2 * T * C);
        defer buf.deinit();
        var pos: u32 = 0;
        while (pos < 18) : (pos += feed) {
            @memcpy(buf.floats()[0 .. feed * C], inputs[pos * C ..][0 .. feed * C]);
            var p: gpu.Params = .{ .rows = feed, .pos = pos, .x = 0, .xs = C, .n = C, .capacity = capacity };
            try gpu.dispatch(ctx, "mk_store", &.{ buf, ring }, &p, feed * C, 32);
            p = .{ .rows = feed, .pos = pos, .y = T * C, .ys = C, .n = C, .taps = taps, .capacity = capacity };
            try gpu.dispatch(ctx, "mk_conv", &.{ ring, w, buf }, &p, feed * C, 32);
            try ctx.finish();
            @memcpy(outs[mode][pos * C ..][0 .. feed * C], buf.floats()[T * C ..][0 .. feed * C]);
        }
    }
    try testing.expectEqualSlices(u32, std.mem.bytesAsSlice(u32, std.mem.sliceAsBytes(outs[0][0 .. 18 * C])), std.mem.bytesAsSlice(u32, std.mem.sliceAsBytes(outs[1][0 .. 18 * C])));
    for (0..18) |t| for (0..C) |c| {
        var sum: f64 = 0;
        for (0..taps) |k| {
            const q = @as(i64, @intCast(t)) - taps + 1 + @as(i64, @intCast(k));
            if (q >= 0) sum += @as(f64, w.floats()[c * taps + k]) * inputs[@as(usize, @intCast(q)) * C + c];
        }
        try close(sum, outs[0][t * C + c], 1e-5);
    };
}

test "rope with a zero-tail frequency table leaves the tail unrotated" {
    const ctx = try gpu.open();
    defer ctx.deinit();
    const dim = 8;
    const heads = 2;
    const buf = try ctx.alloc(2 * heads * dim);
    defer buf.deinit();
    const tables = try ctx.alloc(dim / 2);
    defer tables.deinit();
    @memcpy(tables.floats(), &[_]f32{ 1, 0.1, 0, 0 });
    const f = buf.floats();
    for (f, 0..) |*v, i| v.* = @as(f32, @floatFromInt(i % 7)) - 2;
    var copy: [32]f32 = undefined;
    @memcpy(&copy, f);
    const p: gpu.Params = .{ .rows = 2, .pos = 5, .xs = heads * dim, .heads = heads, .dim = dim, .rot = dim, .z = 0 };
    try gpu.dispatch(ctx, "mk_rope", &.{ buf, tables }, &p, 2 * heads * dim / 2, 8);
    try ctx.finish();
    for (0..2) |row| for (0..heads) |h| for (0..dim / 2) |j| {
        const base = row * heads * dim + h * dim;
        const angle = @as(f64, @floatFromInt(5 + row)) * @as(f64, tables.floats()[j]);
        const x1: f64 = copy[base + j];
        const x2: f64 = copy[base + j + dim / 2];
        try close(x1 * @cos(angle) - x2 * @sin(angle), f[base + j], 1e-6);
        try close(x1 * @sin(angle) + x2 * @cos(angle), f[base + j + dim / 2], 1e-6);
        if (j >= 2) try testing.expectEqual(copy[base + j], f[base + j]);
    };
}

test "GQA attention: causal window, ring slots, CPU match, batch equals one row" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const ctx = try gpu.open();
    defer ctx.deinit();
    const heads = 4;
    const kvh = 2;
    const dim = 64;
    const T = 4;
    var r = rng(9);
    for ([_]u32{ 0, 5 }) |window| {
        const positions = 23;
        const capacity: u32 = if (window == 0) positions else window - 1 + T;
        const keys = try a.alloc(f32, positions * kvh * dim);
        const values = try a.alloc(f32, positions * kvh * dim);
        for (keys) |*v| v.* = r.random().floatNorm(f32);
        for (values) |*v| v.* = r.random().floatNorm(f32);
        const q = try a.alloc(f32, positions * heads * dim);
        for (q) |*v| v.* = r.random().floatNorm(f32);
        var results: [2][]f32 = undefined;
        for ([_]u32{ 1, T }, 0..) |feed, mode| {
            results[mode] = try a.alloc(f32, positions * heads * dim);
            const K = try ctx.alloc(capacity * kvh * dim);
            defer K.deinit();
            const V = try ctx.alloc(capacity * kvh * dim);
            defer V.deinit();
            const buf = try ctx.alloc(3 * T * heads * dim);
            defer buf.deinit();
            var pos: u32 = 0;
            while (pos + feed <= 20) : (pos += feed) {
                for (0..feed) |row| {
                    const slot = (pos + row) % capacity;
                    @memcpy(K.floats()[slot * kvh * dim ..][0 .. kvh * dim], keys[(pos + row) * kvh * dim ..][0 .. kvh * dim]);
                    @memcpy(V.floats()[slot * kvh * dim ..][0 .. kvh * dim], values[(pos + row) * kvh * dim ..][0 .. kvh * dim]);
                }
                @memcpy(buf.floats()[0 .. feed * heads * dim], q[pos * heads * dim ..][0 .. feed * heads * dim]);
                const p: gpu.Params = .{ .rows = feed, .pos = pos, .xs = heads * dim, .y = T * heads * dim, .ys = heads * dim, .heads = heads, .kv_heads = kvh, .dim = dim, .window = window, .capacity = capacity, .scale = 0.125 };
                try gpu.dispatch(ctx, "mk_attention", &.{ buf, K, V }, &p, feed * heads * 256, 256);
                try ctx.finish();
                @memcpy(results[mode][pos * heads * dim ..][0 .. feed * heads * dim], buf.floats()[T * heads * dim ..][0 .. feed * heads * dim]);
            }
        }
        const used = 20 * heads * dim;
        try testing.expectEqualSlices(u32, std.mem.bytesAsSlice(u32, std.mem.sliceAsBytes(results[0][0..used])), std.mem.bytesAsSlice(u32, std.mem.sliceAsBytes(results[1][0..used])));
        for (0..20) |pos| for (0..heads) |h| {
            const g = h / (heads / kvh);
            const start = if (window != 0 and pos + 1 > window) pos + 1 - window else 0;
            var scores: [32]f64 = undefined;
            var mx: f64 = -std.math.inf(f64);
            for (start..pos + 1) |t| {
                var d: f64 = 0;
                for (0..dim) |e| d += 0.125 * @as(f64, q[(pos * heads + h) * dim + e]) * keys[(t * kvh + g) * dim + e];
                scores[t - start] = d;
                mx = @max(mx, d);
            }
            var total: f64 = 0;
            for (start..pos + 1) |t| total += @exp(scores[t - start] - mx);
            for (0..dim) |e| {
                var o: f64 = 0;
                for (start..pos + 1) |t| o += @exp(scores[t - start] - mx) / total * values[(t * kvh + g) * dim + e];
                try close(o, results[0][(pos * heads + h) * dim + e], 1e-5);
            }
        };
    }
}

test "argmax: lowest index wins ties, non-finite gives the sentinel, embed dequantizes" {
    const ctx = try gpu.open();
    defer ctx.deinit();
    const n = 5000;
    const buf = try ctx.alloc(2 * n + 4);
    defer buf.deinit();
    const f = buf.floats();
    for (f[0 .. 2 * n], 0..) |*v, i| v.* = @floatFromInt(i % 97);
    f[3001] = 500;
    f[4000] = 500;
    f[n + 77] = std.math.nan(f32);
    const p: gpu.Params = .{ .rows = 2, .xs = n, .y = 2 * n, .ys = 1, .n = n };
    try gpu.dispatch(ctx, "mk_argmax", &.{buf}, &p, 2 * 1024, 1024);
    try ctx.finish();
    try testing.expectEqual(@as(u32, 3001), buf.structs(u32)[2 * n]);
    try testing.expectEqual(@as(u32, 0xffffffff), buf.structs(u32)[2 * n + 1]);

    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    var r = rng(1);
    const q = try quantize(ctx, arena.allocator(), &r, 10, 64, 32);
    defer {
        q.w.deinit();
        q.s.deinit();
        q.b.deinit();
    }
    const e = try ctx.alloc(2 + 2 * 64);
    defer e.deinit();
    e.structs(u32)[0] = 7;
    e.structs(u32)[1] = 2;
    const ep: gpu.Params = .{ .rows = 2, .x = 0, .xs = 1, .y = 2, .ys = 64, .n = 64, .bits = 4, .group = 32, .sdtype = 2, .bdtype = 2 };
    try gpu.dispatch(ctx, "mk_embed", &.{ q.w, q.s, q.b, e }, &ep, 2 * 64, 64);
    try ctx.finish();
    for (0..64) |c| {
        try close(q.dense[7 * 64 + c], e.floats()[2 + c], 1e-6);
        try close(q.dense[2 * 64 + c], e.floats()[2 + 64 + c], 1e-6);
    }
}

test "fused sandwich norm and gated conv equal their reference dispatch sequences bit for bit" {
    const ctx = try gpu.open();
    defer ctx.deinit();
    var r = rng(21);
    const n = 2048;
    const T = 3;
    const w = try ctx.alloc(n);
    defer w.deinit();
    const halves = std.mem.bytesAsSlice(u16, std.mem.sliceAsBytes(w.floats()));
    for (halves) |*h| h.* = bf16(1 + 0.2 * r.random().floatNorm(f32));
    // arena: y | r | t | out  (each T rows of n)
    var results: [2][]f32 = undefined;
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const init = try arena.allocator().alloc(f32, 2 * T * n);
    for (init) |*v| v.* = r.random().floatNorm(f32) * 2;
    for (0..2) |mode| {
        const buf = try ctx.alloc(4 * T * n);
        defer buf.deinit();
        @memcpy(buf.floats()[0 .. 2 * T * n], init);
        if (mode == 0) {
            var p: gpu.Params = .{ .rows = T, .x = 0, .xs = n, .y = 2 * T * n, .ys = n, .n = n, .heads = 1, .dtype = 2, .flags = gpu.flag_has_weight, .eps = 1e-5 };
            try gpu.dispatch(ctx, "mk_rmsnorm", &.{ w, buf }, &p, T * 512, 512);
            p = .{ .rows = T, .x = T * n, .xs = n, .y = 2 * T * n, .ys = n, .z = T * n, .zs = n, .n = n, .every = 1, .kind = 0 };
            try gpu.dispatch(ctx, "mk_binary", &.{buf}, &p, T * n, 256);
            p = .{ .rows = T, .x = T * n, .xs = n, .y = 3 * T * n, .ys = n, .n = n, .heads = 1, .w = 2048, .dtype = 2, .flags = gpu.flag_has_weight, .eps = 1e-5 };
            try gpu.dispatch(ctx, "mk_rmsnorm", &.{ w, buf }, &p, T * 512, 512);
        } else {
            const p: gpu.Params = .{ .rows = T, .x = 0, .xs = n, .y = T * n, .ys = n, .z = 3 * T * n, .zs = n, .n = n, .w = 0, .dtype = 2, .s = 2048, .sdtype = 2, .eps = 1e-5 };
            try gpu.dispatch(ctx, "mk_norm_add_norm", &.{ w, buf }, &p, T * 512, 512);
        }
        try ctx.finish();
        results[mode] = try arena.allocator().dupe(f32, buf.floats()[T * n .. 2 * T * n]);
        results[mode] = try std.mem.concat(arena.allocator(), f32, &.{ results[mode], buf.floats()[3 * T * n .. 4 * T * n] });
    }
    try testing.expectEqualSlices(u32, std.mem.bytesAsSlice(u32, std.mem.sliceAsBytes(results[0])), std.mem.bytesAsSlice(u32, std.mem.sliceAsBytes(results[1])));

    // Gated conv: u = a*b -> ring -> conv -> c*v, over two batches.
    const C = 64;
    const taps = 4;
    const cap = taps - 1 + 8;
    const cw = try ctx.alloc(C * taps);
    defer cw.deinit();
    for (cw.floats()) |*v| v.* = r.random().floatNorm(f32);
    const src = try arena.allocator().alloc(f32, 3 * 8 * C);
    for (src) |*v| v.* = r.random().floatNorm(f32);
    // mode 2 (prefill-mlx, knob conv_rows): mk_gated_conv_store then mk_gated_conv_rows.
    var outs: [3][]f32 = undefined;
    for (0..3) |mode| {
        const ring = try ctx.alloc(cap * C);
        defer ring.deinit();
        const buf = try ctx.alloc(6 * 8 * C);
        defer buf.deinit();
        outs[mode] = try arena.allocator().alloc(f32, 2 * 5 * C);
        for (0..2) |batch| {
            const pos: u32 = @intCast(batch * 5);
            @memcpy(buf.floats()[0 .. 3 * 8 * C], src);
            // a | b | c | u | v | y, row stride C, 5 rows.
            if (mode == 0) {
                var p: gpu.Params = .{ .rows = 5, .pos = pos, .x = 0, .xs = C, .y = 8 * C, .ys = C, .z = 3 * 8 * C, .zs = C, .n = C, .every = 1, .kind = 1 };
                try gpu.dispatch(ctx, "mk_binary", &.{buf}, &p, 5 * C, 64);
                p = .{ .rows = 5, .pos = pos, .x = 3 * 8 * C, .xs = C, .n = C, .capacity = cap };
                try gpu.dispatch(ctx, "mk_store", &.{ buf, ring }, &p, 5 * C, 64);
                p = .{ .rows = 5, .pos = pos, .y = 4 * 8 * C, .ys = C, .n = C, .taps = taps, .capacity = cap };
                try gpu.dispatch(ctx, "mk_conv", &.{ ring, cw, buf }, &p, 5 * C, 64);
                p = .{ .rows = 5, .pos = pos, .x = 2 * 8 * C, .xs = C, .y = 4 * 8 * C, .ys = C, .z = 5 * 8 * C, .zs = C, .n = C, .every = 1, .kind = 1 };
                try gpu.dispatch(ctx, "mk_binary", &.{buf}, &p, 5 * C, 64);
            } else {
                const p: gpu.Params = .{ .rows = 5, .pos = pos, .x = 0, .xs = C, .y = 8 * C, .ys = C, .z = 2 * 8 * C, .zs = C, .m = 5 * 8 * C, .every = C, .n = C, .taps = taps, .capacity = cap, .kind = 0 };
                if (mode == 1) {
                    try gpu.dispatch(ctx, "mk_gated_conv", &.{ ring, cw, buf }, &p, C, 64);
                } else {
                    try gpu.dispatch(ctx, "mk_gated_conv_store", &.{ ring, buf }, &p, 5 * C, 256);
                    try gpu.dispatch(ctx, "mk_gated_conv_rows", &.{ ring, cw, buf }, &p, 5 * C, 256);
                }
            }
            try ctx.finish();
            @memcpy(outs[mode][batch * 5 * C ..][0 .. 5 * C], buf.floats()[5 * 8 * C ..][0 .. 5 * C]);
        }
    }
    try testing.expectEqualSlices(u32, std.mem.bytesAsSlice(u32, std.mem.sliceAsBytes(outs[0])), std.mem.bytesAsSlice(u32, std.mem.sliceAsBytes(outs[1])));
    try testing.expectEqualSlices(u32, std.mem.bytesAsSlice(u32, std.mem.sliceAsBytes(outs[0])), std.mem.bytesAsSlice(u32, std.mem.sliceAsBytes(outs[2])));
}

test "4-bit linear and GLU tiles: a weight 2 bytes off a word boundary gives the aligned results bit for bit" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const ctx = try gpu.open();
    defer ctx.deinit();
    var r = rng(31);
    const m = 12;
    const n = 256;
    const group = 64;
    const T = 3;
    const q = try quantize(ctx, a, &r, m, n, group);
    defer {
        q.w.deinit();
        q.s.deinit();
        q.b.deinit();
    }
    const wbytes = m * n / 2;
    const sbytes = m * n / group * 2;
    // One buffer, LFM3-style: [2 pad bytes][codes][scales][biases], all at offsets = 2 mod 4.
    const packed_buf = try ctx.alloc((2 + wbytes + 2 * sbytes) / 4 + 1);
    defer packed_buf.deinit();
    const bytes = std.mem.sliceAsBytes(packed_buf.floats());
    @memcpy(bytes[2..][0..wbytes], std.mem.sliceAsBytes(q.w.floats())[0..wbytes]);
    @memcpy(bytes[2 + wbytes ..][0..sbytes], std.mem.sliceAsBytes(q.s.floats())[0..sbytes]);
    @memcpy(bytes[2 + wbytes + sbytes ..][0..sbytes], std.mem.sliceAsBytes(q.b.floats())[0..sbytes]);
    const buf = try ctx.alloc(T * n + 4 * T * m);
    defer buf.deinit();
    const f = buf.floats();
    for (f[0 .. T * n]) |*v| v.* = r.random().floatNorm(f32);
    var p: gpu.Params = .{ .rows = T, .xs = n, .y = T * n, .ys = m, .n = n, .m = m, .bits = 4, .group = group, .sdtype = 2, .bdtype = 2 };
    try gpu.dispatch(ctx, "mk_linear_q4_r2_t2", &.{ q.w, q.s, q.b, buf }, &p, ((m + 1) / 2) * ((T + 1) / 2) * 32, 64);
    p.y = T * n + T * m;
    p.w = 2;
    p.s = 2 + wbytes;
    p.b = 2 + wbytes + sbytes;
    try gpu.dispatch(ctx, "mk_linear_q4_r2_t2", &.{ packed_buf, packed_buf, packed_buf, buf }, &p, ((m + 1) / 2) * ((T + 1) / 2) * 32, 64);
    try ctx.finish();
    const out = std.mem.bytesAsSlice(u32, std.mem.sliceAsBytes(f[T * n ..][0 .. 2 * T * m]));
    try testing.expectEqualSlices(u32, out[0 .. T * m], out[T * m .. 2 * T * m]);
    // GLU with the same matrix twice, misaligned, equals silu(Wx) * (Wx) from the aligned linear.
    var x: gpu.ParamsX = .{ .p = .{ .rows = T, .xs = n, .y = T * n + 2 * T * m, .ys = m, .n = n, .group = group }, .count = 2, .act_a = 1, .kind = 1 };
    x.mat[0] = .{ .w = 2, .s = 2 + wbytes, .b = 2 + wbytes + sbytes, .m = m, .sdtype = 2, .bdtype = 2 };
    x.mat[1] = x.mat[0];
    try gpu.dispatch(ctx, "mk_glu_q4_r2_t1", &.{ packed_buf, buf }, &x, ((m + 1) / 2) * T * 32, 64);
    var b: gpu.Params = .{ .rows = T, .x = T * n, .xs = m, .y = T * n, .ys = m, .z = T * n + 3 * T * m, .zs = m, .n = m, .every = 1, .kind = 1, .act_a = 1 };
    try gpu.dispatch(ctx, "mk_binary", &.{buf}, &b, T * m, 64);
    try ctx.finish();
    const g = std.mem.bytesAsSlice(u32, std.mem.sliceAsBytes(f[T * n + 2 * T * m ..][0 .. 2 * T * m]));
    try testing.expectEqualSlices(u32, g[T * m .. 2 * T * m], g[0 .. T * m]);
}

// ---- prefill objective (prefill search) ----
test "matrix-matrix 4-bit linear and GLU (prep + mm): every compiled tile equals the one-row reference bit for bit" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const ctx = try gpu.open();
    defer ctx.deinit();
    const Knobs = @import("knobs.zig").Knobs;
    var r = rng(47);
    // n = 96: a partial 32-segment step; 1536: 3 full steps; 2048 at LFM3's group 64.
    for ([_][3]usize{ .{ 7, 96, 32 }, .{ 13, 2048, 64 }, .{ 9, 1536, 64 } }) |shape| {
        const m = shape[0];
        const n = shape[1];
        const group = shape[2];
        const T = 19; // not a multiple of any token tile
        const row0 = 2; // a dispatch that starts inside the batch, as head rows do
        const rows = T - row0;
        const qs = [2]Quant{ try quantize(ctx, a, &r, m, n, group), try quantize(ctx, a, &r, m, n, group) };
        defer for (qs) |q| {
            q.w.deinit();
            q.s.deinit();
            q.b.deinit();
        };
        const wbytes = m * n / 2;
        const sbytes = m * n / group * 2;
        // Both matrices in one buffer at offsets = 2 mod 4, as in the LFM3 checkpoint.
        const mat_bytes = wbytes + 2 * sbytes;
        const packed_buf = try ctx.alloc((2 + 2 * mat_bytes) / 4 + 1);
        defer packed_buf.deinit();
        const bytes = std.mem.sliceAsBytes(packed_buf.floats());
        var mats: [2]gpu.Mat = undefined;
        for (qs, 0..) |q, k| {
            const o = 2 + k * mat_bytes;
            @memcpy(bytes[o..][0..wbytes], std.mem.sliceAsBytes(q.w.floats())[0..wbytes]);
            @memcpy(bytes[o + wbytes ..][0..sbytes], std.mem.sliceAsBytes(q.s.floats())[0..sbytes]);
            @memcpy(bytes[o + wbytes + sbytes ..][0..sbytes], std.mem.sliceAsBytes(q.b.floats())[0..sbytes]);
            mats[k] = .{ .w = o, .s = o + wbytes, .b = o + wbytes + sbytes, .m = @intCast(m), .sdtype = 2, .bdtype = 2 };
        }
        // arena: x [T, n], then outputs [T, m] each: ref0, ref1, ref glu, mm, mm glu.
        const out0 = T * n;
        const buf = try ctx.alloc(T * n + 5 * T * m);
        defer buf.deinit();
        const f = buf.floats();
        for (f[0 .. T * n]) |*v| v.* = r.random().floatNorm(f32);
        // Reference: one row per dispatch with mk_linear_q4_r1_t1, then mk_binary for silu(g) * u.
        for (0..2) |k| for (row0..T) |t| {
            var p: gpu.Params = .{ .rows = 1, .row0 = @intCast(t), .xs = @intCast(n), .y = @intCast(out0 + k * T * m), .ys = @intCast(m), .n = @intCast(n), .m = @intCast(m), .bits = 4, .group = @intCast(group), .w = mats[k].w, .s = mats[k].s, .b = mats[k].b, .sdtype = 2, .bdtype = 2 };
            try gpu.dispatch(ctx, "mk_linear_q4_r1_t1", &.{ packed_buf, packed_buf, packed_buf, buf }, &p, m * 32, 32);
        };
        var bp: gpu.Params = .{ .rows = rows, .row0 = row0, .x = @intCast(out0), .xs = @intCast(m), .y = @intCast(out0 + T * m), .ys = @intCast(m), .z = @intCast(out0 + 2 * T * m), .zs = @intCast(m), .n = @intCast(m), .every = 1, .kind = 1, .act_a = 1 };
        try gpu.dispatch(ctx, "mk_binary", &.{buf}, &bp, rows * m, 64);
        try ctx.finish();
        for (row0..T) |t| for (0..m) |row| {
            var sum: f64 = 0;
            for (0..n) |c| sum += @as(f64, qs[0].dense[row * n + c]) * f[t * n + c];
            try close(sum, f[out0 + t * m + row], 1e-4);
        };
        const segs = n / 16;
        const padded = (segs + 31) / 32 * 512;
        const prep = try ctx.alloc(rows * (padded + segs));
        defer prep.deinit();
        var pp: gpu.Params = .{ .rows = rows, .row0 = row0, .xs = @intCast(n), .n = @intCast(n), .z = @intCast(rows * padded) };
        try gpu.dispatch(ctx, "mk_q4_prep", &.{ buf, prep }, &pp, rows * segs, 64);
        for (Knobs.mm_tiles) |tile| {
            const sfx: []const u8 = if (tile[2]) "h" else "f";
            const simds = ((m + tile[0] - 1) / tile[0]) * ((rows + tile[1] - 1) / tile[1]);
            @memset(f[out0 + 3 * T * m ..][0 .. 2 * T * m], 0);
            var p: gpu.Params = .{ .rows = rows, .row0 = row0, .y = @intCast(out0 + 3 * T * m), .ys = @intCast(m), .z = @intCast(rows * padded), .n = @intCast(n), .m = @intCast(m), .bits = 4, .group = @intCast(group), .w = mats[0].w, .s = mats[0].s, .b = mats[0].b, .sdtype = 2, .bdtype = 2 };
            const name = try std.fmt.allocPrintSentinel(a, "mk_mm_q4_r{d}_t{d}_{s}", .{ tile[0], tile[1], sfx }, 0);
            try gpu.dispatch(ctx, name, &.{ packed_buf, packed_buf, packed_buf, buf, prep }, &p, simds * 32, 128);
            var x: gpu.ParamsX = .{ .p = .{ .rows = rows, .row0 = row0, .y = @intCast(out0 + 4 * T * m), .ys = @intCast(m), .z = @intCast(rows * padded), .n = @intCast(n), .group = @intCast(group) }, .count = 2, .act_a = 1, .kind = 1 };
            x.mat[0] = mats[0];
            x.mat[1] = mats[1];
            const gname = try std.fmt.allocPrintSentinel(a, "mk_mm_glu_q4_r{d}_t{d}_{s}", .{ tile[0], tile[1], sfx }, 0);
            try gpu.dispatch(ctx, gname, &.{ packed_buf, buf, prep }, &x, simds * 32, 64);
            try ctx.finish();
            const words = std.mem.bytesAsSlice(u32, std.mem.sliceAsBytes(f));
            const lo = out0 + row0 * m;
            const len = rows * m;
            try testing.expectEqualSlices(u32, words[lo..][0..len], words[lo + 3 * T * m ..][0..len]);
            try testing.expectEqualSlices(u32, words[lo + 2 * T * m ..][0..len], words[lo + 4 * T * m ..][0..len]);
        }
    }
}

// ---- decode search ----

fn bitsEqual(expected: []const f32, actual: []const f32) !void {
    try testing.expectEqualSlices(u32, std.mem.bytesAsSlice(u32, std.mem.sliceAsBytes(expected)), std.mem.bytesAsSlice(u32, std.mem.sliceAsBytes(actual)));
}

test "split decode attention (scores + block chain) equals mk_attention and mk_attention_gated bit for bit" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const ctx = try gpu.open();
    defer ctx.deinit();
    var r = rng(77);
    // (heads, kv heads, dim, window): LFM3's full and sliding shapes, and others.
    for ([_][4]u32{ .{ 16, 4, 128, 0 }, .{ 16, 4, 128, 37 }, .{ 8, 1, 64, 0 }, .{ 4, 4, 32, 9 }, .{ 2, 1, 256, 0 }, .{ 4, 2, 16, 5 }, .{ 4, 1, 40, 0 } }) |shape| {
        const heads = shape[0];
        const kvh = shape[1];
        const dim = shape[2];
        const window = shape[3];
        const positions: u32 = if (window == 0) 700 else 140;
        const capacity: u32 = if (window == 0) positions else window - 1 + 4;
        const span: u32 = if (window == 0) capacity else window;
        const K = try ctx.alloc(capacity * kvh * dim);
        defer K.deinit();
        const V = try ctx.alloc(capacity * kvh * dim);
        defer V.deinit();
        const S = try ctx.alloc(heads * span);
        defer S.deinit();
        const qw = heads * dim;
        // arena: q | gate | ref out | split out | gated ref | gated split
        const buf = try ctx.alloc(qw + heads + 4 * qw);
        defer buf.deinit();
        const f = buf.floats();
        const E = (dim + 31) / 32;
        const G = heads / kvh;
        const name = try std.fmt.allocPrintSentinel(a, "mk_dec_scores_e{d}_g{d}", .{ E, G }, 0);
        const last = try a.alloc(f32, qw);
        var pos: u32 = 0;
        while (pos < positions) : (pos += 1) {
            const slot = pos % capacity;
            for (K.floats()[slot * kvh * dim ..][0 .. kvh * dim]) |*v| v.* = 2 * r.random().floatNorm(f32);
            for (V.floats()[slot * kvh * dim ..][0 .. kvh * dim]) |*v| v.* = r.random().floatNorm(f32);
            for (f[0 .. qw + heads]) |*v| v.* = 2 * r.random().floatNorm(f32);
            // Sparse checks keep the test quick; every position near the window edge and wrap.
            if (pos > 20 and pos % 7 != 0 and pos + 1 != positions and !(window != 0 and pos % capacity < 3)) continue;
            var p: gpu.Params = .{ .rows = 1, .pos = pos, .x = 0, .xs = qw, .ys = qw, .heads = heads, .kv_heads = kvh, .dim = dim, .window = window, .capacity = capacity, .scale = 0.088388, .z = qw, .zs = heads, .act_b = 2 };
            p.y = qw + heads;
            try gpu.dispatch(ctx, "mk_attention", &.{ buf, K, V }, &p, heads * 256, 256);
            p.y = qw + heads + 2 * qw;
            try gpu.dispatch(ctx, "mk_attention_gated", &.{ buf, K, V }, &p, heads * 256, 256);
            const start: u32 = if (window != 0 and pos + 1 > window) pos + 1 - window else 0;
            const chunks = (pos - start + 1 + 7) / 8;
            p.every = span;
            p.m = chunks;
            try gpu.dispatch(ctx, name, &.{ buf, K, S }, &p, kvh * chunks * 32, 256);
            const o = qw + heads;
            p.y = qw + heads + qw;
            try gpu.dispatch(ctx, "mk_dec_chain_block", &.{ buf, V, S }, &p, heads * E * 256, 256);
            p.flags = gpu.flag_dec_gated;
            p.y = qw + heads + 3 * qw;
            try gpu.dispatch(ctx, "mk_dec_chain_block", &.{ buf, V, S }, &p, heads * E * 256, 256);
            try ctx.finish();
            try bitsEqual(f[o..][0..qw], f[o + qw ..][0..qw]);
            try bitsEqual(f[o + 2 * qw ..][0..qw], f[o + 3 * qw ..][0..qw]);
            if (pos + 1 == positions) @memcpy(last, f[o + qw ..][0..qw]);
        }
        // Plain CPU softmax attention at the last position (float64).
        const lp = positions - 1;
        const start: u32 = if (window != 0 and lp + 1 > window) lp + 1 - window else 0;
        for (0..heads) |h| {
            const g = h / G;
            const scores = try a.alloc(f64, lp + 1 - start);
            var mx: f64 = -std.math.inf(f64);
            for (start..lp + 1, scores) |t, *sc| {
                var d: f64 = 0;
                for (0..dim) |e| d += @as(f64, @as(f32, 0.088388) * f[h * dim + e]) * K.floats()[((t % capacity) * kvh + g) * dim + e];
                sc.* = d;
                mx = @max(mx, d);
            }
            var total: f64 = 0;
            for (scores) |sc| total += @exp(sc - mx);
            for (0..dim) |e| {
                var o: f64 = 0;
                for (start..lp + 1, scores) |t, sc| o += @exp(sc - mx) / total * V.floats()[((t % capacity) * kvh + g) * dim + e];
                try close(o, last[h * dim + e], 1e-4);
            }
        }
    }
}

test "fused attention prep equals rmsnorm x2, rope x2, store x2 bit for bit" {
    const ctx = try gpu.open();
    defer ctx.deinit();
    var r = rng(123);
    // (heads, kv heads, dim, rotated pairs); rows 1 and 3; a ring that wraps.
    for ([_][4]u32{ .{ 16, 4, 128, 32 }, .{ 16, 4, 128, 64 }, .{ 4, 2, 16, 4 }, .{ 8, 8, 64, 16 } }) |shape| {
        const heads = shape[0];
        const kvh = shape[1];
        const dim = shape[2];
        const qw = heads * dim;
        const kw = kvh * dim;
        const capacity: u32 = 7;
        const wbuf = try ctx.alloc(dim); // two bf16 weights of dim each
        defer wbuf.deinit();
        const halves = std.mem.bytesAsSlice(u16, std.mem.sliceAsBytes(wbuf.floats()));
        for (halves) |*h| h.* = bf16(0.5 + r.random().floatNorm(f32) * 0.3);
        const tables = try ctx.alloc(dim / 2);
        defer tables.deinit();
        for (tables.floats(), 0..) |*t, j| t.* = if (j < shape[3]) std.math.pow(f32, 10000, -@as(f32, @floatFromInt(2 * j)) / @as(f32, @floatFromInt(dim))) else 0;
        for ([_]u32{ 1, 3 }) |rows| {
            var caches: [2][2]metal.Buffer = undefined;
            var arenas: [2]metal.Buffer = undefined;
            for (0..2) |m| {
                caches[m][0] = try ctx.alloc(capacity * kw);
                caches[m][1] = try ctx.alloc(capacity * kw);
                @memset(caches[m][0].floats(), 0);
                @memset(caches[m][1].floats(), 0);
                arenas[m] = try ctx.alloc(rows * (qw + 2 * kw));
            }
            defer for (0..2) |m| {
                caches[m][0].deinit();
                caches[m][1].deinit();
                arenas[m].deinit();
            };
            var pos: u32 = 0;
            while (pos < 20) : (pos += rows) {
                for (arenas[0].floats()) |*v| v.* = 3 * r.random().floatNorm(f32);
                @memcpy(arenas[1].floats(), arenas[0].floats());
                const qo: u32 = 0;
                const ko: u32 = rows * qw;
                const vo: u32 = rows * (qw + kw);
                // Reference sequence on arena 0.
                var p: gpu.Params = .{ .rows = rows, .pos = pos, .eps = 1e-5, .flags = gpu.flag_has_weight, .dtype = 2 };
                p.x = qo;
                p.xs = qw;
                p.y = qo;
                p.ys = qw;
                p.n = dim;
                p.heads = heads;
                const tg = gpu.normThreads(dim);
                try gpu.dispatch(ctx, "mk_rmsnorm", &.{ wbuf, arenas[0] }, &p, rows * heads * tg, tg);
                p.x = ko;
                p.xs = kw;
                p.y = ko;
                p.ys = kw;
                p.heads = kvh;
                p.w = dim; // second weight: bf16 halves dim/2 .. dim (byte offset dim)
                try gpu.dispatch(ctx, "mk_rmsnorm", &.{ wbuf, arenas[0] }, &p, rows * kvh * tg, tg);
                var rp: gpu.Params = .{ .rows = rows, .pos = pos, .x = qo, .xs = qw, .heads = heads, .dim = dim, .rot = dim, .z = 0 };
                try gpu.dispatch(ctx, "mk_rope", &.{ arenas[0], tables }, &rp, rows * heads * (dim / 2), 64);
                rp.x = ko;
                rp.xs = kw;
                rp.heads = kvh;
                try gpu.dispatch(ctx, "mk_rope", &.{ arenas[0], tables }, &rp, rows * kvh * (dim / 2), 64);
                var sp: gpu.Params = .{ .rows = rows, .pos = pos, .x = ko, .xs = kw, .n = kw, .capacity = capacity };
                try gpu.dispatch(ctx, "mk_store", &.{ arenas[0], caches[0][0] }, &sp, rows * kw, 256);
                sp.x = vo;
                try gpu.dispatch(ctx, "mk_store", &.{ arenas[0], caches[0][1] }, &sp, rows * kw, 256);
                // Fused on arena 1.
                var fp: gpu.Params = .{ .rows = rows, .pos = pos, .x = qo, .xs = qw, .y = ko, .ys = kw, .z = vo, .zs = kw, .heads = heads, .kv_heads = kvh, .dim = dim, .rot = dim, .n = 0, .eps = 1e-5, .w = 0, .dtype = 2, .s = dim, .sdtype = 2, .capacity = capacity };
                try gpu.dispatch(ctx, "mk_dec_attn_prep", &.{ wbuf, wbuf, arenas[1], tables, caches[1][0], caches[1][1] }, &fp, rows * (heads + 2 * kvh) * tg, tg);
                try ctx.finish();
                try bitsEqual(arenas[0].floats()[0 .. rows * (qw + kw)], arenas[1].floats()[0 .. rows * (qw + kw)]);
                try bitsEqual(caches[0][0].floats(), caches[1][0].floats());
                try bitsEqual(caches[0][1].floats(), caches[1][1].floats());
            }
        }
    }
}

test "two-stage argmax equals mk_argmax: ties, signed zeros, -inf rows, non-finite sentinel" {
    const ctx = try gpu.open();
    defer ctx.deinit();
    var r = rng(31);
    const n = 128000;
    const rows = 3;
    const groups = 64;
    const buf = try ctx.alloc(rows * n + 2 * rows);
    defer buf.deinit();
    const part = try ctx.alloc(rows * groups * 3);
    defer part.deinit();
    const f = buf.floats();
    for (0..12) |case| {
        for (f[0 .. rows * n]) |*v| v.* = r.random().floatNorm(f32);
        switch (case) {
            1 => for (0..rows) |row| {
                f[row * n + 777] = 9;
                f[row * n + 100000] = 9;
                f[row * n + 5] = 9;
            },
            2 => for (f[0 .. rows * n]) |*v| {
                v.* = if (v.* > 0) @as(f32, 0) else @as(f32, -0.0);
            },
            3 => @memset(f[0 .. rows * n], -std.math.inf(f32)),
            4 => f[n + 64000] = std.math.nan(f32),
            5 => f[2 * n + 3] = std.math.inf(f32),
            6 => for (f[0 .. rows * n]) |*v| {
                v.* = @round(v.* * 2);
            },
            else => {},
        }
        var p: gpu.Params = .{ .rows = rows, .x = 0, .xs = n, .y = rows * n, .ys = 1, .n = n };
        try gpu.dispatch(ctx, "mk_argmax", &.{buf}, &p, rows * 1024, 1024);
        p.y = rows * n + rows;
        p.every = groups;
        p.m = (n + groups - 1) / groups;
        try gpu.dispatch(ctx, "mk_dec_argmax_part", &.{ buf, part }, &p, rows * groups * 256, 256);
        try gpu.dispatch(ctx, "mk_dec_argmax_final", &.{ buf, part }, &p, rows * 64, 64);
        try ctx.finish();
        try bitsEqual(f[rows * n ..][0..rows], f[rows * n + rows ..][0..rows]);
        // CPU: lowest index of the maximum, sentinel on any non-finite value.
        for (0..rows) |row| {
            var best: f32 = -std.math.inf(f32);
            var idx: u32 = 0xffffffff;
            var bad = false;
            for (f[row * n ..][0..n], 0..) |v, i| {
                if (!std.math.isFinite(v)) bad = true;
                if (v > best) {
                    best = v;
                    idx = @intCast(i);
                }
            }
            const want: u32 = if (bad) 0xffffffff else idx;
            try testing.expectEqual(want, @as(u32, @bitCast(f[rows * n + rows + row])));
        }
    }
}

// ---- prefill-mlx (mlx-arith tier) ----
// Bit identity with stock MLX is tools/nano_mlx_ops.py's (it needs MLX); here: the
// kernels compute the product (CPU, float64, loose bound: MLX's order is not the
// reference's), and the multi-matrix launch equals one launch per matrix bit for bit.
test "mlx qmm port: plain and split-K match CPU; mk_mlx_qmm_multi equals one launch per matrix bit for bit" {
    const exec = @import("exec.zig");
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const ctx = try gpu.open();
    defer ctx.deinit();
    var r = rng(77);
    const K: u32 = 256;
    const group: u32 = 64;
    const ns = [_]u32{ 96, 40, 64 };
    // One weight buffer, data 2 bytes off a word boundary like LFM3's: per matrix
    // codes [N, K/2 bytes], bf16 scales and biases [N, K/group].
    var total: usize = 2;
    for (ns) |n| total += n * K / 2 + 2 * (n * K / group) * 2;
    const wbuf = try ctx.alloc((total + 3) / 4);
    defer wbuf.deinit();
    const bytes = wbuf.structs(u8);
    var mats: [3]gpu.Mat = undefined;
    var dense: [3][]f32 = undefined;
    var off: usize = 2;
    var yoff: u32 = 0;
    const M: u32 = 70; // two full 32-row tiles and a ragged one
    for (ns, 0..) |n, j| {
        dense[j] = try a.alloc(f32, n * K);
        const w_off = off;
        const s_off = w_off + n * K / 2;
        const b_off = s_off + (n * K / group) * 2;
        off = b_off + (n * K / group) * 2;
        for (0..n * K / group) |g| {
            std.mem.writeInt(u16, bytes[s_off + 2 * g ..][0..2], bf16(0.01 + 0.05 * r.random().float(f32)), .little);
            std.mem.writeInt(u16, bytes[b_off + 2 * g ..][0..2], bf16(-0.3 + 0.1 * r.random().float(f32)), .little);
        }
        for (0..n * K / 2) |i| bytes[w_off + i] = r.random().int(u8);
        for (0..n * K) |i| {
            const code: u32 = (bytes[w_off + i / 2] >> @intCast((i % 2) * 4)) & 15;
            const g = i / group;
            dense[j][i] = @as(f32, @floatFromInt(code)) * fromBf16(std.mem.readInt(u16, bytes[s_off + 2 * g ..][0..2], .little)) +
                fromBf16(std.mem.readInt(u16, bytes[b_off + 2 * g ..][0..2], .little));
        }
        mats[j] = .{ .w = w_off, .s = s_off, .b = b_off, .m = n, .y = M * K + yoff, .ys = n, .sdtype = 2, .bdtype = 2 };
        yoff += M * n;
    }
    // arena: x [M, K] | one launch per matrix | multi-matrix launch
    const buf = try ctx.alloc(M * K + 2 * yoff);
    defer buf.deinit();
    const splitk = try ctx.alloc(exec.mlx_splitk_words);
    defer splitk.deinit();
    const f = buf.floats();
    for (f[0 .. M * K]) |*v| v.* = r.random().floatNorm(f32);
    // The unsplit kernel directly (at these small shapes MLX itself would split K).
    for (mats) |mt| {
        const p: gpu.Params = .{ .rows = M, .x = 0, .xs = K, .y = mt.y, .ys = mt.ys, .n = K, .m = mt.m, .w = mt.w, .s = mt.s, .b = mt.b, .sdtype = 2, .bdtype = 2, .bits = 4, .group = group, .kind = 1, .dim = K };
        try gpu.dispatch(ctx, "mk_mlx_qmm_t_bf16_g64", &.{ wbuf, wbuf, wbuf, buf, splitk }, &p, ((mt.m + 31) / 32) * ((M + 31) / 32) * 128, 128);
    }
    var x: gpu.ParamsX = .{ .p = .{ .rows = M, .x = 0, .xs = K, .n = K, .group = group }, .count = 3 };
    var tgs: usize = 0;
    for (mats, 0..) |mt, j| {
        x.mat[j] = mt;
        x.mat[j].y += yoff;
        tgs += ((mt.m + 31) / 32) * ((M + 31) / 32);
    }
    try gpu.dispatch(ctx, "mk_mlx_qmm_multi_bf16_g64", &.{ wbuf, buf }, &x, tgs * 128, 128);
    try ctx.finish();
    try bitsEqual(f[M * K ..][0..yoff], f[M * K + yoff ..][0..yoff]);
    for (mats, 0..) |mt, j| for (0..M) |t| for (0..mt.m) |o| {
        var acc: f64 = 0;
        for (0..K) |k| acc += @as(f64, f[t * K + k]) * dense[j][o * K + k];
        try close(acc, f[mt.y + t * mt.m + o], 1e-4);
    };
    // Split K (MLX qmm_t_splitk + gemm_splitk_accum): 40 x 40 tiles give 4 partitions of K.
    const Ms: u32 = 40;
    var p: gpu.Params = .{ .rows = Ms, .x = 0, .xs = K, .y = M * K, .ys = 40, .n = K, .m = 40, .w = mats[1].w, .s = mats[1].s, .b = mats[1].b, .sdtype = 2, .bdtype = 2, .bits = 4, .group = group };
    try testing.expectEqual(@as(u32, 4), try exec.mlxLinear(ctx, .{ wbuf, wbuf, wbuf }, buf, splitk, &p));
    try ctx.finish();
    for (0..Ms) |t| for (0..40) |o| {
        var acc: f64 = 0;
        for (0..K) |k| acc += @as(f64, f[t * K + k]) * dense[1][o * K + k];
        try close(acc, f[M * K + t * 40 + o], 1e-4);
    };
}

// ---- c2-prefill-mlx (prefill, mlx-arith tier) ----
// The restructured kernels of mlx2.metal against the kernels they replace, bit for bit,
// on random and adversarial inputs (signed zeros, subnormals, infinities, NaN, heavy
// cancellation, ragged sizes, weights 2 bytes off a word boundary). Bit identity of
// the replaced kernels with stock MLX is tools/nano_mlx_ops.py's.

/// Random normal values, or (adversarial) a mix of +-0, subnormals, +-inf, NaN,
/// cancelling +-1e30 pairs and ordinary values.
fn c2Fill(r: *std.Random.DefaultPrng, out: []f32, adversarial: bool, scale: f32) void {
    const special = [_]f32{ 0.0, -0.0, std.math.floatTrueMin(f32), -std.math.floatTrueMin(f32), 1e-39, -3e-40, std.math.inf(f32), -std.math.inf(f32), std.math.nan(f32), 1e30, -1e30, 65504.0, -1.0e-30 };
    for (out) |*v| {
        v.* = scale * r.random().floatNorm(f32);
        if (adversarial and r.random().uintLessThan(u32, 8) == 0) v.* = special[r.random().uintLessThan(usize, special.len)];
    }
}

test "c2-prefill-mlx: mk_mlx2_qmm, mk_mlx2_multi and mk_mlx2_glu equal mk_mlx_qmm_t (and mk_binary) bit for bit" {
    const ctx = try gpu.open();
    defer ctx.deinit();
    var r = rng(2026);
    const K: u32 = 256;
    const group: u32 = 64;
    const ns = [_]u32{ 96, 40, 40 }; // the last two are a GLU's gate and up (same width)
    for ([_]bool{ false, true }) |adversarial| {
        var total: usize = 2;
        for (ns) |n| total += n * K / 2 + 2 * (n * K / group) * 2;
        const wbuf = try ctx.alloc((total + 3) / 4);
        defer wbuf.deinit();
        const bytes = wbuf.structs(u8);
        var mats: [3]gpu.Mat = undefined;
        var off: usize = 2; // data 2 bytes off a word boundary, as LFM3's
        const M: u32 = 70; // a full 64-row tile and a ragged one; 40 outputs: 2.5 tiles of 16
        var yoff: u32 = M * K;
        for (ns, 0..) |n, j| {
            const w_off = off;
            const s_off = w_off + n * K / 2;
            const b_off = s_off + (n * K / group) * 2;
            off = b_off + (n * K / group) * 2;
            for (0..n * K / group) |g| {
                var s = bf16(0.01 + 0.05 * r.random().float(f32));
                var b = bf16(-0.3 + 0.1 * r.random().float(f32));
                if (adversarial) switch (g % 9) {
                    0 => {
                        s = 0;
                        b = 0x8000;
                    }, // +0 scale, -0 bias
                    1 => s = 0x0001, // subnormal scale
                    2 => s = bf16(-0.04), // negative scale
                    3 => b = bf16(1e30), // cancelling bias
                    4 => s = 0x7f80, // +inf scale
                    5 => b = 0x7fc0, // NaN bias
                    else => {},
                };
                std.mem.writeInt(u16, bytes[s_off + 2 * g ..][0..2], s, .little);
                std.mem.writeInt(u16, bytes[b_off + 2 * g ..][0..2], b, .little);
            }
            for (0..n * K / 2) |i| bytes[w_off + i] = r.random().int(u8);
            mats[j] = .{ .w = w_off, .s = s_off, .b = b_off, .m = n, .y = yoff, .ys = n, .sdtype = 2, .bdtype = 2 };
            yoff += M * n;
        }
        // arena: x | reference per matrix | mlx2 per matrix | multi | glu reference | glu
        const per = yoff - M * K;
        const buf = try ctx.alloc(M * K + 3 * per + 2 * M * ns[1]);
        defer buf.deinit();
        const splitk = try ctx.alloc(64);
        defer splitk.deinit();
        const f = buf.floats();
        c2Fill(&r, f[0 .. M * K], adversarial, 1);
        for (mats) |mt| {
            var p: gpu.Params = .{ .rows = M, .x = 0, .xs = K, .y = mt.y, .ys = mt.ys, .n = K, .m = mt.m, .w = mt.w, .s = mt.s, .b = mt.b, .sdtype = 2, .bdtype = 2, .bits = 4, .group = group, .kind = 1, .dim = K };
            try gpu.dispatch(ctx, "mk_mlx_qmm_t_bf16_g64", &.{ wbuf, wbuf, wbuf, buf, splitk }, &p, ((mt.m + 31) / 32) * ((M + 31) / 32) * 128, 128);
            p.y += per;
            try gpu.dispatch(ctx, "mk_mlx2_qmm_bf16_g64", &.{ wbuf, wbuf, wbuf, buf }, &p, ((mt.m + 15) / 16) * ((M + 63) / 64) * 64, 64);
        }
        var x: gpu.ParamsX = .{ .p = .{ .rows = M, .x = 0, .xs = K, .n = K, .group = group }, .count = 3 };
        var tgs: usize = 0;
        for (mats, 0..) |mt, j| {
            x.mat[j] = mt;
            x.mat[j].y += 2 * per;
            tgs += ((mt.m + 15) / 16) * ((M + 63) / 64);
        }
        try gpu.dispatch(ctx, "mk_mlx2_multi_bf16_g64", &.{ wbuf, buf }, &x, tgs * 64, 64);
        // GLU: silu(gate) * up through mk_binary on the reference products, and fused.
        const g0 = M * K + 3 * per;
        const bp: gpu.Params = .{ .rows = M, .x = mats[1].y, .xs = ns[1], .y = mats[2].y, .ys = ns[1], .z = g0, .zs = ns[1], .n = ns[1], .every = 1, .kind = 1, .act_a = 1, .act_b = 0 };
        try gpu.dispatch(ctx, "mk_binary", &.{buf}, &bp, M * ns[1], 256);
        var gx: gpu.ParamsX = .{ .p = .{ .rows = M, .x = 0, .xs = K, .n = K, .group = group, .y = g0 + M * ns[1], .ys = ns[1] }, .count = 2, .act_a = 1, .act_b = 0, .kind = 1 };
        gx.mat[0] = mats[1];
        gx.mat[1] = mats[2];
        try gpu.dispatch(ctx, "mk_mlx2_glu_bf16_g64", &.{ wbuf, buf }, &gx, ((ns[1] + 15) / 16) * ((M + 63) / 64) * 64, 64);
        try ctx.finish();
        try bitsEqual(f[M * K ..][0..per], f[M * K + per ..][0..per]);
        try bitsEqual(f[M * K ..][0..per], f[M * K + 2 * per ..][0..per]);
        try bitsEqual(f[g0..][0 .. M * ns[1]], f[g0 + M * ns[1] ..][0 .. M * ns[1]]);
        if (!adversarial) for (f[M * K ..][0..per]) |v| try testing.expect(std.math.isFinite(v));
    }
}

test "c2-prefill-mlx: mk_mlx2_attention[_gated] equals mk_mlx_attention[_gated] bit for bit" {
    const ctx = try gpu.open();
    defer ctx.deinit();
    var r = rng(4242);
    // (T, KL, window, capacity): ragged, a later chunk, sliding windows in stock's key layout, ring wrap.
    for ([_][4]u32{ .{ 70, 70, 0, 70 }, .{ 40, 300, 0, 300 }, .{ 9, 9, 0, 9 }, .{ 64, 64, 256, 319 }, .{ 100, 700, 256, 355 }, .{ 300, 300, 256, 555 } }) |c| {
        for ([_]bool{ false, true }) |adversarial| {
            const T = c[0];
            const KL = c[1];
            const cap = c[3];
            const qw: u32 = 16 * 128;
            const kw: u32 = 4 * 128;
            const Kc = try ctx.alloc(cap * kw);
            defer Kc.deinit();
            const Vc = try ctx.alloc(cap * kw);
            defer Vc.deinit();
            // arena: q | gate | out | out2 | gated | gated2
            const buf = try ctx.alloc(T * qw + T * 16 + 4 * T * qw);
            defer buf.deinit();
            const f = buf.floats();
            c2Fill(&r, f[0 .. T * qw + T * 16], adversarial, 2);
            for (0..KL) |t| {
                const slot = t % cap;
                c2Fill(&r, Kc.floats()[slot * kw ..][0..kw], adversarial, 2);
                c2Fill(&r, Vc.floats()[slot * kw ..][0..kw], adversarial, 1);
            }
            const o = T * qw + T * 16;
            var p: gpu.Params = .{ .rows = T, .pos = KL - T, .x = 0, .xs = qw, .ys = qw, .heads = 16, .kv_heads = 4, .dim = 128, .window = c[2], .capacity = cap, .scale = 0.08838834764831845, .z = T * qw, .zs = 16, .act_b = 2 };
            const tgs = ((T + 31) / 32) * 16 * 128;
            for ([_][:0]const u8{ "mk_mlx_attention", "mk_mlx2_attention", "mk_mlx_attention_gated", "mk_mlx2_attention_gated" }, 0..) |name, k| {
                p.y = o + @as(u32, @intCast(k)) * T * qw;
                try gpu.dispatch(ctx, name, &.{ buf, Kc, Vc }, &p, tgs, 128);
            }
            try ctx.finish();
            try bitsEqual(f[o..][0 .. T * qw], f[o + T * qw ..][0 .. T * qw]);
            try bitsEqual(f[o + 2 * T * qw ..][0 .. T * qw], f[o + 3 * T * qw ..][0 .. T * qw]);
        }
    }
}

test "c2-prefill-mlx: mk_gated_conv_rows2 equals mk_gated_conv_rows bit for bit" {
    const ctx = try gpu.open();
    defer ctx.deinit();
    var r = rng(99);
    const C = 64;
    for ([_]u32{ 4, 7 }) |taps| for ([_]bool{ false, true }) |adversarial| {
        const rows = 5;
        const cap = taps - 1 + rows;
        const cw = try ctx.alloc(C * taps / 2 + 1);
        defer cw.deinit();
        const halves = std.mem.bytesAsSlice(u16, std.mem.sliceAsBytes(cw.floats()));
        for (halves) |*h| h.* = bf16(r.random().floatNorm(f32));
        if (adversarial) {
            halves[0] = 0x8000; // -0
            halves[1] = 0x0001; // subnormal
        }
        var outs: [2][]f32 = undefined;
        var arena = std.heap.ArenaAllocator.init(testing.allocator);
        defer arena.deinit();
        const src = try arena.allocator().alloc(f32, 4 * 3 * rows * C);
        c2Fill(&r, src, adversarial, 1);
        for (0..2) |mode| {
            const ring = try ctx.alloc(cap * C);
            defer ring.deinit();
            const buf = try ctx.alloc(4 * rows * C);
            defer buf.deinit();
            outs[mode] = try arena.allocator().alloc(f32, 4 * rows * C);
            // Four batches of 5 rows from position 0: the first taps - 1 positions read
            // zeros before the start, later batches wrap the ring.
            for (0..4) |batch| {
                @memcpy(buf.floats()[0 .. 3 * rows * C], src[batch * 3 * rows * C ..][0 .. 3 * rows * C]);
                const p: gpu.Params = .{ .rows = rows, .pos = @intCast(batch * rows), .x = 0, .xs = C, .y = rows * C, .ys = C, .z = 2 * rows * C, .zs = C, .m = 3 * rows * C, .every = C, .n = C, .taps = taps, .capacity = cap, .kind = 1, .act_b = 1, .dtype = 2 };
                try gpu.dispatch(ctx, "mk_gated_conv_store", &.{ ring, buf }, &p, rows * C, 256);
                try gpu.dispatch(ctx, if (mode == 0) "mk_gated_conv_rows" else "mk_gated_conv_rows2", &.{ ring, cw, buf }, &p, rows * C, 256);
                try ctx.finish();
                @memcpy(outs[mode][batch * rows * C ..][0 .. rows * C], buf.floats()[3 * rows * C ..][0 .. rows * C]);
            }
        }
        try bitsEqual(outs[0], outs[1]);
    };
}

// ---- climb 2, exact prefill ----

const exact2 = @import("exact2.zig");

/// An adversarial float: signed zeros, subnormals, tiny normals, infinities, NaN, huge.
fn c2Special(r: *std.Random.DefaultPrng) f32 {
    const v = [_]f32{ 0.0, -0.0, std.math.inf(f32), -std.math.inf(f32), std.math.nan(f32), 1e30, -1e30, 1e-40, -1e-40, 3e38, -3e38 };
    return v[r.random().uintLessThan(usize, v.len)];
}

test "exact prefill (climb 2): matrix-unit 4-bit linear (mk_x2_prep + mk_x2mm_q4) equals the one-row reference bit for bit" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const ctx = try gpu.open();
    defer ctx.deinit();
    var r = rng(2026);
    // (m, n, group, scale dtype): ragged rows and a partial 32-segment step (n 96),
    // LFM3's 2048 / group 64, group 16 and 128, all three scale dtypes.
    for ([_][4]u32{ .{ 13, 96, 32, 2 }, .{ 37, 2048, 64, 2 }, .{ 40, 512, 16, 1 }, .{ 21, 1536, 128, 0 }, .{ 70, 7680, 64, 2 } }) |shape| {
        const m = shape[0];
        const n = shape[1];
        const group = shape[2];
        const sdt = shape[3];
        const esz: usize = if (sdt == 0) 4 else 2;
        const code_bytes = @as(usize, m) * n / 2;
        const groups = @as(usize, m) * n / group;
        // Codes 2 bytes off a word boundary (as LFM3's); f32 scales word-aligned.
        const s_off: usize = if (sdt == 0) (2 + code_bytes + 3) / 4 * 4 else 2 + code_bytes;
        const b_off = s_off + groups * esz;
        const wbuf = try ctx.alloc((b_off + groups * esz + 3) / 4 + 1);
        defer wbuf.deinit();
        const bytes = std.mem.sliceAsBytes(wbuf.floats());
        r.random().bytes(bytes[2 .. 2 + code_bytes]);
        // All-zero half-words (all-zero code quads) on every fifth row, one row of zero codes.
        for (0..m) |row| {
            const rb = bytes[2 + row * n / 2 ..][0 .. n / 2];
            if (row == 3) @memset(rb, 0);
            if (row % 5 == 0) {
                var i: usize = 0;
                while (i < rb.len) : (i += 2) if (r.random().boolean()) {
                    rb[i] = 0;
                    rb[i + 1] = 0;
                };
            }
        }
        for (0..groups) |g| {
            const sv: f32 = if (g % 11 == 0) 0 else 0.01 + 0.05 * r.random().float(f32);
            const bv: f32 = switch (g % 7) {
                0 => -0.0,
                1 => 0.0,
                else => -0.3 + 0.1 * r.random().float(f32),
            };
            for ([_]f32{ sv, bv }, [_]usize{ s_off, b_off }) |v, off| switch (sdt) {
                0 => std.mem.writeInt(u32, bytes[off + g * 4 ..][0..4], @bitCast(v), .little),
                1 => std.mem.writeInt(u16, bytes[off + g * 2 ..][0..2], @bitCast(@as(f16, @floatCast(v))), .little),
                else => std.mem.writeInt(u16, bytes[off + g * 2 ..][0..2], bf16(v), .little),
            };
        }
        const T = 45; // two 32-token tiles, the second partial
        const row0 = 2;
        const rows = T - row0;
        const buf = try ctx.alloc(T * n + 2 * T * m);
        defer buf.deinit();
        const f = buf.floats();
        const prep = try ctx.alloc(exact2.prepLayout(rows, n).words);
        defer prep.deinit();
        // Pass 0: no tiny inputs (every tile on the matrix units); pass 1: tiny rows in the
        // second token tile only (it takes the scalar fallback, the first does not).
        for (0..2) |pass| {
            for (0..T) |t| {
                const kind = t % 8;
                for (f[t * n ..][0..n], 0..) |*v, c| v.* = switch (kind) {
                    1 => ([_]f32{ 0.0, -0.0, 1.0, -1.0 })[r.random().uintLessThan(usize, 4)],
                    2 => -@abs(r.random().floatNorm(f32)) - if (c % 3 == 0) @as(f32, 0) else 1e-3,
                    3 => if (pass == 1 and t >= row0 + 32)
                        (if (r.random().boolean()) @as(f32, 1) else -1) * std.math.ldexp(1 + r.random().float(f32), -100 - @as(i32, @intCast(r.random().uintLessThan(u32, 30))))
                    else
                        r.random().floatNorm(f32),
                    4 => if (r.random().uintLessThan(u32, 16) == 0) c2Special(&r) else r.random().floatNorm(f32),
                    5 => if (c % 2 == 0) 1e4 * r.random().floatNorm(f32) else -f[t * n + c - 1],
                    6 => 1e35 * r.random().floatNorm(f32),
                    else => r.random().floatNorm(f32),
                };
            }
            const ref = T * n;
            const got = T * n + T * m;
            @memset(f[ref..][0 .. 2 * T * m], 0);
            for (row0..T) |t| {
                var p: gpu.Params = .{ .rows = 1, .row0 = @intCast(t), .xs = n, .y = @intCast(ref), .ys = m, .n = n, .m = m, .bits = 4, .group = group, .w = 2, .s = s_off, .b = b_off, .sdtype = sdt, .bdtype = sdt };
                try gpu.dispatch(ctx, "mk_linear_q4_r1_t1", &.{ wbuf, wbuf, wbuf, buf }, &p, m * 32, 32);
            }
            const p: gpu.Params = .{ .rows = rows, .row0 = row0, .xs = n, .y = @intCast(got), .ys = m, .n = n, .m = m, .bits = 4, .group = group, .w = 2, .s = s_off, .b = b_off, .sdtype = sdt, .bdtype = sdt };
            for ([_]bool{ false, true }) |half_codes| {
                @memset(f[got..][0 .. T * m], 0);
                var ph = p;
                try exact2.encodeLinear(ctx, .{ wbuf, wbuf, wbuf }, buf, prep, &ph, true, half_codes);
                try ctx.finish();
                const words = std.mem.bytesAsSlice(u32, std.mem.sliceAsBytes(f));
                try testing.expectEqualSlices(u32, words[ref + row0 * m ..][0 .. rows * m], words[got + row0 * m ..][0 .. rows * m]);
            }
            // The fallback flag marks exactly the tiny rows.
            const flags = prep.structs(u32)[exact2.prepLayout(rows, n).flags..][0..rows];
            for (flags, row0..) |fl, t| try testing.expectEqual(@as(u32, @intFromBool(pass == 1 and t % 8 == 3 and t >= row0 + 32)), fl);
        }
    }
}

test "exact prefill (climb 2): batched exact attention equals mk_attention and mk_attention_gated bit for bit" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const ctx = try gpu.open();
    defer ctx.deinit();
    var r = rng(4096);
    // (heads, kv heads, dim, window, cached positions before the batch, batch rows):
    // LFM3's full and sliding shapes, every group size and head-dim class, a wrapping ring.
    for ([_][6]u32{ .{ 16, 4, 128, 0, 300, 45 }, .{ 16, 4, 128, 37, 100, 45 }, .{ 16, 4, 128, 256, 0, 300 }, .{ 4, 2, 16, 5, 0, 19 }, .{ 8, 1, 64, 0, 0, 33 }, .{ 4, 4, 32, 9, 50, 20 }, .{ 2, 1, 256, 0, 10, 9 }, .{ 4, 1, 40, 0, 3, 17 } }) |shape| {
        const heads = shape[0];
        const kvh = shape[1];
        const dim = shape[2];
        const window = shape[3];
        const pos0 = shape[4];
        const T = shape[5];
        const row0 = 1;
        const rows = T - row0;
        const capacity: u32 = if (window == 0) pos0 + T else window - 1 + T;
        const span: u32 = if (window == 0) capacity else @min(window, capacity);
        const K = try ctx.alloc(capacity * kvh * dim);
        defer K.deinit();
        const V = try ctx.alloc(capacity * kvh * dim);
        defer V.deinit();
        const qw = heads * dim;
        const gate = T * qw;
        const out = gate + T * heads;
        const buf = try ctx.alloc(out + 4 * T * qw);
        defer buf.deinit();
        const f = buf.floats();
        for ([_]u32{ 7, rows }) |chunk| {
            const S = try ctx.alloc(chunk * heads * span);
            defer S.deinit();
            const Fb = try ctx.alloc(chunk * heads * span);
            defer Fb.deinit();
            const ML = try ctx.alloc(chunk * heads * 16);
            defer ML.deinit();
            // Kinds: random; adversarial values; equal keys with zero and -0 queries.
            for (0..3) |kind| {
                const first = (pos0 + T) -| capacity;
                for (first..pos0 + T) |t| {
                    const slot = t % capacity;
                    for (K.floats()[slot * kvh * dim ..][0 .. kvh * dim], V.floats()[slot * kvh * dim ..][0 .. kvh * dim], 0..) |*k, *v, i| {
                        k.* = 2 * r.random().floatNorm(f32);
                        v.* = r.random().floatNorm(f32);
                        if (kind == 1 and r.random().uintLessThan(u32, 40) == 0) k.* = c2Special(&r);
                        if (kind == 1 and r.random().uintLessThan(u32, 40) == 0) v.* = c2Special(&r);
                        if (kind == 2) k.* = if (i % 2 == 0) 0.5 else -0.0;
                    }
                }
                for (f[0 .. T * qw + T * heads], 0..) |*v, i| {
                    v.* = 2 * r.random().floatNorm(f32);
                    if (kind == 1 and r.random().uintLessThan(u32, 60) == 0) v.* = c2Special(&r);
                    if (kind == 2 and i < T * qw and (i / qw) % 3 == 0) v.* = if (i % 2 == 0) 0.0 else -0.0;
                }
                @memset(f[out..][0 .. 4 * T * qw], 0);
                var p: gpu.Params = .{ .rows = rows, .row0 = row0, .pos = pos0, .x = 0, .xs = qw, .ys = qw, .heads = heads, .kv_heads = kvh, .dim = dim, .window = window, .capacity = capacity, .scale = 0.088388, .z = gate, .zs = heads, .act_b = 2, .every = span };
                p.y = out;
                try gpu.dispatch(ctx, "mk_attention", &.{ buf, K, V }, &p, rows * heads * 256, 256);
                p.y = out + 2 * T * qw;
                try gpu.dispatch(ctx, "mk_attention_gated", &.{ buf, K, V }, &p, rows * heads * 256, 256);
                p.y = out + T * qw;
                try exact2.encodeAttention(ctx, buf, K, V, .{ S, Fb, ML }, &p, chunk);
                p.y = out + 3 * T * qw;
                p.flags = exact2.att_gated;
                try exact2.encodeAttention(ctx, buf, K, V, .{ S, Fb, ML }, &p, chunk);
                try ctx.finish();
                const lo = out + row0 * qw;
                try bitsEqual(f[lo..][0 .. rows * qw], f[lo + T * qw ..][0 .. rows * qw]);
                try bitsEqual(f[lo + 2 * T * qw ..][0 .. rows * qw], f[lo + 3 * T * qw ..][0 .. rows * qw]);
            }
        }
    }
}

test "relative attention matches CPU with denominator sink, distance bias, and window" {
    const ctx = try gpu.open();
    defer ctx.deinit();
    const arena = try ctx.alloc(80);
    defer arena.deinit();
    const keys = try ctx.alloc(24);
    defer keys.deinit();
    const values = try ctx.alloc(24);
    defer values.deinit();
    const f = arena.floats();
    for (0..2) |row| for (0..2) |head| for (0..4) |d| {
        f[row * 8 + head * 4 + d] = @as(f32, @floatFromInt((row * 5 + head * 3 + d * 2) % 7)) * 0.25 - 0.5;
    };
    for (0..2) |row| for (0..2) |head| for (0..3) |distance| {
        const code: i32 = @intCast((row + head * 2 + distance) % 5);
        f[16 + row * 6 + head * 3 + distance] = @as(f32, @floatFromInt(code - 2)) * 0.2;
    };
    for (0..4) |i| f[32 + i] = @as(f32, @floatFromInt(i)) * 0.3 - 0.4;

    for ([_]struct { window: u32, capacity: u32 }{ .{ .window = 0, .capacity = 6 }, .{ .window = 2, .capacity = 3 } }) |case| {
        @memset(keys.floats(), 0);
        @memset(values.floats(), 0);
        const fill_start: u32 = if (case.window == 0) 0 else 3;
        for (fill_start..6) |position| {
            const slot = position % case.capacity;
            for (0..4) |d| {
                const kcode: i32 = @intCast((position * 3 + d * 2) % 9);
                const vcode: i32 = @intCast((position * 5 + d * 4) % 11);
                keys.floats()[slot * 4 + d] = @as(f32, @floatFromInt(kcode - 4)) * 0.15;
                values.floats()[slot * 4 + d] = @as(f32, @floatFromInt(vcode - 5)) * 0.2;
            }
        }
        const out: u32 = if (case.window == 0) 40 else 56;
        @memset(f[out..][0..16], 0);
        const p: gpu.Params = .{ .rows = 2, .pos = 4, .x = 0, .xs = 8, .y = out, .ys = 8, .z = 16, .zs = 6, .s = 32, .every = 2, .n = 4, .m = 3, .heads = 2, .kv_heads = 1, .dim = 4, .window = case.window, .capacity = case.capacity, .scale = 0.5 };
        try gpu.dispatch(ctx, "mk_relative_attention", &.{ arena, keys, values }, &p, 2 * 2 * 32, 32);
        try ctx.finish();
        var expected_split: [16]f32 = undefined;
        @memcpy(&expected_split, f[out..][0..16]);
        const parts = try ctx.alloc(2 * 2 * 3 * (4 + 2));
        defer parts.deinit();
        var split = p;
        split.rows = 1;
        split.row0 = 1;
        split.taps = 3; // also exercises an empty partition for the two-key window
        try gpu.dispatch(ctx, "mk_relative_split_keys", &.{ arena, keys, values, parts }, &split, 2 * 3 * 32, 32);
        try gpu.dispatch(ctx, "mk_relative_split_merge", &.{ arena, parts }, &split, 2 * 32, 32);
        try ctx.finish();
        for (expected_split[8..], f[out + 8 ..][0..8]) |expected, actual| try close(expected, actual, 1e-5);
        split.row0 = 0;
        split.rows = 2;
        try gpu.dispatch(ctx, "mk_relative_split_keys", &.{ arena, keys, values, parts }, &split, 2 * 2 * 3 * 32, 32);
        try gpu.dispatch(ctx, "mk_relative_split_merge", &.{ arena, parts }, &split, 2 * 2 * 32, 32);
        try ctx.finish();
        for (expected_split, f[out..][0..16]) |expected, actual| try close(expected, actual, 1e-5);
        for (0..2) |row| for (0..2) |head| {
            const position: u32 = 4 + @as(u32, @intCast(row));
            const first: u32 = if (case.window != 0 and position + 1 > case.window) position + 1 - case.window else 0;
            var max_logit: f64 = f[32 + row * 2 + head];
            var logits: [6]f64 = undefined;
            var count: usize = 0;
            for (first..position + 1) |t| {
                var dot: f64 = 0;
                for (0..4) |d| dot += @as(f64, f[row * 8 + head * 4 + d]) * keys.floats()[(t % case.capacity) * 4 + d];
                const distance = position - @as(u32, @intCast(t));
                const bias: f64 = if (distance < 3) f[16 + row * 6 + head * 3 + distance] else 0;
                const score = dot * 0.5 + bias;
                logits[count] = score;
                max_logit = @max(max_logit, score);
                count += 1;
            }
            var denominator = @exp(@as(f64, f[32 + row * 2 + head]) - max_logit);
            for (logits[0..count]) |score| denominator += @exp(score - max_logit);
            for (0..4) |d| {
                var expected: f64 = 0;
                count = 0;
                for (first..position + 1) |t| {
                    var dot: f64 = 0;
                    for (0..4) |j| dot += @as(f64, f[row * 8 + head * 4 + j]) * keys.floats()[(t % case.capacity) * 4 + j];
                    const distance = position - @as(u32, @intCast(t));
                    const bias: f64 = if (distance < 3) f[16 + row * 6 + head * 3 + distance] else 0;
                    expected += @exp(dot * 0.5 + bias - max_logit) * values.floats()[(t % case.capacity) * 4 + d];
                    count += 1;
                }
                try close(expected / denominator, f[out + row * 8 + head * 4 + d], 1e-5);
            }
        };
    }
}

test "tiled relative attention handles window wrap and a partial query tile" {
    const ctx = try gpu.open();
    defer ctx.deinit();
    const rows = 35;
    const row0 = 1;
    const heads = 2;
    const dim = 128;
    const stride = heads * dim;
    const pos = 38;
    const extent = 13;
    const seq_end = pos + row0 + rows;
    const bias_off = (rows + row0) * stride * 3;
    const sink_off = bias_off + (rows + row0) * heads * extent;
    const arena = try ctx.alloc(sink_off + (rows + row0) * heads);
    defer arena.deinit();
    const f = arena.floats();
    for (f[0 .. (rows + row0) * stride], 0..) |*v, i| {
        v.* = @as(f32, @floatFromInt(@as(i32, @intCast(i % 11)) - 5)) * 0.125;
    }
    for (f[bias_off..], 0..) |*v, i| {
        v.* = @as(f32, @floatFromInt(@as(i32, @intCast(i % 17)) - 8)) * 0.0625;
    }
    for ([_]u32{ 0, 21 }) |window| {
        const capacity = if (window == 0) seq_end else window + rows - 1;
        const keys = try ctx.alloc(capacity * dim);
        defer keys.deinit();
        const values = try ctx.alloc(capacity * dim);
        defer values.deinit();
        for (seq_end - capacity..seq_end) |t| for (0..dim) |d| {
            const i = t % capacity * dim + d;
            keys.floats()[i] = @as(f32, @floatFromInt(@as(i32, @intCast((t * 7 + d * 3) % 13)) - 6)) * 0.0625;
            values.floats()[i] = @as(f32, @floatFromInt(@as(i32, @intCast((t * 5 + d * 2) % 19)) - 9)) * 0.0625;
        };
        const out_off = (rows + row0) * stride;
        var p: gpu.Params = .{
            .rows = rows,
            .row0 = row0,
            .pos = pos,
            .x = 0,
            .xs = stride,
            .y = out_off,
            .ys = stride,
            .z = bias_off,
            .zs = heads * extent,
            .s = sink_off,
            .every = heads,
            .m = extent,
            .heads = heads,
            .kv_heads = 1,
            .dim = dim,
            .window = window,
            .capacity = capacity,
            .scale = 0.0883883476,
        };
        try gpu.dispatch(ctx, "mk_relative_attention", &.{ arena, keys, values }, &p, heads * rows * 32, 32);
        p.y += out_off;
        @memset(f[p.y..][0..stride], -7);
        try gpu.dispatch(ctx, "mk_relative_attention_prefill", &.{ arena, keys, values }, &p, 2 * heads * 128, 128);
        try ctx.finish();
        for (f[p.y..][0..stride]) |v| try std.testing.expectEqual(@as(f32, -7), v);
        const first = row0 * stride;
        for (f[out_off + first ..][0 .. rows * stride], f[p.y + first ..][0 .. rows * stride]) |expected, actual| {
            try close(expected, actual, 0.0003);
        }

        const parts = try ctx.alloc(heads * 3 * (dim + 2));
        defer parts.deinit();
        p.rows = 1;
        p.row0 = 33;
        p.taps = 3;
        try gpu.dispatch(ctx, "mk_relative_split_keys", &.{ arena, keys, values, parts }, &p, heads * p.taps * 32, 32);
        try ctx.finish();
        var before: [heads * 3 * (dim + 2)]f32 = undefined;
        @memcpy(&before, parts.floats()[0..before.len]);
        try gpu.dispatch(ctx, "mk_relative_split_keys4", &.{ arena, keys, values, parts }, &p, heads * p.taps * 128, 128);
        try ctx.finish();
        try std.testing.expectEqualSlices(f32, &before, parts.floats()[0..before.len]);
    }
}

test "route top-k uses stable ties, corrected ranking, and raw-score normalization" {
    const ctx = try gpu.open();
    defer ctx.deinit();
    const arena = try ctx.alloc(256);
    defer arena.deinit();
    const correction = try ctx.alloc(128);
    defer correction.deinit();
    const f = arena.floats();
    const bias = correction.floats();
    const p: gpu.Params = .{ .rows = 1, .x = 0, .xs = 128, .y = 128, .ys = 6, .n = 128, .m = 3, .s = 0, .eps = 1e-20, .scale = 8 };

    @memset(f[0..128], -10);
    @memset(bias, 0);
    f[0] = 0;
    f[1] = 1;
    f[2] = -1;
    f[3] = -2;
    bias[0] = 1.0;
    bias[1] = 0.0;
    bias[2] = 1.3;
    try gpu.dispatch(ctx, "mk_route_topk", &.{ arena, correction }, &p, 32, 256);
    try ctx.finish();
    try std.testing.expectEqualSlices(f32, &.{ 2, 0, 1 }, f[128..131]);
    const score0: f64 = 0.5;
    const score1: f64 = 1.0 / (1.0 + @exp(-1.0));
    const score2: f64 = 1.0 / (1.0 + @exp(1.0));
    const score_sum = score0 + score1 + score2;
    try close(score2 / score_sum * 8, f[131], 1e-6);
    try close(score0 / score_sum * 8, f[132], 1e-6);
    try close(score1 / score_sum * 8, f[133], 1e-6);

    @memset(f[0..128], 0);
    @memset(bias, 0);
    try gpu.dispatch(ctx, "mk_route_topk", &.{ arena, correction }, &p, 32, 256);
    try ctx.finish();
    try std.testing.expectEqualSlices(f32, &.{ 0, 1, 2 }, f[128..131]);
    try close(8.0 / 3.0, f[131], 1e-6);
    try close(8.0 / 3.0, f[132], 1e-6);
    try close(8.0 / 3.0, f[133], 1e-6);
}

test "vector BF16 decode handles strided nonzero rows" {
    const ctx = try gpu.open();
    defer ctx.deinit();
    const n = 256;
    const m = 8;
    const xs = n + 4;
    const ys = m + 4;
    const x = 4;
    const y = x + 2 * xs;
    const arena = try ctx.alloc(y + 2 * ys);
    defer arena.deinit();
    const w = try ctx.alloc(2 + m * n / 2);
    defer w.deinit();
    const halves = std.mem.bytesAsSlice(u16, std.mem.sliceAsBytes(w.floats()))[4..];
    const f = arena.floats();
    @memset(f, -9);
    var r = rng(0x18c0);
    for (f[x + xs ..][0..n]) |*v| v.* = 2 * r.random().float(f32) - 1;
    for (halves) |*v| v.* = bf16(2 * r.random().float(f32) - 1);
    const params: gpu.Params = .{ .x = x, .xs = xs, .y = y, .ys = ys, .w = 8, .row0 = 1, .rows = 1, .m = m, .n = n };
    try gpu.dispatch(ctx, "mk_bf16_linear_4x1_vec", &.{ w, arena }, &params, m / 4 * 32, 128);
    try ctx.finish();
    for (0..m) |j| {
        var expected: f64 = 0;
        for (0..n) |k| expected += @as(f64, f[x + xs + k]) * fromBf16(halves[j * n + k]);
        try testing.expect(std.math.isFinite(f[y + ys + j]));
        try close(expected, f[y + ys + j], 2e-6);
    }
    try testing.expectEqual(@as(f32, -9), f[y]);
    try testing.expectEqual(@as(f32, -9), f[y + ys + m]);
}

test "BF16 prefill matrix and vector tiles respect strides and row bounds" {
    const ctx = try gpu.open();
    defer ctx.deinit();
    const n = 128;
    const m = 64;
    const xs = n + 4;
    const ys = m + 4;
    const x = 4;
    const y = x + 37 * xs;
    const arena = try ctx.alloc(y + 37 * ys);
    defer arena.deinit();
    const w = try ctx.alloc(2 + m * n / 2);
    defer w.deinit();
    const halves = std.mem.bytesAsSlice(u16, std.mem.sliceAsBytes(w.floats()))[4..];
    const f = arena.floats();
    var r = rng(0x478e);
    for (f[x..y]) |*v| v.* = @as(f32, @floatFromInt(r.random().intRangeAtMost(i32, -7, 7))) * 0.125;
    for (halves) |*v| v.* = bf16(@as(f32, @floatFromInt(r.random().intRangeAtMost(i32, -7, 7))) * 0.125);
    const cases = [_]struct { kernel: [:0]const u8, rows: u32, groups: usize }{
        .{ .kernel = "mk_bf16_prefill_matrix16", .rows = 35, .groups = 3 },
        .{ .kernel = "mk_bf16_prefill_matrix32", .rows = 35, .groups = 2 },
        .{ .kernel = "mk_bf16_linear_4x8_full_vec", .rows = 32, .groups = m / 4 },
    };
    for (cases) |case| {
        @memset(f[y..], -99);
        const params: gpu.Params = .{ .x = x, .xs = xs, .y = y, .ys = ys, .w = 8, .row0 = 1, .rows = case.rows, .m = m, .n = n };
        try gpu.dispatch(ctx, case.kernel, &.{ w, arena }, &params, case.groups * 128, 128);
        try ctx.finish();
        for (1..case.rows + 1) |t| {
            for (0..m) |j| {
                var expected: f32 = 0;
                for (0..n) |k| expected += f[x + t * xs + k] * fromBf16(halves[j * n + k]);
                try testing.expectEqual(expected, f[y + t * ys + j]);
            }
            for (f[y + t * ys + m ..][0 .. ys - m]) |value| try testing.expectEqual(@as(f32, -99), value);
        }
        for (f[y..][0..ys]) |value| try testing.expectEqual(@as(f32, -99), value);
        for (f[y + (case.rows + 1) * ys ..]) |value| try testing.expectEqual(@as(f32, -99), value);
    }
}

test "Inkling L2 clipping matches the threshold formula" {
    const ctx = try gpu.open();
    defer ctx.deinit();
    const buf = try ctx.alloc(32);
    defer buf.deinit();
    const f = buf.floats();
    @memcpy(f[0..8], &[_]f32{ 3, 4, 0, 0, 0.5, -0.5, 0.5, -0.5 });
    const p: gpu.Params = .{ .rows = 1, .x = 0, .xs = 8, .z = 16, .zs = 8, .n = 4, .heads = 2, .eps = 2.0 };
    try gpu.dispatch(ctx, "mk_l2_clip", &.{ buf, buf }, &p, 2 * 256, 256);
    try ctx.finish();
    const factor: f64 = 2.0 / (5.0 + 1e-6);
    for (0..4) |i| try close(@as(f64, f[i]) * factor, f[16 + i], 1e-6);
    for (0..4) |i| try close(f[4 + i], f[20 + i], 1e-6);
}

test "Inkling scaled L2 clipping scales before measuring the norm" {
    const ctx = try gpu.open();
    defer ctx.deinit();
    const buf = try ctx.alloc(32);
    defer buf.deinit();
    const scale = try ctx.alloc(8);
    defer scale.deinit();
    const f = buf.floats();
    @memcpy(f[0..8], &[_]f32{ 3, 4, 0, 0, 0.5, -0.5, 0.5, -0.5 });
    @memcpy(scale.floats(), &[_]f32{ 2, 1, 0.5, 1, 1, 2, 1, 0.5 });
    const p: gpu.Params = .{ .rows = 1, .x = 0, .xs = 8, .z = 16, .zs = 8, .n = 4, .heads = 2, .eps = 2.0, .w = 0, .dtype = 0, .flags = gpu.flag_has_weight };
    try gpu.dispatch(ctx, "mk_l2_clip", &.{ buf, scale }, &p, 2 * 256, 256);
    try ctx.finish();
    const factor: f64 = 2.0 / (@sqrt(@as(f64, 52)) + 1e-6);
    for (0..4) |i| try close((&[_]f64{ 6, 4, 0, 0 })[i] * factor, f[16 + i], 1e-6);
    for (0..4) |i| try close((&[_]f64{ 0.5, -1, 0.5, -0.25 })[i], f[20 + i], 1e-6);
}

test "Inkling short convolution matches causal conv plus residual normalization" {
    const ctx = try gpu.open();
    defer ctx.deinit();
    const C = 3;
    const taps = 3;
    const T = 5;
    const capacity = taps - 1 + T;
    const ring = try ctx.alloc(capacity * C);
    defer ring.deinit();
    @memset(ring.floats(), std.math.nan(f32));
    const w = try ctx.alloc(C * taps);
    defer w.deinit();
    @memcpy(w.floats(), &[_]f32{ 0.5, -0.25, 0.75, -0.5, 0.125, 0.25, 1, -1, 0.5 });
    const buf = try ctx.alloc(2 * T * C);
    defer buf.deinit();
    const f = buf.floats();
    var inputs: [T * C]f32 = undefined;
    var residual: [T * C]f32 = undefined;
    for (0..T * C) |i| {
        inputs[i] = @as(f32, @floatFromInt(i)) / 7.0 - 0.75;
        residual[i] = 0.25 - @as(f32, @floatFromInt(i)) / 19.0;
    }
    @memcpy(f[0 .. T * C], &inputs);
    @memcpy(f[T * C .. 2 * T * C], &residual);
    var p: gpu.Params = .{ .rows = T, .pos = 0, .x = 0, .xs = C, .n = C, .capacity = capacity };
    try gpu.dispatch(ctx, "mk_store", &.{ buf, ring }, &p, T * C, 32);
    p = .{ .rows = T, .pos = 0, .x = T * C, .xs = C, .z = 0, .zs = C, .n = C, .taps = taps, .capacity = capacity, .scale = 0.5 };
    try gpu.dispatch(ctx, "mk_inkling_short_conv", &.{ ring, w, buf }, &p, T * C, 32);
    try ctx.finish();
    for (0..T) |t| for (0..C) |c| {
        var sum: f64 = @floatCast(residual[t * C + c]);
        for (0..taps) |k| {
            const q = @as(i64, @intCast(t)) - taps + 1 + @as(i64, @intCast(k));
            if (q >= 0) sum += @as(f64, w.floats()[c * taps + k]) * inputs[@as(usize, @intCast(q)) * C + c];
        }
        try close(sum / 2.0, f[t * C + c], 1e-5);
    };
}

test "Inkling dynamic convolution matches per-head taps, static taps, and reversed history" {
    const ctx = try gpu.open();
    defer ctx.deinit();
    const C = 4;
    const heads = 2;
    const taps = 3;
    const T = 4;
    const capacity = taps - 1 + T;
    const ring = try ctx.alloc(capacity * C);
    defer ring.deinit();
    @memset(ring.floats(), std.math.nan(f32));
    const w = try ctx.alloc(taps * C);
    defer w.deinit();
    for (w.floats(), 0..) |*x, i| x.* = @as(f32, @floatFromInt(i + 1)) / 20.0 - 0.2;
    const buf = try ctx.alloc(2 * T * C + T * heads * taps);
    defer buf.deinit();
    const f = buf.floats();
    var inputs: [T * C]f32 = undefined;
    var dynamic: [T * heads * taps]f32 = undefined;
    var residual: [T * C]f32 = undefined;
    for (0..T * C) |i| {
        inputs[i] = @as(f32, @floatFromInt(i)) / 11.0 - 0.4;
        residual[i] = 0.1 - @as(f32, @floatFromInt(i)) / 37.0;
    }
    for (0..T * heads * taps) |i| dynamic[i] = @as(f32, @floatFromInt(i + 2)) / 29.0 - 0.3;
    @memcpy(f[0 .. T * C], &inputs);
    @memcpy(f[T * C .. T * C + T * heads * taps], &dynamic);
    @memcpy(f[T * C + T * heads * taps .. 2 * T * C + T * heads * taps], &residual);
    var p: gpu.Params = .{ .rows = T, .pos = 0, .x = 0, .xs = C, .n = C, .capacity = capacity };
    try gpu.dispatch(ctx, "mk_store", &.{ buf, ring }, &p, T * C, 32);
    p = .{ .rows = T, .pos = 0, .x = T * C, .xs = heads * taps, .y = T * C + T * heads * taps, .ys = C, .z = 0, .zs = C, .n = C, .heads = heads, .taps = taps, .capacity = capacity, .scale = 1.0 / @sqrt(7.0) };
    try gpu.dispatch(ctx, "mk_inkling_dynamic_conv", &.{ ring, w, buf }, &p, T * C, 32);
    try ctx.finish();
    for (0..T) |t| for (0..C) |c| {
        const head = c / (C / heads);
        var sum: f64 = @floatCast(residual[t * C + c]);
        for (0..taps) |k| {
            const q = @as(i64, @intCast(t)) - @as(i64, @intCast(k));
            if (q >= 0) sum += @as(f64, inputs[@as(usize, @intCast(q)) * C + c]) * (w.floats()[k * C + c] + dynamic[t * heads * taps + head * taps + k]);
        }
        try close(sum / @sqrt(7.0), f[t * C + c], 1e-5);
    };
}

test "Inkling message gate merge matches controller formula" {
    const ctx = try gpu.open();
    defer ctx.deinit();
    const H = 4;
    const buf = try ctx.alloc(5 * H);
    defer buf.deinit();
    const f = buf.floats();
    @memcpy(f[0..H], &[_]f32{ -1.0, -0.25, 0.5, 1.25 });
    @memcpy(f[H .. 2 * H], &[_]f32{ 0.5, -0.75, 1.0, -1.5 });
    @memcpy(f[2 * H .. 4 * H], &[_]f32{ -2.0, -0.5, 0.0, 0.75, 1.5, 0.25, -1.0, 2.0 });
    const p: gpu.Params = .{ .rows = 1, .x = 0, .xs = H, .y = H, .ys = H, .z = 2 * H, .zs = 2 * H, .w = 4 * H, .every = H, .n = H };
    try gpu.dispatch(ctx, "mk_inkling_message_merge", &.{buf}, &p, H, 32);
    try ctx.finish();
    for (0..H) |i| {
        const forget = 2.0 / (1.0 + @exp(-2.0 * @as(f64, f[2 * H + i]))) - 1.0;
        const gain = 2.0 / (1.0 + @exp(-2.0 * @as(f64, f[3 * H + i]))) - 1.0;
        const expected = (1.0 - forget) * @as(f64, f[i]) + gain * @as(f64, f[H + i]);
        try close(expected, f[4 * H + i], 1e-6);
    }
}

test "packed experts decode both formats and combine selected routes" {
    const ctx = try gpu.open();
    defer ctx.deinit();
    const arena = try ctx.alloc(224);
    defer arena.deinit();
    const w = try ctx.alloc(2 * 64 * 32 / 4);
    defer w.deinit();
    const scales = try ctx.alloc(2 * 64 * 2 / 4);
    defer scales.deinit();
    const amax = try ctx.alloc(2);
    defer amax.deinit();
    const denominator = try ctx.alloc(1);
    defer denominator.deinit();
    @memcpy(amax.floats(), &[_]f32{ 1, 4 });
    denominator.floats()[0] = 2;

    const f = arena.floats();
    @memset(f, 0);
    f[32] = 2;
    f[33] = 1;
    f[48] = 1;
    f[49] = 0;
    f[50] = 0.25;
    f[51] = 0.75;
    const bytes = std.mem.sliceAsBytes(w.floats());
    const sb = std.mem.sliceAsBytes(scales.floats());
    for ([_]u32{ 0, 1 }) |nv| {
        @memset(bytes, 0);
        @memset(sb, if (nv == 1) 0x38 else 128);
        for (0..2) |expert| for (0..64) |j| {
            const gate = j % 2 == 0;
            if (nv == 1) {
                // g: [1, -0.5]; u: [1.5, 2]. Scale by 0.5 / 2 per expert.
                bytes[(expert * 64 + j) * 16] = if (gate) 0x92 else 0x43;
            } else {
                // The same values after E8M0 scaling. Exercise signed and
                // subnormal E4M3 in otherwise zero input columns.
                const base = (expert * 64 + j) * 32;
                bytes[base] = if (gate) 0x30 else 0x34;
                bytes[base + 1] = if (gate) 0xa8 else 0x38;
                bytes[base + 2] = 0x81;
            }
        };
        var p: gpu.ExpertParams = .{ .rows = 1, .row0 = 1, .x = 0, .xs = 32, .routes = 44, .rs = 4, .y = 64, .ys = 64, .k = 32, .n = 32, .experts = 2, .top_k = 2, .nvfp4 = nv, .gate_cap = 4, .up_cap = 25 };
        try gpu.dispatch(ctx, "mk_expert_gate_up", &.{ arena, w, scales, amax, denominator }, &p, 64, 32);
        try ctx.finish();
        var expected: [2]f64 = undefined;
        for (0..2) |slot| {
            const factor: f64 = if (nv == 0) 1 else if (slot == 0) 2 else 0.5;
            const g = 1.5 * factor;
            const u = 5.0 * factor;
            const gate_cap = 4 * (2 / (1 + @exp(-2 * g / 4)) - 1);
            const up_cap = 25 * (2 / (1 + @exp(-2 * u / 25)) - 1);
            expected[slot] = gate_cap / (1 + @exp(-g)) * up_cap;
            for (0..32) |j| try close(expected[slot], f[128 + slot * 32 + j], 2e-5);
        }

        if (nv == 1) {
            @memset(f[128..192], 0);
            try gpu.dispatch(ctx, "mk_expert_gate_up_dot8", &.{ arena, w, scales, amax, denominator }, &p, 32 * 32, 256);
            try ctx.finish();
            for (0..2) |slot| for (0..32) |j| try close(expected[slot], f[128 + slot * 32 + j], 2e-5);
        }

        @memset(bytes, 0);
        for (0..2) |i| {
            if (nv == 1) bytes[i * 16] = 0x02 else bytes[i * 32] = 0x30;
        }
        p.x = 64;
        p.xs = 64;
        p.y = 200;
        p.ys = 1;
        p.n = 1;
        try gpu.dispatch(ctx, "mk_expert_down", &.{ arena, w, scales, amax, denominator }, &p, 1, 32);
        try ctx.finish();
        const high: f64 = if (nv == 1) 2 else 1;
        const low: f64 = if (nv == 1) 0.5 else 1;
        try close(expected[0] * high * 0.25 + expected[1] * low * 0.75, f[201], 2e-5);
        try std.testing.expectEqual(@as(f32, 0), f[200]);
        if (nv == 1) {
            p.n = 4;
            p.ys = 4;
            @memset(f[200..208], 0);
            @memset(bytes[0..128], 0);
            for (0..2) |expert| for (0..4) |column| {
                bytes[(expert * 4 + column) * 16] = @as(u8, @intCast(column)) + 2;
            };
            try gpu.dispatch(ctx, "mk_expert_down_dot8", &.{ arena, w, scales, amax, denominator }, &p, 256, 256);
            try ctx.finish();
            const combined = expected[0] * high * 0.25 + expected[1] * low * 0.75;
            for ([_]f64{ 1, 1.5, 2, 3 }, f[204..208]) |factor, actual| try close(combined * factor, actual, 8e-5);
            try std.testing.expectEqualSlices(f32, &.{ 0, 0, 0, 0 }, f[200..204]);
        }
    }
}

test "Inkling split expert tiles cover full blocks and tails without writing unused routes" {
    const ctx = try gpu.open();
    defer ctx.deinit();
    const rows = 192;
    const k = 64;
    const n = 32;
    const arena = try ctx.alloc((rows + 1) * k);
    defer arena.deinit();
    const weights = try ctx.alloc(n * k / 8);
    defer weights.deinit();
    const scales = try ctx.alloc(n * k / 64);
    defer scales.deinit();
    const scale = try ctx.alloc(1);
    defer scale.deinit();
    const routes = try ctx.alloc(rows + 1);
    defer routes.deinit();
    const partial = try ctx.alloc(rows * n);
    defer partial.deinit();
    scale.floats()[0] = 1;
    const bytes = std.mem.sliceAsBytes(weights.floats());
    @memset(bytes, 0);
    @memset(std.mem.sliceAsBytes(scales.floats()), 0x38);
    for (0..n) |i| bytes[i * k / 2] = if (i % 2 == 0) 0x02 else 0x0b; // 1 or -1.5
    @memset(arena.floats(), 0);
    for (0..rows) |i| arena.floats()[(i + 1) * k] = @floatFromInt(i + 1);
    const list = std.mem.bytesAsSlice(u32, std.mem.sliceAsBytes(routes.floats()));
    for (0..rows) |i| list[i + 1] = @intCast(rows - 1 - i);
    const p: gpu.ExpertParams = .{ .rows = rows, .row0 = 1, .xs = k, .k = k, .n = n, .experts = 1, .top_k = 1, .nvfp4 = 1, .round_bf16 = 1 };
    for ([_]usize{ 0, 8, 12, 20, 32, 37, 48, 63, 64, 81, 97, 128, 133, 192 }) |count| {
        list[0] = @intCast(count);
        @memset(partial.floats(), -999);
        try gpu.dispatch(ctx, "mk_expert_block_down_nv", &.{ arena, weights, scales, scale, scale, routes, partial }, &p, 4 * 128, 128);
        try ctx.finish();
        for (0..rows) |r| for (0..n) |c| {
            const x0: f32 = @floatFromInt(r + 1);
            const dot = x0 * (if (c % 2 == 0) @as(f32, 1) else -1.5);
            const expected = if (r < rows - count) -999 else fromBf16(bf16(dot));
            try testing.expectEqual(expected, partial.floats()[r * n + c]);
        };
    }
}
