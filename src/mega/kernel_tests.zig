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
    // (heads, kv heads, dim, window): LFM2.5 and LFM3's full, sliding, and other shapes.
    for ([_][4]u32{ .{ 16, 4, 128, 0 }, .{ 32, 8, 64, 0 }, .{ 16, 4, 128, 37 }, .{ 8, 1, 64, 0 }, .{ 4, 4, 32, 9 }, .{ 2, 1, 256, 0 }, .{ 4, 2, 16, 5 }, .{ 4, 1, 40, 0 } }) |shape| {
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
            if (window == 0 and G == 4 and (dim == 64 or dim == 128)) {
                const linear_name = if (dim == 64) "mk_dec_scores_linear_e2_g4" else "mk_dec_scores_linear_e4_g4";
                try gpu.dispatch(ctx, linear_name, &.{ buf, K, S }, &p, kvh * chunks * 32, 256);
                p.flags = 0;
                p.y = o + qw;
                try gpu.dispatch(ctx, "mk_dec_chain_linear", &.{ buf, V, S }, &p, heads * E * 256, 256);
                p.flags = gpu.flag_dec_gated;
                p.y = o + 3 * qw;
                try gpu.dispatch(ctx, "mk_dec_chain_linear", &.{ buf, V, S }, &p, heads * E * 256, 256);
                try ctx.finish();
                try bitsEqual(f[o..][0..qw], f[o + qw ..][0..qw]);
                try bitsEqual(f[o + 2 * qw ..][0..qw], f[o + 3 * qw ..][0..qw]);
            }
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
    const packed_off = sink_off + (rows + row0) * heads;
    const arena = try ctx.alloc(packed_off + (rows + row0) * heads * extent);
    defer arena.deinit();
    const f = arena.floats();
    for (f[0 .. (rows + row0) * stride], 0..) |*v, i| {
        v.* = @as(f32, @floatFromInt(@as(i32, @intCast(i % 11)) - 5)) * 0.125;
    }
    for (f[bias_off..packed_off], 0..) |*v, i| {
        v.* = @as(f32, @floatFromInt(@as(i32, @intCast(i % 17)) - 8)) * 0.0625;
    }
    // The same bias as 16-bit words, each row's at the start of its own words.
    for (0..rows + row0) |r| {
        const src = f[bias_off + r * heads * extent ..][0 .. heads * extent];
        const dst = std.mem.bytesAsSlice(u16, std.mem.sliceAsBytes(f[packed_off + r * heads * extent ..][0 .. heads * extent]));
        for (src, dst[0..src.len]) |v, *h| h.* = @intCast(@as(u32, @bitCast(v)) >> 16);
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

        // The same tiles in stripes of keys, loaded from the 16-bit mirrors (padded
        // with their first 16 rows, which the wrapped window's tiles read): one
        // stripe, uneven stripes, and more stripes than a row tile has tiles.
        const half_keys = try ctx.alloc((capacity + 16) * dim / 2);
        defer half_keys.deinit();
        const half_values = try ctx.alloc((capacity + 16) * dim / 2);
        defer half_values.deinit();
        for ([_][2]@TypeOf(keys){ .{ keys, half_keys }, .{ values, half_values } }) |pair| {
            const half = std.mem.bytesAsSlice(u16, std.mem.sliceAsBytes(pair[1].floats()));
            for (pair[0].floats(), half[0 .. capacity * dim]) |v, *h| {
                try std.testing.expectEqual(@as(u32, 0), @as(u32, @bitCast(v)) & 0xffff);
                h.* = @intCast(@as(u32, @bitCast(v)) >> 16);
            }
            @memcpy(half[capacity * dim ..][0 .. 16 * dim], half[0 .. 16 * dim]);
        }
        const tiled = try std.testing.allocator.dupe(f32, f[p.y + first ..][0 .. rows * stride]);
        defer std.testing.allocator.free(tiled);
        for ([_]u32{ 1, 3, 16 }) |taps| {
            const records = try ctx.alloc(rows * heads * taps * (dim + 2));
            defer records.deinit();
            p.taps = taps;
            @memset(f[p.y..][0 .. (rows + row0) * stride], -7);
            try gpu.dispatch(ctx, "mk_relative_prefill_stripes", &.{ arena, half_keys, half_values, records }, &p, 2 * heads * taps * 128, 128);
            try gpu.dispatch(ctx, "mk_relative_prefill_merge", &.{ arena, records }, &p, rows * heads * 128, 128);
            try ctx.finish();
            for (f[p.y..][0..stride]) |v| try std.testing.expectEqual(@as(f32, -7), v);
            for (tiled, f[p.y + first ..][0 .. rows * stride]) |expected, actual| try close(expected, actual, 0.0001);
        }

        // The reference's staged tiles off the same mirrors: the tiled kernel's bits on an
        // unwindowed cache; on the wrapped window its tiles start at position 0, not at the
        // window's first key.
        @memset(f[p.y..][0 .. (rows + row0) * stride], -7);
        try gpu.dispatch(ctx, "mk_relative_prefill_tiles", &.{ arena, half_keys, half_values }, &p, 2 * heads * 128, 128);
        try ctx.finish();
        for (f[p.y..][0..stride]) |v| try std.testing.expectEqual(@as(f32, -7), v);
        if (window == 0) {
            try std.testing.expectEqualSlices(f32, tiled, f[p.y + first ..][0 .. rows * stride]);
        } else for (tiled, f[p.y + first ..][0 .. rows * stride]) |expected, actual| try close(expected, actual, 0.0001);
        // The bias read as 16-bit words gives the same bits.
        const unpacked = try std.testing.allocator.dupe(f32, f[p.y + first ..][0 .. rows * stride]);
        defer std.testing.allocator.free(unpacked);
        @memset(f[p.y..][0 .. (rows + row0) * stride], -7);
        p.z = packed_off;
        p.flags = gpu.flag_bias16;
        try gpu.dispatch(ctx, "mk_relative_prefill_tiles", &.{ arena, half_keys, half_values }, &p, 2 * heads * 128, 128);
        try ctx.finish();
        try std.testing.expectEqualSlices(f32, unpacked, f[p.y + first ..][0 .. rows * stride]);
        p.z = bias_off;
        p.flags = 0;

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

test "the packed distance bias holds the tiled projection's bfloat16 values" {
    const ctx = try gpu.open();
    defer ctx.deinit();
    const rows = 8;
    const row0 = 4;
    const heads = 3;
    const extent = 7;
    const taps = 5; // a causal prefix: distances beyond the last query position are not written
    const x_off = 0;
    const out_off = (rows + row0) * heads * 16;
    const packed_off = out_off + (rows + row0) * heads * extent;
    const arena = try ctx.alloc(packed_off + (rows + row0) * heads * extent);
    defer arena.deinit();
    const f = arena.floats();
    for (f[0..out_off], 0..) |*v, i| v.* = @as(f32, @floatFromInt(@as(i32, @intCast(i * 7 % 23)) - 11)) * 0.03125;
    @memset(f[out_off..], -7);
    const weights = try ctx.alloc(16 * extent / 2 + 1);
    defer weights.deinit();
    const w16 = std.mem.bytesAsSlice(u16, std.mem.sliceAsBytes(weights.floats()));
    for (w16[0 .. 16 * extent], 0..) |*h, i| {
        const v: f32 = @as(f32, @floatFromInt(@as(i32, @intCast(i * 5 % 29)) - 14)) * 0.0078125;
        h.* = @intCast(@as(u32, @bitCast(v)) >> 16);
    }
    var p: gpu.Params = .{ .rows = rows, .row0 = row0, .x = x_off, .xs = heads * 16, .y = out_off, .ys = heads * extent, .n = heads * extent, .m = extent, .dim = 16, .dtype = 2, .taps = taps, .flags = gpu.flag_round_bf16 };
    try gpu.dispatch(ctx, "mk_grouped_project_bf16x4", &.{ arena, weights }, &p, rows / 4 * heads * taps, 256);
    p.y = packed_off;
    try gpu.dispatch(ctx, "mk_grouped_project_bf16x4_pack", &.{ arena, weights }, &p, rows / 4 * heads * taps, 256);
    try ctx.finish();
    var nonzero: usize = 0;
    for (0..rows + row0) |r| {
        const plain = f[out_off + r * heads * extent ..][0 .. heads * extent];
        const halves = std.mem.bytesAsSlice(u16, std.mem.sliceAsBytes(f[packed_off + r * heads * extent ..][0 .. heads * extent]));
        for (plain, 0..) |v, i| {
            const written = r >= row0 and i % extent < taps;
            if (!written) {
                try std.testing.expectEqual(@as(f32, -7), v);
                continue;
            }
            try std.testing.expectEqual(@as(u32, 0), @as(u32, @bitCast(v)) & 0xffff);
            try std.testing.expectEqual(@as(u16, @intCast(@as(u32, @bitCast(v)) >> 16)), halves[i]);
            nonzero += @intFromBool(v != 0);
        }
    }
    try std.testing.expect(nonzero > rows * heads * taps / 2);
}

test "eight distances a thread give the tiled distance bias's bits, plain and packed" {
    const ctx = try gpu.open();
    defer ctx.deinit();
    const rows = 8;
    const row0 = 2;
    const heads = 3;
    const extent = 24;
    const out_off = (rows + row0) * heads * 16;
    const region = (rows + row0) * heads * extent;
    const arena = try ctx.alloc(out_off + 4 * region);
    defer arena.deinit();
    const f = arena.floats();
    for (f[0..out_off], 0..) |*v, i| v.* = @as(f32, @floatFromInt(@as(i32, @intCast(i * 7 % 23)) - 11)) * 0.03125;
    const weights = try ctx.alloc(16 * extent / 2);
    defer weights.deinit();
    const w16 = std.mem.bytesAsSlice(u16, std.mem.sliceAsBytes(weights.floats()));
    for (w16[0 .. 16 * extent], 0..) |*h, i| {
        const v: f32 = @as(f32, @floatFromInt(@as(i32, @intCast(i * 5 % 29)) - 14)) * 0.0078125;
        h.* = @intCast(@as(u32, @bitCast(v)) >> 16);
    }
    // A full extent, and causal prefixes that end inside and at the end of a block of eight.
    for ([_]u32{ 24, 13, 16, 1 }) |taps| {
        @memset(f[out_off..], -7);
        var p: gpu.Params = .{ .rows = rows, .row0 = row0, .x = 0, .xs = heads * 16, .y = out_off, .ys = heads * extent, .n = heads * extent, .m = extent, .dim = 16, .dtype = 2, .taps = taps, .flags = gpu.flag_round_bf16 };
        const kernels = [_][:0]const u8{ "mk_grouped_project_bf16x4", "mk_grouped_project_bf16d8", "mk_grouped_project_bf16x4_pack", "mk_grouped_project_bf16d8_pack" };
        for (kernels, 0..) |kernel, k| {
            p.y = out_off + @as(u32, @intCast(k)) * region;
            const threads: usize = if (k % 2 == 0) rows / 4 * heads * taps else rows / 2 * heads * ((taps + 7) / 8);
            try gpu.dispatch(ctx, kernel, &.{ arena, weights }, &p, threads, 256);
        }
        try ctx.finish();
        try std.testing.expectEqualSlices(f32, f[out_off..][0..region], f[out_off + region ..][0..region]);
        try std.testing.expectEqualSlices(f32, f[out_off + 2 * region ..][0..region], f[out_off + 3 * region ..][0..region]);
        try std.testing.expect(f[out_off + row0 * heads * extent] != -7);
    }
}

test "striped relative attention matches the reference over 16-bit K and V, a wrapped window and uneven stripes" {
    const ctx = try gpu.open();
    defer ctx.deinit();
    const heads = 6;
    const kv_heads = 2;
    const dim = 128;
    const stride = heads * dim;
    const width = kv_heads * dim;
    const extent = 9;
    const row0 = 2;
    const pos = 700;
    const end = pos + row0 + 1;
    const bias_off = (row0 + 1) * stride * 3;
    const sink_off = bias_off + (row0 + 1) * heads * extent;
    const arena = try ctx.alloc(sink_off + (row0 + 1) * heads);
    defer arena.deinit();
    const f = arena.floats();
    for (f[0 .. (row0 + 1) * stride], 0..) |*v, i| v.* = @as(f32, @floatFromInt(@as(i32, @intCast(i * 5 % 23)) - 11)) * 0.0625;
    for (f[bias_off..sink_off], 0..) |*v, i| v.* = @as(f32, @floatFromInt(@as(i32, @intCast(i % 17)) - 8)) * 0.125;
    for (f[sink_off..], 0..) |*v, i| v.* = @as(f32, @floatFromInt(@as(i32, @intCast(i % 5)) - 2)) * 0.5;
    const out_off = (row0 + 1) * stride;
    for ([_][2]u32{ .{ 0, 3 }, .{ 0, 2 }, .{ 300, 2 }, .{ 300, 7 }, .{ 1, 1 } }) |case| {
        const window = case[0];
        const capacity: u32 = if (window == 0) end else window + 40;
        const keys = try ctx.alloc(capacity * width);
        defer keys.deinit();
        const values = try ctx.alloc(capacity * width);
        defer values.deinit();
        const half_keys = try ctx.alloc((capacity + 16) * width / 2);
        defer half_keys.deinit();
        const half_values = try ctx.alloc((capacity + 16) * width / 2);
        defer half_values.deinit();
        // Stored through the engine's own kernels, a row at a time, so the mirror
        // is checked as the float cache's upper halves.
        var store: gpu.Params = .{ .rows = 1, .x = 0, .xs = width, .n = width, .capacity = capacity };
        const row = try ctx.alloc(width);
        defer row.deinit();
        for ([_][2]@TypeOf(keys){ .{ keys, half_keys }, .{ values, half_values } }, 0..) |pair, which| {
            for (end - @min(end, capacity)..end) |t| {
                for (row.floats(), 0..) |*v, d| v.* = @as(f32, @floatFromInt(@as(i32, @intCast((t * (7 - 2 * which) + d * 3) % (13 + 6 * which))) - 6)) * 0.0625;
                store.pos = @intCast(t);
                try gpu.dispatch(ctx, "mk_store", &.{ row, pair[0] }, &store, width, 256);
                try gpu.dispatch(ctx, "mk_pack_bf16", &.{ row, pair[1] }, &store, width, 256);
                try ctx.finish();
            }
            const half = std.mem.bytesAsSlice(u16, std.mem.sliceAsBytes(pair[1].floats()));
            for (pair[0].floats(), half[0 .. capacity * width]) |v, h| try std.testing.expectEqual(@as(u32, @bitCast(v)), @as(u32, h) << 16);
            try std.testing.expectEqualSlices(u16, half[0 .. 16 * width], half[capacity * width ..][0 .. 16 * width]);
        }
        var p: gpu.Params = .{ .rows = 1, .row0 = row0, .pos = pos, .x = 0, .xs = stride, .y = out_off, .ys = stride, .z = bias_off, .zs = heads * extent, .s = sink_off, .every = heads, .m = extent, .heads = heads, .kv_heads = kv_heads, .dim = dim, .window = window, .capacity = capacity, .scale = 0.0883883476 };
        try gpu.dispatch(ctx, "mk_relative_attention", &.{ arena, keys, values }, &p, heads * 32, 32);
        p.y += out_off;
        p.taps = case[1];
        const parts = try ctx.alloc(heads * p.taps * 130);
        defer parts.deinit();
        @memset(f[p.y..][0 .. (row0 + 1) * stride], -7);
        try gpu.dispatch(ctx, "mk_relative_long_keys", &.{ arena, half_keys, half_values, parts }, &p, kv_heads * p.taps * 512, 512);
        try gpu.dispatch(ctx, "mk_relative_long_merge", &.{ arena, parts }, &p, heads * 128, 128);
        try ctx.finish();
        for (f[p.y..][0 .. row0 * stride]) |v| try std.testing.expectEqual(@as(f32, -7), v);
        for (f[out_off + row0 * stride ..][0..stride], f[p.y + row0 * stride ..][0..stride]) |expected, actual| try close(expected, actual, 1e-5);
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

test "ring-stepped short convolutions give the reference kernels' bits on a wrapped ring" {
    const ctx = try gpu.open();
    defer ctx.deinit();
    const C = 4;
    const heads = 2;
    const taps = 4;
    const rows = 6;
    const row0 = 1;
    const capacity = 8;
    const ring = try ctx.alloc(capacity * C);
    defer ring.deinit();
    for (ring.floats(), 0..) |*v, i| v.* = @as(f32, @floatFromInt(@as(i32, @intCast(i * 5 % 13)) - 6)) * 0.0625;
    // Short: one weight per (channel, tap). Dynamic: one static weight per (tap, channel).
    const w = try ctx.alloc(C * taps);
    defer w.deinit();
    for (w.floats(), 0..) |*v, i| v.* = @as(f32, @floatFromInt(@as(i32, @intCast(i * 3 % 11)) - 5)) * 0.125;
    const T = rows + row0;
    const residual_off = 0;
    const dynamic_off = T * C;
    const out_off = dynamic_off + T * heads * taps;
    const buf = try ctx.alloc(out_off + 2 * T * C);
    defer buf.deinit();
    const f = buf.floats();
    for (f[0..out_off], 0..) |*v, i| v.* = @as(f32, @floatFromInt(@as(i32, @intCast(i * 7 % 17)) - 8)) * 0.03125;
    // Position 0 (taps before the sequence read zeros), then positions whose taps wrap the ring.
    for ([_]u32{ 0, 13 }) |pos| {
        for ([_][2][:0]const u8{ .{ "mk_inkling_short_conv", "mk_inkling_short_conv_ring" }, .{ "mk_inkling_dynamic_conv", "mk_inkling_dynamic_conv_ring" } }, 0..) |pair, kind| {
            for (pair, 0..) |kernel, which| {
                @memset(f[out_off + which * T * C ..][0 .. T * C], -7);
                var p: gpu.Params = if (kind == 0)
                    .{ .rows = rows, .row0 = row0, .pos = pos, .x = residual_off, .xs = C, .z = out_off + @as(u32, @intCast(which)) * T * C, .zs = C, .n = C, .taps = taps, .capacity = capacity, .scale = 0.4472136, .flags = 1 }
                else
                    .{ .rows = rows, .row0 = row0, .pos = pos, .x = dynamic_off, .xs = heads * taps, .y = residual_off, .ys = C, .z = out_off + @as(u32, @intCast(which)) * T * C, .zs = C, .n = C, .heads = heads, .taps = taps, .capacity = capacity, .scale = 0.3333333, .flags = 1 };
                try gpu.dispatch(ctx, kernel, &.{ ring, w, buf }, &p, rows * C, 32);
            }
            try ctx.finish();
            const reference = f[out_off..][0 .. T * C];
            for (reference[0 .. row0 * C]) |v| try std.testing.expectEqual(@as(f32, -7), v);
            for (reference[row0 * C ..]) |v| try std.testing.expect(v != -7 and std.math.isFinite(v));
            try std.testing.expectEqualSlices(f32, reference, f[out_off + T * C ..][0 .. T * C]);
        }
    }
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

test "route lists built from segments equal the end-to-end scan's" {
    const ctx = try gpu.open();
    defer ctx.deinit();
    const experts = 7;
    const top_k = 3;
    const row0 = 2;
    // Segments of one block of 32, a last segment that is short, and segments past the end.
    for ([_]u32{ 300, 11, 1 }) |rows| {
        const total = rows * top_k;
        const arena = try ctx.alloc((rows + row0) * 2 * top_k);
        defer arena.deinit();
        for (0..rows + row0) |r| for (0..top_k) |s| {
            arena.floats()[r * 2 * top_k + s] = @floatFromInt((r * 5 + s * 3 + r / 7) % experts);
        };
        const words = experts * (total + 1 + 32);
        const scan = try ctx.alloc(words);
        defer scan.deinit();
        const split = try ctx.alloc(words);
        defer split.deinit();
        @memset(scan.structs(u32), 0xffffffff);
        @memset(split.structs(u32), 0xffffffff);
        var p: gpu.ExpertParams = .{ .rows = rows, .row0 = row0, .routes = 0, .rs = 2 * top_k, .experts = experts, .top_k = top_k };
        try gpu.dispatch(ctx, "mk_expert_row_lists_simd", &.{ arena, scan }, &p, experts * 32, 128);
        try gpu.dispatch(ctx, "mk_expert_row_counts", &.{ arena, split }, &p, experts * 32 * 32, 128);
        try gpu.dispatch(ctx, "mk_expert_row_fill", &.{ arena, split }, &p, experts * 32 * 32, 128);
        try ctx.finish();
        var listed: usize = 0;
        for (0..experts) |e| {
            const a = scan.structs(u32)[e * (total + 1) ..][0 .. total + 1];
            const b = split.structs(u32)[e * (total + 1) ..][0 .. total + 1];
            try std.testing.expectEqualSlices(u32, a[0 .. a[0] + 1], b[0 .. a[0] + 1]);
            listed += a[0];
        }
        try std.testing.expectEqual(@as(usize, total), listed);
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

test "LoopCD fixed and adaptive contrast kernels match FP64 CPU oracle" {
    const ctx = try gpu.open();
    defer ctx.deinit();
    var prng = rng(42);
    const r = prng.random();

    const rows: u32 = 4;
    const row0: u32 = 1;
    const total_rows: u32 = row0 + rows + 1;
    const n: u32 = 600;
    const xs: u32 = n + 7;
    const ys: u32 = n + 11;
    const zs: u32 = n + 15;
    const strong_off: u32 = 0;
    const weak_off: u32 = total_rows * xs;
    const out_off: u32 = weak_off + total_rows * ys;
    const arena = try ctx.alloc(out_off + total_rows * zs);
    defer arena.deinit();
    const af = arena.floats();

    for (0..total_rows) |row_i| {
        for (0..n) |c| {
            af[strong_off + row_i * xs + c] = fromBf16(bf16(0.5 * r.floatNorm(f32)));
            af[weak_off + row_i * ys + c] = fromBf16(bf16(0.5 * r.floatNorm(f32)));
        }
    }
    // Row 1 (row0+0): tied top-1 max -> p1 == p2 -> w = omega.
    af[strong_off + (row0 + 0) * xs + 3] = 4.5;
    af[strong_off + (row0 + 0) * xs + 411] = 4.5;
    // Row 2 (row0+1): peaked confident distribution -> w ~ 0.
    for (0..n) |c| af[strong_off + (row0 + 1) * xs + c] = -12.0;
    af[strong_off + (row0 + 1) * xs + 17] = 10.0;
    af[strong_off + (row0 + 1) * xs + 299] = -10.0;
    // Row 3 (row0+2): non-finite logit ignored cleanly by top-2/softmax.
    af[strong_off + (row0 + 2) * xs + 5] = std.math.inf(f32);
    af[strong_off + (row0 + 2) * xs + 7] = std.math.nan(f32);

    // 1. Fixed omega=0 must be bit-exact (sentinel rows untouched).
    @memset(af[out_off..][0 .. total_rows * zs], -9.0);
    var p: gpu.Params = .{
        .rows = rows,
        .row0 = row0,
        .x = strong_off,
        .xs = xs,
        .y = weak_off,
        .ys = ys,
        .z = out_off,
        .zs = zs,
        .n = n,
        .scale = 0.0,
        .flags = gpu.flag_round_bf16,
    };
    try gpu.dispatch(ctx, "mk_loopcd_contrast_fixed", &.{arena}, &p, rows * n, 256);
    try ctx.finish();
    for (0..n) |c| {
        try testing.expectEqual(@as(f32, -9.0), af[out_off + 0 * zs + c]);
        try testing.expectEqual(@as(f32, -9.0), af[out_off + (row0 + rows) * zs + c]);
        for (row0..row0 + rows) |row_i| {
            try testing.expectEqual(@as(u32, @bitCast(af[strong_off + row_i * xs + c])), @as(u32, @bitCast(af[out_off + row_i * zs + c])));
        }
    }

    // 2. Fixed omega=0.75 without and with BF16 rounding.
    p.scale = 0.75;
    p.flags = 0;
    try gpu.dispatch(ctx, "mk_loopcd_contrast_fixed", &.{arena}, &p, rows * n, 256);
    try ctx.finish();
    for (row0..row0 + rows) |row_i| for (0..n) |c| {
        const s = af[strong_off + row_i * xs + c];
        const w = af[weak_off + row_i * ys + c];
        if (!std.math.isFinite(s)) continue;
        const expected = @as(f64, s) + 0.75 * (@as(f64, s) - @as(f64, w));
        try close(expected, af[out_off + row_i * zs + c], 1e-5);
    };

    // 3. Adaptive omega=1.2 vs FP64 CPU oracle across tied, peaked, non-finite, and random rows.
    p.scale = 1.2;
    p.flags = 0;
    @memset(af[out_off..][0 .. total_rows * zs], -9.0);
    try gpu.dispatch(ctx, "mk_loopcd_contrast_adaptive", &.{arena}, &p, rows * 256, 256);
    try ctx.finish();
    for (row0..row0 + rows) |row_i| {
        var m1: f32 = -std.math.inf(f32);
        var m2: f32 = -std.math.inf(f32);
        for (0..n) |c| {
            const v = af[strong_off + row_i * xs + c];
            if (!std.math.isFinite(v)) continue;
            if (v >= m1) {
                m2 = m1;
                m1 = v;
            } else if (v > m2) {
                m2 = v;
            }
        }
        var z: f64 = 0;
        for (0..n) |c| {
            const v = af[strong_off + row_i * xs + c];
            if (std.math.isFinite(v)) z += @exp(@as(f64, v - m1));
        }
        const p1 = 1.0 / z;
        const p2 = if (std.math.isFinite(m2)) @exp(@as(f64, m2 - m1)) / z else 0.0;
        const gap = std.math.clamp(p1 - p2, 0.0, 1.0);
        const weight = 1.2 * (1.0 - gap);
        if (row_i == row0 + 0) try testing.expectApproxEqRel(@as(f64, 1.2), weight, 1e-6);
        if (row_i == row0 + 1) try testing.expectApproxEqAbs(@as(f64, 0.0), weight, 1e-6);
        for (0..n) |c| {
            const s = af[strong_off + row_i * xs + c];
            const wk = af[weak_off + row_i * ys + c];
            if (!std.math.isFinite(s)) continue;
            const expected = @as(f64, s) + weight * (@as(f64, s) - @as(f64, wk));
            try close(expected, af[out_off + row_i * zs + c], 2e-5);
        }
    }

    // 4. n=1 adaptive edge case (gap == 1 => w == 0).
    var p1_cfg = p;
    p1_cfg.rows = 1;
    p1_cfg.row0 = 0;
    p1_cfg.n = 1;
    af[strong_off + 0] = 3.25;
    af[weak_off + 0] = -1.5;
    try gpu.dispatch(ctx, "mk_loopcd_contrast_adaptive", &.{arena}, &p1_cfg, 256, 256);
    try ctx.finish();
    try testing.expectEqual(@as(f32, 3.25), af[out_off + 0]);

    // 5. In-place out == strong matches out-of-place bit-for-bit (fixed and adaptive).
    var p_inp = p;
    p_inp.scale = 0.85;
    p_inp.flags = 0;
    p_inp.z = out_off;
    p_inp.zs = zs;
    const ref_fixed = try testing.allocator.alloc(f32, rows * n);
    defer testing.allocator.free(ref_fixed);
    const ref_adapt = try testing.allocator.alloc(f32, rows * n);
    defer testing.allocator.free(ref_adapt);
    try gpu.dispatch(ctx, "mk_loopcd_contrast_fixed", &.{arena}, &p_inp, rows * n, 256);
    try ctx.finish();
    for (0..rows) |r_i| @memcpy(ref_fixed[r_i * n ..][0..n], af[out_off + (row0 + r_i) * zs ..][0..n]);
    try gpu.dispatch(ctx, "mk_loopcd_contrast_adaptive", &.{arena}, &p_inp, rows * 256, 256);
    try ctx.finish();
    for (0..rows) |r_i| @memcpy(ref_adapt[r_i * n ..][0..n], af[out_off + (row0 + r_i) * zs ..][0..n]);

    const strong_saved = try testing.allocator.alloc(f32, rows * n);
    defer testing.allocator.free(strong_saved);
    for (0..rows) |r_i| @memcpy(strong_saved[r_i * n ..][0..n], af[strong_off + (row0 + r_i) * xs ..][0..n]);

    p_inp.z = strong_off;
    p_inp.zs = xs;
    try gpu.dispatch(ctx, "mk_loopcd_contrast_fixed", &.{arena}, &p_inp, rows * n, 256);
    try ctx.finish();
    for (0..rows) |r_i| for (0..n) |c| {
        try testing.expectEqual(@as(u32, @bitCast(ref_fixed[r_i * n + c])), @as(u32, @bitCast(af[strong_off + (row0 + r_i) * xs + c])));
        af[strong_off + (row0 + r_i) * xs + c] = strong_saved[r_i * n + c];
    };
    try gpu.dispatch(ctx, "mk_loopcd_contrast_adaptive", &.{arena}, &p_inp, rows * 256, 256);
    try ctx.finish();
    for (0..rows) |r_i| for (0..n) |c| {
        try testing.expectEqual(@as(u32, @bitCast(ref_adapt[r_i * n + c])), @as(u32, @bitCast(af[strong_off + (row0 + r_i) * xs + c])));
    };
}

test "Fused 4x2 head softcap LoopCD (fixed & dual) and parallel 2-pass adaptive contrast" {
    const ctx = try gpu.open();
    defer ctx.deinit();
    const inkling_plan = @import("inkling_plan.zig");

    var prng = rng(44);
    const r = prng.random();
    const n: u32 = 256;
    const m: u32 = 4099; // non-multiple of 4 and large enough to span 64 groups
    const rows: u32 = 2;
    const scale: f32 = 30.0;
    const groups = inkling_plan.argmax_groups;
    const span = (m + groups - 1) / groups;

    const w_buf = try ctx.alloc((@as(usize, m) * n + 1) / 2);
    defer w_buf.deinit();
    const wu = std.mem.bytesAsSlice(u16, std.mem.sliceAsBytes(w_buf.floats()));
    for (wu[0 .. @as(usize, m) * n]) |*elem| elem.* = bf16(0.15 * r.floatNorm(f32));

    const xs_off: u32 = 0;
    const xw_off: u32 = xs_off + rows * n;
    const y_off: u32 = xw_off + rows * n;
    const wlogits_off: u32 = y_off + rows * m;
    const ref_s_off: u32 = wlogits_off + rows * m;
    const ref_w_off: u32 = ref_s_off + rows * m;
    const ref_out_off: u32 = ref_w_off + rows * m;
    const arena = try ctx.alloc(ref_out_off + rows * m);
    defer arena.deinit();
    const af = arena.floats();
    const partials = try ctx.alloc(@as(usize, rows) * groups * 3);
    defer partials.deinit();

    for (af[xs_off .. xs_off + rows * n]) |*v| v.* = fromBf16(bf16(0.5 * r.floatNorm(f32)));
    for (af[xw_off .. xw_off + rows * n]) |*v| v.* = fromBf16(bf16(0.5 * r.floatNorm(f32)));

    var ps_ref: gpu.Params = .{ .rows = 1, .row0 = 0, .x = xs_off, .xs = n, .y = ref_s_off, .ys = m, .n = n, .m = m, .scale = scale };
    var pw_ref: gpu.Params = .{ .rows = 1, .row0 = 0, .x = xw_off, .xs = n, .y = ref_w_off, .ys = m, .n = n, .m = m, .scale = scale };
    var p_dual: gpu.Params = .{
        .rows = 1,
        .row0 = 0,
        .x = xs_off,
        .xs = n,
        .z = xw_off,
        .zs = n,
        .y = y_off,
        .ys = m,
        .every = wlogits_off,
        .pad_k = m,
        .n = n,
        .m = m,
        .scale = scale,
        .eps = 0.75,
    };
    const Variant = struct { ref_k: [:0]const u8, dual_k: [:0]const u8, fix_k: [:0]const u8 };
    for ([_]Variant{
        .{ .ref_k = "mk_bf16_linear_4x1_softcap", .dual_k = "mk_bf16_linear_4x2_softcap_dual", .fix_k = "mk_bf16_linear_4x2_softcap_loopcd" },
        .{ .ref_k = "mk_bf16_linear_4x1_softcap_vec8", .dual_k = "mk_bf16_linear_4x2_softcap_dual_vec8", .fix_k = "mk_bf16_linear_4x2_softcap_loopcd_vec8" },
    }) |vk| {
        try gpu.dispatch(ctx, vk.ref_k, &.{ w_buf, arena }, &ps_ref, ((@as(usize, m) + 3) / 4) * 32, 128);
        try gpu.dispatch(ctx, vk.ref_k, &.{ w_buf, arena }, &pw_ref, ((@as(usize, m) + 3) / 4) * 32, 128);

        // 1. Dual 4x2 must be bit-identical to separate 4x1 softcap projections of the same vector width.
        @memset(af[y_off .. y_off + m], -1.0);
        @memset(af[wlogits_off .. wlogits_off + m], -1.0);
        try gpu.dispatch(ctx, vk.dual_k, &.{ w_buf, arena }, &p_dual, ((@as(usize, m) + 3) / 4) * 32, 128);
        try ctx.finish();
        for (0..m) |c| {
            try testing.expectEqual(@as(u32, @bitCast(af[ref_s_off + c])), @as(u32, @bitCast(af[y_off + c])));
            try testing.expectEqual(@as(u32, @bitCast(af[ref_w_off + c])), @as(u32, @bitCast(af[wlogits_off + c])));
        }

        // 2. Fixed 4x2 (omega=0 bit-exact, omega=0.75 bit-identical to reference 4x1 + fixed contrast).
        var p_fix_ref: gpu.Params = .{ .rows = 1, .row0 = 0, .x = ref_s_off, .xs = m, .y = ref_w_off, .ys = m, .z = ref_out_off, .zs = m, .n = m, .scale = 0.75 };
        try gpu.dispatch(ctx, "mk_loopcd_contrast_fixed", &.{arena}, &p_fix_ref, m, 256);

        p_dual.eps = 0.0;
        try gpu.dispatch(ctx, vk.fix_k, &.{ w_buf, arena }, &p_dual, ((@as(usize, m) + 3) / 4) * 32, 128);
        try ctx.finish();
        for (0..m) |c| try testing.expectEqual(@as(u32, @bitCast(af[ref_s_off + c])), @as(u32, @bitCast(af[y_off + c])));

        p_dual.eps = 0.75;
        try gpu.dispatch(ctx, vk.fix_k, &.{ w_buf, arena }, &p_dual, ((@as(usize, m) + 3) / 4) * 32, 128);
        try ctx.finish();
        for (0..m) |c| try testing.expectEqual(@as(u32, @bitCast(af[ref_out_off + c])), @as(u32, @bitCast(af[y_off + c])));
    }

    // 3. Parallel 2-pass adaptive contrast across rows=2 (ties in row 0, peaked in row 1, in-place out == strong).
    for (0..rows) |r_i| for (0..m) |c| {
        af[ref_s_off + r_i * m + c] = 0.25 * r.floatNorm(f32);
        af[ref_w_off + r_i * m + c] = 0.25 * r.floatNorm(f32);
    };
    af[ref_s_off + 3] = 5.0;
    af[ref_s_off + 2001] = 5.0;
    af[ref_s_off + m + 99] = 12.0;
    af[ref_s_off + m + 2500] = -10.0;
    @memcpy(af[y_off .. y_off + rows * m], af[ref_s_off .. ref_s_off + rows * m]);

    var p_ad_ref: gpu.Params = .{ .rows = rows, .row0 = 0, .x = ref_s_off, .xs = m, .y = ref_w_off, .ys = m, .z = ref_out_off, .zs = m, .n = m, .scale = 1.0 };
    var p_ad_par: gpu.Params = .{ .rows = rows, .row0 = 0, .x = y_off, .xs = m, .y = ref_w_off, .ys = m, .z = y_off, .zs = m, .n = m, .m = span, .every = groups, .scale = 1.0 };
    try gpu.dispatch(ctx, "mk_loopcd_contrast_adaptive", &.{arena}, &p_ad_ref, rows * 256, 256);
    try gpu.dispatch(ctx, "mk_loopcd_adaptive_part", &.{ arena, partials }, &p_ad_par, rows * groups * 256, 256);
    try gpu.dispatch(ctx, "mk_loopcd_adaptive_apply", &.{ arena, partials }, &p_ad_par, rows * groups * 256, 256);
    try ctx.finish();
    for (0..rows) |r_i| for (0..m) |c| {
        try close(af[ref_out_off + r_i * m + c], af[y_off + r_i * m + c], 2e-5);
    };
}

fn lsaAlign256(v: u64) u64 {
    return (v + 255) & ~@as(u64, 255);
}

test "LSA MH Q4 pooling and 1-stage / 2-stage needle recovery with learned sink" {
    const ctx = try gpu.open();
    defer ctx.deinit();

    const heads: u32 = 6;
    const kv_heads: u32 = 2;
    const dim: u32 = 128;
    const budget_blocks: u32 = 8;
    const window_blocks: u32 = 8;
    const coarse_cap: u32 = 16;
    const list_stride: u32 = 1 + budget_blocks + window_blocks;
    const meta_bytes: u32 = kv_heads * 4;
    const row_bytes: u32 = ((meta_bytes + kv_heads * (dim / 2)) + 3) & ~@as(u32, 3);
    const max_keys: u32 = 6144;
    const max_blocks: u32 = max_keys / 4;
    const score_items: u32 = max_blocks;
    const tiles_x: u32 = (max_blocks + 255) / 256;

    var cur: u64 = 0;
    const off_hs = cur;
    cur = lsaAlign256(cur + @as(u64, score_items) * heads * 2);
    const off_ts = cur;
    cur = lsaAlign256(cur + @as(u64, heads) * @max(tiles_x, 256) * 8);
    const off_hl = cur;
    cur = lsaAlign256(cur + @as(u64, heads) * 4);
    const off_sc = cur;
    cur = lsaAlign256(cur + @as(u64, score_items) * 4);
    const off_hi = cur;
    cur = lsaAlign256(cur + 512 * 4);
    const off_cc = cur;
    cur = lsaAlign256(cur + 4);
    const off_cm = cur;
    cur = lsaAlign256(cur + 4096 * 4);
    const off_cu = cur;
    cur = lsaAlign256(cur + 4096 * 4);
    const off_fc = cur;
    cur = lsaAlign256(cur + @as(u64, coarse_cap) * 4 * 4);
    const off_fn = cur;
    cur = lsaAlign256(cur + 4);

    const q_off: u32 = 0;
    const sink_off: u32 = heads * dim;
    const arena = try ctx.alloc(sink_off + heads);
    defer arena.deinit();
    const k32 = try ctx.alloc(@as(usize, max_keys) * kv_heads * dim);
    defer k32.deinit();
    const k16 = try ctx.alloc(@as(usize, max_keys) * kv_heads * dim / 2);
    defer k16.deinit();
    const fine_f32 = try ctx.alloc((@as(usize, max_blocks) * row_bytes + 3) / 4);
    defer fine_f32.deinit();
    const fine_bf16 = try ctx.alloc((@as(usize, max_blocks) * row_bytes + 3) / 4);
    defer fine_bf16.deinit();
    const coarse_buf = try ctx.alloc((@as(usize, max_blocks / 4) * row_bytes + 3) / 4);
    defer coarse_buf.deinit();
    const lsa_scratch = try ctx.alloc(@intCast((cur + 3) / 4));
    defer lsa_scratch.deinit();
    @memset(std.mem.sliceAsBytes(lsa_scratch.floats()), 0);
    const block_list = try ctx.alloc(list_stride);
    defer block_list.deinit();
    const block_count = try ctx.alloc(1);
    defer block_count.deinit();

    const af = arena.floats();
    @memset(af, 0);
    // 3 planted needle signatures mapped across heads (0,1)->band0, (2,3)->band1, (4,5)->band2.
    for (0..heads) |qh| {
        const band = qh / 2;
        for (0..32) |j| af[q_off + qh * dim + band * 32 + j] = 2.0;
    }
    for (0..heads) |qh| af[sink_off + qh] = -5.0;

    var prng = rng(99);
    const r = prng.random();
    const k32f = k32.floats();
    const k16u = std.mem.bytesAsSlice(u16, std.mem.sliceAsBytes(k16.floats()));
    for (0..max_keys) |t| {
        for (0..kv_heads) |kvh| {
            for (0..dim) |d| {
                const bg = if (d >= 96) fromBf16(bf16(0.05 * r.floatNorm(f32))) else 0.0;
                const idx = (t * kv_heads + kvh) * dim + d;
                k32f[idx] = bg;
                k16u[idx] = bf16(bg);
            }
        }
    }
    // Pool F32 vs BF16 parity on needle-rich 1-stage and 2-stage configurations.
    const Case = struct { keys: u32, needles: [3]u32, two_stage: bool };
    for ([_]Case{
        .{ .keys = 512, .needles = .{ 15, 53, 91 }, .two_stage = false },
        .{ .keys = 4096, .needles = .{ 77, 412, 889 }, .two_stage = true },
        .{ .keys = 5120, .needles = .{ 77, 512, 1180 }, .two_stage = false },
    }) |tc| {
        for (tc.needles, 0..) |nb, bi| {
            for (0..4) |off| {
                const t = nb * 4 + off;
                for (0..kv_heads) |kvh| {
                    for (0..32) |j| {
                        const idx = (t * kv_heads + kvh) * dim + bi * 32 + j;
                        k32f[idx] = 2.0;
                        k16u[idx] = bf16(2.0);
                    }
                }
            }
        }
        var lp: gpu.LSAParams = .{
            .off_head_scores = off_hs,
            .off_tg_stats = off_ts,
            .off_head_lse = off_hl,
            .off_scores = off_sc,
            .off_hist = off_hi,
            .off_cand_count = off_cc,
            .off_cand_m = off_cm,
            .off_cand_u = off_cu,
            .off_fine_cand = off_fc,
            .off_fine_count = off_fn,
            .units = 1,
            .rows = 1,
            .pos0 = tc.keys - 1,
            .dim = dim,
            .heads = heads,
            .kv_heads = kv_heads,
            .capacity = max_keys,
            .q_off = q_off,
            .q_stride = heads * dim,
            .sink_off = sink_off,
            .sink_stride = heads,
            .budget_blocks = budget_blocks,
            .window_blocks = window_blocks,
            .coarse_cap = coarse_cap,
            .pooled_meta_bytes = meta_bytes,
            .pooled_row_bytes = row_bytes,
            .num_blks = tc.keys / 4,
            .list_stride = list_stride,
            .score_stride = score_items,
            .head_score_stride = score_items,
            .cand_cap = 4096,
            .score_tiles_x = tiles_x,
            .has_sink = 1,
            .scale = 1.0 / @sqrt(@as(f32, dim)),
        };
        try gpu.dispatch(ctx, "mk_lsa_pool_fine_f32", &.{ k32, fine_f32 }, &lp, @as(usize, lp.num_blks) * 128, 128);
        try gpu.dispatch(ctx, "mk_lsa_pool_fine_bf16", &.{ k16, fine_bf16 }, &lp, @as(usize, lp.num_blks) * 128, 128);
        try ctx.finish();
        const pool_bytes = @as(usize, lp.num_blks) * row_bytes;
        try testing.expectEqualSlices(u8, std.mem.sliceAsBytes(fine_f32.floats())[0..pool_bytes], std.mem.sliceAsBytes(fine_bf16.floats())[0..pool_bytes]);

        if (!tc.two_stage) {
            lp.stage_mode = 0;
            lp.bq_tile = 1;
            lp.score_tiles_x = @max((tc.keys / 4 + 255) / 256, 1);
            try gpu.dispatch(ctx, "mk_lsa_mh_score", &.{ arena, fine_bf16, lsa_scratch }, &lp, @as(usize, lp.score_tiles_x) * 256, 256);
            try gpu.dispatch(ctx, "mk_lsa_mh_reduce_lse", &.{ arena, lsa_scratch }, &lp, heads * 32, 32);
            try gpu.dispatch(ctx, "mk_lsa_mh_agg_hist", &.{lsa_scratch}, &lp, @as(usize, lp.score_tiles_x) * 256, 256);
            try gpu.dispatch(ctx, "mk_lsa_mh_collect", &.{lsa_scratch}, &lp, @as(usize, lp.score_tiles_x) * 256, 256);
            try gpu.dispatch(ctx, "mk_lsa_mh_finalize", &.{ lsa_scratch, block_list, block_count }, &lp, 256, 256);
        } else {
            lp.num_blks = tc.keys / 16;
            try gpu.dispatch(ctx, "mk_lsa_pool_coarse", &.{ fine_bf16, coarse_buf }, &lp, @as(usize, lp.num_blks) * 128, 128);
            lp.stage_mode = 1;
            lp.bq_tile = 1;
            lp.score_tiles_x = @max((tc.keys / 16 + 255) / 256, 1);
            try gpu.dispatch(ctx, "mk_lsa_mh_score", &.{ arena, coarse_buf, lsa_scratch }, &lp, @as(usize, lp.score_tiles_x) * 256, 256);
            try gpu.dispatch(ctx, "mk_lsa_mh_reduce_lse", &.{ arena, lsa_scratch }, &lp, heads * 32, 32);
            try gpu.dispatch(ctx, "mk_lsa_mh_agg_hist", &.{lsa_scratch}, &lp, @as(usize, lp.score_tiles_x) * 256, 256);
            try gpu.dispatch(ctx, "mk_lsa_mh_collect", &.{lsa_scratch}, &lp, @as(usize, lp.score_tiles_x) * 256, 256);
            try gpu.dispatch(ctx, "mk_lsa_mh_coarse_expand", &.{lsa_scratch}, &lp, 256, 256);
            lp.stage_mode = 2;
            lp.score_tiles_x = @max((coarse_cap * 4 + 1 + window_blocks + 255) / 256, 1);
            try gpu.dispatch(ctx, "mk_lsa_mh_score", &.{ arena, fine_bf16, lsa_scratch }, &lp, @as(usize, lp.score_tiles_x) * 256, 256);
            try gpu.dispatch(ctx, "mk_lsa_mh_reduce_lse", &.{ arena, lsa_scratch }, &lp, heads * 32, 32);
            try gpu.dispatch(ctx, "mk_lsa_mh_agg_hist", &.{lsa_scratch}, &lp, @as(usize, lp.score_tiles_x) * 256, 256);
            try gpu.dispatch(ctx, "mk_lsa_mh_collect", &.{lsa_scratch}, &lp, @as(usize, lp.score_tiles_x) * 256, 256);
            try gpu.dispatch(ctx, "mk_lsa_mh_finalize", &.{ lsa_scratch, block_list, block_count }, &lp, 256, 256);
        }
        try ctx.finish();

        const cnt = block_count.structs(u32)[0];
        try testing.expectEqual(list_stride, cnt);
        const list = block_list.structs(u32)[0..cnt];
        try testing.expectEqual(@as(u32, 0), list[0]);
        for (1..cnt) |i| try testing.expect(list[i - 1] < list[i]);
        for (tc.needles) |nb| {
            var found = false;
            for (list[1 .. 1 + budget_blocks]) |b| if (b == nb) {
                found = true;
                break;
            };
            try testing.expect(found);
        }
        const total_blks = tc.keys / 4;
        for (0..window_blocks) |w| {
            try testing.expectEqual(total_blks - window_blocks + @as(u32, @intCast(w)), list[1 + budget_blocks + w]);
        }
        // Scratch counters must be zeroed after finalize.
        const sbytes = std.mem.sliceAsBytes(lsa_scratch.floats());
        for (std.mem.bytesAsSlice(u32, sbytes[@intCast(off_hi)..][0 .. 512 * 4])) |v| try testing.expectEqual(@as(u32, 0), v);
        try testing.expectEqual(@as(u32, 0), std.mem.bytesAsSlice(u32, sbytes[@intCast(off_cc)..][0..4])[0]);

        // Fast single-row decode selection (mk_lsa_mh_score_dec + dynamic tile_items) matches reference 256-item selection.
        const dec_block_list = try ctx.alloc(list_stride);
        defer dec_block_list.deinit();
        const dec_block_count = try ctx.alloc(1);
        defer dec_block_count.deinit();
        const dec_scores_ref = try testing.allocator.alloc(u16, @as(usize, max_blocks) * heads);
        defer testing.allocator.free(dec_scores_ref);
        const dec_lse_ref = try testing.allocator.alloc(f32, heads);
        defer testing.allocator.free(dec_lse_ref);
        @memcpy(dec_scores_ref, std.mem.bytesAsSlice(u16, sbytes[@intCast(off_hs)..][0 .. @as(usize, max_blocks) * heads * 2]));
        @memcpy(dec_lse_ref, std.mem.bytesAsSlice(f32, sbytes[@intCast(off_hl)..][0 .. @as(usize, heads) * 4]));
        const ref_list_copy = try testing.allocator.dupe(u32, list);
        defer testing.allocator.free(ref_list_copy);

        const exec_mod = @import("exec.zig");
        var lp_dec = lp;
        lp_dec.tile_items = 8;
        if (!tc.two_stage) {
            const items = tc.keys / 4;
            const dtile = exec_mod.Engine.lsaDecodeScoreTile(items);
            lp_dec.stage_mode = 0;
            lp_dec.bq_tile = 1;
            lp_dec.tile_items = dtile;
            lp_dec.score_tiles_x = @max((items + dtile - 1) / dtile, 1);
            try gpu.dispatch(ctx, "mk_lsa_mh_score_dec", &.{ arena, fine_bf16, lsa_scratch }, &lp_dec, @as(usize, lp_dec.score_tiles_x) * 256, 256);
            try gpu.dispatch(ctx, "mk_lsa_mh_reduce_lse", &.{ arena, lsa_scratch }, &lp_dec, heads * 32, 32);
            lp_dec.tile_items = 32;
            lp_dec.score_tiles_x = @max((items + 31) / 32, 1);
            try gpu.dispatch(ctx, "mk_lsa_mh_agg_hist", &.{lsa_scratch}, &lp_dec, @as(usize, lp_dec.score_tiles_x) * 256, 256);
            lp_dec.tile_items = 256;
            lp_dec.score_tiles_x = @max((items + 255) / 256, 1);
            try gpu.dispatch(ctx, "mk_lsa_mh_collect", &.{lsa_scratch}, &lp_dec, @as(usize, lp_dec.score_tiles_x) * 256, 256);
            try gpu.dispatch(ctx, "mk_lsa_mh_finalize", &.{ lsa_scratch, dec_block_list, dec_block_count }, &lp_dec, 256, 256);
        } else {
            const c_items = tc.keys / 16;
            const dtile1 = exec_mod.Engine.lsaDecodeScoreTile(c_items);
            lp_dec.stage_mode = 1;
            lp_dec.bq_tile = 1;
            lp_dec.tile_items = dtile1;
            lp_dec.score_tiles_x = @max((c_items + dtile1 - 1) / dtile1, 1);
            try gpu.dispatch(ctx, "mk_lsa_mh_score_dec", &.{ arena, coarse_buf, lsa_scratch }, &lp_dec, @as(usize, lp_dec.score_tiles_x) * 256, 256);
            try gpu.dispatch(ctx, "mk_lsa_mh_reduce_lse", &.{ arena, lsa_scratch }, &lp_dec, heads * 32, 32);
            lp_dec.tile_items = 32;
            lp_dec.score_tiles_x = @max((c_items + 31) / 32, 1);
            try gpu.dispatch(ctx, "mk_lsa_mh_agg_hist", &.{lsa_scratch}, &lp_dec, @as(usize, lp_dec.score_tiles_x) * 256, 256);
            lp_dec.tile_items = 256;
            lp_dec.score_tiles_x = @max((c_items + 255) / 256, 1);
            try gpu.dispatch(ctx, "mk_lsa_mh_collect", &.{lsa_scratch}, &lp_dec, @as(usize, lp_dec.score_tiles_x) * 256, 256);
            try gpu.dispatch(ctx, "mk_lsa_mh_coarse_expand", &.{lsa_scratch}, &lp_dec, 256, 256);

            const s2_items = coarse_cap * 4 + 1 + window_blocks;
            const dtile2 = exec_mod.Engine.lsaDecodeScoreTile(s2_items);
            lp_dec.stage_mode = 2;
            lp_dec.tile_items = dtile2;
            lp_dec.score_tiles_x = @max((s2_items + dtile2 - 1) / dtile2, 1);
            try gpu.dispatch(ctx, "mk_lsa_mh_score_dec", &.{ arena, fine_bf16, lsa_scratch }, &lp_dec, @as(usize, lp_dec.score_tiles_x) * 256, 256);
            try gpu.dispatch(ctx, "mk_lsa_mh_reduce_lse", &.{ arena, lsa_scratch }, &lp_dec, heads * 32, 32);
            lp_dec.tile_items = 32;
            lp_dec.score_tiles_x = @max((s2_items + 31) / 32, 1);
            try gpu.dispatch(ctx, "mk_lsa_mh_agg_hist", &.{lsa_scratch}, &lp_dec, @as(usize, lp_dec.score_tiles_x) * 256, 256);
            lp_dec.tile_items = 256;
            lp_dec.score_tiles_x = @max((s2_items + 255) / 256, 1);
            try gpu.dispatch(ctx, "mk_lsa_mh_collect", &.{lsa_scratch}, &lp_dec, @as(usize, lp_dec.score_tiles_x) * 256, 256);
            try gpu.dispatch(ctx, "mk_lsa_mh_finalize", &.{ lsa_scratch, dec_block_list, dec_block_count }, &lp_dec, 256, 256);
        }
        try ctx.finish();
        const active_items = if (!tc.two_stage) (tc.keys / 4 - 1 - window_blocks) else (coarse_cap * 4);
        const dec_scores_got = std.mem.bytesAsSlice(u16, sbytes[@intCast(off_hs)..][0 .. active_items * heads * 2]);
        for (dec_scores_ref[0 .. active_items * heads], dec_scores_got) |rv, gv| {
            try testing.expectApproxEqAbs(@as(f32, @floatCast(@as(f16, @bitCast(rv)))), @as(f32, @floatCast(@as(f16, @bitCast(gv)))), 1e-2);
        }
        const dec_lse_got = std.mem.bytesAsSlice(f32, sbytes[@intCast(off_hl)..][0 .. @as(usize, heads) * 4]);
        for (dec_lse_ref, dec_lse_got) |rl, gl| try close(@as(f64, rl), gl, 1e-4);
        try testing.expectEqual(list_stride, dec_block_count.structs(u32)[0]);
        try testing.expectEqualSlices(u32, ref_list_copy, dec_block_list.structs(u32)[0..list_stride]);

        // 3-dispatch decode selection (mk_lsa_mh_score_dec -> mk_lsa_mh_agg_hist_dec -> mk_lsa_mh_finalize_dec) matches reference bit-for-bit.
        const fused_block_list = try ctx.alloc(list_stride);
        defer fused_block_list.deinit();
        const fused_block_count = try ctx.alloc(1);
        defer fused_block_count.deinit();
        var lp_fused = lp;
        if (!tc.two_stage) {
            const items = tc.keys / 4;
            const n_mid = items - 1 - window_blocks;
            const dtile = exec_mod.Engine.lsaDecodeScoreTile(items);
            lp_fused.stage_mode = 0;
            lp_fused.bq_tile = 1;
            lp_fused.tile_items = dtile;
            lp_fused.score_tiles_x = @max((items + dtile - 1) / dtile, 1);
            try gpu.dispatch(ctx, "mk_lsa_mh_score_dec", &.{ arena, fine_bf16, lsa_scratch }, &lp_fused, @as(usize, lp_fused.score_tiles_x) * 256, 256);
            const agg0 = exec_mod.Engine.lsaDecodeAggTile(n_mid);
            lp_fused.bq_tile = agg0;
            const agg0_tiles = @max((n_mid + agg0 - 1) / agg0, 1);
            try gpu.dispatch(ctx, "mk_lsa_mh_agg_hist_dec", &.{ arena, lsa_scratch }, &lp_fused, @as(usize, agg0_tiles) * 256, 256);
            try gpu.dispatch(ctx, "mk_lsa_mh_finalize_dec", &.{ lsa_scratch, fused_block_list, fused_block_count }, &lp_fused, 256, 256);
        } else {
            const c_items = tc.keys / 16;
            const dtile1 = exec_mod.Engine.lsaDecodeScoreTile(c_items);
            lp_fused.stage_mode = 1;
            lp_fused.bq_tile = 1;
            lp_fused.tile_items = dtile1;
            lp_fused.score_tiles_x = @max((c_items + dtile1 - 1) / dtile1, 1);
            try gpu.dispatch(ctx, "mk_lsa_mh_score_dec", &.{ arena, coarse_buf, lsa_scratch }, &lp_fused, @as(usize, lp_fused.score_tiles_x) * 256, 256);
            const agg1 = exec_mod.Engine.lsaDecodeAggTile(c_items);
            lp_fused.bq_tile = agg1;
            const agg1_tiles = @max((c_items + agg1 - 1) / agg1, 1);
            try gpu.dispatch(ctx, "mk_lsa_mh_agg_hist_dec", &.{ arena, lsa_scratch }, &lp_fused, @as(usize, agg1_tiles) * 256, 256);
            try gpu.dispatch(ctx, "mk_lsa_mh_finalize_dec", &.{ lsa_scratch, fused_block_list, fused_block_count }, &lp_fused, 256, 256);

            const s2_fine = coarse_cap * 4;
            const s2_items = s2_fine + 1 + window_blocks;
            const dtile2 = exec_mod.Engine.lsaDecodeScoreTile(s2_items);
            lp_fused.stage_mode = 2;
            lp_fused.tile_items = dtile2;
            lp_fused.score_tiles_x = @max((s2_items + dtile2 - 1) / dtile2, 1);
            try gpu.dispatch(ctx, "mk_lsa_mh_score_dec", &.{ arena, fine_bf16, lsa_scratch }, &lp_fused, @as(usize, lp_fused.score_tiles_x) * 256, 256);
            const agg2 = exec_mod.Engine.lsaDecodeAggTile(s2_fine);
            lp_fused.bq_tile = agg2;
            const agg2_tiles = @max((s2_fine + agg2 - 1) / agg2, 1);
            try gpu.dispatch(ctx, "mk_lsa_mh_agg_hist_dec", &.{ arena, lsa_scratch }, &lp_fused, @as(usize, agg2_tiles) * 256, 256);
            try gpu.dispatch(ctx, "mk_lsa_mh_finalize_dec", &.{ lsa_scratch, fused_block_list, fused_block_count }, &lp_fused, 256, 256);
        }
        try ctx.finish();
        try testing.expectEqual(list_stride, fused_block_count.structs(u32)[0]);
        try testing.expectEqualSlices(u32, ref_list_copy, fused_block_list.structs(u32)[0..list_stride]);
        const hist_cleared = std.mem.bytesAsSlice(u32, sbytes[@intCast(off_hi)..][0 .. 256 * 4]);
        for (hist_cleared) |hv| try testing.expectEqual(@as(u32, 0), hv);

        if (!tc.two_stage and (tc.keys / 4 - 1 - window_blocks) > 1024) {
            const items = tc.keys / 4;
            const n_mid = items - 1 - window_blocks;
            const dtile = exec_mod.Engine.lsaDecodeScoreTile(items);
            var lp_hyb = lp;
            lp_hyb.stage_mode = 0;
            lp_hyb.bq_tile = 1;
            lp_hyb.tile_items = dtile;
            lp_hyb.score_tiles_x = @max((items + dtile - 1) / dtile, 1);
            try gpu.dispatch(ctx, "mk_lsa_mh_score_dec", &.{ arena, fine_bf16, lsa_scratch }, &lp_hyb, @as(usize, lp_hyb.score_tiles_x) * 256, 256);
            const agg0 = exec_mod.Engine.lsaDecodeAggTile(n_mid);
            lp_hyb.bq_tile = agg0;
            const agg0_tiles = @max((n_mid + agg0 - 1) / agg0, 1);
            try gpu.dispatch(ctx, "mk_lsa_mh_agg_hist_dec", &.{ arena, lsa_scratch }, &lp_hyb, @as(usize, agg0_tiles) * 256, 256);
            lp_hyb.tile_items = 256;
            lp_hyb.score_tiles_x = @max((n_mid + 255) / 256, 1);
            try gpu.dispatch(ctx, "mk_lsa_mh_collect", &.{lsa_scratch}, &lp_hyb, @as(usize, lp_hyb.score_tiles_x) * 256, 256);
            try gpu.dispatch(ctx, "mk_lsa_mh_finalize", &.{ lsa_scratch, fused_block_list, fused_block_count }, &lp_hyb, 256, 256);
            try ctx.finish();
            try testing.expectEqual(list_stride, fused_block_count.structs(u32)[0]);
            try testing.expectEqualSlices(u32, ref_list_copy, fused_block_list.structs(u32)[0..list_stride]);
            for (hist_cleared) |hv| try testing.expectEqual(@as(u32, 0), hv);
        }

        // Also test dense bypass (total_blocks <= 1 + budget + window) writes blockCount == 0.
        var lp_dense = lp;
        lp_dense.stage_mode = 0;
        lp_dense.pos0 = 63; // 16 blocks <= 17
        try gpu.dispatch(ctx, "mk_lsa_mh_finalize", &.{ lsa_scratch, block_list, block_count }, &lp_dense, 256, 256);
        try gpu.dispatch(ctx, "mk_lsa_mh_finalize_dec", &.{ lsa_scratch, fused_block_list, fused_block_count }, &lp_dense, 256, 256);
        try ctx.finish();
        try testing.expectEqual(@as(u32, 0), block_count.structs(u32)[0]);
        try testing.expectEqual(@as(u32, 0), fused_block_count.structs(u32)[0]);
        for (hist_cleared) |hv| try testing.expectEqual(@as(u32, 0), hv);

        // Restore background noise before next case so needles don't leak across cases.
        for (tc.needles) |nb| {
            for (0..4) |off| {
                const t = nb * 4 + off;
                for (0..kv_heads) |kvh| {
                    for (0..32) |j| {
                        const bi_off: usize = if (tc.needles[0] == nb) 0 else if (tc.needles[1] == nb) 32 else 64;
                        const idx = (t * kv_heads + kvh) * dim + bi_off + j;
                        k32f[idx] = 0.0;
                        k16u[idx] = 0;
                    }
                }
            }
        }
    }
}

test "LSA sparse attention ref, fast decode, inline bias, and BF16 MMA prefill match FP64 oracle" {
    const ctx = try gpu.open();
    defer ctx.deinit();
    var prng = rng(123);
    const r = prng.random();

    const heads: u32 = 6;
    const kv_heads: u32 = 2;
    const dim: u32 = 128;
    const m: u32 = 16;
    const rel_dim: u32 = 16;
    const capacity: u32 = 128;
    const rows: u32 = 40;
    const row0: u32 = 2;
    const total_rows: u32 = row0 + rows;
    const pos0: u32 = 46; // covers keys up to pos0 + rows - 1 = 85
    const list_stride: u32 = 16;
    const units: u32 = 2; // covers 40 rows under both row_step=32 (2 tiles) and ref test

    const q_off: u32 = 0;
    const bias_off: u32 = total_rows * heads * dim;
    const r_off: u32 = bias_off + total_rows * heads * m;
    const sink_off: u32 = r_off + total_rows * heads * rel_dim;
    const gate_off: u32 = sink_off + total_rows * heads;
    const out_off: u32 = gate_off + total_rows * heads * dim;
    const arena = try ctx.alloc(out_off + total_rows * heads * dim);
    defer arena.deinit();
    const af = arena.floats();

    const k32 = try ctx.alloc(@as(usize, capacity) * kv_heads * dim);
    defer k32.deinit();
    const v32 = try ctx.alloc(@as(usize, capacity) * kv_heads * dim);
    defer v32.deinit();
    const k16 = try ctx.alloc(@as(usize, capacity) * kv_heads * dim / 2);
    defer k16.deinit();
    const v16 = try ctx.alloc(@as(usize, capacity) * kv_heads * dim / 2);
    defer v16.deinit();
    const parts = try ctx.alloc(@as(usize, heads) * 64 * 130);
    defer parts.deinit();
    const rel_w_buf = try ctx.alloc((@as(usize, rel_dim) * m + 1) / 2);
    defer rel_w_buf.deinit();
    const block_list = try ctx.alloc(@as(usize, rows) * list_stride);
    defer block_list.deinit();
    const block_count = try ctx.alloc(rows);
    defer block_count.deinit();

    for (af[q_off..][0 .. total_rows * heads * dim]) |*v| v.* = fromBf16(bf16(0.4 * r.floatNorm(f32)));
    for (af[bias_off..][0 .. total_rows * heads * m]) |*v| v.* = fromBf16(bf16(0.25 * r.floatNorm(f32)));
    for (af[r_off..][0 .. total_rows * heads * rel_dim]) |*v| v.* = fromBf16(bf16(0.3 * r.floatNorm(f32)));
    for (af[sink_off..][0 .. total_rows * heads]) |*v| v.* = fromBf16(bf16(0.5 * r.floatNorm(f32)));
    for (af[gate_off..][0 .. total_rows * heads * dim]) |*v| v.* = fromBf16(bf16(r.floatNorm(f32)));

    const rel_wu = std.mem.bytesAsSlice(u16, std.mem.sliceAsBytes(rel_w_buf.floats()));
    for (rel_wu[0 .. rel_dim * m]) |*w| w.* = bf16(0.25 * r.floatNorm(f32));

    for ([_][2]metal.Buffer{ .{ k32, k16 }, .{ v32, v16 } }) |pair| {
        const pf = pair[0].floats();
        const pu = std.mem.bytesAsSlice(u16, std.mem.sliceAsBytes(pair[1].floats()));
        for (pf, pu) |*fv, *hv| {
            fv.* = fromBf16(bf16(0.4 * r.floatNorm(f32)));
            hv.* = bf16(fv.*);
        }
    }

    const bcnt = block_count.structs(u32);
    const blist = block_list.structs(u32);
    // Unit 0: dense bypass (cnt = 0); Unit 1: sorted sparse blocks spanning prefix + middle + recent.
    bcnt[0] = 0;
    const u1_blks = [_]u32{ 0, 2, 5, 9, 13, 16, 17, 18, 19, 20 };
    bcnt[1] = u1_blks.len;
    @memcpy(blist[1 * list_stride ..][0..u1_blks.len], &u1_blks);
    _ = units;

    const scale = 1.0 / @sqrt(@as(f32, dim));
    const expected_tile = try testing.allocator.alloc(f32, rows * heads * dim);
    defer testing.allocator.free(expected_tile);
    const k32f = k32.floats();
    const v32f = v32.floats();

    // FP64 oracle for row_step=32 across all 40 rows (mixed dense tile 0 + sparse tile 1).
    for (0..rows) |r_i| {
        const row = row0 + r_i;
        const pos = pos0 + r_i;
        const unit = r_i / 32;
        const ref_r = @min(unit * 32 + 31, rows - 1);
        const ref_pos = pos0 + @as(u32, @intCast(ref_r));
        const cnt = bcnt[unit];
        const tail0: u32 = if (cnt > 0) ((ref_pos + 1) & ~@as(u32, 3)) else 0;
        const tail_end = pos + 1;
        const tail_cnt = if (tail_end > tail0) tail_end - tail0 else 0;
        const total_keys = cnt * 4 + tail_cnt;
        for (0..heads) |h| {
            const kvh = h / (heads / kv_heads);
            var max_s: f64 = @floatCast(af[sink_off + row * heads + h]);
            var denom: f64 = 1.0;
            var acc: [dim]f64 = @splat(0);
            for (0..total_keys) |idx| {
                const t: u32 = if (cnt > 0 and idx < cnt * 4)
                    (blist[unit * list_stride + idx / 4] << 2) | @as(u32, @intCast(idx & 3))
                else
                    @intCast(tail0 + (if (cnt > 0) idx - cnt * 4 else idx));
                if (t > pos) continue;
                var dot: f64 = 0;
                for (0..dim) |d| {
                    dot += @as(f64, af[q_off + row * heads * dim + h * dim + d]) * @as(f64, k32f[(t * kv_heads + kvh) * dim + d]);
                }
                const dist = pos - t;
                const rel: f64 = if (dist < m) @floatCast(af[bias_off + row * heads * m + h * m + dist]) else 0;
                const score = dot * scale + rel;
                const next_m = @max(max_s, score);
                const prev = @exp(max_s - next_m);
                const w = @exp(score - next_m);
                denom = denom * prev + w;
                for (0..dim) |d| acc[d] = acc[d] * prev + w * @as(f64, v32f[(t * kv_heads + kvh) * dim + d]);
                max_s = next_m;
            }
            for (0..dim) |d| expected_tile[(r_i * heads + h) * dim + d] = @floatCast(acc[d] / denom);
        }
    }

    // 1. General FP32 sparse reference with row_step = 32.
    var lp: gpu.LSAParams = .{
        .rows = rows,
        .row0 = row0,
        .pos0 = pos0,
        .row_step = 32,
        .dim = dim,
        .heads = heads,
        .kv_heads = kv_heads,
        .capacity = capacity,
        .q_off = q_off,
        .q_stride = heads * dim,
        .sink_off = sink_off,
        .sink_stride = heads,
        .bias_off = bias_off,
        .bias_stride = heads * m,
        .out_off = out_off,
        .out_stride = heads * dim,
        .m = m,
        .list_stride = list_stride,
        .has_sink = 1,
        .scale = scale,
    };
    @memset(af[out_off..], -7.0);
    try gpu.dispatch(ctx, "mk_relative_lsa_ref", &.{ arena, k32, v32, block_list, block_count }, &lp, rows * heads * 32, 32);
    try ctx.finish();
    for (0..rows) |r_i| for (0..heads * dim) |c| {
        try close(expected_tile[r_i * heads * dim + c], af[out_off + (row0 + r_i) * heads * dim + c], 2e-4);
    };

    // 2. Tiled BF16 MMA sparse prefill against FP64 oracle (and ref).
    @memset(af[out_off..], -7.0);
    try gpu.dispatch(ctx, "mk_relative_lsa_prefill", &.{ arena, k16, v16, block_list, block_count }, &lp, ((rows + 31) / 32) * heads * 128, 128);
    try ctx.finish();
    for (0..rows) |r_i| for (0..heads * dim) |c| {
        try close(expected_tile[r_i * heads * dim + c], af[out_off + (row0 + r_i) * heads * dim + c], 2e-2);
    };

    // 3. Fast striped BF16 sparse decode + inline bias + gated merge on row 1 of unit 1 (pos = 85).
    const dec_row: u32 = row0 + 39;
    const dec_pos: u32 = pos0 + 39; // 85
    const dec_cnt: u32 = 8;
    const dec_blks = [_]u32{ 0, 2, 5, 9, 13, 17, 19, 20 };
    bcnt[0] = dec_cnt;
    @memcpy(blist[0..dec_cnt], &dec_blks);
    const dec_tail0 = (dec_pos + 1) & ~@as(u32, 3);
    const dec_keys_total = dec_cnt * 4 + (dec_pos + 1 - dec_tail0);

    var lp_dec = lp;
    lp_dec.rows = 1;
    lp_dec.row0 = dec_row;
    lp_dec.pos0 = dec_pos;
    lp_dec.row_step = 1;
    lp_dec.taps = 4;
    var p_merge: gpu.Params = .{
        .rows = 1,
        .row0 = dec_row,
        .heads = heads,
        .y = out_off,
        .ys = heads * dim,
        .s = sink_off,
        .every = heads,
        .taps = 4,
    };

    try gpu.dispatch(ctx, "mk_relative_lsa_decode_keys_fast", &.{ arena, k16, v16, parts, block_list, block_count }, &lp_dec, @as(usize, kv_heads) * lp_dec.taps * 256, 256);
    try gpu.dispatch(ctx, "mk_relative_long_merge_fast", &.{ arena, parts }, &p_merge, heads * 128, 128);
    try ctx.finish();

    for (0..heads) |h| {
        const kvh = h / (heads / kv_heads);
        var max_s: f64 = @floatCast(af[sink_off + dec_row * heads + h]);
        var denom: f64 = 1.0;
        var acc: [dim]f64 = @splat(0);
        for (0..dec_keys_total) |idx| {
            const t: u32 = if (idx < dec_cnt * 4) (dec_blks[idx / 4] << 2) | @as(u32, @intCast(idx & 3)) else dec_tail0 + @as(u32, @intCast(idx - dec_cnt * 4));
            var dot: f64 = 0;
            for (0..dim) |d| dot += @as(f64, af[q_off + dec_row * heads * dim + h * dim + d]) * @as(f64, k32f[(t * kv_heads + kvh) * dim + d]);
            const dist = dec_pos - t;
            const rel: f64 = if (dist < m) @floatCast(af[bias_off + dec_row * heads * m + h * m + dist]) else 0;
            const score = dot * scale + rel;
            const next_m = @max(max_s, score);
            const prev = @exp(max_s - next_m);
            const w = @exp(score - next_m);
            denom = denom * prev + w;
            for (0..dim) |d| acc[d] = acc[d] * prev + w * @as(f64, v32f[(t * kv_heads + kvh) * dim + d]);
            max_s = next_m;
        }
        for (0..dim) |d| try close(acc[d] / denom, af[out_off + dec_row * heads * dim + h * dim + d], 2e-3);
    }

    // Inline bias + gated merge variant.
    lp_dec.bias_off = r_off;
    lp_dec.bias_stride = heads * rel_dim;
    p_merge.z = gate_off;
    p_merge.zs = heads * dim;
    p_merge.flags = gpu.flag_round_bf16 | gpu.flag_round_out;
    try gpu.dispatch(ctx, "mk_relative_lsa_decode_keys_inline_fast", &.{ arena, k16, v16, parts, block_list, block_count, rel_w_buf }, &lp_dec, @as(usize, kv_heads) * lp_dec.taps * 256, 256);
    try gpu.dispatch(ctx, "mk_relative_long_merge_gated", &.{ arena, parts }, &p_merge, heads * 128, 128);
    try ctx.finish();

    for (0..heads) |h| {
        const kvh = h / (heads / kv_heads);
        var max_s: f64 = @floatCast(af[sink_off + dec_row * heads + h]);
        var denom: f64 = 1.0;
        var acc: [dim]f64 = @splat(0);
        for (0..dec_keys_total) |idx| {
            const t: u32 = if (idx < dec_cnt * 4) (dec_blks[idx / 4] << 2) | @as(u32, @intCast(idx & 3)) else dec_tail0 + @as(u32, @intCast(idx - dec_cnt * 4));
            var dot: f64 = 0;
            for (0..dim) |d| dot += @as(f64, af[q_off + dec_row * heads * dim + h * dim + d]) * @as(f64, k32f[(t * kv_heads + kvh) * dim + d]);
            const dist = dec_pos - t;
            var rel: f64 = 0;
            if (dist < m) {
                var rsum: f32 = 0;
                for (0..rel_dim) |k| rsum += af[r_off + dec_row * heads * rel_dim + h * rel_dim + k] * fromBf16(rel_wu[k * m + dist]);
                rel = @floatCast(fromBf16(bf16(rsum)));
            }
            const score = dot * scale + rel;
            const next_m = @max(max_s, score);
            const prev = @exp(max_s - next_m);
            const w = @exp(score - next_m);
            denom = denom * prev + w;
            for (0..dim) |d| acc[d] = acc[d] * prev + w * @as(f64, v32f[(t * kv_heads + kvh) * dim + d]);
            max_s = next_m;
        }
        for (0..dim) |d| {
            const g_val = @as(f64, af[gate_off + dec_row * heads * dim + h * dim + d]);
            const sig = 1.0 / (1.0 + @exp(-g_val));
            const gated = (acc[d] / denom) * sig;
            const expected_bf16 = fromBf16(bf16(@floatCast(gated)));
            try close(expected_bf16, af[out_off + dec_row * heads * dim + h * dim + d], 2e-3);
        }
    }
}

test "select_halves splits interleaved group halves into contiguous outputs" {
    const ctx = try gpu.open();
    defer ctx.deinit();
    const groups: u32 = 4;
    const half: u32 = 5;
    const width: u32 = groups * 2 * half;
    const rows: u32 = 3;
    const row0: u32 = 1;
    const arena = try ctx.alloc((rows + row0) * width * 2);
    defer arena.deinit();
    const f = arena.floats();
    @memset(f, -777);
    for (row0..row0 + rows) |r| {
        for (0..groups) |g| {
            for (0..half) |d| {
                f[r * width + g * (2 * half) + d] = @floatFromInt(r * 1000 + g * 10 + d);
                f[r * width + g * (2 * half) + half + d] = @floatFromInt(-(@as(i64, @intCast(r * 1000 + g * 10 + d)) + 1));
            }
        }
    }
    const out_off: u32 = (rows + row0) * width;
    const p: gpu.Params = .{
        .rows = rows,
        .row0 = row0,
        .x = 0,
        .xs = width,
        .y = out_off,
        .ys = width,
        .n = width,
        .m = width / 2,
        .heads = groups,
        .dim = half,
    };
    try gpu.dispatch(ctx, "mk_select_halves", &.{arena}, &p, @as(usize, rows) * width, 256);
    try ctx.finish();
    for (0..width) |c| try testing.expectEqual(@as(f32, -777), f[out_off + c]);
    for (row0..row0 + rows) |r| {
        for (0..groups) |g| {
            for (0..half) |d| {
                const base = out_off + r * width;
                try testing.expectEqual(@as(f32, @floatFromInt(r * 1000 + g * 10 + d)), f[base + g * half + d]);
                try testing.expectEqual(@as(f32, @floatFromInt(-(@as(i64, @intCast(r * 1000 + g * 10 + d)) + 1))), f[base + groups * half + g * half + d]);
            }
        }
    }
}

test "gated_delta_net matches CPU reference and batched prefill equals 1-by-1 decode" {
    const ctx = try gpu.open();
    defer ctx.deinit();
    const hk: u32 = 2;
    const hv: u32 = 4;
    const dk: u32 = 128;
    const dv: u32 = 6;
    const rows: u32 = 5;
    const key_dim = hk * dk;
    const val_dim = hv * dv;
    const qkv_w = 2 * key_dim + val_dim;
    const row_w = qkv_w + hv + hv + val_dim;
    const eps: f32 = 1e-6;

    const a_log_buf = try ctx.alloc(hv);
    defer a_log_buf.deinit();
    const dt_bias_buf = try ctx.alloc(hv);
    defer dt_bias_buf.deinit();
    const arena = try ctx.alloc(rows * row_w);
    defer arena.deinit();
    const st_batch = try ctx.alloc(hv * dv * dk);
    defer st_batch.deinit();
    const st_seq = try ctx.alloc(hv * dv * dk);
    defer st_seq.deinit();

    const a_log_u16 = std.mem.bytesAsSlice(u16, std.mem.sliceAsBytes(a_log_buf.floats()))[0..hv];
    const dt_bias_u16 = std.mem.bytesAsSlice(u16, std.mem.sliceAsBytes(dt_bias_buf.floats()))[0..hv];
    var a_log_f64: [4]f64 = undefined;
    var dt_bias_f64: [4]f64 = undefined;
    for (0..hv) |h| {
        const av = bf16(@as(f32, @floatFromInt(h)) * 0.35 - 0.7);
        const dv_b = bf16(@as(f32, @floatFromInt(h)) * 0.25 - 0.3);
        a_log_u16[h] = av;
        dt_bias_u16[h] = dv_b;
        a_log_f64[h] = @floatCast(fromBf16(av));
        dt_bias_f64[h] = @floatCast(fromBf16(dv_b));
    }

    var r = rng(4242);
    const rand = r.random();
    const af = arena.floats();
    for (0..rows) |t| {
        const base = t * row_w;
        for (0..qkv_w) |i| af[base + i] = (rand.float(f32) - 0.5) * 0.8;
        for (0..hv) |h| {
            af[base + qkv_w + h] = (rand.float(f32) - 0.5) * 3.0;
            af[base + qkv_w + hv + h] = (rand.float(f32) - 0.5) * 3.0;
        }
        @memset(af[base + qkv_w + 2 * hv .. base + row_w], -999);
    }

    const qkv_off: u32 = 0;
    const a_off: u32 = qkv_w;
    const b_off: u32 = qkv_w + hv;
    const out_off: u32 = qkv_w + 2 * hv;
    const total_simds = @as(usize, hv) * dv;
    const total_threads = total_simds * 32;

    @memset(st_batch.floats(), 0);
    var pb: gpu.Params = .{
        .rows = rows,
        .row0 = 0,
        .x = qkv_off,
        .xs = row_w,
        .y = a_off,
        .ys = row_w,
        .z = b_off,
        .zs = row_w,
        .every = out_off,
        .pad_k = row_w,
        .kv_heads = hk,
        .heads = hv,
        .dim = dk,
        .m = dv,
        .dtype = 2,
        .sdtype = 2,
        .eps = eps,
    };
    try gpu.dispatch(ctx, "mk_gated_delta_net", &.{ a_log_buf, dt_bias_buf, arena, st_batch }, &pb, total_threads, @min(total_threads, 128));
    try ctx.finish();

    var batch_out: [rows * val_dim]f32 = undefined;
    for (0..rows) |t| @memcpy(batch_out[t * val_dim ..][0..val_dim], af[t * row_w + out_off ..][0..val_dim]);
    for (0..rows) |t| @memset(af[t * row_w + out_off ..][0..val_dim], -999);
    @memset(st_seq.floats(), 0);

    var seq_out: [rows * val_dim]f32 = undefined;
    for (0..rows) |t| {
        var ps = pb;
        ps.rows = 1;
        ps.row0 = @intCast(t);
        try gpu.dispatch(ctx, "mk_gated_delta_net", &.{ a_log_buf, dt_bias_buf, arena, st_seq }, &ps, total_threads, @min(total_threads, 128));
        try ctx.finish();
        @memcpy(seq_out[t * val_dim ..][0..val_dim], af[t * row_w + out_off ..][0..val_dim]);
    }

    try testing.expectEqualSlices(u32, std.mem.bytesAsSlice(u32, std.mem.sliceAsBytes(&batch_out)), std.mem.bytesAsSlice(u32, std.mem.sliceAsBytes(&seq_out)));
    try testing.expectEqualSlices(u32, st_batch.structs(u32), st_seq.structs(u32));

    var cpu_state = try testing.allocator.alloc(f64, hv * dv * dk);
    defer testing.allocator.free(cpu_state);
    @memset(cpu_state, 0);
    for (0..rows) |t| {
        const base = t * row_w;
        for (0..hv) |h| {
            const h_k = h / (hv / hk);
            var q2: f64 = 0;
            var k2: f64 = 0;
            for (0..dk) |i| {
                const qv: f64 = af[base + h_k * dk + i];
                const kv: f64 = af[base + key_dim + h_k * dk + i];
                q2 += qv * qv;
                k2 += kv * kv;
            }
            const q_inv = 1.0 / (@sqrt(q2 + eps) * @sqrt(@as(f64, dk)));
            const k_inv = 1.0 / @sqrt(k2 + eps);
            const sp_in: f64 = @as(f64, af[base + a_off + h]) + dt_bias_f64[h];
            const sp: f64 = if (sp_in > 20.0) sp_in else @log(1.0 + @exp(sp_in));
            const g: f64 = @exp(-@exp(a_log_f64[h]) * sp);
            const bv: f64 = af[base + b_off + h];
            const beta: f64 = 1.0 / (1.0 + @exp(-bv));
            for (0..dv) |d| {
                const st_row = cpu_state[(h * dv + d) * dk ..][0..dk];
                var kv_mem: f64 = 0;
                for (0..dk) |i| {
                    st_row[i] *= g;
                    kv_mem += st_row[i] * (@as(f64, af[base + key_dim + h_k * dk + i]) * k_inv);
                }
                const delta = (@as(f64, af[base + 2 * key_dim + h * dv + d]) - kv_mem) * beta;
                var out_v: f64 = 0;
                for (0..dk) |i| {
                    st_row[i] += (@as(f64, af[base + key_dim + h_k * dk + i]) * k_inv) * delta;
                    out_v += st_row[i] * (@as(f64, af[base + h_k * dk + i]) * q_inv);
                }
                try close(out_v, batch_out[t * val_dim + h * dv + d], 1e-5);
            }
        }
    }
    for (cpu_state, st_batch.floats()) |exp_s, act_s| try close(exp_s, act_s, 1e-5);
}

test "odd safetensors byte offsets match aligned offsets bit for bit" {
    const ctx = try gpu.open();
    defer ctx.deinit();
    var r = rng(777);
    const rand = r.random();
    const pad: usize = 3;

    // 1. mk_load (BF16 + F32) via mk_rmsnorm
    const nw: u32 = 32;
    const nw_f32_aligned = try ctx.alloc(nw);
    defer nw_f32_aligned.deinit();
    const nw_f32_odd = try ctx.alloc(nw + 2);
    defer nw_f32_odd.deinit();
    const nw_bf16_aligned = try ctx.alloc(nw);
    defer nw_bf16_aligned.deinit();
    const nw_bf16_odd = try ctx.alloc(nw + 2);
    defer nw_bf16_odd.deinit();
    for (nw_f32_aligned.floats()) |*v| v.* = 0.5 + rand.float(f32);
    @memcpy(std.mem.sliceAsBytes(nw_f32_odd.floats())[pad..][0 .. nw * 4], std.mem.sliceAsBytes(nw_f32_aligned.floats())[0 .. nw * 4]);
    const bf_a = std.mem.bytesAsSlice(u16, std.mem.sliceAsBytes(nw_bf16_aligned.floats()))[0..nw];
    for (bf_a) |*h| h.* = bf16(0.5 + rand.float(f32));
    @memcpy(std.mem.sliceAsBytes(nw_bf16_odd.floats())[pad..][0 .. nw * 2], std.mem.sliceAsBytes(nw_bf16_aligned.floats())[0 .. nw * 2]);

    const norm_arena = try ctx.alloc(3 * nw);
    defer norm_arena.deinit();
    for (norm_arena.floats()[0..nw]) |*v| v.* = rand.floatNorm(f32);
    for ([_]u32{ 0, 2 }) |dt| {
        const wa = if (dt == 0) nw_f32_aligned else nw_bf16_aligned;
        const wo = if (dt == 0) nw_f32_odd else nw_bf16_odd;
        var p0: gpu.Params = .{ .rows = 1, .x = 0, .xs = nw, .y = nw, .ys = nw, .n = nw, .heads = 1, .w = 0, .dtype = dt, .flags = gpu.flag_has_weight, .eps = 1e-5 };
        var p3 = p0;
        p3.y = 2 * nw;
        p3.w = pad;
        try gpu.dispatch(ctx, "mk_rmsnorm", &.{ wa, norm_arena }, &p0, 32, 32);
        try gpu.dispatch(ctx, "mk_rmsnorm", &.{ wo, norm_arena }, &p3, 32, 32);
        try ctx.finish();
        try testing.expectEqualSlices(u32, norm_arena.structs(u32)[nw .. 2 * nw], norm_arena.structs(u32)[2 * nw .. 3 * nw]);
    }

    // 2. mk_word + mk_load + mk_mlx_sb via mk_linear_q4_r2_t1, mk_mlx_qmm_t, mk_mlx2_qmm
    const rows: u32 = 8;
    const m: u32 = 4;
    const n: u32 = 64;
    const group: u32 = 32;
    var arena_alloc = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena_alloc.deinit();
    const q = try quantize(ctx, arena_alloc.allocator(), &r, m, n, group);
    defer {
        q.w.deinit();
        q.s.deinit();
        q.b.deinit();
    }
    const w_odd = try ctx.alloc(q.w.floats().len + 2);
    defer w_odd.deinit();
    const s_odd = try ctx.alloc(q.s.floats().len + 2);
    defer s_odd.deinit();
    const b_odd = try ctx.alloc(q.b.floats().len + 2);
    defer b_odd.deinit();
    const w_bytes = m * n / 2;
    const sb_bytes = m * (n / group) * 2;
    @memcpy(std.mem.sliceAsBytes(w_odd.floats())[pad..][0..w_bytes], std.mem.sliceAsBytes(q.w.floats())[0..w_bytes]);
    @memcpy(std.mem.sliceAsBytes(s_odd.floats())[pad..][0..sb_bytes], std.mem.sliceAsBytes(q.s.floats())[0..sb_bytes]);
    @memcpy(std.mem.sliceAsBytes(b_odd.floats())[pad..][0..sb_bytes], std.mem.sliceAsBytes(q.b.floats())[0..sb_bytes]);

    const x_buf = try ctx.alloc(rows * n + 2 * rows * m);
    defer x_buf.deinit();
    const partials = try ctx.alloc(16);
    defer partials.deinit();
    for (x_buf.floats()[0 .. rows * n]) |*v| v.* = rand.floatNorm(f32);

    const out0 = rows * n;
    const out1 = rows * n + rows * m;
    // q4 fast single-row kernel (mk_word + mk_load)
    var p_ref: gpu.Params = .{ .rows = 1, .x = 0, .xs = n, .y = out0, .ys = m, .n = n, .m = m, .bits = 4, .group = group, .sdtype = 2, .bdtype = 2 };
    var p_odd = p_ref;
    p_odd.w = pad;
    p_odd.s = pad;
    p_odd.b = pad;
    p_odd.y = out1;
    try gpu.dispatch(ctx, "mk_linear_q4_r2_t1", &.{ q.w, q.s, q.b, x_buf }, &p_ref, ((m + 1) / 2) * 32, 32);
    try gpu.dispatch(ctx, "mk_linear_q4_r2_t1", &.{ w_odd, s_odd, b_odd, x_buf }, &p_odd, ((m + 1) / 2) * 32, 32);
    try ctx.finish();
    try testing.expectEqualSlices(u32, x_buf.structs(u32)[out0 .. out0 + m], x_buf.structs(u32)[out1 .. out1 + m]);

    // mlx + mlx2 multi-row kernels (mk_mlx_sb + bytewise code load)
    p_ref.rows = rows;
    p_odd.rows = rows;
    try gpu.dispatch(ctx, "mk_mlx_qmm_t_bf16_g32", &.{ q.w, q.s, q.b, x_buf, partials }, &p_ref, 128, 128);
    try gpu.dispatch(ctx, "mk_mlx_qmm_t_bf16_g32", &.{ w_odd, s_odd, b_odd, x_buf, partials }, &p_odd, 128, 128);
    try ctx.finish();
    try testing.expectEqualSlices(u32, x_buf.structs(u32)[out0 .. out0 + rows * m], x_buf.structs(u32)[out1 .. out1 + rows * m]);

    try gpu.dispatch(ctx, "mk_mlx2_qmm_bf16_g32", &.{ q.w, q.s, q.b, x_buf }, &p_ref, 64, 64);
    try gpu.dispatch(ctx, "mk_mlx2_qmm_bf16_g32", &.{ w_odd, s_odd, b_odd, x_buf }, &p_odd, 64, 64);
    try ctx.finish();
    try testing.expectEqualSlices(u32, x_buf.structs(u32)[out0 .. out0 + rows * m], x_buf.structs(u32)[out1 .. out1 + rows * m]);
}

test "dflash2 kernels match f64 CPU reference" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const ctx = try gpu.open();
    defer ctx.deinit();
    var prng = rng(9001);
    const rand = prng.random();

    // 1. df_bf16_linear (rows=7, m=13 unaligned, k=64), df_bf16_linear_tile (rows=37, m=37, k=64),
    //    hardware MMA df_bf16_matrix8 / df_bf16_matrix32 (rows=7, 37; m=130 with w_rows=192, k=64),
    //    and coalesced df_bf16_linear_2xR_vec / 2xR_vec8 / splitk_rows (rows=7, 8; m=130; k=256).
    for ([_][4]u32{ .{ 7, 13, 13, 64 }, .{ 37, 37, 37, 64 }, .{ 7, 130, 192, 64 }, .{ 37, 130, 192, 64 } }) |shape| {
        const rows = shape[0];
        const m = shape[1];
        const w_rows = shape[2];
        const k = shape[3];
        const mma = (w_rows > m);
        const pad_bytes: u32 = if (!mma and rows <= 8) 2 else 0; // test unaligned BF16 weight offset
        const x_buf = try ctx.alloc(rows * k);
        defer x_buf.deinit();
        const w_buf = try ctx.alloc((w_rows * k + 1) / 2 + 2);
        defer w_buf.deinit();
        const y_buf = try ctx.alloc(rows * m + 64);
        defer y_buf.deinit();
        @memset(y_buf.floats(), -999.0);
        for (x_buf.floats()) |*v| v.* = rand.floatNorm(f32);
        const w_u16 = std.mem.bytesAsSlice(u16, std.mem.sliceAsBytes(w_buf.floats())[pad_bytes..]);
        const w_f32 = try a.alloc(f32, w_rows * k);
        for (w_f32, 0..) |*wf, idx| {
            const h = bf16(rand.floatNorm(f32) * 0.25);
            w_u16[idx] = h;
            wf.* = fromBf16(h);
        }
        var p: gpu.Params = .{ .w = pad_bytes, .dtype = 2, .rows = rows, .row0 = 0, .x = 0, .xs = k, .y = 0, .ys = m, .n = k, .m = m };
        if (mma) {
            const tiles_n = (@as(usize, m) + 63) / 64;
            if (rows <= 8) {
                try gpu.dispatch(ctx, "df_bf16_matrix8", &.{ x_buf, w_buf, y_buf }, &p, tiles_n * 128, 128);
            } else {
                const tiles_r = (@as(usize, rows) + 31) / 32;
                try gpu.dispatch(ctx, "df_bf16_matrix32", &.{ x_buf, w_buf, y_buf }, &p, tiles_n * tiles_r * 128, 128);
            }
        } else if (rows <= 8) {
            try gpu.dispatch(ctx, "df_bf16_linear", &.{ x_buf, w_buf, y_buf }, &p, ((@as(usize, m) + 15) / 16) * 128, 128);
        } else {
            const tm = (@as(usize, m) + 31) / 32;
            const tr = (@as(usize, rows) + 31) / 32;
            try gpu.dispatch(ctx, "df_bf16_linear_tile", &.{ x_buf, w_buf, y_buf }, &p, tm * tr * 128, 128);
        }
        try ctx.finish();
        const yf = y_buf.floats();
        const xf = x_buf.floats();
        const tol: f64 = if (mma) 2e-3 else 1e-4;
        for (0..rows) |r| {
            for (0..m) |j| {
                var ref_sum: f64 = 0;
                for (0..k) |kk| {
                    const xv: f64 = if (mma) fromBf16(bf16(xf[r * k + kk])) else xf[r * k + kk];
                    ref_sum += xv * w_f32[j * k + kk];
                }
                try close(ref_sum, yf[r * m + j], tol);
            }
        }
        if (mma) try testing.expectEqual(@as(f32, -999.0), yf[rows * m]);
    }

    // 2. df_grouped_dyn_conv (stage 0 prepare + stage 1 finish + residual)
    {
        const B: u32 = 5;
        const H: u32 = 16;
        const grp_sz: u32 = 4;
        const groups: u32 = H / grp_sz;
        const taps: u32 = 2;
        const dyn_w: u32 = 2 * taps * groups;
        const ab = try ctx.alloc(B * H * 4 + B * dyn_w);
        defer ab.deinit();
        const base_buf = try ctx.alloc((2 * taps * H + 1) / 2);
        defer base_buf.deinit();
        const af = ab.floats();
        const x0: u32 = 0;
        const res0: u32 = B * H;
        const y0: u32 = 2 * B * H;
        const dyn0: u32 = 4 * B * H;
        for (af[x0 .. x0 + B * H]) |*v| v.* = rand.floatNorm(f32);
        for (af[res0 .. res0 + B * H]) |*v| v.* = rand.floatNorm(f32);
        for (af[dyn0 .. dyn0 + B * dyn_w]) |*v| v.* = rand.floatNorm(f32) * 0.3;
        const bu16 = std.mem.bytesAsSlice(u16, std.mem.sliceAsBytes(base_buf.floats()));
        for (0..2 * taps * H) |i| bu16[i] = bf16(rand.floatNorm(f32) * 0.5);

        for ([_]u32{ 0, 1 }) |stage| {
            var p: gpu.Params = .{
                .rows = B,
                .row0 = 0,
                .kind = stage,
                .heads = groups,
                .dim = grp_sz,
                .taps = taps,
                .n = H,
                .x = x0,
                .xs = H,
                .y = y0,
                .ys = H,
                .z = dyn0,
                .zs = dyn_w,
                .pad_r = res0,
                .pad_k = H,
                .w = 0,
                .dtype = 2,
            };
            try gpu.dispatch(ctx, "df_grouped_dyn_conv", &.{ ab, base_buf }, &p, @as(usize, B) * H, 256);
            try ctx.finish();
            for (0..B) |r| {
                for (0..H) |c| {
                    var acc: f64 = 0;
                    const g = c / grp_sz;
                    for (0..taps) |off| {
                        if (r >= off) {
                            const xv: f64 = af[x0 + (r - off) * H + c];
                            const bw: f64 = fromBf16(bu16[(stage * taps + off) * H + c]);
                            const dw: f64 = af[dyn0 + r * dyn_w + (stage * taps + off) * groups + g];
                            acc += bw * xv + dw * xv;
                        }
                    }
                    if (stage == 1) acc += af[res0 + r * H + c];
                    try close(acc, af[y0 + r * H + c], 1e-5);
                }
            }
        }
    }

    // 3. df_attention (sliding window BF16 ring wrap-around + bidirectional noise block)
    {
        const B: u32 = 4;
        const ctx_pos: u32 = 10;
        const cap: u32 = 8;
        const window: u32 = 6;
        const Hq: u32 = 4;
        const Hkv: u32 = 2;
        const D: u32 = 64;
        const q_dim = Hq * D;
        const kv_dim = Hkv * D;
        const scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(D)));

        const ab = try ctx.alloc(B * (2 * q_dim + 2 * kv_dim));
        defer ab.deinit();
        const k_ring = try ctx.alloc((cap * kv_dim + 1) / 2);
        defer k_ring.deinit();
        const v_ring = try ctx.alloc((cap * kv_dim + 1) / 2);
        defer v_ring.deinit();

        const af = ab.floats();
        const q_off: u32 = 0;
        const k_off: u32 = B * q_dim;
        const v_off: u32 = k_off + B * kv_dim;
        const out_off: u32 = v_off + B * kv_dim;
        for (af[0..out_off]) |*v| v.* = rand.floatNorm(f32) * 0.4;
        const kr_u16 = std.mem.bytesAsSlice(u16, std.mem.sliceAsBytes(k_ring.floats()));
        const vr_u16 = std.mem.bytesAsSlice(u16, std.mem.sliceAsBytes(v_ring.floats()));
        for (0..cap * kv_dim) |i| {
            kr_u16[i] = bf16(rand.floatNorm(f32) * 0.4);
            vr_u16[i] = bf16(rand.floatNorm(f32) * 0.4);
        }

        var ap: gpu.Params = .{
            .rows = B,
            .pos = ctx_pos,
            .heads = Hq,
            .kv_heads = Hkv,
            .dim = D,
            .window = window,
            .capacity = cap,
            .scale = scale,
            .x = q_off,
            .xs = q_dim,
            .y = k_off,
            .ys = kv_dim,
            .z = v_off,
            .zs = kv_dim,
            .every = out_off,
            .pad_k = q_dim,
        };
        try gpu.dispatch(ctx, "df_attention", &.{ ab, k_ring, v_ring }, &ap, @as(usize, B) * Hq * 256, 256);
        try ctx.finish();

        for (0..B) |r| {
            const q_pos = ctx_pos + r;
            const cstart = if (q_pos + 1 > window) @min(ctx_pos, q_pos + 1 - window) else 0;
            const tot = (ctx_pos - cstart) + B;
            for (0..Hq) |h| {
                const kvh = h / (Hq / Hkv);
                var scores = try a.alloc(f64, tot);
                var max_s: f64 = -std.math.inf(f64);
                for (0..tot) |j| {
                    const k_pos = cstart + j;
                    var dot: f64 = 0;
                    for (0..D) |d| {
                        const kv: f64 = if (k_pos < ctx_pos)
                            fromBf16(kr_u16[(k_pos % cap) * kv_dim + kvh * D + d])
                        else
                            af[k_off + (k_pos - ctx_pos) * kv_dim + kvh * D + d];
                        dot += @as(f64, af[q_off + r * q_dim + h * D + d]) * kv;
                    }
                    scores[j] = dot * scale;
                    max_s = @max(max_s, scores[j]);
                }
                var denom: f64 = 0;
                for (scores) |*s| {
                    s.* = @exp(s.* - max_s);
                    denom += s.*;
                }
                for (0..D) |d| {
                    var acc: f64 = 0;
                    for (0..tot) |j| {
                        const k_pos = cstart + j;
                        const vv: f64 = if (k_pos < ctx_pos)
                            fromBf16(vr_u16[(k_pos % cap) * kv_dim + kvh * D + d])
                        else
                            af[v_off + (k_pos - ctx_pos) * kv_dim + kvh * D + d];
                        acc += (scores[j] / denom) * vv;
                    }
                    try close(acc, af[out_off + r * q_dim + h * D + d], 1e-4);
                }
            }
        }
    }

    // 4. df_top16_part + df_top16_merge + df_select_path (after_correction softcap=30.0)
    {
        const rows: u32 = 3;
        const vocab: u32 = 500;
        const rank: u32 = 64;
        const top_k: u32 = 16;
        const shards: u32 = 64;
        const anchor: u32 = 17;
        const softcap: f32 = 30.0;

        const logits = try ctx.alloc(rows * vocab);
        defer logits.deinit();
        const part_v = try ctx.alloc(rows * shards * top_k);
        defer part_v.deinit();
        const part_i = try ctx.alloc(rows * shards * top_k);
        defer part_i.deinit();
        const top_v = try ctx.alloc(rows * top_k);
        defer top_v.deinit();
        const top_i = try ctx.alloc(rows * top_k);
        defer top_i.deinit();
        const h_sel = try ctx.alloc(rows * rank);
        defer h_sel.deinit();
        const pred_cb = try ctx.alloc((vocab * rank + 1) / 2);
        defer pred_cb.deinit();
        const succ_cb = try ctx.alloc((vocab * rank + 1) / 2);
        defer succ_cb.deinit();
        const out_tok = try ctx.alloc(1 + rows);
        defer out_tok.deinit();

        const lf = logits.floats();
        for (lf) |*v| v.* = rand.floatNorm(f32) * 15.0;
        const hf = h_sel.floats();
        for (hf) |*v| v.* = rand.floatNorm(f32) * 0.2;
        const pu16 = std.mem.bytesAsSlice(u16, std.mem.sliceAsBytes(pred_cb.floats()));
        const su16 = std.mem.bytesAsSlice(u16, std.mem.sliceAsBytes(succ_cb.floats()));
        for (0..vocab * rank) |i| {
            pu16[i] = bf16(rand.floatNorm(f32) * 0.25);
            su16[i] = bf16(rand.floatNorm(f32) * 0.25);
        }

        var p1: gpu.Params = .{ .rows = rows, .x = 0, .xs = vocab, .m = vocab, .heads = shards, .n = top_k };
        var p2: gpu.Params = .{ .rows = rows, .heads = shards, .n = top_k };
        var p3: gpu.Params = .{ .rows = rows, .n = rank, .m = top_k, .pos = anchor, .x = 0, .xs = rank, .dtype = 2, .eps = softcap };
        try gpu.dispatch(ctx, "df_top16_part", &.{ logits, part_v, part_i }, &p1, rows * shards * 256, 256);
        try gpu.dispatch(ctx, "df_top16_merge", &.{ part_v, part_i, top_v, top_i }, &p2, rows * 256, 256);
        try gpu.dispatch(ctx, "df_select_path", &.{ h_sel, pred_cb, succ_cb, top_v, top_i, out_tok }, &p3, 512, 512);
        try ctx.finish();

        const ot = out_tok.structs(u32);
        try testing.expectEqual(anchor, ot[0]);
        var prev: u32 = anchor;
        for (0..rows) |pos| {
            var used = try a.alloc(bool, vocab);
            @memset(used, false);
            var best_s: f64 = -std.math.inf(f64);
            var best_c: u32 = 0;
            for (0..top_k) |_| {
                var bv: f32 = -std.math.inf(f32);
                var bi: u32 = 0;
                for (0..vocab) |vid| {
                    if (used[vid]) continue;
                    const val = lf[pos * vocab + vid];
                    if (val > bv) {
                        bv = val;
                        bi = @intCast(vid);
                    }
                }
                used[bi] = true;
                var dot: f64 = 0;
                for (0..rank) |d| {
                    dot += @as(f64, fromBf16(pu16[prev * rank + d])) * hf[pos * rank + d] * fromBf16(su16[bi * rank + d]);
                }
                const sc = @as(f64, softcap) * std.math.tanh((@as(f64, bv) + dot) / @as(f64, softcap));
                if (sc > best_s) {
                    best_s = sc;
                    best_c = bi;
                }
            }
            prev = best_c;
            try testing.expectEqual(prev, ot[1 + pos]);
        }
    }

    // 5. Coalesced df_bf16_linear_2xR_vec8, df_bf16_linear_2xR_vec, df_bf16_linear_splitk_rows
    //    (rows = 7, 8; m = 130 unaligned mod-4; k = 256; including round_in / round_out).
    for ([_]u32{ 7, 8 }) |rows| {
        const m: u32 = 130;
        const k: u32 = 256;
        const x_buf = try ctx.alloc(rows * k);
        defer x_buf.deinit();
        const w_buf = try ctx.alloc((m * k + 1) / 2 + 4);
        defer w_buf.deinit();
        const y_buf = try ctx.alloc(rows * m + 32);
        defer y_buf.deinit();
        for (x_buf.floats()) |*v| v.* = rand.floatNorm(f32);
        const w_u16 = std.mem.bytesAsSlice(u16, std.mem.sliceAsBytes(w_buf.floats()));
        const w_f32 = try a.alloc(f32, m * k);
        for (w_f32, 0..) |*wf, idx| {
            const h = bf16(rand.floatNorm(f32) * 0.25);
            w_u16[idx] = h;
            wf.* = fromBf16(h);
        }
        const names = [_][:0]const u8{ "df_bf16_linear_2xR_vec8", "df_bf16_linear_2xR_vec", "df_bf16_linear_splitk_rows", "df_bf16_mma_m8n16" };
        for (names, 0..) |kname, ki| {
            const flags: u32 = if (ki == 0) (gpu.flag_round_bf16 | gpu.flag_round_out) else gpu.flag_round_out;
            @memset(y_buf.floats(), -999.0);
            var p: gpu.Params = .{ .w = 0, .dtype = 2, .rows = rows, .row0 = 0, .x = 0, .xs = k, .y = 0, .ys = m, .n = k, .m = m, .flags = flags };
            if (ki == 2) {
                try gpu.dispatch(ctx, kname, &.{ x_buf, w_buf, y_buf }, &p, @as(usize, m) * 256, 256);
            } else if (ki == 3) {
                try gpu.dispatch(ctx, kname, &.{ x_buf, w_buf, y_buf }, &p, ((@as(usize, m) + 15) / 16) * 64, 64);
            } else {
                try gpu.dispatch(ctx, kname, &.{ x_buf, w_buf, y_buf }, &p, ((@as(usize, m) + 1) / 2) * 32, 128);
            }
            try ctx.finish();
            const yf = y_buf.floats();
            const xf = x_buf.floats();
            const use_bf16_x = (flags & gpu.flag_round_bf16) != 0 or ki == 3;
            for (0..rows) |r| {
                for (0..m) |j| {
                    var ref_sum: f64 = 0;
                    for (0..k) |kk| {
                        const xv: f64 = if (use_bf16_x) fromBf16(bf16(xf[r * k + kk])) else xf[r * k + kk];
                        ref_sum += xv * w_f32[j * k + kk];
                    }
                    const ref_val: f64 = fromBf16(bf16(@floatCast(ref_sum)));
                    try close(ref_val, yf[r * m + j], 2e-3);
                }
            }
            try testing.expectEqual(@as(f32, -999.0), yf[rows * m]);
        }
    }

    // 6. df_bf16_mma_m8n16_direct (splits = 1 and splits = 4 + df_bf16_splitk_reduce).
    for ([_]u32{ 7, 8 }) |rows| {
        const m: u32 = 128;
        const k: u32 = 256;
        const x_buf = try ctx.alloc(rows * k);
        defer x_buf.deinit();
        const w_buf = try ctx.alloc((m * k + 1) / 2 + 4);
        defer w_buf.deinit();
        const y_buf = try ctx.alloc(rows * m + 32);
        defer y_buf.deinit();
        const splitk_buf = try ctx.alloc(4 * rows * m);
        defer splitk_buf.deinit();
        for (x_buf.floats()) |*v| v.* = rand.floatNorm(f32);
        const w_u16 = std.mem.bytesAsSlice(u16, std.mem.sliceAsBytes(w_buf.floats()));
        const w_f32 = try a.alloc(f32, m * k);
        for (w_f32, 0..) |*wf, idx| {
            const h = bf16(rand.floatNorm(f32) * 0.25);
            w_u16[idx] = h;
            wf.* = fromBf16(h);
        }
        const n_tiles: usize = m / 16;
        for ([_]u32{ 1, 4 }) |splits| {
            @memset(y_buf.floats(), -999.0);
            var p: gpu.Params = .{ .w = 0, .dtype = 2, .rows = rows, .row0 = 0, .x = 0, .xs = k, .y = 0, .ys = m, .n = k, .m = m, .taps = splits, .flags = gpu.flag_round_out };
            if (splits == 1) {
                try gpu.dispatch(ctx, "df_bf16_mma_m8n16_direct", &.{ x_buf, w_buf, y_buf }, &p, n_tiles * 64, 64);
            } else {
                try gpu.dispatch(ctx, "df_bf16_mma_m8n16_direct", &.{ x_buf, w_buf, splitk_buf }, &p, @as(usize, splits) * n_tiles * 64, 64);
                try gpu.dispatch(ctx, "df_bf16_splitk_reduce", &.{ splitk_buf, y_buf }, &p, @as(usize, rows) * (m / 4), 256);
            }
            try ctx.finish();
            const yf = y_buf.floats();
            const xf = x_buf.floats();
            for (0..rows) |r| {
                for (0..m) |j| {
                    var ref_sum: f64 = 0;
                    for (0..k) |kk| {
                        const xv: f64 = fromBf16(bf16(xf[r * k + kk]));
                        ref_sum += xv * w_f32[j * k + kk];
                    }
                    const ref_val: f64 = fromBf16(bf16(@floatCast(ref_sum)));
                    try close(ref_val, yf[r * m + j], 2e-3);
                }
            }
            try testing.expectEqual(@as(f32, -999.0), yf[rows * m]);
        }
    }
}

test "inkling 1xR, 2xR, and splitk_rows bf16 kernels match T=1 kernels bit-for-bit" {
    const ctx = try gpu.open();
    defer ctx.deinit();
    var prng = rng(202604);
    const rand = prng.random();

    // 1. mk_bf16_linear_{2xR_vec8, 1xR_vec8} == mk_bf16_linear_4x1_vec8 (rows = 2, 7, 8, m = 130)
    //    and mk_bf16_linear_2xR_softcap_vec8 == mk_bf16_linear_4x1_softcap_vec8 (rows = 7, 8, m = 130 & 131).
    for ([_]u32{ 130, 131 }) |m| {
        const max_rows: u32 = 8;
        const n: u32 = 256;
        const w_buf = try ctx.alloc((m * n + 1) / 2 + 4);
        defer w_buf.deinit();
        const w_u16 = std.mem.bytesAsSlice(u16, std.mem.sliceAsBytes(w_buf.floats()));
        for (0..m * n) |i| w_u16[i] = bf16(rand.floatNorm(f32) * 0.25);

        const row0: u32 = 1;
        const xs: u32 = n + 4;
        const ys: u32 = m + 3;
        const total_r = row0 + max_rows;
        const ref_off: u32 = total_r * xs;
        const got_off: u32 = ref_off + total_r * ys;
        const one_off: u32 = got_off + total_r * ys;
        const arena = try ctx.alloc(one_off + total_r * ys);
        defer arena.deinit();
        const af = arena.floats();
        for (af[0..ref_off]) |*v| v.* = rand.floatNorm(f32);

        for ([_]u32{ 2, 7, 8 }) |rows| {
            if (m == 130) {
                @memset(af[ref_off..], -777.0);
                for (0..rows) |r| {
                    var p1: gpu.Params = .{ .w = 0, .dtype = 2, .rows = 1, .row0 = row0 + @as(u32, @intCast(r)), .x = 0, .xs = xs, .y = ref_off, .ys = ys, .n = n, .m = m, .flags = gpu.flag_round_out };
                    try gpu.dispatch(ctx, "mk_bf16_linear_4x1_vec8", &.{ w_buf, arena }, &p1, ((@as(usize, m) + 3) / 4) * 32, 128);
                }
                var pr: gpu.Params = .{ .w = 0, .dtype = 2, .rows = rows, .row0 = row0, .x = 0, .xs = xs, .y = got_off, .ys = ys, .n = n, .m = m, .flags = gpu.flag_round_out };
                try gpu.dispatch(ctx, "mk_bf16_linear_2xR_vec8", &.{ w_buf, arena }, &pr, ((@as(usize, m) + 1) / 2) * 32, 128);
                var p1x = pr;
                p1x.y = one_off;
                try gpu.dispatch(ctx, "mk_bf16_linear_1xR_vec8", &.{ w_buf, arena }, &p1x, @as(usize, m) * 32, 128);
                try ctx.finish();
                for (0..rows) |r| {
                    const rr = row0 + @as(u32, @intCast(r));
                    const u = arena.structs(u32);
                    try testing.expectEqualSlices(u32, u[ref_off + rr * ys ..][0..m], u[got_off + rr * ys ..][0..m]);
                    try testing.expectEqualSlices(u32, u[ref_off + rr * ys ..][0..m], u[one_off + rr * ys ..][0..m]);
                }
            }

            if (rows >= 7) {
                @memset(af[ref_off..], -777.0);
                for (0..rows) |r| {
                    var p1: gpu.Params = .{ .w = 0, .dtype = 2, .rows = 1, .row0 = row0 + @as(u32, @intCast(r)), .x = 0, .xs = xs, .y = ref_off, .ys = ys, .n = n, .m = m, .scale = 30.0 };
                    try gpu.dispatch(ctx, "mk_bf16_linear_4x1_softcap_vec8", &.{ w_buf, arena }, &p1, ((@as(usize, m) + 3) / 4) * 32, 128);
                }
                var ps: gpu.Params = .{ .w = 0, .dtype = 2, .rows = rows, .row0 = row0, .x = 0, .xs = xs, .y = got_off, .ys = ys, .n = n, .m = m, .scale = 30.0 };
                try gpu.dispatch(ctx, "mk_bf16_linear_2xR_softcap_vec8", &.{ w_buf, arena }, &ps, ((@as(usize, m) + 1) / 2) * 32, 128);
                try ctx.finish();
                for (0..rows) |r| {
                    const rr = row0 + @as(u32, @intCast(r));
                    const u = arena.structs(u32);
                    try testing.expectEqualSlices(u32, u[ref_off + rr * ys ..][0..m], u[got_off + rr * ys ..][0..m]);
                }
            }
        }
    }

    // 2. mk_multi_bf16_{2xR_vec8, 1xR_vec8} == mk_multi_bf16_4x1_vec8 for rows = 8 across 3 projections (128, 64, 32).
    {
        const rows: u32 = 8;
        const row0: u32 = 1;
        const n: u32 = 256;
        const ms = [3]u32{ 128, 64, 32 };
        const sum_m: u32 = ms[0] + ms[1] + ms[2];
        const w_buf = try ctx.alloc((sum_m * n + 1) / 2 + 16);
        defer w_buf.deinit();
        const w_u16 = std.mem.bytesAsSlice(u16, std.mem.sliceAsBytes(w_buf.floats()));
        for (w_u16) |*h| h.* = bf16(rand.floatNorm(f32) * 0.25);

        const total_r = row0 + rows;
        const ref_base: u32 = total_r * n;
        const got_base: u32 = ref_base + total_r * sum_m;
        const one_base: u32 = got_base + total_r * sum_m;
        const arena = try ctx.alloc(one_base + total_r * sum_m);
        defer arena.deinit();
        for (arena.floats()[0..ref_base]) |*v| v.* = rand.floatNorm(f32);

        var x1: gpu.ParamsX = .{ .p = .{ .rows = 1, .row0 = row0, .x = 0, .xs = n, .n = n }, .count = 3, .kind = 0b101 };
        var xr: gpu.ParamsX = .{ .p = .{ .rows = rows, .row0 = row0, .x = 0, .xs = n, .n = n }, .count = 3, .kind = 0b101 };
        var x1r = xr;
        var w_off_b: u64 = 0;
        var cur_ref: u32 = ref_base;
        var cur_got: u32 = got_base;
        var cur_one: u32 = one_base;
        var simds4: usize = 0;
        for (ms, 0..) |mk, idx| {
            x1.mat[idx] = .{ .w = w_off_b, .m = mk, .y = cur_ref, .ys = mk };
            xr.mat[idx] = .{ .w = w_off_b, .m = mk, .y = cur_got, .ys = mk };
            x1r.mat[idx] = .{ .w = w_off_b, .m = mk, .y = cur_one, .ys = mk };
            w_off_b += @as(u64, mk) * n * 2;
            cur_ref += total_r * mk;
            cur_got += total_r * mk;
            cur_one += total_r * mk;
            simds4 += mk / 4;
        }
        for (0..rows) |r| {
            x1.p.row0 = row0 + @as(u32, @intCast(r));
            try gpu.dispatch(ctx, "mk_multi_bf16_4x1_vec8", &.{ w_buf, arena }, &x1, simds4 * 32, 128);
        }
        try gpu.dispatch(ctx, "mk_multi_bf16_2xR_vec8", &.{ w_buf, arena }, &xr, (simds4 * 2) * 32, 128);
        try gpu.dispatch(ctx, "mk_multi_bf16_1xR_vec8", &.{ w_buf, arena }, &x1r, simds4 * 4 * 32, 128);
        try ctx.finish();
        const u = arena.structs(u32);
        for (0..3) |idx| {
            for (0..rows) |r| {
                const rr = row0 + @as(u32, @intCast(r));
                const r0 = x1.mat[idx].y + rr * ms[idx];
                const g0 = xr.mat[idx].y + rr * ms[idx];
                const o0 = x1r.mat[idx].y + rr * ms[idx];
                try testing.expectEqualSlices(u32, u[r0 .. r0 + ms[idx]], u[g0 .. g0 + ms[idx]]);
                try testing.expectEqualSlices(u32, u[r0 .. r0 + ms[idx]], u[o0 .. o0 + ms[idx]]);
            }
        }
    }

    // 3. 1-row-per-TG mk_bf16_linear_splitk_rows == mk_bf16_linear_splitk for rows = 7, 8, m = 64 and m = 24, n = 256.
    for ([_]u32{ 7, 8 }) |rows| {
        for ([_]u32{ 64, 24 }) |m| {
            const row0: u32 = 1;
            const n: u32 = 256;
            const xs: u32 = n + 4;
            const ys: u32 = m + 4;
            const w_buf = try ctx.alloc(m * n / 2 + 4);
            defer w_buf.deinit();
            const w_u16 = std.mem.bytesAsSlice(u16, std.mem.sliceAsBytes(w_buf.floats()));
            for (w_u16) |*h| h.* = bf16(rand.floatNorm(f32) * 0.25);

            const total_r = row0 + rows;
            const ref_off: u32 = total_r * xs;
            const got_off: u32 = ref_off + total_r * ys;
            const arena = try ctx.alloc(got_off + total_r * ys);
            defer arena.deinit();
            for (arena.floats()[0..ref_off]) |*v| v.* = rand.floatNorm(f32);

            const splitk_threads: usize = @import("inkling_plan.zig").splitk_threads;
            for (0..rows) |r| {
                var p1: gpu.Params = .{ .w = 0, .dtype = 2, .rows = 1, .row0 = row0 + @as(u32, @intCast(r)), .x = 0, .xs = xs, .y = ref_off, .ys = ys, .n = n, .m = m, .flags = gpu.flag_round_out };
                try gpu.dispatch(ctx, "mk_bf16_linear_splitk", &.{ w_buf, arena }, &p1, @as(usize, m) * splitk_threads, splitk_threads);
            }
            var pr: gpu.Params = .{ .w = 0, .dtype = 2, .rows = rows, .row0 = row0, .x = 0, .xs = xs, .y = got_off, .ys = ys, .n = n, .m = m, .flags = gpu.flag_round_out };
            try gpu.dispatch(ctx, "mk_bf16_linear_splitk_rows", &.{ w_buf, arena }, &pr, @as(usize, m) * splitk_threads, splitk_threads);
            try ctx.finish();
            const u = arena.structs(u32);
            for (0..rows) |r| {
                const rr = row0 + @as(u32, @intCast(r));
                try testing.expectEqualSlices(u32, u[ref_off + rr * ys ..][0..m], u[got_off + rr * ys ..][0..m]);
            }
        }
    }
}

test "grouped MoE and multi-row striped attention match T=1 kernels bit-for-bit" {
    const ctx = try gpu.open();
    defer ctx.deinit();
    var prng = rng(202611);
    const rand = prng.random();
    const inkling_plan = @import("inkling_plan.zig");

    // 1. Grouped NVFP4 + MXFP8 MoE (count=1, count=2, and count=5 fast paths; round_bf16 = 0 & 1) == 8 single-row T=1 calls bit-for-bit.
    {
        const rows: u32 = 8;
        const row0: u32 = 1;
        const total_r = row0 + rows;
        const experts: u32 = 16;
        const top_k: u32 = 8;
        const u_k: u32 = 256;
        const u_n: u32 = 128; // also d_k
        const d_n: u32 = 256;

        const w_up = try ctx.alloc(@as(usize, experts) * 2 * u_n * u_k / 4 + 16);
        defer w_up.deinit();
        const s_up = try ctx.alloc(@as(usize, experts) * 2 * u_n * (u_k / 16) / 4 + 16);
        defer s_up.deinit();
        const w_dn = try ctx.alloc(@as(usize, experts) * d_n * u_n / 4 + 16);
        defer w_dn.deinit();
        const s_dn = try ctx.alloc(@as(usize, experts) * d_n * (u_n / 16) / 4 + 16);
        defer s_dn.deinit();
        const amax_up = try ctx.alloc(experts);
        defer amax_up.deinit();
        const amax_dn = try ctx.alloc(experts);
        defer amax_dn.deinit();
        const denom = try ctx.alloc(1);
        defer denom.deinit();

        const lists = try ctx.alloc(2048);
        defer lists.deinit();
        const partial = try ctx.alloc(rows * top_k * d_n);
        defer partial.deinit();

        const x_off: u32 = 0;
        const r_off: u32 = total_r * u_k;
        const up_ref_off: u32 = r_off + total_r * (2 * top_k);
        const up_grp_off: u32 = up_ref_off + total_r * (top_k * u_n);
        const out_ref_off: u32 = up_grp_off + total_r * (top_k * u_n);
        const out_grp_off: u32 = out_ref_off + total_r * d_n;
        const arena = try ctx.alloc(out_grp_off + total_r * d_n);
        defer arena.deinit();
        const af = arena.floats();

        for (af[x_off..r_off]) |*v| v.* = @as(f32, @floatFromInt(rand.intRangeAtMost(i32, -8, 8))) * 0.125;
        for (amax_up.floats(), amax_dn.floats()) |*au, *ad| {
            au.* = 0.5 + rand.float(f32);
            ad.* = 0.5 + rand.float(f32);
        }
        denom.floats()[0] = 2.0;

        // Distinct top_k=8 routes per row giving count=1 (expert 0), count=2 (expert 1), count=5 (expert 2), count=3..7 (others).
        const route_table = [8][8]u32{
            .{ 0, 1, 2, 3, 4, 5, 6, 7 },
            .{ 1, 2, 3, 4, 5, 6, 7, 8 },
            .{ 2, 3, 4, 5, 6, 7, 8, 9 },
            .{ 2, 4, 5, 6, 7, 8, 9, 10 },
            .{ 2, 5, 6, 7, 8, 9, 10, 11 },
            .{ 6, 7, 8, 9, 10, 11, 12, 13 },
            .{ 7, 8, 9, 10, 11, 12, 13, 14 },
            .{ 8, 9, 10, 11, 12, 13, 14, 15 },
        };
        for (0..rows) |r| {
            const rb = r_off + (row0 + r) * (2 * top_k);
            var wsum: f32 = 0;
            for (0..top_k) |s| {
                af[rb + s] = @floatFromInt(route_table[r][s]);
                const wv = @as(f32, @floatFromInt(s + 1)) * 0.125;
                af[rb + top_k + s] = wv;
                wsum += wv;
            }
            for (0..top_k) |s| af[rb + top_k + s] /= wsum;
        }

        // Fill signed random weight bytes (avoiding FP8 e4m3 NaN 0x7f/0xff) and valid FP8 e4m3 scales cleanly.
        for ([_]metal.Buffer{ w_up, w_dn }) |wb| {
            const bytes = std.mem.sliceAsBytes(wb.floats());
            for (bytes) |*b| {
                var bv = rand.int(u8);
                if ((bv & 0x7f) == 0x7f) bv &= 0xfe;
                b.* = bv;
            }
        }
        for ([_]metal.Buffer{ s_up, s_dn }) |sb| {
            const bytes = std.mem.sliceAsBytes(sb.floats());
            for (bytes) |*b| b.* = 0x38 + (rand.int(u8) % 8);
        }

        for ([_]u32{ 0, 1 }) |round_bf16| {
            for ([_]u32{ 1, 0 }) |nvfp4| {
                @memset(af[up_ref_off..], 0);
                const up1_k: [:0]const u8 = if (nvfp4 != 0)
                    (if (round_bf16 != 0) "mk_expert_gate_up_nv16_lut_bf16" else "mk_expert_gate_up_nv16_lut")
                else
                    (if (round_bf16 != 0) "mk_expert_gate_up_mx_lut_bf16" else "mk_expert_gate_up_mx_lut");
                const dn1_k: [:0]const u8 = if (nvfp4 != 0)
                    (if (round_bf16 != 0) "mk_expert_down_nv8_lut_bf16" else "mk_expert_down_nv8_lut")
                else
                    (if (round_bf16 != 0) "mk_expert_down_mx_lut_bf16" else "mk_expert_down_mx_lut");
                const upg_k: [:0]const u8 = if (nvfp4 != 0)
                    (if (round_bf16 != 0) "mk_expert_grouped_up_nv16_lut_bf16" else "mk_expert_grouped_up_nv16_lut")
                else
                    (if (round_bf16 != 0) "mk_expert_grouped_up_mx_lut_bf16" else "mk_expert_grouped_up_mx_lut");
                const dng_k: [:0]const u8 = if (nvfp4 != 0)
                    (if (round_bf16 != 0) "mk_expert_grouped_down_nv8_lut_bf16" else "mk_expert_grouped_down_nv8_lut")
                else
                    (if (round_bf16 != 0) "mk_expert_grouped_down_mx_lut_bf16" else "mk_expert_grouped_down_mx_lut");

                for (0..rows) |r| {
                    const rr = row0 + @as(u32, @intCast(r));
                    var eu1: gpu.ExpertParams = .{ .rows = 1, .row0 = rr, .x = x_off, .xs = u_k, .routes = r_off, .rs = 2 * top_k, .y = up_ref_off, .ys = top_k * u_n, .k = u_k, .n = u_n, .experts = experts, .top_k = top_k, .nvfp4 = nvfp4, .round_bf16 = round_bf16, .gate_cap = 20.0, .up_cap = 20.0 };
                    const up_simds = @as(usize, top_k) * (if (nvfp4 != 0) (u_n / 4) else (u_n / 2));
                    try gpu.dispatch(ctx, up1_k, &.{ arena, w_up, s_up, amax_up, denom }, &eu1, up_simds * 32, if (nvfp4 != 0) 128 else 256);
                    var ed1: gpu.ExpertParams = .{ .rows = 1, .row0 = rr, .x = up_ref_off, .xs = top_k * u_n, .routes = r_off, .rs = 2 * top_k, .y = out_ref_off, .ys = d_n, .k = u_n, .n = d_n, .experts = experts, .top_k = top_k, .nvfp4 = nvfp4, .round_bf16 = round_bf16 };
                    if (nvfp4 != 0) {
                        try gpu.dispatch(ctx, dn1_k, &.{ arena, w_dn, s_dn, amax_dn, denom }, &ed1, @as(usize, d_n / 4) * 64, 64);
                    } else {
                        try gpu.dispatch(ctx, dn1_k, &.{ arena, w_dn, s_dn, amax_dn, denom }, &ed1, @as(usize, d_n / 4) * 256, 256);
                    }
                }

                var eug: gpu.ExpertParams = .{ .rows = rows, .row0 = row0, .x = x_off, .xs = u_k, .routes = r_off, .rs = 2 * top_k, .y = up_grp_off, .ys = top_k * u_n, .k = u_k, .n = u_n, .experts = experts, .top_k = top_k, .nvfp4 = nvfp4, .round_bf16 = round_bf16, .gate_cap = 20.0, .up_cap = 20.0 };
                try gpu.dispatch(ctx, "mk_expert_group_tiles", &.{ arena, lists }, &eug, 256, 256);
                const max_tiles: usize = inkling_plan.moeMaxTiles(experts, rows * top_k);
                const up_units = max_tiles * (if (nvfp4 != 0) (u_n / 4) else (u_n / 2));
                try gpu.dispatch(ctx, upg_k, &.{ arena, w_up, s_up, amax_up, denom, lists, partial }, &eug, up_units * 32, 128);

                var edg: gpu.ExpertParams = .{ .rows = rows, .row0 = row0, .x = up_grp_off, .xs = top_k * u_n, .routes = r_off, .rs = 2 * top_k, .y = out_grp_off, .ys = d_n, .k = u_n, .n = d_n, .experts = experts, .top_k = top_k, .nvfp4 = nvfp4, .round_bf16 = round_bf16 };
                const dn_threads = max_tiles * (d_n / 4) * @as(usize, if (nvfp4 != 0) 8 else 32);
                try gpu.dispatch(ctx, dng_k, &.{ arena, w_dn, s_dn, amax_dn, denom, lists, partial }, &edg, dn_threads, 128);
                try gpu.dispatch(ctx, "mk_expert_combine_sum4", &.{ arena, partial }, &edg, @as(usize, rows) * (d_n / 4), 256);
                try ctx.finish();

                const u = arena.structs(u32);
                for (0..rows) |r| {
                    const rr = row0 + @as(u32, @intCast(r));
                    try testing.expectEqualSlices(u32, u[out_ref_off + rr * d_n ..][0..d_n], u[out_grp_off + rr * d_n ..][0..d_n]);
                }
            }
        }
    }

    // 2. Multi-row mk_relative_long_keys_inline_fast + mk_relative_long_merge_gated across 64-key stripe boundary (pos 61..68) == T=1 bit-for-bit.
    {
        const rows: u32 = 8;
        const row0: u32 = 1;
        const total_r = row0 + rows;
        const heads: u32 = 3;
        const kv_heads: u32 = 1;
        const dim: u32 = 128;
        const rel_dim: u32 = 16;
        const rel_ext: u32 = 16;
        const pos0: u32 = 60; // row0=1 -> positions 61..68 (visible=62..69, crossing 64->65 1->2 stripes)
        const capacity: u32 = 80;
        const half_keys = try ctx.alloc((capacity + 16) * dim / 2);
        defer half_keys.deinit();
        const half_vals = try ctx.alloc((capacity + 16) * dim / 2);
        defer half_vals.deinit();
        const rel_w = try ctx.alloc((rel_dim * rel_ext + 1) / 2);
        defer rel_w.deinit();

        for ([_]metal.Buffer{ half_keys, half_vals }) |b| {
            const h16 = std.mem.bytesAsSlice(u16, std.mem.sliceAsBytes(b.floats()));
            for (h16) |*h| h.* = bf16(rand.floatNorm(f32) * 0.25);
        }
        const rw16 = std.mem.bytesAsSlice(u16, std.mem.sliceAsBytes(rel_w.floats()));
        for (rw16) |*h| h.* = bf16(rand.floatNorm(f32) * 0.25);

        const q_w = heads * dim;
        const r_w = heads * rel_dim;
        const q_off: u32 = 0;
        const r_off: u32 = total_r * q_w;
        const g_off: u32 = r_off + total_r * r_w;
        const s_off: u32 = g_off + total_r * q_w;
        const out_ref_off: u32 = s_off + total_r * heads;
        const out_grp_off: u32 = out_ref_off + total_r * q_w;
        const arena = try ctx.alloc(out_grp_off + total_r * q_w);
        defer arena.deinit();
        for (arena.floats()[0..out_ref_off]) |*v| v.* = rand.floatNorm(f32) * 0.25;

        const max_taps: u32 = 2;
        const parts = try ctx.alloc(rows * heads * max_taps * 130);
        defer parts.deinit();

        for (0..rows) |r| {
            const rr = row0 + @as(u32, @intCast(r));
            const vis = pos0 + rr + 1;
            const taps: u32 = (vis + 63) / 64;
            var p1: gpu.Params = .{
                .rows = 1,
                .row0 = rr,
                .pos = pos0,
                .x = q_off,
                .xs = q_w,
                .z = r_off,
                .zs = r_w,
                .s = s_off,
                .every = heads,
                .w = 0,
                .heads = heads,
                .kv_heads = kv_heads,
                .dim = dim,
                .m = rel_ext,
                .window = 0,
                .capacity = capacity,
                .scale = 1.0 / @sqrt(128.0),
                .taps = taps,
            };
            try gpu.dispatch(ctx, "mk_relative_long_keys_inline_fast", &.{ arena, half_keys, half_vals, parts, rel_w }, &p1, @as(usize, kv_heads) * taps * 256, 256);
            p1.y = out_ref_off;
            p1.ys = q_w;
            p1.z = g_off;
            p1.zs = q_w;
            p1.flags = gpu.flag_round_bf16 | gpu.flag_round_out;
            try gpu.dispatch(ctx, "mk_relative_long_merge_gated", &.{ arena, parts }, &p1, @as(usize, heads) * 128, 128);
        }

        var pg: gpu.Params = .{
            .rows = rows,
            .row0 = row0,
            .pos = pos0,
            .x = q_off,
            .xs = q_w,
            .z = r_off,
            .zs = r_w,
            .s = s_off,
            .every = heads,
            .w = 0,
            .heads = heads,
            .kv_heads = kv_heads,
            .dim = dim,
            .m = rel_ext,
            .window = 0,
            .capacity = capacity,
            .scale = 1.0 / @sqrt(128.0),
            .taps = max_taps,
        };
        try gpu.dispatch(ctx, "mk_relative_long_keys_inline_fast", &.{ arena, half_keys, half_vals, parts, rel_w }, &pg, @as(usize, rows) * kv_heads * max_taps * 256, 256);
        pg.y = out_grp_off;
        pg.ys = q_w;
        pg.z = g_off;
        pg.zs = q_w;
        pg.flags = gpu.flag_round_bf16 | gpu.flag_round_out;
        try gpu.dispatch(ctx, "mk_relative_long_merge_gated", &.{ arena, parts }, &pg, @as(usize, rows) * heads * 128, 128);
        try ctx.finish();

        const u = arena.structs(u32);
        for (0..rows) |r| {
            const rr = row0 + @as(u32, @intCast(r));
            try testing.expectEqualSlices(u32, u[out_ref_off + rr * q_w ..][0..q_w], u[out_grp_off + rr * q_w ..][0..q_w]);
        }
    }
}

test "Gemma 4: gelu_tanh activation and row_transform scale / weight_scale" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const ctx = try gpu.open();
    defer ctx.deinit();

    const inputs = [_]f32{ -100.0, -10.0, -3.5, -1.0, -0.25, 0.0, 0.25, 1.0, 3.5, 10.0, 100.0 };
    const n: u32 = inputs.len;
    const buf = try ctx.alloc(n * 4);
    defer buf.deinit();
    const f = buf.floats();
    @memcpy(f[0..n], &inputs);

    var pa: gpu.Params = .{ .rows = 1, .x = 0, .xs = n, .z = n, .zs = n, .n = n, .act_a = 4 };
    try gpu.dispatch(ctx, "mk_activate", &.{buf}, &pa, n, 32);

    @memcpy(f[2 * n .. 3 * n], &[_]f32{ 1.0, 2.0, -0.5, 0.25, 4.0, -2.0, 0.5, 1.5, -1.0, 0.1, 0.25 });
    var pb: gpu.Params = .{ .rows = 1, .x = 0, .xs = n, .y = 2 * n, .ys = n, .z = 3 * n, .zs = n, .n = n, .every = 1, .kind = 1, .act_a = 4 };
    try gpu.dispatch(ctx, "mk_binary", &.{buf}, &pb, n, 32);
    try ctx.finish();

    const k = @sqrt(2.0 / std.math.pi);
    for (inputs, 0..) |x, i| {
        const xf: f64 = x;
        const exp_g = 0.5 * xf * (1.0 + std.math.tanh(k * (xf + 0.044715 * xf * xf * xf)));
        try testing.expect(std.math.isFinite(f[n + i]));
        try testing.expect(std.math.isFinite(f[3 * n + i]));
        try close(exp_g, f[n + i], 1e-5);
        try close(exp_g * @as(f64, f[2 * n + i]), f[3 * n + i], 1e-5);
    }

    // row_transform: constant scale, f32 scalar at unaligned offset 1, bf16 scalar at odd offset 3.
    const rows: u32 = 3;
    const width: u32 = 8;
    const rbuf = try ctx.alloc(rows * width * 4);
    defer rbuf.deinit();
    const rf = rbuf.floats();
    var r = rng(701);
    for (rf[0 .. rows * width]) |*v| v.* = r.random().floatNorm(f32);

    const wbuf = try ctx.alloc(16);
    defer wbuf.deinit();
    const wb = std.mem.sliceAsBytes(wbuf.floats());
    @memset(wb, 0);
    const f32_scalar: f32 = 0.375;
    @memcpy(wb[1..5], std.mem.asBytes(&f32_scalar));
    const bf16_scalar_bits: u16 = bf16(-1.25);
    @memcpy(wb[7..9], std.mem.asBytes(&bf16_scalar_bits));

    var ps: gpu.Params = .{ .rows = rows, .x = 0, .xs = width, .y = rows * width, .ys = width, .n = width, .kind = 3, .scale = 61.9677335 };
    try gpu.dispatch(ctx, "mk_row_transform", &.{ rbuf, wbuf }, &ps, rows * width, 64);
    var pwf: gpu.Params = .{ .rows = rows, .x = 0, .xs = width, .y = 2 * rows * width, .ys = width, .n = width, .kind = 4, .w = 1, .dtype = 0 };
    try gpu.dispatch(ctx, "mk_row_transform", &.{ rbuf, wbuf }, &pwf, rows * width, 64);
    var pwb: gpu.Params = .{ .rows = rows, .x = 0, .xs = width, .y = 3 * rows * width, .ys = width, .n = width, .kind = 4, .w = 7, .dtype = 2 };
    try gpu.dispatch(ctx, "mk_row_transform", &.{ rbuf, wbuf }, &pwb, rows * width, 64);
    try ctx.finish();

    for (0..rows * width) |i| {
        const xv: f64 = rf[i];
        try close(xv * 61.9677335, rf[rows * width + i], 1e-6);
        try close(xv * @as(f64, f32_scalar), rf[2 * rows * width + i], 1e-6);
        try close(xv * @as(f64, fromBf16(bf16_scalar_bits)), rf[3 * rows * width + i], 1e-6);
    }
}

test "Gemma 4: mk_attention_d512 and mk_dec_scores_e16_sub{1,2,4} match CPU reference and each other" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const ctx = try gpu.open();
    defer ctx.deinit();
    var r = rng(702);

    const Case = struct { heads: u32, kv_heads: u32, dim: u32, window: u32, positions: u32 };
    for ([_]Case{
        .{ .heads = 16, .kv_heads = 1, .dim = 512, .window = 0, .positions = 18 },
        .{ .heads = 6, .kv_heads = 1, .dim = 512, .window = 7, .positions = 14 },
        .{ .heads = 3, .kv_heads = 1, .dim = 300, .window = 0, .positions = 11 },
    }) |tc| {
        const cap: u32 = if (tc.window == 0) tc.positions else tc.window + 3;
        const span: u32 = if (tc.window == 0) cap else tc.window;
        const K = try ctx.alloc(cap * tc.kv_heads * tc.dim);
        defer K.deinit();
        const V = try ctx.alloc(cap * tc.kv_heads * tc.dim);
        defer V.deinit();
        const S = try ctx.alloc(tc.heads * span);
        defer S.deinit();
        const qw = tc.heads * tc.dim;
        const buf = try ctx.alloc(3 * qw);
        defer buf.deinit();
        const f = buf.floats();

        const cpu_k = try a.alloc(f32, K.floats().len);
        const cpu_v = try a.alloc(f32, V.floats().len);
        const scores_cpu = try a.alloc(f64, tc.positions);

        for (0..tc.positions) |pos_u| {
            const pos: u32 = @intCast(pos_u);
            const slot = pos % cap;
            for (0..tc.kv_heads * tc.dim) |j| {
                const kv_v = r.random().floatNorm(f32) * 0.5;
                const vv_v = r.random().floatNorm(f32) * 0.5;
                K.floats()[slot * tc.kv_heads * tc.dim + j] = kv_v;
                V.floats()[slot * tc.kv_heads * tc.dim + j] = vv_v;
                cpu_k[slot * tc.kv_heads * tc.dim + j] = kv_v;
                cpu_v[slot * tc.kv_heads * tc.dim + j] = vv_v;
            }
            for (f[0..qw]) |*v| v.* = r.random().floatNorm(f32) * 0.5;

            var p: gpu.Params = .{
                .rows = 1,
                .pos = pos,
                .x = 0,
                .xs = qw,
                .y = qw,
                .ys = qw,
                .heads = tc.heads,
                .kv_heads = tc.kv_heads,
                .dim = tc.dim,
                .window = tc.window,
                .capacity = cap,
                .scale = 1.0,
                .every = span,
            };
            try gpu.dispatch(ctx, "mk_attention_d512", &.{ buf, K, V }, &p, tc.heads * 256, 256);

            const start: u32 = if (tc.window != 0 and pos + 1 > tc.window) pos + 1 - tc.window else 0;
            const chunks = (pos - start + 1 + 7) / 8;
            p.m = chunks;
            const E = (tc.dim + 31) / 32;
            if (E == 16) {
                const G = tc.heads / tc.kv_heads;
                const g_sub: u32 = if (G % 4 == 0) 4 else if (G % 2 == 0) 2 else 1;
                const n_sub = G / g_sub;
                const kname: [:0]const u8 = switch (g_sub) {
                    4 => "mk_dec_scores_e16_sub4",
                    2 => "mk_dec_scores_e16_sub2",
                    else => "mk_dec_scores_e16_sub1",
                };
                try gpu.dispatch(ctx, kname, &.{ buf, K, S }, &p, @as(usize, tc.kv_heads) * n_sub * chunks * 32, 256);
                p.y = 2 * qw;
                try gpu.dispatch(ctx, "mk_dec_chain_block", &.{ buf, V, S }, &p, @as(usize, tc.heads) * E * 256, 256);
            }
            try ctx.finish();

            if (E == 16) {
                const u = buf.structs(u32);
                try testing.expectEqualSlices(u32, u[qw .. 2 * qw], u[2 * qw .. 3 * qw]);
            }

            if (pos + 1 == tc.positions) {
                const G = tc.heads / tc.kv_heads;
                for (0..tc.heads) |h| {
                    const kvh = h / G;
                    var mscore: f64 = -std.math.inf(f64);
                    for (start..pos + 1) |t| {
                        const s = t % cap;
                        var dot: f64 = 0;
                        for (0..tc.dim) |d| dot += @as(f64, f[h * tc.dim + d]) * cpu_k[s * tc.kv_heads * tc.dim + kvh * tc.dim + d];
                        scores_cpu[t - start] = dot;
                        mscore = @max(mscore, dot);
                    }
                    var denom: f64 = 0;
                    for (start..pos + 1) |t| denom += @exp(scores_cpu[t - start] - mscore);
                    for (0..tc.dim) |d| {
                        var acc: f64 = 0;
                        for (start..pos + 1) |t| {
                            const w = @exp(scores_cpu[t - start] - mscore) / denom;
                            acc += w * @as(f64, cpu_v[(t % cap) * tc.kv_heads * tc.dim + kvh * tc.dim + d]);
                        }
                        try close(acc, f[qw + h * tc.dim + d], 1e-4);
                    }
                }
            }
        }
    }
}

test "Gemma 4: mk_mlx2_attention_d{256,512} match reference attention across sliding & causal prefill" {
    const ctx = try gpu.open();
    defer ctx.deinit();
    var r = rng(703);

    const Case = struct { kernel: [:0]const u8, ref_kernel: [:0]const u8, heads: u32, kv_heads: u32, dim: u32, window: u32 };
    for ([_]Case{
        .{ .kernel = "mk_mlx2_attention_d256", .ref_kernel = "mk_attention", .heads = 16, .kv_heads = 8, .dim = 256, .window = 12 },
        .{ .kernel = "mk_mlx2_attention_d256", .ref_kernel = "mk_attention", .heads = 16, .kv_heads = 8, .dim = 256, .window = 0 },
        .{ .kernel = "mk_mlx2_attention_d512", .ref_kernel = "mk_attention_d512", .heads = 16, .kv_heads = 1, .dim = 512, .window = 0 },
    }) |tc| {
        const pos0: u32 = 6;
        const rows: u32 = 40;
        const total = pos0 + rows;
        const cap: u32 = if (tc.window == 0) total else tc.window + rows;
        const kvw = tc.kv_heads * tc.dim;
        const qw = tc.heads * tc.dim;
        const K = try ctx.alloc(cap * kvw);
        defer K.deinit();
        const V = try ctx.alloc(cap * kvw);
        defer V.deinit();
        for (0..total) |t| {
            const slot = t % cap;
            for (0..kvw) |d| {
                K.floats()[slot * kvw + d] = r.random().floatNorm(f32) * 0.35;
                V.floats()[slot * kvw + d] = r.random().floatNorm(f32) * 0.35;
            }
        }
        const arena = try ctx.alloc(rows * qw * 3);
        defer arena.deinit();
        const f = arena.floats();
        for (f[0 .. rows * qw]) |*v| v.* = r.random().floatNorm(f32) * 0.35;

        const pref: gpu.Params = .{
            .rows = rows,
            .pos = pos0,
            .x = 0,
            .xs = qw,
            .y = rows * qw,
            .ys = qw,
            .heads = tc.heads,
            .kv_heads = tc.kv_heads,
            .dim = tc.dim,
            .window = tc.window,
            .capacity = cap,
            .scale = 1.0,
        };
        try gpu.dispatch(ctx, tc.ref_kernel, &.{ arena, K, V }, &pref, @as(usize, rows) * tc.heads * 256, 256);

        var pwide = pref;
        pwide.y = 2 * rows * qw;
        const nd: usize = if (tc.dim == 512) 2 else 1;
        const tgs = @as(usize, (rows + 31) / 32) * tc.heads * nd;
        try gpu.dispatch(ctx, tc.kernel, &.{ arena, K, V }, &pwide, tgs * 128, 128);
        try ctx.finish();

        var max_abs: f32 = 0;
        for (0..rows * qw) |i| {
            const diff = @abs(f[rows * qw + i] - f[2 * rows * qw + i]);
            if (diff > max_abs) max_abs = diff;
        }
        try testing.expect(max_abs < 1e-4);
    }
}

test "Gemma 4: static-K Q4 linear, multi, and gelu_tanh GLU kernels equal tiled reference bit for bit" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const ctx = try gpu.open();
    defer ctx.deinit();
    var r = rng(704);

    for ([_]struct { k: usize, name: [:0]const u8 }{
        .{ .k = 3840, .name = "mk_linear_q4_static_3840" },
        .{ .k = 4096, .name = "mk_linear_q4_static_4096" },
        .{ .k = 15360, .name = "mk_linear_q4_static_15360" },
    }) |tc| {
        const m: usize = 8;
        const q = try quantize(ctx, a, &r, m, tc.k, 64);
        defer {
            q.w.deinit();
            q.s.deinit();
            q.b.deinit();
        }
        const buf = try ctx.alloc(tc.k + 2 * m);
        defer buf.deinit();
        const f = buf.floats();
        for (f[0..tc.k]) |*v| v.* = r.random().floatNorm(f32);
        var p: gpu.Params = .{ .rows = 1, .x = 0, .xs = @intCast(tc.k), .y = @intCast(tc.k), .ys = @intCast(m), .n = @intCast(tc.k), .m = @intCast(m), .bits = 4, .group = 64, .sdtype = 2, .bdtype = 2 };
        try gpu.dispatch(ctx, "mk_linear_q4_r2_t1", &.{ q.w, q.s, q.b, buf }, &p, ((m + 1) / 2) * 32, 128);
        p.y = @intCast(tc.k + m);
        try gpu.dispatch(ctx, tc.name, &.{ q.w, q.s, q.b, buf }, &p, ((m + 1) / 2) * 32, 128);
        try ctx.finish();
        const u = buf.structs(u32);
        try testing.expectEqualSlices(u32, u[tc.k .. tc.k + m], u[tc.k + m .. tc.k + 2 * m]);
    }

    // Fused multi & gelu_tanh GLU static_3840.
    const m: usize = 8;
    const k: usize = 3840;
    const qs = [2]Quant{ try quantize(ctx, a, &r, m, k, 64), try quantize(ctx, a, &r, m, k, 64) };
    defer for (qs) |q| {
        q.w.deinit();
        q.s.deinit();
        q.b.deinit();
    };
    const wbytes = m * k / 2;
    const sbytes = (m * k / 64) * 2;
    const stride = wbytes + 2 * sbytes;
    const packed_buf = try ctx.alloc((2 * stride + 64) / 4);
    defer packed_buf.deinit();
    const bytes = std.mem.sliceAsBytes(packed_buf.floats());
    for (qs, 0..) |q, idx| {
        const off = 4 + idx * stride;
        @memcpy(bytes[off..][0..wbytes], std.mem.sliceAsBytes(q.w.floats())[0..wbytes]);
        @memcpy(bytes[off + wbytes ..][0..sbytes], std.mem.sliceAsBytes(q.s.floats())[0..sbytes]);
        @memcpy(bytes[off + wbytes + sbytes ..][0..sbytes], std.mem.sliceAsBytes(q.b.floats())[0..sbytes]);
    }
    const buf = try ctx.alloc(k + 6 * m);
    defer buf.deinit();
    const f = buf.floats();
    for (f[0..k]) |*v| v.* = r.random().floatNorm(f32);

    var xm: gpu.ParamsX = .{ .p = .{ .rows = 1, .xs = k, .n = k, .group = 64 }, .count = 2 };
    xm.mat[0] = .{ .w = 4, .s = 4 + wbytes, .b = 4 + wbytes + sbytes, .m = m, .y = k, .ys = m, .sdtype = 2, .bdtype = 2 };
    xm.mat[1] = .{ .w = 4 + stride, .s = 4 + stride + wbytes, .b = 4 + stride + wbytes + sbytes, .m = m, .y = k + m, .ys = m, .sdtype = 2, .bdtype = 2 };
    try gpu.dispatch(ctx, "mk_multi_q4_r2_t1", &.{ packed_buf, buf }, &xm, 2 * ((m + 1) / 2) * 32, 128);
    xm.mat[0].y = k + 2 * m;
    xm.mat[1].y = k + 3 * m;
    try gpu.dispatch(ctx, "mk_multi_q4_static_3840", &.{ packed_buf, buf }, &xm, 2 * ((m + 1) / 2) * 32, 128);

    var xg: gpu.ParamsX = .{ .p = .{ .rows = 1, .xs = k, .y = k + 4 * m, .ys = m, .n = k, .group = 64 }, .count = 2, .act_a = 4, .act_b = 0, .kind = 1 };
    xg.mat[0] = .{ .w = 4, .s = 4 + wbytes, .b = 4 + wbytes + sbytes, .m = m, .sdtype = 2, .bdtype = 2 };
    xg.mat[1] = .{ .w = 4 + stride, .s = 4 + stride + wbytes, .b = 4 + stride + wbytes + sbytes, .m = m, .sdtype = 2, .bdtype = 2 };
    try gpu.dispatch(ctx, "mk_glu_q4_r2_t1", &.{ packed_buf, buf }, &xg, ((m + 1) / 2) * 32, 128);
    xg.p.y = k + 5 * m;
    try gpu.dispatch(ctx, "mk_glu_q4_static_3840", &.{ packed_buf, buf }, &xg, ((m + 1) / 2) * 32, 128);
    try ctx.finish();

    const u = buf.structs(u32);
    try testing.expectEqualSlices(u32, u[k .. k + 2 * m], u[k + 2 * m .. k + 4 * m]);
    try testing.expectEqualSlices(u32, u[k + 4 * m .. k + 5 * m], u[k + 5 * m .. k + 6 * m]);
}
