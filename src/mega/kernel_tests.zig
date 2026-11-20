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
    try ctx.finish();
    for (0..8) |i| {
        const x: f64 = @as(f64, @floatFromInt(i)) - 3.5;
        const g: f64 = @as(f64, @floatFromInt(i / 4)) * 2 - 1;
        try close(x / (1 + @exp(-g)), f[32 + i], 1e-6);
        try close(x / (1 + @exp(-x)) + x, f[48 + i], 1e-6);
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
                    0 => { s = 0; b = 0x8000; }, // +0 scale, -0 bias
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
