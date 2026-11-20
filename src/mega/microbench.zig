//! decode search: `nano microbench`. GPU time of single-row 4-bit linear kernels at
//! LFM3's shapes on random weights placed 2 bytes off a word boundary (as in the
//! real checkpoint), each repeated in one command buffer with the bridge's
//! barriers between dispatches. Every variant's output is checked bit for bit
//! against mk_linear_q4_r1_t1. Timing guidance only; run it under the inference
//! lock (it uses the GPU) and confirm any gain with `nano bench`.
const std = @import("std");
const gpu = @import("gpu.zig");
const io = @import("io.zig");
const metal = @import("weights").metal;

const Variant = struct { kernel: [:0]const u8, rows: u32, group: u32 };

fn bf16(v: f32) u16 {
    const bits: u32 = @bitCast(v);
    return @intCast((bits + 0x7fff + ((bits >> 16) & 1)) >> 16);
}

pub fn run(a: std.mem.Allocator, repeats: u32) !void {
    const ctx = try gpu.open();
    defer ctx.deinit();
    var prng = std.Random.DefaultPrng.init(42);
    const r = prng.random();
    const shapes = [_][2]u32{ .{ 2048, 2048 }, .{ 6144, 2048 }, .{ 512, 2048 }, .{ 7680, 2048 }, .{ 2048, 7680 }, .{ 128000, 2048 } };
    var variants: std.ArrayList(Variant) = .empty;
    for ([_]u32{ 1, 2, 4, 8 }) |R| for ([_]u32{ 64, 128, 256 }) |g| {
        try variants.append(a, .{ .kernel = try std.fmt.allocPrintSentinel(a, "mk_linear_q4_r{d}_t1", .{R}, 0), .rows = R, .group = g });
    };
    for (shapes) |shape| {
        const m = shape[0];
        const n = shape[1];
        const group = 64;
        // One buffer: 2 pad bytes, codes, scales, biases (bf16), like an MLX safetensors blob at offset 2 mod 4.
        const code_bytes: usize = @as(usize, m) * n / 2;
        const sb_bytes: usize = @as(usize, m) * (n / group) * 2;
        const wbuf = try ctx.alloc((2 + code_bytes + 2 * sb_bytes + 3) / 4);
        defer wbuf.deinit();
        const bytes = std.mem.sliceAsBytes(wbuf.floats());
        r.bytes(bytes[2 .. 2 + code_bytes]);
        const scales = bytes[2 + code_bytes ..][0..sb_bytes];
        const biases = bytes[2 + code_bytes + sb_bytes ..][0..sb_bytes];
        var i: usize = 0;
        while (i < sb_bytes) : (i += 2) {
            std.mem.writeInt(u16, scales[i..][0..2], bf16(0.01 + 0.05 * r.float(f32)), .little);
            std.mem.writeInt(u16, biases[i..][0..2], bf16(-0.3 + 0.1 * r.float(f32)), .little);
        }
        const arena = try ctx.alloc(n + 2 * m);
        defer arena.deinit();
        const f = arena.floats();
        for (f[0..n]) |*v| v.* = r.floatNorm(f32);
        var p: gpu.Params = .{ .rows = 1, .w = 2, .s = 2 + code_bytes, .b = 2 + code_bytes + sb_bytes, .x = 0, .xs = n, .y = n, .ys = m, .n = n, .m = m, .bits = 4, .group = group, .sdtype = 2, .bdtype = 2 };
        try gpu.dispatch(ctx, "mk_linear_q4_r1_t1", &.{ wbuf, wbuf, wbuf, arena }, &p, @as(usize, m) * 32, 128);
        try ctx.finish();
        const reference = try a.dupe(f32, f[n..][0..m]);
        const mb = @as(f64, @floatFromInt(code_bytes + 2 * sb_bytes)) / 1e6;
        try io.print(a, "shape {d}x{d} ({d:.2} MB)\n", .{ m, n, mb });
        for (variants.items) |v| {
            p.y = n + m;
            @memset(f[n + m ..][0..m], 0);
            const threads = (@as(usize, m) + v.rows - 1) / v.rows * 32;
            const before = ctx.stats();
            for (0..repeats) |_| try gpu.dispatch(ctx, v.kernel, &.{ wbuf, wbuf, wbuf, arena }, &p, threads, v.group);
            try ctx.finish();
            const after = ctx.stats();
            const us = (after.gpu_seconds - before.gpu_seconds) / @as(f64, @floatFromInt(repeats)) * 1e6;
            const same = std.mem.eql(u32, std.mem.bytesAsSlice(u32, std.mem.sliceAsBytes(reference)), std.mem.bytesAsSlice(u32, std.mem.sliceAsBytes(f[n + m ..][0..m])));
            try io.print(a, "  {s:<24} tg {d:>4}: {d:>8.2} us  {d:>6.0} GB/s  {s}\n", .{ v.kernel, v.group, us, mb * 1e-3 / (us * 1e-6), if (same) "bit-identical" else "DIFFERS" });
        }
    }
}
