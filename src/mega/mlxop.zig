//! prefill-mlx: `nano mlx-op linear`, one MLX-arithmetic operation on given inputs
//! (no checkpoint), through the same encoder a forward uses (exec.mlxLinear), for
//! tools/nano_mlx_ops.py to compare bit for bit with stock MLX.
//!
//!   nano mlx-op linear --m M --k K --n N --group G --weights W.bin --x X.f32 --out Y.f32
//!       [--variant mlx|exact] [--repeats R] [--sdtype 0|1|2]
//!
//! W.bin: 2 pad bytes (so the data starts 2 bytes off a word boundary, like the
//! LFM3 checkpoint's), the packed codes [N, K/8] u32, then scales and biases
//! [N, K/G] of dtype --sdtype (2 bfloat16, the default; 1 float16; 0 float32,
//! after 2 more pad bytes so they are word-aligned). X.f32: [M, K] float32.
//! Y.f32 receives [M, N] float32.
//!   nano mlx-op attention --t T --kl KL --heads H --kv-heads KV --window W
//!       --capacity C --q Q.f32 --k K.f32 --v V.f32 --out O.f32 [--gate G.f32] [--repeats R]
//!
//! Attention: queries [T, H * 128] are the last T of KL positions; keys and
//! values [KL, KV * 128] are written in position order into caches of C slots
//! (slot = position % C, so C < KL exercises a ring). O.f32 receives the
//! ungated output [T, H * 128]; with --gate G.f32 ([T, H]) the gated kernel is
//! also run and checked bit for bit against mk_binary(output, sigmoid(gate)).
//! Prints one JSON line: the kernels run and MLX's split; with --repeats R the
//! operation is encoded R more times in one command buffer and the mean GPU
//! time per operation is reported (timing guidance; run under the lock).
const std = @import("std");
const gpu = @import("gpu.zig");
const io = @import("io.zig");
const exec = @import("exec.zig");
const loader = @import("loader.zig");

pub fn run(a: std.mem.Allocator, argv: []const []const u8) !void {
    if (argv.len >= 1 and std.mem.eql(u8, argv[0], "attention")) return attention(a, argv);
    if (argv.len < 1 or !std.mem.eql(u8, argv[0], "linear")) return error.UnknownMlxOp;
    var m: u32 = 0;
    var k: u32 = 0;
    var n: u32 = 0;
    var group: u32 = 64;
    var wpath: ?[]const u8 = null;
    var xpath: ?[]const u8 = null;
    var ypath: ?[]const u8 = null;
    var variant: []const u8 = "mlx";
    var repeats: u32 = 0;
    var sdtype: u32 = 2;
    var i: usize = 1;
    while (i + 1 < argv.len) : (i += 2) {
        const flag = argv[i];
        const value = argv[i + 1];
        if (std.mem.eql(u8, flag, "--m")) m = try std.fmt.parseInt(u32, value, 10) else if (std.mem.eql(u8, flag, "--k")) k = try std.fmt.parseInt(u32, value, 10) else if (std.mem.eql(u8, flag, "--n")) n = try std.fmt.parseInt(u32, value, 10) else if (std.mem.eql(u8, flag, "--group")) group = try std.fmt.parseInt(u32, value, 10) else if (std.mem.eql(u8, flag, "--weights")) wpath = value else if (std.mem.eql(u8, flag, "--x")) xpath = value else if (std.mem.eql(u8, flag, "--out")) ypath = value else if (std.mem.eql(u8, flag, "--variant")) variant = value else if (std.mem.eql(u8, flag, "--repeats")) repeats = try std.fmt.parseInt(u32, value, 10) else if (std.mem.eql(u8, flag, "--sdtype")) sdtype = try std.fmt.parseInt(u32, value, 10) else return error.UnknownFlag;
    }
    if (i != argv.len) return error.MissingValue;
    if (sdtype > 2) return error.InvalidDtype;
    if (m == 0 or k == 0 or n == 0 or group == 0 or k % group != 0 or k % 8 != 0) return error.InvalidShape;
    const wbytes = try loader.readFile(a, wpath orelse return error.MissingWeights);
    const xbytes = try loader.readFile(a, xpath orelse return error.MissingInput);
    const code_bytes = @as(usize, n) * k / 2;
    const sb_bytes = @as(usize, n) * (k / group) * @as(usize, if (sdtype == 0) 4 else 2);
    const s_off = 2 + code_bytes + @as(usize, if (sdtype == 0) 2 else 0);
    if (wbytes.len != s_off + 2 * sb_bytes) return error.InvalidWeightsFile;
    if (xbytes.len != @as(usize, m) * k * 4) return error.InvalidInputFile;

    const ctx = try gpu.open();
    defer ctx.deinit();
    const wbuf = try ctx.alloc((wbytes.len + 3) / 4);
    defer wbuf.deinit();
    @memcpy(std.mem.sliceAsBytes(wbuf.floats())[0..wbytes.len], wbytes);
    const arena = try ctx.alloc(@as(usize, m) * k + @as(usize, m) * n);
    defer arena.deinit();
    @memcpy(std.mem.sliceAsBytes(arena.floats())[0..xbytes.len], xbytes);
    const splitk = try ctx.alloc(exec.mlx_splitk_words);
    defer splitk.deinit();
    const p: gpu.Params = .{
        .rows = m, .x = 0, .xs = k, .y = m * k, .ys = n, .n = k, .m = n,
        .w = 2, .s = s_off, .b = s_off + sb_bytes, .dtype = 0, .sdtype = sdtype, .bdtype = sdtype, .bits = 4, .group = group,
    };
    var kernels: []const u8 = undefined;
    var split: u32 = 0;
    var us: f64 = 0;
    for (0..repeats + 1) |rep_i| {
        if (rep_i == 1) {
            try ctx.finish();
            const before = ctx.stats();
            for (0..repeats) |_| _ = try encodeOnce(ctx, variant, wbuf, arena, splitk, p, m, k, n);
            try ctx.finish();
            us = (ctx.stats().gpu_seconds - before.gpu_seconds) / @as(f64, @floatFromInt(repeats)) * 1e6;
            break;
        }
        const r = try encodeOnce(ctx, variant, wbuf, arena, splitk, p, m, k, n);
        kernels = r.kernels;
        split = r.split;
    }
    try ctx.finish();
    try io.writeFile(a, ypath orelse return error.MissingOutput, std.mem.sliceAsBytes(arena.floats()[m * k ..][0 .. @as(usize, m) * n]));
    try io.print(a, "{{\"kernels\":\"{s}\",\"split_k\":{d},\"m\":{d},\"k\":{d},\"n\":{d},\"repeats\":{d},\"us_per_op\":{d:.2}}}\n", .{ kernels, split, m, k, n, repeats, us });
}

fn encodeOnce(ctx: @import("weights").metal.Context, variant: []const u8, wbuf: @import("weights").metal.Buffer, arena: @import("weights").metal.Buffer, splitk: @import("weights").metal.Buffer, p0: gpu.Params, m: u32, k: u32, n: u32) !struct { kernels: []const u8, split: u32 } {
    var p = p0;
    var kernels: []const u8 = undefined;
    var split: u32 = 0;
    const group = p.group;
    if (std.mem.eql(u8, variant, "mlx")) {
        if (m < exec.mlxVectorLimit(k, n)) return error.BelowMlxVectorLimit;
        if (!(group == 32 or group == 64 or group == 128) or k % 32 != 0) return error.NotPorted;
        split = try exec.mlxLinear(ctx, .{ wbuf, wbuf, wbuf }, arena, splitk, &p);
        kernels = if (split > 1) "mk_mlx_qmm_t_splitk_<dtype>_g*+mk_mlx_splitk_accum" else "mk_mlx_qmm_t_<dtype>_g*";
    } else if (std.mem.eql(u8, variant, "exact")) {
        const tiles = (@as(usize, n) + 1) / 2 * ((@as(usize, m) + 3) / 4);
        try gpu.dispatch(ctx, "mk_linear_q4_r2_t4", &.{ wbuf, wbuf, wbuf, arena }, &p, tiles * 32, 128);
        kernels = "mk_linear_q4_r2_t4";
    } else return error.UnknownVariant;
    return .{ .kernels = kernels, .split = split };
}

fn attention(a: std.mem.Allocator, argv: []const []const u8) !void {
    var t: u32 = 0;
    var kl: u32 = 0;
    var heads: u32 = 16;
    var kv_heads: u32 = 4;
    var window: u32 = 0;
    var capacity: u32 = 0;
    var repeats: u32 = 0;
    var paths: [5]?[]const u8 = @splat(null); // q, k, v, out, gate
    var i: usize = 1;
    while (i + 1 < argv.len) : (i += 2) {
        const flag = argv[i];
        const value = argv[i + 1];
        const names = [_][]const u8{ "--q", "--k", "--v", "--out", "--gate" };
        var matched = false;
        for (names, 0..) |name, j| if (std.mem.eql(u8, flag, name)) {
            paths[j] = value;
            matched = true;
        };
        if (matched) continue;
        const v = try std.fmt.parseInt(u32, value, 10);
        if (std.mem.eql(u8, flag, "--t")) t = v else if (std.mem.eql(u8, flag, "--kl")) kl = v else if (std.mem.eql(u8, flag, "--heads")) heads = v else if (std.mem.eql(u8, flag, "--kv-heads")) kv_heads = v else if (std.mem.eql(u8, flag, "--window")) window = v else if (std.mem.eql(u8, flag, "--capacity")) capacity = v else if (std.mem.eql(u8, flag, "--repeats")) repeats = v else return error.UnknownFlag;
    }
    if (i != argv.len) return error.MissingValue;
    if (capacity == 0) capacity = kl;
    if (t == 0 or kl < t or !exec.mlxAttentionApplies(t, heads, kv_heads, 128)) return error.NotPorted;
    const width = heads * 128;
    const kv_width = kv_heads * 128;
    const qbytes = try loader.readFile(a, paths[0] orelse return error.MissingInput);
    const kbytes = try loader.readFile(a, paths[1] orelse return error.MissingInput);
    const vbytes = try loader.readFile(a, paths[2] orelse return error.MissingInput);
    if (qbytes.len != @as(usize, t) * width * 4 or kbytes.len != @as(usize, kl) * kv_width * 4 or vbytes.len != kbytes.len) return error.InvalidInputFile;
    const ctx = try gpu.open();
    defer ctx.deinit();
    // Arena: q, out, gate, gated out, mk_binary reference.
    const tw = @as(usize, t) * width;
    const arena = try ctx.alloc(4 * tw + @as(usize, t) * heads);
    defer arena.deinit();
    const f = arena.floats();
    @memcpy(std.mem.sliceAsBytes(f[0..tw]), qbytes);
    const gate_off = 2 * tw;
    const gated_off = gate_off + @as(usize, t) * heads;
    const ref_off = gated_off + tw;
    if (paths[4]) |g| {
        const gbytes = try loader.readFile(a, g);
        if (gbytes.len != @as(usize, t) * heads * 4) return error.InvalidInputFile;
        @memcpy(std.mem.sliceAsBytes(f[gate_off..][0 .. @as(usize, t) * heads]), gbytes);
    }
    const kc = try ctx.alloc(@as(usize, capacity) * kv_width);
    defer kc.deinit();
    const vc = try ctx.alloc(@as(usize, capacity) * kv_width);
    defer vc.deinit();
    const ks = std.mem.bytesAsSlice(f32, kbytes);
    const vs = std.mem.bytesAsSlice(f32, vbytes);
    for (0..kl) |pos| {
        const slot = pos % capacity;
        for (0..kv_width) |d| {
            kc.floats()[slot * kv_width + d] = ks[pos * kv_width + d];
            vc.floats()[slot * kv_width + d] = vs[pos * kv_width + d];
        }
    }
    var p: gpu.Params = .{ .rows = t, .row0 = 0, .pos = kl - t, .x = 0, .xs = width, .y = @intCast(tw), .ys = width, .heads = heads, .kv_heads = kv_heads, .dim = 128, .window = window, .capacity = capacity, .scale = @floatCast(1.0 / @sqrt(@as(f64, 128))) };
    try exec.mlxAttention(ctx, arena, kc, vc, &p, false);
    var us: f64 = 0;
    if (repeats > 0) {
        try ctx.finish();
        const before = ctx.stats();
        for (0..repeats) |_| try exec.mlxAttention(ctx, arena, kc, vc, &p, false);
        try ctx.finish();
        us = (ctx.stats().gpu_seconds - before.gpu_seconds) / @as(f64, @floatFromInt(repeats)) * 1e6;
    }
    var gate_ok = true;
    if (paths[4] != null) {
        var g = p;
        g.y = @intCast(gated_off);
        g.z = @intCast(gate_off);
        g.zs = heads;
        g.act_b = 2; // sigmoid
        try exec.mlxAttention(ctx, arena, kc, vc, &g, true);
        var b: gpu.Params = .{ .rows = t, .x = @intCast(tw), .xs = width, .y = @intCast(gate_off), .ys = heads, .z = @intCast(ref_off), .zs = width, .n = width, .every = 128, .kind = 1, .act_a = 0, .act_b = 2 };
        try gpu.dispatch(ctx, "mk_binary", &.{arena}, &b, tw, 256);
        try ctx.finish();
        gate_ok = std.mem.eql(u32, std.mem.bytesAsSlice(u32, std.mem.sliceAsBytes(f[gated_off..][0..tw])), std.mem.bytesAsSlice(u32, std.mem.sliceAsBytes(f[ref_off..][0..tw])));
    }
    try ctx.finish();
    try io.writeFile(a, paths[3] orelse return error.MissingOutput, std.mem.sliceAsBytes(f[tw..][0..tw]));
    try io.print(a, "{{\"kernels\":\"mk_mlx_attention\",\"t\":{d},\"kl\":{d},\"window\":{d},\"capacity\":{d},\"gate_epilogue_bit_identical\":{},\"repeats\":{d},\"us_per_op\":{d:.2}}}\n", .{ t, kl, window, capacity, gate_ok, repeats, us });
}
