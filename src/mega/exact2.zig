//! climb 2, exact prefill: exact multi-row kernels on the matrix units (exact2.metal).
//! `encodeLinear` is the executor's entry (knob x2_linear); `run` is
//! `nano exact2-bench [tokens] [repeats]`: GPU time of the exact multi-row 4-bit
//! linears at LFM3's shapes on random weights placed 2 bytes off a word boundary
//! (as in the real checkpoint), each output checked bit for bit against the
//! reference executor's multi-row kernel (mk_linear_q4_r2_t4). Timing guidance
//! only; run it under the inference lock and confirm any gain with `nano bench`.
const std = @import("std");
const gpu = @import("gpu.zig");
const io = @import("io.zig");
const metal = @import("weights").metal;

/// The compiled tile of mk_x2mm_q4: blocks per SIMD group, SIMD groups per
/// threadgroup, segments per stage (exact2-bench screened the others).
pub const tile = .{ .mt = 2, .nt = 2, .wm = 2, .wn = 2, .ks = 4 };
const bm = 8 * tile.mt * tile.wm;
const bn = 8 * tile.nt * tile.wn;
const threads = 32 * tile.wm * tile.wn;

/// Kernel per codes-as-half (knob x2_half) and scale/bias dtype (MK dtype code 0 f32, 1 f16, 2 bf16).
pub const kernels = [2][3][:0]const u8{
    .{ "mk_x2mm_q4_m2_n2_w2x2_k4_f32", "mk_x2mm_q4_m2_n2_w2x2_k4_f16", "mk_x2mm_q4_m2_n2_w2x2_k4_bf16" },
    .{ "mk_x2mmh_q4_m2_n2_w2x2_k4_f32", "mk_x2mmh_q4_m2_n2_w2x2_k4_f16", "mk_x2mmh_q4_m2_n2_w2x2_k4_bf16" },
};

/// mk_x2_att_chain flag: apply mk_attention_gated's epilogue (X2_ATT_GATED).
pub const att_gated: u32 = 8;

/// Words of mk_x2_prep's buffer for `rows` rows of `n` inputs, and its region offsets.
pub fn prepLayout(rows: u32, n: u32) struct { words: usize, xsum: u32, flags: u32 } {
    const segs = n / 16;
    const xsum = rows * n;
    const flags = xsum + rows * segs;
    return .{ .words = @as(usize, flags) + rows, .xsum = xsum, .flags = flags };
}

/// Whether mk_x2mm applies on top of the 4-bit fast form's conditions (exec.q4Fast):
/// one dtype for scales and biases, known to the kernel.
pub fn applies(sdtype: u32, bdtype: u32) bool {
    return sdtype == bdtype and sdtype <= 2;
}

/// Prepare (optionally) and run one multi-row linear: `p` carries the linear's
/// operands (rows, row0, x, xs, y, ys, n, m, w/s/b, group, dtypes); z and every
/// are set to the prepared buffer's regions.
pub fn encodeLinear(ctx: metal.Context, qb: [3]metal.Buffer, arena: metal.Buffer, prep: metal.Buffer, p: *gpu.Params, prepare: bool, half_codes: bool) !void {
    try encodeLinearTile(ctx, kernels[@intFromBool(half_codes)][p.sdtype], .{ bm, bn, threads }, qb, arena, prep, p, prepare);
}
/// encodeLinear with a given kernel and its (tokens, rows, threads) per threadgroup.
pub fn encodeLinearTile(ctx: metal.Context, name: [:0]const u8, t: [3]u32, qb: [3]metal.Buffer, arena: metal.Buffer, prep: metal.Buffer, p: *gpu.Params, prepare: bool) !void {
    const lay = prepLayout(p.rows, p.n);
    p.z = lay.xsum;
    p.every = lay.flags;
    if (prepare) try gpu.dispatch(ctx, "mk_x2_prep", &.{ arena, prep }, p, @as(usize, p.rows) * 32, 256);
    const tgs = ((@as(usize, p.m) + t[1] - 1) / t[1]) * ((@as(usize, p.rows) + t[0] - 1) / t[0]);
    try gpu.dispatch(ctx, name, &.{ qb[0], qb[1], qb[2], arena, prep }, p, tgs * t[2], t[2]);
}

/// exact2-bench's screened variants: kernel and (tokens, rows, threads) per threadgroup.
const bench_variants = [_]struct { [:0]const u8, [3]u32 }{
    .{ kernels[0][2], .{ bm, bn, threads } },
    .{ kernels[1][2], .{ bm, bn, threads } },
};

/// Whether the batched attention kernels cover this shape (compiled classes).
pub fn attentionApplies(heads: u32, kv_heads: u32, dim: u32) bool {
    if (kv_heads == 0 or heads % kv_heads != 0 or dim == 0 or dim > 256) return false;
    const e = (dim + 31) / 32;
    const g = heads / kv_heads;
    return std.math.isPowerOfTwo(e) and e <= 8 and std.math.isPowerOfTwo(g) and g <= 8;
}

/// Encode batched exact attention for p.rows rows from p.row0 in chunks of at most
/// `chunk` rows. `p`: pos, x/xs (queries), y/ys (output), heads, kv_heads, dim,
/// window, capacity, scale, every (= score stride, at least the widest span) and,
/// gated, z/zs, act_b and flags |= att_gated. `bufs`: scores/es, f, (m, l), each
/// sized for `chunk` rows. Restores p.rows and p.row0.
pub fn encodeAttention(ctx: metal.Context, arena: metal.Buffer, keys: metal.Buffer, values: metal.Buffer, bufs: [3]metal.Buffer, p: *gpu.Params, chunk: u32) !void {
    const E = (p.dim + 31) / 32;
    const G = p.heads / p.kv_heads;
    const ei = std.math.log2_int(u32, E);
    const gi = std.math.log2_int(u32, G);
    const scores = comptime blk: {
        var n: [4][4][:0]const u8 = undefined;
        for (.{ 1, 2, 4, 8 }, 0..) |e, i| for (.{ 1, 2, 4, 8 }, 0..) |g, j| {
            n[i][j] = std.fmt.comptimePrint("mk_x2_att_scores_e{d}_g{d}_r4", .{ e, g });
        };
        break :blk n;
    };
    const chains = [2][4][:0]const u8{
        .{ "mk_x2_att_chain_g1_r1_own", "mk_x2_att_chain_g2_r1_own", "mk_x2_att_chain_g4_r1_own", "mk_x2_att_chain_g8_r1_own" },
        .{ "mk_x2_att_chain_g1_r16_shared", "mk_x2_att_chain_g2_r8_shared", "mk_x2_att_chain_g4_r4_shared", "mk_x2_att_chain_g8_r2_shared" },
    };
    const full = p.window == 0;
    const rows = p.rows;
    const row0 = p.row0;
    defer {
        p.rows = rows;
        p.row0 = row0;
    }
    var c0: u32 = 0;
    while (c0 < rows) : (c0 += chunk) {
        const cr = @min(chunk, rows - c0);
        p.rows = cr;
        p.row0 = row0 + c0;
        // Scores: 4 rows per SIMD group, chunks of 8 positions from each block's first start.
        const end = p.pos + p.row0 + cr; // one past the last position
        p.m = if (full) (end + 7) / 8 else (@min(p.window, end) + 3 + 7) / 8;
        try gpu.dispatch(ctx, scores[ei][gi], &.{ arena, keys, bufs[0] }, p, @as(usize, (cr + 3) / 4) * p.kv_heads * p.m * 32, 256);
        try gpu.dispatch(ctx, "mk_x2_att_soft", &.{ bufs[0], bufs[1], bufs[2] }, p, @as(usize, cr) * p.heads * 2 * 32, 256);
        const rb: u32 = if (full) 16 / G else 1;
        const tgs = @as(usize, (cr + rb - 1) / rb) * p.kv_heads * E;
        try gpu.dispatch(ctx, chains[@intFromBool(full)][gi], &.{ arena, values, bufs[0], bufs[1], bufs[2] }, p, tgs * 256, 256);
    }
}

fn bf16(v: f32) u16 {
    const bits: u32 = @bitCast(v);
    return @intCast((bits + 0x7fff + ((bits >> 16) & 1)) >> 16);
}

pub fn run(a: std.mem.Allocator, tokens: u32, repeats: u32) !void {
    const ctx = try gpu.open();
    defer ctx.deinit();
    var prng = std.Random.DefaultPrng.init(42);
    const r = prng.random();
    const shapes = [_][2]u32{ .{ 2048, 2048 }, .{ 6144, 2048 }, .{ 7680, 2048 }, .{ 2048, 7680 }, .{ 512, 2048 } };
    const T = tokens;
    for (shapes) |shape| {
        const m = shape[0];
        const n = shape[1];
        const group = 64;
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
        const xw = @as(usize, T) * n;
        const ow = @as(usize, T) * m;
        const arena = try ctx.alloc(xw + (2 + bench_variants.len) * ow);
        defer arena.deinit();
        const f = arena.floats();
        for (f[0..xw]) |*v| v.* = r.floatNorm(f32);
        const qb = [3]metal.Buffer{ wbuf, wbuf, wbuf };
        const base: gpu.Params = .{ .rows = T, .w = 2, .s = 2 + code_bytes, .b = 2 + code_bytes + sb_bytes, .x = 0, .xs = n, .ys = m, .n = n, .m = m, .bits = 4, .group = group, .sdtype = 2, .bdtype = 2 };
        // Reference executor's multi-row kernel.
        var p = base;
        p.y = @intCast(xw);
        try gpu.dispatch(ctx, "mk_linear_q4_r2_t4", &.{ wbuf, wbuf, wbuf, arena }, &p, ((@as(usize, m) + 1) / 2) * ((@as(usize, T) + 3) / 4) * 32, 128);
        try ctx.finish();
        const words = std.mem.bytesAsSlice(u32, std.mem.sliceAsBytes(f));
        const ref = words[xw..][0..ow];
        const macs = @as(f64, @floatFromInt(ow)) * @as(f64, @floatFromInt(n));
        try io.print(a, "shape {d}x{d}, {d} tokens\n", .{ m, n, T });
        const segs = n / 16;
        const padded = (segs + 31) / 32 * 512;
        const prep_old = try ctx.alloc(@as(usize, T) * (padded + segs));
        defer prep_old.deinit();
        const prep_new = try ctx.alloc(prepLayout(T, n).words);
        defer prep_new.deinit();
        for (0..1 + bench_variants.len) |variant| {
            p = base;
            p.y = @intCast(xw + (1 + variant) * ow);
            @memset(f[p.y..][0..ow], 0);
            const before = ctx.stats();
            for (0..repeats) |_| {
                if (variant == 0) {
                    // The prefill-mm elite: mk_q4_prep + mk_mm_q4_r2_t8_h.
                    p.z = T * padded;
                    try gpu.dispatch(ctx, "mk_q4_prep", &.{ arena, prep_old }, &p, @as(usize, T) * segs, 256);
                    try gpu.dispatch(ctx, "mk_mm_q4_r2_t8_h", &.{ wbuf, wbuf, wbuf, arena, prep_old }, &p, ((@as(usize, m) + 1) / 2) * ((@as(usize, T) + 7) / 8) * 32, 128);
                } else try encodeLinearTile(ctx, bench_variants[variant - 1][0], bench_variants[variant - 1][1], qb, arena, prep_new, &p, true);
            }
            try ctx.finish();
            const us = (ctx.stats().gpu_seconds - before.gpu_seconds) / @as(f64, @floatFromInt(repeats)) * 1e6;
            var diffs: usize = 0;
            for (ref, words[p.y..][0..ow]) |x, y| diffs += @intFromBool(x != y);
            const name = if (variant == 0) "mk_mm_q4_r2_t8_h (+prep)" else bench_variants[variant - 1][0];
            try io.print(a, "  {s:<32} {d:>9.1} us {d:>6.2} TMAC/s  {s} ({d} words differ)\n", .{ name, us, macs / (us * 1e-6) / 1e12, if (diffs == 0) "bit-identical" else "DIFFERS", diffs });
        }
    }
}
