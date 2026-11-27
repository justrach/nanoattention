//! Megakernel use of the shared headless bridge (src/metal.m). The bridge is
//! unchanged: this file only compiles the megakernel library and passes its own
//! parameter block. metal.Context methods (alloc, wrap, finish, stats) apply.
const std = @import("std");
const metal = @import("weights").metal;

extern fn fm_open([*:0]const u8) ?*anyopaque;
extern fn fm_error() [*:0]const u8;
extern fn fm_dispatch(*anyopaque, [*:0]const u8, [*]const *anyopaque, usize, *const anyopaque, usize, usize, usize) c_int;

/// Mirror of MK in src/mega/kernels.metal.
pub const Params = extern struct {
    w: u64 = 0,
    s: u64 = 0,
    b: u64 = 0,
    rows: u32 = 0,
    row0: u32 = 0,
    pos: u32 = 0,
    pad_r: u32 = 0,
    x: u32 = 0,
    xs: u32 = 0,
    y: u32 = 0,
    ys: u32 = 0,
    z: u32 = 0,
    zs: u32 = 0,
    n: u32 = 0,
    m: u32 = 0,
    dtype: u32 = 0,
    bits: u32 = 0,
    group: u32 = 0,
    sdtype: u32 = 0,
    bdtype: u32 = 0,
    flags: u32 = 0,
    heads: u32 = 0,
    kv_heads: u32 = 0,
    dim: u32 = 0,
    window: u32 = 0,
    capacity: u32 = 0,
    taps: u32 = 0,
    rot: u32 = 0,
    every: u32 = 0,
    kind: u32 = 0,
    act_a: u32 = 0,
    act_b: u32 = 0,
    pad_k: u32 = 0,
    eps: f32 = 0,
    scale: f32 = 0,
    pad_f0: f32 = 0,
    pad_f1: f32 = 0,
};
/// Mirror of MKExpert for a single packed expert projection.
pub const ExpertParams = extern struct {
    w: u64 = 0,
    s: u64 = 0,
    amax: u64 = 0,
    denominator: u64 = 0,
    rows: u32 = 0,
    row0: u32 = 0,
    x: u32 = 0,
    xs: u32 = 0,
    routes: u32 = 0,
    rs: u32 = 0,
    y: u32 = 0,
    ys: u32 = 0,
    k: u32 = 0,
    n: u32 = 0,
    experts: u32 = 0,
    top_k: u32 = 0,
    nvfp4: u32 = 0,
    round_bf16: u32 = 0,
    gate_cap: f32 = 0,
    up_cap: f32 = 0,
};

/// One matrix of a multi-matrix launch (MKMat in fused.metal).
pub const Mat = extern struct {
    w: u64 = 0,
    s: u64 = 0,
    b: u64 = 0,
    m: u32 = 0,
    y: u32 = 0,
    ys: u32 = 0,
    sdtype: u32 = 0,
    bdtype: u32 = 0,
    pad: u32 = 0,
};
/// Mirror of MKX in src/mega/fused.metal.
pub const ParamsX = extern struct {
    p: Params = .{},
    mat: [6]Mat = @splat(.{}),
    count: u32 = 0,
    act_a: u32 = 0,
    act_b: u32 = 0,
    kind: u32 = 0,
};
/// Mirror of MKLSA in src/mega/inkling_lsa.metal (240 bytes).
pub const LSAParams = extern struct {
    off_head_scores: u64 = 0,
    off_tg_stats: u64 = 0,
    off_head_lse: u64 = 0,
    off_scores: u64 = 0,
    off_hist: u64 = 0,
    off_cand_count: u64 = 0,
    off_cand_m: u64 = 0,
    off_cand_u: u64 = 0,
    off_fine_cand: u64 = 0,
    off_fine_count: u64 = 0,
    rel_w_off: u64 = 0,
    units: u32 = 0,
    rows: u32 = 0,
    row0: u32 = 0,
    pos0: u32 = 0,
    row_step: u32 = 1,
    unit_offset: u32 = 0,
    dim: u32 = 0,
    heads: u32 = 0,
    kv_heads: u32 = 0,
    capacity: u32 = 0,
    q_off: u32 = 0,
    q_stride: u32 = 0,
    sink_off: u32 = 0,
    sink_stride: u32 = 0,
    bias_off: u32 = 0,
    bias_stride: u32 = 0,
    out_off: u32 = 0,
    out_stride: u32 = 0,
    m: u32 = 0,
    budget_blocks: u32 = 0,
    window_blocks: u32 = 0,
    coarse_cap: u32 = 0,
    pooled_meta_bytes: u32 = 0,
    pooled_row_bytes: u32 = 0,
    start_blk: u32 = 0,
    num_blks: u32 = 0,
    list_stride: u32 = 0,
    score_stride: u32 = 0,
    head_score_stride: u32 = 0,
    cand_cap: u32 = 0,
    score_tiles_x: u32 = 0,
    stage_mode: u32 = 0,
    bq_tile: u32 = 1,
    taps: u32 = 0,
    flags: u32 = 0,
    has_sink: u32 = 0,
    scale: f32 = 0,
    tile_items: u32 = 0,
};
pub const flag_round_bf16: u32 = 1;
pub const flag_has_weight: u32 = 2;
pub const flag_fast_trig: u32 = 4;
/// decode search: mk_dec_chain_block applies the attention gate epilogue.
pub const flag_dec_gated: u32 = 8;
pub const flag_round_out: u32 = 16;
/// mk_relative_prefill_tiles: the distance bias is 16-bit (knobs.relative_bias16).
pub const flag_bias16: u32 = 32;
/// mk_relative_long_merge_*: keep host-specified `p.taps` instead of recomputing dense taps.
pub const flag_fixed_taps: u32 = 128;

comptime {
    if (@sizeOf(ExpertParams) != 96) @compileError("ExpertParams must match MKExpert in inkling_ops.metal");
    if (@sizeOf(Params) != 160) @compileError("Params must match MK in kernels.metal");
    if (@sizeOf(Mat) != 48 or @sizeOf(ParamsX) != 160 + 6 * 48 + 16) @compileError("ParamsX must match MKX in fused.metal");
    if (@sizeOf(LSAParams) != 240) @compileError("LSAParams must match MKLSA in inkling_lsa.metal");
}

pub const lfm_source = @embedFile("kernels.metal") ++ "\n" ++ @embedFile("fused.metal") ++ "\n" ++ @embedFile("decode.metal") ++ "\n" ++ @embedFile("prefill.metal") ++ "\n" ++ @embedFile("mlx.metal") ++
    "\n" ++ @embedFile("mlx2.metal") ++ "\n" ++ @embedFile("decode2.metal") ++ "\n" ++ @embedFile("exact2.metal");

pub const source = @embedFile("kernels.metal") ++ "\n" ++ @embedFile("inkling_plan.metal") ++ "\n" ++ @embedFile("inkling_ops.metal") ++ "\n" ++ @embedFile("inkling_long.metal") ++ "\n" ++ @embedFile("inkling_lsa.metal") ++ "\n" ++ @embedFile("inkling_tiles.metal") ++ "\n" ++ @embedFile("fused.metal") ++ "\n" ++ @embedFile("inkling_fast.metal") ++ "\n" ++ @embedFile("decode.metal") ++ "\n" ++ @embedFile("prefill.metal") ++ "\n" ++ @embedFile("mlx.metal") ++
    // second climb: prefill in MLX arithmetic, decode, then exact prefill on the matrix units
    // (exact2.metal uses mk_mlx_sb, so it stays after mlx.metal)
    "\n" ++ @embedFile("mlx2.metal") ++ "\n" ++ @embedFile("decode2.metal") ++ "\n" ++ @embedFile("exact2.metal") ++ "\n" ++ @embedFile("dflash2.metal");

pub fn openSource(src: [:0]const u8) !metal.Context {
    const handle = fm_open(src) orelse {
        std.debug.print("Metal: {s}\n", .{std.mem.span(fm_error())});
        return error.MetalFailure;
    };
    return .{ .handle = handle };
}

pub fn openForModel(model_type: []const u8) !metal.Context {
    if (std.mem.eql(u8, model_type, "lfm2") or
        std.mem.eql(u8, model_type, "lfm3") or
        std.mem.eql(u8, model_type, "qwen3_5") or
        std.mem.eql(u8, model_type, "qwen3_5_text") or
        std.mem.eql(u8, model_type, "gemma4") or
        std.mem.eql(u8, model_type, "gemma4_text") or
        std.mem.eql(u8, model_type, "gemma4_unified") or
        std.mem.eql(u8, model_type, "gemma4_unified_text"))
    {
        return openSource(lfm_source);
    }
    return openSource(source);
}

pub fn open() !metal.Context {
    return openSource(source);
}

pub fn dispatch(ctx: metal.Context, kernel: [:0]const u8, buffers: []const metal.Buffer, p: anytype, threads: usize, group: usize) !void {
    const T = @typeInfo(@TypeOf(p)).pointer.child;
    comptime if (T != Params and T != ParamsX and T != ExpertParams and T != LSAParams) @compileError("unknown parameter block");
    if (buffers.len > 8 or threads == 0 or group == 0) return error.InvalidDispatch;
    var handles: [8]*anyopaque = undefined;
    for (buffers, 0..) |buf, i| handles[i] = buf.handle;
    const ok = if (ctx.nano_dispatch_flags) |flags|
        fm_d2_dispatch(ctx.handle, kernel, &handles, buffers.len, p, @sizeOf(T), threads, group, flags)
    else
        fm_dispatch(ctx.handle, kernel, &handles, buffers.len, p, @sizeOf(T), threads, group);
    if (ok == 0) {
        std.debug.print("Metal: {s}\n", .{std.mem.span(fm_error())});
        return error.MetalFailure;
    }
}

/// RMSNorm threadgroup: one thread per 4 words, a SIMD multiple, at most 1024.
/// A function of the group length alone, so every schedule reduces identically.
pub fn normThreads(n: u32) u32 {
    const t = (n + 3) / 4;
    return @min(1024, (t + 31) / 32 * 32);
}

// ---- decode climb 2 (c2-decode) ----
// Asynchronous commits and waits (src/metal.m, fm_d2_*): the gated decode
// (exec.d2GatedForward) and the measurements of `nano d2-probe`.
extern fn fm_d2_dispatch(*anyopaque, [*:0]const u8, [*]const *anyopaque, usize, *const anyopaque, usize, usize, usize, c_int) c_int;
extern fn fm_d2_commit(*anyopaque) c_int;
extern fn fm_d2_drain(*anyopaque, usize) c_int;
extern fn fm_d2_pending_count(*anyopaque) usize;
extern fn fm_d2_finish(*anyopaque, c_int, ?*[4]f64) c_int;
extern fn fm_d2_gate(*anyopaque, u64) c_int;
extern fn fm_d2_signal(*anyopaque, u64) c_int;
extern fn fm_d2_wait_oldest(*anyopaque, c_int, ?*[4]f64) c_int;
extern fn fm_d2_spin(*anyopaque, usize, *const volatile u32, u32) c_int;
extern fn fm_d2_reap(*anyopaque) c_int;
fn d2check(ok: c_int) !void {
    if (ok == 0) {
        std.debug.print("Metal: {s}\n", .{std.mem.span(fm_error())});
        return error.MetalFailure;
    }
}
pub const d2_barrier: c_int = 1;
pub const d2_concurrent: c_int = 2;
/// d2-probe: dispatch with an explicit policy (d2_barrier: a buffer barrier
/// before it; d2_concurrent: a new encoder is concurrent) instead of fm_dispatch's
/// barrier between every two dispatches of a serial encoder.
pub fn dispatchFlags(ctx: metal.Context, kernel: [:0]const u8, buffers: []const metal.Buffer, p: *const Params, threads: usize, group: usize, flags: c_int) !void {
    var handles: [8]*anyopaque = undefined;
    for (buffers, 0..) |buf, i| handles[i] = buf.handle;
    try d2check(fm_d2_dispatch(ctx.handle, kernel, &handles, buffers.len, p, @sizeOf(Params), threads, group, flags));
}
/// Commit the open command buffer without waiting.
pub fn commitAsync(ctx: metal.Context) !void {
    try d2check(fm_d2_commit(ctx.handle));
}
/// Wait until at most `keep` asynchronous commits are still pending.
pub fn drain(ctx: metal.Context, keep: usize) !void {
    try d2check(fm_d2_drain(ctx.handle, keep));
}
pub fn pending(ctx: metal.Context) usize {
    return fm_d2_pending_count(ctx.handle);
}
/// Finish and wait (spin: on the status instead of waitUntilCompleted).
/// times: commit, GPU start, GPU end, return (host seconds).
pub fn finishWait(ctx: metal.Context, spin_wait: bool, times: ?*[4]f64) !void {
    try d2check(fm_d2_finish(ctx.handle, @intFromBool(spin_wait), times));
}
/// d2-probe: the open (or a new) command buffer waits for a shared event to reach `value`.
pub fn gate(ctx: metal.Context, value: u64) !void {
    try d2check(fm_d2_gate(ctx.handle, value));
}
pub fn signal(ctx: metal.Context, value: u64) !void {
    try d2check(fm_d2_signal(ctx.handle, value));
}
/// Wait for the oldest asynchronous commit; times as finishWait (the commit or a later signal first).
pub fn waitOldest(ctx: metal.Context, spin_wait: bool, times: ?*[4]f64) !void {
    try d2check(fm_d2_wait_oldest(ctx.handle, @intFromBool(spin_wait), times));
}
/// Spin until `watch` leaves `sentinel` or pending buffer `index` (0 = oldest) finishes.
pub fn spin(ctx: metal.Context, index: usize, watch: *const volatile u32, sentinel: u32) !void {
    try d2check(fm_d2_spin(ctx.handle, index, watch, sentinel));
}
/// Release the leading finished pending buffers; fails if any of them failed.
pub fn reap(ctx: metal.Context) !void {
    try d2check(fm_d2_reap(ctx.handle));
}
