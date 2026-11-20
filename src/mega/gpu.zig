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
    mat: [4]Mat = @splat(.{}),
    count: u32 = 0,
    act_a: u32 = 0,
    act_b: u32 = 0,
    kind: u32 = 0,
};
pub const flag_round_bf16: u32 = 1;
pub const flag_has_weight: u32 = 2;
pub const flag_fast_trig: u32 = 4;
/// decode search: mk_dec_chain_block applies the attention gate epilogue.
pub const flag_dec_gated: u32 = 8;

comptime {
    if (@sizeOf(Params) != 160) @compileError("Params must match MK in kernels.metal");
    if (@sizeOf(Mat) != 48 or @sizeOf(ParamsX) != 160 + 4 * 48 + 16) @compileError("ParamsX must match MKX in fused.metal");
}

pub const source = @embedFile("kernels.metal") ++ "\n" ++ @embedFile("fused.metal") ++ "\n" ++ @embedFile("decode.metal") ++ "\n" ++ @embedFile("prefill.metal") ++ "\n" ++ @embedFile("mlx.metal");

pub fn open() !metal.Context {
    const handle = fm_open(source) orelse {
        std.debug.print("Metal: {s}\n", .{std.mem.span(fm_error())});
        return error.MetalFailure;
    };
    return .{ .handle = handle };
}

pub fn dispatch(ctx: metal.Context, kernel: [:0]const u8, buffers: []const metal.Buffer, p: anytype, threads: usize, group: usize) !void {
    const T = @typeInfo(@TypeOf(p)).pointer.child;
    comptime if (T != Params and T != ParamsX) @compileError("unknown parameter block");
    if (buffers.len > 8 or threads == 0 or group == 0) return error.InvalidDispatch;
    var handles: [8]*anyopaque = undefined;
    for (buffers, 0..) |buf, i| handles[i] = buf.handle;
    if (fm_dispatch(ctx.handle, kernel, &handles, buffers.len, p, @sizeOf(T), threads, group) == 0) {
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
