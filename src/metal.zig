const std = @import("std");
extern fn fm_open([*:0]const u8) ?*anyopaque;
extern fn fm_close(*anyopaque) void;
extern fn fm_is_m3_ultra(*anyopaque) c_int;
extern fn fm_device_name(*anyopaque) [*:0]const u8;
extern fn fm_error() [*:0]const u8;
extern fn fm_finish(*anyopaque) c_int;
extern fn fm_profile_begin(*anyopaque) c_int;
extern fn fm_alloc(*anyopaque, usize) ?*anyopaque;
extern fn fm_wrap(*anyopaque, *anyopaque, usize) ?*anyopaque;
extern fn fm_contents(*anyopaque) *anyopaque;
extern fn fm_release(*anyopaque) void;
extern fn fm_address(*anyopaque) u64;
extern fn fm_resident(*anyopaque, *anyopaque) c_int;
extern fn fm_resident_write(*anyopaque, *anyopaque) c_int;
extern fn fm_stats(*anyopaque, *Stats) void;

pub const Stats = extern struct { submissions: u64, dispatches: u64, barriers: u64, gpu_seconds: f64 = 0 };

/// GPU-address ABI, mirrored by Matrix in kernels.metal. Backing resources must be resident.
pub const Matrix = extern struct {
    weight: u64,
    scales: u64 = 0,
    biases: u64 = 0,
    rows: u32,
    cols: u32,
    dtype: u32,
    bits: u32 = 0,
    group_size: u32 = 0,
    scale_dtype: u32 = 0,
    bias_dtype: u32 = 0,
    reserved: u32 = 0,
};
pub const MLP = extern struct { gate: Matrix, up: Matrix, down: Matrix };
extern fn fm_replay_prepare(*anyopaque) c_int;
extern fn fm_replay_enable(*anyopaque, c_int) c_int;
extern fn fm_replay_clear(*anyopaque) c_int;
extern fn fm_dispatch(*anyopaque, [*:0]const u8, [*]const *anyopaque, usize, *const anyopaque, usize, usize, usize) c_int;
extern fn fm_map([*:0]const u8, *usize, *usize) ?[*]u8;
extern fn fm_unmap(*anyopaque, usize) void;

pub const Params = extern struct {
    offset: u64 = 0,
    scale_offset: u64 = 0,
    bias_offset: u64 = 0,
    n: u32 = 0,
    m: u32 = 0,
    k: u32 = 0,
    dtype: u32 = 0,
    pos: u32 = 0,
    capacity: u32 = 0,
    heads: u32 = 0,
    kv_heads: u32 = 0,
    dim: u32 = 0,
    window: u32 = 0,
    bits: u32 = 0,
    scale_dtype: u32 = 0,
    bias_dtype: u32 = 0,
    eps: f32 = 1e-6,
    theta: f32 = 10000,
    scale: f32 = 1,
};

fn failure() error{MetalFailure} {
    std.debug.print("Metal: {s}\n", .{std.mem.span(fm_error())});
    return error.MetalFailure;
}

pub const Mapping = struct {
    bytes: []u8,
    mapped_length: usize,

    pub fn open(allocator: std.mem.Allocator, path: []const u8) !Mapping {
        const z = try allocator.dupeSentinel(u8, path, 0);
        defer allocator.free(z);
        var len: usize = 0;
        var mapped: usize = 0;
        const ptr = fm_map(z, &len, &mapped) orelse return error.FileMappingFailed;
        return .{ .bytes = ptr[0..len], .mapped_length = mapped };
    }
    pub fn deinit(self: Mapping) void {
        fm_unmap(self.bytes.ptr, self.mapped_length);
    }
};

pub const Buffer = struct {
    handle: *anyopaque,
    len: usize,

    pub fn address(self: Buffer) u64 {
        return fm_address(self.handle);
    }
    pub fn structs(self: Buffer, comptime T: type) []T {
        const ptr: [*]T = @ptrCast(@alignCast(fm_contents(self.handle)));
        return ptr[0 .. self.len / @sizeOf(T)];
    }
    pub fn deinit(self: Buffer) void {
        fm_release(self.handle);
    }
    // Reading/writing shared memory is only valid after Context.finish().
    pub fn floats(self: Buffer) []f32 {
        const ptr: [*]f32 = @ptrCast(@alignCast(fm_contents(self.handle)));
        return ptr[0 .. self.len / @sizeOf(f32)];
    }
};

pub const Context = struct {
    handle: *anyopaque,

    pub fn init() !Context {
        return initMatched(false);
    }
    pub fn initMatched(matched: bool) !Context {
        const source = @embedFile("kernels.metal") ++ "\n" ++ @embedFile("resident_plan.metal") ++ "\n" ++ @embedFile("row_tiling.metal") ++ "\n" ++ @embedFile("row_tiling_affine.metal") ++ "\n" ++ @embedFile("row_tiling_prepared.metal") ++ "\n" ++ @embedFile("row_tiling_prepared_fused.metal") ++ "\n" ++ @embedFile("row_tiling_fixed.metal") ++ "\n" ++ @embedFile("routing.metal") ++ "\n" ++ @embedFile("dense_matvec.metal") ++ "\n" ++ @embedFile("attention_parallel.metal") ++ "\n" ++ @embedFile("attention_split.metal") ++ "\n" ++ @embedFile("attention_ordered.metal") ++ "\n" ++ @embedFile("attention_mlx.metal") ++ "\n" ++ @embedFile("attention_mlx_wide.metal") ++ "\n" ++ @embedFile("attention_mlx_ring.metal") ++ "\n" ++ @embedFile("attention_mlx_fixed.metal") ++ "\n" ++ @embedFile("attention_gqa_reuse.metal") ++ "\n" ++ @embedFile("attention_gqa_stream.metal") ++ "\n" ++ @embedFile("attention_gqa_grouped.metal") ++ "\n" ++ @embedFile("normalization.metal") ++ "\n" ++ @embedFile("normalization_aligned.metal") ++ "\n" ++ @embedFile("normalization_prepared.metal") ++ "\n" ++ @embedFile("kv_bf16.metal") ++ "\n" ++ @embedFile("resident.metal") ++ "\n" ++ @embedFile("token_batch.metal") ++ "\n" ++ @embedFile("token_batch_attention.metal") ++ "\n" ++ @embedFile("token_batch_weight_reuse.metal") ++ "\n" ++ @embedFile("token_batch_greedy.metal") ++ "\n" ++ @embedFile("token_batch_mlp_weight_reuse.metal");
        return .{ .handle = fm_open(if (matched) "#define FLINT_MLX_MATH 1\n" ++ source else source) orelse return failure() };
    }
    pub fn isM3Ultra(self: Context) bool {
        return fm_is_m3_ultra(self.handle) != 0;
    }
    pub fn deviceName(self: Context) []const u8 {
        return std.mem.span(fm_device_name(self.handle));
    }
    pub fn deinit(self: Context) void {
        fm_close(self.handle);
    }
    pub fn prepareReplay(self: Context) !void {
        comptime {
            if (@sizeOf(Params) > @import("resident_plan.zig").replay_param_stride)
                @compileError("Metal Params exceed the Lean-certified replay slot");
        }
        if (fm_replay_prepare(self.handle) == 0) return failure();
    }
    pub fn setReplay(self: Context, enabled: bool) !void {
        if (fm_replay_enable(self.handle, @intFromBool(enabled)) == 0) return failure();
    }
    pub fn clearReplay(self: Context) !void {
        if (fm_replay_clear(self.handle) == 0) return failure();
    }
    pub fn beginProfile(self: Context) !void {
        if (fm_profile_begin(self.handle) == 0) return failure();
    }
    pub fn finish(self: Context) !void {
        if (fm_finish(self.handle) == 0) return failure();
    }
    pub fn resident(self: Context, buffer: Buffer) !void {
        if (fm_resident(self.handle, buffer.handle) == 0) return failure();
    }
    pub fn residentWrite(self: Context, buffer: Buffer) !void {
        if (fm_resident_write(self.handle, buffer.handle) == 0) return failure();
    }
    pub fn stats(self: Context) Stats {
        var result: Stats = undefined;
        fm_stats(self.handle, &result);
        return result;
    }
    pub fn alloc(self: Context, count: usize) !Buffer {
        const bytes = try std.math.mul(usize, count, @sizeOf(f32));
        return .{ .handle = fm_alloc(self.handle, bytes) orelse return failure(), .len = bytes };
    }
    pub fn wrap(self: Context, mapping: Mapping) !Buffer {
        return .{ .handle = fm_wrap(self.handle, mapping.bytes.ptr, mapping.mapped_length) orelse return failure(), .len = mapping.bytes.len };
    }
    pub fn dispatch(self: Context, kernel: [:0]const u8, buffers: []const Buffer, p: Params, threads: usize, group: usize) !void {
        if (buffers.len > 8 or threads == 0) return error.InvalidDispatch;
        var handles: [8]*anyopaque = undefined;
        for (buffers, 0..) |buf, i| handles[i] = buf.handle;
        if (fm_dispatch(self.handle, kernel, &handles, buffers.len, &p, @sizeOf(Params), threads, group) == 0) return failure();
    }
};

test "headless Metal matvec, RMSNorm, split-half RoPE, SwiGLU, residual" {
    const ctx = try Context.init();
    defer ctx.deinit();
    const w = try ctx.alloc(8);
    defer w.deinit();
    const x = try ctx.alloc(4);
    defer x.deinit();
    const y = try ctx.alloc(4);
    defer y.deinit();
    @memcpy(w.floats(), &[_]f32{ 1, 2, 3, 4, -1, 0, 1, 2 });
    @memcpy(x.floats(), &[_]f32{ 1, 2, 3, 4 });
    try ctx.dispatch("matvec", &.{ w, x, y }, .{ .n = 4, .m = 2 }, 64, 32);
    try ctx.finish();
    try std.testing.expectApproxEqAbs(@as(f32, 30), y.floats()[0], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 10), y.floats()[1], 1e-5);
    @memset(w.floats(), 1);
    try ctx.dispatch("rmsnorm", &.{ w, x, y }, .{ .n = 4, .m = 1 }, 32, 32);
    try ctx.finish();
    try std.testing.expectApproxEqAbs(1 / @sqrt(@as(f32, 7.5) + 1e-6), y.floats()[0], 1e-5);
    try ctx.dispatch("rope", &.{x}, .{ .dim = 4, .heads = 1, .pos = 1 }, 2, 2);
    try ctx.finish();
    try std.testing.expectApproxEqAbs(@cos(@as(f32, 1)) - 3 * @sin(@as(f32, 1)), x.floats()[0], 1e-5);
    try std.testing.expectApproxEqAbs(2 * @cos(@as(f32, 0.01)) - 4 * @sin(@as(f32, 0.01)), x.floats()[1], 1e-5);
    @memcpy(x.floats(), &[_]f32{ 0, 1, -1, 2 });
    @memset(w.floats(), 2);
    try ctx.dispatch("swiglu", &.{ x, w, y }, .{ .n = 4 }, 4, 4);
    try ctx.dispatch("add_scaled", &.{ x, y }, .{ .n = 4, .scale = 0.5 }, 4, 4);
    try ctx.finish();
    try std.testing.expectApproxEqAbs(1 + 1 / (1 + @exp(@as(f32, -1))), x.floats()[1], 1e-5);
}

test "learned attention sink and sliding ring buffer include current token" {
    const ctx = try Context.init();
    defer ctx.deinit();
    const q = try ctx.alloc(4);
    defer q.deinit();
    const k = try ctx.alloc(2);
    defer k.deinit();
    const v = try ctx.alloc(2);
    defer v.deinit();
    const kc = try ctx.alloc(4);
    defer kc.deinit();
    const vc = try ctx.alloc(4);
    defer vc.deinit();
    const sink = try ctx.alloc(2);
    defer sink.deinit();
    const out = try ctx.alloc(4);
    defer out.deinit();
    @memset(q.floats(), 0);
    @memset(k.floats(), 0);
    @memset(sink.floats(), 0);
    for (0..4) |pos| {
        @memset(v.floats(), @as(f32, @floatFromInt(pos + 1)));
        const p: Params = .{ .n = 2, .pos = @intCast(pos), .capacity = 2, .heads = 2, .kv_heads = 1, .dim = 2, .window = 2 };
        try ctx.dispatch("cache_write", &.{ k, v, kc, vc }, p, 2, 2);
        try ctx.dispatch("attention", &.{ q, kc, vc, sink, out }, p, 64, 32);
        try ctx.finish();
        const expected: f32 = if (pos == 0) 0.5 else @as(f32, @floatFromInt(2 * pos + 1)) / 3;
        for (out.floats()) |value| try std.testing.expectApproxEqAbs(expected, value, 1e-5);
    }
}

test "GPU top-k bias and ties, explicit dispatch barriers, and one submission" {
    try std.testing.expectEqual(@as(usize, 56), @sizeOf(Matrix));
    try std.testing.expectEqual(@as(usize, 168), @sizeOf(MLP));
    const ctx = try Context.init();
    defer ctx.deinit();
    const logits = try ctx.alloc(4);
    defer logits.deinit();
    const bias = try ctx.alloc(4);
    defer bias.deinit();
    const ids = try ctx.alloc(2);
    defer ids.deinit();
    const weights = try ctx.alloc(2);
    defer weights.deinit();
    const flag = try ctx.alloc(1);
    defer flag.deinit();
    @memcpy(logits.floats(), &[_]f32{ 0, 2, -2, 2 });
    @memcpy(bias.floats(), &[_]f32{ 0, 0, 2, 0 });
    flag.structs(u32)[0] = 0;
    const before = ctx.stats();
    try ctx.dispatch("route_topk", &.{ logits, bias, ids, weights, flag }, .{ .n = 2, .m = 4 }, 32, 32);
    // Second dispatch consumes the first one's output within the same encoder.
    try ctx.dispatch("add_scaled", &.{ weights, weights }, .{ .n = 2, .scale = 1 }, 2, 2);
    try ctx.finish();
    const after = ctx.stats();
    try std.testing.expectEqual(@as(u64, 1), after.submissions - before.submissions);
    try std.testing.expectEqual(@as(u64, 2), after.dispatches - before.dispatches);
    try std.testing.expectEqual(@as(u64, 1), after.barriers - before.barriers);
    try std.testing.expectEqualSlices(u32, &.{ 2, 1 }, ids.structs(u32));
    try std.testing.expectApproxEqAbs(@as(f32, 2) / (1 + @exp(@as(f32, 2))), weights.floats()[0], 1e-6);
    try std.testing.expectEqual(@as(u32, 0), flag.structs(u32)[0]);
    @memset(logits.floats(), std.math.nan(f32));
    try ctx.dispatch("route_topk", &.{ logits, bias, ids, weights, flag }, .{ .n = 2, .m = 4 }, 32, 32);
    try ctx.finish();
    try std.testing.expectEqual(@as(u32, 1), flag.structs(u32)[0]);
    for (ids.structs(u32)) |id| try std.testing.expect(id < 4);
}
