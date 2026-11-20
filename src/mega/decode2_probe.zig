//! decode climb 2: `nano d2-probe [--checkpoint DIR] [--reps N] [--matvec-only 1]`.
//! Costs that bound single-token decode, measured on this bridge: an empty launch
//! with and without a buffer barrier, in a serial and a concurrent encoder; where
//! a synchronous command buffer's round trip goes (commit -> GPU start -> GPU end
//! -> host return); the gap between back-to-back asynchronous command buffers;
//! a shared-event signal from the host; and, given a checkpoint, submission with
//! the 1 GB mapping bound and the 4-bit matvec's bandwidth over the mapped
//! checkpoint (tensors 2 bytes off a word) against a word-aligned copy, by rows
//! per SIMD group and threadgroup size. Timing guidance only: run it under the
//! inference lock and confirm anything with `nano bench`.
const std = @import("std");
const gpu = @import("gpu.zig");
const io = @import("io.zig");
const weights = @import("weights");
const metal = weights.metal;

fn median(v: []f64) f64 {
    std.mem.sort(f64, v, {}, std.sort.asc(f64));
    return v[v.len / 2];
}

const Samples = struct {
    a: []f64,
    b: []f64,
    c: []f64,
    fn init(al: std.mem.Allocator, n: usize) !Samples {
        return .{ .a = try al.alloc(f64, n), .b = try al.alloc(f64, n), .c = try al.alloc(f64, n) };
    }
    /// times: commit (or signal), GPU start, GPU end, return.
    fn put(self: Samples, i: usize, t: [4]f64) void {
        self.a[i] = (t[1] - t[0]) * 1e6;
        self.b[i] = (t[2] - t[1]) * 1e6;
        self.c[i] = (t[3] - t[2]) * 1e6;
    }
    fn print(self: Samples, al: std.mem.Allocator, label: []const u8) !void {
        try io.print(al, "{s}: commit->GPU start {d:>7.1} us, GPU {d:>7.1} us, GPU end->return {d:>7.1} us\n", .{ label, median(self.a), median(self.b), median(self.c) });
    }
};

pub fn run(a: std.mem.Allocator, args: []const []const u8) !void {
    var checkpoint: ?[]const u8 = null;
    var reps: usize = 20;
    var matvec_only = false;
    var i: usize = 0;
    while (i + 1 < args.len) : (i += 2) {
        if (std.mem.eql(u8, args[i], "--checkpoint")) {
            checkpoint = args[i + 1];
        } else if (std.mem.eql(u8, args[i], "--reps")) {
            reps = try std.fmt.parseInt(usize, args[i + 1], 10);
        } else if (std.mem.eql(u8, args[i], "--matvec-only")) {
            matvec_only = std.mem.eql(u8, args[i + 1], "1");
        } else return error.UnknownFlag;
    }
    if (reps == 0) return error.InvalidReps;
    const ctx = try gpu.open();
    defer ctx.deinit();
    const arena = try ctx.alloc(1024);
    defer arena.deinit();
    const p: gpu.Params = .{};
    try gpu.dispatch(ctx, "mk_d2_empty", &.{arena}, &p, 32, 32);
    try ctx.finish();
    const s = try Samples.init(a, reps);
    if (!matvec_only) try latency(a, ctx, arena, reps, s);
    if (checkpoint) |dir| try matvecs(a, ctx, arena, dir, reps, s, matvec_only);
}

fn latency(a: std.mem.Allocator, ctx: metal.Context, arena: metal.Buffer, reps: usize, s: Samples) !void {
    const p: gpu.Params = .{};
    const modes = [_]struct { []const u8, ?c_int }{
        .{ "serial, barrier (fm_dispatch)", null },
        .{ "serial, no barrier", 0 },
        .{ "concurrent, barrier", gpu.d2_barrier | gpu.d2_concurrent },
        .{ "concurrent, no barrier", gpu.d2_concurrent },
    };
    const n_disp = 200;
    for ([_]usize{ 1, 1024 }) |groups| {
        try io.print(a, "{d} dispatches of an empty kernel ({d} threadgroups of 32) per command buffer, {d} reps, medians:\n", .{ n_disp, groups, reps });
        for (modes) |m| {
            for (0..reps) |r| {
                const t0 = try io.now();
                for (0..n_disp) |_| {
                    if (m[1]) |f| try gpu.dispatchFlags(ctx, "mk_d2_empty", &.{arena}, &p, groups * 32, 32, f) else try gpu.dispatch(ctx, "mk_d2_empty", &.{arena}, &p, groups * 32, 32);
                }
                const t1 = try io.now();
                const before = ctx.stats();
                try ctx.finish();
                const t2 = try io.now();
                s.a[r] = (t1 - t0) / n_disp * 1e6;
                s.b[r] = (ctx.stats().gpu_seconds - before.gpu_seconds) / n_disp * 1e6;
                s.c[r] = (t2 - t0) * 1e6;
            }
            try io.print(a, "  {s:<30} encode {d:>6.3} us/dispatch, GPU {d:>6.3} us/dispatch, wall {d:>8.1} us/buffer\n", .{ m[0], median(s.a), median(s.b), median(s.c) });
        }
    }
    for ([_]usize{ 1, 200 }) |nd| for ([_]bool{ false, true }) |spin| {
        for (0..reps + 3) |r| {
            for (0..nd) |_| try gpu.dispatch(ctx, "mk_d2_empty", &.{arena}, &p, 32, 32);
            var t: [4]f64 = undefined;
            try gpu.finishWait(ctx, spin, &t);
            if (r >= 3) s.put(r - 3, t);
        }
        try s.print(a, try std.fmt.allocPrint(a, "synchronous, {d:>3} empty dispatches, {s:<18}", .{ nd, if (spin) "spin on status" else "waitUntilCompleted" }));
    };
    {
        var gaps: [16]f64 = undefined;
        for (0..16) |_| {
            for (0..200) |_| try gpu.dispatch(ctx, "mk_d2_empty", &.{arena}, &p, 32, 32);
            try gpu.commitAsync(ctx);
        }
        var prev_end: f64 = 0;
        for (0..16) |k| {
            var t: [4]f64 = undefined;
            try gpu.waitOldest(ctx, true, &t);
            gaps[k] = if (k == 0) 0 else (t[1] - prev_end) * 1e6;
            prev_end = t[2];
        }
        try io.print(a, "back to back, 16 asynchronous buffers of 200 empty dispatches: GPU end -> next GPU start, median {d:.1} us\n", .{median(gaps[1..])});
    }
    for (0..reps + 3) |r| {
        const v: u64 = r + 1;
        try gpu.gate(ctx, v);
        for (0..200) |_| try gpu.dispatch(ctx, "mk_d2_empty", &.{arena}, &p, 32, 32);
        try gpu.commitAsync(ctx);
        const t0 = try io.now();
        while ((try io.now()) - t0 < 0.002) {}
        try gpu.signal(ctx, v);
        var t: [4]f64 = undefined;
        try gpu.waitOldest(ctx, true, &t);
        if (r >= 3) s.put(r - 3, t);
    }
    try s.print(a, "shared event, 200 empty dispatches committed 2 ms before the host signal (signal->GPU start)");
}

fn matvecs(a: std.mem.Allocator, ctx: metal.Context, arena: metal.Buffer, dir: []const u8, reps: usize, s: Samples, matvec_only: bool) !void {
    const p: gpu.Params = .{};
    const path = try std.fs.path.join(a, &.{ dir, "model.safetensors" });
    const mapping = try metal.Mapping.open(a, path);
    defer mapping.deinit();
    const mapped = try ctx.wrap(mapping);
    defer mapped.deinit();
    // The same bytes 2 bytes further on: every tensor (2 mod 4 in the file) word-aligned.
    const aligned = try ctx.alloc((mapping.bytes.len + 2 + 3) / 4);
    defer aligned.deinit();
    @memcpy(std.mem.sliceAsBytes(aligned.floats())[2..][0..mapping.bytes.len], mapping.bytes);
    const out = try ctx.alloc(1 << 20);
    defer out.deinit();
    @memset(out.floats(), 0.01);
    if (!matvec_only) for ([_]struct { []const u8, ?metal.Buffer }{ .{ "arena only", null }, .{ "+ mapped checkpoint", mapped }, .{ "+ Metal-allocated copy", aligned } }) |src| {
        for (0..reps + 3) |r| {
            for (0..20) |_| {
                if (src[1]) |b| try gpu.dispatch(ctx, "mk_d2_empty", &.{ arena, b }, &p, 32, 32) else try gpu.dispatch(ctx, "mk_d2_empty", &.{arena}, &p, 32, 32);
            }
            var t: [4]f64 = undefined;
            try gpu.finishWait(ctx, true, &t);
            if (r >= 3) s.put(r - 3, t);
        }
        try s.print(a, try std.fmt.allocPrint(a, "synchronous, 20 empty dispatches, {s}", .{src[0]}));
    };
    const shapes = [_][2]u32{ .{ 128000, 2048 }, .{ 7680, 2048 }, .{ 6144, 2048 }, .{ 2048, 7680 }, .{ 2048, 2048 } };
    for (shapes) |shape| {
        const m = shape[0];
        const n = shape[1];
        const code_bytes: u64 = @as(u64, m) * n / 2;
        const sb: u64 = @as(u64, m) * (n / 64) * 2;
        const total = code_bytes + 2 * sb;
        // Consecutive regions of the file (bf16 scales and biases, group 64), enough
        // of them that the sweep does not fit in the system-level cache.
        const count: usize = @intCast(@min(64, @max(1, (mapping.bytes.len - 4096) / total)));
        for ([_][2]u32{ .{ 1, 128 }, .{ 2, 64 }, .{ 2, 128 }, .{ 2, 256 }, .{ 4, 128 } }) |rt| {
            const kernel = try std.fmt.allocPrintSentinel(a, "mk_linear_q4_r{d}_t1", .{rt[0]}, 0);
            for ([_]struct { []const u8, metal.Buffer, u64 }{ .{ "mapped", mapped, 0 }, .{ "aligned", aligned, 2 } }) |src| {
                for (0..reps) |r| {
                    const before = ctx.stats();
                    for (0..count) |k| {
                        const base: u64 = 75002 + src[2] + @as(u64, k) * total;
                        const q: gpu.Params = .{ .rows = 1, .w = base, .s = base + code_bytes, .b = base + code_bytes + sb, .x = 0, .xs = n, .y = 8192, .ys = m, .n = n, .m = m, .bits = 4, .group = 64, .sdtype = 2, .bdtype = 2 };
                        try gpu.dispatch(ctx, kernel, &.{ src[1], src[1], src[1], out }, &q, (@as(usize, m) + rt[0] - 1) / rt[0] * 32, rt[1]);
                    }
                    try ctx.finish();
                    s.a[r] = (ctx.stats().gpu_seconds - before.gpu_seconds) / @as(f64, @floatFromInt(count)) * 1e6;
                }
                const us = median(s.a);
                try io.print(a, "  {d}x{d} r{d} tg{d} {s}: {d:>8.2} us/matvec, {d:>6.0} GB/s ({d} distinct regions)\n", .{ m, n, rt[0], rt[1], src[0], us, @as(f64, @floatFromInt(total)) / (us * 1e3), count });
            }
        }
    }
}
