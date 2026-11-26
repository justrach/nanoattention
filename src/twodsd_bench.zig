//! CPU benchmark: 2DSD vs plain AR on the tiny exact looped target.
//! No Metal. Reports calls (loop-steps) and wall time. Does not claim GPU speedups.
const std = @import("std");
const twodsd = @import("twodsd.zig");

const Timespec = extern struct { sec: i64, nsec: i64 };
extern "c" fn write(c_int, [*]const u8, usize) isize;
extern "c" fn clock_gettime(c_int, *Timespec) c_int;

fn nowNs() u64 {
    var ts: Timespec = undefined;
    if (clock_gettime(1, &ts) != 0) return 0; // CLOCK_MONOTONIC
    return @as(u64, @intCast(ts.sec)) * 1_000_000_000 + @as(u64, @intCast(ts.nsec));
}

fn writeAll(bytes: []const u8) !void {
    var pos: usize = 0;
    while (pos < bytes.len) {
        const n = write(1, bytes[pos..].ptr, bytes.len - pos);
        if (n <= 0) return error.OutputFailed;
        pos += @intCast(n);
    }
}

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;
    const a = init.arena.allocator();

    var tokens: u32 = 64;
    var prompt_len: u32 = 8;
    var k: u32 = 4;
    var stage: twodsd.Stage = .stage2b;
    var repeats: u32 = 5;
    var causal = true;
    var args = std.process.Args.Iterator.init(init.minimal.args);
    _ = args.skip();
    while (args.next()) |flag| {
        const value = args.next() orelse return error.MissingArgumentValue;
        if (std.mem.eql(u8, flag, "--tokens")) {
            tokens = try std.fmt.parseInt(u32, value, 10);
        } else if (std.mem.eql(u8, flag, "--prompt")) {
            prompt_len = try std.fmt.parseInt(u32, value, 10);
        } else if (std.mem.eql(u8, flag, "--k")) {
            k = try std.fmt.parseInt(u32, value, 10);
        } else if (std.mem.eql(u8, flag, "--stage")) {
            stage = twodsd.parseStage(value) orelse return error.InvalidStage;
        } else if (std.mem.eql(u8, flag, "--repeats")) {
            repeats = try std.fmt.parseInt(u32, value, 10);
        } else if (std.mem.eql(u8, flag, "--causal")) {
            causal = (try std.fmt.parseInt(u1, value, 10)) != 0;
        } else return error.UnknownFlag;
    }
    if (tokens < 1 or prompt_len < 1 or repeats < 1) return error.InvalidBenchmark;

    const prompt = try a.alloc(u32, prompt_len);
    for (prompt, 0..) |*t, i| t.* = @intCast(3 + i % 17);

    var ar_buf = try a.alloc(u32, tokens);
    const ar_n = twodsd.arGreedy(prompt, tokens, causal, &.{}, ar_buf);
    const ar_out = ar_buf[0..ar_n];

    // Sequential AR cost: R loops per decode token after the prompt, plus R * prompt for prefill.
    const ar_decode_calls = twodsd.tiny_R * (tokens -| 1);

    var twodsd_calls: u64 = 0;
    var twodsd_ns: u64 = 0;
    var last_stats: twodsd.Stats = .{};
    for (0..repeats) |_| {
        var runner = twodsd.TinyRunner.init(gpa, 8, causal, .growing);
        defer runner.deinit();
        var ex = try twodsd.Executor.init(gpa, &runner, .{
            .k = k,
            .stage = stage,
            .d1 = 1,
            .d2 = 2,
            .drafter = if (k > 1) .oracle else .none,
        });
        defer ex.deinit();
        const t0 = nowNs();
        const result = try ex.generate(a, prompt, tokens, &.{});
        const t1 = nowNs();
        if (!std.mem.eql(u32, result.output, ar_out)) return error.OutputMismatch;
        twodsd_calls += result.stats.calls;
        twodsd_ns += t1 - t0;
        last_stats = result.stats;
    }
    const mean_calls = @as(f64, @floatFromInt(twodsd_calls)) / @as(f64, @floatFromInt(repeats));
    const mean_ns = @as(f64, @floatFromInt(twodsd_ns)) / @as(f64, @floatFromInt(repeats));
    const ar_tok_s = if (mean_ns > 0)
        @as(f64, @floatFromInt(tokens -| 1)) / (mean_ns / 1e9)
    else
        0.0;
    const call_speedup = @as(f64, @floatFromInt(ar_decode_calls)) / mean_calls;
    const kv_full = twodsd.historyKvBytes(12, 4, prompt_len + tokens, 2048, 4, .full, 0);
    const kv_shared = twodsd.historyKvBytes(12, 4, prompt_len + tokens, 2048, 4, .shared_last, 64);
    const kv_long_full = twodsd.inklingLoopKvBytes(4096, 2048, 4, twodsd.inkling_R, .full, 0);
    const kv_long_shared = twodsd.inklingLoopKvBytes(4096, 2048, 4, twodsd.inkling_R, .shared_last, 64);

    const json = try std.fmt.allocPrint(a,
        \\{{"host":"cpu","metal":false,"causal":{s},"prompt":{d},"tokens":{d},"k":{d},"stage":"{s}","repeats":{d},"ar_decode_loop_calls":{d},"twodsd_mean_calls":{d:.3},"call_speedup_vs_ar":{d:.3},"twodsd_mean_decode_ns":{d:.0},"twodsd_decode_tok_s":{d:.1},"commit_sources":{{"main":{d},"frozen":{d},"fork":{d},"bonus":{d}}},"kv_bytes":{{"full":{d},"shared_last_w64":{d},"ratio":{d:.3}}},"kv_bytes_inkling_seq4096":{{"full":{d},"shared_last_w64":{d},"ratio":{d:.3}}},"note":"Tiny exact looped CPU model (R=4, vocab=97). Call speedup counts loop-steps, not Metal. Oracle drafter is the exact continuation; real drafters accept less."}}
        \\
    , .{
        if (causal) "true" else "false",
        prompt_len,
        tokens,
        k,
        @tagName(stage),
        repeats,
        ar_decode_calls,
        mean_calls,
        call_speedup,
        mean_ns,
        ar_tok_s,
        last_stats.commit_sources.main,
        last_stats.commit_sources.frozen,
        last_stats.commit_sources.fork,
        last_stats.commit_sources.bonus,
        kv_full,
        kv_shared,
        @as(f64, @floatFromInt(kv_shared)) / @as(f64, @floatFromInt(kv_full)),
        kv_long_full,
        kv_long_shared,
        @as(f64, @floatFromInt(kv_long_shared)) / @as(f64, @floatFromInt(kv_long_full)),
    });
    try writeAll(json);
}
