//! Teacher-forced benchmark (docs/QD.md quality stage): an actual prompt of N tokens
//! is prefilled in batches, then D decode tokens are forwarded one at a time
//! with the GPU argmax read back after each. One untimed warm-up run, then R
//! timed repeats from a reset state; medians reported. Loading is excluded.
const std = @import("std");
const exec = @import("exec.zig");
const io = @import("io.zig");
const Knobs = @import("knobs.zig").Knobs;

pub const Input = struct {
    prompt_tokens: u32 = 64,
    decode_tokens: u32 = 256,
    repeats: u32 = 3,
    /// Optional workload file {"bos": id, "code": [...], "decode": [...]}: the
    /// prompt is bos then code repeated and cut to prompt_tokens (the stock-MLX
    /// baseline's workload, tools/nano_bench.py tokens); decode is teacher-forced.
    tokens: ?[]const u8 = null,
    /// Decode objective: start the bridge's per-dispatch profiler (FLINT_PROFILE
    /// report, printed at exit) before the last repeat's decode, so the report
    /// covers decode forwards only. Attribution only; it perturbs timing.
    profile_decode: bool = false,
    /// decode climb 2: print, on stderr, the last repeat's decode time per token
    /// split into host encoding, finish-to-completion wait, command GPU time and
    /// the rest (read-back and loop), as one JSON line.
    decode_detail: bool = false,
};

const Workload = struct { bos: u32, code: []const u32, decode: []const u32 };

fn workload(a: std.mem.Allocator, input: Input, vocab: u32) !struct { prompt: []u32, decode: []u32 } {
    const n = input.prompt_tokens;
    const prompt = try a.alloc(u32, n);
    const decode = try a.alloc(u32, input.decode_tokens);
    if (input.tokens) |bytes| {
        const w = try std.json.parseFromSliceLeaky(Workload, a, bytes, .{ .ignore_unknown_fields = true });
        if (w.code.len == 0 or w.decode.len < decode.len) return error.InvalidBenchmarkTokens;
        prompt[0] = w.bos;
        for (prompt[1..], 0..) |*t, i| t.* = w.code[i % w.code.len];
        @memcpy(decode, w.decode[0..decode.len]);
    } else {
        const s = try stream(a, @as(usize, n) + decode.len, vocab);
        @memcpy(prompt, s[0..n]);
        @memcpy(decode, s[n..]);
    }
    for (prompt) |t| if (t >= vocab) return error.InvalidToken;
    for (decode) |t| if (t >= vocab) return error.InvalidToken;
    return .{ .prompt = prompt, .decode = decode };
}

/// Deterministic token stream inside the vocabulary (seeded LCG), so every
/// candidate sees the same prompt and teacher-forced continuation.
pub fn stream(a: std.mem.Allocator, n: usize, vocab: u32) ![]u32 {
    const out = try a.alloc(u32, n);
    var state: u64 = 0x9e3779b97f4a7c15;
    const span = @min(vocab, 100_000);
    for (out) |*t| {
        state = state *% 6364136223846793005 +% 1442695040888963407;
        t.* = @intCast((state >> 33) % span);
    }
    return out;
}

fn median(values: []f64) f64 {
    std.mem.sort(f64, values, {}, std.sort.asc(f64));
    return values[values.len / 2];
}

pub fn run(engine: *exec.Engine, a: std.mem.Allocator, input: Input, knobs: Knobs) !void {
    if (input.repeats == 0 or input.prompt_tokens == 0) return error.InvalidBenchmark;
    const total = @as(u64, input.prompt_tokens) + input.decode_tokens;
    if (total > engine.context) return error.ContextExhausted;
    const w = try workload(a, input, engine.program.meta.vocab);
    const prompt = w.prompt;
    const decode = w.decode;
    const prefill_s = try a.alloc(f64, input.repeats);
    const decode_s = try a.alloc(f64, input.repeats);
    var prefill_dispatches: u64 = 0;
    var decode_dispatches: u64 = 0;
    var prefill_forwards: u64 = 0;
    var checksum: u64 = 0;
    for (0..input.repeats + 1) |rep| {
        engine.reset();
        prefill_dispatches = 0;
        prefill_forwards = 0;
        decode_dispatches = 0;
        const t0 = try io.now();
        var i: usize = 0;
        while (i < prompt.len) {
            const n = @min(engine.batch, prompt.len - i);
            const last = i + n == prompt.len;
            try engine.forward(prompt[i..][0..n], if (last) @intCast(n - 1) else @intCast(n));
            prefill_dispatches += engine.last.dispatches;
            prefill_forwards += 1;
            i += n;
        }
        var sum: u64 = try engine.next(engine.out_end - 1);
        if (input.profile_decode and rep == input.repeats) try engine.ctx.beginProfile();
        const t1 = try io.now();
        var detail: [3]f64 = @splat(0); // decode climb 2: encode, wait, GPU seconds
        for (decode) |t| {
            try engine.forward(&.{t}, 0);
            decode_dispatches += engine.last.dispatches;
            detail[0] += engine.last.encode_seconds;
            detail[1] += engine.last.wait_seconds;
            detail[2] += engine.last.gpu_seconds;
            sum +%= try engine.next(0);
        }
        const t2 = try io.now();
        try engine.d2Settle(); // decode climb 2 (d2_gate): nothing left at the gate before the next run
        if (input.decode_detail and rep == input.repeats and decode.len > 0) {
            const n: f64 = @floatFromInt(decode.len);
            const us = 1e6 / n;
            std.debug.print("{{\"type\":\"nano_decode_detail\",\"prompt_tokens\":{d},\"wall_us\":{d:.2},\"encode_us\":{d:.2},\"wait_us\":{d:.2},\"gpu_us\":{d:.2},\"other_us\":{d:.2},\"gate_redos\":{d},\"dispatches\":{d:.1}}}\n", .{ input.prompt_tokens, (t2 - t1) * us, detail[0] * us, detail[1] * us, detail[2] * us, (t2 - t1 - detail[0] - detail[1]) * us, engine.d2_redos, @as(f64, @floatFromInt(decode_dispatches)) / n });
        }
        if (rep == 0) {
            checksum = sum;
            continue;
        }
        if (sum != checksum) return error.NondeterministicBenchmark;
        prefill_s[rep - 1] = t1 - t0;
        decode_s[rep - 1] = t2 - t1;
    }
    const prefill_copy = try a.dupe(f64, prefill_s);
    const decode_copy = try a.dupe(f64, decode_s);
    const pm = median(prefill_copy);
    const dm = median(decode_copy);
    const result = .{
        .type = "nano_bench",
        .version = 1,
        .model_type = engine.program.meta.model_type,
        .prompt_tokens = input.prompt_tokens,
        .decode_tokens = input.decode_tokens,
        .repeats = input.repeats,
        .warmup_runs = 1,
        .workload = if (input.tokens != null) "bos + code repeated (bench tokens file); teacher-forced decode" else "seeded LCG token stream",
        .batch = engine.batch,
        .knobs = knobs,
        .prefill_seconds = prefill_s,
        .decode_seconds = decode_s,
        .prefill_tokens_per_second = @as(f64, @floatFromInt(input.prompt_tokens)) / pm,
        .decode_tokens_per_second = if (input.decode_tokens == 0) 0 else @as(f64, @floatFromInt(input.decode_tokens)) / dm,
        .prefill_forwards = prefill_forwards,
        .dispatches_per_prefill_forward = @as(f64, @floatFromInt(prefill_dispatches)) / @as(f64, @floatFromInt(prefill_forwards)),
        .dispatches_per_decode_token = if (input.decode_tokens == 0) 0 else @as(f64, @floatFromInt(decode_dispatches)) / @as(f64, @floatFromInt(input.decode_tokens)),
        .prediction_checksum = checksum,
    };
    try io.out(try std.json.Stringify.valueAlloc(a, result, .{}));
    try io.out("\n");
}
