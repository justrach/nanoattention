const std = @import("std");
const metal = @import("metal.zig");

// Parent integration: append kv_bf16.metal AFTER normalization_aligned.metal,
// attention_mlx.metal and attention_gqa_reuse.metal in Context.initMatched(),
// then import this test file. Production parity tests use that actual library,
// not a replacement math mode or test ABI. FP32 remains available/default.
// Headless, bounded synthetic allocations only; no UI, model, server or process.
// These gates establish storage-path parity on already-rounded input, NOT full
// model accuracy relative to FP32 KV. The parent owns that separate accuracy gate.
const dim: u32 = 128;
const kv_heads: u32 = 2;
const heads: u32 = kv_heads * 8;
const guard_words: usize = 16;
const guard_bits: u32 = 0x4b123456;
const guard_half: u16 = 0x5a6d;
const original_poison: u32 = 0x7fc12345;
const candidate_poison: u32 = 0x7fc54321;
const cache_poison: u16 = 0x7fc5;
const blocks_list = [_]u32{ 128, 512, 1024 };

// Independent CPU integer reference: compare the discarded field to halfway
// instead of implementing the GPU's rounding-bias addition.
fn roundBf16(value: f32) u16 {
    const bits: u32 = @bitCast(value);
    var upper: u16 = @truncate(bits >> 16);
    if ((bits & 0x7fffffff) > 0x7f800000) return upper | 0x0040;
    const lower: u16 = @truncate(bits);
    if (lower > 0x8000 or (lower == 0x8000 and (upper & 1) != 0)) upper +%= 1;
    return upper;
}

fn widenBf16(value: u16) f32 {
    return @bitCast(@as(u32, value) << 16);
}

// Positive/negative even/odd ties and their neighbors; signed zeros, tiny
// subnormal ties/round-up, normal boundary, max finite overflow, infinities,
// quiet/signaling NaNs including payloads entirely in the discarded bits.
const conversion_inputs = [_]u32{
    0x00000000, 0x80000000, 0x3f800000, 0xbf800000,
    0x3f807fff, 0x3f808000, 0x3f808001, 0x3f817fff,
    0x3f818000, 0x3f818001, 0xbf807fff, 0xbf808000,
    0xbf808001, 0xbf817fff, 0xbf818000, 0xbf818001,
    0x00000001, 0x00007fff, 0x00008000, 0x00008001,
    0x00018000, 0x80000001, 0x80008000, 0x80008001,
    0x80018000, 0x007f8000, 0x807f8000, 0x00800000,
    0x7f7f0000, 0x7f7f7fff, 0x7f7f8000, 0xff7f8000,
    0x7f800000, 0xff800000, 0x7f800001, 0xff800001,
    0x7fc12345, 0xffc12345, 0x7fffffff, 0xffffffff,
};
const conversion_expected = [_]u16{
    0x0000, 0x8000, 0x3f80, 0xbf80,
    0x3f80, 0x3f80, 0x3f81, 0x3f81,
    0x3f82, 0x3f82, 0xbf80, 0xbf80,
    0xbf81, 0xbf81, 0xbf82, 0xbf82,
    0x0000, 0x0000, 0x0000, 0x0001,
    0x0002, 0x8000, 0x8000, 0x8001,
    0x8002, 0x0080, 0x8080, 0x0080,
    0x7f7f, 0x7f7f, 0x7f80, 0xff80,
    0x7f80, 0xff80, 0x7fc0, 0xffc0,
    0x7fc1, 0xffc1, 0x7fff, 0xffff,
};

fn matchedContext() !metal.Context {
    const ctx = try metal.Context.initMatched(true);
    if (!ctx.isM3Ultra()) {
        ctx.deinit();
        return error.SkipZigTest;
    }
    return ctx;
}

fn poison(buffer: metal.Buffer, payload: usize, bits: u32) void {
    @memset(buffer.structs(u32)[0..payload], bits);
    @memset(buffer.structs(u32)[payload..], guard_bits);
}

fn guard(buffer: metal.Buffer, payload: usize) !void {
    try std.testing.expectEqual(payload + guard_words, buffer.structs(u32).len);
    for (buffer.structs(u32)[payload..]) |bits| try std.testing.expectEqual(guard_bits, bits);
}

// alloc() counts FP32 words; halve the cache payload explicitly, not just its
// logical index count. All test cache element counts are even.
fn allocBf16(ctx: metal.Context, elements: usize) !metal.Buffer {
    std.debug.assert(elements % 2 == 0);
    return ctx.alloc(elements / 2 + guard_words);
}

fn poisonBf16(buffer: metal.Buffer, elements: usize, bits: u16) void {
    @memset(buffer.structs(u16)[0..elements], bits);
    @memset(buffer.structs(u32)[elements / 2 ..], guard_bits);
}

fn equalWords(expected: metal.Buffer, actual: metal.Buffer, payload: usize, label: []const u8, p: metal.Params) !void {
    try guard(expected, payload);
    try guard(actual, payload);
    for (expected.structs(u32)[0..payload], actual.structs(u32)[0..payload], 0..) |a, b, i| {
        if (a != b or !std.math.isFinite(@as(f32, @bitCast(b)))) {
            std.debug.print("BF16 KV {s}: N={d} B={d} dtype={d} word={d} FP32=0x{x} BF16=0x{x}\n", .{ label, p.pos + 1, p.k, p.dtype, i, a, b });
            try std.testing.expectEqual(a, b);
            try std.testing.expect(std.math.isFinite(@as(f32, @bitCast(b))));
        }
    }
}

fn writeSink(sink: metal.Buffer, p: metal.Params, dominant: bool) void {
    @memset(sink.structs(u8), 0xff);
    const offset: usize = @intCast(p.offset);
    for (0..heads) |h| {
        const value: f32 = if (dominant) 12 + @as(f32, @floatFromInt(h)) * 0.125 else -2 - @as(f32, @floatFromInt(h)) * 0.0625;
        switch (p.dtype) {
            0 => sink.floats()[offset / 4 + h] = value,
            1 => sink.structs(u16)[offset / 2 + h] = @bitCast(@as(f16, @floatCast(value))),
            2 => sink.structs(u16)[offset / 2 + h] = roundBf16(value),
            else => unreachable,
        }
    }
}

const Case = struct {
    tokens: u32,
    split: bool,
    dtype: u32 = 0,
    offset: u64 = 12,
    dominant: bool = false,
};

fn runAttention(ctx: metal.Context, case: Case) !void {
    var p: metal.Params = .{
        .heads = heads,
        .kv_heads = kv_heads,
        .dim = dim,
        .pos = case.tokens - 1,
        .capacity = case.tokens + 7,
        .dtype = case.dtype,
        .offset = case.offset,
        .scale = if (case.dominant) 0.01 else 0.13,
    };
    const count: usize = heads * dim;
    const cache_count = @as(usize, p.capacity) * kv_heads * dim;
    const active_count = @as(usize, case.tokens) * kv_heads * dim;
    const sink_bytes: usize = if (p.dtype == 0) 4 else 2;
    const sink_words = (@as(usize, @intCast(p.offset)) + heads * sink_bytes + 3) / 4;
    const q = try ctx.alloc(count + guard_words);
    defer q.deinit();
    const keys = try ctx.alloc(cache_count + guard_words);
    defer keys.deinit();
    const values = try ctx.alloc(cache_count + guard_words);
    defer values.deinit();
    const bk = try allocBf16(ctx, cache_count);
    defer bk.deinit();
    const bv = try allocBf16(ctx, cache_count);
    defer bv.deinit();
    const sink = try ctx.alloc(sink_words + guard_words);
    defer sink.deinit();
    const original = try ctx.alloc(count + guard_words);
    defer original.deinit();
    const candidate = try ctx.alloc(count + guard_words);
    defer candidate.deinit();
    // Drain any encoded work before buffers are released, including failures.
    defer ctx.finish() catch {};
    poison(q, count, original_poison);
    poison(keys, cache_count, original_poison);
    poison(values, cache_count, original_poison);
    poisonBf16(bk, cache_count, cache_poison);
    poisonBf16(bv, cache_count, cache_poison);
    writeSink(sink, p, case.dominant);
    @memset(sink.structs(u32)[sink_words..], guard_bits);
    const sink_snapshot = try std.testing.allocator.dupe(u32, sink.structs(u32));
    defer std.testing.allocator.free(sink_snapshot);
    for (q.floats()[0..count], 0..) |*v, i|
        v.* = @as(f32, @floatFromInt((i * 19 + i / dim * 7) % 257)) / 129 - 1;
    for (0..case.tokens) |t| {
        for (0..kv_heads) |h| {
            for (0..dim) |d| {
                const i = (t * kv_heads + h) * dim + d;
                // Deliberately NOT already BF16-exact binary fractions. Both
                // cache versions derive from the same independently rounded bits.
                const k = @as(f32, @floatFromInt((t * 17 + t / 31 * 13 + h * 43 + d * 7) % 257)) / 131 - 1;
                const v = @as(f32, @floatFromInt((t * 29 + t / 17 * 11 + h * 19 + d * 37) % 257)) / 259 - 0.25;
                bk.structs(u16)[i] = roundBf16(k);
                bv.structs(u16)[i] = roundBf16(v);
                keys.floats()[i] = widenBf16(bk.structs(u16)[i]);
                values.floats()[i] = widenBf16(bv.structs(u16)[i]);
            }
        }
    }
    if (!case.split) {
        poison(original, count, original_poison);
        poison(candidate, count, candidate_poison);
        try ctx.dispatch("attention_mlx_single", &.{ q, keys, values, sink, original }, p, @as(usize, heads) * 1024, 1024);
        try ctx.dispatch("attention_mlx_single_bf16", &.{ q, bk, bv, sink, candidate }, p, @as(usize, heads) * 1024, 1024);
        try ctx.finish();
        try equalWords(original, candidate, count, "single output", p);
    } else {
        for (blocks_list) |blocks| {
            p.k = blocks;
            const state_count = @as(usize, heads) * blocks * 130;
            const os = try ctx.alloc(state_count + guard_words);
            defer os.deinit();
            const bs = try ctx.alloc(state_count + guard_words);
            defer bs.deinit();
            defer ctx.finish() catch {};
            poison(os, state_count, original_poison);
            poison(original, count, original_poison);
            const groups = @as(usize, kv_heads) * blocks;
            try ctx.dispatch("attention_mlx_split", &.{ q, keys, values, sink, os }, p, groups * 256, 256);
            try ctx.dispatch("attention_mlx_merge", &.{ os, original }, p, @as(usize, heads) * 1024, 1024);
            try ctx.finish();
            // Both candidates use the SAME independently CPU-rounded FP32
            // baseline, never the scalar BF16 candidate as the vector oracle.
            // Cache allocations are Metal-aligned with zero buffer offsets;
            // each D128 token/head stride and lane*4 ushort offset is divisible
            // by four, so every vector load is 8-byte aligned and in bounds.
            // Nonzero sink p.offset (including 6/10 bytes) is independent of
            // the cache offsets and need not be aligned to eight bytes.
            for ([_][:0]const u8{
                "attention_mlx_split_gqa2_bf16",
                "attention_mlx_split_gqa2_bf16_vector",
            }) |entrypoint| {
                poison(bs, state_count, candidate_poison);
                poison(candidate, count, candidate_poison);
                try ctx.dispatch(entrypoint, &.{ q, bk, bv, sink, bs }, p, groups * 128, 128);
                try ctx.dispatch("attention_mlx_merge", &.{ bs, candidate }, p, @as(usize, heads) * 1024, 1024);
                try ctx.finish();
                // All raw_o/max/sum and merged output words, compared as u32;
                // entrypoint identifies either state or output parity failures.
                try equalWords(os, bs, state_count, entrypoint, p);
                try equalWords(original, candidate, count, entrypoint, p);
            }
        }
    }
    try guard(q, count);
    try guard(keys, cache_count);
    try guard(values, cache_count);
    try guard(bk, cache_count / 2);
    try guard(bv, cache_count / 2);
    try guard(sink, sink_words);
    try std.testing.expectEqualSlices(u32, sink_snapshot, sink.structs(u32));
    for (keys.structs(u32)[active_count..cache_count]) |bits| try std.testing.expectEqual(original_poison, bits);
    for (values.structs(u32)[active_count..cache_count]) |bits| try std.testing.expectEqual(original_poison, bits);
    for (bk.structs(u16)[active_count..cache_count]) |bits| try std.testing.expectEqual(cache_poison, bits);
    for (bv.structs(u16)[active_count..cache_count]) |bits| try std.testing.expectEqual(cache_poison, bits);
    // Read-only inputs must still encode the same cache at every active index.
    for (0..active_count) |i| {
        try std.testing.expectEqual(@as(u32, @bitCast(widenBf16(bk.structs(u16)[i]))), keys.structs(u32)[i]);
        try std.testing.expectEqual(@as(u32, @bitCast(widenBf16(bv.structs(u16)[i]))), values.structs(u32)[i]);
    }
}

test "BF16 KV single every output bit agrees with CPU-rounded FP32 cache at N1/N513 for each sink dtype" {
    const ctx = try matchedContext();
    defer ctx.deinit();
    for ([_]u32{ 0, 1, 2 }) |dtype| {
        for ([_]u32{ 1, 513 }) |tokens| try runAttention(ctx, .{
            .tokens = tokens,
            .split = false,
            .dtype = dtype,
            .offset = if (dtype == 0) 12 else if (dtype == 1) 6 else 10,
            .dominant = tokens == 1,
        });
    }
}

test "BF16 KV scalar/vector GQA2 every producer and original merge bit at N1/N1024/N1025/N16384/N32768 and B128/B512/B1024" {
    const ctx = try matchedContext();
    defer ctx.deinit();
    for ([_]Case{
        .{ .tokens = 1, .split = true, .dtype = 1, .offset = 6, .dominant = true }, // Empty stripes must overwrite poison.
        .{ .tokens = 1024, .split = true, .dominant = true },
        .{ .tokens = 1025, .split = true, .dtype = 2, .offset = 10 }, // TK4 plus scalar tails.
        .{ .tokens = 16384, .split = true, .dtype = 1, .offset = 6 },
        .{ .tokens = 32768, .split = true, .dtype = 2, .offset = 10 },
    }) |case| try runAttention(ctx, case);
}

// A test-only entry point calls the EXACT helpers from kv_bf16.metal. It is
// appended to the same original production definitions/math settings, never
// copied into a production dispatch path. No native half/BF16 conversions.
extern fn fm_open([*:0]const u8) ?*anyopaque;
fn conversionContext() !metal.Context {
    const source = "#define FLINT_MLX_MATH 1\n" ++
        @embedFile("kernels.metal") ++ "\n" ++ @embedFile("resident_plan.metal") ++ "\n" ++
        @embedFile("normalization.metal") ++ "\n" ++ @embedFile("normalization_aligned.metal") ++ "\n" ++
        @embedFile("attention_mlx.metal") ++ "\n" ++ @embedFile("attention_gqa_reuse.metal") ++ "\n" ++
        @embedFile("kv_bf16.metal") ++ "\n" ++
        \\kernel void kv_bf16_conversion_probe(device const float *input [[buffer(0)]],
        \\                                      device ushort *encoded [[buffer(1)]],
        \\                                      device float *wide [[buffer(2)]],
        \\                                      constant P &p [[buffer(8)]],
        \\                                      uint i [[thread_position_in_grid]]) {
        \\    if (i >= p.n) return;
        \\    ushort b = kv_bf16_from_float(input[i]);
        \\    encoded[i] = b;
        \\    wide[i] = kv_bf16_to_float(ushort(as_type<uint>(input[i]) >> 16));
        \\}
    ;
    const ctx: metal.Context = .{ .handle = fm_open(source) orelse return error.MetalFailure };
    if (!ctx.isM3Ultra()) {
        ctx.deinit();
        return error.SkipZigTest;
    }
    return ctx;
}

test "BF16 RNE explicit ties signed specials and all 65536 widening patterns" {
    // CPU assertions use hard-coded expected answers, not only two copies of
    // the implementation. GPU round/store and widen must preserve exact bits.
    for (conversion_inputs, conversion_expected) |bits, wanted|
        try std.testing.expectEqual(wanted, roundBf16(@bitCast(bits)));
    const ctx = try conversionContext();
    defer ctx.deinit();
    const n: usize = 65536 + conversion_inputs.len;
    const input = try ctx.alloc(n + guard_words);
    defer input.deinit();
    const encoded = try allocBf16(ctx, n);
    defer encoded.deinit();
    const wide = try ctx.alloc(n + guard_words);
    defer wide.deinit();
    defer ctx.finish() catch {};
    poison(input, n, original_poison);
    poisonBf16(encoded, n, guard_half);
    poison(wide, n, candidate_poison);
    @memcpy(input.structs(u32)[0..conversion_inputs.len], &conversion_inputs);
    for (0..65536) |i| input.structs(u32)[conversion_inputs.len + i] = @as(u32, @intCast(i)) << 16;
    try ctx.dispatch("kv_bf16_conversion_probe", &.{ input, encoded, wide }, .{ .n = @intCast(n) }, ((n + 127) / 128) * 128, 128);
    try ctx.finish();
    for (input.structs(u32)[0..n], 0..) |bits, i| {
        const wanted = if (i < conversion_expected.len) conversion_expected[i] else roundBf16(@bitCast(bits));
        try std.testing.expectEqual(wanted, encoded.structs(u16)[i]);
        // Widen the RAW upper bits, independently of rounding/NaN quieting,
        // so all 65536 patterns (including signaling NaNs) are exercised.
        try std.testing.expectEqual(bits & 0xffff0000, wide.structs(u32)[i]);
    }
    try guard(input, n);
    try guard(encoded, n / 2);
    try guard(wide, n);
}

fn normWeights(buffer: metal.Buffer, offset: usize, dtype: u32) void {
    @memset(buffer.structs(u8), 0xa5);
    for (0..dim) |i| {
        const value: f32 = @as(f32, @floatFromInt(i % 13 + 1)) / 17;
        switch (dtype) {
            0 => buffer.floats()[offset / 4 + i] = value,
            1 => buffer.structs(u16)[offset / 2 + i] = @bitCast(@as(f16, @floatCast(value))),
            2 => buffer.structs(u16)[offset / 2 + i] = roundBf16(value),
            else => unreachable,
        }
    }
}

test "BF16 aligned QK leaves Q/K bits unchanged and rounds only cache stores after norm and RoPE" {
    const ctx = try matchedContext();
    defer ctx.deinit();
    const qn: usize = heads * dim;
    const kn: usize = kv_heads * dim;
    const capacity: u32 = 519;
    const cn: usize = capacity * kn;
    const q = try ctx.alloc(qn + guard_words);
    defer q.deinit();
    const k = try ctx.alloc(kn + guard_words);
    defer k.deinit();
    const bq = try ctx.alloc(qn + guard_words);
    defer bq.deinit();
    const bk = try ctx.alloc(kn + guard_words);
    defer bk.deinit();
    const v = try ctx.alloc(kn + guard_words);
    defer v.deinit();
    const qw = try ctx.alloc(dim + guard_words);
    defer qw.deinit();
    const kw = try ctx.alloc(dim + guard_words);
    defer kw.deinit();
    const keys = try ctx.alloc(cn + guard_words);
    defer keys.deinit();
    const values = try ctx.alloc(cn + guard_words);
    defer values.deinit();
    const bkeys = try allocBf16(ctx, cn);
    defer bkeys.deinit();
    const bvalues = try allocBf16(ctx, cn);
    defer bvalues.deinit();
    defer ctx.finish() catch {};
    for ([_]u32{ 0, 1, 2 }) |dtype| {
        const kdtype = (dtype + 1) % 3;
        const qo: usize = if (dtype == 0) 12 else 6;
        const ko: usize = if (kdtype == 0) 8 else 10;
        normWeights(qw, qo, dtype);
        normWeights(kw, ko, kdtype);
        const qw_snapshot = try std.testing.allocator.dupe(u32, qw.structs(u32));
        defer std.testing.allocator.free(qw_snapshot);
        const kw_snapshot = try std.testing.allocator.dupe(u32, kw.structs(u32));
        defer std.testing.allocator.free(kw_snapshot);
        for ([_]bool{ false, true }) |rope| {
            for ([_]u32{ 0, 1, 513 }) |pos| {
                const p: metal.Params = .{
                    .heads = heads,
                    .kv_heads = kv_heads,
                    .dim = dim,
                    .capacity = capacity,
                    .pos = pos,
                    .bits = @intFromBool(rope),
                    .dtype = dtype,
                    .offset = qo,
                    .scale_dtype = kdtype,
                    .scale_offset = ko,
                    .theta = @log2(@as(f32, 10000)),
                };
                poison(q, qn, original_poison);
                poison(k, kn, original_poison);
                poison(bq, qn, candidate_poison);
                poison(bk, kn, candidate_poison);
                poison(v, kn, original_poison);
                poison(keys, cn, original_poison);
                poison(values, cn, original_poison);
                poisonBf16(bkeys, cn, guard_half);
                poisonBf16(bvalues, cn, guard_half);
                for (q.floats()[0..qn], 0..) |*x, i|
                    x.* = @as(f32, @floatFromInt((i * 7 + pos + 3) % 31)) / 29 - 0.75;
                for (k.floats()[0..kn], 0..) |*x, i|
                    x.* = @as(f32, @floatFromInt((i * 11 + pos + 9) % 37)) / 39 - 0.5;
                @memcpy(bq.structs(u32)[0..qn], q.structs(u32)[0..qn]);
                @memcpy(bk.structs(u32)[0..kn], k.structs(u32)[0..kn]);
                // V bypasses normalization and RoPE: include ties and all special
                // categories to verify the actual cache-store conversion too.
                for (v.structs(u32)[0..kn], 0..) |*bits, i| bits.* = conversion_inputs[i % conversion_inputs.len];
                const group: usize = if (dtype == 1) 128 else 32;
                // 24 SIMDs for 18 real Q/K heads exercises padded head tails.
                try ctx.dispatch("qk_rope_cache_aligned", &.{ q, k, v, qw, kw, keys, values }, p, 24 * 32, group);
                try ctx.dispatch("qk_rope_cache_aligned_bf16", &.{ bq, bk, v, qw, kw, bkeys, bvalues }, p, 24 * 32, group);
                try ctx.finish();
                try equalWords(q, bq, qn, "QK query output", p);
                try equalWords(k, bk, kn, "QK key output", p);
                const slot = @as(usize, pos) * kn;
                for (0..cn) |i| {
                    if (i >= slot and i < slot + kn) {
                        try std.testing.expectEqual(roundBf16(keys.floats()[i]), bkeys.structs(u16)[i]);
                        try std.testing.expectEqual(roundBf16(values.floats()[i]), bvalues.structs(u16)[i]);
                        try std.testing.expectEqual(k.structs(u32)[i - slot], keys.structs(u32)[i]);
                        try std.testing.expectEqual(v.structs(u32)[i - slot], values.structs(u32)[i]);
                    } else {
                        try std.testing.expectEqual(original_poison, keys.structs(u32)[i]);
                        try std.testing.expectEqual(original_poison, values.structs(u32)[i]);
                        try std.testing.expectEqual(guard_half, bkeys.structs(u16)[i]);
                        try std.testing.expectEqual(guard_half, bvalues.structs(u16)[i]);
                    }
                }
                for (v.structs(u32)[0..kn], 0..) |bits, i| try std.testing.expectEqual(conversion_inputs[i % conversion_inputs.len], bits);
                try std.testing.expectEqualSlices(u32, qw_snapshot, qw.structs(u32));
                try std.testing.expectEqualSlices(u32, kw_snapshot, kw.structs(u32));
                try guard(v, kn);
                try guard(keys, cn);
                try guard(values, cn);
                try guard(bkeys, cn / 2);
                try guard(bvalues, cn / 2);
            }
        }
    }
}

const Rejection = enum {
    dim,
    zero_heads,
    zero_kv,
    nonintegral_gqa,
    other_gqa,
    sliding,
    zero_capacity,
    nonresident,
    too_long,
    overflow_pos,
    dtype,
    sink_alignment,
    blocks,
    group,
    norm_dtype,
    norm_alignment,
};

test "BF16 unsupported attention and QK parameters reject without output or cache writes" {
    const ctx = try matchedContext();
    defer ctx.deinit();
    const qn: usize = heads * dim;
    const kn: usize = kv_heads * dim;
    const sn: usize = heads * 128 * 130;
    const q = try ctx.alloc(qn + guard_words);
    defer q.deinit();
    const k = try ctx.alloc(kn + guard_words);
    defer k.deinit();
    const v = try ctx.alloc(kn + guard_words);
    defer v.deinit();
    const w = try ctx.alloc(heads + dim + guard_words);
    defer w.deinit();
    const keys = try allocBf16(ctx, kn);
    defer keys.deinit();
    const values = try allocBf16(ctx, kn);
    defer values.deinit();
    const out = try ctx.alloc(sn + guard_words);
    defer out.deinit();
    defer ctx.finish() catch {};
    for (std.enums.values(Rejection)) |reason| {
        var p: metal.Params = .{ .heads = heads, .kv_heads = kv_heads, .dim = dim, .capacity = 1, .k = 128 };
        switch (reason) {
            .dim => p.dim = 64,
            .zero_heads => p.heads = 0,
            .zero_kv => p.kv_heads = 0,
            .nonintegral_gqa => p.heads = heads - 1,
            .other_gqa => p.heads = kv_heads * 4,
            .sliding => p.window = 1,
            .zero_capacity => p.capacity = 0,
            .nonresident => p.pos = 1,
            .too_long => {
                p.pos = 262144;
                p.capacity = 262145;
            },
            .overflow_pos => p.pos = std.math.maxInt(u32),
            .dtype => p.dtype = 3,
            .sink_alignment => p.offset = 1,
            .blocks => p.k = 64,
            .group => {},
            .norm_dtype => p.scale_dtype = 3,
            .norm_alignment => p.scale_offset = 1,
        }
        // Norm ABI only applies to the store; B/group applies only to attention.
        for ([_][:0]const u8{ "attention_mlx_single_bf16", "attention_mlx_split_gqa2_bf16", "qk_rope_cache_aligned_bf16" }, 0..) |kernel, variant| {
            if (variant != 2 and (reason == .norm_dtype or reason == .norm_alignment)) continue;
            if (variant != 1 and reason == .blocks) continue;
            if (variant == 2 and reason == .group) continue; // QK accepts any complete-SIMD group.
            poison(q, qn, original_poison);
            poison(k, kn, original_poison);
            poison(v, kn, original_poison);
            poison(w, heads + dim, original_poison);
            poisonBf16(keys, kn, guard_half);
            poisonBf16(values, kn, guard_half);
            poison(out, sn, candidate_poison);
            var group: usize = if (variant == 0) 1024 else if (variant == 1) 128 else 32;
            if (reason == .group) group /= 2;
            if (variant == 2) {
                try ctx.dispatch(kernel, &.{ q, k, v, w, w, keys, values }, p, 24 * 32, group);
            } else {
                const groups: usize = if (variant == 0) heads else kv_heads * 128;
                try ctx.dispatch(kernel, &.{ q, keys, values, w, out }, p, groups * group, group);
            }
            try ctx.finish();
            // Never merge a rejected producer; rejection is not host fallback.
            for (out.structs(u32)[0..sn]) |bits| try std.testing.expectEqual(candidate_poison, bits);
            for (q.structs(u32)[0..qn]) |bits| try std.testing.expectEqual(original_poison, bits);
            for (k.structs(u32)[0..kn]) |bits| try std.testing.expectEqual(original_poison, bits);
            for (v.structs(u32)[0..kn]) |bits| try std.testing.expectEqual(original_poison, bits);
            for (keys.structs(u16)[0..kn]) |bits| try std.testing.expectEqual(guard_half, bits);
            for (values.structs(u16)[0..kn]) |bits| try std.testing.expectEqual(guard_half, bits);
            try guard(out, sn);
            try guard(q, qn);
            try guard(k, kn);
            try guard(v, kn);
            try guard(w, heads + dim);
            try guard(keys, kn / 2);
            try guard(values, kn / 2);
        }
    }
}
