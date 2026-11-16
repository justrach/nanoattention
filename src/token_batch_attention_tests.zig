//! Standalone, headless parity gates for token_batch_attention.metal.
//! Import this file from the test root; its Context compiles the new source
//! independently of production imports. No server, UI, or private paths.
const std = @import("std");
const metal = @import("metal.zig");
extern fn fm_open([*:0]const u8) ?*anyopaque;
extern fn fm_error() [*:0]const u8;

// Independent mirror: importing token_batch_tests.zig would also collect its
// unrelated GPU tests. No production ABI or existing file is changed here.
const TokenScratch = extern struct {
    x: u64 = 0,
    norm: u64 = 0,
    q: u64 = 0,
    k: u64 = 0,
    v: u64 = 0,
    attention: u64 = 0,
    projection: u64 = 0,
    activation: u64 = 0,
    router: u64 = 0,
    route_ids: u64 = 0,
    route_weights: u64 = 0,
    route_error: u64 = 0,
    logits: u64 = 0,
};
comptime {
    if (@sizeOf(TokenScratch) != 104 or @alignOf(TokenScratch) != 8 or
        @offsetOf(TokenScratch, "q") != 16 or @offsetOf(TokenScratch, "attention") != 40)
        @compileError("TokenScratch GPU-address ABI mismatch");
}
const max_batch = 32;
const heads: u32 = 32;
const kv_heads: u32 = 4;
const dim: u32 = 128;
const query_words: usize = heads * dim;
const guards: usize = 16;
const guard: u32 = 0x4b123456;
const reference_poison: u32 = 0x7fc12345;
const batch_poison: u32 = 0x7fc54321;
const bf16_poison: u16 = 0x7fc1;
const blocks_list = [_]u32{ 128, 512, 1024 };
const Variant = struct { kernel: [:0]const u8, queries: u32 };
const variants = [_]Variant{
    .{ .kernel = "token_batch_attention_split", .queries = 2 },
    .{ .kernel = "token_batch_attention_split_q4", .queries = 4 },
    .{ .kernel = "token_batch_attention_split_q8", .queries = 8 },
};

fn context() !metal.Context {
    const source = "#define FLINT_MLX_MATH 1\n" ++
        @embedFile("kernels.metal") ++ "\n" ++
        @embedFile("resident_plan.metal") ++ "\n" ++
        @embedFile("row_tiling.metal") ++ "\n" ++
        @embedFile("normalization.metal") ++ "\n" ++
        @embedFile("normalization_aligned.metal") ++ "\n" ++
        @embedFile("attention_mlx.metal") ++ "\n" ++
        @embedFile("attention_gqa_reuse.metal") ++ "\n" ++
        @embedFile("kv_bf16.metal") ++ "\n" ++
        @embedFile("token_batch.metal") ++ "\n" ++
        @embedFile("token_batch_attention.metal");
    return .{ .handle = fm_open(source) orelse {
        std.debug.print("Metal batch attention: {s}\n", .{std.mem.span(fm_error())});
        return error.MetalFailure;
    } };
}

fn poison(buffer: metal.Buffer, words: usize, bits: u32) void {
    @memset(buffer.structs(u32)[0..words], bits);
    @memset(buffer.structs(u32)[words..], guard);
}
fn expectGuard(buffer: metal.Buffer, words: usize) !void {
    try std.testing.expectEqual(words + guards, buffer.structs(u32).len);
    for (buffer.structs(u32)[words..]) |bits| try std.testing.expectEqual(guard, bits);
}
fn expectBits(expected: []const u32, actual: []const u32, p: metal.Params, token: usize, label: []const u8, variant: Variant) !void {
    try std.testing.expectEqual(expected.len, actual.len);
    for (expected, actual, 0..) |a, b, i| {
        if (a != b or !std.math.isFinite(@as(f32, @bitCast(b)))) {
            std.debug.print("batch attention {s} {s}: pos={d}, token={d}, B={d}, BF16={d}, dtype={d}, word={d}, expected=0x{x}, actual=0x{x}\n", .{
                variant.kernel, label, p.pos, token, p.k, p.bits, p.dtype, i, a, b,
            });
            try std.testing.expectEqual(a, b);
            try std.testing.expect(std.math.isFinite(@as(f32, @bitCast(b))));
        }
    }
}
fn bf16(value: f32) u16 {
    const bits: u32 = @bitCast(value);
    if (bits & 0x7fffffff > 0x7f800000) return @truncate((bits >> 16) | 0x40);
    return @truncate((bits +% 0x7fff +% ((bits >> 16) & 1)) >> 16);
}
fn writeCache(buffer: metal.Buffer, index: usize, value: f32, compressed: bool) void {
    if (compressed) buffer.structs(u16)[index] = bf16(value) else buffer.floats()[index] = value;
}
fn writeSink(sink: metal.Buffer, p: metal.Params, dominant: bool) void {
    const offset: usize = @intCast(p.offset);
    for (0..heads) |head| {
        const value: f32 = if (dominant)
            12 + @as(f32, @floatFromInt(head)) * 0.125
        else
            -2 - @as(f32, @floatFromInt(head)) * 0.0625;
        switch (p.dtype) {
            0 => sink.floats()[offset / 4 + head] = value,
            1 => sink.structs(u16)[offset / 2 + head] = @bitCast(@as(f16, @floatCast(value))),
            2 => sink.structs(u16)[offset / 2 + head] = bf16(value),
            else => unreachable,
        }
    }
}

const Token = struct {
    q: metal.Buffer,
    actual: metal.Buffer,
    expected: metal.Buffer,
    fn init(ctx: metal.Context, index: usize) !Token {
        const q = try ctx.alloc(query_words + guards);
        errdefer q.deinit();
        const actual = try ctx.alloc(query_words + guards);
        errdefer actual.deinit();
        const expected = try ctx.alloc(query_words + guards);
        poison(q, query_words, reference_poison);
        poison(actual, query_words, batch_poison);
        poison(expected, query_words, reference_poison);
        for (q.floats()[0..query_words], 0..) |*value, i|
            value.* = @as(f32, @floatFromInt((i * 19 + i / dim * 7 + index * 43) % 257)) / 128 - 1;
        return .{ .q = q, .actual = actual, .expected = expected };
    }
    fn deinit(self: Token) void {
        self.q.deinit();
        self.actual.deinit();
        self.expected.deinit();
    }
};
const Case = struct {
    first: u32,
    batch: u32,
    dtype: u32 = 0,
    dominant_sink: bool = false,
    poison_last_value: bool = false,
};

fn dispatchBatch(ctx: metal.Context, tokens: []const Token, scratch: metal.Buffer, keys: metal.Buffer, values: metal.Buffer, sink: metal.Buffer, states: metal.Buffer, p: metal.Params, extra: usize, variant: Variant) !void {
    for (tokens) |token| {
        try ctx.resident(token.q);
        try ctx.residentWrite(token.actual);
    }
    const before = ctx.stats();
    const groups = @as(usize, kv_heads) * p.k * ((p.m + variant.queries - 1) / variant.queries);
    try ctx.dispatch(variant.kernel, &.{ scratch, keys, values, sink, states }, p, (groups + extra) * 128, 128);
    try ctx.dispatch("token_batch_attention_merge", &.{ states, scratch }, p, (@as(usize, p.m) * heads + extra) * 1024, 1024);
    try ctx.finish();
    const after = ctx.stats();
    try std.testing.expectEqual(@as(u64, 2), after.dispatches - before.dispatches);
    try std.testing.expectEqual(@as(u64, 1), after.submissions - before.submissions);
}

fn verifyAgainstScalar(ctx: metal.Context, tokens: []const Token, keys: metal.Buffer, values: metal.Buffer, sink: metal.Buffer, states: metal.Buffer, reference_states: metal.Buffer, p: metal.Params, verified_tokens: usize, variant: Variant) !void {
    const state_words = @as(usize, heads) * p.k * 130;
    const baseline: [:0]const u8 = if (p.bits == 0) "attention_mlx_split_gqa2" else "attention_mlx_split_gqa2_bf16";
    for (tokens[0..verified_tokens], 0..) |token, t| {
        poison(reference_states, state_words, reference_poison);
        poison(token.expected, query_words, reference_poison);
        var scalar = p;
        scalar.pos += @intCast(t);
        // An ordinary one-query producer and ORIGINAL merge, same stripe count
        // and same cache storage. P.m is deliberately irrelevant to baseline.
        try ctx.dispatch(baseline, &.{ token.q, keys, values, sink, reference_states }, scalar, @as(usize, kv_heads) * p.k * 128, 128);
        try ctx.dispatch("attention_mlx_merge", &.{ reference_states, token.expected }, scalar, @as(usize, heads) * 1024, 1024);
        try ctx.finish();
        try expectBits(reference_states.structs(u32)[0..state_words], states.structs(u32)[t * state_words ..][0..state_words], p, t, "producer states", variant);
        try expectBits(token.expected.structs(u32)[0..query_words], token.actual.structs(u32)[0..query_words], p, t, "merged output", variant);
        try expectGuard(reference_states, state_words);
        try expectGuard(token.expected, query_words);
    }
}

fn runCase(ctx: metal.Context, case: Case, compressed: bool, variant: Variant) !void {
    var p: metal.Params = .{
        .m = case.batch,
        .pos = case.first,
        .capacity = case.first + case.batch + 7,
        .heads = heads,
        .kv_heads = kv_heads,
        .dim = dim,
        .bits = @intFromBool(compressed),
        .dtype = case.dtype,
        .offset = switch (case.dtype) {
            0 => 12,
            1 => 6,
            2 => 10,
            else => unreachable,
        },
        .scale = if (case.dominant_sink) 0.01 else 0.13,
    };
    var tokens: [max_batch]Token = undefined;
    var initialized: usize = 0;
    defer for (tokens[0..initialized]) |token| token.deinit();
    for (&tokens, 0..) |*token, t| {
        token.* = try Token.init(ctx, t);
        initialized += 1;
    }
    const scratch = try ctx.alloc(max_batch * @sizeOf(TokenScratch) / 4);
    defer scratch.deinit();
    for (tokens, 0..) |token, t| scratch.structs(TokenScratch)[t] = .{ .q = token.q.address(), .attention = token.actual.address() };
    const cache_elements = @as(usize, p.capacity) * kv_heads * dim;
    const cache_words = if (compressed) cache_elements / 2 else cache_elements;
    const keys = try ctx.alloc(cache_words + guards);
    defer keys.deinit();
    const values = try ctx.alloc(cache_words + guards);
    defer values.deinit();
    const sink_words: usize = (@as(usize, @intCast(p.offset)) + heads * @as(usize, if (case.dtype == 0) 4 else 2) + 3) / 4;
    const sink = try ctx.alloc(sink_words + guards);
    defer sink.deinit();
    defer ctx.finish() catch {};
    poison(keys, cache_words, reference_poison);
    poison(values, cache_words, reference_poison);
    if (compressed) {
        @memset(keys.structs(u16)[0..cache_elements], bf16_poison);
        @memset(values.structs(u16)[0..cache_elements], bf16_poison);
    }
    poison(sink, sink_words, reference_poison);
    writeSink(sink, p, case.dominant_sink);
    const sink_snapshot = try std.testing.allocator.dupe(u32, sink.structs(u32));
    defer std.testing.allocator.free(sink_snapshot);
    const last_n = case.first + case.batch;
    for (0..last_n) |t| {
        for (0..kv_heads) |head| {
            for (0..dim) |d| {
                const index = (t * kv_heads + head) * dim + d;
                const k = @as(f32, @floatFromInt((t * 17 + t / 31 * 13 + head * 43 + d * 7) % 257)) / 173 - 0.7;
                const v = @as(f32, @floatFromInt((t * 29 + t / 17 * 11 + head * 19 + d * 37) % 257)) / 263 - 0.25;
                writeCache(keys, index, k, compressed);
                writeCache(values, index, v, compressed);
            }
        }
    }
    for (blocks_list) |blocks| {
        p.k = blocks;
        const state_words = @as(usize, heads) * blocks * 130;
        const total_words = @as(usize, case.batch) * state_words;
        const states = try ctx.alloc(total_words + guards);
        defer states.deinit();
        const reference_states = try ctx.alloc(state_words + guards);
        defer reference_states.deinit();
        defer ctx.finish() catch {};
        // Test the exact launch and overdispatch independently. Different
        // poison payloads make unwritten states/output fail even if both omit.
        for ([_]usize{ 0, 3 }) |extra| {
            poison(states, total_words, batch_poison);
            for (tokens) |token| poison(token.actual, query_words, batch_poison);
            try dispatchBatch(ctx, &tokens, scratch, keys, values, sink, states, p, extra, variant);
            try verifyAgainstScalar(ctx, &tokens, keys, values, sink, states, reference_states, p, case.batch, variant);
            try expectGuard(states, total_words);
            for (tokens, 0..) |token, t| {
                try expectGuard(token.q, query_words);
                try expectGuard(token.actual, query_words);
                if (t >= case.batch) for (token.actual.structs(u32)[0..query_words]) |bits|
                    try std.testing.expectEqual(batch_poison, bits);
            }
        }
        if (case.poison_last_value) {
            // The last query in the last group genuinely needs this value;
            // every earlier query must skip it. NaN makes speculative loads
            // followed by accidental masked multiplication observable. This
            // also covers odd query-group tails and TK4/prefix boundaries.
            const future_start = @as(usize, last_n - 1) * kv_heads * dim;
            var saved: [kv_heads * dim]f32 = undefined;
            for (&saved, 0..) |*value, i| {
                value.* = if (compressed) @bitCast(@as(u32, values.structs(u16)[future_start + i]) << 16) else values.floats()[future_start + i];
                writeCache(values, future_start + i, std.math.nan(f32), compressed);
            }
            poison(states, total_words, batch_poison);
            for (tokens) |token| poison(token.actual, query_words, batch_poison);
            try dispatchBatch(ctx, &tokens, scratch, keys, values, sink, states, p, 0, variant);
            try verifyAgainstScalar(ctx, &tokens, keys, values, sink, states, reference_states, p, case.batch - 1, variant);
            try expectGuard(states, total_words);
            for (saved, 0..) |value, i| writeCache(values, future_start + i, value, compressed);
        }
        try expectGuard(keys, cache_words);
        try expectGuard(values, cache_words);
        try std.testing.expectEqualSlices(u32, sink_snapshot, sink.structs(u32));
        // Entire unused global capacity remains poisoned, not just its guard.
        const unused = @as(usize, last_n) * kv_heads * dim;
        if (compressed) {
            for (keys.structs(u16)[unused..cache_elements]) |bits| try std.testing.expectEqual(bf16_poison, bits);
            for (values.structs(u16)[unused..cache_elements]) |bits| try std.testing.expectEqual(bf16_poison, bits);
        } else {
            for (keys.structs(u32)[unused..cache_elements]) |bits| try std.testing.expectEqual(reference_poison, bits);
            for (values.structs(u32)[unused..cache_elements]) |bits| try std.testing.expectEqual(reference_poison, bits);
        }
    }
}

test "batch attention Q2 Q4 Q8 preserve GQA2 states and MLX merge bits for FP32 and BF16" {
    const ctx = try context();
    defer ctx.deinit();
    // Query prefixes straddle 1024 and 16K, including full TK4/tail changes for
    // all three B values. Keep the original 1/3/7/8 critical counts and add
    // 16/31/32, including the new maximum and odd multi-group tails. EVERY
    // variant is checked; larger query groups are not assumed to be faster.
    for (variants) |variant| {
        for ([_]bool{ false, true }) |compressed| {
            for ([_]Case{
                .{ .first = 0, .batch = 1, .dominant_sink = true },
                .{ .first = 0, .batch = 8, .dtype = 2 },
                .{ .first = 1, .batch = 2, .dtype = 1, .poison_last_value = true },
                .{ .first = 1021, .batch = 4, .poison_last_value = true },
                .{ .first = 1023, .batch = 3, .dtype = 1 },
                .{ .first = 4093, .batch = 7, .dtype = 2, .dominant_sink = true },
                .{ .first = 16381, .batch = 7, .dtype = 2, .poison_last_value = true },
                .{ .first = 16383, .batch = 2, .dtype = 1, .dominant_sink = true },
                .{ .first = 16384, .batch = 4 },
                .{ .first = 32760, .batch = 8, .dtype = 2, .poison_last_value = true },
                .{ .first = 1020, .batch = 16, .poison_last_value = true },
                .{ .first = 16380, .batch = 31, .dtype = 1, .poison_last_value = true },
                .{ .first = 32736, .batch = max_batch, .dtype = 2, .poison_last_value = true },
            }) |case| try runCase(ctx, case, compressed, variant);
        }
    }
}

const Rejection = enum {
    zero_batch,
    large_batch,
    wrong_heads,
    zero_kv,
    wrong_dim,
    sliding,
    zero_capacity,
    nonresident,
    overflow,
    too_long,
    bad_blocks,
    bad_dtype,
    unaligned_sink,
    bad_bits,
    wrong_group,
};

test "batch attention unsupported uniform geometry leaves all states and output untouched" {
    const ctx = try context();
    defer ctx.deinit();
    const q = try ctx.alloc(query_words + guards);
    defer q.deinit();
    const out = try ctx.alloc(query_words + guards);
    defer out.deinit();
    const cache = try ctx.alloc(kv_heads * dim + guards);
    defer cache.deinit();
    const sink = try ctx.alloc(heads + guards);
    defer sink.deinit();
    const scratch = try ctx.alloc(max_batch * @sizeOf(TokenScratch) / 4);
    defer scratch.deinit();
    const state_words: usize = max_batch * heads * 128 * 130;
    const states = try ctx.alloc(state_words + guards);
    defer states.deinit();
    defer ctx.finish() catch {};
    poison(q, query_words, reference_poison);
    poison(cache, kv_heads * dim, reference_poison);
    poison(sink, heads, reference_poison);
    for (scratch.structs(TokenScratch)) |*token| token.* = .{ .q = q.address(), .attention = out.address() };
    for (variants) |variant| {
        for ([_]Rejection{
            .zero_batch,     .large_batch, .wrong_heads, .zero_kv,  .wrong_dim,  .sliding,
            .zero_capacity,  .nonresident, .overflow,    .too_long, .bad_blocks, .bad_dtype,
            .unaligned_sink, .bad_bits,    .wrong_group,
        }) |reason| {
            var p: metal.Params = .{ .m = 1, .heads = heads, .kv_heads = kv_heads, .dim = dim, .capacity = 1, .k = 128 };
            var split_group: usize = 128;
            var merge_group: usize = 1024;
            switch (reason) {
                .zero_batch => p.m = 0,
                .large_batch => p.m = max_batch + 1,
                .wrong_heads => p.heads = 31,
                .zero_kv => p.kv_heads = 0,
                .wrong_dim => p.dim = 64,
                .sliding => p.window = 1,
                .zero_capacity => p.capacity = 0,
                .nonresident => p.m = 2,
                .overflow => p.pos = std.math.maxInt(u32),
                .too_long => {
                    p.pos = 262144;
                    p.capacity = 262145;
                },
                .bad_blocks => p.k = 64,
                .bad_dtype => p.dtype = 3,
                .unaligned_sink => p.offset = 2,
                .bad_bits => p.bits = 2,
                .wrong_group => {
                    split_group = 64;
                    merge_group = 512;
                },
            }
            poison(states, state_words, batch_poison);
            poison(out, query_words, batch_poison);
            try ctx.resident(q);
            try ctx.residentWrite(out);
            // Test split and merge separately: a rejected producer is NOT a valid
            // input to merge. Here the same invalid P must make each reject itself.
            try ctx.dispatch(variant.kernel, &.{ scratch, cache, cache, sink, states }, p, split_group, split_group);
            try ctx.dispatch("token_batch_attention_merge", &.{ states, scratch }, p, merge_group, merge_group);
            try ctx.finish();
            for (states.structs(u32)[0..state_words]) |bits| try std.testing.expectEqual(batch_poison, bits);
            for (out.structs(u32)[0..query_words]) |bits| try std.testing.expectEqual(batch_poison, bits);
            try expectGuard(states, state_words);
            try expectGuard(out, query_words);
        }
    }
}
