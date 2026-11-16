//! Import from the test root to integrate; Context below compiles an isolated
//! library including token_batch.metal, so production imports need not change.
//! Headless GPU tests: run only when the coordinating caller schedules them.
const std = @import("std");
const metal = @import("metal.zig");
const plan = @import("resident_plan.zig");
extern fn fm_open([*:0]const u8) ?*anyopaque;
extern fn fm_error() [*:0]const u8;

/// Shared with token_batch.metal. GPU addresses, NOT host pointers.
pub const TokenScratch = extern struct {
    x: u64,
    norm: u64,
    q: u64,
    k: u64,
    v: u64,
    attention: u64,
    projection: u64,
    activation: u64,
    router: u64,
    route_ids: u64,
    route_weights: u64,
    route_error: u64,
    logits: u64,
};
const scratch_fields = [_][]const u8{ "x", "norm", "q", "k", "v", "attention", "projection", "activation", "router", "route_ids", "route_weights", "route_error", "logits" };
comptime {
    if (@sizeOf(TokenScratch) != 104 or @alignOf(TokenScratch) != 8)
        @compileError("TokenScratch must contain thirteen packed GPU addresses");
    for (scratch_fields, 0..) |field, index| {
        if (@offsetOf(TokenScratch, field) != index * 8)
            @compileError("TokenScratch field order mismatch");
    }
}

const sentinel: u32 = 0x7fc12345;
const max_batch = 32;
// Retain every original count and probe both the new maximum and its odd tail.
const batch_counts = [_]usize{ 1, 2, 3, 4, 5, 6, 7, 8, 16, 31, max_batch };
const capacity = 512;
const hidden = 129; // Down row tails for every tile > 1.
const input_cols = 128;
const gate_rows = 65; // Gate/up tails, with deliberately poisoned stride padding.
const down_cols = 64;
const stride = 80;
const expert_count = 4;
const shared = 3;
const Kind = enum { quant, fp32, mixed };
const Slot = enum { x, norm, q, k, v, attention, projection, activation, router, route_ids, route_weights, route_error, logits };

fn context(matched: bool) !metal.Context {
    const source = @embedFile("kernels.metal") ++ "\n" ++ @embedFile("resident_plan.metal") ++ "\n" ++ @embedFile("row_tiling.metal") ++ "\n" ++ @embedFile("token_batch.metal");
    return .{ .handle = fm_open(if (matched) "#define FLINT_MLX_MATH 1\n" ++ source else source) orelse {
        std.debug.print("Metal batch test: {s}\n", .{std.mem.span(fm_error())});
        return error.MetalFailure;
    } };
}

const MatrixStorage = struct {
    w: metal.Buffer,
    scales: metal.Buffer,
    biases: metal.Buffer,
    matrix: metal.Matrix,

    fn init(ctx: metal.Context, rows: u32, cols: u32, quant: bool, seed: usize) !MatrixStorage {
        const elements = @as(usize, rows) * cols;
        const groups = elements / 64;
        const w = try ctx.alloc(elements + 2);
        errdefer w.deinit();
        const scales = try ctx.alloc(groups + 1);
        errdefer scales.deinit();
        const biases = try ctx.alloc(groups + 2);
        errdefer biases.deinit();
        @memset(w.structs(u8), 0);
        @memset(scales.structs(u32), sentinel);
        @memset(biases.structs(u32), sentinel);
        for (0..groups) |g| {
            scales.floats()[1 + g] = @as(f32, @floatFromInt((g * 3 + seed * 5) % 17 + 1)) / 1024;
            biases.floats()[2 + g] = @as(f32, @floatFromInt((g + seed * 3) % 19)) / 256 - 0.03125;
        }
        for (0..elements) |i| {
            if (quant) {
                const code: u8 = @intCast((i * 7 + seed * 11 + i / cols) % 16);
                w.structs(u8)[4 + i / 2] |= code << @as(u3, @intCast(i % 2 * 4));
            } else {
                w.floats()[2 + i] = @as(f32, @floatFromInt((i * 13 + seed * 17 + i / cols) % 53)) / 257 - 0.1;
            }
        }
        return .{
            .w = w,
            .scales = scales,
            .biases = biases,
            .matrix = .{
                .weight = w.address() + @as(u64, if (quant) 4 else 8),
                .scales = scales.address() + 4,
                .biases = biases.address() + 8,
                .rows = rows,
                .cols = cols,
                .dtype = if (quant) 3 else 0,
                .bits = if (quant) 4 else 0,
                .group_size = if (quant) 64 else 0,
                .reserved = if (quant) 1 else 0,
            },
        };
    }
    fn deinit(self: MatrixStorage) void {
        self.w.deinit();
        self.scales.deinit();
        self.biases.deinit();
    }
    fn resident(self: MatrixStorage, ctx: metal.Context) !void {
        try ctx.resident(self.w);
        try ctx.resident(self.scales);
        try ctx.resident(self.biases);
    }
};

const Token = struct {
    buffers: [13]metal.Buffer,
    fn init(ctx: metal.Context) !Token {
        var result: Token = undefined;
        var initialized: usize = 0;
        errdefer for (result.buffers[0..initialized]) |buffer| buffer.deinit();
        for (&result.buffers) |*buffer| {
            buffer.* = try ctx.alloc(capacity);
            initialized += 1;
        }
        return result;
    }
    fn deinit(self: Token) void {
        for (self.buffers) |buffer| buffer.deinit();
    }
    fn buf(self: Token, slot: Slot) metal.Buffer {
        return self.buffers[@backingInt(slot)];
    }
    fn descriptor(self: Token) TokenScratch {
        var result: TokenScratch = undefined;
        inline for (scratch_fields, 0..) |field, i|
            @field(result, field) = self.buffers[i].address();
        return result;
    }
    fn reset(self: Token, token: usize, routes: usize) void {
        for (self.buffers) |buffer| @memset(buffer.structs(u32), sentinel);
        fill(self.buf(.norm).floats()[0..input_cols], token * 37 + 1);
        fill(self.buf(.attention).floats()[0..input_cols], token * 43 + 19);
        fill(self.buf(.x).floats()[0..hidden], token * 13 + 7);
        for (0..routes) |route| {
            self.buf(.route_ids).structs(u32)[route] = @intCast((token + route * 2) % shared);
            self.buf(.route_weights).floats()[route] = if (route == 0) 0.17 + @as(f32, @floatFromInt(token)) / 100 else 0.63;
        }
        self.buf(.route_error).structs(u32)[0] = 0;
    }
    fn resident(self: Token, ctx: metal.Context) !void {
        // All scratch resources are indirect and may be read or written.
        for (self.buffers) |buffer| try ctx.residentWrite(buffer);
    }
};

fn fill(values: []f32, seed: usize) void {
    for (values, 0..) |*value, i|
        value.* = @as(f32, @floatFromInt((i * 5 + seed * 7) % 97)) / 61 - 0.75;
}

fn compareTokens(expected: []const Token, actual: []const Token) !void {
    for (expected, actual) |reference, result| {
        for (reference.buffers, result.buffers) |a, b|
            try std.testing.expectEqualSlices(u32, a.structs(u32), b.structs(u32));
    }
}

fn expectOutput(token: Token, slot: Slot, start: usize, len: usize) !void {
    for (token.buf(slot).floats()[start..][0..len]) |value|
        try std.testing.expect(std.math.isFinite(value));
}

fn stagedName(comptime op: enum { qkv, gate_up, down, qmatvec }, tile: u32) [:0]const u8 {
    return switch (tile) {
        1 => "staged_" ++ @tagName(op) ++ "_t1",
        2 => "staged_" ++ @tagName(op) ++ "_t2",
        4 => "staged_" ++ @tagName(op) ++ "_t4",
        8 => "staged_" ++ @tagName(op) ++ "_t8",
        else => unreachable,
    };
}
fn tiles(rows: usize, tile: usize) usize {
    return (rows + tile - 1) / tile;
}
fn expectOneDispatch(ctx: metal.Context, before: metal.Stats) !void {
    const after = ctx.stats();
    try std.testing.expectEqual(@as(u64, 1), after.dispatches - before.dispatches);
    try std.testing.expectEqual(@as(u64, 1), after.submissions - before.submissions);
}

test "token batch wrappers are bitwise equal to scalar staged kernels in matched FP32 mode" {
    try checkBatches(true);
}

test "token batch wrappers preserve legacy FP32 residual and mixture order" {
    try checkBatches(false);
}

fn checkBatches(matched: bool) !void {
    const ctx = try context(matched);
    defer ctx.deinit();
    var reference: [max_batch]Token = undefined;
    var actual: [max_batch]Token = undefined;
    var initialized: usize = 0;
    defer {
        for (reference[0..initialized]) |token| token.deinit();
        for (actual[0..initialized]) |token| token.deinit();
    }
    for (0..max_batch) |i| {
        reference[i] = try Token.init(ctx);
        actual[i] = Token.init(ctx) catch |err| {
            reference[i].deinit();
            return err;
        };
        initialized += 1;
    }
    const scratch = try ctx.alloc(max_batch * @sizeOf(TokenScratch) / 4);
    defer scratch.deinit();
    for (actual, 0..) |token, i| scratch.structs(TokenScratch)[i] = token.descriptor();

    for ([_]Kind{ .quant, .fp32, .mixed }) |kind| {
        var storage: [3 + expert_count * 3]MatrixStorage = undefined;
        var matrix_count: usize = 0;
        defer for (storage[0..matrix_count]) |matrix| matrix.deinit();
        for (&storage, 0..) |*matrix, i| {
            const rows: u32 = if (i < 3) ([_]u32{ 5, 3, 1 })[i] else if ((i - 3) % 3 == 2) hidden else gate_rows;
            const cols: u32 = if (i >= 3 and (i - 3) % 3 == 2) down_cols else input_cols;
            matrix.* = try MatrixStorage.init(ctx, rows, cols, kind == .quant or (kind == .mixed and i % 2 == 0), i);
            matrix_count += 1;
        }
        const qkv = try ctx.alloc(3 * @sizeOf(metal.Matrix) / 4);
        defer qkv.deinit();
        for (0..3) |i| qkv.structs(metal.Matrix)[i] = storage[i].matrix;
        const experts = try ctx.alloc(expert_count * @sizeOf(metal.MLP) / 4);
        defer experts.deinit();
        for (experts.structs(metal.MLP), 0..) |*expert, i| expert.* = .{
            .gate = storage[3 + i * 3].matrix,
            .up = storage[4 + i * 3].matrix,
            .down = storage[5 + i * 3].matrix,
        };
        for (batch_counts) |count| {
            for ([_]u32{ 0, 2 }) |routes| {
                for (plan.row_tiles) |tile| {
                    // Extra SIMD groups test token-count bounds in addition to
                    // the exact contract. Entire allocations are compared, so
                    // tails, unused fields, inactive tokens, and padding guard
                    // against cross-token aliases and writes past valid rows.
                    for ([_]usize{ 0, 128 }) |extra| {
                        for (0..max_batch) |i| {
                            reference[i].reset(i, routes);
                            actual[i].reset(i, routes);
                        }
                        for (storage) |matrix| try matrix.resident(ctx);
                        for (actual) |token| try token.resident(ctx);
                        for (reference[0..count]) |token|
                            try ctx.dispatch(stagedName(.qkv, tile), &.{ qkv, token.buf(.norm), token.buf(.q), token.buf(.k), token.buf(.v) }, .{}, (tiles(5, tile) + tiles(3, tile) + tiles(1, tile)) * 32, 128);
                        try ctx.finish();
                        for (storage) |matrix| try matrix.resident(ctx);
                        for (actual) |token| try token.resident(ctx);
                        var before = ctx.stats();
                        try ctx.dispatch("token_batch_qkv", &.{ qkv, scratch }, .{ .m = @intCast(count), .n = tile }, count * (tiles(5, tile) + tiles(3, tile) + tiles(1, tile)) * 32 + extra, 128);
                        try ctx.finish();
                        try expectOneDispatch(ctx, before);
                        try compareTokens(&reference, &actual);
                        for (actual[0..count]) |token| {
                            try expectOutput(token, .q, 0, 5);
                            try expectOutput(token, .k, 0, 3);
                            try expectOutput(token, .v, 0, 1);
                        }

                        const p: metal.Params = .{ .n = routes, .m = shared, .k = stride, .dim = hidden, .heads = @intCast(count), .bits = tile };
                        for (storage) |matrix| try matrix.resident(ctx);
                        for (actual) |token| try token.resident(ctx);
                        for (reference[0..count]) |token|
                            try ctx.dispatch(stagedName(.gate_up, tile), &.{ experts, token.buf(.norm), token.buf(.route_ids), token.buf(.activation) }, p, (routes + 1) * tiles(stride, tile) * 32, 128);
                        try ctx.finish();
                        for (storage) |matrix| try matrix.resident(ctx);
                        for (actual) |token| try token.resident(ctx);
                        before = ctx.stats();
                        try ctx.dispatch("token_batch_gate_up", &.{ experts, scratch }, p, count * (routes + 1) * tiles(stride, tile) * 32 + extra, 128);
                        try ctx.finish();
                        try expectOneDispatch(ctx, before);
                        try compareTokens(&reference, &actual);
                        for (actual[0..count]) |token| {
                            for (0..routes + 1) |route| try expectOutput(token, .activation, route * stride, gate_rows);
                        }

                        for (storage) |matrix| try matrix.resident(ctx);
                        for (actual) |token| try token.resident(ctx);
                        for (reference[0..count]) |token|
                            try ctx.dispatch(stagedName(.down, tile), &.{ experts, token.buf(.activation), token.buf(.route_ids), token.buf(.route_weights), token.buf(.x) }, p, tiles(hidden, tile) * 32, 128);
                        try ctx.finish();
                        for (storage) |matrix| try matrix.resident(ctx);
                        for (actual) |token| try token.resident(ctx);
                        before = ctx.stats();
                        try ctx.dispatch("token_batch_down", &.{ experts, scratch }, p, count * tiles(hidden, tile) * 32 + extra, 128);
                        try ctx.finish();
                        try expectOneDispatch(ctx, before);
                        try compareTokens(&reference, &actual);
                        for (actual[0..count]) |token| try expectOutput(token, .x, 0, hidden);
                    }
                }
            }
        }
    }
    try checkMatvec(ctx, &reference, &actual, scratch);
}

fn checkMatvec(ctx: metal.Context, reference: *[max_batch]Token, actual: *[max_batch]Token, scratch: metal.Buffer) !void {
    const storage = try MatrixStorage.init(ctx, 13, input_cols, true, 97);
    defer storage.deinit();
    const descriptor = try ctx.alloc(@sizeOf(metal.Matrix) / 4);
    defer descriptor.deinit();
    descriptor.structs(metal.Matrix)[0] = storage.matrix;
    for (batch_counts) |count| {
        for (plan.row_tiles) |tile| {
            for ([_]u32{ 0, 1 }) |selector| {
                for ([_]usize{ 0, 128 }) |extra| {
                    for (0..max_batch) |i| {
                        reference[i].reset(i, 0);
                        actual[i].reset(i, 0);
                    }
                    const in: Slot = if (selector == 0) .attention else .norm;
                    const out: Slot = if (selector == 0) .projection else .logits;
                    for (reference[0..count]) |token|
                        try ctx.dispatch(stagedName(.qmatvec, tile), &.{ storage.w, storage.scales, storage.biases, token.buf(in), token.buf(out) }, .{ .offset = 4, .scale_offset = 4, .bias_offset = 8, .n = input_cols, .m = 13, .k = 64, .bits = 4, .dtype = 3 }, tiles(13, tile) * 32, 128);
                    try ctx.finish();
                    try storage.resident(ctx);
                    for (actual) |token| try token.resident(ctx);
                    const before = ctx.stats();
                    try ctx.dispatch("token_batch_matvec", &.{ descriptor, scratch }, .{ .m = @intCast(count), .n = tile, .bits = selector }, count * tiles(13, tile) * 32 + extra, 128);
                    try ctx.finish();
                    try expectOneDispatch(ctx, before);
                    try compareTokens(reference, actual);
                    for (actual[0..count]) |token| try expectOutput(token, out, 0, 13);
                }
            }
        }
    }
    // Malformed uniform fields and unsupported dense/router descriptors must
    // return without touching scratch (including the unsupported selector).
    for (actual, 0..) |token, i| token.reset(i, 0);
    for (reference, 0..) |token, i| token.reset(i, 0);
    for ([_]metal.Params{
        .{ .m = 0, .n = 1 },                    .{ .m = max_batch + 1, .n = 1 },
        .{ .m = max_batch, .n = 0 },            .{ .m = max_batch, .n = 3 },
        .{ .m = max_batch, .n = 8, .bits = 2 },
    }) |p| {
        try storage.resident(ctx);
        for (actual) |token| try token.resident(ctx);
        try ctx.dispatch("token_batch_matvec", &.{ descriptor, scratch }, p, 256, 128);
        try ctx.finish();
        try compareTokens(reference, actual);
    }
    descriptor.structs(metal.Matrix)[0].reserved = 0;
    descriptor.structs(metal.Matrix)[0].bits = 0;
    descriptor.structs(metal.Matrix)[0].dtype = 0;
    try storage.resident(ctx);
    for (actual) |token| try token.resident(ctx);
    try ctx.dispatch("token_batch_matvec", &.{ descriptor, scratch }, .{ .m = max_batch, .n = 1 }, max_batch * 13 * 32, 128);
    try ctx.finish();
    try compareTokens(reference, actual);
}
