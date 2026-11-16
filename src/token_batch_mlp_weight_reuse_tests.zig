//! Scheduled/headless parity gates for the bounded MLP weight-pair candidate.
//! This context embeds only original dependencies and the new source. Fixtures
//! are synthetic checkpoint-format BF16/F32 affine INT4, dense F32, and mixed.
//! Run GPU tests only after the coordinating worker grants the GPU lock.
const std = @import("std");
const metal = @import("metal.zig");
extern fn fm_open([*:0]const u8) ?*anyopaque;
extern fn fm_error() [*:0]const u8;
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
        @offsetOf(TokenScratch, "logits") != 96)
        @compileError("TokenScratch thirteen-address ABI mismatch");
}
const hidden = 13;
const shared = 3;
const expert_count = 4;
const guard: u32 = 0x4b123456;
const ref_poison: u32 = 0x7fc12345;
const new_poison: u32 = 0x7fc54321;
const Shape = struct { input: usize, gate: usize, down: usize, stride: usize };
const shapes = [_]Shape{
    .{ .input = 576, .gate = 577, .down = 576, .stride = 592 },
    .{ .input = 1088, .gate = 1153, .down = 1152, .stride = 1168 },
};
const Kind = enum { quant_bf16, quant_f32, dense, mixed };
fn context(matched: bool) !metal.Context {
    const source = @embedFile("kernels.metal") ++ "\n" ++
        @embedFile("resident_plan.metal") ++ "\n" ++
        @embedFile("row_tiling.metal") ++ "\n" ++
        @embedFile("token_batch.metal") ++ "\n" ++
        @embedFile("token_batch_mlp_weight_reuse.metal");
    return .{ .handle = fm_open(if (matched) "#define FLINT_MLX_MATH 1\n" ++ source else source) orelse {
        std.debug.print("Metal MLP weight-pair test: {s}\n", .{std.mem.span(fm_error())});
        return error.MetalFailure;
    } };
}
fn tiles(n: usize, tile: usize) usize {
    return (n + tile - 1) / tile;
}
fn bf16(value: f32) u16 {
    const bits: u32 = @bitCast(value);
    return @truncate((bits +% 0x7fff +% ((bits >> 16) & 1)) >> 16);
}
fn affine(buffer: metal.Buffer, index: usize, value: f32, dtype: u32) void {
    if (dtype == 2) buffer.structs(u16)[index] = bf16(value) else buffer.floats()[index] = value;
}
const Storage = struct {
    w: metal.Buffer,
    scales: metal.Buffer,
    biases: metal.Buffer,
    matrix: metal.Matrix,
    fn init(ctx: metal.Context, rows: usize, cols: usize, quant: bool, dtype: u32, seed: usize) !Storage {
        const elements = rows * cols;
        const groups = elements / 64;
        const w = try ctx.alloc((if (quant) elements / 8 else elements) + 4);
        errdefer w.deinit();
        const scales = try ctx.alloc(groups + 4);
        errdefer scales.deinit();
        const biases = try ctx.alloc(groups + 4);
        errdefer biases.deinit();
        @memset(w.structs(u32), guard);
        @memset(scales.structs(u32), guard);
        @memset(biases.structs(u32), guard);
        if (quant) @memset(w.structs(u8)[4..][0 .. elements / 2], 0);
        for (0..elements) |i| {
            if (quant) {
                const mixed = (i *% 1664525 +% seed *% 1013904223) ^ (i >> 3) ^ (i >> 9);
                const code: u8 = @intCast((mixed >> 5) & 15);
                w.structs(u8)[4 + i / 2] |= code << @as(u3, @intCast((i % 2) * 4));
            } else {
                w.floats()[1 + i] = (@as(f32, @floatFromInt((i * 29 + seed * 43 + i / cols) % 211)) - 105) / 13739;
            }
        }
        for (0..groups) |g| {
            const scale = @as(f32, @floatFromInt((g * 13 + seed * 7) % 71 + 1)) / 7139;
            const bias = @as(f32, @floatFromInt((g * 17 + seed * 3) % 89)) / 2791 - 0.021;
            affine(scales, g + (if (dtype == 2) @as(usize, 2) else 1), scale, dtype);
            affine(biases, g + (if (dtype == 2) @as(usize, 4) else 2), bias, dtype);
        }
        return .{ .w = w, .scales = scales, .biases = biases, .matrix = .{
            .weight = w.address() + 4,
            .scales = scales.address() + 4,
            .biases = biases.address() + 8,
            .rows = @intCast(rows),
            .cols = @intCast(cols),
            .dtype = if (quant) 3 else 0,
            .bits = if (quant) 4 else 0,
            .group_size = if (quant) 64 else 0,
            .scale_dtype = dtype,
            .bias_dtype = dtype,
            .reserved = if (quant) 1 else 0,
        } };
    }
    fn deinit(self: Storage) void {
        self.w.deinit();
        self.scales.deinit();
        self.biases.deinit();
    }
    fn resident(self: Storage, ctx: metal.Context) !void {
        try ctx.resident(self.w);
        try ctx.resident(self.scales);
        try ctx.resident(self.biases);
    }
};
const Slot = enum { x, norm, activation, route_ids, route_weights };
const slots = [_]Slot{ .x, .norm, .activation, .route_ids, .route_weights };
const Token = struct {
    buffers: [slots.len]metal.Buffer,
    fn init(ctx: metal.Context, shape: Shape) !Token {
        const sizes = [_]usize{ hidden + 16, shape.input + 16, 3 * shape.stride + 16, 18, 18 };
        var result: Token = undefined;
        var count: usize = 0;
        errdefer for (result.buffers[0..count]) |b| b.deinit();
        for (&result.buffers, sizes) |*b, size| {
            b.* = try ctx.alloc(size);
            count += 1;
        }
        return result;
    }
    fn deinit(self: Token) void {
        for (self.buffers) |b| b.deinit();
    }
    fn buf(self: Token, slot: Slot) metal.Buffer {
        return self.buffers[@backingInt(slot)];
    }
    fn descriptor(self: Token, routes: usize) TokenScratch {
        var s: TokenScratch = .{};
        inline for (slots) |slot| @field(s, @tagName(slot)) = self.buf(slot).address();
        if (routes == 0) {
            s.route_ids = 0;
            s.route_weights = 0;
        }
        return s;
    }
    fn reset(self: Token, shape: Shape, token: usize, routes: usize, pattern: usize, poison: u32) void {
        for (self.buffers) |b| @memset(b.structs(u32), guard);
        fill(self.buf(.norm).floats()[0..shape.input], token * 37 + 1);
        fill(self.buf(.x).floats()[0..hidden], token * 43 + 9);
        for (0..routes) |route| {
            // 0: equal partners, 1: different partners, 2: one route matches
            // and one differs. Shared expert matches in every pattern.
            const key = if (pattern == 0 or (pattern == 2 and route == 0)) token / 2 else token;
            self.buf(.route_ids).structs(u32)[route] = @intCast((key + route) % shared);
            self.buf(.route_weights).floats()[route] = if (route == 0)
                0.173 + @as(f32, @floatFromInt(token)) / 997
            else
                0.637 - @as(f32, @floatFromInt(token)) / 719;
        }
        @memset(self.buf(.activation).structs(u32)[0 .. (routes + 1) * shape.stride], poison);
    }
    fn resident(self: Token, ctx: metal.Context) !void {
        for (self.buffers) |b| try ctx.residentWrite(b);
    }
};
fn fill(values: []f32, seed: usize) void {
    for (values, 0..) |*v, i| {
        const raw = @as(f32, @floatFromInt((i * 23 + seed * 47) % 197)) - 98;
        const scale: f32 = switch (i % 4) {
            0 => 0.00031,
            1 => 0.00397,
            2 => 0.1131,
            else => 0.0373,
        };
        v.* = raw * scale + @as(f32, @floatFromInt(seed % 13)) / 997;
    }
}
fn expectGate(a: Token, b: Token, shape: Shape, token: usize, batch: usize, routes: usize, storage: []const Storage) !void {
    const aw = a.buf(.activation).structs(u32);
    const bw = b.buf(.activation).structs(u32);
    for (0..3) |route| {
        const active = token < batch and route <= routes;
        const expert: usize = if (route < routes) b.buf(.route_ids).structs(u32)[route] else shared;
        const written: usize = if (active) storage[expert * 3].matrix.rows else 0;
        for (0..shape.stride) |i| {
            const index = route * shape.stride + i;
            if (i < written) {
                try std.testing.expectEqual(aw[index], bw[index]);
                try std.testing.expect(std.math.isFinite(@as(f32, @bitCast(bw[index]))));
            } else if (route <= routes) {
                try std.testing.expectEqual(ref_poison, aw[index]);
                try std.testing.expectEqual(new_poison, bw[index]);
            } else {
                try std.testing.expectEqual(guard, aw[index]);
                try std.testing.expectEqual(guard, bw[index]);
            }
        }
    }
    for (aw[3 * shape.stride ..]) |v| try std.testing.expectEqual(guard, v);
    for (bw[3 * shape.stride ..]) |v| try std.testing.expectEqual(guard, v);
    for ([_]Slot{ .x, .norm, .route_ids, .route_weights }) |slot|
        try std.testing.expectEqualSlices(u32, a.buf(slot).structs(u32), b.buf(slot).structs(u32));
}
fn expectDown(a: Token, b: Token) !void {
    // Gate padding intentionally retains different NaN payloads between tokens'
    // reference/candidate buffers. Down must not read or modify that padding.
    for ([_]Slot{ .x, .norm, .route_ids, .route_weights }) |slot|
        try std.testing.expectEqualSlices(u32, a.buf(slot).structs(u32), b.buf(slot).structs(u32));
    for (b.buf(.x).floats()[0..hidden]) |value| try std.testing.expect(std.math.isFinite(value));
    for (b.buf(.x).structs(u32)[hidden..]) |value| try std.testing.expectEqual(guard, value);
}

fn check(shape: Shape, kind: Kind, matched: bool) !void {
    const ctx = try context(matched);
    defer ctx.deinit();
    var storage: [expert_count * 3]Storage = undefined;
    var matrices: usize = 0;
    defer for (storage[0..matrices]) |s| s.deinit();
    const experts = try ctx.alloc(expert_count * @sizeOf(metal.MLP) / 4);
    defer experts.deinit();
    for (&storage, 0..) |*s, index| {
        const expert = index / 3;
        const component = index % 3;
        const quant = switch (kind) {
            .quant_bf16, .quant_f32 => true,
            .dense => false,
            .mixed => expert == shared or (expert == 1 and component != 1) or (expert == 2 and component != 2),
        };
        const cols = if (component == 2) shape.down - (if (expert % 2 == 1) @as(usize, 64) else 0) else shape.input;
        s.* = try Storage.init(ctx, if (component == 2) hidden else shape.gate + expert * 2, cols, quant, if (kind == .quant_f32) 0 else 2, index + 1);
        matrices += 1;
    }
    for (experts.structs(metal.MLP), 0..) |*m, expert| m.* = .{
        .gate = storage[expert * 3].matrix,
        .up = storage[expert * 3 + 1].matrix,
        .down = storage[expert * 3 + 2].matrix,
    };
    var reference: [8]Token = undefined;
    var candidate: [8]Token = undefined;
    var count: usize = 0;
    defer for (0..count) |i| {
        reference[i].deinit();
        candidate[i].deinit();
    };
    for (0..8) |i| {
        reference[i] = try Token.init(ctx, shape);
        candidate[i] = Token.init(ctx, shape) catch |err| {
            reference[i].deinit();
            return err;
        };
        count += 1;
    }
    const refs = try ctx.alloc(8 * @sizeOf(TokenScratch) / 4);
    defer refs.deinit();
    const news = try ctx.alloc(8 * @sizeOf(TokenScratch) / 4);
    defer news.deinit();
    for ([_]usize{ 0, 2 }) |routes| {
        for (0..3) |pattern| {
            for ([_]u32{ 1, 2, 4, 8 }) |tile| {
                for (1..9) |batch| {
                    for (reference, candidate, 0..) |a, b, token| {
                        a.reset(shape, token, routes, pattern, ref_poison);
                        b.reset(shape, token, routes, pattern, new_poison);
                        refs.structs(TokenScratch)[token] = if (token < batch) a.descriptor(routes) else .{};
                        news.structs(TokenScratch)[token] = if (token < batch) b.descriptor(routes) else .{};
                    }
                    const p: metal.Params = .{ .n = @intCast(routes), .m = shared, .k = @intCast(shape.stride), .dim = hidden, .heads = @intCast(batch), .bits = tile };
                    for (storage) |s| try s.resident(ctx);
                    for (reference, candidate) |a, b| {
                        try a.resident(ctx);
                        try b.resident(ctx);
                    }
                    const gate_tiles = (routes + 1) * tiles(shape.stride, tile);
                    try ctx.dispatch("token_batch_gate_up", &.{ experts, refs }, p, batch * gate_tiles * 32, 128);
                    try ctx.dispatch("token_batch_gate_up_weight2", &.{ experts, news }, p, ((batch + 1) / 2 * gate_tiles + 1) * 32, 128);
                    try ctx.finish();
                    for (reference, candidate, 0..) |a, b, token| try expectGate(a, b, shape, token, batch, routes, &storage);
                    for (storage) |s| try s.resident(ctx);
                    for (reference, candidate) |a, b| {
                        try a.resident(ctx);
                        try b.resident(ctx);
                    }
                    const down_tiles = tiles(hidden, tile);
                    try ctx.dispatch("token_batch_down", &.{ experts, refs }, p, batch * down_tiles * 32, 128);
                    try ctx.dispatch("token_batch_down_weight2", &.{ experts, news }, p, ((batch + 1) / 2 * down_tiles + 1) * 32, 128);
                    try ctx.finish();
                    for (reference, candidate, 0..) |a, b, token| {
                        try expectDown(a, b);
                        // Recheck exact gate results and all padding after down;
                        // x is compared separately because it is now residual.
                        const ax = a.buf(.x).structs(u32);
                        const bx = b.buf(.x).structs(u32);
                        try std.testing.expectEqualSlices(u32, ax, bx);
                        if (token >= batch) {
                            var initial: [hidden]f32 = undefined;
                            fill(&initial, token * 43 + 9);
                            try std.testing.expectEqualSlices(f32, &initial, b.buf(.x).floats()[0..hidden]);
                        }
                        try expectGate(a, b, shape, token, batch, routes, &storage);
                    }
                }
            }
        }
    }
}

test "MLP weight2 matched bits: equal/different routes, shared-only, odd batches, tails and dense fallback" {
    for (shapes) |shape| for ([_]Kind{ .quant_bf16, .quant_f32, .dense, .mixed }) |kind| try check(shape, kind, true);
}
test "MLP weight2 legacy mode retains original fallback and residual order" {
    try check(shapes[0], .mixed, false);
}
