//! Isolated, headless producer parity gates. Import into a scheduled GPU test
//! root; do not run while another worker owns the GPU. No production imports
//! are changed. BF16 affine buffers use the checkpoint's dtype=2 representation;
//! all fixtures are synthetic, with separate F32 and mixed-affine coverage.
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
const Slot = enum { norm, attention, q, k, v, projection, logits };
const slots = [_]Slot{ .norm, .attention, .q, .k, .v, .projection, .logits };
const rows = [_]usize{ 13, 7, 5, 11, 19 };
const guard: u32 = 0x4b123456;
const reference_poison: u32 = 0x7fc12345;
const candidate_poison: u32 = 0x7fc54321;
const capacity = 1280;

fn context() !metal.Context {
    const source = "#define FLINT_MLX_MATH 1\n" ++
        @embedFile("kernels.metal") ++ "\n" ++
        @embedFile("resident_plan.metal") ++ "\n" ++
        @embedFile("row_tiling.metal") ++ "\n" ++
        @embedFile("token_batch.metal") ++ "\n" ++
        @embedFile("token_batch_weight_reuse.metal");
    return .{ .handle = fm_open(source) orelse {
        std.debug.print("Metal weight-pair test: {s}\n", .{std.mem.span(fm_error())});
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
    fn init(ctx: metal.Context, n: usize, cols: usize, scale_dtype: u32, bias_dtype: u32, seed: usize) !Storage {
        const elements = n * cols;
        const groups = elements / 64;
        const w = try ctx.alloc(elements / 8 + 4);
        errdefer w.deinit();
        const scales = try ctx.alloc(groups + 4);
        errdefer scales.deinit();
        const biases = try ctx.alloc(groups + 4);
        errdefer biases.deinit();
        @memset(w.structs(u32), guard);
        @memset(scales.structs(u32), guard);
        @memset(biases.structs(u32), guard);
        // Nonzero aligned offsets also exercise indirect descriptor addresses.
        @memset(w.structs(u8)[4..][0 .. elements / 2], 0);
        for (0..elements) |i| {
            const mixed = (i *% 1664525 +% seed *% 1013904223) ^ (i >> 3) ^ (i >> 9);
            const code: u8 = @intCast((mixed >> 5) & 15);
            w.structs(u8)[4 + i / 2] |= code << @as(u3, @intCast((i % 2) * 4));
        }
        for (0..groups) |g| {
            const scale = @as(f32, @floatFromInt((g * 13 + seed * 7) % 71 + 1)) / 7139;
            const bias = @as(f32, @floatFromInt((g * 17 + seed * 3) % 89)) / 2791 - 0.021;
            affine(scales, g + (if (scale_dtype == 2) @as(usize, 2) else 1), scale, scale_dtype);
            affine(biases, g + (if (bias_dtype == 2) @as(usize, 4) else 2), bias, bias_dtype);
        }
        return .{ .w = w, .scales = scales, .biases = biases, .matrix = .{
            .weight = w.address() + 4,
            .scales = scales.address() + 4,
            .biases = biases.address() + 8,
            .rows = @intCast(n),
            .cols = @intCast(cols),
            .dtype = 3,
            .bits = 4,
            .group_size = 64,
            .scale_dtype = scale_dtype,
            .bias_dtype = bias_dtype,
            .reserved = 1,
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
const Token = struct {
    buffers: [slots.len]metal.Buffer,
    fn init(ctx: metal.Context) !Token {
        var result: Token = undefined;
        var initialized: usize = 0;
        errdefer for (result.buffers[0..initialized]) |b| b.deinit();
        for (&result.buffers) |*b| {
            b.* = try ctx.alloc(capacity);
            initialized += 1;
        }
        return result;
    }
    fn deinit(self: Token) void {
        for (self.buffers) |b| b.deinit();
    }
    fn buf(self: Token, slot: Slot) metal.Buffer {
        return self.buffers[@backingInt(slot)];
    }
    fn descriptor(self: Token) TokenScratch {
        var s: TokenScratch = .{};
        inline for (slots) |slot| @field(s, @tagName(slot)) = self.buf(slot).address();
        return s;
    }
    fn reset(self: Token, token: usize, poison: u32, input_cols: usize) void {
        for (self.buffers) |b| @memset(b.structs(u32), guard);
        for ([_]Slot{ .norm, .attention }, 0..) |slot, kind| {
            for (self.buf(slot).floats()[0..input_cols], 0..) |*value, i| {
                const raw = @as(f32, @floatFromInt((i * 23 + token * 47 + kind * 31) % 197)) - 98;
                const magnitude: f32 = switch (i % 4) {
                    0 => 0.00031,
                    1 => 0.0397,
                    2 => 1.131,
                    else => 0.373,
                };
                value.* = raw * magnitude + @as(f32, @floatFromInt(token)) / 997;
            }
        }
        for ([_]Slot{ .q, .k, .v, .projection, .logits }, rows) |slot, n|
            @memset(self.buf(slot).structs(u32)[0..n], poison);
    }
    fn resident(self: Token, ctx: metal.Context) !void {
        for (self.buffers) |b| try ctx.residentWrite(b);
    }
};
const Fixture = struct {
    reference: [8]Token,
    candidate: [8]Token,
    reference_scratch: metal.Buffer,
    candidate_scratch: metal.Buffer,
    fn init(ctx: metal.Context) !Fixture {
        var result: Fixture = undefined;
        var count: usize = 0;
        errdefer for (0..count) |i| {
            result.reference[i].deinit();
            result.candidate[i].deinit();
        };
        for (0..8) |i| {
            result.reference[i] = try Token.init(ctx);
            result.candidate[i] = Token.init(ctx) catch |err| {
                result.reference[i].deinit();
                return err;
            };
            count += 1;
        }
        result.reference_scratch = try ctx.alloc(8 * @sizeOf(TokenScratch) / 4);
        errdefer result.reference_scratch.deinit();
        result.candidate_scratch = try ctx.alloc(8 * @sizeOf(TokenScratch) / 4);
        for (0..8) |i| {
            result.reference_scratch.structs(TokenScratch)[i] = result.reference[i].descriptor();
            result.candidate_scratch.structs(TokenScratch)[i] = result.candidate[i].descriptor();
        }
        return result;
    }
    fn deinit(self: Fixture) void {
        for (self.reference, self.candidate) |a, b| {
            a.deinit();
            b.deinit();
        }
        self.reference_scratch.deinit();
        self.candidate_scratch.deinit();
    }
    fn reset(self: Fixture, cols: usize) void {
        for (self.reference, self.candidate, 0..) |a, b, i| {
            a.reset(i, reference_poison, cols);
            b.reset(i, candidate_poison, cols);
            self.reference_scratch.structs(TokenScratch)[i] = a.descriptor();
            self.candidate_scratch.structs(TokenScratch)[i] = b.descriptor();
        }
    }
    fn resident(self: Fixture, ctx: metal.Context) !void {
        for (self.reference, self.candidate) |a, b| {
            try a.resident(ctx);
            try b.resident(ctx);
        }
    }
    fn expect(self: Fixture, batch: usize, op: usize, cols: usize) !void {
        for (self.reference, self.candidate, 0..) |a, b, token| {
            for ([_]Slot{ .norm, .attention }) |slot| {
                try std.testing.expectEqualSlices(u32, a.buf(slot).structs(u32), b.buf(slot).structs(u32));
                for (b.buf(slot).structs(u32)[cols..]) |bits| try std.testing.expectEqual(guard, bits);
            }
            for ([_]Slot{ .q, .k, .v, .projection, .logits }, rows, 0..) |slot, n, index| {
                const written = token < batch and (if (op == 0) index < 3 else index == op + 2);
                if (written) {
                    try std.testing.expectEqualSlices(u32, a.buf(slot).structs(u32)[0..n], b.buf(slot).structs(u32)[0..n]);
                    for (b.buf(slot).floats()[0..n]) |value| try std.testing.expect(std.math.isFinite(value));
                } else {
                    for (a.buf(slot).structs(u32)[0..n]) |bits| try std.testing.expectEqual(reference_poison, bits);
                    for (b.buf(slot).structs(u32)[0..n]) |bits| try std.testing.expectEqual(candidate_poison, bits);
                }
                for (a.buf(slot).structs(u32)[n..]) |bits| try std.testing.expectEqual(guard, bits);
                for (b.buf(slot).structs(u32)[n..]) |bits| try std.testing.expectEqual(guard, bits);
            }
        }
    }
};

test "weight2 matched producers are bitwise original across batches, tiled tails and affine dtypes" {
    const ctx = try context();
    defer ctx.deinit();
    const fixture = try Fixture.init(ctx);
    defer fixture.deinit();
    const descriptors = try ctx.alloc(5 * @sizeOf(metal.Matrix) / 4);
    defer descriptors.deinit();
    const single = try ctx.alloc(@sizeOf(metal.Matrix) / 4);
    defer single.deinit();
    // 64: inactive lanes; 576: lanes with a second segment; 1088: three
    // segments for early lanes. Each Q/K/V has its own valid column count.
    for ([_]usize{ 64, 576, 1088 }) |cols| {
        for ([_][2]u32{ .{ 2, 2 }, .{ 0, 0 }, .{ 2, 0 }, .{ 0, 2 } }) |dtypes| {
            var storage: [5]Storage = undefined;
            var initialized: usize = 0;
            defer for (storage[0..initialized]) |s| s.deinit();
            for (&storage, rows, 0..) |*s, n, index| {
                const offset: usize = switch (index) {
                    1, 4 => 64,
                    2 => 128,
                    else => 0,
                };
                s.* = try Storage.init(ctx, n, cols + offset, dtypes[0], dtypes[1], index + 1);
                initialized += 1;
                descriptors.structs(metal.Matrix)[index] = s.matrix;
            }
            for ([_]u32{ 1, 2, 4, 8 }) |tile| {
                for (1..9) |batch| {
                    for (0..3) |op| {
                        fixture.reset(cols + 128);
                        // Absent odd partners and inactive tokens have no valid
                        // addresses, rather than silently sharing live buffers.
                        for (batch..8) |inactive| {
                            fixture.reference_scratch.structs(TokenScratch)[inactive] = .{};
                            fixture.candidate_scratch.structs(TokenScratch)[inactive] = .{};
                        }
                        try fixture.resident(ctx);
                        for (storage) |s| try s.resident(ctx);
                        var total: usize = 0;
                        var descriptor = descriptors;
                        if (op == 0) {
                            for (rows[0..3]) |n| total += tiles(n, tile);
                        } else {
                            single.structs(metal.Matrix)[0] = storage[op + 2].matrix;
                            descriptor = single;
                            total = tiles(rows[op + 2], tile);
                        }
                        const p: metal.Params = .{ .m = @intCast(batch), .n = tile, .bits = if (op == 0) 0 else @intCast(op - 1) };
                        const old: [:0]const u8 = if (op == 0) "token_batch_qkv" else "token_batch_matvec";
                        const new: [:0]const u8 = if (op == 0) "token_batch_qkv_weight2" else "token_batch_matvec_weight2";
                        try ctx.dispatch(old, &.{ descriptor, fixture.reference_scratch }, p, batch * total * 32, 128);
                        // One extra SIMD tests bounds/partial 128-thread groups.
                        try ctx.dispatch(new, &.{ descriptor, fixture.candidate_scratch }, p, ((batch + 1) / 2 * total + 1) * 32, 128);
                        try ctx.finish();
                        try fixture.expect(batch, op, cols + 128);
                    }
                }
            }
        }
    }
}

test "weight2 rejects malformed parameters, descriptors, and missing token inputs without writes" {
    const ctx = try context();
    defer ctx.deinit();
    const fixture = try Fixture.init(ctx);
    defer fixture.deinit();
    const storage = try Storage.init(ctx, rows[0], 64, 2, 2, 7);
    defer storage.deinit();
    const descriptors = try ctx.alloc(3 * @sizeOf(metal.Matrix) / 4);
    defer descriptors.deinit();
    const valid: metal.Params = .{ .m = 3, .n = 4 };
    const invalid = [_]metal.Params{
        .{ .m = 0, .n = 1 }, .{ .m = 33, .n = 1 },
        .{ .m = 3, .n = 0 }, .{ .m = 3, .n = 3 },
    };
    // No accepted geometry is dispatched with undersized resources.
    for (0..17) |case| {
        fixture.reset(192);
        var p = valid;
        var m = storage.matrix;
        if (case < invalid.len) {
            p = invalid[case];
        } else switch (case) {
            4 => m.reserved = 0,
            5 => m.bits = 8,
            6 => m.group_size = 32,
            7 => m.cols = 0,
            8 => m.cols = 63,
            9 => m.weight = 0,
            10 => m.scales = 0,
            11 => m.biases = 0,
            12 => m.scale_dtype = 3,
            13 => m.bias_dtype = 3,
            14 => m.rows = 0, // empty combined row space
            15 => m.rows = std.math.maxInt(u32), // uint grid overflow
            16 => {
                m.cols = 0xffffffc0;
                m.rows = 1024;
            }, // affine uint index overflow
            else => unreachable,
        }
        @memset(descriptors.structs(metal.Matrix), m);
        try storage.resident(ctx);
        try fixture.resident(ctx);
        try ctx.dispatch("token_batch_qkv_weight2", &.{ descriptors, fixture.candidate_scratch }, p, 128, 128);
        try ctx.dispatch("token_batch_matvec_weight2", &.{ descriptors, fixture.candidate_scratch }, p, 128, 128);
        try ctx.finish();
        try fixture.expect(0, 0, 192);
    }
    // Unsupported matvec selector must be rejected even with a valid matrix.
    @memset(descriptors.structs(metal.Matrix), storage.matrix);
    fixture.reset(192);
    try storage.resident(ctx);
    try fixture.resident(ctx);
    try ctx.dispatch("token_batch_matvec_weight2", &.{ descriptors, fixture.candidate_scratch }, .{ .m = 3, .n = 4, .bits = 2 }, 128, 128);
    try ctx.finish();
    try fixture.expect(0, 0, 192);
    // An invalid descriptor in either later projection rejects the whole QKV
    // launch before touching any producer output.
    for (1..3) |projection| {
        fixture.reset(192);
        @memset(descriptors.structs(metal.Matrix), storage.matrix);
        descriptors.structs(metal.Matrix)[projection].reserved = 0;
        try storage.resident(ctx);
        try fixture.resident(ctx);
        try ctx.dispatch("token_batch_qkv_weight2", &.{ descriptors, fixture.candidate_scratch }, valid, 768, 128);
        try ctx.finish();
        try fixture.expect(0, 0, 192);
    }
    @memset(descriptors.structs(metal.Matrix), storage.matrix);
    // Missing input on either real partner rejects the pair uniformly rather
    // than treating the missing activation as zero or writing its partner.
    for (0..2) |missing| {
        fixture.reset(192);
        fixture.candidate_scratch.structs(TokenScratch)[missing] = .{};
        try storage.resident(ctx);
        try fixture.resident(ctx);
        try ctx.dispatch("token_batch_qkv_weight2", &.{ descriptors, fixture.candidate_scratch }, .{ .m = 2, .n = 4 }, 384, 128);
        try ctx.dispatch("token_batch_matvec_weight2", &.{ descriptors, fixture.candidate_scratch }, .{ .m = 2, .n = 4 }, 128, 128);
        try ctx.dispatch("token_batch_matvec_weight2", &.{ descriptors, fixture.candidate_scratch }, .{ .m = 2, .n = 4, .bits = 1 }, 128, 128);
        try ctx.finish();
        try fixture.expect(0, 0, 192);
    }
}
