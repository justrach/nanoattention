//! Run-time performance knobs. Each knob selects among compiled-in variants;
//! no value changes any token row's arithmetic (docs/QD.md admission checks that).
//! Defaults are the reference executor: one dispatch per op, no fusion.
const std = @import("std");

pub const Knobs = struct {
    /// Maximum token rows per forward (prefill chunk). Decode uses 1.
    batch: u32 = 64,
    /// Affine 4-bit linear: output rows and token rows per SIMD group,
    /// separately for single-row (decode) and multi-row (prefill) forwards.
    decode_rows: u32 = 2,
    decode_tokens: u32 = 1,
    prefill_rows: u32 = 2,
    prefill_tokens: u32 = 4,
    /// Threads per threadgroup for the 4-bit linear kernels.
    linear_group: u32 = 128,
    /// Fusions (src/mega/plan.zig). All off = the reference executor.
    fuse_projections: bool = false,
    fuse_glu: bool = false,
    fuse_norms: bool = false,
    fuse_conv: bool = false,
    fuse_gate: bool = false,
    // ---- decode search (decode objective) ----
    /// Single-row attention: 0 = mk_attention (reference); 2 = split into a
    /// parallel scores launch and the online-softmax chain with its max/exp
    /// computed 32 steps at a time (decode.metal). 1 (a per-step chain, the
    /// stepping stone) was removed; see benchmarks/lfm3-decode-search.md.
    decode_attention: u32 = 0,
    /// q/k per-head norms, both ropes and the K/V stores of an attention layer in one launch.
    fuse_attn_prep: bool = false,
    /// Argmax: 0 = mk_argmax (one threadgroup per row), 1 = two-stage (decode.metal).
    decode_argmax: u32 = 0,

    // ---- prefill objective (prefill search) ----
    /// Multi-row affine 4-bit linears (plain, GLU, projections) through
    /// mk_q4_prep plus the matrix-matrix kernels of src/mega/prefill.metal.
    /// Single-row forwards (decode, the head) are unchanged.
    prefill_mm: bool = false,
    /// Output rows and token rows per SIMD group of the matrix-matrix kernels,
    /// and whether unpacked codes are held as half (exact) or float. Only the
    /// combinations in `mm_tiles` are compiled (the prefill-mm elite's; the
    /// climb's other tiles were removed, benchmarks/lfm3-prefill-search.md).
    mm_rows: u32 = 2,
    mm_tokens: u32 = 8,
    mm_half: bool = true,
    /// Threads per threadgroup of the matrix-matrix kernels.
    mm_group: u32 = 128,

    // ---- prefill-mlx (exact) ----
    /// Multi-row forwards run the fused gated conv as two launches
    /// (mk_gated_conv_store, mk_gated_conv_rows: every row's product stored,
    /// then every (row, channel) convolved in parallel) instead of one thread
    /// per channel walking the rows. Same arithmetic; needs fuse_conv.
    conv_rows: bool = false,
    /// Rows without logits (every row of a prefill forward but the last) stop after
    /// the program's last store to a persistent state (exec.trimmedHead): for LFM3,
    /// the last layer's conv output, out_proj and feed-forward run for the logits
    /// rows only, as in stock MLX's chunked prefill. Same arithmetic per row run.
    head_trim: bool = false,
    // ---- prefill-mlx (mlx-arith tier) ----
    /// NOT the reference arithmetic: multi-row affine 4-bit linears (group a
    /// multiple of 32, at least MLX's vector limit of rows) through the port of
    /// MLX's affine_qmm_t / qmm_t_splitk (mlx.metal), bit-identical to stock
    /// MLX's float32 mx.quantized_matmul at the same batch (tools/nano_mlx_ops.py).
    /// Admitted only in the mlx-arith tier (docs/QD.md); decode is unchanged.
    mlx_linear: bool = false,
    /// NOT the reference arithmetic: multi-row attention (more than 8 rows, head
    /// dim 128) through the port of MLX's steel attention (mlx.metal),
    /// bit-identical to stock MLX's float32 mx.fast.scaled_dot_product_attention
    /// over the same keys (tools/nano_mlx_ops.py). mlx-arith tier only.
    mlx_attention: bool = false,
    /// With mlx_linear and fuse_projections / fuse_glu: unsplit MLX-arithmetic
    /// products that share an input (q/k/v projections; w1 and w3) in one launch
    /// (mk_mlx_qmm_multi): every tile is mk_mlx_qmm_t's, so the bits are the same.
    mlx_multi: bool = false,
    // ---- c2-prefill-mlx (prefill, mlx-arith tier) ----
    /// NOT the reference arithmetic: with mlx_linear, the products stock MLX runs
    /// unsplit (plain and, with mlx_multi, multi-matrix) through mk_mlx2_qmm /
    /// mk_mlx2_multi (mlx2.metal): 64 x 16 tiles, x read straight from device memory,
    /// double-buffered dequantized weights. Each output is the same ascending-k fma
    /// chain as stock's affine_qmm_t, so the bits are stock MLX's
    /// (tools/nano_mlx_ops.py check --variants mlx,mlx2). mlx-arith tier only.
    mlx_qmm2: bool = false,
    /// With mlx_linear and fuse_glu: the gate and up products (both unsplit in stock
    /// MLX) and the GLU product in one launch (mk_mlx2_glu): equal bit for bit to
    /// mk_mlx_qmm_t twice then mk_binary. mlx-arith tier only.
    mlx_glu2: bool = false,
    /// NOT the reference arithmetic: with mlx_attention, multi-row attention through
    /// mk_mlx2_attention[_gated] (mlx2.metal): mk_mlx_attention's arithmetic step for
    /// step, scheduled heaviest query blocks first, queries held in registers, separate K
    /// and V tiles (two barriers per key block instead of four). Bits are stock MLX's
    /// (tools/nano_mlx_ops.py check --variants ..mlx2). mlx-arith tier only.
    mlx_attention2: bool = false,
    /// With conv_rows: mk_gated_conv_rows2 (mlx2.metal), the same arithmetic with the
    /// ring slot advanced per tap instead of a 64-bit modulo per tap. Exact tier.
    conv_rows2: bool = false,

    /// (rows, tokens, half) of every compiled mk_mm_q4 / mk_mm_glu_q4 variant.
    pub const mm_tiles = [_]struct { u32, u32, bool }{.{ 2, 8, true }};

    pub fn fusion(self: Knobs) @import("plan.zig").Fusion {
        return .{ .projections = self.fuse_projections, .glu = self.fuse_glu, .norms = self.fuse_norms, .conv = self.fuse_conv, .gate = self.fuse_gate, .attn_prep = self.fuse_attn_prep };
    }

    pub fn validate(self: Knobs) !void {
        if (self.batch == 0 or self.batch > 4096) return error.InvalidBatch;
        for ([_]u32{ self.decode_rows, self.prefill_rows }) |r| if (r != 1 and r != 2 and r != 4 and r != 8) return error.InvalidLinearRows;
        for ([_]u32{ self.decode_tokens, self.prefill_tokens }) |t| if (t != 1 and t != 2 and t != 4) return error.InvalidLinearTokens;
        if (self.linear_group == 0 or self.linear_group % 32 != 0 or self.linear_group > 1024) return error.InvalidLinearGroup;
        // prefill objective (prefill search)
        var tile_ok = false;
        for (mm_tiles) |t| tile_ok = tile_ok or (t[0] == self.mm_rows and t[1] == self.mm_tokens and t[2] == self.mm_half);
        if (!tile_ok) return error.InvalidMmTile;
        if (self.mm_group == 0 or self.mm_group % 32 != 0 or self.mm_group > 1024) return error.InvalidMmGroup;
        if (self.decode_attention != 0 and self.decode_attention != 2) return error.InvalidDecodeAttention;
        if (self.decode_argmax > 1) return error.InvalidDecodeArgmax;
    }

    /// Apply `name=value` (flag form) to the knob of that name.
    pub fn set(self: *Knobs, name: []const u8, value: []const u8) !void {
        const info = @typeInfo(Knobs).@"struct";
        inline for (info.field_names, info.field_types) |field, T| {
            if (std.mem.eql(u8, field, name)) {
                switch (@typeInfo(T)) {
                    .int => @field(self, field) = try std.fmt.parseInt(T, value, 10),
                    .bool => @field(self, field) = std.mem.eql(u8, value, "1") or std.mem.eql(u8, value, "true"),
                    .@"enum" => @field(self, field) = std.meta.stringToEnum(T, value) orelse return error.InvalidKnobValue,
                    else => @compileError("unsupported knob type"),
                }
                return;
            }
        }
        return error.UnknownKnob;
    }

    /// A JSON object of knob values; unknown names fail closed.
    pub fn parseJson(a: std.mem.Allocator, bytes: []const u8) !Knobs {
        const parsed = try std.json.parseFromSliceLeaky(Knobs, a, bytes, .{});
        try parsed.validate();
        return parsed;
    }
};

test "knob flags and JSON configuration select variants and fail closed" {
    var k: Knobs = .{};
    try k.set("decode_rows", "4");
    try k.set("fuse_glu", "true");
    try std.testing.expect(k.fusion().glu and !k.fusion().norms);
    try std.testing.expectEqual(@as(u32, 4), k.decode_rows);
    try std.testing.expectError(error.UnknownKnob, k.set("typo", "1"));
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const j = try Knobs.parseJson(arena.allocator(), "{\"batch\":16,\"prefill_tokens\":2}");
    try std.testing.expectEqual(@as(u32, 16), j.batch);
    try std.testing.expectEqual(@as(u32, 2), j.prefill_tokens);
    try std.testing.expectError(error.UnknownField, Knobs.parseJson(arena.allocator(), "{\"nope\":1}"));
    try std.testing.expectError(error.InvalidLinearRows, Knobs.parseJson(arena.allocator(), "{\"decode_rows\":3}"));
}
