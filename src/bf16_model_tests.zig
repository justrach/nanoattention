const std = @import("std");
const metal = @import("metal.zig");
const Model = @import("model.zig").Model;

// Minimal inert Model fixture: only fields used by the pre-encoding geometry
// guards/allocation setter are initialized. No checkpoint or inference executes.
test "BF16 stripe tuning grows scratch before launch and rejects unsafe direct overrides" {
    const ctx = try metal.Context.initMatched(true);
    defer ctx.deinit();
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const initial = try ctx.alloc(32 * 512 * 130);
    defer initial.deinit();
    var model: Model = undefined;
    model.ctx = ctx;
    model.arena = &arena;
    model.buffers = .empty;
    model.batch = null;
    defer for (model.buffers.items) |buffer| buffer.deinit();
    model.cfg = undefined;
    model.cfg.num_attention_heads = 32;
    model.scratch = undefined;
    model.scratch.attention_scores = initial;
    model.position = 0;
    model.failed = false;
    model.kv_bf16 = true;
    model.mlx_attention = true;
    model.bf16_blocks = 1024;
    // Reject before reading any other fixture fields or encoding GPU work.
    try std.testing.expectError(error.Bf16ScratchTooSmall, model.forward(0));
    try std.testing.expect(!model.failed);
    try std.testing.expectError(error.InvalidBf16Blocks, model.setBf16Blocks(64));
    model.position = 1;
    try std.testing.expectError(error.Bf16TuningRequiresFreshSequence, model.setBf16Blocks(1024));
    model.position = 0;
    try model.setBf16Blocks(1024);
    try std.testing.expect(model.scratch.attention_scores.len >= 32 * 1024 * 130 * 4);
    try std.testing.expectEqual(@as(usize, 1), model.buffers.items.len);
    try model.setBf16Blocks(512);
    try std.testing.expectEqual(@as(usize, 1), model.buffers.items.len);
    model.head_row_tile = 0;
    model.qkv_row_tile = 3;
    model.gate_row_tile = 0;
    model.projection_row_tile = 0;
    model.down_row_tile = 0;
    try std.testing.expectError(error.InvalidStagedRowTile, model.forward(0));
    model.kv_bf16 = false;
    try std.testing.expectError(error.Bf16KvRequiresMatchedStaged, model.setBf16Blocks(128));
}
