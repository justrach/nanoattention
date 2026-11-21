//! Physical storage for Inkling-2 Mini's released packed expert tensors.
//! The execution path binds these bytes as-is, without affine reinterpretation.
const std = @import("std");
const ir = @import("ir.zig");
const weights = @import("weights");
const st = weights.st;

pub const Format = enum { nvfp4, mxfp8 };

pub const Names = struct {
    w13: []const u8,
    w13_scales: []const u8,
    w2: []const u8,
    w2_scales: []const u8,
    w13_amax: ?[]const u8 = null,
    w13_outer_scale_denominator: ?[]const u8 = null,
    w2_amax: ?[]const u8 = null,
    w2_outer_scale_denominator: ?[]const u8 = null,
};

/// Names in the pinned Transformers safetensors checkpoint.
pub fn namesForLayer(a: std.mem.Allocator, layer: usize) !Names {
    const format = try formatForLayer(layer);
    const prefix = try std.fmt.allocPrint(a, "model.llm.layers.{d}.mlp.experts", .{layer});
    return .{
        .w13 = try std.fmt.allocPrint(a, "{s}.w13_weight", .{prefix}),
        .w13_scales = try std.fmt.allocPrint(a, "{s}.w13_weight_scale", .{prefix}),
        .w2 = try std.fmt.allocPrint(a, "{s}.w2_weight", .{prefix}),
        .w2_scales = try std.fmt.allocPrint(a, "{s}.w2_weight_scale", .{prefix}),
        .w13_amax = if (format == .nvfp4) try std.fmt.allocPrint(a, "{s}.w13_weight_amax", .{prefix}) else null,
        .w13_outer_scale_denominator = if (format == .nvfp4) try std.fmt.allocPrint(a, "{s}.w13_weight_outer_scale_denominator", .{prefix}) else null,
        .w2_amax = if (format == .nvfp4) try std.fmt.allocPrint(a, "{s}.w2_weight_amax", .{prefix}) else null,
        .w2_outer_scale_denominator = if (format == .nvfp4) try std.fmt.allocPrint(a, "{s}.w2_weight_outer_scale_denominator", .{prefix}) else null,
    };
}

pub fn declareLayer(b: *ir.Builder, layer: usize, experts: u32, hidden: u32, intermediate: u32) !struct { gate_up: ir.PackedProjection, down: ir.PackedProjection } {
    const n = try namesForLayer(b.allocator(), layer);
    const format = try formatForLayer(layer);
    return .{
        .gate_up = try projection(b, format, n.w13, n.w13_scales, n.w13_amax, n.w13_outer_scale_denominator, experts, 2 * intermediate, hidden),
        .down = try projection(b, format, n.w2, n.w2_scales, n.w2_amax, n.w2_outer_scale_denominator, experts, hidden, intermediate),
    };
}

fn projection(b: *ir.Builder, format: Format, w: []const u8, s: []const u8, a: ?[]const u8, d: ?[]const u8, experts: u32, n: u32, k: u32) !ir.PackedProjection {
    return switch (format) {
        .nvfp4 => .{
            .payload = try b.rawWeight(w, &.{ experts, n, k / 2 }, .u8),
            .scales = try b.rawWeight(s, &.{ experts, n, k / 16 }, .f8_e4m3),
            .amax = try b.rawWeight(a.?, &.{experts}, .f32),
            .denominator = try b.rawWeight(d.?, &.{1}, .f32),
        },
        .mxfp8 => .{
            .payload = try b.rawWeight(w, &.{ experts, n, k }, .f8_e4m3),
            .scales = try b.rawWeight(s, &.{ experts, n, k / 32 }, .u8),
        },
    };
}

/// Raw mapped tensors for one layer. Packed buffers stay in checkpoint form;
/// specialized kernels, not Store.get's affine-quant path, consume them.
pub const Bank = struct {
    format: Format,
    w13: weights.Weight,
    w13_scales: weights.Weight,
    w2: weights.Weight,
    w2_scales: weights.Weight,
    w13_amax: ?weights.Weight,
    w13_outer_scale_denominator: ?weights.Weight,
    w2_amax: ?weights.Weight,
    w2_outer_scale_denominator: ?weights.Weight,
};

fn required(store: *const weights.Store, name: []const u8) !weights.Weight {
    return (try store.find(name)) orelse error.MissingPackedTensor;
}
fn optional(store: *const weights.Store, name: ?[]const u8) !?weights.Weight {
    const key = name orelse return null;
    return (try store.find(key)) orelse error.MissingPackedTensor;
}

/// Bind raw checkpoint tensors with explicit format validation. No tensor is
/// expanded, affine-dequantized, or silently converted during binding.
pub fn bindLayer(store: *const weights.Store, layer: usize, names: Names) !Bank {
    const w13 = try required(store, names.w13);
    const w13_scales = try required(store, names.w13_scales);
    const w2 = try required(store, names.w2);
    const w2_scales = try required(store, names.w2_scales);
    const w13_amax = try optional(store, names.w13_amax);
    const w13_denominator = try optional(store, names.w13_outer_scale_denominator);
    const w2_amax = try optional(store, names.w2_amax);
    const w2_denominator = try optional(store, names.w2_outer_scale_denominator);
    const format = try validateLayer(
        layer,
        w13.tensor,
        w13_scales.tensor,
        w2.tensor,
        w2_scales.tensor,
        if (w13_amax) |value| value.tensor else null,
        if (w13_denominator) |value| value.tensor else null,
        if (w2_amax) |value| value.tensor else null,
        if (w2_denominator) |value| value.tensor else null,
    );
    return .{
        .format = format,
        .w13 = w13,
        .w13_scales = w13_scales,
        .w2 = w2,
        .w2_scales = w2_scales,
        .w13_amax = w13_amax,
        .w13_outer_scale_denominator = w13_denominator,
        .w2_amax = w2_amax,
        .w2_outer_scale_denominator = w2_denominator,
    };
}

pub fn formatForLayer(layer: usize) !Format {
    if (layer < 20) return .nvfp4;
    if (layer < 24) return .mxfp8;
    return error.InvalidExpertLayer;
}

fn expect(value: st.Tensor, dtype: st.DType, shape: []const usize) !void {
    if (value.dtype != dtype) return error.PackedTensorDTypeMismatch;
    if (!std.mem.eql(usize, value.shape, shape)) return error.PackedTensorShapeMismatch;
}

/// Validate one released expert bank. NVFP4 stores two E2M1 values per U8
/// byte and uses 16-value block scales; its global scale metadata is required.
/// MXFP8 stores E4M3 values and U8 E8M0 scales in 32-value blocks.
pub fn validateLayer(
    layer: usize,
    w13: st.Tensor,
    w13_scales: st.Tensor,
    w2: st.Tensor,
    w2_scales: st.Tensor,
    w13_amax: ?st.Tensor,
    w13_outer_scale_denominator: ?st.Tensor,
    w2_amax: ?st.Tensor,
    w2_outer_scale_denominator: ?st.Tensor,
) !Format {
    return switch (try formatForLayer(layer)) {
        .nvfp4 => blk: {
            try expect(w13, .u8, &.{ 128, 2304, 1152 });
            try expect(w13_scales, .f8_e4m3, &.{ 128, 2304, 144 });
            try expect(w2, .u8, &.{ 128, 2304, 576 });
            try expect(w2_scales, .f8_e4m3, &.{ 128, 2304, 72 });
            try expect(w13_amax orelse return error.MissingNVFP4Metadata, .f32, &.{128});
            try expect(w13_outer_scale_denominator orelse return error.MissingNVFP4Metadata, .f32, &.{1});
            try expect(w2_amax orelse return error.MissingNVFP4Metadata, .f32, &.{128});
            try expect(w2_outer_scale_denominator orelse return error.MissingNVFP4Metadata, .f32, &.{1});
            break :blk .nvfp4;
        },
        .mxfp8 => blk: {
            if (w13_amax != null or w13_outer_scale_denominator != null or w2_amax != null or w2_outer_scale_denominator != null)
                return error.UnexpectedNVFP4Metadata;
            try expect(w13, .f8_e4m3, &.{ 128, 2304, 2304 });
            try expect(w13_scales, .u8, &.{ 128, 2304, 72 });
            try expect(w2, .f8_e4m3, &.{ 128, 2304, 1152 });
            try expect(w2_scales, .u8, &.{ 128, 2304, 36 });
            break :blk .mxfp8;
        },
    };
}

fn fixtureTensor(dtype: st.DType, shape: []const usize) st.Tensor {
    return .{ .dtype = dtype, .shape = shape, .offset = 0, .len = 0 };
}

test "released Inkling expert layer format split is explicit" {
    try std.testing.expectEqual(Format.nvfp4, try formatForLayer(0));
    try std.testing.expectEqual(Format.nvfp4, try formatForLayer(19));
    try std.testing.expectEqual(Format.mxfp8, try formatForLayer(20));
    try std.testing.expectEqual(Format.mxfp8, try formatForLayer(23));
    try std.testing.expectError(error.InvalidExpertLayer, formatForLayer(24));
}

test "packed expert names match the pinned checkpoint" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const nv = try namesForLayer(arena.allocator(), 0);
    try std.testing.expectEqualStrings("model.llm.layers.0.mlp.experts.w13_weight", nv.w13);
    try std.testing.expectEqualStrings("model.llm.layers.0.mlp.experts.w13_weight_scale", nv.w13_scales);
    try std.testing.expectEqualStrings("model.llm.layers.0.mlp.experts.w13_weight_amax", nv.w13_amax.?);
    try std.testing.expectEqualStrings("model.llm.layers.0.mlp.experts.w13_weight_outer_scale_denominator", nv.w13_outer_scale_denominator.?);
    const mx = try namesForLayer(arena.allocator(), 23);
    try std.testing.expectEqualStrings("model.llm.layers.23.mlp.experts.w2_weight", mx.w2);
    try std.testing.expect(mx.w13_amax == null and mx.w13_outer_scale_denominator == null and mx.w2_amax == null and mx.w2_outer_scale_denominator == null);
}

test "validate packed expert tensor dtypes and physical shapes" {
    const nvfp4 = try validateLayer(
        0,
        fixtureTensor(.u8, &.{ 128, 2304, 1152 }),
        fixtureTensor(.f8_e4m3, &.{ 128, 2304, 144 }),
        fixtureTensor(.u8, &.{ 128, 2304, 576 }),
        fixtureTensor(.f8_e4m3, &.{ 128, 2304, 72 }),
        fixtureTensor(.f32, &.{128}),
        fixtureTensor(.f32, &.{1}),
        fixtureTensor(.f32, &.{128}),
        fixtureTensor(.f32, &.{1}),
    );
    try std.testing.expectEqual(Format.nvfp4, nvfp4);

    const mxfp8 = try validateLayer(
        20,
        fixtureTensor(.f8_e4m3, &.{ 128, 2304, 2304 }),
        fixtureTensor(.u8, &.{ 128, 2304, 72 }),
        fixtureTensor(.f8_e4m3, &.{ 128, 2304, 1152 }),
        fixtureTensor(.u8, &.{ 128, 2304, 36 }),
        null,
        null,
        null,
        null,
    );
    try std.testing.expectEqual(Format.mxfp8, mxfp8);
}

test "packed expert validation rejects wrong storage rather than requantizing" {
    try std.testing.expectError(error.PackedTensorDTypeMismatch, validateLayer(
        20,
        fixtureTensor(.u8, &.{ 128, 2304, 2304 }),
        fixtureTensor(.u8, &.{ 128, 2304, 72 }),
        fixtureTensor(.f8_e4m3, &.{ 128, 2304, 1152 }),
        fixtureTensor(.u8, &.{ 128, 2304, 36 }),
        null,
        null,
        null,
        null,
    ));
    try std.testing.expectError(error.PackedTensorShapeMismatch, validateLayer(
        20,
        fixtureTensor(.f8_e4m3, &.{ 128, 2304, 1152 }),
        fixtureTensor(.u8, &.{ 128, 2304, 72 }),
        fixtureTensor(.f8_e4m3, &.{ 128, 2304, 1152 }),
        fixtureTensor(.u8, &.{ 128, 2304, 36 }),
        null,
        null,
        null,
        null,
    ));
    try std.testing.expectError(error.UnexpectedNVFP4Metadata, validateLayer(
        20,
        fixtureTensor(.f8_e4m3, &.{ 128, 2304, 2304 }),
        fixtureTensor(.u8, &.{ 128, 2304, 72 }),
        fixtureTensor(.f8_e4m3, &.{ 128, 2304, 1152 }),
        fixtureTensor(.u8, &.{ 128, 2304, 36 }),
        fixtureTensor(.f32, &.{128}),
        null,
        null,
        null,
    ));
    try std.testing.expectError(error.MissingNVFP4Metadata, validateLayer(
        0,
        fixtureTensor(.u8, &.{ 128, 2304, 1152 }),
        fixtureTensor(.f8_e4m3, &.{ 128, 2304, 144 }),
        fixtureTensor(.u8, &.{ 128, 2304, 576 }),
        fixtureTensor(.f8_e4m3, &.{ 128, 2304, 72 }),
        null,
        null,
        null,
        null,
    ));
    try std.testing.expectError(error.MissingNVFP4Metadata, validateLayer(
        0,
        fixtureTensor(.u8, &.{ 128, 2304, 1152 }),
        fixtureTensor(.f8_e4m3, &.{ 128, 2304, 144 }),
        fixtureTensor(.u8, &.{ 128, 2304, 576 }),
        fixtureTensor(.f8_e4m3, &.{ 128, 2304, 72 }),
        fixtureTensor(.f32, &.{128}),
        fixtureTensor(.f32, &.{1}),
        null,
        null,
    ));
}
