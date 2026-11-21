//! Strict, allocation-owned safetensors metadata parser; no IO or weight copy.
//! The caller retains the file/mmap. Tensor offsets are absolute file offsets.
const std = @import("std");

/// Codes 0/1/2 are the floating backend ABI and must remain stable. Unsigned
/// tensors are packed quantized storage, NOT ordinary floating-point weights;
/// callers must unpack/dequantize them before using a floating backend path.
pub const DType = enum(u32) {
    f32 = 0,
    f16 = 1,
    bf16 = 2,
    u8 = 3,
    u32 = 4,
    f8_e4m3 = 5,

    pub fn isFloat(self: DType) bool {
        return switch (self) {
            .f32, .f16, .bf16 => true,
            .u8, .u32, .f8_e4m3 => false,
        };
    }

    pub fn byteSize(self: DType) usize {
        return switch (self) {
            .u8, .f8_e4m3 => 1,
            .f16, .bf16 => 2,
            .f32, .u32 => 4,
        };
    }
};

pub const Tensor = struct {
    dtype: DType,
    shape: []const usize,
    offset: usize,
    len: usize,
};

pub const Header = struct {
    // JSON strings and values are independently owned; shapes and the lookup
    // table live in arena. Neither retains references into the caller's bytes.
    json: std.json.Parsed(std.json.Value),
    arena: std.heap.ArenaAllocator,
    tensors: std.StringHashMapUnmanaged(Tensor),

    pub fn parse(allocator: std.mem.Allocator, file_bytes: []const u8) !Header {
        if (file_bytes.len < 8) return error.TruncatedFile;
        const header_u64 = std.mem.readInt(u64, file_bytes[0..8], .little);
        // Safetensors' standard 100 MB cap prevents metadata allocation bombs.
        if (header_u64 == 0 or header_u64 > 100_000_000) return error.InvalidHeaderLength;
        const header_len = std.math.cast(usize, header_u64) orelse return error.InvalidHeaderLength;
        const data_start = std.math.add(usize, 8, header_len) catch return error.InvalidHeaderLength;
        if (data_start > file_bytes.len) return error.TruncatedFile;
        const bytes = file_bytes[8..data_start];
        if (bytes[0] != '{') return error.InvalidHeader;
        const json = try std.json.parseFromSlice(std.json.Value, allocator, bytes, .{
            .allocate = .alloc_always,
            .duplicate_field_behavior = .@"error",
        });
        errdefer json.deinit();
        if (json.value != .object) return error.InvalidHeader;
        var arena = std.heap.ArenaAllocator.init(allocator);
        errdefer arena.deinit();
        const a = arena.allocator();
        var tensors: std.StringHashMapUnmanaged(Tensor) = .empty;
        var ranges: std.ArrayList(Tensor) = .empty;
        var fields = json.value.object.iterator();
        while (fields.next()) |field| {
            const name = field.key_ptr.*;
            if (std.mem.eql(u8, name, "__metadata__")) continue;
            const tensor = try parseTensor(a, field.value_ptr.*, data_start, file_bytes.len);
            try tensors.put(a, name, tensor);
            if (tensor.len != 0) try ranges.append(a, tensor);
        }
        // Safetensors requires a complete, non-overlapping byte buffer. Sorting
        // gives O(n log n) validation even for a many-expert checkpoint shard.
        std.mem.sort(Tensor, ranges.items, {}, lessThan);
        var cursor = data_start;
        for (ranges.items) |tensor| {
            if (tensor.offset < cursor) return error.OverlappingTensors;
            if (tensor.offset != cursor) return error.InvalidDataLayout;
            cursor = try std.math.add(usize, tensor.offset, tensor.len);
        }
        if (cursor != file_bytes.len) return error.InvalidDataLayout;
        return .{ .json = json, .arena = arena, .tensors = tensors };
    }

    pub fn deinit(self: *Header) void {
        self.json.deinit();
        self.arena.deinit();
        self.* = undefined;
    }

    /// Shape and name storage remains valid until Header.deinit().
    pub fn get(self: *const Header, name: []const u8) ?Tensor {
        return self.tensors.get(name);
    }
};

fn lessThan(_: void, lhs: Tensor, rhs: Tensor) bool {
    return lhs.offset < rhs.offset;
}

fn parseTensor(allocator: std.mem.Allocator, value: std.json.Value, data_start: usize, file_len: usize) !Tensor {
    if (value != .object or value.object.count() != 3) return error.InvalidTensor;
    const dtype_value = value.object.get("dtype") orelse return error.InvalidTensor;
    const shape_value = value.object.get("shape") orelse return error.InvalidTensor;
    const offsets_value = value.object.get("data_offsets") orelse return error.InvalidTensor;
    if (dtype_value != .string) return error.InvalidTensor;
    const dtype: DType = if (std.mem.eql(u8, dtype_value.string, "F32")) .f32 else if (std.mem.eql(u8, dtype_value.string, "F16")) .f16 else if (std.mem.eql(u8, dtype_value.string, "BF16")) .bf16 else if (std.mem.eql(u8, dtype_value.string, "U8")) .u8 else if (std.mem.eql(u8, dtype_value.string, "U32")) .u32 else if (std.mem.eql(u8, dtype_value.string, "F8_E4M3")) .f8_e4m3 else return error.UnsupportedDType;
    if (shape_value != .array or offsets_value != .array or offsets_value.array.items.len != 2)
        return error.InvalidTensor;
    const shape = try allocator.alloc(usize, shape_value.array.items.len);
    var empty = false;
    for (shape_value.array.items, 0..) |dim, i| {
        shape[i] = try natural(dim);
        empty = empty or shape[i] == 0;
    }
    // Scalars have one element; any zero dimension means an empty tensor.
    // Detect emptiness before multiplying so huge but empty shapes don't
    // overflow merely due to the order in which their dimensions occur.
    var elements: usize = if (empty) 0 else 1;
    if (!empty) for (shape) |dim| {
        elements = std.math.mul(usize, elements, dim) catch return error.ShapeOverflow;
    };
    const element_size = dtype.byteSize();
    const expected_len = std.math.mul(usize, elements, element_size) catch return error.ShapeOverflow;
    const begin = try natural(offsets_value.array.items[0]);
    const end = try natural(offsets_value.array.items[1]);
    if (end < begin or end > file_len - data_start) return error.InvalidOffsets;
    if (end - begin != expected_len) return error.TensorSizeMismatch;
    return .{
        .dtype = dtype,
        .shape = shape,
        .offset = std.math.add(usize, data_start, begin) catch return error.InvalidOffsets,
        .len = expected_len,
    };
}

fn natural(value: std.json.Value) !usize {
    // JSON integer spelling is required: fractions/exponents are not shapes or
    // offsets. number_string preserves integers beyond std.json's i64 range.
    return switch (value) {
        .integer => |integer| std.math.cast(usize, integer) orelse return error.InvalidInteger,
        .number_string => |string| std.fmt.parseInt(usize, string, 10) catch return error.InvalidInteger,
        else => error.InvalidInteger,
    };
}

fn fixture(allocator: std.mem.Allocator, json: []const u8, payload_len: usize) ![]u8 {
    const bytes = try allocator.alloc(u8, 8 + json.len + payload_len);
    @memset(bytes, 0);
    std.mem.writeInt(u64, bytes[0..8], @intCast(json.len), .little);
    @memcpy(bytes[8..][0..json.len], json);
    return bytes;
}

fn expectInvalid(expected: anyerror, json: []const u8, payload_len: usize) !void {
    const bytes = try fixture(std.testing.allocator, json, payload_len);
    defer std.testing.allocator.free(bytes);
    try std.testing.expectError(expected, Header.parse(std.testing.allocator, bytes));
}

test "dtype ABI and packed integer storage remain distinct from floats" {
    try std.testing.expectEqual(@as(u32, 0), @backingInt(DType.f32));
    try std.testing.expectEqual(@as(u32, 1), @backingInt(DType.f16));
    try std.testing.expectEqual(@as(u32, 2), @backingInt(DType.bf16));
    try std.testing.expectEqual(@as(u32, 3), @backingInt(DType.u8));
    try std.testing.expectEqual(@as(u32, 4), @backingInt(DType.u32));
    try std.testing.expectEqual(@as(u32, 5), @backingInt(DType.f8_e4m3));
    try std.testing.expect(DType.f32.isFloat());
    try std.testing.expect(DType.f16.isFloat());
    try std.testing.expect(DType.bf16.isFloat());
    try std.testing.expect(!DType.u8.isFloat());
    try std.testing.expect(!DType.u32.isFloat());
    try std.testing.expect(!DType.f8_e4m3.isFloat());
    try std.testing.expectEqual(@as(usize, 1), DType.f8_e4m3.byteSize());
}

test "F8_E4M3 metadata uses one byte elements without scalar float semantics" {
    const json =
        \\{"scales":{"dtype":"F8_E4M3","shape":[3],"data_offsets":[0,3]}}
    ;
    const bytes = try fixture(std.testing.allocator, json, 3);
    defer std.testing.allocator.free(bytes);
    var header = try Header.parse(std.testing.allocator, bytes);
    defer header.deinit();
    const scales = header.get("scales").?;
    try std.testing.expectEqual(DType.f8_e4m3, scales.dtype);
    try std.testing.expectEqual(@as(usize, 3), scales.len);
}

test "packed U8 and U32 tensors use exact one and four byte elements" {
    const json =
        \\{"q8":{"dtype":"U8","shape":[2,3],"data_offsets":[0,6]},
        \\"q4":{"dtype":"U32","shape":[2,2],"data_offsets":[6,22]},
        \\"scalar":{"dtype":"U32","shape":[],"data_offsets":[22,26]},
        \\"empty":{"dtype":"U8","shape":[0,8],"data_offsets":[26,26]}}
    ;
    const bytes = try fixture(std.testing.allocator, json, 26);
    defer std.testing.allocator.free(bytes);
    var header = try Header.parse(std.testing.allocator, bytes);
    defer header.deinit();
    const q8 = header.get("q8").?;
    try std.testing.expectEqual(DType.u8, q8.dtype);
    try std.testing.expectEqual(@as(usize, 6), q8.len);
    try std.testing.expectEqual(8 + json.len, q8.offset);
    const q4 = header.get("q4").?;
    try std.testing.expectEqual(DType.u32, q4.dtype);
    try std.testing.expectEqualSlices(usize, &.{ 2, 2 }, q4.shape);
    try std.testing.expectEqual(@as(usize, 16), q4.len);
    try std.testing.expectEqual(8 + json.len + 6, q4.offset);
    try std.testing.expectEqual(@as(usize, 4), header.get("scalar").?.len);
    try std.testing.expectEqual(@as(usize, 0), header.get("empty").?.len);
}

test "reject incorrect packed integer byte lengths and integer overflow" {
    try expectInvalid(error.TensorSizeMismatch, "{\"x\":{\"dtype\":\"U8\",\"shape\":[2],\"data_offsets\":[0,4]}}", 4);
    try expectInvalid(error.TensorSizeMismatch, "{\"x\":{\"dtype\":\"U32\",\"shape\":[2],\"data_offsets\":[0,4]}}", 4);
    try expectInvalid(error.TensorSizeMismatch, "{\"x\":{\"dtype\":\"U32\",\"shape\":[],\"data_offsets\":[0,2]}}", 2);
    try expectInvalid(error.ShapeOverflow, "{\"x\":{\"dtype\":\"U32\",\"shape\":[9223372036854775807],\"data_offsets\":[0,0]}}", 0);
    try expectInvalid(error.UnsupportedDType, "{\"x\":{\"dtype\":\"U16\",\"shape\":[1],\"data_offsets\":[0,2]}}", 2);
}

test "mixed dtype tensors, absolute offsets, owned metadata, lookup" {
    const json =
        \\{"__metadata__":{"format":"pt"},
        \\"a":{"dtype":"BF16","shape":[2,2],"data_offsets":[0,8]},
        \\"b":{"dtype":"F32","shape":[],"data_offsets":[8,12]},
        \\"c":{"dtype":"F16","shape":[2],"data_offsets":[12,16]}}
    ;
    const bytes = try fixture(std.testing.allocator, json, 16);
    var header = try Header.parse(std.testing.allocator, bytes);
    defer header.deinit();
    @memset(bytes, 0);
    std.testing.allocator.free(bytes);
    const tensor = header.get("a").?;
    try std.testing.expectEqual(DType.bf16, tensor.dtype);
    try std.testing.expectEqualSlices(usize, &.{ 2, 2 }, tensor.shape);
    try std.testing.expectEqual(8 + json.len, tensor.offset);
    try std.testing.expectEqual(@as(usize, 8), tensor.len);
    try std.testing.expectEqual(DType.f32, header.get("b").?.dtype);
    try std.testing.expectEqual(@as(usize, 0), header.get("b").?.shape.len);
    try std.testing.expectEqual(DType.f16, header.get("c").?.dtype);
    try std.testing.expect(header.get("missing") == null);
    try std.testing.expect(header.get("__metadata__") == null);
}

fn allocationProbe(allocator: std.mem.Allocator, bytes: []const u8) !void {
    var header = try Header.parse(allocator, bytes);
    defer header.deinit();
    try std.testing.expectEqual(@as(usize, 4), header.get("x").?.len);
}

test "allocation failures clean up metadata, map and shape storage" {
    const bytes = try fixture(std.testing.allocator, "{\"x\":{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[0,4]}}", 4);
    defer std.testing.allocator.free(bytes);
    try std.testing.checkAllAllocationFailures(std.testing.allocator, allocationProbe, .{bytes});
}

test "empty tensors and zero dimensions" {
    const json = "{\"empty\":{\"dtype\":\"F32\",\"shape\":[0,10],\"data_offsets\":[0,0]}}";
    const bytes = try fixture(std.testing.allocator, json, 0);
    defer std.testing.allocator.free(bytes);
    var header = try Header.parse(std.testing.allocator, bytes);
    defer header.deinit();
    try std.testing.expectEqual(@as(usize, 0), header.get("empty").?.len);
}

test "header prefix truncation and little endian" {
    try std.testing.expectError(error.TruncatedFile, Header.parse(std.testing.allocator, &.{ 0, 1, 2 }));
    var bytes: [10]u8 = @splat(0);
    try std.testing.expectError(error.InvalidHeaderLength, Header.parse(std.testing.allocator, &bytes));
    std.mem.writeInt(u64, bytes[0..8], 3, .little);
    try std.testing.expectError(error.TruncatedFile, Header.parse(std.testing.allocator, &bytes));
    std.mem.writeInt(u64, bytes[0..8], 2, .big);
    try std.testing.expectError(error.InvalidHeaderLength, Header.parse(std.testing.allocator, &bytes));
}

test "reject unsupported dtype, inconsistent size, malformed shape and offsets" {
    try expectInvalid(error.UnsupportedDType, "{\"x\":{\"dtype\":\"I8\",\"shape\":[1],\"data_offsets\":[0,1]}}", 1);
    try expectInvalid(error.TensorSizeMismatch, "{\"x\":{\"dtype\":\"F32\",\"shape\":[2],\"data_offsets\":[0,4]}}", 4);
    try expectInvalid(error.InvalidInteger, "{\"x\":{\"dtype\":\"F32\",\"shape\":[-1],\"data_offsets\":[0,4]}}", 4);
    try expectInvalid(error.InvalidInteger, "{\"x\":{\"dtype\":\"F32\",\"shape\":[1.0],\"data_offsets\":[0,4]}}", 4);
    try expectInvalid(error.InvalidOffsets, "{\"x\":{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[0,8]}}", 4);
    try expectInvalid(error.InvalidOffsets, "{\"x\":{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[4,0]}}", 4);
    try expectInvalid(error.ShapeOverflow, "{\"x\":{\"dtype\":\"F32\",\"shape\":[9223372036854775807,8],\"data_offsets\":[0,0]}}", 0);
}

test "reject overlapping tensors, gaps and unindexed trailing data" {
    try expectInvalid(error.OverlappingTensors,
        \\{"a":{"dtype":"F32","shape":[1],"data_offsets":[0,4]},
        \\"b":{"dtype":"F32","shape":[1],"data_offsets":[2,6]}}
    , 6);
    try expectInvalid(error.InvalidDataLayout, "{\"x\":{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[4,8]}}", 8);
    try expectInvalid(error.InvalidDataLayout, "{\"x\":{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[0,4]}}", 8);
    try expectInvalid(error.DuplicateField, "{\"x\":{},\"x\":{}}", 0);
}
