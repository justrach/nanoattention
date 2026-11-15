const std = @import("std");
const metal = @import("metal.zig");
const st = @import("safetensors.zig");

pub const Quant = struct {
    scales: metal.Buffer,
    biases: metal.Buffer,
    scale_offset: u64,
    bias_offset: u64,
    scale_dtype: u32,
    bias_dtype: u32,
    bits: u32,
    group_size: u32,
};
pub const Weight = struct {
    buffer: metal.Buffer,
    tensor: st.Tensor,
    quant: ?Quant = null,

    pub fn params(self: Weight) metal.Params {
        var p: metal.Params = .{ .offset = self.tensor.offset, .dtype = @backingInt(self.tensor.dtype) };
        if (self.quant) |q| {
            p.scale_offset = q.scale_offset;
            p.bias_offset = q.bias_offset;
            p.scale_dtype = q.scale_dtype;
            p.bias_dtype = q.bias_dtype;
            p.bits = q.bits;
            p.k = q.group_size;
        }
        return p;
    }
    pub fn matrix(self: Weight) metal.Matrix {
        var m: metal.Matrix = .{
            .weight = self.buffer.address() + self.tensor.offset,
            .rows = @intCast(self.tensor.shape[0]),
            .cols = @intCast(self.tensor.shape[1]),
            .dtype = @backingInt(self.tensor.dtype),
        };
        if (self.quant) |q| {
            m.scales = q.scales.address() + q.scale_offset;
            m.biases = q.biases.address() + q.bias_offset;
            m.bits = q.bits;
            m.group_size = q.group_size;
            m.scale_dtype = q.scale_dtype;
            m.bias_dtype = q.bias_dtype;
            m.reserved = @intFromBool(q.bits == 4 and q.group_size == 64 and m.cols % 64 == 0 and m.weight % 4 == 0);
        }
        return m;
    }
    pub fn scalar(self: Weight, i: usize) f32 {
        const bytes: [*]const u8 = @ptrCast(self.buffer.floats().ptr);
        const p = bytes + self.tensor.offset;
        return switch (self.tensor.dtype) {
            .f32 => @bitCast(std.mem.readInt(u32, p[i * 4 ..][0..4], .little)),
            .f16 => @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, p[i * 2 ..][0..2], .little)))),
            .bf16 => @bitCast(@as(u32, std.mem.readInt(u16, p[i * 2 ..][0..2], .little)) << 16),
            .u8, .u32 => unreachable,
        };
    }
    pub fn expert(self: Weight, index: usize) Weight {
        var w = self;
        const elements = self.tensor.shape[1] * self.tensor.shape[2];
        w.tensor.shape = self.tensor.shape[1..];
        if (w.quant) |*q| {
            w.tensor.offset += index * elements * q.bits / 8;
            const groups = elements / q.group_size;
            q.scale_offset += index * groups * floatBytes(q.scale_dtype);
            q.bias_offset += index * groups * floatBytes(q.bias_dtype);
        } else {
            w.tensor.offset += index * elements * floatBytes(@backingInt(w.tensor.dtype));
        }
        return w;
    }
};
fn floatBytes(dtype: u32) usize {
    return if (dtype == 0) 4 else 2;
}
fn floating(w: Weight) bool {
    return w.tensor.dtype == .f32 or w.tensor.dtype == .f16 or w.tensor.dtype == .bf16;
}

const Shard = struct {
    mapping: metal.Mapping,
    buffer: metal.Buffer,
    header: st.Header,
    fn deinit(self: *Shard) void {
        self.header.deinit();
        self.buffer.deinit();
        self.mapping.deinit();
    }
};

pub const Store = struct {
    allocator: std.mem.Allocator,
    shards: std.ArrayList(Shard) = .empty,
    bytes: usize = 0,
    group_size: u32,

    pub fn init(allocator: std.mem.Allocator, ctx: metal.Context, directory: []const u8, group_size: u32) !Store {
        var self: Store = .{ .allocator = allocator, .group_size = group_size };
        errdefer self.deinit();
        const index_path = try std.fs.path.join(allocator, &.{ directory, "model.safetensors.index.json" });
        defer allocator.free(index_path);
        const index = metal.Mapping.open(allocator, index_path) catch {
            self.add(ctx, directory, "model.safetensors") catch |err| {
                if (err != error.FileMappingFailed) return err;
                // Official MLX export uses contiguous model-00000.safetensors shards without an index.
                var i: usize = 0;
                while (i < 1024) : (i += 1) {
                    var name_buf: [64]u8 = undefined;
                    const name = try std.fmt.bufPrint(&name_buf, "model-{d:0>5}.safetensors", .{i});
                    self.add(ctx, directory, name) catch |shard_error| {
                        if (shard_error == error.FileMappingFailed and i > 0) break;
                        return shard_error;
                    };
                }
            };
            return self;
        };
        defer index.deinit();
        const parsed = try std.json.parseFromSlice(std.json.Value, allocator, index.bytes, .{});
        defer parsed.deinit();
        if (parsed.value != .object) return error.InvalidWeightIndex;
        const map = parsed.value.object.get("weight_map") orelse return error.InvalidWeightIndex;
        if (map != .object or map.object.count() == 0) return error.InvalidWeightIndex;
        var names: std.StringHashMap(void) = .init(allocator);
        defer names.deinit();
        var iter = map.object.iterator();
        while (iter.next()) |entry| {
            if (entry.value_ptr.* != .string) return error.InvalidWeightIndex;
            const name = entry.value_ptr.string;
            if (name.len == 0 or std.mem.indexOfAny(u8, name, "/\\") != null or
                std.mem.eql(u8, name, ".") or std.mem.eql(u8, name, "..")) return error.UnsafeShardPath;
            const result = try names.getOrPut(name);
            if (!result.found_existing) try self.add(ctx, directory, name);
        }
        return self;
    }
    fn add(self: *Store, ctx: metal.Context, directory: []const u8, name: []const u8) !void {
        const path = try std.fs.path.join(self.allocator, &.{ directory, name });
        defer self.allocator.free(path);
        const mapping = try metal.Mapping.open(self.allocator, path);
        errdefer mapping.deinit();
        var header = try st.Header.parse(self.allocator, mapping.bytes);
        errdefer header.deinit();
        const buffer = try ctx.wrap(mapping);
        errdefer buffer.deinit();
        try self.shards.append(self.allocator, .{ .mapping = mapping, .header = header, .buffer = buffer });
        self.bytes += mapping.bytes.len;
    }
    pub fn resident(self: *const Store, ctx: metal.Context) !void {
        for (self.shards.items) |shard| try ctx.resident(shard.buffer);
    }
    pub fn deinit(self: *Store) void {
        for (self.shards.items) |*shard| shard.deinit();
        self.shards.deinit(self.allocator);
    }
    pub fn find(self: *const Store, name: []const u8) !?Weight {
        var result: ?Weight = null;
        for (self.shards.items) |*shard| {
            if (shard.header.get(name)) |tensor| {
                if (result != null) return error.DuplicateWeight;
                result = .{ .buffer = shard.buffer, .tensor = tensor };
            }
        }
        return result;
    }
    pub fn get(self: *const Store, name: []const u8, shape: []const usize) !Weight {
        var w = (try self.find(name)) orelse {
            std.debug.print("Missing checkpoint tensor: {s}\n", .{name});
            return error.MissingWeight;
        };
        if (floating(w)) {
            if (!std.mem.eql(usize, shape, w.tensor.shape)) return error.WeightShapeMismatch;
            return w;
        }
        if (self.group_size == 0 or shape.len < 2 or shape.len != w.tensor.shape.len or !std.mem.endsWith(u8, name, ".weight")) return error.InvalidQuantizedWeight;
        const last = shape.len - 1;
        if (!std.mem.eql(usize, shape[0..last], w.tensor.shape[0..last])) return error.WeightShapeMismatch;
        const row_bytes = w.tensor.shape[last] * @as(usize, if (w.tensor.dtype == .u8) 1 else 4);
        const bits = row_bytes * 8 / shape[last];
        if ((bits != 4 and bits != 8) or row_bytes * 8 != shape[last] * bits or shape[last] % self.group_size != 0) return error.InvalidQuantizedWeight;
        var buf: [256]u8 = undefined;
        const prefix = name[0 .. name.len - ".weight".len];
        const scales = (try self.find(try std.fmt.bufPrint(&buf, "{s}.scales", .{prefix}))) orelse return error.MissingQuantizationScales;
        const biases = (try self.find(try std.fmt.bufPrint(&buf, "{s}.biases", .{prefix}))) orelse return error.MissingQuantizationBiases;
        if (!floating(scales) or !floating(biases) or scales.tensor.shape.len != shape.len or !std.mem.eql(usize, scales.tensor.shape, biases.tensor.shape) or
            !std.mem.eql(usize, shape[0..last], scales.tensor.shape[0..last]) or scales.tensor.shape[last] != shape[last] / self.group_size) return error.InvalidQuantizationScales;
        w.quant = .{
            .scales = scales.buffer,
            .biases = biases.buffer,
            .scale_offset = scales.tensor.offset,
            .bias_offset = biases.tensor.offset,
            .scale_dtype = @backingInt(scales.tensor.dtype),
            .bias_dtype = @backingInt(biases.tensor.dtype),
            .bits = @intCast(bits),
            .group_size = self.group_size,
        };
        w.tensor.shape = try self.allocator.dupe(usize, shape);
        return w;
    }
};
