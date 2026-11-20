//! In-memory rewind checkpoints for `flint session` `truncate`. Only sliding
//! rings are copied: lowering the position hides later full-attention entries,
//! and every entry is rewritten before it can be read again. A checkpoint at c
//! is valid only while the first c cached tokens are unchanged, so callers drop
//! checkpoints above any rewind point and clear them on reset or disk restore.
//! Host memory only; the tests below are CPU-only.
const std = @import("std");
const Layer = @import("prefix_cache.zig").Layer;
const Allocator = std.mem.Allocator;

pub const default_limit = 4;

/// True when rewinding from `from` to `to` needs no ring restore: no ring slot
/// holding one of the last `capacity` positions before `to` was overwritten.
pub fn ringsIntact(layers: []const Layer, from: u32, to: u32) bool {
    if (to >= from) return true;
    for (layers) |layer| if (layer.ring and from > layer.capacity) return false;
    return true;
}

fn ringBytes(layers: []const Layer, position: u32) usize {
    var total: usize = 0;
    for (layers) |layer| if (layer.ring) {
        const slots = @min(layer.capacity, position);
        total += slots * (layer.keys.len / layer.capacity + layer.values.len / layer.capacity);
    };
    return total;
}

pub const Checkpoints = struct {
    pub const Slot = struct {
        position: u32 = 0,
        /// Requested with checkpoint_at (or restored from disk): evicted after
        /// automatic checkpoints, so client-marked boundaries survive longer.
        pinned: bool = false,
        used: bool = false,
        serial: u64 = 0,
        rings: []u8 = &.{},
    };
    gpa: Allocator,
    slots: []Slot,
    serial: u64 = 0,

    pub fn init(gpa: Allocator, limit: usize) !Checkpoints {
        const slots = try gpa.alloc(Slot, limit);
        @memset(slots, .{});
        return .{ .gpa = gpa, .slots = slots };
    }
    pub fn deinit(self: *Checkpoints) void {
        for (self.slots) |slot| self.gpa.free(slot.rings);
        self.gpa.free(self.slots);
    }
    pub fn clear(self: *Checkpoints) void {
        for (self.slots) |*slot| slot.used = false;
    }
    pub fn dropAbove(self: *Checkpoints, position: u32) void {
        for (self.slots) |*slot| if (slot.position > position) {
            slot.used = false;
        };
    }
    /// Newest-position checkpoint at or below `position`.
    pub fn best(self: *Checkpoints, position: u32) ?*Slot {
        var result: ?*Slot = null;
        for (self.slots) |*slot| if (slot.used and slot.position <= position and (result == null or slot.position > result.?.position)) {
            result = slot;
        };
        return result;
    }
    pub fn count(self: *const Checkpoints) usize {
        var n: usize = 0;
        for (self.slots) |slot| n += @intFromBool(slot.used);
        return n;
    }

    /// Copy the valid ring slots at `position`. The device must be idle.
    pub fn take(self: *Checkpoints, layers: []const Layer, position: u32, pinned: bool) !void {
        if (self.slots.len == 0) return;
        self.serial += 1;
        for (self.slots) |*slot| if (slot.used and slot.position == position) {
            slot.pinned = slot.pinned or pinned;
            slot.serial = self.serial;
            return;
        };
        var victim: ?*Slot = null;
        for (self.slots) |*slot| if (!slot.used) {
            victim = slot;
            break;
        };
        // Oldest automatic checkpoint first; pinned ones only when all are pinned.
        if (victim == null) for ([_]bool{ false, true }) |pass| {
            for (self.slots) |*slot| if (slot.pinned == pass and (victim == null or slot.serial < victim.?.serial)) {
                victim = slot;
            };
            if (victim != null) break;
        };
        const slot = victim.?;
        const bytes = ringBytes(layers, position);
        if (slot.rings.len < bytes) {
            self.gpa.free(slot.rings);
            slot.rings = &.{};
            slot.used = false;
            // Size for a full ring once; later checkpoints reuse the allocation.
            slot.rings = try self.gpa.alloc(u8, @max(bytes, ringBytes(layers, std.math.maxInt(u32))));
        }
        var offset: usize = 0;
        for (layers) |layer| if (layer.ring) {
            const slots = @min(layer.capacity, position);
            const keys = slots * (layer.keys.len / layer.capacity);
            const values = slots * (layer.values.len / layer.capacity);
            @memcpy(slot.rings[offset..][0..keys], layer.keys[0..keys]);
            offset += keys;
            @memcpy(slot.rings[offset..][0..values], layer.values[0..values]);
            offset += values;
        };
        slot.* = .{ .position = position, .pinned = pinned, .used = true, .serial = self.serial, .rings = slot.rings };
    }

    /// Copy a checkpoint's rings back. The caller then lowers the position to
    /// `slot.position`; full-attention entries below it are already intact.
    pub fn restore(slot: *const Slot, layers: []const Layer) void {
        var offset: usize = 0;
        for (layers) |layer| if (layer.ring) {
            const slots = @min(layer.capacity, slot.position);
            const keys = slots * (layer.keys.len / layer.capacity);
            const values = slots * (layer.values.len / layer.capacity);
            @memcpy(layer.keys[0..keys], slot.rings[offset..][0..keys]);
            offset += keys;
            @memcpy(layer.values[0..values], slot.rings[offset..][0..values]);
            offset += values;
        };
    }
};

const testing = std.testing;

test "rewind needs no ring restore only before any ring slot is reused" {
    var k: [12]u8 = undefined;
    var v: [12]u8 = undefined;
    var fk: [64]u8 = undefined;
    const layers = [_]Layer{
        .{ .keys = &fk, .values = &fk, .capacity = 16, .ring = false },
        .{ .keys = &k, .values = &v, .capacity = 3, .ring = true },
    };
    try testing.expect(ringsIntact(&layers, 3, 0));
    try testing.expect(ringsIntact(&layers, 3, 2));
    try testing.expect(!ringsIntact(&layers, 4, 3));
    try testing.expect(!ringsIntact(&layers, 9, 0));
    try testing.expect(ringsIntact(&layers, 9, 9));
    try testing.expect(ringsIntact(layers[0..1], 15, 1));
}

test "checkpoints copy only ring slots, restore them and select the newest at or below" {
    var ring_k: [12]u8 = @splat(0);
    var ring_v: [6]u8 = @splat(0);
    var full: [32]u8 = @splat(7);
    const layers = [_]Layer{
        .{ .keys = &full, .values = &full, .capacity = 8, .ring = false },
        .{ .keys = &ring_k, .values = &ring_v, .capacity = 3, .ring = true },
    };
    var checkpoints = try Checkpoints.init(testing.allocator, 3);
    defer checkpoints.deinit();
    ring_k[0..4].* = .{ 1, 1, 1, 1 };
    ring_v[0..2].* = .{ 2, 2 };
    try checkpoints.take(&layers, 1, false);
    @memset(&ring_k, 3);
    @memset(&ring_v, 4);
    try checkpoints.take(&layers, 5, false);
    @memset(&ring_k, 9);
    @memset(&ring_v, 9);
    try testing.expect(checkpoints.best(0) == null);
    try testing.expectEqual(@as(u32, 1), checkpoints.best(4).?.position);
    Checkpoints.restore(checkpoints.best(4).?, &layers);
    try testing.expectEqualSlices(u8, &.{ 1, 1, 1, 1, 9, 9, 9, 9, 9, 9, 9, 9 }, &ring_k);
    try testing.expectEqualSlices(u8, &.{ 2, 2, 9, 9, 9, 9 }, &ring_v);
    Checkpoints.restore(checkpoints.best(5).?, &layers);
    try testing.expectEqualSlices(u8, &@as([12]u8, @splat(3)), &ring_k);
    try testing.expectEqualSlices(u8, &@as([6]u8, @splat(4)), &ring_v);
    try testing.expectEqualSlices(u8, &@as([32]u8, @splat(7)), &full);
    checkpoints.dropAbove(4);
    try testing.expectEqual(@as(u32, 1), checkpoints.best(100).?.position);
    checkpoints.clear();
    try testing.expect(checkpoints.best(100) == null);
}

test "checkpoint eviction keeps pinned boundaries and deduplicates positions" {
    var ring: [6]u8 = @splat(0);
    const layers = [_]Layer{.{ .keys = &ring, .values = &ring, .capacity = 3, .ring = true }};
    var checkpoints = try Checkpoints.init(testing.allocator, 3);
    defer checkpoints.deinit();
    try checkpoints.take(&layers, 10, true);
    try checkpoints.take(&layers, 20, false);
    try checkpoints.take(&layers, 30, false);
    try checkpoints.take(&layers, 20, false); // refresh, no new slot
    try testing.expectEqual(@as(usize, 3), checkpoints.count());
    try checkpoints.take(&layers, 40, false); // evicts 30, the oldest automatic
    try testing.expectEqual(@as(u32, 20), checkpoints.best(39).?.position);
    try checkpoints.take(&layers, 50, false); // evicts 20
    try testing.expectEqual(@as(u32, 10), checkpoints.best(39).?.position);
    try checkpoints.take(&layers, 40, true);
    try checkpoints.take(&layers, 60, true); // evicts 50, the only automatic
    try testing.expectEqual(@as(u32, 40), checkpoints.best(59).?.position);
    try checkpoints.take(&layers, 70, true); // all pinned: evicts oldest pinned, 10
    try testing.expect(checkpoints.best(39) == null);
    var none = try Checkpoints.init(testing.allocator, 0);
    defer none.deinit();
    try none.take(&layers, 1, true);
    try testing.expect(none.best(1) == null);
}
