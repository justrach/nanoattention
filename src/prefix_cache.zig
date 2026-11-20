//! Persistent prefix snapshots for `flint session --prefix-cache DIR`.
//! A snapshot holds exact token IDs and the valid KV state after them: the
//! first min(position, capacity) slots of every layer in physical slot order,
//! so a restore reproduces full-attention prefixes and sliding rings alike.
//! Matching compares token IDs, never a hash alone. A version, identity,
//! settings, layout, length or checksum mismatch makes a file unusable.
//! Writes go to a hidden temporary file renamed into place; LRU order is the
//! file mtime, refreshed on every use. Host little-endian layout (Apple Silicon).
//! Plain libc file I/O and no GPU work, so the tests below are CPU-only.
const std = @import("std");
const c = std.c;
const Allocator = std.mem.Allocator;
const Sha256 = std.crypto.hash.sha2.Sha256;
const XxHash3 = std.hash.XxHash3;
extern "c" fn kill(pid: c_int, signal: c_int) c_int;

pub const version: u32 = 1;
pub const magic = "FLKVSNAP".*;
pub const extension = ".kvsnap";
pub const default_cap_bytes: u64 = 4 << 30;

/// KV storage as the engine holds it: `keys`/`values` span all `capacity` slots.
/// `ring` marks sliding layers whose slots are reused modulo `capacity`.
pub const Layer = struct { keys: []u8, values: []u8, capacity: u32, ring: bool };

/// identity: checkpoint config bytes plus shard names/sizes. settings: every
/// engine setting, device and binary that can change stored layout or values.
pub const Key = struct { identity: [32]u8, settings: [32]u8 };

const Header = extern struct {
    magic: [8]u8,
    version: u32,
    layer_count: u32,
    token_count: u32,
    reserved: u32 = 0,
    identity: [32]u8,
    settings: [32]u8,
    /// Tokens, layer records and state; the file is exactly header + payload.
    payload_bytes: u64,
    checksum: u64,
};
const Record = extern struct { key_stride: u64, value_stride: u64, capacity: u32, slots: u32, ring: u32, reserved: u32 = 0 };

fn record(layer: Layer, position: u32) !Record {
    if (layer.capacity == 0 or layer.keys.len % layer.capacity != 0 or layer.values.len % layer.capacity != 0)
        return error.InvalidKvGeometry;
    if (!layer.ring and position > layer.capacity) return error.InvalidKvGeometry;
    return .{
        .key_stride = layer.keys.len / layer.capacity,
        .value_stride = layer.values.len / layer.capacity,
        // Only a ring's capacity changes slot placement. A full-attention layer
        // stores positions 0..n in order whatever the configured context.
        .capacity = if (layer.ring) layer.capacity else 0,
        .slots = @min(layer.capacity, position),
        .ring = @intFromBool(layer.ring),
    };
}

/// Bytes of valid KV state at `position`, the same rule a snapshot persists.
pub fn stateBytes(layers: []const Layer, position: u32) !u64 {
    var total: u64 = 0;
    for (layers) |layer| {
        const r = try record(layer, position);
        total = try std.math.add(u64, total, try std.math.mul(u64, r.slots, r.key_stride + r.value_stride));
    }
    return total;
}

fn payloadBytes(layers: []const Layer, tokens: usize) !u64 {
    const fixed = try std.math.add(u64, try std.math.mul(u64, tokens, 4), try std.math.mul(u64, layers.len, @sizeOf(Record)));
    return std.math.add(u64, fixed, try stateBytes(layers, std.math.cast(u32, tokens) orelse return error.ContextExhausted));
}

pub fn fileBytes(layers: []const Layer, tokens: usize) !u64 {
    return std.math.add(u64, @sizeOf(Header), try payloadBytes(layers, tokens));
}

/// File name: content address of key and tokens. Lookup still reads and
/// compares the stored token IDs; the name only deduplicates writes.
pub fn fileName(key: Key, tokens: []const u32) [32 + extension.len]u8 {
    var h = Sha256.init(.{});
    h.update(&key.identity);
    h.update(&key.settings);
    h.update(std.mem.sliceAsBytes(tokens));
    var digest: [32]u8 = undefined;
    h.final(&digest);
    return std.fmt.bytesToHex(digest[0..16].*, .lower) ++ extension.*;
}

fn writeAll(fd: c.fd_t, bytes: []const u8) !void {
    var offset: usize = 0;
    while (offset < bytes.len) {
        const n = c.write(fd, bytes[offset..].ptr, @min(bytes.len - offset, 1 << 30));
        if (n <= 0) return error.SnapshotWriteFailed;
        offset += @intCast(n);
    }
}

fn readAt(fd: c.fd_t, bytes: []u8, start: u64) !void {
    var offset: usize = 0;
    while (offset < bytes.len) {
        const n = c.pread(fd, bytes[offset..].ptr, @min(bytes.len - offset, 1 << 30), @intCast(start + offset));
        if (n <= 0) return error.SnapshotTruncated;
        offset += @intCast(n);
    }
}

fn fileSize(fd: c.fd_t) !u64 {
    var st: c.Stat = undefined;
    if (c.fstat(fd, &st) != 0) return error.SnapshotStatFailed;
    return @intCast(st.size);
}

pub const Restored = struct {
    position: u32 = 0,
    bytes: u64 = 0,
    /// A candidate passed header checks but failed its checksum after writing
    /// into the layers. Every state the caller held in them is then invalid.
    clobbered: bool = false,
};

pub const Written = struct {
    outcome: enum { written, existing, too_large } = .written,
    bytes: u64 = 0,
};

pub const Cache = struct {
    gpa: Allocator,
    dir: []const u8,
    cap: u64,
    key: Key,
    temp_serial: u32 = 0,

    pub fn init(gpa: Allocator, dir: []const u8, cap: u64, key: Key) !Cache {
        if (dir.len == 0) return error.InvalidPrefixCacheDirectory;
        const owned = try gpa.dupe(u8, dir);
        errdefer gpa.free(owned);
        const z = try gpa.dupeSentinel(u8, dir, 0);
        defer gpa.free(z);
        if (c.mkdir(z, 0o755) != 0 and c.errno(@as(c_int, -1)) != .EXIST) return error.PrefixCacheDirectoryFailed;
        const self: Cache = .{ .gpa = gpa, .dir = owned, .cap = cap, .key = key };
        try self.removeOrphans();
        return self;
    }

    /// Remove `.PID-N.part` files whose writer no longer exists (a crash or
    /// kill mid-write); they are invisible to lookup and to the size cap.
    fn removeOrphans(self: *const Cache) !void {
        const z = try self.gpa.dupeSentinel(u8, self.dir, 0);
        defer self.gpa.free(z);
        const handle = c.opendir(z) orelse return error.PrefixCacheDirectoryFailed;
        defer _ = c.closedir(handle);
        while (c.readdir(handle)) |entry| {
            const name = entry.name[0..entry.namlen];
            if (name.len < 2 or name[0] != '.' or !std.mem.endsWith(u8, name, ".part")) continue;
            const dash = std.mem.indexOfScalar(u8, name, '-') orelse continue;
            const pid = std.fmt.parseInt(c_int, name[1..dash], 10) catch continue;
            if (pid <= 0 or kill(pid, 0) == 0 or c.errno(@as(c_int, -1)) != .SRCH) continue;
            const full = try self.path(name);
            defer self.gpa.free(full);
            _ = c.unlink(full);
        }
    }
    pub fn deinit(self: *Cache) void {
        self.gpa.free(self.dir);
    }
    fn path(self: *const Cache, name: []const u8) ![:0]u8 {
        return std.fmt.allocPrintSentinel(self.gpa, "{s}/{s}", .{ self.dir, name }, 0);
    }

    /// Persist the valid state after `tokens`. Atomic: a reader sees either no
    /// file or a complete one. A same-named file of the right size is reused.
    pub fn write(self: *Cache, tokens: []const u32, layers: []const Layer) !Written {
        if (tokens.len == 0) return error.EmptyPrompt;
        const position = std.math.cast(u32, tokens.len) orelse return error.ContextExhausted;
        const total = try fileBytes(layers, tokens.len);
        if (total > self.cap) return .{ .outcome = .too_large };
        const name = fileName(self.key, tokens);
        const final = try self.path(&name);
        defer self.gpa.free(final);
        const existing = c.open(final, .{ .ACCMODE = .RDONLY });
        if (existing >= 0) {
            defer _ = c.close(existing);
            if ((fileSize(existing) catch 0) == total) {
                _ = c.futimens(existing, null);
                return .{ .outcome = .existing };
            }
        }
        try self.evict(total);
        var header: Header = .{
            .magic = magic,
            .version = version,
            .layer_count = @intCast(layers.len),
            .token_count = position,
            .identity = self.key.identity,
            .settings = self.key.settings,
            .payload_bytes = total - @sizeOf(Header),
            .checksum = 0,
        };
        const records = try self.gpa.alloc(Record, layers.len);
        defer self.gpa.free(records);
        var hash = XxHash3.init(0);
        hash.update(std.mem.sliceAsBytes(tokens));
        for (layers, records) |layer, *r| r.* = try record(layer, position);
        hash.update(std.mem.sliceAsBytes(records));
        for (layers, records) |layer, r| {
            hash.update(layer.keys[0 .. r.slots * r.key_stride]);
            hash.update(layer.values[0 .. r.slots * r.value_stride]);
        }
        header.checksum = hash.final();
        self.temp_serial += 1;
        var temp_name: [64]u8 = undefined;
        const temp = try self.path(try std.fmt.bufPrint(&temp_name, ".{d}-{d}.part", .{ c.getpid(), self.temp_serial }));
        defer self.gpa.free(temp);
        const fd = c.open(temp, .{ .ACCMODE = .WRONLY, .CREAT = true, .EXCL = true }, @as(c_uint, 0o644));
        if (fd < 0) return error.SnapshotWriteFailed;
        var renamed = false;
        defer if (!renamed) {
            _ = c.unlink(temp);
        };
        {
            defer _ = c.close(fd);
            try writeAll(fd, std.mem.asBytes(&header));
            try writeAll(fd, std.mem.sliceAsBytes(tokens));
            try writeAll(fd, std.mem.sliceAsBytes(records));
            for (layers, records) |layer, r| {
                try writeAll(fd, layer.keys[0 .. r.slots * r.key_stride]);
                try writeAll(fd, layer.values[0 .. r.slots * r.value_stride]);
            }
        }
        if (c.rename(temp, final) != 0) return error.SnapshotWriteFailed;
        renamed = true;
        return .{ .bytes = total };
    }

    /// Validate everything except the state checksum. Never touches `layers`.
    fn candidate(self: *const Cache, fd: c.fd_t, tokens: []const u32, above: u32, at_most: u32, layers: []const Layer) !?u32 {
        var header: Header = undefined;
        readAt(fd, std.mem.asBytes(&header), 0) catch return null;
        if (!std.mem.eql(u8, &header.magic, &magic) or header.version != version or
            !std.mem.eql(u8, &header.identity, &self.key.identity) or
            !std.mem.eql(u8, &header.settings, &self.key.settings) or
            header.layer_count != layers.len) return null;
        const count = header.token_count;
        if (count <= above or count > at_most or count > tokens.len) return null;
        const expected = fileBytes(layers, count) catch return null;
        if (header.payload_bytes != expected - @sizeOf(Header)) return error.CorruptSnapshot;
        if ((try fileSize(fd)) != expected) return error.CorruptSnapshot;
        // Token comparison is the match; read in bounded pieces.
        var buffer: [4096]u32 = undefined;
        var offset: usize = 0;
        while (offset < count) {
            const n = @min(buffer.len, count - offset);
            try readAt(fd, std.mem.sliceAsBytes(buffer[0..n]), @sizeOf(Header) + offset * 4);
            if (!std.mem.eql(u32, buffer[0..n], tokens[offset..][0..n])) return null;
            offset += n;
        }
        const table = @sizeOf(Header) + @as(u64, count) * 4;
        for (layers, 0..) |layer, i| {
            var stored: Record = undefined;
            try readAt(fd, std.mem.asBytes(&stored), table + i * @sizeOf(Record));
            const live = record(layer, count) catch return null;
            if (!std.mem.eql(u8, std.mem.asBytes(&stored), std.mem.asBytes(&live))) return null;
        }
        return count;
    }

    fn load(fd: c.fd_t, tokens: []const u32, count: u32, layers: []const Layer, gpa: Allocator) !bool {
        var header: Header = undefined;
        try readAt(fd, std.mem.asBytes(&header), 0);
        var hash = XxHash3.init(0);
        hash.update(std.mem.sliceAsBytes(tokens[0..count]));
        const records = try gpa.alloc(Record, layers.len);
        defer gpa.free(records);
        var offset: u64 = @sizeOf(Header) + @as(u64, count) * 4;
        try readAt(fd, std.mem.sliceAsBytes(records), offset);
        hash.update(std.mem.sliceAsBytes(records));
        offset += layers.len * @sizeOf(Record);
        for (layers, records) |layer, r| {
            const keys = layer.keys[0 .. r.slots * r.key_stride];
            const values = layer.values[0 .. r.slots * r.value_stride];
            try readAt(fd, keys, offset);
            offset += keys.len;
            try readAt(fd, values, offset);
            offset += values.len;
            hash.update(keys);
            hash.update(values);
        }
        return hash.final() == header.checksum;
    }

    /// Restore the longest snapshot whose token IDs are exactly `tokens[0..n]`
    /// with above < n <= at_most. Files that fail validation for this key are
    /// removed; files for other checkpoints, settings or versions are left alone.
    pub fn restoreLongest(self: *Cache, tokens: []const u32, above: u32, at_most: u32, layers: []const Layer) !Restored {
        const Found = struct { name: []u8, count: u32 };
        var found: std.ArrayList(Found) = .empty;
        defer {
            for (found.items) |f| self.gpa.free(f.name);
            found.deinit(self.gpa);
        }
        {
            const z = try self.gpa.dupeSentinel(u8, self.dir, 0);
            defer self.gpa.free(z);
            const handle = c.opendir(z) orelse return error.PrefixCacheDirectoryFailed;
            defer _ = c.closedir(handle);
            while (c.readdir(handle)) |entry| {
                const name = entry.name[0..entry.namlen];
                if (name.len == 0 or name[0] == '.' or !std.mem.endsWith(u8, name, extension)) continue;
                const full = try self.path(name);
                defer self.gpa.free(full);
                const fd = c.open(full, .{ .ACCMODE = .RDONLY });
                if (fd < 0) continue;
                defer _ = c.close(fd);
                const count = self.candidate(fd, tokens, above, at_most, layers) catch {
                    _ = c.unlink(full);
                    continue;
                } orelse continue;
                try found.append(self.gpa, .{ .name = try self.gpa.dupe(u8, name), .count = count });
            }
        }
        std.mem.sort(Found, found.items, {}, struct {
            fn longer(_: void, a: Found, b: Found) bool {
                return a.count > b.count or (a.count == b.count and std.mem.lessThan(u8, a.name, b.name));
            }
        }.longer);
        var result: Restored = .{};
        for (found.items) |f| {
            const full = try self.path(f.name);
            defer self.gpa.free(full);
            const fd = c.open(full, .{ .ACCMODE = .RDONLY });
            if (fd < 0) continue;
            defer _ = c.close(fd);
            // Recheck: the file may have been replaced since the scan.
            const count = (self.candidate(fd, tokens, above, at_most, layers) catch null) orelse continue;
            if (count != f.count) continue;
            result.clobbered = true;
            if (load(fd, tokens, count, layers, self.gpa) catch false) {
                _ = c.futimens(fd, null);
                result.position = count;
                result.bytes = try fileBytes(layers, count);
                return result;
            }
            _ = c.unlink(full);
        }
        return result;
    }

    /// Remove least recently used snapshots until `incoming` more bytes fit.
    pub fn evict(self: *Cache, incoming: u64) !void {
        const Entry = struct { name: []u8, size: u64, sec: i64, nsec: i64 };
        var entries: std.ArrayList(Entry) = .empty;
        defer {
            for (entries.items) |e| self.gpa.free(e.name);
            entries.deinit(self.gpa);
        }
        var total: u64 = 0;
        {
            const z = try self.gpa.dupeSentinel(u8, self.dir, 0);
            defer self.gpa.free(z);
            const handle = c.opendir(z) orelse return error.PrefixCacheDirectoryFailed;
            defer _ = c.closedir(handle);
            while (c.readdir(handle)) |entry| {
                const name = entry.name[0..entry.namlen];
                if (name.len == 0 or name[0] == '.' or !std.mem.endsWith(u8, name, extension)) continue;
                const full = try self.path(name);
                defer self.gpa.free(full);
                var st: c.Stat = undefined;
                if (c.stat(full, &st) != 0) continue;
                const size: u64 = @intCast(st.size);
                total += size;
                try entries.append(self.gpa, .{ .name = try self.gpa.dupe(u8, name), .size = size, .sec = st.mtimespec.sec, .nsec = st.mtimespec.nsec });
            }
        }
        std.mem.sort(Entry, entries.items, {}, struct {
            fn older(_: void, a: Entry, b: Entry) bool {
                if (a.sec != b.sec) return a.sec < b.sec;
                if (a.nsec != b.nsec) return a.nsec < b.nsec;
                return std.mem.lessThan(u8, a.name, b.name);
            }
        }.older);
        for (entries.items) |e| {
            if (total + incoming <= self.cap) break;
            const full = try self.path(e.name);
            defer self.gpa.free(full);
            if (c.unlink(full) == 0) total -= e.size;
        }
    }
};

const testing = std.testing;

const Fixture = struct {
    storage: [4][]u8,
    layers: [2]Layer,
    fn init(seed: u8) !Fixture {
        var f: Fixture = undefined;
        // Full layer: 8 slots of 4/8 bytes; ring: 3 slots of 4/4 bytes.
        const sizes = [_]usize{ 32, 64, 12, 12 };
        for (&f.storage, sizes, 0..) |*s, size, i| {
            s.* = try testing.allocator.alloc(u8, size);
            for (s.*, 0..) |*b, j| b.* = @truncate(seed +% @as(u8, @intCast(i * 61 + j)));
        }
        f.layers = .{
            .{ .keys = f.storage[0], .values = f.storage[1], .capacity = 8, .ring = false },
            .{ .keys = f.storage[2], .values = f.storage[3], .capacity = 3, .ring = true },
        };
        return f;
    }
    fn deinit(self: *Fixture) void {
        for (self.storage) |s| testing.allocator.free(s);
    }
};

fn testDir(buffer: []u8) ![:0]const u8 {
    _ = c.mkdir("target", 0o755);
    const dir = try std.fmt.bufPrintSentinel(buffer, "target/prefix-cache-unit-{d}", .{c.getpid()}, 0);
    removeDir(dir);
    return dir;
}

fn removeDir(dir: [:0]const u8) void {
    const handle = c.opendir(dir) orelse return;
    var buffer: [512]u8 = undefined;
    while (c.readdir(handle)) |entry| {
        const name = entry.name[0..entry.namlen];
        if (std.mem.eql(u8, name, ".") or std.mem.eql(u8, name, "..")) continue;
        const full = std.fmt.bufPrintSentinel(&buffer, "{s}/{s}", .{ dir, name }, 0) catch continue;
        _ = c.unlink(full);
    }
    _ = c.closedir(handle);
    _ = c.rmdir(dir);
}

fn testKey(identity: u8, settings: u8) Key {
    return .{ .identity = @splat(identity), .settings = @splat(settings) };
}

fn snapshotPath(buffer: []u8, dir: []const u8, k: Key, tokens: []const u32) ![:0]u8 {
    return std.fmt.bufPrintSentinel(buffer, "{s}/{s}", .{ dir, &fileName(k, tokens) }, 0);
}

test "prefix snapshot round trip restores only valid slots, in physical order" {
    var name: [128]u8 = undefined;
    const dir = try testDir(&name);
    defer removeDir(dir);
    var source = try Fixture.init(1);
    defer source.deinit();
    var target = try Fixture.init(200);
    defer target.deinit();
    var cache = try Cache.init(testing.allocator, dir, 1 << 20, testKey(1, 2));
    defer cache.deinit();
    const tokens = [_]u32{ 5, 6, 7, 8, 9, 10 };
    const written = try cache.write(&tokens, &source.layers);
    try testing.expect(written.outcome == .written);
    try testing.expectEqual(try fileBytes(&source.layers, 6), written.bytes);
    // 6 full slots of 12 bytes plus a full 3-slot ring of 8 bytes.
    try testing.expectEqual(@as(u64, 6 * 12 + 3 * 8), try stateBytes(&source.layers, 6));
    try testing.expect((try cache.write(&tokens, &source.layers)).outcome == .existing);
    const untouched = try testing.allocator.dupe(u8, target.storage[0][24..]);
    defer testing.allocator.free(untouched);
    const restored = try cache.restoreLongest(&(tokens ++ [_]u32{11}), 0, 6, &target.layers);
    try testing.expectEqual(@as(u32, 6), restored.position);
    try testing.expectEqualSlices(u8, source.storage[0][0..24], target.storage[0][0..24]);
    try testing.expectEqualSlices(u8, untouched, target.storage[0][24..]);
    try testing.expectEqualSlices(u8, source.storage[1][0..48], target.storage[1][0..48]);
    try testing.expectEqualSlices(u8, source.storage[2], target.storage[2]);
    try testing.expectEqualSlices(u8, source.storage[3], target.storage[3]);
    // A snapshot equal to the whole request is excluded by at_most.
    try testing.expectEqual(@as(u32, 0), (try cache.restoreLongest(&tokens, 0, 5, &target.layers)).position);
}

test "prefix snapshot rejects version, identity, settings and layout mismatches" {
    var name: [128]u8 = undefined;
    const dir = try testDir(&name);
    defer removeDir(dir);
    var f = try Fixture.init(3);
    defer f.deinit();
    const tokens = [_]u32{ 1, 2, 3, 4 };
    const request = [_]u32{ 1, 2, 3, 4, 5 };
    var cache = try Cache.init(testing.allocator, dir, 1 << 20, testKey(1, 2));
    defer cache.deinit();
    _ = try cache.write(&tokens, &f.layers);
    for ([_]Key{ testKey(9, 2), testKey(1, 9) }) |other| {
        var foreign = try Cache.init(testing.allocator, dir, 1 << 20, other);
        defer foreign.deinit();
        const r = try foreign.restoreLongest(&request, 0, 4, &f.layers);
        try testing.expectEqual(@as(u32, 0), r.position);
        try testing.expect(!r.clobbered);
    }
    // Different ring capacity: same bytes, different layout.
    var reshaped = f.layers;
    reshaped[1].capacity = 1;
    try testing.expectEqual(@as(u32, 0), (try cache.restoreLongest(&request, 0, 4, &reshaped)).position);
    // A full-attention layer's capacity (the context) does not change layout.
    var wider = f.layers;
    wider[0].keys = f.storage[0][0..16];
    wider[0].values = f.storage[1][0..32];
    wider[0].capacity = 4;
    try testing.expectEqual(@as(u32, 4), (try cache.restoreLongest(&request, 0, 4, &wider)).position);
    wider[0].capacity = 2;
    wider[0].keys = f.storage[0][0..8];
    wider[0].values = f.storage[1][0..16];
    try testing.expectEqual(@as(u32, 0), (try cache.restoreLongest(&request, 0, 4, &wider)).position);
    try testing.expectEqual(@as(u32, 0), (try cache.restoreLongest(&request, 0, 4, f.layers[0..1])).position);
    // Foreign files are ignored, not removed.
    try testing.expectEqual(@as(u32, 4), (try cache.restoreLongest(&request, 0, 4, &f.layers)).position);
    var buffer: [256]u8 = undefined;
    const file = try snapshotPath(&buffer, dir, cache.key, &tokens);
    const fd = c.open(file, .{ .ACCMODE = .RDWR });
    try testing.expect(fd >= 0);
    defer _ = c.close(fd);
    const future = version + 1;
    try testing.expectEqual(@as(isize, 4), c.pwrite(fd, std.mem.asBytes(&future), 4, 8));
    const r = try cache.restoreLongest(&request, 0, 4, &f.layers);
    try testing.expectEqual(@as(u32, 0), r.position);
    try testing.expect(!r.clobbered);
}

test "prefix snapshot detects truncated and corrupted files" {
    var name: [128]u8 = undefined;
    const dir = try testDir(&name);
    defer removeDir(dir);
    var f = try Fixture.init(5);
    defer f.deinit();
    var cache = try Cache.init(testing.allocator, dir, 1 << 20, testKey(4, 4));
    defer cache.deinit();
    const tokens = [_]u32{ 1, 2, 3 };
    const request = [_]u32{ 1, 2, 3, 4 };
    var buffer: [256]u8 = undefined;
    const file = try snapshotPath(&buffer, dir, cache.key, &tokens);
    _ = try cache.write(&tokens, &f.layers);
    const size = try fileBytes(&f.layers, 3);
    {
        const fd = c.open(file, .{ .ACCMODE = .RDWR });
        defer _ = c.close(fd);
        try testing.expectEqual(@as(c_int, 0), c.ftruncate(fd, @intCast(size - 1)));
    }
    const truncated = try cache.restoreLongest(&request, 0, 3, &f.layers);
    try testing.expectEqual(@as(u32, 0), truncated.position);
    try testing.expect(!truncated.clobbered);
    try testing.expect(c.open(file, .{ .ACCMODE = .RDONLY }) < 0);
    // Same size, one flipped state byte: caught by the checksum after loading.
    _ = try cache.write(&tokens, &f.layers);
    {
        const fd = c.open(file, .{ .ACCMODE = .RDWR });
        defer _ = c.close(fd);
        const flipped = [_]u8{f.storage[3][0] ^ 1};
        try testing.expectEqual(@as(isize, 1), c.pwrite(fd, &flipped, 1, @intCast(size - 8)));
    }
    const corrupt = try cache.restoreLongest(&request, 0, 3, &f.layers);
    try testing.expectEqual(@as(u32, 0), corrupt.position);
    try testing.expect(corrupt.clobbered);
    try testing.expect(c.open(file, .{ .ACCMODE = .RDONLY }) < 0);
}

test "prefix lookup picks the longest exact token prefix" {
    var name: [128]u8 = undefined;
    const dir = try testDir(&name);
    defer removeDir(dir);
    var f = try Fixture.init(7);
    defer f.deinit();
    var cache = try Cache.init(testing.allocator, dir, 1 << 20, testKey(6, 6));
    defer cache.deinit();
    _ = try cache.write(&.{ 1, 2 }, &f.layers);
    _ = try cache.write(&.{ 1, 2, 3, 4 }, &f.layers);
    _ = try cache.write(&.{ 1, 2, 9, 9, 9 }, &f.layers);
    _ = try cache.write(&.{ 7, 2, 3, 4, 5, 6 }, &f.layers);
    const Case = struct { request: []const u32, at_most: u32, above: u32 = 0, expected: u32 };
    for ([_]Case{
        .{ .request = &.{ 1, 2, 3, 4, 5, 6, 7 }, .at_most = 6, .expected = 4 },
        .{ .request = &.{ 1, 2, 9, 9, 9, 9 }, .at_most = 5, .expected = 5 },
        .{ .request = &.{ 1, 2, 3 }, .at_most = 2, .expected = 2 },
        .{ .request = &.{ 1, 2, 3, 4 }, .at_most = 3, .expected = 2 },
        .{ .request = &.{ 1, 2, 3, 4, 5 }, .at_most = 4, .above = 2, .expected = 4 },
        .{ .request = &.{ 1, 2, 3, 4, 5 }, .at_most = 4, .above = 4, .expected = 0 },
        .{ .request = &.{ 1, 3 }, .at_most = 1, .expected = 0 },
        .{ .request = &.{ 2, 2, 3, 4, 5, 6, 7 }, .at_most = 6, .expected = 0 },
    }) |case| {
        const r = try cache.restoreLongest(case.request, case.above, case.at_most, &f.layers);
        try testing.expectEqual(case.expected, r.position);
        try testing.expect(r.position != 0 or !r.clobbered);
    }
}

test "prefix cache removes temporary files only when their writer is gone" {
    var name: [128]u8 = undefined;
    const dir = try testDir(&name);
    defer removeDir(dir);
    _ = c.mkdir(dir, 0o755);
    var buffer: [2][256]u8 = undefined;
    // PID 2147483646 cannot exist on macOS; this process does.
    const dead = try std.fmt.bufPrintSentinel(&buffer[0], "{s}/.2147483646-1.part", .{dir}, 0);
    const live = try std.fmt.bufPrintSentinel(&buffer[1], "{s}/.{d}-1.part", .{ dir, c.getpid() }, 0);
    for ([_][:0]u8{ dead, live }) |p| {
        const fd = c.open(p, .{ .ACCMODE = .WRONLY, .CREAT = true }, @as(c_uint, 0o644));
        try testing.expect(fd >= 0);
        _ = c.close(fd);
    }
    var cache = try Cache.init(testing.allocator, dir, 1 << 20, testKey(1, 1));
    defer cache.deinit();
    try testing.expect(c.open(dead, .{ .ACCMODE = .RDONLY }) < 0);
    const fd = c.open(live, .{ .ACCMODE = .RDONLY });
    try testing.expect(fd >= 0);
    _ = c.close(fd);
}

test "prefix cache evicts least recently used snapshots and skips oversized ones" {
    var name: [128]u8 = undefined;
    const dir = try testDir(&name);
    defer removeDir(dir);
    var f = try Fixture.init(9);
    defer f.deinit();
    const each = try fileBytes(&f.layers, 2);
    var cache = try Cache.init(testing.allocator, dir, 2 * each, testKey(8, 8));
    defer cache.deinit();
    const a = [_]u32{ 1, 1 };
    const b = [_]u32{ 2, 2 };
    const d = [_]u32{ 3, 3 };
    var buffer: [3][256]u8 = undefined;
    const paths = [_][:0]u8{ try snapshotPath(&buffer[0], dir, cache.key, &a), try snapshotPath(&buffer[1], dir, cache.key, &b), try snapshotPath(&buffer[2], dir, cache.key, &d) };
    _ = try cache.write(&a, &f.layers);
    _ = try cache.write(&b, &f.layers);
    // Make a older than b, then use a: b becomes least recently used.
    for (paths[0..2], [_]i64{ 100, 200 }) |p, sec| {
        const times = [2]c.timespec{ .{ .sec = sec, .nsec = 0 }, .{ .sec = sec, .nsec = 0 } };
        try testing.expectEqual(@as(c_int, 0), c.utimensat(c.AT.FDCWD, p, &times, 0));
    }
    try testing.expectEqual(@as(u32, 2), (try cache.restoreLongest(&.{ 1, 1, 5 }, 0, 2, &f.layers)).position);
    try testing.expect((try cache.write(&d, &f.layers)).outcome == .written);
    try testing.expect(c.open(paths[1], .{ .ACCMODE = .RDONLY }) < 0);
    for ([_][:0]u8{ paths[0], paths[2] }) |p| {
        const fd = c.open(p, .{ .ACCMODE = .RDONLY });
        try testing.expect(fd >= 0);
        _ = c.close(fd);
    }
    var small = try Cache.init(testing.allocator, dir, each - 1, testKey(8, 8));
    defer small.deinit();
    try testing.expect((try small.write(&.{ 4, 4 }, &f.layers)).outcome == .too_large);
    // A full-attention layer cannot hold more positions than its capacity.
    try testing.expectError(error.InvalidKvGeometry, stateBytes(&f.layers, 9));
}
