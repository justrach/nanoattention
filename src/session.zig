//! Persistent JSONL inference. The caller owns the one loaded Model.
//! History is append-only unless a request asks for an explicit `reset` or a
//! `truncate` rewind; opt-in `--prefix-cache` restores exact on-disk prefixes.
//! No sockets, implicit resets, or GPU work in the CPU-only tests.
const std = @import("std");
const Model = @import("model.zig").Model;
const prefix_cache = @import("prefix_cache.zig");
const rewind = @import("rewind.zig");
const Allocator = std.mem.Allocator;
const Sha256 = std.crypto.hash.sha2.Sha256;

extern "c" fn read(c_int, [*]u8, usize) isize;
extern "c" fn write(c_int, [*]const u8, usize) isize;
const Timespec = extern struct { sec: i64, nsec: i64 };
extern "c" fn clock_gettime(c_int, *Timespec) c_int;

pub const max_line_bytes = 1024 * 1024;
pub const max_checkpoint_at = 64;
pub const max_rewind_checkpoints = 64;

pub const Options = struct {
    prefix_cache: ?[]const u8 = null,
    prefix_cache_bytes: u64 = prefix_cache.default_cap_bytes,
    rewind_checkpoints: u32 = rewind.default_limit,
};

pub const Request = struct {
    id: u32 = 0,
    tokens: []const u32,
    max_tokens: u32 = 128,
    reset: bool = false,
    return_logits: bool = false,
    /// Roll the cache back to its first N history tokens before this request.
    truncate: ?u32 = null,
    /// Absolute cached-token positions inside this request's prefill span.
    checkpoint_at: []const u32 = &.{},
};

pub const Response = struct {
    id: u32,
    tokens: []const u32,
    eos: bool,
    cache_before: u32,
    cache_after: u32,
    prefill_tokens: usize,
    decode_forward_tokens: usize,
    reused_prefix_tokens: u32,
    prefill_seconds: f64,
    decode_seconds: f64,
    kv_allocated_bytes: u64,
    kv_valid_bytes: u64,
    kv_storage: []const u8 = "fp32",
    speculative_batches: usize = 0,
    speculative_proposed: usize = 0,
    speculative_accepted: usize = 0,
    prefill_logits: ?[]const f32,
    final_logits: ?[]const f32,
};

/// Fields appended only when a request or flag uses rewind or the prefix cache,
/// so responses without them stay byte-identical to the append-only protocol.
pub const Extra = struct {
    truncated_from: ?u32 = null,
    rewind_checkpoint: ?u32 = null,
    replayed_tokens: ?u32 = null,
    replay_seconds: ?f64 = null,
    /// "none" (no-op), "ring_intact", "memory", "disk" or "start".
    rewind_source: ?[]const u8 = null,
    restored_prefix_tokens: ?u32 = null,
    restore_seconds: ?f64 = null,
    snapshots_written: ?u32 = null,
    snapshot_bytes: ?u64 = null,
    snapshot_seconds: ?f64 = null,
    snapshot_failures: ?u32 = null,
    checkpoints_taken: ?u32 = null,
    checkpoint_seconds: ?f64 = null,
};

/// Splice extra fields into a complete JSON object. Both inputs are complete
/// serializations; "{}" (no extras) returns the base unchanged.
pub fn merge(a: Allocator, base: []const u8, extra: []const u8) ![]const u8 {
    if (base.len < 2 or base[0] != '{' or base[base.len - 1] != '}' or extra.len < 2 or extra[0] != '{' or extra[extra.len - 1] != '}')
        return error.InvalidResponseJson;
    if (extra.len == 2) return base;
    if (base.len == 2) return extra;
    return std.mem.concat(a, u8, &.{ base[0 .. base.len - 1], ",", extra[1..] });
}

/// Bounded framing, independent of file descriptors. A returned slice remains
/// valid until the next push. Newline is not counted in the 1 MiB limit; a CR
/// (if present) is JSON whitespace and is counted. Empty lines are requests too,
/// and consequently fail parsing instead of being silently ignored.
pub const LineBuffer = struct {
    storage: []u8,
    len: usize = 0,

    pub fn push(self: *LineBuffer, byte: u8) !?[]const u8 {
        if (byte == '\n') {
            const line = self.storage[0..self.len];
            self.len = 0;
            return line;
        }
        if (self.len >= @min(self.storage.len, max_line_bytes)) return error.LineTooLong;
        self.storage[self.len] = byte;
        self.len += 1;
        return null;
    }

    /// EOF emits one final unterminated line, but never invents an empty one.
    pub fn finish(self: *LineBuffer) ?[]const u8 {
        if (self.len == 0) return null;
        const line = self.storage[0..self.len];
        self.len = 0;
        return line;
    }
};

fn jsonU32(value: std.json.Value) !u32 {
    return switch (value) {
        .integer => |n| std.math.cast(u32, n) orelse error.InvalidRequestInteger,
        else => error.InvalidRequestInteger,
    };
}

fn jsonBool(value: std.json.Value) !bool {
    return switch (value) {
        .bool => |b| b,
        else => error.InvalidRequestBoolean,
    };
}

/// All allocations belong to a caller-supplied arena. Reject coercions (numeric
/// strings and floats), duplicate/unknown fields, and missing tokens explicitly.
pub fn parse(a: Allocator, line: []const u8) !Request {
    if (line.len > max_line_bytes) return error.LineTooLong;
    const value = try std.json.parseFromSliceLeaky(std.json.Value, a, line, .{
        .allocate = .alloc_always,
        .duplicate_field_behavior = .@"error",
        .max_value_len = max_line_bytes,
    });
    const object = switch (value) {
        .object => |object| object,
        else => return error.InvalidRequestObject,
    };
    var fields = object.iterator();
    while (fields.next()) |field| {
        const name = field.key_ptr.*;
        if (!std.mem.eql(u8, name, "id") and !std.mem.eql(u8, name, "tokens") and
            !std.mem.eql(u8, name, "max_tokens") and !std.mem.eql(u8, name, "reset") and
            !std.mem.eql(u8, name, "return_logits") and !std.mem.eql(u8, name, "truncate") and
            !std.mem.eql(u8, name, "checkpoint_at")) return error.UnknownRequestField;
    }
    const token_value = object.get("tokens") orelse return error.MissingTokens;
    const items = switch (token_value) {
        .array => |array| array.items,
        else => return error.InvalidTokens,
    };
    if (items.len == 0) return error.EmptyPrompt;
    const tokens = try a.alloc(u32, items.len);
    for (items, tokens) |item, *token| token.* = try jsonU32(item);
    var checkpoint_at: []u32 = &.{};
    if (object.get("checkpoint_at")) |field| {
        const positions = switch (field) {
            .array => |array| array.items,
            else => return error.InvalidCheckpointPositions,
        };
        if (positions.len > max_checkpoint_at) return error.TooManyCheckpointPositions;
        checkpoint_at = try a.alloc(u32, positions.len);
        for (positions, checkpoint_at) |item, *position| position.* = try jsonU32(item);
    }
    return .{
        .id = if (object.get("id")) |v| try jsonU32(v) else 0,
        .tokens = tokens,
        .max_tokens = if (object.get("max_tokens")) |v| try jsonU32(v) else 128,
        .reset = if (object.get("reset")) |v| try jsonBool(v) else false,
        .return_logits = if (object.get("return_logits")) |v| try jsonBool(v) else false,
        .truncate = if (object.get("truncate")) |v| try jsonU32(v) else null,
        .checkpoint_at = checkpoint_at,
    };
}

pub const Preflight = struct { cache_before: u32, required: u64 };

/// Reserve a forward for EVERY generated token, including EOS and the final
/// max_tokens token. Using u64 prevents the u32 context arithmetic wrapping.
/// A reset or truncate changes only the effective position here, never the
/// actual model. checkpoint_at positions must strictly increase inside
/// (cache_before, cache_before + tokens.len].
pub fn validate(request: Request, position: u32, context: u32, vocab_size: u32) !Preflight {
    if (request.tokens.len == 0) return error.EmptyPrompt;
    if (context == 0) return error.InvalidContextLength;
    if (vocab_size == 0) return error.InvalidVocabulary;
    for (request.tokens) |token| if (token >= vocab_size) return error.InvalidToken;
    if (request.truncate) |n| {
        if (request.reset) return error.TruncateWithReset;
        if (n > position) return error.InvalidTruncate;
    }
    const before = if (request.reset) @as(u32, 0) else request.truncate orelse position;
    const with_prompt = std.math.add(u64, before, @intCast(request.tokens.len)) catch return error.ContextExhausted;
    const required = std.math.add(u64, with_prompt, request.max_tokens) catch return error.ContextExhausted;
    if (required > context) return error.ContextExhausted;
    var previous: u64 = before;
    for (request.checkpoint_at) |p| {
        if (p <= previous or p > with_prompt) return error.InvalidCheckpointPosition;
        previous = p;
    }
    return .{ .cache_before = before, .required = required };
}

pub const KvBytes = struct { allocated: u64 = 0, valid: u64 = 0 };

/// metal.Buffer.len is BYTES (Context.alloc multiplies float count by 4).
/// Each K/V buffer can have its own byte stride; sliding ring validity saturates
/// at capacity instead of growing with the total sequence position.
pub fn layerKvBytes(keys_bytes: usize, values_bytes: usize, capacity: u32, position: u32) !KvBytes {
    if (capacity == 0 or keys_bytes == 0 or values_bytes == 0 or
        keys_bytes % capacity != 0 or values_bytes % capacity != 0) return error.InvalidKvGeometry;
    const allocated = try std.math.add(u64, @intCast(keys_bytes), @intCast(values_bytes));
    const stride = try std.math.add(u64, @intCast(keys_bytes / capacity), @intCast(values_bytes / capacity));
    return .{
        .allocated = allocated,
        .valid = try std.math.mul(u64, @min(position, capacity), stride),
    };
}

fn kvBytes(model: *const Model, position: u32) !KvBytes {
    var total: KvBytes = .{};
    for (model.layers) |layer| {
        const bytes = try layerKvBytes(layer.keys.len, layer.values.len, layer.capacity, position);
        total.allocated = try std.math.add(u64, total.allocated, bytes.allocated);
        total.valid = try std.math.add(u64, total.valid, bytes.valid);
    }
    return total;
}

fn finiteLogits(logits: []const f32, vocab_size: u32) !void {
    if (logits.len != vocab_size) return error.InvalidLogitCount;
    for (logits) |value| if (!std.math.isFinite(value)) return error.NonFiniteLogits;
}

fn now() !f64 {
    var ts: Timespec = undefined;
    // Darwin CLOCK_MONOTONIC, as used by benchmark.zig.
    if (clock_gettime(6, &ts) != 0) return error.ClockFailed;
    const seconds = @as(f64, @floatFromInt(ts.sec)) + @as(f64, @floatFromInt(ts.nsec)) * 1e-9;
    if (!std.math.isFinite(seconds)) return error.NonFiniteTiming;
    return seconds;
}

fn elapsed(start: f64, end: f64) !f64 {
    const seconds = end - start;
    if (!std.math.isFinite(seconds) or seconds < 0) return error.InvalidTiming;
    return seconds;
}

fn output(bytes: []const u8) !void {
    var offset: usize = 0;
    while (offset < bytes.len) {
        const n = write(1, bytes[offset..].ptr, bytes.len - offset);
        if (n <= 0) return error.OutputFailed;
        offset += @intCast(n);
    }
}

const Session = struct {
    model: *Model,
    gpa: Allocator,
    history: std.ArrayList(u32) = .empty,
    layers: []prefix_cache.Layer,
    checkpoints: rewind.Checkpoints,
    cache: ?prefix_cache.Cache = null,

    /// In-memory checkpoint at the current position. The device must be idle.
    fn checkpoint(self: *Session, pinned: bool, extra: *Extra) !void {
        try self.model.ctx.finish();
        const start = try now();
        try self.checkpoints.take(self.layers, self.model.position, pinned);
        if (extra.checkpoint_seconds) |*seconds| seconds.* += try elapsed(start, try now());
    }

    /// Disk snapshot of the current position. A failed write is reported, not
    /// fatal: the in-memory state is unchanged and stays exact.
    fn snapshot(self: *Session, extra: *Extra) !void {
        const cache = if (self.cache) |*c| c else return;
        try self.model.ctx.finish();
        const start = try now();
        const position = self.model.position;
        const written = cache.write(self.history.items[0..position], self.layers) catch |err| {
            std.debug.print("Prefix snapshot at {d} not written: {s}\n", .{ position, @errorName(err) });
            extra.snapshot_failures.? += 1;
            extra.snapshot_seconds.? += try elapsed(start, try now());
            return;
        };
        switch (written.outcome) {
            .written => {
                extra.snapshots_written.? += 1;
                extra.snapshot_bytes.? += written.bytes;
            },
            .existing => {},
            .too_large => std.debug.print("Prefix snapshot at {d} exceeds --prefix-cache-bytes; not written\n", .{position}),
        }
        extra.snapshot_seconds.? += try elapsed(start, try now());
    }

    /// Serve `truncate: n`: newest of (intact rings, memory checkpoint, disk
    /// snapshot of history[0..c]) at c <= n, then replay recorded history c..n.
    fn rewindTo(self: *Session, n: u32, extra: *Extra) !void {
        const model = self.model;
        const from = model.position;
        if (self.history.items.len != from) return error.HistoryPositionMismatch;
        try model.ctx.finish();
        var c: u32 = n;
        var source: []const u8 = "none";
        if (n != from and rewind.ringsIntact(self.layers, from, n)) {
            source = "ring_intact";
        } else if (n != from) {
            c = 0;
            source = "start";
            var memory = self.checkpoints.best(n);
            var disk = false;
            if (self.cache) |*cache| {
                const start = try now();
                // A failed lookup may have written into the layers: assume it did.
                const result = cache.restoreLongest(self.history.items[0..n], if (memory) |m| m.position else 0, n, self.layers) catch |err| failed: {
                    std.debug.print("Prefix cache lookup failed: {s}\n", .{@errorName(err)});
                    break :failed prefix_cache.Restored{ .clobbered = true };
                };
                extra.restore_seconds = try elapsed(start, try now());
                if (result.position != 0) {
                    c = result.position;
                    source = "disk";
                    disk = true;
                }
                // A failed load overwrote cached KV: only a replay from 0 is exact.
                if (result.clobbered and !disk) {
                    self.checkpoints.clear();
                    memory = null;
                }
            }
            if (!disk) if (memory) |m| {
                const start = try now();
                rewind.Checkpoints.restore(m, self.layers);
                if (extra.checkpoint_seconds) |*seconds| seconds.* += try elapsed(start, try now());
                c = m.position;
                source = "memory";
            };
        }
        self.checkpoints.dropAbove(n);
        try model.restorePosition(c);
        var replay_seconds: f64 = 0;
        if (n > c) {
            const start = try now();
            _ = try model.prefill(self.history.items[c..n]);
            replay_seconds = try elapsed(start, try now());
        }
        if (model.position != n) return error.CachePositionMismatch;
        self.history.shrinkRetainingCapacity(n);
        extra.truncated_from = from;
        extra.rewind_checkpoint = c;
        extra.replayed_tokens = n - c;
        extra.replay_seconds = replay_seconds;
        extra.rewind_source = source;
    }

    fn handle(self: *Session, line: []const u8) !void {
        const model = self.model;
        const gpa = self.gpa;
        const history = &self.history;
        var arena = std.heap.ArenaAllocator.init(gpa);
        defer arena.deinit();
        const a = arena.allocator();
        const request = try parse(a, line);
        const preflight = try validate(request, model.position, model.context, model.cfg.vocab_size);
        if (model.failed and !request.reset) return error.ModelNeedsReset;
        // Validate geometry and worst-case accounting before reset or any forward.
        _ = try kvBytes(model, @intCast(preflight.required));
        const generated = try a.alloc(u32, request.max_tokens);
        const prefill_copy: ?[]f32 = if (request.return_logits) try a.alloc(f32, model.cfg.vocab_size) else null;
        const final_copy: ?[]f32 = if (request.return_logits) try a.alloc(f32, model.cfg.vocab_size) else null;
        // Also detect a failed clock before mutating the model.
        _ = try now();
        try history.ensureUnusedCapacity(gpa, request.tokens.len + request.max_tokens + 1);
        var extra: Extra = .{};
        if (self.cache != null) extra = .{ .restored_prefix_tokens = 0, .restore_seconds = 0, .snapshots_written = 0, .snapshot_bytes = 0, .snapshot_seconds = 0, .snapshot_failures = 0 };
        if (self.cache != null or request.truncate != null or request.checkpoint_at.len != 0) extra.checkpoint_seconds = 0;
        if (request.checkpoint_at.len != 0) extra.checkpoints_taken = 0;
        var restored: u32 = 0;
        if (request.reset) {
            try model.reset();
            history.clearRetainingCapacity();
            self.checkpoints.clear();
            if (self.cache) |*cache| {
                // Leave at least one token to prefill: it produces the logits.
                const start = try now();
                // A failed lookup leaves position 0: prefilling everything is exact.
                const result = cache.restoreLongest(request.tokens, 0, @intCast(request.tokens.len - 1), self.layers) catch |err| failed: {
                    std.debug.print("Prefix cache lookup failed: {s}\n", .{@errorName(err)});
                    break :failed prefix_cache.Restored{};
                };
                if (result.position != 0) {
                    try model.restorePosition(result.position);
                    restored = result.position;
                    history.appendSliceAssumeCapacity(request.tokens[0..restored]);
                }
                extra.restore_seconds = try elapsed(start, try now());
                extra.restored_prefix_tokens = restored;
                if (restored != 0) try self.checkpoint(true, &extra);
            }
        } else if (request.truncate) |n| try self.rewindTo(n, &extra);
        const before = preflight.cache_before;
        if (model.position != before + restored or history.items.len != model.position) return error.CachePositionMismatch;
        history.appendSliceAssumeCapacity(request.tokens[restored..]);

        // Prefill in segments ending at each checkpoint_at position. Positions
        // inside a restored snapshot are already on disk; the restore point
        // itself is an in-memory checkpoint above.
        var prefill_seconds: f64 = 0;
        var logits: []const f32 = &.{};
        var start: usize = restored;
        for (request.checkpoint_at) |p| {
            const end = p - before;
            if (end < restored) continue;
            if (end > start) {
                const segment = try now();
                logits = try model.prefill(request.tokens[start..end]);
                prefill_seconds += try elapsed(segment, try now());
                start = end;
            }
            try self.checkpoint(true, &extra);
            if (end > restored) try self.snapshot(&extra);
            extra.checkpoints_taken.? += 1;
        }
        if (start < request.tokens.len) {
            const segment = try now();
            logits = try model.prefill(request.tokens[start..]);
            prefill_seconds += try elapsed(segment, try now());
        }
        try finiteLogits(logits, model.cfg.vocab_size);
        if (prefill_copy) |copy| @memcpy(copy, logits);
        try self.checkpoint(false, &extra);

        const decode_start = try now();
        const result = try @import("speculative_generation.zig").generate(model, gpa, history, generated, logits);
        const count = result.count;
        const eos = result.eos;
        logits = result.logits;
        try finiteLogits(logits, model.cfg.vocab_size);
        const decode_seconds = try elapsed(decode_start, try now());
        if (final_copy) |copy| @memcpy(copy, logits);
        const expected: u64 = @as(u64, preflight.cache_before) + request.tokens.len + count;
        if (model.position != expected) return error.CachePositionMismatch;
        if (count != 0) try self.checkpoint(false, &extra);
        const bytes = try kvBytes(model, model.position);
        const response: Response = .{
            .id = request.id,
            .tokens = generated[0..count],
            .eos = eos,
            .cache_before = preflight.cache_before,
            .cache_after = model.position,
            .prefill_tokens = request.tokens.len,
            .decode_forward_tokens = count,
            .reused_prefix_tokens = preflight.cache_before,
            .prefill_seconds = prefill_seconds,
            .decode_seconds = decode_seconds,
            .kv_allocated_bytes = bytes.allocated,
            .kv_valid_bytes = bytes.valid,
            .kv_storage = if (model.kv_bf16) "bf16_global_fp32_sliding" else "fp32",
            .speculative_batches = result.batches,
            .speculative_proposed = result.proposed,
            .speculative_accepted = result.accepted,
            .prefill_logits = prefill_copy,
            .final_logits = final_copy,
        };
        // Serialize completely before writing: failures never emit fake success or
        // a partially serialized response. Transport failures still propagate.
        const json = try merge(a, try std.json.Stringify.valueAlloc(a, response, .{}), try std.json.Stringify.valueAlloc(a, extra, .{ .emit_null_optional_fields = false }));
        try output(json);
        try output("\n");
    }
};

fn kvLayers(gpa: Allocator, model: *const Model) ![]prefix_cache.Layer {
    const layers = try gpa.alloc(prefix_cache.Layer, model.layers.len);
    for (model.layers, layers, 0..) |layer, *view, i| view.* = .{
        .keys = layer.keys.structs(u8),
        .values = layer.values.structs(u8),
        .capacity = layer.capacity,
        .ring = model.cfg.sliding(i),
    };
    return layers;
}

fn hashExecutable(digest: *[32]u8) !void {
    var buffer: [4096]u8 = undefined;
    var size: u32 = buffer.len;
    if (std.c._NSGetExecutablePath(&buffer, &size) != 0) return error.ExecutablePathFailed;
    const path = std.mem.sliceTo(@as([*:0]u8, @ptrCast(&buffer)), 0);
    const fd = std.c.open(path, .{ .ACCMODE = .RDONLY });
    if (fd < 0) return error.ExecutableReadFailed;
    defer _ = std.c.close(fd);
    var h = Sha256.init(.{});
    var chunk: [65536]u8 = undefined;
    while (true) {
        const n = read(fd, &chunk, chunk.len);
        if (n < 0) return error.ExecutableReadFailed;
        if (n == 0) break;
        h.update(chunk[0..@intCast(n)]);
    }
    h.final(digest);
}

pub const shard_samples = 32;
pub const shard_sample_bytes = 4096;

/// The safetensors header (tensor names, dtypes, offsets) plus `shard_samples`
/// evenly spaced 4 KiB pages: a re-quantized checkpoint with the same sizes
/// almost surely differs in them. About 128 KiB of reads per shard, not a hash
/// of the weights: an edit that avoids every sampled page goes undetected.
pub fn sampleShard(h: *Sha256, bytes: []const u8) void {
    if (bytes.len >= 8) {
        const header = std.mem.readInt(u64, bytes[0..8], .little);
        h.update(bytes[0..@intCast(@min(bytes.len, 8 +| header))]);
    }
    const page = @min(bytes.len, shard_sample_bytes);
    for (0..shard_samples) |i| {
        const start = (bytes.len - page) * i / (shard_samples - 1);
        h.update(bytes[start..][0..page]);
    }
}

/// Snapshot identity and settings. Identity: config bytes plus every shard's
/// name, size, header and sampled pages (no 22 GB hash). Settings: device, this exact executable (any
/// rebuild invalidates), and every Model field that selects kernels, storage
/// or batched verification, since any of them can change stored bits.
pub fn snapshotKey(model: *const Model) !prefix_cache.Key {
    var key: prefix_cache.Key = undefined;
    var h = Sha256.init(.{});
    h.update("flint-prefix-identity-v2\x00");
    h.update(std.mem.asBytes(&@as(u64, model.config_map.bytes.len)));
    h.update(model.config_map.bytes);
    for (model.store.shards.items) |shard| {
        h.update(std.mem.asBytes(&@as(u64, shard.name.len)));
        h.update(shard.name);
        h.update(std.mem.asBytes(&@as(u64, shard.mapping.bytes.len)));
        sampleShard(&h, shard.mapping.bytes);
    }
    h.final(&key.identity);
    var executable: [32]u8 = undefined;
    try hashExecutable(&executable);
    const m = model;
    const resident_workers: u32 = if (m.resident_execution) |e| e.workers else 0;
    const resident_row_tile: u32 = if (m.resident_execution) |e| e.descriptor.structs(@import("resident.zig").Descriptor)[0].row_tile else 0;
    const fields = .{
        .{ "mlx_attention", @intFromBool(m.mlx_attention) },                   .{ "kv_bf16", @intFromBool(m.kv_bf16) },
        .{ "bf16_blocks", m.bf16_blocks },                                     .{ "bf16_vector_loads", @intFromBool(m.bf16_vector_loads) },
        .{ "staged_row_tile", m.staged_row_tile },                             .{ "head_row_tile", m.head_row_tile },
        .{ "qkv_row_tile", m.qkv_row_tile },                                   .{ "gate_row_tile", m.gate_row_tile },
        .{ "projection_row_tile", m.projection_row_tile },                     .{ "down_row_tile", m.down_row_tile },
        .{ "attention_chunk", m.attention_chunk },                             .{ "split_attention", @intFromBool(m.splitAttentionSupported()) },
        .{ "resident_workers", resident_workers },                             .{ "resident_row_tile", resident_row_tile },
        .{ "packed_router", @intFromBool(m.packed_router) },                   .{ "fixed_loops", @intFromBool(m.fixed_loops) },
        .{ "affine_loads", @intFromBool(m.affine_loads) },                     .{ "fixed_attention", @intFromBool(m.fixed_attention) },
        .{ "gqa_reuse", m.gqa_reuse },                                         .{ "gqa_blocks", m.gqa_blocks },
        .{ "gqa_prefetch", m.gqa_prefetch },                                   .{ "wide_attention", @intFromBool(m.wide_attention) },
        .{ "ring_attention", @intFromBool(m.ring_attention) },                 .{ "prepared_inputs", @intFromBool(m.prepared_inputs) },
        .{ "prepared_fused_norm", @intFromBool(m.prepared_fused_norm) },       .{ "prepared_fused_gate", @intFromBool(m.prepared_fused_gate) },
        .{ "parallel_down", @intFromBool(m.parallel_down) },                   .{ "speculative_tokens", m.speculative_tokens },
        .{ "batch_query_group", m.batch_query_group },                         .{ "batch_weight_reuse", @intFromBool(m.batch_weight_reuse) },
        .{ "batch_mlp_weight_reuse", @intFromBool(m.batch_mlp_weight_reuse) }, .{ "batch_shared_attention", @intFromBool(m.batch_shared_attention) },
    };
    var settings = Sha256.init(.{});
    var text: [256]u8 = undefined;
    settings.update(try std.fmt.bufPrint(&text, "flint-prefix-settings-v1 device={s} executable={s}", .{ m.ctx.deviceName(), &std.fmt.bytesToHex(executable, .lower) }));
    inline for (fields) |field| settings.update(try std.fmt.bufPrint(&text, " {s}={d}", .{ field[0], @as(u32, field[1]) }));
    settings.final(&key.settings);
    return key;
}

/// Consume stdin through EOF, retaining the caller's Model between requests.
/// Propagate every invalid request / I/O / inference error to main (!void), so
/// the process exits nonzero; stderr has a diagnostic, stdout has no error JSON.
pub fn run(model: *Model, gpa: Allocator, options: Options) !void {
    return runImpl(model, gpa, options) catch |err| {
        std.debug.print("Session failed: {s}\n", .{@errorName(err)});
        return err;
    };
}

fn runImpl(model: *Model, gpa: Allocator, options: Options) !void {
    if (options.rewind_checkpoints > max_rewind_checkpoints) return error.InvalidRewindCheckpoints;
    const storage = try gpa.alloc(u8, max_line_bytes);
    defer gpa.free(storage);
    var lines: LineBuffer = .{ .storage = storage };
    var session: Session = .{
        .model = model,
        .gpa = gpa,
        .layers = try kvLayers(gpa, model),
        .checkpoints = try rewind.Checkpoints.init(gpa, options.rewind_checkpoints),
    };
    defer {
        session.history.deinit(gpa);
        session.checkpoints.deinit();
        gpa.free(session.layers);
        if (session.cache) |*cache| cache.deinit();
    }
    if (options.prefix_cache) |directory|
        session.cache = try prefix_cache.Cache.init(gpa, directory, options.prefix_cache_bytes, try snapshotKey(model));
    var chunk: [8192]u8 = undefined;
    while (true) {
        const n = read(0, &chunk, chunk.len);
        if (n < 0) return error.InputFailed;
        if (n == 0) {
            if (lines.finish()) |line| try session.handle(line);
            return;
        }
        for (chunk[0..@intCast(n)]) |byte| {
            if (try lines.push(byte)) |line| try session.handle(line);
        }
    }
}

test "session EOF preserves final partial line and has no phantom request" {
    var storage: [32]u8 = undefined;
    var lines: LineBuffer = .{ .storage = &storage };
    try std.testing.expect(lines.finish() == null);
    for ("one\ntwo") |byte| {
        if (try lines.push(byte)) |line| try std.testing.expectEqualStrings("one", line);
    }
    try std.testing.expectEqualStrings("two", lines.finish().?);
    try std.testing.expect(lines.finish() == null);
    try std.testing.expectEqualStrings("", (try lines.push('\n')).?);
    try std.testing.expect(lines.finish() == null);
}

test "session line size accepts exact 1 MiB and rejects overflow before write" {
    const storage = try std.testing.allocator.alloc(u8, max_line_bytes + 1);
    defer std.testing.allocator.free(storage);
    var lines: LineBuffer = .{ .storage = storage };
    for (0..max_line_bytes) |_| try std.testing.expect((try lines.push(' ')) == null);
    try std.testing.expectError(error.LineTooLong, lines.push('x'));
    try std.testing.expectEqual(@as(usize, max_line_bytes), lines.len);
    try std.testing.expectEqual(@as(usize, max_line_bytes), (try lines.push('\n')).?.len);
    try std.testing.expect((try lines.push('x')) == null);
    try std.testing.expectEqualStrings("x", lines.finish().?);
    try std.testing.expectError(error.LineTooLong, parse(std.testing.allocator, storage));
}

test "session default request and explicit zero generation" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const request = try parse(arena.allocator(), "{\"tokens\":[2,3]}");
    try std.testing.expectEqual(@as(u32, 0), request.id);
    try std.testing.expectEqual(@as(u32, 128), request.max_tokens);
    try std.testing.expect(!request.reset and !request.return_logits);
    try std.testing.expectEqualSlices(u32, &.{ 2, 3 }, request.tokens);
    const zero = try parse(arena.allocator(), "{\"id\":9,\"tokens\":[1],\"max_tokens\":0,\"reset\":true,\"return_logits\":true}");
    try std.testing.expectEqual(@as(u32, 9), zero.id);
    try std.testing.expectEqual(@as(u32, 0), zero.max_tokens);
    try std.testing.expect(zero.reset and zero.return_logits);
    const checked = try validate(zero, 10, 10, 4);
    try std.testing.expectEqual(@as(u32, 0), checked.cache_before);
    try std.testing.expectEqual(@as(u64, 1), checked.required);
}

test "session rejects missing empty malformed and coercible request fields" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    try std.testing.expectError(error.MissingTokens, parse(a, "{}"));
    try std.testing.expectError(error.EmptyPrompt, parse(a, "{\"tokens\":[]}"));
    try std.testing.expectError(error.InvalidTokens, parse(a, "{\"tokens\":null}"));
    try std.testing.expectError(error.InvalidRequestObject, parse(a, "[]"));
    try std.testing.expectError(error.InvalidRequestInteger, parse(a, "{\"tokens\":[-1]}"));
    try std.testing.expectError(error.InvalidRequestInteger, parse(a, "{\"tokens\":[4294967296]}"));
    try std.testing.expectError(error.InvalidRequestInteger, parse(a, "{\"tokens\":[\"1\"]}"));
    try std.testing.expectError(error.InvalidRequestInteger, parse(a, "{\"tokens\":[1.0]}"));
    try std.testing.expectError(error.InvalidRequestBoolean, parse(a, "{\"tokens\":[1],\"reset\":1}"));
    try std.testing.expectError(error.UnknownRequestField, parse(a, "{\"tokens\":[1],\"typo\":0}"));
    try std.testing.expectError(error.DuplicateField, parse(a, "{\"tokens\":[1],\"tokens\":[2]}"));
}

test "session preflight validates every token before a requested reset" {
    const request: Request = .{ .tokens = &.{ 1, 4 }, .max_tokens = 0, .reset = true };
    try std.testing.expectError(error.InvalidToken, validate(request, 99, 4, 4));
    try std.testing.expectError(error.EmptyPrompt, validate(.{ .tokens = &.{} }, 0, 200, 4));
    try std.testing.expectError(error.InvalidContextLength, validate(.{ .tokens = &.{0} }, 0, 0, 4));
    try std.testing.expectError(error.InvalidVocabulary, validate(.{ .tokens = &.{0} }, 0, 200, 0));
}

test "session context preflight reserves final commit and uses u64 arithmetic" {
    const request: Request = .{ .tokens = &.{1}, .max_tokens = 2 };
    const checked = try validate(request, 7, 10, 4);
    try std.testing.expectEqual(@as(u64, 10), checked.required);
    try std.testing.expectEqual(@as(u32, 7), checked.cache_before);
    try std.testing.expectError(error.ContextExhausted, validate(request, 8, 10, 4));
    try std.testing.expectError(error.ContextExhausted, validate(.{ .tokens = &.{1}, .max_tokens = std.math.maxInt(u32) }, 1, std.math.maxInt(u32), 4));
    try std.testing.expectError(error.ContextExhausted, validate(.{ .tokens = &.{1}, .max_tokens = 0 }, std.math.maxInt(u32), std.math.maxInt(u32), 4));
    const reset = try validate(.{ .tokens = &.{1}, .max_tokens = 2, .reset = true }, 10, 10, 4);
    try std.testing.expectEqual(@as(u32, 0), reset.cache_before);
    try std.testing.expectEqual(@as(u64, 3), reset.required);
}

test "session KV accounting uses byte lengths and saturates sliding validity" {
    const empty = try layerKvBytes(64, 128, 4, 0);
    try std.testing.expectEqual(@as(u64, 192), empty.allocated);
    try std.testing.expectEqual(@as(u64, 0), empty.valid);
    try std.testing.expectEqual(@as(u64, 96), (try layerKvBytes(64, 128, 4, 2)).valid);
    try std.testing.expectEqual(@as(u64, 192), (try layerKvBytes(64, 128, 4, 100)).valid);
    try std.testing.expectError(error.InvalidKvGeometry, layerKvBytes(64, 128, 0, 1));
    try std.testing.expectError(error.InvalidKvGeometry, layerKvBytes(65, 128, 4, 1));
}

test "session finite guards include final logits and timing" {
    try finiteLogits(&.{ 1, -2 }, 2);
    try std.testing.expectError(error.InvalidLogitCount, finiteLogits(&.{1}, 2));
    try std.testing.expectError(error.NonFiniteLogits, finiteLogits(&.{ 1, std.math.nan(f32) }, 2));
    try std.testing.expectError(error.NonFiniteLogits, finiteLogits(&.{std.math.inf(f32)}, 1));
    try std.testing.expectEqual(@as(f64, 0.5), try elapsed(1, 1.5));
    try std.testing.expectError(error.InvalidTiming, elapsed(2, 1));
    try std.testing.expectError(error.InvalidTiming, elapsed(0, std.math.inf(f64)));
}

test "session rewind and checkpoint fields parse strictly and validate before mutation" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const request = try parse(a, "{\"tokens\":[1,2,3],\"truncate\":4,\"checkpoint_at\":[5,7]}");
    try std.testing.expectEqual(@as(?u32, 4), request.truncate);
    try std.testing.expectEqualSlices(u32, &.{ 5, 7 }, request.checkpoint_at);
    const checked = try validate(request, 9, 200, 4);
    try std.testing.expectEqual(@as(u32, 4), checked.cache_before);
    try std.testing.expectError(error.ContextExhausted, validate(request, 9, 16, 4));
    try std.testing.expectEqual(@as(u64, 4 + 1 + 128), (try validate(.{ .tokens = &.{1}, .truncate = 4 }, 9, 200, 4)).required);
    try std.testing.expectError(error.InvalidTruncate, validate(request, 3, 16, 4));
    try std.testing.expectError(error.TruncateWithReset, validate(.{ .tokens = &.{1}, .truncate = 0, .reset = true }, 3, 16, 4));
    try std.testing.expectError(error.InvalidRequestInteger, parse(a, "{\"tokens\":[1],\"truncate\":-1}"));
    try std.testing.expectError(error.InvalidRequestInteger, parse(a, "{\"tokens\":[1],\"truncate\":null}"));
    try std.testing.expectError(error.InvalidCheckpointPositions, parse(a, "{\"tokens\":[1],\"checkpoint_at\":3}"));
    try std.testing.expectError(error.InvalidRequestInteger, parse(a, "{\"tokens\":[1],\"checkpoint_at\":[\"3\"]}"));
    var many: std.ArrayList(u8) = .empty;
    try many.appendSlice(a, "{\"tokens\":[1],\"checkpoint_at\":[");
    for (0..max_checkpoint_at + 1) |i| try many.appendSlice(a, if (i == 0) "1" else ",1");
    try many.appendSlice(a, "]}");
    try std.testing.expectError(error.TooManyCheckpointPositions, parse(a, many.items));
    // Positions must strictly increase inside (cache_before, cache_before + tokens].
    const span: Request = .{ .tokens = &.{ 1, 2, 3 }, .max_tokens = 0 };
    for ([_][]const u32{ &.{3}, &.{ 4, 4 }, &.{ 5, 4 }, &.{7} }) |positions| {
        var bad = span;
        bad.checkpoint_at = positions;
        try std.testing.expectError(error.InvalidCheckpointPosition, validate(bad, 3, 16, 4));
    }
    var good = span;
    good.checkpoint_at = &.{ 4, 6 };
    _ = try validate(good, 3, 16, 4);
    good.reset = true;
    good.checkpoint_at = &.{ 1, 3 };
    _ = try validate(good, 3, 16, 4);
}

test "snapshot identity samples shard header and pages without reading everything" {
    const bytes = try std.testing.allocator.alloc(u8, 1 << 20);
    defer std.testing.allocator.free(bytes);
    for (bytes, 0..) |*b, i| b.* = @truncate(i *% 31);
    std.mem.writeInt(u64, bytes[0..8], 100, .little);
    const digest = struct {
        fn of(data: []const u8) [32]u8 {
            var h = Sha256.init(.{});
            sampleShard(&h, data);
            var out: [32]u8 = undefined;
            h.final(&out);
            return out;
        }
    }.of;
    const base = digest(bytes);
    try std.testing.expectEqual(base, digest(bytes));
    for ([_]usize{ 50, 107, bytes.len - 1 }) |offset| {
        bytes[offset] ^= 1;
        try std.testing.expect(!std.mem.eql(u8, &base, &digest(bytes)));
        bytes[offset] ^= 1;
    }
    // A byte between sampled pages is not covered.
    bytes[shard_sample_bytes + 1] ^= 1;
    try std.testing.expectEqual(base, digest(bytes));
    _ = digest(bytes[0..3]);
    _ = digest(&.{});
}

test "session extra fields are appended only when present" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const none = try std.json.Stringify.valueAlloc(a, Extra{}, .{ .emit_null_optional_fields = false });
    try std.testing.expectEqualStrings("{}", none);
    try std.testing.expectEqualStrings("{\"id\":1}", try merge(a, "{\"id\":1}", none));
    const some = try std.json.Stringify.valueAlloc(a, Extra{ .truncated_from = 9, .rewind_source = "memory" }, .{ .emit_null_optional_fields = false });
    try std.testing.expectEqualStrings("{\"id\":1,\"truncated_from\":9,\"rewind_source\":\"memory\"}", try merge(a, "{\"id\":1}", some));
    try std.testing.expectError(error.InvalidResponseJson, merge(a, "[1]", some));
}
