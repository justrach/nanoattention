//! Persistent, append-only JSONL inference. The caller owns the one loaded Model.
//! No sockets, prompt replay, implicit resets, or GPU work in the CPU-only tests.
const std = @import("std");
const Model = @import("model.zig").Model;
const sampling = @import("sampling.zig");
const Allocator = std.mem.Allocator;

extern "c" fn read(c_int, [*]u8, usize) isize;
extern "c" fn write(c_int, [*]const u8, usize) isize;
const Timespec = extern struct { sec: i64, nsec: i64 };
extern "c" fn clock_gettime(c_int, *Timespec) c_int;

pub const max_line_bytes = 1024 * 1024;

pub const Request = struct {
    id: u32 = 0,
    tokens: []const u32,
    max_tokens: u32 = 128,
    reset: bool = false,
    return_logits: bool = false,
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
    prefill_logits: ?[]const f32,
    final_logits: ?[]const f32,
};

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
            !std.mem.eql(u8, name, "return_logits")) return error.UnknownRequestField;
    }
    const token_value = object.get("tokens") orelse return error.MissingTokens;
    const items = switch (token_value) {
        .array => |array| array.items,
        else => return error.InvalidTokens,
    };
    if (items.len == 0) return error.EmptyPrompt;
    const tokens = try a.alloc(u32, items.len);
    for (items, tokens) |item, *token| token.* = try jsonU32(item);
    return .{
        .id = if (object.get("id")) |v| try jsonU32(v) else 0,
        .tokens = tokens,
        .max_tokens = if (object.get("max_tokens")) |v| try jsonU32(v) else 128,
        .reset = if (object.get("reset")) |v| try jsonBool(v) else false,
        .return_logits = if (object.get("return_logits")) |v| try jsonBool(v) else false,
    };
}

pub const Preflight = struct { cache_before: u32, required: u64 };

/// Reserve a forward for EVERY generated token, including EOS and the final
/// max_tokens token. Using u64 prevents the u32 context arithmetic wrapping.
/// A reset changes only the effective position here, never the actual model.
pub fn validate(request: Request, position: u32, context: u32, vocab_size: u32) !Preflight {
    if (request.tokens.len == 0) return error.EmptyPrompt;
    if (context == 0) return error.InvalidContextLength;
    if (vocab_size == 0) return error.InvalidVocabulary;
    for (request.tokens) |token| if (token >= vocab_size) return error.InvalidToken;
    const before = if (request.reset) @as(u32, 0) else position;
    const with_prompt = std.math.add(u64, before, @intCast(request.tokens.len)) catch return error.ContextExhausted;
    const required = std.math.add(u64, with_prompt, request.max_tokens) catch return error.ContextExhausted;
    if (required > context) return error.ContextExhausted;
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

fn handle(model: *Model, gpa: Allocator, line: []const u8) !void {
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
    if (request.reset) try model.reset();

    const prefill_start = try now();
    var logits = try model.prefill(request.tokens);
    try finiteLogits(logits, model.cfg.vocab_size);
    const prefill_seconds = try elapsed(prefill_start, try now());
    if (prefill_copy) |copy| @memcpy(copy, logits);

    var count: usize = 0;
    var eos = false;
    const decode_start = try now();
    while (count < request.max_tokens) {
        const token = try sampling.greedy(logits);
        generated[count] = token;
        count += 1;
        // Do not defer this forward to a later request: every returned token is
        // already committed, even if it is EOS or exhausts max_tokens.
        logits = try model.forward(token);
        try finiteLogits(logits, model.cfg.vocab_size);
        eos = std.mem.indexOfScalar(u32, model.cfg.eos_token_id, token) != null;
        if (eos) break;
    }
    const decode_seconds = try elapsed(decode_start, try now());
    if (final_copy) |copy| @memcpy(copy, logits);
    const expected: u64 = @as(u64, preflight.cache_before) + request.tokens.len + count;
    if (model.position != expected) return error.CachePositionMismatch;
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
        .prefill_logits = prefill_copy,
        .final_logits = final_copy,
    };
    // Serialize completely before writing: failures never emit fake success or
    // a partially serialized response. Transport failures still propagate.
    const json = try std.json.Stringify.valueAlloc(a, response, .{});
    try output(json);
    try output("\n");
}

/// Consume stdin through EOF, retaining the caller's Model between requests.
/// Propagate every invalid request / I/O / inference error to main (!void), so
/// the process exits nonzero; stderr has a diagnostic, stdout has no error JSON.
pub fn run(model: *Model, gpa: Allocator) !void {
    return runImpl(model, gpa) catch |err| {
        std.debug.print("Session failed: {s}\n", .{@errorName(err)});
        return err;
    };
}

fn runImpl(model: *Model, gpa: Allocator) !void {
    const storage = try gpa.alloc(u8, max_line_bytes);
    defer gpa.free(storage);
    var lines: LineBuffer = .{ .storage = storage };
    var chunk: [8192]u8 = undefined;
    while (true) {
        const n = read(0, &chunk, chunk.len);
        if (n < 0) return error.InputFailed;
        if (n == 0) {
            if (lines.finish()) |line| try handle(model, gpa, line);
            return;
        }
        for (chunk[0..@intCast(n)]) |byte| {
            if (try lines.push(byte)) |line| try handle(model, gpa, line);
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
