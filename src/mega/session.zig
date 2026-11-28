//! Persistent JSONL session over the megakernel, the `flint session` protocol:
//! request {id, tokens, max_tokens, reset, return_logits}; tokens are only the
//! new suffix; every generated token (EOS included) is committed to state.
const std = @import("std");
const exec = @import("exec.zig");
const dflash2 = @import("dflash2.zig");
const inkling_vision = @import("inkling_vision.zig");
const io = @import("io.zig");
const kv_cartridge = @import("kv_cartridge.zig");
const Allocator = std.mem.Allocator;

extern "c" fn read(c_int, [*]u8, usize) isize;

pub const max_line_bytes = 64 * 1024 * 1024;

pub const Request = struct {
    id: u32 = 0,
    tokens: []const u32,
    images: []const []const u8 = &.{},
    max_image_tokens: u32 = inkling_vision.default_max_image_tokens,
    max_tokens: u32 = 128,
    reset: bool = false,
    return_logits: bool = false,
    kv_in: ?[]const u8 = null,
    kv_out: ?[]const u8 = null,
    kv_dtype: ?[]const u8 = null,
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
    prefill_forwards: usize,
    dispatches: u64,
    prefill_logits: ?[]const f32,
    final_logits: ?[]const f32,
    image_token_counts: ?[]const u32 = null,
};

/// Bounded framing; same rules as src/session.zig's LineBuffer.
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
fn jsonString(value: std.json.Value) ![]const u8 {
    return switch (value) {
        .string => |s| s,
        else => error.InvalidRequestString,
    };
}

/// Rejects coercions, duplicate and unknown fields, and missing tokens.
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
        inline for (.{ "id", "tokens", "images", "max_image_tokens", "max_tokens", "reset", "return_logits", "kv_in", "kv_out", "kv_dtype" }) |known| {
            if (std.mem.eql(u8, name, known)) break;
        } else return error.UnknownRequestField;
    }
    const token_value = object.get("tokens") orelse return error.MissingTokens;
    const items = switch (token_value) {
        .array => |array| array.items,
        else => return error.InvalidTokens,
    };
    if (items.len == 0) return error.EmptyPrompt;
    const tokens = try a.alloc(u32, items.len);
    for (items, tokens) |item, *token| token.* = try jsonU32(item);
    var images: []const []const u8 = &.{};
    if (object.get("images")) |iv| {
        const arr = switch (iv) {
            .array => |array| array.items,
            else => return error.InvalidImages,
        };
        const out = try a.alloc([]const u8, arr.len);
        for (arr, out) |item, *dst| dst.* = try jsonString(item);
        images = out;
    }
    return .{
        .id = if (object.get("id")) |v| try jsonU32(v) else 0,
        .tokens = tokens,
        .images = images,
        .max_image_tokens = if (object.get("max_image_tokens")) |v| try jsonU32(v) else inkling_vision.default_max_image_tokens,
        .max_tokens = if (object.get("max_tokens")) |v| try jsonU32(v) else 128,
        .reset = if (object.get("reset")) |v| try jsonBool(v) else false,
        .return_logits = if (object.get("return_logits")) |v| try jsonBool(v) else false,
        .kv_in = if (object.get("kv_in")) |v| try jsonString(v) else null,
        .kv_out = if (object.get("kv_out")) |v| try jsonString(v) else null,
        .kv_dtype = if (object.get("kv_dtype")) |v| try jsonString(v) else null,
    };
}

pub const Preflight = struct { cache_before: u32, required: u64 };

/// Reserve a forward for every generated token, including EOS and the last.
pub fn validate(request: Request, position: u32, context: u32, vocab: u32) !Preflight {
    if (request.tokens.len == 0) return error.EmptyPrompt;
    for (request.tokens) |token| if (token >= vocab) return error.InvalidToken;
    const before = if (request.reset) @as(u32, 0) else position;
    const required = @as(u64, before) + request.tokens.len + request.max_tokens;
    if (required > context) return error.ContextExhausted;
    return .{ .cache_before = before, .required = required };
}

fn finite(logits: []const f32) !void {
    for (logits) |v| if (!std.math.isFinite(v)) return error.NonFiniteLogits;
}

/// Expand single `<|unused_200054|>` image placeholders into `counts[k]` copies if
/// `tokens` has one placeholder per image; otherwise verify exact total count match.
pub fn expandImagePlaceholders(a: Allocator, tokens: []const u32, counts: []const u32, total_vis: u32, token_id: u32) ![]const u32 {
    var placeholders: usize = 0;
    for (tokens) |t| if (t == token_id) {
        placeholders += 1;
    };
    if (placeholders == total_vis) return tokens;
    if (placeholders != counts.len) return error.VisionTokenCountMismatch;
    var out: std.ArrayList(u32) = .empty;
    try out.ensureTotalCapacity(a, tokens.len - placeholders + total_vis);
    var img_i: usize = 0;
    for (tokens) |t| {
        if (t == token_id) {
            for (0..counts[img_i]) |_| try out.append(a, token_id);
            img_i += 1;
        } else {
            try out.append(a, t);
        }
    }
    return out.items;
}

fn handle(engine: *exec.Engine, gpa: Allocator, line: []const u8, draft: ?*dflash2.Draft, vision: ?*const inkling_vision.VisionEncoder) !void {
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    const request = try parse(a, line);
    if (draft != null and request.kv_in != null) return error.DraftWithKvCartridgeUnsupported;

    var encoded_batch: ?inkling_vision.EncodedBatch = null;
    defer if (encoded_batch) |*eb| eb.deinit(gpa);
    var prompt_tokens: []const u32 = request.tokens;
    if (request.images.len > 0) {
        if (request.max_image_tokens == 0) return error.InvalidRequestInteger;
        const enc = vision orelse return error.VisionUnsupported;
        encoded_batch = try enc.encodeFiles(a, request.images, request.max_image_tokens);
        prompt_tokens = try expandImagePlaceholders(a, request.tokens, encoded_batch.?.counts, encoded_batch.?.total_tokens, inkling_vision.image_token_id);
    }

    if (request.kv_in) |kv_path| {
        _ = try kv_cartridge.load(engine, a, kv_path);
    }
    var check_req = request;
    check_req.tokens = prompt_tokens;
    if (request.kv_in != null) check_req.reset = false;
    const preflight = try validate(check_req, engine.position, engine.context, engine.program.meta.vocab);
    if (engine.failed and !check_req.reset and request.kv_in == null) return error.ModelNeedsReset;
    const vocab = engine.program.meta.vocab;
    const generated = try a.alloc(u32, request.max_tokens);
    const prefill_copy: ?[]f32 = if (request.return_logits) try a.alloc(f32, vocab) else null;
    const final_copy: ?[]f32 = if (request.return_logits) try a.alloc(f32, vocab) else null;
    if (request.kv_in == null and request.reset) {
        engine.reset();
        if (draft) |d| d.reset();
    }

    var vis_stream: ?exec.VisionStream = if (encoded_batch) |eb|
        try exec.VisionStream.init(engine.ctx, engine.batch, eb.buffer, eb.width, eb.total_tokens, inkling_vision.image_token_id)
    else
        null;
    defer if (vis_stream) |*vs| vs.deinit();

    var dispatches: u64 = 0;
    var forwards: usize = 0;
    const prefill_start = try io.now();
    var row: u32 = 0;
    if (draft) |d| {
        const pf = try d.prefillWithVision(engine, prompt_tokens, if (vis_stream) |*vs| vs else null);
        dispatches += pf.dispatches;
        forwards += pf.forwards;
        row = pf.last_row;
    } else {
        var i: usize = 0;
        while (i < prompt_tokens.len) {
            const n = @min(engine.batch, prompt_tokens.len - i);
            const chunk = prompt_tokens[i..][0..n];
            const last = i + n == prompt_tokens.len;
            var ov_val: ?exec.EmbedOverride = if (vis_stream) |*vs| try vs.prepareChunk(chunk) else null;
            engine.embed_override = if (ov_val) |*ov| ov else null;
            defer engine.embed_override = null;
            try engine.forward(chunk, if (last) @intCast(n - 1) else @intCast(n));
            dispatches += engine.last.dispatches;
            forwards += 1;
            row = @intCast(n - 1);
            i += n;
        }
        if (vis_stream) |*vs| try vs.finish();
    }
    try finite(try engine.logits(row));
    const prefill_seconds = try io.now() - prefill_start;
    if (prefill_copy) |c| @memcpy(c, try engine.logits(row));

    const decode_start = try io.now();
    var count: usize = 0;
    var eos = false;
    var spec_batches: usize = 0;
    var spec_proposed: usize = 0;
    var spec_accepted: usize = 0;
    if (draft) |d| {
        if (request.max_tokens > 0) {
            const first_tok = try engine.next(row);
            const res = try d.decodeGreedy(engine, first_tok, generated, .{ .stop_on_eos = true, .commit_all = true });
            count = res.count;
            eos = res.eos;
            row = res.last_row;
            spec_batches = res.batches;
            spec_proposed = res.proposed;
            spec_accepted = res.accepted;
            dispatches += res.dispatches;
        }
    } else {
        while (count < request.max_tokens) {
            const token = try engine.next(row);
            try engine.forward(&.{token}, 0);
            dispatches += engine.last.dispatches;
            row = 0;
            generated[count] = token;
            count += 1;
            if (std.mem.indexOfScalar(u32, engine.program.meta.eos, token) != null) {
                eos = true;
                break;
            }
        }
    }
    try finite(try engine.logits(row));
    const decode_seconds = try io.now() - decode_start;
    if (final_copy) |c| @memcpy(c, try engine.logits(row));
    if (engine.position != preflight.cache_before + prompt_tokens.len + count) return error.CachePositionMismatch;
    if (request.kv_out) |out_path| {
        const dtype = if (request.kv_dtype) |ds| kv_cartridge.parseDType(ds) orelse return error.InvalidKvDType else .f32;
        _ = try kv_cartridge.save(engine, a, out_path, dtype, "{}");
    }
    const bytes = engine.stateBytes(engine.position);
    const response: Response = .{
        .id = request.id,
        .tokens = generated[0..count],
        .eos = eos,
        .cache_before = preflight.cache_before,
        .cache_after = engine.position,
        .prefill_tokens = prompt_tokens.len,
        .decode_forward_tokens = count,
        .reused_prefix_tokens = preflight.cache_before,
        .prefill_seconds = prefill_seconds,
        .decode_seconds = decode_seconds,
        .kv_allocated_bytes = bytes.allocated,
        .kv_valid_bytes = bytes.valid,
        .speculative_batches = spec_batches,
        .speculative_proposed = spec_proposed,
        .speculative_accepted = spec_accepted,
        .prefill_forwards = forwards,
        .dispatches = dispatches,
        .prefill_logits = prefill_copy,
        .final_logits = final_copy,
        .image_token_counts = if (encoded_batch) |eb| eb.counts else null,
    };
    const json = try std.json.Stringify.valueAlloc(a, response, .{});
    try io.out(json);
    try io.out("\n");
}

pub fn run(engine: *exec.Engine, gpa: Allocator) !void {
    return runWithVision(engine, gpa, null, null);
}

pub fn runWithDraft(engine: *exec.Engine, gpa: Allocator, draft: ?*dflash2.Draft) !void {
    return runWithVision(engine, gpa, draft, null);
}

pub fn runWithVision(engine: *exec.Engine, gpa: Allocator, draft: ?*dflash2.Draft, vision: ?*const inkling_vision.VisionEncoder) !void {
    return runImpl(engine, gpa, draft, vision) catch |err| {
        std.debug.print("Session failed: {s}\n", .{@errorName(err)});
        return err;
    };
}

fn runImpl(engine: *exec.Engine, gpa: Allocator, draft: ?*dflash2.Draft, vision: ?*const inkling_vision.VisionEncoder) !void {
    var line_buf: std.ArrayList(u8) = .empty;
    defer line_buf.deinit(gpa);
    try line_buf.ensureTotalCapacity(gpa, 64 * 1024);
    var chunk: [65536]u8 = undefined;
    while (true) {
        const n = read(0, &chunk, chunk.len);
        if (n < 0) return error.InputFailed;
        if (n == 0) {
            if (line_buf.items.len > 0) {
                try handle(engine, gpa, line_buf.items, draft, vision);
                line_buf.clearRetainingCapacity();
            }
            return;
        }
        const data = chunk[0..@intCast(n)];
        var start: usize = 0;
        while (std.mem.indexOfScalarPos(u8, data, start, '\n')) |nl| {
            const part = data[start..nl];
            if (line_buf.items.len + part.len > max_line_bytes) return error.LineTooLong;
            try line_buf.appendSlice(gpa, part);
            try handle(engine, gpa, line_buf.items, draft, vision);
            line_buf.clearRetainingCapacity();
            start = nl + 1;
        }
        const tail = data[start..];
        if (tail.len > 0) {
            if (line_buf.items.len + tail.len > max_line_bytes) return error.LineTooLong;
            try line_buf.appendSlice(gpa, tail);
        }
    }
}

test "session request parsing matches the Flint protocol" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const r = try parse(a, "{\"tokens\":[2,3]}");
    try std.testing.expectEqual(@as(u32, 128), r.max_tokens);
    try std.testing.expect(!r.reset and !r.return_logits);
    try std.testing.expectError(error.UnknownRequestField, parse(a, "{\"tokens\":[1],\"typo\":0}"));
    try std.testing.expectError(error.InvalidRequestInteger, parse(a, "{\"tokens\":[1.0]}"));
    try std.testing.expectError(error.EmptyPrompt, parse(a, "{\"tokens\":[]}"));
    try std.testing.expectError(error.DuplicateField, parse(a, "{\"tokens\":[1],\"tokens\":[2]}"));
}

test "session preflight reserves every generated token and checks tokens first" {
    const req: Request = .{ .tokens = &.{1}, .max_tokens = 2 };
    try std.testing.expectEqual(@as(u64, 10), (try validate(req, 7, 10, 4)).required);
    try std.testing.expectError(error.ContextExhausted, validate(req, 8, 10, 4));
    try std.testing.expectError(error.InvalidToken, validate(.{ .tokens = &.{ 1, 4 }, .reset = true }, 9, 100, 4));
    try std.testing.expectEqual(@as(u32, 0), (try validate(.{ .tokens = &.{1}, .max_tokens = 0, .reset = true }, 10, 10, 4)).cache_before);
}

test "session framing keeps a final unterminated line" {
    var storage: [16]u8 = undefined;
    var lines: LineBuffer = .{ .storage = &storage };
    for ("one\ntwo") |byte| if (try lines.push(byte)) |line| try std.testing.expectEqualStrings("one", line);
    try std.testing.expectEqualStrings("two", lines.finish().?);
    try std.testing.expect(lines.finish() == null);
}

test "session request parses images and expands image placeholders" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const r = try parse(a, "{\"tokens\":[1,200054,2,200054,3],\"images\":[\"a.png\",\"b.jpg\"],\"max_image_tokens\":512}");
    try std.testing.expectEqual(@as(usize, 2), r.images.len);
    try std.testing.expectEqualStrings("a.png", r.images[0]);
    try std.testing.expectEqualStrings("b.jpg", r.images[1]);
    try std.testing.expectEqual(@as(u32, 512), r.max_image_tokens);
    try std.testing.expectError(error.InvalidImages, parse(a, "{\"tokens\":[1],\"images\":\"a.png\"}"));

    const expanded = try expandImagePlaceholders(a, r.tokens, &.{ 3, 2 }, 5, 200054);
    try std.testing.expectEqualSlices(u32, &.{ 1, 200054, 200054, 200054, 2, 200054, 200054, 3 }, expanded);
    const passthrough = try expandImagePlaceholders(a, expanded, &.{ 3, 2 }, 5, 200054);
    try std.testing.expectEqualSlices(u32, expanded, passthrough);
    try std.testing.expectError(error.VisionTokenCountMismatch, expandImagePlaceholders(a, &.{ 1, 200054, 2 }, &.{ 3, 2 }, 5, 200054));
}
