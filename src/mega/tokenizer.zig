//! Native Hugging Face tokenizer (a checkpoint's tokenizer.json): byte-level BPE
//! with added tokens. Encoding and decoding are identical to the `tokenizers`
//! library (0.23.3) for every file it accepts; tools/nano_tokenizer_check.py is
//! the evidence. Anything in the file this code does not implement is an
//! error naming the feature (`Diagnostic`), never silently different ids.
//!
//! Encode: input bytes (invalid UTF-8 -> U+FFFD per maximal invalid subpart,
//! as Python's "replace" and Rust's from_utf8_lossy) -> split on added tokens
//! (leftmost-longest; non-normalized tokens first, then normalized ones in the
//! gaps; there is no normalizer) -> the pre-tokenizer regex, matched by hand
//! over Unicode 16.0.0 classes (unicode_tables.zig, the version the library's
//! Oniguruma uses) -> byte-level -> BPE (lowest merge rank first, leftmost on
//! ties; `ignore_merges` looks the whole word up first).
const std = @import("std");
const tables = @import("unicode_tables.zig");
const Allocator = std.mem.Allocator;

/// The one pre-tokenizer regex implemented (Llama 3's, used by LFM3), matched by `pieceEnd`.
pub const split_pattern = "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";

pub const replacement = "\u{FFFD}";

pub const Diagnostic = struct {
    buf: [512]u8 = undefined,
    len: usize = 0,

    pub fn message(self: *const Diagnostic) []const u8 {
        return self.buf[0..self.len];
    }
    fn fail(self: *Diagnostic, comptime fmt: []const u8, args: anytype) error{UnsupportedTokenizer} {
        const s = std.fmt.bufPrint(&self.buf, fmt, args) catch self.buf[0..];
        self.len = s.len;
        return error.UnsupportedTokenizer;
    }
};

/// GPT-2 bytes_to_unicode: printable Latin-1 bytes map to themselves, the rest to 256 + n.
const byte_to_cp: [256]u21 = blk: {
    var t: [256]u21 = undefined;
    var n: u21 = 0;
    for (0..256) |b| {
        if ((b >= 0x21 and b <= 0x7E) or (b >= 0xA1 and b <= 0xAC) or (b >= 0xAE and b <= 0xFF)) {
            t[b] = b;
        } else {
            t[b] = 256 + n;
            n += 1;
        }
    }
    break :blk t;
};
const cp_to_byte: [324]i16 = blk: {
    var t: [324]i16 = @splat(-1);
    for (byte_to_cp, 0..) |cp, b| t[cp] = b;
    break :blk t;
};

/// Bytes of a token string through the ByteLevel decoder: every character mapped
/// back to its byte, or (if any character is outside the byte alphabet) the
/// string's own UTF-8, as tokenizers' ByteLevel::decode_chain does.
fn byteLevelDecode(a: Allocator, s: []const u8) ![]const u8 {
    const out = try a.alloc(u8, s.len);
    var n: usize = 0;
    var view = std.unicode.Utf8View.init(s) catch return s;
    var it = view.iterator();
    while (it.nextCodepoint()) |cp| {
        if (cp >= cp_to_byte.len or cp_to_byte[cp] < 0) return s;
        out[n] = @intCast(cp_to_byte[cp]);
        n += 1;
    }
    return out[0..n];
}

// ---- UTF-8 ----

/// Append `bytes` as UTF-8, each maximal invalid subpart replaced by U+FFFD (the
/// Unicode-recommended practice: Python's "replace", Rust's from_utf8_lossy).
/// With `final` false an incomplete, so far valid sequence at the end is left
/// unconsumed. Returns the bytes consumed.
pub fn appendLossy(gpa: Allocator, out: *std.ArrayList(u8), bytes: []const u8, final: bool) !usize {
    var i: usize = 0;
    while (i < bytes.len) {
        const b = bytes[i];
        if (b < 0x80) {
            try out.append(gpa, b);
            i += 1;
            continue;
        }
        const need: usize, const lo: u8, const hi: u8 = switch (b) {
            0xC2...0xDF => .{ 1, 0x80, 0xBF },
            0xE0 => .{ 2, 0xA0, 0xBF },
            0xE1...0xEC, 0xEE, 0xEF => .{ 2, 0x80, 0xBF },
            0xED => .{ 2, 0x80, 0x9F },
            0xF0 => .{ 3, 0x90, 0xBF },
            0xF1...0xF3 => .{ 3, 0x80, 0xBF },
            0xF4 => .{ 3, 0x80, 0x8F },
            else => {
                try out.appendSlice(gpa, replacement);
                i += 1;
                continue;
            },
        };
        var k: usize = 1;
        while (k <= need) : (k += 1) {
            if (i + k >= bytes.len) {
                if (!final) return i;
                break;
            }
            const c = bytes[i + k];
            if (!(if (k == 1) c >= lo and c <= hi else c >= 0x80 and c <= 0xBF)) break;
        }
        if (k > need) {
            try out.appendSlice(gpa, bytes[i..][0 .. need + 1]);
            i += need + 1;
        } else {
            try out.appendSlice(gpa, replacement);
            i += k;
        }
    }
    return i;
}

fn validUtf8(bytes: []const u8) bool {
    var i: usize = 0;
    while (i < bytes.len) {
        const b = bytes[i];
        if (b < 0x80) {
            i += 1;
            continue;
        }
        const need: usize, const lo: u8, const hi: u8 = switch (b) {
            0xC2...0xDF => .{ 1, 0x80, 0xBF },
            0xE0 => .{ 2, 0xA0, 0xBF },
            0xE1...0xEC, 0xEE, 0xEF => .{ 2, 0x80, 0xBF },
            0xED => .{ 2, 0x80, 0x9F },
            0xF0 => .{ 3, 0x90, 0xBF },
            0xF1...0xF3 => .{ 3, 0x80, 0xBF },
            0xF4 => .{ 3, 0x80, 0x8F },
            else => return false,
        };
        if (i + need >= bytes.len) return false;
        if (bytes[i + 1] < lo or bytes[i + 1] > hi) return false;
        for (bytes[i + 2 ..][0 .. need - 1]) |c| if (c < 0x80 or c > 0xBF) return false;
        i += need + 1;
    }
    return true;
}

const Cp = struct { cp: u21, len: u3 };

/// Decode at `i` of known-valid UTF-8.
inline fn cpAt(s: []const u8, i: usize) Cp {
    const b = s[i];
    if (b < 0x80) return .{ .cp = b, .len = 1 };
    if (b < 0xE0) return .{ .cp = (@as(u21, b & 0x1F) << 6) | (s[i + 1] & 0x3F), .len = 2 };
    if (b < 0xF0) return .{ .cp = (@as(u21, b & 0x0F) << 12) | (@as(u21, s[i + 1] & 0x3F) << 6) | (s[i + 2] & 0x3F), .len = 3 };
    return .{ .cp = (@as(u21, b & 0x07) << 18) | (@as(u21, s[i + 1] & 0x3F) << 12) | (@as(u21, s[i + 2] & 0x3F) << 6) | (s[i + 3] & 0x3F), .len = 4 };
}

// ---- the pre-tokenizer regex ----

const Class = tables.Class;

fn classSearch(cp: u21) Class {
    var lo: usize = 0;
    var hi: usize = tables.ranges.len;
    while (lo < hi) {
        const mid = (lo + hi) / 2;
        const r = tables.ranges[mid];
        if (cp < r.lo) {
            hi = mid;
        } else if (cp > r.hi) {
            lo = mid + 1;
        } else return r.class;
    }
    return .other;
}
const ascii_class: [128]Class = blk: {
    @setEvalBranchQuota(100000);
    var t: [128]Class = undefined;
    for (&t, 0..) |*c, i| c.* = classSearch(i);
    break :blk t;
};
pub inline fn classOf(cp: u21) Class {
    return if (cp < 128) ascii_class[cp] else classSearch(cp);
}

fn runOf(s: []const u8, start: usize, class: Class) usize {
    var j = start;
    while (j < s.len) {
        const c = cpAt(s, j);
        if (classOf(c.cp) != class) break;
        j += c.len;
    }
    return j;
}

/// End of the regex match starting at `i` (< s.len) of valid UTF-8. Alternatives
/// in order, each with the backtracking Oniguruma would do; every position has
/// a match, so successive matches tile the input (behaviour Isolated).
pub fn pieceEnd(s: []const u8, i: usize) usize {
    const c0 = cpAt(s, i);
    const k0 = classOf(c0.cp);
    const j1 = i + c0.len;
    // (?i:'s|'t|'re|'ve|'m|'ll|'d): ASCII case pairs, and U+017F (long s) folds to s.
    if (c0.cp == '\'' and j1 < s.len) {
        const c1 = cpAt(s, j1);
        const j2 = j1 + c1.len;
        switch (c1.cp) {
            's', 'S', 0x17F, 't', 'T', 'm', 'M', 'd', 'D' => return j2,
            'r', 'R', 'v', 'V' => if (j2 < s.len and (s[j2] == 'e' or s[j2] == 'E')) return j2 + 1,
            'l', 'L' => if (j2 < s.len and (s[j2] == 'l' or s[j2] == 'L')) return j2 + 1,
            else => {},
        }
    }
    // [^\r\n\p{L}\p{N}]?\p{L}+
    if (k0 == .letter) return runOf(s, j1, .letter);
    if (k0 != .number and c0.cp != '\r' and c0.cp != '\n' and j1 < s.len) {
        const c1 = cpAt(s, j1);
        if (classOf(c1.cp) == .letter) return runOf(s, j1 + c1.len, .letter);
    }
    // \p{N}{1,3}
    if (k0 == .number) {
        var j = j1;
        var n: usize = 1;
        while (n < 3 and j < s.len) : (n += 1) {
            const c = cpAt(s, j);
            if (classOf(c.cp) != .number) break;
            j += c.len;
        }
        return j;
    }
    // ' '?[^\s\p{L}\p{N}]+[\r\n]*
    var other_start: ?usize = null;
    if (k0 == .other) {
        other_start = j1;
    } else if (c0.cp == ' ' and j1 < s.len) {
        const c1 = cpAt(s, j1);
        if (classOf(c1.cp) == .other) other_start = j1 + c1.len;
    }
    if (other_start) |start| {
        var j = runOf(s, start, .other);
        while (j < s.len and (s[j] == '\r' or s[j] == '\n')) j += 1;
        return j;
    }
    // c0 is whitespace. Its run is [i, j); `last` starts the run's last character.
    var j = j1;
    var last = i;
    var newline: ?usize = if (c0.cp == '\r' or c0.cp == '\n') i else null;
    while (j < s.len) {
        const c = cpAt(s, j);
        if (classOf(c.cp) != .space) break;
        if (c.cp == '\r' or c.cp == '\n') newline = j;
        last = j;
        j += c.len;
    }
    // \s*[\r\n]+: \s* backs off to the run's last CR or LF, which ends the match.
    if (newline) |p| return p + 1;
    // \s+(?!\S): the whole run at the end of input, else all but its last character.
    if (j == s.len) return j;
    if (last > i) return last;
    // \s+
    return j;
}

// ---- the tokenizer ----

const Merge = struct { rank: u32, id: u32 };
pub const AddedToken = struct { content: []const u8, id: u32, special: bool };

const Symbol = struct { c: u32, prev: i32, next: i32, len: u32 };
const Pending = struct {
    pos: u32,
    rank: u32,
    id: u32,
    fn before(x: Pending, y: Pending) bool {
        return x.rank < y.rank or (x.rank == y.rank and x.pos < y.pos);
    }
};

/// Reusable buffers for one encode.
const Scratch = struct {
    symbols: std.ArrayList(Symbol) = .empty,
    heap: std.ArrayList(Pending) = .empty,
    fixed: std.ArrayList(u8) = .empty,
    fn deinit(self: *Scratch, gpa: Allocator) void {
        self.symbols.deinit(gpa);
        self.heap.deinit(gpa);
        self.fixed.deinit(gpa);
    }
};

const AddedSet = struct {
    tokens: []AddedToken = &.{},
    first: [256]bool = @splat(false),

    /// Longest token matching at `i`, if any.
    fn longestAt(self: *const AddedSet, s: []const u8, i: usize) ?AddedToken {
        if (!self.first[s[i]]) return null;
        var best: ?AddedToken = null;
        for (self.tokens) |t| {
            if (t.content.len <= s.len - i and std.mem.eql(u8, s[i..][0..t.content.len], t.content)) {
                if (best == null or t.content.len > best.?.content.len) best = t;
            }
        }
        return best;
    }
};

pub const Tokenizer = struct {
    arena: std.heap.ArenaAllocator,
    /// Raw bytes (byte-level characters mapped back) -> id, for model tokens
    /// whose string lies in the byte alphabet (no others can match a word).
    vocab: std.StringHashMapUnmanaged(u32) = .empty,
    merges: std.AutoHashMapUnmanaged(u64, Merge) = .empty,
    byte_ids: [256]?u32 = @splat(null),
    ignore_merges: bool = false,
    non_normalized: AddedSet = .{},
    normalized: AddedSet = .{},
    added: []AddedToken = &.{},
    /// id -> decoded bytes (null: no token has this id); id -> special.
    pieces: []?[]const u8 = &.{},
    special: []bool = &.{},

    pub fn deinit(self: *Tokenizer) void {
        self.arena.deinit();
    }

    /// Number of ids (one past the largest).
    pub fn size(self: *const Tokenizer) u32 {
        return @intCast(self.pieces.len);
    }

    /// Id of an added token by its content (e.g. "<|im_end|>").
    pub fn addedId(self: *const Tokenizer, content: []const u8) ?u32 {
        for (self.added) |t| if (std.mem.eql(u8, t.content, content)) return t.id;
        return null;
    }

    pub fn isSpecial(self: *const Tokenizer, id: u32) bool {
        return id < self.special.len and self.special[id];
    }

    /// Decoded bytes of one id (possibly a partial UTF-8 sequence), null if unknown.
    pub fn piece(self: *const Tokenizer, id: u32) ?[]const u8 {
        return if (id < self.pieces.len) self.pieces[id] else null;
    }

    pub fn init(gpa: Allocator, json_bytes: []const u8, diag: *Diagnostic) !Tokenizer {
        var self: Tokenizer = .{ .arena = std.heap.ArenaAllocator.init(gpa) };
        errdefer self.arena.deinit();
        var tmp = std.heap.ArenaAllocator.init(gpa);
        defer tmp.deinit();
        try self.load(tmp.allocator(), json_bytes, diag);
        return self;
    }

    fn load(self: *Tokenizer, t: Allocator, json_bytes: []const u8, diag: *Diagnostic) !void {
        const a = self.arena.allocator();
        const root = std.json.parseFromSliceLeaky(std.json.Value, t, json_bytes, .{ .allocate = .alloc_always }) catch
            return diag.fail("tokenizer.json is not valid JSON", .{});
        if (root != .object) return diag.fail("tokenizer.json is not a JSON object", .{});
        const o = root.object;
        for ([_][]const u8{ "truncation", "padding", "normalizer" }) |key| {
            if (o.get(key)) |v| if (v != .null) return diag.fail("\"{s}\" is not supported (only null)", .{key});
        }
        try checkPreTokenizer(o.get("pre_tokenizer"), diag);
        if (o.get("post_processor")) |v| if (v != .null) {
            // ByteLevel post-processing only trims offsets; it adds no tokens.
            const ty = typeName(v);
            if (!std.mem.eql(u8, ty, "ByteLevel")) return diag.fail("post_processor \"{s}\" is not supported (only ByteLevel or null)", .{ty});
        };
        const decoder = o.get("decoder") orelse return diag.fail("no decoder (only ByteLevel is supported)", .{});
        if (decoder == .null or !std.mem.eql(u8, typeName(decoder), "ByteLevel"))
            return diag.fail("decoder \"{s}\" is not supported (only ByteLevel)", .{typeName(decoder)});

        const model = o.get("model") orelse return diag.fail("no model", .{});
        if (model != .object) return diag.fail("model is not an object", .{});
        const m = model.object;
        const ty = typeName(model);
        if (!std.mem.eql(u8, ty, "BPE")) return diag.fail("model \"{s}\" is not supported (only BPE)", .{ty});
        for ([_][]const u8{ "dropout", "unk_token", "continuing_subword_prefix", "end_of_word_suffix" }) |key| {
            if (m.get(key)) |v| if (v != .null) return diag.fail("BPE \"{s}\" is not supported (only null)", .{key});
        }
        if (m.get("byte_fallback")) |v| if (v != .bool or v.bool) return diag.fail("BPE byte_fallback is not supported", .{});
        if (m.get("ignore_merges")) |v| {
            if (v != .bool) return diag.fail("BPE ignore_merges is not a boolean", .{});
            self.ignore_merges = v.bool;
        }
        const vocab_v = m.get("vocab") orelse return diag.fail("BPE has no vocab", .{});
        if (vocab_v != .object) return diag.fail("BPE vocab is not an object", .{});
        const vocab = vocab_v.object;
        const model_size: u32 = @intCast(vocab.count());

        // Ids: the model's, then the added tokens' (checked below).
        var max_id: u32 = 0;
        var it = vocab.iterator();
        while (it.next()) |e| {
            const id = asId(e.value_ptr.*) orelse return diag.fail("vocab entry \"{s}\" has no valid id", .{trunc(e.key_ptr.*)});
            max_id = @max(max_id, id);
        }
        const added_v = o.get("added_tokens") orelse std.json.Value{ .array = .init(t) };
        if (added_v != .array) return diag.fail("added_tokens is not an array", .{});
        for (added_v.array.items) |item| {
            if (item != .object) return diag.fail("an added token is not an object", .{});
            const id = asId(item.object.get("id") orelse .null) orelse return diag.fail("an added token has no valid id", .{});
            max_id = @max(max_id, id);
        }
        const n_ids = @as(usize, max_id) + 1;
        self.pieces = try a.alloc(?[]const u8, n_ids);
        @memset(self.pieces, null);
        self.special = try a.alloc(bool, n_ids);
        @memset(self.special, false);

        try self.vocab.ensureTotalCapacity(a, model_size);
        it = vocab.iterator();
        while (it.next()) |e| {
            const id = asId(e.value_ptr.*).?;
            if (self.pieces[id] != null) return diag.fail("vocab id {d} appears twice", .{id});
            const raw = try byteLevelDecode(a, e.key_ptr.*);
            self.pieces[id] = raw;
            if (raw.ptr != e.key_ptr.*.ptr or e.key_ptr.*.len == 0) {
                // In the byte alphabet (or empty): reachable by a word.
                self.vocab.putAssumeCapacity(raw, id);
                if (raw.len == 1 and std.unicode.utf8CountCodepoints(e.key_ptr.*) catch 0 == 1) self.byte_ids[raw[0]] = id;
            } else {
                self.pieces[id] = try a.dupe(u8, raw);
            }
        }

        const merges_v = m.get("merges") orelse return diag.fail("BPE has no merges", .{});
        if (merges_v != .array) return diag.fail("BPE merges is not an array", .{});
        try self.merges.ensureTotalCapacity(a, @intCast(merges_v.array.items.len));
        var joined: std.ArrayList(u8) = .empty;
        for (merges_v.array.items, 0..) |item, rank| {
            var left: []const u8 = undefined;
            var right: []const u8 = undefined;
            switch (item) {
                .array => |pair| {
                    if (pair.items.len != 2 or pair.items[0] != .string or pair.items[1] != .string) return diag.fail("merge {d} is not a pair of strings", .{rank});
                    left = pair.items[0].string;
                    right = pair.items[1].string;
                },
                .string => |s| {
                    var parts = std.mem.splitScalar(u8, s, ' ');
                    left = parts.next().?;
                    right = parts.next() orelse return diag.fail("merge {d} is not \"a b\"", .{rank});
                    if (parts.next() != null) return diag.fail("merge {d} is not \"a b\"", .{rank});
                },
                else => return diag.fail("merge {d} is neither a pair nor a string", .{rank}),
            }
            const l = asId(vocab.get(left) orelse .null) orelse return diag.fail("merge {d}: \"{s}\" is not in the vocab", .{ rank, trunc(left) });
            const r = asId(vocab.get(right) orelse .null) orelse return diag.fail("merge {d}: \"{s}\" is not in the vocab", .{ rank, trunc(right) });
            joined.clearRetainingCapacity();
            try joined.appendSlice(t, left);
            try joined.appendSlice(t, right);
            const id = asId(vocab.get(joined.items) orelse .null) orelse return diag.fail("merge {d}: \"{s}\" is not in the vocab", .{ rank, trunc(joined.items) });
            // A repeated pair takes its later rank, as the library's map does.
            self.merges.putAssumeCapacity(pairKey(l, r), .{ .rank = @intCast(rank), .id = id });
        }

        // Added tokens: ids as the library assigns them (its vocab id if the
        // content is a model token, else one past the largest so far).
        var added: std.ArrayList(AddedToken) = .empty;
        var non_norm: std.ArrayList(AddedToken) = .empty;
        var norm: std.ArrayList(AddedToken) = .empty;
        var specials: std.ArrayList([]const u8) = .empty;
        var largest: ?u32 = null;
        for (added_v.array.items) |item| {
            const ao = item.object;
            const content_v = ao.get("content") orelse .null;
            if (content_v != .string) return diag.fail("an added token has no content", .{});
            const content = try a.dupe(u8, content_v.string);
            for ([_][]const u8{ "single_word", "lstrip", "rstrip" }) |key| {
                if (ao.get(key)) |v| if (v != .bool or v.bool) return diag.fail("added token \"{s}\": {s} is not supported", .{ trunc(content), key });
            }
            const special = if (ao.get("special")) |v| (v == .bool and v.bool) else false;
            const normalized = if (ao.get("normalized")) |v| (v == .bool and v.bool) else !special;
            if (content.len == 0) continue;
            for (added.items) |prev| if (std.mem.eql(u8, prev.content, content)) return diag.fail("added token \"{s}\" appears twice", .{trunc(content)});
            const id = asId(ao.get("id").?).?;
            const expected: u32 = if (vocab.get(content)) |v| asId(v).? else if (largest) |x| (if (x >= model_size) x + 1 else model_size) else model_size;
            if (id != expected) return diag.fail("added token \"{s}\" has id {d}; the tokenizers library assigns {d}", .{ trunc(content), id, expected });
            largest = @max(largest orelse 0, id);
            const tok: AddedToken = .{ .content = content, .id = id, .special = special };
            try added.append(a, tok);
            if (special) try specials.append(t, content);
            try (if (normalized) &norm else &non_norm).append(a, tok);
            self.pieces[id] = try byteLevelDecode(a, content);
        }
        self.added = added.items;
        self.non_normalized = .{ .tokens = non_norm.items };
        self.normalized = .{ .tokens = norm.items };
        for (self.non_normalized.tokens) |tk| self.non_normalized.first[tk.content[0]] = true;
        for (self.normalized.tokens) |tk| self.normalized.first[tk.content[0]] = true;
        // Special is by content: an id is skipped in decoding if its token string is a special token's.
        for (specials.items) |content| {
            for (self.added) |tk| if (std.mem.eql(u8, tk.content, content)) {
                self.special[tk.id] = true;
            };
        }
    }

    fn pairKey(l: u32, r: u32) u64 {
        return (@as(u64, l) << 32) | r;
    }

    fn bpe(self: *const Tokenizer, gpa: Allocator, word: []const u8, sc: *Scratch, out: *std.ArrayList(u32)) !void {
        if (self.ignore_merges) if (self.vocab.get(word)) |id| return out.append(gpa, id);
        const syms = &sc.symbols;
        syms.clearRetainingCapacity();
        for (word) |b| {
            // A byte without a token is dropped (no unk token), as the library does.
            const id = self.byte_ids[b] orelse continue;
            const n: i32 = @intCast(syms.items.len);
            if (n > 0) syms.items[@intCast(n - 1)].next = n;
            try syms.append(gpa, .{ .c = id, .prev = n - 1, .next = -1, .len = 1 });
        }
        const s = syms.items;
        const heap = &sc.heap;
        heap.clearRetainingCapacity();
        if (s.len > 1) for (0..s.len - 1) |i| {
            if (self.merges.get(pairKey(s[i].c, s[i + 1].c))) |m| try heapPush(gpa, heap, .{ .pos = @intCast(i), .rank = m.rank, .id = m.id });
        };
        while (heapPop(heap)) |top| {
            const cur = &s[top.pos];
            if (cur.len == 0 or cur.next == -1) continue;
            const next_pos: usize = @intCast(cur.next);
            const right = s[next_pos];
            const target = self.merges.get(pairKey(cur.c, right.c)) orelse continue;
            if (target.id != top.id) continue;
            cur.c = top.id;
            cur.len += right.len;
            cur.next = right.next;
            s[next_pos].len = 0;
            if (right.next > -1 and right.next < s.len) s[@intCast(right.next)].prev = @intCast(top.pos);
            if (cur.prev >= 0) {
                if (self.merges.get(pairKey(s[@intCast(cur.prev)].c, cur.c))) |m| try heapPush(gpa, heap, .{ .pos = @intCast(cur.prev), .rank = m.rank, .id = m.id });
            }
            if (cur.next >= 0 and cur.next < s.len) {
                if (self.merges.get(pairKey(cur.c, s[@intCast(cur.next)].c))) |m| try heapPush(gpa, heap, .{ .pos = top.pos, .rank = m.rank, .id = m.id });
            }
        }
        for (s) |sym| if (sym.len != 0) try out.append(gpa, sym.c);
    }

    fn pretokenized(self: *const Tokenizer, gpa: Allocator, text: []const u8, sc: *Scratch, out: *std.ArrayList(u32)) !void {
        var i: usize = 0;
        while (i < text.len) {
            const end = pieceEnd(text, i);
            try self.bpe(gpa, text[i..end], sc, out);
            i = end;
        }
    }

    fn splitNormalized(self: *const Tokenizer, gpa: Allocator, text: []const u8, sc: *Scratch, out: *std.ArrayList(u32)) !void {
        var start: usize = 0;
        var i: usize = 0;
        while (i < text.len) {
            if (self.normalized.longestAt(text, i)) |tok| {
                try self.pretokenized(gpa, text[start..i], sc, out);
                try out.append(gpa, tok.id);
                i += tok.content.len;
                start = i;
            } else i += 1;
        }
        try self.pretokenized(gpa, text[start..], sc, out);
    }

    /// Append the ids of `text` (any bytes) to `out`, as tokenizers' encode with
    /// add_special_tokens (this file's post-processor adds none) would.
    pub fn encode(self: *const Tokenizer, gpa: Allocator, text_in: []const u8, out: *std.ArrayList(u32)) !void {
        var sc: Scratch = .{};
        defer sc.deinit(gpa);
        var text = text_in;
        if (!validUtf8(text)) {
            _ = try appendLossy(gpa, &sc.fixed, text, true);
            text = sc.fixed.items;
        }
        var start: usize = 0;
        var i: usize = 0;
        while (i < text.len) {
            if (self.non_normalized.longestAt(text, i)) |tok| {
                try self.splitNormalized(gpa, text[start..i], &sc, out);
                try out.append(gpa, tok.id);
                i += tok.content.len;
                start = i;
            } else i += 1;
        }
        try self.splitNormalized(gpa, text[start..], &sc, out);
    }

    /// tokenizers' decode: unknown ids dropped, special tokens dropped when
    /// `skip_special`, ByteLevel bytes joined, then lossy UTF-8.
    pub fn decode(self: *const Tokenizer, gpa: Allocator, ids: []const u32, skip_special: bool, out: *std.ArrayList(u8)) !void {
        var bytes: std.ArrayList(u8) = .empty;
        defer bytes.deinit(gpa);
        for (ids) |id| {
            if (skip_special and self.isSpecial(id)) continue;
            if (self.piece(id)) |p| try bytes.appendSlice(gpa, p);
        }
        _ = try appendLossy(gpa, out, bytes.items, true);
    }
};

fn heapPush(gpa: Allocator, heap: *std.ArrayList(Pending), item: Pending) !void {
    try heap.append(gpa, item);
    const h = heap.items;
    var i = h.len - 1;
    while (i > 0) {
        const parent = (i - 1) / 2;
        if (!h[i].before(h[parent])) break;
        std.mem.swap(Pending, &h[i], &h[parent]);
        i = parent;
    }
}
fn heapPop(heap: *std.ArrayList(Pending)) ?Pending {
    const h = heap.items;
    if (h.len == 0) return null;
    const top = h[0];
    h[0] = h[h.len - 1];
    heap.items.len -= 1;
    const n = heap.items.len;
    var i: usize = 0;
    while (true) {
        const l = 2 * i + 1;
        if (l >= n) break;
        var best = l;
        if (l + 1 < n and h[l + 1].before(h[l])) best = l + 1;
        if (!h[best].before(h[i])) break;
        std.mem.swap(Pending, &h[i], &h[best]);
        i = best;
    }
    return top;
}

/// Streaming decode: each id's bytes are appended, and only complete UTF-8
/// (or U+FFFD for what can no longer become valid) is released, so the
/// concatenated output equals `decode` of all ids.
pub const Stream = struct {
    pending: std.ArrayList(u8) = .empty,

    pub fn deinit(self: *Stream, gpa: Allocator) void {
        self.pending.deinit(gpa);
    }
    pub fn push(self: *Stream, gpa: Allocator, tok: *const Tokenizer, id: u32, skip_special: bool, out: *std.ArrayList(u8)) !void {
        if (skip_special and tok.isSpecial(id)) return;
        try self.pending.appendSlice(gpa, tok.piece(id) orelse return);
        const used = try appendLossy(gpa, out, self.pending.items, false);
        const rest = self.pending.items.len - used;
        std.mem.copyForwards(u8, self.pending.items[0..rest], self.pending.items[used..]);
        self.pending.items.len = rest;
    }
    pub fn finish(self: *Stream, gpa: Allocator, out: *std.ArrayList(u8)) !void {
        _ = try appendLossy(gpa, out, self.pending.items, true);
        self.pending.clearRetainingCapacity();
    }
};

fn typeName(v: std.json.Value) []const u8 {
    if (v != .object) return "?";
    const t = v.object.get("type") orelse return "?";
    return if (t == .string) t.string else "?";
}
fn asId(v: std.json.Value) ?u32 {
    return switch (v) {
        .integer => |n| std.math.cast(u32, n),
        else => null,
    };
}
fn trunc(s: []const u8) []const u8 {
    return s[0..@min(s.len, 80)];
}

fn checkPreTokenizer(v: ?std.json.Value, diag: *Diagnostic) !void {
    const p = v orelse return diag.fail("no pre_tokenizer (only Sequence[Split, ByteLevel] is supported)", .{});
    const unsupported = "pre_tokenizer: only Sequence[Split(the Llama 3 regex, Isolated), ByteLevel(no prefix space, no regex)] is supported";
    if (!std.mem.eql(u8, typeName(p), "Sequence")) return diag.fail("pre_tokenizer \"{s}\" is not supported; {s}", .{ typeName(p), unsupported });
    const list = p.object.get("pretokenizers") orelse return diag.fail("{s}", .{unsupported});
    if (list != .array or list.array.items.len != 2) return diag.fail("{s}", .{unsupported});
    const split = list.array.items[0];
    const byte_level = list.array.items[1];
    if (!std.mem.eql(u8, typeName(split), "Split")) return diag.fail("pre_tokenizer \"{s}\" is not supported; {s}", .{ typeName(split), unsupported });
    const pattern = split.object.get("pattern") orelse return diag.fail("{s}", .{unsupported});
    const regex = if (pattern == .object) pattern.object.get("Regex") else null;
    if (regex == null or regex.? != .string) return diag.fail("Split pattern must be a Regex; {s}", .{unsupported});
    if (!std.mem.eql(u8, regex.?.string, split_pattern)) return diag.fail("Split regex {s} is not implemented (only the Llama 3 pattern is)", .{trunc(regex.?.string)});
    const behavior = split.object.get("behavior") orelse .null;
    if (behavior != .string or !std.mem.eql(u8, behavior.string, "Isolated")) return diag.fail("Split behavior must be Isolated", .{});
    if (split.object.get("invert")) |inv| if (inv != .bool or inv.bool) return diag.fail("Split invert is not supported", .{});
    if (!std.mem.eql(u8, typeName(byte_level), "ByteLevel")) return diag.fail("pre_tokenizer \"{s}\" is not supported; {s}", .{ typeName(byte_level), unsupported });
    for ([_][]const u8{ "add_prefix_space", "use_regex" }) |key| {
        if (byte_level.object.get(key)) |f| if (f != .bool or f.bool) return diag.fail("ByteLevel pre-tokenizer {s} is not supported", .{key});
    }
}

// ---- tests (CPU only; fixture made by tools/nano_tokenizer_check.py fixture) ----

const fixture_json = @embedFile("testdata/tokenizer-fixture.json");
const fixture_cases = @embedFile("testdata/tokenizer-fixture-cases.json");

test "tokenizer fixture: encode and decode equal the tokenizers library" {
    const gpa = std.testing.allocator;
    var diag: Diagnostic = .{};
    var tok = Tokenizer.init(gpa, fixture_json, &diag) catch |err| {
        std.debug.print("{s}\n", .{diag.message()});
        return err;
    };
    defer tok.deinit();
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const cases = try std.json.parseFromSliceLeaky(std.json.Value, arena.allocator(), fixture_cases, .{});
    var ids: std.ArrayList(u32) = .empty;
    defer ids.deinit(gpa);
    var text: std.ArrayList(u8) = .empty;
    defer text.deinit(gpa);
    var n: usize = 0;
    for (cases.array.items) |case| {
        const input = try std.base64.standard.Decoder.calcSizeForSlice(case.object.get("b64").?.string);
        const raw = try arena.allocator().alloc(u8, input);
        try std.base64.standard.Decoder.decode(raw, case.object.get("b64").?.string);
        ids.clearRetainingCapacity();
        try tok.encode(gpa, raw, &ids);
        const want = case.object.get("ids").?.array.items;
        try std.testing.expectEqual(want.len, ids.items.len);
        for (want, ids.items) |w, got| try std.testing.expectEqual(@as(u32, @intCast(w.integer)), got);
        text.clearRetainingCapacity();
        try tok.decode(gpa, ids.items, true, &text);
        try std.testing.expectEqualStrings(case.object.get("text").?.string, text.items);
        text.clearRetainingCapacity();
        try tok.decode(gpa, ids.items, false, &text);
        try std.testing.expectEqualStrings(case.object.get("text_special").?.string, text.items);
        n += 1;
    }
    try std.testing.expect(n >= 40);
}

test "streaming decode releases only complete UTF-8 and equals one-shot decode" {
    const gpa = std.testing.allocator;
    var out: std.ArrayList(u8) = .empty;
    defer out.deinit(gpa);
    // "é" split across two calls, then a lone continuation byte and a truncated 4-byte lead.
    const bytes = "a\xC3\xA9\x80\xF0\x9F";
    var used = try appendLossy(gpa, &out, bytes[0..2], false);
    try std.testing.expectEqualStrings("a", out.items);
    try std.testing.expectEqual(@as(usize, 1), used);
    used += try appendLossy(gpa, &out, bytes[used..], false);
    try std.testing.expectEqualStrings("a\u{E9}\u{FFFD}", out.items);
    _ = try appendLossy(gpa, &out, bytes[used..], true);
    try std.testing.expectEqualStrings("a\u{E9}\u{FFFD}\u{FFFD}", out.items);
    // A surrogate's encoding is three maximal subparts: ED, A0, 80.
    out.clearRetainingCapacity();
    _ = try appendLossy(gpa, &out, "\xED\xA0\x80", true);
    try std.testing.expectEqualStrings("\u{FFFD}\u{FFFD}\u{FFFD}", out.items);
    try std.testing.expect(!validUtf8("\xED\xA0\x80") and validUtf8("\xF4\x8F\xBF\xBF") and !validUtf8("\xF4\x90\x80\x80"));
}

test "pre-tokenizer regex alternatives" {
    const cases = [_]struct { []const u8, []const []const u8 }{
        .{ "Hello world", &.{ "Hello", " world" } },
        .{ "I'm 12345 ok", &.{ "I", "'m", " ", "123", "45", " ok" } },
        .{ "'SA 'ſx", &.{ "'S", "A", " '", "ſx" } },
        .{ "a  \n\n  b", &.{ "a", "  \n\n", " ", " b" } },
        .{ "x   ", &.{ "x", "   " } },
        .{ "x\t\ty", &.{ "x", "\t", "\ty" } },
        .{ " !!\r\n\r\nz", &.{ " !!\r\n\r\n", "z" } },
        .{ "\u{3000}\u{4E2D}\u{6587}", &.{"\u{3000}\u{4E2D}\u{6587}"} },
    };
    for (cases) |case| {
        var i: usize = 0;
        for (case[1]) |want| {
            const end = pieceEnd(case[0], i);
            try std.testing.expectEqualStrings(want, case[0][i..end]);
            i = end;
        }
        try std.testing.expectEqual(case[0].len, i);
    }
}

test "unsupported tokenizer features are named" {
    var diag: Diagnostic = .{};
    try std.testing.expectError(error.UnsupportedTokenizer, Tokenizer.init(std.testing.allocator, "{\"normalizer\":{\"type\":\"NFC\"}}", &diag));
    try std.testing.expect(std.mem.indexOf(u8, diag.message(), "normalizer") != null);
    try std.testing.expectError(error.UnsupportedTokenizer, Tokenizer.init(std.testing.allocator, "{\"pre_tokenizer\":{\"type\":\"Whitespace\"}}", &diag));
    try std.testing.expect(std.mem.indexOf(u8, diag.message(), "Whitespace") != null);
}
