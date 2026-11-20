//! Chat templates as front-end knowledge (frontend.chatTemplate): the text a
//! model expects around system, user and assistant turns. Text only; tools,
//! images and video are rejected by the CLI before a template is used.
const std = @import("std");
const tables = @import("unicode_tables.zig");
const Allocator = std.mem.Allocator;

pub const Role = enum { system, user, assistant };
pub const Message = struct { role: Role, content: []const u8 };

pub const Options = struct {
    /// The template's thinking switch (enable_thinking).
    thinking: bool = false,
    /// Append the generation prompt (add_generation_prompt).
    generate: bool = true,
};

pub const Template = struct {
    /// The whole conversation as text, special tokens written as their strings.
    render: *const fn (Allocator, []const Message, Options) anyerror![]u8,
    /// Text that follows a reply already in the model's state and adds one user
    /// turn and the generation prompt. `closed`: the reply ended with the stop
    /// token (otherwise the text closes it first).
    next_turn: *const fn (Allocator, bool, []const u8, Options) anyerror![]u8,
    /// Markers of a reasoning segment in the model's output (added tokens).
    think_open: []const u8,
    think_close: []const u8,
};

fn pythonSpace(cp: u21) bool {
    for (tables.python_space) |r| if (cp >= r[0] and cp <= r[1]) return true;
    return false;
}

/// Jinja's `trim`: Python str.strip(), which removes str.isspace characters
/// (White_Space plus U+001C..U+001F) from both ends. Invalid UTF-8 is not trimmed past.
pub fn trim(s: []const u8) []const u8 {
    var start: usize = 0;
    while (start < s.len) {
        const n = std.unicode.utf8ByteSequenceLength(s[start]) catch break;
        if (start + n > s.len) break;
        const cp = std.unicode.utf8Decode(s[start..][0..n]) catch break;
        if (!pythonSpace(cp)) break;
        start += n;
    }
    var end = s.len;
    while (end > start) {
        var b = end - 1;
        while (b > start and s[b] & 0xC0 == 0x80 and end - b < 4) b -= 1;
        const n = std.unicode.utf8ByteSequenceLength(s[b]) catch break;
        if (b + n != end) break;
        const cp = std.unicode.utf8Decode(s[b..end]) catch break;
        if (!pythonSpace(cp)) break;
        end = b;
    }
    return s[start..end];
}

test "trim strips what Python's str.strip strips" {
    try std.testing.expectEqualStrings("a b", trim(" \t\n\x1c\u{3000}a b\u{2029}\u{85}\x0b "));
    try std.testing.expectEqualStrings("\u{200B}x\u{FEFF}", trim("\u{200B}x\u{FEFF}"));
    try std.testing.expectEqualStrings("", trim(" \n "));
    try std.testing.expectEqualStrings("é", trim("é"));
    try std.testing.expectEqualStrings("\x85a\xff", trim(" \x85a\xff "));
}
