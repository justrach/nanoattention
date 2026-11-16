//! CPU-only history drafting and exact greedy speculative acceptance.
//! No model, logits, allocation, cache mutation, or GPU dependencies.
const std = @import("std");

pub const max_draft: usize = 32;

/// Copy at most 32 tokens of an earlier occurrence's *observed* continuation.
/// `history` must contain only the actual prompt and committed output, never a
/// reference answer or uncommitted draft. `out` must not overlap `history`.
/// Choose the longest nonempty matching suffix with a historical continuation;
/// ties choose the most recent occurrence. Occurrences may overlap the current
/// suffix, but continuation never extends beyond history (no cyclic invention).
/// Empty history, no match, or empty output deterministically produces no draft.
/// Allocation-free; worst-case quadratic time in history length.
pub fn propose(history: []const u32, out: []u32) usize {
    if (history.len < 2 or out.len == 0) return 0;
    var best_len: usize = 0;
    var best_end: usize = 0;
    var end = history.len - 1; // Exclusive match end; at least one continuation token.
    while (end > best_len) : (end -= 1) {
        var len: usize = 0;
        while (len < end and history[end - 1 - len] == history[history.len - 1 - len]) : (len += 1) {}
        if (len > best_len) {
            best_len = len;
            best_end = end;
        }
    }
    if (best_len == 0) return 0;
    const count = @min(max_draft, out.len, history.len - best_end);
    @memcpy(out[0..count], history[best_end..][0..count]);
    return count;
}

pub const Acceptance = struct {
    /// Commit proposed[0..accepted], including EOS if it is the last accepted token.
    accepted: usize = 0,
    /// Commit after the accepted prefix: correction at first mismatch, or bonus
    /// after full acceptance (also the ordinary next token for an empty draft).
    /// Null when EOS or the output budget already ended the prefix.
    next: ?u32 = null,
    /// True iff the tokens this result says to commit end in EOS.
    eos: bool = false,
};

/// Exact greedy acceptance; no probabilities, tolerance, or approximate matching.
/// `first` is the target greedy prediction BEFORE feeding any draft token.
/// For k proposed inputs d0..d(k-1), `after_each` must have exactly k entries:
/// batch logits row i, after input di, greedily predicts d(i+1), NOT di.
/// Thus d0 is checked against `first`, di against after_each[i-1], and the
/// all-accepted bonus is after_each[k-1]. Use the same argmax/tie policy as the
/// ordinary greedy decoder. Rows after the first rejection are never used.
///
/// `budget` is remaining output-token capacity, including EOS and any next token.
/// EOS is optional; when present it is returned/counted once, with nothing after.
/// An empty draft needs no rows and can emit `first` if budget permits.
///
/// Caller must discard speculative KV entries beyond `accepted`. The `next`
/// correction/bonus has NOT been fed by the batch and must be processed normally
/// before requesting another target prediction. This function does not certify
/// batch-vs-sequential logit parity; that must be tested by the model integration.
pub fn accept(
    proposed: []const u32,
    first: u32,
    after_each: []const u32,
    eos: ?u32,
    budget: usize,
) error{ DraftTooLong, InvalidPredictionCount }!Acceptance {
    if (proposed.len > max_draft) return error.DraftTooLong;
    if (after_each.len != proposed.len) return error.InvalidPredictionCount;
    var result: Acceptance = .{};
    var expected = first;
    while (result.accepted < proposed.len and result.accepted < budget) {
        const token = proposed[result.accepted];
        if (token != expected) {
            result.next = expected;
            result.eos = isEos(expected, eos);
            return result;
        }
        result.accepted += 1;
        if (isEos(token, eos)) {
            result.eos = true;
            return result;
        }
        expected = after_each[result.accepted - 1];
    }
    if (result.accepted < budget) {
        result.next = expected;
        result.eos = isEos(expected, eos);
    }
    return result;
}

fn isEos(token: u32, eos: ?u32) bool {
    return if (eos) |stop| token == stop else false;
}

test "draft overlapping suffix copies only committed continuation" {
    var out: [8]u32 = @splat(99);
    const n = propose(&.{ 1, 2, 1, 2, 1 }, &out);
    try std.testing.expectEqualSlices(u32, &.{ 2, 1 }, out[0..n]);
    // A repeated token is not permission to extrapolate beyond actual history.
    try std.testing.expectEqual(@as(usize, 1), propose(&.{ 7, 7, 7, 7 }, &out));
    try std.testing.expectEqual(@as(u32, 7), out[0]);
}

test "draft empty and no matching suffix leave output untouched" {
    var out: [8]u32 = @splat(99);
    try std.testing.expectEqual(@as(usize, 0), propose(&.{}, &out));
    try std.testing.expectEqual(@as(usize, 0), propose(&.{1}, &out));
    try std.testing.expectEqual(@as(usize, 0), propose(&.{ 1, 2, 3 }, &out));
    try std.testing.expectEqual(@as(usize, 0), propose(&.{ 1, 1 }, out[0..0]));
    for (out) |token| try std.testing.expectEqual(@as(u32, 99), token);
}

test "draft longest suffix wins over newer shorter match and ties choose latest" {
    var out: [8]u32 = undefined;
    const history = [_]u32{ 1, 2, 8, 9, 2, 7, 1, 2 };
    const n = propose(&history, &out);
    try std.testing.expectEqualSlices(u32, history[2..], out[0..n]);
    const tied = [_]u32{ 1, 2, 8, 1, 2, 9, 1, 2 };
    const m = propose(&tied, &out);
    try std.testing.expectEqualSlices(u32, &.{ 9, 1, 2 }, out[0..m]);
    var again: [8]u32 = undefined;
    try std.testing.expectEqual(m, propose(&tied, &again));
    try std.testing.expectEqualSlices(u32, out[0..m], again[0..m]);
}

test "draft bounded by maximum output capacity and observed continuation" {
    try std.testing.expectEqual(@as(usize, 32), max_draft);
    var history: [max_draft + 5]u32 = undefined;
    for (&history, 0..) |*token, i| token.* = @intCast(i);
    history[0] = 1000;
    history[history.len - 1] = 1000;
    var out: [max_draft + 3]u32 = @splat(99);
    try std.testing.expectEqual(max_draft, propose(&history, &out));
    try std.testing.expectEqualSlices(u32, history[1..][0..max_draft], out[0..max_draft]);
    for (out[max_draft..]) |token| try std.testing.expectEqual(@as(u32, 99), token);
    try std.testing.expectEqual(@as(usize, 16), propose(&history, out[0..16]));
    try std.testing.expectEqualSlices(u32, history[1..17], out[0..16]);
    try std.testing.expectEqual(@as(usize, 3), propose(&history, out[0..3]));
    try std.testing.expectEqualSlices(u32, history[1..4], out[0..3]);
    // Even with room for 32, a shorter observed continuation is not extrapolated.
    const short = [_]u32{ 42, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 42 };
    @memset(&out, 99);
    try std.testing.expectEqual(short.len - 1, propose(&short, &out));
    try std.testing.expectEqualSlices(u32, short[1..], out[0 .. short.len - 1]);
    try std.testing.expectEqual(@as(u32, 99), out[short.len - 1]);
}

test "accept first rejection ignores every batch prediction" {
    try std.testing.expectEqual(Acceptance{ .next = 4 }, try accept(&.{ 1, 2, 3 }, 4, &.{ 2, 3, 9 }, null, 9));
}

test "accept middle rejection uses preceding row not current row" {
    try std.testing.expectEqual(Acceptance{ .accepted = 1, .next = 7 }, try accept(&.{ 1, 2, 3 }, 1, &.{ 7, 3, 9 }, null, 9));
    try std.testing.expectEqual(Acceptance{ .accepted = 2, .next = 8 }, try accept(&.{ 1, 2, 3 }, 1, &.{ 2, 8, 9 }, null, 9));
}

test "accept all emits final row bonus and empty draft emits known next" {
    try std.testing.expectEqual(Acceptance{ .accepted = 3, .next = 9 }, try accept(&.{ 1, 2, 3 }, 1, &.{ 2, 3, 9 }, null, 9));
    try std.testing.expectEqual(Acceptance{ .next = 1 }, try accept(&.{}, 1, &.{}, null, 9));
}

test "accept budget forbids both excess draft and correction or bonus" {
    try std.testing.expectEqual(Acceptance{}, try accept(&.{ 1, 2, 3 }, 1, &.{ 2, 3, 9 }, null, 0));
    try std.testing.expectEqual(Acceptance{ .accepted = 1 }, try accept(&.{ 1, 2, 3 }, 1, &.{ 7, 3, 9 }, null, 1));
    try std.testing.expectEqual(Acceptance{ .accepted = 3 }, try accept(&.{ 1, 2, 3 }, 1, &.{ 2, 3, 9 }, null, 3));
    try std.testing.expectEqual(Acceptance{}, try accept(&.{}, 1, &.{}, null, 0));
}

test "accept EOS in prefix correction bonus or empty draft stops immediately" {
    try std.testing.expectEqual(Acceptance{ .accepted = 1, .eos = true }, try accept(&.{ 1, 2, 3 }, 1, &.{ 2, 3, 9 }, 1, 9));
    try std.testing.expectEqual(Acceptance{ .accepted = 2, .eos = true }, try accept(&.{ 1, 2, 3 }, 1, &.{ 2, 3, 9 }, 2, 9));
    try std.testing.expectEqual(Acceptance{ .accepted = 1, .next = 7, .eos = true }, try accept(&.{ 1, 2, 3 }, 1, &.{ 7, 3, 9 }, 7, 9));
    try std.testing.expectEqual(Acceptance{ .accepted = 3, .next = 9, .eos = true }, try accept(&.{ 1, 2, 3 }, 1, &.{ 2, 3, 9 }, 9, 9));
    try std.testing.expectEqual(Acceptance{ .next = 9, .eos = true }, try accept(&.{}, 9, &.{}, 9, 1));
    // A rejected draft EOS is not an emitted EOS.
    try std.testing.expectEqual(Acceptance{ .next = 4 }, try accept(&.{9}, 4, &.{9}, 9, 1));
}

test "accept validates exact row count and draft bound" {
    try std.testing.expectError(error.InvalidPredictionCount, accept(&.{1}, 1, &.{}, null, 2));
    try std.testing.expectError(error.InvalidPredictionCount, accept(&.{1}, 1, &.{ 2, 3 }, null, 2));
    const too_long: [max_draft + 1]u32 = @splat(1);
    try std.testing.expectError(error.DraftTooLong, accept(&too_long, 1, &too_long, null, 20));
}

test "accept 16 and 32 token drafts checks late rejection budget and EOS" {
    var draft: [max_draft]u32 = undefined;
    var rows: [max_draft]u32 = undefined;
    for ([_]usize{ 16, 32 }) |len| {
        for (draft[0..len], rows[0..len], 0..) |*token, *prediction, i| {
            token.* = @intCast(i + 1);
            prediction.* = @intCast(i + 2);
        }
        try std.testing.expectEqual(Acceptance{ .accepted = len, .next = @intCast(len + 1) }, try accept(draft[0..len], 1, rows[0..len], null, len + 1));
        try std.testing.expectEqual(Acceptance{ .accepted = len }, try accept(draft[0..len], 1, rows[0..len], null, len));
        try std.testing.expectEqual(Acceptance{ .accepted = len - 1 }, try accept(draft[0..len], 1, rows[0..len], null, len - 1));
        try std.testing.expectEqual(Acceptance{ .accepted = len, .eos = true }, try accept(draft[0..len], 1, rows[0..len], @intCast(len), len + 1));
        // Reject the final draft token, well beyond the former eight-token limit.
        rows[len - 2] = 999;
        try std.testing.expectEqual(Acceptance{ .accepted = len - 1, .next = 999 }, try accept(draft[0..len], 1, rows[0..len], null, len + 1));
    }
}

test "exhaustive small binary drafts agree with indexed greedy prefix oracle" {
    var draft: [4]u32 = undefined;
    var rows: [4]u32 = undefined;
    for (0..5) |len| {
        const possibilities = @as(usize, 1) << @as(u6, @intCast(len));
        for (0..possibilities) |draft_bits| {
            for (draft[0..len], 0..) |*token, i| token.* = @intCast((draft_bits >> @as(u6, @intCast(i))) & 1);
            for (0..possibilities * 2) |prediction_bits| {
                const first: u32 = @intCast(prediction_bits & 1);
                for (rows[0..len], 0..) |*token, i| token.* = @intCast((prediction_bits >> @as(u6, @intCast(i + 1))) & 1);
                for (0..3) |stop| {
                    const eos: ?u32 = if (stop < 2) @intCast(stop) else null;
                    for (0..len + 3) |budget| {
                        const result = try accept(draft[0..len], first, rows[0..len], eos, budget);
                        var oracle: Acceptance = .{};
                        for (0..@min(len + 1, budget)) |i| {
                            const expected: u32 = @intCast((prediction_bits >> @as(u6, @intCast(i))) & 1);
                            const matched = i < len and draft[i] == expected;
                            if (matched) oracle.accepted += 1 else oracle.next = expected;
                            oracle.eos = isEos(expected, eos);
                            if (!matched or oracle.eos) break;
                        }
                        try std.testing.expectEqual(oracle, result);
                    }
                }
            }
        }
    }
}
