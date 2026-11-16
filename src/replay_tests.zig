const std = @import("std");
const metal = @import("metal.zig");

// Synthetic shared buffers only: no model/checkpoint, GUI, or precision lowering.
// The integrating caller imports this file after wiring Context's replay API.
const buffer_count = 4;
const active_capacity = 97;
const word_count = 128;
const guard_word: u32 = 0x7fc12345;
const Words = [buffer_count][word_count]u32;

const Op = struct {
    kernel: [:0]const u8 = "add_scaled",
    first: usize = 0,
    second: usize = 1,
    params: metal.Params,
    threads: usize = 128,
    group: usize = 32,
};

const Fixture = struct {
    ctx: metal.Context,
    buffers: [buffer_count]metal.Buffer,

    fn init(replay: bool) !Fixture {
        const ctx = try metal.Context.initMatched(true);
        errdefer ctx.deinit();
        if (replay) {
            // Pipeline descriptors must opt in before the first kernel lookup.
            try ctx.prepareReplay();
            try ctx.setReplay(true);
        }
        var buffers: [buffer_count]metal.Buffer = undefined;
        var allocated: usize = 0;
        errdefer {
            ctx.finish() catch {};
            ctx.clearReplay() catch {};
            for (buffers[0..allocated]) |buffer| buffer.deinit();
        }
        for (&buffers) |*buffer| {
            buffer.* = try ctx.alloc(word_count);
            allocated += 1;
        }
        return .{ .ctx = ctx, .buffers = buffers };
    }

    fn deinit(self: Fixture) void {
        // This order is also required on assertion/dispatch failure: first drain
        // any pending batch, then release cached ICB resources, then our buffers.
        self.ctx.finish() catch {};
        self.ctx.clearReplay() catch {};
        for (self.buffers) |buffer| buffer.deinit();
        self.ctx.deinit();
    }

    fn reset(self: Fixture, seed: usize) void {
        // Called only while idle. Small dyadic inputs/scales make every add and
        // multiply exact, independent of CPU/GPU multiply-add contraction.
        for (self.buffers, 0..) |buffer, b| {
            @memset(buffer.structs(u32), guard_word);
            for (buffer.floats()[0..active_capacity], 0..) |*value, i| {
                const integer: i32 = @intCast((i * 7 + b * 5 + seed) % 17);
                value.* = @as(f32, @floatFromInt(integer - 8)) / 8;
            }
        }
    }

    fn snapshot(self: Fixture) Words {
        var result: Words = undefined;
        for (self.buffers, 0..) |buffer, b| {
            @memcpy(result[b][0..], buffer.structs(u32));
        }
        return result;
    }

    fn expectWords(self: Fixture, expected: *const Words) !void {
        for (self.buffers, 0..) |buffer, b| {
            // Includes every read-only buffer, dynamically excluded element,
            // and NaN-payload guard word, not just the active output prefix.
            try std.testing.expectEqualSlices(u32, expected.*[b][0..], buffer.structs(u32));
        }
    }

    fn enqueue(self: Fixture, ops: []const Op) !void {
        // Reuse host parameter storage; each command must retain its own copy
        // in a separate aligned parameter slot rather than the last Params.
        var params: metal.Params = .{};
        for (ops) |op| {
            params = op.params;
            try self.ctx.dispatch(op.kernel, &.{ self.buffers[op.first], self.buffers[op.second] }, params, op.threads, op.group);
        }
    }
};

fn reference(words: *Words, ops: []const Op) void {
    for (ops) |op| {
        const count = @min(@as(usize, op.params.n), op.threads);
        if (std.mem.eql(u8, op.kernel, "add_scaled")) {
            // kernels.metal: buffer(0)[i] += p.scale * buffer(1)[i].
            for (0..count) |i| {
                const x: f32 = @bitCast(words.*[op.first][i]);
                const y: f32 = @bitCast(words.*[op.second][i]);
                const value: f32 = x + op.params.scale * y;
                words.*[op.first][i] = @bitCast(value);
            }
        } else {
            std.debug.assert(std.mem.eql(u8, op.kernel, "embedding"));
            std.debug.assert(op.params.dtype == 0);
            std.debug.assert(op.params.offset % @sizeOf(f32) == 0);
            // embedding has the same two-buffer argument count but reverses
            // their read/write roles, making a stale cached pipeline visible.
            const offset: usize = @intCast(op.params.offset / @sizeOf(f32));
            const base = offset + @as(usize, op.params.pos) * @as(usize, op.params.n);
            for (0..count) |i| words.*[op.second][i] = words.*[op.first][base + i];
        }
    }
}

fn expectBatchStats(before: metal.Stats, after: metal.Stats, dispatches: usize) !void {
    try std.testing.expectEqual(@as(u64, @intCast(dispatches)), after.dispatches - before.dispatches);
    try std.testing.expectEqual(@as(u64, 1), after.submissions - before.submissions);
}

fn compareBatch(normal: Fixture, replay: Fixture, ops: []const Op, seed: usize) !void {
    normal.reset(seed);
    replay.reset(seed);
    var expected = normal.snapshot();
    reference(&expected, ops);

    const normal_before = normal.ctx.stats();
    try normal.enqueue(ops);
    try normal.ctx.finish();
    try expectBatchStats(normal_before, normal.ctx.stats(), ops.len);
    try normal.expectWords(&expected);

    const replay_before = replay.ctx.stats();
    try replay.enqueue(ops);
    try replay.ctx.finish();
    try expectBatchStats(replay_before, replay.ctx.stats(), ops.len);
    try replay.expectWords(&expected);
    const normal_words = normal.snapshot();
    try replay.expectWords(&normal_words);
}

test "Metal replay preserves dependent FP32 adds and dynamic per-command parameters bit for bit" {
    const normal = try Fixture.init(false);
    defer normal.deinit();
    const replay = try Fixture.init(true);
    defer replay.deinit();

    const lengths = [_]u32{ 97, 65, 17, 0, 96, 97 };
    const scales = [_]f32{ 0.5, -1, 2, 0, -0.25, 1 };
    for (lengths, scales, 0..) |n, scale, frame| {
        // Unchanged kernels, bindings, and geometry across finishes: only the
        // Params contents change. Later commands consume earlier GPU writes.
        const ops = [_]Op{
            .{ .params = .{ .n = n, .scale = scale } },
            .{ .first = 1, .second = 0, .params = .{ .n = n / 2, .scale = -0.5 } },
            .{ .first = 2, .second = 1, .params = .{ .n = n, .scale = 2 } },
            .{ .first = 0, .second = 2, .params = .{ .n = n / 3, .scale = 0.25 } },
        };
        try compareBatch(normal, replay, &ops, frame);
    }
}

test "Metal replay re-records geometry kernels bindings and changing command counts" {
    const normal = try Fixture.init(false);
    defer normal.deinit();
    const replay = try Fixture.init(true);
    defer replay.deinit();

    var ops = [_]Op{
        .{ .params = .{ .n = 97, .scale = 0.5 } },
        .{ .first = 1, .second = 0, .params = .{ .n = 65, .scale = -1 }, .threads = 96 },
        .{ .first = 2, .second = 1, .params = .{ .n = 33, .scale = 2 }, .threads = 64 },
    };
    for (0..9) |frame| {
        switch (frame) {
            0 => {},
            // n remains 97: changing the grid really changes the written span.
            1 => ops[0].threads = 64,
            2 => ops[0].group = 64,
            3 => ops[0].threads = 128,
            4 => ops[1].first = 3,
            5 => ops[1].second = 2,
            6 => {
                ops[2].kernel = "embedding";
                ops[2].params = .{ .n = 33, .offset = 8, .dtype = 0 };
            },
            7 => ops[0].params.n = 17,
            8 => {},
            else => unreachable,
        }
        // Shrink then grow: stale commands beyond the logical batch must not
        // execute, and growing again must restore all dependent commands.
        const count: usize = if (frame == 7) 1 else ops.len;
        try compareBatch(normal, replay, ops[0..count], frame + 11);
    }
}

test "Metal replay toggles while idle rejects profiling and clears for reuse" {
    const normal = try Fixture.init(false);
    defer normal.deinit();
    const replay = try Fixture.init(true);
    defer replay.deinit();

    // Must reject before opening a profiling session; no profiling capability
    // or timestamp-counter support is needed by this negative test.
    try std.testing.expectError(error.MetalFailure, replay.ctx.beginProfile());
    const ops = [_]Op{
        .{ .params = .{ .n = 65, .scale = -0.5 } },
        .{ .first = 1, .second = 0, .params = .{ .n = 33, .scale = 2 } },
    };
    try compareBatch(normal, replay, &ops, 21);
    try replay.ctx.setReplay(false);
    try compareBatch(normal, replay, &ops, 22);
    try replay.ctx.setReplay(true);
    try compareBatch(normal, replay, &ops, 23);

    try replay.ctx.clearReplay();
    try replay.ctx.clearReplay();
    // Clearing is idempotent and keeps the original pipelines replay-capable;
    // no second prepareReplay (or new Context) is allowed to hide a bad reset.
    try replay.ctx.setReplay(true);
    try compareBatch(normal, replay, &ops, 24);
    try replay.ctx.setReplay(false);
    try compareBatch(normal, replay, &ops, 25);
    try replay.ctx.setReplay(true);
    try compareBatch(normal, replay, &ops, 26);

    const idle_before = replay.ctx.stats();
    try replay.ctx.finish();
    const idle_after = replay.ctx.stats();
    try std.testing.expectEqual(idle_before.dispatches, idle_after.dispatches);
    try std.testing.expectEqual(idle_before.submissions, idle_after.submissions);
}

test "Metal replay aborts the entire batch after a missing kernel and recovers" {
    const normal = try Fixture.init(false);
    defer normal.deinit();
    const replay = try Fixture.init(true);
    defer replay.deinit();
    const ops = [_]Op{
        .{ .params = .{ .n = 97, .scale = 2 } },
        .{ .first = 1, .second = 0, .params = .{ .n = 65, .scale = -0.5 } },
    };
    // First populate the ICB so that abort must also discard cached commands
    // from a previous successful batch rather than just an empty recording.
    try compareBatch(normal, replay, &ops, 31);

    replay.reset(32);
    const untouched = replay.snapshot();
    try replay.enqueue(ops[0..1]);
    const missing_kernel: [:0]const u8 = "__replay_test_missing_kernel__";
    try std.testing.expectError(error.MetalFailure, replay.ctx.dispatch(
        missing_kernel,
        &.{ replay.buffers[1], replay.buffers[0] },
        .{ .n = 65, .scale = -0.5 },
        128,
        32,
    ));
    // The dispatch failure aborts ALL recorded work. finish only drains the
    // declaration-only encoder; even the earlier valid add must never run.
    try replay.ctx.finish();
    try replay.expectWords(&untouched);

    // A later valid batch must not inherit the failure or execute stale slots.
    try compareBatch(normal, replay, &ops, 33);
}
