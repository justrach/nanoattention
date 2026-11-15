//! Synchronous, teacher-forced benchmark. Loading, reset, and warmup are untimed.
//! Link this root module with the same metal.m, libc, and frameworks as main.zig.
const std = @import("std");
const metal = @import("metal.zig");
const Model = @import("model.zig").Model;
const sampling = @import("sampling.zig");

const Timespec = extern struct { sec: i64, nsec: i64 };
extern "c" fn clock_gettime(c_int, *Timespec) c_int;
extern "c" fn write(c_int, [*]const u8, usize) isize;

const Input = struct {
    checkpoint: []const u8,
    prompt: []const u32,
    decode: []const u32,
    repeats: u32 = 3,
    resident_workers: u32 = 0,
    resident_row_tile: u32 = 1,
    staged_row_tile: u32 = 1,
    emit_predictions: bool = false,
};

fn now() !f64 {
    var ts: Timespec = undefined;
    // Darwin CLOCK_MONOTONIC (not wall time or process CPU time).
    if (clock_gettime(6, &ts) != 0) return error.ClockFailed;
    return @as(f64, @floatFromInt(ts.sec)) + @as(f64, @floatFromInt(ts.nsec)) * 1e-9;
}

fn output(bytes: []const u8) !void {
    var pos: usize = 0;
    while (pos < bytes.len) {
        const n = write(1, bytes[pos..].ptr, bytes.len - pos);
        if (n <= 0) return error.OutputFailed;
        pos += @intCast(n);
    }
}

pub fn main(init: std.process.Init) !void {
    var args = std.process.Args.Iterator.init(init.minimal.args);
    _ = args.skip();
    const path = args.next() orelse return error.MissingInput;
    if (std.mem.eql(u8, path, "--help")) {
        try output("Usage: flint-bench INPUT_JSON\nJSON: {\"checkpoint\":\"local-directory\",\"prompt\":[1],\"decode\":[2],\"repeats\":3,\"staged_row_tile\":1}\nStaged row tiles: 1/2/4/8; nondefault conflicts with resident_workers.\n");
        return;
    }
    if (args.next() != null) return error.UnexpectedArgument;
    const mapping = try metal.Mapping.open(init.gpa, path);
    defer mapping.deinit();
    const parsed = try std.json.parseFromSlice(Input, init.gpa, mapping.bytes, .{});
    defer parsed.deinit();
    const input = parsed.value;
    if (input.checkpoint.len == 0) return error.MissingCheckpoint;
    if (input.prompt.len == 0) return error.EmptyPrompt;
    if (input.repeats == 0) return error.InvalidRepeats;
    if (input.staged_row_tile != 1 and input.staged_row_tile != 2 and input.staged_row_tile != 4 and input.staged_row_tile != 8) return error.InvalidStagedRowTile;
    if (input.resident_workers != 0 and input.staged_row_tile != 1) return error.StagedTuningRequiresStaged;
    if (input.resident_workers == 0 and input.resident_row_tile != 1) return error.ResidentTuningRequiresResident;
    const required = try std.math.add(usize, input.prompt.len, input.decode.len);
    const context = std.math.cast(u32, @max(1, required)) orelse return error.ContextExhausted;

    const ctx = try metal.Context.init();
    defer ctx.deinit();
    var model = try Model.init(init.gpa, ctx, input.checkpoint, context);
    defer model.deinit();
    if (input.resident_workers != 0) {
        try model.enableResident(input.resident_workers);
        try model.setResidentRowTile(input.resident_row_tile);
    } else {
        try model.setStagedRowTile(input.staged_row_tile);
    }
    // Validate all inputs before mutating the cache; repetitions never deduplicate
    // input tokens, stop at EOS, or feed the predicted tokens back into the model.
    for (input.prompt) |token| if (token >= model.cfg.vocab_size) return error.InvalidToken;
    for (input.decode) |token| if (token >= model.cfg.vocab_size) return error.InvalidToken;

    for (input.prompt) |token| _ = try model.forward(token);
    for (input.decode[0..@min(8, input.decode.len)]) |token|
        _ = try sampling.greedy(try model.forward(token));

    const predictions: []u32 = if (input.emit_predictions) try init.arena.allocator().alloc(u32, input.decode.len) else &.{};
    for (0..input.repeats) |repeat| {
        try model.reset();
        const gpu_start = ctx.stats().gpu_seconds;
        const start = try now();
        for (input.prompt) |token| _ = try model.forward(token);
        const prefilled = try now();
        const gpu_prefilled = ctx.stats().gpu_seconds;
        var checksum: u64 = 0;
        for (input.decode, 0..) |token, i| {
            const prediction = try sampling.greedy(try model.forward(token));
            checksum += prediction;
            if (input.emit_predictions) predictions[i] = prediction;
        }
        const finished = try now();
        const gpu_finished = ctx.stats().gpu_seconds;
        var buffer: [512]u8 = undefined;
        try output(try std.fmt.bufPrint(
            &buffer,
            "{{\"engine\":\"zig_fp32\",\"repeat\":{d},\"prefill_seconds\":{d:.9},\"decode_seconds\":{d:.9},\"checksum\":{d},\"submissions_per_token\":{d},\"dispatches_per_token\":{d},\"prefill_gpu_seconds\":{d:.9},\"decode_gpu_seconds\":{d:.9},\"resident_row_tile\":{d},\"staged_row_tile\":{d}",
            .{ repeat, prefilled - start, finished - prefilled, checksum, model.last_execution.submissions, model.last_execution.dispatches, gpu_prefilled - gpu_start, gpu_finished - gpu_prefilled, input.resident_row_tile, model.staged_row_tile },
        ));
        if (input.emit_predictions) {
            try output(",\"predictions\":[");
            for (predictions, 0..) |prediction, i| {
                if (i > 0) try output(",");
                try output(try std.fmt.bufPrint(&buffer, "{d}", .{prediction}));
            }
            try output("]");
        }
        try output("}\n");
    }
}
