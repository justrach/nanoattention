//! nano: the model-agnostic megakernel executable. The front end is chosen by
//! the checkpoint's config.json model_type.
const std = @import("std");
const ir = @import("ir.zig");
const exec = @import("exec.zig");
const loader = @import("loader.zig");
const frontend = @import("frontend.zig");
const session = @import("session.zig");
const bench = @import("bench.zig");
const io = @import("io.zig");
const Knobs = @import("knobs.zig").Knobs;
const weights = @import("weights");

const usage =
    \\nano: model-agnostic Zig + Metal megakernel (Apple Silicon)
    \\  nano inspect CHECKPOINT
    \\  nano eval CHECKPOINT --tokens-file F [--feed T] [--positions-file F | --positions all|i,j,..]
    \\        [--logits-out PATH] [--hidden-out PATH] [--greedy N]
    \\  nano generate CHECKPOINT --tokens-file F [--max-tokens N]
    \\  nano session CHECKPOINT [--context N]
    \\  nano mlx-op linear --m M --k K --n N --group G --weights W --x X --out Y  (src/mega/mlxop.zig)
    \\  nano bench CHECKPOINT --prompt-tokens N [--decode-tokens 256] [--repeats 3] [--bench-tokens F]
    \\        [--profile-decode 1]  (per-kernel GPU times of the last repeat's decode, on stderr)
    \\Common: --context N (default 4096), --knobs FILE.json, --knob name=value (repeatable,
    \\        applied after --knobs),
    \\        --sliding-visible N, --embed-round none|bf16.
    \\Token files are a JSON array or {"tokens":[...]}. eval feeds T tokens per
    \\forward (default 1) and writes raw little-endian float32: logits
    \\[positions, vocab] and residual streams [traces, tokens, width].
    \\Every generated token, including EOS, is committed to state in session.
    \\
;

pub const Common = struct {
    context: u32 = 4096,
    knobs: Knobs = .{},
    options: frontend.Options = .{},
};

fn parseTokens(a: std.mem.Allocator, bytes: []const u8, key: []const u8) ![]u32 {
    const value = try std.json.parseFromSliceLeaky(std.json.Value, a, bytes, .{});
    const array = switch (value) {
        .array => |arr| arr,
        .object => |o| switch (o.get(key) orelse return error.InvalidTokenFile) {
            .array => |arr| arr,
            else => return error.InvalidTokenFile,
        },
        else => return error.InvalidTokenFile,
    };
    const out = try a.alloc(u32, array.items.len);
    for (array.items, out) |item, *t| t.* = switch (item) {
        .integer => |n| std.math.cast(u32, n) orelse return error.InvalidTokenFile,
        else => return error.InvalidTokenFile,
    };
    return out;
}

pub fn main(init: std.process.Init) !void {
    const a = init.arena.allocator();
    const gpa = init.gpa;
    var args = std.process.Args.Iterator.init(init.minimal.args);
    _ = args.skip();
    const command = args.next() orelse return io.out(usage);
    if (std.mem.eql(u8, command, "--help") or std.mem.eql(u8, command, "help")) return io.out(usage);
    // decode search: kernel timing on synthetic weights (no checkpoint); run under the inference lock.
    if (std.mem.eql(u8, command, "microbench")) {
        const reps = if (args.next()) |v| try std.fmt.parseInt(u32, v, 10) else 50;
        return @import("microbench.zig").run(a, reps);
    }
    // prefill-mlx: one MLX-arithmetic operation on given inputs (tools/nano_mlx_ops.py).
    if (std.mem.eql(u8, command, "mlx-op")) {
        var rest: std.ArrayList([]const u8) = .empty;
        while (args.next()) |v| try rest.append(a, v);
        return @import("mlxop.zig").run(a, rest.items);
    }
    const directory = args.next() orelse return error.MissingCheckpoint;
    var common: Common = .{};
    var tokens: ?[]u32 = null;
    var positions_spec: ?[]const u8 = null;
    var positions_file: ?[]const u8 = null;
    var logits_out: ?[]const u8 = null;
    var hidden_out: ?[]const u8 = null;
    var feed: u32 = 1;
    var greedy: u32 = 0;
    var max_tokens: u32 = 32;
    var bench_input: bench.Input = .{};
    // --knob flags apply after --knobs FILE whatever their order on the line.
    var knob_flags: std.ArrayList([]const u8) = .empty;
    while (args.next()) |flag| {
        const value = args.next() orelse return error.MissingArgumentValue;
        if (std.mem.eql(u8, flag, "--context")) {
            common.context = try std.fmt.parseInt(u32, value, 10);
        } else if (std.mem.eql(u8, flag, "--knobs")) {
            common.knobs = try Knobs.parseJson(a, try loader.readFile(a, value));
        } else if (std.mem.eql(u8, flag, "--knob")) {
            try knob_flags.append(a, value);
        } else if (std.mem.eql(u8, flag, "--sliding-visible")) {
            common.options.sliding_visible = try std.fmt.parseInt(u32, value, 10);
        } else if (std.mem.eql(u8, flag, "--embed-round")) {
            common.options.embed_round = std.meta.stringToEnum(ir.Round, value) orelse return error.InvalidEmbedRound;
        } else if (std.mem.eql(u8, flag, "--tokens-file")) {
            tokens = try parseTokens(a, try loader.readFile(a, value), "tokens");
        } else if (std.mem.eql(u8, flag, "--tokens")) {
            var list: std.ArrayList(u32) = .empty;
            var iter = std.mem.splitScalar(u8, value, ',');
            while (iter.next()) |item| try list.append(a, try std.fmt.parseInt(u32, item, 10));
            tokens = list.items;
        } else if (std.mem.eql(u8, flag, "--positions")) {
            positions_spec = value;
        } else if (std.mem.eql(u8, flag, "--positions-file")) {
            positions_file = value;
        } else if (std.mem.eql(u8, flag, "--logits-out")) {
            logits_out = value;
        } else if (std.mem.eql(u8, flag, "--hidden-out")) {
            hidden_out = value;
        } else if (std.mem.eql(u8, flag, "--feed")) {
            feed = try std.fmt.parseInt(u32, value, 10);
        } else if (std.mem.eql(u8, flag, "--greedy")) {
            greedy = try std.fmt.parseInt(u32, value, 10);
        } else if (std.mem.eql(u8, flag, "--max-tokens")) {
            max_tokens = try std.fmt.parseInt(u32, value, 10);
        } else if (std.mem.eql(u8, flag, "--prompt-tokens")) {
            bench_input.prompt_tokens = try std.fmt.parseInt(u32, value, 10);
        } else if (std.mem.eql(u8, flag, "--decode-tokens")) {
            bench_input.decode_tokens = try std.fmt.parseInt(u32, value, 10);
        } else if (std.mem.eql(u8, flag, "--bench-tokens")) {
            bench_input.tokens = try loader.readFile(a, value);
        } else if (std.mem.eql(u8, flag, "--profile-decode")) {
            bench_input.profile_decode = std.mem.eql(u8, value, "1");
        } else if (std.mem.eql(u8, flag, "--repeats")) {
            bench_input.repeats = try std.fmt.parseInt(u32, value, 10);
        } else {
            std.debug.print("Unknown flag {s}\n", .{flag});
            return error.UnknownFlag;
        }
    }
    for (knob_flags.items) |value| {
        const eq = std.mem.indexOfScalar(u8, value, '=') orelse return error.InvalidKnob;
        try common.knobs.set(value[0..eq], value[eq + 1 ..]);
    }
    try common.knobs.validate();

    if (std.mem.eql(u8, command, "inspect")) return inspect(gpa, a, directory, common);
    const loaded = try loader.load(gpa, a, directory, common.knobs, common.context, common.options);
    defer {
        loaded.deinit();
        gpa.destroy(loaded);
    }
    std.debug.print("nano: {s}, {d} ops, {d} states, context={d}, batch={d}, {d} mapped bytes\n", .{ loaded.program.meta.model_type, loaded.program.ops.len, loaded.program.states.len, loaded.engine.context, loaded.engine.batch, loaded.store.bytes });
    if (std.mem.eql(u8, command, "session")) return session.run(&loaded.engine, gpa);
    if (std.mem.eql(u8, command, "bench")) return bench.run(&loaded.engine, a, bench_input, common.knobs);
    const prompt = tokens orelse return error.MissingTokens;
    if (prompt.len == 0) return error.EmptyPrompt;
    if (std.mem.eql(u8, command, "generate")) return generate(&loaded.engine, a, prompt, max_tokens);
    if (std.mem.eql(u8, command, "eval")) {
        var positions: []u32 = undefined;
        if (positions_file) |f| {
            positions = try parseTokens(a, try loader.readFile(a, f), "positions");
        } else if (positions_spec == null or std.mem.eql(u8, positions_spec.?, "all")) {
            positions = try a.alloc(u32, prompt.len);
            for (positions, 0..) |*p, i| p.* = @intCast(i);
        } else {
            var list: std.ArrayList(u32) = .empty;
            var iter = std.mem.splitScalar(u8, positions_spec.?, ',');
            while (iter.next()) |item| try list.append(a, try std.fmt.parseInt(u32, item, 10));
            positions = list.items;
        }
        return evaluate(&loaded.engine, a, prompt, positions, feed, logits_out, hidden_out, greedy);
    }
    return error.UnknownCommand;
}

fn inspect(gpa: std.mem.Allocator, a: std.mem.Allocator, directory: []const u8, common: Common) !void {
    const ctx = try @import("gpu.zig").open();
    defer ctx.deinit();
    const config = try loader.readFile(a, try std.fs.path.join(a, &.{ directory, "config.json" }));
    var store = try weights.Store.init(a, ctx, directory, try exec.quantGroup(a, config));
    defer store.deinit();
    var program = try loader.buildProgram(gpa, a, directory, &store, common.options);
    defer program.deinit();
    var names: std.ArrayList([]const u8) = .empty;
    var counts: std.ArrayList(u32) = .empty;
    for (program.ops) |op| {
        for (names.items, 0..) |name, i| {
            if (std.mem.eql(u8, name, op.name())) {
                counts.items[i] += 1;
                break;
            }
        } else {
            try names.append(a, op.name());
            try counts.append(a, 1);
        }
    }
    try io.print(a, "model_type={s} vocab={d} max_context={d} weights={d} states={d} ops={d} head_ops={d} traces={d}\n", .{ program.meta.model_type, program.meta.vocab, program.meta.max_context, program.weights.len, program.states.len, program.ops.len, program.ops.len - program.head_start, program.traces.len });
    for (names.items, counts.items) |name, count| try io.print(a, "  op {s}: {d}\n", .{ name, count });
    var kinds: [5]u32 = @splat(0);
    for (program.states) |s| kinds[@backingInt(s.kind)] += 1;
    try io.print(a, "  states: tokens={d} scratch={d} sequence={d} ring={d} position={d}\n", .{ kinds[0], kinds[1], kinds[2], kinds[3], kinds[4] });
    var bound: usize = 0;
    for (program.weights) |w| {
        _ = try store.get(w.name, w.shape);
        bound += 1;
    }
    try io.print(a, "  bound {d}/{d} weights from {d} mapped bytes\n", .{ bound, program.weights.len, store.bytes });
}

/// Prefill in chunks of the engine batch; returns with the last row's logits.
pub fn prefill(engine: *exec.Engine, tokens: []const u32) !u32 {
    var i: usize = 0;
    var last_row: u32 = 0;
    while (i < tokens.len) {
        const n = @min(engine.batch, tokens.len - i);
        const last = i + n == tokens.len;
        try engine.forward(tokens[i..][0..n], if (last) @intCast(n - 1) else @intCast(n));
        last_row = @intCast(n - 1);
        i += n;
    }
    return last_row;
}

fn isEos(engine: *const exec.Engine, token: u32) bool {
    return std.mem.indexOfScalar(u32, engine.program.meta.eos, token) != null;
}

fn generate(engine: *exec.Engine, a: std.mem.Allocator, prompt: []const u32, max_tokens: u32) !void {
    const required: u64 = @as(u64, prompt.len) + if (max_tokens == 0) 0 else max_tokens - 1;
    if (required > engine.context) return error.ContextExhausted;
    var row = try prefill(engine, prompt);
    var out: std.ArrayList(u32) = .empty;
    for (0..max_tokens) |i| {
        const token = try engine.next(row);
        try out.append(a, token);
        if (isEos(engine, token)) break;
        if (i + 1 < max_tokens) {
            try engine.forward(&.{token}, 0);
            row = 0;
        }
    }
    try io.out(try std.json.Stringify.valueAlloc(a, .{ .tokens = out.items }, .{}));
    try io.out("\n");
}

fn evaluate(engine: *exec.Engine, a: std.mem.Allocator, prompt: []const u32, positions: []const u32, feed_arg: u32, logits_out: ?[]const u8, hidden_out: ?[]const u8, greedy: u32) !void {
    // A feed beyond a context-clamped batch (engine.batch == context >= prompt)
    // means the whole prompt in one forward; it was rejected as InvalidFeed,
    // which made QD admission reject every batch > context (prefill search).
    const feed = if (feed_arg > engine.batch and engine.batch == engine.context) engine.batch else feed_arg;
    if (feed == 0 or feed > engine.batch) return error.InvalidFeed;
    const T: u32 = @intCast(prompt.len);
    for (positions, 0..) |p, i| {
        if (p >= T) return error.InvalidLogitPosition;
        if (i > 0 and p <= positions[i - 1]) return error.PositionsNotIncreasing;
    }
    if (@as(u64, T) + greedy > engine.context) return error.ContextExhausted;
    const vocab = engine.program.meta.vocab;
    const logits = try a.alloc(f32, positions.len * vocab);
    const top1 = try a.alloc(u32, positions.len);
    const traces = engine.program.traces;
    var width: u32 = 0;
    for (traces) |t| width = @max(width, t.view.width);
    const hidden: ?[]f32 = if (hidden_out != null) try a.alloc(f32, traces.len * @as(usize, T) * width) else null;
    const chunk_trace: ?[]f32 = if (hidden_out != null) try a.alloc(f32, traces.len * @as(usize, feed) * width) else null;
    var dispatches: u64 = 0;
    var forwards: u64 = 0;
    var next_pos: usize = 0;
    var i: u32 = 0;
    while (i < T) {
        const n = @min(feed, T - i);
        // Head rows: from the first stored position in this chunk (always the prompt's last).
        var first: u32 = n;
        if (next_pos < positions.len and positions[next_pos] < i + n) first = positions[next_pos] - i;
        if (i + n == T) first = @min(first, n - 1);
        try engine.forwardTraced(prompt[i..][0..n], first, if (chunk_trace) |c| .{ .data = c, .width = width } else null);
        dispatches += engine.last.dispatches;
        forwards += 1;
        if (hidden) |h| for (0..traces.len) |l| for (0..n) |r| {
            @memcpy(h[(l * T + i + r) * width ..][0..width], chunk_trace.?[(l * n + r) * width ..][0..width]);
        };
        while (next_pos < positions.len and positions[next_pos] < i + n) : (next_pos += 1) {
            const row = positions[next_pos] - i;
            @memcpy(logits[next_pos * vocab ..][0..vocab], try engine.logits(row));
            top1[next_pos] = try engine.next(row);
        }
        i += n;
    }
    var generated: std.ArrayList(u32) = .empty;
    var row: u32 = engine.out_end - 1;
    for (0..greedy) |g| {
        const token = try engine.next(row);
        try generated.append(a, token);
        if (isEos(engine, token) or g + 1 == greedy) break;
        try engine.forward(&.{token}, 0);
        row = 0;
    }
    if (logits_out) |path| try io.writeFile(a, path, std.mem.sliceAsBytes(logits));
    if (hidden_out) |path| try io.writeFile(a, path, std.mem.sliceAsBytes(hidden.?));
    const summary = .{
        .tokens = T,
        .feed = feed,
        .positions = positions,
        .top1 = top1,
        .greedy = generated.items,
        .vocab = vocab,
        .traces = traces.len,
        .trace_width = width,
        .forwards = forwards,
        .dispatches = dispatches,
    };
    try io.out(try std.json.Stringify.valueAlloc(a, summary, .{}));
    try io.out("\n");
}

test {
    _ = @import("ir.zig");
    _ = @import("lfm3.zig");
    _ = @import("knobs.zig");
    _ = @import("session.zig");
    _ = @import("kernel_tests.zig");
}
test { _ = @import("plan.zig"); }
