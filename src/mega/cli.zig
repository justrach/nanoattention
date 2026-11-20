//! nano as a command-line tool (docs/CLI.md): `run`, `chat`, `tokenize`,
//! `detokenize`, `--help`, `--version`. Text in, text out, no Python: the
//! native tokenizer (tokenizer.zig), the front end's chat template
//! (frontend.chatTemplate) and the tuned knobs (presets/*.json, embedded at
//! build time) around the same engine `nano session` drives. Greedy decoding
//! is the engine's GPU argmax; sampling reads the logits back (sample.zig).
const std = @import("std");
const exec = @import("exec.zig");
const loader = @import("loader.zig");
const frontend = @import("frontend.zig");
const chat = @import("chat.zig");
const tokenizer = @import("tokenizer.zig");
const sample = @import("sample.zig");
const io = @import("io.zig");
const Knobs = @import("knobs.zig").Knobs;
const metal = @import("weights").metal;
const Allocator = std.mem.Allocator;

pub const version = "0.1.0";
const exact_preset = @embedFile("presets/exact.json");
const mlx_preset = @embedFile("presets/mlx-prefill.json");

extern "c" fn isatty(c_int) c_int;
extern "c" fn read(c_int, [*]u8, usize) isize;
extern "c" fn access(path: [*:0]const u8, mode: c_int) c_int;

const usage =
    \\nano: language models on Apple Silicon (Zig + Metal). Text in, text out.
    \\  nano run [MODEL] "prompt"       one reply, streamed to standard output
    \\  nano chat [MODEL]               a conversation (interactive, or one turn per input line)
    \\  nano tokenize [MODEL] "text"    token ids as a JSON array
    \\  nano detokenize [MODEL] IDS...  the text of token ids
    \\  nano COMMAND --help, nano --version
    \\MODEL is a checkpoint folder (config.json, tokenizer.json, weights). If omitted:
    \\$NANO_MODEL, then $LFM3_CHECKPOINT. See docs/CLI.md.
    \\
    \\Engine commands (token ids in JSON files):
    \\
;

const run_help =
    \\usage: nano run [MODEL] [PROMPT] [options]
    \\The prompt is PROMPT, --prompt TEXT, --file PATH, or standard input. The reply is
    \\streamed to standard output as it is decoded; nothing else is written there.
    \\  --system TEXT            system message
    \\  --max-tokens N           reply length limit (default 1024)
    \\  --raw                    no chat template: the prompt text is the model's input
    \\  --think, --no-think      the chat template's thinking switch (default off)
    \\  --no-thinking-output     leave out the model's thinking (shown dimmed on a terminal)
    \\  --json                   one JSON object at the end (text, ids, counts, timings)
    \\  --stats                  timings on standard error: load, prompt tokens and tokens a
    \\                           second, time to first token, reply tokens and tokens a second
    \\  --context N              positions the model's state holds (default 4096)
    \\  --prefill exact|mlx      how the prompt is read (below; default exact)
    \\  --knobs FILE.json, --knob name=value   engine knobs instead of / over the preset
    \\Sampling (default greedy: deterministic, the engine's GPU argmax):
    \\  --temperature T  --top-k K  --top-p P  --repetition-penalty R  --seed S
    \\  --sampling model         the checkpoint's generation_config.json; flags after it override
    \\Prefill: `exact` (the embedded presets/exact.json) reads the prompt with arithmetic that is
    \\bit-identical to feeding it one token at a time. `mlx` (presets/mlx-prefill.json) uses MLX's
    \\batched kernels for the prompt: several times faster on long prompts, but not bit-identical
    \\to token-at-a-time, so a long reply can differ. Decoding is the same in both.
    \\Stops at the model's stop token, --max-tokens, or a full context (said on standard error).
    \\
;

const chat_help =
    \\usage: nano chat [MODEL] [options]
    \\Type a message and press Enter. Ctrl-C stops the current reply; Ctrl-D leaves.
    \\  /reset          start over          /system TEXT   set the system message and start over
    \\  /stats          last reply's timings and the state's size
    \\  /help           this help           /exit          leave
    \\A line starting with // sends the text after the first /. When standard input is not a
    \\terminal, each non-empty line is one turn. The model keeps its state between turns and
    \\reads only the new tokens.
    \\Options: --system TEXT, --max-tokens N (per reply, default 1024), --think, --no-think,
    \\--no-thinking-output, --json (one JSON line per turn: fed ids, reply ids, text, timings),
    \\--stats (after every reply), --context N, --prefill exact|mlx, --knobs, --knob, and the
    \\sampling options of `nano run --help`.
    \\
;

const tokenize_help =
    \\usage: nano tokenize [MODEL] [TEXT] [--chat [--system TEXT] [--think]]
    \\Prints the token ids of TEXT (or standard input) as a JSON array, as the Hugging Face
    \\tokenizers library encodes them (added and special tokens in the text included). --chat
    \\wraps TEXT as one user turn in the model's chat template with the generation prompt.
    \\
;

const detokenize_help =
    \\usage: nano detokenize [MODEL] IDS... [--special]
    \\Prints the text of token ids (arguments or standard input; separated by spaces or commas,
    \\or a JSON array). Special tokens are left out unless --special.
    \\
;

const Command = enum { run, chat, tokenize, detokenize };

/// Called by main.zig before its own commands; true when the command was handled here.
pub fn dispatch(init: std.process.Init, args_in: std.process.Args.Iterator, engine_usage: []const u8) !bool {
    var args = args_in;
    const first = args.next() orelse {
        try io.out(usage);
        try io.out(engine_usage);
        return true;
    };
    if (eql(first, "--help") or eql(first, "-h") or eql(first, "help")) {
        try io.out(usage);
        try io.out(engine_usage);
        return true;
    }
    if (eql(first, "--version") or eql(first, "-V") or eql(first, "version")) {
        try io.out("nano " ++ version ++ "\n");
        return true;
    }
    const command = std.meta.stringToEnum(Command, first) orelse return false;
    var rest: std.ArrayList([]const u8) = .empty;
    const a = init.arena.allocator();
    while (args.next()) |arg| try rest.append(a, arg);
    var ctx: Context = .{ .gpa = init.gpa, .a = a, .env = init.environ_map };
    ctx.start = io.now() catch 0;
    execute(&ctx, command, rest.items) catch |err| {
        if (err != error.Reported and err != error.Usage) printErr("nano: {s}\n", .{@errorName(err)});
        std.process.exit(if (err == error.Usage) 2 else 1);
    };
    return true;
}

const Context = struct {
    gpa: Allocator,
    a: Allocator,
    env: *std.process.Environ.Map,
    start: f64 = 0,
};

fn eql(x: []const u8, y: []const u8) bool {
    return std.mem.eql(u8, x, y);
}

fn printErr(comptime fmt: []const u8, args: anytype) void {
    if (@import("builtin").is_test) return;
    var buf: [4096]u8 = undefined;
    const s = std.fmt.bufPrint(&buf, fmt, args) catch buf[0..];
    io.writeAll(2, s) catch {};
}

fn fail(comptime fmt: []const u8, args: anytype) error{Reported} {
    printErr("nano: " ++ fmt ++ "\n", args);
    return error.Reported;
}

fn usageError(comptime fmt: []const u8, args: anytype) error{Usage} {
    printErr("nano: " ++ fmt ++ "\n", args);
    return error.Usage;
}

// ---- options ----

const Prefill = enum { exact, mlx };

const Options = struct {
    positionals: std.ArrayList([]const u8) = .empty,
    model: ?[]const u8 = null,
    system: ?[]const u8 = null,
    prompt: ?[]const u8 = null,
    file: ?[]const u8 = null,
    max_tokens: u32 = 1024,
    raw: bool = false,
    json: bool = false,
    stats: bool = false,
    thinking: bool = false,
    thinking_output: bool = true,
    context: u32 = 4096,
    prefill: Prefill = .exact,
    knobs_file: ?[]const u8 = null,
    knob_flags: std.ArrayList([]const u8) = .empty,
    sampling_model: bool = false,
    temperature: ?f32 = null,
    top_k: ?u32 = null,
    top_p: ?f32 = null,
    repetition_penalty: ?f32 = null,
    seed: ?u64 = null,
    chat: bool = false,
    special: bool = false,
    batch: ?[]const u8 = null,
    help: bool = false,
};

const generation_options = [_][]const u8{
    "--system",     "--max-tokens", "--raw",         "--json",               "--stats",     "--think", "--no-think", "--no-thinking-output",
    "--context",    "--prefill",    "--knobs",       "--knob",               "--sampling",  "--temperature",
    "--top-k",      "--top-p",      "--repetition-penalty", "--seed",
};

fn allowed(command: Command, name: []const u8) bool {
    const list: []const []const u8 = switch (command) {
        .run => &(generation_options ++ [_][]const u8{ "--prompt", "--file" }),
        .chat => &generation_options,
        .tokenize => &.{ "--chat", "--system", "--think", "--no-think", "--batch" },
        .detokenize => &.{"--special"},
    };
    for (list) |item| if (eql(item, name)) return true;
    return false;
}

fn takesValue(name: []const u8) bool {
    for ([_][]const u8{ "--system", "--max-tokens", "--context", "--prefill", "--knobs", "--knob", "--sampling", "--temperature", "--top-k", "--top-p", "--repetition-penalty", "--seed", "--prompt", "--file", "--batch" }) |v|
        if (eql(v, name)) return true;
    return false;
}

fn parseNumber(comptime T: type, name: []const u8, value: []const u8) !T {
    return (switch (@typeInfo(T)) {
        .int => std.fmt.parseInt(T, value, 10),
        .float => std.fmt.parseFloat(T, value),
        else => unreachable,
    }) catch usageError("{s} expects a number, got \"{s}\"", .{ name, value });
}

fn parseOptions(a: Allocator, command: Command, args: []const []const u8) !Options {
    var o: Options = .{};
    var i: usize = 0;
    var only_positional = false;
    while (i < args.len) : (i += 1) {
        const arg = args[i];
        if (only_positional or arg.len < 2 or arg[0] != '-') {
            try o.positionals.append(a, arg);
            continue;
        }
        if (eql(arg, "--")) {
            only_positional = true;
            continue;
        }
        if (eql(arg, "-h") or eql(arg, "--help")) {
            o.help = true;
            continue;
        }
        // Negative numbers are positionals (detokenize never gets them, but be plain about it).
        if (arg[0] == '-' and arg[1] != '-') return usageError("unknown option {s} (see nano {s} --help)", .{ arg, @tagName(command) });
        var name = arg;
        var inline_value: ?[]const u8 = null;
        if (std.mem.indexOfScalar(u8, arg, '=')) |eq| {
            name = arg[0..eq];
            inline_value = arg[eq + 1 ..];
        }
        if (!allowed(command, name)) return usageError("unknown option {s} for {s} (see nano {s} --help)", .{ name, @tagName(command), @tagName(command) });
        var value: []const u8 = "";
        if (takesValue(name)) {
            if (inline_value) |v| {
                value = v;
            } else {
                i += 1;
                if (i >= args.len) return usageError("{s} needs a value", .{name});
                value = args[i];
            }
        } else if (inline_value != null) return usageError("{s} takes no value", .{name});
        if (eql(name, "--system")) {
            o.system = value;
        } else if (eql(name, "--max-tokens")) {
            o.max_tokens = try parseNumber(u32, name, value);
        } else if (eql(name, "--raw")) {
            o.raw = true;
        } else if (eql(name, "--json")) {
            o.json = true;
        } else if (eql(name, "--stats")) {
            o.stats = true;
        } else if (eql(name, "--think")) {
            o.thinking = true;
        } else if (eql(name, "--no-think")) {
            o.thinking = false;
        } else if (eql(name, "--no-thinking-output")) {
            o.thinking_output = false;
        } else if (eql(name, "--context")) {
            o.context = try parseNumber(u32, name, value);
        } else if (eql(name, "--prefill")) {
            o.prefill = std.meta.stringToEnum(Prefill, value) orelse return usageError("--prefill is exact or mlx, not \"{s}\"", .{value});
        } else if (eql(name, "--knobs")) {
            o.knobs_file = value;
        } else if (eql(name, "--knob")) {
            try o.knob_flags.append(a, value);
        } else if (eql(name, "--sampling")) {
            if (!eql(value, "model")) return usageError("--sampling takes `model` (the checkpoint's generation_config.json)", .{});
            o.sampling_model = true;
        } else if (eql(name, "--temperature")) {
            o.temperature = try parseNumber(f32, name, value);
        } else if (eql(name, "--top-k")) {
            o.top_k = try parseNumber(u32, name, value);
        } else if (eql(name, "--top-p")) {
            o.top_p = try parseNumber(f32, name, value);
        } else if (eql(name, "--repetition-penalty")) {
            o.repetition_penalty = try parseNumber(f32, name, value);
        } else if (eql(name, "--seed")) {
            o.seed = try parseNumber(u64, name, value);
        } else if (eql(name, "--prompt")) {
            o.prompt = value;
        } else if (eql(name, "--file")) {
            o.file = value;
        } else if (eql(name, "--chat")) {
            o.chat = true;
        } else if (eql(name, "--special")) {
            o.special = true;
        } else if (eql(name, "--batch")) {
            o.batch = value;
        } else unreachable;
    }
    return o;
}

fn exists(a: Allocator, dir: []const u8, name: []const u8) bool {
    const path = std.fmt.allocPrintSentinel(a, "{s}/{s}", .{ dir, name }, 0) catch return false;
    return access(path, 0) == 0;
}

/// MODEL: the first positional if it is a checkpoint folder, else $NANO_MODEL, then $LFM3_CHECKPOINT.
fn resolveModel(ctx: *Context, o: *Options) ![]const u8 {
    if (o.positionals.items.len > 0) {
        const p = o.positionals.items[0];
        if (exists(ctx.a, p, "config.json") or exists(ctx.a, p, "tokenizer.json")) {
            _ = o.positionals.orderedRemove(0);
            return p;
        }
    }
    for ([_][]const u8{ "NANO_MODEL", "LFM3_CHECKPOINT" }) |name| {
        if (ctx.env.get(name)) |v| if (v.len > 0) {
            if (!exists(ctx.a, v, "config.json") and !exists(ctx.a, v, "tokenizer.json"))
                return fail("${s} is \"{s}\", which is not a checkpoint folder (no config.json or tokenizer.json)", .{ name, v });
            return v;
        };
    }
    return usageError("no model: pass a checkpoint folder (nano run MODEL \"prompt\") or set NANO_MODEL", .{});
}

fn loadTokenizer(ctx: *Context, dir: []const u8) !tokenizer.Tokenizer {
    const path = try std.fs.path.join(ctx.a, &.{ dir, "tokenizer.json" });
    const map = metal.Mapping.open(ctx.a, path) catch return fail("cannot read {s}", .{path});
    defer map.deinit();
    var diag: tokenizer.Diagnostic = .{};
    return tokenizer.Tokenizer.init(ctx.gpa, map.bytes, &diag) catch |err| switch (err) {
        error.UnsupportedTokenizer => fail("{s}: {s}", .{ path, diag.message() }),
        else => err,
    };
}

fn templateFor(ctx: *Context, dir: []const u8) !chat.Template {
    const path = try std.fs.path.join(ctx.a, &.{ dir, "config.json" });
    const config = loader.readFile(ctx.a, path) catch return fail("cannot read {s}", .{path});
    const model_type = exec.modelType(ctx.a, config) catch return fail("{s} has no model_type", .{path});
    return frontend.chatTemplate(model_type) orelse fail("no chat template for model_type \"{s}\"; use --raw", .{model_type});
}

fn readAll(a: Allocator, fd: c_int) ![]u8 {
    var out: std.ArrayList(u8) = .empty;
    var chunk: [65536]u8 = undefined;
    while (true) {
        const n = read(fd, &chunk, chunk.len);
        if (n < 0) {
            if (std.posix.errno(n) == .INTR) continue;
            return error.InputFailed;
        }
        if (n == 0) return out.items;
        try out.appendSlice(a, chunk[0..@intCast(n)]);
    }
}

fn tty(fd: c_int) bool {
    return isatty(fd) == 1;
}

/// Invalid UTF-8 in typed text becomes U+FFFD, as the tokenizer would make it.
fn sanitize(a: Allocator, text: []const u8) ![]const u8 {
    var out: std.ArrayList(u8) = .empty;
    _ = try tokenizer.appendLossy(a, &out, text, true);
    return out.items;
}

fn execute(ctx: *Context, command: Command, args: []const []const u8) !void {
    var o = try parseOptions(ctx.a, command, args);
    if (o.help) return io.out(switch (command) {
        .run => run_help,
        .chat => chat_help,
        .tokenize => tokenize_help,
        .detokenize => detokenize_help,
    });
    const dir = try resolveModel(ctx, &o);
    switch (command) {
        .tokenize => return cmdTokenize(ctx, dir, &o),
        .detokenize => return cmdDetokenize(ctx, dir, &o),
        .run => return cmdRun(ctx, dir, &o),
        .chat => return cmdChat(ctx, dir, &o),
    }
}

// ---- tokenize / detokenize ----

fn writeJsonIds(a: Allocator, ids: []const u32) !void {
    try io.out(try std.json.Stringify.valueAlloc(a, ids, .{}));
    try io.out("\n");
}

fn cmdTokenize(ctx: *Context, dir: []const u8, o: *Options) !void {
    var tok = try loadTokenizer(ctx, dir);
    defer tok.deinit();
    if (o.batch) |path| return batch(ctx, dir, &tok, path);
    if (o.positionals.items.len > 1) return usageError("tokenize takes one TEXT (quote it), got {d} arguments", .{o.positionals.items.len});
    var text: []const u8 = if (o.positionals.items.len == 1) o.positionals.items[0] else if (!tty(0)) try readAll(ctx.a, 0) else return usageError("no text: nano tokenize [MODEL] \"text\" (or text on standard input)", .{});
    if (o.chat) {
        const template = try templateFor(ctx, dir);
        text = try renderTurn(ctx.a, template, o.system, try sanitize(ctx.a, text), o.thinking);
    }
    var ids: std.ArrayList(u32) = .empty;
    try tok.encode(ctx.a, text, &ids);
    try writeJsonIds(ctx.a, ids.items);
}

fn renderTurn(a: Allocator, template: chat.Template, system: ?[]const u8, user: []const u8, thinking: bool) ![]u8 {
    var messages: std.ArrayList(chat.Message) = .empty;
    if (system) |s| try messages.append(a, .{ .role = .system, .content = try sanitize(a, s) });
    try messages.append(a, .{ .role = .user, .content = user });
    return template.render(a, messages.items, .{ .thinking = thinking, .generate = true });
}

fn cmdDetokenize(ctx: *Context, dir: []const u8, o: *Options) !void {
    var tok = try loadTokenizer(ctx, dir);
    defer tok.deinit();
    var source: std.ArrayList(u8) = .empty;
    if (o.positionals.items.len > 0) {
        for (o.positionals.items) |p| {
            try source.appendSlice(ctx.a, p);
            try source.append(ctx.a, ' ');
        }
    } else if (!tty(0)) {
        try source.appendSlice(ctx.a, try readAll(ctx.a, 0));
    } else return usageError("no ids: nano detokenize [MODEL] 1 2 3 (or ids on standard input)", .{});
    var ids: std.ArrayList(u32) = .empty;
    var it = std.mem.tokenizeAny(u8, source.items, " \t\r\n,[]");
    while (it.next()) |item| {
        const id = std.fmt.parseInt(u32, item, 10) catch return usageError("\"{s}\" is not a token id", .{item});
        try ids.append(ctx.a, id);
    }
    var text: std.ArrayList(u8) = .empty;
    try tok.decode(ctx.a, ids.items, !o.special, &text);
    try io.out(text.items);
}

/// `nano tokenize MODEL --batch FILE`, for tools/nano_tokenizer_check.py. Records:
/// kind byte, u32 LE length, payload. 'E' bytes -> {"ids"}; 'D' u32 ids ->
/// {"text","special","stream"}; 'C' JSON {messages, thinking, generate} -> {"text","ids"}
/// (the chat template); 'N' JSON {closed, user, thinking} -> {"text"} (the next turn).
/// One JSON line per record on standard output.
fn batch(ctx: *Context, dir: []const u8, tok: *const tokenizer.Tokenizer, path: []const u8) !void {
    const load_seconds = (io.now() catch 0) - ctx.start;
    const map = metal.Mapping.open(ctx.a, path) catch return fail("cannot read {s}", .{path});
    defer map.deinit();
    const data = map.bytes;
    var template: ?chat.Template = null;
    var out: std.ArrayList(u8) = .empty;
    defer out.deinit(ctx.gpa);
    var ids: std.ArrayList(u32) = .empty;
    defer ids.deinit(ctx.gpa);
    var text: std.ArrayList(u8) = .empty;
    defer text.deinit(ctx.gpa);
    var encode_seconds: f64 = 0;
    var encode_bytes: usize = 0;
    var off: usize = 0;
    while (off < data.len) {
        if (off + 5 > data.len) return fail("batch file truncated", .{});
        const kind = data[off];
        const len = std.mem.readInt(u32, data[off + 1 ..][0..4], .little);
        off += 5;
        const size: usize = if (kind == 'D') @as(usize, len) * 4 else len;
        if (off + size > data.len) return fail("batch file truncated", .{});
        const payload = data[off..][0..size];
        off += size;
        var arena = std.heap.ArenaAllocator.init(ctx.gpa);
        defer arena.deinit();
        const a = arena.allocator();
        var line: []const u8 = undefined;
        switch (kind) {
            'E' => {
                ids.clearRetainingCapacity();
                const t0 = try io.now();
                try tok.encode(ctx.gpa, payload, &ids);
                encode_seconds += try io.now() - t0;
                encode_bytes += payload.len;
                line = try std.json.Stringify.valueAlloc(a, .{ .ids = ids.items }, .{});
            },
            'D' => {
                const list = try a.alloc(u32, len);
                for (list, 0..) |*id, i| id.* = std.mem.readInt(u32, payload[i * 4 ..][0..4], .little);
                var plain: std.ArrayList(u8) = .empty;
                try tok.decode(a, list, true, &plain);
                var special: std.ArrayList(u8) = .empty;
                try tok.decode(a, list, false, &special);
                var streamed: std.ArrayList(u8) = .empty;
                var stream: tokenizer.Stream = .{};
                for (list) |id| try stream.push(a, tok, id, true, &streamed);
                try stream.finish(a, &streamed);
                line = try std.json.Stringify.valueAlloc(a, .{ .text = plain.items, .special = special.items, .stream = streamed.items }, .{});
            },
            'C', 'N' => {
                if (template == null) template = try templateFor(ctx, dir);
                const v = try std.json.parseFromSliceLeaky(std.json.Value, a, payload, .{});
                const thinking = if (v.object.get("thinking")) |t| t == .bool and t.bool else false;
                if (kind == 'N') {
                    const closed = v.object.get("closed").?.bool;
                    const rendered = try template.?.next_turn(a, closed, v.object.get("user").?.string, .{ .thinking = thinking });
                    line = try std.json.Stringify.valueAlloc(a, .{ .text = rendered }, .{});
                } else {
                    var messages: std.ArrayList(chat.Message) = .empty;
                    for (v.object.get("messages").?.array.items) |m| {
                        const role = std.meta.stringToEnum(chat.Role, m.object.get("role").?.string) orelse return fail("batch: unsupported role", .{});
                        try messages.append(a, .{ .role = role, .content = m.object.get("content").?.string });
                    }
                    const generate = if (v.object.get("generate")) |g| g == .bool and g.bool else true;
                    const rendered = try template.?.render(a, messages.items, .{ .thinking = thinking, .generate = generate });
                    var rendered_ids: std.ArrayList(u32) = .empty;
                    try tok.encode(a, rendered, &rendered_ids);
                    line = try std.json.Stringify.valueAlloc(a, .{ .text = rendered, .ids = rendered_ids.items }, .{});
                }
            },
            else => return fail("batch: unknown record kind {d}", .{kind}),
        }
        try out.appendSlice(ctx.gpa, line);
        try out.append(ctx.gpa, '\n');
        if (out.items.len > 1 << 20) {
            try io.out(out.items);
            out.clearRetainingCapacity();
        }
    }
    try io.out(out.items);
    printErr("tokenizer loaded in {d:.3} s; encoded {d} bytes in {d:.3} s ({d:.2} MB/s)\n", .{ load_seconds, encode_bytes, encode_seconds, @as(f64, @floatFromInt(encode_bytes)) / 1e6 / @max(encode_seconds, 1e-9) });
}

// ---- generation ----

const Stop = enum { eos, max_tokens, context, interrupted };

const Turn = struct {
    prompt_tokens: usize = 0,
    reply: std.ArrayList(u32) = .empty,
    stop: Stop = .eos,
    prefill_seconds: f64 = 0,
    first_token_seconds: f64 = 0,
    decode_seconds: f64 = 0,

    fn promptRate(t: Turn) f64 {
        return @as(f64, @floatFromInt(t.prompt_tokens)) / @max(t.prefill_seconds, 1e-9);
    }
    /// Tokens after the first over the time after the first.
    fn replyRate(t: Turn) f64 {
        if (t.reply.items.len < 2) return 0;
        return @as(f64, @floatFromInt(t.reply.items.len - 1)) / @max(t.decode_seconds, 1e-9);
    }
};

/// What the user sees of the reply: special tokens never; thinking (between
/// the template's think markers) dimmed on a terminal, between literal markers
/// otherwise, or not at all; complete UTF-8 only.
const Display = struct {
    out_tty: bool,
    show_thinking: bool,
    quiet: bool,
    think_open: ?u32,
    think_close: ?u32,
    in_think: bool = false,
    announced: bool = false,
    dim: bool = false,
    stream: tokenizer.Stream = .{},
    buf: std.ArrayList(u8) = .empty,
    answer: std.ArrayList(u8) = .empty,
    thinking: std.ArrayList(u8) = .empty,

    fn begin(self: *Display, in_think: bool) void {
        self.in_think = in_think;
        self.announced = false;
        self.answer.clearRetainingCapacity();
        self.thinking.clearRetainingCapacity();
    }

    fn emit(self: *Display, bytes: []const u8) !void {
        if (!self.quiet and bytes.len > 0) try io.out(bytes);
    }

    fn text(self: *Display, gpa: Allocator, bytes: []const u8) !void {
        if (bytes.len == 0) return;
        if (!self.in_think) {
            try self.answer.appendSlice(gpa, bytes);
            return self.emit(bytes);
        }
        try self.thinking.appendSlice(gpa, bytes);
        if (!self.show_thinking) return;
        if (self.out_tty) {
            if (!self.dim) try self.emit("\x1b[2m");
            self.dim = true;
        } else if (!self.announced) {
            try self.emit("<think>");
            self.announced = true;
        }
        try self.emit(bytes);
    }

    fn flush(self: *Display, gpa: Allocator) !void {
        self.buf.clearRetainingCapacity();
        try self.stream.finish(gpa, &self.buf);
        try self.text(gpa, self.buf.items);
    }

    fn closeThink(self: *Display) !void {
        if (self.dim) try self.emit("\x1b[0m");
        self.dim = false;
        if (self.announced and self.show_thinking) try self.emit("</think>");
        self.announced = false;
    }

    fn token(self: *Display, gpa: Allocator, tok: *const tokenizer.Tokenizer, id: u32) !void {
        if (self.think_open != null and id == self.think_open.?) {
            try self.flush(gpa);
            self.in_think = true;
            return;
        }
        if (self.think_close != null and id == self.think_close.?) {
            try self.flush(gpa);
            try self.closeThink();
            self.in_think = false;
            return;
        }
        self.buf.clearRetainingCapacity();
        try self.stream.push(gpa, tok, id, true, &self.buf);
        try self.text(gpa, self.buf.items);
    }

    fn end(self: *Display, gpa: Allocator) !void {
        try self.flush(gpa);
        try self.closeThink();
    }
};

var interrupted = std.atomic.Value(bool).init(false);

fn onInterrupt(_: std.posix.SIG) callconv(.c) void {
    interrupted.store(true, .seq_cst);
}

/// Ctrl-C sets a flag (the reply loop stops at the next token) and interrupts a
/// blocked read (no SA_RESTART), instead of ending the process.
fn catchInterrupt() void {
    var act: std.posix.Sigaction = .{ .handler = .{ .handler = onInterrupt }, .mask = std.posix.sigemptyset(), .flags = 0 };
    std.posix.sigaction(.INT, &act, null);
}

const Generator = struct {
    engine: *exec.Engine,
    tok: *const tokenizer.Tokenizer,
    sampler: ?*sample.Sampler,
    display: *Display,
    eos: []const u32,
    /// The sampler's allocator; replies and display text go to the turn's `a`.
    gpa: Allocator,

    fn isEos(self: *const Generator, id: u32) bool {
        return std.mem.indexOfScalar(u32, self.eos, id) != null;
    }

    /// Feed `prompt` after the state and generate up to `max_tokens`. With
    /// `commit`, every generated token (the stop token included) enters the
    /// state, as `nano session` does, so the next turn continues after it.
    fn turn(self: *Generator, a: Allocator, prompt: []const u32, max_tokens: u32, commit: bool, t: *Turn) !void {
        const engine = self.engine;
        t.prompt_tokens = prompt.len;
        t.reply.clearRetainingCapacity();
        if (self.sampler) |s| s.observe(prompt);
        const start = try io.now();
        var i: usize = 0;
        var row: u32 = 0;
        while (i < prompt.len) {
            const n = @min(engine.batch, prompt.len - i);
            const last = i + n == prompt.len;
            try engine.forward(prompt[i..][0..n], if (last) @intCast(n - 1) else @intCast(n));
            row = @intCast(n - 1);
            i += n;
        }
        t.prefill_seconds = try io.now() - start;
        var first: f64 = 0;
        t.stop = .max_tokens;
        while (t.reply.items.len < max_tokens) {
            if (interrupted.load(.seq_cst)) {
                t.stop = .interrupted;
                break;
            }
            const id = if (self.sampler) |s| try s.sample(self.gpa, try engine.logits(row)) else try engine.next(row);
            const now = try io.now();
            if (t.reply.items.len == 0) {
                first = now;
                t.first_token_seconds = now - start;
            }
            t.decode_seconds = now - first;
            try t.reply.append(a, id);
            if (self.sampler) |s| s.observe(&.{id});
            const stop = self.isEos(id);
            if (!stop) try self.display.token(a, self.tok, id);
            const done = stop or t.reply.items.len == max_tokens;
            if (stop) t.stop = .eos;
            if (done and !commit) break;
            if (engine.position >= engine.context) {
                t.stop = .context;
                break;
            }
            try engine.forward(&.{id}, 0);
            row = 0;
            if (done) break;
        }
        try self.display.end(a);
    }
};

const Engine = struct {
    loaded: *loader.Loaded,
    sampler: ?sample.Sampler,
};

fn knobsFor(ctx: *Context, o: *const Options) !Knobs {
    var knobs = if (o.knobs_file) |path|
        Knobs.parseJson(ctx.a, loader.readFile(ctx.a, path) catch return fail("cannot read {s}", .{path})) catch |err| return fail("{s}: {s}", .{ path, @errorName(err) })
    else
        try Knobs.parseJson(ctx.a, if (o.prefill == .mlx) mlx_preset else exact_preset);
    for (o.knob_flags.items) |value| {
        const eq = std.mem.indexOfScalar(u8, value, '=') orelse return usageError("--knob takes name=value, got \"{s}\"", .{value});
        knobs.set(value[0..eq], value[eq + 1 ..]) catch |err| return fail("--knob {s}: {s}", .{ value, @errorName(err) });
    }
    knobs.validate() catch |err| return fail("knobs: {s}", .{@errorName(err)});
    return knobs;
}

fn samplingFor(ctx: *Context, dir: []const u8, o: *const Options) !sample.Params {
    var p: sample.Params = .{};
    if (o.sampling_model) {
        const path = try std.fs.path.join(ctx.a, &.{ dir, "generation_config.json" });
        const bytes = loader.readFile(ctx.a, path) catch return fail("--sampling model: cannot read {s}", .{path});
        p = sample.Params.fromGenerationConfig(ctx.a, bytes) catch |err| return fail("{s}: {s}", .{ path, @errorName(err) });
    }
    if (o.temperature) |v| p.temperature = v;
    if (o.top_k) |v| p.top_k = v;
    if (o.top_p) |v| p.top_p = v;
    if (o.repetition_penalty) |v| p.repetition_penalty = v;
    if (o.seed) |v| p.seed = v;
    p.validate() catch |err| return usageError("sampling: {s}", .{@errorName(err)});
    return p;
}

fn loadEngine(ctx: *Context, dir: []const u8, o: *const Options) !*loader.Loaded {
    const knobs = try knobsFor(ctx, o);
    if (o.context == 0) return usageError("--context must be positive", .{});
    return loader.load(ctx.gpa, ctx.a, dir, knobs, o.context, .{}) catch |err| fail("cannot load {s}: {s}", .{ dir, @errorName(err) });
}

fn stopName(s: Stop) []const u8 {
    return switch (s) {
        .eos => "stop-token",
        .max_tokens => "max-tokens",
        .context => "context-full",
        .interrupted => "interrupted",
    };
}

fn printStats(load_seconds: ?f64, t: *const Turn) void {
    if (load_seconds) |l| printErr("load {d:.2} s | ", .{l});
    printErr("prompt {d} tokens, {d:.3} s, {d:.1} tok/s | first token {d:.3} s | reply {d} tokens, {d:.1} tok/s | stop: {s}\n", .{
        t.prompt_tokens, t.prefill_seconds, t.promptRate(), t.first_token_seconds, t.reply.items.len, t.replyRate(), stopName(t.stop),
    });
}

const JsonTurn = struct {
    text: []const u8,
    thinking: ?[]const u8,
    stop: []const u8,
    prompt_tokens: usize,
    reply_tokens: usize,
    prompt_ids: ?[]const u32,
    reply_ids: []const u32,
    load_seconds: ?f64,
    prefill_seconds: f64,
    prompt_tokens_per_second: f64,
    time_to_first_token_seconds: f64,
    decode_seconds: f64,
    reply_tokens_per_second: f64,
};

fn jsonTurn(a: Allocator, d: *const Display, t: *const Turn, load_seconds: ?f64, prompt_ids: ?[]const u32) ![]u8 {
    const v: JsonTurn = .{
        .text = d.answer.items,
        .thinking = if (d.thinking.items.len > 0) d.thinking.items else null,
        .stop = stopName(t.stop),
        .prompt_tokens = t.prompt_tokens,
        .reply_tokens = t.reply.items.len,
        .prompt_ids = prompt_ids,
        .reply_ids = t.reply.items,
        .load_seconds = load_seconds,
        .prefill_seconds = t.prefill_seconds,
        .prompt_tokens_per_second = t.promptRate(),
        .time_to_first_token_seconds = t.first_token_seconds,
        .decode_seconds = t.decode_seconds,
        .reply_tokens_per_second = t.replyRate(),
    };
    return std.json.Stringify.valueAlloc(a, v, .{});
}

fn stopNotice(t: *const Turn, context: u32) void {
    if (t.stop == .context) printErr("nano: stopped: the context is full ({d} positions; --context sets it)\n", .{context});
}

fn cmdRun(ctx: *Context, dir: []const u8, o: *Options) !void {
    const sources = @as(u32, @intFromBool(o.positionals.items.len > 0)) + @intFromBool(o.prompt != null) + @intFromBool(o.file != null);
    if (sources > 1) return usageError("give the prompt one way: PROMPT, --prompt or --file", .{});
    if (o.positionals.items.len > 1) return usageError("run takes one PROMPT (quote it), got {d} arguments", .{o.positionals.items.len});
    const prompt_text: []const u8 = if (o.positionals.items.len == 1)
        o.positionals.items[0]
    else if (o.prompt) |p|
        p
    else if (o.file) |path|
        loader.readFile(ctx.a, path) catch return fail("cannot read {s}", .{path})
    else if (!tty(0))
        try readAll(ctx.a, 0)
    else
        return usageError("no prompt: nano run [MODEL] \"prompt\" (or --prompt, --file, standard input)", .{});

    var tok = try loadTokenizer(ctx, dir);
    defer tok.deinit();
    var template: ?chat.Template = null;
    const text = if (o.raw) prompt_text else blk: {
        template = try templateFor(ctx, dir);
        break :blk try renderTurn(ctx.a, template.?, o.system, try sanitize(ctx.a, prompt_text), o.thinking);
    };
    var ids: std.ArrayList(u32) = .empty;
    try tok.encode(ctx.a, text, &ids);
    if (ids.items.len == 0) return usageError("the prompt is empty", .{});
    if (ids.items.len > o.context) return fail("the prompt is {d} tokens; the context holds {d} (--context N)", .{ ids.items.len, o.context });
    const params = try samplingFor(ctx, dir, o);

    const loaded = try loadEngine(ctx, dir, o);
    defer {
        loaded.deinit();
        ctx.gpa.destroy(loaded);
    }
    var sampler: ?sample.Sampler = if (params.greedy()) null else try sample.Sampler.init(ctx.gpa, params, loaded.program.meta.vocab);
    defer if (sampler) |*s| s.deinit(ctx.gpa);
    const load_seconds = try io.now() - ctx.start;

    const out_tty = tty(1);
    var display: Display = .{
        .out_tty = out_tty,
        .show_thinking = o.thinking_output,
        .quiet = o.json,
        .think_open = if (template) |t| tok.addedId(t.think_open) else null,
        .think_close = if (template) |t| tok.addedId(t.think_close) else null,
    };
    display.begin(template != null and o.thinking);
    var gen: Generator = .{ .engine = &loaded.engine, .tok = &tok, .sampler = if (sampler) |*s| s else null, .display = &display, .eos = loaded.program.meta.eos, .gpa = ctx.gpa };
    var turn: Turn = .{};
    try gen.turn(ctx.a, ids.items, o.max_tokens, false, &turn);
    if (o.json) {
        try io.out(try jsonTurn(ctx.a, &display, &turn, load_seconds, ids.items));
        try io.out("\n");
    } else if (out_tty) try io.out("\n");
    stopNotice(&turn, o.context);
    if (o.stats) printStats(load_seconds, &turn);
}

/// Lines from standard input; a read interrupted by Ctrl-C returns error.Interrupted.
const LineReader = struct {
    buf: [65536]u8 = undefined,
    start: usize = 0,
    end: usize = 0,
    eof: bool = false,
    line: std.ArrayList(u8) = .empty,

    fn next(self: *LineReader, a: Allocator) !?[]const u8 {
        self.line.clearRetainingCapacity();
        while (true) {
            if (std.mem.indexOfScalar(u8, self.buf[self.start..self.end], '\n')) |nl| {
                try self.line.appendSlice(a, self.buf[self.start..][0..nl]);
                self.start += nl + 1;
                return self.line.items;
            }
            try self.line.appendSlice(a, self.buf[self.start..self.end]);
            self.start = 0;
            self.end = 0;
            if (self.eof) return if (self.line.items.len > 0) self.line.items else null;
            const n = read(0, &self.buf, self.buf.len);
            if (n < 0) {
                if (std.posix.errno(n) == .INTR) {
                    self.line.clearRetainingCapacity();
                    return error.Interrupted;
                }
                return error.InputFailed;
            }
            if (n == 0) self.eof = true;
            self.end = @intCast(n);
        }
    }
};

fn cmdChat(ctx: *Context, dir: []const u8, o: *Options) !void {
    if (o.positionals.items.len > 0) return usageError("chat takes no prompt argument (got \"{s}\"); type it, or pipe lines in", .{o.positionals.items[0]});
    var tok = try loadTokenizer(ctx, dir);
    defer tok.deinit();
    const template = try templateFor(ctx, dir);
    const params = try samplingFor(ctx, dir, o);
    const loaded = try loadEngine(ctx, dir, o);
    defer {
        loaded.deinit();
        ctx.gpa.destroy(loaded);
    }
    var sampler: ?sample.Sampler = if (params.greedy()) null else try sample.Sampler.init(ctx.gpa, params, loaded.program.meta.vocab);
    defer if (sampler) |*s| s.deinit(ctx.gpa);
    const load_seconds = try io.now() - ctx.start;
    const engine = &loaded.engine;

    const in_tty = tty(0);
    const out_tty = tty(1);
    var display: Display = .{
        .out_tty = out_tty,
        .show_thinking = o.thinking_output,
        .quiet = o.json,
        .think_open = tok.addedId(template.think_open),
        .think_close = tok.addedId(template.think_close),
    };
    var gen: Generator = .{ .engine = engine, .tok = &tok, .sampler = if (sampler) |*s| s else null, .display = &display, .eos = loaded.program.meta.eos, .gpa = ctx.gpa };
    catchInterrupt();
    if (in_tty) printErr("nano chat: {s} (load {d:.2} s). /help for commands, Ctrl-D to leave.\n", .{ dir, load_seconds });

    var system: ?[]const u8 = o.system;
    var started = false;
    var closed = true;
    var full = false;
    var turn: Turn = .{};
    var turns: usize = 0;
    var reader: LineReader = .{};
    while (true) {
        if (in_tty) printErr("> ", .{});
        interrupted.store(false, .seq_cst);
        const raw_line = reader.next(ctx.a) catch |err| switch (err) {
            error.Interrupted => {
                printErr("\n", .{});
                continue;
            },
            else => return err,
        } orelse break;
        const line = std.mem.trimEnd(u8, raw_line, "\r");
        if (std.mem.trim(u8, line, " \t").len == 0) continue;
        var message = line;
        if (line[0] == '/') {
            if (line.len > 1 and line[1] == '/') {
                message = line[1..];
            } else {
                var words = std.mem.tokenizeAny(u8, line, " \t");
                const cmd = words.next().?;
                if (eql(cmd, "/exit") or eql(cmd, "/quit")) break;
                if (eql(cmd, "/help")) {
                    printErr("{s}", .{chat_help});
                } else if (eql(cmd, "/reset")) {
                    engine.reset();
                    if (sampler) |*s| s.reset();
                    started = false;
                    full = false;
                    printErr("(conversation reset)\n", .{});
                } else if (eql(cmd, "/system")) {
                    const rest = std.mem.trim(u8, line["/system".len..], " \t");
                    system = if (rest.len > 0) try ctx.a.dupe(u8, rest) else null;
                    engine.reset();
                    if (sampler) |*s| s.reset();
                    started = false;
                    full = false;
                    printErr("(system message {s}; conversation reset)\n", .{if (system != null) "set" else "cleared"});
                } else if (eql(cmd, "/stats")) {
                    if (turns > 0) printStats(null, &turn) else printErr("no reply yet\n", .{});
                    printErr("state: {d} of {d} positions\n", .{ engine.position, engine.context });
                } else printErr("unknown command {s} (/help)\n", .{cmd});
                continue;
            }
        }
        if (full) {
            printErr("nano: the context is full ({d} positions); /reset to start over\n", .{engine.context});
            continue;
        }
        const user = try sanitize(ctx.a, message);
        const text = if (!started)
            try renderTurn(ctx.a, template, system, user, o.thinking)
        else
            try template.next_turn(ctx.a, closed, user, .{ .thinking = o.thinking });
        var ids: std.ArrayList(u32) = .empty;
        try tok.encode(ctx.a, text, &ids);
        if (@as(u64, engine.position) + ids.items.len >= engine.context) {
            printErr("nano: this turn needs {d} more positions; {d} of {d} are used. /reset to start over\n", .{ ids.items.len, engine.position, engine.context });
            continue;
        }
        if (engine.failed) engine.reset();
        display.begin(o.thinking);
        gen.turn(ctx.a, ids.items, o.max_tokens, true, &turn) catch |err| {
            printErr("\nnano: {s}; the conversation was reset\n", .{@errorName(err)});
            engine.reset();
            started = false;
            continue;
        };
        turns += 1;
        started = true;
        closed = turn.stop == .eos;
        if (turn.stop == .context) full = true;
        if (o.json) {
            try io.out(try jsonTurn(ctx.a, &display, &turn, if (turns == 1) load_seconds else null, ids.items));
            try io.out("\n");
        } else try io.out("\n");
        if (turn.stop == .interrupted) printErr("(stopped)\n", .{});
        stopNotice(&turn, engine.context);
        if (o.stats) printStats(null, &turn);
    }
    if (in_tty) printErr("\n", .{});
}

test "options: model-free parsing, values, unknown options" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const o = try parseOptions(a, .run, &.{ "hello world", "--max-tokens", "8", "--temperature=0.5", "--raw", "--knob", "batch=16" });
    try std.testing.expectEqualStrings("hello world", o.positionals.items[0]);
    try std.testing.expectEqual(@as(u32, 8), o.max_tokens);
    try std.testing.expectEqual(@as(f32, 0.5), o.temperature.?);
    try std.testing.expect(o.raw and o.knob_flags.items.len == 1);
    try std.testing.expectError(error.Usage, parseOptions(a, .run, &.{"--bogus"}));
    try std.testing.expectError(error.Usage, parseOptions(a, .tokenize, &.{"--temperature=1"}));
    try std.testing.expectError(error.Usage, parseOptions(a, .run, &.{"--max-tokens"}));
    const d = try parseOptions(a, .detokenize, &.{ "--", "1", "2" });
    try std.testing.expectEqual(@as(usize, 2), d.positionals.items.len);
}

test "the embedded presets are valid knob sets" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    _ = try Knobs.parseJson(arena.allocator(), exact_preset);
    _ = try Knobs.parseJson(arena.allocator(), mlx_preset);
}
