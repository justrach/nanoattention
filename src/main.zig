const std = @import("std");
const metal = @import("metal.zig");
const config = @import("config.zig");
const Model = @import("model.zig").Model;
const sampling = @import("sampling.zig");
extern "c" fn write(c_int, [*]const u8, usize) isize;

fn output(bytes: []const u8) !void {
    var pos: usize = 0;
    while (pos < bytes.len) {
        const n = write(1, bytes[pos..].ptr, bytes.len - pos);
        if (n <= 0) return error.OutputFailed;
        pos += @intCast(n);
    }
}
fn print(comptime fmt: []const u8, args: anytype) !void {
    var buffer: [512]u8 = undefined;
    try output(try std.fmt.bufPrint(&buffer, fmt, args));
}
const usage =
    \\Flint: native Zig + Metal inference baseline (Apple Silicon)
    \\  flint inspect CHECKPOINT
    \\  flint generate CHECKPOINT --tokens 150000,123,456 [--max-tokens 32] [--context 2048]
    \\  flint generate CHECKPOINT --tokens-file prompt.json [--max-tokens 32] [--context 2048]
    \\  flint eval CHECKPOINT --tokens 0,2,3 [--context 2048]
    \\
    \\generate emits a JSON object with greedy-generated token IDs, including EOS.
    \\eval emits one JSON logits array per input token, for numerical verification.
    \\Optional: --resident-workers N (1..32) selects the Lean-plan single-kernel path.
    \\Tuning: --resident-row-tile 1|2|4|8 (requires resident mode).
    \\Tuning: --staged-row-tile 1|2|4|8 (default 1; nondefault requires staged mode).
    \\Use the checkpoint's tokenizer/chat template externally; raw text is not token IDs.
    \\
;

pub fn main(init: std.process.Init) !void {
    const a = init.arena.allocator();
    var args = std.process.Args.Iterator.init(init.minimal.args);
    _ = args.skip();
    const command = args.next() orelse {
        try output(usage);
        return;
    };
    if (std.mem.eql(u8, command, "--help") or std.mem.eql(u8, command, "help")) {
        try output(usage);
        return;
    }
    const inspect = std.mem.eql(u8, command, "inspect");
    const evaluate = std.mem.eql(u8, command, "eval");
    if (!inspect and !evaluate and !std.mem.eql(u8, command, "generate")) return error.UnknownCommand;
    const directory = args.next() orelse return error.MissingCheckpoint;
    if (inspect) {
        if (args.next() != null) return error.UnexpectedArgument;
        const path = try std.fs.path.join(a, &.{ directory, "config.json" });
        const map = try metal.Mapping.open(a, path);
        defer map.deinit();
        const parsed = try config.parse(a, map.bytes);
        defer parsed.deinit();
        const c = parsed.value;
        var sliding: u32 = 0;
        var dense: u32 = 0;
        for (0..c.num_hidden_layers) |i| {
            sliding += @intFromBool(c.sliding(i));
            dense += @intFromBool(c.dense(i));
        }
        try print("Flint: {d} layers ({d} dense, {d} MoE), hidden={d}, vocab={d}\n", .{ c.num_hidden_layers, dense, c.num_hidden_layers - dense, c.hidden_size, c.vocab_size });
        try print("Attention: {d} query / {d} KV heads, dim={d}; {d} sliding (window={d}), {d} full\n", .{ c.num_attention_heads, c.num_key_value_heads, c.head_dim, sliding, c.sliding_window, c.num_hidden_layers - sliding });
        try print("MoE: {d}/{d} routed experts plus shared; context ceiling={d}\n", .{ c.num_experts_per_tok, c.num_experts, c.max_position_embeddings });
        return;
    }
    var context: u32 = 2048;
    var max_tokens: u32 = 32;
    var resident_workers: u32 = 0;
    var resident_row_tile: u32 = 1;
    var staged_row_tile: u32 = 1;
    var tokens: std.ArrayList(u32) = .empty;
    var has_tokens = false;
    while (args.next()) |flag| {
        const value = args.next() orelse return error.MissingArgumentValue;
        if (std.mem.eql(u8, flag, "--context")) {
            context = try std.fmt.parseInt(u32, value, 10);
        } else if (std.mem.eql(u8, flag, "--resident-workers")) {
            resident_workers = try std.fmt.parseInt(u32, value, 10);
            if (resident_workers == 0 or resident_workers > @import("resident.zig").certified.max_workers) return error.InvalidResidentWorkers;
        } else if (std.mem.eql(u8, flag, "--resident-row-tile")) {
            resident_row_tile = try std.fmt.parseInt(u32, value, 10);
            if (!@import("resident.zig").certified.validRowTile(resident_row_tile)) return error.InvalidResidentRowTile;
        } else if (std.mem.eql(u8, flag, "--staged-row-tile")) {
            staged_row_tile = try std.fmt.parseInt(u32, value, 10);
            if (staged_row_tile != 1 and staged_row_tile != 2 and staged_row_tile != 4 and staged_row_tile != 8) return error.InvalidStagedRowTile;
        } else if (std.mem.eql(u8, flag, "--max-tokens")) {
            max_tokens = try std.fmt.parseInt(u32, value, 10);
        } else if (std.mem.eql(u8, flag, "--tokens")) {
            if (has_tokens) return error.DuplicateTokens;
            has_tokens = true;
            var iter = std.mem.splitScalar(u8, value, ',');
            while (iter.next()) |item| try tokens.append(a, try std.fmt.parseInt(u32, item, 10));
        } else if (std.mem.eql(u8, flag, "--tokens-file")) {
            if (has_tokens) return error.DuplicateTokens;
            has_tokens = true;
            const map = try metal.Mapping.open(a, value);
            defer map.deinit();
            const parsed = try std.json.parseFromSlice([]u32, a, map.bytes, .{});
            defer parsed.deinit();
            try tokens.appendSlice(a, parsed.value);
        } else return error.UnknownFlag;
    }
    if (resident_workers != 0 and staged_row_tile != 1) return error.StagedTuningRequiresStaged;
    if (resident_workers == 0 and resident_row_tile != 1) return error.ResidentTuningRequiresResident;
    if (tokens.items.len == 0) return error.EmptyPrompt;
    const required: u64 = @as(u64, tokens.items.len) + if (evaluate or max_tokens == 0) @as(u64, 0) else max_tokens - 1;
    if (required > context) return error.ContextExhausted;
    const ctx = try metal.Context.init();
    defer ctx.deinit();
    var model = try Model.init(init.gpa, ctx, directory, context);
    defer model.deinit();
    if (resident_workers != 0) {
        try model.enableResident(resident_workers);
        try model.setResidentRowTile(resident_row_tile);
    } else {
        try model.setStagedRowTile(staged_row_tile);
    }
    // Validate the complete prompt before mutating the KV cache.
    for (tokens.items) |token| if (token >= model.cfg.vocab_size) return error.InvalidToken;
    std.debug.print("Mapped {d} checkpoint bytes in {d} shard(s); context={d}.\n", .{ model.store.bytes, model.store.shards.items.len, context });
    var logits: []const f32 = &.{};
    for (tokens.items) |token| {
        logits = try model.forward(token);
        if (evaluate) {
            try output("[");
            for (logits, 0..) |v, i| {
                if (!std.math.isFinite(v)) return error.NonFiniteLogits;
                if (i > 0) try output(",");
                try print("{d}", .{v});
            }
            try output("]\n");
        }
    }
    if (evaluate) return;
    try output("{\"tokens\":[");
    for (0..max_tokens) |i| {
        const token = try sampling.greedy(logits);
        if (i > 0) try output(",");
        try print("{d}", .{token});
        if (std.mem.indexOfScalar(u32, model.cfg.eos_token_id, token) != null) break;
        if (i + 1 < max_tokens) logits = try model.forward(token);
    }
    try output("]}\n");
}

test {
    _ = @import("config.zig");
    _ = @import("safetensors.zig");
    _ = @import("sampling.zig");
    _ = @import("metal.zig");
    _ = @import("optimized_tests.zig");
    _ = @import("resident.zig");
    _ = @import("tiling_tests.zig");
    _ = @import("model.zig");
    _ = @import("attention_parallel_tests.zig");
    _ = @import("normalization_tests.zig");
    _ = @import("routing_tests.zig");
    _ = @import("dense_matvec_tests.zig");
}
