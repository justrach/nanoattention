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
    \\  flint session CHECKPOINT --context 32768 --mlx-attention 1 --staged-row-tile 2
    \\session accepts JSONL on stdin with new tokens/max_tokens and retains KV until EOF.
    \\Every generated token, including EOS, is committed to KV before its response.
    \\
    \\generate emits a JSON object with greedy-generated token IDs, including EOS.
    \\eval emits one JSON logits array per input token, for numerical verification.
    \\Optional eval --logits-from N emits only positions >= N; all inputs still execute.
    \\Staged attention: --attention-chunk 0|128|256|512|1024 (default 0; split attention is experimental).
    \\Experimental: --mlx-attention 1 uses MLX-layout attention and matched FP32 reductions.
    \\It requires staged D128 attention and conflicts with --attention-chunk.
    \\Opt-in: --kv-bf16 1 stores global KV as BF16 (sliding KV stays FP32).
    \\It requires --mlx-attention 1 and may change logits/tokens; default is FP32 KV.
\\Opt-in: --speculative-tokens 1..32 enables greedy history drafts and batched target verification.
\\It requires matched staged INT4/g64 projections; 0 disables speculation (default).
\\Opt-in: --twodsd 0|1 requests 2DSD (token+depth speculation). Flint is not looped, so 1 fails closed.
\\Optional: --twodsd-k 1..16, --twodsd-stage stage1|stage2a|stage2b, --twodsd-d1 N, --twodsd-d2 N,
\\--twodsd-kv full|shared_last, --twodsd-window N. Tuning without --twodsd 1 is rejected.
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
    const session_mode = std.mem.eql(u8, command, "session");
    if (!inspect and !evaluate and !session_mode and !std.mem.eql(u8, command, "generate")) return error.UnknownCommand;
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
    var logits_from: ?u32 = null;
    var attention_chunk: ?u32 = null;
    var mlx_attention = false;
    var kv_bf16 = false;
    var speculative_tokens: u32 = 0;
    var speculative_query_group: u32 = 2;
    var speculative_expert_tile: u32 = 0;
    var speculative_weight_reuse = false;
    var twodsd_cli: @import("twodsd.zig").Cli = .{};
    var trace_position: ?u32 = null;
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
        } else if (std.mem.eql(u8, flag, "--attention-chunk")) {
            attention_chunk = try std.fmt.parseInt(u32, value, 10);
            if (attention_chunk.? != 0 and !@import("resident.zig").certified.validAttentionChunk(attention_chunk.?)) return error.InvalidAttentionChunk;
        } else if (std.mem.eql(u8, flag, "--mlx-attention")) {
            const enabled = try std.fmt.parseInt(u1, value, 10);
            mlx_attention = enabled != 0;
        } else if (std.mem.eql(u8, flag, "--kv-bf16")) {
            kv_bf16 = (try std.fmt.parseInt(u1, value, 10)) != 0;
        } else if (std.mem.eql(u8, flag, "--speculative-tokens")) {
            speculative_tokens = try std.fmt.parseInt(u32, value, 10);
            if (speculative_tokens > Model.max_speculative_tokens) return error.InvalidSpeculativeLength;
        } else if (std.mem.eql(u8, flag, "--speculative-query-group")) {
            speculative_query_group = try std.fmt.parseInt(u32, value, 10);
            if (speculative_query_group != 2 and speculative_query_group != 4 and speculative_query_group != 8) return error.InvalidBatchQueryGroup;
        } else if (std.mem.eql(u8, flag, "--speculative-expert-tile")) {
            speculative_expert_tile = try std.fmt.parseInt(u32, value, 10);
            if (speculative_expert_tile != 0 and !@import("resident.zig").certified.validRowTile(speculative_expert_tile)) return error.InvalidStagedRowTile;
        } else if (std.mem.eql(u8, flag, "--speculative-weight-reuse")) {
            speculative_weight_reuse = (try std.fmt.parseInt(u1, value, 10)) != 0;
        } else if (std.mem.eql(u8, flag, "--twodsd")) {
            twodsd_cli.enabled = @import("twodsd.zig").parseEnable(value) orelse return error.InvalidTwoDsdFlag;
        } else if (std.mem.eql(u8, flag, "--twodsd-k")) {
            twodsd_cli.k = try std.fmt.parseInt(u32, value, 10);
        } else if (std.mem.eql(u8, flag, "--twodsd-stage")) {
            twodsd_cli.stage = @import("twodsd.zig").parseStage(value) orelse return error.InvalidTwoDsdStage;
        } else if (std.mem.eql(u8, flag, "--twodsd-d1")) {
            twodsd_cli.d1 = try std.fmt.parseInt(u32, value, 10);
        } else if (std.mem.eql(u8, flag, "--twodsd-d2")) {
            twodsd_cli.d2 = try std.fmt.parseInt(u32, value, 10);
        } else if (std.mem.eql(u8, flag, "--twodsd-kv")) {
            twodsd_cli.kv_policy = @import("twodsd.zig").parseKvPolicy(value) orelse return error.InvalidTwoDsdKvPolicy;
        } else if (std.mem.eql(u8, flag, "--twodsd-window")) {
            twodsd_cli.kv_window = try std.fmt.parseInt(u32, value, 10);
        } else if (std.mem.eql(u8, flag, "--trace-position")) {
            trace_position = try std.fmt.parseInt(u32, value, 10);
        } else if (std.mem.eql(u8, flag, "--logits-from")) {
            logits_from = try std.fmt.parseInt(u32, value, 10);
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
    if (resident_workers != 0 and (staged_row_tile != 1 or attention_chunk != null)) return error.StagedTuningRequiresStaged;
    if (resident_workers == 0 and resident_row_tile != 1) return error.ResidentTuningRequiresResident;
    if (session_mode and (has_tokens or logits_from != null or trace_position != null)) return error.SessionRequiresJsonlInput;
    if (!session_mode and tokens.items.len == 0) return error.EmptyPrompt;
    if (mlx_attention and (resident_workers != 0 or attention_chunk != null)) return error.ConflictingAttentionMode;
    if (kv_bf16 and !mlx_attention) return error.Bf16KvRequiresMatchedStaged;
    if (speculative_tokens != 0 and (!mlx_attention or evaluate)) return error.SpeculationRequiresMatchedGeneration;
    if (evaluate and twodsd_cli.enabled) return error.TwoDsdRequiresGeneration;
    if (twodsd_cli.enabled or twodsd_cli.anyTuning()) {
        const path = try std.fs.path.join(a, &.{ directory, "config.json" });
        const map = try metal.Mapping.open(a, path);
        defer map.deinit();
        const parsed = try config.parse(a, map.bytes);
        defer parsed.deinit();
        try twodsd_cli.apply(parsed.value.model_type);
    }
    if (trace_position != null and (!evaluate or resident_workers != 0)) return error.TraceRequiresStagedEval;
    if (trace_position != null and trace_position.? >= tokens.items.len) return error.InvalidTracePosition;
    if (logits_from != null and !evaluate) return error.LogitsFromRequiresEval;
    if (evaluate and (logits_from orelse 0) >= tokens.items.len) return error.InvalidLogitRange;
    const required: u64 = if (session_mode) 0 else @as(u64, tokens.items.len) + if (evaluate or max_tokens == 0) @as(u64, 0) else if (speculative_tokens != 0) max_tokens else max_tokens - 1;
    if (required > context) return error.ContextExhausted;
    const ctx = try metal.Context.initMatched(mlx_attention);
    defer ctx.deinit();
    var model = try Model.initWithKv(init.gpa, ctx, directory, context, kv_bf16);
    defer model.deinit();
    if (resident_workers != 0) {
        try model.enableResident(resident_workers);
        try model.setResidentRowTile(resident_row_tile);
    } else {
        try model.setStagedRowTile(staged_row_tile);
        if (attention_chunk) |chunk| try model.setAttentionChunk(chunk);
    }
    if (mlx_attention) try model.enableMlxAttention();
    model.trace_position = trace_position;
    model.speculative_tokens = speculative_tokens;
    model.batch_query_group = speculative_query_group;
    if (speculative_tokens == 0 and (speculative_query_group != 2 or speculative_expert_tile != 0 or speculative_weight_reuse)) return error.SpeculativeTuningRequiresSpeculation;
    model.gate_row_tile = speculative_expert_tile;
    model.down_row_tile = speculative_expert_tile;
    model.batch_weight_reuse = speculative_weight_reuse;
    if (speculative_tokens != 0) try model.prepareBatch();
    // Validate the complete prompt before mutating the KV cache.
    for (tokens.items) |token| if (token >= model.cfg.vocab_size) return error.InvalidToken;
    std.debug.print("Mapped {d} checkpoint bytes in {d} shard(s); context={d}.\n", .{ model.store.bytes, model.store.shards.items.len, context });
    if (session_mode) return @import("session.zig").run(&model, init.gpa);
    var logits: []const f32 = &.{};
    if (!evaluate) logits = try model.prefill(tokens.items);
    if (evaluate) for (tokens.items, 0..) |token, position| {
        logits = try model.forward(token);
        if (evaluate and position >= (logits_from orelse 0)) {
            try output("[");
            for (logits, 0..) |v, i| {
                if (!std.math.isFinite(v)) return error.NonFiniteLogits;
                if (i > 0) try output(",");
                try print("{d}", .{v});
            }
            try output("]\n");
        }
    };
    if (evaluate) return;
    if (speculative_tokens != 0) {
        const generated = try a.alloc(u32, max_tokens);
        const result = try @import("speculative_generation.zig").generate(&model, a, &tokens, generated, logits);
        const json = try std.json.Stringify.valueAlloc(a, .{ .tokens = generated[0..result.count], .speculative_batches = result.batches, .speculative_proposed = result.proposed, .speculative_accepted = result.accepted }, .{});
        try output(json);
        try output("\n");
        return;
    }
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
    _ = @import("twodsd.zig");
    _ = @import("speculative.zig");
    _ = @import("token_batch_tests.zig");
    _ = @import("token_batch_attention_tests.zig");
    _ = @import("attention_gqa_stream_tests.zig");
    _ = @import("attention_mlx_wide_tests.zig");
    _ = @import("prepared_input_tests.zig");
    _ = @import("normalization_prepared_tests.zig");
    _ = @import("prepared_gate_fused_tests.zig");
    _ = @import("token_batch_weight_reuse_tests.zig");
    _ = @import("token_batch_greedy_tests.zig");
    _ = @import("token_batch_mlp_weight_reuse_tests.zig");
    _ = @import("config.zig");
    _ = @import("safetensors.zig");
    _ = @import("sampling.zig");
    _ = @import("metal.zig");
    _ = @import("optimized_tests.zig");
    _ = @import("resident.zig");
    _ = @import("tiling_tests.zig");
    _ = @import("model.zig");
    _ = @import("session.zig");
    _ = @import("attention_parallel_tests.zig");
    _ = @import("attention_split_tests.zig");
    _ = @import("attention_mlx_tests.zig");
    _ = @import("normalization_tests.zig");
    _ = @import("normalization_aligned_tests.zig");
    _ = @import("routing_tests.zig");
    _ = @import("dense_matvec_tests.zig");
    _ = @import("down_parallel_tests.zig");
    _ = @import("fixed_loop_tests.zig");
    _ = @import("attention_fixed_tests.zig");
    _ = @import("attention_gqa_reuse_tests.zig");
    _ = @import("kv_bf16_tests.zig");
    _ = @import("bf16_model_tests.zig");
    _ = @import("affine_load_tests.zig");
    _ = @import("router_bf16_tests.zig");
    _ = @import("replay_tests.zig");
}
