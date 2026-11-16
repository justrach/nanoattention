//! Exact greedy admission followed by free-running history-draft timing.
//! Forced future tokens appear ONLY in untimed target-row/rollback checks.
const std = @import("std");
const metal = @import("metal.zig");
const Model = @import("model.zig").Model;
const generation = @import("speculative_generation.zig");

const Timespec = extern struct { sec: i64, nsec: i64 };
extern "c" fn clock_gettime(c_int, *Timespec) c_int;
extern "c" fn write(c_int, [*]const u8, usize) isize;

const Input = struct {
    checkpoint: []const u8,
    prompt_tokens: []const u32,
    max_tokens: u32 = 256,
    draft_tokens: u32 = 4,
    draft_candidates: []const u32 = &.{},
    query_groups: []const u32 = &.{2},
    expert_tiles: []const u32 = &.{0},
    compare_mlp_weight_reuse: bool = false,
    repeats: u32 = 3,
    kv_bf16: bool = false,
    weight_reuse: bool = false,
    mlp_weight_reuse: bool = false,
};
const Cache = struct { keys: []u8, values: []u8 };
const Prefix = struct {
    position: u32,
    caches: []Cache,
    logits: []f32,

    fn save(a: std.mem.Allocator, model: *Model, logits: []const f32) !Prefix {
        try model.ctx.finish();
        const caches = try a.alloc(Cache, model.layers.len);
        var initialized: usize = 0;
        errdefer {
            for (caches[0..initialized]) |cache| {
                a.free(cache.keys);
                a.free(cache.values);
            }
            a.free(caches);
        }
        for (model.layers, caches) |layer, *cache| {
            // Preserve physical slots, not chronological order. Once wrapped,
            // every slot in a sliding ring is valid and must be restored.
            const slots: usize = @min(model.position, layer.capacity);
            const keys = try a.dupe(u8, layer.keys.structs(u8)[0 .. slots * (layer.keys.len / layer.capacity)]);
            errdefer a.free(keys);
            const values = try a.dupe(u8, layer.values.structs(u8)[0 .. slots * (layer.values.len / layer.capacity)]);
            cache.* = .{ .keys = keys, .values = values };
            initialized += 1;
        }
        return .{ .position = model.position, .caches = caches, .logits = try a.dupe(f32, logits) };
    }
    fn deinit(self: Prefix, a: std.mem.Allocator) void {
        for (self.caches) |cache| {
            a.free(cache.keys);
            a.free(cache.values);
        }
        a.free(self.caches);
        a.free(self.logits);
    }
    fn restore(self: Prefix, model: *Model) !void {
        try model.reset();
        for (model.layers, self.caches) |layer, cache| {
            @memcpy(layer.keys.structs(u8)[0..cache.keys.len], cache.keys);
            @memcpy(layer.values.structs(u8)[0..cache.values.len], cache.values);
        }
        @memcpy(model.scratch.logits.floats(), self.logits);
        model.position = self.position;
    }
    fn check(self: Prefix, model: *Model) !void {
        if (model.position != self.position) return error.RollbackPositionMismatch;
        try equalLogits(self.logits, model.scratch.logits.floats());
        for (model.layers, self.caches) |layer, cache| {
            if (!std.mem.eql(u8, cache.keys, layer.keys.structs(u8)[0..cache.keys.len]) or
                !std.mem.eql(u8, cache.values, layer.values.structs(u8)[0..cache.values.len]))
                return error.RollbackCacheMismatch;
        }
    }
};
const Mode = struct {
    draft: u32,
    query: u32 = 2,
    tile: u32 = 0,
    weight_reuse: bool = false,
    mlp_weight_reuse: bool = false,

    fn apply(self: Mode, model: *Model) void {
        model.speculative_tokens = self.draft;
        // Ordinary reference, warmup, timed baseline, and serial row checks
        // always disable the candidate-only reuse path.
        model.batch_weight_reuse = self.draft != 0 and self.weight_reuse;
        model.batch_mlp_weight_reuse = self.draft != 0 and self.mlp_weight_reuse;
        model.batch_query_group = self.query;
        model.gate_row_tile = self.tile;
        model.down_row_tile = self.tile;
    }
};
const Trial = struct {
    draft_tokens: u32 = 0,
    query_group: u32 = 2,
    expert_tile: u32 = 0,
    mlp_weight_reuse: bool = false,
    round: usize = 0,
    order: usize = 0,
    seconds: f64,
    gpu_seconds: f64,
    count: usize,
    eos: bool,
    batches: usize,
    proposed: usize,
    accepted: usize,
};

fn now() !f64 {
    var ts: Timespec = undefined;
    if (clock_gettime(6, &ts) != 0) return error.ClockFailed;
    return @as(f64, @floatFromInt(ts.sec)) + @as(f64, @floatFromInt(ts.nsec)) * 1e-9;
}
fn output(bytes: []const u8) !void {
    var offset: usize = 0;
    while (offset < bytes.len) {
        const n = write(1, bytes[offset..].ptr, bytes.len - offset);
        if (n <= 0) return error.OutputFailed;
        offset += @intCast(n);
    }
}
fn equalLogits(expected: []const f32, actual: []const f32) !void {
    if (!std.mem.eql(u8, std.mem.sliceAsBytes(expected), std.mem.sliceAsBytes(actual)))
        return error.ExactLogitBitsMismatch;
}
fn timedGenerate(model: *Model, a: std.mem.Allocator, history: *std.ArrayList(u32), tokens: []u32, initial: []const f32) !struct { trial: Trial, logits: []const f32 } {
    const gpu_start = model.ctx.stats().gpu_seconds;
    const start = try now();
    const result = try generation.generate(model, a, history, tokens, initial);
    const seconds = (try now()) - start;
    const gpu_seconds = model.ctx.stats().gpu_seconds - gpu_start;
    if (seconds <= 0 or !std.math.isFinite(seconds)) return error.InvalidTiming;
    // GPU command clocks are distinct from the wall clock; do not impose a
    // wall-time upper bound or interpret their difference as pure CPU time.
    if (gpu_seconds < 0 or !std.math.isFinite(gpu_seconds)) return error.InvalidGpuTiming;
    if (model.position != history.items.len) return error.HistoryPositionMismatch;
    return .{
        .trial = .{ .seconds = seconds, .gpu_seconds = gpu_seconds, .count = result.count, .eos = result.eos, .batches = result.batches, .proposed = result.proposed, .accepted = result.accepted },
        .logits = result.logits,
    };
}

// Full-vocabulary bit comparisons; generated reference tokens are allowed only
// here. No forced token enters a timed speculative generation call.
fn validateRows(a: std.mem.Allocator, model: *Model, prefix: Prefix, forced: []const u32) !usize {
    if (forced.len == 0) return error.NoValidationTokens;
    const mode: Mode = .{ .draft = model.speculative_tokens, .query = model.batch_query_group, .tile = model.gate_row_tile, .weight_reuse = model.batch_weight_reuse, .mlp_weight_reuse = model.batch_mlp_weight_reuse };
    const vocab: usize = model.cfg.vocab_size;
    const serial = try a.alloc(f32, try std.math.mul(usize, forced.len, vocab));
    defer a.free(serial);
    const batched = try a.alloc(f32, serial.len);
    defer a.free(batched);
    try prefix.restore(model);
    const rows = try model.forwardBatch(forced);
    for (rows, 0..) |row, i| @memcpy(batched[i * vocab ..][0..vocab], row.logits.floats());
    const rolled_back = try model.finishBatch(0);
    try equalLogits(prefix.logits, rolled_back);
    try prefix.check(model);
    // Intentionally do NOT restore here: serial evaluation tests zero-commit
    // rollback itself, including overwritten sliding slots. Serial rows use
    // ordinary defaults, not candidate-specific MLP tiling.
    (Mode{ .draft = 0 }).apply(model);
    for (forced, 0..) |token, i| {
        const logits = try model.forward(token);
        @memcpy(serial[i * vocab ..][0..vocab], logits);
        try equalLogits(batched[i * vocab ..][0..vocab], logits);
    }
    var checks: usize = 0;
    if (forced.len > 1) {
        const commits = [_]usize{ 1, forced.len - 1 };
        for (commits, 0..) |accepted, index| {
            if (index == 1 and accepted == commits[0]) continue;
            try prefix.restore(model);
            mode.apply(model);
            _ = try model.forwardBatch(forced);
            const retained = try model.finishBatch(accepted);
            if (model.position != prefix.position + accepted) return error.PartialCommitPositionMismatch;
            try equalLogits(serial[(accepted - 1) * vocab ..][0..vocab], retained);
            (Mode{ .draft = 0 }).apply(model);
            for (forced[accepted..], accepted..) |token, i| {
                try equalLogits(serial[i * vocab ..][0..vocab], try model.forward(token));
            }
            if (model.position != prefix.position + forced.len) return error.PartialCommitPositionMismatch;
            checks += 1;
        }
    }
    try prefix.restore(model);
    mode.apply(model);
    return checks;
}
fn restorePrefix(model: *Model, history: *std.ArrayList(u32), prefix: Prefix, prompt: []const u32) !void {
    try prefix.restore(model);
    history.clearRetainingCapacity();
    history.appendSliceAssumeCapacity(prompt);
}
fn emitTrial(trial: Trial) !void {
    var buffer: [512]u8 = undefined;
    try output(try std.fmt.bufPrint(&buffer, "{{\"draft_tokens\":{d},\"query_group\":{d},\"expert_tile\":{d},\"mlp_weight_reuse\":{},\"round\":{d},\"order\":{d},\"decode_seconds\":{d:.9},\"gpu_seconds\":{d:.9},\"committed_tokens\":{d},\"tokens_per_second\":{d:.6},\"eos\":{},\"batches\":{d},\"proposed\":{d},\"accepted\":{d}}}", .{ trial.draft_tokens, trial.query_group, trial.expert_tile, trial.mlp_weight_reuse, trial.round, trial.order, trial.seconds, trial.gpu_seconds, trial.count, @as(f64, @floatFromInt(trial.count)) / trial.seconds, trial.eos, trial.batches, trial.proposed, trial.accepted }));
}

pub fn main(init: std.process.Init) !void {
    var args = std.process.Args.Iterator.init(init.minimal.args);
    _ = args.skip();
    const path = args.next() orelse return error.MissingInput;
    if (std.mem.eql(u8, path, "--help")) {
        try output("Usage: flint-speculative-bench INPUT_JSON\nJSON: checkpoint, prompt_tokens (nonempty u32 array), max_tokens=256, draft_tokens=4 (1..32), draft_candidates=[] (empty uses draft_tokens), repeats=3, kv_bf16=false. query_groups=[2] (2/4/8), expert_tiles=[0] (0/1/2/4/8), at most 16 combinations. weight_reuse=false and mlp_weight_reuse=false independently apply only to speculative modes; ordinary always disables both. compare_mlp_weight_reuse=false; true tests both MLP settings per mode within the 16-mode limit. Schema v3: rotated same-prefix ordinary/candidate trials; round/order are zero-based.\n");
        return;
    }
    if (args.next() != null) return error.UnexpectedArgument;
    const a = init.gpa;
    const mapping = try metal.Mapping.open(a, path);
    defer mapping.deinit();
    const parsed = try std.json.parseFromSlice(Input, a, mapping.bytes, .{});
    defer parsed.deinit();
    const input = parsed.value;
    if (input.checkpoint.len == 0) return error.MissingCheckpoint;
    if (input.prompt_tokens.len == 0) return error.EmptyPrompt;
    if (input.max_tokens == 0 or input.repeats == 0) return error.InvalidBudget;
    const fallback = [_]u32{input.draft_tokens};
    const candidates: []const u32 = if (input.draft_candidates.len == 0) &fallback else input.draft_candidates;
    for (candidates, 0..) |candidate, i| {
        if (candidate == 0 or candidate > Model.max_speculative_tokens) return error.InvalidSpeculativeLength;
        if (std.mem.indexOfScalar(u32, candidates[0..i], candidate) != null) return error.DuplicateDraftCandidate;
    }
    if (input.query_groups.len == 0 or input.expert_tiles.len == 0) return error.EmptyModeAxis;
    for (input.query_groups, 0..) |query, i| {
        if (query != 2 and query != 4 and query != 8) return error.InvalidQueryGroup;
        if (std.mem.indexOfScalar(u32, input.query_groups[0..i], query) != null) return error.DuplicateQueryGroup;
    }
    for (input.expert_tiles, 0..) |tile, i| {
        if (tile != 0 and tile != 1 and tile != 2 and tile != 4 and tile != 8) return error.InvalidExpertTile;
        if (std.mem.indexOfScalar(u32, input.expert_tiles[0..i], tile) != null) return error.DuplicateExpertTile;
    }
    const mlp_options = [_]bool{ false, true };
    const requested_mlp = [_]bool{input.mlp_weight_reuse};
    const mlp_modes: []const bool = if (input.compare_mlp_weight_reuse) &mlp_options else &requested_mlp;
    const base_combinations = try std.math.mul(usize, try std.math.mul(usize, candidates.len, input.query_groups.len), input.expert_tiles.len);
    const combination_count = try std.math.mul(usize, base_combinations, mlp_modes.len);
    if (combination_count > 16) return error.TooManyModes;
    var combinations: [16]Mode = undefined;
    var combination_index: usize = 0;
    for (candidates) |draft| {
        for (input.query_groups) |query| {
            for (input.expert_tiles) |tile| {
                for (mlp_modes) |mlp_reuse| {
                    combinations[combination_index] = .{ .draft = draft, .query = query, .tile = tile, .weight_reuse = input.weight_reuse, .mlp_weight_reuse = mlp_reuse };
                    combination_index += 1;
                }
            }
        }
    }
    const required = try std.math.add(usize, input.prompt_tokens.len, input.max_tokens);
    const context = std.math.cast(u32, required) orelse return error.ContextExhausted;
    const ctx = try metal.Context.initMatched(true);
    defer ctx.deinit();
    var model = try Model.initWithKv(a, ctx, input.checkpoint, context, input.kv_bf16);
    defer model.deinit();
    try model.setStagedRowTile(2);
    try model.enableMlxAttention();
    try model.prepareBatch();
    for (input.prompt_tokens) |token| if (token >= model.cfg.vocab_size) return error.InvalidToken;
    var kv_bytes: u64 = 0;
    var wrapped_layers: usize = 0;
    for (model.layers, 0..) |layer, i| {
        kv_bytes += layer.keys.len + layer.values.len;
        if (model.cfg.sliding(i) and input.prompt_tokens.len > layer.capacity) wrapped_layers += 1;
    }
    var history: std.ArrayList(u32) = .empty;
    defer history.deinit(a);
    try history.ensureTotalCapacity(a, try std.math.add(usize, required, 1));
    history.appendSliceAssumeCapacity(input.prompt_tokens);
    const reference_tokens = try a.alloc(u32, input.max_tokens);
    defer a.free(reference_tokens);
    const candidate_tokens = try a.alloc(u32, input.max_tokens);
    defer a.free(candidate_tokens);
    const ordinary_trials = try a.alloc(Trial, input.repeats);
    defer a.free(ordinary_trials);
    const speculative_trials = try a.alloc(Trial, try std.math.mul(usize, input.repeats, combination_count));
    defer a.free(speculative_trials);

    const prefill_start = try now();
    const initial = try model.prefill(input.prompt_tokens);
    const prefill_seconds = (try now()) - prefill_start;
    const prefix = try Prefix.save(a, &model, initial);
    defer prefix.deinit(a);
    // The first ordinary run is an untimed admission reference, never a speed
    // sample. Its future tokens/logits are used only after generation/checks.
    (Mode{ .draft = 0 }).apply(&model);
    const reference = try generation.generate(&model, a, &history, reference_tokens, prefix.logits);
    if (model.position != history.items.len) return error.HistoryPositionMismatch;
    const final_logits = try a.dupe(f32, reference.logits);
    defer a.free(final_logits);
    const forced_count = @min(@as(usize, Model.max_speculative_tokens), reference.count);
    var partial_checks: usize = 0;
    // Admit full-vocabulary rows and zero/partial rollback for EVERY knob combination.
    for (combinations[0..combination_count]) |mode| {
        mode.apply(&model);
        partial_checks += try validateRows(a, &model, prefix, reference_tokens[0..forced_count]);
    }

    // Warm each mode from the identical prefix, outside all decode timings.
    // No reference tokens are passed to generate/propose, even for warmup.
    const warmup_max_tokens = @min(@as(usize, 32), input.max_tokens);
    const modes = combination_count + 1;
    for (0..modes) |slot| {
        try restorePrefix(&model, &history, prefix, input.prompt_tokens);
        const mode: Mode = if (slot == 0) .{ .draft = 0 } else combinations[slot - 1];
        mode.apply(&model);
        const warmup = try generation.generate(&model, a, &history, candidate_tokens[0..warmup_max_tokens], prefix.logits);
        if (warmup.count != @min(reference.count, warmup_max_tokens) or
            warmup.eos != (reference.eos and reference.count <= warmup_max_tokens) or
            !std.mem.eql(u32, reference_tokens[0..warmup.count], candidate_tokens[0..warmup.count]))
            return error.ExactGreedyTokenMismatch;
        if (model.position != history.items.len) return error.HistoryPositionMismatch;
        if (warmup.count == reference.count) try equalLogits(final_logits, warmup.logits);
    }
    try restorePrefix(&model, &history, prefix, input.prompt_tokens);

    // Each round includes ordinary plus EVERY candidate, with a cyclically
    // rotated order. Every timed run uses the full requested output budget.
    for (ordinary_trials, 0..) |*ordinary_trial, round| {
        for (0..modes) |order| {
            const slot = (order + round % modes) % modes;
            try restorePrefix(&model, &history, prefix, input.prompt_tokens);
            const mode: Mode = if (slot == 0) .{ .draft = 0 } else combinations[slot - 1];
            mode.apply(&model);
            var candidate = try timedGenerate(&model, a, &history, candidate_tokens, prefix.logits);
            if (candidate.trial.count != reference.count or candidate.trial.eos != reference.eos or
                !std.mem.eql(u32, reference_tokens[0..reference.count], candidate_tokens[0..candidate.trial.count]))
                return error.ExactGreedyTokenMismatch;
            try equalLogits(final_logits, candidate.logits);
            candidate.trial.draft_tokens = model.speculative_tokens;
            candidate.trial.query_group = mode.query;
            candidate.trial.expert_tile = mode.tile;
            candidate.trial.mlp_weight_reuse = mode.mlp_weight_reuse;
            candidate.trial.round = round;
            candidate.trial.order = order;
            if (slot == 0) {
                ordinary_trial.* = candidate.trial;
            } else {
                speculative_trials[round * combination_count + slot - 1] = candidate.trial;
            }
        }
    }
    // No stdout speed data is emitted until ALL correctness checks pass.
    var buffer: [2048]u8 = undefined;
    try output(try std.fmt.bufPrint(&buffer, "{{\"schema_version\":3,\"engine\":\"native_matched_greedy\",\"exact_tokens\":true,\"exact_final_logit_bits\":true,\"exact_batch_row_bits\":true,\"forced_rows_correctness_only\":true,\"speed_workload\":\"free_running_history_draft\",\"reference_untimed\":true,\"warmup_timed\":false,\"warmup_max_tokens\":{d},\"timing_order\":\"rotated_by_round\",\"actual_prefix_tokens\":{d},\"max_tokens\":{d},\"prefill_seconds\":{d:.9},\"prefill_samples\":1,\"prefix_restore_timed\":false,\"forced_validation_rows\":{d},\"partial_commit_checks\":{d},\"wrapped_sliding_layers\":{d},\"kv_allocated_bytes\":{d},\"kv_storage\":\"{s}\",\"draft_candidates\":[", .{ warmup_max_tokens, prefix.position, input.max_tokens, prefill_seconds, forced_count, partial_checks, wrapped_layers, kv_bytes, if (input.kv_bf16) "bf16_global_fp32_sliding" else "fp32" }));
    for (candidates, 0..) |candidate, i| {
        if (i != 0) try output(",");
        try output(try std.fmt.bufPrint(&buffer, "{d}", .{candidate}));
    }
    try output(try std.fmt.bufPrint(&buffer, "],\"weight_reuse\":{},\"mlp_weight_reuse\":{},\"compare_mlp_weight_reuse\":{},\"weight_reuse_scope\":\"speculative_modes_only\",\"query_groups\":[", .{ input.weight_reuse, input.mlp_weight_reuse, input.compare_mlp_weight_reuse }));
    for (input.query_groups, 0..) |query, i| {
        if (i != 0) try output(",");
        try output(try std.fmt.bufPrint(&buffer, "{d}", .{query}));
    }
    try output("],\"expert_tiles\":[");
    for (input.expert_tiles, 0..) |tile, i| {
        if (i != 0) try output(",");
        try output(try std.fmt.bufPrint(&buffer, "{d}", .{tile}));
    }
    try output("],\"ordinary\":[");
    for (ordinary_trials, 0..) |trial, i| {
        if (i != 0) try output(",");
        try emitTrial(trial);
    }
    try output("],\"speculative\":[");
    for (speculative_trials, 0..) |trial, i| {
        if (i != 0) try output(",");
        try emitTrial(trial);
    }
    try output("]}\n");
}
