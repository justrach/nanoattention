//! Synchronous, teacher-forced benchmark. Loading, reset, and warmup are untimed.
//! Link this root module with the same metal.m, libc, and frameworks as main.zig.
const std = @import("std");
const metal = @import("metal.zig");
const Model = @import("model.zig").Model;
const sampling = @import("sampling.zig");

const Timespec = extern struct { sec: i64, nsec: i64 };
extern "c" fn clock_gettime(c_int, *Timespec) c_int;
extern "c" fn clock() c_ulong;
extern "c" fn write(c_int, [*]const u8, usize) isize;

const DecodeCandidate = struct {
    head_row_tile: u32 = 2,
    qkv_row_tile: u32 = 0,
    gate_row_tile: u32 = 0,
    projection_row_tile: u32 = 0,
    packed_router: bool = false,
    replay: bool = false,
    bf16_vector_loads: bool = false,
    fixed_loops: bool = false,
    affine_loads: bool = false,
    fixed_attention: bool = false,
    gqa_reuse: u32 = 1,
    gqa_blocks: u32 = 1,
    gqa_prefetch: u32 = 4,
    wide_attention: bool = false,
    ring_attention: bool = false,
    prepared_inputs: bool = false,
    prepared_fused_norm: bool = false,
    prepared_fused_gate: bool = false,
    parallel_down: bool = false,
    down_row_tile: u32 = 0,
};

const Input = struct {
    checkpoint: []const u8,
    prompt: []const u32,
    decode: []const u32,
    repeats: u32 = 3,
    resident_workers: u32 = 0,
    resident_row_tile: u32 = 1,
    staged_row_tile: u32 = 1,
    emit_predictions: bool = false,
    reuse_prefill: bool = false,
    mlx_attention: bool = false,
    kv_bf16: bool = false,
    bf16_blocks: u32 = 0,
    attention_chunk: ?u32 = null,
    attention_candidates: ?[]const u32 = null,
    decode_candidates: ?[]const DecodeCandidate = null,
    profile_decode: bool = false,
    profile_after_sweep: bool = false,
    profile_candidate: u32 = 1,
    prepare_replay: bool = false,
};

// Decode-only trials may restore a prefix computed by real model inference.
// Snapshot/restore is outside decode timing; no cache values or positions are
// fabricated. The single prefill sample is labeled explicitly in every row.
const Prefix = struct {
    position: u32,
    caches: []const struct { keys: []u8, values: []u8 },

    fn save(a: std.mem.Allocator, model: *Model) !Prefix {
        const caches = try a.alloc(@typeInfo(@FieldType(Prefix, "caches")).pointer.child, model.layers.len);
        for (model.layers, caches) |layer, *cache| {
            const bytes = try std.math.mul(usize, @min(model.position, layer.capacity), layer.keys.len / layer.capacity);
            cache.* = .{
                .keys = try a.dupe(u8, layer.keys.structs(u8)[0..bytes]),
                .values = try a.dupe(u8, layer.values.structs(u8)[0..bytes]),
            };
        }
        return .{ .position = model.position, .caches = caches };
    }
    fn restore(self: Prefix, model: *Model) void {
        for (model.layers, self.caches) |layer, cache| {
            @memcpy(layer.keys.structs(u8)[0..cache.keys.len], cache.keys);
            @memcpy(layer.values.structs(u8)[0..cache.values.len], cache.values);
        }
        model.position = self.position;
        model.failed = false;
    }
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
    if (input.mlx_attention and (input.resident_workers != 0 or input.attention_chunk != null or input.attention_candidates != null)) return error.ConflictingAttentionMode;
    if (input.kv_bf16 and !input.mlx_attention) return error.InvalidBf16KvBenchmark;
    if (input.bf16_blocks != 0 and (!input.kv_bf16 or (input.bf16_blocks != 128 and input.bf16_blocks != 512 and input.bf16_blocks != 1024))) return error.InvalidBf16Blocks;
    const single = [_]u32{input.attention_chunk orelse 0};
    const candidates = input.attention_candidates orelse &single;
    if (candidates.len == 0) return error.EmptyAttentionCandidates;
    if (candidates.len > 1 and !input.reuse_prefill) return error.AttentionCandidatesRequirePrefixRestore;
    if (input.resident_workers != 0 and (input.attention_chunk != null or input.attention_candidates != null)) return error.AttentionTuningRequiresStaged;
    for (candidates) |chunk| if (chunk != 0 and !@import("resident.zig").certified.validAttentionChunk(chunk)) return error.InvalidAttentionChunk;
    if (input.decode_candidates) |configs| {
        if (!input.reuse_prefill or !input.mlx_attention or input.resident_workers != 0 or configs.len == 0 or configs.len > 16 or input.decode.len == 0) return error.InvalidDecodeSweep;
        if (configs[0].head_row_tile != input.staged_row_tile or configs[0].qkv_row_tile != 0 or configs[0].gate_row_tile != 0 or configs[0].projection_row_tile != 0 or configs[0].packed_router or configs[0].replay or configs[0].bf16_vector_loads or configs[0].fixed_loops or configs[0].affine_loads or configs[0].fixed_attention or configs[0].gqa_reuse != 1 or configs[0].gqa_blocks != 1 or configs[0].gqa_prefetch != 4 or configs[0].wide_attention or configs[0].ring_attention or configs[0].prepared_inputs or configs[0].prepared_fused_norm or configs[0].prepared_fused_gate or configs[0].parallel_down or (configs[0].down_row_tile != 0 and configs[0].down_row_tile != input.staged_row_tile)) return error.DecodeSweepRequiresBaselineFirst;
        for (configs) |config| {
            if ((config.prepared_fused_norm or config.prepared_fused_gate) and !config.prepared_inputs) return error.PreparedFusionRequiresPreparedInputs;
            if (config.prepared_inputs and (input.prepare_replay or config.parallel_down)) return error.PreparedInputsRequireOrdinaryDecode;
            if (config.replay and !input.prepare_replay) return error.ReplayNotPrepared;
            for ([_]u32{ config.qkv_row_tile, config.gate_row_tile, config.projection_row_tile }) |tile|
                if (tile != 0 and !@import("resident.zig").certified.validRowTile(tile)) return error.InvalidStagedRowTile;
            if ((config.wide_attention or config.ring_attention) and input.kv_bf16) return error.WideAttentionRequiresFp32Kv;
            if (config.bf16_vector_loads and !input.kv_bf16) return error.InvalidBf16VectorLoads;
            if (config.gqa_reuse != 1 and config.gqa_reuse != 2 and config.gqa_reuse != 4) return error.InvalidGqaReuse;
            if (config.gqa_prefetch != 4 and !((config.gqa_reuse == 2 or config.gqa_reuse == 4) and config.gqa_blocks == 1 and !input.kv_bf16 and (config.gqa_prefetch == 1 or config.gqa_prefetch == 2))) return error.InvalidGqaPrefetch;
            if (config.gqa_blocks != 1 and !(config.gqa_reuse == 2 and (config.gqa_blocks == 2 or config.gqa_blocks == 4)) and !(config.gqa_reuse == 4 and config.gqa_blocks == 4)) return error.InvalidGqaBlocks;
            if (!@import("resident.zig").certified.validRowTile(config.head_row_tile) or
                (config.down_row_tile != 0 and !@import("resident.zig").certified.validRowTile(config.down_row_tile))) return error.InvalidStagedRowTile;
        }
    }
    if (input.profile_after_sweep and (input.decode_candidates == null or input.resident_workers != 0 or input.decode.len == 0 or input.profile_decode or input.prepare_replay)) return error.InvalidSweepProfile;
    if (input.profile_after_sweep and input.profile_candidate >= input.decode_candidates.?.len) return error.InvalidProfileCandidate;
    if (input.profile_decode and (input.decode_candidates != null or input.decode.len == 0 or input.decode.len > 8 or input.repeats != 1)) return error.InvalidProfileRequest;
    if (input.prepare_replay and (input.resident_workers != 0 or input.profile_decode)) return error.InvalidReplayBenchmark;
    const required = try std.math.add(usize, input.prompt.len, input.decode.len);
    const context = std.math.cast(u32, @max(1, required)) orelse return error.ContextExhausted;

    const ctx = try metal.Context.initMatched(input.mlx_attention);
    defer ctx.deinit();
    if (input.prepare_replay) try ctx.prepareReplay();
    var model = try Model.initWithKv(init.gpa, ctx, input.checkpoint, context, input.kv_bf16);
    defer model.deinit();
    try model.setBf16Blocks(input.bf16_blocks);
    if (input.resident_workers != 0) {
        try model.enableResident(input.resident_workers);
        try model.setResidentRowTile(input.resident_row_tile);
    } else {
        try model.setStagedRowTile(input.staged_row_tile);
        try model.setAttentionChunk(candidates[0]);
    }
    if (input.mlx_attention) try model.enableMlxAttention();
    // Sweeps keep the seed prefix on the reference policy; candidates change
    // decode only, and every candidate sees the identical genuine KV snapshot.
    if (input.decode_candidates != null) model.gqa_reuse = 1;
    // Validate all inputs before mutating the cache; repetitions never deduplicate
    // input tokens, stop at EOS, or feed the predicted tokens back into the model.
    for (input.prompt) |token| if (token >= model.cfg.vocab_size) return error.InvalidToken;
    for (input.decode) |token| if (token >= model.cfg.vocab_size) return error.InvalidToken;

    const seed_gpu_start = ctx.stats().gpu_seconds;
    const seed_start = try now();
    _ = try model.prefill(input.prompt);
    const seed_prefill_seconds = (try now()) - seed_start;
    const seed_prefill_gpu_seconds = ctx.stats().gpu_seconds - seed_gpu_start;
    if (input.profile_decode) {
        try ctx.beginProfile();
        for (input.decode) |token| _ = try sampling.greedy(try model.forward(token));
        try output("{\"phase\":\"profile\",\"benchmark\":false}\n");
        return;
    }
    const prefix: ?Prefix = if (input.reuse_prefill) try Prefix.save(init.arena.allocator(), &model) else null;
    if (input.decode_candidates != null) return decodeSweep(&model, input, prefix.?, seed_prefill_seconds, seed_prefill_gpu_seconds, init.arena.allocator());
    if (input.prepare_replay) try ctx.setReplay(true);
    var kv_allocated_bytes: u64 = 0;
    for (model.layers) |layer| kv_allocated_bytes += layer.keys.len + layer.values.len;
    const predictions: []u32 = if (input.emit_predictions) try init.arena.allocator().alloc(u32, input.decode.len) else &.{};
    for (candidates) |chunk| {
        // Tuning changes only scheduling; every candidate receives the same
        // genuinely computed prefix, never synthetic KV values or positions.
        model.attention_chunk = chunk;
        if (prefix) |snapshot| snapshot.restore(&model);
        for (input.decode[0..@min(8, input.decode.len)]) |token|
            _ = try sampling.greedy(try model.forward(token));
        for (0..input.repeats) |repeat| {
            if (prefix) |snapshot| snapshot.restore(&model) else try model.reset();
            const gpu_start = ctx.stats().gpu_seconds;
            const start = try now();
            if (!input.reuse_prefill) _ = try model.prefill(input.prompt);
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
            var buffer: [768]u8 = undefined;
            try output(try std.fmt.bufPrint(
                &buffer,
                "{{\"engine\":\"zig_fp32\",\"repeat\":{d},\"prefill_seconds\":{d:.9},\"decode_seconds\":{d:.9},\"checksum\":{d},\"submissions_per_token\":{d},\"dispatches_per_token\":{d},\"prefill_gpu_seconds\":{d:.9},\"decode_gpu_seconds\":{d:.9},\"resident_row_tile\":{d},\"staged_row_tile\":{d},\"prefill_reused\":{},\"prefill_samples\":{d},\"attention_chunk\":{d}",
                .{ repeat, if (input.reuse_prefill) seed_prefill_seconds else prefilled - start, finished - prefilled, checksum, model.last_execution.submissions, model.last_execution.dispatches, if (input.reuse_prefill) seed_prefill_gpu_seconds else gpu_prefilled - gpu_start, gpu_finished - gpu_prefilled, input.resident_row_tile, model.staged_row_tile, input.reuse_prefill, if (input.reuse_prefill) @as(u32, 1) else input.repeats, chunk },
            ));
            if (model.mlx_attention) try output(",\"attention_implementation\":\"mlx_layout_adaptive_stripes\"");
            try output(if (model.kv_bf16) ",\"kv_storage\":\"bf16_global_fp32_sliding\"" else ",\"kv_storage\":\"fp32\"");
            try output(try std.fmt.bufPrint(&buffer, ",\"kv_allocated_bytes\":{d},\"bf16_blocks\":{d}", .{ kv_allocated_bytes, input.bf16_blocks }));
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
}

// Candidate admission compares every vocabulary bit at every decode position,
// outside timing. Timing alternates candidates against one real prefix snapshot.
fn decodeSweep(model: *Model, input: Input, prefix: Prefix, prefill_seconds: f64, prefill_gpu_seconds: f64, a: std.mem.Allocator) !void {
    const configs = input.decode_candidates.?;
    for (configs) |config| if (config.prepared_inputs) {
        try model.prepareQuantInputs();
        break;
    };
    for (configs) |config| if (config.parallel_down) {
        try model.prepareParallelDown();
        break;
    };
    const vocab: usize = model.cfg.vocab_size;
    const reference = try a.alloc(u32, try std.math.mul(usize, input.decode.len, vocab));
    const admitted = try a.alloc(bool, configs.len);
    @memset(admitted, true);
    for (configs, 0..) |config, index| {
        if (input.prepare_replay) try model.ctx.setReplay(config.replay);
        model.head_row_tile = config.head_row_tile;
        model.packed_router = config.packed_router;
        model.bf16_vector_loads = config.bf16_vector_loads;
        model.fixed_loops = config.fixed_loops;
        model.affine_loads = config.affine_loads;
        model.fixed_attention = config.fixed_attention;
        model.gqa_reuse = config.gqa_reuse;
        model.gqa_blocks = config.gqa_blocks;
        model.gqa_prefetch = config.gqa_prefetch;
        model.wide_attention = config.wide_attention;
        model.ring_attention = config.ring_attention;
        model.prepared_inputs = config.prepared_inputs;
        model.prepared_fused_norm = config.prepared_fused_norm;
        model.prepared_fused_gate = config.prepared_fused_gate;
        model.parallel_down = config.parallel_down;
        model.down_row_tile = config.down_row_tile;
        model.qkv_row_tile = config.qkv_row_tile;
        model.gate_row_tile = config.gate_row_tile;
        model.projection_row_tile = config.projection_row_tile;
        prefix.restore(model);
        for (input.decode, 0..) |token, position| {
            const logits = try model.forward(token);
            _ = try sampling.greedy(logits);
            const bits = std.mem.bytesAsSlice(u32, std.mem.sliceAsBytes(logits));
            const expected = reference[position * vocab ..][0..vocab];
            if (index == 0) @memcpy(expected, bits) else if (!std.mem.eql(u32, expected, bits)) {
                std.debug.print("Decode candidate {d} changes vocabulary bits at position {d}\n", .{ index, position });
                admitted[index] = false;
                break;
            }
        }
        try output(try std.json.Stringify.valueAlloc(a, .{ .phase = "validation", .candidate = index, .config = config, .positions = input.decode.len, .vocabulary = vocab, .logits_bit_exact = admitted[index] }, .{}));
        try output("\n");
    }
    var model_buffer_bytes: u64 = 0;
    var kv_allocated_bytes: u64 = 0;
    for (model.buffers.items) |buffer| model_buffer_bytes += buffer.len;
    for (model.layers) |layer| kv_allocated_bytes += layer.keys.len + layer.values.len;
    const predictions = try a.alloc(u32, input.decode.len);
    for (0..input.repeats) |repeat| {
        for (0..configs.len) |offset| {
            // Rotate order to avoid always granting one candidate the same
            // thermal/CPU-scheduling position within a round.
            const index = (offset + repeat) % configs.len;
            if (!admitted[index]) continue;
            const config = configs[index];
            if (input.prepare_replay) try model.ctx.setReplay(config.replay);
            model.head_row_tile = config.head_row_tile;
            model.packed_router = config.packed_router;
            model.bf16_vector_loads = config.bf16_vector_loads;
            model.fixed_loops = config.fixed_loops;
            model.affine_loads = config.affine_loads;
            model.fixed_attention = config.fixed_attention;
            model.gqa_reuse = config.gqa_reuse;
            model.gqa_blocks = config.gqa_blocks;
            model.gqa_prefetch = config.gqa_prefetch;
            model.wide_attention = config.wide_attention;
            model.ring_attention = config.ring_attention;
            model.prepared_inputs = config.prepared_inputs;
            model.prepared_fused_norm = config.prepared_fused_norm;
            model.prepared_fused_gate = config.prepared_fused_gate;
            model.parallel_down = config.parallel_down;
            model.down_row_tile = config.down_row_tile;
            model.qkv_row_tile = config.qkv_row_tile;
            model.gate_row_tile = config.gate_row_tile;
            model.projection_row_tile = config.projection_row_tile;
            prefix.restore(model);
            for (input.decode[0..@min(8, input.decode.len)]) |token| _ = try sampling.greedy(try model.forward(token));
            prefix.restore(model);
            const cpu_start = clock();
            const gpu_start = model.ctx.stats().gpu_seconds;
            const start = try now();
            var checksum: u64 = 0;
            for (input.decode, predictions) |token, *prediction| {
                prediction.* = try sampling.greedy(try model.forward(token));
                checksum += prediction.*;
            }
            const seconds = (try now()) - start;
            const gpu_seconds = model.ctx.stats().gpu_seconds - gpu_start;
            // Darwin CLOCKS_PER_SEC is 1,000,000; includes all process threads.
            const cpu_seconds = @as(f64, @floatFromInt(clock() - cpu_start)) / 1000000.0;
            const row = .{
                .phase = "timing",
                .engine = "zig_fp32",
                .repeat = repeat,
                .candidate = index,
                .config = config,
                .prefill_seconds = prefill_seconds,
                .prefill_gpu_seconds = prefill_gpu_seconds,
                .prefill_samples = @as(u32, 1),
                .prefill_reused = true,
                .decode_seconds = seconds,
                .decode_gpu_seconds = gpu_seconds,
                .decode_cpu_seconds = cpu_seconds,
                .checksum = checksum,
                .predictions = predictions,
                .submissions_per_token = model.last_execution.submissions,
                .dispatches_per_token = model.last_execution.dispatches,
                .checkpoint_mapped_bytes = model.store.bytes,
                .model_buffer_bytes = model_buffer_bytes,
                .kv_allocated_bytes = kv_allocated_bytes,
                .reference_logit_bytes = reference.len * @sizeOf(u32),
                .kv_storage = if (model.kv_bf16) "bf16_global_fp32_sliding" else "fp32",
                .bf16_blocks = input.bf16_blocks,
            };
            try output(try std.json.Stringify.valueAlloc(a, row, .{}));
            try output("\n");
        }
    }
    if (input.profile_after_sweep) {
        const index = input.profile_candidate;
        if (!admitted[index]) return error.ProfileCandidateRejected;
        const config = configs[index];
        model.head_row_tile = config.head_row_tile;
        model.packed_router = config.packed_router;
        model.bf16_vector_loads = config.bf16_vector_loads;
        model.fixed_loops = config.fixed_loops;
        model.affine_loads = config.affine_loads;
        model.fixed_attention = config.fixed_attention;
        model.gqa_reuse = config.gqa_reuse;
        model.gqa_blocks = config.gqa_blocks;
        model.gqa_prefetch = config.gqa_prefetch;
        model.wide_attention = config.wide_attention;
        model.ring_attention = config.ring_attention;
        model.prepared_inputs = config.prepared_inputs;
        model.prepared_fused_norm = config.prepared_fused_norm;
        model.prepared_fused_gate = config.prepared_fused_gate;
        model.parallel_down = config.parallel_down;
        model.down_row_tile = config.down_row_tile;
        model.qkv_row_tile = config.qkv_row_tile;
        model.gate_row_tile = config.gate_row_tile;
        model.projection_row_tile = config.projection_row_tile;
        prefix.restore(model);
        try model.ctx.beginProfile();
        const tokens = @min(8, input.decode.len);
        for (input.decode[0..tokens]) |token| _ = try sampling.greedy(try model.forward(token));
        try output(try std.json.Stringify.valueAlloc(a, .{ .phase = "profile", .benchmark = false, .candidate = index, .tokens = tokens }, .{}));
        try output("\n");
    }
}
