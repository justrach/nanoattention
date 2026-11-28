//! DFlash 2 block-diffusion speculative decoding draft engine (`DFlash2DraftModel`).
//! Runs natively on Apple Silicon via `src/mega/dflash2.metal` + shared `kernels.metal`
//! alongside a target `exec.Engine` (e.g. Inkling 2 Mini) using `exec.Capture` and `rewind`.
const std = @import("std");
const gpu = @import("gpu.zig");
const exec = @import("exec.zig");
const inkling_plan = @import("inkling_plan.zig");
const loader = @import("loader.zig");
const io = @import("io.zig");
const weights = @import("weights");
const metal = weights.metal;

pub const SoftcapMode = enum { before_correction, after_correction };

pub const Config = struct {
    hidden_size: u32,
    intermediate_size: u32,
    num_hidden_layers: u32,
    num_attention_heads: u32,
    num_key_value_heads: u32,
    head_dim: u32,
    sliding_window: u32,
    rms_norm_eps: f32,
    rope_theta: f32,
    vocab_size: u32,
    block_size: u32,
    conv_group_size: u32,
    conv_kernel_size: u32,
    final_logit_softcapping: f32,
    mask_token_id: u32,
    selector_rank: u32,
    selector_top_k: u32,
    selector_softcap_mode: SoftcapMode,
    num_target_layers: u32,
    target_layer_ids: []u32,
};

fn jsonObj(v: ?std.json.Value) !std.json.ObjectMap {
    const val = v orelse return error.MissingConfigField;
    return switch (val) {
        .object => |o| o,
        else => error.InvalidConfigObject,
    };
}

fn jsonU32(v: ?std.json.Value, default: ?u32) !u32 {
    const val = v orelse return default orelse error.MissingConfigField;
    return switch (val) {
        .integer => |n| std.math.cast(u32, n) orelse error.InvalidConfigInteger,
        else => error.InvalidConfigInteger,
    };
}

fn jsonF32(v: ?std.json.Value, default: ?f32) !f32 {
    const val = v orelse return default orelse error.MissingConfigField;
    return switch (val) {
        .integer => |n| @floatFromInt(n),
        .float => |n| @floatCast(n),
        else => error.InvalidConfigNumber,
    };
}

pub fn parseConfig(gpa: std.mem.Allocator, scratch: std.mem.Allocator, bytes: []const u8) !Config {
    const parsed = try std.json.parseFromSliceLeaky(std.json.Value, scratch, bytes, .{});
    const root = try jsonObj(parsed);
    const dc = try jsonObj(root.get("dflash_config"));
    const rope_theta: f32 = if (root.get("rope_parameters")) |rp| blk: {
        const o = try jsonObj(rp);
        break :blk try jsonF32(o.get("rope_theta"), 10000.0);
    } else if (root.get("rope_scaling")) |rs| blk: {
        const o = try jsonObj(rs);
        break :blk try jsonF32(o.get("rope_theta"), 10000.0);
    } else try jsonF32(root.get("rope_theta"), 10000.0);

    const ids_val = dc.get("target_layer_ids") orelse return error.MissingConfigField;
    const ids_arr = switch (ids_val) {
        .array => |arr| arr.items,
        else => return error.InvalidConfigField,
    };
    if (ids_arr.len == 0) return error.InvalidDraftCaptureLayers;
    const ids = try gpa.alloc(u32, ids_arr.len);
    errdefer gpa.free(ids);
    for (ids_arr, ids) |item, *dst| dst.* = try jsonU32(item, null);

    const mode_str = if (dc.get("selector_softcap_mode")) |mv| switch (mv) {
        .string => |s| s,
        else => return error.InvalidConfigField,
    } else "before_correction";
    const mode: SoftcapMode = if (std.mem.eql(u8, mode_str, "after_correction"))
        .after_correction
    else if (std.mem.eql(u8, mode_str, "before_correction"))
        .before_correction
    else
        return error.UnsupportedSelectorSoftcapMode;

    const cfg: Config = .{
        .hidden_size = try jsonU32(root.get("hidden_size"), null),
        .intermediate_size = try jsonU32(root.get("intermediate_size"), null),
        .num_hidden_layers = try jsonU32(root.get("num_hidden_layers"), null),
        .num_attention_heads = try jsonU32(root.get("num_attention_heads"), null),
        .num_key_value_heads = try jsonU32(root.get("num_key_value_heads"), null),
        .head_dim = try jsonU32(root.get("head_dim"), null),
        .sliding_window = try jsonU32(root.get("sliding_window"), 2048),
        .rms_norm_eps = try jsonF32(root.get("rms_norm_eps"), 1e-6),
        .rope_theta = rope_theta,
        .vocab_size = try jsonU32(root.get("vocab_size"), null),
        .block_size = try jsonU32(dc.get("block_size"), 8),
        .conv_group_size = try jsonU32(dc.get("conv_group_size"), 16),
        .conv_kernel_size = try jsonU32(dc.get("conv_kernel_size"), 2),
        .final_logit_softcapping = try jsonF32(dc.get("final_logit_softcapping"), 0.0),
        .mask_token_id = try jsonU32(dc.get("mask_token_id"), null),
        .selector_rank = try jsonU32(dc.get("selector_rank"), 256),
        .selector_top_k = try jsonU32(dc.get("selector_top_k"), 16),
        .selector_softcap_mode = mode,
        .num_target_layers = try jsonU32(root.get("num_target_layers"), ids[ids.len - 1]),
        .target_layer_ids = ids,
    };
    if (cfg.block_size < 2 or cfg.hidden_size % cfg.conv_group_size != 0 or cfg.hidden_size % 4 != 0 or cfg.head_dim % 2 != 0 or cfg.head_dim > 256) return error.InvalidDraftConfig;
    if (cfg.num_attention_heads % cfg.num_key_value_heads != 0) return error.InvalidDraftConfig;
    if (cfg.selector_top_k == 0 or cfg.selector_top_k > 16 or cfg.sliding_window < 2) return error.InvalidDraftConfig;
    return cfg;
}

const LayerWeights = struct {
    input_layernorm: weights.Weight,
    post_attention_layernorm: weights.Weight,
    attn_conv_base: weights.Weight,
    attn_conv_proj: weights.Weight,
    mlp_conv_base: weights.Weight,
    mlp_conv_proj: weights.Weight,
    q_proj: weights.Weight,
    k_proj: weights.Weight,
    v_proj: weights.Weight,
    o_proj: weights.Weight,
    q_norm: weights.Weight,
    k_norm: weights.Weight,
    gate_proj: weights.Weight,
    up_proj: weights.Weight,
    down_proj: weights.Weight,
};

pub const PrefillResult = struct {
    last_row: u32,
    dispatches: u64,
    forwards: usize,
};

pub const StreamHook = struct {
    ctx: *anyopaque,
    on_token: *const fn (ctx: *anyopaque, tok: u32, is_eos: bool) anyerror!void,
    should_stop: ?*const fn (ctx: *anyopaque) bool = null,
};

pub const DecodeOptions = struct {
    stop_on_eos: bool = true,
    commit_all: bool = false,
    stream: ?StreamHook = null,
};

pub const DecodeResult = struct {
    count: usize,
    eos: bool,
    interrupted: bool = false,
    context_full: bool = false,
    last_row: u32,
    next_token: u32,
    batches: usize,
    proposed: usize,
    accepted: usize,
    dispatches: u64,
    draft_seconds: f64,
    verify_seconds: f64,
};

pub const DCutState = struct {
    pub const default_verify_ratios = [_]f64{ 0.368, 0.480, 0.590, 0.658, 0.761, 0.846, 0.930, 1.000 };
    initialized: bool = false,
    draft_ms: f64 = 4.2,
    verify_ms: [16]f64 = initVerifyMs(),
    gamma: f64 = 1.0,

    fn initVerifyMs() [16]f64 {
        var v: [16]f64 = undefined;
        const base_8: f64 = 45.6;
        for (0..16) |i| {
            const ratio = if (i < default_verify_ratios.len) default_verify_ratios[i] else (1.0 + 0.08 * @as(f64, @floatFromInt(i + 1 - default_verify_ratios.len)));
            v[i] = base_8 * ratio;
        }
        return v;
    }

    pub fn reset(self: *DCutState) void {
        self.* = .{};
    }

    pub fn selectVerifyLen(self: *const DCutState, confs: []const f32, max_len: u32, r_min_in: u32) u32 {
        if (max_len <= 1) return max_len;
        const r_min = @min(max_len, @max(@as(u32, 1), r_min_in));
        var surv: [16]f64 = undefined;
        var util: [16]f64 = undefined;
        surv[0] = 1.0;
        util[0] = 1.0;
        var k: usize = 1;
        while (k < max_len) : (k += 1) {
            const c_raw: f64 = if (k - 1 < confs.len) @floatCast(confs[k - 1]) else 0.0;
            const c = std.math.clamp(c_raw, 1e-6, 1.0);
            surv[k] = surv[k - 1] * std.math.pow(f64, c, self.gamma);
            util[k] = util[k - 1] + surv[k];
        }
        var best_r: u32 = max_len;
        var best_tp: f64 = -1.0;
        var r: u32 = r_min;
        while (r <= max_len) : (r += 1) {
            const idx = @as(usize, r - 1);
            const cost = @max(1e-3, self.draft_ms + self.verify_ms[idx]);
            const tp = util[idx] / cost;
            if (tp > best_tp) {
                best_tp = tp;
                best_r = r;
            }
        }
        return best_r;
    }

    pub fn observe(self: *DCutState, draft_ms_obs: f64, verify_r: u32, verify_ms_obs: f64, confs: []const f32, acc: u32, max_len: u32) void {
        if (verify_r == 0 or verify_r > 16) return;
        const r_idx = @as(usize, verify_r - 1);
        if (!self.initialized) {
            self.initialized = true;
            if (draft_ms_obs > 0.1 and draft_ms_obs < 1000.0) self.draft_ms = draft_ms_obs;
            const r_ref = if (r_idx < default_verify_ratios.len) default_verify_ratios[r_idx] else 1.0;
            if (verify_ms_obs > 0.5 and verify_ms_obs < 5000.0) {
                for (0..@min(@as(usize, max_len), 16)) |j| {
                    const ratio = if (j < default_verify_ratios.len) default_verify_ratios[j] else (1.0 + 0.08 * @as(f64, @floatFromInt(j + 1 - default_verify_ratios.len)));
                    self.verify_ms[j] = @max(0.5, verify_ms_obs * (ratio / r_ref));
                }
            }
        } else {
            const alpha: f64 = 0.25;
            if (draft_ms_obs > 0.1 and draft_ms_obs < 1000.0) {
                self.draft_ms = (1.0 - alpha) * self.draft_ms + alpha * draft_ms_obs;
            }
            if (verify_ms_obs > 0.5 and verify_ms_obs < 5000.0) {
                self.verify_ms[r_idx] = (1.0 - alpha) * self.verify_ms[r_idx] + alpha * verify_ms_obs;
                var j: usize = r_idx + 1;
                while (j < 16) : (j += 1) {
                    if (self.verify_ms[j] < self.verify_ms[j - 1] + 0.5) {
                        self.verify_ms[j] = self.verify_ms[j - 1] + 0.5;
                    }
                }
                j = r_idx;
                while (j > 0) : (j -= 1) {
                    if (self.verify_ms[j - 1] > self.verify_ms[j] - 0.5) {
                        self.verify_ms[j - 1] = @max(0.5, self.verify_ms[j] - 0.5);
                    }
                }
            }
        }
        if (verify_r > 1 and acc + 1 <= verify_r) {
            var s: f64 = 1.0;
            var pred: f64 = 0.0;
            var k: usize = 0;
            while (k + 1 < verify_r and k < confs.len) : (k += 1) {
                const c = std.math.clamp(@as(f64, @floatCast(confs[k])), 1e-6, 1.0);
                s *= std.math.pow(f64, c, self.gamma);
                pred += s;
            }
            const err = std.math.clamp((pred - @as(f64, @floatFromInt(acc))) / @as(f64, @floatFromInt(verify_r - 1)), -1.0, 1.0);
            self.gamma = std.math.clamp(self.gamma * @exp(0.08 * err), 0.25, 4.0);
        }
    }
};

pub const Draft = struct {
    gpa: std.mem.Allocator,
    cfg: Config,
    valid_vocab: u32,
    store: weights.Store,

    trace_indices: []u32,
    include_final_norm: bool,
    num_capture: u32,
    capture_cap: u32,
    commit_chunk: u32,
    ring_cap: u32,
    shards: u32 = 64,

    // Bound draft weights
    fc: weights.Weight,
    hidden_norm: weights.Weight,
    norm: weights.Weight,
    sel_hidden_proj: weights.Weight,
    sel_pred_cb: weights.Weight,
    sel_succ_cb: weights.Weight,
    layers: []LayerWeights,

    // Shared target weights (owned by target Store)
    embed: weights.Weight,
    embed_norm: ?weights.Weight,
    unembed: weights.Weight,

    // GPU buffers (owned by Draft; released before target.ctx.deinit)
    capture_buf: metal.Buffer,
    k_rings: []metal.Buffer,
    v_rings: []metal.Buffer,
    arena: metal.Buffer,
    rope_tables: metal.Buffer,
    top_part_vals: metal.Buffer,
    top_part_ids: metal.Buffer,
    top_vals: metal.Buffer,
    top_ids: metal.Buffer,
    out_tokens: metal.Buffer,
    splitk_buf: metal.Buffer,
    splitk_max_m: u32,

    // Scratch offsets inside `arena` (words)
    off_ctx_proj: u32,
    off_k_ctx: u32,
    off_v_ctx: u32,
    off_tokens: u32,
    off_hidden: u32,
    off_norm: u32,
    off_conv_dyn: u32,
    off_conv_x: u32,
    off_q: u32,
    off_k: u32,
    off_v: u32,
    off_att: u32,
    off_proj: u32,
    off_gate: u32,
    off_up: u32,
    off_hsel: u32,
    off_logits: u32,

    d2_spin: bool = false,
    dcut_enabled: bool = false,
    dcut_r_min: u32 = 1,
    dcut: DCutState = .{},
    ctx_pos: u32 = 0,

    pub fn init(gpa: std.mem.Allocator, target: *const exec.Engine, draft_dir: []const u8) !*Draft {
        var scratch_arena = std.heap.ArenaAllocator.init(gpa);
        defer scratch_arena.deinit();
        const sa = scratch_arena.allocator();

        const cfg_path = try std.fs.path.join(sa, &.{ draft_dir, "config.json" });
        const cfg_bytes = try loader.readFile(sa, cfg_path);
        const cfg = try parseConfig(gpa, sa, cfg_bytes);
        errdefer gpa.free(cfg.target_layer_ids);

        const valid_vocab = target.program.meta.vocab;
        if (valid_vocab > cfg.vocab_size) return error.DraftVocabMismatch;
        if (cfg.selector_softcap_mode != .after_correction and cfg.final_logit_softcapping > 0) {
            return error.UnsupportedSelectorSoftcapMode;
        }

        // Resolve target capture slots.
        const total_ids = cfg.target_layer_ids.len;
        if (total_ids == 0) return error.InvalidDraftCaptureLayers;
        const traces_len = target.program.traces.len;
        var include_final_norm = false;
        var trace_count = total_ids;
        if (traces_len > 0 and cfg.target_layer_ids[total_ids - 1] == traces_len - 1 and target.program.draft_final_norm != null) {
            include_final_norm = true;
            trace_count = total_ids - 1;
        } else if (std.mem.eql(u8, target.program.meta.model_type, "inkling2_loop") and
            traces_len > 1 and cfg.target_layer_ids[total_ids - 1] >= traces_len and
            cfg.target_layer_ids[total_ids - 1] == cfg.num_target_layers and
            target.program.draft_final_norm != null)
        {
            include_final_norm = true;
            trace_count = total_ids - 1;
        }
        const trace_indices = try gpa.alloc(u32, trace_count);
        errdefer gpa.free(trace_indices);
        const remapping = traces_len > 1 and std.mem.eql(u8, target.program.meta.model_type, "inkling2_loop") and
            cfg.target_layer_ids[total_ids - 1] >= traces_len;
        if (remapping) {
            if (std.c.getenv("NANO_DFLASH_REMAP")) |raw| {
                const spec = std.mem.span(raw);
                if (spec.len > 0) {
                    var it = std.mem.splitScalar(u8, spec, ',');
                    var idx: usize = 0;
                    while (it.next()) |tok| {
                        const trimmed = std.mem.trim(u8, tok, " \t");
                        if (trimmed.len == 0 or idx >= trace_count) return error.InvalidDraftCaptureLayers;
                        const tid = std.fmt.parseInt(u32, trimmed, 10) catch return error.InvalidDraftCaptureLayers;
                        if (tid >= traces_len) return error.InvalidDraftCaptureLayers;
                        trace_indices[idx] = tid;
                        idx += 1;
                    }
                    if (idx != trace_count) return error.InvalidDraftCaptureLayers;
                } else if (traces_len == inkling_plan.dflash_2p_traces and trace_count == 3) {
                    trace_indices[0] = inkling_plan.dflash_2p_remap_0;
                    trace_indices[1] = inkling_plan.dflash_2p_remap_1;
                    trace_indices[2] = inkling_plan.dflash_2p_remap_2;
                } else {
                    for (0..trace_count) |i| {
                        const tid = cfg.target_layer_ids[i];
                        trace_indices[i] = @min(@as(u32, @intCast(traces_len - 1)), @as(u32, @intCast((@as(u64, tid) * (traces_len - 1) + cfg.num_target_layers / 2) / cfg.num_target_layers)));
                    }
                }
            } else if (traces_len == inkling_plan.dflash_2p_traces and trace_count == 3) {
                trace_indices[0] = inkling_plan.dflash_2p_remap_0;
                trace_indices[1] = inkling_plan.dflash_2p_remap_1;
                trace_indices[2] = inkling_plan.dflash_2p_remap_2;
            } else {
                for (0..trace_count) |i| {
                    const tid = cfg.target_layer_ids[i];
                    trace_indices[i] = @min(@as(u32, @intCast(traces_len - 1)), @as(u32, @intCast((@as(u64, tid) * (traces_len - 1) + cfg.num_target_layers / 2) / cfg.num_target_layers)));
                }
            }
            for (trace_indices) |tid| {
                if (tid >= traces_len) return error.InvalidDraftCaptureLayers;
                if (target.program.traces[tid].view.width != cfg.hidden_size) return error.DraftHiddenSizeMismatch;
            }
        } else {
            for (0..trace_count) |i| {
                const tid = cfg.target_layer_ids[i];
                if (tid >= traces_len) return error.InvalidDraftCaptureLayers;
                if (target.program.traces[tid].view.width != cfg.hidden_size) return error.DraftHiddenSizeMismatch;
                trace_indices[i] = tid;
            }
        }
        if (include_final_norm) {
            if (target.program.draft_final_norm.?.view.width != cfg.hidden_size) return error.DraftHiddenSizeMismatch;
        }

        const embed = target.findWeight("model.llm.embed.weight") orelse target.findWeight("embed.weight") orelse return error.MissingWeight;
        const embed_norm = target.findWeight("model.llm.embed_norm.weight") orelse target.findWeight("embed_norm.weight");
        const unembed = target.findWeight("model.llm.unembed.weight") orelse target.findWeight("unembed.weight") orelse return error.MissingWeight;
        if (embed.tensor.shape.len != 2 or embed.tensor.shape[1] != cfg.hidden_size or embed.tensor.shape[0] < valid_vocab) return error.DraftWeightShapeMismatch;
        if (unembed.tensor.shape.len != 2 or unembed.tensor.shape[1] != cfg.hidden_size or unembed.tensor.shape[0] < valid_vocab) return error.DraftWeightShapeMismatch;
        if (unembed.quant != null or embed.quant != null) return error.UnsupportedQuantizedDraftTargetHead;
        if (embed_norm) |en| if (en.tensor.shape.len != 1 or en.tensor.shape[0] != cfg.hidden_size) return error.DraftWeightShapeMismatch;

        var store = try weights.Store.init(gpa, target.ctx, draft_dir, 0);
        errdefer store.deinit();

        const H = cfg.hidden_size;
        const I = cfg.intermediate_size;
        const q_dim = cfg.num_attention_heads * cfg.head_dim;
        const kv_dim = cfg.num_key_value_heads * cfg.head_dim;
        const groups = H / cfg.conv_group_size;
        const conv_proj_dim = 2 * cfg.conv_kernel_size * groups;
        const num_capture: u32 = @intCast(total_ids);

        const fc = try store.get("fc.weight", &.{ H, num_capture * H });
        const hidden_norm = try store.get("hidden_norm.weight", &.{H});
        const norm = try store.get("norm.weight", &.{H});
        const sel_hidden_proj = try store.get("candidate_selector.hidden_projection.weight", &.{ cfg.selector_rank, H });
        const sel_pred_cb = try store.get("candidate_selector.predecessor_codebook", &.{ cfg.vocab_size, cfg.selector_rank });
        const sel_succ_cb = try store.get("candidate_selector.successor_codebook", &.{ cfg.vocab_size, cfg.selector_rank });

        const layers = try gpa.alloc(LayerWeights, cfg.num_hidden_layers);
        errdefer gpa.free(layers);
        for (layers, 0..) |*lw, l| {
            lw.* = .{
                .input_layernorm = try store.get(try std.fmt.allocPrint(sa, "layers.{d}.input_layernorm.weight", .{l}), &.{H}),
                .post_attention_layernorm = try store.get(try std.fmt.allocPrint(sa, "layers.{d}.post_attention_layernorm.weight", .{l}), &.{H}),
                .attn_conv_base = try store.get(try std.fmt.allocPrint(sa, "layers.{d}.attention_conv.base_kernel", .{l}), &.{ 2, cfg.conv_kernel_size, H }),
                .attn_conv_proj = try store.get(try std.fmt.allocPrint(sa, "layers.{d}.attention_conv.kernel_projection.weight", .{l}), &.{ conv_proj_dim, H }),
                .mlp_conv_base = try store.get(try std.fmt.allocPrint(sa, "layers.{d}.mlp_conv.base_kernel", .{l}), &.{ 2, cfg.conv_kernel_size, H }),
                .mlp_conv_proj = try store.get(try std.fmt.allocPrint(sa, "layers.{d}.mlp_conv.kernel_projection.weight", .{l}), &.{ conv_proj_dim, H }),
                .q_proj = try store.get(try std.fmt.allocPrint(sa, "layers.{d}.self_attn.q_proj.weight", .{l}), &.{ q_dim, H }),
                .k_proj = try store.get(try std.fmt.allocPrint(sa, "layers.{d}.self_attn.k_proj.weight", .{l}), &.{ kv_dim, H }),
                .v_proj = try store.get(try std.fmt.allocPrint(sa, "layers.{d}.self_attn.v_proj.weight", .{l}), &.{ kv_dim, H }),
                .o_proj = try store.get(try std.fmt.allocPrint(sa, "layers.{d}.self_attn.o_proj.weight", .{l}), &.{ H, q_dim }),
                .q_norm = try store.get(try std.fmt.allocPrint(sa, "layers.{d}.self_attn.q_norm.weight", .{l}), &.{cfg.head_dim}),
                .k_norm = try store.get(try std.fmt.allocPrint(sa, "layers.{d}.self_attn.k_norm.weight", .{l}), &.{cfg.head_dim}),
                .gate_proj = try store.get(try std.fmt.allocPrint(sa, "layers.{d}.mlp.gate_proj.weight", .{l}), &.{ I, H }),
                .up_proj = try store.get(try std.fmt.allocPrint(sa, "layers.{d}.mlp.up_proj.weight", .{l}), &.{ I, H }),
                .down_proj = try store.get(try std.fmt.allocPrint(sa, "layers.{d}.mlp.down_proj.weight", .{l}), &.{ H, I }),
            };
        }

        const capture_cap = @max(cfg.block_size, @min(cfg.sliding_window - 1, target.context));
        const commit_chunk = @min(@as(u32, 512), capture_cap);
        const ring_cap = cfg.sliding_window;
        const B = cfg.block_size;
        const P = B - 1;
        const shards: u32 = @max(@as(u32, 64), (valid_vocab + 24 * 256 - 1) / (24 * 256));

        const capture_buf = try target.ctx.alloc(@as(usize, capture_cap) * num_capture * H);
        errdefer capture_buf.deinit();

        const k_rings = try gpa.alloc(metal.Buffer, cfg.num_hidden_layers);
        errdefer gpa.free(k_rings);
        const v_rings = try gpa.alloc(metal.Buffer, cfg.num_hidden_layers);
        errdefer gpa.free(v_rings);
        var allocated_rings: usize = 0;
        errdefer for (0..allocated_rings) |j| {
            k_rings[j].deinit();
            v_rings[j].deinit();
        };
        const ring_words = (@as(usize, ring_cap) * kv_dim + 1) / 2;
        for (0..cfg.num_hidden_layers) |j| {
            k_rings[j] = try target.ctx.alloc(ring_words);
            errdefer k_rings[j].deinit();
            v_rings[j] = try target.ctx.alloc(ring_words);
            allocated_rings += 1;
        }

        // Lay out unified scratch `arena`. Region A (`ctx_proj`, `k_ctx`, `v_ctx`) and
        // Region B (block execution) do not overlap temporally, so we can overlap or keep separate.
        const align64 = struct {
            fn pad(x: usize) usize {
                return (x + 63) / 64 * 64;
            }
        }.pad;

        const nextOff = struct {
            fn alloc(c: *usize, words: usize) u32 {
                const res: u32 = @intCast(c.*);
                c.* += (words + 63) / 64 * 64;
                return res;
            }
        }.alloc;

        var cur: usize = 0;
        const off_tokens = nextOff(&cur, B);
        // Commit region starts strictly after `off_tokens` so `commitContext` and CPU token setup
        // never alias inside the same command buffer.
        const off_ctx_proj: u32 = @intCast(cur);
        const off_k_ctx: u32 = @intCast(@as(usize, off_ctx_proj) + align64(@as(usize, commit_chunk) * H));
        const off_v_ctx: u32 = @intCast(@as(usize, off_k_ctx) + align64(@as(usize, commit_chunk) * kv_dim));
        const commit_words = @as(usize, off_v_ctx) + align64(@as(usize, commit_chunk) * kv_dim);

        const off_hidden = nextOff(&cur, B * H);
        const off_norm = nextOff(&cur, B * H);
        const off_conv_dyn = nextOff(&cur, B * conv_proj_dim);
        const off_conv_x = nextOff(&cur, B * H);
        const off_q = nextOff(&cur, B * q_dim);
        const off_k = nextOff(&cur, B * kv_dim);
        const off_v = nextOff(&cur, B * kv_dim);
        const off_att = nextOff(&cur, B * q_dim);
        const off_proj = nextOff(&cur, B * H);
        const off_gate = nextOff(&cur, B * I);
        const off_up = nextOff(&cur, B * I);
        const off_hsel = nextOff(&cur, P * cfg.selector_rank);
        const off_logits = nextOff(&cur, P * valid_vocab);
        const block_words = cur;

        const arena = try target.ctx.alloc(@max(commit_words, block_words));
        errdefer arena.deinit();

        const rope_tables = try target.ctx.alloc(cfg.head_dim / 2);
        errdefer rope_tables.deinit();
        {
            const inv = rope_tables.floats();
            const d_f64: f64 = @floatFromInt(cfg.head_dim);
            const theta_f64: f64 = @floatCast(cfg.rope_theta);
            for (0..cfg.head_dim / 2) |j| {
                const exp: f64 = @as(f64, @floatFromInt(2 * j)) / d_f64;
                inv[j] = @floatCast(1.0 / std.math.pow(f64, theta_f64, exp));
            }
        }

        const top_part_vals = try target.ctx.alloc(P * shards * (cfg.selector_top_k + 2));
        errdefer top_part_vals.deinit();
        const top_part_ids = try target.ctx.alloc(P * shards * cfg.selector_top_k);
        errdefer top_part_ids.deinit();
        const top_vals = try target.ctx.alloc(P * (cfg.selector_top_k + 1));
        errdefer top_vals.deinit();
        const top_ids = try target.ctx.alloc(P * cfg.selector_top_k);
        errdefer top_ids.deinit();
        const out_tokens = try target.ctx.alloc(2 * B);
        errdefer out_tokens.deinit();
        const splitk_max_m: u32 = @max(H, q_dim);
        const splitk_buf = try target.ctx.alloc(8 * @as(usize, B) * splitk_max_m);
        errdefer splitk_buf.deinit();

        var d2_spin = target.knobs.d2_spin;
        if (std.c.getenv("NANO_DFLASH_SPIN")) |raw| {
            const s = std.mem.span(raw);
            if (std.mem.eql(u8, s, "1") or std.mem.eql(u8, s, "true")) {
                d2_spin = true;
            } else if (std.mem.eql(u8, s, "0") or std.mem.eql(u8, s, "false")) {
                d2_spin = false;
            }
        }
        var dcut_enabled = target.knobs.dflash_dcut;
        if (std.c.getenv("NANO_DFLASH_DCUT")) |raw| {
            const s = std.mem.span(raw);
            if (std.mem.eql(u8, s, "1") or std.mem.eql(u8, s, "true")) {
                dcut_enabled = true;
            } else if (std.mem.eql(u8, s, "0") or std.mem.eql(u8, s, "false")) {
                dcut_enabled = false;
            }
        }
        var dcut_r_min: u32 = 1;
        if (std.c.getenv("NANO_DFLASH_DCUT_RMIN")) |raw| {
            const s = std.mem.trim(u8, std.mem.span(raw), " \t");
            if (s.len > 0) {
                dcut_r_min = std.fmt.parseInt(u32, s, 10) catch return error.InvalidDraftBlock;
            }
        }

        const self = try gpa.create(Draft);
        self.* = .{
            .gpa = gpa,
            .cfg = cfg,
            .valid_vocab = valid_vocab,
            .store = store,
            .trace_indices = trace_indices,
            .include_final_norm = include_final_norm,
            .num_capture = num_capture,
            .capture_cap = capture_cap,
            .commit_chunk = commit_chunk,
            .ring_cap = ring_cap,
            .shards = shards,
            .fc = fc,
            .hidden_norm = hidden_norm,
            .norm = norm,
            .sel_hidden_proj = sel_hidden_proj,
            .sel_pred_cb = sel_pred_cb,
            .sel_succ_cb = sel_succ_cb,
            .layers = layers,
            .embed = embed,
            .embed_norm = embed_norm,
            .unembed = unembed,
            .capture_buf = capture_buf,
            .k_rings = k_rings,
            .v_rings = v_rings,
            .arena = arena,
            .rope_tables = rope_tables,
            .top_part_vals = top_part_vals,
            .top_part_ids = top_part_ids,
            .top_vals = top_vals,
            .top_ids = top_ids,
            .out_tokens = out_tokens,
            .splitk_buf = splitk_buf,
            .splitk_max_m = splitk_max_m,
            .off_ctx_proj = off_ctx_proj,
            .off_k_ctx = off_k_ctx,
            .off_v_ctx = off_v_ctx,
            .off_tokens = off_tokens,
            .off_hidden = off_hidden,
            .off_norm = off_norm,
            .off_conv_dyn = off_conv_dyn,
            .off_conv_x = off_conv_x,
            .off_q = off_q,
            .off_k = off_k,
            .off_v = off_v,
            .off_att = off_att,
            .off_proj = off_proj,
            .off_gate = off_gate,
            .off_up = off_up,
            .off_hsel = off_hsel,
            .off_logits = off_logits,
            .d2_spin = d2_spin,
            .dcut_enabled = dcut_enabled,
            .dcut_r_min = dcut_r_min,
        };
        return self;
    }

    pub fn deinit(self: *Draft) void {
        self.splitk_buf.deinit();
        self.out_tokens.deinit();
        self.top_ids.deinit();
        self.top_vals.deinit();
        self.top_part_ids.deinit();
        self.top_part_vals.deinit();
        self.rope_tables.deinit();
        self.arena.deinit();
        for (self.k_rings, self.v_rings) |kr, vr| {
            kr.deinit();
            vr.deinit();
        }
        self.gpa.free(self.k_rings);
        self.gpa.free(self.v_rings);
        self.capture_buf.deinit();
        self.gpa.free(self.layers);
        self.store.deinit();
        self.gpa.free(self.trace_indices);
        self.gpa.free(self.cfg.target_layer_ids);
        self.gpa.destroy(self);
    }

    pub fn reset(self: *Draft) void {
        self.ctx_pos = 0;
        self.dcut.reset();
    }

    pub fn makeCapture(self: *const Draft, row0: u32, rows: u32, dst_row0: u32) exec.Capture {
        return .{
            .trace_indices = self.trace_indices,
            .include_final_norm = self.include_final_norm,
            .row0 = row0,
            .rows = rows,
            .dest = self.capture_buf,
            .width = self.cfg.hidden_size,
            .stride = self.num_capture * self.cfg.hidden_size,
            .dst_row0 = dst_row0,
        };
    }

    fn dispatchLinear(
        self: *const Draft,
        ctx: metal.Context,
        src_buf: metal.Buffer,
        src_off: u32,
        src_stride: u32,
        w: weights.Weight,
        dst_buf: metal.Buffer,
        dst_off: u32,
        dst_stride: u32,
        rows: u32,
        m: u32,
        n: u32,
        round_in: bool,
        round_out: bool,
        concurrent_first: bool,
    ) !void {
        if (n % 4 != 0) return error.InvalidDraftLinearWidth;
        var first_ctx = ctx;
        if (concurrent_first) first_ctx.nano_dispatch_flags = gpu.d2_concurrent;
        var barrier_ctx = ctx;
        barrier_ctx.nano_dispatch_flags = null;
        var p: gpu.Params = .{
            .w = w.tensor.offset,
            .dtype = @backingInt(w.tensor.dtype),
            .rows = rows,
            .row0 = 0,
            .x = src_off,
            .xs = src_stride,
            .y = dst_off,
            .ys = dst_stride,
            .n = n,
            .m = m,
            .flags = (if (round_in) gpu.flag_round_bf16 else 0) | (if (round_out) gpu.flag_round_out else 0),
        };
        if (rows <= 8) {
            const w_rows: u32 = @intCast(w.tensor.shape[0]);
            const tile16 = inkling_plan.bf16_mma_tile_n16;
            const tile32 = inkling_plan.bf16_mma_tile_n32;
            const pad16_m: u32 = ((m + tile16 - 1) / tile16) * tile16;
            const pad32_m: u32 = ((m + tile32 - 1) / tile32) * tile32;
            if (w.tensor.dtype == .bf16 and m <= 256 and n % 256 == 0 and w.tensor.offset % 16 == 0 and src_off % 4 == 0 and src_stride % 4 == 0) {
                try gpu.dispatch(first_ctx, "df_bf16_linear_splitk_rows_vec8", &.{ src_buf, w.buffer, dst_buf }, &p, @as(usize, m) * 256, 256);
            } else if (w.tensor.dtype == .bf16 and m <= 256 and n % 128 == 0 and w.tensor.offset % 8 == 0 and src_off % 4 == 0 and src_stride % 4 == 0) {
                try gpu.dispatch(first_ctx, "df_bf16_linear_splitk_rows", &.{ src_buf, w.buffer, dst_buf }, &p, @as(usize, m) * 256, 256);
            } else if (w.tensor.dtype == .bf16 and m >= 16384 and w_rows >= pad32_m and n % 64 == 0 and w.tensor.offset % 16 == 0 and src_off % 4 == 0 and src_stride % 4 == 0) {
                try gpu.dispatch(first_ctx, "df_bf16_mma_m8n32_direct", &.{ src_buf, w.buffer, dst_buf }, &p, ((@as(usize, m) + 31) / 32) * 128, 128);
            } else if (w.tensor.dtype == .bf16 and (m % 16 == 0 or w_rows >= pad16_m) and n % 64 == 0 and w.tensor.offset % 16 == 0 and src_off % 4 == 0 and src_stride % 4 == 0 and (m % 16 != 0 or (dst_off % 4 == 0 and dst_stride % 4 == 0))) {
                const n_tiles: u32 = (m + 15) / 16;
                const k_blocks: u32 = n / 64;
                const desired: u32 = if (n_tiles >= 256 or (n_tiles >= 120 and n <= 2304)) 1 else (384 + n_tiles - 1) / n_tiles;
                const max_by_k: u32 = @max(@as(u32, 1), k_blocks / 4);
                const can_split = (m % 16 == 0) and (dst_off % 4 == 0) and (dst_stride % 4 == 0) and (m <= self.splitk_max_m);
                const splits: u32 = if (can_split) @min(@as(u32, 8), @min(desired, max_by_k)) else 1;
                p.taps = splits;
                if (splits == 1) {
                    try gpu.dispatch(first_ctx, "df_bf16_mma_m8n16_direct", &.{ src_buf, w.buffer, dst_buf }, &p, @as(usize, n_tiles) * 64, 64);
                } else {
                    try gpu.dispatch(first_ctx, "df_bf16_mma_m8n16_direct", &.{ src_buf, w.buffer, self.splitk_buf }, &p, @as(usize, splits) * n_tiles * 64, 64);
                    try gpu.dispatch(barrier_ctx, "df_bf16_splitk_reduce", &.{ self.splitk_buf, dst_buf }, &p, @as(usize, rows) * (m / 4), 256);
                }
            } else if (w.tensor.dtype == .bf16 and n % 64 == 0 and w.tensor.offset % 16 == 0 and src_off % 4 == 0 and src_stride % 4 == 0) {
                try gpu.dispatch(first_ctx, "df_bf16_mma_m8n16", &.{ src_buf, w.buffer, dst_buf }, &p, ((@as(usize, m) + 15) / 16) * 64, 64);
            } else if (w.tensor.dtype == .bf16 and n % 128 == 0 and w.tensor.offset % 8 == 0 and src_off % 4 == 0 and src_stride % 4 == 0) {
                try gpu.dispatch(first_ctx, "df_bf16_linear_2xR_vec", &.{ src_buf, w.buffer, dst_buf }, &p, ((@as(usize, m) + 1) / 2) * 32, 128);
            } else {
                const threads = ((@as(usize, m) + 15) / 16) * 128;
                try gpu.dispatch(first_ctx, "df_bf16_linear", &.{ src_buf, w.buffer, dst_buf }, &p, threads, 128);
            }
            return;
        }
        const w_rows: u32 = @intCast(w.tensor.shape[0]);
        const tiles_n: u32 = (m + 63) / 64;
        const can_mma = w.tensor.dtype == .bf16 and w_rows >= tiles_n * 64 and n >= 32 and n % 32 == 0 and
            w.tensor.offset % 8 == 0 and src_off % 4 == 0 and src_stride % 4 == 0;
        if (can_mma) {
            const tiles_r = (@as(usize, rows) + 31) / 32;
            try gpu.dispatch(first_ctx, "df_bf16_matrix32", &.{ src_buf, w.buffer, dst_buf }, &p, @as(usize, tiles_n) * tiles_r * 128, 128);
            return;
        }
        const tiles_m = (@as(usize, m) + 31) / 32;
        const tiles_r = (@as(usize, rows) + 31) / 32;
        try gpu.dispatch(first_ctx, "df_bf16_linear_tile", &.{ src_buf, w.buffer, dst_buf }, &p, tiles_m * tiles_r * 128, 128);
    }

    fn dispatchNorm(ctx: metal.Context, buf: metal.Buffer, w: weights.Weight, x_off: u32, xs: u32, y_off: u32, ys: u32, rows: u32, heads: u32, n: u32, eps: f32, round_out: bool) !void {
        var p: gpu.Params = .{
            .w = w.tensor.offset,
            .dtype = @backingInt(w.tensor.dtype),
            .rows = rows,
            .row0 = 0,
            .x = x_off,
            .xs = xs,
            .y = y_off,
            .ys = ys,
            .heads = heads,
            .n = n,
            .eps = eps,
            .flags = gpu.flag_has_weight | (if (round_out) gpu.flag_round_out else 0),
        };
        const grp = gpu.normThreads(n);
        const fast = n % 4 == 0 and n <= 4 * grp and x_off % 4 == 0 and xs % 4 == 0 and y_off % 4 == 0 and ys % 4 == 0 and
            ((w.tensor.dtype == .bf16 and w.tensor.offset % 8 == 0) or (w.tensor.dtype == .f32 and w.tensor.offset % 16 == 0));
        try gpu.dispatch(ctx, if (fast) "mk_rmsnorm_vec" else "mk_rmsnorm", &.{ w.buffer, buf }, &p, @as(usize, rows) * heads * grp, grp);
    }

    fn fastWeightAligned(w: weights.Weight) bool {
        return switch (w.tensor.dtype) {
            .f32 => w.tensor.offset % 16 == 0,
            .f16, .bf16 => w.tensor.offset % 8 == 0,
            else => false,
        };
    }

    /// Project captured target features into draft KV rings for `rows` consecutive positions starting at `self.ctx_pos`.
    pub fn commitContext(self: *Draft, ctx: metal.Context, rows: u32, finish_after: bool) !void {
        if (rows == 0) return;
        var mctx = ctx;
        const H = self.cfg.hidden_size;
        const in_w = self.num_capture * H;
        const kv_dim = self.cfg.num_key_value_heads * self.cfg.head_dim;
        const Hkv = self.cfg.num_key_value_heads;
        const D = self.cfg.head_dim;
        const can_kv_fused = Hkv * D == kv_dim and D == 128 and self.off_k_ctx % 4 == 0 and kv_dim % 4 == 0 and fastWeightAligned(self.layers[0].k_norm);
        const can_v_vec4 = self.off_v_ctx % 4 == 0 and kv_dim % 4 == 0;

        var start: u32 = 0;
        while (start < rows) {
            const chunk = @min(self.commit_chunk, rows - start);
            const src_off = start * in_w;
            try self.dispatchLinear(mctx, self.capture_buf, src_off, in_w, self.fc, self.arena, self.off_ctx_proj, H, chunk, H, in_w, false, false, start == 0);
            try dispatchNorm(mctx, self.arena, self.hidden_norm, self.off_ctx_proj, H, self.off_ctx_proj, H, chunk, 1, H, self.cfg.rms_norm_eps, false);

            for (self.layers, self.k_rings, self.v_rings) |lw, k_ring, v_ring| {
                const can_multi_kv = chunk <= 8 and lw.k_proj.buffer.handle == lw.v_proj.buffer.handle and
                    lw.k_proj.tensor.dtype == .bf16 and lw.v_proj.tensor.dtype == .bf16 and
                    H % 64 == 0 and kv_dim % 16 == 0 and self.off_ctx_proj % 4 == 0 and H % 4 == 0 and
                    lw.k_proj.tensor.offset % 16 == 0 and lw.v_proj.tensor.offset % 16 == 0 and
                    self.off_k_ctx % 4 == 0 and self.off_v_ctx % 4 == 0 and kv_dim % 4 == 0;
                if (can_multi_kv) {
                    var mx: gpu.ParamsX = .{
                        .p = .{ .rows = chunk, .row0 = 0, .x = self.off_ctx_proj, .xs = H, .n = H },
                        .count = 2,
                    };
                    mx.mat[0] = .{ .w = lw.k_proj.tensor.offset, .m = kv_dim, .y = self.off_k_ctx, .ys = kv_dim };
                    mx.mat[1] = .{ .w = lw.v_proj.tensor.offset, .m = kv_dim, .y = self.off_v_ctx, .ys = kv_dim };
                    const total_tiles: usize = 2 * @as(usize, kv_dim / 16);
                    try gpu.dispatch(ctx, "df_multi_bf16_mma_m8n16_direct", &.{ self.arena, lw.k_proj.buffer, self.arena }, &mx, total_tiles * 64, 64);
                } else {
                    try self.dispatchLinear(ctx, self.arena, self.off_ctx_proj, H, lw.k_proj, self.arena, self.off_k_ctx, kv_dim, chunk, kv_dim, H, false, false, false);
                    try self.dispatchLinear(ctx, self.arena, self.off_ctx_proj, H, lw.v_proj, self.arena, self.off_v_ctx, kv_dim, chunk, kv_dim, H, false, false, false);
                }

                if (can_kv_fused and fastWeightAligned(lw.k_norm)) {
                    var kr_p: gpu.Params = .{
                        .w = lw.k_norm.tensor.offset,
                        .dtype = @backingInt(lw.k_norm.tensor.dtype),
                        .rows = chunk,
                        .row0 = 0,
                        .pos = self.ctx_pos + start,
                        .capacity = self.ring_cap,
                        .x = self.off_k_ctx,
                        .xs = kv_dim,
                        .heads = Hkv,
                        .dim = D,
                        .rot = D,
                        .z = 0,
                        .eps = self.cfg.rms_norm_eps,
                    };
                    const total_heads = @as(usize, chunk) * Hkv;
                    try gpu.dispatch(ctx, "df_rmsnorm_rope_ring_store", &.{ lw.k_norm.buffer, self.arena, self.rope_tables, k_ring }, &kr_p, ((total_heads + 3) / 4) * 128, 128);
                } else {
                    try dispatchNorm(ctx, self.arena, lw.k_norm, self.off_k_ctx, kv_dim, self.off_k_ctx, kv_dim, chunk, Hkv, D, self.cfg.rms_norm_eps, false);

                    var rope_p: gpu.Params = .{
                        .rows = chunk,
                        .row0 = 0,
                        .pos = self.ctx_pos + start,
                        .x = self.off_k_ctx,
                        .xs = kv_dim,
                        .heads = Hkv,
                        .dim = D,
                        .rot = D,
                        .z = 0,
                    };
                    try gpu.dispatch(ctx, "mk_rope", &.{ self.arena, self.rope_tables }, &rope_p, @as(usize, chunk) * Hkv * (D / 2), 256);

                    var sk_p: gpu.Params = .{ .rows = chunk, .row0 = 0, .pos = self.ctx_pos + start, .capacity = self.ring_cap, .x = self.off_k_ctx, .xs = kv_dim, .n = kv_dim };
                    try gpu.dispatch(ctx, if (can_kv_fused or (self.off_k_ctx % 4 == 0 and kv_dim % 4 == 0)) "df_kv_ring_store_vec4" else "df_kv_ring_store", &.{ self.arena, k_ring }, &sk_p, @as(usize, chunk) * (if (self.off_k_ctx % 4 == 0 and kv_dim % 4 == 0) kv_dim / 4 else kv_dim), 256);
                }

                // V store reads off_v_ctx and writes v_ring cleanly in parallel with K norm+rope+ring_store.
                mctx.nano_dispatch_flags = gpu.d2_concurrent;
                defer mctx.nano_dispatch_flags = null;
                var sv_p: gpu.Params = .{ .rows = chunk, .row0 = 0, .pos = self.ctx_pos + start, .capacity = self.ring_cap, .x = self.off_v_ctx, .xs = kv_dim, .n = kv_dim };
                if (can_v_vec4) {
                    try gpu.dispatch(mctx, "df_kv_ring_store_vec4", &.{ self.arena, v_ring }, &sv_p, @as(usize, chunk) * (kv_dim / 4), 256);
                } else {
                    try gpu.dispatch(mctx, "df_kv_ring_store", &.{ self.arena, v_ring }, &sv_p, @as(usize, chunk) * kv_dim, 256);
                }
            }
            start += chunk;
        }
        self.ctx_pos += rows;
        if (finish_after) try gpu.finishWait(mctx, self.d2_spin, null);
    }

    /// Propose `verify_size - 1` draft tokens after `anchor`; writes `[anchor, d0, ...]` to `out_block[0..verify_size]`,
    /// and optionally writes post-selector top-1 confidence for each proposed token to `out_confs[0..verify_size-1]`.
    pub fn propose(self: *Draft, ctx: metal.Context, anchor: u32, verify_size: u32, fresh_cmd: bool, out_block: []u32, out_confs: ?[]f32) !void {
        if (verify_size < 2 or verify_size > self.cfg.block_size or out_block.len < verify_size) return error.InvalidDraftBlock;
        var mctx = ctx;
        const B = verify_size;
        const P = B - 1;
        const H = self.cfg.hidden_size;
        const I = self.cfg.intermediate_size;
        const Hq = self.cfg.num_attention_heads;
        const Hkv = self.cfg.num_key_value_heads;
        const D = self.cfg.head_dim;
        const q_dim = Hq * D;
        const kv_dim = Hkv * D;
        const groups = H / self.cfg.conv_group_size;
        const conv_proj_dim = 2 * self.cfg.conv_kernel_size * groups;
        const vec4_conv_ok = (H % 4 == 0) and (self.cfg.conv_group_size % 4 == 0) and
            (self.off_norm % 4 == 0) and (self.off_conv_x % 4 == 0) and (self.off_proj % 4 == 0) and (self.off_hidden % 4 == 0);
        const can_gqa4 = (Hq == inkling_plan.dflash_gqa4_ratio * Hkv) and (D == inkling_plan.dflash_head_dim) and (kv_dim % 4 == 0) and (q_dim % 4 == 0) and
            (self.off_q % 4 == 0) and (self.off_k % 4 == 0) and (self.off_v % 4 == 0) and (self.off_att % 4 == 0);

        const tok_words = self.arena.structs(u32);
        tok_words[self.off_tokens] = anchor;
        for (1..B) |i| tok_words[self.off_tokens + i] = self.cfg.mask_token_id;

        if (fresh_cmd) mctx.nano_dispatch_flags = gpu.d2_concurrent;
        var emb_p: gpu.Params = .{
            .w = self.embed.tensor.offset,
            .dtype = @backingInt(self.embed.tensor.dtype),
            .rows = B,
            .row0 = 0,
            .x = self.off_tokens,
            .xs = 1,
            .y = self.off_hidden,
            .ys = H,
            .n = H,
        };
        {
            defer mctx.nano_dispatch_flags = null;
            try gpu.dispatch(mctx, "mk_embed", &.{ self.embed.buffer, self.embed.buffer, self.embed.buffer, self.arena }, &emb_p, @as(usize, B) * H, 256);
        }
        if (self.embed_norm) |en| {
            try dispatchNorm(ctx, self.arena, en, self.off_hidden, H, self.off_hidden, H, B, 1, H, self.cfg.rms_norm_eps, true);
        }

        for (self.layers, self.k_rings, self.v_rings) |lw, k_ring, v_ring| {
            // Attention block
            try dispatchNorm(ctx, self.arena, lw.input_layernorm, self.off_hidden, H, self.off_norm, H, B, 1, H, self.cfg.rms_norm_eps, false);
            try self.dispatchLinear(ctx, self.arena, self.off_norm, H, lw.attn_conv_proj, self.arena, self.off_conv_dyn, conv_proj_dim, B, conv_proj_dim, H, false, false, false);
            var ac0: gpu.Params = .{
                .rows = B,
                .row0 = 0,
                .kind = 0,
                .heads = groups,
                .dim = self.cfg.conv_group_size,
                .taps = self.cfg.conv_kernel_size,
                .n = H,
                .x = self.off_norm,
                .xs = H,
                .y = self.off_conv_x,
                .ys = H,
                .z = self.off_conv_dyn,
                .zs = conv_proj_dim,
                .w = lw.attn_conv_base.tensor.offset,
                .dtype = @backingInt(lw.attn_conv_base.tensor.dtype),
            };
            if (vec4_conv_ok and fastWeightAligned(lw.attn_conv_base)) {
                try gpu.dispatch(ctx, "df_grouped_dyn_conv_vec4", &.{ self.arena, lw.attn_conv_base.buffer }, &ac0, @as(usize, B) * (H / 4), 256);
            } else {
                try gpu.dispatch(ctx, "df_grouped_dyn_conv", &.{ self.arena, lw.attn_conv_base.buffer }, &ac0, @as(usize, B) * H, 256);
            }

            const can_multi_qkv = B <= 8 and
                lw.q_proj.buffer.handle == lw.k_proj.buffer.handle and
                lw.q_proj.buffer.handle == lw.v_proj.buffer.handle and
                lw.q_proj.tensor.dtype == .bf16 and lw.k_proj.tensor.dtype == .bf16 and lw.v_proj.tensor.dtype == .bf16 and
                H % 64 == 0 and q_dim % 16 == 0 and kv_dim % 16 == 0 and self.off_conv_x % 4 == 0 and H % 4 == 0 and
                lw.q_proj.tensor.offset % 16 == 0 and lw.k_proj.tensor.offset % 16 == 0 and lw.v_proj.tensor.offset % 16 == 0 and
                self.off_q % 4 == 0 and self.off_k % 4 == 0 and self.off_v % 4 == 0 and q_dim % 4 == 0 and kv_dim % 4 == 0;
            if (can_multi_qkv) {
                var mx: gpu.ParamsX = .{
                    .p = .{ .rows = B, .row0 = 0, .x = self.off_conv_x, .xs = H, .n = H },
                    .count = 3,
                };
                mx.mat[0] = .{ .w = lw.q_proj.tensor.offset, .m = q_dim, .y = self.off_q, .ys = q_dim };
                mx.mat[1] = .{ .w = lw.k_proj.tensor.offset, .m = kv_dim, .y = self.off_k, .ys = kv_dim };
                mx.mat[2] = .{ .w = lw.v_proj.tensor.offset, .m = kv_dim, .y = self.off_v, .ys = kv_dim };
                const total_tiles: usize = @as(usize, q_dim / 16) + 2 * @as(usize, kv_dim / 16);
                try gpu.dispatch(ctx, "df_multi_bf16_mma_m8n16_direct", &.{ self.arena, lw.q_proj.buffer, self.arena }, &mx, total_tiles * 64, 64);
            } else {
                try self.dispatchLinear(ctx, self.arena, self.off_conv_x, H, lw.q_proj, self.arena, self.off_q, q_dim, B, q_dim, H, false, false, false);
                try self.dispatchLinear(ctx, self.arena, self.off_conv_x, H, lw.k_proj, self.arena, self.off_k, kv_dim, B, kv_dim, H, false, false, false);
                try self.dispatchLinear(ctx, self.arena, self.off_conv_x, H, lw.v_proj, self.arena, self.off_v, kv_dim, B, kv_dim, H, false, false, false);
            }

            if (D == 128 and q_dim % 4 == 0 and kv_dim % 4 == 0 and self.off_q % 4 == 0 and self.off_k % 4 == 0 and fastWeightAligned(lw.q_norm) and fastWeightAligned(lw.k_norm)) {
                var rnq: gpu.Params = .{
                    .w = lw.q_norm.tensor.offset,
                    .dtype = @backingInt(lw.q_norm.tensor.dtype),
                    .rows = B,
                    .row0 = 0,
                    .pos = self.ctx_pos,
                    .x = self.off_q,
                    .xs = q_dim,
                    .heads = Hq,
                    .dim = D,
                    .rot = D,
                    .z = 0,
                    .eps = self.cfg.rms_norm_eps,
                };
                try gpu.dispatch(ctx, "df_rmsnorm_rope", &.{ lw.q_norm.buffer, self.arena, self.rope_tables }, &rnq, ((@as(usize, B) * Hq + 3) / 4) * 128, 128);

                mctx.nano_dispatch_flags = gpu.d2_concurrent;
                defer mctx.nano_dispatch_flags = null;
                var rnk: gpu.Params = .{
                    .w = lw.k_norm.tensor.offset,
                    .dtype = @backingInt(lw.k_norm.tensor.dtype),
                    .rows = B,
                    .row0 = 0,
                    .pos = self.ctx_pos,
                    .x = self.off_k,
                    .xs = kv_dim,
                    .heads = Hkv,
                    .dim = D,
                    .rot = D,
                    .z = 0,
                    .eps = self.cfg.rms_norm_eps,
                };
                try gpu.dispatch(mctx, "df_rmsnorm_rope", &.{ lw.k_norm.buffer, self.arena, self.rope_tables }, &rnk, ((@as(usize, B) * Hkv + 3) / 4) * 128, 128);
            } else {
                try dispatchNorm(mctx, self.arena, lw.q_norm, self.off_q, q_dim, self.off_q, q_dim, B, Hq, D, self.cfg.rms_norm_eps, false);
                try dispatchNorm(mctx, self.arena, lw.k_norm, self.off_k, kv_dim, self.off_k, kv_dim, B, Hkv, D, self.cfg.rms_norm_eps, false);

                var rq: gpu.Params = .{ .rows = B, .pos = self.ctx_pos, .x = self.off_q, .xs = q_dim, .heads = Hq, .dim = D, .rot = D, .z = 0 };
                try gpu.dispatch(mctx, "mk_rope", &.{ self.arena, self.rope_tables }, &rq, @as(usize, B) * Hq * (D / 2), 256);
                mctx.nano_dispatch_flags = gpu.d2_concurrent;
                defer mctx.nano_dispatch_flags = null;
                var rk: gpu.Params = .{ .rows = B, .pos = self.ctx_pos, .x = self.off_k, .xs = kv_dim, .heads = Hkv, .dim = D, .rot = D, .z = 0 };
                try gpu.dispatch(mctx, "mk_rope", &.{ self.arena, self.rope_tables }, &rk, @as(usize, B) * Hkv * (D / 2), 256);
            }

            var att_p: gpu.Params = .{
                .rows = B,
                .pos = self.ctx_pos,
                .heads = Hq,
                .kv_heads = Hkv,
                .dim = D,
                .window = self.cfg.sliding_window,
                .capacity = self.ring_cap,
                .scale = 1.0 / @sqrt(@as(f32, @floatFromInt(D))),
                .x = self.off_q,
                .xs = q_dim,
                .y = self.off_k,
                .ys = kv_dim,
                .z = self.off_v,
                .zs = kv_dim,
                .every = self.off_att,
                .pad_k = q_dim,
            };
            if (can_gqa4) {
                try gpu.dispatch(ctx, "df_attention_gqa4", &.{ self.arena, k_ring, v_ring }, &att_p, @as(usize, B) * Hkv * 256, 256);
            } else {
                try gpu.dispatch(ctx, "df_attention", &.{ self.arena, k_ring, v_ring }, &att_p, @as(usize, B) * Hq * 256, 256);
            }

            try self.dispatchLinear(ctx, self.arena, self.off_att, q_dim, lw.o_proj, self.arena, self.off_proj, H, B, H, q_dim, false, false, false);
            var ac1: gpu.Params = .{
                .rows = B,
                .row0 = 0,
                .kind = 1,
                .heads = groups,
                .dim = self.cfg.conv_group_size,
                .taps = self.cfg.conv_kernel_size,
                .n = H,
                .x = self.off_proj,
                .xs = H,
                .y = self.off_hidden,
                .ys = H,
                .z = self.off_conv_dyn,
                .zs = conv_proj_dim,
                .pad_r = self.off_hidden,
                .pad_k = H,
                .w = lw.attn_conv_base.tensor.offset,
                .dtype = @backingInt(lw.attn_conv_base.tensor.dtype),
            };
            if (vec4_conv_ok and fastWeightAligned(lw.attn_conv_base)) {
                try gpu.dispatch(ctx, "df_grouped_dyn_conv_vec4", &.{ self.arena, lw.attn_conv_base.buffer }, &ac1, @as(usize, B) * (H / 4), 256);
            } else {
                try gpu.dispatch(ctx, "df_grouped_dyn_conv", &.{ self.arena, lw.attn_conv_base.buffer }, &ac1, @as(usize, B) * H, 256);
            }

            // MLP block
            try dispatchNorm(ctx, self.arena, lw.post_attention_layernorm, self.off_hidden, H, self.off_norm, H, B, 1, H, self.cfg.rms_norm_eps, false);
            try self.dispatchLinear(ctx, self.arena, self.off_norm, H, lw.mlp_conv_proj, self.arena, self.off_conv_dyn, conv_proj_dim, B, conv_proj_dim, H, false, false, false);
            var mc0: gpu.Params = .{
                .rows = B,
                .row0 = 0,
                .kind = 0,
                .heads = groups,
                .dim = self.cfg.conv_group_size,
                .taps = self.cfg.conv_kernel_size,
                .n = H,
                .x = self.off_norm,
                .xs = H,
                .y = self.off_conv_x,
                .ys = H,
                .z = self.off_conv_dyn,
                .zs = conv_proj_dim,
                .w = lw.mlp_conv_base.tensor.offset,
                .dtype = @backingInt(lw.mlp_conv_base.tensor.dtype),
            };
            if (vec4_conv_ok and fastWeightAligned(lw.mlp_conv_base)) {
                try gpu.dispatch(ctx, "df_grouped_dyn_conv_vec4", &.{ self.arena, lw.mlp_conv_base.buffer }, &mc0, @as(usize, B) * (H / 4), 256);
            } else {
                try gpu.dispatch(ctx, "df_grouped_dyn_conv", &.{ self.arena, lw.mlp_conv_base.buffer }, &mc0, @as(usize, B) * H, 256);
            }

            const gate_rows: u32 = @intCast(lw.gate_proj.tensor.shape[0]);
            const up_rows: u32 = @intCast(lw.up_proj.tensor.shape[0]);
            const pad16_I: u32 = ((I + 15) / 16) * 16;
            const can_fused_glu = B <= 8 and
                lw.gate_proj.buffer.handle == lw.up_proj.buffer.handle and
                lw.gate_proj.tensor.dtype == .bf16 and lw.up_proj.tensor.dtype == .bf16 and
                (I % 16 == 0 or (gate_rows >= pad16_I and up_rows >= pad16_I)) and
                H % 64 == 0 and lw.gate_proj.tensor.offset % 16 == 0 and lw.up_proj.tensor.offset % 16 == 0 and
                self.off_conv_x % 4 == 0 and H % 4 == 0;
            if (can_fused_glu) {
                var gu_p: gpu.Params = .{
                    .w = lw.gate_proj.tensor.offset,
                    .s = lw.up_proj.tensor.offset,
                    .dtype = @backingInt(lw.gate_proj.tensor.dtype),
                    .rows = B,
                    .row0 = 0,
                    .x = self.off_conv_x,
                    .xs = H,
                    .y = self.off_gate,
                    .ys = I,
                    .n = H,
                    .m = I,
                };
                try gpu.dispatch(ctx, "df_bf16_gate_up_swiglu_mma", &.{ self.arena, lw.gate_proj.buffer, self.arena }, &gu_p, ((@as(usize, I) + 15) / 16) * 64, 64);
            } else {
                try self.dispatchLinear(ctx, self.arena, self.off_conv_x, H, lw.gate_proj, self.arena, self.off_gate, I, B, I, H, false, false, false);
                try self.dispatchLinear(ctx, self.arena, self.off_conv_x, H, lw.up_proj, self.arena, self.off_up, I, B, I, H, false, false, false);
                var sg_p: gpu.Params = .{ .rows = B, .n = I, .x = self.off_gate, .xs = I, .y = self.off_up, .ys = I, .z = self.off_gate, .zs = I };
                try gpu.dispatch(ctx, "df_swiglu_mul", &.{self.arena}, &sg_p, @as(usize, B) * I, 256);
            }
            try self.dispatchLinear(ctx, self.arena, self.off_gate, I, lw.down_proj, self.arena, self.off_proj, H, B, H, I, false, false, false);

            var mc1: gpu.Params = .{
                .rows = B,
                .row0 = 0,
                .kind = 1,
                .heads = groups,
                .dim = self.cfg.conv_group_size,
                .taps = self.cfg.conv_kernel_size,
                .n = H,
                .x = self.off_proj,
                .xs = H,
                .y = self.off_hidden,
                .ys = H,
                .z = self.off_conv_dyn,
                .zs = conv_proj_dim,
                .pad_r = self.off_hidden,
                .pad_k = H,
                .w = lw.mlp_conv_base.tensor.offset,
                .dtype = @backingInt(lw.mlp_conv_base.tensor.dtype),
            };
            if (vec4_conv_ok and fastWeightAligned(lw.mlp_conv_base)) {
                try gpu.dispatch(ctx, "df_grouped_dyn_conv_vec4", &.{ self.arena, lw.mlp_conv_base.buffer }, &mc1, @as(usize, B) * (H / 4), 256);
            } else {
                try gpu.dispatch(ctx, "df_grouped_dyn_conv", &.{ self.arena, lw.mlp_conv_base.buffer }, &mc1, @as(usize, B) * H, 256);
            }
        }

        // Proposal rows [1 .. B-1] (logits_start = 1).
        try dispatchNorm(ctx, self.arena, self.norm, self.off_hidden + H, H, self.off_norm, H, P, 1, H, self.cfg.rms_norm_eps, false);
        try self.dispatchLinear(ctx, self.arena, self.off_norm, H, self.unembed, self.arena, self.off_logits, self.valid_vocab, P, self.valid_vocab, H, false, false, false);
        // Disjoint output (`off_hsel` vs `off_logits`) reading same immutable `off_norm`.
        try self.dispatchLinear(ctx, self.arena, self.off_norm, H, self.sel_hidden_proj, self.arena, self.off_hsel, self.cfg.selector_rank, P, self.cfg.selector_rank, H, false, false, true);

        const softcap: f32 = if (self.cfg.selector_softcap_mode == .after_correction) self.cfg.final_logit_softcapping else 0.0;
        const need_conf: u32 = if (out_confs != null) 1 else 0;
        var t1_p: gpu.Params = .{ .rows = P, .row0 = 0, .x = self.off_logits, .xs = self.valid_vocab, .m = self.valid_vocab, .heads = self.shards, .n = self.cfg.selector_top_k, .kind = need_conf, .eps = softcap };
        try gpu.dispatch(ctx, "df_top16_part", &.{ self.arena, self.top_part_vals, self.top_part_ids }, &t1_p, @as(usize, P) * self.shards * 256, 256);

        var t2_p: gpu.Params = .{ .rows = P, .heads = self.shards, .n = self.cfg.selector_top_k, .kind = need_conf, .eps = softcap };
        try gpu.dispatch(ctx, "df_top16_merge", &.{ self.top_part_vals, self.top_part_ids, self.top_vals, self.top_ids }, &t2_p, @as(usize, P) * 256, 256);

        var sp_p: gpu.Params = .{
            .rows = P,
            .n = self.cfg.selector_rank,
            .m = self.cfg.selector_top_k,
            .pos = anchor,
            .x = self.off_hsel,
            .xs = self.cfg.selector_rank,
            .w = self.sel_pred_cb.tensor.offset,
            .s = self.sel_succ_cb.tensor.offset,
            .dtype = @backingInt(self.sel_pred_cb.tensor.dtype),
            .kind = need_conf,
            .eps = softcap,
        };
        try gpu.dispatch(ctx, "df_select_path", &.{ self.arena, self.sel_pred_cb.buffer, self.sel_succ_cb.buffer, self.top_vals, self.top_ids, self.out_tokens }, &sp_p, 512, 512);

        try gpu.finishWait(ctx, self.d2_spin, null);
        const out_u32 = self.out_tokens.structs(u32);
        for (0..B) |i| {
            const tok = out_u32[i];
            if (tok >= self.valid_vocab) return error.InvalidDraftToken;
            out_block[i] = tok;
        }
        if (out_confs) |confs| {
            if (confs.len < P) return error.InvalidDraftBlock;
            for (0..P) |i| confs[i] = @bitCast(out_u32[B + i]);
        }
    }

    pub fn prefill(self: *Draft, target: *exec.Engine, prompt: []const u32) !PrefillResult {
        return self.prefillWithVision(target, prompt, null);
    }

    pub fn prefillWithVision(self: *Draft, target: *exec.Engine, prompt: []const u32, vis: ?*exec.VisionStream) !PrefillResult {
        if (prompt.len == 0) return error.EmptyPrompt;
        const pos0: u64 = target.position;
        const total_end: u64 = pos0 + prompt.len;
        const window_len: u64 = self.cfg.sliding_window - 1;
        const window_start: u64 = total_end -| window_len;

        var dispatches: u64 = 0;
        var forwards: usize = 0;
        var last_row: u32 = 0;
        var i: usize = 0;
        while (i < prompt.len) {
            const n = @min(target.batch, prompt.len - i);
            const chunk = prompt[i..][0..n];
            const chunk_start_pos: u64 = pos0 + i;
            const chunk_end_pos: u64 = chunk_start_pos + n;
            const last = i + n == prompt.len;
            const req_out: u32 = if (last) @intCast(n - 1) else @intCast(n);

            var ov_val: ?exec.EmbedOverride = if (vis) |vs| try vs.prepareChunk(chunk) else null;
            target.embed_override = if (ov_val) |*ov| ov else null;
            defer target.embed_override = null;

            if (chunk_end_pos <= window_start) {
                try target.forward(chunk, req_out);
                self.ctx_pos = @intCast(chunk_end_pos);
            } else {
                const cap_start: u64 = @max(chunk_start_pos, window_start);
                const row0: u32 = @intCast(cap_start - chunk_start_pos);
                const rows: u32 = @intCast(chunk_end_pos - cap_start);
                var cap = self.makeCapture(row0, rows, 0);
                target.capture = &cap;
                defer target.capture = null;
                try target.forward(chunk, req_out);
                self.ctx_pos = @intCast(cap_start);
                try self.commitContext(target.ctx, rows, true);
            }
            dispatches += target.last.dispatches;
            forwards += 1;
            last_row = @intCast(n - 1);
            i += n;
        }
        if (vis) |vs| try vs.finish();
        return .{ .last_row = last_row, .dispatches = dispatches, .forwards = forwards };
    }

    fn isEos(target: *const exec.Engine, tok: u32) bool {
        return std.mem.indexOfScalar(u32, target.program.meta.eos, tok) != null;
    }

    pub fn decodeGreedy(
        self: *Draft,
        target: *exec.Engine,
        first_token: u32,
        out_tokens: []u32,
        opts: DecodeOptions,
    ) !DecodeResult {
        const limit = out_tokens.len;
        if (limit == 0) {
            return .{ .count = 0, .eos = false, .last_row = target.out_end -| 1, .next_token = first_token, .batches = 0, .proposed = 0, .accepted = 0, .dispatches = 0, .draft_seconds = 0, .verify_seconds = 0 };
        }
        if (target.batch < self.cfg.block_size and !opts.commit_all) {
            // Fall through safely; `verify_size` clamps by `target.batch` below.
        }
        var block_buf: [16]u32 = undefined;
        var conf_buf: [16]f32 = undefined;
        if (self.cfg.block_size > block_buf.len) return error.InvalidDraftBlock;
        const block = block_buf[0..self.cfg.block_size];
        var verify_cap: u32 = self.cfg.block_size;
        if (std.c.getenv("NANO_DFLASH_VERIFY_MAX")) |raw| {
            const s = std.mem.trim(u8, std.mem.span(raw), " \t");
            if (s.len > 0) {
                const v = std.fmt.parseInt(u32, s, 10) catch return error.InvalidDraftBlock;
                if (v < 1 or v > self.cfg.block_size) return error.InvalidDraftBlock;
                verify_cap = v;
            }
        }

        var anchor = first_token;
        var count: usize = 0;
        var last_row: u32 = target.out_end -| 1;
        var next_token: u32 = first_token;
        var pending_commit: u32 = 0;
        var batches: usize = 0;
        var proposed: usize = 0;
        var accepted_total: usize = 0;
        var dispatches: u64 = 0;
        var draft_seconds: f64 = 0;
        var verify_seconds: f64 = 0;
        var eos = false;
        var interrupted = false;
        var context_full = false;

        while (count < limit) {
            if (opts.stream) |sh| {
                if (sh.should_stop) |stop_fn| if (stop_fn(sh.ctx)) {
                    if (opts.commit_all and pending_commit > 0) {
                        const td_int = try io.now();
                        try self.commitContext(target.ctx, pending_commit, true);
                        pending_commit = 0;
                        draft_seconds += (try io.now()) - td_int;
                    }
                    interrupted = true;
                    break;
                };
            }
            const remaining = limit - count;
            const avail_ctx = target.context -| target.position;
            if ((opts.stop_on_eos and isEos(target, anchor)) or remaining == 1 or avail_ctx == 0) {
                const anchor_eos = opts.stop_on_eos and isEos(target, anchor);
                out_tokens[count] = anchor;
                count += 1;
                eos = anchor_eos;
                if (opts.stream) |sh| try sh.on_token(sh.ctx, anchor, anchor_eos);
                if (avail_ctx == 0) {
                    if (!eos) context_full = true;
                    if (opts.commit_all and pending_commit > 0) {
                        const td_ctx = try io.now();
                        try self.commitContext(target.ctx, pending_commit, true);
                        pending_commit = 0;
                        draft_seconds += (try io.now()) - td_ctx;
                    }
                } else if (opts.commit_all) {
                    const tv0 = try io.now();
                    if (pending_commit > 0) {
                        try self.commitContext(target.ctx, pending_commit, true);
                        pending_commit = 0;
                    }
                    var cap = self.makeCapture(0, 1, 0);
                    target.capture = &cap;
                    defer target.capture = null;
                    try target.forward(&.{anchor}, 0);
                    dispatches += target.last.dispatches;
                    verify_seconds += (try io.now()) - tv0;
                    last_row = 0;
                    next_token = try target.next(0);
                    const td0 = try io.now();
                    try self.commitContext(target.ctx, 1, true);
                    draft_seconds += (try io.now()) - td0;
                }
                break;
            }

            const max_block = @min(verify_cap, @min(self.cfg.block_size, target.batch));
            const propose_size: u32 = @intCast(@min(max_block, @min(remaining, avail_ctx)));
            var verify_size: u32 = propose_size;

            const td0 = try io.now();
            if (propose_size > 1) {
                const fresh_cmd = (pending_commit == 0);
                if (pending_commit > 0) {
                    try self.commitContext(target.ctx, pending_commit, false);
                    pending_commit = 0;
                }
                const conf_slice: ?[]f32 = if (self.dcut_enabled) conf_buf[0 .. propose_size - 1] else null;
                try self.propose(target.ctx, anchor, propose_size, fresh_cmd, block[0..propose_size], conf_slice);
                if (self.dcut_enabled) {
                    verify_size = self.dcut.selectVerifyLen(conf_buf[0 .. propose_size - 1], propose_size, self.dcut_r_min);
                }
                proposed += verify_size - 1;
            } else {
                if (pending_commit > 0) {
                    try self.commitContext(target.ctx, pending_commit, true);
                    pending_commit = 0;
                }
                block[0] = anchor;
            }
            const draft_dt = (try io.now()) - td0;
            draft_seconds += draft_dt;
            batches += 1;

            const tv0 = try io.now();
            const pos_before = target.position;
            var cap = self.makeCapture(0, verify_size, 0);
            target.capture = &cap;
            defer target.capture = null;
            try target.forward(block[0..verify_size], 0);
            dispatches += target.last.dispatches;
            const verify_dt = (try io.now()) - tv0;
            verify_seconds += verify_dt;

            var acc: u32 = 0;
            while (acc + 1 < verify_size) : (acc += 1) {
                const expected = try target.next(acc);
                if (block[1 + acc] != expected) break;
            }
            if (self.dcut_enabled and propose_size > 1) {
                self.dcut.observe(draft_dt * 1000.0, verify_size, verify_dt * 1000.0, conf_buf[0 .. propose_size - 1], acc, propose_size);
            }
            accepted_total += acc;

            // Emit `anchor` + matched prefix up to EOS or `limit`.
            out_tokens[count] = anchor;
            count += 1;
            if (opts.stream) |sh| try sh.on_token(sh.ctx, anchor, false);
            var kept: u32 = 1;
            for (0..acc) |k| {
                const tok = block[1 + k];
                out_tokens[count] = tok;
                count += 1;
                kept += 1;
                const tok_eos = opts.stop_on_eos and isEos(target, tok);
                if (opts.stream) |sh| try sh.on_token(sh.ctx, tok, tok_eos);
                if (tok_eos) {
                    eos = true;
                    break;
                }
            }
            last_row = kept - 1;
            const bonus = try target.next(last_row);
            next_token = bonus;

            if (kept < verify_size) try target.rewind(pos_before + kept);
            pending_commit = kept;

            if (eos or count >= limit) {
                if (opts.commit_all and pending_commit > 0) {
                    const td1 = try io.now();
                    try self.commitContext(target.ctx, pending_commit, true);
                    pending_commit = 0;
                    draft_seconds += (try io.now()) - td1;
                }
                break;
            }
            anchor = bonus;
        }

        return .{
            .count = count,
            .eos = eos,
            .interrupted = interrupted,
            .context_full = context_full,
            .last_row = last_row,
            .next_token = next_token,
            .batches = batches,
            .proposed = proposed,
            .accepted = accepted_total,
            .dispatches = dispatches,
            .draft_seconds = draft_seconds,
            .verify_seconds = verify_seconds,
        };
    }
};
