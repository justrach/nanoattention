//! Front-end registry: model_type -> program builder. The only place that
//! names architectures; everything below a front end is model-agnostic.
const std = @import("std");
const ir = @import("ir.zig");
const inkling2_loop = @import("inkling2_loop.zig");
pub const lfm3 = @import("lfm3.zig");
const lfm2 = @import("lfm2.zig");

pub const Checkpoint = lfm3.Checkpoint;

pub const LoopCdMode = inkling2_loop.LoopCdMode;
pub const parseLoopCdMode = inkling2_loop.parseLoopCdMode;
pub const NoPeMode = inkling2_loop.NoPeMode;
pub const parseNoPeMode = inkling2_loop.parseNoPeMode;

pub const Options = struct {
    sliding_visible: ?u32 = null,
    embed_round: ir.Round = .none,
    loop_passes: ?u32 = null,
    loopcd: LoopCdMode = .off,
    loopcd_omega: f32 = 1.0,
    loopcd_weak_pass: u32 = 1,
    nope: NoPeMode = .off,
};

pub fn build(gpa: std.mem.Allocator, model_type: []const u8, config_bytes: []const u8, checkpoint: Checkpoint, options: Options) !ir.Program {
    const has_loop_opts = options.loop_passes != null or options.loopcd != .off or options.loopcd_omega != 1.0 or options.loopcd_weak_pass != 1 or options.nope != .off;
    const opts: lfm3.Options = .{ .sliding_visible = options.sliding_visible, .embed_round = options.embed_round };
    if (std.mem.eql(u8, model_type, "lfm3")) {
        if (has_loop_opts) return error.UnsupportedLoopOptions;
        return lfm3.build(gpa, config_bytes, checkpoint, opts);
    }
    if (std.mem.eql(u8, model_type, "lfm2")) {
        if (has_loop_opts) return error.UnsupportedLoopOptions;
        return lfm2.build(gpa, config_bytes, checkpoint, opts);
    }
    if (std.mem.eql(u8, model_type, "qwen3_5") or std.mem.eql(u8, model_type, "qwen3_5_text")) {
        if (has_loop_opts) return error.UnsupportedLoopOptions;
        if (options.sliding_visible != null or options.embed_round != .none) return error.UnsupportedQwen35Options;
        return @import("qwen3_5.zig").build(gpa, config_bytes, checkpoint);
    }
    if (std.mem.eql(u8, model_type, "gemma4") or
        std.mem.eql(u8, model_type, "gemma4_text") or
        std.mem.eql(u8, model_type, "gemma4_unified") or
        std.mem.eql(u8, model_type, "gemma4_unified_text"))
    {
        if (has_loop_opts) return error.UnsupportedLoopOptions;
        if (options.sliding_visible != null or options.embed_round != .none) return error.UnsupportedGemma4Options;
        return @import("gemma4.zig").build(gpa, config_bytes, checkpoint);
    }
    if (std.mem.eql(u8, model_type, "inkling2_loop")) {
        if (options.sliding_visible != null or options.embed_round != .none) return error.UnsupportedInklingOptions;
        return inkling2_loop.buildWithOptions(gpa, config_bytes, .{
            .loop_passes = options.loop_passes,
            .loopcd = options.loopcd,
            .loopcd_omega = options.loopcd_omega,
            .loopcd_weak_pass = options.loopcd_weak_pass,
            .nope = options.nope,
        });
    }
    std.debug.print("No megakernel front end for model_type \"{s}\"\n", .{model_type});
    return error.UnsupportedModelType;
}

// ---- cli: chat templates (src/mega/chat.zig) ----
pub const chat = @import("chat.zig");

/// The model's chat template, if its front end supplies one.
pub fn chatTemplate(model_type: []const u8) ?chat.Template {
    if (std.mem.eql(u8, model_type, "lfm3")) return lfm3.chat_template;
    return null;
}
