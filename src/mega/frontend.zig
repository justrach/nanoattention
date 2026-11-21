//! Front-end registry: model_type -> program builder. The only place that
//! names architectures; everything below a front end is model-agnostic.
const std = @import("std");
const ir = @import("ir.zig");
const inkling2_loop = @import("inkling2_loop.zig");
pub const lfm3 = @import("lfm3.zig");

pub const Checkpoint = lfm3.Checkpoint;

pub const Options = struct {
    sliding_visible: ?u32 = null,
    embed_round: ir.Round = .none,
};

pub fn build(gpa: std.mem.Allocator, model_type: []const u8, config_bytes: []const u8, checkpoint: Checkpoint, options: Options) !ir.Program {
    if (std.mem.eql(u8, model_type, "lfm3"))
        return lfm3.build(gpa, config_bytes, checkpoint, .{ .sliding_visible = options.sliding_visible, .embed_round = options.embed_round });
    if (std.mem.eql(u8, model_type, "inkling2_loop")) {
        if (options.sliding_visible != null or options.embed_round != .none) return error.UnsupportedInklingOptions;
        return inkling2_loop.build(gpa, config_bytes);
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
