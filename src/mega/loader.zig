//! Checkpoint directory -> Metal context, mapped weights, program and engine.
const std = @import("std");
const ir = @import("ir.zig");
const gpu = @import("gpu.zig");
const exec = @import("exec.zig");
const frontend = @import("frontend.zig");
const Knobs = @import("knobs.zig").Knobs;
const weights = @import("weights");
const metal = weights.metal;

pub fn readFile(a: std.mem.Allocator, path: []const u8) ![]u8 {
    const map = try metal.Mapping.open(a, path);
    defer map.deinit();
    return a.dupe(u8, map.bytes);
}

fn storeHas(context: *const anyopaque, name: []const u8) bool {
    const store: *const weights.Store = @ptrCast(@alignCast(context));
    return (store.find(name) catch return false) != null;
}

pub const Loaded = struct {
    gpa: std.mem.Allocator,
    ctx: metal.Context,
    store: *weights.Store,
    program: *ir.Program,
    engine: exec.Engine,

    pub fn deinit(self: *Loaded) void {
        self.engine.deinit();
        self.program.deinit();
        self.gpa.destroy(self.program);
        self.store.deinit();
        self.gpa.destroy(self.store);
        self.ctx.deinit();
    }
};

/// The program is also returned unbound by `buildProgram`, for `inspect`.
pub fn buildProgram(gpa: std.mem.Allocator, a: std.mem.Allocator, directory: []const u8, store: *const weights.Store, options: frontend.Options) !ir.Program {
    const config = try readFile(a, try std.fs.path.join(a, &.{ directory, "config.json" }));
    const model_type = try exec.modelType(a, config);
    return frontend.build(gpa, model_type, config, .{ .context = store, .has = storeHas }, options);
}

pub fn load(gpa: std.mem.Allocator, a: std.mem.Allocator, directory: []const u8, knobs: Knobs, context: u32, options: frontend.Options) !*Loaded {
    const config = try readFile(a, try std.fs.path.join(a, &.{ directory, "config.json" }));
    const ctx = try gpu.open();
    errdefer ctx.deinit();
    const store = try gpa.create(weights.Store);
    errdefer gpa.destroy(store);
    store.* = try weights.Store.init(a, ctx, directory, try exec.quantGroup(a, config));
    errdefer store.deinit();
    const program = try gpa.create(ir.Program);
    errdefer gpa.destroy(program);
    program.* = try buildProgram(gpa, a, directory, store, options);
    errdefer program.deinit();
    const loaded = try gpa.create(Loaded);
    errdefer gpa.destroy(loaded);
    loaded.* = .{ .gpa = gpa, .ctx = ctx, .store = store, .program = program, .engine = undefined };
    loaded.engine = try exec.Engine.init(gpa, program, ctx, store, knobs, context);
    return loaded;
}
