const std = @import("std");

fn addMetalSdk(b: *std.Build, module: *std.Build.Module, sdk: ?[]const u8) void {
    const flags: []const []const u8 = if (sdk) |dir|
        b.allocator.dupe([]const u8, &.{ "-std=c11", "-fno-objc-arc", "-isysroot", dir }) catch @panic("OOM")
    else
        &.{ "-std=c11", "-fno-objc-arc" };
    module.addCSourceFile(.{ .file = b.path("src/metal.m"), .flags = flags });
    module.addCSourceFile(.{ .file = b.path("src/image.m"), .flags = flags });
    if (sdk) |dir| {
        module.addSystemFrameworkPath(.{ .cwd_relative = b.pathJoin(&.{ dir, "System/Library/Frameworks" }) });
        module.addLibraryPath(.{ .cwd_relative = b.pathJoin(&.{ dir, "usr/lib" }) });
        module.addSystemIncludePath(.{ .cwd_relative = b.pathJoin(&.{ dir, "usr/include" }) });
    }
    module.linkFramework("Foundation", .{});
    module.linkFramework("Metal", .{});
    module.linkFramework("CoreGraphics", .{});
    module.linkFramework("ImageIO", .{});
}

fn addSmokeExe(
    b: *std.Build,
    name: []const u8,
    target: std.Build.ResolvedTarget,
    optimize: std.builtin.OptimizeMode,
    sdk: ?[]const u8,
    nano_lib: *std.Build.Step.Compile,
) *std.Build.Step.Compile {
    const smoke_mod = b.createModule(.{
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    const flags: []const []const u8 = if (sdk) |dir|
        b.allocator.dupe([]const u8, &.{ "-std=c99", "-Wall", "-Wextra", "-isysroot", dir }) catch @panic("OOM")
    else
        &.{ "-std=c99", "-Wall", "-Wextra" };
    smoke_mod.addCSourceFile(.{ .file = b.path("tools/ios_c_api_smoke.c"), .flags = flags });
    smoke_mod.addIncludePath(b.path("include"));
    smoke_mod.linkLibrary(nano_lib);
    if (sdk) |dir| {
        smoke_mod.addSystemFrameworkPath(.{ .cwd_relative = b.pathJoin(&.{ dir, "System/Library/Frameworks" }) });
        smoke_mod.addLibraryPath(.{ .cwd_relative = b.pathJoin(&.{ dir, "usr/lib" }) });
        smoke_mod.addSystemIncludePath(.{ .cwd_relative = b.pathJoin(&.{ dir, "usr/include" }) });
    }
    smoke_mod.linkFramework("Foundation", .{});
    smoke_mod.linkFramework("Metal", .{});
    smoke_mod.linkFramework("CoreGraphics", .{});
    smoke_mod.linkFramework("ImageIO", .{});
    return b.addExecutable(.{ .name = name, .root_module = smoke_mod });
}

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});
    if ((target.result.os.tag != .macos and target.result.os.tag != .ios) or target.result.cpu.arch != .aarch64)
        @panic("nanoattention Metal currently requires Apple Silicon macOS or iOS");
    const sdk: ?[]const u8 = if (b.option([]const u8, "sdk", "Apple SDK path for non-native target builds")) |s|
        s
    else if (!target.query.isNative())
        std.zig.system.darwin.getSdk(b.allocator, b.graph.io, &target.result)
    else
        null;
    const proofs = b.addSystemCommand(&.{"python3"});
    proofs.addFileArg(b.path("tools/check_resident_proofs.py"));
    b.step("proofs", "Build Lean proofs and check exact Zig/Metal plan bindings").dependOn(&proofs.step);

    // nano: the model-agnostic megakernel. Shares the weights/metal/safetensors
    // files through one module so their types are identical to Flint's.
    const shared = b.createModule(.{
        .root_source_file = b.path("src/weights.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    const nano_lib_module = b.createModule(.{
        .root_source_file = b.path("src/mega/c_api.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .imports = &.{.{ .name = "weights", .module = shared }},
    });
    addMetalSdk(b, nano_lib_module, sdk);
    const nano_lib = b.addLibrary(.{ .linkage = .static, .name = "nano", .root_module = nano_lib_module });
    nano_lib.step.dependOn(&proofs.step);
    b.installArtifact(nano_lib);
    b.getInstallStep().dependOn(&b.addInstallFile(b.path("include/nano.h"), "include/nano.h").step);

    const nano_module = b.createModule(.{
        .root_source_file = b.path("src/mega/main.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .imports = &.{.{ .name = "weights", .module = shared }},
    });
    addMetalSdk(b, nano_module, sdk);

    if (target.result.os.tag == .ios) {
        if (target.result.abi == .simulator) {
            const nano_sim = b.addExecutable(.{ .name = "nano", .root_module = nano_module });
            nano_sim.step.dependOn(&proofs.step);
            b.installArtifact(nano_sim);
            const smoke_sim = addSmokeExe(b, "nano-ios-smoke", target, optimize, sdk, nano_lib);
            smoke_sim.step.dependOn(&proofs.step);
            b.installArtifact(smoke_sim);
        }
        return;
    }

    const module = b.createModule(.{
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    addMetalSdk(b, module, sdk);
    const exe = b.addExecutable(.{ .name = "flint", .root_module = module });
    exe.step.dependOn(&proofs.step);
    b.installArtifact(exe);
    const bench_module = b.createModule(.{
        .root_source_file = b.path("src/benchmark.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    addMetalSdk(b, bench_module, sdk);
    const bench = b.addExecutable(.{ .name = "flint-bench", .root_module = bench_module });
    bench.step.dependOn(&proofs.step);
    b.installArtifact(bench);
    const speculative_module = b.createModule(.{
        .root_source_file = b.path("src/speculative_bench.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    addMetalSdk(b, speculative_module, sdk);
    const speculative_bench = b.addExecutable(.{ .name = "flint-speculative-bench", .root_module = speculative_module });
    speculative_bench.step.dependOn(&proofs.step);
    b.installArtifact(speculative_bench);
    const nano = b.addExecutable(.{ .name = "nano", .root_module = nano_module });
    nano.step.dependOn(&proofs.step);
    b.installArtifact(nano);
    const smoke_mac = addSmokeExe(b, "nano-c-smoke", target, optimize, sdk, nano_lib);
    smoke_mac.step.dependOn(&proofs.step);
    b.step("c-smoke", "Build native macOS C API smoke test").dependOn(&b.addInstallArtifact(smoke_mac, .{}).step);
    const nano_tests = b.addTest(.{ .root_module = nano_module });
    nano_tests.step.dependOn(&proofs.step);
    const run_nano_tests = b.addRunArtifact(nano_tests);
    b.step("test-mega", "Run megakernel IR, front-end, state and headless Metal kernel tests").dependOn(&run_nano_tests.step);
    const tests = b.addTest(.{ .root_module = module });
    tests.step.dependOn(&proofs.step);
    const run_tests = b.addRunArtifact(tests);
    const test_step = b.step("test", "Run checkpoint and headless Metal kernel tests");
    test_step.dependOn(&run_tests.step);
    test_step.dependOn(&run_nano_tests.step);
    const parity = b.addSystemCommand(&.{"python3"});
    parity.addFileArg(b.path("tools/verify.py"));
    parity.addArtifactArg(exe);
    const verify = b.step("verify", "Run unit tests and synthetic Flint Metal/NumPy parity (requires numpy)");
    verify.dependOn(&run_tests.step);
    verify.dependOn(&parity.step);
    const resident_one = b.addSystemCommand(&.{"python3"});
    resident_one.addFileArg(b.path("tools/verify.py"));
    resident_one.addArtifactArg(exe);
    resident_one.addArgs(&.{ "--resident-workers", "1" });
    resident_one.step.dependOn(&run_tests.step);
    const resident_four = b.addSystemCommand(&.{"python3"});
    resident_four.addFileArg(b.path("tools/verify.py"));
    resident_four.addArtifactArg(exe);
    resident_four.addArgs(&.{ "--resident-workers", "4" });
    resident_four.step.dependOn(&resident_one.step);
    b.step("verify-resident", "Check one-dispatch certified resident execution with one and four workers").dependOn(&resident_four.step);
}
