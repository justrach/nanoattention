const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});
    if (target.result.os.tag != .macos or target.result.cpu.arch != .aarch64)
        @panic("Flint Metal currently requires Apple Silicon macOS");
    const module = b.createModule(.{
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    module.addCSourceFile(.{ .file = b.path("src/metal.m"), .flags = &.{ "-std=c11", "-fno-objc-arc" } });
    module.linkFramework("Foundation", .{});
    module.linkFramework("Metal", .{});
    const proofs = b.addSystemCommand(&.{"python3"});
    proofs.addFileArg(b.path("tools/check_resident_proofs.py"));
    b.step("proofs", "Build Lean proofs and check exact Zig/Metal plan bindings").dependOn(&proofs.step);
    const exe = b.addExecutable(.{ .name = "flint", .root_module = module });
    exe.step.dependOn(&proofs.step);
    b.installArtifact(exe);
    const bench_module = b.createModule(.{
        .root_source_file = b.path("src/benchmark.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    bench_module.addCSourceFile(.{ .file = b.path("src/metal.m"), .flags = &.{ "-std=c11", "-fno-objc-arc" } });
    bench_module.linkFramework("Foundation", .{});
    bench_module.linkFramework("Metal", .{});
    const bench = b.addExecutable(.{ .name = "flint-bench", .root_module = bench_module });
    bench.step.dependOn(&proofs.step);
    b.installArtifact(bench);
    const tests = b.addTest(.{ .root_module = module });
    tests.step.dependOn(&proofs.step);
    const run_tests = b.addRunArtifact(tests);
    b.step("test", "Run checkpoint and headless Metal kernel tests").dependOn(&run_tests.step);
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
