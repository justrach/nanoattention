//! Minimal libc I/O: stdout, whole-file writes, a monotonic clock.
const std = @import("std");

extern "c" fn write(c_int, [*]const u8, usize) isize;
extern "c" fn open(path: [*:0]const u8, flags: c_int, ...) c_int;
extern "c" fn close(c_int) c_int;
extern "c" fn rename([*:0]const u8, [*:0]const u8) c_int;
const Timespec = extern struct { sec: i64, nsec: i64 };
extern "c" fn clock_gettime(c_int, *Timespec) c_int;

pub fn writeAll(fd: c_int, bytes: []const u8) !void {
    var offset: usize = 0;
    while (offset < bytes.len) {
        const n = write(fd, bytes[offset..].ptr, bytes.len - offset);
        if (n <= 0) return error.OutputFailed;
        offset += @intCast(n);
    }
}
pub fn out(bytes: []const u8) !void {
    return writeAll(1, bytes);
}
pub fn print(a: std.mem.Allocator, comptime fmt: []const u8, args: anytype) !void {
    return out(try std.fmt.allocPrint(a, fmt, args));
}

/// Write via a temporary file and rename, so readers never see a partial file.
pub fn writeFile(a: std.mem.Allocator, path: []const u8, bytes: []const u8) !void {
    const tmp = try std.fmt.allocPrintSentinel(a, "{s}.tmp", .{path}, 0);
    const final = try a.dupeSentinel(u8, path, 0);
    // Darwin: O_WRONLY | O_CREAT | O_TRUNC.
    const fd = open(tmp, 0x1 | 0x200 | 0x400, @as(c_uint, 0o644));
    if (fd < 0) return error.CannotCreateFile;
    writeAll(fd, bytes) catch |err| {
        _ = close(fd);
        return err;
    };
    if (close(fd) != 0) return error.OutputFailed;
    if (rename(tmp, final) != 0) return error.CannotRenameFile;
}

/// Darwin CLOCK_MONOTONIC seconds, as the Flint session and benchmarks use.
pub fn now() !f64 {
    var ts: Timespec = undefined;
    if (clock_gettime(6, &ts) != 0) return error.ClockFailed;
    return @as(f64, @floatFromInt(ts.sec)) + @as(f64, @floatFromInt(ts.nsec)) * 1e-9;
}
