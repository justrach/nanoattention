//! Opt-in numerical traces only. Never enabled in benchmark mode.
const std = @import("std");
extern "c" fn write(c_int, [*]const u8, usize) isize;

fn output(bytes: []const u8) !void {
    var offset: usize = 0;
    while (offset < bytes.len) {
        const count = write(2, bytes[offset..].ptr, bytes.len - offset);
        if (count <= 0) return error.TraceOutputFailed;
        offset += @intCast(count);
    }
}

pub fn emit(position: u32, layer: usize, stage: []const u8, values: anytype) !void {
    const bytes = try std.json.Stringify.valueAlloc(std.heap.page_allocator, .{
        .position = position,
        .layer = layer,
        .stage = stage,
        .values = values,
    }, .{});
    defer std.heap.page_allocator.free(bytes);
    try output("FLINT_TRACE ");
    try output(bytes);
    try output("\n");
}
