//! Native Inkling 2 Vision encoder (`inkling2_vision`):
//! ImageIO RGB decode -> FP32 Lanczos3 resize & right/bottom 40px padding ->
//! 20x20 patch embedding + CLS + bilinear 2D pos emb + ln_pre ->
//! 23-layer ViT with interleaved 2D RoPE, bidirectional MHA, LayerScale & GELU ->
//! 2x2 spatial merge -> Linear(4096->4096) + GELU + Linear(4096->2304).
const std = @import("std");
const gpu = @import("gpu.zig");
const weights = @import("weights");
const metal = weights.metal;
const Allocator = std.mem.Allocator;

extern fn fm_load_image_rgba(path: [*:0]const u8, out_rgba: *?[*]u8, out_w: *u32, out_h: *u32) c_int;
extern fn fm_load_image_rgba_bytes(data: [*]const u8, len: usize, out_rgba: *?[*]u8, out_w: *u32, out_h: *u32) c_int;
extern fn fm_load_image_rgba_url(url: [*:0]const u8, out_rgba: *?[*]u8, out_w: *u32, out_h: *u32) c_int;
extern fn fm_free_image(pixels: [*]u8) void;

pub const ImageInput = union(enum) {
    source: []const u8,
    bytes: []const u8,
};

pub const image_token_id: u32 = 200054;
pub const default_max_image_tokens: u32 = 1024;
pub const default_max_image_dim: u32 = 4096;

pub const PatchLayout = enum { hwc, chw };

pub const Config = struct {
    text_hidden_size: u32,
    native_patch_size: u32,
    adapter_merge_size: u32,
    projection_hidden_dim: u32,
    projector_bias: bool,
    hidden_size: u32,
    num_hidden_layers: u32,
    num_attention_heads: u32,
    mlp_ratio: f32,
    posemb_grid_size: u32,
    layer_norm_eps: f32,
    rope_theta: f32 = 10000.0,
    patch_layout: PatchLayout = .hwc,

    pub fn headDim(self: Config) u32 {
        return self.hidden_size / self.num_attention_heads;
    }
    pub fn mlpDim(self: Config) u32 {
        return @intFromFloat(@round(@as(f32, @floatFromInt(self.hidden_size)) * self.mlp_ratio));
    }
    pub fn patchPixels(self: Config) u32 {
        return self.native_patch_size * self.native_patch_size * 3;
    }
    pub fn tilePixels(self: Config) u32 {
        return self.native_patch_size * self.adapter_merge_size;
    }
};

fn jsonObj(v: ?std.json.Value) !std.json.ObjectMap {
    const val = v orelse return error.MissingVisionConfig;
    return switch (val) {
        .object => |o| o,
        else => error.InvalidVisionConfig,
    };
}
fn jsonU32(v: ?std.json.Value, default: ?u32) !u32 {
    const val = v orelse return default orelse error.MissingVisionConfigField;
    return switch (val) {
        .integer => |n| std.math.cast(u32, n) orelse error.InvalidVisionConfig,
        else => error.InvalidVisionConfig,
    };
}
fn jsonF32(v: ?std.json.Value, default: ?f32) !f32 {
    const val = v orelse return default orelse error.MissingVisionConfigField;
    return switch (val) {
        .integer => |n| @floatFromInt(n),
        .float => |n| @floatCast(n),
        else => error.InvalidVisionConfig,
    };
}
fn jsonBool(v: ?std.json.Value, default: bool) !bool {
    const val = v orelse return default;
    return switch (val) {
        .bool => |b| b,
        else => error.InvalidVisionConfig,
    };
}

pub fn hasVisionConfig(a: Allocator, config_bytes: []const u8) bool {
    const parsed = std.json.parseFromSliceLeaky(std.json.Value, a, config_bytes, .{}) catch return false;
    if (parsed != .object) return false;
    const vc = parsed.object.get("vision_config") orelse return false;
    return vc == .object;
}

pub fn parseConfig(a: Allocator, config_bytes: []const u8) !Config {
    const parsed = try std.json.parseFromSliceLeaky(std.json.Value, a, config_bytes, .{});
    const root = try jsonObj(parsed);
    const vc = try jsonObj(root.get("vision_config"));
    if (vc.get("model_type")) |mt| {
        if (mt != .string or !std.mem.eql(u8, mt.string, "inkling2_vision")) return error.UnsupportedVisionModelType;
    } else return error.MissingVisionConfigField;

    const cfg: Config = .{
        .text_hidden_size = try jsonU32(vc.get("text_hidden_size"), null),
        .native_patch_size = try jsonU32(vc.get("native_patch_size"), 20),
        .adapter_merge_size = try jsonU32(vc.get("adapter_merge_size"), 2),
        .projection_hidden_dim = try jsonU32(vc.get("projection_hidden_dim"), 4096),
        .projector_bias = try jsonBool(vc.get("projector_bias"), false),
        .hidden_size = try jsonU32(vc.get("hidden_size"), 1024),
        .num_hidden_layers = try jsonU32(vc.get("num_hidden_layers"), 23),
        .num_attention_heads = try jsonU32(vc.get("num_attention_heads"), 16),
        .mlp_ratio = try jsonF32(vc.get("mlp_ratio"), 4.0),
        .posemb_grid_size = try jsonU32(vc.get("posemb_grid_size"), 32),
        .layer_norm_eps = try jsonF32(vc.get("layer_norm_eps"), 1e-5),
    };
    if (cfg.hidden_size == 0 or cfg.num_hidden_layers == 0 or cfg.num_attention_heads == 0 or
        cfg.hidden_size % cfg.num_attention_heads != 0 or cfg.headDim() % 4 != 0 or cfg.headDim() > 128 or
        cfg.native_patch_size == 0 or cfg.adapter_merge_size == 0 or cfg.posemb_grid_size == 0 or
        cfg.text_hidden_size == 0 or cfg.projection_hidden_dim == 0 or cfg.projector_bias)
    {
        return error.UnsupportedVisionConfig;
    }
    return cfg;
}

// ---- Native image decode + FP32 Lanczos3 resize + 40px padding --------------

pub const RgbImage = struct {
    pixels: [*]u8,
    width: u32,
    height: u32,
    channels: u32 = 4,
    owned_by_c: bool = true,

    pub fn slice(self: RgbImage) []const u8 {
        return self.pixels[0 .. @as(usize, self.width) * self.height * self.channels];
    }

    pub fn deinit(self: RgbImage) void {
        if (self.owned_by_c) fm_free_image(self.pixels);
    }
};

pub fn decodeImageBytes(bytes: []const u8) !RgbImage {
    if (bytes.len == 0) return error.ImageDecodeFailed;
    var px_ptr: ?[*]u8 = null;
    var w: u32 = 0;
    var h: u32 = 0;
    if (fm_load_image_rgba_bytes(bytes.ptr, bytes.len, &px_ptr, &w, &h) == 0 or px_ptr == null or w == 0 or h == 0) {
        return error.ImageDecodeFailed;
    }
    return .{ .pixels = px_ptr.?, .width = w, .height = h, .channels = 4, .owned_by_c = true };
}

fn hexNibble(ch: u8) ?u8 {
    return switch (ch) {
        '0'...'9' => ch - '0',
        'a'...'f' => ch - 'a' + 10,
        'A'...'F' => ch - 'A' + 10,
        else => null,
    };
}

pub fn decodeFileUri(a: Allocator, uri: []const u8) ![]u8 {
    if (!std.mem.startsWith(u8, uri, "file://")) return error.InvalidImageUri;
    var rest = uri["file://".len..];
    if (std.mem.startsWith(u8, rest, "localhost/")) {
        rest = rest["localhost".len..];
    } else if (rest.len == 0 or rest[0] != '/') {
        return error.InvalidImageUri;
    }
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(a);
    try out.ensureTotalCapacity(a, rest.len);
    var i: usize = 0;
    while (i < rest.len) {
        if (rest[i] == '%' and i + 2 < rest.len) {
            const hi = hexNibble(rest[i + 1]) orelse return error.InvalidImageUri;
            const lo = hexNibble(rest[i + 2]) orelse return error.InvalidImageUri;
            try out.append(a, (hi << 4) | lo);
            i += 3;
        } else {
            try out.append(a, rest[i]);
            i += 1;
        }
    }
    return out.toOwnedSlice(a);
}

pub fn decodeDataUriBytes(a: Allocator, uri: []const u8) ![]u8 {
    if (!std.mem.startsWith(u8, uri, "data:")) return error.InvalidImageUri;
    const comma = std.mem.indexOfScalar(u8, uri, ',') orelse return error.InvalidImageUri;
    const header = uri["data:".len..comma];
    var is_base64 = false;
    var parts = std.mem.splitScalar(u8, header, ';');
    while (parts.next()) |part| {
        const trimmed = std.mem.trim(u8, part, " \t");
        if (std.ascii.eqlIgnoreCase(trimmed, "base64")) {
            is_base64 = true;
            break;
        }
    }
    if (!is_base64) return error.InvalidImageUri;

    const raw_payload = uri[comma + 1 ..];
    var clean: std.ArrayList(u8) = .empty;
    defer clean.deinit(a);
    try clean.ensureTotalCapacity(a, raw_payload.len);
    for (raw_payload) |ch| {
        if (ch == ' ' or ch == '\t' or ch == '\r' or ch == '\n') continue;
        try clean.append(a, ch);
    }
    const decoded_len = std.base64.standard.Decoder.calcSizeForSlice(clean.items) catch return error.InvalidImageUri;
    const out = try a.alloc(u8, decoded_len);
    errdefer a.free(out);
    std.base64.standard.Decoder.decode(out, clean.items) catch return error.InvalidImageUri;
    return out;
}

/// Decode an image from a local path, `file://` URI, `data:...;base64,...` URI, or `http(s)://` URL.
pub fn decodeImageFile(a: Allocator, source: []const u8) !RgbImage {
    const trimmed = std.mem.trim(u8, source, " \t\r\n");
    if (trimmed.len == 0) return error.ImageDecodeFailed;
    if (std.mem.startsWith(u8, trimmed, "data:")) {
        const bytes = try decodeDataUriBytes(a, trimmed);
        defer a.free(bytes);
        return decodeImageBytes(bytes);
    }
    if (std.mem.startsWith(u8, trimmed, "http://") or std.mem.startsWith(u8, trimmed, "https://")) {
        const zurl = try a.dupeSentinel(u8, trimmed, 0);
        defer a.free(zurl);
        var px_ptr: ?[*]u8 = null;
        var w: u32 = 0;
        var h: u32 = 0;
        if (fm_load_image_rgba_url(zurl, &px_ptr, &w, &h) == 0 or px_ptr == null or w == 0 or h == 0) {
            return error.ImageDecodeFailed;
        }
        return .{ .pixels = px_ptr.?, .width = w, .height = h, .channels = 4, .owned_by_c = true };
    }
    var owned_path: ?[]u8 = null;
    defer if (owned_path) |p| a.free(p);
    const fs_path: []const u8 = if (std.mem.startsWith(u8, trimmed, "file://")) blk: {
        owned_path = try decodeFileUri(a, trimmed);
        break :blk owned_path.?;
    } else trimmed;

    const zpath = try a.dupeSentinel(u8, fs_path, 0);
    defer a.free(zpath);
    var px_ptr: ?[*]u8 = null;
    var w: u32 = 0;
    var h: u32 = 0;
    if (fm_load_image_rgba(zpath, &px_ptr, &w, &h) == 0 or px_ptr == null or w == 0 or h == 0) {
        return error.ImageDecodeFailed;
    }
    return .{ .pixels = px_ptr.?, .width = w, .height = h, .channels = 4, .owned_by_c = true };
}

pub const TargetDims = struct {
    resized_w: u32,
    resized_h: u32,
    padded_w: u32,
    padded_h: u32,
    grid_h: u32,
    grid_w: u32,
    visual_tokens: u32,
};

pub fn computeTargetDims(w: u32, h: u32, cfg: Config, max_image_tokens: u32, max_dim: u32) TargetDims {
    const tile = cfg.tilePixels(); // 40
    var rw: u32 = @max(@as(u32, 1), w);
    var rh: u32 = @max(@as(u32, 1), h);

    // Enforce max_dim while preserving aspect ratio.
    if (rw > max_dim or rh > max_dim) {
        const s = @as(f64, @floatFromInt(max_dim)) / @as(f64, @floatFromInt(@max(rw, rh)));
        rw = @max(@as(u32, 1), @as(u32, @intFromFloat(@round(@as(f64, @floatFromInt(rw)) * s))));
        rh = @max(@as(u32, 1), @as(u32, @intFromFloat(@round(@as(f64, @floatFromInt(rh)) * s))));
    }

    // Enforce max_image_tokens budget (each visual token is tile x tile pixels).
    var tw = ((rw + tile - 1) / tile);
    var th = ((rh + tile - 1) / tile);
    if (@as(u64, tw) * th > max_image_tokens) {
        const scale = @sqrt(@as(f64, @floatFromInt(max_image_tokens)) / @as(f64, @floatFromInt(@as(u64, tw) * th)));
        rw = @max(@as(u32, 1), @as(u32, @intFromFloat(@floor(@as(f64, @floatFromInt(rw)) * scale))));
        rh = @max(@as(u32, 1), @as(u32, @intFromFloat(@floor(@as(f64, @floatFromInt(rh)) * scale))));
        tw = (rw + tile - 1) / tile;
        th = (rh + tile - 1) / tile;
        while (@as(u64, tw) * th > max_image_tokens and (rw > tile or rh > tile)) {
            if (rw >= rh and rw > tile) {
                rw = (tw - 1) * tile;
            } else if (rh > tile) {
                rh = (th - 1) * tile;
            } else break;
            tw = (rw + tile - 1) / tile;
            th = (rh + tile - 1) / tile;
        }
    }

    const padded_w = tw * tile;
    const padded_h = th * tile;
    return .{
        .resized_w = rw,
        .resized_h = rh,
        .padded_w = padded_w,
        .padded_h = padded_h,
        .grid_h = padded_h / cfg.native_patch_size,
        .grid_w = padded_w / cfg.native_patch_size,
        .visual_tokens = tw * th,
    };
}

fn lanczos3Weight(x_in: f32) f32 {
    const x = @abs(x_in);
    if (x < 1e-6) return 1.0;
    if (x >= 3.0) return 0.0;
    const pi_x = std.math.pi * x;
    const pi_x_3 = pi_x / 3.0;
    return (@sin(pi_x) / pi_x) * (@sin(pi_x_3) / pi_x_3);
}

const Vec4 = @Vector(4, f32);

const LanczosTable = struct {
    first: []u32,
    count: []u32,
    weights: []f32,
    stride: usize,

    fn build(a: Allocator, src_len: u32, dst_len: u32) !LanczosTable {
        const scale = @as(f32, @floatFromInt(src_len)) / @as(f32, @floatFromInt(dst_len));
        const filter_scale: f32 = @max(1.0, scale);
        const support: f32 = 3.0 * filter_scale;
        const stride: usize = @intFromFloat(@ceil(2.0 * support) + 3.0);

        const first = try a.alloc(u32, dst_len);
        errdefer a.free(first);
        const count = try a.alloc(u32, dst_len);
        errdefer a.free(count);
        const weights_buf = try a.alloc(f32, @as(usize, dst_len) * stride);
        errdefer a.free(weights_buf);

        for (0..dst_len) |d| {
            const center = (@as(f32, @floatFromInt(d)) + 0.5) * scale - 0.5;
            const lo: i32 = @max(0, @as(i32, @intFromFloat(@floor(center - support))));
            const hi: i32 = @min(@as(i32, @intCast(src_len)) - 1, @as(i32, @intFromFloat(@ceil(center + support))));
            const base = d * stride;
            var w_sum: f32 = 0;
            var cnt: usize = 0;
            var s = lo;
            while (s <= hi) : (s += 1) {
                const wt = lanczos3Weight((@as(f32, @floatFromInt(s)) - center) / filter_scale);
                weights_buf[base + cnt] = wt;
                w_sum += wt;
                cnt += 1;
            }
            const inv_w: f32 = if (w_sum > 0) 1.0 / w_sum else 1.0;
            for (weights_buf[base .. base + cnt]) |*w_ptr| w_ptr.* *= inv_w;
            first[d] = @intCast(lo);
            count[d] = @intCast(cnt);
        }
        return .{ .first = first, .count = count, .weights = weights_buf, .stride = stride };
    }

    fn deinit(self: LanczosTable, a: Allocator) void {
        a.free(self.first);
        a.free(self.count);
        a.free(self.weights);
    }
};

/// Precomputed-table SIMD separable FP32 Lanczos3 resize supporting 3-channel RGB or 4-channel RGBA.
/// Output is `[dh * dw]` Vec4 pixels `{r, g, b, 0}` clamped to `[0, 255]`.
fn resizeLanczos3Vec4(a: Allocator, src: []const u8, sw: u32, sh: u32, channels: u32, dw: u32, dh: u32) ![]Vec4 {
    const h_tbl = try LanczosTable.build(a, sw, dw);
    defer h_tbl.deinit(a);
    const v_tbl = try LanczosTable.build(a, sh, dh);
    defer v_tbl.deinit(a);

    const tmp = try a.alloc(Vec4, @as(usize, sh) * dw);
    defer a.free(tmp);
    const out = try a.alloc(Vec4, @as(usize, dh) * dw);
    errdefer a.free(out);

    const zero_v: Vec4 = @splat(0.0);
    const max_v: Vec4 = @splat(255.0);
    const ch: usize = channels;

    // Horizontal pass: outer y, inner dx (sequential streaming over scanlines).
    for (0..sh) |y| {
        const src_row = src[y * sw * ch ..][0 .. @as(usize, sw) * ch];
        const tmp_row = tmp[y * dw ..][0..dw];
        for (0..dw) |dx| {
            const lo: usize = h_tbl.first[dx];
            const cnt: usize = h_tbl.count[dx];
            const wts = h_tbl.weights[dx * h_tbl.stride ..][0..cnt];
            var acc: Vec4 = zero_v;
            for (0..cnt) |t| {
                const off = (lo + t) * ch;
                const px: Vec4 = .{
                    @floatFromInt(src_row[off + 0]),
                    @floatFromInt(src_row[off + 1]),
                    @floatFromInt(src_row[off + 2]),
                    0.0,
                };
                acc += px * @as(Vec4, @splat(wts[t]));
            }
            tmp_row[dx] = @min(@max(acc, zero_v), max_v);
        }
    }

    // Vertical pass: outer dy, inner x (vectorized row accumulation).
    for (0..dh) |dy| {
        const lo: usize = v_tbl.first[dy];
        const cnt: usize = v_tbl.count[dy];
        const wts = v_tbl.weights[dy * v_tbl.stride ..][0..cnt];
        const out_row = out[dy * dw ..][0..dw];
        @memset(out_row, zero_v);
        for (0..cnt) |t| {
            const wt_v: Vec4 = @splat(wts[t]);
            const tmp_row = tmp[(lo + t) * dw ..][0..dw];
            for (out_row, tmp_row) |*dst_px, src_px| {
                dst_px.* += src_px * wt_v;
            }
        }
        for (out_row) |*dst_px| {
            dst_px.* = @min(@max(dst_px.*, zero_v), max_v);
        }
    }
    return out;
}

/// Separable FP32 Lanczos3 resize matching Pillow/PyTorch coordinate centers.
pub fn resizeLanczos3Rgb(a: Allocator, src: []const u8, sw: u32, sh: u32, dw: u32, dh: u32) ![]f32 {
    const v4 = try resizeLanczos3Vec4(a, src, sw, sh, 3, dw, dh);
    defer a.free(v4);
    const out = try a.alloc(f32, @as(usize, dh) * dw * 3);
    for (v4, 0..) |px, i| {
        out[i * 3 + 0] = px[0];
        out[i * 3 + 1] = px[1];
        out[i * 3 + 2] = px[2];
    }
    return out;
}

pub const PreprocessedImage = struct {
    dims: TargetDims,
    /// Flattened patches [grid_h * grid_w, native_patch_size * native_patch_size * 3].
    patches: []f32,
};

/// Right/bottom padding pixel: -1 on the 0..255 scale, normalized with mean=0.5, std=0.5.
pub const pad_pixel: f32 = (-1.0 * (1.0 / 255.0) - 0.5) / 0.5;

/// Preprocess 3-channel RGB or 4-channel RGBA pixels directly into `dst_patches`
/// (`dims.grid_h * dims.grid_w * cfg.patchPixels()` floats) with zero extra patch buffer copy.
pub fn preprocessPixelsInto(
    a: Allocator,
    pixels: []const u8,
    w: u32,
    h: u32,
    channels: u32,
    cfg: Config,
    dims: TargetDims,
    dst_patches: []f32,
) !void {
    if (channels != 3 and channels != 4) return error.InvalidImageSize;
    if (pixels.len != @as(usize, w) * h * channels) return error.InvalidImageSize;
    const p = cfg.native_patch_size;
    const patch_dim = cfg.patchPixels();
    const n_patches = @as(usize, dims.grid_h) * dims.grid_w;
    if (dst_patches.len < n_patches * patch_dim) return error.InvalidImageSize;

    var resized_opt: ?[]Vec4 = null;
    defer if (resized_opt) |r| a.free(r);
    if (dims.resized_w != w or dims.resized_h != h) {
        resized_opt = try resizeLanczos3Vec4(a, pixels, w, h, channels, dims.resized_w, dims.resized_h);
    }

    @memset(dst_patches[0 .. n_patches * patch_dim], pad_pixel);

    const scale: f32 = 2.0 / 255.0;
    const ch: usize = channels;
    const rw: usize = dims.resized_w;
    const rh: usize = dims.resized_h;
    for (0..dims.grid_h) |gy| {
        const y0 = gy * p;
        const y_lim = @min(@as(usize, p), rh -| y0);
        for (0..dims.grid_w) |gx| {
            const x0 = gx * p;
            const x_lim = @min(@as(usize, p), rw -| x0);
            const patch_base = (gy * dims.grid_w + gx) * patch_dim;
            for (0..y_lim) |dy| {
                const py = y0 + dy;
                if (resized_opt) |rf| {
                    const row = rf[py * rw + x0 ..][0..x_lim];
                    switch (cfg.patch_layout) {
                        .hwc => {
                            const dst_row = dst_patches[patch_base + dy * p * 3 ..];
                            for (row, 0..) |px, dx| {
                                dst_row[dx * 3 + 0] = px[0] * scale - 1.0;
                                dst_row[dx * 3 + 1] = px[1] * scale - 1.0;
                                dst_row[dx * 3 + 2] = px[2] * scale - 1.0;
                            }
                        },
                        .chw => {
                            const plane = p * p;
                            for (row, 0..) |px, dx| {
                                const spatial = dy * p + dx;
                                dst_patches[patch_base + 0 * plane + spatial] = px[0] * scale - 1.0;
                                dst_patches[patch_base + 1 * plane + spatial] = px[1] * scale - 1.0;
                                dst_patches[patch_base + 2 * plane + spatial] = px[2] * scale - 1.0;
                            }
                        },
                    }
                } else {
                    const row_off = (py * rw + x0) * ch;
                    switch (cfg.patch_layout) {
                        .hwc => {
                            const dst_row = dst_patches[patch_base + dy * p * 3 ..];
                            for (0..x_lim) |dx| {
                                const s = row_off + dx * ch;
                                dst_row[dx * 3 + 0] = @as(f32, @floatFromInt(pixels[s + 0])) * scale - 1.0;
                                dst_row[dx * 3 + 1] = @as(f32, @floatFromInt(pixels[s + 1])) * scale - 1.0;
                                dst_row[dx * 3 + 2] = @as(f32, @floatFromInt(pixels[s + 2])) * scale - 1.0;
                            }
                        },
                        .chw => {
                            const plane = p * p;
                            for (0..x_lim) |dx| {
                                const s = row_off + dx * ch;
                                const spatial = dy * p + dx;
                                dst_patches[patch_base + 0 * plane + spatial] = @as(f32, @floatFromInt(pixels[s + 0])) * scale - 1.0;
                                dst_patches[patch_base + 1 * plane + spatial] = @as(f32, @floatFromInt(pixels[s + 1])) * scale - 1.0;
                                dst_patches[patch_base + 2 * plane + spatial] = @as(f32, @floatFromInt(pixels[s + 2])) * scale - 1.0;
                            }
                        },
                    }
                }
            }
        }
    }
}

pub fn preprocessRgb(a: Allocator, rgb: []const u8, w: u32, h: u32, cfg: Config, max_image_tokens: u32) !PreprocessedImage {
    const dims = computeTargetDims(w, h, cfg, max_image_tokens, default_max_image_dim);
    const n_patches = @as(usize, dims.grid_h) * dims.grid_w;
    const patches = try a.alloc(f32, n_patches * cfg.patchPixels());
    errdefer a.free(patches);
    try preprocessPixelsInto(a, rgb, w, h, 3, cfg, dims, patches);
    return .{ .dims = dims, .patches = patches };
}

// ---- GPU Vision Encoder -----------------------------------------------------

const LayerWeights = struct {
    ln_1_w: weights.Weight,
    ln_1_b: weights.Weight,
    ln_2_w: weights.Weight,
    ln_2_b: weights.Weight,
    ls_1: weights.Weight,
    ls_2: weights.Weight,
    q_w: weights.Weight,
    q_b: weights.Weight,
    k_w: weights.Weight,
    k_b: weights.Weight,
    v_w: weights.Weight,
    v_b: weights.Weight,
    out_w: weights.Weight,
    out_b: weights.Weight,
    fc_w: weights.Weight,
    fc_b: weights.Weight,
    proj_w: weights.Weight,
    proj_b: weights.Weight,
};

pub const EncodedImage = struct {
    /// GPU buffer containing [tokens, text_hidden_size] float32 (BF16-rounded) visual embeddings.
    buffer: metal.Buffer,
    tokens: u32,
    grid_h: u32,
    grid_w: u32,
    width: u32,

    pub fn deinit(self: EncodedImage) void {
        self.buffer.deinit();
    }
};

pub const EncodedBatch = struct {
    /// Single contiguous GPU buffer holding all images' visual embeddings [total_tokens, width].
    buffer: metal.Buffer,
    /// Visual token count per image, in order.
    counts: []u32,
    total_tokens: u32,
    width: u32,

    pub fn deinit(self: *EncodedBatch, gpa: Allocator) void {
        self.buffer.deinit();
        gpa.free(self.counts);
    }
};

const Scratch = struct {
    buf: ?metal.Buffer = null,
    words: usize = 0,
    rope_buf: ?metal.Buffer = null,
    rope_words: usize = 0,

    fn deinit(self: *Scratch) void {
        if (self.buf) |b| b.deinit();
        if (self.rope_buf) |b| b.deinit();
    }

    fn ensureBuf(self: *Scratch, ctx: metal.Context, need: usize) !metal.Buffer {
        if (self.buf != null and self.words >= need) return self.buf.?;
        const new_buf = try ctx.alloc(need);
        if (self.buf) |old| old.deinit();
        self.buf = new_buf;
        self.words = need;
        return new_buf;
    }

    fn ensureRope(self: *Scratch, ctx: metal.Context, need: usize) !metal.Buffer {
        if (self.rope_buf != null and self.rope_words >= need) return self.rope_buf.?;
        const new_buf = try ctx.alloc(need);
        if (self.rope_buf) |old| old.deinit();
        self.rope_buf = new_buf;
        self.rope_words = need;
        return new_buf;
    }
};

pub const VisionEncoder = struct {
    gpa: Allocator,
    ctx: metal.Context,
    cfg: Config,
    scratch: *Scratch,

    patch_emb_w: weights.Weight,
    class_emb: weights.Weight,
    cls_pos_emb: weights.Weight,
    pos_emb: weights.Weight,
    ln_pre_w: weights.Weight,
    ln_pre_b: weights.Weight,
    layers: []LayerWeights,
    proj_in_w: weights.Weight,
    proj_out_w: weights.Weight,

    pub fn init(gpa: Allocator, ctx: metal.Context, store: *const weights.Store, config_bytes: []const u8) !VisionEncoder {
        var scratch_arena = std.heap.ArenaAllocator.init(gpa);
        defer scratch_arena.deinit();
        const sa = scratch_arena.allocator();

        const cfg = try parseConfig(sa, config_bytes);
        const D = cfg.hidden_size;
        const M = cfg.mlpDim();
        const P = cfg.patchPixels();
        const G = cfg.posemb_grid_size;
        const merge_dim = cfg.adapter_merge_size * cfg.adapter_merge_size * D;

        const patch_emb_w = try store.get("model.visual.backbone.embeddings.patch_embedding.weight", &.{ D, P });

        const class_emb = try store.get("model.visual.backbone.embeddings.class_embedding", &.{D});
        const cls_pos_emb = try store.get("model.visual.backbone.embeddings.cls_position_embedding", &.{D});
        const pos_emb = try store.get("model.visual.backbone.embeddings.position_embedding", &.{ G * G, D });
        const ln_pre_w = try store.get("model.visual.backbone.ln_pre.weight", &.{D});
        const ln_pre_b = try store.get("model.visual.backbone.ln_pre.bias", &.{D});
        const proj_in_w = try store.get("model.visual.proj_in.weight", &.{ cfg.projection_hidden_dim, merge_dim });
        const proj_out_w = try store.get("model.visual.proj_out.weight", &.{ cfg.text_hidden_size, cfg.projection_hidden_dim });

        const layers = try gpa.alloc(LayerWeights, cfg.num_hidden_layers);
        errdefer gpa.free(layers);
        for (layers, 0..) |*lw, l| {
            const pfx = try std.fmt.allocPrint(sa, "model.visual.backbone.layers.{d}", .{l});
            lw.* = .{
                .ln_1_w = try store.get(try std.fmt.allocPrint(sa, "{s}.ln_1.weight", .{pfx}), &.{D}),
                .ln_1_b = try store.get(try std.fmt.allocPrint(sa, "{s}.ln_1.bias", .{pfx}), &.{D}),
                .ln_2_w = try store.get(try std.fmt.allocPrint(sa, "{s}.ln_2.weight", .{pfx}), &.{D}),
                .ln_2_b = try store.get(try std.fmt.allocPrint(sa, "{s}.ln_2.bias", .{pfx}), &.{D}),
                .ls_1 = try store.get(try std.fmt.allocPrint(sa, "{s}.ls_1.gamma", .{pfx}), &.{D}),
                .ls_2 = try store.get(try std.fmt.allocPrint(sa, "{s}.ls_2.gamma", .{pfx}), &.{D}),
                .q_w = try store.get(try std.fmt.allocPrint(sa, "{s}.self_attn.q_proj.weight", .{pfx}), &.{ D, D }),
                .q_b = try store.get(try std.fmt.allocPrint(sa, "{s}.self_attn.q_proj.bias", .{pfx}), &.{D}),
                .k_w = try store.get(try std.fmt.allocPrint(sa, "{s}.self_attn.k_proj.weight", .{pfx}), &.{ D, D }),
                .k_b = try store.get(try std.fmt.allocPrint(sa, "{s}.self_attn.k_proj.bias", .{pfx}), &.{D}),
                .v_w = try store.get(try std.fmt.allocPrint(sa, "{s}.self_attn.v_proj.weight", .{pfx}), &.{ D, D }),
                .v_b = try store.get(try std.fmt.allocPrint(sa, "{s}.self_attn.v_proj.bias", .{pfx}), &.{D}),
                .out_w = try store.get(try std.fmt.allocPrint(sa, "{s}.self_attn.out_proj.weight", .{pfx}), &.{ D, D }),
                .out_b = try store.get(try std.fmt.allocPrint(sa, "{s}.self_attn.out_proj.bias", .{pfx}), &.{D}),
                .fc_w = try store.get(try std.fmt.allocPrint(sa, "{s}.mlp.c_fc.weight", .{pfx}), &.{ M, D }),
                .fc_b = try store.get(try std.fmt.allocPrint(sa, "{s}.mlp.c_fc.bias", .{pfx}), &.{M}),
                .proj_w = try store.get(try std.fmt.allocPrint(sa, "{s}.mlp.c_proj.weight", .{pfx}), &.{ D, M }),
                .proj_b = try store.get(try std.fmt.allocPrint(sa, "{s}.mlp.c_proj.bias", .{pfx}), &.{D}),
            };
        }

        const gpu_scratch = try gpa.create(Scratch);
        gpu_scratch.* = .{};

        return .{
            .gpa = gpa,
            .ctx = ctx,
            .cfg = cfg,
            .scratch = gpu_scratch,
            .patch_emb_w = patch_emb_w,
            .class_emb = class_emb,
            .cls_pos_emb = cls_pos_emb,
            .pos_emb = pos_emb,
            .ln_pre_w = ln_pre_w,
            .ln_pre_b = ln_pre_b,
            .layers = layers,
            .proj_in_w = proj_in_w,
            .proj_out_w = proj_out_w,
        };
    }

    pub fn deinit(self: *VisionEncoder) void {
        self.scratch.deinit();
        self.gpa.destroy(self.scratch);
        self.gpa.free(self.layers);
    }

    fn dispatchLinear(
        self: *const VisionEncoder,
        x_buf: metal.Buffer,
        x_off: u32,
        w: weights.Weight,
        bias_opt: ?weights.Weight,
        y_buf: metal.Buffer,
        y_off: u32,
        ls_opt: ?weights.Weight,
        rows: u32,
        m: u32,
        n: u32,
        apply_gelu: bool,
        round_out: bool,
    ) !void {
        const b_w = bias_opt orelse w;
        const g_w = ls_opt orelse w;
        const fuse_ls = ls_opt != null and ls_opt.?.tensor.dtype == w.tensor.dtype;
        const act_a: u32 = if (apply_gelu) 4 else if (fuse_ls) 5 else 0;
        var p: gpu.Params = .{
            .w = w.tensor.offset,
            .b = if (bias_opt) |bw| bw.tensor.offset else 0,
            .s = if (fuse_ls) ls_opt.?.tensor.offset else 0,
            .x = x_off,
            .xs = n,
            .y = y_off,
            .ys = m,
            .rows = rows,
            .n = n,
            .m = m,
            .dtype = @backingInt(w.tensor.dtype),
            .bdtype = if (bias_opt) |bw| @backingInt(bw.tensor.dtype) else 0,
            .flags = (if (bias_opt != null) gpu.flag_has_weight else 0) |
                (if (round_out) gpu.flag_round_out else 0),
            .act_a = act_a,
        };
        if (w.tensor.dtype == .bf16 or w.tensor.dtype == .f16) {
            const tgs: usize = ((@as(usize, m) + 63) / 64) * ((@as(usize, rows) + 63) / 64);
            try gpu.dispatch(self.ctx, "mk_vis_linear_mma_h64x64", &.{ x_buf, w.buffer, b_w.buffer, y_buf, g_w.buffer }, &p, tgs * 256, 256);
        } else {
            const tgs: usize = ((@as(usize, m) + 31) / 32) * ((@as(usize, rows) + 31) / 32);
            try gpu.dispatch(self.ctx, "mk_vis_linear_mma32", &.{ x_buf, w.buffer, b_w.buffer, y_buf, g_w.buffer }, &p, tgs * 128, 128);
        }
        if (ls_opt != null and !fuse_ls) {
            return error.UnsupportedVisionConfig;
        }
    }

    fn dispatchLayerNorm(
        self: *const VisionEncoder,
        x_buf: metal.Buffer,
        x_off: u32,
        w: weights.Weight,
        b: weights.Weight,
        y_buf: metal.Buffer,
        y_off: u32,
        rows: u32,
        dim: u32,
    ) !void {
        var p: gpu.Params = .{
            .w = w.tensor.offset,
            .b = b.tensor.offset,
            .x = x_off,
            .xs = dim,
            .y = y_off,
            .ys = dim,
            .rows = rows,
            .n = dim,
            .dtype = @backingInt(w.tensor.dtype),
            .bdtype = @backingInt(b.tensor.dtype),
            .eps = self.cfg.layer_norm_eps,
        };
        const kernel: [:0]const u8 = if (dim <= 1024 and dim % 4 == 0 and x_off % 4 == 0 and y_off % 4 == 0)
            "mk_vis_layernorm_1024"
        else
            "mk_vis_layernorm";
        try gpu.dispatch(self.ctx, kernel, &.{ x_buf, w.buffer, b.buffer, y_buf }, &p, @as(usize, rows) * 256, 256);
    }

    fn fillRopeTable(dst: []f32, gh: u32, gw: u32, hd: u32, theta: f32) void {
        const axis_pairs = hd / 4;
        const half_dim_f: f64 = @floatFromInt(hd / 2);
        const theta_f: f64 = @floatCast(theta);
        for (0..axis_pairs) |k| {
            const inv_freq = 1.0 / std.math.pow(f64, theta_f, @as(f64, @floatFromInt(2 * k)) / half_dim_f);
            for (0..gw) |px| {
                const angle = @as(f64, @floatFromInt(px + 1)) * inv_freq;
                const off = (px * axis_pairs + k) * 2;
                dst[off + 0] = @floatCast(@cos(angle));
                dst[off + 1] = @floatCast(@sin(angle));
            }
            for (0..gh) |py| {
                const angle = @as(f64, @floatFromInt(py + 1)) * inv_freq;
                const off = ((@as(usize, gw) + py) * axis_pairs + k) * 2;
                dst[off + 0] = @floatCast(@cos(angle));
                dst[off + 1] = @floatCast(@sin(angle));
            }
        }
    }

    fn scratchLayout(self: *const VisionEncoder, dims: TargetDims) struct {
        h_off: u32,
        norm_off: u32,
        work_off: u32,
        total_words: usize,
        n_patches: u32,
        seq_len: u32,
    } {
        const c = self.cfg;
        const D = c.hidden_size;
        const M = c.mlpDim();
        const P = c.patchPixels();
        const n_patches: u32 = dims.grid_h * dims.grid_w;
        const seq_len: u32 = 1 + n_patches;
        const n_vis: u32 = dims.visual_tokens;
        const merge_dim: u32 = c.adapter_merge_size * c.adapter_merge_size * D;

        // Align every region to 4 floats for float4 vector loads.
        const seq_d = ((@as(usize, seq_len) * D + 3) / 4) * 4;
        const h_off: usize = 0;
        const norm_off: usize = seq_d;
        const work_off: usize = 2 * seq_d;
        const patch_words = ((@as(usize, n_patches) * P + 3) / 4) * 4 + ((@as(usize, n_patches) * D + 3) / 4) * 4;
        const attn_words = 4 * seq_d;
        const mlp_words = ((@as(usize, seq_len) * M + 3) / 4) * 4;
        const proj_words = ((@as(usize, n_vis) * merge_dim + 3) / 4) * 4 + ((@as(usize, n_vis) * c.projection_hidden_dim + 3) / 4) * 4;
        const work_words = @max(@max(patch_words, attn_words), @max(mlp_words, proj_words));
        return .{
            .h_off = @intCast(h_off),
            .norm_off = @intCast(norm_off),
            .work_off = @intCast(work_off),
            .total_words = work_off + work_words,
            .n_patches = n_patches,
            .seq_len = seq_len,
        };
    }

    fn encodeFromScratch(self: *const VisionEncoder, dims: TargetDims, scratch_buf: metal.Buffer) !EncodedImage {
        const c = self.cfg;
        const D = c.hidden_size;
        const M = c.mlpDim();
        const P = c.patchPixels();
        const gh = dims.grid_h;
        const gw = dims.grid_w;
        const n_vis: u32 = dims.visual_tokens;
        const merge_dim: u32 = c.adapter_merge_size * c.adapter_merge_size * D;
        const lay = self.scratchLayout(dims);
        const n_patches = lay.n_patches;
        const seq_len = lay.seq_len;
        const seq_d: u32 = @intCast(((@as(usize, seq_len) * D + 3) / 4) * 4);

        const patches_off: u32 = lay.work_off;
        const patch_proj_off: u32 = lay.work_off + @as(u32, @intCast(((@as(usize, n_patches) * P + 3) / 4) * 4));
        const q_off: u32 = lay.work_off;
        const k_off: u32 = lay.work_off + seq_d;
        const v_off: u32 = lay.work_off + 2 * seq_d;
        const att_off: u32 = lay.work_off + 3 * seq_d;
        const mlp_off: u32 = lay.work_off;
        const merged_off: u32 = lay.work_off;
        const proj_mid_off: u32 = lay.work_off + @as(u32, @intCast(((@as(usize, n_vis) * merge_dim + 3) / 4) * 4));

        const hd = c.headDim();
        const rope_words = (@as(usize, gw) + gh) * (hd / 4) * 2;
        const rope_buf = try self.scratch.ensureRope(self.ctx, rope_words);
        fillRopeTable(rope_buf.floats()[0..rope_words], gh, gw, hd, c.rope_theta);

        const out_buf = try self.ctx.alloc(@as(usize, n_vis) * c.text_hidden_size);
        errdefer out_buf.deinit();

        // 1. Patch projection: [n_patches, P] @ W^T -> [n_patches, D]
        try self.dispatchLinear(scratch_buf, patches_off, self.patch_emb_w, null, scratch_buf, patch_proj_off, null, n_patches, D, P, false, false);

        // 2. Prepend CLS + add bilinearly interpolated 2D position embeddings -> norm_off
        {
            var p: gpu.Params = .{
                .w = self.pos_emb.tensor.offset,
                .s = self.class_emb.tensor.offset,
                .b = self.cls_pos_emb.tensor.offset,
                .x = patch_proj_off,
                .xs = D,
                .y = lay.norm_off,
                .ys = D,
                .n = D,
                .heads = gh,
                .kv_heads = gw,
                .dim = c.posemb_grid_size,
                .dtype = @backingInt(self.pos_emb.tensor.dtype),
            };
            try gpu.dispatch(
                self.ctx,
                "mk_vis_embed_prep",
                &.{ scratch_buf, self.pos_emb.buffer, self.class_emb.buffer, self.cls_pos_emb.buffer, scratch_buf },
                &p,
                @as(usize, seq_len) * D,
                256,
            );
        }

        // 3. ln_pre: norm_off -> h_off
        try self.dispatchLayerNorm(scratch_buf, lay.norm_off, self.ln_pre_w, self.ln_pre_b, scratch_buf, lay.h_off, seq_len, D);

        // 4. 23 ViT transformer layers
        const att_scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(hd)));
        const use_flash64 = (hd == 64 and D % 4 == 0 and q_off % 4 == 0 and k_off % 4 == 0 and v_off % 4 == 0 and att_off % 4 == 0);
        for (self.layers) |lw| {
            // Pre-attention LayerNorm: h_off -> norm_off
            try self.dispatchLayerNorm(scratch_buf, lay.h_off, lw.ln_1_w, lw.ln_1_b, scratch_buf, lay.norm_off, seq_len, D);

            // Q, K, V projections
            try self.dispatchLinear(scratch_buf, lay.norm_off, lw.q_w, lw.q_b, scratch_buf, q_off, null, seq_len, D, D, false, false);
            try self.dispatchLinear(scratch_buf, lay.norm_off, lw.k_w, lw.k_b, scratch_buf, k_off, null, seq_len, D, D, false, false);
            try self.dispatchLinear(scratch_buf, lay.norm_off, lw.v_w, lw.v_b, scratch_buf, v_off, null, seq_len, D, D, false, false);

            // Fused in-place 2D RoPE on BOTH Q and K using precomputed (cos, sin) table
            {
                var rp: gpu.Params = .{
                    .rows = seq_len,
                    .x = q_off,
                    .xs = D,
                    .y = k_off,
                    .ys = D,
                    .heads = c.num_attention_heads,
                    .dim = hd,
                    .window = gh,
                    .capacity = gw,
                    .scale = c.rope_theta,
                };
                const rope_threads = @as(usize, seq_len) * c.num_attention_heads * (hd / 2);
                try gpu.dispatch(self.ctx, "mk_vis_rope2d_qk", &.{ scratch_buf, scratch_buf, rope_buf }, &rp, rope_threads, 256);
            }

            // Bidirectional MHA (FlashAttention tiled MMA for head_dim == 64)
            {
                var ap: gpu.Params = .{
                    .rows = seq_len,
                    .x = q_off,
                    .xs = D,
                    .y = k_off,
                    .ys = D,
                    .z = v_off,
                    .zs = D,
                    .pos = att_off,
                    .window = D,
                    .heads = c.num_attention_heads,
                    .dim = hd,
                    .scale = att_scale,
                };
                if (use_flash64) {
                    const q_blocks: usize = (seq_len + 31) / 32;
                    try gpu.dispatch(self.ctx, "mk_vis_attention_flash64", &.{ scratch_buf, scratch_buf, scratch_buf, scratch_buf }, &ap, q_blocks * c.num_attention_heads * 128, 128);
                } else {
                    const att_threads = @as(usize, seq_len) * c.num_attention_heads * 32;
                    try gpu.dispatch(self.ctx, "mk_vis_attention", &.{ scratch_buf, scratch_buf, scratch_buf, scratch_buf }, &ap, att_threads, 128);
                }
            }

            // Output projection + fused LayerScale 1 residual directly into h_off
            try self.dispatchLinear(scratch_buf, att_off, lw.out_w, lw.out_b, scratch_buf, lay.h_off, lw.ls_1, seq_len, D, D, false, false);

            // Pre-MLP LayerNorm: h_off -> norm_off
            try self.dispatchLayerNorm(scratch_buf, lay.h_off, lw.ln_2_w, lw.ln_2_b, scratch_buf, lay.norm_off, seq_len, D);

            // MLP: c_fc + GELU -> mlp_off, c_proj + fused LayerScale 2 residual directly into h_off
            try self.dispatchLinear(scratch_buf, lay.norm_off, lw.fc_w, lw.fc_b, scratch_buf, mlp_off, null, seq_len, M, D, true, false);
            try self.dispatchLinear(scratch_buf, mlp_off, lw.proj_w, lw.proj_b, scratch_buf, lay.h_off, lw.ls_2, seq_len, D, M, false, false);
        }

        // 5. Drop CLS + 2x2 spatial merge: h_off -> merged_off
        {
            var mp: gpu.Params = .{
                .x = lay.h_off,
                .xs = D,
                .y = merged_off,
                .ys = merge_dim,
                .n = D,
                .heads = gh,
                .kv_heads = gw,
                .dim = c.adapter_merge_size,
            };
            try gpu.dispatch(self.ctx, "mk_vis_spatial_merge", &.{ scratch_buf, scratch_buf }, &mp, @as(usize, n_vis) * merge_dim, 256);
        }

        // 6. Multimodal projector: proj_in + GELU -> proj_mid_off, proj_out -> out_buf (BF16 rounded)
        try self.dispatchLinear(scratch_buf, merged_off, self.proj_in_w, null, scratch_buf, proj_mid_off, null, n_vis, c.projection_hidden_dim, merge_dim, true, false);
        try self.dispatchLinear(scratch_buf, proj_mid_off, self.proj_out_w, null, out_buf, 0, null, n_vis, c.text_hidden_size, c.projection_hidden_dim, false, true);
        try self.ctx.finish();

        return .{
            .buffer = out_buf,
            .tokens = n_vis,
            .grid_h = gh / c.adapter_merge_size,
            .grid_w = gw / c.adapter_merge_size,
            .width = c.text_hidden_size,
        };
    }

    pub fn encodePreprocessed(self: *const VisionEncoder, prep: PreprocessedImage) !EncodedImage {
        const lay = self.scratchLayout(prep.dims);
        const scratch_buf = try self.scratch.ensureBuf(self.ctx, lay.total_words);
        @memcpy(scratch_buf.floats()[lay.work_off..][0..prep.patches.len], prep.patches);
        return self.encodeFromScratch(prep.dims, scratch_buf);
    }

    pub fn encodePixels(self: *const VisionEncoder, a: Allocator, pixels: []const u8, w: u32, h: u32, channels: u32, max_image_tokens: u32) !EncodedImage {
        const dims = computeTargetDims(w, h, self.cfg, max_image_tokens, default_max_image_dim);
        const lay = self.scratchLayout(dims);
        const scratch_buf = try self.scratch.ensureBuf(self.ctx, lay.total_words);
        const patch_words = @as(usize, lay.n_patches) * self.cfg.patchPixels();
        try preprocessPixelsInto(a, pixels, w, h, channels, self.cfg, dims, scratch_buf.floats()[lay.work_off..][0..patch_words]);
        return self.encodeFromScratch(dims, scratch_buf);
    }

    pub fn encodeRgb(self: *const VisionEncoder, a: Allocator, rgb: []const u8, w: u32, h: u32, max_image_tokens: u32) !EncodedImage {
        return self.encodePixels(a, rgb, w, h, 3, max_image_tokens);
    }

    pub fn encodeBytes(self: *const VisionEncoder, a: Allocator, bytes: []const u8, max_image_tokens: u32) !EncodedImage {
        const img = try decodeImageBytes(bytes);
        defer img.deinit();
        return self.encodePixels(a, img.slice(), img.width, img.height, img.channels, max_image_tokens);
    }

    pub fn encodeFile(self: *const VisionEncoder, a: Allocator, path: []const u8, max_image_tokens: u32) !EncodedImage {
        const img = try decodeImageFile(a, path);
        defer img.deinit();
        return self.encodePixels(a, img.slice(), img.width, img.height, img.channels, max_image_tokens);
    }

    pub fn encodeInput(self: *const VisionEncoder, a: Allocator, input: ImageInput, max_image_tokens: u32) !EncodedImage {
        return switch (input) {
            .source => |s| self.encodeFile(a, s, max_image_tokens),
            .bytes => |b| self.encodeBytes(a, b, max_image_tokens),
        };
    }

    /// Encode one or more image inputs (paths/URLs/data URIs or raw encoded bytes) into one contiguous GPU buffer.
    pub fn encodeInputs(self: *const VisionEncoder, a: Allocator, inputs: []const ImageInput, max_image_tokens: u32) !EncodedBatch {
        if (inputs.len == 0) return error.NoImages;
        const counts = try self.gpa.alloc(u32, inputs.len);
        errdefer self.gpa.free(counts);

        // Zero-copy fast path for the common single-image case.
        if (inputs.len == 1) {
            const e = try self.encodeInput(a, inputs[0], max_image_tokens);
            counts[0] = e.tokens;
            return .{ .buffer = e.buffer, .counts = counts, .total_tokens = e.tokens, .width = e.width };
        }

        const encoded = try a.alloc(EncodedImage, inputs.len);
        var built: usize = 0;
        errdefer for (encoded[0..built]) |e| e.deinit();
        defer for (encoded[0..built]) |e| e.deinit();

        var total: u32 = 0;
        for (inputs, 0..) |inp, i| {
            encoded[i] = try self.encodeInput(a, inp, max_image_tokens);
            built += 1;
            counts[i] = encoded[i].tokens;
            total += encoded[i].tokens;
        }
        const W = self.cfg.text_hidden_size;
        const buf = try self.ctx.alloc(@as(usize, total) * W);
        errdefer buf.deinit();
        const dst = buf.floats();
        var row: usize = 0;
        for (encoded) |e| {
            const words = @as(usize, e.tokens) * W;
            @memcpy(dst[row * W ..][0..words], e.buffer.floats()[0..words]);
            row += e.tokens;
        }
        return .{ .buffer = buf, .counts = counts, .total_tokens = total, .width = W };
    }

    /// Encode one or more image files/URIs and pack their visual embeddings into one contiguous GPU buffer.
    pub fn encodeFiles(self: *const VisionEncoder, a: Allocator, paths: []const []const u8, max_image_tokens: u32) !EncodedBatch {
        const inputs = try a.alloc(ImageInput, paths.len);
        defer a.free(inputs);
        for (paths, 0..) |p, i| inputs[i] = .{ .source = p };
        return self.encodeInputs(a, inputs, max_image_tokens);
    }
};

// ---- Unit tests -------------------------------------------------------------

fn cpuErf(x_in: f64) f64 {
    const sign: f64 = if (x_in < 0) -1.0 else 1.0;
    const x = @abs(x_in);
    const t = 1.0 / (1.0 + 0.3275911 * x);
    const poly = (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t;
    return sign * (1.0 - poly * @exp(-x * x));
}

fn cpuGelu(x: f64) f64 {
    return 0.5 * x * (1.0 + cpuErf(x * std.math.sqrt1_2));
}

test "parseConfig accepts Inkling 2 Mini vision_config and computes dimensions" {
    const cfg_json =
        \\{"model_type":"inkling2_loop","vision_config":{
        \\"model_type":"inkling2_vision","text_hidden_size":2304,"temporal_patch_size":2,
        \\"native_patch_size":20,"adapter_merge_size":2,"projection_hidden_dim":4096,
        \\"projector_bias":false,"hidden_size":1024,"num_hidden_layers":23,
        \\"num_attention_heads":16,"mlp_ratio":4.0,"posemb_grid_size":32,"layer_norm_eps":1e-5}}
    ;
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const cfg = try parseConfig(arena.allocator(), cfg_json);
    try std.testing.expectEqual(@as(u32, 2304), cfg.text_hidden_size);
    try std.testing.expectEqual(@as(u32, 64), cfg.headDim());
    try std.testing.expectEqual(@as(u32, 4096), cfg.mlpDim());
    try std.testing.expectEqual(@as(u32, 1200), cfg.patchPixels());
    try std.testing.expectEqual(@as(u32, 40), cfg.tilePixels());

    // Exact multiple of 40 -> no resize
    const d1 = computeTargetDims(80, 120, cfg, 1024, 4096);
    try std.testing.expectEqual(@as(u32, 80), d1.padded_w);
    try std.testing.expectEqual(@as(u32, 120), d1.padded_h);
    try std.testing.expectEqual(@as(u32, 6), d1.visual_tokens);
    try std.testing.expectEqual(@as(u32, 6), d1.grid_h);
    try std.testing.expectEqual(@as(u32, 4), d1.grid_w);

    // Non-multiple of 40 -> padded right/bottom
    const d2 = computeTargetDims(50, 30, cfg, 1024, 4096);
    try std.testing.expectEqual(@as(u32, 50), d2.resized_w);
    try std.testing.expectEqual(@as(u32, 30), d2.resized_h);
    try std.testing.expectEqual(@as(u32, 80), d2.padded_w);
    try std.testing.expectEqual(@as(u32, 40), d2.padded_h);
    try std.testing.expectEqual(@as(u32, 2), d2.visual_tokens);

    // Large image respects max_image_tokens budget
    const d3 = computeTargetDims(3840, 2160, cfg, 256, 4096);
    try std.testing.expect(d3.visual_tokens <= 256);
    try std.testing.expect(d3.padded_w % 40 == 0 and d3.padded_h % 40 == 0);
}

test "vision Metal kernels match CPU float64 reference" {
    const gpa = std.testing.allocator;
    const ctx = try gpu.open();
    defer ctx.deinit();

    // 1. mk_vis_linear_mma (with bias + GELU) on non-multiple-of-32 dims (rows=5, N=12, M=19)
    {
        const R: u32 = 5;
        const N: u32 = 12;
        const M: u32 = 19;
        const x = try ctx.alloc(R * N);
        defer x.deinit();
        const w = try ctx.alloc(M * N);
        defer w.deinit();
        const b = try ctx.alloc(M);
        defer b.deinit();
        const y = try ctx.alloc(R * M);
        defer y.deinit();
        for (x.floats(), 0..) |*v, i| v.* = @as(f32, @floatFromInt(@as(i32, @intCast(i % 13)) - 6)) * 0.25;
        for (w.floats(), 0..) |*v, i| v.* = @as(f32, @floatFromInt(@as(i32, @intCast((i * 7) % 11)) - 5)) * 0.15;
        for (b.floats(), 0..) |*v, i| v.* = @as(f32, @floatFromInt(@as(i32, @intCast(i % 5)) - 2)) * 0.1;

        var p: gpu.Params = .{ .rows = R, .xs = N, .ys = M, .n = N, .m = M, .dtype = 0, .bdtype = 0, .flags = gpu.flag_has_weight, .act_a = 4 };
        try gpu.dispatch(ctx, "mk_vis_linear_mma", &.{ x, w, b, y }, &p, 128, 128);
        try ctx.finish();

        for (0..R) |r| {
            for (0..M) |m_i| {
                var acc: f64 = b.floats()[m_i];
                for (0..N) |k| acc += @as(f64, x.floats()[r * N + k]) * w.floats()[m_i * N + k];
                const expected: f32 = @floatCast(cpuGelu(acc));
                try std.testing.expectApproxEqAbs(expected, y.floats()[r * M + m_i], 1e-4);
            }
        }
    }

    // 2. mk_vis_layernorm
    {
        const R: u32 = 3;
        const D: u32 = 32;
        const x = try ctx.alloc(R * D);
        defer x.deinit();
        const g = try ctx.alloc(D);
        defer g.deinit();
        const b = try ctx.alloc(D);
        defer b.deinit();
        const y = try ctx.alloc(R * D);
        defer y.deinit();
        for (x.floats(), 0..) |*v, i| v.* = @as(f32, @floatFromInt(@as(i32, @intCast(i % 17)) - 8)) * 0.3;
        for (g.floats(), 0..) |*v, i| v.* = 1.0 + @as(f32, @floatFromInt(i % 5)) * 0.1;
        for (b.floats(), 0..) |*v, i| v.* = @as(f32, @floatFromInt(@as(i32, @intCast(i % 7)) - 3)) * 0.05;

        var p: gpu.Params = .{ .rows = R, .xs = D, .ys = D, .n = D, .dtype = 0, .bdtype = 0, .eps = 1e-5 };
        try gpu.dispatch(ctx, "mk_vis_layernorm", &.{ x, g, b, y }, &p, R * 256, 256);
        try ctx.finish();

        for (0..R) |r| {
            var mean: f64 = 0;
            for (0..D) |c| mean += x.floats()[r * D + c];
            mean /= D;
            var vr: f64 = 0;
            for (0..D) |c| {
                const d = x.floats()[r * D + c] - mean;
                vr += d * d;
            }
            const inv = 1.0 / @sqrt(vr / D + 1e-5);
            for (0..D) |c| {
                const exp: f32 = @floatCast((x.floats()[r * D + c] - mean) * inv * g.floats()[c] + b.floats()[c]);
                try std.testing.expectApproxEqAbs(exp, y.floats()[r * D + c], 1e-5);
            }
        }
    }

    // 3. mk_vis_rope2d + mk_vis_attention + mk_vis_spatial_merge + mk_scatter_embeds
    {
        const gh: u32 = 2;
        const gw: u32 = 2;
        const S: u32 = 1 + gh * gw;
        const heads: u32 = 2;
        const hd: u32 = 8;
        const D: u32 = heads * hd;

        const q = try ctx.alloc(S * D);
        defer q.deinit();
        const k = try ctx.alloc(S * D);
        defer k.deinit();
        const v = try ctx.alloc(S * D);
        defer v.deinit();
        const out = try ctx.alloc(S * D);
        defer out.deinit();
        for (q.floats(), 0..) |*val, i| val.* = @as(f32, @floatFromInt(@as(i32, @intCast(i % 9)) - 4)) * 0.2;
        for (k.floats(), 0..) |*val, i| val.* = @as(f32, @floatFromInt(@as(i32, @intCast((i * 3) % 9)) - 4)) * 0.2;
        for (v.floats(), 0..) |*val, i| val.* = @as(f32, @floatFromInt(@as(i32, @intCast((i * 5) % 11)) - 5)) * 0.1;

        const q_cls_before = try gpa.dupe(f32, q.floats()[0..D]);
        defer gpa.free(q_cls_before);
        const q_patch1_before = try gpa.dupe(f32, q.floats()[2 * D .. 3 * D]); // patch idx 1 => (py=0, px=1) => (row=1, col=2)
        defer gpa.free(q_patch1_before);

        var rp: gpu.Params = .{ .rows = S, .xs = D, .heads = heads, .dim = hd, .window = gh, .capacity = gw, .scale = 10000.0 };
        try gpu.dispatch(ctx, "mk_vis_rope2d", &.{q}, &rp, S * heads * (hd / 2), 64);
        try gpu.dispatch(ctx, "mk_vis_rope2d", &.{k}, &rp, S * heads * (hd / 2), 64);
        var ap: gpu.Params = .{ .rows = S, .xs = D, .ys = D, .zs = D, .window = D, .heads = heads, .dim = hd, .scale = 1.0 / @sqrt(@as(f32, @floatFromInt(hd))) };
        try gpu.dispatch(ctx, "mk_vis_attention", &.{ q, k, v, out }, &ap, S * heads * 32, 128);
        try ctx.finish();

        // CLS row (row 0) must be unrotated by 2D RoPE
        try std.testing.expectEqualSlices(f32, q_cls_before, q.floats()[0..D]);

        // Patch 1 (py=0, px=1) rotates first half (pair 0) by col+1=2.0, second half (pair 2) by row+1=1.0
        {
            const a0 = q_patch1_before[0];
            const b0 = q_patch1_before[1];
            try std.testing.expectApproxEqAbs(a0 * @cos(@as(f32, 2.0)) - b0 * @sin(@as(f32, 2.0)), q.floats()[2 * D + 0], 1e-5);
            try std.testing.expectApproxEqAbs(a0 * @sin(@as(f32, 2.0)) + b0 * @cos(@as(f32, 2.0)), q.floats()[2 * D + 1], 1e-5);
            const a2 = q_patch1_before[4];
            const b2 = q_patch1_before[5];
            try std.testing.expectApproxEqAbs(a2 * @cos(@as(f32, 1.0)) - b2 * @sin(@as(f32, 1.0)), q.floats()[2 * D + 4], 1e-5);
            try std.testing.expectApproxEqAbs(a2 * @sin(@as(f32, 1.0)) + b2 * @cos(@as(f32, 1.0)), q.floats()[2 * D + 5], 1e-5);
        }

        // Verify bidirectional attention vs CPU reference
        for (0..S) |r| {
            for (0..heads) |h_i| {
                var scores: [5]f64 = undefined;
                var m: f64 = -std.math.inf(f64);
                for (0..S) |j| {
                    var dot: f64 = 0;
                    for (0..hd) |d_i| dot += @as(f64, q.floats()[r * D + h_i * hd + d_i]) * k.floats()[j * D + h_i * hd + d_i];
                    scores[j] = dot / @sqrt(@as(f64, @floatFromInt(hd)));
                    m = @max(m, scores[j]);
                }
                var denom: f64 = 0;
                for (0..S) |j| {
                    scores[j] = @exp(scores[j] - m);
                    denom += scores[j];
                }
                for (0..hd) |d_i| {
                    var acc: f64 = 0;
                    for (0..S) |j| acc += (scores[j] / denom) * v.floats()[j * D + h_i * hd + d_i];
                    try std.testing.expectApproxEqAbs(@as(f32, @floatCast(acc)), out.floats()[r * D + h_i * hd + d_i], 1e-5);
                }
            }
        }

        // Spatial merge: [1 + 2*2, D] -> [1, 4*D], dropping CLS at row 0
        const merged = try ctx.alloc(4 * D);
        defer merged.deinit();
        var mp: gpu.Params = .{ .xs = D, .ys = 4 * D, .n = D, .heads = gh, .kv_heads = gw, .dim = 2 };
        try gpu.dispatch(ctx, "mk_vis_spatial_merge", &.{ out, merged }, &mp, 4 * D, 64);
        try ctx.finish();
        for (0..4) |sub| {
            try std.testing.expectEqualSlices(f32, out.floats()[(1 + sub) * D ..][0..D], merged.floats()[sub * D ..][0..D]);
        }

        // Scatter embeds into arena
        const arena_buf = try ctx.alloc(4 * D);
        defer arena_buf.deinit();
        @memset(arena_buf.floats(), 99.0);
        const vis_buf = try ctx.alloc(2 * D);
        defer vis_buf.deinit();
        for (vis_buf.floats(), 0..) |*val, i| val.* = @as(f32, @floatFromInt(i + 1));
        const row_map = try ctx.alloc(4);
        defer row_map.deinit();
        @memcpy(row_map.structs(i32), &[_]i32{ -1, 0, 1, -1 });
        var sp: gpu.Params = .{ .rows = 4, .n = D, .xs = D, .ys = D };
        try gpu.dispatch(ctx, "mk_scatter_embeds", &.{ arena_buf, vis_buf, row_map }, &sp, 4 * D, 64);
        try ctx.finish();
        try std.testing.expectEqual(@as(f32, 99.0), arena_buf.floats()[0]);
        try std.testing.expectEqualSlices(f32, vis_buf.floats()[0..D], arena_buf.floats()[D .. 2 * D]);
        try std.testing.expectEqualSlices(f32, vis_buf.floats()[D .. 2 * D], arena_buf.floats()[2 * D .. 3 * D]);
        try std.testing.expectEqual(@as(f32, 99.0), arena_buf.floats()[3 * D]);
    }
}

test "inkling_vision decodes file:// URIs, data: base64 URIs, and raw PNG bytes" {
    const a = std.testing.allocator;
    const p1 = try decodeFileUri(a, "file:///tmp/hello%20world/img%231.png");
    defer a.free(p1);
    try std.testing.expectEqualStrings("/tmp/hello world/img#1.png", p1);

    const p2 = try decodeFileUri(a, "file://localhost/var/img.jpg");
    defer a.free(p2);
    try std.testing.expectEqualStrings("/var/img.jpg", p2);

    try std.testing.expectError(error.InvalidImageUri, decodeFileUri(a, "http://example.com/a.png"));
    try std.testing.expectError(error.InvalidImageUri, decodeDataUriBytes(a, "data:image/png,notbase64"));

    // 1x1 opaque red PNG (#FF0000), decoded zero-copy as RGBA8888
    const red_png_uri = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGP4z8AAAAMBAQDJ/pLvAAAAAElFTkSuQmCC";
    const img = try decodeImageFile(a, red_png_uri);
    defer img.deinit();
    try std.testing.expectEqual(@as(u32, 1), img.width);
    try std.testing.expectEqual(@as(u32, 1), img.height);
    try std.testing.expectEqual(@as(u32, 4), img.channels);
    try std.testing.expectEqualSlices(u8, &.{ 255, 0, 0, 255 }, img.slice());

    const raw_bytes = try decodeDataUriBytes(a, red_png_uri);
    defer a.free(raw_bytes);
    const img2 = try decodeImageBytes(raw_bytes);
    defer img2.deinit();
    try std.testing.expectEqual(@as(u32, 1), img2.width);
    try std.testing.expectEqual(@as(u32, 1), img2.height);
    try std.testing.expectEqual(@as(u32, 4), img2.channels);
    try std.testing.expectEqualSlices(u8, &.{ 255, 0, 0, 255 }, img2.slice());
}

test "fast vision kernels (flash64, mma_h64x64+layerscale, mma32, layernorm_1024, rope2d_qk, lanczos4) match reference" {
    const alloc = std.testing.allocator;
    const ctx = try gpu.open();
    defer ctx.deinit();

    var prng = std.Random.DefaultPrng.init(20261128);
    const rnd = prng.random();
    const toBf16 = struct {
        fn f(v: f32) u16 {
            const bits: u32 = @bitCast(v);
            return @truncate((bits + 0x7fff + ((bits >> 16) & 1)) >> 16);
        }
        fn back(u: u16) f32 {
            return @bitCast(@as(u32, u) << 16);
        }
    };

    // 0. Precomputed SIMD Lanczos3 RGB vs RGBA parity on non-square resize
    {
        const sw: u32 = 137;
        const sh: u32 = 89;
        const dw: u32 = 60;
        const dh: u32 = 40;
        const rgb = try alloc.alloc(u8, sw * sh * 3);
        defer alloc.free(rgb);
        const rgba = try alloc.alloc(u8, sw * sh * 4);
        defer alloc.free(rgba);
        for (0..sw * sh) |i| {
            const r: u8 = @intCast((i * 73 + 11) % 256);
            const g: u8 = @intCast((i * 151 + 29) % 256);
            const b: u8 = @intCast((i * 211 + 53) % 256);
            rgb[i * 3 + 0] = r;
            rgb[i * 3 + 1] = g;
            rgb[i * 3 + 2] = b;
            rgba[i * 4 + 0] = r;
            rgba[i * 4 + 1] = g;
            rgba[i * 4 + 2] = b;
            rgba[i * 4 + 3] = 255;
        }
        const out_rgb = try resizeLanczos3Vec4(alloc, rgb, sw, sh, 3, dw, dh);
        defer alloc.free(out_rgb);
        const out_rgba = try resizeLanczos3Vec4(alloc, rgba, sw, sh, 4, dw, dh);
        defer alloc.free(out_rgba);
        try std.testing.expectEqualSlices(Vec4, out_rgb, out_rgba);
    }

    // 1. mk_vis_layernorm_1024 (256 rows x D=1024, BF16 weight+bias) vs CPU f64 + determinism check
    {
        const rows: u32 = 256;
        const D: u32 = 1024;
        const x = try ctx.alloc(rows * D);
        defer x.deinit();
        const y = try ctx.alloc(rows * D);
        defer y.deinit();
        const y2 = try ctx.alloc(rows * D);
        defer y2.deinit();
        const w = try ctx.alloc(D / 2);
        defer w.deinit();
        const b = try ctx.alloc(D / 2);
        defer b.deinit();
        for (x.floats()) |*v| v.* = rnd.floatNorm(f32);
        const wu = std.mem.bytesAsSlice(u16, std.mem.sliceAsBytes(w.floats()));
        const bu = std.mem.bytesAsSlice(u16, std.mem.sliceAsBytes(b.floats()));
        for (wu) |*v| v.* = toBf16.f(0.8 + 0.4 * rnd.float(f32));
        for (bu) |*v| v.* = toBf16.f(0.2 * rnd.floatNorm(f32));
        var lp: gpu.Params = .{ .rows = rows, .xs = D, .ys = D, .n = D, .dtype = 2, .bdtype = 2, .eps = 1e-5 };
        try gpu.dispatch(ctx, "mk_vis_layernorm_1024", &.{ x, w, b, y }, &lp, rows * 256, 256);
        try gpu.dispatch(ctx, "mk_vis_layernorm_1024", &.{ x, w, b, y2 }, &lp, rows * 256, 256);
        try ctx.finish();
        try std.testing.expectEqualSlices(f32, y.floats(), y2.floats());
        for ([_]usize{ 0, 1, 63, 128, 255 }) |r| {
            var mean: f64 = 0;
            for (0..D) |c| mean += x.floats()[r * D + c];
            mean /= D;
            var vsum: f64 = 0;
            for (0..D) |c| {
                const d = x.floats()[r * D + c] - mean;
                vsum += d * d;
            }
            const inv = 1.0 / @sqrt(vsum / D + 1e-5);
            for (0..D) |c| {
                const exp_v: f32 = @floatCast((x.floats()[r * D + c] - mean) * inv * toBf16.back(wu[c]) + toBf16.back(bu[c]));
                try std.testing.expectApproxEqAbs(exp_v, y.floats()[r * D + c], 1e-4);
            }
        }
    }

    // 2. mk_vis_linear_mma_h64x64 + mk_vis_linear_mma32 (non-multiple-of-64 R=71, M=130, N=132, unaligned BF16 + LayerScale)
    {
        const R: u32 = 71;
        const M: u32 = 130;
        const N: u32 = 132;
        const x = try ctx.alloc(R * N);
        defer x.deinit();
        const w = try ctx.alloc((M * N + 2) / 2);
        defer w.deinit();
        const b = try ctx.alloc((M + 2) / 2);
        defer b.deinit();
        const g = try ctx.alloc((M + 2) / 2);
        defer g.deinit();
        const y = try ctx.alloc(R * M);
        defer y.deinit();
        const y32 = try ctx.alloc(R * M);
        defer y32.deinit();
        const h0 = try alloc.alloc(f32, R * M);
        defer alloc.free(h0);

        for (x.floats()) |*v| v.* = 0.25 * rnd.floatNorm(f32);
        // Deliberately use byte_off = 2 (unaligned to 8 bytes) to exercise real safetensors BF16 alignment!
        const wu = std.mem.bytesAsSlice(u16, std.mem.sliceAsBytes(w.floats())[2..]);
        const bu = std.mem.bytesAsSlice(u16, std.mem.sliceAsBytes(b.floats())[2..]);
        const gu = std.mem.bytesAsSlice(u16, std.mem.sliceAsBytes(g.floats())[2..]);
        for (0..M * N) |i| wu[i] = toBf16.f(0.15 * rnd.floatNorm(f32));
        for (0..M) |i| {
            bu[i] = toBf16.f(0.1 * rnd.floatNorm(f32));
            gu[i] = toBf16.f(0.5 + 0.5 * rnd.float(f32));
        }
        for (h0, y.floats(), y32.floats()) |*src, *dst, *dst32| {
            const v = 0.3 * rnd.floatNorm(f32);
            src.* = v;
            dst.* = v;
            dst32.* = v;
        }
        var mp: gpu.Params = .{ .w = 2, .b = 2, .s = 2, .rows = R, .xs = N, .ys = M, .n = N, .m = M, .dtype = 2, .bdtype = 2, .flags = gpu.flag_has_weight, .act_a = 5 };
        const tgs64: usize = ((M + 63) / 64) * ((R + 63) / 64);
        const tgs32: usize = ((M + 31) / 32) * ((R + 31) / 32);
        try gpu.dispatch(ctx, "mk_vis_linear_mma_h64x64", &.{ x, w, b, y, g }, &mp, tgs64 * 256, 256);
        try gpu.dispatch(ctx, "mk_vis_linear_mma32", &.{ x, w, b, y32, g }, &mp, tgs32 * 128, 128);
        try ctx.finish();

        for ([_]usize{ 0, 17, 63, 64, 70 }) |r| {
            for ([_]usize{ 0, 31, 64, 127, 129 }) |m| {
                var acc: f64 = 0;
                for (0..N) |k| acc += @as(f64, x.floats()[r * N + k]) * @as(f64, toBf16.back(wu[m * N + k]));
                acc += toBf16.back(bu[m]);
                const expected: f32 = @floatCast(@as(f64, h0[r * M + m]) + @as(f64, toBf16.back(gu[m])) * acc);
                try std.testing.expectApproxEqAbs(expected, y.floats()[r * M + m], 3e-3);
                try std.testing.expectApproxEqAbs(expected, y32.floats()[r * M + m], 1e-3);
            }
        }
    }

    // 3. mk_vis_rope2d_qk (precomputed table) vs mk_vis_rope2d reference
    {
        const gh: u32 = 5;
        const gw: u32 = 7;
        const S: u32 = 1 + gh * gw;
        const heads: u32 = 4;
        const hd: u32 = 64;
        const D: u32 = heads * hd;
        const q_ref = try ctx.alloc(S * D);
        defer q_ref.deinit();
        const k_ref = try ctx.alloc(S * D);
        defer k_ref.deinit();
        const q_fast = try ctx.alloc(S * D);
        defer q_fast.deinit();
        const k_fast = try ctx.alloc(S * D);
        defer k_fast.deinit();
        for (q_ref.floats(), q_fast.floats()) |*a, *b| {
            const v = rnd.floatNorm(f32);
            a.* = v;
            b.* = v;
        }
        for (k_ref.floats(), k_fast.floats()) |*a, *b| {
            const v = rnd.floatNorm(f32);
            a.* = v;
            b.* = v;
        }
        const rope_words = (@as(usize, gw) + gh) * (hd / 4) * 2;
        const rope_buf = try ctx.alloc(rope_words);
        defer rope_buf.deinit();
        VisionEncoder.fillRopeTable(rope_buf.floats(), gh, gw, hd, 10000.0);

        var rp: gpu.Params = .{ .rows = S, .xs = D, .ys = D, .heads = heads, .dim = hd, .window = gh, .capacity = gw, .scale = 10000.0 };
        const threads = @as(usize, S) * heads * (hd / 2);
        try gpu.dispatch(ctx, "mk_vis_rope2d", &.{q_ref}, &rp, threads, 256);
        try gpu.dispatch(ctx, "mk_vis_rope2d", &.{k_ref}, &rp, threads, 256);
        try gpu.dispatch(ctx, "mk_vis_rope2d_qk", &.{ q_fast, k_fast, rope_buf }, &rp, threads, 256);
        try ctx.finish();

        for (q_ref.floats(), q_fast.floats()) |ev, av| try std.testing.expectApproxEqAbs(ev, av, 1e-5);
        for (k_ref.floats(), k_fast.floats()) |ev, av| try std.testing.expectApproxEqAbs(ev, av, 1e-5);
    }

    // 4. mk_vis_attention_flash64 (D=64, heads=4, non-multiple-of-32 S=71) vs mk_vis_attention reference
    {
        const S: u32 = 71;
        const heads: u32 = 4;
        const hd: u32 = 64;
        const D: u32 = heads * hd;
        const q = try ctx.alloc(S * D);
        defer q.deinit();
        const k = try ctx.alloc(S * D);
        defer k.deinit();
        const v = try ctx.alloc(S * D);
        defer v.deinit();
        const out_ref = try ctx.alloc(S * D);
        defer out_ref.deinit();
        const out_flash = try ctx.alloc(S * D);
        defer out_flash.deinit();
        for (q.floats()) |*x| x.* = 0.5 * rnd.floatNorm(f32);
        for (k.floats()) |*x| x.* = 0.5 * rnd.floatNorm(f32);
        for (v.floats()) |*x| x.* = 0.5 * rnd.floatNorm(f32);
        @memset(out_ref.floats(), 0);
        @memset(out_flash.floats(), 0);

        var ap: gpu.Params = .{
            .rows = S,
            .xs = D,
            .ys = D,
            .zs = D,
            .window = D,
            .heads = heads,
            .dim = hd,
            .scale = 1.0 / @sqrt(@as(f32, @floatFromInt(hd))),
        };
        try gpu.dispatch(ctx, "mk_vis_attention", &.{ q, k, v, out_ref }, &ap, S * heads * 32, 128);
        const q_blocks: usize = (S + 31) / 32;
        try gpu.dispatch(ctx, "mk_vis_attention_flash64", &.{ q, k, v, out_flash }, &ap, q_blocks * heads * 128, 128);
        try ctx.finish();

        for (out_ref.floats(), out_flash.floats()) |ev, av| {
            try std.testing.expect(std.math.isFinite(av));
            try std.testing.expectApproxEqAbs(ev, av, 1e-3);
        }
    }

    // 5. Full aliased-scratch 1-layer D=1024, heads=16, hd=64, M=4096 comparison (non-square gh=4, gw=6)
    {
        const gh: u32 = 4;
        const gw: u32 = 6;
        const n_patches: u32 = gh * gw;
        const S: u32 = 1 + n_patches;
        const D: u32 = 1024;
        const M: u32 = 4096;
        const heads: u32 = 16;
        const hd: u32 = 64;
        const seq_d: u32 = S * D;

        // Allocate BF16 weights for 1 full layer
        const allocBf16 = struct {
            fn f(c: metal.Context, r_gen: std.Random, count: usize, scale: f32) !metal.Buffer {
                const buf = try c.alloc((count + 1) / 2);
                const u = std.mem.bytesAsSlice(u16, std.mem.sliceAsBytes(buf.floats()));
                for (0..count) |i| u[i] = toBf16.f(scale * r_gen.floatNorm(f32));
                return buf;
            }
        }.f;

        const ln1_w = try allocBf16(ctx, rnd, D, 1.0);
        defer ln1_w.deinit();
        const ln1_b = try allocBf16(ctx, rnd, D, 0.1);
        defer ln1_b.deinit();
        const q_w = try allocBf16(ctx, rnd, D * D, 0.05);
        defer q_w.deinit();
        const q_b = try allocBf16(ctx, rnd, D, 0.05);
        defer q_b.deinit();
        const k_w = try allocBf16(ctx, rnd, D * D, 0.05);
        defer k_w.deinit();
        const k_b = try allocBf16(ctx, rnd, D, 0.05);
        defer k_b.deinit();
        const v_w = try allocBf16(ctx, rnd, D * D, 0.05);
        defer v_w.deinit();
        const v_b = try allocBf16(ctx, rnd, D, 0.05);
        defer v_b.deinit();
        const out_w = try allocBf16(ctx, rnd, D * D, 0.05);
        defer out_w.deinit();
        const out_b = try allocBf16(ctx, rnd, D, 0.05);
        defer out_b.deinit();
        const ls1 = try allocBf16(ctx, rnd, D, 0.5);
        defer ls1.deinit();
        const ln2_w = try allocBf16(ctx, rnd, D, 1.0);
        defer ln2_w.deinit();
        const ln2_b = try allocBf16(ctx, rnd, D, 0.1);
        defer ln2_b.deinit();
        const fc_w = try allocBf16(ctx, rnd, M * D, 0.04);
        defer fc_w.deinit();
        const fc_b = try allocBf16(ctx, rnd, M, 0.05);
        defer fc_b.deinit();
        const proj_w = try allocBf16(ctx, rnd, D * M, 0.04);
        defer proj_w.deinit();
        const proj_b = try allocBf16(ctx, rnd, D, 0.05);
        defer proj_b.deinit();
        const ls2 = try allocBf16(ctx, rnd, D, 0.5);
        defer ls2.deinit();

        // Reference path (separate buffers, original kernels)
        const h_ref = try ctx.alloc(seq_d);
        defer h_ref.deinit();
        const norm_ref = try ctx.alloc(seq_d);
        defer norm_ref.deinit();
        const q_ref = try ctx.alloc(seq_d);
        defer q_ref.deinit();
        const k_ref = try ctx.alloc(seq_d);
        defer k_ref.deinit();
        const v_ref = try ctx.alloc(seq_d);
        defer v_ref.deinit();
        const att_ref = try ctx.alloc(seq_d);
        defer att_ref.deinit();
        const mlp_ref = try ctx.alloc(S * M);
        defer mlp_ref.deinit();
        for (h_ref.floats()) |*x| x.* = 0.3 * rnd.floatNorm(f32);

        // Aliased scratch path (new kernels)
        const scratch = try ctx.alloc(2 * seq_d + @max(4 * seq_d, S * M));
        defer scratch.deinit();
        @memcpy(scratch.floats()[0..seq_d], h_ref.floats());

        const rope_words = (@as(usize, gw) + gh) * (hd / 4) * 2;
        const rope_buf = try ctx.alloc(rope_words);
        defer rope_buf.deinit();
        VisionEncoder.fillRopeTable(rope_buf.floats(), gh, gw, hd, 10000.0);

        // Run Reference Layer
        {
            var lp: gpu.Params = .{ .rows = S, .xs = D, .ys = D, .n = D, .dtype = 2, .bdtype = 2, .eps = 1e-5 };
            try gpu.dispatch(ctx, "mk_vis_layernorm", &.{ h_ref, ln1_w, ln1_b, norm_ref }, &lp, S * 256, 256);
            var mm: gpu.Params = .{ .rows = S, .xs = D, .ys = D, .n = D, .m = D, .dtype = 2, .bdtype = 2, .flags = gpu.flag_has_weight };
            const tgs32: usize = ((D + 31) / 32) * ((S + 31) / 32);
            try gpu.dispatch(ctx, "mk_vis_linear_mma", &.{ norm_ref, q_w, q_b, q_ref }, &mm, tgs32 * 128, 128);
            try gpu.dispatch(ctx, "mk_vis_linear_mma", &.{ norm_ref, k_w, k_b, k_ref }, &mm, tgs32 * 128, 128);
            try gpu.dispatch(ctx, "mk_vis_linear_mma", &.{ norm_ref, v_w, v_b, v_ref }, &mm, tgs32 * 128, 128);
            var rp: gpu.Params = .{ .rows = S, .xs = D, .heads = heads, .dim = hd, .window = gh, .capacity = gw, .scale = 10000.0 };
            try gpu.dispatch(ctx, "mk_vis_rope2d", &.{q_ref}, &rp, S * heads * (hd / 2), 256);
            try gpu.dispatch(ctx, "mk_vis_rope2d", &.{k_ref}, &rp, S * heads * (hd / 2), 256);
            var ap: gpu.Params = .{ .rows = S, .xs = D, .ys = D, .zs = D, .window = D, .heads = heads, .dim = hd, .scale = 1.0 / @sqrt(@as(f32, @floatFromInt(hd))) };
            try gpu.dispatch(ctx, "mk_vis_attention", &.{ q_ref, k_ref, v_ref, att_ref }, &ap, S * heads * 32, 128);
            try gpu.dispatch(ctx, "mk_vis_linear_mma", &.{ att_ref, out_w, out_b, norm_ref }, &mm, tgs32 * 128, 128);
            var lsp: gpu.Params = .{ .rows = S, .xs = D, .ys = D, .n = D, .dtype = 2 };
            try gpu.dispatch(ctx, "mk_vis_layerscale_add", &.{ h_ref, norm_ref, ls1 }, &lsp, seq_d, 256);
            try gpu.dispatch(ctx, "mk_vis_layernorm", &.{ h_ref, ln2_w, ln2_b, norm_ref }, &lp, S * 256, 256);
            var mm_fc: gpu.Params = .{ .rows = S, .xs = D, .ys = M, .n = D, .m = M, .dtype = 2, .bdtype = 2, .flags = gpu.flag_has_weight, .act_a = 4 };
            try gpu.dispatch(ctx, "mk_vis_linear_mma", &.{ norm_ref, fc_w, fc_b, mlp_ref }, &mm_fc, ((M + 31) / 32) * ((S + 31) / 32) * 128, 128);
            var mm_pr: gpu.Params = .{ .rows = S, .xs = M, .ys = D, .n = M, .m = D, .dtype = 2, .bdtype = 2, .flags = gpu.flag_has_weight };
            try gpu.dispatch(ctx, "mk_vis_linear_mma", &.{ mlp_ref, proj_w, proj_b, att_ref }, &mm_pr, tgs32 * 128, 128);
            try gpu.dispatch(ctx, "mk_vis_layerscale_add", &.{ h_ref, att_ref, ls2 }, &lsp, seq_d, 256);
            try ctx.finish();
        }

        // Run New Aliased-Scratch Fast Layer
        {
            const h_off: u32 = 0;
            const norm_off: u32 = seq_d;
            const work_off: u32 = 2 * seq_d;
            const q_off = work_off;
            const k_off = work_off + seq_d;
            const v_off = work_off + 2 * seq_d;
            const att_off = work_off + 3 * seq_d;
            const mlp_off = work_off;
            var lp: gpu.Params = .{ .x = h_off, .xs = D, .y = norm_off, .ys = D, .rows = S, .n = D, .dtype = 2, .bdtype = 2, .eps = 1e-5 };
            try gpu.dispatch(ctx, "mk_vis_layernorm_1024", &.{ scratch, ln1_w, ln1_b, scratch }, &lp, S * 256, 256);
            const tgs64_d: usize = ((D + 63) / 64) * ((S + 63) / 64);
            var mq: gpu.Params = .{ .x = norm_off, .xs = D, .y = q_off, .ys = D, .rows = S, .n = D, .m = D, .dtype = 2, .bdtype = 2, .flags = gpu.flag_has_weight };
            try gpu.dispatch(ctx, "mk_vis_linear_mma_h64x64", &.{ scratch, q_w, q_b, scratch, q_w }, &mq, tgs64_d * 256, 256);
            mq.y = k_off;
            try gpu.dispatch(ctx, "mk_vis_linear_mma_h64x64", &.{ scratch, k_w, k_b, scratch, k_w }, &mq, tgs64_d * 256, 256);
            mq.y = v_off;
            try gpu.dispatch(ctx, "mk_vis_linear_mma_h64x64", &.{ scratch, v_w, v_b, scratch, v_w }, &mq, tgs64_d * 256, 256);
            var rp: gpu.Params = .{ .rows = S, .x = q_off, .xs = D, .y = k_off, .ys = D, .heads = heads, .dim = hd, .window = gh, .capacity = gw, .scale = 10000.0 };
            try gpu.dispatch(ctx, "mk_vis_rope2d_qk", &.{ scratch, scratch, rope_buf }, &rp, S * heads * (hd / 2), 256);
            var ap: gpu.Params = .{ .rows = S, .x = q_off, .xs = D, .y = k_off, .ys = D, .z = v_off, .zs = D, .pos = att_off, .window = D, .heads = heads, .dim = hd, .scale = 1.0 / @sqrt(@as(f32, @floatFromInt(hd))) };
            try gpu.dispatch(ctx, "mk_vis_attention_flash64", &.{ scratch, scratch, scratch, scratch }, &ap, ((S + 31) / 32) * heads * 128, 128);
            var mo: gpu.Params = .{ .x = att_off, .xs = D, .y = h_off, .ys = D, .rows = S, .n = D, .m = D, .dtype = 2, .bdtype = 2, .flags = gpu.flag_has_weight, .act_a = 5 };
            try gpu.dispatch(ctx, "mk_vis_linear_mma_h64x64", &.{ scratch, out_w, out_b, scratch, ls1 }, &mo, tgs64_d * 256, 256);
            try gpu.dispatch(ctx, "mk_vis_layernorm_1024", &.{ scratch, ln2_w, ln2_b, scratch }, &lp, S * 256, 256);
            var mfc: gpu.Params = .{ .x = norm_off, .xs = D, .y = mlp_off, .ys = M, .rows = S, .n = D, .m = M, .dtype = 2, .bdtype = 2, .flags = gpu.flag_has_weight, .act_a = 4 };
            try gpu.dispatch(ctx, "mk_vis_linear_mma_h64x64", &.{ scratch, fc_w, fc_b, scratch, fc_w }, &mfc, ((M + 63) / 64) * ((S + 63) / 64) * 256, 256);
            var mpr: gpu.Params = .{ .x = mlp_off, .xs = M, .y = h_off, .ys = D, .rows = S, .n = M, .m = D, .dtype = 2, .bdtype = 2, .flags = gpu.flag_has_weight, .act_a = 5 };
            try gpu.dispatch(ctx, "mk_vis_linear_mma_h64x64", &.{ scratch, proj_w, proj_b, scratch, ls2 }, &mpr, tgs64_d * 256, 256);
            try ctx.finish();
        }

        for (h_ref.floats(), scratch.floats()[0..seq_d]) |ev, av| {
            try std.testing.expectApproxEqAbs(ev, av, 1e-2);
        }
    }
}
