//! Supported inference configuration for RecursiveSI/flint.
//! Defaults mirror the published config. EOS must be a JSON array (including
//! for a single token); the scalar Transformers shorthand is deliberately not
//! accepted. Unknown fields/options fail closed instead of changing inference.
const std = @import("std");

const full_attention_layers = [_]u32{ 0, 1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 45, 48 };
const default_layer_types = blk: {
    var layers: [49][]const u8 = @splat("sliding_attention");
    for (full_attention_layers) |i| layers[i] = "full_attention";
    break :blk layers;
};

/// MLX affine groupwise quantization: packed unsigned weights, floating scales
/// and affine biases. Other quantization schemes are not supported.
pub const Quantization = struct {
    bits: u32 = 4,
    group_size: u32 = 64,

    pub fn validate(self: Quantization) !void {
        if (!supportedBits(self.bits) or self.group_size == 0)
            return error.UnsupportedQuantization;
    }
};

pub const MLXExport = struct {
    bits: u32 = 4,
    embedding_bits: u32 = 8,
    full_attention_layers: []const u32 = &full_attention_layers,
    routed_expert_bits: u32 = 4,
    scale_rule: []const u8 = "mse",
};

pub const RopeParameters = struct {
    rope_theta: f64 = 10000.0,
    rope_type: []const u8 = "default",
};

pub const Config = struct {
    architectures: []const []const u8 = &.{"FlintForCausalLM"},
    model_type: []const u8 = "flint",
    attention_bias: bool = false,
    attention_dropout: f64 = 0.0,
    attention_sink: bool = true,
    bos_token_id: u32 = 150000,
    decoder_sparse_step: u32 = 1,
    dtype: ?[]const u8 = null,
    eos_token_id: []const u32 = &.{ 150001, 150003, 150005, 150018 },
    head_dim: u32 = 128,
    hidden_act: []const u8 = "silu",
    hidden_size: u32 = 2048,
    initializer_range: f64 = 0.02,
    intermediate_size: u32 = 6144,
    layer_types: []const []const u8 = &default_layer_types,
    // Export metadata only: runtime KV caches remain FP32. Compressed caches
    // and their groupwise quantization are NOT implemented by this engine.
    kv_group_size: ?u32 = null,
    kv_key_bits: ?u32 = null,
    kv_value_bits: ?u32 = null,
    max_position_embeddings: u32 = 262144,
    mlp_only_layers: []const u32 = &.{ 0, 1 },
    moe_intermediate_size: u32 = 512,
    mlx_export: ?MLXExport = null,
    norm_topk_prob: bool = true,
    num_attention_heads: u32 = 32,
    num_experts: u32 = 256,
    num_local_experts: ?u32 = null,
    num_experts_per_tok: u32 = 6,
    num_hidden_layers: u32 = 49,
    num_key_value_heads: u32 = 4,
    no_rope_layer_indices: []const u32 = &full_attention_layers,
    output_router_logits: bool = false,
    pad_token_id: u32 = 150001,
    quantization: ?Quantization = null,
    rms_norm_eps: f64 = 1e-6,
    rope_parameters: ?RopeParameters = null,
    rope_theta: f64 = 10000.0,
    router_aux_loss_coef: f64 = 0.0005,
    router_score_function: []const u8 = "sigmoid",
    router_expert_bias: bool = false,
    scoring_func: ?[]const u8 = null,
    shared_expert_intermediate_size: u32 = 1024,
    sliding_window: u32 = 513,
    softmax_type: []const u8 = "learnable",
    softmax_type_source: ?[]const u8 = null,
    tie_word_embeddings: bool = false,
    torch_dtype: []const u8 = "bfloat16",
    use_cache: bool = true,
    use_sliding_window: bool = true,
    vocab_size: u32 = 150144,
    transformers_version: []const u8 = "5.12.1",

    pub fn validate(self: Config) !void {
        if (!std.mem.eql(u8, self.model_type, "flint") or self.architectures.len != 1 or
            !std.mem.eql(u8, self.architectures[0], "FlintForCausalLM"))
            return error.UnsupportedArchitecture;
        // These are the math paths implemented by the Flint engine. In
        // particular, learnable softmax is NOT ordinary softmax, and routing
        // uses normalized top-k sigmoid scores, not softmax over all experts.
        if (self.attention_bias or self.attention_dropout != 0 or !self.attention_sink or
            self.router_expert_bias or
            self.decoder_sparse_step != 1 or !self.norm_topk_prob or
            self.output_router_logits or self.tie_word_embeddings or
            !std.mem.eql(u8, self.hidden_act, "silu") or
            !std.mem.eql(u8, self.router_score_function, "sigmoid") or
            !std.mem.eql(u8, self.softmax_type, "learnable"))
            return error.UnsupportedOption;
        if (!std.mem.eql(u8, self.torch_dtype, "bfloat16") and
            !std.mem.eql(u8, self.torch_dtype, "float16") and
            !std.mem.eql(u8, self.torch_dtype, "float32"))
            return error.UnsupportedDType;
        if (self.dtype) |dtype| {
            if (!std.mem.eql(u8, dtype, self.torch_dtype)) return error.UnsupportedDType;
        }
        if (self.num_local_experts) |count| {
            if (count != self.num_experts) return error.InvalidDimensions;
        }
        if (self.scoring_func) |scoring| {
            if (!std.mem.eql(u8, scoring, self.router_score_function)) return error.UnsupportedOption;
        }
        if (self.softmax_type_source) |source| {
            if (!std.mem.eql(u8, source, "model.attention_pattern.softmax_type")) return error.UnsupportedOption;
        }
        // Aliases must agree: the math continues to use top-level rope_theta.
        // No scaled RoPE or silent nested-parameter override is supported.
        if (self.rope_parameters) |parameters| {
            if (!std.mem.eql(u8, parameters.rope_type, "default")) return error.UnsupportedOption;
            if (!positiveFinite(parameters.rope_theta) or parameters.rope_theta != self.rope_theta)
                return error.InvalidNumericValue;
        }
        if (self.kv_group_size) |size| {
            if (size == 0) return error.UnsupportedQuantization;
        }
        if (self.kv_key_bits) |bits| {
            if (!supportedBits(bits)) return error.UnsupportedQuantization;
        }
        if (self.kv_value_bits) |bits| {
            if (!supportedBits(bits)) return error.UnsupportedQuantization;
        }
        if (self.quantization) |quantization| try quantization.validate();
        if (self.mlx_export) |export_config| {
            const quantization = self.quantization orelse return error.UnsupportedQuantization;
            if (export_config.bits != quantization.bits or
                export_config.routed_expert_bits != quantization.bits or
                !supportedBits(export_config.embedding_bits) or
                !std.mem.eql(u8, export_config.scale_rule, "mse"))
                return error.UnsupportedQuantization;
            try validateIndices(export_config.full_attention_layers, self.num_hidden_layers);
            var full_count: usize = 0;
            for (self.layer_types, 0..) |kind, layer| {
                const full = std.mem.eql(u8, kind, "full_attention");
                if (full) full_count += 1;
                if (contains(export_config.full_attention_layers, layer) != full)
                    return error.InvalidLayerTypes;
            }
            if (full_count != export_config.full_attention_layers.len) return error.InvalidLayerTypes;
        }
        if (self.hidden_size == 0 or self.head_dim == 0 or self.head_dim % 2 != 0 or
            self.num_attention_heads == 0 or self.num_key_value_heads == 0 or
            self.num_attention_heads % self.num_key_value_heads != 0 or
            self.num_hidden_layers == 0 or self.intermediate_size == 0 or
            self.moe_intermediate_size == 0 or self.shared_expert_intermediate_size == 0 or
            self.num_experts == 0 or self.num_experts_per_tok == 0 or
            self.num_experts_per_tok > self.num_experts or
            self.max_position_embeddings == 0 or self.vocab_size == 0)
            return error.InvalidDimensions;
        // Flint's Q projection is wider than hidden_size (4096 vs 2048).
        // Do not enforce hidden_size == num_attention_heads * head_dim.
        _ = std.math.mul(u32, self.num_attention_heads, self.head_dim) catch return error.InvalidDimensions;
        if (!positiveFinite(self.rms_norm_eps) or !positiveFinite(self.rope_theta) or
            !nonnegativeFinite(self.initializer_range) or !nonnegativeFinite(self.router_aux_loss_coef))
            return error.InvalidNumericValue;
        if (self.layer_types.len != self.num_hidden_layers) return error.InvalidLayerTypes;
        for (self.layer_types) |kind| {
            if (std.mem.eql(u8, kind, "sliding_attention")) {
                if (!self.use_sliding_window or self.sliding_window == 0) return error.UnsupportedOption;
            } else if (!std.mem.eql(u8, kind, "full_attention")) return error.InvalidLayerTypes;
        }
        try validateIndices(self.mlp_only_layers, self.num_hidden_layers);
        try validateIndices(self.no_rope_layer_indices, self.num_hidden_layers);
        if (self.bos_token_id >= self.vocab_size or self.pad_token_id >= self.vocab_size or
            self.eos_token_id.len == 0) return error.InvalidTokenId;
        for (self.eos_token_id) |id| if (id >= self.vocab_size) return error.InvalidTokenId;
    }

    /// True for a sliding-window attention layer. Out-of-range indices are false.
    pub fn sliding(self: Config, layer: usize) bool {
        return layer < self.num_hidden_layers and layer < self.layer_types.len and
            self.use_sliding_window and std.mem.eql(u8, self.layer_types[layer], "sliding_attention");
    }

    /// True for a dense FFN instead of MoE. These layers STILL have attention.
    pub fn dense(self: Config, layer: usize) bool {
        return layer < self.num_hidden_layers and contains(self.mlp_only_layers, layer);
    }

    /// True when the attention layer applies rotary position embeddings.
    pub fn rope(self: Config, layer: usize) bool {
        return layer < self.num_hidden_layers and !contains(self.no_rope_layer_indices, layer);
    }
};

/// All parsed strings/slices are owned by the returned arena; call deinit().
/// Unknown fields (including rope_scaling/quantization_config) are rejected.
pub fn parse(allocator: std.mem.Allocator, bytes: []const u8) !std.json.Parsed(Config) {
    // std.json's typed integer parser also accepts numeric strings. Export
    // metadata is a strict schema: reject such coercions before typed parsing.
    const metadata = try std.json.parseFromSlice(std.json.Value, allocator, bytes, .{
        .duplicate_field_behavior = .@"error",
    });
    defer metadata.deinit();
    try validateMetadataTypes(metadata.value);
    const parsed = try std.json.parseFromSlice(Config, allocator, bytes, .{
        .allocate = .alloc_always,
        .ignore_unknown_fields = false,
        .duplicate_field_behavior = .@"error",
    });
    errdefer parsed.deinit();
    try parsed.value.validate();
    return parsed;
}

fn validateMetadataTypes(value: std.json.Value) !void {
    if (value != .object) return error.UnexpectedToken;
    inline for (.{ "kv_group_size", "kv_key_bits", "kv_value_bits", "num_local_experts" }) |name|
        try requireType(value, name, .integer, true);
    inline for (.{ "dtype", "scoring_func", "softmax_type_source" }) |name|
        try requireType(value, name, .string, true);
    inline for (.{ "attention_sink", "router_expert_bias" }) |name|
        try requireType(value, name, .bool, false);
    inline for (.{ "quantization", "mlx_export", "rope_parameters" }) |name| {
        try requireType(value, name, .object, true);
        if (value.object.get(name)) |nested| {
            if (nested != .null) {
                if (comptime std.mem.eql(u8, name, "quantization")) {
                    try requireType(nested, "bits", .integer, false);
                    try requireType(nested, "group_size", .integer, false);
                } else if (comptime std.mem.eql(u8, name, "mlx_export")) {
                    inline for (.{ "bits", "embedding_bits", "routed_expert_bits" }) |field|
                        try requireType(nested, field, .integer, false);
                    try requireType(nested, "scale_rule", .string, false);
                    try requireType(nested, "full_attention_layers", .array, false);
                    if (nested.object.get("full_attention_layers")) |layers| {
                        for (layers.array.items) |layer| {
                            if (layer != .integer) return error.UnexpectedToken;
                        }
                    }
                } else {
                    try requireType(nested, "rope_type", .string, false);
                    if (nested.object.get("rope_theta")) |theta| {
                        if (theta != .integer and theta != .float) return error.UnexpectedToken;
                    }
                }
            }
        }
    }
}

fn requireType(object: std.json.Value, name: []const u8, tag: std.meta.Tag(std.json.Value), nullable: bool) !void {
    if (object.object.get(name)) |value| {
        if (nullable and value == .null) return;
        if (std.meta.activeTag(value) != tag) return error.UnexpectedToken;
    }
}

fn contains(indices: []const u32, layer: usize) bool {
    for (indices) |index| if (index == layer) return true;
    return false;
}

fn validateIndices(indices: []const u32, count: u32) !void {
    for (indices, 0..) |index, i| {
        if (index >= count) return error.InvalidLayerIndex;
        for (indices[0..i]) |previous| if (previous == index) return error.InvalidLayerIndex;
    }
}

fn supportedBits(bits: u32) bool {
    return bits == 4 or bits == 8;
}

fn positiveFinite(value: f64) bool {
    return std.math.isFinite(value) and value > 0;
}

fn nonnegativeFinite(value: f64) bool {
    return std.math.isFinite(value) and value >= 0;
}

const tiny_json =
    \\{"hidden_size":4,"head_dim":2,"num_attention_heads":4,"num_key_value_heads":2,
    \\"num_hidden_layers":2,"intermediate_size":8,"moe_intermediate_size":2,
    \\"shared_expert_intermediate_size":4,"num_experts":3,"num_experts_per_tok":2,
    \\"layer_types":["full_attention","sliding_attention"],"mlp_only_layers":[0],
    \\"no_rope_layer_indices":[0],"vocab_size":16,"bos_token_id":0,
    \\"pad_token_id":1,"eos_token_id":[1,2],"sliding_window":3}
;

const int4_metadata_json =
    \\{"attention_sink":true,"dtype":"bfloat16","kv_group_size":64,
    \\"kv_key_bits":8,"kv_value_bits":8,"num_local_experts":256,
    \\"quantization":{"bits":4,"group_size":64},
    \\"mlx_export":{"bits":4,"embedding_bits":8,
    \\"full_attention_layers":[0,1,5,9,13,17,21,25,29,33,37,41,45,48],
    \\"routed_expert_bits":4,"scale_rule":"mse"},
    \\"rope_parameters":{"rope_theta":10000.0,"rope_type":"default"},
    \\"router_expert_bias":false,"scoring_func":"sigmoid",
    \\"softmax_type_source":"model.attention_pattern.softmax_type"}
;

test "local MLX int4 checkpoint metadata preserves Flint math" {
    const parsed = try parse(std.testing.allocator, int4_metadata_json);
    defer parsed.deinit();
    try std.testing.expectEqual(@as(u32, 4), parsed.value.quantization.?.bits);
    try std.testing.expectEqual(@as(u32, 64), parsed.value.quantization.?.group_size);
    try std.testing.expectEqual(@as(u32, 8), parsed.value.mlx_export.?.embedding_bits);
    try std.testing.expectEqual(@as(u32, 8), parsed.value.kv_key_bits.?);
    try std.testing.expectEqual(@as(u32, 8), parsed.value.kv_value_bits.?);
    try std.testing.expectEqual(@as(f64, 10000), parsed.value.rope_theta);
    try std.testing.expectEqual(@as(u32, 49), parsed.value.num_hidden_layers);
    try std.testing.expectEqual(@as(u32, 513), parsed.value.sliding_window);
    try std.testing.expect(!parsed.value.rope(48));
    try std.testing.expect(parsed.value.sliding(47));
}

test "MLX supports four and eight bit affine groupwise weights" {
    const parsed = try parse(std.testing.allocator,
        \\{"quantization":{"bits":8,"group_size":32},
        \\"mlx_export":{"bits":8,"routed_expert_bits":8,"embedding_bits":4}}
    );
    defer parsed.deinit();
    try std.testing.expectEqual(@as(u32, 8), parsed.value.quantization.?.bits);
    try std.testing.expectEqual(@as(u32, 32), parsed.value.quantization.?.group_size);
    const unquantized = try parse(std.testing.allocator, "{}");
    defer unquantized.deinit();
    try std.testing.expect(unquantized.value.quantization == null);
    try std.testing.expect(unquantized.value.mlx_export == null);
}

test "reject unsupported or inconsistent MLX quantization metadata" {
    const invalid = [_][]const u8{
        "{\"quantization\":{\"bits\":2}}",
        "{\"quantization\":{\"group_size\":0}}",
        "{\"mlx_export\":{}}",
        "{\"quantization\":{},\"mlx_export\":{\"bits\":8}}",
        "{\"quantization\":{},\"mlx_export\":{\"routed_expert_bits\":8}}",
        "{\"quantization\":{},\"mlx_export\":{\"embedding_bits\":2}}",
        "{\"quantization\":{},\"mlx_export\":{\"scale_rule\":\"symmetric\"}}",
        "{\"kv_group_size\":0}",
        "{\"kv_key_bits\":3}",
        "{\"kv_value_bits\":16}",
    };
    for (invalid) |json| try std.testing.expectError(error.UnsupportedQuantization, parse(std.testing.allocator, json));
    try std.testing.expectError(error.InvalidLayerTypes, parse(std.testing.allocator, "{\"quantization\":{},\"mlx_export\":{\"full_attention_layers\":[0]}}"));
    try std.testing.expectError(error.InvalidLayerIndex, parse(std.testing.allocator, "{\"quantization\":{},\"mlx_export\":{\"full_attention_layers\":[0,0]}}"));
    try std.testing.expectError(error.UnknownField, parse(std.testing.allocator, "{\"quantization\":{\"mode\":\"affine\"}}"));
    try std.testing.expectError(error.UnknownField, parse(std.testing.allocator, "{\"mlx_export\":{\"unknown\":1}}"));
    try std.testing.expectError(error.UnexpectedToken, parse(std.testing.allocator, "{\"quantization\":{\"bits\":\"4\"}}"));
    try std.testing.expectError(error.UnexpectedToken, parse(std.testing.allocator, "{\"quantization\":{\"bits\":4.0}}"));
    try std.testing.expectError(error.UnexpectedToken, parse(std.testing.allocator, "{\"kv_key_bits\":\"8\"}"));
    try std.testing.expectError(error.UnexpectedToken, parse(std.testing.allocator, "{\"mlx_export\":{\"full_attention_layers\":[\"0\"]}}"));
    try std.testing.expectError(error.UnexpectedToken, parse(std.testing.allocator, "{\"rope_parameters\":{\"rope_theta\":\"10000\"}}"));
}

test "reject aliases that would silently change inference" {
    try std.testing.expectError(error.UnsupportedOption, parse(std.testing.allocator, "{\"attention_sink\":false}"));
    try std.testing.expectError(error.UnsupportedOption, parse(std.testing.allocator, "{\"router_expert_bias\":true}"));
    try std.testing.expectError(error.UnsupportedOption, parse(std.testing.allocator, "{\"scoring_func\":\"softmax\"}"));
    try std.testing.expectError(error.UnsupportedOption, parse(std.testing.allocator, "{\"softmax_type_source\":\"other\"}"));
    try std.testing.expectError(error.UnsupportedDType, parse(std.testing.allocator, "{\"dtype\":\"float16\"}"));
    try std.testing.expectError(error.InvalidDimensions, parse(std.testing.allocator, "{\"num_local_experts\":128}"));
    try std.testing.expectError(error.InvalidNumericValue, parse(std.testing.allocator, "{\"rope_parameters\":{\"rope_theta\":5000}}"));
    try std.testing.expectError(error.UnsupportedOption, parse(std.testing.allocator, "{\"rope_parameters\":{\"rope_type\":\"linear\"}}"));
    try std.testing.expectError(error.UnknownField, parse(std.testing.allocator, "{\"rope_parameters\":{\"factor\":2}}"));
    try std.testing.expectError(error.UnexpectedToken, parse(std.testing.allocator, "{\"attention_sink\":\"true\"}"));
    const matching = try parse(std.testing.allocator, "{\"rope_theta\":5000,\"rope_parameters\":{\"rope_theta\":5000},\"torch_dtype\":\"float16\",\"dtype\":\"float16\"}");
    defer matching.deinit();
    try std.testing.expectEqual(@as(f64, 5000), matching.value.rope_theta);
}

fn int4AllocationProbe(allocator: std.mem.Allocator) !void {
    const parsed = try parse(allocator, int4_metadata_json);
    defer parsed.deinit();
    try parsed.value.validate();
}

test "MLX metadata allocation failures release nested storage" {
    try std.testing.checkAllAllocationFailures(std.testing.allocator, int4AllocationProbe, .{});
}

test "official defaults and Flint projection width" {
    const parsed = try parse(std.testing.allocator, "{}");
    defer parsed.deinit();
    try std.testing.expectEqual(@as(u32, 49), parsed.value.num_hidden_layers);
    try std.testing.expect(parsed.value.dense(0));
    try std.testing.expect(!parsed.value.dense(2));
    try std.testing.expect(!parsed.value.sliding(48));
    try std.testing.expect(parsed.value.sliding(47));
    try std.testing.expect(!parsed.value.rope(48));
    try std.testing.expect(parsed.value.rope(47));
    try std.testing.expect(!parsed.value.rope(49));
    try std.testing.expect(!parsed.value.dense(49));
    try std.testing.expect(!parsed.value.sliding(49));
}

test "published config schema" {
    // Snapshot of RecursiveSI/flint/config.json, including every published key.
    const json =
        \\{"architectures":["FlintForCausalLM"],"model_type":"flint",
        \\"attention_bias":false,"attention_dropout":0.0,"bos_token_id":150000,
        \\"decoder_sparse_step":1,"eos_token_id":[150001,150003,150005,150018],
        \\"head_dim":128,"hidden_act":"silu","hidden_size":2048,
        \\"initializer_range":0.02,"intermediate_size":6144,
        \\"layer_types":["full_attention","full_attention","sliding_attention","sliding_attention",
        \\"sliding_attention","full_attention","sliding_attention","sliding_attention",
        \\"sliding_attention","full_attention","sliding_attention","sliding_attention",
        \\"sliding_attention","full_attention","sliding_attention","sliding_attention",
        \\"sliding_attention","full_attention","sliding_attention","sliding_attention",
        \\"sliding_attention","full_attention","sliding_attention","sliding_attention",
        \\"sliding_attention","full_attention","sliding_attention","sliding_attention",
        \\"sliding_attention","full_attention","sliding_attention","sliding_attention",
        \\"sliding_attention","full_attention","sliding_attention","sliding_attention",
        \\"sliding_attention","full_attention","sliding_attention","sliding_attention",
        \\"sliding_attention","full_attention","sliding_attention","sliding_attention",
        \\"sliding_attention","full_attention","sliding_attention","sliding_attention","full_attention"],
        \\"max_position_embeddings":262144,"mlp_only_layers":[0,1],
        \\"moe_intermediate_size":512,"norm_topk_prob":true,"num_attention_heads":32,
        \\"num_experts":256,"num_experts_per_tok":6,"num_hidden_layers":49,
        \\"num_key_value_heads":4,"no_rope_layer_indices":[0,1,5,9,13,17,21,25,29,33,37,41,45,48],
        \\"output_router_logits":false,"pad_token_id":150001,"rms_norm_eps":1e-6,
        \\"rope_theta":10000.0,"router_aux_loss_coef":0.0005,"router_score_function":"sigmoid",
        \\"shared_expert_intermediate_size":1024,"sliding_window":513,"softmax_type":"learnable",
        \\"tie_word_embeddings":false,"torch_dtype":"bfloat16","use_cache":true,
        \\"use_sliding_window":true,"vocab_size":150144,"transformers_version":"5.12.1"}
    ;
    const parsed = try parse(std.testing.allocator, json);
    defer parsed.deinit();
    const defaults: Config = .{};
    try std.testing.expectEqualDeep(defaults, parsed.value);
}

fn allocationProbe(allocator: std.mem.Allocator) !void {
    const parsed = try parse(allocator, tiny_json);
    defer parsed.deinit();
    try parsed.value.validate();
}

test "allocation failures clean up all config storage" {
    try std.testing.checkAllAllocationFailures(std.testing.allocator, allocationProbe, .{});
}

test "tiny config supports wider Q projection and owns its strings" {
    const bytes = try std.testing.allocator.dupe(u8, tiny_json);
    const parsed = try parse(std.testing.allocator, bytes);
    defer parsed.deinit();
    @memset(bytes, 'x');
    std.testing.allocator.free(bytes);
    try parsed.value.validate();
    try std.testing.expect(parsed.value.sliding(1));
    try std.testing.expect(parsed.value.rope(1));
    try std.testing.expect(parsed.value.dense(0));
}

test "reject incompatible options and malformed JSON config" {
    try std.testing.expectError(error.UnsupportedArchitecture, parse(std.testing.allocator, "{\"model_type\":\"llama\"}"));
    try std.testing.expectError(error.UnsupportedOption, parse(std.testing.allocator, "{\"softmax_type\":\"standard\"}"));
    try std.testing.expectError(error.UnsupportedOption, parse(std.testing.allocator, "{\"norm_topk_prob\":false}"));
    try std.testing.expectError(error.UnknownField, parse(std.testing.allocator, "{\"rope_scaling\":{}}"));
    try std.testing.expectError(error.DuplicateField, parse(std.testing.allocator, "{\"head_dim\":2,\"head_dim\":4}"));
    try std.testing.expectError(error.UnexpectedToken, parse(std.testing.allocator, "{\"eos_token_id\":1}"));
}

test "validate dimension, numerical, token and layer bounds" {
    var config: Config = .{};
    config.num_key_value_heads = 3;
    try std.testing.expectError(error.InvalidDimensions, config.validate());
    config = .{};
    config.head_dim = 3;
    try std.testing.expectError(error.InvalidDimensions, config.validate());
    config = .{};
    config.rms_norm_eps = std.math.nan(f64);
    try std.testing.expectError(error.InvalidNumericValue, config.validate());
    config = .{};
    config.eos_token_id = &.{150144};
    try std.testing.expectError(error.InvalidTokenId, config.validate());
    config = .{};
    config.mlp_only_layers = &.{ 0, 0 };
    try std.testing.expectError(error.InvalidLayerIndex, config.validate());
    config = .{};
    config.no_rope_layer_indices = &.{49};
    try std.testing.expectError(error.InvalidLayerIndex, config.validate());
    config = .{};
    config.layer_types = &.{"full_attention"};
    try std.testing.expectError(error.InvalidLayerTypes, config.validate());
}
