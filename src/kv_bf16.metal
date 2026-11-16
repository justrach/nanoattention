/*
 * Adapted from MLX v0.32.4 mlx/backend/metal/kernels/sdpa_vector.h.
 * Copyright © 2024 Apple Inc.
 * Copyright © 2023 Apple Inc. (MLX MIT license).
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

// Append AFTER normalization_aligned.metal, attention_mlx.metal, and
// attention_gqa_reuse.metal in the SAME matched production library.
// Opt-in STORAGE precision only: global KV is ushort BF16; Q/K working vectors,
// attention arithmetic, sink ABI, outputs and 130-word states remain FP32.
// FP32 remains the host default; sliding caches MUST remain FP32.
// Host must gate on the certified M3 device and successful device parity gates.
// Supported: D128, GQA8, window=0, 1<=pos+1<=min(capacity,262144).
// QK buffers 0..4 and P8 are unchanged; cache buffers 5/6 are ushort elements.
// QK launch: one full SIMD/head, padded head tails allowed; group multiple of 32.
// Single: 1024 threads/head; split: 128 threads/group, kv_heads*B groups,
// B=P.k in {128,512,1024}. Use ORIGINAL attention_mlx_merge afterwards.
// Sink dtype 0/1/2 and its aligned BYTE offset are unchanged. Q/K norm weight
// dtypes/offsets have the same independent ABI. Bounds and residency are host
// duties. Rejection is NOT fallback: do not merge untouched scratch. Host must
// select a complete FP32 producer/cache path for unsupported configurations.
// No fast-math changes: source association is preserved, not proven device bits.

inline ushort kv_bf16_from_float(float value) {
    uint bits = as_type<uint>(value);
    // Preserve NaN sign/high payload and force a quiet nonzero BF16 payload;
    // even a signaling NaN whose payload only occupies discarded bits stays NaN.
    if ((bits & 0x7fffffffu) > 0x7f800000u)
        return ushort((bits >> 16) | 0x0040u);
    // Round to nearest, ties to even, including signed zeros/subnormals and
    // finite overflow to infinity. No native half or floating conversion.
    return ushort((bits + 0x7fffu + ((bits >> 16) & 1u)) >> 16);
}

inline float kv_bf16_to_float(ushort value) {
    return as_type<float>(uint(value) << 16);
}

// Experimental split-only load: data must be 8-byte aligned and base must
// be a multiple of four ushort elements. Metal cache buffer bases/offsets must
// preserve that alignment; D128 token/head strides and lane*4 preserve it here.
// The packed uint2 load and integer-only widening preserve every BF16 bit,
// including NaN payloads and signed zeros (no floating-point conversion).
inline float4 kv_bf16_load4(device const ushort *data, ulong base) {
    const uint2 packed = *reinterpret_cast<device const uint2 *>(data + base);
    const uint4 bits = (uint4(packed.x, packed.x, packed.y, packed.y) <<
                        uint4(16, 0, 16, 0)) & uint4(0xffff0000u);
    return as_type<float4>(bits);
}

inline bool kv_bf16_valid(constant P &p) {
    return attention_mlx_valid(p) && p.heads / p.kv_heads == 8 &&
        p.window == 0 && p.pos < 262144 && p.pos + 1 <= p.capacity &&
        p.dtype <= 2 && p.offset % (p.dtype == 0 ? 4 : 2) == 0;
}

kernel void qk_rope_cache_aligned_bf16(device float *q [[buffer(0)]],
                                 device float *k [[buffer(1)]],
                                 device const float *v [[buffer(2)]],
                                 device const uchar *qw [[buffer(3)]],
                                 device const uchar *kw [[buffer(4)]],
                                 device ushort *keys [[buffer(5)]],
                                 device ushort *values [[buffer(6)]],
                                 constant P &p [[buffer(8)]],
                                 uint i [[thread_position_in_grid]],
                                 uint lane [[thread_index_in_simdgroup]]) {
    if (!kv_bf16_valid(p) || p.scale_dtype > 2 ||
        p.scale_offset % (p.scale_dtype == 0 ? 4 : 2) != 0) return;
    uint head = i / 32;
    if (p.dim != aligned_norm_head || head >= p.heads + p.kv_heads) return;
    bool query = head < p.heads;
    uint local_head = query ? head : head - p.heads;
    uint base = local_head * p.dim;
    device float *x = query ? q : k;
    device const uchar *w = query ? qw : kw;
    ulong offset = query ? p.offset : p.scale_offset;
    uint dtype = query ? p.dtype : p.scale_dtype;
    float sum = 0.0f;
    for (uint j = 0; j < aligned_norm_reads; ++j) {
        float a = x[base + aligned_norm_reads * lane + j];
        sum += a * a;
    }
    float inv = precise::rsqrt(simd_sum(sum) / float(p.dim) + p.eps);
    uint half_dim = p.dim / 2;
    for (uint j = lane; j < half_dim; j += 32) {
        float a = x[base + j] * inv * weight(w, offset, dtype, j);
        float b = x[base + j + half_dim] * inv * weight(w, offset, dtype, j + half_dim);
        if (p.bits) {
            float fraction = float(j) / float(half_dim);
            float frequency = exp2(-fraction * p.theta);
            float angle = float(p.pos) * frequency;
            float c = fast::cos(angle), s = fast::sin(angle);
            float rotated_a = a * c - b * s;
            b = a * s + b * c;
            a = rotated_a;
        }
        x[base + j] = a;
        x[base + j + half_dim] = b;
        if (!query) {
            ulong slot = ulong(p.pos % p.capacity) * p.kv_heads * p.dim + base;
            keys[slot + j] = kv_bf16_from_float(a);
            keys[slot + j + half_dim] = kv_bf16_from_float(b);
            values[slot + j] = kv_bf16_from_float(v[base + j]);
            values[slot + j + half_dim] = kv_bf16_from_float(v[base + j + half_dim]);
        }
    }
}

kernel void attention_mlx_single_bf16(device const float *queries [[buffer(0)]],
                                 device const ushort *keys [[buffer(1)]],
                                 device const ushort *values [[buffer(2)]],
                                 device const uchar *sink [[buffer(3)]],
                                 device float *out [[buffer(4)]],
                                 constant P &p [[buffer(8)]],
                                 uint3 grid [[threadgroup_position_in_grid]],
                                 uint3 tptg [[threads_per_threadgroup]],
                                 uint sg [[simdgroup_index_in_threadgroup]],
                                 uint lane [[thread_index_in_simdgroup]]) {
    const uint head = grid.x;
    // All early exits are uniform over the entire group, before any barrier.
    if (!kv_bf16_valid(p) || head >= p.heads ||
        tptg.x != 1024 || tptg.y != 1 || tptg.z != 1 ||
        grid.y != 0 || grid.z != 0) return;
    const uint start = attention_mlx_start(p);
    const uint n = attention_mlx_length(p);
    const uint kv_head = head / (p.heads / p.kv_heads);
    float q[4], k[4], o[4];
    for (int j = 0; j < 4; ++j) q[j] = p.scale * queries[ulong(head) * 128 + lane * 4 + j];
    for (int j = 0; j < 4; ++j) o[j] = 0;
    float maximum = -3.4028234663852886e38f;
    float denom = 0;
    if (sg == 0) {
        maximum = weight(sink, p.offset, p.dtype, head);
        denom = 1;
    }
    for (uint i = sg; i < n; i += 32) {
        const ulong base = attention_mlx_base(p, start + i, kv_head, lane);
        for (int j = 0; j < 4; ++j) k[j] = kv_bf16_to_float(keys[base + j]);
        float score = 0;
        for (int j = 0; j < 4; ++j) score += q[j] * k[j];
        score = simd_sum(score);
        float next = max(maximum, score);
        float factor = fast::exp(maximum - next);
        float probability = fast::exp(score - next);
        maximum = next;
        denom = denom * factor + probability;
        for (int j = 0; j < 4; ++j) o[j] = o[j] * factor + probability * kv_bf16_to_float(values[base + j]);
    }

    threadgroup float outputs[32 * 32];
    threadgroup float maxima[32], denominators[32];
    if (lane == 0) {
        maxima[sg] = maximum;
        denominators[sg] = denom;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    maximum = maxima[lane];
    float next = simd_max(maximum);
    float factor = fast::exp(maximum - next);
    denom = simd_sum(denominators[lane] * factor);
    for (int j = 0; j < 4; ++j) {
        outputs[lane * 32 + sg] = o[j];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        o[j] = simd_sum(outputs[sg * 32 + lane] * factor);
        o[j] = denom == 0 ? o[j] : o[j] / denom;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lane == 0) {
        for (int j = 0; j < 4; ++j) out[ulong(head) * 128 + sg * 4 + j] = o[j];
    }
}

template<uint B, bool VectorLoads = false>
inline void attention_mlx_gqa2_bf16_body(device const float *queries,
                                        device const ushort *keys,
                                        device const ushort *values,
                                        device const uchar *sink,
                                        device float *states,
                                        constant P &p,
                                        uint group, uint sg, uint lane) {
    static_assert(B == 128 || B == 512 || B == 1024, "unsupported GQA stripe count");
    constexpr uint H = 2;
    const uint kv_head = group / B, block = group % B;
    if (kv_head >= p.kv_heads) return;
    const uint first_head = kv_head * 8 + sg * H;
    const uint n = p.pos + 1;
    float q[H][4], o[H][4];
    float maximum[H], denom[H];
    #pragma unroll
    for (uint h = 0; h < H; ++h) {
        const uint head = first_head + h;
        for (int j = 0; j < 4; ++j) q[h][j] = p.scale * queries[ulong(head) * 128 + lane * 4 + j];
        for (int j = 0; j < 4; ++j) o[h][j] = 0;
        maximum[h] = -3.4028234663852886e38f;
        denom[h] = 0;
        if (block == 0) {
            maximum[h] = weight(sink, p.offset, p.dtype, head);
            denom[h] = 1;
        }
    }

    // TK=4: load four keys, then four values ONCE per SIMD, and reuse each
    // loaded lane's four components for H queries. All heads see tokens
    // block, block+B, ... in the original ascending stripe order. Do not split
    // the lanes between heads or combine reductions from different heads.
    uint i = block;
    for (; i + 3 * B < n; i += 4 * B) {
        float kr[4][4], vr[4][4];
        for (int t = 0; t < 4; ++t) {
            const ulong base = (ulong(i + uint(t) * B) * p.kv_heads + kv_head) * 128 + lane * 4;
            if (VectorLoads) {
                const float4 loaded = kv_bf16_load4(keys, base);
                for (int j = 0; j < 4; ++j) kr[t][j] = loaded[j];
            } else {
                for (int j = 0; j < 4; ++j) kr[t][j] = kv_bf16_to_float(keys[base + j]);
            }
        }
        for (int t = 0; t < 4; ++t) {
            const ulong base = (ulong(i + uint(t) * B) * p.kv_heads + kv_head) * 128 + lane * 4;
            if (VectorLoads) {
                const float4 loaded = kv_bf16_load4(values, base);
                for (int j = 0; j < 4; ++j) vr[t][j] = loaded[j];
            } else {
                for (int j = 0; j < 4; ++j) vr[t][j] = kv_bf16_to_float(values[base + j]);
            }
        }
        for (int t = 0; t < 4; ++t) {
            #pragma unroll
            for (uint h = 0; h < H; ++h) {
                float score = 0;
                for (int j = 0; j < 4; ++j) score += q[h][j] * kr[t][j];
                score = simd_sum(score);
                float next = max(maximum[h], score);
                float factor = fast::exp(maximum[h] - next);
                float probability = fast::exp(score - next);
                maximum[h] = next;
                denom[h] = denom[h] * factor + probability;
                for (int j = 0; j < 4; ++j) o[h][j] = o[h][j] * factor + probability * vr[t][j];
            }
        }
    }
    for (; i < n; i += B) {
        const ulong base = (ulong(i) * p.kv_heads + kv_head) * 128 + lane * 4;
        float kr[4], vr[4];
        if (VectorLoads) {
            const float4 loaded_keys = kv_bf16_load4(keys, base);
            const float4 loaded_values = kv_bf16_load4(values, base);
            for (int j = 0; j < 4; ++j) kr[j] = loaded_keys[j];
            for (int j = 0; j < 4; ++j) vr[j] = loaded_values[j];
        } else {
            for (int j = 0; j < 4; ++j) kr[j] = kv_bf16_to_float(keys[base + j]);
            for (int j = 0; j < 4; ++j) vr[j] = kv_bf16_to_float(values[base + j]);
        }
        #pragma unroll
        for (uint h = 0; h < H; ++h) {
            float score = 0;
            for (int j = 0; j < 4; ++j) score += q[h][j] * kr[j];
            score = simd_sum(score);
            float next = max(maximum[h], score);
            float factor = fast::exp(maximum[h] - next);
            float probability = fast::exp(score - next);
            maximum[h] = next;
            denom[h] = denom[h] * factor + probability;
            for (int j = 0; j < 4; ++j) o[h][j] = o[h][j] * factor + probability * vr[j];
        }
    }
    #pragma unroll
    for (uint h = 0; h < H; ++h) {
        const ulong state = (ulong(first_head + h) * B + block) * 130;
        if (lane == 0) {
            states[state + 128] = maximum[h];
            states[state + 129] = denom[h];
        }
        for (int j = 0; j < 4; ++j) states[state + lane * 4 + j] = o[h][j];
    }
}

kernel void attention_mlx_split_gqa2_bf16(device const float *queries [[buffer(0)]],
                                          device const ushort *keys [[buffer(1)]],
                                          device const ushort *values [[buffer(2)]],
                                          device const uchar *sink [[buffer(3)]],
                                          device float *states [[buffer(4)]],
                                          constant P &p [[buffer(8)]],
                                          uint3 grid [[threadgroup_position_in_grid]],
                                          uint3 tptg [[threads_per_threadgroup]],
                                          uint sg [[simdgroup_index_in_threadgroup]],
                                          uint lane [[thread_index_in_simdgroup]]) {
    // Entire group rejects uniformly before reads/writes. No implicit fallback.
    if (!kv_bf16_valid(p) || tptg.x != 128 || tptg.y != 1 || tptg.z != 1 ||
        grid.y != 0 || grid.z != 0) return;
    switch (p.k) {
        case 128:
            attention_mlx_gqa2_bf16_body<128>(queries, keys, values, sink, states, p, grid.x, sg, lane);
            break;
        case 512:
            attention_mlx_gqa2_bf16_body<512>(queries, keys, values, sink, states, p, grid.x, sg, lane);
            break;
        case 1024:
            attention_mlx_gqa2_bf16_body<1024>(queries, keys, values, sink, states, p, grid.x, sg, lane);
            break;
        default:
            return;
    }
}

// Opt-in experimental entry point only. Same ABI/grid/128-thread groups and
// stripe/arithmetic order as the scalar path; cache buffer offsets must retain
// 8-byte alignment. The sink's independent p.offset remains a BYTE offset and
// is not used to address either cache. The original entry point stays scalar.
kernel void attention_mlx_split_gqa2_bf16_vector(device const float *queries [[buffer(0)]],
                                                 device const ushort *keys [[buffer(1)]],
                                                 device const ushort *values [[buffer(2)]],
                                                 device const uchar *sink [[buffer(3)]],
                                                 device float *states [[buffer(4)]],
                                                 constant P &p [[buffer(8)]],
                                                 uint3 grid [[threadgroup_position_in_grid]],
                                                 uint3 tptg [[threads_per_threadgroup]],
                                                 uint sg [[simdgroup_index_in_threadgroup]],
                                                 uint lane [[thread_index_in_simdgroup]]) {
    if (!kv_bf16_valid(p) || tptg.x != 128 || tptg.y != 1 || tptg.z != 1 ||
        grid.y != 0 || grid.z != 0) return;
    switch (p.k) {
        case 128:
            attention_mlx_gqa2_bf16_body<128, true>(queries, keys, values, sink, states, p, grid.x, sg, lane);
            break;
        case 512:
            attention_mlx_gqa2_bf16_body<512, true>(queries, keys, values, sink, states, p, grid.x, sg, lane);
            break;
        case 1024:
            attention_mlx_gqa2_bf16_body<1024, true>(queries, keys, values, sink, states, p, grid.x, sg, lane);
            break;
        default:
            return;
    }
}
