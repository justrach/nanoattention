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

// Opt-in candidate: append after kernels.metal and resident_plan.metal.
// One decode query; float32 Q/K/V/output; D=V=128; learned sinks always present.
// No arbitrary mask: all tokens in the effective causal/windowed prefix are used.
// Preconditions: heads/kv_heads is integral, 1 <= GQA <= 32, capacity > 0,
// pos < UINT_MAX, effective-prefix length <= 262144, and every effective-prefix
// token is still resident in the ring. These are caller preconditions, not masks.
// Single/merge: 1024 threads (32 SIMDs) per head. Split: 32*GQA threads per group.
// Split grid: kv_heads * p.k groups; group = kv_head * p.k + block.
// p.k is the certified B=64/128/512/1024 stripe count, NOT a token chunk size.
// Scratch: float states[heads * B * 130], record = {raw_o[128], max, sum}.
// A completed producer dispatch must precede merge; merge must precede consumers.
// Preserve MLX compiler floating-point settings when assessing parity: this file
// preserves source association, not a guarantee of compiler/runtime bit identity.

#include <metal_simdgroup>

// Specializations below use these literals; reject stale proof bindings.
static_assert(STAGED_MLX_ATTENTION_DIM == 128 && STAGED_MLX_ATTENTION_LANES == 32 &&
    STAGED_MLX_ATTENTION_ELEMENTS_PER_LANE == 4 && STAGED_MLX_ATTENTION_SINGLE_SIMDS == 32 &&
    STAGED_MLX_ATTENTION_MERGE_SIMDS == 32 && STAGED_MLX_ATTENTION_STATE_STRIDE == 130 &&
    STAGED_MLX_ATTENTION_MAX_FIELD == 128 && STAGED_MLX_ATTENTION_SUM_FIELD == 129 &&
    STAGED_MLX_ATTENTION_BLOCKS_64 == 64 && STAGED_MLX_ATTENTION_BLOCKS_128 == 128 &&
    STAGED_MLX_ATTENTION_BLOCKS_512 == 512 && STAGED_MLX_ATTENTION_BLOCKS_1024 == 1024 &&
    STAGED_MLX_ATTENTION_MAX_GQA == 32, "MLX attention certificate geometry mismatch");

inline bool attention_mlx_valid_blocks(uint blocks) {
    return blocks == STAGED_MLX_ATTENTION_BLOCKS_64 || blocks == STAGED_MLX_ATTENTION_BLOCKS_128 ||
        blocks == STAGED_MLX_ATTENTION_BLOCKS_512 || blocks == STAGED_MLX_ATTENTION_BLOCKS_1024;
}

inline bool attention_mlx_valid(constant P &p) {
    return p.dim == 128 && p.capacity > 0 && p.heads > 0 && p.kv_heads > 0 &&
        p.heads % p.kv_heads == 0 && p.heads / p.kv_heads <= 32;
}

inline uint attention_mlx_start(constant P &p) {
    // Stock MLX RotatingKVCache returns the physical buffer, not a chronological
    // rotation, once full. A permutation preserves real softmax but changes its
    // FP32 reduction order; match that order for the model's exact-sized ring.
    if (p.window && p.capacity == p.window && p.pos + 1 >= p.window) return 0;
    return p.window && p.pos + 1 > p.window ? p.pos + 1 - p.window : 0;
}

inline uint attention_mlx_length(constant P &p) {
    return p.window ? min(p.pos + 1, p.window) : p.pos + 1;
}

inline ulong attention_mlx_base(constant P &p, uint token, uint kv_head, uint lane) {
    return (ulong(token % p.capacity) * p.kv_heads + kv_head) * 128 + lane * 4;
}

kernel void attention_mlx_single(device const float *queries [[buffer(0)]],
                                 device const float *keys [[buffer(1)]],
                                 device const float *values [[buffer(2)]],
                                 device const uchar *sink [[buffer(3)]],
                                 device float *out [[buffer(4)]],
                                 constant P &p [[buffer(8)]],
                                 uint3 grid [[threadgroup_position_in_grid]],
                                 uint3 tptg [[threads_per_threadgroup]],
                                 uint sg [[simdgroup_index_in_threadgroup]],
                                 uint lane [[thread_index_in_simdgroup]]) {
    const uint head = grid.x;
    // All early exits are uniform over the entire group, before any barrier.
    if (!attention_mlx_valid(p) || head >= p.heads ||
        tptg.x * tptg.y * tptg.z != 1024) return;
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
        for (int j = 0; j < 4; ++j) k[j] = keys[base + j];
        float score = 0;
        for (int j = 0; j < 4; ++j) score += q[j] * k[j];
        score = simd_sum(score);
        float next = max(maximum, score);
        float factor = fast::exp(maximum - next);
        float probability = fast::exp(score - next);
        maximum = next;
        denom = denom * factor + probability;
        for (int j = 0; j < 4; ++j) o[j] = o[j] * factor + probability * values[base + j];
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

kernel void attention_mlx_split(device const float *queries [[buffer(0)]],
                                device const float *keys [[buffer(1)]],
                                device const float *values [[buffer(2)]],
                                device const uchar *sink [[buffer(3)]],
                                device float *states [[buffer(4)]],
                                constant P &p [[buffer(8)]],
                                uint3 grid [[threadgroup_position_in_grid]],
                                uint3 tptg [[threads_per_threadgroup]],
                                uint sg [[simdgroup_index_in_threadgroup]],
                                uint lane [[thread_index_in_simdgroup]]) {
    if (!attention_mlx_valid(p) || !attention_mlx_valid_blocks(p.k)) return;
    const uint gqa = p.heads / p.kv_heads;
    if (tptg.x * tptg.y * tptg.z != 32 * gqa) return;
    const uint kv_head = grid.x / p.k, block = grid.x % p.k;
    if (kv_head >= p.kv_heads) return;
    const uint head = kv_head * gqa + sg;
    const uint start = attention_mlx_start(p);
    const uint n = attention_mlx_length(p);
    float q[4], o[4] = {0};
    for (int j = 0; j < 4; ++j) q[j] = p.scale * queries[ulong(head) * 128 + lane * 4 + j];
    float maximum = -3.4028234663852886e38f;
    float denom = 0;
    if (block == 0) {
        maximum = weight(sink, p.offset, p.dtype, head);
        denom = 1;
    }

    // MLX TK=4: prefetch all four keys, then all four values, and consume
    // them in ascending stripe order. Each SIMD keeps its own head's state.
    uint i = block;
    for (; i + 3 * p.k < n; i += 4 * p.k) {
        float kr[4][4], vr[4][4];
        for (int t = 0; t < 4; ++t) {
            const ulong base = attention_mlx_base(p, start + i + uint(t) * p.k, kv_head, lane);
            for (int j = 0; j < 4; ++j) kr[t][j] = keys[base + j];
        }
        for (int t = 0; t < 4; ++t) {
            const ulong base = attention_mlx_base(p, start + i + uint(t) * p.k, kv_head, lane);
            for (int j = 0; j < 4; ++j) vr[t][j] = values[base + j];
        }
        for (int t = 0; t < 4; ++t) {
            float score = 0;
            for (int j = 0; j < 4; ++j) score += q[j] * kr[t][j];
            score = simd_sum(score);
            float next = max(maximum, score);
            float factor = fast::exp(maximum - next);
            float probability = fast::exp(score - next);
            maximum = next;
            denom = denom * factor + probability;
            for (int j = 0; j < 4; ++j) o[j] = o[j] * factor + probability * vr[t][j];
        }
    }
    for (; i < n; i += p.k) {
        const ulong base = attention_mlx_base(p, start + i, kv_head, lane);
        float score = 0;
        for (int j = 0; j < 4; ++j) score += q[j] * keys[base + j];
        score = simd_sum(score);
        float next = max(maximum, score);
        float factor = fast::exp(maximum - next);
        float probability = fast::exp(score - next);
        maximum = next;
        denom = denom * factor + probability;
        for (int j = 0; j < 4; ++j) o[j] = o[j] * factor + probability * values[base + j];
    }
    const ulong state = (ulong(head) * p.k + block) * 130;
    if (lane == 0) {
        states[state + 128] = maximum;
        states[state + 129] = denom;
    }
    for (int j = 0; j < 4; ++j) states[state + lane * 4 + j] = o[j];
}

kernel void attention_mlx_merge(device const float *states [[buffer(0)]],
                                device float *out [[buffer(1)]],
                                constant P &p [[buffer(8)]],
                                uint3 grid [[threadgroup_position_in_grid]],
                                uint3 tptg [[threads_per_threadgroup]],
                                uint sg [[simdgroup_index_in_threadgroup]],
                                uint lane [[thread_index_in_simdgroup]]) {
    const uint head = grid.x;
    if (!attention_mlx_valid(p) || head >= p.heads ||
        !attention_mlx_valid_blocks(p.k) || tptg.x * tptg.y * tptg.z != 1024) return;
    const ulong first = ulong(head) * p.k * 130;
    float o[4] = {0};
    float denom = 0;
    float maximum = -3.4028234663852886e38f;
    threadgroup float outputs[32 * 32];

    // Each lane handles l, l+32, ... before the SIMD reduction. Do not replace
    // this with one sequential sum over blocks or normalize producer states.
    for (uint b = 0; b < p.k / 32; ++b) {
        const ulong state = first + ulong(lane + 32 * b) * 130;
        maximum = max(maximum, states[state + 128]);
    }
    maximum = simd_max(maximum);
    for (uint b = 0; b < p.k / 32; ++b) {
        const ulong state = first + ulong(lane + 32 * b) * 130;
        float factor = fast::exp(states[state + 128] - maximum);
        denom += factor * states[state + 129];
    }
    denom = simd_sum(denom);

    // SIMD g accumulates blocks g, g+32, ... for dimensions 4*l+j.
    for (uint b = 0; b < p.k / 32; ++b) {
        const ulong state = first + ulong(sg + 32 * b) * 130;
        float factor = fast::exp(states[state + 128] - maximum);
        for (int j = 0; j < 4; ++j) o[j] += factor * states[state + lane * 4 + j];
    }
    for (int j = 0; j < 4; ++j) {
        outputs[lane * 32 + sg] = o[j];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        o[j] = simd_sum(outputs[sg * 32 + lane]);
        o[j] = denom == 0 ? o[j] : o[j] / denom;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lane == 0) {
        for (int j = 0; j < 4; ++j) out[ulong(head) * 128 + sg * 4 + j] = o[j];
    }
}
