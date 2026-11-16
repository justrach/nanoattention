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

// Exact full physical-ring specialization: removes runtime modulo and loop bound.
// Never usable before the complete capacity-513/window-513 ring is valid.
// Same per-SIMD token order, arithmetic, transpose and reductions as original.

kernel void attention_mlx_single_ring513(device const float *queries [[buffer(0)]],
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
    if (!attention_mlx_valid(p) || p.capacity != 513 || p.window != 513 || p.pos < 512 || head >= p.heads ||
        tptg.x * tptg.y * tptg.z != 1024) return;
    constexpr uint n = 513;
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
        const ulong base = (ulong(i) * p.kv_heads + kv_head) * 128 + lane * 4;
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
