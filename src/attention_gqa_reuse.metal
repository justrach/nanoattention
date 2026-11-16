/*
 * Adapted from attention_mlx.metal and MLX v0.32.4 sdpa_vector.h.
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

// Append AFTER attention_mlx.metal in the same production Metal compilation.
// Opt-in global GQA8/D128 decode producer, not a new precision or partition.
// Preconditions (unsupported parameters/groups uniformly return without writes):
//   * FP32 Q/K/V; heads = kv_heads*8, kv_heads>0; learned sink present per head.
//   * window=0, 1<=pos+1<=min(capacity,262144); all tokens resident linearly.
//   * B=p.k is 128, 512 or 1024, exactly the ORIGINAL stripe count.
//   * One-dimensional grid of kv_heads*B groups. gqa2: 128 threads/group
//     (four SIMDs, two query heads/SIMD); gqa4: 64 threads/group
//     (two SIMDs, four query heads/SIMD). Every head still uses all 32 lanes.
//   * Original weight()/offset/dtype sink ABI: dtype=0/1/2 and offset aligned
//     to its 4/2/2-byte element. Buffer bounds/residency remain caller duties.
// Scratch: heads*B*130 FP32 words, {raw_o[128], max, sum}, unnormalized.
// Complete this producer before the ORIGINAL attention_mlx_merge consumes it.
// Host MUST select the original producer for unsupported input: a reject is
// not a fallback, and merging rejected/uninitialized scratch is invalid.
// Keep production math flags unchanged. Per-head source association/order is
// preserved; compilation can still change bits, so gate use on device tests.

template<uint B, uint H>
inline void attention_mlx_gqa_reuse_body(device const float *queries,
                                        device const float *keys,
                                        device const float *values,
                                        device const uchar *sink,
                                        device float *states,
                                        constant P &p,
                                        uint group, uint sg, uint lane) {
    static_assert(B == 128 || B == 512 || B == 1024, "unsupported GQA stripe count");
    static_assert(H == 2 || H == 4, "unsupported heads per SIMD");
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
            for (int j = 0; j < 4; ++j) kr[t][j] = keys[base + j];
        }
        for (int t = 0; t < 4; ++t) {
            const ulong base = (ulong(i + uint(t) * B) * p.kv_heads + kv_head) * 128 + lane * 4;
            for (int j = 0; j < 4; ++j) vr[t][j] = values[base + j];
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
        for (int j = 0; j < 4; ++j) kr[j] = keys[base + j];
        for (int j = 0; j < 4; ++j) vr[j] = values[base + j];
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

template<uint H>
inline void attention_mlx_gqa_reuse_dispatch(device const float *queries,
                                            device const float *keys,
                                            device const float *values,
                                            device const uchar *sink,
                                            device float *states,
                                            constant P &p,
                                            uint3 grid, uint3 tptg,
                                            uint sg, uint lane) {
    // Parameter checks and specialization selection are uniform across every
    // lane/SIMD. Check the bounded position before forming pos+1, including
    // UINT_MAX. No barriers or data-dependent rejection paths are introduced.
    if (!attention_mlx_valid(p) || p.heads / p.kv_heads != 8 ||
        p.window != 0 || p.pos >= 262144 || p.pos + 1 > p.capacity ||
        p.dtype > 2 || p.offset % (p.dtype == 0 ? 4 : 2) != 0 ||
        tptg.x != 32 * (8 / H) || tptg.y != 1 || tptg.z != 1 ||
        grid.y != 0 || grid.z != 0) return;
    switch (p.k) {
        case 128:
            attention_mlx_gqa_reuse_body<128, H>(queries, keys, values, sink, states, p, grid.x, sg, lane);
            break;
        case 512:
            attention_mlx_gqa_reuse_body<512, H>(queries, keys, values, sink, states, p, grid.x, sg, lane);
            break;
        case 1024:
            attention_mlx_gqa_reuse_body<1024, H>(queries, keys, values, sink, states, p, grid.x, sg, lane);
            break;
        default:
            return;
    }
}

kernel void attention_mlx_split_gqa2(device const float *queries [[buffer(0)]],
                                     device const float *keys [[buffer(1)]],
                                     device const float *values [[buffer(2)]],
                                     device const uchar *sink [[buffer(3)]],
                                     device float *states [[buffer(4)]],
                                     constant P &p [[buffer(8)]],
                                     uint3 grid [[threadgroup_position_in_grid]],
                                     uint3 tptg [[threads_per_threadgroup]],
                                     uint sg [[simdgroup_index_in_threadgroup]],
                                     uint lane [[thread_index_in_simdgroup]]) {
    attention_mlx_gqa_reuse_dispatch<2>(queries, keys, values, sink, states, p, grid, tptg, sg, lane);
}

kernel void attention_mlx_split_gqa4(device const float *queries [[buffer(0)]],
                                     device const float *keys [[buffer(1)]],
                                     device const float *values [[buffer(2)]],
                                     device const uchar *sink [[buffer(3)]],
                                     device float *states [[buffer(4)]],
                                     constant P &p [[buffer(8)]],
                                     uint3 grid [[threadgroup_position_in_grid]],
                                     uint3 tptg [[threads_per_threadgroup]],
                                     uint sg [[simdgroup_index_in_threadgroup]],
                                     uint lane [[thread_index_in_simdgroup]]) {
    attention_mlx_gqa_reuse_dispatch<4>(queries, keys, values, sink, states, p, grid, tptg, sg, lane);
}
