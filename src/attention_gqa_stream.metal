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

// Append AFTER attention_gqa_reuse.metal in the production Metal library.
// Opt-in ordinary global GQA2/GQA4 candidates: same ABI, guards, stripe partition,
// head mapping and FP32 math as attention_mlx_split_gqa2; only TK=4 becomes
// TK=1 or TK=2 to reduce live K/V prefetch registers (not a speed guarantee).
// Requirements: FP32 Q/K/V, D128, heads=kv_heads*8, kv_heads>0, window=0,
// 1<=pos+1<=min(capacity,262144), linearly resident K/V, learned sink per head
// using weight()/offset/dtype ABI (dtype 0/1/2, offset aligned to 4/2/2 bytes).
// B=p.k is the ORIGINAL 128/512/1024 stripe count, never the prefetch chunk.
// Dispatch kv_heads*B one-dimensional groups: GQA2 uses 128 threads (four
// SIMDs, two heads/SIMD); GQA4 uses 64 threads (two SIMDs, four heads/SIMD).
// Every head still uses all 32 lanes.
// Scratch remains heads*B*130 FP32 words: {raw_o[128], max, sum}, unnormalized.
// Complete the producer before the unchanged attention_mlx_merge consumes it.
// Unsupported parameters/groups uniformly return WITHOUT writing scratch;
// rejection is not fallback. The host must route unsupported inputs elsewhere.
// Buffer bounds/residency remain caller duties. Preserve production math flags
// and gate opt-in on device bit-parity tests; source order alone is not proof.

template<uint B, uint TK, uint H>
inline void attention_mlx_gqa_stream_body(device const float *queries,
                                        device const float *keys,
                                        device const float *values,
                                        device const uchar *sink,
                                        device float *states,
                                        constant P &p,
                                        uint group, uint sg, uint lane) {
    static_assert(B == 128 || B == 512 || B == 1024, "unsupported GQA stripe count");
    static_assert(TK == 1 || TK == 2, "unsupported prefetch chunk");
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

    // Change ONLY the prefetch chunk from the original TK=4 to TK=1/2.
    // Load TK keys, then TK values ONCE per SIMD and reuse them for H
    // queries. Each head still consumes block, block+B, ... in ascending
    // order with the original arithmetic and full 32-lane simd_sum.
    uint i = block;
    for (; i + (TK - 1) * B < n; i += TK * B) {
        float kr[TK][4], vr[TK][4];
        for (uint t = 0; t < TK; ++t) {
            const ulong base = (ulong(i + uint(t) * B) * p.kv_heads + kv_head) * 128 + lane * 4;
            for (int j = 0; j < 4; ++j) kr[t][j] = keys[base + j];
        }
        for (uint t = 0; t < TK; ++t) {
            const ulong base = (ulong(i + uint(t) * B) * p.kv_heads + kv_head) * 128 + lane * 4;
            for (int j = 0; j < 4; ++j) vr[t][j] = values[base + j];
        }
        for (uint t = 0; t < TK; ++t) {
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

template<uint TK, uint H>
inline void attention_mlx_gqa_stream_dispatch(device const float *queries,
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
            attention_mlx_gqa_stream_body<128, TK, H>(queries, keys, values, sink, states, p, grid.x, sg, lane);
            break;
        case 512:
            attention_mlx_gqa_stream_body<512, TK, H>(queries, keys, values, sink, states, p, grid.x, sg, lane);
            break;
        case 1024:
            attention_mlx_gqa_stream_body<1024, TK, H>(queries, keys, values, sink, states, p, grid.x, sg, lane);
            break;
        default:
            return;
    }
}

kernel void attention_mlx_split_gqa2_tk1(device const float *queries [[buffer(0)]],
                                     device const float *keys [[buffer(1)]],
                                     device const float *values [[buffer(2)]],
                                     device const uchar *sink [[buffer(3)]],
                                     device float *states [[buffer(4)]],
                                     constant P &p [[buffer(8)]],
                                     uint3 grid [[threadgroup_position_in_grid]],
                                     uint3 tptg [[threads_per_threadgroup]],
                                     uint sg [[simdgroup_index_in_threadgroup]],
                                     uint lane [[thread_index_in_simdgroup]]) {
    attention_mlx_gqa_stream_dispatch<1, 2>(queries, keys, values, sink, states, p, grid, tptg, sg, lane);
}

kernel void attention_mlx_split_gqa2_tk2(device const float *queries [[buffer(0)]],
                                     device const float *keys [[buffer(1)]],
                                     device const float *values [[buffer(2)]],
                                     device const uchar *sink [[buffer(3)]],
                                     device float *states [[buffer(4)]],
                                     constant P &p [[buffer(8)]],
                                     uint3 grid [[threadgroup_position_in_grid]],
                                     uint3 tptg [[threads_per_threadgroup]],
                                     uint sg [[simdgroup_index_in_threadgroup]],
                                     uint lane [[thread_index_in_simdgroup]]) {
    attention_mlx_gqa_stream_dispatch<2, 2>(queries, keys, values, sink, states, p, grid, tptg, sg, lane);
}

kernel void attention_mlx_split_gqa4_tk1(device const float *queries [[buffer(0)]],
                                     device const float *keys [[buffer(1)]],
                                     device const float *values [[buffer(2)]],
                                     device const uchar *sink [[buffer(3)]],
                                     device float *states [[buffer(4)]],
                                     constant P &p [[buffer(8)]],
                                     uint3 grid [[threadgroup_position_in_grid]],
                                     uint3 tptg [[threads_per_threadgroup]],
                                     uint sg [[simdgroup_index_in_threadgroup]],
                                     uint lane [[thread_index_in_simdgroup]]) {
    attention_mlx_gqa_stream_dispatch<1, 4>(queries, keys, values, sink, states, p, grid, tptg, sg, lane);
}

kernel void attention_mlx_split_gqa4_tk2(device const float *queries [[buffer(0)]],
                                     device const float *keys [[buffer(1)]],
                                     device const float *values [[buffer(2)]],
                                     device const uchar *sink [[buffer(3)]],
                                     device float *states [[buffer(4)]],
                                     constant P &p [[buffer(8)]],
                                     uint3 grid [[threadgroup_position_in_grid]],
                                     uint3 tptg [[threads_per_threadgroup]],
                                     uint sg [[simdgroup_index_in_threadgroup]],
                                     uint lane [[thread_index_in_simdgroup]]) {
    attention_mlx_gqa_stream_dispatch<2, 4>(queries, keys, values, sink, states, p, grid, tptg, sg, lane);
}
