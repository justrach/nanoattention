/*
 * Adapted from attention_gqa_reuse.metal, attention_mlx.metal, and
 * MLX v0.32.4 mlx/backend/metal/kernels/sdpa_vector.h.
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

// Append after token_batch.metal, kv_bf16.metal and the MLX/GQA helpers.
// Global cache only; batch count 1..32. All batch KV positions must be written
// before this split. Query groups remain 2/4/8 regardless of batch count.
// No TokenScratch ABI change: states is a separate contiguous buffer with
// record ((token*heads+head)*B+block)*130 = {raw_o[128], max, sum}.
// The active layout uses P.k as B, not the allocation's maximum stripe count.
// Split: 128 threads/group, kv_heads*B*ceil(batch/Queries) 1-D groups.
// Queries=2 for the original split, 4/8 for experimental _q4/_q8 entries.
// Group order: query group, KV head, stripe (stripe fastest). Each SIMD owns
// two GQA heads and at most Queries positions, reusing each K/V load up to
// 2*Queries ways. Each query/head retains all 32 lanes and its recurrence.
// Larger query groups increase register pressure; they are not presumed faster.
// Merge: 1024 threads/group, batch*heads one-dimensional groups.
// Scratch Q/output pointers must be disjoint per token, valid, and resident;
// buffer allocation bounds/residency remain host duties. Reject is NOT fallback.
#ifndef FLINT_TOKEN_BATCH_ATTENTION_METAL
#define FLINT_TOKEN_BATCH_ATTENTION_METAL

inline bool token_batch_attention_valid(constant P &p) {
    // Widen before adding positions; reject UINT_MAX and nonresident ranges
    // uniformly before reading TokenScratch, cache, sink, or producer states.
    return p.m >= 1 && p.m <= 32 && p.heads == 32 && p.kv_heads == 4 &&
        p.dim == 128 && p.window == 0 && p.capacity > 0 && p.bits <= 1 &&
        p.pos < 262144 && ulong(p.pos) + p.m <= min(ulong(p.capacity), 262144ul) &&
        (p.k == 128 || p.k == 512 || p.k == 1024) &&
        p.dtype <= 2 && p.offset % (p.dtype == 0 ? 4 : 2) == 0;
}

template<bool BF16>
inline void token_batch_attention_load(device const uchar *data, ulong base,
                                        thread float *result) {
    if (BF16) {
        device const ushort *cache = reinterpret_cast<device const ushort *>(data);
        for (int j = 0; j < 4; ++j) result[j] = kv_bf16_to_float(cache[base + j]);
    } else {
        device const float *cache = reinterpret_cast<device const float *>(data);
        for (int j = 0; j < 4; ++j) result[j] = cache[base + j];
    }
}

template<uint B, uint Queries, bool BF16>
inline void token_batch_attention_body(device const TokenScratch *tokens,
    device const uchar *keys, device const uchar *values, device const uchar *sink,
    device float *states, constant P &p, uint group, uint sg, uint lane) {
    static_assert(B == 128 || B == 512 || B == 1024, "Unsupported batch stripe count");
    static_assert(Queries == 2 || Queries == 4 || Queries == 8, "Unsupported batch query group");
    const uint query_group = group / (4 * B);
    const uint kv_head = (group / B) % 4, block = group % B;
    const uint first_token = query_group * Queries;
    if (first_token >= p.m) return;
    const uint count = min(Queries, p.m - first_token);
    const uint first_head = kv_head * 8 + sg * 2;
    const uint n = p.pos + first_token + count;
    for (uint r = 0; r < count; ++r)
        if (!tokens[first_token + r].q) return;
    float q[Queries][2][4], o[Queries][2][4];
    float maximum[Queries][2], denom[Queries][2];
    #pragma unroll
    for (uint r = 0; r < Queries; ++r) {
        if (r >= count) continue;
        device const float *queries = tokens[first_token + r].q;
        #pragma unroll
        for (uint h = 0; h < 2; ++h) {
            const uint head = first_head + h;
            for (int j = 0; j < 4; ++j) q[r][h][j] = p.scale * queries[ulong(head) * 128 + lane * 4 + j];
            for (int j = 0; j < 4; ++j) o[r][h][j] = 0;
            maximum[r][h] = -3.4028234663852886e38f;
            denom[r][h] = 0;
            if (block == 0) {
                maximum[r][h] = weight(sink, p.offset, p.dtype, head);
                denom[r][h] = 1;
            }
        }
    }

    // TK4: four keys first, then four values, shared across both heads and
    // positions. Later queries may need extra causal-prefix elements;
    // an earlier query NEVER uses those values, even if NaN/Inf. Each
    // valid query consumes only block, block+B, ... < pos+token+1, in order.
    uint i = block;
    for (; i + 3 * B < n; i += 4 * B) {
        float kr[4][4], vr[4][4];
        for (int t = 0; t < 4; ++t) {
            const ulong base = (ulong(i + uint(t) * B) * 4 + kv_head) * 128 + lane * 4;
            token_batch_attention_load<BF16>(keys, base, kr[t]);
        }
        for (int t = 0; t < 4; ++t) {
            const ulong base = (ulong(i + uint(t) * B) * 4 + kv_head) * 128 + lane * 4;
            token_batch_attention_load<BF16>(values, base, vr[t]);
        }
        for (int t = 0; t < 4; ++t) {
            #pragma unroll
            for (uint r = 0; r < Queries; ++r) {
                if (r >= count || i + uint(t) * B >= p.pos + first_token + r + 1) continue;
                #pragma unroll
                for (uint h = 0; h < 2; ++h) {
                    float score = 0;
                    for (int j = 0; j < 4; ++j) score += q[r][h][j] * kr[t][j];
                    score = simd_sum(score);
                    float next = max(maximum[r][h], score);
                    float factor = fast::exp(maximum[r][h] - next);
                    float probability = fast::exp(score - next);
                    maximum[r][h] = next;
                    denom[r][h] = denom[r][h] * factor + probability;
                    for (int j = 0; j < 4; ++j) o[r][h][j] = o[r][h][j] * factor + probability * vr[t][j];
                }
            }
        }
    }
    for (; i < n; i += B) {
        const ulong base = (ulong(i) * 4 + kv_head) * 128 + lane * 4;
        float kr[4], vr[4];
        token_batch_attention_load<BF16>(keys, base, kr);
        token_batch_attention_load<BF16>(values, base, vr);
        #pragma unroll
        for (uint r = 0; r < Queries; ++r) {
            if (r >= count || i >= p.pos + first_token + r + 1) continue;
            #pragma unroll
            for (uint h = 0; h < 2; ++h) {
                float score = 0;
                for (int j = 0; j < 4; ++j) score += q[r][h][j] * kr[j];
                score = simd_sum(score);
                float next = max(maximum[r][h], score);
                float factor = fast::exp(maximum[r][h] - next);
                float probability = fast::exp(score - next);
                maximum[r][h] = next;
                denom[r][h] = denom[r][h] * factor + probability;
                for (int j = 0; j < 4; ++j) o[r][h][j] = o[r][h][j] * factor + probability * vr[j];
            }
        }
    }
    #pragma unroll
    for (uint r = 0; r < Queries; ++r) {
        if (r >= count) continue;
        #pragma unroll
        for (uint h = 0; h < 2; ++h) {
            const ulong state = ((ulong(first_token + r) * 32 + first_head + h) * B + block) * 130;
            if (lane == 0) {
                states[state + 128] = maximum[r][h];
                states[state + 129] = denom[r][h];
            }
            for (int j = 0; j < 4; ++j) states[state + lane * 4 + j] = o[r][h][j];
        }
    }
}

template<uint Queries, bool BF16>
inline void token_batch_attention_dispatch(device const TokenScratch *tokens,
    device const uchar *keys, device const uchar *values, device const uchar *sink,
    device float *states, constant P &p, uint group, uint sg, uint lane) {
    switch (p.k) {
        case 128: token_batch_attention_body<128, Queries, BF16>(tokens, keys, values, sink, states, p, group, sg, lane); break;
        case 512: token_batch_attention_body<512, Queries, BF16>(tokens, keys, values, sink, states, p, group, sg, lane); break;
        case 1024: token_batch_attention_body<1024, Queries, BF16>(tokens, keys, values, sink, states, p, group, sg, lane); break;
    }
}

template<uint Queries>
inline void token_batch_attention_entry(device const TokenScratch *tokens,
    device const uchar *keys, device const uchar *values, device const uchar *sink,
    device float *states, constant P &p, uint3 grid, uint3 tptg, uint sg, uint lane) {
    static_assert(Queries == 2 || Queries == 4 || Queries == 8, "Unsupported batch query group");
    if (!token_batch_attention_valid(p) || tptg.x != 128 || tptg.y != 1 || tptg.z != 1 ||
        grid.y != 0 || grid.z != 0 || grid.x >= 4 * p.k * ((p.m + Queries - 1) / Queries)) return;
    if (p.bits)
        token_batch_attention_dispatch<Queries, true>(tokens, keys, values, sink, states, p, grid.x, sg, lane);
    else
        token_batch_attention_dispatch<Queries, false>(tokens, keys, values, sink, states, p, grid.x, sg, lane);
}

// These differ ONLY in query grouping/grid geometry, never the state ABI or
// the per-query score/softmax/merge association. Keep the original name at Q2.
#define TOKEN_BATCH_ATTENTION_ENTRY(Name, Queries) \
kernel void Name(device const TokenScratch *tokens [[buffer(0)]], \
    device const uchar *keys [[buffer(1)]], device const uchar *values [[buffer(2)]], \
    device const uchar *sink [[buffer(3)]], device float *states [[buffer(4)]], \
    constant P &p [[buffer(8)]], uint3 grid [[threadgroup_position_in_grid]], \
    uint3 tptg [[threads_per_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]], \
    uint lane [[thread_index_in_simdgroup]]) { \
    token_batch_attention_entry<Queries>(tokens, keys, values, sink, states, p, grid, tptg, sg, lane); \
}
TOKEN_BATCH_ATTENTION_ENTRY(token_batch_attention_split, 2)
TOKEN_BATCH_ATTENTION_ENTRY(token_batch_attention_split_q4, 4)
TOKEN_BATCH_ATTENTION_ENTRY(token_batch_attention_split_q8, 8)
#undef TOKEN_BATCH_ATTENTION_ENTRY

// Original attention_mlx_merge arithmetic, association, and transpose barriers.
// Only the state-record base and final output pointer include a token index.
kernel void token_batch_attention_merge(device const float *states [[buffer(0)]],
    device const TokenScratch *tokens [[buffer(1)]], constant P &p [[buffer(8)]],
    uint3 grid [[threadgroup_position_in_grid]], uint3 tptg [[threads_per_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    if (!token_batch_attention_valid(p) || grid.x >= p.m * p.heads ||
        tptg.x != 1024 || tptg.y != 1 || tptg.z != 1 || grid.y != 0 || grid.z != 0) return;
    const uint token = grid.x / p.heads, head = grid.x % p.heads;
    device float *out = tokens[token].attention;
    if (!out) return;
    const ulong first = (ulong(token) * p.heads + head) * p.k * 130;
    float o[4] = {0};
    float denom = 0;
    float maximum = -3.4028234663852886e38f;
    threadgroup float outputs[32 * 32];
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
#endif
