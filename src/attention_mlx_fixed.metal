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

// Append after attention_mlx.metal in the SAME production Metal compilation.
// Opt-in global NoPE decode specialization, not a different attention algorithm.
// Caller preconditions (unsupported shapes return without writing any state):
//   * FP32 Q/K/V; D=V=128; heads/kv_heads=8, kv_heads>0.
//   * window=0, pos<UINT_MAX, 1<=pos+1<=min(capacity,262144).
//   * B=p.k is exactly 128, 512 or 1024, with the ORIGINAL stripe partition.
//   * One-dimensional grid: kv_heads*B groups, 256 threads (eight SIMDs) each.
//   * Sink uses the original weight()/offset/dtype ABI and is present once/head.
// Scratch and synchronization are unchanged: heads*B*130 FP32 words, record
// {raw_o[128], max, sum}; use the ORIGINAL attention_mlx_merge after producer.
// No normalized scratch, quantization, altered FP32 association or math flags.
// Source association is preserved; bitwise device parity still requires tests.

template<uint B>
inline void attention_mlx_split_fixed_body(device const float *queries,
                                            device const float *keys,
                                            device const float *values,
                                            device const uchar *sink,
                                            device float *states,
                                            constant P &p,
                                            uint group, uint sg, uint lane) {
    static_assert(B == 128 || B == 512 || B == 1024, "unsupported fixed stripe count");
    const uint kv_head = group / B, block = group % B;
    if (kv_head >= p.kv_heads) return;
    const uint head = kv_head * 8 + sg;
    const uint n = p.pos + 1;
    float q[4], o[4] = {0};
    #pragma unroll
    for (int j = 0; j < 4; ++j) q[j] = p.scale * queries[ulong(head) * 128 + lane * 4 + j];
    float maximum = -3.4028234663852886e38f;
    float denom = 0;
    if (block == 0) {
        maximum = weight(sink, p.offset, p.dtype, head);
        denom = 1;
    }

    // Same TK=4 prefetch and consumption order as attention_mlx_split. B is a
    // compile-time stripe count, not a new chunk size. Every token is resident
    // at its linear index, so the original token%capacity is exactly token.
    uint i = block;
    for (; i + 3 * B < n; i += 4 * B) {
        float kr[4][4], vr[4][4];
        #pragma unroll
        for (int t = 0; t < 4; ++t) {
            const ulong base = (ulong(i + uint(t) * B) * p.kv_heads + kv_head) * 128 + lane * 4;
            #pragma unroll
            for (int j = 0; j < 4; ++j) kr[t][j] = keys[base + j];
        }
        #pragma unroll
        for (int t = 0; t < 4; ++t) {
            const ulong base = (ulong(i + uint(t) * B) * p.kv_heads + kv_head) * 128 + lane * 4;
            #pragma unroll
            for (int j = 0; j < 4; ++j) vr[t][j] = values[base + j];
        }
        #pragma unroll
        for (int t = 0; t < 4; ++t) {
            float score = 0;
            #pragma unroll
            for (int j = 0; j < 4; ++j) score += q[j] * kr[t][j];
            score = simd_sum(score);
            float next = max(maximum, score);
            float factor = fast::exp(maximum - next);
            float probability = fast::exp(score - next);
            maximum = next;
            denom = denom * factor + probability;
            #pragma unroll
            for (int j = 0; j < 4; ++j) o[j] = o[j] * factor + probability * vr[t][j];
        }
    }
    for (; i < n; i += B) {
        const ulong base = (ulong(i) * p.kv_heads + kv_head) * 128 + lane * 4;
        float score = 0;
        #pragma unroll
        for (int j = 0; j < 4; ++j) score += q[j] * keys[base + j];
        score = simd_sum(score);
        float next = max(maximum, score);
        float factor = fast::exp(maximum - next);
        float probability = fast::exp(score - next);
        maximum = next;
        denom = denom * factor + probability;
        #pragma unroll
        for (int j = 0; j < 4; ++j) o[j] = o[j] * factor + probability * values[base + j];
    }
    const ulong state = (ulong(head) * B + block) * 130;
    if (lane == 0) {
        states[state + 128] = maximum;
        states[state + 129] = denom;
    }
    #pragma unroll
    for (int j = 0; j < 4; ++j) states[state + lane * 4 + j] = o[j];
}

kernel void attention_mlx_split_fixed(device const float *queries [[buffer(0)]],
                                      device const float *keys [[buffer(1)]],
                                      device const float *values [[buffer(2)]],
                                      device const uchar *sink [[buffer(3)]],
                                      device float *states [[buffer(4)]],
                                      constant P &p [[buffer(8)]],
                                      uint3 grid [[threadgroup_position_in_grid]],
                                      uint3 tptg [[threads_per_threadgroup]],
                                      uint sg [[simdgroup_index_in_threadgroup]],
                                      uint lane [[thread_index_in_simdgroup]]) {
    // All rejects and B selection are uniform over the entire threadgroup.
    // Deliberately no generic fallback: the host must select the original
    // attention_mlx_split for every unsupported shape/window/ring/stripe count.
    if (!attention_mlx_valid(p) || p.heads / p.kv_heads != 8 ||
        p.window != 0 || p.pos >= 262144 || p.pos + 1 > p.capacity ||
        tptg.x != 256 || tptg.y != 1 || tptg.z != 1) return;
    switch (p.k) {
        case 128:
            attention_mlx_split_fixed_body<128>(queries, keys, values, sink, states, p, grid.x, sg, lane);
            break;
        case 512:
            attention_mlx_split_fixed_body<512>(queries, keys, values, sink, states, p, grid.x, sg, lane);
            break;
        case 1024:
            attention_mlx_split_fixed_body<1024>(queries, keys, values, sink, states, p, grid.x, sg, lane);
            break;
        default:
            return;
    }
}
