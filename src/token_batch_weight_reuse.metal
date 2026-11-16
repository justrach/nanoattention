// Append after token_batch.metal (and its kernels/resident_plan/row_tiling dependencies).
// Bounded matched-math INT4/g64 candidate; no fallback and no residual fusion.
// ABI: buffers 0/1/8 and P match token_batch_qkv / token_batch_matvec.
// p.m=batch 1..32, p.n=row tile 1/2/4/8. Matvec p.bits: 0=O, 1=head.
// Grid: ceil(batch/2)*sum(ceil(rows/tile))*32, group=128, SIMD width=32.
// Pair is the fastest SIMD index. Tail SIMD groups are safe. Host must provide
// disjoint token allocations, adequate per-projection input lengths, resident
// indirect resources, and a total grid representable by uint thread indices.
#ifndef FLINT_TOKEN_BATCH_WEIGHT_REUSE_METAL
#define FLINT_TOKEN_BATCH_WEIGHT_REUSE_METAL

inline bool token_weight2_matrix(Matrix m) {
#if !FLINT_MLX_MATH
    return false; // This candidate is deliberately matched-math only.
#else
    return m.reserved == 1 && m.bits == 4 && m.group_size == 64 &&
           m.cols != 0 && m.cols % 64 == 0 && m.w && m.scales && m.biases &&
           m.scale_dtype <= 2 && m.bias_dtype <= 2 &&
           ulong(m.rows) * (m.cols / 64) <= 0x100000000ul;
#endif
}

// Exactly matrix_dot_rows' matched quartet preprocessing and accumulation for
// each token. Only the codes and affine loads are shared between the tokens.
template<uint Rows>
inline void token_weight2_rows(Matrix m, device const float *x0,
                               device const float *x1, device float *out0,
                               device float *out1, bool second,
                               uint first, uint lane) {
    float sums0[Rows], sums1[Rows];
    #pragma unroll
    for (uint r = 0; r < Rows; ++r) { sums0[r] = 0; sums1[r] = 0; }
    uint row_words = m.cols / 8, row_groups = m.cols / 64;
    device const uint *packed = (device const uint *)m.w;
    for (uint segment = lane; segment < m.cols / 16; segment += 32) {
        float values0[16], values1[16], input_sum0 = 0, input_sum1 = 0;
        #pragma unroll
        for (uint j = 0; j < 16; j += 4) {
            float value0 = x0[segment * 16 + j], value1 = x0[segment * 16 + j + 1];
            float value2 = x0[segment * 16 + j + 2], value3 = x0[segment * 16 + j + 3];
            input_sum0 += value0 + value1 + value2 + value3;
            values0[j] = value0;
            values0[j + 1] = value1 / 16.0f;
            values0[j + 2] = value2 / 256.0f;
            values0[j + 3] = value3 / 4096.0f;
            if (second) {
                float other0 = x1[segment * 16 + j], other1 = x1[segment * 16 + j + 1];
                float other2 = x1[segment * 16 + j + 2], other3 = x1[segment * 16 + j + 3];
                input_sum1 += other0 + other1 + other2 + other3;
                values1[j] = other0;
                values1[j + 1] = other1 / 16.0f;
                values1[j + 2] = other2 / 256.0f;
                values1[j + 3] = other3 / 4096.0f;
            }
        }
        #pragma unroll
        for (uint r = 0; r < Rows; ++r) {
            uint row = first + r;
            if (row >= m.rows) continue;
            ulong base = ulong(row) * row_words + segment * 2;
            uint codes0 = packed[base], codes1 = packed[base + 1];
            uint group = row * row_groups + segment / 4;
            float scale = weight(m.scales, 0, m.scale_dtype, group);
            float bias = weight(m.biases, 0, m.bias_dtype, group);
            float dot0 = 0, dot1 = 0;
            #pragma unroll
            for (uint j = 0; j < 16; j += 4) {
                uint codes = j < 8 ? codes0 : codes1;
                uint half_word = (j & 7u) < 4 ? codes : codes >> 16;
                dot0 += (values0[j] * float(half_word & 0x000fu) +
                         values0[j + 1] * float(half_word & 0x00f0u) +
                         values0[j + 2] * float(half_word & 0x0f00u) +
                         values0[j + 3] * float(half_word & 0xf000u));
                if (second)
                    dot1 += (values1[j] * float(half_word & 0x000fu) +
                             values1[j + 1] * float(half_word & 0x00f0u) +
                             values1[j + 2] * float(half_word & 0x0f00u) +
                             values1[j + 3] * float(half_word & 0xf000u));
            }
            sums0[r] += scale * dot0 + input_sum0 * bias;
            if (second) sums1[r] += scale * dot1 + input_sum1 * bias;
        }
    }
    #pragma unroll
    for (uint r = 0; r < Rows; ++r) {
        float sum0 = simd_sum(sums0[r]);
        if (lane == 0 && first + r < m.rows) out0[first + r] = sum0;
        if (second) {
            float sum1 = simd_sum(sums1[r]);
            if (lane == 0 && first + r < m.rows) out1[first + r] = sum1;
        }
    }
}

inline void token_weight2_dispatch(Matrix m, device const float *x0,
                                   device const float *x1, device float *out0,
                                   device float *out1, bool second,
                                   uint tile, uint rows, uint lane) {
    switch (rows) {
        case 1: token_weight2_rows<1>(m, x0, x1, out0, out1, second, tile, lane); break;
        case 2: token_weight2_rows<2>(m, x0, x1, out0, out1, second, tile * 2, lane); break;
        case 4: token_weight2_rows<4>(m, x0, x1, out0, out1, second, tile * 4, lane); break;
        case 8: token_weight2_rows<8>(m, x0, x1, out0, out1, second, tile * 8, lane); break;
    }
}

kernel void token_batch_qkv_weight2(device const Matrix *matrices [[buffer(0)]],
                                    device const TokenScratch *tokens [[buffer(1)]],
                                    constant P &p [[buffer(8)]],
                                    uint i [[thread_position_in_grid]],
                                    uint lane [[thread_index_in_simdgroup]]) {
    if (!token_batch_count(p.m) || !token_batch_tile(p.n)) return;
    // Validate each projection independently; their column counts may differ.
    for (uint projection = 0; projection < 3; ++projection)
        if (!token_weight2_matrix(matrices[projection])) return;
    ulong qt = token_batch_tiles(matrices[0].rows, p.n);
    ulong kt = token_batch_tiles(matrices[1].rows, p.n);
    ulong tiles = qt + kt + token_batch_tiles(matrices[2].rows, p.n);
    uint pairs = (p.m + 1) / 2;
    if (!token_batch_geometry(tiles * pairs)) return;
    ulong simd = i / 32;
    if (simd >= tiles * pairs) return;
    uint token = uint(simd % pairs) * 2;
    ulong tile = simd / pairs;
    uint projection = tile < qt ? 0 : (tile < qt + kt ? 1 : 2);
    tile -= projection == 0 ? 0 : (projection == 1 ? qt : qt + kt);
    device const TokenScratch &s0 = tokens[token];
    if (!s0.norm || !s0.q || !s0.k || !s0.v) return;
    device const float *x1 = s0.norm;
    device float *out0 = projection == 0 ? s0.q : (projection == 1 ? s0.k : s0.v);
    device float *out1 = out0;
    bool second = token + 1 < p.m;
    if (second) {
        device const TokenScratch &s1 = tokens[token + 1];
        if (!s1.norm || !s1.q || !s1.k || !s1.v) return;
        x1 = s1.norm;
        out1 = projection == 0 ? s1.q : (projection == 1 ? s1.k : s1.v);
    }
    token_weight2_dispatch(matrices[projection], s0.norm, x1, out0, out1,
                           second, uint(tile), p.n, lane);
}

kernel void token_batch_matvec_weight2(device const Matrix &m [[buffer(0)]],
                                       device const TokenScratch *tokens [[buffer(1)]],
                                       constant P &p [[buffer(8)]],
                                       uint i [[thread_position_in_grid]],
                                       uint lane [[thread_index_in_simdgroup]]) {
    if (!token_batch_count(p.m) || !token_batch_tile(p.n) || p.bits > 1 ||
        !token_weight2_matrix(m)) return;
    ulong tiles = token_batch_tiles(m.rows, p.n);
    uint pairs = (p.m + 1) / 2;
    if (!token_batch_geometry(tiles * pairs)) return;
    ulong simd = i / 32;
    if (simd >= tiles * pairs) return;
    uint token = uint(simd % pairs) * 2;
    device const TokenScratch &s0 = tokens[token];
    device const float *x0 = p.bits == 0 ? s0.attention : s0.norm;
    device float *out0 = p.bits == 0 ? s0.projection : s0.logits;
    if (!x0 || !out0) return;
    device const float *x1 = x0;
    device float *out1 = out0;
    bool second = token + 1 < p.m;
    if (second) {
        device const TokenScratch &s1 = tokens[token + 1];
        x1 = p.bits == 0 ? s1.attention : s1.norm;
        out1 = p.bits == 0 ? s1.projection : s1.logits;
        if (!x1 || !out1) return;
    }
    token_weight2_dispatch(m, x0, x1, out0, out1, second, uint(simd / pairs), p.n, lane);
}
#endif
