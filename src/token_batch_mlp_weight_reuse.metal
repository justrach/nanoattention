// Append after token_batch.metal and its dependencies. No prior weight2 source
// is required. Buffers 0=MLP[], 1=TokenScratch[], 8=P retain the old batch ABI.
// P: heads=batch 1..32, bits=tile 1/2/4/8, n=routed count, m=shared expert,
// k=activation stride, dim=hidden. Pair is fastest SIMD index, group=128.
// Gate grid: ceil(batch/2)*(n+1)*ceil(k/tile)*32.
// Down grid: ceil(batch/2)*ceil(dim/tile)*32.
// Host validates expert IDs/allocation lengths and makes indirect resources
// resident. Tokens must have disjoint writable allocations. No absent partner
// is dereferenced. Unsupported sharing cases retain the original helpers.
#ifndef FLINT_TOKEN_BATCH_MLP_WEIGHT_REUSE_METAL
#define FLINT_TOKEN_BATCH_MLP_WEIGHT_REUSE_METAL

inline bool mlpw2_shareable(Matrix m) {
#if FLINT_MLX_MATH
    return m.reserved == 1 && m.bits == 4 && m.group_size == 64 &&
           m.cols != 0 && m.cols % 64 == 0 && m.w && m.scales && m.biases &&
           m.scale_dtype <= 2 && m.bias_dtype <= 2 &&
           ulong(m.rows) * (m.cols / 64) <= 0x100000000ul;
#else
    return false;
#endif
}

// Count=2 shares the activation preprocessing across gate and up as well as
// sharing each matrix's weight loads across the two tokens. Count=1 is down.
// Results are lane-local; every caller keeps the original SIMD reduction.
template<uint Rows, uint Count>
inline void mlpw2_dot(Matrix a, Matrix b, device const float *x0,
                      device const float *x1, uint first, uint lane,
                      thread float *sums0, thread float *sums1) {
    #pragma unroll
    for (uint r = 0; r < Rows * Count; ++r) { sums0[r] = 0; sums1[r] = 0; }
    for (uint segment = lane; segment < a.cols / 16; segment += 32) {
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
            float other0 = x1[segment * 16 + j], other1 = x1[segment * 16 + j + 1];
            float other2 = x1[segment * 16 + j + 2], other3 = x1[segment * 16 + j + 3];
            input_sum1 += other0 + other1 + other2 + other3;
            values1[j] = other0;
            values1[j + 1] = other1 / 16.0f;
            values1[j + 2] = other2 / 256.0f;
            values1[j + 3] = other3 / 4096.0f;
        }
        #pragma unroll
        for (uint which = 0; which < Count; ++which) {
            Matrix m = which == 0 ? a : b;
            uint row_words = m.cols / 8, row_groups = m.cols / 64;
            device const uint *packed = (device const uint *)m.w;
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
                    dot1 += (values1[j] * float(half_word & 0x000fu) +
                             values1[j + 1] * float(half_word & 0x00f0u) +
                             values1[j + 2] * float(half_word & 0x0f00u) +
                             values1[j + 3] * float(half_word & 0xf000u));
                }
                sums0[which * Rows + r] += scale * dot0 + input_sum0 * bias;
                sums1[which * Rows + r] += scale * dot1 + input_sum1 * bias;
            }
        }
    }
}

inline bool mlpw2_gate_valid(device const TokenScratch &s, device const MLP *experts,
                             constant P &p, uint route) {
    if (!s.norm || !s.activation || (p.n && !s.route_ids)) return false;
    uint expert = route < p.n ? s.route_ids[route] : p.m;
    device const MLP &m = experts[expert];
    return m.gate.rows <= p.k && m.up.rows == m.gate.rows && m.up.cols == m.gate.cols;
}

template<uint Rows>
inline void mlpw2_gate(device const MLP *experts, device const TokenScratch &s0,
                       device const TokenScratch &s1, bool second,
                       constant P &p, uint local_tile, uint lane) {
    uint route_tiles = uint(token_batch_tiles(p.k, Rows));
    uint route = local_tile / route_tiles;
    bool valid0 = mlpw2_gate_valid(s0, experts, p, route);
    bool valid1 = second && mlpw2_gate_valid(s1, experts, p, route);
    uint e0 = valid0 ? (route < p.n ? s0.route_ids[route] : p.m) : 0;
    uint e1 = valid1 ? (route < p.n ? s1.route_ids[route] : p.m) : 0;
    if (valid0 && valid1 && e0 == e1 && mlpw2_shareable(experts[e0].gate) &&
        mlpw2_shareable(experts[e0].up)) {
        device const MLP &mlp = experts[e0];
        uint first = (local_tile % route_tiles) * Rows;
        if (first >= mlp.gate.rows) return;
        float sums0[Rows * 2], sums1[Rows * 2];
        mlpw2_dot<Rows, 2>(mlp.gate, mlp.up, s0.norm, s1.norm, first, lane, sums0, sums1);
        #pragma unroll
        for (uint r = 0; r < Rows; ++r) {
            float gate0 = simd_sum(sums0[r]), up0 = simd_sum(sums0[Rows + r]);
            float gate1 = simd_sum(sums1[r]), up1 = simd_sum(sums1[Rows + r]);
            if (lane == 0 && first + r < mlp.gate.rows) {
                s0.activation[route * p.k + first + r] = swiglu_product(gate0, up0);
                s1.activation[route * p.k + first + r] = swiglu_product(gate1, up1);
            }
        }
    } else {
        // Same original local scalar index for each token, not pair index.
        if (valid0) staged_gate_up<Rows>(experts, s0.norm, s0.route_ids, s0.activation, p, local_tile * 32, lane);
        if (valid1) staged_gate_up<Rows>(experts, s1.norm, s1.route_ids, s1.activation, p, local_tile * 32, lane);
    }
}

inline bool mlpw2_down_valid(device const TokenScratch &s, device const MLP *experts,
                             constant P &p) {
    if (!s.activation || !s.x || (p.n && (!s.route_ids || !s.route_weights))) return false;
    for (uint route = 0; route <= p.n; ++route) {
        uint expert = route < p.n ? s.route_ids[route] : p.m;
        if (experts[expert].down.rows != p.dim || experts[expert].down.cols > p.k) return false;
    }
    return true;
}

template<uint Rows>
inline void mlpw2_down(device const MLP *experts, device const TokenScratch &s0,
                       device const TokenScratch &s1, bool second,
                       constant P &p, uint tile, uint lane) {
    bool valid0 = mlpw2_down_valid(s0, experts, p);
    bool valid1 = second && mlpw2_down_valid(s1, experts, p);
    uint first = tile * Rows;
    if (!valid0 || !valid1) {
        if (valid0) staged_down<Rows>(experts, s0.activation, s0.route_ids, s0.route_weights, s0.x, p, tile * 32, lane);
        if (valid1) staged_down<Rows>(experts, s1.activation, s1.route_ids, s1.route_weights, s1.x, p, tile * 32, lane);
        return;
    }
    float outputs0[Rows], outputs1[Rows];
    #pragma unroll
    for (uint r = 0; r < Rows; ++r) {
#if FLINT_MLX_MATH
        outputs0[r] = 0; outputs1[r] = 0;
#else
        outputs0[r] = lane == 0 && first + r < p.dim ? s0.x[first + r] : 0;
        outputs1[r] = lane == 0 && first + r < p.dim ? s1.x[first + r] : 0;
#endif
    }
    for (uint route = 0; route <= p.n; ++route) {
        uint e0 = route < p.n ? s0.route_ids[route] : p.m;
        uint e1 = route < p.n ? s1.route_ids[route] : p.m;
        float sums0[Rows], sums1[Rows];
        device const float *x0 = s0.activation + route * p.k;
        device const float *x1 = s1.activation + route * p.k;
        if (e0 == e1 && mlpw2_shareable(experts[e0].down)) {
            mlpw2_dot<Rows, 1>(experts[e0].down, experts[e0].down, x0, x1, first, lane, sums0, sums1);
        } else {
            matrix_dot_rows<Rows>(experts[e0].down, x0, first, lane, sums0);
            matrix_dot_rows<Rows>(experts[e1].down, x1, first, lane, sums1);
        }
        #pragma unroll
        for (uint r = 0; r < Rows; ++r) {
            float sum0 = simd_sum(sums0[r]);
            float sum1 = simd_sum(sums1[r]);
            if (lane == 0) {
                outputs0[r] += mixture_product(sum0, route < p.n ? s0.route_weights[route] : 1.0f);
                outputs1[r] += mixture_product(sum1, route < p.n ? s1.route_weights[route] : 1.0f);
            }
        }
    }
    #pragma unroll
    for (uint r = 0; r < Rows; ++r) {
        if (lane == 0 && first + r < p.dim) {
#if FLINT_MLX_MATH
            s0.x[first + r] = s0.x[first + r] + outputs0[r];
            s1.x[first + r] = s1.x[first + r] + outputs1[r];
#else
            s0.x[first + r] = outputs0[r];
            s1.x[first + r] = outputs1[r];
#endif
        }
    }
}

kernel void token_batch_gate_up_weight2(device const MLP *experts [[buffer(0)]],
                                        device const TokenScratch *tokens [[buffer(1)]],
                                        constant P &p [[buffer(8)]],
                                        uint i [[thread_position_in_grid]],
                                        uint lane [[thread_index_in_simdgroup]]) {
    if (!token_batch_count(p.heads) || !token_batch_tile(p.bits) || !p.k ||
        p.n > STAGED_ROUTER_SLOTS) return;
    uint pairs = (p.heads + 1) / 2;
    ulong tiles = (ulong(p.n) + 1) * token_batch_tiles(p.k, p.bits);
    if (!token_batch_geometry(tiles * pairs) || ulong(p.n + 1) * p.k > 0x100000000ul) return;
    ulong simd = i / 32;
    if (simd >= tiles * pairs) return;
    uint token = uint(simd % pairs) * 2, local_tile = uint(simd / pairs);
    bool second = token + 1 < p.heads;
    device const TokenScratch &s0 = tokens[token];
    device const TokenScratch &s1 = tokens[second ? token + 1 : token];
    switch (p.bits) {
        case 1: mlpw2_gate<1>(experts, s0, s1, second, p, local_tile, lane); break;
        case 2: mlpw2_gate<2>(experts, s0, s1, second, p, local_tile, lane); break;
        case 4: mlpw2_gate<4>(experts, s0, s1, second, p, local_tile, lane); break;
        case 8: mlpw2_gate<8>(experts, s0, s1, second, p, local_tile, lane); break;
    }
}

kernel void token_batch_down_weight2(device const MLP *experts [[buffer(0)]],
                                     device const TokenScratch *tokens [[buffer(1)]],
                                     constant P &p [[buffer(8)]],
                                     uint i [[thread_position_in_grid]],
                                     uint lane [[thread_index_in_simdgroup]]) {
    if (!token_batch_count(p.heads) || !token_batch_tile(p.bits) || !p.k ||
        p.n > STAGED_ROUTER_SLOTS) return;
    uint pairs = (p.heads + 1) / 2;
    ulong tiles = token_batch_tiles(p.dim, p.bits);
    if (!token_batch_geometry(tiles * pairs) || ulong(p.n + 1) * p.k > 0x100000000ul) return;
    ulong simd = i / 32;
    if (simd >= tiles * pairs) return;
    uint token = uint(simd % pairs) * 2, tile = uint(simd / pairs);
    bool second = token + 1 < p.heads;
    device const TokenScratch &s0 = tokens[token];
    device const TokenScratch &s1 = tokens[second ? token + 1 : token];
    switch (p.bits) {
        case 1: mlpw2_down<1>(experts, s0, s1, second, p, tile, lane); break;
        case 2: mlpw2_down<2>(experts, s0, s1, second, p, tile, lane); break;
        case 4: mlpw2_down<4>(experts, s0, s1, second, p, tile, lane); break;
        case 8: mlpw2_down<8>(experts, s0, s1, second, p, tile, lane); break;
    }
}
#endif
