// Append after row_tiling.metal. Fixed-column candidate, same row ownership,
// descriptor ABI, FP32 expressions, lane segment order and SIMD reductions.
// reserved == 1 retains the original descriptor's INT4/g64 contract. Unsupported
// widths and other descriptor paths call the unmodified generic helpers.
// This changes only integer addressing/trip counts, with explicit outer unroll;
// no math mode, reduction, sigmoid, mixture or residual-order changes.

template<uint Rows, uint K>
inline void matrix_dot_rows_width(Matrix m, device const float *x,
                            uint first, uint lane, thread float *sums) {
    static_assert(Rows == RESIDENT_ROW_TILE_1 || Rows == RESIDENT_ROW_TILE_2 ||
                  Rows == RESIDENT_ROW_TILE_4 || Rows == RESIDENT_ROW_TILE_8, "Uncertified row tile");
    #pragma unroll
    for (uint r = 0; r < Rows; ++r) sums[r] = 0;
    if (m.reserved != 1) {
        #pragma unroll
        for (uint r = 0; r < Rows; ++r)
            if (first + r < m.rows) sums[r] = matrix_dot(m, x, first + r, lane);
        return;
    }
    static_assert(K == 512 || K == 1024 || K == 2048 || K == 4096 || K == 6144, "Unsupported fixed width");
    constexpr uint row_words = K / 8, row_groups = K / 64;
    device const uint *packed = (device const uint *)m.w;
    // Every lane visits precisely the same ordered segments as the generic loop.
    #pragma unroll
    for (uint step = 0; step < K / 512; ++step) {
        uint segment = lane + step * 32;
        float values[16], input_sum = 0;
#if FLINT_MLX_MATH
        #pragma unroll
        for (uint j = 0; j < 16; j += 4) {
            float value0 = x[segment * 16 + j], value1 = x[segment * 16 + j + 1];
            float value2 = x[segment * 16 + j + 2], value3 = x[segment * 16 + j + 3];
            input_sum += value0 + value1 + value2 + value3;
            values[j] = value0;
            values[j + 1] = value1 / 16.0f;
            values[j + 2] = value2 / 256.0f;
            values[j + 3] = value3 / 4096.0f;
        }
#else
        #pragma unroll
        for (uint j = 0; j < 16; ++j) {
            float value = x[segment * 16 + j];
            input_sum += value;
            values[j] = value * (1.0f / float(1u << ((j & 3u) * 4)));
        }
#endif
        #pragma unroll
        for (uint r = 0; r < Rows; ++r) {
            uint row = first + r;
            if (row >= m.rows) continue;
            ulong base = ulong(row) * row_words + segment * 2;
            uint codes0 = packed[base], codes1 = packed[base + 1];
            uint group = row * row_groups + segment / 4;
            float scale = weight(m.scales, 0, m.scale_dtype, group);
            float bias = weight(m.biases, 0, m.bias_dtype, group);
            float dot = 0;
#if FLINT_MLX_MATH
            #pragma unroll
            for (uint j = 0; j < 16; j += 4) {
                uint codes = j < 8 ? codes0 : codes1;
                uint half_word = (j & 7u) < 4 ? codes : codes >> 16;
                dot += (values[j] * float(half_word & 0x000fu) +
                        values[j + 1] * float(half_word & 0x00f0u) +
                        values[j + 2] * float(half_word & 0x0f00u) +
                        values[j + 3] * float(half_word & 0xf000u));
            }
            sums[r] += scale * dot + input_sum * bias;
#else
            #pragma unroll
            for (uint j = 0; j < 16; ++j) {
                uint codes = j < 8 ? codes0 : codes1;
                uint half_word = (j & 7u) < 4 ? codes : codes >> 16;
                dot += float(half_word & (15u << ((j & 3u) * 4))) * values[j];
            }
            sums[r] += scale * dot + bias * input_sum;
#endif
        }
    }
}


template<uint Rows>
inline void matrix_dot_rows_fixed(Matrix m, device const float *x,
                                  uint first, uint lane, thread float *sums) {
    if (m.reserved == 1) {
        switch (m.cols) {
            case 512: matrix_dot_rows_width<Rows, 512>(m, x, first, lane, sums); return;
            case 1024: matrix_dot_rows_width<Rows, 1024>(m, x, first, lane, sums); return;
            case 2048: matrix_dot_rows_width<Rows, 2048>(m, x, first, lane, sums); return;
            case 4096: matrix_dot_rows_width<Rows, 4096>(m, x, first, lane, sums); return;
            case 6144: matrix_dot_rows_width<Rows, 6144>(m, x, first, lane, sums); return;
        }
    }
    matrix_dot_rows<Rows>(m, x, first, lane, sums);
}

template<uint Rows, uint K>
inline void matrix_gate_up_rows_width(Matrix gate, Matrix up, device const float *x,
                                uint first, uint lane, thread float *gates, thread float *ups) {
    if (gate.reserved != 1 || up.reserved != 1 || gate.cols != up.cols || gate.rows != up.rows) {
        matrix_dot_rows<Rows>(gate, x, first, lane, gates);
        matrix_dot_rows<Rows>(up, x, first, lane, ups);
        return;
    }
    #pragma unroll
    for (uint r = 0; r < Rows; ++r) { gates[r] = 0; ups[r] = 0; }
    static_assert(K == 512 || K == 1024 || K == 2048 || K == 4096 || K == 6144, "Unsupported fixed width");
    constexpr uint row_words = K / 8, row_groups = K / 64;
    device const uint *gw = (device const uint *)gate.w;
    device const uint *uw = (device const uint *)up.w;
    // Every lane visits precisely the same ordered segments as the generic loop.
    #pragma unroll
    for (uint step = 0; step < K / 512; ++step) {
        uint segment = lane + step * 32;
        float values[16], input_sum = 0;
#if FLINT_MLX_MATH
        #pragma unroll
        for (uint j = 0; j < 16; j += 4) {
            float value0 = x[segment * 16 + j], value1 = x[segment * 16 + j + 1];
            float value2 = x[segment * 16 + j + 2], value3 = x[segment * 16 + j + 3];
            input_sum += value0 + value1 + value2 + value3;
            values[j] = value0;
            values[j + 1] = value1 / 16.0f;
            values[j + 2] = value2 / 256.0f;
            values[j + 3] = value3 / 4096.0f;
        }
#else
        #pragma unroll
        for (uint j = 0; j < 16; ++j) {
            float value = x[segment * 16 + j];
            input_sum += value;
            values[j] = value * (1.0f / float(1u << ((j & 3u) * 4)));
        }
#endif
        #pragma unroll
        for (uint r = 0; r < Rows; ++r) {
            uint row = first + r;
            if (row >= gate.rows) continue;
            ulong base = ulong(row) * row_words + segment * 2;
            uint g0 = gw[base], g1 = gw[base + 1], u0 = uw[base], u1 = uw[base + 1];
            uint group = row * row_groups + segment / 4;
            float gd = 0, ud = 0;
#if FLINT_MLX_MATH
            #pragma unroll
            for (uint j = 0; j < 16; j += 4) {
                uint gc = j < 8 ? g0 : g1, uc = j < 8 ? u0 : u1;
                uint gh = (j & 7u) < 4 ? gc : gc >> 16;
                uint uh = (j & 7u) < 4 ? uc : uc >> 16;
                gd += (values[j] * float(gh & 0x000fu) +
                       values[j + 1] * float(gh & 0x00f0u) +
                       values[j + 2] * float(gh & 0x0f00u) +
                       values[j + 3] * float(gh & 0xf000u));
                ud += (values[j] * float(uh & 0x000fu) +
                       values[j + 1] * float(uh & 0x00f0u) +
                       values[j + 2] * float(uh & 0x0f00u) +
                       values[j + 3] * float(uh & 0xf000u));
            }
            gates[r] += weight(gate.scales, 0, gate.scale_dtype, group) * gd +
                        input_sum * weight(gate.biases, 0, gate.bias_dtype, group);
            ups[r] += weight(up.scales, 0, up.scale_dtype, group) * ud +
                      input_sum * weight(up.biases, 0, up.bias_dtype, group);
#else
            #pragma unroll
            for (uint j = 0; j < 16; ++j) {
                uint gc = j < 8 ? g0 : g1, uc = j < 8 ? u0 : u1;
                uint gh = (j & 7u) < 4 ? gc : gc >> 16;
                uint uh = (j & 7u) < 4 ? uc : uc >> 16;
                uint mask = 15u << ((j & 3u) * 4);
                gd += float(gh & mask) * values[j];
                ud += float(uh & mask) * values[j];
            }
            gates[r] += weight(gate.scales, 0, gate.scale_dtype, group) * gd +
                        weight(gate.biases, 0, gate.bias_dtype, group) * input_sum;
            ups[r] += weight(up.scales, 0, up.scale_dtype, group) * ud +
                      weight(up.biases, 0, up.bias_dtype, group) * input_sum;
#endif
        }
    }
}

template<uint Rows>
inline void matrix_gate_up_rows_fixed(Matrix gate, Matrix up, device const float *x,
                                      uint first, uint lane, thread float *gates, thread float *ups) {
    if (gate.reserved == 1 && up.reserved == 1 && gate.cols == up.cols && gate.rows == up.rows) {
        switch (gate.cols) {
            case 512: matrix_gate_up_rows_width<Rows, 512>(gate, up, x, first, lane, gates, ups); return;
            case 1024: matrix_gate_up_rows_width<Rows, 1024>(gate, up, x, first, lane, gates, ups); return;
            case 2048: matrix_gate_up_rows_width<Rows, 2048>(gate, up, x, first, lane, gates, ups); return;
            case 4096: matrix_gate_up_rows_width<Rows, 4096>(gate, up, x, first, lane, gates, ups); return;
            case 6144: matrix_gate_up_rows_width<Rows, 6144>(gate, up, x, first, lane, gates, ups); return;
        }
    }
    matrix_gate_up_rows<Rows>(gate, up, x, first, lane, gates, ups);
}

template<uint Rows>
inline void staged_qmatvec_fixed(device const uchar *w, device const uchar *scales,
                            device const uchar *biases, device const float *x,
                            device float *out, constant P &p, uint i, uint lane) {
    uint first = (i / 32) * Rows;
    if (first >= p.m) return;
    Matrix m = { w + p.offset, scales + p.scale_offset, biases + p.bias_offset,
                 p.m, p.n, p.dtype, 4, 64, p.scale_dtype, p.bias_dtype, 1 };
    float sums[Rows];
    matrix_dot_rows_fixed<Rows>(m, x, first, lane, sums);
    #pragma unroll
    for (uint r = 0; r < Rows; ++r) {
        float sum = simd_sum(sums[r]);
        if (lane == 0 && first + r < p.m) out[first + r] = sum;
    }
}
// Launch fusion only: each projection keeps its own tiled row space and
// accumulation. The generated helpers certify which interval owns each tile.
template<uint Rows>
inline void staged_qkv_fixed(device const Matrix *matrices, device const float *x,
                       device float *q, device float *k, device float *v,
                       uint i, uint lane) {
    uint qt = (matrices[0].rows + Rows - 1) / Rows;
    uint kt = (matrices[1].rows + Rows - 1) / Rows;
    uint tile = i / 32;
    uint projection = staged_qkv_projection(tile, qt, kt);
    uint first = staged_qkv_local_tile(tile, qt, kt) * Rows;
    device const Matrix &m = matrices[projection];
    if (first >= m.rows) return;
    float sums[Rows];
    matrix_dot_rows_fixed<Rows>(m, x, first, lane, sums);
    device float *out = projection == 0 ? q : (projection == 1 ? k : v);
    #pragma unroll
    for (uint r = 0; r < Rows; ++r) {
        float sum = simd_sum(sums[r]);
        if (lane == 0 && first + r < m.rows) out[first + r] = sum;
    }
}
template<uint Rows>
inline void staged_gate_up_fixed(device const MLP *experts, device const float *x,
                           device const uint *ids, device float *activation,
                           constant P &p, uint i, uint lane) {
    uint tiles_per_route = (p.k + Rows - 1) / Rows;
    uint tile = i / 32, route = tile / tiles_per_route;
    if (route > p.n) return;
    uint first = (tile % tiles_per_route) * Rows;
    uint expert = route < p.n ? ids[route] : p.m;
    device const MLP &mlp = experts[expert];
    if (first >= mlp.gate.rows) return;
    float gates[Rows], ups[Rows];
    matrix_gate_up_rows_fixed<Rows>(mlp.gate, mlp.up, x, first, lane, gates, ups);
    #pragma unroll
    for (uint r = 0; r < Rows; ++r) {
        float gate = simd_sum(gates[r]), up = simd_sum(ups[r]);
        if (lane == 0 && first + r < mlp.gate.rows)
            activation[route * p.k + first + r] = swiglu_product(gate, up);
    }
}
template<uint Rows>
inline void staged_down_fixed(device const MLP *experts, device const float *activation,
                        device const uint *ids, device const float *weights,
                        device float *residual, constant P &p, uint i, uint lane) {
    uint first = (i / 32) * Rows;
    if (first >= p.dim) return;
    float outputs[Rows];
    #pragma unroll
    for (uint r = 0; r < Rows; ++r) {
#if defined(FLINT_MLX_MATH) && FLINT_MLX_MATH
        outputs[r] = 0;
#else
        outputs[r] = lane == 0 && first + r < p.dim ? residual[first + r] : 0;
#endif
    }
    for (uint route = 0; route <= p.n; ++route) {
        uint expert = route < p.n ? ids[route] : p.m;
        float sums[Rows];
        matrix_dot_rows_fixed<Rows>(experts[expert].down, activation + route * p.k, first, lane, sums);
        #pragma unroll
        for (uint r = 0; r < Rows; ++r) {
            float sum = simd_sum(sums[r]);
            if (lane == 0) outputs[r] += mixture_product(sum, route < p.n ? weights[route] : 1.0f);
        }
    }
    #pragma unroll
    for (uint r = 0; r < Rows; ++r)
        if (lane == 0 && first + r < p.dim) {
#if defined(FLINT_MLX_MATH) && FLINT_MLX_MATH
            residual[first + r] = residual[first + r] + outputs[r];
#else
            residual[first + r] = outputs[r];
#endif
        }
}
#define STAGED_FIXED_TILE_ENTRY(Rows) \
kernel void staged_qmatvec_fixed_t##Rows(device const uchar *w [[buffer(0)]], device const uchar *scales [[buffer(1)]], \
    device const uchar *biases [[buffer(2)]], device const float *x [[buffer(3)]], device float *out [[buffer(4)]], \
    constant P &p [[buffer(8)]], uint i [[thread_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) { \
    staged_qmatvec_fixed<Rows>(w, scales, biases, x, out, p, i, lane); \
} \
kernel void staged_qkv_fixed_t##Rows(device const Matrix *matrices [[buffer(0)]], device const float *x [[buffer(1)]], \
    device float *q [[buffer(2)]], device float *k [[buffer(3)]], device float *v [[buffer(4)]], \
    uint i [[thread_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) { \
    staged_qkv_fixed<Rows>(matrices, x, q, k, v, i, lane); \
} \
kernel void staged_gate_up_fixed_t##Rows(device const MLP *experts [[buffer(0)]], device const float *x [[buffer(1)]], \
    device const uint *ids [[buffer(2)]], device float *activation [[buffer(3)]], constant P &p [[buffer(8)]], \
    uint i [[thread_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) { \
    staged_gate_up_fixed<Rows>(experts, x, ids, activation, p, i, lane); \
} \
kernel void staged_down_fixed_t##Rows(device const MLP *experts [[buffer(0)]], device const float *activation [[buffer(1)]], \
    device const uint *ids [[buffer(2)]], device const float *weights [[buffer(3)]], device float *residual [[buffer(4)]], \
    constant P &p [[buffer(8)]], uint i [[thread_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) { \
    staged_down_fixed<Rows>(experts, activation, ids, weights, residual, p, i, lane); \
}
STAGED_FIXED_TILE_ENTRY(1)
STAGED_FIXED_TILE_ENTRY(2)
STAGED_FIXED_TILE_ENTRY(4)
STAGED_FIXED_TILE_ENTRY(8)
#undef STAGED_FIXED_TILE_ENTRY
