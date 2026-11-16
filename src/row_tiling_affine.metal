// Append after row_tiling.metal. BF16 affine-load candidate only: the
// reserved == 1 INT4/g64 descriptor contract and all relevant affine dtypes
// must match before specializing weight's dtype argument to literal 2.
// Other descriptors call the original helpers, including the fused gate/up
// helper for mismatched affine dtypes or row/column geometry.
// Keep dynamic K segment loops, FP32 expressions, input_sum, lane reductions,
// sigmoid/mixture/residual order, launch geometry and certified QKV helpers.

template<uint Rows>
inline void matrix_dot_rows_affine(Matrix m, device const float *x,
                            uint first, uint lane, thread float *sums) {
    static_assert(Rows == RESIDENT_ROW_TILE_1 || Rows == RESIDENT_ROW_TILE_2 ||
                  Rows == RESIDENT_ROW_TILE_4 || Rows == RESIDENT_ROW_TILE_8, "Uncertified row tile");
    if (m.reserved != 1 || m.scale_dtype != 2 || m.bias_dtype != 2) {
        matrix_dot_rows<Rows>(m, x, first, lane, sums);
        return;
    }
    #pragma unroll
    for (uint r = 0; r < Rows; ++r) sums[r] = 0;
    uint row_words = m.cols / 8, row_groups = m.cols / 64;
    device const uint *packed = (device const uint *)m.w;
    for (uint segment = lane; segment < m.cols / 16; segment += 32) {
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
            float scale = weight(m.scales, 0, 2, group);
            float bias = weight(m.biases, 0, 2, group);
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
inline void staged_qmatvec_affine(device const uchar *w, device const uchar *scales,
                            device const uchar *biases, device const float *x,
                            device float *out, constant P &p, uint i, uint lane) {
    uint first = (i / 32) * Rows;
    if (first >= p.m) return;
    Matrix m = { w + p.offset, scales + p.scale_offset, biases + p.bias_offset,
                 p.m, p.n, p.dtype, 4, 64, p.scale_dtype, p.bias_dtype, 1 };
    float sums[Rows];
    matrix_dot_rows_affine<Rows>(m, x, first, lane, sums);
    #pragma unroll
    for (uint r = 0; r < Rows; ++r) {
        float sum = simd_sum(sums[r]);
        if (lane == 0 && first + r < p.m) out[first + r] = sum;
    }
}
// Launch fusion only: each projection keeps its own tiled row space and
// accumulation. The generated helpers certify which interval owns each tile.
template<uint Rows>
inline void staged_qkv_affine(device const Matrix *matrices, device const float *x,
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
    matrix_dot_rows_affine<Rows>(m, x, first, lane, sums);
    device float *out = projection == 0 ? q : (projection == 1 ? k : v);
    #pragma unroll
    for (uint r = 0; r < Rows; ++r) {
        float sum = simd_sum(sums[r]);
        if (lane == 0 && first + r < m.rows) out[first + r] = sum;
    }
}
// Gate and up share each activation segment, while each dot keeps the same
// per-segment accumulation order as matrix_dot_rows_affine.
template<uint Rows>
inline void matrix_gate_up_rows_affine(Matrix gate, Matrix up, device const float *x,
                                uint first, uint lane, thread float *gates, thread float *ups) {
    if (gate.reserved != 1 || up.reserved != 1 || gate.cols != up.cols || gate.rows != up.rows ||
        gate.scale_dtype != 2 || gate.bias_dtype != 2 || up.scale_dtype != 2 || up.bias_dtype != 2) {
        matrix_gate_up_rows<Rows>(gate, up, x, first, lane, gates, ups);
        return;
    }
    #pragma unroll
    for (uint r = 0; r < Rows; ++r) { gates[r] = 0; ups[r] = 0; }
    uint row_words = gate.cols / 8, row_groups = gate.cols / 64;
    device const uint *gw = (device const uint *)gate.w;
    device const uint *uw = (device const uint *)up.w;
    for (uint segment = lane; segment < gate.cols / 16; segment += 32) {
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
            gates[r] += weight(gate.scales, 0, 2, group) * gd +
                        input_sum * weight(gate.biases, 0, 2, group);
            ups[r] += weight(up.scales, 0, 2, group) * ud +
                      input_sum * weight(up.biases, 0, 2, group);
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
            gates[r] += weight(gate.scales, 0, 2, group) * gd +
                        weight(gate.biases, 0, 2, group) * input_sum;
            ups[r] += weight(up.scales, 0, 2, group) * ud +
                      weight(up.biases, 0, 2, group) * input_sum;
#endif
        }
    }
}
template<uint Rows>
inline void staged_gate_up_affine(device const MLP *experts, device const float *x,
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
    matrix_gate_up_rows_affine<Rows>(mlp.gate, mlp.up, x, first, lane, gates, ups);
    #pragma unroll
    for (uint r = 0; r < Rows; ++r) {
        float gate = simd_sum(gates[r]), up = simd_sum(ups[r]);
        if (lane == 0 && first + r < mlp.gate.rows)
            activation[route * p.k + first + r] = swiglu_product(gate, up);
    }
}
template<uint Rows>
inline void staged_down_affine(device const MLP *experts, device const float *activation,
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
        matrix_dot_rows_affine<Rows>(experts[expert].down, activation + route * p.k, first, lane, sums);
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
#define STAGED_AFFINE_TILE_ENTRY(Rows) \
kernel void staged_qmatvec_affine_t##Rows(device const uchar *w [[buffer(0)]], device const uchar *scales [[buffer(1)]], \
    device const uchar *biases [[buffer(2)]], device const float *x [[buffer(3)]], device float *out [[buffer(4)]], \
    constant P &p [[buffer(8)]], uint i [[thread_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) { \
    staged_qmatvec_affine<Rows>(w, scales, biases, x, out, p, i, lane); \
} \
kernel void staged_qkv_affine_t##Rows(device const Matrix *matrices [[buffer(0)]], device const float *x [[buffer(1)]], \
    device float *q [[buffer(2)]], device float *k [[buffer(3)]], device float *v [[buffer(4)]], \
    uint i [[thread_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) { \
    staged_qkv_affine<Rows>(matrices, x, q, k, v, i, lane); \
} \
kernel void staged_gate_up_affine_t##Rows(device const MLP *experts [[buffer(0)]], device const float *x [[buffer(1)]], \
    device const uint *ids [[buffer(2)]], device float *activation [[buffer(3)]], constant P &p [[buffer(8)]], \
    uint i [[thread_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) { \
    staged_gate_up_affine<Rows>(experts, x, ids, activation, p, i, lane); \
} \
kernel void staged_down_affine_t##Rows(device const MLP *experts [[buffer(0)]], device const float *activation [[buffer(1)]], \
    device const uint *ids [[buffer(2)]], device const float *weights [[buffer(3)]], device float *residual [[buffer(4)]], \
    constant P &p [[buffer(8)]], uint i [[thread_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) { \
    staged_down_affine<Rows>(experts, activation, ids, weights, residual, p, i, lane); \
}
STAGED_AFFINE_TILE_ENTRY(1)
STAGED_AFFINE_TILE_ENTRY(2)
STAGED_AFFINE_TILE_ENTRY(4)
STAGED_AFFINE_TILE_ENTRY(8)
#undef STAGED_AFFINE_TILE_ENTRY
