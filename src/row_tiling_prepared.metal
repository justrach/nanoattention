// Opt-in prepared-input candidate. Append once after row_tiling_affine.metal.
// Not an admission certificate: exact output-bit checks are required before
// enabling this path. Only reserved==1 INT4/g64 descriptors are supported.
// Each raw N-vector becomes [N scaled floats][N/16 segment sums], stride
// N+N/16. Qmatvec/QKV/gate-up consume one vector; down consumes p.n+1 vectors
// with N=p.k (possibly larger than a selected down.cols). Gate/up OUTPUT
// remains raw, so prepare it before prepared down. Only real down.cols segments
// are read; stale activation padding cannot contribute to valid output rows.
// Descriptor/geometry rejection never falls back to a raw-input helper.

inline bool prepared_matrix_valid(Matrix m) {
    return m.reserved == 1 && (m.dtype == 3 || m.dtype == 4) && m.bits == 4 && m.group_size == 64 &&
           m.rows != 0 && m.cols != 0 && m.cols % 64 == 0 &&
           m.scale_dtype <= 2 && m.bias_dtype <= 2;
}
inline float prepared_affine_weight(device const uchar *bytes, uint dtype, uint group) {
    if (dtype == 2) return weight(bytes, 0, 2, group);
    return weight(bytes, 0, dtype, group);
}

// p.n=N (positive multiple of 64), p.m=number of vectors. One grid thread per
// 16-input segment, flattened over vectors. Input/output must not overlap.
kernel void prepare_quant_input(device const float *raw [[buffer(0)]],
                                device float *prepared [[buffer(1)]],
                                constant P &p [[buffer(8)]], uint i [[thread_position_in_grid]]) {
    if (p.n == 0 || p.n % 64 != 0 || p.m == 0) return;
    uint segments = p.n / 16;
    if (ulong(i) >= ulong(p.m) * segments) return;
    uint vector = i / segments, segment = i % segments;
    device const float *x = raw + ulong(vector) * p.n;
    device float *out = prepared + ulong(vector) * (ulong(p.n) + segments);
    float input_sum = 0;
#if FLINT_MLX_MATH
    #pragma unroll
    for (uint j = 0; j < 16; j += 4) {
        float value0 = x[segment * 16 + j], value1 = x[segment * 16 + j + 1];
        float value2 = x[segment * 16 + j + 2], value3 = x[segment * 16 + j + 3];
        input_sum += value0 + value1 + value2 + value3;
        out[segment * 16 + j] = value0;
        out[segment * 16 + j + 1] = value1 / 16.0f;
        out[segment * 16 + j + 2] = value2 / 256.0f;
        out[segment * 16 + j + 3] = value3 / 4096.0f;
    }
#else
    #pragma unroll
    for (uint j = 0; j < 16; ++j) {
        float value = x[segment * 16 + j];
        input_sum += value;
        out[segment * 16 + j] = value * (1.0f / float(1u << ((j & 3u) * 4)));
    }
#endif
    out[p.n + segment] = input_sum;
}

template<uint Rows>
inline void matrix_dot_rows_prepared(Matrix m, device const float *x, uint sum_offset,
                            uint first, uint lane, thread float *sums) {
    static_assert(Rows == RESIDENT_ROW_TILE_1 || Rows == RESIDENT_ROW_TILE_2 ||
                  Rows == RESIDENT_ROW_TILE_4 || Rows == RESIDENT_ROW_TILE_8, "Uncertified row tile");
    #pragma unroll
    for (uint r = 0; r < Rows; ++r) sums[r] = 0;
    if (!prepared_matrix_valid(m) || sum_offset < m.cols || sum_offset % 64 != 0) return;
    uint row_words = m.cols / 8, row_groups = m.cols / 64;
    device const uint *packed = (device const uint *)m.w;
    for (uint segment = lane; segment < m.cols / 16; segment += 32) {
        float values[16];
        float input_sum = x[sum_offset + segment];
        #pragma unroll
        for (uint j = 0; j < 16; ++j) values[j] = x[segment * 16 + j];
        #pragma unroll
        for (uint r = 0; r < Rows; ++r) {
            uint row = first + r;
            if (row >= m.rows) continue;
            ulong base = ulong(row) * row_words + segment * 2;
            uint codes0 = packed[base], codes1 = packed[base + 1];
            uint group = row * row_groups + segment / 4;
            float scale = prepared_affine_weight(m.scales, m.scale_dtype, group);
            float bias = prepared_affine_weight(m.biases, m.bias_dtype, group);
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
inline void staged_qmatvec_prepared(device const uchar *w, device const uchar *scales,
                            device const uchar *biases, device const float *x,
                            device float *out, constant P &p, uint i, uint lane) {
    uint first = (i / 32) * Rows;
    if (first >= p.m || p.n == 0 || p.n % 64 != 0 ||
        p.bits != 4 || p.k != 64 || (p.dtype != 3 && p.dtype != 4) ||
        p.scale_dtype > 2 || p.bias_dtype > 2) return;
    Matrix m = { w + p.offset, scales + p.scale_offset, biases + p.bias_offset,
                 p.m, p.n, p.dtype, 4, 64, p.scale_dtype, p.bias_dtype, 1 };
    float sums[Rows];
    matrix_dot_rows_prepared<Rows>(m, x, m.cols, first, lane, sums);
    #pragma unroll
    for (uint r = 0; r < Rows; ++r) {
        float sum = simd_sum(sums[r]);
        if (lane == 0 && first + r < p.m) out[first + r] = sum;
    }
}
// Launch fusion only: each projection keeps its own tiled row space and
// accumulation. The generated helpers certify which interval owns each tile.
template<uint Rows>
inline void staged_qkv_prepared(device const Matrix *matrices, device const float *x,
                       device float *q, device float *k, device float *v,
                       uint i, uint lane) {
    if (!prepared_matrix_valid(matrices[0]) || !prepared_matrix_valid(matrices[1]) ||
        !prepared_matrix_valid(matrices[2]) || matrices[0].cols != matrices[1].cols ||
        matrices[0].cols != matrices[2].cols) return;
    uint qt = (matrices[0].rows + Rows - 1) / Rows;
    uint kt = (matrices[1].rows + Rows - 1) / Rows;
    uint tile = i / 32;
    uint projection = staged_qkv_projection(tile, qt, kt);
    uint first = staged_qkv_local_tile(tile, qt, kt) * Rows;
    device const Matrix &m = matrices[projection];
    if (first >= m.rows) return;
    float sums[Rows];
    matrix_dot_rows_prepared<Rows>(m, x, m.cols, first, lane, sums);
    device float *out = projection == 0 ? q : (projection == 1 ? k : v);
    #pragma unroll
    for (uint r = 0; r < Rows; ++r) {
        float sum = simd_sum(sums[r]);
        if (lane == 0 && first + r < m.rows) out[first + r] = sum;
    }
}
// Gate and up share each activation segment, while each dot keeps the same
// per-segment accumulation order as matrix_dot_rows_prepared.
template<uint Rows>
inline void matrix_gate_up_rows_prepared(Matrix gate, Matrix up, device const float *x,
                                uint first, uint lane, thread float *gates, thread float *ups) {
    #pragma unroll
    for (uint r = 0; r < Rows; ++r) { gates[r] = 0; ups[r] = 0; }
    if (!prepared_matrix_valid(gate) || !prepared_matrix_valid(up) ||
        gate.cols != up.cols || gate.rows != up.rows) return;
    uint row_words = gate.cols / 8, row_groups = gate.cols / 64;
    device const uint *gw = (device const uint *)gate.w;
    device const uint *uw = (device const uint *)up.w;
    for (uint segment = lane; segment < gate.cols / 16; segment += 32) {
        float values[16];
        float input_sum = x[gate.cols + segment];
        #pragma unroll
        for (uint j = 0; j < 16; ++j) values[j] = x[segment * 16 + j];
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
            gates[r] += prepared_affine_weight(gate.scales, gate.scale_dtype, group) * gd +
                        input_sum * prepared_affine_weight(gate.biases, gate.bias_dtype, group);
            ups[r] += prepared_affine_weight(up.scales, up.scale_dtype, group) * ud +
                      input_sum * prepared_affine_weight(up.biases, up.bias_dtype, group);
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
            gates[r] += prepared_affine_weight(gate.scales, gate.scale_dtype, group) * gd +
                        prepared_affine_weight(gate.biases, gate.bias_dtype, group) * input_sum;
            ups[r] += prepared_affine_weight(up.scales, up.scale_dtype, group) * ud +
                      prepared_affine_weight(up.biases, up.bias_dtype, group) * input_sum;
#endif
        }
    }
}
template<uint Rows>
inline void staged_gate_up_prepared(device const MLP *experts, device const float *x,
                           device const uint *ids, device float *activation,
                           constant P &p, uint i, uint lane) {
    if (p.k == 0) return;
    uint tiles_per_route = (p.k + Rows - 1) / Rows;
    uint tile = i / 32, route = tile / tiles_per_route;
    if (route > p.n) return;
    uint first = (tile % tiles_per_route) * Rows;
    uint expert = route < p.n ? ids[route] : p.m;
    device const MLP &mlp = experts[expert];
    if (!prepared_matrix_valid(mlp.gate) || !prepared_matrix_valid(mlp.up) ||
        mlp.gate.cols != mlp.up.cols || mlp.gate.rows != mlp.up.rows ||
        mlp.gate.rows > p.k || first >= mlp.gate.rows) return;
    float gates[Rows], ups[Rows];
    matrix_gate_up_rows_prepared<Rows>(mlp.gate, mlp.up, x, first, lane, gates, ups);
    #pragma unroll
    for (uint r = 0; r < Rows; ++r) {
        float gate = simd_sum(gates[r]), up = simd_sum(ups[r]);
        if (lane == 0 && first + r < mlp.gate.rows)
            activation[route * p.k + first + r] = swiglu_product(gate, up);
    }
}
template<uint Rows>
inline void staged_down_prepared(device const MLP *experts, device const float *activation,
                        device const uint *ids, device const float *weights,
                        device float *residual, constant P &p, uint i, uint lane) {
    uint first = (i / 32) * Rows;
    if (first >= p.dim || p.k == 0 || p.k % 64 != 0) return;
    // Reject each route before its activation read. Residual writes happen only
    // after ALL routes succeed, so rejection cannot partially modify output.
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
        Matrix down = experts[expert].down;
        if (!prepared_matrix_valid(down) || down.cols > p.k || down.rows != p.dim) return;
        float sums[Rows];
        // Sum cache starts after the full route stride, not down.cols.
        matrix_dot_rows_prepared<Rows>(down,
            activation + ulong(route) * (ulong(p.k) + p.k / 16), p.k, first, lane, sums);
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
#define STAGED_PREPARED_TILE_ENTRY(Rows) \
kernel void staged_qmatvec_prepared_t##Rows(device const uchar *w [[buffer(0)]], device const uchar *scales [[buffer(1)]], \
    device const uchar *biases [[buffer(2)]], device const float *x [[buffer(3)]], device float *out [[buffer(4)]], \
    constant P &p [[buffer(8)]], uint i [[thread_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) { \
    staged_qmatvec_prepared<Rows>(w, scales, biases, x, out, p, i, lane); \
} \
kernel void staged_qkv_prepared_t##Rows(device const Matrix *matrices [[buffer(0)]], device const float *x [[buffer(1)]], \
    device float *q [[buffer(2)]], device float *k [[buffer(3)]], device float *v [[buffer(4)]], \
    uint i [[thread_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) { \
    staged_qkv_prepared<Rows>(matrices, x, q, k, v, i, lane); \
} \
kernel void staged_gate_up_prepared_t##Rows(device const MLP *experts [[buffer(0)]], device const float *x [[buffer(1)]], \
    device const uint *ids [[buffer(2)]], device float *activation [[buffer(3)]], constant P &p [[buffer(8)]], \
    uint i [[thread_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) { \
    staged_gate_up_prepared<Rows>(experts, x, ids, activation, p, i, lane); \
} \
kernel void staged_down_prepared_t##Rows(device const MLP *experts [[buffer(0)]], device const float *activation [[buffer(1)]], \
    device const uint *ids [[buffer(2)]], device const float *weights [[buffer(3)]], device float *residual [[buffer(4)]], \
    constant P &p [[buffer(8)]], uint i [[thread_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) { \
    staged_down_prepared<Rows>(experts, activation, ids, weights, residual, p, i, lane); \
}
STAGED_PREPARED_TILE_ENTRY(1)
STAGED_PREPARED_TILE_ENTRY(2)
STAGED_PREPARED_TILE_ENTRY(4)
STAGED_PREPARED_TILE_ENTRY(8)
#undef STAGED_PREPARED_TILE_ENTRY
