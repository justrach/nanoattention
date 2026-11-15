// MLX-style multi-row activation reuse. The supported row tiles and integer
// ownership/index transformations are certified in the Lean binding. The
// compiler and FP32 evaluation remain part of the measured/tested boundary.
// INT4/g64 uses MLX-style 16-input lane segments and masked nibble products.
// Lane partition/reduction association changes require numerical validation.
// Independent output rows share inputs; no barrier or spin-limit weakening.

template<uint Rows>
inline void matrix_dot_rows(Matrix m, device const float *x,
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
    uint row_words = m.cols / 8, row_groups = m.cols / 64;
    device const uint *packed = (device const uint *)m.w;
    for (uint segment = lane; segment < m.cols / 16; segment += 32) {
        float values[16], input_sum = 0;
        #pragma unroll
        for (uint j = 0; j < 16; ++j) {
            float value = x[segment * 16 + j];
            input_sum += value;
            values[j] = value * (1.0f / float(1u << ((j & 3u) * 4)));
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
            float dot = 0;
            #pragma unroll
            for (uint j = 0; j < 16; ++j) {
                uint codes = j < 8 ? codes0 : codes1;
                uint half_word = (j & 7u) < 4 ? codes : codes >> 16;
                dot += float(half_word & (15u << ((j & 3u) * 4))) * values[j];
            }
            sums[r] += scale * dot + bias * input_sum;
        }
    }
}

template<uint Rows>
inline void row_tile_matvec(device const Matrix &m, device const float *x, device float *out,
                            uint simd, uint lane, uint simds) {
    for (uint first = simd * Rows; first < m.rows; first += simds * Rows) {
        float sums[Rows];
        matrix_dot_rows<Rows>(m, x, first, lane, sums);
        #pragma unroll
        for (uint r = 0; r < Rows; ++r) {
            float sum = simd_sum(sums[r]);
            if (lane == 0 && first + r < m.rows) out[first + r] = sum;
        }
    }
}

// A small direct probe also covers non-multiple row tails and indirect resources.
kernel void row_tile_probe(device const Matrix &m [[buffer(0)]], device const float *x [[buffer(1)]],
                           device float *out [[buffer(2)]], constant P &p [[buffer(8)]],
                           uint i [[thread_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
    uint simd = i / 32;
    uint simds = (m.rows + p.n - 1) / p.n;
    switch (p.n) {
        case 1: row_tile_matvec<1>(m, x, out, simd, lane, simds); break;
        case 2: row_tile_matvec<2>(m, x, out, simd, lane, simds); break;
        case 4: row_tile_matvec<4>(m, x, out, simd, lane, simds); break;
        case 8: row_tile_matvec<8>(m, x, out, simd, lane, simds); break;
    }
}

// Staged dispatches retain global dependency boundaries while reusing inputs
// across certified independent rows. Unlike the resident path, the grid can
// contain many more workgroups than can be simultaneously resident.
template<uint Rows>
inline void staged_qmatvec(device const uchar *w, device const uchar *scales,
                            device const uchar *biases, device const float *x,
                            device float *out, constant P &p, uint i, uint lane) {
    uint first = (i / 32) * Rows;
    if (first >= p.m) return;
    Matrix m = { w + p.offset, scales + p.scale_offset, biases + p.bias_offset,
                 p.m, p.n, p.dtype, 4, 64, p.scale_dtype, p.bias_dtype, 1 };
    float sums[Rows];
    matrix_dot_rows<Rows>(m, x, first, lane, sums);
    #pragma unroll
    for (uint r = 0; r < Rows; ++r) {
        float sum = simd_sum(sums[r]);
        if (lane == 0 && first + r < p.m) out[first + r] = sum;
    }
}
// Launch fusion only: each projection keeps its own tiled row space and
// accumulation. The generated helpers certify which interval owns each tile.
template<uint Rows>
inline void staged_qkv(device const Matrix *matrices, device const float *x,
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
    matrix_dot_rows<Rows>(m, x, first, lane, sums);
    device float *out = projection == 0 ? q : (projection == 1 ? k : v);
    #pragma unroll
    for (uint r = 0; r < Rows; ++r) {
        float sum = simd_sum(sums[r]);
        if (lane == 0 && first + r < m.rows) out[first + r] = sum;
    }
}
// Gate and up share each activation segment, while each dot keeps the same
// per-segment accumulation order as matrix_dot_rows.
template<uint Rows>
inline void matrix_gate_up_rows(Matrix gate, Matrix up, device const float *x,
                                uint first, uint lane, thread float *gates, thread float *ups) {
    if (gate.reserved != 1 || up.reserved != 1 || gate.cols != up.cols || gate.rows != up.rows) {
        matrix_dot_rows<Rows>(gate, x, first, lane, gates);
        matrix_dot_rows<Rows>(up, x, first, lane, ups);
        return;
    }
    #pragma unroll
    for (uint r = 0; r < Rows; ++r) { gates[r] = 0; ups[r] = 0; }
    uint row_words = gate.cols / 8, row_groups = gate.cols / 64;
    device const uint *gw = (device const uint *)gate.w;
    device const uint *uw = (device const uint *)up.w;
    for (uint segment = lane; segment < gate.cols / 16; segment += 32) {
        float values[16], input_sum = 0;
        #pragma unroll
        for (uint j = 0; j < 16; ++j) {
            float value = x[segment * 16 + j];
            input_sum += value;
            values[j] = value * (1.0f / float(1u << ((j & 3u) * 4)));
        }
        #pragma unroll
        for (uint r = 0; r < Rows; ++r) {
            uint row = first + r;
            if (row >= gate.rows) continue;
            ulong base = ulong(row) * row_words + segment * 2;
            uint g0 = gw[base], g1 = gw[base + 1], u0 = uw[base], u1 = uw[base + 1];
            uint group = row * row_groups + segment / 4;
            float gd = 0, ud = 0;
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
        }
    }
}
template<uint Rows>
inline void staged_gate_up(device const MLP *experts, device const float *x,
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
    matrix_gate_up_rows<Rows>(mlp.gate, mlp.up, x, first, lane, gates, ups);
    #pragma unroll
    for (uint r = 0; r < Rows; ++r) {
        float gate = simd_sum(gates[r]), up = simd_sum(ups[r]);
        if (lane == 0 && first + r < mlp.gate.rows)
            activation[route * p.k + first + r] = (gate / (1.0f + exp(-gate))) * up;
    }
}
template<uint Rows>
inline void staged_down(device const MLP *experts, device const float *activation,
                        device const uint *ids, device const float *weights,
                        device float *residual, constant P &p, uint i, uint lane) {
    uint first = (i / 32) * Rows;
    if (first >= p.dim) return;
    float outputs[Rows];
    #pragma unroll
    for (uint r = 0; r < Rows; ++r) outputs[r] = lane == 0 && first + r < p.dim ? residual[first + r] : 0;
    for (uint route = 0; route <= p.n; ++route) {
        uint expert = route < p.n ? ids[route] : p.m;
        float sums[Rows];
        matrix_dot_rows<Rows>(experts[expert].down, activation + route * p.k, first, lane, sums);
        #pragma unroll
        for (uint r = 0; r < Rows; ++r) {
            float sum = simd_sum(sums[r]);
            if (lane == 0) outputs[r] += sum * (route < p.n ? weights[route] : 1.0f);
        }
    }
    #pragma unroll
    for (uint r = 0; r < Rows; ++r)
        if (lane == 0 && first + r < p.dim) residual[first + r] = outputs[r];
}
#define STAGED_TILE_ENTRY(Rows) \
kernel void staged_qmatvec_t##Rows(device const uchar *w [[buffer(0)]], device const uchar *scales [[buffer(1)]], \
    device const uchar *biases [[buffer(2)]], device const float *x [[buffer(3)]], device float *out [[buffer(4)]], \
    constant P &p [[buffer(8)]], uint i [[thread_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) { \
    staged_qmatvec<Rows>(w, scales, biases, x, out, p, i, lane); \
} \
kernel void staged_qkv_t##Rows(device const Matrix *matrices [[buffer(0)]], device const float *x [[buffer(1)]], \
    device float *q [[buffer(2)]], device float *k [[buffer(3)]], device float *v [[buffer(4)]], \
    uint i [[thread_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) { \
    staged_qkv<Rows>(matrices, x, q, k, v, i, lane); \
} \
kernel void staged_gate_up_t##Rows(device const MLP *experts [[buffer(0)]], device const float *x [[buffer(1)]], \
    device const uint *ids [[buffer(2)]], device float *activation [[buffer(3)]], constant P &p [[buffer(8)]], \
    uint i [[thread_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) { \
    staged_gate_up<Rows>(experts, x, ids, activation, p, i, lane); \
} \
kernel void staged_down_t##Rows(device const MLP *experts [[buffer(0)]], device const float *activation [[buffer(1)]], \
    device const uint *ids [[buffer(2)]], device const float *weights [[buffer(3)]], device float *residual [[buffer(4)]], \
    constant P &p [[buffer(8)]], uint i [[thread_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) { \
    staged_down<Rows>(experts, activation, ids, weights, residual, p, i, lane); \
}
STAGED_TILE_ENTRY(1)
STAGED_TILE_ENTRY(2)
STAGED_TILE_ENTRY(4)
STAGED_TILE_ENTRY(8)
#undef STAGED_TILE_ENTRY
