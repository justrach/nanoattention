// ---- c2-prefill-mlx (prefill, mlx-arith tier): stock MLX's qmm_t output bits, restructured ----
// NOT the reference arithmetic. Same contract as mk_mlx_qmm_t (mlx.metal): every
// output of stock MLX's unsplit affine_qmm_t (float32, transpose, 4-bit) is one chain
//   W[o][k] = s * code + b            (odd k: (s / 16) * (byte & 0xf0) + b, the same value)
//   C = +0; for k ascending: C = fma(x[t][k], W[o][k], C)
// because simdgroup_multiply_accumulate computes each element of an 8 x 8 x 8 step as
// acc = C; for k = 0..7: acc = fma(A[r][k], B[k][c], acc) (lean/proofs/Emu/Mlx.lean).
// That chain does not depend on the tile shape, on what is staged in threadgroup
// memory or on how blocks are loaded, so the kernels below are free to choose them;
// tools/nano_mlx_ops.py (--variants mlx2) checks the bits against stock MLX. Only
// unsplit products may use them (split K partitions are part of stock's bits).
// Tile: 64 tokens x 16 outputs per matrix per threadgroup of 2 SIMD groups. SIMD group
// g owns tokens 8 g + 16 i (i < 4) and outputs 8 j (j < 2) in 8 x 8 fragments. x
// fragments are read straight from device memory (float2 per lane: x offsets and row
// stride must be even); the weights of a BK-wide K block are dequantized once per
// threadgroup into one of two threadgroup buffers while the other is multiplied, and
// the next block's code bytes, scales and biases are prefetched into registers.
// One barrier per K block.
constant uint MK_MLX2_BM = 64, MK_MLX2_BN = 16;

// The K loop. C[mat][i][j]: matrix mat's fragment (tokens 8 g + 16 i, outputs 8 j).
// xr: x row 0 of the forward (arena + x + row0 * xs). wo/so/bo: byte offsets of each
// matrix's codes, scales and biases in wb/sb/bb. Requires K % BK == 0.
template <uint BK, uint NMAT, uint GROUP, uint SDT>
inline void mk_mlx2_mm(thread simdgroup_float8x8 (&C)[NMAT][4][2], device const float *xr, uint xs, uint M, uint K, uint N,
                       device const uchar *wb, device const uchar *sb, device const uchar *bb,
                       thread const ulong (&wo)[NMAT], thread const ulong (&so)[NMAT], thread const ulong (&bo)[NMAT],
                       uint y_row, uint y_col, uint lid, uint simd, uint lane, threadgroup float *Ws) {
    constexpr uint LDW = BK + 4;                        // MLX's padded tile row
    constexpr uint WTILE = MK_MLX2_BN * LDW;            // floats per matrix per buffer
    constexpr uint UNITS = NMAT * MK_MLX2_BN * BK / 8;  // 8-weight units per K block
    constexpr uint WU = UNITS / 64;
    static_assert(UNITS % 64 == 0, "every thread dequantizes the same number of units");
    constexpr uint esz = SDT == 0 ? 4 : 2;
    uint qid = lane / 4;
    uint sm = (qid & 4) + (lane / 2) % 4, sn = (qid & 2) * 2 + (lane % 2) * 2;
    MK_UNROLL
    for (uint m = 0; m < NMAT; ++m) for (uint i = 0; i < 4; ++i) for (uint j = 0; j < 2; ++j)
        C[m][i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    // A fragment i of this lane: x[row 8 g + 16 i + sm][k + sn, k + sn + 1]; rows past M read row M - 1
    // (they only feed outputs that are not stored).
    device const float *xa[4];
    MK_UNROLL
    for (uint i = 0; i < 4; ++i) xa[i] = xr + min(y_row + 8 * simd + 16 * i + sm, M - 1) * xs + sn;
    // Dequantization units: thread lid fills 8 consecutive inputs `col` of output row `row` of matrix `mat`.
    uint mat[WU], row[WU], col[WU];
    device const uchar *wp[WU], *sp[WU], *bp[WU];
    MK_UNROLL
    for (uint r = 0; r < WU; ++r) {
        uint c = lid + 64 * r, cc = c % (MK_MLX2_BN * BK / 8);
        mat[r] = c / (MK_MLX2_BN * BK / 8);
        row[r] = cc / (BK / 8);
        col[r] = (cc % (BK / 8)) * 8;
        uint o = min(y_col + row[r], N - 1); // rows past N only feed outputs that are not stored
        wp[r] = wb + wo[mat[r]] + ulong(o) * (K / 2) + col[r] / 2;
        sp[r] = sb + so[mat[r]] + ulong(o) * (K / GROUP) * esz;
        bp[r] = bb + bo[mat[r]] + ulong(o) * (K / GROUP) * esz;
    }
    uchar code[WU][4];
    float s[WU], b[WU];
    auto fetch = [&](uint k0) {
        MK_UNROLL
        for (uint r = 0; r < WU; ++r) {
            MK_UNROLL
            for (uint q = 0; q < 4; ++q) code[r][q] = wp[r][k0 / 2 + q];
            uint g = (k0 + col[r]) / GROUP;
            s[r] = mk_mlx_sb<SDT>(sp[r] + g * esz);
            b[r] = mk_mlx_sb<SDT>(bp[r] + g * esz);
        }
    };
    // MLX's dequantize<float, 2, 4>, written into the tile as MLX's loader does.
    auto store = [&](threadgroup float *wt) {
        MK_UNROLL
        for (uint r = 0; r < WU; ++r) {
            float v[8];
            MK_UNROLL
            for (uint q = 0; q < 4; ++q) {
                uint byte = code[r][q];
                v[2 * q] = s[r] * float(byte & 0x0f) + b[r];
                v[2 * q + 1] = (s[r] / 16.0f) * float(byte & 0xf0) + b[r];
            }
            threadgroup float *d = wt + mat[r] * WTILE + row[r] * LDW + col[r];
            ((threadgroup float4 *)d)[0] = float4(v[0], v[1], v[2], v[3]);
            ((threadgroup float4 *)d)[1] = float4(v[4], v[5], v[6], v[7]);
        }
    };
    fetch(0);
    store(Ws);
    if (BK < K) fetch(BK);
    uint buf = 0;
    for (uint k0 = 0; k0 < K; k0 += BK) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        threadgroup const float *wt = Ws + buf * NMAT * WTILE;
        MK_UNROLL
        for (uint kk = 0; kk < BK; kk += 8) {
            simdgroup_float8x8 A[4], B[NMAT][2];
            MK_UNROLL
            for (uint i = 0; i < 4; ++i) {
                float2 v = *(device const float2 *)(xa[i] + k0 + kk);
                A[i].thread_elements()[0] = v.x;
                A[i].thread_elements()[1] = v.y;
            }
            MK_UNROLL
            for (uint m = 0; m < NMAT; ++m) for (uint j = 0; j < 2; ++j) {
                threadgroup const float *p = wt + m * WTILE + (8 * j + sn) * LDW + kk + sm; // B[k][o] = W[o][k]
                B[m][j].thread_elements()[0] = p[0];
                B[m][j].thread_elements()[1] = p[LDW];
            }
            MK_UNROLL
            for (uint m = 0; m < NMAT; ++m) for (uint i = 0; i < 4; ++i) for (uint j = 0; j < 2; ++j)
                simdgroup_multiply_accumulate(C[m][i][j], A[i], B[m][j], C[m][i][j]);
        }
        // The other buffer was last read before this block's barrier: safe to refill.
        if (k0 + BK < K) {
            store(Ws + (1 - buf) * NMAT * WTILE);
            if (k0 + 2 * BK < K) fetch(k0 + 2 * BK);
        }
        buf = 1 - buf;
    }
}
// Store matrix 0's tile (a copy): rows < M, columns < N.
inline void mk_mlx2_store(thread simdgroup_float8x8 (&C)[4][2], device float *y, uint ys, uint M, uint N,
                          uint y_row, uint y_col, uint simd, uint lane) {
    uint qid = lane / 4;
    uint sm = (qid & 4) + (lane / 2) % 4, sn = (qid & 2) * 2 + (lane % 2) * 2;
    MK_UNROLL
    for (uint i = 0; i < 4; ++i) for (uint j = 0; j < 2; ++j) {
        uint t = y_row + 8 * simd + 16 * i + sm, o = y_col + 8 * j + sn;
        if (t >= M) continue;
        for (uint e = 0; e < 2; ++e) if (o + e < N) y[ulong(t) * ys + o + e] = C[i][j].thread_elements()[e];
    }
}
// One matrix: buffers and parameters as mk_mlx_qmm_t (MK: rows = M, n = K, m = N, x/xs,
// y/ys, w/s/b). Grid: ceil(N / 16) * ceil(M / 64) threadgroups of 64 threads.
#define MK_MLX2_QMM(NAME, GROUP, SDT) \
[[kernel]] void NAME(device const uchar *w [[buffer(0)]], device const uchar *s [[buffer(1)]], \
        device const uchar *b [[buffer(2)]], device float *arena [[buffer(3)]], \
        constant MK &p [[buffer(8)]], uint tgid [[threadgroup_position_in_grid]], \
        uint simd [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]], \
        uint lid [[thread_index_in_threadgroup]]) { \
    threadgroup float Ws[2 * MK_MLX2_BN * (64 + 4)]; \
    uint n_tiles = (p.m + MK_MLX2_BN - 1) / MK_MLX2_BN; \
    uint y_col = (tgid % n_tiles) * MK_MLX2_BN, y_row = (tgid / n_tiles) * MK_MLX2_BM; \
    const ulong wo[1] = {p.w}, so[1] = {p.s}, bo[1] = {p.b}; \
    simdgroup_float8x8 C[1][4][2]; \
    mk_mlx2_mm<64, 1, GROUP, SDT>(C, arena + p.x + ulong(p.row0) * p.xs, p.xs, p.rows, p.n, p.m, w, s, b, wo, so, bo, \
                                  y_row, y_col, lid, simd, lane, Ws); \
    mk_mlx2_store(C[0], arena + p.y + ulong(p.row0) * p.ys, p.ys, p.rows, p.m, y_row, y_col, simd, lane); \
}
// Up to 4 matrices of one input in one launch (q/k/v), tiles of each matrix in order
// (as mk_mlx_qmm_multi): MKX x.mat[0..count) (w/s/b in one buffer, m = N, y/ys); x.p: rows, row0, n = K, x/xs.
#define MK_MLX2_MULTI(NAME, GROUP, SDT) \
[[kernel]] void NAME(device const uchar *w [[buffer(0)]], device float *arena [[buffer(1)]], \
        constant MKX &x [[buffer(8)]], uint tgid [[threadgroup_position_in_grid]], \
        uint simd [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]], \
        uint lid [[thread_index_in_threadgroup]]) { \
    threadgroup float Ws[2 * MK_MLX2_BN * (64 + 4)]; \
    uint m_tiles = (x.p.rows + MK_MLX2_BM - 1) / MK_MLX2_BM, t = tgid, k = 0; \
    for (; k + 1 < x.count; ++k) { \
        uint nt = ((x.mat[k].m + MK_MLX2_BN - 1) / MK_MLX2_BN) * m_tiles; \
        if (t < nt) break; \
        t -= nt; \
    } \
    uint N = x.mat[k].m, n_tiles = (N + MK_MLX2_BN - 1) / MK_MLX2_BN; \
    uint y_col = (t % n_tiles) * MK_MLX2_BN, y_row = (t / n_tiles) * MK_MLX2_BM; \
    const ulong wo[1] = {x.mat[k].w}, so[1] = {x.mat[k].s}, bo[1] = {x.mat[k].b}; \
    simdgroup_float8x8 C[1][4][2]; \
    mk_mlx2_mm<64, 1, GROUP, SDT>(C, arena + x.p.x + ulong(x.p.row0) * x.p.xs, x.p.xs, x.p.rows, x.p.n, N, w, w, w, wo, so, bo, \
                                  y_row, y_col, lid, simd, lane, Ws); \
    mk_mlx2_store(C[0], arena + x.mat[k].y + ulong(x.p.row0) * x.mat[k].ys, x.mat[k].ys, x.p.rows, N, y_row, y_col, simd, lane); \
}
// GLU: gate (mat[0]) and up (mat[1]) products of one input, both of N outputs, over one
// x tile, and the product out = mk_act(gate, act_a) * mk_act(up, act_b) (kind 1; kind 0
// adds) written to x.p.y/ys: mk_binary's formula on the same two values, so the output
// equals mk_mlx_qmm_t twice then mk_binary bit for bit. BK 32 (two matrices' buffers).
#define MK_MLX2_GLU(NAME, GROUP, SDT) \
[[kernel]] void NAME(device const uchar *w [[buffer(0)]], device float *arena [[buffer(1)]], \
        constant MKX &x [[buffer(8)]], uint tgid [[threadgroup_position_in_grid]], \
        uint simd [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]], \
        uint lid [[thread_index_in_threadgroup]]) { \
    threadgroup float Ws[2 * 2 * MK_MLX2_BN * (32 + 4)]; \
    uint N = x.mat[0].m, M = x.p.rows, n_tiles = (N + MK_MLX2_BN - 1) / MK_MLX2_BN; \
    uint y_col = (tgid % n_tiles) * MK_MLX2_BN, y_row = (tgid / n_tiles) * MK_MLX2_BM; \
    const ulong wo[2] = {x.mat[0].w, x.mat[1].w}, so[2] = {x.mat[0].s, x.mat[1].s}, bo[2] = {x.mat[0].b, x.mat[1].b}; \
    simdgroup_float8x8 C[2][4][2]; \
    mk_mlx2_mm<32, 2, GROUP, SDT>(C, arena + x.p.x + ulong(x.p.row0) * x.p.xs, x.p.xs, M, x.p.n, N, w, w, w, wo, so, bo, \
                                  y_row, y_col, lid, simd, lane, Ws); \
    device float *y = arena + x.p.y + ulong(x.p.row0) * x.p.ys; \
    uint qid = lane / 4; \
    uint sm = (qid & 4) + (lane / 2) % 4, sn = (qid & 2) * 2 + (lane % 2) * 2; \
    for (uint i = 0; i < 4; ++i) for (uint j = 0; j < 2; ++j) { \
        uint t = y_row + 8 * simd + 16 * i + sm, o = y_col + 8 * j + sn; \
        if (t >= M) continue; \
        for (uint e = 0; e < 2; ++e) if (o + e < N) { \
            float a = mk_act(C[0][i][j].thread_elements()[e], x.act_a); \
            float b = mk_act(C[1][i][j].thread_elements()[e], x.act_b); \
            y[ulong(t) * x.p.ys + o + e] = x.kind == 0 ? a + b : a * b; \
        } \
    } \
}
#define MK_MLX2_DT(DT, SDT) \
MK_MLX2_QMM(mk_mlx2_qmm_##DT##_g32, 32, SDT) MK_MLX2_QMM(mk_mlx2_qmm_##DT##_g64, 64, SDT) \
MK_MLX2_QMM(mk_mlx2_qmm_##DT##_g128, 128, SDT) MK_MLX2_MULTI(mk_mlx2_multi_##DT##_g32, 32, SDT) \
MK_MLX2_MULTI(mk_mlx2_multi_##DT##_g64, 64, SDT) MK_MLX2_MULTI(mk_mlx2_multi_##DT##_g128, 128, SDT) \
MK_MLX2_GLU(mk_mlx2_glu_##DT##_g32, 32, SDT) MK_MLX2_GLU(mk_mlx2_glu_##DT##_g64, 64, SDT) \
MK_MLX2_GLU(mk_mlx2_glu_##DT##_g128, 128, SDT)
MK_MLX2_DT(f32, 0) MK_MLX2_DT(f16, 1) MK_MLX2_DT(bf16, 2)

// ---- c2-prefill-mlx (prefill, mlx-arith tier): stock MLX's steel attention, rescheduled ----
// Same arithmetic as mk_mlx_attention (mlx.metal: the port of steel_attention, float32,
// head dim 128, BQ 32, BK 16, WM 4, WN 1): every (row, key block) step, every 8 x 8 x 8
// operand sequence and every reduction is that kernel's, so the output bits are the
// same (tools/nano_mlx_ops.py checks them against stock MLX). What differs is only
// scheduling and staging, none of which reaches an output:
//   - threadgroup order: heaviest query blocks first (the last block of a causal
//     forward has the most keys), heads fastest, so the long blocks do not form a tail;
//   - the query fragments (16 per SIMD group) stay in registers, read once from device
//     memory (float2 per lane: query offset and row stride must be even), instead of
//     being re-read from threadgroup memory for every key block;
//   - K (transposed) and V tiles get their own threadgroup buffers and are both loaded
//     after one barrier, so a key block costs two barriers instead of four.
template <bool GATED>
inline void mk_mlx2_attention_tile(device float *arena, device const float *Kc, device const float *Vc, constant MK &p,
                                   uint tgid, uint simd, uint lane, uint lid,
                                   threadgroup float *Ks, threadgroup float *Vs) {
    uint NQ = (p.rows + 31) / 32;
    uint q0 = (NQ - 1 - tgid / p.heads) * 32, h = tgid % p.heads;
    uint kvh = h / (p.heads / p.kv_heads);
    uint first = p.pos + p.row0;          // position of batch row 0
    uint kL = first + p.rows;             // keys: positions [origin, kL)
    uint origin = (p.window != 0 && first + 1 > p.window) ? first + 1 - p.window : 0;
    uint qpos0 = first + q0;
    uint qlast = first + min(q0 + 32, p.rows) - 1;
    uint slot_words = p.kv_heads * 128;
    uint qid = lane / 4;
    uint sm = (qid & 4) + (lane / 2) % 4, sn = (qid & 2) * 2 + (lane % 2) * 2;
    uint tm = 8 * simd;
    uint row_pos = qpos0 + tm + sm;
    // Query fragments: Q[row tm + sm][8 dd + sn, + 1]; rows past the batch are zero (as Qs's).
    simdgroup_float8x8 Q[16];
    {
        uint r = q0 + tm + sm;
        device const float *qp = arena + p.x + (p.row0 + min(r, p.rows - 1)) * p.xs + h * 128 + sn;
        MK_UNROLL
        for (uint dd = 0; dd < 16; ++dd) {
            float2 v = r < p.rows ? *(device const float2 *)(qp + 8 * dd) : float2(0.0f);
            Q[dd].thread_elements()[0] = v.x;
            Q[dd].thread_elements()[1] = v.y;
        }
    }
    uint kb_first = 0;
    if (p.window != 0 && qpos0 + 1 > p.window) kb_first = (qpos0 + 1 - p.window - origin) / 16;
    uint kb_end = min((kL - origin + 15) / 16, (qlast + 1 - origin + 15) / 16);
    const float scale = p.scale * M_LOG2E_F;
    const float neg = -FLT_MAX;
    float max_score = neg, sum_score = 0;
    simdgroup_float8x8 O[16];
    MK_UNROLL
    for (uint id = 0; id < 16; ++id) O[id] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    for (uint kb = kb_first; kb < kb_end; ++kb) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // K block, transposed: Ks[d * LDK + j] = K[key origin + 16 kb + j][d]; V block:
        // Vs[j * LDV + d] = V[key][d]. Keys past kL are zero. Thread lid copies 16 dims of
        // key row lid / 8 of each (one slot lookup per row).
        {
            uint j = lid / 8, d0 = (lid % 8) * 16, key = origin + kb * 16 + j;
            threadgroup float4 *vd = (threadgroup float4 *)(Vs + j * MK_MLXA_LDV + d0);
            if (key < kL) {
                ulong slot = ulong(key < p.capacity ? key : key % p.capacity) * slot_words + kvh * 128 + d0;
                device const float4 *ks = (device const float4 *)(Kc + slot);
                device const float4 *vs = (device const float4 *)(Vc + slot);
                MK_UNROLL
                for (uint e = 0; e < 4; ++e) {
                    float4 v = ks[e];
                    Ks[(d0 + 4 * e) * MK_MLXA_LDK + j] = v.x;
                    Ks[(d0 + 4 * e + 1) * MK_MLXA_LDK + j] = v.y;
                    Ks[(d0 + 4 * e + 2) * MK_MLXA_LDK + j] = v.z;
                    Ks[(d0 + 4 * e + 3) * MK_MLXA_LDK + j] = v.w;
                    vd[e] = vs[e];
                }
            } else {
                for (uint e = 0; e < 16; ++e) Ks[(d0 + e) * MK_MLXA_LDK + j] = 0.0f;
                for (uint e = 0; e < 4; ++e) vd[e] = 0.0f;
            }
        }
        simdgroup_float8x8 S[2];
        S[0] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        S[1] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        MK_UNROLL
        for (uint dd = 0; dd < 16; ++dd) {
            simdgroup_float8x8 Kf[2];
            simdgroup_barrier(mem_flags::mem_none);
            MK_UNROLL
            for (uint j = 0; j < 2; ++j) {
                threadgroup const float *k = Ks + (dd * 8 + sm) * MK_MLXA_LDK + j * 8 + sn;
                Kf[j].thread_elements()[0] = k[0];
                Kf[j].thread_elements()[1] = k[1];
            }
            simdgroup_barrier(mem_flags::mem_none);
            MK_UNROLL
            for (uint j = 0; j < 2; ++j) simdgroup_multiply_accumulate(S[j], Q[dd], Kf[j], S[j]);
        }
        uint c0 = origin + kb * 16;
        bool all_visible = c0 + 15 < kL && c0 + 15 <= qpos0 && (p.window == 0 || qlast < c0 + p.window);
        MK_UNROLL
        for (uint j = 0; j < 2; ++j) for (uint jj = 0; jj < 2; ++jj) {
            float s = S[j].thread_elements()[jj] * scale;
            uint col = c0 + j * 8 + sn + jj;
            bool visible = all_visible || (col < kL && col <= row_pos && (p.window == 0 || row_pos < col + p.window));
            S[j].thread_elements()[jj] = visible ? s : neg;
        }
        // Online softmax in base 2 (mk_mlx_attention_tile's, line for line).
        float new_max = max_score;
        MK_UNROLL
        for (uint j = 0; j < 2; ++j) {
            float thr = max(S[j].thread_elements()[0], S[j].thread_elements()[1]);
            float qgr = max(thr, simd_shuffle_xor(thr, ushort(1)));
            float sgr = max(qgr, simd_shuffle_xor(qgr, ushort(8)));
            new_max = max(new_max, sgr);
        }
        MK_UNROLL
        for (uint j = 0; j < 2; ++j) for (uint jj = 0; jj < 2; ++jj)
            S[j].thread_elements()[jj] = fast::exp2(S[j].thread_elements()[jj] - new_max);
        float factor = fast::exp2(max_score - new_max);
        max_score = new_max;
        float sum_tmp = 0;
        MK_UNROLL
        for (uint j = 0; j < 2; ++j) {
            float thr = S[j].thread_elements()[0] + S[j].thread_elements()[1];
            float qgr = thr + simd_shuffle_xor(thr, ushort(1));
            float sgr = qgr + simd_shuffle_xor(qgr, ushort(8));
            sum_tmp = sum_tmp + sgr;
        }
        sum_score = sum_score * factor + sum_tmp;
        MK_UNROLL
        for (uint id = 0; id < 16; ++id) {
            O[id].thread_elements()[0] = O[id].thread_elements()[0] * factor;
            O[id].thread_elements()[1] = O[id].thread_elements()[1] * factor;
        }
        MK_UNROLL
        for (uint id = 0; id < 16; ++id) for (uint ik = 0; ik < 2; ++ik) {
            simdgroup_float8x8 Vf;
            simdgroup_barrier(mem_flags::mem_none);
            threadgroup const float *v = Vs + (ik * 8 + sm) * MK_MLXA_LDV + id * 8 + sn;
            Vf.thread_elements()[0] = v[0];
            Vf.thread_elements()[1] = v[1];
            simdgroup_barrier(mem_flags::mem_none);
            simdgroup_multiply_accumulate(O[id], S[ik], Vf, O[id]);
        }
    }
    uint r = q0 + tm + sm;
    if (r >= p.rows) return;
    uint row = p.row0 + r;
    device float *out = arena + p.y + row * p.ys + h * 128;
    float g = GATED ? mk_act(arena[p.z + row * p.zs + h], p.act_b) : 1.0f;
    MK_UNROLL
    for (uint id = 0; id < 16; ++id) for (uint jj = 0; jj < 2; ++jj) {
        float a = O[id].thread_elements()[jj] / sum_score;
        out[id * 8 + sn + jj] = GATED ? a * g : a;
    }
}
// Buffers and parameters as mk_mlx_attention[_gated]; grid ceil(rows / 32) * heads
// threadgroups of 128 threads.
#define MK_MLX2_ATT(NAME, GATED) \
[[kernel, max_total_threads_per_threadgroup(128)]] void NAME(device float *arena [[buffer(0)]], device const float *K [[buffer(1)]], \
        device const float *V [[buffer(2)]], constant MK &p [[buffer(8)]], uint tgid [[threadgroup_position_in_grid]], \
        uint simd [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]], \
        uint lid [[thread_index_in_threadgroup]]) { \
    threadgroup float Ks[128 * MK_MLXA_LDK], Vs[16 * MK_MLXA_LDV]; \
    mk_mlx2_attention_tile<GATED>(arena, K, V, p, tgid, simd, lane, lid, Ks, Vs); \
}
MK_MLX2_ATT(mk_mlx2_attention, false) MK_MLX2_ATT(mk_mlx2_attention_gated, true)

// ---- c2-prefill-mlx (knob conv_rows2): mk_gated_conv_rows without a modulo per tap ----
// Same arithmetic as mk_gated_conv_rows (prefill.metal), operation for operation:
// acc = 0; for k ascending: acc += w[c][k] * u(q = position - taps + 1 + k), u = 0 for
// q < 0; then the gate product. Only the ring slot of q is found differently: once,
// with 32-bit arithmetic, then advanced with a wrap (slot(q + 1) = slot(q) + 1, or 0
// at the capacity) instead of a 64-bit q % capacity per tap. Positions fit in 32 bits.
kernel void mk_gated_conv_rows2(device const float *ring [[buffer(0)]], device const uchar *w [[buffer(1)]],
                                device float *arena [[buffer(2)]], constant MK &p [[buffer(8)]],
                                uint i [[thread_position_in_grid]]) {
    uint r = i / p.n, c = i % p.n;
    if (r >= p.rows) return;
    uint row = p.row0 + r;
    uint position = p.pos + row;
    // q = position + 1 - taps + k; q < 0 exactly for the first taps - 1 - position taps.
    uint skip = position + 1 >= p.taps ? 0 : p.taps - 1 - position;
    uint slot = skip == 0 ? (position + 1 - p.taps) % p.capacity : 0; // slot of the first q >= 0
    float acc = 0;
    for (uint k = 0; k < p.taps; ++k) {
        float u = 0.0f;
        if (k >= skip) {
            u = ring[ulong(slot) * p.n + c];
            slot = slot + 1 == p.capacity ? 0 : slot + 1;
        }
        acc += mk_load(w, p.w, p.dtype, ulong(c) * p.taps + k) * u;
    }
    float other = arena[p.z + row * p.zs + c];
    float left = (p.kind & 1u) ? mk_act(acc, p.act_b & 3u) : mk_act(other, p.act_b & 3u);
    float right = (p.kind & 1u) ? mk_act(other, (p.act_b >> 2) & 3u) : mk_act(acc, (p.act_b >> 2) & 3u);
    arena[p.m + row * p.every + c] = left * right;
}
