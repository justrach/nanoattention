// ---- mlx-arith tier (prefill-mlx): MLX's batched affine 4-bit matrix multiply -------------
/*
 * Adapted from MLX v0.32.4 mlx/backend/metal/kernels/quantized.h (qmm_t_impl,
 * QuantizedBlockLoader, dequantize, affine_qmm_t, affine_qmm_t_splitk) and
 * mlx/backend/metal/kernels/steel/gemm/{loader.h, mma.h} (BlockLoader, BlockMMA)
 * and steel/gemm/kernels/steel_gemm_splitk.h (gemm_splitk_accum).
 * Copyright © 2023-2024 Apple Inc. (MLX MIT license, licenses/mlx-MIT.txt).
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions: the above copyright
 * notice and this permission notice shall be included in all copies or
 * substantial portions of the Software. THE SOFTWARE IS PROVIDED "AS IS",
 * WITHOUT WARRANTY OF ANY KIND.
 */
// NOT the reference arithmetic. These kernels reproduce stock MLX's float32
// `mx.quantized_matmul(x, w, scales, biases, transpose=True, group_size, bits=4)`
// bit for bit for M >= MLX's vector limit (exec.mlxLinear chooses the grid and
// split exactly as MLX's QuantizedMatmul::eval_gpu does; tools/nano_mlx_ops.py
// checks the bits). Every output y[t][o] is, in this order:
//   W[o][k] = scale * code + bias (float; MLX's dequantize, so 4-bit codes of
//             the high nibble are (scale / 16) * (code << 4): the same value)
//   C = 0; for each 32-wide K block k0 of the partition, ascending:
//          for kk = 0, 8, 16, 24: C = A * B + C, one simdgroup 8 x 8 x 8
//          multiply-accumulate with A[t][j] = x[t][k0 + kk + j] and
//          B[j][o] = W[o][k0 + kk + j]
//   split K (MLX qmm_t_splitk): each partition's C is a float partial; the
//          output is ((0 + P0) + P1) + ... in partition order (gemm_splitk_accum).
// The 8 x 8 x 8 step's internal order is the hardware's; using the same
// operands in the same sequence as MLX is what makes the bits equal.
// Tile: 32 tokens x 32 outputs x 32 inputs per threadgroup of 4 SIMD groups
// (MLX BM = BN = BK = 32, WM = WN = 2); SIMD group s owns tokens tm + {0, 16}
// and outputs tn + {0, 16} in 8 x 8 blocks, tm = 8 * (s / 2), tn = 8 * (s % 2).
// Parameters (MK): rows = M, n = K, m = N, x/xs input, y/ys output, w/s/b byte
// offsets, group (32, 64 or 128: one kernel each); kind = split_k (1 = no
// split), dim = K partition length.
constant uint MK_MLX_LD = 36; // BK + 16 / sizeof(float): MLX's padded tile row
#define MK_UNROLL _Pragma("clang loop unroll(full)") // MLX's STEEL_PRAGMA_UNROLL

// Scale/bias element of dtype SDT (MK dtype codes: 0 float32, 1 float16, 2 bfloat16)
// as a float: exact, as MLX's float32 model holds them after set_dtype.
template <uint SDT>
inline float mk_mlx_sb(device const uchar *a) {
    if (SDT == 0) return *(device const float *)a;
    if (SDT == 1) return float(*(device const half *)a);
    return as_type<float>(uint(*(device const ushort *)a) << 16);
}
// The K loop (MLX's load_unsafe path when FULL: every row and column of the tile in range).
template <bool FULL, uint GROUP, uint SDT>
inline void mk_mlx_qmm_loop(thread simdgroup_float8x8 (&C)[2][2], device const float *x, device const uchar *w,
                            device const uchar *sp, device const uchar *bp, uint k_start, uint k_len,
                            uint li, uint num_els, uint num_outs, threadgroup float *xd, threadgroup float *wd,
                            threadgroup const float *Xs, threadgroup const float *Ws,
                            threadgroup const float *As, threadgroup const float *Bs) {
    constexpr uint esz = SDT == 0 ? 4 : 2;
    for (uint k0 = k_start; k0 < k_start + k_len;
         sp += (k0 + 32) % GROUP == 0 ? esz : 0, bp += (k0 + 32) % GROUP == 0 ? esz : 0, k0 += 32, x += 32, w += 16) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // x tile (BlockLoader): a copy; rows past M are zero.
        if (FULL || li < num_els) {
            ((threadgroup float4 *)xd)[0] = ((device const float4 *)x)[0];
            ((threadgroup float4 *)xd)[1] = ((device const float4 *)x)[1];
        } else {
            ((threadgroup float4 *)xd)[0] = 0;
            ((threadgroup float4 *)xd)[1] = 0;
        }
        // W tile (QuantizedBlockLoader + dequantize<float, 2, 4>): rows past N are zero.
        if (FULL || li < num_outs) {
            float s = mk_mlx_sb<SDT>(sp), b = mk_mlx_sb<SDT>(bp);
            // The thread's 4 code bytes (bytewise, as MLX's loader; data may be 2-byte aligned).
            uint bytes[4] = {uint(w[0]), uint(w[1]), uint(w[2]), uint(w[3])};
            float v[8];
            MK_UNROLL
            for (uint j = 0; j < 4; ++j) {
                v[2 * j] = s * float(bytes[j] & 0x0f) + b;
                v[2 * j + 1] = (s / 16.0f) * float(bytes[j] & 0xf0) + b;
            }
            ((threadgroup float4 *)wd)[0] = float4(v[0], v[1], v[2], v[3]);
            ((threadgroup float4 *)wd)[1] = float4(v[4], v[5], v[6], v[7]);
        } else {
            ((threadgroup float4 *)wd)[0] = 0;
            ((threadgroup float4 *)wd)[1] = 0;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // BlockMMA::mma: four 8-wide steps along K; C = A * B + C per 8 x 8 block.
        MK_UNROLL
        for (uint kk = 0; kk < 32; kk += 8) {
            simdgroup_float8x8 A[2], B[2];
            simdgroup_barrier(mem_flags::mem_none);
            MK_UNROLL
            for (uint i = 0; i < 2; ++i) {
                threadgroup const float *a = As + 16 * i * MK_MLX_LD + kk;
                A[i].thread_elements()[0] = a[0];
                A[i].thread_elements()[1] = a[1];
            }
            simdgroup_barrier(mem_flags::mem_none);
            MK_UNROLL
            for (uint j = 0; j < 2; ++j) {
                threadgroup const float *b = Bs + 16 * j * MK_MLX_LD + kk;
                B[j].thread_elements()[0] = b[0];
                B[j].thread_elements()[1] = b[MK_MLX_LD];
            }
            simdgroup_barrier(mem_flags::mem_none);
            MK_UNROLL
            for (uint i = 0; i < 2; ++i) for (uint jj = 0; jj < 2; ++jj) {
                uint j = (i % 2) ? 1 - jj : jj; // MLX tile_matmad's serpentine order (independent blocks)
                simdgroup_multiply_accumulate(C[i][j], A[i], B[j], C[i][j]);
            }
        }
    }
}
// PR: `constant MK &` (one matrix) or `thread const MK &` (one matrix of mk_mlx_qmm_multi).
template <bool SPLIT, uint GROUP, uint SDT, typename PR>
inline void mk_mlx_qmm_tile(device const uchar *wb, device const uchar *sb, device const uchar *bb,
                            device float *arena, device float *partials, PR p,
                            uint tgid, uint simd, uint lane, uint lid,
                            threadgroup float *Xs, threadgroup float *Ws) {
    uint M = p.rows, N = p.m, K = p.n;
    uint n_tiles = (N + 31) / 32, m_tiles = (M + 31) / 32;
    uint y_col = (tgid % n_tiles) * 32, y_row = ((tgid / n_tiles) % m_tiles) * 32;
    uint part = tgid / (n_tiles * m_tiles);
    uint k_len = SPLIT ? p.dim : K;
    uint k_start = part * k_len;
    uint num_els = min(32u, M - y_row), num_outs = min(32u, N - y_col);
    // Loaders: thread lid fills 8 consecutive columns 8 * (lid % 4) of tile row lid / 4.
    uint li = lid / 4, lj = 8 * (lid % 4);
    uint tm = 8 * (simd / 2), tn = 8 * (simd % 2);
    threadgroup float *xd = Xs + li * MK_MLX_LD + lj;
    threadgroup float *wd = Ws + li * MK_MLX_LD + lj;
    // Per-thread sources, advanced by one K block per step as MLX's loaders do.
    device const float *x = arena + p.x + (p.row0 + y_row + min(li, num_els - 1)) * p.xs + k_start + lj;
    uint wrow = y_col + min(li, num_outs - 1);
    device const uchar *w = wb + p.w + ulong(wrow) * (K / 2) + (k_start + lj) / 2;
    // Scale and bias of the thread's group (lj < 32 <= GROUP: one group per K block),
    // advanced as MLX's loader does. Compile-time dtype: a run-time switch here cost ~7%.
    constexpr uint esz = SDT == 0 ? 4 : 2;
    device const uchar *sp = sb + p.s + (ulong(wrow) * (K / GROUP) + k_start / GROUP) * esz;
    device const uchar *bp = bb + p.b + (ulong(wrow) * (K / GROUP) + k_start / GROUP) * esz;
    // BlockMMA's fragment layout: lane holds elements (sm, sn) and (sm, sn + 1) of each 8 x 8 block.
    uint qid = lane / 4;
    uint sm = (qid & 4) + (lane / 2) % 4, sn = (qid & 2) * 2 + (lane % 2) * 2;
    threadgroup const float *As = Xs + (tm + sm) * MK_MLX_LD + sn;  // A[t][k] = Xs[t][k]
    threadgroup const float *Bs = Ws + (tn + sn) * MK_MLX_LD + sm;  // B[k][o] = Ws[o][k]
    simdgroup_float8x8 C[2][2];
    for (uint i = 0; i < 2; ++i) for (uint j = 0; j < 2; ++j) C[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    if (num_els == 32 && num_outs == 32)
        mk_mlx_qmm_loop<true, GROUP, SDT>(C, x, w, sp, bp, k_start, k_len, li, num_els, num_outs, xd, wd, Xs, Ws, As, Bs);
    else
        mk_mlx_qmm_loop<false, GROUP, SDT>(C, x, w, sp, bp, k_start, k_len, li, num_els, num_outs, xd, wd, Xs, Ws, As, Bs);
    // Store (a copy). Full tiles straight from the simdgroup matrices; partial
    // tiles staged in Xs (no extra threadgroup memory: MLX's 9 KB per group),
    // then only the rows and columns in range written.
    if (num_els == 32 && num_outs == 32) {
        for (uint i = 0; i < 2; ++i) for (uint j = 0; j < 2; ++j) {
            uint t = y_row + tm + 16 * i, o = y_col + tn + 16 * j;
            if (SPLIT) simdgroup_store(C[i][j], partials + (ulong(part) * M + t) * N + o, N);
            else simdgroup_store(C[i][j], arena + p.y + ulong(p.row0 + t) * p.ys + o, p.ys);
        }
        return;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = 0; i < 2; ++i) for (uint j = 0; j < 2; ++j)
        simdgroup_store(C[i][j], Xs + (tm + 16 * i) * 32 + tn + 16 * j, 32);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint e = lid; e < 32 * 32; e += 128) {
        uint r = e / 32, c = e % 32;
        if (r >= num_els || c >= num_outs) continue;
        uint t = y_row + r, o = y_col + c;
        if (SPLIT) partials[(ulong(part) * M + t) * N + o] = Xs[e];
        else arena[p.y + (p.row0 + t) * p.ys + o] = Xs[e];
    }
}
#define MK_MLX_QMM(NAME, SPLIT, GROUP, SDT) \
[[kernel]] void NAME(device const uchar *w [[buffer(0)]], device const uchar *s [[buffer(1)]], \
        device const uchar *b [[buffer(2)]], device float *arena [[buffer(3)]], device float *partials [[buffer(4)]], \
        constant MK &p [[buffer(8)]], uint tgid [[threadgroup_position_in_grid]], \
        uint simd [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]], \
        uint lid [[thread_index_in_threadgroup]]) { \
    threadgroup float Xs[32 * MK_MLX_LD], Ws[32 * MK_MLX_LD]; \
    mk_mlx_qmm_tile<SPLIT, GROUP, SDT, constant MK &>(w, s, b, arena, partials, p, tgid, simd, lane, lid, Xs, Ws); \
}
// MLX instantiates group sizes 32, 64 and 128 (a compile-time constant there too);
// here also one kernel per scale/bias dtype (mk_mlx_qmm_t[_splitk]_<dtype>_g<group>).
#define MK_MLX_QMM_DT(DT, SDT) \
MK_MLX_QMM(mk_mlx_qmm_t_##DT##_g32, false, 32, SDT) MK_MLX_QMM(mk_mlx_qmm_t_##DT##_g64, false, 64, SDT) \
MK_MLX_QMM(mk_mlx_qmm_t_##DT##_g128, false, 128, SDT) MK_MLX_QMM(mk_mlx_qmm_t_splitk_##DT##_g32, true, 32, SDT) \
MK_MLX_QMM(mk_mlx_qmm_t_splitk_##DT##_g64, true, 64, SDT) MK_MLX_QMM(mk_mlx_qmm_t_splitk_##DT##_g128, true, 128, SDT)
MK_MLX_QMM_DT(f32, 0) MK_MLX_QMM_DT(f16, 1) MK_MLX_QMM_DT(bf16, 2)
// knob mlx_multi: up to 4 unsplit products of one input (q/k/v projections; w1 and
// w3) in one launch. Threadgroup tgid takes the matrices' tiles in order; each tile
// is exactly mk_mlx_qmm_t's for that matrix (same tile, same K order, same store),
// so every output has stock MLX's bits; only launches and barriers go. Matrices:
// x.mat[0..count) (w/s/b offsets in one buffer, m = N, y/ys); x.p: rows, row0, n = K, x/xs.
#define MK_MLX_QMM_MULTI(NAME, GROUP, SDT) \
[[kernel]] void NAME(device const uchar *w [[buffer(0)]], device float *arena [[buffer(1)]], \
        constant MKX &x [[buffer(8)]], uint tgid [[threadgroup_position_in_grid]], \
        uint simd [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]], \
        uint lid [[thread_index_in_threadgroup]]) { \
    threadgroup float Xs[32 * MK_MLX_LD], Ws[32 * MK_MLX_LD]; \
    uint m_tiles = (x.p.rows + 31) / 32, t = tgid, k = 0; \
    for (; k + 1 < x.count; ++k) { \
        uint nt = ((x.mat[k].m + 31) / 32) * m_tiles; \
        if (t < nt) break; \
        t -= nt; \
    } \
    MK q = x.p; \
    q.w = x.mat[k].w; q.s = x.mat[k].s; q.b = x.mat[k].b; q.m = x.mat[k].m; q.y = x.mat[k].y; q.ys = x.mat[k].ys; \
    mk_mlx_qmm_tile<false, GROUP, SDT, thread const MK &>(w, w, w, arena, arena, q, t, simd, lane, lid, Xs, Ws); \
}
#define MK_MLX_QMM_MULTI_DT(DT, SDT) \
MK_MLX_QMM_MULTI(mk_mlx_qmm_multi_##DT##_g32, 32, SDT) MK_MLX_QMM_MULTI(mk_mlx_qmm_multi_##DT##_g64, 64, SDT) \
MK_MLX_QMM_MULTI(mk_mlx_qmm_multi_##DT##_g128, 128, SDT)
MK_MLX_QMM_MULTI_DT(f32, 0) MK_MLX_QMM_MULTI_DT(f16, 1) MK_MLX_QMM_MULTI_DT(bf16, 2)
// gemm_splitk_accum: out = 0; out += P[i] for partitions i ascending. Grid M * N.
kernel void mk_mlx_splitk_accum(device const float *partials [[buffer(0)]], device float *arena [[buffer(1)]],
                                constant MK &p [[buffer(8)]], uint i [[thread_position_in_grid]]) {
    uint M = p.rows, N = p.m;
    uint t = i / N, o = i % N;
    if (t >= M) return;
    float out = 0;
    for (uint k = 0; k < p.kind; ++k) out += partials[(ulong(k) * M + t) * N + o];
    arena[p.y + (p.row0 + t) * p.ys + o] = out;
}

// ---- mlx-arith tier (prefill-mlx): MLX's steel attention, float32, head dim 128 -----------
// Adapted from MLX v0.32.4 mlx/backend/metal/kernels/steel/attn/kernels/steel_attention.h
// and steel/attn/{mma.h, loader.h} (same copyright and MIT notice as above). The
// instantiation stock MLX runs for float32 queries longer than 8 rows at head dim
// 128 on a non-NAX GPU: BQ = 32 query rows, BK = 16 keys, BD = BV = 128, WM = 4
// SIMD groups of 8 rows each, WN = 1, Q staged in threadgroup memory.
// Per query row (thread elements in MLX's 8 x 8 fragment layout), for each key
// block kb in order:
//   S = 0; for dd = 0..15: S = Q[:, 8dd..] * K^T[8dd.., :] + S   (8 x 8 x 8 steps)
//   S *= scale * log2(e)                         (float product computed here)
//   masked entries (key past the last row's position, key after the query,
//     key before the window) := -FLT_MAX        (MLX's finite_min)
//   m' = max(m, rowmax(S)); S = fast::exp2(S - m'); f = fast::exp2(m - m')
//   l = l * f + rowsum(S); O *= f; for 8-key halves ik: O = S_ik * V_ik + O
// and finally O /= l. rowmax/rowsum over a fragment's 8 columns:
//   ((c0 op c1) op (c2 op c3)) op ((c4 op c5) op (c6 op c7))   (shuffle xor 1, then 8)
// folded into the running value fragment by fragment. m starts at -FLT_MAX, l at 0.
// Keys are positions [origin, pos + row0 + rows); query rows are the last `rows`
// of them. Full attention: origin 0 (mlx_vlm's KVCache keeps every position).
// Sliding windows: origin = max(0, first - (window - 1)) for a forward whose first
// row is at position `first`, because mlx_vlm's RotatingKVCache hands a multi-row
// chunk only the last window - 1 cached keys plus the chunk's own; MLX's 16-key
// blocks start at the origin, so the blocks (and the rounding) follow it.
// MLX applies an array mask there and visits every key block;
// here only blocks that hold a visible key for some row of the query block
// are visited. The others are exact no-ops in MLX (all-masked blocks after a
// row's first visible key contribute exp2(-FLT_MAX - m) = 0 with factor 1;
// blocks before it are cleared by a factor of exp2(-FLT_MAX - m') = 0), so the
// result is the same up to the sign of an exactly-zero output.
// K/V rows come from the engine's caches: position t is slot t % capacity,
// p.kv_heads * 128 words per slot (cache buffers are 16-byte aligned, read as float4). Parameters (MK): rows, row0, pos, heads,
// kv_heads, dim (128), window (0 = full), scale, capacity, x/xs queries, y/ys
// output, z/zs gate and act_b (GATED: output * act(gate[head]), mk_binary's formula).
constant uint MK_MLXA_LDQ = 132, MK_MLXA_LDK = 20, MK_MLXA_LDV = 132;

template <bool GATED>
inline void mk_mlx_attention_tile(device float *arena, device const float *Kc, device const float *Vc, constant MK &p,
                                  uint tgid, uint simd, uint lane, uint lid,
                                  threadgroup float *Qs, threadgroup float *KVs) {
    uint NQ = (p.rows + 31) / 32;
    uint q0 = (tgid % NQ) * 32, h = tgid / NQ;
    uint kvh = h / (p.heads / p.kv_heads);
    uint first = p.pos + p.row0;          // position of batch row 0
    uint kL = first + p.rows;             // keys: positions [origin, kL)
    uint origin = (p.window != 0 && first + 1 > p.window) ? first + 1 - p.window : 0;
    uint qpos0 = first + q0;
    uint qlast = first + min(q0 + 32, p.rows) - 1;
    uint slot_words = p.kv_heads * 128;
    // Fragment coordinates (BaseMMAFrag::get_coord): this lane holds row sm, columns sn, sn + 1.
    uint qid = lane / 4;
    uint sm = (qid & 4) + (lane / 2) % 4, sn = (qid & 2) * 2 + (lane % 2) * 2;
    uint tm = 8 * simd;
    uint row_pos = qpos0 + tm + sm;       // this thread's query position
    // Q tile: rows past the batch are zero.
    for (uint i = lid; i < 32 * 128; i += 128) {
        uint r = i / 128, d = i % 128;
        Qs[r * MK_MLXA_LDQ + d] = q0 + r < p.rows ? arena[p.x + (p.row0 + q0 + r) * p.xs + h * 128 + d] : 0.0f;
    }
    // Key block kb holds positions origin + 16 kb .. origin + 16 kb + 15.
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
        // K block, transposed: KVs[d * LDK + j] = K[key origin + kb * 16 + j][d]; keys past kL are zero.
        // Thread lid copies 16 dims of key row lid / 8 (one slot lookup per row).
        {
            uint j = lid / 8, d0 = (lid % 8) * 16, key = origin + kb * 16 + j;
            if (key < kL) {
                device const float4 *src = (device const float4 *)(Kc + ulong(key < p.capacity ? key : key % p.capacity) * slot_words + kvh * 128 + d0);
                MK_UNROLL
                for (uint e = 0; e < 4; ++e) {
                    float4 v = src[e];
                    KVs[(d0 + 4 * e) * MK_MLXA_LDK + j] = v.x;
                    KVs[(d0 + 4 * e + 1) * MK_MLXA_LDK + j] = v.y;
                    KVs[(d0 + 4 * e + 2) * MK_MLXA_LDK + j] = v.z;
                    KVs[(d0 + 4 * e + 3) * MK_MLXA_LDK + j] = v.w;
                }
            } else {
                for (uint e = 0; e < 16; ++e) KVs[(d0 + e) * MK_MLXA_LDK + j] = 0.0f;
            }
        }
        simdgroup_float8x8 S[2];
        S[0] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        S[1] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        MK_UNROLL
        for (uint dd = 0; dd < 16; ++dd) {
            simdgroup_float8x8 Qf, Kf[2];
            simdgroup_barrier(mem_flags::mem_none);
            threadgroup const float *q = Qs + (tm + sm) * MK_MLXA_LDQ + dd * 8 + sn;
            Qf.thread_elements()[0] = q[0];
            Qf.thread_elements()[1] = q[1];
            MK_UNROLL
            for (uint j = 0; j < 2; ++j) {
                threadgroup const float *k = KVs + (dd * 8 + sm) * MK_MLXA_LDK + j * 8 + sn;
                Kf[j].thread_elements()[0] = k[0];
                Kf[j].thread_elements()[1] = k[1];
            }
            simdgroup_barrier(mem_flags::mem_none);
            MK_UNROLL
            for (uint j = 0; j < 2; ++j) simdgroup_multiply_accumulate(S[j], Qf, Kf[j], S[j]);
        }
        // Mask only blocks with an invisible (row, key) pair for some row of the query
        // block (MLX: causal masking near the diagonal only); elsewhere just the scale.
        uint c0 = origin + kb * 16;
        bool all_visible = c0 + 15 < kL && c0 + 15 <= qpos0 && (p.window == 0 || qlast < c0 + p.window);
        MK_UNROLL
        for (uint j = 0; j < 2; ++j) for (uint jj = 0; jj < 2; ++jj) {
            float s = S[j].thread_elements()[jj] * scale;
            uint col = c0 + j * 8 + sn + jj;
            bool visible = all_visible || (col < kL && col <= row_pos && (p.window == 0 || row_pos < col + p.window));
            S[j].thread_elements()[jj] = visible ? s : neg;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // V block: KVs[j * LDV + d] = V[key origin + kb * 16 + j][d]; keys past kL are zero.
        {
            uint j = lid / 8, d0 = (lid % 8) * 16, key = origin + kb * 16 + j;
            threadgroup float4 *dst = (threadgroup float4 *)(KVs + j * MK_MLXA_LDV + d0);
            if (key < kL) {
                device const float4 *src = (device const float4 *)(Vc + ulong(key < p.capacity ? key : key % p.capacity) * slot_words + kvh * 128 + d0);
                for (uint e = 0; e < 4; ++e) dst[e] = src[e];
            } else {
                for (uint e = 0; e < 4; ++e) dst[e] = 0.0f;
            }
        }
        // Online softmax in base 2.
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
        threadgroup_barrier(mem_flags::mem_threadgroup);
        MK_UNROLL
        for (uint id = 0; id < 16; ++id) for (uint ik = 0; ik < 2; ++ik) {
            simdgroup_float8x8 Vf;
            simdgroup_barrier(mem_flags::mem_none);
            threadgroup const float *v = KVs + (ik * 8 + sm) * MK_MLXA_LDV + id * 8 + sn;
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
// max_total_threads_per_threadgroup(128) as MLX declares it: lets the compiler keep
// the 16 output fragments in registers (no arithmetic effect).
#define MK_MLX_ATT(NAME, GATED) \
[[kernel, max_total_threads_per_threadgroup(128)]] void NAME(device float *arena [[buffer(0)]], device const float *K [[buffer(1)]], \
        device const float *V [[buffer(2)]], constant MK &p [[buffer(8)]], uint tgid [[threadgroup_position_in_grid]], \
        uint simd [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]], \
        uint lid [[thread_index_in_threadgroup]]) { \
    threadgroup float Qs[32 * MK_MLXA_LDQ], KVs[128 * MK_MLXA_LDK]; \
    mk_mlx_attention_tile<GATED>(arena, K, V, p, tgid, simd, lane, lid, Qs, KVs); \
}
MK_MLX_ATT(mk_mlx_attention, false) MK_MLX_ATT(mk_mlx_attention_gated, true)
