// ===== climb 2, exact prefill: exact prefill on the matrix units ===========================
// Every kernel here reproduces, for each (token, output) word, the reference
// kernels' expression tree (kernels.metal) bit for bit; tests in kernel_tests.zig
// (climb 2 exact-prefill block) check that on random and adversarial inputs.
//
// ---- 4-bit linear on simdgroup 8 x 8 x 8 steps (knob x2_linear) --------------------
// The reference output y[t][r] of mk_q4_tile (Lean Emu.Mega.linearQ4) is
//   acc_L = +0; for seg = L, L + 32, ... (ascending):
//     dot = +0; for quad q = 0..3: dot = dot + i3(q), where, with prescaled inputs
//       x0..x3 = v0, v1 / 16, v2 / 256, v3 / 4096 and in-place-masked codes c0..c3,
//       i1 = fma(x0, c0, x1 * c1); i2 = fma(x2, c2, i1); i3 = fma(x3, c3, i2)
//     acc_L = acc_L + fma(scale, dot, xsum * bias)
//   y = adjacent-pair tree (simd_sum) over acc_0 .. acc_31.
// simdgroup_multiply_accumulate computes each element as acc = C; for k = 0..7:
// acc = fma(A[r][k], B[k][c], acc) (measured, Lean Emu.Mlx header). The kernel
// fills A (tokens x slots) and B (slots x rows) with two quads q, q + 1: slots
// 0..3 = quad q's inputs and masked codes in input order (1, 0, 2, 3), slots
// 4..7 = quad q + 1's. Per lane, A_top keeps slots 0..3 and zeroes 4..7, A_bot
// the reverse. With C = +0:
//   A_top x B: j3 = fma(x3, c3, fma(x2, c2, fma(x0, c0, fma(x1, c1, +0)))) for
//     quad q, then fma(+0, c, j3) four times; c is a finite code >= +0, so these
//     add +0 and return j3 unless j3 is -0 (see below: it is not).
//   A_bot x B: fma(+0, c, +0) = +0 four times, then j3 of quad q + 1 from +0.
// j3 equals the reference's i3 except in one case: all four products are exact
// -0, where i3 = -0 and j3 = +0 (any nonzero product makes the two chains equal
// from that product on). That slip changes `dot` only if dot is -0 when it is
// added, and dot (from +0) can become -0 only by the flush of an exact nonzero
// partial sum below 2^-126 (Metal flushes such results; Lean Emu.F32 Mode.metal).
// If every prescaled input of the token is a multiple of 2^-126 (zero, or
// |x| >= 2^-103), every product, chain value and partial sum is a multiple of
// 2^-126 and none is flushed: j3 and dot are never -0 and the slip cannot reach
// dot. mk_x2_prep flags each token row with a nonzero prescaled input below
// 2^-103; a threadgroup whose tokens include a flagged row computes its whole
// tile with the reference's scalar tree instead.
// Everything after the quad chains (dot sums, scale and bias term, per-lane
// accumulation in segment order, the pair tree over the 32 lane accumulators)
// is the reference's element-wise arithmetic on the step outputs.

// Prepared inputs (mk_x2_prep), per dispatch row t (batch-relative), rows R:
//   X: float4 ((seg * 4 + q) * R + t) = (x1, x0, x2, x3) of quad q of segment seg
//   S: word xo + seg * R + t = xsum of (t, seg)       (xo = p.z)
//   F: word fo + t = 1 if row t has a nonzero prescaled input below 2^-103 (fo = p.every)
// Grid: one SIMD group per row; lane L prepares segments L, L + 32, ...
// The xsum and prescaled inputs are mk_q4_prep's expressions.
kernel void mk_x2_prep(device const float *arena [[buffer(0)]], device float *prep [[buffer(1)]],
                       constant MK &p [[buffer(8)]], uint i [[thread_position_in_grid]],
                       uint lane [[thread_index_in_simdgroup]]) {
    uint segs = p.n / 16;
    uint t = i / 32;
    if (t >= p.rows) return;
    device float4 *X = (device float4 *)prep;
    bool tiny = false;
    for (uint seg = lane; seg < segs; seg += 32) {
        device const float *x = arena + p.x + (p.row0 + t) * p.xs + seg * 16;
        float xsum = 0;
        for (uint j = 0; j < 16; j += 4) {
            float v0 = x[j], v1 = x[j + 1], v2 = x[j + 2], v3 = x[j + 3];
            xsum += v0 + v1 + v2 + v3;
            float4 q = float4(v1 / 16.0f, v0, v2 / 256.0f, v3 / 4096.0f);
            X[ulong(seg * 4 + j / 4) * p.rows + t] = q;
            float4 a = abs(q);
            tiny = tiny || any(a < 0x1p-103f && a != 0.0f);
        }
        prep[p.z + ulong(seg) * p.rows + t] = xsum;
    }
    tiny = simd_any(tiny);
    if (lane == 0) ((device uint *)prep)[p.every + t] = tiny ? 1u : 0u;
}

#define X2U _Pragma("clang loop unroll(full)")
// Fold lane accumulator acc_L (L = 0..31 in order) into the adjacent-pair tree:
// s_k holds the pending left subtree of level k; after L = 31, s0 is the total.
#define MK_X2_FOLD(L, a, s0, s1, s2, s3, s4)                                       \
    {                                                                              \
        float v_ = (a);                                                            \
        if (!((L) & 1)) s0 = v_;                                                   \
        else {                                                                     \
            v_ = s0 + v_;                                                          \
            if (!((L) & 2)) s1 = v_;                                               \
            else {                                                                 \
                v_ = s1 + v_;                                                      \
                if (!((L) & 4)) s2 = v_;                                           \
                else {                                                             \
                    v_ = s2 + v_;                                                  \
                    if (!((L) & 8)) s3 = v_;                                       \
                    else {                                                         \
                        v_ = s3 + v_;                                              \
                        if (!((L) & 16)) s4 = v_;                                  \
                        else s0 = s4 + v_;                                         \
                    }                                                              \
                }                                                                  \
            }                                                                      \
        }                                                                          \
    }

// Scalar fallback: the reference tree for one (token, row) pair, all 32 lanes.
inline float mk_x2_scalar(device const uchar *wb, device const uchar *sb, device const uchar *bb,
                          device const float *prep, constant MK &p, uint t, uint row, uint lane) {
    uint segs = p.n / 16, R = p.rows;
    uint row_words = p.n / 8, row_groups = p.n / p.group, seg_group = p.group / 16;
    device const float4 *X = (device const float4 *)prep;
    float acc = 0;
    for (uint seg = lane; seg < segs; seg += 32) {
        ulong base = ulong(row) * row_words + seg * 2;
        uint c0 = mk_word(wb, p.w, base), c1 = mk_word(wb, p.w, base + 1);
        uint group = row * row_groups + seg / seg_group;
        float scale = mk_load(sb, p.s, p.sdtype, group), bias = mk_load(bb, p.b, p.bdtype, group);
        float xsum = prep[p.z + ulong(seg) * R + t];
        float dot = 0;
        for (uint j = 0; j < 16; j += 4) {
            float4 q = X[ulong(seg * 4 + j / 4) * R + t];
            uint codes = j < 8 ? c0 : c1;
            uint hw = (j & 7u) < 4 ? codes : codes >> 16;
            dot += (q.y * float(hw & 0x000fu) + q.x * float(hw & 0x00f0u) +
                    q.z * float(hw & 0x0f00u) + q.w * float(hw & 0xf000u));
        }
        acc += scale * dot + xsum * bias;
    }
    return simd_sum(acc);
}

// The kernel. A threadgroup of WM x WN SIMD groups owns BM = 8 MT WM tokens x
// BN = 8 NT WN rows; SIMD group (wm, wn) owns MT x NT blocks of 8 x 8 (fragment
// layout: lane holds row sm, columns sn and sn + 1 of a block). All SIMD groups
// walk the reference's virtual lanes L = 0..31 in order and, for each, its
// segments L, L + 32, ... in order, KS segments per stage. A stage's prepared
// inputs, unpacked codes, scales, biases and xsums sit in threadgroup memory,
// two quads per fragment:
//   Xa[kk][quad pair][token][8] (A: rows = tokens), Wb[kk][quad pair][slot][BN + 4] (B)
// The next stage's device loads are issued into registers before the current
// stage is computed and stored after it (software pipeline). Each lane
// accumulator acc_L is folded into the pair tree as soon as L is complete.
template <uint MT, uint NT, uint WM, uint WN, uint KS, uint SDT, typename WT = float>
inline void mk_x2mm_tile(device const uchar *wb, device const uchar *sb, device const uchar *bb,
                       device const float *prep, device float *arena, constant MK &p,
                       uint tgid, uint simd, uint lane, uint lid,
                       threadgroup float *Xa, threadgroup WT *Wb, threadgroup float *Sc, threadgroup float *Bi,
                       threadgroup float *Xsum) {
    constexpr uint BM = 8 * MT * WM, BN = 8 * NT * WN, TG = 32 * WM * WN, BNP = BN + 4;
    constexpr uint NX = (KS * 4 * BM + TG - 1) / TG, NW = (KS * 2 * BN + TG - 1) / TG;
    constexpr uint NS = (KS * BN + TG - 1) / TG, NM = (KS * BM + TG - 1) / TG;
    uint segs = p.n / 16, R = p.rows;
    uint row_words = p.n / 8, row_groups = p.n / p.group, seg_group = p.group / 16;
    uint tiles_n = (p.m + BN - 1) / BN;
    uint n0 = (tgid % tiles_n) * BN, m0 = (tgid / tiles_n) * BM;
    uint tn = (simd % WN) * 8 * NT, tm = (simd / WN) * 8 * MT;
    uint qid = lane / 4;
    uint sm = (qid & 4) + (lane / 2) % 4, sn = (qid & 2) * 2 + (lane % 2) * 2;
    device const uint *F = (device const uint *)prep + p.every;
    bool flagged = false;
    for (uint t = lane; t < BM; t += 32) flagged = flagged || F[min(m0 + t, R - 1)] != 0;
    if (simd_any(flagged)) {
        for (uint i = 0; i < 8 * MT; ++i) {
            uint t = m0 + tm + i;
            if (t >= R) break;
            for (uint j = 0; j < 8 * NT; ++j) {
                uint row = n0 + tn + j;
                if (row >= p.m) break;
                float y = mk_x2_scalar(wb, sb, bb, prep, p, t, row, lane);
                if (lane == 0) arena[p.y + (p.row0 + t) * p.ys + row] = y;
            }
        }
        return;
    }
    device const float4 *X = (device const float4 *)prep;
    constexpr uint esz = SDT == 0 ? 4 : 2;
    // Prefetch registers for one stage (L, k0) with ks segments.
    float4 rx[NX];
    uint rw[NW];
    float rs[NS], rb[NS], rm[NM];
    auto nkOf = [&](uint L) { return L < segs ? (segs - L + 31) / 32 : 0u; };
    auto fetch = [&](uint L, uint k0) {
        uint ks = min(KS, nkOf(L) - k0);
        X2U for (uint c = 0; c < NX; ++c) {
            uint k = lid + c * TG;
            uint kk = min(k / (4 * BM), ks - 1), r = k % (4 * BM);
            uint seg = L + 32 * (k0 + kk);
            rx[c] = X[ulong(seg * 4 + r / BM) * R + min(m0 + r % BM, R - 1)];
        }
        X2U for (uint c = 0; c < NW; ++c) {
            uint k = lid + c * TG;
            uint kk = min(k / (2 * BN), ks - 1), r = k % (2 * BN);
            uint seg = L + 32 * (k0 + kk);
            rw[c] = mk_word(wb, p.w, ulong(min(n0 + r % BN, p.m - 1)) * row_words + seg * 2 + r / BN);
        }
        X2U for (uint c = 0; c < NS; ++c) {
            uint k = lid + c * TG;
            uint kk = min(k / BN, ks - 1);
            uint seg = L + 32 * (k0 + kk);
            ulong g = ulong(min(n0 + k % BN, p.m - 1)) * row_groups + seg / seg_group;
            rs[c] = mk_mlx_sb<SDT>(sb + p.s + g * esz);
            rb[c] = mk_mlx_sb<SDT>(bb + p.b + g * esz);
        }
        X2U for (uint c = 0; c < NM; ++c) {
            uint k = lid + c * TG;
            uint kk = min(k / BM, ks - 1);
            uint seg = L + 32 * (k0 + kk);
            rm[c] = prep[p.z + ulong(seg) * R + min(m0 + k % BM, R - 1)];
        }
    };
    auto store = [&]() {
        X2U for (uint c = 0; c < NX; ++c) {
            uint k = lid + c * TG;
            if (k >= KS * 4 * BM) continue;
            // Xa[kk][quad pair][token][8]: slots 4 (q % 2) .. +3 = quad q's inputs 1, 0, 2, 3.
            uint kk = k / (4 * BM), r = k % (4 * BM), q = r / BM, t = r % BM;
            ((threadgroup float4 *)Xa)[((kk * 2 + q / 2) * BM + t) * 2 + q % 2] = rx[c];
        }
        X2U for (uint c = 0; c < NW; ++c) {
            uint k = lid + c * TG;
            if (k >= KS * 2 * BN) continue;
            uint kk = k / (2 * BN), r = k % (2 * BN), wsel = r / BN, n = r % BN;
            X2U for (uint h = 0; h < 2; ++h) {
                uint hw = rw[c] >> (16 * h);
                // Wb[kk][quad pair][slot][BN + 4], slots 4 (q % 2) .. +3 = quad q's masked codes.
                threadgroup WT *w = Wb + ((kk * 2 + wsel) * 8 + h * 4) * BNP + n;
                w[0] = WT(hw & 0x00f0u);
                w[BNP] = WT(hw & 0x000fu);
                w[2 * BNP] = WT(hw & 0x0f00u);
                w[3 * BNP] = WT(hw & 0xf000u);
            }
        }
        X2U for (uint c = 0; c < NS; ++c) {
            uint k = lid + c * TG;
            if (k < KS * BN) { Sc[k] = rs[c]; Bi[k] = rb[c]; }
        }
        X2U for (uint c = 0; c < NM; ++c) {
            uint k = lid + c * TG;
            if (k < KS * BM) Xsum[k] = rm[c];
        }
    };
    // First stage.
    uint fL = 0;
    while (fL < 32 && nkOf(fL) == 0) ++fL;
    if (fL < 32) fetch(fL, 0);
    float s0[MT][NT][2], s1[MT][NT][2], s2[MT][NT][2], s3[MT][NT][2], s4[MT][NT][2];
    simdgroup_float8x8 Z = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    for (uint L = 0; L < 32; ++L) {
        float acc[MT][NT][2];
        X2U for (uint i = 0; i < MT; ++i) X2U for (uint j = 0; j < NT; ++j) X2U for (uint e = 0; e < 2; ++e) acc[i][j][e] = 0;
        uint nk = nkOf(L);
        for (uint k0 = 0; k0 < nk; k0 += KS) {
            uint ks = min(KS, nk - k0);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            store();
            threadgroup_barrier(mem_flags::mem_threadgroup);
            // Next stage's loads, in flight during this stage's compute.
            uint nL = L, nk0 = k0 + KS;
            if (nk0 >= nk) {
                nk0 = 0;
                ++nL;
                while (nL < 32 && nkOf(nL) == 0) ++nL;
            }
            if (nL < 32) fetch(nL, nk0);
            for (uint kk = 0; kk < ks; ++kk) {
                float dot[MT][NT][2];
                X2U for (uint i = 0; i < MT; ++i) X2U for (uint j = 0; j < NT; ++j) X2U for (uint e = 0; e < 2; ++e) dot[i][j][e] = 0;
                {
                    // Two quads per fragment pair: A_top (slots 4..7 zeroed) gives quad 2 qp's
                    // chain then fma(+0, code, acc) four times (acc unchanged: codes are finite
                    // and acc is never -0 here); A_bot (slots 0..3 zeroed) gives +0 four times
                    // then quad 2 qp + 1's chain from +0.
                    X2U for (uint qp = 0; qp < 2; ++qp) {
                        simdgroup_float8x8 At[MT], Ab[MT], B[NT];
                        X2U for (uint i = 0; i < MT; ++i) {
                            simdgroup_float8x8 A;
                            simdgroup_load(A, Xa + ((kk * 2 + qp) * BM + tm + 8 * i) * 8, 8);
                            X2U for (uint e = 0; e < 2; ++e) {
                                float v = A.thread_elements()[e];
                                At[i].thread_elements()[e] = sn < 4 ? v : 0.0f;
                                Ab[i].thread_elements()[e] = sn < 4 ? 0.0f : v;
                            }
                        }
                        X2U for (uint j = 0; j < NT; ++j) {
                            if (is_same<WT, half>::value) {
                                // Masked codes are exact in half (at most 15 * 4096, 4 significant bits).
                                half2 c = *(threadgroup const half2 *)(Wb + ((kk * 2 + qp) * 8 + sm) * BNP + tn + 8 * j + sn);
                                B[j].thread_elements()[0] = float(c.x);
                                B[j].thread_elements()[1] = float(c.y);
                            } else simdgroup_load(B[j], (threadgroup const float *)Wb + (kk * 2 + qp) * 8 * BNP + tn + 8 * j, BNP);
                        }
                        X2U for (uint h = 0; h < 2; ++h) X2U for (uint i = 0; i < MT; ++i) X2U for (uint j = 0; j < NT; ++j) {
                            simdgroup_float8x8 D;
                            simdgroup_multiply_accumulate(D, h ? Ab[i] : At[i], B[j], Z);
                            X2U for (uint e = 0; e < 2; ++e) dot[i][j][e] = dot[i][j][e] + D.thread_elements()[e];
                        }
                    }
                }
                X2U for (uint j = 0; j < NT; ++j) X2U for (uint e = 0; e < 2; ++e) {
                    float scale = Sc[kk * BN + tn + 8 * j + sn + e], bias = Bi[kk * BN + tn + 8 * j + sn + e];
                    X2U for (uint i = 0; i < MT; ++i)
                        acc[i][j][e] = acc[i][j][e] + fma(scale, dot[i][j][e], Xsum[kk * BM + tm + 8 * i + sm] * bias);
                }
            }
        }
        X2U for (uint i = 0; i < MT; ++i) X2U for (uint j = 0; j < NT; ++j) X2U for (uint e = 0; e < 2; ++e)
            MK_X2_FOLD(L, acc[i][j][e], s0[i][j][e], s1[i][j][e], s2[i][j][e], s3[i][j][e], s4[i][j][e]);
    }
    X2U for (uint i = 0; i < MT; ++i) {
        uint t = m0 + tm + 8 * i + sm;
        if (t >= R) continue;
        X2U for (uint j = 0; j < NT; ++j) X2U for (uint e = 0; e < 2; ++e) {
            uint row = n0 + tn + 8 * j + sn + e;
            if (row < p.m) arena[p.y + (p.row0 + t) * p.ys + row] = s0[i][j][e];
        }
    }
}
#define MK_X2MM(MT, NT, WM, WN, KS, SDT, SN) \
kernel void mk_x2mm_q4_m##MT##_n##NT##_w##WM##x##WN##_k##KS##_##SN(device const uchar *w [[buffer(0)]], device const uchar *s [[buffer(1)]], \
        device const uchar *b [[buffer(2)]], device float *arena [[buffer(3)]], device const float *prep [[buffer(4)]], \
        constant MK &p [[buffer(8)]], uint tgid [[threadgroup_position_in_grid]], uint lid [[thread_position_in_threadgroup]], \
        uint simd [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) { \
    constexpr uint BM = 8 * MT * WM, BN = 8 * NT * WN; \
    threadgroup float Xa[KS * 2 * BM * 8], Wb[KS * 2 * 8 * (BN + 4)]; \
    threadgroup float Sc[KS * BN], Bi[KS * BN], Xsum[KS * BM]; \
    mk_x2mm_tile<MT, NT, WM, WN, KS, SDT>(w, s, b, prep, arena, p, tgid, simd, lane, lid, Xa, Wb, Sc, Bi, Xsum); \
}
// Scale and bias dtype at compile time (both the same; MK dtype codes 0 f32, 1 f16, 2 bf16).
MK_X2MM(2, 2, 2, 2, 4, 0, f32) MK_X2MM(2, 2, 2, 2, 4, 1, f16) MK_X2MM(2, 2, 2, 2, 4, 2, bf16)
// Knob x2_half: the unpacked codes staged as half (exact: integers <= 15 * 4096 with
// 4 significant bits), half the threadgroup traffic of the B operand.
#define MK_X2MMH(MT, NT, WM, WN, KS, SDT, SN) \
kernel void mk_x2mmh_q4_m##MT##_n##NT##_w##WM##x##WN##_k##KS##_##SN(device const uchar *w [[buffer(0)]], device const uchar *s [[buffer(1)]], \
        device const uchar *b [[buffer(2)]], device float *arena [[buffer(3)]], device const float *prep [[buffer(4)]], \
        constant MK &p [[buffer(8)]], uint tgid [[threadgroup_position_in_grid]], uint lid [[thread_position_in_threadgroup]], \
        uint simd [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) { \
    constexpr uint BM = 8 * MT * WM, BN = 8 * NT * WN; \
    threadgroup float Xa[KS * 2 * BM * 8]; \
    threadgroup half Wb[KS * 2 * 8 * (BN + 4)]; \
    threadgroup float Sc[KS * BN], Bi[KS * BN], Xsum[KS * BM]; \
    mk_x2mm_tile<MT, NT, WM, WN, KS, SDT, half>(w, s, b, prep, arena, p, tgid, simd, lane, lid, Xa, Wb, Sc, Bi, Xsum); \
}
MK_X2MMH(2, 2, 2, 2, 4, 0, f32) MK_X2MMH(2, 2, 2, 2, 4, 1, f16) MK_X2MMH(2, 2, 2, 2, 4, 2, bf16)

// ---- batched exact attention (knob x2_attention) ------------------------------------
// mk_attention (and mk_attention_gated) for many rows, as three launches. Per
// (row, query head) the reference runs 8 SIMD classes s; class s visits positions
// start + s, start + s + 8, ... <= position with an online softmax:
//   dot = simd_sum over lanes of (fma chain over e of q[lane + 32 e] * K[t][...]),
//     q = scale * Q (a product), the chain from +0;
//   nm = max(m, dot); f = exp(m - nm); es = exp(dot - nm); m = nm;
//   l = l * f + es; o[d] = o[d] * f + es * V[t][d];
// then merges the classes in order s = 0..7 (M = max of the m_s; f_s = l_s == 0 ?
// 0 : exp(m_s - M); total += l_s * f_s; acc[d] += o_s[d] * f_s; out = acc / total,
// gated: (acc / total) * act(gate)). The launches:
//   mk_x2_att_scores: every visible (row, head, position)'s dot, exactly as above;
//     a SIMD group loads 8 keys once for RB rows x G query heads of one KV head.
//   mk_x2_att_soft: per (row, head, s), m, f, es and l in step order; the running
//     max is a prefix max (exact in any association; Metal's max ignores a NaN
//     operand in either position, and a signed-zero difference in m cannot change
//     exp(m - nm) or exp(dot - nm)), computed 32 steps at a time by a shuffle scan
//     as in mk_dec_chain_block. Writes es over the dot, f beside it, and (m, l).
//   mk_x2_att_chain: per (RB rows, KV head, 32 dims), SIMD group s applies
//     o = o * f + es * V for its class in step order for every row and query head
//     (V loaded once for all of them), then the merge in class order.
// Layout: S/F word (r * heads + h) * p.every + (t - start_r), r dispatch-relative;
// ML word ((r * heads + h) * 8 + s) * 2 + {0: m, 1: l}.
constant uint X2_ATT_GATED = 8u;
inline uint mk_x2_start(uint position, uint window) {
    return window && position + 1 > window ? position + 1 - window : 0;
}
// Grid: SIMD group (block, kvh, chunk), chunk of 8 positions from the block's
// first row's start; p.m = chunks per block.
template <uint E, uint G, uint RB>
inline void mk_x2_att_scores_impl(device const float *arena, device const float *K, device float *S, constant MK &p,
                                  uint simd, uint lane) {
    uint chunk = simd % p.m, kvh = (simd / p.m) % p.kv_heads, blk = simd / (p.m * p.kv_heads);
    uint r0 = blk * RB;
    if (r0 >= p.rows) return;
    uint first = mk_x2_start(p.pos + p.row0 + r0, p.window);
    uint last = p.pos + p.row0 + min(r0 + RB, p.rows) - 1;
    uint t0 = first + chunk * 8;
    if (t0 > last) return;
    float k[8][E];
    X2U for (uint c = 0; c < 8; ++c) {
        uint t = min(t0 + c, last);
        ulong base = ulong(t % p.capacity) * (p.kv_heads * p.dim) + kvh * p.dim;
        X2U for (uint e = 0; e < E; ++e) k[c][e] = lane + 32 * e < p.dim ? K[base + lane + 32 * e] : 0;
    }
    X2U for (uint i = 0; i < RB; ++i) {
        uint r = r0 + i;
        if (r >= p.rows) break;
        uint position = p.pos + p.row0 + r;
        uint start = mk_x2_start(position, p.window);
        float q[G][E];
        X2U for (uint g = 0; g < G; ++g) {
            device const float *qp = arena + p.x + (p.row0 + r) * p.xs + (kvh * G + g) * p.dim;
            X2U for (uint e = 0; e < E; ++e) q[g][e] = lane + 32 * e < p.dim ? p.scale * qp[lane + 32 * e] : 0;
        }
        X2U for (uint c = 0; c < 8; ++c) {
            uint t = t0 + c;
            if (t < start || t > position) continue;
            X2U for (uint g = 0; g < G; ++g) {
                float dot = 0;
                X2U for (uint e = 0; e < E; ++e) if (lane + 32 * e < p.dim) dot += q[g][e] * k[c][e];
                dot = simd_sum(dot);
                if (lane == 0) S[(ulong(r) * p.heads + kvh * G + g) * p.every + (t - start)] = dot;
            }
        }
    }
}
#define MK_X2_SCORES(E, G, RB) \
kernel void mk_x2_att_scores_e##E##_g##G##_r##RB(device const float *arena [[buffer(0)]], device const float *K [[buffer(1)]], \
        device float *S [[buffer(2)]], constant MK &p [[buffer(8)]], uint i [[thread_position_in_grid]], \
        uint lane [[thread_index_in_simdgroup]]) { \
    mk_x2_att_scores_impl<E, G, RB>(arena, K, S, p, i / 32, lane); \
}
// Every head-dim class E = ceil(dim / 32) and group size G; 4 rows per SIMD group.
MK_X2_SCORES(1, 1, 4) MK_X2_SCORES(2, 1, 4) MK_X2_SCORES(4, 1, 4) MK_X2_SCORES(8, 1, 4)
MK_X2_SCORES(1, 2, 4) MK_X2_SCORES(2, 2, 4) MK_X2_SCORES(4, 2, 4) MK_X2_SCORES(8, 2, 4)
MK_X2_SCORES(1, 4, 4) MK_X2_SCORES(2, 4, 4) MK_X2_SCORES(4, 4, 4) MK_X2_SCORES(8, 4, 4)
MK_X2_SCORES(1, 8, 4) MK_X2_SCORES(2, 8, 4) MK_X2_SCORES(4, 8, 4) MK_X2_SCORES(8, 8, 4)

// Grid: SIMD group (r, h, class quarter c4); it runs the four classes s = 4 c4 .. 4 c4 + 3
// side by side (independent chains, each in its own step order).
kernel void mk_x2_att_soft(device float *S [[buffer(0)]], device float *F [[buffer(1)]], device float *ML [[buffer(2)]],
                           constant MK &p [[buffer(8)]], uint i [[thread_position_in_grid]],
                           uint lane [[thread_index_in_simdgroup]]) {
    uint sg = i / 32;
    uint c4 = sg % 2, h = (sg / 2) % p.heads, r = sg / (2 * p.heads);
    if (r >= p.rows) return;
    uint position = p.pos + p.row0 + r;
    uint start = mk_x2_start(position, p.window);
    uint steps[4], maxsteps = 0;
    X2U for (uint c = 0; c < 4; ++c) {
        uint s = 4 * c4 + c;
        steps[c] = position >= start + s ? (position - start - s) / 8 + 1 : 0;
        maxsteps = max(maxsteps, steps[c]);
    }
    ulong base = (ulong(r) * p.heads + h) * p.every + 4 * c4;
    float m[4], l[4];
    X2U for (uint c = 0; c < 4; ++c) { m[c] = -INFINITY; l[c] = 0; }
    for (uint b0 = 0; b0 < maxsteps; b0 += 32) {
        uint j = b0 + lane;
        float f[4], es[4], nm[4];
        uint cnt[4];
        X2U for (uint c = 0; c < 4; ++c) {
            cnt[c] = steps[c] > b0 ? min(32u, steps[c] - b0) : 0u;
            float dot = lane < cnt[c] ? S[base + c + 8 * j] : -INFINITY;
            nm[c] = max(m[c], dot);
            for (uint off = 1; off < 32; off <<= 1) {
                float up = simd_shuffle_up(nm[c], off);
                if (lane >= off) nm[c] = max(nm[c], up);
            }
            float prev = simd_shuffle_up(nm[c], 1);
            if (lane == 0) prev = m[c];
            f[c] = metal::precise::exp(prev - nm[c]);
            es[c] = metal::precise::exp(dot - nm[c]);
            if (lane < cnt[c]) { S[base + c + 8 * j] = es[c]; F[base + c + 8 * j] = f[c]; }
        }
        X2U for (uint jj = 0; jj < 32; ++jj) X2U for (uint c = 0; c < 4; ++c) {
            float fj = simd_shuffle(f[c], jj), ej = simd_shuffle(es[c], jj);
            if (jj < cnt[c]) l[c] = l[c] * fj + ej;
        }
        X2U for (uint c = 0; c < 4; ++c) if (cnt[c]) m[c] = simd_shuffle(nm[c], cnt[c] - 1);
    }
    if (lane < 4) {
        device float *o = ML + ((ulong(r) * p.heads + h) * 8 + 4 * c4 + lane) * 2;
        float mv = lane == 0 ? m[0] : lane == 1 ? m[1] : lane == 2 ? m[2] : m[3];
        float lv = lane == 0 ? l[0] : lane == 1 ? l[1] : lane == 2 ? l[2] : l[3];
        o[0] = mv;
        o[1] = lv;
    }
}

// Grid: threadgroup (block, kvh, part) of 8 SIMD groups; SIMD group s, lane = dim
// part * 32 + lane. The rows of a block visit the same class-s positions: one row
// (`own`: its own window start), or RB rows of a full-attention layer (`shared`:
// start 0), so V is loaded once for all of them (and all G query heads).
template <uint G, uint RB>
inline void mk_x2_att_chain_impl(device float *arena, device const float *V, device const float *S,
                                 device const float *F, device const float *ML, constant MK &p,
                                 uint tgid, uint s, uint lane, threadgroup float *outs) {
    uint parts = (p.dim + 31) / 32;
    uint part = tgid % parts, kvh = (tgid / parts) % p.kv_heads, blk = tgid / (parts * p.kv_heads);
    uint r0 = blk * RB;
    if (r0 >= p.rows) return;
    uint d = part * 32 + lane;
    bool live = d < p.dim;
    uint dd = min(d, p.dim - 1);
    ulong rowlen = p.kv_heads * p.dim, kvoff = kvh * p.dim + dd;
    uint nr = min(RB, p.rows - r0);
    uint start[RB], steps[RB], maxsteps = 0;
    X2U for (uint i = 0; i < RB; ++i) {
        uint position = p.pos + p.row0 + r0 + min(i, nr - 1);
        start[i] = mk_x2_start(position, p.window);
        steps[i] = i < nr && position >= start[i] + s ? (position - start[i] - s) / 8 + 1 : 0;
        maxsteps = max(maxsteps, steps[i]);
    }
    float o[RB][G];
    X2U for (uint i = 0; i < RB; ++i) X2U for (uint g = 0; g < G; ++g) o[i][g] = 0;
    for (uint b0 = 0; b0 < maxsteps; b0 += 32) {
        float fb[RB][G], eb[RB][G];
        X2U for (uint i = 0; i < RB; ++i) X2U for (uint g = 0; g < G; ++g) {
            ulong base = (ulong(r0 + min(i, nr - 1)) * p.heads + kvh * G + g) * p.every + s + 8 * (b0 + lane);
            bool ok = b0 + lane < steps[i];
            fb[i][g] = ok ? F[base] : 0.0f;
            eb[i][g] = ok ? S[base] : 0.0f;
        }
        // The block's 32 values of this lane's dim, all issued before the serial steps
        // (clamped, unconditional loads; only steps < steps[i] are applied).
        float vv[32];
        {
            uint t = start[0] + s + 8 * b0;
            uint slot = t % p.capacity;
            X2U for (uint jj = 0; jj < 32; ++jj) {
                vv[jj] = V[ulong(slot) * rowlen + kvoff];
                slot += 8;
                if (slot >= p.capacity) slot -= p.capacity;
            }
        }
        X2U for (uint jj = 0; jj < 32; ++jj) {
            uint j = b0 + jj;
            X2U for (uint i = 0; i < RB; ++i) {
                bool on = j < steps[i];
                X2U for (uint g = 0; g < G; ++g) {
                    float fj = simd_shuffle(fb[i][g], jj), ej = simd_shuffle(eb[i][g], jj);
                    if (on) o[i][g] = o[i][g] * fj + ej * vv[jj];
                }
            }
        }
    }
    X2U for (uint i = 0; i < RB; ++i) X2U for (uint g = 0; g < G; ++g) outs[((s * RB + i) * G + g) * 32 + lane] = o[i][g];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // Merge: SIMD group s merges the (row, head) pairs s, s + 8, ... of the block.
    for (uint pr = s; pr < nr * G; pr += 8) {
        uint i = pr / G, g = pr % G, h = kvh * G + g, r = r0 + i;
        device const float *ml = ML + (ulong(r) * p.heads + h) * 16;
        float M = ml[0];
        for (uint c = 1; c < 8; ++c) M = max(M, ml[2 * c]);
        float total = 0;
        float acc = 0;
        for (uint c = 0; c < 8; ++c) {
            float f = ml[2 * c + 1] == 0 ? 0.0f : metal::precise::exp(ml[2 * c] - M);
            total += ml[2 * c + 1] * f;
            acc += outs[((c * RB + i) * G + g) * 32 + lane] * f;
        }
        if (!live) continue;
        device float *out = arena + p.y + (p.row0 + r) * p.ys + h * p.dim;
        if (p.flags & X2_ATT_GATED) {
            float gt = mk_act(arena[p.z + (p.row0 + r) * p.zs + h], p.act_b);
            float a = acc / total;
            out[d] = a * gt;
        } else {
            out[d] = acc / total;
        }
    }
}
#define MK_X2_CHAIN(G, RB, SN) \
kernel void mk_x2_att_chain_g##G##_r##RB##_##SN(device float *arena [[buffer(0)]], device const float *V [[buffer(1)]], \
        device const float *S [[buffer(2)]], device const float *F [[buffer(3)]], device const float *ML [[buffer(4)]], \
        constant MK &p [[buffer(8)]], uint tgid [[threadgroup_position_in_grid]], \
        uint s [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) { \
    threadgroup float outs[8 * RB * G * 32]; \
    mk_x2_att_chain_impl<G, RB>(arena, V, S, F, ML, p, tgid, s, lane, outs); \
}
// Per group size: one row per threadgroup (rows with their own window start), or
// 16 / G rows sharing start 0 (full attention).
MK_X2_CHAIN(1, 1, own) MK_X2_CHAIN(2, 1, own) MK_X2_CHAIN(4, 1, own) MK_X2_CHAIN(8, 1, own)
MK_X2_CHAIN(1, 16, shared) MK_X2_CHAIN(2, 8, shared) MK_X2_CHAIN(4, 4, shared) MK_X2_CHAIN(8, 2, shared)
