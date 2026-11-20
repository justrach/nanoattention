// ---- prefill objective (prefill search): matrix-matrix 4-bit linear ------------------
// mk_q4_tile's per-output arithmetic, reorganised for many token rows. Lane L of
// the SIMD group that owns output (row, token) still owns 16-input segments L,
// L+32, ... in ascending order; per segment it adds `scale * dot + xsum * bias`
// with the reference dot and xsum expressions; one simd_sum at the end. What
// changes is only who computes what, and when:
//  - mk_q4_prep computes each (token, segment)'s xsum and prescaled inputs once
//    (the reference recomputes them in every SIMD group) and stores them;
//  - a SIMD group unpacks a segment's codes for R rows once and applies them to
//    TT token rows, loading each token's prepared segment once for all R rows.
// SIMD groups of one threadgroup share a token tile (consecutive row tiles), so
// the prepared inputs they read are the same cache lines.

// Prepared input layout, per token row r of the dispatch (batch-relative):
// float4 index ((r * ksteps + k) * 4 + q) * 32 + L holds inputs 4q..4q+3 of
// segment 32k + L (prescaled as in the reference); xsum of segment s of row r is
// word p.z + r * segs + s. ksteps = ceil(segs / 32); segs = n / 16.
kernel void mk_q4_prep(device const float *arena [[buffer(0)]], device float *prep [[buffer(1)]],
                       constant MK &p [[buffer(8)]], uint i [[thread_position_in_grid]]) {
    uint segs = p.n / 16, ksteps = (segs + 31) / 32;
    uint r = i / segs, seg = i % segs;
    if (r >= p.rows) return;
    device const float *x = arena + p.x + (p.row0 + r) * p.xs + seg * 16;
    device float4 *xp = (device float4 *)prep + ((ulong(r) * ksteps + seg / 32) * 4) * 32 + seg % 32;
    float xsum = 0;
    for (uint j = 0; j < 16; j += 4) {
        float v0 = x[j], v1 = x[j + 1], v2 = x[j + 2], v3 = x[j + 3];
        xsum += v0 + v1 + v2 + v3;
        xp[(j / 4) * 32] = float4(v0, v1 / 16.0f, v2 / 256.0f, v3 / 4096.0f);
    }
    prep[p.z + ulong(r) * segs + seg] = xsum;
}

// Codes of one 16-input segment of one row, masked in place as the reference
// does (float(hw & 0x00f0u) etc.), stored as CT. half holds every masked code
// exactly (at most 15 * 4096 = 61440, 4 significant bits).
template <typename CT>
inline void mk_mm_unpack(uint c0, uint c1, thread CT *w) {
    for (uint j = 0; j < 16; j += 4) {
        uint codes = j < 8 ? c0 : c1;
        uint hw = (j & 7u) < 4 ? codes : codes >> 16;
        w[j] = CT(hw & 0x000fu);
        w[j + 1] = CT(hw & 0x00f0u);
        w[j + 2] = CT(hw & 0x0f00u);
        w[j + 3] = CT(hw & 0xf000u);
    }
}

// The reference dot of mk_q4_tile with the masked codes already converted:
// the same products, the same left-to-right sums, the same contraction shape.
#define MK_MM_DOT(dot, xv, w)                                                   \
    {                                                                           \
        dot = 0;                                                                \
        for (uint j = 0; j < 16; j += 4) {                                      \
            dot += (xv[j] * float(w[j]) + xv[j + 1] * float(w[j + 1]) +         \
                    xv[j + 2] * float(w[j + 2]) + xv[j + 3] * float(w[j + 3])); \
        }                                                                       \
    }

template <uint R, uint TT, typename CT>
inline void mk_mm_tile(device const uchar *wb, device const uchar *sb, device const uchar *bb,
                       device const float *prep, device float *arena, constant MK &p, uint simd, uint lane) {
    uint tiles_r = (p.m + R - 1) / R;
    uint first = (simd % tiles_r) * R, t0 = (simd / tiles_r) * TT;
    if (t0 >= p.rows) return;
    uint segs = p.n / 16, ksteps = (segs + 31) / 32;
    uint row_words = p.n / 8, row_groups = p.n / p.group, seg_group = p.group / 16;
    device const float4 *xp = (device const float4 *)prep;
    float acc[TT][R];
    for (uint t = 0; t < TT; ++t) for (uint r = 0; r < R; ++r) acc[t][r] = 0;
    for (uint seg = lane, k = 0; seg < segs; seg += 32, ++k) {
        CT w[R][16];
        float scale[R], bias[R];
        for (uint r = 0; r < R; ++r) {
            uint row = min(first + r, p.m - 1);
            ulong base = ulong(row) * row_words + seg * 2;
            mk_mm_unpack<CT>(mk_word(wb, p.w, base), mk_word(wb, p.w, base + 1), w[r]);
            uint group = row * row_groups + seg / seg_group;
            scale[r] = mk_load(sb, p.s, p.sdtype, group);
            bias[r] = mk_load(bb, p.b, p.bdtype, group);
        }
        for (uint t = 0; t < TT; ++t) {
            if (t0 + t >= p.rows) break;
            device const float4 *xr = xp + ((ulong(t0 + t) * ksteps + k) * 4) * 32 + lane;
            float4 q0 = xr[0], q1 = xr[32], q2 = xr[64], q3 = xr[96];
            float xv[16] = {q0.x, q0.y, q0.z, q0.w, q1.x, q1.y, q1.z, q1.w,
                            q2.x, q2.y, q2.z, q2.w, q3.x, q3.y, q3.z, q3.w};
            float xsum = prep[p.z + ulong(t0 + t) * segs + seg];
            for (uint r = 0; r < R; ++r) {
                float dot;
                MK_MM_DOT(dot, xv, w[r]);
                acc[t][r] += scale[r] * dot + xsum * bias[r];
            }
        }
    }
    for (uint t = 0; t < TT; ++t) {
        if (t0 + t >= p.rows) break;
        device float *out = arena + p.y + (p.row0 + t0 + t) * p.ys;
        for (uint r = 0; r < R; ++r) {
            float sum = simd_sum(acc[t][r]);
            if (lane == 0 && first + r < p.m) out[first + r] = sum;
        }
    }
}
#define MK_MM(R, TT, CT, C) \
kernel void mk_mm_q4_r##R##_t##TT##_##C(device const uchar *w [[buffer(0)]], device const uchar *s [[buffer(1)]], \
        device const uchar *b [[buffer(2)]], device float *arena [[buffer(3)]], device const float *prep [[buffer(4)]], \
        constant MK &p [[buffer(8)]], uint i [[thread_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) { \
    mk_mm_tile<R, TT, CT>(w, s, b, prep, arena, p, i / 32, lane); \
}
// The prefill-mm elite's tile (2 rows x 8 tokens, half codes); the climb's other
// tiles were removed (benchmarks/lfm3-prefill-search.md).
MK_MM(2, 8, half, h)

// Gated pair (mk_glu_tile's arithmetic): both matrices' rows share the prepared
// token segment; the epilogue is mk_binary's formula.
template <uint R, uint TT, typename CT>
inline void mk_mm_glu_tile(device const uchar *wb, device const float *prep, device float *arena, constant MKX &x,
                           uint simd, uint lane) {
    constant MK &p = x.p;
    constant MKMat &m0 = x.mat[0];
    constant MKMat &m1 = x.mat[1];
    uint tiles_r = (m0.m + R - 1) / R;
    uint first = (simd % tiles_r) * R, t0 = (simd / tiles_r) * TT;
    if (t0 >= p.rows) return;
    uint segs = p.n / 16, ksteps = (segs + 31) / 32;
    uint row_words = p.n / 8, row_groups = p.n / p.group, seg_group = p.group / 16;
    device const float4 *xp = (device const float4 *)prep;
    float a0[TT][R], a1[TT][R];
    for (uint t = 0; t < TT; ++t) for (uint r = 0; r < R; ++r) { a0[t][r] = 0; a1[t][r] = 0; }
    for (uint seg = lane, k = 0; seg < segs; seg += 32, ++k) {
        CT g[R][16], u[R][16];
        float s0[R], b0[R], s1[R], b1[R];
        for (uint r = 0; r < R; ++r) {
            uint row = min(first + r, m0.m - 1);
            ulong base = ulong(row) * row_words + seg * 2;
            mk_mm_unpack<CT>(mk_word(wb, m0.w, base), mk_word(wb, m0.w, base + 1), g[r]);
            mk_mm_unpack<CT>(mk_word(wb, m1.w, base), mk_word(wb, m1.w, base + 1), u[r]);
            uint group = row * row_groups + seg / seg_group;
            s0[r] = mk_load(wb, m0.s, m0.sdtype, group);
            b0[r] = mk_load(wb, m0.b, m0.bdtype, group);
            s1[r] = mk_load(wb, m1.s, m1.sdtype, group);
            b1[r] = mk_load(wb, m1.b, m1.bdtype, group);
        }
        for (uint t = 0; t < TT; ++t) {
            if (t0 + t >= p.rows) break;
            device const float4 *xr = xp + ((ulong(t0 + t) * ksteps + k) * 4) * 32 + lane;
            float4 q0 = xr[0], q1 = xr[32], q2 = xr[64], q3 = xr[96];
            float xv[16] = {q0.x, q0.y, q0.z, q0.w, q1.x, q1.y, q1.z, q1.w,
                            q2.x, q2.y, q2.z, q2.w, q3.x, q3.y, q3.z, q3.w};
            float xsum = prep[p.z + ulong(t0 + t) * segs + seg];
            for (uint r = 0; r < R; ++r) {
                float d0, d1;
                MK_MM_DOT(d0, xv, g[r]);
                a0[t][r] += s0[r] * d0 + xsum * b0[r];
                MK_MM_DOT(d1, xv, u[r]);
                a1[t][r] += s1[r] * d1 + xsum * b1[r];
            }
        }
    }
    for (uint t = 0; t < TT; ++t) {
        if (t0 + t >= p.rows) break;
        device float *out = arena + p.y + (p.row0 + t0 + t) * p.ys;
        for (uint r = 0; r < R; ++r) {
            float gs = simd_sum(a0[t][r]);
            float us = simd_sum(a1[t][r]);
            float a = mk_act(gs, x.act_a), b = mk_act(us, x.act_b);
            if (lane == 0 && first + r < m0.m) out[first + r] = x.kind == 0 ? a + b : a * b;
        }
    }
}
#define MK_MM_GLU(R, TT, CT, C) \
kernel void mk_mm_glu_q4_r##R##_t##TT##_##C(device const uchar *w [[buffer(0)]], device float *arena [[buffer(1)]], \
        device const float *prep [[buffer(2)]], constant MKX &x [[buffer(8)]], uint i [[thread_position_in_grid]], \
        uint lane [[thread_index_in_simdgroup]]) { \
    mk_mm_glu_tile<R, TT, CT>(w, prep, arena, x, i / 32, lane); \
}
MK_MM_GLU(2, 8, half, h)

// ---- prefill-mlx (exact): the gated short conv over many rows in two launches --------------
// mk_gated_conv walks a batch's rows serially per channel (one thread per
// channel) so each row's ring write precedes the later rows' reads. Here the
// first launch writes every row's u = act(a) * act(b) to the ring, the second
// convolves and gates every (row, channel) in parallel. Formulas, tap order
// and operand order are mk_gated_conv's; the ring holds the same u values when
// they are read (capacity = taps - 1 + batch, so no row of this batch
// overwrites a slot another row of it still reads). Parameters as mk_gated_conv.
kernel void mk_gated_conv_store(device float *ring [[buffer(0)]], device const float *arena [[buffer(1)]],
                                constant MK &p [[buffer(8)]], uint i [[thread_position_in_grid]]) {
    uint r = i / p.n, c = i % p.n;
    if (r >= p.rows) return;
    uint row = p.row0 + r;
    float a = mk_act(arena[p.x + row * p.xs + c], p.act_a & 3u);
    float b = mk_act(arena[p.y + row * p.ys + c], (p.act_a >> 2) & 3u);
    ring[ulong((p.pos + row) % p.capacity) * p.n + c] = a * b;
}
kernel void mk_gated_conv_rows(device const float *ring [[buffer(0)]], device const uchar *w [[buffer(1)]],
                               device float *arena [[buffer(2)]], constant MK &p [[buffer(8)]],
                               uint i [[thread_position_in_grid]]) {
    uint r = i / p.n, c = i % p.n;
    if (r >= p.rows) return;
    uint row = p.row0 + r;
    long position = long(p.pos) + row;
    float acc = 0;
    for (uint k = 0; k < p.taps; ++k) {
        long q = position - long(p.taps) + 1 + k;
        float u = q < 0 ? 0.0f : ring[ulong(q % p.capacity) * p.n + c];
        acc += mk_load(w, p.w, p.dtype, ulong(c) * p.taps + k) * u;
    }
    float other = arena[p.z + row * p.zs + c];
    float left = (p.kind & 1u) ? mk_act(acc, p.act_b & 3u) : mk_act(other, p.act_b & 3u);
    float right = (p.kind & 1u) ? mk_act(other, (p.act_b >> 2) & 3u) : mk_act(acc, (p.act_b >> 2) & 3u);
    arena[p.m + row * p.every + c] = left * right;
}
