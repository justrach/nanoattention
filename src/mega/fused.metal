// Megakernel fused variants. Each is selected by a knob and must reproduce,
// for every token row, the reference kernels' expression trees exactly
// (kernels.metal): same lane ownership, same accumulation order, same
// formulas; only launches, stores of dead intermediates and loads are removed.
// Admission (tools/nano_qd.py) checks bit identity against the reference.

// One matrix of a multi-matrix launch: weight/scale/bias byte offsets in the
// single bound weight buffer, output rows, output word offset and stride.
struct MKMat {
    ulong w, s, b;
    uint m, y, ys, sdtype, bdtype, pad;
};
// Host mirror: ParamsX in src/mega/gpu.zig (160 + 4 * 48 + 16 bytes).
struct MKX {
    MK p;
    MKMat mat[4];
    uint count, act_a, act_b, kind;
};

// The reference segment load of mk_q4_tile, for one token row.
inline float mk_q4_load(device const float *x, thread float *xv) {
    float xsum = 0;
    for (uint j = 0; j < 16; j += 4) {
        float v0 = x[j], v1 = x[j + 1], v2 = x[j + 2], v3 = x[j + 3];
        xsum += v0 + v1 + v2 + v3;
        xv[j] = v0;
        xv[j + 1] = v1 / 16.0f;
        xv[j + 2] = v2 / 256.0f;
        xv[j + 3] = v3 / 4096.0f;
    }
    return xsum;
}
// The reference per-segment row contribution of mk_q4_tile.
inline float mk_q4_dot(thread const float *xv, uint c0, uint c1) {
    float dot = 0;
    for (uint j = 0; j < 16; j += 4) {
        uint codes = j < 8 ? c0 : c1;
        uint hw = (j & 7u) < 4 ? codes : codes >> 16;
        dot += (xv[j] * float(hw & 0x000fu) +
                xv[j + 1] * float(hw & 0x00f0u) +
                xv[j + 2] * float(hw & 0x0f00u) +
                xv[j + 3] * float(hw & 0xf000u));
    }
    return dot;
}

// ---- multi-projection: up to 4 linears sharing one input --------------------
// SIMD tiles enumerate matrix 0's tiles, then matrix 1's, ... Each matrix keeps
// mk_q4_tile's per-row arithmetic; this is launch fusion only.
template <uint R, uint TT>
inline void mk_multi_tile(device const uchar *wb, device float *arena, constant MKX &x, uint simd, uint lane) {
    constant MK &p = x.p;
    uint tiles_t = (p.rows + TT - 1) / TT;
    uint which = 0, local = simd;
    for (; which < x.count; ++which) {
        uint tiles = ((x.mat[which].m + R - 1) / R) * tiles_t;
        if (local < tiles) break;
        local -= tiles;
    }
    if (which >= x.count) return;
    constant MKMat &mat = x.mat[which];
    uint first = (local / tiles_t) * R, t0 = (local % tiles_t) * TT;
    uint row_words = p.n / 8, row_groups = p.n / p.group, seg_group = p.group / 16;
    float acc[TT][R];
    for (uint t = 0; t < TT; ++t) for (uint r = 0; r < R; ++r) acc[t][r] = 0;
    for (uint seg = lane; seg < p.n / 16; seg += 32) {
        float xv[TT][16], xsum[TT];
        for (uint t = 0; t < TT; ++t) {
            xsum[t] = 0;
            if (t0 + t < p.rows) xsum[t] = mk_q4_load(arena + p.x + (p.row0 + t0 + t) * p.xs + seg * 16, xv[t]);
        }
        for (uint r = 0; r < R; ++r) {
            uint row = first + r;
            if (row >= mat.m) continue;
            ulong base = ulong(row) * row_words + seg * 2;
            uint c0 = mk_word(wb, mat.w, base), c1 = mk_word(wb, mat.w, base + 1);
            uint group = row * row_groups + seg / seg_group;
            float scale = mk_load(wb, mat.s, mat.sdtype, group), bias = mk_load(wb, mat.b, mat.bdtype, group);
            for (uint t = 0; t < TT; ++t) {
                if (t0 + t >= p.rows) continue;
                float dot = mk_q4_dot(xv[t], c0, c1);
                acc[t][r] += scale * dot + xsum[t] * bias;
            }
        }
    }
    for (uint t = 0; t < TT; ++t) {
        if (t0 + t >= p.rows) continue;
        device float *out = arena + mat.y + (p.row0 + t0 + t) * mat.ys;
        for (uint r = 0; r < R; ++r) {
            float sum = simd_sum(acc[t][r]);
            if (lane == 0 && first + r < mat.m) out[first + r] = sum;
        }
    }
}
#define MK_MULTI(R, TT) \
kernel void mk_multi_q4_r##R##_t##TT(device const uchar *w [[buffer(0)]], device float *arena [[buffer(1)]], \
        constant MKX &x [[buffer(8)]], uint i [[thread_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) { \
    mk_multi_tile<R, TT>(w, arena, x, i / 32, lane); \
}
MK_MULTI(1, 1) MK_MULTI(2, 1) MK_MULTI(4, 1) MK_MULTI(8, 1)
MK_MULTI(1, 2) MK_MULTI(2, 2) MK_MULTI(4, 2) MK_MULTI(8, 2)
MK_MULTI(1, 4) MK_MULTI(2, 4) MK_MULTI(4, 4) MK_MULTI(8, 4)

// ---- gated pair: out = act_a(W0 x) (+|*) act_b(W1 x), intermediates never stored --
// Both matrices have the same shape; each row's two dot products are the
// reference ones, combined by mk_binary's formula.
template <uint R, uint TT>
inline void mk_glu_tile(device const uchar *wb, device float *arena, constant MKX &x, uint simd, uint lane) {
    constant MK &p = x.p;
    constant MKMat &m0 = x.mat[0];
    constant MKMat &m1 = x.mat[1];
    uint tiles_t = (p.rows + TT - 1) / TT;
    uint first = (simd / tiles_t) * R, t0 = (simd % tiles_t) * TT;
    if (first >= m0.m) return;
    uint row_words = p.n / 8, row_groups = p.n / p.group, seg_group = p.group / 16;
    float a0[TT][R], a1[TT][R];
    for (uint t = 0; t < TT; ++t) for (uint r = 0; r < R; ++r) { a0[t][r] = 0; a1[t][r] = 0; }
    for (uint seg = lane; seg < p.n / 16; seg += 32) {
        float xv[TT][16], xsum[TT];
        for (uint t = 0; t < TT; ++t) {
            xsum[t] = 0;
            if (t0 + t < p.rows) xsum[t] = mk_q4_load(arena + p.x + (p.row0 + t0 + t) * p.xs + seg * 16, xv[t]);
        }
        for (uint r = 0; r < R; ++r) {
            uint row = first + r;
            if (row >= m0.m) continue;
            ulong base = ulong(row) * row_words + seg * 2;
            uint g0 = mk_word(wb, m0.w, base), g1 = mk_word(wb, m0.w, base + 1);
            uint u0 = mk_word(wb, m1.w, base), u1 = mk_word(wb, m1.w, base + 1);
            uint group = row * row_groups + seg / seg_group;
            float s0 = mk_load(wb, m0.s, m0.sdtype, group), b0 = mk_load(wb, m0.b, m0.bdtype, group);
            float s1 = mk_load(wb, m1.s, m1.sdtype, group), b1 = mk_load(wb, m1.b, m1.bdtype, group);
            for (uint t = 0; t < TT; ++t) {
                if (t0 + t >= p.rows) continue;
                float d0 = mk_q4_dot(xv[t], g0, g1);
                a0[t][r] += s0 * d0 + xsum[t] * b0;
                float d1 = mk_q4_dot(xv[t], u0, u1);
                a1[t][r] += s1 * d1 + xsum[t] * b1;
            }
        }
    }
    for (uint t = 0; t < TT; ++t) {
        if (t0 + t >= p.rows) continue;
        device float *out = arena + p.y + (p.row0 + t0 + t) * p.ys;
        for (uint r = 0; r < R; ++r) {
            float g = simd_sum(a0[t][r]);
            float u = simd_sum(a1[t][r]);
            float a = mk_act(g, x.act_a), b = mk_act(u, x.act_b);
            if (lane == 0 && first + r < m0.m) out[first + r] = x.kind == 0 ? a + b : a * b;
        }
    }
}
#define MK_GLU(R, TT) \
kernel void mk_glu_q4_r##R##_t##TT(device const uchar *w [[buffer(0)]], device float *arena [[buffer(1)]], \
        constant MKX &x [[buffer(8)]], uint i [[thread_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) { \
    mk_glu_tile<R, TT>(w, arena, x, i / 32, lane); \
}
MK_GLU(1, 1) MK_GLU(2, 1) MK_GLU(4, 1) MK_GLU(8, 1)
MK_GLU(1, 2) MK_GLU(2, 2) MK_GLU(4, 2) MK_GLU(8, 2)
MK_GLU(1, 4) MK_GLU(2, 4) MK_GLU(4, 4) MK_GLU(8, 4)

// ---- sandwich residual: r = r (+) norm(y) * w0; out = norm(r) * w1 --------------
// One threadgroup per row with mk_rmsnorm's size; each thread owns the same
// words in all three passes, so both reductions follow mk_inv_rms's order.
// p.x = y, p.y = r (updated in place), p.z = out; p.w / p.s = byte offsets of
// the two norm weights in buffer 0, p.dtype / p.sdtype their dtypes;
// p.kind 0: r + t, 1: t + r (the reference add's operand order).
inline float mk_reduce_inv(float acc, uint n, float eps, uint simd, uint lane, threadgroup float *partial) {
    acc = simd_sum(acc);
    if (simd == 0) partial[lane] = 0;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0) partial[simd] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd == 0) {
        acc = simd_sum(partial[lane]);
        if (lane == 0) partial[32] = metal::precise::rsqrt(acc / float(n) + eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float inv = partial[32];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return inv;
}
kernel void mk_norm_add_norm(device const uchar *w [[buffer(0)]], device float *arena [[buffer(1)]],
                             constant MK &p [[buffer(8)]], uint tgid [[threadgroup_position_in_grid]],
                             uint lid [[thread_position_in_threadgroup]], uint tg [[threads_per_threadgroup]],
                             uint simd [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float partial[33];
    if (tgid >= p.rows) return;
    uint row = p.row0 + tgid;
    device const float *y = arena + p.x + row * p.xs;
    device float *res = arena + p.y + row * p.ys;
    device float *out = arena + p.z + row * p.zs;
    float acc = 0;
    for (uint base = lid * 4; base < p.n; base += tg * 4)
        for (uint j = 0; j < 4; ++j) if (base + j < p.n) { float v = y[base + j]; acc += v * v; }
    float inv = mk_reduce_inv(acc, p.n, p.eps, simd, lane, partial);
    acc = 0;
    for (uint base = lid * 4; base < p.n; base += tg * 4)
        for (uint j = 0; j < 4; ++j) if (base + j < p.n) {
            float v = y[base + j] * inv;
            float t = mk_load(w, p.w, p.dtype, base + j) * v;
            float r = p.kind == 0 ? res[base + j] + t : t + res[base + j];
            res[base + j] = r;
            acc += r * r;
        }
    float inv2 = mk_reduce_inv(acc, p.n, p.eps, simd, lane, partial);
    for (uint base = lid * 4; base < p.n; base += tg * 4)
        for (uint j = 0; j < 4; ++j) if (base + j < p.n) {
            float v = res[base + j] * inv2;
            out[base + j] = mk_load(w, p.s, p.sdtype, base + j) * v;
        }
}

// ---- gated short conv: u = a*b; ring <- u; v = conv(ring); out = c * v ---------
// One thread per channel walks the batch rows in order, so its ring writes
// precede its own reads (same thread, same channel). p.x / p.y = the two
// factors of u, p.z = the other factor of out, p.m = out word offset, p.every =
// out stride. p.act_a = act(a) | act(b) << 2; p.act_b = left act | right act << 2;
// p.kind bit 0: the conv output is the left operand of out.
kernel void mk_gated_conv(device float *ring [[buffer(0)]], device const uchar *w [[buffer(1)]],
                          device float *arena [[buffer(2)]], constant MK &p [[buffer(8)]],
                          uint c [[thread_position_in_grid]]) {
    if (c >= p.n) return;
    for (uint r = 0; r < p.rows; ++r) {
        uint row = p.row0 + r;
        long position = long(p.pos) + row;
        float a = mk_act(arena[p.x + row * p.xs + c], p.act_a & 3u);
        float b = mk_act(arena[p.y + row * p.ys + c], (p.act_a >> 2) & 3u);
        ring[ulong((p.pos + row) % p.capacity) * p.n + c] = a * b;
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
}

// ---- attention with a per-head gate epilogue -----------------------------------
// mk_attention's arithmetic, then out = att * act(gate[head]) (mk_binary with
// b_every = dim). p.z = gate word offset, p.zs = its row stride, p.act_b its act.
kernel void mk_attention_gated(device float *arena [[buffer(0)]], device const float *K [[buffer(1)]],
                               device const float *V [[buffer(2)]], constant MK &p [[buffer(8)]],
                               uint tgid [[threadgroup_position_in_grid]], uint simd [[simdgroup_index_in_threadgroup]],
                               uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float maxes[MK_ATT_SIMDS], sums[MK_ATT_SIMDS], outs[MK_ATT_SIMDS][256];
    uint r = tgid / p.heads, h = tgid % p.heads;
    if (r >= p.rows) return;
    uint row = p.row0 + r;
    uint position = p.pos + row;
    uint kvh = h / (p.heads / p.kv_heads);
    uint start = p.window && position + 1 > p.window ? position + 1 - p.window : 0;
    uint E = (p.dim + 31) / 32;
    float q[8], o[8];
    device const float *qp = arena + p.x + row * p.xs + h * p.dim;
    for (uint e = 0; e < 8; ++e) {
        uint d = lane + 32 * e;
        q[e] = (e < E && d < p.dim) ? p.scale * qp[d] : 0;
        o[e] = 0;
    }
    float m = -INFINITY, l = 0;
    for (uint t = start + simd; t <= position; t += MK_ATT_SIMDS) {
        ulong base = ulong(t % p.capacity) * (p.kv_heads * p.dim) + kvh * p.dim;
        float dot = 0;
        for (uint e = 0; e < E; ++e) { uint d = lane + 32 * e; if (d < p.dim) dot += q[e] * K[base + d]; }
        dot = simd_sum(dot);
        float nm = max(m, dot);
        float f = metal::precise::exp(m - nm), es = metal::precise::exp(dot - nm);
        m = nm;
        l = l * f + es;
        for (uint e = 0; e < E; ++e) { uint d = lane + 32 * e; if (d < p.dim) o[e] = o[e] * f + es * V[base + d]; }
    }
    if (lane == 0) { maxes[simd] = m; sums[simd] = l; }
    for (uint e = 0; e < E; ++e) { uint d = lane + 32 * e; if (d < p.dim) outs[simd][d] = o[e]; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd != 0) return;
    float M = maxes[0];
    for (uint s = 1; s < MK_ATT_SIMDS; ++s) M = max(M, maxes[s]);
    float total = 0;
    float acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    for (uint s = 0; s < MK_ATT_SIMDS; ++s) {
        float f = sums[s] == 0 ? 0.0f : metal::precise::exp(maxes[s] - M);
        total += sums[s] * f;
        for (uint e = 0; e < E; ++e) { uint d = lane + 32 * e; if (d < p.dim) acc[e] += outs[s][d] * f; }
    }
    float g = mk_act(arena[p.z + row * p.zs + h], p.act_b);
    device float *out = arena + p.y + row * p.ys + h * p.dim;
    for (uint e = 0; e < E; ++e) {
        uint d = lane + 32 * e;
        if (d < p.dim) { float a = acc[e] / total; out[d] = a * g; }
    }
}
