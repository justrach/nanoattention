// ===== decode search: decode-objective variants (docs/QD.md). Each reproduces the
// reference kernels' per-row expression trees (kernels.metal) bit for bit;
// admission (tools/nano_qd.py) checks that. Single-row (decode) forwards only
// unless stated; the host falls back to the reference kernel otherwise.

// ---- split attention (knob decode_attention = 2) --------------------------------
// mk_attention for one token row, as two launches:
//   mk_dec_scores<E, G>: every visible position's q.k for the G query heads of a
//     KV head: lane L owns dims L + 32 e, the products accumulate in e order
//     (dot += q[e] * k[e], a fused multiply-add as in the reference), then one
//     simd_sum. Positions are spread over many SIMD groups; each SIMD loads a
//     key row once for all G heads.
//   mk_dec_chain_block (below): mk_attention's online softmax per (head, SIMD s)
//     over the stored scores, positions start + s, start + s + 8, ... in order,
//     then the merge in SIMD order. The per-dim recurrence is independent across
//     dims, so each threadgroup takes 32 dims (one per lane) and recomputes the
//     scalar chain (max, exp, l), which is the same arithmetic in every part.
// Any head dim up to 256 whose ceil(dim / 32) is 1, 2, 4 or 8; lanes past dim
// contribute exactly 0, as in mk_attention.
// Scores: S[h * p.every + (t - start)]. p.m = position groups per SIMD launch.
// p.flags bit MK_DEC_GATED: the chain applies mk_attention_gated's epilogue,
// out = att * act_b(gate[head]) (p.z / p.zs the gate view); else out = att.
constant uint MK_DEC_GATED = 8u;
constant uint MK_DEC_CHUNK = 8;
template <uint E, uint G>
inline void mk_dec_scores_impl(device const float *arena, device const float *K, device float *S, constant MK &p,
                               uint simd, uint lane) {
    uint kvh = simd / p.m, chunk = simd % p.m;
    if (kvh >= p.kv_heads) return;
    uint position = p.pos + p.row0;
    uint start = p.window && position + 1 > p.window ? position + 1 - p.window : 0;
    uint t0 = start + chunk * MK_DEC_CHUNK;
    if (t0 > position) return;
    float q[G][E];
    for (uint g = 0; g < G; ++g) {
        device const float *qp = arena + p.x + p.row0 * p.xs + (kvh * G + g) * p.dim;
        for (uint e = 0; e < E; ++e) q[g][e] = lane + 32 * e < p.dim ? p.scale * qp[lane + 32 * e] : 0;
    }
    float k[MK_DEC_CHUNK][E];
    for (uint c = 0; c < MK_DEC_CHUNK; ++c) {
        uint t = min(t0 + c, position);
        ulong base = ulong(t % p.capacity) * (p.kv_heads * p.dim) + kvh * p.dim;
        for (uint e = 0; e < E; ++e) k[c][e] = lane + 32 * e < p.dim ? K[base + lane + 32 * e] : 0;
    }
    for (uint c = 0; c < MK_DEC_CHUNK; ++c) {
        uint t = t0 + c;
        if (t > position) break;
        for (uint g = 0; g < G; ++g) {
            float dot = 0;
            for (uint e = 0; e < E; ++e) if (lane + 32 * e < p.dim) dot += q[g][e] * k[c][e];
            dot = simd_sum(dot);
            if (lane == 0) S[(kvh * G + g) * p.every + (t - start)] = dot;
        }
    }
}
#define MK_DEC_SCORES(E, G) \
kernel void mk_dec_scores_e##E##_g##G(device const float *arena [[buffer(0)]], device const float *K [[buffer(1)]], \
        device float *S [[buffer(2)]], constant MK &p [[buffer(8)]], uint i [[thread_position_in_grid]], \
        uint lane [[thread_index_in_simdgroup]]) { \
    mk_dec_scores_impl<E, G>(arena, K, S, p, i / 32, lane); \
}
MK_DEC_SCORES(1, 1) MK_DEC_SCORES(2, 1) MK_DEC_SCORES(4, 1) MK_DEC_SCORES(8, 1)
MK_DEC_SCORES(1, 2) MK_DEC_SCORES(2, 2) MK_DEC_SCORES(4, 2) MK_DEC_SCORES(8, 2)
MK_DEC_SCORES(1, 4) MK_DEC_SCORES(2, 4) MK_DEC_SCORES(4, 4) MK_DEC_SCORES(8, 4)
MK_DEC_SCORES(1, 8) MK_DEC_SCORES(2, 8) MK_DEC_SCORES(4, 8) MK_DEC_SCORES(8, 8)

// ---- attention prep (knob fuse_attn_prep): q/k per-head RMSNorm, both ropes, K/V stores ----
// Replaces rmsnorm(q), rmsnorm(k), rope(q), rope(k), store(k), store(v): one
// threadgroup of mk_norm_threads(dim) threads per (row, unit), unit = a query
// head, a key head or a value head. Norms use mk_inv_rms with the reference
// threadgroup size and write in place exactly as mk_rmsnorm; after a device
// barrier the same threadgroup applies mk_rope's per-pair arithmetic; key heads
// are then copied to their cache slot, value heads copied directly (mk_store).
// p.x/xs = q, p.y/ys = k, p.z/zs = v views; p.heads/kv_heads/dim/rot; p.w/dtype and
// p.s/sdtype = q and k norm weights (buffers 0 and 1); p.n = table offset (words);
// p.capacity = K/V cache rows; p.eps; p.flags MK_FAST_TRIG.
kernel void mk_dec_attn_prep(device const uchar *wq [[buffer(0)]], device const uchar *wk [[buffer(1)]],
                             device float *arena [[buffer(2)]], device const float *tables [[buffer(3)]],
                             device float *K [[buffer(4)]], device float *V [[buffer(5)]], constant MK &p [[buffer(8)]],
                             uint tgid [[threadgroup_position_in_grid]], uint lid [[thread_position_in_threadgroup]],
                             uint tg [[threads_per_threadgroup]], uint simd [[simdgroup_index_in_threadgroup]],
                             uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float partial[33];
    uint units = p.heads + 2 * p.kv_heads;
    uint r = tgid / units, u = tgid % units;
    if (r >= p.rows) return;
    uint row = p.row0 + r;
    uint slot = (p.pos + row) % p.capacity;
    uint kvw = p.kv_heads * p.dim;
    if (u >= p.heads + p.kv_heads) {
        uint vh = u - p.heads - p.kv_heads;
        device const float *v = arena + p.z + row * p.zs + vh * p.dim;
        for (uint c = lid; c < p.dim; c += tg) V[ulong(slot) * kvw + vh * p.dim + c] = v[c];
        return;
    }
    bool is_q = u < p.heads;
    uint h = is_q ? u : u - p.heads;
    device float *x = arena + (is_q ? p.x + row * p.xs : p.y + row * p.ys) + h * p.dim;
    device const uchar *w = is_q ? wq : wk;
    ulong woff = is_q ? p.w : p.s;
    uint wdt = is_q ? p.dtype : p.sdtype;
    float inv = mk_inv_rms(x, p.dim, p.eps, lid, tg, simd, lane, partial);
    for (uint base = lid * 4; base < p.dim; base += tg * 4)
        for (uint j = 0; j < 4; ++j) if (base + j < p.dim) {
            float v = x[base + j] * inv;
            x[base + j] = mk_load(w, woff, wdt, base + j) * v;
        }
    threadgroup_barrier(mem_flags::mem_device);
    uint half_rot = p.rot / 2;
    for (uint j = lid; j < half_rot; j += tg) {
        float theta = float(p.pos + row) * tables[p.n + j];
        float c, s;
        if (p.flags & MK_FAST_TRIG) { c = metal::fast::cos(theta); s = metal::fast::sin(theta); }
        else { c = metal::precise::cos(theta); s = metal::precise::sin(theta); }
        float a = x[j], b = x[j + half_rot];
        x[j] = a * c - b * s;
        x[j + half_rot] = a * s + b * c;
    }
    if (is_q) return;
    threadgroup_barrier(mem_flags::mem_device);
    for (uint c = lid; c < p.dim; c += tg) K[ulong(slot) * kvw + h * p.dim + c] = x[c];
}

// ---- two-stage argmax (knob decode_argmax = 1) ----------------------------------
// mk_argmax's result: the lowest index whose value equals the row maximum
// (comparisons as in mk_argmax, so +0 and -0 tie), or the all-ones sentinel if
// any value is non-finite. That is an integer, the same for any reduction
// order. Stage 1: threadgroup g of a row scans the slice [g * p.m, (g + 1) * p.m)
// and writes (value, index, bad) to P[(r * groups + g) * 3]; stage 2: one
// threadgroup per row reduces its groups' triples. p.every = groups per row.
inline void mk_dec_argmax_reduce(float best, uint index, uint invalid, uint tg, uint simd, uint lane,
                                 threadgroup float *best_v, threadgroup uint *best_i, threadgroup uint *bad,
                                 thread float &v_out, thread uint &i_out, thread uint &b_out) {
    float top = simd_max(best);
    uint winner = simd_min(best == top ? index : 0xffffffffu);
    uint any_bad = simd_max(invalid);
    uint nsimd = (tg + 31) / 32;
    if (lane == 0) { best_v[simd] = top; best_i[simd] = winner; bad[simd] = any_bad; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float v = lane < nsimd ? best_v[lane] : -INFINITY;
    uint i = lane < nsimd ? best_i[lane] : 0xffffffffu;
    uint b = lane < nsimd ? bad[lane] : 0u;
    float t = simd_max(v);
    i_out = simd_min(v == t ? i : 0xffffffffu);
    b_out = simd_max(b);
    v_out = t;
}
kernel void mk_dec_argmax_part(device const float *arena [[buffer(0)]], device float *P [[buffer(1)]],
                               constant MK &p [[buffer(8)]], uint tgid [[threadgroup_position_in_grid]],
                               uint lid [[thread_position_in_threadgroup]], uint tg [[threads_per_threadgroup]],
                               uint simd [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float best_v[32];
    threadgroup uint best_i[32], bad[32];
    uint r = tgid / p.every, g = tgid % p.every;
    if (r >= p.rows) return;
    device const float *x = arena + p.x + (p.row0 + r) * p.xs;
    uint lo = g * p.m, hi = min(lo + p.m, p.n);
    float best = -INFINITY;
    uint index = 0xffffffffu, invalid = 0;
    for (uint i = lo + lid; i < hi; i += tg) {
        float v = x[i];
        invalid |= isfinite(v) ? 0u : 1u;
        if (v > best) { best = v; index = i; }
    }
    float v;
    uint i, b;
    mk_dec_argmax_reduce(best, index, invalid, tg, simd, lane, best_v, best_i, bad, v, i, b);
    if (lid == 0) {
        device float *o = P + (r * p.every + g) * 3;
        o[0] = v;
        o[1] = as_type<float>(i);
        o[2] = as_type<float>(b);
    }
}
kernel void mk_dec_argmax_final(device float *arena [[buffer(0)]], device const float *P [[buffer(1)]],
                                constant MK &p [[buffer(8)]], uint tgid [[threadgroup_position_in_grid]],
                                uint lid [[thread_position_in_threadgroup]], uint tg [[threads_per_threadgroup]],
                                uint simd [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float best_v[32];
    threadgroup uint best_i[32], bad[32];
    uint r = tgid;
    if (r >= p.rows) return;
    float best = -INFINITY;
    uint index = 0xffffffffu, invalid = 0;
    for (uint g = lid; g < p.every; g += tg) {
        device const float *q = P + (r * p.every + g) * 3;
        float v = q[0];
        uint i = as_type<uint>(q[1]);
        invalid |= as_type<uint>(q[2]);
        // Groups ascend in index, so strict > keeps the lowest index of a tie.
        if (v > best || (v == best && i < index)) { best = v; index = i; }
    }
    float v;
    uint i, b;
    mk_dec_argmax_reduce(best, index, invalid, tg, simd, lane, best_v, best_i, bad, v, i, b);
    if (lid == 0) arena[p.y + (p.row0 + r) * p.ys] = as_type<float>(b ? 0xffffffffu : i);
}

// ---- split attention, block chain (knob decode_attention = 2) ----------------------
// The chain of the split attention, with the scalar part of each SIMD's online softmax computed 32
// steps at a time. Step j of SIMD s (position start + s + 8 j) uses
// nm_j = max(m_{j-1}, dot_j), f_j = exp(m_{j-1} - nm_j), es_j = exp(dot_j - nm_j);
// the running max is a prefix max (exact in any association), so lane j of a
// block computes nm_j by a shuffle scan seeded with the carried max and then
// f_j and es_j, all in parallel. The recurrences l = l * f + es and
// o = o * f + es * V are then applied serially in step order, exactly as before.
// (A NaN score could make the scan's max differ from the serial one; scores are
// finite whenever q and the keys are.)
kernel void mk_dec_chain_block(device float *arena [[buffer(0)]], device const float *V [[buffer(1)]],
                               device const float *S [[buffer(2)]], constant MK &p [[buffer(8)]],
                               uint tgid [[threadgroup_position_in_grid]], uint simd [[simdgroup_index_in_threadgroup]],
                               uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float maxes[MK_ATT_SIMDS], sums[MK_ATT_SIMDS], outs[MK_ATT_SIMDS][32];
    uint parts = (p.dim + 31) / 32;
    uint h = tgid / parts, part = tgid % parts;
    if (h >= p.heads) return;
    uint row = p.row0;
    uint position = p.pos + row;
    uint kvh = h / (p.heads / p.kv_heads);
    uint start = p.window && position + 1 > p.window ? position + 1 - p.window : 0;
    uint d = lane + 32 * part;
    device const float *sc = S + h * p.every;
    float m = -INFINITY, l = 0, o = 0;
    uint steps = position >= start + simd ? (position - start - simd) / MK_ATT_SIMDS + 1 : 0;
    bool live = d < p.dim;
    uint dd = min(d, p.dim - 1);
    ulong kvoff = kvh * p.dim + dd, rowlen = p.kv_heads * p.dim;
    // Software pipeline: block b + 1's score and 32 value loads (clamped,
    // unconditional; slots advance by 8 mod capacity) are issued before block b's
    // scan and serial recurrences, which run over the valid steps in order.
    float dot_n = 0, vv_n[32];
    auto fetch = [&](uint b0) {
        uint j = b0 + lane;
        dot_n = j < steps ? sc[simd + MK_ATT_SIMDS * j] : -INFINITY;
        uint t = start + simd + MK_ATT_SIMDS * b0;
        uint slot = t % p.capacity;
        _Pragma("unroll") for (uint jj = 0; jj < 32; ++jj) {
            vv_n[jj] = V[ulong(t <= position ? slot : 0) * rowlen + kvoff];
            t += MK_ATT_SIMDS;
            slot += MK_ATT_SIMDS;
            if (slot >= p.capacity) slot -= p.capacity;
        }
    };
    if (steps) fetch(0);
    for (uint b0 = 0; b0 < steps; b0 += 32) {
        uint cnt = min(32u, steps - b0);
        float dot = dot_n, vv[32];
        _Pragma("unroll") for (uint jj = 0; jj < 32; ++jj) vv[jj] = vv_n[jj];
        if (b0 + 32 < steps) fetch(b0 + 32);
        float nm = max(m, dot);
        for (uint off = 1; off < 32; off <<= 1) {
            float up = simd_shuffle_up(nm, off);
            if (lane >= off) nm = max(nm, up);
        }
        float prev = simd_shuffle_up(nm, 1);
        if (lane == 0) prev = m;
        float f = metal::precise::exp(prev - nm), es = metal::precise::exp(dot - nm);
        _Pragma("unroll") for (uint jj = 0; jj < 32; ++jj) {
            float fj = simd_shuffle(f, jj), ej = simd_shuffle(es, jj);
            if (jj < cnt) {
                l = l * fj + ej;
                if (live) o = o * fj + ej * vv[jj];
            }
        }
        m = simd_shuffle(nm, cnt - 1);
    }
    if (lane == 0) { maxes[simd] = m; sums[simd] = l; }
    outs[simd][lane] = o;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd != 0) return;
    float M = maxes[0];
    for (uint s = 1; s < MK_ATT_SIMDS; ++s) M = max(M, maxes[s]);
    float total = 0;
    float acc = 0;
    for (uint s = 0; s < MK_ATT_SIMDS; ++s) {
        float f = sums[s] == 0 ? 0.0f : metal::precise::exp(maxes[s] - M);
        total += sums[s] * f;
        acc += outs[s][lane] * f;
    }
    if (d >= p.dim) return;
    device float *out = arena + p.y + row * p.ys + h * p.dim;
    if (p.flags & MK_DEC_GATED) {
        float g = mk_act(arena[p.z + row * p.zs + h], p.act_b);
        float a = acc / total;
        out[d] = a * g;
    } else {
        out[d] = acc / total;
    }
}
