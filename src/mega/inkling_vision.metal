// Native Metal kernels for Inkling 2 Vision (23-layer Perception-Encoder ViT +
// 2x2 spatial merge + GELU MLP projector) and multimodal embedding scatter.
// Uses the shared MK parameter block and mk_load / mk_bf16 helpers from kernels.metal.

inline float mk_erf_approx(float x) {
    float sign = x < 0.0f ? -1.0f : 1.0f;
    float ax = metal::abs(x);
    float t = 1.0f / (1.0f + 0.3275911f * ax);
    float poly = (((((1.061405429f * t - 1.453152027f) * t) + 1.421413741f) * t - 0.284496736f) * t + 0.254829592f) * t;
    return sign * (1.0f - poly * metal::precise::exp(-ax * ax));
}

inline float mk_gelu(float x) {
    return 0.5f * x * (1.0f + mk_erf_approx(x * 0.70710678118654752440f));
}

inline float4 mk_load_bf16x4(device const uchar *buf, ulong byte_off, ulong elem_idx) {
    device const ushort *ptr = (device const ushort *)(buf + byte_off) + elem_idx;
    return float4(
        as_type<float>(uint(ptr[0]) << 16),
        as_type<float>(uint(ptr[1]) << 16),
        as_type<float>(uint(ptr[2]) << 16),
        as_type<float>(uint(ptr[3]) << 16));
}

// ---- Prepend CLS + add bilinearly interpolated 2D position embeddings -------
// Grid: (1 + gh * gw) * D threads.
// p.w = pos_emb offset, p.s = class_emb offset, p.b = cls_pos_emb offset,
// p.n = D, p.heads = gh, p.kv_heads = gw, p.dim = posemb_grid_size (32), p.dtype = weight dtype.
kernel void mk_vis_embed_prep(device const float *patch_proj [[buffer(0)]],
                              device const uchar *pos_buf [[buffer(1)]],
                              device const uchar *cls_buf [[buffer(2)]],
                              device const uchar *cls_pos_buf [[buffer(3)]],
                              device float *seq_out [[buffer(4)]],
                              constant MK &p [[buffer(8)]],
                              uint gid [[thread_position_in_grid]]) {
    uint D = p.n;
    uint gh = p.heads;
    uint gw = p.kv_heads;
    uint total_rows = 1u + gh * gw;
    uint r = gid / D;
    uint c = gid % D;
    if (r >= total_rows) return;
    if (r == 0u) {
        float cls_v = mk_load(cls_buf, p.s, p.dtype, c);
        float cls_p = mk_load(cls_pos_buf, p.b, p.dtype, c);
        seq_out[p.y + c] = cls_v + cls_p;
        return;
    }
    uint idx = r - 1u;
    uint py = idx / gw;
    uint px = idx % gw;
    uint G = p.dim;
    float sy = (float(py) + 0.5f) * (float(G) / float(gh)) - 0.5f;
    float sx = (float(px) + 0.5f) * (float(G) / float(gw)) - 0.5f;
    sy = clamp(sy, 0.0f, float(G - 1u));
    sx = clamp(sx, 0.0f, float(G - 1u));
    uint y0 = uint(floor(sy));
    uint x0 = uint(floor(sx));
    uint y1 = min(y0 + 1u, G - 1u);
    uint x1 = min(x0 + 1u, G - 1u);
    float wy = sy - float(y0);
    float wx = sx - float(x0);
    float p00 = mk_load(pos_buf, p.w, p.dtype, ulong(y0 * G + x0) * D + c);
    float p01 = mk_load(pos_buf, p.w, p.dtype, ulong(y0 * G + x1) * D + c);
    float p10 = mk_load(pos_buf, p.w, p.dtype, ulong(y1 * G + x0) * D + c);
    float p11 = mk_load(pos_buf, p.w, p.dtype, ulong(y1 * G + x1) * D + c);
    float pe = (1.0f - wy) * ((1.0f - wx) * p00 + wx * p01) + wy * ((1.0f - wx) * p10 + wx * p11);
    seq_out[p.y + ulong(r) * p.ys + c] = patch_proj[p.x + ulong(idx) * p.xs + c] + pe;
}

// ---- Generic LayerNorm with weight and bias ---------------------------------
// One threadgroup (256 threads) per row r in [0, p.rows).
kernel void mk_vis_layernorm(device const float *x_buf [[buffer(0)]],
                             device const uchar *w_buf [[buffer(1)]],
                             device const uchar *b_buf [[buffer(2)]],
                             device float *y_buf [[buffer(3)]],
                             constant MK &p [[buffer(8)]],
                             uint tgid [[threadgroup_position_in_grid]],
                             uint lid [[thread_position_in_threadgroup]],
                             uint tg [[threads_per_threadgroup]],
                             uint simd [[simdgroup_index_in_threadgroup]],
                             uint lane [[thread_index_in_simdgroup]]) {
    if (tgid >= p.rows) return;
    uint n = p.n;
    device const float *xr = x_buf + p.x + ulong(tgid) * p.xs;
    device float *yr = y_buf + p.y + ulong(tgid) * p.ys;

    float sum = 0.0f;
    for (uint c = lid; c < n; c += tg) sum += xr[c];
    sum = simd_sum(sum);
    threadgroup float partial[32];
    threadgroup float mean_s;
    threadgroup float inv_s;
    if (lane == 0u) partial[simd] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint n_simds = (tg + 31u) / 32u;
    if (simd == 0u) {
        float v = lane < n_simds ? partial[lane] : 0.0f;
        v = simd_sum(v);
        if (lane == 0u) mean_s = v / float(n);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float mean = mean_s;

    float var_sum = 0.0f;
    for (uint c = lid; c < n; c += tg) {
        float d = xr[c] - mean;
        var_sum += d * d;
    }
    var_sum = simd_sum(var_sum);
    if (lane == 0u) partial[simd] = var_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd == 0u) {
        float v = lane < n_simds ? partial[lane] : 0.0f;
        v = simd_sum(v);
        if (lane == 0u) inv_s = metal::precise::rsqrt(v / float(n) + p.eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float inv = inv_s;

    bool round_out = (p.flags & MK_ROUND_OUT) != 0u;
    for (uint c = lid; c < n; c += tg) {
        float g = mk_load(w_buf, p.w, p.dtype, c);
        float b = mk_load(b_buf, p.b, p.bdtype, c);
        float v = (xr[c] - mean) * inv * g + b;
        yr[c] = round_out ? mk_bf16(v) : v;
    }
}

// ---- Single-read register-cached LayerNorm for n <= 1024, n % 4 == 0 --------
// One threadgroup (256 threads, 1 float4 per thread) per row: 3x less read traffic.
kernel void mk_vis_layernorm_1024(device const float *x_buf [[buffer(0)]],
                                  device const uchar *w_buf [[buffer(1)]],
                                  device const uchar *b_buf [[buffer(2)]],
                                  device float *y_buf [[buffer(3)]],
                                  constant MK &p [[buffer(8)]],
                                  uint tgid [[threadgroup_position_in_grid]],
                                  uint lid [[thread_position_in_threadgroup]],
                                  uint simd [[simdgroup_index_in_threadgroup]],
                                  uint lane [[thread_index_in_simdgroup]]) {
    if (tgid >= p.rows) return;
    uint n = p.n;
    uint c = lid * 4u;
    bool active = c < n;
    device const float *xr = x_buf + p.x + ulong(tgid) * p.xs;
    device float *yr = y_buf + p.y + ulong(tgid) * p.ys;

    float4 xv = active ? *(device const float4 *)(xr + c) : float4(0.0f);
    float s = simd_sum(xv.x + xv.y + xv.z + xv.w);
    threadgroup float partial[8];
    threadgroup float mean_s;
    threadgroup float inv_s;
    if (lane == 0u) partial[simd] = s;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd == 0u) {
        float v = lane < 8u ? partial[lane] : 0.0f;
        v = simd_sum(v);
        if (lane == 0u) mean_s = v / float(n);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float mean = mean_s;

    float4 dv = active ? (xv - float4(mean)) : float4(0.0f);
    float vs = simd_sum(dot(dv, dv));
    if (lane == 0u) partial[simd] = vs;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd == 0u) {
        float v = lane < 8u ? partial[lane] : 0.0f;
        v = simd_sum(v);
        if (lane == 0u) inv_s = metal::precise::rsqrt(v / float(n) + p.eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (!active) return;

    float inv = inv_s;
    float4 gv = (p.dtype == 2u && (p.w & 1ul) == 0ul)
        ? mk_load_bf16x4(w_buf, p.w, c)
        : float4(mk_load(w_buf, p.w, p.dtype, c), mk_load(w_buf, p.w, p.dtype, c + 1u),
                 mk_load(w_buf, p.w, p.dtype, c + 2u), mk_load(w_buf, p.w, p.dtype, c + 3u));
    float4 bv = (p.bdtype == 2u && (p.b & 1ul) == 0ul)
        ? mk_load_bf16x4(b_buf, p.b, c)
        : float4(mk_load(b_buf, p.b, p.bdtype, c), mk_load(b_buf, p.b, p.bdtype, c + 1u),
                 mk_load(b_buf, p.b, p.bdtype, c + 2u), mk_load(b_buf, p.b, p.bdtype, c + 3u));
    float4 out = dv * inv * gv + bv;
    if ((p.flags & MK_ROUND_OUT) != 0u) {
        out = float4(mk_bf16(out.x), mk_bf16(out.y), mk_bf16(out.z), mk_bf16(out.w));
    }
    *(device float4 *)(yr + c) = out;
}

// ---- Tiled 32x32 simdgroup_matrix dense linear (reference / small tiles) ----
kernel void mk_vis_linear_mma(device const float *x_buf [[buffer(0)]],
                              device const uchar *w_buf [[buffer(1)]],
                              device const uchar *b_buf [[buffer(2)]],
                              device float *y_buf [[buffer(3)]],
                              constant MK &p [[buffer(8)]],
                              uint tgid [[threadgroup_position_in_grid]],
                              uint lid [[thread_position_in_threadgroup]],
                              uint simd [[simdgroup_index_in_threadgroup]],
                              uint lane [[thread_index_in_simdgroup]]) {
    uint M = p.m;
    uint N = p.n;
    uint R = p.rows;
    uint tiles_m = (M + 31u) / 32u;
    uint tm = tgid % tiles_m;
    uint tr = tgid / tiles_m;
    uint row_base = tr * 32u;
    uint col_base = tm * 32u;
    if (row_base >= R || col_base >= M) return;

    uint sg_r = (simd / 2u) * 16u;
    uint sg_c = (simd % 2u) * 16u;

    simdgroup_float8x8 acc[2][2];
    for (uint i = 0u; i < 2u; i++) {
        for (uint j = 0u; j < 2u; j++) {
            acc[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        }
    }

    threadgroup float As[32][36];
    threadgroup float Bs[32][36];

    bool round_in = (p.flags & MK_ROUND_BF16) != 0u;

    for (uint k0 = 0u; k0 < N; k0 += 32u) {
        for (uint e = lid; e < 1024u; e += 128u) {
            uint lr = e / 32u;
            uint lk = e % 32u;
            uint gr = row_base + lr;
            uint gk = k0 + lk;
            float xv = (gr < R && gk < N) ? x_buf[p.x + ulong(gr) * p.xs + gk] : 0.0f;
            As[lr][lk] = round_in ? mk_bf16(xv) : xv;

            uint gm = col_base + lr;
            float wv = (gm < M && gk < N) ? mk_load(w_buf, p.w, p.dtype, ulong(gm) * N + gk) : 0.0f;
            Bs[lk][lr] = wv;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint kk = 0u; kk < 32u; kk += 8u) {
            simdgroup_float8x8 a0, a1, b0, b1;
            simdgroup_load(a0, &As[sg_r + 0u][kk], 36);
            simdgroup_load(a1, &As[sg_r + 8u][kk], 36);
            simdgroup_load(b0, &Bs[kk][sg_c + 0u], 36);
            simdgroup_load(b1, &Bs[kk][sg_c + 8u], 36);
            simdgroup_multiply_accumulate(acc[0][0], a0, b0, acc[0][0]);
            simdgroup_multiply_accumulate(acc[0][1], a0, b1, acc[0][1]);
            simdgroup_multiply_accumulate(acc[1][0], a1, b0, acc[1][0]);
            simdgroup_multiply_accumulate(acc[1][1], a1, b1, acc[1][1]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (uint i = 0u; i < 2u; i++) {
        for (uint j = 0u; j < 2u; j++) {
            simdgroup_store(acc[i][j], &As[sg_r + i * 8u][sg_c + j * 8u], 36);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    bool has_bias = (p.flags & MK_HAS_WEIGHT) != 0u;
    bool apply_gelu = (p.act_a == 4u);
    bool round_out = (p.flags & MK_ROUND_OUT) != 0u;

    for (uint e = lid; e < 1024u; e += 128u) {
        uint lr = e / 32u;
        uint lc = e % 32u;
        uint gr = row_base + lr;
        uint gc = col_base + lc;
        if (gr < R && gc < M) {
            float v = As[lr][lc];
            if (has_bias) v += mk_load(b_buf, p.b, p.bdtype, gc);
            if (apply_gelu) v = mk_gelu(v);
            if (round_out) v = mk_bf16(v);
            y_buf[p.y + ulong(gr) * p.ys + gc] = v;
        }
    }
}

// ---- Fast 32x32 tiled simdgroup_matrix dense linear -------------------------
// 4 simdgroups (2x2 of 16x16 = 4 accs/sg, zero register spilling), vectorized
// float4/BF16x4 loads, padded shared memory, optional bias + GELU (act_a == 4)
// OR fused LayerScale residual add (act_a == 5: y[r,c] += gamma[c] * (v + bias[c])).
// Grid: ceil(M/32) * ceil(rows/32) * 128 threads, threadgroup = 128.
kernel void mk_vis_linear_mma32(device const float *x_buf [[buffer(0)]],
                                device const uchar *w_buf [[buffer(1)]],
                                device const uchar *b_buf [[buffer(2)]],
                                device float *y_buf [[buffer(3)]],
                                device const uchar *g_buf [[buffer(4)]],
                                constant MK &p [[buffer(8)]],
                                uint tgid [[threadgroup_position_in_grid]],
                                uint lid [[thread_position_in_threadgroup]],
                                uint simd [[simdgroup_index_in_threadgroup]]) {
    uint M = p.m;
    uint N = p.n;
    uint R = p.rows;
    uint tiles_m = (M + 31u) / 32u;
    uint tm = tgid % tiles_m;
    uint tr = tgid / tiles_m;
    uint row_base = tr * 32u;
    uint col_base = tm * 32u;
    if (row_base >= R || col_base >= M) return;

    uint sg_r = (simd / 2u) * 16u;
    uint sg_c = (simd % 2u) * 16u;

    simdgroup_float8x8 acc[2][2];
    acc[0][0] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    acc[0][1] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    acc[1][0] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    acc[1][1] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

    threadgroup float As[32][36];
    threadgroup float Bs[32][36];

    bool is_bf16 = (p.dtype == 2u);
    bool x_vec4 = (p.xs % 4u == 0u) && (p.x % 4u == 0u);
    bool w_vec4 = is_bf16 && (N % 4u == 0u) && ((p.w & 1ul) == 0ul);

    for (uint k0 = 0u; k0 < N; k0 += 32u) {
        // 256 float4 chunks across 128 threads = 2 iterations/thread.
        for (uint v = lid; v < 256u; v += 128u) {
            uint lr = v >> 3;           // 0..31
            uint lk = (v & 7u) << 2;    // 0, 4, ..., 28
            uint gr = row_base + lr;
            uint gk = k0 + lk;

            float4 xv = float4(0.0f);
            if (gr < R) {
                if (x_vec4 && gk + 3u < N) {
                    xv = *(device const float4 *)(x_buf + p.x + ulong(gr) * p.xs + gk);
                } else {
                    if (gk + 0u < N) xv.x = x_buf[p.x + ulong(gr) * p.xs + gk + 0u];
                    if (gk + 1u < N) xv.y = x_buf[p.x + ulong(gr) * p.xs + gk + 1u];
                    if (gk + 2u < N) xv.z = x_buf[p.x + ulong(gr) * p.xs + gk + 2u];
                    if (gk + 3u < N) xv.w = x_buf[p.x + ulong(gr) * p.xs + gk + 3u];
                }
            }
            As[lr][lk + 0u] = xv.x;
            As[lr][lk + 1u] = xv.y;
            As[lr][lk + 2u] = xv.z;
            As[lr][lk + 3u] = xv.w;

            uint gm = col_base + lr;
            float4 wv = float4(0.0f);
            if (gm < M) {
                if (w_vec4 && gk + 3u < N) {
                    wv = mk_load_bf16x4(w_buf, p.w, ulong(gm) * N + gk);
                } else {
                    if (gk + 0u < N) wv.x = mk_load(w_buf, p.w, p.dtype, ulong(gm) * N + gk + 0u);
                    if (gk + 1u < N) wv.y = mk_load(w_buf, p.w, p.dtype, ulong(gm) * N + gk + 1u);
                    if (gk + 2u < N) wv.z = mk_load(w_buf, p.w, p.dtype, ulong(gm) * N + gk + 2u);
                    if (gk + 3u < N) wv.w = mk_load(w_buf, p.w, p.dtype, ulong(gm) * N + gk + 3u);
                }
            }
            Bs[lk + 0u][lr] = wv.x;
            Bs[lk + 1u][lr] = wv.y;
            Bs[lk + 2u][lr] = wv.z;
            Bs[lk + 3u][lr] = wv.w;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint kk = 0u; kk < 32u; kk += 8u) {
            simdgroup_float8x8 a0, a1, b0, b1;
            simdgroup_load(a0, &As[sg_r + 0u][kk], 36);
            simdgroup_load(a1, &As[sg_r + 8u][kk], 36);
            simdgroup_load(b0, &Bs[kk][sg_c + 0u], 36);
            simdgroup_load(b1, &Bs[kk][sg_c + 8u], 36);
            simdgroup_multiply_accumulate(acc[0][0], a0, b0, acc[0][0]);
            simdgroup_multiply_accumulate(acc[0][1], a0, b1, acc[0][1]);
            simdgroup_multiply_accumulate(acc[1][0], a1, b0, acc[1][0]);
            simdgroup_multiply_accumulate(acc[1][1], a1, b1, acc[1][1]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    bool has_bias = (p.flags & MK_HAS_WEIGHT) != 0u;
    bool apply_gelu = (p.act_a == 4u);
    bool apply_ls_add = (p.act_a == 5u);
    bool round_out = (p.flags & MK_ROUND_OUT) != 0u;

    threadgroup float bias_s[32];
    threadgroup float ls_s[32];
    if (lid < 32u) {
        uint gc = col_base + lid;
        bias_s[lid] = (has_bias && gc < M) ? mk_load(b_buf, p.b, p.bdtype, gc) : 0.0f;
        ls_s[lid] = (apply_ls_add && gc < M) ? mk_load(g_buf, p.s, p.dtype, gc) : 1.0f;
    }

    for (uint i = 0u; i < 2u; i++) {
        for (uint j = 0u; j < 2u; j++) {
            simdgroup_store(acc[i][j], &As[sg_r + i * 8u][sg_c + j * 8u], 36);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint e = lid; e < 1024u; e += 128u) {
        uint lr = e >> 5;
        uint lc = e & 31u;
        uint gr = row_base + lr;
        uint gc = col_base + lc;
        if (gr < R && gc < M) {
            float v = As[lr][lc] + bias_s[lc];
            if (apply_gelu) v = mk_gelu(v);
            ulong y_off = p.y + ulong(gr) * p.ys + gc;
            if (apply_ls_add) v = y_buf[y_off] + ls_s[lc] * v;
            if (round_out) v = mk_bf16(v);
            y_buf[y_off] = v;
        }
    }
}

// ---- High-throughput 64x64 FP16-input FP32-accum simdgroup_matrix linear ----
// 8 simdgroups (256 threads, 4x2 of 16x32 = 8 accs/sg, zero register spilling),
// 9.7 KB shared memory, 2x less X and W bandwidth vs 32x32, vectorized float4 /
// BF16x4 / half4 loads & stores, fused bias + GELU (act_a == 4) OR LayerScale
// residual add (act_a == 5), direct register-to-global epilogue (no epilogue barrier).
// Grid: ceil(M/64) * ceil(rows/64) * 256 threads, threadgroup = 256.
kernel void mk_vis_linear_mma_h64x64(device const float *x_buf [[buffer(0)]],
                                     device const uchar *w_buf [[buffer(1)]],
                                     device const uchar *b_buf [[buffer(2)]],
                                     device float *y_buf [[buffer(3)]],
                                     device const uchar *g_buf [[buffer(4)]],
                                     constant MK &p [[buffer(8)]],
                                     uint tgid [[threadgroup_position_in_grid]],
                                     uint lid [[thread_position_in_threadgroup]],
                                     uint simd [[simdgroup_index_in_threadgroup]],
                                     uint lane [[thread_index_in_simdgroup]]) {
    uint M = p.m;
    uint N = p.n;
    uint R = p.rows;
    uint tiles_m = (M + 63u) / 64u;
    uint tm = tgid % tiles_m;
    uint tr = tgid / tiles_m;
    uint row_base = tr * 64u;
    uint col_base = tm * 64u;
    if (row_base >= R || col_base >= M) return;

    uint sg_r = (simd >> 1) * 16u;
    uint sg_c = (simd & 1u) * 32u;

    simdgroup_float8x8 acc[2][4];
    for (uint i = 0u; i < 2u; i++) {
        for (uint j = 0u; j < 4u; j++) {
            acc[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        }
    }

    threadgroup half As[64][40];
    threadgroup half Bs[32][72];

    bool is_bf16 = (p.dtype == 2u);
    bool x_vec4 = (p.xs % 4u == 0u) && (p.x % 4u == 0u);
    bool w_vec4 = is_bf16 && (N % 4u == 0u) && ((p.w & 1ul) == 0ul);

    for (uint k0 = 0u; k0 < N; k0 += 32u) {
        // 512 float4 chunks across 256 threads = 2 iterations/thread for both A and B.
        for (uint v = lid; v < 512u; v += 256u) {
            uint lr = v >> 3;           // 0..63
            uint lk = (v & 7u) << 2;    // 0, 4, ..., 28
            uint gr = row_base + lr;
            uint gk = k0 + lk;

            float4 xv = float4(0.0f);
            if (gr < R) {
                if (x_vec4 && gk + 3u < N) {
                    xv = *(device const float4 *)(x_buf + p.x + ulong(gr) * p.xs + gk);
                } else {
                    if (gk + 0u < N) xv.x = x_buf[p.x + ulong(gr) * p.xs + gk + 0u];
                    if (gk + 1u < N) xv.y = x_buf[p.x + ulong(gr) * p.xs + gk + 1u];
                    if (gk + 2u < N) xv.z = x_buf[p.x + ulong(gr) * p.xs + gk + 2u];
                    if (gk + 3u < N) xv.w = x_buf[p.x + ulong(gr) * p.xs + gk + 3u];
                }
            }
            *(threadgroup half4 *)(&As[lr][lk]) = half4(xv);

            uint gm = col_base + lr;
            float4 wv = float4(0.0f);
            if (gm < M) {
                if (w_vec4 && gk + 3u < N) {
                    wv = mk_load_bf16x4(w_buf, p.w, ulong(gm) * N + gk);
                } else {
                    if (gk + 0u < N) wv.x = mk_load(w_buf, p.w, p.dtype, ulong(gm) * N + gk + 0u);
                    if (gk + 1u < N) wv.y = mk_load(w_buf, p.w, p.dtype, ulong(gm) * N + gk + 1u);
                    if (gk + 2u < N) wv.z = mk_load(w_buf, p.w, p.dtype, ulong(gm) * N + gk + 2u);
                    if (gk + 3u < N) wv.w = mk_load(w_buf, p.w, p.dtype, ulong(gm) * N + gk + 3u);
                }
            }
            half4 wh = half4(wv);
            Bs[lk + 0u][lr] = wh.x;
            Bs[lk + 1u][lr] = wh.y;
            Bs[lk + 2u][lr] = wh.z;
            Bs[lk + 3u][lr] = wh.w;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint kk = 0u; kk < 32u; kk += 8u) {
            simdgroup_half8x8 a0, a1, b0, b1, b2, b3;
            simdgroup_load(a0, &As[sg_r + 0u][kk], 40);
            simdgroup_load(a1, &As[sg_r + 8u][kk], 40);
            simdgroup_load(b0, &Bs[kk][sg_c + 0u], 72);
            simdgroup_load(b1, &Bs[kk][sg_c + 8u], 72);
            simdgroup_load(b2, &Bs[kk][sg_c + 16u], 72);
            simdgroup_load(b3, &Bs[kk][sg_c + 24u], 72);
            simdgroup_multiply_accumulate(acc[0][0], a0, b0, acc[0][0]);
            simdgroup_multiply_accumulate(acc[0][1], a0, b1, acc[0][1]);
            simdgroup_multiply_accumulate(acc[0][2], a0, b2, acc[0][2]);
            simdgroup_multiply_accumulate(acc[0][3], a0, b3, acc[0][3]);
            simdgroup_multiply_accumulate(acc[1][0], a1, b0, acc[1][0]);
            simdgroup_multiply_accumulate(acc[1][1], a1, b1, acc[1][1]);
            simdgroup_multiply_accumulate(acc[1][2], a1, b2, acc[1][2]);
            simdgroup_multiply_accumulate(acc[1][3], a1, b3, acc[1][3]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    uint qid = lane >> 2;
    uint sm = (qid & 4u) + ((lane >> 1) & 3u);
    uint sn = ((qid & 2u) << 1) + ((lane & 1u) << 1);
    bool has_bias = (p.flags & MK_HAS_WEIGHT) != 0u;
    bool apply_gelu = (p.act_a == 4u);
    bool apply_ls_add = (p.act_a == 5u);
    bool round_out = (p.flags & MK_ROUND_OUT) != 0u;

    for (uint j = 0u; j < 4u; j++) {
        uint gc0 = col_base + sg_c + j * 8u + sn;
        uint gc1 = gc0 + 1u;
        float b0 = (has_bias && gc0 < M) ? mk_load(b_buf, p.b, p.bdtype, gc0) : 0.0f;
        float b1 = (has_bias && gc1 < M) ? mk_load(b_buf, p.b, p.bdtype, gc1) : 0.0f;
        float g0 = (apply_ls_add && gc0 < M) ? mk_load(g_buf, p.s, p.dtype, gc0) : 1.0f;
        float g1 = (apply_ls_add && gc1 < M) ? mk_load(g_buf, p.s, p.dtype, gc1) : 1.0f;
        for (uint i = 0u; i < 2u; i++) {
            uint gr = row_base + sg_r + i * 8u + sm;
            if (gr >= R) continue;
            float v0 = acc[i][j].thread_elements()[0] + b0;
            float v1 = acc[i][j].thread_elements()[1] + b1;
            if (apply_gelu) {
                v0 = mk_gelu(v0);
                v1 = mk_gelu(v1);
            }
            ulong base_y = p.y + ulong(gr) * p.ys;
            if (gc0 < M) {
                if (apply_ls_add) v0 = y_buf[base_y + gc0] + g0 * v0;
                if (round_out) v0 = mk_bf16(v0);
                y_buf[base_y + gc0] = v0;
            }
            if (gc1 < M) {
                if (apply_ls_add) v1 = y_buf[base_y + gc1] + g1 * v1;
                if (round_out) v1 = mk_bf16(v1);
                y_buf[base_y + gc1] = v1;
            }
        }
    }
}

// ---- In-place interleaved 2D RoPE on Q or K (reference) ---------------------
kernel void mk_vis_rope2d(device float *buf [[buffer(0)]],
                          constant MK &p [[buffer(8)]],
                          uint gid [[thread_position_in_grid]]) {
    uint pairs_per_head = p.dim / 2u;
    uint pairs_per_row = p.heads * pairs_per_head;
    uint r = gid / pairs_per_row;
    if (r == 0u || r >= p.rows) return;
    uint rem = gid % pairs_per_row;
    uint h = rem / pairs_per_head;
    uint pair_idx = rem % pairs_per_head;

    uint patch_idx = r - 1u;
    uint gw = p.capacity;
    uint py = patch_idx / gw;
    uint px = patch_idx % gw;

    uint axis_pairs = pairs_per_head / 2u;
    bool is_row = pair_idx >= axis_pairs;
    uint k = is_row ? (pair_idx - axis_pairs) : pair_idx;
    float pos = is_row ? float(py + 1u) : float(px + 1u);
    float inv_freq = 1.0f / metal::precise::pow(p.scale, float(2u * k) / float(p.dim / 2u));
    float angle = pos * inv_freq;
    float cs = metal::precise::cos(angle);
    float sn = metal::precise::sin(angle);

    ulong off = p.x + ulong(r) * p.xs + ulong(h) * p.dim + ulong(2u * pair_idx);
    float a = buf[off];
    float b = buf[off + 1u];
    buf[off] = a * cs - b * sn;
    buf[off + 1u] = a * sn + b * cs;
}

// ---- Fused in-place 2D RoPE on BOTH Q and K using precomputed (cos, sin) ----
// rope_tbl layout: [(gw + gh), dim/4, 2] floats — first gw entries for x=px+1,
// next gh entries for y=py+1. Eliminates all GPU pow/cos/sin and halves dispatches.
kernel void mk_vis_rope2d_qk(device float *q_buf [[buffer(0)]],
                             device float *k_buf [[buffer(1)]],
                             device const float *rope_tbl [[buffer(2)]],
                             constant MK &p [[buffer(8)]],
                             uint gid [[thread_position_in_grid]]) {
    uint pairs_per_head = p.dim / 2u;
    uint pairs_per_row = p.heads * pairs_per_head;
    uint r = gid / pairs_per_row;
    if (r == 0u || r >= p.rows) return;
    uint rem = gid % pairs_per_row;
    uint h = rem / pairs_per_head;
    uint pair_idx = rem % pairs_per_head;

    uint patch_idx = r - 1u;
    uint gw = p.capacity;
    uint py = patch_idx / gw;
    uint px = patch_idx % gw;

    uint axis_pairs = pairs_per_head / 2u;
    bool is_row = pair_idx >= axis_pairs;
    uint k = is_row ? (pair_idx - axis_pairs) : pair_idx;
    uint tbl_row = is_row ? (gw + py) : px;
    uint tbl_off = (tbl_row * axis_pairs + k) * 2u;
    float cs = rope_tbl[tbl_off];
    float sn = rope_tbl[tbl_off + 1u];

    ulong q_off = p.x + ulong(r) * p.xs + ulong(h) * p.dim + ulong(2u * pair_idx);
    ulong k_off = p.y + ulong(r) * p.ys + ulong(h) * p.dim + ulong(2u * pair_idx);
    float qa = q_buf[q_off];
    float qb = q_buf[q_off + 1u];
    q_buf[q_off] = qa * cs - qb * sn;
    q_buf[q_off + 1u] = qa * sn + qb * cs;

    float ka = k_buf[k_off];
    float kb = k_buf[k_off + 1u];
    k_buf[k_off] = ka * cs - kb * sn;
    k_buf[k_off + 1u] = ka * sn + kb * cs;
}

// ---- Bidirectional multi-head self-attention (reference / general head_dim) -
kernel void mk_vis_attention(device const float *q_buf [[buffer(0)]],
                             device const float *k_buf [[buffer(1)]],
                             device const float *v_buf [[buffer(2)]],
                             device float *out_buf [[buffer(3)]],
                             constant MK &p [[buffer(8)]],
                             uint gid [[thread_position_in_grid]],
                             uint lane [[thread_index_in_simdgroup]]) {
    uint task = gid / 32u;
    uint total_tasks = p.rows * p.heads;
    if (task >= total_tasks) return;
    uint r = task / p.heads;
    uint h = task % p.heads;
    uint dim = p.dim;
    uint head_off = h * dim;

    float q_reg[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float o_reg[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    device const float *qr = q_buf + p.x + ulong(r) * p.xs + head_off;
    uint iters = (dim + 31u) / 32u;
    for (uint u = 0u; u < iters; u++) {
        uint d = lane + u * 32u;
        if (d < dim) q_reg[u] = qr[d];
    }

    float m = -INFINITY;
    float l_sum = 0.0f;
    for (uint j = 0u; j < p.rows; j++) {
        device const float *kj = k_buf + p.y + ulong(j) * p.ys + head_off;
        float dot = 0.0f;
        for (uint u = 0u; u < iters; u++) {
            uint d = lane + u * 32u;
            if (d < dim) dot += q_reg[u] * kj[d];
        }
        dot = simd_sum(dot) * p.scale;

        float m_new = max(m, dot);
        float alpha = m == -INFINITY ? 0.0f : metal::precise::exp(m - m_new);
        float w = metal::precise::exp(dot - m_new);
        l_sum = l_sum * alpha + w;
        m = m_new;

        device const float *vj = v_buf + p.z + ulong(j) * p.zs + head_off;
        for (uint u = 0u; u < iters; u++) {
            uint d = lane + u * 32u;
            if (d < dim) o_reg[u] = o_reg[u] * alpha + w * vj[d];
        }
    }

    float inv_l = l_sum > 0.0f ? (1.0f / l_sum) : 0.0f;
    device float *orow = out_buf + p.pos + ulong(r) * p.window + head_off;
    for (uint u = 0u; u < iters; u++) {
        uint d = lane + u * 32u;
        if (d < dim) orow[d] = o_reg[u] * inv_l;
    }
}

// ---- FlashAttention-style tiled FP16-MMA / FP32-accum bidirectional MHA (D = 64) -
// BQ = 32 queries, BK = 32 keys, D = 64, 128 threads = 4 simdgroups (8 Q rows/sg).
// Shared memory: half K_smem[64][36] (pre-transposed) + half V_smem[32][68] +
// float S_smem[32][36] (in-place reused for half P) + float D_smem[4][8][8] = 14.3 KB
// (< 16 KB -> 2 concurrent threadgroups/core), zero .thread_elements() fragment spills.
// Grid: ceil(rows/32) * heads * 128 threads, threadgroup = 128.
kernel void mk_vis_attention_flash64(device const float *q_buf [[buffer(0)]],
                                     device const float *k_buf [[buffer(1)]],
                                     device const float *v_buf [[buffer(2)]],
                                     device float *out_buf [[buffer(3)]],
                                     constant MK &p [[buffer(8)]],
                                     uint tgid [[threadgroup_position_in_grid]],
                                     uint lid [[thread_position_in_threadgroup]],
                                     uint simd [[simdgroup_index_in_threadgroup]],
                                     uint lane [[thread_index_in_simdgroup]]) {
    uint S = p.rows;
    uint q_blocks = (S + 31u) / 32u;
    uint qb = tgid % q_blocks;
    uint h = tgid / q_blocks;
    if (h >= p.heads) return;

    uint q_base = qb * 32u;
    uint head_off = h * 64u;
    uint sg_q = simd * 8u;

    threadgroup half K_smem[64][36];
    threadgroup half V_smem[32][68];
    threadgroup float S_smem[32][36];
    threadgroup float D_smem[4][8][8];

    // Step 0: Cooperative float4 load of Q[q_base..q_base+31, 0..63] into V_smem as half4,
    // then hold this simdgroup's 8x64 Q tile permanently in FP16 MMA registers.
    for (uint v = lid; v < 512u; v += 128u) {
        uint lr = v >> 4;           // 0..31
        uint ld = (v & 15u) << 2;   // 0, 4, ..., 60
        uint gr = q_base + lr;
        float4 qv = (gr < S)
            ? *(device const float4 *)(q_buf + p.x + ulong(gr) * p.xs + head_off + ld)
            : float4(0.0f);
        *(threadgroup half4 *)(&V_smem[lr][ld]) = half4(qv);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    simdgroup_half8x8 Q_frag[8];
    simdgroup_float8x8 O_frag[8];
    for (uint db = 0u; db < 8u; db++) {
        simdgroup_load(Q_frag[db], &V_smem[sg_q][db * 8u], 68);
        O_frag[db] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float m_row[8];
    float l_row[8];
    for (uint r = 0u; r < 8u; r++) {
        m_row[r] = -INFINITY;
        l_row[r] = 0.0f;
    }

    const float scale_log2e = p.scale * 1.44269504088896340736f;

    for (uint k_base = 0u; k_base < S; k_base += 32u) {
        // 1. Cooperative float4 load of K (pre-transposed into K_smem[64][36]) and V (half4 into V_smem[32][68]).
        for (uint v = lid; v < 512u; v += 128u) {
            uint lr = v >> 4;
            uint ld = (v & 15u) << 2;
            uint gk = k_base + lr;
            if (gk < S) {
                half4 kh = half4(*(device const float4 *)(k_buf + p.y + ulong(gk) * p.ys + head_off + ld));
                K_smem[ld + 0u][lr] = kh.x;
                K_smem[ld + 1u][lr] = kh.y;
                K_smem[ld + 2u][lr] = kh.z;
                K_smem[ld + 3u][lr] = kh.w;
                *(threadgroup half4 *)(&V_smem[lr][ld]) = half4(*(device const float4 *)(v_buf + p.z + ulong(gk) * p.zs + head_off + ld));
            } else {
                K_smem[ld + 0u][lr] = half(0.0f);
                K_smem[ld + 1u][lr] = half(0.0f);
                K_smem[ld + 2u][lr] = half(0.0f);
                K_smem[ld + 3u][lr] = half(0.0f);
                *(threadgroup half4 *)(&V_smem[lr][ld]) = half4(0.0h);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // 2. S_frag[0..3] = Q_frag[0..7] @ K_smem  ([8, 64] @ [64, 32] -> [8, 32], non-transposed).
        simdgroup_float8x8 S_frag[4];
        for (uint kb = 0u; kb < 4u; kb++) {
            S_frag[kb] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        }
        for (uint db = 0u; db < 8u; db++) {
            for (uint kb = 0u; kb < 4u; kb++) {
                simdgroup_half8x8 k_m;
                simdgroup_load(k_m, &K_smem[db * 8u][kb * 8u], 36);
                simdgroup_multiply_accumulate(S_frag[kb], Q_frag[db], k_m, S_frag[kb]);
            }
        }
        for (uint kb = 0u; kb < 4u; kb++) {
            simdgroup_store(S_frag[kb], &S_smem[sg_q][kb * 8u], 36);
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);

        // 3. Online softmax per simdgroup over its 8 rows x 32 keys; write half P back into S_smem[sg_q..sg_q+7].
        bool valid_k = (k_base + lane) < S;
        float s_row[8];
        for (uint r = 0u; r < 8u; r++) {
            s_row[r] = valid_k ? (S_smem[sg_q + r][lane] * scale_log2e) : -INFINITY;
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);

        threadgroup half *P_sg = (threadgroup half *)(&S_smem[sg_q][0]);
        float alpha_row[8];
        bool need_rescale = false;
        for (uint r = 0u; r < 8u; r++) {
            float s_log2 = s_row[r];
            float bmax = simd_max(s_log2);
            float m_old = m_row[r];
            float m_new = max(m_old, bmax);
            float alpha = (m_old == -INFINITY) ? 0.0f : metal::fast::exp2(m_old - m_new);
            float pv = valid_k ? metal::fast::exp2(s_log2 - m_new) : 0.0f;
            float bsum = simd_sum(pv);
            l_row[r] = l_row[r] * alpha + bsum;
            m_row[r] = m_new;
            alpha_row[r] = alpha;
            if (k_base > 0u && alpha != 1.0f) need_rescale = true;
            P_sg[r * 36u + lane] = half(pv);
        }

        // 4. Rescale O_frag by diag(alpha_row) ONLY on rare max-update blocks (~5 / 126).
        if (need_rescale) {
            uint r0 = lane >> 3;
            uint c0 = lane & 7u;
            D_smem[simd][r0][c0] = (r0 == c0) ? alpha_row[r0] : 0.0f;
            uint r1 = (lane + 32u) >> 3;
            uint c1 = (lane + 32u) & 7u;
            D_smem[simd][r1][c1] = (r1 == c1) ? alpha_row[r1] : 0.0f;
            simdgroup_barrier(mem_flags::mem_threadgroup);
            simdgroup_float8x8 d_mat;
            simdgroup_load(d_mat, &D_smem[simd][0][0], 8);
            for (uint db = 0u; db < 8u; db++) {
                simdgroup_float8x8 tmp;
                simdgroup_multiply(tmp, d_mat, O_frag[db]);
                O_frag[db] = tmp;
            }
        } else {
            simdgroup_barrier(mem_flags::mem_threadgroup);
        }

        // 5. O_frag[0..7] += P_sg @ V_smem  ([8, 32] @ [32, 64] -> [8, 64], FP16-MMA).
        for (uint kb = 0u; kb < 4u; kb++) {
            simdgroup_half8x8 p_m;
            simdgroup_load(p_m, &P_sg[kb * 8u], 36);
            for (uint db = 0u; db < 8u; db++) {
                simdgroup_half8x8 v_m;
                simdgroup_load(v_m, &V_smem[kb * 8u][db * 8u], 68);
                simdgroup_multiply_accumulate(O_frag[db], p_m, v_m, O_frag[db]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Final normalization via D_smem and coalesced float4 store using S_smem+K_smem/V_smem?
    // Reuse K_smem + V_smem? Simpler: normalize into float staging via S_smem in 2 halves of 32 cols!
    {
        uint r0 = lane >> 3;
        uint c0 = lane & 7u;
        D_smem[simd][r0][c0] = (r0 == c0) ? (l_row[r0] > 0.0f ? 1.0f / l_row[r0] : 0.0f) : 0.0f;
        uint r1 = (lane + 32u) >> 3;
        uint c1 = (lane + 32u) & 7u;
        D_smem[simd][r1][c1] = (r1 == c1) ? (l_row[r1] > 0.0f ? 1.0f / l_row[r1] : 0.0f) : 0.0f;
        simdgroup_barrier(mem_flags::mem_threadgroup);
        simdgroup_float8x8 inv_mat;
        simdgroup_load(inv_mat, &D_smem[simd][0][0], 8);
        for (uint half_d = 0u; half_d < 2u; half_d++) {
            for (uint db = 0u; db < 4u; db++) {
                simdgroup_float8x8 tmp;
                simdgroup_multiply(tmp, inv_mat, O_frag[half_d * 4u + db]);
                simdgroup_store(tmp, &S_smem[sg_q][db * 8u], 36);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint v = lid; v < 256u; v += 128u) {
                uint lr = v >> 3;           // 0..31
                uint ld = (v & 7u) << 2;    // 0, 4, ..., 28
                uint gr = q_base + lr;
                if (gr < S) {
                    float4 ov = float4(S_smem[lr][ld + 0u], S_smem[lr][ld + 1u], S_smem[lr][ld + 2u], S_smem[lr][ld + 3u]);
                    *(device float4 *)(out_buf + p.pos + ulong(gr) * p.window + head_off + half_d * 32u + ld) = ov;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
}

// ---- LayerScale residual add: h[r, c] += gamma[c] * x[r, c] -----------------
kernel void mk_vis_layerscale_add(device float *h_buf [[buffer(0)]],
                                  device const float *x_buf [[buffer(1)]],
                                  device const uchar *g_buf [[buffer(2)]],
                                  constant MK &p [[buffer(8)]],
                                  uint gid [[thread_position_in_grid]]) {
    uint r = gid / p.n;
    uint c = gid % p.n;
    if (r >= p.rows) return;
    float g = mk_load(g_buf, p.w, p.dtype, c);
    ulong h_off = p.y + ulong(r) * p.ys + c;
    ulong x_off = p.x + ulong(r) * p.xs + c;
    h_buf[h_off] = h_buf[h_off] + g * x_buf[x_off];
}

// ---- 2x2 Spatial Merge (drops CLS at row 0) ---------------------------------
// Grid: (mh * mw) * (merge * merge * D) threads.
// p.heads = gh, p.kv_heads = gw, p.dim = merge (2), p.n = D (1024).
kernel void mk_vis_spatial_merge(device const float *seq_buf [[buffer(0)]],
                                 device float *merged_buf [[buffer(1)]],
                                 constant MK &p [[buffer(8)]],
                                 uint gid [[thread_position_in_grid]]) {
    uint merge = p.dim;
    uint D = p.n;
    uint out_w = merge * merge * D;
    uint mh = p.heads / merge;
    uint mw = p.kv_heads / merge;
    uint n_tok = mh * mw;
    uint t = gid / out_w;
    uint col = gid % out_w;
    if (t >= n_tok) return;

    uint my = t / mw;
    uint mx = t % mw;
    uint sub = col / D;
    uint c = col % D;
    uint dy = sub / merge;
    uint dx = sub % merge;
    uint sy = my * merge + dy;
    uint sx = mx * merge + dx;
    uint src_r = 1u + sy * p.kv_heads + sx;
    merged_buf[p.y + ulong(t) * p.ys + col] = seq_buf[p.x + ulong(src_r) * p.xs + c];
}

// ---- Scatter visual embeddings into LLM arena after embed_norm --------------
// Grid: rows * n threads.
// row_map[r] < 0 => leave arena untouched; row_map[r] >= 0 => copy vis_buf[row_map[r]].
kernel void mk_scatter_embeds(device float *arena [[buffer(0)]],
                              device const float *vis_buf [[buffer(1)]],
                              device const int *row_map [[buffer(2)]],
                              constant MK &p [[buffer(8)]],
                              uint gid [[thread_position_in_grid]]) {
    uint r = gid / p.n;
    uint c = gid % p.n;
    if (r >= p.rows) return;
    int src = row_map[r];
    if (src < 0) return;
    float v = vis_buf[p.x + ulong(uint(src)) * p.xs + c];
    if (p.flags & MK_ROUND_OUT) v = mk_bf16(v);
    arena[p.y + ulong(p.row0 + r) * p.ys + c] = v;
}
