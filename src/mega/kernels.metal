// Megakernel reference library: one generic kernel per IR operation.
// Every kernel processes `rows` token rows; each row's arithmetic depends only
// on that row's data and absolute position, never on the batch size or on how
// rows are grouped into threads, so a batched forward is bit-identical to the
// same tokens fed one at a time. Variants (row/token tiles, fusions) must keep
// each row's expression tree; see docs/MEGAKERNEL.md.
#include <metal_stdlib>
using namespace metal;

// Host mirror: Params in src/mega/gpu.zig (160 bytes).
struct MK {
    ulong w, s, b;              // byte offsets of weight, scales, biases
    uint rows, row0, pos, pad_r;
    uint x, xs, y, ys, z, zs;   // word offsets and row strides of operands
    uint n, m, dtype, bits, group, sdtype, bdtype, flags;
    uint heads, kv_heads, dim, window, capacity, taps, rot, every;
    uint kind, act_a, act_b, pad_k;
    float eps, scale, pad_f0, pad_f1;
};

constant uint MK_ROUND_BF16 = 1u;
constant uint MK_HAS_WEIGHT = 2u;
constant uint MK_FAST_TRIG = 4u;
constant uint MK_ROUND_OUT = 16u;
constant uint MK_BIAS16 = 32u;
constant uint MK_FIXED_TAPS = 128u;

inline float mk_load(device const uchar *base, ulong offset, uint dtype, ulong i) {
    if (dtype == 0) {
        if ((offset & 3u) == 0) return ((device const float *)(base + offset))[i];
        device const uchar *q = base + offset + i * 4ul;
        uint u = uint(q[0]) | (uint(q[1]) << 8) | (uint(q[2]) << 16) | (uint(q[3]) << 24);
        return as_type<float>(u);
    }
    ushort h;
    if ((offset & 1u) == 0) {
        h = ((device const ushort *)(base + offset))[i];
    } else {
        device const uchar *q = base + offset + i * 2ul;
        h = ushort(uint(q[0]) | (uint(q[1]) << 8));
    }
    return dtype == 1 ? float(as_type<half>(h)) : as_type<float>(uint(h) << 16);
}
inline float mk_bf16(float v) {
    // Round to nearest even, as a float -> bfloat16 conversion does.
    uint bits = as_type<uint>(v);
    if ((bits & 0x7f800000u) == 0x7f800000u) return v;
    bits = (bits + 0x7fffu + ((bits >> 16) & 1u)) & 0xffff0000u;
    return as_type<float>(bits);
}
// MLX's numerically stable sigmoid.
inline float mk_sigmoid(float x) {
    float y = 1.0f / (1.0f + metal::precise::exp(metal::abs(x)));
    return x < 0 ? y : 1.0f - y;
}
inline float mk_gelu_tanh(float x) {
    constexpr float k = 0.7978845608028654f; // sqrt(2 / pi)
    float u = k * (x + 0.044715f * x * x * x);
    float t = metal::abs(u) > 20.0f ? (u > 0.0f ? 1.0f : -1.0f) : tanh(u);
    return 0.5f * x * (1.0f + t);
}
inline float mk_act(float x, uint act) {
    if (act == 1) return x * mk_sigmoid(x);
    if (act == 2) return mk_sigmoid(x);
    if (act == 3) return tanh(x);
    if (act == 4) return mk_gelu_tanh(x);
    return x;
}
// Packed code word i of a weight at byte `offset`. Safetensors data in
// unpadded multi-shard files can start at any byte offset (`(offset & 3) != 0`),
// including odd offsets; handle 4-byte, 2-byte, and unaligned byte loads.
inline uint mk_word(device const uchar *w, ulong offset, ulong i) {
    if ((offset & 3u) == 0) return ((device const uint *)(w + offset))[i];
    if ((offset & 1u) == 0) {
        device const ushort *h = (device const ushort *)(w + offset);
        return uint(h[2 * i]) | (uint(h[2 * i + 1]) << 16);
    }
    device const uchar *q = w + offset + i * 4ul;
    return uint(q[0]) | (uint(q[1]) << 8) | (uint(q[2]) << 16) | (uint(q[3]) << 24);
}
inline float mk_code(device const uchar *w, ulong offset, uint bits, ulong i) {
    ulong bit = i * bits;
    uint lo = w[offset + bit / 8];
    uint value = lo >> (bit % 8);
    if (bit % 8 + bits > 8) value |= uint(w[offset + bit / 8 + 1]) << (8 - bit % 8);
    return float(value & ((1u << bits) - 1u));
}

// ---- embedding ------------------------------------------------------------
// Grid: rows * n threads. Dense (bits 0) or affine table: code * scale + bias.
kernel void mk_embed(device const uchar *w [[buffer(0)]], device const uchar *s [[buffer(1)]],
                     device const uchar *b [[buffer(2)]], device float *arena [[buffer(3)]],
                     constant MK &p [[buffer(8)]], uint i [[thread_position_in_grid]]) {
    uint r = i / p.n, c = i % p.n;
    if (r >= p.rows) return;
    uint row = p.row0 + r;
    uint token = as_type<uint>(arena[p.x + row * p.xs]);
    ulong index = ulong(token) * p.n + c;
    float v;
    if (p.bits == 0) v = mk_load(w, p.w, p.dtype, index);
    else {
        ulong group = index / p.group;
        float scale = mk_load(s, p.s, p.sdtype, group), bias = mk_load(b, p.b, p.bdtype, group);
        float code = mk_code(w, p.w, p.bits, index);
        if (p.flags & MK_ROUND_BF16) v = mk_bf16(mk_bf16(scale * code) + bias);
        else v = scale * code + bias;
    }
    arena[p.y + row * p.ys + c] = v;
}

// ---- RMSNorm ----------------------------------------------------------------
// One threadgroup per (row, group) of n words, MLX rms_single_row structure:
// each thread squares 4 contiguous words, SIMD sums, then SIMD 0 sums partials.
// Threadgroup size is a function of n alone (host: mk_norm_threads).
inline float mk_inv_rms(device const float *x, uint n, float eps, uint lid, uint tg, uint simd, uint lane,
                        threadgroup float *partial) {
    float acc = 0;
    for (uint base = lid * 4; base < n; base += tg * 4)
        for (uint j = 0; j < 4; ++j) if (base + j < n) { float v = x[base + j]; acc += v * v; }
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
    return partial[32];
}
kernel void mk_rmsnorm(device const uchar *w [[buffer(0)]], device float *arena [[buffer(1)]],
                       constant MK &p [[buffer(8)]], uint tgid [[threadgroup_position_in_grid]],
                       uint lid [[thread_position_in_threadgroup]], uint tg [[threads_per_threadgroup]],
                       uint simd [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float partial[33];
    uint r = tgid / p.heads, g = tgid % p.heads;
    if (r >= p.rows) return;
    uint row = p.row0 + r;
    device const float *x = arena + p.x + row * p.xs + g * p.n;
    device float *y = arena + p.y + row * p.ys + g * p.n;
    float inv = mk_inv_rms(x, p.n, p.eps, lid, tg, simd, lane, partial);
    // Each thread rewrites only words it read itself: in-place is safe.
    for (uint base = lid * 4; base < p.n; base += tg * 4)
        for (uint j = 0; j < 4; ++j) if (base + j < p.n) {
            float v = x[base + j] * inv;
            if (p.flags & MK_ROUND_BF16) v = mk_bf16(v);
            float result = (p.flags & MK_HAS_WEIGHT) ? mk_load(w, p.w, p.dtype, base + j) * v : v;
            y[base + j] = (p.flags & MK_ROUND_OUT) ? mk_bf16(result) : result;
        }
}

// ---- linear -----------------------------------------------------------------
// Reference arithmetic for affine 4-bit (group % 16 == 0, n % 16 == 0, weight
// offset even): MLX
// qmv_fast. Lane L owns 16-input segments L, L+32, ...; per segment the inputs
// are summed and prescaled, codes are masked in place, and the row accumulates
// scale * dot + sum * bias; a final simd_sum reduces lanes. A SIMD group owns
// R rows x TT tokens; neither tile changes any row's expression tree.
template <uint R, uint TT, uint K = 0>
inline void mk_q4_tile(device const uchar *wb, device const uchar *sb, device const uchar *bb,
                       device float *arena, constant MK &p, uint simd, uint lane) {
    const uint tiles_t = K ? 1u : (p.rows + TT - 1) / TT;
    uint first = (simd / tiles_t) * R, t0 = (simd % tiles_t) * TT;
    if (first >= p.m) return;
    const uint n = K ? K : p.n, group_size = K ? 64u : p.group;
    uint row_words = n / 8, row_groups = n / group_size, seg_group = group_size / 16;
    float acc[TT][R];
    for (uint t = 0; t < TT; ++t) for (uint r = 0; r < R; ++r) acc[t][r] = 0;
    for (uint seg = lane; seg < n / 16; seg += 32) {
        float xv[TT][16], xsum[TT];
        for (uint t = 0; t < TT; ++t) {
            xsum[t] = 0;
            if (!K && t0 + t >= p.rows) continue;
            device const float *x = arena + p.x + (p.row0 + t0 + t) * p.xs + seg * 16;
            for (uint j = 0; j < 16; j += 4) {
                float v0 = x[j], v1 = x[j + 1], v2 = x[j + 2], v3 = x[j + 3];
                xsum[t] += v0 + v1 + v2 + v3;
                xv[t][j] = v0;
                xv[t][j + 1] = v1 / 16.0f;
                xv[t][j + 2] = v2 / 256.0f;
                xv[t][j + 3] = v3 / 4096.0f;
            }
        }
        for (uint r = 0; r < R; ++r) {
            uint row = first + r;
            if (row >= p.m) continue;
            ulong base = ulong(row) * row_words + seg * 2;
            uint c0 = mk_word(wb, p.w, base), c1 = mk_word(wb, p.w, base + 1);
            uint group = row * row_groups + seg / seg_group;
            float scale = mk_load(sb, p.s, K ? 2u : p.sdtype, group), bias = mk_load(bb, p.b, K ? 2u : p.bdtype, group);
            for (uint t = 0; t < TT; ++t) {
                if (!K && t0 + t >= p.rows) continue;
                float dot = 0;
                for (uint j = 0; j < 16; j += 4) {
                    uint codes = j < 8 ? c0 : c1;
                    uint hw = (j & 7u) < 4 ? codes : codes >> 16;
                    dot += (xv[t][j] * float(hw & 0x000fu) +
                            xv[t][j + 1] * float(hw & 0x00f0u) +
                            xv[t][j + 2] * float(hw & 0x0f00u) +
                            xv[t][j + 3] * float(hw & 0xf000u));
                }
                acc[t][r] += scale * dot + xsum[t] * bias;
            }
        }
    }
    for (uint t = 0; t < TT; ++t) {
        if (!K && t0 + t >= p.rows) continue;
        device float *out = arena + p.y + (p.row0 + t0 + t) * p.ys;
        for (uint r = 0; r < R; ++r) {
            float sum = simd_sum(acc[t][r]);
            if (lane == 0 && first + r < p.m) out[first + r] = sum;
        }
    }
}
#define MK_Q4(R, TT) \
kernel void mk_linear_q4_r##R##_t##TT(device const uchar *w [[buffer(0)]], device const uchar *s [[buffer(1)]], \
        device const uchar *b [[buffer(2)]], device float *arena [[buffer(3)]], constant MK &p [[buffer(8)]], \
        uint i [[thread_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) { \
    mk_q4_tile<R, TT>(w, s, b, arena, p, i / 32, lane); \
}
MK_Q4(1, 1) MK_Q4(2, 1) MK_Q4(4, 1) MK_Q4(8, 1)
MK_Q4(1, 2) MK_Q4(2, 2) MK_Q4(4, 2) MK_Q4(8, 2)
MK_Q4(1, 4) MK_Q4(2, 4) MK_Q4(4, 4) MK_Q4(8, 4)

// Fixed input width, group-64 and BF16 metadata for the common single-row shapes.
#define MK_Q4_STATIC(K) \
kernel void mk_linear_q4_static_##K(device const uchar *w [[buffer(0)]], device const uchar *s [[buffer(1)]], \
        device const uchar *b [[buffer(2)]], device float *arena [[buffer(3)]], constant MK &p [[buffer(8)]], \
        uint i [[thread_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) { \
    mk_q4_tile<2, 1, K>(w, s, b, arena, p, i / 32, lane); \
}
MK_Q4_STATIC(2048) MK_Q4_STATIC(3840) MK_Q4_STATIC(4096) MK_Q4_STATIC(7680) MK_Q4_STATIC(8192) MK_Q4_STATIC(15360)

// Any affine width/group and dense F32/F16/BF16: lane-strided columns, one
// SIMD per (row, token). Reference for shapes outside the 4-bit fast form.
kernel void mk_linear_generic(device const uchar *w [[buffer(0)]], device const uchar *s [[buffer(1)]],
                              device const uchar *b [[buffer(2)]], device float *arena [[buffer(3)]],
                              constant MK &p [[buffer(8)]], uint i [[thread_position_in_grid]],
                              uint lane [[thread_index_in_simdgroup]]) {
    uint simd = i / 32;
    uint row = simd / p.rows, t = simd % p.rows;
    if (row >= p.m) return;
    device const float *x = arena + p.x + (p.row0 + t) * p.xs;
    float acc = 0;
    for (uint col = lane; col < p.n; col += 32) {
        ulong index = ulong(row) * p.n + col;
        float wv;
        if (p.bits == 0) wv = mk_load(w, p.w, p.dtype, index);
        else {
            ulong group = index / p.group;
            wv = mk_code(w, p.w, p.bits, index) * mk_load(s, p.s, p.sdtype, group) + mk_load(b, p.b, p.bdtype, group);
        }
        acc += wv * x[col];
    }
    acc = simd_sum(acc);
    if (lane == 0) arena[p.y + (p.row0 + t) * p.ys + row] = (p.flags & MK_ROUND_OUT) ? mk_bf16(acc) : acc;
}

// ---- elementwise ------------------------------------------------------------
// out = act_a(a) op act_b(b[c / every]); kind 0 add, 1 mul. Grid rows * n.
kernel void mk_binary(device float *arena [[buffer(0)]], constant MK &p [[buffer(8)]],
                      uint i [[thread_position_in_grid]]) {
    uint r = i / p.n, c = i % p.n;
    if (r >= p.rows) return;
    uint row = p.row0 + r;
    float a = mk_act(arena[p.x + row * p.xs + c], p.act_a);
    float b = mk_act(arena[p.y + row * p.ys + c / p.every], p.act_b);
    float result = p.kind == 0 ? a + b : (p.kind == 1 ? a * b : a - b);
    arena[p.z + row * p.zs + c] = (p.flags & MK_ROUND_OUT) ? mk_bf16(result) : result;
}

kernel void mk_concat(device float *arena [[buffer(0)]], constant MK &p [[buffer(8)]],
                      uint i [[thread_position_in_grid]]) {
    uint width = p.n + p.m;
    uint r = i / width, c = i % width;
    if (r >= p.rows) return;
    uint row = p.row0 + r;
    float value = c < p.n
        ? arena[p.x + row * p.xs + c]
        : arena[p.y + row * p.ys + c - p.n];
    arena[p.z + row * p.zs + c] = value;
}

kernel void mk_activate(device float *arena [[buffer(0)]], constant MK &p [[buffer(8)]],
                        uint i [[thread_position_in_grid]]) {
    uint r = i / p.n, c = i % p.n;
    if (r >= p.rows) return;
    uint row = p.row0 + r;
    arena[p.z + row * p.zs + c] = mk_act(arena[p.x + row * p.xs + c], p.act_a);
}

// ---- state writes -----------------------------------------------------------
// Row r goes to slot (pos + row) % capacity of a sequence/ring state.
kernel void mk_store(device const float *arena [[buffer(0)]], device float *dst [[buffer(1)]],
                     constant MK &p [[buffer(8)]], uint i [[thread_position_in_grid]]) {
    uint r = i / p.n, c = i % p.n;
    if (r >= p.rows) return;
    uint row = p.row0 + r;
    uint slot = (p.pos + row) % p.capacity;
    dst[ulong(slot) * p.n + c] = arena[p.x + row * p.xs + c];
}

// ---- causal depthwise conv ----------------------------------------------------
// out[c] = sum_k w[c][k] * u(p - taps + 1 + k)[c], k ascending, zeros before 0.
kernel void mk_conv(device const float *ring [[buffer(0)]], device const uchar *w [[buffer(1)]],
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
    arena[p.y + row * p.ys + c] = acc;
}

// ---- rotary embedding -----------------------------------------------------------
// Pairs (j, j + rot/2) of each head; angle = float(position) * freq[j].
kernel void mk_rope(device float *arena [[buffer(0)]], device const float *tables [[buffer(1)]],
                    constant MK &p [[buffer(8)]], uint i [[thread_position_in_grid]]) {
    uint half_rot = p.rot / 2, per_row = p.heads * half_rot;
    uint r = i / per_row, rem = i % per_row, h = rem / half_rot, j = rem % half_rot;
    if (r >= p.rows) return;
    uint row = p.row0 + r;
    float theta = float(p.pos + row) * tables[p.z + j];
    float c, s;
    if (p.flags & MK_FAST_TRIG) { c = metal::fast::cos(theta); s = metal::fast::sin(theta); }
    else { c = metal::precise::cos(theta); s = metal::precise::sin(theta); }
    device float *x = arena + p.x + row * p.xs + h * p.dim;
    float a = x[j], b = x[j + half_rot];
    x[j] = a * c - b * s;
    x[j + half_rot] = a * s + b * c;
}

// ---- attention ------------------------------------------------------------------
// One threadgroup of MK_ATT_SIMDS SIMD groups per (row, query head). Lane L owns
// dims L, L+32, ... (any head dim up to 256). SIMD s visits positions start+s,
// start+s+S, ... with an online softmax; partial states merge in SIMD order.
// The visit order is fixed by the row's own position range, so batching
// cannot change it.
constant uint MK_ATT_SIMDS = 8;
template <uint MAX_E>
inline void mk_attention_impl(device float *arena, device const float *K, device const float *V,
                              constant MK &p, uint tgid, uint simd, uint lane,
                              threadgroup float *maxes, threadgroup float *sums,
                              threadgroup float (*outs)[MAX_E * 32]) {
    uint r = tgid / p.heads, h = tgid % p.heads;
    if (r >= p.rows) return;
    uint row = p.row0 + r;
    uint position = p.pos + row;
    uint kvh = h / (p.heads / p.kv_heads);
    uint start = p.window && position + 1 > p.window ? position + 1 - p.window : 0;
    uint E = (p.dim + 31) / 32;
    float q[MAX_E], o[MAX_E];
    device const float *qp = arena + p.x + row * p.xs + h * p.dim;
    for (uint e = 0; e < MAX_E; ++e) {
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
    float acc[MAX_E];
    for (uint e = 0; e < MAX_E; ++e) acc[e] = 0;
    for (uint s = 0; s < MK_ATT_SIMDS; ++s) {
        float f = sums[s] == 0 ? 0.0f : metal::precise::exp(maxes[s] - M);
        total += sums[s] * f;
        for (uint e = 0; e < E; ++e) { uint d = lane + 32 * e; if (d < p.dim) acc[e] += outs[s][d] * f; }
    }
    device float *out = arena + p.y + row * p.ys + h * p.dim;
    for (uint e = 0; e < E; ++e) { uint d = lane + 32 * e; if (d < p.dim) out[d] = acc[e] / total; }
}
kernel void mk_attention(device float *arena [[buffer(0)]], device const float *K [[buffer(1)]],
                         device const float *V [[buffer(2)]], constant MK &p [[buffer(8)]],
                         uint tgid [[threadgroup_position_in_grid]], uint simd [[simdgroup_index_in_threadgroup]],
                         uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float maxes[MK_ATT_SIMDS], sums[MK_ATT_SIMDS], outs[MK_ATT_SIMDS][256];
    mk_attention_impl<8>(arena, K, V, p, tgid, simd, lane, maxes, sums, outs);
}
kernel void mk_attention_d512(device float *arena [[buffer(0)]], device const float *K [[buffer(1)]],
                              device const float *V [[buffer(2)]], constant MK &p [[buffer(8)]],
                              uint tgid [[threadgroup_position_in_grid]], uint simd [[simdgroup_index_in_threadgroup]],
                              uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float maxes[MK_ATT_SIMDS], sums[MK_ATT_SIMDS], outs[MK_ATT_SIMDS][512];
    mk_attention_impl<16>(arena, K, V, p, tgid, simd, lane, maxes, sums, outs);
}

// ---- argmax ----------------------------------------------------------------------
// One 1024-thread group per row. Lowest index wins ties; any non-finite value
// in the row yields the 0xffffffff sentinel.
kernel void mk_argmax(device float *arena [[buffer(0)]], constant MK &p [[buffer(8)]],
                      uint tgid [[threadgroup_position_in_grid]], uint lid [[thread_position_in_threadgroup]],
                      uint tg [[threads_per_threadgroup]], uint simd [[simdgroup_index_in_threadgroup]],
                      uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float best_v[32];
    threadgroup uint best_i[32], bad[32];
    uint r = tgid;
    if (r >= p.rows) return;
    uint row = p.row0 + r;
    device const float *x = arena + p.x + row * p.xs;
    float best = -INFINITY;
    uint index = 0xffffffffu, invalid = 0;
    for (uint i = lid; i < p.n; i += tg) {
        float v = x[i];
        invalid |= isfinite(v) ? 0u : 1u;
        if (v > best) { best = v; index = i; }
    }
    float top = simd_max(best);
    uint winner = simd_min(best == top ? index : 0xffffffffu);
    uint any_bad = simd_max(invalid);
    uint nsimd = (tg + 31) / 32;
    if (lane == 0) { best_v[simd] = top; best_i[simd] = winner; bad[simd] = any_bad; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd != 0) return;
    float v = lane < nsimd ? best_v[lane] : -INFINITY;
    uint i = lane < nsimd ? best_i[lane] : 0xffffffffu;
    uint b = lane < nsimd ? bad[lane] : 0u;
    float t = simd_max(v);
    uint w = simd_min(v == t ? i : 0xffffffffu);
    b = simd_max(b);
    if (lane == 0) arena[p.y + row * p.ys] = as_type<float>(b ? 0xffffffffu : w);
}

// ---- select_halves ------------------------------------------------------------
// Grid: rows * n threads (n = 2 * heads * dim). Splits interleaved
// [heads, 2, dim] rows into contiguous [first_halves, second_halves].
kernel void mk_select_halves(device float *arena [[buffer(0)]], constant MK &p [[buffer(8)]],
                             uint i [[thread_position_in_grid]]) {
    uint r = i / p.n, c = i % p.n;
    if (r >= p.rows) return;
    uint row = p.row0 + r;
    uint half_n = p.m;
    bool second = c >= half_n;
    uint k = second ? c - half_n : c;
    uint g = k / p.dim;
    uint d = k % p.dim;
    uint src_c = g * (2u * p.dim) + (second ? p.dim : 0u) + d;
    arena[p.y + row * p.ys + c] = arena[p.x + row * p.xs + src_c];
}

// ---- gated_delta_net ----------------------------------------------------------
// One SIMD group (32 threads) per (v_head, v_dim), walking rows sequentially so
// batched prefill is bit-identical to tokens fed one at a time.
kernel void mk_gated_delta_net(device const uchar *a_log [[buffer(0)]],
                               device const uchar *dt_bias [[buffer(1)]],
                               device float *arena [[buffer(2)]],
                               device float *state [[buffer(3)]],
                               constant MK &p [[buffer(8)]],
                               uint gid [[thread_position_in_grid]],
                               uint lane [[thread_index_in_simdgroup]]) {
    uint simd_id = gid / 32u;
    if (simd_id >= p.heads * p.m) return;
    uint hv = simd_id / p.m;
    uint dv = simd_id % p.m;
    uint hk = hv / (p.heads / p.kv_heads);
    uint key_dim = p.kv_heads * p.dim;
    uint E = (p.dim + 31u) / 32u;

    device float *st_ptr = state + (ulong(hv) * p.m + dv) * p.dim;
    float st[8];
    for (uint e = 0; e < E; ++e) {
        uint c = E * lane + e;
        st[e] = (c < p.dim) ? st_ptr[c] : 0.0f;
    }

    float a_exp = metal::precise::exp(mk_load(a_log, p.w, p.dtype, hv));
    float dt_b = mk_load(dt_bias, p.s, p.sdtype, hv);
    float scale_q = metal::rsqrt(float(p.dim));

    for (uint r = 0; r < p.rows; ++r) {
        uint row = p.row0 + r;
        device const float *row_qkv = arena + p.x + ulong(row) * p.xs;
        device const float *q_ptr = row_qkv + ulong(hk) * p.dim;
        device const float *k_ptr = row_qkv + key_dim + ulong(hk) * p.dim;
        float v_val = row_qkv[2u * key_dim + ulong(hv) * p.m + dv];
        float a_val = arena[p.y + ulong(row) * p.ys + hv];
        float b_val = arena[p.z + ulong(row) * p.zs + hv];

        float sp_in = a_val + dt_b;
        float sp = sp_in > 20.0f ? sp_in : metal::precise::log(1.0f + metal::precise::exp(sp_in));
        float g_val = metal::precise::exp(-a_exp * sp);
        float beta_val = mk_sigmoid(b_val);

        float qv[8], kv[8], q2 = 0.0f, k2 = 0.0f;
        for (uint e = 0; e < E; ++e) {
            uint c = E * lane + e;
            float qq = (c < p.dim) ? q_ptr[c] : 0.0f;
            float kk = (c < p.dim) ? k_ptr[c] : 0.0f;
            qv[e] = qq;
            kv[e] = kk;
            q2 += qq * qq;
            k2 += kk * kk;
        }
        float q_inv = metal::rsqrt(simd_sum(q2) + p.eps) * scale_q;
        float k_inv = metal::rsqrt(simd_sum(k2) + p.eps);

        float kv_mem = 0.0f;
        for (uint e = 0; e < E; ++e) {
            st[e] *= g_val;
            kv_mem += st[e] * (kv[e] * k_inv);
        }
        kv_mem = simd_sum(kv_mem);
        float delta = (v_val - kv_mem) * beta_val;

        float out = 0.0f;
        for (uint e = 0; e < E; ++e) {
            st[e] += (kv[e] * k_inv) * delta;
            out += st[e] * (qv[e] * q_inv);
        }
        out = simd_sum(out);
        if (lane == 0) arena[p.every + ulong(row) * p.pad_k + ulong(hv) * p.m + dv] = out;
    }

    for (uint e = 0; e < E; ++e) {
        uint c = E * lane + e;
        if (c < p.dim) st_ptr[c] = st[e];
    }
}

kernel void mk_copy_slot(
    device const float *arena [[buffer(0)]],
    device float *dst [[buffer(1)]],
    constant MK &p [[buffer(8)]],
    uint gid [[thread_position_in_grid]]) {
    ulong total = ulong(p.rows) * p.n;
    if (gid >= total) return;
    uint r = uint(gid / p.n);
    uint c = uint(gid % p.n);
    dst[ulong(p.row0 + r) * p.ys + p.y + c] = arena[ulong(p.x) + ulong(p.pos + r) * p.xs + c];
}

kernel void mk_row_transform(device float *arena [[buffer(0)]], device const uchar *w [[buffer(1)]],
                              constant MK &p [[buffer(8)]], uint i [[thread_position_in_grid]]) {
    uint r = i / p.n, c = i % p.n;
    if (r >= p.rows) return;
    uint row = p.row0 + r;
    float x = arena[p.x + row * p.xs + c];
    if (p.kind == 1) x = tanh(x / p.scale) * p.scale;
    if (p.kind == 2) x *= 1.0f + p.scale * metal::precise::log(max(float(p.pos + row + 1) / float(p.m), 1.0f));
    if (p.kind == 3) x *= p.scale;
    if (p.kind == 4) x *= mk_load(w, p.w, p.dtype, 0);
    arena[p.y + row * p.ys + c] = (p.flags & MK_ROUND_BF16) ? mk_bf16(x) : x;
}
