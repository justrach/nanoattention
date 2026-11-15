#include <metal_stdlib>
using namespace metal;

struct P {
    ulong offset, scale_offset, bias_offset;
    uint n, m, k, dtype, pos, capacity, heads, kv_heads, dim, window;
    uint bits, scale_dtype, bias_dtype;
    float eps, theta, scale;
};
inline float weight(device const uchar *bytes, ulong offset, uint dtype, ulong i) {
    if (dtype == 0) return ((device const float *)(bytes + offset))[i];
    ushort bits = ((device const ushort *)(bytes + offset))[i];
    return dtype == 1 ? float(as_type<half>(bits)) : as_type<float>(uint(bits) << 16);
}
kernel void embedding(device const uchar *w [[buffer(0)]], device float *out [[buffer(1)]],
                      constant P &p [[buffer(8)]], uint i [[thread_position_in_grid]]) {
    if (i < p.n) out[i] = weight(w, p.offset, p.dtype, ulong(p.pos) * p.n + i);
}
// One SIMD group per output row. Activations accumulate in FP32 for all checkpoint dtypes.
kernel void matvec(device const uchar *w [[buffer(0)]], device const float *x [[buffer(1)]],
                   device float *out [[buffer(2)]], constant P &p [[buffer(8)]],
                   uint i [[thread_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
    uint row = i / 32;
    if (row >= p.m) return;
    float sum = 0;
    for (uint col = lane; col < p.n; col += 32)
        sum += weight(w, p.offset, p.dtype, ulong(row) * p.n + col) * x[col];
    sum = simd_sum(sum);
    if (lane == 0) out[row] = sum;
}
inline float affine_weight(device const uchar *w, device const uchar *scales, device const uchar *biases, constant P &p, ulong i) {
    uint packed = w[p.offset + i * p.bits / 8];
    uint code = (packed >> ((i * p.bits) % 8)) & ((1u << p.bits) - 1u);
    ulong group = i / p.k;
    return float(code) * weight(scales, p.scale_offset, p.scale_dtype, group) + weight(biases, p.bias_offset, p.bias_dtype, group);
}
kernel void qembedding(device const uchar *w [[buffer(0)]], device const uchar *scales [[buffer(1)]],
                       device const uchar *biases [[buffer(2)]], device float *out [[buffer(3)]],
                       constant P &p [[buffer(8)]], uint i [[thread_position_in_grid]]) {
    if (i < p.n) out[i] = affine_weight(w, scales, biases, p, ulong(p.pos) * p.n + i);
}
// Flint's hot INT4/group-64 path uses two packed words / 16 inputs per lane,
// like MLX qmv_fast. Each segment stays within one 64-weight affine group.
// This changes FP32 reduction association; parity tests remain mandatory.
inline float dot4_g64(device const uchar *w, device const uchar *scales, device const uchar *biases,
                      uint scale_dtype, uint bias_dtype, device const float *x,
                      uint cols, uint row, uint lane) {
    device const uint *packed = (device const uint *)(w + ulong(row) * (cols / 2));
    float sum = 0;
    for (uint segment = lane; segment < cols / 16; segment += 32) {
        uint codes0 = packed[segment * 2], codes1 = packed[segment * 2 + 1];
        uint group = row * (cols / 64) + segment / 4;
        float scale = weight(scales, 0, scale_dtype, group), bias = weight(biases, 0, bias_dtype, group);
        float dot = 0, input_sum = 0;
#if FLINT_MLX_MATH
        #pragma unroll
        for (uint j = 0; j < 16; j += 4) {
            float value0 = x[segment * 16 + j], value1 = x[segment * 16 + j + 1];
            float value2 = x[segment * 16 + j + 2], value3 = x[segment * 16 + j + 3];
            input_sum += value0 + value1 + value2 + value3;
            value1 = value1 / 16.0f;
            value2 = value2 / 256.0f;
            value3 = value3 / 4096.0f;
            uint codes = j < 8 ? codes0 : codes1;
            uint half_word = (j & 7u) < 4 ? codes : codes >> 16;
            dot += (value0 * float(half_word & 0x000fu) +
                    value1 * float(half_word & 0x00f0u) +
                    value2 * float(half_word & 0x0f00u) +
                    value3 * float(half_word & 0xf000u));
        }
        sum += scale * dot + input_sum * bias;
#else
        #pragma unroll
        for (uint j = 0; j < 16; ++j) {
            float value = x[segment * 16 + j];
            uint codes = j < 8 ? codes0 : codes1;
            uint shift = (j & 3u) * 4;
            uint half_word = (j & 7u) < 4 ? codes : codes >> 16;
            float prescaled = value * (1.0f / float(1u << shift));
            dot += float(half_word & (15u << shift)) * prescaled;
            input_sum += value;
        }
        sum += scale * dot + bias * input_sum;
#endif
    }
    return sum;
}
kernel void qmatvec4_g64(device const uchar *w [[buffer(0)]], device const uchar *scales [[buffer(1)]],
                         device const uchar *biases [[buffer(2)]], device const float *x [[buffer(3)]],
                         device float *out [[buffer(4)]], constant P &p [[buffer(8)]],
                         uint i [[thread_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
    uint row = i / 32;
    if (row >= p.m) return;
    float sum = dot4_g64(w + p.offset, scales + p.scale_offset, biases + p.bias_offset,
                         p.scale_dtype, p.bias_dtype, x, p.n, row, lane);
    sum = simd_sum(sum);
    if (lane == 0) out[row] = sum;
}
kernel void qmatvec(device const uchar *w [[buffer(0)]], device const uchar *scales [[buffer(1)]],
                    device const uchar *biases [[buffer(2)]], device const float *x [[buffer(3)]],
                    device float *out [[buffer(4)]], constant P &p [[buffer(8)]],
                    uint i [[thread_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
    uint row = i / 32;
    if (row >= p.m) return;
    float sum = 0;
    for (uint col = lane; col < p.n; col += 32)
        sum += affine_weight(w, scales, biases, p, ulong(row) * p.n + col) * x[col];
    sum = simd_sum(sum);
    if (lane == 0) out[row] = sum;
}
kernel void rmsnorm(device const uchar *w [[buffer(0)]], device const float *x [[buffer(1)]],
                    device float *out [[buffer(2)]], constant P &p [[buffer(8)]],
                    uint i [[thread_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
    uint row = i / 32;
    if (row >= p.m) return;
    float sum = 0;
    for (uint j = lane; j < p.n; j += 32) { float v = x[row * p.n + j]; sum += v * v; }
    float inv = rsqrt(simd_sum(sum) / float(p.n) + p.eps);
    for (uint j = lane; j < p.n; j += 32)
        out[row * p.n + j] = x[row * p.n + j] * inv * weight(w, p.offset, p.dtype, j);
}
// Hugging Face / NeoX split-half RoPE, not interleaved adjacent-pair RoPE.
kernel void rope(device float *x [[buffer(0)]], constant P &p [[buffer(8)]], uint i [[thread_position_in_grid]]) {
    uint half_dim = p.dim / 2;
    if (i >= p.heads * half_dim) return;
    uint j = i % half_dim, base = (i / half_dim) * p.dim + j;
    float angle = float(p.pos) * pow(p.theta, -2.0f * float(j) / float(p.dim));
    float a = x[base], b = x[base + half_dim], c = cos(angle), s = sin(angle);
    x[base] = a * c - b * s;
    x[base + half_dim] = b * c + a * s;
}
kernel void cache_write(device const float *k [[buffer(0)]], device const float *v [[buffer(1)]],
                        device float *keys [[buffer(2)]], device float *values [[buffer(3)]],
                        constant P &p [[buffer(8)]], uint i [[thread_position_in_grid]]) {
    if (i >= p.n) return;
    uint index = (p.pos % p.capacity) * p.n + i;
    keys[index] = k[i]; values[index] = v[i];
}
// Online softmax includes the learned sink in the denominator, never in the value sum.
// Ring slots are addressed by absolute token position; window includes the current token.
kernel void attention(device const float *q [[buffer(0)]], device const float *keys [[buffer(1)]],
                      device const float *values [[buffer(2)]], device const uchar *sink [[buffer(3)]],
                      device float *out [[buffer(4)]], constant P &p [[buffer(8)]],
                      uint i [[thread_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
    uint head = i / 32;
    if (head >= p.heads) return;
    uint kv_head = head / (p.heads / p.kv_heads);
    uint start = p.window && p.pos + 1 > p.window ? p.pos + 1 - p.window : 0;
    float maximum = weight(sink, p.offset, p.dtype, head), denom = 1;
    float acc[8] = {0,0,0,0,0,0,0,0};
    for (uint t = start; t <= p.pos; ++t) {
        uint base = (t % p.capacity) * p.kv_heads * p.dim + kv_head * p.dim;
        float dot = 0;
        for (uint d = lane; d < p.dim; d += 32) dot += q[head * p.dim + d] * keys[base + d];
        float score = simd_sum(dot) * p.scale;
        float next_max = max(maximum, score), old_scale = exp(maximum - next_max), prob = exp(score - next_max);
        denom = denom * old_scale + prob;
        for (uint d = lane, j = 0; d < p.dim; d += 32, ++j) acc[j] = acc[j] * old_scale + prob * values[base + d];
        maximum = next_max;
    }
    for (uint d = lane, j = 0; d < p.dim; d += 32, ++j) out[head * p.dim + d] = acc[j] / denom;
}
inline float mixture_product(float value, float scale) {
#if defined(FLINT_MLX_MATH) && FLINT_MLX_MATH
    // MLX materializes expert*weight before reducing experts. Preserve that
    // FP32 rounding boundary instead of contracting it into the next addition.
    volatile float product = value * scale;
    return product;
#else
    return value * scale;
#endif
}
inline float swiglu_product(float gate, float up) {
#if defined(FLINT_MLX_MATH) && FLINT_MLX_MATH
    float y = 1.0f / (1.0f + metal::precise::exp(abs(gate)));
    float sigmoid = gate < 0 ? y : 1.0f - y;
    return (gate * sigmoid) * up;
#else
    return (gate / (1.0f + exp(-gate))) * up;
#endif
}
kernel void swiglu(device const float *gate [[buffer(0)]], device const float *up [[buffer(1)]],
                   device float *out [[buffer(2)]], constant P &p [[buffer(8)]], uint i [[thread_position_in_grid]]) {
    if (i < p.n) { float g = gate[i]; out[i] = swiglu_product(g, up[i]); }
}
kernel void add_scaled(device float *x [[buffer(0)]], device const float *y [[buffer(1)]],
                       constant P &p [[buffer(8)]], uint i [[thread_position_in_grid]]) {
    if (i < p.n) x[i] += p.scale * y[i];
}

// GPU virtual addresses refer directly to resident checkpoint buffers. The host
// declares indirect resources once per token; no expert weights are copied.
struct Matrix {
    device const uchar *w;
    device const uchar *scales;
    device const uchar *biases;
    uint rows, cols, dtype, bits, group_size, scale_dtype, bias_dtype, reserved;
};
struct MLP { Matrix gate, up, down; };
inline float matrix_weight(Matrix m, ulong i) {
    if (!m.bits) return weight(m.w, 0, m.dtype, i);
    uint packed = m.w[i * m.bits / 8];
    uint code = (packed >> ((i * m.bits) % 8)) & ((1u << m.bits) - 1u);
    ulong group = i / m.group_size;
    return float(code) * weight(m.scales, 0, m.scale_dtype, group) + weight(m.biases, 0, m.bias_dtype, group);
}

inline float matrix_dot(Matrix m, device const float *x, uint row, uint lane) {
    if (m.reserved == 1) return dot4_g64(m.w, m.scales, m.biases, m.scale_dtype, m.bias_dtype, x, m.cols, row, lane);
    float sum = 0;
    for (uint col = lane; col < m.cols; col += 32) sum += matrix_weight(m, ulong(row) * m.cols + col) * x[col];
    return sum;
}

inline float router_sigmoid(float x) {
#if FLINT_MLX_MATH
    float y = 1.0f / (1.0f + metal::precise::exp(abs(x)));
    return x < 0 ? y : 1.0f - y;
#else
    return 1.0f / (1.0f + exp(-x));
#endif
}

// One SIMD group selects all routes, with deterministic low-index tie breaks.
// p.m = experts, p.n = top-k. Invalid logits are flagged for the single final CPU fence.
kernel void route_topk(device const float *logits [[buffer(0)]], device const float *bias [[buffer(1)]],
                       device uint *ids [[buffer(2)]], device float *weights [[buffer(3)]],
                       device atomic_uint *error [[buffer(4)]], constant P &p [[buffer(8)]],
                       uint lane [[thread_index_in_simdgroup]]) {
    float total = 0;
    for (uint k = 0; k < p.n; ++k) {
        float best = -INFINITY;
        uint best_id = p.m;
        for (uint e = lane; e < p.m; e += 32) {
            bool used = false;
            for (uint j = 0; j < k; ++j) used |= ids[j] == e;
            if (used) continue;
            float score = router_sigmoid(logits[e]);
            if (!isfinite(logits[e]) || !isfinite(bias[e])) {
                atomic_fetch_or_explicit(error, 1u, memory_order_relaxed);
                continue;
            }
            float choice = score + bias[e];
            if (choice > best || (choice == best && e < best_id)) { best = choice; best_id = e; }
        }
        float top = simd_max(best);
        uint winner = simd_min(best == top ? best_id : p.m);
        if (winner >= p.m) winner = 0; // Keep invalid-input execution in bounds until the final error check.
        float w = router_sigmoid(logits[winner]);
        if (!isfinite(w)) w = 0;
        if (lane == 0) { ids[k] = winner; weights[k] = w; }
        total += w;
        simdgroup_barrier(mem_flags::mem_device);
    }
    if (lane == 0) for (uint k = 0; k < p.n; ++k) weights[k] /= total + 1e-20f;
}

// Fuses selected-expert gather, both projections, and SwiGLU. The extra route
// is the always-on shared expert (or the single dense MLP when p.n == 0).
// p.n = routed count, p.m = shared descriptor index, p.k = activation row stride.
kernel void fused_gate_up(device const MLP *experts [[buffer(0)]], device const float *x [[buffer(1)]],
                          device const uint *ids [[buffer(2)]], device float *activation [[buffer(3)]],
                          constant P &p [[buffer(8)]], uint i [[thread_position_in_grid]],
                          uint lane [[thread_index_in_simdgroup]]) {
    uint flat = i / 32, route = flat / p.k, row = flat % p.k;
    if (route > p.n) return;
    uint expert = route < p.n ? ids[route] : p.m;
    device const MLP &mlp = experts[expert];
    if (row >= mlp.gate.rows) return;
    float gate = matrix_dot(mlp.gate, x, row, lane);
    float up = matrix_dot(mlp.up, x, row, lane);
    gate = simd_sum(gate); up = simd_sum(up);
    if (lane == 0) activation[route * p.k + row] = swiglu_product(gate, up);
}

// Fuses every selected down projection, mixture weighting, shared expert, and
// residual addition. A dispatch boundary supplies the required device-wide barrier.
kernel void fused_down_reduce(device const MLP *experts [[buffer(0)]], device const float *activation [[buffer(1)]],
                              device const uint *ids [[buffer(2)]], device const float *weights [[buffer(3)]],
                              device float *residual [[buffer(4)]], constant P &p [[buffer(8)]],
                              uint i [[thread_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
    uint row = i / 32;
    if (row >= p.dim) return;
#if defined(FLINT_MLX_MATH) && FLINT_MLX_MATH
    float output = 0;
#else
    float output = lane == 0 ? residual[row] : 0;
#endif
    for (uint route = 0; route <= p.n; ++route) {
        uint expert = route < p.n ? ids[route] : p.m;
        device const Matrix &down = experts[expert].down;
        float sum = matrix_dot(down, activation + route * p.k, row, lane);
        sum = simd_sum(sum);
        if (lane == 0) output += mixture_product(sum, route < p.n ? weights[route] : 1.0f);
    }
#if defined(FLINT_MLX_MATH) && FLINT_MLX_MATH
    if (lane == 0) residual[row] = residual[row] + output;
#else
    if (lane == 0) residual[row] = output;
#endif
}
