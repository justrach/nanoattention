// Concatenated after kernels.metal: P and weight() are shared with the staged path.
// Single vector only (p.m = 1). Dispatch exactly 256 threads in one threadgroup.
// x and out may alias. n is positive and at most 8192; weights may be F32/F16/BF16.
kernel void rmsnorm_parallel(device const uchar *w [[buffer(0)]],
                             device const float *x [[buffer(1)]],
                             device float *out [[buffer(2)]],
                             constant P &p [[buffer(8)]],
                             uint tid [[thread_index_in_threadgroup]],
                             uint lane [[thread_index_in_simdgroup]],
                             uint simd [[simdgroup_index_in_threadgroup]]) {
    threadgroup float partial[8];
    float sum = 0;
    for (uint j = tid; j < p.n; j += 256) {
        float v = x[j];
        sum += v * v;
    }
    sum = simd_sum(sum);
    if (lane == 0) partial[simd] = sum;
    // Also completes every input read before an in-place output can overwrite it.
    threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
    float total = simd_sum(lane < 8 ? partial[lane] : 0.0f);
    float inv = rsqrt(total / float(p.n) + p.eps);
    for (uint j = tid; j < p.n; j += 256)
        out[j] = x[j] * inv * weight(w, p.offset, p.dtype, j);
}

// Buffers: weights, residual (updated in place), projection, normalized output.
// The output must not alias residual/projection: the residual remains unnormalized.
// p.n = hidden dimension; same one-group/256-thread contract as rmsnorm_parallel.
kernel void residual_norm(device const uchar *w [[buffer(0)]],
                          device float *x [[buffer(1)]],
                          device const float *projection [[buffer(2)]],
                          device float *out [[buffer(3)]],
                          constant P &p [[buffer(8)]],
                          uint tid [[thread_index_in_threadgroup]],
                          uint lane [[thread_index_in_simdgroup]],
                          uint simd [[simdgroup_index_in_threadgroup]]) {
    threadgroup float partial[8];
    float sum = 0;
    for (uint j = tid; j < p.n; j += 256) {
        float v = x[j] + projection[j];
        x[j] = v;
        sum += v * v;
    }
    sum = simd_sum(sum);
    if (lane == 0) partial[simd] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
    float total = simd_sum(lane < 8 ? partial[lane] : 0.0f);
    float inv = rsqrt(total / float(p.n) + p.eps);
    for (uint j = tid; j < p.n; j += 256)
        out[j] = x[j] * inv * weight(w, p.offset, p.dtype, j);
}

// One SIMD per Q head followed by one SIMD per K head; padded head tails are safe.
// Dispatch (heads + kv_heads) * 32 threads, group = 32 or a multiple of 32.
// p.dim must be even, 2..256; capacity > 0. Q/K, V, and cache storage are disjoint.
// p.offset/dtype select Q norm weights; p.scale_offset/scale_dtype select K weights.
// p.bits != 0 enables split-half RoPE; p.pos is absolute (only cache uses modulo).
kernel void qk_rope_cache_fused(device float *q [[buffer(0)]],
                                device float *k [[buffer(1)]],
                                device const float *v [[buffer(2)]],
                                device const uchar *qw [[buffer(3)]],
                                device const uchar *kw [[buffer(4)]],
                                device float *keys [[buffer(5)]],
                                device float *values [[buffer(6)]],
                                constant P &p [[buffer(8)]],
                                uint i [[thread_position_in_grid]],
                                uint lane [[thread_index_in_simdgroup]]) {
    uint head = i / 32;
    if (head >= p.heads + p.kv_heads) return;
    bool query = head < p.heads;
    uint local_head = query ? head : head - p.heads;
    uint base = local_head * p.dim;
    device float *x = query ? q : k;
    device const uchar *w = query ? qw : kw;
    ulong offset = query ? p.offset : p.scale_offset;
    uint dtype = query ? p.dtype : p.scale_dtype;
    float sum = 0;
    // Keep the existing per-head RMSNorm lane accumulation/reduction order.
    for (uint col = lane; col < p.dim; col += 32) {
        float a = x[base + col];
        sum += a * a;
    }
    float inv = rsqrt(simd_sum(sum) / float(p.dim) + p.eps);
    uint half_dim = p.dim / 2;
    for (uint j = lane; j < half_dim; j += 32) {
        float a = x[base + j] * inv * weight(w, offset, dtype, j);
        float b = x[base + j + half_dim] * inv * weight(w, offset, dtype, j + half_dim);
        if (p.bits) {
            float angle = float(p.pos) * pow(p.theta, -2.0f * float(j) / float(p.dim));
            float c = cos(angle), s = sin(angle);
            float rotated_a = a * c - b * s;
            b = b * c + a * s;
            a = rotated_a;
        }
        x[base + j] = a;
        x[base + j + half_dim] = b;
        if (!query) {
            uint slot = (p.pos % p.capacity) * p.kv_heads * p.dim + base;
            keys[slot + j] = a;
            keys[slot + j + half_dim] = b;
            values[slot + j] = v[base + j];
            values[slot + j + half_dim] = v[base + j + half_dim];
        }
    }
}
