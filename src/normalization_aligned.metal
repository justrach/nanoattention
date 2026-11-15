// Append after kernels.metal + normalization.metal (shared P and weight()).
// Candidate MLX 0.32.4 adjacent-four FP32 RMS reduction for Flint only.
constant uint aligned_norm_threads = STAGED_ALIGNED_NORM_THREADS;
constant uint aligned_norm_reads = STAGED_ALIGNED_NORM_ELEMENTS_PER_THREAD;
constant uint aligned_norm_hidden = STAGED_ALIGNED_NORM_DIM;
constant uint aligned_norm_head = STAGED_ALIGNED_HEAD_DIM;
static_assert(STAGED_ALIGNED_NORM_THREADS == 512 && STAGED_ALIGNED_NORM_ELEMENTS_PER_THREAD == 4 &&
    STAGED_ALIGNED_NORM_DIM == 2048 && STAGED_ALIGNED_HEAD_DIM == 128 &&
    STAGED_ALIGNED_HEAD_THREADS == 32 && STAGED_ALIGNED_HEAD_ELEMENTS_PER_THREAD == 4,
    "Aligned normalization certificate geometry mismatch");

inline float aligned_norm_inverse(float sum, threadgroup float *partial,
                                 threadgroup float &inverse, uint tid,
                                 uint lane, uint simd, float eps) {
    // Sixteen SIMD partials, then a full zero-padded SIMD reduction.
    if (tid < 32) partial[tid] = 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    sum = simd_sum(sum);
    if (lane == 0) partial[simd] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd == 0) {
        float total = simd_sum(partial[lane]);
        if (lane == 0) inverse = precise::rsqrt(total / float(aligned_norm_hidden) + eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return inverse;
}

// One vector, exactly one 512-thread (16-SIMD) group, n = 2048.
// Cached inputs allow x/out aliasing. Weight offsets are byte offsets.
kernel void rmsnorm_aligned(device const uchar *w [[buffer(0)]],
                            device const float *x [[buffer(1)]],
                            device float *out [[buffer(2)]],
                            constant P &p [[buffer(8)]],
                            uint tid [[thread_index_in_threadgroup]],
                            uint lane [[thread_index_in_simdgroup]],
                            uint simd [[simdgroup_index_in_threadgroup]],
                            uint3 group [[threads_per_threadgroup]]) {
    if (p.n != aligned_norm_hidden || group.x != aligned_norm_threads || group.y != 1 || group.z != 1) return;
    threadgroup float partial[32], inverse;
    float cached[aligned_norm_reads];
    float sum = 0.0f;
    for (uint j = 0; j < aligned_norm_reads; ++j) {
        float a = x[aligned_norm_reads * tid + j];
        cached[j] = a;
        sum += a * a;
    }
    float inv = aligned_norm_inverse(sum, partial, inverse, tid, lane, simd, p.eps);
    for (uint j = 0; j < aligned_norm_reads; ++j) {
        uint col = aligned_norm_reads * tid + j;
        out[col] = cached[j] * inv * weight(w, p.offset, p.dtype, col);
    }
}

// Same fixed dispatch as rmsnorm_aligned. Output must not alias residual or
// projection; the updated residual stays unnormalized, projection is read-only.
kernel void residual_norm_aligned(device const uchar *w [[buffer(0)]],
                                 device float *residual [[buffer(1)]],
                                 device const float *projection [[buffer(2)]],
                                 device float *out [[buffer(3)]],
                                 constant P &p [[buffer(8)]],
                                 uint tid [[thread_index_in_threadgroup]],
                                 uint lane [[thread_index_in_simdgroup]],
                                 uint simd [[simdgroup_index_in_threadgroup]],
                                 uint3 group [[threads_per_threadgroup]]) {
    if (p.n != aligned_norm_hidden || group.x != aligned_norm_threads || group.y != 1 || group.z != 1) return;
    threadgroup float partial[32], inverse;
    float cached[aligned_norm_reads];
    float sum = 0.0f;
    for (uint j = 0; j < aligned_norm_reads; ++j) {
        uint col = aligned_norm_reads * tid + j;
        float a = residual[col] + projection[col];
        cached[j] = a;
        residual[col] = a;
        sum += a * a;
    }
    float inv = aligned_norm_inverse(sum, partial, inverse, tid, lane, simd, p.eps);
    for (uint j = 0; j < aligned_norm_reads; ++j) {
        uint col = aligned_norm_reads * tid + j;
        out[col] = cached[j] * inv * weight(w, p.offset, p.dtype, col);
    }
}

// Same ABI, head padding, normalized pair/RoPE/cache write order as the fused
// legacy kernel. The dim-128 path also matches MLX's host-log2/exp2 RoPE phase;
// p.theta is the host-computed FP32 log2(base), not base, in this entry point.
kernel void qk_rope_cache_aligned(device float *q [[buffer(0)]],
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
    if (p.dim != aligned_norm_head || head >= p.heads + p.kv_heads) return;
    bool query = head < p.heads;
    uint local_head = query ? head : head - p.heads;
    uint base = local_head * p.dim;
    device float *x = query ? q : k;
    device const uchar *w = query ? qw : kw;
    ulong offset = query ? p.offset : p.scale_offset;
    uint dtype = query ? p.dtype : p.scale_dtype;
    float sum = 0.0f;
    for (uint j = 0; j < aligned_norm_reads; ++j) {
        float a = x[base + aligned_norm_reads * lane + j];
        sum += a * a;
    }
    float inv = precise::rsqrt(simd_sum(sum) / float(p.dim) + p.eps);
    uint half_dim = p.dim / 2;
    for (uint j = lane; j < half_dim; j += 32) {
        float a = x[base + j] * inv * weight(w, offset, dtype, j);
        float b = x[base + j + half_dim] * inv * weight(w, offset, dtype, j + half_dim);
        if (p.bits) {
            float fraction = float(j) / float(half_dim);
            float frequency = exp2(-fraction * p.theta);
            float angle = float(p.pos) * frequency;
            float c = fast::cos(angle), s = fast::sin(angle);
            float rotated_a = a * c - b * s;
            b = a * s + b * c;
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
