// Append after normalization_aligned.metal and row_tiling_prepared.metal.
// Production FLINT_MLX_MATH quartet order: one 2048-vector, one 512-thread
// group. Raw normalization and residual update retain the aligned kernels'
// arithmetic. Prepared output is [2048 prescaled values][128 segment sums].
// Prepared storage must be distinct from all inputs and the raw output.
inline void aligned_norm_store_prepared(thread const float *values,
                                        device float *prepared, uint tid, uint lane) {
    uint col = aligned_norm_reads * tid;
    prepared[col] = values[0];
    prepared[col + 1] = values[1] / 16.0f;
    prepared[col + 2] = values[2] / 256.0f;
    prepared[col + 3] = values[3] / 4096.0f;

    float quartet = values[0] + values[1] + values[2] + values[3];
    uint laneBase = lane & ~3u;
    // Every lane participates, including lanes that do not own a sum store.
    float quartet0 = simd_shuffle(quartet, laneBase);
    float quartet1 = simd_shuffle(quartet, laneBase + 1);
    float quartet2 = simd_shuffle(quartet, laneBase + 2);
    float quartet3 = simd_shuffle(quartet, laneBase + 3);
    float input_sum = 0;
    input_sum += quartet0;
    input_sum += quartet1;
    input_sum += quartet2;
    input_sum += quartet3;
    if (lane % 4 == 0) prepared[aligned_norm_hidden + tid / 4] = input_sum;
}

// Cached inputs permit x/rawout aliasing, as in rmsnorm_aligned.
kernel void rmsnorm_aligned_prepared(device const uchar *w [[buffer(0)]],
                                     device const float *x [[buffer(1)]],
                                     device float *rawout [[buffer(2)]],
                                     device float *prepared [[buffer(3)]],
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
    float values[aligned_norm_reads];
    for (uint j = 0; j < aligned_norm_reads; ++j) {
        uint col = aligned_norm_reads * tid + j;
        values[j] = cached[j] * inv * weight(w, p.offset, p.dtype, col);
        rawout[col] = values[j];
    }
    aligned_norm_store_prepared(values, prepared, tid, lane);
}

// Raw output, residual, projection and prepared storage must be distinct.
// Residual remains unnormalized; projection is read-only.
kernel void residual_norm_aligned_prepared(device const uchar *w [[buffer(0)]],
                                          device float *residual [[buffer(1)]],
                                          device const float *projection [[buffer(2)]],
                                          device float *rawout [[buffer(3)]],
                                          device float *prepared [[buffer(4)]],
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
    float values[aligned_norm_reads];
    for (uint j = 0; j < aligned_norm_reads; ++j) {
        uint col = aligned_norm_reads * tid + j;
        values[j] = cached[j] * inv * weight(w, p.offset, p.dtype, col);
        rawout[col] = values[j];
    }
    aligned_norm_store_prepared(values, prepared, tid, lane);
}
