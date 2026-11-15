// Appended after kernels.metal: reuses P and the same matvec buffer ABI.
// One SIMD group per row, including when threadgroup size is 64 or 128.
// The kernel name selects the checkpoint dtype; p.dtype is not consulted.
inline float dense_matvec_value(float value) { return value; }
inline float dense_matvec_value(half value) { return float(value); }
inline float dense_matvec_value(ushort value) { return as_type<float>(uint(value) << 16); }

template<typename T>
inline void dense_matvec_row(device const T *w, device const float *x,
                             device float *out, constant P &p, uint row, uint lane) {
    if (row >= p.m) return;
    device const T *weights = w + ulong(row) * p.n;
    float sum = 0;
    uint blocks = p.n / 128;
    for (uint block = 0; block < blocks; ++block) {
        uint col = block * 128 + lane;
        // Four consecutive iterations of the old lane-strided loop. Keep one
        // accumulator and this exact += order, not four partial sums or dot().
        sum += dense_matvec_value(weights[col]) * x[col];
        sum += dense_matvec_value(weights[col + 32]) * x[col + 32];
        sum += dense_matvec_value(weights[col + 64]) * x[col + 64];
        sum += dense_matvec_value(weights[col + 96]) * x[col + 96];
    }
    for (uint col = blocks * 128 + lane; col < p.n; col += 32)
        sum += dense_matvec_value(weights[col]) * x[col];
    sum = simd_sum(sum);
    if (lane == 0) out[row] = sum;
}

// p.offset is bytes; F32 requires 4-byte alignment, F16/BF16 only 2-byte
// alignment. No vector load or padding requirement is imposed on odd rows.
kernel void dense_matvec_f32(device const uchar *w [[buffer(0)]],
                             device const float *x [[buffer(1)]], device float *out [[buffer(2)]],
                             constant P &p [[buffer(8)]], uint i [[thread_position_in_grid]],
                             uint lane [[thread_index_in_simdgroup]]) {
    dense_matvec_row((device const float *)(w + p.offset), x, out, p, i / 32, lane);
}
kernel void dense_matvec_f16(device const uchar *w [[buffer(0)]],
                             device const float *x [[buffer(1)]], device float *out [[buffer(2)]],
                             constant P &p [[buffer(8)]], uint i [[thread_position_in_grid]],
                             uint lane [[thread_index_in_simdgroup]]) {
    dense_matvec_row((device const half *)(w + p.offset), x, out, p, i / 32, lane);
}
kernel void dense_matvec_bf16(device const uchar *w [[buffer(0)]],
                              device const float *x [[buffer(1)]], device float *out [[buffer(2)]],
                              constant P &p [[buffer(8)]], uint i [[thread_position_in_grid]],
                              uint lane [[thread_index_in_simdgroup]]) {
    dense_matvec_row((device const ushort *)(w + p.offset), x, out, p, i / 32, lane);
}

// One-row-per-threadgroup path. Dispatch exactly 256 threads
// (eight 32-lane SIMD groups) per threadgroup, with one group per output row.
// Every thread in an excess group returns together, before the barrier.
template<typename T>
inline void dense_matvec_split_row(device const T *w, device const float *x,
                                   device float *out, constant P &p, uint row,
                                   uint lane, uint simd, threadgroup float *partials) {
    static_assert(STAGED_DENSE_SIMDS * STAGED_ROUTER_LANES == RESIDENT_THREADGROUP_SIZE,
                  "Dense ownership certificate must match the launch geometry");
    if (row >= p.m) return;
    device const T *weights = w + ulong(row) * p.n;
    float sum = 0;
    for (uint col = simd * STAGED_ROUTER_LANES + lane; col < p.n; col += RESIDENT_THREADGROUP_SIZE)
        sum += dense_matvec_value(weights[col]) * x[col];
    sum = simd_sum(sum);
    if (lane == 0) partials[simd] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd == 0) {
        float total = lane < STAGED_DENSE_SIMDS ? partials[lane] : 0.0f;
        total = simd_sum(total);
        if (lane == 0) out[row] = total;
    }
}

kernel void dense_matvec_split_f32(device const uchar *w [[buffer(0)]],
                                   device const float *x [[buffer(1)]], device float *out [[buffer(2)]],
                                   constant P &p [[buffer(8)]], uint3 group [[threadgroup_position_in_grid]],
                                   uint lane [[thread_index_in_simdgroup]],
                                   uint simd [[simdgroup_index_in_threadgroup]]) {
    threadgroup float partials[STAGED_DENSE_SIMDS];
    dense_matvec_split_row((device const float *)(w + p.offset), x, out, p, group.x, lane, simd, partials);
}
kernel void dense_matvec_split_f16(device const uchar *w [[buffer(0)]],
                                   device const float *x [[buffer(1)]], device float *out [[buffer(2)]],
                                   constant P &p [[buffer(8)]], uint3 group [[threadgroup_position_in_grid]],
                                   uint lane [[thread_index_in_simdgroup]],
                                   uint simd [[simdgroup_index_in_threadgroup]]) {
    threadgroup float partials[STAGED_DENSE_SIMDS];
    dense_matvec_split_row((device const half *)(w + p.offset), x, out, p, group.x, lane, simd, partials);
}
kernel void dense_matvec_split_bf16(device const uchar *w [[buffer(0)]],
                                    device const float *x [[buffer(1)]], device float *out [[buffer(2)]],
                                    constant P &p [[buffer(8)]], uint3 group [[threadgroup_position_in_grid]],
                                    uint lane [[thread_index_in_simdgroup]],
                                    uint simd [[simdgroup_index_in_threadgroup]]) {
    threadgroup float partials[STAGED_DENSE_SIMDS];
    dense_matvec_split_row((device const ushort *)(w + p.offset), x, out, p, group.x, lane, simd, partials);
}
