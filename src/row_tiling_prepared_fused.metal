// Append once after row_tiling_prepared.metal. Matched-math gate/up producer:
// retain the raw activation and prepare only its real rows for prepared down.
// ABI: experts=0, prepared norm=1, ids=2, raw activation=3, prepared=4, P=8.
// P is unchanged from gate/up: n selected routes, m shared expert, k raw stride.
// Raw stride must be a positive multiple of 64. Prepared route stride is k+k/16.
// Buffers must not overlap. Padding values AND padding sums remain untouched;
// prepared down may read only real columns. Host owns expert/id buffer bounds.
// Exactly 16 output rows per complete 1-D group, using 512/Rows threads and
// 16/Rows 32-lane SIMDs. All rejecting conditions are uniform before the barrier.
template<uint Rows>
inline void staged_gate_up_prepared_fused(device const MLP *experts,
                                         device const float *x, device const uint *ids,
                                         device float *raw, device float *prepared,
                                         constant P &p, uint tid, uint lane, uint simd,
                                         uint3 group_id, uint3 group, uint3 grid,
                                         uint simd_width, uint simds,
                                         threadgroup float *tile) {
    static_assert(Rows == 1 || Rows == 2 || Rows == 4 || Rows == 8, "Uncertified row tile");
#if !FLINT_MLX_MATH
    // The fused producer's quartet sum is the matched-math prepare contract.
    return;
#endif
    constexpr uint threads = 512 / Rows;
    if (group.x != threads || group.y != 1 || group.z != 1 ||
        grid.y != 1 || grid.z != 1 || grid.x % threads != 0 ||
        simd_width != 32 || simds != 16 / Rows ||
        p.k == 0 || p.k % 64 != 0) return;
    uint groups_per_route = p.k / 16;
    uint route = group_id.x / groups_per_route;
    if (route > p.n) return;
    uint first = (group_id.x % groups_per_route) * 16;
    uint expert = route < p.n ? ids[route] : p.m;
    Matrix gate = experts[expert].gate, up = experts[expert].up;
    if (!prepared_matrix_valid(gate) || !prepared_matrix_valid(up) ||
        gate.cols != up.cols || gate.rows != up.rows ||
        gate.rows > p.k || gate.rows % 16 != 0 || first >= gate.rows) return;

    float gates[Rows], ups[Rows];
    uint local_first = simd * Rows;
    matrix_gate_up_rows_prepared<Rows>(gate, up, x, first + local_first, lane, gates, ups);
    #pragma unroll
    for (uint r = 0; r < Rows; ++r) {
        float gate_value = simd_sum(gates[r]), up_value = simd_sum(ups[r]);
        if (lane == 0) {
            float value = swiglu_product(gate_value, up_value);
            raw[ulong(route) * p.k + first + local_first + r] = value;
            tile[local_first + r] = value;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    device float *out = prepared + ulong(route) * (ulong(p.k) + p.k / 16);
    if (tid < 16) {
        float value = tile[tid];
        // Keep the same compile-time divisors as prepare_quant_input.
        switch (tid % 4) {
            case 0: out[first + tid] = value; break;
            case 1: out[first + tid] = value / 16.0f; break;
            case 2: out[first + tid] = value / 256.0f; break;
            default: out[first + tid] = value / 4096.0f; break;
        }
    }
    if (tid == 0) {
        float input_sum = 0;
        #pragma unroll
        for (uint j = 0; j < 16; j += 4)
            input_sum += tile[j] + tile[j + 1] + tile[j + 2] + tile[j + 3];
        out[p.k + first / 16] = input_sum;
    }
}
#define STAGED_PREPARED_FUSED_ENTRY(Rows) \
kernel void staged_gate_up_prepared_fused_t##Rows( \
    device const MLP *experts [[buffer(0)]], device const float *x [[buffer(1)]], \
    device const uint *ids [[buffer(2)]], device float *raw [[buffer(3)]], \
    device float *prepared [[buffer(4)]], constant P &p [[buffer(8)]], \
    uint tid [[thread_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]], \
    uint simd [[simdgroup_index_in_threadgroup]], uint3 group_id [[threadgroup_position_in_grid]], \
    uint3 group [[threads_per_threadgroup]], uint3 grid [[threads_per_grid]], \
    uint simd_width [[threads_per_simdgroup]], uint simds [[simdgroups_per_threadgroup]]) { \
    threadgroup float tile[16]; \
    staged_gate_up_prepared_fused<Rows>(experts, x, ids, raw, prepared, p, tid, lane, simd, \
                                      group_id, group, grid, simd_width, simds, tile); \
}
STAGED_PREPARED_FUSED_ENTRY(1)
STAGED_PREPARED_FUSED_ENTRY(2)
STAGED_PREPARED_FUSED_ENTRY(4)
STAGED_PREPARED_FUSED_ENTRY(8)
#undef STAGED_PREPARED_FUSED_ENTRY
