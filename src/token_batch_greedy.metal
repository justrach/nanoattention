// Append after kernels.metal, resident_plan.metal, row_tiling.metal and
// token_batch.metal. No TokenScratch ABI change.
// buffer(0): at least P.m TokenScratch records; .logits addresses P.n f32s.
// buffer(1): at least 2*P.m uints: {lowest-ID argmax, any-nonfinite flag}.
// buffer(8): P, with m=batch in [1,32], n=vocabulary > 0.
// Launch EXACTLY batch*256 threads, 256 threads/group, one 1-D group/token.
// Requires 32-wide SIMD. All guards before the barrier are group-uniform.
// Allocation bounds, disjoint outputs and indirect logits residency are caller
// preconditions. Enqueue after logits, before finish; read results only AFTER
// finish and reject ANY nonzero flag before using ANY of the returned IDs.
#ifndef FLINT_TOKEN_BATCH_GREEDY_METAL
#define FLINT_TOKEN_BATCH_GREEDY_METAL

kernel void token_batch_greedy(device const TokenScratch *tokens [[buffer(0)]],
                               device uint *results [[buffer(1)]],
                               constant P &p [[buffer(8)]],
                               uint3 group [[threadgroup_position_in_grid]],
                               uint3 groups [[threadgroups_per_grid]],
                               uint3 size [[threads_per_threadgroup]],
                               uint tid [[thread_index_in_threadgroup]],
                               uint sg [[simdgroup_index_in_threadgroup]],
                               uint lane [[thread_index_in_simdgroup]],
                               uint width [[threads_per_simdgroup]]) {
    if (!token_batch_count(p.m) || p.n == 0 || width != 32 ||
        size.x != 256 || size.y != 1 || size.z != 1 ||
        groups.x != p.m || groups.y != 1 || groups.z != 1 ||
        group.x >= p.m || group.y != 0 || group.z != 0) return;
    device const float *logits = tokens[group.x].logits;
    if (!logits) return;

    // The ID sentinel distinguishes an empty lane from a real -FLT_MAX value.
    // Explicit equality/index comparisons preserve earliest ties, even +/-0.
    const float lowest = -3.4028234663852886e38f;
    const uint no_id = 0xffffffffu;
    float best = lowest;
    uint id = no_id;
    bool invalid = false;
    // Widen the stride so even a vocabulary near UINT_MAX cannot wrap.
    for (ulong i = tid; i < ulong(p.n); i += 256) {
        float value = logits[i];
        // Inspect raw exponents: do not let fast-math isfinite assumptions hide
        // a losing NaN/Inf. Invalid rows have no usable greedy ID.
        bool finite = (as_type<uint>(value) & 0x7f800000u) != 0x7f800000u;
        invalid = invalid || !finite;
        if (finite && (value > best || (value == best && uint(i) < id))) {
            best = value;
            id = uint(i);
        }
    }

    float simd_best = simd_max(best);
    uint simd_id = simd_min(best == simd_best ? id : no_id);
    bool simd_invalid = simd_any(invalid);
    threadgroup float values[8];
    threadgroup uint ids[8];
    threadgroup uint flags[8];
    if (lane == 0) {
        values[sg] = simd_best;
        ids[sg] = simd_id;
        flags[sg] = uint(simd_invalid);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sg == 0) {
        best = lane < 8 ? values[lane] : lowest;
        id = lane < 8 ? ids[lane] : no_id;
        invalid = lane < 8 ? flags[lane] != 0 : false;
        float maximum = simd_max(best);
        uint winner = simd_min(best == maximum ? id : no_id);
        bool any_invalid = simd_any(invalid);
        if (lane == 0) {
            results[2 * group.x] = winner;
            results[2 * group.x + 1] = uint(any_invalid);
        }
    }
}
#endif
