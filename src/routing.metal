// One SIMD caches each expert's unbiased sigmoid and biased selection score.
// Register ownership/capacity comes from the Lean-generated binding. This keeps
// the original selection, tie-breaking and normalization order, but avoids
// recalculating sigmoid and rereading device IDs for every top-k iteration.
kernel void route_topk_cached(device const float *logits [[buffer(0)]],
                              device const float *bias [[buffer(1)]],
                              device uint *ids [[buffer(2)]],
                              device float *weights [[buffer(3)]],
                              device atomic_uint *error [[buffer(4)]],
                              constant P &p [[buffer(8)]],
                              uint lane [[thread_index_in_simdgroup]]) {
    static_assert(STAGED_ROUTER_LANES == 32, "Apple SIMD width contract");
    if (p.m > STAGED_CACHED_ROUTER_EXPERTS || p.n == 0 || p.n > p.m) {
        if (lane == 0) atomic_fetch_or_explicit(error, 1u, memory_order_relaxed);
        return;
    }
    float probabilities[STAGED_ROUTER_SLOTS], choices[STAGED_ROUTER_SLOTS];
    #pragma unroll
    for (uint slot = 0; slot < STAGED_ROUTER_SLOTS; ++slot) {
        uint expert = lane + slot * STAGED_ROUTER_LANES;
        probabilities[slot] = 0;
        choices[slot] = -INFINITY;
        if (expert < p.m) {
            float logit = logits[expert], adjustment = bias[expert];
            if (!isfinite(logit) || !isfinite(adjustment)) {
                atomic_fetch_or_explicit(error, 1u, memory_order_relaxed);
            } else {
                float score = 1.0f / (1.0f + exp(-logit));
                probabilities[slot] = score;
                choices[slot] = score + adjustment;
            }
        }
    }
    float total = 0;
    for (uint k = 0; k < p.n; ++k) {
        float best = -INFINITY;
        uint best_id = p.m;
        #pragma unroll
        for (uint slot = 0; slot < STAGED_ROUTER_SLOTS; ++slot) {
            uint expert = lane + slot * STAGED_ROUTER_LANES;
            float choice = choices[slot];
            if (expert < p.m && (choice > best || (choice == best && expert < best_id))) {
                best = choice;
                best_id = expert;
            }
        }
        float top = simd_max(best);
        uint winner = simd_min(best == top ? best_id : p.m);
        if (winner >= p.m) winner = 0; // Error cases remain bounded until host rejection.
        uint owner = winner % STAGED_ROUTER_LANES;
        uint slot = winner / STAGED_ROUTER_LANES;
        float local_probability = 0;
        #pragma unroll
        for (uint j = 0; j < STAGED_ROUTER_SLOTS; ++j) {
            if (j == slot) local_probability = probabilities[j];
            if (lane == owner && j == slot) choices[j] = -INFINITY;
        }
        float probability = simd_shuffle(local_probability, owner);
        if (lane == 0) { ids[k] = winner; weights[k] = probability; }
        total += probability;
    }
    if (lane == 0)
        for (uint k = 0; k < p.n; ++k) weights[k] /= total + 1e-20f;
}
