// Parallelize QK scores, but preserve the old eight-SIMD online value/softmax
// reduction verbatim. Reassociating complete chunk states changed long-prefix
// rounding enough to flip MoE expert choices, despite passing local tolerances.
kernel void attention_scores(device const float *q [[buffer(0)]],
                             device const float *keys [[buffer(1)]],
                             device float *scores [[buffer(2)]], constant P &p [[buffer(8)]],
                             uint3 group [[threadgroup_position_in_grid]],
                             uint sg [[simdgroup_index_in_threadgroup]],
                             uint lane [[thread_index_in_simdgroup]]) {
    uint head = group.x / p.k, chunk = group.x % p.k;
    if (head >= p.heads) return;
    uint start = p.window && p.pos + 1 > p.window ? p.pos + 1 - p.window : 0;
    uint length = p.pos + 1 - start;
    uint begin = start + chunk * p.bits;
    uint end = min(p.pos + 1, begin + p.bits);
    uint kv_head = head / (p.heads / p.kv_heads);
    for (uint t = begin + sg; t < end; t += STAGED_ATTENTION_SIMDS) {
        uint base = (t % p.capacity) * p.kv_heads * p.dim + kv_head * p.dim;
        float dot = 0;
        for (uint d = lane; d < p.dim; d += 32)
            dot += q[head * p.dim + d] * keys[base + d];
        float score = simd_sum(dot) * p.scale;
        if (lane == 0) scores[head * length + t - start] = score;
    }
}

kernel void attention_ordered_values(device const float *scores [[buffer(0)]],
                                     device const float *values [[buffer(1)]],
                                     device const uchar *sink [[buffer(2)]],
                                     device float *out [[buffer(3)]], constant P &p [[buffer(8)]],
                                     uint3 group [[threadgroup_position_in_grid]],
                                     uint sg [[simdgroup_index_in_threadgroup]],
                                     uint lane [[thread_index_in_simdgroup]]) {
    uint head = group.x;
    if (head >= p.heads) return; // Uniform across the entire threadgroup.
    uint kv_head = head / (p.heads / p.kv_heads);
    uint start = p.window && p.pos + 1 > p.window ? p.pos + 1 - p.window : 0;
    uint length = p.pos + 1 - start;
    // The learned sink has no value contribution, and belongs to exactly one
    // partition. Empty other partitions retain (-infinity, 0, zero vector).
    float maximum = sg == 0 ? weight(sink, p.offset, p.dtype, head) : -INFINITY;
    float denom = sg == 0 ? 1.0f : 0.0f;
    float acc[8] = {0,0,0,0,0,0,0,0};
    for (uint t = start + sg; t <= p.pos; t += 8) {
        uint base = (t % p.capacity) * p.kv_heads * p.dim + kv_head * p.dim;
        float score = scores[head * length + t - start];
        float next_max = max(maximum, score);
        float old_scale = exp(maximum - next_max), prob = exp(score - next_max);
        denom = denom * old_scale + prob;
        for (uint d = lane, j = 0; d < p.dim; d += 32, ++j)
            acc[j] = acc[j] * old_scale + prob * values[base + d];
        maximum = next_max;
    }

    threadgroup float maxima[8], denominators[8], partial[8 * 256];
    if (lane == 0) { maxima[sg] = maximum; denominators[sg] = denom; }
    for (uint d = lane, j = 0; d < p.dim; d += 32, ++j)
        partial[sg * 256 + d] = acc[j];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg != 0) return;

    float merge_max = simd_max(lane < 8 ? maxima[lane] : -INFINITY);
    float factors[8];
    float merge_denom = 0;
    for (uint g = 0; g < 8; ++g) {
        // Do not evaluate exp(-infinity - -infinity) for an idle partition.
        // With finite inputs the sink makes merge_max finite and denom > 0.
        factors[g] = denominators[g] > 0 ? exp(maxima[g] - merge_max) : 0;
        merge_denom += denominators[g] * factors[g];
    }
    for (uint d = lane; d < p.dim; d += 32) {
        float sum = 0;
        for (uint g = 0; g < 8; ++g) sum += partial[g * 256 + d] * factors[g];
        out[head * p.dim + d] = sum / merge_denom;
    }
}
