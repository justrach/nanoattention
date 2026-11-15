// Split-K attention: independent token chunks expose work across the device.
// Each producer writes an unnormalized online-softmax state. Dispatch barriers,
// not a cross-threadgroup spin protocol, order producer -> merge -> projection.
kernel void attention_split(device const float *q [[buffer(0)]],
                            device const float *keys [[buffer(1)]],
                            device const float *values [[buffer(2)]],
                            device const uchar *sink [[buffer(3)]],
                            device float *states [[buffer(4)]],
                            constant P &p [[buffer(8)]],
                            uint3 group [[threadgroup_position_in_grid]],
                            uint sg [[simdgroup_index_in_threadgroup]],
                            uint lane [[thread_index_in_simdgroup]]) {
    uint head = group.x / p.k, chunk = group.x % p.k;
    if (head >= p.heads) return;
    uint start = p.window && p.pos + 1 > p.window ? p.pos + 1 - p.window : 0;
    uint begin = start + chunk * p.bits;
    uint end = min(p.pos + 1, begin + p.bits);
    uint kv_head = head / (p.heads / p.kv_heads);
    bool owns_sink = chunk == 0 && sg == 0;
    float maximum = owns_sink ? weight(sink, p.offset, p.dtype, head) : -INFINITY;
    float denom = owns_sink ? 1.0f : 0.0f;
    float acc[8] = {0,0,0,0,0,0,0,0};
    for (uint t = begin + sg; t < end; t += STAGED_ATTENTION_SIMDS) {
        uint base = (t % p.capacity) * p.kv_heads * p.dim + kv_head * p.dim;
        float dot = 0;
        for (uint d = lane; d < p.dim; d += STAGED_ATTENTION_LANES)
            dot += q[head * p.dim + d] * keys[base + d];
        float score = simd_sum(dot) * p.scale;
        float next_max = max(maximum, score);
        float old_scale = exp(maximum - next_max), prob = exp(score - next_max);
        denom = denom * old_scale + prob;
        for (uint d = lane, j = 0; d < p.dim; d += STAGED_ATTENTION_LANES, ++j)
            acc[j] = acc[j] * old_scale + prob * values[base + d];
        maximum = next_max;
    }
    threadgroup float maxima[STAGED_ATTENTION_SIMDS], denominators[STAGED_ATTENTION_SIMDS];
    threadgroup float partial[STAGED_ATTENTION_SIMDS * STAGED_ATTENTION_MAX_DIM];
    if (lane == 0) { maxima[sg] = maximum; denominators[sg] = denom; }
    for (uint d = lane, j = 0; d < p.dim; d += STAGED_ATTENTION_LANES, ++j)
        partial[sg * STAGED_ATTENTION_MAX_DIM + d] = acc[j];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg != 0) return;
    float merge_max = simd_max(lane < STAGED_ATTENTION_SIMDS ? maxima[lane] : -INFINITY);
    float factors[STAGED_ATTENTION_SIMDS], merge_denom = 0;
    for (uint g = 0; g < STAGED_ATTENTION_SIMDS; ++g) {
        factors[g] = denominators[g] > 0 ? exp(maxima[g] - merge_max) : 0;
        merge_denom += denominators[g] * factors[g];
    }
    uint state = (head * p.k + chunk) * (p.dim + 2);
    if (lane == 0) {
        states[state + p.dim] = merge_max;
        states[state + p.dim + 1] = merge_denom;
    }
    for (uint d = lane; d < p.dim; d += STAGED_ATTENTION_LANES) {
        float sum = 0;
        for (uint g = 0; g < STAGED_ATTENTION_SIMDS; ++g)
            sum += partial[g * STAGED_ATTENTION_MAX_DIM + d] * factors[g];
        states[state + d] = sum;
    }
}

// A single SIMD per head merges only states written by this token's producers.
// One global maximum, ascending chunk order, and exactly one normalization.
kernel void attention_merge(device const float *states [[buffer(0)]],
                            device float *out [[buffer(1)]], constant P &p [[buffer(8)]],
                            uint head [[threadgroup_position_in_grid]],
                            uint lane [[thread_index_in_simdgroup]]) {
    if (head >= p.heads) return;
    uint first = head * p.k * (p.dim + 2);
    float maximum = -INFINITY;
    for (uint chunk = lane; chunk < p.k; chunk += STAGED_ATTENTION_LANES)
        maximum = max(maximum, states[first + chunk * (p.dim + 2) + p.dim]);
    maximum = simd_max(maximum);
    float denom = 0, acc[8] = {0,0,0,0,0,0,0,0};
    for (uint chunk = 0; chunk < p.k; ++chunk) {
        uint state = first + chunk * (p.dim + 2);
        float z = states[state + p.dim + 1];
        float factor = z > 0 ? exp(states[state + p.dim] - maximum) : 0;
        denom += z * factor;
        for (uint d = lane, j = 0; d < p.dim; d += STAGED_ATTENTION_LANES, ++j)
            acc[j] += states[state + d] * factor;
    }
    for (uint d = lane, j = 0; d < p.dim; d += STAGED_ATTENTION_LANES, ++j)
        out[head * p.dim + d] = acc[j] / denom;
}
