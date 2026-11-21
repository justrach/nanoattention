// Inkling-2 loop and attention reference kernels; MK and helpers are declared in kernels.metal.

kernel void mk_depth_attention(device float *arena [[buffer(0)]], device const uchar *query [[buffer(1)]],
                               constant MK &p [[buffer(8)]], uint tid [[thread_index_in_threadgroup]],
                               uint group [[threadgroup_position_in_grid]],
                               uint lane [[thread_index_in_simdgroup]], uint simd [[simdgroup_index_in_threadgroup]]) {
    uint row = p.row0 + group;
    threadgroup float partials[8][8];
    threadgroup float weights[8];
    for (uint c = 0; c < p.heads; ++c) {
        float acc = 0.0f;
        for (uint d = tid; d < p.n; d += 256) {
            float q = mk_load(query, p.w, p.dtype, d);
            float v = c < p.m
                ? arena[p.x + row * p.xs + c * p.n + d]
                : arena[p.y + row * p.ys + d];
            acc += q * v;
        }
        float sum = simd_sum(acc);
        if (lane == 0) partials[simd][c] = sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float maximum = -INFINITY;
        float logits[8];
        for (uint c = 0; c < p.heads; ++c) {
            float dot = 0.0f;
            for (uint s = 0; s < 8; ++s) dot += partials[s][c];
            logits[c] = dot * p.scale;
            maximum = max(maximum, logits[c]);
        }
        float denominator = 0.0f;
        for (uint c = 0; c < p.heads; ++c) {
            weights[c] = metal::precise::exp(logits[c] - maximum);
            denominator += weights[c];
        }
        for (uint c = 0; c < p.heads; ++c) weights[c] /= denominator;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint d = tid; d < p.n; d += 256) {
        float value = 0.0f;
        for (uint c = 0; c < p.heads; ++c) {
            float state = c < p.m
                ? arena[p.x + row * p.xs + c * p.n + d]
                : arena[p.y + row * p.ys + d];
            value += weights[c] * state;
        }
        arena[p.z + row * p.zs + d] = value;
    }
}

kernel void mk_relative_attention(device float *arena [[buffer(0)]], device const float *K [[buffer(1)]],
                                  device const float *V [[buffer(2)]], constant MK &p [[buffer(8)]],
                                  uint lane [[thread_index_in_threadgroup]], uint group [[threadgroup_position_in_grid]]) {
    uint r = group / p.heads, h = group % p.heads;
    if (r >= p.rows) return;
    uint row = p.row0 + r;
    uint position = p.pos + row;
    uint kvh = h / (p.heads / p.kv_heads);
    uint first = p.window && position + 1 > p.window ? position + 1 - p.window : 0;
    uint E = (p.dim + 31) / 32;
    float q[8], acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    device const float *qp = arena + p.x + row * p.xs + h * p.dim;
    for (uint e = 0; e < E; ++e) {
        uint d = lane + 32 * e;
        q[e] = d < p.dim ? qp[d] : 0.0f;
    }
    float maximum = arena[p.s + row * p.every + h];
    float denominator = 1.0f;
    uint slot = first % p.capacity;
    for (uint t = first; t <= position; ++t) {
        ulong base = ulong(slot) * (p.kv_heads * p.dim) + kvh * p.dim;
        if (++slot == p.capacity) slot = 0;
        float dot = 0.0f;
        for (uint e = 0; e < E; ++e) {
            uint d = lane + 32 * e;
            if (d < p.dim) dot += q[e] * K[base + d];
        }
        dot = simd_sum(dot);
        uint distance = position - t;
        float relative = distance < p.m ? arena[p.z + row * p.zs + h * p.m + distance] : 0.0f;
        float score = dot * p.scale + relative;
        float next_maximum = max(maximum, score);
        float old_weight = metal::precise::exp(maximum - next_maximum);
        float key_weight = metal::precise::exp(score - next_maximum);
        denominator = denominator * old_weight + key_weight;
        for (uint e = 0; e < E; ++e) {
            uint d = lane + 32 * e;
            if (d < p.dim) acc[e] = acc[e] * old_weight + key_weight * V[base + d];
        }
        maximum = next_maximum;
    }
    device float *out = arena + p.y + row * p.ys + h * p.dim;
    for (uint e = 0; e < E; ++e) {
        uint d = lane + 32 * e;
        if (d < p.dim) out[d] = acc[e] / denominator;
    }
}


// 32 queries share each 16-key tile. This path uses FP16 fragments and
// FP32 online normalization; the denominator-only sink starts the reduction.
kernel void mk_relative_attention_prefill(device float *arena [[buffer(0)]], device const float *keys [[buffer(1)]],
                                          device const float *values [[buffer(2)]], constant MK &p [[buffer(8)]],
                                          uint group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                                          uint lane [[thread_index_in_simdgroup]], uint simd [[simdgroup_index_in_threadgroup]]) {
    constexpr uint D = 128, Q = 32, B = 16, Ks = B + 8, Vs = D + 8;
    const uint h = group % p.heads, kv = h / (p.heads / p.kv_heads);
    const uint q0 = ((p.rows + Q - 1) / Q - 1 - group / p.heads) * Q;
    const uint start = p.pos + p.row0, end = start + p.rows;
    const uint earliest = p.window ? start + 1 - min(start + 1, p.window) : 0;
    const uint qfirst = start + q0, qlast = start + min(q0 + Q, p.rows) - 1;
    const uint lower = p.window ? qfirst + 1 - min(qfirst + 1, p.window) : 0;
    const uint fr = ((lane / 4) & 4) + (lane / 2) % 4;
    const uint fc = (((lane / 4) & 2) + lane % 2) * 2;
    const uint local = q0 + 8 * simd + fr, row = p.row0 + local, pos = start + local;
    const bool active = local < p.rows;
    simdgroup_half8x8 q_frag[D/8];
    simdgroup_float8x8 out[D/8];
    #pragma unroll
    for (uint j = 0; j < D/8; ++j) {
        for (uint e = 0; e < 2; ++e)
            q_frag[j].thread_elements()[e] = active ? half(arena[p.x + row * p.xs + h * D + j * 8 + fc + e]) : 0.0h;
        out[j] = make_filled_simdgroup_matrix<float,8,8>(0);
    }
    threadgroup half kt[D * Ks], vt[B * Vs];
    float maximum = active ? arena[p.s + row * p.every + h] * M_LOG2E_F : 0;
    float denominator = 1;
    const float scale = p.scale * M_LOG2E_F;
    for (uint block = earliest + (lower - earliest) / B * B; block <= qlast; block += B) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const uint t = block + tid / 8, col = (tid % 8) * 16;
        const ulong base = (ulong(t % p.capacity) * p.kv_heads + kv) * D + col;
        #pragma unroll
        for (uint j = 0; j < 16; ++j) {
            kt[(col + j) * Ks + tid / 8] = t < end ? half(keys[base + j]) : 0.0h;
            vt[(tid / 8) * Vs + col + j] = t < end ? half(values[base + j]) : 0.0h;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (q0 + simd * 8 >= p.rows) continue;
        simdgroup_float8x8 scores[B/8];
        simdgroup_half8x8 probs[B/8];
        float bound = -INFINITY;
        #pragma unroll
        for (uint j = 0; j < B/8; ++j) {
            scores[j] = make_filled_simdgroup_matrix<float,8,8>(0);
            #pragma unroll
            for (uint k = 0; k < D/8; ++k) {
                simdgroup_half8x8 wk;
                simdgroup_load(wk, kt + 8 * (k * Ks + j), Ks);
                simdgroup_multiply_accumulate(scores[j], q_frag[k], wk, scores[j]);
            }
            #pragma unroll
            for (uint e = 0; e < 2; ++e) {
                const uint key = block + j * 8 + fc + e;
                const uint distance = pos - key;
                float score = -INFINITY;
                if (active && key <= pos && (p.window == 0 || distance < p.window)) {
                    score = scores[j].thread_elements()[e] * scale;
                    if (distance < p.m) score += arena[p.z + row * p.zs + h * p.m + distance] * M_LOG2E_F;
                }
                scores[j].thread_elements()[e] = score;
                bound = max(bound, score);
            }
        }
        bound = max(bound, simd_shuffle_xor(bound, ushort(1)));
        bound = max(bound, simd_shuffle_xor(bound, ushort(8)));
        bound = max(bound, maximum);
        const float rescale = fast::exp2(maximum - bound);
        maximum = bound;
        float weight_sum = 0;
        #pragma unroll
        for (uint j = 0; j < B/8; ++j) for (uint e = 0; e < 2; ++e) {
            float w = fast::exp2(scores[j].thread_elements()[e] - bound);
            weight_sum += w;
            probs[j].thread_elements()[e] = half(w);
        }
        weight_sum += simd_shuffle_xor(weight_sum, ushort(1));
        weight_sum += simd_shuffle_xor(weight_sum, ushort(8));
        denominator = rescale * denominator + weight_sum;
        #pragma unroll
        for (uint j = 0; j < D/8; ++j) {
            out[j].thread_elements()[0] *= rescale;
            out[j].thread_elements()[1] *= rescale;
            #pragma unroll
            for (uint k = 0; k < B/8; ++k) {
                simdgroup_half8x8 vv;
                simdgroup_load(vv, vt + k * 8 * Vs + j * 8, Vs);
                simdgroup_multiply_accumulate(out[j], probs[k], vv, out[j]);
            }
        }
    }
    if (active) for (uint j = 0; j < D/8; ++j) for (uint e = 0; e < 2; ++e)
        arena[p.y + row * p.ys + h * D + j * 8 + fc + e] = out[j].thread_elements()[e] / denominator;
}

kernel void mk_route_topk(device float *arena [[buffer(0)]], device const uchar *correction [[buffer(1)]],
                           constant MK &p [[buffer(8)]], uint lane [[thread_index_in_simdgroup]],
                           uint simd [[simdgroup_index_in_threadgroup]], uint group [[threadgroup_position_in_grid]]) {
    uint local_row = group * 8 + simd;
    if (local_row >= p.rows) return;
    uint row = p.row0 + local_row;
    float score[4], biased[4];
    for (uint i = 0; i < 4; ++i) {
        uint expert = lane + 32 * i;
        float raw = mk_sigmoid(arena[p.x + row * p.xs + expert]);
        score[i] = raw;
        biased[i] = raw + mk_load(correction, p.s, 0, expert);
    }
    float chosen[32];
    uint winners[32];
    for (uint k = 0; k < p.m; ++k) {
        float best = -INFINITY;
        uint best_expert = p.n;
        for (uint i = 0; i < 4; ++i) {
            uint expert = lane + 32 * i;
            if (biased[i] > best || (biased[i] == best && expert < best_expert)) {
                best = biased[i];
                best_expert = expert;
            }
        }
        float top = simd_max(best);
        uint winner = simd_min(best == top ? best_expert : p.n);
        float selected_score = 0.0f;
        for (uint i = 0; i < 4; ++i) {
            uint expert = lane + 32 * i;
            if (expert == winner) {
                selected_score = score[i];
                biased[i] = -INFINITY;
            }
        }
        chosen[k] = simd_sum(selected_score);
        winners[k] = winner;
    }
    if (lane == 0) {
        float total = 0.0f;
        for (uint k = 0; k < p.m; ++k) total = chosen[k] + total;
        for (uint k = 0; k < p.m; ++k) {
            arena[p.y + row * p.ys + k] = float(winners[k]);
            arena[p.y + row * p.ys + p.m + k] = chosen[k] / (total + p.eps) * p.scale;
        }
    }
}

kernel void mk_inkling_message_merge(device float *arena [[buffer(0)]], constant MK &p [[buffer(8)]],
                                     uint i [[thread_position_in_grid]]) {
    uint r = i / p.n, c = i % p.n;
    if (r >= p.rows) return;
    uint row = p.row0 + r;
    float hidden = arena[p.x + row * p.xs + c];
    float message = arena[p.y + row * p.ys + c];
    float forget = tanh(arena[p.z + row * p.zs + c]);
    float gain = tanh(arena[p.z + row * p.zs + p.n + c]);
    float output = (1.0f - forget) * hidden + gain * message;
    if (p.flags & MK_ROUND_BF16) output = mk_bf16(output);
    arena[p.w + row * p.every + c] = output;
}

kernel void mk_inkling_dynamic_conv(device const float *ring [[buffer(0)]], device const uchar *w [[buffer(1)]],
                                      device float *arena [[buffer(2)]], constant MK &p [[buffer(8)]],
                                      uint i [[thread_position_in_grid]]) {
    uint r = i / p.n, c = i % p.n;
    if (r >= p.rows) return;
    uint row = p.row0 + r;
    long position = long(p.pos) + row;
    uint channels_per_head = p.n / p.heads;
    uint head = c / channels_per_head;
    float acc = 0.0f;
    for (uint k = 0; k < p.taps; ++k) {
        long q = position - long(k);
        float value = q < 0 ? 0.0f : ring[ulong(q % p.capacity) * p.n + c];
        float static_tap = mk_load(w, p.w, p.dtype, ulong(k) * p.n + c);
        float dynamic_tap = arena[p.x + row * p.xs + head * p.taps + k];
        acc += value * (static_tap + dynamic_tap);
    }
    float residual = arena[p.y + row * p.ys + c];
    float output = (acc + residual) * p.scale;
    if (p.flags & MK_ROUND_BF16) output = mk_bf16(output);
    arena[p.z + row * p.zs + c] = output;
}

kernel void mk_inkling_short_conv(device const float *ring [[buffer(0)]], device const uchar *w [[buffer(1)]],
                                   device float *arena [[buffer(2)]], constant MK &p [[buffer(8)]],
                                   uint i [[thread_position_in_grid]]) {
    uint r = i / p.n, c = i % p.n;
    if (r >= p.rows) return;
    uint row = p.row0 + r;
    long position = long(p.pos) + row;
    float acc = 0.0f;
    for (uint k = 0; k < p.taps; ++k) {
        long q = position - long(p.taps) + 1 + k;
        float value = q < 0 ? 0.0f : ring[ulong(q % p.capacity) * p.n + c];
        if (p.dtype == 1) value = float(half(value));
        else if (p.dtype == 2) value = mk_bf16(value);
        acc += mk_load(w, p.w, p.dtype, ulong(c) * p.taps + k) * value;
    }
    if (p.dtype == 1) acc = float(half(acc));
    else if (p.dtype == 2) acc = mk_bf16(acc);
    float residual = arena[p.x + row * p.xs + c];
    float output = (acc + residual) * p.scale;
    if (p.flags & MK_ROUND_BF16) output = mk_bf16(output);
    arena[p.z + row * p.zs + c] = output;
}

kernel void mk_l2_clip(device float *arena [[buffer(0)]], device const uchar *scale [[buffer(1)]],
                       constant MK &p [[buffer(8)]], uint tid [[thread_index_in_threadgroup]],
                       uint simd [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
                       uint group [[threadgroup_position_in_grid]]) {
    uint r = group / p.heads, h = group % p.heads;
    if (r >= p.rows) return;
    uint row = p.row0 + r;
    device const float *x = arena + p.x + row * p.xs + h * p.n;
    float sum = 0.0f;
    for (uint d = tid; d < p.n; d += 256) {
        float value = x[d];
        if (p.flags & MK_HAS_WEIGHT) value *= mk_load(scale, p.w, p.dtype, ulong(h) * p.n + d);
        sum += value * value;
    }
    sum = simd_sum(sum);
    threadgroup float partial[8];
    threadgroup float factor;
    if (lane == 0) partial[simd] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd == 0) {
        float total = simd_sum(lane < 8 ? partial[lane] : 0.0f);
        if (lane == 0) factor = min(p.eps / (sqrt(total) + 1e-6f), 1.0f);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    device float *out = arena + p.z + row * p.zs + h * p.n;
    for (uint d = tid; d < p.n; d += 256) {
        float value = x[d];
        if (p.flags & MK_HAS_WEIGHT) value *= mk_load(scale, p.w, p.dtype, ulong(h) * p.n + d);
        float result = value * factor;
        out[d] = (p.flags & MK_ROUND_OUT) ? mk_bf16(result) : result;
    }
}

kernel void mk_inkling_softcapped_swiglu(device float *arena [[buffer(0)]], constant MK &p [[buffer(8)]],
                                         uint i [[thread_position_in_grid]]) {
    uint r = i / p.n, c = i % p.n;
    if (r >= p.rows) return;
    uint row = p.row0 + r;
    float gate = arena[p.x + row * p.xs + c];
    float up = arena[p.y + row * p.ys + c];
    float gate_cap = tanh(gate / p.eps) * p.eps;
    float up_cap = tanh(up / p.scale) * p.scale;
    float value = gate_cap * mk_sigmoid(gate) * up_cap;
    if (p.flags & MK_ROUND_BF16) value = mk_bf16(value);
    arena[p.z + row * p.zs + c] = value;
}

// Reference packed expert projections. Every output owns a fixed ascending-k
// sum, independent of token batching. No full-size dequantized tensor exists.
struct MKExpert {
    ulong w, s, amax, denominator;
    uint rows, row0, x, xs, routes, rs, y, ys;
    uint k, n, experts, top_k, nvfp4, round_bf16;
    float gate_cap, up_cap;
};

inline float mk_fp8_e4m3(uchar code) {
    int exponent = (code >> 3) & 15;
    uint mantissa = code & 7;
    if (exponent == 15 && mantissa == 7) return as_type<float>(0x7fc00000u);
    float value = exponent == 0 ? ldexp(float(mantissa), -9) : ldexp(1.0f + float(mantissa) * 0.125f, exponent - 7);
    return code & 128 ? -value : value;
}

inline float mk_fp4_e2m1(uint code) {
    constexpr float mag[] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
    return code & 8 ? -mag[code & 7] : mag[code & 7];
}

inline float mk_expert_dot(device const float *x, device const uchar *w, device const uchar *s,
                           device const uchar *amax, device const uchar *denom,
                           constant MKExpert &p, uint expert, ulong matrix_row) {
    const ulong base = matrix_row * p.k;
    float accum = 0.0f;
    if (p.nvfp4) {
        for (uint j = 0; j < p.k; ++j) {
            uint code = (w[p.w + (base + j) / 2] >> ((j & 1) * 4)) & 15;
            float scale = mk_fp8_e4m3(s[p.s + (base + j) / 16]);
            accum = fma(x[j], mk_fp4_e2m1(code) * scale, accum);
        }
        // Checkpoint encodes the per-expert factor as amax / denominator.
        accum *= mk_load(amax, p.amax, 0, expert) / mk_load(denom, p.denominator, 0, 0);
    } else {
        for (uint j = 0; j < p.k; ++j) {
            uint scale = s[p.s + (base + j) / 32];
            float weight = scale == 255 ? as_type<float>(0x7fc00000u) : ldexp(mk_fp8_e4m3(w[p.w + base + j]), int(scale) - 127);
            accum = fma(x[j], weight, accum);
        }
    }
    return p.round_bf16 ? mk_bf16(accum) : accum;
}

kernel void mk_expert_gate_up(device float *arena [[buffer(0)]], device const uchar *w [[buffer(1)]],
                              device const uchar *s [[buffer(2)]], device const uchar *amax [[buffer(3)]],
                              device const uchar *denom [[buffer(4)]], constant MKExpert &p [[buffer(8)]],
                              uint i [[thread_position_in_grid]]) {
    uint r = i / (p.top_k * p.n), slot = (i / p.n) % p.top_k, j = i % p.n;
    if (r >= p.rows) return;
    r += p.row0;
    uint expert = uint(arena[p.routes + r * p.rs + slot]);
    float value = 0.0f;
    if (expert < p.experts) {
        ulong base = (ulong(expert) * p.n + j) * 2;
        device const float *x = arena + p.x + r * p.xs;
        float gate = mk_expert_dot(x, w, s, amax, denom, p, expert, base);
        float up = mk_expert_dot(x, w, s, amax, denom, p, expert, base + 1);
        value = ((tanh(gate / p.gate_cap) * p.gate_cap) * mk_sigmoid(gate)) * (tanh(up / p.up_cap) * p.up_cap);
    }
    arena[p.y + r * p.ys + slot * p.n + j] = p.round_bf16 ? mk_bf16(value) : value;
}

kernel void mk_expert_down(device float *arena [[buffer(0)]], device const uchar *w [[buffer(1)]],
                           device const uchar *s [[buffer(2)]], device const uchar *amax [[buffer(3)]],
                           device const uchar *denom [[buffer(4)]], constant MKExpert &p [[buffer(8)]],
                           uint i [[thread_position_in_grid]]) {
    uint r = i / p.n, j = i % p.n;
    if (r >= p.rows) return;
    r += p.row0;
    device const float *routes = arena + p.routes + r * p.rs;
    float value = 0.0f;
    for (uint slot = 0; slot < p.top_k; ++slot) {
        uint expert = uint(routes[slot]);
        if (expert >= p.experts) continue;
        device const float *x = arena + p.x + r * p.xs + slot * p.k;
        float dot = mk_expert_dot(x, w, s, amax, denom, p, expert, ulong(expert) * p.n + j);
        value += dot * routes[p.top_k + slot];
    }
    arena[p.y + r * p.ys + j] = p.round_bf16 ? mk_bf16(value) : value;
}

kernel void mk_row_transform(device float *arena [[buffer(0)]], constant MK &p [[buffer(8)]],
                              uint i [[thread_position_in_grid]]) {
    uint r = i / p.n, c = i % p.n;
    if (r >= p.rows) return;
    uint row = p.row0 + r;
    float x = arena[p.x + row * p.xs + c];
    if (p.kind == 1) x = tanh(x / p.scale) * p.scale;
    if (p.kind == 2) x *= 1.0f + p.scale * metal::precise::log(max(float(p.pos + row + 1) / float(p.m), 1.0f));
    arena[p.y + row * p.ys + c] = (p.flags & MK_ROUND_BF16) ? mk_bf16(x) : x;
}

kernel void mk_grouped_project(device float *arena [[buffer(0)]], device const uchar *w [[buffer(1)]],
                                constant MK &p [[buffer(8)]], uint i [[thread_position_in_grid]]) {
    const uint width = p.n / p.m * p.taps;
    const uint r = i / width;
    if (r >= p.rows) return;
    const uint row = p.row0 + r, h = i % width / p.taps, d = i % p.taps;
    device const float *x = arena + p.x + row * p.xs + h * p.dim;
    float sum = 0.0f;
    for (uint k = 0; k < p.dim; ++k) sum = fma(x[k], mk_load(w, p.w, p.dtype, ulong(k) * p.m + d), sum);
    arena[p.y + row * p.ys + h * p.m + d] = (p.flags & MK_ROUND_BF16) ? mk_bf16(sum) : sum;
}

// Four causal queries share one BF16 weight load; each FMA chain is in
// the same order as mk_grouped_project.
kernel void mk_grouped_project_bf16x4(device float *arena [[buffer(0)]], device const uchar *w [[buffer(1)]],
                                      constant MK &p [[buffer(8)]], uint i [[thread_position_in_grid]]) {
    const uint width = p.n / p.m * p.taps, r = i / width * 4;
    if (r >= p.rows) return;
    const uint row = p.row0 + r, h = i % width / p.taps, d = i % p.taps;
    const device float *x = arena + p.x + row * p.xs + h * 16;
    const device ushort *weight = reinterpret_cast<const device ushort *>(w + p.w) + d;
    float4 sum = 0;
    #pragma unroll
    for (uint k = 0; k < 16; ++k) {
        const float wi = as_type<float>(uint(weight[k * p.m]) << 16);
        const float4 xi(x[k], x[p.xs+k], x[p.xs*2+k], x[p.xs*3+k]);
        sum = fma(xi, float4(wi), sum);
    }
    #pragma unroll
    for (uint t=0; t<4; ++t)
        arena[p.y + (row+t)*p.ys + h*p.m+d] = (p.flags & MK_ROUND_BF16) ? mk_bf16(sum[t]) : sum[t];
}

// Packed SIMD candidate: lanes load contiguous packed bytes and accumulate
// independent partials. The scalar reference remains the default.
inline float mk_expert_e4m3(uchar code) {
    uint exp = (code >> 3) & 15u, mant = code & 7u;
    uint sign = uint(code & 128u) << 24;
    if (exp == 0) return as_type<float>(as_type<uint>(float(mant) * (1.0f / 512.0f)) | sign);
    if (exp == 15 && mant == 7) return as_type<float>(0x7fc00000u);
    return as_type<float>(sign | ((exp + 120u) << 23) | (mant << 20));
}

inline float mk_expert_dot_simd(device const float *x, device const uchar *w, device const uchar *s,
                                device const uchar *amax, device const uchar *denom,
                                constant MKExpert &p, uint expert, ulong matrix_row, uint lane) {
    ulong base = matrix_row * p.k;
    float accum = 0.0f;
    if (p.nvfp4) {
        device const uchar *wb = w + p.w + base / 2;
        device const uchar *sb = s + p.s + base / 16;
        for (uint j = 2 * lane; j < p.k; j += 64) {
            uint byte = wb[j / 2];
            float scale = mk_expert_e4m3(sb[j / 16]);
            accum = fma(x[j], mk_fp4_e2m1(byte & 15u) * scale, accum);
            accum = fma(x[j + 1], mk_fp4_e2m1(byte >> 4) * scale, accum);
        }
        accum = simd_sum(accum);
        if (lane == 0) accum *= mk_load(amax, p.amax, 0, expert) / mk_load(denom, p.denominator, 0, 0);
    } else {
        device const uchar *wb = w + p.w + base;
        device const uchar *sb = s + p.s + base / 32;
        for (uint j = lane; j < p.k; j += 32) {
            uint e = sb[j / 32];
            float v = mk_expert_e4m3(wb[j]);
            float scale = as_type<float>(e == 0 ? 0x00400000u : e == 255 ? 0x7fc00000u : e << 23);
            accum = fma(x[j], v * scale, accum);
        }
        accum = simd_sum(accum);
    }
    return p.round_bf16 ? mk_bf16(accum) : accum;
}

kernel void mk_expert_gate_up_simd(device float *arena [[buffer(0)]], device const uchar *w [[buffer(1)]],
                                   device const uchar *s [[buffer(2)]], device const uchar *amax [[buffer(3)]],
                                   device const uchar *denom [[buffer(4)]], constant MKExpert &p [[buffer(8)]],
                                   uint gid [[thread_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
    uint i = gid / 32;
    uint r = i / (p.top_k * p.n), slot = (i / p.n) % p.top_k, j = i % p.n;
    if (r >= p.rows) return;
    r += p.row0;
    uint expert = uint(arena[p.routes + r * p.rs + slot]);
    float value = 0.0f;
    if (expert < p.experts) {
        ulong base = (ulong(expert) * p.n + j) * 2;
        device const float *x = arena + p.x + r * p.xs;
        float gate = mk_expert_dot_simd(x, w, s, amax, denom, p, expert, base, lane);
        float up = mk_expert_dot_simd(x, w, s, amax, denom, p, expert, base + 1, lane);
        if (lane == 0)
            value = ((tanh(gate / p.gate_cap) * p.gate_cap) * mk_sigmoid(gate)) * (tanh(up / p.up_cap) * p.up_cap);
    }
    if (lane == 0) arena[p.y + r * p.ys + slot * p.n + j] = p.round_bf16 ? mk_bf16(value) : value;
}

kernel void mk_expert_gate_up_pair(device float *arena [[buffer(0)]], device const uchar *w [[buffer(1)]],
                                   device const uchar *s [[buffer(2)]], device const uchar *amax [[buffer(3)]],
                                   device const uchar *denom [[buffer(4)]], constant MKExpert &p [[buffer(8)]],
                                   uint gid [[thread_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
    uint i = gid / 32;
    uint r = i / (p.top_k * p.n), slot = (i / p.n) % p.top_k, j = i % p.n;
    if (r >= p.rows) return;
    r += p.row0;
    uint expert = uint(arena[p.routes + r * p.rs + slot]);
    float value = 0.0f;
    if (expert < p.experts) {
        ulong base = (ulong(expert) * p.n + j) * 2;
        device const float *x = arena + p.x + r * p.xs;
        const ulong elem = base * p.k;
        device const uchar *wp = w + p.w + elem / 2;
        device const uchar *sp = s + p.s + elem / 16;
        const uint wd = p.k / 2, sd = p.k / 16;
        float2 accum = 0.0f;
        for (uint k = 2 * lane; k < p.k; k += 64) {
            uint2 bytes(wp[k / 2], wp[k / 2 + wd]);
            float2 scale(mk_expert_e4m3(sp[k / 16]), mk_expert_e4m3(sp[k / 16 + sd]));
            float2 w0(mk_fp4_e2m1(bytes.x & 15u), mk_fp4_e2m1(bytes.y & 15u));
            float2 w1(mk_fp4_e2m1(bytes.x >> 4), mk_fp4_e2m1(bytes.y >> 4));
            accum = fma(float2(x[k]), w0 * scale, accum);
            accum = fma(float2(x[k + 1]), w1 * scale, accum);
        }
        float gate = simd_sum(accum.x), up = simd_sum(accum.y);
        if (lane == 0) {
            float global = mk_load(amax, p.amax, 0, expert) / mk_load(denom, p.denominator, 0, 0);
            gate *= global;
            up *= global;
            if (p.round_bf16) { gate = mk_bf16(gate); up = mk_bf16(up); }
        }
        if (lane == 0)
            value = ((tanh(gate / p.gate_cap) * p.gate_cap) * mk_sigmoid(gate)) * (tanh(up / p.up_cap) * p.up_cap);
    }
    if (lane == 0) arena[p.y + r * p.ys + slot * p.n + j] = p.round_bf16 ? mk_bf16(value) : value;
}

// A SIMD group computes two gate/up pairs from one 8-value load per lane.
inline float mk_dot_nvfp4x8(uint w, float4 a, float4 b) {
    const float4 lo(mk_fp4_e2m1(w & 15u), mk_fp4_e2m1((w >> 4) & 15u),
                    mk_fp4_e2m1((w >> 8) & 15u), mk_fp4_e2m1((w >> 12) & 15u));
    w >>= 16;
    const float4 hi(mk_fp4_e2m1(w & 15u), mk_fp4_e2m1((w >> 4) & 15u),
                    mk_fp4_e2m1((w >> 8) & 15u), mk_fp4_e2m1((w >> 12) & 15u));
    const float4 l = a * lo, h = b * hi;
    return ((l.x + l.y) + l.z + l.w) + ((h.x + h.y) + h.z + h.w);
}

// Four adjacent output rows share the same input and global scale.
inline float4 mk_expert_nvfp4x4(device const float *x, device const uchar *w, device const uchar *s,
                                 device const uchar *amax, device const uchar *denom, constant MKExpert &p,
                                 ulong base, uint expert, uint lane) {
    float4 acc = 0;
    const uint wd = p.k / 8, sd = p.k / 16;
    device const uint *wp = reinterpret_cast<device const uint *>(w + p.w + base / 2);
    device const uchar *sp = s + p.s + base / 16;
    for (uint k = lane * 8; k < p.k; k += 256) {
        const float4 a(x[k], x[k + 1], x[k + 2], x[k + 3]);
        const float4 b(x[k + 4], x[k + 5], x[k + 6], x[k + 7]);
        #pragma unroll
        for (uint c = 0; c < 4; ++c) {
            const float part = mk_dot_nvfp4x8(wp[k / 8 + c * wd], a, b);
            const float scale = mk_expert_e4m3(sp[k / 16 + c * sd]);
            acc[c] += part * scale;
        }
    }
    acc = float4(simd_sum(acc.x), simd_sum(acc.y), simd_sum(acc.z), simd_sum(acc.w));
    if (lane == 0) acc *= mk_load(amax, p.amax, 0, expert) / mk_load(denom, p.denominator, 0, 0);
    return acc;
}

// MXFP8's regular and fast matvecs reduce 4 and 8 contiguous values per
// lane, respectively. Scale the group dot before accumulating K blocks.
template <uint D>
inline float4 mk_expert_mxfp8x4(device const float *x, device const uchar *w, device const uchar *s,
                                 constant MKExpert &p, ulong base, uint lane) {
    float4 acc = 0;
    device const uchar *wp = w + p.w + base;
    device const uchar *sp = s + p.s + base / 32;
    for (uint k = lane * D; k < p.k; k += D * 32) {
        float input[D];
        #pragma unroll
        for (uint i = 0; i < D; ++i) input[i] = x[k+i];
        #pragma unroll
        for (uint c = 0; c < 4; ++c) {
            const uint e = sp[k/32 + c*(p.k/32)];
            const float scale = as_type<float>(e == 0 ? 0x00400000u : e == 255 ? 0x7fc00000u : e << 23);
            const device uchar *codes = wp + k + c*p.k;
            float part = 0;
            #pragma unroll
            for (uint i = 0; i < D; ++i) part = fma(input[i], mk_expert_e4m3(codes[i]), part);
            acc[c] += part * scale;
        }
    }
    return float4(simd_sum(acc.x), simd_sum(acc.y), simd_sum(acc.z), simd_sum(acc.w));
}

inline float4 mk_expert_dot4(device const float *x, device const uchar *w, device const uchar *s,
                            device const uchar *amax, device const uchar *denom, constant MKExpert &p,
                            ulong base, uint expert, uint lane) {
    if (p.nvfp4) return mk_expert_nvfp4x4(x, w, s, amax, denom, p, base, expert, lane);
    if (p.k % 256 == 0) return mk_expert_mxfp8x4<8>(x, w, s, p, base, lane);
    return mk_expert_mxfp8x4<4>(x, w, s, p, base, lane);
}

kernel void mk_expert_gate_up_dot8(device float *arena [[buffer(0)]], device const uchar *w [[buffer(1)]],
                                    device const uchar *s [[buffer(2)]], device const uchar *amax [[buffer(3)]],
                                    device const uchar *denom [[buffer(4)]], constant MKExpert &p [[buffer(8)]],
                                    uint gid [[thread_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
    const uint columns = p.n / 2, i = gid / 32;
    uint r = i / (p.top_k * columns);
    if (r >= p.rows) return;
    const uint slot = (i / columns) % p.top_k, j = (i % columns) * 2;
    r += p.row0;
    const uint expert = uint(arena[p.routes + r * p.rs + slot]);
    float4 acc = 0.0f;
    if (expert < p.experts) {
        const ulong base = (ulong(expert) * p.n + j) * 2 * p.k;
        device const float *x = arena + p.x + r * p.xs;
        acc = mk_expert_dot4(x, w, s, amax, denom, p, base, expert, lane);
    }
    if (lane == 0) {
        device float *out = arena + p.y + r * p.ys + slot * p.n + j;
        for (uint c = 0; c < 2; ++c) {
            float gate = acc[c * 2], up = acc[c * 2 + 1];
            if (p.round_bf16) { gate = mk_bf16(gate); up = mk_bf16(up); }
            const float value = ((tanh(gate / p.gate_cap) * p.gate_cap) * mk_sigmoid(gate)) * (tanh(up / p.up_cap) * p.up_cap);
            out[c] = p.round_bf16 ? mk_bf16(value) : value;
        }
    }
}

kernel void mk_expert_down_simd(device float *arena [[buffer(0)]], device const uchar *w [[buffer(1)]],
                                device const uchar *s [[buffer(2)]], device const uchar *amax [[buffer(3)]],
                                device const uchar *denom [[buffer(4)]], constant MKExpert &p [[buffer(8)]],
                                uint i [[threadgroup_position_in_grid]], uint slot [[simdgroup_index_in_threadgroup]],
                                uint lane [[thread_index_in_simdgroup]]) {
    uint r = i / p.n, j = i % p.n;
    if (r >= p.rows) return;
    r += p.row0;
    threadgroup float dots[8];
    if (slot < p.top_k) {
        device const float *routes = arena + p.routes + r * p.rs;
        uint expert = uint(routes[slot]);
        float dot = 0.0f;
        if (expert < p.experts) {
            device const float *x = arena + p.x + r * p.xs + slot * p.k;
            dot = mk_expert_dot_simd(x, w, s, amax, denom, p, expert, ulong(expert) * p.n + j, lane) * routes[p.top_k + slot];
        }
        if (lane == 0) dots[slot] = dot;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (slot == 0 && lane == 0) {
        float value = 0.0f;
        for (uint k = 0; k < p.top_k; ++k) value += dots[k];
        arena[p.y + r * p.ys + j] = p.round_bf16 ? mk_bf16(value) : value;
    }
}

// Four neighboring outputs share a load of each routed activation block.
kernel void mk_expert_down_dot8(device float *arena [[buffer(0)]], device const uchar *w [[buffer(1)]],
                                 device const uchar *s [[buffer(2)]], device const uchar *amax [[buffer(3)]],
                                 device const uchar *denom [[buffer(4)]], constant MKExpert &p [[buffer(8)]],
                                 uint i [[threadgroup_position_in_grid]], uint slot [[simdgroup_index_in_threadgroup]],
                                 uint lane [[thread_index_in_simdgroup]]) {
    const uint columns = p.n / 4, j = (i % columns) * 4;
    uint r = i / columns;
    if (r >= p.rows) return;
    r += p.row0;
    threadgroup float4 dots[8];
    if (slot < p.top_k) {
        device const float *routes = arena + p.routes + r * p.rs;
        const uint expert = uint(routes[slot]);
        float4 acc = 0;
        if (expert < p.experts) {
            const ulong base = (ulong(expert) * p.n + j) * p.k;
            device const float *x = arena + p.x + r * p.xs + slot * p.k;
            acc = mk_expert_dot4(x, w, s, amax, denom, p, base, expert, lane);
        }
        if (lane == 0) {
            if (p.round_bf16) for (uint c = 0; c < 4; ++c) acc[c] = mk_bf16(acc[c]);
            dots[slot] = acc * routes[p.top_k + slot];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (slot == 0 && lane < 4) {
        float sum = 0;
        for (uint k = 0; k < p.top_k; ++k) sum += dots[k][lane];
        arena[p.y + r * p.ys + j + lane] = p.round_bf16 ? mk_bf16(sum) : sum;
    }
}

// Keep mk_linear_generic's lane-strided summation order. Tile across outputs
// to reuse each input load and across tokens to reuse each weight load.
template <uint T>
inline void mk_bf16_linear_tile(device const ushort *weights, device float *arena,
                                 constant MK &p, uint simd, uint lane) {
    uint tiles = (p.rows + T - 1) / T;
    uint row = (simd / tiles) * 4;
    uint t0 = (simd % tiles) * T;
    if (row >= p.m) return;
    const uint valid = min(T, p.rows - t0);
    float acc[T][4] = {};
    for (uint k = lane; k < p.n; k += 32) {
        float w[4];
        for (uint j = 0; j < 4; ++j) w[j] = as_type<float>(uint(weights[ulong(row + j) * p.n + k]) << 16);
        for (uint t = 0; t < T; ++t) {
            if (t >= valid) break;
            float x = arena[p.x + (p.row0 + t0 + t) * p.xs + k];
            for (uint j = 0; j < 4; ++j) acc[t][j] += w[j] * x;
        }
    }
    for (uint t = 0; t < T; ++t) {
        if (t >= valid) break;
        for (uint j = 0; j < 4; ++j) {
            float sum = simd_sum(acc[t][j]);
            if (lane == 0) arena[p.y + (p.row0 + t0 + t) * p.ys + row + j] = (p.flags & MK_ROUND_OUT) ? mk_bf16(sum) : sum;
        }
    }
}

#define MK_BF16_LINEAR(T) \
kernel void mk_bf16_linear_4x##T(device const uchar *w [[buffer(0)]], device float *arena [[buffer(1)]], \
                                constant MK &p [[buffer(8)]], uint gid [[thread_position_in_grid]], \
                                uint lane [[thread_index_in_simdgroup]]) { \
    mk_bf16_linear_tile<T>((device const ushort *)(w + p.w), arena, p, gid / 32, lane); \
}
MK_BF16_LINEAR(1)
MK_BF16_LINEAR(4)
#undef MK_BF16_LINEAR

// Full tiles have no tail, so all accumulators can stay in registers.
template <uint T>
inline void mk_bf16_full_tile(device const uchar *w, device float *arena,
                              constant MK &p, uint gid, uint lane) {
    const uint tiles = p.rows / T, simd = gid / 32;
    const uint row = (simd / tiles) * 4, t0 = p.row0 + (simd % tiles) * T;
    device const ushort *src = (device const ushort *)(w + p.w) + ulong(row) * p.n;
    device const float *x = arena + p.x + t0 * p.xs;
    float4 acc[T] = {};
    for (uint k = lane; k < p.n; k += 32) {
        uint4 bits(src[k], src[k + p.n], src[k + 2ul * p.n], src[k + 3ul * p.n]);
        float4 b = as_type<float4>(bits << 16);
        #pragma unroll
        for (uint t = 0; t < T; ++t) acc[t] += b * x[t * p.xs + k];
    }
    #pragma unroll
    for (uint t = 0; t < T; ++t) {
        #pragma unroll
        for (uint j = 0; j < 4; ++j) {
            float sum = simd_sum(acc[t][j]);
            if (lane == 0) arena[p.y + (t0 + t) * p.ys + row + j] = (p.flags & MK_ROUND_OUT) ? mk_bf16(sum) : sum;
        }
    }
}
#define MK_BF16_FULL(T) \
kernel void mk_bf16_linear_4x##T##_full(device const uchar *w [[buffer(0)]], device float *arena [[buffer(1)]], \
                                       constant MK &p [[buffer(8)]], uint gid [[thread_position_in_grid]], \
                                       uint lane [[thread_index_in_simdgroup]]) { mk_bf16_full_tile<T>(w, arena, p, gid, lane); }
MK_BF16_FULL(8)

kernel void mk_bf16_linear_4x8_full_vec(device const uchar *w [[buffer(0)]], device float *arena [[buffer(1)]],
                                       constant MK &p [[buffer(8)]], uint gid [[thread_position_in_grid]],
                                       uint lane [[thread_index_in_simdgroup]]) {
    constexpr uint T = 8;
    const uint tiles = p.rows / T, simd = gid / 32;
    const uint row = (simd / tiles) * 4, t0 = p.row0 + (simd % tiles) * T;
    const uint stride = p.n / 4;
    const device ushort4 *src = (device const ushort4 *)(w + p.w) + ulong(row) * stride;
    const device float4 *x = (device const float4 *)(arena + p.x + t0 * p.xs);
    float4 acc[T] = {};
    for (uint k = lane; k < stride; k += 32) {
        float4 b[4];
        #pragma unroll
        for (uint j = 0; j < 4; ++j) b[j] = as_type<float4>(uint4(src[k + ulong(j) * stride]) << 16);
        #pragma unroll
        for (uint t = 0; t < T; ++t) {
            const float4 a = x[t * (p.xs / 4) + k];
            #pragma unroll
            for (uint i = 0; i < 4; ++i) {
                const float4 w = float4(b[0][i], b[1][i], b[2][i], b[3][i]);
                acc[t] = fma(w, a[i], acc[t]);
            }
        }
    }
    #pragma unroll
    for (uint t = 0; t < T; ++t) {
        #pragma unroll
        for (uint j = 0; j < 4; ++j) {
            const float sum = simd_sum(acc[t][j]);
            if (lane == 0) arena[p.y + (t0 + t) * p.ys + row + j] = (p.flags & MK_ROUND_OUT) ? mk_bf16(sum) : sum;
        }
    }
}

MK_BF16_FULL(16)
MK_BF16_FULL(32)
#undef MK_BF16_FULL

kernel void mk_bf16_linear_4x1_vec(device const uchar *w [[buffer(0)]], device float *arena [[buffer(1)]],
                                  constant MK &p [[buffer(8)]], uint gid [[thread_position_in_grid]],
                                  uint lane [[thread_index_in_simdgroup]]) {
    uint row = gid / 32 * 4;
    if (row >= p.m) return;
    const uint stride = p.n / 4;
    device const ushort4 *weights = (device const ushort4 *)(w + p.w) + ulong(row) * stride;
    device const float4 *x = (device const float4 *)(arena + p.x + p.row0 * p.xs);
    float partial[4] = {};
    #pragma unroll 4
    for (uint k = lane; k < stride; k += 32) {
        float4 a = x[k];
        #pragma unroll
        for (uint j = 0; j < 4; ++j) {
            float4 b = as_type<float4>(uint4(weights[ulong(j) * stride + k]) << 16);
            for (uint i = 0; i < 4; ++i) partial[j] = fma(a[i], b[i], partial[j]);
        }
    }
    #pragma unroll
    for (uint j = 0; j < 4; ++j) {
        float sum = simd_sum(partial[j]);
        if (lane == 0) arena[p.y + p.row0 * p.ys + row + j] = (p.flags & MK_ROUND_OUT) ? mk_bf16(sum) : sum;
    }
}

#include <metal_simdgroup_matrix>

// Each SIMD reuses the same 16 weight columns for two or four token fragments.
template <uint M>
inline void mk_bf16_matrix_tile(const device uchar *weights, device float *arena, constant MK &p,
                                uint group, uint tid, uint simd, uint lane, threadgroup bfloat *a) {
    constexpr uint K = 32, L = K + 8, N = 64;
    const uint r0 = group / (p.m / N) * M, c0 = group % (p.m / N) * N;
    const uint fr = ((lane / 4) & 4) + (lane / 2) % 4;
    const uint fc = (((lane / 4) & 2) + lane % 2) * 2;
    device const bfloat *w = reinterpret_cast<device const bfloat *>(weights + p.w) + ulong(c0+simd*16)*p.n;
    simdgroup_float8x8 acc[M/8][2];
    #pragma unroll
    for (uint i=0; i<M/8; ++i) for (uint j=0; j<2; ++j) acc[i][j] = make_filled_simdgroup_matrix<float,8,8>(0);
    const uint rr = tid / (K/4), ac = tid % (K/4);
    const bool lo = r0+rr < p.rows, hi = r0+rr+16 < p.rows;
    const device float4 *xlo = reinterpret_cast<const device float4 *>(arena + p.x + (p.row0 + (lo ? r0+rr : 0)) * p.xs);
    const device float4 *xhi = reinterpret_cast<const device float4 *>(arena + p.x + (p.row0 + (hi ? r0+rr+16 : 0)) * p.xs);
    for (uint k = 0; k < p.n; k += K) {
        reinterpret_cast<threadgroup bfloat4 *>(a + rr*L)[ac] = bfloat4(lo ? xlo[k/4+ac] : float4(0));
        if constexpr (M > 16) reinterpret_cast<threadgroup bfloat4 *>(a + (rr+16)*L)[ac] = bfloat4(hi ? xhi[k/4+ac] : float4(0));
        threadgroup_barrier(mem_flags::mem_threadgroup);
        #pragma unroll
        for (uint kf=0; kf<K; kf+=8) {
            simdgroup_bfloat8x8 af[M/8], bf[2];
            #pragma unroll
            for (uint i=0; i<M/8; ++i) simdgroup_load(af[i],a+i*8*L+kf,L);
            #pragma unroll
            for (uint i=0; i<2; ++i) simdgroup_load(bf[i],w + ulong(i*8)*p.n+k+kf,p.n,ulong2(0),true);
            #pragma unroll
            for (uint i=0; i<M/8; ++i) for (uint j=0; j<2; ++j)
                simdgroup_multiply_accumulate(acc[i][j],af[i],bf[j],acc[i][j]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    #pragma unroll
    for (uint i=0; i<M/8; ++i) {
        const uint row = r0+i*8+fr;
        if (row >= p.rows) continue;
        device float *out = arena+p.y+(p.row0+row)*p.ys+c0+simd*16+fc;
        #pragma unroll
        for (uint j=0; j<2; ++j) for (uint e=0; e<2; ++e) {
            const float value = acc[i][j].thread_elements()[e];
            out[j*8+e] = (p.flags & MK_ROUND_OUT) ? mk_bf16(value) : value;
        }
    }
}

#define MK_BF16_MATRIX(M) \
    kernel void mk_bf16_prefill_matrix##M(device const uchar *weights [[buffer(0)]], device float *arena [[buffer(1)]], \
                                      constant MK &p [[buffer(8)]], uint group [[threadgroup_position_in_grid]], \
                                      uint tid [[thread_index_in_threadgroup]], uint simd [[simdgroup_index_in_threadgroup]], \
                                      uint lane [[thread_index_in_simdgroup]]) { \
        alignas(8) threadgroup bfloat a[M*40]; \
        mk_bf16_matrix_tile<M>(weights,arena,p,group,tid,simd,lane,a); \
    }
MK_BF16_MATRIX(16)
MK_BF16_MATRIX(32)
#undef MK_BF16_MATRIX

// Build a compact list of route slots for each expert.
kernel void mk_expert_row_lists(device const float *arena [[buffer(0)]], device uint *lists [[buffer(1)]],
                                constant MKExpert &p [[buffer(8)]], uint expert [[thread_position_in_grid]]) {
    if (expert >= p.experts) return;
    device uint *entries = lists + expert * (p.rows * p.top_k + 1);
    uint n = 0;
    for (uint r = 0; r < p.rows; ++r) {
        device const float *routes = arena + p.routes + (p.row0 + r) * p.rs;
        for (uint s = 0; s < p.top_k; ++s) if (uint(routes[s]) == expert)
            entries[++n] = r * p.top_k + s;
    }
    entries[0] = n;
}

// Stable compact of the row-major routes. Lanes own consecutive routes,
// and each SIMD group writes the list of a different expert.
kernel void mk_expert_row_lists_simd(device const float *arena [[buffer(0)]], device uint *lists [[buffer(1)]],
                                     constant MKExpert &p [[buffer(8)]], uint gid [[thread_position_in_grid]],
                                     uint lane [[thread_index_in_simdgroup]]) {
    const uint expert = gid / 32;
    if (expert >= p.experts) return;
    const uint total = p.rows * p.top_k;
    device uint *entries = lists + expert * (total + 1);
    uint n = 0;
    for (uint block = 0; block < total; block += 32) {
        const uint i = block + lane;
        uint match = 0;
        if (i < total) {
            const uint row = p.row0 + i / p.top_k;
            match = uint(uint(arena[p.routes + row * p.rs + i % p.top_k]) == expert);
        }
        const uint rank = simd_prefix_exclusive_sum(match);
        if (match) entries[1 + n + rank] = i;
        n += simd_sum(match);
    }
    if (lane == 0) entries[0] = n;
}

// One workgroup per expert/32 output rows. Its four SIMD groups share a
// dequantized KxN tile for up to eight tokens routed to that expert.
template <bool Down, uint K, typename T>
inline void mk_expert_mm(device float *arena, device const uchar *w, device const uchar *s,
                         device const uchar *amax, device const uchar *denom, device const uint *lists,
                         device float *partial, constant MKExpert &p, uint group, uint tid, uint simd,
                         threadgroup T *xs, threadgroup T *ws, threadgroup float *ys) {
    const uint N = Down ? p.n : 2 * p.n;
    const uint expert = group / (N / 32), c0 = (group % (N / 32)) * 32;
    device const uint *list = lists + expert * (p.rows * p.top_k + 1);
    uint count = list[0];
    if (count == 0) return;
    float global = p.nvfp4 ? mk_load(amax, p.amax, 0, expert) / mk_load(denom, p.denominator, 0, 0) : 1.0f;
    for (uint t0 = 0; t0 < count; t0 += 8) {
        simdgroup_float8x8 acc = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        for (uint k0 = 0; k0 < p.k; k0 += K) {
            for (uint j = tid; j < 8 * K; j += 128) {
                uint t = t0 + j / K;
                float x = 0;
                if (t < count) {
                    uint route = list[1 + t];
                    x = arena[p.x + (p.row0 + route / p.top_k) * p.xs + (Down ? (route % p.top_k) * p.k : 0) + k0 + j % K];
                }
                xs[j] = T(x);
            }
            if (K == 128) {
                // Each thread owns a 32-value K segment. Reuse a scale
                // across a whole block and keep weights in row-major scratch.
                const uint lane_k = (tid % 4) * 32;
                for (uint c = tid / 4; c < 32; c += 32) {
                    const ulong index = (ulong(expert) * N + c0 + c) * p.k + k0 + lane_k;
                    threadgroup T *dst = ws + c * K + lane_k;
                    if (p.nvfp4) {
                        for (uint block = 0; block < 2; ++block) {
                            const float scale = mk_expert_e4m3(s[p.s + index / 16 + block]);
                            for (uint j = 0; j < 8; ++j) {
                                uint byte = w[p.w + index / 2 + block * 8 + j];
                                dst[block * 16 + 2 * j] = T(mk_fp4_e2m1(byte & 15u) * scale);
                                dst[block * 16 + 2 * j + 1] = T(mk_fp4_e2m1(byte >> 4) * scale);
                            }
                        }
                    } else {
                        uint s8 = s[p.s + index / 32];
                        float scale = as_type<float>(s8 == 0 ? 0x00400000u : s8 == 255 ? 0x7fc00000u : s8 << 23);
                        for (uint j = 0; j < 32; ++j) dst[j] = T(mk_expert_e4m3(w[p.w + index + j]) * scale);
                    }
                }
            } else for (uint j = tid; j < 1024; j += 128) {
                uint c = j / 32, k = j % 32;
                ulong index = (ulong(expert) * N + c0 + c) * p.k + k0 + k;
                float value;
                if (p.nvfp4) {
                    uint code = (w[p.w + index / 2] >> ((k & 1) * 4)) & 15u;
                    value = mk_fp4_e2m1(code) * mk_expert_e4m3(s[p.s + index / 16]);
                } else {
                    uint s8 = s[p.s + index / 32];
                    float scale = as_type<float>(s8 == 0 ? 0x00400000u : s8 == 255 ? 0x7fc00000u : s8 << 23);
                    value = mk_expert_e4m3(w[p.w + index]) * scale;
                }
                ws[k * 32 + c] = T(value);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint k = 0; k < K; k += 8) {
                simdgroup_matrix<T,8,8> a, b;
                simdgroup_load(a, xs + k, K);
                if (K == 128) simdgroup_load(b, ws + simd * 8 * K + k, K, ulong2(0), true);
                else simdgroup_load(b, ws + k * 32 + simd * 8, 32);
                simdgroup_multiply_accumulate(acc, a, b, acc);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        simdgroup_store(acc, ys + simd * 8, 32);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const uint cols = Down ? 32 : 16;
        for (uint j = tid; j < 8 * cols; j += 128) {
            uint t = t0 + j / cols, c = j % cols;
            if (t >= count) continue;
            uint route = list[1 + t];
            if (Down) {
                float value = ys[j] * global;
                partial[ulong(route) * p.n + c0 + c] = p.round_bf16 ? mk_bf16(value) : value;
            } else {
                float gate = ys[2 * j] * global, up = ys[2 * j + 1] * global;
                if (p.round_bf16) { gate = mk_bf16(gate); up = mk_bf16(up); }
                float value = ((tanh(gate / p.gate_cap) * p.gate_cap) * mk_sigmoid(gate)) * (tanh(up / p.up_cap) * p.up_cap);
                arena[p.y + (p.row0 + route / p.top_k) * p.ys + (route % p.top_k) * p.n + c0 / 2 + c] = p.round_bf16 ? mk_bf16(value) : value;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

#define MK_EXPERT_MM(NAME, DOWN, K, T) \
kernel void NAME(device float *arena [[buffer(0)]], device const uchar *w [[buffer(1)]], \
                 device const uchar *s [[buffer(2)]], device const uchar *amax [[buffer(3)]], \
                 device const uchar *denom [[buffer(4)]], device const uint *list [[buffer(5)]], \
                 device float *partial [[buffer(6)]], constant MKExpert &p [[buffer(8)]], \
                 uint group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]], \
                 uint simd [[simdgroup_index_in_threadgroup]]) { \
    threadgroup T xs[8 * K], ws[32 * K]; threadgroup float ys[256]; \
    mk_expert_mm<DOWN, K, T>(arena, w, s, amax, denom, list, partial, p, group, tid, simd, xs, ws, ys); \
}
MK_EXPERT_MM(mk_expert_gate_up_mm, false, 32, float)
MK_EXPERT_MM(mk_expert_down_mm, true, 32, float)
MK_EXPERT_MM(mk_expert_gate_up_wide, false, 128, float)
MK_EXPERT_MM(mk_expert_down_wide, true, 128, float)
MK_EXPERT_MM(mk_expert_gate_up_bf16, false, 128, bfloat)
MK_EXPERT_MM(mk_expert_down_bf16, true, 128, bfloat)
#undef MK_EXPERT_MM

inline bfloat4 mk_nv_weights4(uint packed, float scale) {
    const uint4 code = (uint4(packed) >> uint4(0, 4, 8, 12)) & 15u;
    const uint4 mag = code & 7u;
    const uint4 ordinary = 0x3f000000u + (mag << 22);
    const uint4 bits = select(ordinary, mag * 0x3f000000u, mag < 2u);
    return bfloat4(as_type<float4>(bits | ((code & 8u) << 28)) * scale);
}

// Batched expert path: 8, 16 or 32 routed rows x 32 outputs x 64 K,
// split across four SIMD groups. Round the complete NVFP4 scale or MXFP8 weight to BF16
// before FP32 matrix accumulation. Input rows already crossed a BF16 boundary.
template <bool Down, bool Nv, uint M>
inline void mk_expert_block(device float *arena, device const uchar *w, device const uchar *s,
                             device const uchar *amax, device const uchar *denom, device const uint *lists,
                             device float *partial, constant MKExpert &p, uint g, uint tid, uint simd, uint lane,
                             uint start, uint stop, uint step, threadgroup bfloat *a, threadgroup bfloat *b) {
    constexpr uint N = 32, B = 64, L = B + 8, TG = 128;
    constexpr uint Loads = M * B / 4 / TG;
    const uint n = Down ? p.n : 2 * p.n;
    const uint expert = g / (n / N), c0 = (g % (n / N)) * N;
    device const uint *list = lists + expert * (p.rows * p.top_k + 1);
    const uint count = list[0];
    if (count == 0) return;
    const float outer = Nv ? mk_load(amax,p.amax,0,expert) / mk_load(denom,p.denominator,0,0) : 1.0f;
    const uint fr = ((lane / 4) & 4) + (lane / 2) % 4;
    const uint fc = (((lane / 4) & 2) + lane % 2) * 2;
    for (uint t0 = start; t0 < stop; t0 += step) {
        device const float *xs[Loads];
        bool active[Loads];
        #pragma unroll
        for (uint i=0; i<Loads; ++i) {
            const uint id = tid + i*TG, xr = t0 + id / (B/4), xk = id % (B/4) * 4;
            active[i] = xr < count;
            const uint route = active[i] ? list[xr+1] : 0;
            xs[i] = arena + p.x + (p.row0 + route / p.top_k) * p.xs + (Down ? route % p.top_k * p.k : 0) + xk;
        }
        const uint wc = tid / 4, wk = tid % 4 * (B/4);
        const ulong offset = (ulong(expert) * n + c0 + wc) * p.k + wk;
        simdgroup_float8x8 sums[M/8];
        #pragma unroll
        for (uint i=0; i<M/8; ++i)
            sums[i] = make_filled_simdgroup_matrix<float,8,8>(0);
        for (uint k=0; k<p.k; k+=B) {
            #pragma unroll
            for (uint i=0; i<Loads; ++i) {
                const uint id = tid + i*TG;
                const uint ar = id / (B/4), ak = id % (B/4) * 4;
                const device float4 *src = reinterpret_cast<const device float4 *>(xs[i]+k);
                reinterpret_cast<threadgroup bfloat4 *>(a+ar*L+ak)[0] = bfloat4(active[i] ? src[0] : float4(0));
            }
            const ulong windex = offset + k;
            if constexpr (Nv) {
                device const uint *bytes = reinterpret_cast<device const uint *>(w + p.w + windex/2);
                #pragma unroll
                for (uint block=0; block<B/64; ++block) {
                    const float scale = mk_expert_e4m3(s[p.s + windex/16 + block]) * outer;
                    const uint lo = bytes[2*block], hi = bytes[2*block+1];
                    threadgroup bfloat4 *dst = reinterpret_cast<threadgroup bfloat4 *>(b + wc*L+wk+block*16);
                    dst[0] = mk_nv_weights4(lo, scale);
                    dst[1] = mk_nv_weights4(lo >> 16, scale);
                    dst[2] = mk_nv_weights4(hi, scale);
                    dst[3] = mk_nv_weights4(hi >> 16, scale);
                }
            } else {
                uint e = s[p.s + windex / 32];
                float factor = as_type<float>(e == 0 ? 0x00400000u : e == 255 ? 0x7fc00000u : e << 23);
                #pragma unroll
                for (uint d=0; d<B/4; ++d) b[wc*L+wk+d] = bfloat(mk_expert_e4m3(w[p.w+windex+d]) * factor);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            #pragma unroll
            for (uint kf=0; kf<B; kf+=8) {
                simdgroup_bfloat8x8 bf;
                simdgroup_load(bf,b+simd*8*L+kf,L,ulong2(0),true);
                #pragma unroll
                for (uint i=0; i<M/8; ++i) {
                    simdgroup_bfloat8x8 af;
                    simdgroup_load(af,a+i*8*L+kf,L);
                    simdgroup_multiply_accumulate(sums[i],af,bf,sums[i]);
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        #pragma unroll
        for (uint i=0; i<M/8; ++i) {
            const uint t = t0 + i*8 + fr;
            if (t >= count) continue;
            const uint r = list[t+1];
            {
                const uint c = c0 + simd*8 + fc;
                float lo = sums[i].thread_elements()[0], hi = sums[i].thread_elements()[1];
                if (p.round_bf16) { lo = mk_bf16(lo); hi = mk_bf16(hi); }
                if constexpr (Down) {
                    partial[ulong(r)*p.n+c] = lo;
                    partial[ulong(r)*p.n+c+1] = hi;
                } else {
                    const float out = (tanh(lo / p.gate_cap) * p.gate_cap * mk_sigmoid(lo)) * (tanh(hi / p.up_cap) * p.up_cap);
                    arena[p.y + (p.row0 + r/p.top_k)*p.ys + (r%p.top_k)*p.n + c/2] = p.round_bf16 ? mk_bf16(out) : out;
                }
            }
        }
    }
}
// Stripe complete row tiles across workgroups so popular experts do not
// leave a single workgroup doing their trailing tiles. Each group still owns
// distinct route/column pairs, and only one group processes the final tail.
#define MK_EXPERT_BLOCK(NAME, DOWN, NV) \
    kernel void NAME(device float *arena [[buffer(0)]],device const uchar *w [[buffer(1)]],device const uchar *s [[buffer(2)]], \
                     device const uchar *amax [[buffer(3)]],device const uchar *denom [[buffer(4)]],device const uint *lists [[buffer(5)]], \
                     device float *partial [[buffer(6)]],constant MKExpert &p [[buffer(8)]],uint g [[threadgroup_position_in_grid]], \
                     uint tid [[thread_index_in_threadgroup]],uint simd [[simdgroup_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]]) { \
        alignas(8) threadgroup bfloat a[32*72], b[32*72]; \
        const uint n = DOWN ? p.n : 2 * p.n; \
        const uint cols = n / 32, parts = p.rows >= 128 ? 4 : p.rows >= 64 ? 2 : 1; \
        const uint expert = g / (cols * parts), part = g / cols % parts; \
        const uint count = lists[expert * (p.rows * p.top_k + 1)]; \
        const uint full = count / 32 * 32; \
        const uint tail = count - full; \
        g = expert * cols + g % cols; \
        if (part * 32 < full) { \
            mk_expert_block<DOWN,NV,32>(arena,w,s,amax,denom,lists,partial,p,g,tid,simd,lane,part*32,full,32*parts,a,b); \
        } \
        if (full/32 % parts != part) return; \
        if (tail != 0 && tail <= 8) { \
            mk_expert_block<DOWN,NV,8>(arena,w,s,amax,denom,lists,partial,p,g,tid,simd,lane,full,count,8,a,b); \
        } else if (tail > 8 && tail <= 16) { \
            mk_expert_block<DOWN,NV,16>(arena,w,s,amax,denom,lists,partial,p,g,tid,simd,lane,full,count,16,a,b); \
        } else if (tail > 16) { \
            mk_expert_block<DOWN,NV,32>(arena,w,s,amax,denom,lists,partial,p,g,tid,simd,lane,full,count,32,a,b); \
        } \
    }
MK_EXPERT_BLOCK(mk_expert_block_up_nv, false, true)
MK_EXPERT_BLOCK(mk_expert_block_down_nv, true, true)
MK_EXPERT_BLOCK(mk_expert_block_up_mx, false, false)
MK_EXPERT_BLOCK(mk_expert_block_down_mx, true, false)
#undef MK_EXPERT_BLOCK

kernel void mk_expert_combine_mm(device float *arena [[buffer(0)]], device const float *partial [[buffer(1)]],
                                 constant MKExpert &p [[buffer(8)]], uint i [[thread_position_in_grid]]) {
    uint r = i / p.n, j = i % p.n;
    if (r >= p.rows) return;
    device const float *routes = arena + p.routes + (p.row0 + r) * p.rs;
    float sum = 0;
    for (uint s = 0; s < p.top_k; ++s) {
        if (uint(routes[s]) < p.experts) sum += partial[(ulong(r) * p.top_k + s) * p.n + j] * routes[p.top_k + s];
    }
    arena[p.y + (p.row0 + r) * p.ys + j] = p.round_bf16 ? mk_bf16(sum) : sum;
}

// Partition a query's visible keys; only the merge includes the sink.
kernel void mk_relative_split_keys(device const float *arena [[buffer(0)]], device const float *key [[buffer(1)]],
                                   device const float *val [[buffer(2)]], device float *parts [[buffer(3)]],
                                   constant MK &p [[buffer(8)]], uint lane [[thread_index_in_simdgroup]],
                                   uint g [[threadgroup_position_in_grid]]) {
    uint qh = g / p.taps, part = g % p.taps;
    uint h = qh % p.heads, row = p.row0 + qh / p.heads;
    uint pos = p.pos + row;
    uint kvh = h / (p.heads / p.kv_heads);
    uint visible = p.window ? min(p.window, pos + 1) : pos + 1;
    uint first = pos + 1 - visible;
    uint start = first + uint(ulong(visible) * part / p.taps);
    uint end = first + uint(ulong(visible) * (part + 1) / p.taps);
    uint E = (p.dim + 31) / 32;
    float q[8], acc[8] = {};
    device const float *qp = arena + p.x + row * p.xs + h * p.dim;
    device const float *r = arena + p.z + row * p.zs + h * p.m;
    for (uint e = 0; e < E; ++e) {
        uint d = lane + 32 * e;
        q[e] = d < p.dim ? qp[d] : 0.0f;
    }
    float maximum = -INFINITY, denominator = 0.0f;
    uint slot = start % p.capacity;
    for (uint t = start; t < end; ++t) {
        ulong base = ulong(slot) * (p.kv_heads * p.dim) + kvh * p.dim;
        if (++slot == p.capacity) slot = 0;
        float dot = 0.0f;
        for (uint e = 0; e < E; ++e) {
            uint d = lane + 32 * e;
            if (d < p.dim) dot += q[e] * key[base + d];
        }
        dot = simd_sum(dot);
        uint distance = pos - t;
        float score = dot * p.scale + (distance < p.m ? r[distance] : 0.0f);
        float m = max(maximum, score);
        float prev = t == start ? 0.0f : metal::precise::exp(maximum - m);
        float w = metal::precise::exp(score - m);
        denominator = denominator * prev + w;
        for (uint e = 0; e < E; ++e) {
            uint d = lane + 32 * e;
            if (d < p.dim) acc[e] = acc[e] * prev + w * val[base + d];
        }
        maximum = m;
    }
    device float *dst = parts + ulong(g) * (p.dim + 2);
    for (uint e = 0; e < E; ++e) {
        uint d = lane + 32 * e;
        if (d < p.dim) dst[d] = acc[e];
    }
    if (lane == 0) { dst[p.dim] = maximum; dst[p.dim + 1] = denominator; }
}

// Compute four key dots concurrently, then replay the original per-part
// online reduction in ascending key order. Limited to head size 128,
// partitions of at most 32 positions, and four SIMD groups per threadgroup.
kernel void mk_relative_split_keys4(device const float *arena [[buffer(0)]], device const float *key [[buffer(1)]],
                                    device const float *val [[buffer(2)]], device float *parts [[buffer(3)]],
                                    constant MK &p [[buffer(8)]], uint tid [[thread_index_in_threadgroup]],
                                    uint lane [[thread_index_in_simdgroup]], uint simd [[simdgroup_index_in_threadgroup]],
                                    uint g [[threadgroup_position_in_grid]]) {
    const uint qh = g / p.taps, part = g % p.taps;
    const uint h = qh % p.heads, row = p.row0 + qh / p.heads;
    const uint pos = p.pos + row;
    const uint kvh = h / (p.heads / p.kv_heads);
    const uint visible = p.window ? min(p.window, pos + 1) : pos + 1;
    const uint first = pos + 1 - visible;
    const uint start = first + uint(ulong(visible) * part / p.taps);
    const uint end = first + uint(ulong(visible) * (part + 1) / p.taps);
    const uint stride = p.kv_heads * 128;
    device const float *q = arena + p.x + row * p.xs + h * 128;
    device const float *bias = arena + p.z + row * p.zs + h * p.m;
    threadgroup float score[32];
    float q_lane[4];
    #pragma unroll
    for (uint e = 0; e < 4; ++e) q_lane[e] = q[lane + 32 * e];
    for (uint t = start + simd; t < end; t += 4) {
        device const float *k = key + ulong(t % p.capacity) * stride + kvh * 128;
        float dot = 0.0f;
        #pragma unroll
        for (uint e = 0; e < 4; ++e) dot += q_lane[e] * k[lane + 32 * e];
        dot = simd_sum(dot);
        const uint distance = pos - t;
        if (lane == 0) score[t - start] = dot * p.scale + (distance < p.m ? bias[distance] : 0.0f);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float maximum = -INFINITY, denominator = 0.0f, acc = 0.0f;
    uint slot = start % p.capacity;
    for (uint t = start; t < end; ++t) {
        float s = score[t - start];
        float m = max(maximum, s);
        float prev = t == start ? 0.0f : metal::precise::exp(maximum - m);
        float w = metal::precise::exp(s - m);
        denominator = denominator * prev + w;
        acc = acc * prev + w * val[ulong(slot) * stride + kvh * 128 + tid];
        maximum = m;
        if (++slot == p.capacity) slot = 0;
    }
    device float *dst = parts + ulong(g) * 130;
    dst[tid] = acc;
    if (tid == 0) { dst[128] = maximum; dst[129] = denominator; }
}

kernel void mk_relative_split_merge(device float *arena [[buffer(0)]], device const float *parts [[buffer(1)]],
                                    constant MK &p [[buffer(8)]], uint lane [[thread_index_in_simdgroup]],
                                    uint qh [[threadgroup_position_in_grid]]) {
    uint h = qh % p.heads, row = p.row0 + qh / p.heads;
    float maximum = arena[p.s + row * p.every + h];
    float denominator = 1.0f, acc[8] = {};
    uint E = (p.dim + 31) / 32;
    device const float *src = parts + ulong(qh) * p.taps * (p.dim + 2);
    for (uint k = 0; k < p.taps; ++k, src += p.dim + 2) {
        if (src[p.dim + 1] == 0.0f) continue;
        float m = max(maximum, src[p.dim]);
        float prev = metal::precise::exp(maximum - m);
        float w = metal::precise::exp(src[p.dim] - m);
        denominator = denominator * prev + w * src[p.dim + 1];
        for (uint e = 0; e < E; ++e) {
            uint d = lane + 32 * e;
            if (d < p.dim) acc[e] = acc[e] * prev + w * src[d];
        }
        maximum = m;
    }
    device float *dst = arena + p.y + row * p.ys + h * p.dim;
    for (uint e = 0; e < E; ++e) {
        uint d = lane + 32 * e;
        if (d < p.dim) dst[d] = acc[e] / denominator;
    }
}
