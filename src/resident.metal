// Resident decode and phase-counter barrier design adapted from recursive-oss/rec
// (Apache-2.0). See THIRD_PARTY_NOTICES.md and licenses/rec-APACHE-2.0.txt.
// Modified for native Zig descriptors, FP32 arithmetic, a Lean-certified plan,
// generic tiny-model validation, and bounded, opt-in worker counts.
// The abstract proof does NOT establish GPU residency, Metal memory semantics,
// compiler correctness, or floating-point equivalence. Those remain assumptions/tests.

struct ResidentLayer {
    Matrix input_norm, post_norm, q, k, v, o, q_norm, k_norm, sink, router;
    device const MLP *experts;
    device const float *bias;
    device float *keys;
    device float *values;
    uint capacity, window, rope, routed, shared_index, stride;
};
struct ResidentDescriptor {
    Matrix embedding, final_norm, head;
    device const ResidentLayer *layers;
    device float *x, *norm, *q, *k, *v, *attention, *projection, *activation, *router;
    device uint *route_ids;
    device float *route_weights, *logits;
    uint hidden, heads, kv_heads, dim, layer_count, position, token, workers;
    float eps, theta;
    uint row_tile, reserved;
};

// A bounded arrival/phase barrier, NOT a portable guarantee that GPU workgroups
// are co-resident. A timeout poisons the command; the host must reject all output.
inline bool resident_sync(device atomic_uint *control, uint tid, uint workers, uint phase,
                          threadgroup uint *ready) {
    atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
    threadgroup_barrier(mem_flags::mem_device);
    if (tid == 0) {
        uint arrived = atomic_fetch_add_explicit(control, 1u, memory_order_relaxed);
        if (arrived + 1 == workers) {
            atomic_store_explicit(control, 0u, memory_order_relaxed);
            atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
            atomic_store_explicit(control + 1, phase + 1, memory_order_relaxed);
        }
        uint spins = 0;
        while (atomic_load_explicit(control + 1, memory_order_relaxed) == phase &&
               atomic_load_explicit(control + 2, memory_order_relaxed) == 0 && spins < RESIDENT_SPIN_LIMIT) ++spins;
        ready[0] = atomic_load_explicit(control + 1, memory_order_relaxed) == phase + 1 &&
                   atomic_load_explicit(control + 2, memory_order_relaxed) == 0;
        if (!ready[0]) atomic_fetch_max_explicit(control + 2, 1u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
    return ready[0] != 0;
}

inline void resident_matvec_scalar(device const Matrix &m, device const float *x, device float *out,
                            uint simd, uint lane, uint simds) {
    for (uint row = simd; row < m.rows; row += simds) {
        float value = simd_sum(matrix_dot(m, x, row, lane));
        if (lane == 0) out[row] = value;
    }
}
inline void resident_norm(device const Matrix &w, device const float *x, device float *out,
                          uint dim, float eps, uint simd, uint lane) {
    if (simd != 0) return;
    float sum = 0;
    for (uint d = lane; d < dim; d += 32) sum += x[d] * x[d];
    float inv = rsqrt(simd_sum(sum) / float(dim) + eps);
    for (uint d = lane; d < dim; d += 32) out[d] = x[d] * inv * weight(w.w, 0, w.dtype, d);
}
inline void resident_qk(device const ResidentDescriptor &s, device const ResidentLayer &l,
                        uint simd, uint lane, uint simds) {
    for (uint head = simd; head < s.heads + s.kv_heads; head += simds) {
        bool query = head < s.heads;
        uint local = query ? head : head - s.heads;
        device float *x = query ? s.q : s.k;
        device const Matrix &w = query ? l.q_norm : l.k_norm;
        uint base = local * s.dim;
        float sum = 0;
        for (uint d = lane; d < s.dim; d += 32) sum += x[base + d] * x[base + d];
        float inv = rsqrt(simd_sum(sum) / float(s.dim) + s.eps);
        uint half_dim = s.dim / 2;
        for (uint d = lane; d < half_dim; d += 32) {
            float a = x[base + d] * inv * weight(w.w, 0, w.dtype, d);
            float b = x[base + d + half_dim] * inv * weight(w.w, 0, w.dtype, d + half_dim);
            if (l.rope) {
                float angle = float(s.position) * pow(s.theta, -2.0f * float(d) / float(s.dim));
                float c = cos(angle), sn = sin(angle), first = a * c - b * sn;
                b = b * c + a * sn; a = first;
            }
            x[base + d] = a; x[base + d + half_dim] = b;
            if (!query) {
                uint at = (s.position % l.capacity) * s.kv_heads * s.dim + base + d;
                l.keys[at] = a; l.keys[at + half_dim] = b;
            }
        }
        if (!query) for (uint d = lane; d < s.dim; d += 32) {
            uint at = (s.position % l.capacity) * s.kv_heads * s.dim + base + d;
            l.values[at] = s.v[base + d];
        }
    }
}
inline void resident_attention(device const ResidentDescriptor &s, device const ResidentLayer &l,
                               uint simd, uint lane, uint simds) {
    for (uint head = simd; head < s.heads; head += simds) {
        uint kv_head = head / (s.heads / s.kv_heads);
        uint start = l.window && s.position + 1 > l.window ? s.position + 1 - l.window : 0;
        float maximum = weight(l.sink.w, 0, l.sink.dtype, head), denom = 1;
        float acc[8] = {0,0,0,0,0,0,0,0};
        for (uint token = start; token <= s.position; ++token) {
            uint base = (token % l.capacity) * s.kv_heads * s.dim + kv_head * s.dim;
            float dot = 0;
            for (uint d = lane; d < s.dim; d += 32) dot += s.q[head * s.dim + d] * l.keys[base + d];
            float score = simd_sum(dot) / sqrt(float(s.dim));
            float next_max = max(maximum, score), old_scale = exp(maximum - next_max), prob = exp(score - next_max);
            denom = denom * old_scale + prob;
            for (uint d = lane, j = 0; d < s.dim; d += 32, ++j) acc[j] = acc[j] * old_scale + prob * l.values[base + d];
            maximum = next_max;
        }
        for (uint d = lane, j = 0; d < s.dim; d += 32, ++j) s.attention[head * s.dim + d] = acc[j] / denom;
    }
}
inline void resident_post_norm(device const ResidentDescriptor &s, device const ResidentLayer &l, uint simd, uint lane) {
    if (simd != 0) return;
    float sum = 0;
    for (uint d = lane; d < s.hidden; d += 32) { float v = s.x[d] + s.projection[d]; sum += v * v; }
    float inv = rsqrt(simd_sum(sum) / float(s.hidden) + s.eps);
    for (uint d = lane; d < s.hidden; d += 32) {
        float value = s.x[d] + s.projection[d];
        s.x[d] = value;
        s.norm[d] = value * inv * weight(l.post_norm.w, 0, l.post_norm.dtype, d);
    }
}

inline void resident_route(device const ResidentDescriptor &s, device const ResidentLayer &l,
                           device atomic_uint *control, uint simd, uint lane) {
    if (simd != 0 || !l.routed) return;
    float total = 0;
    for (uint k = 0; k < l.routed; ++k) {
        float best = -INFINITY; uint best_id = l.router.rows;
        for (uint e = lane; e < l.router.rows; e += 32) {
            bool used = false;
            for (uint j = 0; j < k; ++j) used |= s.route_ids[j] == e;
            if (used) continue;
            if (!isfinite(s.router[e]) || !isfinite(l.bias[e])) {
                atomic_store_explicit(control + 2, 2u, memory_order_relaxed); continue;
            }
            float choice = 1.0f / (1.0f + exp(-s.router[e])) + l.bias[e];
            if (choice > best || (choice == best && e < best_id)) { best = choice; best_id = e; }
        }
        float top = simd_max(best);
        uint winner = simd_min(best == top ? best_id : l.router.rows);
        if (winner >= l.router.rows) winner = 0;
        float w = 1.0f / (1.0f + exp(-s.router[winner]));
        if (!isfinite(w)) w = 0;
        if (lane == 0) { s.route_ids[k] = winner; s.route_weights[k] = w; }
        total += w;
        simdgroup_barrier(mem_flags::mem_device);
    }
    if (lane == 0) for (uint k = 0; k < l.routed; ++k) s.route_weights[k] /= total + 1e-20f;
}
inline void resident_gate_up_scalar(device const ResidentDescriptor &s, device const ResidentLayer &l,
                             uint simd, uint lane, uint simds) {
    for (uint flat = simd; flat < (l.routed + 1) * l.stride; flat += simds) {
        uint route = flat / l.stride, row = flat % l.stride;
        uint expert = route < l.routed ? s.route_ids[route] : l.shared_index;
        device const MLP &mlp = l.experts[expert];
        if (row >= mlp.gate.rows) continue;
        float gate = simd_sum(matrix_dot(mlp.gate, s.norm, row, lane));
        float up = simd_sum(matrix_dot(mlp.up, s.norm, row, lane));
        if (lane == 0) s.activation[flat] = (gate / (1.0f + exp(-gate))) * up;
    }
}
inline void resident_down_scalar(device const ResidentDescriptor &s, device const ResidentLayer &l,
                          uint simd, uint lane, uint simds) {
    for (uint row = simd; row < s.hidden; row += simds) {
        float output = lane == 0 ? s.x[row] : 0;
        for (uint route = 0; route <= l.routed; ++route) {
            uint expert = route < l.routed ? s.route_ids[route] : l.shared_index;
            float sum = simd_sum(matrix_dot(l.experts[expert].down, s.activation + route * l.stride, row, lane));
            if (lane == 0) output += sum * (route < l.routed ? s.route_weights[route] : 1.0f);
        }
        if (lane == 0) s.x[row] = output;
    }
}

template<uint Rows>
inline void resident_matvec(device const Matrix &m, device const float *x, device float *out,
                            uint simd, uint lane, uint simds) {
    if constexpr (Rows == 1) resident_matvec_scalar(m, x, out, simd, lane, simds);
    else row_tile_matvec<Rows>(m, x, out, simd, lane, simds);
}
template<uint Rows>
inline void resident_gate_up_rows(device const ResidentDescriptor &s, device const ResidentLayer &l,
                                  uint simd, uint lane, uint simds) {
    // Tile the work-unit domain, not raw activations: tails must never cross
    // a route's stride boundary (tiny supported models need this guard too).
    uint tiles_per_route = (l.stride + Rows - 1) / Rows;
    for (uint tile = simd; tile < (l.routed + 1) * tiles_per_route; tile += simds) {
        uint route = tile / tiles_per_route, first = (tile % tiles_per_route) * Rows;
        uint expert = route < l.routed ? s.route_ids[route] : l.shared_index;
        device const MLP &mlp = l.experts[expert];
        if (first >= mlp.gate.rows) continue;
        float gates[Rows], ups[Rows];
        matrix_dot_rows<Rows>(mlp.gate, s.norm, first, lane, gates);
        matrix_dot_rows<Rows>(mlp.up, s.norm, first, lane, ups);
        #pragma unroll
        for (uint r = 0; r < Rows; ++r) {
            float gate = simd_sum(gates[r]), up = simd_sum(ups[r]);
            if (lane == 0 && first + r < mlp.gate.rows)
                s.activation[route * l.stride + first + r] = (gate / (1.0f + exp(-gate))) * up;
        }
    }
}
template<uint Rows>
inline void resident_down_rows(device const ResidentDescriptor &s, device const ResidentLayer &l,
                               uint simd, uint lane, uint simds) {
    for (uint first = simd * Rows; first < s.hidden; first += simds * Rows) {
        float output[Rows];
        #pragma unroll
        for (uint r = 0; r < Rows; ++r) output[r] = lane == 0 && first + r < s.hidden ? s.x[first + r] : 0;
        for (uint route = 0; route <= l.routed; ++route) {
            uint expert = route < l.routed ? s.route_ids[route] : l.shared_index;
            float sums[Rows];
            matrix_dot_rows<Rows>(l.experts[expert].down, s.activation + route * l.stride, first, lane, sums);
            #pragma unroll
            for (uint r = 0; r < Rows; ++r) {
                float sum = simd_sum(sums[r]);
                if (lane == 0) output[r] += sum * (route < l.routed ? s.route_weights[route] : 1.0f);
            }
        }
        #pragma unroll
        for (uint r = 0; r < Rows; ++r) if (lane == 0 && first + r < s.hidden) s.x[first + r] = output[r];
    }
}
template<uint Rows>
inline void resident_gate_up(device const ResidentDescriptor &s, device const ResidentLayer &l,
                             uint simd, uint lane, uint simds) {
    if constexpr (Rows == 1) resident_gate_up_scalar(s, l, simd, lane, simds);
    else resident_gate_up_rows<Rows>(s, l, simd, lane, simds);
}
template<uint Rows>
inline void resident_down(device const ResidentDescriptor &s, device const ResidentLayer &l,
                          uint simd, uint lane, uint simds) {
    if constexpr (Rows == 1) resident_down_scalar(s, l, simd, lane, simds);
    else resident_down_rows<Rows>(s, l, simd, lane, simds);
}

// Exactly one kernel invocation consumes the Lean-emitted schedule. This is a
// one-token path; sampling and multi-token packets are intentionally separate.
template<uint Rows>
inline void resident_decode(device const ResidentDescriptor &s, device const uint *program,
                            device atomic_uint *control, constant P &p,
                            uint tid, uint group, uint local_simd, uint lane, threadgroup uint *ready) {
    static_assert(Rows == RESIDENT_ROW_TILE_1 || Rows == RESIDENT_ROW_TILE_2 ||
                  Rows == RESIDENT_ROW_TILE_4 || Rows == RESIDENT_ROW_TILE_8, "Uncertified row tile");
    uint index = group * RESIDENT_THREADGROUP_SIZE + tid;
    // Descriptor-wide guard: every thread exits before the first barrier on a
    // mismatch. Never silently run a different tile than the host validated.
    if (s.row_tile != Rows) {
        if (index == 0) atomic_store_explicit(control + 2, 3u, memory_order_relaxed);
        return;
    }
    uint threads = s.workers * RESIDENT_THREADGROUP_SIZE;
    uint simd = group * (RESIDENT_THREADGROUP_SIZE / 32) + local_simd;
    uint simds = s.workers * (RESIDENT_THREADGROUP_SIZE / 32);
    for (uint phase = 0; phase < p.n; ++phase) {
        uint instruction = program[phase], opcode = instruction & 255u, layer = instruction >> 8;
        if (opcode >= RESIDENT_INPUT_NORM && opcode <= RESIDENT_DOWN_REDUCE && layer >= s.layer_count) {
            if (index == 0) atomic_store_explicit(control + 2, 3u, memory_order_relaxed);
            if (!resident_sync(control, tid, s.workers, phase, ready)) return;
        }
        device const ResidentLayer &l = s.layers[layer];
        switch (opcode) {
            case RESIDENT_EMBEDDING:
                for (uint d = index; d < s.hidden; d += threads) s.x[d] = matrix_weight(s.embedding, ulong(s.token) * s.hidden + d);
                break;
            case RESIDENT_INPUT_NORM:
                resident_norm(l.input_norm, s.x, s.norm, s.hidden, s.eps, simd, lane); break;
            case RESIDENT_QKV:
                resident_matvec<Rows>(l.q, s.norm, s.q, simd, lane, simds);
                resident_matvec<Rows>(l.k, s.norm, s.k, simd, lane, simds);
                resident_matvec<Rows>(l.v, s.norm, s.v, simd, lane, simds); break;
            case RESIDENT_QK_ROPE_CACHE:
                resident_qk(s, l, simd, lane, simds); break;
            case RESIDENT_ATTENTION:
                resident_attention(s, l, simd, lane, simds); break;
            case RESIDENT_OUTPUT_PROJECTION:
                resident_matvec<Rows>(l.o, s.attention, s.projection, simd, lane, simds); break;
            case RESIDENT_RESIDUAL_POST_NORM:
                resident_post_norm(s, l, simd, lane); break;
            case RESIDENT_ROUTER:
                if (l.routed) resident_matvec<Rows>(l.router, s.norm, s.router, simd, lane, simds); break;
            case RESIDENT_ROUTE_TOPK:
                resident_route(s, l, control, simd, lane); break;
            case RESIDENT_GATE_UP:
                resident_gate_up<Rows>(s, l, simd, lane, simds); break;
            case RESIDENT_DOWN_REDUCE:
                resident_down<Rows>(s, l, simd, lane, simds); break;
            case RESIDENT_FINAL_NORM:
                resident_norm(s.final_norm, s.x, s.norm, s.hidden, s.eps, simd, lane); break;
            case RESIDENT_HEAD:
                resident_matvec<Rows>(s.head, s.norm, s.logits, simd, lane, simds); break;
            default:
                if (index == 0) atomic_store_explicit(control + 2, 3u, memory_order_relaxed);
        }
        if (!resident_sync(control, tid, s.workers, phase, ready)) return;
    }
    if (index == 0) atomic_store_explicit(control + 3, 1u, memory_order_relaxed);
}

// Each entrypoint instantiates only its certified row tile, so unrelated row
// accumulators and runtime tile branches do not enter the resident interpreter.
#define RESIDENT_DECODE_ENTRY(Rows) \
kernel void resident_decode_t##Rows(device const ResidentDescriptor &s [[buffer(0)]], \
                                    device const uint *program [[buffer(1)]], \
                                    device atomic_uint *control [[buffer(2)]], constant P &p [[buffer(8)]], \
                                    uint tid [[thread_index_in_threadgroup]], uint group [[threadgroup_position_in_grid]], \
                                    uint local_simd [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) { \
    threadgroup uint ready[1]; \
    resident_decode<Rows>(s, program, control, p, tid, group, local_simd, lane, ready); \
}
RESIDENT_DECODE_ENTRY(1)
RESIDENT_DECODE_ENTRY(2)
RESIDENT_DECODE_ENTRY(4)
RESIDENT_DECODE_ENTRY(8)
#undef RESIDENT_DECODE_ENTRY

// Isolates phase-counter progress and fail-closed abort handling before model tests.
kernel void resident_barrier_probe(device atomic_uint *control [[buffer(0)]], device uint *diagnostic [[buffer(1)]],
                                   constant P &p [[buffer(8)]], uint tid [[thread_index_in_threadgroup]],
                                   uint group [[threadgroup_position_in_grid]]) {
    threadgroup uint ready[1];
    for (uint phase = 0; phase < p.n; ++phase) {
        if (!resident_sync(control, tid, p.m, phase, ready)) {
            if (tid == 0) diagnostic[group] = phase + 1;
            return;
        }
    }
    if (tid == 0) diagnostic[group] = 0;
}
