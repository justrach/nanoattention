// ===== decode climb 2 (c2-decode): single-row decode variants, each behind a knob
// whose default leaves the reference schedule unchanged. Every kernel here
// reproduces the per-row expression trees of the kernels it replaces
// (kernels.metal, fused.metal, decode.metal) bit for bit; kernel_tests.zig
// checks each against its reference dispatch sequence.

// An empty launch, for measuring dispatch and barrier cost (nano d2-probe).
kernel void mk_d2_empty(device float *arena [[buffer(0)]], constant MK &p [[buffer(8)]],
                        uint i [[thread_position_in_grid]]) {
    if (p.flags == 0xffffffffu && i == 0) arena[0] = 1.0f;
}

// ---- gated pre-encoded decode (knob d2_gate) ----
// The token arrives packed with a generation number in one word, written by the
// host after the previous token was read back: gate[0] = gen << 20 | token.
// mk_d2_gate_packed holds the command buffer until it is written: one thread
// spins until gate[0] carries generation p.every, at most p.n iterations, and
// records the iterations in gate[2]. It touches nothing else, so no result
// depends on it; a gate that gives up early costs a recomputation (below).
// mk_d2_embed_gated is mk_embed with the token taken from that word; if the
// word does not carry the expected generation (the gate gave up first) every
// output is NaN, so the forward's argmax returns the non-finite sentinel, and
// gate[1] is set so the host knows to recompute the token synchronously.
kernel void mk_d2_gate_packed(device atomic_uint *gate [[buffer(0)]], constant MK &p [[buffer(8)]],
                              uint i [[thread_position_in_grid]]) {
    if (i != 0) return;
    uint it = 0;
    for (; it < p.n; ++it)
        if ((atomic_load_explicit(gate, memory_order_relaxed) >> 20) == p.every) break;
    atomic_store_explicit(gate + 2, it, memory_order_relaxed);
}
kernel void mk_d2_embed_gated(device const uchar *w [[buffer(0)]], device const uchar *s [[buffer(1)]],
                              device const uchar *b [[buffer(2)]], device float *arena [[buffer(3)]],
                              device atomic_uint *gate [[buffer(4)]],
                              constant MK &p [[buffer(8)]], uint i [[thread_position_in_grid]]) {
    uint r = i / p.n, c = i % p.n;
    if (r >= p.rows) return;
    uint row = p.row0 + r;
    uint word = atomic_load_explicit(gate, memory_order_relaxed);
    if ((word >> 20) != p.every) {
        if (c == 0) atomic_store_explicit(gate + 1, 1u, memory_order_relaxed);
        arena[p.y + row * p.ys + c] = NAN;
        return;
    }
    uint token = word & 0xfffffu;
    ulong index = ulong(token) * p.n + c;
    float v;
    if (p.bits == 0) v = mk_load(w, p.w, p.dtype, index);
    else {
        ulong group = index / p.group;
        float scale = mk_load(s, p.s, p.sdtype, group), bias = mk_load(b, p.b, p.bdtype, group);
        float code = mk_code(w, p.w, p.bits, index);
        if (p.flags & MK_ROUND_BF16) v = mk_bf16(mk_bf16(scale * code) + bias);
        else v = scale * code + bias;
    }
    arena[p.y + row * p.ys + c] = v;
}

// ---- sandwich norm with operands held in registers (knob d2_norm_regs) ----
// mk_norm_add_norm for n <= 4 * threads (one word group per thread): each thread
// loads its y, r and both weights once, keeps r in registers for the second
// norm and the output (mk_norm_add_norm stores r and reads it back), and writes
// r and out once. Same threads, same words, same expressions, same reductions.
kernel void mk_d2_norm_add_norm(device const uchar *w [[buffer(0)]], device float *arena [[buffer(1)]],
                                constant MK &p [[buffer(8)]], uint tgid [[threadgroup_position_in_grid]],
                                uint lid [[thread_position_in_threadgroup]], uint tg [[threads_per_threadgroup]],
                                uint simd [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float partial[33];
    if (tgid >= p.rows) return;
    uint row = p.row0 + tgid;
    device const float *y = arena + p.x + row * p.xs;
    device float *res = arena + p.y + row * p.ys;
    device float *out = arena + p.z + row * p.zs;
    uint base = lid * 4;
    float yv[4], rv[4], w0[4], w1[4];
    for (uint j = 0; j < 4; ++j) if (base + j < p.n) {
        yv[j] = y[base + j];
        rv[j] = res[base + j];
        w0[j] = mk_load(w, p.w, p.dtype, base + j);
        w1[j] = mk_load(w, p.s, p.sdtype, base + j);
    }
    float acc = 0;
    for (uint j = 0; j < 4; ++j) if (base + j < p.n) { float v = yv[j]; acc += v * v; }
    float inv = mk_reduce_inv(acc, p.n, p.eps, simd, lane, partial);
    acc = 0;
    for (uint j = 0; j < 4; ++j) if (base + j < p.n) {
        float v = yv[j] * inv;
        float t = w0[j] * v;
        float r = p.kind == 0 ? rv[j] + t : t + rv[j];
        rv[j] = r;
        acc += r * r;
    }
    float inv2 = mk_reduce_inv(acc, p.n, p.eps, simd, lane, partial);
    for (uint j = 0; j < 4; ++j) if (base + j < p.n) {
        float v = rv[j] * inv2;
        res[base + j] = rv[j];
        out[base + j] = w1[j] * v;
    }
}
