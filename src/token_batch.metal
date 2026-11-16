// Launch batching only: a SIMD group owns one token and one original row tile.
// All tokens address the same read-only checkpoint descriptors (and GPU weight
// cache), but never share accumulators or scratch. Token is the fastest SIMD
// index, placing the same weight rows for different tokens next to each other
// in the grid for cache reuse. No FP32 expression, lane
// partition, reduction, SwiGLU, mixture, or residual order is changed.
// Batch count: 1..32. Row tiles remain 1/2/4/8; all math/schedules unchanged.
// Concatenate after kernels.metal, resident_plan.metal, and row_tiling.metal.
#ifndef FLINT_TOKEN_BATCH_METAL
#define FLINT_TOKEN_BATCH_METAL

// Host mirror: thirteen u64 GPU addresses, in this order (104 bytes, align 8).
struct TokenScratch {
    device float *x, *norm, *q, *k, *v, *attention, *projection, *activation, *router;
    device uint *route_ids;
    device float *route_weights;
    device uint *route_error;
    device float *logits;
};

inline bool token_batch_tile(uint tile) {
    return tile == 1 || tile == 2 || tile == 4 || tile == 8;
}
inline bool token_batch_count(uint count) { return count >= 1 && count <= 32; }
inline ulong token_batch_tiles(uint rows, uint tile) {
    return (ulong(rows) + tile - 1) / tile;
}
// The original helpers receive a uint scalar-thread index. Reject geometry
// that cannot be represented before narrowing or dereferencing token scratch.
inline bool token_batch_geometry(ulong tiles) {
    return tiles != 0 && tiles <= 0x08000000ul;
}

// p.m = token count, p.n = row tile. Grid: count*(Qtiles+Ktiles+Vtiles)*32.
kernel void token_batch_qkv(device const Matrix *matrices [[buffer(0)]],
                            device const TokenScratch *tokens [[buffer(1)]],
                            constant P &p [[buffer(8)]],
                            uint i [[thread_position_in_grid]],
                            uint lane [[thread_index_in_simdgroup]]) {
    if (!token_batch_count(p.m) || !token_batch_tile(p.n)) return;
    ulong tiles = token_batch_tiles(matrices[0].rows, p.n) +
                  token_batch_tiles(matrices[1].rows, p.n) +
                  token_batch_tiles(matrices[2].rows, p.n);
    if (!token_batch_geometry(tiles)) return;
    ulong simd = i / 32;
    if (simd >= tiles * p.m) return;
    uint token = uint(simd % p.m);
    device const TokenScratch &s = tokens[token];
    if (!s.norm || !s.q || !s.k || !s.v) return;
    uint local = uint(simd / p.m) * 32;
    switch (p.n) {
        case 1: staged_qkv<1>(matrices, s.norm, s.q, s.k, s.v, local, lane); break;
        case 2: staged_qkv<2>(matrices, s.norm, s.q, s.k, s.v, local, lane); break;
        case 4: staged_qkv<4>(matrices, s.norm, s.q, s.k, s.v, local, lane); break;
        case 8: staged_qkv<8>(matrices, s.norm, s.q, s.k, s.v, local, lane); break;
    }
}

// Gate/down P preserves the staged ABI: n=routed, m=shared descriptor index,
// k=activation stride, dim=hidden. heads=token count, bits=row tile.
// The caller validates descriptor allocation/route IDs (no expert count is
// available in this ABI), disjoint token allocations, and resource residency.
kernel void token_batch_gate_up(device const MLP *experts [[buffer(0)]],
                                device const TokenScratch *tokens [[buffer(1)]],
                                constant P &p [[buffer(8)]],
                                uint i [[thread_position_in_grid]],
                                uint lane [[thread_index_in_simdgroup]]) {
    if (!token_batch_count(p.heads) || !token_batch_tile(p.bits) || p.k == 0 ||
        p.n > STAGED_ROUTER_SLOTS) return;
    ulong route_tiles = token_batch_tiles(p.k, p.bits);
    ulong tiles = (ulong(p.n) + 1) * route_tiles;
    if (!token_batch_geometry(tiles)) return;
    ulong simd = i / 32;
    if (simd >= tiles * p.heads) return;
    uint token = uint(simd % p.heads);
    device const TokenScratch &s = tokens[token];
    if (!s.norm || !s.activation || (p.n && !s.route_ids)) return;
    uint local_tile = uint(simd / p.heads);
    uint route = uint(ulong(local_tile) / route_tiles);
    uint expert = route < p.n ? s.route_ids[route] : p.m;
    device const MLP &mlp = experts[expert];
    if (mlp.gate.rows > p.k || mlp.up.rows != mlp.gate.rows ||
        mlp.up.cols != mlp.gate.cols) return;
    uint local = local_tile * 32;
    switch (p.bits) {
        case 1: staged_gate_up<1>(experts, s.norm, s.route_ids, s.activation, p, local, lane); break;
        case 2: staged_gate_up<2>(experts, s.norm, s.route_ids, s.activation, p, local, lane); break;
        case 4: staged_gate_up<4>(experts, s.norm, s.route_ids, s.activation, p, local, lane); break;
        case 8: staged_gate_up<8>(experts, s.norm, s.route_ids, s.activation, p, local, lane); break;
    }
}

// Grid: count*ceil(hidden/tile)*32; writes residual s.x in original route order.
kernel void token_batch_down(device const MLP *experts [[buffer(0)]],
                             device const TokenScratch *tokens [[buffer(1)]],
                             constant P &p [[buffer(8)]],
                             uint i [[thread_position_in_grid]],
                             uint lane [[thread_index_in_simdgroup]]) {
    if (!token_batch_count(p.heads) || !token_batch_tile(p.bits) || p.k == 0 ||
        p.n > STAGED_ROUTER_SLOTS) return;
    ulong tiles = token_batch_tiles(p.dim, p.bits);
    if (!token_batch_geometry(tiles)) return;
    ulong simd = i / 32;
    if (simd >= tiles * p.heads) return;
    uint token = uint(simd % p.heads);
    device const TokenScratch &s = tokens[token];
    if (!s.activation || !s.x || (p.n && (!s.route_ids || !s.route_weights))) return;
    // Uniform within each SIMD; do not let malformed geometry cross a route's
    // activation stride or the residual allocation before calling the helper.
    for (uint route = 0; route <= p.n; ++route) {
        uint expert = route < p.n ? s.route_ids[route] : p.m;
        if (experts[expert].down.rows != p.dim || experts[expert].down.cols > p.k) return;
    }
    uint local = uint(simd / p.heads) * 32;
    switch (p.bits) {
        case 1: staged_down<1>(experts, s.activation, s.route_ids, s.route_weights, s.x, p, local, lane); break;
        case 2: staged_down<2>(experts, s.activation, s.route_ids, s.route_weights, s.x, p, local, lane); break;
        case 4: staged_down<4>(experts, s.activation, s.route_ids, s.route_weights, s.x, p, local, lane); break;
        case 8: staged_down<8>(experts, s.activation, s.route_ids, s.route_weights, s.x, p, local, lane); break;
    }
}

template<uint Rows>
inline void token_batch_matvec_rows(Matrix m, device const float *x, device float *out,
                                    uint tile, uint lane) {
    uint first = tile * Rows;
    float sums[Rows];
    matrix_dot_rows<Rows>(m, x, first, lane, sums);
    #pragma unroll
    for (uint r = 0; r < Rows; ++r) {
        float sum = simd_sum(sums[r]);
        if (lane == 0 && first + r < m.rows) out[first + r] = sum;
    }
}

// p.m=count, p.n=row tile, p.bits=0 attention->projection or 1 norm->logits.
// INT4/g64 ONLY. Dense router must retain its dedicated dispatch/reduction.
kernel void token_batch_matvec(device const Matrix &m [[buffer(0)]],
                               device const TokenScratch *tokens [[buffer(1)]],
                               constant P &p [[buffer(8)]],
                               uint i [[thread_position_in_grid]],
                               uint lane [[thread_index_in_simdgroup]]) {
    if (!token_batch_count(p.m) || !token_batch_tile(p.n) || p.bits > 1) return;
    if (m.reserved != 1 || m.bits != 4 || m.group_size != 64 ||
        m.cols == 0 || m.cols % 64 || !m.w || !m.scales || !m.biases) return;
    ulong tiles = token_batch_tiles(m.rows, p.n);
    if (!token_batch_geometry(tiles)) return;
    ulong simd = i / 32;
    if (simd >= tiles * p.m) return;
    uint token = uint(simd % p.m);
    device const TokenScratch &s = tokens[token];
    device const float *x = p.bits == 0 ? s.attention : s.norm;
    device float *out = p.bits == 0 ? s.projection : s.logits;
    if (!x || !out) return;
    uint tile = uint(simd / p.m);
    switch (p.n) {
        case 1: token_batch_matvec_rows<1>(m, x, out, tile, lane); break;
        case 2: token_batch_matvec_rows<2>(m, x, out, tile, lane); break;
        case 4: token_batch_matvec_rows<4>(m, x, out, tile, lane); break;
        case 8: token_batch_matvec_rows<8>(m, x, out, tile, lane); break;
    }
}
#endif
