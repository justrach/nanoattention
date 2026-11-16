// Group multiple independent certified stripes without changing a head's
// SIMD reduction or online recurrence. Append after attention_gqa_reuse.metal.
template<uint H, uint Blocks>
inline void attention_mlx_grouped(device const float *queries, device const float *keys,
    device const float *values, device const uchar *sink, device float *states,
    constant P &p, uint3 grid, uint3 tptg, uint sg, uint lane) {
    static_assert((H == 2 || H == 4) && (Blocks == 2 || Blocks == 4), "Unsupported grouped geometry");
    constexpr uint simds = 8 / H;
    if (tptg.x != 32 * simds * Blocks || tptg.y != 1 || tptg.z != 1) return;
    // B=128/512/1024 is divisible by Blocks, so no group crosses a KV head.
    attention_mlx_gqa_reuse_dispatch<H>(queries, keys, values, sink, states, p,
        uint3(grid.x * Blocks + sg / simds, grid.y, grid.z),
        uint3(32 * simds, 1, 1), sg % simds, lane);
}

kernel void attention_mlx_split_gqa2_b2(
    device const float *queries [[buffer(0)]], device const float *keys [[buffer(1)]],
    device const float *values [[buffer(2)]], device const uchar *sink [[buffer(3)]],
    device float *states [[buffer(4)]], constant P &p [[buffer(8)]],
    uint3 grid [[threadgroup_position_in_grid]], uint3 tptg [[threads_per_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    attention_mlx_grouped<2, 2>(queries, keys, values, sink, states, p, grid, tptg, sg, lane);
}

kernel void attention_mlx_split_gqa2_b4(
    device const float *queries [[buffer(0)]], device const float *keys [[buffer(1)]],
    device const float *values [[buffer(2)]], device const uchar *sink [[buffer(3)]],
    device float *states [[buffer(4)]], constant P &p [[buffer(8)]],
    uint3 grid [[threadgroup_position_in_grid]], uint3 tptg [[threads_per_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    attention_mlx_grouped<2, 4>(queries, keys, values, sink, states, p, grid, tptg, sg, lane);
}

kernel void attention_mlx_split_gqa4_b4(
    device const float *queries [[buffer(0)]], device const float *keys [[buffer(1)]],
    device const float *values [[buffer(2)]], device const uchar *sink [[buffer(3)]],
    device float *states [[buffer(4)]], constant P &p [[buffer(8)]],
    uint3 grid [[threadgroup_position_in_grid]], uint3 tptg [[threads_per_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    attention_mlx_grouped<4, 4>(queries, keys, values, sink, states, p, grid, tptg, sg, lane);
}
