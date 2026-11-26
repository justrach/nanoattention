# Resident execution-plan certificate

This standalone Lean 4.35.0 project uses only `Std`: no packages, downloads,
extra axioms, `sorry`, `admit`, `native_decide`, or unsafe proof shortcuts.

From the repository root:

```sh
python3 tools/check_resident_proofs.py --generate  # intentional regeneration
python3 tools/check_resident_proofs.py             # build proofs + exact drift check
PYTHONDONTWRITEBYTECODE=1 python3 proofs/test_verifier.py
zig test --dep plan -Mroot=proofs/check_bindings.zig -Mplan=src/resident_plan.zig
(cd proofs && lake env lean Audit.lean)            # inspect proof axiom dependencies
(cd proofs && lake env lean TwoDsd.lean)           # 2DSD window / KV / rejection contracts
```

The verifier invokes `lake build` in this directory, executes the built Lean
emitter once per language, and byte-compares both outputs without rewriting
anything in verification mode. A missing/stale binding causes a nonzero exit.
The native resident build should run this verifier before compilation. Neither
binding should be reformatted independently of the Lean emitter.

## Bound interface

`Main.lean` emits `src/resident_plan.zig` and `src/resident_plan.metal` from the
same definitions in `ResidentProofs.lean` that the kernel checks. The Zig module
exports `threadgroup_size = 256`, `max_workers = 32`, `spin_limit = 10000000`,
`max_layers = 49`, `row_tiles = [_]u32{1, 2, 4, 8}`,
`router_lanes = 32`, `router_slots = 8`, `cached_router_experts = 256`,
`dense_simds = 8`, `dense_max_cols = 8192`,
`attention_chunks = [_]u32{128, 256, 512, 1024}`, `validAttentionChunk`,
`attention_max_context = 262144`, `attention_max_heads = 64`,
`attention_max_dim = 256`, `attention_simds = 8`, `attention_lanes = 32`,
`validRowTile(tile: u32) bool`, enum `Op`, and
`program(layers: u32) ?[]const u32`.
`program` returns a distinct immutable static array for each count **1..49**
and `null` otherwise. The Metal binding exports matching
`RESIDENT_THREADGROUP_SIZE`, `RESIDENT_MAX_WORKERS`, `RESIDENT_SPIN_LIMIT`,
`RESIDENT_MAX_LAYERS`, `RESIDENT_ROW_TILE_1/2/4/8` (each expands to its
numeric tile size), `STAGED_ROUTER_LANES`, `STAGED_ROUTER_SLOTS`,
`STAGED_CACHED_ROUTER_EXPERTS`, and enum `ResidentOp`. The tile list and Zig predicate
are emitted from `rowTiles`, the same definition used by `validRowTile` and
`row_tile_admissible`. Unsupported sizes return false, including zero.
The existing 1..49 schedules and all previous exported APIs are unchanged.

The actual schedule, not an opaque hash, is generated:

```
[1] ++ concat(layer in 0..layers-1: [layer*256+2, ..., layer*256+11]) ++ [12,13]
```

Opcodes: embedding=1, input_norm=2, qkv=3, qk_rope_cache=4, attention=5,
output_projection=6, residual_post_norm=7, router=8, route_topk=9,
gate_up=10, down_reduce=11, final_norm=12, head=13. Dense layers keep 8/9
as no-ops so their control schedule is identical. The interpreter must consume
this array and perform its inter-worker barrier after **every** instruction.

## Kernel-checked facts

* `layerProgram_length`, `layerPrograms_length`, `makePlan_length`: exactly
  ten instructions per layer and **10*layers+3** per program.
  `supported_length_bounds` establishes **13..493** instructions for 1..49 layers;
  `boundary_instructions` fixes embedding and the final normalization/head positions.
* `schedule_certificate`, `supported_schedule`: for every supported count,
  the concatenated plan equals an independently indexed specification. The
  finite certificate actually covers 0..49; the exported API excludes zero.
* `indexed_layer_instruction`, `supported_instruction_at`, and
  `supported_layer_instruction`: layer `l` opcode `o` (2..11) occupies zero-based
  position `10*l+o-1` in the actual emitted plan. `dependencies_in_layer` and
  `dependencies_between_layers` order all layer stages and adjacent layers.
* `instruction_bounds` proves byte decoding and UInt32 representability for
  layer instructions. `instruction_bounds_certificate` and
  `supported_instruction_bounds` cover **every** emitted instruction, including
  embedding/final/head: nonzero opcode at most 13, layer less than 49,
  unsigned 32-bit representability.
* `row_worker_coverage`, `row_worker_unique`: every row has exactly one
  worker/iteration decomposition `row = worker + iteration*workers` with
  `worker < workers`, provided `workers > 0`. These facts have no fixed row
  limit and use mathematical natural numbers, not wrapping machine arithmetic.
* `barrier_phase_advance_iff_last`, `barrier_nonlast_preserves_phase`,
  `barrier_last_resets`: an abstract arrival counter advances phase only on
  the last arrival and resets its count. `OneArrivalPerWorker` explicitly
  requires a duplicate-free list of valid worker IDs and a fresh, valid next
  arrival; the theorem also assumes the counter equals that list's length.
  `one_arrival_per_worker_bound` derives the counter bound from these premises.
  The simple counter transition itself does **not** track worker identity or
  prove that the native implementation maintains those premises.
* `binding_limits` checks the four interface constants and
  `opcode_table_complete` checks the numeric opcode table is exactly 1..13.

## Register-cached routing bounds

`router_register_ownership` proves that every expert below the generated
capacity has a valid lane/slot decomposition, with lane `< 32`, slot `< 8`,
and exact reconstruction `lane + slot*32 = expert`. `router_register_bounds`
proves the converse index bound for each valid lane/slot. The host uses the
emitted capacity to select the cached kernel or the generic fallback; the
shader's arrays and iteration limits use the matching emitted macros.

These facts certify the integer register layout, not top-k selection semantics,
sigmoid arithmetic, Metal shuffles, or compiler register allocation. Those are
checked empirically against the CPU reference and legacy GPU routing, including
ties, partial expert counts, full-capacity selection, and nonfinite inputs.

## MLX-layout and aligned-normalization certificates

The opt-in MLX-layout candidate exports `mlx_attention_dim = 128`, 32 lanes,
four adjacent elements per lane, 32 single/merge SIMDs, and admissible stripe
counts 64/128/512/1024. Its packed state stride is 130 floats: 128 numerator elements,
maximum at 128, denominator at 129. The shaders statically assert these
specializations against the generated Metal binding; host allocation and launch
sizes consume the corresponding Zig constants.

Certificates establish adjacent-element coverage/uniqueness, interleaved token
ownership `block + iteration*blocks`, guarded TK4 lookahead/advancement, GQA
head ownership, unique sink ownership by block zero, merge coverage, and packed
state UInt32 bounds (at most 8,519,680 float elements). Aligned normalization
also certifies `2048 = 512*4` and `128 = 32*4` element ownership. The axiom audit
and binding tests include these facts.

The generated `mlxLargeDeviceBlocks(length)` policy selects 128 stripes below
16,384 keys, 512 below 65,536, and 1024 thereafter. Integer certificates and CPU
binding tests cover its threshold neighbors and validity through the context
ceiling. This models the observed MLX `applegpu_g15d` GQA8 policy, not universal
Apple GPU tuning. The host gates contexts of at least 16K to the validated
M3 Ultra/GQA8 configuration and sizes scratch for the largest required stripe
count. Changing block counts also changes FP32 association, so the policy must
be validated independently with full-model numerical checks.

These are integer-layout proofs, **not proofs of FP32 agreement with MLX**.
Reduction association, sigmoid/exponential implementations, the Metal compiler,
cache residency, synchronization and measured speed remain separate empirical
acceptance requirements. Passing Lean does not turn a failed logit check into
a pass.

## Split-attention indexing certificates

`attentionChunks` defines the admitted 128/256/512/1024-token chunks; the
emitter exports the same list/predicate and matching Metal constants. The host
checks dimensions/context against the emitted bounds, and the shaders consume
the generated eight-SIMD, 32-lane geometry. Larger unsupported shapes retain
the prior attention path rather than using an uncertified split launch.

Relative to the effective sliding-window start, chunk `c`, SIMD `s`, iteration
`i` owns `width*c + s + 8*i`. Certificates establish constructive coverage,
unique ownership with within-chunk iteration bounds, nonempty chunk count,
partial-chunk masking, guarded absolute reads, and nonwrapping initialization
and loop advancement. Exactly chunk 0 / SIMD 0 owns the denominator-only sink.
Head-major state indexing and its `dim+2` fields have unique ownership and
UInt32 bounds; a separate theorem bounds row-major KV element indices under
the stated context/head/dimension premises. Byte allocations require wider
host arithmetic; the theorem does not claim a 16 GB float buffer is available.

Each producer merges its local eight SIMD states into an **unnormalized**
(maximum, denominator, weighted-value vector) record. A dependent merge dispatch
combines only this token's active chunk records and normalizes once. Lean does
not prove this floating-point algebra, implementation memory visibility, or
performance. GPU tests separately compare against independent FP64 softmax at
unchanged 1e-5 tolerance, including partial chunks, 1K/4K/8K, nonaligned sliding
starts, ring wrap, GQA, partial dimensions, and sink dtype/offset cases.

## Fused QKV and split dense projection certificates

The staged INT4 QKV launch concatenates independently rounded-up Q, K, and V
tile ranges. `Main.lean` renders `staged_qkv_projection` and
`staged_qkv_local_tile` directly from the expression definitions checked in
`ResidentProofs.lean`; Zig exports matching `stagedQkvProjection` and
`stagedQkvLocalTile`. Certificates establish half-open interval ownership,
local bounds, inverse coverage, reconstruction, and nonwrapping prefix sums
under the explicit total-tile bound. Existing tiled-row proofs then cover each
projection's guarded partial row tile. The shader consumes these helpers;
weights remain separate mapped tensors.

The split dense reduction uses eight SIMDs of 32 lanes, with columns
`simd * 32 + lane + iteration * 256`. Lean proves constructive coverage,
unique ownership, guarded read bounds, and nonwrapping advancement for widths
at most 8192. Generated `STAGED_DENSE_SIMDS` / `STAGED_DENSE_MAX_COLS` and the
matching Zig constants bind shader geometry and host eligibility. Compile-time
assertions tie eight SIMDs to the existing 256-thread launch constant.

Neither certificate proves FP32 reduction equivalence, Metal memory ordering,
compiler behavior, or throughput. The split reduction changes summation
association; CPU/GPU and full-model MLX comparisons remain separate empirical
obligations. Binding tests cover empty projection ranges, independently partial
QKV tiles, dense exact-once ownership, and integer boundary cases.

## Tiled row and packed INT4 certificates

For an abstract launch of **1..32 workers**, each with 256 threads and 32-lane
SIMDs, there are `simds = workers*8`, hence **8..256 SIMDs**. The worker maximum
bounds the currently admitted experiments; it does not certify simultaneous
residency, device-memory coherence, or eventual barrier completion. A global SIMD index
`s < simds` owns this tile in iteration `k`:

```
row0 = s*R + k*(simds*R)
row = row0 + j           # 0 <= j < R; access only when row < rows
R in {1,2,4,8}
```

* `row_tile_admissible` proves the checked predicate accepts exactly those
  four sizes; `row_tile_bounds` gives `0 < R <= 8`.
* `tiled_row_coverage` constructs the owner of any natural-number row:
  `s = (row/R)%simds`, `k = (row/R)/simds`, `j = row%R`.
  `guarded_tiled_coverage` additionally shows that a valid output row's
  owner passes the outer `row0 < rows` loop test, so the row guard cannot
  exclude a valid row. `tiled_row_unique` proves all three coordinates unique when
  `simds > 0`, `R > 0`, `s < simds`, and `j < R`.
  Together these give neither omissions nor duplicate ownership, including
  partial final tiles. They do not assert that hardware executes each owner.
* `tiled_row_guard` identifies the exact partial-tile mask
  `j < rows-row0` for a tested `row0 < rows`.
  `guarded_tiled_row_bounds` derives an output index below `rows` and `2^32`
  when `rows <= 1048576`. Native code must actually apply the guard.
* `staged_row_coverage` proves that `ceil(rows/R)` staged SIMDs cover every
  valid output row, including the final partial tile. `staged_row_u32_bounds`
  proves nonwrapping tile starts/candidates for `rows <= 1048576`; the final
  `candidate < rows` guard remains mandatory. Staged grids do not inherit the
  resident worker-count cap or require simultaneous group residency.
* `launch_simd_bounds` derives the positive SIMD count and its upper bound
  from the supported worker count. `tiled_u32_bounds` assumes
  `rows <= 1048576`, `simds <= 256`, a valid tile, `s < simds`,
  `start < rows`, and `j < R`. It proves stride at most **2048**, the initial
  `s*R` representable, and both `start+j` and the **next** loop start
  `start+simds*R` below `2^32` (also at most `rows+simds*R`).
  `tiled_iteration_u32` bounds the multiplication `k*(simds*R)` and its sum
  with `s*R` for an iteration whose tile start is below `rows`.
  These are natural-number bounds supporting a nonwrapping implementation,
  not a blanket theorem about arbitrary UInt32 counter expressions.
* `tiled_row_result` transports **any opaque per-row result function** through
  the ownership mapping unchanged. It justifies rescheduling the same row
  computation, not changing its evaluation order or combining row reductions.
* `int4_chunk_indices` proves that a 64-weight group contains eight packed
  words of eight weights, with word/nibble quotient-remainder reconstruction.
  `int4_group_word_bounds` bounds group and word indices assuming
  `column < columns` and `columns % 64 = 0`.
* `int4_pair_indices`, `int4_pair_group`, and `int4_pair_word_bounds` check
  the 16-input/two-packed-word lane layout: exact word/column reconstruction,
  one shared group-64 scale/bias pair for the entire segment, and bounded word
  loads under the aligned-column premise. These are integer properties, not
  proof that changing lane/reduction association preserves IEEE754 rounding.
* `int4Code word lane = (word / 16^lane) % 16` specifies low-nibble-first
  extraction. `int4_code_bounds` gives **0..15** for every lane;
  `int4_low_nibble_first` fixes lane zero to `word%16`.
  `int4_split_reconstruction` gives exact quotient/remainder reconstruction
  at any nibble boundary. `int4_word_reconstruction` proves all eight
  extracted nibbles reconstruct a word when `word < 2^32`.
* `int4_affine_sum` proves exact distributivity for any list of mathematical
  integer `(code, activation)` pairs, including lists of length eight:
  `sum((scale*code+bias)*x) = scale*sum(code*x)+bias*sum(x)`.
  **It is not an IEEE754 equivalence and does not authorize reassociation
  of native floating-point operations.**

## Assumptions and nonclaims

These are integer schedule and abstract control-state theorems, **not a proof
of the shader or a proof of performance**. Native code must use the generated
values/schedules faithfully, decode instructions correctly, implement each
opcode correctly, preserve the one-arrival-per-worker invariant, and barrier
between instructions. The tiled row-index bounds and ideal INT4 packing facts above are conditional
certificates, not proofs of arbitrary shader accesses. Device allocation sizes,
row/column strides and address arithmetic, the correspondence of Metal bitwise
shifts/masks to `int4Code`, actual threadgroup/lane configuration, all other
integer counters, and spin-limit failure handling remain implementation
obligations.

The generated native emitter and textual artifact check remain part of the
binding trust boundary: Lean does not prove the behavior of Zig/Metal parsers
or the operating system's file writes. `Audit.lean` reports the standard Lean
logical axioms used by generic tactic proofs; the two finite schedule/bounds
certificates themselves depend on no axioms.

The model does not establish Metal acquire/release semantics, visibility of
non-atomic device buffers after atomic operations, compiler correctness,
float/half rounding or numerical agreement, lack of data races, sufficient
simultaneously resident threadgroups, GPU scheduling fairness, eventual
arrival, or successful completion before the spin limit. There is no
unconditional GPU-residency/liveness theorem. On-device execution, numerical
checks, timeout handling and performance measurements remain necessary.

## Inkling 2 Mini megakernel certificates (`proofs/Inkling.lean`)

`proofs/Inkling.lean` (`Std`-only, zero `sorry`/`admit`/`native_decide`, standard axioms only) formalizes integer indexing, bit-decode, and overlap contracts for the Inkling 2 Mini (`inkling2_loop`) decode and prefill kernels in **49 public theorems** (`53` total including 4 private helper lemmas). `proofs/Main.lean` emits matching compile-time constants and inline Metal helpers into `src/mega/inkling_plan.zig` and `src/mega/inkling_plan.metal`, verified against `tools/check_resident_proofs.py` on every build:

- **Schedule & state-slot injectivity:** `inkling_visit_count` (48 logical visits over 24 physical layers), `inkling_physical_layer_bounds`, and `inkling_state_slot_injective` (each logical visit receives a distinct persistent state slot).
- **Speculative `d2_gate` ring & token packing:** `d2_gate_ring_disjoint` (stalled/recomputed ahead-decode cannot overwrite live ring horizon entries when `capacity >= horizon + 2`) and `d2_token_pack_roundtrip` (`gen << 20 | token` bijective packing in the 32-bit gate word, `d2GenShift = 20`, `d2MaxVocab = 2^20`).
- **Short-convolution ring step:** `ring_back_bounds`, `ring_back_eq_mod` (branchless `inkling_ring_back` equals `(slot + capacity - back) % capacity` without 64-bit division), and `ring_back_ne_current`.
- **Sub-SIMD NVFP4 gate+up & down decode:** `nvfp4_bit_decode_exact` (branchless E2M1 bit-manipulation in `inkling_nvfp4_decode4` equals the 16-entry LUT identically), `nvfp4_subsimd_coverage`, `nvfp4_subsimd_unique`, `nvfp4_subsimd_scale_block`, `nvfp4_k_divisibility` (`16`-lane sub-SIMD gate+up), and `nvfp4_subsimd8_coverage`, `nvfp4_subsimd8_unique`, `nvfp4_subsimd8_scale_block`, `nvfp4_down_k_divisibility` (`8`-lane sub-SIMD down projection).
- **Fused multi-projection & split-K BF16 decode:** `multi_proj_local_bounds`, `multi_proj_inverse_coverage`, `multi_proj_u32_bounds` (covering `inkling_multi_proj_index` and `inkling_multi_proj_local_tile` up to 6 matrices), plus `splitk_bf16_coverage` and `splitk_bf16_unique` (`8`-SIMD × `32`-lane vec4 stride partitioning).
- **Register-resident depth attention & RMSNorm:** `depth_reg_2304_exact`, `depth_reg_ownership`, `depth_vec_coverage`, `depth_reg_simds`, and `depth_reg_unique` (`576` threads = `18` SIMDs × `32` lanes × `4` elements covering `hidden = 2304`).
- **Striped GQA3 & prefill far-band relative attention:** `gqa3_ownership` (`3` query heads per KV head), `stripe_partition_bounds`, `stripe_partition_coverage`, `stripe_partition_disjoint` (for `inkling_stripe_start`/`inkling_stripe_end`), `global_linear_slot`, and `prefill_far_band_exact` (`inkling_far_band_end` proves that keys below the boundary are strictly causal `<= diag`, past `rel_extent`, and fit linearly without ring modulo when `far_end <= capacity`).
- **Two-stage decode argmax & `N=64` expert prefill tiles:** `argmax_part_coverage`, `argmax_part_unique` (`64` partition groups), `expert_block_n64_divisibility`, `expert_block_n64_coverage`, `expert_block_n64_unique`, and `expert_tail_partition` (unique part assignment for partial `tail` tiles across `parts` striped workgroups).
- **Constant LUT FP8/FP4 bit-decode (`expert_fp_lut`):** `e4m3_bit_decode_nan`, `e4m3_bit_decode_zero`, `e4m3_bit_decode_u32`, `fp4_pair_decompose`, `fp4_bit_decode_u32`, and `fp4_pair_lut_exact` (exact byte/word equivalence for E4M3FN scales and packed FP4 pairs).
- **Compact MoE tile packing & bounds (`expert_compact_tiles`):** `moe_tile_pack_roundtrip` and `moe_tile_count_bound` (with 4 private sum-decomposition helper lemmas) backing `inkling_moe_tile_block`, `inkling_moe_pack_tile_r`, `inkling_moe_unpack_bm`, `inkling_moe_unpack_tid_y`, and `inkling_moe_max_tiles`.
- **Batched QKVR sconv ring safety & inline relative bias skip (`fuse_qkvr_prep`, `relative_inline_bias`):** `sconv_batch_ring_disjoint` and `inline_rel_skip_far`.

As with the resident certificates, these prove integer schedule/index/bit-decode invariants only — FP32/FP16/BF16 reassociation, Metal barrier ordering, memory coherence, and runtime speedup are validated empirically by the `nano` unit tests, real-prompt logit parity checks, and `tools/nano_inkling_attention_check.py`.
