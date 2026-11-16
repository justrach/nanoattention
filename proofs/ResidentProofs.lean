import Std

/-!
Integer execution-plan certificates for the resident Flint interpreter.
No assertion here verifies floating point arithmetic, the Metal compiler/memory
model, occupancy, scheduling fairness, or an actual GPU barrier implementation.
-/
namespace Resident

set_option maxRecDepth 100000
set_option maxHeartbeats 0

 def threadgroupSize : Nat := 256
 def maxWorkers : Nat := 32
 def spinLimit : Nat := 10000000
 def maxLayers : Nat := 49

/-- The opcode table is also the source for both generated language bindings. -/
def operations : List (String × Nat) :=
  [("embedding", 1), ("input_norm", 2), ("qkv", 3),
   ("qk_rope_cache", 4), ("attention", 5), ("output_projection", 6),
   ("residual_post_norm", 7), ("router", 8), ("route_topk", 9),
   ("gate_up", 10), ("down_reduce", 11), ("final_norm", 12), ("head", 13)]

theorem opcode_table_complete :
    operations.map Prod.snd = (List.range 13).map (· + 1) := by decide

def encode (layer opcode : Nat) : Nat := layer * 256 + opcode

def layerProgram (layer : Nat) : List Nat :=
  (List.range 10).map (fun offset => encode layer (offset + 2))

def layerPrograms : Nat → List Nat
  | 0 => []
  | n + 1 => layerPrograms n ++ layerProgram n

/-- Dense and routed layers deliberately share this schedule: 8/9 are no-ops
for dense layers, not omitted instructions. -/
def makePlan (layers : Nat) : List Nat :=
  [1] ++ layerPrograms layers ++ [12, 13]

theorem layerProgram_length (layer : Nat) : (layerProgram layer).length = 10 := by
  simp [layerProgram]

theorem layerPrograms_length (layers : Nat) : (layerPrograms layers).length = 10 * layers := by
  induction layers with
  | zero => simp [layerPrograms]
  | succ n ih => simp [layerPrograms, ih, layerProgram_length]; omega

theorem makePlan_length (layers : Nat) : (makePlan layers).length = 10 * layers + 3 := by
  simp [makePlan, layerPrograms_length]

theorem supported_length_bounds (layers : Nat)
    (hmin : 1 ≤ layers) (hmax : layers ≤ maxLayers) :
    13 ≤ (makePlan layers).length ∧ (makePlan layers).length ≤ 493 := by
  rw [makePlan_length]
  unfold maxLayers at hmax
  omega

/-- Per-position specification: embedding; then each layer's 2..11 in
increasing order; then final normalization and head. -/
def instructionAt (layers position : Nat) : Nat :=
  if position = 0 then 1
  else if position < 10 * layers + 1 then
    encode ((position - 1) / 10) ((position - 1) % 10 + 2)
  else if position = 10 * layers + 1 then 12 else 13

def indexedPlan (layers : Nat) : List Nat :=
  (List.range (10 * layers + 3)).map (instructionAt layers)

/-- A finite, kernel-reduced certificate covers every exported layer count.
The independent indexed specification fixes both ordering and multiplicity. -/
theorem schedule_certificate :
    ∀ layers ∈ List.range 50, makePlan layers = indexedPlan layers := by decide

theorem supported_schedule (layers : Nat) (h : layers ≤ maxLayers) :
    makePlan layers = indexedPlan layers := by
  apply schedule_certificate layers
  simp only [List.mem_range]
  simp only [maxLayers] at h
  omega

theorem supported_instruction_at (layers position : Nat)
    (hl : layers ≤ maxLayers) (hp : position < 10 * layers + 3) :
    (makePlan layers)[position]? = some (instructionAt layers position) := by
  rw [supported_schedule layers hl]
  simp only [indexedPlan, List.getElem?_map, List.getElem?_range hp, Option.map_some]

theorem boundary_instructions (layers : Nat) (hl : layers ≤ maxLayers) :
    (makePlan layers)[0]? = some 1 ∧
    (makePlan layers)[10 * layers + 1]? = some 12 ∧
    (makePlan layers)[10 * layers + 2]? = some 13 := by
  constructor
  · rw [supported_instruction_at layers 0 hl (by omega)]
    simp [instructionAt]
  · constructor
    · rw [supported_instruction_at layers (10 * layers + 1) hl (by omega)]
      simp [instructionAt]
    · rw [supported_instruction_at layers (10 * layers + 2) hl (by omega)]
      simp [instructionAt, show ¬10 * layers + 2 < 10 * layers + 1 by omega]

/-- The position of a layer operation in the entire program (zero based). -/
def operationPosition (layer opcode : Nat) : Nat := 10 * layer + opcode - 1

theorem dependencies_in_layer (layer a b : Nat)
    (ha : 2 ≤ a) (hab : a < b) :
    operationPosition layer a < operationPosition layer b := by
  unfold operationPosition
  omega

theorem dependencies_between_layers (layer : Nat) :
    operationPosition layer 11 < operationPosition (layer + 1) 2 := by
  unfold operationPosition
  omega

theorem indexed_layer_instruction (layers layer opcode : Nat)
    (hl : layer < layers) (ho : 2 ≤ opcode) (ho' : opcode ≤ 11) :
    instructionAt layers (operationPosition layer opcode) = encode layer opcode := by
  unfold instructionAt operationPosition
  have hpos : 10 * layer + opcode - 1 ≠ 0 := by omega
  have hbody : 10 * layer + opcode - 1 < 10 * layers + 1 := by omega
  simp only [hpos, ↓reduceIte, hbody]
  have hquot : (10 * layer + opcode - 1 - 1) / 10 = layer := by omega
  have hrem : (10 * layer + opcode - 1 - 1) % 10 + 2 = opcode := by omega
  rw [hquot, hrem]

theorem supported_layer_instruction (layers layer opcode : Nat)
    (hn : layers ≤ maxLayers) (hl : layer < layers)
    (ho : 2 ≤ opcode) (ho' : opcode ≤ 11) :
    (makePlan layers)[operationPosition layer opcode]? = some (encode layer opcode) := by
  have hp : operationPosition layer opcode < 10 * layers + 3 := by
    unfold operationPosition
    omega
  rw [supported_instruction_at layers _ hn hp,
    indexed_layer_instruction layers layer opcode hl ho ho']

/-- Encoding is within UInt32, with unambiguous byte-sized opcode and layer. -/
theorem instruction_bounds (layer opcode : Nat)
    (hl : layer < maxLayers) (ho : 2 ≤ opcode) (ho' : opcode ≤ 11) :
    encode layer opcode < 2^32 ∧
    encode layer opcode / 256 = layer ∧
    encode layer opcode % 256 = opcode := by
  unfold encode maxLayers at *
  omega

def allInstructionsBounded (layers : Nat) : Prop :=
  ∀ i ∈ makePlan layers, i < 2^32 ∧ i % 256 ≥ 1 ∧ i % 256 ≤ 13 ∧ i / 256 < maxLayers

instance (layers : Nat) : Decidable (allInstructionsBounded layers) :=
  inferInstanceAs (Decidable (∀ i ∈ makePlan layers,
    i < 2^32 ∧ i % 256 ≥ 1 ∧ i % 256 ≤ 13 ∧ i / 256 < maxLayers))

theorem instruction_bounds_certificate :
    ∀ layers ∈ List.range 50, allInstructionsBounded layers := by decide

theorem supported_instruction_bounds (layers : Nat) (h : layers ≤ maxLayers) :
    allInstructionsBounded layers := by
  apply instruction_bounds_certificate layers
  simp only [List.mem_range]
  simp only [maxLayers] at h
  omega

/-- Row worker w visits w, w+workers, ... . This identifies a unique worker
and iteration for every row; callers must supply a nonzero worker count. -/
theorem row_worker_coverage (row workers : Nat) (hw : 0 < workers) :
    row % workers < workers ∧
    row % workers + (row / workers) * workers = row := by
  constructor
  · exact Nat.mod_lt row hw
  · simpa [Nat.mul_comm] using Nat.mod_add_div row workers

theorem row_worker_unique (workers w₁ w₂ k₁ k₂ : Nat)
    (hw : 0 < workers) (h₁ : w₁ < workers) (h₂ : w₂ < workers)
    (heq : w₁ + k₁ * workers = w₂ + k₂ * workers) :
    w₁ = w₂ ∧ k₁ = k₂ := by
  have hdiv₁ : (w₁ + k₁ * workers) / workers = k₁ := by
    simp [Nat.add_mul_div_right _ _ hw, Nat.div_eq_of_lt h₁]
  have hdiv₂ : (w₂ + k₂ * workers) / workers = k₂ := by
    simp [Nat.add_mul_div_right _ _ hw, Nat.div_eq_of_lt h₂]
  have hk : k₁ = k₂ := by rw [heq] at hdiv₁; omega
  constructor
  · rw [hk] at heq
    exact Nat.add_right_cancel heq
  · exact hk

/-- Abstract single-arrival transition. The premise arrivals < workers is
supplied by one arrival per participating worker in the current phase.
It does not follow from a GPU scheduler or the integer representation. -/
structure BarrierState where
  phase : Nat
  arrivals : Nat
  deriving DecidableEq, Repr

def arrive (workers : Nat) (s : BarrierState) : BarrierState :=
  if s.arrivals + 1 = workers then ⟨s.phase + 1, 0⟩
  else ⟨s.phase, s.arrivals + 1⟩

theorem barrier_counter_advance_iff_last (workers : Nat) (s : BarrierState)
    (_arrivalBound : s.arrivals < workers) :
    (arrive workers s).phase = s.phase + 1 ↔ s.arrivals + 1 = workers := by
  unfold arrive
  split <;> simp_all

/-- Distinct worker identities seen in this phase, and a fresh valid arrival. -/
def OneArrivalPerWorker (workers : Nat) (seen : List Nat) (next : Nat) : Prop :=
  seen.Nodup ∧ (∀ worker ∈ seen, worker < workers) ∧
  next < workers ∧ next ∉ seen

theorem one_arrival_per_worker_bound (workers : Nat) (seen : List Nat) (next : Nat)
    (h : OneArrivalPerWorker workers seen next) : seen.length < workers := by
  have hn : (next :: seen).Nodup := List.nodup_cons.mpr ⟨h.2.2.2, h.1⟩
  have hs : next :: seen ⊆ List.range workers := by
    intro worker hw
    apply List.mem_range.mpr
    simp only [List.mem_cons] at hw
    rcases hw with heq | hmem
    · simpa [heq] using h.2.2.1
    · exact h.2.1 worker hmem
  have hlen := hn.length_le_of_subset hs
  simp only [List.length_cons, List.length_range] at hlen
  omega

/-- No phase advance except on the last arrival, under an explicit worker-ID
uniqueness premise and the relation between seen identities and the counter. -/
theorem barrier_phase_advance_iff_last (workers : Nat) (s : BarrierState)
    (seen : List Nat) (next : Nat)
    (h : OneArrivalPerWorker workers seen next) (hc : s.arrivals = seen.length) :
    (arrive workers s).phase = s.phase + 1 ↔ s.arrivals + 1 = workers := by
  apply barrier_counter_advance_iff_last
  rw [hc]
  exact one_arrival_per_worker_bound workers seen next h

theorem barrier_nonlast_preserves_phase (workers : Nat) (s : BarrierState)
    (h : s.arrivals + 1 < workers) :
    (arrive workers s).phase = s.phase ∧
    (arrive workers s).arrivals = s.arrivals + 1 := by
  simp [arrive, show s.arrivals + 1 ≠ workers by omega]

theorem barrier_last_resets (workers : Nat) (s : BarrierState)
    (h : s.arrivals + 1 = workers) :
    arrive workers s = ⟨s.phase + 1, 0⟩ := by simp [arrive, h]

theorem binding_limits :
    threadgroupSize = 256 ∧ maxWorkers = 32 ∧
    spinLimit = 10000000 ∧ maxLayers = 49 := by decide

/-! Register-cached routing has one slot per lane-strided expert. -/
def routerLanes : Nat := 32
def routerSlots : Nat := 8
def cachedRouterExperts : Nat := routerLanes * routerSlots

theorem router_register_ownership (expert : Nat) (h : expert < cachedRouterExperts) :
    expert % routerLanes < routerLanes ∧ expert / routerLanes < routerSlots ∧
    expert % routerLanes + expert / routerLanes * routerLanes = expert := by
  simp only [routerLanes, routerSlots, cachedRouterExperts] at *
  omega

theorem router_register_bounds (lane slot : Nat)
    (hl : lane < routerLanes) (hs : slot < routerSlots) :
    lane + slot * routerLanes < cachedRouterExperts := by
  simp only [routerLanes, routerSlots, cachedRouterExperts] at *
  omega

/-! Dense split reduction: integer column ownership only, not floating-point
reduction arithmetic, barrier memory ordering, or performance. -/
def denseSimds : Nat := 8
def denseMaxCols : Nat := 8192

/-- Each of eight SIMDs contributes 32 lanes, advancing by one threadgroup. -/
def denseColumn (simd lane iteration : Nat) : Nat :=
  simd * routerLanes + lane + iteration * threadgroupSize

theorem dense_geometry :
    denseSimds = 8 ∧ routerLanes = 32 ∧ threadgroupSize = 256 ∧
    denseSimds * routerLanes = threadgroupSize ∧ denseMaxCols = 8192 := by decide

/-- Constructive coverage of every guarded input column, including partial
threadgroups and the empty-input case (where the premise is impossible). -/
theorem dense_column_coverage (col n : Nat)
    (hn : n ≤ denseMaxCols) (hc : col < n) :
    (col % threadgroupSize) / routerLanes < denseSimds ∧
    col % routerLanes < routerLanes ∧
    col / threadgroupSize < denseMaxCols / threadgroupSize ∧
    denseColumn ((col % threadgroupSize) / routerLanes)
      (col % routerLanes) (col / threadgroupSize) = col := by
  simp only [denseSimds, denseMaxCols, routerLanes, threadgroupSize, denseColumn] at *
  omega

/-- No two valid lane/SIMD/iteration coordinates own the same column. -/
theorem dense_column_unique (s₁ l₁ k₁ s₂ l₂ k₂ : Nat)
    (hs₁ : s₁ < denseSimds) (hl₁ : l₁ < routerLanes)
    (hs₂ : s₂ < denseSimds) (hl₂ : l₂ < routerLanes)
    (heq : denseColumn s₁ l₁ k₁ = denseColumn s₂ l₂ k₂) :
    s₁ = s₂ ∧ l₁ = l₂ ∧ k₁ = k₂ := by
  simp only [denseColumn, denseSimds, routerLanes, threadgroupSize] at *
  omega

/-- The loop guard certifies a valid input index and a representable UInt32
increment even on the final iteration; the next column need not be in bounds. -/
theorem dense_guarded_bounds (simd lane iteration n : Nat)
    (hs : simd < denseSimds) (hl : lane < routerLanes)
    (hn : n ≤ denseMaxCols) (guard : denseColumn simd lane iteration < n) :
    simd * routerLanes + lane < threadgroupSize ∧
    iteration < denseMaxCols / threadgroupSize ∧
    denseColumn simd lane iteration < denseMaxCols ∧
    denseColumn simd lane iteration < 2^32 ∧
    denseColumn simd lane iteration + threadgroupSize < 2^32 := by
  simp only [denseColumn, denseSimds, denseMaxCols, routerLanes, threadgroupSize] at *
  omega

/-- Natural-number successor matches the shader's non-wrapping col += 256. -/
theorem dense_column_advance (simd lane iteration : Nat) :
    denseColumn simd lane (iteration + 1) =
      denseColumn simd lane iteration + threadgroupSize := by
  simp [denseColumn, Nat.add_mul, Nat.add_assoc]

/-! Split attention integer contracts. Coordinates are relative to the sliding
window start, not the absolute cache position. These statements do not certify
online-softmax arithmetic, memory ordering, device execution, or barriers. -/
def attentionChunks : List Nat := [128, 256, 512, 1024]
def validAttentionChunk (c : Nat) : Bool := attentionChunks.contains c
def attentionMaxContext : Nat := 262144
def attentionMaxHeads : Nat := 64
def attentionMaxDim : Nat := 256
def attentionSimds : Nat := 8
def attentionLanes : Nat := 32
def attentionChunkCount (n c : Nat) : Nat := (n + c - 1) / c
def attentionToken (c chunk simd iteration : Nat) : Nat :=
  c * chunk + simd + attentionSimds * iteration

theorem attention_chunk_admissible (c : Nat) :
    validAttentionChunk c = true ↔ c = 128 ∨ c = 256 ∨ c = 512 ∨ c = 1024 := by
  simp [validAttentionChunk, attentionChunks]

theorem attention_geometry :
    attentionSimds = 8 ∧ attentionLanes = 32 ∧
    attentionSimds * attentionLanes = threadgroupSize ∧
    attentionMaxContext = 262144 ∧ attentionMaxHeads = 64 ∧ attentionMaxDim = 256 := by decide

theorem attention_chunk_bounds (c : Nat) (hc : validAttentionChunk c = true) :
    128 ≤ c ∧ c ≤ 1024 ∧ c % attentionSimds = 0 := by
  rcases (attention_chunk_admissible c).mp hc with rfl | rfl | rfl | rfl <;> decide

/-- Canonical decomposition of every actual token, including a partial final
chunk. The lane coordinate is deliberately absent: a SIMD cooperates on a token. -/
theorem attention_token_coverage (r n c : Nat)
    (hc : validAttentionChunk c = true) (hr : r < n) :
    r / c < attentionChunkCount n c ∧ (r % c) % attentionSimds < attentionSimds ∧
    (r % c) / attentionSimds < c / attentionSimds ∧
    attentionToken c (r / c) ((r % c) % attentionSimds)
      ((r % c) / attentionSimds) = r := by
  rcases (attention_chunk_admissible c).mp hc with rfl | rfl | rfl | rfl <;>
    simp only [attentionChunkCount, attentionToken, attentionSimds] <;> omega

/-- Uniqueness requires a coordinate inside its chunk; a loop guard alone
would not prevent an iteration spilling into the next chunk. -/
theorem attention_token_unique (c a s i b t j : Nat)
    (hc : validAttentionChunk c = true)
    (hs : s < attentionSimds) (ht : t < attentionSimds)
    (hi : i < c / attentionSimds) (hj : j < c / attentionSimds)
    (he : attentionToken c a s i = attentionToken c b t j) :
    a = b ∧ s = t ∧ i = j := by
  rcases (attention_chunk_admissible c).mp hc with rfl | rfl | rfl | rfl <;>
    simp only [attentionToken, attentionSimds] at * <;> omega

theorem attention_chunk_count_bounds (n c : Nat)
    (hn : n ≤ attentionMaxContext) (hc : validAttentionChunk c = true) :
    attentionChunkCount n c ≤ 2048 ∧
    (attentionChunkCount n c = 0 ↔ n = 0) ∧ n + c - 1 < 2^32 := by
  rcases (attention_chunk_admissible c).mp hc with rfl | rfl | rfl | rfl <;>
    simp only [attentionChunkCount, attentionMaxContext] at * <;> omega

/-- Exactly the nonempty chunks are launched; n=0 launches none. -/
theorem attention_active_chunk (n c chunk : Nat)
    (hc : validAttentionChunk c = true) :
    chunk < attentionChunkCount n c ↔ c * chunk < n := by
  rcases (attention_chunk_admissible c).mp hc with rfl | rfl | rfl | rfl <;>
    simp only [attentionChunkCount] <;> omega

/-- The natural-number loop recurrence matches a nonwrapping += 8. -/
theorem attention_token_advance (c chunk simd iteration : Nat) :
    attentionToken c chunk simd (iteration + 1) =
      attentionToken c chunk simd iteration + attentionSimds := by
  simp [attentionToken, Nat.mul_add, Nat.add_assoc]

/-- Both guards are required: the per-chunk end and the overall active
window. With start+n bounded, their absolute cache read is below max_context. -/
theorem attention_guarded_read_bounds (start n c chunk simd iteration : Nat)
    (hc : validAttentionChunk c = true) (hn : start + n ≤ attentionMaxContext)
    (hs : simd < attentionSimds)
    (localGuard : simd + attentionSimds * iteration < c)
    (readGuard : attentionToken c chunk simd iteration < n) :
    chunk < attentionChunkCount n c ∧ iteration < c / attentionSimds ∧
    start + attentionToken c chunk simd iteration < attentionMaxContext ∧
    start + attentionToken c chunk simd iteration < 2^32 := by
  rcases (attention_chunk_admissible c).mp hc with rfl | rfl | rfl | rfl <;>
    simp only [attentionToken, attentionChunkCount, attentionSimds, attentionMaxContext] at * <;> omega

/-- The partial-chunk mask is exactly the remaining relative token count. -/
theorem attention_partial_chunk_guard (n c chunk offset : Nat)
    (hb : c * chunk < n) :
    c * chunk + offset < n ↔ offset < n - c * chunk := by omega

/-- Includes begin+width and the final SIMD loop increment, even if they
are outside the active token window. Neither operation wraps a UInt32. -/
theorem attention_loop_u32_bounds (start n c chunk simd iteration : Nat)
    (hc : validAttentionChunk c = true) (hn : start + n ≤ attentionMaxContext)
    (hchunk : chunk < attentionChunkCount n c)
    (hs : simd < attentionSimds) (hi : iteration < c / attentionSimds) :
    start + c * chunk + c ≤ attentionMaxContext + 1024 ∧
    start + attentionToken c chunk simd iteration < attentionMaxContext + 1024 ∧
    start + attentionToken c chunk simd iteration + attentionSimds < 2^32 := by
  rcases (attention_chunk_admissible c).mp hc with rfl | rfl | rfl | rfl <;>
    simp only [attentionChunkCount, attentionToken, attentionSimds, attentionMaxContext] at * <;> omega

/-- One distinguished producer pair includes the sink exactly once. It
exists only for a nonempty window (there is no producer when n=0). -/
def attentionSinkOwner (chunk simd : Nat) : Prop := chunk = 0 ∧ simd = 0

theorem attention_unique_sink (n c : Nat) (hn : 0 < n)
    (hc : validAttentionChunk c = true) :
    0 < attentionChunkCount n c ∧ 0 < attentionSimds ∧
    ∀ chunk simd, attentionSinkOwner chunk simd ↔ chunk = 0 ∧ simd = 0 := by
  have hp : 0 < attentionChunkCount n c := by
    rcases (attention_chunk_admissible c).mp hc with rfl | rfl | rfl | rfl <;>
      simp only [attentionChunkCount] <;> omega
  exact ⟨hp, by decide, fun _ _ => Iff.rfl⟩

/-- Head-major flattened state index. Storage is one (dim+2)-float state
per head/chunk pair, including the two scalar softmax statistics. -/
def attentionStateSlot (chunks head chunk : Nat) : Nat := head * chunks + chunk

theorem attention_state_flatten (heads chunks head chunk : Nat)
    (hh : head < heads) (hc : chunk < chunks) :
    attentionStateSlot chunks head chunk < heads * chunks ∧
    attentionStateSlot chunks head chunk / chunks = head ∧
    attentionStateSlot chunks head chunk % chunks = chunk := by
  have hp : 0 < chunks := by omega
  have hm := Nat.mul_le_mul_right chunks (show head + 1 ≤ heads by omega)
  have hd : (head * chunks + chunk) / chunks = head := by
    rw [Nat.add_comm]
    simp [Nat.add_mul_div_right _ _ hp, Nat.div_eq_of_lt hc]
  have hr : (head * chunks + chunk) % chunks = chunk := by
    simp [Nat.add_mod, Nat.mod_eq_of_lt hc]
  simp only [attentionStateSlot]
  constructor
  · simp only [Nat.add_mul, Nat.one_mul] at hm
    omega
  · exact ⟨hd, hr⟩

theorem attention_state_unique (chunks h₁ c₁ h₂ c₂ : Nat)
    (hc₁ : c₁ < chunks) (hc₂ : c₂ < chunks)
    (he : attentionStateSlot chunks h₁ c₁ = attentionStateSlot chunks h₂ c₂) :
    h₁ = h₂ ∧ c₁ = c₂ := by
  have ha := attention_state_flatten (h₁ + 1) chunks h₁ c₁ (by omega) hc₁
  have hb := attention_state_flatten (h₂ + 1) chunks h₂ c₂ (by omega) hc₂
  rw [he] at ha
  omega

theorem attention_state_u32_bounds (heads chunks dim head chunk field : Nat)
    (hh : heads ≤ attentionMaxHeads) (hc : chunks ≤ 2048) (hd : dim ≤ attentionMaxDim)
    (hhead : head < heads) (hchunk : chunk < chunks) (hf : field < dim + 2) :
    heads * chunks * (dim + 2) ≤ 33816576 ∧
    attentionStateSlot chunks head chunk * (dim + 2) + field <
      heads * chunks * (dim + 2) ∧
    attentionStateSlot chunks head chunk * (dim + 2) + field < 2^32 := by
  have hb := (attention_state_flatten heads chunks head chunk hhead hchunk).1
  have hm := Nat.mul_le_mul (show heads ≤ 64 from hh) hc
  have ht := Nat.mul_le_mul hm (show dim + 2 ≤ 258 by simpa [attentionMaxDim] using Nat.add_le_add_right hd 2)
  have hi := Nat.mul_le_mul_right (dim + 2) (show attentionStateSlot chunks head chunk + 1 ≤ heads * chunks by omega)
  simp only [Nat.add_mul, Nat.one_mul] at hi
  constructor
  · exact ht
  · constructor <;> omega

/-- A dim+2 stride keeps fields of distinct states disjoint. -/
theorem attention_state_field_unique (dim slot₁ field₁ slot₂ field₂ : Nat)
    (hf₁ : field₁ < dim + 2) (hf₂ : field₂ < dim + 2)
    (he : slot₁ * (dim + 2) + field₁ = slot₂ * (dim + 2) + field₂) :
    slot₁ = slot₂ ∧ field₁ = field₂ := by
  have hp : 0 < dim + 2 := by omega
  have hd₁ : (slot₁ * (dim + 2) + field₁) / (dim + 2) = slot₁ := by
    rw [Nat.add_comm]
    simp [Nat.add_mul_div_right _ _ hp, Nat.div_eq_of_lt hf₁]
  have hd₂ : (slot₂ * (dim + 2) + field₂) / (dim + 2) = slot₂ := by
    rw [Nat.add_comm]
    simp [Nat.add_mul_div_right _ _ hp, Nat.div_eq_of_lt hf₂]
  rw [he] at hd₁
  have hs : slot₁ = slot₂ := by omega
  subst slot₂
  exact ⟨rfl, by omega⟩

/-- Parallel QK scores use one head-major float per actual relative token.
This is an index/ownership bound, not an assertion about FP32 dot products. -/
theorem attention_score_u32_bounds (heads length head relative : Nat)
    (hh : heads ≤ attentionMaxHeads) (hl : length ≤ attentionMaxContext)
    (hhead : head < heads) (hr : relative < length) :
    heads * length ≤ 16777216 ∧ head * length + relative < heads * length ∧
    head * length + relative < 2^32 := by
  have hb := (attention_state_flatten heads length head relative hhead hr).1
  have hm := Nat.mul_le_mul (show heads ≤ 64 from hh) (show length ≤ 262144 from hl)
  simp only [attentionStateSlot] at hb
  constructor
  · exact hm
  · exact ⟨hb, by omega⟩

/-- Row-major cache element offset, not bytes or allocation size. With
pos<262144, heads<=64, dim<=256 the maximum allocation has exactly 2^32
float elements; every guarded element index is strictly below 2^32.
A byte offset or a combined K+V allocation requires wider arithmetic. -/
theorem attention_cache_u32_index (pos kvheads dim head lane : Nat)
    (hp : pos < attentionMaxContext) (hk : kvheads ≤ attentionMaxHeads)
    (hd : dim ≤ attentionMaxDim) (hh : head < kvheads) (hl : lane < dim) :
    (pos * kvheads + head) * dim + lane < 2^32 := by
  have hrow : pos * kvheads + head < attentionMaxContext * kvheads := by
    have hm := Nat.mul_le_mul_right kvheads (show pos + 1 ≤ attentionMaxContext by omega)
    simp only [Nat.add_mul, Nat.one_mul] at hm
    omega
  have hm := Nat.mul_le_mul_right dim (show pos * kvheads + head + 1 ≤ attentionMaxContext * kvheads by omega)
  have hcap := Nat.mul_le_mul (Nat.mul_le_mul_left attentionMaxContext hk) hd
  simp only [Nat.add_mul, Nat.one_mul] at hm
  simp only [Nat.add_mul, attentionMaxContext, attentionMaxHeads, attentionMaxDim] at *
  omega

/-! Tiled row ownership. A SIMD owns R adjacent rows per iteration. -/
def rowTiles : List Nat := [1, 2, 4, 8]

def validRowTile (tile : Nat) : Bool := rowTiles.contains tile

theorem row_tile_admissible (tile : Nat) :
    validRowTile tile = true ↔ tile = 1 ∨ tile = 2 ∨ tile = 4 ∨ tile = 8 := by
  simp [validRowTile, rowTiles]

theorem row_tile_bounds (tile : Nat) (h : validRowTile tile = true) :
    0 < tile ∧ tile ≤ 8 := by
  have := (row_tile_admissible tile).mp h
  omega

/-- s is a global SIMD index, k an iteration, j a row within the tile. -/
def tiledRow (simds tile s k j : Nat) : Nat :=
  s * tile + k * (simds * tile) + j

theorem tiled_row_factor (simds tile s k j : Nat) :
    tiledRow simds tile s k j = (s + k * simds) * tile + j := by
  simp [tiledRow, Nat.add_mul, Nat.mul_assoc]

/-- Constructive coverage, including the unique proposed owner coordinates. -/
theorem tiled_row_coverage (row simds tile : Nat)
    (hs : 0 < simds) (ht : 0 < tile) :
    (row / tile) % simds < simds ∧ row % tile < tile ∧
    tiledRow simds tile ((row / tile) % simds)
      ((row / tile) / simds) (row % tile) = row := by
  obtain ⟨hs', heq⟩ := row_worker_coverage (row / tile) simds hs
  refine ⟨hs', Nat.mod_lt row ht, ?_⟩
  rw [tiled_row_factor, heq]
  simpa [Nat.add_comm, Nat.mul_comm] using Nat.mod_add_div row tile

/-- Every valid output row's owner passes the outer tile-start test as well
as the inner row guard, so partial-tile masking cannot remove that row. -/
theorem guarded_tiled_coverage (row rows simds tile : Nat)
    (hr : row < rows) (hs : 0 < simds) (ht : 0 < tile) :
    let s := (row / tile) % simds
    let k := (row / tile) / simds
    let j := row % tile
    s < simds ∧ j < tile ∧ tiledRow simds tile s k 0 < rows ∧
      tiledRow simds tile s k j = row := by
  obtain ⟨hs', hj', heq⟩ := tiled_row_coverage row simds tile hs ht
  refine ⟨hs', hj', ?_, heq⟩
  unfold tiledRow at *
  omega

theorem tiled_row_unique (simds tile s₁ s₂ k₁ k₂ j₁ j₂ : Nat)
    (hs : 0 < simds) (ht : 0 < tile)
    (hs₁ : s₁ < simds) (hs₂ : s₂ < simds)
    (hj₁ : j₁ < tile) (hj₂ : j₂ < tile)
    (heq : tiledRow simds tile s₁ k₁ j₁ = tiledRow simds tile s₂ k₂ j₂) :
    s₁ = s₂ ∧ k₁ = k₂ ∧ j₁ = j₂ := by
  rw [tiled_row_factor, tiled_row_factor] at heq
  have hd₁ : ((s₁ + k₁ * simds) * tile + j₁) / tile = s₁ + k₁ * simds := by
    rw [Nat.add_comm]
    simp [Nat.add_mul_div_right _ _ ht, Nat.div_eq_of_lt hj₁]
  have hd₂ : ((s₂ + k₂ * simds) * tile + j₂) / tile = s₂ + k₂ * simds := by
    rw [Nat.add_comm]
    simp [Nat.add_mul_div_right _ _ ht, Nat.div_eq_of_lt hj₂]
  have hq : s₁ + k₁ * simds = s₂ + k₂ * simds := by
    rw [heq] at hd₁
    omega
  obtain ⟨hsame, hksame⟩ := row_worker_unique simds s₁ s₂ k₁ k₂ hs hs₁ hs₂ hq
  refine ⟨hsame, hksame, ?_⟩
  rw [hq] at heq
  omega

/-- For a tested tile start, this is exactly the final-partial-tile mask. -/
theorem tiled_row_guard (rows simds tile s k j : Nat)
    (hstart : tiledRow simds tile s k 0 < rows) :
    tiledRow simds tile s k j < rows ↔
      j < rows - tiledRow simds tile s k 0 := by
  unfold tiledRow at *
  omega

/-- The partial-tile mask gives an in-bounds, representable output row. -/
theorem guarded_tiled_row_bounds (rows simds tile s k j : Nat)
    (hr : rows ≤ 1048576) (hstart : tiledRow simds tile s k 0 < rows)
    (guard : j < rows - tiledRow simds tile s k 0) :
    tiledRow simds tile s k j < rows ∧ tiledRow simds tile s k j < 2^32 := by
  have hb := (tiled_row_guard rows simds tile s k j hstart).mpr guard
  omega

/-- Tiling only changes ownership of opaque row computations; it does not
rewrite a row's arithmetic or make any floating-point equivalence claim. -/
theorem tiled_row_result {α : Type} (compute : Nat → α) (row simds tile : Nat)
    (hs : 0 < simds) (ht : 0 < tile) :
    compute (tiledRow simds tile ((row / tile) % simds)
      ((row / tile) / simds) (row % tile)) = compute row := by
  rw [(tiled_row_coverage row simds tile hs ht).2.2]

/-- A staged grid launches ceil(rows/tile) SIMDs, with exactly one owner
for every guarded row, independently of simultaneous residency. -/
theorem staged_row_coverage (row rows tile : Nat) (hr : row < rows)
    (ht : validRowTile tile = true) :
    row / tile < (rows + tile - 1) / tile ∧ row % tile < tile ∧
      row / tile * tile + row % tile = row := by
  rcases (row_tile_admissible tile).mp ht with rfl | rfl | rfl | rfl <;> omega

/-- Even a partial final tile has nonwrapping candidate indices; the shader
must still guard candidate < rows before every output access. -/
theorem staged_row_u32_bounds (rows tile simd j : Nat)
    (hr : rows ≤ 1048576) (ht : validRowTile tile = true)
    (hs : simd < (rows + tile - 1) / tile) (hj : j < tile) :
    simd * tile < rows ∧ simd * tile + j < 2^32 := by
  rcases (row_tile_admissible tile).mp ht with rfl | rfl | rfl | rfl <;> omega

/-! Independent Q/K/V grids concatenate tile intervals, not output rows.
The expression tree is evaluated here and rendered by Main, so the generated
branch conditions and subtraction are the definitions used in these proofs. -/
inductive QkvExpr where
  | tile | qtiles | ktiles
  | literal (n : Nat)
  | add (a b : QkvExpr)
  | sub (a b : QkvExpr)
  | ifLt (a b yes no : QkvExpr)

def QkvExpr.eval (tile qtiles ktiles : Nat) : QkvExpr → Nat
  | .tile => tile
  | .qtiles => qtiles
  | .ktiles => ktiles
  | .literal n => n
  | .add a b => a.eval tile qtiles ktiles + b.eval tile qtiles ktiles
  | .sub a b => a.eval tile qtiles ktiles - b.eval tile qtiles ktiles
  | .ifLt a b yes no =>
      if a.eval tile qtiles ktiles < b.eval tile qtiles ktiles then
        yes.eval tile qtiles ktiles else no.eval tile qtiles ktiles

def qkvProjectionExpr : QkvExpr :=
  .ifLt .tile .qtiles (.literal 0)
    (.ifLt .tile (.add .qtiles .ktiles) (.literal 1) (.literal 2))

def qkvOffsetExpr : QkvExpr :=
  .ifLt .tile .qtiles (.literal 0)
    (.ifLt .tile (.add .qtiles .ktiles) .qtiles (.add .qtiles .ktiles))

def qkvLocalTileExpr : QkvExpr := .sub .tile qkvOffsetExpr

def qkvProjection (tile qtiles ktiles : Nat) : Nat :=
  qkvProjectionExpr.eval tile qtiles ktiles

def qkvLocalTile (tile qtiles ktiles : Nat) : Nat :=
  qkvLocalTileExpr.eval tile qtiles ktiles

def qkvTileCount (projection qtiles ktiles vtiles : Nat) : Nat :=
  if projection = 0 then qtiles else if projection = 1 then ktiles else vtiles

def qkvGlobalTile (projection localTile qtiles ktiles : Nat) : Nat :=
  (if projection = 0 then 0 else if projection = 1 then qtiles else qtiles + ktiles) +
    localTile

/-- Exact half-open interval ownership, including empty projections. -/
theorem qkv_interval_ownership (tile q k v : Nat) (h : tile < q + k + v) :
    (qkvProjection tile q k = 0 ↔ tile < q) ∧
    (qkvProjection tile q k = 1 ↔ q ≤ tile ∧ tile < q + k) ∧
    (qkvProjection tile q k = 2 ↔ q + k ≤ tile ∧ tile < q + k + v) := by
  simp only [qkvProjection, qkvProjectionExpr, QkvExpr.eval]
  by_cases hq : tile < q <;> by_cases hk : tile < q + k <;>
    simp only [hq, hk, ite_true, ite_false] <;> simp_all <;> omega

/-- Decode any launched tile to a valid local tile and reconstruct it exactly. -/
theorem qkv_local_bounds (tile q k v : Nat) (h : tile < q + k + v) :
    qkvProjection tile q k < 3 ∧
    qkvLocalTile tile q k < qkvTileCount (qkvProjection tile q k) q k v ∧
    qkvGlobalTile (qkvProjection tile q k) (qkvLocalTile tile q k) q k = tile := by
  simp only [qkvProjection, qkvProjectionExpr, qkvLocalTile, qkvLocalTileExpr,
    qkvOffsetExpr, QkvExpr.eval, qkvTileCount, qkvGlobalTile]
  by_cases hq : tile < q <;> by_cases hk : tile < q + k <;>
    simp only [hq, hk, ite_true, ite_false] <;> simp_all <;> omega

/-- Conversely every local tile has a unique global encoding that decodes back. -/
theorem qkv_inverse_coverage (projection localTile q k v : Nat)
    (hp : projection < 3) (hl : localTile < qkvTileCount projection q k v) :
    qkvGlobalTile projection localTile q k < q + k + v ∧
    qkvProjection (qkvGlobalTile projection localTile q k) q k = projection ∧
    qkvLocalTile (qkvGlobalTile projection localTile q k) q k = localTile := by
  have cases : projection = 0 ∨ projection = 1 ∨ projection = 2 := by omega
  by_cases hq : qkvGlobalTile projection localTile q k < q <;>
    by_cases hk : qkvGlobalTile projection localTile q k < q + k <;>
    simp only [qkvProjection, qkvProjectionExpr, qkvLocalTile, qkvLocalTileExpr,
      qkvOffsetExpr, QkvExpr.eval, hq, hk, ite_true, ite_false] <;>
    rcases cases with rfl | rfl | rfl <;>
    simp_all [qkvTileCount, qkvGlobalTile] <;> omega

/-- Bounded totals make both prefix additions nonwrapping, and the selected
subtraction cannot underflow. A tile outside the launched total is not covered. -/
theorem qkv_u32_bounds (tile q k v : Nat)
    (ht : tile < q + k + v) (hb : q + k + v < 2^32) :
    q < 2^32 ∧ k < 2^32 ∧ v < 2^32 ∧ q + k < 2^32 ∧ tile < 2^32 ∧
    qkvOffsetExpr.eval tile q k ≤ tile ∧
    qkvLocalTile tile q k < 2^32 ∧
    (q + k) % 2^32 = q + k ∧ (q + k + v) % 2^32 = q + k + v := by
  have hprefix : q + k < 2^32 := by omega
  have hm₁ := Nat.mod_eq_of_lt hprefix
  have hm₂ := Nat.mod_eq_of_lt hb
  simp only [qkvLocalTile, qkvLocalTileExpr, qkvOffsetExpr, QkvExpr.eval]
  by_cases hq : tile < q <;> by_cases hk : tile < q + k <;>
    simp only [hq, hk, ite_true, ite_false] <;> simp_all <;> omega

/-- Reuse the existing staged-row certificate independently for each projection.
Ceiling is taken before concatenation, so partial Q/K/V tails cannot be merged.
The decoded local tile/row is exactly the original opaque row computation. -/
theorem qkv_staged_row_coverage (projection row qrows krows vrows width : Nat)
    (hp : projection < 3)
    (hr : row < qkvTileCount projection qrows krows vrows)
    (hw : validRowTile width = true) :
    let q := (qrows + width - 1) / width
    let k := (krows + width - 1) / width
    let v := (vrows + width - 1) / width
    let globalTile := qkvGlobalTile projection (row / width) q k
    globalTile < q + k + v ∧ qkvProjection globalTile q k = projection ∧
      qkvLocalTile globalTile q k * width + row % width = row ∧ row % width < width := by
  dsimp only
  obtain ⟨ht, hj, heq⟩ := staged_row_coverage row
    (qkvTileCount projection qrows krows vrows) width hr hw
  have hc : row / width < qkvTileCount projection
      ((qrows + width - 1) / width) ((krows + width - 1) / width)
      ((vrows + width - 1) / width) := by
    have cases : projection = 0 ∨ projection = 1 ∨ projection = 2 := by omega
    rcases cases with rfl | rfl | rfl <;> simpa [qkvTileCount] using ht
  obtain ⟨hg, hp', hl⟩ := qkv_inverse_coverage projection (row / width)
    ((qrows + width - 1) / width) ((krows + width - 1) / width)
    ((vrows + width - 1) / width) hp hc
  exact ⟨hg, hp', by rw [hl]; exact heq, hj⟩

/-- Native launch contract: 256 threads / 32 lanes = 8 SIMDs per group. -/
theorem launch_simd_bounds (workers : Nat) (hmin : 1 ≤ workers)
    (hmax : workers ≤ maxWorkers) :
    0 < workers * 8 ∧ workers * 8 ≤ 256 := by
  unfold maxWorkers at hmax
  omega

/-- Explicit bounded arithmetic certificate for the tested native loop.
The current tile start is tested before its body. Even the next start after
its last successful test and every guarded row candidate fit in UInt32.
No claim is made for an unbounded iteration counter or an unchecked launch. -/
theorem tiled_u32_bounds (rows simds tile s start j : Nat)
    (hr : rows ≤ 1048576) (hs : simds ≤ 256)
    (ht : validRowTile tile = true) (howner : s < simds)
    (hstart : start < rows) (hj : j < tile) :
    simds * tile ≤ 2048 ∧ s * tile < 2^32 ∧
    start + simds * tile ≤ rows + simds * tile ∧
    start + simds * tile < 2^32 ∧ start + j < 2^32 := by
  have htile := row_tile_bounds tile ht
  have hstride : simds * tile ≤ 256 * 8 := Nat.mul_le_mul hs htile.2
  have hinit : s * tile ≤ 256 * 8 := Nat.mul_le_mul (by omega) htile.2
  omega

/-- In a tested iteration, the iteration*stride intermediate itself is
bounded by the tile start, so the factored native expression cannot wrap. -/
theorem tiled_iteration_u32 (rows simds tile s k : Nat)
    (hr : rows ≤ 1048576) (hstart : tiledRow simds tile s k 0 < rows) :
    k * (simds * tile) < 2^32 ∧
    s * tile + k * (simds * tile) < 2^32 := by
  unfold tiledRow at hstart
  omega

/-! Ideal packed INT4 indexing. These are natural-number arithmetic facts;
actual buffer allocations and the shader's shifts/masks remain obligations. -/
def int4Code (word lane : Nat) : Nat := (word / 16^lane) % 16

theorem int4_code_bounds (word lane : Nat) : int4Code word lane < 16 := by
  exact Nat.mod_lt _ (by decide)

theorem int4_low_nibble_first (word : Nat) : int4Code word 0 = word % 16 := by
  simp [int4Code]

/-- A 64-weight quantization group contains eight 8-weight packed words. -/
theorem int4_chunk_indices (column : Nat) :
    column % 64 / 8 < 8 ∧ column % 8 < 8 ∧
    column / 64 * 8 + column % 64 / 8 = column / 8 ∧
    (column / 64 * 8 + column % 64 / 8) * 8 + column % 8 = column := by
  omega

theorem int4_group_word_bounds (column columns : Nat)
    (hc : column < columns) (haligned : columns % 64 = 0) :
    column / 64 < columns / 64 ∧ column / 8 < columns / 8 := by
  omega

/-- The two-word lane layout covers sixteen adjacent columns. Each lane
segment stays in one affine group and maps to exactly the original words. -/
theorem int4_pair_indices (column : Nat) :
    column % 16 < 16 ∧ column / 16 * 16 + column % 16 = column ∧
    column / 16 * 2 + column % 16 / 8 = column / 8 := by omega

theorem int4_pair_group (segment j : Nat) (hj : j < 16) :
    (segment * 16 + j) / 64 = segment / 4 := by omega

theorem int4_pair_word_bounds (segment columns : Nat)
    (hs : segment < columns / 16) (ha : columns % 64 = 0) :
    segment * 2 + 1 < columns / 8 := by omega

/-- Euclidean decomposition at any nibble: the high quotient and low residue
reconstruct the original word exactly, without a machine overflow model. -/
theorem int4_split_reconstruction (word lane : Nat) :
    word % 16^lane + 16^lane * (word / 16^lane) = word := by
  exact Nat.mod_add_div word (16^lane)

/-- Eight low-first nibbles reconstruct any unsigned 32-bit packed word. -/
theorem int4_word_reconstruction (word : Nat) (hw : word < 2^32) :
    word = int4Code word 0 + 16 * int4Code word 1 +
      256 * int4Code word 2 + 4096 * int4Code word 3 +
      65536 * int4Code word 4 + 1048576 * int4Code word 5 +
      16777216 * int4Code word 6 + 268435456 * int4Code word 7 := by
  simp only [int4Code, Nat.reducePow, Nat.div_one] at *
  omega

/-- Exact distributivity over mathematical integers, NOT IEEE754 arithmetic.
Pairs are (code, activation), so this also applies to each list of 8 chunks. -/
theorem int4_affine_sum (scale bias : Int) (xs : List (Int × Int)) :
    (xs.map (fun p => (scale * p.1 + bias) * p.2)).sum =
      scale * (xs.map (fun p => p.1 * p.2)).sum +
      bias * (xs.map Prod.snd).sum := by
  induction xs with
  | nil => simp
  | cons p xs ih =>
    simp only [List.map_cons, List.sum_cons, ih]
    simp [Int.add_mul, Int.mul_add, Int.mul_assoc, Int.add_assoc, Int.add_comm, Int.add_left_comm]

/-! MLX-style attention candidate: integer geometry only. No floating-point,
Metal memory-model, compiler, scheduling, or liveness claim is made here. -/
def mlxAttentionDim : Nat := 128
def mlxAttentionLanes : Nat := 32
def mlxAttentionElementsPerLane : Nat := 4
def mlxAttentionSingleSimds : Nat := 32
def mlxAttentionMergeSimds : Nat := 32
def mlxAttentionMaxGqa : Nat := 32
def mlxAttentionMaxHeads : Nat := 64
def mlxAttentionMaxContext : Nat := 262144
def mlxAttentionStateStride : Nat := 130
def mlxAttentionMaxField : Nat := 128
def mlxAttentionSumField : Nat := 129
def mlxAttentionMaxBlocks : Nat := 1024
def mlxAttentionMaxStateElements : Nat :=
  mlxAttentionMaxHeads * mlxAttentionMaxBlocks * mlxAttentionStateStride
def mlxAttentionBlocks : List Nat := [64, 128, 512, 1024]
def validMlxAttentionBlocks (blocks : Nat) : Bool := mlxAttentionBlocks.contains blocks

/-- Host policy for Flint GQA8 on apple_g15d only, not a universal MLX schedule.
This definition certifies integer eligibility, not measured performance. -/
def mlxLargeDevice512MinLength : Nat := 16384
def mlxLargeDevice1024MinLength : Nat := 65536
def mlxLargeDeviceBlocks (length : Nat) : Nat :=
  if length < mlxLargeDevice512MinLength then 128
  else if length < mlxLargeDevice1024MinLength then 512 else 1024
def alignedNormDim : Nat := 2048
def alignedNormThreads : Nat := 512
def alignedNormElementsPerThread : Nat := 4
def alignedHeadDim : Nat := 128
def alignedHeadThreads : Nat := 32
def alignedHeadElementsPerThread : Nat := 4

/-- Exported by the emitter in both languages from these checked definitions. -/
def mlxGeometryConstants : List (String × Nat) :=
  [("mlx_attention_dim", mlxAttentionDim),
   ("mlx_attention_lanes", mlxAttentionLanes),
   ("mlx_attention_elements_per_lane", mlxAttentionElementsPerLane),
   ("mlx_attention_single_simds", mlxAttentionSingleSimds),
   ("mlx_attention_merge_simds", mlxAttentionMergeSimds),
   ("mlx_attention_max_gqa", mlxAttentionMaxGqa),
   ("mlx_attention_max_heads", mlxAttentionMaxHeads),
   ("mlx_attention_max_context", mlxAttentionMaxContext),
   ("mlx_attention_max_blocks", mlxAttentionMaxBlocks),
   ("mlx_attention_max_state_elements", mlxAttentionMaxStateElements),
   ("mlx_attention_state_stride", mlxAttentionStateStride),
   ("mlx_attention_max_field", mlxAttentionMaxField),
   ("mlx_attention_sum_field", mlxAttentionSumField),
   ("aligned_norm_dim", alignedNormDim),
   ("aligned_norm_threads", alignedNormThreads),
   ("aligned_norm_elements_per_thread", alignedNormElementsPerThread),
   ("aligned_head_dim", alignedHeadDim),
   ("aligned_head_threads", alignedHeadThreads),
   ("aligned_head_elements_per_thread", alignedHeadElementsPerThread)]

def mlxAttentionComponent (lane j : Nat) : Nat :=
  lane * mlxAttentionElementsPerLane + j

theorem mlx_attention_geometry :
    mlxAttentionDim = mlxAttentionLanes * mlxAttentionElementsPerLane ∧
    mlxAttentionSingleSimds * mlxAttentionLanes = 1024 ∧
    mlxAttentionMergeSimds * mlxAttentionLanes = 1024 ∧
    mlxAttentionMaxGqa * mlxAttentionLanes = 1024 ∧
    mlxAttentionStateStride = mlxAttentionDim + 2 ∧
    mlxAttentionMaxField = mlxAttentionDim ∧
    mlxAttentionSumField = mlxAttentionDim + 1 := by decide

theorem mlx_attention_blocks_admissible (blocks : Nat) :
    validMlxAttentionBlocks blocks = true ↔
      blocks = 64 ∨ blocks = 128 ∨ blocks = 512 ∨ blocks = 1024 := by
  simp [validMlxAttentionBlocks, mlxAttentionBlocks]

theorem mlx_attention_blocks_bounds (blocks : Nat)
    (hb : validMlxAttentionBlocks blocks = true) :
    0 < blocks ∧ blocks ≤ mlxAttentionMaxBlocks := by
  have := (mlx_attention_blocks_admissible blocks).mp hb
  simp only [mlxAttentionMaxBlocks]
  omega

theorem mlx_attention_max_state_elements : mlxAttentionMaxStateElements = 8519680 := by decide

/-- All policy results are admissible, including lengths outside the supported
context range. Actual launches must separately enforce the context limit. -/
theorem mlx_large_device_blocks_valid (length : Nat) :
    validMlxAttentionBlocks (mlxLargeDeviceBlocks length) = true := by
  unfold mlxLargeDeviceBlocks
  split
  · decide
  · split <;> decide

theorem mlx_large_device_blocks_policy (length : Nat) :
    (length < 16384 → mlxLargeDeviceBlocks length = 128) ∧
    (16384 ≤ length → length < 65536 → mlxLargeDeviceBlocks length = 512) ∧
    (65536 ≤ length → mlxLargeDeviceBlocks length = 1024) := by
  simp only [mlxLargeDeviceBlocks, mlxLargeDevice512MinLength, mlxLargeDevice1024MinLength]
  by_cases h : length < 16384 <;> by_cases h' : length < 65536 <;>
    simp [h, h'] <;> omega

theorem mlx_large_device_blocks_boundaries :
    mlxLargeDeviceBlocks 0 = 128 ∧
    mlxLargeDeviceBlocks 16383 = 128 ∧ mlxLargeDeviceBlocks 16384 = 512 ∧
    mlxLargeDeviceBlocks 65535 = 512 ∧ mlxLargeDeviceBlocks 65536 = 1024 ∧
    mlxLargeDeviceBlocks mlxAttentionMaxContext = 1024 ∧
    mlxLargeDeviceBlocks (2^32 - 1) = 1024 := by decide

/-- Four adjacent components per lane, including the inverse coordinates. -/
theorem mlx_attention_component_coverage (d : Nat) (hd : d < mlxAttentionDim) :
    d / mlxAttentionElementsPerLane < mlxAttentionLanes ∧
    d % mlxAttentionElementsPerLane < mlxAttentionElementsPerLane ∧
    mlxAttentionComponent (d / mlxAttentionElementsPerLane)
      (d % mlxAttentionElementsPerLane) = d := by
  simp only [mlxAttentionDim, mlxAttentionLanes, mlxAttentionElementsPerLane,
    mlxAttentionComponent] at *
  omega

theorem mlx_attention_component_unique (l₁ j₁ l₂ j₂ : Nat)
    (hj₁ : j₁ < mlxAttentionElementsPerLane) (hj₂ : j₂ < mlxAttentionElementsPerLane)
    (he : mlxAttentionComponent l₁ j₁ = mlxAttentionComponent l₂ j₂) :
    l₁ = l₂ ∧ j₁ = j₂ := by
  simp only [mlxAttentionComponent, mlxAttentionElementsPerLane] at *
  omega

theorem mlx_attention_component_bounds (lane j : Nat)
    (hl : lane < mlxAttentionLanes) (hj : j < mlxAttentionElementsPerLane) :
    mlxAttentionComponent lane j < mlxAttentionDim ∧
    mlxAttentionComponent lane j ≠ mlxAttentionMaxField ∧
    mlxAttentionComponent lane j ≠ mlxAttentionSumField := by
  simp only [mlxAttentionComponent, mlxAttentionElementsPerLane, mlxAttentionLanes,
    mlxAttentionDim, mlxAttentionMaxField, mlxAttentionSumField] at *
  omega

/-- B is a stripe count, not a contiguous-token chunk width. -/
def mlxAttentionToken (blocks block iteration : Nat) : Nat := block + iteration * blocks

theorem mlx_attention_token_coverage (t n blocks : Nat)
    (hb : validMlxAttentionBlocks blocks = true) (ht : t < n)
    (hn : n ≤ mlxAttentionMaxContext) :
    t % blocks < blocks ∧ t / blocks < mlxAttentionMaxContext / blocks ∧
    mlxAttentionToken blocks (t % blocks) (t / blocks) = t := by
  rcases (mlx_attention_blocks_admissible blocks).mp hb with rfl | rfl | rfl | rfl <;>
    simp only [mlxAttentionToken, mlxAttentionMaxContext] at * <;> omega

theorem mlx_attention_token_unique (blocks b₁ i₁ b₂ i₂ : Nat)
    (hb : validMlxAttentionBlocks blocks = true) (h₁ : b₁ < blocks) (h₂ : b₂ < blocks)
    (he : mlxAttentionToken blocks b₁ i₁ = mlxAttentionToken blocks b₂ i₂) :
    b₁ = b₂ ∧ i₁ = i₂ := by
  exact row_worker_unique blocks b₁ b₂ i₁ i₂
    (by have := (mlx_attention_blocks_admissible blocks).mp hb; omega) h₁ h₂ he

/-- Both scalar +=B and four-token prefetch +=4B, including the loop test's
lookahead, fit UInt32. Only guarded relative indices are read; absolute ring
addressing and resident-token validity remain caller obligations. -/
theorem mlx_attention_token_u32_bounds (blocks block iteration n : Nat)
    (hb : validMlxAttentionBlocks blocks = true) (hn : n ≤ mlxAttentionMaxContext)
    (guard : mlxAttentionToken blocks block iteration < n) :
    mlxAttentionToken blocks block iteration < mlxAttentionMaxContext ∧
    mlxAttentionToken blocks block iteration + 3 * blocks < 2^32 ∧
    mlxAttentionToken blocks block iteration + 4 * blocks < 2^32 := by
  rcases (mlx_attention_blocks_admissible blocks).mp hb with rfl | rfl | rfl | rfl <;>
    simp only [mlxAttentionMaxContext] at * <;> omega

/-- Initial loop-test lookahead is safe even for empty stripes (n=0 or n<B). -/
theorem mlx_attention_token_init_u32_bounds (blocks block : Nat)
    (hb : validMlxAttentionBlocks blocks = true) (hblock : block < blocks) :
    block < 2^32 ∧ block + 3 * blocks < 2^32 ∧ block + 4 * blocks < 2^32 := by
  have := mlx_attention_blocks_bounds blocks hb
  simp only [mlxAttentionMaxBlocks] at *
  omega

/-- Single-pass token ownership uses the same division/remainder construction. -/
theorem mlx_attention_single_token_coverage (t : Nat) :
    t % mlxAttentionSingleSimds < mlxAttentionSingleSimds ∧
    t % mlxAttentionSingleSimds + (t / mlxAttentionSingleSimds) *
      mlxAttentionSingleSimds = t := by
  exact row_worker_coverage t mlxAttentionSingleSimds (by decide)

/-- In the merge, each SIMD/lane owns blocks s, s+32, ... . -/
theorem mlx_attention_merge_block_coverage (block blocks : Nat)
    (hb : validMlxAttentionBlocks blocks = true) (h : block < blocks) :
    block % mlxAttentionMergeSimds < mlxAttentionMergeSimds ∧
    block / mlxAttentionMergeSimds < blocks / mlxAttentionMergeSimds ∧
    block % mlxAttentionMergeSimds + (block / mlxAttentionMergeSimds) *
      mlxAttentionMergeSimds = block := by
  rcases (mlx_attention_blocks_admissible blocks).mp hb with rfl | rfl | rfl | rfl <;>
    simp only [mlxAttentionMergeSimds] at * <;> omega

/-- GQA producer SIMD s owns head kv_head*GQA+s, exactly once. -/
theorem mlx_attention_gqa_ownership (heads kvheads gqa kvhead simd : Nat)
    (hg : 0 < gqa) (hmax : gqa ≤ mlxAttentionMaxGqa)
    (hheads : heads = kvheads * gqa) (hk : kvhead < kvheads) (hs : simd < gqa) :
    kvhead * gqa + simd < heads ∧
    (kvhead * gqa + simd) / gqa = kvhead ∧
    (kvhead * gqa + simd) % gqa = simd ∧
    gqa * mlxAttentionLanes ≤ 1024 := by
  have hf := attention_state_flatten kvheads gqa kvhead simd hk hs
  simp only [attentionStateSlot] at hf
  refine ⟨by omega, hf.2.1, hf.2.2, ?_⟩
  simp only [mlxAttentionMaxGqa, mlxAttentionLanes] at *
  omega

def mlxAttentionStateIndex (blocks head block field : Nat) : Nat :=
  (head * blocks + block) * mlxAttentionStateStride + field

/-- State allocation is at most 8,519,680 float elements, not bytes. -/
theorem mlx_attention_state_u32_bounds (heads blocks head block field : Nat)
    (hh : heads ≤ mlxAttentionMaxHeads) (hb : validMlxAttentionBlocks blocks = true)
    (hhead : head < heads) (hblock : block < blocks) (hf : field < mlxAttentionStateStride) :
    heads * blocks * mlxAttentionStateStride ≤ mlxAttentionMaxStateElements ∧
    mlxAttentionStateIndex blocks head block field < heads * blocks * mlxAttentionStateStride ∧
    mlxAttentionStateIndex blocks head block field < 2^32 := by
  have flat := (attention_state_flatten heads blocks head block hhead hblock).1
  have bmax : blocks ≤ mlxAttentionMaxBlocks := (mlx_attention_blocks_bounds blocks hb).2
  have allocation := Nat.mul_le_mul_right mlxAttentionStateStride
    (Nat.mul_le_mul (show heads ≤ 64 from hh) bmax)
  have index := Nat.mul_le_mul_right mlxAttentionStateStride
    (show attentionStateSlot blocks head block + 1 ≤ heads * blocks by omega)
  simp only [attentionStateSlot, mlxAttentionStateStride, mlxAttentionMaxBlocks,
    mlxAttentionMaxStateElements, mlxAttentionMaxHeads, Nat.add_mul, Nat.one_mul,
    mlxAttentionStateIndex] at *
  exact ⟨allocation, by omega, by omega⟩

theorem mlx_attention_state_unique (blocks h₁ b₁ f₁ h₂ b₂ f₂ : Nat)
    (hb₁ : b₁ < blocks) (hb₂ : b₂ < blocks)
    (hf₁ : f₁ < mlxAttentionStateStride) (hf₂ : f₂ < mlxAttentionStateStride)
    (he : mlxAttentionStateIndex blocks h₁ b₁ f₁ = mlxAttentionStateIndex blocks h₂ b₂ f₂) :
    h₁ = h₂ ∧ b₁ = b₂ ∧ f₁ = f₂ := by
  have slots := attention_state_field_unique mlxAttentionDim
    (attentionStateSlot blocks h₁ b₁) f₁ (attentionStateSlot blocks h₂ b₂) f₂ hf₁ hf₂ he
  have owners := attention_state_unique blocks h₁ b₁ h₂ b₂ hb₁ hb₂ slots.1
  exact ⟨owners.1, owners.2, slots.2⟩

/-- Every head has B allocated records, even for n<B or n=0. Only block zero
owns its sink. This is ownership, not a softmax or sink-value theorem. -/
def mlxAttentionSinkOwner (block : Nat) : Prop := block = 0

theorem mlx_attention_unique_sink (blocks : Nat) (hb : validMlxAttentionBlocks blocks = true) :
    ∃ block, block < blocks ∧ mlxAttentionSinkOwner block ∧
      ∀ other, other < blocks → mlxAttentionSinkOwner other → other = block := by
  have hp : 0 < blocks := by
    have := (mlx_attention_blocks_admissible blocks).mp hb
    omega
  exact ⟨0, hp, rfl, fun _ _ h => h⟩

/-! Aligned normalization: contiguous four-component integer ownership only. -/
theorem aligned_norm_geometry :
    alignedNormDim = alignedNormThreads * alignedNormElementsPerThread ∧
    alignedHeadDim = alignedHeadThreads * alignedHeadElementsPerThread := by decide

theorem aligned_norm_ownership (d : Nat) (hd : d < alignedNormDim) :
    d / alignedNormElementsPerThread < alignedNormThreads ∧
    d % alignedNormElementsPerThread < alignedNormElementsPerThread ∧
    (d / alignedNormElementsPerThread) * alignedNormElementsPerThread +
      d % alignedNormElementsPerThread = d := by
  simp only [alignedNormDim, alignedNormThreads, alignedNormElementsPerThread] at *
  omega

theorem aligned_norm_unique (t₁ j₁ t₂ j₂ : Nat)
    (hj₁ : j₁ < alignedNormElementsPerThread) (hj₂ : j₂ < alignedNormElementsPerThread)
    (he : t₁ * alignedNormElementsPerThread + j₁ = t₂ * alignedNormElementsPerThread + j₂) :
    t₁ = t₂ ∧ j₁ = j₂ := by
  simp only [alignedNormElementsPerThread] at *
  omega

theorem aligned_norm_bounds (t j : Nat) (ht : t < alignedNormThreads)
    (hj : j < alignedNormElementsPerThread) :
    t * alignedNormElementsPerThread + j < alignedNormDim := by
  simp only [alignedNormThreads, alignedNormElementsPerThread, alignedNormDim] at *
  omega

theorem aligned_head_ownership (d : Nat) (hd : d < alignedHeadDim) :
    d / alignedHeadElementsPerThread < alignedHeadThreads ∧
    d % alignedHeadElementsPerThread < alignedHeadElementsPerThread ∧
    (d / alignedHeadElementsPerThread) * alignedHeadElementsPerThread +
      d % alignedHeadElementsPerThread = d := by
  exact mlx_attention_component_coverage d hd

/-- Fixed-width unrolling visits exactly the original lane-strided segment.
This is integer-only: neither compiler FP reassociation nor device parity is proved. -/
def fixedSegment (lane step : Nat) : Nat := lane + step * 32

theorem fixed_segment_order (lane a b : Nat) (h : a < b) :
    fixedSegment lane a < fixedSegment lane b := by
  unfold fixedSegment
  omega

theorem fixed_segment_coverage (width lane segment : Nat)
    (aligned : width % 512 = 0) (hlane : lane < 32)
    (hsegment : segment < width / 16) (owner : segment % 32 = lane) :
    ∃ step, step < width / 512 ∧ fixedSegment lane step = segment := by
  refine ⟨segment / 32, ?_, ?_⟩
  · omega
  · unfold fixedSegment
    omega

theorem fixed_segment_unique (lane a b : Nat)
    (h : fixedSegment lane a = fixedSegment lane b) : a = b := by
  unfold fixedSegment at h
  omega

theorem fixed_segment_bounds (width lane step : Nat)
    (hlane : lane < 32) (hstep : step < width / 512) :
    fixedSegment lane step < width / 16 := by
  unfold fixedSegment
  omega

/-- Global cache specialization may remove modulo only for a resident linear
prefix. Windowed/ring attention is deliberately excluded by the host guard. -/
theorem linear_cache_index (token capacity : Nat) (resident : token < capacity) :
    token % capacity = token := Nat.mod_eq_of_lt resident

/-- Sharing K/V registers does not share or split a query's SIMD lanes:
each GQA8 local head has one (SIMD, register-head) owner. -/
theorem gqa_reuse_coverage (head h : Nat) (admissible : h = 2 ∨ h = 4)
    (bound : head < 8) :
    ∃ sg slot, sg < 8 / h ∧ slot < h ∧ sg * h + slot = head := by
  rcases admissible with rfl | rfl
  · exact ⟨head / 2, head % 2, by omega⟩
  · exact ⟨head / 4, head % 4, by omega⟩

theorem gqa_reuse_unique (h a x b y : Nat) (admissible : h = 2 ∨ h = 4)
    (hx : x < h) (hy : y < h) (same : a * h + x = b * h + y) :
    a = b ∧ x = y := by
  rcases admissible with rfl | rfl <;> omega

/-- Packing independent stripes into a physical group never crosses a KV
head, since the certified stripe counts are divisible by the packing factor. -/
theorem gqa_grouped_kv_owner (group slot stripes blocks : Nat)
    (hb : stripes = 128 ∨ stripes = 512 ∨ stripes = 1024)
    (hc : blocks = 2 ∨ blocks = 4) (hs : slot < blocks) :
    (group * blocks + slot) / stripes = group / (stripes / blocks) := by
  rcases hb with rfl | rfl | rfl <;> rcases hc with rfl | rfl <;> omega

theorem gqa_grouped_inverse (stripe blocks : Nat) (hc : blocks = 2 ∨ blocks = 4) :
    stripe / blocks * blocks + stripe % blocks = stripe := by
  rcases hc with rfl | rfl <;> omega

/-- Integer specification of discarding 16 low mantissa bits with ties to even.
NaN handling, float interpretation and device code are separate runtime checks. -/
def bf16RoundedHigh (high low : Nat) : Nat :=
  high + if low > 32768 ∨ (low = 32768 ∧ high % 2 = 1) then 1 else 0

theorem bf16_rounding_bias (high low : Nat) (bound : low < 65536) :
    (high * 65536 + low + 32767 + high % 2) / 65536 =
      bf16RoundedHigh high low := by
  unfold bf16RoundedHigh
  split <;> omega

theorem bf16_tie_even (high : Nat) :
    bf16RoundedHigh high 32768 % 2 = 0 := by
  unfold bf16RoundedHigh
  split <;> omega

theorem bf16_finite_round_no_u32_wrap (bits : Nat) (finite_bound : bits ≤ 4286578687) :
    bits + 32767 + (bits / 65536) % 2 < 4294967296 := by omega

/-- Replay parameter-table geometry, emitted to the C bridge and Zig. These
certify indices only, not the semantics of Metal indirect-command barriers. -/
def replayCommands : Nat := 1024
def replayParamStride : Nat := 256

theorem replay_param_alignment (command : Nat) :
    command * replayParamStride % 256 = 0 := by
  unfold replayParamStride
  omega

theorem replay_param_bounds (command bytes : Nat)
    (hc : command < replayCommands) (hb : bytes ≤ replayParamStride) :
    command * replayParamStride + bytes ≤ replayCommands * replayParamStride := by
  unfold replayParamStride replayCommands at *
  omega

theorem replay_param_disjoint (a b bytes : Nat)
    (order : a < b) (bound : bytes ≤ replayParamStride) :
    a * replayParamStride + bytes ≤ b * replayParamStride := by
  unfold replayParamStride at *
  omega

end Resident
