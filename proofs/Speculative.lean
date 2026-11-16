import Std

/-!
Standalone specification of speculative greedy prefix acceptance. No model/GPU
or floating-point claims: the caller must supply exact target greedy predictions
with the ordinary decoder's argmax/tie policy. For draft inputs d0..d(k-1), the
prediction list is `first :: after_each`: first predicts d0; batch row i predicts
d(i+1). Only predictions through the first mismatch are on the committed path.

`acceptedCount` mirrors the matched-prefix branch of src/speculative.zig's
`accept`, including the output budget and an accepted EOS. The correction/bonus
is outside this count. This is a specification proof, not a verified compilation
or a claim that batch logits equal sequential logits.
-/
namespace Speculative

def acceptedCount : List Nat → List Nat → Option Nat → Nat → Nat
  | _, _, _, 0 => 0
  | [], _, _, _ => 0
  | _, [], _, _ => 0
  | d :: ds, t :: ts, eos, budget + 1 =>
      if d = t then
        if eos = some d then 1 else acceptedCount ds ts eos budget + 1
      else 0

/-- Every accepted draft token equals the target prediction at the same position,
including early termination at EOS and truncation by the remaining budget. -/
theorem accepted_prefix_eq (draft target : List Nat) (eos : Option Nat) (budget : Nat) :
    draft.take (acceptedCount draft target eos budget) =
      target.take (acceptedCount draft target eos budget) := by
  induction budget generalizing draft target with
  | zero => simp [acceptedCount]
  | succ budget ih =>
      cases draft with
      | nil => simp [acceptedCount]
      | cons d ds =>
          cases target with
          | nil => simp [acceptedCount]
          | cons t ts =>
              by_cases equal : d = t
              · subst t
                by_cases stop : eos = some d
                · simp [acceptedCount, stop]
                · simpa [acceptedCount, stop] using congrArg (List.cons d) (ih ds ts)
              · simp [acceptedCount, equal]

/-- The accepted portion alone never exceeds the remaining output budget. -/
theorem accepted_le_budget (draft target : List Nat) (eos : Option Nat) (budget : Nat) :
    acceptedCount draft target eos budget ≤ budget := by
  induction budget generalizing draft target with
  | zero => simp [acceptedCount]
  | succ budget ih =>
      cases draft with
      | nil => simp [acceptedCount]
      | cons d ds =>
          cases target with
          | nil => simp [acceptedCount]
          | cons t ts =>
              simp only [acceptedCount]
              split
              · split
                · omega
                · exact Nat.add_le_add_right (ih ds ts) 1
              · omega

/-!
Integer-only specifications for the layer-major batch dispatch and sliding-ring
journal. These definitions describe mathematical natural numbers, not finite
machine arithmetic, generated kernel code, floating-point evaluation, execution
ordering, or physical cache restoration. Integration must separately establish
that actual dispatch sizes and arithmetic implement these definitions.
-/

def laneFor (i : Nat) : Nat := i % 32

def tokenFor (count i : Nat) : Nat := (i / 32) % count

def rowTileFor (count i : Nat) : Nat := (i / 32) / count

/-- Inverse layout: lane within SIMD, then token, then row tile. -/
def threadIndex (count rowTile token lane : Nat) : Nat :=
  (rowTile * count + token) * 32 + lane

/-- Every thread index is recovered from its decoded coordinates. -/
theorem mapping_inverse (count i : Nat) (_hc : 0 < count) :
    threadIndex count (rowTileFor count i) (tokenFor count i) (laneFor i) = i := by
  unfold threadIndex rowTileFor tokenFor laneFor
  have block := Nat.div_add_mod (i / 32) count
  have thread := Nat.div_add_mod i 32
  rw [Nat.mul_comm ((i / 32) / count) count, block]
  omega

/-- Each legal coordinate triple has a thread with exactly those coordinates. -/
theorem mapping_coverage (count rowTile token lane : Nat) (hc : 0 < count)
    (ht : token < count) (hl : lane < 32) :
    let i := threadIndex count rowTile token lane
    rowTileFor count i = rowTile ∧ tokenFor count i = token ∧ laneFor i = lane := by
  have divided : threadIndex count rowTile token lane / 32 = rowTile * count + token := by
    unfold threadIndex
    omega
  have row : (rowTile * count + token) / count = rowTile := by
    rw [Nat.mul_comm rowTile count, Nat.mul_add_div hc, Nat.div_eq_of_lt ht]
    omega
  have tok : (rowTile * count + token) % count = token := by
    rw [Nat.mul_comm rowTile count, Nat.mul_add_mod, Nat.mod_eq_of_lt ht]
  dsimp only
  refine ⟨?_, ?_, ?_⟩
  · simpa only [rowTileFor, divided] using row
  · simpa only [tokenFor, divided] using tok
  · unfold laneFor threadIndex
    omega

/-- A bounded dispatch decodes only legal row, token, and SIMD-lane indices. -/
theorem mapping_bounds (count tiles i : Nat) (hc : 0 < count)
    (hi : i < tiles * count * 32) :
    rowTileFor count i < tiles ∧ tokenFor count i < count ∧ laneFor i < 32 := by
  have block : i / 32 < tiles * count := (Nat.div_lt_iff_lt_mul (by decide)).2 hi
  refine ⟨?_, Nat.mod_lt (i / 32) hc, Nat.mod_lt i (by decide)⟩
  exact (Nat.div_lt_iff_lt_mul hc).2 block

/-- Every legal triple lies inside the corresponding finite dispatch. -/
theorem covered_index_bounds (count tiles rowTile token lane : Nat) (_hc : 0 < count)
    (hr : rowTile < tiles) (ht : token < count) (hl : lane < 32) :
    threadIndex count rowTile token lane < tiles * count * 32 := by
  have next : (rowTile + 1) * count ≤ tiles * count :=
    Nat.mul_le_mul_right count (by omega)
  rw [Nat.add_mul, Nat.one_mul] at next
  have block : rowTile * count + token < tiles * count := by omega
  unfold threadIndex
  omega

/-- Journal destination for speculative input j, relative to the starting position. -/
def ringSlot (start capacity j : Nat) : Nat := (start + j) % capacity

/-- Every journal destination is a valid ring index. -/
theorem ring_slot_bounds (start capacity j : Nat) (hc : 0 < capacity) :
    ringSlot start capacity j < capacity := Nat.mod_lt (start + j) hc

/-- At most one wrap occurs after adding any offset smaller than capacity;
therefore translating a bounded set of offsets around the ring is injective. -/
theorem ring_slots_injective (start capacity count j k : Nat)
    (hc : 0 < capacity) (hcount : count ≤ capacity)
    (hj : j < count) (hk : k < count)
    (he : ringSlot start capacity j = ringSlot start capacity k) : j = k := by
  have hjc : j < capacity := by omega
  have hkc : k < capacity := by omega
  have hb := Nat.mod_lt start hc
  have wrap (x : Nat) (hx : x < capacity + capacity) :
      x % capacity = if x < capacity then x else x - capacity := by
    split
    · exact Nat.mod_eq_of_lt (by assumption)
    · rw [Nat.mod_eq_sub_mod (a := x) (b := capacity) (by omega)]
      exact Nat.mod_eq_of_lt (by omega)
  unfold ringSlot at he
  rw [Nat.add_mod start j capacity, Nat.add_mod start k capacity,
    Nat.mod_eq_of_lt hjc, Nat.mod_eq_of_lt hkc,
    wrap (start % capacity + j) (by omega),
    wrap (start % capacity + k) (by omega)] at he
  split at he <;> split at he <;> omega

/-- Different speculative inputs never journal the same ring slot when the
batch count is no larger than the ring capacity. -/
theorem ring_slots_distinct (start capacity count j k : Nat)
    (hc : 0 < capacity) (hcount : count ≤ capacity)
    (hj : j < count) (hk : k < count) (hne : j ≠ k) :
    ringSlot start capacity j ≠ ringSlot start capacity k := by
  intro he
  exact hne (ring_slots_injective start capacity count j k hc hcount hj hk he)

/-- The configured maximum batch of 32 fits a 513-slot sliding ring. -/
theorem max_batch_fits_ring (count : Nat) (hcount : count ≤ 32) : count ≤ 513 := by
  omega

/-- Keeping only an accepted prefix cannot move the committed position beyond
its speculative endpoint, which itself must have been checked against context. -/
theorem accepted_position_bounds (start accepted count context : Nat)
    (ha : accepted ≤ count) (hc : start + count ≤ context) :
    start + accepted ≤ start + count ∧ start + count ≤ context := by
  exact ⟨Nat.add_le_add_left ha start, hc⟩

/-- An accepted journal slot is distinct from every rejected journal slot.
This separates the integer destinations; it does not prove physical restoration. -/
theorem accepted_rejected_slots_distinct (start capacity count accepted j k : Nat)
    (hc : 0 < capacity) (hcount : count ≤ capacity) (ha : accepted ≤ count)
    (hj : j < accepted) (hklo : accepted ≤ k) (hkhi : k < count) :
    ringSlot start capacity j ≠ ringSlot start capacity k := by
  apply ring_slots_distinct start capacity count j k hc hcount (by omega) hkhi
  omega

end Speculative

#print axioms Speculative.accepted_prefix_eq
#print axioms Speculative.accepted_le_budget
#print axioms Speculative.mapping_inverse
#print axioms Speculative.mapping_coverage
#print axioms Speculative.mapping_bounds
#print axioms Speculative.covered_index_bounds
#print axioms Speculative.ring_slot_bounds
#print axioms Speculative.ring_slots_injective
#print axioms Speculative.ring_slots_distinct
#print axioms Speculative.max_batch_fits_ring
#print axioms Speculative.accepted_position_bounds
#print axioms Speculative.accepted_rejected_slots_distinct
