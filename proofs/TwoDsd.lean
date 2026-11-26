import Std

/-!
Standalone integer specification of 2DSD scheduler contracts. No model, GPU,
Metal overlapping decode, or floating-point claims. Mirrors `src/twodsd.zig`:
the shared-last window constraint `R * k ≤ W`, last-loop KV accounting, greedy
prefix acceptance at loop `R`, and standard rejection-sampling residuals.

Kim, Chavan, Seo, Hu, Lee, Kang, Mahoney, Keutzer, Hooper and Gholami,
*2DSD: Two-Dimensional Speculative Decoding for Looped Language Models* (2026),
https://github.com/SqueezeAILab/2DSD
-/
namespace TwoDsd

def windowFits (R k W : Nat) : Prop := R * k ≤ W

theorem window_eq_ok (R k : Nat) : windowFits R k (R * k) := Nat.le_refl _

theorem window_too_small {R k W : Nat} (h : W < R * k) : ¬ windowFits R k W :=
  Nat.not_le.mpr h

/-- History KV bytes: `layers × seq × width × elem × R` for per-loop `full`. -/
def historyKvFull (layers R seq width elem : Nat) : Nat :=
  layers * seq * width * elem * R

/-- Last-loop sharing: one committed history plus an in-flight window per loop. -/
def historyKvSharedLast (layers R seq width elem W : Nat) : Nat :=
  layers * seq * width * elem + layers * W * width * elem * R

theorem shared_last_eq_history_plus_window
    (layers R seq width elem W : Nat) :
    historyKvSharedLast layers R seq width elem W =
      layers * seq * width * elem + layers * W * width * elem * R := rfl

theorem full_is_R_histories
    (layers R seq width elem : Nat) :
    historyKvFull layers R seq width elem =
      (layers * seq * width * elem) * R := by
  simp [historyKvFull, Nat.mul_assoc]

def stage2Legal (d1 d2 R : Nat) : Prop := d1 < d2 ∧ d2 < R

theorem stage2_has_room {d1 d2 R : Nat} (h : stage2Legal d1 d2 R) : d1 + 1 < R := by
  obtain ⟨h1, h2⟩ := h
  omega

/-- Greedy prefix acceptance at loop R, matching `src/speculative.zig` / Speculative.lean. -/
def acceptedCount : List Nat → List Nat → Option Nat → Nat → Nat
  | _, _, _, 0 => 0
  | [], _, _, _ => 0
  | _, [], _, _ => 0
  | d :: ds, t :: ts, eos, budget + 1 =>
      if d = t then
        if eos = some d then 1 else acceptedCount ds ts eos budget + 1
      else 0

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

/-- Accept draft token `x` with probability `min(1, t/g)`: `u < t/g` iff `u_num * g < t * u_den`. -/
def accept (t g u_num u_den : Nat) : Bool := decide (u_num * g < t * u_den)

theorem accept_iff (t g u_num u_den : Nat) :
    accept t g u_num u_den = true ↔ u_num * g < t * u_den := by
  simp [accept]

/-- If the target mass is at least the proposal mass, every `u ∈ [0, 1)` accepts. -/
theorem certain_accept
    {t g u_num u_den : Nat} (h : g ≤ t) (hu : u_num < u_den) (hg : 0 < g) :
    accept t g u_num u_den = true := by
  simp [accept]
  have h1 : u_num * g < u_den * g := Nat.mul_lt_mul_of_pos_right hu hg
  have h2 : u_den * g ≤ u_den * t := Nat.mul_le_mul_left u_den h
  omega

/-- Residual `t − g` after a rejected proposal (truncating subtraction; renormalization is out of scope). -/
def residual (t g : List Nat) : List Nat :=
  List.zipWith (fun a b => a - b) t g

theorem residual_length (t g : List Nat) :
    (residual t g).length = min t.length g.length := by
  simp [residual, List.length_zipWith]

end TwoDsd
