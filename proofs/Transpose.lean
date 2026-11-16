import Std

/-! Integer ownership for four disjoint 32x32 transpose planes. Metal barrier
    semantics and FP32 code generation remain device-tested obligations. -/
namespace Transpose

def index (component lane simd : Nat) : Nat := component * 1024 + lane * 32 + simd

theorem bounds (component lane simd : Nat)
    (hc : component < 4) (hl : lane < 32) (hs : simd < 32) :
    index component lane simd < 4096 := by
  unfold index
  omega

theorem unique (c₁ l₁ s₁ c₂ l₂ s₂ : Nat)
    (_hc₁ : c₁ < 4) (hl₁ : l₁ < 32) (hs₁ : s₁ < 32)
    (_hc₂ : c₂ < 4) (hl₂ : l₂ < 32) (hs₂ : s₂ < 32)
    (he : index c₁ l₁ s₁ = index c₂ l₂ s₂) :
    c₁ = c₂ ∧ l₁ = l₂ ∧ s₁ = s₂ := by
  unfold index at he
  omega

theorem read_is_owned_write (component lane simd : Nat) :
    index component simd lane = component * 1024 + simd * 32 + lane := rfl

#print axioms bounds
#print axioms unique
end Transpose
