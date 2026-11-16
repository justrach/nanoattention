import Std

/-! Integer traversal only: smaller register prefetch batches must visit precisely
    the original ascending attention stripe. Floating-point compilation and
    actual Metal execution are checked separately by bitwise device tests. -/
namespace Prefetch

def supported (chunk : Nat) : Prop := chunk = 1 ∨ chunk = 2 ∨ chunk = 4

theorem supported_positive (chunk : Nat) (h : supported chunk) : 0 < chunk := by
  rcases h with h | h | h <;> omega

theorem ordinal_reconstruction (ordinal chunk : Nat) :
    chunk * (ordinal / chunk) + ordinal % chunk = ordinal := by
  exact Nat.div_add_mod ordinal chunk

theorem stripe_reconstruction (ordinal chunk blocks stripe : Nat) :
    (chunk * (ordinal / chunk) + ordinal % chunk) * blocks + stripe =
      ordinal * blocks + stripe := by
  rw [ordinal_reconstruction]

theorem component_bounds (ordinal chunk : Nat) (h : supported chunk) :
    ordinal % chunk < chunk := by
  exact Nat.mod_lt ordinal (supported_positive chunk h)

theorem chunk_component_unique (q₁ r₁ q₂ r₂ chunk : Nat)
    (h : supported chunk) (h₁ : r₁ < chunk) (h₂ : r₂ < chunk)
    (he : chunk * q₁ + r₁ = chunk * q₂ + r₂) : q₁ = q₂ ∧ r₁ = r₂ := by
  rcases h with h | h | h <;> subst chunk <;> omega

theorem physical_ring513_index (slot : Nat) (h : slot < 513) : slot % 513 = slot := by
  exact Nat.mod_eq_of_lt h

#print axioms physical_ring513_index
#print axioms stripe_reconstruction
#print axioms chunk_component_unique
end Prefetch
