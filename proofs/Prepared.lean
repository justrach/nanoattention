import Std

/-! Prepared input layout: N scaled values followed by N/16 cached sums.
    These integer proofs do not assert floating-point reassociation laws. -/
namespace Prepared

theorem scaled_bounds (n segment component : Nat)
    (hn : n % 16 = 0) (hs : segment < n / 16) (hc : component < 16) :
    segment * 16 + component < n := by omega

theorem sum_bounds (n segment : Nat) (hs : segment < n / 16) :
    n + segment < n + n / 16 := by omega

theorem scaled_sum_disjoint (n segment component other : Nat)
    (hn : n % 16 = 0) (hs : segment < n / 16) (hc : component < 16) :
    segment * 16 + component ≠ n + other := by
  have := scaled_bounds n segment component hn hs hc
  omega

theorem segment_component_unique (s₁ c₁ s₂ c₂ : Nat)
    (hc₁ : c₁ < 16) (hc₂ : c₂ < 16)
    (he : s₁ * 16 + c₁ = s₂ * 16 + c₂) : s₁ = s₂ ∧ c₁ = c₂ := by omega

theorem padded_route_sum_bounds (stride columns segment : Nat)
    (hc : columns ≤ stride) (hs : segment < columns / 16) :
    stride + segment < stride + stride / 16 := by omega

#print axioms scaled_sum_disjoint
#print axioms padded_route_sum_bounds
end Prepared
