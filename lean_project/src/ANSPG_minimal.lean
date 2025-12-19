/-
ANSPG - Adversarial Neuro-Symbolic Proving Ground
Minimal version for testing version alignment
-/

namespace ANSPG

/-- Basic arithmetic theorem -/
theorem nat_add_zero : ∀ n : Nat, n + 0 = n := by
  intro n
  exact Nat.add_zero n

end ANSPG
