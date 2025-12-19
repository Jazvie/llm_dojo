/-
ANSPG - Adversarial Neuro-Symbolic Proving Ground
Main Lean 4 library for H.O.R.S.E. theorem proving game.

This file contains example theorems and utilities used in the game.
-/

import Mathlib

namespace ANSPG

/-! ## Basic Arithmetic Theorems

These are simple theorems commonly used in the H.O.R.S.E. game
as seed theorems for mutation-based conjecture generation.
-/

/-- Addition with zero on the right is identity -/
theorem nat_add_zero : ∀ n : ℕ, n + 0 = n := by
  intro n
  rfl

/-- Addition with zero on the left is identity -/
theorem nat_zero_add : ∀ n : ℕ, 0 + n = n := by
  intro n
  simp

/-- Addition is commutative for natural numbers -/
theorem nat_add_comm : ∀ n m : ℕ, n + m = m + n := by
  intros n m
  ring

/-- Addition is associative for natural numbers -/
theorem nat_add_assoc : ∀ a b c : ℕ, (a + b) + c = a + (b + c) := by
  intros a b c
  ring

/-- Multiplication with zero yields zero -/
theorem nat_mul_zero : ∀ n : ℕ, n * 0 = 0 := by
  intro n
  simp

/-- Multiplication with one is identity -/
theorem nat_mul_one : ∀ n : ℕ, n * 1 = n := by
  intro n
  simp

/-- Multiplication is commutative -/
theorem nat_mul_comm : ∀ n m : ℕ, n * m = m * n := by
  intros n m
  ring

/-- Distributive property -/
theorem nat_distrib : ∀ a b c : ℕ, a * (b + c) = a * b + a * c := by
  intros a b c
  ring

/-! ## Integer Theorems -/

/-- Double negation is identity -/
theorem int_neg_neg : ∀ n : ℤ, -(-n) = n := by
  intro n
  ring

/-- Adding negation yields zero -/
theorem int_add_neg : ∀ n : ℤ, n + (-n) = 0 := by
  intro n
  ring

/-! ## Slightly Harder Theorems

These require more sophisticated tactics or multiple steps.
Good for H.O.R.S.E. challenges with constraints.
-/

/-- Sum of first n natural numbers -/
-- TODO: Fix this proof for Mathlib v4.3.0
theorem sum_first_n (n : ℕ) : 2 * (Finset.range (n + 1)).sum id = n * (n + 1) := by
  sorry

/-- Square of sum identity -/
theorem square_of_sum (a b : ℤ) : (a + b)^2 = a^2 + 2*a*b + b^2 := by
  ring

/-- Difference of squares -/
theorem diff_of_squares (a b : ℤ) : a^2 - b^2 = (a + b) * (a - b) := by
  ring

/-! ## Proof Challenges for H.O.R.S.E.

These theorems are designed as challenges with interesting constraint possibilities.
-/

/-- Challenge: Prove without using `simp` -/
theorem challenge_no_simp (_n : ℕ) : _n + 0 = 0 + _n := by
  -- This can be proved with: rw [Nat.add_zero, Nat.zero_add]
  rw [Nat.add_zero, Nat.zero_add]

/-- Challenge: Prove with induction only -/
theorem challenge_induction_only (n : ℕ) : 0 + n = n := by
  induction n with
  | zero => rfl
  | succ k ih => rw [Nat.add_succ, ih]

/-- Challenge: Prove in term mode -/
theorem challenge_term_mode : ∀ n : ℕ, n = n :=
  fun n => rfl

end ANSPG
