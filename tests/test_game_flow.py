"""
Tests for the H.O.R.S.E. game flow mechanics.

Key invariants tested:
1. Statement Sanitizer: Defender never sees challenger's proof
2. Forking Pattern: Both agents face same environment state
"""

import pytest
from uuid import uuid4

from src.models import Shot, ProofAttempt


class TestStatementSanitizer:
    """Tests for the Statement Sanitizer (get_defender_view)."""

    def test_defender_view_hides_proof(self):
        """The defender's view of a shot must not include the proof."""
        proof = ProofAttempt(
            tactics=["simp", "ring", "exact h"],
            is_complete=True,
            is_valid=True,
        )

        shot = Shot(
            theorem_name="secret_theorem",
            theorem_statement="∀ n : ℕ, n + 0 = n",
            challenger_proof=proof,
        )

        defender_view = shot.get_defender_view()

        # Critical: proof must be hidden
        assert defender_view.challenger_proof is None
        # But statement must be visible
        assert defender_view.theorem_statement == shot.theorem_statement
        assert defender_view.theorem_name == shot.theorem_name

    def test_defender_view_preserves_metadata(self):
        """Metadata like difficulty should be preserved."""
        shot = Shot(
            theorem_name="meta_theorem",
            theorem_statement="∀ x : ℤ, x + 0 = x",
            challenger_proof=ProofAttempt(tactics=["simp"]),
            difficulty_estimate=0.7,
            source="mathlib",
        )

        defender_view = shot.get_defender_view()

        assert defender_view.difficulty_estimate == 0.7
        assert defender_view.source == "mathlib"

    def test_defender_view_is_independent_copy(self):
        """Modifying defender view should not affect original shot."""
        shot = Shot(
            theorem_name="original",
            theorem_statement="∀ n : ℕ, n = n",
            challenger_proof=ProofAttempt(tactics=["rfl"]),
        )

        defender_view = shot.get_defender_view()

        # Modify the defender view
        defender_view.theorem_name = "modified"

        # Original should be unchanged
        assert shot.theorem_name == "original"


class TestForkingPattern:
    """
    Tests for the Forking Pattern.

    The key invariant: Both challenger and defender must face
    the SAME Lean environment state. The challenger's proof
    should not pollute the defender's environment.
    """

    def test_shot_ids_are_preserved(self):
        """The shot ID should be the same in defender view for tracking."""
        shot = Shot(
            theorem_name="fork_test",
            theorem_statement="∀ n : ℕ, 0 + n = n",
            challenger_proof=ProofAttempt(tactics=["simp"]),
        )

        defender_view = shot.get_defender_view()

        # Same ID means we can track it's the same challenge
        assert defender_view.id == shot.id


class TestShotValidity:
    """Tests for Shot.is_valid() method."""

    def test_shot_with_valid_proof_is_valid(self):
        """A shot with a complete, valid proof should be valid."""
        shot = Shot(
            theorem_name="valid_shot",
            theorem_statement="∀ n : ℕ, n + 0 = n",
            challenger_proof=ProofAttempt(
                tactics=["simp"],
                is_complete=True,
                is_valid=True,
            ),
        )

        assert shot.is_valid() is True

    def test_shot_without_proof_is_invalid(self):
        """A shot without a proof should be invalid."""
        shot = Shot(
            theorem_name="no_proof_shot",
            theorem_statement="∀ n : ℕ, n + 0 = n",
        )

        assert shot.is_valid() is False

    def test_shot_with_incomplete_proof_is_invalid(self):
        """A shot with an incomplete proof should be invalid."""
        shot = Shot(
            theorem_name="incomplete_shot",
            theorem_statement="∀ n : ℕ, n + 0 = n",
            challenger_proof=ProofAttempt(
                tactics=["intro n"],
                is_complete=False,
                is_valid=True,
            ),
        )

        assert shot.is_valid() is False
