"""
Tests for the rulebook and grounded proving system.

Tests cover:
1. Rulebook presets and tactic validation
2. Proof search configuration
"""

import pytest

from src.rulebook import (
    Rulebook,
    TacticSet,
    TacticInfo,
    TacticCategory,
    create_basic_rulebook,
    create_mathlib_rulebook,
    create_competition_rulebook,
    create_custom_rulebook,
)


class TestRulebookPresets:
    """Tests for rulebook preset factories."""

    def test_basic_rulebook_has_core_tactics(self):
        """Basic rulebook should have core Lean 4 tactics."""
        rb = create_basic_rulebook()

        assert rb.name == "basic"
        assert rb.tactics.is_available("rfl")
        assert rb.tactics.is_available("intro")
        assert rb.tactics.is_available("cases")
        assert rb.tactics.is_available("simp")

        # Should NOT have mathlib automation
        assert not rb.tactics.is_available("omega")
        assert not rb.tactics.is_available("aesop")

    def test_mathlib_rulebook_has_automation(self):
        """Mathlib rulebook should have full automation tactics."""
        rb = create_mathlib_rulebook()

        assert rb.name == "mathlib"
        assert rb.tactics.is_available("omega")
        assert rb.tactics.is_available("linarith")
        assert rb.tactics.is_available("aesop")
        assert rb.tactics.is_available("ring")
        assert rb.tactics.is_available("exact?")

    def test_competition_rulebook_no_search(self):
        """Competition rulebook should not have search tactics."""
        rb = create_competition_rulebook()

        assert rb.name == "competition"
        # Has standard tactics
        assert rb.tactics.is_available("omega")
        assert rb.tactics.is_available("simp")

        # But NOT search tactics
        assert not rb.tactics.is_available("aesop")
        assert not rb.tactics.is_available("exact?")
        assert not rb.tactics.is_available("apply?")

    def test_custom_rulebook(self):
        """Custom rulebook should only have specified tactics."""
        rb = create_custom_rulebook(
            name="minimal",
            allowed_tactics=["intro", "exact", "rfl"],
            max_steps=5,
        )

        assert rb.name == "minimal"
        assert rb.tactics.is_available("intro")
        assert rb.tactics.is_available("exact")
        assert rb.tactics.is_available("rfl")
        assert not rb.tactics.is_available("simp")
        assert rb.max_steps == 5


class TestTacticValidation:
    """Tests for tactic validation against rulebook."""

    def test_validate_allowed_tactic(self):
        """Allowed tactics should pass validation."""
        rb = create_mathlib_rulebook()

        is_valid, error = rb.validate_tactic("simp")
        assert is_valid
        assert error is None

    def test_validate_tactic_with_args(self):
        """Tactics with arguments should be validated by base name."""
        rb = create_mathlib_rulebook()

        is_valid, _ = rb.validate_tactic("rw [Nat.add_comm]")
        assert is_valid

        is_valid, _ = rb.validate_tactic("intro x y z")
        assert is_valid

    def test_validate_disallowed_tactic(self):
        """Tactics not in rulebook should fail validation."""
        rb = create_basic_rulebook()

        is_valid, error = rb.validate_tactic("omega")
        assert not is_valid
        assert "omega" in error


class TestTacticSet:
    """Tests for TacticSet operations."""

    def test_add_and_get_tactic(self):
        """Should be able to add and retrieve tactics."""
        ts = TacticSet()
        tactic = TacticInfo("test_tactic", TacticCategory.CLOSING, "A test tactic")

        ts.add(tactic)

        assert ts.get("test_tactic") == tactic
        assert ts.is_available("test_tactic")

    def test_by_category(self):
        """Should filter tactics by category."""
        rb = create_mathlib_rulebook()

        closing = rb.tactics.by_category(TacticCategory.CLOSING)
        assert any(t.name == "rfl" for t in closing)

        arithmetic = rb.tactics.by_category(TacticCategory.ARITHMETIC)
        assert any(t.name == "omega" for t in arithmetic)

    def test_to_prompt_text(self):
        """Should generate readable prompt text."""
        rb = create_basic_rulebook()
        prompt = rb.tactics.to_prompt_text()

        assert "Available tactics:" in prompt
        assert "rfl" in prompt
        assert "intro" in prompt


class TestRulebookPromptGeneration:
    """Tests for generating LLM prompts from rulebook."""

    def test_to_system_prompt(self):
        """Should generate a complete system prompt."""
        rb = create_mathlib_rulebook()
        prompt = rb.to_system_prompt()

        assert "mathlib" in prompt.lower()
        assert "Available tactics:" in prompt
        assert "Available types:" in prompt
        assert "Nat" in prompt

    def test_to_dict(self):
        """Should serialize to dict."""
        rb = create_mathlib_rulebook()
        data = rb.to_dict()

        assert data["name"] == "mathlib"
        assert "simp" in data["tactics"]
        assert "Nat" in data["types"]
