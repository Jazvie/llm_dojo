"""
LeanDojo-based Referee for ANSPG H.O.R.S.E. game (LEGACY/OPTIONAL).

WARNING: This referee only works with theorems that exist in a pre-traced
repository. For dynamic theorem generation (the primary use case), use
REPLReferee from repl_referee.py instead.

This module is kept for compatibility with static repository analysis where
theorems are known ahead of time.

Unlike the REPL-based referee, LeanDojo provides:
- Step-by-step tactic execution with full tracing
- Access to proof state AST and premises
- Deterministic success/failure outcomes

But it CANNOT verify dynamically generated theorems.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any
from uuid import UUID
from pathlib import Path

from ..leandojo import (
    LeanDojoClient,
    TracedTacticState,
    TacticApplicationResult,
    TracedPremise,
    create_leandojo_client,
)
from ..models import (
    Shot,
    ProofAttempt,
    TurnResult,
    TacticState,
    TacticResult,
)


@dataclass
class LeanDojoValidationResult:
    """Result of proof validation via LeanDojo."""

    is_valid: bool
    proof_complete: bool = False
    error_message: str | None = None

    # Traced data for analysis
    tactics_applied: list[str] = field(default_factory=list)
    final_state: TracedTacticState | None = None
    premises_used: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "is_valid": self.is_valid,
            "proof_complete": self.proof_complete,
            "error_message": self.error_message,
            "tactics_applied": self.tactics_applied,
            "premises_used": self.premises_used,
        }


class LeanDojoReferee:
    """
    Deterministic referee using LeanDojo for proof verification.

    This is the ground truth for the H.O.R.S.E. game. All proof
    verification goes through LeanDojo's traced execution.

    Responsibilities:
    - Verify proofs by executing tactics step-by-step
    - Provide deterministic pass/fail outcomes
    """

    def __init__(self, repo_path: Path | str):
        """
        Initialize the LeanDojo referee.

        Args:
            repo_path: Path to the Lean repository to use for tracing
        """
        self._repo_path = Path(repo_path)
        self._client: LeanDojoClient | None = None
        self._initialized = False

    async def initialize(self) -> bool:
        """
        Initialize LeanDojo with the repository.

        Returns:
            True if initialization succeeded
        """
        self._client = create_leandojo_client()

        if not self._client.is_available:
            raise RuntimeError(
                "LeanDojo is not installed. This system requires LeanDojo for "
                "deterministic proof verification. Install with: pip install lean-dojo"
            )

        success = await self._client.initialize_repo(self._repo_path)
        self._initialized = success
        return success

    async def validate_shot(self, shot: Shot) -> LeanDojoValidationResult:
        """
        Validate a challenger's shot using LeanDojo.

        This executes the proof step-by-step and verifies:
        1. Each tactic succeeds
        2. The proof completes (no remaining goals)

        Args:
            shot: The shot to validate

        Returns:
            LeanDojoValidationResult with detailed outcome
        """
        if not self._initialized or not self._client:
            return LeanDojoValidationResult(is_valid=False, error_message="Referee not initialized")

        if not shot.challenger_proof or not shot.challenger_proof.tactics:
            return LeanDojoValidationResult(is_valid=False, error_message="Shot has no proof")

        try:
            # Enter the theorem
            state = await self._client.enter_theorem(shot.theorem_name)
            applied_tactics = []
            premises_used = []

            # Execute each tactic
            for tactic in shot.challenger_proof.tactics:
                result = await self._client.run_tactic(state, tactic)

                if not result.success:
                    await self._client.exit_proof()
                    return LeanDojoValidationResult(
                        is_valid=False,
                        error_message=f"Tactic failed: {tactic} - {result.error_message}",
                        tactics_applied=applied_tactics,
                    )

                applied_tactics.append(tactic)

                if result.new_state:
                    state = result.new_state
                    # Track premises from state
                    if state.accessible_premises:
                        premises_used.extend(p.name for p in state.accessible_premises[:5])

                    if state.is_complete:
                        break

            # Check if proof is complete
            proof_complete = state.is_complete if state else False

            if not proof_complete:
                await self._client.exit_proof()
                return LeanDojoValidationResult(
                    is_valid=False,
                    error_message="Proof incomplete - goals remaining",
                    tactics_applied=applied_tactics,
                    final_state=state,
                )

            await self._client.exit_proof()

            return LeanDojoValidationResult(
                is_valid=True,
                proof_complete=True,
                tactics_applied=applied_tactics,
                final_state=state,
                premises_used=list(set(premises_used)),
            )

        except Exception as e:
            if self._client and self._client.is_in_proof:
                await self._client.exit_proof()
            return LeanDojoValidationResult(
                is_valid=False,
                error_message=f"Validation error: {str(e)}",
            )

    async def validate_defense(
        self,
        shot: Shot,
        defense: ProofAttempt,
    ) -> LeanDojoValidationResult:
        """
        Validate a defender's proof attempt using LeanDojo.

        Args:
            shot: The original shot (theorem to prove)
            defense: The defender's proof attempt

        Returns:
            LeanDojoValidationResult with detailed outcome
        """
        if not self._initialized or not self._client:
            return LeanDojoValidationResult(is_valid=False, error_message="Referee not initialized")

        if not defense.tactics:
            return LeanDojoValidationResult(is_valid=False, error_message="Defense has no tactics")

        try:
            # Enter the theorem
            state = await self._client.enter_theorem(shot.theorem_name)
            applied_tactics = []
            premises_used = []

            # Execute each tactic
            for tactic in defense.tactics:
                result = await self._client.run_tactic(state, tactic)

                if not result.success:
                    await self._client.exit_proof()
                    return LeanDojoValidationResult(
                        is_valid=False,
                        error_message=f"Tactic failed: {tactic} - {result.error_message}",
                        tactics_applied=applied_tactics,
                    )

                applied_tactics.append(tactic)

                if result.new_state:
                    state = result.new_state
                    if state.accessible_premises:
                        premises_used.extend(p.name for p in state.accessible_premises[:5])

                    if state.is_complete:
                        break

            proof_complete = state.is_complete if state else False

            if not proof_complete:
                await self._client.exit_proof()
                return LeanDojoValidationResult(
                    is_valid=False,
                    error_message="Defense incomplete - goals remaining",
                    tactics_applied=applied_tactics,
                    final_state=state,
                )

            await self._client.exit_proof()

            return LeanDojoValidationResult(
                is_valid=True,
                proof_complete=True,
                tactics_applied=applied_tactics,
                final_state=state,
                premises_used=list(set(premises_used)),
            )

        except Exception as e:
            if self._client and self._client.is_in_proof:
                await self._client.exit_proof()
            return LeanDojoValidationResult(
                is_valid=False,
                error_message=f"Defense validation error: {str(e)}",
            )

    async def handle_bluff_call(self, shot: Shot) -> tuple[bool, str]:
        """
        Handle a bluff call by re-validating the challenger's proof.

        Returns:
            (bluff_successful, message)
            - bluff_successful: True if challenger's proof was invalid
        """
        if not shot.challenger_proof:
            return True, "Challenger has no proof - bluff successful"

        result = await self.validate_shot(shot)

        if not result.is_valid:
            return True, f"Bluff successful! Proof invalid: {result.error_message}"
        else:
            return False, "Bluff failed - challenger's proof is valid"

    def determine_outcome(
        self,
        shot: Shot,
        defense_result: LeanDojoValidationResult | None,
        defender_timed_out: bool = False,
        bluff_called: bool = False,
        bluff_successful: bool = False,
    ) -> TurnResult:
        """
        Determine the outcome of a turn based on LeanDojo validation.
        """
        turn_result = TurnResult(
            shot=shot,
            bluff_called=bluff_called,
            bluff_successful=bluff_successful,
        )

        if bluff_called:
            turn_result.defender_succeeded = bluff_successful
            return turn_result

        if defender_timed_out:
            turn_result.defender_succeeded = False
            return turn_result

        if defense_result and defense_result.is_valid:
            turn_result.defender_succeeded = True
        else:
            turn_result.defender_succeeded = False

        return turn_result

    @property
    def is_ready(self) -> bool:
        """Check if the referee is initialized and ready."""
        return self._initialized and self._client is not None


async def create_leandojo_referee(repo_path: Path | str) -> LeanDojoReferee:
    """
    Factory to create and initialize a LeanDojo referee.

    Args:
        repo_path: Path to the Lean repository

    Returns:
        Initialized LeanDojoReferee
    """
    referee = LeanDojoReferee(repo_path)
    await referee.initialize()
    return referee
