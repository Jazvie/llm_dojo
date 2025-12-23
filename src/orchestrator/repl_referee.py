"""
REPL-based Referee for ANSPG H.O.R.S.E. game.

Uses the Lean REPL for dynamic theorem verification.

Key capabilities:
- Validate dynamically generated theorems
- Execute tactics step-by-step
- Forking pattern: each proof attempt gets a fresh environment
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..lean_repl.client import LeanREPLClient, create_repl_client
from ..lean_repl.errors import LeanErrorCategory
from ..models import (
    Shot,
    ProofAttempt,
    TacticState,
    TacticResult as ModelTacticResult,
)


@dataclass
class REPLValidationResult:
    """Result of proof validation via REPL."""

    is_valid: bool
    proof_complete: bool = False
    error_message: str | None = None

    # Traced data for analysis
    tactics_applied: list[str] = field(default_factory=list)
    premises_used: list[str] = field(default_factory=list)
    final_goals: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "is_valid": self.is_valid,
            "proof_complete": self.proof_complete,
            "error_message": self.error_message,
            "tactics_applied": self.tactics_applied,
            "premises_used": self.premises_used,
            "final_goals": self.final_goals,
        }


class REPLReferee:
    """
    Deterministic referee using Lean REPL for proof verification.

    Responsibilities:
    - Verify arbitrary new theorems by executing them in Lean
    - Provide deterministic pass/fail outcomes
    """

    def __init__(
        self,
        repl_path: Path | str,
        lean_project: Path | str,
        init_timeout: float = 500.0,
        verbose: bool = False,
    ):
        """
        Initialize the REPL referee.

        Args:
            repl_path: Path to the REPL executable (lake exe repl)
            lean_project: Path to the Lean project with Mathlib imports
            init_timeout: Timeout for initialization (importing Mathlib can be slow)
            verbose: Enable debug output for timing info
        """
        self._repl_path = Path(repl_path)
        self._lean_project = Path(lean_project)
        self._init_timeout = init_timeout
        self._verbose = verbose
        self._client: LeanREPLClient | None = None
        self._initialized = False

    async def initialize(self) -> bool:
        """
        Initialize REPL with the Lean project.

        Returns:
            True if initialization succeeded
        """
        if not self._repl_path.exists():
            raise FileNotFoundError(
                f"REPL not found at {self._repl_path}. Build it with: cd repl && lake build"
            )

        if not self._lean_project.exists():
            raise FileNotFoundError(f"Lean project not found: {self._lean_project}")

        # Create REPL client with longer timeout for Mathlib import
        self._client = create_repl_client(
            str(self._repl_path), timeout_seconds=self._init_timeout, verbose=self._verbose
        )

        # Start the REPL process
        success = await self._client.start(self._lean_project)
        if not success:
            return False

        # Initialize with Mathlib imports
        await self._client.initialize(repo_path=self._lean_project, imports=["Mathlib"])

        self._initialized = True
        return True

    async def validate_shot(self, shot: Shot) -> REPLValidationResult:
        """
        Validate a challenger's shot using REPL.

        Uses command mode to compile the full theorem with proof tactics.
        This ensures all Mathlib tactics are available (unlike tactic mode).

        Args:
            shot: The shot to validate

        Returns:
            REPLValidationResult with detailed outcome
        """
        if not self._initialized or not self._client:
            return REPLValidationResult(is_valid=False, error_message="Referee not initialized")

        if not shot.challenger_proof or not shot.challenger_proof.tactics:
            return REPLValidationResult(is_valid=False, error_message="Shot has no proof")

        tactics = shot.challenger_proof.tactics

        try:
            # Build the complete theorem declaration with all tactics
            tactics_str = "\n  ".join(tactics)
            cmd = f"theorem {shot.theorem_name} : {shot.theorem_statement} := by\n  {tactics_str}"

            # Send as a command (not tactic mode) - this has access to all Mathlib tactics
            # Use the current environment ID that has Mathlib imported
            env_id = self._client._state.current_env_id
            response = await self._client._send_command(cmd, env_id=env_id)

            if response.has_error:
                return REPLValidationResult(
                    is_valid=False,
                    error_message=f"Proof failed: {response.error_message}",
                    tactics_applied=tactics,
                )

            # Check for sorries in the response - if there are sorries, the proof is incomplete
            if response.sorries:
                return REPLValidationResult(
                    is_valid=False,
                    error_message="Proof incomplete - contains sorry",
                    tactics_applied=tactics,
                )

            # Proof compiled successfully
            return REPLValidationResult(
                is_valid=True,
                proof_complete=True,
                tactics_applied=tactics,
            )

        except Exception as e:
            return REPLValidationResult(
                is_valid=False,
                error_message=f"Validation error: {str(e)}",
            )

    async def validate_defense(
        self,
        shot: Shot,
        defense: ProofAttempt,
    ) -> REPLValidationResult:
        """
        Validate a defender's proof attempt using REPL.

        Uses command mode to compile the full theorem with proof tactics.
        This ensures all Mathlib tactics are available.

        Args:
            shot: The original shot (theorem statement)
            defense: The defender's proof attempt

        Returns:
            REPLValidationResult with detailed outcome
        """
        if not self._initialized or not self._client:
            return REPLValidationResult(is_valid=False, error_message="Referee not initialized")

        if not defense.tactics:
            return REPLValidationResult(is_valid=False, error_message="Defense has no tactics")

        tactics = defense.tactics

        try:
            # Build the complete theorem declaration with all tactics
            # Use a different name to avoid collision with the challenger's theorem
            theorem_name = f"{shot.theorem_name}_defense"
            tactics_str = "\n  ".join(tactics)
            cmd = f"theorem {theorem_name} : {shot.theorem_statement} := by\n  {tactics_str}"

            # Send as a command (not tactic mode) - this has access to all Mathlib tactics
            # Use the current environment ID that has Mathlib imported
            env_id = self._client._state.current_env_id
            response = await self._client._send_command(cmd, env_id=env_id)

            if response.has_error:
                return REPLValidationResult(
                    is_valid=False,
                    error_message=f"Defense failed: {response.error_message}",
                    tactics_applied=tactics,
                )

            # Check for sorries in the response - if there are sorries, the proof is incomplete
            if response.sorries:
                return REPLValidationResult(
                    is_valid=False,
                    error_message="Defense incomplete - contains sorry",
                    tactics_applied=tactics,
                )

            # Proof compiled successfully
            return REPLValidationResult(
                is_valid=True,
                proof_complete=True,
                tactics_applied=tactics,
            )

        except Exception as e:
            return REPLValidationResult(
                is_valid=False,
                error_message=f"Defense validation error: {str(e)}",
            )

    async def close(self) -> None:
        """Close the REPL process."""
        if self._client:
            await self._client.close()

    @property
    def is_ready(self) -> bool:
        """Check if the referee is initialized and ready."""
        return self._initialized and self._client is not None


async def create_repl_referee(repl_path: Path | str, lean_project: Path | str) -> REPLReferee:
    """
    Factory to create and initialize a REPL referee.

    Args:
        repl_path: Path to the REPL executable
        lean_project: Path to the Lean project

    Returns:
        Initialized REPLReferee

    Raises:
        RuntimeError: If initialization fails
    """
    referee = REPLReferee(repl_path, lean_project)
    success = await referee.initialize()
    if not success:
        raise RuntimeError(
            f"Failed to initialize REPL referee. Check that the REPL is built "
            f"(cd {repl_path.parent if isinstance(repl_path, Path) else 'repl'} && lake build) "
            f"and the Lean project exists at {lean_project}"
        )
    return referee
