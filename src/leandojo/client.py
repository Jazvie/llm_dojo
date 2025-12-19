"""
LeanDojo integration for ANSPG (LEGACY/OPTIONAL).

WARNING: LeanDojo only works with theorems that exist in a pre-traced repository.
For dynamic theorem generation (the primary use case), use the REPL-based
referee in src/orchestrator/repl_referee.py instead.

This module provides traced AST, proof states, and premise retrieval using LeanDojo.
It's useful for static repository analysis but NOT for the dynamic H.O.R.S.E. game.

LeanDojo must be installed separately:
    pip install lean-dojo

Usage (for static repos only):
    client = LeanDojoClient()
    await client.initialize_repo("/path/to/traced/project")

    # Enter an EXISTING theorem (must be in traced repo)
    state = await client.enter_theorem("MyTheorem")

    # Run tactics
    new_state = await client.run_tactic(state, "intro x")

    # Get premise suggestions
    premises = await client.get_relevant_premises(state, k=10)
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TYPE_CHECKING

# LeanDojo types are imported lazily to handle optional dependency
_lean_dojo_available = False
_LeanGitRepo = None
_Dojo = None
_TacticState = None
_ProofFinished = None
_LeanError = None


def _import_lean_dojo():
    """Lazily import LeanDojo."""
    global _lean_dojo_available, _LeanGitRepo, _Dojo, _TacticState, _ProofFinished, _LeanError

    if _lean_dojo_available:
        return True

    try:
        from lean_dojo import LeanGitRepo, Dojo, TacticState, ProofFinished, LeanError

        _LeanGitRepo = LeanGitRepo
        _Dojo = Dojo
        _TacticState = TacticState
        _ProofFinished = ProofFinished
        _LeanError = LeanError
        _lean_dojo_available = True
        return True
    except ImportError:
        return False


@dataclass
class TracedPremise:
    """A premise from traced Lean data."""

    name: str
    statement: str
    module_path: str
    file_path: str
    start_line: int
    end_line: int

    # Dependency information
    dependencies: list[str] = field(default_factory=list)
    dependents: list[str] = field(default_factory=list)


@dataclass
class TracedTacticState:
    """Tactic state with traced information."""

    goals: list[str]
    hypotheses: list[dict[str, str]]  # [{name: type}, ...]

    # Accessible premises from the traced data
    accessible_premises: list[TracedPremise] = field(default_factory=list)

    # Is the proof complete?
    is_complete: bool = False

    # Raw LeanDojo state for advanced usage
    _raw_state: Any = None

    @property
    def pretty(self) -> str:
        """Pretty-print the state."""
        lines = []
        for i, goal in enumerate(self.goals):
            lines.append(f"Goal {i + 1}: {goal}")
        if self.hypotheses:
            lines.append("Hypotheses:")
            for h in self.hypotheses:
                for name, typ in h.items():
                    lines.append(f"  {name} : {typ}")
        return "\n".join(lines)


@dataclass
class TacticApplicationResult:
    """Result of applying a tactic via LeanDojo."""

    success: bool
    new_state: TracedTacticState | None = None
    error_message: str | None = None

    # Time taken for the tactic
    time_ms: float = 0.0


class LeanDojoClient:
    """
    Client for interacting with Lean through LeanDojo.

    Provides:
    - Traced theorem/proof data
    - Tactic execution with rollback
    - Premise retrieval based on proof state
    """

    def __init__(self):
        self._repo = None
        self._dojo = None
        self._current_theorem = None
        self._initialized = False

    @property
    def is_available(self) -> bool:
        """Check if LeanDojo is available."""
        return _import_lean_dojo()

    async def initialize_repo(
        self,
        repo_path: Path | str,
        commit: str | None = None,
    ) -> bool:
        """
        Initialize with a Lean repository.

        Args:
            repo_path: Path to a Lean git repository
            commit: Optional specific commit to use

        Returns:
            True if initialization succeeded
        """
        if not self.is_available:
            raise RuntimeError("LeanDojo is not installed. Install with: pip install lean-dojo")

        repo_path = Path(repo_path)
        if not repo_path.exists():
            raise FileNotFoundError(f"Repository not found: {repo_path}")

        # LeanDojo operations are blocking, run in executor
        loop = asyncio.get_event_loop()

        def _init():
            return _LeanGitRepo(str(repo_path), commit)

        self._repo = await loop.run_in_executor(None, _init)
        self._initialized = True
        return True

    async def enter_theorem(
        self,
        theorem_name: str,
        file_path: str | None = None,
    ) -> TracedTacticState:
        """
        Enter proof mode for a theorem.

        Args:
            theorem_name: Name of the theorem to prove
            file_path: Optional file path if theorem name is ambiguous

        Returns:
            Initial tactic state
        """
        if not self._initialized or not self._repo:
            raise RuntimeError("Repository not initialized")

        loop = asyncio.get_event_loop()

        def _enter():
            # Find the theorem in traced data
            traced_repo = self._repo.trace()

            for traced_file in traced_repo.traced_files:
                for theorem in traced_file.theorems:
                    if theorem.name == theorem_name:
                        if file_path and str(traced_file.path) != file_path:
                            continue

                        # Enter the Dojo
                        dojo = _Dojo(theorem)
                        initial_state = dojo.__enter__()
                        return dojo, initial_state, theorem

            raise ValueError(f"Theorem not found: {theorem_name}")

        self._dojo, raw_state, self._current_theorem = await loop.run_in_executor(None, _enter)

        return self._convert_state(raw_state)

    async def run_tactic(
        self,
        state: TracedTacticState,
        tactic: str,
    ) -> TacticApplicationResult:
        """
        Run a tactic in the current proof.

        Args:
            state: Current tactic state
            tactic: Tactic to apply

        Returns:
            Result with new state or error
        """
        if not self._dojo:
            raise RuntimeError("Not in proof mode")

        loop = asyncio.get_event_loop()
        import time

        def _run():
            start = time.time()
            try:
                raw_state = state._raw_state
                result = self._dojo.run_tac(raw_state, tactic)
                elapsed = (time.time() - start) * 1000

                if isinstance(result, _ProofFinished):
                    return TacticApplicationResult(
                        success=True,
                        new_state=TracedTacticState(
                            goals=[],
                            hypotheses=[],
                            is_complete=True,
                        ),
                        time_ms=elapsed,
                    )
                elif isinstance(result, _LeanError):
                    return TacticApplicationResult(
                        success=False,
                        error_message=str(result),
                        time_ms=elapsed,
                    )
                else:
                    # New TacticState
                    new_state = self._convert_state(result)
                    return TacticApplicationResult(
                        success=True,
                        new_state=new_state,
                        time_ms=elapsed,
                    )
            except Exception as e:
                return TacticApplicationResult(
                    success=False,
                    error_message=str(e),
                )

        return await loop.run_in_executor(None, _run)

    async def get_accessible_premises(
        self,
        state: TracedTacticState,
        k: int = 20,
    ) -> list[TracedPremise]:
        """
        Get premises accessible from the current state.

        Args:
            state: Current tactic state
            k: Maximum number of premises to return

        Returns:
            List of accessible premises
        """
        if not self._dojo or not state._raw_state:
            return []

        loop = asyncio.get_event_loop()

        def _get_premises():
            try:
                raw_state = state._raw_state
                # LeanDojo provides accessible premises
                if hasattr(raw_state, "get_accessible_premises"):
                    premises = raw_state.get_accessible_premises()
                    return [
                        TracedPremise(
                            name=p.full_name,
                            statement=p.statement if hasattr(p, "statement") else "",
                            module_path=p.module_path if hasattr(p, "module_path") else "",
                            file_path=str(p.file_path) if hasattr(p, "file_path") else "",
                            start_line=p.start_line if hasattr(p, "start_line") else 0,
                            end_line=p.end_line if hasattr(p, "end_line") else 0,
                        )
                        for p in premises[:k]
                    ]
            except Exception:
                pass
            return []

        return await loop.run_in_executor(None, _get_premises)

    async def exit_proof(self) -> None:
        """Exit the current proof mode."""
        if self._dojo:
            loop = asyncio.get_event_loop()

            def _exit():
                self._dojo.__exit__(None, None, None)

            await loop.run_in_executor(None, _exit)
            self._dojo = None
            self._current_theorem = None

    def _convert_state(self, raw_state: Any) -> TracedTacticState:
        """Convert a LeanDojo TacticState to our format."""
        goals = []
        hypotheses = []

        if hasattr(raw_state, "goals"):
            goals = [str(g) for g in raw_state.goals]

        if hasattr(raw_state, "hypotheses"):
            for h in raw_state.hypotheses:
                if hasattr(h, "name") and hasattr(h, "type"):
                    hypotheses.append({h.name: str(h.type)})

        return TracedTacticState(
            goals=goals,
            hypotheses=hypotheses,
            is_complete=len(goals) == 0,
            _raw_state=raw_state,
        )

    @property
    def is_in_proof(self) -> bool:
        """Check if currently in proof mode."""
        return self._dojo is not None


class LeanDojoProver:
    """
    High-level prover using LeanDojo.

    Combines LeanDojo's traced data with tactic search.
    """

    def __init__(self, client: LeanDojoClient):
        self.client = client

    async def prove_with_trace(
        self,
        theorem_name: str,
        tactics: list[str],
        file_path: str | None = None,
    ) -> tuple[bool, list[str]]:
        """
        Attempt to prove a theorem with a sequence of tactics.

        Args:
            theorem_name: Name of the theorem
            tactics: List of tactics to apply
            file_path: Optional file path

        Returns:
            (success, list of applied tactics)
        """
        try:
            state = await self.client.enter_theorem(theorem_name, file_path)
            applied = []

            for tactic in tactics:
                result = await self.client.run_tactic(state, tactic)
                if not result.success:
                    break
                applied.append(tactic)
                if result.new_state and result.new_state.is_complete:
                    return True, applied
                state = result.new_state

            return False, applied
        finally:
            await self.client.exit_proof()

    async def get_premise_based_tactics(
        self,
        state: TracedTacticState,
        k: int = 5,
    ) -> list[str]:
        """
        Generate tactic suggestions based on accessible premises.

        Args:
            state: Current tactic state
            k: Number of suggestions

        Returns:
            List of suggested tactics
        """
        premises = await self.client.get_accessible_premises(state, k=k * 2)
        tactics = []

        for premise in premises:
            # Generate tactics that use this premise
            name = premise.name
            tactics.append(f"apply {name}")
            tactics.append(f"exact {name}")
            if "eq" in premise.statement.lower():
                tactics.append(f"rw [{name}]")

        return tactics[:k]


def create_leandojo_client() -> LeanDojoClient:
    """
    Factory to create a LeanDojo client.

    Raises RuntimeError if LeanDojo is not installed - it is REQUIRED
    for deterministic proof verification.
    """
    if not _import_lean_dojo():
        raise RuntimeError(
            "LeanDojo is required but not installed.\n"
            "Install with: pip install lean-dojo\n"
            "See https://leandojo.org/ for setup instructions."
        )
    return LeanDojoClient()
