"""
Grounded Prover: LLM-driven proof search with REPL validation.

This module implements the core proving loop where:
1. LLM proposes tactics based on goal state + rulebook
2. REPL validates the tactic
3. LLM sees the result (new goal or error)
4. Repeat until proof complete or budget exhausted

Philosophy:
- NO heuristic fallbacks - the LLM is the prover
- Everything is grounded in Lean - no "assumed valid" proofs
- Rulebook-aware prompting - LLM knows what tactics are available
- Error feedback enables learning/refinement
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..lean_repl.client import LeanREPLClient, ProofState, TacticResult as REPLTacticResult
    from .llm_client import LLMClient

from ..models import Shot, ProofAttempt
from ..rulebook import Rulebook
from .base import get_difficulty_guidance


@dataclass
class ProofSearchConfig:
    """Configuration for proof search."""

    # Budget limits
    max_tactics: int = 20  # Max tactics to try per proof
    max_depth: int = 10  # Max proof depth (for tree search)
    max_retries_per_step: int = 3  # Retries if tactic fails

    # LLM settings
    num_suggestions: int = 5  # Tactics to request from LLM
    temperature: float = 0.3  # LLM temperature

    # Timeout
    tactic_timeout_ms: float = 5000  # Per-tactic timeout

    # Search strategy
    strategy: str = "greedy"  # "greedy", "beam", "mcts" (future)
    beam_width: int = 3  # For beam search


@dataclass
class ProofStep:
    """A single step in the proof."""

    tactic: str
    success: bool
    goals_before: list[str] = field(default_factory=list)
    goals_after: list[str] = field(default_factory=list)
    error_message: str | None = None
    time_ms: float = 0.0


@dataclass
class ProofSearchResult:
    """Result of a proof search."""

    success: bool
    tactics: list[str] = field(default_factory=list)
    steps: list[ProofStep] = field(default_factory=list)
    error_message: str | None = None

    # Metrics
    total_time_ms: float = 0.0
    tactics_tried: int = 0
    llm_calls: int = 0

    def to_proof_attempt(self) -> ProofAttempt:
        """Convert to a ProofAttempt for the game."""
        return ProofAttempt(
            tactics=self.tactics,
            is_complete=self.success,
            is_valid=self.success,
            total_steps=len(self.tactics),
            elapsed_ms=int(self.total_time_ms),
        )


class GroundedProver:
    """
    LLM-driven prover with REPL grounding.

    This is the main proving engine. It:
    1. Takes a theorem statement + rulebook
    2. Interactively queries the LLM for tactics
    3. Validates each tactic via REPL
    4. Provides feedback to LLM on errors
    5. Returns a complete proof or failure

    The key insight is that the LLM sees:
    - The current goal state (from REPL)
    - Available tactics (from Rulebook)
    - Error messages when tactics fail
    - History of what was tried

    This grounds all proof attempts in actual Lean validation.
    """

    # System prompt template for proof search
    SYSTEM_PROMPT = """You are a Lean 4 theorem proving agent. Your task is to prove theorems by suggesting tactics.

{rulebook_section}

IMPORTANT:
- Suggest ONE tactic at a time
- Use exact Lean 4 syntax
- If a tactic fails, learn from the error and try something different
- The proof is complete when there are no remaining goals
"""

    # User prompt for tactic suggestion
    TACTIC_PROMPT = """Current proof state:

GOAL:
{goal}

{hypotheses_section}

TACTICS APPLIED SO FAR:
{tactics_history}

{error_section}

Suggest the next tactic to apply. Respond with ONLY the tactic, no explanation.
"""

    def __init__(
        self,
        repl_client: "LeanREPLClient",
        llm_client: "LLMClient",
        rulebook: "Rulebook",
        config: ProofSearchConfig | None = None,
    ):
        """
        Initialize the grounded prover.

        Args:
            repl_client: REPL client for Lean interaction
            llm_client: LLM client for tactic suggestions
            rulebook: Available tactics/premises
            config: Search configuration
        """
        self.repl = repl_client
        self.llm = llm_client
        self.rulebook = rulebook
        self.config = config or ProofSearchConfig()

    async def prove(
        self,
        theorem_name: str,
        theorem_statement: str,
    ) -> ProofSearchResult:
        """
        Attempt to prove a theorem.

        Args:
            theorem_name: Unique name for the theorem
            theorem_statement: The type to prove

        Returns:
            ProofSearchResult with success status and tactics
        """
        import time

        start_time = time.time()
        result = ProofSearchResult(success=False)

        try:
            # Enter proof mode
            state = await self.repl.enter_theorem(theorem_name, theorem_statement)

            if state.is_complete:
                # Trivially true (shouldn't happen, but handle it)
                result.success = True
                return result

            # Main proof loop
            last_error: str | None = None
            retries_this_step = 0

            while result.tactics_tried < self.config.max_tactics:
                if len(result.tactics) >= self.config.max_depth:
                    result.error_message = f"Max depth ({self.config.max_depth}) reached"
                    break

                # Get tactic suggestion from LLM
                tactic = await self._get_tactic_suggestion(
                    state=state,
                    tactics_so_far=result.tactics,
                    last_error=last_error,
                )
                result.llm_calls += 1
                result.tactics_tried += 1

                if not tactic:
                    result.error_message = "LLM returned no tactic"
                    break

                # Validate against rulebook
                is_allowed, rulebook_error = self.rulebook.validate_tactic(tactic)
                if not is_allowed:
                    last_error = rulebook_error
                    retries_this_step += 1
                    if retries_this_step >= self.config.max_retries_per_step:
                        result.error_message = f"Rulebook violation: {rulebook_error}"
                        break
                    continue

                # Apply tactic via REPL
                tactic_result = await self.repl.run_tactic(
                    state,
                    tactic,
                    timeout_ms=self.config.tactic_timeout_ms,
                )

                step = ProofStep(
                    tactic=tactic,
                    success=tactic_result.success,
                    goals_before=state.goals.copy(),
                    goals_after=tactic_result.new_state.goals if tactic_result.new_state else [],
                    error_message=tactic_result.error.message if tactic_result.error else None,
                    time_ms=tactic_result.time_ms,
                )
                result.steps.append(step)

                if tactic_result.success:
                    result.tactics.append(tactic)
                    if tactic_result.new_state is not None:
                        state = tactic_result.new_state
                    last_error = None
                    retries_this_step = 0

                    # Check if proof is complete
                    if state.is_complete:
                        result.success = True
                        break
                else:
                    # Tactic failed - provide error to LLM
                    last_error = (
                        tactic_result.error.message if tactic_result.error else "Unknown error"
                    )
                    retries_this_step += 1

                    if retries_this_step >= self.config.max_retries_per_step:
                        # Too many failures at this step, give up
                        result.error_message = (
                            f"Failed after {retries_this_step} retries: {last_error}"
                        )
                        break

        except Exception as e:
            result.error_message = f"Proof search error: {str(e)}"

        finally:
            result.total_time_ms = (time.time() - start_time) * 1000
            await self.repl.exit_proof()

        return result

    async def prove_shot(self, shot: Shot) -> ProofSearchResult:
        """
        Prove a shot (convenience wrapper).

        Args:
            shot: The shot to prove

        Returns:
            ProofSearchResult
        """
        return await self.prove(
            theorem_name=f"{shot.theorem_name}_attempt",
            theorem_statement=shot.theorem_statement,
        )

    async def _get_tactic_suggestion(
        self,
        state: "ProofState",
        tactics_so_far: list[str],
        last_error: str | None,
    ) -> str | None:
        """
        Get a tactic suggestion from the LLM.

        Builds a prompt with:
        - Rulebook info
        - Current goal state
        - Tactics applied so far
        - Error feedback if last tactic failed
        """
        # Build system prompt with rulebook
        rulebook_section = self.rulebook.to_system_prompt()

        system_prompt = self.SYSTEM_PROMPT.format(
            rulebook_section=rulebook_section,
        )

        # Build user prompt with current state
        goal = state.goals[0] if state.goals else "No goals"

        hypotheses_section = ""
        if state.hypotheses:
            hyp_lines = [f"  {h.get('name', '?')}: {h.get('type', '?')}" for h in state.hypotheses]
            hypotheses_section = "HYPOTHESES:\n" + "\n".join(hyp_lines)

        tactics_history = (
            "\n".join(f"  {t}" for t in tactics_so_far) if tactics_so_far else "  (none)"
        )

        error_section = ""
        if last_error:
            error_section = f"LAST ERROR:\n{last_error}\n\nPlease try a different approach."

        user_prompt = self.TACTIC_PROMPT.format(
            goal=goal,
            hypotheses_section=hypotheses_section,
            tactics_history=tactics_history,
            error_section=error_section,
        )

        # Query LLM
        try:
            client = self.llm._get_client()
            response = await client.chat.completions.create(
                model=self.llm.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=self.config.temperature,
                max_tokens=100,
            )

            content = response.choices[0].message.content or ""
            # Clean up response - take first non-empty line
            for line in content.strip().split("\n"):
                line = line.strip()
                if line and not line.startswith("#") and not line.startswith("--"):
                    # Remove markdown code blocks if present
                    if line.startswith("`") and line.endswith("`"):
                        line = line[1:-1]
                    return line

            return None

        except Exception as e:
            # Log error but don't crash
            import sys

            print(f"[GroundedProver] LLM error: {e}", file=sys.stderr)
            return None


class GroundedConjecturer:
    """
    LLM-driven conjecture generator with REPL validation.

    Generates novel theorems by:
    1. Asking LLM to propose a theorem + proof
    2. Validating the proof via REPL
    3. If validation fails, asking LLM to fix it
    4. Repeat until valid or budget exhausted

    This ensures all generated shots are PROVABLE.
    """

    CONJECTURE_PROMPT = """You are a Lean 4 mathematician generating theorems for a proving competition.

{rulebook_section}

Generate a theorem at difficulty level {difficulty:.1f}/1.0.

{difficulty_guidance}

{previous_attempts_section}

Respond with JSON ONLY:
{{
  "name": "short_identifier",
  "statement": "forall n : Nat, ...",
  "tactics": ["intro n", "ring"],
  "reasoning": "brief explanation of why this is true"
}}

CRITICAL: 
- The statement must be the TYPE only (no "theorem" keyword)
- Use ASCII: Nat, Int, Real, forall (not Unicode symbols)
- Provide tactics that ACTUALLY prove the statement
"""

    REPAIR_PROMPT = """Your proof attempt failed with error:

{error}

Theorem: {statement}
Attempted tactics: {tactics}

Please provide a corrected proof. Respond with JSON:
{{
  "tactics": ["corrected", "tactics", "here"],
  "reasoning": "what went wrong and how this fixes it"
}}
"""

    def __init__(
        self,
        repl_client: "LeanREPLClient",
        llm_client: "LLMClient",
        rulebook: "Rulebook",
        max_repair_attempts: int = 3,
    ):
        self.repl = repl_client
        self.llm = llm_client
        self.rulebook = rulebook
        self.max_repair_attempts = max_repair_attempts
        self._generated: set[str] = set()

    async def generate_shot(
        self,
        difficulty: float = 0.4,
    ) -> Shot | None:
        """
        Generate a validated shot.

        Args:
            difficulty: Target difficulty (0-1)

        Returns:
            A Shot with a validated proof, or None if generation fails
        """
        from uuid import uuid4

        previous_attempts: list[str] = []
        proposal: dict | None = None
        last_error: str = ""

        for attempt in range(self.max_repair_attempts + 1):
            # Generate or repair
            if attempt == 0:
                proposal = await self._generate_proposal(difficulty, previous_attempts)
            elif proposal is not None:
                proposal = await self._repair_proposal(
                    proposal.get("statement", ""),
                    proposal.get("tactics", []),
                    last_error,
                )

            if not proposal:
                return None

            statement = proposal.get("statement", "")
            tactics = proposal.get("tactics", [])
            name = proposal.get("name", f"gen_{uuid4().hex[:6]}")

            if not statement or not tactics:
                previous_attempts.append(f"Empty statement or tactics")
                continue

            # Skip duplicates
            if statement in self._generated:
                previous_attempts.append(f"Duplicate: {statement}")
                continue

            # Validate via REPL
            theorem_name = f"{name}_{uuid4().hex[:4]}"
            is_valid, last_error = await self._validate_proof(theorem_name, statement, tactics)

            if is_valid:
                self._generated.add(statement)
                return Shot(
                    theorem_name=theorem_name,
                    theorem_statement=statement,
                    challenger_proof=ProofAttempt(
                        tactics=tactics,
                        is_complete=True,
                        is_valid=True,
                        total_steps=len(tactics),
                    ),
                    difficulty_estimate=difficulty,
                    source="llm_generated",
                )

            previous_attempts.append(f"Failed: {last_error}")

        return None

    async def _generate_proposal(
        self,
        difficulty: float,
        previous_attempts: list[str],
    ) -> dict | None:
        """Generate a new theorem proposal from LLM."""
        rulebook_section = self.rulebook.to_system_prompt()
        difficulty_guidance = get_difficulty_guidance(difficulty)

        prev_section = ""
        if previous_attempts:
            prev_section = "Previous failed attempts:\n" + "\n".join(
                f"  - {a}" for a in previous_attempts[-3:]
            )
            prev_section += "\n\nPlease try something different."

        prompt = self.CONJECTURE_PROMPT.format(
            rulebook_section=rulebook_section,
            difficulty=difficulty,
            difficulty_guidance=difficulty_guidance,
            previous_attempts_section=prev_section,
        )

        try:
            client = self.llm._get_client()
            response = await client.chat.completions.create(
                model=self.llm.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.5,
                max_tokens=500,
            )

            content = response.choices[0].message.content or ""
            return self._extract_json(content)

        except Exception:
            return None

    async def _repair_proposal(
        self,
        statement: str,
        tactics: list[str],
        error: str,
    ) -> dict | None:
        """Ask LLM to repair a failed proof."""
        prompt = self.REPAIR_PROMPT.format(
            error=error,
            statement=statement,
            tactics=tactics,
        )

        try:
            client = self.llm._get_client()
            response = await client.chat.completions.create(
                model=self.llm.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=300,
            )

            content = response.choices[0].message.content or ""
            result = self._extract_json(content)

            if result and "tactics" in result:
                # Keep original statement, update tactics
                result["statement"] = statement
                return result

            return None

        except Exception:
            return None

    async def _validate_proof(
        self,
        name: str,
        statement: str,
        tactics: list[str],
    ) -> tuple[bool, str]:
        """
        Validate a proof via REPL.

        Returns:
            (is_valid, error_message)
        """
        try:
            # Build complete theorem
            tactics_str = "\n  ".join(tactics)
            cmd = f"theorem {name} : {statement} := by\n  {tactics_str}"

            # Send to REPL
            env_id = self.repl._state.current_env_id
            response = await self.repl._send_command(cmd, env_id=env_id)

            if response.has_error:
                return False, response.error_message

            if response.sorries:
                return False, "Proof incomplete (contains sorry)"

            return True, ""

        except Exception as e:
            return False, str(e)

    def _extract_json(self, content: str) -> dict | None:
        """Extract JSON from LLM response."""
        # Find JSON object
        start = content.find("{")
        end = content.rfind("}")

        if start == -1 or end == -1 or end <= start:
            return None

        try:
            return json.loads(content[start : end + 1])
        except json.JSONDecodeError:
            return None
