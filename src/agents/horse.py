"""
HorseAgent: Unified agent for H.O.R.S.E. theorem proving game.

This is the main agent class that can both:
- take_shot(): Propose AND prove a theorem (challenger role)
- match_shot(): Prove someone else's theorem (defender role)

KEY DESIGN PRINCIPLE:
Both challenger and defender use the SAME mechanism:
1. LLM proposes a complete proof (list of tactics)
2. REPL validates the proof in one shot
3. If valid → success; if invalid → MISS (no fallback!)

This mirrors real H.O.R.S.E.:
- You take a shot (propose + make it) → if you miss, you miss
- Opponent tries to match (make the same shot their way) → if they miss, they get a letter

NO FALLBACKS. If your proof doesn't work, you missed your shot. Period.

DUPLICATE STATEMENT POLICY:
- Same statement, WITHIN a turn (retry): ALLOWED - lets agent fix proof errors
- Same statement, ACROSS turns: BLOCKED - prevents "farming" a theorem
- Same statement, different agent: ALLOWED - each agent proposes independently

This prevents an agent from repeatedly proposing the same theorem that their
opponent keeps failing on, encouraging variety in the game.

Enforced via the optional DuplicateStatementTracker passed to the agent.
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING
from uuid import uuid4

if TYPE_CHECKING:
    from ..lean_repl.client import LeanREPLClient, ProofState

from ..models import Shot, ProofAttempt, DuplicateStatementTracker
from ..rulebook import Rulebook
from ..game_config import SimpPolicy
from .base import get_difficulty_guidance
from .llm_client import LLMClient


@dataclass
class HorseAgentConfig:
    """Configuration for a H.O.R.S.E. agent."""

    name: str = "Agent"
    model: str = "gpt-4o-mini"

    # LLM settings
    temperature: float = 0.3
    max_tokens: int = 500

    # Conjecture settings (challenger)
    difficulty_target: float = 0.4  # 0 = trivial, 1 = very hard
    max_conjecture_attempts: int = 3  # Retries for generating a valid theorem

    # Simp policy (game-level setting passed down)
    simp_policy: SimpPolicy = SimpPolicy.ALLOWED


@dataclass
class ShotResult:
    """Result of a shot attempt (take_shot or match_shot)."""

    success: bool
    tactics: list[str] = field(default_factory=list)
    error_message: str | None = None
    time_ms: float = 0.0

    def to_proof_attempt(self) -> ProofAttempt:
        """Convert to a ProofAttempt for the game model."""
        return ProofAttempt(
            tactics=self.tactics,
            is_complete=self.success,
            is_valid=self.success,
            total_steps=len(self.tactics),
            elapsed_ms=int(self.time_ms),
        )


@dataclass
class AgentStats:
    """Statistics for an agent's performance."""

    shots_proposed: int = 0
    shots_validated: int = 0  # Own proofs that passed
    shots_matched: int = 0  # Successfully defended
    shots_failed: int = 0  # Failed to defend
    total_tactics_used: int = 0
    total_time_ms: float = 0.0

    # Debugging - track last failure details
    last_failure_reason: str | None = None
    last_failure_attempts: list[str] = field(default_factory=list)


class HorseAgent:
    """
    Unified agent for H.O.R.S.E. theorem proving.

    Capabilities:
    - take_shot(): Generate a novel theorem and prove it
    - match_shot(): Prove a theorem proposed by opponent

    Both operations use the SAME validation mechanism:
    1. LLM proposes complete tactics
    2. REPL validates in one shot
    3. No fallbacks - miss is a miss
    """

    # Prompt for generating novel theorems (challenger)
    CONJECTURE_PROMPT = """You are a Lean 4 mathematician in a H.O.R.S.E. proving competition.

{rulebook_section}

Your task: Generate a NOVEL theorem at difficulty level {difficulty:.1f}/1.0.

{difficulty_guidance}

GAME MECHANICS:
- You have {max_attempts} attempts to propose a valid theorem+proof
- This is attempt {current_attempt} of {max_attempts}
- Your opponent gets only ONE attempt to match your proof
- Strategy: Balance difficulty - too easy and opponent matches; too hard and you might fail

{already_used_section}

{previous_section}

Respond with JSON ONLY:
{{
  "name": "descriptive_snake_case_name",
  "statement": "<your novel theorem statement here>",
  "tactics": ["tactic1", "tactic2", "..."],
  "reasoning": "brief explanation of why this theorem is interesting and provable"
}}

CRITICAL RULES:
- "statement" is the TYPE only (no "theorem" keyword)
- Use ASCII: Nat, Int, Real, forall (NOT Unicode symbols)
- "tactics" must be a COMPLETE proof - all tactics needed to close all goals
- Your tactics will be validated - if they don't compile, you MISS
- DO NOT repeat theorems you have already used (see list above if any)
- Be CREATIVE - propose something different from basic identities
"""

    # Prompt for matching a shot (defender)
    MATCH_PROMPT = """You are a Lean 4 mathematician defending in a H.O.R.S.E. competition.

{rulebook_section}

Your opponent proposed this theorem. You must prove it to avoid getting a letter.

THEOREM TO PROVE:
{statement}

GAME MECHANICS:
- You have exactly ONE attempt to prove this theorem
- If your proof fails, you get a letter (H, O, R, S, or E)
- Spell HORSE and you lose the game

Respond with JSON ONLY:
{{
  "tactics": ["intro n", "ring", "..."],
  "reasoning": "brief explanation of proof strategy"
}}

CRITICAL RULES:
- "tactics" must be a COMPLETE proof - all tactics needed to close all goals
- Use only tactics from the rulebook above
- If you can't prove it, you MISS and get a letter
- Your tactics will be validated - if they don't compile, you MISS
"""

    def __init__(
        self,
        repl_client: "LeanREPLClient",
        llm_client: LLMClient,
        rulebook: Rulebook,
        config: HorseAgentConfig | None = None,
        duplicate_tracker: DuplicateStatementTracker | None = None,
    ):
        """
        Initialize the agent.

        Args:
            repl_client: REPL for proof validation
            llm_client: LLM for tactic suggestions
            rulebook: Available tactics/premises
            config: Agent configuration
            duplicate_tracker: Optional shared tracker for cross-turn statement duplicates
        """
        self.repl = repl_client
        self.llm = llm_client
        self.rulebook = rulebook
        self.config = config or HorseAgentConfig()
        self.stats = AgentStats()
        # Statement duplicate tracker (shared across agents for game-level tracking)
        self._duplicate_tracker = duplicate_tracker

    @property
    def name(self) -> str:
        return self.config.name

    async def take_shot(self) -> Shot | None:
        """
        Take a shot: propose a theorem AND prove it.

        This is the challenger's action. The agent must:
        1. Generate a theorem statement + proof tactics
        2. Have it validated by the REPL
        3. If valid → Shot accepted
        4. If invalid → try again (up to max_conjecture_attempts)

        Duplicate Policy:
            - Same statement, within this turn: ALLOWED (retry with different proof)
            - Same statement, from previous turn: BLOCKED (no farming)

        Returns:
            A Shot with validated proof, or None if all attempts fail
        """
        self.stats.shots_proposed += 1
        previous_failures: list[str] = []

        for attempt in range(self.config.max_conjecture_attempts):
            # Step 1: Generate a theorem + proof from LLM
            proposal = await self._generate_proposal(previous_failures, current_attempt=attempt + 1)

            if not proposal:
                previous_failures.append("LLM returned no proposal")
                continue

            statement = proposal.get("statement", "")
            name = proposal.get("name", f"thm_{uuid4().hex[:6]}")
            tactics = proposal.get("tactics", [])

            if not statement:
                previous_failures.append("Empty statement")
                continue

            if not tactics:
                previous_failures.append("No tactics provided")
                continue

            # Check for duplicate statements across turns (game-level)
            # This prevents "farming" the same theorem against a struggling opponent
            if self._duplicate_tracker and self._duplicate_tracker.is_duplicate(
                self.config.name, statement
            ):
                previous_failures.append(
                    f"Duplicate statement (already used in previous turn): {statement[:50]}..."
                )
                continue

            # Step 2: Validate the proof via REPL (ONE SHOT, no fallback)
            theorem_name = f"{name}_{uuid4().hex[:4]}"
            is_valid, error = await self._validate_proof(theorem_name, statement, tactics)

            if is_valid:
                # Step 3: Triviality check (if NO_AUTO_SIMP policy is active)
                # Reject theorems that can be solved by simp alone
                if self.config.simp_policy == SimpPolicy.NO_AUTO_SIMP:
                    trivial_name = f"trivial_check_{uuid4().hex[:4]}"
                    is_trivial, _ = await self._validate_proof(trivial_name, statement, ["simp"])
                    if is_trivial:
                        previous_failures.append(
                            f"Theorem too trivial (solvable by 'simp' alone): {statement[:50]}..."
                        )
                        continue

                # Shot succeeded. Record the statement to prevent reuse across turns
                if self._duplicate_tracker:
                    self._duplicate_tracker.record_statement(self.config.name, statement)
                self.stats.shots_validated += 1
                self.stats.total_tactics_used += len(tactics)

                return Shot(
                    theorem_name=theorem_name,
                    theorem_statement=statement,
                    challenger_proof=ProofAttempt(
                        tactics=tactics,
                        is_complete=True,
                        is_valid=True,
                        total_steps=len(tactics),
                    ),
                    difficulty_estimate=self.config.difficulty_target,
                    source="llm_generated",
                )
            else:
                # Shot missed - record failure and try a DIFFERENT theorem
                previous_failures.append(f"Proof failed for '{statement[:40]}...': {error}")

        # All attempts exhausted - record for debugging
        self.stats.last_failure_reason = (
            f"All {self.config.max_conjecture_attempts} attempts failed"
        )
        self.stats.last_failure_attempts = previous_failures.copy()
        return None

    async def match_shot(self, shot: Shot) -> ShotResult:
        """
        Match a shot: prove a theorem proposed by opponent.

        This is the defender's action. The agent:
        1. Sees only the theorem statement (NOT the challenger's proof)
        2. Proposes their own complete proof
        3. Has it validated by the REPL
        4. If valid → MATCH (defender becomes challenger)
        5. If invalid → MISS (defender gets a letter)

        This is symmetric with take_shot:
        - Same mechanism: propose tactics → validate → done
        - Only difference: defender doesn't propose the theorem

        Note: No duplicate checking here - defender is matching, not proposing.

        Args:
            shot: The shot to match (from get_defender_view() - no proof visible)

        Returns:
            ShotResult with success status and tactics
        """
        start_time = time.time()

        # Step 1: Ask LLM to generate proof tactics
        proof_proposal = await self._generate_match_proof(shot)

        if not proof_proposal:
            self.stats.shots_failed += 1
            return ShotResult(
                success=False,
                error_message="LLM returned no proof proposal",
                time_ms=(time.time() - start_time) * 1000,
            )

        tactics = proof_proposal.get("tactics", [])

        if not tactics:
            self.stats.shots_failed += 1
            return ShotResult(
                success=False,
                error_message="No tactics provided",
                time_ms=(time.time() - start_time) * 1000,
            )

        # Step 2: Validate the proof via REPL (ONE SHOT, no fallback)
        # Note: No duplicate check here - defender is matching, not proposing
        defense_name = f"{shot.theorem_name}_defense_{uuid4().hex[:4]}"
        is_valid, error = await self._validate_proof(defense_name, shot.theorem_statement, tactics)

        elapsed = (time.time() - start_time) * 1000

        if is_valid:
            self.stats.shots_matched += 1
            self.stats.total_tactics_used += len(tactics)
            self.stats.total_time_ms += elapsed

            return ShotResult(
                success=True,
                tactics=tactics,
                time_ms=elapsed,
            )
        else:
            self.stats.shots_failed += 1
            self.stats.total_time_ms += elapsed

            return ShotResult(
                success=False,
                tactics=tactics,  # Include what was tried
                error_message=error,
                time_ms=elapsed,
            )

    async def _generate_proposal(
        self,
        previous_failures: list[str],
        current_attempt: int = 1,
    ) -> dict | None:
        """Generate a theorem + proof proposal from the LLM."""
        rulebook_section = self.rulebook.to_system_prompt()
        difficulty_guidance = get_difficulty_guidance(self.config.difficulty_target)

        # Build section showing previously used statements (from successful shots)
        already_used_section = ""
        if self._duplicate_tracker:
            used_statements = self._duplicate_tracker._used_statements.get(self.config.name, set())
            if used_statements:
                # Show the original (non-normalized) statements from history
                history = [
                    stmt
                    for agent, stmt in self._duplicate_tracker._statement_history
                    if agent == self.config.name
                ]
                if history:
                    already_used_section = (
                        "THEOREMS YOU HAVE ALREADY USED (do NOT repeat these):\n"
                        + "\n".join(f"  - {stmt}" for stmt in history[-10:])  # Last 10
                    )

        # Build section showing failures in current turn
        previous_section = ""
        if previous_failures:
            recent = previous_failures
            previous_section = "PREVIOUS ATTEMPTS THIS TURN (failed):\n" + "\n".join(
                f"  - {f}" for f in recent
            )

        prompt = self.CONJECTURE_PROMPT.format(
            rulebook_section=rulebook_section,
            difficulty=self.config.difficulty_target,
            difficulty_guidance=difficulty_guidance,
            max_attempts=self.config.max_conjecture_attempts,
            current_attempt=current_attempt,
            already_used_section=already_used_section,
            previous_section=previous_section,
        )

        return await self._query_llm_json(prompt)

    async def _generate_match_proof(self, shot: Shot) -> dict | None:
        """Generate a proof for matching a shot."""
        rulebook_section = self.rulebook.to_system_prompt()

        prompt = self.MATCH_PROMPT.format(
            rulebook_section=rulebook_section,
            statement=shot.theorem_statement,
        )

        return await self._query_llm_json(prompt)

    async def _validate_proof(
        self,
        name: str,
        statement: str,
        tactics: list[str],
    ) -> tuple[bool, str]:
        """
        Validate a proof via REPL command mode.

        This is the core validation - same for challenger and defender.
        One shot, no retries, no fallbacks.

        Args:
            name: Theorem name for REPL
            statement: Theorem statement (type)
            tactics: List of tactics to apply

        Returns:
            (is_valid, error_message)
        """
        try:
            # Validate tactics against rulebook first
            for tactic in tactics:
                is_allowed, error = self.rulebook.validate_tactic(tactic)
                if not is_allowed:
                    return False, f"Rulebook violation: {error}"

            # Build the complete theorem command
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

    async def _query_llm_json(self, prompt: str) -> dict | None:
        """Query the LLM and parse JSON response."""
        try:
            client = self.llm._get_client()
            response = await client.chat.completions.create(
                model=self.llm.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            )

            content = response.choices[0].message.content or ""
            return self._extract_json(content)

        except Exception:
            return None

    def _extract_json(self, content: str) -> dict | None:
        """Extract JSON object from LLM response."""
        start = content.find("{")
        end = content.rfind("}")

        if start == -1 or end == -1 or end <= start:
            return None

        try:
            return json.loads(content[start : end + 1])
        except json.JSONDecodeError:
            return None

    def get_stats(self) -> dict:
        """Get agent statistics as a dict."""
        return {
            "name": self.config.name,
            "model": self.config.model,
            "shots_proposed": self.stats.shots_proposed,
            "shots_validated": self.stats.shots_validated,
            "shots_matched": self.stats.shots_matched,
            "shots_failed": self.stats.shots_failed,
            "total_tactics": self.stats.total_tactics_used,
            "total_time_ms": self.stats.total_time_ms,
        }
