"""
Base Agent class for ANSPG.

Defines the interface that all agents (Prover, Conjecturer, LLM-backed) must implement.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any
from uuid import UUID, uuid4

from ..models import Shot, ProofAttempt

#this can and probabily will be changed in the future, but mostly is here for debugging purposes.
def get_difficulty_guidance(difficulty: float) -> str:
    """Generate concrete guidance for theorem difficulty level.

    Args:
        difficulty: Float from 0.0 (trivial) to 1.0 (very hard)

    Returns:
        String with concrete examples and expectations for the difficulty level
    """
    if difficulty <= 0.2:
        return """DIFFICULTY 0.0-0.2 (Trivial):
- Simple reflexivity or direct computation (e.g., "forall n : Nat, n = n")
- One-step proofs using rfl, trivial, or decide
- Basic arithmetic identities (e.g., "forall n : Nat, 0 + n = n")
- Proof should be 1-2 tactics maximum"""
    elif difficulty <= 0.4:
        return """DIFFICULTY 0.2-0.4 (Easy):
- Simple properties requiring intro + one closing tactic
- Basic commutativity/associativity (e.g., "forall a b : Nat, a + b = b + a")
- Properties solvable by omega, simp, or ring alone
- Proof should be 2-3 tactics"""
    elif difficulty <= 0.6:
        return """DIFFICULTY 0.4-0.6 (Medium):
- Properties requiring case analysis or simple induction
- Theorems needing multiple rewrites or lemma applications
- Inequalities requiring linarith with setup (e.g., "forall n : Nat, n <= n + 1")
- Proof should be 3-5 tactics"""
    elif difficulty <= 0.8:
        return """DIFFICULTY 0.6-0.8 (Hard):
- Theorems requiring induction with non-trivial inductive step
- Properties involving multiple quantifiers
- Results needing careful case splits and multiple lemmas
- Proof should be 5-8 tactics"""
    else:
        return """DIFFICULTY 0.8-1.0 (Very Hard):
- Complex inductions with multiple base cases
- Theorems requiring auxiliary lemmas or clever witnesses
- Properties involving nested quantifiers or complex types
- Proof may need 8+ tactics with careful structuring"""


@dataclass
class AgentConfig:
    """Configuration for an agent."""

    name: str = "Agent"
    model: str = "gpt-4"  # or "claude-3-opus", etc.
    temperature: float = 0.7
    max_tokens: int = 4096

    # Proof search parameters
    max_search_depth: int = 50
    beam_width: int = 5
    timeout_per_tactic_ms: int = 5000

    # Conjecturer parameters
    mutation_rate: float = 0.3
    novelty_weight: float = 0.5
    difficulty_target: float = 0.5  # Target pass rate


class BaseAgent(ABC):
    """
    Abstract base class for all ANSPG agents.

    Agents must implement:
    - generate_shot(): Create a theorem + proof
    - attempt_proof(): Try to prove a given theorem
    """

    def __init__(
        self,
        config: AgentConfig | None = None,
    ):
        self._id = uuid4()
        self.config = config or AgentConfig()

        # Statistics
        self.proofs_attempted = 0
        self.proofs_succeeded = 0
        self.shots_generated = 0
        self.total_tokens_used = 0

    @property
    def id(self) -> UUID:
        return self._id

    @property
    def name(self) -> str:
        return self.config.name

    @abstractmethod
    async def generate_shot(self) -> Shot | None:
        """
        Generate a shot (theorem + proof).

        This is called when the agent is the Challenger.

        Returns:
            A Shot with:
            - theorem_name: Unique name for the theorem
            - theorem_statement: The Lean type signature
            - challenger_proof: The agent's proof (kept secret)

            Returns None if the agent fails to generate a valid shot.
        """
        pass

    @abstractmethod
    async def attempt_proof(self, shot: Shot) -> ProofAttempt | None:
        """
        Attempt to prove a theorem.

        This is called when the agent is the Defender.

        Args:
            shot: The Shot to prove (theorem, proof hidden)

        Returns:
            A ProofAttempt with the sequence of tactics used.
            Returns None if the agent fails to find a proof.
        """
        pass

    async def decide_bluff(self, shot: Shot) -> bool:
        """
        Decide whether to call a bluff on the challenger's shot.

        If the agent believes the shot is invalid (unprovable or
        the challenger's proof is wrong), they can call a bluff.

        Default implementation never calls bluff.

        Args:
            shot: The shot to evaluate

        Returns:
            True to call bluff, False to attempt the proof
        """
        return False

    def update_stats(
        self,
        proof_attempted: bool = False,
        proof_succeeded: bool = False,
        shot_generated: bool = False,
        tokens_used: int = 0,
    ) -> None:
        """Update agent statistics."""
        if proof_attempted:
            self.proofs_attempted += 1
        if proof_succeeded:
            self.proofs_succeeded += 1
        if shot_generated:
            self.shots_generated += 1
        self.total_tokens_used += tokens_used

    @property
    def success_rate(self) -> float:
        """Calculate proof success rate."""
        if self.proofs_attempted == 0:
            return 0.0
        return self.proofs_succeeded / self.proofs_attempted

    def to_dict(self) -> dict[str, Any]:
        """Serialize agent info to dict."""
        return {
            "id": str(self._id),
            "name": self.config.name,
            "model": self.config.model,
            "proofs_attempted": self.proofs_attempted,
            "proofs_succeeded": self.proofs_succeeded,
            "shots_generated": self.shots_generated,
            "success_rate": self.success_rate,
            "total_tokens_used": self.total_tokens_used,
        }


class RandomAgent(BaseAgent):
    """
    A simple random agent for testing.

    Generates trivial theorems and attempts basic tactics.
    """

    async def generate_shot(self) -> Shot | None:
        """Generate a simple theorem."""
        import random

        # Simple arithmetic theorems
        templates = [
            ("add_zero_{n}", "forall n : Nat, n + 0 = n", ["rfl"]),
            ("zero_add_{n}", "forall n : Nat, 0 + n = n", ["induction n", "rfl", "simp [*]"]),
            (
                "add_comm_{n}_{m}",
                "forall n m : Nat, n + m = m + n",
                ["intro n", "intro m", "omega"],
            ),
        ]

        template = random.choice(templates)
        n = random.randint(1, 100)

        name = template[0].format(n=n, m=random.randint(1, 100))
        statement = template[1]
        tactics = template[2]

        proof = ProofAttempt(tactics=tactics, is_complete=True, is_valid=True)

        shot = Shot(
            theorem_name=name,
            theorem_statement=statement,
            challenger_proof=proof,
        )

        self.update_stats(shot_generated=True)
        return shot

    async def attempt_proof(self, shot: Shot) -> ProofAttempt | None:
        """Try basic tactics."""
        self.update_stats(proof_attempted=True)

        # Try common tactics
        basic_tactics = [
            ["rfl"],
            ["trivial"],
            ["simp"],
            ["omega"],
            ["decide"],
            ["aesop"],
        ]

        # For now, just try one approach
        import random

        tactics = random.choice(basic_tactics)

        proof = ProofAttempt(
            tactics=tactics,
            total_steps=len(tactics),
        )

        return proof
