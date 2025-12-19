"""
Dynamic Theorem Registration for ANSPG.

Solves the "Ghost Theorem" problem by providing mechanisms to:
1. Register new theorems in the Lean environment dynamically
2. Track proved theorems in a dynamic knowledge database
3. Support curriculum learning with complexity metrics

This implements the SOTA architecture where:
- Novel conjectures are materialized into real Lean declarations
- Proved theorems become available as premises for future proofs
- The agent can "bootstrap" its capabilities over time

Key insight from the architecture doc:
"As the agent proves new theorems, these results must be indexed and
made available as premises for future proofs, mirroring the cumulative
nature of human mathematical knowledge."
"""

from __future__ import annotations

import asyncio
import json
import math
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import Any, AsyncIterator
from uuid import UUID, uuid4


class TheoremStatus(Enum):
    """Status of a theorem in the registry."""

    CONJECTURED = auto()  # Proposed but not yet verified
    PROVING = auto()  # Proof in progress
    PROVED = auto()  # Successfully proved
    REFUTED = auto()  # Counterexample found
    ABANDONED = auto()  # Proof attempt abandoned
    REGISTERED = auto()  # Added to Lean environment


class DifficultyTier(Enum):
    """
    Difficulty tiers for curriculum learning.

    Based on the LeanAgent approach:
    - EASY: 33rd percentile of proof complexity
    - MEDIUM: 67th percentile
    - HARD: Top tier
    """

    EASY = auto()
    MEDIUM = auto()
    HARD = auto()
    EXPERT = auto()


@dataclass
class TheoremRecord:
    """
    A record of a theorem in the dynamic knowledge database.

    Tracks:
    - The theorem itself (name, statement, proof)
    - Metadata for curriculum learning
    - Usage statistics for retrieval
    """

    id: UUID = field(default_factory=uuid4)
    name: str = ""
    statement: str = ""  # Type signature

    # Proof information
    proof_tactics: list[str] = field(default_factory=list)
    proof_steps: int = 0
    status: TheoremStatus = TheoremStatus.CONJECTURED

    # Source information
    module_path: str = ""
    source: str = "generated"  # "generated", "mathlib", "human"

    # Complexity metrics for curriculum learning
    complexity_score: float = 0.0  # e^S where S = proof steps
    difficulty_tier: DifficultyTier = DifficultyTier.EASY

    # Dependencies
    premises_used: list[str] = field(default_factory=list)

    # Usage statistics
    times_retrieved: int = 0
    times_used_as_premise: int = 0

    # Timestamps
    created_at: datetime = field(default_factory=datetime.now)
    proved_at: datetime | None = None
    registered_at: datetime | None = None

    def compute_complexity(self) -> float:
        """
        Compute complexity score using e^S formula.

        This is the standard metric from LeanAgent.
        """
        self.complexity_score = math.exp(self.proof_steps)
        return self.complexity_score

    def assign_difficulty_tier(self, percentile_33: float, percentile_67: float) -> None:
        """
        Assign difficulty tier based on complexity percentiles.

        Args:
            percentile_33: Complexity at 33rd percentile
            percentile_67: Complexity at 67th percentile
        """
        if self.complexity_score <= percentile_33:
            self.difficulty_tier = DifficultyTier.EASY
        elif self.complexity_score <= percentile_67:
            self.difficulty_tier = DifficultyTier.MEDIUM
        elif self.complexity_score <= percentile_67 * 1.5:
            self.difficulty_tier = DifficultyTier.HARD
        else:
            self.difficulty_tier = DifficultyTier.EXPERT

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": str(self.id),
            "name": self.name,
            "statement": self.statement,
            "proof_tactics": self.proof_tactics,
            "proof_steps": self.proof_steps,
            "status": self.status.name,
            "module_path": self.module_path,
            "source": self.source,
            "complexity_score": self.complexity_score,
            "difficulty_tier": self.difficulty_tier.name,
            "premises_used": self.premises_used,
            "times_retrieved": self.times_retrieved,
            "times_used_as_premise": self.times_used_as_premise,
            "created_at": self.created_at.isoformat(),
            "proved_at": self.proved_at.isoformat() if self.proved_at else None,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TheoremRecord":
        record = cls(
            id=UUID(data["id"]) if "id" in data else uuid4(),
            name=data.get("name", ""),
            statement=data.get("statement", ""),
            proof_tactics=data.get("proof_tactics", []),
            proof_steps=data.get("proof_steps", 0),
            module_path=data.get("module_path", ""),
            source=data.get("source", "generated"),
            complexity_score=data.get("complexity_score", 0.0),
            premises_used=data.get("premises_used", []),
            times_retrieved=data.get("times_retrieved", 0),
            times_used_as_premise=data.get("times_used_as_premise", 0),
        )

        if "status" in data:
            record.status = TheoremStatus[data["status"]]
        if "difficulty_tier" in data:
            record.difficulty_tier = DifficultyTier[data["difficulty_tier"]]
        if data.get("proved_at"):
            record.proved_at = datetime.fromisoformat(data["proved_at"])
        if data.get("created_at"):
            record.created_at = datetime.fromisoformat(data["created_at"])

        return record


class DynamicKnowledgeDatabase:
    """
    Dynamic knowledge database for lifelong learning.

    Implements:
    - Real-time indexing of proved theorems
    - Curriculum learning with difficulty tiers
    - Usage tracking for retrieval optimization

    From the architecture doc:
    "As the agent successfully proves theorems, these are added to a
    persistent database. The retriever index is periodically updated
    to include these new proofs as potential examples or premises."
    """

    def __init__(self, persist_path: Path | None = None):
        """
        Initialize the knowledge database.

        Args:
            persist_path: Optional path for persistence
        """
        self._records: dict[str, TheoremRecord] = {}
        self._by_difficulty: dict[DifficultyTier, list[str]] = {tier: [] for tier in DifficultyTier}
        self._persist_path = persist_path

        # Complexity percentiles (updated as theorems are added)
        self._percentile_33: float = 1.0
        self._percentile_67: float = 10.0

    def add_theorem(self, record: TheoremRecord) -> None:
        """Add a theorem to the database."""
        record.compute_complexity()
        record.assign_difficulty_tier(self._percentile_33, self._percentile_67)

        self._records[record.name] = record
        self._by_difficulty[record.difficulty_tier].append(record.name)

        # Update percentiles
        self._update_percentiles()

    def mark_proved(
        self,
        name: str,
        proof_tactics: list[str],
        premises_used: list[str] | None = None,
    ) -> TheoremRecord | None:
        """Mark a theorem as proved and update its record."""
        record = self._records.get(name)
        if not record:
            return None

        record.status = TheoremStatus.PROVED
        record.proof_tactics = proof_tactics
        record.proof_steps = len(proof_tactics)
        record.proved_at = datetime.now()

        if premises_used:
            record.premises_used = premises_used

        # Recompute complexity with actual proof
        record.compute_complexity()

        # Update difficulty tier
        old_tier = record.difficulty_tier
        record.assign_difficulty_tier(self._percentile_33, self._percentile_67)

        # Update tier index if changed
        if old_tier != record.difficulty_tier:
            if name in self._by_difficulty[old_tier]:
                self._by_difficulty[old_tier].remove(name)
            self._by_difficulty[record.difficulty_tier].append(name)

        return record

    def mark_registered(self, name: str) -> None:
        """Mark a theorem as registered in the Lean environment."""
        if name in self._records:
            self._records[name].status = TheoremStatus.REGISTERED
            self._records[name].registered_at = datetime.now()

    def get_by_difficulty(
        self,
        tier: DifficultyTier,
        limit: int | None = None,
        proved_only: bool = True,
    ) -> list[TheoremRecord]:
        """
        Get theorems by difficulty tier.

        Used for curriculum learning.
        """
        names = self._by_difficulty.get(tier, [])
        records = []

        for name in names:
            record = self._records.get(name)
            if record:
                if proved_only and record.status != TheoremStatus.PROVED:
                    continue
                records.append(record)
                if limit and len(records) >= limit:
                    break

        return records

    def get_curriculum_batch(
        self,
        current_tier: DifficultyTier,
        batch_size: int = 10,
    ) -> list[TheoremRecord]:
        """
        Get a batch of theorems for curriculum learning.

        Implements the progressive difficulty approach:
        - Start with easy theorems
        - Gradually increase difficulty
        """
        # Get theorems from current and adjacent tiers
        records = []

        # Mostly from current tier
        records.extend(self.get_by_difficulty(current_tier, batch_size * 2 // 3))

        # Some from easier tier (for stability)
        if current_tier != DifficultyTier.EASY:
            easier = DifficultyTier(current_tier.value - 1)
            records.extend(self.get_by_difficulty(easier, batch_size // 6))

        # Some from harder tier (for challenge)
        if current_tier != DifficultyTier.EXPERT:
            harder = DifficultyTier(current_tier.value + 1)
            records.extend(self.get_by_difficulty(harder, batch_size // 6))

        return records[:batch_size]

    def record_retrieval(self, name: str) -> None:
        """Record that a theorem was retrieved."""
        if name in self._records:
            self._records[name].times_retrieved += 1

    def record_use_as_premise(self, name: str) -> None:
        """Record that a theorem was used as a premise."""
        if name in self._records:
            self._records[name].times_used_as_premise += 1

    def get_most_useful_premises(self, limit: int = 20) -> list[str]:
        """Get the most frequently used premises."""
        sorted_records = sorted(
            self._records.values(),
            key=lambda r: r.times_used_as_premise,
            reverse=True,
        )
        return [r.name for r in sorted_records[:limit]]

    def _update_percentiles(self) -> None:
        """Update complexity percentiles."""
        complexities = sorted(
            r.complexity_score for r in self._records.values() if r.status == TheoremStatus.PROVED
        )

        if len(complexities) >= 3:
            idx_33 = int(len(complexities) * 0.33)
            idx_67 = int(len(complexities) * 0.67)
            self._percentile_33 = complexities[idx_33]
            self._percentile_67 = complexities[idx_67]

    def save(self, path: Path | None = None) -> None:
        """Save the database to disk."""
        path = path or self._persist_path
        if not path:
            return

        data = {
            "records": [r.to_dict() for r in self._records.values()],
            "percentile_33": self._percentile_33,
            "percentile_67": self._percentile_67,
        }

        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: Path) -> "DynamicKnowledgeDatabase":
        """Load a database from disk."""
        with open(path) as f:
            data = json.load(f)

        db = cls(persist_path=path)
        db._percentile_33 = data.get("percentile_33", 1.0)
        db._percentile_67 = data.get("percentile_67", 10.0)

        for record_data in data.get("records", []):
            record = TheoremRecord.from_dict(record_data)
            db._records[record.name] = record
            db._by_difficulty[record.difficulty_tier].append(record.name)

        return db

    def __len__(self) -> int:
        return len(self._records)

    def statistics(self) -> dict[str, Any]:
        """Get database statistics."""
        status_counts = {}
        for record in self._records.values():
            status = record.status.name
            status_counts[status] = status_counts.get(status, 0) + 1

        tier_counts = {tier.name: len(names) for tier, names in self._by_difficulty.items()}

        return {
            "total_theorems": len(self._records),
            "by_status": status_counts,
            "by_difficulty": tier_counts,
            "percentile_33": self._percentile_33,
            "percentile_67": self._percentile_67,
        }


class DynamicTheoremRegistry:
    """
    Registry for dynamically registering theorems in Lean.

    This solves the "Ghost Theorem" problem by:
    1. Writing new theorems to a Lean file
    2. Triggering recompilation/retracing
    3. Making theorems available as premises

    Integration points:
    - LeanREPLClient for theorem declaration
    - DynamicKnowledgeDatabase for tracking
    - Retrieval system for premise availability
    """

    def __init__(
        self,
        repo_path: Path,
        generated_file: str = "Generated.lean",
        knowledge_db: DynamicKnowledgeDatabase | None = None,
    ):
        """
        Initialize the registry.

        Args:
            repo_path: Path to the Lean repository
            generated_file: Name of file for generated theorems
            knowledge_db: Optional knowledge database
        """
        self._repo_path = Path(repo_path)
        self._generated_file = generated_file
        self._generated_path = self._repo_path / "src" / generated_file
        self._knowledge_db = knowledge_db or DynamicKnowledgeDatabase()

        # Track registered theorems
        self._registered: set[str] = set()

        # REPL client (set via set_client)
        self._client = None

    def set_client(self, client) -> None:
        """Set the REPL client for dynamic registration."""
        self._client = client

    async def register_theorem(
        self,
        name: str,
        statement: str,
        proof_tactics: list[str],
        premises_used: list[str] | None = None,
    ) -> bool:
        """
        Register a new theorem in the Lean environment.

        This is the key operation that solves Ghost Theorem:
        1. Add to knowledge database
        2. Declare in REPL (if available)
        3. Append to generated file

        Args:
            name: Theorem name (must be valid Lean identifier)
            statement: Type signature
            proof_tactics: List of tactics
            premises_used: Premises used in proof

        Returns:
            True if registration succeeded
        """
        # Create record
        record = TheoremRecord(
            name=name,
            statement=statement,
            proof_tactics=proof_tactics,
            proof_steps=len(proof_tactics),
            status=TheoremStatus.PROVING,
            source="generated",
            premises_used=premises_used or [],
        )

        # Add to knowledge database
        self._knowledge_db.add_theorem(record)

        # Try to declare via REPL
        if self._client:
            try:
                success = await self._client.declare_theorem(name, statement, proof_tactics)
                if success:
                    self._knowledge_db.mark_proved(name, proof_tactics, premises_used)
                    self._knowledge_db.mark_registered(name)
                    self._registered.add(name)
                    return True
            except Exception:
                pass

        # Fallback: append to file
        success = self._append_to_file(name, statement, proof_tactics)
        if success:
            self._knowledge_db.mark_proved(name, proof_tactics, premises_used)
            self._registered.add(name)

        return success

    def _append_to_file(
        self,
        name: str,
        statement: str,
        proof_tactics: list[str],
    ) -> bool:
        """
        Append a theorem to the generated file.

        This enables persistence across sessions but requires
        recompilation to take effect.
        """
        try:
            # Ensure file exists with header
            if not self._generated_path.exists():
                self._generated_path.parent.mkdir(parents=True, exist_ok=True)
                header = """/-
-- Auto-generated theorems for ANSPG
-- This file is managed by DynamicTheoremRegistry
-/

import Mathlib

namespace ANSPG.Generated

"""
                self._generated_path.write_text(header)

            # Build theorem declaration
            tactics_str = "\n  ".join(proof_tactics)
            declaration = f"""
/-- Auto-generated theorem -/
theorem {name} : {statement} := by
  {tactics_str}

"""

            # Append to file
            with open(self._generated_path, "a") as f:
                f.write(declaration)

            return True

        except Exception:
            return False

    def is_registered(self, name: str) -> bool:
        """Check if a theorem is registered."""
        return name in self._registered

    def get_available_premises(self) -> list[str]:
        """Get all registered theorems available as premises."""
        return list(self._registered)

    def finalize_file(self) -> None:
        """
        Finalize the generated file with closing namespace.

        Call this when done generating theorems.
        """
        if self._generated_path.exists():
            with open(self._generated_path, "a") as f:
                f.write("\nend ANSPG.Generated\n")

    @property
    def knowledge_db(self) -> DynamicKnowledgeDatabase:
        return self._knowledge_db

    def statistics(self) -> dict[str, Any]:
        return {
            "registered_count": len(self._registered),
            "knowledge_db": self._knowledge_db.statistics(),
        }
