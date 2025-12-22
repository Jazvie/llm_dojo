"""
Core data models for the ANSPG H.O.R.S.E. game.

Defines the fundamental structures:
- Shot (Theorem + Proof)
- Game State
- Agent Actions
- Verification Results
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any
from uuid import UUID, uuid4


class GamePhase(Enum):
    """Current phase of the H.O.R.S.E. game."""

    WAITING = auto()  # Waiting for players
    CONJECTURE = auto()  # Challenger is setting a shot
    VERIFICATION = auto()  # Referee is validating the shot
    SOLVING = auto()  # Defender is attempting to match
    SCORING = auto()  # Determining outcome
    GAME_OVER = auto()  # H.O.R.S.E. spelled out


class TacticState(Enum):
    """State of a tactic execution."""

    SUCCESS = auto()
    ERROR = auto()
    TIMEOUT = auto()
    NO_GOALS = auto()  # Proof complete


class AgentRole(Enum):
    """Role of an agent in the current turn."""

    CHALLENGER = auto()  # Setting the shot
    DEFENDER = auto()  # Matching the shot
    SPECTATOR = auto()  # Observing only


@dataclass
class TacticResult:
    """Result of executing a single tactic."""

    state: TacticState
    goals_before: list[str]
    goals_after: list[str]
    error_message: str | None = None
    tactic_used: str = ""
    elapsed_ms: int = 0


@dataclass
class ProofAttempt:
    """
    A proof attempt by an agent.

    Maps to P in the Shot tuple.
    """

    id: UUID = field(default_factory=uuid4)
    tactics: list[str] = field(default_factory=list)
    tactic_results: list[TacticResult] = field(default_factory=list)
    is_complete: bool = False
    is_valid: bool = False
    total_steps: int = 0
    ast_depth: int = 0
    elapsed_ms: int = 0

    def add_tactic(self, tactic: str, result: TacticResult) -> None:
        self.tactics.append(tactic)
        self.tactic_results.append(result)
        self.total_steps += 1
        if result.state == TacticState.NO_GOALS:
            self.is_complete = True


@dataclass
class Shot:
    """
    A Shot in the H.O.R.S.E. game.

    The fundamental unit: S = (T, P)
    - T: The theorem statement
    - P: The proof certificate (hidden from defender)
    """

    id: UUID = field(default_factory=uuid4)

    # T: The Theorem
    theorem_name: str = ""
    theorem_statement: str = ""  # The Lean type signature
    theorem_full: str = ""  # Full Lean code including theorem keyword

    # P: The Proof (challenger's proof, kept secret during solving)
    challenger_proof: ProofAttempt | None = None

    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    difficulty_estimate: float = 0.5  # 0.0 = trivial, 1.0 = extremely hard
    source: str = "generated"  # "generated", "mathlib", "user"

    # Progress prediction (for filtering impossible shots)
    predicted_steps: float | None = None
    is_filtered_as_impossible: bool = False

    def is_valid(self) -> bool:
        """Check if the shot has a valid challenger proof."""
        return (
            self.challenger_proof is not None
            and self.challenger_proof.is_complete
            and self.challenger_proof.is_valid
        )

    def get_defender_view(self) -> "Shot":
        """
        Get the defender's view of this shot.

        This is the 'Statement Sanitizer' - the defender sees:
        - The theorem statement (type)
        - NOT the challenger's proof

        This ensures fair H.O.R.S.E. gameplay where the defender
        must independently prove the theorem.
        """
        return Shot(
            id=self.id,
            theorem_name=self.theorem_name,
            theorem_statement=self.theorem_statement,
            theorem_full=self.theorem_full,
            challenger_proof=None,  # Hidden from defender!
            created_at=self.created_at,
            difficulty_estimate=self.difficulty_estimate,
            source=self.source,
            predicted_steps=self.predicted_steps,
            is_filtered_as_impossible=self.is_filtered_as_impossible,
        )


@dataclass
class AgentState:
    """State of an agent in the game."""

    id: UUID = field(default_factory=uuid4)
    name: str = ""
    role: AgentRole = AgentRole.SPECTATOR
    letters: str = ""  # Accumulated letters (e.g., "HO" = 2 losses)
    is_eliminated: bool = False

    # Statistics
    shots_set: int = 0
    shots_matched: int = 0
    shots_missed: int = 0
    total_proof_time_ms: int = 0

    def add_letter(self) -> str:
        """Add a letter to the agent's score. Returns the new letter."""
        horse = "HORSE"
        if len(self.letters) < 5:
            new_letter = horse[len(self.letters)]
            self.letters += new_letter
            if self.letters == "HORSE":
                self.is_eliminated = True
            return new_letter
        return ""

    @property
    def letters_remaining(self) -> int:
        return 5 - len(self.letters)


@dataclass
class TurnResult:
    """Result of a single turn in the game."""

    shot: Shot
    defender_attempt: ProofAttempt | None = None
    defender_succeeded: bool = False
    letter_awarded_to: UUID | None = None
    bluff_called: bool = False
    bluff_successful: bool = False  # True if challenger's proof was invalid


@dataclass
class GameState:
    """
    Complete state of an H.O.R.S.E. game.

    Supports N-player rotation where:
    - Players are arranged in a circular order
    - The challenger proposes a theorem
    - All other players (defenders) try to match in rotation order
    - First defender to fail gets a letter
    - Ball passes according to game rules
    """

    id: UUID = field(default_factory=uuid4)

    # Players
    agents: list[AgentState] = field(default_factory=list)

    # Rotation order (indices into agents list)
    # This is the "line" of players - first is challenger, rest are defenders in order
    rotation_order: list[int] = field(default_factory=list)

    # For 2-player backwards compatibility
    current_challenger_idx: int = 0
    current_defender_idx: int = 1

    # Game progress
    phase: GamePhase = GamePhase.WAITING
    current_shot: Shot | None = None
    turn_history: list[TurnResult] = field(default_factory=list)
    turn_number: int = 0

    # Configuration
    time_limit_ms: int = 300_000  # 5 minutes per turn
    token_budget: int = 100_000

    # Timestamps
    created_at: datetime = field(default_factory=datetime.now)
    started_at: datetime | None = None
    ended_at: datetime | None = None

    def initialize_rotation(self, randomize: bool = False) -> None:
        """Initialize the rotation order.

        Args:
            randomize: If True, shuffle the order randomly
        """
        import random

        self.rotation_order = list(range(len(self.agents)))
        if randomize:
            random.shuffle(self.rotation_order)

        # Set initial challenger/defender for backwards compatibility
        if len(self.rotation_order) >= 1:
            self.current_challenger_idx = self.rotation_order[0]
        if len(self.rotation_order) >= 2:
            self.current_defender_idx = self.rotation_order[1]

        # Update roles
        self._update_roles()

    def _update_roles(self) -> None:
        """Update agent roles based on current rotation."""
        for i, agent in enumerate(self.agents):
            if i == self.current_challenger_idx:
                agent.role = AgentRole.CHALLENGER
            elif i == self.current_defender_idx:
                agent.role = AgentRole.DEFENDER
            else:
                agent.role = AgentRole.SPECTATOR

    def get_active_defenders(self) -> list[int]:
        """Get indices of all defenders (non-eliminated, non-challenger) in rotation order."""
        challenger_pos = self.rotation_order.index(self.current_challenger_idx)
        defenders = []
        n = len(self.rotation_order)

        for i in range(1, n):
            idx = self.rotation_order[(challenger_pos + i) % n]
            if not self.agents[idx].is_eliminated:
                defenders.append(idx)

        return defenders

    def advance_rotation(self) -> None:
        """Advance the rotation: next player in line becomes challenger."""
        if not self.rotation_order:
            return

        # Find current challenger position in rotation
        current_pos = self.rotation_order.index(self.current_challenger_idx)
        n = len(self.rotation_order)

        # Find next non-eliminated player
        for i in range(1, n + 1):
            next_pos = (current_pos + i) % n
            next_idx = self.rotation_order[next_pos]
            if not self.agents[next_idx].is_eliminated:
                self.current_challenger_idx = next_idx
                break

        # Set defender as next non-eliminated player after challenger
        challenger_pos = self.rotation_order.index(self.current_challenger_idx)
        for i in range(1, n + 1):
            next_pos = (challenger_pos + i) % n
            next_idx = self.rotation_order[next_pos]
            if not self.agents[next_idx].is_eliminated and next_idx != self.current_challenger_idx:
                self.current_defender_idx = next_idx
                break

        self._update_roles()

    @property
    def challenger(self) -> AgentState | None:
        if self.agents and 0 <= self.current_challenger_idx < len(self.agents):
            return self.agents[self.current_challenger_idx]
        return None

    @property
    def defender(self) -> AgentState | None:
        if self.agents and 0 <= self.current_defender_idx < len(self.agents):
            return self.agents[self.current_defender_idx]
        return None

    @property
    def active_player_count(self) -> int:
        """Number of non-eliminated players."""
        return sum(1 for agent in self.agents if not agent.is_eliminated)

    @property
    def is_game_over(self) -> bool:
        # Game over when only 1 player remains (or phase is set to GAME_OVER)
        return self.phase == GamePhase.GAME_OVER or self.active_player_count <= 1

    @property
    def winner(self) -> AgentState | None:
        if not self.is_game_over:
            return None
        for agent in self.agents:
            if not agent.is_eliminated:
                return agent
        return None

    def swap_roles(self) -> None:
        """Legacy 2-player role swap. For N-player, use advance_rotation()."""
        self.current_challenger_idx, self.current_defender_idx = (
            self.current_defender_idx,
            self.current_challenger_idx,
        )
        self._update_roles()


@dataclass
class LeanEnvironment:
    file_content: str = ""
    cursor_line: int = 0
    cursor_column: int = 0
    goals: list[str] = field(default_factory=list)
    hypotheses: list[dict[str, str]] = field(default_factory=list)
    diagnostics: list[dict[str, Any]] = field(default_factory=list)

    @property
    def is_proof_complete(self) -> bool:
        return len(self.goals) == 0 and len(self.diagnostics) == 0

    @property
    def current_goal(self) -> str | None:
        return self.goals[0] if self.goals else None


# Type aliases for clarity
TheoremStatement = str
TacticScript = str
ProofTerm = str


@dataclass
class DuplicateStatementTracker:
    """
    Tracks theorem statements across turns to prevent repetitive conjectures.

    This encourages variety in the game by preventing agents from:
    - Repeatedly proposing the same theorem statement across different turns
    - "Farming" a theorem that the opponent keeps failing on

    DUPLICATE POLICY:
        +-----------------------+-------------------+---------------------------+
        | Scenario              | Allowed?          | Rationale                 |
        +-----------------------+-------------------+---------------------------+
        | Same statement,       | NO (blocked)      | Prevents farming the same |
        | later turn            |                   | theorem against a         |
        |                       |                   | struggling opponent       |
        +-----------------------+-------------------+---------------------------+
        | Same statement,       | YES               | Allows retry with         |
        | same turn (retry)     |                   | different tactics if      |
        |                       |                   | proof failed              |
        +-----------------------+-------------------+---------------------------+
        | Same statement,       | YES               | Each agent independently  |
        | different agent       |                   | can propose theorems      |
        +-----------------------+-------------------+---------------------------+

    Key design decisions:
    - We track theorem STATEMENTS (normalized), not proof tactics
    - Statements are normalized (whitespace) before comparison
    - Tracking is per-agent: Agent A's statements don't block Agent B
    - Only successful shots are recorded (failed attempts can be retried)

    Usage:
        tracker = DuplicateStatementTracker()
        # Inject into both agents
        agent_a = HorseAgent(..., duplicate_tracker=tracker)
        agent_b = HorseAgent(..., duplicate_tracker=tracker)
    """

    # Map: agent_name -> set of statement fingerprints used across turns
    _used_statements: dict[str, set[str]] = field(default_factory=dict)

    # History for debugging/analysis
    _statement_history: list[tuple[str, str]] = field(default_factory=list)

    def _normalize(self, statement: str) -> str:
        """
        Normalize a theorem statement for comparison.

        We normalize to catch equivalent statements:
        - Strip whitespace
        - Normalize internal whitespace
        """
        # Normalize: strip, collapse whitespace
        return " ".join(statement.split())

    def is_duplicate(self, agent_name: str, statement: str) -> bool:
        """
        Check if this statement has been used by this agent before.
        """
        if agent_name not in self._used_statements:
            return False

        normalized = self._normalize(statement)
        return normalized in self._used_statements[agent_name]

    def record_statement(self, agent_name: str, statement: str) -> None:
        """
        Record a statement for duplicate tracking.

        Should be called after a successful shot is taken.
        """
        if agent_name not in self._used_statements:
            self._used_statements[agent_name] = set()

        normalized = self._normalize(statement)
        self._used_statements[agent_name].add(normalized)

        # Keep history for debugging
        self._statement_history.append((agent_name, statement))

    def get_agent_statement_count(self, agent_name: str) -> int:
        return len(self._used_statements.get(agent_name, set()))

    def clear(self) -> None:
        """Reset the tracker (for new games)."""
        self._used_statements.clear()
        self._statement_history.clear()
