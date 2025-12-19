"""
State management for Lean REPL with O(1) backtracking.

Implements the SOTA architecture for proof search:
- Environment handles for instant state restoration
- Proof tree representation for MCTS/BFS
- Immutable snapshots (no mutable reference bugs)

The key insight is that the REPL returns a handle (integer ID) for each
environment state. By storing these handles, we can backtrack to any
previous state without re-executing tactics.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Iterator
from uuid import UUID, uuid4


class NodeStatus(Enum):
    """Status of a proof tree node."""

    PENDING = auto()  # Not yet explored
    EXPLORING = auto()  # Currently being explored
    PROVED = auto()  # Goals discharged
    FAILED = auto()  # Dead end (error or timeout)
    PRUNED = auto()  # Abandoned (e.g., hallucination detected)


@dataclass
class StateHandle:
    """
    A handle to a Lean environment state.

    This is the key abstraction for O(1) backtracking. The REPL
    maintains internal state indexed by these handles, allowing
    us to "jump" to any previous point in the proof.

    Attributes:
        id: Unique identifier for this handle
        env_id: The environment ID returned by the REPL
        parent_id: ID of the parent state (for tree navigation)
        goals: Goal strings at this state
        tactic_applied: The tactic that led to this state
        depth: Depth in the proof tree
        created_at: Timestamp for debugging/metrics
    """

    id: UUID = field(default_factory=uuid4)
    env_id: int = 0  # REPL environment handle
    parent_id: UUID | None = None

    # Proof state
    goals: list[str] = field(default_factory=list)
    hypotheses: list[dict[str, str]] = field(default_factory=list)

    # Transition info
    tactic_applied: str | None = None
    tactic_time_ms: float = 0.0

    # Tree position
    depth: int = 0

    # Metadata
    created_at: float = field(default_factory=time.time)

    @property
    def is_solved(self) -> bool:
        """Whether all goals are discharged."""
        return len(self.goals) == 0

    @property
    def num_goals(self) -> int:
        return len(self.goals)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": str(self.id),
            "env_id": self.env_id,
            "parent_id": str(self.parent_id) if self.parent_id else None,
            "goals": self.goals,
            "hypotheses": self.hypotheses,
            "tactic_applied": self.tactic_applied,
            "depth": self.depth,
            "is_solved": self.is_solved,
        }


@dataclass
class ProofNode:
    """
    A node in the proof search tree.

    Extends StateHandle with search-specific metadata for
    algorithms like MCTS and Best-First Search.
    """

    state: StateHandle
    status: NodeStatus = NodeStatus.PENDING

    # Search metadata
    visit_count: int = 0
    value_estimate: float = 0.0  # For MCTS UCB
    prior_probability: float = 1.0  # From policy network

    # Progress prediction (dense reward)
    estimated_steps_remaining: int | None = None

    # Children (tactics tried from this state)
    children: dict[str, "ProofNode"] = field(default_factory=dict)

    # Error info if failed
    error_message: str | None = None

    @property
    def id(self) -> UUID:
        return self.state.id

    @property
    def is_leaf(self) -> bool:
        return len(self.children) == 0

    @property
    def is_terminal(self) -> bool:
        return self.status in {NodeStatus.PROVED, NodeStatus.FAILED, NodeStatus.PRUNED}

    def ucb_score(self, exploration_constant: float = 1.41) -> float:
        """
        Upper Confidence Bound score for MCTS.

        UCB = Q(s,a) + c * P(s,a) * sqrt(N(s)) / (1 + N(s,a))

        where:
        - Q(s,a) = average value of this node
        - P(s,a) = prior probability from policy network
        - N(s) = parent visit count
        - N(s,a) = this node's visit count
        """
        import math

        if self.visit_count == 0:
            return float("inf")  # Prioritize unvisited

        # Exploitation term
        q_value = self.value_estimate / self.visit_count

        # Exploration term (assumes parent visit count available)
        # Simplified: use prior * sqrt(log(total) / visits)
        exploration = (
            exploration_constant
            * self.prior_probability
            * math.sqrt(math.log(self.visit_count + 1) / self.visit_count)
        )

        return q_value + exploration

    def add_child(self, tactic: str, child_state: StateHandle) -> "ProofNode":
        """Add a child node resulting from applying a tactic."""
        child = ProofNode(state=child_state, status=NodeStatus.PENDING)
        self.children[tactic] = child
        return child

    def get_path_to_root(self) -> list[str]:
        """Get the tactic sequence from root to this node."""
        # This requires tree traversal - implemented in ProofTree
        raise NotImplementedError("Use ProofTree.get_path_to_node()")


class ProofTree:
    """
    A proof search tree with efficient state management.

    This is the central data structure for tree search algorithms.
    It maintains:
    - A root node (initial proof state)
    - All explored nodes indexed by ID
    - Methods for navigation and backtracking
    """

    def __init__(self, root_state: StateHandle):
        self.root = ProofNode(state=root_state, status=NodeStatus.EXPLORING)
        self._nodes: dict[UUID, ProofNode] = {root_state.id: self.root}
        self._state_to_node: dict[int, ProofNode] = {root_state.env_id: self.root}

    def add_node(self, parent: ProofNode, tactic: str, state: StateHandle) -> ProofNode:
        """Add a new node to the tree."""
        node = parent.add_child(tactic, state)
        self._nodes[state.id] = node
        self._state_to_node[state.env_id] = node
        return node

    def get_node(self, node_id: UUID) -> ProofNode | None:
        """Get a node by its ID."""
        return self._nodes.get(node_id)

    def get_node_by_env(self, env_id: int) -> ProofNode | None:
        """Get a node by its REPL environment ID."""
        return self._state_to_node.get(env_id)

    def get_path_to_node(self, node: ProofNode) -> list[str]:
        """Get the tactic sequence from root to a node."""
        path = []
        current = node

        while current.state.parent_id is not None:
            if current.state.tactic_applied:
                path.append(current.state.tactic_applied)
            parent = self._nodes.get(current.state.parent_id)
            if parent is None:
                break
            current = parent

        path.reverse()
        return path

    def get_leaves(self) -> Iterator[ProofNode]:
        """Iterate over all leaf nodes."""
        for node in self._nodes.values():
            if node.is_leaf and not node.is_terminal:
                yield node

    def get_proved_paths(self) -> list[list[str]]:
        """Get all successful proof paths."""
        paths = []
        for node in self._nodes.values():
            if node.status == NodeStatus.PROVED:
                paths.append(self.get_path_to_node(node))
        return paths

    @property
    def is_solved(self) -> bool:
        """Whether any path reaches a proved state."""
        return any(n.status == NodeStatus.PROVED for n in self._nodes.values())

    @property
    def num_nodes(self) -> int:
        return len(self._nodes)

    @property
    def max_depth(self) -> int:
        return max(n.state.depth for n in self._nodes.values())

    def statistics(self) -> dict[str, Any]:
        """Get tree statistics for debugging/metrics."""
        status_counts = {s: 0 for s in NodeStatus}
        for node in self._nodes.values():
            status_counts[node.status] += 1

        return {
            "num_nodes": self.num_nodes,
            "max_depth": self.max_depth,
            "is_solved": self.is_solved,
            "status_counts": {s.name: c for s, c in status_counts.items()},
        }


class StateManager:
    """
    High-level state manager for proof search.

    Provides:
    - State handle allocation and tracking
    - Backtracking via handle restoration
    - Integration with the REPL client

    This is the "glue" between the ProofTree and the REPL.
    """

    def __init__(self):
        self._handles: dict[UUID, StateHandle] = {}
        self._env_to_handle: dict[int, StateHandle] = {}
        self._current: StateHandle | None = None
        self._trees: dict[str, ProofTree] = {}  # theorem_name -> tree

    def create_handle(
        self,
        env_id: int,
        goals: list[str],
        hypotheses: list[dict[str, str]] | None = None,
        parent: StateHandle | None = None,
        tactic: str | None = None,
        tactic_time_ms: float = 0.0,
    ) -> StateHandle:
        """Create a new state handle."""
        handle = StateHandle(
            env_id=env_id,
            parent_id=parent.id if parent else None,
            goals=goals,
            hypotheses=hypotheses or [],
            tactic_applied=tactic,
            tactic_time_ms=tactic_time_ms,
            depth=(parent.depth + 1) if parent else 0,
        )

        self._handles[handle.id] = handle
        self._env_to_handle[env_id] = handle
        self._current = handle

        return handle

    def get_handle(self, handle_id: UUID) -> StateHandle | None:
        """Get a handle by ID."""
        return self._handles.get(handle_id)

    def get_handle_by_env(self, env_id: int) -> StateHandle | None:
        """Get a handle by REPL environment ID."""
        return self._env_to_handle.get(env_id)

    @property
    def current(self) -> StateHandle | None:
        """The current state handle."""
        return self._current

    def set_current(self, handle: StateHandle) -> None:
        """Set the current state (for backtracking)."""
        self._current = handle

    def start_proof(self, theorem_name: str, initial_state: StateHandle) -> ProofTree:
        """Start a new proof tree for a theorem."""
        tree = ProofTree(initial_state)
        self._trees[theorem_name] = tree
        return tree

    def get_tree(self, theorem_name: str) -> ProofTree | None:
        """Get the proof tree for a theorem."""
        return self._trees.get(theorem_name)

    def clear(self) -> None:
        """Clear all state (for cleanup)."""
        self._handles.clear()
        self._env_to_handle.clear()
        self._trees.clear()
        self._current = None
