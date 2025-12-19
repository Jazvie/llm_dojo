"""
Lean 4 REPL Integration Layer for ANSPG.

This module provides a persistent, state-aware interface to Lean 4 via the
leanprover-community/repl tool. It implements the SOTA architecture for
AI-driven theorem proving:

- Persistent process (no restart per tactic)
- State handles for O(1) backtracking
- Structured error parsing
- AST-level data extraction

See: https://github.com/leanprover-community/repl
"""

# Import submodules - these are always available
from .errors import (
    LeanError,
    LeanErrorCategory,
    parse_lean_error,
    StructuredError,
    categorize_for_search,
)
from .state_manager import (
    StateManager,
    StateHandle,
    ProofTree,
    ProofNode,
    NodeStatus,
)
from .client import (
    LeanREPLClient,
    REPLState,
    REPLResponse,
    REPLStatus,
    TacticResult,
    ProofState,
    create_repl_client,
)

__all__ = [
    # Client
    "LeanREPLClient",
    "REPLState",
    "REPLStatus",
    "REPLResponse",
    "TacticResult",
    "ProofState",
    "create_repl_client",
    # Errors
    "LeanError",
    "LeanErrorCategory",
    "parse_lean_error",
    "StructuredError",
    "categorize_for_search",
    # State management
    "StateManager",
    "StateHandle",
    "ProofTree",
    "ProofNode",
    "NodeStatus",
]
