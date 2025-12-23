"""Orchestrator module for ANSPG.

Public API uses the Lean REPL for deterministic proof verification:
- `REPLReferee`: Primary referee for dynamic theorem verification via Lean REPL.
- `REPLValidationResult`: Result type for REPL-based validation.
"""

from .repl_referee import REPLReferee, REPLValidationResult, create_repl_referee

__all__ = [
    "REPLReferee",
    "REPLValidationResult",
    "create_repl_referee",
]
