"""Orchestrator module for ANSPG.

Public API uses the Lean REPL for deterministic proof verification:
- `REPLReferee`: Primary referee for dynamic theorem verification via Lean REPL.
- `REPLValidationResult`: Result type for REPL-based validation.

The REPL approach supports dynamic theorem generation, unlike LeanDojo which
requires theorems to exist in a pre-traced repository.

Legacy LeanDojo referee remains available for static repository analysis.
"""

from .repl_referee import REPLReferee, REPLValidationResult, create_repl_referee

# Legacy LeanDojo exports (optional, for static repos only)
# from .leandojo_referee import LeanDojoReferee, LeanDojoValidationResult, create_leandojo_referee

__all__ = [
    # Primary REPL-based referee
    "REPLReferee",
    "REPLValidationResult",
    "create_repl_referee",
]
