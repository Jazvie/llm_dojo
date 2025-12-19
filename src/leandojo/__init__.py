"""
LeanDojo integration for ANSPG (LEGACY/OPTIONAL).

NOTE: This module is kept for compatibility with static repository analysis.
For dynamic theorem verification (the primary use case), use the REPL-based
referee in src/orchestrator/repl_referee.py instead.

LeanDojo only works with theorems that exist in a pre-traced repository.
It cannot verify dynamically generated theorems, which is the core requirement
for the H.O.R.S.E. game.

If you need LeanDojo, install it separately:
    pip install lean-dojo
"""

from .client import (
    LeanDojoClient,
    LeanDojoProver,
    TracedTacticState,
    TracedPremise,
    TacticApplicationResult,
    create_leandojo_client,
)

__all__ = [
    "LeanDojoClient",
    "LeanDojoProver",
    "TracedTacticState",
    "TracedPremise",
    "TacticApplicationResult",
    "create_leandojo_client",
]
