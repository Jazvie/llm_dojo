"""
Agent implementations for ANSPG H.O.R.S.E.

Main classes:
- HorseAgent: Unified agent that can take_shot() and match_shot()
- GroundedProver: Interactive proof search with REPL validation
- LLMClient: OpenAI-compatible LLM interface

The HorseAgent is the primary class for playing H.O.R.S.E. games.
It uses the same proving machinery for both proposing and defending,
ensuring fair play.
"""

from .base import BaseAgent, AgentConfig
from .horse import HorseAgent, HorseAgentConfig, AgentStats, ShotResult
from .grounded import GroundedProver, GroundedConjecturer, ProofSearchConfig, ProofSearchResult
from .llm_client import LLMClient, create_llm_client

__all__ = [
    # Base
    "BaseAgent",
    "AgentConfig",
    # Main agent
    "HorseAgent",
    "HorseAgentConfig",
    "AgentStats",
    "ShotResult",
    # Proof search (still used by GroundedAgent)
    "GroundedProver",
    "GroundedConjecturer",
    "ProofSearchConfig",
    "ProofSearchResult",
    # LLM
    "LLMClient",
    "create_llm_client",
]
