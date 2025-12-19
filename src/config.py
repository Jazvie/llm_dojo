"""
Configuration for ANSPG.

Environment variables:
- ANSPG_LLM_API_KEY or OPENAI_API_KEY: API key for LLM
- ANSPG_LLM_MODEL: Model to use (default: gpt-4o-mini)
- ANSPG_LLM_BASE_URL: Custom base URL for OpenAI-compatible API
- ANSPG_LEAN_REPO: Path to traced Lean repository
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # python-dotenv not available, use environment variables directly
    pass

from .agents.llm_client import LLMClient, create_llm_client


@dataclass(frozen=True)
class Settings:
    """ANSPG configuration settings."""

    llm_base_url: Optional[str] = None
    llm_model: Optional[str] = None
    llm_api_key: Optional[str] = None
    lean_repo_path: Optional[str] = None

    @property
    def llm_enabled(self) -> bool:
        return bool(self.llm_api_key)

    @property
    def lean_repo(self) -> Path | None:
        if self.lean_repo_path:
            return Path(self.lean_repo_path)
        return None


@lru_cache(maxsize=1)
def load_settings() -> Settings:
    """Load settings from environment variables."""
    return Settings(
        llm_base_url=os.getenv("ANSPG_LLM_BASE_URL"),
        llm_model=os.getenv("ANSPG_LLM_MODEL", "gpt-4o-mini"),
        llm_api_key=os.getenv("ANSPG_LLM_API_KEY") or os.getenv("OPENAI_API_KEY"),
        lean_repo_path=os.getenv("ANSPG_LEAN_REPO"),
    )


def create_llm_client_from_settings(settings: Settings) -> LLMClient | None:
    """Create an LLM client from settings."""
    if not settings.llm_enabled:
        return None
    return create_llm_client(
        api_key=settings.llm_api_key,
        base_url=settings.llm_base_url,
        model=settings.llm_model,
    )
