"""
Comprehensive configuration system for ANSPG.

Supports YAML config files with:
- Game settings (time limits, turns, rulebook)
- Agent configurations (per-agent hyperparameters)
- Logging settings
- Environment variable overrides

Config file search order:
1. Explicit --config path
2. ./anspg.yaml (current directory)
3. ~/.config/anspg/config.yaml (user config)
4. Defaults

Example anspg.yaml:
```yaml
game:
  time_limit_s: 300
  max_turns: 10
  rulebook: mathlib

agent_defaults:
  temperature: 0.3
  max_tokens: 500
  difficulty_target: 0.4
  max_conjecture_attempts: 5

agents:
  agent_a:
    name: Agent-A
    model: gpt-4o-mini
    temperature: 0.5  # Override default
  agent_b:
    name: Agent-B
    model: gpt-4o

logging:
  verbose: false
  show_failed_attempts: true
```
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:
    yaml = None  # type: ignore


class SimpPolicy(Enum):
    """
    Policy for handling `simp` and other automation tactics.

    Controls whether agents can "abuse" simp to propose trivial theorems.

    Options:
    - ALLOWED: No restrictions on simp (default, good for beginners)
    - NO_AUTO_SIMP: Theorems that can be solved by simp alone are rejected.
                    Forces challengers to propose non-trivial theorems.
                    simp can still be used as part of larger proofs.
    - BANNED: simp and related automation tactics are removed from the rulebook.
              Forces fully manual proofs (hardcore mode).
    """

    ALLOWED = "allowed"
    NO_AUTO_SIMP = "no_auto_simp"
    BANNED = "banned"

    @classmethod
    def from_string(cls, value: str | None) -> SimpPolicy:
        """Parse from string, defaulting to ALLOWED."""
        if value is None:
            return cls.ALLOWED
        value_lower = value.lower().replace("-", "_")
        for policy in cls:
            if policy.value == value_lower:
                return policy
        # Fallback to ALLOWED if unrecognized
        return cls.ALLOWED


@dataclass
class GameConfig:
    """Game-level configuration."""

    time_limit_s: int = 300
    max_turns: int = 10
    rulebook: str = "mathlib"
    simp_policy: SimpPolicy = SimpPolicy.ALLOWED


@dataclass
class AgentConfig:
    """Configuration for a single agent."""

    name: str = "Agent"
    model: str = "gpt-4o-mini"

    # LLM settings
    temperature: float = 0.3
    max_tokens: int = 500

    # Conjecture/proof settings
    difficulty_target: float = 0.4  # 0 = trivial, 1 = very hard
    max_conjecture_attempts: int = 3  # Retries for generating valid theorems

    @classmethod
    def from_dict(cls, data: dict[str, Any], defaults: AgentConfig | None = None) -> AgentConfig:
        """Create AgentConfig from dict, using defaults for missing values."""
        if defaults:
            # Start with defaults, override with provided values
            config_dict = {
                "name": data.get("name", defaults.name),
                "model": data.get("model", defaults.model),
                "temperature": data.get("temperature", defaults.temperature),
                "max_tokens": data.get("max_tokens", defaults.max_tokens),
                "difficulty_target": data.get("difficulty_target", defaults.difficulty_target),
                "max_conjecture_attempts": data.get(
                    "max_conjecture_attempts", defaults.max_conjecture_attempts
                ),
            }
        else:
            config_dict = data

        return cls(**{k: v for k, v in config_dict.items() if k in cls.__dataclass_fields__})


@dataclass
class LoggingConfig:
    """Logging configuration."""

    verbose: bool = False
    show_failed_attempts: bool = True
    show_proof_details: bool = False


@dataclass
class ANSPGConfig:
    """Complete ANSPG configuration."""

    game: GameConfig = field(default_factory=GameConfig)
    agent_defaults: AgentConfig = field(default_factory=AgentConfig)
    agents: dict[str, AgentConfig] = field(default_factory=dict)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    @classmethod
    def from_yaml(cls, path: Path) -> ANSPGConfig:
        """Load configuration from YAML file."""
        if yaml is None:
            raise ImportError(
                "PyYAML is required for config files. Install with: pip install pyyaml"
            )

        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path) as f:
            data = yaml.safe_load(f) or {}

        # Parse game config
        game_data = data.get("game", {})
        game = GameConfig(
            time_limit_s=game_data.get("time_limit_s", 300),
            max_turns=game_data.get("max_turns", 10),
            rulebook=game_data.get("rulebook", "mathlib"),
            simp_policy=SimpPolicy.from_string(game_data.get("simp_policy")),
        )

        # Parse agent defaults
        defaults_data = data.get("agent_defaults", {})
        agent_defaults = AgentConfig.from_dict(defaults_data) if defaults_data else AgentConfig()

        # Parse per-agent configs
        agents_data = data.get("agents", {})
        agents = {}
        for agent_id, agent_data in agents_data.items():
            agents[agent_id] = AgentConfig.from_dict(agent_data, defaults=agent_defaults)

        # Parse logging config
        logging_data = data.get("logging", {})
        logging = LoggingConfig(
            verbose=logging_data.get("verbose", False),
            show_failed_attempts=logging_data.get("show_failed_attempts", True),
            show_proof_details=logging_data.get("show_proof_details", False),
        )

        return cls(
            game=game,
            agent_defaults=agent_defaults,
            agents=agents,
            logging=logging,
        )

    def to_yaml(self, path: Path) -> None:
        """Save configuration to YAML file."""
        if yaml is None:
            raise ImportError("PyYAML is required. Install with: pip install pyyaml")

        data = {
            "game": {
                "time_limit_s": self.game.time_limit_s,
                "max_turns": self.game.max_turns,
                "rulebook": self.game.rulebook,
                "simp_policy": self.game.simp_policy.value,
            },
            "agent_defaults": {
                "temperature": self.agent_defaults.temperature,
                "max_tokens": self.agent_defaults.max_tokens,
                "difficulty_target": self.agent_defaults.difficulty_target,
                "max_conjecture_attempts": self.agent_defaults.max_conjecture_attempts,
            },
            "agents": {
                agent_id: {
                    "name": agent.name,
                    "model": agent.model,
                    "temperature": agent.temperature,
                    "max_tokens": agent.max_tokens,
                    "difficulty_target": agent.difficulty_target,
                    "max_conjecture_attempts": agent.max_conjecture_attempts,
                }
                for agent_id, agent in self.agents.items()
            },
            "logging": {
                "verbose": self.logging.verbose,
                "show_failed_attempts": self.logging.show_failed_attempts,
                "show_proof_details": self.logging.show_proof_details,
            },
        }

        with open(path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)

    def get_agent_config(self, agent_id: str) -> AgentConfig:
        """Get config for a specific agent, falling back to defaults.

        Always returns a COPY to avoid mutation issues when CLI flags override values.
        """
        from copy import deepcopy

        if agent_id in self.agents:
            return deepcopy(self.agents[agent_id])
        return deepcopy(self.agent_defaults)


def find_config_file() -> Path | None:
    """
    Find config file in standard locations.

    Search order:
    1. ./anspg.yaml (current directory)
    2. ~/.config/anspg/config.yaml (user config)
    3. None (use defaults)
    """
    # Current directory
    local_config = Path("anspg.yaml")
    if local_config.exists():
        return local_config

    # User config directory
    user_config_dir = Path.home() / ".config" / "anspg"
    user_config = user_config_dir / "config.yaml"
    if user_config.exists():
        return user_config

    return None


def load_config(path: Path | None = None) -> tuple[ANSPGConfig, Path | None]:
    """
    Load configuration from file or defaults.

    Args:
        path: Explicit config file path (optional)

    Returns:
        Tuple of (ANSPGConfig instance, path that was loaded or None if defaults)
    """
    if path is not None:
        return ANSPGConfig.from_yaml(path), path

    # Try to find config file
    found_path = find_config_file()
    if found_path:
        return ANSPGConfig.from_yaml(found_path), found_path

    # Use defaults
    return ANSPGConfig(), None


def create_default_config(path: Path) -> None:
    """Create a default config file at the given path."""
    config = ANSPGConfig()

    # Add example agents
    config.agents["agent_a"] = AgentConfig(
        name="Agent-A",
        model="gpt-4o-mini",
    )
    config.agents["agent_b"] = AgentConfig(
        name="Agent-B",
        model="gpt-4o-mini",
    )

    config.to_yaml(path)
