"""
Configuration system for ANSPG.

Config file search order:
1. Explicit --config path
2. ./anspg.yaml (current directory)
3. ~/.config/anspg/config.yaml (user config)
4. Defaults

See anspg.example.yaml for full configuration options.
"""

from __future__ import annotations

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
        """Parse from string, defaulting to ALLOWED if None."""
        if value is None:
            return cls.ALLOWED
        value_lower = value.lower().replace("-", "_")
        for policy in cls:
            if policy.value == value_lower:
                return policy
        valid_options = ", ".join(p.value for p in cls)
        raise ValueError(f"Invalid simp_policy '{value}'. Valid options: {valid_options}")


@dataclass
class GameConfig:
    """Game-level configuration."""

    time_limit_s: int = 300
    max_turns: int = 10
    rulebook: str = "mathlib"
    simp_policy: SimpPolicy = SimpPolicy.ALLOWED

    # Multi-agent H.O.R.S.E. options
    randomize_order: bool = False
    challenger_takes_letter_on_miss: bool = False


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
    agents: list[AgentConfig] = field(default_factory=list)
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
            randomize_order=game_data.get("randomize_order", False),
            challenger_takes_letter_on_miss=game_data.get("challenger_takes_letter_on_miss", False),
        )

        # Parse agent defaults
        defaults_data = data.get("agent_defaults", {})
        agent_defaults = AgentConfig.from_dict(defaults_data) if defaults_data else AgentConfig()

        # Parse agents list
        agents_data = data.get("agents", [])
        agents = []
        for i, agent_data in enumerate(agents_data):
            agent_config = AgentConfig.from_dict(agent_data, defaults=agent_defaults)
            if not agent_data.get("name"):
                agent_config.name = f"Agent{i + 1}"
            agents.append(agent_config)

        # Parse logging config
        logging_data = data.get("logging", {})
        logging_config = LoggingConfig(
            verbose=logging_data.get("verbose", False),
            show_failed_attempts=logging_data.get("show_failed_attempts", True),
            show_proof_details=logging_data.get("show_proof_details", False),
        )

        return cls(
            game=game,
            agent_defaults=agent_defaults,
            agents=agents,
            logging=logging_config,
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
                "randomize_order": self.game.randomize_order,
                "challenger_takes_letter_on_miss": self.game.challenger_takes_letter_on_miss,
            },
            "agent_defaults": {
                "temperature": self.agent_defaults.temperature,
                "max_tokens": self.agent_defaults.max_tokens,
                "difficulty_target": self.agent_defaults.difficulty_target,
                "max_conjecture_attempts": self.agent_defaults.max_conjecture_attempts,
            },
            "agents": [
                {
                    "name": agent.name,
                    "model": agent.model,
                    "temperature": agent.temperature,
                    "max_tokens": agent.max_tokens,
                    "difficulty_target": agent.difficulty_target,
                    "max_conjecture_attempts": agent.max_conjecture_attempts,
                }
                for agent in self.agents
            ],
            "logging": {
                "verbose": self.logging.verbose,
                "show_failed_attempts": self.logging.show_failed_attempts,
                "show_proof_details": self.logging.show_proof_details,
            },
        }

        with open(path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)

    def get_agent_configs(self) -> list[AgentConfig]:
        """Get all agent configs. Returns copies to avoid mutation issues."""
        from copy import deepcopy

        if len(self.agents) < 2:
            raise ValueError(
                "At least 2 agents must be configured. "
                "See anspg.example.yaml for configuration format."
            )
        return [deepcopy(agent) for agent in self.agents]


def find_config_file() -> Path | None:
    """Find config file in ./anspg.yaml or ~/.config/anspg/config.yaml."""
    local_config = Path("anspg.yaml")
    if local_config.exists():
        return local_config

    user_config = Path.home() / ".config" / "anspg" / "config.yaml"
    if user_config.exists():
        return user_config

    return None


def load_config(path: Path | None = None) -> tuple[ANSPGConfig, Path | None]:
    """Load configuration from file or use defaults."""
    if path is not None:
        return ANSPGConfig.from_yaml(path), path

    found_path = find_config_file()
    if found_path:
        return ANSPGConfig.from_yaml(found_path), found_path

    return ANSPGConfig(), None


def create_default_config(path: Path) -> None:
    """Create a default config file at the given path."""
    config = ANSPGConfig()
    config.agents = [
        AgentConfig(name="Alice", model="gpt-4o-mini"),
        AgentConfig(name="Bob", model="gpt-4o-mini"),
    ]
    config.to_yaml(path)
