<img src="img/the_horse_turned_groom.jpg" style="display: block; margin: 30 auto;" />


# ANSPG - Adversarial Neuro-Symbolic Proving Ground

A H.O.R.S.E.-style theorem proving game where AI agents compete by proposing and solving Lean 4 theorems, with deterministic verification via the Lean REPL. Supports 2 or more players.

## Overview

ANSPG measures LLM capability at formal mathematics by having models:
1. Generate theorems with proofs (as challenger)
2. Solve theorems independently (as defender)
3. All proofs verified deterministically through the Lean REPL

This is a CLI benchmark for comparing LLM math reasoning via Lean-grounded verification.

## Game Rules

Agents take turns in rotation as Challenger and Defenders:

1. **Challenger** proposes a theorem with a proof
2. **System** verifies the challenger's proof via Lean REPL
3. **Defenders** (all other players, in rotation order) see only the theorem statement and must prove it independently
4. **Scoring**:
   - If challenger's proof is invalid → ball passes to next player (traditional HORSE rules)
   - If a defender fails to prove → that defender gets a letter, challenger shoots again
   - If all defenders succeed → ball passes to next player
5. First agent to spell **H-O-R-S-E** is eliminated
6. Last agent standing wins

### Multi-Player Flow

In a game with 3+ players, the rotation works like real HORSE:
- Player 1 (challenger) makes a shot
- Players 2, 3, ... (defenders) each try to match in order
- First defender to miss gets a letter; challenger keeps the ball
- If all defenders match, ball passes to Player 2 (who becomes challenger)

### Key Design: Statement Sanitizer

The defender receives a "sanitized" view of the shot—they see the theorem statement, but **never** the challenger's proof. This ensures fair competition where both agents must independently solve the same problem.

### Key Design: Symmetric Validation

Both challenger and defender use the **same validation mechanism**:
1. LLM proposes complete proof tactics
2. REPL validates in one shot
3. If valid → success; if invalid → MISS

By default, this follows traditional HORSE rules where missing your own shot doesn't give you a letter—the ball just passes. You can change this behavior with `challenger_takes_letter_on_miss: true` in the config if you want challengers penalized for failed conjectures.

The challenger gets multiple attempts (configurable via `max_conjecture_attempts`) to generate a valid theorem+proof, but the defender only gets one shot to match.

## Architecture

```
LLM Agent 1  <->  REPL Referee  <->  LLM Agent 2  <->  ...  <->  LLM Agent N
      |               |                   |                          |
   Generate      Verify via              Solve                      Solve
   theorems    Lean execution          theorems                   theorems
```

The REPL provides:
- **Inject Syntax**: Submit arbitrary theorem strings and receive immediate feedback
- **Maintain State**: Persist defined theorems for subsequent proof steps
- **Branch/Backtrack**: Support non-linear exploration via environment IDs

## Installation

### Prerequisites

- Python 3.11+
- Git

### Step 1: Install Lean 4

```bash
curl https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh -sSf | sh
```

Verify installation:
```bash
lean --version
lake --version
```

### Step 2: Build the Lean Project

```bash
cd lean_project
lake build
```

> **Note:** This downloads and compiles Mathlib, which can take **30-60 minutes** and requires **~8GB disk space**.

### Step 3: Install the Lean REPL

```bash
git clone https://github.com/leanprover-community/repl.git
cd repl
lake build
```

Test the REPL:
```bash
lake exe repl
```

Type `{"cmd": "def x := 1"}` and press Enter and then Ctrl+D. You should get a JSON response with an `env` field.

### Step 4: Install Python Dependencies

```bash
uv sync
```

or

```bash
pip install -e .
```

### Step 5: Configure Environment

```bash
export ANSPG_REPL_PATH=/path/to/repl/.lake/build/bin/repl
export ANSPG_LEAN_PROJECT=/path/to/anspg/lean_project
export OPENAI_API_KEY=sk-your-key-here
```

Or create a `.env` file in the project root:

```bash
ANSPG_REPL_PATH=/path/to/repl/.lake/build/bin/repl
ANSPG_LEAN_PROJECT=/path/to/anspg/lean_project
OPENAI_API_KEY=sk-your-key-here
```

## Usage

### Quick Start

```bash
# Copy example config
cp anspg.example.yaml anspg.yaml

# Run a battle (uses anspg.yaml automatically)
anspg battle
```

### With CLI Flags (2-Player)
This CLI flag will probabily be removed later in favor of the more flexible config with more agents

```bash
anspg battle \
  --model-a openai/gpt-oss-20b \
  --model-b z-ai/glm-4-32b \
  --difficulty 0.5 \
  --temperature 0.3 \
  --turns 10
```

### Multi-Player Game

Add more agents to your `anspg.yaml`:

```yaml
agents:
  - name: Alice
    model: qwen/qwen3-next-80b-a3b-instruct
  - name: Bob
    model: openai/gpt-oss-20b
  - name: Charlie
    model: z-ai/glm-4-32b
```

Use `--randomize` to shuffle the initial player order:
```bash
anspg battle --randomize
```

### With Explicit Paths

```bash
anspg battle \
  --repl-path ./repl/.lake/build/bin/repl \
  --lean-project ./lean_project \
  --config my_config.yaml
```

### Initialize a New Project

```bash
anspg init ./my-project
```

## Configuration

### YAML Configuration (Recommended)

Create `anspg.yaml` in your project directory:

```yaml
game:
  time_limit_s: 300
  max_turns: 10
  rulebook: mathlib
  simp_policy: allowed
  randomize_order: false
  challenger_takes_letter_on_miss: false

agent_defaults:
  temperature: 0.3
  difficulty_target: 0.4
  max_conjecture_attempts: 5

agents:
  - name: Alice
    model: gpt-4o
    temperature: 0.5
  - name: Bob
    model: claude-3-5-sonnet

logging:
  verbose: true
  show_failed_attempts: true
```

See `anspg.example.yaml` for all options.

### Environment Variables

| Variable | Description |
|----------|-------------|
| `OPENAI_API_KEY` or `ANSPG_LLM_API_KEY` | API key for LLM agents |
| `ANSPG_REPL_PATH` | Path to Lean REPL executable |
| `ANSPG_LEAN_PROJECT` | Path to Lean project with Mathlib |
| `OPENAI_BASE_URL` | Custom API base URL (e.g., OpenRouter) |

### CLI Options

| Option | Description |
|--------|-------------|
| `--config` / `-c` | Path to YAML config file |
| `--repl-path` | Path to REPL (overrides env var) |
| `--lean-project` | Path to Lean project (overrides env var) |
| `--model-a` / `--model-b` | LLM models for 2-player mode (override config) |
| `--difficulty` / `-d` | Difficulty target 0-1 (override config) |
| `--temperature` | LLM temperature (override config) |
| `--turns` / `-n` | Maximum number of turns (override config) |
| `--time-limit` / `-t` | Seconds per turn (override config) |
| `--verbose` / `-v` | Show detailed logging (override config) |
| `--randomize` | Randomize initial player order |
| `--challenger-takes-letter` | Challenger gets letter if they miss their own shot |

### OpenRouter Support

For OpenRouter models:

```bash
export OPENAI_API_KEY="sk-or-v1-..."
export OPENAI_BASE_URL="https://openrouter.ai/api/v1"
```

The CLI auto-detects OpenRouter keys (prefix `sk-or-`) and sets the base URL automatically.

## Project Structure

```
src/
├── orchestrator/
│   └── repl_referee.py      # REPL-based proof verification
├── lean_repl/
│   ├── client.py            # Lean REPL process management
│   ├── state_manager.py     # Proof state tracking
│   └── errors.py            # Error parsing
├── agents/
│   ├── horse.py             # HorseAgent: unified challenger/defender
│   ├── grounded.py          # GroundedProver: LLM+REPL proof search
│   ├── llm_client.py        # LLM API client
│   └── base.py              # Base agent interface
├── retrieval/               # Semantic search for premises
├── rulebook.py              # Rulebook system for allowed tactics
├── game_config.py           # YAML configuration system
├── models.py                # Core data structures
└── cli.py                   # CLI with 'battle' command
```

## HorseAgent Architecture

The unified `HorseAgent` class handles both challenger and defender roles:

### take_shot() - Challenger Role

```
For attempt in range(max_conjecture_attempts):  # Default: 5
  1. LLM generates theorem + proof tactics (JSON)
  2. Validate tactics against Rulebook
  3. Send to REPL for validation
  4. If valid → return Shot
  5. If invalid → record failure, try DIFFERENT theorem

All attempts failed → return None (challenger gets letter)
```

### match_shot() - Defender Role

```
1. LLM sees theorem statement only (proof hidden)
2. LLM generates proof tactics (JSON)
3. Validate tactics against Rulebook
4. Send to REPL for validation
5. If valid → MATCH (become challenger)
6. If invalid → MISS (get letter)

ONE ATTEMPT ONLY - no retries for defender
```

### Attempt Asymmetry

| Role | Attempts | Configurable? |
|------|----------|---------------|
| Challenger | `max_conjecture_attempts` (default 5) | Yes, in config |
| Defender | 1 (no retries) | No (by design) |

## Rulebooks

Rulebooks define what tactics and types are available:

```bash
anspg rulebooks  # List available rulebooks
```

| Rulebook | Description |
|----------|-------------|
| `basic` | Core Lean 4 tactics only |
| `mathlib` | Full Mathlib environment (default) |
| `competition` | No automated search tactics (aesop, exact?, etc.) |

## Development

```bash
# Install dev dependencies
uv sync --group dev

# Run tests
uv run pytest tests/ -v

# Type checking
uv run mypy src/

# Code formatting
uv run black src/
uv run ruff check src/
```

## Troubleshooting

### "lake: command not found"

Make sure you ran the elan installer and restarted your terminal:
```bash
source ~/.profile  # or ~/.bashrc or ~/.zshrc
```

### "REPL not found"

Check that `ANSPG_REPL_PATH` points to the correct binary:
```bash
ls -la $ANSPG_REPL_PATH
```

### "Mathlib build failed"

This usually means:
1. Not enough disk space (need ~8GB free)
2. Not enough RAM (need ~4GB minimum)
3. Network issues downloading Mathlib

Try again with:
```bash
cd lean_project
lake clean
lake build
```

### "Timeout waiting for REPL"

The first time you use the REPL with Mathlib, it needs to import everything. This can take 10-30 seconds. Subsequent runs are faster due to caching.

### "anspg: command not found"

Make sure you installed the Python package:
```bash
pip install -e .
```

## Links

- [Lean 4](https://leanprover.github.io/lean4/doc/) - Theorem prover
- [leanprover-community/repl](https://github.com/leanprover-community/repl) - Lean REPL
- [Pantograph](https://github.com/lenianiva/Pantograph) - Advanced REPL interface


