"""
CLI for ANSPG - Adversarial Neuro-Symbolic Proving Ground.

Commands:
- anspg battle: Run LLM 1v1 benchmark with Lean REPL grounding
- anspg init: Initialize a new project
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

# Load environment variables from .env file
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    # python-dotenv not available, use environment variables directly
    pass

app = typer.Typer(
    name="anspg",
    help="Adversarial Neuro-Symbolic Proving Ground - LLM 1v1 Benchmark for Lean 4",
)
console = Console()


@app.command("battle")
def battle(
    repl_path: Path = typer.Option(
        None,
        "--repl-path",
        envvar="ANSPG_REPL_PATH",
        help="Path to Lean REPL executable (or set ANSPG_REPL_PATH env var)",
    ),
    lean_project: Path = typer.Option(
        None,
        "--lean-project",
        envvar="ANSPG_LEAN_PROJECT",
        help="Path to Lean project with Mathlib (or set ANSPG_LEAN_PROJECT env var)",
    ),
    config_file: Path = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to YAML config file (defaults to anspg.yaml if exists)",
    ),
    model_a: str = typer.Option(
        None, "--model-a", "-a", help="Model for agent A (overrides config)"
    ),
    model_b: str = typer.Option(
        None, "--model-b", "-b", help="Model for agent B (overrides config)"
    ),
    name_a: str = typer.Option(None, "--name-a", help="Name for agent A (overrides config)"),
    name_b: str = typer.Option(None, "--name-b", help="Name for agent B (overrides config)"),
    rulebook: str = typer.Option(
        None,
        "--rulebook",
        "-r",
        help="Rulebook preset: basic, mathlib, competition (overrides config)",
    ),
    time_limit: int = typer.Option(
        None, "--time-limit", "-t", help="Time limit per turn in seconds (overrides config)"
    ),
    turns: int = typer.Option(
        None, "--turns", "-n", help="Maximum number of turns (overrides config)"
    ),
    verbose: bool = typer.Option(
        None, "--verbose", "-v", help="Show detailed proof steps (overrides config)"
    ),
    difficulty: float = typer.Option(
        None, "--difficulty", "-d", help="Difficulty target 0-1 (overrides config)"
    ),
    temperature: float = typer.Option(
        None, "--temperature", help="LLM temperature (overrides config)"
    ),
):
    """
    Run an LLM 1v1 battle grounded in Lean REPL.

    This is the core benchmark: two LLMs compete in H.O.R.S.E. with
    deterministic proof verification via the Lean REPL.

    Example:
        anspg battle --repl-path ./repl/.lake/build/bin/repl --lean-project ./lean_project

    Or with environment variables:
        export ANSPG_REPL_PATH=./repl/.lake/build/bin/repl
        export ANSPG_LEAN_PROJECT=./lean_project
        anspg battle --model-a gpt-4o --model-b claude-3-opus
    """
    from .game_config import load_config

    console.print(
        Panel.fit(
            "[bold blue]ANSPG - LLM 1v1 Battle[/bold blue]\n"
            "[dim]H.O.R.S.E. Theorem Proving - Grounded in Lean REPL[/dim]",
            border_style="blue",
        )
    )

    # Load configuration (file or defaults)
    config, config_path = load_config(config_file)

    # CLI arguments override config file
    final_rulebook = rulebook if rulebook is not None else config.game.rulebook
    final_time_limit = time_limit if time_limit is not None else config.game.time_limit_s
    final_max_turns = turns if turns is not None else config.game.max_turns
    final_verbose = verbose if verbose is not None else config.logging.verbose

    # Agent A config
    agent_a_config = config.get_agent_config("agent_a")
    if model_a is not None:
        agent_a_config.model = model_a
    if name_a is not None:
        agent_a_config.name = name_a
    elif model_a is not None and agent_a_config.name == "Agent":
        # Auto-name based on model if no name provided
        agent_a_config.name = "Agent-A"
    if difficulty is not None:
        agent_a_config.difficulty_target = difficulty
    if temperature is not None:
        agent_a_config.temperature = temperature

    # Agent B config
    agent_b_config = config.get_agent_config("agent_b")
    if model_b is not None:
        agent_b_config.model = model_b
    if name_b is not None:
        agent_b_config.name = name_b
    elif model_b is not None and agent_b_config.name == "Agent":
        # Auto-name based on model if no name provided
        agent_b_config.name = "Agent-B"
    if difficulty is not None:
        agent_b_config.difficulty_target = difficulty
    if temperature is not None:
        agent_b_config.temperature = temperature

    # Validate REPL configuration
    if not repl_path or not lean_project:
        console.print("[red]Error: REPL configuration missing![/red]")
        console.print("\nYou must provide either:")
        console.print("  1. Command line options: --repl-path and --lean-project")
        console.print("  2. Environment variables: ANSPG_REPL_PATH and ANSPG_LEAN_PROJECT")
        console.print("\nSee SETUP.md for installation instructions.")
        raise typer.Exit(1)

    if not repl_path.exists():
        console.print(f"[red]REPL not found: {repl_path}[/red]")
        console.print("[dim]Build it with: cd repl && lake build[/dim]")
        raise typer.Exit(1)

    if not lean_project.exists():
        console.print(f"[red]Lean project not found: {lean_project}[/red]")
        console.print("[dim]Initialize with: anspg init[/dim]")
        raise typer.Exit(1)

    console.print(f"\n[bold]Configuration:[/bold]")
    if config_path:
        console.print(f"  Config: {config_path}")
    else:
        console.print(f"  Config: [dim](using defaults)[/dim]")
    console.print(f"  REPL: {repl_path}")
    console.print(f"  Project: {lean_project}")
    console.print(f"  Rulebook: {final_rulebook}")
    console.print(
        f"  {agent_a_config.name}: {agent_a_config.model} (temp={agent_a_config.temperature}, difficulty={agent_a_config.difficulty_target})"
    )
    console.print(
        f"  {agent_b_config.name}: {agent_b_config.model} (temp={agent_b_config.temperature}, difficulty={agent_b_config.difficulty_target})"
    )
    console.print(f"  Time limit: {final_time_limit}s per turn")
    console.print(f"  Max turns: {final_max_turns}")

    asyncio.run(
        _run_battle(
            repl_path=repl_path,
            lean_project=lean_project,
            agent_a_config=agent_a_config,
            agent_b_config=agent_b_config,
            rulebook_name=final_rulebook,
            time_limit_s=final_time_limit,
            max_turns=final_max_turns,
            verbose=final_verbose,
        )
    )


async def _run_battle(
    repl_path: Path,
    lean_project: Path,
    agent_a_config: "AgentConfig",
    agent_b_config: "AgentConfig",
    rulebook_name: str,
    time_limit_s: int,
    max_turns: int,
    verbose: bool,
):
    """
    Run an LLM battle with Lean REPL grounding.

    Game Flow (H.O.R.S.E. style):
    1. Challenger takes a shot: proposes theorem AND proves it
    2. If challenger's proof fails: challenger gets a letter
    3. Defender tries to match: proves the same theorem their way
    4. If defender fails: defender gets a letter, challenger stays
    5. If defender succeeds: roles swap
    6. First to spell H-O-R-S-E loses
    """
    from .orchestrator.repl_referee import REPLReferee
    from .agents.horse import HorseAgent, HorseAgentConfig
    from .agents.llm_client import create_llm_client
    from .game_config import AgentConfig
    from .rulebook import (
        create_mathlib_rulebook,
        create_basic_rulebook,
        create_competition_rulebook,
    )

    # Extract names for convenience
    name_a = agent_a_config.name
    name_b = agent_b_config.name
    model_a = agent_a_config.model
    model_b = agent_b_config.model

    # Select rulebook
    if rulebook_name == "basic":
        rulebook = create_basic_rulebook()
    elif rulebook_name == "competition":
        rulebook = create_competition_rulebook()
    else:
        rulebook = create_mathlib_rulebook()

    console.print(f"\n[dim]Using rulebook: {rulebook.name}[/dim]")
    if verbose:
        console.print(f"[dim]Available tactics: {len(rulebook.tactics.all_names())}[/dim]")

    # Initialize REPL referee
    console.print("\n[dim]Initializing Lean REPL (this may take a moment for Mathlib)...[/dim]")

    referee = REPLReferee(
        repl_path=repl_path,
        lean_project=lean_project,
        verbose=verbose,
    )

    try:
        await referee.initialize()
        console.print("[green]REPL initialized successfully[/green]")
    except FileNotFoundError as e:
        console.print(f"[red]REPL not found: {e}[/red]")
        console.print("[dim]See SETUP.md for installation instructions[/dim]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Failed to initialize REPL: {e}[/red]")
        console.print("[dim]Make sure Lean 4 and the REPL are properly installed[/dim]")
        raise typer.Exit(1)

    # Create LLM clients
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("ANSPG_LLM_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL") or os.getenv("ANSPG_LLM_BASE_URL")

    if not api_key:
        console.print("[red]No API key found. Set OPENAI_API_KEY or ANSPG_LLM_API_KEY[/red]")
        await referee.close()
        raise typer.Exit(1)

    # Detect OpenRouter keys
    if api_key.startswith("sk-or-") and not base_url:
        console.print("[yellow]OpenRouter key detected, setting base_url[/yellow]")
        base_url = "https://openrouter.ai/api/v1"

    llm_a = create_llm_client(api_key=api_key, base_url=base_url, model=model_a)
    llm_b = create_llm_client(api_key=api_key, base_url=base_url, model=model_b)

    # Create agents - both use the SAME REPL client (shared environment)
    # The referee's client has Mathlib imported
    repl_client = referee._client

    agent_a = HorseAgent(
        repl_client=repl_client,
        llm_client=llm_a,
        rulebook=rulebook,
        config=HorseAgentConfig(
            name=agent_a_config.name,
            model=agent_a_config.model,
            temperature=agent_a_config.temperature,
            max_tokens=agent_a_config.max_tokens,
            difficulty_target=agent_a_config.difficulty_target,
            max_conjecture_attempts=agent_a_config.max_conjecture_attempts,
        ),
    )

    agent_b = HorseAgent(
        repl_client=repl_client,
        llm_client=llm_b,
        rulebook=rulebook,
        config=HorseAgentConfig(
            name=agent_b_config.name,
            model=agent_b_config.model,
            temperature=agent_b_config.temperature,
            max_tokens=agent_b_config.max_tokens,
            difficulty_target=agent_b_config.difficulty_target,
            max_conjecture_attempts=agent_b_config.max_conjecture_attempts,
        ),
    )

    # Game state
    scores = {name_a: "", name_b: ""}
    challenger_idx = 0
    agents = [agent_a, agent_b]
    names = [name_a, name_b]

    def display_scores():
        table = Table(title="Scoreboard", show_header=True)
        table.add_column("Player", style="cyan")
        table.add_column("Model", style="dim")
        table.add_column("Letters", style="red")
        table.add_column("Status", style="green")

        for name, letters in scores.items():
            model = model_a if name == name_a else model_b
            status = "ELIMINATED" if letters == "HORSE" else "Playing"
            display_letters = letters if letters else "-"
            table.add_row(name, model, display_letters, status)

        console.print(table)

    def add_letter(player_name: str, reason: str) -> str:
        """Add a letter to a player's score. Returns the letter added."""
        horse = "HORSE"
        current = scores[player_name]
        if len(current) < 5:
            letter = horse[len(current)]
            scores[player_name] += letter
            console.print(f"[red]{player_name} receives '{letter}' - {reason}[/red]")
            return letter
        return ""

    display_scores()

    # Main game loop
    try:
        for turn in range(1, max_turns + 1):
            # Check for winner
            if scores[name_a] == "HORSE":
                console.print(f"\n[bold green]{name_b} WINS![/bold green]")
                break
            if scores[name_b] == "HORSE":
                console.print(f"\n[bold green]{name_a} WINS![/bold green]")
                break

            challenger = agents[challenger_idx]
            defender = agents[1 - challenger_idx]
            challenger_name = names[challenger_idx]
            defender_name = names[1 - challenger_idx]

            console.print(f"\n[bold]{'=' * 50}[/bold]")
            console.print(f"[bold]Turn {turn}[/bold]")
            console.print(
                f"  Challenger: [cyan]{challenger_name}[/cyan] ({challenger.config.model})"
            )
            console.print(f"  Defender: [cyan]{defender_name}[/cyan] ({defender.config.model})")

            # ============================================================
            # PHASE 1: Challenger takes a shot (propose + prove)
            # ============================================================
            console.print(f"\n[bold]Phase 1:[/bold] {challenger_name} taking a shot...")

            shot = None
            try:
                shot = await asyncio.wait_for(
                    challenger.take_shot(),
                    timeout=time_limit_s,
                )
            except asyncio.TimeoutError:
                console.print(f"[yellow]{challenger_name} timed out[/yellow]")
            except Exception as e:
                console.print(f"[red]Error: {e}[/red]")
                if verbose:
                    import traceback

                    console.print(f"[dim]{traceback.format_exc()}[/dim]")

            if not shot:
                add_letter(challenger_name, "failed to take shot")

                # Show failure details in verbose mode
                if verbose and challenger.stats.last_failure_reason:
                    console.print(f"  [dim]Reason: {challenger.stats.last_failure_reason}[/dim]")
                    if challenger.stats.last_failure_attempts:
                        console.print(f"  [dim]Attempts:[/dim]")
                        for i, attempt in enumerate(challenger.stats.last_failure_attempts[:3], 1):
                            console.print(f"    [dim]{i}. {attempt}[/dim]")
                        if len(challenger.stats.last_failure_attempts) > 3:
                            console.print(
                                f"    [dim]... and {len(challenger.stats.last_failure_attempts) - 3} more[/dim]"
                            )

                display_scores()
                # Swap roles - defender becomes challenger
                challenger_idx = 1 - challenger_idx
                continue

            # Shot was taken successfully (challenger proved it)
            console.print(f"\n[green]Shot taken![/green]")
            console.print(f"  Theorem: [cyan]{shot.theorem_statement}[/cyan]")
            if shot.challenger_proof and shot.challenger_proof.tactics:
                tactics_display = " ; ".join(shot.challenger_proof.tactics[:5])
                if len(shot.challenger_proof.tactics) > 5:
                    tactics_display += f" ... (+{len(shot.challenger_proof.tactics) - 5} more)"
                console.print(f"  Proof: [dim]{tactics_display}[/dim]")

            # ============================================================
            # PHASE 2: Defender tries to match
            # ============================================================
            console.print(f"\n[bold]Phase 2:[/bold] {defender_name} attempting to match...")

            # CRITICAL: Defender sees only the theorem (Statement Sanitizer)
            defender_shot = shot.get_defender_view()
            console.print(f"  [dim](Defender sees only the statement, not the proof)[/dim]")

            defense_result = None
            try:
                defense_result = await asyncio.wait_for(
                    defender.match_shot(defender_shot),
                    timeout=time_limit_s,
                )
            except asyncio.TimeoutError:
                console.print(f"[yellow]{defender_name} timed out[/yellow]")
            except Exception as e:
                console.print(f"[red]Error: {e}[/red]")
                if verbose:
                    import traceback

                    console.print(f"[dim]{traceback.format_exc()}[/dim]")

            if defense_result and defense_result.success:
                tactics_display = " ; ".join(defense_result.tactics[:5])
                if len(defense_result.tactics) > 5:
                    tactics_display += f" ... (+{len(defense_result.tactics) - 5} more)"
                console.print(f"  Defense: [dim]{tactics_display}[/dim]")
                console.print(
                    f"[green]{defender_name} matched the shot! "
                    f"({len(defense_result.tactics)} tactics, {defense_result.time_ms:.0f}ms)[/green]"
                )
                # Defender matched - they become challenger
                challenger_idx = 1 - challenger_idx
            else:
                error_msg = defense_result.error_message if defense_result else "no attempt"
                if verbose:
                    if defense_result and defense_result.error_message:
                        console.print(f"  [dim]Failed: {error_msg}[/dim]")
                        if defense_result.tactics:
                            tactics_tried = " ; ".join(defense_result.tactics[:3])
                            if len(defense_result.tactics) > 3:
                                tactics_tried += f" ... (+{len(defense_result.tactics) - 3} more)"
                            console.print(f"  [dim]Tactics tried: {tactics_tried}[/dim]")
                    else:
                        console.print(f"  [dim]No defense attempt generated[/dim]")
                add_letter(defender_name, "failed to match")

            display_scores()

    finally:
        await referee.close()

    # Final results
    console.print("\n" + "=" * 50)
    console.print("[bold]Final Results[/bold]")
    display_scores()

    # Show agent stats
    if verbose:
        console.print("\n[bold]Agent Statistics:[/bold]")
        for agent in agents:
            stats = agent.get_stats()
            console.print(f"  {stats['name']} ({stats['model']}):")
            console.print(f"    Shots proposed: {stats['shots_proposed']}")
            console.print(f"    Shots validated: {stats['shots_validated']}")
            console.print(f"    Shots matched: {stats['shots_matched']}")
            console.print(f"    Shots failed: {stats['shots_failed']}")
            console.print(f"    Total tactics: {stats['total_tactics']}")
            console.print(f"    Total time: {stats['total_time_ms']:.0f}ms")

    if scores[name_a] == "HORSE":
        console.print(f"\n[bold green]Winner: {name_b}[/bold green]")
    elif scores[name_b] == "HORSE":
        console.print(f"\n[bold green]Winner: {name_a}[/bold green]")
    else:
        console.print("\n[yellow]Game ended without elimination[/yellow]")


@app.command()
def init(
    project_dir: Path = typer.Argument(Path("."), help="Directory to initialize"),
):
    """
    Initialize a new ANSPG project with Lean 4 setup.
    """
    project_dir = project_dir.resolve()

    console.print(f"[bold]Initializing ANSPG project in {project_dir}[/bold]")

    # Create directories
    (project_dir / "theorems").mkdir(parents=True, exist_ok=True)

    # Create lakefile
    lakefile = project_dir / "lakefile.lean"
    if not lakefile.exists():
        lakefile.write_text("""import Lake
open Lake DSL

package anspg_game where
  version := v!"0.1.0"

require mathlib from git
  "https://github.com/leanprover-community/mathlib4"

@[default_target]
lean_lib ANSPGGame where
  srcDir := "theorems"
""")
        console.print("  Created lakefile.lean")

    # Create example theorem
    example_file = project_dir / "theorems" / "Example.lean"
    if not example_file.exists():
        example_file.write_text("""import Mathlib

-- Example theorem for H.O.R.S.E. game
theorem nat_add_zero : forall n : Nat, n + 0 = n := by
  intro n
  rfl

-- A slightly harder theorem
theorem nat_add_comm : forall n m : Nat, n + m = m + n := by
  intros n m
  omega
""")
        console.print("  Created theorems/Example.lean")

    # Create config
    from .game_config import create_default_config

    config_file = project_dir / "anspg.yaml"
    if not config_file.exists():
        create_default_config(config_file)
        console.print("  Created anspg.yaml")

    console.print("\n[green]Project initialized![/green]")
    console.print("\nNext steps:")
    console.print("  1. Install Lean 4 (see SETUP.md)")
    console.print("  2. Run 'lake build' to build the Lean project")
    console.print("  3. Set up the REPL (see SETUP.md)")
    console.print("  4. Set OPENAI_API_KEY environment variable")
    console.print("  5. Run 'anspg battle' to start a game")


@app.command()
def rulebooks():
    """
    List available rulebook presets.
    """
    from .rulebook import (
        create_basic_rulebook,
        create_mathlib_rulebook,
        create_competition_rulebook,
    )

    console.print("[bold]Available Rulebooks:[/bold]\n")

    for name, factory in [
        ("basic", create_basic_rulebook),
        ("mathlib", create_mathlib_rulebook),
        ("competition", create_competition_rulebook),
    ]:
        rb = factory()
        console.print(f"[cyan]{name}[/cyan]")
        console.print(f"  {rb.description}")
        console.print(f"  Tactics: {len(rb.tactics.all_names())}")
        console.print(f"  Types: {', '.join(rb.types.all_lean_names())}")
        console.print()


if __name__ == "__main__":
    app()
