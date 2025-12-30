"""
CLI for ANSPG - Adversarial Neuro-Symbolic Proving Ground.

Commands:
- anspg battle: Run LLM H.O.R.S.E. battle with Lean REPL grounding
- anspg init: Initialize a new project
"""

from __future__ import annotations

import asyncio
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
    help="Adversarial Neuro-Symbolic Proving Ground - LLM H.O.R.S.E. Benchmark for Lean 4",
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
    max_conjecture_attempts: int = typer.Option(
        None, "--max-attempts", help="Max conjecture attempts per shot (overrides config)"
    ),
    simp_policy: str = typer.Option(
        None,
        "--simp-policy",
        "-s",
        help="Simp policy: allowed, no_auto_simp, banned (overrides config)",
    ),
    randomize_order: bool = typer.Option(
        None, "--randomize", help="Randomize player order (overrides config)"
    ),
    challenger_takes_letter: bool = typer.Option(
        None,
        "--challenger-takes-letter",
        help="Challenger gets letter if they miss their own shot (overrides config)",
    ),
):
    """
    Run an LLM H.O.R.S.E. battle grounded in Lean REPL.

    Configure agents in anspg.yaml:

        agents:
          - name: Alice
            model: gpt-4o
          - name: Bob
            model: claude-3-opus

    Then run: anspg battle
    """
    from .game_config import load_config, SimpPolicy

    config, config_path = load_config(config_file)
    agent_configs = config.get_agent_configs()

    # Apply global CLI overrides to all agents
    for agent_config in agent_configs:
        if difficulty is not None:
            agent_config.difficulty_target = difficulty
        if temperature is not None:
            agent_config.temperature = temperature
        if max_conjecture_attempts is not None:
            agent_config.max_conjecture_attempts = max_conjecture_attempts

    # CLI arguments override config file
    final_rulebook = rulebook if rulebook is not None else config.game.rulebook
    final_time_limit = time_limit if time_limit is not None else config.game.time_limit_s
    final_max_turns = turns if turns is not None else config.game.max_turns
    final_verbose = verbose if verbose is not None else config.logging.verbose
    final_simp_policy = (
        SimpPolicy.from_string(simp_policy) if simp_policy is not None else config.game.simp_policy
    )
    final_randomize = (
        randomize_order if randomize_order is not None else config.game.randomize_order
    )
    final_challenger_takes_letter = (
        challenger_takes_letter
        if challenger_takes_letter is not None
        else config.game.challenger_takes_letter_on_miss
    )

    # Determine game mode text
    n_players = len(agent_configs)
    mode_text = f"{n_players}-Player" if n_players > 2 else "1v1"

    console.print(
        Panel.fit(
            f"[bold blue]ANSPG - LLM {mode_text} Battle[/bold blue]\n"
            "[dim]H.O.R.S.E. Theorem Proving - Grounded in Lean REPL[/dim]",
            border_style="blue",
        )
    )

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
    console.print(f"  Players: {n_players}")

    for i, agent_config in enumerate(agent_configs):
        console.print(
            f"    [{i + 1}] {agent_config.name}: {agent_config.model} "
            f"(temp={agent_config.temperature}, difficulty={agent_config.difficulty_target}, "
            f"max_attempts={agent_config.max_conjecture_attempts})"
        )

    console.print(f"  Time limit: {final_time_limit}s per turn")
    console.print(f"  Max turns: {final_max_turns}")
    console.print(f"  Simp policy: {final_simp_policy.value}")
    console.print(f"  Randomize order: {final_randomize}")
    console.print(f"  Challenger takes letter on miss: {final_challenger_takes_letter}")

    asyncio.run(
        _run_battle(
            repl_path=repl_path,
            lean_project=lean_project,
            agent_configs=agent_configs,
            rulebook_name=final_rulebook,
            time_limit_s=final_time_limit,
            max_turns=final_max_turns,
            verbose=final_verbose,
            simp_policy=final_simp_policy,
            randomize_order=final_randomize,
            challenger_takes_letter_on_miss=final_challenger_takes_letter,
        )
    )


async def _run_battle(
    repl_path: Path,
    lean_project: Path,
    agent_configs: list,  # List of AgentConfig
    rulebook_name: str,
    time_limit_s: int,
    max_turns: int,
    verbose: bool,
    simp_policy,  # SimpPolicy enum
    randomize_order: bool = False,
    challenger_takes_letter_on_miss: bool = False,
):
    """Run an N-player LLM battle with Lean REPL grounding."""
    import random
    from .orchestrator.repl_referee import REPLReferee
    from .agents.horse import HorseAgent, HorseAgentConfig
    from .agents.llm_client import create_llm_client
    from .game_config import SimpPolicy
    from .models import DuplicateStatementTracker, GameStateView
    from .rulebook import (
        create_mathlib_rulebook,
        create_basic_rulebook,
        create_competition_rulebook,
        apply_simp_ban,
    )

    n_players = len(agent_configs)

    # Select rulebook
    if rulebook_name == "basic":
        rulebook = create_basic_rulebook()
    elif rulebook_name == "competition":
        rulebook = create_competition_rulebook()
    else:
        rulebook = create_mathlib_rulebook()

    # Apply simp ban if policy requires it
    if simp_policy == SimpPolicy.BANNED:
        rulebook = apply_simp_ban(rulebook)
        console.print("[yellow]Simp policy: BANNED - simp and related tactics removed[/yellow]")
    elif simp_policy == SimpPolicy.NO_AUTO_SIMP:
        console.print(
            "[yellow]Simp policy: NO_AUTO_SIMP - theorems solvable by simp alone will be rejected[/yellow]"
        )

    console.print(f"\n[dim]Using rulebook: {rulebook.name}[/dim]")
    if verbose:
        console.print(f"[dim]Available tactics: {len(rulebook.tactics.all_names())}[/dim]")

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

    # Create LLM clients for each agent
    llm_clients = [
        create_llm_client(api_key=api_key, base_url=base_url, model=cfg.model)
        for cfg in agent_configs
    ]

    # Get REPL client from referee
    repl_client = referee._client
    if repl_client is None:
        console.print("[red]REPL client not initialized[/red]")
        await referee.close()
        raise typer.Exit(1)

    # Shared statement tracker prevents agents from proposing the same theorem
    duplicate_tracker = DuplicateStatementTracker()

    # Create HorseAgent for each player
    agents: list[HorseAgent] = []
    for i, (cfg, llm) in enumerate(zip(agent_configs, llm_clients)):
        agent = HorseAgent(
            repl_client=repl_client,
            llm_client=llm,
            rulebook=rulebook,
            config=HorseAgentConfig(
                name=cfg.name,
                model=cfg.model,
                temperature=cfg.temperature,
                max_tokens=cfg.max_tokens,
                difficulty_target=cfg.difficulty_target,
                max_conjecture_attempts=cfg.max_conjecture_attempts,
                simp_policy=simp_policy,
                prompt_context=cfg.get_prompt_context(),
            ),
            duplicate_tracker=duplicate_tracker,
        )
        agents.append(agent)

    # Initialize game state
    names = [cfg.name for cfg in agent_configs]
    models = [cfg.model for cfg in agent_configs]
    scores = {name: "" for name in names}
    eliminated = set()  # Set of eliminated player names

    # Create rotation order (indices into agents/names lists)
    rotation_order = list(range(n_players))
    if randomize_order:
        random.shuffle(rotation_order)
        console.print(
            f"[yellow]Randomized player order: {[names[i] for i in rotation_order]}[/yellow]"
        )

    # Current position in rotation (index into rotation_order)
    current_pos = 0

    def get_active_players() -> list[int]:
        """Get indices of non-eliminated players in rotation order."""
        return [i for i in rotation_order if names[i] not in eliminated]

    def get_challenger_idx() -> int:
        """Get the current challenger's index."""
        active = get_active_players()
        if not active:
            return -1
        return active[current_pos % len(active)]

    def get_defenders(challenger_idx: int) -> list[int]:
        """Get indices of defenders (all active players except challenger) in rotation order."""
        active = get_active_players()
        challenger_pos_in_active = active.index(challenger_idx)
        # Return players after challenger in rotation, wrapping around
        defenders = []
        for i in range(1, len(active)):
            idx = active[(challenger_pos_in_active + i) % len(active)]
            defenders.append(idx)
        return defenders

    def advance_turn():
        """Advance to the next player in rotation."""
        nonlocal current_pos
        active = get_active_players()
        if active:
            current_pos = (current_pos + 1) % len(active)

    def display_scores():
        table = Table(title="Scoreboard", show_header=True)
        table.add_column("Order", style="dim")
        table.add_column("Player", style="cyan")
        table.add_column("Model", style="dim")
        table.add_column("Letters", style="red")
        table.add_column("Status", style="green")

        # Display in rotation order
        for pos, idx in enumerate(rotation_order):
            name = names[idx]
            model = models[idx]
            letters = scores[name]

            if name in eliminated:
                status = "ELIMINATED"
                status_style = "red"
            else:
                status = "Playing"
                status_style = "green"

            display_letters = letters if letters else "-"
            order_marker = f"[{pos + 1}]"

            table.add_row(
                order_marker,
                name,
                model,
                display_letters,
                f"[{status_style}]{status}[/{status_style}]",
            )

        console.print(table)

    def add_letter(player_name: str, reason: str) -> str:
        """Add a letter to a player's score. Returns the letter added."""
        horse = "HORSE"
        current = scores[player_name]
        if len(current) < 5:
            letter = horse[len(current)]
            scores[player_name] += letter
            console.print(f"[red]{player_name} receives '{letter}' - {reason}[/red]")

            # Check for elimination
            if scores[player_name] == "HORSE":
                eliminated.add(player_name)
                console.print(f"[bold red]{player_name} has been ELIMINATED![/bold red]")

            return letter
        return ""

    def build_game_state_view(
        agent_name: str,
        turn_num: int,
        challenger_name: str | None = None,
    ) -> GameStateView:
        return GameStateView(
            standings=dict(scores),
            turn_number=turn_num,
            challenger_name=challenger_name,
            my_letters=scores.get(agent_name, ""),
            my_name=agent_name,
        )

    display_scores()

    # Main game loop
    try:
        for turn in range(1, max_turns + 1):
            active_players = get_active_players()

            # Check for winner (only 1 player remaining)
            if len(active_players) <= 1:
                if active_players:
                    winner_name = names[active_players[0]]
                    console.print(f"\n[bold green]{winner_name} WINS![/bold green]")
                break

            challenger_idx = get_challenger_idx()
            if challenger_idx < 0:
                break

            challenger = agents[challenger_idx]
            challenger_name = names[challenger_idx]
            defenders_idx = get_defenders(challenger_idx)

            console.print(f"\n[bold]{'=' * 60}[/bold]")
            console.print(f"[bold]Turn {turn}[/bold]")
            console.print(
                f"  Challenger: [cyan]{challenger_name}[/cyan] ({challenger.config.model})"
            )
            if len(defenders_idx) == 1:
                console.print(
                    f"  Defender: [cyan]{names[defenders_idx[0]]}[/cyan] ({agents[defenders_idx[0]].config.model})"
                )
            else:
                console.print(f"  Defenders ({len(defenders_idx)}):")
                for i, def_idx in enumerate(defenders_idx):
                    console.print(
                        f"    [{i + 1}] [cyan]{names[def_idx]}[/cyan] ({agents[def_idx].config.model})"
                    )

            # Phase 1: Challenger takes a shot (propose + prove)
            console.print(f"\n[bold]Phase 1:[/bold] {challenger_name} taking a shot...")

            # Set game state context for challenger
            challenger.set_game_state(
                build_game_state_view(challenger_name, turn, challenger_name=None)
            )

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
                # Challenger failed to take shot
                if challenger_takes_letter_on_miss:
                    add_letter(challenger_name, "failed to take shot")
                else:
                    console.print(
                        f"[yellow]{challenger_name} missed their shot - no letter assigned (traditional rules)[/yellow]"
                    )

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
                # Ball passes to next player (they choose their shot)
                advance_turn()
                continue

            # Shot was taken successfully (challenger proved it)
            console.print(f"\n[green]Shot taken![/green]")
            console.print(f"  Theorem: [cyan]{shot.theorem_statement}[/cyan]")
            if shot.challenger_proof and shot.challenger_proof.tactics:
                tactics_display = " ; ".join(shot.challenger_proof.tactics[:5])
                if len(shot.challenger_proof.tactics) > 5:
                    tactics_display += f" ... (+{len(shot.challenger_proof.tactics) - 5} more)"
                console.print(f"  Proof: [dim]{tactics_display}[/dim]")

            # Phase 2: Each defender tries to match (in rotation order)
            defender_failed = False

            for def_idx in defenders_idx:
                if names[def_idx] in eliminated:
                    continue  # Skip eliminated players

                defender = agents[def_idx]
                defender_name = names[def_idx]

                console.print(f"\n[bold]Phase 2:[/bold] {defender_name} attempting to match...")

                # CRITICAL: Defender sees only the theorem (Statement Sanitizer)
                defender_shot = shot.get_defender_view()
                console.print(f"  [dim](Defender sees only the statement, not the proof)[/dim]")

                # Set game state context for defender (includes who proposed the theorem)
                defender.set_game_state(
                    build_game_state_view(defender_name, turn, challenger_name=challenger_name)
                )

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
                else:
                    error_msg = defense_result.error_message if defense_result else "no attempt"
                    if verbose:
                        if defense_result and defense_result.error_message:
                            console.print(f"  [dim]Failed: {error_msg}[/dim]")
                            if defense_result.tactics:
                                tactics_tried = " ; ".join(defense_result.tactics[:3])
                                if len(defense_result.tactics) > 3:
                                    tactics_tried += (
                                        f" ... (+{len(defense_result.tactics) - 3} more)"
                                    )
                                console.print(f"  [dim]Tactics tried: {tactics_tried}[/dim]")
                        else:
                            console.print(f"  [dim]No defense attempt generated[/dim]")

                    add_letter(defender_name, "failed to match")
                    defender_failed = True
                    break  # Stop checking other defenders once one fails

            # Determine next turn
            if defender_failed:
                # A defender failed - challenger shoots again (stays at current position)
                console.print(f"[dim]{challenger_name} keeps the ball[/dim]")
            else:
                # All defenders matched - ball passes to next player
                console.print(f"[dim]All defenders matched! Ball passes to next player[/dim]")
                advance_turn()

            display_scores()

    finally:
        await referee.close()

    # Final results
    console.print("\n" + "=" * 60)
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

    # Determine and announce winner
    active_players = get_active_players()
    if len(active_players) == 1:
        winner_name = names[active_players[0]]
        console.print(f"\n[bold green]Winner: {winner_name}[/bold green]")
    elif len(active_players) == 0:
        console.print("\n[yellow]All players eliminated![/yellow]")
    else:
        console.print("\n[yellow]Game ended without elimination[/yellow]")
        # Show standings
        standings = sorted(
            [(name, scores[name]) for name in names if name not in eliminated],
            key=lambda x: len(x[1]),
        )
        console.print("Final standings (by fewest letters):")
        for i, (name, letters) in enumerate(standings, 1):
            console.print(f"  {i}. {name}: {letters if letters else '-'}")


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
