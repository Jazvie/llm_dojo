"""
Lean 4 REPL Client with Persistent Process.

Implements the SOTA architecture for AI-driven theorem proving:
- Persistent Lean process (no restart per tactic)
- JSON protocol over stdin/stdout
- State handles for O(1) backtracking
- Structured error parsing

This replaces the naive "spawn process per tactic" approach that
introduces 1-5 second latency per step. With a persistent REPL,
step latency drops to 10-50ms, enabling MCTS and other search algorithms.

Compatible with: https://github.com/leanprover-community/repl
"""

from __future__ import annotations

import asyncio
import json
import subprocess
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, AsyncIterator

from .errors import (
    LeanError,
    LeanErrorCategory,
    StructuredError,
    parse_lean_error,
)
from .state_manager import StateHandle, StateManager


class REPLStatus(Enum):
    """Status of the REPL process."""

    NOT_STARTED = auto()
    STARTING = auto()
    READY = auto()
    BUSY = auto()
    ERROR = auto()
    TERMINATED = auto()


@dataclass
class ProofState:
    """
    The current state of a proof in progress.

    Tracks goals, hypotheses, and REPL state IDs for backtracking.
    """

    goals: list[str]
    hypotheses: list[dict[str, str]] = field(default_factory=list)

    # REPL environment ID (for command mode)
    env_id: int = 0

    # REPL proof state ID (for tactic mode) - different from env_id!
    proof_state_id: int = 0

    # Is the proof complete?
    is_complete: bool = False

    # Pretty-printed state from Lean
    pretty_state: str = ""

    @property
    def num_goals(self) -> int:
        return len(self.goals)

    def to_dict(self) -> dict[str, Any]:
        return {
            "goals": self.goals,
            "hypotheses": self.hypotheses,
            "env_id": self.env_id,
            "proof_state_id": self.proof_state_id,
            "is_complete": self.is_complete,
            "num_goals": self.num_goals,
        }


@dataclass
class TacticResult:
    """Result of applying a tactic."""

    success: bool
    new_state: ProofState | None = None
    error: StructuredError | None = None

    # Performance metrics
    time_ms: float = 0.0

    # For backtracking
    env_id: int = 0  # New environment ID after tactic

    @property
    def is_proof_complete(self) -> bool:
        return self.success and self.new_state is not None and self.new_state.is_complete


@dataclass
class REPLResponse:
    """
    A response from the Lean REPL.

    The REPL communicates via JSON with the following structure:
    Command mode response:
    {
        "env": <int>,           // Environment handle
        "messages": [...],      // Diagnostic messages
        "sorries": [...],       // Sorry placeholders with proofState IDs
    }

    Tactic mode response:
    {
        "proofState": <int>,    // New proof state ID
        "goals": [...],         // Remaining goals as strings
    }
    """

    env_id: int = 0
    messages: list[dict[str, Any]] = field(default_factory=list)
    goals: list[str] = field(default_factory=list)
    proof_state_id: int | None = None  # For tactic mode responses
    sorries: list[dict[str, Any]] = field(default_factory=list)

    # Error info
    has_error: bool = False
    error_message: str = ""

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> "REPLResponse":
        """Parse a JSON response from the REPL."""
        # Check for error
        has_error = False
        error_message = ""

        # Command mode errors: {"messages": [{"severity": "error", "data": "..."}], ...}
        messages = data.get("messages", [])
        for msg in messages:
            if msg.get("severity") == "error":
                has_error = True
                error_message = msg.get("data", "Unknown error")
                break

        # Tactic mode errors: {"message": "Lean error:\n..."}
        if "message" in data and not has_error:
            msg_text = data["message"]
            if "error" in msg_text.lower():
                has_error = True
                error_message = msg_text

        # Handle goals - can be a list of strings (tactic mode) or empty
        goals = data.get("goals", [])

        # proofState can be an int (tactic response) - different from "proofState" in sorries
        proof_state_id = data.get("proofState") if isinstance(data.get("proofState"), int) else None

        return cls(
            env_id=data.get("env", 0),
            messages=messages,
            goals=goals,
            proof_state_id=proof_state_id,
            sorries=data.get("sorries", []),
            has_error=has_error,
            error_message=error_message,
        )


@dataclass
class REPLState:
    """
    Complete state of the REPL session.

    Tracks the current environment, imports, and theorem context.
    """

    status: REPLStatus = REPLStatus.NOT_STARTED
    current_env_id: int = 0
    imports: list[str] = field(default_factory=list)

    # Current theorem (if in proof mode)
    current_theorem: str | None = None
    in_proof_mode: bool = False

    # Statistics
    commands_executed: int = 0
    total_time_ms: float = 0.0


class LeanREPLClient:
    """
    Client for interacting with Lean 4 via a persistent REPL.

    This is the foundation of the SOTA architecture. It maintains
    a single Lean process and communicates via JSON, enabling:

    1. Fast tactic execution (10-50ms vs 1-5s)
    2. State handles for O(1) backtracking
    3. Structured error feedback for AI agents

    Usage:
        async with LeanREPLClient() as client:
            await client.initialize("/path/to/repo")
            state = await client.enter_theorem("MyTheorem")
            result = await client.run_tactic(state, "intro x")
    """

    def __init__(
        self,
        repl_path: str | None = None,
        timeout_seconds: float = 60.0,
        verbose: bool = False,
    ):
        """
        Initialize the REPL client.

        Args:
            repl_path: Path to the repl executable. If None, uses "repl" from PATH.
            timeout_seconds: Default timeout for commands.
            verbose: Enable debug output.
        """
        self._repl_path = repl_path or "repl"
        self._timeout = timeout_seconds
        self._verbose = verbose

        self._process: subprocess.Popen | None = None
        self._state = REPLState()
        self._state_manager = StateManager()

        # Lock for sequential command execution
        self._lock = asyncio.Lock()

    async def __aenter__(self) -> "LeanREPLClient":
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.close()

    @property
    def is_ready(self) -> bool:
        return self._state.status == REPLStatus.READY

    @property
    def state(self) -> REPLState:
        return self._state

    async def start(self, repo_path: Path | str) -> bool:
        """
        Start the REPL process.

        Args:
            repo_path: Path to the Lean repository (for lake env)

        Returns:
            True if started successfully
        """
        if self._state.status == REPLStatus.READY:
            return True

        self._state.status = REPLStatus.STARTING
        repo_path = Path(repo_path)

        try:
            # Start the REPL process with lake environment
            # The repl tool expects to be run from a Lake project
            self._process = subprocess.Popen(
                ["lake", "env", self._repl_path],
                cwd=str(repo_path),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                # Avoid deadlocks when Lean writes a lot to stderr (e.g., Mathlib import)
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,  # Line buffered
            )

            # The REPL is ready immediately after spawning - it waits for input
            # No initial greeting/response is sent by the REPL
            self._state.status = REPLStatus.READY
            self._state.current_env_id = 0
            return True

        except FileNotFoundError:
            self._state.status = REPLStatus.ERROR
            raise RuntimeError(
                f"REPL not found at '{self._repl_path}'. "
                "Install with: lake build repl (in a Lake project)"
            )
        except Exception as e:
            self._state.status = REPLStatus.ERROR
            raise RuntimeError(f"Failed to start REPL: {e}")

    async def initialize(
        self,
        repo_path: Path | str,
        imports: list[str] | None = None,
    ) -> bool:
        """
        Initialize the REPL with a repository and imports.

        Args:
            repo_path: Path to the Lean repository
            imports: Optional list of imports (e.g., ["Mathlib"])

        Returns:
            True if initialization succeeded
        """
        # Start the process
        if not await self.start(repo_path):
            return False

        # Process imports
        if imports:
            for imp in imports:
                cmd = f"import {imp}"
                if self._verbose:
                    print(f"[REPL] Importing {imp} (this may take a while for Mathlib)...")
                import time

                start_time = time.time()
                response = await self._send_command(cmd)
                elapsed = time.time() - start_time
                if self._verbose:
                    print(f"[REPL] Import {imp} completed in {elapsed:.1f}s")
                if response.has_error:
                    raise LeanError(parse_lean_error(response.error_message))
                self._state.imports.append(imp)

        return True

    async def enter_theorem(
        self,
        theorem_name: str,
        theorem_statement: str | None = None,
    ) -> ProofState:
        """
        Enter proof mode for a theorem.

        If theorem_statement is provided, declares a new theorem with a sorry.
        The sorry creates a proof state that we can then solve with tactics.

        Args:
            theorem_name: Name of the theorem
            theorem_statement: Optional type signature for new theorems

        Returns:
            Initial proof state with proof_state_id for tactic mode
        """
        if not self.is_ready:
            raise RuntimeError("REPL not initialized")

        if theorem_statement:
            # Declare a new theorem with sorry
            cmd = f"theorem {theorem_name} : {theorem_statement} := by\n  sorry"
        else:
            # Try to enter an existing theorem
            # This requires the theorem to exist in the environment
            cmd = f"#check @{theorem_name}"

        response = await self._send_command(cmd)

        if response.has_error:
            raise LeanError(parse_lean_error(response.error_message))

        # Extract proofState from the first sorry
        # The response format is: {"sorries": [{"proofState": 0, "goal": "...", ...}], "env": 1}
        proof_state_id = 0
        goals: list[str] = []
        pretty_state = ""

        if response.sorries:
            first_sorry = response.sorries[0]
            proof_state_id = first_sorry.get("proofState", 0)
            goal_text = first_sorry.get("goal", "")
            if goal_text:
                goals = [goal_text]
                pretty_state = goal_text

        # Create proof state with the proof_state_id for tactic mode
        state = ProofState(
            goals=goals,
            env_id=response.env_id,
            proof_state_id=proof_state_id,
            is_complete=len(goals) == 0,
            pretty_state=pretty_state,
        )

        # Track state
        self._state.current_theorem = theorem_name
        self._state.in_proof_mode = True
        self._state.current_env_id = response.env_id

        # Create state handle for backtracking
        self._state_manager.create_handle(
            env_id=response.env_id,
            goals=goals,
        )

        return state

    async def run_tactic(
        self,
        state: ProofState,
        tactic: str,
        timeout_ms: float | None = None,
    ) -> TacticResult:
        """
        Run a tactic in the current proof state.

        Uses the REPL's tactic mode: {"tactic": "...", "proofState": N}

        Args:
            state: Current proof state (must have valid proof_state_id)
            tactic: Tactic to apply
            timeout_ms: Optional timeout in milliseconds

        Returns:
            TacticResult with new state or error
        """
        if not self._state.in_proof_mode:
            return TacticResult(
                success=False,
                error=StructuredError(
                    category=LeanErrorCategory.INTERNAL,
                    message="Not in proof mode",
                ),
            )

        start_time = time.time()

        # Send tactic using tactic mode protocol
        response = await self._send_tactic(
            tactic,
            proof_state_id=state.proof_state_id,
            timeout_ms=timeout_ms,
        )

        elapsed_ms = (time.time() - start_time) * 1000

        if response.has_error:
            error = parse_lean_error(response.error_message, state.goals)
            return TacticResult(
                success=False,
                error=error,
                time_ms=elapsed_ms,
                env_id=state.env_id,  # Unchanged on error
            )

        # Check if proof is complete - goals is a list of strings in tactic mode
        goals = response.goals or []
        is_complete = len(goals) == 0

        # Get the new proof state ID from the response
        new_proof_state_id = (
            response.proof_state_id if response.proof_state_id is not None else state.proof_state_id
        )

        new_state = ProofState(
            goals=goals,
            env_id=state.env_id,  # env_id doesn't change in tactic mode
            proof_state_id=new_proof_state_id,
            is_complete=is_complete,
            pretty_state="\n".join(goals) if goals else "",
        )

        # Track state for backtracking
        parent_handle = self._state_manager.get_handle_by_env(state.env_id)
        self._state_manager.create_handle(
            env_id=state.env_id,
            goals=goals,
            parent=parent_handle,
            tactic=tactic,
            tactic_time_ms=elapsed_ms,
        )

        return TacticResult(
            success=True,
            new_state=new_state,
            time_ms=elapsed_ms,
            env_id=state.env_id,
        )

    async def backtrack(self, env_id: int) -> ProofState | None:
        """
        Backtrack to a previous proof state.

        This is O(1) because we just switch to a stored environment handle.

        Args:
            env_id: Environment ID to restore

        Returns:
            The restored proof state, or None if not found
        """
        handle = self._state_manager.get_handle_by_env(env_id)
        if handle is None:
            return None

        # The REPL allows us to reference previous environments
        # by including the env ID in commands
        self._state.current_env_id = env_id
        self._state_manager.set_current(handle)

        return ProofState(
            goals=handle.goals,
            hypotheses=handle.hypotheses,
            env_id=env_id,
            is_complete=handle.is_solved,
        )

    async def exit_proof(self) -> None:
        """Exit the current proof mode."""
        self._state.current_theorem = None
        self._state.in_proof_mode = False

    async def declare_theorem(
        self,
        name: str,
        statement: str,
        proof_tactics: list[str],
    ) -> bool:
        """
        Declare and prove a new theorem.

        This solves the "Ghost Theorem" problem by actually
        registering the theorem in the environment.

        Args:
            name: Theorem name
            statement: Type signature
            proof_tactics: List of tactics

        Returns:
            True if theorem was proved and added to environment
        """
        # Build the complete theorem declaration
        tactics_str = "\n  ".join(proof_tactics)
        cmd = f"theorem {name} : {statement} := by\n  {tactics_str}"

        response = await self._send_command(cmd)

        if response.has_error:
            return False

        # Theorem is now in the environment and can be used as a premise
        self._state.current_env_id = response.env_id
        return True

    async def check_expr(self, expr: str) -> tuple[bool, str]:
        """
        Check if an expression is well-typed.

        Useful for validating hallucinated premises.

        Args:
            expr: Expression to check

        Returns:
            (is_valid, type_or_error)
        """
        response = await self._send_command(f"#check {expr}")

        if response.has_error:
            return False, response.error_message

        # Extract type from response
        # Format: "expr : type"
        for msg in response.messages:
            if "data" in msg:
                return True, msg["data"]

        return True, ""

    async def get_constants(self) -> AsyncIterator[str]:
        """
        Get all constants in the current environment.

        Useful for building the premise database.
        """
        # This requires a custom REPL command or metaprogramming
        # For now, return empty - would need REPL extension
        return
        yield  # Make this a generator

    async def close(self) -> None:
        """Close the REPL process."""
        if self._process:
            self._process.terminate()
            try:
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._process.kill()
            self._process = None

        self._state.status = REPLStatus.TERMINATED
        self._state_manager.clear()

    async def _send_command(
        self,
        command: str,
        env_id: int | None = None,
        timeout_ms: float | None = None,
    ) -> REPLResponse:
        """
        Send a command to the REPL and get the response.

        Commands are sent as JSON:
        {
            "cmd": "<command>",
            "env": <env_id>  // Optional: reference previous environment
        }
        """
        async with self._lock:
            if not self._process or self._process.poll() is not None:
                return REPLResponse(has_error=True, error_message="REPL process not running")

            self._state.status = REPLStatus.BUSY

            try:
                # Build JSON command
                cmd_obj: dict[str, Any] = {"cmd": command}
                if env_id is not None:
                    cmd_obj["env"] = env_id

                # Commands must be terminated by a blank line (double newline)
                # per the REPL protocol: "Commands should be separated by blank lines."
                cmd_json = json.dumps(cmd_obj) + "\n\n"

                if self._verbose:
                    print(f"[REPL] > {cmd_json.strip()}")

                # Send command
                self._process.stdin.write(cmd_json)
                self._process.stdin.flush()

                # Read response
                response = await self._read_response(timeout_ms)

                self._state.commands_executed += 1
                self._state.status = REPLStatus.READY

                return response

            except Exception as e:
                self._state.status = REPLStatus.ERROR
                return REPLResponse(has_error=True, error_message=str(e))

    async def _send_tactic(
        self,
        tactic: str,
        proof_state_id: int,
        timeout_ms: float | None = None,
    ) -> REPLResponse:
        """
        Send a tactic to the REPL in tactic mode and get the response.

        Tactics are sent as JSON:
        {
            "tactic": "<tactic>",
            "proofState": <proof_state_id>
        }

        Response format:
        {
            "proofState": <new_id>,
            "goals": ["goal1", "goal2", ...]
        }
        """
        async with self._lock:
            if not self._process or self._process.poll() is not None:
                return REPLResponse(has_error=True, error_message="REPL process not running")

            self._state.status = REPLStatus.BUSY

            try:
                # Build JSON tactic command
                tactic_obj: dict[str, Any] = {
                    "tactic": tactic,
                    "proofState": proof_state_id,
                }

                # Commands must be terminated by a blank line (double newline)
                tactic_json = json.dumps(tactic_obj) + "\n\n"

                if self._verbose:
                    print(f"[REPL] > {tactic_json.strip()}")

                # Send tactic
                self._process.stdin.write(tactic_json)
                self._process.stdin.flush()

                # Read response
                response = await self._read_response(timeout_ms)

                self._state.commands_executed += 1
                self._state.status = REPLStatus.READY

                return response

            except Exception as e:
                self._state.status = REPLStatus.ERROR
                return REPLResponse(has_error=True, error_message=str(e))

    async def _read_response(self, timeout_ms: float | None = None) -> REPLResponse:
        """
        Read a JSON response from the REPL.

        The REPL returns multi-line JSON responses terminated by a blank line.
        We accumulate lines until we hit a blank line, then parse the complete JSON.
        """
        if not self._process:
            return REPLResponse(has_error=True, error_message="No process")

        timeout = (timeout_ms / 1000) if timeout_ms else self._timeout
        loop = asyncio.get_event_loop()

        def read_line():
            return self._process.stdout.readline()

        try:
            # Accumulate lines until we hit a blank line (response terminator)
            response_lines: list[str] = []
            start_time = time.time()

            while True:
                # Calculate remaining timeout
                elapsed = time.time() - start_time
                remaining_timeout = timeout - elapsed
                if remaining_timeout <= 0:
                    return REPLResponse(
                        has_error=True,
                        error_message="Timeout waiting for REPL response",
                    )

                line = await asyncio.wait_for(
                    loop.run_in_executor(None, read_line),
                    timeout=remaining_timeout,
                )

                if not line:
                    # EOF reached
                    break

                # Blank line signals end of response
                if line.strip() == "":
                    break

                response_lines.append(line)

            if not response_lines:
                return REPLResponse(has_error=True, error_message="Empty response")

            # Join lines and parse JSON
            full_response = "".join(response_lines)

            if self._verbose:
                print(f"[REPL] < {full_response.strip()}")

            data = json.loads(full_response)
            return REPLResponse.from_json(data)

        except asyncio.TimeoutError:
            return REPLResponse(
                has_error=True,
                error_message="Timeout waiting for REPL response",
            )
        except json.JSONDecodeError as e:
            return REPLResponse(
                has_error=True,
                error_message=f"Invalid JSON from REPL: {e}",
            )


def create_repl_client(
    repl_path: str | None = None,
    timeout_seconds: float = 60.0,
    verbose: bool = False,
) -> LeanREPLClient:
    """
    Factory to create a REPL client.

    Args:
        repl_path: Path to repl executable
        timeout_seconds: Default command timeout
        verbose: Enable debug output

    Returns:
        Configured LeanREPLClient
    """
    return LeanREPLClient(
        repl_path=repl_path,
        timeout_seconds=timeout_seconds,
        verbose=verbose,
    )
