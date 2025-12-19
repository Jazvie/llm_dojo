"""
Structured error parsing for Lean 4 REPL responses.

Implements error categorization essential for:
- Reflexion loops (analyzing errors to generate corrected tactics)
- Hallucination detection (unknown identifier errors)
- Search guidance (distinguishing recoverable vs fatal errors)

Error categories align with Lean's internal error types:
- Syntax errors: Parsing failures
- Type errors: Type mismatches, universe issues
- Tactic errors: Failed unification, no applicable tactics
- Unknown identifier: Hallucinated premises
- Timeout: Resource exhaustion
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any


class LeanErrorCategory(Enum):
    """
    Categorization of Lean errors for AI feedback loops.

    These categories determine how the agent should respond:
    - SYNTAX: Fix the tactic string format
    - TYPE_MISMATCH: Wrong type, try different approach
    - UNKNOWN_IDENTIFIER: Hallucinated premise, remove or prove it
    - TACTIC_FAILED: Tactic doesn't apply, try alternatives
    - TIMEOUT: Resource limit, simplify or abort branch
    - GOAL_NOT_PROVABLE: May indicate false conjecture
    - INTERNAL: Lean bug or system issue
    """

    SYNTAX = auto()
    TYPE_MISMATCH = auto()
    UNKNOWN_IDENTIFIER = auto()
    UNKNOWN_TACTIC = auto()
    TACTIC_FAILED = auto()
    UNIFICATION_FAILED = auto()
    TIMEOUT = auto()
    DETERMINISTIC_TIMEOUT = auto()
    GOAL_NOT_PROVABLE = auto()
    INTERNAL = auto()
    UNKNOWN = auto()


@dataclass
class StructuredError:
    """
    A structured representation of a Lean error.

    Provides rich information for AI feedback:
    - category: What kind of error (for routing)
    - message: Human-readable description
    - location: Where in the tactic the error occurred
    - suggestions: Possible fixes from Lean's error messages
    - context: Additional diagnostic information
    """

    category: LeanErrorCategory
    message: str
    raw_message: str = ""

    # Location information
    line: int | None = None
    column: int | None = None
    end_line: int | None = None
    end_column: int | None = None

    # For unknown identifier errors
    unknown_name: str | None = None
    similar_names: list[str] = field(default_factory=list)

    # For type errors
    expected_type: str | None = None
    actual_type: str | None = None

    # For tactic failures
    goals_before: list[str] = field(default_factory=list)

    # Suggestions from Lean
    suggestions: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "category": self.category.name,
            "message": self.message,
            "raw_message": self.raw_message,
            "location": {
                "line": self.line,
                "column": self.column,
            }
            if self.line
            else None,
            "unknown_name": self.unknown_name,
            "similar_names": self.similar_names,
            "expected_type": self.expected_type,
            "actual_type": self.actual_type,
            "suggestions": self.suggestions,
        }

    @property
    def is_recoverable(self) -> bool:
        """Whether this error suggests trying alternative tactics."""
        return self.category in {
            LeanErrorCategory.TACTIC_FAILED,
            LeanErrorCategory.UNIFICATION_FAILED,
            LeanErrorCategory.TYPE_MISMATCH,
        }

    @property
    def is_hallucination(self) -> bool:
        """Whether this error indicates a hallucinated premise."""
        return self.category in {
            LeanErrorCategory.UNKNOWN_IDENTIFIER,
            LeanErrorCategory.UNKNOWN_TACTIC,
        }

    @property
    def feedback_for_agent(self) -> str:
        """Generate feedback string for the AI agent."""
        if self.category == LeanErrorCategory.UNKNOWN_IDENTIFIER:
            msg = f"Unknown identifier: '{self.unknown_name}'"
            if self.similar_names:
                msg += f". Did you mean: {', '.join(self.similar_names[:3])}?"
            return msg

        if self.category == LeanErrorCategory.TYPE_MISMATCH:
            if self.expected_type and self.actual_type:
                return f"Type mismatch: expected {self.expected_type}, got {self.actual_type}"
            return f"Type mismatch: {self.message}"

        if self.category == LeanErrorCategory.TACTIC_FAILED:
            return f"Tactic failed: {self.message}"

        if self.category == LeanErrorCategory.SYNTAX:
            return f"Syntax error: {self.message}"

        return self.message


class LeanError(Exception):
    """Exception wrapping a structured Lean error."""

    def __init__(self, structured: StructuredError):
        self.structured = structured
        super().__init__(structured.message)

    @property
    def category(self) -> LeanErrorCategory:
        return self.structured.category


# Regex patterns for parsing Lean error messages
_PATTERNS = {
    # Unknown identifier: "unknown identifier 'foo'"
    "unknown_id": re.compile(
        r"unknown (?:identifier|constant|declaration) ['\"]?(\w+(?:\.\w+)*)['\"]?", re.IGNORECASE
    ),
    # Unknown tactic: "unknown tactic 'foo'"
    "unknown_tactic": re.compile(r"unknown tactic ['\"]?(\w+)['\"]?", re.IGNORECASE),
    # Type mismatch: "type mismatch ... has type ... but is expected to have type"
    "type_mismatch": re.compile(
        r"type mismatch.*?has type\s+(.+?)\s+but is expected to have type\s+(.+)",
        re.IGNORECASE | re.DOTALL,
    ),
    # Alternative type mismatch format
    "type_mismatch_alt": re.compile(r"expected\s+(.+?),?\s+got\s+(.+)", re.IGNORECASE),
    # Tactic failed: "tactic 'foo' failed"
    "tactic_failed": re.compile(r"tactic ['\"]?(\w+)['\"]? failed", re.IGNORECASE),
    # Unification failed
    "unification": re.compile(
        r"(?:failed to unify|cannot unify|unification failed)", re.IGNORECASE
    ),
    # Syntax error
    "syntax": re.compile(r"(?:syntax error|unexpected token|expected|parse error)", re.IGNORECASE),
    # Timeout
    "timeout": re.compile(
        r"(?:timeout|time limit|maximum recursion depth|deterministic timeout)", re.IGNORECASE
    ),
    # Did you mean suggestions
    "did_you_mean": re.compile(r"did you mean ['\"]?(\w+(?:\.\w+)*)['\"]?", re.IGNORECASE),
    # Multiple suggestions: "unknown ... (possible matches: a, b, c)"
    "possible_matches": re.compile(r"possible (?:matches|alternatives):\s*(.+)", re.IGNORECASE),
}


def parse_lean_error(error_text: str, goals: list[str] | None = None) -> StructuredError:
    """
    Parse a Lean error message into a structured format.

    This is the main entry point for error categorization.

    Args:
        error_text: Raw error message from Lean
        goals: Current goal state (for context)

    Returns:
        StructuredError with category and extracted information
    """
    error_text = error_text.strip()
    lower_text = error_text.lower()

    # Check for unknown identifier (hallucination)
    if match := _PATTERNS["unknown_id"].search(error_text):
        unknown_name = match.group(1)
        similar = _extract_suggestions(error_text)
        return StructuredError(
            category=LeanErrorCategory.UNKNOWN_IDENTIFIER,
            message=f"Unknown identifier: {unknown_name}",
            raw_message=error_text,
            unknown_name=unknown_name,
            similar_names=similar,
            goals_before=goals or [],
        )

    # Check for unknown tactic
    if match := _PATTERNS["unknown_tactic"].search(error_text):
        tactic_name = match.group(1)
        return StructuredError(
            category=LeanErrorCategory.UNKNOWN_TACTIC,
            message=f"Unknown tactic: {tactic_name}",
            raw_message=error_text,
            unknown_name=tactic_name,
        )

    # Check for type mismatch
    if match := _PATTERNS["type_mismatch"].search(error_text):
        return StructuredError(
            category=LeanErrorCategory.TYPE_MISMATCH,
            message="Type mismatch",
            raw_message=error_text,
            actual_type=match.group(1).strip(),
            expected_type=match.group(2).strip(),
            goals_before=goals or [],
        )

    if match := _PATTERNS["type_mismatch_alt"].search(error_text):
        return StructuredError(
            category=LeanErrorCategory.TYPE_MISMATCH,
            message="Type mismatch",
            raw_message=error_text,
            expected_type=match.group(1).strip(),
            actual_type=match.group(2).strip(),
            goals_before=goals or [],
        )

    # Check for unification failure
    if _PATTERNS["unification"].search(error_text):
        return StructuredError(
            category=LeanErrorCategory.UNIFICATION_FAILED,
            message="Unification failed",
            raw_message=error_text,
            goals_before=goals or [],
        )

    # Check for syntax error
    if _PATTERNS["syntax"].search(error_text):
        return StructuredError(
            category=LeanErrorCategory.SYNTAX,
            message=_extract_first_line(error_text),
            raw_message=error_text,
        )

    # Check for timeout
    if _PATTERNS["timeout"].search(error_text):
        is_deterministic = "deterministic" in lower_text
        return StructuredError(
            category=(
                LeanErrorCategory.DETERMINISTIC_TIMEOUT
                if is_deterministic
                else LeanErrorCategory.TIMEOUT
            ),
            message="Timeout exceeded",
            raw_message=error_text,
        )

    # Check for tactic failure
    if match := _PATTERNS["tactic_failed"].search(error_text):
        return StructuredError(
            category=LeanErrorCategory.TACTIC_FAILED,
            message=f"Tactic '{match.group(1)}' failed",
            raw_message=error_text,
            goals_before=goals or [],
        )

    # Generic tactic failure patterns
    if any(kw in lower_text for kw in ["failed", "no goals", "goals accomplished"]):
        return StructuredError(
            category=LeanErrorCategory.TACTIC_FAILED,
            message=_extract_first_line(error_text),
            raw_message=error_text,
            goals_before=goals or [],
        )

    # Unknown error
    return StructuredError(
        category=LeanErrorCategory.UNKNOWN,
        message=_extract_first_line(error_text),
        raw_message=error_text,
        goals_before=goals or [],
    )


def _extract_suggestions(error_text: str) -> list[str]:
    """Extract 'did you mean' suggestions from error text."""
    suggestions = []

    # Single suggestion
    for match in _PATTERNS["did_you_mean"].finditer(error_text):
        suggestions.append(match.group(1))

    # Multiple suggestions
    if match := _PATTERNS["possible_matches"].search(error_text):
        matches_text = match.group(1)
        # Split by comma or whitespace
        for name in re.split(r"[,\s]+", matches_text):
            name = name.strip("'\"")
            if name and name not in suggestions:
                suggestions.append(name)

    return suggestions


def _extract_first_line(text: str) -> str:
    """Extract the first meaningful line from error text."""
    for line in text.split("\n"):
        line = line.strip()
        if line and not line.startswith("error:"):
            return line
    return text.split("\n")[0].strip() if text else "Unknown error"


def categorize_for_search(error: StructuredError) -> str:
    """
    Return a category string for search algorithm decisions.

    Returns:
        "backtrack": Try different tactic at this state
        "prune": Abandon this branch entirely
        "retry": Retry with modification
        "fatal": Stop the proof attempt
    """
    if error.category in {
        LeanErrorCategory.TACTIC_FAILED,
        LeanErrorCategory.UNIFICATION_FAILED,
        LeanErrorCategory.TYPE_MISMATCH,
    }:
        return "backtrack"

    if error.category == LeanErrorCategory.UNKNOWN_IDENTIFIER:
        return "prune"  # Hallucinated premise

    if error.category == LeanErrorCategory.SYNTAX:
        return "retry"  # Fix syntax and retry

    if error.category in {
        LeanErrorCategory.TIMEOUT,
        LeanErrorCategory.DETERMINISTIC_TIMEOUT,
    }:
        return "prune"  # Resource exhaustion

    if error.category == LeanErrorCategory.INTERNAL:
        return "fatal"

    return "backtrack"
