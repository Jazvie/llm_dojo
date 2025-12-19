"""
Rulebook: Modular definition of the proving environment.

The Rulebook defines what tactics, premises, and types are available
to agents. This allows for:
- Consistent rules known to all agents upfront
- Easy difficulty scaling (add/remove tactics)
- Reproducible experiments

Philosophy:
- Rulebook = global playing field (what tactics/types CAN be used)
- All agents must only use tactics defined in the Rulebook
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any


class TacticCategory(Enum):
    """Categories of tactics for organization and difficulty scaling."""

    # Finishing tactics - close goals directly
    CLOSING = auto()  # rfl, trivial, decide, native_decide

    # Simplification - automated rewriting
    SIMPLIFICATION = auto()  # simp, simp_all, norm_num, ring, field_simp

    # Arithmetic solvers - decision procedures
    ARITHMETIC = auto()  # omega, linarith, polyrith, nlinarith

    # Structural - manipulate goal/hypothesis structure
    STRUCTURAL = auto()  # intro, cases, induction, constructor, etc.

    # Rewriting - explicit term rewriting
    REWRITING = auto()  # rw, rewrite, conv, unfold

    # Search - automated proof search
    SEARCH = auto()  # aesop, exact?, apply?, rfl?

    # Advanced - require more expertise
    ADVANCED = auto()  # rcases, obtain, calc, have, let, suffices

    # Meta - tactic combinators and control flow
    META = auto()  # repeat, first, all_goals, any_goals, try, <;>


@dataclass
class TacticInfo:
    """Information about a single tactic."""

    name: str
    category: TacticCategory
    description: str = ""
    example: str = ""  # Example usage
    requires_args: bool = False  # Does it need arguments?
    common_args: list[str] = field(default_factory=list)  # Common argument patterns


@dataclass
class TacticSet:
    """
    A set of available tactics.

    Tactics are grouped by category for:
    - LLM prompting (show relevant tactics for goal type)
    - Difficulty scaling (enable/disable categories)
    - Documentation generation
    """

    tactics: dict[str, TacticInfo] = field(default_factory=dict)

    def add(self, tactic: TacticInfo) -> None:
        """Add a tactic to the set."""
        self.tactics[tactic.name] = tactic

    def get(self, name: str) -> TacticInfo | None:
        """Get tactic info by name."""
        return self.tactics.get(name)

    def by_category(self, category: TacticCategory) -> list[TacticInfo]:
        """Get all tactics in a category."""
        return [t for t in self.tactics.values() if t.category == category]

    def all_names(self) -> list[str]:
        """Get all tactic names."""
        return list(self.tactics.keys())

    def is_available(self, name: str) -> bool:
        """Check if a tactic is available."""
        # Handle tactics with arguments (e.g., "rw [foo]" -> check "rw")
        base_name = name.split()[0] if name else ""
        return base_name in self.tactics

    def to_prompt_text(self, categories: list[TacticCategory] | None = None) -> str:
        """
        Generate prompt text describing available tactics.

        Args:
            categories: Optional filter for specific categories
        """
        lines = ["Available tactics:"]

        cats = categories or list(TacticCategory)
        for cat in cats:
            tactics = self.by_category(cat)
            if tactics:
                lines.append(f"\n{cat.name}:")
                for t in tactics:
                    if t.example:
                        lines.append(f"  - {t.name}: {t.description} (e.g., `{t.example}`)")
                    else:
                        lines.append(f"  - {t.name}: {t.description}")

        return "\n".join(lines)


@dataclass
class PremiseInfo:
    """Information about an available premise (lemma/theorem)."""

    name: str
    type_signature: str
    description: str = ""
    module: str = ""  # e.g., "Mathlib.Algebra.Ring.Basic"


@dataclass
class PremiseSet:
    """
    A set of available premises.

    Premises are lemmas/theorems that can be used in proofs.
    This enables RAG-style retrieval and explicit premise hints.
    """

    premises: dict[str, PremiseInfo] = field(default_factory=dict)
    modules: set[str] = field(default_factory=set)  # Imported modules

    def add(self, premise: PremiseInfo) -> None:
        """Add a premise."""
        self.premises[premise.name] = premise
        if premise.module:
            self.modules.add(premise.module)

    def search(self, query: str, limit: int = 10) -> list[PremiseInfo]:
        """
        Search for premises by name or description.

        Note: For production, this should use vector similarity.
        """
        query_lower = query.lower()
        matches = [
            p
            for p in self.premises.values()
            if query_lower in p.name.lower() or query_lower in p.description.lower()
        ]
        return matches[:limit]

    def to_prompt_text(self, relevant: list[str] | None = None) -> str:
        """Generate prompt text for relevant premises."""
        if relevant:
            lines = ["Potentially useful lemmas:"]
            for name in relevant:
                if name in self.premises:
                    p = self.premises[name]
                    lines.append(f"  - {p.name} : {p.type_signature}")
            return "\n".join(lines)
        return ""


@dataclass
class TypeInfo:
    """Information about an available type."""

    name: str
    lean_name: str  # The actual Lean name (e.g., "Nat" not "N")
    description: str = ""
    operations: list[str] = field(default_factory=list)  # Available operations


@dataclass
class TypeSet:
    """
    A set of available types.

    Defines what mathematical objects agents can work with.
    """

    types: dict[str, TypeInfo] = field(default_factory=dict)

    def add(self, type_info: TypeInfo) -> None:
        """Add a type."""
        self.types[type_info.name] = type_info

    def all_lean_names(self) -> list[str]:
        """Get all Lean type names."""
        return [t.lean_name for t in self.types.values()]

    def to_prompt_text(self) -> str:
        """Generate prompt text describing available types."""
        lines = ["Available types:"]
        for t in self.types.values():
            ops = ", ".join(t.operations) if t.operations else "standard"
            lines.append(f"  - {t.lean_name}: {t.description} (operations: {ops})")
        return "\n".join(lines)


@dataclass
class Rulebook:
    """
    Complete definition of the proving environment.

    The Rulebook is the "game manual" - it tells agents exactly
    what they can use. This is communicated to LLMs in prompts
    and used for validation.

    Key principle: Everything in the Rulebook is ALLOWED.
    Agents can only use tactics defined in the Rulebook.
    """

    name: str = "default"
    description: str = ""

    tactics: TacticSet = field(default_factory=TacticSet)
    premises: PremiseSet = field(default_factory=PremiseSet)
    types: TypeSet = field(default_factory=TypeSet)

    # Lean imports required for this rulebook
    imports: list[str] = field(default_factory=list)

    # Optional: max proof depth/steps as a global limit
    max_steps: int | None = None
    max_depth: int | None = None

    def to_system_prompt(self) -> str:
        """
        Generate a system prompt section describing the rules.

        This is included in LLM prompts so agents know what's available.
        """
        sections = [
            f"# Proving Environment: {self.name}",
            "",
            self.description if self.description else "Standard Lean 4 proving environment.",
            "",
            self.tactics.to_prompt_text(),
            "",
            self.types.to_prompt_text(),
        ]

        if self.max_steps:
            sections.append(f"\nProof step limit: {self.max_steps}")

        return "\n".join(sections)

    def validate_tactic(self, tactic: str) -> tuple[bool, str | None]:
        """
        Check if a tactic is allowed by this rulebook.

        Returns:
            (is_valid, error_message)
        """
        if self.tactics.is_available(tactic):
            return True, None
        else:
            base_name = tactic.split()[0] if tactic else ""
            return False, f"Tactic '{base_name}' is not in the rulebook"

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict for storage/transmission."""
        return {
            "name": self.name,
            "description": self.description,
            "tactics": [t.name for t in self.tactics.tactics.values()],
            "types": [t.lean_name for t in self.types.types.values()],
            "imports": self.imports,
            "max_steps": self.max_steps,
        }


# =============================================================================
# Preset Rulebooks
# =============================================================================


def _create_basic_tactics() -> TacticSet:
    """Create the basic tactic set available in core Lean 4."""
    ts = TacticSet()

    # Closing tactics
    ts.add(TacticInfo("rfl", TacticCategory.CLOSING, "Reflexivity - proves a = a"))
    ts.add(TacticInfo("trivial", TacticCategory.CLOSING, "Solves trivial goals"))
    ts.add(TacticInfo("decide", TacticCategory.CLOSING, "Decides decidable propositions"))
    ts.add(TacticInfo("assumption", TacticCategory.CLOSING, "Uses a hypothesis directly"))

    # Structural tactics
    ts.add(
        TacticInfo(
            "intro",
            TacticCategory.STRUCTURAL,
            "Introduce a variable/hypothesis",
            example="intro x",
            requires_args=True,
        )
    )
    ts.add(
        TacticInfo(
            "intros",
            TacticCategory.STRUCTURAL,
            "Introduce multiple variables",
            example="intros x y z",
        )
    )
    ts.add(
        TacticInfo(
            "cases",
            TacticCategory.STRUCTURAL,
            "Case split on an inductive type",
            example="cases h",
            requires_args=True,
        )
    )
    ts.add(
        TacticInfo(
            "induction",
            TacticCategory.STRUCTURAL,
            "Induction on a variable",
            example="induction n with d hd",
            requires_args=True,
        )
    )
    ts.add(TacticInfo("constructor", TacticCategory.STRUCTURAL, "Apply constructor of goal type"))
    ts.add(TacticInfo("left", TacticCategory.STRUCTURAL, "Choose left side of disjunction"))
    ts.add(TacticInfo("right", TacticCategory.STRUCTURAL, "Choose right side of disjunction"))
    ts.add(
        TacticInfo(
            "exists",
            TacticCategory.STRUCTURAL,
            "Provide witness for existential",
            example="exists 0",
            requires_args=True,
        )
    )
    ts.add(
        TacticInfo(
            "use",
            TacticCategory.STRUCTURAL,
            "Provide witness (Mathlib style)",
            example="use 42",
            requires_args=True,
        )
    )
    ts.add(
        TacticInfo(
            "exact",
            TacticCategory.STRUCTURAL,
            "Provide exact proof term",
            example="exact h",
            requires_args=True,
        )
    )
    ts.add(
        TacticInfo(
            "apply",
            TacticCategory.STRUCTURAL,
            "Apply a lemma/hypothesis",
            example="apply Nat.add_comm",
            requires_args=True,
        )
    )
    ts.add(
        TacticInfo(
            "refine",
            TacticCategory.STRUCTURAL,
            "Partial proof term with holes",
            example="refine ?_ + ?_",
            requires_args=True,
        )
    )

    # Rewriting tactics
    ts.add(
        TacticInfo(
            "rw",
            TacticCategory.REWRITING,
            "Rewrite using equality",
            example="rw [Nat.add_comm]",
            requires_args=True,
        )
    )
    ts.add(
        TacticInfo(
            "simp",
            TacticCategory.SIMPLIFICATION,
            "Simplification using simp lemmas",
            example="simp",
        )
    )
    ts.add(
        TacticInfo(
            "simp only",
            TacticCategory.SIMPLIFICATION,
            "Simplify with specific lemmas only",
            example="simp only [Nat.add_zero]",
            requires_args=True,
        )
    )

    return ts


def _create_mathlib_tactics() -> TacticSet:
    """Create tactic set with Mathlib tactics."""
    ts = _create_basic_tactics()

    # Arithmetic solvers
    ts.add(TacticInfo("omega", TacticCategory.ARITHMETIC, "Linear integer arithmetic solver"))
    ts.add(
        TacticInfo("linarith", TacticCategory.ARITHMETIC, "Linear arithmetic over ordered rings")
    )
    ts.add(
        TacticInfo(
            "nlinarith",
            TacticCategory.ARITHMETIC,
            "Nonlinear arithmetic (limited)",
        )
    )
    ts.add(
        TacticInfo(
            "polyrith",
            TacticCategory.ARITHMETIC,
            "Polynomial arithmetic (requires network)",
        )
    )
    ts.add(TacticInfo("norm_num", TacticCategory.ARITHMETIC, "Numeric normalization"))
    ts.add(TacticInfo("ring", TacticCategory.ARITHMETIC, "Ring equation solver"))
    ts.add(TacticInfo("ring_nf", TacticCategory.ARITHMETIC, "Ring normal form"))
    ts.add(TacticInfo("field_simp", TacticCategory.SIMPLIFICATION, "Simplify field expressions"))

    # Search tactics
    ts.add(TacticInfo("aesop", TacticCategory.SEARCH, "Automated proof search"))
    ts.add(TacticInfo("exact?", TacticCategory.SEARCH, "Search for exact proof"))
    ts.add(TacticInfo("apply?", TacticCategory.SEARCH, "Search for applicable lemma"))
    ts.add(TacticInfo("rw?", TacticCategory.SEARCH, "Search for rewrite lemma"))

    # Advanced tactics
    ts.add(
        TacticInfo(
            "rcases",
            TacticCategory.ADVANCED,
            "Recursive case split",
            example="rcases h with ⟨a, b, c⟩",
            requires_args=True,
        )
    )
    ts.add(
        TacticInfo(
            "obtain",
            TacticCategory.ADVANCED,
            "Destructure and introduce",
            example="obtain ⟨x, hx⟩ := h",
            requires_args=True,
        )
    )
    ts.add(
        TacticInfo(
            "have",
            TacticCategory.ADVANCED,
            "Introduce intermediate goal",
            example="have h : P := by ...",
            requires_args=True,
        )
    )
    ts.add(
        TacticInfo(
            "let",
            TacticCategory.ADVANCED,
            "Local definition",
            example="let x := 5",
            requires_args=True,
        )
    )
    ts.add(
        TacticInfo(
            "suffices",
            TacticCategory.ADVANCED,
            "It suffices to show",
            example="suffices h : P by exact h",
            requires_args=True,
        )
    )
    ts.add(
        TacticInfo(
            "calc",
            TacticCategory.ADVANCED,
            "Calculational proof",
            requires_args=True,
        )
    )
    ts.add(
        TacticInfo(
            "conv",
            TacticCategory.REWRITING,
            "Targeted rewriting",
            example="conv => lhs; rw [h]",
            requires_args=True,
        )
    )
    ts.add(TacticInfo("push_neg", TacticCategory.SIMPLIFICATION, "Push negations inward"))
    ts.add(TacticInfo("contrapose", TacticCategory.STRUCTURAL, "Prove contrapositive"))
    ts.add(TacticInfo("by_contra", TacticCategory.STRUCTURAL, "Proof by contradiction"))
    ts.add(
        TacticInfo(
            "by_cases",
            TacticCategory.STRUCTURAL,
            "Case split on decidable prop",
            example="by_cases h : P",
            requires_args=True,
        )
    )

    # Meta/combinators
    ts.add(
        TacticInfo(
            "repeat",
            TacticCategory.META,
            "Repeat tactic until failure",
            example="repeat rfl",
            requires_args=True,
        )
    )
    ts.add(
        TacticInfo(
            "first",
            TacticCategory.META,
            "Try tactics in order",
            example="first | rfl | simp",
            requires_args=True,
        )
    )
    ts.add(
        TacticInfo(
            "all_goals",
            TacticCategory.META,
            "Apply to all goals",
            example="all_goals simp",
            requires_args=True,
        )
    )
    ts.add(TacticInfo("try", TacticCategory.META, "Try tactic, don't fail", example="try simp"))

    return ts


def _create_basic_types() -> TypeSet:
    """Create basic type set."""
    ts = TypeSet()

    ts.add(
        TypeInfo(
            "natural",
            "Nat",
            "Natural numbers (0, 1, 2, ...)",
            operations=["+", "-", "*", "/", "%", "^", "<", "<=", "="],
        )
    )
    ts.add(
        TypeInfo(
            "integer",
            "Int",
            "Integers (..., -1, 0, 1, ...)",
            operations=["+", "-", "*", "/", "%", "^", "<", "<=", "=", "neg"],
        )
    )
    ts.add(
        TypeInfo(
            "rational",
            "Rat",
            "Rational numbers",
            operations=["+", "-", "*", "/", "^", "<", "<=", "=", "neg", "inv"],
        )
    )
    ts.add(
        TypeInfo(
            "bool",
            "Bool",
            "Boolean values",
            operations=["&&", "||", "!", "=="],
        )
    )
    ts.add(
        TypeInfo(
            "list",
            "List",
            "Polymorphic lists",
            operations=["++", "::", "length", "map", "filter"],
        )
    )

    return ts


def _create_extended_types() -> TypeSet:
    """Create extended type set with Mathlib types."""
    ts = _create_basic_types()

    ts.add(
        TypeInfo(
            "real",
            "Real",
            "Real numbers",
            operations=["+", "-", "*", "/", "^", "<", "<=", "=", "neg", "inv", "abs"],
        )
    )
    ts.add(
        TypeInfo(
            "complex",
            "Complex",
            "Complex numbers",
            operations=["+", "-", "*", "/", "^", "=", "neg", "inv", "conj", "abs"],
        )
    )
    ts.add(
        TypeInfo(
            "finset",
            "Finset",
            "Finite sets",
            operations=["union", "inter", "card", "sum", "prod"],
        )
    )

    return ts


def create_basic_rulebook() -> Rulebook:
    """
    Create a basic rulebook with core Lean 4 tactics.

    Good for simple arithmetic and logic proofs.
    """
    return Rulebook(
        name="basic",
        description="Core Lean 4 tactics without Mathlib automation.",
        tactics=_create_basic_tactics(),
        types=_create_basic_types(),
        imports=[],  # No imports needed for core Lean
    )


def create_mathlib_rulebook() -> Rulebook:
    """
    Create a full Mathlib rulebook.

    This is the standard environment for mathematical proving.
    """
    return Rulebook(
        name="mathlib",
        description="Full Mathlib environment with all tactics and types.",
        tactics=_create_mathlib_tactics(),
        types=_create_extended_types(),
        imports=["Mathlib"],
    )


def create_competition_rulebook() -> Rulebook:
    """
    Create a competition-style rulebook.

    Disables heavy automation (aesop, exact?) to require more skill.
    """
    tactics = _create_mathlib_tactics()

    # Remove search tactics
    del tactics.tactics["aesop"]
    del tactics.tactics["exact?"]
    del tactics.tactics["apply?"]
    del tactics.tactics["rw?"]

    return Rulebook(
        name="competition",
        description="Competition mode: no automated search tactics.",
        tactics=tactics,
        types=_create_extended_types(),
        imports=["Mathlib"],
    )


def create_custom_rulebook(
    name: str,
    allowed_tactics: list[str],
    allowed_types: list[str] | None = None,
    imports: list[str] | None = None,
    max_steps: int | None = None,
) -> Rulebook:
    """
    Create a custom rulebook with specific tactics.

    Args:
        name: Rulebook name
        allowed_tactics: List of tactic names to include
        allowed_types: List of type names (defaults to basic types)
        imports: Required imports
        max_steps: Optional step limit
    """
    # Start with full mathlib tactics, then filter
    full_tactics = _create_mathlib_tactics()
    filtered = TacticSet()

    for tactic_name in allowed_tactics:
        if tactic_name in full_tactics.tactics:
            filtered.add(full_tactics.tactics[tactic_name])
        else:
            # Allow unknown tactics (user might know what they're doing)
            filtered.add(
                TacticInfo(
                    tactic_name,
                    TacticCategory.ADVANCED,
                    f"Custom tactic: {tactic_name}",
                )
            )

    return Rulebook(
        name=name,
        description=f"Custom rulebook: {name}",
        tactics=filtered,
        types=_create_extended_types()
        if "Real" in (allowed_types or [])
        else _create_basic_types(),
        imports=imports or ["Mathlib"],
        max_steps=max_steps,
    )


# Convenience: default rulebook
DEFAULT_RULEBOOK = create_mathlib_rulebook()
