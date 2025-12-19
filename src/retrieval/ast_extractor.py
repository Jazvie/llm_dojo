"""
AST-based premise extraction for Lean 4.

Implements the SOTA approach for premise analysis:
- Expr traversal using foldConsts pattern
- Precise identification of used constants (not regex)
- Dependency graph construction

This replaces the naive text-based approach that fails to capture:
- Implicit arguments
- Lemmas invoked by tactics like simp, linarith
- Type class instances

The key insight from the architecture doc is that Expr.foldConsts
traverses the elaborated proof term to collect every Expr.const node,
giving 100% precision in identifying used premises.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Iterator, Set


class ExprKind(Enum):
    """
    Lean 4 Expr constructors.

    These match the Lean.Expr type:
    - bvar: Bound variable (De Bruijn index)
    - fvar: Free variable (hypothesis)
    - const: Global constant (theorem, def, axiom)
    - app: Function application
    - lam: Lambda abstraction
    - forallE: Pi type / forall
    - etc.
    """

    BVAR = auto()  # Bound variable
    FVAR = auto()  # Free variable
    MVAR = auto()  # Metavariable
    SORT = auto()  # Type universe
    CONST = auto()  # Global constant
    APP = auto()  # Application
    LAM = auto()  # Lambda
    FORALL = auto()  # Pi type
    LET = auto()  # Let binding
    LIT = auto()  # Literal
    MDATA = auto()  # Metadata
    PROJ = auto()  # Projection


@dataclass
class ExtractedConstant:
    """
    A constant extracted from an Expr tree.

    Represents a global theorem, definition, or axiom used in a proof.
    """

    name: str  # Full qualified name (e.g., "Nat.add_comm")
    universe_levels: list[str] = field(default_factory=list)  # Universe parameters

    # Source location (if available)
    module_path: str = ""
    file_path: str = ""
    line_number: int = 0

    # Whether this is a type class instance
    is_instance: bool = False

    # Whether this is from the standard library
    is_stdlib: bool = False

    def __hash__(self) -> int:
        return hash(self.name)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, ExtractedConstant):
            return self.name == other.name
        return False


@dataclass
class ExprNode:
    """
    A node in a serialized Expr tree.

    This is our Python representation of Lean's Expr type,
    extracted from JSON or S-expression format.
    """

    kind: ExprKind

    # For CONST nodes
    const_name: str | None = None
    const_levels: list[str] = field(default_factory=list)

    # For BVAR nodes
    bvar_index: int | None = None

    # For APP nodes
    fn: "ExprNode | None" = None
    arg: "ExprNode | None" = None

    # For LAM/FORALL nodes
    binder_name: str | None = None
    binder_type: "ExprNode | None" = None
    body: "ExprNode | None" = None

    # For LET nodes
    let_name: str | None = None
    let_type: "ExprNode | None" = None
    let_value: "ExprNode | None" = None
    let_body: "ExprNode | None" = None

    # Children for generic traversal
    children: list["ExprNode"] = field(default_factory=list)


class ExprTraverser:
    """
    Traverses Expr trees to extract used constants.

    Implements the foldConsts pattern from Lean metaprogramming.
    This is the correct way to find all premises used in a proof.
    """

    def __init__(self):
        self._constants: Set[ExtractedConstant] = set()
        self._visited: Set[int] = set()  # Track visited nodes by id

    def fold_consts(self, expr: ExprNode) -> Set[ExtractedConstant]:
        """
        Extract all constants from an Expr tree.

        This mirrors Lean.Expr.foldConsts which traverses the
        expression and collects all Expr.const nodes.

        Args:
            expr: Root of the Expr tree

        Returns:
            Set of all constants found
        """
        self._constants = set()
        self._visited = set()
        self._traverse(expr)
        return self._constants

    def _traverse(self, expr: ExprNode) -> None:
        """Recursively traverse the Expr tree."""
        # Avoid cycles (shouldn't happen in well-formed Expr, but be safe)
        node_id = id(expr)
        if node_id in self._visited:
            return
        self._visited.add(node_id)

        # Collect constants
        if expr.kind == ExprKind.CONST and expr.const_name:
            self._constants.add(
                ExtractedConstant(
                    name=expr.const_name,
                    universe_levels=expr.const_levels,
                )
            )

        # Traverse children
        if expr.fn:
            self._traverse(expr.fn)
        if expr.arg:
            self._traverse(expr.arg)
        if expr.binder_type:
            self._traverse(expr.binder_type)
        if expr.body:
            self._traverse(expr.body)
        if expr.let_type:
            self._traverse(expr.let_type)
        if expr.let_value:
            self._traverse(expr.let_value)
        if expr.let_body:
            self._traverse(expr.let_body)

        for child in expr.children:
            self._traverse(child)


def parse_expr_json(data: dict[str, Any]) -> ExprNode:
    """
    Parse a JSON representation of an Expr.

    Expected format (from Lean export tools):
    {
        "kind": "const",
        "name": "Nat.add",
        "levels": ["u"]
    }
    or
    {
        "kind": "app",
        "fn": {...},
        "arg": {...}
    }
    """
    kind_str = data.get("kind", "").upper()

    try:
        kind = ExprKind[kind_str]
    except KeyError:
        kind = ExprKind.MDATA  # Unknown, treat as metadata

    node = ExprNode(kind=kind)

    if kind == ExprKind.CONST:
        node.const_name = data.get("name")
        node.const_levels = data.get("levels", [])

    elif kind == ExprKind.BVAR:
        node.bvar_index = data.get("index")

    elif kind == ExprKind.APP:
        if "fn" in data:
            node.fn = parse_expr_json(data["fn"])
        if "arg" in data:
            node.arg = parse_expr_json(data["arg"])

    elif kind in {ExprKind.LAM, ExprKind.FORALL}:
        node.binder_name = data.get("name")
        if "type" in data:
            node.binder_type = parse_expr_json(data["type"])
        if "body" in data:
            node.body = parse_expr_json(data["body"])

    elif kind == ExprKind.LET:
        node.let_name = data.get("name")
        if "type" in data:
            node.let_type = parse_expr_json(data["type"])
        if "value" in data:
            node.let_value = parse_expr_json(data["value"])
        if "body" in data:
            node.let_body = parse_expr_json(data["body"])

    # Handle generic children array
    if "children" in data:
        node.children = [parse_expr_json(c) for c in data["children"]]

    return node


def parse_expr_sexp(sexp: str) -> ExprNode:
    """
    Parse an S-expression representation of an Expr.

    Format: (const Nat.add [u v]) or (app (const f []) (const x []))
    """
    # Simple recursive descent parser for S-expressions
    tokens = _tokenize_sexp(sexp)
    return _parse_sexp_tokens(iter(tokens))


def _tokenize_sexp(s: str) -> list[str]:
    """Tokenize an S-expression."""
    tokens = []
    current = ""

    for char in s:
        if char in "()[]":
            if current:
                tokens.append(current)
                current = ""
            tokens.append(char)
        elif char.isspace():
            if current:
                tokens.append(current)
                current = ""
        else:
            current += char

    if current:
        tokens.append(current)

    return tokens


def _parse_sexp_tokens(tokens: Iterator[str]) -> ExprNode:
    """Parse tokens into an ExprNode."""
    try:
        token = next(tokens)
    except StopIteration:
        return ExprNode(kind=ExprKind.MDATA)

    if token == "(":
        # Get the expression type
        kind_str = next(tokens).upper()

        try:
            kind = ExprKind[kind_str]
        except KeyError:
            kind = ExprKind.MDATA

        node = ExprNode(kind=kind)

        # Parse based on type
        if kind == ExprKind.CONST:
            # (const Name [levels])
            node.const_name = next(tokens)
            # Parse levels if present
            if _peek(tokens) == "[":
                next(tokens)  # consume [
                while (t := next(tokens)) != "]":
                    node.const_levels.append(t)

        elif kind == ExprKind.APP:
            # (app fn arg)
            node.fn = _parse_sexp_tokens(tokens)
            node.arg = _parse_sexp_tokens(tokens)

        elif kind in {ExprKind.LAM, ExprKind.FORALL}:
            # (lam name type body) or (forall name type body)
            node.binder_name = next(tokens)
            node.binder_type = _parse_sexp_tokens(tokens)
            node.body = _parse_sexp_tokens(tokens)

        # Consume closing paren
        while True:
            try:
                t = next(tokens)
                if t == ")":
                    break
            except StopIteration:
                break

        return node

    else:
        # Atomic token - treat as const
        return ExprNode(kind=ExprKind.CONST, const_name=token)


def _peek(it: Iterator[str]) -> str | None:
    """Peek at next token without consuming."""
    # This is a simplification - real impl would use itertools.tee
    return None


@dataclass
class DependencyInfo:
    """
    Dependency information for a theorem.

    Maps theorem -> {premises used}
    """

    theorem_name: str
    premises_used: Set[str] = field(default_factory=set)

    # Import path for this theorem
    module_path: str = ""

    # Direct imports (for dependency graph)
    imports: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "theorem": self.theorem_name,
            "premises": list(self.premises_used),
            "module": self.module_path,
            "imports": self.imports,
        }


class DependencyGraph:
    """
    Dependency graph for theorems and modules.

    This implements the SOTA approach where premises are filtered
    by the transitive closure of imports. A theorem in module B
    can only use premises from modules that B imports.
    """

    def __init__(self):
        # Module -> list of imports
        self._module_imports: dict[str, list[str]] = {}

        # Theorem -> DependencyInfo
        self._theorem_deps: dict[str, DependencyInfo] = {}

        # Cached transitive closures
        self._transitive_cache: dict[str, Set[str]] = {}

    def add_module(self, module_path: str, imports: list[str]) -> None:
        """Register a module and its imports."""
        self._module_imports[module_path] = imports
        self._transitive_cache.clear()  # Invalidate cache

    def add_theorem(
        self,
        theorem_name: str,
        module_path: str,
        premises_used: Set[str],
    ) -> None:
        """Register a theorem and its premise dependencies."""
        self._theorem_deps[theorem_name] = DependencyInfo(
            theorem_name=theorem_name,
            module_path=module_path,
            premises_used=premises_used,
            imports=self._module_imports.get(module_path, []),
        )

    def get_accessible_modules(self, module_path: str) -> Set[str]:
        """
        Get all modules accessible from a given module.

        This is the transitive closure of imports.
        """
        if module_path in self._transitive_cache:
            return self._transitive_cache[module_path]

        accessible = {module_path}
        to_process = list(self._module_imports.get(module_path, []))

        while to_process:
            mod = to_process.pop()
            if mod not in accessible:
                accessible.add(mod)
                to_process.extend(self._module_imports.get(mod, []))

        self._transitive_cache[module_path] = accessible
        return accessible

    def filter_premises(
        self,
        candidates: list[str],
        current_module: str,
    ) -> list[str]:
        """
        Filter premise candidates to only accessible ones.

        This prevents the "hallucinated premise" problem where
        the agent suggests a premise that exists but isn't imported.
        """
        accessible = self.get_accessible_modules(current_module)

        # Filter candidates whose module is accessible
        result = []
        for premise in candidates:
            # Get the module for this premise
            premise_module = self._get_premise_module(premise)
            if premise_module in accessible:
                result.append(premise)

        return result

    def _get_premise_module(self, premise_name: str) -> str:
        """Get the module path for a premise."""
        # Look up in theorem deps
        if premise_name in self._theorem_deps:
            return self._theorem_deps[premise_name].module_path

        # Infer from name (e.g., "Mathlib.Data.Nat.Basic.add_comm" -> "Mathlib.Data.Nat.Basic")
        parts = premise_name.rsplit(".", 1)
        if len(parts) > 1:
            return parts[0]

        return ""

    def get_premises_for(self, theorem_name: str) -> Set[str]:
        """Get the premises used by a theorem."""
        if theorem_name in self._theorem_deps:
            return self._theorem_deps[theorem_name].premises_used
        return set()

    def save(self, path: Path) -> None:
        """Save the dependency graph to disk."""
        data = {
            "modules": self._module_imports,
            "theorems": {name: info.to_dict() for name, info in self._theorem_deps.items()},
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: Path) -> "DependencyGraph":
        """Load a dependency graph from disk."""
        with open(path) as f:
            data = json.load(f)

        graph = cls()
        graph._module_imports = data.get("modules", {})

        for name, info_dict in data.get("theorems", {}).items():
            graph._theorem_deps[name] = DependencyInfo(
                theorem_name=info_dict["theorem"],
                module_path=info_dict.get("module", ""),
                premises_used=set(info_dict.get("premises", [])),
                imports=info_dict.get("imports", []),
            )

        return graph


def extract_constants_from_proof(
    proof_expr: dict[str, Any] | str,
) -> Set[str]:
    """
    Extract all constants used in a proof expression.

    This is the main entry point for AST-based premise extraction.

    Args:
        proof_expr: Either a JSON dict or S-expression string

    Returns:
        Set of constant names used in the proof
    """
    # Parse the expression
    if isinstance(proof_expr, dict):
        expr = parse_expr_json(proof_expr)
    elif isinstance(proof_expr, str):
        if proof_expr.startswith("("):
            expr = parse_expr_sexp(proof_expr)
        else:
            # Try JSON
            try:
                expr = parse_expr_json(json.loads(proof_expr))
            except json.JSONDecodeError:
                return set()
    else:
        return set()

    # Traverse to find constants
    traverser = ExprTraverser()
    constants = traverser.fold_consts(expr)

    return {c.name for c in constants}
