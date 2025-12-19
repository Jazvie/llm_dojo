"""
Index builder for Mathlib theorem corpus.

Provides utilities to:
- Extract theorems from Lean/Mathlib source
- Build embeddings and FAISS index
- Support for traced data when available
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np

from .embeddings import MathlibEmbedder
from .vector_store import MathlibVectorStore, TheoremMetadata


@dataclass
class ExtractedTheorem:
    """A theorem extracted from source."""

    name: str
    statement: str
    docstring: str
    module_path: str
    source_file: str
    line_number: int


class TheoremIndexBuilder:
    """
    Builds a searchable index of Mathlib theorems.

    Supports multiple data sources:
    1. JSON export (simplest)
    2. Lean source files (regex-based extraction)
    3. Traced repos (most accurate, when available)
    """

    # Regex patterns for extracting theorems from .lean files
    _THEOREM_PATTERN = re.compile(
        r"(?:^|\n)"
        r"(?:/--\s*(.*?)\s*-/\s*)?"  # Optional docstring
        r"(theorem|lemma|def)\s+"  # Declaration keyword
        r"(\w+(?:\.\w+)*)"  # Name
        r"\s*(?:\{[^}]*\})?"  # Optional implicit args
        r"\s*(?:\[[^\]]*\])?"  # Optional instance args
        r"\s*(?:\([^)]*\))*"  # Optional explicit args
        r"\s*:\s*"  # Colon
        r"([^:=]+)"  # Type/statement (until := or where)
        r"(?::=|where)",
        re.DOTALL | re.MULTILINE,
    )

    def __init__(
        self,
        embedder: MathlibEmbedder | None = None,
        model_name: str = "all-MiniLM-L6-v2",
    ):
        """
        Initialize the builder.

        Args:
            embedder: Optional pre-configured embedder
            model_name: Model name if creating new embedder
        """
        self.embedder = embedder or MathlibEmbedder(model_name)

    def extract_from_lean_file(
        self,
        file_path: Path | str,
        module_prefix: str = "",
    ) -> list[ExtractedTheorem]:
        """
        Extract theorems from a .lean file using regex.

        Note: This is heuristic and may miss some theorems or
        extract incorrect statements. For accuracy, use traced data.

        Args:
            file_path: Path to .lean file
            module_prefix: Module path prefix (e.g., "Mathlib.Algebra")

        Returns:
            List of extracted theorems
        """
        file_path = Path(file_path)
        if not file_path.exists():
            return []

        content = file_path.read_text(encoding="utf-8", errors="replace")
        theorems = []

        for match in self._THEOREM_PATTERN.finditer(content):
            docstring = match.group(1) or ""
            decl_type = match.group(2)  # theorem, lemma, or def
            name = match.group(3)
            statement = match.group(4).strip()

            # Clean up statement
            statement = re.sub(r"\s+", " ", statement)

            # Calculate line number
            line_number = content[: match.start()].count("\n") + 1

            # Build full name with module prefix
            full_name = f"{module_prefix}.{name}" if module_prefix else name

            theorems.append(
                ExtractedTheorem(
                    name=full_name,
                    statement=statement,
                    docstring=docstring.strip(),
                    module_path=module_prefix,
                    source_file=str(file_path),
                    line_number=line_number,
                )
            )

        return theorems

    def extract_from_directory(
        self,
        dir_path: Path | str,
        module_prefix: str = "",
    ) -> Iterator[ExtractedTheorem]:
        """
        Extract theorems from all .lean files in a directory.

        Args:
            dir_path: Directory path
            module_prefix: Base module prefix

        Yields:
            ExtractedTheorem objects
        """
        dir_path = Path(dir_path)

        for lean_file in dir_path.rglob("*.lean"):
            # Build module path from file path
            rel_path = lean_file.relative_to(dir_path)
            module_parts = list(rel_path.parent.parts) + [lean_file.stem]
            module_path = (
                ".".join([module_prefix] + module_parts)
                if module_prefix
                else ".".join(module_parts)
            )

            for theorem in self.extract_from_lean_file(lean_file, module_path):
                yield theorem

    def load_from_json(self, json_path: Path | str) -> list[ExtractedTheorem]:
        """
        Load theorems from a JSON export.

        Expected format:
        [
            {
                "name": "Nat.add_comm",
                "statement": "∀ n m : ℕ, n + m = m + n",
                "docstring": "Addition is commutative",
                "module_path": "Mathlib.Data.Nat.Basic"
            },
            ...
        ]

        Args:
            json_path: Path to JSON file

        Returns:
            List of ExtractedTheorem objects
        """
        with open(json_path) as f:
            data = json.load(f)

        return [
            ExtractedTheorem(
                name=t.get("name", ""),
                statement=t.get("statement", ""),
                docstring=t.get("docstring", ""),
                module_path=t.get("module_path", ""),
                source_file=t.get("source_file", ""),
                line_number=t.get("line_number", 0),
            )
            for t in data
        ]

    def build_index(
        self,
        theorems: list[ExtractedTheorem] | list[dict],
        batch_size: int = 32,
        show_progress: bool = True,
    ) -> MathlibVectorStore:
        """
        Build a vector store from extracted theorems.

        Args:
            theorems: List of theorems (ExtractedTheorem or dict)
            batch_size: Batch size for embedding
            show_progress: Show progress bar

        Returns:
            Populated MathlibVectorStore
        """
        # Convert to dict format for embedder
        theorem_dicts = []
        for t in theorems:
            if isinstance(t, ExtractedTheorem):
                theorem_dicts.append(
                    {
                        "name": t.name,
                        "statement": t.statement,
                        "docstring": t.docstring,
                        "module_path": t.module_path,
                    }
                )
            else:
                theorem_dicts.append(t)

        if not theorem_dicts:
            raise ValueError("No theorems to index")

        # Generate embeddings
        embedded = self.embedder.embed_theorems(
            theorem_dicts,
            batch_size=batch_size,
            show_progress=show_progress,
        )

        # Extract embeddings as numpy array
        embeddings = np.array([e.embedding for e in embedded], dtype=np.float32)

        # Create metadata
        metadata = [
            TheoremMetadata(
                name=e.name,
                statement=e.statement,
                docstring=e.docstring,
                module_path=e.module_path,
            )
            for e in embedded
        ]

        # Build vector store
        store = MathlibVectorStore(dimension=embeddings.shape[1])
        store.build(embeddings, metadata)

        return store

    def build_index_from_directory(
        self,
        dir_path: Path | str,
        module_prefix: str = "",
        batch_size: int = 32,
        show_progress: bool = True,
    ) -> MathlibVectorStore:
        """
        Build index from a directory of .lean files.

        Args:
            dir_path: Directory path
            module_prefix: Module prefix for names
            batch_size: Embedding batch size
            show_progress: Show progress bar

        Returns:
            Populated MathlibVectorStore
        """
        theorems = list(self.extract_from_directory(dir_path, module_prefix))
        return self.build_index(theorems, batch_size, show_progress)


# Seed theorems for bootstrap - common Mathlib theorems useful for H.O.R.S.E.
SEED_THEOREMS = [
    {
        "name": "Nat.add_comm",
        "statement": "∀ n m : ℕ, n + m = m + n",
        "docstring": "Addition of natural numbers is commutative",
        "module_path": "Mathlib.Data.Nat.Basic",
    },
    {
        "name": "Nat.add_assoc",
        "statement": "∀ n m k : ℕ, n + m + k = n + (m + k)",
        "docstring": "Addition of natural numbers is associative",
        "module_path": "Mathlib.Data.Nat.Basic",
    },
    {
        "name": "Nat.mul_comm",
        "statement": "∀ n m : ℕ, n * m = m * n",
        "docstring": "Multiplication of natural numbers is commutative",
        "module_path": "Mathlib.Data.Nat.Basic",
    },
    {
        "name": "Nat.mul_assoc",
        "statement": "∀ n m k : ℕ, n * m * k = n * (m * k)",
        "docstring": "Multiplication of natural numbers is associative",
        "module_path": "Mathlib.Data.Nat.Basic",
    },
    {
        "name": "Nat.add_zero",
        "statement": "∀ n : ℕ, n + 0 = n",
        "docstring": "Zero is a right identity for addition",
        "module_path": "Mathlib.Data.Nat.Basic",
    },
    {
        "name": "Nat.zero_add",
        "statement": "∀ n : ℕ, 0 + n = n",
        "docstring": "Zero is a left identity for addition",
        "module_path": "Mathlib.Data.Nat.Basic",
    },
    {
        "name": "Nat.mul_one",
        "statement": "∀ n : ℕ, n * 1 = n",
        "docstring": "One is a right identity for multiplication",
        "module_path": "Mathlib.Data.Nat.Basic",
    },
    {
        "name": "Nat.one_mul",
        "statement": "∀ n : ℕ, 1 * n = n",
        "docstring": "One is a left identity for multiplication",
        "module_path": "Mathlib.Data.Nat.Basic",
    },
    {
        "name": "Nat.left_distrib",
        "statement": "∀ n m k : ℕ, n * (m + k) = n * m + n * k",
        "docstring": "Multiplication distributes over addition (left)",
        "module_path": "Mathlib.Data.Nat.Basic",
    },
    {
        "name": "Nat.right_distrib",
        "statement": "∀ n m k : ℕ, (n + m) * k = n * k + m * k",
        "docstring": "Multiplication distributes over addition (right)",
        "module_path": "Mathlib.Data.Nat.Basic",
    },
    {
        "name": "Int.add_comm",
        "statement": "∀ a b : ℤ, a + b = b + a",
        "docstring": "Addition of integers is commutative",
        "module_path": "Mathlib.Data.Int.Basic",
    },
    {
        "name": "Int.mul_comm",
        "statement": "∀ a b : ℤ, a * b = b * a",
        "docstring": "Multiplication of integers is commutative",
        "module_path": "Mathlib.Data.Int.Basic",
    },
    {
        "name": "Int.neg_neg",
        "statement": "∀ a : ℤ, -(-a) = a",
        "docstring": "Double negation of an integer",
        "module_path": "Mathlib.Data.Int.Basic",
    },
    {
        "name": "Int.add_neg_self",
        "statement": "∀ a : ℤ, a + (-a) = 0",
        "docstring": "An integer plus its negation is zero",
        "module_path": "Mathlib.Data.Int.Basic",
    },
    {
        "name": "List.length_append",
        "statement": "∀ (l₁ l₂ : List α), (l₁ ++ l₂).length = l₁.length + l₂.length",
        "docstring": "Length of concatenated lists",
        "module_path": "Mathlib.Data.List.Basic",
    },
    {
        "name": "List.reverse_reverse",
        "statement": "∀ (l : List α), l.reverse.reverse = l",
        "docstring": "Reversing a list twice yields the original",
        "module_path": "Mathlib.Data.List.Basic",
    },
    {
        "name": "Bool.not_not",
        "statement": "∀ b : Bool, !!b = b",
        "docstring": "Double negation of a boolean",
        "module_path": "Mathlib.Data.Bool.Basic",
    },
    {
        "name": "And.comm",
        "statement": "∀ p q : Prop, p ∧ q ↔ q ∧ p",
        "docstring": "Conjunction is commutative",
        "module_path": "Mathlib.Logic.Basic",
    },
    {
        "name": "Or.comm",
        "statement": "∀ p q : Prop, p ∨ q ↔ q ∨ p",
        "docstring": "Disjunction is commutative",
        "module_path": "Mathlib.Logic.Basic",
    },
    {
        "name": "Eq.symm",
        "statement": "∀ {α : Type*} {a b : α}, a = b → b = a",
        "docstring": "Equality is symmetric",
        "module_path": "Init.Core",
    },
]


def build_seed_index() -> MathlibVectorStore:
    """
    Build a small index from seed theorems.

    Useful for testing and demos without full Mathlib.

    Returns:
        MathlibVectorStore with seed theorems
    """
    builder = TheoremIndexBuilder()
    return builder.build_index(SEED_THEOREMS, show_progress=False)
