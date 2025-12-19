"""
Embedding generation for Mathlib theorems.

Uses sentence-transformers to create dense vector representations
of theorem names, statements, and docstrings for semantic search.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Sequence

# Lazy import to avoid loading model at module import time
_model = None


def _get_model():
    """Lazily load the sentence transformer model."""
    global _model
    if _model is None:
        try:
            from sentence_transformers import SentenceTransformer

            # Use a model good for code/math - all-MiniLM is fast and decent
            # For better quality, consider "all-mpnet-base-v2" or specialized math models
            _model = SentenceTransformer("all-MiniLM-L6-v2")
        except ImportError:
            raise ImportError(
                "sentence-transformers is required for embeddings. "
                "Install with: pip install sentence-transformers"
            )
    return _model


@dataclass
class TheoremEmbedding:
    """A theorem with its embedding vector."""

    name: str
    statement: str
    docstring: str
    module_path: str
    embedding: np.ndarray

    @property
    def dimension(self) -> int:
        return len(self.embedding)


class MathlibEmbedder:
    """
    Generates embeddings for Mathlib theorems.

    Combines theorem name, statement, and docstring into a single
    text representation for embedding.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the embedder.

        Args:
            model_name: The sentence-transformers model to use.
                        Options: "all-MiniLM-L6-v2" (fast), "all-mpnet-base-v2" (better quality)
        """
        self._model_name = model_name
        self._model = None

    @property
    def model(self):
        """Lazy load the model."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer

                self._model = SentenceTransformer(self._model_name)
            except ImportError:
                raise ImportError(
                    "sentence-transformers is required. "
                    "Install with: pip install sentence-transformers"
                )
        return self._model

    @property
    def dimension(self) -> int:
        """Return the embedding dimension."""
        # MiniLM-L6-v2 produces 384-dimensional embeddings
        # mpnet-base-v2 produces 768-dimensional embeddings
        return self.model.get_sentence_embedding_dimension()

    def _format_theorem_text(
        self,
        name: str,
        statement: str,
        docstring: str = "",
    ) -> str:
        """
        Format theorem info into a single text for embedding.

        The format emphasizes the mathematical content while including
        the name for disambiguation.
        """
        parts = []

        # Include theorem name (often descriptive like "Nat.add_comm")
        if name:
            # Convert dot-notation to readable form
            readable_name = name.replace(".", " ").replace("_", " ")
            parts.append(readable_name)

        # Main content is the statement
        if statement:
            parts.append(statement)

        # Docstring provides natural language context
        if docstring:
            parts.append(docstring)

        return " | ".join(parts)

    def embed_theorem(
        self,
        name: str,
        statement: str,
        docstring: str = "",
    ) -> np.ndarray:
        """
        Generate an embedding for a single theorem.

        Args:
            name: Theorem name (e.g., "Nat.add_comm")
            statement: The Lean type signature/statement
            docstring: Optional docstring

        Returns:
            Embedding vector as numpy array
        """
        text = self._format_theorem_text(name, statement, docstring)
        return self.model.encode(text, convert_to_numpy=True)

    def embed_theorems(
        self,
        theorems: Sequence[dict],
        batch_size: int = 32,
        show_progress: bool = True,
    ) -> list[TheoremEmbedding]:
        """
        Generate embeddings for multiple theorems.

        Args:
            theorems: List of dicts with keys: name, statement, docstring (optional), module_path (optional)
            batch_size: Batch size for encoding
            show_progress: Show progress bar

        Returns:
            List of TheoremEmbedding objects
        """
        # Format all texts
        texts = [
            self._format_theorem_text(
                t.get("name", ""),
                t.get("statement", ""),
                t.get("docstring", ""),
            )
            for t in theorems
        ]

        # Batch encode
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
        )

        # Create TheoremEmbedding objects
        results = []
        for t, emb in zip(theorems, embeddings):
            results.append(
                TheoremEmbedding(
                    name=t.get("name", ""),
                    statement=t.get("statement", ""),
                    docstring=t.get("docstring", ""),
                    module_path=t.get("module_path", ""),
                    embedding=emb,
                )
            )

        return results

    def embed_query(self, query: str) -> np.ndarray:
        """
        Generate an embedding for a search query.

        Args:
            query: Natural language query (e.g., "commutativity of addition")

        Returns:
            Query embedding vector
        """
        return self.model.encode(query, convert_to_numpy=True)

    def embed_goal_state(
        self,
        goal: str,
        hypotheses: list[str] | None = None,
    ) -> np.ndarray:
        """
        Generate an embedding for a proof goal state.

        This is useful for finding relevant lemmas given the current tactic state.

        Args:
            goal: The current proof goal
            hypotheses: Optional list of hypotheses in context

        Returns:
            Goal state embedding vector
        """
        parts = [goal]
        if hypotheses:
            parts.extend(hypotheses[:5])  # Limit to avoid too long text
        text = " ; ".join(parts)
        return self.model.encode(text, convert_to_numpy=True)
