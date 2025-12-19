"""
Vector store for Mathlib theorem retrieval.

Uses FAISS for efficient similarity search over theorem embeddings.
"""

from __future__ import annotations

import json
import numpy as np
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Sequence

# Lazy import for FAISS
_faiss = None


def _get_faiss():
    """Lazily import faiss."""
    global _faiss
    if _faiss is None:
        try:
            import faiss

            _faiss = faiss
        except ImportError:
            raise ImportError(
                "faiss-cpu is required for vector search. Install with: pip install faiss-cpu"
            )
    return _faiss


@dataclass
class TheoremMetadata:
    """Metadata for a theorem in the index."""

    name: str
    statement: str
    docstring: str = ""
    module_path: str = ""
    # Additional fields for retrieval
    tactic_tags: list[str] | None = None  # e.g., ["simp", "ring"]
    difficulty_score: float = 0.0  # Estimated difficulty

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "TheoremMetadata":
        return cls(
            name=d.get("name", ""),
            statement=d.get("statement", ""),
            docstring=d.get("docstring", ""),
            module_path=d.get("module_path", ""),
            tactic_tags=d.get("tactic_tags"),
            difficulty_score=d.get("difficulty_score", 0.0),
        )


@dataclass
class SearchResult:
    """A single search result."""

    metadata: TheoremMetadata
    score: float  # Similarity score (higher = more similar)
    rank: int


class MathlibVectorStore:
    """
    FAISS-based vector store for Mathlib theorem retrieval.

    Supports:
    - Similarity search by query embedding
    - Filtering by module path or tags
    - Persistence to disk
    """

    def __init__(
        self,
        dimension: int = 384,  # MiniLM-L6-v2 default
        index_type: str = "flat",  # "flat" for exact, "ivf" for approximate
    ):
        """
        Initialize the vector store.

        Args:
            dimension: Embedding dimension
            index_type: Type of FAISS index ("flat" for exact search, "ivf" for faster approximate)
        """
        self.dimension = dimension
        self.index_type = index_type
        self._index = None
        self._metadata: list[TheoremMetadata] = []
        self._initialized = False

    def _create_index(self, dimension: int):
        """Create the FAISS index."""
        faiss = _get_faiss()

        if self.index_type == "flat":
            # Exact search using inner product (for normalized vectors = cosine similarity)
            self._index = faiss.IndexFlatIP(dimension)
        elif self.index_type == "ivf":
            # Approximate search for large datasets
            quantizer = faiss.IndexFlatIP(dimension)
            # nlist = number of clusters, adjust based on dataset size
            self._index = faiss.IndexIVFFlat(quantizer, dimension, 100)
        else:
            raise ValueError(f"Unknown index type: {self.index_type}")

    def build(
        self,
        embeddings: np.ndarray,
        metadata: Sequence[TheoremMetadata],
    ) -> None:
        """
        Build the index from embeddings and metadata.

        Args:
            embeddings: numpy array of shape (n_theorems, dimension)
            metadata: List of TheoremMetadata for each embedding
        """
        if len(embeddings) != len(metadata):
            raise ValueError("Number of embeddings must match metadata")

        faiss = _get_faiss()

        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)

        # Create and populate index
        self._create_index(embeddings.shape[1])
        self.dimension = embeddings.shape[1]

        if self.index_type == "ivf":
            # IVF index needs training
            self._index.train(embeddings)

        self._index.add(embeddings)
        self._metadata = list(metadata)
        self._initialized = True

    def search(
        self,
        query_embedding: np.ndarray,
        k: int = 10,
        module_filter: str | None = None,
    ) -> list[SearchResult]:
        """
        Search for similar theorems.

        Args:
            query_embedding: Query vector
            k: Number of results to return
            module_filter: Optional filter by module path prefix (e.g., "Mathlib.Algebra")

        Returns:
            List of SearchResult objects
        """
        if not self._initialized:
            return []

        faiss = _get_faiss()

        # Normalize query for cosine similarity
        query = query_embedding.reshape(1, -1).astype(np.float32)
        faiss.normalize_L2(query)

        # Search for more results if filtering
        search_k = k * 3 if module_filter else k

        scores, indices = self._index.search(query, search_k)

        results = []
        for rank, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx < 0:  # FAISS returns -1 for missing
                continue

            meta = self._metadata[idx]

            # Apply filter
            if module_filter and not meta.module_path.startswith(module_filter):
                continue

            results.append(
                SearchResult(
                    metadata=meta,
                    score=float(score),
                    rank=len(results),
                )
            )

            if len(results) >= k:
                break

        return results

    def search_by_text(
        self,
        query: str,
        embedder,  # MathlibEmbedder
        k: int = 10,
        module_filter: str | None = None,
    ) -> list[SearchResult]:
        """
        Search by natural language query.

        Args:
            query: Natural language query
            embedder: MathlibEmbedder instance for encoding
            k: Number of results
            module_filter: Optional module path filter

        Returns:
            List of SearchResult objects
        """
        query_emb = embedder.embed_query(query)
        return self.search(query_emb, k=k, module_filter=module_filter)

    def save(self, path: Path | str) -> None:
        """
        Save the index and metadata to disk.

        Args:
            path: Directory path to save to
        """
        if not self._initialized:
            raise RuntimeError("Index not initialized")

        faiss = _get_faiss()
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save FAISS index
        faiss.write_index(self._index, str(path / "index.faiss"))

        # Save metadata
        meta_list = [m.to_dict() for m in self._metadata]
        with open(path / "metadata.json", "w") as f:
            json.dump(meta_list, f)

        # Save config
        config = {
            "dimension": self.dimension,
            "index_type": self.index_type,
            "num_theorems": len(self._metadata),
        }
        with open(path / "config.json", "w") as f:
            json.dump(config, f)

    @classmethod
    def load(cls, path: Path | str) -> "MathlibVectorStore":
        """
        Load an index from disk.

        Args:
            path: Directory path to load from

        Returns:
            Loaded MathlibVectorStore instance
        """
        faiss = _get_faiss()
        path = Path(path)

        # Load config
        with open(path / "config.json") as f:
            config = json.load(f)

        store = cls(
            dimension=config["dimension"],
            index_type=config["index_type"],
        )

        # Load FAISS index
        store._index = faiss.read_index(str(path / "index.faiss"))

        # Load metadata
        with open(path / "metadata.json") as f:
            meta_list = json.load(f)
        store._metadata = [TheoremMetadata.from_dict(m) for m in meta_list]

        store._initialized = True
        return store

    def __len__(self) -> int:
        return len(self._metadata)

    @property
    def is_initialized(self) -> bool:
        return self._initialized
