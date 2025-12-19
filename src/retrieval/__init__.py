"""
Retrieval module for semantic search over Mathlib.

Provides:
- MathlibEmbedder: Embeds theorem statements for vector search
- MathlibVectorStore: FAISS-based vector database for premise retrieval
- TheoremIndex: Manages the theorem corpus and metadata
"""

from .embeddings import MathlibEmbedder
from .vector_store import MathlibVectorStore, TheoremMetadata
from .index_builder import TheoremIndexBuilder

__all__ = [
    "MathlibEmbedder",
    "MathlibVectorStore",
    "TheoremMetadata",
    "TheoremIndexBuilder",
]
