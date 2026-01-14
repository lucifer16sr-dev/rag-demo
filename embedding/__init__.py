"""Embedding module for RAG Knowledge Assistant."""

from .create_embeddings import (
    EmbeddingStore,
    create_embeddings_from_documents,
    load_embedding_store,
)

__all__ = [
    'EmbeddingStore',
    'create_embeddings_from_documents',
    'load_embedding_store',
]
