"""Compatibility names for the shared OpenAI-compatible response parser."""

from embedding_lab.adapters.openai_embeddings import (
    OpenAIEmbeddingResponseError,
    parse_embedding_response,
)

VllmResponseError = OpenAIEmbeddingResponseError

__all__ = ["VllmResponseError", "parse_embedding_response"]
