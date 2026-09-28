"""Embedding boundary consumed by research use cases."""

from typing import Protocol

from embedding_lab.domain.embedding import EmbeddingResponse


class EmbeddingService(Protocol):
    """Return one vector per text, preserving input order."""

    async def embed(self, texts: list[str]) -> EmbeddingResponse: ...
