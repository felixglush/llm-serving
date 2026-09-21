"""The small boundary consumed by embedding application use cases."""

from typing import Protocol

from embedding_lab.domain.embedding import EmbeddingResponse


class EmbeddingService(Protocol):
    """Return one vector per text, in the same order as the input."""

    async def embed(self, texts: list[str]) -> EmbeddingResponse: ...
