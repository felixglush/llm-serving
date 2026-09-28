"""Application boundaries owned by the model simulator."""

from typing import Protocol

from model_simulator.domain import EmbeddingResponse


class EmbeddingService(Protocol):
    """Return one vector per text, preserving input order."""

    async def embed(self, texts: list[str]) -> EmbeddingResponse: ...
