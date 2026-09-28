"""Deterministic synthetic model owned by the model-simulator service."""

import hashlib

from model_simulator.domain import Embedding, EmbeddingResponse


class SyntheticEmbeddingService:
    """Produce stable nonzero vectors without downloads or inference."""

    def __init__(self, dimensions: int = 8) -> None:
        if isinstance(dimensions, bool) or not isinstance(dimensions, int) or dimensions < 1:
            raise ValueError("dimensions must be a positive integer")
        self.dimensions = dimensions

    async def embed(self, texts: list[str]) -> EmbeddingResponse:
        vectors = []
        for text in texts:
            values = []
            for index in range(self.dimensions):
                digest = hashlib.sha256(f"{index}:{text}".encode()).digest()
                values.append((int.from_bytes(digest[:4], "big") + 1) / (2**32))
            vectors.append(Embedding(values=tuple(values)))
        return EmbeddingResponse(embeddings=tuple(vectors), input_tokens=len(texts))
