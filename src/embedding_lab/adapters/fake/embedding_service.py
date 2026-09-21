"""An offline deterministic test adapter, not a semantic embedding model."""

import hashlib

from embedding_lab.domain.embedding import Embedding, EmbeddingResponse


class FakeEmbeddingService:
    """Produce stable nonzero vectors without network access or model downloads.

    No model tokenizer runs. The fake reports one synthetic input token per text,
    only to exercise the service contract in offline tests. Its numbers must not
    be interpreted as serving measurements.
    """

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
                # Strictly positive values make even empty input nonzero.
                values.append((int.from_bytes(digest[:4], "big") + 1) / (2**32))
            vectors.append(Embedding(values=tuple(values)))
        return EmbeddingResponse(embeddings=tuple(vectors), input_tokens=len(texts))
