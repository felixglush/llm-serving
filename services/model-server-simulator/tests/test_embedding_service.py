import math
from dataclasses import FrozenInstanceError

import pytest

from model_server_simulator.domain import Embedding, EmbeddingResponse
from model_server_simulator.ports import EmbeddingService
from model_server_simulator.synthetic_model import SyntheticEmbeddingService


async def test_synthetic_model_satisfies_port_preserves_order_and_is_stable() -> None:
    service: EmbeddingService = SyntheticEmbeddingService(dimensions=5)
    response = await service.embed(["hello", "world", "hello", ""])
    repeated = await SyntheticEmbeddingService(dimensions=5).embed(["hello"])

    assert len(response.embeddings) == 4
    assert response.embeddings[0] == response.embeddings[2] == repeated.embeddings[0]
    assert response.embeddings[0] != response.embeddings[1]
    for embedding in response.embeddings:
        assert len(embedding.values) == 5
        assert all(math.isfinite(value) for value in embedding.values)
        assert any(value != 0 for value in embedding.values)
    assert response.input_tokens == 4  # Synthetic fake usage: one token per text.


async def test_empty_batch_returns_no_vectors_or_tokens() -> None:
    response = await SyntheticEmbeddingService().embed([])
    assert response == EmbeddingResponse(embeddings=(), input_tokens=0)


@pytest.mark.parametrize("dimensions", [0, -1, True])
def test_synthetic_model_rejects_invalid_dimensions(dimensions: int) -> None:
    with pytest.raises(ValueError, match="dimensions"):
        SyntheticEmbeddingService(dimensions=dimensions)


def test_response_requires_request_level_input_tokens() -> None:
    response = EmbeddingResponse((Embedding((1.0,)), Embedding((2.0,))), input_tokens=7)
    assert response.input_tokens == 7
    assert EmbeddingResponse((), input_tokens=0).input_tokens == 0
    for invalid in (-1, None, True):
        with pytest.raises(ValueError, match="input_tokens"):
            EmbeddingResponse((), input_tokens=invalid)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="positive input-token count"):
        EmbeddingResponse((Embedding((1.0,)),), input_tokens=0)
    with pytest.raises(FrozenInstanceError):
        response.input_tokens = 8  # type: ignore[misc]
