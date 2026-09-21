import math

import pytest

from embedding_lab.adapters.fake.embedding_service import FakeEmbeddingService
from embedding_lab.application.verify import verify_embeddings
from embedding_lab.domain.embedding import Embedding, EmbeddingResponse


def response(*vectors: tuple[float, ...]) -> EmbeddingResponse:
    return EmbeddingResponse(
        embeddings=tuple(Embedding(values=values) for values in vectors),
        input_tokens=len(vectors),
    )


class StubService:
    def __init__(self, result: EmbeddingResponse) -> None:
        self.result = result
        self.calls: list[list[str]] = []

    async def embed(self, texts: list[str]) -> EmbeddingResponse:
        self.calls.append(texts)
        return self.result


async def test_fake_service_passes_verification_without_infrastructure() -> None:
    result = await verify_embeddings(
        FakeEmbeddingService(dimensions=5), ["alpha", "beta"], expected_dimensions=5
    )

    assert result.passed
    assert result.input_count == 2
    assert result.dimensions == 5
    assert result.failures == ()


async def test_verification_sends_one_batch_to_the_service() -> None:
    service = StubService(response((1.0, 2.0), (3.0, 4.0)))

    result = await verify_embeddings(service, ["alpha", "beta"])

    assert result.passed
    assert service.calls == [["alpha", "beta"]]


@pytest.mark.parametrize(
    ("result", "texts", "failure"),
    [
        (response((1.0,)), ["alpha", "beta"], "returned 1 vectors"),
        (
            response((1.0,), (2.0, 3.0)),
            ["alpha", "beta"],
            "inconsistent dimensions",
        ),
        (response((math.nan,)), ["alpha"], "non-finite"),
        (response((math.inf,)), ["alpha"], "non-finite"),
        (response((10**1000,)), ["alpha"], "non-finite"),
        (response((0.0,)), ["alpha"], "all zero"),
    ],
)
async def test_verification_reports_invalid_embeddings(
    result: EmbeddingResponse, texts: list[str], failure: str
) -> None:
    service = StubService(result)

    verification = await verify_embeddings(service, texts)

    assert not verification.passed
    assert any(failure in problem for problem in verification.failures)


async def test_expected_dimensions_are_checked() -> None:
    result = await verify_embeddings(
        FakeEmbeddingService(dimensions=5), ["alpha"], expected_dimensions=4
    )

    assert not result.passed
    assert result.dimensions == 5
    assert any("expected 4" in problem for problem in result.failures)


async def test_empty_verification_batch_is_rejected() -> None:
    with pytest.raises(ValueError, match="at least one"):
        await verify_embeddings(FakeEmbeddingService(), [])


async def test_plain_string_is_rejected_as_batch() -> None:
    with pytest.raises(ValueError, match="at least one input text"):
        await verify_embeddings(FakeEmbeddingService(), "alpha")


async def test_invalid_expected_dimensions_are_rejected() -> None:
    with pytest.raises(ValueError, match="expected_dimensions"):
        await verify_embeddings(FakeEmbeddingService(), ["alpha"], expected_dimensions=0)
