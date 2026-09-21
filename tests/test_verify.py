import math

import pytest

from embedding_lab.adapters.fake.embedding_service import FakeEmbeddingService
from embedding_lab.application.verify import VerifyEmbeddingService
from embedding_lab.domain.embedding import Embedding, EmbeddingResponse


def response(*vectors: tuple[float, ...]) -> EmbeddingResponse:
    return EmbeddingResponse(
        embeddings=tuple(Embedding(values=values) for values in vectors),
        input_tokens=len(vectors),
    )


class SequenceService:
    def __init__(self, first: EmbeddingResponse, repeated: EmbeddingResponse) -> None:
        self.responses = [first, repeated]
        self.calls: list[list[str]] = []

    async def embed(self, texts: list[str]) -> EmbeddingResponse:
        self.calls.append(texts)
        return self.responses.pop(0)


async def test_fake_service_passes_verification_without_infrastructure() -> None:
    result = await VerifyEmbeddingService(FakeEmbeddingService(dimensions=5)).run(
        ["alpha", "beta"], expected_dimensions=5
    )

    assert result.passed
    assert result.input_count == 2
    assert result.dimensions == 5
    assert result.failures == ()


async def test_repeated_batch_uses_the_same_inputs_and_tolerates_small_drift() -> None:
    service = SequenceService(
        response((1.0, 2.0), (3.0, 4.0)),
        response((1.000005, 2.0), (3.0, 4.0)),
    )

    result = await VerifyEmbeddingService(service).run(["alpha", "beta"])

    assert result.passed
    assert service.calls == [["alpha", "beta"], ["alpha", "beta"]]


@pytest.mark.parametrize(
    ("first", "repeated", "failure"),
    [
        (response((1.0,)), response((1.0,), (2.0,)), "first run returned 1 vectors"),
        (response((1.0,), (2.0,)), response((1.0,)), "repeated run returned 1 vectors"),
        (
            response((1.0,), (2.0, 3.0)),
            response((1.0,), (2.0, 3.0)),
            "inconsistent dimensions",
        ),
        (response((math.nan,)), response((1.0,)), "non-finite"),
        (response((math.inf,)), response((1.0,)), "non-finite"),
        (response((10**1000,)), response((1.0,)), "non-finite"),
        (response((0.0,)), response((1.0,)), "all zero"),
        (response((1.0,)), response((2.0,)), "changed between runs"),
    ],
)
async def test_verification_reports_invalid_embeddings(
    first: EmbeddingResponse, repeated: EmbeddingResponse, failure: str
) -> None:
    service = SequenceService(first, repeated)
    texts = (
        ["alpha", "beta"]
        if len(first.embeddings) == 2 or len(repeated.embeddings) == 2
        else ["alpha"]
    )

    result = await VerifyEmbeddingService(service).run(texts)

    assert not result.passed
    assert any(failure in problem for problem in result.failures)


async def test_expected_dimensions_are_checked() -> None:
    result = await VerifyEmbeddingService(FakeEmbeddingService(dimensions=5)).run(
        ["alpha"], expected_dimensions=4
    )

    assert not result.passed
    assert result.dimensions == 5
    assert any("expected 4" in problem for problem in result.failures)


async def test_dimensions_must_remain_consistent_between_runs() -> None:
    service = SequenceService(response((1.0, 2.0)), response((1.0, 2.0, 3.0)))

    result = await VerifyEmbeddingService(service).run(["alpha"])

    assert not result.passed
    assert "vector dimensions changed between runs" in result.failures


async def test_empty_verification_batch_is_rejected() -> None:
    with pytest.raises(ValueError, match="at least one"):
        await VerifyEmbeddingService(FakeEmbeddingService()).run([])


async def test_plain_string_is_rejected_as_batch() -> None:
    with pytest.raises(ValueError, match="at least one input text"):
        await VerifyEmbeddingService(FakeEmbeddingService()).run("alpha")


async def test_invalid_expected_dimensions_are_rejected() -> None:
    with pytest.raises(ValueError, match="expected_dimensions"):
        await VerifyEmbeddingService(FakeEmbeddingService()).run(["alpha"], expected_dimensions=0)
