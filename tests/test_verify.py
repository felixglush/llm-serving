import math

import pytest

from embedding_lab.application.verify import validate_embeddings
from embedding_lab.domain.embedding import Embedding


def embeddings(*vectors: tuple[float, ...]) -> tuple[Embedding, ...]:
    return tuple(Embedding(values=values) for values in vectors)


def test_valid_embeddings_pass_without_a_service() -> None:
    result = validate_embeddings(
        ["alpha", "beta"],
        embeddings((1.0, 2.0), (3.0, 4.0)),
        expected_dimensions=2,
    )

    assert result.passed
    assert result.input_count == 2
    assert result.dimensions == 2
    assert result.failures == ()


@pytest.mark.parametrize(
    ("result", "texts", "failure"),
    [
        (embeddings((1.0,)), ["alpha", "beta"], "received 1 vectors"),
        (
            embeddings((1.0,), (2.0, 3.0)),
            ["alpha", "beta"],
            "inconsistent dimensions",
        ),
        (embeddings((math.nan,)), ["alpha"], "non-finite"),
        (embeddings((math.inf,)), ["alpha"], "non-finite"),
        (embeddings((10**1000,)), ["alpha"], "non-finite"),
        (embeddings((0.0,)), ["alpha"], "all zero"),
    ],
)
def test_validation_reports_invalid_embeddings(
    result: tuple[Embedding, ...], texts: list[str], failure: str
) -> None:
    verification = validate_embeddings(texts, result)

    assert not verification.passed
    assert any(failure in problem for problem in verification.failures)


def test_expected_dimensions_are_checked() -> None:
    result = validate_embeddings(
        ["alpha"], embeddings((1.0, 2.0, 3.0, 4.0, 5.0)), expected_dimensions=4
    )

    assert not result.passed
    assert result.dimensions == 5
    assert any("expected 4" in problem for problem in result.failures)


def test_empty_verification_batch_is_rejected() -> None:
    with pytest.raises(ValueError, match="at least one"):
        validate_embeddings([], ())


def test_plain_string_is_rejected_as_batch() -> None:
    with pytest.raises(ValueError, match="at least one input text"):
        validate_embeddings("alpha", ())


def test_invalid_expected_dimensions_are_rejected() -> None:
    with pytest.raises(ValueError, match="expected_dimensions"):
        validate_embeddings(["alpha"], embeddings((1.0,)), expected_dimensions=0)
