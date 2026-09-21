"""Check the observable contract of an embedding service."""

import math
from collections.abc import Sequence
from dataclasses import dataclass

from embedding_lab.domain.embedding import EmbeddingResponse
from embedding_lab.ports.embedding_service import EmbeddingService

DEFAULT_TEXTS = ("A short example sentence.", "A different example sentence.")


@dataclass(frozen=True, slots=True)
class VerificationResult:
    """Observed vector shape and any correctness failures."""

    input_count: int
    dimensions: int | None
    failures: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return not self.failures


async def verify_embeddings(
    service: EmbeddingService,
    texts: Sequence[str] = DEFAULT_TEXTS,
    *,
    expected_dimensions: int | None = None,
) -> VerificationResult:
    """Check any embedding-service adapter by running the same batch twice."""

    if isinstance(texts, str) or not texts:
        raise ValueError("verification requires at least one input text")
    if expected_dimensions is not None and (
        isinstance(expected_dimensions, bool)
        or not isinstance(expected_dimensions, int)
        or expected_dimensions < 1
    ):
        raise ValueError("expected_dimensions must be a positive integer")

    first = await service.embed(list(texts))
    repeated = await service.embed(list(texts))
    failures: list[str] = []
    first_dimensions, first_valid = _inspect(
        first, "first", len(texts), expected_dimensions, failures
    )
    repeated_dimensions, repeated_valid = _inspect(
        repeated, "repeated", len(texts), expected_dimensions, failures
    )

    if first_valid and repeated_valid and first_dimensions != repeated_dimensions:
        failures.append("vector dimensions changed between runs")
    if first_valid and repeated_valid and first_dimensions == repeated_dimensions:
        for index, (original, again) in enumerate(
            zip(first.embeddings, repeated.embeddings, strict=True)
        ):
            if any(
                not math.isclose(left, right, rel_tol=1e-5, abs_tol=1e-6)
                for left, right in zip(original.values, again.values, strict=True)
            ):
                failures.append(f"embedding {index} changed between runs")

    return VerificationResult(
        input_count=len(texts),
        dimensions=first_dimensions,
        failures=tuple(failures),
    )


def _inspect(
    response: EmbeddingResponse,
    label: str,
    expected_count: int,
    expected_dimensions: int | None,
    failures: list[str],
) -> tuple[int | None, bool]:
    """Check one batch and return its consistent dimension, when established."""

    valid = True
    consistent_dimensions = True
    if len(response.embeddings) != expected_count:
        failures.append(
            f"{label} run returned {len(response.embeddings)} vectors for {expected_count} texts"
        )
        valid = False
        consistent_dimensions = False

    dimensions: int | None = None
    for index, embedding in enumerate(response.embeddings):
        size = len(embedding.values)
        if size == 0:
            failures.append(f"{label} embedding {index} is empty")
            valid = False
            consistent_dimensions = False
        if dimensions is None:
            dimensions = size
        elif size != dimensions:
            failures.append(f"{label} embedding {index} has inconsistent dimensions")
            valid = False
            consistent_dimensions = False
        if expected_dimensions is not None and size != expected_dimensions:
            failures.append(
                f"{label} embedding {index} has {size} dimensions, expected {expected_dimensions}"
            )
            valid = False
        if any(not _is_finite_number(value) for value in embedding.values):
            failures.append(f"{label} embedding {index} contains a non-finite or non-numeric value")
            valid = False
        elif not any(value != 0 for value in embedding.values):
            failures.append(f"{label} embedding {index} is all zero")
            valid = False

    return dimensions if consistent_dimensions else None, valid


def _is_finite_number(value: object) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False
