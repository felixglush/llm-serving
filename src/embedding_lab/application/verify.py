"""Check the observable contract of an embedding service."""

import math
from collections.abc import Sequence
from dataclasses import dataclass

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
    """Check one batch from any embedding-service adapter."""

    if isinstance(texts, str) or not texts:
        raise ValueError("verification requires at least one input text")
    if expected_dimensions is not None and (
        isinstance(expected_dimensions, bool)
        or not isinstance(expected_dimensions, int)
        or expected_dimensions < 1
    ):
        raise ValueError("expected_dimensions must be a positive integer")

    response = await service.embed(list(texts))
    failures: list[str] = []
    consistent_dimensions = len(response.embeddings) == len(texts)
    if not consistent_dimensions:
        failures.append(
            f"service returned {len(response.embeddings)} vectors for {len(texts)} texts"
        )

    dimensions: int | None = None
    for index, embedding in enumerate(response.embeddings):
        size = len(embedding.values)
        if size == 0:
            failures.append(f"embedding {index} is empty")
            consistent_dimensions = False
        if dimensions is None:
            dimensions = size
        elif size != dimensions:
            failures.append(f"embedding {index} has inconsistent dimensions")
            consistent_dimensions = False
        if expected_dimensions is not None and size != expected_dimensions:
            failures.append(
                f"embedding {index} has {size} dimensions, expected {expected_dimensions}"
            )
        if any(not _is_finite_number(value) for value in embedding.values):
            failures.append(f"embedding {index} contains a non-finite or non-numeric value")
        elif not any(value != 0 for value in embedding.values):
            failures.append(f"embedding {index} is all zero")

    return VerificationResult(
        input_count=len(texts),
        dimensions=dimensions if consistent_dimensions else None,
        failures=tuple(failures),
    )


def _is_finite_number(value: object) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False
