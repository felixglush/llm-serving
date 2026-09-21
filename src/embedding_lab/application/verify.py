"""Validate a batch of embeddings against its source texts."""

import math
from collections.abc import Sequence
from dataclasses import dataclass

from embedding_lab.domain.embedding import Embedding


@dataclass(frozen=True, slots=True)
class VerificationResult:
    """Observed vector shape and any correctness failures."""

    input_count: int
    dimensions: int | None
    failures: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return not self.failures


def validate_embeddings(
    texts: Sequence[str],
    embeddings: Sequence[Embedding],
    *,
    expected_dimensions: int | None = None,
) -> VerificationResult:
    """Check vector count, dimensions, and numeric values without performing I/O."""

    if isinstance(texts, str) or not texts:
        raise ValueError("verification requires at least one input text")
    if expected_dimensions is not None and (
        isinstance(expected_dimensions, bool)
        or not isinstance(expected_dimensions, int)
        or expected_dimensions < 1
    ):
        raise ValueError("expected_dimensions must be a positive integer")

    failures: list[str] = []
    consistent_dimensions = len(embeddings) == len(texts)
    if not consistent_dimensions:
        failures.append(
            f"received {len(embeddings)} vectors for {len(texts)} texts"
        )

    dimensions: int | None = None
    for index, embedding in enumerate(embeddings):
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
