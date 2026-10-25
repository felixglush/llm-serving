"""Workload shapes used by benchmark use cases."""

import random
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Workload:
    """A closed-loop request workload.

    ``target_tokens_per_text`` describes the generated input size used for planning. A benchmark
    records provider-reported usage separately because this target is only an approximation.
    """

    concurrency: int
    requests: int
    texts_per_request: int
    target_tokens_per_text: int
    warmup_requests: int = 0
    seed: int = 0
    corpus_version: str = "synthetic-v1"

    def __post_init__(self) -> None:
        if self.concurrency < 1:
            raise ValueError("concurrency must be positive")
        if self.requests < 1:
            raise ValueError("requests must be positive")
        if self.texts_per_request < 1:
            raise ValueError("texts_per_request must be positive")
        if self.target_tokens_per_text < 1:
            raise ValueError("target_tokens_per_text must be positive")
        if self.warmup_requests < 0:
            raise ValueError("warmup_requests must not be negative")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise ValueError("seed must be an integer")
        if not self.corpus_version.strip():
            raise ValueError("corpus_version must not be blank")

    @property
    def worker_count(self) -> int:
        """Return the bounded number of workers needed for this run."""

        return min(self.concurrency, self.requests)

    @property
    def measured_vectors(self) -> int:
        """Return how many vectors successful requests would produce."""

        return self.requests * self.texts_per_request

    @property
    def estimated_input_tokens(self) -> int:
        """Return the planned token count, which is distinct from reported usage."""

        return self.measured_vectors * self.target_tokens_per_text

    def texts_for_request(self, request_index: int) -> tuple[str, ...]:
        """Generate a stable synthetic batch for one request index.

        The generated text is a repeatable workload fixture, not a tokenizer-backed token
        count. Provider-reported usage remains authoritative for benchmark token throughput.
        """

        return synthetic_texts(
            seed=self.seed,
            corpus_version=self.corpus_version,
            request_index=request_index,
            texts_per_request=self.texts_per_request,
            target_tokens_per_text=self.target_tokens_per_text,
        )


def synthetic_texts(
    *,
    seed: int,
    corpus_version: str,
    request_index: int,
    texts_per_request: int,
    target_tokens_per_text: int,
) -> tuple[str, ...]:
    """Generate one deterministic synthetic request without choosing a load policy."""

    if request_index < 0:
        raise ValueError("request_index must not be negative")
    return tuple(
        _synthetic_text(
            seed=seed,
            corpus_version=corpus_version,
            request_index=request_index,
            text_index=text_index,
            target_tokens=target_tokens_per_text,
        )
        for text_index in range(texts_per_request)
    )


def _synthetic_text(
    *, seed: int, corpus_version: str, request_index: int, text_index: int, target_tokens: int
) -> str:
    rng = random.Random(f"{corpus_version}:{seed}:{request_index}:{text_index}")
    words = (f"w{rng.randrange(65_536):04x}" for _ in range(target_tokens))
    return " ".join(words)
