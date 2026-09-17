"""Workload shapes used by benchmark use cases."""

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
