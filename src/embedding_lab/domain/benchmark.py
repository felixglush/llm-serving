"""Benchmark result value object."""

import math
from dataclasses import dataclass


def _validate_percentile(name: str, value: float | None) -> None:
    if value is not None and (not math.isfinite(value) or value < 0):
        raise ValueError(f"{name} must be finite and non-negative when provided")


@dataclass(frozen=True, slots=True)
class BenchmarkResult:
    """Measured outcomes from successful and failed requests.

    Latencies are successful-request percentiles. ``None`` means there were no successful
    requests, so callers cannot accidentally read a fabricated zero. Input tokens are the
    server's reported usage for successful requests, not generated output tokens.
    """

    successful_requests: int
    failed_requests: int
    p50_latency_ms: float | None
    p95_latency_ms: float | None
    p99_latency_ms: float | None
    vectors_per_second: float
    elapsed_seconds: float
    input_tokens: int
    errors: tuple[str, ...] = ()
    warmup_requests: int = 0

    def __post_init__(self) -> None:
        for name, value in (
            ("successful_requests", self.successful_requests),
            ("failed_requests", self.failed_requests),
            ("warmup_requests", self.warmup_requests),
        ):
            if value < 0:
                raise ValueError(f"{name} must not be negative")
        if not isinstance(self.input_tokens, int) or isinstance(self.input_tokens, bool):
            raise ValueError("input_tokens must be a reported integer")
        if self.input_tokens < 0:
            raise ValueError("input_tokens must not be negative")
        if self.successful_requests > 0 and self.input_tokens == 0:
            raise ValueError("successful requests require reported input tokens")
        if self.successful_requests == 0 and self.input_tokens != 0:
            raise ValueError("input_tokens must be zero when no requests succeeded")
        if not math.isfinite(self.elapsed_seconds) or self.elapsed_seconds <= 0:
            raise ValueError("elapsed_seconds must be positive and finite")
        if not math.isfinite(self.vectors_per_second) or self.vectors_per_second < 0:
            raise ValueError("vectors_per_second must be finite and non-negative")
        _validate_percentile("p50_latency_ms", self.p50_latency_ms)
        _validate_percentile("p95_latency_ms", self.p95_latency_ms)
        _validate_percentile("p99_latency_ms", self.p99_latency_ms)
        latencies = (self.p50_latency_ms, self.p95_latency_ms, self.p99_latency_ms)
        if self.successful_requests == 0 and any(value is not None for value in latencies):
            raise ValueError("latency percentiles require successful requests")
        if self.successful_requests > 0 and any(value is None for value in latencies):
            raise ValueError("successful requests require all latency percentiles")
        if (
            self.p50_latency_ms is not None
            and self.p95_latency_ms is not None
            and self.p50_latency_ms > self.p95_latency_ms
        ):
            raise ValueError("p50_latency_ms must not exceed p95_latency_ms")
        if (
            self.p95_latency_ms is not None
            and self.p99_latency_ms is not None
            and self.p95_latency_ms > self.p99_latency_ms
        ):
            raise ValueError("p95_latency_ms must not exceed p99_latency_ms")

    @property
    def total_requests(self) -> int:
        return self.successful_requests + self.failed_requests

    @property
    def failure_rate(self) -> float:
        if self.total_requests == 0:
            return 0.0
        return self.failed_requests / self.total_requests

    @property
    def tokens_per_second(self) -> float:
        """Reported embedding input tokens divided by measured wall time."""

        return self.input_tokens / self.elapsed_seconds
