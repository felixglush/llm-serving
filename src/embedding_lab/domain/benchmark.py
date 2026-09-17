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
    requests, so callers cannot accidentally read a fabricated zero. Token throughput is also
    optional because some providers do not return usage information.
    """

    successful_requests: int
    failed_requests: int
    p50_latency_ms: float | None
    p95_latency_ms: float | None
    p99_latency_ms: float | None
    vectors_per_second: float
    tokens_per_second: float | None
    elapsed_seconds: float
    actual_input_tokens: int | None
    estimated_input_tokens: int
    errors: tuple[str, ...] = ()
    warmup_requests: int = 0

    def __post_init__(self) -> None:
        for name, value in (
            ("successful_requests", self.successful_requests),
            ("failed_requests", self.failed_requests),
            ("estimated_input_tokens", self.estimated_input_tokens),
            ("warmup_requests", self.warmup_requests),
        ):
            if value < 0:
                raise ValueError(f"{name} must not be negative")
        if self.actual_input_tokens is not None and self.actual_input_tokens < 0:
            raise ValueError("actual_input_tokens must not be negative when provided")
        if not math.isfinite(self.elapsed_seconds) or self.elapsed_seconds <= 0:
            raise ValueError("elapsed_seconds must be positive and finite")
        if not math.isfinite(self.vectors_per_second) or self.vectors_per_second < 0:
            raise ValueError("vectors_per_second must be finite and non-negative")
        if self.tokens_per_second is not None and (
            not math.isfinite(self.tokens_per_second) or self.tokens_per_second < 0
        ):
            raise ValueError("tokens_per_second must be finite and non-negative when provided")
        _validate_percentile("p50_latency_ms", self.p50_latency_ms)
        _validate_percentile("p95_latency_ms", self.p95_latency_ms)
        _validate_percentile("p99_latency_ms", self.p99_latency_ms)
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
    def token_count_source(self) -> str:
        if self.actual_input_tokens is None:
            return "unavailable"
        return "reported"
