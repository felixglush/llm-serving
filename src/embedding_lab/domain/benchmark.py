"""Benchmark result value object."""

import math
from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True, slots=True)
class RequestObservation:
    """One request attempt, retained for reproducibility and later analysis."""

    phase: Literal["warmup", "measured"]
    request_index: int
    success: bool
    latency_ms: float
    vectors: int = 0
    input_tokens: int = 0
    error: str | None = None

    def __post_init__(self) -> None:
        if self.phase not in {"warmup", "measured"}:
            raise ValueError("phase must be warmup or measured")
        if self.request_index < 0:
            raise ValueError("request_index must not be negative")
        if not math.isfinite(self.latency_ms) or self.latency_ms < 0:
            raise ValueError("latency_ms must be finite and non-negative")
        if self.vectors < 0 or self.input_tokens < 0:
            raise ValueError("vectors and input_tokens must not be negative")
        if self.success and self.error is not None:
            raise ValueError("successful observations must not contain an error")
        if not self.success and not self.error:
            raise ValueError("failed observations require an error")


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
    warmup_successful_requests: int = 0
    warmup_failed_requests: int = 0
    warmup_errors: tuple[str, ...] = ()
    successful_vectors: int = 0
    observations: tuple[RequestObservation, ...] = ()

    def __post_init__(self) -> None:
        for name, value in (
            ("successful_requests", self.successful_requests),
            ("failed_requests", self.failed_requests),
            ("warmup_requests", self.warmup_requests),
            ("warmup_successful_requests", self.warmup_successful_requests),
            ("warmup_failed_requests", self.warmup_failed_requests),
            ("successful_vectors", self.successful_vectors),
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
        if self.observations:
            measured = tuple(item for item in self.observations if item.phase == "measured")
            warmup = tuple(item for item in self.observations if item.phase == "warmup")
            if len(measured) != self.total_requests:
                raise ValueError("observations must include every measured request")
            if len(warmup) != self.warmup_requests:
                raise ValueError("observations must include every warmup request")
            if sum(item.success for item in measured) != self.successful_requests:
                raise ValueError("measured observations disagree with successful_requests")
            if sum(item.success for item in warmup) != self.warmup_successful_requests:
                raise ValueError("warmup observations disagree with warmup_successful_requests")
            if len(warmup) - self.warmup_successful_requests != self.warmup_failed_requests:
                raise ValueError("warmup observations disagree with warmup_failed_requests")
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

    @property
    def requests_per_second(self) -> float:
        """Successful measured requests per second."""

        return self.successful_requests / self.elapsed_seconds
