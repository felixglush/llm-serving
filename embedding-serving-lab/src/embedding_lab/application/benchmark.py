"""Provider-neutral closed-loop benchmark use case."""

import asyncio
import math
import time
from collections.abc import Callable, Sequence
from typing import Literal

from embedding_lab.domain.benchmark import BenchmarkResult, RequestObservation
from embedding_lab.domain.workload import Workload
from embedding_lab.ports.embedding_service import EmbeddingService

Clock = Callable[[], float]
Phase = Literal["warmup", "measured"]


class RunBenchmark:
    """Measure one fixed workload against any embedding service implementation.

    Workers use a closed loop: a worker sends one request, waits for its response, and only
    then sends the next request. Warm-up requests run before the measured clock starts.
    """

    def __init__(self, service: EmbeddingService, *, clock: Clock = time.perf_counter) -> None:
        self._service = service
        self._clock = clock

    async def run(self, workload: Workload) -> BenchmarkResult:
        warmups = await self._run_phase(workload, phase="warmup", count=workload.warmup_requests)
        started = self._clock()
        measured = await self._run_phase(workload, phase="measured", count=workload.requests)
        elapsed_seconds = max(self._clock() - started, 1e-12)

        successful = tuple(item for item in measured if item.success)
        failures = tuple(item.error or "request failed" for item in measured if not item.success)
        warmup_failures = tuple(
            item.error or "request failed" for item in warmups if not item.success
        )
        latencies = tuple(item.latency_ms for item in successful)
        successful_vectors = sum(item.vectors for item in successful)
        input_tokens = sum(item.input_tokens for item in successful)

        return BenchmarkResult(
            successful_requests=len(successful),
            failed_requests=len(measured) - len(successful),
            p50_latency_ms=_percentile(latencies, 0.50),
            p95_latency_ms=_percentile(latencies, 0.95),
            p99_latency_ms=_percentile(latencies, 0.99),
            vectors_per_second=successful_vectors / elapsed_seconds,
            elapsed_seconds=elapsed_seconds,
            input_tokens=input_tokens,
            errors=failures,
            warmup_requests=len(warmups),
            warmup_successful_requests=sum(item.success for item in warmups),
            warmup_failed_requests=len(warmups) - sum(item.success for item in warmups),
            warmup_errors=warmup_failures,
            successful_vectors=successful_vectors,
            observations=tuple(
                sorted(
                    (*warmups, *measured),
                    key=lambda item: (item.phase, item.request_index),
                )
            ),
        )

    async def _run_phase(
        self, workload: Workload, *, phase: Phase, count: int
    ) -> list[RequestObservation]:
        if count == 0:
            return []
        next_index = 0
        lock = asyncio.Lock()
        observations: list[RequestObservation] = []

        async def worker() -> None:
            nonlocal next_index
            while True:
                async with lock:
                    if next_index >= count:
                        return
                    request_index = next_index
                    next_index += 1
                observations.append(
                    await self._request(workload, phase=phase, request_index=request_index)
                )

        await asyncio.gather(*(worker() for _ in range(workload.worker_count)))
        return observations

    async def _request(
        self, workload: Workload, *, phase: Phase, request_index: int
    ) -> RequestObservation:
        started = self._clock()
        try:
            response = await self._service.embed(list(workload.texts_for_request(request_index)))
        except Exception as error:  # A failed request is part of the measured result.
            message = str(error).strip() or type(error).__name__
            return RequestObservation(
                phase=phase,
                request_index=request_index,
                success=False,
                latency_ms=_elapsed_ms(self._clock() - started),
                error=message[:500],
            )
        return RequestObservation(
            phase=phase,
            request_index=request_index,
            success=True,
            latency_ms=_elapsed_ms(self._clock() - started),
            vectors=len(response.embeddings),
            input_tokens=response.input_tokens,
        )


def _elapsed_ms(value: float) -> float:
    return max(value * 1_000, 0.0)


def _percentile(values: Sequence[float], fraction: float) -> float | None:
    """Return a linearly interpolated percentile, or None when no request succeeded."""

    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * weight
