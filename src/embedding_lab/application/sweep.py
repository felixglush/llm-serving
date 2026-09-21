"""Provider-neutral concurrency sweep orchestration."""

from dataclasses import dataclass, replace

from embedding_lab.application.benchmark import RunBenchmark
from embedding_lab.domain.benchmark import BenchmarkResult
from embedding_lab.domain.workload import Workload
from embedding_lab.ports.embedding_service import EmbeddingService


@dataclass(frozen=True, slots=True)
class ConcurrencySweepPoint:
    """One benchmark result with the concurrency that produced it."""

    workload: Workload
    result: BenchmarkResult

    def __post_init__(self) -> None:
        if self.workload.concurrency < 1:
            raise ValueError("sweep point concurrency must be positive")

    @property
    def concurrency(self) -> int:
        return self.workload.concurrency


class ConcurrencySweep:
    """Run the same workload at several concurrency levels sequentially."""

    def __init__(self, service: EmbeddingService) -> None:
        self._service = service

    async def run(
        self,
        workload: Workload,
        concurrency_values: tuple[int, ...] | list[int],
    ) -> tuple[ConcurrencySweepPoint, ...]:
        values = _normalize_concurrency_values(concurrency_values)
        points: list[ConcurrencySweepPoint] = []
        for concurrency in values:
            point_workload = replace(workload, concurrency=concurrency)
            result = await RunBenchmark(self._service).run(point_workload)
            points.append(ConcurrencySweepPoint(point_workload, result))
        return tuple(points)


def _normalize_concurrency_values(values: tuple[int, ...] | list[int]) -> tuple[int, ...]:
    if not values:
        raise ValueError("concurrency_values must not be empty")
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 1 for value in values):
        raise ValueError("concurrency_values must contain positive integers")
    if len(set(values)) != len(values):
        raise ValueError("concurrency_values must not contain duplicates")
    return tuple(sorted(values))
