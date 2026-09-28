import asyncio
import json

import pytest

from embedding_lab.application.benchmark import RunBenchmark
from embedding_lab.cli.main import main
from embedding_lab.domain.embedding import Embedding, EmbeddingResponse
from embedding_lab.domain.workload import Workload


class InstrumentedService:
    def __init__(self, failures: set[int] | None = None) -> None:
        self.calls = 0
        self.active = 0
        self.max_active = 0
        self.failures = failures or set()

    async def embed(self, texts: list[str]) -> EmbeddingResponse:
        call = self.calls
        self.calls += 1
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        try:
            await asyncio.sleep(0)
            if call in self.failures:
                raise RuntimeError(f"synthetic failure {call}")
            return EmbeddingResponse(
                embeddings=tuple(Embedding((float(index),)) for index in range(len(texts))),
                input_tokens=len(texts) * 3,
            )
        finally:
            self.active -= 1


async def test_runner_attempts_exact_requests_and_bounds_closed_loop_concurrency() -> None:
    service = InstrumentedService()
    workload = Workload(
        concurrency=3,
        requests=7,
        texts_per_request=2,
        target_tokens_per_text=4,
        warmup_requests=2,
        seed=12,
        corpus_version="fixture-a",
    )

    result = await RunBenchmark(service).run(workload)

    assert service.calls == 9
    assert service.max_active <= 3
    assert result.total_requests == 7
    assert result.successful_requests == 7
    assert result.failed_requests == 0
    assert result.warmup_requests == 2
    assert result.warmup_successful_requests == 2
    assert result.successful_vectors == 14
    assert result.input_tokens == 42
    assert len(result.observations) == 9
    assert [item.phase for item in result.observations].count("warmup") == 2
    assert [item.phase for item in result.observations].count("measured") == 7


async def test_runner_keeps_failures_and_excludes_warmups_from_metrics() -> None:
    service = InstrumentedService(failures={0, 3})
    workload = Workload(
        concurrency=1,
        requests=3,
        texts_per_request=1,
        target_tokens_per_text=4,
        warmup_requests=1,
    )

    result = await RunBenchmark(service).run(workload)

    assert result.successful_requests == 2
    assert result.failed_requests == 1
    assert result.warmup_successful_requests == 0
    assert result.warmup_failed_requests == 1
    assert result.errors == ("synthetic failure 3",)
    assert result.warmup_errors == ("synthetic failure 0",)
    assert result.p50_latency_ms is not None
    assert result.p95_latency_ms is not None
    assert result.p99_latency_ms is not None
    assert result.failure_rate == pytest.approx(1 / 3)


async def test_runner_reports_no_latency_when_every_measured_request_fails() -> None:
    service = InstrumentedService(failures={0, 1})
    result = await RunBenchmark(service).run(
        Workload(
            concurrency=2,
            requests=2,
            texts_per_request=1,
            target_tokens_per_text=1,
        )
    )

    assert result.successful_requests == 0
    assert result.p50_latency_ms is None
    assert result.p95_latency_ms is None
    assert result.p99_latency_ms is None
    assert result.vectors_per_second == 0
    assert result.tokens_per_second == 0
    assert result.failure_rate == 1.0


def test_workload_text_generation_is_repeatable_and_seeded() -> None:
    workload = Workload(
        concurrency=1,
        requests=1,
        texts_per_request=2,
        target_tokens_per_text=3,
        seed=7,
        corpus_version="fixture-a",
    )

    assert workload.texts_for_request(4) == workload.texts_for_request(4)
    assert workload.texts_for_request(4) != Workload(
        concurrency=1,
        requests=1,
        texts_per_request=2,
        target_tokens_per_text=3,
        seed=8,
        corpus_version="fixture-a",
    ).texts_for_request(4)


def test_benchmark_command_emits_result_json(capsys: pytest.CaptureFixture[str]) -> None:
    code = main(
        [
            "benchmark",
            "--requests",
            "3",
            "--concurrency",
            "2",
            "--texts-per-request",
            "2",
            "--warmup-requests",
            "1",
        ]
    )

    assert code == 0
    output = json.loads(capsys.readouterr().out)
    assert output["service"] == "fake"
    assert output["workload"]["requests"] == 3
    assert output["result"]["successful_requests"] == 3
    assert len(output["result"]["observations"]) == 4
