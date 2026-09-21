import pytest

from embedding_lab.domain import BenchmarkResult, ModelConfig, ReplicaTopology, Workload


def test_domain_objects_capture_vendor_neutral_concepts() -> None:
    model = ModelConfig("Qwen/Qwen3-Embedding-0.6B", revision="main")
    topology = ReplicaTopology(replicas=1, tensor_parallel_size=2)
    workload = Workload(
        concurrency=8,
        requests=10,
        texts_per_request=2,
        target_tokens_per_text=256,
        warmup_requests=1,
    )
    result = BenchmarkResult(
        successful_requests=9,
        failed_requests=1,
        p50_latency_ms=10.0,
        p95_latency_ms=20.0,
        p99_latency_ms=25.0,
        vectors_per_second=900.0,
        elapsed_seconds=0.02,
        input_tokens=4_608,
        warmup_requests=1,
    )

    assert model.embedding_dimensions is None
    assert topology.worker_count == 2
    assert workload.worker_count == 8
    assert workload.measured_vectors == 20
    assert result.failure_rate == pytest.approx(0.1)
    assert result.tokens_per_second == pytest.approx(230_400.0)


@pytest.mark.parametrize(
    ("factory", "message"),
    [
        (lambda: ModelConfig(""), "model_id"),
        (lambda: ReplicaTopology(replicas=0), "replicas"),
        (
            lambda: Workload(
                concurrency=0,
                requests=1,
                texts_per_request=1,
                target_tokens_per_text=1,
            ),
            "concurrency",
        ),
    ],
)
def test_domain_objects_reject_invalid_values(factory: object, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        factory()  # type: ignore[operator]


def test_all_failed_result_has_no_fake_latency() -> None:
    result = BenchmarkResult(
        successful_requests=0,
        failed_requests=2,
        p50_latency_ms=None,
        p95_latency_ms=None,
        p99_latency_ms=None,
        vectors_per_second=0.0,
        elapsed_seconds=0.1,
        input_tokens=0,
    )

    assert result.p95_latency_ms is None
    assert result.tokens_per_second == 0.0
    assert result.failure_rate == 1.0


@pytest.mark.parametrize(
    ("successful_requests", "p50", "p95", "p99"),
    [
        (0, 0.0, 0.0, 0.0),
        (1, None, None, None),
        (1, 1.0, None, 3.0),
    ],
)
def test_latency_percentiles_match_success_count(
    successful_requests: int,
    p50: float | None,
    p95: float | None,
    p99: float | None,
) -> None:
    with pytest.raises(ValueError, match="latency percentiles"):
        BenchmarkResult(
            successful_requests=successful_requests,
            failed_requests=1,
            p50_latency_ms=p50,
            p95_latency_ms=p95,
            p99_latency_ms=p99,
            vectors_per_second=0.0,
            elapsed_seconds=1.0,
            input_tokens=1 if successful_requests else 0,
        )


@pytest.mark.parametrize(
    ("successful_requests", "input_tokens"),
    [(1, 0), (0, 1), (1, None)],
)
def test_benchmark_requires_reported_input_tokens(
    successful_requests: int, input_tokens: int | None
) -> None:
    with pytest.raises(ValueError, match="input_tokens|reported input tokens"):
        BenchmarkResult(
            successful_requests=successful_requests,
            failed_requests=1,
            p50_latency_ms=1.0 if successful_requests else None,
            p95_latency_ms=1.0 if successful_requests else None,
            p99_latency_ms=1.0 if successful_requests else None,
            vectors_per_second=1.0 if successful_requests else 0.0,
            elapsed_seconds=1.0,
            input_tokens=input_tokens,  # type: ignore[arg-type]
        )
