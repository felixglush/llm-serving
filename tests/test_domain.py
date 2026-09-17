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
        tokens_per_second=230_400.0,
        elapsed_seconds=0.02,
        actual_input_tokens=4_608,
        estimated_input_tokens=5_120,
        warmup_requests=1,
    )

    assert model.embedding_dimensions is None
    assert topology.worker_count == 2
    assert workload.worker_count == 8
    assert workload.measured_vectors == 20
    assert result.failure_rate == pytest.approx(0.1)
    assert result.token_count_source == "reported"


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
        tokens_per_second=None,
        elapsed_seconds=0.1,
        actual_input_tokens=None,
        estimated_input_tokens=512,
    )

    assert result.p95_latency_ms is None
    assert result.tokens_per_second is None
    assert result.failure_rate == 1.0
