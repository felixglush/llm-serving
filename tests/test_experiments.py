from pathlib import Path
from typing import Any

import pytest

from embedding_lab.adapters.wandb.publisher import WandbPublisher
from embedding_lab.application.experiments import ExperimentStore
from embedding_lab.domain.benchmark import BenchmarkResult, RequestObservation
from embedding_lab.domain.deployment import Deployment
from embedding_lab.domain.experiment import Experiment
from embedding_lab.domain.model import ModelConfig
from embedding_lab.domain.topology import ReplicaTopology
from embedding_lab.domain.workload import Workload


def make_experiment() -> Experiment:
    observations = (
        RequestObservation(
            phase="measured",
            request_index=0,
            success=True,
            latency_ms=12.0,
            vectors=2,
            input_tokens=8,
        ),
    )
    result = BenchmarkResult(
        successful_requests=1,
        failed_requests=0,
        p50_latency_ms=12.0,
        p95_latency_ms=12.0,
        p99_latency_ms=12.0,
        vectors_per_second=166.0,
        elapsed_seconds=0.012,
        input_tokens=8,
        successful_vectors=2,
        observations=observations,
    )
    return Experiment(
        experiment_id="exp123",
        created_at="2026-09-21T12:00:00Z",
        model=ModelConfig("example/model", revision="revision-sha"),
        topology=ReplicaTopology(),
        deployment=Deployment("deployment-a", "https://example.test", "fingerprint-a"),
        serving_config={"gpu_type": "L4", "max_num_seqs": 32},
        workload=Workload(
            concurrency=1,
            requests=1,
            texts_per_request=2,
            target_tokens_per_text=4,
            seed=5,
            corpus_version="fixture-a",
        ),
        result=result,
    )


def test_experiment_store_round_trips_json_and_observations(tmp_path: Path) -> None:
    store = ExperimentStore(tmp_path)
    experiment = make_experiment()

    path = store.save(experiment)
    loaded = store.load("exp123")

    assert path == tmp_path / "exp123" / "experiment.json"
    assert loaded == experiment
    assert (tmp_path / "exp123" / "observations.json").exists()


def test_experiment_requires_an_immutable_model_revision() -> None:
    with pytest.raises(ValueError, match="immutable"):
        Experiment(
            experiment_id="exp123",
            created_at="2026-09-21T12:00:00Z",
            model=ModelConfig("example/model"),
            topology=ReplicaTopology(),
            deployment=Deployment("deployment-a", "https://example.test", "fingerprint-a"),
            serving_config={},
            workload=Workload(
                concurrency=1,
                requests=1,
                texts_per_request=1,
                target_tokens_per_text=1,
            ),
            result=BenchmarkResult(
                successful_requests=1,
                failed_requests=0,
                p50_latency_ms=1.0,
                p95_latency_ms=1.0,
                p99_latency_ms=1.0,
                vectors_per_second=1.0,
                elapsed_seconds=1.0,
                input_tokens=1,
            ),
        )


class FakeArtifact:
    def __init__(self, name: str, type: str) -> None:
        self.name = name
        self.type = type
        self.files: list[str] = []

    def add_file(self, path: str) -> None:
        self.files.append(path)


class FakeRun:
    def __init__(self) -> None:
        self.id = "wandb123"
        self.url = "https://wandb.ai/user/project/runs/wandb123"
        self.logs: list[dict[str, Any]] = []
        self.artifacts: list[FakeArtifact] = []

    def __enter__(self) -> "FakeRun":
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def log(self, metrics: dict[str, Any]) -> None:
        self.logs.append(metrics)

    def log_artifact(self, artifact: FakeArtifact) -> None:
        self.artifacts.append(artifact)


class FakeWandb:
    def __init__(self) -> None:
        self.init_calls: list[dict[str, Any]] = []
        self.runs: list[FakeRun] = []

    def init(self, **kwargs: Any) -> FakeRun:
        self.init_calls.append(kwargs)
        run = FakeRun()
        self.runs.append(run)
        return run

    def Artifact(self, name: str, type: str) -> FakeArtifact:  # noqa: N802
        return FakeArtifact(name, type)


def test_wandb_publisher_maps_config_metrics_artifact_and_is_idempotent(
    tmp_path: Path,
) -> None:
    store = ExperimentStore(tmp_path)
    experiment = make_experiment()
    fake = FakeWandb()
    publisher = WandbPublisher(project="llm-serving-benchmarks", entity="user", client=fake)

    first = publisher.publish(experiment, store)
    second = publisher.publish(experiment, store)

    assert first == second
    assert len(fake.init_calls) == 1
    init = fake.init_calls[0]
    assert init["id"] == "exp123"
    assert init["resume"] == "allow"
    assert init["config"]["model_revision"] == "revision-sha"
    assert init["config"]["workload_seed"] == 5
    assert init["group"] == "deployment-fingerprint-a"
    run = fake.runs[0]
    assert run.logs[0]["p95_latency_ms"] == 12.0
    assert len(run.artifacts) == 1
    assert {Path(path).name for path in run.artifacts[0].files} == {
        "experiment.json",
        "observations.json",
    }
    assert store.load_receipt("exp123") == first


class FailingWandb(FakeWandb):
    def init(self, **kwargs: Any) -> FakeRun:
        raise RuntimeError("synthetic W&B outage")


def test_wandb_failure_keeps_local_record_and_writes_no_receipt(tmp_path: Path) -> None:
    store = ExperimentStore(tmp_path)
    experiment = make_experiment()

    with pytest.raises(RuntimeError, match="outage"):
        WandbPublisher(project="llm-serving-benchmarks", client=FailingWandb()).publish(
            experiment, store
        )

    assert (tmp_path / "exp123" / "experiment.json").exists()
    assert store.load_receipt("exp123") is None


def test_wandb_publisher_requires_a_project(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="WANDB_PROJECT"):
        WandbPublisher(client=FakeWandb()).publish(make_experiment(), ExperimentStore(tmp_path))
