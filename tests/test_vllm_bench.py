import json
import sys
from pathlib import Path
from typing import Any

import pytest

from embedding_lab.adapters.wandb.publisher import WandbPublisher
from embedding_lab.application.vllm_bench import (
    VllmBenchInvocation,
    VllmBenchStore,
    normalize_metrics,
)
from embedding_lab.domain.deployment import Deployment
from embedding_lab.domain.model import ModelConfig
from embedding_lab.domain.topology import ReplicaTopology
from embedding_lab.domain.vllm_benchmark import VllmBenchRecord


def test_normalize_metrics_maps_embedding_fields_and_percentiles() -> None:
    metrics = normalize_metrics(
        {
            "completed": 8,
            "failed": 2,
            "total_input": 320,
            "total_input_sequences": 8,
            "request_throughput": 4.0,
            "input_sequence_throughput": 4.0,
            "total_token_throughput": 160.0,
            "percentiles_e2el_ms": [[50, 12.0], [95, 20.0], [99, 28.0]],
        }
    )

    assert metrics == {
        "successful_requests": 8,
        "failed_requests": 2,
        "input_tokens": 320,
        "vectors": 8,
        "requests_per_second": 4.0,
        "vectors_per_second": 4.0,
        "input_tokens_per_second": 160.0,
        "p50_latency_ms": 12.0,
        "p95_latency_ms": 20.0,
        "p99_latency_ms": 28.0,
    }


def test_normalize_metrics_maps_vllm_saved_result_shape() -> None:
    metrics = normalize_metrics(
        {
            "completed": 8,
            "errors": ["", "failed"],
            "total_input_tokens": 256,
            "total_input_sequences": 8,
            "request_throughput": 5.0,
            "input_sequence_throughput": 5.0,
            "total_token_throughput": 160.0,
            "p50_e2el_ms": 12.0,
            "p95_e2el_ms": 20.0,
            "p99_e2el_ms": 28.0,
        }
    )

    assert metrics["failed_requests"] == 1
    assert metrics["input_tokens"] == 256
    assert metrics["p50_latency_ms"] == 12.0
    assert metrics["p95_latency_ms"] == 20.0
    assert metrics["p99_latency_ms"] == 28.0


def test_vllm_invocation_preserves_raw_result_and_uses_expected_flags(tmp_path: Path) -> None:
    executable = tmp_path / "fake-vllm"
    executable.write_text(
        "#!/usr/bin/env " + sys.executable + "\n"
        "import json, pathlib, sys\n"
        "args = sys.argv\n"
        "directory = pathlib.Path(args[args.index('--result-dir') + 1])\n"
        "filename = args[args.index('--result-filename') + 1]\n"
        "directory.mkdir(parents=True, exist_ok=True)\n"
        "(directory / filename).write_text(json.dumps({'completed': 1}))\n",
        encoding="utf-8",
    )
    executable.chmod(0o755)
    invocation = VllmBenchInvocation(
        executable=str(executable),
        base_url="https://example.modal.run",
        model="example/model",
        result_dir=tmp_path / "result",
        num_prompts=1,
        max_concurrency=2,
        num_warmups=1,
    )

    payload, command, result_path = invocation.run()

    assert payload == {"completed": 1}
    assert result_path.read_text(encoding="utf-8") == '{"completed": 1}'
    assert command[0] == str(executable)
    assert "--backend" in command
    assert command[command.index("--backend") + 1] == "openai-embeddings"
    assert command[command.index("--max-concurrency") + 1] == "2"
    assert command[command.index("--ready-check-timeout-sec") + 1] == "900"
    assert command[command.index("--label") + 1] == "vllm-embeddings"


def make_record() -> VllmBenchRecord:
    return VllmBenchRecord(
        experiment_id="vllm123",
        created_at="2026-09-21T12:00:00Z",
        model=ModelConfig("example/model", revision="revision-sha"),
        topology=ReplicaTopology(),
        deployment=Deployment("deployment-a", "https://example.test", "fingerprint-a"),
        serving_config={"vllm_version": "0.29.0", "client_location": "local"},
        workload={"num_prompts": 2, "max_concurrency": 1, "seed": 5},
        command=("vllm", "bench", "serve"),
        metrics={"p95_latency_ms": 12.0, "input_tokens_per_second": 100.0},
        raw_result={"completed": 2, "failed": 0},
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
        self.id = "wandb-vllm-123"
        self.url = "https://wandb.ai/user/project/runs/wandb-vllm-123"
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


class FailingWandb(FakeWandb):
    def init(self, **kwargs: Any) -> FakeRun:
        raise RuntimeError("synthetic W&B outage")


def test_vllm_publication_maps_config_metrics_artifact_and_is_idempotent(tmp_path: Path) -> None:
    store = VllmBenchStore(tmp_path)
    record = make_record()
    raw_path = tmp_path / "raw.json"
    raw_path.write_text(json.dumps(record.raw_result), encoding="utf-8")
    fake = FakeWandb()

    publisher = WandbPublisher(project="llm-serving-benchmarks", entity="user", client=fake)
    first = publisher.publish_vllm_bench(record, store, str(raw_path))
    second = publisher.publish_vllm_bench(record, store, str(raw_path))

    assert first == second
    assert len(fake.init_calls) == 1
    assert fake.init_calls[0]["config"]["benchmark_source"] == "vllm-bench"
    assert fake.init_calls[0]["config"]["input_tokens_source"] == "vllm_prompt_len"
    assert fake.runs[0].logs == [record.metrics]
    assert {Path(path).name for path in fake.runs[0].artifacts[0].files} == {
        "vllm-bench.json",
        "raw.json",
    }
    assert store.load_receipt(record.experiment_id) == first


def test_vllm_publication_failure_keeps_record_without_receipt(tmp_path: Path) -> None:
    store = VllmBenchStore(tmp_path)
    record = make_record()
    raw_path = tmp_path / "raw.json"
    raw_path.write_text(json.dumps(record.raw_result), encoding="utf-8")

    with pytest.raises(RuntimeError, match="outage"):
        WandbPublisher(project="llm-serving-benchmarks", client=FailingWandb()).publish_vllm_bench(
            record, store, str(raw_path)
        )

    assert (tmp_path / "vllm123" / "vllm-bench.json").exists()
    assert store.load_receipt(record.experiment_id) is None
