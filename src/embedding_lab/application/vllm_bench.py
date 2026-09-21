"""Run and normalize vLLM's external online serving benchmark."""

import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from embedding_lab.application.experiments import PublicationReceipt
from embedding_lab.domain.vllm_benchmark import VllmBenchRecord


class VllmBenchError(RuntimeError):
    """Raised when the vLLM benchmark process or result is invalid."""


class VllmBenchStore:
    """Store reference benchmark records and their W&B publication receipts."""

    def __init__(self, root: Path | str = "results") -> None:
        self.root = Path(root)

    def directory(self, experiment_id: str) -> Path:
        return self.root / experiment_id

    def save(self, record: VllmBenchRecord) -> Path:
        directory = self.directory(record.experiment_id)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "vllm-bench.json"
        _atomic_json_write(path, record.to_dict())
        return path

    def receipt_path(self, experiment_id: str) -> Path:
        return self.directory(experiment_id) / "wandb-publication.json"

    def load_receipt(self, experiment_id: str) -> PublicationReceipt | None:
        path = self.receipt_path(experiment_id)
        if not path.exists():
            return None
        return PublicationReceipt(**json.loads(path.read_text(encoding="utf-8")))

    def save_receipt(self, receipt: PublicationReceipt) -> Path:
        path = self.receipt_path(receipt.experiment_id)
        _atomic_json_write(path, receipt.to_dict())
        return path


@dataclass(frozen=True, slots=True)
class VllmBenchInvocation:
    """Inputs for one vLLM embedding serving benchmark invocation."""

    executable: str
    base_url: str
    model: str
    result_dir: Path
    num_prompts: int = 100
    max_concurrency: int = 1
    input_len: int = 32
    num_warmups: int = 1
    request_rate: str = "inf"
    seed: int = 0
    endpoint: str = "/v1/embeddings"
    ready_check_timeout_sec: int = 900

    def command(self, result_filename: str) -> list[str]:
        return [
            self.executable,
            "bench",
            "serve",
            "--backend",
            "openai-embeddings",
            "--base-url",
            self.base_url,
            "--endpoint",
            self.endpoint,
            "--model",
            self.model,
            "--dataset-name",
            "random",
            "--num-prompts",
            str(self.num_prompts),
            "--max-concurrency",
            str(self.max_concurrency),
            "--input-len",
            str(self.input_len),
            "--num-warmups",
            str(self.num_warmups),
            "--request-rate",
            self.request_rate,
            "--ready-check-timeout-sec",
            str(self.ready_check_timeout_sec),
            "--seed",
            str(self.seed),
            "--percentile-metrics",
            "e2el",
            "--metric-percentiles",
            "50,95,99",
            "--save-result",
            "--save-detailed",
            "--result-dir",
            str(self.result_dir),
            "--result-filename",
            result_filename,
            "--disable-tqdm",
        ]

    def run(
        self, result_filename: str = "vllm-result.json"
    ) -> tuple[dict[str, Any], list[str], Path]:
        self._validate()
        self.result_dir.mkdir(parents=True, exist_ok=True)
        command = self.command(result_filename)
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            env=os.environ.copy(),
        )
        if completed.returncode != 0:
            detail = completed.stderr.strip() or completed.stdout.strip() or "no output"
            raise VllmBenchError(
                "vllm bench serve failed with "
                f"exit code {completed.returncode}: {detail}"
            )
        result_path = self.result_dir / result_filename
        if not result_path.exists():
            raise VllmBenchError(f"vllm benchmark did not write {result_path}")
        try:
            payload = json.loads(result_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise VllmBenchError(
                f"vllm benchmark result is not valid JSON: {result_path}"
            ) from error
        if not isinstance(payload, dict):
            raise VllmBenchError("vllm benchmark result must be a JSON object")
        return payload, command, result_path

    def _validate(self) -> None:
        if not self.base_url.strip():
            raise ValueError("base_url must not be empty")
        if not self.model.strip():
            raise ValueError("model must not be empty")
        if self.num_prompts < 1 or self.max_concurrency < 1 or self.input_len < 1:
            raise ValueError("num_prompts, max_concurrency, and input_len must be positive")
        if self.num_warmups < 0:
            raise ValueError("num_warmups must not be negative")
        if self.ready_check_timeout_sec < 1:
            raise ValueError("ready_check_timeout_sec must be positive")
        if not self.endpoint.startswith("/"):
            raise ValueError("endpoint must start with '/'")


def normalize_metrics(payload: dict[str, Any]) -> dict[str, float | int]:
    """Map stable vLLM embedding fields without rewriting the raw result."""

    metrics: dict[str, float | int] = {}
    _copy_number(payload, metrics, "completed", "successful_requests")
    _copy_number(payload, metrics, "failed", "failed_requests")
    _copy_number(payload, metrics, "total_input", "input_tokens")
    _copy_number(payload, metrics, "total_input_sequences", "vectors")
    _copy_number(payload, metrics, "request_throughput", "requests_per_second")
    _copy_number(payload, metrics, "input_sequence_throughput", "vectors_per_second")
    _copy_number(payload, metrics, "total_token_throughput", "input_tokens_per_second")
    _copy_number(payload, metrics, "mean_e2el_ms", "mean_latency_ms")
    _copy_number(payload, metrics, "std_e2el_ms", "std_latency_ms")
    _copy_number(payload, metrics, "median_e2el_ms", "p50_latency_ms")
    percentiles = payload.get("percentiles_e2el_ms")
    if isinstance(percentiles, list):
        for item in percentiles:
            if isinstance(item, (list, tuple)) and len(item) == 2:
                percentile, value = item
                if isinstance(percentile, (int, float)) and isinstance(value, (int, float)):
                    if percentile in {50, 95, 99}:
                        metrics[f"p{int(percentile)}_latency_ms"] = value
    duration = (
        payload.get("duration")
        or payload.get("elapsed_time")
        or payload.get("elapsed_seconds")
    )
    if isinstance(duration, (int, float)):
        metrics["elapsed_seconds"] = duration
    return metrics


def _copy_number(
    payload: dict[str, Any], metrics: dict[str, float | int], source: str, target: str
) -> None:
    value = payload.get(source)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        metrics[target] = value


def _atomic_json_write(path: Path, payload: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)
