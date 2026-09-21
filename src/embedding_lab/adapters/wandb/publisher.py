"""Publish local experiment records to W&B Runs and Artifacts."""

import os
from pathlib import Path
from typing import Any

from embedding_lab.application.experiments import (
    ExperimentStore,
    PublicationReceipt,
)
from embedding_lab.domain.experiment import Experiment


class WandbPublisher:
    """Publish one experiment as one resumable W&B Run."""

    def __init__(
        self,
        *,
        project: str | None = None,
        entity: str | None = None,
        client: Any | None = None,
    ) -> None:
        self.project = project or os.environ.get("WANDB_PROJECT")
        self.entity = entity or os.environ.get("WANDB_ENTITY")
        self._client = client

    def publish(self, experiment: Experiment, store: ExperimentStore) -> PublicationReceipt:
        if not self.project:
            raise ValueError("WANDB_PROJECT or an explicit project is required")
        existing = store.load_receipt(experiment.experiment_id)
        if existing is not None:
            if existing.project != self.project or existing.entity != self.entity:
                raise ValueError("existing W&B publication uses a different project or entity")
            return existing

        client = self._client or _load_wandb()
        experiment_path = store.save(experiment)
        artifact_name = f"embedding-experiment-{experiment.experiment_id}"
        config = _config_for(experiment)
        metrics = _metrics_for(experiment)
        group = f"deployment-{experiment.deployment.configuration_fingerprint}"
        init_kwargs: dict[str, Any] = {
            "project": self.project,
            "id": experiment.experiment_id,
            "resume": "allow",
            "name": f"benchmark-{experiment.experiment_id}",
            "job_type": "benchmark",
            "group": group,
            "config": config,
            "tags": ["embedding-benchmark"],
        }
        if self.entity:
            init_kwargs["entity"] = self.entity

        with client.init(**init_kwargs) as run:
            run.log(metrics)
            artifact = client.Artifact(artifact_name, type="benchmark")
            artifact.add_file(str(experiment_path))
            artifact.add_file(str(experiment_path.with_name("observations.json")))
            run.log_artifact(artifact)
            receipt = PublicationReceipt(
                experiment_id=experiment.experiment_id,
                entity=self.entity,
                project=self.project,
                run_id=str(run.id),
                run_url=getattr(run, "url", None),
                artifact_name=artifact_name,
            )
        store.save_receipt(receipt)
        return receipt


def _load_wandb() -> Any:
    import wandb

    return wandb


def _config_for(experiment: Experiment) -> dict[str, Any]:
    return {
        "experiment_id": experiment.experiment_id,
        "model_id": experiment.model.model_id,
        "model_revision": experiment.model.revision,
        "model_max_sequence_length": experiment.model.max_sequence_length,
        "embedding_dimensions": experiment.model.embedding_dimensions,
        "deployment_name": experiment.deployment.name,
        "deployment_fingerprint": experiment.deployment.configuration_fingerprint,
        "endpoint_url": experiment.deployment.endpoint_url,
        "replicas": experiment.topology.replicas,
        "tensor_parallel_size": experiment.topology.tensor_parallel_size,
        "workload_concurrency": experiment.workload.concurrency,
        "workload_requests": experiment.workload.requests,
        "workload_texts_per_request": experiment.workload.texts_per_request,
        "workload_target_tokens_per_text": experiment.workload.target_tokens_per_text,
        "workload_warmup_requests": experiment.workload.warmup_requests,
        "workload_seed": experiment.workload.seed,
        "workload_corpus_version": experiment.workload.corpus_version,
        **{f"serving_{key}": value for key, value in experiment.serving_config.items()},
    }


def _metrics_for(experiment: Experiment) -> dict[str, float | int]:
    result = experiment.result
    metrics: dict[str, float | int | None] = {
        "successful_requests": result.successful_requests,
        "failed_requests": result.failed_requests,
        "failure_rate": result.failure_rate,
        "p50_latency_ms": result.p50_latency_ms,
        "p95_latency_ms": result.p95_latency_ms,
        "p99_latency_ms": result.p99_latency_ms,
        "requests_per_second": result.requests_per_second,
        "vectors_per_second": result.vectors_per_second,
        "input_tokens": result.input_tokens,
        "tokens_per_second": result.tokens_per_second,
        "elapsed_seconds": result.elapsed_seconds,
        "warmup_successful_requests": result.warmup_successful_requests,
        "warmup_failed_requests": result.warmup_failed_requests,
    }
    return {key: value for key, value in metrics.items() if value is not None}
