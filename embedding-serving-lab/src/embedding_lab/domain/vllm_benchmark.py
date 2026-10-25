"""Persisted records produced by the vLLM reference serving benchmark."""

from dataclasses import dataclass
from typing import Any

from embedding_lab.domain.deployment import Deployment
from embedding_lab.domain.model import ModelConfig
from embedding_lab.domain.topology import ReplicaTopology


@dataclass(frozen=True, slots=True)
class VllmBenchRecord:
    """A versioned wrapper around one raw ``vllm bench serve`` result."""

    experiment_id: str
    created_at: str
    model: ModelConfig
    topology: ReplicaTopology
    deployment: Deployment
    serving_config: dict[str, Any]
    workload: dict[str, Any]
    command: tuple[str, ...]
    metrics: dict[str, float | int]
    raw_result: dict[str, Any]
    schema_version: int = 1

    def __post_init__(self) -> None:
        if not self.experiment_id.strip():
            raise ValueError("experiment_id must not be empty")
        if "/" in self.experiment_id or "\\" in self.experiment_id:
            raise ValueError("experiment_id must be a path-safe identifier")
        if not self.created_at.strip():
            raise ValueError("created_at must not be empty")
        if self.model.revision is None:
            raise ValueError("vllm benchmark model revision must be immutable and provided")
        if self.schema_version != 1:
            raise ValueError("unsupported vllm benchmark schema version")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "experiment_id": self.experiment_id,
            "created_at": self.created_at,
            "source": "vllm-bench",
            "model": {
                "model_id": self.model.model_id,
                "revision": self.model.revision,
                "max_sequence_length": self.model.max_sequence_length,
                "embedding_dimensions": self.model.embedding_dimensions,
            },
            "topology": {
                "replicas": self.topology.replicas,
                "tensor_parallel_size": self.topology.tensor_parallel_size,
            },
            "deployment": {
                "name": self.deployment.name,
                "endpoint_url": self.deployment.endpoint_url,
                "configuration_fingerprint": self.deployment.configuration_fingerprint,
            },
            "serving_config": dict(self.serving_config),
            "workload": dict(self.workload),
            "command": list(self.command),
            "metrics": dict(self.metrics),
            "raw_result": self.raw_result,
        }
