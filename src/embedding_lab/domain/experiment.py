"""Reproducible experiment records independent of a tracking provider."""

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any

from embedding_lab.domain.benchmark import BenchmarkResult, RequestObservation
from embedding_lab.domain.deployment import Deployment
from embedding_lab.domain.model import ModelConfig
from embedding_lab.domain.topology import ReplicaTopology
from embedding_lab.domain.workload import Workload


@dataclass(frozen=True, slots=True)
class Experiment:
    """A saved benchmark and all configuration needed to interpret it."""

    experiment_id: str
    created_at: str
    model: ModelConfig
    topology: ReplicaTopology
    deployment: Deployment
    serving_config: Mapping[str, Any]
    workload: Workload
    result: BenchmarkResult
    schema_version: int = 1

    def __post_init__(self) -> None:
        if not self.experiment_id.strip():
            raise ValueError("experiment_id must not be empty")
        if "/" in self.experiment_id or "\\" in self.experiment_id:
            raise ValueError("experiment_id must be a path-safe identifier")
        if not self.created_at.strip():
            raise ValueError("created_at must not be empty")
        if self.model.revision is None:
            raise ValueError("experiment model revision must be immutable and provided")
        if self.schema_version != 1:
            raise ValueError("unsupported experiment schema version")
        object.__setattr__(self, "serving_config", dict(self.serving_config))

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation with derived metrics included."""

        result = asdict(self.result)
        result.update(
            {
                "failure_rate": self.result.failure_rate,
                "requests_per_second": self.result.requests_per_second,
                "tokens_per_second": self.result.tokens_per_second,
            }
        )
        return {
            "schema_version": self.schema_version,
            "experiment_id": self.experiment_id,
            "created_at": self.created_at,
            "model": asdict(self.model),
            "topology": asdict(self.topology),
            "deployment": asdict(self.deployment),
            "serving_config": dict(self.serving_config),
            "workload": asdict(self.workload),
            "result": result,
        }

    @property
    def observation_dicts(self) -> list[dict[str, Any]]:
        return [asdict(observation) for observation in self.result.observations]

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "Experiment":
        if payload.get("schema_version") != 1:
            raise ValueError("unsupported experiment schema version")
        raw_result = dict(_mapping(payload, "result"))
        raw_observations = raw_result.pop("observations", [])
        for key in ("failure_rate", "requests_per_second", "tokens_per_second"):
            raw_result.pop(key, None)
        raw_result["errors"] = tuple(raw_result.get("errors", ()))
        raw_result["warmup_errors"] = tuple(raw_result.get("warmup_errors", ()))
        result = BenchmarkResult(
            **raw_result,
            observations=tuple(
                RequestObservation(**_mapping_value(item)) for item in raw_observations
            ),
        )
        return cls(
            experiment_id=_string(payload, "experiment_id"),
            created_at=_string(payload, "created_at"),
            model=ModelConfig(**_mapping(payload, "model")),
            topology=ReplicaTopology(**_mapping(payload, "topology")),
            deployment=Deployment(**_mapping(payload, "deployment")),
            serving_config=_mapping(payload, "serving_config"),
            workload=Workload(**_mapping(payload, "workload")),
            result=result,
            schema_version=payload["schema_version"],
        )


def _mapping(payload: Mapping[str, Any], key: str) -> dict[str, Any]:
    value = payload.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"{key} must be an object")
    return dict(value)


def _mapping_value(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("observation must be an object")
    return dict(value)


def _string(payload: Mapping[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str):
        raise ValueError(f"{key} must be a string")
    return value
