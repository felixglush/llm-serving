"""Persistence for sweep manifests that link individual experiments."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class SweepRecord:
    """A versioned index of the experiments in one concurrency sweep."""

    sweep_id: str
    created_at: str
    concurrency_values: tuple[int, ...]
    experiment_ids: tuple[str, ...]
    schema_version: int = 1

    def __post_init__(self) -> None:
        if not self.sweep_id.strip() or "/" in self.sweep_id or "\\" in self.sweep_id:
            raise ValueError("sweep_id must be a path-safe identifier")
        if not self.created_at.strip():
            raise ValueError("created_at must not be empty")
        if self.schema_version != 1:
            raise ValueError("unsupported sweep schema version")
        if not self.concurrency_values:
            raise ValueError("concurrency_values must not be empty")
        if any(value < 1 for value in self.concurrency_values):
            raise ValueError("concurrency_values must contain positive integers")
        if tuple(sorted(self.concurrency_values)) != self.concurrency_values:
            raise ValueError("concurrency_values must be sorted")
        if len(set(self.concurrency_values)) != len(self.concurrency_values):
            raise ValueError("concurrency_values must not contain duplicates")
        if len(self.concurrency_values) != len(self.experiment_ids):
            raise ValueError("each concurrency value must have one experiment")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "sweep_id": self.sweep_id,
            "created_at": self.created_at,
            "concurrency_values": list(self.concurrency_values),
            "experiment_ids": list(self.experiment_ids),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "SweepRecord":
        if payload.get("schema_version") != 1:
            raise ValueError("unsupported sweep schema version")
        return cls(
            sweep_id=_required_string(payload, "sweep_id"),
            created_at=_required_string(payload, "created_at"),
            concurrency_values=tuple(payload.get("concurrency_values", ())),
            experiment_ids=tuple(payload.get("experiment_ids", ())),
            schema_version=payload["schema_version"],
        )


class SweepStore:
    """Store sweep manifests below the same results root as experiments."""

    def __init__(self, root: Path | str = "results") -> None:
        self.root = Path(root)

    def save(self, sweep: SweepRecord) -> Path:
        path = self.root / "sweeps" / sweep.sweep_id / "sweep.json"
        _atomic_json_write(path, sweep.to_dict())
        return path

    def load(self, sweep_id: str) -> SweepRecord:
        path = self.root / "sweeps" / sweep_id / "sweep.json"
        return SweepRecord.from_dict(json.loads(path.read_text(encoding="utf-8")))


def _required_string(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str):
        raise ValueError(f"{key} must be a string")
    return value


def _atomic_json_write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
