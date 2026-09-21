"""Local experiment persistence and publication coordination."""

import json
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from embedding_lab.domain.experiment import Experiment


@dataclass(frozen=True, slots=True)
class PublicationReceipt:
    """The W&B identity recorded after a successful publication."""

    experiment_id: str
    entity: str | None
    project: str
    run_id: str
    run_url: str | None
    artifact_name: str

    def to_dict(self) -> dict[str, str | None]:
        return {
            "experiment_id": self.experiment_id,
            "entity": self.entity,
            "project": self.project,
            "run_id": self.run_id,
            "run_url": self.run_url,
            "artifact_name": self.artifact_name,
        }


class ExperimentStore:
    """Write and read versioned experiment directories under a results root."""

    def __init__(self, root: Path | str = "results") -> None:
        self.root = Path(root)

    def save(self, experiment: Experiment) -> Path:
        directory = self.root / experiment.experiment_id
        directory.mkdir(parents=True, exist_ok=True)
        experiment_path = directory / "experiment.json"
        _atomic_json_write(experiment_path, experiment.to_dict())
        _atomic_json_write(directory / "observations.json", experiment.observation_dicts)
        return experiment_path

    def load(self, experiment_id: str) -> Experiment:
        path = self.root / experiment_id / "experiment.json"
        return Experiment.from_dict(json.loads(path.read_text(encoding="utf-8")))

    def receipt_path(self, experiment_id: str) -> Path:
        return self.root / experiment_id / "wandb-publication.json"

    def load_receipt(self, experiment_id: str) -> PublicationReceipt | None:
        path = self.receipt_path(experiment_id)
        if not path.exists():
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
        return PublicationReceipt(**payload)

    def save_receipt(self, receipt: PublicationReceipt) -> Path:
        path = self.receipt_path(receipt.experiment_id)
        _atomic_json_write(path, receipt.to_dict())
        return path


def new_experiment_id() -> str:
    return uuid.uuid4().hex[:12]


def utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _atomic_json_write(path: Path, payload: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
