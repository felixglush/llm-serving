"""Immutable local JSON storage for gateway load experiments."""

import json
import shutil
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any

from embedding_lab.domain.gateway_load import GatewayLoadExperiment


class GatewayLoadStore:
    """Persist one gateway load record and its raw observations."""

    def __init__(self, root: Path | str = "results/gateway-load") -> None:
        self.root = Path(root)

    def save(self, experiment: GatewayLoadExperiment) -> Path:
        self.root.mkdir(parents=True, exist_ok=True)
        directory = self.root / experiment.experiment_id
        if directory.exists():
            raise FileExistsError(f"experiment already exists: {experiment.experiment_id}")
        temporary = Path(tempfile.mkdtemp(prefix=f".{experiment.experiment_id}-", dir=self.root))
        try:
            _json_write(temporary / "gateway-load.json", experiment.to_dict())
            _json_write(
                temporary / "observations.json",
                [asdict(item) for item in experiment.result.observations],
            )
            temporary.rename(directory)
        except BaseException:
            shutil.rmtree(temporary, ignore_errors=True)
            raise
        return directory / "gateway-load.json"

    def load(self, experiment_id: str) -> GatewayLoadExperiment:
        path = self.root / experiment_id / "gateway-load.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("gateway load record must be an object")
        return GatewayLoadExperiment.from_dict(payload)


def _json_write(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
