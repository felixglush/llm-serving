import asyncio
import json
from pathlib import Path

import pytest

from embedding_lab.application.sweep import ConcurrencySweep
from embedding_lab.application.sweeps import SweepRecord, SweepStore
from embedding_lab.cli.main import main
from embedding_lab.domain.embedding import Embedding, EmbeddingResponse
from embedding_lab.domain.workload import Workload


class InstrumentedService:
    def __init__(self) -> None:
        self.calls = 0
        self.active = 0
        self.max_active = 0
        self.batches: list[tuple[str, ...]] = []

    async def embed(self, texts: list[str]) -> EmbeddingResponse:
        self.calls += 1
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        self.batches.append(tuple(texts))
        try:
            await asyncio.sleep(0)
            return EmbeddingResponse(
                embeddings=tuple(Embedding((1.0,)) for _ in texts),
                input_tokens=len(texts),
            )
        finally:
            self.active -= 1


async def test_sweep_runs_sorted_points_with_fixed_workload_identity() -> None:
    service = InstrumentedService()
    workload = Workload(
        concurrency=99,
        requests=3,
        texts_per_request=2,
        target_tokens_per_text=4,
        warmup_requests=1,
        seed=7,
        corpus_version="fixture-a",
    )

    points = await ConcurrencySweep(service).run(workload, [4, 1, 2])

    assert [point.concurrency for point in points] == [1, 2, 4]
    assert all(point.workload.requests == 3 for point in points)
    assert all(point.workload.seed == 7 for point in points)
    assert all(point.result.successful_requests == 3 for point in points)
    assert service.calls == 3 * (3 + 1)
    assert service.max_active <= 4


async def test_sweep_reuses_the_same_generated_requests() -> None:
    service = InstrumentedService()
    workload = Workload(
        concurrency=1,
        requests=2,
        texts_per_request=1,
        target_tokens_per_text=2,
        seed=11,
        corpus_version="fixture-a",
    )

    await ConcurrencySweep(service).run(workload, [1, 2])

    assert service.batches[:2] == service.batches[2:]


@pytest.mark.parametrize("values", [[], [0], [1, 1]])
async def test_sweep_rejects_invalid_concurrency_values(values: list[int]) -> None:
    with pytest.raises(ValueError):
        await ConcurrencySweep(
            InstrumentedService()
        ).run(
            Workload(
                concurrency=1,
                requests=1,
                texts_per_request=1,
                target_tokens_per_text=1,
            ),
            values,
        )


def test_sweep_store_round_trips_manifest(tmp_path: Path) -> None:
    record = SweepRecord(
        sweep_id="sweep-a",
        created_at="2026-09-21T12:00:00Z",
        concurrency_values=(1, 4),
        experiment_ids=("sweep-a-c1", "sweep-a-c4"),
    )

    store = SweepStore(tmp_path)
    path = store.save(record)

    assert path == tmp_path / "sweeps" / "sweep-a" / "sweep.json"
    assert store.load("sweep-a") == record


def test_sweep_cli_persists_each_experiment_and_manifest(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = main(
        [
            "sweep",
            "--concurrency-values",
            "4,1,2",
            "--requests",
            "2",
            "--warmup-requests",
            "1",
            "--results-dir",
            str(tmp_path),
            "--sweep-id",
            "fixture-sweep",
        ]
    )

    assert code == 0
    output = json.loads(capsys.readouterr().out)
    assert output["sweep"]["concurrency_values"] == [1, 2, 4]
    assert [row["concurrency"] for row in output["rows"]] == [1, 2, 4]
    assert output["sweep_path"] == str(tmp_path / "sweeps" / "fixture-sweep" / "sweep.json")
    for experiment in output["experiments"]:
        path = Path(experiment["experiment_path"])
        assert path.exists()
        assert json.loads(path.read_text())["workload"]["concurrency"] in {1, 2, 4}
