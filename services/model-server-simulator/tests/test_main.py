import httpx
import pytest
from fastapi import FastAPI

from model_server_simulator import main


async def test_model_server_simulator_entrypoint_composes_app_and_runs_one_worker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def capture_run(app: FastAPI, **kwargs: object) -> None:
        captured["app"] = app
        captured["kwargs"] = kwargs

    monkeypatch.setattr("model_server_simulator.main.uvicorn.run", capture_run)

    result = main.main(
        [
            "--backend-id",
            "backend-b",
            "--model",
            "lab-embedding",
            "--dimensions",
            "4",
            "--delay-seconds",
            "0",
            "--max-concurrency",
            "2",
            "--host",
            "127.0.0.1",
            "--port",
            "8123",
            "--fail-every-n-requests",
            "3",
        ]
    )

    assert result == 0
    app = captured["app"]
    assert isinstance(app, FastAPI)
    kwargs = captured["kwargs"]
    assert isinstance(kwargs, dict)
    assert kwargs["workers"] == 1
    assert kwargs["host"] == "127.0.0.1"
    assert kwargs["port"] == 8123

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://testserver",
    ) as client:
        health = await client.get("/health")

    assert health.status_code == 200
    assert health.json() == {
        "status": "ok",
        "backend_id": "backend-b",
        "model": "lab-embedding",
        "dimensions": 4,
        "active_requests": 0,
        "max_concurrency": 2,
    }
