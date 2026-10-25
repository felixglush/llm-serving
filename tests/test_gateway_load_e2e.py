"""Real bounded HTTP acceptance test for the fixed-arrival gateway load runner."""

import os
import socket
import subprocess
import time
from pathlib import Path
from typing import Any

import httpx
import pytest

from embedding_lab.adapters.filesystem.gateway_load_store import GatewayLoadStore
from embedding_lab.adapters.gateway.http_client import HttpxGatewayClient
from embedding_lab.application.experiments import new_experiment_id, utc_now
from embedding_lab.application.gateway_load import RunGatewayLoad
from embedding_lab.domain.gateway_load import (
    FixedArrivalPlan,
    GatewayLoadClientConfig,
    GatewayLoadExperiment,
    GatewayLoadTarget,
)

RUN_GATEWAY_LOAD_E2E = os.environ.get("RUN_GATEWAY_LOAD_E2E") == "1"
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
COMPOSE_FILE = REPOSITORY_ROOT / "deploy" / "observability" / "compose.yaml"


def _free_port() -> int:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def _compose(
    project: str,
    environment: dict[str, str],
    *arguments: str,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            "docker",
            "compose",
            "--project-name",
            project,
            "--file",
            str(COMPOSE_FILE),
            *arguments,
        ],
        cwd=REPOSITORY_ROOT,
        env=environment,
        check=check,
        capture_output=True,
        text=True,
        timeout=180,
    )


def _wait_for_ready(port: int, timeout_seconds: float = 45.0) -> None:
    deadline = time.monotonic() + timeout_seconds
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            response = httpx.get(f"http://127.0.0.1:{port}/ready", timeout=1)
            response.raise_for_status()
            if response.json() == {"status": "ready"}:
                return
        except (httpx.HTTPError, ValueError) as error:
            last_error = error
        time.sleep(0.2)
    raise AssertionError(f"gateway did not become ready: {last_error}")


def _prometheus_scalar(port: int, expression: str) -> float:
    response = httpx.get(
        f"http://127.0.0.1:{port}/api/v1/query",
        params={"query": expression},
        timeout=3,
    )
    response.raise_for_status()
    payload: dict[str, Any] = response.json()
    result = payload["data"]["result"]
    assert len(result) == 1
    return float(result[0]["value"][1])


@pytest.mark.gateway_load_e2e
@pytest.mark.skipif(
    not RUN_GATEWAY_LOAD_E2E,
    reason="set RUN_GATEWAY_LOAD_E2E=1 to run the real Docker Compose load acceptance test",
)
async def test_fixed_arrival_load_over_real_gateway_records_synthetic_saturation(
    tmp_path: Path,
) -> None:
    ports = {
        "GATEWAY_PORT": str(_free_port()),
        "BACKEND_A_PORT": str(_free_port()),
        "BACKEND_B_PORT": str(_free_port()),
        "PROMETHEUS_PORT": str(_free_port()),
        "GRAFANA_PORT": str(_free_port()),
    }
    environment = (
        os.environ
        | ports
        | {
            "GRAFANA_ADMIN_PASSWORD": "load-e2e-admin",
            "BACKEND_A_DELAY_SECONDS": "0.05",
            "BACKEND_B_DELAY_SECONDS": "0.5",
            "BACKEND_A_MAX_CONCURRENCY": "1",
            "BACKEND_B_MAX_CONCURRENCY": "1",
            "GATEWAY_REQUEST_TIMEOUT_SECONDS": "2",
        }
    )
    project = f"llm-serving-gateway-load-e2e-{os.getpid()}"
    gateway_port = int(ports["GATEWAY_PORT"])
    prometheus_port = int(ports["PROMETHEUS_PORT"])
    base_url = f"http://127.0.0.1:{gateway_port}"

    _compose(project, environment, "config", "--quiet")
    try:
        _compose(project, environment, "up", "--build", "--detach", "--wait")
        _wait_for_ready(gateway_port)

        load_plan = FixedArrivalPlan(
            request_rate=40.0,
            duration_seconds=1.0,
            max_requests=40,
            max_outstanding=16,
            drain_timeout_seconds=3.0,
            texts_per_request=1,
            target_tokens_per_text=4,
            seed=19,
            corpus_version="synthetic-gateway-load-e2e-v1",
        )
        async with HttpxGatewayClient(
            base_url=base_url,
            model="lab-embedding",
            timeout_seconds=2.5,
        ) as client:
            result = await RunGatewayLoad(client).run(load_plan)

        experiment = GatewayLoadExperiment(
            experiment_id=new_experiment_id(),
            created_at=utc_now(),
            target=GatewayLoadTarget(
                base_url=base_url,
                model="lab-embedding",
                label="local-gateway-two-simulators-saturation",
                kind="synthetic",
            ),
            client=GatewayLoadClientConfig(request_timeout_seconds=2.5),
            plan=load_plan,
            result=result,
        )
        store = GatewayLoadStore(tmp_path)
        path = store.save(experiment)

        assert path.exists()
        assert store.load(experiment.experiment_id) == experiment
        assert result.scheduled_requests == 40
        assert result.dispatched_requests + result.generator_dropped_requests == 40
        assert result.max_observed_outstanding <= 16
        assert result.successful_requests < result.scheduled_requests
        assert result.outcome_counts.get("http_error", 0) > 0
        assert result.status_code_counts.get(503, 0) > 0
        assert result.target_request_rate > result.successful_completion_rate_during_arrival_window
        assert result.backend_counts.get("backend-a", 0) > 0
        assert result.backend_counts.get("backend-b", 0) > 0

        deadline = time.monotonic() + 8
        while time.monotonic() < deadline:
            if (
                _prometheus_scalar(prometheus_port, "sum(gateway_upstream_requests_outstanding)")
                == 0
            ):
                break
            time.sleep(0.2)
        assert (
            _prometheus_scalar(prometheus_port, "sum(gateway_upstream_requests_outstanding)") == 0
        )
    finally:
        _compose(
            project,
            environment,
            "down",
            "--volumes",
            "--remove-orphans",
            check=False,
        )
