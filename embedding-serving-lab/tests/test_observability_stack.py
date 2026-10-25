"""Real Docker Compose acceptance test for the Stage 3 observability stack."""

import asyncio
import os
import socket
import subprocess
import time
from pathlib import Path
from typing import Any

import httpx
import pytest

RUN_OBSERVABILITY_E2E = os.environ.get("RUN_OBSERVABILITY_E2E") == "1"
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


def _wait_for_json(
    url: str,
    *,
    auth: tuple[str, str] | None = None,
    timeout_seconds: float = 45,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_seconds
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            response = httpx.get(url, auth=auth, timeout=1)
            response.raise_for_status()
            payload = response.json()
            if isinstance(payload, dict):
                return payload
        except (httpx.HTTPError, ValueError) as error:
            last_error = error
        time.sleep(0.2)
    raise AssertionError(f"{url} did not return JSON: {last_error}")


def _wait_for_http(url: str, *, timeout_seconds: float = 45) -> httpx.Response:
    deadline = time.monotonic() + timeout_seconds
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            response = httpx.get(url, timeout=1)
            response.raise_for_status()
            return response
        except httpx.HTTPError as error:
            last_error = error
        time.sleep(0.2)
    raise AssertionError(f"{url} did not become ready: {last_error}")


def _prometheus_query(port: int, expression: str) -> list[dict[str, Any]]:
    response = httpx.get(
        f"http://127.0.0.1:{port}/api/v1/query",
        params={"query": expression},
        timeout=3,
    )
    response.raise_for_status()
    payload = response.json()
    assert payload["status"] == "success"
    result = payload["data"]["result"]
    assert isinstance(result, list)
    return result


def _scalar(result: list[dict[str, Any]]) -> float:
    assert len(result) == 1
    return float(result[0]["value"][1])


@pytest.mark.observability_e2e
@pytest.mark.skipif(
    not RUN_OBSERVABILITY_E2E,
    reason="set RUN_OBSERVABILITY_E2E=1 to run the real Docker Compose acceptance test",
)
async def test_live_prometheus_and_grafana_observe_known_gateway_traffic() -> None:
    ports = {
        "GATEWAY_PORT": str(_free_port()),
        "BACKEND_A_PORT": str(_free_port()),
        "BACKEND_B_PORT": str(_free_port()),
        "PROMETHEUS_PORT": str(_free_port()),
        "GRAFANA_PORT": str(_free_port()),
    }
    environment = os.environ | ports | {"GRAFANA_ADMIN_PASSWORD": "e2e-admin"}
    project = f"llm-serving-observability-e2e-{os.getpid()}"
    gateway_port = int(ports["GATEWAY_PORT"])
    prometheus_port = int(ports["PROMETHEUS_PORT"])
    grafana_port = int(ports["GRAFANA_PORT"])

    _compose(project, environment, "config", "--quiet")
    try:
        _compose(project, environment, "up", "--build", "--detach", "--wait")

        assert _wait_for_json(f"http://127.0.0.1:{gateway_port}/ready") == {"status": "ready"}
        assert "Ready" in _wait_for_http(f"http://127.0.0.1:{prometheus_port}/-/ready").text
        assert _wait_for_json(f"http://127.0.0.1:{grafana_port}/api/health")["database"] == ("ok")

        async with httpx.AsyncClient(timeout=10) as client:
            requests = [
                asyncio.create_task(
                    client.post(
                        f"http://127.0.0.1:{gateway_port}/v1/embeddings",
                        json={
                            "model": "lab-embedding",
                            "input": [f"known-{index}"],
                            "encoding_format": "float",
                        },
                    )
                )
                for index in range(6)
            ]

            async with asyncio.timeout(8):
                while True:
                    outstanding_b = _prometheus_query(
                        prometheus_port,
                        'gateway_upstream_requests_outstanding{backend="backend-b"}',
                    )
                    if outstanding_b and _scalar(outstanding_b) > 0:
                        break
                    await asyncio.sleep(0.2)

            responses = await asyncio.gather(*requests)

        assert all(response.status_code == 200 for response in responses)
        assert [response.headers["X-Backend-ID"] for response in responses].count("backend-a") == 3
        assert [response.headers["X-Backend-ID"] for response in responses].count("backend-b") == 3

        async with asyncio.timeout(8):
            while True:
                received = _prometheus_query(
                    prometheus_port,
                    "gateway_embedding_requests_received_total",
                )
                outstanding = _prometheus_query(
                    prometheus_port,
                    "sum(gateway_upstream_requests_outstanding)",
                )
                if (
                    received
                    and _scalar(received) == 6
                    and outstanding
                    and _scalar(outstanding) == 0
                ):
                    break
                await asyncio.sleep(0.2)

        selected_a = _scalar(
            _prometheus_query(
                prometheus_port,
                'gateway_backend_selections_total{backend="backend-a"}',
            )
        )
        selected_b = _scalar(
            _prometheus_query(
                prometheus_port,
                'gateway_backend_selections_total{backend="backend-b"}',
            )
        )
        assert (selected_a, selected_b) == (3, 3)

        average_a = _scalar(
            _prometheus_query(
                prometheus_port,
                "gateway_upstream_request_duration_seconds_sum"
                '{backend="backend-a",outcome="succeeded"}'
                " / gateway_upstream_request_duration_seconds_count"
                '{backend="backend-a",outcome="succeeded"}',
            )
        )
        average_b = _scalar(
            _prometheus_query(
                prometheus_port,
                "gateway_upstream_request_duration_seconds_sum"
                '{backend="backend-b",outcome="succeeded"}'
                " / gateway_upstream_request_duration_seconds_count"
                '{backend="backend-b",outcome="succeeded"}',
            )
        )
        assert average_b > average_a

        targets = _prometheus_query(prometheus_port, 'up{job=~"gateway|model-server-simulator"}')
        assert len(targets) == 3
        assert all(float(item["value"][1]) == 1 for item in targets)

        auth = ("admin", "e2e-admin")
        data_source = _wait_for_json(
            f"http://127.0.0.1:{grafana_port}/api/datasources/uid/prometheus/health",
            auth=auth,
        )
        assert data_source["status"] == "OK"
        dashboard = _wait_for_json(
            f"http://127.0.0.1:{grafana_port}/api/dashboards/uid/embedding-gateway-traffic",
            auth=auth,
        )
        assert dashboard["dashboard"]["title"] == "Embedding Gateway Traffic"
        assert len(dashboard["dashboard"]["panels"]) >= 8

        through_grafana = httpx.get(
            f"http://127.0.0.1:{grafana_port}/api/datasources/proxy/uid/prometheus/api/v1/query",
            params={"query": "sum(rate(gateway_embedding_requests_received_total[1m]))"},
            auth=auth,
            timeout=3,
        )
        through_grafana.raise_for_status()
        assert through_grafana.json()["data"]["result"]
    finally:
        _compose(
            project,
            environment,
            "down",
            "--volumes",
            "--remove-orphans",
            check=False,
        )
