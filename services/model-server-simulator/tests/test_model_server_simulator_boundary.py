"""Acceptance checks for the independently packaged model-server-simulator service."""

import os
import socket
import subprocess
import sys
import time
from importlib.metadata import requires
from pathlib import Path

import httpx

from model_server_simulator.http_server import ModelServerSimulatorSettings


def test_model_server_simulator_package_owns_its_http_adapter() -> None:
    """The process package must be importable without an embedding_lab HTTP module."""

    assert ModelServerSimulatorSettings.__module__ == "model_server_simulator.http_server"


def test_service_distribution_has_no_repository_runtime_dependencies() -> None:
    """The deployable service owns its domain and synthetic implementation."""

    dependencies = requires("model-server-simulator") or []
    forbidden = {
        "embedding-serving-lab",
        "model-serving-contracts",
        "synthetic-embedding-model",
        "model-server",
    }

    assert not any(dependency.split(" ", 1)[0] in forbidden for dependency in dependencies)


def test_container_build_context_is_the_service_directory() -> None:
    """The image build must not copy source or metadata from sibling projects."""

    dockerfile = (Path(__file__).parents[1] / "Dockerfile").read_text()

    assert "COPY services/" not in dockerfile
    assert "COPY packages/" not in dockerfile
    assert "COPY pyproject.toml uv.lock" not in dockerfile
    assert "COPY requirements.lock pyproject.toml" in dockerfile


def test_installed_module_runs_as_a_real_process_outside_the_checkout(tmp_path: Path) -> None:
    """Exercise the network boundary with no repository path injected into Python."""

    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]

    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "model_server_simulator.main",
            "--backend-id",
            "process-test",
            "--dimensions",
            "4",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
        ],
        cwd=tmp_path,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    base_url = f"http://127.0.0.1:{port}"
    try:
        deadline = time.monotonic() + 10
        while True:
            if process.poll() is not None:
                output = process.stdout.read() if process.stdout is not None else ""
                raise AssertionError(f"model server simulator exited before readiness:\n{output}")
            try:
                health = httpx.get(f"{base_url}/health", timeout=0.2)
                if health.status_code == 200:
                    break
            except httpx.HTTPError:
                pass
            if time.monotonic() >= deadline:
                raise AssertionError(
                    "model server simulator did not become healthy within 10 seconds"
                )
            time.sleep(0.05)

        response = httpx.post(
            f"{base_url}/v1/embeddings",
            json={
                "model": "lab-embedding",
                "input": ["alpha", "beta"],
                "encoding_format": "float",
            },
            timeout=2,
        )

        assert response.status_code == 200
        assert response.headers["X-Backend-ID"] == "process-test"
        assert [item["index"] for item in response.json()["data"]] == [0, 1]
        assert all(len(item["embedding"]) == 4 for item in response.json()["data"])
    finally:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)
