"""End-to-end acceptance for the gateway and two simulator processes."""

import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx


def _ephemeral_ports(count: int) -> tuple[int, ...]:
    probes: list[socket.socket] = []
    try:
        for _ in range(count):
            probe = socket.socket()
            probe.bind(("127.0.0.1", 0))
            probes.append(probe)
        return tuple(probe.getsockname()[1] for probe in probes)
    finally:
        for probe in probes:
            probe.close()


def _start_process(
    arguments: list[str], *, cwd: Path, environment: dict[str, str]
) -> subprocess.Popen[str]:
    return subprocess.Popen(
        arguments,
        cwd=cwd,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )


def _wait_for_listener(process: subprocess.Popen[str], *, port: int, service_name: str) -> None:
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        if process.poll() is not None:
            output, _ = process.communicate()
            raise AssertionError(
                f"{service_name} exited before readiness with code {process.returncode}:\n{output}"
            )
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.2):
                return
        except OSError:
            time.sleep(0.05)
    raise AssertionError(f"{service_name} did not listen on port {port} within 10 seconds")


def _stop_processes(processes: list[subprocess.Popen[str]]) -> list[str]:
    for process in reversed(processes):
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
    outputs = []
    for process in processes:
        output, _ = process.communicate(timeout=1)
        outputs.append(output)
        assert process.poll() is not None
    return outputs


def test_gateway_routes_embeddings_to_two_real_simulators_in_round_robin(
    tmp_path: Path,
) -> None:
    """Forward four real HTTP calls with stable vectors and alternating backends."""

    simulator_a_port, simulator_b_port, gateway_port = _ephemeral_ports(3)
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    processes: list[subprocess.Popen[str]] = []

    try:
        for backend_id, port in (
            ("backend-a", simulator_a_port),
            ("backend-b", simulator_b_port),
        ):
            simulator = _start_process(
                [
                    sys.executable,
                    "-m",
                    "model_server_simulator.main",
                    "--backend-id",
                    backend_id,
                    "--model",
                    "lab-embedding",
                    "--dimensions",
                    "4",
                    "--host",
                    "127.0.0.1",
                    "--port",
                    str(port),
                ],
                cwd=tmp_path,
                environment=environment,
            )
            processes.append(simulator)
            _wait_for_listener(simulator, port=port, service_name=backend_id)

        gateway = _start_process(
            [
                sys.executable,
                "-m",
                "gateway.main",
                "--backend",
                f"backend-a=http://127.0.0.1:{simulator_a_port}",
                "--backend",
                f"backend-b=http://127.0.0.1:{simulator_b_port}",
                "--model",
                "lab-embedding",
                "--dimensions",
                "4",
                "--request-timeout-seconds",
                "2",
                "--host",
                "127.0.0.1",
                "--port",
                str(gateway_port),
            ],
            cwd=tmp_path,
            environment=environment,
        )
        processes.append(gateway)
        _wait_for_listener(gateway, port=gateway_port, service_name="gateway")

        base_url = f"http://127.0.0.1:{gateway_port}"
        request_body = {
            "model": "lab-embedding",
            "input": ["alpha", "beta"],
            "encoding_format": "float",
        }
        responses = [
            httpx.post(f"{base_url}/v1/embeddings", json=request_body, timeout=3) for _ in range(4)
        ]

        for response in responses:
            assert response.status_code == 200, response.text
        assert [response.headers["X-Backend-ID"] for response in responses] == [
            "backend-a",
            "backend-b",
            "backend-a",
            "backend-b",
        ]

        request_ids = [response.headers["X-Request-ID"] for response in responses]
        assert all(request_ids)
        assert len(set(request_ids)) == 4

        expected_data = [
            {
                "object": "embedding",
                "index": 0,
                "embedding": [
                    0.4056282336823642,
                    0.034747765166684985,
                    0.3293414353393018,
                    0.7117408665362746,
                ],
            },
            {
                "object": "embedding",
                "index": 1,
                "embedding": [
                    0.31422656681388617,
                    0.2911738574039191,
                    0.005320429569110274,
                    0.38812300097197294,
                ],
            },
        ]
        payloads = [response.json() for response in responses]
        assert all(payload["model"] == "lab-embedding" for payload in payloads)
        assert all(payload["data"] == expected_data for payload in payloads)
        assert all(
            [item["index"] for item in payload["data"]] == [0, 1]
            and all(len(item["embedding"]) == 4 for item in payload["data"])
            for payload in payloads
        )

        outputs = _stop_processes(processes)
        processes.clear()
        gateway_output = outputs[-1]
        assert gateway_output.count('"event":"embedding_request_completed"') == 4
        assert '"backend_id":"backend-a"' in gateway_output
        assert '"backend_id":"backend-b"' in gateway_output
        assert all(f'"request_id":"{request_id}"' in gateway_output for request_id in request_ids)
        assert "alpha" not in gateway_output
        assert "beta" not in gateway_output
    finally:
        _stop_processes(processes)


def test_unavailable_selected_backend_is_not_retried_by_the_real_gateway(
    tmp_path: Path,
) -> None:
    unavailable_port, simulator_port, gateway_port = _ephemeral_ports(3)
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    processes: list[subprocess.Popen[str]] = []

    try:
        simulator = _start_process(
            [
                sys.executable,
                "-m",
                "model_server_simulator.main",
                "--backend-id",
                "backend-b",
                "--model",
                "lab-embedding",
                "--dimensions",
                "4",
                "--host",
                "127.0.0.1",
                "--port",
                str(simulator_port),
            ],
            cwd=tmp_path,
            environment=environment,
        )
        processes.append(simulator)
        _wait_for_listener(simulator, port=simulator_port, service_name="backend-b")

        gateway = _start_process(
            [
                sys.executable,
                "-m",
                "gateway.main",
                "--backend",
                f"backend-a=http://127.0.0.1:{unavailable_port}",
                "--backend",
                f"backend-b=http://127.0.0.1:{simulator_port}",
                "--model",
                "lab-embedding",
                "--dimensions",
                "4",
                "--request-timeout-seconds",
                "1",
                "--host",
                "127.0.0.1",
                "--port",
                str(gateway_port),
            ],
            cwd=tmp_path,
            environment=environment,
        )
        processes.append(gateway)
        _wait_for_listener(gateway, port=gateway_port, service_name="gateway")

        request_body = {
            "model": "lab-embedding",
            "input": ["alpha"],
            "encoding_format": "float",
        }
        first = httpx.post(
            f"http://127.0.0.1:{gateway_port}/v1/embeddings",
            json=request_body,
            timeout=2,
        )
        second = httpx.post(
            f"http://127.0.0.1:{gateway_port}/v1/embeddings",
            json=request_body,
            timeout=2,
        )

        assert first.status_code == 503
        assert first.json()["error"]["type"] == "upstream_unavailable"
        assert second.status_code == 200
        assert second.headers["X-Backend-ID"] == "backend-b"

        outputs = _stop_processes(processes)
        processes.clear()
        gateway_output = outputs[-1]
        assert '"backend_id":"backend-a"' in gateway_output
        assert '"outcome":"upstream_unavailable"' in gateway_output
    finally:
        _stop_processes(processes)


def test_real_gateway_deadline_does_not_prevent_the_next_backend_request(
    tmp_path: Path,
) -> None:
    slow_port, fast_port, gateway_port = _ephemeral_ports(3)
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    processes: list[subprocess.Popen[str]] = []

    try:
        for backend_id, port, delay in (
            ("backend-a", slow_port, "0.2"),
            ("backend-b", fast_port, "0"),
        ):
            simulator = _start_process(
                [
                    sys.executable,
                    "-m",
                    "model_server_simulator.main",
                    "--backend-id",
                    backend_id,
                    "--model",
                    "lab-embedding",
                    "--dimensions",
                    "4",
                    "--delay-seconds",
                    delay,
                    "--host",
                    "127.0.0.1",
                    "--port",
                    str(port),
                ],
                cwd=tmp_path,
                environment=environment,
            )
            processes.append(simulator)
            _wait_for_listener(simulator, port=port, service_name=backend_id)

        gateway = _start_process(
            [
                sys.executable,
                "-m",
                "gateway.main",
                "--backend",
                f"backend-a=http://127.0.0.1:{slow_port}",
                "--backend",
                f"backend-b=http://127.0.0.1:{fast_port}",
                "--model",
                "lab-embedding",
                "--dimensions",
                "4",
                "--request-timeout-seconds",
                "0.05",
                "--host",
                "127.0.0.1",
                "--port",
                str(gateway_port),
            ],
            cwd=tmp_path,
            environment=environment,
        )
        processes.append(gateway)
        _wait_for_listener(gateway, port=gateway_port, service_name="gateway")

        request_body = {
            "model": "lab-embedding",
            "input": ["alpha"],
            "encoding_format": "float",
        }
        first = httpx.post(
            f"http://127.0.0.1:{gateway_port}/v1/embeddings",
            json=request_body,
            timeout=2,
        )
        second = httpx.post(
            f"http://127.0.0.1:{gateway_port}/v1/embeddings",
            json=request_body,
            timeout=2,
        )

        assert first.status_code == 504
        assert first.json()["error"]["type"] == "deadline_exceeded"
        assert second.status_code == 200
        assert second.headers["X-Backend-ID"] == "backend-b"

        outputs = _stop_processes(processes)
        processes.clear()
        gateway_output = outputs[-1]
        assert '"backend_id":"backend-a"' in gateway_output
        assert '"outcome":"deadline_exceeded"' in gateway_output
    finally:
        _stop_processes(processes)
