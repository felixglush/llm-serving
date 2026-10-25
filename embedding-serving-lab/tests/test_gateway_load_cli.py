import importlib
import json
from pathlib import Path
from types import TracebackType

import pytest

from embedding_lab.ports.gateway_client import GatewayCallResult

cli_module = importlib.import_module("embedding_lab.cli.main")


class FakeGatewayClient:
    created_with: dict[str, object] = {}

    def __init__(self, **kwargs: object) -> None:
        type(self).created_with = kwargs
        self.calls = 0

    async def embed(self, texts: list[str]) -> GatewayCallResult:
        backend_id = "backend-a" if self.calls % 2 == 0 else "backend-b"
        self.calls += 1
        return GatewayCallResult(
            backend_id=backend_id,
            vectors=len(texts),
            input_tokens=len(texts),
        )

    async def __aenter__(self) -> "FakeGatewayClient":
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        return None


def test_gateway_load_command_runs_and_persists_before_reporting_success(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cli_module, "HttpxGatewayClient", FakeGatewayClient, raising=False)

    code = cli_module.main(
        [
            "gateway-load",
            "--base-url",
            "http://gateway.test",
            "--model",
            "lab-embedding",
            "--request-rate",
            "1000",
            "--duration-seconds",
            "1",
            "--max-requests",
            "2",
            "--max-outstanding",
            "2",
            "--drain-timeout-seconds",
            "1",
            "--request-timeout-seconds",
            "2",
            "--texts-per-request",
            "2",
            "--target-tokens-per-text",
            "3",
            "--target-label",
            "test-gateway",
            "--target-kind",
            "synthetic",
            "--results-dir",
            str(tmp_path),
        ]
    )

    assert code == 0
    output = json.loads(capsys.readouterr().out)
    record_path = Path(output["record_path"])
    assert record_path.exists()
    assert output["target"]["kind"] == "synthetic"
    assert output["client"]["request_timeout_seconds"] == 2.0
    assert output["result"]["successful_requests"] == 2
    assert output["result"]["backend_counts"] == {
        "backend-a": 1,
        "backend-b": 1,
    }
    assert "observations" not in output["result"]
    assert output["capacity_claim"] == "synthetic evidence; not production capacity"
    assert FakeGatewayClient.created_with["base_url"] == "http://gateway.test"
    assert FakeGatewayClient.created_with["timeout_seconds"] == 2.0


def test_invalid_gateway_load_config_creates_no_artifact(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit, match="2"):
        cli_module.main(
            [
                "gateway-load",
                "--base-url",
                "http://gateway.test",
                "--request-rate",
                "0",
                "--results-dir",
                str(tmp_path),
            ]
        )

    assert "request_rate must be positive and finite" in capsys.readouterr().err
    assert list(tmp_path.iterdir()) == []


def test_gateway_load_does_not_silently_use_the_model_server_environment_url(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("EMBEDDING_BASE_URL", "https://model-server.test")
    monkeypatch.delenv("GATEWAY_BASE_URL", raising=False)

    with pytest.raises(SystemExit, match="2"):
        cli_module.main(["gateway-load"])

    assert "--base-url or GATEWAY_BASE_URL is required" in capsys.readouterr().err


def test_gateway_load_does_not_inherit_the_direct_model_server_model(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cli_module, "HttpxGatewayClient", FakeGatewayClient, raising=False)
    monkeypatch.setenv("GATEWAY_BASE_URL", "http://gateway.test")
    monkeypatch.setenv("EMBEDDING_MODEL", "Qwen/Qwen3-Embedding-0.6B")
    monkeypatch.delenv("GATEWAY_MODEL", raising=False)

    code = cli_module.main(
        [
            "gateway-load",
            "--request-rate",
            "1000",
            "--duration-seconds",
            "1",
            "--max-requests",
            "1",
            "--results-dir",
            str(tmp_path),
        ]
    )

    assert code == 0
    output = json.loads(capsys.readouterr().out)
    assert FakeGatewayClient.created_with["model"] == "lab-embedding"
    assert output["target"]["model"] == "lab-embedding"
