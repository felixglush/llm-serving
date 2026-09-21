import json

import pytest

from embedding_lab.adapters.modal.deployment import ModalDeploymentBackend
from embedding_lab.cli.main import main
from embedding_lab.domain.deployment import Deployment


def test_verify_command_runs_fake_service(capsys: pytest.CaptureFixture[str]) -> None:
    code = main(["verify", "--text", "alpha", "--text", "beta", "--expected-dimensions", "8"])

    assert code == 0
    output = json.loads(capsys.readouterr().out)
    assert output == {
        "service": "fake",
        "passed": True,
        "input_count": 2,
        "dimensions": 8,
        "failures": [],
    }


def test_verify_command_exits_nonzero_when_dimensions_disagree(
    capsys: pytest.CaptureFixture[str],
) -> None:
    code = main(["verify", "--expected-dimensions", "7"])

    assert code == 1
    output = json.loads(capsys.readouterr().out)
    assert not output["passed"]
    assert any("expected 7" in failure for failure in output["failures"])


def test_vllm_verify_requires_an_endpoint(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("EMBEDDING_BASE_URL", raising=False)

    with pytest.raises(SystemExit, match="2"):
        main(["verify", "--service", "vllm"])

    assert "--base-url or EMBEDDING_BASE_URL is required" in capsys.readouterr().err


def test_vllm_experiment_requires_an_immutable_revision(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit, match="2"):
        main(["experiment", "--service", "vllm", "--base-url", "https://example.test"])

    assert "--revision is required for a persisted vllm experiment" in capsys.readouterr().err


def test_deploy_command_reports_the_modal_endpoint(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    def deploy(
        self: ModalDeploymentBackend, model: object, topology: object
    ) -> Deployment:
        return Deployment(
            name="embedding-lab-model-r1-tp1-abc123",
            endpoint_url="https://example--embeddings.modal.run",
            configuration_fingerprint="abc123",
        )

    monkeypatch.setattr(ModalDeploymentBackend, "deploy", deploy)

    code = main(
        [
            "deploy",
            "--provider",
            "modal",
            "--model",
            "Qwen/Qwen3-Embedding-0.6B",
            "--revision",
            "revision-a",
        ]
    )

    assert code == 0
    assert json.loads(capsys.readouterr().out) == {
        "name": "embedding-lab-model-r1-tp1-abc123",
        "endpoint_url": "https://example--embeddings.modal.run",
        "configuration_fingerprint": "abc123",
    }
