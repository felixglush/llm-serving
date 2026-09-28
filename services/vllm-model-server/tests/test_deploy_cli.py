import json

import pytest

from vllm_model_server.cli import main
from vllm_model_server.domain import Deployment
from vllm_model_server.modal_deployment import ModalDeploymentBackend


def test_vllm_deploy_command_reports_the_modal_endpoint(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    def deploy(self: ModalDeploymentBackend, model: object, topology: object) -> Deployment:
        return Deployment(
            name="embedding-lab-model-r1-tp1-abc123",
            endpoint_url="https://example--embeddings.modal.run",
            configuration_fingerprint="abc123",
        )

    monkeypatch.setattr(ModalDeploymentBackend, "deploy", deploy)

    code = main(
        [
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


def test_deploy_command_does_not_print_success_when_configuration_is_invalid(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit, match="2"):
        main(["--max-num-seqs", "0"])

    captured = capsys.readouterr()
    assert captured.out == ""
    assert "max_num_seqs must be a positive integer" in captured.err
