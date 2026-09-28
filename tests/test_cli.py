import json

import pytest

from embedding_lab.cli.main import main


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
