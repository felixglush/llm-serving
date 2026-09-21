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
