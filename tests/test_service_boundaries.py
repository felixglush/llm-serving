"""Distribution boundaries for independently deployable processes."""

from importlib.metadata import requires


def test_research_lab_does_not_import_service_implementation_packages() -> None:
    dependencies = requires("embedding-serving-lab") or []
    forbidden = {
        "model-serving-contracts",
        "model-simulator",
        "synthetic-embedding-model",
        "vllm-model-server",
    }

    assert not any(dependency.split(" ", 1)[0] in forbidden for dependency in dependencies)
