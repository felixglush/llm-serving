"""Distribution boundaries for independently deployable processes."""

from importlib.metadata import distribution, requires
from pathlib import Path


def test_service_directories_use_public_service_names() -> None:
    services = Path(__file__).parents[1] / "services"

    assert (services / "model-server-simulator").is_dir()
    assert (services / "model-server").is_dir()
    assert not (services / "model-simulator").exists()
    assert not (services / "vllm-model-server").exists()


def test_service_packages_publish_only_the_new_command_names() -> None:
    simulator_commands = {
        entry_point.name: entry_point.value
        for entry_point in distribution("model-server-simulator").entry_points
        if entry_point.group == "console_scripts"
    }
    server_commands = {
        entry_point.name: entry_point.value
        for entry_point in distribution("model-server").entry_points
        if entry_point.group == "console_scripts"
    }

    assert simulator_commands == {"model-server-simulator": "model_server_simulator.main:main"}
    assert server_commands == {"model-server-deploy": "model_server.cli:main"}


def test_research_lab_does_not_import_service_implementation_packages() -> None:
    dependencies = requires("embedding-serving-lab") or []
    forbidden = {
        "model-serving-contracts",
        "model-server-simulator",
        "synthetic-embedding-model",
        "model-server",
    }

    assert not any(dependency.split(" ", 1)[0] in forbidden for dependency in dependencies)
