"""Package-boundary checks for the real vLLM model-serving service."""

from importlib.metadata import requires
from pathlib import Path

from model_server.modal_deployment import ModalDeploymentSettings
from model_server.runtime import VllmEngineConfig


def test_model_server_and_modal_configuration_are_owned_by_the_service() -> None:
    assert VllmEngineConfig.__module__ == "model_server.runtime"
    assert ModalDeploymentSettings.__module__ == "model_server.modal_deployment"


def test_service_distribution_has_no_repository_runtime_dependencies() -> None:
    dependencies = requires("model-server") or []
    forbidden = {
        "embedding-serving-lab",
        "model-serving-contracts",
        "model-server-simulator",
        "synthetic-embedding-model",
    }

    assert not any(dependency.split(" ", 1)[0] in forbidden for dependency in dependencies)


def test_nvidia_image_does_not_copy_sibling_project_files() -> None:
    dockerfile = (Path(__file__).parents[1] / "Dockerfile").read_text()

    assert "COPY " not in dockerfile
