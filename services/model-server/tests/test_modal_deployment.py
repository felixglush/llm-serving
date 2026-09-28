import pytest

import model_server.modal_deployment as modal_deployment
from model_server.domain import ModelConfig, ReplicaTopology
from model_server.modal_deployment import (
    ModalDeploymentBackend,
    ModalDeploymentPlan,
    ModalDeploymentSettings,
    build_modal_app,
)


def test_modal_plan_keeps_one_replica_on_one_gpu() -> None:
    backend = ModalDeploymentBackend()

    plan = backend.plan(
        ModelConfig(
            "Qwen/Qwen3-Embedding-0.6B",
            revision="main",
            max_sequence_length=8_192,
            embedding_dimensions=1_024,
        ),
        ReplicaTopology(replicas=1, tensor_parallel_size=1),
    )

    assert plan.gpu == "L4"
    assert plan.max_containers == 1
    assert plan.container_concurrency == 32
    assert plan.command == (
        "vllm",
        "serve",
        "Qwen/Qwen3-Embedding-0.6B",
        "--runner",
        "pooling",
        "--host",
        "0.0.0.0",
        "--port",
        "8000",
        "--dtype",
        "float16",
        "--revision",
        "main",
        "--max-model-len",
        "8192",
        "--max-num-seqs",
        "32",
        "--max-num-batched-tokens",
        "8192",
        "--tensor-parallel-size",
        "1",
    )


def test_each_serving_configuration_has_a_stable_distinct_identity() -> None:
    backend = ModalDeploymentBackend()
    topology = ReplicaTopology(replicas=1, tensor_parallel_size=1)

    qwen = backend.plan(
        ModelConfig("Qwen/Qwen3-Embedding-0.6B", revision="revision-a"),
        topology,
    )
    qwen_again = backend.plan(
        ModelConfig("Qwen/Qwen3-Embedding-0.6B", revision="revision-a"),
        topology,
    )
    other_revision = backend.plan(
        ModelConfig("Qwen/Qwen3-Embedding-0.6B", revision="revision-b"),
        topology,
    )
    other_model = backend.plan(
        ModelConfig("BAAI/bge-small-en-v1.5", revision="revision-a"),
        topology,
    )

    assert qwen.app_name == qwen_again.app_name
    assert qwen.configuration_fingerprint == qwen_again.configuration_fingerprint
    assert other_revision.app_name != qwen.app_name
    assert other_model.app_name != qwen.app_name
    assert qwen.app_name.startswith("model-server-qwen-qwen3-embedding-0-6b-")


@pytest.mark.parametrize(
    "topology",
    [
        ReplicaTopology(replicas=2, tensor_parallel_size=1),
        ReplicaTopology(replicas=1, tensor_parallel_size=2),
    ],
)
def test_service_rejects_topologies_it_cannot_yet_provision(
    topology: ReplicaTopology,
) -> None:
    with pytest.raises(ValueError, match="model-server supports exactly one replica on one worker"):
        ModalDeploymentBackend().plan(ModelConfig("model"), topology)


@pytest.mark.parametrize("gpu_type", ["L4:2", "L4:1", "L4,L4"])
def test_service_rejects_multi_gpu_resource_syntax(gpu_type: str) -> None:
    with pytest.raises(ValueError, match="gpu_type must name one GPU type"):
        ModalDeploymentSettings(gpu_type=gpu_type)


def test_runtime_and_provider_changes_get_separate_deployments() -> None:
    model = ModelConfig("Qwen/Qwen3-Embedding-0.6B", revision="revision-a")
    topology = ReplicaTopology()
    baseline = ModalDeploymentBackend().plan(model, topology)

    assert (
        ModalDeploymentBackend(ModalDeploymentSettings(gpu_type="A10G"))
        .plan(model, topology)
        .app_name
        != baseline.app_name
    )
    assert (
        ModalDeploymentBackend(ModalDeploymentSettings(container_concurrency=16))
        .plan(model, topology)
        .app_name
        != baseline.app_name
    )
    assert (
        ModalDeploymentBackend(ModalDeploymentSettings(vllm_version="0.28.0"))
        .plan(model, topology)
        .app_name
        != baseline.app_name
    )


def test_unpinned_model_does_not_pass_a_revision_flag() -> None:
    plan = ModalDeploymentBackend().plan(ModelConfig("model"), ReplicaTopology())

    assert "--revision" not in plan.command


def test_deploy_returns_modal_endpoint_for_the_planned_configuration(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    captured_plans: list[ModalDeploymentPlan] = []
    deploy_calls: list[tuple[str, str]] = []

    class FakeApp:
        def deploy(self, *, name: str, strategy: str) -> None:
            deploy_calls.append((name, strategy))

    class FakeWebFunction:
        def get_web_url(self) -> str:
            return "https://example--embeddings.modal.run"

    def fake_build(plan: ModalDeploymentPlan) -> tuple[FakeApp, FakeWebFunction]:
        captured_plans.append(plan)
        return FakeApp(), FakeWebFunction()

    monkeypatch.setattr(modal_deployment, "build_modal_app", fake_build)
    result = ModalDeploymentBackend().deploy(
        ModelConfig("Qwen/Qwen3-Embedding-0.6B", revision="revision-a"),
        ReplicaTopology(),
    )

    assert result.name == captured_plans[0].app_name
    assert result.configuration_fingerprint == captured_plans[0].configuration_fingerprint
    assert result.endpoint_url == "https://example--embeddings.modal.run"
    assert deploy_calls == [(captured_plans[0].app_name, "recreate")]
    assert capsys.readouterr().out == ""


def test_modal_app_can_be_constructed_without_deploying() -> None:
    modal = pytest.importorskip("modal")

    plan = ModalDeploymentBackend().plan(ModelConfig("model"), ReplicaTopology())

    app, web_function = build_modal_app(plan)

    assert isinstance(app, modal.App)
    assert isinstance(web_function, modal.Function)
