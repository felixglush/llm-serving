"""Translate portable serving configuration into a Modal deployment plan."""

import hashlib
import json
import re
import sys
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any

from embedding_lab.adapters.vllm.runtime import VllmEngineConfig
from embedding_lab.domain.deployment import Deployment
from embedding_lab.domain.model import ModelConfig
from embedding_lab.domain.topology import ReplicaTopology


@dataclass(frozen=True, slots=True)
class ModalDeploymentSettings:
    """Modal- and vLLM-specific settings for one serving configuration."""

    app_prefix: str = "embedding-lab"
    gpu_type: str = "L4"
    vllm_version: str = "0.29.0"
    python_version: str = "3.13"
    container_concurrency: int = 32
    max_num_seqs: int = 32
    max_num_batched_tokens: int = 8_192
    port: int = 8_000
    min_containers: int = 0
    max_containers: int = 1
    startup_timeout_seconds: int = 900
    huggingface_cache_volume: str = "embedding-lab-huggingface-cache"
    vllm_cache_volume: str = "embedding-lab-vllm-cache"

    def __post_init__(self) -> None:
        for name in (
            "app_prefix",
            "gpu_type",
            "vllm_version",
            "python_version",
            "huggingface_cache_volume",
            "vllm_cache_volume",
        ):
            if not getattr(self, name).strip():
                raise ValueError(f"{name} must not be empty")
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9+!-]*", self.gpu_type):
            raise ValueError("gpu_type must name one GPU type, without a device count")
        if not re.fullmatch(r"[a-z][a-z0-9-]*", self.app_prefix):
            raise ValueError("app_prefix must be a lowercase Modal app name prefix")
        if len(self.app_prefix) > 45:
            raise ValueError("app_prefix must leave room for a model and deployment identity")
        for name in (
            "container_concurrency",
            "max_num_seqs",
            "max_num_batched_tokens",
            "port",
            "max_containers",
            "startup_timeout_seconds",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.min_containers not in (0, 1):
            raise ValueError("min_containers must be zero or one")
        if self.max_containers != 1:
            raise ValueError("Stage 6 requires max_containers=1")


@dataclass(frozen=True, slots=True)
class ModalDeploymentPlan:
    """A side-effect-free description of the Modal resources to create."""

    app_name: str
    configuration_fingerprint: str
    gpu: str
    vllm_version: str
    python_version: str
    min_containers: int
    max_containers: int
    container_concurrency: int
    startup_timeout_seconds: int
    port: int
    huggingface_cache_volume: str
    vllm_cache_volume: str
    command: tuple[str, ...]


class ModalDeploymentBackend:
    """Prepare and deploy vLLM embedding services on Modal."""

    def __init__(
        self,
        settings: ModalDeploymentSettings | None = None,
        *,
        deployer: Callable[[ModalDeploymentPlan], str] | None = None,
    ) -> None:
        self._settings = settings or ModalDeploymentSettings()
        self._deployer = deployer or _deploy_to_modal

    def deploy(
        self,
        model: ModelConfig,
        topology: ReplicaTopology,
    ) -> Deployment:
        """Deploy a planned app and return its provider-neutral identity."""

        plan = self.plan(model, topology)
        endpoint_url = self._deployer(plan)
        return Deployment(
            name=plan.app_name,
            endpoint_url=endpoint_url,
            configuration_fingerprint=plan.configuration_fingerprint,
        )

    def plan(
        self,
        model: ModelConfig,
        topology: ReplicaTopology,
    ) -> ModalDeploymentPlan:
        """Return the concrete single-replica Modal plan without cloud I/O."""

        if topology != ReplicaTopology(replicas=1, tensor_parallel_size=1):
            raise ValueError("Stage 6 supports exactly one replica on one worker")
        engine = VllmEngineConfig.from_topology(
            topology,
            max_model_len=model.max_sequence_length,
            max_num_seqs=self._settings.max_num_seqs,
            max_num_batched_tokens=self._settings.max_num_batched_tokens,
        )
        command = [
            "vllm",
            "serve",
            model.model_id,
            "--runner",
            "pooling",
            "--host",
            "0.0.0.0",
            "--port",
            str(self._settings.port),
            "--dtype",
            "float16",
        ]
        if model.revision is not None:
            command.extend(("--revision", model.revision))
        command.extend(engine.to_cli_args())

        fingerprint = _configuration_fingerprint(model, topology, self._settings)
        model_slug = re.sub(r"[^a-z0-9]+", "-", model.model_id.lower()).strip("-")
        suffix = f"-r{topology.replicas}-tp{topology.tensor_parallel_size}-{fingerprint[:10]}"
        model_slug = model_slug[: 64 - len(self._settings.app_prefix) - len(suffix) - 1]
        app_name = f"{self._settings.app_prefix}-{model_slug}{suffix}"

        return ModalDeploymentPlan(
            app_name=app_name,
            configuration_fingerprint=fingerprint,
            gpu=self._settings.gpu_type,
            vllm_version=self._settings.vllm_version,
            python_version=self._settings.python_version,
            min_containers=self._settings.min_containers,
            max_containers=self._settings.max_containers,
            container_concurrency=self._settings.container_concurrency,
            startup_timeout_seconds=self._settings.startup_timeout_seconds,
            port=self._settings.port,
            huggingface_cache_volume=self._settings.huggingface_cache_volume,
            vllm_cache_volume=self._settings.vllm_cache_volume,
            command=tuple(command),
        )


def _configuration_fingerprint(
    model: ModelConfig,
    topology: ReplicaTopology,
    settings: ModalDeploymentSettings,
) -> str:
    settings_data = asdict(settings)
    settings_data.pop("app_prefix")
    payload = {
        "model": asdict(model),
        "topology": asdict(topology),
        "modal": settings_data,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def build_modal_app(plan: ModalDeploymentPlan) -> tuple[Any, Any]:
    """Build lazy Modal objects without creating cloud resources."""

    local_python = f"{sys.version_info.major}.{sys.version_info.minor}"
    if local_python != plan.python_version:
        raise RuntimeError(
            "serialized Modal launcher requires matching local and container Python versions: "
            f"local {local_python}, container {plan.python_version}"
        )
    try:
        import modal
    except ImportError as error:
        raise RuntimeError(
            "Modal deployment requires the 'modal' extra: uv sync --extra modal"
        ) from error

    image = modal.Image.debian_slim(
        python_version=plan.python_version
    ).uv_pip_install(f"vllm=={plan.vllm_version}")
    huggingface_cache = modal.Volume.from_name(
        plan.huggingface_cache_volume,
        create_if_missing=True,
    )
    vllm_cache = modal.Volume.from_name(
        plan.vllm_cache_volume,
        create_if_missing=True,
    )
    app = modal.App(plan.app_name)
    command = plan.command

    @app.function(
        image=image,
        gpu=plan.gpu,
        serialized=True,
        min_containers=plan.min_containers,
        max_containers=plan.max_containers,
        volumes={
            "/root/.cache/huggingface": huggingface_cache,
            "/root/.cache/vllm": vllm_cache,
        },
        startup_timeout=plan.startup_timeout_seconds,
        timeout=plan.startup_timeout_seconds,
        name="embeddings",
    )
    @modal.concurrent(max_inputs=plan.container_concurrency)
    @modal.web_server(plan.port, startup_timeout=plan.startup_timeout_seconds)
    def serve() -> None:
        import subprocess

        subprocess.Popen(command)

    return app, serve


def _deploy_to_modal(plan: ModalDeploymentPlan) -> str:
    app, web_function = build_modal_app(plan)
    app.deploy(name=plan.app_name, strategy="recreate")
    endpoint_url = web_function.get_web_url()
    if not isinstance(endpoint_url, str) or not endpoint_url:
        raise RuntimeError("Modal did not expose an HTTP endpoint")
    return endpoint_url
