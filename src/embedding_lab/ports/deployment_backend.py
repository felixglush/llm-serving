"""Boundary implemented by infrastructure-specific deployment adapters."""

from typing import Protocol

from embedding_lab.domain.deployment import Deployment
from embedding_lab.domain.model import ModelConfig
from embedding_lab.domain.topology import ReplicaTopology


class DeploymentBackend(Protocol):
    """Deploy one model using a provider-neutral replica topology."""

    def deploy(
        self,
        model: ModelConfig,
        topology: ReplicaTopology,
    ) -> Deployment: ...
