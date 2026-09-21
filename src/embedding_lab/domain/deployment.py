"""Provider-neutral facts produced by a deployment backend."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Deployment:
    """A deployed serving configuration and its public endpoint."""

    name: str
    endpoint_url: str
    configuration_fingerprint: str

    def __post_init__(self) -> None:
        for field_name in ("name", "endpoint_url", "configuration_fingerprint"):
            if not getattr(self, field_name).strip():
                raise ValueError(f"{field_name} must not be empty")
