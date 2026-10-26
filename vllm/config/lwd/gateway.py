# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Gateway billing plane config: endpoint, listen socket and stream timings.

Deliberately separate from `coordination`, which describes cross-edge prefix
cache coordination and is only legal in the `cloud_share` scene. Billing
through the platform gateway is orthogonal: it applies to a single edge and a
single cloud, needs no tenant key, and reserves nothing.
"""

from typing import Annotated, Any
from urllib.parse import urlsplit

from pydantic import ConfigDict, Field, StrictBool, field_validator
from pydantic.dataclasses import dataclass

from .topology import LwdPort, LwdPositiveSeconds, LwdString


@dataclass(frozen=True, config=ConfigDict(extra="forbid"))
class LwdGatewayConfig:
    enabled: StrictBool = False
    # Edge-side fields
    control_url: LwdString | None = None
    """Full gateway URL including /v1/chat/completions."""
    consumer_id: LwdString | None = None
    """Billing label; sent as X-Mse-Consumer when set."""
    # Cloud-side fields
    listen_host: LwdString = "0.0.0.0"
    listen_port: LwdPort = 8100
    instance_id: LwdString = "cloud-0"
    """Echoed in the control message so the edge can tell clouds apart."""
    # Shared stream timings
    connect_timeout: Annotated[float, Field(strict=True, gt=0, allow_inf_nan=False)] = (
        10.0
    )
    heartbeat_interval: LwdPositiveSeconds = 10.0
    """Must stay below the gateway's stream idle timeout."""
    usage_wait_timeout: LwdPositiveSeconds = 600.0
    """Upper bound for holding the stream open waiting for the real usage.
    On expiry the stream ends without a usage frame -- never with zeros."""

    @field_validator("control_url")
    @classmethod
    def validate_url(cls, value: str | None) -> str | None:
        if value is not None:
            url = urlsplit(value)
            if url.scheme not in ("http", "https") or not url.hostname:
                raise ValueError("[LWD] control_url must be an HTTP(S) endpoint")
        return value

    @classmethod
    def from_dict(
        cls, raw: dict[str, Any], *, cloud_id: int | None = None
    ) -> "LwdGatewayConfig":
        defaults = {} if cloud_id is None else {"instance_id": f"cloud-{cloud_id}"}
        return cls(**(defaults | raw))

    def validate_role(self, role: str) -> None:
        if not self.enabled:
            return
        if role == "edge" and not self.control_url:
            raise ValueError("[LWD] edge gateway billing requires control_url")
        if role == "cloud" and not 1 <= self.listen_port <= 65535:
            raise ValueError("[LWD] listen_port must be between 1 and 65535")
        if self.heartbeat_interval >= self.usage_wait_timeout:
            raise ValueError(
                "[LWD] heartbeat_interval must be below usage_wait_timeout"
            )
