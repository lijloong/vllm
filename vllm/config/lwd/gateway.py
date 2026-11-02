# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Gateway billing plane config: endpoint, listen socket and stream timings.

Deliberately separate from `coordination`, which describes cross-edge prefix
cache coordination and is only legal in the `cloud_share` scene. Billing
through the platform gateway is orthogonal: it needs no tenant key and
reserves nothing.

Both endpoints are per cloud DP. Every cloud DP is its own process that
publishes usage only for the requests it ran, so a multi-DP deployment has to
address each DP's endpoint separately; a single value means "one endpoint
serves every DP".
"""

from typing import Annotated, Any

from pydantic import ConfigDict, Field, StrictBool, field_validator
from pydantic.dataclasses import dataclass

from .topology import (
    LwdPerDpPort,
    LwdPerDpUrl,
    LwdPositiveSeconds,
    LwdString,
    check_http_url,
)


@dataclass(frozen=True, config=ConfigDict(extra="forbid"))
class LwdGatewayConfig:
    enabled: StrictBool = False
    # Edge-side fields
    control_url: LwdPerDpUrl | None = None
    """Full gateway URL including /v1/chat/completions.

    One URL serves every cloud DP; a list has one entry per cloud DP in
    dp_idx order. Multi-DP needs the list form: usage is published only by
    the DP process that ran the request, so the billing stream has to reach
    that exact DP or it never resolves.
    """
    consumer_id: LwdString | None = None
    """Billing label; sent as X-Mse-Consumer when set."""
    # Cloud-side fields
    listen_host: LwdString = "0.0.0.0"
    listen_port: LwdPerDpPort = 8100
    """Listen port, one per cloud DP when the DPs share a host."""
    instance_id: LwdString = "cloud-0"
    """Echoed in the control message so the edge can tell clouds apart; the
    cloud engine appends the DP rank when several DPs share the instance."""
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
    def validate_url(cls, value: LwdPerDpUrl | None) -> LwdPerDpUrl | None:
        if value is None:
            return value
        urls = (value,) if isinstance(value, str) else value
        for url in urls:
            check_http_url(url)
        return value

    @classmethod
    def from_dict(
        cls, raw: dict[str, Any], *, cloud_id: int | None = None
    ) -> "LwdGatewayConfig":
        defaults = {} if cloud_id is None else {"instance_id": f"cloud-{cloud_id}"}
        return cls(**(defaults | raw))

    def control_url_for(self, dp_idx: int | None) -> str | None:
        """Billing endpoint of one cloud DP, or None when it cannot be known.

        A single URL applies to every DP. With a per-DP list, an unknown or
        out-of-range dp_idx has no endpoint: picking one would meter a DP that
        never ran the request, so that stream is declined instead.
        """
        if self.control_url is None:
            return None
        if isinstance(self.control_url, str):
            return self.control_url
        if dp_idx is None or not 0 <= dp_idx < len(self.control_url):
            return None
        return self.control_url[dp_idx]

    def listen_port_for(self, dp_rank: int) -> int:
        """Listen port of one cloud DP (a per-DP list is indexed by DP)."""
        port = self.listen_port
        if isinstance(port, int):
            return port
        if not 0 <= dp_rank < len(port):
            raise ValueError(
                f"[LWD] no gateway listen_port for cloud DP {dp_rank}; "
                f"the list has {len(port)} entries"
            )
        return port[dp_rank]

    def validate_role(
        self, role: str, *, cloud_dp_size: int | None = None
    ) -> None:
        if not self.enabled:
            return
        if role == "edge":
            if not self.control_url:
                raise ValueError("[LWD] edge gateway billing requires control_url")
            self._check_per_dp("control_url", self.control_url, cloud_dp_size, role)
        if role == "cloud":
            ports = (
                (self.listen_port,)
                if isinstance(self.listen_port, int)
                else self.listen_port
            )
            for port in ports:
                if not 1 <= port <= 65535:
                    raise ValueError("[LWD] listen_port must be between 1 and 65535")
            self._check_per_dp("listen_port", self.listen_port, cloud_dp_size, role)
        if self.heartbeat_interval >= self.usage_wait_timeout:
            raise ValueError(
                "[LWD] heartbeat_interval must be below usage_wait_timeout"
            )

    @staticmethod
    def _check_per_dp(
        name: str,
        value: LwdPerDpUrl | LwdPerDpPort | None,
        cloud_dp_size: int | None,
        role: str,
    ) -> None:
        """A per-DP list must cover every cloud DP, no more and no less."""
        if cloud_dp_size is None or value is None or isinstance(value, (str, int)):
            return
        if len(value) != cloud_dp_size:
            raise ValueError(
                f"[LWD] {role} gateway {name} has {len(value)} entries; "
                f"one per cloud DP is required ({cloud_dp_size})"
            )
