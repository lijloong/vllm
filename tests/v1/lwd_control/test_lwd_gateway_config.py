# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Gateway config contract: minimal surface, role-checked, scene-independent.

Unlike `coordination` (prefix cache coordination, `cloud_share` only), the
billing gateway must work for a single edge and a single cloud, so it lives in
its own block and is validated per role instead of per scene.
"""

from __future__ import annotations

import dataclasses

import pytest

from vllm.config.lwd.entry import LwdConfig
from vllm.config.lwd.gateway import LwdGatewayConfig
from vllm.config.lwd.topology import LwdGateway

pytestmark = pytest.mark.cpu_test


def test_defaults_are_inert() -> None:
    gateway = LwdGatewayConfig()
    assert gateway.enabled is False
    assert gateway.control_url is None
    assert gateway.listen_port == 8100
    assert gateway.instance_id == "cloud-0"
    assert gateway.heartbeat_interval == 10.0
    assert gateway.usage_wait_timeout == 600.0
    # disabled means no role requirement at all
    gateway.validate_role("edge")
    gateway.validate_role("cloud")


def test_edge_requires_control_url() -> None:
    with pytest.raises(ValueError, match="control_url"):
        LwdGatewayConfig(enabled=True).validate_role("edge")
    LwdGatewayConfig(
        enabled=True, control_url="https://gw.example.com/v1/chat/completions"
    ).validate_role("edge")


@pytest.mark.parametrize("port", [0, -1, 70000])
def test_cloud_listen_port_range(port: int) -> None:
    with pytest.raises(ValueError):
        LwdGatewayConfig(enabled=True, listen_port=port).validate_role("cloud")


def test_heartbeat_must_stay_below_the_usage_wait() -> None:
    with pytest.raises(ValueError, match="heartbeat_interval"):
        LwdGatewayConfig(
            enabled=True, heartbeat_interval=600.0, usage_wait_timeout=600.0
        ).validate_role("cloud")


@pytest.mark.parametrize(
    "url", ["file:///tmp/x", "ftp://gw/x", "gw.example.com/v1/chat/completions", ""]
)
def test_control_url_must_be_http(url: str) -> None:
    with pytest.raises(ValueError):
        LwdGatewayConfig(control_url=url)


def test_from_dict_derives_the_instance_id_on_the_cloud() -> None:
    cloud = LwdGatewayConfig.from_dict({"enabled": True}, cloud_id=2)
    assert cloud.instance_id == "cloud-2"
    edge = LwdGatewayConfig.from_dict({"enabled": True, "instance_id": "keep"})
    assert edge.instance_id == "keep"


def test_unknown_keys_and_bad_numbers_are_rejected() -> None:
    with pytest.raises(Exception):
        LwdGatewayConfig(unknown=1)  # type: ignore[call-arg]
    with pytest.raises(Exception):
        LwdGatewayConfig(connect_timeout=0)
    with pytest.raises(Exception):
        LwdGatewayConfig(usage_wait_timeout=float("nan"))


def test_topology_twin_round_trips() -> None:
    """The YAML block and the runtime config carry the same fields."""
    yaml_block = LwdGateway(
        enabled=True,
        control_url="http://gw.internal:9133/v1/chat/completions",
        consumer_id="tenant-a",
        listen_port=8100,
        connect_timeout=5.0,
    )
    runtime = LwdGatewayConfig.from_dict(dataclasses.asdict(yaml_block))
    assert runtime.enabled is True
    assert runtime.control_url == yaml_block.control_url
    assert runtime.consumer_id == "tenant-a"
    assert runtime.connect_timeout == 5.0
    with pytest.raises(Exception):
        LwdGateway(not_a_field=1)  # type: ignore[call-arg]


def test_lwd_config_carries_an_optional_gateway_block() -> None:
    names = {field.name for field in dataclasses.fields(LwdConfig)}
    assert "gateway" in names
    defaults = {
        field.name: field.default
        for field in dataclasses.fields(LwdConfig)
        if field.default is not dataclasses.MISSING
    }
    assert defaults["gateway"] is None
