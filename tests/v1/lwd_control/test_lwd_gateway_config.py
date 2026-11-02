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


def test_per_dp_control_url_must_cover_every_cloud_dp() -> None:
    gateway = LwdGatewayConfig(
        enabled=True,
        control_url=(
            "http://gw-a:8100/v1/chat/completions",
            "http://gw-b:8100/v1/chat/completions",
        ),
    )
    gateway.validate_role("edge", cloud_dp_size=2)
    assert gateway.control_url_for(0) == "http://gw-a:8100/v1/chat/completions"
    assert gateway.control_url_for(1) == "http://gw-b:8100/v1/chat/completions"
    with pytest.raises(ValueError, match="one per cloud DP"):
        gateway.validate_role("edge", cloud_dp_size=3)


def test_one_control_url_still_serves_every_cloud_dp() -> None:
    url = "https://gw.example.com/v1/chat/completions"
    gateway = LwdGatewayConfig(enabled=True, control_url=url)
    gateway.validate_role("edge", cloud_dp_size=2)
    assert {gateway.control_url_for(dp) for dp in (0, 1)} == {url}
    assert gateway.control_url_for(None) == url


def test_unknown_dp_has_no_endpoint_with_a_per_dp_list() -> None:
    """Guessing a DP would meter usage the owner DP never published."""
    gateway = LwdGatewayConfig(
        enabled=True,
        control_url=(
            "http://gw-a:8100/v1/chat/completions",
            "http://gw-b:8100/v1/chat/completions",
        ),
    )
    assert gateway.control_url_for(None) is None
    assert gateway.control_url_for(2) is None


def test_a_bad_url_inside_the_per_dp_list_is_rejected() -> None:
    with pytest.raises(ValueError):
        LwdGatewayConfig(
            control_url=(
                "http://gw-a:8100/v1/chat/completions",
                "ftp://gw-b/v1/chat/completions",
            )
        )


def test_cloud_listen_ports_are_per_dp() -> None:
    gateway = LwdGatewayConfig(enabled=True, listen_port=(9188, 9189))
    gateway.validate_role("cloud", cloud_dp_size=2)
    assert [gateway.listen_port_for(dp) for dp in (0, 1)] == [9188, 9189]
    with pytest.raises(ValueError, match="one per cloud DP"):
        gateway.validate_role("cloud", cloud_dp_size=3)


def test_topology_twin_carries_the_per_dp_lists() -> None:
    yaml_block = LwdGateway(
        enabled=True,
        control_url=(
            "http://gw-a:8100/v1/chat/completions",
            "http://gw-b:8100/v1/chat/completions",
        ),
        listen_port=(9188, 9189),
    )
    runtime = LwdGatewayConfig.from_dict(dataclasses.asdict(yaml_block))
    assert runtime.control_url == yaml_block.control_url
    assert runtime.listen_port == yaml_block.listen_port


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
