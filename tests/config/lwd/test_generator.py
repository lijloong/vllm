# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import subprocess
import sys
from pathlib import Path

import pytest
import yaml
from lwd_config_under_test import LwdTopology
from sample_topologies import topology_payload

ROOT = Path(__file__).resolve().parents[3]
GENERATOR = ROOT / "tools/gen_lwd_topology.py"


def generate(
    output: Path, **options: str | int | bool | None
) -> subprocess.CompletedProcess[str]:
    command = [sys.executable, str(GENERATOR), "-o", str(output)]
    for name, value in options.items():
        if value is None:
            continue
        flag = f"--{name.replace('_', '-')}"
        # True means a bare store_true flag; anything else takes a value.
        command.append(flag) if value is True else command.extend([flag, str(value)])
    return subprocess.run(command, capture_output=True, text=True)


@pytest.mark.parametrize(
    "scene,edges,clouds,prefix",
    [
        ("single_instance", "10.0.0.1", "10.1.0.1", "s1"),
        ("edge_share", "10.0.0.1", "0@10.1.0.1,1@10.1.0.2", "s2"),
        ("cloud_share", "0@10.0.0.1,1@10.0.0.2", "10.1.0.1", "s3"),
        ("lwd_cluster", "10.0.0.1,10.0.0.2", "10.1.0.1,10.1.0.2", "s4"),
    ],
)
@pytest.mark.parametrize("dp", [1, 2])
def test_explicit_card_counts_reproduce_original_payloads(
    tmp_path: Path, scene: str, edges: str, clouds: str, prefix: str, dp: int
) -> None:
    output = tmp_path / "topology.yaml"
    generate(
        output,
        scene=scene,
        edge_machines=edges,
        cloud_machines=clouds,
        cloud_cards="0:8" if prefix in ("s1", "s3") else "0:8,1:8",
        dp=str(dp),
    ).check_returncode()
    expected = topology_payload(f"{prefix}_{scene}_{dp}dp.yaml")
    assert yaml.safe_load(output.read_text()) == expected
    assert list(tmp_path.iterdir()) == [output]


def test_explicit_colocated_ids_generate_independent_ranks(tmp_path: Path) -> None:
    output = tmp_path / "topology.yaml"
    generate(
        output,
        scene="cloud_share",
        edge_machines="1@76.76.26.17,0@76.76.26.17",
        cloud_machines="76.76.26.232",
        cloud_cards="0:8",
        dp="2",
        port_base="15783",
    ).check_returncode()
    raw = yaml.safe_load(output.read_text())
    assert [[d["ranks"] for d in e["dp"]] for e in raw["edges"]] == [
        [[0], [0]],
        [[1], [1]],
    ]
    assert [d["ctrl_port"] for d in raw["clouds"][0]["dp"]] == [15783, 15784]
    assert raw["deployment"]["hccl_world_size"] == 10


@pytest.mark.parametrize(
    "clouds,cards,dp,expected_ranks,world",
    [
        ("0@10.1.0.1,1@10.1.0.1", "0:2,1:4", 2, [[[1], [2]], [[3, 4], [5, 6]]], 7),
        ("0@10.1.0.1,1@10.1.0.2", "0:4,1:2", 1, [[[1, 2, 3, 4]], [[5, 6]]], 7),
        ("1@10.1.0.1,0@10.1.0.1", "1:4,0:4", 1, [[[1, 2, 3, 4]], [[5, 6, 7, 8]]], 9),
        (
            "0@10.1.0.1,1@10.1.0.1",
            "0:4,1:4",
            2,
            [[[1, 2], [3, 4]], [[5, 6], [7, 8]]],
            9,
        ),
        (
            "0@10.1.0.1,1@10.1.0.2",
            "0:8,1:8",
            1,
            [[list(range(1, 9))], [list(range(9, 17))]],
            17,
        ),
        (
            "0@10.1.0.1,1@10.1.0.2",
            "0:8,1:8",
            2,
            [[[1, 2, 3, 4], [5, 6, 7, 8]], [[9, 10, 11, 12], [13, 14, 15, 16]]],
            17,
        ),
    ],
)
def test_edge_share_allocates_explicit_cards_per_instance(
    tmp_path: Path, clouds: str, cards: str, dp: int, expected_ranks: list, world: int
) -> None:
    output = tmp_path / "topology.yaml"
    generate(
        output,
        scene="edge_share",
        edge_machines="10.0.0.1",
        cloud_machines=clouds,
        cloud_cards=cards,
        dp=str(dp),
        port_base="6020",
    ).check_returncode()
    raw = yaml.safe_load(output.read_text())
    assert raw["deployment"]["hccl_world_size"] == world
    assert [[d["ranks"] for d in c["dp"]] for c in raw["clouds"]] == expected_ranks
    assert [[d["ctrl_port"] for d in c["dp"]] for c in raw["clouds"]] == [
        list(range(6020, 6020 + dp))
    ] * 2
    assert {tuple(d["ranks"]) for e in raw["edges"] for d in e["dp"]} == {(0,)}
    assert raw["instance_links"] == [{"edge": 0, "cloud": 0}, {"edge": 1, "cloud": 1}]


@pytest.mark.parametrize(
    "scene,edges",
    [
        ("single_instance", "10.0.0.1"),
        ("cloud_share", "0@10.0.0.1,1@10.0.0.1"),
    ],
)
@pytest.mark.parametrize("cards,dp", [(4, 1), (4, 2), (16, 2)])
def test_cloud_size_is_not_inferred_from_host_address(
    tmp_path: Path, scene: str, edges: str, cards: int, dp: int
) -> None:
    output = tmp_path / "topology.yaml"
    generate(
        output,
        scene=scene,
        edge_machines=edges,
        cloud_machines="10.1.0.1",
        cloud_cards=f"0:{cards}",
        dp=str(dp),
    ).check_returncode()
    topology = LwdTopology.from_file(str(output))
    edge_count = 1 if scene == "single_instance" else 2
    assert topology.deployment.hccl_world_size == edge_count + cards
    assert [len(d.ranks) for d in topology.clouds[0].dp] == [cards // dp] * dp


@pytest.mark.parametrize(
    "scene,edges,clouds,cards",
    [
        ("single_instance", "10.1.0.1", "10.1.0.1", "0:6"),
        ("cloud_share", "0@10.0.0.1,1@10.0.0.1,2@10.0.0.2", "10.1.0.1", "0:12"),
        ("edge_share", "10.0.0.1", "0@10.1.0.1,1@10.1.0.2,2@10.1.0.1", "0:3,1:6,2:9"),
        (
            "lwd_cluster",
            "0@10.0.0.1,1@10.0.0.1,2@10.0.0.2",
            "0@10.1.0.1,1@10.1.0.2,2@10.1.0.1",
            "0:3,1:6,2:9",
        ),
    ],
)
def test_allocation_rules_scale_beyond_two_instances_and_two_dps(
    tmp_path: Path, scene: str, edges: str, clouds: str, cards: str
) -> None:
    output = tmp_path / "topology.yaml"
    result = generate(
        output,
        scene=scene,
        edge_machines=edges,
        cloud_machines=clouds,
        cloud_cards=cards,
        dp=3,
        port_base=6201,
    )
    assert result.returncode == 0, result.stderr
    topology = LwdTopology.from_file(str(output))
    counts = {
        int(identity): int(count)
        for identity, count in (entry.split(":") for entry in cards.split(","))
    }
    for cloud in topology.clouds:
        assert [len(dp.ranks) for dp in cloud.dp] == [counts[cloud.id] // 3] * 3
    edge_ranks = {r for edge in topology.edges for dp in edge.dp for r in dp.ranks}
    cloud_ranks = [r for cloud in topology.clouds for dp in cloud.dp for r in dp.ranks]
    assert len(cloud_ranks) == len(set(cloud_ranks)) == sum(counts.values())
    assert edge_ranks.isdisjoint(cloud_ranks)
    assert edge_ranks | set(cloud_ranks) == set(
        range(topology.deployment.hccl_world_size)
    )
    if scene == "lwd_cluster":
        endpoints = [
            (dp.addr, dp.ctrl_port) for cloud in topology.clouds for dp in cloud.dp
        ]
        assert len(set(endpoints)) == len(endpoints)


_SCENE_INPUTS = {
    "single_instance": ("10.0.0.1", "10.1.0.1", "0:8"),
    "edge_share": ("10.0.0.1", "0@10.1.0.1,1@10.1.0.2", "0:4,1:4"),
    "cloud_share": ("0@10.0.0.1,1@10.0.0.2", "10.1.0.1", "0:8"),
    "lwd_cluster": ("10.0.0.1,10.0.0.2", "0@10.1.0.1,1@10.1.0.1", "0:4,1:4"),
}


@pytest.mark.parametrize(
    "scene,overrides,match",
    [
        (
            "cloud_share",
            {"edge_machines": "0@10.0.0.1,0@10.0.0.2"},
            "unique and contiguous",
        ),
        (
            "cloud_share",
            {"edge_machines": "0@10.0.0.1,2@10.0.0.2"},
            "unique and contiguous",
        ),
        ("cloud_share", {"edge_machines": "0@10.0.0.1,10.0.0.2"}, "Do not mix"),
        ("cloud_share", {"dp": "1,2"}, "invalid int value"),
        # Match card IDs so these cases reach duplicate/gapped instance ID checks.
        (
            "edge_share",
            {"cloud_machines": "0@10.1.0.1,0@10.1.0.2", "cloud_cards": "0:4"},
            "unique and contiguous",
        ),
        (
            "edge_share",
            {"cloud_machines": "0@10.1.0.1,2@10.1.0.2", "cloud_cards": "0:4,2:4"},
            "unique and contiguous",
        ),
        ("edge_share", {"cloud_machines": "0@10.1.0.1,10.1.0.2"}, "Do not mix"),
        (
            "edge_share",
            {"cloud_machines": "10.1.0.1,10.1.0.1"},
            "--cloud-machines ID@IP",
        ),
        (
            "edge_share",
            {"cloud_machines": "0@10.1.0.1,1@10.1.0.1,2@10.1.0.1"},
            "every cloud instance",
        ),
        ("edge_share", {"dp": 8}, "divisible"),
        ("edge_share", {"dp": 2, "port_base": 65535}, "within 1..65535"),
        ("single_instance", {"cloud_cards": None}, "required"),
        ("single_instance", {"cloud_cards": "0:0"}, "positive"),
        ("single_instance", {"cloud_cards": "0:-1"}, "ID:count"),
        ("single_instance", {"cloud_cards": "0:3", "dp": 2}, "divisible"),
        ("single_instance", {"cloud_cards": "0:4,0:4"}, "duplicate"),
        ("single_instance", {"cloud_cards": "1:4"}, "every cloud instance"),
        ("single_instance", {"cloud_cards": "0:4,1:4"}, "every cloud instance"),
        ("single_instance", {"cloud_cards": "4"}, "ID:count"),
        ("single_instance", {"cloud_cards": "0:4", "dp": 0}, "positive integer"),
        ("lwd_cluster", {"port_base": 65535}, "ctrl_port"),
        (
            "edge_share",
            {"cloud_machines": "10.1.0.1,10.1.0.2"},
            "--cloud-machines ID@IP",
        ),
        ("edge_share", {"edge_machines": "0@10.0.0.1,1@10.0.0.1"}, "one plain edge IP"),
        (
            "cloud_share",
            {"edge_machines": "10.0.0.1,10.0.0.2"},
            "--edge-machines ID@IP",
        ),
    ],
)
def test_rejected_input_preserves_existing_yaml(
    tmp_path: Path, scene: str, overrides: dict[str, str | int | None], match: str
) -> None:
    edges, clouds, cards = _SCENE_INPUTS[scene]
    options: dict[str, str | int | None] = {
        "scene": scene,
        "edge_machines": edges,
        "cloud_machines": clouds,
        "cloud_cards": cards,
        "dp": 1,
        "port_base": 6020,
    }
    output = tmp_path / "topology.yaml"
    output.write_text("keep existing yaml\n")
    result = generate(output, **(options | overrides))
    assert result.returncode != 0
    assert match in result.stderr
    assert output.read_text() == "keep existing yaml\n"


# --------------------------------------------------------------------- #
# gateway billing endpoints                                             #
# --------------------------------------------------------------------- #
_GATEWAY_BASE = {
    "scene": "single_instance",
    "edge_machines": "76.76.26.194",
    "cloud_machines": "76.76.26.192",
    "cloud_cards": "0:16",
    "listen_port": 9188,
    "enable_gateway": True,
}


def _gateway_topology(tmp_path: Path, **options) -> dict:
    output = tmp_path / "topology.yaml"
    generate(output, **(_GATEWAY_BASE | options)).check_returncode()
    return yaml.safe_load(output.read_text())


def test_gateway_single_dp_keeps_scalar_endpoints(tmp_path: Path) -> None:
    raw = _gateway_topology(tmp_path, dp=1)
    assert (
        raw["edges"][0]["gateway"]["control_url"]
        == "http://76.76.26.192:9188/v1/chat/completions"
    )
    assert raw["clouds"][0]["gateway"]["listen_port"] == 9188


def test_gateway_multi_dp_gets_one_endpoint_per_dp(tmp_path: Path) -> None:
    """Every cloud DP is its own process with its own endpoint, so a multi-DP
    deployment must not be handed one shared URL or one shared port."""
    raw = _gateway_topology(tmp_path, dp=2)
    assert raw["clouds"][0]["gateway"]["listen_port"] == [9188, 9189]
    assert raw["edges"][0]["gateway"]["control_url"] == [
        f"http://76.76.26.192:{port}/v1/chat/completions" for port in (9188, 9189)
    ]


def test_gateway_multi_dp_takes_one_explicit_url_per_dp(tmp_path: Path) -> None:
    urls = [
        "http://gw.example.com/dp0/v1/chat/completions",
        "http://gw.example.com/dp1/v1/chat/completions",
    ]
    raw = _gateway_topology(
        tmp_path,
        dp=2,
        gateway_control_urls=",".join(urls),
        listen_port=8100,
    )
    assert raw["edges"][0]["gateway"]["control_url"] == urls


def test_gateway_single_dp_still_reuses_the_control_url(tmp_path: Path) -> None:
    url = "http://gw.internal:9133/v1/chat/completions"
    raw = _gateway_topology(tmp_path, dp=1, control_url=url)
    assert raw["edges"][0]["gateway"]["control_url"] == url


def test_gateway_refuses_one_url_for_several_dps(tmp_path: Path) -> None:
    output = tmp_path / "topology.yaml"
    result = generate(
        output,
        **(
            _GATEWAY_BASE
            | {"dp": 2, "control_url": "http://gw.example.com/v1/chat/completions"}
        ),
    )
    assert result.returncode != 0
    assert "--gateway-control-urls" in result.stderr
    assert not output.exists()


def test_gateway_ports_must_not_collide_with_ctrl_port(tmp_path: Path) -> None:
    output = tmp_path / "topology.yaml"
    result = generate(
        output,
        **(_GATEWAY_BASE | {"dp": 2, "port_base": 9188}),
    )
    assert result.returncode != 0
    assert "collide" in result.stderr
    assert not output.exists()


# --------------------------------------------------------------------- #
# per-DP cloud hosts                                                    #
# --------------------------------------------------------------------- #
def test_cloud_dps_can_live_on_separate_hosts(tmp_path: Path) -> None:
    """`ID@IP1+IP2` places one DP per host; ranks still split by DP."""
    output = tmp_path / "topology.yaml"
    generate(
        output,
        scene="single_instance",
        edge_machines="76.76.26.194",
        cloud_machines="0@76.76.26.192+76.76.26.193",
        cloud_cards="0:16",
        dp=2,
    ).check_returncode()
    raw = yaml.safe_load(output.read_text())
    assert [dp["addr"] for dp in raw["clouds"][0]["dp"]] == [
        "76.76.26.192",
        "76.76.26.193",
    ]
    assert [dp["ranks"] for dp in raw["clouds"][0]["dp"]] == [
        list(range(1, 9)),
        list(range(9, 17)),
    ]
    # separate hosts may share a port; they are not bind conflicts
    assert [dp["ctrl_port"] for dp in raw["clouds"][0]["dp"]] == [5550, 5550]


@pytest.mark.parametrize(
    "clouds,match",
    [
        ("0@76.76.26.192+76.76.26.193+76.76.26.194", "one host per DP (2), got 3"),
        ("0@76.76.26.192+76.76.26.192", "must be distinct"),
    ],
)
def test_cloud_per_dp_hosts_are_validated(
    tmp_path: Path, clouds: str, match: str
) -> None:
    output = tmp_path / "topology.yaml"
    result = generate(
        output,
        scene="single_instance",
        edge_machines="76.76.26.194",
        cloud_machines=clouds,
        cloud_cards="0:16",
        dp=2,
    )
    assert result.returncode != 0
    assert match in result.stderr
    assert not output.exists()


def test_gateway_follows_the_host_of_each_dp(tmp_path: Path) -> None:
    """One DP per host: same port everywhere, URL host per DP."""
    output = tmp_path / "topology.yaml"
    generate(
        output,
        scene="single_instance",
        edge_machines="76.76.26.194",
        cloud_machines="0@76.76.26.192+76.76.26.193",
        cloud_cards="0:16",
        dp=2,
        listen_port=9188,
        enable_gateway=True,
    ).check_returncode()
    raw = yaml.safe_load(output.read_text())
    assert raw["clouds"][0]["gateway"]["listen_port"] == 9188
    assert raw["edges"][0]["gateway"]["control_url"] == [
        "http://76.76.26.192:9188/v1/chat/completions",
        "http://76.76.26.193:9188/v1/chat/completions",
    ]
