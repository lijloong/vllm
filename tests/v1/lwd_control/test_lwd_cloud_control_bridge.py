# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Behavioral tests for direct-DP cloud HTTP coordination."""

from __future__ import annotations

import asyncio
import json
import queue
import socket
import time
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import httpx
import pytest
import pytest_asyncio
import uvicorn

from vllm.v1.lwd_control.control_cloud_scheduler.lwd_cloud_control import (
    LwdCloudControlBridge,
    LwdCloudControlProcessor,
    create_lwd_cloud_control_app,
    start_lwd_cloud_control_server,
)
from vllm.v1.lwd_control.lwd_comm.lwd_prefix import (
    HEADER_EDGE_ID,
    HEADER_ENGINE_INDEX,
    LwdPrefixManifest,
    LwdProbeResult,
    LwdUsage,
)

BridgeFactory = Callable[
    [int],
    tuple[LwdCloudControlBridge, list[queue.Queue[Any]], queue.Queue[Any]],
]


@asynccontextmanager
async def _live_server(app: Any) -> AsyncIterator[str]:
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen()
    port = listener.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(app, log_level="warning"))
    serve_task = asyncio.create_task(server.serve(sockets=[listener]))
    try:
        deadline = time.monotonic() + 5
        while not server.started:
            if serve_task.done():
                await serve_task
            if time.monotonic() >= deadline:
                raise TimeoutError("test HTTP server did not start")
            await asyncio.sleep(0.01)
        yield f"http://127.0.0.1:{port}"
    finally:
        server.should_exit = True
        await serve_task
        listener.close()


@pytest_asyncio.fixture
async def bridge_factory() -> AsyncIterator[BridgeFactory]:
    bridges: list[LwdCloudControlBridge] = []

    def make_bridge(
        engine_count: int,
    ) -> tuple[LwdCloudControlBridge, list[queue.Queue[Any]], queue.Queue[Any]]:
        command_queues: list[queue.Queue[Any]] = [
            queue.Queue() for _ in range(engine_count)
        ]
        event_queue: queue.Queue[Any] = queue.Queue()
        bridge = LwdCloudControlBridge(command_queues, event_queue)
        bridge.start()
        bridges.append(bridge)
        return bridge, command_queues, event_queue

    yield make_bridge

    for bridge in bridges:
        bridge.close()
        bridge._dispatch_thread.join(timeout=1)
        assert not bridge._dispatch_thread.is_alive()


def _manifest(request_id: str) -> LwdPrefixManifest:
    return LwdPrefixManifest(
        request_id=request_id,
        prompt_tokens=16,
        block_size=16,
        full_block_hashes=(bytes(range(32)),),
    )


def test_tail_digest_is_accepted_as_final_message() -> None:
    manifest = LwdPrefixManifest(
        request_id="tail",
        prompt_tokens=18,
        block_size=16,
        full_block_hashes=(bytes(range(32)),),
        tail_hash=bytes(range(31, -1, -1)),
    )
    messages = manifest.to_messages()
    assert messages[0]["content"].startswith("ecb1:")
    assert messages[1]["content"].startswith("ect1:")
    headers = manifest.to_headers()
    assert (
        LwdPrefixManifest.from_openai_request(headers, {"messages": messages})
        == manifest
    )


def _result(request_id: str, hit_blocks: int, engine_index: int = 0) -> LwdProbeResult:
    return LwdProbeResult(
        request_id=request_id,
        instance_id="cloud-test",
        block_size=16,
        hit_blocks=hit_blocks,
        hit_tokens=hit_blocks * 16,
        cloud_id=0,
        engine_index=engine_index,
    )


def _usage_event(
    request_id: str,
    usage: LwdUsage,
    *,
    engine_index: int = 0,
    prefix_nonce: int = 7,
    kind: str = "usage",
) -> dict[str, Any]:
    return {
        "type": kind,
        "request_id": request_id,
        "usage": usage,
        "engine_index": engine_index,
        "prefix_nonce": prefix_nonce,
    }


async def _start_probe(
    bridge: LwdCloudControlBridge,
    command_queue: queue.Queue[Any],
    request_id: str,
    dp_idx: int = 0,
    reply_key: str | None = None,
    timeout: float = 1.0,
) -> tuple[asyncio.Task[LwdProbeResult], dict[str, Any]]:
    key = reply_key or f"{request_id}#dp{dp_idx}"
    task = asyncio.create_task(
        bridge.probe(dp_idx, _manifest(request_id), reply_key=key, timeout=timeout)
    )
    await asyncio.sleep(0)
    command = command_queue.get_nowait()
    assert command["type"] == "probe"
    assert command["engine_index"] == dp_idx
    assert command["reply_key"] == key
    assert command["deadline"] == command["t_cmd"] + timeout
    return task, command


def _reply(
    event_queue: queue.Queue[Any],
    reply_key: str,
    engine_index: int,
    result: LwdProbeResult | None = None,
    error: str | None = None,
) -> None:
    event_queue.put(
        {
            "type": "probe",
            "request_id": reply_key,
            "engine_index": engine_index,
            "ok": result is not None,
            "result": result,
            "error": error,
        }
    )


@pytest.mark.asyncio
async def test_same_request_concurrent_dps_use_independent_reply_keys(
    bridge_factory: BridgeFactory,
) -> None:
    bridge, command_queues, event_queue = bridge_factory(2)
    probe0, _ = await _start_probe(
        bridge, command_queues[0], "shared", dp_idx=0, reply_key="shared#0"
    )
    probe1, _ = await _start_probe(
        bridge, command_queues[1], "shared", dp_idx=1, reply_key="shared#1"
    )
    assert not probe0.done() and not probe1.done()
    assert all(command_queue.empty() for command_queue in command_queues)

    _reply(event_queue, "shared#1", 1, _result("shared", 1, 1))
    _reply(event_queue, "shared#0", 0, _result("shared", 0, 0))
    assert (await probe1).engine_index == 1
    assert (await probe0).engine_index == 0
    assert bridge.registration("shared") is None


@pytest.mark.asyncio
async def test_probe_timeout_cleans_only_probe_and_drops_late_reply(
    bridge_factory: BridgeFactory,
) -> None:
    bridge, command_queues, event_queue = bridge_factory(1)
    probe_task, _ = await _start_probe(
        bridge, command_queues[0], "timeout", reply_key="timeout#0", timeout=0.01
    )
    with pytest.raises(TimeoutError):
        await asyncio.wait_for(probe_task, timeout=1)
    assert not bridge._probe_futures

    _reply(event_queue, "timeout#0", 0, _result("timeout", 1))
    await asyncio.sleep(0.15)
    assert not bridge._probe_futures


@pytest.mark.asyncio
async def test_activate_owns_usage_and_old_release_preserves_new_generation(
    bridge_factory: BridgeFactory,
) -> None:
    bridge, _, event_queue = bridge_factory(1)
    old_registration = bridge.activate("usage", 0, 7)
    new_registration = bridge.activate("usage", 1, 8)
    assert (
        await bridge.wait_usage("usage", timeout=1, registration=old_registration)
        is None
    )
    bridge.release("usage", old_registration)
    assert bridge.registration("usage") is new_registration

    expected = LwdUsage(prompt_tokens=16, completion_tokens=4, cached_tokens=8)
    event_queue.put(
        _usage_event(
            "usage", expected, engine_index=1, prefix_nonce=8, kind="usage_start"
        )
    )
    event_queue.put(_usage_event("usage", expected, engine_index=1, prefix_nonce=8))
    assert (
        await bridge.wait_usage("usage", timeout=1, registration=new_registration)
        is expected
    )
    assert bridge.registration("usage") is None


@pytest.mark.asyncio
@pytest.mark.parametrize(("old_dp", "old_nonce"), [(1, 7)])
async def test_usage_rejects_mismatched_generation(
    bridge_factory: BridgeFactory, old_dp: int, old_nonce: int
) -> None:
    bridge, _, event_queue = bridge_factory(1)
    registration = bridge.activate("generation", 1, 8)
    old_usage = LwdUsage(prompt_tokens=10, completion_tokens=1, cached_tokens=0)
    expected = LwdUsage(prompt_tokens=16, completion_tokens=4, cached_tokens=8)
    waiter = asyncio.create_task(
        bridge.wait_usage("generation", timeout=1, registration=registration)
    )
    event_queue.put(
        _usage_event(
            "generation",
            old_usage,
            engine_index=old_dp,
            prefix_nonce=old_nonce,
            kind="usage_start",
        )
    )
    event_queue.put(
        _usage_event(
            "generation", old_usage, engine_index=old_dp, prefix_nonce=old_nonce
        )
    )
    await asyncio.sleep(0.05)
    assert not waiter.done()

    event_queue.put(
        _usage_event(
            "generation", expected, engine_index=1, prefix_nonce=8, kind="usage_start"
        )
    )
    event_queue.put(
        _usage_event("generation", expected, engine_index=1, prefix_nonce=8)
    )
    assert await waiter is expected


@pytest.mark.asyncio
async def test_usage_from_wrong_dp_is_rejected_immediately(
    bridge_factory: BridgeFactory,
) -> None:
    bridge, _, event_queue = bridge_factory(1)
    registration = bridge.activate("wrong-dp", 1, 8)
    wrong_usage = LwdUsage(prompt_tokens=16, completion_tokens=4, cached_tokens=8)
    waiter = asyncio.create_task(
        bridge.wait_usage("wrong-dp", timeout=1, registration=registration)
    )
    event_queue.put(
        _usage_event("wrong-dp", wrong_usage, engine_index=0, prefix_nonce=8)
    )

    assert await waiter is None
    assert bridge.registration("wrong-dp") is None


@pytest.mark.asyncio
async def test_request_notify_timeout_releases_registration(
    bridge_factory: BridgeFactory,
) -> None:
    bridge, _, _ = bridge_factory(1)
    registration = bridge.activate("admission", 0, 7)
    assert (
        await bridge.wait_usage(
            "admission",
            timeout=1,
            request_notify_timeout=0.01,
            registration=registration,
        )
        is None
    )
    assert bridge.registration("admission") is None


@pytest.mark.asyncio
async def test_started_usage_wait_still_has_total_timeout(
    bridge_factory: BridgeFactory,
) -> None:
    bridge, _, event_queue = bridge_factory(1)
    registration = bridge.activate("long", 0, 7)
    expected = LwdUsage(prompt_tokens=16, completion_tokens=4, cached_tokens=8)
    event_queue.put(_usage_event("long", expected, kind="usage_start"))
    await asyncio.sleep(0.02)
    waiter = asyncio.create_task(
        bridge.wait_usage("long", timeout=0.01, registration=registration)
    )
    await asyncio.sleep(0.02)
    assert await waiter is None
    assert bridge.registration("long") is None


@pytest.mark.asyncio
async def test_probe_cancellation_does_not_remove_activation(
    bridge_factory: BridgeFactory,
) -> None:
    bridge, command_queues, _ = bridge_factory(1)
    registration = bridge.activate("cancel", 0, 7)
    probe_task, _ = await _start_probe(
        bridge, command_queues[0], "cancel", reply_key="cancel#0"
    )
    probe_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await probe_task
    assert not bridge._probe_futures
    assert bridge.registration("cancel") is registration
    bridge.release("cancel")


def test_processor_always_peeks_and_echoes_dp_reply_key() -> None:
    pending_queue: queue.Queue[Any] = queue.Queue()
    event_queue: queue.Queue[Any] = queue.Queue()
    manifest = _manifest("r")
    pending_queue.put(
        {
            "type": "probe",
            "manifest": manifest,
            "reply_key": "r#nonce:7:0:1",
            "engine_index": 1,
        }
    )
    coordinator = Mock()
    coordinator.peek.return_value = _result("r", 1, 1)

    assert LwdCloudControlProcessor(pending_queue, event_queue).poll(coordinator) == 1

    coordinator.peek.assert_called_once_with(manifest)
    coordinator.probe.assert_not_called()
    event = event_queue.get_nowait()
    assert event["request_id"] == "r#nonce:7:0:1"
    assert event["engine_index"] == 1
    assert event["result"] is coordinator.peek.return_value


def test_processor_skips_expired_probe_without_peek() -> None:
    pending_queue: queue.Queue[Any] = queue.Queue()
    event_queue: queue.Queue[Any] = queue.Queue()
    manifest = _manifest("expired")
    now = time.monotonic()
    for index, deadline in enumerate((now - 1, now + 60)):
        pending_queue.put(
            {
                "type": "probe",
                "manifest": manifest,
                "reply_key": f"expired#{index}",
                "engine_index": 0,
                "deadline": deadline,
            }
        )
    coordinator = Mock()
    coordinator.peek.return_value = _result("expired", 0, 0)

    assert LwdCloudControlProcessor(pending_queue, event_queue).poll(coordinator) == 2

    coordinator.peek.assert_called_once_with(manifest)
    expired = event_queue.get_nowait()
    assert not expired["ok"] and expired["error"] == "probe command expired"
    assert event_queue.get_nowait()["ok"]


def test_processor_poll_budget_preserves_remaining_commands() -> None:
    pending_queue: queue.Queue[Any] = queue.Queue()
    event_queue: queue.Queue[Any] = queue.Queue()
    manifest = _manifest("budget")
    for index in range(33):
        pending_queue.put(
            {
                "type": "probe",
                "manifest": manifest,
                "reply_key": f"budget#{index}",
                "engine_index": 0,
                "deadline": time.monotonic() + 60,
            }
        )
    coordinator = Mock()
    coordinator.peek.return_value = _result("budget", 0, 0)
    processor = LwdCloudControlProcessor(pending_queue, event_queue)

    assert processor.poll(coordinator) == 32
    assert pending_queue.qsize() == 1
    assert event_queue.qsize() == 32
    assert processor.poll(coordinator) == 1
    assert pending_queue.empty()
    assert event_queue.qsize() == 33


def test_processor_custom_poll_budget_preserves_remaining_commands() -> None:
    pending_queue: queue.Queue[Any] = queue.Queue()
    event_queue: queue.Queue[Any] = queue.Queue()
    manifest = _manifest("custom-budget")
    for index in range(3):
        pending_queue.put(
            {
                "type": "probe",
                "manifest": manifest,
                "reply_key": f"custom-budget#{index}",
                "engine_index": 0,
                "deadline": time.monotonic() + 60,
            }
        )
    coordinator = Mock()
    coordinator.peek.return_value = _result("custom-budget", 0, 0)
    processor = LwdCloudControlProcessor(pending_queue, event_queue, poll_budget=2)

    assert processor.poll(coordinator) == 2
    assert pending_queue.qsize() == 1
    assert event_queue.qsize() == 2
    assert processor.poll(coordinator) == 1
    assert pending_queue.empty()
    assert event_queue.qsize() == 3


def test_probe_result_supports_scenario_based_dp_identity() -> None:
    result = _result("r", 1, 0)
    headers = result.to_headers()
    message = result.to_control_message()
    assert headers[HEADER_ENGINE_INDEX] == "0"
    assert message["engine_index"] == 0
    assert message["cloud_id"] == 0
    assert LwdProbeResult.from_headers(headers) == result
    assert LwdProbeResult.from_control_message(message) == result

    minimal = {
        "type": "edge_cloud_probe",
        "protocol": "edge-cloud-prefix-v1",
        "request_id": "r",
        "instance_id": "cloud-test",
        "block_size": 16,
        "hit_blocks": 1,
        "hit_tokens": 16,
    }
    inferred = LwdProbeResult.from_control_message(minimal)
    assert inferred.cloud_id is None
    assert inferred.engine_index is None
    assert inferred.alignment_tokens == 0
    assert LwdProbeResult.from_headers(inferred.to_headers()) == inferred


def _http_request(request_id: str = "r") -> tuple[dict[str, str], dict[str, Any]]:
    manifest = _manifest(request_id)
    headers = {**manifest.to_headers(), HEADER_EDGE_ID: "2"}
    body = {
        "model": "test-model",
        "messages": manifest.to_messages(),
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    return headers, body


async def _answer_next_probe(
    command_queue: queue.Queue[Any], event_queue: queue.Queue[Any]
) -> None:
    while command_queue.empty():
        await asyncio.sleep(0)
    command = command_queue.get_nowait()
    event_queue.put(
        {
            "type": "probe",
            "request_id": command["reply_key"],
            "engine_index": command["engine_index"],
            "ok": True,
            "result": _result("e2-r", 1, command["engine_index"]),
        }
    )


@pytest.mark.asyncio
async def test_contract_streams_startup_heartbeat_and_real_usage(
    bridge_factory: BridgeFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "vllm.v1.lwd_control.control_cloud_scheduler.lwd_cloud_control."
        "_USAGE_HEARTBEAT_INTERVAL",
        0.01,
    )
    bridge, command_queues, event_queue = bridge_factory(2)
    app = create_lwd_cloud_control_app(bridge, cloud_id=0, dp_idx=1)
    headers, body = _http_request()
    expected = LwdUsage(prompt_tokens=16, completion_tokens=3, cached_tokens=16)
    responder = asyncio.create_task(_answer_next_probe(command_queues[1], event_queue))

    async with (
        _live_server(app) as base_url,
        httpx.AsyncClient(base_url=base_url) as client,
        client.stream(
            "POST", "/v1/chat/completions", headers=headers, json=body
        ) as response,
    ):
        assert response.status_code == 200
        frames = response.aiter_text()
        startup_frame = await anext(frames)
        empty_frame = await anext(frames)
        startup = json.loads(startup_frame.removeprefix("data: "))
        empty = json.loads(empty_frame.removeprefix("data: "))
        delta = startup["choices"][0]["delta"]
        assert delta["role"] == "assistant"
        assert isinstance(delta["content"], str)
        control = json.loads(delta["content"])
        assert control["type"] == "edge_cloud_probe"
        assert control["request_id"] == "r"
        assert control["cloud_id"] == 0
        assert control["engine_index"] == 1
        assert empty["choices"][0] == {
            "index": 0,
            "delta": {},
            "finish_reason": None,
        }
        assert "usage" not in startup and "usage" not in empty
        assert response.headers["x-edge-cloud-instance-id"] == "cloud-test"
        assert response.headers["x-edge-cloud-prefix-hit-tokens"] == "16"
        assert response.headers["x-accel-buffering"] == "no"

        assert await anext(frames) == ": edge-cloud-usage-pending\n\n"
        registration = bridge.registration("e2-r")
        assert registration is not None
        assert registration.engine_index == control["engine_index"]
        assert control["prefix_nonce"] == registration.prefix_nonce
        event_queue.put(
            _usage_event(
                "e2-r",
                expected,
                engine_index=registration.engine_index,
                prefix_nonce=registration.prefix_nonce,
                kind="usage_start",
            )
        )
        event_queue.put(
            _usage_event(
                "e2-r",
                expected,
                engine_index=registration.engine_index,
                prefix_nonce=registration.prefix_nonce,
            )
        )
        remaining = [frame async for frame in frames]

    await responder
    usage_chunk = json.loads(remaining[0].removeprefix("data: "))
    assert usage_chunk["choices"] == []
    assert '"completion_tokens":3' in remaining[0]
    assert '"cached_tokens":16' in remaining[0]
    assert remaining == [remaining[0], "data: [DONE]\n\n"]
    assert bridge.registration("e2-r") is None


@pytest.mark.asyncio
async def test_bridge_close_after_startup_ends_usage_unavailable(
    bridge_factory: BridgeFactory,
) -> None:
    bridge, command_queues, event_queue = bridge_factory(1)
    app = create_lwd_cloud_control_app(bridge, cloud_id=0, dp_idx=0)
    headers, body = _http_request()
    responder = asyncio.create_task(_answer_next_probe(command_queues[0], event_queue))

    async with (
        _live_server(app) as base_url,
        httpx.AsyncClient(base_url=base_url) as client,
        client.stream(
            "POST", "/v1/chat/completions", headers=headers, json=body
        ) as response,
    ):
        frames = response.aiter_text()
        await anext(frames)
        await anext(frames)
        bridge.close()
        remaining = [frame async for frame in frames]

    await responder
    assert remaining[-2:] == [
        ": edge-cloud-usage-unavailable\n\n",
        "data: [DONE]\n\n",
    ]
    assert all('"usage"' not in frame for frame in remaining)


@pytest.mark.asyncio
async def test_contract_usage_unavailable_never_emits_zero_usage(
    bridge_factory: BridgeFactory,
) -> None:
    bridge, command_queues, event_queue = bridge_factory(1)
    app = create_lwd_cloud_control_app(bridge, cloud_id=0, dp_idx=0)
    headers, body = _http_request()

    async def no_usage(*args: Any, **kwargs: Any) -> None:
        return None

    bridge.wait_usage = no_usage
    responder = asyncio.create_task(_answer_next_probe(command_queues[0], event_queue))
    async with (
        _live_server(app) as base_url,
        httpx.AsyncClient(base_url=base_url) as client,
        client.stream(
            "POST", "/v1/chat/completions", headers=headers, json=body
        ) as response,
    ):
        frames = [frame async for frame in response.aiter_text()]

    await responder
    assert frames[-2:] == [
        ": edge-cloud-usage-unavailable\n\n",
        "data: [DONE]\n\n",
    ]
    assert all('"usage"' not in frame for frame in frames)


@pytest.mark.asyncio
@pytest.mark.parametrize(("field", "value"), [("stream", False), ("max_tokens", 1)])
async def test_contract_rejects_non_whitelisted_body(
    bridge_factory: BridgeFactory, field: str, value: Any
) -> None:
    bridge, command_queues, _ = bridge_factory(1)
    app = create_lwd_cloud_control_app(bridge, cloud_id=0, dp_idx=0)
    transport = httpx.ASGITransport(app=app)
    headers, body = _http_request()
    body[field] = value
    async with httpx.AsyncClient(
        transport=transport, base_url="http://cloud"
    ) as client:
        response = await client.post("/v1/chat/completions", headers=headers, json=body)
    assert response.status_code == 400
    assert command_queues[0].empty()
    assert bridge.registration("e2-r") is None


@pytest.mark.asyncio
async def test_legacy_probe_and_activate_routes_are_removed(
    bridge_factory: BridgeFactory,
) -> None:
    bridge, command_queues, _ = bridge_factory(1)
    app = create_lwd_cloud_control_app(bridge, cloud_id=0, dp_idx=0)
    transport = httpx.ASGITransport(app=app)
    headers, body = _http_request()
    async with httpx.AsyncClient(
        transport=transport, base_url="http://cloud"
    ) as client:
        for path in ("/v1/lwd/probe", "/v1/lwd/activate"):
            response = await client.post(path, headers=headers, json=body)
            assert response.status_code == 404
    assert command_queues[0].empty()
    assert bridge.registration("e2-r") is None


def test_any_dp_bind_failure_fails_whole_server_startup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dps = [SimpleNamespace(dp_idx=index, coord_port=8100 + index) for index in range(2)]
    vllm_config = SimpleNamespace(
        lwd_config=SimpleNamespace(
            role="cloud",
            coordination=SimpleNamespace(enabled=True, listen_host="127.0.0.1"),
            instance=SimpleNamespace(dp=dps),
        )
    )

    def fail_bind(host: str, cloud_dps: Any) -> list[Any]:
        raise OSError("second DP address is occupied")

    monkeypatch.setattr(
        "vllm.v1.lwd_control.control_cloud_scheduler.lwd_cloud_control."
        "_bind_control_sockets",
        fail_bind,
    )
    with pytest.raises(OSError, match="second DP"):
        start_lwd_cloud_control_server(vllm_config)
