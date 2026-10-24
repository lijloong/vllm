# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Cloud gateway endpoint contract, driven without a socket.

The route functions are called directly with a duck-typed Request, so no
uvicorn/httpx is needed. What is pinned here is what the platform gateway
sees: frame order, the usage shape, and that a request whose usage never
arrives ends *without* a usage frame instead of billing zeros.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from vllm.config.lwd.gateway import LwdGatewayConfig
from vllm.v1.lwd_control.lwd_gateway import lwd_gateway_wire as wire
from vllm.v1.lwd_control.lwd_gateway.lwd_gateway_server import (
    LwdGatewayControlServer,
    create_lwd_gateway_app,
)

pytestmark = pytest.mark.cpu_test

RID = "chatcmpl-abc123"
MODEL = "qwen3.5"


class _FakeRequest:
    def __init__(self, headers: dict[str, str], body: bytes) -> None:
        self.headers = {key.lower(): value for key, value in headers.items()}
        self._body = body

    async def json(self):
        return json.loads(self._body)


def _gateway(**overrides) -> LwdGatewayConfig:
    base = {
        "enabled": True,
        "listen_port": 8100,
        "instance_id": "cloud-0",
        "heartbeat_interval": 5.0,
        "usage_wait_timeout": 5.0,
    }
    return LwdGatewayConfig(**(base | overrides))


def _server(**overrides) -> LwdGatewayControlServer:
    return LwdGatewayControlServer(_gateway(**overrides))


def _handler(app, path: str, method: str):
    for route in app.routes:
        if route.path == path and method in getattr(route, "methods", ()):
            return route.endpoint
    raise AssertionError(f"route {method} {path} not registered")


def _post(app, headers: dict[str, str], body: dict):
    request = _FakeRequest(headers, json.dumps(body).encode())
    return asyncio.run(_handler(app, "/v1/chat/completions", "POST")(request))


def _classify(chunk) -> tuple[str, object]:
    text = (chunk.decode() if isinstance(chunk, bytes) else chunk).strip()
    if text.startswith(":"):
        return "comment", text.lstrip(":").strip()
    assert text.startswith("data:"), text
    payload = text[len("data:") :].strip()
    if payload == "[DONE]":
        return "done", None
    return "data", json.loads(payload)


def _stream(response):
    async def _collect():
        return [_classify(chunk) async for chunk in response.body_iterator]

    return response.status_code, dict(response.headers), asyncio.run(_collect())


def _request(headers: dict[str, str], body: dict) -> _FakeRequest:
    return _FakeRequest(headers, json.dumps(body).encode())


def _shadow(prompt_tokens: int = 1024, **header_overrides):
    headers, body = wire.build_shadow_request(RID, prompt_tokens, MODEL)
    headers.update(header_overrides)
    return headers, body


def _control(events) -> dict:
    for kind, payload in events:
        if kind != "data":
            continue
        choices = payload.get("choices") or []
        if not choices:
            continue
        content = (choices[0].get("delta") or {}).get("content")
        if content:
            return json.loads(content)
    raise AssertionError("no control message frame")


# --------------------------------------------------------------------- #
# happy path                                                             #
# --------------------------------------------------------------------- #
def test_stream_carries_control_message_then_usage_then_done() -> None:
    server = _server()
    # Usage finished before the billing stream showed up (the edge never
    # waits, so this order is normal): the late cache must still bill it.
    server.publish_usage(RID, 1024, 128)
    app = create_lwd_gateway_app(server)
    status, headers, events = _stream(_post(app, *_shadow()))

    assert status == 200
    assert headers["content-type"].startswith("text/event-stream")
    assert headers[wire.HEADER_REQUEST_ID.lower()] == RID
    assert headers[wire.HEADER_INSTANCE_ID.lower()] == "cloud-0"
    assert headers["x-accel-buffering"] == "no"

    kinds = [kind for kind, _ in events]
    assert kinds == ["data", "data", "data", "done"]
    assert _control(events) == wire.control_message(RID, "cloud-0", 256)
    assert events[1][1]["choices"] == [
        {"index": 0, "delta": {}, "finish_reason": None}
    ]
    assert events[2][1]["choices"] == []
    assert events[2][1]["usage"] == {
        "prompt_tokens": 1024,
        "completion_tokens": 128,
        "total_tokens": 1152,
        "prompt_tokens_details": {"cached_tokens": 0},
    }
    assert server.pending() == 0


def test_heartbeats_keep_the_stream_alive_until_usage_lands() -> None:
    server = _server(heartbeat_interval=0.01, usage_wait_timeout=5.0)
    app = create_lwd_gateway_app(server)

    async def scenario():
        server.attach_loop(asyncio.get_running_loop())
        publish = asyncio.create_task(_publish_after(server, 0.05, RID, 64, 8))
        response = await _handler(app, "/v1/chat/completions", "POST")(
            _request(*_shadow())
        )
        events = [_classify(chunk) async for chunk in response.body_iterator]
        await publish
        return events

    events = asyncio.run(scenario())
    comments = [payload for kind, payload in events if kind == "comment"]
    assert comments == ["edge-cloud-usage-pending"] * len(comments)
    assert len(comments) >= 1
    assert events[-1] == ("done", None)
    assert events[-2][1]["usage"]["prompt_tokens"] == 64


async def _publish_after(server, delay, rid, prompt, completion) -> None:
    await asyncio.sleep(delay)
    server.publish_usage(rid, prompt, completion)


# --------------------------------------------------------------------- #
# usage never arrives                                                    #
# --------------------------------------------------------------------- #
def test_missing_usage_ends_the_stream_without_billing_zeros() -> None:
    server = _server(heartbeat_interval=0.01, usage_wait_timeout=0.05)
    app = create_lwd_gateway_app(server)
    _, _, events = _stream(_post(app, *_shadow()))

    data_frames = [payload for kind, payload in events if kind == "data"]
    assert all("usage" not in frame for frame in data_frames)
    assert ("comment", "edge-cloud-usage-unavailable") in events
    assert events[-1] == ("done", None)
    assert server.pending() == 0


def test_disconnect_right_after_the_control_message_leaves_nothing() -> None:
    """Disconnect before the waiter exists: nothing was registered yet."""
    server = _server()
    app = create_lwd_gateway_app(server)

    async def scenario():
        response = await _handler(app, "/v1/chat/completions", "POST")(
            _request(*_shadow())
        )
        stream = response.body_iterator
        await stream.__anext__()  # control message reaches the gateway
        await stream.aclose()  # ...then the client goes away
        assert server.pending() == 0

    asyncio.run(scenario())
    assert server.pending() == 0


def test_abort_settles_a_waiting_stream_immediately_with_no_usage() -> None:
    """A declined abort ends the stream now, with no usage frame.

    The aborted request's declared prompt is only *nominal* (its prefill might
    be incomplete), so it must never be billed -- but leaving the gateway
    connection parked until usage_wait_timeout would be pure overhead.
    """
    server = _server(heartbeat_interval=0.01, usage_wait_timeout=30.0)
    app = create_lwd_gateway_app(server)
    post = _handler(app, "/v1/chat/completions", "POST")

    async def scenario():
        server.attach_loop(asyncio.get_running_loop())
        response = await post(_request(*_shadow()))

        async def collect():
            return [_classify(chunk) async for chunk in response.body_iterator]

        task = asyncio.create_task(collect())
        await asyncio.sleep(0.02)
        assert server.pending() == 1
        server.publish_aborted(RID)
        # If abort lost the waiter, this hits the wait_for timeout instead.
        return await asyncio.wait_for(task, timeout=1.0)

    events = asyncio.run(scenario())
    data_frames = [payload for kind, payload in events if kind == "data"]
    assert all("usage" not in frame for frame in data_frames)
    assert ("comment", "edge-cloud-usage-unavailable") in events
    assert events[-1] == ("done", None)
    assert server.pending() == 0


def test_abort_marker_does_not_poison_a_later_same_rid_stream() -> None:
    """Aborts are never late-cached: a retry for the same id still bills.

    An abort marker settling a future with nothing would also poison any
    *later* stream for the same request id (client retry or id reuse), which
    is exactly the real attempt we must not lose.
    """
    server = _server(heartbeat_interval=0.01, usage_wait_timeout=30.0)
    app = create_lwd_gateway_app(server)
    post = _handler(app, "/v1/chat/completions", "POST")

    async def scenario():
        server.attach_loop(asyncio.get_running_loop())
        server.publish_aborted(RID)  # nobody is waiting: must be a no-op
        await asyncio.sleep(0.05)
        server.publish_usage(RID, 100, 20)
        response = await post(_request(*_shadow()))
        return [_classify(chunk) async for chunk in response.body_iterator]

    events = asyncio.run(scenario())
    usage_frames = [
        payload for kind, payload in events if kind == "data" and "usage" in payload
    ]
    assert len(usage_frames) == 1
    assert usage_frames[0]["usage"]["prompt_tokens"] == 100
    assert events[-1] == ("done", None)
    assert server.pending() == 0


def test_disconnect_during_heartbeats_drops_the_waiter_at_once() -> None:
    """A disconnect mid-inference must not leave the id registered.

    The client going away surfaces inside the frame generator as
    GeneratorExit, which is *not* CancelledError: without a finally around the
    heartbeat loop the waiter keeps the request id until usage_wait_timeout,
    and a usage event arriving in that window is consumed by nobody -- a
    retried stream for the same id would then bill nothing.
    """
    server = _server(heartbeat_interval=0.01, usage_wait_timeout=30.0)
    app = create_lwd_gateway_app(server)

    async def scenario():
        response = await _handler(app, "/v1/chat/completions", "POST")(
            _request(*_shadow())
        )
        stream = response.body_iterator
        await stream.__anext__()  # control message
        await stream.__anext__()  # empty delta
        kind, _ = _classify(await stream.__anext__())  # first heartbeat
        assert kind == "comment"
        assert server.pending() == 1  # the waiter is live now
        await stream.aclose()
        # Cancel only; the waiter's own finally runs on the next loop pass.
        for _ in range(3):
            await asyncio.sleep(0)
        return server.pending()

    assert asyncio.run(scenario()) == 0
    assert server.pending() == 0


def test_stop_fails_streams_that_are_still_waiting() -> None:
    server = _server(usage_wait_timeout=30.0)
    app = create_lwd_gateway_app(server)

    async def scenario():
        server.attach_loop(asyncio.get_running_loop())
        response = await _handler(app, "/v1/chat/completions", "POST")(
            _request(*_shadow())
        )

        async def drain():
            return [chunk async for chunk in response.body_iterator]

        task = asyncio.create_task(drain())
        await asyncio.sleep(0.02)
        server.stop()
        with pytest.raises(RuntimeError, match="gateway control plane stopped"):
            await task

    asyncio.run(scenario())


# --------------------------------------------------------------------- #
# request validation                                                     #
# --------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "header_overrides,body_overrides",
    [
        ({wire.HEADER_PROTOCOL: "edge-cloud-prefix-v9"}, {}),
        ({wire.HEADER_PROMPT_TOKENS: "abc"}, {}),
        ({}, {"messages": [{"role": "user", "content": "hello"}]}),
        ({}, {"model": ""}),
    ],
)
def test_bad_requests_are_rejected_before_any_stream(
    header_overrides, body_overrides
) -> None:
    server = _server()
    app = create_lwd_gateway_app(server)
    headers, body = _shadow()
    headers.update(header_overrides)
    body.update(body_overrides)
    assert _post(app, headers, body).status_code == 400
    assert server.pending() == 0


def test_health_reports_pending_streams() -> None:
    server = _server()
    app = create_lwd_gateway_app(server)
    payload = asyncio.run(_handler(app, "/health", "GET")())
    assert payload == {"status": "ok", "pending_streams": 0}


# --------------------------------------------------------------------- #
# two streams, one request id                                            #
# --------------------------------------------------------------------- #
def test_duplicate_stream_is_superseded_not_starved() -> None:
    """A second stream for one request id takes over the single usage event.

    The older stream must end promptly *without* a usage frame instead of
    holding a gateway connection open until ``usage_wait_timeout``; the newer
    one gets the real numbers. This also pins the release rule: the superseded
    stream must not unregister the future that replaced it, or the survivor
    would time out and bill nothing.
    """
    server = _server(heartbeat_interval=0.01, usage_wait_timeout=30.0)
    app = create_lwd_gateway_app(server)
    post = _handler(app, "/v1/chat/completions", "POST")

    async def scenario():
        server.attach_loop(asyncio.get_running_loop())

        async def collect(response):
            return [_classify(c) async for c in response.body_iterator]

        first = asyncio.create_task(collect(await post(_request(*_shadow()))))
        await asyncio.sleep(0.02)
        assert server.pending() == 1
        second = asyncio.create_task(collect(await post(_request(*_shadow()))))
        await asyncio.sleep(0.02)
        assert server.pending() == 1  # one id, one live registration
        server.publish_usage(RID, 100, 20)
        return await first, await second

    stale_events, live_events = asyncio.run(scenario())

    stale_data = [p for k, p in stale_events if k == "data"]
    assert all("usage" not in frame for frame in stale_data)
    assert ("comment", "edge-cloud-usage-unavailable") in stale_events
    assert stale_events[-1] == ("done", None)

    live_usage = [p for k, p in live_events if k == "data" and "usage" in p]
    assert len(live_usage) == 1
    assert live_usage[0]["usage"]["prompt_tokens"] == 100
    assert live_usage[0]["usage"]["completion_tokens"] == 20
    assert live_events[-1] == ("done", None)
    assert server.pending() == 0
