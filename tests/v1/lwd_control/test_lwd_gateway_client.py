# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Edge billing client contract: fire-and-forget, drain-to-DONE, never block.

No socket is opened: the aiohttp session is replaced, so what is pinned is the
client's own discipline -- the shadow request shape, that a rejected or broken
stream degrades to a warning, and that two streams for the same request id both
stay referenced (a registry keyed by request id would drop the first one
mid-drain and lose its usage).
"""

from __future__ import annotations

import asyncio
import json
import threading
import time

import pytest

from vllm.config.lwd.gateway import LwdGatewayConfig
from vllm.v1.lwd_control.lwd_gateway import lwd_gateway_client as client_module
from vllm.v1.lwd_control.lwd_gateway import lwd_gateway_wire as wire
from vllm.v1.lwd_control.lwd_gateway.lwd_gateway_client import (
    LwdGatewayBillingClient,
    _iter_sse_data,
)

pytestmark = pytest.mark.cpu_test

RID = "chatcmpl-abc123"
MODEL = "qwen3.5"
CONTROL_URL = "https://gateway.example.com/v1/chat/completions"


class _AsyncLines:
    def __init__(self, text: str) -> None:
        self._lines = text.splitlines(keepends=True)
        self.read = 0

    def __aiter__(self):
        async def _emit():
            for line in self._lines:
                self.read += 1
                yield line.encode()

        return _emit()


class _FakeResponse:
    def __init__(self, sse_text: str = "", status: int = 200) -> None:
        self.status = status
        self.content = _AsyncLines(sse_text)
        self.close_calls = 0

    def close(self) -> None:
        self.close_calls += 1

    async def read(self, limit: int = -1) -> bytes:
        return b'{"error":"upstream said no"}'


class _FakeSession:
    def __init__(self, response: _FakeResponse) -> None:
        self.response = response
        self.calls: list[dict] = []

    async def post(self, url, json=None, headers=None, timeout=None):
        self.calls.append(
            {"url": url, "body": json, "headers": headers, "timeout": timeout}
        )
        return self.response

    async def close(self) -> None:
        pass


def _gateway(**overrides) -> LwdGatewayConfig:
    base = {"enabled": True, "control_url": CONTROL_URL, "consumer_id": "t1"}
    return LwdGatewayConfig(**(base | overrides))


def _client(gateway: LwdGatewayConfig | None = None) -> LwdGatewayBillingClient:
    return LwdGatewayBillingClient(
        gateway or _gateway(), MODEL, edge_id=0
    )


def _sse_stream(prompt: int = 1024, completion: int = 128) -> str:
    message = wire.control_message(RID, "cloud-0", 256)
    return (
        wire.probe_frame(RID, MODEL, message)
        + wire.empty_delta_frame(RID, MODEL)
        + wire.HEARTBEAT_COMMENT
        + wire.usage_frame(RID, MODEL, wire.usage_dict(prompt, completion))
        + wire.DONE_FRAME
    )


def _attach(client: LwdGatewayBillingClient, response: _FakeResponse):
    """Replace the lazily created aiohttp session with the fake one."""
    session = _FakeSession(response)

    async def _ensure():
        return session

    client._ensure_session = _ensure  # type: ignore[method-assign]
    return session


def _wait_until(predicate, what: str, deadline_s: float = 5.0) -> None:
    """Poll a real signal; never infer completion from an empty registry
    alone (the stream task may not have started yet)."""
    end = time.monotonic() + deadline_s
    while time.monotonic() < end:
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError(f"timed out waiting for {what}")


# --------------------------------------------------------------------- #
# stream shape                                                           #
# --------------------------------------------------------------------- #
def test_submit_sends_the_shadow_request_and_drains_to_done() -> None:
    client = _client()
    response = _FakeResponse(_sse_stream())
    session = _attach(client, response)
    try:
        client.submit(RID, 1024)
        _wait_until(lambda: response.close_calls == 1, "stream close")
        _wait_until(lambda: not client._streams, "registry drain")
    finally:
        client.close()

    assert len(session.calls) == 1
    call = session.calls[0]
    assert call["url"] == CONTROL_URL
    assert call["timeout"].total is None  # the stream outlives the inference
    headers, body = call["headers"], call["body"]
    assert headers[wire.HEADER_REQUEST_ID] == RID
    assert headers[wire.HEADER_PROMPT_TOKENS] == "1024"
    assert headers["X-Mse-Consumer"] == "t1"
    assert "Authorization" not in headers
    assert set(body) == {"model", "messages", "stream", "stream_options"}
    # every frame was read: the cloud only finishes the stream once drained
    assert response.content.read >= 5
    assert response.close_calls == 1


def test_api_key_is_sent_as_bearer(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VLLM_LWD_GATEWAY_API_KEY", "sk-gw-token")
    client = _client()
    response = _FakeResponse(_sse_stream())
    session = _attach(client, response)
    try:
        client.submit(RID, 8)
        _wait_until(lambda: response.close_calls == 1, "stream close")
    finally:
        client.close()
    assert session.calls[0]["headers"]["Authorization"] == "Bearer sk-gw-token"


def test_rejected_stream_is_a_warning_not_a_hang() -> None:
    client = _client()
    response = _FakeResponse("", status=503)
    session = _attach(client, response)
    try:
        client.submit(RID, 8)
        _wait_until(lambda: response.close_calls == 1, "stream close")
    finally:
        client.close()
    assert len(session.calls) == 1
    assert response.close_calls == 1


def test_two_streams_for_one_request_id_both_stay_referenced() -> None:
    """The registry must not be keyed by request id.

    A re-submitted id (client retry, or an id reused after abort) would
    otherwise overwrite the previous task's only strong reference; a task
    parked on a socket read has no pending callback, so it can be collected
    mid-drain and that request's usage is lost.
    """
    gate = threading.Event()

    class _GatedLines(_AsyncLines):
        def __aiter__(self):
            async def _emit():
                for line in self._lines:
                    while not gate.is_set():
                        await asyncio.sleep(0.01)
                    yield line.encode()

            return _emit()

    client = _client()
    first = _FakeResponse(_sse_stream())
    first.content = _GatedLines(_sse_stream())
    _attach(client, first)
    second = _FakeResponse(_sse_stream())
    second.content = _GatedLines(_sse_stream())
    sessions = [first, second]
    responses = iter(sessions)

    async def _ensure():
        return _FakeSession(next(responses))

    client._ensure_session = _ensure  # type: ignore[method-assign]
    try:
        client.submit(RID, 8)
        client.submit(RID, 8)
        _wait_until(lambda: len(client._streams) == 2, "both streams registered")
        gate.set()
        _wait_until(lambda: not client._streams, "registry drain")
    finally:
        gate.set()
        client.close()


def test_submit_after_close_does_not_raise() -> None:
    client = _client()
    _attach(client, _FakeResponse(_sse_stream()))
    client.close()
    client._loop.stop()
    time.sleep(0.2)
    client.submit(RID, 8)  # must degrade to a warning


# --------------------------------------------------------------------- #
# SSE reader                                                             #
# --------------------------------------------------------------------- #
def _read(text: str) -> list[str]:
    async def _collect():
        return [payload async for payload in _iter_sse_data(_AsyncLines(text))]

    return asyncio.run(_collect())


def test_sse_reader_keeps_multiline_data_and_skips_comments() -> None:
    text = (
        ": heartbeat\n"
        "\n"
        "data: {\"a\":1}\n"
        "\n"
        "data: first\n"
        "data: second\n"
        "\n"
        "data: [DONE]\n"
        "\n"
    )
    assert _read(text) == ['{"a":1}', "first\nsecond", "[DONE]"]


def test_sse_reader_rejects_oversized_frames() -> None:
    huge = "data: " + "x" * 20000 + "\n\n"
    with pytest.raises(ValueError, match="size limit"):
        _read(huge)


def test_control_message_payload_is_valid_json() -> None:
    """What the client parses out of the first frame must round-trip."""
    stream = _sse_stream()
    payloads = _read(stream)
    first = json.loads(payloads[0])
    message = json.loads(first["choices"][0]["delta"]["content"])
    assert message == wire.control_message(RID, "cloud-0", 256)


# --------------------------------------------------------------------- #
# control message split across frames                                    #
# --------------------------------------------------------------------- #
class _LogRecorder:
    def __init__(self) -> None:
        self.warnings: list[str] = []
        self.infos: list[str] = []

    @staticmethod
    def _render(msg, args):
        return msg % args if args else msg

    def warning(self, msg, *args, **kwargs):
        self.warnings.append(self._render(msg, args))

    def info(self, msg, *args, **kwargs):
        self.infos.append(self._render(msg, args))

    def error(self, msg, *args, **kwargs):
        self.warnings.append(self._render(msg, args))

    def debug(self, *args, **kwargs):
        pass


def _content_frame(text: str) -> str:
    chunk = {
        "id": RID,
        "object": "chat.completion.chunk",
        "created": 0,
        "model": MODEL,
        "choices": [
            {
                "index": 0,
                "delta": {"role": "assistant", "content": text},
                "finish_reason": None,
            }
        ],
    }
    return "data: " + json.dumps(chunk, separators=(",", ":")) + "\n\n"


def _split_control_message(message_rid: str) -> str:
    message = json.dumps(
        wire.control_message(message_rid, "cloud-0", 256), separators=(",", ":")
    )
    half = len(message) // 2
    return _content_frame(message[:half]) + _content_frame(message[half:])


def _run_stream(client, response) -> None:
    _attach(client, response)
    try:
        client.submit(RID, 64)
        _wait_until(lambda: response.close_calls == 1, "stream close")
    finally:
        client.close()


def test_fragmented_control_message_is_reassembled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A gateway may split delta.content across frames.

    Without accumulation the control message never parses, so the instance id
    stays unknown in the usage log -- billing still works (usage has its own
    frame), which is exactly why this would go unnoticed in production.
    """
    recorder = _LogRecorder()
    monkeypatch.setattr(client_module, "lwd_logger", recorder)
    stream = (
        _split_control_message(RID)
        + wire.usage_frame(RID, MODEL, wire.usage_dict(64, 8))
        + wire.DONE_FRAME
    )
    _run_stream(_client(), _FakeResponse(stream))
    assert any("cloud=cloud-0" in line for line in recorder.infos), recorder.infos
    assert not recorder.warnings


def test_abort_closes_only_that_ids_streams() -> None:
    """abort(rid) cancels exactly the aborted request's billing streams.

    The aborted stream closes at once (the cloud releases its waiter via the
    heartbeat loop's finally, so both sides are freed instead of hanging for
    usage_wait_timeout), and other ids keep draining afterwards.
    """
    gate = threading.Event()

    class _GatedLines(_AsyncLines):
        def __aiter__(self):
            async def _emit():
                while not gate.is_set():
                    await asyncio.sleep(0.01)
                for line in self._lines:
                    yield line.encode()

            return _emit()

    class _RoutedSession:
        def __init__(self, responses):
            self._responses = responses
            self.calls: list[dict] = []

        async def post(self, url, json=None, headers=None, timeout=None):
            self.calls.append({"url": url, "headers": headers})
            return self._responses[headers[wire.HEADER_REQUEST_ID]]

    other_rid = "chatcmpl-keep!"
    aborted_resp = _FakeResponse("")
    aborted_resp.content = _GatedLines(_sse_stream())
    other_resp = _FakeResponse(_sse_stream())
    session = _RoutedSession({RID: aborted_resp, other_rid: other_resp})

    client = _client()

    async def _ensure():
        return session

    client._ensure_session = _ensure  # type: ignore[method-assign]
    try:
        client.submit(RID, 8)
        _wait_until(lambda: len(client._streams) == 1, "first stream live")
        client.abort(RID)
        _wait_until(lambda: aborted_resp.close_calls == 1, "aborted stream closed")
        _wait_until(lambda: not client._streams, "registry empty after abort")
        # the client is unaffected for other ids afterwards
        assert other_rid not in client._rid_tasks
        client.submit(other_rid, 64)
        _wait_until(lambda: other_resp.close_calls == 1, "other stream drained")
    finally:
        gate.set()
        client.close()


def test_unrelated_json_content_frame_does_not_fake_a_misroute(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A foreign but well-formed JSON content frame must not warn.

    Our own control message is flat, so no proper prefix of it parses as JSON;
    a gateway plugin notice does parse, and without the request_id guard it
    would trip the "another request" warning and poison the instance id.
    """
    recorder = _LogRecorder()
    monkeypatch.setattr(client_module, "lwd_logger", recorder)
    stream = (
        _content_frame('{"role":"assistant","notice":"injected"}')
        + _split_control_message(RID)
        + wire.usage_frame(RID, MODEL, wire.usage_dict(64, 8))
        + wire.DONE_FRAME
    )
    _run_stream(_client(), _FakeResponse(stream))
    assert not any("another request" in line for line in recorder.warnings), (
        recorder.warnings
    )
    assert any("cloud=cloud-0" in line for line in recorder.infos), recorder.infos


def test_fragmented_message_for_another_request_still_warns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The misroute signal must survive fragmentation too."""
    recorder = _LogRecorder()
    monkeypatch.setattr(client_module, "lwd_logger", recorder)
    stream = (
        _split_control_message("chatcmpl-someone-else")
        + wire.usage_frame(RID, MODEL, wire.usage_dict(64, 8))
        + wire.DONE_FRAME
    )
    _run_stream(_client(), _FakeResponse(stream))
    assert any("another request" in line for line in recorder.warnings), (
        recorder.warnings
    )
