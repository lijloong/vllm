# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
import json
import logging

import pytest

from vllm.v1.lwd_control.control_edge_scheduler.lwd_edge_client import (
    LwdEdgePrefixClient,
    _iter_sse_data,
)
from vllm.v1.lwd_control.control_edge_scheduler.lwd_gateway_client import (
    LwdGatewayClient,
)
from vllm.v1.lwd_control.lwd_comm.lwd_prefix import (
    HEADER_EDGE_ID,
    HEADER_PROBE_NONCE,
    HEADER_USAGE_EXECUTION_TIMEOUT_MS,
)


class _Content:
    def __init__(self, payload: bytes) -> None:
        self._payload = payload
        self.limit: int | None = None

    async def read(self, limit: int) -> bytes:
        self.limit = limit
        return self._payload


class _Response:
    def __init__(self, payload: bytes, message: dict | None = None) -> None:
        self.content = _Content(payload)
        self.status = 200
        self._message = message
        self.closed = False

    async def json(self) -> dict:
        assert self._message is not None
        return self._message

    def close(self) -> None:
        self.closed = True

    async def read(self, *args: object) -> bytes:
        # aiohttp's ClientResponse.read() accepts no size argument; mirror
        # that so a regression to response.read(limit) fails here.
        raise TypeError(
            "ClientResponse.read() takes 1 positional argument but 2 were given"
        )


def test_error_body_reads_the_content_stream() -> None:
    response = _Response(b'{"error":"boom"}\n')

    body = asyncio.run(LwdEdgePrefixClient._read_error_body(response))

    assert body == '{"error":"boom"}'
    assert response.content.limit == 512


def _probe_message(*, cloud_id: int = 1, dp_idx: int = 1) -> dict:
    return {
        "type": "edge_cloud_probe",
        "protocol": "edge-cloud-prefix-v1",
        "request_id": "req",
        "instance_id": "cloud-1",
        "block_size": 4,
        "hit_blocks": 1,
        "hit_tokens": 4,
        "alignment_tokens": 0,
        "cloud_id": cloud_id,
        "engine_index": dp_idx,
    }


class _Session:
    def __init__(self, response: _Response) -> None:
        self.response = response
        self.calls: list[tuple[str, dict]] = []

    async def post(self, url: str, **kwargs: dict) -> _Response:
        self.calls.append((url, kwargs))
        return self.response


def test_direct_dp_probe_uses_logical_edge_and_closes_short_response() -> None:
    response = _Response(b"", _probe_message())
    session = _Session(response)
    client = LwdEdgePrefixClient("http://cloud1:8101", 4, b"tenant-key-16bytes")

    result = asyncio.run(
        client.probe(
            session,
            "req",
            list(range(8)),
            edge_id=3,
            cloud_id=1,
            dp_idx=1,
            probe_nonce=7,
        )
    )

    assert result.hit_tokens == 4
    assert result.engine_index == 1
    assert response.closed
    url, kwargs = session.calls[0]
    assert url == "http://cloud1:8101/v1/lwd/probe"
    assert kwargs["headers"][HEADER_EDGE_ID] == "3"
    assert kwargs["headers"][HEADER_PROBE_NONCE] == "7"


def test_direct_dp_probe_rejects_misrouted_response() -> None:
    response = _Response(b"", _probe_message(dp_idx=0))
    session = _Session(response)
    client = LwdEdgePrefixClient("http://cloud1:8101", 4, b"tenant-key-16bytes")

    with pytest.raises(RuntimeError, match="different DP"):
        asyncio.run(
            client.probe(
                session,
                "req",
                list(range(8)),
                edge_id=0,
                cloud_id=1,
                dp_idx=1,
                probe_nonce=1,
            )
        )
    assert response.closed


class _StreamingContent:
    def __init__(self, events: list[bytes], *, block: bool = False) -> None:
        self.events = events
        self.block = block

    def __aiter__(self):
        async def generate():
            for event in self.events:
                yield event
            if self.block:
                await asyncio.Future()

        return generate()


def test_cancelled_usage_drain_closes_response() -> None:
    async def run() -> None:
        response = _Response(b"")
        response.content = _StreamingContent([], block=True)
        client = LwdEdgePrefixClient("http://cloud1:8101", 4, b"tenant-key-16bytes")
        task = client._start_drain("req", response, _iter_sse_data(response.content))

        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        await asyncio.sleep(0)

        assert response.closed
        assert not client._drain_tasks

    asyncio.run(run())


def test_completed_usage_drain_closes_response() -> None:
    async def run() -> None:
        response = _Response(b"")
        response.content = _StreamingContent([b"data: [DONE]\n\n"])
        client = LwdEdgePrefixClient("http://cloud1:8101", 4, b"tenant-key-16bytes")
        task = client._start_drain("req", response, _iter_sse_data(response.content))

        await task

        assert response.closed
        assert not client._drain_tasks

    asyncio.run(run())


def _content_chunk(content: str | None) -> bytes:
    delta = {} if content is None else {"role": "assistant", "content": content}
    chunk = {
        "id": "req",
        "object": "chat.completion.chunk",
        "created": 1,
        "model": "test",
        "choices": [{"index": 0, "delta": delta, "finish_reason": None}],
    }
    return f"data: {json.dumps(chunk)}\n\n".encode()


def test_gateway_execution_timeout_adds_token_allowance_to_constant_base() -> None:
    client = LwdGatewayClient(
        "https://gateway.example.com/v1/chat/completions",
        4,
        b"tenant-key-16bytes",
    )

    assert client._usage_execution_timeout_ms(100) == 700000
    assert client._usage_execution_timeout_ms(1200) == 1800000


def test_gateway_posts_complete_url_and_recovers_fragmented_startup_content() -> None:
    async def run() -> None:
        control = {
            **_probe_message(cloud_id=1, dp_idx=1),
            "prefix_nonce": 9,
        }
        encoded = json.dumps(control, separators=(",", ":"))
        response = _Response(b"")
        response.content = _StreamingContent(
            [
                line
                for event in [
                    _content_chunk("gateway-injected-text"),
                    _content_chunk(encoded[:20]),
                    _content_chunk("not-json"),
                    _content_chunk(encoded),
                    _content_chunk(None),
                    b"data: [DONE]\n\n",
                ]
                for line in event.splitlines(keepends=True)
            ]
        )
        session = _Session(response)
        url = "https://gateway.example.com/tenant-a/v1/chat/completions?route=blue"
        client = LwdGatewayClient(url, 4, b"tenant-key-16bytes")

        activation = await client.activate_gateway(
            session,
            "req",
            list(range(8)),
            edge_id=3,
            model="test",
            max_tokens=1200,
        )
        await activation.drain_task

        assert activation.result.cloud_id == 1
        assert activation.result.engine_index == 1
        assert activation.prefix_nonce == 9
        request_url, kwargs = session.calls[0]
        assert request_url == url
        assert set(kwargs["json"]) == {
            "model",
            "messages",
            "stream",
            "stream_options",
        }
        assert kwargs["json"]["stream"] is True
        assert kwargs["json"]["stream_options"] == {"include_usage": True}
        assert HEADER_PROBE_NONCE not in kwargs["headers"]
        assert kwargs["headers"][HEADER_USAGE_EXECUTION_TIMEOUT_MS] == "1800000"
        assert response.closed

    asyncio.run(run())


def test_gateway_activation_rejects_non_200_and_closes_response() -> None:
    async def run() -> None:
        response = _Response(b'{"error":"busy"}')
        response.status = 503
        session = _Session(response)
        client = LwdGatewayClient(
            "https://gateway.example.com/v1/chat/completions",
            4,
            b"tenant-key-16bytes",
        )

        with pytest.raises(RuntimeError, match="gateway activation HTTP 503"):
            await client.activate_gateway(
                session,
                "req",
                list(range(8)),
                edge_id=3,
                model="test",
            )

        assert response.closed

    asyncio.run(run())


def test_gateway_activation_startup_timeout_closes_response() -> None:
    async def run() -> None:
        response = _Response(b"")
        response.content = _StreamingContent([], block=True)
        session = _Session(response)
        client = LwdGatewayClient(
            "https://gateway.example.com/v1/chat/completions",
            4,
            b"tenant-key-16bytes",
            probe_timeout=0.01,
        )

        with pytest.raises(asyncio.TimeoutError):
            await client.activate_gateway(
                session,
                "req",
                list(range(8)),
                edge_id=3,
                model="test",
            )

        assert response.closed

    asyncio.run(run())


def test_usage_unavailable_comment_accepts_sse_whitespace(
    caplog: pytest.LogCaptureFixture,
) -> None:
    async def run() -> None:
        response = _Response(b"")
        response.content = _StreamingContent(
            [
                b":edge-cloud-usage-unavailable \n",
                b"\n",
                b"data: [DONE]\n",
                b"\n",
            ]
        )
        client = LwdEdgePrefixClient("http://cloud1:8101", 4, b"tenant-key-16bytes")
        task = client._start_drain("req", response, _iter_sse_data(response.content))

        await task

        assert response.closed

    with caplog.at_level(logging.INFO):
        asyncio.run(run())
    assert "usage unavailable in stream" in caplog.text
