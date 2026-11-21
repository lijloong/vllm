# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
import json
from typing import Any

import pytest

from vllm.v1.lwd_control.control_edge_scheduler.lwd_edge_client import (
    LwdEdgePrefixClient,
    _iter_sse_data,
)
from vllm.v1.lwd_control.lwd_comm.lwd_prefix import HEADER_EDGE_ID


class _Content:
    def __init__(self, payload: bytes) -> None:
        self._payload = payload
        self.limit: int | None = None

    async def read(self, limit: int) -> bytes:
        self.limit = limit
        return self._payload


class _Response:
    def __init__(self, payload: bytes = b"") -> None:
        self.content: Any = _Content(payload)
        self.headers: dict[str, str] = {}
        self.status = 200
        self.closed = False

    def close(self) -> None:
        self.closed = True


class _Session:
    def __init__(self, response: _Response) -> None:
        self.response = response
        self.calls: list[tuple[str, dict]] = []

    async def post(self, url: str, **kwargs: dict) -> _Response:
        self.calls.append((url, kwargs))
        return self.response


class _StreamingContent:
    def __init__(self, events: list[bytes], *, block: bool = False) -> None:
        self.events = events
        self.block = block

    def __aiter__(self):
        async def generate():
            for event in self.events:
                for line in event.splitlines(keepends=True):
                    yield line
            if self.block:
                await asyncio.Future()

        return generate()


def _probe_message(request_id: str = "req") -> dict:
    return {
        "type": "edge_cloud_probe",
        "protocol": "edge-cloud-prefix-v1",
        "request_id": request_id,
        "instance_id": "cloud-1",
        "block_size": 4,
        "hit_blocks": 1,
        "hit_tokens": 4,
        "alignment_tokens": 0,
        "cloud_id": 1,
        "engine_index": 1,
        "prefix_nonce": 7,
    }


def _chunk_frame(
    *, content: str | None = None, empty: bool = False, usage: dict | None = None
) -> bytes:
    if usage is not None:
        choices = []
    elif empty:
        choices = [{"index": 0, "delta": {}, "finish_reason": None}]
    else:
        choices = [
            {
                "index": 0,
                "delta": {"role": "assistant", "content": content or ""},
                "finish_reason": None,
            }
        ]
    chunk = {
        "id": "req",
        "object": "chat.completion.chunk",
        "created": 1700000000,
        "model": "test",
        "choices": choices,
    }
    if usage is not None:
        chunk["usage"] = usage
    return f"data: {json.dumps(chunk, separators=(',', ':'))}\n\n".encode()


def _client() -> LwdEdgePrefixClient:
    return LwdEdgePrefixClient("http://gateway:8101", 4, b"tenant-key-16bytes")


def _activate_and_drain(response: _Response) -> Any:
    async def run() -> Any:
        activation = await _client().activate(
            _Session(response),
            "req",
            list(range(8)),
            edge_id=0,
            model="test",
        )
        await activation.drain_task
        return activation.result

    return asyncio.run(run())


def test_error_body_reads_the_content_stream() -> None:
    response = _Response(b'{"error":"boom"}')

    body = asyncio.run(LwdEdgePrefixClient._read_error_body(response))

    assert body == '{"error":"boom"}'
    assert response.content.limit == 512


def test_activate_uses_one_contract_request_and_reassembles_startup() -> None:
    message = json.dumps(_probe_message(), separators=(",", ":"))
    midpoint = len(message) // 2
    response = _Response()
    response.content = _StreamingContent(
        [
            _chunk_frame(content=message[:midpoint]),
            _chunk_frame(content=message[midpoint:]),
            _chunk_frame(empty=True),
            _chunk_frame(
                usage={
                    "prompt_tokens": 8,
                    "completion_tokens": 2,
                    "total_tokens": 10,
                    "prompt_tokens_details": {"cached_tokens": 4},
                }
            ),
            b"data: [DONE]\n\n",
        ]
    )
    session = _Session(response)
    client = _client()

    async def run() -> None:
        activation = await client.activate(
            session,
            "req",
            list(range(8)),
            edge_id=3,
            model="test",
        )
        assert not response.closed
        await activation.drain_task
        assert activation.result.hit_tokens == 4
        assert activation.result.engine_index == 1
        assert activation.prefix_nonce == 7

    asyncio.run(run())

    assert response.closed
    url, kwargs = session.calls[0]
    assert url == "http://gateway:8101/v1/chat/completions"
    assert kwargs["headers"][HEADER_EDGE_ID] == "3"
    assert kwargs["headers"]["Accept"] == "text/event-stream"
    assert "X-Lwd-Probe-Nonce" not in kwargs["headers"]
    assert kwargs["json"] == {
        "model": "test",
        "messages": kwargs["json"]["messages"],
        "stream": True,
        "stream_options": {"include_usage": True},
    }


def test_activate_does_not_require_response_headers() -> None:
    response = _Response()
    response.headers = {}
    response.content = _StreamingContent(
        [
            _chunk_frame(content=json.dumps(_probe_message())),
            b"data: [DONE]\n\n",
        ]
    )
    client = _client()

    async def run() -> None:
        activation = await client.activate(
            _Session(response),
            "req",
            list(range(8)),
            edge_id=0,
            model="test",
        )
        await activation.drain_task
        assert activation.result.instance_id == "cloud-1"
        assert activation.result.engine_index == 1

    asyncio.run(run())
    assert response.headers == {}
    assert response.closed


def test_activate_ignores_non_json_plugin_content() -> None:
    response = _Response()
    response.content = _StreamingContent(
        [
            _chunk_frame(content="gateway plugin notice"),
            _chunk_frame(content=json.dumps(_probe_message())),
            b"data: [DONE]\n\n",
        ]
    )

    result = _activate_and_drain(response)

    assert result.request_id == "req"
    assert response.closed


def test_activate_resynchronizes_after_plugin_between_fragments() -> None:
    message = json.dumps(_probe_message(), separators=(",", ":"))
    response = _Response()
    response.content = _StreamingContent(
        [
            _chunk_frame(content=message[:40]),
            _chunk_frame(content="gateway plugin notice"),
            _chunk_frame(content=message),
            b"data: [DONE]\n\n",
        ]
    )

    result = _activate_and_drain(response)

    assert result.request_id == "req"
    assert response.closed


def test_activate_preserves_nested_json_fragmentation() -> None:
    message = _probe_message() | {"metadata": {"node": 1}}
    serialized = json.dumps(message, separators=(",", ":"))
    split = serialized.index('{"node"')
    response = _Response()
    response.content = _StreamingContent(
        [
            _chunk_frame(content=serialized[:split]),
            _chunk_frame(content=serialized[split:]),
            b"data: [DONE]\n\n",
        ]
    )

    result = _activate_and_drain(response)

    assert result.request_id == "req"
    assert response.closed


def test_sse_reader_preserves_usage_unavailable_comment() -> None:
    async def run() -> list[str]:
        content = _StreamingContent(
            [b": edge-cloud-usage-unavailable\n\n", b"data: [DONE]\n\n"]
        )
        return [payload async for payload in _iter_sse_data(content)]

    assert asyncio.run(run()) == [": edge-cloud-usage-unavailable", "[DONE]"]


def test_activate_does_not_admit_at_http_200_without_startup() -> None:
    response = _Response()
    response.content = _StreamingContent([], block=True)
    client = _client()

    async def run() -> None:
        task = asyncio.create_task(
            client.activate(
                _Session(response),
                "req",
                list(range(8)),
                edge_id=0,
                model="test",
            )
        )
        await asyncio.sleep(0.01)
        assert not task.done()
        assert not response.closed
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        assert response.closed

    asyncio.run(run())


def test_activate_rejects_wrong_request_id() -> None:
    response = _Response()
    response.content = _StreamingContent(
        [_chunk_frame(content=json.dumps(_probe_message("other")))]
    )
    client = _client()

    async def run() -> None:
        with pytest.raises(RuntimeError, match="different request ID"):
            await client.activate(
                _Session(response),
                "req",
                list(range(8)),
                edge_id=0,
                model="test",
            )

    asyncio.run(run())
    assert response.closed


def test_activate_rejects_missing_prefix_nonce() -> None:
    message = _probe_message()
    del message["prefix_nonce"]
    response = _Response()
    response.content = _StreamingContent([_chunk_frame(content=json.dumps(message))])
    client = _client()

    async def run() -> None:
        with pytest.raises(RuntimeError, match="invalid prefix nonce"):
            await client.activate(
                _Session(response),
                "req",
                list(range(8)),
                edge_id=0,
                model="test",
            )

    asyncio.run(run())
    assert response.closed


def test_cancelled_usage_drain_closes_response() -> None:
    async def run() -> None:
        response = _Response()
        response.content = _StreamingContent([], block=True)
        client = _client()
        task = client._start_drain("req", response, _iter_sse_data(response.content))

        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        await asyncio.sleep(0)

        assert response.closed
        assert not client._drain_tasks

    asyncio.run(run())


def test_completed_usage_drain_closes_response() -> None:
    async def run() -> None:
        response = _Response()
        response.content = _StreamingContent([b"data: [DONE]\n\n"])
        client = _client()
        task = client._start_drain("req", response, _iter_sse_data(response.content))

        await task

        assert response.closed
        assert not client._drain_tasks

    asyncio.run(run())
