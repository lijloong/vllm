# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Cloud HTTP app implementing the explicit gateway SSE contract."""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncIterator, Iterator, Mapping
from dataclasses import replace as dataclass_replace
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from vllm.logger import init_logger
from vllm.v1.lwd_control.control_cloud_scheduler.lwd_cloud_control import (
    LwdCloudControlBridge,
    _LwdUsageRegistration,
    _negotiate_lwd_protocol_headers,
    _required_decimal_header,
    _sse_data,
)
from vllm.v1.lwd_control.lwd_comm.lwd_id_adapter import wrap_req_id
from vllm.v1.lwd_control.lwd_comm.lwd_mm_identity import mm_abi_header_value
from vllm.v1.lwd_control.lwd_comm.lwd_prefix import (
    GATEWAY_USAGE_EXECUTION_TIMEOUT_S,
    HEADER_EDGE_ID,
    HEADER_MM_ABI,
    HEADER_PROTOCOL,
    HEADER_USAGE_EXECUTION_TIMEOUT_MS,
    PROTOCOL_VERSION,
    PROTOCOL_VERSION_MM,
    LwdPrefixManifest,
)

logger = init_logger(__name__)

_USAGE_HEARTBEAT_INTERVAL = 10.0


def _gateway_chunk(
    request_id: str,
    model: str,
    choices: list[Any],
    usage: dict[str, Any] | None = None,
) -> dict[str, Any]:
    chunk: dict[str, Any] = {
        "id": request_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": choices,
    }
    if usage is not None:
        chunk["usage"] = usage
    return chunk


def _startup_frames(
    display_id: str, model: str, control_message: Mapping[str, Any]
) -> Iterator[str]:
    yield _sse_data(
        _gateway_chunk(
            display_id,
            model,
            [
                {
                    "index": 0,
                    "delta": {
                        "role": "assistant",
                        "content": json.dumps(control_message, separators=(",", ":")),
                    },
                    "finish_reason": None,
                }
            ],
        )
    )
    yield _sse_data(
        _gateway_chunk(
            display_id,
            model,
            [{"index": 0, "delta": {}, "finish_reason": None}],
        )
    )


def _parse_usage_execution_timeout(headers: Mapping[str, str]) -> float:
    normalized = {k.lower(): v for k, v in headers.items()}
    raw = normalized.get(HEADER_USAGE_EXECUTION_TIMEOUT_MS.lower())
    if raw is None:
        return GATEWAY_USAGE_EXECUTION_TIMEOUT_S
    timeout_ms = _required_decimal_header(normalized, HEADER_USAGE_EXECUTION_TIMEOUT_MS)
    if timeout_ms == 0:
        raise ValueError(
            f"{HEADER_USAGE_EXECUTION_TIMEOUT_MS} must be a positive integer"
        )
    return timeout_ms / 1000


async def _usage_frames(
    bridge: LwdCloudControlBridge,
    request_id: str,
    display_id: str,
    model: str,
    registration: _LwdUsageRegistration,
    execution_timeout: float,
) -> AsyncIterator[str]:
    usage_task = asyncio.create_task(
        bridge.wait_gateway_usage(
            request_id,
            execution_timeout=execution_timeout,
            registration=registration,
        )
    )
    try:
        while not usage_task.done():
            completed, _ = await asyncio.wait(
                {usage_task}, timeout=_USAGE_HEARTBEAT_INTERVAL
            )
            if not completed:
                yield ": edge-cloud-usage-pending\n\n"
        usage = usage_task.result()
    except Exception as exc:
        logger.warning(
            "[LWD][ctrl] gateway usage wait failed req=%s: %s",
            request_id,
            exc,
        )
        usage = None
    finally:
        if not usage_task.done():
            usage_task.cancel()
        await asyncio.gather(usage_task, return_exceptions=True)
    if usage is None:
        logger.warning(
            "[LWD][ctrl] gateway usage unavailable req=%s",
            request_id,
        )
        yield ": edge-cloud-usage-unavailable\n\n"
        yield "data: [DONE]\n\n"
        return
    yield _sse_data(
        _gateway_chunk(
            display_id,
            model,
            [],
            usage.to_openai_dict(),
        )
    )
    yield "data: [DONE]\n\n"


def create_lwd_gateway_control_app(
    bridge: LwdCloudControlBridge,
    cloud_id: int,
    dp_idx: int,
    processor_fingerprint: bytes | None = None,
    *,
    enforce_mm_abi_match: bool = True,
    probe_timeout: float = 20.0,
) -> Any:
    """Build one fixed-DP app with the single Chat Completions route."""
    if cloud_id < 0 or dp_idx < 0:
        raise ValueError("cloud_id and dp_idx must be non-negative")
    app = FastAPI()
    expected_mm_abi = (
        mm_abi_header_value(processor_fingerprint)
        if processor_fingerprint is not None
        else None
    )

    async def parse_request(
        request: Request,
    ) -> tuple[LwdPrefixManifest, Mapping[str, Any], str, str | None, float]:
        headers = dict(request.headers)
        body = await request.json()
        if not isinstance(body, Mapping):
            raise ValueError("request body must be a JSON object")
        required_fields = {"model", "messages", "stream", "stream_options"}
        if set(body) != required_fields:
            raise ValueError(
                "request body must contain exactly model, messages, stream "
                "and stream_options"
            )
        if body.get("stream") is not True:
            raise ValueError("edge-cloud requests must set stream=true")
        stream_options = body.get("stream_options")
        if (
            not isinstance(stream_options, Mapping)
            or stream_options.get("include_usage") is not True
        ):
            raise ValueError("stream_options.include_usage must be true")
        if not isinstance(body.get("model"), str) or not body["model"]:
            raise ValueError("model must be a non-empty string")
        mm_abi_header = _negotiate_lwd_protocol_headers(
            headers,
            expected_mm_abi,
            enforce_mm_abi_match=enforce_mm_abi_match,
        )
        parse_headers: Mapping[str, str] = headers
        if mm_abi_header is not None:
            parse_headers = {
                k: v for k, v in headers.items() if k.lower() != HEADER_PROTOCOL.lower()
            }
            parse_headers[HEADER_PROTOCOL] = PROTOCOL_VERSION
        manifest = LwdPrefixManifest.from_openai_request(parse_headers, body)
        headers_lc = {k.lower(): v for k, v in headers.items()}
        edge_id = _required_decimal_header(headers_lc, HEADER_EDGE_ID)
        display_request_id = manifest.request_id
        manifest = dataclass_replace(
            manifest, request_id=wrap_req_id(edge_id, display_request_id)
        )
        execution_timeout = _parse_usage_execution_timeout(headers_lc)
        return manifest, body, display_request_id, mm_abi_header, execution_timeout

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request) -> Any:
        try:
            (
                manifest,
                body,
                display_id,
                mm_abi,
                execution_timeout,
            ) = await parse_request(request)
            registration = bridge.activate(manifest.request_id, dp_idx)
        except ValueError as exc:
            return JSONResponse({"error": f"bad request: {exc}"}, status_code=400)
        except Exception as exc:  # noqa: BLE001
            return JSONResponse({"error": str(exc)}, status_code=503)

        reply_key = (
            f"{manifest.request_id}#probe:{registration.prefix_nonce}:"
            f"{cloud_id}:{dp_idx}"
        )
        try:
            result = await bridge.probe(
                dp_idx, manifest, reply_key=reply_key, timeout=probe_timeout
            )
        except asyncio.CancelledError:
            bridge.release(manifest.request_id, registration)
            raise
        except TimeoutError as exc:
            bridge.release(manifest.request_id, registration)
            logger.warning(
                "[LWD][ctrl] gateway probe timeout req=%s dp=%d: %s",
                manifest.request_id,
                dp_idx,
                exc,
            )
            return JSONResponse({"error": str(exc)}, status_code=503)
        except Exception as exc:  # noqa: BLE001
            bridge.release(manifest.request_id, registration)
            logger.exception("[LWD][ctrl] gateway probe raised")
            return JSONResponse({"error": str(exc)}, status_code=503)

        result = dataclass_replace(
            result,
            request_id=display_id,
            cloud_id=cloud_id,
            engine_index=dp_idx,
        )
        protocol = PROTOCOL_VERSION_MM if mm_abi is not None else PROTOCOL_VERSION
        control_message = result.to_control_message(protocol, mm_abi)
        control_message["prefix_nonce"] = registration.prefix_nonce
        model = body["model"]

        async def events() -> AsyncIterator[str]:
            try:
                for frame in _startup_frames(display_id, model, control_message):
                    yield frame
                async for frame in _usage_frames(
                    bridge,
                    manifest.request_id,
                    display_id,
                    model,
                    registration,
                    execution_timeout,
                ):
                    yield frame
            finally:
                bridge.release(manifest.request_id, registration)

        headers_out = result.to_headers()
        headers_out[HEADER_PROTOCOL] = protocol
        headers_out["Cache-Control"] = "no-cache"
        headers_out["X-Accel-Buffering"] = "no"
        if mm_abi is not None:
            headers_out[HEADER_MM_ABI] = mm_abi
        return StreamingResponse(
            events(), media_type="text/event-stream", headers=headers_out
        )

    return app
