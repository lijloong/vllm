# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Gateway-facing wire shape for the edge->cloud billing stream.

Pure stdlib, shared by both roles: the edge builds the shadow request, the
cloud parses it and renders the SSE frames. Frame order follows the
edge-gateway-cloud interaction spec: a control message inside the first
``delta.content``, an empty delta frame, ``:``-comment heartbeats while the
inference runs, a standard ``usage`` chunk with ``choices: []``, ``[DONE]``.

This deployment has no prefix cache and no cloud reuse, so the manifest
messages carry a *request identity* digest only -- one message instead of a
per-block hash chain, no tenant key, and the cloud does not cross-check digest
counts against the prompt length. ``hit_blocks``/``hit_tokens`` stay 0 and
``cached_tokens`` stays 0; they are part of the wire contract, not of the
feature set here.
"""

from __future__ import annotations

import base64
import hashlib
import json
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

PROTOCOL_VERSION = "edge-cloud-prefix-v1"
DIGEST_PREFIX = "ecb1:"
DEFAULT_BLOCK_SIZE = 256

HEADER_PROTOCOL = "X-Edge-Cloud-Protocol"
HEADER_REQUEST_ID = "X-Edge-Cloud-Request-ID"
HEADER_EDGE_ID = "X-Edge-Cloud-Edge-Id"
HEADER_INSTANCE_ID = "X-Edge-Cloud-Instance-ID"
HEADER_BLOCK_SIZE = "X-Edge-Cloud-Block-Size"
HEADER_PROMPT_TOKENS = "X-Edge-Cloud-Prompt-Tokens"
HEADER_HIT_BLOCKS = "X-Edge-Cloud-Prefix-Hit-Blocks"
HEADER_HIT_TOKENS = "X-Edge-Cloud-Prefix-Hit-Tokens"

HEARTBEAT_COMMENT = ": edge-cloud-usage-pending\n\n"
USAGE_UNAVAILABLE_COMMENT = ": edge-cloud-usage-unavailable\n\n"
DONE_FRAME = "data: [DONE]\n\n"

DEFAULT_OBJECT = "chat.completion.chunk"


@dataclass(frozen=True)
class LwdGatewayRequest:
    """Everything the cloud needs from one shadow request."""

    request_id: str
    prompt_tokens: int
    block_size: int
    edge_id: int
    model: str


def identity_digest(request_id: str) -> str:
    """Placeholder manifest digest: request identity, not a cache key."""
    raw = hashlib.sha256(request_id.encode("utf-8")).digest()
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def _check_api_key(api_key: str) -> None:
    if (
        not api_key
        or not api_key.isascii()
        or not api_key.isprintable()
        or any(character.isspace() for character in api_key)
    ):
        raise ValueError(
            "the gateway API key must be a non-empty ASCII token "
            "without whitespace"
        )


def build_shadow_request(
    request_id: str,
    prompt_tokens: int,
    model: str,
    *,
    block_size: int = DEFAULT_BLOCK_SIZE,
    edge_id: int = 0,
    consumer_id: str | None = None,
    api_key: str | None = None,
) -> tuple[dict[str, str], dict[str, Any]]:
    """Build the whitelisted shadow request (headers, body).

    Only ``model``/``messages``/``stream``/``stream_options`` cross the
    gateway: the real prompt, sampling parameters and any media stay on the
    edge. ``stream`` is fixed to true regardless of what the user asked the
    edge for, because this stream exists to carry the control message and the
    final usage.
    """
    if not request_id:
        raise ValueError("request_id must not be empty")
    if prompt_tokens < 0:
        raise ValueError("prompt_tokens must not be negative")
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    if api_key is not None:
        _check_api_key(api_key)
    headers = {
        HEADER_PROTOCOL: PROTOCOL_VERSION,
        HEADER_REQUEST_ID: request_id,
        HEADER_EDGE_ID: str(edge_id),
        HEADER_BLOCK_SIZE: str(block_size),
        HEADER_PROMPT_TOKENS: str(prompt_tokens),
        "Accept": "text/event-stream",
    }
    if consumer_id:
        headers["X-Mse-Consumer"] = consumer_id
    if api_key is not None:
        headers["Authorization"] = f"Bearer {api_key}"
    body: dict[str, Any] = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": DIGEST_PREFIX + identity_digest(request_id),
            }
        ],
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    return headers, body


def parse_control_request(
    headers: Mapping[str, str], body: Mapping[str, Any]
) -> LwdGatewayRequest:
    """Parse and validate one shadow request; raises ValueError on any breach.

    ``prompt_tokens`` is read from the header only (gateways strip non-standard
    body fields) and must be plain ASCII decimal: ``int()`` would also accept
    ``"+1024"``, ``" 1024"`` or full-width digits, and a value that only
    *looks* numeric turns a wire breach into a silent accounting mismatch.
    """
    normalized = {key.lower(): value for key, value in headers.items()}

    def required(name: str) -> str:
        try:
            return normalized[name.lower()]
        except KeyError as exc:
            raise ValueError(f"missing required header {name}") from exc

    protocol = required(HEADER_PROTOCOL)
    if protocol != PROTOCOL_VERSION:
        raise ValueError(f"unsupported edge-cloud protocol {protocol!r}")
    request_id = required(HEADER_REQUEST_ID)
    if not request_id:
        raise ValueError(f"{HEADER_REQUEST_ID} must not be empty")
    prompt_tokens_raw = required(HEADER_PROMPT_TOKENS)
    if not (prompt_tokens_raw.isascii() and prompt_tokens_raw.isdecimal()):
        raise ValueError(
            f"{HEADER_PROMPT_TOKENS} must be a non-negative integer"
        )
    # required() must stay outside the int() try: otherwise a missing header
    # is rewrapped as "not an integer" and the real cause disappears.
    block_size_raw = required(HEADER_BLOCK_SIZE)
    try:
        block_size = int(block_size_raw)
    except ValueError as exc:
        raise ValueError("edge-cloud block size must be an integer") from exc
    if block_size <= 0:
        raise ValueError("edge-cloud block size must be positive")
    edge_id_raw = normalized.get(HEADER_EDGE_ID.lower(), "0")
    try:
        edge_id = int(edge_id_raw)
    except ValueError as exc:
        raise ValueError(f"invalid {HEADER_EDGE_ID} {edge_id_raw!r}") from exc
    if edge_id < 0:
        raise ValueError(f"negative {HEADER_EDGE_ID}")

    messages = body.get("messages")
    if not isinstance(messages, Sequence) or not messages:
        raise ValueError("messages must be a non-empty list")
    for message in messages:
        if not isinstance(message, Mapping) or message.get("role") != "user":
            raise ValueError("every manifest message must have role=user")
        content = message.get("content")
        if not isinstance(content, str) or not content.startswith(DIGEST_PREFIX):
            raise ValueError("message content is not an edge-cloud digest")
    model = body.get("model")
    if not isinstance(model, str) or not model:
        raise ValueError("model must be a non-empty string")
    return LwdGatewayRequest(
        request_id=request_id,
        prompt_tokens=int(prompt_tokens_raw),
        block_size=block_size,
        edge_id=edge_id,
        model=model,
    )


def control_message(
    request_id: str, instance_id: str, block_size: int
) -> dict[str, Any]:
    """The JSON object carried inside the first ``delta.content``."""
    return {
        "type": "edge_cloud_probe",
        "protocol": PROTOCOL_VERSION,
        "request_id": request_id,
        "instance_id": instance_id,
        "block_size": block_size,
        # No prefix cache in this deployment: nothing is ever reserved or hit.
        "hit_blocks": 0,
        "hit_tokens": 0,
    }


def usage_dict(
    prompt_tokens: int, completion_tokens: int, cached_tokens: int = 0
) -> dict[str, Any]:
    """Standard OpenAI streaming usage shape (cached detail included)."""
    if min(prompt_tokens, completion_tokens, cached_tokens) < 0:
        raise ValueError("usage counters must not be negative")
    if cached_tokens > prompt_tokens:
        raise ValueError("cached_tokens must not exceed prompt_tokens")
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
        "prompt_tokens_details": {"cached_tokens": cached_tokens},
    }


def _sse(payload: Any) -> str:
    return f"data: {json.dumps(payload, separators=(',', ':'))}\n\n"


def _chunk(
    request_id: str,
    model: str,
    choices: list[Any],
    usage: dict[str, Any] | None = None,
) -> dict[str, Any]:
    chunk: dict[str, Any] = {
        "id": request_id,
        "object": DEFAULT_OBJECT,
        "created": int(time.time()),
        "model": model,
        "choices": choices,
    }
    if usage is not None:
        chunk["usage"] = usage
    return chunk


def probe_frame(request_id: str, model: str, message: Mapping[str, Any]) -> str:
    """Event one: the control message inside a content delta."""
    return _sse(
        _chunk(
            request_id,
            model,
            [
                {
                    "index": 0,
                    "delta": {
                        "role": "assistant",
                        "content": json.dumps(message, separators=(",", ":")),
                    },
                    "finish_reason": None,
                }
            ],
        )
    )


def empty_delta_frame(request_id: str, model: str) -> str:
    """Event two: a valid data frame that adds no content and ends nothing.

    It exists to flush gateways that hold the last chunk back for usage
    extraction.
    """
    return _sse(
        _chunk(
            request_id,
            model,
            [{"index": 0, "delta": {}, "finish_reason": None}],
        )
    )


def usage_frame(
    request_id: str, model: str, usage: Mapping[str, Any]
) -> str:
    """Event four: ``choices: []`` plus the standard usage object."""
    return _sse(_chunk(request_id, model, [], usage=dict(usage)))


def response_headers(
    request_id: str, instance_id: str, block_size: int
) -> dict[str, str]:
    """Observability headers; the handshake itself reads the SSE content."""
    return {
        HEADER_PROTOCOL: PROTOCOL_VERSION,
        HEADER_REQUEST_ID: request_id,
        HEADER_INSTANCE_ID: instance_id,
        HEADER_BLOCK_SIZE: str(block_size),
        HEADER_HIT_BLOCKS: "0",
        HEADER_HIT_TOKENS: "0",
        "Cache-Control": "no-cache",
        "X-Accel-Buffering": "no",
    }
