# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Gateway wire contract: shadow request, control message, SSE frames.

These are the bytes the platform gateway sees, so they are pinned literally:
a rename or a reordered field breaks billing/observability without any error
on either side.
"""

from __future__ import annotations

import base64
import json

import pytest

from vllm.v1.lwd_control.lwd_gateway import lwd_gateway_wire as wire

pytestmark = pytest.mark.cpu_test

MODEL = "qwen3.5"
RID = "chatcmpl-abc123"


def _headers(**overrides):
    headers, _ = wire.build_shadow_request(RID, 1024, MODEL)
    headers.update(overrides)
    return headers


def _body(**overrides):
    _, body = wire.build_shadow_request(RID, 1024, MODEL)
    body.update(overrides)
    return body


# --------------------------------------------------------------------- #
# shadow request                                                         #
# --------------------------------------------------------------------- #
def test_shadow_body_is_the_openai_whitelist() -> None:
    _, body = wire.build_shadow_request(RID, 1024, MODEL)
    assert set(body) == {"model", "messages", "stream", "stream_options"}
    assert body["stream"] is True
    assert body["stream_options"] == {"include_usage": True}
    assert body["model"] == MODEL
    # no sampling params, no output cap, no real prompt
    assert not {"max_tokens", "temperature", "top_p", "n"} & set(body)


def test_shadow_messages_carry_one_identity_digest() -> None:
    _, body = wire.build_shadow_request(RID, 1024, MODEL)
    assert body["messages"] == [
        {"role": "user", "content": "ecb1:" + wire.identity_digest(RID)}
    ]


def test_identity_digest_is_url_safe_and_stable() -> None:
    digest = wire.identity_digest(RID)
    assert len(digest) == 43  # 32 bytes, base64url without padding
    assert "=" not in digest and "+" not in digest and "/" not in digest
    assert base64.urlsafe_b64decode(digest + "=") == (
        __import__("hashlib").sha256(RID.encode()).digest()
    )
    assert digest == wire.identity_digest(RID)
    assert digest != wire.identity_digest(RID + "-other")


def test_request_headers_match_the_spec() -> None:
    headers, _ = wire.build_shadow_request(
        RID, 1024, MODEL, block_size=256, edge_id=0, consumer_id="tenant-a"
    )
    assert headers == {
        "X-Edge-Cloud-Protocol": "edge-cloud-prefix-v1",
        "X-Edge-Cloud-Request-ID": RID,
        "X-Edge-Cloud-Edge-Id": "0",
        "X-Edge-Cloud-Block-Size": "256",
        "X-Edge-Cloud-Prompt-Tokens": "1024",
        "Accept": "text/event-stream",
        "X-Mse-Consumer": "tenant-a",
    }
    assert "Authorization" not in headers


def test_bearer_token_shape() -> None:
    headers, _ = wire.build_shadow_request(RID, 8, MODEL, api_key="sk-gw")
    assert headers["Authorization"] == "Bearer sk-gw"
    for bad in ("", " ", "sk with space", "sk-ünïcode"):
        with pytest.raises(ValueError):
            wire.build_shadow_request(RID, 8, MODEL, api_key=bad)


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"prompt_tokens": -1}, "prompt_tokens"),
        ({"block_size": 0}, "block_size"),
        ({"request_id": ""}, "request_id"),
    ],
)
def test_shadow_request_rejects_bad_inputs(kwargs, match) -> None:
    kwargs = {"prompt_tokens": 8, "block_size": 256, "request_id": RID} | kwargs
    with pytest.raises(ValueError, match=match):
        wire.build_shadow_request(
            kwargs.pop("request_id"),
            kwargs.pop("prompt_tokens"),
            MODEL,
            **kwargs,
        )


# --------------------------------------------------------------------- #
# parsing                                                                #
# --------------------------------------------------------------------- #
def test_parse_round_trips_the_shadow_request() -> None:
    headers, body = wire.build_shadow_request(RID, 1024, MODEL, edge_id=3)
    parsed = wire.parse_control_request(headers, body)
    assert parsed == wire.LwdGatewayRequest(
        request_id=RID,
        prompt_tokens=1024,
        block_size=256,
        edge_id=3,
        model=MODEL,
    )


def test_parse_is_case_insensitive_on_headers() -> None:
    headers, body = wire.build_shadow_request(RID, 16, MODEL)
    lowered = {key.lower(): value for key, value in headers.items()}
    assert wire.parse_control_request(lowered, body).request_id == RID


def test_missing_headers_are_rejected() -> None:
    for name in (
        wire.HEADER_PROTOCOL,
        wire.HEADER_REQUEST_ID,
        wire.HEADER_PROMPT_TOKENS,
        wire.HEADER_BLOCK_SIZE,
    ):
        headers = _headers()
        del headers[name]
        with pytest.raises(ValueError, match=name):
            wire.parse_control_request(headers, _body())


@pytest.mark.parametrize(
    "value", ["", " ", "-10", "10.0", "abc", "+10", "1_0", " 10", "１０"]
)
def test_prompt_tokens_must_be_plain_decimal(value) -> None:
    with pytest.raises(ValueError):
        wire.parse_control_request(
            _headers(**{wire.HEADER_PROMPT_TOKENS: value}), _body()
        )


def test_unknown_protocol_is_rejected() -> None:
    with pytest.raises(ValueError, match="unsupported edge-cloud protocol"):
        wire.parse_control_request(
            _headers(**{wire.HEADER_PROTOCOL: "edge-cloud-prefix-v2"}), _body()
        )


@pytest.mark.parametrize(
    "messages",
    [
        [],
        "not-a-list",
        [{"role": "assistant", "content": "ecb1:AAAA"}],
        [{"role": "user", "content": "please summarize this"}],
        [{"role": "user", "content": 42}],
    ],
)
def test_manifest_messages_must_be_digests(messages) -> None:
    with pytest.raises(ValueError):
        wire.parse_control_request(_headers(), _body(messages=messages))


def test_model_is_required() -> None:
    for model in (None, "", 42):
        with pytest.raises(ValueError, match="model"):
            wire.parse_control_request(_headers(), _body(model=model))


def test_negative_edge_id_is_rejected() -> None:
    with pytest.raises(ValueError):
        wire.parse_control_request(
            _headers(**{wire.HEADER_EDGE_ID: "-1"}), _body()
        )


# --------------------------------------------------------------------- #
# frames                                                                 #
# --------------------------------------------------------------------- #
def test_control_message_reports_zero_hit() -> None:
    message = wire.control_message(RID, "cloud-0", 256)
    assert message == {
        "type": "edge_cloud_probe",
        "protocol": "edge-cloud-prefix-v1",
        "request_id": RID,
        "instance_id": "cloud-0",
        "block_size": 256,
        "hit_blocks": 0,
        "hit_tokens": 0,
    }


def test_usage_keeps_the_openai_shape() -> None:
    assert wire.usage_dict(1024, 128) == {
        "prompt_tokens": 1024,
        "completion_tokens": 128,
        "total_tokens": 1152,
        "prompt_tokens_details": {"cached_tokens": 0},
    }
    with pytest.raises(ValueError):
        wire.usage_dict(10, 1, cached_tokens=11)
    with pytest.raises(ValueError):
        wire.usage_dict(-1, 0)


def _data(frame: str) -> dict:
    assert frame.startswith("data: ") and frame.endswith("\n\n")
    return json.loads(frame[len("data: ") : -2])


def test_probe_frame_carries_the_control_message_in_content() -> None:
    message = wire.control_message(RID, "cloud-0", 256)
    chunk = _data(wire.probe_frame(RID, MODEL, message))
    assert chunk["object"] == "chat.completion.chunk"
    assert chunk["id"] == RID and chunk["model"] == MODEL
    (choice,) = chunk["choices"]
    assert choice["finish_reason"] is None
    assert choice["delta"]["role"] == "assistant"
    assert json.loads(choice["delta"]["content"]) == message
    assert "usage" not in chunk


def test_empty_delta_frame_adds_nothing_and_ends_nothing() -> None:
    chunk = _data(wire.empty_delta_frame(RID, MODEL))
    assert chunk["choices"] == [{"index": 0, "delta": {}, "finish_reason": None}]
    assert "usage" not in chunk


def test_usage_frame_has_empty_choices() -> None:
    usage = wire.usage_dict(64, 8)
    chunk = _data(wire.usage_frame(RID, MODEL, usage))
    assert chunk["choices"] == []
    assert chunk["usage"] == usage


def test_comment_and_done_literals() -> None:
    assert wire.HEARTBEAT_COMMENT == ": edge-cloud-usage-pending\n\n"
    assert wire.USAGE_UNAVAILABLE_COMMENT == ": edge-cloud-usage-unavailable\n\n"
    assert wire.DONE_FRAME == "data: [DONE]\n\n"


def test_response_headers_are_observability_only() -> None:
    assert wire.response_headers(RID, "cloud-0", 256) == {
        "X-Edge-Cloud-Protocol": "edge-cloud-prefix-v1",
        "X-Edge-Cloud-Request-ID": RID,
        "X-Edge-Cloud-Instance-ID": "cloud-0",
        "X-Edge-Cloud-Block-Size": "256",
        "X-Edge-Cloud-Prefix-Hit-Blocks": "0",
        "X-Edge-Cloud-Prefix-Hit-Tokens": "0",
        "Cache-Control": "no-cache",
        "X-Accel-Buffering": "no",
    }
