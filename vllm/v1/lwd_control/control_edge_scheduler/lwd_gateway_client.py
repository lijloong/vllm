# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Edge client for the explicit gateway single-stream contract."""

from __future__ import annotations

import asyncio
import json
import math
from collections.abc import AsyncIterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from vllm.v1.lwd_control.control_edge_scheduler.lwd_edge_client import (
    LwdEdgePrefixClient,
    _iter_sse_data,
)
from vllm.v1.lwd_control.lwd_comm.lwd_prefix import (
    GATEWAY_USAGE_EXECUTION_TIMEOUT_S,
    HEADER_USAGE_EXECUTION_TIMEOUT_MS,
    LwdPrefixManifest,
    LwdProbeResult,
)

_MAX_CONTROL_MESSAGE_CHARS = 16 * 1024
_MAX_STARTUP_CANDIDATES = 4


@dataclass(frozen=True)
class LwdGatewayActivation:
    result: LwdProbeResult
    prefix_nonce: int
    drain_task: asyncio.Task[None]


class LwdGatewayClient(LwdEdgePrefixClient):
    """Post to one complete gateway URL without appending a route path."""

    def _usage_execution_timeout_ms(self, max_tokens: int | None) -> int:
        if max_tokens is not None and (type(max_tokens) is not int or max_tokens < 0):
            raise ValueError("max_tokens must be a non-negative integer or None")
        output_allowance_s = max_tokens or 0
        return math.ceil(
            (GATEWAY_USAGE_EXECUTION_TIMEOUT_S + output_allowance_s) * 1000
        )

    def _decode_probe(self, message: Mapping[str, Any]) -> LwdProbeResult:
        return LwdProbeResult.from_gateway_control_message(message)

    def build_gateway_request(
        self,
        manifest: LwdPrefixManifest,
        *,
        edge_id: int,
        model: str,
        multimodal: bool = False,
    ) -> tuple[dict[str, str], dict[str, Any]]:
        headers, body = self.build_control_request(
            manifest,
            edge_id=edge_id,
            model=model,
            multimodal=multimodal,
        )
        headers["Accept"] = "text/event-stream"
        return headers, body

    async def activate_gateway(
        self,
        session: Any,
        request_id: str,
        prompt_token_ids: list[int],
        *,
        edge_id: int,
        model: str,
        manifest: LwdPrefixManifest | None = None,
        media_items: Sequence[Any] = (),
        max_tokens: int | None = None,
    ) -> LwdGatewayActivation:
        """Return only after the startup content identifies the serving DP."""
        if manifest is None:
            manifest = self.build_manifest(request_id, prompt_token_ids, media_items)
        multimodal = bool(media_items)
        headers, body = self.build_gateway_request(
            manifest,
            edge_id=edge_id,
            model=model,
            multimodal=multimodal,
        )
        headers[HEADER_USAGE_EXECUTION_TIMEOUT_MS] = str(
            self._usage_execution_timeout_ms(max_tokens)
        )
        response = await self._post_url(self._control_url, session, headers, body)
        if response.status != 200:
            try:
                detail = await self._read_error_body(response)
                raise RuntimeError(
                    f"gateway activation HTTP {response.status} "
                    f"url={self._control_url} body={detail}"
                )
            finally:
                response.close()

        sse_data = _iter_sse_data(response.content)
        try:
            probe, prefix_nonce = await asyncio.wait_for(
                self._read_startup(sse_data, request_id, prompt_token_ids, multimodal),
                self._probe_timeout,
            )
        except BaseException:
            response.close()
            raise
        drain_task = self._start_drain(request_id, response, sse_data)
        return LwdGatewayActivation(probe, prefix_nonce, drain_task)

    async def _read_startup(
        self,
        sse_data: AsyncIterator[str],
        request_id: str,
        prompt_token_ids: list[int],
        multimodal: bool,
    ) -> tuple[LwdProbeResult, int]:
        candidates: list[str] = []
        async for payload in sse_data:
            if payload == "[DONE]":
                raise RuntimeError("gateway stream ended before the startup message")
            try:
                chunk = json.loads(payload)
            except json.JSONDecodeError:
                continue
            if not isinstance(chunk, dict):
                continue
            choices = chunk.get("choices") or []
            if not choices:
                continue
            content = (choices[0].get("delta") or {}).get("content")
            if not content:
                continue
            if content.lstrip().startswith("{"):
                candidates = [candidate + content for candidate in candidates]
                candidates.append(content)
                if len(candidates) > _MAX_STARTUP_CANDIDATES:
                    candidates.pop(0)
            elif candidates:
                candidates = [candidate + content for candidate in candidates]
            else:
                continue
            for candidate in list(candidates):
                if len(candidate) > _MAX_CONTROL_MESSAGE_CHARS:
                    raise RuntimeError("gateway startup message exceeds the size limit")
                try:
                    message = json.loads(candidate)
                except json.JSONDecodeError:
                    continue
                candidates.remove(candidate)
                if not isinstance(message, dict) or "request_id" not in message:
                    continue
                probe = self._parse_probe(
                    message, request_id, prompt_token_ids, multimodal
                )
                prefix_nonce = message.get("prefix_nonce")
                if type(prefix_nonce) is not int or prefix_nonce <= 0:
                    raise RuntimeError("cloud returned an invalid prefix nonce")
                return probe, prefix_nonce
        raise RuntimeError("gateway stream ended before the startup message")
