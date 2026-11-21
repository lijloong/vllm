# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Edge client for the single-request gateway SSE contract."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterable, AsyncIterator, Sequence
from dataclasses import dataclass
from typing import Any

from vllm import envs
from vllm.logger import init_logger
from vllm.v1.lwd_control.lwd_comm.lwd_mm_identity import (
    mm_abi_header_value,
    negotiate_lwd_protocol,
)
from vllm.v1.lwd_control.lwd_comm.lwd_prefix import (
    HEADER_EDGE_ID,
    HEADER_MM_ABI,
    HEADER_PROTOCOL,
    PROTOCOL_VERSION,
    PROTOCOL_VERSION_MM,
    LwdPrefixHasher,
    LwdPrefixManifest,
    LwdProbeResult,
)

logger = init_logger(__name__)

_MAX_CONTROL_MESSAGE_CHARS = 16 * 1024
_MAX_STARTUP_CANDIDATES = 4


@dataclass(frozen=True)
class LwdGatewayActivation:
    result: LwdProbeResult
    prefix_nonce: int
    drain_task: asyncio.Task[None]


async def _iter_sse_data(content: AsyncIterable[bytes]) -> AsyncIterator[str]:
    """Read SSE events, preserving multiline data and ignoring heartbeats."""
    data: list[str] = []
    size = 0
    async for raw_line in content:
        line = raw_line.decode("utf-8").rstrip("\r\n")
        if not line:
            if data:
                payload = "\n".join(data)
                data.clear()
                size = 0
                yield payload
        elif line.startswith(":"):
            yield line
        elif line == "data" or line.startswith("data:"):
            value = line.partition(":")[2].removeprefix(" ")
            size += len(value) + 1
            if size > _MAX_CONTROL_MESSAGE_CHARS:
                raise ValueError(
                    "edge-cloud SSE event exceeds the control message limit"
                )
            data.append(value)
    if data:
        yield "\n".join(data)


class LwdEdgePrefixClient:
    """Client bound to one gateway or direct-DP coordination endpoint."""

    def __init__(
        self,
        control_url: str,
        block_size: int,
        tenant_key: bytes,
        consumer_id: str | None = None,
        connect_timeout: float = 10.0,
        processor_fingerprint: bytes | None = None,
        probe_timeout: float = 30.0,
    ) -> None:
        if not control_url.startswith(("http://", "https://")):
            raise ValueError("control_url must use http:// or https://")
        if probe_timeout <= 0:
            raise ValueError("probe_timeout must be positive")
        self._control_url = control_url.rstrip("/")
        self._hasher = LwdPrefixHasher(tenant_key, block_size, processor_fingerprint)
        self._mm_abi_header: str | None = (
            mm_abi_header_value(processor_fingerprint)
            if processor_fingerprint is not None
            else None
        )
        self._consumer_id = consumer_id
        self._connect_timeout = connect_timeout
        self._probe_timeout = probe_timeout
        self._drain_tasks: set[Any] = set()
        api_key = envs.VLLM_LWD_CLOUD_API_KEY
        if api_key is not None and (
            not api_key
            or not api_key.isascii()
            or not api_key.isprintable()
            or any(c.isspace() for c in api_key)
        ):
            raise ValueError(
                "VLLM_LWD_CLOUD_API_KEY must contain a non-empty ASCII "
                "API token without whitespace"
            )
        self._api_key: str | None = api_key

    @property
    def block_size(self) -> int:
        return self._hasher.block_size

    def build_manifest(
        self,
        request_id: str,
        prompt_token_ids: list[int],
        media_items: Sequence[Any] = (),
    ) -> LwdPrefixManifest:
        return self._hasher.build_manifest(
            request_id, prompt_token_ids, media_items=media_items
        )

    def build_control_request(
        self,
        manifest: LwdPrefixManifest,
        *,
        edge_id: int,
        model: str,
        multimodal: bool = False,
    ) -> tuple[dict[str, str], dict[str, Any]]:
        headers = manifest.to_headers()
        if multimodal:
            if self._mm_abi_header is None:
                raise ValueError(
                    "multimodal probe needs a processor fingerprint for "
                    f"{HEADER_MM_ABI}"
                )
            headers[HEADER_PROTOCOL] = PROTOCOL_VERSION_MM
            headers[HEADER_MM_ABI] = self._mm_abi_header
        headers[HEADER_EDGE_ID] = str(edge_id)
        headers["Accept"] = "text/event-stream"
        if self._consumer_id:
            headers["X-Mse-Consumer"] = self._consumer_id
        if self._api_key is not None:
            headers["Authorization"] = f"Bearer {self._api_key}"
        body: dict[str, Any] = {
            "model": model,
            "messages": manifest.to_messages(),
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        return headers, body

    async def activate(
        self,
        session: Any,
        request_id: str,
        prompt_token_ids: list[int],
        *,
        edge_id: int,
        model: str,
        manifest: LwdPrefixManifest | None = None,
        media_items: Sequence[Any] = (),
    ) -> LwdGatewayActivation:
        """Wait for the startup content, then drain usage in the background."""
        if manifest is None:
            manifest = self.build_manifest(request_id, prompt_token_ids, media_items)
        multimodal = bool(media_items)
        headers, body = self.build_control_request(
            manifest,
            edge_id=edge_id,
            model=model,
            multimodal=multimodal,
        )
        response = await self._post("/v1/chat/completions", session, headers, body)
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

    async def _post(
        self, path: str, session: Any, headers: dict[str, str], body: dict[str, Any]
    ) -> Any:
        import aiohttp

        timeout = aiohttp.ClientTimeout(total=None, connect=self._connect_timeout)
        try:
            return await asyncio.wait_for(
                session.post(
                    f"{self._control_url}{path}",
                    json=body,
                    headers=headers,
                    timeout=timeout,
                ),
                self._probe_timeout,
            )
        except asyncio.TimeoutError as exc:
            raise RuntimeError(
                f"control request timed out after {self._probe_timeout}s "
                f"url={self._control_url}{path}"
            ) from exc

    def _parse_probe(
        self,
        message: Any,
        request_id: str,
        prompt_token_ids: list[int],
        multimodal: bool,
    ) -> LwdProbeResult:
        expected_protocol = PROTOCOL_VERSION_MM if multimodal else PROTOCOL_VERSION
        protocol = message.get("protocol") if isinstance(message, dict) else None
        mm_abi = message.get("mm_abi") if isinstance(message, dict) else None
        try:
            negotiate_lwd_protocol(
                protocol,
                mm_abi,
                expected_protocol=expected_protocol,
                expected_mm_abi=self._mm_abi_header,
            )
        except ValueError as exc:
            if protocol != expected_protocol:
                raise RuntimeError(
                    f"cloud did not acknowledge {expected_protocol}"
                ) from exc
            if multimodal:
                raise RuntimeError("cloud MM-ABI fingerprint does not match") from exc
            raise RuntimeError("cloud returned MM-ABI for a text-only probe") from exc
        probe = LwdProbeResult.from_control_message(message)
        if probe.request_id != request_id:
            raise RuntimeError("cloud returned a different request ID")
        if probe.hit_tokens > len(prompt_token_ids):
            raise RuntimeError("cloud prefix hit exceeds the prompt length")
        return probe

    def _start_drain(
        self, request_id: str, response: Any, sse_data: AsyncIterator[str]
    ) -> asyncio.Task[None]:
        task = asyncio.create_task(self._drain_stream(request_id, response, sse_data))
        self._drain_tasks.add(task)

        def finish(_task: asyncio.Task[None]) -> None:
            self._drain_tasks.discard(_task)
            try:
                response.close()
            except Exception:  # noqa: BLE001
                logger.debug(
                    "[LWD][edge-client] gateway stream close failed req=%s",
                    request_id,
                    exc_info=True,
                )

        task.add_done_callback(finish)
        return task

    async def _drain_stream(
        self, request_id: str, response: Any, sse_data: AsyncIterator[str]
    ) -> None:
        usage: dict | None = None
        unavailable = False
        try:
            async for payload in sse_data:
                if payload == "[DONE]":
                    break
                if payload == ": edge-cloud-usage-unavailable":
                    unavailable = True
                    continue
                try:
                    chunk = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                if chunk.get("usage") is not None:
                    usage = chunk["usage"]
            if usage is None:
                if unavailable:
                    logger.info(
                        "[LWD][edge-client] usage unavailable in gateway stream req=%s",
                        request_id,
                    )
                else:
                    logger.warning(
                        "[LWD][edge-client] usage missing in gateway stream req=%s",
                        request_id,
                    )
                return
            details = usage.get("prompt_tokens_details") or {}
            logger.info(
                "[LWD][edge-client] usage req=%s prompt=%s completion=%s "
                "total=%s cached=%s",
                request_id,
                usage.get("prompt_tokens"),
                usage.get("completion_tokens"),
                usage.get("total_tokens"),
                details.get("cached_tokens"),
            )
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001
            logger.warning(
                "[LWD][edge-client] gateway stream drain failed req=%s",
                request_id,
                exc_info=True,
            )
        finally:
            try:
                response.close()
            except Exception:  # noqa: BLE001
                logger.debug(
                    "[LWD][edge-client] gateway stream close failed req=%s",
                    request_id,
                    exc_info=True,
                )

    @staticmethod
    async def _read_error_body(response: Any, limit: int = 512) -> str:
        try:
            raw = await response.content.read(limit)
            return " ".join(raw.decode("utf-8", "replace").split())
        except Exception as exc:  # noqa: BLE001
            return f"<unreadable: {type(exc).__name__}>"
