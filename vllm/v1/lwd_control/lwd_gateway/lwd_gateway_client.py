# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Edge-side gateway billing client: one stream per request, never waited on.

This deployment has no reuse decision to make, so admission is *not* gated on
the gateway round trip: the shadow request is fired and forgotten, its frames
are drained in the background, and every failure degrades to a warning.
Billing must never add latency to, or break, user inference.
"""

from __future__ import annotations

import asyncio
import json
import threading
from typing import Any

from vllm import envs
from vllm.config.lwd.gateway import LwdGatewayConfig
from vllm.lwd_logger import lwd_logger
from vllm.v1.lwd_control.lwd_gateway.lwd_gateway_wire import (
    build_shadow_request,
)

# Cap on one control/usage frame; a gateway should never send more.
_MAX_FRAME_CHARS = 16 * 1024
_ERROR_BODY_LIMIT = 512


async def _iter_sse_data(content: Any):
    """Yield SSE data payloads, preserving multiline data, ignoring comments."""
    data: list[str] = []
    size = 0
    async for raw_line in content:
        line = raw_line.decode("utf-8", "replace").rstrip("\r\n")
        if not line:
            if data:
                payload = "\n".join(data)
                data.clear()
                size = 0
                yield payload
        elif line == "data" or line.startswith("data:"):
            value = line.partition(":")[2].removeprefix(" ")
            size += len(value) + 1
            if size > _MAX_FRAME_CHARS:
                raise ValueError("gateway SSE frame exceeds the size limit")
            data.append(value)
    if data:
        yield "\n".join(data)


class LwdGatewayBillingClient:
    """Fire-and-forget billing streams on a private event loop thread."""

    def __init__(
        self,
        gateway: LwdGatewayConfig,
        model: str,
        *,
        edge_id: int = 0,
    ) -> None:
        self._gateway = gateway
        self._model = model
        self._edge_id = edge_id
        self._api_key = envs.VLLM_LWD_GATEWAY_API_KEY
        # asyncio keeps only weak references to tasks, and a task parked on a
        # socket read has no pending callback: this set is its only strong
        # reference, so it must not be keyed by request id (a re-submitted id
        # would drop the previous stream mid-drain and lose its usage). The
        # rid index below exists only to *cancel* whole groups on abort; the
        # strong reference contract still lives in the set.
        self._streams: set[asyncio.Task] = set()
        self._rid_tasks: dict[str, set[asyncio.Task]] = {}
        self._session: Any = None
        self._closed = False
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._run_loop, name="lwd-gateway-billing", daemon=True
        )
        self._thread.start()

    def _run_loop(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def submit(self, request_id: str, prompt_tokens: int) -> None:
        """Open this request's billing stream; returns immediately.

        Called from the engine thread. Never raises: a dead loop means no
        billing for this request, which is still better than failing admission.
        """
        if self._closed:
            # Scheduling into a stopping loop would silently drop the
            # coroutine ("never awaited") instead of saying so.
            lwd_logger.warning(
                "[gateway] billing client closed, req=%s not billed", request_id
            )
            return
        coroutine = self._stream(request_id, prompt_tokens)
        try:
            asyncio.run_coroutine_threadsafe(coroutine, self._loop)
        except RuntimeError:
            # Loop already stopped: close the coroutine explicitly, otherwise
            # it is reported as "never awaited" on garbage collection.
            coroutine.close()
            lwd_logger.warning(
                "[gateway] billing loop gone, req=%s not billed", request_id
            )

    def abort(self, request_id: str) -> None:
        """Close every live billing stream for this request id.

        Engine-thread entry for an aborted request. Cancelling drops the
        stream on the cloud side too (its heartbeat loop cancels the usage
        waiter in its finally), so the aborted stream ends *now* instead of
        hanging until usage_wait_timeout -- matching the rule that an abort
        without output is declined, not billed. Other ids keep draining.

        Lossless only because the cloud-side abort notification is not
        landed yet: an edge-aborted request produces no cloud usage event
        awaiting this stream. Once that feature lands, an abort with
        accepted tokens WILL have usage to bill, and this unconditional
        cancel becomes a guaranteed bill loss -- revisit then.
        """
        tasks = self._rid_tasks.pop(request_id, None)
        if not tasks or self._closed:
            return
        for task in tasks:
            try:
                self._loop.call_soon_threadsafe(task.cancel)
            except RuntimeError:
                pass

    def _track(self, task: asyncio.Task, request_id: str) -> None:
        self._streams.add(task)
        self._rid_tasks.setdefault(request_id, set()).add(task)

    def _untrack(self, task: asyncio.Task, request_id: str) -> None:
        self._streams.discard(task)
        group = self._rid_tasks.get(request_id)
        if group is not None:
            group.discard(task)
            if not group:
                self._rid_tasks.pop(request_id, None)

    def close(self) -> None:
        """Stop the loop and release the shared session."""
        self._closed = True

        def _shutdown() -> None:
            if self._session is not None:
                asyncio.ensure_future(self._session.close())
            self._loop.call_later(1.0, self._loop.stop)

        try:
            self._loop.call_soon_threadsafe(_shutdown)
        except RuntimeError:
            pass

    async def _ensure_session(self) -> Any:
        if self._session is None:
            import aiohttp

            self._session = aiohttp.ClientSession()
        return self._session

    async def _stream(self, request_id: str, prompt_tokens: int) -> None:
        import aiohttp

        task = asyncio.current_task()
        if task is not None:
            self._track(task, request_id)
        try:
            headers, body = build_shadow_request(
                request_id,
                prompt_tokens,
                self._model,
                edge_id=self._edge_id,
                consumer_id=self._gateway.consumer_id,
                api_key=self._api_key,
            )
            session = await self._ensure_session()
            timeout = aiohttp.ClientTimeout(
                total=None, connect=self._gateway.connect_timeout
            )
            response = await session.post(
                self._gateway.control_url,
                json=body,
                headers=headers,
                timeout=timeout,
            )
            try:
                if response.status != 200:
                    lwd_logger.warning(
                        "[gateway] billing stream rejected req=%s status=%d "
                        "body=%s",
                        request_id,
                        response.status,
                        await self._error_body(response),
                    )
                    return
                lwd_logger.debug(
                    "[gateway] billing stream open req=%s via=%s",
                    request_id,
                    self._gateway.control_url,
                )
                await self._drain(request_id, response)
            finally:
                response.close()
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - billing is best effort
            lwd_logger.warning(
                "[gateway] billing stream failed req=%s: %s: %s",
                request_id,
                type(exc).__name__,
                exc,
            )
        finally:
            if task is not None:
                self._untrack(task, request_id)

    async def _drain(self, request_id: str, response: Any) -> None:
        """Read the stream to [DONE]: the cloud only finishes it once read."""
        usage: dict | None = None
        instance_id = "?"
        # A gateway may split delta.content across frames, so the control
        # message is accumulated until it parses. Billing does not depend on
        # this (usage comes from its own frame); without it the instance id and
        # the misrouted-request warning would silently never fire.
        content_buf = ""
        async for payload in _iter_sse_data(response.content):
            if payload == "[DONE]":
                break
            try:
                chunk = json.loads(payload)
            except json.JSONDecodeError:
                continue
            if not isinstance(chunk, dict):
                continue
            if chunk.get("usage") is not None:
                usage = chunk["usage"]
                continue
            choices = chunk.get("choices") or []
            if not choices or usage is not None:
                continue
            content = (choices[0].get("delta") or {}).get("content")
            if not content or instance_id != "?":
                continue
            content_buf += content
            if len(content_buf) > _MAX_FRAME_CHARS:
                lwd_logger.warning(
                    "[gateway] control message for req=%s exceeds %d chars; "
                    "giving up on it (billing unaffected)",
                    request_id,
                    _MAX_FRAME_CHARS,
                )
                instance_id = "<oversize>"
                continue
            try:
                message = json.loads(content_buf)
            except json.JSONDecodeError:
                continue
            if not isinstance(message, dict):
                continue
            if "request_id" not in message:
                # Not our control message: an unrelated JSON content frame (a
                # gateway plugin notice, say). It parsed, so it consumed
                # everything accumulated so far -- restart the buffer instead
                # of poisoning it, and do not warn about a request id we
                # cannot read. Our own message is flat, so no proper prefix of
                # it parses as JSON; a foreign frame does.
                content_buf = ""
                continue
            if message["request_id"] != request_id:
                lwd_logger.warning(
                    "[gateway] control message for another request: got=%s "
                    "want=%s (gateway routing?)",
                    message["request_id"],
                    request_id,
                )
            instance_id = str(message.get("instance_id", "?"))
        if usage is None:
            lwd_logger.warning(
                "[gateway] no usage in billing stream req=%s", request_id
            )
            return
        details = usage.get("prompt_tokens_details") or {}
        lwd_logger.info(
            "[gateway] usage req=%s cloud=%s prompt=%s completion=%s total=%s "
            "cached=%s",
            request_id,
            instance_id,
            usage.get("prompt_tokens"),
            usage.get("completion_tokens"),
            usage.get("total_tokens"),
            details.get("cached_tokens"),
        )

    @staticmethod
    async def _error_body(response: Any) -> str:
        try:
            raw = await response.read(_ERROR_BODY_LIMIT)
            return " ".join(raw.decode("utf-8", "replace").split())
        except Exception as exc:  # noqa: BLE001 - diagnostics must not raise
            return f"<unreadable: {type(exc).__name__}>"
