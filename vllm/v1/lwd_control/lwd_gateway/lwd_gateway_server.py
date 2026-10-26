# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Cloud-side gateway billing endpoint (OpenAI-compatible SSE).

Runs inside the cloud EngineCore process: usage is produced there, so the
HTTP server lives there too and needs no cross-process bridge. One daemon
thread owns uvicorn and its event loop; the engine thread hands finished
requests over with ``call_soon_threadsafe`` and never blocks on the loop.

Waiting costs one coroutine per in-flight request (no thread pool), so the
number of open billing streams is not a capacity knob.
"""

from __future__ import annotations

import asyncio
import threading
import time
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from vllm.config.lwd.gateway import LwdGatewayConfig
from vllm.lwd_logger import lwd_logger
from vllm.v1.lwd_control.lwd_gateway.lwd_gateway_wire import (
    DONE_FRAME,
    HEARTBEAT_COMMENT,
    USAGE_UNAVAILABLE_COMMENT,
    control_message,
    empty_delta_frame,
    parse_control_request,
    probe_frame,
    response_headers,
    usage_dict,
    usage_frame,
)

# A finished request whose billing stream has not shown up yet. The edge opens
# that stream without waiting for anything, so arrival order is not
# guaranteed; without this cache a fast request would bill nothing at all.
_LATE_USAGE_TTL_S = 300.0
_LATE_USAGE_MAX = 4096


class LwdGatewayControlServer:
    """SSE control endpoint plus the usage hand-off from the engine thread."""

    def __init__(self, gateway: LwdGatewayConfig) -> None:
        self._gateway = gateway
        self._loop: asyncio.AbstractEventLoop | None = None
        self._waiters: dict[str, asyncio.Future] = {}
        self._late: dict[str, tuple[Any, float]] = {}
        self._lock = threading.Lock()
        self._closed = False
        self._server: Any = None
        self._thread: threading.Thread | None = None

    @property
    def gateway(self) -> LwdGatewayConfig:
        return self._gateway

    # ------------------------------------------------------------------ #
    # Lifecycle                                                          #
    # ------------------------------------------------------------------ #
    def start(self) -> None:
        """Serve in a daemon thread; the loop is captured on startup."""
        import uvicorn

        config = uvicorn.Config(
            create_lwd_gateway_app(self),
            host=self._gateway.listen_host,
            port=self._gateway.listen_port,
            log_level="warning",
        )
        self._server = uvicorn.Server(config)
        self._thread = threading.Thread(
            target=self._server.run, name="lwd-gateway-ctrl", daemon=True
        )
        self._thread.start()
        lwd_logger.info(
            "[gateway] control endpoint listening on %s:%s instance=%s",
            self._gateway.listen_host,
            self._gateway.listen_port,
            self._gateway.instance_id,
        )

    def attach_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """Called from the app startup hook, on the serving loop's thread."""
        self._loop = loop

    def stop(self) -> None:
        """Stop serving and fail the streams that are still waiting."""
        self._closed = True
        if self._server is not None:
            self._server.should_exit = True
        loop, pending = self._loop, []
        with self._lock:
            pending = list(self._waiters.values())
            self._waiters.clear()
            self._late.clear()
        alive = [future for future in pending if not future.done()]
        if loop is None or not alive:
            return
        error = RuntimeError("gateway control plane stopped")

        def _fail() -> None:
            for future in alive:
                if not future.done():
                    future.set_exception(error)
                    # Often nobody is left to retrieve it: clear the log flag.
                    future.exception()

        try:
            loop.call_soon_threadsafe(_fail)
        except RuntimeError:
            lwd_logger.info(
                "[gateway] stop: loop already closed, dropped %d waiter(s)",
                len(alive),
            )

    # ------------------------------------------------------------------ #
    # Usage hand-off                                                     #
    # ------------------------------------------------------------------ #
    def publish_usage(
        self, request_id: str, prompt_tokens: int, completion_tokens: int
    ) -> None:
        """Engine-thread entry point; never blocks on the serving loop."""
        if self._closed:
            return
        try:
            usage = usage_dict(prompt_tokens, completion_tokens)
        except ValueError:
            lwd_logger.warning(
                "[gateway] refusing to publish invalid usage req=%s "
                "prompt=%d completion=%d",
                request_id,
                prompt_tokens,
                completion_tokens,
            )
            return
        loop = self._loop
        if loop is None:
            # Endpoint not up yet: keep it for the stream that will arrive.
            self._stash_late(request_id, usage)
            return
        try:
            loop.call_soon_threadsafe(self._on_usage, request_id, usage)
        except RuntimeError:
            self._stash_late(request_id, usage)

    def publish_aborted(self, request_id: str) -> None:
        """End the billing stream of an aborted request without usage.

        Unlike ``publish_usage`` this is *not* late-cached: a late arrival
        would settle a *newer* attempt for the same request id (a retry or an
        id reused after abort) with nothing for data -- turning a real attempt
        into "usage unavailable". Settling only whoever is already waiting
        covers the cases that matter, and any earlier stream that missed the
        marker still ends itself at usage_wait_timeout with the same outcome:
        nothing billed.
        """
        if self._closed:
            return
        loop = self._loop
        if loop is None:
            # Endpoint not up (or already down): nobody can be waiting.
            return
        try:
            loop.call_soon_threadsafe(self._deliver_abort, request_id)
        except RuntimeError:
            # Loop already closed; nothing left to settle.
            lwd_logger.info(
                "[gateway] abort for req=%s arrived after loop close",
                request_id,
            )

    def _deliver_abort(self, request_id: str) -> None:
        """Settle a waiting stream with None = end now, no usage frame.

        Synchronous on the serving loop, so dropping the waiter and resolving
        it cannot interleave with a concurrent wait_usage registration.
        """
        with self._lock:
            future = self._waiters.pop(request_id, None)
        if future is not None and not future.done():
            future.set_result(None)

    def _stash_late(self, request_id: str, usage: Any) -> None:
        now = time.monotonic()
        with self._lock:
            self._late[request_id] = (usage, now)
            for key, (_, stamp) in list(self._late.items()):
                if now - stamp > _LATE_USAGE_TTL_S:
                    self._late.pop(key, None)
            while len(self._late) > _LATE_USAGE_MAX:
                self._late.pop(next(iter(self._late)), None)

    def _on_usage(self, request_id: str, usage: Any) -> None:
        """Serving-loop side: settle the waiting stream or stash the usage.

        Runs as a ``call_soon_threadsafe`` callback, i.e. synchronously on the
        serving loop, so there is no await between the two lock acquisitions
        here and a concurrent ``wait_usage`` cannot interleave with them. The
        engine-thread path (``publish_usage`` before the loop exists) takes the
        same lock inside ``_stash_late``.
        """
        with self._lock:
            future = self._waiters.pop(request_id, None)
        if future is None:
            self._stash_late(request_id, usage)
            return
        if not future.done():
            future.set_result(usage)

    async def wait_usage(
        self, request_id: str, timeout: float
    ) -> Any | None:
        """Wait for the real usage; None means "this request has no upstream
        usage", which the stream reports without inventing zeros."""
        loop = asyncio.get_running_loop()
        future: asyncio.Future = loop.create_future()
        with self._lock:
            cached = self._late.pop(request_id, None)
            stale = None if cached is not None else self._waiters.get(request_id)
            if cached is None:
                self._waiters[request_id] = future
        if cached is not None:
            return cached[0]
        if stale is not None and not stale.done():
            # Second stream for one request id (client retry, or an id reused
            # after abort). Only one usage event will ever arrive for it, and
            # it belongs to the live attempt, so the newest registration owns
            # it: settle the older stream now instead of letting it hold a
            # gateway connection open for the whole usage_wait_timeout.
            lwd_logger.warning(
                "[gateway] duplicate billing stream req=%s: superseding the "
                "previous one, which ends without a usage frame",
                request_id,
            )
            stale.set_result(None)
        try:
            return await asyncio.wait_for(future, timeout)
        except asyncio.TimeoutError:
            lwd_logger.warning(
                "[gateway] usage wait timeout req=%s after %.0fs",
                request_id,
                timeout,
            )
            return None
        finally:
            # Covers timeout, cancellation and client disconnect alike: the
            # future is created here, so this is the one place that knows which
            # registration belongs to this stream.
            self.release(request_id, future)

    def release(
        self, request_id: str, future: asyncio.Future | None = None
    ) -> None:
        """Drop a registration (idempotent).

        When ``future`` is given it is dropped only if it is still the
        registered one -- a superseded stream must not unregister the stream
        that replaced it.
        """
        with self._lock:
            current = self._waiters.get(request_id)
            if current is None:
                return
            if future is not None and current is not future:
                return
            self._waiters.pop(request_id, None)

    def pending(self) -> int:
        with self._lock:
            return len(self._waiters)


def create_lwd_gateway_app(server: LwdGatewayControlServer) -> FastAPI:
    """FastAPI app: one SSE billing stream per request, plus /health."""
    gateway = server.gateway

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        server.attach_loop(asyncio.get_running_loop())
        try:
            yield
        finally:
            server.stop()

    app = FastAPI(title="vLLM LWD gateway billing plane", lifespan=lifespan)

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {"status": "ok", "pending_streams": server.pending()}

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request) -> Any:
        try:
            headers = dict(request.headers)
            body = await request.json()
            parsed = parse_control_request(headers, body)
        except ValueError as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)
        except Exception as exc:  # noqa: BLE001 - malformed body from a gateway
            return JSONResponse(
                {"error": f"bad request: {exc}"}, status_code=400
            )

        request_id = parsed.request_id
        model = parsed.model
        instance_id = gateway.instance_id

        async def _frames():
            # Event one: control message inside delta.content (survives
            # gateway DTO conversion, unlike response headers).
            yield probe_frame(
                request_id,
                model,
                control_message(request_id, instance_id, parsed.block_size),
            )
            # Event two: empty delta, flushes gateways holding the last chunk.
            yield empty_delta_frame(request_id, model)
            # Event three: heartbeats while the inference runs.
            usage_task = asyncio.create_task(
                server.wait_usage(request_id, gateway.usage_wait_timeout)
            )
            try:
                while not usage_task.done():
                    done, _ = await asyncio.wait(
                        {usage_task}, timeout=gateway.heartbeat_interval
                    )
                    if not done:
                        yield HEARTBEAT_COMMENT
                usage = usage_task.result()
            finally:
                # A client disconnect surfaces here as GeneratorExit (not
                # CancelledError), so this has to be a finally: left running,
                # the waiter would keep the request id registered until
                # usage_wait_timeout, and a usage event landing in that window
                # is consumed by nobody -- a retried stream for the same id
                # would then find neither a waiter nor a late-cache entry and
                # bill nothing. Cancel only, do not await: the generator is
                # being closed, and the task's own finally drops the
                # registration on the next loop pass.
                if not usage_task.done():
                    usage_task.cancel()
            if usage is None:
                # No real numbers, so no usage frame: zeros would bill this
                # request as free. The stream still ends normally -- dropping
                # it mid-way makes the gateway log an upstream error instead.
                lwd_logger.error(
                    "[gateway] usage unavailable req=%s: ending stream "
                    "without a usage frame",
                    request_id,
                )
                yield USAGE_UNAVAILABLE_COMMENT
                yield DONE_FRAME
                return
            # Events four and five: real usage, then the end of the stream.
            yield usage_frame(request_id, model, usage)
            yield DONE_FRAME

        async def events():
            async for frame in _frames():
                yield frame
            # No release here on purpose: the registration is created inside
            # wait_usage, whose finally drops exactly this stream's future --
            # including on cancellation and on a client disconnect. Releasing
            # by request id from the outside would unregister whichever stream
            # currently holds the id, i.e. possibly a newer one.

        return StreamingResponse(
            events(),
            media_type="text/event-stream",
            headers=response_headers(
                request_id, instance_id, parsed.block_size
            ),
        )

    return app
