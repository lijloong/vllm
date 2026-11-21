# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Asynchronous single-stream gateway activation and prefix negotiation."""

from __future__ import annotations

import hashlib
import queue
from collections.abc import Callable
from dataclasses import dataclass
from dataclasses import replace as dataclass_replace
from enum import Enum
from typing import Any

from vllm.config.lwd.topology import LwdConnectionId
from vllm.logger import init_logger
from vllm.v1.lwd_control.control_edge_scheduler.lwd_cloud_health import (
    LwdCloudHealthTracker,
)
from vllm.v1.lwd_control.control_edge_scheduler.lwd_edge_client import (
    LwdEdgePrefixClient,
)
from vllm.v1.lwd_control.control_edge_scheduler.lwd_probe_runtime import (
    LwdProbeRuntime,
)
from vllm.v1.lwd_control.lwd_comm.lwd_mm_identity import compute_processor_fingerprint
from vllm.v1.lwd_control.lwd_comm.lwd_prefix import PROBE_HEADROOM_S, LwdProbeResult

logger = init_logger(__name__)


class LwdProbeSubmission(Enum):
    SUBMITTED = "submitted"
    UNAVAILABLE = "unavailable"
    FAILED = "failed"


@dataclass(frozen=True)
class _MediaIdentity:
    modality: str
    digest: bytes
    offset: int
    length: int


@dataclass(frozen=True)
class LwdProbeCompletion:
    request: Any
    probe_nonce: int = 0
    connection_id: LwdConnectionId | None = None
    result: LwdProbeResult | None = None
    chain: list[bytes] | None = None
    activation_error: str | None = None


@dataclass
class _InflightProbe:
    request: Any
    generation: int
    edge_id: int
    media_items: tuple[_MediaIdentity, ...]
    future: Any = None
    connection_id: LwdConnectionId | None = None
    drain_task: Any = None
    prefix_nonce: int = 0


class LwdEdgeProbeManager:
    """Run one gateway stream per request off the engine thread."""

    def __init__(self, vllm_config: Any, wakeup: Callable[[], None]) -> None:
        self._vllm_config = vllm_config
        self._wakeup = wakeup
        self._runtime: LwdProbeRuntime | None = None
        self._done: queue.Queue[tuple[str, int, Any]] = queue.Queue()
        self._inflight: dict[str, _InflightProbe] = {}
        self._generation = 0
        self._clients: dict[tuple[int, int, int], LwdEdgePrefixClient] = {}
        self._hash_sizes: dict[tuple[int, int], int] = {}
        self._processor_fingerprint: bytes | None = None
        self._tenant_key: bytes | None = None
        self._targets: dict[LwdConnectionId, str] = {}
        for connection in vllm_config.lwd_config.connections:
            connection_id = connection.connection_id
            cloud = connection.cloud
            url = cloud.coord_url or f"http://{cloud.addr}:{cloud.coord_port}"
            self._targets[connection_id] = url.rstrip("/")
        self._health = LwdCloudHealthTracker(
            tuple((target.cloud_id, target.dp_idx) for target in self._targets)
        )

    @property
    def enabled(self) -> bool:
        coordination = self._vllm_config.lwd_config.coordination
        return bool(coordination and coordination.enabled and self._targets)

    @property
    def pending(self) -> bool:
        return bool(self._inflight) or not self._done.empty()

    def _edge_id(self, request: Any) -> int:
        if self._vllm_config.lwd_config.topology.deployment.scene == "edge_share":
            return request.client_index
        return self._vllm_config.lwd_config.instance_id

    def _fingerprint(self) -> bytes:
        if self._processor_fingerprint is None:
            self._processor_fingerprint = compute_processor_fingerprint(
                self._vllm_config.model_config
            )
        return self._processor_fingerprint

    def _key(self) -> bytes:
        if self._tenant_key is None:
            path = self._vllm_config.lwd_config.coordination.tenant_key_file
            if path:
                with open(path, "rb") as key_file:
                    self._tenant_key = key_file.read()
            else:
                self._tenant_key = b""
        return self._tenant_key

    def _client(self, target: LwdConnectionId, block_size: int) -> LwdEdgePrefixClient:
        key = (target.cloud_id, target.dp_idx, block_size)
        client = self._clients.get(key)
        if client is not None:
            return client
        coordination = self._vllm_config.lwd_config.coordination
        client = LwdEdgePrefixClient(
            self._targets[target],
            block_size,
            self._key(),
            consumer_id=coordination.consumer_id,
            connect_timeout=coordination.connect_timeout,
            processor_fingerprint=self._fingerprint(),
            probe_timeout=coordination.probe_timeout,
        )
        for stale in tuple(self._clients):
            if stale[:2] == key[:2] and stale != key:
                del self._clients[stale]
        self._clients[key] = client
        return client

    @staticmethod
    def _media_items(request: Any) -> tuple[_MediaIdentity, ...]:
        return tuple(
            _MediaIdentity(
                modality=feature.modality,
                digest=hashlib.sha256(feature.identifier.encode("utf-8")).digest(),
                offset=feature.mm_position.offset,
                length=feature.mm_position.length,
            )
            for feature in getattr(request, "mm_features", ())
            if feature.modality == "image"
        )

    def _visible_targets(self, edge_id: int) -> tuple[LwdConnectionId, ...]:
        return tuple(target for target in self._targets if target.edge_id == edge_id)

    def _uses_prefix_hit(self, request: Any) -> bool:
        coordination = self._vllm_config.lwd_config.coordination
        return bool(
            self._vllm_config.cache_config.enable_prefix_caching
            and not coordination.placement_only
            and not request.skip_reading_prefix_cache
        )

    def _selected_target(self, edge_id: int) -> LwdConnectionId | None:
        visible = self._visible_targets(edge_id)
        allowed = {(target.cloud_id, target.dp_idx): target for target in visible}
        healthy = self._health.candidates()
        targets = tuple(allowed[key] for key in healthy if key in allowed)
        if not targets:
            return None
        selected = self._health.next(
            allowed={(target.cloud_id, target.dp_idx) for target in targets}
        )
        if selected is None:
            return None
        return allowed[selected]

    def _resolve_connection(
        self, result: LwdProbeResult, edge_id: int
    ) -> tuple[LwdConnectionId, LwdProbeResult]:
        visible = self._visible_targets(edge_id)
        cloud_ids = {target.cloud_id for target in visible}
        cloud_id = result.cloud_id
        if cloud_id is None:
            if len(cloud_ids) != 1:
                raise RuntimeError(
                    "gateway must return cloud_id when multiple Clouds are visible"
                )
            cloud_id = next(iter(cloud_ids))
        dp_indices = {
            target.dp_idx for target in visible if target.cloud_id == cloud_id
        }
        if not dp_indices:
            raise RuntimeError(f"gateway returned an unconfigured Cloud: {cloud_id}")
        engine_index = result.engine_index
        if engine_index is None:
            if len(dp_indices) != 1:
                raise RuntimeError(
                    "gateway must return engine_index when multiple DPs are visible"
                )
            engine_index = next(iter(dp_indices))
        connection = next(
            (
                candidate
                for candidate in visible
                if (candidate.cloud_id, candidate.dp_idx) == (cloud_id, engine_index)
            ),
            None,
        )
        if connection is None:
            raise RuntimeError(
                "gateway returned an unconfigured Cloud/DP: "
                f"({cloud_id}, {engine_index})"
            )
        expected_instance_id = f"cloud-{cloud_id}"
        if result.instance_id != expected_instance_id:
            raise RuntimeError(
                "gateway returned a different instance: "
                f"expected={expected_instance_id} actual={result.instance_id}"
            )
        normalized = dataclass_replace(
            result, cloud_id=cloud_id, engine_index=engine_index
        )
        return connection, normalized

    def submit(self, request: Any) -> LwdProbeSubmission:
        if request.request_id in self._inflight:
            self.abort(request.request_id)
        if not self.enabled:
            return LwdProbeSubmission.UNAVAILABLE
        edge_id = self._edge_id(request)
        target = self._selected_target(edge_id)
        if target is None:
            return LwdProbeSubmission.FAILED
        if self._runtime is None:
            self._runtime = LwdProbeRuntime()
        runtime = self._runtime
        coordination = self._vllm_config.lwd_config.coordination
        self._generation += 1
        generation = self._generation
        media_items = self._media_items(request)
        inflight = _InflightProbe(request, generation, edge_id, media_items)
        inflight.connection_id = target
        self._inflight[request.request_id] = inflight
        placement_only = not self._uses_prefix_hit(request)

        async def job() -> tuple[Any, ...]:
            session = await runtime.ensure_session()
            block_size = self._hash_sizes.get(
                (target.cloud_id, target.dp_idx),
                self._vllm_config.cache_config.block_size,
            )
            client = self._client(target, block_size)
            manifest = client.build_manifest(
                request.request_id,
                list(request.prompt_token_ids),
                media_items=media_items,
            )
            activation = await client.activate(
                session,
                request.request_id,
                list(request.prompt_token_ids),
                edge_id=edge_id,
                manifest=manifest,
                media_items=media_items,
                model=self._model_name(),
            )
            connection, result = self._resolve_connection(activation.result, edge_id)
            chain = None if placement_only else list(manifest.full_block_hashes)
            if self._inflight.get(request.request_id) is inflight:
                inflight.connection_id = connection
                inflight.drain_task = activation.drain_task
                inflight.prefix_nonce = activation.prefix_nonce
            return connection, result, chain, block_size, activation.prefix_nonce

        coroutine = job()
        try:
            future = runtime.submit(
                coroutine, timeout=coordination.probe_timeout + PROBE_HEADROOM_S
            )
        except Exception:
            coroutine.close()
            logger.exception("[LWD] failed to submit gateway activation")
            self._inflight.pop(request.request_id, None)
            return LwdProbeSubmission.FAILED
        inflight.future = future

        def done(future: Any) -> None:
            try:
                outcome = future.result()
            except BaseException as exc:
                outcome = str(exc)
            self._done.put((request.request_id, generation, outcome))
            self._wakeup()

        future.add_done_callback(done)
        return LwdProbeSubmission.SUBMITTED

    def _model_name(self) -> str:
        model_config = self._vllm_config.model_config
        served = model_config.served_model_name
        if isinstance(served, (list, tuple)):
            return str(served[0] if served else model_config.model)
        return str(served or model_config.model)

    def mark_admitted(self, request_id: str, nonce: int) -> None:
        inflight = self._inflight.get(request_id)
        if inflight is not None and inflight.prefix_nonce == nonce:
            self._inflight.pop(request_id, None)

    def abort(self, request_id: str) -> None:
        inflight = self._inflight.pop(request_id, None)
        if inflight is None:
            return
        if inflight.drain_task is None and inflight.future is not None:
            inflight.future.cancel()

    def drain(self) -> list[LwdProbeCompletion]:
        completions: list[LwdProbeCompletion] = []
        while True:
            try:
                request_id, generation, outcome = self._done.get_nowait()
            except queue.Empty:
                break
            inflight = self._inflight.get(request_id)
            if inflight is None or inflight.generation != generation:
                continue
            if isinstance(outcome, str):
                if inflight.connection_id is not None:
                    self._health.record_failure(
                        (
                            inflight.connection_id.cloud_id,
                            inflight.connection_id.dp_idx,
                        )
                    )
                completions.append(
                    LwdProbeCompletion(
                        inflight.request,
                        probe_nonce=inflight.prefix_nonce,
                        connection_id=inflight.connection_id,
                        activation_error=outcome,
                    )
                )
                continue
            connection, result, chain, build_size, prefix_nonce = outcome
            dp_id = (connection.cloud_id, connection.dp_idx)
            self._health.record_success(dp_id)
            learned = int(result.block_size)
            self._hash_sizes[dp_id] = learned
            if chain is not None and learned != build_size:
                rebuilt = self._client(connection, learned).build_manifest(
                    request_id,
                    list(inflight.request.prompt_token_ids),
                    media_items=inflight.media_items,
                )
                chain = list(rebuilt.full_block_hashes)
            completions.append(
                LwdProbeCompletion(
                    inflight.request,
                    probe_nonce=prefix_nonce,
                    connection_id=connection,
                    result=result,
                    chain=chain,
                )
            )
        return completions

    def close(self) -> None:
        inflights = list(self._inflight.values())
        self._inflight.clear()
        runtime = self._runtime
        for inflight in inflights:
            if inflight.future is not None:
                inflight.future.cancel()
            if inflight.drain_task is not None and runtime is not None:
                runtime.cancel_task(inflight.drain_task)
        if runtime is not None:
            runtime.stop()
            self._runtime = None
