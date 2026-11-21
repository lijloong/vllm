# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
import threading
import time
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from vllm.config.lwd.topology import LwdConnectionId
from vllm.v1.lwd_control.control_edge_scheduler.lwd_edge_client import (
    LwdGatewayActivation,
)
from vllm.v1.lwd_control.control_edge_scheduler.lwd_edge_engine import (
    LwdEdgeEngineCore,
)
from vllm.v1.lwd_control.control_edge_scheduler.lwd_edge_probe import (
    LwdEdgeProbeManager,
    LwdProbeCompletion,
    LwdProbeSubmission,
)
from vllm.v1.lwd_control.control_edge_scheduler.lwd_probe_runtime import (
    LwdProbeRuntime,
)
from vllm.v1.lwd_control.lwd_comm.lwd_prefix import LwdPrefixManifest, LwdProbeResult


def _connection(edge_id: int, cloud_id: int, dp_idx: int) -> SimpleNamespace:
    return SimpleNamespace(
        connection_id=LwdConnectionId(edge_id, cloud_id, dp_idx),
        cloud=SimpleNamespace(
            coord_url=f"http://cloud{cloud_id}:{8100 + dp_idx}",
            coord_port=8100 + dp_idx,
            addr=f"cloud{cloud_id}",
        ),
    )


def _config(prefix_enabled: bool) -> SimpleNamespace:
    return SimpleNamespace(
        lwd_config=SimpleNamespace(
            connections=[_connection(0, 0, 0), _connection(0, 0, 1)],
            coordination=SimpleNamespace(
                enabled=True,
                placement_only=False,
                consumer_id=None,
                connect_timeout=1.0,
                probe_timeout=1.0,
                tenant_key_file=None,
            ),
            topology=SimpleNamespace(
                deployment=SimpleNamespace(scene="single_instance")
            ),
            instance_id=0,
        ),
        cache_config=SimpleNamespace(
            enable_prefix_caching=prefix_enabled,
            block_size=4,
        ),
        model_config=SimpleNamespace(served_model_name="test", model="test"),
    )


def _request() -> SimpleNamespace:
    return SimpleNamespace(
        request_id="req",
        client_index=0,
        skip_reading_prefix_cache=False,
        prompt_token_ids=list(range(16)),
        num_prompt_tokens=16,
        mm_features=[],
    )


class _Client:
    def __init__(
        self,
        target: LwdConnectionId,
        calls: list[tuple[int, int]],
        gateway_target: LwdConnectionId | None = None,
        advertise_cloud: bool = True,
        advertise_engine: bool = True,
        instance_id: str = "cloud-0",
    ) -> None:
        self.target = target
        self.gateway_target = gateway_target or target
        self.calls = calls
        self.advertise_cloud = advertise_cloud
        self.advertise_engine = advertise_engine
        self.instance_id = instance_id
        self.last_drain_task: asyncio.Task[None] | None = None

    def build_manifest(
        self, request_id: str, prompt_token_ids: list[int], media_items: tuple
    ) -> LwdPrefixManifest:
        return LwdPrefixManifest(
            request_id=request_id,
            prompt_tokens=len(prompt_token_ids),
            block_size=4,
            full_block_hashes=(bytes(32), bytes(32), bytes(32), bytes(32)),
            tail_hash=None,
        )

    async def activate(
        self, _session: object, request_id: str, _prompt: list[int], **kwargs
    ) -> LwdGatewayActivation:
        assert kwargs["edge_id"] == 0
        self.calls.append((self.target.cloud_id, self.target.dp_idx))
        hit_blocks = self.gateway_target.dp_idx + 1
        result = LwdProbeResult(
            request_id=request_id,
            instance_id=self.instance_id,
            block_size=4,
            hit_blocks=hit_blocks,
            hit_tokens=hit_blocks * 4,
            cloud_id=(self.gateway_target.cloud_id if self.advertise_cloud else None),
            engine_index=(
                self.gateway_target.dp_idx if self.advertise_engine else None
            ),
        )
        self.last_drain_task = asyncio.create_task(asyncio.sleep(60))
        return LwdGatewayActivation(result, 7, self.last_drain_task)


def _drain(manager: LwdEdgeProbeManager) -> LwdProbeCompletion:
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        completions = manager.drain()
        if completions:
            return completions[0]
        time.sleep(0.005)
    raise AssertionError("gateway activation did not arrive")


@pytest.mark.parametrize("prefix_enabled", [True, False])
def test_submit_opens_one_stream_and_binds_gateway_selected_dp(
    prefix_enabled: bool,
) -> None:
    manager = LwdEdgeProbeManager(_config(prefix_enabled), lambda: None)
    calls: list[tuple[int, int]] = []
    selected = LwdConnectionId(0, 0, 1)
    clients = {}
    for target in manager._targets:
        gateway_target = selected if target.dp_idx == 0 else target
        clients[target] = _Client(target, calls, gateway_target)
    manager._client = lambda target, _block_size: clients[target]
    try:
        assert manager.submit(_request()) is LwdProbeSubmission.SUBMITTED
        completion = _drain(manager)
        assert calls == [(0, 0)]
        assert completion.connection_id == selected
        assert completion.result is not None
        assert completion.result.engine_index == 1
        assert completion.probe_nonce == 7
        if prefix_enabled:
            assert completion.chain is not None
            assert len(completion.chain) == 4
        else:
            assert completion.chain is None

        manager.mark_admitted("req", completion.probe_nonce)
        assert not manager._inflight
    finally:
        manager.close()


def test_stale_activation_does_not_cancel_established_drain() -> None:
    manager = LwdEdgeProbeManager(_config(True), lambda: None)
    calls: list[tuple[int, int]] = []
    target = LwdConnectionId(0, 0, 0)
    client = _Client(target, calls)
    manager._client = lambda _target, _block_size: client
    resolve = manager._resolve_connection

    def abort_and_resolve(result: LwdProbeResult, edge_id: int):
        manager.abort("req")
        return resolve(result, edge_id)

    manager._resolve_connection = abort_and_resolve
    try:
        assert manager.submit(_request()) is LwdProbeSubmission.SUBMITTED
        deadline = time.monotonic() + 2
        while client.last_drain_task is None and time.monotonic() < deadline:
            time.sleep(0.005)
        assert client.last_drain_task is not None
        time.sleep(0.05)
        assert not client.last_drain_task.done()
        manager._runtime.cancel_task(client.last_drain_task)
    finally:
        manager.close()


def test_resolve_failure_does_not_cancel_established_drain() -> None:
    manager = LwdEdgeProbeManager(_config(True), lambda: None)
    calls: list[tuple[int, int]] = []
    target = LwdConnectionId(0, 0, 9)
    client = _Client(target, calls)
    manager._client = lambda _target, _block_size: client
    try:
        assert manager.submit(_request()) is LwdProbeSubmission.SUBMITTED
        completion = _drain(manager)
        assert completion.activation_error is not None
        assert client.last_drain_task is not None
        assert not client.last_drain_task.done()
        manager._runtime.cancel_task(client.last_drain_task)
    finally:
        manager.close()


def test_gateway_instance_id_must_match_cloud() -> None:
    manager = LwdEdgeProbeManager(_config(True), lambda: None)
    calls: list[tuple[int, int]] = []
    target = LwdConnectionId(0, 0, 0)
    client = _Client(target, calls, instance_id="cloud-1")
    manager._client = lambda _target, _block_size: client
    try:
        assert manager.submit(_request()) is LwdProbeSubmission.SUBMITTED
        completion = _drain(manager)
        assert completion.activation_error is not None
        assert "different instance" in completion.activation_error
    finally:
        manager.close()


def test_single_cloud_single_dp_infers_omitted_identity() -> None:
    config = _config(True)
    config.lwd_config.connections = [_connection(0, 0, 0)]
    manager = LwdEdgeProbeManager(config, lambda: None)
    calls: list[tuple[int, int]] = []
    target = LwdConnectionId(0, 0, 0)
    manager._client = lambda _target, _block_size: _Client(
        target, calls, advertise_cloud=False, advertise_engine=False
    )
    try:
        assert manager.submit(_request()) is LwdProbeSubmission.SUBMITTED
        completion = _drain(manager)
        assert completion.connection_id == target
        assert completion.result is not None
        assert completion.result.cloud_id == 0
        assert completion.result.engine_index == 0
        manager.mark_admitted("req", completion.probe_nonce)
    finally:
        manager.close()


def test_single_cloud_multi_dp_requires_engine_index() -> None:
    manager = LwdEdgeProbeManager(_config(True), lambda: None)
    calls: list[tuple[int, int]] = []
    target = LwdConnectionId(0, 0, 0)
    manager._client = lambda _target, _block_size: _Client(
        target, calls, advertise_engine=False
    )
    try:
        assert manager.submit(_request()) is LwdProbeSubmission.SUBMITTED
        completion = _drain(manager)
        assert completion.activation_error is not None
        assert "engine_index" in completion.activation_error
    finally:
        manager.close()


def test_multi_cloud_requires_cloud_id() -> None:
    config = _config(True)
    config.lwd_config.connections = [
        _connection(0, 0, 0),
        _connection(0, 1, 0),
    ]
    manager = LwdEdgeProbeManager(config, lambda: None)
    calls: list[tuple[int, int]] = []
    target = LwdConnectionId(0, 0, 0)
    manager._client = lambda _target, _block_size: _Client(
        target, calls, advertise_cloud=False
    )
    try:
        assert manager.submit(_request()) is LwdProbeSubmission.SUBMITTED
        completion = _drain(manager)
        assert completion.activation_error is not None
        assert "cloud_id" in completion.activation_error
    finally:
        manager.close()


def test_single_cloud_multi_dp_infers_cloud_id() -> None:
    manager = LwdEdgeProbeManager(_config(True), lambda: None)
    calls: list[tuple[int, int]] = []
    selected = LwdConnectionId(0, 0, 1)
    clients = {}
    for target in manager._targets:
        gateway_target = selected if target.dp_idx == 0 else target
        clients[target] = _Client(target, calls, gateway_target, advertise_cloud=False)
    manager._client = lambda target, _block_size: clients[target]
    try:
        assert manager.submit(_request()) is LwdProbeSubmission.SUBMITTED
        completion = _drain(manager)
        assert completion.connection_id == selected
        assert completion.result is not None
        assert completion.result.cloud_id == 0
        assert completion.result.engine_index == 1
        manager.mark_admitted("req", completion.probe_nonce)
    finally:
        manager.close()


def test_activate_style_submit_timeout_covers_semaphore_queue() -> None:
    runtime = LwdProbeRuntime(max_inflight=1)
    started = threading.Event()
    release = threading.Event()

    async def hold_permit() -> None:
        started.set()
        while not release.is_set():
            await asyncio.sleep(0.001)

    first = runtime.submit(hold_permit())
    assert started.wait(timeout=1)
    try:
        queued = runtime.submit(asyncio.sleep(0), timeout=0.01)
        with pytest.raises(TimeoutError):
            queued.result(timeout=1)
    finally:
        release.set()
        first.result(timeout=1)
        runtime.stop()


def test_bounded_cancellation_closes_unstarted_coroutine() -> None:
    async def run() -> None:
        runtime = object.__new__(LwdProbeRuntime)
        runtime._inflight = asyncio.Semaphore(0)
        coroutine = Mock()
        task = asyncio.create_task(runtime._bounded(coroutine))
        await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        coroutine.close.assert_called_once_with()

    asyncio.run(run())


def test_submit_failure_cleans_inflight_and_fails_closed() -> None:
    class _StoppedRuntime:
        def submit(self, _coroutine, timeout: float):
            raise RuntimeError("event loop stopped")

        def stop(self) -> None:
            pass

    manager = LwdEdgeProbeManager(_config(True), lambda: None)
    manager._runtime = _StoppedRuntime()

    assert manager.submit(_request()) is LwdProbeSubmission.FAILED
    assert not manager._inflight
    manager.close()


def test_abort_does_not_disconnect_established_usage_drain() -> None:
    manager = LwdEdgeProbeManager(_config(True), lambda: None)
    calls: list[tuple[int, int]] = []
    clients = {target: _Client(target, calls) for target in manager._targets}
    manager._client = lambda target, _block_size: clients[target]
    try:
        assert manager.submit(_request()) is LwdProbeSubmission.SUBMITTED
        _drain(manager)
        task = manager._inflight["req"].drain_task
        manager.abort("req")
        time.sleep(0.01)
        assert not task.done()
        assert not manager._inflight
        manager._runtime.cancel_task(task)
    finally:
        manager.close()


def test_activation_failure_aborts_without_admitting() -> None:
    request = SimpleNamespace(request_id="r")
    completion = LwdProbeCompletion(request, activation_error="startup missing")
    engine = object.__new__(LwdEdgeEngineCore)
    engine._lwd_probe_manager = Mock()
    engine._lwd_probe_manager.drain.return_value = [completion]
    engine._lwd_abort_internally = Mock()
    engine._lwd_add_request_after_probe = Mock()

    engine._lwd_drain_probe_completions()

    engine._lwd_abort_internally.assert_called_once_with(request)
    engine._lwd_add_request_after_probe.assert_not_called()
    assert request.lwd_prefix_nonce == 0


def test_admit_marks_usage_drain_only_after_request_notify() -> None:
    calls = []
    connection = LwdConnectionId(0, 0, 0)
    request = SimpleNamespace(request_id="r", lwd_prefix_nonce=7)
    engine = object.__new__(LwdEdgeEngineCore)
    engine.scheduler = SimpleNamespace(lwd_admit_request=Mock(return_value=connection))
    engine._lwd_coord_clients = {}
    engine._lwd_notify_request = lambda _dp, _request: calls.append("notify")
    engine._lwd_probe_manager = Mock()
    engine._lwd_probe_manager.mark_admitted.side_effect = lambda _rid, _nonce: (
        calls.append("mark")
    )

    engine._lwd_admit_and_notify(request, connection)

    assert calls == ["notify", "mark"]
    engine._lwd_probe_manager.mark_admitted.assert_called_once_with("r", 7)


def test_admit_validation_failure_releases_unadmitted_activation() -> None:
    request = SimpleNamespace(request_id="r", lwd_prefix_nonce=7)
    engine = object.__new__(LwdEdgeEngineCore)
    engine.scheduler = SimpleNamespace(
        lwd_admit_request=Mock(side_effect=ValueError("invalid request")),
        lwd_edge_cancel_admission=Mock(),
    )
    engine._lwd_coord_clients = {}
    engine._lwd_notify_request = Mock()
    engine._lwd_probe_manager = Mock()

    with pytest.raises(ValueError, match="invalid request"):
        engine._lwd_admit_and_notify(request)

    engine._lwd_probe_manager.abort.assert_called_once_with("r")
    engine._lwd_notify_request.assert_not_called()
    engine._lwd_probe_manager.mark_admitted.assert_not_called()
    engine.scheduler.lwd_edge_cancel_admission.assert_called_once_with("r")


def test_admit_failure_releases_unadmitted_activation() -> None:
    connection = LwdConnectionId(0, 0, 0)
    request = SimpleNamespace(request_id="r", lwd_prefix_nonce=7)
    engine = object.__new__(LwdEdgeEngineCore)
    engine.scheduler = SimpleNamespace(
        lwd_admit_request=Mock(return_value=connection),
        lwd_edge_cancel_admission=Mock(),
    )
    engine._lwd_coord_clients = {}
    engine._lwd_notify_request = Mock(side_effect=RuntimeError("notify failed"))
    engine._lwd_probe_manager = Mock()

    with pytest.raises(RuntimeError, match="notify failed"):
        engine._lwd_admit_and_notify(request, connection)

    engine._lwd_probe_manager.abort.assert_called_once_with("r")
    engine._lwd_probe_manager.mark_admitted.assert_not_called()
    engine.scheduler.lwd_edge_cancel_admission.assert_called_once_with("r")
