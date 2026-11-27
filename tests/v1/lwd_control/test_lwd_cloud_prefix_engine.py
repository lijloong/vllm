# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import queue
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from vllm.config.lwd import LwdConnectionId
from vllm.v1.core.kv_cache_manager import KVCacheBlocks, KVCacheManager
from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.engine import EngineCoreRequestType
from vllm.v1.lwd_control.control_cloud_scheduler.lwd_cloud_engine import (
    LwdCloudEngineCoreMixin,
)
from vllm.v1.lwd_control.control_cloud_scheduler.lwd_cloud_ledger import (
    LwdCloudLedger,
)
from vllm.v1.lwd_control.control_scheduler.lwd_base_engine import LwdBaseEngineCore
from vllm.v1.lwd_control.lwd_comm.lwd_link_protocol import LwdPeer
from vllm.v1.lwd_control.lwd_comm.lwd_messages import (
    LwdAbortNotify,
    LwdRequestNotify,
)
from vllm.v1.request import RequestStatus

_CONNECTION = LwdConnectionId(0, 0, 0)


def _cloud_engine() -> LwdCloudEngineCoreMixin:
    engine = object.__new__(LwdCloudEngineCoreMixin)
    engine._lwd_prefix_enabled = True
    # Retirement (abort) touches these two scheduler maps; a bare
    # SimpleNamespace would raise AttributeError inside retire_request.
    engine.scheduler = SimpleNamespace(lwd_recv_rows={}, pending_c2e=[])
    engine._lwd_ledger = LwdCloudLedger(engine.scheduler)
    # A legitimate ADD always follows a RequestNotify that registered the route.
    engine._lwd_ledger.request_routes["r"] = object()
    engine._lwd_send_prefix_ready = Mock()
    engine._lwd_engine_index = 0
    engine.cloud_control_event_queue = queue.Queue()
    return engine


def test_add_request_reserves_before_native_add_then_replies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = _cloud_engine()
    reservations = Mock()
    reservations.reserve.return_value = 32
    engine.scheduler.lwd_prefix_reservations = reservations
    request = SimpleNamespace(
        request_id="r", lwd_prefix_nonce=7, lwd_wire_hashes_adopted=True
    )
    added = []
    monkeypatch.setattr(
        LwdBaseEngineCore, "add_request", lambda _self, req, wave=0: added.append(req)
    )

    engine.add_request(request)

    reservations.reserve.assert_called_once_with(request)
    assert added == [request]
    engine._lwd_send_prefix_ready.assert_called_once_with("r", 32, 7)
    assert engine.cloud_control_event_queue.get_nowait() == {
        "type": "usage_start",
        "request_id": "r",
        "engine_index": 0,
        "prefix_nonce": 7,
    }


def test_add_request_without_prefix_cache_uses_native_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = _cloud_engine()
    engine._lwd_prefix_enabled = False
    request = SimpleNamespace(
        request_id="r", lwd_prefix_nonce=7, lwd_wire_hashes_adopted=False
    )
    added = []
    monkeypatch.setattr(
        LwdBaseEngineCore, "add_request", lambda _self, req, wave=0: added.append(req)
    )

    engine.add_request(request)

    assert added == [request]
    engine._lwd_send_prefix_ready.assert_not_called()
    assert engine.cloud_control_event_queue.get_nowait()["prefix_nonce"] == 7


def test_add_request_without_prefix_cache_does_not_start_failed_add(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = _cloud_engine()
    engine._lwd_prefix_enabled = False
    request = SimpleNamespace(request_id="r", lwd_prefix_nonce=7)
    monkeypatch.setattr(
        LwdBaseEngineCore,
        "add_request",
        Mock(side_effect=RuntimeError("native add failed")),
    )

    with pytest.raises(RuntimeError, match="native add failed"):
        engine.add_request(request)
    assert engine.cloud_control_event_queue.empty()


def test_coordination_poll_wakes_engine_for_remaining_commands() -> None:
    engine = object.__new__(LwdCloudEngineCoreMixin)
    engine._lwd_probe_query = Mock()
    processor = Mock()
    processor.poll.return_value = 32
    engine.cloud_control_processor = processor
    engine._lwd_control_processor = processor
    engine._lwd_coord_pending = queue.Queue()
    engine._lwd_coord_pending.put({"type": "probe"})
    engine.input_queue = queue.Queue()

    engine._lwd_poll_coordination()

    request_type, payload = engine.input_queue.get_nowait()
    assert request_type.name == "WAKEUP"
    assert payload is None


def test_reserve_failure_replies_error_without_native_add(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = _cloud_engine()
    reservations = Mock()
    reservations.reserve.side_effect = RuntimeError("no blocks")
    engine.scheduler.lwd_prefix_reservations = reservations
    request = SimpleNamespace(
        request_id="r", lwd_prefix_nonce=7, lwd_wire_hashes_adopted=True
    )
    native_add = Mock()
    monkeypatch.setattr(LwdBaseEngineCore, "add_request", native_add)

    engine.add_request(request)

    native_add.assert_not_called()
    engine._lwd_send_prefix_ready.assert_called_once_with("r", 0, 7, error="no blocks")
    assert engine.cloud_control_event_queue.empty()


def test_active_duplicate_replies_error_without_native_add(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = _cloud_engine()
    engine.scheduler.requests = {"r": object()}
    engine.scheduler.lwd_prefix_reservations = Mock()
    request = SimpleNamespace(
        request_id="r", lwd_prefix_nonce=7, lwd_wire_hashes_adopted=True
    )
    native_add = Mock()
    monkeypatch.setattr(LwdBaseEngineCore, "add_request", native_add)

    engine.add_request(request)

    native_add.assert_not_called()
    engine.scheduler.lwd_prefix_reservations.reserve.assert_not_called()
    engine._lwd_send_prefix_ready.assert_called_once_with(
        "r", 0, 7, error="duplicate active request id"
    )


def test_request_build_adopts_wire_hashes_and_skip_reading() -> None:
    engine = object.__new__(LwdCloudEngineCoreMixin)
    engine.request_block_hasher = Mock(return_value=[])
    engine._lwd_hash_block_size = 2
    engine.vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(
            get_hidden_size=lambda: 8,
            dtype=torch.float32,
        )
    )
    notify = LwdRequestNotify(
        "r",
        4,
        block_hashes=[b"h0", b"h1"],
        prefix_nonce=7,
        skip_reading_prefix_cache=True,
    )

    request = engine._lwd_build_request(notify)

    assert request.block_hashes == [b"h0", b"h1"]
    assert request.lwd_wire_hashes_adopted
    assert request.lwd_prefix_nonce == 7
    assert request.skip_reading_prefix_cache


@pytest.mark.parametrize("wire_hashes", [[], [b"h0"]])
def test_request_build_never_hashes_meta_embeds_on_wire_chain_mismatch(
    wire_hashes: list[bytes],
) -> None:
    """A missing/partial edge chain must not fall back to local hashing.

    The cloud request owns a meta prompt-embeds placeholder, so the local
    hasher raises ``Cannot copy out of meta tensor`` (seen under concurrent
    gateway/probe requests whose chain length differs from the cloud block
    count).  Admission must fail closed through ``lwd_wire_hashes_adopted``
    instead.
    """
    engine = object.__new__(LwdCloudEngineCoreMixin)
    engine.request_block_hasher = Mock(
        side_effect=AssertionError("local hasher read placeholder input")
    )
    engine._lwd_hash_block_size = 2
    engine.vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(
            get_hidden_size=lambda: 8,
            dtype=torch.float32,
        )
    )
    notify = LwdRequestNotify(
        "r",
        4,
        block_hashes=wire_hashes,
        prefix_nonce=7,
    )

    request = engine._lwd_build_request(notify)

    assert request.block_hashes == wire_hashes
    assert not request.lwd_wire_hashes_adopted
    engine.request_block_hasher.assert_not_called()


@pytest.mark.parametrize("prompt_tokens", [3, 4])
def test_decode_never_hashes_or_caches_output_blocks(prompt_tokens: int) -> None:
    engine = object.__new__(LwdCloudEngineCoreMixin)
    engine.request_block_hasher = Mock(
        side_effect=AssertionError("local hasher read placeholder input")
    )
    engine._lwd_hash_block_size = 2
    engine.vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(
            get_hidden_size=lambda: 8,
            dtype=torch.float32,
        )
    )
    wire_hashes = [f"h{i}".encode() for i in range(prompt_tokens // 2)]
    request = engine._lwd_build_request(
        LwdRequestNotify(
            "r",
            prompt_tokens,
            block_hashes=wire_hashes,
            prefix_nonce=7,
        )
    )
    assert request.block_hashes == wire_hashes
    assert request.shared_prefix_boundary == 0

    request.append_output_token_ids([11, 12, 13])

    assert request.num_output_tokens == 3
    assert request.block_hashes == wire_hashes
    engine.request_block_hasher.assert_not_called()

    coordinator = Mock()
    coordinator.get_num_blocks_to_allocate.return_value = 0
    coordinator.allocate_new_blocks.return_value = ([],)
    manager = KVCacheManager.__new__(KVCacheManager)
    manager.max_model_len = 100
    manager.lwd_prefix_caching = True
    manager.retained_hit_group_ids = ()
    manager.empty_kv_cache_blocks = KVCacheBlocks(((),))
    manager.coordinator = coordinator
    manager.block_pool = SimpleNamespace(get_num_free_blocks=lambda: 100)
    manager.watermark_blocks = 0

    new_blocks = manager.allocate_slots(
        request,
        request.num_tokens,
        has_scheduled_reqs=False,
    )

    assert new_blocks is not None
    coordinator.cache_blocks.assert_called_once_with(request, prompt_tokens)


@pytest.mark.parametrize("prompt_tokens", [3, 4])
def test_async_output_update_never_hashes_or_caches_output_blocks(
    prompt_tokens: int,
) -> None:
    engine = object.__new__(LwdCloudEngineCoreMixin)
    engine.request_block_hasher = Mock(
        side_effect=AssertionError("local hasher read placeholder input")
    )
    engine._lwd_hash_block_size = 2
    engine.vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(
            get_hidden_size=lambda: 8,
            dtype=torch.float32,
        )
    )
    wire_hashes = [f"h{i}".encode() for i in range(prompt_tokens // 2)]
    request = engine._lwd_build_request(
        LwdRequestNotify("r", prompt_tokens, block_hashes=wire_hashes)
    )
    request.status = RequestStatus.RUNNING
    output_token_ids = [11, 12, 13]
    request.num_computed_tokens = request.num_tokens + len(output_token_ids)
    request.num_output_placeholders = len(output_token_ids)

    manager = KVCacheManager.__new__(KVCacheManager)
    manager.enable_caching = True
    manager.lwd_prefix_caching = True
    manager.coordinator = Mock()
    scheduler = object.__new__(AsyncScheduler)
    scheduler.max_model_len = 100
    scheduler.kv_cache_manager = manager

    new_token_ids, stopped = scheduler._update_request_with_output(
        request, output_token_ids
    )

    assert new_token_ids == output_token_ids
    assert not stopped
    assert request.block_hashes == wire_hashes
    engine.request_block_hasher.assert_not_called()
    manager.coordinator.cache_blocks.assert_called_once_with(request, prompt_tokens)


@pytest.mark.parametrize(
    ("done", "expected_args"),
    [
        (("refill_done", "r", 0), ("r", 0)),
        (("refill_done", "r", 7), ("r", 7)),
    ],
)
def test_refill_done_bypasses_watermark_recording(
    monkeypatch: pytest.MonkeyPatch,
    done: tuple[object, ...],
    expected_args: tuple[object, ...],
) -> None:
    class DoneMq:
        def __init__(self) -> None:
            self.items = [done]

        def dequeue(self, timeout: int) -> tuple[object, ...]:
            if self.items:
                return self.items.pop(0)
            raise TimeoutError

    engine = object.__new__(LwdCloudEngineCoreMixin)
    engine._lwd_done_mqs = {_CONNECTION: DoneMq()}
    engine.scheduler = SimpleNamespace(lwd_refill_done=Mock())
    from vllm_ascend.worker.lwd_cloud_model import lwd_pre_recv

    record = Mock()
    monkeypatch.setattr(lwd_pre_recv, "record_irecv_completions", record)

    engine._lwd_drain_irecv_completions()

    engine.scheduler.lwd_refill_done.assert_called_once_with(*expected_args)
    record.assert_not_called()


def test_send_refill_carries_generation() -> None:
    engine = object.__new__(LwdCloudEngineCoreMixin)
    peer = object()
    engine._lwd_ledger = LwdCloudLedger(SimpleNamespace())
    engine._lwd_ledger.request_routes = {"r": peer}
    engine._lwd_send_until_queued = Mock()

    engine._lwd_send_refill("r", 16, 32, 7)

    actual_peer, notify = engine._lwd_send_until_queued.call_args.args
    assert actual_peer is peer
    assert (notify.request_id, notify.prefix_nonce) == ("r", 7)


def test_prefix_with_kv_connector_fails_before_startup() -> None:
    config = SimpleNamespace(
        cache_config=SimpleNamespace(enable_prefix_caching=True),
        kv_transfer_config=SimpleNamespace(),
    )

    with pytest.raises(ValueError, match="external hits bypass.*READY"):
        LwdCloudEngineCoreMixin(vllm_config=config, lwd_dp_size=1)


def test_stale_add_after_abort_is_dropped_without_scheduling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Abort retiring the generation before the ADD is consumed must not
    resurrect the request: no native add, no READY, no reservation, no raise."""
    engine = _cloud_engine()
    engine._lwd_ledger.request_routes.pop("r")
    engine._lwd_ledger.retired_routes["r"] = object()
    engine._lwd_ledger.retire_pending_add("r", 7)
    reservations = Mock()
    engine.scheduler.lwd_prefix_reservations = reservations
    request = SimpleNamespace(
        request_id="r", lwd_prefix_nonce=7, lwd_wire_hashes_adopted=True
    )
    native_add = Mock()
    monkeypatch.setattr(LwdBaseEngineCore, "add_request", native_add)

    engine.add_request(request)

    native_add.assert_not_called()
    reservations.reserve.assert_not_called()
    engine._lwd_send_prefix_ready.assert_not_called()
    assert engine.cloud_control_event_queue.empty()
    # The tombstone is consumed exactly once.
    assert engine._lwd_ledger.retired_pending_add == {}


def test_add_without_route_or_tombstone_fails_fast() -> None:
    """An ADD that never followed RequestNotify is a wiring bug, not a race."""
    engine = _cloud_engine()
    engine._lwd_ledger.request_routes.pop("r")
    request = SimpleNamespace(
        request_id="r", lwd_prefix_nonce=7, lwd_wire_hashes_adopted=True
    )

    with pytest.raises(RuntimeError, match="ADD without a request route"):
        engine.add_request(request)


def test_new_generation_after_abort_is_admitted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A different nonce is a fresh generation and must still be admitted."""
    engine = _cloud_engine()
    reservations = Mock()
    reservations.reserve.return_value = 32
    engine.scheduler.lwd_prefix_reservations = reservations
    engine._lwd_ledger.retire_pending_add("r", 6)
    request = SimpleNamespace(
        request_id="r", lwd_prefix_nonce=7, lwd_wire_hashes_adopted=True
    )
    added = []
    monkeypatch.setattr(
        LwdBaseEngineCore, "add_request", lambda _self, req, wave=0: added.append(req)
    )

    engine.add_request(request)

    assert added == [request]
    engine._lwd_send_prefix_ready.assert_called_once_with("r", 32, 7)
    # A fresh generation releases the stale tombstone on its way in.
    assert engine._lwd_ledger.retired_pending_add == {}


def test_send_prefix_ready_without_route_does_not_raise() -> None:
    engine = object.__new__(LwdCloudEngineCoreMixin)
    engine._lwd_ledger = LwdCloudLedger(SimpleNamespace())

    engine._lwd_send_prefix_ready("gone", 0, 7, error="boom")


def test_send_refill_without_route_does_not_raise() -> None:
    engine = object.__new__(LwdCloudEngineCoreMixin)
    engine._lwd_ledger = LwdCloudLedger(SimpleNamespace())

    engine._lwd_send_refill("gone", 16, 32, 7)


def test_abort_before_queued_add_drops_stale_generation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """End-to-end of the race: RequestNotify registered the route, AbortNotify
    is processed first, then the queued ADD must be dropped silently."""
    engine = _cloud_engine()
    engine.scheduler.requests = {}
    engine.scheduler._lwd_refill_gate = set()
    engine.scheduler._lwd_refill_pending = {}
    engine.scheduler.lwd_prefix_reservations = Mock()
    engine.aborts_queue = queue.Queue()
    engine.input_queue = queue.Queue()
    engine._lwd_ledger.request_nonces["r"] = 7
    request = SimpleNamespace(
        request_id="r", lwd_prefix_nonce=7, lwd_wire_hashes_adopted=True
    )
    native_add = Mock()
    monkeypatch.setattr(LwdBaseEngineCore, "add_request", native_add)

    engine._lwd_handle_abort_notify(LwdAbortNotify("r", prefix_nonce=7))
    assert engine._lwd_ledger.retired_pending_add == {"r": 7}

    engine.add_request(request)

    native_add.assert_not_called()
    engine._lwd_send_prefix_ready.assert_not_called()
    assert engine._lwd_ledger.retired_pending_add == {}


def test_abort_overtakes_queued_add(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """FIFO-faithful race reproduction.

    RequestNotify and AbortNotify both land on the IO thread first. Processing
    RequestNotify appends its ADD to the queue tail, so the abort is consumed
    before that ADD; the abort retires the route and the late ADD must be
    dropped silently instead of resurrecting the cancelled generation.
    """
    engine = object.__new__(LwdCloudEngineCoreMixin)
    engine._lwd_prefix_enabled = True
    engine.lwd_config = SimpleNamespace(instance_id=0)
    engine.engines_running = False
    engine.scheduler = SimpleNamespace(
        requests={},
        _lwd_refill_gate=set(),
        _lwd_refill_pending={},
        lwd_recv_rows={},
        pending_c2e=[],
        lwd_prefix_reservations=Mock(),
    )
    engine.scheduler.lwd_prefix_reservations.reserve.return_value = 32
    engine._lwd_ledger = LwdCloudLedger(engine.scheduler)
    engine._lwd_engine_index = 0
    engine.cloud_control_event_queue = queue.Queue()
    engine.aborts_queue = queue.Queue()
    engine.input_queue = queue.Queue()
    engine._lwd_send_prefix_ready = Mock()
    # This case pins queue ordering and route retirement, not request building.
    monkeypatch.setattr(
        LwdCloudEngineCoreMixin,
        "_lwd_build_request",
        lambda _self, notify: SimpleNamespace(
            request_id=notify.request_id,
            lwd_prefix_nonce=notify.prefix_nonce,
            lwd_wire_hashes_adopted=True,
            use_structured_output=False,
        ),
    )
    added: list[object] = []
    monkeypatch.setattr(
        LwdBaseEngineCore,
        "add_request",
        lambda _self, req, wave=0: added.append(req),
    )

    peer = LwdPeer(0, 0)
    engine.input_queue.put_nowait(
        (EngineCoreRequestType.WAKEUP, (peer, LwdRequestNotify("r", 4, prefix_nonce=7)))
    )
    engine.input_queue.put_nowait(
        (EngineCoreRequestType.WAKEUP, (peer, LwdAbortNotify("r", prefix_nonce=7)))
    )

    # Drain in FIFO order exactly as the engine thread would: RequestNotify
    # appends its ADD behind the queued abort, so the abort runs first.
    while not engine.input_queue.empty():
        request_type, payload = engine.input_queue.get_nowait()
        if request_type == EngineCoreRequestType.WAKEUP:
            engine._lwd_handle_edge_message(*payload)
        elif request_type == EngineCoreRequestType.ADD:
            engine.add_request(payload[0], payload[1])

    assert added == []
    engine._lwd_send_prefix_ready.assert_not_called()
    assert engine._lwd_ledger.retired_pending_add == {}
