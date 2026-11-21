import errno
import logging
from collections import Counter
from types import SimpleNamespace

import pytest
import torch

from miles.backends.megatron_utils import optimizer_state_numa as numa


class Policy:
    def __init__(self):
        self.bound = False
        self.denied = False
        self.pages = Counter()

    def bind(self, node):
        assert node == 1
        if self.denied:
            raise OSError(errno.EPERM, "denied")
        self.bound = True
        return "previous"

    def restore(self, previous):
        assert previous == "previous"
        self.bound = False

    def page_bytes(self, address, nbytes):
        return self.pages


def test_visible_pci_address_selects_the_numa_node(tmp_path):
    device = SimpleNamespace(pci_domain_id=0, pci_bus_id=73, pci_device_id=0)
    pci = tmp_path / "0000:49:00.0"
    pci.mkdir()
    (pci / "numa_node").write_text("2\n")
    assert numa._gpu_node(device, tmp_path) == 2


def test_headroom_counts_only_free_and_reclaimable_file_pages(tmp_path):
    node = tmp_path / "node1"
    node.mkdir()
    info = node / "meminfo"
    info.write_text(
        "Node 1 MemFree: 12 kB\nNode 1 Active(file): 7 kB\nNode 1 Inactive(file): 3 kB\nNode 1 Shmem: 99 kB\n"
    )
    assert numa._available_bytes(1, tmp_path) == 22 * 1024
    info.write_text("Node 1 MemFree: 12 kB\n")
    with pytest.raises(ValueError, match="Incomplete"):
        numa._available_bytes(1, tmp_path)


@pytest.mark.parametrize(
    "mode,node,available,denied,expected_bound",
    [
        ("bind", 1, numa._HEADROOM_BYTES + 16, False, True),
        ("bind", 1, numa._HEADROOM_BYTES + 15, False, False),
        ("bind", 1, numa._HEADROOM_BYTES + 16, True, False),
        ("bind", -1, 0, False, False),
        ("report", 1, numa._HEADROOM_BYTES + 16, False, False),
    ],
)
def test_binding_scoped_to_allocation(monkeypatch, mode, node, available, denied, expected_bound):
    policy = Policy()
    policy.denied = denied
    monkeypatch.setattr(numa, "_available_bytes", lambda node: available)
    placement = numa.OptimizerStatePlacement(mode, node, policy)

    def allocate(tensor):
        assert policy.bound == expected_bound
        return torch.empty_like(tensor)

    tensor = torch.ones(3)  # 12 bytes, rounded up to a 16-byte allocation.
    assert placement.wrap_allocator(allocate)(tensor).shape == tensor.shape
    assert not policy.bound


def test_allocation_error_restores_thread_policy(monkeypatch):
    policy = Policy()
    placement = numa.OptimizerStatePlacement("bind", 1, policy)
    monkeypatch.setattr(numa, "_available_bytes", lambda node: 2 * numa._HEADROOM_BYTES)

    def fail(tensor):
        assert policy.bound
        raise RuntimeError("allocation failed")

    with pytest.raises(RuntimeError, match="allocation failed"):
        placement.wrap_allocator(fail)(torch.ones(4))
    assert not policy.bound


@pytest.mark.parametrize(
    "node,counts,expected",
    [
        (1, {1: 8, 0: 8}, "50.0%"),
        (1, {0: 16}, "0.0%"),
        (-1, {1: 16}, "unknown"),
        (1, {1: 8, -errno.ENOENT: 8}, "unknown"),
    ],
)
def test_sampled_local_share(caplog, node, counts, expected):
    policy = Policy()
    policy.pages.update(counts)
    placement = numa.OptimizerStatePlacement("report", node, policy)
    buffer = placement.wrap_allocator(torch.empty_like)(torch.ones(4))
    assert buffer.nbytes == 16
    with caplog.at_level(logging.INFO, logger=numa.__name__):
        placement.report()
        placement.report()
    (record,) = caplog.records
    assert record.message.endswith(f"local share {expected}")
    unknown = counts.get(-errno.ENOENT, 0)
    assert f"unknown bytes {unknown}," in record.message


def test_denied_query_or_no_live_buffers_has_unknown_share(caplog):
    policy = Policy()

    def denied(address, nbytes):
        raise OSError(errno.EPERM, "denied")

    policy.page_bytes = denied
    placement = numa.OptimizerStatePlacement("report", 1, policy)
    empty = numa.OptimizerStatePlacement("report", 1, policy)
    buffer = placement.wrap_allocator(torch.empty_like)(torch.ones(4))
    with caplog.at_level(logging.INFO, logger=numa.__name__):
        placement.report()
        empty.report()
    assert f"unknown bytes {buffer.nbytes}, local share unknown" in caplog.records[0].message
    assert "unknown bytes 0, local share unknown" in caplog.records[1].message
