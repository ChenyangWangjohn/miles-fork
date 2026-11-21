"""Linux NUMA placement for Megatron's pinned chunked optimizer-state buffers."""

from __future__ import annotations

import ctypes
import functools
import logging
import os
import sys
import weakref
from collections import Counter
from contextlib import contextmanager
from pathlib import Path

import torch

logger = logging.getLogger(__name__)

_MPOL_BIND = 2
_SAMPLE_BYTES = 2 << 20
# Multiple ranks on a socket may observe the same free and reclaimable memory concurrently.
_HEADROOM_BYTES = 16 << 30


def _gpu_node(properties, pci_root=Path("/sys/bus/pci/devices")) -> int:
    # CUDA returns the PCI address of the visible device, including under CUDA_VISIBLE_DEVICES.
    address = f"{properties.pci_domain_id:04x}:{properties.pci_bus_id:02x}:{properties.pci_device_id:02x}.0"
    return int((pci_root / address / "numa_node").read_text())


def _available_bytes(node: int, root=Path("/sys/devices/system/node")) -> int:
    fields = {"MemFree:", "Active(file):", "Inactive(file):"}
    values = {}
    for line in (root / f"node{node}" / "meminfo").read_text().splitlines():
        parts = line.split()
        if len(parts) == 5 and parts[2] in fields and parts[4] == "kB":
            values[parts[2]] = int(parts[3]) * 1024
    if fields != values.keys():
        raise ValueError(f"Incomplete NUMA meminfo for node {node}")
    return sum(values.values())


class _LinuxPolicy:
    """Read and restore the allocating thread's policy through libnuma."""

    def __init__(self):
        self.lib = ctypes.CDLL("libnuma.so.1", use_errno=True)
        bits = self.lib.numa_num_possible_nodes()
        if bits <= 0:
            raise OSError("libnuma could not determine the node mask size")
        self.word_bits = 8 * ctypes.sizeof(ctypes.c_ulong)
        self.mask = ctypes.c_ulong * ((bits + self.word_bits - 1) // self.word_bits)
        self.maxnode = ctypes.c_ulong(ctypes.sizeof(self.mask) * 8 + 1)

    @staticmethod
    def _checked(result):
        if result < 0:
            code = ctypes.get_errno()
            raise OSError(code, os.strerror(code))

    def bind(self, node):
        if not 0 <= node < ctypes.sizeof(self.mask) * 8:
            raise ValueError(f"NUMA node {node} is outside the kernel node mask")
        previous_mode, previous_mask = ctypes.c_int(), self.mask()
        self._checked(self.lib.get_mempolicy(ctypes.byref(previous_mode), previous_mask, self.maxnode, None, 0))
        mask = self.mask()
        mask[node // self.word_bits] = 1 << (node % self.word_bits)
        self._checked(self.lib.set_mempolicy(_MPOL_BIND, mask, self.maxnode))
        return previous_mode.value, previous_mask

    def restore(self, previous):
        mode, mask = previous
        self._checked(self.lib.set_mempolicy(mode, mask, self.maxnode))

    def page_bytes(self, address: int, nbytes: int) -> Counter:
        offsets = range(0, nbytes, _SAMPLE_BYTES)
        pages = (ctypes.c_void_p * len(offsets))(*(address + offset for offset in offsets))
        statuses = (ctypes.c_int * len(offsets))()
        self._checked(self.lib.move_pages(0, ctypes.c_ulong(len(pages)), pages, None, statuses, 0))
        sampled = Counter()
        for offset, node in zip(offsets, statuses, strict=True):
            sampled[node] += min(_SAMPLE_BYTES, nbytes - offset)
        return sampled


class OptimizerStatePlacement:
    def __init__(self, mode, node, policy):
        self.mode = mode
        self.node = node
        self.policy = policy
        self._buffers = []
        self._warned = False

    @contextmanager
    def _allocation(self, nbytes):
        previous = None
        if self.mode == "bind" and self.node >= 0:
            rounded = 1 << max(nbytes - 1, 0).bit_length()
            try:
                if _available_bytes(self.node) >= rounded + _HEADROOM_BYTES:
                    previous = self.policy.bind(self.node)
                elif not self._warned:
                    logger.warning("NUMA node %s lacks headroom for an optimizer-state buffer", self.node)
                    self._warned = True
            except (OSError, ValueError) as exc:
                if not self._warned:
                    logger.warning("Allocating optimizer-state buffer with the existing NUMA policy: %s", exc)
                    self._warned = True
        try:
            yield
        finally:
            if previous is not None:
                self.policy.restore(previous)

    def wrap_allocator(self, allocate):
        @functools.wraps(allocate)
        def wrapped(tensor):
            with self._allocation(tensor.nbytes):
                buffer = allocate(tensor)
            if self._buffers is not None:
                self._buffers.append(weakref.ref(buffer))
            return buffer

        return wrapped

    def report(self):
        """Sample live buffers once after the first train step; keep no strong references."""
        if self._buffers is None:
            return
        refs, self._buffers = self._buffers, None
        pages = Counter()
        unknown = 0
        for ref in refs:
            buffer = ref()
            if buffer is None or not buffer.nbytes:
                continue
            try:
                for node, count in self.policy.page_bytes(buffer.data_ptr(), buffer.nbytes).items():
                    if node < 0:  # A negative move_pages status is a per-page errno.
                        unknown += count
                    else:
                        pages[node] += count
            except OSError:
                unknown += buffer.nbytes
        total = sum(pages.values()) + unknown
        share = f"{pages[self.node] / total:.1%}" if self.node >= 0 and total and not unknown else "unknown"
        logger.info(
            "Optimizer-state NUMA placement: target node %s, sampled bytes by node %s, unknown bytes %s, local share %s",
            self.node,
            dict(pages),
            unknown,
            share,
        )


def install_optimizer_state_numa(mode: str) -> OptimizerStatePlacement | None:
    """Install at actor initialization, before constructing the chunked optimizer or restoring a checkpoint."""
    if sys.platform != "linux" or not torch.cuda.is_available() or torch.version.hip is not None:
        logger.warning("Optimizer-state NUMA placement requires Linux and CUDA; using the existing allocator")
        return None

    # Optional Megatron API: importing the actor must still work with versions that lack chunked offload.
    from megatron.core.optimizer.cpu_offloading.chunked_optimizer_state_offload import ChunkedOptimizerStateOffloader

    current = ChunkedOptimizerStateOffloader._new_cpu_buffer
    placement = getattr(current, "_miles_optimizer_state_placement", None)
    if placement is not None:
        return placement
    try:
        node = _gpu_node(torch.cuda.get_device_properties(torch.cuda.current_device()))
        policy = _LinuxPolicy()
    except (AttributeError, OSError, ValueError) as exc:
        logger.warning("Cannot determine optimizer-state NUMA placement: %s; using the existing allocator", exc)
        return None
    if node < 0:
        logger.warning("GPU has no NUMA node; optimizer-state buffers use the existing allocator")
    placement = OptimizerStatePlacement(mode, node, policy)
    allocate = placement.wrap_allocator(current)
    allocate._miles_optimizer_state_placement = placement
    ChunkedOptimizerStateOffloader._new_cpu_buffer = staticmethod(allocate)
    return placement
