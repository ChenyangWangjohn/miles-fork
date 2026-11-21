"""Compare optimizer updates in fresh processes so each starts with a cold pinned allocator."""

import hashlib
import json
import subprocess
import sys

import pytest
import torch
from tests.ci.ci_register import register_cuda_ci

from miles.backends.megatron_utils import optimizer_state_numa as numa

register_cuda_ci(est_time=60, suite="stage-b-2-gpu-h200", labels=["megatron"], hardware=["hopper", "blackwell"])


def _worker(mode):
    from megatron.core.optimizer.cpu_offloading.chunked_optimizer_state_offload import ChunkedOptimizerStateOffloader

    if mode == "bind":
        placement = numa.install_optimizer_state_numa(mode)
        if placement is None or placement.node < 0:
            return {"skip": "GPU NUMA topology is unavailable"}
        try:
            if numa._available_bytes(placement.node) < numa._HEADROOM_BYTES + (32 << 20):
                return {"skip": "insufficient local headroom to bind the test buffers"}
            policy = placement.policy
            previous = policy.bind(placement.node)
            policy.restore(previous)
        except OSError as exc:
            return {"skip": f"NUMA policy syscalls are unavailable: {exc}"}

    param = torch.nn.Parameter(torch.ones(4 << 20, device="cuda"))
    optimizer = torch.optim.SGD([param], lr=0.1, momentum=0.9)
    offloader = ChunkedOptimizerStateOffloader(
        optimizer, master_params=[], state_dtypes=[torch.float32], chunk_size_bytes=param.nbytes, offload_fraction=1.0
    )
    for step in (1, 2):
        param.grad = torch.full_like(param, step)
        offloader.step()
    torch.cuda.synchronize()
    buffer = offloader._cpu_state[param]["momentum_buffer"]
    assert buffer.is_pinned()
    result = {"digest": hashlib.sha256(param.detach().cpu().numpy().tobytes()).hexdigest()}
    if mode == "bind":
        try:
            pages = placement.policy.page_bytes(buffer.data_ptr(), buffer.nbytes)
        except OSError as exc:
            return {"skip": f"NUMA page locations are unavailable: {exc}"}
        if any(node < 0 for node in pages):
            return {"skip": "some sampled pages could not be located"}
        assert pages == {placement.node: buffer.nbytes}, pages
    return result


def test_optimizer_state_placement_preserves_updates():
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    results = []
    for mode in ("off", "bind"):
        proc = subprocess.run(
            [sys.executable, __file__, "--worker", mode], text=True, capture_output=True, timeout=180, check=False
        )
        assert proc.returncode == 0, proc.stderr
        result = json.loads(proc.stdout.strip().splitlines()[-1])
        if "skip" in result:
            pytest.skip(result["skip"])
        results.append(result["digest"])
    assert results[0] == results[1]


if __name__ == "__main__":
    if sys.argv[1:2] == ["--worker"]:
        print(json.dumps(_worker(sys.argv[2])))
    else:
        sys.exit(pytest.main([__file__, "-v", "-s"]))
