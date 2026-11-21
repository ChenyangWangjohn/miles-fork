"""Check the first optimizer-state chunk at the reduction boundary and after three updates."""

import torch
from megatron.core.optimizer import OptimizerConfig
from megatron.core.optimizer.optimizer import Float16OptimizerWithFloat16Params
from tests.ci.ci_register import register_cuda_ci

from miles.backends.megatron_utils import model as model_module

register_cuda_ci(est_time=30, suite="stage-b-2-gpu-h200", labels=["megatron"], hardware=["hopper", "blackwell"])


def test_prefetch_restores_first_chunk_before_reductions_without_changing_updates(monkeypatch):
    def run(prefetch_enabled):
        param = torch.nn.Parameter(torch.arange(32, device="cuda", dtype=torch.bfloat16))
        adam = torch.optim.AdamW([param], lr=0.01)
        config = OptimizerConfig(bf16=True, use_distributed_optimizer=True, chunked_optimizer_state_offload=True)
        optimizer = Float16OptimizerWithFloat16Params(adam, config, grad_scaler=None, init_state_fn=None)
        master = adam.param_groups[0]["params"][0]
        optimizer.enable_chunked_optimizer_state_offload(
            master_params=[master], state_dtypes=(torch.float32, torch.float32)
        )
        snapshots = []

        def check_residency(chunks):
            if snapshots:  # Adam allocates the moments during its first update.
                assert adam.state[master]["exp_avg"].is_cuda == prefetch_enabled

        monkeypatch.setattr(model_module, "finalize_model_grads", check_residency)
        for step in range(3):
            if snapshots:
                assert adam.state[master]["exp_avg"].is_cpu
            param.grad = torch.full_like(param, (step + 1) / 16)
            prefetch = optimizer.prefetch_optimizer_state_for_gradient_finalization if prefetch_enabled else None
            model_module.finalize_model_grads_with_empty_cache([], prefetch_optimizer_state=prefetch)
            assert not optimizer.prepare_grads()
            optimizer.step_with_ready_grads()
            # Moment writes to the canonical host tensors are asynchronous.
            torch.cuda.synchronize()
            snapshots.append(
                (
                    param.detach().cpu(),
                    master.detach().cpu(),
                    adam.state[master]["exp_avg"].clone(),
                    adam.state[master]["exp_avg_sq"].clone(),
                )
            )
        return snapshots

    for reference, prefetched in zip(run(False), run(True), strict=True):
        for expected, actual in zip(reference, prefetched, strict=True):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
