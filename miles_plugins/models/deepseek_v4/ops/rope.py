import math
import operator
from functools import lru_cache
from threading import Lock

import torch
from megatron.core.transformer import TransformerConfig


@lru_cache(maxsize=32)
def _prefix_for_settings(dim, original_seq_len, base, factor, beta_fast, beta_slow, output_device):
    table = None
    allocation_stream = None
    lock = Lock()

    def prefix(length):
        nonlocal table, allocation_stream
        stream = torch.cuda.current_stream(output_device) if output_device.type == "cuda" else None
        with lock:
            if table is None or table.size(0) < length:
                capacity = 1 << max(length - 1, 0).bit_length()
                # A table built during inference must also be usable in a later autograd pass.
                with torch.inference_mode(False):
                    table = _build_freqs_cis(
                        dim, capacity, original_seq_len, base, factor, beta_fast, beta_slow, "cpu"
                    ).to(output_device)
                allocation_stream = stream
            elif stream is not None and stream != allocation_stream:
                # A later eviction may drop the cache's reference while this stream is still using the table.
                table.record_stream(stream)
            return table[:length]

    return prefix


def precompute_freqs_cis(
    dim, seqlen, original_seq_len, base, factor, beta_fast, beta_slow, device=None
) -> torch.Tensor:
    """Return a prefix of the rotary frequency table for this RoPE/YaRN setting and device.

    The frequencies are independent of the requested length. Growing each setting to the next
    power of two lets varying packed sequence lengths share a read-only view of one table. Up to
    32 settings are retained for CPU-built tables used on CPU or CUDA.
    """
    build_device = torch.get_default_device()
    output_device = build_device if device is None else torch.device(device)
    if output_device.type == "cuda" and output_device.index is None:
        output_device = torch.device("cuda", torch.cuda.current_device())
    settings = (dim, original_seq_len, base, factor, beta_fast, beta_slow)
    try:
        length = operator.index(seqlen)
    except TypeError:
        length = None
    # Only share CPU-built constants. Device-built tables can still have work pending on the caller's stream.
    # The exact path also preserves arange's handling of fractional endpoints and differentiable settings.
    if (
        build_device.type != "cpu"
        or output_device.type not in ("cpu", "cuda")
        or length is None
        or (isinstance(seqlen, torch.Tensor) and seqlen.ndim != 0)
        or any(isinstance(x, torch.Tensor) for x in settings)
    ):
        return _build_freqs_cis(dim, seqlen, original_seq_len, base, factor, beta_fast, beta_slow, build_device).to(
            output_device
        )
    if length < 0:
        raise RuntimeError("upper bound and lower bound inconsistent with step sign")
    return _prefix_for_settings(*settings, output_device)(length)


precompute_freqs_cis.cache_clear = _prefix_for_settings.cache_clear
precompute_freqs_cis.cache_info = _prefix_for_settings.cache_info
precompute_freqs_cis.cache_parameters = _prefix_for_settings.cache_parameters


def _build_freqs_cis(dim, seqlen, original_seq_len, base, factor, beta_fast, beta_slow, build_device=None):
    def find_correction_dim(num_rotations, dim, base, max_seq_len):
        return dim * math.log(max_seq_len / (num_rotations * 2 * math.pi)) / (2 * math.log(base))

    def find_correction_range(low_rot, high_rot, dim, base, max_seq_len):
        low = math.floor(find_correction_dim(low_rot, dim, base, max_seq_len))
        high = math.ceil(find_correction_dim(high_rot, dim, base, max_seq_len))
        return max(low, 0), min(high, dim - 1)

    def linear_ramp_factor(min, max, dim):
        if min == max:
            max += 0.001
        linear_func = (torch.arange(dim, dtype=torch.float32, device=build_device) - min) / (max - min)
        ramp_func = torch.clamp(linear_func, 0, 1)
        return ramp_func

    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32, device=build_device) / dim))
    if original_seq_len > 0:
        low, high = find_correction_range(beta_fast, beta_slow, dim, base, original_seq_len)
        smooth = 1 - linear_ramp_factor(low, high, dim // 2)
        freqs = freqs / factor * (1 - smooth) + freqs * smooth

    t = torch.arange(seqlen, device=build_device)
    freqs = torch.outer(t, freqs)
    return torch.polar(torch.ones_like(freqs), freqs)


def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor, inverse: bool = False) -> torch.Tensor:
    """Apply RoPE in-place to the last dim of ``x``.

    ``x`` has shape ``[..., dim]`` where ``dim`` is even; the last-dim pairs are
    treated as complex numbers multiplied by ``freqs_cis``. When ``inverse=True``
    the conjugate rotation is applied (used for the indexer's inverse rope).
    """
    y = x
    x = torch.view_as_complex(x.float().unflatten(-1, (-1, 2)))
    if inverse:
        freqs_cis = freqs_cis.conj()
    if x.ndim == 3:
        freqs_cis = freqs_cis.view(1, x.size(1), x.size(-1))
    else:
        freqs_cis = freqs_cis.view(1, x.size(1), 1, x.size(-1))
    x = torch.view_as_real(x * freqs_cis).flatten(-2)
    y.copy_(x)
    return y


def wrapped_precompute_freqs_cis(
    config: TransformerConfig, rope_head_dim: int, base: float, yarn_disabled: bool, max_seq_len: int, device
):
    # max_seq_len = global length (cp_size * seqlen_local); a view of the setting's cached table (precompute_freqs_cis).
    # yarn_disabled=True → original_seq_len=0, which makes precompute_freqs_cis skip the YaRN
    # correction-range interpolation. Used by 0415 for pure-window (compress_ratio==0) layers.
    original_seq_len = 0 if yarn_disabled else config.original_max_position_embeddings

    inputs = dict(
        dim=rope_head_dim,
        seqlen=max_seq_len,
        original_seq_len=original_seq_len,
        base=base,
        factor=config.rotary_scaling_factor,
        beta_fast=config.beta_fast,
        beta_slow=config.beta_slow,
    )

    assert config.rotary_scaling_factor in (4, 16), f"Unexpected rotary_scaling_factor: {config.rotary_scaling_factor}"
    expected_original = 0 if yarn_disabled else 65536
    assert inputs == dict(
        dim=rope_head_dim,
        seqlen=max_seq_len,
        original_seq_len=expected_original,
        base=base,
        factor=config.rotary_scaling_factor,
        beta_fast=32,
        beta_slow=1,
    )

    return precompute_freqs_cis(**inputs, device=device)
