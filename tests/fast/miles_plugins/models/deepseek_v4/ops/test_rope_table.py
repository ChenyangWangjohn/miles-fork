"""The rope table cache (miles_plugins/models/deepseek_v4/ops/rope.py precompute_freqs_cis): one table per setting at
a power-of-two length, sliced to the length asked for, is the table a build at that length gives, bit for bit, in
any order of lengths and at any thread count; a length the cached table holds builds nothing."""

import random

import pytest
import torch

from miles_plugins.models.deepseek_v4.ops import rope
from miles_plugins.models.deepseek_v4.ops.thd_utils import get_q_positions_thd

# DeepSeek-V4.1's two settings: window-only layers (rotary base, no YaRN) and compressed layers (YaRN 16x of 64K)
SETTINGS = {
    "window": dict(dim=64, original_seq_len=0, base=10000, factor=16, beta_fast=32, beta_slow=1),
    "compressed": dict(dim=64, original_seq_len=65536, base=160000, factor=16, beta_fast=32, beta_slow=1),
}
LENGTHS = [1, 2, 3, 7, 63, 64, 65, 127, 128, 129, 1000, 4095, 4096, 4097, 21504, 32767, 32768, 32769, 65537]


@pytest.fixture(autouse=True)
def _empty_cache():
    rope.precompute_freqs_cis.cache_clear()
    yield
    rope.precompute_freqs_cis.cache_clear()


def _bits(t: torch.Tensor) -> torch.Tensor:
    return torch.view_as_real(t).contiguous().view(torch.int32)


def _assert_same(got: torch.Tensor, want: torch.Tensor) -> None:
    assert got.shape == want.shape and got.dtype == want.dtype == torch.complex64
    assert got.is_contiguous()
    assert torch.equal(_bits(got), _bits(want))


@pytest.mark.parametrize("setting", list(SETTINGS))
def test_a_slice_of_the_cached_table_is_the_table_of_that_length(setting):
    kwargs = SETTINGS[setting]
    lengths = LENGTHS.copy()
    random.Random(0).shuffle(lengths)  # the cache grows between requests
    for seqlen in lengths:
        _assert_same(
            rope.precompute_freqs_cis(seqlen=seqlen, **kwargs), rope._build_freqs_cis(seqlen=seqlen, **kwargs)
        )


def test_the_rows_do_not_depend_on_the_thread_count():
    threads = torch.get_num_threads()
    kwargs = SETTINGS["compressed"]
    try:
        torch.set_num_threads(1)
        one = rope._build_freqs_cis(seqlen=65536, **kwargs)
        torch.set_num_threads(max(threads, 4))
        _assert_same(rope.precompute_freqs_cis(seqlen=40000, **kwargs), one[:40000])
    finally:
        torch.set_num_threads(threads)


def test_a_length_the_table_holds_builds_nothing(monkeypatch):
    built = []

    def counting_build(dim, seqlen, *args):
        built.append(seqlen)
        return build(dim, seqlen, *args)

    build = rope._build_freqs_cis
    monkeypatch.setattr(rope, "_build_freqs_cis", counting_build)
    kwargs = SETTINGS["compressed"]
    first = rope.precompute_freqs_cis(seqlen=1000, **kwargs)
    for seqlen in (3000, 1024, 4096, 17, 5000, 8192):
        rope.precompute_freqs_cis(seqlen=seqlen, **kwargs)
    assert built == [1024, 4096, 8192]
    # a hit is a view of the cached table, no copy
    table = rope.precompute_freqs_cis(seqlen=8192, **kwargs)
    assert rope.precompute_freqs_cis(seqlen=100, **kwargs).data_ptr() == table.data_ptr()
    # the setting is part of the key: the window layers' table is its own
    rope.precompute_freqs_cis(seqlen=100, **SETTINGS["window"])
    assert built == [1024, 4096, 8192, 128] and rope.precompute_freqs_cis.cache_info().currsize == 2
    # a table handed out before the cache grew keeps its values
    _assert_same(first, build(seqlen=1000, **kwargs))


def test_packed_rows_index_the_same_values():
    """THD: each CP rank's q rows are positions within their own sequence, gathered from the stream's table."""
    lens, cp = [9000, 1, 20000, 128, 3000, 887], 8
    total = sum(lens)
    cu_seqlens = torch.tensor([0, *torch.tensor(lens).cumsum(0).tolist()], dtype=torch.int32)
    for kwargs in SETTINGS.values():
        cached = rope.precompute_freqs_cis(seqlen=total, **kwargs)
        exact = rope._build_freqs_cis(seqlen=total, **kwargs)
        for rank in range(cp):
            positions = get_q_positions_thd(cu_seqlens, total // cp, rank * (total // cp))
            _assert_same(cached.index_select(0, positions), exact.index_select(0, positions))


def test_inference_warmup_can_feed_training_autograd():
    with torch.inference_mode():
        warm = rope.precompute_freqs_cis(seqlen=128, **SETTINGS["window"])
    freqs = rope.precompute_freqs_cis(seqlen=64, **SETTINGS["window"])
    assert freqs.data_ptr() == warm.data_ptr() and not freqs.is_inference()
    x = torch.ones_like(freqs, requires_grad=True)
    (x * freqs).real.sum().backward()
    torch.testing.assert_close(x.grad, freqs.conj())


def test_cache_evicts_settings_and_can_be_cleared():
    kwargs = SETTINGS["window"]
    oldest = rope.precompute_freqs_cis(seqlen=8, **kwargs)
    for i in range(32):
        rope.precompute_freqs_cis(seqlen=8, **{**kwargs, "base": 20000 + i})
    assert rope.precompute_freqs_cis.cache_info().currsize == 32
    assert rope.precompute_freqs_cis(seqlen=8, **kwargs).data_ptr() != oldest.data_ptr()
    rope.precompute_freqs_cis.cache_clear()
    assert rope.precompute_freqs_cis.cache_info().currsize == 0


def test_implicit_output_device_tracks_torch_default():
    kwargs = SETTINGS["window"]
    cpu = rope.precompute_freqs_cis(seqlen=16, **kwargs)
    with torch.device("meta"):
        assert rope.precompute_freqs_cis(seqlen=8, **kwargs).device.type == "meta"
    assert rope.precompute_freqs_cis(seqlen=8, **kwargs).data_ptr() == cpu.data_ptr()


def test_length_edges_preserve_arange_behavior():
    kwargs = SETTINGS["window"]
    rope.precompute_freqs_cis(seqlen=8, **kwargs)
    with pytest.raises(RuntimeError):
        rope.precompute_freqs_cis(seqlen=-1, **kwargs)
    assert rope.precompute_freqs_cis(seqlen=0, **kwargs).shape == (0, 32)
    for length in (3.5, torch.tensor(3), torch.tensor(3.5)):
        _assert_same(
            rope.precompute_freqs_cis(seqlen=length, **kwargs), rope._build_freqs_cis(seqlen=length, **kwargs)
        )
    with pytest.raises(TypeError):
        rope.precompute_freqs_cis(seqlen=torch.tensor([3]), **kwargs)


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two CUDA devices")
def test_unindexed_cuda_device_tracks_current_ordinal():
    for index in range(2):
        with torch.cuda.device(index):
            freqs = rope.precompute_freqs_cis(seqlen=12 - index, device="cuda", **SETTINGS["window"])
            assert freqs.device == torch.device("cuda", index)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_cuda_factory_builds_on_the_consuming_stream():
    kwargs = SETTINGS["window"]
    with torch.device("cuda"):
        first = rope.precompute_freqs_cis(seqlen=256, **kwargs)
        with torch.cuda.stream(torch.cuda.Stream()):
            second = rope.precompute_freqs_cis(seqlen=64, **kwargs)
            _assert_same(second, rope._build_freqs_cis(seqlen=64, **kwargs))
            assert second.data_ptr() != first.data_ptr()  # different streams have independent pending builds
