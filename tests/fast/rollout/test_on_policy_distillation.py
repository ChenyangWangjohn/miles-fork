import asyncio
import math
from argparse import Namespace

import httpx
import pytest
from tests.ci.ci_register import register_cpu_ci

import miles.rollout.on_policy_distillation as opd
from miles.rollout.on_policy_distillation import (
    _compute_topk_reverse_kl,
    _score_payload,
    _scoring_post,
    _teacher_sampled_log_probs,
    reward_func,
)
from miles.utils.types import RewardSpec, Sample

register_cpu_ci(est_time=60, suite="stage-a-cpu")


def _entry(prob: float, token_id: int):
    return [math.log(prob), token_id]


def _args(strategy: str, weight_mode: str = "student_p"):
    return Namespace(
        opd_top_k_strategy=strategy,
        opd_reward_weight_mode=weight_mode,
    )


def _sample():
    return Sample(
        tokens=[10, 11, 12],
        response_length=2,
        metadata={
            "opd_student_top_logprobs": [
                [_entry(0.6, 1), _entry(0.4, 2)],
                [_entry(0.7, 4), _entry(0.3, 5)],
            ]
        },
    )


def _teacher_payload():
    return {
        "teacher": {
            "meta_info": {
                "input_top_logprobs": [
                    None,
                    [_entry(0.5, 2), _entry(0.5, 3)],
                    [_entry(0.8, 4), _entry(0.2, 6)],
                ],
                "input_token_ids_logprobs": [
                    None,
                    [_entry(0.3, 1), _entry(0.7, 2)],
                    [_entry(0.4, 4), _entry(0.6, 5)],
                ],
            }
        },
        "student_on_teacher": {
            "meta_info": {
                "input_token_ids_logprobs": [
                    None,
                    [_entry(0.4, 2), _entry(0.2, 3)],
                    [_entry(0.7, 4), _entry(0.1, 6)],
                ]
            }
        },
    }


def test_topk_only_student_uses_student_probability_weights():
    reverse_kl = _compute_topk_reverse_kl(_args("only-student"), _sample(), _teacher_payload())

    expected_0 = 0.6 * math.log(0.6 / 0.3) + 0.4 * math.log(0.4 / 0.7)
    expected_1 = 0.7 * math.log(0.7 / 0.4) + 0.3 * math.log(0.3 / 0.6)

    assert reverse_kl.tolist() == pytest.approx([expected_0, expected_1])


def test_topk_intersection_uses_overlap_only():
    reverse_kl = _compute_topk_reverse_kl(_args("intersection", "none"), _sample(), _teacher_payload())

    assert reverse_kl.tolist() == pytest.approx(
        [
            math.log(0.4 / 0.5),
            math.log(0.7 / 0.8),
        ]
    )


def test_topk_only_teacher_does_not_need_student_top_logprobs():
    sample = Sample(tokens=[10, 11, 12], response_length=2)

    reverse_kl = _compute_topk_reverse_kl(_args("only-teacher"), sample, _teacher_payload())

    expected_0 = (2 / 3) * math.log(0.4 / 0.5) + (1 / 3) * math.log(0.2 / 0.5)
    expected_1 = (7 / 8) * math.log(0.7 / 0.8) + (1 / 8) * math.log(0.1 / 0.2)

    assert reverse_kl.tolist() == pytest.approx([expected_0, expected_1])


def test_topk_xor_uses_symmetric_difference_without_normalization():
    reverse_kl = _compute_topk_reverse_kl(_args("xor", "none"), _sample(), _teacher_payload())

    expected_0 = math.log(0.6 / 0.3) + math.log(0.2 / 0.5)
    expected_1 = math.log(0.3 / 0.6) + math.log(0.1 / 0.2)

    assert reverse_kl.tolist() == pytest.approx([expected_0, expected_1])


# ---------------------------------------------------------------------------
# Scoring payload: response window
# ---------------------------------------------------------------------------


def test_score_payload_materializes_only_the_response_window():
    payload = _score_payload([10, 11, 12, 13], response_length=2)

    assert payload["input_ids"] == [10, 11, 12, 13]
    # Two prompt tokens; logprobs start one token before the response window.
    assert payload["logprob_start_len"] == 1
    assert payload["sampling_params"]["max_new_tokens"] == 0


def test_score_payload_empty_response_starts_at_last_prompt_token():
    payload = _score_payload([10, 11, 12], response_length=0)

    assert payload["logprob_start_len"] == 2


def test_score_payload_rejects_out_of_bounds_response_window():
    with pytest.raises(ValueError, match="out of bounds"):
        _score_payload([10, 11], response_length=3)
    with pytest.raises(ValueError, match="out of bounds"):
        _score_payload([10, 11], response_length=-1)


def test_score_payload_rejects_windows_without_a_prompt_token():
    with pytest.raises(ValueError, match="at least one prompt token"):
        _score_payload([10, 11], response_length=2)


# ---------------------------------------------------------------------------
# Sampled log-prob extraction: alignment guard
# ---------------------------------------------------------------------------


def _scored_sample() -> Sample:
    return Sample(tokens=[10, 11, 12, 13], response_length=2)


def _reply(entries: list[list]) -> dict:
    return {"meta_info": {"input_token_logprobs": entries}}


def test_sampled_log_probs_match_between_full_and_window_replies():
    sample = _scored_sample()
    full_reply = _reply([[None, 10, None], [-0.5, 11, None], [-1.0, 12, None], [-2.0, 13, None]])
    window_reply = _reply([[None, 11, None], [-1.0, 12, None], [-2.0, 13, None]])

    full = _teacher_sampled_log_probs(full_reply, sample)
    window = _teacher_sampled_log_probs(window_reply, sample)

    assert full.tolist() == window.tolist() == [-1.0, -2.0]


def test_sampled_log_probs_reject_misaligned_tokens():
    reply = _reply([[None, 11, None], [-1.0, 99, None], [-2.0, 13, None]])

    with pytest.raises(ValueError, match="token alignment mismatch"):
        _teacher_sampled_log_probs(reply, _scored_sample())


def test_sampled_log_probs_reject_wrong_position_count():
    reply = _reply([[None, 11, None], [-2.0, 13, None]])

    with pytest.raises(ValueError, match="expected 2"):
        _teacher_sampled_log_probs(reply, _scored_sample())


def test_sampled_log_probs_reject_missing_and_non_finite_values():
    with pytest.raises(ValueError, match="None for a response-token"):
        _teacher_sampled_log_probs(_reply([[None, 11, None], [None, 12, None], [-2.0, 13, None]]), _scored_sample())
    with pytest.raises(ValueError, match="non-finite"):
        _teacher_sampled_log_probs(
            _reply([[None, 11, None], [math.inf, 12, None], [-2.0, 13, None]]), _scored_sample()
        )


def test_sampled_log_probs_empty_response_returns_empty_tensor():
    sample = Sample(tokens=[10, 11], response_length=0)

    assert _teacher_sampled_log_probs(_reply([]), sample).numel() == 0


# ---------------------------------------------------------------------------
# Bounded scoring transport
# ---------------------------------------------------------------------------


def _scoring_args(**overrides) -> Namespace:
    defaults = {
        "opd_log_task_reward": False,
        "opd_scoring_timeout": 5.0,
        "opd_scoring_max_inflight": 0,
        "opd_scoring_retries": 0,
    }
    defaults.update(overrides)
    return Namespace(**defaults)


def test_scoring_post_retries_after_timeout_then_succeeds(monkeypatch):
    monkeypatch.setattr(opd, "_SCORING_RETRY_BACKOFF_S", 0.01)
    calls = {"count": 0}

    async def flaky_post(url, payload, max_retries=1):
        calls["count"] += 1
        if calls["count"] == 1:
            raise TimeoutError("first attempt times out")
        return {"ok": True}

    monkeypatch.setattr(opd, "post", flaky_post)
    sample = _scored_sample()

    result = asyncio.run(
        _scoring_post(
            _scoring_args(opd_scoring_retries=1), "http://teacher", {"input_ids": [1]}, sample=sample, target="teacher"
        )
    )

    assert result == {"ok": True}
    assert calls["count"] == 2


def test_student_weight_version_reads_router_model_info(monkeypatch):
    seen = {}

    async def fake_scoring_get(args, url, *, sample, target):
        seen["url"] = url
        seen["target"] = target
        return {"weight_version": 12}

    monkeypatch.setattr(opd, "_scoring_get", fake_scoring_get)

    version = asyncio.run(
        opd._student_weight_version(
            _scoring_args(sglang_router_ip="student", sglang_router_port=30000),
            _scored_sample(),
        )
    )

    assert version == "12"
    assert seen == {
        "url": "http://student:30000/model_info",
        "target": "student-version",
    }


def test_student_weight_version_requires_version_metadata(monkeypatch):
    async def fake_scoring_get(args, url, *, sample, target):
        return {"model_path": "student"}

    monkeypatch.setattr(opd, "_scoring_get", fake_scoring_get)

    with pytest.raises(ValueError, match="missing weight_version"):
        asyncio.run(
            opd._student_weight_version(
                _scoring_args(sglang_router_ip="student", sglang_router_port=30000),
                _scored_sample(),
            )
        )


def test_scoring_post_fails_fast_with_zero_retries(monkeypatch):
    async def failing_post(url, payload, max_retries=1):
        raise httpx.ConnectError("connection refused")

    monkeypatch.setattr(opd, "post", failing_post)
    sample = _scored_sample()

    with pytest.raises(RuntimeError, match="failed after 1 attempt"):
        asyncio.run(
            _scoring_post(_scoring_args(), "http://teacher", {"input_ids": [1]}, sample=sample, target="teacher")
        )


def test_scoring_post_shares_one_deadline_across_retries(monkeypatch):
    monkeypatch.setattr(opd, "_SCORING_RETRY_BACKOFF_S", 0.01)
    calls = {"count": 0}

    async def hanging_post(url, payload, max_retries=1):
        calls["count"] += 1
        await asyncio.sleep(60)

    monkeypatch.setattr(opd, "post", hanging_post)

    async def run() -> float:
        loop = asyncio.get_running_loop()
        start = loop.time()
        with pytest.raises(RuntimeError, match="failed after"):
            await _scoring_post(
                _scoring_args(opd_scoring_timeout=0.2, opd_scoring_retries=5),
                "http://teacher",
                {"input_ids": [1]},
                sample=_scored_sample(),
                target="teacher",
            )
        return loop.time() - start

    # Five retries share the 0.2s deadline instead of each getting a fresh one.
    assert asyncio.run(run()) < 5.0
    assert calls["count"] == 1


def test_scoring_post_bounds_inflight_requests(monkeypatch):
    state = {"current": 0, "max": 0}

    async def tracked_post(url, payload, max_retries=1):
        state["current"] += 1
        state["max"] = max(state["max"], state["current"])
        await asyncio.sleep(0.01)
        state["current"] -= 1
        return {"ok": True}

    monkeypatch.setattr(opd, "post", tracked_post)
    args = _scoring_args(opd_scoring_max_inflight=1)

    async def run():
        await asyncio.gather(
            *(
                _scoring_post(args, "http://teacher", {"input_ids": [1]}, sample=_scored_sample(), target="teacher")
                for _ in range(4)
            )
        )

    asyncio.run(run())

    assert state["max"] == 1


def test_scoring_post_deadline_includes_inflight_wait(monkeypatch):
    entered_post = asyncio.Event()
    release_post = asyncio.Event()

    async def blocked_post(url, payload, max_retries=1):
        entered_post.set()
        await release_post.wait()
        return {"ok": True}

    monkeypatch.setattr(opd, "post", blocked_post)
    long_args = _scoring_args(opd_scoring_timeout=5.0, opd_scoring_max_inflight=1)
    short_args = _scoring_args(opd_scoring_timeout=0.05, opd_scoring_max_inflight=1)

    async def run():
        first = asyncio.create_task(
            _scoring_post(long_args, "http://teacher", {"input_ids": [1]}, sample=_scored_sample(), target="teacher")
        )
        await entered_post.wait()
        with pytest.raises(RuntimeError, match="failed after 0 attempt"):
            await _scoring_post(
                short_args,
                "http://teacher",
                {"input_ids": [1]},
                sample=_scored_sample(),
                target="teacher",
            )
        release_post.set()
        assert await first == {"ok": True}

    asyncio.run(run())


def test_scoring_post_unbounded_when_inflight_limit_disabled(monkeypatch):
    state = {"current": 0, "max": 0}

    async def tracked_post(url, payload, max_retries=1):
        state["current"] += 1
        state["max"] = max(state["max"], state["current"])
        await asyncio.sleep(0.01)
        state["current"] -= 1
        return {"ok": True}

    monkeypatch.setattr(opd, "post", tracked_post)
    args = _scoring_args(opd_scoring_max_inflight=0)

    async def run():
        await asyncio.gather(
            *(
                _scoring_post(args, "http://teacher", {"input_ids": [1]}, sample=_scored_sample(), target="teacher")
                for _ in range(4)
            )
        )

    asyncio.run(run())

    assert state["max"] == 4


def test_reward_func_uses_response_window(monkeypatch):
    seen = {}

    async def fake_post(url, payload, max_retries=1):
        seen["url"] = url
        seen["payload"] = payload
        return _reply([[None, 11, None], [-1.0, 12, None], [-2.0, 13, None]])

    monkeypatch.setattr(opd, "post", fake_post)
    args = _scoring_args(opd_log_prob_top_k=0, rm_url="http://teacher/generate")
    sample = _scored_sample()

    response = asyncio.run(reward_func(args, sample))

    assert seen["url"] == "http://teacher/generate"
    assert seen["payload"]["logprob_start_len"] == 1
    assert _teacher_sampled_log_probs(response, sample).tolist() == [-1.0, -2.0]


# ---------------------------------------------------------------------------
# Position-blocked top-k scoring
# ---------------------------------------------------------------------------


def _blocked_top_k_args(strategy: str, *, block_size: int = 2, weight_mode: str = "student_p") -> Namespace:
    return _scoring_args(
        opd_log_prob_top_k=2,
        opd_top_k_strategy=strategy,
        opd_reward_weight_mode=weight_mode,
        opd_top_k_scoring_block_size=block_size,
        rm_url="http://teacher/generate",
        sglang_router_ip="student",
        sglang_router_port=30000,
    )


def _blocked_sample() -> Sample:
    return Sample(
        tokens=[10, 11, 12, 13, 14, 15],
        response_length=4,
        metadata={
            "opd_student_top_logprobs": [
                [_entry(0.6, 1), _entry(0.4, 2)],
                [_entry(0.7, 2), _entry(0.3, 3)],
                [_entry(0.8, 4), _entry(0.2, 5)],
                [_entry(0.9, 5), _entry(0.1, 6)],
            ]
        },
    )


def _candidate_score(target: str, position: int, token_id: int) -> float:
    target_offset = 0.2 if target == "teacher" else 0.4
    return -(target_offset + 0.05 * position + 0.01 * token_id)


def _blocked_reply(
    sample: Sample,
    payload: dict,
    target: str,
    *,
    weight_version: str | None = None,
) -> dict:
    prompt_length = len(sample.tokens) - sample.response_length
    start = payload["logprob_start_len"] + 1 - prompt_length
    end = len(payload["input_ids"]) - prompt_length
    response_tokens = sample.tokens[prompt_length + start : prompt_length + end]
    candidate_ids = payload["token_ids_logprob"]
    meta_info = {
        "input_token_logprobs": [None, *[[-1.0, token_id] for token_id in response_tokens]],
        "input_token_ids_logprobs": [
            None,
            *[
                [[_candidate_score(target, position, token_id), token_id] for token_id in candidate_ids]
                for position in range(start, end)
            ],
        ],
    }
    if weight_version is not None:
        meta_info["weight_version"] = weight_version
    return {"meta_info": meta_info}


def _global_candidate_reply(sample: Sample, candidate_rows: list[list], target: str) -> dict:
    candidate_ids = sorted({int(entry[1]) for row in candidate_rows for entry in row})
    return {
        "meta_info": {
            "input_token_ids_logprobs": [
                None,
                *[
                    [[_candidate_score(target, position, token_id), token_id] for token_id in candidate_ids]
                    for position in range(sample.response_length)
                ],
            ]
        }
    }


@pytest.mark.parametrize("weight_mode", ["student_p", "teacher_p", "none"])
def test_only_student_block_scoring_matches_response_wide_union(monkeypatch, weight_mode):
    sample = _blocked_sample()
    args = _blocked_top_k_args("only-student", weight_mode=weight_mode)
    calls = []

    async def fake_scoring_post(args, url, payload, *, sample, target):
        calls.append((url, payload, target))
        return _blocked_reply(sample, payload, target)

    monkeypatch.setattr(opd, "_scoring_post", fake_scoring_post)

    blocked_payload = asyncio.run(reward_func(args, sample))
    legacy_payload = {
        "teacher": _global_candidate_reply(
            sample,
            sample.metadata["opd_student_top_logprobs"],
            "teacher",
        )
    }

    assert _compute_topk_reverse_kl(args, sample, blocked_payload).tolist() == pytest.approx(
        _compute_topk_reverse_kl(args, sample, legacy_payload).tolist()
    )
    assert [call[1]["input_ids"] for call in calls] == [
        sample.tokens[:4],
        sample.tokens[:6],
    ]
    assert [call[1]["logprob_start_len"] for call in calls] == [1, 3]
    assert [call[1]["token_ids_logprob"] for call in calls] == [[1, 2, 3], [4, 5, 6]]
    compact_rows = blocked_payload["teacher"]["meta_info"]["input_token_ids_logprobs"][1:]
    assert [len(row) for row in compact_rows] == [2, 2, 2, 2]


@pytest.mark.parametrize("weight_mode", ["student_p", "teacher_p", "none"])
def test_only_teacher_block_scoring_matches_response_wide_union(monkeypatch, weight_mode):
    sample = Sample(tokens=[10, 11, 12, 13, 14, 15], response_length=4)
    args = _blocked_top_k_args("only-teacher", weight_mode=weight_mode)
    teacher_top = [
        [_entry(0.6, 1), _entry(0.4, 2)],
        [_entry(0.7, 2), _entry(0.3, 3)],
        [_entry(0.8, 4), _entry(0.2, 5)],
        [_entry(0.9, 5), _entry(0.1, 6)],
    ]
    teacher_response = {
        "meta_info": {
            "input_token_logprobs": [None, *[[-1.0, token_id] for token_id in sample.tokens[-4:]]],
            "input_top_logprobs": [None, *teacher_top],
        }
    }
    calls = []

    async def fake_scoring_post(args, url, payload, *, sample, target):
        calls.append((url, payload, target))
        if target == "teacher":
            return teacher_response
        return _blocked_reply(sample, payload, target, weight_version="7")

    async def fake_student_weight_version(args, sample):
        return "7"

    monkeypatch.setattr(opd, "_scoring_post", fake_scoring_post)
    monkeypatch.setattr(opd, "_student_weight_version", fake_student_weight_version)

    blocked_payload = asyncio.run(reward_func(args, sample))
    legacy_payload = {
        "teacher": teacher_response,
        "student_on_teacher": _global_candidate_reply(sample, teacher_top, "student"),
    }

    assert _compute_topk_reverse_kl(args, sample, blocked_payload).tolist() == pytest.approx(
        _compute_topk_reverse_kl(args, sample, legacy_payload).tolist()
    )
    assert len(calls) == 3
    assert calls[0][2] == "teacher"
    assert "token_ids_logprob" not in calls[0][1]
    assert [call[1]["token_ids_logprob"] for call in calls[1:]] == [[1, 2, 3], [4, 5, 6]]
    assert blocked_payload["student_on_teacher"]["meta_info"]["weight_version"] == "7"


def test_only_teacher_retries_all_blocks_after_student_version_change(monkeypatch):
    sample = Sample(tokens=[10, 11, 12, 13, 14, 15], response_length=4)
    args = _blocked_top_k_args("only-teacher")
    args.opd_scoring_retries = 1
    teacher_top = [
        [_entry(0.6, 1), _entry(0.4, 2)],
        [_entry(0.7, 2), _entry(0.3, 3)],
        [_entry(0.8, 4), _entry(0.2, 5)],
        [_entry(0.9, 5), _entry(0.1, 6)],
    ]
    teacher_response = {
        "meta_info": {
            "input_token_logprobs": [None, *[[-1.0, token_id] for token_id in sample.tokens[-4:]]],
            "input_top_logprobs": [None, *teacher_top],
        }
    }
    expected_versions = iter(["7", "8"])
    block_versions = iter(["7", "8", "8", "8"])
    student_calls = []

    async def fake_student_weight_version(args, sample):
        return next(expected_versions)

    async def fake_scoring_post(args, url, payload, *, sample, target):
        if target == "teacher":
            return teacher_response
        student_calls.append(payload)
        return _blocked_reply(sample, payload, target, weight_version=next(block_versions))

    monkeypatch.setattr(opd, "_SCORING_RETRY_BACKOFF_S", 0)
    monkeypatch.setattr(opd, "_student_weight_version", fake_student_weight_version)
    monkeypatch.setattr(opd, "_scoring_post", fake_scoring_post)

    reward_payload = asyncio.run(reward_func(args, sample))

    assert len(student_calls) == 4
    assert reward_payload["student_on_teacher"]["meta_info"]["weight_version"] == "8"
    assert [payload["token_ids_logprob"] for payload in student_calls] == [
        [1, 2, 3],
        [4, 5, 6],
        [1, 2, 3],
        [4, 5, 6],
    ]


def test_only_teacher_rejects_mixed_student_versions_after_retry_budget(monkeypatch):
    sample = Sample(tokens=[10, 11, 12, 13, 14, 15], response_length=4)
    args = _blocked_top_k_args("only-teacher")
    args.opd_scoring_retries = 1
    teacher_top = [
        [_entry(0.6, 1), _entry(0.4, 2)],
        [_entry(0.7, 2), _entry(0.3, 3)],
        [_entry(0.8, 4), _entry(0.2, 5)],
        [_entry(0.9, 5), _entry(0.1, 6)],
    ]
    teacher_response = {
        "meta_info": {
            "input_token_logprobs": [None, *[[-1.0, token_id] for token_id in sample.tokens[-4:]]],
            "input_top_logprobs": [None, *teacher_top],
        }
    }
    block_versions = iter(["7", "8", "7", "8"])

    async def fake_student_weight_version(args, sample):
        return "7"

    async def fake_scoring_post(args, url, payload, *, sample, target):
        if target == "teacher":
            return teacher_response
        return _blocked_reply(sample, payload, target, weight_version=next(block_versions))

    monkeypatch.setattr(opd, "_SCORING_RETRY_BACKOFF_S", 0)
    monkeypatch.setattr(opd, "_student_weight_version", fake_student_weight_version)
    monkeypatch.setattr(opd, "_scoring_post", fake_scoring_post)

    with pytest.raises(RuntimeError, match="refusing to assemble mixed-version scores"):
        asyncio.run(reward_func(args, sample))


def test_top_k_block_scoring_rejects_missing_candidate(monkeypatch):
    sample = _blocked_sample()
    args = _blocked_top_k_args("only-student")

    async def fake_scoring_post(args, url, payload, *, sample, target):
        response = _blocked_reply(sample, payload, target)
        response["meta_info"]["input_token_ids_logprobs"][1] = []
        return response

    monkeypatch.setattr(opd, "_scoring_post", fake_scoring_post)

    with pytest.raises(ValueError, match="missing candidate token id"):
        asyncio.run(reward_func(args, sample))


def test_top_k_block_scoring_rejects_shifted_response_tokens(monkeypatch):
    sample = _blocked_sample()
    args = _blocked_top_k_args("only-student")

    async def fake_scoring_post(args, url, payload, *, sample, target):
        response = _blocked_reply(sample, payload, target)
        response["meta_info"]["input_token_logprobs"][1][1] = 999
        return response

    monkeypatch.setattr(opd, "_scoring_post", fake_scoring_post)

    with pytest.raises(ValueError, match="token alignment mismatch"):
        asyncio.run(reward_func(args, sample))


def test_zero_block_size_preserves_legacy_response_wide_union(monkeypatch):
    sample = _blocked_sample()
    args = _blocked_top_k_args("only-student", block_size=0)
    calls = []

    async def fake_scoring_post(args, url, payload, *, sample, target):
        calls.append((url, payload, target))
        return {"meta_info": {}}

    monkeypatch.setattr(opd, "_scoring_post", fake_scoring_post)

    asyncio.run(reward_func(args, sample))

    assert len(calls) == 1
    assert calls[0][1]["input_ids"] == sample.tokens
    assert calls[0][1]["token_ids_logprob"] == [1, 2, 3, 4, 5, 6]


# ---------------------------------------------------------------------------
# Observed task reward (--opd-log-task-reward)
# ---------------------------------------------------------------------------


def test_reward_func_records_observed_task_reward_before_teacher_scoring(monkeypatch):
    args = _scoring_args(
        opd_log_task_reward=True,
        opd_log_prob_top_k=0,
        rm_url="http://teacher/generate",
    )
    sample = _scored_sample()
    calls = []

    async def fake_async_rm(task_args, task_sample):
        calls.append("task_reward")
        assert task_args.custom_rm_path is None
        assert task_sample is not sample
        return 0.75

    async def fake_scoring_post(args, url, payload, *, sample, target):
        calls.append("teacher_scoring")
        assert sample.metadata[opd.OPD_TASK_REWARD_METADATA_KEY] == 0.75
        return {"meta_info": {}}

    monkeypatch.setattr("miles.rollout.rm_hub.async_rm", fake_async_rm)
    monkeypatch.setattr(opd, "_scoring_post", fake_scoring_post)

    asyncio.run(reward_func(args, sample))

    assert calls == ["task_reward", "teacher_scoring"]


def test_observed_task_reward_uses_builtin_rm_without_mutating_training_args(monkeypatch):
    training_args = Namespace(
        opd_log_task_reward=True,
        custom_rm_path="miles.rollout.on_policy_distillation.reward_func",
        rm_type="deepscaler",
    )
    sample = Sample(
        response="answer",
        label="42",
        metadata={"dataset": "math", "rm_type": "remote_rm"},
        reward_spec=RewardSpec(rm_type="remote_rm", custom_rm_path="pkg.remote_reward"),
    )
    call = {}

    async def fake_async_rm(args, received_sample):
        call.update(args=args, sample=received_sample)
        return 1

    monkeypatch.setattr("miles.rollout.rm_hub.async_rm", fake_async_rm)

    asyncio.run(opd._record_observed_task_reward(training_args, sample))

    assert call["args"] is not training_args
    assert call["args"].custom_rm_path is None
    assert call["args"].rm_type == "deepscaler"
    assert call["sample"] is not sample
    assert call["sample"].metadata == {"dataset": "math"}
    assert call["sample"].reward_spec is None
    assert training_args.custom_rm_path == "miles.rollout.on_policy_distillation.reward_func"
    assert sample.reward_spec == RewardSpec(rm_type="remote_rm", custom_rm_path="pkg.remote_reward")
    assert sample.metadata == {
        "dataset": "math",
        "rm_type": "remote_rm",
        opd.OPD_TASK_REWARD_METADATA_KEY: 1.0,
    }


@pytest.mark.parametrize(
    ("bad_value", "match"),
    [
        (None, "must be scalar"),
        (float("nan"), "must be finite"),
    ],
)
def test_observed_task_reward_rejects_non_scalar_and_non_finite_scores(monkeypatch, bad_value, match):
    args = Namespace(opd_log_task_reward=True)

    async def fake_async_rm(*_args, **_kwargs):
        return bad_value

    monkeypatch.setattr("miles.rollout.rm_hub.async_rm", fake_async_rm)
    with pytest.raises(ValueError, match=match):
        asyncio.run(opd._record_observed_task_reward(args, Sample(response="x", label="1")))


def test_observed_task_reward_is_logged_raw_but_optimization_reward_stays_zero():
    sample = Sample(tokens=[10, 11, 12], response_length=2)
    sample.reward = {"meta_info": {"input_token_logprobs": [None, [-0.2, 11], [-0.3, 12]]}}
    sample.metadata[opd.OPD_TASK_REWARD_METADATA_KEY] = 1.0

    raw_rewards, rewards = opd.post_process_rewards(
        Namespace(opd_log_prob_top_k=0, opd_log_task_reward=True, reward_key=""),
        [sample],
    )

    assert raw_rewards == [1.0]
    assert rewards == [0.0]
    assert sample.teacher_log_probs.tolist() == pytest.approx([-0.2, -0.3])


def test_observed_task_reward_missing_from_a_sample_fails_loud():
    sample = Sample(tokens=[10, 11, 12], response_length=2)
    sample.reward = {"meta_info": {"input_token_logprobs": [None, [-0.2, 11], [-0.3, 12]]}}

    with pytest.raises(ValueError, match="has no observed task reward"):
        opd.post_process_rewards(
            Namespace(opd_log_prob_top_k=0, opd_log_task_reward=True, reward_key=""),
            [sample],
        )
