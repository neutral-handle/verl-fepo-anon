# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""FEPO: entropy / continuation-F helpers (top-k logprobs from vLLM)."""

from __future__ import annotations

import math
import os
from typing import Any

import numpy as np


def vllm_max_logprobs_allowed() -> int:
    return int(os.environ.get("VLLM_ENTROPY_MAX_LOGPROBS", "20"))


def clamp_vllm_logprobs_topk(requested: int) -> int:
    cap = vllm_max_logprobs_allowed()
    return min(max(0, int(requested)), cap)


def _sanitize_logprob_values(lps: np.ndarray) -> np.ndarray:
    return np.nan_to_num(lps, nan=-1e30, neginf=-1e30, posinf=0.0)


def entropy_from_logprobs_topk(logprobs_dict: dict[int, float]) -> float:
    """Entropy over renormalized top-K logprobs (vLLM); not full-vocab softmax."""
    if not logprobs_dict:
        return 0.0
    lps = _sanitize_logprob_values(np.array(list(logprobs_dict.values()), dtype=np.float64))
    if not np.any(np.isfinite(lps)):
        return 0.0
    m = float(np.max(lps))
    p = np.exp(np.clip(lps - m, -80.0, 0.0))
    z = float(p.sum())
    if z <= 0.0 or not np.isfinite(z):
        return 0.0
    p = p / z
    return float(-np.sum(p * np.log(np.clip(p, 1e-20, 1.0))))


def vllm_request_step_logprobs_to_float_dicts(o: Any) -> list[dict[int, float]]:
    out: list[dict[int, float]] = []
    for step_lp in o.logprobs or []:
        d: dict[int, float] = {}
        for tid, info in step_lp.items():
            d[int(tid)] = float(info.logprob)
        out.append(d)
    return out


def continuation_F_from_gen_ids_and_step_logprobs(
    gen_ids: list[int],
    step_lps_float: list[dict[int, float]],
    *,
    f_continuation_mode: str,
    tokenizer: Any | None,
    stop_fn: Any | None,
    normalize_by_continuation_length: bool,
) -> float:
    """Sentence-level or full continuation F; aligned with entropy_ce experiment."""
    if f_continuation_mode == "first_sentence":
        if tokenizer is None:
            raise ValueError("tokenizer is required when f_continuation_mode='first_sentence'")
        if stop_fn is None:
            raise ValueError("stop_fn is required when f_continuation_mode='first_sentence'")
        from verl.utils.fepo_sentence_stop import truncate_gen_ids_to_first_sentence

        padded = list(step_lps_float)
        while len(padded) < len(gen_ids):
            padded.append({})
        keep_k = truncate_gen_ids_to_first_sentence(gen_ids, tokenizer, stop_fn)
        raw_sum = float(sum(entropy_from_logprobs_topk(padded[i]) for i in range(keep_k)))
        denom = max(int(keep_k), 1)
        return raw_sum / float(denom) if normalize_by_continuation_length else raw_sum
    if normalize_by_continuation_length:
        entropies = [entropy_from_logprobs_topk(s) for s in step_lps_float]
        n = len(entropies)
        return float(sum(entropies)) / float(max(n, 1))
    return float(sum(entropy_from_logprobs_topk(s) for s in step_lps_float))
