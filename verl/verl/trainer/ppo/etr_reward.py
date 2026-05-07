# Copyright 2025 the VERL authors
#
# Entropy Trend Reward (ETR), arXiv:2604.05355 — trajectory-level entropy shaping for concise CoT,
# combined with GRPO outcome rewards. This module applies the paper's scalar reward to the terminal
# response token after rule-based correctness is known and per-token policy entropy is available.
"""ETR (Entropy Trend Reward) shaping for GRPO-style outcome rewards.

Reference: Xiong et al., "ETR: Entropy Trend Reward for Efficient Chain-of-Thought Reasoning",
https://arxiv.org/abs/2604.05355

VERL integration notes
----------------------
- Correctness is still produced by ``NaiveRewardManager`` + ``compute_score``. After **old_log_prob writes**
  ``batch['token_entropy']``, this module replaces the scalar reward on the **last response token** using the
  paper-style formulation: ``R = -1`` (incorrect), ``R = 1 + λ * R_entropy`` (correct).
- ``R_entropy`` uses momentum-accumulated inter-segment entropy differences (paper Eq. (7)(8)).
  ``H_t`` is the mean token entropy within each "reasoning segment".
  Segmentation modes:
  - ``newline``: match the token-id pattern from ``tokenizer.encode('\\n\\n')`` directly on response token ids (no decode).
    Falls back to ``chunk`` if fewer than 2 non-empty segments are found.
  - ``chunk``: fixed-size segments of ``chunk_tokens`` tokens.

This file intentionally depends only on torch / numpy / typing for easier testing and static checking.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch


def momentum_entropy_reward_from_H(H: list[float], gamma: float) -> float:
    """Paper Eq. (5)-(8): S_1=0, S_t = γ S_{t-1} + Δ_t, Δ_t = H_{t-1}-H_t, R_entropy = Σ_{t=2}^T S_t."""
    if len(H) < 2:
        return 0.0
    g = float(gamma)
    if not (0.0 < g < 1.0) or not all(math.isfinite(x) for x in H):
        return 0.0
    s = 0.0
    acc = 0.0
    for tau in range(2, len(H) + 1):
        delta = float(H[tau - 2]) - float(H[tau - 1])
        s = g * s + delta
        acc += s
    return float(acc)


_NEWLINE_PATTERN_IDS_CACHE: dict[int, list[int]] = {}


def _newline_pattern_ids(tokenizer: Any) -> list[int]:
    """Token ids for the literal ``\\n\\n`` under this tokenizer (no decode of rollout needed)."""
    tid = id(tokenizer)
    if tid not in _NEWLINE_PATTERN_IDS_CACHE:
        _NEWLINE_PATTERN_IDS_CACHE[tid] = list(tokenizer.encode("\n\n", add_special_tokens=False))
    return _NEWLINE_PATTERN_IDS_CACHE[tid]


def _split_ids_by_subsequence(ids: list[int], pat: list[int]) -> list[tuple[int, int]]:
    """Non-overlapping splits: segments are token ranges *between* occurrences of ``pat`` (``pat`` excluded)."""
    n = len(ids)
    m = len(pat)
    if m == 0:
        return [(0, n)]
    out: list[tuple[int, int]] = []
    start = 0
    i = 0
    while i <= n - m:
        if ids[i : i + m] == pat:
            out.append((start, i))
            start = i + m
            i = start
        else:
            i += 1
    out.append((start, n))
    return out


def _segment_ranges_fixed(resp_len: int, chunk_tokens: int) -> list[tuple[int, int]]:
    if resp_len <= 0 or chunk_tokens <= 0:
        return []
    out: list[tuple[int, int]] = []
    s = 0
    while s < resp_len:
        e = min(s + chunk_tokens, resp_len)
        out.append((s, e))
        s = e
    return out


def _segment_ranges_newline(
    tokenizer: Any,
    response_ids: torch.Tensor,
    resp_len: int,
) -> list[tuple[int, int]] | None:
    """Split response token ids at occurrences of the encoded ``\\n\\n`` pattern. Returns None → use chunk fallback."""
    if resp_len <= 0:
        return []
    pat = _newline_pattern_ids(tokenizer)
    if not pat:
        return None
    ids = response_ids[:resp_len].detach().cpu().tolist()
    raw_ranges = _split_ids_by_subsequence(ids, pat)
    nonempty = [(s, e) for s, e in raw_ranges if e > s]
    if len(nonempty) < 2:
        return None
    return nonempty


def _mean_H_per_segment(ent_1d: torch.Tensor, ranges: list[tuple[int, int]]) -> list[float]:
    H: list[float] = []
    for s, e in ranges:
        if e <= s:
            continue
        seg = ent_1d[s:e].float()
        if seg.numel() == 0:
            continue
        v = seg.mean().item()
        if math.isfinite(v):
            H.append(float(v))
    return H


def _row_correct(batch: Any, i: int, seq_score: float) -> bool:
    acc = batch.non_tensor_batch.get("acc")
    if acc is not None:
        v = acc[i]
        if isinstance(v, np.ndarray):
            if v.shape == ():
                v = v.item()
            else:
                v = float(v.flat[0])
        return bool(v) if isinstance(v, (bool, np.bool_)) else float(v) > 0.5
    return float(seq_score) > 0.5


def compute_etr_scalar_for_row(
    tokenizer: Any,
    response_ids_1d: torch.Tensor,
    ent_1d: torch.Tensor,
    *,
    segment_mode: str,
    chunk_tokens: int,
    gamma: float,
    lambda_coef: float,
    correct: bool,
    clip_r_entropy: float | None,
    correct_base: float,
    incorrect_reward: float,
) -> tuple[float, float]:
    """Returns (terminal_scalar_reward, r_entropy_before_clip_or_same)."""
    resp_len = int(response_ids_1d.shape[0])
    if resp_len <= 0:
        return (incorrect_reward if not correct else correct_base, 0.0)

    mode = (segment_mode or "chunk").strip().lower()
    ranges: list[tuple[int, int]] | None = None
    if mode == "newline":
        ranges = _segment_ranges_newline(tokenizer, response_ids_1d, resp_len)
    if ranges is None:
        ranges = _segment_ranges_fixed(resp_len, max(1, int(chunk_tokens)))

    H = _mean_H_per_segment(ent_1d[:resp_len], ranges)
    r_ent = momentum_entropy_reward_from_H(H, gamma)
    if clip_r_entropy is not None and clip_r_entropy > 0:
        r_ent = float(np.clip(r_ent, -clip_r_entropy, clip_r_entropy))

    if not correct:
        return float(incorrect_reward), float(r_ent)
    lam = float(lambda_coef)
    return float(correct_base + lam * r_ent), float(r_ent)


def apply_etr_shaping_to_token_scores(
    batch: Any,
    tokenizer: Any,
    etr_cfg: dict[str, Any],
) -> dict[str, float]:
    """Mutates ``batch.batch['token_level_scores']`` in place (terminal token only).

    Preconditions: ``token_entropy`` and ``response_mask`` on batch; scores already placed on last
    valid response token (NaiveRewardManager convention).
    """
    if "token_entropy" not in batch.batch:
        raise ValueError("ETR requires batch['token_entropy'] (enable ETR in ray_trainer or use GRPO-S path).")
    scores = batch.batch["token_level_scores"]
    resp_mask = batch.batch["response_mask"].bool()
    ent = batch.batch["token_entropy"]
    responses = batch.batch["responses"]

    gamma = float(etr_cfg.get("gamma", 0.9))
    lam = float(etr_cfg.get("lambda_coef", etr_cfg.get("lambda", 0.1)))
    segment_mode = str(etr_cfg.get("segment_mode", "chunk"))
    chunk_tokens = int(etr_cfg.get("chunk_tokens", 32))
    clip_r = etr_cfg.get("clip_r_entropy", None)
    clip_r_f = float(clip_r) if clip_r is not None else None
    correct_base = float(etr_cfg.get("correct_base_reward", 1.0))
    incorrect_reward = float(etr_cfg.get("incorrect_reward", -1.0))

    bsz = int(scores.shape[0])
    r_ent_list: list[float] = []
    new_terminal: list[float] = []

    for i in range(bsz):
        seq_score = float((scores[i] * resp_mask[i].float()).sum().item())
        correct = _row_correct(batch, i, seq_score)
        rl = int(resp_mask[i].sum().item())
        if rl <= 0:
            new_terminal.append(float(incorrect_reward if not correct else correct_base))
            r_ent_list.append(0.0)
            continue
        last_idx = rl - 1
        r_seq, r_e = compute_etr_scalar_for_row(
            tokenizer,
            responses[i],
            ent[i],
            segment_mode=segment_mode,
            chunk_tokens=chunk_tokens,
            gamma=gamma,
            lambda_coef=lam,
            correct=correct,
            clip_r_entropy=clip_r_f,
            correct_base=correct_base,
            incorrect_reward=incorrect_reward,
        )
        scores[i].zero_()
        scores[i, last_idx] = r_seq
        new_terminal.append(r_seq)
        r_ent_list.append(r_e)

    out: dict[str, float] = {
        "etr/terminal_reward_mean": float(np.mean(new_terminal)) if new_terminal else 0.0,
        "etr/r_entropy_mean": float(np.mean(r_ent_list)) if r_ent_list else 0.0,
    }
    return out
