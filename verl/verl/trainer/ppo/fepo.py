# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""FEPO: token-level future-entropy advantage shaping (adds sparse bonus to ``advantages``)."""

from __future__ import annotations

import math
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Optional

import numpy as np
import ray
import torch

from verl import DataProto


_SUFFIX_LEN_DEBIAS_STATE: dict[str, Any] = {
    "bins": {},  # bin_id -> {"count": int, "mean": float, "m2": float}
    "global": {"count": 0, "mean": 0.0, "m2": 0.0},
}


def _rankdata_average(x: np.ndarray) -> np.ndarray:
    """Average-tie rankdata (1-based), scipy-free."""
    n = int(x.shape[0])
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(n, dtype=np.float64)
    i = 0
    while i < n:
        j = i + 1
        xi = x[order[i]]
        while j < n and x[order[j]] == xi:
            j += 1
        # average rank in [i+1, j]
        avg_rank = 0.5 * ((i + 1) + j)
        ranks[order[i:j]] = avg_rank
        i = j
    return ranks


def _spearmanr_safe(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman correlation with tie-aware ranks; returns nan when undefined."""
    if x.size < 3 or y.size < 3:
        return float("nan")
    rx = _rankdata_average(x)
    ry = _rankdata_average(y)
    sx = np.std(rx)
    sy = np.std(ry)
    if sx <= 0.0 or sy <= 0.0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def _welford_update(state: dict[str, float], x: float) -> None:
    c = int(state.get("count", 0)) + 1
    mean = float(state.get("mean", 0.0))
    m2 = float(state.get("m2", 0.0))
    delta = x - mean
    mean = mean + delta / float(c)
    delta2 = x - mean
    m2 = m2 + delta * delta2
    state["count"] = c
    state["mean"] = mean
    state["m2"] = m2


def _welford_var(state: dict[str, float]) -> float:
    c = int(state.get("count", 0))
    if c <= 1:
        return 0.0
    return float(state.get("m2", 0.0)) / float(c - 1)


def _suffix_len_bin_id(length: int, bin_width: int) -> int:
    bw = max(int(bin_width), 1)
    l = max(int(length), 1)
    return (l - 1) // bw


def _pre_high_mask_top_entropy_ratio(
    entropy: torch.Tensor,
    response_mask: torch.Tensor,
    top_ratio: float,
) -> torch.Tensor:
    """Per sequence, mark the top ``ceil(n_valid * top_ratio)`` highest-entropy response tokens."""
    r = float(np.clip(top_ratio, 0.0, 1.0))
    out = torch.zeros_like(response_mask, dtype=torch.bool)
    B = int(entropy.size(0))
    for b in range(B):
        mask_b = response_mask[b].bool()
        valid_idx = torch.where(mask_b)[0]
        n = int(valid_idx.numel())
        if n == 0:
            continue
        k = max(1, int(math.ceil(n * r)))
        k = min(k, n)
        vals = entropy[b, valid_idx].float()
        _, sub_i = torch.topk(vals, k=k, largest=True)
        chosen = valid_idx[sub_i]
        out[b, chosen] = True
    return out


def _prompt_group_indices(prompts: torch.Tensor) -> dict[bytes, list[int]]:
    """Group batch rows by identical prompt token ids."""
    p = prompts.detach().cpu().contiguous()
    groups: dict[bytes, list[int]] = defaultdict(list)
    for i in range(int(p.size(0))):
        key = p[i].numpy().tobytes()
        groups[key].append(i)
    return groups


def _suffix_entropy_rate_exclusive(entropy: torch.Tensor, response_mask: torch.Tensor) -> torch.Tensor:
    """Realized suffix entropy rate F_t = mean_{k>t} H_k on valid response range; NaN if undefined."""
    B, T = entropy.shape
    out = torch.full((B, T), float("nan"), device=entropy.device, dtype=entropy.dtype)
    for b in range(B):
        m = response_mask[b].bool()
        n = int(m.sum().item())
        if n <= 1:
            continue
        row = entropy[b, :n].float()
        # suffix including current: S_t = H_t + ... + H_{n-1}
        s_incl = torch.flip(torch.cumsum(torch.flip(row, dims=[0]), dim=0), dims=[0])
        s_excl = s_incl - row
        den = torch.arange(n - 1, -1, -1, device=row.device, dtype=row.dtype)  # n-t-1
        valid = den > 0
        vals = torch.full((n,), float("nan"), device=row.device, dtype=row.dtype)
        vals[valid] = s_excl[valid] / den[valid]
        out[b, :n] = vals.to(dtype=entropy.dtype)
    return out


def _simple_sentence_stop_check(text: str, *, min_chars: int = 2) -> bool:
    t = text
    if len(t.strip()) < min_chars:
        return False
    if "\n\n" in t:
        return True
    tt = t.rstrip()
    if not tt:
        return False
    if tt[-1] in "。！？!?":
        return True
    # Treat '.' as sentence end unless it's likely decimal tail.
    if tt[-1] == ".":
        prev = tt[:-1].rstrip()
        if prev and prev[-1].isdigit():
            return False
        return True
    return False


def _make_sentence_stop_check(mode: str) -> Callable[[str], bool]:
    if str(mode) == "pysbd":
        try:
            from sentence_stop_utils import make_pysbd_first_sentence_stop_check

            return make_pysbd_first_sentence_stop_check()
        except Exception:
            return _simple_sentence_stop_check
    return _simple_sentence_stop_check


def _suffix_entropy_rate_to_sentence_end(
    entropy: torch.Tensor,
    responses: torch.Tensor,
    response_mask: torch.Tensor,
    *,
    tokenizer: Any,
    sentence_stop_check: Callable[[str], bool],
    min_suffix_tokens: int,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Sentence-bounded suffix entropy-rate from t+1 to sentence end.

    Returns:
      - rate tensor [B,T], NaN where undefined / filtered
      - keep_k tensor [B,T], number of kept suffix tokens for each t
    """
    B, T = entropy.shape
    out = torch.full((B, T), float("nan"), device=entropy.device, dtype=entropy.dtype)
    keep = torch.zeros((B, T), device=entropy.device, dtype=torch.int32)
    n_filtered_short = 0
    min_k = max(int(min_suffix_tokens), 1)
    ent_cpu = entropy.detach().cpu()
    resp_cpu = responses.detach().cpu()
    mask_cpu = response_mask.detach().cpu().bool()
    for b in range(B):
        n = int(mask_cpu[b].sum().item())
        if n <= 1:
            continue
        row_ent = ent_cpu[b, :n]
        row_resp = [int(x) for x in resp_cpu[b, :n].tolist()]
        for t in range(n - 1):
            suffix_ids = row_resp[t + 1 :]
            if not suffix_ids:
                continue
            k = 0
            for kk in range(1, len(suffix_ids) + 1):
                frag = tokenizer.decode(suffix_ids[:kk], skip_special_tokens=True)
                if sentence_stop_check(frag):
                    k = kk
                    break
            if k <= 0:
                k = len(suffix_ids)
            if k < min_k:
                n_filtered_short += 1
                continue
            vals = row_ent[t + 1 : t + 1 + k].float()
            out[b, t] = torch.mean(vals).to(dtype=entropy.dtype)
            keep[b, t] = int(k)
    return out, keep, int(n_filtered_short)


def _suffix_entropy_rate_to_sentence_end_simple_fast(
    entropy: torch.Tensor,
    responses: torch.Tensor,
    response_mask: torch.Tensor,
    *,
    tokenizer: Any,
    min_suffix_tokens: int,
    num_threads: int = 1,
    eligible_mask: Optional[torch.Tensor] = None,
    max_scan_tokens: int = 256,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Fast path for `simple` sentence stop mode.

    Strategy:
    - Decode each token once to text piece.
    - Mark token positions that can terminate a sentence (heuristic aligned with simple mode).
    - Precompute nearest sentence-end index to the right in O(T).
    - Use prefix-sum on entropy to compute mean suffix entropy in O(1) per token.
    """
    B, T = entropy.shape
    out_cpu = torch.full((B, T), float("nan"), dtype=entropy.dtype, device="cpu")
    keep_cpu = torch.zeros((B, T), dtype=torch.int32, device="cpu")
    n_filtered_short = 0
    min_k = max(int(min_suffix_tokens), 1)
    ent_cpu = entropy.detach().cpu().float()
    resp_cpu = responses.detach().cpu()
    mask_cpu = response_mask.detach().cpu().bool()
    eligible_cpu = eligible_mask.detach().cpu().bool() if eligible_mask is not None else None
    row_need_n: dict[int, int] = {}
    # Decode once for all unique token ids in current batch to avoid
    # high lock contention and repeated tokenizer calls across threads.
    unique_ids: set[int] = set()
    active_rows: list[int] = []
    for b in range(B):
        n = int(mask_cpu[b].sum().item())
        if n <= 0:
            continue
        n_need = n
        if eligible_cpu is not None:
            # Skip rows that have no eligible positions at all.
            elig = torch.where(eligible_cpu[b, : max(n - 1, 0)])[0]
            if elig.numel() == 0:
                continue
            # Only decode/scan prefix up to the furthest needed candidate window.
            if max_scan_tokens > 0:
                max_t = int(torch.max(elig).item())
                n_need = min(n, max_t + 1 + max_scan_tokens)
        row_need_n[b] = n_need
        active_rows.append(b)
        unique_ids.update(int(x) for x in resp_cpu[b, :n_need].tolist())
    decode_cache: dict[int, str] = {
        tid: tokenizer.decode([tid], skip_special_tokens=True) for tid in unique_ids
    }
    # Precompute per-token-id stop flag (approximation of simple mode).
    # Ignore the decimal-tail safeguard here: rare and the tradeoff favors speed.
    stop_id_cache: dict[int, bool] = {}
    for tid, p in decode_cache.items():
        p_strip = p.rstrip()
        if not p_strip:
            stop_id_cache[tid] = False
            continue
        tail = p_strip[-1]
        if "\n\n" in p or tail in "。！？!?.":
            stop_id_cache[tid] = True
        else:
            stop_id_cache[tid] = False

    def _process_one_row(b: int) -> int:
        local_filtered = 0
        n = int(mask_cpu[b].sum().item())
        if n <= 1:
            return local_filtered
        n_need = int(row_need_n.get(b, n))
        if n_need <= 1:
            return local_filtered
        row_ent = ent_cpu[b, :n_need]
        row_resp_tensor = resp_cpu[b, :n_need]
        row_resp = row_resp_tensor.tolist()

        # Vectorized stop via cache.
        stop = [stop_id_cache.get(int(tid), False) for tid in row_resp]

        # nearest stop index >= i, else -1
        next_stop = [-1] * n_need
        nxt = -1
        for i in range(n_need - 1, -1, -1):
            if stop[i]:
                nxt = i
            next_stop[i] = nxt

        # Prefix sum for O(1) segment mean.
        csum = torch.zeros((n_need + 1,), dtype=row_ent.dtype)
        csum[1:] = torch.cumsum(row_ent, dim=0)

        for t in range(n_need - 1):
            if eligible_cpu is not None and not bool(eligible_cpu[b, t].item()):
                continue
            s = t + 1
            e = next_stop[s]
            if e < 0:
                e = n_need - 1
            if max_scan_tokens > 0:
                e = min(e, s + max_scan_tokens - 1, n_need - 1)
            k = e - s + 1
            if k < min_k:
                local_filtered += 1
                continue
            seg_sum = csum[e + 1] - csum[s]
            out_cpu[b, t] = (seg_sum / float(k)).to(dtype=entropy.dtype)
            keep_cpu[b, t] = int(k)
        return local_filtered

    workers = max(int(num_threads), 1)
    if workers == 1:
        for b in active_rows:
            n_filtered_short += _process_one_row(b)
    else:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            for local_filtered in ex.map(_process_one_row, active_rows):
                n_filtered_short += int(local_filtered)

    out = out_cpu.to(device=entropy.device)
    keep = keep_cpu.to(device=entropy.device)
    return out, keep, int(n_filtered_short)


def _suffix_entropy_rate_fixed_window(
    entropy: torch.Tensor,
    response_mask: torch.Tensor,
    *,
    window_tokens: int,
    min_suffix_tokens: int,
    eligible_mask: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Fixed-window suffix entropy-rate from t+1 to t+window_tokens."""
    B, T = entropy.shape
    out_cpu = torch.full((B, T), float("nan"), dtype=entropy.dtype, device="cpu")
    keep_cpu = torch.zeros((B, T), dtype=torch.int32, device="cpu")
    n_filtered_short = 0
    min_k = max(int(min_suffix_tokens), 1)
    w = max(int(window_tokens), 1)
    ent_cpu = entropy.detach().cpu().float()
    mask_cpu = response_mask.detach().cpu().bool()
    eligible_cpu = eligible_mask.detach().cpu().bool() if eligible_mask is not None else None

    for b in range(B):
        n = int(mask_cpu[b].sum().item())
        if n <= 1:
            continue
        row_ent = ent_cpu[b, :n]
        csum = torch.zeros((n + 1,), dtype=row_ent.dtype)
        csum[1:] = torch.cumsum(row_ent, dim=0)
        for t in range(n - 1):
            if eligible_cpu is not None and not bool(eligible_cpu[b, t].item()):
                continue
            s = t + 1
            e = min(n - 1, t + w)
            k = e - s + 1
            if k < min_k:
                n_filtered_short += 1
                continue
            seg_sum = csum[e + 1] - csum[s]
            out_cpu[b, t] = (seg_sum / float(k)).to(dtype=entropy.dtype)
            keep_cpu[b, t] = int(k)

    out = out_cpu.to(device=entropy.device)
    keep = keep_cpu.to(device=entropy.device)
    return out, keep, int(n_filtered_short)


def _compute_suffix_len_full_mode(response_mask: torch.Tensor) -> torch.Tensor:
    """Exclusive future suffix length for full mode: L_t = (#valid after t)."""
    B, T = response_mask.shape
    out = torch.zeros((B, T), dtype=torch.int32, device=response_mask.device)
    m_cpu = response_mask.detach().cpu().bool()
    out_cpu = out.detach().cpu()
    for b in range(B):
        n = int(m_cpu[b].sum().item())
        if n <= 1:
            continue
        vals = torch.arange(n - 1, -1, -1, dtype=torch.int32)  # n-t-1
        out_cpu[b, :n] = vals
    return out_cpu.to(device=response_mask.device)


def _build_suffix_len_debiased_score(
    *,
    raw_f_rate: torch.Tensor,
    high_mask: torch.Tensor,
    suffix_len: torch.Tensor,
    enable: bool,
    bin_width: int,
    min_count: int,
    z_clip: float,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, int, int]:
    """Length-binned z-score using previous-step stats; then update stats with current batch."""
    rank_score = raw_f_rate.float().clone()
    if not enable:
        return rank_score, 0, 0

    valid = high_mask & torch.isfinite(raw_f_rate) & (suffix_len > 0)
    idx = torch.nonzero(valid, as_tuple=False)
    if idx.numel() == 0:
        return rank_score, 0, 0

    state = _SUFFIX_LEN_DEBIAS_STATE
    bins_state: dict[int, dict[str, float]] = state["bins"]
    g_state: dict[str, float] = state["global"]
    g_count = int(g_state.get("count", 0))
    g_mean = float(g_state.get("mean", 0.0))
    g_var = _welford_var(g_state)

    n_bin_fallback = 0
    n_global_fallback = 0
    for p in idx.tolist():
        b, t = int(p[0]), int(p[1])
        l = int(suffix_len[b, t].item())
        bid = _suffix_len_bin_id(l, bin_width)
        x = float(raw_f_rate[b, t].item())
        b_state = bins_state.get(bid)
        use_bin = b_state is not None and int(b_state.get("count", 0)) >= int(min_count)
        if use_bin:
            mu = float(b_state.get("mean", 0.0))
            var = _welford_var(b_state)
        elif g_count >= int(min_count):
            mu = g_mean
            var = g_var
            n_global_fallback += 1
        else:
            mu = 0.0
            var = -1.0  # mark unavailable
            n_bin_fallback += 1

        if var >= 0.0:
            z = (x - mu) / float(np.sqrt(var + eps))
            if z_clip > 0:
                z = float(np.clip(z, -z_clip, z_clip))
            rank_score[b, t] = z

    # Update stats AFTER scoring current batch (avoid self-leakage).
    for p in idx.tolist():
        b, t = int(p[0]), int(p[1])
        l = int(suffix_len[b, t].item())
        bid = _suffix_len_bin_id(l, bin_width)
        x = float(raw_f_rate[b, t].item())
        if bid not in bins_state:
            bins_state[bid] = {"count": 0, "mean": 0.0, "m2": 0.0}
        _welford_update(bins_state[bid], x)
        _welford_update(g_state, x)

    return rank_score, int(n_bin_fallback), int(n_global_fallback)


def _run_fepo_lowtail_adv_phase(
    batch: DataProto,
    tokenizer: Any,
    fepo_cfg: dict[str, Any],
) -> tuple[DataProto, dict[str, float], list[dict[str, Any]]]:
    """New FEPO v2: direction from A_t, strength from low-tail suffix entropy-rate quantile.

    On low-tail positions (``q <= beta``): ``adv > 0`` uses ``m = 1 + alpha`` (strengthen);
    ``adv < 0`` uses ``m = max(0, 1 - low_tail_neg_adv_penalty)`` when ``low_tail_neg_adv_penalty > 0``;
    otherwise it preserves the legacy behavior (can be combined with ``low_tail_pos_adv_only``).
    """
    ent_key = "fepo_token_entropy" if "fepo_token_entropy" in batch.batch else "token_entropy"
    if ent_key not in batch.batch:
        raise ValueError(
            "FEPO lowtail_adv requires per-token entropy in batch['fepo_token_entropy'] (preferred) "
            "or batch['token_entropy']."
        )
    entropy = batch.batch[ent_key]
    response_mask = batch.batch["response_mask"].bool()
    advantages = batch.batch["advantages"]
    prompts = batch.batch["prompts"]

    h_threshold = float(fepo_cfg.get("h_threshold", 2.0))
    alpha = float(fepo_cfg.get("alpha", fepo_cfg.get("bonus_pos", 0.2)))
    beta = float(fepo_cfg.get("beta", 0.2))
    high_head_penalty = float(fepo_cfg.get("high_head_penalty", 0.0))
    # For high-head points with adv<0: multiply advantage by (1 + high_head_neg_adv_boost). Default 0 disables.
    high_head_neg_adv_boost = float(fepo_cfg.get("high_head_neg_adv_boost", 0.0))
    suffix_mode = str(fepo_cfg.get("suffix_mode", "full"))  # full / sentence
    f_sentence_stop = str(fepo_cfg.get("f_sentence_stop", "simple"))
    sentence_min_suffix_tokens = int(fepo_cfg.get("sentence_min_suffix_tokens", 5))
    sentence_num_threads = int(fepo_cfg.get("sentence_num_threads", 1))
    sentence_only_high_entropy = bool(fepo_cfg.get("sentence_only_high_entropy", True))
    sentence_max_scan_tokens = int(fepo_cfg.get("sentence_max_scan_tokens", 256))
    sentence_high_entropy_ratio = float(fepo_cfg.get("sentence_high_entropy_ratio", 0.01))
    fixed_window_tokens = int(fepo_cfg.get("fixed_window_tokens", 32))
    rank_scope = str(fepo_cfg.get("rank_scope", "group")).strip().lower()
    if rank_scope not in {"group", "batch"}:
        rank_scope = "group"
    low_tail_pos_adv_only = bool(fepo_cfg.get("low_tail_pos_adv_only", False))
    # For low-tail points with adv<0: multiply advantage by max(0, 1 - low_tail_neg_adv_penalty). Default 0 keeps legacy behavior.
    low_tail_neg_adv_penalty = float(fepo_cfg.get("low_tail_neg_adv_penalty", 0.0))
    high_head_neg_adv_only = bool(fepo_cfg.get("high_head_neg_adv_only", False))
    high_head_pos_adv_only = bool(fepo_cfg.get("high_head_pos_adv_only", False))
    suffix_len_debias_enable = bool(fepo_cfg.get("suffix_len_debias_enable", False))
    suffix_len_bin_width = int(fepo_cfg.get("suffix_len_bin_width", 2))
    suffix_len_min_count = int(fepo_cfg.get("suffix_len_min_count", 100))
    suffix_len_z_clip = float(fepo_cfg.get("suffix_len_z_clip", 3.0))
    alpha = max(alpha, 0.0)
    high_head_penalty = max(high_head_penalty, 0.0)
    high_head_neg_adv_boost = max(high_head_neg_adv_boost, 0.0)
    beta = float(np.clip(beta, 0.0, 1.0))
    low_tail_neg_adv_penalty = float(np.clip(low_tail_neg_adv_penalty, 0.0, 1.0))
    collect_point_records = bool(fepo_cfg.get("__collect_point_records", True))
    offpolicy_diag_enable = bool(fepo_cfg.get("offpolicy_diag_enable", True))

    high_entropy_select_mode = str(fepo_cfg.get("high_entropy_select_mode", "threshold")).strip().lower()
    if high_entropy_select_mode not in {"threshold", "top_ratio"}:
        high_entropy_select_mode = "threshold"
    high_entropy_top_ratio = float(fepo_cfg.get("high_entropy_top_ratio", 0.1))

    # Realized suffix entropy-rate per token (exclusive future).
    # threshold: keep tokens with entropy >= h_threshold, then cap per-seq by sentence_high_entropy_ratio.
    # top_ratio: no fixed entropy cutoff; per sequence take top ceil(n_valid * ratio) tokens by entropy.
    if high_entropy_select_mode == "top_ratio":
        high_entropy_top_ratio = float(np.clip(high_entropy_top_ratio, 1e-6, 1.0))
        pre_high_mask = _pre_high_mask_top_entropy_ratio(entropy, response_mask, high_entropy_top_ratio)
        eligible_cap_mask = pre_high_mask.clone()
    else:
        pre_high_mask = (entropy >= h_threshold) & response_mask
        eligible_cap_mask = torch.zeros_like(response_mask, dtype=torch.bool)
        ratio = max(float(sentence_high_entropy_ratio), 0.0)
        for b in range(int(entropy.size(0))):
            high_ts = torch.where(pre_high_mask[b])[0]
            if high_ts.numel() == 0:
                continue
            resp_len = int(response_mask[b].sum().item())
            if ratio > 0.0:
                k_cap = max(1, int(resp_len * ratio))
                k = min(int(high_ts.numel()), k_cap)
            else:
                k = int(high_ts.numel())
            if k <= 0:
                continue
            vals = entropy[b, high_ts].float()
            _, top_idx = torch.topk(vals, k=k, largest=True)
            chosen = high_ts[top_idx]
            eligible_cap_mask[b, chosen] = True

    keep_k = torch.zeros_like(advantages, dtype=torch.int32)
    if suffix_mode == "sentence":
        responses = batch.batch["responses"]
        eligible_mask = eligible_cap_mask if sentence_only_high_entropy else None
        if str(f_sentence_stop) == "simple":
            f_rate, keep_k, n_filtered_short_sentence = _suffix_entropy_rate_to_sentence_end_simple_fast(
                entropy,
                responses,
                response_mask,
                tokenizer=tokenizer,
                min_suffix_tokens=sentence_min_suffix_tokens,
                num_threads=sentence_num_threads,
                eligible_mask=eligible_mask,
                max_scan_tokens=sentence_max_scan_tokens,
            )
        else:
            stop_check = _make_sentence_stop_check(f_sentence_stop)
            f_rate, keep_k, n_filtered_short_sentence = _suffix_entropy_rate_to_sentence_end(
                entropy,
                responses,
                response_mask,
                tokenizer=tokenizer,
                sentence_stop_check=stop_check,
                min_suffix_tokens=sentence_min_suffix_tokens,
            )
    elif suffix_mode == "fixed_window":
        eligible_mask = eligible_cap_mask if sentence_only_high_entropy else None
        f_rate, keep_k, n_filtered_short_sentence = _suffix_entropy_rate_fixed_window(
            entropy,
            response_mask,
            window_tokens=fixed_window_tokens,
            min_suffix_tokens=sentence_min_suffix_tokens,
            eligible_mask=eligible_mask,
        )
    else:
        f_rate = _suffix_entropy_rate_exclusive(entropy, response_mask)
        n_filtered_short_sentence = 0
    high_mask = pre_high_mask & torch.isfinite(f_rate) & eligible_cap_mask
    if suffix_mode in {"sentence", "fixed_window"}:
        suffix_len = keep_k
    else:
        suffix_len = _compute_suffix_len_full_mode(response_mask)
    rank_score, n_len_bin_fallback, n_len_global_fallback = _build_suffix_len_debiased_score(
        raw_f_rate=f_rate,
        high_mask=high_mask,
        suffix_len=suffix_len,
        enable=suffix_len_debias_enable,
        bin_width=suffix_len_bin_width,
        min_count=suffix_len_min_count,
        z_clip=suffix_len_z_clip,
    )
    if high_entropy_select_mode == "top_ratio":
        low_mask = response_mask & ~pre_high_mask
    else:
        low_mask = (entropy < h_threshold) & response_mask
    m = torch.ones_like(advantages)
    q = torch.full_like(advantages, float("nan"))

    groups = _prompt_group_indices(prompts) if rank_scope == "group" else {b"__batch__": list(range(int(entropy.size(0))))}
    n_high = 0
    n_boost = 0
    n_head_penalized = 0
    n_low_tail_gate_blocked = 0
    n_low_tail_neg_penalized = 0
    n_high_head_gate_blocked = 0
    n_high_head_neg_boosted = 0

    for idxs in groups.values():
        positions: list[tuple[int, int, float]] = []
        for b in idxs:
            ts = torch.where(high_mask[b])[0].tolist()
            for t in ts:
                positions.append((int(b), int(t), float(rank_score[b, t].item())))
        if not positions:
            continue
        positions_sorted = sorted(positions, key=lambda x: x[2])  # low-tail first
        k = len(positions_sorted)
        n_high += int(k)
        for rank, (b, t, _f) in enumerate(positions_sorted):
            qi = (float(rank) / float(k - 1)) if k > 1 else 0.0
            q[b, t] = qi
            adv_t = float(advantages[b, t].item())
            if qi <= beta:
                if adv_t > 0.0:
                    m[b, t] = 1.0 + alpha
                    n_boost += 1
                elif adv_t < 0.0:
                    if low_tail_neg_adv_penalty > 0.0:
                        m[b, t] = max(0.0, 1.0 - low_tail_neg_adv_penalty)
                        n_low_tail_neg_penalized += 1
                    elif not low_tail_pos_adv_only:
                        m[b, t] = 1.0 + alpha
                        n_boost += 1
                    else:
                        n_low_tail_gate_blocked += 1
                else:
                    if not low_tail_pos_adv_only:
                        m[b, t] = 1.0 + alpha
                        n_boost += 1
                    else:
                        n_low_tail_gate_blocked += 1
            elif qi >= (1.0 - beta):
                # High-head branch:
                # 1) Optional: if adv<0, apply high_head_neg_adv_boost (m > 1)
                # 2) Otherwise, apply the original high_head_penalty downweighting (m < 1)
                if adv_t < 0.0 and high_head_neg_adv_boost > 0.0:
                    m[b, t] = 1.0 + high_head_neg_adv_boost
                    n_high_head_neg_boosted += 1
                elif high_head_penalty > 0.0:
                    # high-head penalty: downweight very high-q positions.
                    # When both switches are enabled, positive-only takes precedence.
                    if high_head_pos_adv_only:
                        allow_high_head = adv_t > 0.0
                    elif high_head_neg_adv_only:
                        allow_high_head = adv_t < 0.0
                    else:
                        allow_high_head = True
                    if allow_high_head:
                        m[b, t] = max(0.0, 1.0 - high_head_penalty)
                        n_head_penalized += 1
                    else:
                        n_high_head_gate_blocked += 1

    # Effective split masks on selected high-entropy points.
    q_finite_mask = torch.isfinite(q)
    low_tail_mask = high_mask & q_finite_mask & (q <= beta)
    high_head_mask = high_mask & q_finite_mask & (q >= (1.0 - beta))
    adv_pos_mask = advantages > 0
    adv_neg_mask = advantages <= 0
    n_low_tail = int(low_tail_mask.sum().item())
    n_high_head = int(high_head_mask.sum().item())
    n_adv_pos_high = int((high_mask & adv_pos_mask).sum().item())
    n_adv_neg_high = int((high_mask & adv_neg_mask).sum().item())
    n_low_tail_adv_pos = int((low_tail_mask & adv_pos_mask).sum().item())
    adv_neg_strict_mask = advantages < 0
    n_low_tail_adv_neg_strict = int((low_tail_mask & adv_neg_strict_mask).sum().item())
    n_high_head_adv_pos = int((high_head_mask & adv_pos_mask).sum().item())

    batch.batch["advantages"] = advantages * m * response_mask.float()

    q_valid = q[torch.isfinite(q)]
    m_valid = m[high_mask]
    n_resp = int(response_mask.sum().item())
    n_low = int(low_mask.sum().item())
    low_entropy_vals = entropy[low_mask]
    n_suffix_used = (
        int((keep_k >= max(sentence_min_suffix_tokens, 1)).sum().item())
        if suffix_mode in {"sentence", "fixed_window"}
        else 0
    )
    # Distribution stats on valid suffix-rate positions.
    f_valid_all = f_rate[response_mask & torch.isfinite(f_rate)].float()
    f_valid_high = f_rate[high_mask].float()
    if suffix_mode == "sentence":
        len_valid = keep_k[keep_k > 0].float()
    else:
        len_valid = torch.zeros((0,), device=advantages.device, dtype=torch.float32)

    def _pct(x: torch.Tensor, qv: float) -> float:
        if x.numel() == 0:
            return float("nan")
        return float(torch.quantile(x, qv).item())

    # ------------------------------
    # Off-policy diagnostics:
    # (1) ratio distribution between old_log_probs and rollout_log_probs
    # (2) rank-stability proxy: Spearman between old-policy and rollout-policy
    #     suffix-rate ranks inside the selected high-entropy set
    # ------------------------------
    off_ratio_mean = float("nan")
    off_ratio_p10 = float("nan")
    off_ratio_p50 = float("nan")
    off_ratio_p90 = float("nan")
    off_log_ratio_abs_mean = float("nan")
    off_spearman_mean = float("nan")
    off_spearman_p10 = float("nan")
    off_spearman_p50 = float("nan")
    off_spearman_p90 = float("nan")
    off_spearman_n_prompts = 0.0

    if offpolicy_diag_enable and ("old_log_probs" in batch.batch) and ("rollout_log_probs" in batch.batch):
        old_lp = batch.batch["old_log_probs"].float()
        roll_lp = batch.batch["rollout_log_probs"].float()
        ratio_mask = response_mask & torch.isfinite(old_lp) & torch.isfinite(roll_lp)
        if ratio_mask.any():
            log_r = (old_lp - roll_lp)[ratio_mask]
            r = torch.exp(log_r)
            off_ratio_mean = float(torch.mean(r).item())
            off_ratio_p10 = _pct(r, 0.10)
            off_ratio_p50 = _pct(r, 0.50)
            off_ratio_p90 = _pct(r, 0.90)
            off_log_ratio_abs_mean = float(torch.mean(torch.abs(log_r)).item())

        # Build rollout-policy entropy proxy from rollout log-prob.
        # This is a policy-mismatch proxy: π_old vs π_anchor(old_log_prob policy).
        roll_ent = (-roll_lp).clamp(min=0.0)
        if suffix_mode == "sentence":
            responses = batch.batch["responses"]
            eligible_mask = eligible_cap_mask if sentence_only_high_entropy else None
            if str(f_sentence_stop) == "simple":
                f_rate_roll, _keep_roll, _ = _suffix_entropy_rate_to_sentence_end_simple_fast(
                    roll_ent,
                    responses,
                    response_mask,
                    tokenizer=tokenizer,
                    min_suffix_tokens=sentence_min_suffix_tokens,
                    num_threads=sentence_num_threads,
                    eligible_mask=eligible_mask,
                    max_scan_tokens=sentence_max_scan_tokens,
                )
            else:
                stop_check = _make_sentence_stop_check(f_sentence_stop)
                f_rate_roll, _keep_roll, _ = _suffix_entropy_rate_to_sentence_end(
                    roll_ent,
                    responses,
                    response_mask,
                    tokenizer=tokenizer,
                    sentence_stop_check=stop_check,
                    min_suffix_tokens=sentence_min_suffix_tokens,
                )
        elif suffix_mode == "fixed_window":
            eligible_mask = eligible_cap_mask if sentence_only_high_entropy else None
            f_rate_roll, _keep_roll, _ = _suffix_entropy_rate_fixed_window(
                roll_ent,
                response_mask,
                window_tokens=fixed_window_tokens,
                min_suffix_tokens=sentence_min_suffix_tokens,
                eligible_mask=eligible_mask,
            )
        else:
            f_rate_roll = _suffix_entropy_rate_exclusive(roll_ent, response_mask)

        spearman_vals: list[float] = []
        groups_for_diag = _prompt_group_indices(prompts)
        for idxs in groups_for_diag.values():
            xs: list[float] = []
            ys: list[float] = []
            for b in idxs:
                mask_bt = high_mask[b] & torch.isfinite(f_rate[b]) & torch.isfinite(f_rate_roll[b])
                ts = torch.where(mask_bt)[0].tolist()
                for t in ts:
                    xs.append(float(f_rate[b, t].item()))
                    ys.append(float(f_rate_roll[b, t].item()))
            if len(xs) >= 3:
                rho = _spearmanr_safe(np.asarray(xs, dtype=np.float64), np.asarray(ys, dtype=np.float64))
                if np.isfinite(rho):
                    spearman_vals.append(float(rho))
        if len(spearman_vals) > 0:
            sp = np.asarray(spearman_vals, dtype=np.float64)
            off_spearman_mean = float(np.mean(sp))
            off_spearman_p10 = float(np.quantile(sp, 0.10))
            off_spearman_p50 = float(np.quantile(sp, 0.50))
            off_spearman_p90 = float(np.quantile(sp, 0.90))
            off_spearman_n_prompts = float(sp.shape[0])

    metrics: dict[str, float] = {
        "fepo/n_selected": float(n_high),
        "fepo/lowtail_mode": 1.0,
        "fepo/h_threshold": float(h_threshold),
        "fepo/high_entropy_top_ratio_mode": 1.0 if high_entropy_select_mode == "top_ratio" else 0.0,
        "fepo/high_entropy_top_ratio": float(high_entropy_top_ratio)
        if high_entropy_select_mode == "top_ratio"
        else float("nan"),
        "fepo/alpha": float(alpha),
        "fepo/beta": float(beta),
        "fepo/low_tail_neg_adv_penalty": float(low_tail_neg_adv_penalty),
        "fepo/high_head_penalty": float(high_head_penalty),
        "fepo/high_head_neg_adv_boost": float(high_head_neg_adv_boost),
        "fepo/suffix_len_debias_enable": 1.0 if suffix_len_debias_enable else 0.0,
        "fepo/suffix_len_bin_width": float(max(suffix_len_bin_width, 1)),
        "fepo/suffix_len_min_count": float(max(suffix_len_min_count, 1)),
        "fepo/suffix_len_z_clip": float(max(suffix_len_z_clip, 0.0)),
        "fepo/suffix_len_debias_bin_fallback": float(n_len_bin_fallback),
        "fepo/suffix_len_debias_global_fallback": float(n_len_global_fallback),
        "fepo/low_tail_pos_adv_only": 1.0 if low_tail_pos_adv_only else 0.0,
        "fepo/high_head_neg_adv_only": 1.0 if high_head_neg_adv_only else 0.0,
        "fepo/high_head_pos_adv_only": 1.0 if high_head_pos_adv_only else 0.0,
        "fepo/n_boosted": float(n_boost),
        "fepo/n_head_penalized": float(n_head_penalized),
        "fepo/n_high_head_neg_boosted": float(n_high_head_neg_boosted),
        "fepo/n_low_tail_gate_blocked": float(n_low_tail_gate_blocked),
        "fepo/n_low_tail_neg_penalized": float(n_low_tail_neg_penalized),
        "fepo/n_high_head_gate_blocked": float(n_high_head_gate_blocked),
        "fepo/n_low_tail_effective": float(n_low_tail),
        "fepo/n_high_head_effective": float(n_high_head),
        "fepo/boost_hit_rate": (float(n_boost) / float(n_high)) if n_high > 0 else 0.0,
        "fepo/head_penalty_hit_rate": (float(n_head_penalized) / float(n_high)) if n_high > 0 else 0.0,
        "fepo/high_head_neg_boost_hit_rate": (float(n_high_head_neg_boosted) / float(n_high)) if n_high > 0 else 0.0,
        # Adv-sign diagnostics on effective low-tail / high-head points.
        "fepo/low_tail_adv_pos_ratio": (float(n_low_tail_adv_pos) / float(n_low_tail)) if n_low_tail > 0 else float("nan"),
        "fepo/low_tail_neg_penalized_ratio": (float(n_low_tail_neg_penalized) / float(n_low_tail_adv_neg_strict))
        if n_low_tail_adv_neg_strict > 0
        else float("nan"),
        "fepo/high_head_adv_pos_ratio": (float(n_high_head_adv_pos) / float(n_high_head))
        if n_high_head > 0
        else float("nan"),
        # Composition diagnostics: in positive/negative-adv high points, how many are low-tail/high-head.
        "fepo/adv_pos_low_tail_ratio_in_high": (float(n_low_tail_adv_pos) / float(n_adv_pos_high))
        if n_adv_pos_high > 0
        else float("nan"),
        "fepo/adv_pos_high_head_ratio_in_high": (float(n_high_head_adv_pos) / float(n_adv_pos_high))
        if n_adv_pos_high > 0
        else float("nan"),
        "fepo/adv_neg_low_tail_ratio_in_high": (float(int((low_tail_mask & adv_neg_mask).sum().item())) / float(n_adv_neg_high))
        if n_adv_neg_high > 0
        else float("nan"),
        "fepo/adv_neg_high_head_ratio_in_high": (float(int((high_head_mask & adv_neg_mask).sum().item())) / float(n_adv_neg_high))
        if n_adv_neg_high > 0
        else float("nan"),
        "fepo/q_mean": float(torch.mean(q_valid).item()) if q_valid.numel() > 0 else float("nan"),
        "fepo/m_mean_on_high": float(torch.mean(m_valid).item()) if m_valid.numel() > 0 else 1.0,
        "fepo/n_low_entropy": float(n_low),
        "fepo/low_entropy_ratio": (float(n_low) / float(n_resp)) if n_resp > 0 else 0.0,
        "fepo/low_entropy_mean": float(torch.mean(low_entropy_vals.float()).item())
        if low_entropy_vals.numel() > 0
        else float("nan"),
        "fepo/suffix_mode_sentence": 1.0 if suffix_mode == "sentence" else 0.0,
        "fepo/suffix_mode_fixed_window": 1.0 if suffix_mode == "fixed_window" else 0.0,
        "fepo/rank_scope_batch": 1.0 if rank_scope == "batch" else 0.0,
        "fepo/sentence_min_suffix_tokens": float(max(sentence_min_suffix_tokens, 1)),
        "fepo/sentence_num_threads": float(max(sentence_num_threads, 1)),
        "fepo/sentence_only_high_entropy": 1.0 if sentence_only_high_entropy else 0.0,
        "fepo/sentence_max_scan_tokens": float(max(sentence_max_scan_tokens, 0)),
        "fepo/sentence_high_entropy_ratio": float(max(sentence_high_entropy_ratio, 0.0)),
        "fepo/fixed_window_tokens": float(max(fixed_window_tokens, 1)),
        "fepo/sentence_high_entropy_cap_count": float(eligible_cap_mask.sum().item()),
        "fepo/sentence_positions_used": float(n_suffix_used),
        "fepo/sentence_positions_filtered_short": float(n_filtered_short_sentence),
        "fepo/f_suffix_rate_mean_all_valid": float(torch.mean(f_valid_all).item()) if f_valid_all.numel() > 0 else float("nan"),
        "fepo/f_suffix_rate_p10_all_valid": _pct(f_valid_all, 0.10),
        "fepo/f_suffix_rate_p50_all_valid": _pct(f_valid_all, 0.50),
        "fepo/f_suffix_rate_p90_all_valid": _pct(f_valid_all, 0.90),
        "fepo/f_suffix_rate_mean_high": float(torch.mean(f_valid_high).item()) if f_valid_high.numel() > 0 else float("nan"),
        "fepo/f_suffix_rate_p10_high": _pct(f_valid_high, 0.10),
        "fepo/f_suffix_rate_p50_high": _pct(f_valid_high, 0.50),
        "fepo/f_suffix_rate_p90_high": _pct(f_valid_high, 0.90),
        "fepo/sentence_suffix_len_mean": float(torch.mean(len_valid).item()) if len_valid.numel() > 0 else float("nan"),
        "fepo/sentence_suffix_len_p10": _pct(len_valid, 0.10),
        "fepo/sentence_suffix_len_p50": _pct(len_valid, 0.50),
        "fepo/sentence_suffix_len_p90": _pct(len_valid, 0.90),
        "fepo/offpolicy_diag_enable": 1.0 if offpolicy_diag_enable else 0.0,
        "fepo/offpolicy_ratio_mean": off_ratio_mean,
        "fepo/offpolicy_ratio_p10": off_ratio_p10,
        "fepo/offpolicy_ratio_p50": off_ratio_p50,
        "fepo/offpolicy_ratio_p90": off_ratio_p90,
        "fepo/offpolicy_abs_log_ratio_mean": off_log_ratio_abs_mean,
        "fepo/offpolicy_spearman_mean": off_spearman_mean,
        "fepo/offpolicy_spearman_p10": off_spearman_p10,
        "fepo/offpolicy_spearman_p50": off_spearman_p50,
        "fepo/offpolicy_spearman_p90": off_spearman_p90,
        "fepo/offpolicy_spearman_n_prompts": off_spearman_n_prompts,
    }

    point_records: list[dict[str, Any]] = []
    if collect_point_records and n_high > 0:
        for b in range(int(entropy.size(0))):
            ts = torch.where(high_mask[b])[0].tolist()
            for t in ts:
                q_bt = float(q[b, t].item())
                m_bt = float(m[b, t].item())
                adv_bt = float(advantages[b, t].item())
                point_records.append(
                    {
                        "batch_index": int(b),
                        "response_t": int(t),
                        "h_t": float(entropy[b, t].item()),
                        "f_suffix_rate": float(f_rate[b, t].item()),
                        "f_rank_score": float(rank_score[b, t].item()),
                        "suffix_tokens_used": int(keep_k[b, t].item())
                        if suffix_mode in {"sentence", "fixed_window"}
                        else None,
                        "q": q_bt,
                        "m": m_bt,
                        "adv_before": adv_bt,
                        "adv_after": float((advantages[b, t] * m[b, t]).item()),
                        "boosted": bool(m_bt > 1.0),
                        "low_tail_neg_penalized": bool(
                            np.isfinite(q_bt) and q_bt <= beta and adv_bt < 0.0 and m_bt < 1.0
                        ),
                    }
                )
    return batch, metrics, point_records


def select_high_entropy_positions(
    entropy: torch.Tensor,
    response_mask: torch.Tensor,
    *,
    h_threshold: float,
    max_points_per_seq: int,
    max_points_ratio: float,
) -> list[tuple[int, int]]:
    """Return list of (batch_idx, response_time_idx) with entropy >= threshold, top-k per sequence."""
    B, _T = entropy.shape
    out: list[tuple[int, int]] = []
    for b in range(B):
        mask = response_mask[b].bool()
        if not mask.any():
            continue
        row = entropy[b].float().clone()
        row[~mask] = float("-inf")
        valid_idx = torch.where(mask)[0]
        above = valid_idx[row[valid_idx] >= h_threshold]
        if above.numel() == 0:
            continue
        if int(max_points_per_seq) > 0:
            point_budget = int(max_points_per_seq)
        else:
            valid_len = int(mask.sum().item())
            point_budget = max(1, int(valid_len * float(max_points_ratio)))
        k = min(point_budget, int(above.numel()))
        sub = row[above]
        _, topk = torch.topk(sub, k=k, largest=True)
        chosen = above[topk]
        for t in chosen.tolist():
            out.append((b, int(t)))
    return out


def build_fepo_jobs(
    batch: DataProto,
    positions: list[tuple[int, int]],
    *,
    entropy: torch.Tensor,
    f_sentence_max_new_tokens: int,
    mc_max_new_tokens: int,
) -> tuple[list[dict[str, Any]], list[tuple[int, int]]]:
    """Branching-aligned jobs: ``prefix_before`` = prompt + response[:t] (excl. token t), ``chosen_token`` = response[t].

    Continuation MC cap matches ``compare_bias_sign_bucket_vs_mc`` style:
    ``min(f_sentence_max_new_tokens, mc_max_new_tokens, response_length - t - 1)``.
    """
    input_ids = batch.batch["input_ids"].detach().cpu()
    responses = batch.batch["responses"].detach().cpu()
    response_mask = batch.batch["response_mask"].detach().cpu().bool()
    entropy_cpu = entropy.detach().cpu()
    pl = int(batch.batch["prompts"].size(1))

    # Prompt side is LEFT-padded in verl (input_ids = [pad...pad, prompt, response]).
    # We must strip leading pad tokens using attention_mask, otherwise the prefix sent to vLLM
    # will include pad tokens and produce a next-token distribution very different from rollout.
    attn_mask = batch.batch.get("attention_mask", None)
    if attn_mask is not None:
        attn_mask = attn_mask.detach().cpu().bool()

    jobs: list[dict[str, Any]] = []
    coord: list[tuple[int, int]] = []

    # Group points by sample id so each sample row is materialized once.
    pos_by_b: dict[int, list[int]] = defaultdict(list)
    for b, t in positions:
        pos_by_b[int(b)].append(int(t))

    for b, ts in pos_by_b.items():
        row_mask = response_mask[b]
        valid_len = int(row_mask.sum().item())
        if valid_len <= 1:
            continue

        # Materialize sample rows once; reuse cheap Python slicing for each point.
        input_row = input_ids[b].tolist()
        resp_row = responses[b].tolist()

        # Locate where the real (non-pad) prompt starts within the left-padded prompt block.
        if attn_mask is not None:
            prompt_attn = attn_mask[b, :pl]
            nonzero = prompt_attn.nonzero(as_tuple=False)
            prompt_start = int(nonzero[0].item()) if nonzero.numel() > 0 else 0
        else:
            prompt_start = 0

        for t in ts:
            if t < 0 or t >= valid_len:
                continue

            rem = valid_len - (t + 1)
            if rem <= 0:
                continue

            cont_max = min(int(f_sentence_max_new_tokens), int(mc_max_new_tokens), rem)
            cont_max = max(1, cont_max)
            prefix_before = input_row[prompt_start : pl + t]
            chosen_token = int(resp_row[t])
            suffix_after = resp_row[t + 1 : t + 1 + cont_max]
            jobs.append(
                {
                    "prefix_before": prefix_before,
                    "chosen_token": chosen_token,
                    "cont_max_new_tokens": cont_max,
                    "h_t": float(entropy_cpu[b, t].item()),
                    "suffix_after": [int(x) for x in suffix_after],
                }
            )
            coord.append((b, t))
    return jobs, coord


def deltas_to_sparse_bonus(
    batch_size: int,
    response_length: int,
    device: torch.device,
    dtype: torch.dtype,
    coord: list[tuple[int, int]],
    deltas: list[float],
    ok: list[bool],
    *,
    delta_pos_threshold: float,
    delta_neg_threshold: float,
    bonus_pos: float,
    bonus_neg: float,
) -> torch.Tensor:
    """Map scalar deltas to per-token advantage bonus (sparse)."""
    bonus = torch.zeros(batch_size, response_length, device=device, dtype=dtype)
    for (b, t), d, ok_i in zip(coord, deltas, ok, strict=True):
        if not ok_i or not np.isfinite(d):
            continue
        df = float(d)
        if df >= delta_pos_threshold:
            bonus[b, t] = bonus[b, t] + float(bonus_pos)
        elif df <= -delta_neg_threshold:
            bonus[b, t] = bonus[b, t] - float(bonus_neg)
    return bonus


def run_fepo_advantage_phase(
    batch: DataProto,
    tokenizer: Any,
    fepo_cfg: dict[str, Any],
    async_rollout_manager: Optional[Any],
) -> tuple[DataProto, dict[str, float], list[dict[str, Any]]]:
    """After GRPO advantages are computed, add FEPO token bonus to ``advantages``."""
    if async_rollout_manager is None:
        raise NotImplementedError(
            "FEPO currently requires async rollout (vLLM agent loop). "
            "Set async_rollout_mode / use the standard vLLM agent-loop pipeline."
        )

    enable = bool(fepo_cfg.get("enable", False))
    if not enable:
        return batch, {}, []
    fepo_variant = str(fepo_cfg.get("variant", "legacy_mc_bonus"))
    if fepo_variant == "lowtail_adv":
        return _run_fepo_lowtail_adv_phase(batch, tokenizer, fepo_cfg)
    collect_point_records = bool(fepo_cfg.get("__collect_point_records", True))

    h_threshold = float(fepo_cfg.get("h_threshold", 2.0))
    max_points = int(fepo_cfg.get("max_points_per_seq", 4))
    max_points_ratio = float(fepo_cfg.get("max_points_ratio", 0.01))
    delta_pos_threshold = float(fepo_cfg.get("delta_pos_threshold", 0.05))
    delta_neg_threshold = float(fepo_cfg.get("delta_neg_threshold", 0.05))
    bonus_pos = float(fepo_cfg.get("bonus_pos", 0.02))
    bonus_neg = float(fepo_cfg.get("bonus_neg", 0.02))

    ent_key = "fepo_token_entropy" if "fepo_token_entropy" in batch.batch else "token_entropy"
    if ent_key not in batch.batch:
        raise ValueError(
            "FEPO requires per-token entropy in batch['fepo_token_entropy'] (preferred) or "
            "batch['token_entropy']. Enable recomputation of old_log_prob (non-bypass) or "
            "populate fepo_token_entropy in the trainer."
        )

    entropy = batch.batch[ent_key]
    response_mask = batch.batch["response_mask"]
    positions = select_high_entropy_positions(
        entropy,
        response_mask,
        h_threshold=h_threshold,
        max_points_per_seq=max_points,
        max_points_ratio=max_points_ratio,
    )
    if not positions:
        return batch, {"fepo/n_selected": 0.0}, []

    f_sent = int(fepo_cfg.get("f_sentence_max_new_tokens", 128))
    mc_cap = int(fepo_cfg.get("mc_max_new_tokens", 128))
    jobs, coord = build_fepo_jobs(
        batch,
        positions,
        entropy=entropy,
        f_sentence_max_new_tokens=f_sent,
        mc_max_new_tokens=mc_cap,
    )
    if not jobs:
        return batch, {"fepo/n_selected": float(len(positions)), "fepo/n_jobs_ok": 0.0}, []

    payload = {
        "jobs": jobs,
        "mc_m": int(fepo_cfg.get("mc_m", 1)),
        "mc_temperature": float(fepo_cfg.get("mc_temperature", 1.0)),
        "mc_top_p": float(fepo_cfg.get("mc_top_p", 0.95)),
        "logprobs_k": int(fepo_cfg.get("logprobs_k", 20)),
        "f_continuation_mode": str(fepo_cfg.get("f_continuation_mode", "first_sentence")),
        "f_sentence_max_new_tokens": f_sent,
        "normalize_by_continuation_length": bool(fepo_cfg.get("normalize_by_continuation_length", True)),
        "mc_max_new_tokens": mc_cap,
        "candidate_top_p": float(fepo_cfg.get("candidate_top_p", 0.95)),
        "candidate_max_k": int(fepo_cfg.get("candidate_max_k", 20)),
        "candidate_min_prob": float(fepo_cfg.get("candidate_min_prob", 0.0)),
        "min_candidates": int(fepo_cfg.get("min_candidates", 2)),
        "probe_batch_chunk": int(fepo_cfg.get("probe_batch_chunk", int(fepo_cfg.get("mc_batch_chunk", 32)))),
        "mc_batch_chunk": int(fepo_cfg.get("mc_batch_chunk", 32)),
        "fepo_job_concurrency": int(fepo_cfg.get("job_concurrency", 8)),
        "global_pooling": bool(fepo_cfg.get("global_pooling", False)),
        "detail_full": bool(collect_point_records),
        "f_bar_mode": str(fepo_cfg.get("f_bar_mode", "branching")),
        "f_real_mode": str(fepo_cfg.get("f_real_mode", "chosen_branch_mc")),
    }

    async_rollout_manager.wake_up()
    try:
        server_handles = list(getattr(async_rollout_manager, "server_handles", []) or [])
        if not server_handles:
            raise RuntimeError("FEPO requires async_rollout_manager.server_handles, but none were found.")

        n_jobs = len(jobs)
        n_srv = max(1, len(server_handles))
        # Round-robin sharding to spread FEPO compute across rollout replicas.
        shard_job_idx: list[list[int]] = [[] for _ in range(n_srv)]
        for i in range(n_jobs):
            shard_job_idx[i % n_srv].append(i)

        futures = []
        for sidx, indices in enumerate(shard_job_idx):
            if not indices:
                continue
            sub_payload = dict(payload)
            sub_payload["jobs"] = [jobs[i] for i in indices]
            futures.append((indices, server_handles[sidx].fepo_compute.remote(sub_payload)))

        deltas = [0.0] * n_jobs
        ok = [False] * n_jobs
        details: list[dict[str, Any]] = [{} for _ in range(n_jobs)]
        if futures:
            results = ray.get([f for _idx, f in futures])
            for (indices, _fut), out in zip(futures, results, strict=True):
                sd = out.get("deltas", [])
                so = out.get("ok", [])
                st = out.get("details", [])
                for local_i, global_i in enumerate(indices):
                    if local_i < len(sd):
                        deltas[global_i] = float(sd[local_i])
                    if local_i < len(so):
                        ok[global_i] = bool(so[local_i])
                    if local_i < len(st) and isinstance(st[local_i], dict):
                        details[global_i] = st[local_i]
    finally:
        async_rollout_manager.sleep()
    advantages = batch.batch["advantages"]
    bonus = deltas_to_sparse_bonus(
        advantages.size(0),
        advantages.size(1),
        advantages.device,
        advantages.dtype,
        coord,
        deltas,
        ok,
        delta_pos_threshold=delta_pos_threshold,
        delta_neg_threshold=delta_neg_threshold,
        bonus_pos=bonus_pos,
        bonus_neg=bonus_neg,
    )
    batch.batch["advantages"] = advantages + bonus * response_mask.float()

    n_ok = float(sum(1 for x in ok if x))
    n_jobs_total = float(len(jobs))
    fail_reasons: dict[str, int] = defaultdict(int)
    for k_ok, det in zip(ok, details, strict=False):
        if k_ok:
            continue
        if isinstance(det, dict):
            r = det.get("reason")
            if isinstance(r, str) and r:
                fail_reasons[r] += 1
            else:
                fail_reasons["unknown"] += 1
        else:
            fail_reasons["unknown"] += 1
    min_f_vals = [
        float(d.get("branch_min_f_mc"))
        for d, k in zip(details, ok, strict=False)
        if k and isinstance(d, dict) and d.get("branch_min_f_mc") is not None
    ]
    metrics = {
        "fepo/n_selected": float(len(positions)),
        "fepo/n_jobs": n_jobs_total,
        "fepo/n_jobs_ok": n_ok,
        "fepo/n_jobs_failed": float(len(ok) - int(n_ok)),
        "fepo/mean_delta": float(np.nanmean([float(d) for d, k in zip(deltas, ok, strict=True) if k]))
        if any(ok)
        else 0.0,
        "fepo/branch_steps_min_f_mc": float(np.nanmean(min_f_vals)) if min_f_vals else 0.0,
    }
    for rk, rv in fail_reasons.items():
        metrics[f"fepo/fail_{rk}"] = float(rv)
    point_records: list[dict[str, Any]] = []
    if collect_point_records:
        for i, ((b, t), job) in enumerate(zip(coord, jobs, strict=False)):
            d = details[i] if i < len(details) and isinstance(details[i], dict) else {}
            rec: dict[str, Any] = {
                "batch_index": int(b),
                "response_t": int(t),
                "ok": bool(ok[i]) if i < len(ok) else False,
                "delta": float(deltas[i]) if i < len(deltas) else None,
                "h_t": float(job.get("h_t", 0.0)),
                "prefix_before_token_ids": list(job.get("prefix_before", [])),
                "chosen_token_id": int(job.get("chosen_token", -1)),
                "suffix_after_token_ids": list(job.get("suffix_after", [])),
                "f_bar": d.get("f_bar"),
                "f_real": d.get("f_real"),
                "branch_min_f_mc": d.get("branch_min_f_mc"),
                "f_bar_mode": d.get("f_bar_mode"),
                "f_real_mode": d.get("f_real_mode"),
                "cands": d.get("cands"),
                "cand_probs": d.get("cand_probs"),
                "f_mc": d.get("f_mc"),
                "reason": d.get("reason"),
            }
            # Keep readable context for offline inspection.
            try:
                rec["prefix_before_text"] = tokenizer.decode(rec["prefix_before_token_ids"], skip_special_tokens=True)
                rec["chosen_token_text"] = tokenizer.decode([rec["chosen_token_id"]], skip_special_tokens=True)
                rec["suffix_after_text"] = tokenizer.decode(rec["suffix_after_token_ids"], skip_special_tokens=True)
                rec["context_text"] = rec["prefix_before_text"] + rec["chosen_token_text"]
            except Exception:
                pass
            point_records.append(rec)
    return batch, metrics, point_records
