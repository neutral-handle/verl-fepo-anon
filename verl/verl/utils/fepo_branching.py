# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""FEPO branching: top-p capped candidates from a step logprob dict (aligned with entropy_ce ``infer_topk_f_mc_compare``)."""

from __future__ import annotations

import numpy as np


def topp_capped_candidates_from_step_logprobs(
    step_logprobs: dict[int, float],
    top_p: float,
    max_k: int,
) -> tuple[list[int], list[float]]:
    """Return (token_ids, renorm_probs) over nucleus top-p truncated to at most ``max_k`` tokens."""
    if not step_logprobs:
        return [], []
    items = sorted(step_logprobs.items(), key=lambda x: x[1], reverse=True)
    tids_all = [int(tid) for tid, _ in items]
    lps = np.array([float(lp) for _, lp in items], dtype=np.float64)
    m = float(np.max(lps))
    p = np.exp(np.clip(lps - m, -80.0, 0.0))
    z = float(np.sum(p))
    if z <= 0.0 or not np.isfinite(z):
        p = np.ones(len(tids_all), dtype=np.float64) / float(len(tids_all))
    else:
        p = p / z
    tp = float(np.clip(top_p, 1e-12, 1.0))
    cum = np.cumsum(p)
    idx = int(np.searchsorted(cum, tp, side="left"))
    k_topp = max(1, idx + 1)
    k_used = min(max(1, int(max_k)), k_topp)
    sel_tids = tids_all[:k_used]
    sel_p = p[:k_used]
    z2 = float(np.sum(sel_p))
    if z2 <= 0.0 or not np.isfinite(z2):
        sel_p = np.ones(len(sel_tids), dtype=np.float64) / float(len(sel_tids))
    else:
        sel_p = sel_p / z2
    return sel_tids, [float(x) for x in sel_p.tolist()]
