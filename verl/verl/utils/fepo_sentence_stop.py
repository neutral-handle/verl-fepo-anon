# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""First-sentence truncation for FEPO continuation-F (aligned with examples/entropy_ce/sentence_stop_utils)."""

from __future__ import annotations

import re
from typing import Any, Callable


def _likely_decimal_or_incomplete_number_at_end(s: str) -> bool:
    t = s.rstrip()
    if not t or t[-1] != ".":
        return False
    if re.search(r"\d\.\s*$", t):
        return True
    if re.search(r"\d\.\d+\s*$", t):
        return True
    return False


def _first_paragraph_break_end(s: str) -> int | None:
    p = s.find("\n\n")
    if p < 0:
        return None
    return p + 2


def _first_chinese_sentence_punct_end(s: str) -> int | None:
    for i, ch in enumerate(s):
        if ch in "。！？":
            return i + 1
    return None


def _first_english_bang_question_end(s: str) -> int | None:
    for i, ch in enumerate(s):
        if ch not in "!?":
            continue
        if i == 0:
            continue
        return i + 1
    return None


def _first_valid_sentence_period_end(s: str) -> int | None:
    for i, ch in enumerate(s):
        if ch != ".":
            continue
        prefix = s[: i + 1]
        p2 = prefix.rstrip()
        if not p2.endswith("."):
            continue
        if _likely_decimal_or_incomplete_number_at_end(p2):
            continue
        return i + 1
    return None


def _first_sentence_boundary_end_exclusive(s: str) -> int | None:
    cands: list[int] = []
    e = _first_paragraph_break_end(s)
    if e is not None:
        cands.append(e)
    e = _first_chinese_sentence_punct_end(s)
    if e is not None:
        cands.append(e)
    e = _first_english_bang_question_end(s)
    if e is not None:
        cands.append(e)
    e = _first_valid_sentence_period_end(s)
    if e is not None:
        cands.append(e)
    if not cands:
        return None
    return min(cands)


def truncate_gen_ids_to_first_sentence(
    gen_ids: list[int],
    tokenizer: Any,
    stop_check: Callable[[str], bool],
) -> int:
    for k in range(1, len(gen_ids) + 1):
        frag = tokenizer.decode(gen_ids[:k], skip_special_tokens=True)
        if stop_check(frag):
            return k
    return len(gen_ids)


def completion_should_stop_after_first_sentence_simple(
    completion_text: str,
    *,
    min_chars: int = 2,
) -> bool:
    t = completion_text
    if len(t.strip()) < min_chars:
        return False
    end = _first_sentence_boundary_end_exclusive(t)
    if end is None:
        return False
    if len(t) < end:
        return False
    return True
