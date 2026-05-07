# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Adapted from https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/hendrycks_math/utils.py

import re
from typing import Optional


def last_boxed_only_string(string: str) -> Optional[str]:
    """Extract the last LaTeX boxed expression from a string.

    Args:
        string: Input string containing LaTeX code

    Returns:
        The last boxed expression or None if not found
    """
    idx = string.rfind("\\boxed{")
    if idx < 0:
        return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0

    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    return string[idx : right_brace_idx + 1] if right_brace_idx is not None else None


def remove_boxed(s: str) -> str:
    """Remove the LaTeX boxed command from a string.

    Args:
        s: String with format "\\boxed{content}"

    Returns:
        The content inside the boxed command
    """
    left = "\\boxed{"
    assert s[: len(left)] == left, f"box error: {s}"
    assert s[-1] == "}", f"box error: {s}"
    return s[len(left) : -1]


def _strip_latex_wrappers(s: str) -> str:
    """Strip common LaTeX wrappers that do not affect semantics."""
    s = s.replace("\\left", "").replace("\\right", "")
    s = s.replace("\\tfrac", "\\frac").replace("\\dfrac", "\\frac")
    return s


def _is_wrapped_by(s: str, left: str, right: str) -> bool:
    return len(s) >= 2 and s[0] == left and s[-1] == right


def _split_top_level_commas(s: str) -> list[str]:
    """Split by commas at top-level only, respecting bracket depth."""
    parts: list[str] = []
    depth = 0
    start = 0
    for i, ch in enumerate(s):
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth = max(0, depth - 1)
        elif ch == "," and depth == 0:
            parts.append(s[start:i].strip())
            start = i + 1
    parts.append(s[start:].strip())
    return parts


def _normalize_structured_answer(s: str) -> str:
    """Normalize structured answers like tuples/intervals for robust matching."""
    s = _strip_latex_wrappers(s).strip()
    if _is_wrapped_by(s, "(", ")") or _is_wrapped_by(s, "[", "]"):
        inner = s[1:-1].strip()
        items = _split_top_level_commas(inner)
        if len(items) > 1:
            normalized_items = [normalize_final_answer(item) for item in items]
            return f"{s[0]}{','.join(normalized_items)}{s[-1]}"
    return normalize_final_answer(s)


def extract_answer_candidate(
    solution_str: str, answer_pattern: str = r"(?i)Answer\s*:\s*([^\n]+)"
) -> tuple[str, bool]:
    """Extract final answer candidate and whether it came from \\boxed{...}."""
    matches = re.findall(answer_pattern, solution_str)
    answer_line = matches[-1] if matches else ""
    boxed = last_boxed_only_string(answer_line)
    if boxed is not None:
        return remove_boxed(boxed), True

    # Fallback: sometimes boxed answer is outside the strict Answer-line capture.
    boxed_full = last_boxed_only_string(solution_str)
    if boxed_full is not None:
        return remove_boxed(boxed_full), True

    if answer_line:
        return answer_line, False
    return "[INVALID]", False


# Constants for normalization
SUBSTITUTIONS = [
    ("an ", ""),
    ("a ", ""),
    (".$", "$"),
    ("\\$", ""),
    (r"\ ", ""),
    (" ", ""),
    ("mbox", "text"),
    (",\\text{and}", ","),
    ("\\text{and}", ","),
    ("\\text{m}", "\\text{}"),
]

REMOVED_EXPRESSIONS = [
    "square",
    "ways",
    "integers",
    "dollars",
    "mph",
    "inches",
    "hours",
    "km",
    "units",
    "\\ldots",
    "sue",
    "points",
    "feet",
    "minutes",
    "digits",
    "cents",
    "degrees",
    "cm",
    "gm",
    "pounds",
    "meters",
    "meals",
    "edges",
    "students",
    "childrentickets",
    "multiples",
    "\\text{s}",
    "\\text{.}",
    "\\text{\ns}",
    "\\text{}^2",
    "\\text{}^3",
    "\\text{\n}",
    "\\text{}",
    r"\mathrm{th}",
    r"^\circ",
    r"^{\circ}",
    r"\;",
    r",\!",
    "{,}",
    '"',
    "\\dots",
]


def normalize_final_answer(final_answer: str) -> str:
    """Normalize a final answer to a quantitative reasoning question.

    Args:
        final_answer: The answer string to normalize

    Returns:
        Normalized answer string
    """
    final_answer = final_answer.split("=")[-1]

    # Apply substitutions and removals
    for before, after in SUBSTITUTIONS:
        final_answer = final_answer.replace(before, after)
    for expr in REMOVED_EXPRESSIONS:
        final_answer = final_answer.replace(expr, "")

    # Extract and normalize LaTeX math
    final_answer = re.sub(r"(.*?)(\$)(.*?)(\$)(.*)", "$\\3$", final_answer)
    final_answer = re.sub(r"(\\text\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\textbf\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\overline\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\boxed\{)(.*)(\})", "\\2", final_answer)

    # Normalize shorthand TeX:
    #  \fracab -> \frac{a}{b}
    #  \frac{abc}{bef} -> \frac{abc}{bef}
    #  \fracabc -> \frac{a}{b}c
    #  \sqrta -> \sqrt{a}
    #  \sqrtab -> sqrt{a}b
    final_answer = re.sub(r"(frac)([^{])(.)", "frac{\\2}{\\3}", final_answer)
    final_answer = re.sub(r"(sqrt)([^{])", "sqrt{\\2}", final_answer)
    final_answer = final_answer.replace("$", "")

    # Normalize plain integers:
    # - remove thousands separators
    # - normalize leading zeros so "024" and "24" are treated as equal
    int_like = final_answer.replace(",", "")
    if re.fullmatch(r"[+-]?\d+", int_like):
        sign = ""
        if int_like.startswith(("+", "-")):
            sign, int_like = int_like[0], int_like[1:]
        int_like = int_like.lstrip("0") or "0"
        final_answer = f"{sign}{int_like}"

    return final_answer.strip()


def is_correct_minerva(
    solution_str: str, gt: str, gt_need_extract: bool = False, answer_pattern: str = r"(?i)Answer\s*:\s*([^\n]+)"
) -> tuple[bool, str, bool]:
    """Check if the solution is correct according to Minerva criteria.

    Args:
        solution_str: The solution string to check
        gt: The ground truth answer
        gt_need_extract: Whether the ground truth needs extraction
        answer_pattern: Regex pattern to extract the answer

    Returns:
        Tuple of (is_correct, normalized_prediction)
    """
    # Extract answer from solution (prefer boxed answer)
    extracted_answer, from_boxed = extract_answer_candidate(solution_str, answer_pattern)
    pred = _normalize_structured_answer(extracted_answer)

    # Process ground truth
    if gt_need_extract:
        gt_boxed = last_boxed_only_string(gt)
        gt = _normalize_structured_answer(remove_boxed(gt_boxed)) if gt_boxed is not None else _normalize_structured_answer(gt)
    else:
        gt = _normalize_structured_answer(gt)

    return (pred == gt), pred, from_boxed


def is_correct_strict_box(
    pred: str, gt: str, pause_tokens_index: Optional[list[int]] = None
) -> tuple[int, Optional[str]]:
    """Check if the prediction is correct using strict boxed answer criteria.

    Args:
        pred: The prediction string
        gt: The ground truth answer
        pause_tokens_index: Indices of pause tokens

    Returns:
        Tuple of (score, extracted_prediction)
    """
    # Extract the relevant part of the prediction
    if pause_tokens_index is not None:
        assert len(pause_tokens_index) == 4
        pred = pred[pause_tokens_index[-1] - 100 :]
    else:
        pred = pred[-100:]

    # Extract and check the boxed answer
    boxed_pred = last_boxed_only_string(pred)
    extracted_pred = remove_boxed(boxed_pred) if boxed_pred is not None else None

    return 1 if (extracted_pred == gt) else -1, extracted_pred


def verify(
    solution_str: str, answer: str, strict_box_verify: bool = False, pause_tokens_index: Optional[list[int]] = None
) -> tuple[bool, Optional[str], bool]:
    """Verify if the solution is correct.

    Args:
        solution_str: The solution string to verify
        answer: The ground truth answer
        strict_box_verify: Whether to use strict box verification
        pause_tokens_index: Indices of pause tokens

    Returns:
        True if the solution is correct, False otherwise
    """
    answer = str(answer) if answer is not None else ""

    if strict_box_verify:
        correct, pred = is_correct_strict_box(solution_str, answer, pause_tokens_index)
        return correct == 1, pred, pred is not None

    correct, pred, from_boxed = is_correct_minerva(solution_str, answer)
    return correct, pred, from_boxed


def compute_score(
    solution_str: str,
    ground_truth: str,
    strict_box_verify: bool = False,
    pause_tokens_index: Optional[list[int]] = None,
    format_penalty: float = -0.2,
) -> float:
    """Compute the reward score for a solution.

    Args:
        solution_str: The solution string
        ground_truth: The ground truth answer
        strict_box_verify: Whether to use strict box verification
        pause_tokens_index: Indices of pause tokens

    Returns:
        Reward score (correctness score + format penalty when format is wrong).
    """
    # Limit solution length for efficiency
    solution_str = solution_str[-300:]  # The longest answer in MATH-500 has 159 characters

    # Verify the solution
    correct, pred, from_boxed = verify(solution_str, ground_truth, strict_box_verify, pause_tokens_index)

    format_term = 0.0 if from_boxed else float(format_penalty)
    reward = (1.0 if correct else 0.0) + format_term
    acc = correct

    return {
        "score": reward,
        "acc": acc,
        "pred": pred,
        "format_score": format_term,
        "from_boxed": from_boxed,
    }
