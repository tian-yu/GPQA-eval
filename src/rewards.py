"""Verifiable reward functions and answer extractors for GRPO on GPQA-style MCQs & Dr.SCI numeric tasks."""

from __future__ import annotations

import math
import re

STRICT_MCQ_RE = re.compile(r"The answer is \(([A-D])\)\.")
FLEX_MCQ_RE = re.compile(r"(?:The (?:correct |final )?answer is:?\s*)?\(([A-D])\)")
BOXED_RE = re.compile(r"\\boxed\{([^}]+)\}")
FINAL_NUM_RE = re.compile(r"The final answer is:?\s*(?:\$\\boxed\{)?([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\}?")


def extract_mcq_letter(text: str) -> tuple[str | None, bool]:
    """Return (extracted_letter, is_strict_format)."""
    strict = STRICT_MCQ_RE.findall(text)
    if strict:
        return strict[-1], len(strict) == 1
    flex = FLEX_MCQ_RE.findall(text)
    if flex:
        return flex[-1], False
    return None, False


def extract_numeric(text: str) -> tuple[float | None, bool]:
    """Return (parsed_float, is_strict_boxed_format)."""
    boxed = BOXED_RE.findall(text)
    raw = boxed[-1] if boxed else None
    is_strict = len(boxed) == 1
    if raw is None:
        m = FINAL_NUM_RE.findall(text)
        raw = m[-1] if m else None
    if raw is None:
        return None, False
    cleaned = raw.strip().replace(",", "").replace("$", "")
    try:
        val = float(cleaned)
        if math.isfinite(val):
            return val, is_strict
    except ValueError:
        pass
    return None, False


def score_completion(text: str, gold_answer: str, answer_type: str, max_tokens: int = 1536, n_tokens: int | None = None) -> dict[str, float]:
    """Compute correctness, format, and DAPO-style soft overlong penalty."""
    correctness = 0.0
    fmt = 0.0

    if answer_type == "mcq":
        pred, is_strict = extract_mcq_letter(text)
        if pred == gold_answer:
            correctness = 2.0
        if is_strict:
            fmt = 0.5
        elif pred is not None:
            fmt = 0.15
    elif answer_type == "numeric":
        pred_val, is_strict = extract_numeric(text)
        try:
            gold_val = float(gold_answer)
        except ValueError:
            gold_val = float("nan")
        if pred_val is not None and math.isfinite(gold_val):
            denom = max(abs(gold_val), 1e-9)
            if abs(pred_val - gold_val) / denom <= 0.01:
                correctness = 2.0
        if is_strict:
            fmt = 0.5
        elif pred_val is not None:
            fmt = 0.15

    # DAPO soft overlong penalty: ramp linearly from 0 to -0.5 over last 20% of budget
    overlong = 0.0
    if n_tokens is not None and max_tokens > 0:
        soft_start = int(0.8 * max_tokens)
        if n_tokens > soft_start:
            frac = min(1.0, (n_tokens - soft_start) / max(1, max_tokens - soft_start))
            overlong = -0.5 * frac

    return {
        "correctness": correctness,
        "format": fmt,
        "overlong": overlong,
        "total": correctness + fmt + overlong,
    }


def grpo_reward_func(completions: list[list[dict]], answer: list[str], answer_type: list[str], **kwargs) -> list[float]:
    """TRL GRPOTrainer-compatible reward function."""
    max_tokens = int(kwargs.get("max_completion_length", 1536))
    rewards: list[float] = []
    for comp, gold, atype in zip(completions, answer, answer_type):
        text = comp[0]["content"] if isinstance(comp, list) and comp else str(comp)
        approx_toks = max(1, len(text) // 4)
        rewards.append(score_completion(text, str(gold), str(atype), max_tokens=max_tokens, n_tokens=approx_toks)["total"])
    return rewards
