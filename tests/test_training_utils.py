"""CPU-only unit tests for src/chat_setup.py and src/rewards.py."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import chat_setup  # noqa: E402
import rewards  # noqa: E402


def test_latest_checkpoint(tmp_path: Path):
    assert chat_setup.latest_checkpoint(tmp_path) is None
    (tmp_path / "checkpoint-100").mkdir()
    (tmp_path / "checkpoint-250").mkdir()
    (tmp_path / "other").mkdir()
    assert chat_setup.latest_checkpoint(tmp_path) == str(tmp_path / "checkpoint-250")


def test_mcq_reward_strict_and_overlong():
    good = "Let's calculate the energy.\n\nThe answer is (B)."
    r = rewards.score_completion(good, "B", "mcq", max_tokens=1000, n_tokens=400)
    assert r["correctness"] == 2.0
    assert r["format"] == 0.5
    assert r["overlong"] == 0.0
    assert r["total"] == 2.5

    wrong_overlong = "Wrong reasoning\n\nThe answer is (A)."
    r2 = rewards.score_completion(wrong_overlong, "B", "mcq", max_tokens=1000, n_tokens=1000)
    assert r2["correctness"] == 0.0
    assert r2["format"] == 0.5
    assert r2["overlong"] == -0.5
    assert r2["total"] == 0.0


def test_numeric_reward_tolerance():
    ans = "Step 1... The final answer is: $\\boxed{9.81}$"
    r = rewards.score_completion(ans, "9.80", "numeric", max_tokens=1000, n_tokens=200)
    assert r["correctness"] == 2.0
    assert r["format"] == 0.5


def test_grpo_reward_func_batch():
    comps = [[{"role": "assistant", "content": "Because F=ma.\n\nThe answer is (C)."}], [{"role": "assistant", "content": "No answer"}]]
    out = rewards.grpo_reward_func(comps, ["C", "A"], ["mcq", "mcq"])
    assert out[0] == 2.5
    assert out[1] == 0.0
