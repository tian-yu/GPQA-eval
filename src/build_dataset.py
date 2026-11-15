"""Local M3 dataset curation, 4-gram coverage selection, and 13-gram GPQA decontamination.

Builds all four splits for our 4-hour Colab T4 post-training pipeline with 0.0 GPU hours:
  - sft_train   (~15,000 science examples, <=2560 tokens/row)
  - dpo_train   (~3,500 science/math preference pairs, chosen <=1536 tokens)
  - grpo_pool   (~2,500 verifiable STEM MCQs + Dr.SCI numeric problems)
  - val_heldout (~350 held-out verifiable science problems)

Usage:
    python src/build_dataset.py --push-repo BabyWombat/gpqa-ninja-mix
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import decontam  # noqa: E402
from coverage import select_by_part  # noqa: E402

GPQA_PROMPT = (
    "What is the correct answer to this question:{q}\nChoices:\n"
    "(A) {a}\n(B) {b}\n(C) {c}\n(D) {d}\nLet's think step by step: "
)
LETTERS = "ABCD"
_OPTION_RE = re.compile(r"^[ \t]*\(?([A-J])[\):\.][ \t]+(\S.*?)[ \t]*$", re.M)
_BOXED_LETTER_RE = re.compile(r"\\boxed\{\s*(?:\\text(?:bf)?\{)?\s*\(?([A-J])\)?\s*\}?\s*\}")
_ANSWER_IS_RE = re.compile(r"[Tt]he (?:correct |final )?answer is:?\s*\**\s*\(?([A-J])\)?(?![A-Za-z])")
_FINAL_BOX_RE = re.compile(
    r"(?:\n[ \t]*(?:#{1,6}[ \t]*)?\**(?:Final Answer|Answer|Conclusion)\**:?\**[ \t]*)?"
    r"\s*(?:\\\[|\\\(|\$\$|\$)?\s*\\boxed\{\s*(?:\\text(?:bf)?\{)?\s*\(?[A-J]\)?\s*\}?\s*\}\s*(?:\\\]|\\\)|\$\$|\$)?[\s\.\*]*$"
)


def clean_text(text: str) -> str:
    return " ".join(str(text).replace(" [title]", ". ").split())


def question_bucket(question: str) -> float:
    """Deterministic [0, 1) hash so SFT, GRPO, and Val never share questions."""
    h = hashlib.sha1(" ".join(question.lower().split()).encode()).hexdigest()
    return int(h[:8], 16) / 0x100000000


def parse_mcq(question: str) -> tuple[str, list[str]] | None:
    matches = list(_OPTION_RE.finditer(question))
    if len(matches) != 4:
        return None
    if [m.group(1) for m in matches] != list(LETTERS):
        return None
    stem = question[: matches[0].start()].strip()
    between = [question[a.end() : b.start()] for a, b in zip(matches, matches[1:])] + [question[matches[-1].end() :]]
    if not stem or any(s.strip() for s in between):
        return None
    return stem, [m.group(2).strip() for m in matches]


def final_letter(text: str | None) -> str | None:
    if not text:
        return None
    found = _BOXED_LETTER_RE.findall(text) or _ANSWER_IS_RE.findall(text)
    return found[-1] if found else None


def format_gpqa_solution(answer: str, letter: str) -> str:
    body = _FINAL_BOX_RE.sub("", answer.rstrip()).rstrip()
    return f"{body}\n\nThe answer is ({letter})."


def parse_float(val: str | None) -> float | None:
    if val is None:
        return None
    s = str(val).strip().replace(",", "")
    try:
        x = float(s)
        return x if (x == x and abs(x) != float("inf")) else None
    except ValueError:
        return None


def count_tokens(tokenizer, messages: list[dict], max_tokens: int | None = None) -> int:
    text = "\n".join(str(m.get("content", "")) for m in messages)
    if max_tokens is not None and len(text) > max_tokens * 6:
        return max_tokens + 1
    return len(tokenizer.encode(text, add_special_tokens=False))


def collect_megascience(tokenizer, max_sft_tokens: int, pool_factor: int = 2) -> tuple[list[dict], list[dict], list[dict]]:
    """Stream MegaScience/MegaScience into SFT candidates, GRPO MCQ pool, and Val MCQs."""
    from datasets import load_dataset

    sft_targets = {
        "mcq": 3000,
        "textbook_physics": 3500,
        "textbook_chemistry": 3500,
        "textbook_biology": 2500,
    }
    caps = {k: v * pool_factor for k, v in sft_targets.items()}
    counts: Counter[str] = Counter()
    sft_cands: list[dict] = []
    grpo_mcqs: list[dict] = []
    val_mcqs: list[dict] = []
    grpo_cap, val_cap = 1500, 220

    ds = load_dataset("MegaScience/MegaScience", split="train", streaming=True)
    for idx_row, row in enumerate(ds, 1):
        if idx_row % 50000 == 0:
            print(f"    MegaScience scanned {idx_row:,} rows -> counts={dict(counts)}, grpo={len(grpo_mcqs)}, val={len(val_mcqs)}")
        src = str(row.get("source") or "").lower()
        q = str(row.get("question") or "").strip()
        a = str(row.get("answer") or "").strip()
        if not q or not a:
            continue

        if "nemotro" in src:
            parsed = parse_mcq(q)
            if parsed is None:
                continue
            stem, choices = parsed
            gold = final_letter(row.get("reference_answer"))
            sol = final_letter(a)
            if gold is None or gold not in LETTERS or sol != gold:
                continue
            prompt_str = GPQA_PROMPT.format(
                q=clean_text(stem),
                a=clean_text(choices[0]),
                b=clean_text(choices[1]),
                c=clean_text(choices[2]),
                d=clean_text(choices[3]),
            )
            bucket = question_bucket(stem)
            if bucket < 0.60 and counts["mcq"] < caps["mcq"]:
                msgs = [
                    {"role": "user", "content": prompt_str},
                    {"role": "assistant", "content": format_gpqa_solution(a, gold)},
                ]
                ntok = count_tokens(tokenizer, msgs, max_sft_tokens)
                if ntok <= max_sft_tokens:
                    sft_cands.append({"part": "mcq", "messages": msgs, "n_tokens": ntok, "source": "MegaScience/mcq"})
                    counts["mcq"] += 1
            elif 0.60 <= bucket < 0.90 and len(grpo_mcqs) < grpo_cap:
                grpo_mcqs.append({
                    "prompt": [{"role": "user", "content": prompt_str}],
                    "answer": gold,
                    "answer_type": "mcq",
                    "source": "MegaScience/mcq",
                })
            elif bucket >= 0.90 and len(val_mcqs) < val_cap:
                val_mcqs.append({
                    "prompt": [{"role": "user", "content": prompt_str}],
                    "answer": gold,
                    "answer_type": "mcq",
                    "source": "MegaScience/mcq",
                })

        elif "textbook" in src:
            subj = str(row.get("subject") or "").lower()
            part = None
            for s in ("physics", "chemistry", "biology"):
                if s in subj:
                    part = f"textbook_{s}"
                    break
            if part is None or counts[part] >= caps[part]:
                continue
            msgs = [{"role": "user", "content": q}, {"role": "assistant", "content": a}]
            ntok = count_tokens(tokenizer, msgs)
            if ntok <= max_sft_tokens:
                sft_cands.append({"part": part, "messages": msgs, "n_tokens": ntok, "source": f"MegaScience/{part}"})
                counts[part] += 1

        if all(counts[k] >= caps[k] for k in caps) and len(grpo_mcqs) >= grpo_cap and len(val_mcqs) >= val_cap:
            break

    return sft_cands, grpo_mcqs, val_mcqs


def collect_dolci_sft(tokenizer, max_sft_tokens: int, target: int = 2500, pool_factor: int = 2) -> list[dict]:
    """Stream science examples from allenai/Dolci-Instruct-SFT-7B."""
    from datasets import load_dataset

    cap = target * pool_factor
    cands: list[dict] = []
    ds = load_dataset("allenai/Dolci-Instruct-SFT-7B", split="train", streaming=True)
    for row in ds:
        domain = str(row.get("domain") or "").lower()
        src = str(row.get("source_dataset") or "").lower()
        if "science" not in domain and "science" not in src and "sciriff" not in src:
            continue
        msgs = row.get("messages") or []
        if not isinstance(msgs, list) or len(msgs) < 2:
            continue
        clean_msgs = [{"role": str(m["role"]), "content": str(m["content"])} for m in msgs if m.get("role") in ("user", "assistant", "system")]
        if len(clean_msgs) < 2 or clean_msgs[-1]["role"] != "assistant":
            continue
        ntok = count_tokens(tokenizer, clean_msgs, max_sft_tokens)
        if 32 <= ntok <= max_sft_tokens:
            cands.append({"part": "dolci_science", "messages": clean_msgs, "n_tokens": ntok, "source": f"Dolci-SFT/{src}"})
            if len(cands) >= cap:
                break
    return cands


def collect_dolci_dpo(tokenizer, max_chosen_tokens: int = 1536, target: int = 3500) -> list[dict]:
    """Stream concise science/math preference pairs from allenai/Dolci-Instruct-DPO."""
    from datasets import load_dataset

    pairs: list[dict] = []
    ds = load_dataset("allenai/Dolci-Instruct-DPO", split="train", streaming=True)
    for row in ds:
        pid = str(row.get("prompt_id") or "").lower()
        if not any(k in pid for k in ("science", "sciriff", "math")):
            continue
        chosen = row.get("chosen") or []
        rejected = row.get("rejected") or []
        if len(chosen) < 2 or len(rejected) < 2:
            continue
        prompt_msgs = chosen[:-1]
        chosen_msg = chosen[-1]
        rejected_msg = rejected[-1]
        if chosen_msg.get("role") != "assistant" or rejected_msg.get("role") != "assistant":
            continue
        ntok = count_tokens(tokenizer, chosen, max_chosen_tokens)
        if ntok <= max_chosen_tokens:
            pairs.append({
                "prompt": [{"role": str(m["role"]), "content": str(m["content"])} for m in prompt_msgs],
                "chosen": [{"role": "assistant", "content": str(chosen_msg["content"])}],
                "rejected": [{"role": "assistant", "content": str(rejected_msg["content"])}],
                "preference_type": str(row.get("preference_type") or ""),
            })
            if len(pairs) >= target:
                break
    return pairs


def collect_drsci_numeric(grpo_target: int = 1000, val_target: int = 130) -> tuple[list[dict], list[dict]]:
    """Stream verifiable numeric STEM problems from MiniByte-666/Dr.SCI."""
    from datasets import load_dataset

    grpo_rows: list[dict] = []
    val_rows: list[dict] = []
    ds = load_dataset("MiniByte-666/Dr.SCI", data_files="Dr_SCI_verifiable.parquet", split="train", streaming=True)
    for row in ds:
        info = row.get("extra_info") or {}
        subj = str(info.get("subject") or "").lower()
        if subj not in ("physics", "chemistry", "biology"):
            continue
        if not info.get("match_rule"):
            continue
        diff = info.get("difficulty")
        if diff is None or not (0.25 <= float(diff) <= 0.875):
            continue
        gt = (row.get("reward_model") or {}).get("ground_truth")
        num = parse_float(gt)
        if num is None:
            continue
        prompt = row.get("prompt") or []
        if not prompt:
            continue
        q_text = str(info.get("question") or prompt[0].get("content", ""))
        item = {
            "prompt": [{"role": str(m["role"]), "content": str(m["content"])} for m in prompt],
            "answer": str(num),
            "answer_type": "numeric",
            "source": f"Dr.SCI/{subj}",
        }
        if question_bucket(q_text) < 0.88:
            if len(grpo_rows) < grpo_target:
                grpo_rows.append(item)
        else:
            if len(val_rows) < val_target:
                val_rows.append(item)
        if len(grpo_rows) >= grpo_target and len(val_rows) >= val_target:
            break
    return grpo_rows, val_rows


def decontaminate_rows(rows: list[dict], gpqa_grams: set[tuple[str, ...]]) -> tuple[list[dict], dict]:
    """Run 13-gram GPQA decontamination and drop any flagged rows."""
    report = decontam.check(rows, gpqa_grams)
    flagged = set(report["flagged_row_indices"])
    clean = [r for i, r in enumerate(rows) if i not in flagged]
    return clean, report


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", type=Path, default=ROOT / "data" / "ninja_mix")
    p.add_argument("--tokenizer", default="Qwen/Qwen3-4B-Base")
    p.add_argument("--max-sft-tokens", type=int, default=2560)
    p.add_argument("--push-repo", default=None, help="Private HF dataset repo (e.g. BabyWombat/gpqa-ninja-mix)")
    args = p.parse_args(argv)

    from dotenv import load_dotenv
    load_dotenv(ROOT.parent / ".env")
    from transformers import AutoTokenizer

    print(f"Loading tokenizer {args.tokenizer}...")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    print("Indexing GPQA 13-grams for decontamination...")
    gpqa_grams, n_gpqa = decontam.load_gpqa_ngrams(None)
    print(f"Indexed {n_gpqa} GPQA questions.")

    print("Streaming MegaScience/MegaScience...")
    ms_sft, grpo_mcqs, val_mcqs = collect_megascience(tokenizer, args.max_sft_tokens)
    print(f"  MegaScience -> sft_cands={len(ms_sft)}, grpo_mcqs={len(grpo_mcqs)}, val_mcqs={len(val_mcqs)}")

    print("Streaming allenai/Dolci-Instruct-SFT-7B...")
    dolci_sft = collect_dolci_sft(tokenizer, args.max_sft_tokens)
    print(f"  Dolci SFT -> sft_cands={len(dolci_sft)}")

    print("Running 4-gram lazy greedy coverage selection (EESFT)...")
    sft_targets = {
        "mcq": 3000,
        "textbook_physics": 3500,
        "textbook_chemistry": 3500,
        "textbook_biology": 2500,
        "dolci_science": 2500,
    }
    sft_selected, cov_stats = select_by_part(ms_sft + dolci_sft, sft_targets, norm="sqrt")
    print("  Coverage stats:", json.dumps(cov_stats, indent=2))

    print("Streaming allenai/Dolci-Instruct-DPO...")
    dpo_rows = collect_dolci_dpo(tokenizer)
    print(f"  Dolci DPO -> pairs={len(dpo_rows)}")

    print("Streaming MiniByte-666/Dr.SCI (verifiable numeric)...")
    grpo_num, val_num = collect_drsci_numeric()
    print(f"  Dr.SCI -> grpo_num={len(grpo_num)}, val_num={len(val_num)}")

    splits = {
        "sft_train": [{"messages": r["messages"], "part": r["part"], "source": r["source"]} for r in sft_selected],
        "dpo_train": dpo_rows,
        "grpo_pool": grpo_mcqs + grpo_num,
        "val_heldout": val_mcqs + val_num,
    }

    decontam_summary = {"gpqa_questions_indexed": n_gpqa, "splits": {}}
    clean_splits: dict[str, list[dict]] = {}
    for name, rows in splits.items():
        clean, rep = decontaminate_rows(rows, gpqa_grams)
        clean_splits[name] = clean
        decontam_summary["splits"][name] = {
            "checked": rep["train_rows_checked"],
            "flagged_dropped": rep["train_rows_flagged"],
            "overlap_rate": rep["overlap_rate"],
            "final_count": len(clean),
        }
        write_jsonl(args.out_dir / f"{name}.jsonl", clean)

    decontam_summary["coverage_stats"] = cov_stats
    report_path = args.out_dir / "decontam_report.json"
    report_path.write_text(json.dumps(decontam_summary, indent=2) + "\n")
    print("Decontamination summary:", json.dumps(decontam_summary, indent=2))

    if args.push_repo:
        from datasets import Dataset
        token = os.environ.get("HF_TOKEN")
        print(f"Pushing private dataset configs to {args.push_repo}...")
        for k, v in clean_splits.items():
            Dataset.from_list(v).push_to_hub(args.push_repo, config_name=k, split="train", private=True, token=token)
        print("Push complete!")

    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    raise SystemExit(main())
