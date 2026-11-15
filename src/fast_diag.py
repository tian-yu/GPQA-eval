"""Fast held-out validation & EOS/truncation diagnostics (zero GPQA exposure)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from chat_setup import GLITCH_TOKEN, STOP_STRINGS  # noqa: E402
from rewards import score_completion  # noqa: E402


def run_heldout_eval(
    model,
    tokenizer,
    dataset_repo: str = "BabyWombat/gpqa-ninja-mix",
    split_config: str = "val_heldout",
    max_samples: int = 120,
    max_new_tokens: int = 1536,
    out_json: Path | None = None,
) -> dict:
    """Evaluate model on non-GPQA held-out science problems and return aggregate metrics."""
    import torch
    from datasets import load_dataset
    from unsloth import FastLanguageModel

    FastLanguageModel.for_inference(model)
    ds = load_dataset(dataset_repo, split_config, split="train")
    if max_samples and len(ds) > max_samples:
        ds = ds.select(range(max_samples))

    correct, extracted, truncated, glitched = 0, 0, 0, 0
    lengths: list[int] = []

    for row in ds:
        prompt_text = tokenizer.apply_chat_template(row["prompt"], tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
        in_len = inputs["input_ids"].shape[1]
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                eos_token_id=model.generation_config.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )
        gen_ids = out[0][in_len:]
        n_tok = int(gen_ids.shape[0])
        lengths.append(n_tok)
        text = tokenizer.decode(gen_ids, skip_special_tokens=False)
        for s in STOP_STRINGS:
            if s in text:
                text = text.split(s, 1)[0]

        if n_tok >= max_new_tokens:
            truncated += 1
        if GLITCH_TOKEN in text:
            glitched += 1

        res = score_completion(text, str(row["answer"]), str(row["answer_type"]), max_tokens=max_new_tokens, n_tokens=n_tok)
        if res["correctness"] > 0:
            correct += 1
        if res["format"] > 0:
            extracted += 1

    n = max(1, len(ds))
    metrics = {
        "n_eval": len(ds),
        "val_acc": round(100.0 * correct / n, 2),
        "extract_rate": round(100.0 * extracted / n, 2),
        "trunc_rate": round(100.0 * truncated / n, 2),
        "glitch_rate": round(100.0 * glitched / n, 2),
        "mean_tokens": round(sum(lengths) / n, 1),
    }
    if out_json:
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(metrics, indent=2) + "\n")
    print("Held-out diagnostics:", json.dumps(metrics))
    return metrics
