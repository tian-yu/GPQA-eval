"""Stage 1: Coverage-Selected LoRA SFT for Qwen3-Base on Colab T4 (<=1.75h)."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from chat_setup import (  # noqa: E402
    INSTRUCTION_PART,
    RESPONSE_PART,
    configure_tokenizer_and_eos,
    latest_checkpoint,
    load_colab_secrets,
    setup_wandb,
)
from fast_diag import run_heldout_eval  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-model", default="Qwen/Qwen3-4B-Base")
    p.add_argument("--dataset", default="BabyWombat/gpqa-ninja-mix")
    p.add_argument("--push-model", required=True, help="HF repo to push merged fp16 model")
    p.add_argument("--drive-dir", type=Path, default=Path("/content/ckpt/sft"))
    p.add_argument("--ckpt-repo", default=None, help="Optional private HF repo for intermediate checkpoints")
    p.add_argument("--lora-r", type=int, default=64)
    p.add_argument("--max-seq-len", type=int, default=2560)
    p.add_argument("--max-steps", type=int, default=-1)
    p.add_argument("--max-samples", type=int, default=0, help="0 = full sft_train split")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--eval-samples", type=int, default=120)
    args = p.parse_args(argv)

    load_colab_secrets()
    report_to = setup_wandb(project="gpqa-ninja", run_name=f"sft-{args.base_model.split('/')[-1]}", config=vars(args))

    from datasets import load_dataset
    from trl import SFTConfig, SFTTrainer
    from unsloth import FastLanguageModel
    from unsloth.chat_templates import train_on_responses_only

    t0 = time.time()
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.base_model,
        max_seq_length=args.max_seq_len,
        dtype=None,  # auto float16 on T4
        load_in_4bit=True,
    )
    configure_tokenizer_and_eos(tokenizer, model)

    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        lora_alpha=args.lora_r * 2,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )

    ds = load_dataset(args.dataset, "sft_train", split="train")
    if args.max_samples > 0:
        ds = ds.select(range(min(args.max_samples, len(ds))))

    def format_row(ex):
        return {"text": tokenizer.apply_chat_template(ex["messages"], tokenize=False, add_generation_prompt=False)}

    ds = ds.map(format_row, remove_columns=ds.column_names)

    args.drive_dir.mkdir(parents=True, exist_ok=True)
    resume = latest_checkpoint(args.drive_dir)
    if resume:
        print(f"Resuming SFT from checkpoint: {resume}")

    sft_args = SFTConfig(
        output_dir=str(args.drive_dir),
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        num_train_epochs=1,
        max_steps=args.max_steps,
        logging_steps=10,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=2,
        fp16=True,
        bf16=False,
        optim="adamw_8bit",
        max_length=args.max_seq_len,
        packing=True,
        dataset_text_field="text",
        report_to=report_to,
        seed=42,
        push_to_hub=bool(args.ckpt_repo),
        hub_model_id=args.ckpt_repo,
        hub_strategy="checkpoint" if args.ckpt_repo else "end",
        hub_private_repo=True,
    )

    trainer = SFTTrainer(model=model, processing_class=tokenizer, train_dataset=ds, args=sft_args)
    trainer = train_on_responses_only(trainer, instruction_part=INSTRUCTION_PART, response_part=RESPONSE_PART)
    trainer.train(resume_from_checkpoint=resume)

    hours = round((time.time() - t0) / 3600.0, 3)
    print(f"SFT training finished in {hours} T4 hours. Merging & pushing to {args.push_model}...")

    metrics = run_heldout_eval(
        model,
        tokenizer,
        dataset_repo=args.dataset,
        max_samples=args.eval_samples,
        out_json=args.drive_dir / "sft_diag.json",
    )
    metrics["train_hours"] = hours
    (args.drive_dir / "sft_diag.json").write_text(json.dumps(metrics, indent=2) + "\n")

    if report_to == "wandb":
        import wandb  # type: ignore

        wandb.log({f"val/{k}": v for k, v in metrics.items()})

    model.push_to_hub_merged(args.push_model, tokenizer, save_method="merged_16bit", token=os.environ.get("HF_TOKEN"), private=False)
    print(f"SFT model pushed to https://huggingface.co/{args.push_model}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
