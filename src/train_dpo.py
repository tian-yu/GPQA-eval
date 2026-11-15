"""Stage 2: Length-Normalized / IPO Preference Tuning (DPO) on Colab T4 (<=0.80h)."""

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
    configure_tokenizer_and_eos,
    latest_checkpoint,
    load_colab_secrets,
    setup_wandb,
)
from fast_diag import run_heldout_eval  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--init-model", required=True, help="HF repo of S1 checkpoint")
    p.add_argument("--dataset", default="BabyWombat/gpqa-ninja-mix")
    p.add_argument("--push-model", required=True, help="HF repo to push merged fp16 DPO model")
    p.add_argument("--drive-dir", type=Path, default=Path("/content/drive/MyDrive/gpqa_ninja/dpo"))
    p.add_argument("--lora-r", type=int, default=32)
    p.add_argument("--max-seq-len", type=int, default=2048)
    p.add_argument("--max-steps", type=int, default=-1)
    p.add_argument("--max-samples", type=int, default=0)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--beta", type=float, default=0.1)
    p.add_argument("--eval-samples", type=int, default=120)
    args = p.parse_args(argv)

    load_colab_secrets()
    report_to = setup_wandb(project="gpqa-ninja", run_name=f"dpo-{args.init_model.split('/')[-1]}", config=vars(args))

    from datasets import load_dataset
    from trl import DPOConfig, DPOTrainer
    from unsloth import FastLanguageModel, PatchDPOTrainer

    PatchDPOTrainer()
    t0 = time.time()

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.init_model,
        max_seq_length=args.max_seq_len,
        dtype=None,
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

    ds = load_dataset(args.dataset, "dpo_train", split="train")
    if args.max_samples > 0:
        ds = ds.select(range(min(args.max_samples, len(ds))))

    def format_dpo(ex):
        prompt = tokenizer.apply_chat_template(ex["prompt"], tokenize=False, add_generation_prompt=True)
        chosen = ex["chosen"][0]["content"] + tokenizer.eos_token
        rejected = ex["rejected"][0]["content"] + tokenizer.eos_token
        return {"prompt": prompt, "chosen": chosen, "rejected": rejected}

    ds = ds.map(format_dpo, remove_columns=ds.column_names)

    args.drive_dir.mkdir(parents=True, exist_ok=True)
    resume = latest_checkpoint(args.drive_dir)
    if resume:
        print(f"Resuming DPO from checkpoint: {resume}")

    dpo_args = DPOConfig(
        output_dir=str(args.drive_dir),
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        num_train_epochs=1,
        max_steps=args.max_steps,
        beta=args.beta,
        loss_type="ipo",  # length-robust preference optimization (mitigates 74% longer-chosen bias)
        max_length=args.max_seq_len,
        max_prompt_length=1024,
        logging_steps=10,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=2,
        fp16=True,
        bf16=False,
        optim="adamw_8bit",
        report_to=report_to,
        seed=42,
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        processing_class=tokenizer,
        train_dataset=ds,
        args=dpo_args,
    )
    trainer.train(resume_from_checkpoint=resume)

    hours = round((time.time() - t0) / 3600.0, 3)
    print(f"DPO training finished in {hours} T4 hours. Running held-out validation...")

    metrics = run_heldout_eval(
        model,
        tokenizer,
        dataset_repo=args.dataset,
        max_samples=args.eval_samples,
        out_json=args.drive_dir / "dpo_diag.json",
    )
    metrics["train_hours"] = hours
    (args.drive_dir / "dpo_diag.json").write_text(json.dumps(metrics, indent=2) + "\n")

    if report_to == "wandb":
        import wandb  # type: ignore

        wandb.log({f"val/{k}": v for k, v in metrics.items()})

    model.push_to_hub_merged(args.push_model, tokenizer, save_method="merged_16bit", token=os.environ.get("HF_TOKEN"), private=False)
    print(f"DPO model pushed to https://huggingface.co/{args.push_model}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
