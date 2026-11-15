"""Stage 3: Verifiable Curriculum GRPO on Colab T4 (<=1.05h)."""

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
from rewards import grpo_reward_func  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--init-model", required=True, help="HF repo of D1 or S1 checkpoint")
    p.add_argument("--dataset", default="BabyWombat/gpqa-ninja-mix")
    p.add_argument("--push-model", required=True, help="HF repo to push merged fp16 GRPO model")
    p.add_argument("--drive-dir", type=Path, default=Path("/content/ckpt/grpo"))
    p.add_argument("--ckpt-repo", default=None, help="Optional private HF repo for intermediate checkpoints")
    p.add_argument("--lora-r", type=int, default=32)
    p.add_argument("--max-prompt-len", type=int, default=1024)
    p.add_argument("--max-completion-len", type=int, default=1536)
    p.add_argument("--num-generations", type=int, default=8)
    p.add_argument("--max-steps", type=int, default=80)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--eval-samples", type=int, default=120)
    args = p.parse_args(argv)

    load_colab_secrets()
    report_to = setup_wandb(project="gpqa-ninja", run_name=f"grpo-{args.init_model.split('/')[-1]}", config=vars(args))

    from unsloth import FastLanguageModel
    from datasets import load_dataset
    from trl import GRPOConfig, GRPOTrainer

    t0 = time.time()
    max_seq_len = args.max_prompt_len + args.max_completion_len

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.init_model,
        max_seq_length=max_seq_len,
        dtype=None,
        load_in_4bit=True,
        fast_inference=False,
        token=os.environ.get("HF_TOKEN"),
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

    ds = load_dataset(args.dataset, "grpo_pool", split="train", token=os.environ.get("HF_TOKEN")).shuffle(seed=42)

    args.drive_dir.mkdir(parents=True, exist_ok=True)
    resume = latest_checkpoint(args.drive_dir)
    if resume:
        print(f"Resuming GRPO from checkpoint: {resume}")

    grpo_args = GRPOConfig(
        output_dir=str(args.drive_dir),
        learning_rate=args.lr,
        lr_scheduler_type="constant",
        per_device_train_batch_size=args.num_generations,
        gradient_accumulation_steps=4,
        num_generations=args.num_generations,
        max_prompt_length=args.max_prompt_len,
        max_completion_length=args.max_completion_len,
        max_steps=args.max_steps,
        beta=0.0,
        epsilon=0.2,
        epsilon_high=0.28,  # DAPO clip-higher for exploration
        loss_type="dapo",
        logging_steps=5,
        save_strategy="steps",
        save_steps=25,
        save_total_limit=2,
        fp16=True,
        bf16=False,
        optim="adamw_8bit",
        log_completions=False,  # Never log completions to W&B
        report_to=report_to,
        seed=42,
        push_to_hub=bool(args.ckpt_repo),
        hub_model_id=args.ckpt_repo,
        hub_strategy="checkpoint" if args.ckpt_repo else "end",
        hub_private_repo=True,
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[grpo_reward_func],
        args=grpo_args,
        train_dataset=ds,
    )
    trainer.train(resume_from_checkpoint=resume)

    hours = round((time.time() - t0) / 3600.0, 3)
    print(f"GRPO training finished in {hours} T4 hours. Running held-out validation...")

    metrics = run_heldout_eval(
        model,
        tokenizer,
        dataset_repo=args.dataset,
        max_samples=args.eval_samples,
        out_json=args.drive_dir / "grpo_diag.json",
    )
    metrics["train_hours"] = hours
    (args.drive_dir / "grpo_diag.json").write_text(json.dumps(metrics, indent=2) + "\n")

    if report_to == "wandb":
        import wandb  # type: ignore

        wandb.log({f"val/{k}": v for k, v in metrics.items()})

    model.push_to_hub_merged(args.push_model, tokenizer, save_method="merged_16bit", token=os.environ.get("HF_TOKEN"), private=False)
    print(f"GRPO model pushed to https://huggingface.co/{args.push_model}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
