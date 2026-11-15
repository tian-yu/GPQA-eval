"""Chat template, OLMo-style EOS fix, and private W&B / Drive helpers for Qwen3 Base models."""

from __future__ import annotations

import os
from pathlib import Path

EOT = "<|endoftext|>"
IM_END = "<|im_end|>"
GLITCH_TOKEN = "\tNdrFc"  # Qwen3 vocab id 70564 emitted when <|im_end|> is undertrained

OLMO_CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "{% if message['role'] == 'assistant' %}"
    "{{ '<|im_start|>assistant\\n' + message['content'] }}"
    "{% if loop.last %}{{ '<|endoftext|>' }}{% else %}{{ '<|im_end|>\\n' }}{% endif %}"
    "{% else %}"
    "{{ '<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>\\n' }}"
    "{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}"
)

INSTRUCTION_PART = "<|im_start|>user\n"
RESPONSE_PART = "<|im_start|>assistant\n"
STOP_STRINGS = [EOT, IM_END, "<|im_start|>"]


def configure_tokenizer_and_eos(tokenizer, model=None) -> tuple[int, list[int]]:
    """Apply OLMo-style chat template and set <|endoftext|> as primary EOS token."""
    tokenizer.chat_template = OLMO_CHAT_TEMPLATE
    tokenizer.eos_token = EOT
    if tokenizer.pad_token is None:
        tokenizer.pad_token = EOT

    eot_id = tokenizer.convert_tokens_to_ids(EOT)
    im_end_id = tokenizer.convert_tokens_to_ids(IM_END)
    eos_ids = [i for i in (eot_id, im_end_id) if isinstance(i, int) and i >= 0]
    if not eos_ids:
        eos_ids = [tokenizer.eos_token_id]

    if model is not None and getattr(model, "generation_config", None) is not None:
        model.generation_config.eos_token_id = eos_ids
        model.generation_config.pad_token_id = tokenizer.pad_token_id

    return eot_id, eos_ids


def load_colab_secrets() -> None:
    """Load HF_TOKEN and WANDB_API_KEY from Colab Secrets into os.environ if present."""
    try:
        from google.colab import userdata  # type: ignore
    except ImportError:
        return
    for key in ("HF_TOKEN", "WANDB_API_KEY"):
        if not os.environ.get(key):
            try:
                val = userdata.get(key)
                if val:
                    os.environ[key] = val
            except Exception:
                pass


def setup_wandb(project: str = "gpqa-ninja", run_name: str = "run", config: dict | None = None) -> str:
    """Enable private W&B logging if WANDB_API_KEY is available, else disable cleanly."""
    load_colab_secrets()
    if not os.environ.get("WANDB_API_KEY"):
        os.environ["WANDB_MODE"] = "disabled"
        return "none"

    os.environ["WANDB_PROJECT"] = project
    os.environ["WANDB_LOG_MODEL"] = "false"
    os.environ["WANDB_SILENT"] = "true"
    try:
        import wandb  # type: ignore

        wandb.init(
            project=project,
            name=run_name,
            config=config or {},
            reinit=True,
            settings=wandb.Settings(quiet=True),
        )
        return "wandb"
    except Exception:
        os.environ["WANDB_MODE"] = "disabled"
        return "none"


def latest_checkpoint(ckpt_dir: Path) -> str | None:
    """Return path to latest Hugging Face checkpoint-* subdirectory if one exists."""
    if not ckpt_dir.exists():
        return None
    ckpts = []
    for p in ckpt_dir.iterdir():
        if p.is_dir() and p.name.startswith("checkpoint-"):
            try:
                ckpts.append((int(p.name.split("-")[-1]), p))
            except ValueError:
                continue
    if not ckpts:
        return None
    ckpts.sort(key=lambda x: x[0])
    return str(ckpts[-1][1])
