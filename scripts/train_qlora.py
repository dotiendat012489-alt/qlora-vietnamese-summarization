import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
SEED = 42
FULL_RUN_STEPS = 1875
WARMUP_RATIO = 0.03

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "processed"
OUTPUT_DIR = ROOT / "checkpoints" / "qwen2.5-7b-qlora-vinews"
REPORTS_DIR = ROOT / "reports"


def build_model_and_tokenizer():
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map={"": 0},
        dtype=torch.bfloat16,
    )
    return model, tokenizer


def build_lora_config() -> LoraConfig:
    return LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    )


def build_training_args(
    max_steps: int = -1,
    output_dir: Path = OUTPUT_DIR,
    batch_size: int = 2,
    grad_accum: int = 8,
    grad_checkpointing: bool = True,
    eval_during_smoke: bool = False,
    max_seq_length: int = 1280,
) -> SFTConfig:
    smoke_test = max_steps > 0
    warmup_steps = max(1, round(WARMUP_RATIO * (max_steps if smoke_test else FULL_RUN_STEPS)))
    do_eval = eval_during_smoke if smoke_test else True
    return SFTConfig(
        output_dir=str(output_dir),
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_steps=warmup_steps,
        num_train_epochs=2,
        max_steps=max_steps,
        bf16=True,
        gradient_checkpointing=grad_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False} if grad_checkpointing else None,
        optim="paged_adamw_8bit",
        max_length=max_seq_length,
        assistant_only_loss=True,
        packing=False,
        eval_strategy="steps" if do_eval else "no",
        eval_steps=5 if smoke_test else 200,
        logging_steps=1 if smoke_test else 10,
        save_strategy="no" if smoke_test else "steps",
        save_steps=200,
        save_total_limit=3,
        seed=SEED,
        report_to="none",
    )


def plot_loss_curve(trainer: SFTTrainer) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    history = trainer.state.log_history

    train_steps = [h["step"] for h in history if "loss" in h]
    train_loss = [h["loss"] for h in history if "loss" in h]
    eval_steps = [h["step"] for h in history if "eval_loss" in h]
    eval_loss = [h["eval_loss"] for h in history if "eval_loss" in h]

    plt.figure(figsize=(8, 5))
    plt.plot(train_steps, train_loss, label="train_loss")
    if eval_loss:
        plt.plot(eval_steps, eval_loss, label="eval_loss", marker="o")
    plt.xlabel("step")
    plt.ylabel("loss")
    plt.title("QLoRA training loss — Qwen2.5-7B-Instruct on VietNews summarization")
    plt.legend()
    plt.grid(alpha=0.3)
    out_path = REPORTS_DIR / "training_loss.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved loss curve to {out_path}")

    with open(REPORTS_DIR / "training_log_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--max-steps",
        type=int,
        default=-1,
        help="Smoke-test override: run only N optimizer steps instead of the full 2 epochs "
        "(outline section 7, phase 2: 'chạy thử 100 bước để soát lỗi' before the real run).",
    )
    parser.add_argument("--batch-size", type=int, default=2, help="per_device_train_batch_size")
    parser.add_argument("--grad-accum", type=int, default=8, help="gradient_accumulation_steps")
    parser.add_argument("--no-grad-checkpointing", action="store_true")
    parser.add_argument("--max-seq-length", type=int, default=1280)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from the latest checkpoint-N in OUTPUT_DIR (adapter + optimizer + "
        "scheduler + RNG state), e.g. after pausing training to shut the machine down.",
    )
    parser.add_argument(
        "--data-suffix",
        default="",
        help="Load train{suffix}.jsonl / validation{suffix}.jsonl instead of the plain files, "
        "e.g. --data-suffix _detailed for the detailed-summary retrain. Also namespaces the "
        "checkpoint output dir so it doesn't overwrite a differently-trained adapter.",
    )
    parser.add_argument(
        "--save-smoke-checkpoint",
        action="store_true",
        help="Save the adapter even in --max-steps smoke-test mode, so it can be loaded for a "
        "quick generation check (e.g. verifying the adapter actually shifts output style, not "
        "just checking the loss curve).",
    )
    args = parser.parse_args()
    smoke_test = args.max_steps > 0
    run_output_dir = OUTPUT_DIR.with_name(OUTPUT_DIR.name + args.data_suffix)
    output_dir = ROOT / "checkpoints" / "smoke_test" if smoke_test else run_output_dir

    train_ds = load_dataset("json", data_files=str(DATA_DIR / f"train{args.data_suffix}.jsonl"), split="train")
    val_ds = load_dataset("json", data_files=str(DATA_DIR / f"validation{args.data_suffix}.jsonl"), split="train")
    print(f"train: {len(train_ds)}, validation: {len(val_ds)}")

    model, tokenizer = build_model_and_tokenizer()
    lora_config = build_lora_config()
    training_args = build_training_args(
        max_steps=args.max_steps,
        output_dir=output_dir,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        grad_checkpointing=not args.no_grad_checkpointing,
        max_seq_length=args.max_seq_length,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        peft_config=lora_config,
    )

    print(f"Trainable params: {sum(p.numel() for p in trainer.model.parameters() if p.requires_grad):,}")

    resume_path = None
    if args.resume:
        checkpoints = sorted(
            output_dir.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[-1])
        )
        if not checkpoints:
            raise RuntimeError(f"--resume given but no checkpoint-* found in {output_dir}")
        resume_path = str(checkpoints[-1])
        print(f"Resuming from {resume_path}")

    import time

    start = time.monotonic()
    trainer.train(resume_from_checkpoint=resume_path)
    elapsed = time.monotonic() - start

    peak_vram_gb = torch.cuda.max_memory_allocated() / 1e9
    print(f"\nElapsed: {elapsed:.1f}s for {trainer.state.global_step} steps "
          f"({elapsed / max(trainer.state.global_step, 1):.2f}s/step avg, includes eval passes)")
    print(f"Peak VRAM allocated: {peak_vram_gb:.2f} GB")

    if smoke_test and not args.save_smoke_checkpoint:
        print("\nSmoke test finished (no checkpoint saved). Inspect the loss values above:")
        print("expected to start around 2.0-2.5 per the outline; NaN/inf or a flat loss means")
        print("something in the masking/formatting pipeline is broken before committing to the full run.")
        return

    if smoke_test:
        print("\nSmoke test finished, saving checkpoint (--save-smoke-checkpoint) for a generation check.")

    trainer.save_model(str(run_output_dir / "final"))
    tokenizer.save_pretrained(str(run_output_dir / "final"))
    print(f"Saved final adapter to {run_output_dir / 'final'}")

    plot_loss_curve(trainer)
    print("\nPhase 2 (training) complete. Ready for Phase 3 (evaluation).")


if __name__ == "__main__":
    main()
