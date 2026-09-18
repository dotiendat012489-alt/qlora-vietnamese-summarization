import io
import subprocess
import sys
import threading
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

import torch
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
ROOT = Path(__file__).resolve().parent.parent
ADAPTER_DIR = ROOT / "checkpoints" / "qwen2.5-7b-qlora-vinews" / "final"
DATA_DIR = ROOT / "data" / "processed"

stop_monitor = threading.Event()


def step(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def gpu_monitor():
    while not stop_monitor.is_set():
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader"],
            capture_output=True, text=True,
        ).stdout.strip()
        step(f"  [gpu] {out}")
        stop_monitor.wait(15)


def main():
    ds = load_dataset("json", data_files=str(DATA_DIR / "test.jsonl"), split="train")
    articles = [ds[i]["messages"][0]["content"] for i in range(8)]
    lengths = [len(a) for a in articles]
    step(f"loaded 8 real test articles, char lengths: {lengths}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    texts = [tokenizer.apply_chat_template([{"role": "user", "content": a}], tokenize=False, add_generation_prompt=True) for a in articles]
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=4096)
    step(f"tokenized batch, input_ids shape={inputs['input_ids'].shape}")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    base = AutoModelForCausalLM.from_pretrained(MODEL_ID, quantization_config=bnb_config, device_map={"": 0}, dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(base, str(ADAPTER_DIR))
    model.eval()
    step("model + adapter loaded")

    inputs = inputs.to("cuda")

    monitor_thread = threading.Thread(target=gpu_monitor, daemon=True)
    monitor_thread.start()

    step("calling model.generate() with real batch=8, max_new_tokens=200...")
    t0 = time.monotonic()
    with torch.inference_mode():
        out = model.generate(
            **inputs,
            max_new_tokens=200,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    elapsed = time.monotonic() - t0
    stop_monitor.set()
    step(f"generate() returned after {elapsed:.1f}s")

    gen_only = out[:, inputs["input_ids"].shape[1]:]
    for i, g in enumerate(gen_only):
        text = tokenizer.decode(g, skip_special_tokens=True).strip()
        step(f"  [{i}] {text[:100]}")


if __name__ == "__main__":
    main()
