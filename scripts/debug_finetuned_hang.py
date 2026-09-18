import sys
import time
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
ROOT = Path(__file__).resolve().parent.parent
ADAPTER_DIR = ROOT / "checkpoints" / "qwen2.5-7b-qlora-vinews" / "final"


def step(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    step("start")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    step("tokenizer loaded")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    base = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, quantization_config=bnb_config, device_map={"": 0}, dtype=torch.bfloat16
    )
    step("base model loaded (4-bit)")

    model = PeftModel.from_pretrained(base, str(ADAPTER_DIR))
    step("adapter attached (PeftModel.from_pretrained returned)")

    model.eval()
    step("model.eval() done")

    prompt = [{"role": "user", "content": "Tóm tắt bài báo sau trong 3-4 câu.\n\nHôm nay trời đẹp, nhiều người đi chơi công viên."}]
    text = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
    step(f"chat template applied, len={len(text)} chars")

    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    inputs = tokenizer([text], return_tensors="pt", padding=True).to("cuda")
    step(f"tokenized, input_ids shape={inputs['input_ids'].shape}")

    step("calling model.generate() now...")
    sys.stdout.flush()
    with torch.inference_mode():
        out = model.generate(
            **inputs,
            max_new_tokens=50,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    step("generate() returned")

    result = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    step(f"decoded output: {result!r}")


if __name__ == "__main__":
    main()
