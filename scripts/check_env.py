import sys


def fail(msg: str) -> None:
    print(f"[FAIL] {msg}")
    sys.exit(1)


def ok(msg: str) -> None:
    print(f"[ OK ] {msg}")


def main() -> None:
    import torch

    if not torch.cuda.is_available():
        fail("torch.cuda.is_available() is False — no GPU visible to PyTorch")
    ok(f"PyTorch {torch.__version__}, CUDA {torch.version.cuda}")

    name = torch.cuda.get_device_name(0)
    cc_major, cc_minor = torch.cuda.get_device_capability(0)
    total_vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    ok(f"GPU: {name}, compute capability {cc_major}.{cc_minor}, {total_vram_gb:.1f} GB VRAM")

    if total_vram_gb < 15:
        fail(f"Expected ~16GB VRAM card, found {total_vram_gb:.1f} GB — VRAM budget in the outline won't hold")

    try:
        import bitsandbytes as bnb
    except ImportError as e:
        fail(f"bitsandbytes import failed: {e}")
    ok(f"bitsandbytes {bnb.__version__}")

    try:
        import transformers, peft, trl, accelerate, datasets
    except ImportError as e:
        fail(f"core library import failed: {e}")
    ok(
        f"transformers {transformers.__version__}, peft {peft.__version__}, "
        f"trl {trl.__version__}, accelerate {accelerate.__version__}, datasets {datasets.__version__}"
    )

    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model

    tiny_model_id = "Qwen/Qwen2.5-0.5B-Instruct"
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    try:
        tok = AutoTokenizer.from_pretrained(tiny_model_id)
        model = AutoModelForCausalLM.from_pretrained(
            tiny_model_id, quantization_config=bnb_config, device_map={"": 0}
        )
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        )
        model = get_peft_model(model, lora_config)

        inputs = tok("Xin chào, đây là bài kiểm tra.", return_tensors="pt").to("cuda")
        out = model(**inputs, labels=inputs["input_ids"])
        out.loss.backward()
    except Exception as e:
        fail(f"4-bit NF4 load + LoRA forward/backward failed on this GPU: {e}")

    ok("4-bit NF4 quantized load + LoRA forward/backward pass succeeded")
    print("\nEnvironment check passed. Ready for Phase 1 (data preparation).")


if __name__ == "__main__":
    main()
