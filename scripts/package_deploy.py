import json
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
ROOT = Path(__file__).resolve().parent.parent
SHORT_INSTRUCTION = "Tóm tắt bài báo sau trong 3-4 câu."
DETAILED_INSTRUCTION = (
    "Hãy tóm tắt đầy đủ và chi tiết bài báo sau, nêu rõ các thông tin, sự kiện, "
    "số liệu quan trọng, giúp người đọc nắm được toàn bộ nội dung chính mà không "
    "cần đọc bài báo gốc. Viết thành các đoạn văn liền mạch (không dùng gạch đầu "
    "dòng, không đánh số). Đi thẳng vào nội dung, không mở đầu bằng các câu như "
    "\"Bài báo này nói về\" hay \"Dựa trên thông tin\", và không thêm câu kết luận "
    "kiểu \"Tóm lại\"."
)
LLAMA_CPP_DIR = ROOT.parent / "llama.cpp"
LLAMA_BIN_DIR = ROOT / "checkpoints" / "llama-cpp-bin"


def paths_for(mode: str):
    suffix = "_detailed" if mode == "detailed" else ""
    return {
        "adapter_dir": ROOT / "checkpoints" / f"qwen2.5-7b-qlora-vinews{suffix}" / "final",
        "merged_dir": ROOT / "checkpoints" / f"merged-fp16{suffix}",
        "gguf_dir": ROOT / "checkpoints" / f"gguf{suffix}",
        "instruction": DETAILED_INSTRUCTION if mode == "detailed" else SHORT_INSTRUCTION,
        "ollama_model_name": f"qwen2.5-7b-vinews-qlora{'-detailed' if mode == 'detailed' else ''}",
    }


def merge_adapter(paths):
    merged_dir = paths["merged_dir"]
    if merged_dir.exists() and any(merged_dir.iterdir()):
        print(f"Merged model already exists at {merged_dir}, skipping merge.")
        return
    print("Loading base model in bf16 on CPU (no quantization) for a clean merge...")
    base = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16, device_map="cpu")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = PeftModel.from_pretrained(base, str(paths["adapter_dir"]))
    print("Merging LoRA weights into base weights...")
    merged = model.merge_and_unload()
    merged_dir.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(str(merged_dir))
    tokenizer.save_pretrained(str(merged_dir))
    print(f"Saved merged model to {merged_dir}")


def ensure_llama_cpp():
    if LLAMA_CPP_DIR.exists():
        return
    print(f"llama.cpp not found at {LLAMA_CPP_DIR}, cloning...")
    subprocess.run(["git", "clone", "--depth", "1", "https://github.com/ggml-org/llama.cpp", str(LLAMA_CPP_DIR)], check=True)


def ensure_quantize_binary() -> Path:
    exe_path = LLAMA_BIN_DIR / "llama-quantize.exe"
    if exe_path.exists():
        return exe_path

    LLAMA_BIN_DIR.mkdir(parents=True, exist_ok=True)
    print("Fetching latest llama.cpp Windows CPU release (for llama-quantize.exe)...")
    with urllib.request.urlopen("https://api.github.com/repos/ggml-org/llama.cpp/releases?per_page=1") as resp:
        release = json.loads(resp.read())[0]
    asset = next(a for a in release["assets"] if "bin-win-cpu-x64.zip" in a["name"])
    print(f"Downloading {asset['name']} ({asset['size'] / 1e6:.1f} MB) from release {release['tag_name']}...")

    zip_path = LLAMA_BIN_DIR / asset["name"]
    urllib.request.urlretrieve(asset["browser_download_url"], zip_path)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(LLAMA_BIN_DIR)
    zip_path.unlink()

    found = next(LLAMA_BIN_DIR.rglob("llama-quantize.exe"), None)
    if found is None:
        raise RuntimeError(f"llama-quantize.exe not found after extracting {asset['name']}")
    if found != exe_path:
        shutil.copy(found, exe_path)
    return exe_path


def convert_to_gguf(paths):
    gguf_dir = paths["gguf_dir"]
    gguf_dir.mkdir(parents=True, exist_ok=True)
    fp16_gguf = gguf_dir / "model-fp16.gguf"
    quantized_gguf = gguf_dir / "model-Q4_K_M.gguf"

    if quantized_gguf.exists():
        print(f"Quantized GGUF already exists at {quantized_gguf}, skipping conversion.")
        return quantized_gguf

    ensure_llama_cpp()
    convert_script = LLAMA_CPP_DIR / "convert_hf_to_gguf.py"
    if not fp16_gguf.exists():
        print("Converting merged HF model to GGUF (fp16)...")
        subprocess.run(
            [sys.executable, str(convert_script), str(paths["merged_dir"]), "--outfile", str(fp16_gguf), "--outtype", "f16"],
            check=True,
        )

    quantize_bin = ensure_quantize_binary()
    print("Quantizing to Q4_K_M...")
    subprocess.run([str(quantize_bin), str(fp16_gguf), str(quantized_gguf), "Q4_K_M"], check=True)
    return quantized_gguf


def write_modelfile(paths, gguf_path: Path):
    modelfile_path = paths["gguf_dir"] / "Modelfile"
    modelfile_path.write_text(
        f'''FROM {gguf_path.name}

TEMPLATE """{{{{ if .System }}}}<|im_start|>system
{{{{ .System }}}}<|im_end|>
{{{{ end }}}}{{{{ if .Prompt }}}}<|im_start|>user
{{{{ .Prompt }}}}<|im_end|>
{{{{ end }}}}<|im_start|>assistant
{{{{ .Response }}}}<|im_end|>
"""

SYSTEM """{paths["instruction"]}"""

PARAMETER stop "<|im_end|>"
PARAMETER temperature 0
PARAMETER repeat_penalty 1.3
PARAMETER num_ctx 8192
''',
        encoding="utf-8",
    )
    print(f"Wrote Ollama Modelfile to {modelfile_path}")
    return modelfile_path


def register_with_ollama(paths, modelfile_path: Path):
    ollama_bin = shutil.which("ollama") or r"C:\Users\Admin\AppData\Local\Programs\Ollama\ollama.exe"
    name = paths["ollama_model_name"]
    print(f"Registering model '{name}' with Ollama...")
    subprocess.run([ollama_bin, "create", name, "-f", str(modelfile_path)], check=True, cwd=str(paths["gguf_dir"]))
    print(f"Done. Run with: ollama run {name}")


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["short", "detailed"],
        default="detailed",
        help="'short' packages the original 3-4-sentence adapter; 'detailed' (default, current "
        "project state) packages the Llama-distilled detailed-summary adapter.",
    )
    args = parser.parse_args()
    paths = paths_for(args.mode)

    merge_adapter(paths)
    gguf_path = convert_to_gguf(paths)
    modelfile_path = write_modelfile(paths, gguf_path)
    register_with_ollama(paths, modelfile_path)

    print("\nPhase 4 (packaging) complete.")
    print(f"- LoRA adapter (push to HF Hub manually): {paths['adapter_dir']}")
    print(f"- Merged fp16 model: {paths['merged_dir']}")
    print(f"- GGUF Q4_K_M: {paths['gguf_dir'] / 'model-Q4_K_M.gguf'}")
    print(f"- Ollama model: {paths['ollama_model_name']}")


if __name__ == "__main__":
    main()
