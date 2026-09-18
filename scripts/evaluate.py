import argparse
import json
import random
import statistics
import sys
import time
from pathlib import Path

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
SHORT_INSTRUCTION = "Tóm tắt bài báo sau trong 3-4 câu."
DETAILED_INSTRUCTION = (
    "Hãy tóm tắt đầy đủ và chi tiết bài báo sau, nêu rõ các thông tin, sự kiện, "
    "số liệu quan trọng, giúp người đọc nắm được toàn bộ nội dung chính mà không "
    "cần đọc bài báo gốc. Viết thành các đoạn văn liền mạch (không dùng gạch đầu "
    "dòng, không đánh số). Đi thẳng vào nội dung, không mở đầu bằng các câu như "
    "\"Bài báo này nói về\" hay \"Dựa trên thông tin\", và không thêm câu kết luận "
    "kiểu \"Tóm lại\"."
)
SEED = 42
N_JUDGE_SAMPLES = 200
GEN_BATCH_SIZE = 8
JUDGE_MODEL = "llama3.1:8b"
OLLAMA_URL = "http://localhost:11434/api/generate"
CONFIG_NAMES = ["zero_shot", "few_shot", "qlora_finetuned"]

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "processed"
REPORTS_DIR = ROOT / "reports"

FEW_SHOT_INDICES = [0, 1]

MAX_INPUT_LENGTH = 4096


def load_test_set(suffix: str, limit=None):
    from datasets import load_dataset

    ds = load_dataset("json", data_files=str(DATA_DIR / f"test{suffix}.jsonl"), split="train")
    articles, references = [], []
    for ex in ds:
        articles.append(ex["messages"][0]["content"])
        references.append(ex["messages"][1]["content"])
    if limit:
        articles, references = articles[:limit], references[:limit]
    return articles, references


def load_few_shot_examples(suffix: str):
    from datasets import load_dataset

    ds = load_dataset("json", data_files=str(DATA_DIR / f"train{suffix}.jsonl"), split="train")
    return [(ds[i]["messages"][0]["content"], ds[i]["messages"][1]["content"]) for i in FEW_SHOT_INDICES]


def build_zero_shot_messages(articles):
    return [[{"role": "user", "content": a}] for a in articles]


def build_few_shot_messages(articles, few_shot):
    prefix = []
    for user_content, assistant_content in few_shot:
        prefix.append({"role": "user", "content": user_content})
        prefix.append({"role": "assistant", "content": assistant_content})
    return [prefix + [{"role": "user", "content": a}] for a in articles]


def pred_path(config_name: str, mode: str, limit) -> Path:
    pred_dir = REPORTS_DIR / "predictions" / mode
    suffix = f"_smoke{limit}" if limit else ""
    return pred_dir / f"{config_name}{suffix}.json"


def run_generation_subprocess(config_name: str, mode: str, limit, max_new_tokens: int):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    def build_bnb_config():
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

    def load_base_model():
        return AutoModelForCausalLM.from_pretrained(
            MODEL_ID, quantization_config=build_bnb_config(), device_map={"": 0}, dtype=torch.bfloat16
        )

    @torch.inference_mode()
    def generate_batch(model, tokenizer, prompts):
        texts = [tokenizer.apply_chat_template(p, tokenize=False, add_generation_prompt=True) for p in prompts]
        tokenizer.padding_side = "left"
        tokenizer.truncation_side = "left"
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        inputs = tokenizer(
            texts, return_tensors="pt", padding=True, truncation=True, max_length=MAX_INPUT_LENGTH
        ).to("cuda")
        out = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tokenizer.pad_token_id
        )
        gen_only = out[:, inputs["input_ids"].shape[1]:]
        return [tokenizer.decode(g, skip_special_tokens=True).strip() for g in gen_only]

    def run_generation(model, tokenizer, message_lists):
        predictions = []
        for i in range(0, len(message_lists), GEN_BATCH_SIZE):
            batch = message_lists[i : i + GEN_BATCH_SIZE]
            predictions.extend(generate_batch(model, tokenizer, batch))
            if (i // GEN_BATCH_SIZE) % 10 == 0:
                print(f"  generated {i + len(batch)}/{len(message_lists)}", flush=True)
        return predictions

    dataset_suffix = "_detailed" if mode == "detailed" else ""
    adapter_dir = ROOT / "checkpoints" / f"qwen2.5-7b-qlora-vinews{dataset_suffix}" / "final"

    articles, _ = load_test_set(dataset_suffix, limit)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    if config_name == "zero_shot":
        model = load_base_model()
        messages = build_zero_shot_messages(articles)
    elif config_name == "few_shot":
        model = load_base_model()
        messages = build_few_shot_messages(articles, load_few_shot_examples(dataset_suffix))
    elif config_name == "qlora_finetuned":
        model = PeftModel.from_pretrained(load_base_model(), str(adapter_dir))
        messages = build_zero_shot_messages(articles)
    else:
        raise ValueError(config_name)

    print(f"=== Config: {config_name} ({len(articles)} examples, mode={mode}) ===", flush=True)
    t0 = time.monotonic()
    predictions = run_generation(model, tokenizer, messages)
    print(f"{config_name} generation took {time.monotonic() - t0:.0f}s", flush=True)

    out_path = pred_path(config_name, mode, limit)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(predictions, f, ensure_ascii=False)
    print(f"Saved predictions to {out_path}", flush=True)


def compute_rouge(predictions, references):
    from rouge_score import rouge_scorer

    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=False)
    totals = {"rouge1": [], "rouge2": [], "rougeL": []}
    for pred, ref in zip(predictions, references):
        scores = scorer.score(ref, pred)
        for key in totals:
            totals[key].append(scores[key].fmeasure * 100)
    return {key: (statistics.mean(vals), statistics.pstdev(vals)) for key, vals in totals.items()}


def ollama_judge(article, reference, prediction):
    import requests

    prompt = f"""Bạn là giám khảo đánh giá chất lượng tóm tắt tin tức tiếng Việt.

Bài báo gốc:
{article}

Tóm tắt tham chiếu (chuẩn):
{reference}

Tóm tắt cần chấm điểm:
{prediction}

Chấm điểm tóm tắt cần chấm theo thang 1-10 (10 là tốt nhất) cho hai tiêu chí:
1. accuracy: độ chính xác thông tin so với bài báo gốc (không bịa đặt, không sai sự kiện)
2. fluency: độ tự nhiên, mạch lạc của văn phong tiếng Việt

Chỉ trả lời bằng JSON hợp lệ, không giải thích thêm, đúng định dạng:
{{"accuracy": <số>, "fluency": <số>}}"""

    try:
        resp = requests.post(
            OLLAMA_URL,
            json={"model": JUDGE_MODEL, "prompt": prompt, "stream": False, "format": "json"},
            timeout=120,
        )
        resp.raise_for_status()
        content = json.loads(resp.json()["response"])
        return float(content["accuracy"]), float(content["fluency"])
    except Exception as e:
        print(f"  judge call failed: {e}")
        return None


def run_llm_judge(articles, references, predictions, n_samples, seed, instruction):
    rng = random.Random(seed)
    idx = rng.sample(range(len(articles)), min(n_samples, len(articles)))
    accuracy_scores, fluency_scores = [], []
    for count, i in enumerate(idx, 1):
        article_text = articles[i].replace(f"{instruction}\n\n", "")
        result = ollama_judge(article_text, references[i], predictions[i])
        if result is not None:
            accuracy_scores.append(result[0])
            fluency_scores.append(result[1])
        if count % 25 == 0:
            print(f"  judged {count}/{len(idx)}")
    return {
        "accuracy": (statistics.mean(accuracy_scores), statistics.pstdev(accuracy_scores)),
        "fluency": (statistics.mean(fluency_scores), statistics.pstdev(fluency_scores)),
        "n": len(accuracy_scores),
    }


def run_orchestrator(mode, limit, judge_samples, max_new_tokens):
    import subprocess

    dataset_suffix = "_detailed" if mode == "detailed" else ""
    instruction = DETAILED_INSTRUCTION if mode == "detailed" else SHORT_INSTRUCTION

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    articles, references = load_test_set(dataset_suffix, limit)
    print(f"Test set: {len(articles)} examples (mode={mode})")

    for name in CONFIG_NAMES:
        out_path = pred_path(name, mode, limit)
        if out_path.exists():
            print(f"\n=== Config: {name} === (predictions already exist, skipping generation)")
            continue
        print(f"\nSpawning subprocess for config: {name}")
        cmd = [sys.executable, __file__, "--gen-only", name, "--mode", mode, "--max-new-tokens", str(max_new_tokens)]
        if limit:
            cmd += ["--limit", str(limit)]
        t0 = time.monotonic()
        subprocess.run(cmd, check=True)
        print(f"[{name}] subprocess finished in {time.monotonic() - t0:.0f}s")

    all_predictions = {}
    results = {}
    for name in CONFIG_NAMES:
        with open(pred_path(name, mode, limit), encoding="utf-8") as f:
            all_predictions[name] = json.load(f)
        results[name] = {"rouge": compute_rouge(all_predictions[name], references)}
        print(f"{name} ROUGE: { {k: round(v[0], 2) for k, v in results[name]['rouge'].items()} }")

    print("\n=== LLM-as-judge (Llama 3.1 8B via Ollama) ===")
    t_judge = time.monotonic()
    for name in CONFIG_NAMES:
        print(f"Judging config: {name}")
        results[name]["llm_judge"] = run_llm_judge(
            articles, references, all_predictions[name], judge_samples, SEED, instruction
        )
    print(f"LLM-judge phase took {time.monotonic() - t_judge:.0f}s")

    out_name = f"evaluation_results_{mode}" + ("_smoke.json" if limit else ".json")
    with open(REPORTS_DIR / out_name, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nSaved metrics to {REPORTS_DIR / out_name}")

    qual_indices = [0, 1, 2, 3]
    qual_path = REPORTS_DIR / f"qualitative_examples_{mode}.md"
    with open(qual_path, "w", encoding="utf-8") as f:
        for i in qual_indices:
            f.write(f"## Example {i}\n\n")
            article_text = articles[i].replace(f"{instruction}\n\n", "")
            f.write(f"**Article:** {article_text[:500]}...\n\n")
            f.write(f"**Reference:** {references[i]}\n\n")
            for name in CONFIG_NAMES:
                f.write(f"**{name}:** {all_predictions[name][i]}\n\n")
            f.write("---\n\n")
    print(f"Saved qualitative examples to {qual_path}")
    print("\nPhase 3 (evaluation) complete. Ready for Phase 4 (packaging/deployment).")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Evaluate only the first N test examples (smoke test).")
    parser.add_argument("--judge-samples", type=int, default=N_JUDGE_SAMPLES)
    parser.add_argument(
        "--mode",
        choices=["short", "detailed"],
        default="detailed",
        help="'short' evaluates the original 3-4-sentence adapter/data; 'detailed' (default, current "
        "project state) evaluates the Llama-distilled detailed-summary adapter/data.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=None,
        help="Defaults to 200 for --mode short, 350 for --mode detailed.",
    )
    parser.add_argument(
        "--gen-only",
        choices=CONFIG_NAMES,
        default=None,
        help="Internal: run generation for just this config in the current process, then exit. "
        "Used by the orchestrator to isolate each config in its own subprocess/CUDA context.",
    )
    args = parser.parse_args()
    max_new_tokens = args.max_new_tokens or (700 if args.mode == "detailed" else 200)

    if args.gen_only:
        run_generation_subprocess(args.gen_only, args.mode, args.limit, max_new_tokens)
    else:
        run_orchestrator(args.mode, args.limit, args.judge_samples, max_new_tokens)


if __name__ == "__main__":
    main()
