import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

GENERATOR_MODEL_ID = "NousResearch/Meta-Llama-3.1-8B-Instruct"
OLD_INSTRUCTION = "Tóm tắt bài báo sau trong 3-4 câu."
NEW_INSTRUCTION = (
    "Hãy tóm tắt đầy đủ và chi tiết bài báo sau, nêu rõ các thông tin, sự kiện, "
    "số liệu quan trọng, giúp người đọc nắm được toàn bộ nội dung chính mà không "
    "cần đọc bài báo gốc. Viết thành các đoạn văn liền mạch (không dùng gạch đầu "
    "dòng, không đánh số). Đi thẳng vào nội dung, không mở đầu bằng các câu như "
    "\"Bài báo này nói về\" hay \"Dựa trên thông tin\", và không thêm câu kết luận "
    "kiểu \"Tóm lại\"."
)
MAX_NEW_TOKENS = 700
MAX_INPUT_LENGTH = 2048
GEN_BATCH_SIZE = 8
SPLITS = ["train", "validation", "test"]

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "processed"


def load_articles(split: str, limit=None):
    path = DATA_DIR / f"{split}.jsonl"
    articles = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            ex = json.loads(line)
            user_content = ex["messages"][0]["content"]
            assert user_content.startswith(OLD_INSTRUCTION), f"unexpected format: {user_content[:80]!r}"
            article = user_content[len(OLD_INSTRUCTION) + 2 :]
            articles.append(article)
    if limit:
        articles = articles[:limit]
    return articles


def out_path(split: str, limit) -> Path:
    suffix = f"_smoke{limit}" if limit else ""
    return DATA_DIR / f"{split}_detailed{suffix}.jsonl"


@torch.inference_mode()
def generate_batch(model, tokenizer, articles: list[str]) -> list[str]:
    prompts = [[{"role": "user", "content": f"{NEW_INSTRUCTION}\n\n{a}"}] for a in articles]
    texts = [tokenizer.apply_chat_template(p, tokenize=False, add_generation_prompt=True) for p in prompts]
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    inputs = tokenizer(
        texts, return_tensors="pt", padding=True, truncation=True, max_length=MAX_INPUT_LENGTH
    ).to("cuda")
    out = model.generate(
        **inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False, pad_token_id=tokenizer.pad_token_id
    )
    gen_only = out[:, inputs["input_ids"].shape[1] :]
    return [tokenizer.decode(g, skip_special_tokens=True).strip() for g in gen_only]


def process_split(model, tokenizer, split: str, limit):
    articles = load_articles(split, limit)
    path = out_path(split, limit)

    done = []
    if path.exists():
        with open(path, encoding="utf-8") as f:
            done = [json.loads(l) for l in f]
        print(f"[{split}] resuming: {len(done)}/{len(articles)} already done")

    import time

    start_idx = len(done)
    results = done
    t0 = time.monotonic()
    with open(path, "a", encoding="utf-8") as f:
        for i in range(start_idx, len(articles), GEN_BATCH_SIZE):
            batch = articles[i : i + GEN_BATCH_SIZE]
            summaries = generate_batch(model, tokenizer, batch)
            for article, summary in zip(batch, summaries):
                record = {
                    "messages": [
                        {"role": "user", "content": f"{NEW_INSTRUCTION}\n\n{article}"},
                        {"role": "assistant", "content": summary},
                    ]
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                results.append(record)
            f.flush()
            n_done = i + len(batch) - start_idx
            elapsed = time.monotonic() - t0
            print(
                f"[{split}] {i + len(batch)}/{len(articles)} "
                f"({elapsed / n_done:.2f}s/example avg, this run)",
                flush=True,
            )

    print(f"[{split}] done: {len(results)} examples -> {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N examples per split (smoke test).")
    parser.add_argument("--split", choices=SPLITS, default=None, help="Process only this split (default: all).")
    args = parser.parse_args()

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(GENERATOR_MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        GENERATOR_MODEL_ID, quantization_config=bnb_config, device_map={"": 0}, dtype=torch.bfloat16
    )

    splits = [args.split] if args.split else SPLITS
    for split in splits:
        process_split(model, tokenizer, split, args.limit)

    print("\nDetailed target generation complete.")


if __name__ == "__main__":
    main()
