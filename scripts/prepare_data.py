import hashlib
import json
import random
from pathlib import Path

from datasets import concatenate_datasets, load_dataset
from transformers import AutoTokenizer

SEED = 42
MIN_ARTICLE_WORDS = 100
MIN_SUMMARY_WORDS = 20
N_TRAIN, N_VAL, N_TEST = 15_000, 1_000, 1_000
MAX_SEQ_LENGTH = 1280
INSTRUCTION = "Tóm tắt bài báo sau trong 3-4 câu."
TOKENIZER_ID = "Qwen/Qwen2.5-7B-Instruct"

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"


def word_count(text: str) -> int:
    return len(text.split())


def article_hash(article: str) -> str:
    return hashlib.sha256(article[:200].encode("utf-8")).hexdigest()


def load_pool():
    print("Loading Yuhthe/vietnews (train + validation + test splits)...")
    ds = load_dataset("Yuhthe/vietnews")
    pool = concatenate_datasets([ds["train"], ds["validation"], ds["test"]])
    print(f"Raw pool: {len(pool)} examples")
    return pool


def filter_and_dedup(pool):
    seen_hashes = set()
    kept = []
    for ex in pool:
        article = ex["article"].strip()
        summary = ex["abstract"].strip()
        if word_count(article) < MIN_ARTICLE_WORDS:
            continue
        if word_count(summary) < MIN_SUMMARY_WORDS:
            continue
        h = article_hash(article)
        if h in seen_hashes:
            continue
        seen_hashes.add(h)
        kept.append({"article": article, "summary": summary})

    kept_pct = 100 * len(kept) / len(pool)
    print(f"After filtering + dedup: {len(kept)} / {len(pool)} ({kept_pct:.1f}%)")
    return kept


def truncate_middle(text: str, max_words: int) -> str:
    words = text.split()
    if len(words) <= max_words:
        return text
    head = max_words // 2
    tail = max_words - head
    return " ".join(words[:head] + words[-tail:])


def to_chatml(example: dict) -> dict:
    article = truncate_middle(example["article"], max_words=900)
    return {
        "messages": [
            {"role": "user", "content": f"{INSTRUCTION}\n\n{article}"},
            {"role": "assistant", "content": example["summary"]},
        ]
    }


def compute_token_stats(records, tokenizer):
    lengths = []
    for r in records[:2000]:
        text = tokenizer.apply_chat_template(r["messages"], tokenize=False)
        lengths.append(len(tokenizer(text)["input_ids"]))
    lengths.sort()
    n = len(lengths)
    p50 = lengths[n // 2]
    p90 = lengths[int(n * 0.9)]
    p99 = lengths[int(n * 0.99)]
    coverage = sum(1 for l in lengths if l <= MAX_SEQ_LENGTH) / n
    print(f"Token length (sampled {n}): p50={p50}, p90={p90}, p99={p99}")
    print(f"Coverage at max_seq_length={MAX_SEQ_LENGTH}: {coverage:.1%}")


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    pool = load_pool()
    kept = filter_and_dedup(pool)

    rng = random.Random(SEED)
    rng.shuffle(kept)

    needed = N_TRAIN + N_VAL + N_TEST
    if len(kept) < needed:
        raise RuntimeError(f"Only {len(kept)} examples survived filtering, need {needed}")

    train_raw = kept[:N_TRAIN]
    val_raw = kept[N_TRAIN : N_TRAIN + N_VAL]
    test_raw = kept[N_TRAIN + N_VAL : N_TRAIN + N_VAL + N_TEST]

    splits = {"train": train_raw, "validation": val_raw, "test": test_raw}
    formatted = {name: [to_chatml(ex) for ex in rows] for name, rows in splits.items()}

    for name, rows in formatted.items():
        out_path = PROCESSED_DIR / f"{name}.jsonl"
        with open(out_path, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"Wrote {len(rows)} examples to {out_path}")

    print("\nComputing token-length statistics with the Qwen2.5 tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_ID)
    compute_token_stats(formatted["train"], tokenizer)

    print("\nPhase 1 (data preparation) complete. Ready for Phase 2 (training).")


if __name__ == "__main__":
    main()
