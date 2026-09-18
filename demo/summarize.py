import re

import requests

OLLAMA_URL = "http://localhost:11434/api/generate"

GENERAL_INSTRUCTION = (
    "Hãy tóm tắt văn bản sau, bao quát đầy đủ CÁC Ý CHÍNH (không phải đầy đủ từng câu chữ), "
    "theo đúng các yêu cầu sau:\n"
    "- BẮT BUỘC viết ngắn gọn hơn đáng kể so với văn bản gốc - độ dài bản tóm tắt chỉ nên bằng "
    "khoảng một phần ba đến một nửa văn bản gốc. Nếu bản tóm tắt của bạn dài gần bằng hoặc dài hơn "
    "văn bản gốc, đó là làm sai yêu cầu.\n"
    "- Diễn đạt lại bằng câu chữ của riêng bạn. TUYỆT ĐỐI KHÔNG chép nguyên các câu hoặc đoạn dài "
    "từ văn bản gốc - chỉ được giữ nguyên tên riêng, địa danh, số liệu, ngày tháng, hoặc một trích "
    "dẫn trực tiếp ngắn khi thực sự cần thiết, còn câu văn xung quanh phải được viết lại.\n"
    "- Chỉ dùng thông tin CÓ THẬT trong văn bản gốc. TUYỆT ĐỐI KHÔNG suy diễn, không tự thêm lý do, "
    "nguyên nhân, hay mối liên hệ mà văn bản gốc không nói rõ.\n"
    "- Chia thành các đoạn văn theo từng chủ đề. Mỗi đoạn bắt đầu bằng 1-2 câu giới thiệu chủ đề.\n"
    "- Nếu một đoạn có nhiều ý phụ hoặc chi tiết cụ thể, hãy liệt kê MỖI Ý PHỤ TRÊN MỘT DÒNG RIÊNG, "
    'dòng đó bắt đầu bằng "- ".\n'
    "- Không dùng tiêu đề, không dùng markdown (không dùng dấu #, không dùng chữ in đậm **, "
    "không đánh số thứ tự 1. 2. 3.), chỉ viết đoạn văn thường và gạch đầu dòng như mô tả trên.\n\n"
    "Văn bản cần tóm tắt:"
)

MODES = {
    "news": {
        "label": "Tin tức",
        "model": "qwen2.5-7b-vinews-qlora-detailed",
        "instruction": None,
        "description": "Adapter QLoRA tinh chỉnh riêng cho tóm tắt tin tức báo chí tiếng Việt.",
    },
    "general": {
        "label": "Văn bản khác",
        "model": "qwen2.5:7b",
        "instruction": GENERAL_INSTRUCTION,
        "description": "Qwen2.5-7B-Instruct gốc — phù hợp cho bài văn, tài liệu, ghi chú, văn bản không phải tin tức.",
    },
}

SINGLE_PASS_MAX_WORDS = 600
MAX_CHUNK_WORDS = 500

REDUCE_MODEL = "qwen2.5:7b"
REDUCE_INSTRUCTION = (
    "Dưới đây là các đoạn tóm tắt của những phần liên tiếp trong một văn bản dài. "
    "Hãy tổng hợp lại thành một bản tóm tắt hoàn chỉnh, mạch lạc, theo đúng các yêu cầu sau:\n"
    "- CHỈ dùng thông tin có trong các đoạn tóm tắt dưới đây, không thêm bất kỳ thông tin "
    "nào khác không có trong đó.\n"
    "- TUYỆT ĐỐI KHÔNG suy diễn, không tự thêm lý do, nguyên nhân, hay mối liên hệ mà các đoạn "
    "tóm tắt không nói rõ. Nếu không chắc chắn về một chi tiết, hãy bỏ qua chi tiết đó thay vì đoán.\n"
    "- Các đoạn tóm tắt có thể nói về những chủ đề, sự việc, con người khác nhau và KHÔNG liên quan "
    "đến nhau. TUYỆT ĐỐI KHÔNG gán tên riêng, chức danh, số liệu, hay câu nói của chủ đề/sự việc "
    "này cho chủ đề/sự việc khác - chỉ được kết hợp các chi tiết khi chúng thực sự thuộc cùng một "
    "chủ đề, sự việc.\n"
    "- Bản tổng hợp phải NGẮN GỌN hơn tổng các đoạn tóm tắt bên dưới cộng lại - đây là bước cô đọng "
    "thêm một lần nữa, không phải chỉ nối các đoạn lại và thêm gạch đầu dòng. Loại bỏ những chi tiết "
    "trùng lặp hoặc không quan trọng giữa các đoạn.\n"
    "- Chia thành các đoạn văn theo từng chủ đề, giữ đúng trình tự xuất hiện trong văn bản gốc. "
    "Mỗi đoạn bắt đầu bằng 1-2 câu giới thiệu chủ đề.\n"
    "- Nếu một đoạn có nhiều ý phụ hoặc chi tiết cụ thể, hãy liệt kê MỖI Ý PHỤ TRÊN MỘT DÒNG RIÊNG, "
    'dòng đó bắt đầu bằng "- ". TUYỆT ĐỐI KHÔNG nối nhiều ý phụ trên cùng một dòng bằng dấu " - " '
    "ở giữa câu — mỗi ý phụ phải xuống dòng riêng.\n"
    "- Không dùng tiêu đề, không dùng markdown (không dùng dấu #, không dùng chữ in đậm), "
    "chỉ viết đoạn văn thường và gạch đầu dòng như mô tả trên.\n\n"
    "Ví dụ đúng định dạng:\n"
    "Đây là câu giới thiệu chủ đề của đoạn văn thứ nhất.\n"
    "- Đây là ý phụ thứ nhất.\n"
    "- Đây là ý phụ thứ hai.\n\n"
    "Đây là câu giới thiệu chủ đề của đoạn văn thứ hai, không có ý phụ nào nên không cần gạch đầu dòng.\n\n"
    "Các đoạn tóm tắt cần tổng hợp:"
)
MAX_REDUCE_DEPTH = 5


def split_into_chunks(text: str, max_words: int = MAX_CHUNK_WORDS) -> list[str]:
    paragraphs = [p for p in text.split("\n") if p.strip()]
    chunks, current, current_words = [], [], 0

    def flush():
        if current:
            chunks.append("\n".join(current))

    for para in paragraphs:
        para_words = len(para.split())
        if para_words > max_words:
            flush()
            current, current_words = [], 0
            sentences = para.split(". ")
            piece, piece_words = [], 0
            for sent in sentences:
                sw = len(sent.split())
                if piece_words + sw > max_words and piece:
                    chunks.append(". ".join(piece) + ".")
                    piece, piece_words = [], 0
                piece.append(sent)
                piece_words += sw
            if piece:
                chunks.append(". ".join(piece) + ".")
            continue

        if current_words + para_words > max_words and current:
            flush()
            current, current_words = [], 0
        current.append(para)
        current_words += para_words

    flush()
    return chunks


NEWS_MODEL_OPTIONS = {"temperature": 0, "repeat_penalty": 1.3}
GENERAL_MODEL_OPTIONS = {"temperature": 0}

_CJK_RE = re.compile(r"[一-鿿]")
_RETRY_TEMPERATURES = [0.3, 0.5, 0.7]


def _has_cjk(text: str) -> bool:
    return bool(_CJK_RE.search(text))


def summarize_chunk(chunk: str, model: str, instruction: str | None = None, timeout: int = 240) -> str:
    prompt = f"{instruction}\n\n{chunk}" if instruction else chunk
    base_options = NEWS_MODEL_OPTIONS if model == MODES["news"]["model"] else GENERAL_MODEL_OPTIONS

    def call(options: dict) -> str:
        resp = requests.post(
            OLLAMA_URL,
            json={"model": model, "prompt": prompt, "stream": False, "options": options},
            timeout=timeout,
        )
        resp.raise_for_status()
        return resp.json()["response"].strip()

    result = call(base_options)
    if not _has_cjk(result):
        return result

    for i, temp in enumerate(_RETRY_TEMPERATURES):
        retry_options = {**base_options, "temperature": temp, "seed": i + 1}
        result = call(retry_options)
        if not _has_cjk(result):
            return result

    return result


def _reduce(summaries: list[str], depth: int = 0) -> str:
    combined = "\n\n".join(summaries)
    if len(combined.split()) <= SINGLE_PASS_MAX_WORDS or depth >= MAX_REDUCE_DEPTH:
        return summarize_chunk(combined, REDUCE_MODEL, REDUCE_INSTRUCTION)

    batches = split_into_chunks(combined, max_words=MAX_CHUNK_WORDS)
    reduced_batches = [summarize_chunk(b, REDUCE_MODEL, REDUCE_INSTRUCTION) for b in batches]
    return _reduce(reduced_batches, depth=depth + 1)


def summarize_document(text: str, mode: str = "news") -> tuple[str, int, bool]:
    cfg = MODES[mode]
    model, instruction = cfg["model"], cfg["instruction"]

    if len(text.split()) <= SINGLE_PASS_MAX_WORDS:
        return summarize_chunk(text, model, instruction), 1, False

    chunks = split_into_chunks(text, max_words=MAX_CHUNK_WORDS)
    part_summaries = [summarize_chunk(c, model, instruction) for c in chunks]

    if len(chunks) == 1:
        return part_summaries[0], 1, False

    final = _reduce(part_summaries)
    return final, len(chunks), True
