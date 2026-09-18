# Tóm tắt văn bản tiếng Việt (QLoRA)

Dự án cá nhân gồm hai phần: (1) fine-tune Qwen2.5-7B-Instruct bằng QLoRA (4-bit
NF4 + LoRA) để tóm tắt tiếng Việt, huấn luyện hoàn toàn trên GPU cá
nhân; (2) một demo Streamlit tóm tắt văn bản tiếng
Việt bất kỳ (tản văn, thư từ, báo cáo, ghi chú...), dán trực tiếp hoặc tải file
`.docx`/`.pdf`/`.txt`.

## Kết quả

**Demo tóm tắt văn bản** (model gốc `qwen2.5:7b` + chỉ thị prompt tự viết,
kiến trúc map-reduce cho văn bản dài tùy ý): qua đọc đối chiếu thủ công 110
bản tóm tắt với văn bản gốc — 96% văn bản ngắn (<1000 từ) và 64% văn bản dài
(>1000 từ) đạt chuẩn (đúng ý chính, không bịa đặt, không sao chép nguyên
văn).

**Giới hạn đã biết**: không có LLM nào đảm bảo tuyệt đối 0% bịa đặt hay sai
sót. Luôn đối chiếu lại bản tóm tắt với văn bản gốc trước khi dùng cho việc
quan trọng.

## Cấu trúc project

```
demo/                 # Streamlit demo
  app.py                 # giao diện
  summarize.py            # logic tóm tắt (single-pass + map-reduce), MODES dict
  extract_text.py          # trích văn bản từ .docx/.pdf/.txt
scripts/               # pipeline huấn luyện QLoRA
  check_env.py, prepare_data.py, generate_detailed_targets.py,
  train_qlora.py, evaluate.py, package_deploy.py
checkpoints/            # adapter LoRA, merged fp16, GGUF
data/                   # dữ liệu train/val/test VietNews
reports/                # kết quả đánh giá (ROUGE/judge), biểu đồ loss huấn luyện
```

## Cài đặt & chạy demo

Demo chỉ gọi model qua Ollama (HTTP), **không cần torch/CUDA** — cài nhẹ và
nhanh:

```bash
# 1. Cài Ollama: https://ollama.com, sau đó tải model
ollama pull qwen2.5:7b

# 2. Cài các thư viện demo cần
python -m venv .venv && source .venv/bin/activate   # hoặc conda
pip install streamlit requests python-docx pypdf

# 3. Chạy
streamlit run demo/app.py
```

Mở trình duyệt tại `http://localhost:8501`, dán văn bản hoặc tải file lên,
bấm "Tóm tắt".
