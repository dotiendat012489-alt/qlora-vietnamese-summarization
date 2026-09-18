import html
import re
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
from extract_text import extract_text
from summarize import MODES, summarize_document


_BULLET_PREFIX = re.compile(r"^(?:[-•]|\d{1,2}[.)])\s+")


def _clean_line(ln: str) -> str:
    ln = re.sub(r"^#+\s*", "", ln.strip())
    ln = ln.replace("**", "").replace("__", "")
    return ln.strip()


def summary_to_html(summary: str) -> str:
    blocks = [b for b in re.split(r"\n\s*\n", summary.strip()) if b.strip()]
    html_parts = []
    for block in blocks:
        lines = [_clean_line(ln) for ln in block.split("\n") if ln.strip()]
        paragraph_lines, bullet_lines = [], []
        for ln in lines:
            (bullet_lines if _BULLET_PREFIX.match(ln) else paragraph_lines).append(ln)
        if paragraph_lines:
            text = " ".join(paragraph_lines)
            html_parts.append(f"<p>{html.escape(text)}</p>")
        if bullet_lines:
            items = "".join(
                f"<li>{html.escape(_BULLET_PREFIX.sub('', ln))}</li>" for ln in bullet_lines
            )
            html_parts.append(f"<ul>{items}</ul>")
    return "\n".join(html_parts)

st.set_page_config(page_title="Tóm tắt văn bản tiếng Việt", page_icon="📝", layout="centered")

st.html(
    """
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Newsreader:opsz,wght@6..72,500;6..72,600;6..72,700&family=Inter:wght@400;500;600;700&family=IBM+Plex+Mono:wght@500;600&display=swap" rel="stylesheet">
<style>
  :root {
    --bg: #f5f3ee;
    --surface: #ffffff;
    --surface-2: #ece9e1;
    --ink: #1e2430;
    --ink-muted: #656b78;
    --ink-faint: #8c9099;
    --border: #ddd9cd;
    --accent: #3d5a80;
    --accent-ink: #23405f;
    --accent-soft: #e6ecf3;
    --good: #3d7a5c;
    --good-soft: #e3ede6;
  }
  @media (prefers-color-scheme: dark) {
    :root:not([data-theme="light"]) {
      --bg: #14171c; --surface: #1b1f26; --surface-2: #20252d;
      --ink: #e8e6df; --ink-muted: #a2a6b0; --ink-faint: #6f7480;
      --border: #2c313a; --accent: #7ca3c9; --accent-ink: #bcd6ec;
      --accent-soft: #1e2c3a; --good: #74c39a; --good-soft: #1a2e22;
    }
  }

  .stApp { background: var(--bg); }
  html, body, [class*="css"] { font-family: 'Inter', -apple-system, sans-serif; }

  .block-container { max-width: 720px; padding-top: 3.4rem; padding-bottom: 5rem; }

  /* Masthead */
  .app-brand {
    display: flex; align-items: center; gap: 10px; margin-bottom: 22px;
  }
  .app-brand .mark {
    width: 34px; height: 34px; border-radius: 9px; background: var(--accent);
    display: flex; align-items: center; justify-content: center;
    font-size: 16px; flex-shrink: 0;
  }
  .app-title {
    font-family: 'Newsreader', serif; font-weight: 600; font-size: 25px;
    color: var(--ink); letter-spacing: -.01em;
  }

  /* Tabs (paste vs upload) */
  button[data-baseweb="tab"] { font-family: 'Inter', sans-serif; font-weight: 600; font-size: 14px; }
  div[data-baseweb="tab-highlight"] { background-color: var(--accent) !important; }
  div[data-baseweb="tab-border"] { background-color: var(--border) !important; }

  /* Text area */
  div[data-testid="stTextArea"] textarea {
    background: var(--surface); border: 1.5px solid var(--border); border-radius: 10px;
    font-family: 'Inter', sans-serif; font-size: 14.5px; color: var(--ink);
  }
  div[data-testid="stTextArea"] textarea:focus { border-color: var(--accent); box-shadow: none; }

  /* File uploader */
  div[data-testid="stFileUploaderDropzone"] {
    background: var(--surface); border: 1.5px dashed var(--border); border-radius: 10px;
  }

  /* Primary button */
  div[data-testid="stButton"] button[kind="primary"] {
    background: var(--accent); border: none; border-radius: 8px; font-weight: 600;
    padding: 10px 28px; font-size: 14.5px; letter-spacing: .01em;
    box-shadow: 0 1px 3px rgba(61,90,128,.25);
  }
  div[data-testid="stButton"] button[kind="primary"]:hover {
    background: var(--accent-ink); box-shadow: 0 2px 6px rgba(61,90,128,.32);
  }

  /* Result card */
  .result-card {
    background: var(--surface); border: 1px solid var(--border); border-radius: 14px;
    padding: 26px 28px; margin-top: 6px; box-shadow: 0 2px 10px rgba(20,20,20,.04);
  }
  .result-label {
    font-family: 'IBM Plex Mono', monospace; font-size: 10.5px; letter-spacing: .08em;
    text-transform: uppercase; color: var(--good); font-weight: 600; margin-bottom: 14px;
    display: flex; align-items: center; gap: 8px;
  }
  .result-label .dot { width: 7px; height: 7px; border-radius: 50%; background: var(--good); display: inline-block; }
  .result-text { font-size: 15.5px; line-height: 1.72; color: var(--ink); }
  .result-text p { margin: 0 0 14px; }
  .result-text p:last-child { margin-bottom: 0; }
  .result-text ul { margin: -4px 0 14px; padding-left: 22px; }
  .result-text ul:last-child { margin-bottom: 0; }
  .result-text li { margin-bottom: 5px; }

  .meta-row {
    display: flex; gap: 18px; flex-wrap: wrap; margin-top: 20px; padding-top: 16px;
    border-top: 1px solid var(--border); font-family: 'IBM Plex Mono', monospace;
    font-size: 11.5px; color: var(--ink-faint);
  }
  .meta-row b { color: var(--ink-muted); font-weight: 600; }

  .app-dek { font-size: 13.5px; color: var(--ink-faint); margin: -4px 2px 26px; }

  footer, #MainMenu, header[data-testid="stHeader"] { visibility: hidden; height: 0; }
</style>
"""
)

st.html(
    '<div class="app-brand"><span class="mark">📝</span>'
    '<span class="app-title">Tóm tắt văn bản tiếng Việt</span></div>'
)
st.html(f'<div class="app-dek">{MODES["general"]["description"]}</div>')

mode_key = "general"

tab_paste, tab_upload = st.tabs(["✍️ Dán văn bản", "📎 Tải file lên"])

text_input = ""
with tab_paste:
    text_input = st.text_area(
        "Dán văn bản vào đây:",
        height=260,
        placeholder="Nội dung văn bản tiếng Việt...",
        label_visibility="collapsed",
    )

with tab_upload:
    uploaded = st.file_uploader(
        "Tải file lên", type=["docx", "pdf", "txt"], label_visibility="collapsed"
    )
    if uploaded is not None:
        try:
            extracted = extract_text(uploaded.getvalue(), uploaded.name)
            st.text_area(
                f"Nội dung trích xuất từ {uploaded.name} ({len(extracted.split())} từ) — có thể chỉnh sửa trước khi tóm tắt:",
                value=extracted,
                height=220,
                key="extracted_text",
            )
            text_input = st.session_state.get("extracted_text", extracted)
        except Exception as e:
            st.error(f"Không đọc được file: {e}")

go = st.button("Tóm tắt", type="primary", disabled=not text_input.strip())

if go:
    with st.spinner("Đang tạo tóm tắt..."):
        try:
            summary, n_chunks, used_reduce = summarize_document(text_input, mode=mode_key)
            word_count = len(text_input.split())
            summary_html = summary_to_html(summary)
            st.html(
                f"""
<div class="result-card">
  <div class="result-label"><span class="dot"></span>Tóm tắt</div>
  <div class="result-text">{summary_html}</div>
  <div class="meta-row">
    <span>Văn bản gốc: <b>{word_count}</b> từ</span>
    <span>Tóm tắt: <b>{len(summary.split())}</b> từ</span>
    <span>Nén: <b>{round(len(summary.split())/max(word_count,1)*100)}%</b></span>
    <span>Xử lý: <b>{f'{n_chunks} đoạn + tổng hợp' if used_reduce else '1 lượt'}</b></span>
  </div>
</div>
"""
            )
        except Exception as e:
            if "Connection" in type(e).__name__ or "Connection" in str(e):
                st.error(
                    "Không kết nối được Ollama tại localhost:11434. "
                    'Đảm bảo Ollama đang chạy và đã tải model bằng "ollama pull qwen2.5:7b".'
                )
            else:
                st.error(f"Lỗi: {e}")
