import io


def extract_text(file_bytes: bytes, filename: str) -> str:
    ext = filename.lower().rsplit(".", 1)[-1] if "." in filename else ""

    if ext == "txt":
        return file_bytes.decode("utf-8", errors="replace")

    if ext == "docx":
        import docx

        doc = docx.Document(io.BytesIO(file_bytes))
        paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
        return "\n\n".join(paragraphs)

    if ext == "pdf":
        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(file_bytes))
        pages = [page.extract_text() or "" for page in reader.pages]
        return "\n\n".join(p.strip() for p in pages if p.strip())

    raise ValueError(f"Không hỗ trợ định dạng .{ext} — chỉ nhận .docx, .pdf, .txt")
