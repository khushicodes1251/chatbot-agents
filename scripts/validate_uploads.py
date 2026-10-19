#!/usr/bin/env python3
"""Checks PDF upload indexing and that the web chat ignores the old sample note."""

from __future__ import annotations

import io
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rag_chatbot import config
from rag_chatbot.uploads import (
    chunk_document_text,
    clear_uploads,
    extract_text,
    index_upload,
    list_uploaded_files,
    prepare_document_answer,
    purge_sample_note,
    remove_upload,
)
from web_app import app


def text_pdf(text: str) -> bytes:
    safe = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
    stream = f"BT /F1 18 Tf 72 720 Td ({safe}) Tj ET".encode("latin-1", errors="replace")
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R "
        b"/Resources << /Font << /F1 5 0 R >> >> >>",
        b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    out = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for i, obj in enumerate(objects, start=1):
        offsets.append(len(out))
        out.extend(f"{i} 0 obj\n".encode())
        out.extend(obj)
        out.extend(b"\nendobj\n")
    xref = len(out)
    out.extend(f"xref\n0 {len(objects)+1}\n".encode())
    out.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        out.extend(f"{offset:010d} 00000 n \n".encode())
    out.extend(
        f"trailer << /Size {len(objects)+1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF".encode()
    )
    return bytes(out)


def main() -> int:
    failures = 0

    def check(name: str, ok: bool, detail: str = "") -> None:
        nonlocal failures
        print(("PASS" if ok else "FAIL"), name, detail)
        if not ok:
            failures += 1

    chunks = chunk_document_text("Alpha policy.\n\n" + ("beta " * 400))
    check("chunk_long_text", len(chunks) >= 2, f"n={len(chunks)}")

    clear_uploads()
    sample = config.UPLOAD_DIR / "harbor-note.txt"
    sample.parent.mkdir(parents=True, exist_ok=True)
    sample.write_text("Harbor QuietBuds spare cable ships in a grey pouch with a blue clip.\n", encoding="utf-8")
    purge_sample_note()
    check("sample_note_removed", not sample.exists() and all(f["name"] != "harbor-note.txt" for f in list_uploaded_files()))

    try:
        index_upload("notes.txt", b"hello")
        check("reject_non_pdf", False, "expected ValueError")
    except ValueError as exc:
        check("reject_non_pdf", "PDF" in str(exc), str(exc))

    from pypdf import PdfWriter

    blank_buf = io.BytesIO()
    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)
    writer.write(blank_buf)
    try:
        index_upload("scan.pdf", blank_buf.getvalue())
        check("reject_scanned", False, "expected ValueError")
    except ValueError as exc:
        check("reject_scanned", "OCR" in str(exc), str(exc))

    pdf = text_pdf("Khushi led the local document search project in 2024.")
    check("pdf_has_text", "Khushi" in extract_text_bytes(pdf))
    indexed = index_upload("experience.pdf", pdf)
    check("index_pdf", indexed["chunks"] >= 1 and indexed["pages"] == 1 and not indexed["duplicate"], str(indexed))
    again = index_upload("experience-copy.pdf", pdf)
    check("duplicate_skipped", again.get("duplicate") is True, str(again))

    grounded = prepare_document_answer("What did Khushi lead?", ["experience.pdf"])
    prompt = grounded["messages"][1]["content"] if grounded["messages"] else ""
    check(
        "grounded_in_pdf",
        grounded["messages"] is not None and "Khushi" in prompt and "p.1" in prompt,
        prompt[:140],
    )
    check(
        "no_sample_in_prompt",
        "harbor-note" not in prompt.lower() and "quietbuds" not in prompt.lower(),
        prompt[:80],
    )

    web_source = (ROOT / "web_app.py").read_text(encoding="utf-8")
    check("web_does_not_call_harbor", "iter_answer_tokens" not in web_source and "pipeline" not in web_source)

    other = index_upload("other.pdf", text_pdf("The warehouse closes at 18:00."))
    check("second_pdf", other["name"] == "other.pdf" and not other["duplicate"])
    only_other = prepare_document_answer("What did Khushi lead?", ["other.pdf"])
    other_prompt = only_other["messages"][1]["content"] if only_other["messages"] else ""
    context_only = other_prompt.split("QUESTION:", 1)[0]
    check("selection_excludes_other_pdf", "Khushi" not in context_only and "18:00" in context_only, context_only[:120])

    client = app.test_client()
    page = client.get("/")
    check("api_home", page.status_code == 200 and b"Document chat" in page.data and b"/assets/" in page.data)
    react_app = (ROOT / "frontend" / "src" / "components" / "Sidebar.jsx").read_text(encoding="utf-8")
    check("react_upload_ui", "Upload PDF" in react_app and "Harbor support" not in react_app)

    bad = client.post("/api/upload", data={"file": (io.BytesIO(b"hi"), "notes.docx")})
    check("flask_reject_docx", bad.status_code == 400 and "PDF" in bad.get_json().get("error", ""))

    removed = client.delete("/api/files/experience.pdf")
    names = [item["name"] for item in removed.get_json()["files"]]
    check("remove_pdf", removed.status_code == 200 and "experience.pdf" not in names, str(names))
    empty_chat = client.post(
        "/api/chat",
        json={"question": "What did Khushi lead?", "documents": [], "mode": "harbor"},
    )
    body = empty_chat.data.decode()
    check(
        "no_selection_even_if_harbor_mode",
        "Select at least one" in body and "refund" not in body.lower() and "Khushi led" not in body,
        body[:180],
    )

    clear_uploads()
    if failures:
        print(f"Upload validation: FAIL ({failures})")
        return 1
    print("Upload validation: PASS")
    return 0


def extract_text_bytes(data: bytes) -> str:
    path = config.UPLOAD_DIR / "_probe.pdf"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    try:
        return extract_text(path)
    finally:
        path.unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())
