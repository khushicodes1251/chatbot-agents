"""Load knowledge-base text files and split them into paragraph chunks."""

from pathlib import Path


def load_text_documents(folder: Path) -> list[dict]:
    """Load every .txt file under folder (recursive) into {source, text} dicts.

    `source` is the path relative to the knowledge-base root (posix-style),
    e.g. ``refunds/refund-policy.txt``.
    """
    if not folder.is_dir():
        raise FileNotFoundError(f"Knowledge base folder not found: {folder}")

    documents = []
    for path in sorted(folder.rglob("*.txt")):
        if not path.is_file():
            continue
        rel = path.relative_to(folder).as_posix()
        documents.append(
            {"source": rel, "text": path.read_text(encoding="utf-8").strip()}
        )
    if not documents:
        raise FileNotFoundError(f"No .txt files found in {folder}")
    return documents


def chunk_by_paragraphs(text: str) -> list[str]:
    """Split on blank lines and drop empty pieces."""
    return [part.strip() for part in text.split("\n\n") if part.strip()]


def build_chunk_records(documents: list[dict]) -> list[dict]:
    """Create chunk records with source + chunk_id metadata."""
    records = []
    for doc in documents:
        for i, chunk in enumerate(chunk_by_paragraphs(doc["text"])):
            records.append(
                {"source": doc["source"], "chunk_id": i, "text": chunk}
            )
    return records


def load_chunks(folder: Path) -> list[dict]:
    """Convenience: load files and return paragraph chunks."""
    return build_chunk_records(load_text_documents(folder))
