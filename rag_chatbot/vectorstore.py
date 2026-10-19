"""Local Chroma index: build, open, and query."""

import logging
import os
from pathlib import Path

# Disable Chroma telemetry before importing chromadb (avoids noisy console errors)
os.environ["ANONYMIZED_TELEMETRY"] = "False"

import chromadb
from chromadb.config import Settings

from rag_chatbot import config
from rag_chatbot.documents import load_chunks
from rag_chatbot.ollama_client import embed_text

_CHROMA_SETTINGS = Settings(anonymized_telemetry=False)
# Chroma 0.6 + some posthog versions still log capture() errors even when "disabled"
logging.getLogger("chromadb.telemetry.product.posthog").disabled = True

_CLIENT_CACHE: dict[str, chromadb.PersistentClient] = {}
_COLLECTION_CACHE: dict[str, object] = {}


def _collection_names(client: chromadb.PersistentClient) -> set[str]:
    # Chroma 0.6+ returns name-like objects; str() is the collection name
    return {str(name) for name in client.list_collections()}


def embedding_text(source: str, text: str) -> str:
    """Text used for vectors: include the filename so policy-name questions match better."""
    return f"Harbor & Co. document ({source}):\n{text}"


# Folder keywords → soft distance boost (lower distance = better). Kept small so the
# distance gate (MAX_DISTANCE on the raw best hit) stays meaningful.
_TOPIC_FOLDER_HINTS: list[tuple[tuple[str, ...], str, float]] = [
    (("cancel", "cancellation", "shipped", "in transit", "in-transit"), "cancellations/", 0.04),
    (("refund", "money back", "statement"), "refunds/", 0.035),
    (("return", "exchange", "damaged", "defective"), "returns/", 0.04),
    (("support hours", "business hours", "escalate", "human"), "support/", 0.03),
    (("bluetooth", "pair", "connect", "troubleshoot", "won't charge"), "troubleshooting/", 0.03),
    (("price", "cost", "$", "pricing"), "pricing/", 0.03),
    (("shipping", "delivery", "express"), "shipping/", 0.03),
]


def _folder_boosts(question: str) -> dict[str, float]:
    lowered = question.lower()
    boosts: dict[str, float] = {}
    for keywords, folder, amount in _TOPIC_FOLDER_HINTS:
        if any(k in lowered for k in keywords):
            boosts[folder] = max(boosts.get(folder, 0.0), amount)
    return boosts


def _source_folder(source: str) -> str:
    return source.split("/", 1)[0] + "/" if "/" in source else source


def rerank_hits(question: str, hits: list[dict], top_k: int) -> list[dict]:
    """
    Soft topic boost + folder diversity so multi-policy questions keep return+refund
    (or similar) sources in the prompt instead of three near-duplicate chunks.

    Raw cosine distances on each hit are unchanged. The closest raw hit is always
    retained so the distance gate stays conservative.
    """
    if not hits:
        return hits

    boosts = _folder_boosts(question)
    scored: list[tuple[float, int, dict]] = []
    for i, hit in enumerate(hits):
        folder = _source_folder(hit["source"])
        adj = hit["distance"] - boosts.get(folder, 0.0)
        # Prefer canonical policy folders slightly over faq duplicates when tied
        if folder in boosts and hit["source"].startswith(folder):
            adj -= 0.005
        scored.append((adj, i, hit))
    scored.sort(key=lambda t: (t[0], t[1]))

    selected: list[dict] = []
    seen_folders: set[str] = set()
    remaining = [h for _, _, h in scored]

    if remaining:
        first = remaining.pop(0)
        selected.append(first)
        seen_folders.add(_source_folder(first["source"]))

    still: list[dict] = []
    for hit in remaining:
        folder = _source_folder(hit["source"])
        if folder not in seen_folders and len(selected) < top_k:
            selected.append(hit)
            seen_folders.add(folder)
        else:
            still.append(hit)

    for hit in still:
        if len(selected) >= top_k:
            break
        selected.append(hit)

    selected = selected[:top_k]

    # Ensure the closest raw match remains in the returned set (for gating).
    raw_best = min(hits, key=lambda h: h["distance"])
    if raw_best not in selected and selected:
        selected[-1] = raw_best

    return selected


def get_client(persist_dir: Path | None = None) -> chromadb.PersistentClient:
    path = str((persist_dir or config.CHROMA_DIR).resolve())
    if path not in _CLIENT_CACHE:
        Path(path).mkdir(parents=True, exist_ok=True)
        _CLIENT_CACHE[path] = chromadb.PersistentClient(
            path=path, settings=_CHROMA_SETTINGS
        )
    return _CLIENT_CACHE[path]


def clear_vector_caches() -> None:
    """Drop cached Chroma clients/collections (call after rebuild)."""
    _CLIENT_CACHE.clear()
    _COLLECTION_CACHE.clear()


def build_index(
    kb_dir: Path | None = None,
    persist_dir: Path | None = None,
    collection_name: str | None = None,
) -> int:
    """Rebuild the Chroma collection from the knowledge-base folder. Returns chunk count."""
    kb_dir = kb_dir or config.KB_DIR
    collection_name = collection_name or config.COLLECTION_NAME
    client = get_client(persist_dir)

    chunks = load_chunks(kb_dir)
    if collection_name in _collection_names(client):
        client.delete_collection(collection_name)

    collection = client.create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )

    ids, documents, metadatas, embeddings = [], [], [], []
    for chunk in chunks:
        ids.append(f"{chunk['source']}::{chunk['chunk_id']}")
        # Store clean text for the LLM; embed a source-tagged version for search
        documents.append(chunk["text"])
        metadatas.append(
            {"source": chunk["source"], "chunk_id": int(chunk["chunk_id"])}
        )
        embeddings.append(embed_text(embedding_text(chunk["source"], chunk["text"])))

    collection.add(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
        embeddings=embeddings,
    )
    clear_vector_caches()
    return collection.count()


def get_collection(
    persist_dir: Path | None = None,
    collection_name: str | None = None,
):
    """Open an existing collection or raise a clear error (cached per process)."""
    collection_name = collection_name or config.COLLECTION_NAME
    cache_key = f"{(persist_dir or config.CHROMA_DIR).resolve()}::{collection_name}"
    if cache_key in _COLLECTION_CACHE:
        return _COLLECTION_CACHE[cache_key]

    client = get_client(persist_dir)
    if collection_name not in _collection_names(client):
        raise RuntimeError(
            f"Collection {collection_name!r} not found in {persist_dir or config.CHROMA_DIR}. "
            "Run: python build_index.py"
        )
    collection = client.get_collection(collection_name)
    if collection.count() == 0:
        raise RuntimeError(
            f"Collection {collection_name!r} is empty. Run: python build_index.py"
        )
    _COLLECTION_CACHE[cache_key] = collection
    return collection


def retrieve(
    question: str,
    top_k: int | None = None,
    collection=None,
    *,
    diversify: bool = True,
    where: dict | None = None,
) -> list[dict]:
    """
    Return top-k hits with text, metadata, distance, and approx similarity.

    Fetches a wider candidate pool, then optionally re-ranks with light topic
    boosts and folder diversity (helps multi-policy questions). Raw cosine
    distances are preserved on each hit for gating.
    """
    top_k = top_k or config.TOP_K
    if collection is None:
        collection = get_collection()

    # Wider pool for re-ranking; never ask for more rows than the index holds
    if where is None:
        count = collection.count()
    else:
        matched = collection.get(where=where)
        count = len(matched.get("ids") or [])
    if count <= 0:
        return []
    n_candidates = min(max(top_k * 3, 8), count)

    # Embed the raw question (same as historically). Do not add a Harbor document
    # prefix here — that shifted distances and weakened the unsupported gate.
    query = {
        "query_embeddings": [embed_text(question)],
        "n_results": n_candidates,
        "include": ["documents", "metadatas", "distances"],
    }
    if where is not None:
        query["where"] = where
    result = collection.query(**query)

    hits = []
    for doc, meta, dist in zip(
        result["documents"][0],
        result["metadatas"][0],
        result["distances"][0],
    ):
        distance = float(dist)
        hit = {
            "text": doc,
            "source": meta["source"],
            "chunk_id": meta["chunk_id"],
            "distance": distance,
            "similarity": 1.0 - distance,
        }
        if meta.get("page") is not None:
            hit["page"] = int(meta["page"])
        hits.append(hit)

    if diversify and len(hits) > top_k:
        return rerank_hits(question, hits, top_k)
    if diversify:
        return rerank_hits(question, hits, top_k)
    return hits[:top_k]
