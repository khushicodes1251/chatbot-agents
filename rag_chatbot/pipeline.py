"""End-to-end RAG: retrieve context, gate weak matches, generate with Ollama."""

from __future__ import annotations

from typing import Callable, Iterator

from rag_chatbot import config
from rag_chatbot.ollama_client import chat, chat_stream, require_models
from rag_chatbot.orders import build_order_context, get_order, refuse_bulk_or_foreign_lookup
from rag_chatbot.support_behavior import (
    build_user_prompt,
    clarifying_question,
    insufficient_with_next_steps,
    is_ambiguous,
    is_order_specific,
    needs_order_id,
    order_id_prompt,
    truncate_hit_text,
)
from rag_chatbot.vectorstore import build_index as rebuild_vector_index
from rag_chatbot.vectorstore import retrieve

# Prefer a smaller, tighter context window for support answers (smaller prompt → faster TTFT)
_MAX_HITS_IN_PROMPT = 3
_MAX_CHARS_PER_HIT = 360


def build_index() -> int:
    """Index the knowledge base (embeddings + Chroma). Chat model not required."""
    require_models(need_chat=False, force=True)
    return rebuild_vector_index()


def format_context(hits: list[dict]) -> str:
    blocks = []
    for i, hit in enumerate(hits[:_MAX_HITS_IN_PROMPT], start=1):
        text = truncate_hit_text(hit["text"], _MAX_CHARS_PER_HIT)
        blocks.append(f"[{i}] ({hit['source']}#{hit['chunk_id']})\n{text}")
    return "\n\n".join(blocks)


def _fixed_reply(
    question: str,
    answer: str,
    *,
    hits: list | None = None,
    order_ids: list | None = None,
    gated: bool = True,
) -> dict:
    return {
        "question": question,
        "hits": hits or [],
        "gated": gated,
        "order_ids": order_ids or [],
        "messages": None,
        "answer": answer,
    }


def _prepare(question: str) -> dict:
    """Retrieve + assemble context, or return a fixed support reply (no LLM)."""
    require_models(need_chat=True)

    privacy_refusal = refuse_bulk_or_foreign_lookup(question)
    if privacy_refusal:
        return _fixed_reply(question, privacy_refusal)

    if is_ambiguous(question):
        return _fixed_reply(question, clarifying_question(question))

    if needs_order_id(question):
        return _fixed_reply(question, order_id_prompt())

    order_ids, order_context = build_order_context(question)
    matched_orders = [oid for oid in order_ids if get_order(oid) is not None]
    hits = retrieve(question)
    # Gate on closest raw distance in the set (not display order after re-rank)
    best_distance = min((h["distance"] for h in hits), default=999.0)
    retrieval_ok = bool(hits) and best_distance <= config.MAX_DISTANCE

    if not retrieval_ok and not order_context:
        return _fixed_reply(
            question,
            insufficient_with_next_steps(),
            hits=hits,
            order_ids=order_ids,
        )

    # Order id mentioned but not found in sample DB
    if order_ids and not matched_orders and not retrieval_ok:
        return _fixed_reply(
            question,
            (
                f"I could not find order id {', '.join(order_ids)} in the sample order database. "
                "Please double-check the id, or ask a general policy question. "
                + insufficient_with_next_steps().split(". ", 1)[-1]
            ),
            hits=hits,
            order_ids=order_ids,
        )

    context_parts: list[str] = []
    if order_context:
        context_parts.append(order_context)
    if retrieval_ok:
        context_parts.append(format_context(hits))
    elif matched_orders and hits:
        context_parts.append(format_context(hits[:2]))

    messages = [
        {"role": "system", "content": config.SYSTEM_PROMPT},
        {
            "role": "user",
            "content": build_user_prompt(
                question,
                "\n\n".join(context_parts),
                order_specific=is_order_specific(question, order_ids),
            ),
        },
    ]
    return {
        "question": question,
        "hits": hits,
        "gated": False,
        "order_ids": order_ids,
        "messages": messages,
        "answer": None,
    }


def _append_sources(result: dict, show_sources: bool) -> dict:
    if not show_sources:
        return result
    hits = result.get("hits") or []
    sources = ", ".join(
        f"{h['source']}#{h['chunk_id']} ({h['similarity']:.2f})" for h in hits
    )
    order_ids = result.get("order_ids") or []
    extra = f" | orders: {', '.join(order_ids)}" if order_ids else ""
    result["answer"] = f"{result.get('answer') or ''}\n\nSources: {sources}{extra}"
    return result


def answer_question(question: str, show_sources: bool = False) -> dict:
    """
    Answer one question using RAG (blocking; waits for full generation).

    Returns dict with keys: question, answer, gated, hits, order_ids.
    """
    prepared = _prepare(question)
    if prepared["gated"] or prepared["messages"] is None:
        return _append_sources(
            {
                "question": prepared["question"],
                "hits": prepared["hits"],
                "gated": prepared["gated"],
                "order_ids": prepared["order_ids"],
                "answer": prepared["answer"],
            },
            show_sources,
        )

    answer = chat(prepared["messages"])
    return _append_sources(
        {
            "question": prepared["question"],
            "hits": prepared["hits"],
            "gated": False,
            "order_ids": prepared["order_ids"],
            "answer": answer,
        },
        show_sources,
    )


def answer_question_stream(
    question: str,
    show_sources: bool = False,
    on_token: Callable[[str], None] | None = None,
) -> dict:
    """
    Answer with token streaming when generation is needed.

    Fixed replies (clarify / privacy / insufficient) are returned immediately
    without calling the LLM.
    """
    prepared = _prepare(question)
    if prepared["gated"] or prepared["messages"] is None:
        answer = prepared["answer"] or ""
        if on_token and answer:
            on_token(answer)
        return _append_sources(
            {
                "question": prepared["question"],
                "hits": prepared["hits"],
                "gated": prepared["gated"],
                "order_ids": prepared["order_ids"],
                "answer": answer,
            },
            show_sources,
        )

    pieces: list[str] = []
    for piece in chat_stream(prepared["messages"]):
        pieces.append(piece)
        if on_token:
            on_token(piece)
    answer = "".join(pieces).strip()
    return _append_sources(
        {
            "question": prepared["question"],
            "hits": prepared["hits"],
            "gated": False,
            "order_ids": prepared["order_ids"],
            "answer": answer,
        },
        show_sources,
    )


def iter_answer_tokens(question: str) -> Iterator[str]:
    """Low-level token iterator for generation path (or a single fixed reply)."""
    prepared = _prepare(question)
    if prepared["gated"] or prepared["messages"] is None:
        yield prepared["answer"] or ""
        return
    yield from chat_stream(prepared["messages"])
