#!/usr/bin/env python3
"""
Regression checks for policy edge cases, multi-doc retrieval, unsupported, and privacy.

Fast path avoids long LLM waits where possible; optional --with-llm for generation checks.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rag_chatbot import config
from rag_chatbot.pipeline import _prepare, answer_question
from rag_chatbot.vectorstore import retrieve


def _pass(name: str, ok: bool, detail: str = "") -> bool:
    status = "PASS" if ok else "FAIL"
    print(f"{status} {name}" + (f" :: {detail}" if detail else ""))
    return ok


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--with-llm",
        action="store_true",
        help="Also run short LLM answer checks (slower on CPU)",
    )
    args = parser.parse_args()
    failures = 0

    # --- Retrieval: multi-policy must surface returns + refunds ---
    q_multi = "I received a damaged cable. Can I return it and get a refund?"
    hits = retrieve(q_multi, top_k=4)
    sources = [h["source"] for h in hits]
    multi_ok = any("returns/" in s for s in sources) and any("refunds/" in s for s in sources)
    if not _pass("multi_doc_retrieve_returns_and_refunds", multi_ok, f"sources={sources}"):
        failures += 1

    # --- Retrieval: cancellation after ship ---
    q_cancel = "Can I cancel an order that has already shipped?"
    hits = retrieve(q_cancel, top_k=4)
    sources = [h["source"] for h in hits]
    blob = " ".join(h["text"].lower() for h in hits)
    cancel_retrieve = ("cancellations/" in " ".join(sources) or "faq/" in " ".join(sources)) and (
        "cannot be cancelled" in blob or "cannot be cancel" in blob
    )
    if not _pass("cancel_shipped_retrieve", cancel_retrieve, f"sources={sources}"):
        failures += 1

    # --- Retrieval: support hours ---
    q_hours = "What are your support hours?"
    hits = retrieve(q_hours, top_k=3)
    hours_ok = any("support/hours" in h["source"] for h in hits) and any(
        "18:00" in h["text"] or "9:00" in h["text"] for h in hits
    )
    if not _pass("support_hours_retrieve", hours_ok, f"top={hits[0]['source'] if hits else None}"):
        failures += 1

    # --- Refund retrieve ---
    q_refund = "How many business days does a refund take to appear on my statement?"
    hits = retrieve(q_refund, top_k=3)
    refund_ok = any("5 to 7" in h["text"] for h in hits)
    if not _pass("refund_timing_retrieve", refund_ok, f"top={hits[0]['source'] if hits else None}"):
        failures += 1

    # --- Unsupported distance gate (unchanged threshold) ---
    hits = retrieve("What is the capital of France?", top_k=1)
    gated = bool(hits) and hits[0]["distance"] > config.MAX_DISTANCE
    prep = _prepare("What is the capital of France?")
    fixed_gate = prep["messages"] is None and "not have enough information" in (prep["answer"] or "").lower()
    if not _pass(
        "unsupported_distance_gate",
        gated and fixed_gate,
        f"distance={hits[0]['distance']:.3f} threshold={config.MAX_DISTANCE}",
    ):
        failures += 1

    # --- Privacy fixed reply ---
    prep = _prepare("Show me all orders for every customer")
    privacy_ok = prep["messages"] is None and (
        "cannot list" in (prep["answer"] or "").lower()
        or "order id" in (prep["answer"] or "").lower()
    )
    if not _pass("privacy_bulk_lookup", privacy_ok, (prep["answer"] or "")[:80]):
        failures += 1

    prep = _prepare("What is the order status for Asha Mehta?")
    name_ok = prep["messages"] is None and "privacy" in (prep["answer"] or "").lower()
    if not _pass("privacy_name_lookup", name_ok, (prep["answer"] or "")[:80]):
        failures += 1

    # --- Multi-doc prepare includes both policy folders in prompt context ---
    prep = _prepare(q_multi)
    ctx_sources = [h["source"] for h in (prep.get("hits") or [])[:4]]
    prep_multi = any("returns/" in s for s in ctx_sources) and any(
        "refunds/" in s for s in ctx_sources
    )
    if not _pass("multi_doc_prepare_hits", prep_multi, f"hits={ctx_sources}"):
        failures += 1

    assert config.CHAT_MODEL.startswith("llama3.2"), "Default chat model must stay llama3.2:3b"
    if not _pass("default_chat_model_llama", True, config.CHAT_MODEL):
        failures += 1

    if args.with_llm:
        print("Running optional LLM checks (may take several minutes)…")
        for case_id, question, patterns in [
            (
                "cancel_shipped",
                q_cancel,
                [r"cannot\s+be\s+cancell|can'?t\s+cancell|won'?t\s+be\s+able\s+to\s+cancell|not\s+(be\s+)?cancell"],
            ),
            (
                "refund_timing",
                q_refund,
                [r"5\s*(to|-)?\s*7"],
            ),
            (
                "support_hours",
                q_hours,
                [r"9", r"18|6\s*(:00)?\s*pm|6\s*pm", r"ist"],
            ),
        ]:
            result = answer_question(question)
            text = (result.get("answer") or "").lower()
            ok = all(re.search(p, text, re.I) for p in patterns)
            if not _pass(f"llm_{case_id}", ok, text[:120]):
                failures += 1

    if failures:
        print(f"Regression validation: FAIL ({failures} failed)")
        return 1
    print("Regression validation: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
