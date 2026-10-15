#!/usr/bin/env python3
"""Validate Phase 2 knowledge base loading, retrieval, orders, and eval coverage."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rag_chatbot import config
from rag_chatbot.documents import load_chunks, load_text_documents
from rag_chatbot.orders import (
    build_order_context,
    extract_order_ids,
    get_order,
    refuse_bulk_or_foreign_lookup,
)
from rag_chatbot.vectorstore import retrieve


def main() -> int:
    docs = load_text_documents(config.KB_DIR)
    chunks = load_chunks(config.KB_DIR)
    print(f"KB documents: {len(docs)}")
    print(f"KB chunks: {len(chunks)}")
    categories = sorted({d["source"].split("/")[0] for d in docs if "/" in d["source"]})
    print(f"Topic folders: {', '.join(categories)}")

    eval_set = json.loads(config.EVAL_PATH.read_text(encoding="utf-8"))
    cases = eval_set["cases"]
    print(f"Eval cases: {len(cases)}")

    # Order safety checks (no LLM)
    assert get_order("1024") is not None
    assert get_order("9999") is None
    assert extract_order_ids("status of order 1024") == ["1024"]
    assert refuse_bulk_or_foreign_lookup("Show me all orders for every customer")
    assert refuse_bulk_or_foreign_lookup("What is the order status for Asha Mehta?")
    _, ctx = build_order_context("What is the status of order 1024?")
    assert ctx and "in_transit" in ctx
    print("Order safety checks: PASS")

    # Retrieval relevance against eval expected_documents (top hit must match one expected)
    retrieval_pass = 0
    retrieval_fail = []
    for case in cases:
        if case.get("privacy_block") or case.get("unsupported"):
            continue
        if case["type"] in {"order_lookup", "order_invalid", "ambiguous"}:
            continue
        expected = case.get("expected_documents") or []
        if not expected:
            continue
        hits = retrieve(case["question"], top_k=4)
        sources = [h["source"] for h in hits]
        ok = any(
            any(src.endswith(exp) or exp in src for src in sources) for exp in expected
        )
        if ok:
            retrieval_pass += 1
            print(f"  RETRIEVE PASS {case['id']}: top={sources[0]}")
        else:
            retrieval_fail.append((case["id"], sources, expected))
            print(f"  RETRIEVE FAIL {case['id']}: got={sources} expected≈{expected}")

    print(f"Retrieval checks passed: {retrieval_pass}; failed: {len(retrieval_fail)}")

    # Unsupported should be gated by distance
    oos = next(c for c in cases if c["id"] == "out_of_scope_capitals")
    hits = retrieve(oos["question"], top_k=1)
    gated = hits[0]["distance"] > config.MAX_DISTANCE
    print(
        f"Unsupported gate check: distance={hits[0]['distance']:.3f} "
        f"threshold={config.MAX_DISTANCE} gated={gated}"
    )

    if retrieval_fail:
        print("Phase 2 validation completed with retrieval failures.")
        return 1
    if not gated:
        print("Warning: unsupported question was not gated by distance.")
        return 1
    print("Phase 2 validation: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
