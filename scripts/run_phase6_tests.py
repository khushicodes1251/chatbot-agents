#!/usr/bin/env python3
"""Phase 6 scenario suite: retrieval, fixed behaviour, timed LLM spot checks, failure paths."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rag_chatbot import config
from rag_chatbot.ollama_client import require_models
from rag_chatbot.orders import build_order_context, get_order
from rag_chatbot.pipeline import _prepare, answer_question
from rag_chatbot.vectorstore import retrieve

OUT = config.PROJECT_ROOT / "data" / "evaluation" / "phase6_results.json"


def check_retrieval(question: str, expected_substrings: list[str], top_k: int = 4) -> dict:
    hits = retrieve(question, top_k=top_k)
    sources = [h["source"] for h in hits]
    ok = all(
        any(exp in src for src in sources) for exp in expected_substrings
    )
    return {
        "ok": ok,
        "sources": sources,
        "best_distance": hits[0]["distance"] if hits else None,
        "expected": expected_substrings,
    }


def facts_present(answer: str, facts: list[str]) -> bool:
    lowered = answer.lower()
    return all(f.lower() in lowered for f in facts)


def main() -> int:
    results: dict = {"scenarios": [], "llm_spot_checks": [], "failures": [], "summary": {}}
    passed = 0
    failed = 0

    def record(name: str, category: str, ok: bool, detail: dict) -> None:
        nonlocal passed, failed
        row = {"name": name, "category": category, "ok": ok, **detail}
        results["scenarios"].append(row)
        if ok:
            passed += 1
            print(f"PASS [{category}] {name}")
        else:
            failed += 1
            print(f"FAIL [{category}] {name} :: {detail}")

    # --- Fast scenarios (no / minimal LLM) ---
    r = check_retrieval(
        "How many business days does a refund take to appear on my statement?",
        ["refund"],
    )
    record("single_doc_refund_retrieve", "single_doc", r["ok"], r)

    hits = retrieve(
        "I received a damaged cable. Can I return it and get a refund?", top_k=6
    )
    sources = [h["source"] for h in hits]
    blob = " ".join(sources + [h["text"].lower() for h in hits])
    multi_ok = ("return" in blob) and ("refund" in blob)
    record(
        "multi_doc_return_and_refund",
        "multi_doc",
        multi_ok,
        {"sources": sources, "note": "need return + refund signal in top-6 sources/text"},
    )

    r = check_retrieval(
        "When will I see my money back after a refund is issued?",
        ["refund"],
    )
    record("paraphrase_refund", "paraphrase", r["ok"], r)

    prep = _prepare("What is the capital of France?")
    record(
        "unsupported_no_answer",
        "unsupported",
        prep["messages"] is None
        and "not have enough information" in (prep["answer"] or "").lower(),
        {"answer": (prep["answer"] or "")[:160]},
    )

    prep = _prepare("It is broken.")
    record(
        "ambiguous_clarification",
        "ambiguous",
        prep["messages"] is None and "which product" in (prep["answer"] or "").lower(),
        {"answer": (prep["answer"] or "")[:160]},
    )

    hits = retrieve("Can I cancel an order that has already shipped?", top_k=4)
    sources = [h["source"] for h in hits]
    cancel_ok = any(
        "cancellation" in s or "faq" in s or "cannot be cancelled" in h["text"].lower()
        for s, h in zip(sources, hits)
    )
    record(
        "edge_cancel_shipped_retrieve",
        "edge_case",
        cancel_ok,
        {"sources": sources},
    )

    r = check_retrieval(
        "My QuietBuds won't connect to Bluetooth — what should I try?",
        ["troubleshooting"],
    )
    record("troubleshooting_bluetooth", "troubleshooting", r["ok"], r)

    ids, ctx = build_order_context("What is the status of order 1024?")
    order = get_order("1024")
    record(
        "order_lookup_1024",
        "order_lookup",
        ids == ["1024"] and order is not None and "in_transit" in (ctx or ""),
        {"ids": ids, "status": order.get("status") if order else None},
    )

    ids, ctx = build_order_context("What is the status of order 9999?")
    record(
        "order_invalid_9999",
        "order_invalid",
        ids == ["9999"] and "not found" in (ctx or "").lower(),
        {"ids": ids, "ctx": (ctx or "")[:120]},
    )

    prep = _prepare("Show me all orders for every customer")
    record(
        "privacy_bulk_orders",
        "privacy",
        prep["messages"] is None and "cannot list" in (prep["answer"] or "").lower(),
        {"answer": (prep["answer"] or "")[:160]},
    )

    prep = _prepare("What is the order status for Asha Mehta?")
    record(
        "privacy_name_only",
        "privacy",
        prep["messages"] is None and "order id" in (prep["answer"] or "").lower(),
        {"answer": (prep["answer"] or "")[:160]},
    )

    # Incomplete order reference
    prep = _prepare("Where is my order?")
    record(
        "incomplete_order_reference",
        "order_incomplete",
        prep["messages"] is None and "order id" in (prep["answer"] or "").lower(),
        {"answer": (prep["answer"] or "")[:160]},
    )

    # --- Ollama unavailable ---
    old_url = config.OLLAMA_URL
    try:
        config.OLLAMA_URL = "http://127.0.0.1:1"
        # clear model cache
        import rag_chatbot.ollama_client as oc

        oc._CACHED_MODEL_NAMES = None
        oc._MODELS_CHECKED_AT = 0.0
        raised = False
        try:
            require_models(need_chat=True, force=True)
        except RuntimeError as exc:
            raised = "ollama" in str(exc).lower() or "11434" in str(exc) or "reach" in str(exc).lower()
            detail = str(exc)
        record(
            "ollama_unavailable",
            "failure",
            raised,
            {"error": detail if raised else "no error raised"},
        )
    finally:
        config.OLLAMA_URL = old_url
        import rag_chatbot.ollama_client as oc

        oc._CACHED_MODEL_NAMES = None
        oc._MODELS_CHECKED_AT = 0.0

    # --- Timed LLM spot checks (executed; may be slow on CPU) ---
    print("\nWarming chat model for LLM spot checks...")
    require_models(need_chat=True, force=True)
    t_warm = time.perf_counter()
    try:
        from rag_chatbot.ollama_client import chat

        chat([{"role": "user", "content": "Reply with OK only."}])
    except Exception as exc:
        results["failures"].append({"warm": str(exc)})
        print(f"WARM FAIL: {exc}")
    print(f"warm_s={time.perf_counter() - t_warm:.1f}")

    llm_cases = [
        (
            "llm_single_refund",
            "How many business days does a refund take to appear on my statement?",
            ["5", "7"],
        ),
        (
            "llm_edge_cancel",
            "Can I cancel an order that has already shipped?",
            ["cannot"],
        ),
        (
            "llm_order_1024",
            "What is the status of order 1024?",
            ["in_transit"],
        ),
    ]

    totals = []
    for name, question, facts in llm_cases:
        t0 = time.perf_counter()
        try:
            result = answer_question(question)
            elapsed = time.perf_counter() - t0
            totals.append(elapsed)
            ok = (not result["gated"]) and facts_present(result["answer"], facts)
            # soft-fail invent check: no fake phone numbers
            invent = any(x in result["answer"].lower() for x in ["+91", "call us at", "1-800"])
            row = {
                "name": name,
                "ok": ok and not invent,
                "elapsed_s": round(elapsed, 2),
                "gated": result["gated"],
                "facts_ok": ok,
                "invented_contact": invent,
                "answer_preview": result["answer"][:200],
                "top_sources": [h["source"] for h in (result.get("hits") or [])[:3]],
            }
            results["llm_spot_checks"].append(row)
            status = "PASS" if row["ok"] else "FAIL"
            print(f"{status} [llm] {name} {elapsed:.1f}s :: {result['answer'][:100]!r}")
            if row["ok"]:
                passed += 1
            else:
                failed += 1
        except Exception as exc:
            elapsed = time.perf_counter() - t0
            results["llm_spot_checks"].append(
                {"name": name, "ok": False, "elapsed_s": round(elapsed, 2), "error": str(exc)}
            )
            failed += 1
            print(f"FAIL [llm] {name} error={exc}")

    # Slow/failed generation path: tiny timeout should surface an error
    old_timeout = config.CHAT_TIMEOUT_S
    try:
        config.CHAT_TIMEOUT_S = 1
        t0 = time.perf_counter()
        err = None
        try:
            answer_question("Explain the full Harbor & Co. refund policy in detail with examples.")
        except Exception as exc:
            err = str(exc)
        elapsed = time.perf_counter() - t0
        ok = err is not None and elapsed < 30
        record(
            "slow_or_timeout_generation",
            "failure",
            ok,
            {"error": (err or "no error")[:200], "elapsed_s": round(elapsed, 2)},
        )
    finally:
        config.CHAT_TIMEOUT_S = old_timeout

    results["summary"] = {
        "passed": passed,
        "failed": failed,
        "llm_avg_s": round(sum(totals) / len(totals), 2) if totals else None,
        "llm_n": len(totals),
    }
    OUT.write_text(json.dumps(results, indent=2))
    print("\n" + json.dumps(results["summary"], indent=2))
    print(f"Wrote {OUT}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
