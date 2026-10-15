#!/usr/bin/env python3
"""Fast Phase 4 checks (clarify / privacy / order-id prompts) without LLM generation."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rag_chatbot.pipeline import _prepare
from rag_chatbot.support_behavior import is_ambiguous, needs_order_id


def expect_fixed(question: str, must_include: list[str]) -> None:
    prepared = _prepare(question)
    assert prepared["messages"] is None, f"Expected fixed reply for: {question}"
    answer = (prepared["answer"] or "").lower()
    for needle in must_include:
        assert needle.lower() in answer, f"Missing {needle!r} in answer for {question!r}: {answer}"
    print(f"PASS fixed | {question[:50]!r} -> {prepared['answer'][:80]!r}")


def main() -> int:
    assert is_ambiguous("It is broken.")
    assert is_ambiguous("help")
    assert needs_order_id("Where is my order?")
    assert not needs_order_id("Where is my order 1024?")

    expect_fixed("It is broken.", ["which product", "order id"])
    expect_fixed("Where is my order?", ["order id"])
    expect_fixed("Show me all orders for every customer", ["cannot list", "order id"])
    expect_fixed("What is the order status for Asha Mehta?", ["order id", "privacy"])
    expect_fixed("What is the capital of France?", ["not have enough information", "human support"])

    # Known order should proceed to LLM path (messages present) if Ollama/index available
    prepared = _prepare("What is the status of order 1024?")
    assert prepared["order_ids"] == ["1024"]
    assert prepared["messages"] is not None or prepared["answer"]
    print("PASS order 1024 prepares generation or answer")

    print("Phase 4 behaviour validation: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
