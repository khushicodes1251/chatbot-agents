"""Customer-support behaviour helpers (clarify, escalate, classify intent)."""

from __future__ import annotations

import re

from rag_chatbot.orders import extract_order_ids

# Very short / vague customer messages that need clarification before answering
_AMBIGUOUS_PATTERNS = [
    re.compile(r"^(it|this|that)\s+(is\s+)?(broken|damaged|not working)\.?$", re.I),
    re.compile(r"^(help|hello|hi|hey)\.?$", re.I),
    re.compile(r"^(issue|problem|error)\.?$", re.I),
    re.compile(r"^my (order|package|item|product)\.?$", re.I),
    re.compile(r"^(what can you do|who are you)\??$", re.I),
]

_ORDER_STATUS_PATTERNS = [
    re.compile(r"\b(where is my (order|package)|track( my)? order|order status|status of my order)\b", re.I),
    re.compile(r"\b(has my order shipped|when will my order (arrive|be delivered))\b", re.I),
]

_ESCALATION_HINT = (
    "If you still need help, ask for a human support follow-up during Harbor & Co. "
    "sample support hours (Monday–Friday 9:00–18:00 IST) and share your order id when you have one. "
    "This demo does not publish a phone number."
)


def is_ambiguous(question: str) -> bool:
    text = " ".join(question.strip().split())
    if len(text) < 12 and "?" not in text and not extract_order_ids(text):
        # Extremely short non-questions without an order id
        if text.lower() in {"broken", "damaged", "refund", "return", "cancel", "help"}:
            return True
    return any(p.match(text) for p in _AMBIGUOUS_PATTERNS)


def clarifying_question(question: str) -> str:
    lowered = question.lower()
    if "broken" in lowered or "damaged" in lowered or "not working" in lowered:
        return (
            "I can help with that. Which product is affected "
            "(QuietBuds, SoundBar Mini, Link USB-C, or Wave), and do you have an order id?"
        )
    if "order" in lowered or "package" in lowered:
        return (
            "I can check a sample order when you share your order id "
            "(for example: 1024). What is your order id?"
        )
    if "refund" in lowered or "return" in lowered or "cancel" in lowered:
        return (
            "Happy to help. Are you asking about our general policy, "
            "or about a specific order? If it is a specific order, please share the order id."
        )
    return (
        "Could you share a bit more detail—what product or topic you need help with "
        "(shipping, return, refund, warranty, troubleshooting, or an order id)?"
    )


def needs_order_id(question: str) -> bool:
    """True when the customer asks for personal order status but gave no order id."""
    if extract_order_ids(question):
        return False
    return any(p.search(question) for p in _ORDER_STATUS_PATTERNS)


def order_id_prompt() -> str:
    return (
        "To check a specific sample order, please share your order id "
        "(for example: 1024). For privacy I cannot look up orders by name alone."
    )


def insufficient_with_next_steps() -> str:
    return (
        "I do not have enough information in the Harbor & Co. documents to answer that. "
        + _ESCALATION_HINT
    )


def is_order_specific(question: str, order_ids: list[str]) -> bool:
    return bool(order_ids)


def build_user_prompt(question: str, context: str, order_specific: bool) -> str:
    scope = (
        "Order-specific: use ORDER RECORD for that order only; use policy for what is allowed."
        if order_specific
        else "General question: answer from documents only; do not invent an order status."
    )
    return (
        f"{scope}\n\n"
        f"CONTEXT:\n{context}\n\n"
        f"QUESTION:\n{question}\n\n"
        "Answer directly from CONTEXT only. At most 2 short next steps. "
        "No invented policies or phone numbers. If missing info, say so and mention "
        "escalation Monday–Friday 9:00–18:00 IST."
    )


def truncate_hit_text(text: str, max_chars: int = 500) -> str:
    text = text.strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."
