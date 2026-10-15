"""Safe lookup helpers for fictional sample orders.

Only returns an order when the caller supplies that exact order id.
Never lists the full order database or searches by customer name alone.
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from typing import Any

from rag_chatbot import config

_EXPLICIT_ORDER_PATTERN = re.compile(
    r"order\s*(?:id|#|number)?\s*[=:]?\s*(\d{4})\b",
    re.IGNORECASE,
)
_BARE_FOUR_DIGIT = re.compile(r"\b(\d{4})\b")
_PERSON_NAME = re.compile(r"\b[A-Z][a-z]+ [A-Z][a-z]+\b")


@lru_cache(maxsize=1)
def _load_orders() -> dict[str, dict[str, Any]]:
    path = config.ORDERS_PATH
    if not path.is_file():
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {str(item["order_id"]): item for item in raw}


def extract_order_ids(text: str) -> list[str]:
    """Return unique order ids mentioned in text.

    Accepts explicit phrases like ``order 1024`` / ``order id: 1024``,
    or a bare 4-digit token only when it exists in the sample database.
    """
    seen: set[str] = set()
    ids: list[str] = []

    def add(oid: str) -> None:
        if oid not in seen:
            seen.add(oid)
            ids.append(oid)

    for match in _EXPLICIT_ORDER_PATTERN.finditer(text):
        add(match.group(1))

    known = _load_orders()
    for match in _BARE_FOUR_DIGIT.finditer(text):
        oid = match.group(1)
        if oid in known:
            add(oid)

    return ids


def get_order(order_id: str) -> dict[str, Any] | None:
    """Return one sample order by id, or None if it does not exist."""
    return _load_orders().get(str(order_id))


def format_order_for_context(order: dict[str, Any]) -> str:
    """Format a single order for the LLM context (no other customers included)."""
    return (
        f"order_id: {order.get('order_id')}\n"
        f"customer_name: {order.get('customer_name')}\n"
        f"product: {order.get('product')}\n"
        f"category: {order.get('category')}\n"
        f"status: {order.get('status')}\n"
        f"order_date: {order.get('order_date')}\n"
        f"promised_delivery_date: {order.get('promised_delivery_date')}\n"
        f"delivered_date: {order.get('delivered_date')}\n"
        f"price: {order.get('price')}\n"
        f"payment_method: {order.get('payment_method')}\n"
        f"damaged: {order.get('damaged')}\n"
        f"notes: {order.get('notes')}"
    )


def build_order_context(question: str) -> tuple[list[str], str | None]:
    """Inspect the question for order ids and build a context block."""
    order_ids = extract_order_ids(question)
    if not order_ids:
        return [], None

    blocks: list[str] = []
    for oid in order_ids:
        order = get_order(oid)
        if order is None:
            blocks.append(
                f"ORDER LOOKUP: order id {oid} was not found in the sample order database."
            )
        else:
            blocks.append(
                "ORDER RECORD (only the order id supplied by the customer):\n"
                + format_order_for_context(order)
            )
    return order_ids, "\n\n".join(blocks)


def refuse_bulk_or_foreign_lookup(question: str) -> str | None:
    """Return a refusal message for unsafe order requests, else None."""
    lowered = question.lower()
    bulk_markers = [
        "all orders",
        "every order",
        "list orders",
        "show all orders",
        "everyone's order",
        "other customer",
        "another customer",
        "someone else's",
    ]
    if any(marker in lowered for marker in bulk_markers):
        return (
            "I can only look up an order when you provide your own order id. "
            "I cannot list all orders or share another customer's information."
        )

    if extract_order_ids(question):
        return None

    # Name-only order lookup without an order id
    if "order" in lowered and _PERSON_NAME.search(question):
        return (
            "For privacy, I look up sample orders only by order id, "
            "not by customer name. Please share your order id."
        )
    return None
