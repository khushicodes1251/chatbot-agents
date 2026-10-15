#!/usr/bin/env python3
"""Simple terminal chatbot for the local Harbor & Co. RAG demo."""

import argparse
import os
import sys
import time

# Must run before chromadb is imported (via rag_chatbot)
os.environ["ANONYMIZED_TELEMETRY"] = "False"

from rag_chatbot import config
from rag_chatbot.pipeline import answer_question, answer_question_stream
from rag_chatbot.vectorstore import get_collection


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Chat with a local RAG bot over Harbor & Co. sample documents."
    )
    parser.add_argument(
        "--sources",
        action="store_true",
        help="Append retrieved source chunk ids to each answer",
    )
    parser.add_argument(
        "--question",
        "-q",
        help="Ask one question and exit (useful for quick tests)",
    )
    parser.add_argument(
        "--no-stream",
        action="store_true",
        help="Wait for the full answer instead of streaming tokens",
    )
    return parser.parse_args()


def print_banner() -> None:
    print("Harbor & Co. RAG chatbot (local)")
    print(f"Chat model: {config.CHAT_MODEL}  |  Embeddings: {config.EMBED_MODEL}")
    print(
        f"Streaming: {'on' if config.STREAM_RESPONSES else 'off'}  |  "
        f"max tokens: {config.NUM_PREDICT}"
    )
    print("Ask about products, shipping, returns, refunds, orders, and troubleshooting.")
    print("For a specific sample order, include your order id (e.g. 1024).")
    print("Type 'quit' or 'exit' to leave.\n")


def handle_question(question: str, show_sources: bool, stream: bool) -> None:
    print("\nBot: ", end="", flush=True)
    t0 = time.perf_counter()
    first_token_at = None

    def on_token(piece: str) -> None:
        nonlocal first_token_at
        if first_token_at is None:
            first_token_at = time.perf_counter()
        print(piece, end="", flush=True)

    try:
        if stream and config.STREAM_RESPONSES:
            # Stream answer body only; sources are appended after generation.
            result = answer_question_stream(
                question, show_sources=False, on_token=on_token
            )
            if show_sources:
                hits = result.get("hits") or []
                sources = ", ".join(
                    f"{h['source']}#{h['chunk_id']} ({h['similarity']:.2f})" for h in hits
                )
                order_ids = result.get("order_ids") or []
                extra = f" | orders: {', '.join(order_ids)}" if order_ids else ""
                suffix = f"\n\nSources: {sources}{extra}"
                print(suffix, end="", flush=True)
                result["answer"] = f"{result.get('answer') or ''}{suffix}"
        else:
            result = answer_question(question, show_sources=show_sources)
            first_token_at = time.perf_counter()
            print(result["answer"], end="", flush=True)
    except Exception as exc:
        print(f"\n\nError while generating (timeout or model issue): {exc}", file=sys.stderr)
        return

    total = time.perf_counter() - t0
    ttft = (first_token_at - t0) if first_token_at else total
    print()  # end Bot line
    print(f"(ttft={ttft:.1f}s total={total:.1f}s)\n")


def main() -> None:
    args = parse_args()
    stream = not args.no_stream

    # Fail early if the index was never built; reuse this collection handle via cache
    try:
        collection = get_collection()
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"Loaded collection with {collection.count()} chunks.")

    if args.question:
        handle_question(args.question, show_sources=args.sources, stream=stream)
        return

    print_banner()
    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break

        if not question:
            continue
        if question.lower() in {"quit", "exit", "q"}:
            print("Bye.")
            break

        handle_question(question, show_sources=args.sources, stream=stream)


if __name__ == "__main__":
    main()
