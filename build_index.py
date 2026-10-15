#!/usr/bin/env python3
"""Build (or rebuild) the local Chroma index from data/knowledge_base/."""

import os

os.environ["ANONYMIZED_TELEMETRY"] = "False"

from rag_chatbot import config
from rag_chatbot.pipeline import build_index


def main() -> None:
    print(f"Knowledge base: {config.KB_DIR}")
    print(f"Chroma path:    {config.CHROMA_DIR}")
    print(f"Embed model:    {config.EMBED_MODEL}")
    count = build_index()
    print(f"Indexed {count} chunks into collection {config.COLLECTION_NAME!r}.")
    print("Next: python chat_cli.py")


if __name__ == "__main__":
    main()
