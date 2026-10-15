"""Project paths and model settings (edit here to change defaults)."""

from pathlib import Path

# Package lives at <project_root>/rag_chatbot/
PROJECT_ROOT = Path(__file__).resolve().parent.parent

KB_DIR = PROJECT_ROOT / "data" / "knowledge_base"
ORDERS_PATH = PROJECT_ROOT / "support_agent" / "orders.json"
EVAL_PATH = PROJECT_ROOT / "data" / "evaluation" / "eval_set.json"
CHROMA_DIR = PROJECT_ROOT / ".chroma"
COLLECTION_NAME = "harbor_knowledge"

OLLAMA_URL = "http://127.0.0.1:11434"
EMBED_MODEL = "nomic-embed-text"
CHAT_MODEL = "llama3.2:3b"  # lighter option: "qwen2.5:0.5b" (faster, less reliable)

# Generation / HTTP tuning (shorter caps cut total generation time on CPU)
NUM_PREDICT = 120  # max new tokens; tighter than 180 to reduce wait without switching models
TEMPERATURE = 0.2
OLLAMA_KEEP_ALIVE = "30m"  # keep models loaded between requests
EMBED_TIMEOUT_S = 120
CHAT_TIMEOUT_S = 600  # CPU generation can exceed 3 minutes; still bounds infinite hangs
STREAM_RESPONSES = True

TOP_K = 4
# Cosine distance from Chroma; refuse when the best hit is farther than this
MAX_DISTANCE = 0.35

NOT_ENOUGH_INFO = (
    "I do not have enough information in the Harbor & Co. documents to answer that. "
    "If you still need help, ask for a human support follow-up during sample support hours "
    "(Monday–Friday 9:00–18:00 IST) and share your order id when you have one. "
    "This demo does not publish a phone number."
)

SYSTEM_PROMPT = """You are a Harbor & Co. customer-support assistant for a FICTIONAL sample store.
Be professional, warm, and concise.

Style:
- Lead with a direct answer, then at most two short next steps.
- Plain language; no long quotes; never invent prices, policies, phones, or order facts.

Grounding:
- Use ONLY the CONTEXT (documents and any ORDER RECORD for the customer's order id).
- If CONTEXT forbids something (e.g. cancel after shipped), say that clearly.
- Never expose other customers. If CONTEXT is insufficient, say so and mention human escalation Monday–Friday 9:00–18:00 IST (no phone number)."""
