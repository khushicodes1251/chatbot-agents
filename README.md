# Build Your First RAG AI Chatbot

A beginner-friendly, **fully local** customer-support chatbot for a **fictional** company called Harbor & Co. The bot answers from a sample knowledge base (policies, products, troubleshooting, and more), retrieves relevant text before generating a reply, and says when the documents do not contain enough information.

No OpenAI API key, paid inference, or cloud vector database is required.

---

## 1. Project overview and purpose

This repository teaches and implements a small **RAG** (Retrieval-Augmented Generation) support assistant:

1. Load and chunk Harbor & Co. sample documents  
2. Embed chunks with a local model (Ollama)  
3. Store vectors in a local Chroma database  
4. Retrieve the best chunks for a customer question  
5. Generate an answer with a local chat model (Ollama), grounded in that context  

You can learn the ideas in **Jupyter notebooks**, then run the same pipeline as a **terminal chatbot** (`chat_cli.py`).

Harbor & Co. content is **fictional and educational**. See `data/knowledge_base/00_about_this_dataset.txt`.

---

## 2. What RAG is and why this project uses it

A plain LLM may guess. **RAG** grounds answers in *your* documents:

```
Question → embed → search vector DB → top document chunks
        → prompt (context + question) → local LLM → answer
```

Why it fits customer support:

- Policies and product facts stay in editable text files  
- The model is instructed to use retrieved context, not invent prices or eligibility  
- When retrieval is weak, the app can refuse and suggest escalation  

---

## 3. Key features and limitations

### Features

- Local embeddings (`nomic-embed-text`) and chat (`llama3.2:3b` by default) via Ollama  
- Local Chroma vector store under `.chroma/`  
- Topic-organized knowledge base (products, pricing, shipping, refunds, troubleshooting, etc.)  
- Safe **order lookup** only when the user supplies a sample order id (from `support_agent/orders.json`)  
- Streaming replies in the CLI (optional `--no-stream`)  
- Clarifying questions for ambiguous requests (no LLM wait)  
- Privacy blocks for “all orders” / name-only lookups  
- Evaluation set + validation scripts  

### Limitations

- Runs on **CPU** in typical setups; `llama3.2:3b` can take **minutes** per full answer  
- Streaming improves **time-to-first-token visibility**, not always total generation time  
- Sample store only: no real tickets, email, or phone channel  
- Small models can still misread policy wording; always verify important answers with `--sources`  
- Not a production support platform (no auth, no multi-user sessions, no LangGraph/agents)  

---

## 4. Technology stack

| Component | Choice | Why |
|-----------|--------|-----|
| Language | Python 3.10+ (tested on 3.12) | Simple for beginners |
| Local LLM / embeddings | [Ollama](https://ollama.com) | No API keys; models stay on disk |
| Chat model (default) | `llama3.2:3b` | Better policy adherence than tiny models |
| Embedding model | `nomic-embed-text` | Small local embeddings for RAG |
| Vector DB | Chroma (persistent) | Simple local store; no cloud |
| HTTP | `requests` | Talk to Ollama’s HTTP API |
| Learning | Jupyter notebooks | Experiment before packaging |
| Orchestration | Plain Python (+ light helpers) | LangChain is **not** required for the CLI path |

---

## 5. Architecture and request flow

```
You (chat_cli.py)
    │
    ▼
support checks (privacy / clarify / need order id?) ──► fixed reply (no LLM)
    │
    ▼
retrieve: embed question → Chroma top-k chunks
    │
    ▼
optional ORDER RECORD (only if order id present and found)
    │
    ▼
distance gate (too weak + no order context?) ──► insufficient + escalation text
    │
    ▼
prompt (system + context + question) → Ollama chat (stream or blocking)
    │
    ▼
answer printed in the terminal
```

**Indexing is separate** (do this after changing documents):

```
data/knowledge_base/**/*.txt → paragraph chunks → embeddings → .chroma/
```

Run: `python build_index.py`  
Chat does **not** rebuild embeddings on every message.

---

## 6. Project directory structure

```
chatbot-agents/
├── README.md                 # this file
├── requirements.txt          # Python dependencies
├── build_index.py            # rebuild Chroma index from the knowledge base
├── chat_cli.py               # interactive / one-shot terminal chatbot
├── rag_chatbot/              # application package
│   ├── config.py             # paths, models, TOP_K, NUM_PREDICT, prompts
│   ├── documents.py          # load .txt files (recursive) + paragraph chunking
│   ├── ollama_client.py      # embed + chat/stream helpers
│   ├── vectorstore.py        # Chroma build / open / retrieve
│   ├── orders.py             # safe order-id lookup
│   ├── support_behavior.py   # clarify, escalate, prompt framing
│   └── pipeline.py           # end-to-end answer_question / stream
├── data/
│   ├── knowledge_base/       # fictional Harbor & Co. documents (by topic)
│   └── evaluation/           # eval_set.json + Phase 3 timing JSON
├── scripts/
│   ├── validate_phase2.py    # KB load + retrieval relevance checks
│   └── validate_phase4.py    # clarify / privacy / escalation behaviour
├── notebooks/                # Stage 1–4 learning notebooks
├── support_agent/            # original sample policies + orders.json
└── .chroma/                  # local vector index (gitignored; created by build_index)
```

---

## 7. Prerequisites and system requirements

- **OS:** Linux/macOS/Windows with Python and terminal access  
- **Python:** 3.10 or newer (verified on 3.12)  
- **RAM:** 8 GB+ recommended if using `llama3.2:3b` on CPU; more is better  
- **Disk:** room for Ollama models (`nomic-embed-text` ~0.3 GB, `llama3.2:3b` ~2 GB)  
- **Ollama** installed and able to run `ollama serve`  
- Optional: Jupyter for notebooks  

---

## 8. Python environment setup

From the project root:

```bash
cd /path/to/chatbot-agents
python3 -m venv .venv

# Linux / macOS
source .venv/bin/activate

# Windows (PowerShell)
# .venv\Scripts\Activate.ps1

pip install --upgrade pip
pip install -r requirements.txt
```

---

## 9. Ollama installation and model configuration

1. Install Ollama from [https://ollama.com](https://ollama.com).  
2. Start the server (separate terminal if needed):

```bash
ollama serve
```

3. Pull the models used by this project:

```bash
ollama pull nomic-embed-text
ollama pull llama3.2:3b
```

Optional faster/weaker chat model (already used in experiments; change config to enable):

```bash
ollama pull qwen2.5:0.5b
```

Then set `CHAT_MODEL = "qwen2.5:0.5b"` in `rag_chatbot/config.py`.

Defaults today:

| Setting | Default |
|---------|---------|
| `EMBED_MODEL` | `nomic-embed-text` |
| `CHAT_MODEL` | `llama3.2:3b` |
| `OLLAMA_URL` | `http://127.0.0.1:11434` |

---

## 10. Dependency installation

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

Main packages: `notebook`, `ipykernel`, `nbconvert`, `numpy`, `requests`, `chromadb`.

---

## 11. How to prepare or update the knowledge base

- Edit or add `.txt` files under `data/knowledge_base/` (subfolders are supported).  
- Keep Harbor & Co. clearly labelled as **fictional**.  
- Prefer clear paragraphs separated by blank lines (chunking splits on `\n\n`).  
- After any change, **rebuild the index**:

```bash
python build_index.py
```

Do not edit only `.chroma/` by hand; always regenerate from the text files.

Sample orders for id lookup live in `support_agent/orders.json` (also fictional).

---

## 12. Chunking, embeddings, indexing, and retrieval

1. **Load** — `documents.py` finds all `*.txt` under `data/knowledge_base/` recursively.  
2. **Chunk** — split on blank lines (paragraph chunks).  
3. **Embed** — Ollama `nomic-embed-text`; index text is tagged with the filename for better “where is the X policy?” matches.  
4. **Store** — Chroma collection `harbor_knowledge` in `.chroma/` (cosine space).  
5. **Retrieve** — embed the question, return top-`TOP_K` chunks with distances.  
6. **Gate** — if best distance `>` `MAX_DISTANCE` and there is no usable order context, refuse with escalation text.  

Learning notebooks walk through these steps with small examples before the packaged CLI.

---

## 13. How to build or rebuild the vector index

```bash
source .venv/bin/activate
ollama serve   # if not already running
python build_index.py
```

You should see a chunk count (currently on the order of **~80+** chunks with the expanded KB). Re-run after every knowledge-base edit.

---

## 14. How to start and use the chatbot

```bash
source .venv/bin/activate
ollama serve          # if needed
python build_index.py # once (or after doc changes)
python chat_cli.py
```

Useful flags:

```bash
python chat_cli.py -q "How long do refunds take?"
python chat_cli.py --sources -q "Can I cancel a shipped order?"
python chat_cli.py --no-stream -q "What is the price of Harbor QuietBuds?"
```

- Type normally; answers **stream** by default.  
- Type `quit` or `exit` to leave.  
- For a **specific sample order**, include the id (e.g. `What is the status of order 1024?`).  
- The CLI prints `(ttft=…s total=…s)` after each reply for local timing.  

---

## 15. Example customer questions and expected behaviour

| Question | Expected behaviour |
|----------|--------------------|
| How many business days does a refund take to appear on my statement? | Mentions **5 to 7** business days |
| How long is express delivery after an order leaves the warehouse? | Mentions **2** business days |
| What is the price of Harbor QuietBuds? | **79.99** (sample price) |
| My QuietBuds won't connect to Bluetooth — what should I try? | Troubleshooting steps from the troubleshooting doc |
| Can I cancel an order that has already shipped? | **Cannot** cancel; may return after delivery if eligible |
| What is the status of order 1024? | Sample order **in_transit** (and related fields) |
| What is the status of order 9999? | Order id **not found** in the sample DB |
| Where is my order? | Asks for an **order id** (no LLM needed) |
| It is broken. | Asks **which product** / order id |
| Show me all orders for every customer | **Privacy refusal** |
| What is the order status for Asha Mehta? | **Privacy refusal** (id required, not name) |
| What is the capital of France? | Not enough info + escalation hint |

---

## 16. Configuration options

Edit `rag_chatbot/config.py`:

| Variable | Role |
|----------|------|
| `CHAT_MODEL` / `EMBED_MODEL` | Ollama model names |
| `TOP_K` | How many chunks to retrieve (default `4`) |
| `MAX_DISTANCE` | Cosine distance gate (default `0.35`; larger = more permissive) |
| `NUM_PREDICT` | Max new tokens (default `120`) |
| `TEMPERATURE` | Sampling temperature (default `0.2`) |
| `STREAM_RESPONSES` | Default streaming on/off for CLI logic |
| `OLLAMA_KEEP_ALIVE` | Keep models loaded (default `"30m"`) |
| `CHAT_TIMEOUT_S` / `EMBED_TIMEOUT_S` | HTTP timeouts |
| `SYSTEM_PROMPT` / `NOT_ENOUGH_INFO` | Support wording |
| `KB_DIR` / `ORDERS_PATH` / `CHROMA_DIR` | Paths |

---

## 17. How to run tests and evaluate retrieval quality

```bash
# Knowledge base load + retrieval vs eval expected documents
python scripts/validate_phase2.py

# Clarify / privacy / escalation fixed replies (no long LLM wait)
python scripts/validate_phase4.py

# Full Phase 6 scenario suite (includes timed LLM spot checks; can take many minutes on CPU)
python scripts/run_phase6_tests.py
# Results: data/evaluation/phase6_results.json

# Compare installed chat models without changing the app default (cold + warm timings)
python scripts/compare_chat_models.py --models llama3.2:3b qwen2.5:0.5b --warm-runs 2
# Results: data/evaluation/model_comparison.json

# Latency baseline (representative subset; label before|after)
python scripts/latency_baseline.py --label before --warm-runs 2

# Policy / multi-doc / privacy regressions (fast; add --with-llm for generation checks)
python scripts/validate_regressions.py
```

Evaluation cases live in `data/evaluation/eval_set.json` (questions, expected document paths, expected facts, unsupported/privacy flags).

Phase 3 timing comparisons (historical measurements on this machine):

- `data/evaluation/phase3_baseline.json`  
- `data/evaluation/phase3_after.json`  
- `data/evaluation/model_comparison.json` (llama3.2:3b vs qwen2.5:0.5b; default not switched)  

Manual chat checks: use the table in section 15 with `python chat_cli.py --sources`.

---

## 18. Performance troubleshooting and common errors

| Symptom | Likely cause | What to try |
|---------|--------------|-------------|
| Very slow answers (minutes) | CPU inference with `llama3.2:3b` | Expect slow totals; streaming shows tokens earlier; optional smaller `CHAT_MODEL` / lower `NUM_PREDICT` |
| “Cannot reach Ollama” | Server not running | `ollama serve`; check `OLLAMA_URL` |
| Missing model errors | Model not pulled | `ollama pull nomic-embed-text` / `ollama pull llama3.2:3b` |
| Collection not found / empty | Index never built | `python build_index.py` |
| Answers ignore new docs | Index stale | Rebuild after editing `data/knowledge_base/` |
| Timeout errors | Generation exceeded `CHAT_TIMEOUT_S` | Increase timeout or reduce `NUM_PREDICT` |
| Wrong / weak answers | Retrieval or small-model limits | `chat_cli.py --sources`; tune `MAX_DISTANCE` / docs |
| Feels “stuck” with no text | Older non-streaming behaviour | Use default streaming CLI (current default) |

On one measured CPU run (same three in-KB questions, warmed models): average **total** ~173 s before Phase 3 limits, ~144–164 s after `NUM_PREDICT` / streaming experiments; average **TTFT** with streaming ~139 s (user sees text before the full answer finishes). Streaming did **not** reliably cut total generation time.

---

## 19. Privacy considerations and local-data handling

- All processing is intended to stay on your machine (Ollama + local Chroma).  
- Sample orders are fictional; lookup requires an **exact order id**.  
- The bot refuses bulk “list all orders” and name-only lookups.  
- Do not put real customer PII into the knowledge base or `orders.json` for demos.  
- `.chroma/` is local index data; it is gitignored.  

---

## 20. Current limitations and sensible future improvements

**Current limitations**

- CPU latency with 3B-class models  
- No web UI, authentication, or real ticketing  
- No automated LLM answer scoring (eval focuses on retrieval + behaviour gates)  
- Learning notebooks still reflect earlier, smaller KB examples in places  

**Sensible next steps**

- Compare `qwen2.5:0.5b` vs `llama3.2:3b` for speed/quality (with fresh timings)  
- Add a simple Gradio/Streamlit UI that reuses `pipeline.py`  
- Expand eval with automated answer-fact checks where practical  
- Tune chunking for multi-doc questions (e.g. damaged item → return + refund)  

---

## Learning notebooks (optional path)

```bash
source .venv/bin/activate
jupyter notebook notebooks/01_rag_intro_load_and_chunk.ipynb
# then 02_local_embeddings_and_similarity.ipynb
# then 03_chroma_store_and_retrieve.ipynb
# then 04_grounded_answers_with_ollama.ipynb
```

Use notebooks to learn; use `build_index.py` + `chat_cli.py` for the packaged app.

---

## Quick start (checklist)

1. `python3 -m venv .venv && source .venv/bin/activate`  
2. `pip install -r requirements.txt`  
3. `ollama serve`  
4. `ollama pull nomic-embed-text && ollama pull llama3.2:3b`  
5. `python build_index.py`  
6. `python chat_cli.py`  
7. `python scripts/validate_phase2.py && python scripts/validate_phase4.py`  
