# Document chat

A local chatbot that answers from PDFs you upload. The browser UI is React. Flask serves that page and the API. Embeddings and the vector index stay on this machine. There is no built-in knowledge base and no paid API.

---

## What it does

1. You upload one or more text-based PDFs.
2. The app extracts the text layer, keeps the page number, and splits the text into chunks. Long paragraphs overlap so a sentence in the middle of a page is not cut in half. When several PDFs are selected, retrieval keeps at least the closest chunk from each one.
3. Chunks are embedded with Ollama (`nomic-embed-text`) and stored in a local Chroma collection.
4. A question searches only the PDFs you have selected.
5. The closest chunks are sent to a local vLLM server. If vLLM is down, the app falls back to the Ollama chat model.
6. The answer is streamed and cites the filename and page.

If you have not selected a PDF, the reply is: “Select at least one indexed PDF before asking a question.” If the selected pages do not contain the fact, the model is told to say so.

A leftover file named `harbor-note.txt` is deleted on startup and is never searched.

---

## Layout

```
chatbot-agents/
├── README.md
├── requirements.txt
├── web_app.py                 # Flask page on :5000 and /api routes
├── frontend/                  # React UI (Vite)
│   ├── src/App.jsx
│   ├── src/api.js
│   ├── src/components/        # sidebar and chat pane
│   └── dist/                  # production build served by Flask (gitignored)
├── rag_chatbot/
│   ├── config.py              # paths, models, TOP_K, NUM_PREDICT
│   ├── uploads.py             # PDF extract, chunk, embed, selected-document retrieval
│   ├── vectorstore.py         # Chroma open / query
│   ├── ollama_client.py       # embeddings and chat fallback
│   └── vllm_client.py         # stream answers from local vLLM
├── scripts/validate_uploads.py
├── notebooks/
│   └── 01_pdf_rag_walkthrough.ipynb
├── data/uploads/              # saved PDFs and manifest.json (gitignored)
└── .chroma_uploads/           # vector index (gitignored)
```

Settings live in `rag_chatbot/config.py`. There is no `.env` file.

---

## Requirements

- Python 3.10 or newer
- Node.js, to build the React page
- [Ollama](https://ollama.com) with `nomic-embed-text` (embeddings) and `llama3.2:3b` (chat fallback)
- A local vLLM server at `http://127.0.0.1:8000/v1` serving `Qwen/Qwen2.5-7B-Instruct` for document answers
- Enough disk for those models

OCR is not installed. A scanned or image-only PDF is rejected.

---

## Setup

```bash
cd /path/to/chatbot-agents
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

ollama serve
ollama pull nomic-embed-text
ollama pull llama3.2:3b

cd frontend
npm install
npm run build
cd ..
python web_app.py
```

Open http://127.0.0.1:5000.

For frontend work with live reload, leave Flask running and, in another terminal, run `npm run dev` inside `frontend/`. Open http://127.0.0.1:5173. That dev server proxies `/api` to Flask. After changing React code, run `npm run build` again so port 5000 stays current.

---

## Using the page

1. Drop a PDF on the sidebar, or click **Upload PDF**. The limit is 16 MB. Empty files, non-PDFs, encrypted PDFs that cannot be decrypted, and scanned PDFs are rejected.
2. Wait until the notice says the file is ready. Uploading a file selects it. The same bytes are not indexed twice.
3. Check only the documents you want to query. The question box stays disabled until at least one ready PDF is selected.
4. Ask a question. Under the answer, each retrieved excerpt is shown with its filename and page. A short follow-up uses the previous question to find the right passage. Refreshing the page clears that conversation.
5. **Remove** deletes that PDF and its chunks. No restart is required.

---

## API

| Method | Path | Role |
|--------|------|------|
| `GET` | `/` | Built React page |
| `GET` | `/api/files` | Indexed PDFs |
| `POST` | `/api/upload` | Save, extract, and embed one PDF |
| `DELETE` | `/api/files/<name>` | Remove one PDF and its chunks |
| `POST` | `/api/clear` | Remove every upload |
| `POST` | `/api/chat` | Stream an answer for `{ "question", "documents" }` |

`documents` is the list of selected filenames. Any other field in the JSON body is ignored.

---

## Configuration

Edit `rag_chatbot/config.py`:

| Variable | Role |
|----------|------|
| `EMBED_MODEL` | Ollama embedding model (`nomic-embed-text`) |
| `CHAT_MODEL` | Ollama chat model used only if vLLM is unreachable |
| `VLLM_URL` / `VLLM_MODEL` | Local vLLM endpoint for document answers |
| `TOP_K` | How many chunks go into the prompt (default `6`) |
| `NUM_PREDICT` | Max new tokens (default `512`) |
| `TEMPERATURE` | Sampling temperature (default `0.2`) |
| `UPLOAD_DIR` | Folder for saved PDFs |
| `UPLOAD_CHROMA_DIR` | Folder for the vector index |

---

## Walkthrough notebook

`notebooks/01_pdf_rag_walkthrough.ipynb` explains the current pipeline and runs it: PDF text extraction, paragraph chunking, Ollama embeddings, the Chroma index, selected-document retrieval, and a streamed local answer.

```bash
source .venv/bin/activate
pip install -r requirements.txt
ollama serve
jupyter notebook notebooks/01_pdf_rag_walkthrough.ipynb
```

Run the cells from top to bottom. The notebook indexes `notebook-demo.pdf`, then the last cell deletes that file. Other uploads are left in place. Answer cells use vLLM and fall back to `CHAT_MODEL` if vLLM is unreachable.

---

## Checks

```bash
source .venv/bin/activate
python scripts/validate_uploads.py
```

The script creates temporary PDFs, checks indexing and selection, then deletes them. If you already have uploads, copy `data/uploads/` and `.chroma_uploads/` first. The script clears those folders.

---

## Troubleshooting

| Symptom | What to try |
|---------|-------------|
| Blank page at port 5000 | `cd frontend && npm run build`, restart `python web_app.py`, then hard-refresh |
| Page cannot reach the API | Start `python web_app.py` |
| “Cannot reach the local embedding service” | `ollama serve` |
| Missing model | `ollama pull nomic-embed-text` and `ollama pull llama3.2:3b` |
| Scanned PDF rejected | Upload a PDF with selectable text. OCR is not enabled |
| vLLM is down | Answers fall back to `CHAT_MODEL` on Ollama, which is slower on CPU |

---

## Limits

- No login and no OCR
- Answers can still miss a fact that is in the PDF if retrieval does not surface that page
- Uploaded files stay on this machine under `data/uploads/` and are gitignored
