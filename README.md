# RAG Chatbot — Medical Field (clinical/textbook style)

## Mirrors the exact stages
from the notebook: chunk → embed → store → retrieve → generate + cite.

## Domain: medical field, clinical/textbook style. Corpus covers 6 conditions —
Type 2 Diabetes, Essential Hypertension, Community-Acquired Pneumonia,
Bronchial Asthma, Iron-Deficiency Anemia, and Migraine — each with
pathophysiology, clinical features, diagnosis, and management, written like a
medical reference/textbook entry. This is study/reference material, not a
tool for diagnosing or treating any real patient — the system prompt makes
that boundary explicit.

## Files
- `rag.py`      — the pipeline itself (all 5 stages, one function each)
- `main.py`     — FastAPI server, `/chat` + `/upload` endpoints, serves the chat UI
- `docs/`       — the corpus (6 clinical .md files) — swap for your own topics
- `static/index.html` — simple chatbot UI (plain HTML/JS) + document upload
- `retrieval_eval.py` — retrieval evaluation harness (golden set, hand-written
  precision@k/recall@k/MRR/nDCG@k, dense → hybrid → rewrite → rerank), run
  cell-by-cell in VS Code (`# %%` blocks)
- `EVAL.md`     — retrieval evaluation report: golden set, results table,
  worst-query diagnosis, shipping decision

## Run it
```bash
cd rag-backend
python -m venv venv
source venv/bin/activate        
pip install -r requirements.txt
cp .env.example .env             
uvicorn main:app --reload --port 8000
```

Open http://localhost:8000 in your browser — that's your chatbot.

## How it works, in order
1. On startup, `rag.build_index()` reads every `.md` file in `docs/`, splits
   it into ~45-word chunks (10-word overlap), embeds them all in one Voyage
   call, and stores them in an in-memory Qdrant collection.
2. Each `POST /chat` call embeds your question, finds the top-k closest
   chunks by cosine similarity, stuffs them into a prompt, and asks Groq
   (Llama 3.3 70B) to answer *only* from those chunks and cite the source.
3. If the answer isn't in the retrieved chunks, the model is instructed to
   say "I don't know based on the provided documents" instead of guessing.

## Retrieval evaluation (`retrieval_eval.py` + `EVAL.md`)
Before shipping a change, retrieval quality is measured, not assumed. A
22-query golden set (labelled by content marker, not chunk ID, so it survives
re-chunking) is scored with hand-written `precision@k`, `recall@k`, `MRR`,
and `nDCG@k` across 8 retriever configs — dense, BM25, hybrid (BM25+RRF),
hybrid+rewrite, multi-query, HyDE, hybrid+rerank, and the full stack.

**Result on this corpus:** the cross-encoder reranker on top of hybrid
retrieval was the clear winner on every metric (R@5 0.950→0.966, nDCG@5
0.865→0.924 vs the dense baseline). Plain hybrid alone actually *hurt*
retrieval here — on a small, synonym-friendly corpus, BM25's noisier ranking
diluted an already-strong dense signal. Full findings, the results table, and
the shipping decision are in `EVAL.md`.

Stack used for evaluation: Jina (embeddings + reranker) + Qdrant + Groq
(free tier, per assignment) — production above uses Voyage; only the
qualitative pattern (reranking helps, multi-query doesn't) is expected to
transfer, not the absolute numbers.

## To make it your own
Replace the files in `docs/` with whatever corpus you actually want (5–15
short docs). Everything else keeps working — the pipeline doesn't care what's
inside the docs, only that each file becomes its own "source" for citations.

## Adding a book
Drop a `.pdf` file straight into `docs/` — it gets picked up automatically
next time the server starts (alongside the `.md`/`.txt` files), text gets
extracted, chunked, and embedded like everything else.

## Letting a user upload their own document
There's a `POST /upload` endpoint (and an "Upload document" button on the
chat page) that accepts a `.pdf`, `.txt`, or `.md` file, chunks + embeds it,
and adds it to the running index immediately — no server restart needed.
Ask a question about it right after uploading to see it get retrieved and cited.
