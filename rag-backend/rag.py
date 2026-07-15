

import os
import glob
import io
import voyageai
from pypdf import PdfReader
from qdrant_client import QdrantClient, models
from groq import Groq

EMBED_MODEL = "voyage-3.5"
GEN_MODEL = "llama-3.3-70b-versatile"
COLLECTION = "simple_rag_docs"
DOCS_DIR = os.path.join(os.path.dirname(__file__), "docs")

# Running counter for chunk ids. build_index() resets it; add_document() keeps
# incrementing it so new uploads never collide with existing point ids.
_next_id = 0

SYSTEM_PROMPT = (
    "You are a clinical reference assistant for medical-field study material. Answer "
    "the question using ONLY the numbered context passages given to you, in a "
    "textbook-style tone (pathophysiology, clinical features, diagnosis, management). "
    "Cite the sources you use inline, like [type2_diabetes.md]. If the answer is not "
    "in the context, reply exactly: \"I don't know based on the provided documents.\" "
    "Never use outside knowledge. This is educational reference material, not a "
    "diagnosis or treatment plan for any specific patient."
)

voyage = voyageai.Client(api_key=os.getenv("VOYAGE_API_KEY"))
groq = Groq(api_key=os.getenv("GROQ_API_KEY"))
qdrant = QdrantClient(":memory:")  # in-memory vector DB, resets each time the server restarts


# ---------- Stage 1: chunk ----------
def chunk_text(text, source, size=45, overlap=10):
    """Split into ~size-word chunks with overlap words shared between neighbours."""
    words = text.split()
    if not words:
        return []
    chunks, start, step = [], 0, max(1, size - overlap)
    while start < len(words):
        piece = " ".join(words[start:start + size]).strip()
        if piece:
            chunks.append({"text": piece, "source": source})
        start += step
    return chunks


def read_pdf_bytes(file_bytes):
    """Extract plain text from a PDF (books work fine here too)."""
    reader = PdfReader(io.BytesIO(file_bytes))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def extract_text(filename, file_bytes):
    """Turn raw file bytes into plain text, based on extension."""
    if filename.lower().endswith(".pdf"):
        return read_pdf_bytes(file_bytes)
    return file_bytes.decode("utf-8", errors="ignore")  # .md / .txt


def load_and_chunk_docs(size=45, overlap=10):
    """Load every file in docs/ — .md, .txt, and .pdf (e.g. a book) are all supported."""
    global _next_id
    chunks = []
    paths = sorted(
        glob.glob(os.path.join(DOCS_DIR, "*.md"))
        + glob.glob(os.path.join(DOCS_DIR, "*.txt"))
        + glob.glob(os.path.join(DOCS_DIR, "*.pdf"))
    )
    for path in paths:
        source = os.path.basename(path)
        with open(path, "rb") as f:
            text = extract_text(source, f.read())
        chunks.extend(chunk_text(text, source, size=size, overlap=overlap))
    _next_id = 0
    for c in chunks:
        c["id"] = _next_id
        _next_id += 1
    return chunks


# ---------- Stage 2 + 3: embed + store ----------
def _embed_and_upsert(chunks):
    """Shared helper: embed a list of chunks and upsert them into Qdrant."""
    texts = [c["text"] for c in chunks]
    resp = voyage.embed(texts, model=EMBED_MODEL, input_type="document")
    dense_vectors = resp.embeddings
    points = [
        models.PointStruct(
            id=c["id"],
            vector={"dense": dvec},
            payload={"text": c["text"], "source": c["source"]},
        )
        for c, dvec in zip(chunks, dense_vectors)
    ]
    qdrant.upsert(collection_name=COLLECTION, points=points)
    return len(dense_vectors[0]) if dense_vectors else None


def build_index(size=45, overlap=10):
    """Run once at startup. Chunks every doc in docs/ (including any book PDFs), embeds, stores."""
    chunks = load_and_chunk_docs(size=size, overlap=overlap)
    texts = [c["text"] for c in chunks]

    resp = voyage.embed(texts, model=EMBED_MODEL, input_type="document")
    dense_vectors = resp.embeddings
    dim = len(dense_vectors[0])

    if qdrant.collection_exists(COLLECTION):
        qdrant.delete_collection(COLLECTION)
    qdrant.create_collection(
        collection_name=COLLECTION,
        vectors_config={"dense": models.VectorParams(size=dim, distance=models.Distance.COSINE)},
    )

    points = [
        models.PointStruct(
            id=c["id"],
            vector={"dense": dvec},
            payload={"text": c["text"], "source": c["source"]},
        )
        for c, dvec in zip(chunks, dense_vectors)
    ]
    qdrant.upsert(collection_name=COLLECTION, points=points)
    print(f"[rag] indexed {len(points)} chunks from {DOCS_DIR}")
    return len(points)


# ---------- User uploads: add a document to the already-running index ----------
def add_document(filename, file_bytes, size=45, overlap=10):
    """
    Let a user add their own document (book, notes, PDF, .txt, .md) at runtime,
    without rebuilding the whole index. Same chunk -> embed -> store stages as
    build_index, just applied to one new file and appended instead of replacing.
    """
    global _next_id
    text = extract_text(filename, file_bytes)
    new_chunks = chunk_text(text, filename, size=size, overlap=overlap)
    for c in new_chunks:
        c["id"] = _next_id
        _next_id += 1

    if not new_chunks:
        return 0

    _embed_and_upsert(new_chunks)
    print(f"[rag] added {len(new_chunks)} chunks from uploaded file '{filename}'")
    return len(new_chunks)


# ---------- Stage 4: retrieve ----------
def embed_query(query):
    return voyage.embed([query], model=EMBED_MODEL, input_type="query").embeddings[0]


def retrieve(query, k=5):
    qvec = embed_query(query)
    hits = qdrant.query_points(
        collection_name=COLLECTION, query=qvec, using="dense",
        limit=k, with_payload=True,
    ).points
    return [{"score": h.score, "source": h.payload["source"], "text": h.payload["text"]} for h in hits]


# ---------- Stage 5: generate + cite ----------
def format_context(hits):
    return "\n".join(f"[{i + 1}] (source: {h['source']}) {h['text']}" for i, h in enumerate(hits))


def rag_answer(query, k=5):
    hits = retrieve(query, k=k)
    user_msg = f"Context:\n{format_context(hits)}\n\nQuestion: {query}"
    resp = groq.chat.completions.create(
        model=GEN_MODEL,
        temperature=0,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
    )
    answer = resp.choices[0].message.content
    sources = sorted({h["source"] for h in hits})
    return {"answer": answer, "sources": sources, "chunks_used": hits}
