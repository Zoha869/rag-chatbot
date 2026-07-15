import os
from dotenv import load_dotenv, find_dotenv
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

load_dotenv(find_dotenv())

for name in ("VOYAGE_API_KEY", "GROQ_API_KEY"):
    if not os.getenv(name):
        raise RuntimeError(f"{name} is missing — add it to your .env file.")

import rag

app = FastAPI(title="Simple RAG Chatbot")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    message: str
    k: int = 3


@app.on_event("startup")
def startup():
    # Build the index once when the server starts (in-memory, so it resets on restart)
    rag.build_index(size=45, overlap=10)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat")
def chat(req: ChatRequest):
    result = rag.rag_answer(req.message, k=req.k)
    return result


@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    """Let a user add their own document (book, notes, PDF, .txt, .md) to the index."""
    file_bytes = await file.read()
    n_chunks = rag.add_document(file.filename, file_bytes)
    return {"filename": file.filename, "chunks_added": n_chunks}


# Serve the simple chat UI at http://localhost:8000/
app.mount("/", StaticFiles(directory="static", html=True), name="static")