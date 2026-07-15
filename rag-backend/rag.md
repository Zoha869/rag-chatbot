# RAG Report

## Project
Medical RAG Chatbot

## Corpus

I created a medical knowledge base containing six clinical documents:

- Type 2 Diabetes
- Essential Hypertension
- Community-Acquired Pneumonia
- Asthma
- Anemia
- Migraine

Each document includes:
- Pathophysiology
- Clinical Features
- Diagnosis
- Management

---

## Chunking

Chunk Size: 45 words

Overlap: 10 words

### Why?

A chunk size of 45 words preserves enough medical context while keeping retrieval focused. A 10-word overlap prevents important information from being split across chunks.

---

## Embedding Model

Voyage-3.5



## Vector Database

Qdrant (in-memory)

## Generation Model

Groq Llama-3.3-70B-Versatile

## Top-k

k = 3

### Why?

Retrieving the top 3 chunks provides enough relevant medical information without adding unnecessary context.
Qdrant give relevent chunks
## Top-K
k=5
Retrieving the top 5 chunks provides enough information ful detailed context.
Qdrant give relevent chunks
---

## Guardrail Test

Question:

What is the treatment for appendicitis?

Answer:

"I don't know based on the provided documents."

This demonstrates that the chatbot only answers using the provided medical documents.

---

## Limitation

The chatbot can only answer questions related to the uploaded medical documents. It cannot answer questions about diseases or treatments that are not included in the corpus.

---

## Future Improvements

- Add more medical documents
- Support larger medical textbooks
- Improve retrieval using re-ranking
- Store vectors in a persistent Qdrant database