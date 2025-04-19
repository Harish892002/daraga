# src/main.py

import os
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from src.preprocess import process_book_by_chapter_progress
from src.build_rag import initialize_rag_stack
from llama_index.core.schema import Document

app = FastAPI()

# Allow CORS for any frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryInput(BaseModel):
    book_title: str
    chapter_read: int
    chapter_percent: float
    question: str

def get_documents_upto_progress(docs, sentence_map, chapter_map, chapter_read, chapter_percent):
    # Build cumulative sentence counts per chapter
    total = 0
    limits = {}
    for chap_id, count in sentence_map.items():
        limits[chap_id] = (total, total + count)
        total += count

    # Compute cutoff based on fully‑read chapters + partial current chapter
    cutoff = 0
    for chap_id in sorted(limits):
        if chap_id < chapter_read:
            cutoff += sentence_map[chap_id]
        elif chap_id == chapter_read:
            cutoff += int(sentence_map[chap_id] * (chapter_percent / 100))
            break

    # Return only those chunks whose start is before the cutoff
    allowed = []
    for doc in docs:
        start = doc.metadata.get("sentence_start", 0)
        if start < cutoff:
            allowed.append(doc)
    return allowed

@app.post("/ask")
def ask_question(query_input: QueryInput):
    # Load & chunk the EPUB up to the claimed progress
    docs, sentence_map, chapter_map = process_book_by_chapter_progress(
        query_input.book_title,
        query_input.chapter_read,
        query_input.chapter_percent
    )

    # Filter out any chunks beyond that progress
    context_docs = get_documents_upto_progress(
        docs,
        sentence_map,
        chapter_map,
        query_input.chapter_read,
        query_input.chapter_percent
    )

    # If nothing to retrieve, fall back
    if not context_docs:
        return {"answer": "I don't have context for the asked question."}

    # Build the RAG stack with only the allowed docs
    query_engine = initialize_rag_stack(context_docs, index_name=query_input.book_title)

    # Run the query
    response = query_engine.query(query_input.question)
    return {"answer": str(response)}
