# src/api.py

import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from src.load import get_gutenberg_id, get_epub_path
from src.preprocess import process_book_by_chapter_progress
from src.build_rag import initialize_rag_stack
from llama_index.core.schema import Document
from typing import Union

load_dotenv()
app = FastAPI()

class AskRequest(BaseModel):
    book_title: str
    chapter_read: Union[int, str]
    chapter_percent: float
    question: str

@app.post("/ask")
async def ask_book(request: AskRequest):
    try:
        # Resolve Gutenberg ID and ensure EPUB is downloaded
        gutenberg_id = get_gutenberg_id(request.book_title)
        epub_path = get_epub_path(gutenberg_id)

        # Process & chunk up to the user’s read progress
        docs, sentence_map, chapter_map = process_book_by_chapter_progress(
            str(gutenberg_id),
            request.chapter_read,
            request.chapter_percent
        )

        # Flatten and filter for Document instances
        if docs and isinstance(docs[0], list):
            docs = [d for sub in docs for d in sub]
        docs = [d for d in docs if isinstance(d, Document)]

        # No context?  Fallback
        if not docs:
            return {"answer": "I don't have context for the asked question.", "sources": []}

        # Build RAG and run query
        query_engine = initialize_rag_stack(docs, index_name=str(gutenberg_id))
        response = query_engine.query(request.question)

        # If no source nodes, fallback
        if not getattr(response, "source_nodes", None):
            return {"answer": "I don't have context for the asked question.", "sources": []}

        return {
            "answer": str(response),
            "sources": [node.metadata for node in response.source_nodes]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))