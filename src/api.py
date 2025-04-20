from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv

from src.load import get_gutenberg_id, get_epub_path
from src.loaders.epub import load_epub_documents
from src.build_rag import initialize_rag_stack

load_dotenv()

app = FastAPI()

class AskRequest(BaseModel):
    book_title:      str
    chapter_read:    int
    chapter_percent: float
    question:        str

# Cache full-book engine + metadata maps per book
_index_cache = {}

@app.post("/ask")
def ask(request: AskRequest):
    try:
        # Load book and docs
        book_id = str(get_gutenberg_id(request.book_title))
        epub_path = get_epub_path(book_id)
        docs, sentence_map, chapter_map = load_epub_documents(epub_path)

        # Initialize engine once
        if book_id not in _index_cache:
            engine = initialize_rag_stack(docs, index_name=book_id)
            _index_cache[book_id] = {
                "engine": engine,
                "sentence_map": sentence_map,
                "chapter_map": chapter_map
            }
        cache = _index_cache[book_id]
        engine = cache["engine"]
        sentence_map = cache["sentence_map"]
        chapter_map = cache["chapter_map"]

        # Compute slice
        spine_ids = sorted(chapter_map.keys())
        idx = max(1, min(request.chapter_read, len(spine_ids))) - 1
        target_spine = spine_ids[idx]
        total_sents = sentence_map[target_spine]
        sent_thresh = int((request.chapter_percent / 100.0) * total_sents)

        # Apply metadata filter
        engine.retriever.metadata_filters = {
            "$or": [
                {"spine_index": {"$lt": target_spine}},
                {"$and": [
                    {"spine_index": {"$eq": target_spine}},
                    {"sentence_start": {"$lt": sent_thresh}}
                ]}
            ]
        }

        # Run the query
        response = engine.query(request.question)

        # If no source nodes, fallback
        if not getattr(response, "source_nodes", None):
            return {"answer": "I don't have context for that question.", "sources": []}

        # Return answer and sources
        return {
            "answer":  str(response),
            "sources": [node.metadata for node in response.source_nodes]
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))