from fastapi import FastAPI
from pydantic import BaseModel
from src.loaders.epub_loader import download_epub_from_gutenberg, load_epub_documents
from src.load_book import load_documents_by_chapter_progress
from src.build_rag import initialize_rag_stack
import requests
from bs4 import BeautifulSoup

app = FastAPI()

class AskRequest(BaseModel):
    question: str
    chapter: int
    percent: float
    book_id: str = None
    book_title: str = None

def search_gutenberg_book_id(title: str) -> str:
    query = title.strip().replace(" ", "+")
    search_url = f"https://www.gutenberg.org/ebooks/search/?query={query}"
    response = requests.get(search_url)
    soup = BeautifulSoup(response.text, "html.parser")
    first_link = soup.select_one("li.booklink a.link")
    if first_link and first_link.get("href"):
        return first_link["href"].split("/")[-1]
    raise ValueError("No book found on Project Gutenberg with that title.")

@app.post("/ask")
def ask_question(req: AskRequest):
    try:
        if req.book_id:
            gutenberg_id = req.book_id
        elif req.book_title:
            gutenberg_id = search_gutenberg_book_id(req.book_title)
        else:
            return {"error": "Please provide either 'book_id' or 'book_title'."}

        epub_path = download_epub_from_gutenberg(gutenberg_id)
        full_docs, sentence_map, chapter_map = load_epub_documents(epub_path)
        max_chapter = max(sentence_map.keys())

        if not (1 <= req.chapter <= max_chapter):
            return {"error": f"Chapter must be between 1 and {max_chapter}"}
        if not (0.0 <= req.percent <= 100.0):
            return {"error": "Percent must be between 0 and 100"}

        # 1. Load document chunks based on progress
        docs = load_documents_by_chapter_progress(req.chapter, req.percent, sentence_map, full_docs)

        # 2. Initialize retriever
        query_engine = initialize_rag_stack(docs)

        # 3. Query the index
        response = query_engine.query(req.question)

        # 4. Validate or fallback
        if not response or not response.source_nodes:
            return {"answer": "I don't have context for this question."}

        return {
            "answer": str(response),
            "sources": [
                {
                    "chapter": node.metadata.get("spine_index"),
                    "title": node.metadata.get("chapter_title"),
                    "range": f"{node.metadata.get('sentence_start')}-{node.metadata.get('sentence_end')}"
                }
                for node in response.source_nodes
            ]
        }

    except Exception as e:
        return {"error": str(e)}
