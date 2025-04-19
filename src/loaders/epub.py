import os
import requests
from ebooklib import epub, ITEM_DOCUMENT
from bs4 import BeautifulSoup
import nltk
from llama_index.core.schema import Document
nltk.download('punkt')
from nltk.tokenize import sent_tokenize

DATA_DIR = "data"

def resolve_book_name_to_gutenberg_id(book_name: str) -> str:
    """
    Queries the Gutendex API to find the best matching Gutenberg ID for a given book title.
    """
    try:
        response = requests.get(f"https://gutendex.com/books/?search={book_name}")
        response.raise_for_status()
        results = response.json()["results"]
        if results:
            return str(results[0]["id"])
    except Exception:
        pass
    return None

def download_epub_from_gutenberg(gutenberg_id):
    url = f"https://www.gutenberg.org/ebooks/{gutenberg_id}.epub.images"
    response = requests.get(url, allow_redirects=True)
    if response.status_code != 200:
        raise ValueError(f"Failed to download EPUB: {url}")

    book_path = os.path.join(DATA_DIR, f"{gutenberg_id}.epub")
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(book_path, 'wb') as f:
        f.write(response.content)

    return book_path

def load_epub_documents(epub_path):
    book = epub.read_epub(epub_path)
    sentence_map = {}
    chapter_map = {}
    all_docs = []

    for i, item in enumerate(book.spine):
        idref = item[0]
        chapter = book.get_item_with_id(idref)
        if chapter is None or chapter.get_type() != ITEM_DOCUMENT:
            continue

        title = chapter.get_name() or f"Chapter {i+1}"
        soup = BeautifulSoup(chapter.get_content(), 'html.parser')
        text = soup.get_text().strip()

        if not text:
            continue

        sentences = sent_tokenize(text)
        sentence_map[i] = len(sentences)
        chapter_map[i] = title

        chunk_size = 5
        for j in range(0, len(sentences), chunk_size):
            chunk = " ".join(sentences[j:j+chunk_size])
            if chunk.strip():
                doc = Document(
                    text=chunk,
                    metadata={
                        "spine_index": i,
                        "chapter_title": title,
                        "sentence_start": j,
                        "sentence_end": min(j + chunk_size - 1, len(sentences) - 1)
                    }
                )
                all_docs.append(doc)

    return all_docs, sentence_map, chapter_map
