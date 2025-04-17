import os
import requests
from ebooklib import epub, ITEM_DOCUMENT
from bs4 import BeautifulSoup
from llama_index.core.schema import Document
from tempfile import NamedTemporaryFile
from nltk.tokenize import sent_tokenize
import nltk
import re

nltk.download("punkt")

CHUNK_SIZE = 5  # number of sentences per sub-chunk


def download_epub_from_gutenberg(book_id_or_url: str) -> str:
    if book_id_or_url.isdigit():
        url = f"https://www.gutenberg.org/ebooks/{book_id_or_url}.epub.images"
        response = requests.get(url, allow_redirects=True)
        if response.status_code == 200:
            redirected_url = response.url
            epub_data = requests.get(redirected_url).content
        else:
            raise ValueError(f"Could not fetch EPUB for book ID {book_id_or_url}")
    elif book_id_or_url.startswith("http"):
        epub_data = requests.get(book_id_or_url).content
    else:
        raise ValueError("Provide a valid Project Gutenberg ID or URL.")

    with NamedTemporaryFile(delete=False, suffix=".epub") as tmp_file:
        tmp_file.write(epub_data)
        return tmp_file.name


def load_epub_documents(epub_path: str):
    book = epub.read_epub(epub_path)
    manifest = list(book.get_items_of_type(ITEM_DOCUMENT))

    docs = []
    sentence_map = {}  # chapter index -> total sentences
    chapter_map = {}  # chapter index -> visible chapter name
    chapter_pattern = re.compile(r"chapter\s+(\d+|[ivxlc]+)", re.IGNORECASE)

    chapter_idx = 1

    for item in manifest:
        soup = BeautifulSoup(item.get_content(), "html.parser")
        text = soup.get_text(separator="\n").strip()
        if not text:
            continue

        # Try to detect chapter heading
        heading = soup.find(['h1', 'h2', 'h3'])
        heading_text = heading.get_text(strip=True) if heading else ""
        match = chapter_pattern.search(heading_text)
        if match:
            chapter_map[chapter_idx] = heading_text
        else:
            chapter_map[chapter_idx] = f"(non-chapter) {item.get_name()}"

        # Sentence chunking
        sentences = sent_tokenize(text)
        sentence_map[chapter_idx] = len(sentences)

        for i in range(0, len(sentences), CHUNK_SIZE):
            chunk_sentences = sentences[i:i + CHUNK_SIZE]
            chunk_text = " ".join(chunk_sentences)

            docs.append(Document(
                text=chunk_text,
                metadata={
                    "spine_index": chapter_idx,
                    "sentence_start": i + 1,
                    "sentence_end": min(i + CHUNK_SIZE, len(sentences)),
                    "chapter_title": heading_text,
                    "source_file": item.get_name(),
                    "source": os.path.basename(epub_path),
                }
            ))

        chapter_idx += 1

    return docs, sentence_map, chapter_map
