import os
import nltk
from nltk.tokenize import sent_tokenize
from src.loaders.epub import load_epub_documents
from src.load import get_epub_path

nltk.download('punkt')

def process_book_by_chapter_progress(gutenberg_id: str, chapter_number_or_name, chapter_percent: float):
    epub_path = get_epub_path(gutenberg_id)
    documents, sentence_map, chapter_map = load_epub_documents(epub_path)

    # Parse chapter_read to int
    if isinstance(chapter_number_or_name, str) and chapter_number_or_name.isdigit():
        chapter_read = int(chapter_number_or_name)
    elif isinstance(chapter_number_or_name, int):
        chapter_read = chapter_number_or_name
    else:
        raise ValueError("Invalid chapter number")

    # Count how many sentences to allow in total
    sentence_limit = 0
    for i in range(1, chapter_read):
        sentence_limit += sentence_map.get(i, 0)

    current_chapter_sentences = sentence_map.get(chapter_read, 0)
    current_cutoff = int((chapter_percent / 100.0) * current_chapter_sentences)
    sentence_limit += current_cutoff

    filtered_docs = []
    for doc in documents:
        chap = doc.metadata.get("spine_index", 0)
        start = doc.metadata.get("sentence_start", 0)

        if chap < chapter_read:
            filtered_docs.append(doc)
        elif chap == chapter_read and start <= current_cutoff:
            filtered_docs.append(doc)

    return filtered_docs, sentence_map, chapter_map