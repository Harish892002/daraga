import os
import nltk
from nltk.tokenize import sent_tokenize
from src.loaders.epub import load_epub_documents
from src.load import get_epub_path

nltk.download('punkt')

def process_book_by_chapter_progress(gutenberg_id: str, chapter_number_or_name: str or int, chapter_percent: float):
    epub_path = get_epub_path(gutenberg_id)
    documents, sentence_map, chapter_map = load_epub_documents(epub_path)

    # Total sentence count in the entire book
    total_sentences = sum(sentences for sentences in sentence_map.values())

    # Calculate number of sentences to include based on percentage
    target_sentence_count = int((chapter_percent / 100.0) * total_sentences)

    cumulative_count = 0
    allowed_docs = []
    for doc in documents:
        chapter = doc.metadata.get("chapter")
        sent_range = doc.metadata.get("sentence_range", (0, 0))
        count = sent_range[1] - sent_range[0]
        cumulative_count += count
        allowed_docs.append(doc)
        if cumulative_count >= target_sentence_count:
            break

    return allowed_docs, sentence_map, chapter_map
