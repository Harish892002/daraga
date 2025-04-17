import os
from dotenv import load_dotenv
from src.loaders.epub import download_epub_from_gutenberg, load_epub_documents

load_dotenv()
GUTENBERG_ID = os.getenv("GUTENBERG_ID", "1661")

def load_documents_by_chapter_progress(chapter_read: int, chapter_percent: float):
    epub_path = download_epub_from_gutenberg(GUTENBERG_ID)
    full_docs, sentence_map, chapter_map = load_epub_documents(epub_path)

    # Calculate sentence threshold for access
    sentence_limit = 0
    for i in range(1, chapter_read):
        sentence_limit += sentence_map.get(i, 0)

    chapter_sentences = sentence_map.get(chapter_read, 0)
    cutoff_sentence = int((chapter_percent / 100.0) * chapter_sentences)
    sentence_limit += cutoff_sentence

    print(f"\n📘 Sentence limit: {sentence_limit} (Up to Chapter {chapter_read} at {chapter_percent:.1f}%)")
    print(f"🔎 Chapter Title: {chapter_map.get(chapter_read, 'Unknown')}")

    accessible_docs = []
    for doc in full_docs:
        chap = doc.metadata.get("spine_index", 0)
        start = doc.metadata.get("sentence_start", 0)
        end = doc.metadata.get("sentence_end", 0)

        # Debug filtering info
        # print(f"Checking: Chapter {chap} ({doc.metadata.get('chapter_title', 'N/A')}) | Sentences {start}-{end} -> ", end="")

        if chap < chapter_read:
            accessible_docs.append(doc)
            print("✅ Included (earlier chapter)")
        elif chap == chapter_read and start <= cutoff_sentence:
            accessible_docs.append(doc)
            print("✅ Included (partial chapter)")
        else:
            print("❌ Excluded")

    return accessible_docs