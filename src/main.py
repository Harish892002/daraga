from dotenv import load_dotenv
from src.loaders.epub import download_epub_from_gutenberg, load_epub_documents
from src.load import load_documents_by_chapter_progress
from src.build_rag import initialize_rag_stack
import re

load_dotenv()

# 📘 Load full EPUB manifest to determine max chapters
epub_path = download_epub_from_gutenberg("1661")
docs, sentence_map, chapter_map = load_epub_documents(epub_path)
max_chapter = max(sentence_map.keys())

print(f"\n📘 This book has {max_chapter} manifest chapters.\n")

# 🔢 Show available chapters to user
for idx in range(1, max_chapter + 1):
    print(f"{idx}. {chapter_map.get(idx, 'Unknown')}")

# 🔢 Ask user for input
try:
    chapter_read = int(input(f"\n📖 Up to which chapter have you read? (1–{max_chapter}): "))
    chapter_percent = float(input(f"📊 What percentage of Chapter {chapter_read} have you read? (0–100): "))
    if chapter_read < 1 or chapter_read > max_chapter or not (0 <= chapter_percent <= 100):
        raise ValueError
except ValueError:
    print(f"⚠️ Invalid input. Defaulting to Chapter {max_chapter}, 100%.")
    chapter_read = max_chapter
    chapter_percent = 100.0

# 📄 Load document chunks up to this point
documents = load_documents_by_chapter_progress(chapter_read, chapter_percent)

# 🧠 Build index
query_engine = initialize_rag_stack(documents)

# 💬 Query loop
while True:
    query = input("> Ask a question (or type 'exit'): ")
    if query.lower() in ["exit", "quit"]:
        break
    try:
        response = query_engine.query(query)

        # 📎 Validate context scope
        valid = True
        for node in response.source_nodes:
            chap = node.metadata.get("spine_index", 0)
            start = node.metadata.get("sentence_start", 0)
            if chap > chapter_read or (chap == chapter_read and start > int((chapter_percent / 100.0) * sentence_map[chap])):
                valid = False
                break

        # 🔍 Keyword Sanity Check (better tokenization)
        query_keywords = re.findall(r"\b[a-zA-Z][a-zA-Z]+\b", query.lower())
        keyword_match = any(
            any(kw in node.text.lower() for kw in query_keywords)
            for node in response.source_nodes
        )

        if not response or not valid or not keyword_match:
            print("\n❓ I don't have context for this question.\n")
            continue

        # ✅ Valid and relevant
        print(f"\n🧠 Answer:\n{response}\n")

        # 🔍 Debug: Show source chunks used
        print("📎 Source Chunks Used:")
        for node in response.source_nodes:
            meta = node.metadata
            print(f" - Chapter {meta.get('spine_index')} ({meta.get('chapter_title')}) | Sentences {meta.get('sentence_start')}-{meta.get('sentence_end')}")

    except Exception as e:
        print(f"\n❗ Error: {e}\n")
