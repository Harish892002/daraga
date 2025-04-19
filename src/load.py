import requests
import os

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

def get_gutenberg_id(book_title: str) -> int:
    try:
        response = requests.get(f"https://gutendex.com/books?search={book_title}")
        response.raise_for_status()
        data = response.json()
        if data["count"] == 0:
            raise ValueError("No book found with the given title.")
        return data["results"][0]["id"]
    except Exception as e:
        raise RuntimeError(f"Failed to get Gutenberg ID: {e}")

def get_epub_path(gutenberg_id: int) -> str:
    epub_path = os.path.join(DATA_DIR, f"{gutenberg_id}.epub")
    if os.path.exists(epub_path):
        return epub_path

    print(f"📚 Fetching book from Project Gutenberg: {gutenberg_id}")
    try:
        response = requests.get(
            f"https://www.gutenberg.org/ebooks/{gutenberg_id}.epub.images",
            allow_redirects=True
        )
        if response.status_code == 200:
            real_url = response.url.replace("ebooks", "files").replace(".epub.images", f"/{gutenberg_id}.epub")
            epub_response = requests.get(real_url)
            epub_response.raise_for_status()
            with open(epub_path, "wb") as f:
                f.write(epub_response.content)
            return epub_path
        else:
            raise ValueError("EPUB file not available for this book.")
    except Exception as e:
        raise RuntimeError(f"Failed to download EPUB: {e}")
