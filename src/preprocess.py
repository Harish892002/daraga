from typing import List
from llama_index.core.schema import Document


def process_book(documents: List[Document], max_spine_index: int = None) -> List[Document]:
    if max_spine_index:
        return [doc for doc in documents if doc.metadata.get("spine_index", 0) <= max_spine_index]
    return documents