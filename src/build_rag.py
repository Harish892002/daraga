import os
from dotenv import load_dotenv

from pinecone import Pinecone, ServerlessSpec

from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.postprocessor import LLMRerank

load_dotenv()

PINECONE_API_KEY = os.environ["PINECONE_API_KEY"]
PINECONE_ENV     = os.environ.get("PINECONE_ENV", "us-east-1")
MODEL            = os.environ.get("MODEL", "hermes3:8b")
EMBED_MODEL      = os.environ.get("EMBED_MODEL", "intfloat/multilingual-e5-large")
DEFAULT_DIM      = int(os.environ.get("EMBED_DIM", 1024))

def initialize_rag_stack(documents, index_name: str):
    """
    Ensure a Pinecone index for the full book exists and vectorizes the book only once.
    Returns a RetrieverQueryEngine for querying.
    """
    # 1) Setup Pinecone index
    pc = Pinecone(api_key=PINECONE_API_KEY)
    existing = pc.list_indexes().names()
    if index_name not in existing:
        pc.create_index(
            name=index_name,
            dimension=DEFAULT_DIM,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region=PINECONE_ENV),
        )
    pinecone_index = pc.Index(index_name)

    # 2) Prepare embedding and LLM
    embed_model  = HuggingFaceEmbedding(model_name=EMBED_MODEL)
    llm          = Ollama(model=MODEL)
    vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
    storage_ctx  = StorageContext.from_defaults(vector_store=vector_store)

    # 3) Only upsert if index empty
    stats = pinecone_index.describe_index_stats()
    vector_count = stats.get("namespaces", {}).get("", {}).get("vector_count", 0)
    if vector_count == 0:
        print(f"📥 Index empty, upserting {len(documents)} docs…")
        index = VectorStoreIndex.from_documents(
            documents,
            storage_context=storage_ctx,
            embed_model=embed_model,
            llm=llm
        )
    else:
        print(f"📦 Index already has {vector_count} vectors—skipping upsert.")
        # Wrap existing vectors into an index object without re-upserting
        index = VectorStoreIndex.from_documents(
            [],
            storage_context=storage_ctx,
            embed_model=embed_model,
            llm=llm
        )

    # 4) Build retriever + reranker
    retriever = VectorIndexRetriever(index=index, similarity_top_k=5)
    reranker  = LLMRerank(choice_batch_size=3, llm=llm, top_n=1)
    engine    = RetrieverQueryEngine.from_args(
        retriever=retriever,
        node_postprocessors=[reranker],
        llm=llm
    )
    return engine