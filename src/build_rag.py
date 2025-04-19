import os
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.core import StorageContext, VectorStoreIndex, Settings
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.postprocessor import LLMRerank

load_dotenv()

PINECONE_API_KEY = os.environ["PINECONE_API_KEY"]
PINECONE_ENV = os.environ.get("PINECONE_ENV", "us-east-1")
MODEL = os.environ.get("MODEL", "hermes3:8b")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "intfloat/multilingual-e5-large")
DEFAULT_DIM = int(os.environ.get("EMBED_DIM", 1024))

def initialize_rag_stack(documents, index_name: str):
    pc = Pinecone(api_key=PINECONE_API_KEY)
    existing = pc.list_indexes().names()
    if index_name not in existing:
        print(f"✅ Creating Pinecone index '{index_name}'")
        pc.create_index(
            name=index_name,
            dimension=DEFAULT_DIM,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region=PINECONE_ENV),
        )
    pinecone_index = pc.Index(index_name)

    embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL)
    llm = Ollama(model=MODEL)
    Settings.embed_model = embed_model
    Settings.llm = llm

    storage_context = StorageContext.from_defaults(
        vector_store=PineconeVectorStore(pinecone_index=pinecone_index)
    )

    print("📥 Indexing filtered documents")
    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
    )

    # Get allowed doc_ids
    allowed_ids = [doc.id_ for doc in documents]

    retriever = VectorIndexRetriever(index=index, similarity_top_k=5,doc_ids=allowed_ids)
    reranker = LLMRerank(choice_batch_size=3, llm=llm, top_n=1)

    return RetrieverQueryEngine.from_args(
        retriever=retriever,
        node_postprocessors=[reranker],
        llm=None
    )