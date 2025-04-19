import os
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.embeddings.nomic import NomicEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.core import StorageContext,VectorStoreIndex, Settings
from llama_index.core.query_engine import RetrieverQueryEngine
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

load_dotenv()

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
PINECONE_ENV = os.environ.get("PINECONE_ENV", "us-east-1")
DEFAULT_DIMENSIONS = 1024

# Initialize Pinecone client
pc = Pinecone(api_key=PINECONE_API_KEY)

# Set global settings for embedding and LLM
embed_model = HuggingFaceEmbedding(
    model_name="intfloat/multilingual-e5-large",
    embed_batch_size=32  # optional tuning
)
Settings.embed_model = embed_model
Settings.llm = Ollama(model="hermes3:8b")

def initialize_rag_stack(documents, index_name: str):
    if not documents:
        raise ValueError("No documents provided for indexing.")
    if not index_name:
        raise ValueError("Index name must be provided.")

    print(f"📦 Checking Pinecone index: {index_name}")
    existing_indexes = pc.list_indexes().names()

    if index_name not in existing_indexes:
        print(f"✅ Creating Pinecone index: {index_name}")
        pc.create_index(
            name=index_name,
            dimension=DEFAULT_DIMENSIONS,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region=PINECONE_ENV),
        )
    else:
        print(f"📂 Pinecone index '{index_name}' already exists.")

    pinecone_index = pc.Index(index_name)
    vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    print("📥 Building vector index with documents...")
    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
    )

    query_engine = RetrieverQueryEngine.from_args(
        retriever=index.as_retriever(similarity_top_k=5)
    )

    print("✅ RAG stack initialized.")
    return query_engine