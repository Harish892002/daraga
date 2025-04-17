from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.core.storage import StorageContext
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.postprocessor import LLMRerank
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from pinecone import Pinecone
import os
from dotenv import load_dotenv

load_dotenv()

def initialize_rag_stack(documents):
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    pinecone_env = os.getenv("PINECONE_ENV")
    pinecone_index_name = os.getenv("PINECONE_INDEX", "llama-index-demo")

    # 🧠 LLM setup
    llm = Ollama(model="hermes3:8b")
    embed_model = OllamaEmbedding(model_name="nomic-embed-text")

    # 📦 Pinecone setup
    pc = Pinecone(api_key=pinecone_api_key)

    # ❌ Optional: clear existing index if exists
    if pinecone_index_name in pc.list_indexes().names():
        print(f"⚠️ Deleting existing Pinecone index '{pinecone_index_name}'...")
        pc.delete_index(pinecone_index_name)

    from pinecone import ServerlessSpec
    pc.create_index(
        name=pinecone_index_name,
        dimension=768,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region=pinecone_env)
    )

    index = pc.Index(pinecone_index_name)
    vector_store = PineconeVectorStore(pinecone_index=index)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # 🧱 Build index from docs
    rag_index = VectorStoreIndex.from_documents(
        documents, storage_context=storage_context, llm=llm, embed_model=embed_model
    )

    # 🔄 Enhanced retriever with reranker
    retriever = VectorIndexRetriever(index=rag_index, similarity_top_k=5)
    reranker = LLMRerank(choice_batch_size=3, llm=llm)

    query_engine = RetrieverQueryEngine.from_args(
    retriever=retriever,
    node_postprocessors=[reranker],
    llm=llm  # 👈 This is the key fix!
)

    return query_engine