"""
ToonDB RAG System - Configuration
"""
import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()

@dataclass
class AzureOpenAIConfig:
    api_key: str
    endpoint: str
    api_version: str
    chat_deployment: str
    embedding_deployment: str

@dataclass
class ToonDBConfig:
    db_path: str
    vector_path: str

@dataclass
class RAGConfig:
    chunk_size: int
    chunk_overlap: int
    top_k: int
    max_context_length: int

def get_azure_config() -> AzureOpenAIConfig:
    return AzureOpenAIConfig(
        api_key=os.getenv("AZURE_OPENAI_API_KEY", ""),
        endpoint=os.getenv("AZURE_OPENAI_ENDPOINT", ""),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview"),
        chat_deployment=os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT", "gpt-4.1"),
        embedding_deployment=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "embedding")
    )

def get_toondb_config() -> ToonDBConfig:
    return ToonDBConfig(
        db_path=os.getenv("TOONDB_PATH", "./toondb_data"),
        vector_path=os.getenv("TOONDB_VECTOR_PATH", "./toondb_vectors")
    )

def get_rag_config() -> RAGConfig:
    return RAGConfig(
        chunk_size=int(os.getenv("CHUNK_SIZE", "512")),
        chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "50")),
        top_k=int(os.getenv("TOP_K", "5")),
        max_context_length=int(os.getenv("MAX_CONTEXT_LENGTH", "4000"))
    )
