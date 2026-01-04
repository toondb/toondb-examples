"""
ToonDB Agent Memory System - Configuration
"""
import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()


@dataclass
class AzureOpenAIConfig:
    """Azure OpenAI configuration"""
    api_key: str
    endpoint: str
    api_version: str
    chat_deployment: str
    embedding_deployment: str


@dataclass
class ToonDBConfig:
    """ToonDB configuration"""
    db_path: str


@dataclass
class AgentConfig:
    """Agent behavior configuration"""
    max_context_tokens: int = 4000
    memory_window_hours: int = 24
    top_k_memories: int = 10
    enable_metrics: bool = True


def get_azure_config() -> AzureOpenAIConfig:
    """Load Azure OpenAI configuration from environment"""
    return AzureOpenAIConfig(
        api_key=os.getenv("AZURE_OPENAI_API_KEY", ""),
        endpoint=os.getenv("AZURE_OPENAI_ENDPOINT", ""),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview"),
        chat_deployment=os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT", "gpt-4"),
        embedding_deployment=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-3-small")
    )


def get_toondb_config() -> ToonDBConfig:
    """Load ToonDB configuration from environment"""
    return ToonDBConfig(
        db_path=os.getenv("TOONDB_PATH", "./agent_memory_db")
    )


def get_agent_config() -> AgentConfig:
    """Load agent configuration from environment"""
    return AgentConfig(
        max_context_tokens=int(os.getenv("MAX_CONTEXT_TOKENS", "4000")),
        memory_window_hours=int(os.getenv("MEMORY_WINDOW_HOURS", "24")),
        top_k_memories=int(os.getenv("TOP_K_MEMORIES", "10")),
        enable_metrics=os.getenv("ENABLE_METRICS", "true").lower() == "true"
    )
