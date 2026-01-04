"""
ToonDB RAG System - Embeddings using Azure OpenAI
"""
from typing import List
import numpy as np
from openai import AzureOpenAI

from config import get_azure_config


class AzureEmbeddings:
    """Azure OpenAI Embeddings"""
    
    def __init__(self):
        config = get_azure_config()
        self.client = AzureOpenAI(
            api_key=config.api_key,
            api_version=config.api_version,
            azure_endpoint=config.endpoint
        )
        self.deployment = config.embedding_deployment
        self._dimension = 1536  # text-embedding-ada-002 dimension
    
    def embed(self, texts: List[str]) -> np.ndarray:
        """Embed a list of texts"""
        if not texts:
            return np.array([])
        
        # Handle single text
        if isinstance(texts, str):
            texts = [texts]
        
        # Azure OpenAI has a limit on batch size
        batch_size = 16
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            response = self.client.embeddings.create(
                input=batch,
                model=self.deployment
            )
            batch_embeddings = [e.embedding for e in response.data]
            all_embeddings.extend(batch_embeddings)
        
        return np.array(all_embeddings)
    
    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single query"""
        return self.embed([query])[0]
    
    @property
    def dimension(self) -> int:
        return self._dimension


class LocalEmbeddings:
    """Local embeddings using sentence-transformers (fallback)"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)
    
    def embed(self, texts: List[str]) -> np.ndarray:
        """Embed a list of texts"""
        if isinstance(texts, str):
            texts = [texts]
        return self.model.encode(texts, normalize_embeddings=True)
    
    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single query"""
        return self.embed([query])[0]
    
    @property
    def dimension(self) -> int:
        return self.model.get_sentence_embedding_dimension()


class MockEmbeddings:
    """Mock embeddings for testing/demo without API key"""
    
    def __init__(self, dimension: int = 1536):
        self._dimension = dimension
    
    def embed(self, texts: List[str]) -> np.ndarray:
        """Generate random embeddings"""
        if isinstance(texts, str):
            texts = [texts]
        # Deterministic random based on text length to be consistent
        embeddings = []
        for text in texts:
            np.random.seed(len(text))
            embeddings.append(np.random.rand(self._dimension))
        return np.array(embeddings)
    
    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single query"""
        return self.embed([query])[0]
    
    @property
    def dimension(self) -> int:
        return self._dimension


def get_embeddings(use_azure: bool = True, use_mock: bool = False):
    """Factory function to get embeddings model"""
    if use_mock:
        return MockEmbeddings()
    elif use_azure:
        return AzureEmbeddings()
    else:
        return LocalEmbeddings()

