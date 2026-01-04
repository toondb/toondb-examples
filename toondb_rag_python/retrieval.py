"""
ToonDB RAG System - Retrieval Strategies
"""
from typing import List
from dataclasses import dataclass
import numpy as np

from vector_store import ToonDBVectorStore, SearchResult
from embeddings import AzureEmbeddings


class BasicRetriever:
    """Simple top-k retrieval"""
    
    def __init__(self, vector_store: ToonDBVectorStore, embedder: AzureEmbeddings):
        self.vector_store = vector_store
        self.embedder = embedder
    
    def retrieve(self, query: str, top_k: int = 5) -> List[SearchResult]:
        """Retrieve top-k most similar chunks"""
        query_embedding = self.embedder.embed_query(query)
        return self.vector_store.search(query_embedding, top_k)


class ThresholdRetriever:
    """Retrieval with relevance threshold"""
    
    def __init__(
        self, 
        vector_store: ToonDBVectorStore, 
        embedder: AzureEmbeddings,
        min_score: float = 0.5
    ):
        self.vector_store = vector_store
        self.embedder = embedder
        self.min_score = min_score
    
    def retrieve(self, query: str, top_k: int = 5) -> List[SearchResult]:
        """Retrieve chunks above score threshold"""
        query_embedding = self.embedder.embed_query(query)
        results = self.vector_store.search(query_embedding, top_k * 2)
        
        # Filter by threshold
        filtered = [r for r in results if r.score >= self.min_score]
        return filtered[:top_k]


class MMRRetriever:
    """Maximal Marginal Relevance retrieval for diversity"""
    
    def __init__(
        self, 
        vector_store: ToonDBVectorStore, 
        embedder: AzureEmbeddings,
        lambda_mult: float = 0.5
    ):
        self.vector_store = vector_store
        self.embedder = embedder
        self.lambda_mult = lambda_mult  # Balance between relevance and diversity
    
    def retrieve(self, query: str, top_k: int = 5, fetch_k: int = 20) -> List[SearchResult]:
        """Retrieve diverse set of relevant chunks using MMR"""
        query_embedding = self.embedder.embed_query(query)
        
        # Get initial candidates
        candidates = self.vector_store.search(query_embedding, fetch_k)
        
        if len(candidates) <= top_k:
            return candidates
        
        # Get embeddings for candidates
        candidate_embeddings = []
        for result in candidates:
            # Re-embed the content (in production, store embeddings)
            emb = self.embedder.embed_query(result.chunk.content)
            candidate_embeddings.append(emb)
        
        candidate_embeddings = np.array(candidate_embeddings)
        
        # Normalize
        query_norm = query_embedding / np.linalg.norm(query_embedding)
        candidate_norms = candidate_embeddings / np.linalg.norm(
            candidate_embeddings, axis=1, keepdims=True
        )
        
        # MMR selection
        selected_indices = []
        remaining_indices = list(range(len(candidates)))
        
        for _ in range(top_k):
            if not remaining_indices:
                break
            
            best_score = -float('inf')
            best_idx = -1
            
            for idx in remaining_indices:
                # Relevance to query
                relevance = np.dot(query_norm, candidate_norms[idx])
                
                # Diversity from selected
                if selected_indices:
                    selected_embs = candidate_norms[selected_indices]
                    max_similarity = np.max(np.dot(selected_embs, candidate_norms[idx]))
                else:
                    max_similarity = 0
                
                # MMR score
                mmr_score = self.lambda_mult * relevance - (1 - self.lambda_mult) * max_similarity
                
                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = idx
            
            if best_idx >= 0:
                selected_indices.append(best_idx)
                remaining_indices.remove(best_idx)
        
        return [candidates[i] for i in selected_indices]


def get_retriever(
    vector_store: ToonDBVectorStore,
    embedder: AzureEmbeddings,
    strategy: str = "basic",
    **kwargs
):
    """Factory function to get retriever by strategy name"""
    strategies = {
        "basic": BasicRetriever,
        "threshold": ThresholdRetriever,
        "mmr": MMRRetriever
    }
    
    if strategy not in strategies:
        raise ValueError(f"Unknown retrieval strategy: {strategy}")
    
    return strategies[strategy](vector_store, embedder, **kwargs)
