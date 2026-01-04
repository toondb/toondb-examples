"""
ToonDB RAG System - Vector Store using ToonDB
"""
from typing import List, Optional
from dataclasses import dataclass
import json
import numpy as np

from toondb import Database

from documents import Chunk
from config import get_toondb_config


@dataclass
class SearchResult:
    """Search result from vector store"""
    chunk: Chunk
    score: float


class ToonDBVectorStore:
    """
    Vector Store implementation using ToonDB
    
    Uses ToonDB's key-value store for metadata and HNSW index for vectors.
    
    Key structure:
    - chunks/{chunk_id} -> chunk content and metadata (JSON)
    - vectors/{chunk_id} -> embedding vector (stored as bytes)
    """
    
    def __init__(self, db_path: str = None):
        config = get_toondb_config()
        self.db_path = db_path or config.db_path
        self._db = None
        self._chunks_cache = {}  # In-memory cache for fast lookup
        self._vectors_cache = {}  # In-memory cache for vectors
    
    @property
    def db(self) -> Database:
        """Lazy database connection"""
        if self._db is None:
            self._db = Database.open(self.db_path)
        return self._db
    
    def upsert(self, chunks: List[Chunk], embeddings: np.ndarray):
        """Insert or update chunks with their embeddings"""
        assert len(chunks) == len(embeddings), "Chunks and embeddings must have same length"
        
        for i, chunk in enumerate(chunks):
            chunk_id = chunk.id
            
            # Store chunk metadata
            chunk_data = {
                "content": chunk.content,
                "metadata": chunk.metadata,
                "start_index": chunk.start_index,
                "end_index": chunk.end_index
            }
            self.db.put(
                f"chunks/{chunk_id}".encode(),
                json.dumps(chunk_data).encode()
            )
            
            # Store embedding as bytes
            embedding_bytes = embeddings[i].astype(np.float32).tobytes()
            self.db.put(
                f"vectors/{chunk_id}".encode(),
                embedding_bytes
            )
            
            # Update cache
            self._chunks_cache[chunk_id] = chunk
            self._vectors_cache[chunk_id] = embeddings[i]
        
        print(f"✅ Upserted {len(chunks)} chunks to ToonDB")
    
    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[SearchResult]:
        """
        Search for similar chunks using cosine similarity
        
        This is a brute-force search - for production, use ToonDB's HNSW index
        """
        # Load all vectors if cache is empty
        if not self._vectors_cache:
            self._load_all()
        
        if not self._vectors_cache:
            return []
        
        # Calculate cosine similarities
        query_norm = query_embedding / np.linalg.norm(query_embedding)
        
        scores = []
        for chunk_id, vector in self._vectors_cache.items():
            vector_norm = vector / np.linalg.norm(vector)
            similarity = np.dot(query_norm, vector_norm)
            scores.append((chunk_id, similarity))
        
        # Sort by similarity (descending)
        scores.sort(key=lambda x: x[1], reverse=True)
        
        # Return top-k results
        results = []
        for chunk_id, score in scores[:top_k]:
            chunk = self._chunks_cache.get(chunk_id)
            if chunk:
                results.append(SearchResult(chunk=chunk, score=float(score)))
        
        return results
    
    def _load_all(self):
        """Load all chunks and vectors from database"""
        # Scan for chunks
        try:
            chunk_results = self.db.scan_prefix("chunks/")
            for kv in chunk_results:
                key = kv.key.decode() if isinstance(kv.key, bytes) else kv.key
                chunk_id = key.replace("chunks/", "")
                
                value = kv.value.decode() if isinstance(kv.value, bytes) else kv.value
                data = json.loads(value)
                
                chunk = Chunk(
                    content=data["content"],
                    metadata=data["metadata"],
                    start_index=data["start_index"],
                    end_index=data["end_index"],
                    id=chunk_id
                )
                self._chunks_cache[chunk_id] = chunk
            
            # Scan for vectors
            vector_results = self.db.scan_prefix("vectors/")
            for kv in vector_results:
                key = kv.key.decode() if isinstance(kv.key, bytes) else kv.key
                chunk_id = key.replace("vectors/", "")
                
                value = kv.value if isinstance(kv.value, bytes) else kv.value.encode()
                vector = np.frombuffer(value, dtype=np.float32)
                self._vectors_cache[chunk_id] = vector
                
        except Exception as e:
            print(f"Warning: Could not load from database: {e}")
    
    def delete(self, chunk_ids: List[str]):
        """Delete chunks by ID"""
        for chunk_id in chunk_ids:
            try:
                self.db.delete(f"chunks/{chunk_id}".encode())
                self.db.delete(f"vectors/{chunk_id}".encode())
                self._chunks_cache.pop(chunk_id, None)
                self._vectors_cache.pop(chunk_id, None)
            except Exception as e:
                print(f"Warning: Could not delete {chunk_id}: {e}")
    
    def clear(self):
        """Clear all data"""
        chunk_ids = list(self._chunks_cache.keys())
        self.delete(chunk_ids)
        self._chunks_cache.clear()
        self._vectors_cache.clear()
    
    def count(self) -> int:
        """Get number of stored chunks"""
        if not self._chunks_cache:
            self._load_all()
        return len(self._chunks_cache)
    
    def close(self):
        """Close database connection"""
        if self._db:
            self._db.close()
            self._db = None
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
