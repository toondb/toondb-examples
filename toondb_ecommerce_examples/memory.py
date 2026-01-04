import json
import time
import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Dict
from openai import AzureOpenAI
from toondb import Database
from .config import get_toondb_config, get_azure_config

@dataclass
class Episode:
    """A fact or episode in memory"""
    id: str
    content: str
    source: str
    timestamp: float
    embedding: Optional[np.ndarray] = None
    
    def to_dict(self):
        return {
            "id": self.id,
            "content": self.content,
            "source": self.source,
            "timestamp": self.timestamp
        }
        
    @staticmethod
    def from_dict(data: dict, embedding: Optional[np.ndarray] = None):
        return Episode(
            id=data["id"],
            content=data["content"],
            source=data["source"],
            timestamp=data["timestamp"],
            embedding=embedding
        )

class ToonDBMemory:
    """
    ToonDB-backed long-term memory for the agent.
    Stores "episodes" (facts/interactions) and supports semantic search.
    """
    
    def __init__(self, db_path: str = None):
        self.config = get_toondb_config()
        self.azure_config = get_azure_config()
        self.db_path = db_path or self.config.db_path
        self._db = None
        self._embedder = None
        self._episodes_cache = {}
        self._vectors_cache = {}
        
    @property
    def db(self) -> Database:
        if self._db is None:
            self._db = Database.open(self.db_path)
        return self._db
        
    @property
    def embedder(self) -> AzureOpenAI:
        if self._embedder is None:
            self._embedder = AzureOpenAI(
                api_key=self.azure_config.api_key,
                api_version=self.azure_config.api_version,
                azure_endpoint=self.azure_config.endpoint
            )
        return self._embedder

    def add_episode(self, content: str, source: str = "dialogue", timestamp: float = None):
        """Add a new memory episode"""
        episode_id = f"{uuid.uuid4().hex}"
        ts = timestamp or time.time()
        
        # Get embedding
        resp = self.embedder.embeddings.create(
            input=content,
            model=self.azure_config.embedding_deployment
        )
        embedding = np.array(resp.data[0].embedding, dtype=np.float32)
        
        episode = Episode(
            id=episode_id,
            content=content,
            source=source,
            timestamp=ts,
            embedding=embedding
        )
        
        # Store metadata
        self.db.put(
            f"episodes/{episode_id}".encode(),
            json.dumps(episode.to_dict()).encode()
        )
        
        # Store vector
        self.db.put(
            f"vectors/{episode_id}".encode(),
            embedding.tobytes()
        )
        
        # Update cache
        self._episodes_cache[episode_id] = episode
        self._vectors_cache[episode_id] = embedding
        
    def search(self, query: str, top_k: int = 5) -> str:
        """
        Search memory and return TOON formatted output.
        TOON output is a compact context format.
        """
        # Load cache if empty
        if not self._vectors_cache:
            self._load_all()
            
        if not self._vectors_cache:
            return "NO_MEMORY_FOUND"
            
        # Get query embedding
        resp = self.embedder.embeddings.create(
            input=query,
            model=self.azure_config.embedding_deployment
        )
        query_vec = np.array(resp.data[0].embedding, dtype=np.float32)
        query_norm = query_vec / np.linalg.norm(query_vec)
        
        # Brute force cosine similarity
        scores = []
        for eid, vec in self._vectors_cache.items():
            vec_norm = vec / np.linalg.norm(vec)
            sim = np.dot(query_norm, vec_norm)
            scores.append((eid, sim))
            
        scores.sort(key=lambda x: x[1], reverse=True)
        
        # Format as TOON output using native helper
        records = []
        for eid, score in scores[:top_k]:
            ep = self._episodes_cache[eid]
            records.append({
                "source": ep.source,
                "relevance": round(score, 2),
                "content": ep.content
            })
            
        if not records:
            return "NO_MEMORY_FOUND"
            
        return Database.to_toon("memory", records, ["source", "relevance", "content"])

    def _load_all(self):
        try:
            # Load episodes
            for kv in self.db.scan_prefix("episodes/".encode()):
                key = kv.key.decode()
                eid = key.split("/")[-1]
                data = json.loads(kv.value.decode())
                self._episodes_cache[eid] = Episode.from_dict(data)
                
            # Load vectors
            for kv in self.db.scan_prefix("vectors/".encode()):
                key = kv.key.decode()
                eid = key.split("/")[-1]
                vec = np.frombuffer(kv.value, dtype=np.float32)
                self._vectors_cache[eid] = vec
                if eid in self._episodes_cache:
                    self._episodes_cache[eid].embedding = vec
        except Exception:
            pass
            
    def close(self):
        if self._db:
            self._db.close()

import uuid
