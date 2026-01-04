"""
ToonDB Agent Memory System - Memory Manager
Handles hierarchical storage of observations with timestamps
"""
import time
import json
from typing import List, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import numpy as np

from toondb import Database
from openai import AzureOpenAI

from config import get_azure_config, get_toondb_config


@dataclass
class Memory:
    """A single memory/observation"""
    session_id: str
    turn: int
    content: str
    role: str  # 'user' or 'assistant'
    timestamp: float
    token_count: int
    embedding: Optional[np.ndarray] = None
    
    def to_dict(self) -> dict:
        """Convert to dictionary for storage"""
        return {
            "session_id": self.session_id,
            "turn": self.turn,
            "content": self.content,
            "role": self.role,
            "timestamp": self.timestamp,
            "token_count": self.token_count
        }
    
    @staticmethod
    def from_dict(data: dict, embedding: Optional[np.ndarray] = None) -> 'Memory':
        """Create from dictionary"""
        return Memory(
            session_id=data["session_id"],
            turn=data["turn"],
            content=data["content"],
            role=data["role"],
            timestamp=data["timestamp"],
            token_count=data["token_count"],
            embedding=embedding
        )


class MemoryManager:
    """
    Manages hierarchical memory storage in ToonDB
    
    Path structure: session.{session_id}.observations.turn_{turn_number}
    Each observation stored with metadata: timestamp, role, token_count
    """
    
    def __init__(self):
        self.toondb_config = get_toondb_config()
        self.azure_config = get_azure_config()
        self._db = None
        self._embedder = None
        
    @property
    def db(self) -> Database:
        """Lazy database connection"""
        if self._db is None:
            self._db = Database.open(self.toondb_config.db_path)
        return self._db
    
    @property
    def embedder(self) -> AzureOpenAI:
        """Lazy Azure OpenAI client"""
        if self._embedder is None:
            self._embedder = AzureOpenAI(
                api_key=self.azure_config.api_key,
                api_version=self.azure_config.api_version,
                azure_endpoint=self.azure_config.endpoint
            )
        return self._embedder
    
    def _get_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for text using Azure OpenAI"""
        response = self.embedder.embeddings.create(
            input=text,
            model=self.azure_config.embedding_deployment
        )
        return np.array(response.data[0].embedding, dtype=np.float32)
    
    def _count_tokens(self, text: str) -> int:
        """Estimate token count (rough approximation)"""
        # Simple approximation: ~4 chars per token
        return len(text) // 4
    
    def _get_memory_path(self, session_id: str, turn: int) -> str:
        """Generate hierarchical path for memory"""
        return f"session.{session_id}.observations.turn_{turn}"
    
    def store_observation(
        self, 
        session_id: str, 
        turn: int, 
        content: str, 
        role: str
    ) -> tuple[Memory, float]:
        """
        Store observation in hierarchical path with embedding
        
        Returns:
            (Memory object, write_latency_ms)
        """
        start_time = time.time()
        
        # Create memory object
        memory = Memory(
            session_id=session_id,
            turn=turn,
            content=content,
            role=role,
            timestamp=time.time(),
            token_count=self._count_tokens(content)
        )
        
        # Generate embedding
        embedding = self._get_embedding(content)
        memory.embedding = embedding
        
        # Store metadata
        path = self._get_memory_path(session_id, turn)
        metadata_key = f"{path}.metadata"
        self.db.put(
            metadata_key.encode(),
            json.dumps(memory.to_dict()).encode()
        )
        
        # Store embedding
        embedding_key = f"{path}.embedding"
        self.db.put(
            embedding_key.encode(),
            embedding.tobytes()
        )
        
        write_latency_ms = (time.time() - start_time) * 1000
        return memory, write_latency_ms
    
    def get_recent_memories(
        self, 
        session_id: str, 
        hours: int = 24
    ) -> List[Memory]:
        """
        Retrieve memories from the last N hours
        
        Args:
            session_id: Session identifier
            hours: Time window in hours
            
        Returns:
            List of Memory objects within time window
        """
        cutoff_time = time.time() - (hours * 3600)
        memories = []
        
        # Scan for all observations in this session
        prefix = f"session.{session_id}.observations."
        
        try:
            results = self.db.scan_prefix(prefix.encode())
            
            # Group by turn
            metadata_by_turn = {}
            embeddings_by_turn = {}
            
            for key, value in results:
                key_str = key.decode() if isinstance(key, bytes) else key
                
                if ".metadata" in key_str:
                    turn_str = key_str.split("turn_")[1].split(".")[0]
                    value_str = value.decode() if isinstance(value, bytes) else value
                    metadata_by_turn[turn_str] = json.loads(value_str)
                    
                elif ".embedding" in key_str:
                    turn_str = key_str.split("turn_")[1].split(".")[0]
                    value_bytes = value if isinstance(value, bytes) else value.encode()
                    embeddings_by_turn[turn_str] = np.frombuffer(value_bytes, dtype=np.float32)
            
            # Reconstruct memories
            for turn_str, metadata in metadata_by_turn.items():
                if metadata["timestamp"] >= cutoff_time:
                    embedding = embeddings_by_turn.get(turn_str)
                    memory = Memory.from_dict(metadata, embedding)
                    memories.append(memory)
            
            # Sort by turn number
            memories.sort(key=lambda m: m.turn)
            
        except Exception as e:
            print(f"Warning: Could not retrieve memories: {e}")
        
        return memories
    
    def search_memories(
        self,
        session_id: str,
        query: str,
        top_k: int = 10,
        hours: int = 24
    ) -> tuple[List[tuple[Memory, float]], float]:
        """
        Search for semantically similar memories with timestamp filter
        
        Args:
            session_id: Session identifier
            query: Query text to search for
            top_k: Number of results to return
            hours: Time window filter in hours
            
        Returns:
            (List of (Memory, similarity_score) tuples, search_latency_ms)
        """
        start_time = time.time()
        
        # Get recent memories
        memories = self.get_recent_memories(session_id, hours)
        
        if not memories:
            return [], (time.time() - start_time) * 1000
        
        # Generate query embedding
        query_embedding = self._get_embedding(query)
        query_norm = query_embedding / np.linalg.norm(query_embedding)
        
        # Calculate similarities
        similarities = []
        for memory in memories:
            if memory.embedding is not None:
                mem_norm = memory.embedding / np.linalg.norm(memory.embedding)
                similarity = float(np.dot(query_norm, mem_norm))
                similarities.append((memory, similarity))
        
        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        search_latency_ms = (time.time() - start_time) * 1000
        return similarities[:top_k], search_latency_ms
    
    def close(self):
        """Close database connection"""
        if self._db:
            self._db.close()
            self._db = None
