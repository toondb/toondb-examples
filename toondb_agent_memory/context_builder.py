"""
ToonDB Agent Memory System - Context Builder
Builds conversation context from retrieved memories
"""
import time
from typing import List
from dataclasses import dataclass

from memory_manager import Memory, MemoryManager
from config import get_agent_config


@dataclass
class ContextResult:
    """Result of context assembly"""
    context: str
    memories_used: List[Memory]
    total_tokens: int
    assemble_latency_ms: float
    retrieval_latency_ms: float


class ContextBuilder:
    """
    Builds conversation context from retrieved memories
    
    Features:
    - Vector search for semantically relevant memories
    - Hard filter on timestamp (last N hours)
    - Assembles context targeting max_tokens limit
    - Formats memories for LLM consumption
    """
    
    def __init__(self, memory_manager: MemoryManager):
        self.memory_manager = memory_manager
        self.config = get_agent_config()
    
    def build_context(
        self,
        query: str,
        session_id: str,
        max_tokens: int = None
    ) -> ContextResult:
        """
        Build context from relevant memories
        
        Process:
        1. Embed query
        2. Vector search with timestamp filter
        3. Assemble top-k memories into context
        4. Return formatted context + metrics
        
        Args:
            query: Current user query/message
            session_id: Session identifier
            max_tokens: Maximum tokens for context (default from config)
            
        Returns:
            ContextResult with formatted context and metrics
        """
        assemble_start = time.time()
        
        max_tokens = max_tokens or self.config.max_context_tokens
        
        # Search for relevant memories (includes retrieval latency)
        results, retrieval_latency_ms = self.memory_manager.search_memories(
            session_id=session_id,
            query=query,
            top_k=self.config.top_k_memories,
            hours=self.config.memory_window_hours
        )
        
        if not results:
            return ContextResult(
                context="No previous conversation history available.",
                memories_used=[],
                total_tokens=0,
                assemble_latency_ms=(time.time() - assemble_start) * 1000,
                retrieval_latency_ms=retrieval_latency_ms
            )
        
        # Assemble context from memories
        context_parts = ["=== Relevant Past Conversation Memories ===\n"]
        memories_used = []
        total_tokens = 0
        
        for memory, similarity in results:
            # Check if adding this memory would exceed token limit
            if total_tokens + memory.token_count > max_tokens:
                break
            
            # Format memory
            timestamp_str = self._format_timestamp(memory.timestamp)
            context_parts.append(
                f"[Turn {memory.turn} - {memory.role.upper()} - {timestamp_str} - Relevance: {similarity:.3f}]\n"
                f"{memory.content}\n"
            )
            
            memories_used.append(memory)
            total_tokens += memory.token_count
        
        context_parts.append("\n=== End of Past Memories ===")
        context = "\n".join(context_parts)
        
        assemble_latency_ms = (time.time() - assemble_start) * 1000
        
        return ContextResult(
            context=context,
            memories_used=memories_used,
            total_tokens=total_tokens,
            assemble_latency_ms=assemble_latency_ms,
            retrieval_latency_ms=retrieval_latency_ms
        )
    
    def _format_timestamp(self, timestamp: float) -> str:
        """Format timestamp for display"""
        from datetime import datetime
        dt = datetime.fromtimestamp(timestamp)
        now = datetime.now()
        
        # Calculate time difference
        diff_seconds = (now - dt).total_seconds()
        
        if diff_seconds < 60:
            return "just now"
        elif diff_seconds < 3600:
            minutes = int(diff_seconds / 60)
            return f"{minutes}m ago"
        elif diff_seconds < 86400:
            hours = int(diff_seconds / 3600)
            return f"{hours}h ago"
        else:
            days = int(diff_seconds / 86400)
            return f"{days}d ago"
