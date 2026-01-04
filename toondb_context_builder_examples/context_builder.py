"""
ToonDB Context Query Builder
Demonstrates priority-based context assembly under token budgets
"""
import time
from typing import List, Optional, Tuple
from dataclasses import dataclass
import tiktoken


@dataclass
class ContextComponent:
    """A component of the final context"""
    name: str
    content: str
    priority: int  # Lower = higher priority (0 = must include)
    tokens: int


class ContextQueryBuilder:
    """
    Assembles LLM context under a token budget with priority-based truncation.
    
    Priority Levels:
    0 = Critical (system message, current query) - always included
    1 = High (most recent messages) - included if space allows
    2 = Medium (older history, top retrieval results) - included if space allows
    3 = Low (older retrieval, peripheral context) - only if abundant space
    """
    
    def __init__(self, model: str = "gpt-4", token_budget: int = 4000):
        """
        Initialize the context builder
        
        Args:
            model: Model name for tokenization (gpt-4, gpt-3.5-turbo, etc.)
            token_budget: Maximum tokens for assembled context
        """
        self.token_budget = token_budget
        try:
            self.tokenizer = tiktoken.encoding_for_model(model)
        except:
            # Fallback to cl100k_base (GPT-4 encoding)
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        return len(self.tokenizer.encode(text))
    
    def build_context(
        self,
        system_message: str,
        user_query: str,
        conversation_history: List[Tuple[str, str]] = None,  # [(role, content), ...]
        retrieved_context: List[str] = None,
        metadata: Optional[str] = None
    ) -> Tuple[str, dict]:
        """
        Build context from components under token budget.
        
        Args:
            system_message: System instructions (priority 0)
            user_query: Current user input (priority 0)
            conversation_history: List of (role, content) tuples
            retrieved_context: List of retrieved documents/facts
            metadata: Optional metadata (priority 3)
            
        Returns:
            (assembled_context, stats_dict)
        """
        start_time = time.time()
        
        components = []
        
        # Priority 0: System and current query (always included)
        components.append(ContextComponent(
            name="system",
            content=system_message,
            priority=0,
            tokens=self.count_tokens(system_message)
        ))
        
        components.append(ContextComponent(
            name="user_query",
            content=f"User: {user_query}",
            priority=0,
            tokens=self.count_tokens(user_query) + 3  # "User: " prefix
        ))
        
        # Priority 1 & 2: Conversation history (recent = priority 1, older = priority 2)
        if conversation_history:
            history_count = len(conversation_history)
            for idx, (role, content) in enumerate(conversation_history):
                # Recent messages (last 3) get priority 1, others get priority 2
                priority = 1 if idx >= (history_count - 3) else 2
                formatted = f"{role}: {content}"
                
                components.append(ContextComponent(
                    name=f"history_{idx}",
                    content=formatted,
                    priority=priority,
                    tokens=self.count_tokens(formatted)
                ))
        
        # Priority 2 & 3: Retrieved context (first 3 = priority 2, rest = priority 3)
        if retrieved_context:
            for idx, doc in enumerate(retrieved_context):
                priority = 2 if idx < 3 else 3
                components.append(ContextComponent(
                    name=f"retrieval_{idx}",
                    content=doc,
                    priority=priority,
                    tokens=self.count_tokens(doc)
                ))
        
        # Priority 3: Metadata
        if metadata:
            components.append(ContextComponent(
                name="metadata",
                content=metadata,
                priority=3,
                tokens=self.count_tokens(metadata)
            ))
        
        # Sort by priority, then by order (to maintain coherence within priority levels)
        components.sort(key=lambda c: (c.priority, c.name))
        
        # Assemble under budget
        selected = []
        total_tokens = 0
        
        for comp in components:
            if total_tokens + comp.tokens <= self.token_budget:
                selected.append(comp)
                total_tokens += comp.tokens
            else:
                # Try to fit partial content if priority 0 (critical)
                if comp.priority == 0:
                    # Must fit somehow - truncate if needed
                    remaining = self.token_budget - total_tokens
                    if remaining > 50:  # Minimum useful tokens
                        truncated = self._truncate_to_tokens(comp.content, remaining)
                        selected.append(ContextComponent(
                            name=comp.name,
                            content=truncated,
                            priority=comp.priority,
                            tokens=remaining
                        ))
                        total_tokens += remaining
                break
        
        # Assemble final context
        final_context = self._format_context(selected)
        
        # Stats
        stats = {
            "total_tokens": total_tokens,
            "budget": self.token_budget,
            "utilization": (total_tokens / self.token_budget) * 100,
            "components_included": len(selected),
            "components_total": len(components),
            "latency_ms": (time.time() - start_time) * 1000
        }
        
        return final_context, stats
    
    def _truncate_to_tokens(self, text: str, max_tokens: int) -> str:
        """Truncate text to fit within token budget"""
        tokens = self.tokenizer.encode(text)
        if len(tokens) <= max_tokens:
            return text
        truncated_tokens = tokens[:max_tokens]
        return self.tokenizer.decode(truncated_tokens) + "..."
    
    def _format_context(self, components: List[ContextComponent]) -> str:
        """Format selected components into final context"""
        parts = []
        
        # Group by type for better formatting
        system_parts = [c for c in components if c.name == "system"]
        history_parts = [c for c in components if c.name.startswith("history_")]
        retrieval_parts = [c for c in components if c.name.startswith("retrieval_")]
        query_parts = [c for c in components if c.name == "user_query"]
        metadata_parts = [c for c in components if c.name == "metadata"]
        
        # System
        for comp in system_parts:
            parts.append(comp.content)
        
        # Retrieved context
        if retrieval_parts:
            parts.append("\n=== Retrieved Context ===")
            for comp in retrieval_parts:
                parts.append(comp.content)
            parts.append("=== End Retrieved Context ===\n")
        
        # Conversation history
        if history_parts:
            parts.append("\n=== Conversation History ===")
            for comp in history_parts:
                parts.append(comp.content)
            parts.append("=== End History ===\n")
        
        # Metadata
        for comp in metadata_parts:
            parts.append(f"\n[Metadata: {comp.content}]")
        
        # Current query
        for comp in query_parts:
            parts.append(f"\n{comp.content}")
        
        return "\n".join(parts)
