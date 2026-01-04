"""
ToonDB Agent Memory System - Main Agent
Real LLM-powered agent with memory capabilities
"""
import time
from typing import Optional
from dataclasses import dataclass
from openai import AzureOpenAI

from memory_manager import MemoryManager
from context_builder import ContextBuilder
from performance_tracker import PerformanceTracker
from config import get_azure_config, get_agent_config


@dataclass
class AgentResponse:
    """Response from agent including metrics"""
    message: str
    context_used: str
    memories_count: int
    write_latency_ms: float
    read_latency_ms: float
    assemble_latency_ms: float
    llm_latency_ms: float
    total_latency_ms: float


class Agent:
    """
    Memory-augmented conversational agent
    
    Features:
    - Stores all observations in ToonDB with hierarchical paths
    - Retrieves relevant memories using vector search
    - Assembles context with timestamp filtering
    - Generates responses using Azure OpenAI
    - Tracks performance metrics
    """
    
    def __init__(
        self, 
        session_id: str,
        enable_metrics: bool = True,
        system_prompt: Optional[str] = None
    ):
        self.session_id = session_id
        self.turn_counter = 0
        
        # Components
        self.memory_manager = MemoryManager()
        self.context_builder = ContextBuilder(self.memory_manager)
        
        # LLM
        azure_config = get_azure_config()
        self.llm = AzureOpenAI(
            api_key=azure_config.api_key,
            api_version=azure_config.api_version,
            azure_endpoint=azure_config.endpoint
        )
        self.chat_deployment = azure_config.chat_deployment
        
        # Performance tracking
        self.enable_metrics = enable_metrics
        if enable_metrics:
            self.tracker = PerformanceTracker()
        
        # System prompt
        self.system_prompt = system_prompt or self._default_system_prompt()
    
    def _default_system_prompt(self) -> str:
        """Default system prompt for the agent"""
        return """You are a helpful AI assistant with access to conversation memory.

When you receive relevant past memories, use them to:
- Maintain context across conversations
- Reference previous discussions
- Provide consistent and personalized responses
- Avoid asking for information already provided

The memories show turn numbers, timestamps, and relevance scores. Use this information to understand the conversation flow and prioritize recent or highly relevant information.

Be conversational, helpful, and make good use of the context provided."""
    
    def chat(self, user_message: str) -> AgentResponse:
        """
        Process user message and generate response
        
        Complete cycle:
        1. Store user message as observation
        2. Build context from past memories
        3. Generate LLM response with context
        4. Store assistant response
        5. Return response + performance metrics
        """
        cycle_start = time.time()
        
        # Increment turn counter
        self.turn_counter += 1
        current_turn = self.turn_counter
        
        # Step 1: Store user message
        user_memory, write_latency_1 = self.memory_manager.store_observation(
            session_id=self.session_id,
            turn=current_turn,
            content=user_message,
            role="user"
        )
        
        # Step 2: Build context from memories
        context_result = self.context_builder.build_context(
            query=user_message,
            session_id=self.session_id
        )
        
        read_latency = context_result.retrieval_latency_ms
        assemble_latency = context_result.assemble_latency_ms
        
        # Step 3: Generate LLM response
        llm_start = time.time()
        
        messages = [
            {"role": "system", "content": self.system_prompt}
        ]
        
        # Add context if available
        if context_result.memories_used:
            messages.append({
                "role": "system",
                "content": context_result.context
            })
        
        # Add current user message
        messages.append({"role": "user", "content": user_message})
        
        response = self.llm.chat.completions.create(
            model=self.chat_deployment,
            messages=messages,
            temperature=0.7,
            max_tokens=500
        )
        
        assistant_message = response.choices[0].message.content
        llm_latency = (time.time() - llm_start) * 1000
        
        # Step 4: Store assistant response
        self.turn_counter += 1
        assistant_memory, write_latency_2 = self.memory_manager.store_observation(
            session_id=self.session_id,
            turn=self.turn_counter,
            content=assistant_message,
            role="assistant"
        )
        
        # Calculate total latencies
        write_latency = write_latency_1 + write_latency_2
        total_latency = (time.time() - cycle_start) * 1000
        
        # Record metrics
        if self.enable_metrics:
            self.tracker.record_cycle(
                write_ms=write_latency,
                read_ms=read_latency,
                assemble_ms=assemble_latency,
                llm_ms=llm_latency
            )
        
        return AgentResponse(
            message=assistant_message,
            context_used=context_result.context,
            memories_count=len(context_result.memories_used),
            write_latency_ms=write_latency,
            read_latency_ms=read_latency,
            assemble_latency_ms=assemble_latency,
            llm_latency_ms=llm_latency,
            total_latency_ms=total_latency
        )
    
    def get_performance_report(self):
        """Get performance report"""
        if self.enable_metrics:
            return self.tracker.get_report()
        return None
    
    def close(self):
        """Close connections"""
        self.memory_manager.close()
