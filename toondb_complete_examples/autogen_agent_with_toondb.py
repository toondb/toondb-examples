"""
Real AutoGen Integration with ToonDB Memory
============================================

A complete, working example showing how to integrate ToonDB as the memory layer
for AutoGen multi-agent conversations with persistent storage and retrieval.

Requirements:
    pip install pyautogen toondb-client python-dotenv

Usage:
    python3 autogen_agent_with_toondb.py
    
Features:
    - Multi-agent conversation (Assistant + UserProxy)
    - Automatic message storage in ToonDB
    - Memory retrieval functions for agents
    - Persistent conversation history
    - Session-based isolation
"""

import os
import json
import time
import uuid
from typing import Dict, List, Optional
from dotenv import load_dotenv

# AutoGen imports
from autogen import AssistantAgent, UserProxyAgent, ConversableAgent

# ToonDB
from toondb import Database

# Load environment
load_dotenv()


class ToonDBMemoryStore:
    """
    ToonDB-backed memory store for AutoGen agents
    
    Automatically captures all agent messages and provides
    search/retrieval capabilities for conversation history.
    """
    
    def __init__(self, db_path: str = "./toondb_autogen_data"):
        self.db = Database.open(db_path)
        self.session_id = None
    
    def start_session(self, session_id: Optional[str] = None) -> str:
        """Start a new conversation session"""
        if session_id is None:
            session_id = f"session_{uuid.uuid4().hex[:8]}"
        
        self.session_id = session_id
        self.db.put(
            f"sessions.{session_id}.created_at".encode(),
            str(time.time()).encode()
        )
        
        return session_id
    
    def save_message(
        self, 
        sender: str, 
        recipient: str, 
        content: str,
        message_type: str = "chat"
    ):
        """Save a message from agent conversation"""
        if not self.session_id:
            self.start_session()
        
        # Count messages to get index
        msg_count = 0
        prefix = f"sessions.{self.session_id}.messages."
        for key, _ in self.db.scan_prefix(prefix.encode()):
            if ".content" in key.decode():
                msg_count += 1
        
        msg_idx = msg_count + 1
        path = f"sessions.{self.session_id}.messages.{msg_idx}"
        
        # Store message
        self.db.put(f"{path}.sender".encode(), sender.encode())
        self.db.put(f"{path}.recipient".encode(), recipient.encode())
        self.db.put(f"{path}.content".encode(), content.encode())
        self.db.put(f"{path}.type".encode(), message_type.encode())
        self.db.put(f"{path}.timestamp".encode(), str(time.time()).encode())
    
    def get_conversation_history(self, limit: int = 10) -> List[Dict]:
        """Retrieve recent conversation history"""
        if not self.session_id:
            return []
        
        messages_data = {}
        prefix = f"sessions.{self.session_id}.messages."
        
        for key, value in self.db.scan_prefix(prefix.encode()):
            key_str = key.decode()
            parts = key_str.split(".")
            
            if len(parts) >= 5:
                msg_idx = parts[3]
                field = parts[4]
                
                if msg_idx not in messages_data:
                    messages_data[msg_idx] = {}
                
                messages_data[msg_idx][field] = value.decode()
        
        # Sort and get recent
        sorted_indices = sorted(messages_data.keys(), key=int)
        recent_indices = sorted_indices[-limit:] if len(sorted_indices) > limit else sorted_indices
        
        return [messages_data[idx] for idx in recent_indices]
    
    def search_messages(self, query: str, limit: int = 5) -> List[Dict]:
        """Search through conversation history"""
        if not self.session_id:
            return []
        
        results = []
        query_lower = query.lower()
        history = self.get_conversation_history(limit=100)
        
        for msg in history:
            if query_lower in msg.get("content", "").lower():
                results.append(msg)
                if len(results) >= limit:
                    break
        
        return results
    
    def get_summary(self) -> str:
        """Get a summary of the conversation"""
        history = self.get_conversation_history(limit=10)
        
        if not history:
            return "No conversation history"
        
        summary_parts = [f"Conversation session: {self.session_id}", ""]
        for i, msg in enumerate(history, 1):
            sender = msg.get("sender", "unknown")
            content = msg.get("content", "")[:100]
            summary_parts.append(f"{i}. {sender}: {content}...")
        
        return "\n".join(summary_parts)
    
    def close(self):
        """Close database connection"""
        self.db.close()


# Global memory store
memory = ToonDBMemoryStore()


def message_interceptor(recipient, messages, sender, config):
    """
    Intercept and store all messages in ToonDB
    
    This function is called for every message sent between agents
    """
    # Get the last message
    if messages:
        last_message = messages[-1]
        
        # Extract content
        if isinstance(last_message, dict):
            content = last_message.get("content", str(last_message))
        else:
            content = str(last_message)
        
        # Save to ToonDB
        memory.save_message(
            sender=sender.name if hasattr(sender, 'name') else str(sender),
            recipient=recipient.name if hasattr(recipient, 'name') else str(recipient),
            content=content
        )
    
    # Return None to continue normal processing
    return False, None


def search_memory_function(query: str) -> str:
    """
    Function that agents can call to search conversation history
    
    Args:
        query: Search query string
    
    Returns:
        Formatted search results
    """
    results = memory.search_messages(query, limit=3)
    
    if not results:
        return f"No messages found matching '{query}'"
    
    formatted = []
    for i, msg in enumerate(results, 1):
        sender = msg.get("sender", "unknown")
        content = msg.get("content", "")
        formatted.append(f"{i}. {sender}: {content[:150]}...")
    
    return "Found messages:\n" + "\n".join(formatted)


def get_conversation_summary_function() -> str:
    """
    Function that agents can call to get conversation summary
    
    Returns:
        Summary of recent conversation
    """
    return memory.get_summary()


def create_agents_with_memory():
    """
    Create AutoGen agents with ToonDB memory integration
    
    Returns:
        Tuple of (assistant_agent, user_proxy_agent)
    """
    # Azure OpenAI configuration for AutoGen
    # NOTE: For Azure, the 'model' field is the deployment name
    llm_config = {
        "config_list": [{
            "model": os.getenv("AZURE_OPENAI_DEPLOYMENT"),
            "api_type": "azure",
            "api_key": os.getenv("AZURE_OPENAI_API_KEY"),
            "base_url": os.getenv("AZURE_OPENAI_ENDPOINT"),
            "api_version": os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview"),
        }],
        "temperature": 0,
    }
    
    # Create assistant agent
    assistant = AssistantAgent(
        name="assistant",
        system_message="""You are a helpful AI assistant with access to conversation history.
        
When users ask about what was discussed earlier, use the search_memory or get_summary functions
to retrieve relevant information from past conversations.

Always be helpful and reference past context when relevant.""",
        llm_config=llm_config,
    )
    
    # Create user proxy agent
    user_proxy = UserProxyAgent(
        name="user",
        human_input_mode="TERMINATE",  # Can be "ALWAYS" for interactive
        max_consecutive_auto_reply=10,
        code_execution_config=False,
    )
    
    # Register memory functions with assistant
    assistant.register_function(
        function_map={
            "search_memory": search_memory_function,
            "get_conversation_summary": get_conversation_summary_function,
        }
    )
    
    # Register message interceptor for both agents
    assistant.register_reply(
        ConversableAgent,
        reply_func=message_interceptor,
        position=0,
    )
    
    user_proxy.register_reply(
        ConversableAgent,
        reply_func=message_interceptor,
        position=0,
    )
    
    return assistant, user_proxy


def run_demo_conversation():
    """Run a demo conversation showcasing ToonDB memory"""
    print("="*70)
    print("  AutoGen + ToonDB Memory Demo")
    print("="*70)
    print()
    
    # Start session
    session_id = memory.start_session()
    print(f"Session ID: {session_id}\n")
    
    # Create agents
    assistant, user_proxy = create_agents_with_memory()
    
    # Demo conversations
    conversations = [
        "My name is Alice and I'm working on a machine learning project about image classification.",
        "What have we been discussing so far?",  # Will use memory
        "What was my name again?",  # Will use memory
    ]
    
    for i, user_message in enumerate(conversations, 1):
        print(f"\n{'='*70}")
        print(f"  Conversation Turn {i}")
        print('='*70)
        print(f"\n👤 User: {user_message}\n")
        
        # Initiate chat
        user_proxy.initiate_chat(
            assistant,
            message=user_message,
            clear_history=False,
        )
        
        time.sleep(1)
    
    # Show conversation summary
    print(f"\n{'='*70}")
    print("  Conversation Summary from ToonDB")
    print('='*70)
    print()
    print(memory.get_summary())
    print()
    
    # Show raw message count
    history = memory.get_conversation_history(limit=100)
    print(f"Total messages stored in ToonDB: {len(history)}")
    print()
    
    memory.close()
    print("✅ Demo complete!")


def run_interactive_conversation():
    """Run an interactive conversation with the agents"""
    print("="*70)
    print("  AutoGen + ToonDB Interactive Mode")
    print("="*70)
    print("\nType your messages. The agent has access to conversation history.")
    print("Type 'exit' to quit.\n")
    
    # Start session
    session_id = memory.start_session()
    print(f"Session ID: {session_id}\n")
    
    # Create agents
    assistant, user_proxy = create_agents_with_memory()
    
    # Interactive loop
    while True:
        user_input = input("You: ").strip()
        
        if not user_input:
            continue
        
        if user_input.lower() == 'exit':
            print("\nGoodbye!")
            break
        
        # Chat with assistant
        try:
            user_proxy.initiate_chat(
                assistant,
                message=user_input,
                clear_history=False,
            )
        except Exception as e:
            print(f"\n❌ Error: {e}\n")
        
        print()  # Add spacing
    
    # Show summary
    print(f"\n{'='*70}")
    print("  Session Summary")
    print('='*70)
    print()
    print(memory.get_summary())
    
    memory.close()


def run_multi_agent_collaboration():
    """
    Demo of multiple agents collaborating with shared memory
    
    Shows how ToonDB can store multi-agent conversations
    """
    print("="*70)
    print("  Multi-Agent Collaboration with ToonDB")
    print("="*70)
    print()
    
    session_id = memory.start_session()
    print(f"Session ID: {session_id}\n")
    
    # Azure OpenAI configuration
    llm_config = {
        "config_list": [{
            "model": os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT"),  # Deployment name
            "api_type": "azure",
            "api_key": os.getenv("AZURE_OPENAI_API_KEY"),
            "base_url": os.getenv("AZURE_OPENAI_ENDPOINT"),
            "api_version": os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview"),
        }],
    }
    
    # Create multiple specialized agents
    researcher = AssistantAgent(
        name="researcher",
        system_message="You are a research specialist. Focus on finding facts and data.",
        llm_config=llm_config,
    )
    
    writer = AssistantAgent(
        name="writer",
        system_message="You are a content writer. Create clear, engaging content.",
        llm_config=llm_config,
    )
    
    user = UserProxyAgent(
        name="user",
        human_input_mode="NEVER",
        max_consecutive_auto_reply=3,
        code_execution_config=False,
    )
    
    # Register interceptors
    for agent in [researcher, writer, user]:
        agent.register_reply(
            ConversableAgent,
            reply_func=message_interceptor,
            position=0,
        )
    
    # Task
    task = "Write a short paragraph about the benefits of open source software."
    
    print(f"Task: {task}\n")
    print("Agents collaborating...\n")
    
    # Researcher gathers info
    user.initiate_chat(
        researcher,
        message=f"Research task: {task}",
    )
    
    time.sleep(1)
    
    # Writer creates content
    user.initiate_chat(
        writer,
        message=f"Based on the research, {task}",
    )
    
    # Show all stored messages
    print(f"\n{'='*70}")
    print("  All Messages Stored in ToonDB")
    print('='*70)
    print()
    
    history = memory.get_conversation_history(limit=100)
    for i, msg in enumerate(history, 1):
        sender = msg.get("sender", "unknown")
        recipient = msg.get("recipient", "unknown")
        content = msg.get("content", "")[:100]
        print(f"{i}. {sender} → {recipient}: {content}...")
    
    print(f"\nTotal messages: {len(history)}")
    
    memory.close()
    print("\n✅ Multi-agent demo complete!")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "--demo":
            run_demo_conversation()
        elif sys.argv[1] == "--interactive":
            run_interactive_conversation()
        elif sys.argv[1] == "--multi-agent":
            run_multi_agent_collaboration()
        else:
            print("Usage: python3 autogen_agent_with_toondb.py [--demo|--interactive|--multi-agent]")
    else:
        run_demo_conversation()
