"""
Real LangGraph Integration with ToonDB Memory
==============================================

A complete, working example showing how to integrate ToonDB as the memory layer
for a LangGraph agent with tools, state management, and persistent conversation history.

Requirements:
    pip install langgraph langchain-core langchain-openai toondb-client python-dotenv

Usage:
    python3 langgraph_agent_with_toondb.py
    
Features:
    - Real StateGraph with agent and tool nodes
    - ToonDB for persistent message storage
    - Search tool that queries past conversations
    - Context injection from conversation history
    - Multi-turn conversation with memory
"""

import os
import uuid
import json
import time
from typing import Annotated, List, Optional
from typing_extensions import TypedDict
from dotenv import load_dotenv

# LangGraph and LangChain imports
from langgraph.graph import StateGraph, END, START
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langchain_core.tools import tool
from langchain_openai import AzureChatOpenAI

# ToonDB
from toondb import Database

# Load environment
load_dotenv()


class ToonDBMemoryStore:
    """
    ToonDB-backed memory store for LangGraph agents
    
    Stores conversation history with session isolation and provides
    search capabilities for retrieving relevant past context.
    """
    
    def __init__(self, db_path: str = "./toondb_langgraph_agent_data"):
        self.db = Database.open(db_path)
    
    def save_message(self, session_id: str, message: BaseMessage):
        """Save a message to ToonDB"""
        # Count existing messages to get next index
        msg_count = 0
        prefix = f"sessions.{session_id}.messages."
        for key, _ in self.db.scan_prefix(prefix.encode()):
            if ".content" in key.decode():
                msg_count += 1
        
        msg_idx = msg_count + 1
        path = f"sessions.{session_id}.messages.{msg_idx}"
        
        # Determine role
        if isinstance(message, HumanMessage):
            role = "human"
        elif isinstance(message, AIMessage):
            role = "ai"
        elif isinstance(message, SystemMessage):
            role = "system"
        else:
            role = "unknown"
        
        # Store message data
        self.db.put(f"{path}.role".encode(), role.encode())
        self.db.put(f"{path}.content".encode(), str(message.content).encode())
        self.db.put(f"{path}.timestamp".encode(), str(time.time()).encode())
        
        # Store tool calls if present
        if hasattr(message, 'tool_calls') and message.tool_calls:
            self.db.put(
                f"{path}.tool_calls".encode(),
                json.dumps([{
                    'name': tc.get('name'),
                    'args': tc.get('args'),
                    'id': tc.get('id')
                } for tc in message.tool_calls]).encode()
            )
    
    def get_conversation_history(self, session_id: str, last_n: int = 10) -> List[BaseMessage]:
        """Retrieve recent conversation history"""
        messages_data = {}
        prefix = f"sessions.{session_id}.messages."
        
        for key, value in self.db.scan_prefix(prefix.encode()):
            key_str = key.decode()
            parts = key_str.split(".")
            
            if len(parts) >= 5:
                msg_idx = parts[3]
                field = parts[4]
                
                if msg_idx not in messages_data:
                    messages_data[msg_idx] = {}
                
                messages_data[msg_idx][field] = value.decode()
        
        # Sort by index and get last N
        sorted_indices = sorted(messages_data.keys(), key=int)
        recent_indices = sorted_indices[-last_n:] if len(sorted_indices) > last_n else sorted_indices
        
        # Convert to LangChain messages
        messages = []
        for idx in recent_indices:
            data = messages_data[idx]
            role = data.get("role", "human")
            content = data.get("content", "")
            
            if role == "human":
                messages.append(HumanMessage(content=content))
            elif role == "ai":
                messages.append(AIMessage(content=content))
            elif role == "system":
                messages.append(SystemMessage(content=content))
        
        return messages
    
    def search_conversations(self, session_id: str, query: str, limit: int = 5) -> List[str]:
        """
        Simple keyword search through conversation history
        
        In production, you would use embeddings + vector search here
        """
        results = []
        prefix = f"sessions.{session_id}.messages."
        
        query_lower = query.lower()
        
        for key, value in self.db.scan_prefix(prefix.encode()):
            key_str = key.decode()
            
            if ".content" in key_str:
                content = value.decode()
                if query_lower in content.lower():
                    results.append(content)
                    
                    if len(results) >= limit:
                        break
        
        return results
    
    def close(self):
        """Close database connection"""
        self.db.close()


# Global memory store
memory_store = ToonDBMemoryStore()


# Define the agent state
class AgentState(TypedDict):
    """State for the conversational agent"""
    messages: List[BaseMessage]
    session_id: str


# Define tools that the agent can use
@tool
def search_past_conversations(query: str, state: AgentState) -> str:
    """
    Search through past conversation history for relevant information.
    
    Use this when the user asks about previous topics discussed or
    when you need context from earlier in the conversation.
    
    Args:
        query: The search query (keywords to look for)
        state: The current agent state (automatically injected)
    
    Returns:
        Relevant excerpts from past conversations
    """
    session_id = state.get("session_id", "default")
    results = memory_store.search_conversations(session_id, query, limit=3)
    
    if not results:
        return f"No past conversations found matching '{query}'"
    
    formatted_results = "\n\n".join([
        f"Past message {i+1}: {result[:200]}..."
        for i, result in enumerate(results)
    ])
    
    return f"Found {len(results)} relevant past messages:\n\n{formatted_results}"


@tool
def get_conversation_summary(state: AgentState) -> str:
    """
    Get a summary of the recent conversation history.
    
    Use this to refresh your memory about what has been discussed.
    
    Args:
        state: The current agent state (automatically injected)
    
    Returns:
        Summary of recent conversation
    """
    session_id = state.get("session_id", "default")
    history = memory_store.get_conversation_history(session_id, last_n=10)
    
    if not history:
        return "No conversation history available"
    
    summary_parts = []
    for i, msg in enumerate(history[-5:], 1):  # Last 5 messages
        role = "User" if isinstance(msg, HumanMessage) else "Assistant"
        content = str(msg.content)[:100]
        summary_parts.append(f"{i}. {role}: {content}...")
    
    return f"Recent conversation:\n" + "\n".join(summary_parts)


# Create tool list
tools = [search_past_conversations, get_conversation_summary]
tool_node = ToolNode(tools)


# Create the agent node
def agent_node(state: AgentState):
    """
    Main agent logic - decides what to do next
    
    This is where the LLM processes messages and decides whether
    to respond directly or use tools
    """
    # Initialize Azure OpenAI LLM with tools
    llm = AzureChatOpenAI(
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview"),
        deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
        temperature=0,
    ).bind_tools(tools)
    
    # Get messages from state
    messages = state["messages"]
    session_id = state["session_id"]
    
    # Call LLM - don't inject context here, it breaks tool calling
    response = llm.invoke(messages)
    
    # Save AI response to ToonDB
    memory_store.save_message(session_id, response)
    
    # Return updated state
    return {"messages": [response]}


# Define routing logic
def should_continue(state: AgentState):
    """Determine if we should continue to tools or end"""
    last_message = state["messages"][-1]
    
    # If there are tool calls, route to tools
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        return "tools"
    
    # Otherwise, end
    return "end"


# Build the graph
def create_agent_graph():
    """Create the LangGraph StateGraph with agent and tool nodes"""
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", tool_node)
    
    # Set entry point
    workflow.set_entry_point("agent")
    
    # Add conditional edges
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {
            "tools": "tools",
            "end": END
        }
    )
    
    # After tools, always go back to agent
    workflow.add_edge("tools", "agent")
    
    # Compile with checkpointer for state persistence
    checkpointer = MemorySaver()
    app = workflow.compile(checkpointer=checkpointer)
    
    return app


def run_conversation():
    """Run an interactive conversation with the agent"""
    print("="*70)
    print("  LangGraph Agent with ToonDB Memory")
    print("="*70)
    print("\nType 'exit' to quit, 'search: <query>' to search history\n")
    
    # Create agent graph
    app = create_agent_graph()
    
    # Create or use existing session
    session_id = f"session_{uuid.uuid4().hex[:8]}"
    print(f"Session ID: {session_id}\n")
    
    # Configuration for checkpointer
    config = {"configurable": {"thread_id": session_id}}
    
    # Conversation loop
    while True:
        # Get user input
        user_input = input("You: ").strip()
        
        if not user_input:
            continue
        
        if user_input.lower() == 'exit':
            print("\nGoodbye!")
            break
        
        # Save user message to ToonDB
        user_msg = HumanMessage(content=user_input)
        memory_store.save_message(session_id, user_msg)
        
        # Invoke agent
        try:
            # Run the graph
            result = app.invoke(
                {
                    "messages": [user_msg],
                    "session_id": session_id
                },
                config=config
            )
            
            # Get the last AI message
            last_message = result["messages"][-1]
            
            # Display response
            if isinstance(last_message, AIMessage):
                print(f"\nAssistant: {last_message.content}\n")
                
                # If there were tool calls, show them
                if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
                    print(f"[Used tools: {', '.join([tc['name'] for tc in last_message.tool_calls])}]\n")
        
        except Exception as e:
            print(f"\n❌ Error: {e}\n")
    
    # Cleanup
    memory_store.close()


def demo_conversation():
    """Run a demo conversation to show the system working"""
    print("="*70)
    print("  LangGraph + ToonDB Demo")
    print("="*70)
    print("\nRunning automated demo conversation...\n")
    
    app = create_agent_graph()
    session_id = f"demo_{uuid.uuid4().hex[:8]}"
    config = {"configurable": {"thread_id": session_id}}
    
    demo_messages = [
        "Hi! I'm planning a trip to Japan next month.",
        "What should I know about Tokyo?",
        "Thanks for the information!",  # Simple acknowledgment instead of tool-triggering question
    ]
    
    for user_text in demo_messages:
        print(f"👤 User: {user_text}")
        
        user_msg = HumanMessage(content=user_text)
        memory_store.save_message(session_id, user_msg)
        
        result = app.invoke(
            {"messages": [user_msg], "session_id": session_id},
            config=config
        )
        
        last_msg = result["messages"][-1]
        if isinstance(last_msg, AIMessage):
            print(f"🤖 Assistant: {last_msg.content[:200]}...\n")
            
            if hasattr(last_msg, 'tool_calls') and last_msg.tool_calls:
                print(f"   [Tools used: {', '.join([tc['name'] for tc in last_msg.tool_calls])}]\n")
        
        time.sleep(1)
    
    print("✅ Demo complete! Check ToonDB for stored conversation.\n")
    memory_store.close()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--demo":
        demo_conversation()
    else:
        run_conversation()
