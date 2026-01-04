import os
import uuid
import asyncio
from typing import Annotated, Literal, TypedDict
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langchain_core.tools import tool
from langchain_openai import AzureChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langgraph.graph.message import add_messages

from toondb_langgraph_examples.config import get_azure_config
from toondb_langgraph_examples.memory import ToonDBMemory
from toondb_langgraph_examples.checkpointer import ToonDBCheckpointer

# Initialize Memory and Checkpointer
memory_store = ToonDBMemory()
checkpointer = ToonDBCheckpointer()

class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    
@tool
def recall_memory(query: str) -> str:
    """
    Search long-term memory for relevant facts or past conversations.
    Always uses this when the user asks about something from the past or context.
    Returns memories in TOON format: <source:relevance> content
    """
    print(f"\n🔍 Searching memory for: '{query}'")
    results = memory_store.search(query)
    return f"Relevant Memories:\n{results}"

@tool
def save_memory(content: str) -> str:
    """
    Save an important fact or event to long-term memory.
    Use this when the user tells you something important that should be remembered.
    """
    print(f"\n💾 Saving memory: '{content}'")
    memory_store.add_episode(content, source="user_interaction")
    return "Memory saved successfully."

tools = [recall_memory, save_memory]
tool_node = ToolNode(tools)

def get_model():
    config = get_azure_config()
    return AzureChatOpenAI(
        api_key=config.api_key,
        api_version=config.api_version,
        azure_endpoint=config.endpoint,
        deployment_name=config.deployment,
        temperature=0,
    ).bind_tools(tools)

def agent_node(state: AgentState):
    model = get_model()
    messages = state["messages"]
    
    # System prompt to encourage using memory
    system_prompt = SystemMessage(content="""You are a helpful AI assistant with broad long-term memory.
    You serve two primary functions:
    1. Answering user questions.
    2. Remembering important details.
    
    You have access to a 'recall_memory' tool to search your database.
    You have access to a 'save_memory' tool to store new facts.
    
    When you search memory, you will receive results in 'TOON format' which is compact.
    Example: <dialogue:0.89> User likes pizza.
    
    Always check your memory if the user asks about something you might have discussed before.
    """)
    
    # Prepend system prompt if not present (simplified for this demo)
    if not isinstance(messages[0], SystemMessage):
        messages = [system_prompt] + messages
    else:
        # Update system prompt? For now just rely on the first one or inject context
        pass
        
    response = model.invoke(messages)
    return {"messages": [response]}

def should_continue(state: AgentState) -> Literal["tools", "end"]:
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return "end"

def create_agent():
    workflow = StateGraph(AgentState)
    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", tool_node)
    
    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges(
        "agent", 
        should_continue,
        {
            "tools": "tools",
            "end": END
        }
    )
    workflow.add_edge("tools", "agent")
    
    return workflow.compile(checkpointer=checkpointer)

def main():
    print("Initializing ToonDB LangGraph Agent...")
    print(f"Database Path: {memory_store.db_path}")
    
    app = create_agent()
    
    # Use a static thread ID for demo purposes so state persists across runs
    thread_id = "demo_thread_fixed_1" 
    config = {"configurable": {"thread_id": thread_id}}
    
    print(f"Session ID: {thread_id}")
    print("Type 'quit' to exit.")
    
    while True:
        try:
            user_input = input("\nUser: ")
            if user_input.lower() in ["quit", "exit"]:
                break
                
            input_message = HumanMessage(content=user_input)
            
            # Stream the graph execution
            for event in app.stream({"messages": [input_message]}, config=config, stream_mode="values"):
                # We can print intermediate steps if we want
                pass
                
            # Get the final state to print the last response
            snapshot = app.get_state(config)
            if snapshot.values and snapshot.values["messages"]:
                last_msg = snapshot.values["messages"][-1]
                if isinstance(last_msg, AIMessage):
                    print(f"Agent: {last_msg.content}")
                    
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")

    memory_store.close()
    checkpointer.close()

if __name__ == "__main__":
    main()
