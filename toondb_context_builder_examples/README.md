# Context Query Builder Example

This example demonstrates ToonDB's **Context Query Builder** - a powerful feature for assembling LLM context under strict token budgets.

## What is Context Query Builder?

The Context Query Builder assembles the final prompt for an LLM by combining:
- **System Message**: Instructions and persona for the LLM
- **User Query**: The current user input
- **Conversation History**: Recent exchanges
- **Retrieved Context**: Semantically relevant information from vectorsearch

All of this is done under a **token budget** with **priority-based truncation**, ensuring the most important information fits within the LLM's context window.

## Key Features

- **Token Budget Management**: Automatically fits content within your specified token limit
- **Priority-Based Trucation**: Keeps the most important content (system + current query), then adds history and retrieval as space allows
- **TOON Format Integration**: Uses ToonDB's `to_toon()` for compact retrieval formatting
- **Flexible Truncation**: Intelligently truncates conversation history from the middle, keeping recent and oldest context

## Use Cases

- **Chatbots**: Keep conversation context relevant and within limits
- **RAG Systems**: Balance retrieved documents with conversation flow
- **Long Conversations**: Maintain coherence even after hundreds of turns
- **Multi-Modal Agents**: Coordinate different types of context (text, metadata, tool outputs)

## Running the Example

```bash
pip install -r requirements.txt
python3 runner.py
```

## Example Output

The script demonstrates:
1. Building context with ample token budget → All content fits
2. Building context with tight budget → Intelligent truncation
3. Priority ordering → System and query always included first
