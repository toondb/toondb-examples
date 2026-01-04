# ToonDB Examples

Welcome to the official examples repository for **ToonDB**, the high-performance embedded database designed for AI applications. This repository contains various examples to help you get started with ToonDB across different programming languages and use cases.

## 📂 Repository Structure

The repository is organized by language and use case.

### 🧠 RAG Applications (Retrieval-Augmented Generation)
Complete, production-ready reference implementations of a RAG system using ToonDB as the vector store and Azure OpenAI.

| Language | Directory | Description |
|----------|-----------|-------------|
| **Python** | [`toondb_rag_python/`](./toondb_rag_python) | Full RAG pipeline with ToonDB persistence, chunking, and Azure OpenAI integration. |
| **Node.js** | [`toondb_rag_node/`](./toondb_rag_node) | Node.js implementation using `@sushanth/toondb` SDK. |
| **Rust** | [`toondb_rag_rust/`](./toondb_rag_rust) | High-performance Rust implementation using the native `toondb` crate. |
| **Go** | [External Repo →](https://github.com/toondb/toondb-golang-examples) | Go examples are maintained in a separate repository. |

### 🤖 Agent Memory & Advanced Scenarios
These examples demonstrate how to use ToonDB as a high-performance memory backend for autonomous agents, implementing features like long-term recall, semantic search, and state persistence.

#### 1. LangGraph Agent (`toondb_langgraph_examples`)
**Best for**: Building complex, stateful agents with LangGraph.
- **Persistent State**: Uses ToonDB as a `checkpointer` to save graph state (threads), allowing agents to pause and resume.
- **Long-Term Memory**: Implements a dedicated memory store for recalling past user interactions using vector search.
- **Features**: Time-weighted retrieval, compact memory format.

#### 2. eCommerce RAG (`toondb_ecommerce_examples`)
**Best for**: Shopping assistants, product catalogs, and recommendation systems.
- **Hybrid Search**: Combines semantic search (embeddings) with metadata filtering (e.g., price, category).
- **TOON Format**: Demonstrates `Database.to_toon()` for formatting search results into a token-efficient string for LLMs.
- **Ingestion**: Scripts to ingest and index structured JSON product data.

#### 3. Azure OpenAI "California Politics" (`toondb_azure_openai_examples`)
**Best for**: Knowledge retrieval systems using Azure OpenAI.
- **Fact Retrieval**: Stores and retrieves facts about specific entities (Kamala Harris, Gavin Newsom) from disjointed text chunks.
- **High Accuracy**: Optimized for precision in retrieving "needle in a haystack" facts.
- **Integration**: Direct integration with Azure OpenAI embeddings and chat models.

#### 4. Wizard of Oz (`toondb_wizard_of_oz_examples`)
**Best for**: Long-context narrative understanding and book ingestion.
- **Chunking Strategy**: Demonstrates how to chunk large unstructured text (a novel) into semantic episodes.
- **Narrative Search**: Enables searching for plot points, character details, and thematic elements across a long document.

#### 5. Podcast Search (`toondb_podcast_examples`)
**Best for**: Audio transcripts, meeting notes, and multi-speaker dialogue.
- **Transcript Parsing**: Handles specialized formats (Speaker: Timestamp) and effectively models dialogue turns.
- **Speaker Attribution**: Preserves identifying metadata to allow searching for "What did X say about Y?".

#### 6. Zep Port (`toondb_zep_examples`)
**Best for**: Users migrating from Zep or needing entity-centric memory.
- **Entity Extraction**: Automatically extracts and stores named entities (People, Organizations) alongside conversation history.
- **User Management**: Manages user profiles and metadata associated with conversation threads.

#### 7. Context Query Builder (`toondb_context_builder_examples`)
**Best for**: Managing LLM context under strict token budgets.
- **Priority-Based Truncation**: Intelligently fits system message, query, history, and retrieval results within token limits.
- **Token Budget Management**: Automatically truncates lower-priority content when space is tight.
- **TOON Integration**: Demonstrates using `Database.to_toon()` for ultra-compact retrieval formatting.
- **Flexible Assembly**: Supports different priority levels for different content types.

### 📚 ToonDB SDK Examples
Simple examples demonstrating basic CRUD operations, SQL usage, and connection handling.

| Language | Directory | Description |
|----------|-----------|-------------|
| **Node.js**| [`toondb_node_examples/`](./toondb_node_examples) | JavaScript examples for Node.js, showing setup, basic operations, and SQL. |
| **Rust** | [`toondb_rust_examples/`](./toondb_rust_examples) | Rust native examples showing how to embed ToonDB directly. |
| **Go** | [External Repo →](https://github.com/toondb/toondb-golang-examples) | Go examples are maintained in a separate repository. |

## 🚀 Getting Started

Choose your preferred language and navigate to the corresponding directory. Each example folder contains its own `README.md` with specific setup and running instructions.

### Prerequisites

- **ToonDB Server**: For Go, Node.js, and Python SDKs, ensure the `toondb` or `toondb-server` binary is installed and available in your system's `PATH`.
  - Python: `pip install toondb-client` (includes binaries)
  - Go: `go install github.com/toondb/toondb/cmd/toondb@latest`
  - Rust: Uses the library directly.
- **Language Runtimes**: Python 3.10+, Node.js 18+, Go 1.21+, or Rust 1.75+ depending on the example.
- **API Keys**: For RAG examples, you will need an Azure OpenAI API key (or compatible OpenAI endpoint).

## 🔑 Key Features Demonstrated

- **Vector Search**: Using ToonDB's HNSW index for fast similarity search.
- **Persistence**: Storing embeddings and metadata reliably on disk.
- **Hybrid Search**: Combining vector search with structured filtering (in applicable SDKs).
- **Multi-Language Support**: Consistent API usage patterns across Python, TS/JS, Go, and Rust.

## 🤝 Contributing

Feel free to submit Pull Requests with new examples or improvements to existing ones!

## 📄 License

Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## 🔗 Useful Links

- [Official ToonDB Documentation](https://toondb.io)
- [Python SDK (PyPI)](https://pypi.org/project/toondb-client/)
- [Node.js SDK (npm)](https://www.npmjs.com/package/@sushanth/toondb)
- [Go SDK](https://pkg.go.dev/github.com/toondb/toondb-go)
- [Rust Crate](https://crates.io/crates/toondb)

## Acknowledgements

Some of the agent memory examples (Wizard of Oz, Podcast, Zep ports) are referenced and adapted from the following projects:
- [Zep](https://github.com/getzep/zep)
- [Graphiti](https://github.com/getzep/graphiti)
