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
| **Go** | [`toondb_rag_go/`](./toondb_rag_go) | Go implementation demonstrating the `toondb-go` SDK in a RAG context. |
| **Rust** | [`toondb_rag_rust/`](./toondb_rag_rust) | High-performance Rust implementation using the native `toondb` crate. |

### 📚 ToonDB SDK Examples
Simple examples demonstrating basic CRUD operations, SQL usage, and connection handling.

| Language | Directory | Description |
|----------|-----------|-------------|
| **Go** | [`toondb_go_examples/`](./toondb_go_examples) | Basic usage of the Go SDK, including connection, key-value ops, and SQL checks. |
| **Node.js**| [`toondb_node_examples/`](./toondb_node_examples) | JavaScript examples for Node.js, showing setup, basic operations, and SQL. |
| **Rust** | [`toondb_rust_examples/`](./toondb_rust_examples) | Rust native examples showing how to embed ToonDB directly. |

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
