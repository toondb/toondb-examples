# SochDB Rust Examples

Welcome to the official Rust examples repository for **SochDB**, the high-performance embedded database designed for AI applications.

## 📂 Repository Structure

### 🦀 Basic Examples
Complete, production-ready examples using the official `sochdb` crate from crates.io.

| Directory | Description |
|-----------|-------------|
| [`sochdb-examples/`](./sochdb-examples) | Comprehensive examples using sochdb 0.4.0 from crates.io |
| [`sochdb-rag-example/`](./sochdb-rag-example) | Full RAG pipeline with Azure OpenAI integration |

## 🚀 Quick Start

### Prerequisites

- Rust 1.70+ (2021 edition)
- Cargo

### Running the Basic Examples

```bash
cd sochdb-examples
cargo run --release
```

### What's Included

The basic examples demonstrate:
- ✅ Basic key-value operations (Put, Get, Delete)
- ✅ Path-based hierarchical keys
- ✅ Database statistics
- ✅ Using the published crate from crates.io

## 📦 Using SochDB in Your Project

Add to your `Cargo.toml`:

```toml
[dependencies]
sochdb = "0.4.0"
anyhow = "1"
```

Example code:

```rust
use sochdb::Database;
use anyhow::Result;

fn main() -> Result<()> {
    // Open database
    let db = Database::open("./mydb")?;
    
    // Basic operations
    db.put(b"key", b"value")?;
    let value = db.get(b"key")?;
    
    // Path-based keys
    db.put_path("users/alice/email", b"alice@example.com")?;
    let email = db.get_path("users/alice/email")?;
    
    Ok(())
}
```

## 🧠 RAG Application Example

See [`sochdb-rag-example/`](./sochdb-rag-example) for a complete Retrieval-Augmented Generation system using:
- SochDB as the vector store
- Azure OpenAI for embeddings and chat
- Document chunking and ingestion

## 📚 Documentation

- [SochDB Main Repository](https://github.com/sochdb/sochdb)
- [Rust Crate Documentation](https://docs.rs/sochdb)
- [API Reference](https://sochdb.dev)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

Apache 2.0 - see [LICENSE](LICENSE) for details.
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
