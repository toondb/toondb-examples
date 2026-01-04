# ToonDB RAG System (Rust)

A production-ready RAG system built with ToonDB and Azure OpenAI for Rust.

## Project Structure

```
toondb_rag_rust/
├── src/
│   ├── main.rs         # Demo entry point
│   ├── config.rs       # Configuration
│   ├── documents.rs    # Document loading
│   ├── chunking.rs     # Text chunking
│   ├── embeddings.rs   # Azure OpenAI embeddings
│   ├── vectorstore.rs  # In-memory vector storage
│   ├── generation.rs   # LLM generation
│   └── rag.rs          # Main RAG class
├── .env                # Configuration
└── Cargo.toml          # Dependencies
```

## Quick Start

```bash
# Build
cargo build

# Run demo
cargo run
```

## Usage

```rust
use crate::config::Config;
use crate::rag::{RAGOptions, ToonDBRAG};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let config = Config::load();
    
    let mut rag = ToonDBRAG::new(
        &config,
        RAGOptions {
            db_path: Some("./my_rag_db".to_string()),
            chunking_strategy: "semantic".to_string(),
            use_mock: false,
        },
    );

    // Ingest documents
    rag.ingest_file("./docs/my_doc.md").await?;

    // Query
    let response = rag.query("What is the main topic?").await?;
    println!("{}", response.answer);

    Ok(())
}
```

## Configuration

Edit `.env`:
```
AZURE_OPENAI_API_KEY=your_key
AZURE_OPENAI_ENDPOINT=your_endpoint
AZURE_OPENAI_EMBEDDING_DEPLOYMENT=text-embedding-3-small
AZURE_OPENAI_CHAT_DEPLOYMENT=gpt-4.1
```

## Dependencies

- `toondb = "0.3"` - ToonDB database
- `tokio` - Async runtime
- `reqwest` - HTTP client for Azure API
- `serde` - Serialization
