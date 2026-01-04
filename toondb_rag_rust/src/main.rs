//! ToonDB RAG System Demo (Rust)
mod chunking;
mod config;
mod documents;
mod embeddings;
mod generation;
mod rag;
mod vectorstore;

use config::Config;
use rag::{RAGOptions, ToonDBRAG};
use std::fs;
use std::path::Path;

const SAMPLE_CONTENT: &str = r#"
# ToonDB Documentation

## Overview

ToonDB is a high-performance embedded database designed for AI applications.
It provides key-value storage, vector search, and SQL capabilities.

## Features

### Key-Value Store
ToonDB offers simple get/put/delete operations for key-value data.
Keys can be hierarchical paths like "users/alice/email".

### Vector Search
ToonDB includes HNSW (Hierarchical Navigable Small World) index for 
fast approximate nearest neighbor search. This is ideal for RAG applications.

### SQL Support
ToonDB supports basic SQL operations including:
- CREATE TABLE
- INSERT INTO
- SELECT with WHERE, ORDER BY, LIMIT
- UPDATE and DELETE

### Transactions
All operations support ACID transactions with snapshot isolation.

## Installation

Rust:
```
[dependencies]
toondb = "0.3"
```

Python:
```
pip install toondb-client
```

## Quick Start

```rust
use toondb::Database;

fn main() {
    let db = Database::open("./my_db").unwrap();
    db.put(b"key", b"value").unwrap();
    let value = db.get(b"key").unwrap();
    println!("{:?}", value);
    db.close().unwrap();
}
```

## Performance

ToonDB is optimized for:
- Low latency reads (sub-millisecond)
- High throughput writes
- Efficient vector search with HNSW

## Use Cases

1. RAG (Retrieval-Augmented Generation)
2. Semantic search applications
3. LLM context retrieval
4. Multi-tenant data storage
5. Embedded databases for desktop apps
"#;

fn create_sample_document() -> anyhow::Result<String> {
    let docs_dir = Path::new("documents");
    fs::create_dir_all(docs_dir)?;

    let sample_path = docs_dir.join("sample_toondb_docs.md");
    fs::write(&sample_path, SAMPLE_CONTENT)?;

    println!("📄 Created sample document: {}", sample_path.display());
    Ok(sample_path.to_string_lossy().to_string())
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    println!("{}", "=".repeat(60));
    println!("🚀 ToonDB RAG System Demo (Rust)");
    println!("{}", "=".repeat(60));

    // Create sample document
    let sample_path = create_sample_document()?;

    // Load config
    let config = Config::load();

    // Initialize RAG with real Azure OpenAI
    println!("\n📦 Initializing RAG system...");
    println!("   🔄 Verifying Azure OpenAI connection...");

    let mut rag = ToonDBRAG::new(
        &config,
        RAGOptions {
            db_path: Some("./demo_toondb_data".to_string()),
            chunking_strategy: "semantic".to_string(),
            use_mock: false, // Try real Azure OpenAI first
        },
    );

    // Test connection by doing a simple embed
    match rag.query("test").await {
        Ok(_) => {
            println!("   ✅ Connected to Azure OpenAI");
            // Re-initialize with fresh state
            rag = ToonDBRAG::new(
                &config,
                RAGOptions {
                    db_path: Some("./demo_toondb_data".to_string()),
                    chunking_strategy: "semantic".to_string(),
                    use_mock: false,
                },
            );
        }
        Err(e) => {
            println!("   ⚠️ Could not connect to Azure: {}", e);
            println!("   🔄 Falling back to Mock Mode");
            rag = ToonDBRAG::new(
                &config,
                RAGOptions {
                    db_path: Some("./demo_toondb_data".to_string()),
                    chunking_strategy: "semantic".to_string(),
                    use_mock: true,
                },
            );
        }
    }

    // Clear and ingest
    rag.clear()?;

    println!("\n📥 Ingesting document...");
    let count = rag.ingest_file(&sample_path).await?;
    println!("   ✅ Created {} chunks", count);

    // Stats
    let (chunks, docs) = rag.stats();
    println!("\n📊 Stats: {} chunks from {} documents", chunks, docs);

    // Test queries
    let test_questions = vec![
        "What is ToonDB?",
        "How do I install ToonDB in Rust?",
        "What are the main features of ToonDB?",
        "Does ToonDB support SQL?",
        "What is HNSW in ToonDB?",
    ];

    println!("\n{}", "=".repeat(60));
    println!("🔍 Running Test Queries");
    println!("{}", "=".repeat(60));

    for question in test_questions {
        println!("\n❓ Question: {}", question);
        println!("{}", "-".repeat(40));

        match rag.query(question).await {
            Ok(response) => {
                println!("📊 Confidence: {}", response.confidence);

                let answer = if response.answer.len() > 300 {
                    format!("{}...", &response.answer[..300])
                } else {
                    response.answer.clone()
                };
                println!("💬 Answer: {}", answer);

                if !response.sources.is_empty() {
                    println!("📚 Top source score: {:.3}", response.sources[0].score);
                }
            }
            Err(e) => {
                println!("Error: {}", e);
            }
        }
    }

    println!("\n{}", "=".repeat(60));
    println!("✅ Demo completed!");
    println!("{}", "=".repeat(60));

    Ok(())
}
