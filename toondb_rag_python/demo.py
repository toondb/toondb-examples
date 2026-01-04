#!/usr/bin/env python3
"""
ToonDB RAG System - Demo Script

This script demonstrates the complete RAG pipeline with a sample PDF.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
load_dotenv()

from rag import ToonDBRAG


def create_sample_document():
    """Create a sample document for demo"""
    sample_content = """
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

Python:
```
pip install toondb-client
```

Node.js:
```
npm install @sushanth/toondb
```

## Quick Start

```python
from toondb import Database

with Database.open("./my_db") as db:
    db.put(b"key", b"value")
    value = db.get(b"key")
    print(value)  # b"value"
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
"""
    
    sample_path = Path(__file__).parent / "documents" / "sample_toondb_docs.md"
    sample_path.parent.mkdir(exist_ok=True)
    
    with open(sample_path, 'w') as f:
        f.write(sample_content)
    
    print(f"📄 Created sample document: {sample_path}")
    return sample_path


def run_demo():
    """Run the RAG demo"""
    print("=" * 60)
    print("🚀 ToonDB RAG System Demo")
    print("=" * 60)
    
    # Create sample document
    sample_path = create_sample_document()
    
    # Initialize RAG
    print("\n📦 Initializing RAG system...")
    # Initialize RAG
    print("\n📦 Initializing RAG system...")
    # Initialize real RAG system (will raise error if connection fails)
    rag = ToonDBRAG(db_path="./demo_toondb_data", use_azure=True, use_mock=False)
    
    # Test connection
    print("   🔄 Verify connection to Azure OpenAI...")
    rag.embedder.embed_query("test connection")
    print("   ✅ Connected to Azure OpenAI")

    with rag:
        
        # Clear previous data
        rag.clear()
        
        # Ingest document
        print("\n📥 Ingesting document...")
        count = rag.ingest_file(str(sample_path))
        print(f"   ✅ Created {count} chunks")
        
        # Show stats
        stats = rag.get_stats()
        print(f"\n📊 Stats: {stats['total_chunks']} chunks from {stats['ingested_documents']} documents")
        
        # Test queries
        test_questions = [
            "What is ToonDB?",
            "How do I install ToonDB in Python?",
            "What are the main features of ToonDB?",
            "Does ToonDB support SQL?",
            "What is HNSW in ToonDB?",
        ]
        
        print("\n" + "=" * 60)
        print("🔍 Running Test Queries")
        print("=" * 60)
        
        for question in test_questions:
            print(f"\n❓ Question: {question}")
            print("-" * 40)
            
            response = rag.query(question)
            
            print(f"📊 Confidence: {response.confidence}")
            print(f"💬 Answer: {response.answer[:300]}...")
            
            if response.sources:
                print(f"📚 Top source score: {response.sources[0].score:.3f}")
        
        print("\n" + "=" * 60)
        print("✅ Demo completed!")
        print("=" * 60)
        
        # Interactive option
        print("\n💡 You can now run interactive mode:")
        print("   python main.py interactive")


if __name__ == "__main__":
    run_demo()
