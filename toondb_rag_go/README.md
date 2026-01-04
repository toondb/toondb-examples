# ToonDB RAG System (Go)

A production-ready RAG system built with ToonDB and Azure OpenAI for Go.

## Project Structure

```
toondb_rag_go/
├── cmd/demo/main.go        # Demo entry point
├── internal/
│   ├── config/config.go    # Configuration
│   ├── documents/          # Document loading
│   ├── chunking/           # Text chunking
│   ├── embeddings/         # Azure OpenAI embeddings
│   ├── vectorstore/        # ToonDB storage
│   ├── generation/         # LLM generation
│   └── rag/                # Main RAG class
├── .env                    # Configuration
└── go.mod
```

## Quick Start

```bash
# Build
go build ./cmd/demo

# Run demo (uses mock mode by default)
go run ./cmd/demo
```

## Usage

```go
package main

import (
    "github.com/toondb/toondb-rag-go/internal/config"
    "github.com/toondb/toondb-rag-go/internal/rag"
)

func main() {
    cfg, _ := config.Load()
    
    ragSystem := rag.New(cfg, rag.Options{
        DBPath:           "./my_rag_db",
        ChunkingStrategy: "semantic",
        UseMock:          false, // Use real Azure OpenAI
    })
    defer ragSystem.Close()

    // Ingest documents
    ragSystem.IngestFile("./docs/my_doc.md")

    // Query
    response, _ := ragSystem.Query("What is the main topic?")
    fmt.Println(response.Answer)
}
```

## Configuration

Edit `.env`:
```
AZURE_OPENAI_API_KEY=your_key
AZURE_OPENAI_ENDPOINT=your_endpoint
AZURE_OPENAI_EMBEDDING_DEPLOYMENT=text-embedding-3-small
AZURE_OPENAI_CHAT_DEPLOYMENT=gpt-4.1
TOONDB_PATH=./toondb_data
```

## Note

The Go SDK requires an embedded server to run. For production use, ensure the ToonDB server is running or use the mock mode for testing.
