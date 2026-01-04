// Demo for ToonDB RAG System
package main

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/toondb/toondb-rag-go/internal/config"
	"github.com/toondb/toondb-rag-go/internal/rag"
)

const sampleContent = `
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

Go:
` + "```" + `
go get github.com/toondb/toondb/toondb-go
` + "```" + `

Python:
` + "```" + `
pip install toondb-client
` + "```" + `

## Quick Start

` + "```go" + `
package main

import (
    toondb "github.com/toondb/toondb/toondb-go"
)

func main() {
    db, _ := toondb.Open("./my_db")
    db.Put([]byte("key"), []byte("value"))
    value, _ := db.Get([]byte("key"))
    fmt.Println(string(value)) // "value"
    db.Close()
}
` + "```" + `

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
`

func createSampleDocument() (string, error) {
	docsDir := filepath.Join(".", "documents")
	if err := os.MkdirAll(docsDir, 0755); err != nil {
		return "", err
	}

	samplePath := filepath.Join(docsDir, "sample_toondb_docs.md")
	if err := os.WriteFile(samplePath, []byte(sampleContent), 0644); err != nil {
		return "", err
	}

	fmt.Printf("📄 Created sample document: %s\n", samplePath)
	return samplePath, nil
}

func main() {
	fmt.Println(strings.Repeat("=", 60))
	fmt.Println("🚀 ToonDB RAG System Demo (Go)")
	fmt.Println(strings.Repeat("=", 60))

	// Create sample document
	samplePath, err := createSampleDocument()
	if err != nil {
		fmt.Printf("Error creating sample document: %v\n", err)
		return
	}

	// Load config
	cfg, err := config.Load()
	if err != nil {
		fmt.Printf("Error loading config: %v\n", err)
		return
	}

	// Initialize RAG
	fmt.Println("\n📦 Initializing RAG system...")

	// Use real Azure OpenAI - Go SDK ships with embedded binary
	ragSystem := rag.New(cfg, rag.Options{
		DBPath:           "./demo_toondb_data",
		ChunkingStrategy: "semantic",
		UseMock:          false, // Use real Azure OpenAI
	})
	defer ragSystem.Close()

	// Test connection
	fmt.Println("   🔄 Verifying Azure OpenAI connection...")
	_, err = ragSystem.Search("test", 1)
	if err != nil {
		fmt.Printf("   ⚠️ Could not connect to Azure: %v\n", err)
		fmt.Println("   🔄 Falling back to Mock Mode")
		ragSystem = rag.New(cfg, rag.Options{
			DBPath:           "./demo_toondb_data",
			ChunkingStrategy: "semantic",
			UseMock:          true,
		})
	} else {
		fmt.Println("   ✅ Connected to Azure OpenAI")
	}

	// Clear and ingest
	ragSystem.Clear()

	fmt.Println("\n📥 Ingesting document...")
	count, err := ragSystem.IngestFile(samplePath)
	if err != nil {
		fmt.Printf("Error ingesting: %v\n", err)
		return
	}
	fmt.Printf("   ✅ Created %d chunks\n", count)

	// Stats
	stats := ragSystem.Stats()
	fmt.Printf("\n📊 Stats: %d chunks from %d documents\n",
		stats["totalChunks"], stats["ingestedDocuments"])

	// Test queries
	testQuestions := []string{
		"What is ToonDB?",
		"How do I install ToonDB in Go?",
		"What are the main features of ToonDB?",
		"Does ToonDB support SQL?",
		"What is HNSW in ToonDB?",
	}

	fmt.Println("\n" + strings.Repeat("=", 60))
	fmt.Println("🔍 Running Test Queries")
	fmt.Println(strings.Repeat("=", 60))

	for _, question := range testQuestions {
		fmt.Printf("\n❓ Question: %s\n", question)
		fmt.Println(strings.Repeat("-", 40))

		response, err := ragSystem.Query(question)
		if err != nil {
			fmt.Printf("Error: %v\n", err)
			continue
		}

		fmt.Printf("📊 Confidence: %s\n", response.Confidence)

		answer := response.Answer
		if len(answer) > 300 {
			answer = answer[:300] + "..."
		}
		fmt.Printf("💬 Answer: %s\n", answer)

		if len(response.Sources) > 0 {
			fmt.Printf("📚 Top source score: %.3f\n", response.Sources[0].Score)
		}
	}

	fmt.Println("\n" + strings.Repeat("=", 60))
	fmt.Println("✅ Demo completed!")
	fmt.Println(strings.Repeat("=", 60))
}
