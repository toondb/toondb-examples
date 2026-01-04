// Package rag provides the main RAG system
package rag

import (
	"fmt"

	"github.com/toondb/toondb-rag-go/internal/chunking"
	"github.com/toondb/toondb-rag-go/internal/config"
	"github.com/toondb/toondb-rag-go/internal/documents"
	"github.com/toondb/toondb-rag-go/internal/embeddings"
	"github.com/toondb/toondb-rag-go/internal/generation"
	"github.com/toondb/toondb-rag-go/internal/vectorstore"
)

// Generator interface for LLM generation
type Generator interface {
	GenerateWithSources(question string, results []vectorstore.SearchResult) (*generation.RAGResponse, error)
}

// ToonDBRAG is the main RAG system
type ToonDBRAG struct {
	loader       *documents.DocumentLoader
	preprocessor *documents.TextPreprocessor
	chunker      chunking.Chunker
	embedder     embeddings.Embedder
	vectorStore  vectorstore.VectorStore // Use interface
	generator    Generator
	topK         int
	ingestedDocs []string
}

// Options for creating a RAG system
type Options struct {
	DBPath           string
	ChunkingStrategy string
	UseMock          bool
}

// New creates a new RAG system
func New(cfg *config.Config, opts Options) *ToonDBRAG {
	dbPath := opts.DBPath
	if dbPath == "" {
		dbPath = cfg.ToonDB.Path
	}

	var embedder embeddings.Embedder
	var gen Generator
	var store vectorstore.VectorStore

	if opts.UseMock {
		embedder = embeddings.NewMockEmbeddings()
		gen = generation.NewMockLLMGenerator(cfg.RAG.MaxContextLength)
		store = vectorstore.NewInMemoryVectorStore()
	} else {
		embedder = embeddings.NewAzureEmbeddings(&cfg.Azure)
		gen = generation.NewAzureLLMGenerator(&cfg.Azure, cfg.RAG.MaxContextLength)
		// Use ToonDB for persistent storage
		store = vectorstore.NewToonDBVectorStore(dbPath)
	}

	return &ToonDBRAG{
		loader:       documents.NewDocumentLoader(),
		preprocessor: documents.NewTextPreprocessor(),
		chunker:      chunking.GetChunker(opts.ChunkingStrategy, cfg.RAG.ChunkSize, cfg.RAG.ChunkOverlap),
		embedder:     embedder,
		vectorStore:  store,
		generator:    gen,
		topK:         cfg.RAG.TopK,
	}
}

// Ingest ingests documents into the RAG system
func (r *ToonDBRAG) Ingest(docs []*documents.Document) (int, error) {
	var allChunks []*documents.Chunk

	for _, doc := range docs {
		// Preprocess
		doc.Content = r.preprocessor.Clean(doc.Content)

		// Chunk
		chunks := r.chunker.Chunk(doc)
		allChunks = append(allChunks, chunks...)

		if filename, ok := doc.Metadata["filename"].(string); ok {
			r.ingestedDocs = append(r.ingestedDocs, filename)
		} else {
			r.ingestedDocs = append(r.ingestedDocs, doc.ID)
		}
	}

	if len(allChunks) == 0 {
		fmt.Println("⚠️ No chunks generated from documents")
		return 0, nil
	}

	// Embed
	fmt.Printf("🔄 Embedding %d chunks...\n", len(allChunks))
	texts := make([]string, len(allChunks))
	for i, chunk := range allChunks {
		texts[i] = chunk.Content
	}

	embs, err := r.embedder.Embed(texts)
	if err != nil {
		return 0, err
	}

	// Store
	if err := r.vectorStore.Upsert(allChunks, embs); err != nil {
		return 0, err
	}

	fmt.Printf("✅ Ingested %d documents (%d chunks)\n", len(docs), len(allChunks))
	return len(allChunks), nil
}

// IngestFile ingests a single file
func (r *ToonDBRAG) IngestFile(path string) (int, error) {
	doc, err := r.loader.Load(path)
	if err != nil {
		return 0, err
	}
	return r.Ingest([]*documents.Document{doc})
}

// IngestDirectory ingests all files from a directory
func (r *ToonDBRAG) IngestDirectory(path string, extensions []string) (int, error) {
	docs, err := r.loader.LoadDirectory(path, extensions)
	if err != nil {
		return 0, err
	}
	return r.Ingest(docs)
}

// Query queries the RAG system
func (r *ToonDBRAG) Query(question string) (*generation.RAGResponse, error) {
	// Get embedding
	queryEmb, err := r.embedder.EmbedQuery(question)
	if err != nil {
		return nil, err
	}

	// Search
	results, err := r.vectorStore.Search(queryEmb, r.topK)
	if err != nil {
		return nil, err
	}

	// Generate
	return r.generator.GenerateWithSources(question, results)
}

// Search searches for similar chunks
func (r *ToonDBRAG) Search(query string, topK int) ([]vectorstore.SearchResult, error) {
	queryEmb, err := r.embedder.EmbedQuery(query)
	if err != nil {
		return nil, err
	}

	if topK == 0 {
		topK = r.topK
	}

	return r.vectorStore.Search(queryEmb, topK)
}

// Stats returns system statistics
func (r *ToonDBRAG) Stats() map[string]interface{} {
	return map[string]interface{}{
		"totalChunks":       r.vectorStore.Count(),
		"ingestedDocuments": len(r.ingestedDocs),
		"documentNames":     r.ingestedDocs,
	}
}

// Clear clears all data
func (r *ToonDBRAG) Clear() error {
	r.ingestedDocs = nil
	fmt.Println("🗑️ Cleared all data")
	return r.vectorStore.Clear()
}

// Close closes connections
func (r *ToonDBRAG) Close() error {
	return r.vectorStore.Close()
}
