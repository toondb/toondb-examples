// Package chunking provides text chunking strategies
package chunking

import (
	"strings"

	"github.com/toondb/toondb-examples/toondb_rag_go/internal/documents"
)

// Chunker interface for chunking strategies
type Chunker interface {
	Chunk(doc *documents.Document) []*documents.Chunk
}

// FixedSizeChunker chunks by fixed size with overlap
type FixedSizeChunker struct {
	ChunkSize int
	Overlap   int
}

// NewFixedSizeChunker creates a new fixed size chunker
func NewFixedSizeChunker(chunkSize, overlap int) *FixedSizeChunker {
	return &FixedSizeChunker{
		ChunkSize: chunkSize,
		Overlap:   overlap,
	}
}

// Chunk splits a document into fixed-size chunks
func (c *FixedSizeChunker) Chunk(doc *documents.Document) []*documents.Chunk {
	text := doc.Content
	var chunks []*documents.Chunk
	start := 0

	for start < len(text) {
		end := start + c.ChunkSize
		if end > len(text) {
			end = len(text)
		}

		chunkText := text[start:end]

		// Try to break at word boundary
		if end < len(text) && text[end] != ' ' {
			lastSpace := strings.LastIndex(chunkText, " ")
			if lastSpace > c.ChunkSize/2 {
				end = start + lastSpace
				chunkText = text[start:end]
			}
		}

		metadata := make(map[string]interface{})
		for k, v := range doc.Metadata {
			metadata[k] = v
		}
		metadata["chunkIndex"] = len(chunks)
		metadata["docId"] = doc.ID

		chunks = append(chunks, documents.NewChunk(
			strings.TrimSpace(chunkText),
			metadata,
			start,
			end,
		))

		nextStart := end - c.Overlap
		if nextStart <= start {
			nextStart = end
		}
		if end >= len(text) {
			break
		}
		start = nextStart
	}

	return chunks
}

// SemanticChunker chunks by semantic boundaries (paragraphs)
type SemanticChunker struct {
	MaxChunkSize int
	MinChunkSize int
}

// NewSemanticChunker creates a new semantic chunker
func NewSemanticChunker(maxSize, minSize int) *SemanticChunker {
	return &SemanticChunker{
		MaxChunkSize: maxSize,
		MinChunkSize: minSize,
	}
}

// Chunk splits a document by paragraphs
func (c *SemanticChunker) Chunk(doc *documents.Document) []*documents.Chunk {
	paragraphs := strings.Split(doc.Content, "\n\n")
	var chunks []*documents.Chunk
	currentChunk := ""
	currentStart := 0

	for _, para := range paragraphs {
		para = strings.TrimSpace(para)
		if para == "" {
			continue
		}

		if len(currentChunk)+len(para)+2 <= c.MaxChunkSize {
			currentChunk += para + "\n\n"
		} else {
			if len(currentChunk) >= c.MinChunkSize {
				metadata := make(map[string]interface{})
				for k, v := range doc.Metadata {
					metadata[k] = v
				}
				metadata["chunkIndex"] = len(chunks)
				metadata["docId"] = doc.ID

				chunks = append(chunks, documents.NewChunk(
					strings.TrimSpace(currentChunk),
					metadata,
					currentStart,
					currentStart+len(currentChunk),
				))
			}
			currentStart += len(currentChunk)
			currentChunk = para + "\n\n"
		}
	}

	// Don't forget the last chunk
	if strings.TrimSpace(currentChunk) != "" && len(strings.TrimSpace(currentChunk)) >= c.MinChunkSize {
		metadata := make(map[string]interface{})
		for k, v := range doc.Metadata {
			metadata[k] = v
		}
		metadata["chunkIndex"] = len(chunks)
		metadata["docId"] = doc.ID

		chunks = append(chunks, documents.NewChunk(
			strings.TrimSpace(currentChunk),
			metadata,
			currentStart,
			currentStart+len(currentChunk),
		))
	}

	return chunks
}

// GetChunker returns a chunker by strategy name
func GetChunker(strategy string, chunkSize, overlap int) Chunker {
	switch strategy {
	case "fixed":
		return NewFixedSizeChunker(chunkSize, overlap)
	case "semantic":
		return NewSemanticChunker(chunkSize, chunkSize/4)
	default:
		return NewSemanticChunker(chunkSize, chunkSize/4)
	}
}
