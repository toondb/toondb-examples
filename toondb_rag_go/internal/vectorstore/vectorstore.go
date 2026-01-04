// Package vectorstore provides ToonDB-based vector storage
package vectorstore

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"

	toondb "github.com/toondb/toondb-go"
	"github.com/toondb/toondb-rag-go/internal/documents"
)

// SearchResult represents a search result
type SearchResult struct {
	Chunk *documents.Chunk
	Score float32
}

// VectorStore interface
type VectorStore interface {
	Upsert(chunks []*documents.Chunk, embeddings [][]float32) error
	Search(queryEmbedding []float32, topK int) ([]SearchResult, error)
	Clear() error
	Count() int
	Close() error
}

// ToonDBVectorStore stores vectors in ToonDB
type ToonDBVectorStore struct {
	dbPath       string
	db           *toondb.Database
	chunksCache  map[string]*documents.Chunk
	vectorsCache map[string][]float32
}

// NewToonDBVectorStore creates a new vector store
func NewToonDBVectorStore(dbPath string) *ToonDBVectorStore {
	return &ToonDBVectorStore{
		dbPath:       dbPath,
		chunksCache:  make(map[string]*documents.Chunk),
		vectorsCache: make(map[string][]float32),
	}
}

// Open opens the database
func (s *ToonDBVectorStore) Open() error {
	if s.db != nil {
		return nil
	}

	db, err := toondb.Open(s.dbPath)
	if err != nil {
		return err
	}
	s.db = db
	return nil
}

// Upsert inserts or updates chunks with embeddings
func (s *ToonDBVectorStore) Upsert(chunks []*documents.Chunk, embeddings [][]float32) error {
	if len(chunks) != len(embeddings) {
		return fmt.Errorf("chunks and embeddings must have same length")
	}

	if err := s.Open(); err != nil {
		return err
	}

	for i, chunk := range chunks {
		chunkID := chunk.ID

		// Store chunk metadata as JSON
		chunkData, err := json.Marshal(map[string]interface{}{
			"content":    chunk.Content,
			"metadata":   chunk.Metadata,
			"startIndex": chunk.StartIndex,
			"endIndex":   chunk.EndIndex,
		})
		if err != nil {
			return err
		}

		if err := s.db.Put([]byte("chunks/"+chunkID), chunkData); err != nil {
			return err
		}

		// Store embedding as bytes
		vecBytes := floatsToBytes(embeddings[i])
		if err := s.db.Put([]byte("vectors/"+chunkID), vecBytes); err != nil {
			return err
		}

		// Update cache
		s.chunksCache[chunkID] = chunk
		s.vectorsCache[chunkID] = embeddings[i]
	}

	fmt.Printf("✅ Upserted %d chunks to ToonDB\n", len(chunks))
	return nil
}

// Search finds similar chunks
func (s *ToonDBVectorStore) Search(queryEmbedding []float32, topK int) ([]SearchResult, error) {
	// Load all if cache empty
	if len(s.vectorsCache) == 0 {
		if err := s.loadAll(); err != nil {
			return nil, err
		}
	}

	return searchInMemory(s.chunksCache, s.vectorsCache, queryEmbedding, topK), nil
}

func (s *ToonDBVectorStore) loadAll() error {
	if err := s.Open(); err != nil {
		return err
	}

	// Scan for chunks using v0.3.0 API - Scan takes a prefix string
	chunkResults, err := s.db.Scan("chunks/")
	if err != nil {
		return err
	}

	for _, kv := range chunkResults {
		chunkID := string(kv.Key)[7:] // Remove "chunks/"

		var data struct {
			Content    string                 `json:"content"`
			Metadata   map[string]interface{} `json:"metadata"`
			StartIndex int                    `json:"startIndex"`
			EndIndex   int                    `json:"endIndex"`
		}
		if err := json.Unmarshal(kv.Value, &data); err != nil {
			continue
		}

		chunk := &documents.Chunk{
			ID:         chunkID,
			Content:    data.Content,
			Metadata:   data.Metadata,
			StartIndex: data.StartIndex,
			EndIndex:   data.EndIndex,
		}
		s.chunksCache[chunkID] = chunk
	}

	// Scan for vectors
	vectorResults, err := s.db.Scan("vectors/")
	if err != nil {
		return err
	}

	for _, kv := range vectorResults {
		chunkID := string(kv.Key)[8:] // Remove "vectors/"
		s.vectorsCache[chunkID] = bytesToFloats(kv.Value)
	}

	return nil
}

// Clear removes all data
func (s *ToonDBVectorStore) Clear() error {
	s.chunksCache = make(map[string]*documents.Chunk)
	s.vectorsCache = make(map[string][]float32)
	return nil
}

// Count returns the number of stored chunks
func (s *ToonDBVectorStore) Count() int {
	return len(s.chunksCache)
}

// Close closes the database
func (s *ToonDBVectorStore) Close() error {
	if s.db != nil {
		return s.db.Close()
	}
	return nil
}

// InMemoryVectorStore is a pure in-memory store for testing/mock mode
type InMemoryVectorStore struct {
	chunksCache  map[string]*documents.Chunk
	vectorsCache map[string][]float32
}

// NewInMemoryVectorStore creates an in-memory vector store
func NewInMemoryVectorStore() *InMemoryVectorStore {
	return &InMemoryVectorStore{
		chunksCache:  make(map[string]*documents.Chunk),
		vectorsCache: make(map[string][]float32),
	}
}

// Upsert stores chunks in memory
func (s *InMemoryVectorStore) Upsert(chunks []*documents.Chunk, embeddings [][]float32) error {
	for i, chunk := range chunks {
		s.chunksCache[chunk.ID] = chunk
		s.vectorsCache[chunk.ID] = embeddings[i]
	}
	fmt.Printf("✅ Stored %d chunks in memory\n", len(chunks))
	return nil
}

// Search finds similar chunks
func (s *InMemoryVectorStore) Search(queryEmbedding []float32, topK int) ([]SearchResult, error) {
	return searchInMemory(s.chunksCache, s.vectorsCache, queryEmbedding, topK), nil
}

// Clear removes all data
func (s *InMemoryVectorStore) Clear() error {
	s.chunksCache = make(map[string]*documents.Chunk)
	s.vectorsCache = make(map[string][]float32)
	return nil
}

// Count returns the count
func (s *InMemoryVectorStore) Count() int {
	return len(s.chunksCache)
}

// Close is a no-op
func (s *InMemoryVectorStore) Close() error {
	return nil
}

// Shared search logic
func searchInMemory(chunks map[string]*documents.Chunk, vectors map[string][]float32, queryEmbedding []float32, topK int) []SearchResult {
	if len(vectors) == 0 {
		return []SearchResult{}
	}

	queryNorm := normalize(queryEmbedding)

	type score struct {
		chunkID    string
		similarity float32
	}
	var scores []score

	for chunkID, vector := range vectors {
		vectorNorm := normalize(vector)
		similarity := dot(queryNorm, vectorNorm)
		scores = append(scores, score{chunkID, similarity})
	}

	// Sort by similarity (descending)
	for i := 0; i < len(scores)-1; i++ {
		for j := i + 1; j < len(scores); j++ {
			if scores[j].similarity > scores[i].similarity {
				scores[i], scores[j] = scores[j], scores[i]
			}
		}
	}

	var results []SearchResult
	for i := 0; i < topK && i < len(scores); i++ {
		chunk := chunks[scores[i].chunkID]
		if chunk != nil {
			results = append(results, SearchResult{
				Chunk: chunk,
				Score: scores[i].similarity,
			})
		}
	}

	return results
}

// Helper functions
func floatsToBytes(floats []float32) []byte {
	buf := make([]byte, len(floats)*4)
	for i, f := range floats {
		binary.LittleEndian.PutUint32(buf[i*4:], math.Float32bits(f))
	}
	return buf
}

func bytesToFloats(data []byte) []float32 {
	floats := make([]float32, len(data)/4)
	for i := range floats {
		bits := binary.LittleEndian.Uint32(data[i*4:])
		floats[i] = math.Float32frombits(bits)
	}
	return floats
}

func normalize(vec []float32) []float32 {
	var sum float32
	for _, v := range vec {
		sum += v * v
	}
	norm := float32(math.Sqrt(float64(sum)))
	result := make([]float32, len(vec))
	for i, v := range vec {
		result[i] = v / norm
	}
	return result
}

func dot(a, b []float32) float32 {
	var sum float32
	for i := range a {
		sum += a[i] * b[i]
	}
	return sum
}
