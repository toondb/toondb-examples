// Package embeddings provides Azure OpenAI embeddings
package embeddings

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"net/http"

	"github.com/toondb/toondb-examples/toondb_rag_go/internal/config"
)

// Embedder interface for embedding models
type Embedder interface {
	Embed(texts []string) ([][]float32, error)
	EmbedQuery(query string) ([]float32, error)
	Dimension() int
}

// AzureEmbeddings uses Azure OpenAI for embeddings
type AzureEmbeddings struct {
	cfg       *config.AzureConfig
	client    *http.Client
	dimension int
}

// NewAzureEmbeddings creates a new Azure embeddings client
func NewAzureEmbeddings(cfg *config.AzureConfig) *AzureEmbeddings {
	return &AzureEmbeddings{
		cfg:       cfg,
		client:    &http.Client{},
		dimension: 1536,
	}
}

type embeddingRequest struct {
	Input []string `json:"input"`
}

type embeddingResponse struct {
	Data []struct {
		Embedding []float32 `json:"embedding"`
	} `json:"data"`
}

// Embed embeds a list of texts
func (e *AzureEmbeddings) Embed(texts []string) ([][]float32, error) {
	if len(texts) == 0 {
		return [][]float32{}, nil
	}

	url := fmt.Sprintf("%sopenai/deployments/%s/embeddings?api-version=%s",
		e.cfg.Endpoint, e.cfg.EmbeddingDeployment, e.cfg.APIVersion)

	body, _ := json.Marshal(embeddingRequest{Input: texts})

	req, err := http.NewRequest("POST", url, bytes.NewReader(body))
	if err != nil {
		return nil, err
	}

	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("api-key", e.cfg.APIKey)

	resp, err := e.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != 200 {
		bodyBytes, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("embedding request failed: %s", string(bodyBytes))
	}

	var result embeddingResponse
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, err
	}

	embeddings := make([][]float32, len(result.Data))
	for i, d := range result.Data {
		embeddings[i] = d.Embedding
	}

	return embeddings, nil
}

// EmbedQuery embeds a single query
func (e *AzureEmbeddings) EmbedQuery(query string) ([]float32, error) {
	embeddings, err := e.Embed([]string{query})
	if err != nil {
		return nil, err
	}
	return embeddings[0], nil
}

// Dimension returns the embedding dimension
func (e *AzureEmbeddings) Dimension() int {
	return e.dimension
}

// MockEmbeddings for testing
type MockEmbeddings struct {
	dimension int
}

// NewMockEmbeddings creates mock embeddings
func NewMockEmbeddings() *MockEmbeddings {
	return &MockEmbeddings{dimension: 1536}
}

// Embed returns mock embeddings
func (m *MockEmbeddings) Embed(texts []string) ([][]float32, error) {
	embeddings := make([][]float32, len(texts))
	for i, text := range texts {
		emb := make([]float32, m.dimension)
		for j := 0; j < m.dimension; j++ {
			emb[j] = float32(math.Sin(float64(len(text)+j))*0.5 + 0.5)
		}
		embeddings[i] = emb
	}
	return embeddings, nil
}

// EmbedQuery returns a mock embedding
func (m *MockEmbeddings) EmbedQuery(query string) ([]float32, error) {
	embeddings, _ := m.Embed([]string{query})
	return embeddings[0], nil
}

// Dimension returns the dimension
func (m *MockEmbeddings) Dimension() int {
	return m.dimension
}
