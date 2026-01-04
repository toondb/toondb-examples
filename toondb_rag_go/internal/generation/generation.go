// Package generation provides LLM generation using Azure OpenAI
package generation

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"

	"github.com/toondb/toondb-rag-go/internal/config"
	"github.com/toondb/toondb-rag-go/internal/vectorstore"
)

// RAGResponse represents a RAG response
type RAGResponse struct {
	Answer     string
	Sources    []vectorstore.SearchResult
	Context    string
	Confidence string
}

// ContextAssembler assembles context from search results
type ContextAssembler struct {
	MaxLength int
}

// NewContextAssembler creates a new assembler
func NewContextAssembler(maxLength int) *ContextAssembler {
	return &ContextAssembler{MaxLength: maxLength}
}

// Assemble creates context from results
func (a *ContextAssembler) Assemble(results []vectorstore.SearchResult) string {
	var parts []string
	currentLen := 0

	for i, result := range results {
		source := "Unknown"
		if filename, ok := result.Chunk.Metadata["filename"].(string); ok {
			source = filename
		}

		text := fmt.Sprintf("[Source %d: %s]\n%s\n", i+1, source, result.Chunk.Content)

		if currentLen+len(text) > a.MaxLength {
			break
		}

		parts = append(parts, text)
		currentLen += len(text)
	}

	return strings.Join(parts, "\n")
}

// AzureLLMGenerator generates responses using Azure OpenAI
type AzureLLMGenerator struct {
	cfg       *config.AzureConfig
	client    *http.Client
	assembler *ContextAssembler
}

// NewAzureLLMGenerator creates a new generator
func NewAzureLLMGenerator(cfg *config.AzureConfig, maxContextLen int) *AzureLLMGenerator {
	return &AzureLLMGenerator{
		cfg:       cfg,
		client:    &http.Client{},
		assembler: NewContextAssembler(maxContextLen),
	}
}

type chatRequest struct {
	Messages    []message `json:"messages"`
	MaxTokens   int       `json:"max_tokens"`
	Temperature float32   `json:"temperature"`
}

type message struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type chatResponse struct {
	Choices []struct {
		Message struct {
			Content string `json:"content"`
		} `json:"message"`
	} `json:"choices"`
}

const promptTemplate = `Answer the question based on the provided context.
Cite your sources using [Source N] notation.
If the context doesn't contain enough information, say so.

Context:
%s

Question: %s

Provide a detailed answer with citations:`

// GenerateWithSources generates a response from search results
func (g *AzureLLMGenerator) GenerateWithSources(question string, results []vectorstore.SearchResult) (*RAGResponse, error) {
	// Determine confidence
	confidence := "low"
	if len(results) > 0 {
		topScore := results[0].Score
		if topScore >= 0.8 {
			confidence = "high"
		} else if topScore >= 0.5 {
			confidence = "medium"
		}
	}

	// Build context
	context := g.assembler.Assemble(results)

	// Handle low confidence
	if confidence == "low" && (len(results) == 0 || results[0].Score < 0.3) {
		return &RAGResponse{
			Answer:     "I don't have enough relevant information to answer this question confidently.",
			Sources:    results,
			Context:    context,
			Confidence: confidence,
		}, nil
	}

	// Generate
	prompt := fmt.Sprintf(promptTemplate, context, question)
	answer, err := g.generate(prompt)
	if err != nil {
		return nil, err
	}

	if confidence == "medium" {
		answer = "Based on the available information: " + answer
	}

	return &RAGResponse{
		Answer:     answer,
		Sources:    results,
		Context:    context,
		Confidence: confidence,
	}, nil
}

func (g *AzureLLMGenerator) generate(prompt string) (string, error) {
	url := fmt.Sprintf("%sopenai/deployments/%s/chat/completions?api-version=%s",
		g.cfg.Endpoint, g.cfg.ChatDeployment, g.cfg.APIVersion)

	body, _ := json.Marshal(chatRequest{
		Messages:    []message{{Role: "user", Content: prompt}},
		MaxTokens:   1000,
		Temperature: 0.1,
	})

	req, err := http.NewRequest("POST", url, bytes.NewReader(body))
	if err != nil {
		return "", err
	}

	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("api-key", g.cfg.APIKey)

	resp, err := g.client.Do(req)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()

	if resp.StatusCode != 200 {
		bodyBytes, _ := io.ReadAll(resp.Body)
		return "", fmt.Errorf("chat request failed: %s", string(bodyBytes))
	}

	var result chatResponse
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return "", err
	}

	if len(result.Choices) == 0 {
		return "", fmt.Errorf("no response from LLM")
	}

	return result.Choices[0].Message.Content, nil
}

// MockLLMGenerator for testing
type MockLLMGenerator struct {
	assembler *ContextAssembler
}

// NewMockLLMGenerator creates a mock generator
func NewMockLLMGenerator(maxContextLen int) *MockLLMGenerator {
	return &MockLLMGenerator{
		assembler: NewContextAssembler(maxContextLen),
	}
}

// GenerateWithSources generates a mock response
func (g *MockLLMGenerator) GenerateWithSources(question string, results []vectorstore.SearchResult) (*RAGResponse, error) {
	context := g.assembler.Assemble(results)
	q := strings.ToLower(question)

	answer := "I am a mock AI. "

	if strings.Contains(q, "install") {
		answer += "To install ToonDB, run `go get github.com/toondb/toondb/toondb-go`."
	} else if strings.Contains(q, "features") {
		answer += "ToonDB features include Key-Value Store, Vector Search, and SQL Support."
	} else if strings.Contains(q, "sql") {
		answer += "Yes, ToonDB supports SQL operations."
	} else if strings.Contains(q, "toondb") {
		answer += "ToonDB is a high-performance embedded database for AI applications."
	} else {
		answer += fmt.Sprintf("I found %d relevant sources.", len(results))
	}

	confidence := "low"
	if len(results) > 0 {
		confidence = "high"
	}

	return &RAGResponse{
		Answer:     answer,
		Sources:    results,
		Context:    context,
		Confidence: confidence,
	}, nil
}
