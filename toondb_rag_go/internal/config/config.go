// Package config provides configuration management
package config

import (
	"os"
	"strconv"

	"github.com/joho/godotenv"
)

// Config holds all configuration
type Config struct {
	Azure  AzureConfig
	ToonDB ToonDBConfig
	RAG    RAGConfig
}

// AzureConfig holds Azure OpenAI settings
type AzureConfig struct {
	APIKey              string
	Endpoint            string
	APIVersion          string
	ChatDeployment      string
	EmbeddingDeployment string
}

// ToonDBConfig holds ToonDB settings
type ToonDBConfig struct {
	Path string
}

// RAGConfig holds RAG settings
type RAGConfig struct {
	ChunkSize        int
	ChunkOverlap     int
	TopK             int
	MaxContextLength int
}

// Load loads configuration from environment
func Load() (*Config, error) {
	_ = godotenv.Load() // Ignore error if .env doesn't exist

	return &Config{
		Azure: AzureConfig{
			APIKey:              getEnv("AZURE_OPENAI_API_KEY", ""),
			Endpoint:            getEnv("AZURE_OPENAI_ENDPOINT", ""),
			APIVersion:          getEnv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview"),
			ChatDeployment:      getEnv("AZURE_OPENAI_CHAT_DEPLOYMENT", "gpt-4.1"),
			EmbeddingDeployment: getEnv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-3-small"),
		},
		ToonDB: ToonDBConfig{
			Path: getEnv("TOONDB_PATH", "./toondb_data"),
		},
		RAG: RAGConfig{
			ChunkSize:        getEnvInt("CHUNK_SIZE", 512),
			ChunkOverlap:     getEnvInt("CHUNK_OVERLAP", 50),
			TopK:             getEnvInt("TOP_K", 5),
			MaxContextLength: getEnvInt("MAX_CONTEXT_LENGTH", 4000),
		},
	}, nil
}

func getEnv(key, defaultVal string) string {
	if val := os.Getenv(key); val != "" {
		return val
	}
	return defaultVal
}

func getEnvInt(key string, defaultVal int) int {
	if val := os.Getenv(key); val != "" {
		if i, err := strconv.Atoi(val); err == nil {
			return i
		}
	}
	return defaultVal
}
