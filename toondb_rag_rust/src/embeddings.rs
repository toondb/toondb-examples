//! ToonDB RAG System - Embeddings
use crate::config::AzureConfig;
use anyhow::Result;
use serde::{Deserialize, Serialize};

/// Trait for embedding models
#[async_trait::async_trait]
pub trait Embedder: Send + Sync {
    async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>>;
    async fn embed_query(&self, query: &str) -> Result<Vec<f32>>;
    fn dimension(&self) -> usize;
}

/// Azure OpenAI Embeddings
pub struct AzureEmbeddings {
    client: reqwest::Client,
    config: AzureConfig,
    dimension: usize,
}

#[derive(Serialize)]
struct EmbeddingRequest {
    input: Vec<String>,
}

#[derive(Deserialize)]
struct EmbeddingResponse {
    data: Vec<EmbeddingData>,
}

#[derive(Deserialize)]
struct EmbeddingData {
    embedding: Vec<f32>,
}

impl AzureEmbeddings {
    pub fn new(config: AzureConfig) -> Self {
        AzureEmbeddings {
            client: reqwest::Client::new(),
            config,
            dimension: 1536,
        }
    }
}

#[async_trait::async_trait]
impl Embedder for AzureEmbeddings {
    async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(vec![]);
        }

        let url = format!(
            "{}openai/deployments/{}/embeddings?api-version={}",
            self.config.endpoint, self.config.embedding_deployment, self.config.api_version
        );

        let request = EmbeddingRequest {
            input: texts.to_vec(),
        };

        let response = self
            .client
            .post(&url)
            .header("Content-Type", "application/json")
            .header("api-key", &self.config.api_key)
            .json(&request)
            .send()
            .await?;

        if !response.status().is_success() {
            let error = response.text().await?;
            anyhow::bail!("Embedding request failed: {}", error);
        }

        let result: EmbeddingResponse = response.json().await?;
        Ok(result.data.into_iter().map(|d| d.embedding).collect())
    }

    async fn embed_query(&self, query: &str) -> Result<Vec<f32>> {
        let embeddings = self.embed(&[query.to_string()]).await?;
        Ok(embeddings.into_iter().next().unwrap_or_default())
    }

    fn dimension(&self) -> usize {
        self.dimension
    }
}

/// Mock embeddings for testing
pub struct MockEmbeddings {
    dimension: usize,
}

impl MockEmbeddings {
    pub fn new() -> Self {
        MockEmbeddings { dimension: 1536 }
    }
}

#[async_trait::async_trait]
impl Embedder for MockEmbeddings {
    async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        Ok(texts
            .iter()
            .map(|text| {
                (0..self.dimension)
                    .map(|i| ((text.len() + i) as f32).sin() * 0.5 + 0.5)
                    .collect()
            })
            .collect())
    }

    async fn embed_query(&self, query: &str) -> Result<Vec<f32>> {
        let embeddings = self.embed(&[query.to_string()]).await?;
        Ok(embeddings.into_iter().next().unwrap_or_default())
    }

    fn dimension(&self) -> usize {
        self.dimension
    }
}
