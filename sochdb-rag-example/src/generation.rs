//! ToonDB RAG System - LLM Generation
use crate::config::AzureConfig;
use crate::vectorstore::SearchResult;
use anyhow::Result;
use serde::{Deserialize, Serialize};

/// RAG Response
#[derive(Debug)]
pub struct RAGResponse {
    pub answer: String,
    pub sources: Vec<SearchResult>,
    pub context: String,
    pub confidence: String,
}

/// Trait for generators
#[async_trait::async_trait]
pub trait Generator: Send + Sync {
    async fn generate_with_sources(
        &self,
        question: &str,
        results: Vec<SearchResult>,
    ) -> Result<RAGResponse>;
}

/// Context assembler
pub struct ContextAssembler {
    max_length: usize,
}

impl ContextAssembler {
    pub fn new(max_length: usize) -> Self {
        ContextAssembler { max_length }
    }

    pub fn assemble(&self, results: &[SearchResult]) -> String {
        let mut parts = Vec::new();
        let mut current_len = 0;

        for (i, result) in results.iter().enumerate() {
            let source = result
                .chunk
                .metadata
                .get("filename")
                .cloned()
                .unwrap_or_else(|| "Unknown".to_string());

            let text = format!("[Source {}: {}]\n{}\n", i + 1, source, result.chunk.content);

            if current_len + text.len() > self.max_length {
                break;
            }

            parts.push(text);
            current_len += parts.last().unwrap().len();
        }

        parts.join("\n")
    }
}

/// Azure LLM Generator
pub struct AzureLLMGenerator {
    client: reqwest::Client,
    config: AzureConfig,
    assembler: ContextAssembler,
}

#[derive(Serialize)]
struct ChatRequest {
    messages: Vec<ChatMessage>,
    max_tokens: u32,
    temperature: f32,
}

#[derive(Serialize)]
struct ChatMessage {
    role: String,
    content: String,
}

#[derive(Deserialize)]
struct ChatResponse {
    choices: Vec<ChatChoice>,
}

#[derive(Deserialize)]
struct ChatChoice {
    message: ChatResponseMessage,
}

#[derive(Deserialize)]
struct ChatResponseMessage {
    content: String,
}

const PROMPT_TEMPLATE: &str = r#"Answer the question based on the provided context.
Cite your sources using [Source N] notation.
If the context doesn't contain enough information, say so.

Context:
{context}

Question: {question}

Provide a detailed answer with citations:"#;

impl AzureLLMGenerator {
    pub fn new(config: AzureConfig, max_context_len: usize) -> Self {
        AzureLLMGenerator {
            client: reqwest::Client::new(),
            config,
            assembler: ContextAssembler::new(max_context_len),
        }
    }

    async fn generate(&self, prompt: &str) -> Result<String> {
        let url = format!(
            "{}openai/deployments/{}/chat/completions?api-version={}",
            self.config.endpoint, self.config.chat_deployment, self.config.api_version
        );

        let request = ChatRequest {
            messages: vec![ChatMessage {
                role: "user".to_string(),
                content: prompt.to_string(),
            }],
            max_tokens: 1000,
            temperature: 0.1,
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
            anyhow::bail!("Chat request failed: {}", error);
        }

        let result: ChatResponse = response.json().await?;
        Ok(result
            .choices
            .first()
            .map(|c| c.message.content.clone())
            .unwrap_or_default())
    }
}

#[async_trait::async_trait]
impl Generator for AzureLLMGenerator {
    async fn generate_with_sources(
        &self,
        question: &str,
        results: Vec<SearchResult>,
    ) -> Result<RAGResponse> {
        let confidence = if results.is_empty() {
            "low"
        } else if results[0].score >= 0.8 {
            "high"
        } else if results[0].score >= 0.5 {
            "medium"
        } else {
            "low"
        };

        let context = self.assembler.assemble(&results);

        if confidence == "low" && (results.is_empty() || results[0].score < 0.3) {
            return Ok(RAGResponse {
                answer: "I don't have enough relevant information to answer this question confidently.".to_string(),
                sources: results,
                context,
                confidence: confidence.to_string(),
            });
        }

        let prompt = PROMPT_TEMPLATE
            .replace("{context}", &context)
            .replace("{question}", question);

        let mut answer = self.generate(&prompt).await?;

        if confidence == "medium" {
            answer = format!("Based on the available information: {}", answer);
        }

        Ok(RAGResponse {
            answer,
            sources: results,
            context,
            confidence: confidence.to_string(),
        })
    }
}

/// Mock generator for testing
pub struct MockLLMGenerator {
    assembler: ContextAssembler,
}

impl MockLLMGenerator {
    pub fn new(max_context_len: usize) -> Self {
        MockLLMGenerator {
            assembler: ContextAssembler::new(max_context_len),
        }
    }
}

#[async_trait::async_trait]
impl Generator for MockLLMGenerator {
    async fn generate_with_sources(
        &self,
        question: &str,
        results: Vec<SearchResult>,
    ) -> Result<RAGResponse> {
        let context = self.assembler.assemble(&results);
        let q = question.to_lowercase();

        let mut answer = "I am a mock AI. ".to_string();

        if q.contains("install") {
            answer.push_str("Add `toondb = \"0.3\"` to your Cargo.toml.");
        } else if q.contains("features") {
            answer.push_str("ToonDB features include Key-Value Store, Vector Search, and SQL Support.");
        } else if q.contains("sql") {
            answer.push_str("Yes, ToonDB supports SQL operations.");
        } else if q.contains("toondb") {
            answer.push_str("ToonDB is a high-performance embedded database for AI applications.");
        } else {
            answer.push_str(&format!("I found {} relevant sources.", results.len()));
        }

        let confidence = if results.is_empty() { "low" } else { "high" }.to_string();

        Ok(RAGResponse {
            answer,
            sources: results,
            context,
            confidence,
        })
    }
}
