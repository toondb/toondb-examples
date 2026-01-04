//! ToonDB RAG System - Configuration
use std::env;

#[derive(Clone, Debug)]
pub struct Config {
    pub azure: AzureConfig,
    pub toondb: ToonDBConfig,
    pub rag: RAGConfig,
}

#[derive(Clone, Debug)]
pub struct AzureConfig {
    pub api_key: String,
    pub endpoint: String,
    pub api_version: String,
    pub chat_deployment: String,
    pub embedding_deployment: String,
}

#[derive(Clone, Debug)]
pub struct ToonDBConfig {
    pub path: String,
}

#[derive(Clone, Debug)]
pub struct RAGConfig {
    pub chunk_size: usize,
    pub chunk_overlap: usize,
    pub top_k: usize,
    pub max_context_length: usize,
}

impl Config {
    pub fn load() -> Self {
        dotenv::dotenv().ok();

        Config {
            azure: AzureConfig {
                api_key: env::var("AZURE_OPENAI_API_KEY").unwrap_or_default(),
                endpoint: env::var("AZURE_OPENAI_ENDPOINT").unwrap_or_default(),
                api_version: env::var("AZURE_OPENAI_API_VERSION")
                    .unwrap_or_else(|_| "2024-12-01-preview".to_string()),
                chat_deployment: env::var("AZURE_OPENAI_CHAT_DEPLOYMENT")
                    .unwrap_or_else(|_| "gpt-4.1".to_string()),
                embedding_deployment: env::var("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")
                    .unwrap_or_else(|_| "text-embedding-3-small".to_string()),
            },
            toondb: ToonDBConfig {
                path: env::var("TOONDB_PATH").unwrap_or_else(|_| "./toondb_data".to_string()),
            },
            rag: RAGConfig {
                chunk_size: env::var("CHUNK_SIZE")
                    .unwrap_or_else(|_| "512".to_string())
                    .parse()
                    .unwrap_or(512),
                chunk_overlap: env::var("CHUNK_OVERLAP")
                    .unwrap_or_else(|_| "50".to_string())
                    .parse()
                    .unwrap_or(50),
                top_k: env::var("TOP_K")
                    .unwrap_or_else(|_| "5".to_string())
                    .parse()
                    .unwrap_or(5),
                max_context_length: env::var("MAX_CONTEXT_LENGTH")
                    .unwrap_or_else(|_| "4000".to_string())
                    .parse()
                    .unwrap_or(4000),
            },
        }
    }
}
