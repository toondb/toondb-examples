//! ToonDB RAG System - Main RAG Module
use crate::chunking::{get_chunker, Chunker};
use crate::config::Config;
use crate::documents::{Document, DocumentLoader, TextPreprocessor};
use crate::embeddings::{AzureEmbeddings, Embedder, MockEmbeddings};
use crate::generation::{AzureLLMGenerator, Generator, MockLLMGenerator, RAGResponse};
use crate::vectorstore::{InMemoryVectorStore, VectorStore};
use anyhow::Result;

/// RAG system options
pub struct RAGOptions {
    pub db_path: Option<String>,
    pub chunking_strategy: String,
    pub use_mock: bool,
}

impl Default for RAGOptions {
    fn default() -> Self {
        RAGOptions {
            db_path: None,
            chunking_strategy: "semantic".to_string(),
            use_mock: true,
        }
    }
}

/// ToonDB RAG System
pub struct ToonDBRAG {
    loader: DocumentLoader,
    preprocessor: TextPreprocessor,
    chunker: Box<dyn Chunker>,
    embedder: Box<dyn Embedder>,
    vector_store: Box<dyn VectorStore>,
    generator: Box<dyn Generator>,
    top_k: usize,
    ingested_docs: Vec<String>,
}

impl ToonDBRAG {
    pub fn new(config: &Config, options: RAGOptions) -> Self {
        let embedder: Box<dyn Embedder>;
        let generator: Box<dyn Generator>;
        let vector_store: Box<dyn VectorStore> = Box::new(InMemoryVectorStore::new());

        if options.use_mock {
            embedder = Box::new(MockEmbeddings::new());
            generator = Box::new(MockLLMGenerator::new(config.rag.max_context_length));
        } else {
            // Use real Azure OpenAI
            embedder = Box::new(AzureEmbeddings::new(config.azure.clone()));
            generator = Box::new(AzureLLMGenerator::new(
                config.azure.clone(),
                config.rag.max_context_length,
            ));
        }

        ToonDBRAG {
            loader: DocumentLoader::new(),
            preprocessor: TextPreprocessor::new(),
            chunker: get_chunker(
                &options.chunking_strategy,
                config.rag.chunk_size,
                config.rag.chunk_overlap,
            ),
            embedder,
            vector_store,
            generator,
            top_k: config.rag.top_k,
            ingested_docs: Vec::new(),
        }
    }

    /// Ingest documents
    pub async fn ingest(&mut self, documents: Vec<Document>) -> Result<usize> {
        let mut all_chunks = Vec::new();

        for mut doc in documents {
            doc.content = self.preprocessor.clean(&doc.content);
            let chunks = self.chunker.chunk(&doc);
            all_chunks.extend(chunks);

            if let Some(filename) = doc.metadata.get("filename") {
                self.ingested_docs.push(filename.clone());
            } else {
                self.ingested_docs.push(doc.id.clone());
            }
        }

        if all_chunks.is_empty() {
            println!("⚠️ No chunks generated from documents");
            return Ok(0);
        }

        println!("🔄 Embedding {} chunks...", all_chunks.len());
        let texts: Vec<String> = all_chunks.iter().map(|c| c.content.clone()).collect();
        let embeddings = self.embedder.embed(&texts).await?;

        self.vector_store.upsert(&all_chunks, &embeddings)?;

        println!(
            "✅ Ingested {} documents ({} chunks)",
            self.ingested_docs.len(),
            all_chunks.len()
        );
        Ok(all_chunks.len())
    }

    /// Ingest a single file
    pub async fn ingest_file(&mut self, path: &str) -> Result<usize> {
        let doc = self.loader.load(path)?;
        self.ingest(vec![doc]).await
    }

    /// Query the RAG system
    pub async fn query(&self, question: &str) -> Result<RAGResponse> {
        let query_embedding = self.embedder.embed_query(question).await?;
        let results = self.vector_store.search(&query_embedding, self.top_k)?;
        self.generator.generate_with_sources(question, results).await
    }

    /// Clear all data
    pub fn clear(&mut self) -> Result<()> {
        self.vector_store.clear()?;
        self.ingested_docs.clear();
        println!("🗑️ Cleared all data");
        Ok(())
    }

    /// Get stats
    pub fn stats(&self) -> (usize, usize) {
        (self.vector_store.count(), self.ingested_docs.len())
    }
}
