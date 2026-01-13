//! ToonDB RAG System - Vector Store
use crate::documents::Chunk;
use anyhow::Result;
use std::collections::HashMap;

/// Search result
#[derive(Clone, Debug)]
pub struct SearchResult {
    pub chunk: Chunk,
    pub score: f32,
}

/// Trait for vector stores
pub trait VectorStore: Send + Sync {
    fn upsert(&mut self, chunks: &[Chunk], embeddings: &[Vec<f32>]) -> Result<()>;
    fn search(&self, query_embedding: &[f32], top_k: usize) -> Result<Vec<SearchResult>>;
    fn clear(&mut self) -> Result<()>;
    fn count(&self) -> usize;
}

/// In-memory vector store (for demo/mock mode)
pub struct InMemoryVectorStore {
    chunks: HashMap<String, Chunk>,
    vectors: HashMap<String, Vec<f32>>,
}

impl InMemoryVectorStore {
    pub fn new() -> Self {
        InMemoryVectorStore {
            chunks: HashMap::new(),
            vectors: HashMap::new(),
        }
    }
}

impl VectorStore for InMemoryVectorStore {
    fn upsert(&mut self, chunks: &[Chunk], embeddings: &[Vec<f32>]) -> Result<()> {
        for (chunk, embedding) in chunks.iter().zip(embeddings.iter()) {
            self.chunks.insert(chunk.id.clone(), chunk.clone());
            self.vectors.insert(chunk.id.clone(), embedding.clone());
        }
        println!("✅ Stored {} chunks in memory", chunks.len());
        Ok(())
    }

    fn search(&self, query_embedding: &[f32], top_k: usize) -> Result<Vec<SearchResult>> {
        if self.vectors.is_empty() {
            return Ok(vec![]);
        }

        let query_norm = normalize(query_embedding);

        let mut scores: Vec<(String, f32)> = self
            .vectors
            .iter()
            .map(|(id, vec)| {
                let vec_norm = normalize(vec);
                let similarity = dot(&query_norm, &vec_norm);
                (id.clone(), similarity)
            })
            .collect();

        // Sort by similarity descending
        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let results: Vec<SearchResult> = scores
            .into_iter()
            .take(top_k)
            .filter_map(|(id, score)| {
                self.chunks.get(&id).map(|chunk| SearchResult {
                    chunk: chunk.clone(),
                    score,
                })
            })
            .collect();

        Ok(results)
    }

    fn clear(&mut self) -> Result<()> {
        self.chunks.clear();
        self.vectors.clear();
        Ok(())
    }

    fn count(&self) -> usize {
        self.chunks.len()
    }
}

// Helper functions
fn normalize(vec: &[f32]) -> Vec<f32> {
    let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm == 0.0 {
        vec.to_vec()
    } else {
        vec.iter().map(|x| x / norm).collect()
    }
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}
