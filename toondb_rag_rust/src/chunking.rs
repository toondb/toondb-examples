//! ToonDB RAG System - Text Chunking
use crate::documents::{Chunk, Document};

/// Trait for chunking strategies
pub trait Chunker {
    fn chunk(&self, doc: &Document) -> Vec<Chunk>;
}

/// Fixed-size chunker with overlap
pub struct FixedSizeChunker {
    chunk_size: usize,
    overlap: usize,
}

impl FixedSizeChunker {
    pub fn new(chunk_size: usize, overlap: usize) -> Self {
        FixedSizeChunker { chunk_size, overlap }
    }
}

impl Chunker for FixedSizeChunker {
    fn chunk(&self, doc: &Document) -> Vec<Chunk> {
        let text = &doc.content;
        let chars: Vec<char> = text.chars().collect();
        let mut chunks = Vec::new();
        let mut start = 0;

        while start < chars.len() {
            let end = (start + self.chunk_size).min(chars.len());
            let chunk_text: String = chars[start..end].iter().collect();

            let mut metadata = doc.metadata.clone();
            metadata.insert("chunk_index".to_string(), chunks.len().to_string());
            metadata.insert("doc_id".to_string(), doc.id.clone());

            chunks.push(Chunk::new(
                chunk_text.trim().to_string(),
                metadata,
                start,
                end,
            ));

            let next_start = if end >= chars.len() {
                break;
            } else {
                (end - self.overlap).max(start + 1)
            };

            start = next_start;
        }

        chunks
    }
}

/// Semantic chunker based on paragraphs
pub struct SemanticChunker {
    max_chunk_size: usize,
    min_chunk_size: usize,
}

impl SemanticChunker {
    pub fn new(max_chunk_size: usize, min_chunk_size: usize) -> Self {
        SemanticChunker {
            max_chunk_size,
            min_chunk_size,
        }
    }
}

impl Chunker for SemanticChunker {
    fn chunk(&self, doc: &Document) -> Vec<Chunk> {
        let paragraphs: Vec<&str> = doc.content.split("\n\n").collect();
        let mut chunks = Vec::new();
        let mut current_chunk = String::new();
        let mut current_start = 0;

        for para in paragraphs {
            let para = para.trim();
            if para.is_empty() {
                continue;
            }

            if current_chunk.len() + para.len() + 2 <= self.max_chunk_size {
                current_chunk.push_str(para);
                current_chunk.push_str("\n\n");
            } else {
                if current_chunk.len() >= self.min_chunk_size {
                    let mut metadata = doc.metadata.clone();
                    metadata.insert("chunk_index".to_string(), chunks.len().to_string());
                    metadata.insert("doc_id".to_string(), doc.id.clone());

                    chunks.push(Chunk::new(
                        current_chunk.trim().to_string(),
                        metadata,
                        current_start,
                        current_start + current_chunk.len(),
                    ));
                }
                current_start += current_chunk.len();
                current_chunk = format!("{}\n\n", para);
            }
        }

        // Don't forget the last chunk
        if !current_chunk.trim().is_empty() && current_chunk.trim().len() >= self.min_chunk_size {
            let mut metadata = doc.metadata.clone();
            metadata.insert("chunk_index".to_string(), chunks.len().to_string());
            metadata.insert("doc_id".to_string(), doc.id.clone());

            chunks.push(Chunk::new(
                current_chunk.trim().to_string(),
                metadata,
                current_start,
                current_start + current_chunk.len(),
            ));
        }

        chunks
    }
}

/// Get chunker by strategy name
pub fn get_chunker(strategy: &str, chunk_size: usize, overlap: usize) -> Box<dyn Chunker> {
    match strategy {
        "fixed" => Box::new(FixedSizeChunker::new(chunk_size, overlap)),
        _ => Box::new(SemanticChunker::new(chunk_size, chunk_size / 4)),
    }
}
