//! ToonDB RAG System - Document Models
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::Path;

/// Represents a loaded document
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Document {
    pub id: String,
    pub content: String,
    pub metadata: HashMap<String, String>,
}

impl Document {
    pub fn new(content: String, metadata: HashMap<String, String>) -> Self {
        let id = format!("{:x}", md5::compute(&content));
        Document { id, content, metadata }
    }
}

/// Represents a chunk of a document
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Chunk {
    pub id: String,
    pub content: String,
    pub metadata: HashMap<String, String>,
    pub start_index: usize,
    pub end_index: usize,
}

impl Chunk {
    pub fn new(
        content: String,
        metadata: HashMap<String, String>,
        start_index: usize,
        end_index: usize,
    ) -> Self {
        let id = format!("{:x}", md5::compute(format!("{}{}", content, start_index)));
        Chunk {
            id,
            content,
            metadata,
            start_index,
            end_index,
        }
    }
}

/// Document loader
pub struct DocumentLoader;

impl DocumentLoader {
    pub fn new() -> Self {
        DocumentLoader
    }

    pub fn load(&self, path: &str) -> anyhow::Result<Document> {
        let content = fs::read_to_string(path)?;
        let filename = Path::new(path)
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("unknown")
            .to_string();

        let ext = Path::new(path)
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("")
            .to_lowercase();

        let doc_type = if ext == "md" || ext == "markdown" {
            "markdown"
        } else {
            "text"
        };

        let mut metadata = HashMap::new();
        metadata.insert("source".to_string(), path.to_string());
        metadata.insert("filename".to_string(), filename);
        metadata.insert("type".to_string(), doc_type.to_string());

        Ok(Document::new(content, metadata))
    }

    pub fn load_directory(&self, dir_path: &str, extensions: &[&str]) -> anyhow::Result<Vec<Document>> {
        let mut documents = Vec::new();

        for entry in fs::read_dir(dir_path)? {
            let entry = entry?;
            let path = entry.path();

            if path.is_file() {
                if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
                    if extensions.iter().any(|&e| e.trim_start_matches('.') == ext) {
                        match self.load(path.to_str().unwrap_or("")) {
                            Ok(doc) => {
                                println!("✅ Loaded: {}", path.display());
                                documents.push(doc);
                            }
                            Err(e) => {
                                println!("❌ Failed to load {}: {}", path.display(), e);
                            }
                        }
                    }
                }
            }
        }

        Ok(documents)
    }
}

/// Text preprocessor
pub struct TextPreprocessor;

impl TextPreprocessor {
    pub fn new() -> Self {
        TextPreprocessor
    }

    pub fn clean(&self, text: &str) -> String {
        // Remove excessive whitespace
        let mut result = String::new();
        let mut last_was_space = false;

        for ch in text.chars() {
            if ch.is_whitespace() {
                if !last_was_space {
                    result.push(' ');
                    last_was_space = true;
                }
            } else {
                result.push(ch);
                last_was_space = false;
            }
        }

        result.trim().to_string()
    }
}
