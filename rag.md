# Complete Guide to Building a Production RAG System

## Table of Contents
1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Document Processing Pipeline](#document-processing-pipeline)
4. [Embedding & Vector Storage](#embedding--vector-storage)
5. [Retrieval Strategies](#retrieval-strategies)
6. [Generation with Context](#generation-with-context)
7. [Evaluation & Accuracy Metrics](#evaluation--accuracy-metrics)
8. [Testing Strategies](#testing-strategies)
9. [Monitoring & Observability](#monitoring--observability)
10. [Common Pitfalls & Solutions](#common-pitfalls--solutions)

---

## Overview

### What is RAG?

Retrieval-Augmented Generation (RAG) is a technique that enhances LLM responses by retrieving relevant information from external knowledge bases before generating answers. Instead of relying solely on the model's training data, RAG grounds responses in your specific documents.

### Why RAG?

- **Reduces hallucinations** by grounding responses in actual documents
- **Enables domain-specific knowledge** without fine-tuning
- **Keeps information current** by updating the document store
- **Provides citations** and traceability for answers
- **More cost-effective** than fine-tuning for most use cases

### Core Components

```
┌─────────────────────────────────────────────────────────────────┐
│                        RAG PIPELINE                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐ │
│   │Documents │───▶│ Chunking │───▶│Embedding │───▶│  Vector  │ │
│   │          │    │          │    │          │    │   Store  │ │
│   └──────────┘    └──────────┘    └──────────┘    └──────────┘ │
│                                                        │        │
│                                                        ▼        │
│   ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐ │
│   │ Response │◀───│   LLM    │◀───│ Context  │◀───│ Retrieve │ │
│   │          │    │Generate  │    │ Assembly │    │          │ │
│   └──────────┘    └──────────┘    └──────────┘    └──────────┘ │
│                                                        ▲        │
│                                                        │        │
│                                              ┌──────────┐       │
│                                              │  Query   │       │
│                                              └──────────┘       │
└─────────────────────────────────────────────────────────────────┘
```

---

## Architecture

### High-Level Architecture

```python
# Simplified RAG System Structure
class RAGSystem:
    def __init__(self):
        self.document_processor = DocumentProcessor()
        self.embedder = EmbeddingModel()
        self.vector_store = VectorDatabase()
        self.retriever = Retriever()
        self.generator = LLMGenerator()
    
    def ingest(self, documents: List[Document]):
        """Ingest documents into the system"""
        chunks = self.document_processor.process(documents)
        embeddings = self.embedder.embed(chunks)
        self.vector_store.upsert(chunks, embeddings)
    
    def query(self, question: str) -> Response:
        """Answer a question using RAG"""
        query_embedding = self.embedder.embed(question)
        relevant_chunks = self.retriever.retrieve(query_embedding)
        context = self.assemble_context(relevant_chunks)
        response = self.generator.generate(question, context)
        return response
```

### Technology Stack Options

| Component | Options |
|-----------|---------|
| **Vector Store** | toondb|
| **Embeddings** | OpenAI ada-002, Cohere, Voyage AI, sentence-transformers, BGE |
| **LLM** | Claude, GPT-4, Llama, Mistral |
| **Orchestration** | LangChain, LlamaIndex, Haystack, custom |
| **Document Processing** | Unstructured, LangChain loaders, PyMuPDF, Apache Tika |

---

## Document Processing Pipeline

### Step 1: Document Loading

```python
from pathlib import Path
from typing import List, Dict, Any
import hashlib

class Document:
    def __init__(self, content: str, metadata: Dict[str, Any]):
        self.content = content
        self.metadata = metadata
        self.id = self._generate_id()
    
    def _generate_id(self) -> str:
        return hashlib.md5(self.content.encode()).hexdigest()

class DocumentLoader:
    """Load documents from various sources"""
    
    def load_pdf(self, path: Path) -> Document:
        import fitz  # PyMuPDF
        doc = fitz.open(path)
        text = ""
        for page_num, page in enumerate(doc):
            text += page.get_text()
        return Document(
            content=text,
            metadata={
                "source": str(path),
                "type": "pdf",
                "pages": len(doc)
            }
        )
    
    def load_markdown(self, path: Path) -> Document:
        with open(path, 'r') as f:
            content = f.read()
        return Document(
            content=content,
            metadata={"source": str(path), "type": "markdown"}
        )
    
    def load_html(self, path: Path) -> Document:
        from bs4 import BeautifulSoup
        with open(path, 'r') as f:
            soup = BeautifulSoup(f.read(), 'html.parser')
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
        text = soup.get_text(separator='\n')
        return Document(
            content=text,
            metadata={"source": str(path), "type": "html"}
        )
```

### Step 2: Text Cleaning & Preprocessing

```python
import re
from typing import Optional

class TextPreprocessor:
    """Clean and normalize text before chunking"""
    
    def clean(self, text: str) -> str:
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters that don't add meaning
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', text)
        # Normalize unicode
        text = text.encode('utf-8', errors='ignore').decode('utf-8')
        return text.strip()
    
    def remove_boilerplate(self, text: str, doc_type: str) -> str:
        """Remove headers, footers, and other boilerplate"""
        if doc_type == "pdf":
            # Remove page numbers
            text = re.sub(r'\n\s*\d+\s*\n', '\n', text)
            # Remove common header/footer patterns
            text = re.sub(r'(Page \d+ of \d+)', '', text)
        return text
```

### Step 3: Chunking Strategies

Chunking is critical—it determines the quality of retrieval.

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List

@dataclass
class Chunk:
    content: str
    metadata: Dict[str, Any]
    start_index: int
    end_index: int

class ChunkingStrategy(ABC):
    @abstractmethod
    def chunk(self, document: Document) -> List[Chunk]:
        pass

class FixedSizeChunker(ChunkingStrategy):
    """Simple fixed-size chunking with overlap"""
    
    def __init__(self, chunk_size: int = 512, overlap: int = 50):
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def chunk(self, document: Document) -> List[Chunk]:
        text = document.content
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.chunk_size
            chunk_text = text[start:end]
            
            chunks.append(Chunk(
                content=chunk_text,
                metadata={**document.metadata, "chunk_index": len(chunks)},
                start_index=start,
                end_index=end
            ))
            
            start = end - self.overlap
        
        return chunks

class SemanticChunker(ChunkingStrategy):
    """Chunk based on semantic boundaries (paragraphs, sections)"""
    
    def __init__(self, max_chunk_size: int = 1000, min_chunk_size: int = 100):
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size
    
    def chunk(self, document: Document) -> List[Chunk]:
        # Split by paragraphs first
        paragraphs = re.split(r'\n\n+', document.content)
        chunks = []
        current_chunk = ""
        current_start = 0
        
        for para in paragraphs:
            if len(current_chunk) + len(para) <= self.max_chunk_size:
                current_chunk += para + "\n\n"
            else:
                if len(current_chunk) >= self.min_chunk_size:
                    chunks.append(Chunk(
                        content=current_chunk.strip(),
                        metadata={**document.metadata, "chunk_index": len(chunks)},
                        start_index=current_start,
                        end_index=current_start + len(current_chunk)
                    ))
                current_start += len(current_chunk)
                current_chunk = para + "\n\n"
        
        # Don't forget the last chunk
        if current_chunk.strip():
            chunks.append(Chunk(
                content=current_chunk.strip(),
                metadata={**document.metadata, "chunk_index": len(chunks)},
                start_index=current_start,
                end_index=current_start + len(current_chunk)
            ))
        
        return chunks

class RecursiveChunker(ChunkingStrategy):
    """Recursively split by different separators"""
    
    def __init__(self, chunk_size: int = 512, overlap: int = 50):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.separators = ["\n\n", "\n", ". ", " ", ""]
    
    def chunk(self, document: Document) -> List[Chunk]:
        return self._split_text(document.content, self.separators, document.metadata)
    
    def _split_text(self, text: str, separators: List[str], metadata: dict) -> List[Chunk]:
        chunks = []
        separator = separators[0]
        
        splits = text.split(separator) if separator else list(text)
        
        current_chunk = ""
        for split in splits:
            if len(current_chunk) + len(split) <= self.chunk_size:
                current_chunk += split + separator
            else:
                if current_chunk:
                    if len(current_chunk) > self.chunk_size and len(separators) > 1:
                        # Recursively split with next separator
                        chunks.extend(self._split_text(
                            current_chunk, separators[1:], metadata
                        ))
                    else:
                        chunks.append(Chunk(
                            content=current_chunk.strip(),
                            metadata={**metadata, "chunk_index": len(chunks)},
                            start_index=0,
                            end_index=len(current_chunk)
                        ))
                current_chunk = split + separator
        
        if current_chunk.strip():
            chunks.append(Chunk(
                content=current_chunk.strip(),
                metadata={**metadata, "chunk_index": len(chunks)},
                start_index=0,
                end_index=len(current_chunk)
            ))
        
        return chunks
```

### Chunking Best Practices

| Factor | Recommendation |
|--------|----------------|
| **Chunk Size** | 256-1024 tokens typically works well; tune based on your use case |
| **Overlap** | 10-20% overlap prevents context loss at boundaries |
| **Preserve Structure** | Keep headings, lists, and code blocks intact |
| **Metadata** | Include source, page number, section title for filtering |
| **Small Chunks** | Better for precise retrieval |
| **Large Chunks** | Better for context-rich answers |

---

## Embedding & Vector Storage

### Embedding Models Comparison

```python
from abc import ABC, abstractmethod
import numpy as np
from typing import List

class EmbeddingModel(ABC):
    @abstractmethod
    def embed(self, texts: List[str]) -> np.ndarray:
        pass
    
    @property
    @abstractmethod
    def dimension(self) -> int:
        pass

class OpenAIEmbeddings(EmbeddingModel):
    """OpenAI's text-embedding-3-small or ada-002"""
    
    def __init__(self, model: str = "text-embedding-3-small"):
        from openai import OpenAI
        self.client = OpenAI()
        self.model = model
        self._dimension = 1536 if "ada" in model else 1536
    
    def embed(self, texts: List[str]) -> np.ndarray:
        response = self.client.embeddings.create(
            input=texts,
            model=self.model
        )
        return np.array([e.embedding for e in response.data])
    
    @property
    def dimension(self) -> int:
        return self._dimension

class VoyageEmbeddings(EmbeddingModel):
    """Voyage AI embeddings - excellent for retrieval"""
    
    def __init__(self, model: str = "voyage-2"):
        import voyageai
        self.client = voyageai.Client()
        self.model = model
    
    def embed(self, texts: List[str]) -> np.ndarray:
        result = self.client.embed(texts, model=self.model)
        return np.array(result.embeddings)
    
    @property
    def dimension(self) -> int:
        return 1024

class LocalEmbeddings(EmbeddingModel):
    """Local embeddings using sentence-transformers"""
    
    def __init__(self, model_name: str = "BAAI/bge-base-en-v1.5"):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)
    
    def embed(self, texts: List[str]) -> np.ndarray:
        return self.model.encode(texts, normalize_embeddings=True)
    
    @property
    def dimension(self) -> int:
        return self.model.get_sentence_embedding_dimension()
```

### Vector Store Implementation

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Tuple

@dataclass
class SearchResult:
    chunk: Chunk
    score: float
    
class VectorStore(ABC):
    @abstractmethod
    def upsert(self, chunks: List[Chunk], embeddings: np.ndarray):
        pass
    
    @abstractmethod
    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[SearchResult]:
        pass
    
    @abstractmethod
    def delete(self, ids: List[str]):
        pass

class ChromaDBStore(VectorStore):
    """ChromaDB - great for local development"""
    
    def __init__(self, collection_name: str = "documents"):
        import chromadb
        self.client = chromadb.PersistentClient(path="./chroma_db")
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
    
    def upsert(self, chunks: List[Chunk], embeddings: np.ndarray):
        self.collection.upsert(
            ids=[f"chunk_{i}" for i in range(len(chunks))],
            embeddings=embeddings.tolist(),
            documents=[c.content for c in chunks],
            metadatas=[c.metadata for c in chunks]
        )
    
    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[SearchResult]:
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )
        
        search_results = []
        for i in range(len(results['documents'][0])):
            chunk = Chunk(
                content=results['documents'][0][i],
                metadata=results['metadatas'][0][i],
                start_index=0,
                end_index=0
            )
            # Convert distance to similarity score
            score = 1 - results['distances'][0][i]
            search_results.append(SearchResult(chunk=chunk, score=score))
        
        return search_results

class PineconeStore(VectorStore):
    """Pinecone - production-ready managed vector DB"""
    
    def __init__(self, index_name: str, dimension: int):
        from pinecone import Pinecone
        self.pc = Pinecone()
        self.index = self.pc.Index(index_name)
        self.dimension = dimension
    
    def upsert(self, chunks: List[Chunk], embeddings: np.ndarray):
        vectors = [
            {
                "id": f"chunk_{i}",
                "values": embeddings[i].tolist(),
                "metadata": {
                    "content": chunks[i].content,
                    **chunks[i].metadata
                }
            }
            for i in range(len(chunks))
        ]
        
        # Batch upsert
        batch_size = 100
        for i in range(0, len(vectors), batch_size):
            self.index.upsert(vectors=vectors[i:i+batch_size])
    
    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[SearchResult]:
        results = self.index.query(
            vector=query_embedding.tolist(),
            top_k=top_k,
            include_metadata=True
        )
        
        return [
            SearchResult(
                chunk=Chunk(
                    content=match.metadata.get("content", ""),
                    metadata=match.metadata,
                    start_index=0,
                    end_index=0
                ),
                score=match.score
            )
            for match in results.matches
        ]

class PgVectorStore(VectorStore):
    """PostgreSQL with pgvector - great if you already use Postgres"""
    
    def __init__(self, connection_string: str, table_name: str = "documents"):
        import psycopg2
        self.conn = psycopg2.connect(connection_string)
        self.table_name = table_name
        self._init_table()
    
    def _init_table(self):
        with self.conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS {self.table_name} (
                    id SERIAL PRIMARY KEY,
                    content TEXT,
                    metadata JSONB,
                    embedding vector(1536)
                )
            """)
            cur.execute(f"""
                CREATE INDEX IF NOT EXISTS {self.table_name}_embedding_idx 
                ON {self.table_name} 
                USING ivfflat (embedding vector_cosine_ops)
            """)
        self.conn.commit()
    
    def upsert(self, chunks: List[Chunk], embeddings: np.ndarray):
        import json
        with self.conn.cursor() as cur:
            for i, chunk in enumerate(chunks):
                cur.execute(f"""
                    INSERT INTO {self.table_name} (content, metadata, embedding)
                    VALUES (%s, %s, %s)
                """, (
                    chunk.content,
                    json.dumps(chunk.metadata),
                    embeddings[i].tolist()
                ))
        self.conn.commit()
    
    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[SearchResult]:
        with self.conn.cursor() as cur:
            cur.execute(f"""
                SELECT content, metadata, 1 - (embedding <=> %s) as score
                FROM {self.table_name}
                ORDER BY embedding <=> %s
                LIMIT %s
            """, (query_embedding.tolist(), query_embedding.tolist(), top_k))
            
            results = []
            for row in cur.fetchall():
                chunk = Chunk(
                    content=row[0],
                    metadata=row[1],
                    start_index=0,
                    end_index=0
                )
                results.append(SearchResult(chunk=chunk, score=row[2]))
            
            return results
```

---

## Retrieval Strategies

### Basic Retrieval

```python
class BasicRetriever:
    """Simple top-k retrieval"""
    
    def __init__(self, vector_store: VectorStore, embedder: EmbeddingModel):
        self.vector_store = vector_store
        self.embedder = embedder
    
    def retrieve(self, query: str, top_k: int = 5) -> List[SearchResult]:
        query_embedding = self.embedder.embed([query])[0]
        return self.vector_store.search(query_embedding, top_k)
```

### Hybrid Retrieval (Dense + Sparse)

```python
from rank_bm25 import BM25Okapi
import numpy as np

class HybridRetriever:
    """Combines dense (vector) and sparse (BM25) retrieval"""
    
    def __init__(
        self, 
        vector_store: VectorStore, 
        embedder: EmbeddingModel,
        chunks: List[Chunk],
        alpha: float = 0.5  # Weight for dense vs sparse
    ):
        self.vector_store = vector_store
        self.embedder = embedder
        self.alpha = alpha
        
        # Build BM25 index
        tokenized_corpus = [c.content.lower().split() for c in chunks]
        self.bm25 = BM25Okapi(tokenized_corpus)
        self.chunks = chunks
    
    def retrieve(self, query: str, top_k: int = 5) -> List[SearchResult]:
        # Dense retrieval
        query_embedding = self.embedder.embed([query])[0]
        dense_results = self.vector_store.search(query_embedding, top_k * 2)
        
        # Sparse retrieval (BM25)
        tokenized_query = query.lower().split()
        bm25_scores = self.bm25.get_scores(tokenized_query)
        
        # Normalize scores
        dense_scores = {r.chunk.content: r.score for r in dense_results}
        max_bm25 = max(bm25_scores) if max(bm25_scores) > 0 else 1
        
        # Combine scores
        combined_scores = {}
        for i, chunk in enumerate(self.chunks):
            dense_score = dense_scores.get(chunk.content, 0)
            sparse_score = bm25_scores[i] / max_bm25
            combined_scores[i] = self.alpha * dense_score + (1 - self.alpha) * sparse_score
        
        # Sort and return top-k
        sorted_indices = sorted(combined_scores.keys(), key=lambda x: combined_scores[x], reverse=True)
        
        return [
            SearchResult(chunk=self.chunks[i], score=combined_scores[i])
            for i in sorted_indices[:top_k]
        ]
```

### Reranking

```python
class RerankedRetriever:
    """Two-stage retrieval with reranking"""
    
    def __init__(
        self,
        vector_store: VectorStore,
        embedder: EmbeddingModel,
        reranker_model: str = "BAAI/bge-reranker-base"
    ):
        self.vector_store = vector_store
        self.embedder = embedder
        
        from sentence_transformers import CrossEncoder
        self.reranker = CrossEncoder(reranker_model)
    
    def retrieve(self, query: str, top_k: int = 5, initial_k: int = 20) -> List[SearchResult]:
        # Stage 1: Initial retrieval
        query_embedding = self.embedder.embed([query])[0]
        initial_results = self.vector_store.search(query_embedding, initial_k)
        
        # Stage 2: Rerank
        pairs = [(query, r.chunk.content) for r in initial_results]
        rerank_scores = self.reranker.predict(pairs)
        
        # Sort by rerank score
        reranked = sorted(
            zip(initial_results, rerank_scores),
            key=lambda x: x[1],
            reverse=True
        )
        
        return [
            SearchResult(chunk=r.chunk, score=float(score))
            for r, score in reranked[:top_k]
        ]
```

### Query Expansion / HyDE

```python
class HyDERetriever:
    """Hypothetical Document Embeddings - generate hypothetical answer first"""
    
    def __init__(
        self,
        vector_store: VectorStore,
        embedder: EmbeddingModel,
        llm_client
    ):
        self.vector_store = vector_store
        self.embedder = embedder
        self.llm = llm_client
    
    def retrieve(self, query: str, top_k: int = 5) -> List[SearchResult]:
        # Generate hypothetical answer
        hypothetical_prompt = f"""Given the question: "{query}"
        
Write a short passage that would answer this question. Write as if you're 
writing a paragraph from a document that contains the answer."""
        
        hypothetical_doc = self.llm.generate(hypothetical_prompt)
        
        # Embed the hypothetical document instead of the query
        hyde_embedding = self.embedder.embed([hypothetical_doc])[0]
        
        return self.vector_store.search(hyde_embedding, top_k)
```

### Multi-Query Retrieval

```python
class MultiQueryRetriever:
    """Generate multiple query variations for better recall"""
    
    def __init__(
        self,
        vector_store: VectorStore,
        embedder: EmbeddingModel,
        llm_client
    ):
        self.vector_store = vector_store
        self.embedder = embedder
        self.llm = llm_client
    
    def retrieve(self, query: str, top_k: int = 5, num_queries: int = 3) -> List[SearchResult]:
        # Generate query variations
        variation_prompt = f"""Generate {num_queries} different versions of this question 
that would help find relevant documents. Make them diverse in phrasing.

Original question: {query}

Return only the questions, one per line."""
        
        variations_text = self.llm.generate(variation_prompt)
        queries = [query] + variations_text.strip().split('\n')
        
        # Retrieve for each query
        all_results = {}
        for q in queries:
            q_embedding = self.embedder.embed([q])[0]
            results = self.vector_store.search(q_embedding, top_k)
            for r in results:
                key = r.chunk.content
                if key not in all_results or all_results[key].score < r.score:
                    all_results[key] = r
        
        # Return top-k unique results
        sorted_results = sorted(all_results.values(), key=lambda x: x.score, reverse=True)
        return sorted_results[:top_k]
```

---

## Generation with Context

### Context Assembly

```python
class ContextAssembler:
    """Assemble retrieved chunks into a coherent context"""
    
    def __init__(self, max_context_length: int = 4000):
        self.max_context_length = max_context_length
    
    def assemble(self, results: List[SearchResult]) -> str:
        context_parts = []
        current_length = 0
        
        for i, result in enumerate(results):
            chunk_text = f"[Source {i+1}]\n{result.chunk.content}\n"
            
            if current_length + len(chunk_text) > self.max_context_length:
                break
            
            context_parts.append(chunk_text)
            current_length += len(chunk_text)
        
        return "\n".join(context_parts)
    
    def assemble_with_metadata(self, results: List[SearchResult]) -> str:
        """Include source metadata for citations"""
        context_parts = []
        
        for i, result in enumerate(results):
            source = result.chunk.metadata.get('source', 'Unknown')
            page = result.chunk.metadata.get('page', '')
            
            header = f"[Source {i+1}: {source}"
            if page:
                header += f", Page {page}"
            header += "]"
            
            context_parts.append(f"{header}\n{result.chunk.content}\n")
        
        return "\n".join(context_parts)
```

### RAG Prompt Templates

```python
class RAGPrompts:
    """Prompt templates for RAG generation"""
    
    BASIC_QA = """Answer the question based on the provided context. 
If the context doesn't contain enough information to answer, say so.

Context:
{context}

Question: {question}

Answer:"""

    QA_WITH_CITATIONS = """Answer the question based on the provided context.
Cite your sources using [Source N] notation.
If the context doesn't contain enough information, say so.

Context:
{context}

Question: {question}

Provide a detailed answer with citations:"""

    CONVERSATIONAL = """You are a helpful assistant with access to a knowledge base.
Use the following context to answer the user's question.
If you're unsure or the context doesn't help, be honest about it.

Context from knowledge base:
{context}

User: {question}
Assistant:"""

    STEP_BY_STEP = """Based on the context provided, answer the question step by step.

Context:
{context}

Question: {question}

Think through this step by step:
1. What information from the context is relevant?
2. How does it answer the question?
3. What is the final answer?"""
```

### Complete RAG Generator

```python
class RAGGenerator:
    """Complete RAG generation with configurable prompts"""
    
    def __init__(self, llm_client, prompt_template: str = None):
        self.llm = llm_client
        self.prompt_template = prompt_template or RAGPrompts.QA_WITH_CITATIONS
    
    def generate(
        self, 
        question: str, 
        context: str,
        max_tokens: int = 1000,
        temperature: float = 0.1
    ) -> str:
        prompt = self.prompt_template.format(
            context=context,
            question=question
        )
        
        response = self.llm.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}]
        )
        
        return response.content[0].text
    
    def generate_with_sources(
        self,
        question: str,
        results: List[SearchResult]
    ) -> dict:
        """Generate answer with structured source tracking"""
        context = ContextAssembler().assemble_with_metadata(results)
        answer = self.generate(question, context)
        
        return {
            "answer": answer,
            "sources": [
                {
                    "content": r.chunk.content[:200] + "...",
                    "metadata": r.chunk.metadata,
                    "relevance_score": r.score
                }
                for r in results
            ]
        }
```

---

## Evaluation & Accuracy Metrics

This is crucial—you need to know if your RAG system is actually working.

### Evaluation Framework Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                   RAG EVALUATION FRAMEWORK                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   ┌────────────────┐     ┌────────────────┐                     │
│   │   RETRIEVAL    │     │   GENERATION   │                     │
│   │   EVALUATION   │     │   EVALUATION   │                     │
│   ├────────────────┤     ├────────────────┤                     │
│   │ • Precision@K  │     │ • Faithfulness │                     │
│   │ • Recall@K     │     │ • Relevance    │                     │
│   │ • MRR          │     │ • Coherence    │                     │
│   │ • NDCG         │     │ • Completeness │                     │
│   │ • Hit Rate     │     │ • Correctness  │                     │
│   └────────────────┘     └────────────────┘                     │
│                                                                 │
│   ┌────────────────────────────────────────┐                    │
│   │          END-TO-END EVALUATION         │                    │
│   ├────────────────────────────────────────┤                    │
│   │ • Answer Correctness                   │                    │
│   │ • Context Precision                    │                    │
│   │ • Context Recall                       │                    │
│   │ • Answer Similarity                    │                    │
│   └────────────────────────────────────────┘                    │
└─────────────────────────────────────────────────────────────────┘
```

### Creating an Evaluation Dataset

```python
from dataclasses import dataclass
from typing import List, Optional
import json

@dataclass
class EvalExample:
    """Single evaluation example"""
    question: str
    ground_truth_answer: str
    ground_truth_contexts: List[str]  # Chunks that should be retrieved
    metadata: Optional[dict] = None

class EvalDataset:
    """Evaluation dataset for RAG system"""
    
    def __init__(self, examples: List[EvalExample]):
        self.examples = examples
    
    @classmethod
    def from_json(cls, path: str) -> "EvalDataset":
        with open(path, 'r') as f:
            data = json.load(f)
        examples = [EvalExample(**ex) for ex in data]
        return cls(examples)
    
    @classmethod
    def generate_synthetic(cls, documents: List[Document], llm_client, n_examples: int = 50):
        """Generate synthetic QA pairs from documents"""
        examples = []
        
        for doc in documents[:n_examples]:
            # Extract a chunk to generate question from
            chunk = doc.content[:2000]
            
            prompt = f"""Based on this text, generate a question-answer pair.
The question should be answerable from the text.

Text:
{chunk}

Return in this exact format:
Question: <your question>
Answer: <the answer from the text>"""
            
            response = llm_client.generate(prompt)
            
            # Parse response
            lines = response.strip().split('\n')
            question = lines[0].replace('Question:', '').strip()
            answer = lines[1].replace('Answer:', '').strip()
            
            examples.append(EvalExample(
                question=question,
                ground_truth_answer=answer,
                ground_truth_contexts=[chunk]
            ))
        
        return cls(examples)
    
    def save(self, path: str):
        data = [
            {
                "question": ex.question,
                "ground_truth_answer": ex.ground_truth_answer,
                "ground_truth_contexts": ex.ground_truth_contexts,
                "metadata": ex.metadata
            }
            for ex in self.examples
        ]
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
```

### Retrieval Metrics

```python
import numpy as np
from typing import List, Set

class RetrievalMetrics:
    """Metrics for evaluating retrieval quality"""
    
    @staticmethod
    def precision_at_k(retrieved: List[str], relevant: Set[str], k: int) -> float:
        """What fraction of retrieved docs are relevant?"""
        retrieved_k = retrieved[:k]
        relevant_retrieved = sum(1 for doc in retrieved_k if doc in relevant)
        return relevant_retrieved / k if k > 0 else 0.0
    
    @staticmethod
    def recall_at_k(retrieved: List[str], relevant: Set[str], k: int) -> float:
        """What fraction of relevant docs were retrieved?"""
        retrieved_k = set(retrieved[:k])
        relevant_retrieved = len(retrieved_k & relevant)
        return relevant_retrieved / len(relevant) if relevant else 0.0
    
    @staticmethod
    def mrr(retrieved: List[str], relevant: Set[str]) -> float:
        """Mean Reciprocal Rank - where is the first relevant result?"""
        for i, doc in enumerate(retrieved):
            if doc in relevant:
                return 1.0 / (i + 1)
        return 0.0
    
    @staticmethod
    def ndcg_at_k(retrieved: List[str], relevant: Set[str], k: int) -> float:
        """Normalized Discounted Cumulative Gain"""
        dcg = 0.0
        for i, doc in enumerate(retrieved[:k]):
            if doc in relevant:
                dcg += 1.0 / np.log2(i + 2)  # +2 because i starts at 0
        
        # Ideal DCG
        ideal_dcg = sum(1.0 / np.log2(i + 2) for i in range(min(k, len(relevant))))
        
        return dcg / ideal_dcg if ideal_dcg > 0 else 0.0
    
    @staticmethod
    def hit_rate(retrieved: List[str], relevant: Set[str], k: int) -> float:
        """Is there at least one relevant doc in top-k?"""
        return 1.0 if any(doc in relevant for doc in retrieved[:k]) else 0.0

class RetrievalEvaluator:
    """Evaluate retrieval component of RAG"""
    
    def __init__(self, retriever):
        self.retriever = retriever
        self.metrics = RetrievalMetrics()
    
    def evaluate(self, eval_dataset: EvalDataset, k_values: List[int] = [1, 3, 5, 10]) -> dict:
        results = {f"precision@{k}": [] for k in k_values}
        results.update({f"recall@{k}": [] for k in k_values})
        results.update({f"ndcg@{k}": [] for k in k_values})
        results["mrr"] = []
        results["hit_rate@5"] = []
        
        for example in eval_dataset.examples:
            # Get retrieved docs
            search_results = self.retriever.retrieve(example.question, top_k=max(k_values))
            retrieved = [r.chunk.content for r in search_results]
            relevant = set(example.ground_truth_contexts)
            
            # Calculate metrics
            for k in k_values:
                results[f"precision@{k}"].append(
                    self.metrics.precision_at_k(retrieved, relevant, k)
                )
                results[f"recall@{k}"].append(
                    self.metrics.recall_at_k(retrieved, relevant, k)
                )
                results[f"ndcg@{k}"].append(
                    self.metrics.ndcg_at_k(retrieved, relevant, k)
                )
            
            results["mrr"].append(self.metrics.mrr(retrieved, relevant))
            results["hit_rate@5"].append(self.metrics.hit_rate(retrieved, relevant, 5))
        
        # Average all metrics
        return {metric: np.mean(values) for metric, values in results.items()}
```

### Generation Metrics (LLM-as-Judge)

```python
class GenerationMetrics:
    """Use LLM to evaluate generation quality"""
    
    def __init__(self, llm_client):
        self.llm = llm_client
    
    def faithfulness(self, answer: str, context: str) -> float:
        """Is the answer faithful to the context? (No hallucination)"""
        prompt = f"""Evaluate if the answer is faithful to the provided context.
A faithful answer only contains information that can be directly inferred from the context.

Context:
{context}

Answer:
{answer}

Rate faithfulness from 0 to 1:
- 1.0: Completely faithful, all claims supported by context
- 0.5: Partially faithful, some unsupported claims
- 0.0: Unfaithful, contains hallucinations

Return only a number between 0 and 1."""
        
        response = self.llm.generate(prompt)
        try:
            return float(response.strip())
        except:
            return 0.5
    
    def relevance(self, question: str, answer: str) -> float:
        """Does the answer address the question?"""
        prompt = f"""Evaluate if the answer is relevant to and addresses the question.

Question: {question}

Answer: {answer}

Rate relevance from 0 to 1:
- 1.0: Directly and completely addresses the question
- 0.5: Partially addresses the question
- 0.0: Does not address the question at all

Return only a number between 0 and 1."""
        
        response = self.llm.generate(prompt)
        try:
            return float(response.strip())
        except:
            return 0.5
    
    def correctness(self, answer: str, ground_truth: str) -> float:
        """Is the answer correct compared to ground truth?"""
        prompt = f"""Compare the generated answer to the ground truth answer.
Evaluate if they convey the same information.

Ground Truth: {ground_truth}

Generated Answer: {answer}

Rate correctness from 0 to 1:
- 1.0: Semantically equivalent, same key information
- 0.5: Partially correct, some information matches
- 0.0: Incorrect or contradictory

Return only a number between 0 and 1."""
        
        response = self.llm.generate(prompt)
        try:
            return float(response.strip())
        except:
            return 0.5
    
    def context_precision(self, question: str, contexts: List[str]) -> float:
        """Are the retrieved contexts relevant to the question?"""
        relevant_count = 0
        
        for ctx in contexts:
            prompt = f"""Is this context relevant for answering the question?

Question: {question}

Context: {ctx[:500]}

Answer with only 'yes' or 'no'."""
            
            response = self.llm.generate(prompt).strip().lower()
            if 'yes' in response:
                relevant_count += 1
        
        return relevant_count / len(contexts) if contexts else 0.0

class GenerationEvaluator:
    """Evaluate generation component of RAG"""
    
    def __init__(self, rag_system, llm_client):
        self.rag = rag_system
        self.metrics = GenerationMetrics(llm_client)
    
    def evaluate(self, eval_dataset: EvalDataset) -> dict:
        results = {
            "faithfulness": [],
            "relevance": [],
            "correctness": [],
            "context_precision": []
        }
        
        for example in eval_dataset.examples:
            # Get RAG response
            response = self.rag.query(example.question)
            
            # Evaluate
            results["faithfulness"].append(
                self.metrics.faithfulness(response.answer, response.context)
            )
            results["relevance"].append(
                self.metrics.relevance(example.question, response.answer)
            )
            results["correctness"].append(
                self.metrics.correctness(response.answer, example.ground_truth_answer)
            )
            results["context_precision"].append(
                self.metrics.context_precision(
                    example.question, 
                    [r.chunk.content for r in response.sources]
                )
            )
        
        return {metric: np.mean(values) for metric, values in results.items()}
```

### Using RAGAS Framework

```python
# RAGAS is a popular framework for RAG evaluation
# pip install ragas

from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
    answer_correctness
)
from datasets import Dataset

def evaluate_with_ragas(rag_system, eval_dataset: EvalDataset):
    """Use RAGAS framework for comprehensive evaluation"""
    
    # Prepare data
    questions = []
    answers = []
    contexts = []
    ground_truths = []
    
    for example in eval_dataset.examples:
        response = rag_system.query(example.question)
        
        questions.append(example.question)
        answers.append(response.answer)
        contexts.append([r.chunk.content for r in response.sources])
        ground_truths.append(example.ground_truth_answer)
    
    # Create dataset
    dataset = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths
    })
    
    # Evaluate
    results = evaluate(
        dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
            answer_correctness
        ]
    )
    
    return results
```

### Embedding Quality Evaluation

```python
class EmbeddingEvaluator:
    """Evaluate embedding model quality"""
    
    def __init__(self, embedder: EmbeddingModel):
        self.embedder = embedder
    
    def semantic_similarity_test(self, similar_pairs: List[tuple], dissimilar_pairs: List[tuple]) -> dict:
        """Test if similar texts have higher similarity than dissimilar texts"""
        
        similar_scores = []
        for text1, text2 in similar_pairs:
            emb1 = self.embedder.embed([text1])[0]
            emb2 = self.embedder.embed([text2])[0]
            similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
            similar_scores.append(similarity)
        
        dissimilar_scores = []
        for text1, text2 in dissimilar_pairs:
            emb1 = self.embedder.embed([text1])[0]
            emb2 = self.embedder.embed([text2])[0]
            similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
            dissimilar_scores.append(similarity)
        
        return {
            "avg_similar_score": np.mean(similar_scores),
            "avg_dissimilar_score": np.mean(dissimilar_scores),
            "separation": np.mean(similar_scores) - np.mean(dissimilar_scores)
        }
    
    def query_document_alignment(self, query_doc_pairs: List[tuple]) -> float:
        """Test query-document alignment"""
        scores = []
        
        for query, relevant_doc in query_doc_pairs:
            q_emb = self.embedder.embed([query])[0]
            d_emb = self.embedder.embed([relevant_doc])[0]
            similarity = np.dot(q_emb, d_emb) / (np.linalg.norm(q_emb) * np.linalg.norm(d_emb))
            scores.append(similarity)
        
        return np.mean(scores)
```

---

## Testing Strategies

### Unit Tests

```python
import pytest
from unittest.mock import Mock, patch

class TestDocumentProcessor:
    """Unit tests for document processing"""
    
    def test_chunking_respects_size_limit(self):
        chunker = FixedSizeChunker(chunk_size=100, overlap=10)
        doc = Document(content="a" * 500, metadata={})
        
        chunks = chunker.chunk(doc)
        
        for chunk in chunks:
            assert len(chunk.content) <= 100
    
    def test_chunking_preserves_content(self):
        chunker = FixedSizeChunker(chunk_size=100, overlap=0)
        original_text = "This is the original text content."
        doc = Document(content=original_text, metadata={})
        
        chunks = chunker.chunk(doc)
        reconstructed = "".join(c.content for c in chunks)
        
        assert original_text in reconstructed
    
    def test_text_preprocessor_removes_special_chars(self):
        preprocessor = TextPreprocessor()
        dirty_text = "Hello\x00World\x1f!"
        
        clean = preprocessor.clean(dirty_text)
        
        assert "\x00" not in clean
        assert "\x1f" not in clean

class TestRetriever:
    """Unit tests for retrieval"""
    
    def test_retriever_returns_top_k(self):
        mock_vector_store = Mock()
        mock_vector_store.search.return_value = [
            SearchResult(Chunk("content", {}, 0, 0), 0.9),
            SearchResult(Chunk("content", {}, 0, 0), 0.8),
        ]
        
        mock_embedder = Mock()
        mock_embedder.embed.return_value = np.array([[0.1, 0.2, 0.3]])
        
        retriever = BasicRetriever(mock_vector_store, mock_embedder)
        results = retriever.retrieve("test query", top_k=2)
        
        assert len(results) == 2
    
    def test_retriever_orders_by_score(self):
        mock_vector_store = Mock()
        mock_vector_store.search.return_value = [
            SearchResult(Chunk("high", {}, 0, 0), 0.95),
            SearchResult(Chunk("low", {}, 0, 0), 0.5),
        ]
        
        mock_embedder = Mock()
        mock_embedder.embed.return_value = np.array([[0.1, 0.2, 0.3]])
        
        retriever = BasicRetriever(mock_vector_store, mock_embedder)
        results = retriever.retrieve("test query")
        
        assert results[0].score > results[1].score

class TestEmbeddings:
    """Unit tests for embeddings"""
    
    def test_embedding_dimension(self):
        embedder = LocalEmbeddings("sentence-transformers/all-MiniLM-L6-v2")
        embedding = embedder.embed(["test text"])
        
        assert embedding.shape[1] == embedder.dimension
    
    def test_embedding_normalization(self):
        embedder = LocalEmbeddings("BAAI/bge-base-en-v1.5")
        embedding = embedder.embed(["test text"])[0]
        
        norm = np.linalg.norm(embedding)
        assert abs(norm - 1.0) < 0.01  # Should be normalized
```

### Integration Tests

```python
class TestRAGIntegration:
    """Integration tests for the complete RAG pipeline"""
    
    @pytest.fixture
    def rag_system(self):
        """Set up a test RAG system"""
        embedder = LocalEmbeddings()
        vector_store = ChromaDBStore(collection_name="test_collection")
        retriever = BasicRetriever(vector_store, embedder)
        
        # Ingest test documents
        test_docs = [
            Document("Python is a programming language created by Guido van Rossum.", {"source": "test1"}),
            Document("Machine learning is a subset of artificial intelligence.", {"source": "test2"}),
        ]
        
        chunker = SemanticChunker()
        for doc in test_docs:
            chunks = chunker.chunk(doc)
            embeddings = embedder.embed([c.content for c in chunks])
            vector_store.upsert(chunks, embeddings)
        
        return RAGSystem(retriever=retriever, generator=Mock())
    
    def test_relevant_retrieval(self, rag_system):
        """Test that relevant documents are retrieved"""
        results = rag_system.retriever.retrieve("Who created Python?")
        
        assert len(results) > 0
        assert any("Guido" in r.chunk.content for r in results)
    
    def test_irrelevant_query_handling(self, rag_system):
        """Test behavior with irrelevant queries"""
        results = rag_system.retriever.retrieve("What is the recipe for chocolate cake?")
        
        # Should still return results, but with low scores
        assert all(r.score < 0.5 for r in results)

class TestEndToEnd:
    """End-to-end tests simulating real usage"""
    
    def test_document_ingestion_to_query(self):
        """Test complete flow from ingestion to query"""
        # Set up
        rag = RAGSystem()
        
        # Ingest
        doc = Document(
            content="The Eiffel Tower is located in Paris, France. It was built in 1889.",
            metadata={"source": "landmarks.txt"}
        )
        rag.ingest([doc])
        
        # Query
        response = rag.query("Where is the Eiffel Tower?")
        
        assert "Paris" in response.answer or "France" in response.answer
        assert len(response.sources) > 0
```

### Regression Tests

```python
class TestRegression:
    """Regression tests to catch quality degradation"""
    
    # Golden test cases - these should always work
    GOLDEN_TESTS = [
        {
            "question": "What is photosynthesis?",
            "expected_keywords": ["plants", "light", "energy", "glucose"],
            "min_relevance": 0.7
        },
        {
            "question": "Who was Albert Einstein?",
            "expected_keywords": ["physicist", "relativity", "Nobel"],
            "min_relevance": 0.7
        }
    ]
    
    def test_golden_cases(self, rag_system):
        """Ensure golden test cases still pass"""
        for test_case in self.GOLDEN_TESTS:
            response = rag_system.query(test_case["question"])
            
            # Check keywords present
            answer_lower = response.answer.lower()
            keywords_found = sum(
                1 for kw in test_case["expected_keywords"] 
                if kw.lower() in answer_lower
            )
            
            assert keywords_found >= len(test_case["expected_keywords"]) // 2
    
    def test_no_hallucination_regression(self, rag_system):
        """Ensure system doesn't hallucinate on known topics"""
        # Ask about something NOT in the knowledge base
        response = rag_system.query("What is the population of Mars?")
        
        # Should indicate uncertainty
        uncertainty_phrases = ["don't have", "no information", "not sure", "cannot find"]
        assert any(phrase in response.answer.lower() for phrase in uncertainty_phrases)
```

### Load Testing

```python
import asyncio
import time
from concurrent.futures import ThreadPoolExecutor

class LoadTester:
    """Test system performance under load"""
    
    def __init__(self, rag_system):
        self.rag = rag_system
    
    def measure_latency(self, query: str) -> float:
        """Measure query latency"""
        start = time.time()
        self.rag.query(query)
        return time.time() - start
    
    def run_load_test(
        self, 
        queries: List[str], 
        concurrency: int = 10,
        duration_seconds: int = 60
    ) -> dict:
        """Run load test with concurrent queries"""
        results = {
            "total_queries": 0,
            "latencies": [],
            "errors": 0
        }
        
        end_time = time.time() + duration_seconds
        
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            while time.time() < end_time:
                query = queries[results["total_queries"] % len(queries)]
                
                try:
                    latency = self.measure_latency(query)
                    results["latencies"].append(latency)
                except Exception as e:
                    results["errors"] += 1
                
                results["total_queries"] += 1
        
        return {
            "total_queries": results["total_queries"],
            "errors": results["errors"],
            "avg_latency": np.mean(results["latencies"]),
            "p50_latency": np.percentile(results["latencies"], 50),
            "p95_latency": np.percentile(results["latencies"], 95),
            "p99_latency": np.percentile(results["latencies"], 99),
            "throughput": results["total_queries"] / duration_seconds
        }
```

---

## Monitoring & Observability

### Logging Setup

```python
import logging
import json
from datetime import datetime
from functools import wraps

# Configure structured logging
class StructuredLogger:
    def __init__(self, name: str):
        self.logger = logging.getLogger(name)
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter('%(message)s'))
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
    
    def log(self, event: str, **kwargs):
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "event": event,
            **kwargs
        }
        self.logger.info(json.dumps(log_entry))

logger = StructuredLogger("rag_system")

def log_query(func):
    """Decorator to log queries"""
    @wraps(func)
    def wrapper(self, query: str, *args, **kwargs):
        start_time = time.time()
        
        try:
            result = func(self, query, *args, **kwargs)
            latency = time.time() - start_time
            
            logger.log(
                "query_completed",
                query=query[:100],
                latency_ms=latency * 1000,
                num_sources=len(result.sources) if hasattr(result, 'sources') else 0,
                top_score=result.sources[0].score if result.sources else 0
            )
            
            return result
        except Exception as e:
            logger.log(
                "query_failed",
                query=query[:100],
                error=str(e),
                latency_ms=(time.time() - start_time) * 1000
            )
            raise
    
    return wrapper
```

### Metrics Collection

```python
from dataclasses import dataclass, field
from collections import defaultdict
import threading

@dataclass
class MetricsCollector:
    """Collect and expose metrics"""
    
    query_count: int = 0
    error_count: int = 0
    latencies: List[float] = field(default_factory=list)
    retrieval_scores: List[float] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock)
    
    def record_query(self, latency: float, retrieval_score: float):
        with self._lock:
            self.query_count += 1
            self.latencies.append(latency)
            self.retrieval_scores.append(retrieval_score)
    
    def record_error(self):
        with self._lock:
            self.error_count += 1
    
    def get_metrics(self) -> dict:
        with self._lock:
            return {
                "total_queries": self.query_count,
                "total_errors": self.error_count,
                "error_rate": self.error_count / max(self.query_count, 1),
                "avg_latency_ms": np.mean(self.latencies) * 1000 if self.latencies else 0,
                "p95_latency_ms": np.percentile(self.latencies, 95) * 1000 if self.latencies else 0,
                "avg_retrieval_score": np.mean(self.retrieval_scores) if self.retrieval_scores else 0
            }

# Prometheus-style metrics (if using prometheus_client)
from prometheus_client import Counter, Histogram, Gauge

rag_queries_total = Counter('rag_queries_total', 'Total RAG queries')
rag_query_latency = Histogram('rag_query_latency_seconds', 'RAG query latency')
rag_retrieval_score = Histogram('rag_retrieval_score', 'Top retrieval score')
rag_errors_total = Counter('rag_errors_total', 'Total RAG errors')
```

### Tracing

```python
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode
import uuid

tracer = trace.get_tracer("rag_system")

class TracedRAGSystem:
    """RAG system with distributed tracing"""
    
    def __init__(self, retriever, generator):
        self.retriever = retriever
        self.generator = generator
    
    def query(self, question: str) -> dict:
        with tracer.start_as_current_span("rag_query") as span:
            query_id = str(uuid.uuid4())
            span.set_attribute("query_id", query_id)
            span.set_attribute("question", question[:100])
            
            try:
                # Retrieval span
                with tracer.start_as_current_span("retrieval") as retrieval_span:
                    results = self.retriever.retrieve(question)
                    retrieval_span.set_attribute("num_results", len(results))
                    retrieval_span.set_attribute("top_score", results[0].score if results else 0)
                
                # Generation span
                with tracer.start_as_current_span("generation") as gen_span:
                    context = self._assemble_context(results)
                    gen_span.set_attribute("context_length", len(context))
                    answer = self.generator.generate(question, context)
                    gen_span.set_attribute("answer_length", len(answer))
                
                span.set_status(Status(StatusCode.OK))
                return {"answer": answer, "sources": results, "query_id": query_id}
                
            except Exception as e:
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)
                raise
```

### Alerting Rules

```yaml
# Example Prometheus alerting rules
groups:
  - name: rag_alerts
    rules:
      - alert: HighErrorRate
        expr: rate(rag_errors_total[5m]) / rate(rag_queries_total[5m]) > 0.05
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "RAG error rate above 5%"
          
      - alert: HighLatency
        expr: histogram_quantile(0.95, rate(rag_query_latency_seconds_bucket[5m])) > 5
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "RAG p95 latency above 5 seconds"
          
      - alert: LowRetrievalQuality
        expr: histogram_quantile(0.5, rate(rag_retrieval_score_bucket[1h])) < 0.5
        for: 30m
        labels:
          severity: warning
        annotations:
          summary: "Median retrieval score below 0.5"
```

### Dashboard Metrics to Track

| Category | Metric | Description |
|----------|--------|-------------|
| **Latency** | p50, p95, p99 query latency | How fast is the system? |
| **Throughput** | Queries per second | How much load can it handle? |
| **Quality** | Average retrieval score | Are we finding relevant docs? |
| **Quality** | Faithfulness score (sampled) | Are answers grounded? |
| **Errors** | Error rate | What % of queries fail? |
| **Usage** | Unique queries per day | Is the system being used? |
| **Cost** | Tokens per query | What's the LLM cost? |
| **Vector DB** | Index size, query latency | Is the DB healthy? |

---

## Common Pitfalls & Solutions

### Pitfall 1: Poor Chunking

**Problem**: Chunks split in the middle of important context, losing meaning.

**Solution**:
```python
# Bad: Fixed size regardless of content
chunks = text.split_every(500)

# Good: Respect semantic boundaries
class SmartChunker:
    def chunk(self, text: str) -> List[str]:
        # First split by paragraphs
        paragraphs = text.split('\n\n')
        
        # Then combine small paragraphs, split large ones
        chunks = []
        current = ""
        
        for para in paragraphs:
            if len(current) + len(para) < self.max_size:
                current += para + "\n\n"
            else:
                if current:
                    chunks.append(current.strip())
                # If single paragraph too large, split by sentences
                if len(para) > self.max_size:
                    chunks.extend(self._split_by_sentences(para))
                else:
                    current = para + "\n\n"
        
        return chunks
```

### Pitfall 2: Embedding Model Mismatch

**Problem**: Using a general embedding model for domain-specific content.

**Solution**:
```python
# Option 1: Fine-tune embeddings on your domain
from sentence_transformers import SentenceTransformer, InputExample, losses

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Create training examples from your domain
train_examples = [
    InputExample(texts=['query', 'relevant_doc'], label=1.0),
    InputExample(texts=['query', 'irrelevant_doc'], label=0.0),
]

train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)
train_loss = losses.CosineSimilarityLoss(model)

model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=3)

# Option 2: Use domain-specific model
# For legal: "nlpaueb/legal-bert-base-uncased"
# For medical: "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
# For code: "microsoft/codebert-base"
```

### Pitfall 3: No Reranking

**Problem**: Vector similarity misses nuanced relevance.

**Solution**: Always rerank top results:
```python
class TwoStageRetriever:
    def retrieve(self, query: str, top_k: int = 5) -> List[SearchResult]:
        # Stage 1: Fast vector search (get more than needed)
        candidates = self.vector_store.search(query, top_k=top_k * 4)
        
        # Stage 2: Rerank with cross-encoder
        pairs = [(query, c.chunk.content) for c in candidates]
        scores = self.reranker.predict(pairs)
        
        # Return top-k reranked
        reranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
        return [r[0] for r in reranked[:top_k]]
```

### Pitfall 4: Context Window Stuffing

**Problem**: Stuffing too much context reduces answer quality.

**Solution**: Be selective with context:
```python
class AdaptiveContextBuilder:
    def build(self, results: List[SearchResult], max_tokens: int = 2000) -> str:
        # Only include high-scoring results
        filtered = [r for r in results if r.score > 0.5]
        
        # Diversify sources
        seen_sources = set()
        diverse_results = []
        for r in filtered:
            source = r.chunk.metadata.get('source')
            if source not in seen_sources:
                diverse_results.append(r)
                seen_sources.add(source)
        
        # Build context within token budget
        context_parts = []
        current_tokens = 0
        
        for r in diverse_results:
            chunk_tokens = len(r.chunk.content.split()) * 1.3  # Rough estimate
            if current_tokens + chunk_tokens > max_tokens:
                break
            context_parts.append(r.chunk.content)
            current_tokens += chunk_tokens
        
        return "\n\n---\n\n".join(context_parts)
```

### Pitfall 5: No Fallback for Low-Confidence Retrieval

**Problem**: System hallucinates when retrieval fails.

**Solution**: Detect and handle low-confidence scenarios:
```python
class SafeRAG:
    def query(self, question: str) -> dict:
        results = self.retriever.retrieve(question)
        
        # Check retrieval confidence
        top_score = results[0].score if results else 0
        
        if top_score < 0.3:
            return {
                "answer": "I don't have enough information to answer this question confidently.",
                "confidence": "low",
                "suggestion": "Try rephrasing your question or asking about a different topic."
            }
        
        if top_score < 0.6:
            # Generate but add caveat
            answer = self.generator.generate(question, self._build_context(results))
            return {
                "answer": f"Based on limited information: {answer}",
                "confidence": "medium",
                "sources": results
            }
        
        # High confidence - normal generation
        answer = self.generator.generate(question, self._build_context(results))
        return {
            "answer": answer,
            "confidence": "high", 
            "sources": results
        }
```

### Pitfall 6: Stale Data

**Problem**: Documents change but index doesn't update.

**Solution**: Implement incremental updates:
```python
class IncrementalIndexer:
    def __init__(self, vector_store, embedder):
        self.vector_store = vector_store
        self.embedder = embedder
        self.doc_hashes = {}  # Track document versions
    
    def sync(self, documents: List[Document]):
        """Incrementally sync documents"""
        for doc in documents:
            doc_hash = hashlib.md5(doc.content.encode()).hexdigest()
            doc_id = doc.metadata.get('id')
            
            if doc_id in self.doc_hashes:
                if self.doc_hashes[doc_id] == doc_hash:
                    continue  # No change
                else:
                    # Document changed - delete old chunks
                    self.vector_store.delete_by_metadata({"doc_id": doc_id})
            
            # Index new/changed document
            chunks = self.chunker.chunk(doc)
            for chunk in chunks:
                chunk.metadata['doc_id'] = doc_id
            
            embeddings = self.embedder.embed([c.content for c in chunks])
            self.vector_store.upsert(chunks, embeddings)
            
            self.doc_hashes[doc_id] = doc_hash
```

---

## Quick Start Checklist

### Before Launch

- [ ] Create evaluation dataset with 50+ examples
- [ ] Achieve >0.7 on retrieval metrics (MRR, recall@5)
- [ ] Achieve >0.8 faithfulness score
- [ ] Set up monitoring and alerting
- [ ] Run load tests for expected traffic
- [ ] Document chunking strategy and parameters
- [ ] Implement error handling and fallbacks

### After Launch

- [ ] Monitor latency p95 and p99
- [ ] Track retrieval score distribution
- [ ] Sample and evaluate answers weekly
- [ ] Collect user feedback
- [ ] A/B test improvements
- [ ] Update documents and re-index regularly

---

## Summary

Building a production RAG system requires attention to every component:

1. **Document Processing**: Clean, chunk semantically, preserve metadata
2. **Embeddings**: Choose/fine-tune for your domain
3. **Retrieval**: Use hybrid + reranking for best results
4. **Generation**: Craft prompts, handle low-confidence gracefully
5. **Evaluation**: Measure retrieval AND generation quality
6. **Testing**: Unit, integration, regression, and load tests
7. **Monitoring**: Track latency, quality, errors, and costs

The key insight: **A RAG system is only as good as its retrieval**. Invest heavily in getting retrieval right—the best LLM can't help if it doesn't get the right context.