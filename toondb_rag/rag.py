"""
ToonDB RAG System - Main RAG Class
"""
from typing import List, Optional
from pathlib import Path

from documents import Document, DocumentLoader, TextPreprocessor, Chunk
from chunking import get_chunker, ChunkingStrategy
from embeddings import AzureEmbeddings, get_embeddings
from vector_store import ToonDBVectorStore, SearchResult
from retrieval import BasicRetriever, get_retriever
from generation import AzureLLMGenerator, RAGResponse
from config import get_rag_config


class ToonDBRAG:
    """
    Complete RAG System using ToonDB
    
    Features:
    - PDF, Markdown, Text document ingestion
    - Semantic chunking with configurable strategies
    - Azure OpenAI embeddings
    - ToonDB vector storage
    - Multiple retrieval strategies
    - Response generation with citations
    
    Usage:
        rag = ToonDBRAG()
        
        # Ingest documents
        rag.ingest_directory("./documents")
        
        # Query
        response = rag.query("What is the main topic?")
        print(response.answer)
    """
    
    def __init__(
        self,
        db_path: str = None,
        chunking_strategy: str = "semantic",
        retrieval_strategy: str = "basic",
        use_azure: bool = True,
        use_mock: bool = False
    ):
        # Configuration
        config = get_rag_config()
        
        # Components
        self.loader = DocumentLoader()
        self.preprocessor = TextPreprocessor()
        self.chunker = get_chunker(
            strategy=chunking_strategy,
            max_chunk_size=config.chunk_size,
            min_chunk_size=config.chunk_size // 4
        )
        self.embedder = get_embeddings(use_azure=use_azure, use_mock=use_mock)
        self.vector_store = ToonDBVectorStore(db_path=db_path)
        self.retriever = get_retriever(
            self.vector_store,
            self.embedder,
            strategy=retrieval_strategy
        )
        
        if use_mock:
            from generation import MockLLMGenerator
            self.generator = MockLLMGenerator()
        else:
            self.generator = AzureLLMGenerator()

        
        # Config
        self.top_k = config.top_k
        
        # Track ingested documents
        self._ingested_docs = []
    
    def ingest(self, documents: List[Document]) -> int:
        """Ingest documents into the RAG system"""
        all_chunks = []
        
        for doc in documents:
            # Preprocess
            doc.content = self.preprocessor.clean(doc.content)
            doc.content = self.preprocessor.remove_boilerplate(
                doc.content, 
                doc.metadata.get("type", "")
            )
            
            # Chunk
            chunks = self.chunker.chunk(doc)
            all_chunks.extend(chunks)
            
            self._ingested_docs.append(doc.metadata.get("filename", doc.id))
        
        if not all_chunks:
            print("⚠️ No chunks generated from documents")
            return 0
        
        # Embed
        print(f"🔄 Embedding {len(all_chunks)} chunks...")
        texts = [chunk.content for chunk in all_chunks]
        embeddings = self.embedder.embed(texts)
        
        # Store
        self.vector_store.upsert(all_chunks, embeddings)
        
        print(f"✅ Ingested {len(documents)} documents ({len(all_chunks)} chunks)")
        return len(all_chunks)
    
    def ingest_file(self, file_path: str) -> int:
        """Ingest a single file"""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        doc = self.loader.load(path)
        return self.ingest([doc])
    
    def ingest_directory(self, directory: str, extensions: List[str] = None) -> int:
        """Ingest all documents from a directory"""
        path = Path(directory)
        if not path.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")
        
        documents = self.loader.load_directory(path, extensions)
        return self.ingest(documents)
    
    def query(self, question: str, top_k: int = None) -> RAGResponse:
        """Query the RAG system"""
        top_k = top_k or self.top_k
        
        # Retrieve relevant chunks
        results = self.retriever.retrieve(question, top_k)
        
        # Generate response
        response = self.generator.generate_with_sources(question, results)
        
        return response
    
    def search(self, query: str, top_k: int = None) -> List[SearchResult]:
        """Search for relevant chunks without generation"""
        top_k = top_k or self.top_k
        return self.retriever.retrieve(query, top_k)
    
    def get_stats(self) -> dict:
        """Get system statistics"""
        return {
            "total_chunks": self.vector_store.count(),
            "ingested_documents": len(self._ingested_docs),
            "document_names": self._ingested_docs
        }
    
    def clear(self):
        """Clear all data"""
        self.vector_store.clear()
        self._ingested_docs = []
        print("🗑️ Cleared all data")
    
    def close(self):
        """Close connections"""
        self.vector_store.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


# Convenience function
def create_rag(
    db_path: str = None,
    chunking_strategy: str = "semantic",
    retrieval_strategy: str = "basic"
) -> ToonDBRAG:
    """Create a ToonDB RAG system"""
    return ToonDBRAG(
        db_path=db_path,
        chunking_strategy=chunking_strategy,
        retrieval_strategy=retrieval_strategy
    )
