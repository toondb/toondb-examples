"""
ToonDB RAG System - RAG Integration Tests
"""
import sys
from pathlib import Path
import pytest
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from documents import Document, Chunk
from vector_store import ToonDBVectorStore, SearchResult
from chunking import SemanticChunker


# ============================================================================
# Vector Store Tests
# ============================================================================

class TestToonDBVectorStore:
    """Tests for ToonDB vector store"""
    
    @pytest.fixture
    def vector_store(self, tmp_path):
        """Create a temporary vector store"""
        store = ToonDBVectorStore(db_path=str(tmp_path / "test_db"))
        yield store
        store.close()
    
    def test_upsert_and_count(self, vector_store):
        """Test upserting chunks and counting"""
        chunks = [
            Chunk(content="First chunk", metadata={}, start_index=0, end_index=11),
            Chunk(content="Second chunk", metadata={}, start_index=12, end_index=24),
        ]
        embeddings = np.random.rand(2, 1536).astype(np.float32)
        
        vector_store.upsert(chunks, embeddings)
        
        assert vector_store.count() == 2
    
    def test_search_returns_results(self, vector_store):
        """Test that search returns relevant results"""
        chunks = [
            Chunk(content="Python programming", metadata={}, start_index=0, end_index=18),
            Chunk(content="Java development", metadata={}, start_index=0, end_index=16),
            Chunk(content="Python is great", metadata={}, start_index=0, end_index=15),
        ]
        
        # Create embeddings - make Python ones similar
        embeddings = np.array([
            [1.0, 0.0, 0.0] + [0.0] * 1533,  # Python programming
            [0.0, 1.0, 0.0] + [0.0] * 1533,  # Java (different)
            [0.9, 0.1, 0.0] + [0.0] * 1533,  # Python is great (similar to first)
        ], dtype=np.float32)
        
        vector_store.upsert(chunks, embeddings)
        
        # Search with query similar to Python
        query = np.array([1.0, 0.0, 0.0] + [0.0] * 1533, dtype=np.float32)
        results = vector_store.search(query, top_k=2)
        
        assert len(results) == 2
        assert results[0].score > 0.8  # High similarity
    
    def test_search_scores_are_sorted(self, vector_store):
        """Test that search results are sorted by score"""
        chunks = [
            Chunk(content=f"Chunk {i}", metadata={}, start_index=0, end_index=7)
            for i in range(5)
        ]
        embeddings = np.random.rand(5, 1536).astype(np.float32)
        
        vector_store.upsert(chunks, embeddings)
        
        query = np.random.rand(1536).astype(np.float32)
        results = vector_store.search(query, top_k=5)
        
        # Verify descending order
        for i in range(len(results) - 1):
            assert results[i].score >= results[i + 1].score
    
    def test_delete_chunks(self, vector_store):
        """Test deleting chunks"""
        chunks = [
            Chunk(content="Keep this", metadata={}, start_index=0, end_index=9),
            Chunk(content="Delete this", metadata={}, start_index=0, end_index=11),
        ]
        embeddings = np.random.rand(2, 1536).astype(np.float32)
        
        vector_store.upsert(chunks, embeddings)
        assert vector_store.count() == 2
        
        vector_store.delete([chunks[1].id])
        assert vector_store.count() == 1
    
    def test_clear_removes_all(self, vector_store):
        """Test clearing all data"""
        chunks = [
            Chunk(content=f"Chunk {i}", metadata={}, start_index=0, end_index=7)
            for i in range(3)
        ]
        embeddings = np.random.rand(3, 1536).astype(np.float32)
        
        vector_store.upsert(chunks, embeddings)
        assert vector_store.count() == 3
        
        vector_store.clear()
        assert vector_store.count() == 0


# ============================================================================
# Retrieval Tests (Mock)
# ============================================================================

class TestRetrieval:
    """Tests for retrieval functionality"""
    
    def test_search_result_structure(self):
        """Test SearchResult dataclass"""
        chunk = Chunk(
            content="Test content",
            metadata={"source": "test.txt"},
            start_index=0,
            end_index=12
        )
        result = SearchResult(chunk=chunk, score=0.85)
        
        assert result.chunk.content == "Test content"
        assert result.score == 0.85
    
    def test_cosine_similarity_calculation(self):
        """Test that cosine similarity is calculated correctly"""
        # Identical vectors should have similarity of 1.0
        vec = np.array([1.0, 2.0, 3.0])
        vec_norm = vec / np.linalg.norm(vec)
        similarity = np.dot(vec_norm, vec_norm)
        
        assert abs(similarity - 1.0) < 0.0001
        
        # Orthogonal vectors should have similarity of 0.0
        vec1 = np.array([1.0, 0.0, 0.0])
        vec2 = np.array([0.0, 1.0, 0.0])
        similarity = np.dot(vec1, vec2)
        
        assert abs(similarity) < 0.0001


# ============================================================================
# End-to-End Tests (Without API calls)
# ============================================================================

class TestEndToEnd:
    """End-to-end tests without external API calls"""
    
    def test_document_to_chunks_pipeline(self):
        """Test complete document processing pipeline"""
        # Create document
        doc = Document(
            content="""First paragraph about Python.

Second paragraph about databases.

Third paragraph about AI and machine learning.""",
            metadata={"source": "test.txt"}
        )
        
        # Chunk
        chunker = SemanticChunker(max_chunk_size=100, min_chunk_size=20)
        chunks = chunker.chunk(doc)
        
        # Verify
        assert len(chunks) >= 1
        assert all(c.content for c in chunks)
        assert all("source" in c.metadata for c in chunks)
    
    def test_chunk_metadata_includes_doc_id(self):
        """Test that chunks include document ID"""
        doc = Document(
            content="Test content for testing",
            metadata={"source": "test.txt"}
        )
        
        chunker = SemanticChunker(max_chunk_size=1000, min_chunk_size=10)
        chunks = chunker.chunk(doc)
        
        assert len(chunks) >= 1
        assert all("doc_id" in c.metadata for c in chunks)


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
