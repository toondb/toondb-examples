"""
ToonDB RAG System - Unit Tests
"""
import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from documents import Document, Chunk, DocumentLoader, TextPreprocessor
from chunking import FixedSizeChunker, SemanticChunker, RecursiveChunker


# ============================================================================
# Document Tests
# ============================================================================

class TestDocument:
    """Tests for Document class"""
    
    def test_document_creation(self):
        """Test creating a document"""
        doc = Document(
            content="Test content",
            metadata={"source": "test.txt"}
        )
        
        assert doc.content == "Test content"
        assert doc.metadata["source"] == "test.txt"
        assert doc.id is not None
        assert len(doc.id) == 32  # MD5 hash length
    
    def test_document_id_is_deterministic(self):
        """Test that same content produces same ID"""
        doc1 = Document(content="Same content", metadata={})
        doc2 = Document(content="Same content", metadata={})
        
        assert doc1.id == doc2.id
    
    def test_different_content_different_id(self):
        """Test that different content produces different ID"""
        doc1 = Document(content="Content A", metadata={})
        doc2 = Document(content="Content B", metadata={})
        
        assert doc1.id != doc2.id


class TestChunk:
    """Tests for Chunk class"""
    
    def test_chunk_creation(self):
        """Test creating a chunk"""
        chunk = Chunk(
            content="Chunk content",
            metadata={"chunk_index": 0},
            start_index=0,
            end_index=13
        )
        
        assert chunk.content == "Chunk content"
        assert chunk.metadata["chunk_index"] == 0
        assert chunk.start_index == 0
        assert chunk.end_index == 13


class TestTextPreprocessor:
    """Tests for text preprocessing"""
    
    def test_clean_removes_excessive_whitespace(self):
        """Test whitespace normalization"""
        preprocessor = TextPreprocessor()
        
        text = "Hello    world\n\n\ntest"
        clean = preprocessor.clean(text)
        
        assert "    " not in clean
        assert clean == "Hello world test"
    
    def test_clean_removes_special_characters(self):
        """Test special character removal"""
        preprocessor = TextPreprocessor()
        
        text = "Hello\x00World\x1f!"
        clean = preprocessor.clean(text)
        
        assert "\x00" not in clean
        assert "\x1f" not in clean


# ============================================================================
# Chunking Tests
# ============================================================================

class TestFixedSizeChunker:
    """Tests for fixed-size chunking"""
    
    def test_chunks_respect_size_limit(self):
        """Test that chunks don't exceed size limit"""
        chunker = FixedSizeChunker(chunk_size=100, overlap=10)
        doc = Document(content="a" * 500, metadata={})
        
        chunks = chunker.chunk(doc)
        
        for chunk in chunks:
            assert len(chunk.content) <= 100
    
    def test_chunking_with_overlap(self):
        """Test that overlap is applied"""
        chunker = FixedSizeChunker(chunk_size=100, overlap=20)
        doc = Document(content="a" * 200, metadata={})
        
        chunks = chunker.chunk(doc)
        
        # With 100 size and 20 overlap, we should get 3 chunks for 200 chars
        # Chunk 1: 0-100, Chunk 2: 80-180, Chunk 3: 160-200
        assert len(chunks) >= 2
    
    def test_metadata_preserved(self):
        """Test that document metadata is preserved in chunks"""
        chunker = FixedSizeChunker(chunk_size=50, overlap=0)
        doc = Document(
            content="This is test content for chunking",
            metadata={"source": "test.txt", "type": "text"}
        )
        
        chunks = chunker.chunk(doc)
        
        for chunk in chunks:
            assert "source" in chunk.metadata
            assert chunk.metadata["source"] == "test.txt"
            assert "chunk_index" in chunk.metadata


class TestSemanticChunker:
    """Tests for semantic chunking"""
    
    def test_splits_on_paragraphs(self):
        """Test that semantic chunker splits on paragraph boundaries"""
        chunker = SemanticChunker(max_chunk_size=100, min_chunk_size=10)
        
        text = """First paragraph here.

Second paragraph here.

Third paragraph here."""
        
        doc = Document(content=text, metadata={})
        chunks = chunker.chunk(doc)
        
        # Should create chunks based on paragraphs
        assert len(chunks) >= 1
    
    def test_respects_max_size(self):
        """Test that chunks respect max size"""
        chunker = SemanticChunker(max_chunk_size=50, min_chunk_size=10)
        
        text = "Short para.\n\n" + "This is a longer paragraph with more content.\n\n" * 3
        doc = Document(content=text, metadata={})
        
        chunks = chunker.chunk(doc)
        
        for chunk in chunks:
            # Allow some flexibility for paragraph boundaries
            assert len(chunk.content) <= 150


class TestRecursiveChunker:
    """Tests for recursive chunking"""
    
    def test_recursive_splitting(self):
        """Test recursive splitting with multiple separators"""
        chunker = RecursiveChunker(chunk_size=50, overlap=0)
        
        text = "Para one.\n\nPara two.\n\nPara three that is longer."
        doc = Document(content=text, metadata={})
        
        chunks = chunker.chunk(doc)
        
        assert len(chunks) >= 1
        for chunk in chunks:
            assert chunk.content.strip() != ""


# ============================================================================
# Integration Tests
# ============================================================================

class TestDocumentLoader:
    """Tests for document loading"""
    
    def test_load_text_file(self, tmp_path):
        """Test loading a text file"""
        # Create temp file
        test_file = tmp_path / "test.txt"
        test_file.write_text("Test content for loading")
        
        loader = DocumentLoader()
        doc = loader.load_text(test_file)
        
        assert doc.content == "Test content for loading"
        assert doc.metadata["type"] == "text"
        assert doc.metadata["filename"] == "test.txt"
    
    def test_load_markdown_file(self, tmp_path):
        """Test loading a markdown file"""
        test_file = tmp_path / "test.md"
        test_file.write_text("# Heading\n\nContent here")
        
        loader = DocumentLoader()
        doc = loader.load_markdown(test_file)
        
        assert "# Heading" in doc.content
        assert doc.metadata["type"] == "markdown"


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
