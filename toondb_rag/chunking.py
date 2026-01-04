"""
ToonDB RAG System - Text Chunking Strategies
"""
from typing import List
from abc import ABC, abstractmethod
import re

from documents import Document, Chunk


class ChunkingStrategy(ABC):
    """Abstract base class for chunking strategies"""
    
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
            end = min(start + self.chunk_size, len(text))
            chunk_text = text[start:end]
            
            # Try to break at word boundary
            if end < len(text) and not text[end].isspace():
                last_space = chunk_text.rfind(' ')
                if last_space > self.chunk_size // 2:
                    end = start + last_space
                    chunk_text = text[start:end]
            
            chunks.append(Chunk(
                content=chunk_text.strip(),
                metadata={
                    **document.metadata,
                    "chunk_index": len(chunks),
                    "doc_id": document.id
                },
                start_index=start,
                end_index=end
            ))
            
            # Calculate next start position with overlap
            next_start = end - self.overlap
            
            # Ensure we always advance (prevent infinite loop)
            if next_start <= start:
                next_start = end
            
            # If we've reached the end, break
            if end >= len(text):
                break
                
            start = next_start
        
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
            para = para.strip()
            if not para:
                continue
                
            if len(current_chunk) + len(para) + 2 <= self.max_chunk_size:
                current_chunk += para + "\n\n"
            else:
                if len(current_chunk) >= self.min_chunk_size:
                    chunks.append(Chunk(
                        content=current_chunk.strip(),
                        metadata={
                            **document.metadata,
                            "chunk_index": len(chunks),
                            "doc_id": document.id
                        },
                        start_index=current_start,
                        end_index=current_start + len(current_chunk)
                    ))
                current_start += len(current_chunk)
                current_chunk = para + "\n\n"
        
        # Don't forget the last chunk
        if current_chunk.strip() and len(current_chunk.strip()) >= self.min_chunk_size:
            chunks.append(Chunk(
                content=current_chunk.strip(),
                metadata={
                    **document.metadata,
                    "chunk_index": len(chunks),
                    "doc_id": document.id
                },
                start_index=current_start,
                end_index=current_start + len(current_chunk)
            ))
        
        return chunks


class RecursiveChunker(ChunkingStrategy):
    """Recursively split by different separators"""
    
    def __init__(self, chunk_size: int = 512, overlap: int = 50):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.separators = ["\n\n", "\n", ". ", " "]
    
    def chunk(self, document: Document) -> List[Chunk]:
        raw_chunks = self._split_text(document.content, self.separators)
        
        chunks = []
        for i, content in enumerate(raw_chunks):
            chunks.append(Chunk(
                content=content.strip(),
                metadata={
                    **document.metadata,
                    "chunk_index": i,
                    "doc_id": document.id
                },
                start_index=0,
                end_index=len(content)
            ))
        
        return chunks
    
    def _split_text(self, text: str, separators: List[str]) -> List[str]:
        if not separators:
            return [text] if text.strip() else []
        
        separator = separators[0]
        splits = text.split(separator)
        
        chunks = []
        current_chunk = ""
        
        for split in splits:
            if len(current_chunk) + len(split) + len(separator) <= self.chunk_size:
                current_chunk += split + separator
            else:
                if current_chunk.strip():
                    if len(current_chunk) > self.chunk_size and len(separators) > 1:
                        # Recursively split with next separator
                        chunks.extend(self._split_text(current_chunk, separators[1:]))
                    else:
                        chunks.append(current_chunk.strip())
                current_chunk = split + separator
        
        if current_chunk.strip():
            if len(current_chunk) > self.chunk_size and len(separators) > 1:
                chunks.extend(self._split_text(current_chunk, separators[1:]))
            else:
                chunks.append(current_chunk.strip())
        
        return chunks


def get_chunker(strategy: str = "semantic", **kwargs) -> ChunkingStrategy:
    """Factory function to get chunker by strategy name"""
    strategies = {
        "fixed": FixedSizeChunker,
        "semantic": SemanticChunker,
        "recursive": RecursiveChunker
    }
    
    if strategy not in strategies:
        raise ValueError(f"Unknown chunking strategy: {strategy}")
    
    return strategies[strategy](**kwargs)
