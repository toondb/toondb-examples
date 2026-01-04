"""
ToonDB RAG System - Document Models and Loader
"""
from pathlib import Path
from typing import Dict, Any, List
from dataclasses import dataclass, field
import hashlib
import re


@dataclass
class Document:
    """Represents a loaded document"""
    content: str
    metadata: Dict[str, Any]
    id: str = field(default="")
    
    def __post_init__(self):
        if not self.id:
            self.id = self._generate_id()
    
    def _generate_id(self) -> str:
        return hashlib.md5(self.content.encode()).hexdigest()


@dataclass 
class Chunk:
    """Represents a chunk of a document"""
    content: str
    metadata: Dict[str, Any]
    start_index: int
    end_index: int
    id: str = field(default="")
    
    def __post_init__(self):
        if not self.id:
            self.id = hashlib.md5(
                f"{self.content}{self.start_index}".encode()
            ).hexdigest()


class DocumentLoader:
    """Load documents from various sources"""
    
    def load_pdf(self, path: Path) -> Document:
        """Load a PDF document using PyMuPDF"""
        import fitz  # PyMuPDF
        
        doc = fitz.open(path)
        text_parts = []
        
        for page_num, page in enumerate(doc):
            page_text = page.get_text()
            text_parts.append(f"[Page {page_num + 1}]\n{page_text}")
        
        text = "\n\n".join(text_parts)
        
        return Document(
            content=text,
            metadata={
                "source": str(path),
                "filename": path.name,
                "type": "pdf",
                "pages": len(doc)
            }
        )
    
    def load_markdown(self, path: Path) -> Document:
        """Load a Markdown document"""
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        return Document(
            content=content,
            metadata={
                "source": str(path),
                "filename": path.name,
                "type": "markdown"
            }
        )
    
    def load_text(self, path: Path) -> Document:
        """Load a plain text document"""
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        return Document(
            content=content,
            metadata={
                "source": str(path),
                "filename": path.name,
                "type": "text"
            }
        )
    
    def load(self, path: Path) -> Document:
        """Auto-detect and load document"""
        suffix = path.suffix.lower()
        
        if suffix == '.pdf':
            return self.load_pdf(path)
        elif suffix in ['.md', '.markdown']:
            return self.load_markdown(path)
        elif suffix in ['.txt', '.text']:
            return self.load_text(path)
        else:
            # Try as text
            return self.load_text(path)
    
    def load_directory(self, directory: Path, extensions: List[str] = None) -> List[Document]:
        """Load all documents from a directory"""
        if extensions is None:
            extensions = ['.pdf', '.md', '.txt']
        
        documents = []
        for ext in extensions:
            for file_path in directory.glob(f"*{ext}"):
                try:
                    doc = self.load(file_path)
                    documents.append(doc)
                    print(f"✅ Loaded: {file_path.name}")
                except Exception as e:
                    print(f"❌ Failed to load {file_path.name}: {e}")
        
        return documents


class TextPreprocessor:
    """Clean and normalize text before chunking"""
    
    def clean(self, text: str) -> str:
        """Clean text by removing special characters and normalizing whitespace"""
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
