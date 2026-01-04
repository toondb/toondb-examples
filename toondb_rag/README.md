# ToonDB RAG System

A production-ready RAG (Retrieval-Augmented Generation) system built with ToonDB and Azure OpenAI.

## Features

- 📄 **PDF/Markdown/Text ingestion** - Load documents from various formats
- 🔧 **Semantic chunking** - Smart text splitting for optimal retrieval
- 🧠 **Azure OpenAI embeddings** - High-quality vector representations
- 🗄️ **ToonDB vector storage** - Fast similarity search with HNSW
- 🔍 **Multiple retrieval strategies** - Basic, threshold, and MMR
- 💬 **Citation-aware generation** - Answers with source references
- ✅ **Comprehensive testing** - Unit and integration tests

## Quick Start

### 1. Install Dependencies

```bash
cd toondb_rag
source ../.venv/bin/activate  # Use existing venv
pip install -r requirements.txt
```

### 2. Configure Environment

Edit `.env` with your Azure OpenAI credentials:

```env
AZURE_OPENAI_API_KEY=your_key
AZURE_OPENAI_ENDPOINT=your_endpoint
AZURE_OPENAI_EMBEDDING_DEPLOYMENT=embedding
AZURE_OPENAI_CHAT_DEPLOYMENT=gpt-4.1
```

### 3. Ingest Documents

```bash
# Ingest a single PDF
python main.py ingest ./documents/my_document.pdf

# Ingest a directory
python main.py ingest ./documents/
```

### 4. Query

```bash
# Single query
python main.py query "What is the main topic?"

# Interactive mode
python main.py interactive
```

## Project Structure

```
toondb_rag/
├── __init__.py         # Package init
├── config.py           # Configuration from .env
├── documents.py        # Document/Chunk models, loaders
├── chunking.py         # Text chunking strategies
├── embeddings.py       # Azure OpenAI embeddings
├── vector_store.py     # ToonDB vector storage
├── retrieval.py        # Retrieval strategies
├── generation.py       # LLM generation with Azure
├── rag.py              # Main RAG class
├── main.py             # CLI entry point
├── demo.py             # Demo script
├── requirements.txt    # Dependencies
├── .env                # Configuration (not in git)
├── documents/          # Place your PDFs here
└── tests/
    ├── test_documents.py
    └── test_rag.py
```

## Usage Examples

### Python API

```python
from rag import ToonDBRAG

# Create RAG system
with ToonDBRAG() as rag:
    # Ingest documents
    rag.ingest_directory("./documents")
    
    # Query
    response = rag.query("What are the key features?")
    
    print(f"Answer: {response.answer}")
    print(f"Confidence: {response.confidence}")
    print(f"Sources: {len(response.sources)}")
```

### CLI

```bash
# Show help
python main.py --help

# Ingest and query
python main.py ingest ./my_docs/
python main.py query "Explain the architecture"

# Interactive chat
python main.py interactive

# Check stats
python main.py stats

# Clear all data
python main.py clear
```

## Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_documents.py -v
```

## Architecture

```
┌─────────────┐    ┌───────────┐    ┌────────────┐    ┌─────────┐
│  Documents  │───▶│  Chunking │───▶│ Embeddings │───▶│ ToonDB  │
│  (PDF/MD)   │    │ (Semantic)│    │  (Azure)   │    │ (Store) │
└─────────────┘    └───────────┘    └────────────┘    └─────────┘
                                                           │
┌─────────────┐    ┌───────────┐    ┌────────────┐         │
│  Response   │◀───│   LLM     │◀───│  Retriever │◀────────┘
│  (Answer)   │    │  (GPT-4)  │    │   (Top-K)  │
└─────────────┘    └───────────┘    └────────────┘
```

## Configuration

| Variable | Description | Default |
|----------|-------------|---------|
| `AZURE_OPENAI_API_KEY` | Azure OpenAI API key | Required |
| `AZURE_OPENAI_ENDPOINT` | Azure endpoint URL | Required |
| `AZURE_OPENAI_EMBEDDING_DEPLOYMENT` | Embedding model deployment | `embedding` |
| `AZURE_OPENAI_CHAT_DEPLOYMENT` | Chat model deployment | `gpt-4.1` |
| `TOONDB_PATH` | ToonDB storage path | `./toondb_data` |
| `CHUNK_SIZE` | Max chunk size in chars | `512` |
| `TOP_K` | Number of chunks to retrieve | `5` |
| `MAX_CONTEXT_LENGTH` | Max context for LLM | `4000` |

## License

MIT
