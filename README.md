# ToonDB RAG Examples

Production-ready **Retrieval-Augmented Generation (RAG)** implementations using ToonDB as the vector database, with examples in Python, Node.js, Go, and Rust.

## 🚀 Quick Start

Each implementation follows the same architecture and can be configured with Azure OpenAI or any OpenAI-compatible API.

### Python
```bash
cd toondb_rag
pip install toondb-client openai python-dotenv numpy
cp .env.example .env  # Add your API keys
python demo.py
```

### Node.js
```bash
cd toondb_rag_node
npm install
cp .env.example .env  # Add your API keys
node demo.js
```

### Go
```bash
cd toondb_rag_go
go mod download
cp .env.example .env  # Add your API keys
go run ./cmd/demo
```

### Rust
```bash
cd toondb_rag_rust
cargo build
cp .env.example .env  # Add your API keys
cargo run
```

## 📁 Project Structure

All implementations share the same modular architecture:

```
toondb_rag_<lang>/
├── config       # Configuration management
├── documents    # Document loading & preprocessing
├── chunking     # Text chunking strategies (fixed, semantic)
├── embeddings   # Azure OpenAI embedding integration
├── vectorstore  # ToonDB vector storage
├── generation   # LLM response generation
├── rag          # Main orchestration
└── demo         # Demo entry point
```

## ⚙️ Configuration

Copy `.env.example` to `.env` and configure:

```env
# Azure OpenAI
AZURE_OPENAI_API_KEY=your_key
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_EMBEDDING_DEPLOYMENT=text-embedding-3-small
AZURE_OPENAI_CHAT_DEPLOYMENT=gpt-4

# ToonDB
TOONDB_PATH=./toondb_data

# RAG Settings
CHUNK_SIZE=512
CHUNK_OVERLAP=50
TOP_K=5
MAX_CONTEXT_LENGTH=4000
```

## 📖 Usage Examples

### Python

```python
from rag import ToonDBRAG

# Initialize
rag = ToonDBRAG(db_path="./my_rag_db")

# Ingest documents
rag.ingest_file("./docs/documentation.md")
rag.ingest_directory("./docs", extensions=[".md", ".txt"])

# Query
response = rag.query("What is ToonDB?")
print(response.answer)
print(f"Confidence: {response.confidence}")
print(f"Sources: {len(response.sources)}")
```

### Node.js

```javascript
const { ToonDBRAG } = require('./src/rag');

async function main() {
  const rag = new ToonDBRAG({ dbPath: './my_rag_db' });
  await rag.initialize();
  
  await rag.ingestFile('./docs/documentation.md');
  
  const response = await rag.query('What is ToonDB?');
  console.log(response.answer);
}

main();
```

### Go

```go
package main

import (
    "github.com/toondb/toondb-examples/toondb_rag_go/internal/rag"
    "github.com/toondb/toondb-examples/toondb_rag_go/internal/config"
)

func main() {
    cfg, _ := config.Load()
    ragSystem := rag.New(cfg, rag.Options{
        DBPath: "./my_rag_db",
    })
    defer ragSystem.Close()
    
    ragSystem.IngestFile("./docs/documentation.md")
    
    response, _ := ragSystem.Query("What is ToonDB?")
    fmt.Println(response.Answer)
}
```

### Rust

```rust
use toondb_rag::{ToonDBRAG, RAGOptions};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let config = Config::load();
    let mut rag = ToonDBRAG::new(&config, RAGOptions {
        db_path: Some("./my_rag_db".to_string()),
        ..Default::default()
    });
    
    rag.ingest_file("./docs/documentation.md").await?;
    
    let response = rag.query("What is ToonDB?").await?;
    println!("{}", response.answer);
    
    Ok(())
}
```

## 🔧 Features

| Feature | Python | Node.js | Go | Rust |
|---------|--------|---------|-----|------|
| Azure OpenAI Embeddings | ✅ | ✅ | ✅ | ✅ |
| Azure OpenAI LLM | ✅ | ✅ | ✅ | ✅ |
| ToonDB Persistence | ✅ | ✅ | ✅ | ✅ |
| Semantic Chunking | ✅ | ✅ | ✅ | ✅ |
| Fixed-Size Chunking | ✅ | ✅ | ✅ | ✅ |
| Mock Mode (Testing) | ✅ | ✅ | ✅ | ✅ |
| Confidence Scores | ✅ | ✅ | ✅ | ✅ |
| Source Citations | ✅ | ✅ | ✅ | ✅ |

## 📦 Dependencies

### Python
- `toondb-client` >= 0.3.0
- `openai` >= 1.0.0
- `python-dotenv`
- `numpy`

### Node.js
- `@sushanth/toondb` >= 0.3.0
- `openai` >= 4.0.0
- `dotenv`

### Go
- `github.com/toondb/toondb-go` >= 0.3.0
- `github.com/joho/godotenv`

### Rust
- `toondb` = "0.3"
- `tokio`
- `reqwest`
- `serde`

## 🧪 Testing

Each implementation includes a demo script that:
1. Creates a sample ToonDB documentation file
2. Ingests and chunks the document
3. Generates embeddings via Azure OpenAI
4. Stores vectors in ToonDB (persistent)
5. Runs 5 test queries with real LLM responses

## 📄 License

MIT License - see LICENSE file for details.

## 🔗 Links

- [ToonDB Documentation](https://toondb.io)
- [ToonDB Python SDK](https://pypi.org/project/toondb-client/)
- [ToonDB Node.js SDK](https://www.npmjs.com/package/@sushanth/toondb)
- [ToonDB Go SDK](https://pkg.go.dev/github.com/toondb/toondb-go)
- [ToonDB Rust SDK](https://crates.io/crates/toondb)
