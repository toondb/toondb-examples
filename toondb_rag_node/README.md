# ToonDB RAG System (Node.js)

A production-ready RAG system built with ToonDB and Azure OpenAI for Node.js.

## Quick Start

```bash
# Install dependencies
npm install

# Run demo
node demo.js
```

## Project Structure

```
toondb_rag_node/
├── src/
│   ├── config.js       # Configuration from .env
│   ├── documents.js    # Document/Chunk models, loaders
│   ├── chunking.js     # Text chunking strategies
│   ├── embeddings.js   # Azure OpenAI embeddings
│   ├── vectorStore.js  # ToonDB vector storage
│   ├── generation.js   # LLM generation with Azure
│   └── rag.js          # Main RAG class
├── demo.js             # Demo script
├── .env                # Configuration
└── package.json
```

## Usage

```javascript
const { ToonDBRAG } = require('./src/rag');

async function main() {
  const rag = new ToonDBRAG({ dbPath: './my_rag_db' });

  // Ingest documents
  await rag.ingestFile('./docs/my_doc.md');

  // Query
  const response = await rag.query('What is the main topic?');
  console.log(response.answer);

  await rag.close();
}
```

## Configuration

Edit `.env`:
```
AZURE_OPENAI_API_KEY=your_key
AZURE_OPENAI_ENDPOINT=your_endpoint
AZURE_OPENAI_EMBEDDING_DEPLOYMENT=text-embedding-3-small
AZURE_OPENAI_CHAT_DEPLOYMENT=gpt-4.1
```
