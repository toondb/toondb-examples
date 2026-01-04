/**
 * ToonDB RAG System - Configuration
 */
require('dotenv').config();

const config = {
    azure: {
        apiKey: process.env.AZURE_OPENAI_API_KEY || '',
        endpoint: process.env.AZURE_OPENAI_ENDPOINT || '',
        apiVersion: process.env.AZURE_OPENAI_API_VERSION || '2024-12-01-preview',
        chatDeployment: process.env.AZURE_OPENAI_CHAT_DEPLOYMENT || 'gpt-4.1',
        embeddingDeployment: process.env.AZURE_OPENAI_EMBEDDING_DEPLOYMENT || 'text-embedding-3-small'
    },
    toondb: {
        path: process.env.TOONDB_PATH || './toondb_data'
    },
    rag: {
        chunkSize: parseInt(process.env.CHUNK_SIZE || '512'),
        chunkOverlap: parseInt(process.env.CHUNK_OVERLAP || '50'),
        topK: parseInt(process.env.TOP_K || '5'),
        maxContextLength: parseInt(process.env.MAX_CONTEXT_LENGTH || '4000')
    }
};

module.exports = config;
