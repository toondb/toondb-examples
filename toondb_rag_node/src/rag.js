/**
 * ToonDB RAG System - Main RAG Class
 */
const { DocumentLoader, TextPreprocessor } = require('./documents');
const { SemanticChunker, getChunker } = require('./chunking');
const { getEmbeddings } = require('./embeddings');
const { ToonDBVectorStore } = require('./vectorStore');
const { AzureLLMGenerator, MockLLMGenerator } = require('./generation');
const config = require('./config');

/**
 * Complete RAG System using ToonDB
 */
class ToonDBRAG {
    constructor(options = {}) {
        const {
            dbPath = null,
            chunkingStrategy = 'semantic',
            useMock = false
        } = options;

        // Components
        this.loader = new DocumentLoader();
        this.preprocessor = new TextPreprocessor();
        this.chunker = getChunker(chunkingStrategy, {
            chunkSize: config.rag.chunkSize,
            minChunkSize: Math.floor(config.rag.chunkSize / 4)
        });
        this.embedder = getEmbeddings(true, useMock);
        this.vectorStore = new ToonDBVectorStore(dbPath);
        this.generator = useMock ? new MockLLMGenerator() : new AzureLLMGenerator();

        // Config
        this.topK = config.rag.topK;
        this._ingestedDocs = [];
    }

    /**
     * Ingest documents into the RAG system
     */
    async ingest(documents) {
        const allChunks = [];

        for (const doc of documents) {
            // Preprocess
            doc.content = this.preprocessor.clean(doc.content);
            doc.content = this.preprocessor.removeBoilerplate(
                doc.content,
                doc.metadata?.type || ''
            );

            // Chunk
            const chunks = this.chunker.chunk(doc);
            allChunks.push(...chunks);
            this._ingestedDocs.push(doc.metadata?.filename || doc.id);
        }

        if (allChunks.length === 0) {
            console.log('⚠️ No chunks generated from documents');
            return 0;
        }

        // Embed
        console.log(`🔄 Embedding ${allChunks.length} chunks...`);
        const texts = allChunks.map(chunk => chunk.content);
        const embeddings = await this.embedder.embed(texts);

        // Store
        await this.vectorStore.upsert(allChunks, embeddings);

        console.log(`✅ Ingested ${documents.length} documents (${allChunks.length} chunks)`);
        return allChunks.length;
    }

    /**
     * Ingest a single file
     */
    async ingestFile(filePath) {
        const doc = this.loader.load(filePath);
        return this.ingest([doc]);
    }

    /**
     * Ingest all documents from a directory
     */
    async ingestDirectory(dirPath, extensions = null) {
        const documents = this.loader.loadDirectory(dirPath, extensions);
        return this.ingest(documents);
    }

    /**
     * Query the RAG system
     */
    async query(question, topK = null) {
        topK = topK || this.topK;

        // Retrieve relevant chunks
        const queryEmbedding = await this.embedder.embedQuery(question);
        const results = await this.vectorStore.search(queryEmbedding, topK);

        // Generate response
        const response = await this.generator.generateWithSources(question, results);

        return response;
    }

    /**
     * Search for relevant chunks without generation
     */
    async search(queryText, topK = null) {
        topK = topK || this.topK;
        const queryEmbedding = await this.embedder.embedQuery(queryText);
        return this.vectorStore.search(queryEmbedding, topK);
    }

    /**
     * Get system statistics
     */
    getStats() {
        return {
            totalChunks: this.vectorStore.count(),
            ingestedDocuments: this._ingestedDocs.length,
            documentNames: this._ingestedDocs
        };
    }

    /**
     * Clear all data
     */
    async clear() {
        await this.vectorStore.clear();
        this._ingestedDocs = [];
        console.log('🗑️ Cleared all data');
    }

    /**
     * Close connections
     */
    async close() {
        await this.vectorStore.close();
    }
}

module.exports = { ToonDBRAG };
