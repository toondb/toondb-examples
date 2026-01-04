/**
 * ToonDB RAG System - Embeddings using Azure OpenAI
 */
const { AzureOpenAI } = require('openai');
const config = require('./config');

/**
 * Azure OpenAI Embeddings
 */
class AzureEmbeddings {
    constructor() {
        this.client = new AzureOpenAI({
            apiKey: config.azure.apiKey,
            apiVersion: config.azure.apiVersion,
            endpoint: config.azure.endpoint
        });
        this.deployment = config.azure.embeddingDeployment;
        this._dimension = 1536;
    }

    /**
     * Embed a list of texts
     */
    async embed(texts) {
        if (!texts || texts.length === 0) {
            return [];
        }

        // Handle single text
        if (typeof texts === 'string') {
            texts = [texts];
        }

        // Azure OpenAI has a batch size limit
        const batchSize = 16;
        const allEmbeddings = [];

        for (let i = 0; i < texts.length; i += batchSize) {
            const batch = texts.slice(i, i + batchSize);
            const response = await this.client.embeddings.create({
                input: batch,
                model: this.deployment
            });

            const batchEmbeddings = response.data.map(e => e.embedding);
            allEmbeddings.push(...batchEmbeddings);
        }

        return allEmbeddings;
    }

    /**
     * Embed a single query
     */
    async embedQuery(query) {
        const embeddings = await this.embed([query]);
        return embeddings[0];
    }

    get dimension() {
        return this._dimension;
    }
}

/**
 * Mock embeddings for testing
 */
class MockEmbeddings {
    constructor(dimension = 1536) {
        this._dimension = dimension;
    }

    async embed(texts) {
        if (typeof texts === 'string') {
            texts = [texts];
        }

        return texts.map(text => {
            // Deterministic random based on text length
            const embeddings = new Array(this._dimension);
            for (let i = 0; i < this._dimension; i++) {
                embeddings[i] = Math.sin(text.length + i) * 0.5 + 0.5;
            }
            return embeddings;
        });
    }

    async embedQuery(query) {
        const embeddings = await this.embed([query]);
        return embeddings[0];
    }

    get dimension() {
        return this._dimension;
    }
}

function getEmbeddings(useAzure = true, useMock = false) {
    if (useMock) {
        return new MockEmbeddings();
    }
    return new AzureEmbeddings();
}

module.exports = {
    AzureEmbeddings,
    MockEmbeddings,
    getEmbeddings
};
