/**
 * ToonDB RAG System - LLM Generation using Azure OpenAI
 */
const { AzureOpenAI } = require('openai');
const config = require('./config');

/**
 * RAG Response structure
 */
class RAGResponse {
    constructor(answer, sources, context, confidence) {
        this.answer = answer;
        this.sources = sources;
        this.context = context;
        this.confidence = confidence;
    }
}

/**
 * Prompt templates
 */
const PROMPTS = {
    QA_WITH_CITATIONS: `Answer the question based on the provided context.
Cite your sources using [Source N] notation.
If the context doesn't contain enough information, say so.

Context:
{context}

Question: {question}

Provide a detailed answer with citations:`
};

/**
 * Assemble retrieved chunks into context
 */
class ContextAssembler {
    constructor(maxContextLength = null) {
        this.maxContextLength = maxContextLength || config.rag.maxContextLength;
    }

    assemble(results) {
        const contextParts = [];
        let currentLength = 0;

        for (let i = 0; i < results.length; i++) {
            const result = results[i];
            const source = result.chunk.metadata?.filename || 'Unknown';
            const chunkText = `[Source ${i + 1}: ${source}]\n${result.chunk.content}\n`;

            if (currentLength + chunkText.length > this.maxContextLength) {
                break;
            }

            contextParts.push(chunkText);
            currentLength += chunkText.length;
        }

        return contextParts.join('\n');
    }
}

/**
 * LLM Generator using Azure OpenAI
 */
class AzureLLMGenerator {
    constructor(promptTemplate = null) {
        this.client = new AzureOpenAI({
            apiKey: config.azure.apiKey,
            apiVersion: config.azure.apiVersion,
            endpoint: config.azure.endpoint
        });
        this.deployment = config.azure.chatDeployment;
        this.promptTemplate = promptTemplate || PROMPTS.QA_WITH_CITATIONS;
        this.contextAssembler = new ContextAssembler();
    }

    async generate(question, context, maxTokens = 1000, temperature = 0.1) {
        const prompt = this.promptTemplate
            .replace('{context}', context)
            .replace('{question}', question);

        const response = await this.client.chat.completions.create({
            model: this.deployment,
            max_tokens: maxTokens,
            temperature: temperature,
            messages: [{ role: 'user', content: prompt }]
        });

        return response.choices[0].message.content;
    }

    async generateWithSources(question, results, maxTokens = 1000, temperature = 0.1) {
        // Determine confidence level
        let confidence;
        if (!results || results.length === 0) {
            confidence = 'low';
        } else {
            const topScore = results[0].score;
            if (topScore >= 0.8) {
                confidence = 'high';
            } else if (topScore >= 0.5) {
                confidence = 'medium';
            } else {
                confidence = 'low';
            }
        }

        // Build context
        const context = this.contextAssembler.assemble(results);

        // Handle low confidence
        if (confidence === 'low' && (!results || results[0]?.score < 0.3)) {
            return new RAGResponse(
                "I don't have enough relevant information to answer this question confidently.",
                results,
                context,
                confidence
            );
        }

        // Generate answer
        let answer = await this.generate(question, context, maxTokens, temperature);

        // Add caveat for medium confidence
        if (confidence === 'medium') {
            answer = `Based on the available information: ${answer}`;
        }

        return new RAGResponse(answer, results, context, confidence);
    }
}

/**
 * Mock Generator for testing
 */
class MockLLMGenerator {
    constructor() {
        this.contextAssembler = new ContextAssembler();
    }

    async generateWithSources(question, results) {
        const context = this.contextAssembler.assemble(results);
        const q = question.toLowerCase();

        let answer = 'I am a mock AI. ';

        if (q.includes('install')) {
            answer += 'To install ToonDB, run `npm install @sushanth/toondb` or `pip install toondb-client`.';
        } else if (q.includes('features')) {
            answer += 'ToonDB features include Key-Value Store, Vector Search, and SQL Support.';
        } else if (q.includes('sql')) {
            answer += 'Yes, ToonDB supports SQL operations like CREATE, INSERT, SELECT.';
        } else if (q.includes('toondb')) {
            answer += 'ToonDB is a high-performance embedded database designed for AI applications.';
        } else {
            answer += `I found ${results.length} relevant sources.`;
        }

        return new RAGResponse(
            answer,
            results,
            context,
            results.length > 0 ? 'high' : 'low'
        );
    }
}

module.exports = {
    RAGResponse,
    ContextAssembler,
    AzureLLMGenerator,
    MockLLMGenerator
};
