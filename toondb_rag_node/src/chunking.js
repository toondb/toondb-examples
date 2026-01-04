/**
 * ToonDB RAG System - Text Chunking Strategies
 */
const { Chunk } = require('./documents');

/**
 * Simple fixed-size chunking with overlap
 */
class FixedSizeChunker {
    constructor(chunkSize = 512, overlap = 50) {
        this.chunkSize = chunkSize;
        this.overlap = overlap;
    }

    chunk(document) {
        const text = document.content;
        const chunks = [];
        let start = 0;

        while (start < text.length) {
            let end = Math.min(start + this.chunkSize, text.length);
            let chunkText = text.slice(start, end);

            // Try to break at word boundary
            if (end < text.length && !/\s/.test(text[end])) {
                const lastSpace = chunkText.lastIndexOf(' ');
                if (lastSpace > this.chunkSize / 2) {
                    end = start + lastSpace;
                    chunkText = text.slice(start, end);
                }
            }

            chunks.push(new Chunk(
                chunkText.trim(),
                {
                    ...document.metadata,
                    chunkIndex: chunks.length,
                    docId: document.id
                },
                start,
                end
            ));

            // Calculate next start position
            let nextStart = end - this.overlap;
            if (nextStart <= start) nextStart = end;
            if (end >= text.length) break;
            start = nextStart;
        }

        return chunks;
    }
}

/**
 * Chunk based on semantic boundaries (paragraphs)
 */
class SemanticChunker {
    constructor(maxChunkSize = 1000, minChunkSize = 100) {
        this.maxChunkSize = maxChunkSize;
        this.minChunkSize = minChunkSize;
    }

    chunk(document) {
        const paragraphs = document.content.split(/\n\n+/);
        const chunks = [];
        let currentChunk = '';
        let currentStart = 0;

        for (const para of paragraphs) {
            const trimmed = para.trim();
            if (!trimmed) continue;

            if (currentChunk.length + trimmed.length + 2 <= this.maxChunkSize) {
                currentChunk += trimmed + '\n\n';
            } else {
                if (currentChunk.length >= this.minChunkSize) {
                    chunks.push(new Chunk(
                        currentChunk.trim(),
                        {
                            ...document.metadata,
                            chunkIndex: chunks.length,
                            docId: document.id
                        },
                        currentStart,
                        currentStart + currentChunk.length
                    ));
                }
                currentStart += currentChunk.length;
                currentChunk = trimmed + '\n\n';
            }
        }

        // Don't forget the last chunk
        if (currentChunk.trim() && currentChunk.trim().length >= this.minChunkSize) {
            chunks.push(new Chunk(
                currentChunk.trim(),
                {
                    ...document.metadata,
                    chunkIndex: chunks.length,
                    docId: document.id
                },
                currentStart,
                currentStart + currentChunk.length
            ));
        }

        return chunks;
    }
}

/**
 * Factory function to get chunker by strategy name
 */
function getChunker(strategy = 'semantic', options = {}) {
    const strategies = {
        fixed: FixedSizeChunker,
        semantic: SemanticChunker
    };

    if (!strategies[strategy]) {
        throw new Error(`Unknown chunking strategy: ${strategy}`);
    }

    return new strategies[strategy](options.chunkSize, options.overlap || options.minChunkSize);
}

module.exports = {
    FixedSizeChunker,
    SemanticChunker,
    getChunker
};
