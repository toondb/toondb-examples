/**
 * ToonDB RAG System - Document Models and Loader
 */
const fs = require('fs');
const path = require('path');
const crypto = require('crypto');

/**
 * Represents a loaded document
 */
class Document {
    constructor(content, metadata = {}) {
        this.content = content;
        this.metadata = metadata;
        this.id = this._generateId();
    }

    _generateId() {
        return crypto.createHash('md5').update(this.content).digest('hex');
    }
}

/**
 * Represents a chunk of a document
 */
class Chunk {
    constructor(content, metadata, startIndex, endIndex) {
        this.content = content;
        this.metadata = metadata;
        this.startIndex = startIndex;
        this.endIndex = endIndex;
        this.id = crypto.createHash('md5')
            .update(`${content}${startIndex}`)
            .digest('hex');
    }
}

/**
 * Load documents from various sources
 */
class DocumentLoader {
    /**
     * Load a markdown document
     */
    loadMarkdown(filePath) {
        const content = fs.readFileSync(filePath, 'utf-8');
        return new Document(content, {
            source: filePath,
            filename: path.basename(filePath),
            type: 'markdown'
        });
    }

    /**
     * Load a plain text document
     */
    loadText(filePath) {
        const content = fs.readFileSync(filePath, 'utf-8');
        return new Document(content, {
            source: filePath,
            filename: path.basename(filePath),
            type: 'text'
        });
    }

    /**
     * Auto-detect and load document
     */
    load(filePath) {
        const ext = path.extname(filePath).toLowerCase();

        if (ext === '.md' || ext === '.markdown') {
            return this.loadMarkdown(filePath);
        } else {
            return this.loadText(filePath);
        }
    }

    /**
     * Load all documents from a directory
     */
    loadDirectory(dirPath, extensions = ['.md', '.txt']) {
        const documents = [];
        const files = fs.readdirSync(dirPath);

        for (const file of files) {
            const ext = path.extname(file).toLowerCase();
            if (extensions.includes(ext)) {
                try {
                    const doc = this.load(path.join(dirPath, file));
                    documents.push(doc);
                    console.log(`✅ Loaded: ${file}`);
                } catch (e) {
                    console.log(`❌ Failed to load ${file}: ${e.message}`);
                }
            }
        }

        return documents;
    }
}

/**
 * Clean and normalize text before chunking
 */
class TextPreprocessor {
    clean(text) {
        // Remove excessive whitespace
        text = text.replace(/\s+/g, ' ');
        // Remove special characters
        text = text.replace(/[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]/g, '');
        return text.trim();
    }

    removeBoilerplate(text, docType) {
        if (docType === 'pdf') {
            // Remove page numbers
            text = text.replace(/\n\s*\d+\s*\n/g, '\n');
            text = text.replace(/Page \d+ of \d+/g, '');
        }
        return text;
    }
}

module.exports = {
    Document,
    Chunk,
    DocumentLoader,
    TextPreprocessor
};
