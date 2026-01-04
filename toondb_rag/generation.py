"""
ToonDB RAG System - LLM Generation using Azure OpenAI
"""
from typing import List, Dict, Any
from dataclasses import dataclass
from openai import AzureOpenAI

from config import get_azure_config, get_rag_config
from vector_store import SearchResult


@dataclass
class RAGResponse:
    """Response from RAG system"""
    answer: str
    sources: List[SearchResult]
    context: str
    confidence: str  # "high", "medium", "low"


class ContextAssembler:
    """Assemble retrieved chunks into a coherent context"""
    
    def __init__(self, max_context_length: int = None):
        config = get_rag_config()
        self.max_context_length = max_context_length or config.max_context_length
    
    def assemble(self, results: List[SearchResult]) -> str:
        """Assemble context from search results"""
        context_parts = []
        current_length = 0
        
        for i, result in enumerate(results):
            source = result.chunk.metadata.get('filename', 'Unknown')
            chunk_text = f"[Source {i+1}: {source}]\n{result.chunk.content}\n"
            
            if current_length + len(chunk_text) > self.max_context_length:
                break
            
            context_parts.append(chunk_text)
            current_length += len(chunk_text)
        
        return "\n".join(context_parts)


class RAGPrompts:
    """Prompt templates for RAG generation"""
    
    BASIC_QA = """Answer the question based on the provided context. 
If the context doesn't contain enough information to answer, say so.

Context:
{context}

Question: {question}

Answer:"""

    QA_WITH_CITATIONS = """Answer the question based on the provided context.
Cite your sources using [Source N] notation.
If the context doesn't contain enough information, say so.

Context:
{context}

Question: {question}

Provide a detailed answer with citations:"""

    CONVERSATIONAL = """You are a helpful assistant with access to a knowledge base.
Use the following context to answer the user's question.
If you're unsure or the context doesn't help, be honest about it.

Context from knowledge base:
{context}

User: {question}
Assistant:"""


class AzureLLMGenerator:
    """LLM Generator using Azure OpenAI"""
    
    def __init__(self, prompt_template: str = None):
        config = get_azure_config()
        self.client = AzureOpenAI(
            api_key=config.api_key,
            api_version=config.api_version,
            azure_endpoint=config.endpoint
        )
        self.deployment = config.chat_deployment
        self.prompt_template = prompt_template or RAGPrompts.QA_WITH_CITATIONS
        self.context_assembler = ContextAssembler()
    
    def generate(
        self,
        question: str,
        context: str,
        max_tokens: int = 1000,
        temperature: float = 0.1
    ) -> str:
        """Generate answer from question and context"""
        prompt = self.prompt_template.format(
            context=context,
            question=question
        )
        
        response = self.client.chat.completions.create(
            model=self.deployment,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}]
        )
        
        return response.choices[0].message.content
    
    def generate_with_sources(
        self,
        question: str,
        results: List[SearchResult],
        max_tokens: int = 1000,
        temperature: float = 0.1
    ) -> RAGResponse:
        """Generate answer with structured source tracking"""
        # Determine confidence level
        if not results:
            confidence = "low"
        else:
            top_score = results[0].score
            if top_score >= 0.8:
                confidence = "high"
            elif top_score >= 0.5:
                confidence = "medium"
            else:
                confidence = "low"
        
        # Build context
        context = self.context_assembler.assemble(results)
        
        # Handle low confidence
        if confidence == "low" and (not results or results[0].score < 0.3):
            return RAGResponse(
                answer="I don't have enough relevant information to answer this question confidently. Please try rephrasing your question or asking about a different topic.",
                sources=results,
                context=context,
                confidence=confidence
            )
        
        # Generate answer
        answer = self.generate(question, context)
        
        # Add caveat for medium confidence
        if confidence == "medium":
            answer = f"Based on the available information: {answer}"
        
        return RAGResponse(
            answer=answer,
            sources=results,
            context=context,
            confidence=confidence
        )


class MockLLMGenerator:
    """Mock Generator for testing/demo"""
    
    def __init__(self):
        self.context_assembler = ContextAssembler()
        
    def generate_with_sources(
        self,
        question: str,
        results: List[SearchResult],
        max_tokens: int = 1000,
        temperature: float = 0.1
    ) -> RAGResponse:
        """Generate mock answer"""
        context = self.context_assembler.assemble(results)
        
        # Simple keyword matching for demo
        answer = "I am a mock AI. "
        
        if "install" in question.lower():
            answer += "To install ToonDB, run `pip install toondb-client` or `npm install @sushanth/toondb`."
        elif "features" in question.lower():
            answer += "ToonDB features include Key-Value Store, Vector Search, and SQL Support."
        elif "sql" in question.lower():
            answer += "Yes, ToonDB supports SQL operations like CREATE, INSERT, SELECT."
        elif "hnsw" in question.lower():
            answer += "HNSW is used for fast approximate nearest neighbor search in ToonDB."
        elif "toondb" in question.lower():
            answer += "ToonDB is a high-performance embedded database designed for AI applications."
        else:
            answer += f"I found {len(results)} relevant sources but cannot generate a specific answer in mock mode."
            
        return RAGResponse(
            answer=answer,
            sources=results,
            context=context,
            confidence="high" if results else "low"
        )

