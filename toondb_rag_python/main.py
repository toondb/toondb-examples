#!/usr/bin/env python3
"""
ToonDB RAG System - Main Entry Point

Usage:
    python main.py ingest ./documents     # Ingest documents from a directory
    python main.py ingest file.pdf        # Ingest a single file
    python main.py query "Your question"  # Query the system
    python main.py interactive            # Interactive mode
    python main.py stats                  # Show statistics
    python main.py clear                  # Clear all data
"""
import sys
import argparse
from pathlib import Path

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
load_dotenv()

from rag import ToonDBRAG


def ingest_command(rag: ToonDBRAG, path: str):
    """Ingest documents from path"""
    p = Path(path)
    
    if p.is_file():
        print(f"📄 Ingesting file: {path}")
        count = rag.ingest_file(path)
    elif p.is_dir():
        print(f"📁 Ingesting directory: {path}")
        count = rag.ingest_directory(path)
    else:
        print(f"❌ Path not found: {path}")
        return
    
    print(f"✅ Ingested {count} chunks")


def query_command(rag: ToonDBRAG, question: str):
    """Query the RAG system"""
    print(f"\n🔍 Query: {question}\n")
    
    response = rag.query(question)
    
    print(f"📊 Confidence: {response.confidence}")
    print(f"\n💬 Answer:\n{response.answer}\n")
    
    if response.sources:
        print("📚 Sources:")
        for i, source in enumerate(response.sources):
            filename = source.chunk.metadata.get('filename', 'Unknown')
            score = source.score
            print(f"  [{i+1}] {filename} (score: {score:.3f})")
            print(f"      {source.chunk.content[:100]}...")


def interactive_mode(rag: ToonDBRAG):
    """Interactive query mode"""
    print("\n🤖 ToonDB RAG Interactive Mode")
    print("   Type 'quit' or 'exit' to exit")
    print("   Type 'stats' to see statistics")
    print("-" * 50)
    
    while True:
        try:
            question = input("\n❓ You: ").strip()
            
            if not question:
                continue
            
            if question.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            
            if question.lower() == 'stats':
                stats = rag.get_stats()
                print(f"📊 Total chunks: {stats['total_chunks']}")
                print(f"📄 Documents: {stats['ingested_documents']}")
                continue
            
            response = rag.query(question)
            
            print(f"\n🤖 Assistant ({response.confidence} confidence):")
            print(response.answer)
            
            if response.sources:
                print(f"\n📚 ({len(response.sources)} sources used)")
        
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")


def stats_command(rag: ToonDBRAG):
    """Show statistics"""
    stats = rag.get_stats()
    
    print("\n📊 ToonDB RAG Statistics")
    print("-" * 30)
    print(f"Total chunks:     {stats['total_chunks']}")
    print(f"Documents:        {stats['ingested_documents']}")
    
    if stats['document_names']:
        print("\nIngested files:")
        for name in stats['document_names']:
            print(f"  - {name}")


def clear_command(rag: ToonDBRAG):
    """Clear all data"""
    confirm = input("⚠️  Are you sure you want to clear all data? (yes/no): ")
    if confirm.lower() == 'yes':
        rag.clear()
        print("✅ All data cleared")
    else:
        print("❌ Cancelled")


def main():
    parser = argparse.ArgumentParser(
        description="ToonDB RAG System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Ingest command
    ingest_parser = subparsers.add_parser('ingest', help='Ingest documents')
    ingest_parser.add_argument('path', help='Path to file or directory')
    
    # Query command
    query_parser = subparsers.add_parser('query', help='Query the system')
    query_parser.add_argument('question', help='Question to ask')
    
    # Interactive command
    subparsers.add_parser('interactive', help='Interactive mode')
    
    # Stats command
    subparsers.add_parser('stats', help='Show statistics')
    
    # Clear command
    subparsers.add_parser('clear', help='Clear all data')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Create RAG system
    with ToonDBRAG() as rag:
        if args.command == 'ingest':
            ingest_command(rag, args.path)
        elif args.command == 'query':
            query_command(rag, args.question)
        elif args.command == 'interactive':
            interactive_mode(rag)
        elif args.command == 'stats':
            stats_command(rag)
        elif args.command == 'clear':
            clear_command(rag)


if __name__ == "__main__":
    main()
