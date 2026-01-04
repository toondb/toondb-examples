import asyncio
import os
import sys
from pathlib import Path
from datetime import datetime, timezone

from .memory import ToonDBMemory
from .config import get_toondb_config

def get_wizard_of_oz_text() -> str:
    # Adjust path to where the text file is located
    path = Path("donotuse/wizard_of_oz/woo.txt")
    if not path.exists():
        print(f"File not found: {path}")
        return ""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def chunk_text(text: str, chunk_size: int = 1000) -> list[str]:
    """Simple chunking by paragraphs roughly fitting chunk_size"""
    paragraphs = text.split("\n\n")
    chunks = []
    current_chunk = ""
    
    for p in paragraphs:
        if len(current_chunk) + len(p) < chunk_size:
            current_chunk += p + "\n\n"
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = p + "\n\n"
            
    if current_chunk:
        chunks.append(current_chunk.strip())
        
    return chunks

async def ingest_book(memory: ToonDBMemory):
    print("📦 Ingesting Wizard of Oz...")
    text = get_wizard_of_oz_text()
    if not text:
        return

    # For demo speed, let's take the first 50 chunks
    # akin to the original example taking a subset
    chunks = chunk_text(text)[:50]
    
    print(f"Ingesting {len(chunks)} chunks...")
    for i, chunk in enumerate(chunks):
        if not chunk.strip():
            continue
            
        memory.add_episode(
            content=chunk,
            source=f"Wizard of Oz - Chunk {i}",
            timestamp=datetime.now(timezone.utc).timestamp()
        )
        if (i+1) % 10 == 0:
            print(f"  Processed {i+1} chunks...")
            
    print("✅ Ingestion complete.")

async def demonstrate_search(memory: ToonDBMemory):
    print("\n🔍 Demonstrating Search...")
    queries = [
        "What color are the shoes?",
        "Who is the great Wizard?",
        "What did the scarecrow want?"
    ]
    
    for q in queries:
        print(f"\nQuery: '{q}'")
        result = memory.search(q, top_k=3)
        print(f"--- TOON Context ---\n{result}\n--------------------")

async def main():
    print("=== ToonDB Wizard of Oz Example ===")
    config = get_toondb_config()
    print(f"Database: {config.db_path}")
    
    memory = ToonDBMemory()
    
    try:
        await ingest_book(memory)
        await demonstrate_search(memory)
    finally:
        memory.close()

if __name__ == "__main__":
    asyncio.run(main())
