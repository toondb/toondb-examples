import asyncio
import os
import sys
import re
from pathlib import Path
from datetime import datetime, timezone

from .memory import ToonDBMemory
from .config import get_toondb_config

def get_transcript_text() -> str:
    path = Path("donotuse/podcast/podcast_transcript.txt")
    if not path.exists():
        print(f"File not found: {path}")
        return ""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def parse_transcript(text: str) -> list[dict]:
    """
    Simulated parser for the transcript format:
    Speaker Name (Role): Content...
    """
    # Simply split by lines for now, or regex match speaker patterns
    # Real parser would be more robust, but illustrative for this example
    lines = text.split('\n')
    parsed = []
    
    current_speaker = "Unknown"
    current_text = ""
    
    # Regex for speaker line: "0 (3s):" or "1 (1m 3s):"
    # Matches start of line, some speaker ID, parens with anything inside, colon
    speaker_pattern = re.compile(r'^(.+?)\s*\(([^)]+)\):$')
    
    for line in lines:
        line = line.strip()
        if not line: 
            continue
            
        match = speaker_pattern.match(line)
        if match:
            # If we had a previous speaker, save their text
            if current_speaker != "Unknown":
                parsed.append({
                    "speaker": current_speaker,
                    "content": current_text.strip()
                })
            
            # Start new speaker episode
            current_speaker = match.group(1).strip()
            # Timestamp is match.group(2) if we wanted it
            current_text = ""
        else:
            # It's content
            current_text += " " + line
            
    # Add last segment
    if current_speaker != "Unknown" and current_text:
        parsed.append({
            "speaker": current_speaker,
            "content": current_text.strip()
        })
        
    return parsed

async def ingest_podcast(memory: ToonDBMemory):
    print("📦 Ingesting Podcast Transcript...")
    text = get_transcript_text()
    if not text:
        return

    segments = parse_transcript(text)
    print(f"Found {len(segments)} segments.")
    
    # Ingest first 50 segments for speed
    for i, seg in enumerate(segments[:50]):
        content = f"{seg['speaker']}: {seg['content']}"
        memory.add_episode(
            content=content,
            source="Podcast Transcript",
            timestamp=datetime.now(timezone.utc).timestamp()
        )
        if (i+1) % 10 == 0:
            print(f"  Processed {i+1} segments...")
            
    print("✅ Ingestion complete.")

async def demonstrate_search(memory: ToonDBMemory):
    print("\n🔍 Demonstrating Search...")
    queries = [
        "What is the podcast about?",
        "Who is the host?",
        "Mention of AI or technology"
    ]
    
    for q in queries:
        print(f"\nQuery: '{q}'")
        result = memory.search(q, top_k=3)
        print(f"--- TOON Context ---\n{result}\n--------------------")

async def main():
    print("=== ToonDB Podcast Example ===")
    config = get_toondb_config()
    print(f"Database: {config.db_path}")
    
    memory = ToonDBMemory()
    
    try:
        await ingest_podcast(memory)
        await demonstrate_search(memory)
    finally:
        memory.close()

if __name__ == "__main__":
    asyncio.run(main())
