import asyncio
import json
import sys
from datetime import datetime, timezone

from .memory import ToonDBMemory
from .config import get_toondb_config

async def ingest_politics_data(memory: ToonDBMemory):
    print("📦 Ingesting California Politics data...")
    
    episodes = [
        {
            'content': 'Kamala Harris is the Attorney General of California. She was previously the district attorney for San Francisco.',
            'type': 'text',
            'description': 'podcast transcript',
        },
        {
            'content': 'As AG, Harris was in office from January 3, 2011 – January 3, 2017',
            'type': 'text',
            'description': 'podcast transcript',
        },
        {
            'content': json.dumps({
                'name': 'Gavin Newsom',
                'position': 'Governor',
                'state': 'California',
                'previous_role': 'Lieutenant Governor',
                'previous_location': 'San Francisco',
            }),
            'type': 'json',
            'description': 'podcast metadata',
        },
    ]

    for i, episode in enumerate(episodes):
        memory.add_episode(
            content=episode['content'],
            source=episode['description'],
            timestamp=datetime.now(timezone.utc).timestamp()
        )
        print(f"  Added episode {i}: {episode['description']}")
            
    print("✅ Ingestion complete.")

async def demonstrate_search(memory: ToonDBMemory):
    print("\n🔍 Demonstrating Search...")
    query = 'Who was the California Attorney General?'
    print(f"Query: '{query}'")
    
    result = memory.search(query, top_k=3)
    print(f"--- TOON Context ---\n{result}\n--------------------")

async def main():
    print("=== ToonDB Azure OpenAI Example ===")
    config = get_toondb_config()
    print(f"Database: {config.db_path}")
    
    memory = ToonDBMemory()
    
    try:
        await ingest_politics_data(memory)
        await demonstrate_search(memory)
    finally:
        memory.close()

if __name__ == "__main__":
    asyncio.run(main())
