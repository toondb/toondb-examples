import asyncio
import sys
from pathlib import Path
from datetime import datetime, timezone
import random
from typing import List, Tuple

from .memory import ToonDBMemory
from .config import get_toondb_config
from .runner import ingest_products_data

async def run_accuracy_test():
    print("=== ToonDB eCommerce Accuracy Test ===")
    memory = ToonDBMemory()
    
    # Ensure data is loaded (simulated check)
    # in a real persistent DB we might not need to re-ingest every time if persisted
    # but for this test let's re-ingest to be safe
    await ingest_products_data(memory)
    
    # Populate some conversation history for context mixing tests
    history = [
        ("User", "I am allergic to wool."),
        ("Assistant", "Noted, I will avoid showing you wool products."),
        ("User", "I prefer black shoes."),
        ("Assistant", "We have some great black options.")
    ]
    for role, text in history:
        memory.add_episode(f"{role}: {text}", source="conversation_history")

    test_cases = [
        # Query, Expected Keyword/content in top results
        ("wool shoes", "Wool Runners"),
        ("something for kids", "Little Kids"),
        ("socks", "Sock"), 
        ("black shoes", "Natural Black"),
        ("my allergy", "allergic to wool"), # Should retrieve conversation history
        ("price of size 10", "Price Range"), # generic product price check
    ]
    
    hits = 0
    print("\nStarting Evaluation...")
    print("-" * 50)
    
    for query, expected in test_cases:
        print(f"Query: '{query}'")
        print(f"Expecting: '{expected}'")
        
        # Search top 3
        result = memory.search(query, top_k=3)
        
        # Check if expected string is in the TOON formatted result
        if expected.lower() in result.lower():
            print("✅ PASS")
            hits += 1
        else:
            print("❌ FAIL")
            print(f"Retrieved Context:\n{result}\n")
            
        print("-" * 50)
        
    accuracy = (hits / len(test_cases)) * 100
    print(f"\nOverall Accuracy: {accuracy:.1f}% ({hits}/{len(test_cases)})")
    
    memory.close()

if __name__ == "__main__":
    asyncio.run(run_accuracy_test())
