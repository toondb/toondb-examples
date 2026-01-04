import asyncio
from .memory import ToonDBMemory
from .runner import ingest_book

async def run_accuracy_test():
    print("=== ToonDB Wizard of Oz Accuracy Test ===")
    memory = ToonDBMemory()
    
    await ingest_book(memory)
    
    test_cases = [
        # Original text has silver shoes, not ruby (movie vers.)
        ("What color were the shoes Dorothy took?", "silver"), 
        ("What did the Scarecrow want?", "brains"),
        ("What happened to the house?", "cyclone"),
    ]
    
    hits = 0
    print("\nStarting Evaluation...")
    print("-" * 50)
    
    for query, expected in test_cases:
        print(f"Query: '{query}'")
        print(f"Expecting: '{expected}'")
        
        result = memory.search(query, top_k=5)
        
        if expected.lower() in result.lower():
            print("✅ PASS")
            hits += 1
        else:
            # Fallback for "brains" vs "brain" etc
            if expected == "brains" and "brain" in result.lower():
                 print("✅ PASS (fuzzy match)")
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
