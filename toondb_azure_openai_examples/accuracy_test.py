import asyncio
from .memory import ToonDBMemory
from .runner import ingest_politics_data

async def run_accuracy_test():
    print("=== ToonDB Azure OpenAI Accuracy Test ===")
    memory = ToonDBMemory()
    
    # Ingest data
    await ingest_politics_data(memory)
    
    test_cases = [
        ("Who was the California Attorney General?", "Kamala Harris"),
        ("Gavin Newsom previous role", "Lieutenant Governor"),
        ("Kamala Harris term dates", "January 3, 2011"),
        ("Where was Gavin Newsom before", "San Francisco"),
    ]
    
    hits = 0
    print("\nStarting Evaluation...")
    print("-" * 50)
    
    for query, expected in test_cases:
        print(f"Query: '{query}'")
        print(f"Expecting: '{expected}'")
        
        result = memory.search(query, top_k=3)
        
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
