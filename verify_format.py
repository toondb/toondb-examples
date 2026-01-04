from toondb_langgraph_examples.memory import ToonDBMemory

def verify_output():
    mem = ToonDBMemory()
    
    # Add a memory to ensure we have results
    mem.add_episode("My name is Sushanth.", source="user_input")
    
    query = "name"
    print(f"Searching for '{query}'...")
    result = mem.search(query)
    print("\n--- Result ---")
    print(result)
    print("--------------\n")
    
    if "memory[" in result and "source,relevance,content" in result:
        print("✅ SUCCESS: Output contains TOON format indicators.")
    else:
        print("❌ FAILURE: Output does not look like TOON format.")

if __name__ == "__main__":
    verify_output()
