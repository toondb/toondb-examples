"""
Context Query Builder Demo with ToonDB Integration

Demonstrates priority-based context assembly under token budgets.
"""
import asyncio
from .context_builder import ContextQueryBuilder
from toondb import Database


def demo_basic_assembly():
    """Demo 1: Basic context assembly with ample budget"""
    print("=" * 70)
    print("Demo 1: Context Assembly with Ample Token Budget (4000 tokens)")
    print("=" * 70)
    
    builder = ContextQueryBuilder(model="gpt-4", token_budget=4000)
    
    system_message = "You are a helpful AI assistant with knowledge about Python programming."
    
    user_query = "How do I read a CSV file in Python?"
    
    conversation_history = [
        ("User", "What's the best way to handle dates in Python?"),
        ("Assistant", "The datetime module is great for basic date handling. For more advanced features, consider using the pandas library."),
        ("User", "Can you show me an example with pandas?"),
        ("Assistant", "Sure! Here's an example: import pandas as pd; df = pd.DataFrame({'date': ['2024-01-01', '2024-01-02']})"),
    ]
    
    retrieved_context = [
        "Python CSV Module: The csv module implements classes to read and write tabular data in CSV format.",
        "Pandas read_csv(): pandas.read_csv() is a function to read a comma-separated values (csv) file into DataFrame.",
        "Common use: df = pd.read_csv('file.csv')"
    ]
    
    context, stats = builder.build_context(
        system_message=system_message,
        user_query=user_query,
        conversation_history=conversation_history,
        retrieved_context=retrieved_context
    )
    
    print(f"\n📊 Stats:")
    print(f"  Tokens Used: {stats['total_tokens']}/{stats['budget']}")
    print(f"  Utilization: {stats['utilization']:.1f}%")
    print(f"  Components: {stats['components_included']}/{stats['components_total']}")
    print(f"  Latency: {stats['latency_ms']:.2f}ms")
    
    print(f"\n📄 Assembled Context:")
    print("-" * 70)
    print(context)
    print("-" * 70)


def demo_tight_budget():
    """Demo 2: Context assembly with tight budget showing truncation"""
    print("\n\n" + "=" * 70)
    print("Demo 2: Context Assembly with Tight Budget (500 tokens)")
    print("=" * 70)
    
    builder = ContextQueryBuilder(model="gpt-4", token_budget=500)
    
    system_message = "You are a helpful AI assistant."
    user_query = "What is machine learning?"
    
    # Long conversation history
    conversation_history = [
        ("User", "Tell me about artificial intelligence"),
        ("Assistant", "AI is a broad field focused on creating intelligent machines that can perform tasks requiring human-like intelligence."),
        ("User", "What are the main branches?"),
        ("Assistant", "The main branches include machine learning, natural language processing, computer vision, and robotics."),
        ("User", "Which is most important?"),
        ("Assistant", "Machine learning is foundational as it enables systems to learn from data."),
        ("User", "How does it work?"),
        ("Assistant", "ML algorithms learn patterns from training data and make predictions on new data."),
    ]
    
    retrieved_context = [
        "Machine Learning Definition: A subset of AI that enables systems to learn and improve from experience without being explicitly programmed.",
        "Types: Supervised learning uses labeled data, unsupervised learning finds patterns, reinforcement learning learns through rewards.",
        "Common algorithms: Linear regression, decision trees, neural networks, support vector machines.",
        "Applications: Image recognition, spam filtering, recommendation systems, autonomous vehicles.",
    ]
    
    context, stats = builder.build_context(
        system_message=system_message,
        user_query=user_query,
        conversation_history=conversation_history,
        retrieved_context=retrieved_context
    )
    
    print(f"\n📊 Stats:")
    print(f"  Tokens Used: {stats['total_tokens']}/{stats['budget']}")
    print(f"  Utilization: {stats['utilization']:.1f}%")
    print(f"  Components: {stats['components_included']}/{stats['components_total']} (truncated!)")
    print(f"  Latency: {stats['latency_ms']:.2f}ms")
    
    print(f"\n📄 Assembled Context (Notice Priority-Based Selection):")
    print("-" * 70)
    print(context)
    print("-" * 70)
    
    print(f"\n💡 Note: With tight budget, only highest priority content was included:")
    print(f"   - System message (priority 0) ✓")
    print(f"   - User query (priority 0) ✓")
    print(f"   - Recent history (priority 1) ✓")
    print(f"   - Top retrieval results (priority 2) ✓")
    print(f"   - Older history/retrieval (priority 3) ✗ (truncated)")


def demo_with_toondb():
    """Demo 3: ToonDB integration for retrieval"""
    print("\n\n" + "=" * 70)
    print("Demo 3: Context Builder with ToonDB Retrieval")
    print("=" * 70)
    
    # Store some facts in ToonDB
    db = Database.open("./demo_context_db")
    
    facts = [
        {"topic": "Python CSV", "content": "Python's csv module provides functionality to read and write CSV files."},
        {"topic": "Pandas", "content": "pandas.read_csv() is the most popular way to load CSV data into a DataFrame."},
        {"topic": "NumPy", "content": "NumPy provides genfromtxt() for reading CSV files into arrays."},
    ]
    
    # Store facts
    for idx, fact in enumerate(facts):
        db.put(f"fact:{idx}".encode(), str(fact).encode())
    
    # Retrieve (in real scenario, this would use vector search)
    retrieved = []
    for key, value in db.scan_prefix(b"fact:"):
        retrieved.append(value.decode())
    
    # Use TOON format for compact representation
    toon_context = Database.to_toon("facts", facts, ["topic", "content"])
    
    builder = ContextQueryBuilder(model="gpt-4", token_budget=1000)
    
    context, stats = builder.build_context(
        system_message="You are a Python expert.",
        user_query="How do I load CSV data?",
        retrieved_context=[toon_context]  # Use TOON-formatted retrieval
    )
    
    print(f"\n📊 Stats:")
    print(f"  Tokens Used: {stats['total_tokens']}/{stats['budget']}")
    print(f"  Utilization: {stats['utilization']:.1f}%")
    
    print(f"\n📄 Context with ToonDB TOON Format:")
    print("-" * 70)
    print(context)
    print("-" * 70)
    
    db.close()
    print("\n✓ ToonDB integration complete!")


async def main():
    """Run all demos"""
    print("\n🚀 ToonDB Context Query Builder Examples\n")
    
    demo_basic_assembly()
    demo_tight_budget()
    demo_with_toondb()
    
    print("\n\n" + "=" * 70)
    print("✅ All demos complete!")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
