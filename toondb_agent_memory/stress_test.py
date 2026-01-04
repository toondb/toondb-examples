"""
ToonDB Agent Memory System - Large Scale Stress Test
Tests system performance with 1000+ observations
"""
import time
import argparse
from agent import Agent
from performance_tracker import PerformanceTracker
import uuid


def generate_diverse_messages(count: int):
    """Generate diverse realistic messages for stress testing"""
    
    topics = [
        # Technical questions
        "How do I optimize database queries for large datasets?",
        "What's the difference between SQL and NoSQL databases?",
        "Can you explain how indexing works in databases?",
        "What are the best practices for database schema design?",
        "How do I handle database migrations in production?",
        
        # Research topics
        "Tell me about recent advances in AI and machine learning",
        "What are the environmental impacts of cloud computing?",
        "How does blockchain technology work?",
        "What are the latest developments in quantum computing?",
        "Explain the current state of renewable energy technology",
        
        # Data analysis
        "I have sales data showing a 15% increase quarter over quarter",
        "The dataset contains 50,000 rows with 12 features",
        "Our user engagement metrics show 2.3 million active users",
        "Revenue grew from $1.2M to $1.8M this quarter",
        "Customer churn rate decreased from 5% to 3.2%",
        
        # Problem solving
        "I'm seeing performance issues with my application",
        "The system crashes when handling concurrent requests",
        "Memory usage spikes to 90% during peak hours",
        "API response times increased from 200ms to 2 seconds",
        "Database connection pool is exhausted under load",
        
        # Planning and strategy
        "We need to scale our infrastructure for growth",
        "Should we migrate to microservices architecture?",
        "What's the best cloud provider for our use case?",
        "How do we implement disaster recovery?",
        "What security measures should we prioritize?",
        
        # Follow-up questions
        "Can you elaborate on that point?",
        "What are the trade-offs involved?",
        "How would this work in practice?",
        "Are there any alternatives to consider?",
        "What are the cost implications?",
    ]
    
    messages = []
    for i in range(count):
        # Cycle through topics with variation
        base_msg = topics[i % len(topics)]
        
        # Add variation every 10 messages to create uniqueness
        if i % 10 == 0:
            prefix = f"[Session {i//100}] "
        elif i % 5 == 0:
            prefix = "Follow-up: "
        else:
            prefix = ""
        
        messages.append(prefix + base_msg)
    
    return messages


def run_stress_test(num_turns: int = 500, verbose: bool = False):
    """
    Run stress test with specified number of turns
    
    Args:
        num_turns: Number of conversation turns (default 500 = 1000 observations)
        verbose: Print progress updates
    """
    print("\n" + "="*70)
    print(f"  🔥 ToonDB Large-Scale Stress Test - {num_turns} Turns")
    print("="*70 + "\n")
    
    # Generate messages
    print(f"📝 Generating {num_turns} diverse messages...")
    messages = generate_diverse_messages(num_turns)
    
    # Calculate observations (2 per turn: user + assistant)
    total_observations = num_turns * 2
    print(f"📊 Total observations: {total_observations}")
    print(f"🔑 Session ID: stress_test_{uuid.uuid4().hex[:8]}")
    print(f"⏰ Started at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("\n" + "-"*70 + "\n")
    
    # Create agent
    session_id = f"stress_{uuid.uuid4().hex[:8]}"
    agent = Agent(session_id=session_id, enable_metrics=True)
    
    # Track additional metrics
    start_time = time.time()
    checkpoint_interval = 50  # Report every 50 turns
    
    try:
        for i, user_message in enumerate(messages, 1):
            if verbose or i % checkpoint_interval == 0:
                elapsed = time.time() - start_time
                rate = i / elapsed if elapsed > 0 else 0
                print(f"Turn {i}/{num_turns} | "
                      f"Observations: {i*2} | "
                      f"Elapsed: {elapsed:.1f}s | "
                      f"Rate: {rate:.2f} turns/sec")
            
            # Get agent response
            response = agent.chat(user_message)
            
            # Print occasional sample to show it's working
            if verbose and i % 100 == 0:
                print(f"  Sample Q: {user_message[:60]}...")
                print(f"  Sample A: {response.message[:60]}...")
                print()
    
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        print(f"Completed {i} turns before interruption\n")
    
    finally:
        # Calculate final statistics
        total_time = time.time() - start_time
        actual_observations = agent.turn_counter
        
        print("\n" + "="*70)
        print("📈 STRESS TEST RESULTS")
        print("="*70)
        print(f"\n✅ Completed: {i} turns")
        print(f"📦 Total observations stored: {actual_observations}")
        print(f"⏱️  Total time: {total_time:.2f} seconds")
        print(f"⚡ Average rate: {i/total_time:.2f} turns/second")
        print(f"💾 Estimated database size: ~{actual_observations * 0.05:.1f} MB")
        print()
        
        # Show performance report
        print("="*70)
        print("📊 PERFORMANCE METRICS")
        print("="*70)
        
        report = agent.get_performance_report()
        print(report.summary())
        
        # Cleanup
        agent.close()
        
        print("✅ Stress test complete!")
        print(f"\n💡 ToonDB handled {actual_observations} observations")
        print(f"   Database path: {agent.memory_manager.toondb_config.db_path}")


def main():
    parser = argparse.ArgumentParser(
        description="ToonDB Large-Scale Stress Test"
    )
    
    parser.add_argument(
        "--num-turns",
        type=int,
        default=500,
        help="Number of conversation turns (default: 500 = 1000 observations)"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print progress updates"
    )
    
    args = parser.parse_args()
    
    # Confirm if running large test
    if args.num_turns >= 500:
        print(f"\n⚠️  You are about to run a stress test with {args.num_turns} turns")
        print(f"   This will generate {args.num_turns * 2} observations")
        print(f"   Estimated time: {args.num_turns * 4 / 60:.1f} minutes")
        print(f"   API costs: ~${args.num_turns * 0.01:.2f} (embeddings + completions)")
        
        confirm = input("\n   Continue? [y/N]: ")
        if confirm.lower() != 'y':
            print("Aborted.")
            return
    
    run_stress_test(num_turns=args.num_turns, verbose=args.verbose)


if __name__ == "__main__":
    main()
