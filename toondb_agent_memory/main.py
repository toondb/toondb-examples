"""
ToonDB Agent Memory System - Main Entry Point
"""
import sys
import argparse
import uuid
from datetime import datetime

from agent import Agent
from scenarios.customer_support import get_scenario


def print_header():
    """Print welcome header"""
    print("\n" + "="*70)
    print("  🤖 ToonDB Agent Memory System - Live Demonstration")
    print("="*70 + "\n")


def run_scenario(scenario_name: str, num_turns: int = None, verbose: bool = False):
    """
    Run a predefined conversation scenario
    
    Args:
        scenario_name: Name of scenario to run
        num_turns: Number of turns to execute (None = all)
        verbose: Show detailed metrics per turn
    """
    print_header()
    print(f"📋 Loading scenario: {scenario_name}")
    
    # Load scenario
    scenario = get_scenario(scenario_name)
    messages = scenario.get_messages()
    
    if num_turns:
        messages = messages[:num_turns]
    
    print(f"📊 Turns to execute: {len(messages)}")
    print(f"🔑 Session ID: {uuid.uuid4().hex[:12]}")
    print(f"⏰ Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\n" + "-"*70 + "\n")
    
    # Create agent
    session_id = f"session_{uuid.uuid4().hex[:12]}"
    agent = Agent(session_id=session_id, enable_metrics=True)
    
    # Execute conversation
    for i, user_message in enumerate(messages, 1):
        print(f"\n{'='*70}")
        print(f"Turn {i}/{len(messages)}")
        print(f"{'='*70}\n")
        
        print(f"👤 USER:\n{user_message}\n")
        
        # Get agent response
        response = agent.chat(user_message)
        
        print(f"🤖 AGENT:\n{response.message}\n")
        
        if verbose:
            print(f"📊 METRICS:")
            print(f"  - Memories used: {response.memories_count}")
            print(f"  - Write latency: {response.write_latency_ms:.2f} ms")
            print(f"  - Read latency: {response.read_latency_ms:.2f} ms")
            print(f"  - Assemble latency: {response.assemble_latency_ms:.2f} ms")
            print(f"  - LLM latency: {response.llm_latency_ms:.2f} ms")
            print(f"  - Total latency: {response.total_latency_ms:.2f} ms")
            
            if response.memories_count > 0:
                print(f"\n📝 CONTEXT USED:")
                print(response.context_used)
        
        print()
    
    # Show performance report
    print("\n" + "="*70)
    print("📈 FINAL PERFORMANCE REPORT")
    print("="*70)
    
    report = agent.get_performance_report()
    print(report.summary())
    
    # Cleanup
    agent.close()
    
    print("✅ Scenario complete!")


def run_interactive(session_id: str = None):
    """
    Run interactive chat mode
    
    Args:
        session_id: Optional session ID (new one created if not provided)
    """
    print_header()
    print("💬 Interactive Mode")
    print("Type 'exit' or 'quit' to end the conversation")
    print("Type 'report' to see performance metrics\n")
    
    # Create agent
    if not session_id:
        session_id = f"session_{uuid.uuid4().hex[:12]}"
    
    print(f"🔑 Session ID: {session_id}")
    print(f"⏰ Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\n" + "-"*70 + "\n")
    
    agent = Agent(session_id=session_id, enable_metrics=True)
    
    turn = 0
    try:
        while True:
            # Get user input
            user_message = input("👤 YOU: ").strip()
            
            if not user_message:
                continue
            
            if user_message.lower() in ['exit', 'quit']:
                print("\n👋 Goodbye!")
                break
            
            if user_message.lower() == 'report':
                report = agent.get_performance_report()
                print("\n" + report.summary())
                continue
            
            turn += 1
            
            # Get response
            print()
            response = agent.chat(user_message)
            print(f"🤖 AGENT: {response.message}\n")
            print(f"⚡ Latency: {response.total_latency_ms:.0f}ms | Memories: {response.memories_count}\n")
    
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
    
    finally:
        # Show final report
        if turn > 0:
            print("\n" + "="*70)
            print("📈 SESSION PERFORMANCE REPORT")
            print("="*70)
            
            report = agent.get_performance_report()
            print(report.summary())
        
        agent.close()


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="ToonDB Agent Memory System - Live Agent Demonstration"
    )
    
    parser.add_argument(
        "--mode",
        choices=["scenario", "interactive"],
        default="scenario",
        help="Execution mode (default: scenario)"
    )
    
    parser.add_argument(
        "--scenario",
        default="customer_support",
        help="Scenario name to run (default: customer_support)"
    )
    
    parser.add_argument(
        "--num-turns",
        type=int,
        help="Number of turns to execute (default: all)"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed metrics per turn"
    )
    
    parser.add_argument(
        "--session-id",
        help="Session ID for interactive mode"
    )
    
    args = parser.parse_args()
    
    try:
        if args.mode == "scenario":
            run_scenario(
                scenario_name=args.scenario,
                num_turns=args.num_turns,
                verbose=args.verbose
            )
        else:
            run_interactive(session_id=args.session_id)
    
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
