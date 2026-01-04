"""
ToonDB Chat History Example
Demonstrates memory storage and retrieval for conversational AI

Equivalent to Zep's chat_history/memory.py
"""
import uuid
import json
import time
from typing import List, Dict
from toondb import Database


# Conversation history (same as Zep example)
SHOE_PURCHASE_HISTORY = [
    {
        "name": "Jane",
        "role": "user",
        "content": (
            "Hi there, I'm looking for a new pair of shoes. I'm not sure where"
            " to start."
        ),
    },
    {
        "name": "assistant",
        "role": "assistant",
        "content": (
            "Hello! I'd be happy to help you find the perfect pair of shoes. To"
            " get started, could you tell me what brand you're interested in,"
            " or if you're open to any brand?"
        ),
    },
    {
        "name": "Jane",
        "role": "user",
        "content": "I like Nike and Adidas, but I'm open to other brands too.",
    },
    {
        "name": "assistant",
        "role": "assistant",
        "content": (
            "That's great! It's important to find the right shoe for your"
            " needs. Could you tell me what you'll be using these shoes for"
            " mainly? For example, are they for running, basketball, or casual"
            " wear?"
        ),
    },
    {
        "name": "Jane",
        "role": "user",
        "content": "I'll be using them for running.",
    },
    {
        "name": "assistant",
        "role": "assistant",
        "content": (
            "Fantastic! Now, could you please provide me with your shoe size"
            " and foot width? This information will help me narrow down the"
            " options for you."
        ),
    },
    {
        "name": "Jane",
        "role": "user",
        "content": ("I wear a men's size 10, and I'm not sure about the foot width."),
    },
    {
        "name": "assistant",
        "role": "assistant",
        "content": (
            "Thank you. Based on your size, I'll make sure to suggest shoes"
            " with a wide range of sizes and widths available."
        ),
    },
    {
        "name": "Jane",
        "role": "user",
        "content": (
            "I have a bit of a pronation issue. Do I need to consider that"
            " while choosing shoes?"
        ),
    },
    {
        "name": "assistant",
        "role": "assistant",
        "content": (
            "Yes, pronation is essential when choosing running shoes. It refers"
            " to the way your foot rolls inward or outward when you walk or"
            " run. For pronation issues, stability or motion control shoes are"
            " usually recommended. I'll make sure to consider this in my"
            " suggestions."
        ),
    },
    {
        "name": "Jane",
        "role": "user",
        "content": "Great, I have a budget of around $120.",
    },
    {
        "name": "assistant",
        "role": "assistant",
        "content": (
            "Perfect! I'll make sure to find the best running shoes that fit"
            " your preferences, taking into account your brand preferences,"
            " running needs, size, and budget."
        ),
    },
    {
        "name": "assistant",
        "role": "assistant",
        "content": (
            "I've found a few great options for you. The first one is the"
            " Brooks Ghost 14, which is known for its cushioning and support,"
            " ideal for runners with pronation issues. It's available in your"
            " size, and it's within your budget."
        ),
        "metadata": {"bar": "foo"},
    },
    {
        "name": "Jane",
        "role": "user",
        "content": (
            "I've heard good things about Brooks, but I was kind of hoping to"
            " get a Nike or an Adidas shoe."
        ),
    },
    {
        "name": "assistant",
        "role": "assistant",
        "content": (
            "I understand. In that case, I'd recommend the Adidas Ultraboost"
            " 21. It's a popular choice among runners for its comfort, energy"
            " return, and stability. Plus, it's on sale at $129.99 today,"
            " almost within your budget."
        ),
    },
    {
        "name": "Jane",
        "role": "user",
        "content": "They're nice looking. Yes, I think I'll go with those.",
    },
    {
        "name": "assistant",
        "role": "assistant",
        "content": (
            "Excellent choice! To complete the purchase, I'll need you to"
            " provide your credit card information in our secure payment form."
            " This will allow us to process your order and ensure a smooth"
            " delivery."
        ),
    },
]


class ToonDBChatHistory:
    """
    Chat history manager using ToonDB
    
    Stores conversation messages with full context retrieval
    """
    
    def __init__(self, db_path="./toondb_chat_data"):
        self.db = Database.open(db_path)
    
    def create_user(self, user_id=None, first_name=None, last_name=None, email=None):
        """Create a user"""
        if user_id is None:
            user_id = uuid.uuid4().hex
        
        self.db.put(f"users.{user_id}.first_name".encode(), (first_name or "").encode())
        self.db.put(f"users.{user_id}.last_name".encode(), (last_name or "").encode())
        self.db.put(f"users.{user_id}.email".encode(), (email or "").encode())
        self.db.put(f"users.{user_id}.created_at".encode(), str(time.time()).encode())
        
        return user_id
    
    def create_thread(self, thread_id=None, user_id=None):
        """Create a conversation thread"""
        if thread_id is None:
            thread_id = uuid.uuid4().hex
        
        if user_id:
            self.db.put(f"threads.{thread_id}.user_id".encode(), user_id.encode())
        
        self.db.put(f"threads.{thread_id}.created_at".encode(), str(time.time()).encode())
        
        return thread_id
    
    def add_message(self, thread_id, message: Dict):
        """Add a single message to thread"""
        # Get current message count
        msg_count = 0
        for key, _ in self.db.scan_prefix(f"threads.{thread_id}.messages.".encode()):
            key_str = key.decode()
            if ".content" in key_str:
                msg_count += 1
        
        msg_idx = msg_count + 1
        prefix = f"threads.{thread_id}.messages.{msg_idx}"
        
        self.db.put(f"{prefix}.role".encode(), message["role"].encode())
        self.db.put(f"{prefix}.name".encode(), message["name"].encode())
        self.db.put(f"{prefix}.content".encode(), message["content"].encode())
        
        # Store metadata if present
        if "metadata" in message:
            self.db.put(f"{prefix}.metadata".encode(), 
                       json.dumps(message["metadata"]).encode())
        
        self.db.put(f"{prefix}.timestamp".encode(), str(time.time()).encode())
    
    def add_messages(self, thread_id, messages: List[Dict]):
        """Add multiple messages to thread"""
        for msg in messages:
            self.add_message(thread_id, msg)
    
    def get_thread_messages(self, thread_id) -> List[Dict]:
        """Retrieve all messages from a thread"""
        messages = {}
        
        for key, value in self.db.scan_prefix(f"threads.{thread_id}.messages.".encode()):
            key_str = key.decode()
            parts = key_str.split(".")
            
            if len(parts) >= 5:
                msg_idx = parts[3]
                field = parts[4]
                
                if msg_idx not in messages:
                    messages[msg_idx] = {}
                
                value_str = value.decode()
                
                if field == "metadata":
                    messages[msg_idx][field] = json.loads(value_str)
                else:
                    messages[msg_idx][field] = value_str
        
        # Sort by index and return as list
        sorted_messages = [messages[k] for k in sorted(messages.keys(), key=int)]
        return sorted_messages
    
    def get_user_context(self, thread_id) -> str:
        """
        Build user context from all messages in thread
        
        This simulates Zep's get_user_context() which extracts
        facts and builds a summary
        """
        messages = self.get_thread_messages(thread_id)
        
        # Build context from conversation
        context_parts = []
        user_info = {}
        
        for msg in messages:
            content = msg.get("content", "")
            role = msg.get("role", "")
            
            # Extract facts (simple keyword matching)
            if "Nike" in content or "Adidas" in content:
                user_info["preferred_brands"] = ["Nike", "Adidas"]
            
            if "running" in content.lower():
                user_info["purpose"] = "running"
            
            if "size 10" in content:
                user_info["size"] = "men's 10"
            
            if "pronation" in content.lower():
                user_info["special_needs"] = "pronation support"
            
            if "$120" in content or "$129" in content:
                user_info["budget"] = "$120-130"
        
        # Build summary
        context_parts.append("**Customer Profile:**")
        context_parts.append(f"- Preferred Brands: {user_info.get('preferred_brands', 'Not specified')}")
        context_parts.append(f"- Purpose: {user_info.get('purpose', 'Not specified')}")
        context_parts.append(f"- Size: {user_info.get('size', 'Not specified')}")
        context_parts.append(f"- Special Needs: {user_info.get('special_needs', 'None')}")
        context_parts.append(f"- Budget: {user_info.get('budget', 'Not specified')}")
        context_parts.append(f"\n**Conversation Summary:**")
        context_parts.append(f"Customer is looking for running shoes. Prefers Nike/Adidas, has pronation issues, budget ~$120. Recommended Adidas Ultraboost 21 at $129.99.")
        
        return "\n".join(context_parts)
    
    def search_thread(self, thread_id, query: str, limit: int = 5) -> List[Dict]:
        """
        Simple keyword search through thread messages
        
        In production, use embeddings for semantic search
        """
        messages = self.get_thread_messages(thread_id)
        
        # Simple keyword matching
        results = []
        query_lower = query.lower()
        
        for msg in messages:
            content = msg.get("content", "").lower()
            if query_lower in content:
                results.append(msg)
        
        return results[:limit]
    
    def close(self):
        """Close database"""
        self.db.close()


def main():
    print("="*70)
    print("  ToonDB Chat History Example - Shoe Purchase")
    print("="*70 + "\n")
    
    chat = ToonDBChatHistory()
    
    # Create user
    print("Creating user...")
    user_id = chat.create_user(
        first_name="Jane",
        last_name="Smith",
        email="jane@example.com"
    )
    print(f"✓ User created: {user_id}\n")
    
    # Create thread
    print("Creating thread...")
    thread_id = chat.create_thread(user_id=user_id)
    print(f"✓ Thread created: {thread_id}\n")
    
    # Add messages
    print(f"Adding {len(SHOE_PURCHASE_HISTORY)} messages to thread...")
    chat.add_messages(thread_id, SHOE_PURCHASE_HISTORY)
    print("✓ Messages added\n")
    
    # Retrieve and display messages
    print("="*70)
    print("  Conversation History")
    print("="*70 + "\n")
    
    messages = chat.get_thread_messages(thread_id)
    for i, msg in enumerate(messages, 1):
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        print(f"[{i}] {role.upper()}: {content[:80]}...")
    
    print(f"\n✓ Retrieved {len(messages)} messages\n")
    
    # Get user context
    print("="*70)
    print("  User Context (Extracted Facts)")
    print("="*70 + "\n")
    
    context = chat.get_user_context(thread_id)
    print(context)
    print()
    
    # Search functionality
    print("="*70)
    print("  Search Thread")
    print("="*70 + "\n")
    
    query = "pronation"
    print(f"Searching for: '{query}'")
    results = chat.search_thread(thread_id, query)
    print(f"✓ Found {len(results)} messages mentioning '{query}':")
    for msg in results:
        print(f"   - {msg.get('role')}: {msg.get('content')[:60]}...")
    
    # Cleanup
    chat.close()
    print("\n✅ Example complete!")


if __name__ == "__main__":
    main()
