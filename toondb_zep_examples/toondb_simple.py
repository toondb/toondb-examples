"""
ToonDB Simple Example - Inspired by Zep's simple.py
Demonstrates basic user and conversation thread management
"""
import uuid
from toondb import Database


def main():
    print("="*70)
    print("  ToonDB Simple Example - User & Thread Management")
    print("="*70 + "\n")
    
    # Initialize database
    db = Database.open("./toondb_simple_data")
    print("✅ Database initialized\n")
    
    # User information
    first_name = "John"
    last_name = "Doe"
    email = "john.doe@example.com"
    user_role = f"{first_name} {last_name}"
    assistant_role = "ShoeSalesSupportBot"
    
    # Create user
    user_id = f"user_{uuid.uuid4().hex[:8]}"
    print(f"Creating user: {user_id}")
    
    db.put(f"users.{user_id}.first_name".encode(), first_name.encode())
    db.put(f"users.{user_id}.last_name".encode(), last_name.encode())
    db.put(f"users.{user_id}.email".encode(), email.encode())
    
    print(f"  ✓ User created: {first_name} {last_name} ({email})\n")
    
    # Define conversation threads
    threads = [
        [
            {"role": "user", "name": user_role, "content": "Help me find some new running shoes. Adidas are my favorite"},
            {"role": "assistant", "name": assistant_role, "content": "Can do! How about the Adidas Ultra Boost 21 for $100?"},
            {"role": "user", "name": user_role, "content": "Sounds good to me."},
        ],
        [
            {"role": "user", "name": user_role, "content": "I tried the Adidas ultra boost, and I no longer like Adidas. I want Puma."},
            {"role": "assistant", "name": assistant_role, "content": "I see. Do you want to try the Puma Velocity Nitro 2?"},
            {"role": "user", "name": user_role, "content": "I used to own the Velocity Nitro 2. What's another Puma Shoe I can try?"},
            {"role": "assistant", "name": assistant_role, "content": "I see. Do you want to try the Puma Deviate Nitro Elite?"},
            {"role": "user", "name": user_role, "content": "Sure"},
        ]
    ]
    
    # Create threads and add messages
    for thread_idx, messages in enumerate(threads, 1):
        thread_id = f"thread_{uuid.uuid4().hex[:8]}"
        
        print(f"Creating thread {thread_idx}: {thread_id}")
        
        # Store thread metadata
        db.put(f"threads.{thread_id}.user_id".encode(), user_id.encode())
        db.put(f"threads.{thread_id}.thread_number".encode(), str(thread_idx).encode())
        
        # Store messages
        for msg_idx, msg in enumerate(messages, 1):
            prefix = f"threads.{thread_id}.messages.{msg_idx}"
            
            db.put(f"{prefix}.role".encode(), msg["role"].encode())
            db.put(f"{prefix}.name".encode(), msg["name"].encode())
            db.put(f"{prefix}.content".encode(), msg["content"].encode())
        
        print(f"  ✓ Stored {len(messages)} messages\n")
    
    # Retrieve and display stored data
    print("="*70)
    print("  Retrieving Stored Data")
    print("="*70 + "\n")
    
    # Get user info
    print(f"User: {user_id}")
    first = db.get(f"users.{user_id}.first_name".encode()).decode()
    last = db.get(f"users.{user_id}.last_name".encode()).decode()
    user_email = db.get(f"users.{user_id}.email".encode()).decode()
    print(f"  Name: {first} {last}")
    print(f"  Email: {user_email}\n")
    
    # List all threads for this user
    print(f"Scanning threads for user {user_id}...\n")
    
    thread_count = 0
    # Scan all keys starting with "threads."
    # In a production app, you would use a secondary index: indexes.user_threads.{user_id}.{thread_id}
    # For this simple example, we'll scan unique thread IDs
    
    seen_threads = set()
    
    for key, _ in db.scan_prefix(b"threads."):
        key_str = key.decode()
        parts = key_str.split(".")
        
        if len(parts) >= 2:
            thread_id = parts[1]
            
            if thread_id in seen_threads:
                continue
                
            seen_threads.add(thread_id)
            
            # Check if this thread belongs to our user
            try:
                owner_id_bytes = db.get(f"threads.{thread_id}.user_id".encode())
                if owner_id_bytes and owner_id_bytes.decode() == user_id:
                    thread_count += 1
                    thread_num_bytes = db.get(f"threads.{thread_id}.thread_number".encode())
                    thread_num = thread_num_bytes.decode() if thread_num_bytes else "?"
                    
                    print(f"Thread {thread_num} ({thread_id}):")
                    
                    # Get messages for this thread
                    msg_num = 1
                    while True:
                        role_key = f"threads.{thread_id}.messages.{msg_num}.role".encode()
                        role = db.get(role_key)
                        
                        if role is None:
                            break
                        
                        content_key = f"threads.{thread_id}.messages.{msg_num}.content".encode()
                        content = db.get(content_key).decode()
                        
                        print(f"  [{role.decode()}]: {content}")
                        msg_num += 1
                    print()
            except Exception as e:
                print(f"Error reading thread {thread_id}: {e}")
    
    print(f"✅ Total threads found: {thread_count}")
    
    # Cleanup
    db.close()
    print("\n✅ Example complete!")


if __name__ == "__main__":
    main()
