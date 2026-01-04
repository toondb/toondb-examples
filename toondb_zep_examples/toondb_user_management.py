"""
ToonDB User Management Example - Inspired by Zep's user_example.py
Demonstrates CRUD operations for users and thread management
"""
import uuid
import json
import time
from toondb import Database


class ToonDBUserManager:
    """User management using ToonDB hierarchical paths"""
    
    def __init__(self, db_path="./toondb_user_data"):
        self.db = Database.open(db_path)
    
    def create_user(self, user_id=None, first_name=None, last_name=None, 
                   email=None, metadata=None):
        """Create a new user"""
        if user_id is None:
            user_id = f"user_{uuid.uuid4().hex}"
        
        # Store user fields
        if first_name:
            self.db.put(f"users.{user_id}.first_name".encode(), first_name.encode())
        if last_name:
            self.db.put(f"users.{user_id}.last_name".encode(), last_name.encode())
        if email:
            self.db.put(f"users.{user_id}.email".encode(), email.encode())
        if metadata:
            self.db.put(f"users.{user_id}.metadata".encode(), 
                       json.dumps(metadata).encode())
        
        # Store creation timestamp
        self.db.put(f"users.{user_id}.created_at".encode(), 
                   str(time.time()).encode())
        
        return user_id
    
    def get_user(self, user_id):
        """Retrieve user information"""
        user_data = {}
        
        # Check if user exists
        first_name_key = f"users.{user_id}.first_name".encode()
        if self.db.get(first_name_key) is None:
            return None
        
        # Scan all fields for this user
        for key, value in self.db.scan_prefix(f"users.{user_id}.".encode()):
            key_str = key.decode()
            field = key_str.split(".")[-1]
            
            value_str = value.decode()
            
            # Parse JSON metadata
            if field == "metadata":
                user_data[field] = json.loads(value_str)
            else:
                user_data[field] = value_str
        
        user_data["user_id"] = user_id
        return user_data
    
    def update_user(self, user_id, first_name=None, last_name=None, 
                   email=None, metadata=None):
        """Update user information"""
        # Check if user exists
        if self.get_user(user_id) is None:
            raise ValueError(f"User {user_id} does not exist")
        
        # Update fields
        if first_name:
            self.db.put(f"users.{user_id}.first_name".encode(), first_name.encode())
        if last_name:
            self.db.put(f"users.{user_id}.last_name".encode(), last_name.encode())
        if email:
            self.db.put(f"users.{user_id}.email".encode(), email.encode())
        if metadata:
            self.db.put(f"users.{user_id}.metadata".encode(), 
                       json.dumps(metadata).encode())
        
        return self.get_user(user_id)
    
    def list_users(self, limit=None):
        """List all users"""
        users = {}
        
        for key, value in self.db.scan_prefix(b"users."):
            key_str = key.decode()
            parts = key_str.split(".")
            
            if len(parts) >= 3:
                user_id = parts[1]
                field = parts[2]
                
                if user_id not in users:
                    users[user_id] = {"user_id": user_id}
                
                value_str = value.decode()
                
                if field == "metadata":
                    users[user_id][field] = json.loads(value_str)
                else:
                    users[user_id][field] = value_str
        
        user_list = list(users.values())
        
        if limit:
            return user_list[:limit]
        
        return user_list
    
    def delete_user(self, user_id):
        """Delete a user (soft delete with flag)"""
        # Check if user exists
        if self.get_user(user_id) is None:
            raise ValueError(f"User {user_id} does not exist")
        
        # Soft delete - set deleted flag
        self.db.put(f"users.{user_id}.deleted".encode(), b"true")
        self.db.put(f"users.{user_id}.deleted_at".encode(), 
                   str(time.time()).encode())
    
    def create_thread(self, thread_id=None, user_id=None):
        """Create a conversation thread for a user"""
        if thread_id is None:
            thread_id = f"thread_{uuid.uuid4().hex}"
        
        if user_id:
            self.db.put(f"threads.{thread_id}.user_id".encode(), user_id.encode())
        
        self.db.put(f"threads.{thread_id}.created_at".encode(), 
                   str(time.time()).encode())
        
        return thread_id
    
    def close(self):
        """Close database connection"""
        self.db.close()


def main():
    print("="*70)
    print("  ToonDB User Management Example")
    print("="*70 + "\n")
    
    manager = ToonDBUserManager()
    
    # Create multiple users
    print("Creating 3 users...")
    user_ids = []
    
    for i in range(3):
        user_id = manager.create_user(
            first_name=f"John{i}",
            last_name=f"Doe{i}",
            email=f"user{i}@example.com",
            metadata={"foo": "bar", "index": i}
        )
        user_ids.append(user_id)
        print(f"  ✓ Created user {i+1}: {user_id}")
    
    print()
    
    # Update the first user
    print("Updating first user...")
    updated_user = manager.update_user(
        user_id=user_ids[0],
        email="updated_user@example.com",
        first_name="UpdatedJohn",
        last_name="UpdatedDoe",
        metadata={"foo": "updated_bar"}
    )
    print(f"  ✓ Updated user: {updated_user['user_id']}")
    print(f"    New email: {updated_user['email']}")
    print(f"    New name: {updated_user['first_name']} {updated_user['last_name']}\n")
    
    # Create a thread for the first user
    print("Creating thread for first user...")
    thread_id = manager.create_thread(user_id=user_ids[0])
    print(f"  ✓ Created thread: {thread_id}\n")
    
    # List all users
    print("Listing all users...")
    all_users = manager.list_users()
    print(f"  Found {len(all_users)} users:")
    for user in all_users:
        print(f"    - {user['user_id']}: {user.get('first_name', 'N/A')} "
              f"{user.get('last_name', 'N/A')} ({user.get('email', 'N/A')})")
    print()
    
    # Delete the second user
    print(f"Deleting second user: {user_ids[1]}...")
    manager.delete_user(user_ids[1])
    print(f"  ✓ Deleted user: {user_ids[1]}\n")
    
    # Verify deletion
    deleted_user = manager.get_user(user_ids[1])
    print(f"Verifying deletion...")
    print(f"  User {user_ids[1]} deleted flag: {deleted_user.get('deleted', 'false')}\n")
    
    # Cleanup
    manager.close()
    print("✅ Example complete!")


if __name__ == "__main__":
    main()
