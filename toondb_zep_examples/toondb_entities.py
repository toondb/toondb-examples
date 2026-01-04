"""
ToonDB Entity & Relationship Example - Inspired by Zep's advanced.py
Demonstrates hierarchical entity storage and relationship tracking
"""
import uuid
import json
import time
from toondb import Database
from typing import Dict, Any, List, Optional


class Entity:
    """Base class for entities"""
    def __init__(self, entity_id=None, entity_type=None, **fields):
        self.entity_id = entity_id or f"{entity_type}_{uuid.uuid4().hex[:8]}"
        self.entity_type = entity_type
        self.fields = fields
    
    def to_dict(self):
        return {
            "entity_id": self.entity_id,
            "entity_type": self.entity_type,
            **self.fields
        }


class Relationship:
    """Represents a relationship between entities"""
    def __init__(self, rel_id=None, rel_type=None, source_id=None, 
                 target_id=None, **properties):
        self.rel_id = rel_id or f"{rel_type}_{uuid.uuid4().hex[:8]}"
        self.rel_type = rel_type
        self.source_id = source_id
        self.target_id = target_id
        self.properties = properties
    
    def to_dict(self):
        return {
            "rel_id": self.rel_id,
            "rel_type": self.rel_type,
            "source_id": self.source_id,
            "target_id": self.target_id,
            **self.properties
        }


class ToonDBEntityStore:
    """Entity and relationship storage using ToonDB"""
    
    def __init__(self, db_path="./toondb_entity_data"):
        self.db = Database.open(db_path)
    
    def store_entity(self, entity: Entity):
        """Store an entity"""
        entity_data = entity.to_dict()
        
        # Store each field
        for key, value in entity_data.items():
            path = f"entities.{entity.entity_type}.{entity.entity_id}.{key}"
            self.db.put(path.encode(), str(value).encode())
        
        return entity.entity_id
    
    def get_entity(self, entity_type: str, entity_id: str) -> Optional[Dict]:
        """Retrieve an entity"""
        entity_data = {}
        prefix = f"entities.{entity_type}.{entity_id}."
        
        for key, value in self.db.scan_prefix(prefix.encode()):
            key_str = key.decode()
            field = key_str.split(".")[-1]
            entity_data[field] = value.decode()
        
        return entity_data if entity_data else None
    
    def list_entities_by_type(self, entity_type: str) -> List[Dict]:
        """List all entities of a given type"""
        entities = {}
        prefix = f"entities.{entity_type}."
        
        for key, value in self.db.scan_prefix(prefix.encode()):
            key_str = key.decode()
            parts = key_str.split(".")
            
            if len(parts) >= 4:
                entity_id = parts[2]
                field = parts[3]
                
                if entity_id not in entities:
                    entities[entity_id] = {}
                
                entities[entity_id][field] = value.decode()
        
        return list(entities.values())
    
    def store_relationship(self, relationship: Relationship):
        """Store a relationship"""
        rel_data = relationship.to_dict()
        
        # Store relationship data
        for key, value in rel_data.items():
            path = f"relationships.{relationship.rel_type}.{relationship.rel_id}.{key}"
            self.db.put(path.encode(), str(value).encode())
        
        # Create reverse index for querying
        # Index: source -> list of relationships
        source_index = f"indexes.source.{relationship.source_id}.{relationship.rel_id}"
        self.db.put(source_index.encode(), relationship.rel_type.encode())
        
        # Index: target -> list of relationships
        target_index = f"indexes.target.{relationship.target_id}.{relationship.rel_id}"
        self.db.put(target_index.encode(), relationship.rel_type.encode())
        
        return relationship.rel_id
    
    def get_relationships_from(self, entity_id: str) -> List[Dict]:
        """Get all relationships where entity is the source"""
        relationships = []
        prefix = f"indexes.source.{entity_id}."
        
        for key, value in self.db.scan_prefix(prefix.encode()):
            key_str = key.decode()
            rel_id = key_str.split(".")[-1]
            rel_type = value.decode()
            
            # Get full relationship data
            rel_prefix = f"relationships.{rel_type}.{rel_id}."
            rel_data = {}
            
            for rkey, rvalue in self.db.scan_prefix(rel_prefix.encode()):
                field = rkey.decode().split(".")[-1]
                rel_data[field] = rvalue.decode()
            
            if rel_data:
                relationships.append(rel_data)
        
        return relationships
    
    def get_relationships_to(self, entity_id: str) -> List[Dict]:
        """Get all relationships where entity is the target"""
        relationships = []
        prefix = f"indexes.target.{entity_id}."
        
        for key, value in self.db.scan_prefix(prefix.encode()):
            key_str = key.decode()
            rel_id = key_str.split(".")[-1]
            rel_type = value.decode()
            
            # Get full relationship data
            rel_prefix = f"relationships.{rel_type}.{rel_id}."
            rel_data = {}
            
            for rkey, rvalue in self.db.scan_prefix(rel_prefix.encode()):
                field = rkey.decode().split(".")[-1]
                rel_data[field] = rvalue.decode()
            
            if rel_data:
                relationships.append(rel_data)
        
        return relationships
    
    def close(self):
        """Close database connection"""
        self.db.close()


def main():
    print("="*70)
    print("  ToonDB Entity & Relationship Example")
    print("="*70 + "\n")
    
    store = ToonDBEntityStore()
    
    # Create entities - Travel planning example
    print("Creating entities...\n")
    
    # Create a user/traveler
    user = Entity(
        entity_type="User",
        first_name="John",
        last_name="Doe",
        email="john.doe@example.com"
    )
    user_id = store.store_entity(user)
    print(f"✓ Created User: {user_id}")
    
    # Create destinations
    rome = Entity(
        entity_type="Destination",
        destination_name="Rome",
        country="Italy",
        destination_type="city",
        best_season="Spring/Fall"
    )
    rome_id = store.store_entity(rome)
    print(f"✓ Created Destination: {rome_id} (Rome)")
    
    tokyo = Entity(
        entity_type="Destination",
        destination_name="Tokyo",
        country="Japan",
        destination_type="city",
        best_season="Spring"
    )
    tokyo_id = store.store_entity(tokyo)
    print(f"✓ Created Destination: {tokyo_id} (Tokyo)")
    
    # Create accommodations
    hotel = Entity(
        entity_type="Accommodation",
        accommodation_name="Hotel Roma",
        accommodation_type="hotel",
        star_rating="4",
        location="Centro Storico",
        nightly_rate="200"
   )
    hotel_id = store.store_entity(hotel)
    print(f"✓ Created Accommodation: {hotel_id} (Hotel Roma)")
    
    # Create an experience
    tour = Entity(
        entity_type="Experience",
        experience_name="Colosseum Tour",
        experience_type="tour",
        duration="3 hours",
        price_per_person="50"
    )
    tour_id = store.store_entity(tour)
    print(f"✓ Created Experience: {tour_id} (Colosseum Tour)\n")
    
    # Create relationships
    print("Creating relationships...\n")
    
    # User visits Rome
    visit_rel = Relationship(
        rel_type="VISITS",
        source_id=user_id,
        target_id=rome_id,
        visit_purpose="vacation",
        visit_frequency="first time",
        travel_dates="2024-05-15 to 2024-05-25"
    )
    store.store_relationship(visit_rel)
    print(f"✓ Created relationship: User VISITS Rome")
    
    # User stays at hotel
    stay_rel = Relationship(
        rel_type="STAYS_AT",
        source_id=user_id,
        target_id=hotel_id,
        check_in_date="2024-05-15",
        check_out_date="2024-05-25",
        room_type="Deluxe"
    )
    store.store_relationship(stay_rel)
    print(f"✓ Created relationship: User STAYS_AT Hotel Roma")
    
    # User participates in tour
    participate_rel = Relationship(
        rel_type="PARTICIPATES",
        source_id=user_id,
        target_id=tour_id,
        participation_date="2024-05-17",
        group_size="2"
    )
    store.store_relationship(participate_rel)
    print(f"✓ Created relationship: User PARTICIPATES Colosseum Tour\n")
    
    # Query and display data
    print("="*70)
    print("  Querying Stored Data")
    print("="*70 + "\n")
    
    # List all destinations
    print("All Destinations:")
    destinations = store.list_entities_by_type("Destination")
    for dest in destinations:
        print(f"  - {dest.get('destination_name')}, {dest.get('country')}")
    print()
    
    # Get user's relationships
    print(f"User's travel plans (relationships from {user_id}):")
    user_rels = store.get_relationships_from(user_id)
    for rel in user_rels:
        rel_type = rel.get('rel_type')
        target = rel.get('target_id')
        
        if rel_type == "VISITS":
            dest = store.get_entity("Destination", target)
            print(f"  → VISITS: {dest.get('destination_name')} "
                  f"({rel.get('travel_dates')})")
        
        elif rel_type == "STAYS_AT":
            hotel = store.get_entity("Accommodation", target)
            print(f"  → STAYS_AT: {hotel.get('accommodation_name')} "
                  f"({rel.get('check_in_date')} to {rel.get('check_out_date')})")
        
        elif rel_type == "PARTICIPATES":
            exp = store.get_entity("Experience", target)
            print(f"  → PARTICIPATES: {exp.get('experience_name')} "
                  f"on {rel.get('participation_date')}")
    
    print()
    
    # Cleanup
    store.close()
    print("✅ Example complete!")


if __name__ == "__main__":
    main()
