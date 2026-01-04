"""
ToonDB Advanced Travel Example
Complete sophisticated travel planning with entities and relationships

Equivalent to Zep's advanced.py - demonstrates complex domain modeling
"""
import uuid
import json
import time
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
from toondb import Database


# Entity Type Definitions
@dataclass
class Person:
    """Travel companion (spouse, friend, family, colleague)"""
    person_name: str
    age: Optional[int] = None
    relationship_to_user: Optional[str] = None
    travel_preferences: Optional[str] = None
    special_needs: Optional[str] = None


@dataclass
class Destination:
    """Travel destination"""
    destination_name: str
    country: str
    destination_type: Optional[str] = None  # city, country, region
    best_season: Optional[str] = None
    visa_required: Optional[str] = None


@dataclass
class Accommodation:
    """Lodging"""
    accommodation_name: str
    accommodation_type: str  # hotel, resort, hostel
    star_rating: Optional[int] = None
    location: Optional[str] = None
    nightly_rate: Optional[int] = None


@dataclass
class Experience:
    """Activity or tour"""
    experience_name: str
    experience_type: str  # tour, activity, attraction
    duration: Optional[str] = None
    price_per_person: Optional[int] = None
    skill_level: Optional[str] = None  # beginner, intermediate, advanced


@dataclass
class TravelService:
    """Airlines, restaurants, transport"""
    service_name: str
    service_type: str  # airline, restaurant, transport
    service_class: Optional[str] = None  # economy, business, luxury
    route_or_location: Optional[str] = None
    rating: Optional[str] = None


# Relationship Type Definitions
@dataclass
class Visits:
    """User visits destination"""
    user_id: str
    destination_id: str
    visit_purpose: Optional[str] = None
    visit_frequency: Optional[str] = None
    travel_dates: Optional[str] = None
    satisfaction_level: Optional[str] = None


@dataclass
class StaysAt:
    """User stays at accommodation"""
    user_id: str
    accommodation_id: str
    check_in_date: Optional[str] = None
    check_out_date: Optional[str] = None
    room_type: Optional[str] = None
    booking_status: Optional[str] = None


@dataclass
class Participates:
    """User participates in experience"""
    user_id: str
    experience_id: str
    participation_date: Optional[str] = None
    group_size: Optional[int] = None
    booking_method: Optional[str] = None
    experience_rating: Optional[str] = None


@dataclass
class Books:
    """User books travel service"""
    user_id: str
    service_id: str
    booking_date: Optional[str] = None
    service_date: Optional[str] = None
    confirmation_code: Optional[str] = None
    payment_method: Optional[str] = None


class TravelPlanningSystem:
    """
    Complete travel planning system using ToonDB
    
    Stores entities, relationships, and conversation threads
    """
    
    def __init__(self, db_path="./toondb_travel_data"):
        self.db = Database.open(db_path)
    
    # User Management
    def create_user(self, user_id=None, first_name=None, last_name=None, email=None):
        """Create user"""
        if user_id is None:
            user_id = f"user_{uuid.uuid4().hex[:8]}"
        
        self.db.put(f"users.{user_id}.first_name".encode(), (first_name or "").encode())
        self.db.put(f"users.{user_id}.last_name".encode(), (last_name or "").encode())
        self.db.put(f"users.{user_id}.email".encode(), (email or "").encode())
        self.db.put(f"users.{user_id}.created_at".encode(), str(time.time()).encode())
        
        return user_id
    
    # Thread Management
    def create_thread(self, thread_id=None, user_id=None):
        """Create conversation thread"""
        if thread_id is None:
            thread_id = f"thread_{uuid.uuid4().hex[:8]}"
        
        if user_id:
            self.db.put(f"threads.{thread_id}.user_id".encode(), user_id.encode())
        
        self.db.put(f"threads.{thread_id}.created_at".encode(), str(time.time()).encode())
        return thread_id
    
    def add_message(self, thread_id, message: Dict):
        """Add message to thread"""
        # Count existing messages
        msg_count = 0
        for key, _ in self.db.scan_prefix(f"threads.{thread_id}.messages.".encode()):
            if ".content" in key.decode():
                msg_count += 1
        
        msg_idx = msg_count + 1
        prefix = f"threads.{thread_id}.messages.{msg_idx}"
        
        self.db.put(f"{prefix}.role".encode(), message["role"].encode())
        self.db.put(f"{prefix}.name".encode(), message["name"].encode())
        self.db.put(f"{prefix}.content".encode(), message["content"].encode())
        self.db.put(f"{prefix}.timestamp".encode(), str(time.time()).encode())
    
    # Entity Management
    def store_entity(self, entity_type, entity_id, entity_data: Dict):
        """Store an entity"""
        for key, value in entity_data.items():
            if value is not None:
                path = f"entities.{entity_type}.{entity_id}.{key}"
                self.db.put(path.encode(), str(value).encode())
        
        return entity_id
    
    def get_entity(self, entity_type, entity_id) -> Optional[Dict]:
        """Retrieve an entity"""
        entity_data = {}
        prefix = f"entities.{entity_type}.{entity_id}."
        
        for key, value in self.db.scan_prefix(prefix.encode()):
            key_str = key.decode()
            field = key_str.split(".")[-1]
            entity_data[field] = value.decode()
        
        return entity_data if entity_data else None
    
    def list_entities(self, entity_type) -> List[Dict]:
        """List all entities of type"""
        entities = {}
        prefix = f"entities.{entity_type}."
        
        for key, value in self.db.scan_prefix(prefix.encode()):
            key_str = key.decode()
            parts = key_str.split(".")
            
            if len(parts) >= 4:
                entity_id = parts[2]
                field = parts[3]
                
                if entity_id not in entities:
                    entities[entity_id] = {"_id": entity_id}
                
                entities[entity_id][field] = value.decode()
        
        return list(entities.values())
    
    # Relationship Management
    def store_relationship(self, rel_type, rel_data: Dict):
        """Store a relationship"""
        rel_id = f"{rel_type}_{uuid.uuid4().hex[:8]}"
        
        for key, value in rel_data.items():
            if value is not None:
                path = f"relationships.{rel_type}.{rel_id}.{key}"
                self.db.put(path.encode(), str(value).encode())
        
        # Create indexes for querying
        user_id = rel_data.get("user_id")
        if user_id:
            self.db.put(f"user_relationships.{user_id}.{rel_type}.{rel_id}".encode(), b"1")
        
        return rel_id
    
    def get_user_relationships(self, user_id, rel_type=None) -> List[Dict]:
        """Get all relationships for a user"""
        relationships = []
        
        if rel_type:
            prefix = f"user_relationships.{user_id}.{rel_type}."
        else:
            prefix = f"user_relationships.{user_id}."
        
        for key, _ in self.db.scan_prefix(prefix.encode()):
            key_str = key.decode()
            parts = key_str.split(".")
            
            if len(parts) >= 4:
                actual_rel_type = parts[2]
                rel_id = parts[3]
                
                # Get full relationship data
                rel_prefix = f"relationships.{actual_rel_type}.{rel_id}."
                rel_data = {"_type": actual_rel_type, "_id": rel_id}
                
                for rkey, rvalue in self.db.scan_prefix(rel_prefix.encode()):
                    field = rkey.decode().split(".")[-1]
                    rel_data[field] = rvalue.decode()
                
                relationships.append(rel_data)
        
        return relationships
    
    def close(self):
        """Close database"""
        self.db.close()


# Test and validation functions
def test_entity_storage():
    """Test entity storage and retrieval"""
    print("\n" + "="*70)
    print("  TEST: Entity Storage")
    print("="*70 + "\n")
    
    system = TravelPlanningSystem()
    
    # Create destination
    rome = Destination(
        destination_name="Rome",
        country="Italy",
        destination_type="city",
        best_season="Spring/Fall"
    )
    
    rome_id = f"dest_{uuid.uuid4().hex[:8]}"
    system.store_entity("Destination", rome_id, asdict(rome))
    
    # Retrieve and validate
    retrieved = system.get_entity("Destination", rome_id)
    
    assert retrieved is not None, "Entity not found!"
    assert retrieved["destination_name"] == "Rome", "Name mismatch!"
    assert retrieved["country"] == "Italy", "Country mismatch!"
    
    print(f"✓ Entity stored: {rome_id}")
    print(f"✓ Retrieved: {retrieved['destination_name']}, {retrieved['country']}")
    print("✓ Validation passed!\n")
    
    system.close()
    return True


def test_relationship_tracking():
    """Test relationship storage and querying"""
    print("="*70)
    print("  TEST: Relationship Tracking")
    print("="*70 + "\n")
    
    system = TravelPlanningSystem()
    
    # Create user
    user_id = system.create_user(first_name="John", last_name="Doe")
    print(f"✓ Created user: {user_id}")
    
    # Create entities
    rome_id = f"dest_{uuid.uuid4().hex[:8]}"
    hotel_id = f"hotel_{uuid.uuid4().hex[:8]}"
    
    system.store_entity("Destination", rome_id, {"destination_name": "Rome", "country": "Italy"})
    system.store_entity("Accommodation", hotel_id, {"accommodation_name": "Hotel Roma", "star_rating": "4"})
    
    print(f"✓ Created destination: {rome_id}")
    print(f"✓ Created accommodation: {hotel_id}")
    
    # Create relationships
    visit_rel = Visits(
        user_id=user_id,
        destination_id=rome_id,
        visit_purpose="vacation",
        travel_dates="2024-05-15 to 2024-05-25"
    )
    
    stay_rel = StaysAt(
        user_id=user_id,
        accommodation_id=hotel_id,
        check_in_date="2024-05-15",
        check_out_date="2024-05-25"
    )
    
    visit_id = system.store_relationship("VISITS", asdict(visit_rel))
    stay_id = system.store_relationship("STAYS_AT", asdict(stay_rel))
    
    print(f"✓ Created relationship: VISITS ({visit_id})")
    print(f"✓ Created relationship: STAYS_AT ({stay_id})")
    
    # Query relationships
    user_rels = system.get_user_relationships(user_id)
    
    assert len(user_rels) == 2, f"Expected 2 relationships, got {len(user_rels)}"
    
    print(f"\n✓ Found {len(user_rels)} relationships for user")
    for rel in user_rels:
        print(f"   - {rel['_type']}: {rel.get('travel_dates', 'N/A')}")
    
    print("✓ Validation passed!\n")
    
    system.close()
    return True


def run_full_travel_scenario():
    """Run complete travel planning scenario"""
    print("="*70)
    print("  FULL SCENARIO: Multi-Session Travel Planning")
    print("="*70 + "\n")
    
    system = TravelPlanningSystem()
    
    # Create user
    user_id = system.create_user(
        first_name="John",
        last_name="Doe",
        email="john.doe@example.com"
    )
    print(f"✓ User created: John Doe ({user_id})\n")
    
    # Session 1: Italy trip
    print("Session 1: Planning Italy Trip")
    print("-" * 70)
    
    thread1 = system.create_thread(user_id=user_id)
    
    # Create entities
    maria_id = f"person_{uuid.uuid4().hex[:8]}"
    maria = Person(person_name="Maria", age=28, relationship_to_user="spouse")
    system.store_entity("Person", maria_id, asdict(maria))
    
    rome_id = f"dest_{uuid.uuid4().hex[:8]}"
    rome = Destination(destination_name="Rome", country="Italy", destination_type="city")
    system.store_entity("Destination", rome_id, asdict(rome))
    
    hotel_id = f"hotel_{uuid.uuid4().hex[:8]}"
    hotel = Accommodation(
        accommodation_name="Villa San Michele",
        accommodation_type="hotel",
        star_rating=4,
        nightly_rate=380
    )
    system.store_entity("Accommodation", hotel_id, asdict(hotel))
    
    # Create relationships
    visit = Visits(
        user_id=user_id,
        destination_id=rome_id,
        visit_purpose="vacation",
        travel_dates="September 15-25"
    )
    system.store_relationship("VISITS", asdict(visit))
    
    stay = StaysAt(
        user_id=user_id,
        accommodation_id=hotel_id,
        check_in_date="2024-09-15",
        check_out_date="2024-09-20",
        booking_status="confirmed"
    )
    system.store_relationship("STAYS_AT", asdict(stay))
    
    print(f"✓ Created companion: Maria (spouse, age 28)")
    print(f"✓ Created destination: Rome, Italy")
    print(f"✓ Created accommodation: Villa San Michele ($380/night)")
    print(f"✓ Created relationships: VISITS, STAYS_AT\n")
    
    # Verify data integrity
    user_relationships = system.get_user_relationships(user_id)
    assert len(user_relationships) >= 2, "Missing relationships!"
    
    destinations = system.list_entities("Destination")
    assert len(destinations) >= 1, "No destinations found!"
    
    print(f"✓ Validation: {len(user_relationships)} relationships stored")
    print(f"✓ Validation: {len(destinations)} destinations in system\n")
    
    # Query specific data
    print("Querying travel plans:")
    print("-" * 70)
    
    visits = system.get_user_relationships(user_id, "VISITS")
    for visit_rel in visits:
        dest_id = visit_rel.get("destination_id")
        if dest_id:
            dest = system.get_entity("Destination", dest_id)
            if dest:
                print(f"→ VISITING: {dest.get('destination_name')}, {dest.get('country')}")
                print(f"  Dates: {visit_rel.get('travel_dates')}")
                print(f"  Purpose: {visit_rel.get('visit_purpose')}")
    
    print()
    
    stays = system.get_user_relationships(user_id, "STAYS_AT")
    for stay_rel in stays:
        hotel_id = stay_rel.get("accommodation_id")
        if hotel_id:
            hotel = system.get_entity("Accommodation", hotel_id)
            if hotel:
                print(f"→ STAYING AT: {hotel.get('accommodation_name')}")
                print(f"  Type: {hotel.get('accommodation_type')}")
                print(f"  Check-in: {stay_rel.get('check_in_date')}")
                print(f"  Check-out: {stay_rel.get('check_out_date')}")
                print(f"  Status: {stay_rel.get('booking_status')}")
    
    print("\n✅ Full scenario complete!")
    print(f"✅ All validations passed!\n")
    
    system.close()
    return True


def main():
    """Run all tests"""
    print("="*70)
    print("  ToonDB Advanced Travel Planning System")
    print("  Comprehensive Testing & Validation")
    print("="*70)
    
    try:
        # Run tests
        test_entity_storage()
        test_relationship_tracking()
        run_full_travel_scenario()
        
        print("="*70)
        print("  ✅ ALL TESTS PASSED")
        print("="*70)
        
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        raise
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        raise


if __name__ == "__main__":
    main()
