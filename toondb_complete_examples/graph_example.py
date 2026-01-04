"""
ToonDB Graph Example
Demonstrates storing and querying graph-like data with episodes, nodes, and edges

Equivalent to Zep's graph_example/graph_example.py
"""
import uuid
import json
import time
from typing import List, Dict, Optional
from toondb import Database


class ToonDBGraph:
    """
    Graph storage and retrieval using ToonDB hierarchical paths
    
    Simulates knowledge graph with:
    - Episodes (data points like messages or facts)
    - Nodes (entities extracted from episodes)
    - Edges (relationships between nodes)
    """
    
    def __init__(self, db_path="./toondb_graph_data"):
        self.db = Database.open(db_path)
    
    def create_graph(self, graph_id, name=None, description=None):
        """Create a graph"""
        self.db.put(f"graphs.{graph_id}.name".encode(), (name or "").encode())
        self.db.put(f"graphs.{graph_id}.description".encode(), (description or "").encode())
        self.db.put(f"graphs.{graph_id}.created_at".encode(), str(time.time()).encode())
        
        return {"graph_id": graph_id, "name": name, "description": description}
    
    def add_episode(self, graph_id, data, episode_type="text"):
        """
        Add an episode (data point) to the graph
        
        Episodes are raw input that can be processed into nodes/edges
        """
        episode_id = uuid.uuid4().hex
        
        self.db.put(f"episodes.{graph_id}.{episode_id}.data".encode(), data.encode())
        self.db.put(f"episodes.{graph_id}.{episode_id}.type".encode(), episode_type.encode())
        self.db.put(f"episodes.{graph_id}.{episode_id}.created_at".encode(), 
                   str(time.time()).encode())
        
        # Extract entities if JSON
        if episode_type == "json":
            self._extract_from_json(graph_id, episode_id, data)
        else:
            self._extract_from_text(graph_id, episode_id, data)
        
        return episode_id
    
    def _extract_from_text(self, graph_id, episode_id, text):
        """Simple entity extraction from text"""
        # This is a simplified version - in production, use NLP
        words = text.split()
        
        # Look for capitalized words (simple named entity recognition)
        entities = [w for w in words if w[0].isupper() and len(w) > 2]
        
        for entity in entities:
            node_id = self._create_node(graph_id, entity, "Person")
            
            # Link episode to node
            self.db.put(f"episode_nodes.{episode_id}.{node_id}".encode(), b"mentions")
    
    def _extract_from_json(self, graph_id, episode_id, json_str):
        """Extract entities from JSON data"""
        try:
            data = json.loads(json_str)
            
            # Extract as nodes
            for key, value in data.items():
                if isinstance(value, str):
                    node_id = self._create_node(graph_id, value, key.title())
                    self.db.put(f"episode_nodes.{episode_id}.{node_id}".encode(), 
                               key.encode())
        except:
            pass
    
    def _create_node(self, graph_id, name, node_type):
        """Create or get existing node"""
        # Check if node exists
        node_id = None
        for key, value in self.db.scan_prefix(f"nodes.{graph_id}.".encode()):
            key_str = key.decode()
            if ".name" in key_str and value.decode() == name:
                node_id = key_str.split(".")[2]
                break
        
        if not node_id:
            node_id = uuid.uuid4().hex[:8]
            self.db.put(f"nodes.{graph_id}.{node_id}.name".encode(), name.encode())
            self.db.put(f"nodes.{graph_id}.{node_id}.type".encode(), node_type.encode())
            self.db.put(f"nodes.{graph_id}.{node_id}.created_at".encode(), 
                       str(time.time()).encode())
        
        return node_id
    
    def create_edge(self, graph_id, source_node, target_node, edge_type, properties=None):
        """Create an edge (relationship) between two nodes"""
        edge_id = uuid.uuid4().hex[:8]
        
        self.db.put(f"edges.{graph_id}.{edge_id}.source".encode(), source_node.encode())
        self.db.put(f"edges.{graph_id}.{edge_id}.target".encode(), target_node.encode())
        self.db.put(f"edges.{graph_id}.{edge_id}.type".encode(), edge_type.encode())
        
        if properties:
            self.db.put(f"edges.{graph_id}.{edge_id}.properties".encode(),
                       json.dumps(properties).encode())
        
        # Create indexes for graph traversal
        self.db.put(f"node_edges.{source_node}.out.{edge_id}".encode(), target_node.encode())
        self.db.put(f"node_edges.{target_node}.in.{edge_id}".encode(), source_node.encode())
        
        return edge_id
    
    def get_episodes(self, graph_id, last_n=None) -> List[Dict]:
        """Get episodes from a graph"""
        episodes = {}
        
        for key, value in self.db.scan_prefix(f"episodes.{graph_id}.".encode()):
            key_str = key.decode()
            parts = key_str.split(".")
            
            if len(parts) >= 4:
                episode_id = parts[2]
                field = parts[3]
                
                if episode_id not in episodes:
                    episodes[episode_id] = {"uuid_": episode_id}
                
                episodes[episode_id][field] = value.decode()
        
        episode_list = list(episodes.values())
        
        # Sort by created_at (most recent first)
        episode_list.sort(key=lambda x: float(x.get("created_at", 0)), reverse=True)
        
        if last_n:
            return episode_list[:last_n]
        
        return episode_list
    
    def get_nodes(self, graph_id) -> List[Dict]:
        """Get all nodes from a graph"""
        nodes = {}
        
        for key, value in self.db.scan_prefix(f"nodes.{graph_id}.".encode()):
            key_str = key.decode()
            parts = key_str.split(".")
            
            if len(parts) >= 4:
                node_id = parts[2]
                field = parts[3]
                
                if node_id not in nodes:
                    nodes[node_id] = {"node_id": node_id}
                
                nodes[node_id][field] = value.decode()
        
        return list(nodes.values())
    
    def get_edges(self, graph_id) -> List[Dict]:
        """Get all edges from a graph"""
        edges = {}
        
        for key, value in self.db.scan_prefix(f"edges.{graph_id}.".encode()):
            key_str = key.decode()
            parts = key_str.split(".")
            
            if len(parts) >= 4:
                edge_id = parts[2]
                field = parts[3]
                
                if edge_id not in edges:
                    edges[edge_id] = {"edge_id": edge_id}
                
                value_str = value.decode()
                
                if field == "properties":
                    edges[edge_id][field] = json.loads(value_str)
                else:
                    edges[edge_id][field] = value_str
        
        return list(edges.values())
    
    def search(self, graph_id, query) -> List[Dict]:
        """Search episodes and nodes for query"""
        results = []
        query_lower = query.lower()
        
        # Search episodes
        episodes = self.get_episodes(graph_id)
        for episode in episodes:
            if query_lower in episode.get("data", "").lower():
                results.append({
                    "type": "episode",
                    "id": episode["uuid_"],
                    "data": episode.get("data", ""),
                    "match_type": "episode_content"
                })
        
        # Search nodes
        nodes = self.get_nodes(graph_id)
        for node in nodes:
            if query_lower in node.get("name", "").lower():
                results.append({
                    "type": "node",
                    "id": node["node_id"],
                    "name": node.get("name", ""),
                    "node_type": node.get("type", ""),
                    "match_type": "node_name"
                })
        
        return results
    
    def close(self):
        """Close database"""
        self.db.close()


def main():
    print("="*70)
    print("  ToonDB Graph Example")
    print("="*70 + "\n")
    
    graph = ToonDBGraph()
    
    # Create graph
    graph_id = f"slack:{uuid.uuid4().hex[:8]}"
    print(f"Creating graph: {graph_id}")
    graph.create_graph(
        graph_id=graph_id,
        name="My Knowledge Graph",
        description="This is my knowledge graph"
    )
    print("✓ Graph created\n")
    
    # Add text episode
    print("Adding text episode...")
    graph.add_episode(graph_id, "This is a test episode", "text")
    print("✓ Episode added\n")
    
    # Add meaningful episode
    print("Adding meaningful episode...")
    graph.add_episode(graph_id, "Eric Clapton is a rock star", "text")
    print("✓ Episode added\n")
    
    # Add JSON episode
    print("Adding JSON episode...")
    json_data = '{"name": "Eric Clapton", "age": 78, "genre": "Rock"}'
    graph.add_episode(graph_id, json_data, "json")
    print("✓ JSON episode added\n")
    
    # Get episodes
    print("="*70)
    print("  Episodes")
    print("="*70 + "\n")
    
    episodes = graph.get_episodes(graph_id, last_n=5)
    print(f"Found {len(episodes)} episodes:")
    for ep in episodes:
        print(f"  - [{ep.get('type')}] {ep.get('data')[:60]}...")
    print()
    
    # Get nodes
    print("="*70)
    print("  Nodes (Extracted Entities)")
    print("="*70 + "\n")
    
    nodes = graph.get_nodes(graph_id)
    print(f"Found {len(nodes)} nodes:")
    for node in nodes:
        print(f"  - {node.get('name')} (type: {node.get('type')})")
    print()
    
    # Create some edges
    if len(nodes) >= 2:
        print("="*70)
        print("  Creating Edges")
        print("="*70 + "\n")
        
        # Create relationship between first two nodes
        edge_id = graph.create_edge(
            graph_id,
            nodes[0]["node_id"],
            nodes[1]["node_id"],
            "RELATED_TO",
            {"strength": "high"}
        )
        print(f"✓ Created edge: {nodes[0].get('name')} -> {nodes[1].get('name')}\n")
    
    # Get edges
    print("="*70)
    print("  Edges (Relationships)")
    print("="*70 + "\n")
    
    edges = graph.get_edges(graph_id)
    print(f"Found {len(edges)} edges:")
    for edge in edges:
        source = edge.get("source", "?")
        target = edge.get("target", "?")
        edge_type = edge.get("type", "?")
        print(f"  - {source} --[{edge_type}]--> {target}")
    print()
    
    # Search
    print("="*70)
    print("  Search")
    print("="*70 + "\n")
    
    query = "Eric Clapton"
    print(f"Searching for: '{query}'")
    results = graph.search(graph_id, query)
    print(f"✓ Found {len(results)} results:")
    for result in results:
        if result["type"] == "episode":
            print(f"  - Episode: {result.get('data')[:50]}...")
        else:
            print(f"  - Node: {result.get('name')} ({result.get('node_type')})")
    
    # Cleanup
    graph.close()
    print("\n✅ Example complete!")


if __name__ == "__main__":
    main()
