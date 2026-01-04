import json
import asyncio
from typing import Any, AsyncIterator, Dict, Optional, Sequence, Tuple
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver, Checkpoint, CheckpointMetadata, CheckpointTuple, JsonPlusSerializer
from toondb import Database
from .config import get_toondb_config

class ToonDBCheckpointer(BaseCheckpointSaver):
    """
    ToonDB-backed checkpointer for LangGraph.
    
    Persists graph state to ToonDB using key-value storage.
    Keys are structured as: `checkpoints/{thread_id}/{checkpoint_id}`
    """
    
    def __init__(self, db_path: str = None):
        super().__init__()
        self.config = get_toondb_config()
        self.db_path = db_path or self.config.db_path
        self._db = None
        self.serde = JsonPlusSerializer()

    @property
    def db(self) -> Database:
        if self._db is None:
            self._db = Database.open(self.db_path)
        return self._db

    def get_tuple(self, config: RunnableConfig) -> Optional[CheckpointTuple]:
        """Get a checkpoint tuple from the database."""
        thread_id = config["configurable"]["thread_id"]
        checkpoint_id = config["configurable"].get("checkpoint_id")
        
        if checkpoint_id:
            key = f"checkpoints/{thread_id}/{checkpoint_id}"
            data = self._get_data(key)
        else:
            # Find the latest checkpoint for this thread
            prefix = f"checkpoints/{thread_id}/"
            # ToonDB scan returns keys in lexicographical order. 
            # We assume checkpoint IDs are sortable (e.g. UUIDs or timestamps).
            # We need to find the *last* key for this prefix.
            # Since standard scan is forward, we might need to seek or scan all.
            # Optimization: Use reverse scan if available, or just scan all (ok for demo).
            latest_key = None
            try:
                for kv in self.db.scan_prefix(prefix.encode()):
                    latest_key = kv.key.decode()
            except Exception:
                pass
                
            if latest_key:
                data = self._get_data(latest_key)
            else:
                return None

        if not data:
            return None

        checkpoint = self.serde.loads(data["checkpoint"])
        metadata = self.serde.loads(data["metadata"])
        parent_config = data.get("parent_config")
        
        return CheckpointTuple(
            config=config,
            checkpoint=checkpoint,
            metadata=metadata,
            parent_config=parent_config,
        )

    def list(
        self,
        config: Optional[RunnableConfig],
        *,
        filter: Optional[Dict[str, Any]] = None,
        before: Optional[RunnableConfig] = None,
        limit: Optional[int] = None,
    ) -> AsyncIterator[CheckpointTuple]:
        """List checkpoints matching the criteria."""
        # Simplified implementation for demo - lists all for thread
        if not config:
            return

        thread_id = config["configurable"]["thread_id"]
        prefix = f"checkpoints/{thread_id}/"
        
        count = 0
        # Iterate in reverse would be better, but scan is forward.
        # Collecting all and reversing for "latest first" semantics usually expected
        found_checkpoints = []
        
        try:
            for kv in self.db.scan_prefix(prefix.encode()):
                key = kv.key.decode()
                data = self._get_data(key)
                if data:
                    checkpoint = self.serde.loads(data["checkpoint"])
                    metadata = self.serde.loads(data["metadata"])
                    parent_config = data.get("parent_config")
                    found_checkpoints.append(CheckpointTuple(
                        config={"configurable": {"thread_id": thread_id, "checkpoint_id": checkpoint["id"]}},
                        checkpoint=checkpoint,
                        metadata=metadata,
                        parent_config=parent_config,
                    ))
        except Exception:
            pass
            
        # Reverse to get newest first
        for cp in reversed(found_checkpoints):
            if limit and count >= limit:
                break
            yield cp
            count += 1

    def put(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: Dict[str, Any],
    ) -> RunnableConfig:
        """Save a checkpoint to the database."""
        thread_id = config["configurable"]["thread_id"]
        checkpoint_id = checkpoint["id"]
        
        key = f"checkpoints/{thread_id}/{checkpoint_id}"
        
        data = {
            "checkpoint": self.serde.dumps(checkpoint),
            "metadata": self.serde.dumps(metadata),
            "parent_config": config.get("configurable", {}).get("thread_ts") # approximate parent tracking
        }
        
        self.db.put(key.encode(), json.dumps(data).encode())
        
        return {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_id": checkpoint_id,
            }
        }
        
    def _get_data(self, key: str) -> Optional[Dict]:
        try:
            val = self.db.get(key.encode())
            if val:
                return json.loads(val.decode())
        except Exception:
            pass
        return None

    def close(self):
        if self._db:
            self._db.close()
