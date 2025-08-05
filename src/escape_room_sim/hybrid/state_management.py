"""
Mesa-CrewAI Hybrid State Management System

This module implements comprehensive state management for the hybrid architecture,
ensuring consistent state synchronization between Mesa's spatial/temporal model
and CrewAI's reasoning and memory systems.

Key Responsibilities:
- Unified state representation across frameworks
- Conflict resolution for concurrent state changes
- State versioning and rollback capabilities
- Performance-optimized state synchronization
"""

from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from abc import ABC, abstractmethod
import threading
import copy
import mesa
from crewai import Agent
from unittest.mock import Mock
from .core_architecture import ComponentState, HybridAgent, DecisionData, IStateSynchronizer


class StateType(Enum):
    """Types of state managed by the system"""
    SPATIAL = "spatial"           # Agent positions, environment layout
    TEMPORAL = "temporal"         # Time, scheduling, sequences
    RESOURCE = "resource"         # Resource ownership, availability
    SOCIAL = "social"            # Relationships, trust, communication
    COGNITIVE = "cognitive"      # Agent memories, knowledge, beliefs
    ENVIRONMENTAL = "environmental"  # Room state, hazards, conditions


class StateChangeType(Enum):
    """Types of state changes"""
    CREATE = "create"
    UPDATE = "update"
    DELETE = "delete"
    MOVE = "move"
    TRANSFER = "transfer"


@dataclass
class StateChange:
    """Represents a change to system state"""
    change_id: str
    timestamp: datetime
    change_type: StateChangeType
    state_type: StateType
    entity_id: str
    old_value: Any
    new_value: Any
    source: str  # 'mesa' or 'crewai'
    validated: bool = False
    applied: bool = False


@dataclass
class StateSnapshot:
    """Complete state snapshot for rollback capabilities"""
    snapshot_id: str
    timestamp: datetime
    mesa_state: Dict[str, Any]
    crewai_state: Dict[str, Any]
    unified_state: Dict[str, Any]
    version: int


class IStateValidator(ABC):
    """Interface for validating state changes"""
    
    @abstractmethod
    def validate_change(self, change: StateChange, current_state: Dict[str, Any]) -> bool:
        """Validate if a state change is legal"""
        pass
    
    @abstractmethod
    def resolve_conflict(self, changes: List[StateChange]) -> List[StateChange]:
        """Resolve conflicting state changes"""
        pass


class IStateSerializer(ABC):
    """Interface for serializing/deserializing state"""
    
    @abstractmethod
    def serialize_state(self, state: Dict[str, Any]) -> str:
        """Serialize state for storage"""
        pass
    
    @abstractmethod
    def deserialize_state(self, serialized: str) -> Dict[str, Any]:
        """Deserialize state from storage"""
        pass


class UnifiedStateManager:
    """
    Central state manager that maintains consistency between Mesa and CrewAI
    
    Architecture Decision: Single source of truth with synchronized replicas
    - Unified state is the authoritative source
    - Mesa and CrewAI maintain local replicas for performance
    - Changes flow through central manager for consistency
    - State versioning enables rollback and debugging
    """
    
    def __init__(self, validator: IStateValidator, serializer: IStateSerializer):
        self.validator = validator
        self.serializer = serializer
        
        # Core state storage
        self.unified_state: Dict[str, Any] = {}
        self.mesa_state: Dict[str, Any] = {}
        self.crewai_state: Dict[str, Any] = {}
        
        # Change tracking
        self.pending_changes: List[StateChange] = []
        self.applied_changes: List[StateChange] = []
        self.change_listeners: Dict[StateType, List[Callable]] = {}
        
        # State versioning
        self.current_version = 0
        self.snapshots: Dict[int, StateSnapshot] = {}
        self.max_snapshots = 10
        
        # Synchronization
        self._lock = threading.RLock()
        self._sync_required = {"mesa": False, "crewai": False}
        
        # Performance tracking
        self.sync_metrics = {
            "sync_count": 0,
            "avg_sync_time": 0.0,
            "conflict_count": 0,
            "rollback_count": 0
        }
    
    def initialize_state(self, mesa_model: mesa.Model, crewai_agents: List[Agent]) -> None:
        """Initialize unified state from both frameworks"""
        with self._lock:
            # Extract initial state from Mesa
            self.mesa_state = self._extract_mesa_state(mesa_model)
            
            # Extract initial state from CrewAI
            self.crewai_state = self._extract_crewai_state(crewai_agents)
            
            # Create unified state
            self.unified_state = self._merge_states(self.mesa_state, self.crewai_state)
            
            # Create initial snapshot
            self._create_snapshot("initial")
    
    def register_state_change(self, change: StateChange) -> bool:
        """Register a state change for processing"""
        with self._lock:
            # Validate change
            if not self.validator.validate_change(change, self.unified_state):
                return False
            
            change.validated = True
            self.pending_changes.append(change)
            
            return True
    
    def apply_pending_changes(self) -> List[StateChange]:
        """Apply all pending state changes"""
        with self._lock:
            if not self.pending_changes:
                return []
            
            # Resolve conflicts
            resolved_changes = self.validator.resolve_conflict(self.pending_changes)
            
            # Apply changes
            applied = []
            for change in resolved_changes:
                if self._apply_single_change(change):
                    applied.append(change)
            
            # Clear pending changes
            self.pending_changes.clear()
            
            # Mark synchronization required
            self._sync_required["mesa"] = True
            self._sync_required["crewai"] = True
            
            # Increment version
            self.current_version += 1
            
            return applied
    
    def synchronize_to_mesa(self, mesa_model: mesa.Model) -> None:
        """Synchronize unified state to Mesa model"""
        if not self._sync_required["mesa"]:
            return
        
        start_time = datetime.now()
        
        with self._lock:
            # Update Mesa agents
            self._sync_spatial_state_to_mesa(mesa_model)
            self._sync_resource_state_to_mesa(mesa_model)
            self._sync_environmental_state_to_mesa(mesa_model)
            
            self._sync_required["mesa"] = False
            
            # Update metrics
            sync_time = (datetime.now() - start_time).total_seconds()
            self._update_sync_metrics(sync_time)
    
    def synchronize_to_crewai(self, crewai_agents: List[Agent]) -> None:
        """Synchronize unified state to CrewAI agents"""
        if not self._sync_required["crewai"]:
            return
        
        start_time = datetime.now()
        
        with self._lock:
            # Update CrewAI agent memories
            self._sync_cognitive_state_to_crewai(crewai_agents)
            self._sync_social_state_to_crewai(crewai_agents)
            
            self._sync_required["crewai"] = False
            
            # Update metrics
            sync_time = (datetime.now() - start_time).total_seconds()
            self._update_sync_metrics(sync_time)
    
    def get_unified_state(self) -> Dict[str, Any]:
        """Get current unified state"""
        with self._lock:
            return copy.deepcopy(self.unified_state)
    
    def get_agent_state(self, agent_id: str) -> Dict[str, Any]:
        """Get state for specific agent"""
        with self._lock:
            agents = self.unified_state.get("agents", {})
            return copy.deepcopy(agents.get(agent_id, {}))
    
    def create_snapshot(self, label: str = None) -> str:
        """Create state snapshot for rollback"""
        with self._lock:
            snapshot_id = label or f"snapshot_{self.current_version}_{datetime.now().timestamp()}"
            
            snapshot = StateSnapshot(
                snapshot_id=snapshot_id,
                timestamp=datetime.now(),
                mesa_state=copy.deepcopy(self.mesa_state),
                crewai_state=copy.deepcopy(self.crewai_state),
                unified_state=copy.deepcopy(self.unified_state),
                version=self.current_version
            )
            
            self.snapshots[self.current_version] = snapshot
            
            # Maintain snapshot limit
            if len(self.snapshots) > self.max_snapshots:
                oldest_version = min(self.snapshots.keys())
                del self.snapshots[oldest_version]
            
            return snapshot_id
    
    def rollback_to_snapshot(self, version: int) -> bool:
        """Rollback to a previous state snapshot"""
        with self._lock:
            if version not in self.snapshots:
                return False
            
            snapshot = self.snapshots[version]
            
            # Restore state
            self.mesa_state = copy.deepcopy(snapshot.mesa_state)
            self.crewai_state = copy.deepcopy(snapshot.crewai_state)
            self.unified_state = copy.deepcopy(snapshot.unified_state)
            self.current_version = snapshot.version
            
            # Clear pending changes
            self.pending_changes.clear()
            
            # Mark sync required
            self._sync_required["mesa"] = True
            self._sync_required["crewai"] = True
            
            # Update metrics
            self.sync_metrics["rollback_count"] += 1
            
            return True
    
    def add_change_listener(self, state_type: StateType, callback: Callable) -> None:
        """Add listener for state changes"""
        if state_type not in self.change_listeners:
            self.change_listeners[state_type] = []
        self.change_listeners[state_type].append(callback)
    
    def remove_change_listener(self, state_type: StateType, callback: Callable) -> None:
        """Remove state change listener"""
        if state_type in self.change_listeners:
            self.change_listeners[state_type] = [
                cb for cb in self.change_listeners[state_type] if cb != callback
            ]
    
    # Private methods for state extraction and synchronization
    
    def _extract_mesa_state(self, mesa_model: mesa.Model) -> Dict[str, Any]:
        """Extract current state from Mesa model"""
        state = {
            "model": {
                "step_count": getattr(mesa_model, 'schedule', mesa.time.RandomActivation(mesa_model)).steps,
                "running": getattr(mesa_model, 'running', True),
                "time_remaining": getattr(mesa_model, 'time_remaining', None)
            },
            "agents": {},
            "environment": {},
            "resources": {}
        }
        
        # Extract agent states
        if hasattr(mesa_model, 'schedule'):
            for agent in mesa_model.schedule.agents:
                agent_id = self._get_mesa_agent_id(agent)
                state["agents"][agent_id] = {
                    "position": getattr(agent, 'pos', None),
                    "health": getattr(agent, 'health', 1.0),
                    "energy": getattr(agent, 'energy', 1.0),
                    "resources": getattr(agent, 'resources', []),
                    "status": getattr(agent, 'status', 'active')
                }
        
        # Extract environment state
        if hasattr(mesa_model, 'grid'):
            state["environment"]["bounds"] = {
                "width": mesa_model.grid.width,
                "height": mesa_model.grid.height
            }
        
        # Extract resource state
        if hasattr(mesa_model, 'resource_manager'):
            state["resources"] = mesa_model.resource_manager.get_state_summary()
        
        return state
    
    def _extract_crewai_state(self, crewai_agents: List[Agent]) -> Dict[str, Any]:
        """Extract current state from CrewAI agents"""
        state = {
            "agents": {},
            "memories": {},
            "decisions": {}
        }
        
        for agent in crewai_agents:
            agent_id = agent.role.lower().replace(" ", "_")
            
            state["agents"][agent_id] = {
                "role": agent.role,
                "goal": agent.goal,
                "backstory": agent.backstory,
                "memory_enabled": getattr(agent, 'memory', False)
            }
            
            # Extract memory if available
            if hasattr(agent, 'memory') and agent.memory:
                state["memories"][agent_id] = self._extract_agent_memory(agent)
        
        return state
    
    def _extract_agent_memory(self, agent: Agent) -> Dict[str, Any]:
        """Extract memory from CrewAI agent"""
        memory_data = {}
        
        # This would depend on CrewAI's memory implementation
        if hasattr(agent, 'memory') and agent.memory:
            # Try to extract structured memory data
            if hasattr(agent.memory, 'storage'):
                memory_data["storage"] = agent.memory.storage
            if hasattr(agent.memory, 'short_term'):
                memory_data["short_term"] = agent.memory.short_term
            if hasattr(agent.memory, 'long_term'):
                memory_data["long_term"] = agent.memory.long_term
        
        return memory_data
    
    def _merge_states(self, mesa_state: Dict[str, Any], crewai_state: Dict[str, Any]) -> Dict[str, Any]:
        """Merge Mesa and CrewAI states into unified representation"""
        unified = {
            "version": self.current_version,
            "timestamp": datetime.now(),
            "agents": {},
            "environment": mesa_state.get("environment", {}),
            "resources": mesa_state.get("resources", {}),
            "model": mesa_state.get("model", {}),
            "memories": crewai_state.get("memories", {}),
            "social": {
                "trust_relationships": {},
                "communications": []
            }
        }
        
        # Merge agent data
        mesa_agents = mesa_state.get("agents", {})
        crewai_agents = crewai_state.get("agents", {})
        
        all_agent_ids = set(mesa_agents.keys()) | set(crewai_agents.keys())
        
        for agent_id in all_agent_ids:
            unified["agents"][agent_id] = {
                # Mesa data
                **mesa_agents.get(agent_id, {}),
                # CrewAI data
                **crewai_agents.get(agent_id, {}),
                # Unified fields
                "last_updated": datetime.now(),
                "state_source": "unified"
            }
        
        return unified
    
    def _apply_single_change(self, change: StateChange) -> bool:
        """Apply a single state change to unified state"""
        try:
            # Navigate to the correct state location
            state_path = self._get_state_path(change.state_type, change.entity_id)
            
            # Apply change based on type
            if change.change_type == StateChangeType.CREATE:
                self._set_nested_value(self.unified_state, state_path, change.new_value)
            elif change.change_type == StateChangeType.UPDATE:
                self._set_nested_value(self.unified_state, state_path, change.new_value)
            elif change.change_type == StateChangeType.DELETE:
                self._delete_nested_value(self.unified_state, state_path)
            elif change.change_type == StateChangeType.MOVE:
                # Handle position changes
                self._handle_move_change(change)
            elif change.change_type == StateChangeType.TRANSFER:
                # Handle resource transfers
                self._handle_transfer_change(change)
            
            change.applied = True
            self.applied_changes.append(change)
            
            # Notify listeners
            self._notify_change_listeners(change)
            
            return True
            
        except Exception as e:
            print(f"Failed to apply state change {change.change_id}: {e}")
            return False
    
    def _get_state_path(self, state_type: StateType, entity_id: str) -> List[str]:
        """Get path to state location in unified state"""
        if state_type == StateType.SPATIAL:
            return ["agents", entity_id, "position"]
        elif state_type == StateType.RESOURCE:
            return ["agents", entity_id, "resources"]
        elif state_type == StateType.SOCIAL:
            return ["social", "trust_relationships", entity_id]
        elif state_type == StateType.COGNITIVE:
            return ["memories", entity_id]
        elif state_type == StateType.ENVIRONMENTAL:
            return ["environment", entity_id]
        elif state_type == StateType.TEMPORAL:
            return ["model", entity_id]
        else:
            return ["agents", entity_id]
    
    def _set_nested_value(self, state: Dict[str, Any], path: List[str], value: Any) -> None:
        """Set value at nested path in state dictionary"""
        current = state
        for key in path[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        current[path[-1]] = value
    
    def _delete_nested_value(self, state: Dict[str, Any], path: List[str]) -> None:
        """Delete value at nested path in state dictionary"""
        current = state
        for key in path[:-1]:
            if key not in current:
                return
            current = current[key]
        if path[-1] in current:
            del current[path[-1]]
    
    def _handle_move_change(self, change: StateChange) -> None:
        """Handle agent movement changes"""
        agent_id = change.entity_id
        new_position = change.new_value
        
        # Update agent position
        if "agents" not in self.unified_state:
            self.unified_state["agents"] = {}
        if agent_id not in self.unified_state["agents"]:
            self.unified_state["agents"][agent_id] = {}
        
        self.unified_state["agents"][agent_id]["position"] = new_position
        self.unified_state["agents"][agent_id]["last_moved"] = datetime.now()
    
    def _handle_transfer_change(self, change: StateChange) -> None:
        """Handle resource transfer changes"""
        # Extract transfer details from change
        transfer_data = change.new_value
        from_agent = transfer_data.get("from_agent")
        to_agent = transfer_data.get("to_agent")
        resource_id = transfer_data.get("resource_id")
        
        if from_agent and to_agent and resource_id:
            # Remove from source agent
            from_resources = self.unified_state["agents"][from_agent].get("resources", [])
            if resource_id in from_resources:
                from_resources.remove(resource_id)
            
            # Add to target agent
            to_resources = self.unified_state["agents"][to_agent].get("resources", [])
            if resource_id not in to_resources:
                to_resources.append(resource_id)
    
    def _notify_change_listeners(self, change: StateChange) -> None:
        """Notify registered listeners of state changes"""
        listeners = self.change_listeners.get(change.state_type, [])
        for callback in listeners:
            try:
                callback(change)
            except Exception as e:
                print(f"State change listener failed: {e}")
    
    def _sync_spatial_state_to_mesa(self, mesa_model: mesa.Model) -> None:
        """Synchronize spatial state to Mesa model"""
        if not hasattr(mesa_model, 'schedule'):
            return
        
        unified_agents = self.unified_state.get("agents", {})
        
        for mesa_agent in mesa_model.schedule.agents:
            agent_id = self._get_mesa_agent_id(mesa_agent)
            if agent_id in unified_agents:
                unified_agent = unified_agents[agent_id]
                
                # Update position
                new_pos = unified_agent.get("position")
                if new_pos and hasattr(mesa_agent, 'pos') and mesa_agent.pos != new_pos:
                    if hasattr(mesa_model, 'grid'):
                        mesa_model.grid.move_agent(mesa_agent, new_pos)
                    else:
                        mesa_agent.pos = new_pos
                
                # Update other attributes
                if "health" in unified_agent:
                    mesa_agent.health = unified_agent["health"]
                if "energy" in unified_agent:
                    mesa_agent.energy = unified_agent["energy"]
    
    def _sync_resource_state_to_mesa(self, mesa_model: mesa.Model) -> None:
        """Synchronize resource state to Mesa model"""
        if not hasattr(mesa_model, 'resource_manager'):
            return
        
        unified_resources = self.unified_state.get("resources", {})
        mesa_model.resource_manager.sync_from_unified_state(unified_resources)
    
    def _sync_environmental_state_to_mesa(self, mesa_model: mesa.Model) -> None:
        """Synchronize environmental state to Mesa model"""
        unified_env = self.unified_state.get("environment", {})
        
        # Update model-level environmental properties
        for key, value in unified_env.items():
            if hasattr(mesa_model, key):
                setattr(mesa_model, key, value)
    
    def _sync_cognitive_state_to_crewai(self, crewai_agents: List[Agent]) -> None:
        """Synchronize cognitive state to CrewAI agents"""
        unified_memories = self.unified_state.get("memories", {})
        
        for agent in crewai_agents:
            agent_id = agent.role.lower().replace(" ", "_")
            if agent_id in unified_memories and hasattr(agent, 'memory'):
                # Update agent memory
                self._update_agent_memory(agent, unified_memories[agent_id])
    
    def _sync_social_state_to_crewai(self, crewai_agents: List[Agent]) -> None:
        """Synchronize social state to CrewAI agents"""
        unified_social = self.unified_state.get("social", {})
        
        # This would update trust relationships and communication history
        # in the agents' memory systems
        pass
    
    def _update_agent_memory(self, agent: Agent, memory_data: Dict[str, Any]) -> None:
        """Update CrewAI agent memory with unified state data"""
        if not hasattr(agent, 'memory') or not agent.memory:
            return
        
        # This would depend on CrewAI's memory API
        # Update different memory components based on available data
        pass
    
    def _get_mesa_agent_id(self, mesa_agent: mesa.Agent) -> str:
        """Get agent ID from Mesa agent"""
        if hasattr(mesa_agent, 'agent_id'):
            return mesa_agent.agent_id
        elif hasattr(mesa_agent, 'unique_id'):
            return str(mesa_agent.unique_id)
        else:
            return f"agent_{id(mesa_agent)}"
    
    def _create_snapshot(self, label: str) -> None:
        """Create initial state snapshot"""
        snapshot = StateSnapshot(
            snapshot_id=label,
            timestamp=datetime.now(),
            mesa_state=copy.deepcopy(self.mesa_state),
            crewai_state=copy.deepcopy(self.crewai_state),
            unified_state=copy.deepcopy(self.unified_state),
            version=self.current_version
        )
        
        self.snapshots[self.current_version] = snapshot
    
    def _update_sync_metrics(self, sync_time: float) -> None:
        """Update synchronization performance metrics"""
        self.sync_metrics["sync_count"] += 1
        current_avg = self.sync_metrics["avg_sync_time"]
        count = self.sync_metrics["sync_count"]
        
        # Update running average
        self.sync_metrics["avg_sync_time"] = ((current_avg * (count - 1)) + sync_time) / count
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get state management performance metrics"""
        with self._lock:
            return {
                "sync_metrics": self.sync_metrics.copy(),
                "state_size": {
                    "unified": len(str(self.unified_state)),
                    "mesa": len(str(self.mesa_state)),
                    "crewai": len(str(self.crewai_state))
                },
                "change_tracking": {
                    "pending_changes": len(self.pending_changes),
                    "applied_changes": len(self.applied_changes),
                    "snapshots": len(self.snapshots)
                },
                "current_version": self.current_version
            }


class DefaultStateValidator(IStateValidator):
    """Default implementation of state validator"""
    
    def validate_change(self, change: StateChange, current_state: Dict[str, Any]) -> bool:
        """Validate if a state change is legal"""
        
        # Basic validation rules
        if not change.entity_id or not change.entity_id.strip():
            return False
        
        # State-type specific validation
        if change.state_type == StateType.SPATIAL:
            return self._validate_spatial_change(change, current_state)
        elif change.state_type == StateType.RESOURCE:
            return self._validate_resource_change(change, current_state)
        elif change.state_type == StateType.SOCIAL:
            return self._validate_social_change(change, current_state)
        
        return True
    
    def resolve_conflict(self, changes: List[StateChange]) -> List[StateChange]:
        """Resolve conflicting state changes"""
        if len(changes) <= 1:
            return changes
        
        # Group changes by entity and state type
        change_groups = {}
        for change in changes:
            key = (change.entity_id, change.state_type)
            if key not in change_groups:
                change_groups[key] = []
            change_groups[key].append(change)
        
        resolved = []
        
        for group in change_groups.values():
            if len(group) == 1:
                resolved.extend(group)
            else:
                # Resolve conflicts within group
                resolved.append(self._resolve_group_conflict(group))
        
        return resolved
    
    def _validate_spatial_change(self, change: StateChange, current_state: Dict[str, Any]) -> bool:
        """Validate spatial state changes"""
        if change.change_type == StateChangeType.MOVE:
            new_pos = change.new_value
            if not isinstance(new_pos, (list, tuple)) or len(new_pos) != 2:
                return False
            
            # Check if position is within bounds
            env = current_state.get("environment", {})
            bounds = env.get("bounds", {})
            if bounds:
                width = bounds.get("width", 10)
                height = bounds.get("height", 10)
                if not (0 <= new_pos[0] < width and 0 <= new_pos[1] < height):
                    return False
        
        return True
    
    def _validate_resource_change(self, change: StateChange, current_state: Dict[str, Any]) -> bool:
        """Validate resource state changes"""
        if change.change_type == StateChangeType.TRANSFER:
            transfer_data = change.new_value
            if not isinstance(transfer_data, dict):
                return False
            
            required_keys = ["from_agent", "to_agent", "resource_id"]
            if not all(key in transfer_data for key in required_keys):
                return False
            
            # Check if source agent has the resource
            from_agent = transfer_data["from_agent"]
            resource_id = transfer_data["resource_id"]
            
            agents = current_state.get("agents", {})
            if from_agent in agents:
                agent_resources = agents[from_agent].get("resources", [])
                if resource_id not in agent_resources:
                    return False
        
        return True
    
    def _validate_social_change(self, change: StateChange, current_state: Dict[str, Any]) -> bool:
        """Validate social state changes"""
        # Social changes are generally permissive
        return True
    
    def _resolve_group_conflict(self, changes: List[StateChange]) -> StateChange:
        """Resolve conflicts within a group of changes"""
        # Simple resolution: use the most recent change
        return max(changes, key=lambda c: c.timestamp)


class JSONStateSerializer(IStateSerializer):
    """JSON-based state serializer"""
    
    def serialize_state(self, state: Dict[str, Any]) -> str:
        """Serialize state to JSON string"""
        import json
        
        # Convert datetime objects to strings
        serializable_state = self._make_serializable(state)
        return json.dumps(serializable_state, indent=2)
    
    def deserialize_state(self, serialized: str) -> Dict[str, Any]:
        """Deserialize state from JSON string"""
        import json
        
        state = json.loads(serialized)
        return self._restore_types(state)
    
    def _make_serializable(self, obj: Any) -> Any:
        """Convert objects to JSON-serializable format"""
        if isinstance(obj, datetime):
            return obj.isoformat()
        elif isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        else:
            return obj
    
    def _restore_types(self, obj: Any) -> Any:
        """Restore types from serialized format"""
        if isinstance(obj, str) and self._is_iso_datetime(obj):
            return datetime.fromisoformat(obj)
        elif isinstance(obj, dict):
            return {k: self._restore_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._restore_types(item) for item in obj]
        else:
            return obj
    
    def _is_iso_datetime(self, s: str) -> bool:
        """Check if string is ISO datetime format"""
        try:
            datetime.fromisoformat(s)
            return True
        except ValueError:
            return False


# Concrete Escape Room Implementation

class EscapeRoomStateSynchronizer(IStateSynchronizer):
    """
    Escape room specific state synchronizer implementation
    
    Handles synchronization of escape room specific state between
    Mesa and CrewAI frameworks.
    """
    
    def __init__(self, room_config: Dict[str, Any] = None):
        self.room_config = room_config or {}
        # Create simple validator and serializer for testing
        validator = DefaultStateValidator()
        serializer = JSONStateSerializer()
        self.state_manager = UnifiedStateManager(validator, serializer)
        self.last_sync_timestamp = datetime.now()
        self.sync_history: List[Dict[str, Any]] = []
    
    def sync_mesa_to_crewai(self, mesa_model: mesa.Model) -> None:
        """Sync Mesa state changes to CrewAI agents"""
        try:
            # Extract current Mesa state
            mesa_state = self._extract_mesa_state(mesa_model)
            
            # Update CrewAI agent memories with Mesa state
            self._update_crewai_memories(mesa_state, mesa_model)
            
            # Record sync operation
            self._record_sync_operation("mesa_to_crewai", mesa_state)
            
        except Exception as e:
            print(f"Error in Mesa to CrewAI sync: {e}")
    
    def sync_crewai_to_mesa(self, decisions: Dict[str, DecisionData], 
                          mesa_model: mesa.Model) -> None:
        """Sync CrewAI decisions to Mesa model"""
        try:
            # Apply decisions to Mesa model state
            self._apply_decisions_to_mesa(decisions, mesa_model)
            
            # Update unified state
            unified_state = self._create_unified_state(decisions, mesa_model)
            self.state_manager.update_state(unified_state)
            
            # Record sync operation
            self._record_sync_operation("crewai_to_mesa", decisions)
            
        except Exception as e:
            print(f"Error in CrewAI to Mesa sync: {e}")
    
    def _extract_mesa_state(self, mesa_model: mesa.Model) -> Dict[str, Any]:
        """Extract relevant state from Mesa model"""
        state = {
            'timestamp': datetime.now(),
            'model_step': getattr(mesa_model, 'schedule', Mock()).steps,
            'agents': {},
            'environment': {},
            'resources': {}
        }
        
        # Extract agent states
        if hasattr(mesa_model, 'schedule') and hasattr(mesa_model.schedule, 'agents'):
            for agent in mesa_model.schedule.agents:
                agent_id = getattr(agent, 'agent_id', str(agent.unique_id))
                state['agents'][agent_id] = {
                    'position': getattr(agent, 'pos', None),
                    'health': getattr(agent, 'health', 100),
                    'energy': getattr(agent, 'energy', 1.0),
                    'resources': getattr(agent, 'resources', []),
                    'state': getattr(agent, 'state', {}),
                    'last_action': getattr(agent, 'last_action', None)
                }
        
        # Extract environment state
        state['environment'] = {
            'width': getattr(mesa_model, 'width', 10),
            'height': getattr(mesa_model, 'height', 10),
            'room_objects': getattr(mesa_model, 'room_objects', {}),
            'hazards': getattr(mesa_model, 'hazards', []),
            'temperature': getattr(mesa_model, 'temperature', 20.0),
            'lighting': getattr(mesa_model, 'lighting', 1.0),
            'time_remaining': getattr(mesa_model, 'time_remaining', None)
        }
        
        # Extract resource state
        state['resources'] = getattr(mesa_model, 'global_resources', {})
        
        return state
    
    def _update_crewai_memories(self, mesa_state: Dict[str, Any], mesa_model: mesa.Model) -> None:
        """Update CrewAI agent memories with Mesa state information"""
        for agent_id, agent_state in mesa_state['agents'].items():
            # Create memory entry for agent
            memory_entry = {
                'type': 'environmental_update',
                'timestamp': datetime.now(),
                'position': agent_state['position'],
                'health': agent_state['health'],
                'energy': agent_state['energy'],
                'environment': mesa_state['environment'],
                'nearby_agents': self._find_nearby_agents(agent_id, mesa_state['agents']),
                'available_resources': self._find_available_resources(agent_state['position'], mesa_state)
            }
            
            # Store in state manager (would integrate with actual CrewAI memory in full implementation)
            self.state_manager.record_state_change(StateChange(
                change_id=f"memory_update_{agent_id}_{datetime.now().timestamp()}",
                timestamp=datetime.now(),
                state_type=StateType.COGNITIVE,
                change_type=StateChangeType.UPDATE,
                affected_entities=[agent_id],
                previous_state=None,
                new_state=memory_entry,
                metadata={'sync_source': 'mesa_to_crewai'}
            ))
    
    def _apply_decisions_to_mesa(self, decisions: Dict[str, DecisionData], mesa_model: mesa.Model) -> None:
        """Apply CrewAI decisions to Mesa model state"""
        for agent_id, decision in decisions.items():
            # Find Mesa agent
            mesa_agent = self._find_mesa_agent(agent_id, mesa_model)
            if not mesa_agent:
                continue
            
            # Apply decision based on action type
            if decision.chosen_action == 'move':
                self._apply_move_decision(mesa_agent, decision, mesa_model)
            elif decision.chosen_action == 'communicate':
                self._apply_communication_decision(mesa_agent, decision, mesa_model)
            elif decision.chosen_action in ['examine', 'analyze']:
                self._apply_examination_decision(mesa_agent, decision, mesa_model)
            else:
                # Store decision in agent state for other actions
                if not hasattr(mesa_agent, 'pending_actions'):
                    mesa_agent.pending_actions = []
                mesa_agent.pending_actions.append({
                    'action': decision.chosen_action,
                    'parameters': decision.action_parameters,
                    'timestamp': decision.timestamp
                })
    
    def _apply_move_decision(self, mesa_agent, decision: DecisionData, mesa_model: mesa.Model) -> None:
        """Apply movement decision to Mesa agent"""
        target_pos = decision.action_parameters.get('target_position')
        if target_pos and hasattr(mesa_agent, 'pos'):
            # Validate movement (basic bounds checking)
            width = getattr(mesa_model, 'width', 10)
            height = getattr(mesa_model, 'height', 10)
            
            if 0 <= target_pos[0] < width and 0 <= target_pos[1] < height:
                mesa_agent.pos = target_pos
                mesa_agent.last_action = f"moved_to_{target_pos}"
    
    def _apply_communication_decision(self, mesa_agent, decision: DecisionData, mesa_model: mesa.Model) -> None:
        """Apply communication decision to Mesa agent"""
        target = decision.action_parameters.get('target')
        message = decision.action_parameters.get('message', 'hello')
        
        # Store communication in model (would integrate with actual communication system)
        if not hasattr(mesa_model, 'communications'):
            mesa_model.communications = []
        
        mesa_model.communications.append({
            'sender': getattr(mesa_agent, 'agent_id', mesa_agent.unique_id),
            'target': target,
            'message': message,
            'timestamp': datetime.now()
        })
        
        mesa_agent.last_action = f"communicated_with_{target}"
    
    def _apply_examination_decision(self, mesa_agent, decision: DecisionData, mesa_model: mesa.Model) -> None:
        """Apply examination decision to Mesa agent"""
        target = decision.action_parameters.get('target', 'environment')
        detail_level = decision.action_parameters.get('detail_level', 'medium')
        
        # Store examination results in agent state
        if not hasattr(mesa_agent, 'examination_results'):
            mesa_agent.examination_results = []
        
        mesa_agent.examination_results.append({
            'target': target,
            'detail_level': detail_level,
            'timestamp': datetime.now(),
            'results': f"examined_{target}_at_{detail_level}_detail"
        })
        
        mesa_agent.last_action = f"examined_{target}"
    
    def _find_mesa_agent(self, agent_id: str, mesa_model: mesa.Model):
        """Find Mesa agent by ID"""
        if hasattr(mesa_model, 'schedule') and hasattr(mesa_model.schedule, 'agents'):
            for agent in mesa_model.schedule.agents:
                if (hasattr(agent, 'agent_id') and agent.agent_id == agent_id) or \
                   str(agent.unique_id) == agent_id:
                    return agent
        return None
    
    def _find_nearby_agents(self, agent_id: str, agents_state: Dict[str, Any]) -> List[str]:
        """Find agents near the specified agent"""
        nearby = []
        agent_pos = agents_state.get(agent_id, {}).get('position')
        
        if not agent_pos:
            return nearby
        
        for other_id, other_state in agents_state.items():
            if other_id == agent_id:
                continue
            
            other_pos = other_state.get('position')
            if other_pos:
                distance = ((agent_pos[0] - other_pos[0]) ** 2 + 
                          (agent_pos[1] - other_pos[1]) ** 2) ** 0.5
                if distance <= 3:  # Within 3 units
                    nearby.append(other_id)
        
        return nearby
    
    def _find_available_resources(self, position: Tuple[int, int], mesa_state: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Find resources available near the specified position"""
        resources = []
        
        if not position:
            return resources
        
        # Check room objects for resources
        room_objects = mesa_state.get('environment', {}).get('room_objects', {})
        for pos, obj in room_objects.items():
            if isinstance(pos, (list, tuple)) and len(pos) >= 2:
                distance = ((position[0] - pos[0]) ** 2 + (position[1] - pos[1]) ** 2) ** 0.5
                if distance <= 2 and obj.get('type') in ['key', 'tool', 'resource']:
                    resources.append({
                        'type': obj.get('type'),
                        'position': pos,
                        'available': obj.get('available', True)
                    })
        
        return resources
    
    def _create_unified_state(self, decisions: Dict[str, DecisionData], mesa_model: mesa.Model) -> Dict[str, Any]:
        """Create unified state representation"""
        return {
            'timestamp': datetime.now(),
            'decisions': {agent_id: {
                'action': decision.chosen_action,
                'parameters': decision.action_parameters,
                'confidence': decision.confidence_level
            } for agent_id, decision in decisions.items()},
            'mesa_state': self._extract_mesa_state(mesa_model)
        }
    
    def _record_sync_operation(self, sync_type: str, data: Any) -> None:
        """Record synchronization operation for debugging"""
        self.sync_history.append({
            'type': sync_type,
            'timestamp': datetime.now(),
            'data_size': len(str(data)) if data else 0,
            'success': True
        })
        
        # Keep only recent history (last 100 operations)
        if len(self.sync_history) > 100:
            self.sync_history = self.sync_history[-100:]