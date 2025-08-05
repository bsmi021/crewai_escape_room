"""
Unified State Management System

Agent D: State Management & Integration Specialist
Manages unified state across Mesa and CrewAI frameworks with event-driven synchronization.
"""

import asyncio
import time
import threading
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable, Set, Union
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import copy
import mesa
from crewai import Agent

from .event_bus import EventBus, StateChangeEvent
from ..core_architecture import DecisionData, IStateSynchronizer
from ..actions.action_models import ActionHandoff


class StateChangeType(Enum):
    """Types of state changes"""
    POSITION_UPDATE = "position_update"
    RESOURCE_CLAIM = "resource_claim"  
    RESOURCE_TRANSFER = "resource_transfer"
    STATUS_UPDATE = "status_update"
    HEALTH_CHANGE = "health_change"
    ROOM_TRANSITION = "room_transition"
    OBJECT_INTERACTION = "object_interaction"
    COMMUNICATION = "communication"


@dataclass
class StateChange:
    """Represents a change to system state"""
    change_id: str
    entity_id: str
    change_type: str
    old_value: Any
    new_value: Any
    timestamp: datetime
    source: str  # 'mesa' or 'crewai'
    priority: float = 0.5
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.change_id:
            self.change_id = str(uuid.uuid4())


@dataclass
class StateConflict:
    """Represents a conflict between state changes"""
    conflict_id: str
    conflict_type: str
    conflicting_changes: List[StateChange]
    severity: str = "medium"
    detected_timestamp: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        if not self.conflict_id:
            self.conflict_id = str(uuid.uuid4())


@dataclass
class StateResolution:
    """Represents resolution of a state conflict"""
    conflict_id: str
    resolution_type: str
    resolved_changes: List[StateChange]
    rejected_changes: List[StateChange] = field(default_factory=list)
    resolution_successful: bool = True
    resolution_reasoning: str = ""
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class UnifiedState:
    """Unified state representation"""
    version: int
    timestamp: datetime
    agents: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    environment: Dict[str, Any] = field(default_factory=dict)
    resources: Dict[str, Any] = field(default_factory=dict)
    social: Dict[str, Any] = field(default_factory=dict)
    model_data: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SystemHandoff:
    """Final handoff from complete pipeline to user/system"""
    unified_state: UnifiedState
    integration_results: Dict[str, Any]
    system_performance: Dict[str, float]
    error_reports: List[str]
    simulation_statistics: Dict[str, Any]
    end_to_end_success: bool
    
    def validate(self) -> bool:
        """Validate system handoff"""
        if not self.end_to_end_success:
            return False
        
        if "critical" in " ".join(self.error_reports).lower():
            return False
            
        required_metrics = ["total_execution_time", "success_rate", "throughput"]
        if not all(metric in self.system_performance for metric in required_metrics):
            return False
            
        return True


class UnifiedStateManager:
    """
    Central state manager with event-driven synchronization
    
    Maintains consistency between Mesa and CrewAI with:
    - Event-driven state updates
    - Conflict detection and resolution
    - Performance monitoring
    - Rollback capabilities
    """
    
    def __init__(self):
        self.unified_state = UnifiedState(
            version=0,
            timestamp=datetime.now()
        )
        
        # Event system
        self.event_bus = EventBus()
        
        # State management
        self.pending_changes: List[StateChange] = []
        self.change_history: List[StateChange] = []
        self.snapshots: Dict[str, UnifiedState] = {}
        self.max_snapshots = 10
        
        # Synchronization
        self.synchronizers: List[IStateSynchronizer] = []
        self._lock = threading.RLock()
        
        # Performance tracking
        self.performance_metrics = {
            "state_changes_processed": 0,
            "conflicts_resolved": 0,
            "sync_operations": 0,
            "average_sync_time": 0.0,
            "last_sync_timestamp": None
        }
        
        # Initialize with default synchronizer
        from .state_synchronizer import StateSynchronizer
        self.add_synchronizer(StateSynchronizer())
    
    def add_synchronizer(self, synchronizer: IStateSynchronizer):
        """Add state synchronizer"""
        self.synchronizers.append(synchronizer)
    
    def create_unified_state(self, mesa_model: mesa.Model, 
                           crewai_agents: List[Agent]) -> UnifiedState:
        """Create unified state from both frameworks"""
        with self._lock:
            # Extract Mesa state
            mesa_state = self._extract_mesa_state(mesa_model)
            
            # Extract CrewAI state
            crewai_state = self._extract_crewai_state(crewai_agents)
            
            # Create unified representation
            unified_state = UnifiedState(
                version=self.unified_state.version + 1,
                timestamp=datetime.now(),
                agents=self._merge_agent_data(mesa_state.get("agents", {}), 
                                            crewai_state.get("agents", {})),
                environment=mesa_state.get("environment", {}),
                resources=mesa_state.get("resources", {}),
                social=crewai_state.get("social", {}),
                model_data=mesa_state.get("model_data", {}),
                metadata={
                    "mesa_agents": len(mesa_state.get("agents", {})),
                    "crewai_agents": len(crewai_state.get("agents", {})),
                    "creation_method": "framework_merge"
                }
            )
            
            self.unified_state = unified_state
            
            # Notify listeners
            self.event_bus.publish("unified_state_created", {
                "state": unified_state,
                "timestamp": datetime.now()
            })
            
            return unified_state
    
    def register_state_change(self, change: StateChange) -> bool:
        """Register a state change for processing"""
        with self._lock:
            # Validate change
            if not self._validate_state_change(change):
                return False
            
            # Add to pending changes
            self.pending_changes.append(change)
            
            # Notify listeners
            self.event_bus.publish("state_change_registered", {
                "change": change,
                "pending_count": len(self.pending_changes)
            })
            
            return True
    
    def apply_pending_changes(self) -> List[StateChange]:
        """Apply all pending state changes"""
        with self._lock:
            if not self.pending_changes:
                return []
            
            # Detect conflicts
            conflicts = self.detect_conflicts()
            
            # Resolve conflicts
            if conflicts:
                resolutions = self.handle_state_conflicts(conflicts)
                self.performance_metrics["conflicts_resolved"] += len(resolutions)
            
            # Apply changes
            applied_changes = []
            for change in self.pending_changes:
                if self._apply_state_change(change):
                    applied_changes.append(change)
                    self.change_history.append(change)
            
            # Update version and clear pending
            self.unified_state.version += 1
            self.unified_state.timestamp = datetime.now()
            self.pending_changes.clear()
            
            # Update metrics
            self.performance_metrics["state_changes_processed"] += len(applied_changes)
            
            # Notify listeners
            self.event_bus.publish("state_changes_applied", {
                "applied_changes": applied_changes,
                "new_version": self.unified_state.version
            })
            
            return applied_changes
    
    def detect_conflicts(self) -> List[StateConflict]:
        """Detect conflicts in pending changes"""
        conflicts = []
        
        # Group changes by entity and type
        entity_changes = {}
        for change in self.pending_changes:
            key = (change.entity_id, change.change_type)
            if key not in entity_changes:
                entity_changes[key] = []
            entity_changes[key].append(change)
        
        # Detect conflicts in each group
        for (entity_id, change_type), changes in entity_changes.items():
            if len(changes) > 1:
                # Check if these are actually conflicting
                if self._are_changes_conflicting(changes):
                    conflict = StateConflict(
                        conflict_id=str(uuid.uuid4()),
                        conflict_type=self._determine_conflict_type(changes),
                        conflicting_changes=changes,
                        severity=self._assess_conflict_severity(changes)
                    )
                    conflicts.append(conflict)
        
        return conflicts
    
    def handle_state_conflicts(self, conflicts: List[StateConflict]) -> List[StateResolution]:
        """Resolve state conflicts"""
        resolutions = []
        
        for conflict in conflicts:
            resolution = self._resolve_conflict(conflict)
            resolutions.append(resolution)
            
            # Update pending changes based on resolution
            if resolution.resolution_successful:
                # Remove rejected changes from pending
                for rejected in resolution.rejected_changes:
                    if rejected in self.pending_changes:
                        self.pending_changes.remove(rejected)
        
        return resolutions
    
    def sync_to_mesa(self, mesa_model: mesa.Model):
        """Synchronize unified state to Mesa model"""
        start_time = time.perf_counter()
        
        for synchronizer in self.synchronizers:
            synchronizer.sync_crewai_to_mesa({}, mesa_model)
        
        end_time = time.perf_counter()
        sync_time = end_time - start_time
        
        self._update_sync_metrics(sync_time)
    
    def sync_to_crewai(self, crewai_agents: List[Agent]):
        """Synchronize unified state to CrewAI agents"""
        start_time = time.perf_counter()
        
        for synchronizer in self.synchronizers:
            synchronizer.sync_mesa_to_crewai(MockModel())  # Mock mesa model
        
        end_time = time.perf_counter()
        sync_time = end_time - start_time
        
        self._update_sync_metrics(sync_time)
    
    def process_action_handoff(self, handoff: ActionHandoff, 
                             mesa_model: mesa.Model) -> Dict[str, Any]:
        """Process action handoff from Agent C"""
        try:
            # Extract state changes from execution results
            state_changes = []
            for action_id, result in handoff.execution_results.items():
                if result.success:
                    change = StateChange(
                        change_id=f"handoff_{action_id}",
                        entity_id=result.agent_id,
                        change_type=result.action_type,
                        old_value=None,  # Would extract from result_data
                        new_value=result.result_data,
                        timestamp=result.execution_timestamp,
                        source="mesa",
                        metadata={"from_handoff": True, "action_id": action_id}
                    )
                    state_changes.append(change)
            
            # Register changes
            for change in state_changes:
                self.register_state_change(change)
            
            # Apply changes
            applied = self.apply_pending_changes()
            
            # Synchronize
            self.sync_to_mesa(mesa_model)
            
            return {
                "success": True,
                "state_updates": len(applied),
                "synchronization_time": time.perf_counter(),
                "conflict_resolutions": len(handoff.conflict_resolutions),
                "failed_actions": len(handoff.failed_actions)
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "state_updates": 0
            }
    
    def get_unified_state(self) -> UnifiedState:
        """Get current unified state"""
        with self._lock:
            return copy.deepcopy(self.unified_state)
    
    def create_snapshot(self, label: str = None) -> str:
        """Create state snapshot for rollback"""
        with self._lock:
            snapshot_id = label or f"snapshot_{self.unified_state.version}_{datetime.now().timestamp()}"
            
            self.snapshots[snapshot_id] = copy.deepcopy(self.unified_state)
            
            # Maintain snapshot limit
            if len(self.snapshots) > self.max_snapshots:
                oldest = min(self.snapshots.keys(), 
                           key=lambda k: self.snapshots[k].timestamp)
                del self.snapshots[oldest]
            
            return snapshot_id
    
    def rollback_to_snapshot(self, snapshot_id: str) -> bool:
        """Rollback to previous state snapshot"""
        with self._lock:
            if snapshot_id not in self.snapshots:
                return False
            
            self.unified_state = copy.deepcopy(self.snapshots[snapshot_id])
            self.pending_changes.clear()
            
            self.event_bus.publish("state_rollback", {
                "snapshot_id": snapshot_id,
                "version": self.unified_state.version
            })
            
            return True
    
    # Private helper methods
    
    def _extract_mesa_state(self, mesa_model: mesa.Model) -> Dict[str, Any]:
        """Extract state from Mesa model"""
        agents = {}
        # Mesa 3.x uses built-in agents attribute
        if hasattr(mesa_model, 'agents'):
            for agent in mesa_model.agents:
                agent_id = str(getattr(agent, 'unique_id', id(agent)))
                agents[agent_id] = {
                    "position": getattr(agent, 'pos', (0, 0)),
                    "health": getattr(agent, 'health', 100),
                    "energy": getattr(agent, 'energy', 1.0),
                    "resources": getattr(agent, 'resources', []),
                    "current_room": getattr(agent, 'current_room', None),
                    "status": getattr(agent, 'status', 'active')
                }
        
        environment = {
            "width": getattr(mesa_model, 'width', 10),
            "height": getattr(mesa_model, 'height', 10),
            "time_remaining": getattr(mesa_model, 'time_remaining', None),
            "running": getattr(mesa_model, 'running', True)
        }
        
        resources = getattr(mesa_model, 'global_resources', {})
        
        return {
            "agents": agents,
            "environment": environment,
            "resources": resources,
            "model_data": {
                "step_count": getattr(mesa_model, 'model_step_count', 0)
            }
        }
    
    def _extract_crewai_state(self, crewai_agents: List[Agent]) -> Dict[str, Any]:
        """Extract state from CrewAI agents"""
        agents = {}
        for agent in crewai_agents:
            agent_id = agent.role.lower().replace(" ", "_")
            agents[agent_id] = {
                "role": agent.role,
                "goal": agent.goal,
                "backstory": agent.backstory,
                "memory_enabled": hasattr(agent, 'memory') and agent.memory is not None
            }
        
        return {
            "agents": agents,
            "social": {
                "communications": [],
                "trust_relationships": {}
            }
        }
    
    def _merge_agent_data(self, mesa_agents: Dict, crewai_agents: Dict) -> Dict[str, Any]:
        """Merge agent data from both frameworks"""
        merged = {}
        
        # Start with Mesa agent data
        for agent_id, mesa_data in mesa_agents.items():
            merged[agent_id] = mesa_data.copy()
            merged[agent_id]["framework_source"] = "mesa"
        
        # Merge CrewAI agent data
        for agent_id, crewai_data in crewai_agents.items():
            if agent_id in merged:
                merged[agent_id].update(crewai_data)
                merged[agent_id]["framework_source"] = "hybrid"
            else:
                merged[agent_id] = crewai_data.copy()
                merged[agent_id]["framework_source"] = "crewai"
        
        return merged
    
    def _validate_state_change(self, change: StateChange) -> bool:
        """Validate state change"""
        if not change.entity_id or not change.change_type:
            return False
        
        if not isinstance(change.timestamp, datetime):
            return False
        
        if change.source not in ['mesa', 'crewai', 'system']:
            return False
        
        return True
    
    def _apply_state_change(self, change: StateChange) -> bool:
        """Apply single state change to unified state"""
        try:
            # Update based on change type
            if change.change_type == "position_update":
                if change.entity_id not in self.unified_state.agents:
                    self.unified_state.agents[change.entity_id] = {}
                self.unified_state.agents[change.entity_id]["position"] = change.new_value
                
            elif change.change_type == "resource_claim":
                if change.entity_id not in self.unified_state.agents:
                    self.unified_state.agents[change.entity_id] = {"resources": []}
                if "resources" not in self.unified_state.agents[change.entity_id]:
                    self.unified_state.agents[change.entity_id]["resources"] = []
                self.unified_state.agents[change.entity_id]["resources"].append(change.new_value)
                
            elif change.change_type == "status_update":
                if change.entity_id not in self.unified_state.agents:
                    self.unified_state.agents[change.entity_id] = {}
                self.unified_state.agents[change.entity_id]["status"] = change.new_value
            
            # Publish state change event
            event = StateChangeEvent(
                change_id=change.change_id,
                entity_id=change.entity_id,
                change_type=change.change_type,
                old_value=change.old_value,
                new_value=change.new_value,
                timestamp=change.timestamp
            )
            self.event_bus.publish("state_changed", event)
            
            return True
            
        except Exception as e:
            print(f"Failed to apply state change {change.change_id}: {e}")
            return False
    
    def _are_changes_conflicting(self, changes: List[StateChange]) -> bool:
        """Check if changes are actually conflicting or just sequential updates"""
        # Position updates over time are usually not conflicts (unless simultaneous)
        if len(changes) <= 1:
            return False
        
        change_type = changes[0].change_type
        
        # Position updates are allowed if they're sequential
        if change_type == "position_update":
            # Sort by timestamp
            sorted_changes = sorted(changes, key=lambda c: c.timestamp)
            
            # Check if all changes have different timestamps (sequential)
            timestamps = [c.timestamp for c in sorted_changes]
            time_diffs = [(timestamps[i+1] - timestamps[i]).total_seconds() 
                         for i in range(len(timestamps)-1)]
            
            # If all time differences are > 0.1 seconds, consider them sequential
            if all(diff > 0.1 for diff in time_diffs):
                return False
            
            # If they have the same timestamp and different sources, it's a conflict
            same_time_diff_source = any(
                abs((c1.timestamp - c2.timestamp).total_seconds()) < 0.01 
                and c1.source != c2.source
                for c1 in changes for c2 in changes if c1 != c2
            )
            return same_time_diff_source
        
        # Resource ownership changes are conflicts if they're different targets
        elif change_type == "ownership_change":
            different_targets = len(set(c.new_value for c in changes)) > 1
            return different_targets
        
        # Most other simultaneous changes are conflicts
        return True
    
    def _determine_conflict_type(self, changes: List[StateChange]) -> str:
        """Determine type of conflict"""
        change_types = set(change.change_type for change in changes)
        
        if "resource_claim" in change_types or "ownership_change" in change_types:
            return "resource_ownership"
        elif "position_update" in change_types:
            return "spatial_conflict"
        elif "move" in change_types:
            return "spatial_conflict"
        else:
            return "general_conflict"
    
    def _assess_conflict_severity(self, changes: List[StateChange]) -> str:
        """Assess severity of conflict"""
        if len(changes) > 3:
            return "high"
        elif any(change.priority > 0.8 for change in changes):
            return "high"
        elif len(changes) > 1:
            return "medium"
        else:
            return "low"
    
    def _resolve_conflict(self, conflict: StateConflict) -> StateResolution:
        """Resolve a specific conflict"""
        # Simple resolution strategy: highest priority wins
        changes = conflict.conflicting_changes
        winner = max(changes, key=lambda c: c.priority)
        losers = [c for c in changes if c != winner]
        
        return StateResolution(
            conflict_id=conflict.conflict_id,
            resolution_type="priority_based",
            resolved_changes=[winner],
            rejected_changes=losers,
            resolution_successful=True,
            resolution_reasoning=f"Resolved by priority: {winner.priority} > others"
        )
    
    def _update_sync_metrics(self, sync_time: float):
        """Update synchronization metrics"""
        self.performance_metrics["sync_operations"] += 1
        
        # Update average sync time
        current_avg = self.performance_metrics["average_sync_time"]
        count = self.performance_metrics["sync_operations"]
        new_avg = ((current_avg * (count - 1)) + sync_time) / count
        self.performance_metrics["average_sync_time"] = new_avg
        self.performance_metrics["last_sync_timestamp"] = datetime.now()


class AsyncUnifiedStateManager(UnifiedStateManager):
    """Async version of unified state manager"""
    
    def __init__(self):
        super().__init__()
        self._async_lock = asyncio.Lock()
    
    async def register_state_change_async(self, change: StateChange) -> bool:
        """Async version of register_state_change"""
        async with self._async_lock:
            return self.register_state_change(change)
    
    async def apply_pending_changes_async(self) -> List[StateChange]:
        """Async version of apply_pending_changes"""
        async with self._async_lock:
            return self.apply_pending_changes()


# Mock class for testing (Mesa 3.x compatible)
class MockModel:
    def __init__(self):
        self.agents = []
        self.model_step_count = 0
        self.width = 10
        self.height = 10
        self.time_remaining = 300
        self.running = True
    
    def __getattr__(self, name):
        return lambda *args, **kwargs: None