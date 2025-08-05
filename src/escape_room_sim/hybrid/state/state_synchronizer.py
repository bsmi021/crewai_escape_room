"""
State Synchronizer Implementation

Agent D: State Management & Integration Specialist
Implements state synchronization between Mesa and CrewAI frameworks.
"""

import time
from datetime import datetime
from typing import Dict, List, Any, Optional
from unittest.mock import Mock
import mesa
from crewai import Agent

from ..core_architecture import DecisionData, IStateSynchronizer
from ..actions.action_models import ActionHandoff


class StateSynchronizer(IStateSynchronizer):
    """
    Concrete implementation of state synchronizer
    
    Handles bidirectional synchronization between Mesa and CrewAI.
    """
    
    def __init__(self):
        self.sync_history: List[Dict[str, Any]] = []
        self.performance_metrics = {
            "mesa_to_crewai_syncs": 0,
            "crewai_to_mesa_syncs": 0,
            "average_sync_time": 0.0,
            "failed_syncs": 0
        }
    
    def sync_mesa_to_crewai(self, mesa_model: mesa.Model) -> Dict[str, Any]:
        """Synchronize Mesa model state to CrewAI agents"""
        start_time = time.perf_counter()
        
        try:
            # Extract Mesa state
            mesa_state = self._extract_mesa_state(mesa_model)
            
            # Convert to CrewAI-compatible format
            crewai_updates = self._convert_mesa_to_crewai_format(mesa_state)
            
            # Record sync operation
            sync_record = {
                "direction": "mesa_to_crewai",
                "timestamp": datetime.now(),
                "agent_count": len(mesa_state.get("agents", {})),
                "environment_updates": len(mesa_state.get("environment", {})),
                "success": True
            }
            
            self.sync_history.append(sync_record)
            self.performance_metrics["mesa_to_crewai_syncs"] += 1
            
            end_time = time.perf_counter()
            sync_time = end_time - start_time
            self._update_performance_metrics(sync_time)
            
            return {
                "agent_states": crewai_updates.get("agents", {}),
                "environment_state": crewai_updates.get("environment", {}),
                "sync_time": sync_time,
                "success": True
            }
            
        except Exception as e:
            self.performance_metrics["failed_syncs"] += 1
            return {
                "agent_states": {},
                "environment_state": {},
                "error": str(e),
                "success": False
            }
    
    def sync_crewai_to_mesa(self, decisions: Dict[str, DecisionData], 
                          mesa_model: mesa.Model) -> Dict[str, Any]:
        """Synchronize CrewAI decisions to Mesa model"""
        start_time = time.perf_counter()
        
        try:
            # Convert decisions to Mesa actions
            mesa_updates = self._convert_crewai_to_mesa_format(decisions)
            
            # Apply updates to Mesa model
            applied_updates = self._apply_updates_to_mesa(mesa_updates, mesa_model)
            
            # Record sync operation
            sync_record = {
                "direction": "crewai_to_mesa",
                "timestamp": datetime.now(),
                "decisions_processed": len(decisions),
                "updates_applied": len(applied_updates),
                "success": True
            }
            
            self.sync_history.append(sync_record)
            self.performance_metrics["crewai_to_mesa_syncs"] += 1
            
            end_time = time.perf_counter()
            sync_time = end_time - start_time
            self._update_performance_metrics(sync_time)
            
            return {
                "decisions_applied": applied_updates,
                "mesa_state_updated": True,
                "sync_time": sync_time,
                "success": True
            }
            
        except Exception as e:
            self.performance_metrics["failed_syncs"] += 1
            return {
                "decisions_applied": [],
                "mesa_state_updated": False,
                "error": str(e),
                "success": False
            }
    
    def process_action_handoff(self, handoff: ActionHandoff, 
                             mesa_model: mesa.Model) -> Dict[str, Any]:
        """Process action handoff from Agent C"""
        start_time = time.perf_counter()
        
        try:
            # Extract state changes from execution results
            state_changes = []
            for action_id, result in handoff.execution_results.items():
                if result.success:
                    change_data = {
                        "agent_id": result.agent_id,
                        "action_type": result.action_type,
                        "result_data": result.result_data,
                        "timestamp": result.execution_timestamp
                    }
                    state_changes.append(change_data)
            
            # Apply state changes to unified representation
            unified_updates = self._create_unified_updates(state_changes)
            
            # Synchronize back to frameworks
            mesa_sync_result = self._apply_to_mesa_model(unified_updates, mesa_model)
            
            end_time = time.perf_counter()
            processing_time = end_time - start_time
            
            return {
                "success": True,
                "state_updates": len(state_changes),
                "synchronization_time": processing_time,
                "conflict_resolutions": len(handoff.conflict_resolutions),
                "failed_actions": len(handoff.failed_actions),
                "mesa_sync_success": mesa_sync_result["success"],
                "performance_metrics": handoff.performance_metrics
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "state_updates": 0,
                "synchronization_time": time.perf_counter() - start_time
            }
    
    def create_unified_state(self, mesa_model: mesa.Model, 
                           crewai_state: Dict) -> Dict[str, Any]:
        """Create unified state representation"""
        try:
            mesa_state = self._extract_mesa_state(mesa_model)
            
            unified = {
                "version": 1,
                "timestamp": datetime.now(),
                "agents": self._merge_agent_states(
                    mesa_state.get("agents", {}),
                    crewai_state.get("agents", {})
                ),
                "environment": mesa_state.get("environment", {}),
                "resources": mesa_state.get("resources", {}),
                "social": crewai_state.get("social", {}),
                "metadata": {
                    "mesa_agent_count": len(mesa_state.get("agents", {})),
                    "crewai_agent_count": len(crewai_state.get("agents", {})),
                    "sync_timestamp": datetime.now().isoformat()
                }
            }
            
            return unified
            
        except Exception as e:
            return {
                "version": 0,
                "timestamp": datetime.now(),
                "error": str(e),
                "agents": {},
                "environment": {},
                "resources": {}
            }
    
    def handle_state_conflicts(self, conflicts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Handle state conflicts using resolution strategies"""
        resolutions = []
        
        for conflict in conflicts:
            resolution = self._resolve_conflict(conflict)
            resolutions.append(resolution)
        
        return resolutions
    
    def get_synchronization_metrics(self) -> Dict[str, Any]:
        """Get synchronization performance metrics"""
        return {
            "performance": self.performance_metrics.copy(),
            "sync_history_count": len(self.sync_history),
            "recent_syncs": self.sync_history[-10:] if self.sync_history else [],
            "average_sync_time": self.performance_metrics["average_sync_time"],
            "success_rate": self._calculate_success_rate()
        }
    
    # Private helper methods
    
    def _extract_mesa_state(self, mesa_model: mesa.Model) -> Dict[str, Any]:
        """Extract current state from Mesa model"""
        agents = {}
        
        # Handle both real Mesa models and mocks
        if hasattr(mesa_model, 'schedule') and hasattr(mesa_model.schedule, 'agents'):
            for agent in mesa_model.schedule.agents:
                agent_id = str(getattr(agent, 'unique_id', id(agent)))
                agents[agent_id] = {
                    "position": getattr(agent, 'pos', (0, 0)),
                    "health": getattr(agent, 'health', 100),
                    "energy": getattr(agent, 'energy', 1.0),
                    "resources": getattr(agent, 'resources', []),
                    "current_room": getattr(agent, 'current_room', None),
                    "status": getattr(agent, 'status', 'active'),
                    "last_action": getattr(agent, 'last_action', None)
                }
        
        environment = {
            "width": getattr(mesa_model, 'width', 10),
            "height": getattr(mesa_model, 'height', 10),
            "time_remaining": getattr(mesa_model, 'time_remaining', None),
            "running": getattr(mesa_model, 'running', True),
            "room_objects": getattr(mesa_model, 'room_objects', {}),
            "hazards": getattr(mesa_model, 'hazards', [])
        }
        
        resources = getattr(mesa_model, 'global_resources', {})
        
        return {
            "agents": agents,
            "environment": environment,
            "resources": resources,
            "model_step": getattr(mesa_model.schedule, 'steps', 0) if hasattr(mesa_model, 'schedule') else 0
        }
    
    def _convert_mesa_to_crewai_format(self, mesa_state: Dict[str, Any]) -> Dict[str, Any]:
        """Convert Mesa state to CrewAI-compatible format"""
        crewai_format = {
            "agents": {},
            "environment": mesa_state.get("environment", {}),
            "observations": []
        }
        
        # Convert agent states
        for agent_id, agent_data in mesa_state.get("agents", {}).items():
            crewai_format["agents"][agent_id] = {
                "memory_updates": {
                    "position": agent_data.get("position"),
                    "health": agent_data.get("health"),
                    "resources": agent_data.get("resources", []),
                    "environment_state": mesa_state.get("environment", {}),
                    "last_action_result": agent_data.get("last_action")
                },
                "observations": [
                    f"Current position: {agent_data.get('position')}",
                    f"Health: {agent_data.get('health')}",
                    f"Resources: {agent_data.get('resources', [])}"
                ]
            }
        
        return crewai_format
    
    def _convert_crewai_to_mesa_format(self, decisions: Dict[str, DecisionData]) -> Dict[str, Any]:
        """Convert CrewAI decisions to Mesa-compatible format"""
        mesa_format = {
            "agent_actions": {},
            "state_updates": {}
        }
        
        for agent_id, decision in decisions.items():
            mesa_format["agent_actions"][agent_id] = {
                "action_type": decision.chosen_action,
                "parameters": decision.action_parameters,
                "priority": decision.confidence_level,
                "reasoning": decision.reasoning,
                "timestamp": decision.timestamp
            }
            
            # Convert specific actions to state updates
            if decision.chosen_action == "move":
                target_pos = decision.action_parameters.get("target_position")
                if target_pos:
                    mesa_format["state_updates"][agent_id] = {
                        "position": target_pos
                    }
            elif decision.chosen_action == "communicate":
                mesa_format["state_updates"][agent_id] = {
                    "last_communication": {
                        "target": decision.action_parameters.get("target"),
                        "message": decision.action_parameters.get("message"),
                        "timestamp": decision.timestamp
                    }
                }
        
        return mesa_format
    
    def _apply_updates_to_mesa(self, updates: Dict[str, Any], 
                             mesa_model: mesa.Model) -> List[Dict[str, Any]]:
        """Apply updates to Mesa model"""
        applied_updates = []
        
        # Apply agent actions
        agent_actions = updates.get("agent_actions", {})
        for agent_id, action_data in agent_actions.items():
            # Find corresponding Mesa agent
            mesa_agent = self._find_mesa_agent(agent_id, mesa_model)
            if mesa_agent:
                update_record = {
                    "agent_id": agent_id,
                    "action": action_data["action_type"],
                    "applied": True,
                    "timestamp": datetime.now()
                }
                applied_updates.append(update_record)
        
        # Apply state updates
        state_updates = updates.get("state_updates", {})
        for agent_id, state_data in state_updates.items():
            mesa_agent = self._find_mesa_agent(agent_id, mesa_model)
            if mesa_agent:
                # Apply position updates
                if "position" in state_data:
                    mesa_agent.pos = state_data["position"]
                
                # Apply other state updates
                for key, value in state_data.items():
                    if key != "position":
                        setattr(mesa_agent, key, value)
                
                update_record = {
                    "agent_id": agent_id,
                    "state_update": list(state_data.keys()),
                    "applied": True,
                    "timestamp": datetime.now()
                }
                applied_updates.append(update_record)
        
        return applied_updates
    
    def _create_unified_updates(self, state_changes: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create unified state updates from action results"""
        unified = {
            "agent_updates": {},
            "environment_updates": {},
            "resource_updates": {}
        }
        
        for change in state_changes:
            agent_id = change["agent_id"]
            action_type = change["action_type"]
            result_data = change.get("result_data", {})
            
            if agent_id not in unified["agent_updates"]:
                unified["agent_updates"][agent_id] = {}
            
            # Process different action types
            if action_type == "move":
                if "new_position" in result_data:
                    unified["agent_updates"][agent_id]["position"] = result_data["new_position"]
            
            elif action_type == "examine":
                if "findings" in result_data:
                    unified["agent_updates"][agent_id]["last_examination"] = result_data["findings"]
            
            elif action_type == "communicate":
                if "delivery_status" in result_data:
                    unified["agent_updates"][agent_id]["last_communication"] = result_data
            
            elif action_type == "claim_resource":
                if "claim_success" in result_data and result_data["claim_success"]:
                    resource_id = result_data.get("resource_id")
                    if resource_id:
                        if "resources" not in unified["agent_updates"][agent_id]:
                            unified["agent_updates"][agent_id]["resources"] = []
                        unified["agent_updates"][agent_id]["resources"].append(resource_id)
        
        return unified
    
    def _apply_to_mesa_model(self, updates: Dict[str, Any], 
                           mesa_model: mesa.Model) -> Dict[str, Any]:
        """Apply unified updates to Mesa model"""
        try:
            applied_count = 0
            
            # Apply agent updates
            agent_updates = updates.get("agent_updates", {})
            for agent_id, update_data in agent_updates.items():
                mesa_agent = self._find_mesa_agent(agent_id, mesa_model)
                if mesa_agent:
                    for key, value in update_data.items():
                        if key == "position" and hasattr(mesa_agent, 'pos'):
                            mesa_agent.pos = value
                        elif key == "resources" and hasattr(mesa_agent, 'resources'):
                            mesa_agent.resources = value
                        else:
                            setattr(mesa_agent, key, value)
                    applied_count += 1
            
            return {
                "success": True,
                "updates_applied": applied_count,
                "timestamp": datetime.now()
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "updates_applied": 0
            }
    
    def _find_mesa_agent(self, agent_id: str, mesa_model: mesa.Model):
        """Find Mesa agent by ID"""
        if not hasattr(mesa_model, 'schedule') or not hasattr(mesa_model.schedule, 'agents'):
            return None
        
        for agent in mesa_model.schedule.agents:
            if str(getattr(agent, 'unique_id', id(agent))) == agent_id:
                return agent
        
        return None
    
    def _merge_agent_states(self, mesa_agents: Dict, crewai_agents: Dict) -> Dict[str, Any]:
        """Merge agent states from both frameworks"""
        merged = {}
        
        # Start with Mesa agents
        for agent_id, mesa_data in mesa_agents.items():
            merged[agent_id] = mesa_data.copy()
            merged[agent_id]["source"] = "mesa"
        
        # Add CrewAI agent data
        for agent_id, crewai_data in crewai_agents.items():
            if agent_id in merged:
                # Merge data
                merged[agent_id].update(crewai_data)
                merged[agent_id]["source"] = "hybrid"
            else:
                merged[agent_id] = crewai_data.copy()
                merged[agent_id]["source"] = "crewai"
        
        return merged
    
    def _resolve_conflict(self, conflict: Dict[str, Any]) -> Dict[str, Any]:
        """Resolve a state conflict"""
        # Simple resolution: prefer most recent change
        conflicting_changes = conflict.get("changes", [])
        if not conflicting_changes:
            return {"resolved": False, "reason": "no_changes"}
        
        # Sort by timestamp and take most recent
        sorted_changes = sorted(conflicting_changes, 
                              key=lambda c: c.get("timestamp", datetime.min))
        winning_change = sorted_changes[-1]
        
        return {
            "resolved": True,
            "resolution_type": "timestamp_priority",
            "winning_change": winning_change,
            "rejected_changes": sorted_changes[:-1],
            "timestamp": datetime.now()
        }
    
    def _update_performance_metrics(self, sync_time: float):
        """Update performance tracking metrics"""
        total_syncs = (self.performance_metrics["mesa_to_crewai_syncs"] + 
                      self.performance_metrics["crewai_to_mesa_syncs"])
        
        if total_syncs == 1:
            self.performance_metrics["average_sync_time"] = sync_time
        else:
            current_avg = self.performance_metrics["average_sync_time"]
            new_avg = ((current_avg * (total_syncs - 1)) + sync_time) / total_syncs
            self.performance_metrics["average_sync_time"] = new_avg
    
    def _calculate_success_rate(self) -> float:
        """Calculate synchronization success rate"""
        total_syncs = (self.performance_metrics["mesa_to_crewai_syncs"] + 
                      self.performance_metrics["crewai_to_mesa_syncs"])
        
        if total_syncs == 0:
            return 1.0
        
        failed_syncs = self.performance_metrics["failed_syncs"]
        return (total_syncs - failed_syncs) / total_syncs