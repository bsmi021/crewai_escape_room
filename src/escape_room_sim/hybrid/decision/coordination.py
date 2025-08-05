"""
Coordination Protocols for Synchronized Multi-Agent Actions

Implements coordination protocols for executing synchronized actions
between multiple agents in the escape room scenario.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class CheckpointStatus(Enum):
    """Status of synchronization checkpoints"""
    PENDING = "pending"
    REACHED = "reached"
    MISSED = "missed"
    SKIPPED = "skipped"


@dataclass
class SynchronizationPoint:
    """Represents a synchronization checkpoint"""
    time: float
    agents: List[str]
    checkpoint: str
    status: CheckpointStatus = CheckpointStatus.PENDING
    actual_time: Optional[float] = None
    reached_agents: List[str] = None
    
    def __post_init__(self):
        if self.reached_agents is None:
            self.reached_agents = []


@dataclass
class CoordinatedAction:
    """Represents an action in a coordinated sequence"""
    agent_id: str
    action: str
    duration: float
    dependencies: List[str]
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    status: str = "pending"
    parameters: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.parameters is None:
            self.parameters = {}


class CoordinationProtocol:
    """
    Manages coordination of synchronized actions between agents
    """
    
    def __init__(self, coordination_scenario: Dict[str, Any]):
        """
        Initialize coordination protocol
        
        Args:
            coordination_scenario: Dictionary defining coordinated actions and sync points
        """
        self.scenario = coordination_scenario
        self.coordinated_actions = self._parse_coordinated_actions()
        self.synchronization_points = self._parse_synchronization_points()
        self.start_time: Optional[datetime] = None
        self.status = "initialized"
        
        logger.info(f"Initialized coordination protocol with {len(self.coordinated_actions)} actions")
    
    def _parse_coordinated_actions(self) -> List[CoordinatedAction]:
        """Parse coordinated actions from scenario"""
        actions = []
        
        coordinated_actions_data = self.scenario.get("coordinated_actions", {})
        for agent_id, action_data in coordinated_actions_data.items():
            action = CoordinatedAction(
                agent_id=agent_id,
                action=action_data["action"],
                duration=action_data["duration"],
                dependencies=action_data.get("dependencies", []),
                parameters=action_data.get("parameters", {})
            )
            actions.append(action)
        
        return actions
    
    def _parse_synchronization_points(self) -> List[SynchronizationPoint]:
        """Parse synchronization points from scenario"""
        sync_points = []
        
        sync_points_data = self.scenario.get("synchronization_points", [])
        for point_data in sync_points_data:
            sync_point = SynchronizationPoint(
                time=point_data["time"],
                agents=point_data["agents"],
                checkpoint=point_data["checkpoint"]
            )
            sync_points.append(sync_point)
        
        return sync_points
    
    async def create_execution_plan(self) -> Dict[str, Any]:
        """
        Create detailed execution plan for coordinated actions
        
        Returns:
            Execution plan with timeline, checkpoints, and assignments
        """
        logger.info("Creating execution plan for coordinated actions")
        
        # Create timeline
        timeline = self._create_timeline()
        
        # Validate synchronization points
        valid_sync_points = self._validate_sync_points()
        
        # Create agent assignments
        agent_assignments = self._create_agent_assignments()
        
        # Calculate critical path
        critical_path = self._calculate_critical_path()
        
        execution_plan = {
            "timeline": timeline,
            "checkpoints": valid_sync_points,
            "agent_assignments": agent_assignments,
            "critical_path": critical_path,
            "total_duration": max(action.duration for action in self.coordinated_actions),
            "coordination_complexity": len(self.synchronization_points),
            "created_at": datetime.now().isoformat()
        }
        
        logger.info(f"Execution plan created with {len(timeline)} timeline events")
        return execution_plan
    
    def _create_timeline(self) -> List[Dict[str, Any]]:
        """Create detailed timeline of coordinated actions"""
        timeline = []
        
        # Sort actions by start time (considering dependencies)
        scheduled_actions = self._schedule_actions()
        
        for action in scheduled_actions:
            # Start event
            timeline.append({
                "time": action.start_time,
                "event_type": "action_start",
                "agent_id": action.agent_id,
                "action": action.action,
                "description": f"{action.agent_id} starts {action.action}"
            })
            
            # End event
            timeline.append({
                "time": action.end_time,
                "event_type": "action_end",
                "agent_id": action.agent_id,
                "action": action.action,
                "description": f"{action.agent_id} completes {action.action}"
            })
        
        # Add synchronization points
        for sync_point in self.synchronization_points:
            timeline.append({
                "time": sync_point.time,
                "event_type": "synchronization",
                "agents": sync_point.agents,
                "checkpoint": sync_point.checkpoint,
                "description": f"Sync checkpoint: {sync_point.checkpoint}"
            })
        
        # Sort timeline by time
        timeline.sort(key=lambda x: x["time"])
        
        return timeline
    
    def _schedule_actions(self) -> List[CoordinatedAction]:
        """Schedule actions considering dependencies"""
        scheduled = []
        remaining = self.coordinated_actions.copy()
        current_time = 0.0
        
        while remaining:
            # Find actions with satisfied dependencies
            ready_actions = []
            for action in remaining:
                deps_satisfied = all(
                    dep_action in [s.action for s in scheduled if s.agent_id == action.agent_id]
                    for dep_action in action.dependencies
                )
                if deps_satisfied:
                    ready_actions.append(action)
            
            if not ready_actions:
                # Force schedule to avoid deadlock
                ready_actions = [remaining[0]]
            
            # Schedule ready actions
            for action in ready_actions:
                # Calculate start time based on dependencies
                dep_end_times = []
                for dep_action_name in action.dependencies:
                    for scheduled_action in scheduled:
                        if (scheduled_action.action == dep_action_name and 
                            scheduled_action.agent_id == action.agent_id):
                            dep_end_times.append(scheduled_action.end_time)
                
                start_time = max(dep_end_times) if dep_end_times else current_time
                
                action.start_time = start_time
                action.end_time = start_time + action.duration
                action.status = "scheduled"
                
                scheduled.append(action)
                remaining.remove(action)
                
                current_time = max(current_time, action.end_time)
        
        return scheduled
    
    def _validate_sync_points(self) -> List[Dict[str, Any]]:
        """Validate synchronization points against scheduled actions"""
        valid_points = []
        
        for sync_point in self.synchronization_points:
            is_valid = True
            validation_notes = []
            
            # Check if agents are available at sync time
            for agent_id in sync_point.agents:
                agent_actions = [a for a in self.coordinated_actions if a.agent_id == agent_id]
                
                for action in agent_actions:
                    if (action.start_time <= sync_point.time <= action.end_time):
                        is_valid = False
                        validation_notes.append(
                            f"{agent_id} busy with {action.action} at sync time {sync_point.time}"
                        )
            
            valid_points.append({
                "checkpoint": sync_point.checkpoint,
                "time": sync_point.time,
                "agents": sync_point.agents,
                "valid": is_valid,
                "notes": validation_notes
            })
        
        return valid_points
    
    def _create_agent_assignments(self) -> Dict[str, List[Dict[str, Any]]]:
        """Create agent-specific assignments"""
        assignments = {}
        
        for action in self.coordinated_actions:
            if action.agent_id not in assignments:
                assignments[action.agent_id] = []
            
            assignments[action.agent_id].append({
                "action": action.action,
                "start_time": action.start_time,
                "duration": action.duration,
                "dependencies": action.dependencies,
                "parameters": action.parameters,
                "sync_points": [
                    sp.checkpoint for sp in self.synchronization_points 
                    if action.agent_id in sp.agents and 
                    action.start_time <= sp.time <= action.end_time
                ]
            })
        
        return assignments
    
    def _calculate_critical_path(self) -> List[str]:
        """Calculate critical path through coordinated actions"""
        # Simple critical path: longest sequence of dependent actions
        paths = []
        
        for action in self.coordinated_actions:
            if not action.dependencies:  # Start action
                path = self._find_longest_path(action, [])
                paths.append(path)
        
        if not paths:
            return [action.action for action in self.coordinated_actions]
        
        # Return longest path
        longest_path = max(paths, key=len)
        return [action.action for action in longest_path]
    
    def _find_longest_path(self, current_action: CoordinatedAction, 
                          visited: List[CoordinatedAction]) -> List[CoordinatedAction]:
        """Find longest path from current action"""
        if current_action in visited:
            return visited
        
        visited = visited + [current_action]
        
        # Find actions that depend on current action
        dependent_actions = [
            action for action in self.coordinated_actions
            if current_action.action in action.dependencies and 
            action.agent_id == current_action.agent_id
        ]
        
        if not dependent_actions:
            return visited
        
        # Find longest path among dependents
        longest_path = visited
        for dep_action in dependent_actions:
            path = self._find_longest_path(dep_action, visited)
            if len(path) > len(longest_path):
                longest_path = path
        
        return longest_path
    
    async def execute_coordination(self) -> Dict[str, Any]:
        """
        Execute the coordination protocol
        
        Returns:
            Execution results and metrics
        """
        logger.info("Starting coordination execution")
        
        self.start_time = datetime.now()
        self.status = "executing"
        
        execution_results = {
            "start_time": self.start_time.isoformat(),
            "status": "executing",
            "completed_actions": [],
            "sync_point_results": [],
            "errors": []
        }
        
        try:
            # Execute actions according to schedule
            await self._execute_scheduled_actions(execution_results)
            
            # Check synchronization points
            await self._monitor_synchronization_points(execution_results)
            
            execution_results["status"] = "completed"
            execution_results["end_time"] = datetime.now().isoformat()
            execution_results["total_duration"] = (
                datetime.now() - self.start_time
            ).total_seconds()
            
        except Exception as e:
            logger.error(f"Coordination execution failed: {e}")
            execution_results["status"] = "failed"
            execution_results["error"] = str(e)
        
        self.status = execution_results["status"]
        return execution_results
    
    async def _execute_scheduled_actions(self, results: Dict[str, Any]):
        """Execute scheduled actions"""
        # Create tasks for each action
        action_tasks = []
        
        for action in self.coordinated_actions:
            task = self._execute_action(action)
            action_tasks.append(task)
        
        # Wait for all actions to complete
        completed_actions = await asyncio.gather(*action_tasks, return_exceptions=True)
        
        for i, result in enumerate(completed_actions):
            action = self.coordinated_actions[i]
            
            if isinstance(result, Exception):
                results["errors"].append({
                    "agent_id": action.agent_id,
                    "action": action.action,
                    "error": str(result)
                })
            else:
                results["completed_actions"].append({
                    "agent_id": action.agent_id,
                    "action": action.action,
                    "result": result,
                    "duration": result.get("duration", action.duration)
                })
    
    async def _execute_action(self, action: CoordinatedAction) -> Dict[str, Any]:
        """Execute a single coordinated action"""
        # Wait for start time
        if action.start_time:
            delay = action.start_time
            if delay > 0:
                await asyncio.sleep(delay)
        
        # Simulate action execution
        start_time = datetime.now()
        
        # Mock execution - in real implementation this would call actual action
        await asyncio.sleep(min(action.duration, 0.1))  # Cap simulation time
        
        end_time = datetime.now()
        actual_duration = (end_time - start_time).total_seconds()
        
        return {
            "agent_id": action.agent_id,
            "action": action.action,
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "duration": actual_duration,
            "success": True
        }
    
    async def _monitor_synchronization_points(self, results: Dict[str, Any]):
        """Monitor synchronization points during execution"""
        for sync_point in self.synchronization_points:
            # Wait for sync time
            if self.start_time:
                elapsed = (datetime.now() - self.start_time).total_seconds()
                if elapsed < sync_point.time:
                    await asyncio.sleep(sync_point.time - elapsed)
            
            # Check which agents reached the sync point
            reached_agents = []  # In real implementation, this would check actual agent states
            
            sync_result = {
                "checkpoint": sync_point.checkpoint,
                "scheduled_time": sync_point.time,
                "actual_time": (datetime.now() - self.start_time).total_seconds() if self.start_time else 0,
                "expected_agents": sync_point.agents,
                "reached_agents": reached_agents,
                "success": len(reached_agents) == len(sync_point.agents)
            }
            
            results["sync_point_results"].append(sync_result)
    
    async def handle_coordination_failure(self, failure_scenario: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle coordination failure and create recovery plan
        
        Args:
            failure_scenario: Description of the failure
            
        Returns:
            Recovery plan
        """
        failed_agent = failure_scenario.get("failed_agent")
        failure_time = failure_scenario.get("failure_time", 0.0)
        failure_reason = failure_scenario.get("failure_reason", "unknown")
        
        logger.warning(f"Handling coordination failure: {failed_agent} at {failure_time} ({failure_reason})")
        
        # Find affected actions
        affected_actions = [
            action for action in self.coordinated_actions
            if (action.agent_id == failed_agent or 
                any(dep_action in [a.action for a in self.coordinated_actions if a.agent_id == failed_agent]
                    for dep_action in action.dependencies))
        ]
        
        # Create recovery plan
        recovery_plan = {
            "failure_description": failure_scenario,
            "affected_actions": [
                {"agent": action.agent_id, "action": action.action} 
                for action in affected_actions
            ],
            "replacement_actions": [],
            "timeline_adjustment": {},
            "affected_agents": list(set(action.agent_id for action in affected_actions))
        }
        
        # Generate replacement actions
        for action in affected_actions:
            if action.agent_id == failed_agent:
                # Reassign to another agent
                other_agents = [
                    a.agent_id for a in self.coordinated_actions 
                    if a.agent_id != failed_agent
                ]
                if other_agents:
                    replacement_action = {
                        "original_agent": failed_agent,
                        "replacement_agent": other_agents[0],
                        "action": action.action,
                        "new_duration": action.duration * 1.5,  # Assume slower execution
                        "new_start_time": failure_time + 1.0
                    }
                    recovery_plan["replacement_actions"].append(replacement_action)
            else:
                # Adjust timeline for dependent actions
                recovery_plan["timeline_adjustment"][action.agent_id] = {
                    "action": action.action,
                    "delay": 2.0,  # Add delay for replanning
                    "reason": f"dependency_on_{failed_agent}"
                }
        
        return recovery_plan


def validate_synchronization_points(coordinated_actions: Dict[str, Dict[str, Any]],
                                   synchronization_points: List[Dict[str, Any]]) -> bool:
    """
    Validate that synchronization points are achievable
    
    Args:
        coordinated_actions: Dictionary of coordinated actions
        synchronization_points: List of synchronization points
        
    Returns:
        True if all sync points are valid
    """
    for sync_point in synchronization_points:
        sync_time = sync_point["time"]
        required_agents = sync_point["agents"]
        
        # Check if each required agent can reach the sync point
        for agent_id in required_agents:
            if agent_id not in coordinated_actions:
                continue
            
            action_data = coordinated_actions[agent_id]
            action_duration = action_data["duration"]
            
            # Simple check: agent must not be busy at sync time
            if action_duration > sync_time:
                return False
    
    return True