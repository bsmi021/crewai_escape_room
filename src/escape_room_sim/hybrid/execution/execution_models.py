"""
Execution Pipeline Data Models

Agent C: Action Translation & Execution Specialist
Data models for execution planning, monitoring, and performance tracking.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Any, Optional, Union
from enum import Enum

from ..core_architecture import MesaAction
from ..actions.action_models import ExecutionResult


class ExecutionPhase(Enum):
    """Phases of action execution"""
    PLANNING = "planning"
    SCHEDULING = "scheduling" 
    EXECUTING = "executing"
    MONITORING = "monitoring"
    COMPLETING = "completing"
    FAILED = "failed"


class ExecutionPriority(Enum):
    """Priority levels for execution"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    BACKGROUND = "background"


@dataclass
class ExecutionPlan:
    """
    Plan for executing a set of Mesa actions
    
    Coordinates timing, dependencies, and resource allocation for action execution.
    """
    plan_id: str
    actions: List[MesaAction]
    execution_order: List[str] = field(default_factory=list)
    resource_allocation: Dict[str, Any] = field(default_factory=dict)
    time_constraints: Dict[str, float] = field(default_factory=dict)
    dependency_graph: Dict[str, List[str]] = field(default_factory=dict)
    created_timestamp: datetime = field(default_factory=datetime.now)
    estimated_total_duration: float = 0.0
    priority: str = ExecutionPriority.MEDIUM.value
    
    def __post_init__(self):
        """Calculate execution order and estimated duration"""
        if not self.execution_order:
            self.execution_order = self._calculate_execution_order()
        
        if not self.estimated_total_duration:
            self.estimated_total_duration = self._calculate_total_duration()
    
    def _calculate_execution_order(self) -> List[str]:
        """Calculate optimal execution order based on dependencies"""
        if not self.actions:
            return []
        
        # Simple ordering for now - can be enhanced with proper dependency resolution
        action_ids = []
        
        # Group by agent for coordinated execution
        agent_actions = {}
        for action in self.actions:
            if action.agent_id not in agent_actions:
                agent_actions[action.agent_id] = []
            agent_actions[action.agent_id].append(action)
        
        # Interleave actions from different agents to enable parallel execution
        max_actions = max(len(actions) for actions in agent_actions.values()) if agent_actions else 0
        
        for i in range(max_actions):
            for agent_id, actions in agent_actions.items():
                if i < len(actions):
                    action_id = f"{actions[i].agent_id}_{actions[i].action_type}_{id(actions[i])}"
                    action_ids.append(action_id)
        
        return action_ids
    
    def _calculate_total_duration(self) -> float:
        """Calculate total estimated duration"""
        if not self.actions:
            return 0.0
        
        # For parallel execution, take the maximum duration per agent
        agent_durations = {}
        for action in self.actions:
            if action.agent_id not in agent_durations:
                agent_durations[action.agent_id] = 0.0
            agent_durations[action.agent_id] += action.expected_duration
        
        # Add coordination overhead
        coordination_overhead = len(self.actions) * 0.1
        max_agent_duration = max(agent_durations.values()) if agent_durations else 0.0
        
        return max_agent_duration + coordination_overhead
    
    def get_next_action(self, completed_actions: List[str]) -> Optional[MesaAction]:
        """Get next action to execute based on execution plan"""
        for action_id in self.execution_order:
            if action_id not in completed_actions:
                # Find the corresponding action
                for action in self.actions:
                    current_id = f"{action.agent_id}_{action.action_type}_{id(action)}"
                    if current_id == action_id:
                        # Check dependencies
                        if self._dependencies_satisfied(action, completed_actions):
                            return action
                break
        return None
    
    def _dependencies_satisfied(self, action: MesaAction, completed_actions: List[str]) -> bool:
        """Check if action dependencies are satisfied"""
        dependencies = action.parameters.get('dependencies', [])
        return all(dep in completed_actions for dep in dependencies)
    
    def update_progress(self, completed_action_id: str, result: ExecutionResult):
        """Update plan progress with completed action"""
        # Could update resource allocation, time estimates, etc.
        # This is a placeholder for more sophisticated progress tracking
        pass


@dataclass 
class ExecutionState:
    """
    Current state of action execution system
    
    Tracks active executions, resource usage, and system health.
    """
    active_plans: Dict[str, ExecutionPlan] = field(default_factory=dict)
    executing_actions: Dict[str, MesaAction] = field(default_factory=dict)
    completed_actions: List[str] = field(default_factory=list)
    failed_actions: List[str] = field(default_factory=list)
    resource_usage: Dict[str, float] = field(default_factory=dict)
    system_load: float = 0.0
    last_update: datetime = field(default_factory=datetime.now)
    
    def add_plan(self, plan: ExecutionPlan):
        """Add execution plan to active plans"""
        self.active_plans[plan.plan_id] = plan
        self.last_update = datetime.now()
    
    def remove_plan(self, plan_id: str):
        """Remove completed execution plan"""
        if plan_id in self.active_plans:
            del self.active_plans[plan_id]
        self.last_update = datetime.now()
    
    def start_action_execution(self, action: MesaAction):
        """Mark action as starting execution"""
        action_id = f"{action.agent_id}_{action.action_type}_{id(action)}"
        self.executing_actions[action_id] = action
        self.last_update = datetime.now()
    
    def complete_action_execution(self, action_id: str, success: bool):
        """Mark action execution as completed"""
        if action_id in self.executing_actions:
            del self.executing_actions[action_id]
        
        if success:
            self.completed_actions.append(action_id)
        else:
            self.failed_actions.append(action_id)
        
        self.last_update = datetime.now()
    
    def calculate_system_load(self) -> float:
        """Calculate current system load"""
        # Simple load calculation based on active executions
        base_load = len(self.executing_actions) * 0.2
        plan_load = len(self.active_plans) * 0.1
        
        # Resource usage load
        resource_load = sum(self.resource_usage.values()) * 0.1
        
        total_load = min(1.0, base_load + plan_load + resource_load)
        self.system_load = total_load
        return total_load
    
    def get_status_summary(self) -> Dict[str, Any]:
        """Get execution state summary"""
        return {
            "active_plans": len(self.active_plans),
            "executing_actions": len(self.executing_actions),
            "completed_actions": len(self.completed_actions),
            "failed_actions": len(self.failed_actions),
            "system_load": self.calculate_system_load(),
            "success_rate": self._calculate_success_rate(),
            "last_update": self.last_update.isoformat()
        }
    
    def _calculate_success_rate(self) -> float:
        """Calculate overall success rate"""
        total_completed = len(self.completed_actions) + len(self.failed_actions)
        if total_completed == 0:
            return 1.0
        return len(self.completed_actions) / total_completed


@dataclass
class ActionResult:
    """
    Result of a single action execution
    
    Enhanced version of ExecutionResult with additional execution context.
    """
    action_id: str
    agent_id: str
    action_type: str
    execution_result: ExecutionResult
    mesa_model_state_before: Optional[Dict[str, Any]] = None
    mesa_model_state_after: Optional[Dict[str, Any]] = None
    execution_context: Dict[str, Any] = field(default_factory=dict)
    performance_impact: Dict[str, float] = field(default_factory=dict)
    
    def get_execution_summary(self) -> Dict[str, Any]:
        """Get summary of action execution"""
        return {
            "action_id": self.action_id,
            "agent_id": self.agent_id,
            "action_type": self.action_type,
            "success": self.execution_result.success,
            "duration": self.execution_result.actual_duration,
            "performance_impact": self.performance_impact,
            "timestamp": self.execution_result.execution_timestamp.isoformat()
        }
    
    def calculate_efficiency(self, expected_duration: float) -> float:
        """Calculate execution efficiency"""
        return self.execution_result.calculate_efficiency(expected_duration)


@dataclass
class PerformanceMetrics:
    """
    Performance metrics for execution system
    
    Tracks timing, throughput, and success rates across executions.
    """
    total_actions_executed: int = 0
    successful_actions: int = 0
    failed_actions: int = 0
    average_execution_time: float = 0.0
    peak_system_load: float = 0.0
    actions_per_second: float = 0.0
    conflict_resolution_time: float = 0.0
    memory_usage: float = 0.0
    start_time: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)
    
    def update_with_result(self, result: ActionResult):
        """Update metrics with new action result"""
        self.total_actions_executed += 1
        
        if result.execution_result.success:
            self.successful_actions += 1
        else:
            self.failed_actions += 1
        
        # Update average execution time
        new_duration = result.execution_result.actual_duration
        if self.total_actions_executed == 1:
            self.average_execution_time = new_duration
        else:
            # Running average
            weight = 1.0 / self.total_actions_executed
            self.average_execution_time = (
                (1 - weight) * self.average_execution_time + 
                weight * new_duration
            )
        
        # Update throughput
        elapsed_time = (datetime.now() - self.start_time).total_seconds()
        if elapsed_time > 0:
            self.actions_per_second = self.total_actions_executed / elapsed_time
        
        self.last_updated = datetime.now()
    
    def get_success_rate(self) -> float:
        """Get overall success rate"""
        if self.total_actions_executed == 0:
            return 1.0
        return self.successful_actions / self.total_actions_executed
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary"""
        return {
            "total_actions": self.total_actions_executed,
            "success_rate": self.get_success_rate(),
            "average_execution_time": self.average_execution_time,
            "actions_per_second": self.actions_per_second,
            "peak_system_load": self.peak_system_load,
            "conflict_resolution_time": self.conflict_resolution_time,
            "memory_usage_mb": self.memory_usage,
            "uptime_seconds": (self.last_updated - self.start_time).total_seconds()
        }
    
    def is_performance_degraded(self) -> bool:
        """Check if performance is degraded"""
        # Define performance thresholds
        if self.get_success_rate() < 0.85:  # Less than 85% success rate
            return True
        if self.average_execution_time > 10.0:  # Taking more than 10 seconds per action
            return True
        if self.peak_system_load > 0.9:  # System load over 90%
            return True
        return False
    
    def get_performance_recommendations(self) -> List[str]:
        """Get performance improvement recommendations"""
        recommendations = []
        
        if self.get_success_rate() < 0.9:
            recommendations.append("Investigate action failure patterns")
        
        if self.average_execution_time > 5.0:
            recommendations.append("Optimize action execution pipeline")
        
        if self.actions_per_second < 1.0:
            recommendations.append("Increase parallelization or reduce action complexity")
        
        if self.peak_system_load > 0.8:
            recommendations.append("Consider load balancing or resource scaling")
        
        if self.conflict_resolution_time > 1.0:
            recommendations.append("Optimize conflict resolution algorithms")
        
        return recommendations