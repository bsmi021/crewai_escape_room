"""
Action Translation Data Models

Agent C: Action Translation & Execution Specialist
Data models for action sequences, conflicts, and execution results.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Any, Optional, Union
from enum import Enum

from ..core_architecture import DecisionData, MesaAction


class SequenceType(Enum):
    """Types of action sequences"""
    SINGLE_STEP = "single_step"
    MULTI_STEP = "multi_step"
    CONDITIONAL = "conditional"
    PARALLEL = "parallel"
    COORDINATED = "coordinated"


class ConflictType(Enum):
    """Types of action conflicts"""
    RESOURCE_COMPETITION = "resource_competition"
    SPATIAL_COLLISION = "spatial_collision"
    TEMPORAL_OVERLAP = "temporal_overlap"
    DEPENDENCY_VIOLATION = "dependency_violation"
    PRIORITY_CONFLICT = "priority_conflict"


class ConflictSeverity(Enum):
    """Severity levels for conflicts"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ResolutionType(Enum):
    """Types of conflict resolution strategies"""
    PRIORITY_BASED = "priority_based"
    TRUST_BASED = "trust_based"
    TEMPORAL_SEQUENCING = "temporal_sequencing"
    RESOURCE_SHARING = "resource_sharing"
    ALTERNATIVE_ACTION = "alternative_action"
    NEGOTIATED = "negotiated"


class ExecutionStatus(Enum):
    """Execution status for actions"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"  
    FAILED = "failed"
    CANCELLED = "cancelled"
    WAITING_DEPENDENCY = "waiting_dependency"


@dataclass
class ActionSequence:
    """
    Represents a sequence of related actions for a single agent
    
    Supports multi-step sequences with dependencies and conditional execution.
    """
    sequence_id: str
    agent_id: str
    decisions: List[DecisionData]
    sequence_type: str
    dependencies: List[str] = field(default_factory=list)
    created_timestamp: datetime = field(default_factory=datetime.now)
    estimated_duration: float = 0.0
    priority: float = 0.5
    conditions: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Calculate estimated duration from decisions"""
        if not self.estimated_duration:
            # Estimate based on action types and complexity
            base_durations = {
                "move": 1.0,
                "examine": 2.0, 
                "analyze": 3.0,
                "communicate": 1.5,
                "use_tool": 2.5,
                "solve_puzzle": 5.0,
                "claim_resource": 1.0
            }
            
            total_duration = 0.0
            for decision in self.decisions:
                action_type = decision.chosen_action
                base_duration = base_durations.get(action_type, 2.0)
                
                # Adjust for complexity
                complexity_factor = len(decision.action_parameters) * 0.1
                duration = base_duration * (1.0 + complexity_factor)
                total_duration += duration
            
            # Add sequence overhead for multi-step
            if self.sequence_type in ["multi_step", "conditional"]:
                total_duration *= 1.2
            elif self.sequence_type == "coordinated":
                total_duration *= 1.5
                
            self.estimated_duration = total_duration
    
    def validate_sequence(self) -> bool:
        """Validate sequence integrity"""
        try:
            # Check basic requirements
            if not self.sequence_id or not self.agent_id:
                return False
            
            if not self.decisions:
                return False
                
            # Check agent consistency
            for decision in self.decisions:
                if decision.agent_id != self.agent_id:
                    return False
                    
            # Check sequence type validity
            valid_types = [t.value for t in SequenceType]
            if self.sequence_type not in valid_types:
                return False
                
            # Check dependencies format
            for dep in self.dependencies:
                if not isinstance(dep, str):
                    return False
                    
            return True
            
        except Exception:
            return False
    
    def get_next_decision(self, completed_actions: List[str]) -> Optional[DecisionData]:
        """Get next decision to execute based on completed actions"""
        if self.sequence_type == "single_step":
            return self.decisions[0] if self.decisions else None
        
        elif self.sequence_type == "multi_step":
            # Execute in order
            for i, decision in enumerate(self.decisions):
                decision_id = f"{decision.agent_id}_{decision.chosen_action}_{i}"
                if decision_id not in completed_actions:
                    return decision
            return None
            
        elif self.sequence_type == "conditional":
            # Check conditions for each decision
            for decision in self.decisions:
                if self._check_conditions(decision, completed_actions):
                    decision_id = f"{decision.agent_id}_{decision.chosen_action}"
                    if decision_id not in completed_actions:
                        return decision
            return None
            
        return None
    
    def _check_conditions(self, decision: DecisionData, completed_actions: List[str]) -> bool:
        """Check if conditions are met for a conditional decision"""
        # Simple condition checking - can be enhanced
        action_conditions = self.conditions.get(decision.chosen_action, {})
        
        # Check required previous actions
        required_actions = action_conditions.get("requires", [])
        for required in required_actions:
            if required not in completed_actions:
                return False
                
        return True


@dataclass  
class ActionConflict:
    """
    Represents a conflict between multiple actions
    
    Tracks conflicts for resolution and provides analysis capabilities.
    """
    conflict_id: str
    conflict_type: str
    conflicting_actions: List[MesaAction]
    resource_contested: Optional[str] = None
    severity: str = "medium"
    detected_timestamp: datetime = field(default_factory=datetime.now)
    agents_involved: List[str] = field(default_factory=list)
    conflict_metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Extract agents involved from conflicting actions"""
        if not self.agents_involved:
            self.agents_involved = list(set(
                action.agent_id for action in self.conflicting_actions
            ))
    
    def analyze_conflict(self) -> Dict[str, Any]:
        """Analyze conflict for resolution planning"""
        analysis = {
            "conflict_id": self.conflict_id,
            "conflict_type": self.conflict_type,
            "agents_involved": self.agents_involved,
            "severity": self.severity,
            "resource_type": self.resource_contested,
            "action_types": [action.action_type for action in self.conflicting_actions],
            "priority_scores": [],
            "timing_analysis": {},
            "resolution_suggestions": []
        }
        
        # Analyze priorities
        for action in self.conflicting_actions:
            priority = action.parameters.get("priority", 0.5)
            analysis["priority_scores"].append({
                "agent_id": action.agent_id,
                "priority": priority
            })
        
        # Timing analysis
        if len(self.conflicting_actions) >= 2:
            durations = [action.expected_duration for action in self.conflicting_actions]
            analysis["timing_analysis"] = {
                "min_duration": min(durations),
                "max_duration": max(durations),
                "avg_duration": sum(durations) / len(durations),
                "total_duration": sum(durations)
            }
        
        # Resolution suggestions
        if self.conflict_type == "resource_competition":
            analysis["resolution_suggestions"].extend([
                "priority_based", "temporal_sequencing", "resource_sharing"
            ])
        elif self.conflict_type == "spatial_collision":
            analysis["resolution_suggestions"].extend([
                "temporal_sequencing", "alternative_path", "coordination"
            ])
        elif self.conflict_type == "temporal_overlap":
            analysis["resolution_suggestions"].extend([
                "time_scheduling", "priority_based", "parallel_execution"
            ])
        
        return analysis
    
    def get_conflict_severity_score(self) -> float:
        """Get numeric severity score (0.0 to 1.0)"""
        severity_scores = {
            "low": 0.25,
            "medium": 0.5,
            "high": 0.75,
            "critical": 1.0
        }
        return severity_scores.get(self.severity, 0.5)


@dataclass
class ConflictResolution:
    """
    Represents the resolution of an action conflict
    
    Contains resolved actions and resolution metadata.
    """
    conflict_id: str
    resolution_type: str
    resolved_actions: List[MesaAction]
    rejected_actions: List[MesaAction] = field(default_factory=list)
    resolution_timestamp: datetime = field(default_factory=datetime.now)
    success: bool = True
    resolution_reasoning: str = ""
    confidence: float = 0.8
    alternative_actions: List[MesaAction] = field(default_factory=list)
    resolution_metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get_resolution_summary(self) -> Dict[str, Any]:
        """Get summary of resolution outcome"""
        return {
            "conflict_id": self.conflict_id,
            "resolution_type": self.resolution_type,
            "success": self.success,
            "confidence": self.confidence,
            "actions_resolved": len(self.resolved_actions),
            "actions_rejected": len(self.rejected_actions),
            "agents_affected": list(set(
                action.agent_id for action in (self.resolved_actions + self.rejected_actions)
            )),
            "resolution_reasoning": self.resolution_reasoning,
            "timestamp": self.resolution_timestamp.isoformat()
        }
    
    def validate_resolution(self) -> bool:
        """Validate resolution integrity"""
        try:
            # Must have at least one resolved action
            if not self.resolved_actions:
                return False
                
            # Resolution type must be valid
            valid_types = [t.value for t in ResolutionType]
            if self.resolution_type not in valid_types:
                return False
                
            # Confidence must be in valid range
            if not (0.0 <= self.confidence <= 1.0):
                return False
                
            # All actions must be valid MesaAction objects
            all_actions = self.resolved_actions + self.rejected_actions + self.alternative_actions
            for action in all_actions:
                if not isinstance(action, MesaAction):
                    return False
                if not action.agent_id or not action.action_type:
                    return False
                    
            return True
            
        except Exception:
            return False


@dataclass
class ExecutionResult:
    """
    Result of action execution in Mesa model
    
    Tracks execution outcome and performance metrics.
    """
    action_id: str
    agent_id: str
    action_type: str
    status: str
    execution_timestamp: datetime = field(default_factory=datetime.now)
    actual_duration: float = 0.0
    success: bool = False
    result_data: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    side_effects: List[Dict[str, Any]] = field(default_factory=list)
    
    def __post_init__(self):
        """Set success based on status"""
        if not hasattr(self, '_success_set'):
            self.success = self.status == "completed"
            self._success_set = True
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary for monitoring"""
        return {
            "action_id": self.action_id,
            "agent_id": self.agent_id,
            "action_type": self.action_type,
            "success": self.success,
            "actual_duration": self.actual_duration,
            "status": self.status,
            "performance_metrics": self.performance_metrics,
            "side_effects_count": len(self.side_effects),
            "execution_timestamp": self.execution_timestamp.isoformat()
        }
    
    def calculate_efficiency(self, expected_duration: float) -> float:
        """Calculate execution efficiency compared to expected duration"""
        if expected_duration <= 0:
            return 1.0
            
        if self.actual_duration <= 0:
            return 0.0
            
        # Efficiency = expected / actual (higher is better)
        # Cap at 2.0 to handle cases where execution was much faster than expected
        return min(2.0, expected_duration / self.actual_duration)
    
    def add_side_effect(self, effect_type: str, description: str, impact: str = "minor"):
        """Add a side effect of the action execution"""
        side_effect = {
            "type": effect_type,
            "description": description,
            "impact": impact,
            "timestamp": datetime.now().isoformat()
        }
        self.side_effects.append(side_effect)


@dataclass
class ActionHandoff:
    """
    Handoff data structure from Agent C to Agent D (State Management)
    
    Contains executed actions, results, and performance metrics for state updates.
    """
    actions: List[MesaAction]
    execution_results: Dict[str, ExecutionResult]
    conflict_resolutions: List[ConflictResolution]
    action_timestamp: datetime = field(default_factory=datetime.now)
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    failed_actions: List[MesaAction] = field(default_factory=list)
    rollback_actions: List[str] = field(default_factory=list)
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary for Agent D"""
        total_actions = len(self.actions)
        successful_actions = sum(1 for result in self.execution_results.values() if result.success)
        failed_actions = len(self.failed_actions)
        
        return {
            "total_actions": total_actions,
            "successful_actions": successful_actions,
            "failed_actions": failed_actions,
            "success_rate": successful_actions / total_actions if total_actions > 0 else 0.0,
            "conflicts_resolved": len(self.conflict_resolutions),
            "rollbacks_required": len(self.rollback_actions),
            "performance_metrics": self.performance_metrics,
            "timestamp": self.action_timestamp.isoformat()
        }
    
    def validate_handoff(self) -> bool:
        """Validate handoff data integrity"""
        try:
            # Must have actions
            if not self.actions:
                return False
                
            # Execution results should match actions
            action_ids = {f"{action.agent_id}_{action.action_type}_{id(action)}" for action in self.actions}
            result_ids = set(self.execution_results.keys())
            
            # Not all actions may have results yet (some may be pending)
            # But all results should correspond to actions
            for result_id in result_ids:
                # Basic validation - result should have valid format
                if not result_id or not isinstance(result_id, str):
                    return False
            
            # Validate conflict resolutions
            for resolution in self.conflict_resolutions:
                if not resolution.validate_resolution():
                    return False
            
            # Validate failed actions are valid MesaAction objects
            for failed_action in self.failed_actions:
                if not isinstance(failed_action, MesaAction):
                    return False
                    
            return True
            
        except Exception:
            return False