"""
Execution Pipeline Implementation

Agent C: Action Translation & Execution Specialist
Executes Mesa actions with monitoring, performance tracking, and feedback loops.
"""

import time
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable
from abc import ABC, abstractmethod
import logging
import mesa
from collections import defaultdict

from ..core_architecture import MesaAction
from ..actions.action_models import ExecutionResult, ActionHandoff
from .execution_models import (
    ExecutionPlan, ExecutionState, ActionResult, PerformanceMetrics,
    ExecutionPhase, ExecutionPriority
)

logger = logging.getLogger(__name__)


class MesaActionExecutor:
    """
    Executes Mesa actions within the Mesa model environment
    
    Handles action execution, rollback, and integration with Mesa's step cycle.
    """
    
    def __init__(self, rollback_enabled: bool = True):
        self.rollback_enabled = rollback_enabled
        self.execution_history: List[ActionResult] = []
        self.rollback_stack: List[Dict[str, Any]] = []
        self.execution_callbacks: Dict[str, List[Callable]] = defaultdict(list)
        
    def execute_action(self, action: MesaAction, mesa_model: mesa.Model) -> ExecutionResult:
        """Execute single Mesa action and return result"""
        start_time = time.perf_counter()
        action_id = f"{action.agent_id}_{action.action_type}_{id(action)}"
        
        try:
            # Save state for potential rollback
            if self.rollback_enabled:
                model_state = self._capture_model_state(mesa_model)
                self.rollback_stack.append({
                    'action_id': action_id,
                    'model_state': model_state,
                    'timestamp': datetime.now()
                })
            
            # Execute action
            result_data = self._execute_mesa_action(action, mesa_model)
            
            # Calculate execution time
            end_time = time.perf_counter()
            actual_duration = end_time - start_time
            
            # Create execution result
            execution_result = ExecutionResult(
                action_id=action_id,
                agent_id=action.agent_id,
                action_type=action.action_type,
                status="completed",
                actual_duration=actual_duration,
                success=True,
                result_data=result_data,
                performance_metrics={
                    "execution_time_ms": actual_duration * 1000,
                    "model_step_impact": self._measure_model_impact(mesa_model)
                }
            )
            
            # Track execution
            self._track_execution(action, execution_result, mesa_model)
            
            # Execute callbacks
            self._execute_callbacks("action_completed", action, execution_result)
            
            return execution_result
            
        except Exception as e:
            # Handle execution failure
            end_time = time.perf_counter()
            actual_duration = end_time - start_time
            
            logger.error(f"Action execution failed: {action_id}, Error: {str(e)}")
            
            # Rollback if enabled
            if self.rollback_enabled:
                self._rollback_action(action_id, mesa_model)
            
            execution_result = ExecutionResult(
                action_id=action_id,
                agent_id=action.agent_id,  
                action_type=action.action_type,
                status="failed",
                actual_duration=actual_duration,
                success=False,
                error_message=str(e),
                performance_metrics={
                    "execution_time_ms": actual_duration * 1000,
                    "failure_type": type(e).__name__
                }
            )
            
            # Execute failure callbacks
            self._execute_callbacks("action_failed", action, execution_result)
            
            return execution_result
    
    def execute_action_batch(self, actions: List[MesaAction], 
                           mesa_model: mesa.Model) -> List[ExecutionResult]:
        """Execute multiple actions with coordination"""
        results = []
        
        # Group actions by agent for coordination
        agent_actions = defaultdict(list)
        for action in actions:
            agent_actions[action.agent_id].append(action)
        
        # Execute actions with coordination
        for agent_id, agent_action_list in agent_actions.items():
            for action in agent_action_list:
                result = self.execute_action(action, mesa_model)
                results.append(result)
                
                # If action failed and rollback is enabled, skip remaining actions for this agent
                if not result.success and self.rollback_enabled:
                    logger.warning(f"Skipping remaining actions for {agent_id} due to failure")
                    break
        
        return results
    
    def _execute_mesa_action(self, action: MesaAction, mesa_model: mesa.Model) -> Dict[str, Any]:
        """Execute the actual Mesa action"""
        result_data = {}
        
        # Find the agent in the model
        target_agent = None
        # Mesa 3.x compatibility: use agents attribute
        for agent in mesa_model.agents:
            agent_id = getattr(agent, 'agent_id', str(getattr(agent, 'unique_id', '')))
            if agent_id == action.agent_id:
                target_agent = agent
                break
        
        if not target_agent:
            raise ValueError(f"Agent {action.agent_id} not found in model")
        
        # Execute action based on type
        if action.action_type == 'move':
            result_data = self._execute_move_action(action, target_agent, mesa_model)
        elif action.action_type == 'examine':
            result_data = self._execute_examine_action(action, target_agent, mesa_model)
        elif action.action_type == 'communicate':
            result_data = self._execute_communicate_action(action, target_agent, mesa_model)
        elif action.action_type == 'claim_resource':
            result_data = self._execute_claim_resource_action(action, target_agent, mesa_model)
        elif action.action_type == 'use_tool':
            result_data = self._execute_use_tool_action(action, target_agent, mesa_model)
        else:
            # Generic action execution
            result_data = self._execute_generic_action(action, target_agent, mesa_model)
        
        return result_data
    
    def _execute_move_action(self, action: MesaAction, agent, mesa_model: mesa.Model) -> Dict[str, Any]:
        """Execute movement action"""
        target_pos = action.parameters.get('target_position')
        if not target_pos:
            raise ValueError("Move action missing target_position")
        
        old_pos = getattr(agent, 'pos', (0, 0))
        
        # Update agent position
        if hasattr(mesa_model, 'grid') and hasattr(agent, 'pos'):
            try:
                mesa_model.grid.move_agent(agent, target_pos)
            except:
                # Fallback position update
                agent.pos = target_pos
        else:
            agent.pos = target_pos
        
        return {
            "old_position": old_pos,
            "new_position": target_pos,
            "distance_moved": self._calculate_distance(old_pos, target_pos)
        }
    
    def _execute_examine_action(self, action: MesaAction, agent, mesa_model: mesa.Model) -> Dict[str, Any]:
        """Execute examination action"""
        target = action.parameters.get('target', 'environment')
        detail_level = action.parameters.get('detail_level', 'medium')
        
        examination_results = {
            "target": target,
            "detail_level": detail_level,
            "findings": []
        }
        
        # Simulate examination based on target
        if target == 'environment':
            examination_results["findings"] = [
                "Room layout analyzed",
                "Potential exits identified", 
                "Objects of interest catalogued"
            ]
        elif target == 'door':
            examination_results["findings"] = [
                "Door material: wood",
                "Lock type: standard key lock",
                "Condition: locked"
            ]
        else:
            examination_results["findings"] = [f"Examined {target}"]
        
        return examination_results
    
    def _execute_communicate_action(self, action: MesaAction, agent, mesa_model: mesa.Model) -> Dict[str, Any]:
        """Execute communication action"""
        target = action.parameters.get('target', 'broadcast')
        message = action.parameters.get('message', 'status_update')
        
        # Log communication
        communication_result = {
            "sender": action.agent_id,
            "target": target,
            "message_type": message,
            "timestamp": datetime.now().isoformat(),
            "delivery_status": "sent"
        }
        
        # If model has communication system, use it
        if hasattr(mesa_model, 'communication_log'):
            mesa_model.communication_log.append(communication_result)
        
        return communication_result
    
    def _execute_claim_resource_action(self, action: MesaAction, agent, mesa_model: mesa.Model) -> Dict[str, Any]:
        """Execute resource claim action"""
        resource_id = action.parameters.get('resource_id')
        if not resource_id:
            raise ValueError("Claim resource action missing resource_id")
        
        # Check if model has resource manager
        if hasattr(mesa_model, 'resource_manager'):
            try:
                success = mesa_model.resource_manager.claim_resource(action.agent_id, resource_id)
                return {
                    "resource_id": resource_id,
                    "claim_success": success,
                    "agent_id": action.agent_id
                }
            except Exception as e:
                raise ValueError(f"Resource claim failed: {str(e)}")
        
        # Fallback: update agent resources directly
        agent_resources = getattr(agent, 'resources', [])
        if resource_id not in agent_resources:
            agent_resources.append(resource_id)
            agent.resources = agent_resources
        
        return {
            "resource_id": resource_id,
            "claim_success": True,
            "agent_id": action.agent_id
        }
    
    def _execute_use_tool_action(self, action: MesaAction, agent, mesa_model: mesa.Model) -> Dict[str, Any]:
        """Execute tool use action"""
        tool_id = action.parameters.get('tool_id')
        if not tool_id:
            raise ValueError("Use tool action missing tool_id")
        
        # Check if agent has the tool
        agent_resources = getattr(agent, 'resources', [])
        if tool_id not in agent_resources:
            raise ValueError(f"Agent {action.agent_id} does not have tool {tool_id}")
        
        # Simulate tool usage
        tool_result = {
            "tool_id": tool_id,
            "usage_success": True,
            "tool_effect": f"Used {tool_id} successfully",
            "durability_impact": 0.1  # Tool degrades slightly
        }
        
        return tool_result
    
    def _execute_generic_action(self, action: MesaAction, agent, mesa_model: mesa.Model) -> Dict[str, Any]:
        """Execute generic/unknown action type"""
        return {
            "action_type": action.action_type,
            "parameters": action.parameters,
            "execution_method": "generic",
            "success": True
        }
    
    def _capture_model_state(self, mesa_model: mesa.Model) -> Dict[str, Any]:
        """Capture current model state for rollback"""
        state = {
            "agent_positions": {},
            "agent_resources": {},
            "model_step": getattr(mesa_model, 'model_step_count', 0),
            "timestamp": datetime.now()
        }
        
        # Capture agent states
        # Mesa 3.x compatibility: check for agents attribute
        if hasattr(mesa_model, 'agents'):
            # Mesa 3.x compatibility: use agents attribute
            for agent in mesa_model.agents:
                agent_id = getattr(agent, 'agent_id', str(getattr(agent, 'unique_id', id(agent))))
                state["agent_positions"][agent_id] = getattr(agent, 'pos', None)
                state["agent_resources"][agent_id] = getattr(agent, 'resources', []).copy()
        
        return state
    
    def _rollback_action(self, action_id: str, mesa_model: mesa.Model):
        """Rollback the effects of a failed action"""
        # Find the rollback state for this action
        rollback_state = None
        for i, rollback_entry in enumerate(reversed(self.rollback_stack)):
            if rollback_entry['action_id'] == action_id:
                rollback_state = rollback_entry['model_state']
                # Remove this and all subsequent rollback entries
                self.rollback_stack = self.rollback_stack[:-i-1]
                break
        
        if not rollback_state:
            logger.warning(f"No rollback state found for action {action_id}")
            return
        
        # Restore model state
        try:
            # Mesa 3.x compatibility: check for agents attribute
            if hasattr(mesa_model, 'agents'):
                # Mesa 3.x compatibility: use agents attribute
                for agent in mesa_model.agents:
                    agent_id = getattr(agent, 'agent_id', str(getattr(agent, 'unique_id', id(agent))))
                    
                    # Restore position
                    if agent_id in rollback_state["agent_positions"]:
                        old_pos = rollback_state["agent_positions"][agent_id]
                        if old_pos and hasattr(agent, 'pos'):
                            agent.pos = old_pos
                    
                    # Restore resources
                    if agent_id in rollback_state["agent_resources"]:
                        old_resources = rollback_state["agent_resources"][agent_id]
                        agent.resources = old_resources.copy()
            
            logger.info(f"Successfully rolled back action {action_id}")
            
        except Exception as e:
            logger.error(f"Rollback failed for action {action_id}: {str(e)}")
    
    def _measure_model_impact(self, mesa_model: mesa.Model) -> float:
        """Measure impact of action on model state"""
        # Simple metric - could be enhanced
        return 1.0  # Placeholder
    
    def _calculate_distance(self, pos1, pos2) -> float:
        """Calculate distance between positions"""
        if not pos1 or not pos2:
            return 0.0
        return ((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2) ** 0.5
    
    def _track_execution(self, action: MesaAction, result: ExecutionResult, mesa_model: mesa.Model):
        """Track action execution for analysis"""
        action_result = ActionResult(
            action_id=result.action_id,
            agent_id=action.agent_id,
            action_type=action.action_type,
            execution_result=result,
            execution_context={
                "model_step": getattr(mesa_model, 'model_step_count', 0),
                "execution_timestamp": datetime.now().isoformat()
            }
        )
        
        self.execution_history.append(action_result)
        
        # Keep only recent history (last 1000 executions)
        if len(self.execution_history) > 1000:
            self.execution_history = self.execution_history[-1000:]
    
    def add_execution_callback(self, event_type: str, callback: Callable):
        """Add callback for execution events"""
        self.execution_callbacks[event_type].append(callback)
    
    def _execute_callbacks(self, event_type: str, action: MesaAction, result: ExecutionResult):
        """Execute callbacks for event type"""
        for callback in self.execution_callbacks.get(event_type, []):
            try:
                callback(action, result)
            except Exception as e:
                logger.error(f"Callback execution failed: {str(e)}")


class ExecutionMonitor:
    """
    Monitors action execution and system health
    
    Tracks performance, detects issues, and provides real-time feedback.
    """
    
    def __init__(self, monitoring_interval: float = 1.0):
        self.monitoring_interval = monitoring_interval
        self.execution_state = ExecutionState()
        self.performance_metrics = PerformanceMetrics()
        self.alerts: List[Dict[str, Any]] = []
        self.monitoring_active = False
        
        # Thresholds for alerts
        self.alert_thresholds = {
            "max_execution_time": 10.0,
            "min_success_rate": 0.85,
            "max_system_load": 0.9,
            "max_failed_actions": 10
        }
    
    def start_monitoring(self):
        """Start execution monitoring"""
        self.monitoring_active = True
        logger.info("Execution monitoring started")
    
    def stop_monitoring(self):
        """Stop execution monitoring"""
        self.monitoring_active = False
        logger.info("Execution monitoring stopped")
    
    def update_execution_state(self, plan: ExecutionPlan, action: MesaAction, result: ExecutionResult):
        """Update execution state with new result"""
        # Update execution state
        action_id = result.action_id
        
        if result.success:
            self.execution_state.complete_action_execution(action_id, True)
        else:
            self.execution_state.complete_action_execution(action_id, False)
        
        # Update performance metrics
        action_result = ActionResult(
            action_id=action_id,
            agent_id=action.agent_id,
            action_type=action.action_type,
            execution_result=result
        )
        
        self.performance_metrics.update_with_result(action_result)
        
        # Check for alerts
        self._check_alerts()
        
        # Update system load
        self.execution_state.calculate_system_load()
    
    def _check_alerts(self):
        """Check for performance alerts"""
        current_time = datetime.now()
        
        # Check execution time alert
        if self.performance_metrics.average_execution_time > self.alert_thresholds["max_execution_time"]:
            self._create_alert("high_execution_time", 
                             f"Average execution time {self.performance_metrics.average_execution_time:.2f}s exceeds threshold")
        
        # Check success rate alert  
        success_rate = self.performance_metrics.get_success_rate()
        if success_rate < self.alert_thresholds["min_success_rate"]:
            self._create_alert("low_success_rate",
                             f"Success rate {success_rate:.1%} below threshold {self.alert_thresholds['min_success_rate']:.1%}")
        
        # Check system load alert
        if self.execution_state.system_load > self.alert_thresholds["max_system_load"]:
            self._create_alert("high_system_load",
                             f"System load {self.execution_state.system_load:.1%} exceeds threshold")
        
        # Check failed actions alert
        if len(self.execution_state.failed_actions) > self.alert_thresholds["max_failed_actions"]:
            self._create_alert("too_many_failures",
                             f"Failed actions {len(self.execution_state.failed_actions)} exceeds threshold")
    
    def _create_alert(self, alert_type: str, message: str):
        """Create performance alert"""
        alert = {
            "type": alert_type,
            "message": message,
            "timestamp": datetime.now(),
            "severity": "warning"
        }
        
        # Avoid duplicate alerts
        recent_alerts = [a for a in self.alerts if (datetime.now() - a["timestamp"]).seconds < 60]
        if not any(a["type"] == alert_type for a in recent_alerts):
            self.alerts.append(alert)
            logger.warning(f"Performance alert: {alert_type} - {message}")
    
    def get_monitoring_report(self) -> Dict[str, Any]:
        """Get comprehensive monitoring report"""
        return {
            "execution_state": self.execution_state.get_status_summary(),
            "performance_metrics": self.performance_metrics.get_performance_summary(),
            "active_alerts": len([a for a in self.alerts if (datetime.now() - a["timestamp"]).seconds < 300]),
            "recent_alerts": self.alerts[-10:],  # Last 10 alerts
            "system_health": self._assess_system_health(),
            "recommendations": self.performance_metrics.get_performance_recommendations()
        }
    
    def _assess_system_health(self) -> str:
        """Assess overall system health"""
        if self.performance_metrics.is_performance_degraded():
            return "degraded"
        elif len(self.alerts) > 5:
            return "warning"
        else:
            return "healthy"


class PerformanceTracker:
    """
    Tracks performance metrics and provides optimization insights
    
    Analyzes execution patterns and suggests improvements.
    """
    
    def __init__(self):
        self.metrics_history: List[PerformanceMetrics] = []
        self.execution_patterns: Dict[str, List[float]] = defaultdict(list)
        self.optimization_suggestions: List[str] = []
    
    def track_performance(self, metrics: PerformanceMetrics):
        """Track performance metrics over time"""
        self.metrics_history.append(metrics)
        
        # Keep rolling window of metrics
        if len(self.metrics_history) > 100:
            self.metrics_history = self.metrics_history[-100:]
        
        # Update execution patterns
        self.execution_patterns["success_rate"].append(metrics.get_success_rate())
        self.execution_patterns["avg_execution_time"].append(metrics.average_execution_time)
        self.execution_patterns["actions_per_second"].append(metrics.actions_per_second)
        
        # Generate optimization suggestions
        self._update_optimization_suggestions()
    
    def _update_optimization_suggestions(self):
        """Update optimization suggestions based on patterns"""
        suggestions = []
        
        if len(self.execution_patterns["success_rate"]) >= 10:
            recent_success_rates = self.execution_patterns["success_rate"][-10:]
            if all(rate < 0.9 for rate in recent_success_rates):
                suggestions.append("Consistently low success rate - review action validation logic")
        
        if len(self.execution_patterns["avg_execution_time"]) >= 10:
            recent_times = self.execution_patterns["avg_execution_time"][-10:]
            if all(time > 5.0 for time in recent_times):
                suggestions.append("High execution times - consider action optimization or parallelization")
        
        self.optimization_suggestions = suggestions
    
    def get_performance_trends(self) -> Dict[str, Any]:
        """Get performance trends analysis"""
        if len(self.metrics_history) < 2:
            return {"trend_analysis": "Insufficient data for trend analysis"}
        
        current = self.metrics_history[-1]
        previous = self.metrics_history[-2]
        
        trends = {
            "success_rate_trend": current.get_success_rate() - previous.get_success_rate(),
            "execution_time_trend": current.average_execution_time - previous.average_execution_time,
            "throughput_trend": current.actions_per_second - previous.actions_per_second,
            "optimization_suggestions": self.optimization_suggestions
        }
        
        return trends


class FeedbackProcessor:
    """
    Processes execution feedback and provides learning insights
    
    Analyzes execution results to improve future performance.
    """
    
    def __init__(self):
        self.feedback_data: List[Dict[str, Any]] = []
        self.learning_insights: Dict[str, Any] = {}
        self.improvement_patterns: Dict[str, List[float]] = defaultdict(list)
    
    def process_execution_feedback(self, handoff: ActionHandoff):
        """Process execution feedback from action handoff"""
        feedback_entry = {
            "timestamp": handoff.action_timestamp,
            "total_actions": len(handoff.actions),
            "success_rate": handoff.get_performance_summary()["success_rate"],
            "conflicts_resolved": len(handoff.conflict_resolutions),
            "performance_metrics": handoff.performance_metrics
        }
        
        self.feedback_data.append(feedback_entry)
        self._update_learning_insights(feedback_entry)
    
    def _update_learning_insights(self, feedback: Dict[str, Any]):
        """Update learning insights based on feedback"""
        # Track improvement patterns
        self.improvement_patterns["success_rate"].append(feedback["success_rate"])
        self.improvement_patterns["conflict_resolution"].append(feedback["conflicts_resolved"])
        
        # Generate insights
        self.learning_insights = {
            "average_success_rate": sum(self.improvement_patterns["success_rate"]) / len(self.improvement_patterns["success_rate"]),
            "conflict_resolution_efficiency": sum(self.improvement_patterns["conflict_resolution"]) / len(self.improvement_patterns["conflict_resolution"]),
            "total_feedback_sessions": len(self.feedback_data),
            "last_updated": datetime.now()
        }
    
    def get_learning_recommendations(self) -> List[str]:
        """Get recommendations based on learning insights"""
        recommendations = []
        
        if self.learning_insights.get("average_success_rate", 1.0) < 0.85:
            recommendations.append("Focus on improving action success rates through better validation")
        
        if self.learning_insights.get("conflict_resolution_efficiency", 0) < 0.5:
            recommendations.append("Optimize conflict resolution strategies for better efficiency")
        
        return recommendations


# Mock class for testing
class Mock:
    def __init__(self):
        self.steps = 0
    
    def __getattr__(self, name):
        return Mock()
    
    def __call__(self, *args, **kwargs):
        return Mock()