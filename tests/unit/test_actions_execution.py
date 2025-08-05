"""
Test Suite for Agent C: Action Translation & Execution Specialist
Execution Pipeline Tests

Tests the execution monitoring, performance tracking, and feedback loops.
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, MagicMock, patch
from typing import Dict, List, Any
import mesa

from src.escape_room_sim.hybrid.core_architecture import MesaAction
from src.escape_room_sim.hybrid.actions.action_models import ExecutionResult, ActionHandoff
from src.escape_room_sim.hybrid.execution.execution_models import (
    ExecutionPlan, ExecutionState, ActionResult, PerformanceMetrics
)
from src.escape_room_sim.hybrid.execution.execution_pipeline import (
    MesaActionExecutor, ExecutionMonitor, PerformanceTracker, FeedbackProcessor
)


class TestExecutionPlan:
    """Test ExecutionPlan data model"""
    
    def test_execution_plan_creation(self):
        """Test creation of ExecutionPlan"""
        actions = [
            MesaAction(
                agent_id="strategist",
                action_type="move",
                parameters={"target_position": (5, 5)},
                expected_duration=1.0,
                prerequisites=[]
            ),
            MesaAction(
                agent_id="strategist",
                action_type="examine",
                parameters={"target": "door"},
                expected_duration=2.0,
                prerequisites=[]
            )
        ]
        
        plan = ExecutionPlan(
            plan_id="plan_001",
            actions=actions
        )
        
        assert plan.plan_id == "plan_001"
        assert len(plan.actions) == 2
        assert len(plan.execution_order) > 0
        assert plan.estimated_total_duration > 0
        
    def test_execution_plan_ordering(self):
        """Test execution order calculation"""
        actions = [
            MesaAction(
                agent_id="strategist",
                action_type="move",
                parameters={"target_position": (5, 5)},
                expected_duration=1.0,
                prerequisites=[]
            ),
            MesaAction(
                agent_id="mediator",
                action_type="communicate",
                parameters={"target": "broadcast"},
                expected_duration=1.5,
                prerequisites=[]
            )
        ]
        
        plan = ExecutionPlan(
            plan_id="plan_002",
            actions=actions
        )
        
        # Should have execution order for both actions
        assert len(plan.execution_order) == 2
        
        # Get next action
        next_action = plan.get_next_action([])
        assert next_action is not None
        assert next_action.agent_id in ["strategist", "mediator"]
        
    def test_execution_plan_dependencies(self):
        """Test dependency handling in execution plan"""
        actions = [
            MesaAction(
                agent_id="strategist",
                action_type="move",
                parameters={"target_position": (3, 3)},
                expected_duration=1.0,
                prerequisites=[]
            ),
            MesaAction(
                agent_id="strategist",
                action_type="examine",
                parameters={"target": "door", "dependencies": ["move_complete"]},
                expected_duration=2.0,
                prerequisites=[]
            )
        ]
        
        plan = ExecutionPlan(
            plan_id="plan_003",
            actions=actions
        )
        
        # First action should be available
        first_action = plan.get_next_action([])
        assert first_action is not None
        assert first_action.action_type == "move"
        
        # Second action should require dependency completion
        # (This would be more complex in real implementation)


class TestExecutionState:
    """Test ExecutionState data model"""
    
    def test_execution_state_creation(self):
        """Test creation of ExecutionState"""
        state = ExecutionState()
        
        assert len(state.active_plans) == 0
        assert len(state.executing_actions) == 0
        assert len(state.completed_actions) == 0
        assert state.system_load == 0.0
        
    def test_execution_state_plan_management(self):
        """Test plan management in execution state"""
        state = ExecutionState()
        
        plan = ExecutionPlan(
            plan_id="test_plan",
            actions=[
                MesaAction(
                    agent_id="strategist",
                    action_type="move",
                    parameters={"target_position": (5, 5)},
                    expected_duration=1.0,
                    prerequisites=[]
                )
            ]
        )
        
        # Add plan
        state.add_plan(plan)
        assert "test_plan" in state.active_plans
        assert len(state.active_plans) == 1
        
        # Remove plan
        state.remove_plan("test_plan")
        assert "test_plan" not in state.active_plans
        assert len(state.active_plans) == 0
        
    def test_execution_state_action_tracking(self):
        """Test action execution tracking"""
        state = ExecutionState()
        
        action = MesaAction(
            agent_id="strategist",
            action_type="move",
            parameters={"target_position": (5, 5)},
            expected_duration=1.0,
            prerequisites=[]
        )
        
        # Start action execution
        state.start_action_execution(action)
        action_id = f"strategist_move_{id(action)}"
        assert action_id in state.executing_actions
        
        # Complete action execution
        state.complete_action_execution(action_id, True)
        assert action_id not in state.executing_actions
        assert action_id in state.completed_actions
        
    def test_execution_state_system_load(self):
        """Test system load calculation"""
        state = ExecutionState()
        
        # Add some executing actions
        for i in range(3):
            action = MesaAction(
                agent_id=f"agent_{i}",
                action_type="move",
                parameters={"target_position": (i, i)},
                expected_duration=1.0,
                prerequisites=[]
            )
            state.start_action_execution(action)
        
        # Calculate system load
        load = state.calculate_system_load()
        assert load > 0.0
        assert load <= 1.0
        assert state.system_load == load


class TestMesaActionExecutor:
    """Test MesaActionExecutor class"""
    
    @pytest.fixture
    def executor(self):
        """Create executor instance"""
        return MesaActionExecutor(rollback_enabled=True)
    
    @pytest.fixture
    def mock_mesa_model(self):
        """Create mock Mesa model"""
        model = Mock(spec=mesa.Model)
        model.schedule = Mock()
        model.grid = Mock()
        model.grid.width = 10
        model.grid.height = 10
        
        # Mock agent
        mock_agent = Mock()
        mock_agent.agent_id = "strategist"
        mock_agent.pos = (5, 5)
        mock_agent.resources = []
        mock_agent.energy = 1.0
        
        model.schedule.agents = [mock_agent]
        model.schedule.steps = 0
        
        return model
    
    def test_executor_initialization(self, executor):
        """Test executor initialization"""
        assert executor is not None
        assert executor.rollback_enabled is True
        assert len(executor.execution_history) == 0
        assert len(executor.rollback_stack) == 0
        
    def test_execute_move_action(self, executor, mock_mesa_model):
        """Test execution of move action"""
        action = MesaAction(
            agent_id="strategist",
            action_type="move",
            parameters={"target_position": (7, 7)},
            expected_duration=1.0,
            prerequisites=[]
        )
        
        result = executor.execute_action(action, mock_mesa_model)
        
        assert isinstance(result, ExecutionResult)
        assert result.success is True
        assert result.action_type == "move"
        assert result.agent_id == "strategist"
        assert result.actual_duration > 0
        
    def test_execute_examine_action(self, executor, mock_mesa_model):
        """Test execution of examine action"""
        action = MesaAction(
            agent_id="strategist",
            action_type="examine",
            parameters={"target": "door", "detail_level": "high"},
            expected_duration=2.0,
            prerequisites=[]
        )
        
        result = executor.execute_action(action, mock_mesa_model)
        
        assert result.success is True
        assert result.action_type == "examine"
        assert "findings" in result.result_data
        
    def test_execute_communicate_action(self, executor, mock_mesa_model):
        """Test execution of communicate action"""
        action = MesaAction(
            agent_id="strategist",
            action_type="communicate",
            parameters={"target": "broadcast", "message": "status_update"},
            expected_duration=1.5,
            prerequisites=[]
        )
        
        result = executor.execute_action(action, mock_mesa_model)
        
        assert result.success is True
        assert result.action_type == "communicate"
        assert "sender" in result.result_data
        assert result.result_data["sender"] == "strategist"
        
    def test_execute_claim_resource_action(self, executor, mock_mesa_model):
        """Test execution of claim resource action"""
        action = MesaAction(
            agent_id="strategist",
            action_type="claim_resource",
            parameters={"resource_id": "key_001"},
            expected_duration=1.0,
            prerequisites=[]
        )
        
        result = executor.execute_action(action, mock_mesa_model)
        
        assert result.success is True
        assert result.action_type == "claim_resource"
        assert "resource_id" in result.result_data
        assert result.result_data["resource_id"] == "key_001"
        
    def test_execute_use_tool_action(self, executor, mock_mesa_model):
        """Test execution of use tool action"""
        # Add tool to agent's resources first
        agent = mock_mesa_model.schedule.agents[0]
        agent.resources = ["hammer"]
        
        action = MesaAction(
            agent_id="strategist",
            action_type="use_tool",
            parameters={"tool_id": "hammer"},
            expected_duration=2.5,
            prerequisites=["has_tool"]
        )
        
        result = executor.execute_action(action, mock_mesa_model)
        
        assert result.success is True
        assert result.action_type == "use_tool"
        assert "tool_id" in result.result_data
        
    def test_execute_action_failure(self, executor, mock_mesa_model):
        """Test action execution failure handling"""
        # Action with missing required parameter
        action = MesaAction(
            agent_id="strategist",
            action_type="claim_resource",
            parameters={},  # Missing resource_id
            expected_duration=1.0,
            prerequisites=[]
        )
        
        result = executor.execute_action(action, mock_mesa_model)
        
        assert result.success is False
        assert result.error_message is not None
        assert result.actual_duration > 0
        
    def test_execute_action_batch(self, executor, mock_mesa_model):
        """Test batch action execution"""
        actions = [
            MesaAction(
                agent_id="strategist",
                action_type="move",
                parameters={"target_position": (3, 3)},
                expected_duration=1.0,
                prerequisites=[]
            ),
            MesaAction(
                agent_id="strategist",
                action_type="examine",
                parameters={"target": "environment"},
                expected_duration=2.0,
                prerequisites=[]
            )
        ]
        
        results = executor.execute_action_batch(actions, mock_mesa_model)
        
        assert len(results) == 2
        assert all(isinstance(r, ExecutionResult) for r in results)
        assert all(r.success for r in results)
        
    def test_rollback_functionality(self, executor, mock_mesa_model):
        """Test action rollback functionality"""
        # Execute action that should succeed
        agent = mock_mesa_model.schedule.agents[0]
        original_pos = agent.pos
        
        action = MesaAction(
            agent_id="strategist",
            action_type="move",
            parameters={"target_position": (8, 8)},
            expected_duration=1.0,
            prerequisites=[]
        )
        
        result = executor.execute_action(action, mock_mesa_model)
        assert result.success is True
        
        # Position should have changed (the mock doesn't actually move, so we check the result data)
        assert result.result_data["new_position"] == (8, 8)
        assert result.result_data["old_position"] == original_pos
        
        # Force a failure that triggers rollback
        failing_action = MesaAction(
            agent_id="strategist", 
            action_type="claim_resource",
            parameters={},  # Missing required parameter
            expected_duration=1.0,
            prerequisites=[]
        )
        
        failing_result = executor.execute_action(failing_action, mock_mesa_model)
        assert failing_result.success is False
        
    def test_execution_callbacks(self, executor, mock_mesa_model):
        """Test execution callback system"""
        callback_called = []
        
        def test_callback(action, result):
            callback_called.append((action.action_type, result.success))
        
        executor.add_execution_callback("action_completed", test_callback)
        
        action = MesaAction(
            agent_id="strategist",
            action_type="examine",
            parameters={"target": "environment"},
            expected_duration=1.0,
            prerequisites=[]
        )
        
        result = executor.execute_action(action, mock_mesa_model)
        
        assert result.success is True
        assert len(callback_called) == 1
        assert callback_called[0] == ("examine", True)


class TestExecutionMonitor:
    """Test ExecutionMonitor class"""
    
    @pytest.fixture
    def monitor(self):
        """Create monitor instance"""
        return ExecutionMonitor(monitoring_interval=0.1)
    
    def test_monitor_initialization(self, monitor):
        """Test monitor initialization"""
        assert monitor is not None
        assert monitor.monitoring_interval == 0.1
        assert isinstance(monitor.execution_state, ExecutionState)
        assert isinstance(monitor.performance_metrics, PerformanceMetrics)
        assert monitor.monitoring_active is False
        
    def test_monitor_start_stop(self, monitor):
        """Test monitor start/stop functionality"""
        assert monitor.monitoring_active is False
        
        monitor.start_monitoring()
        assert monitor.monitoring_active is True
        
        monitor.stop_monitoring()
        assert monitor.monitoring_active is False
        
    def test_monitor_execution_update(self, monitor):
        """Test execution state update"""
        plan = ExecutionPlan(
            plan_id="test_plan",
            actions=[
                MesaAction(
                    agent_id="strategist",
                    action_type="move",
                    parameters={"target_position": (5, 5)},
                    expected_duration=1.0,
                    prerequisites=[]
                )
            ]
        )
        
        action = plan.actions[0]
        result = ExecutionResult(
            action_id="test_action",
            agent_id="strategist",
            action_type="move",
            status="completed",
            success=True,
            actual_duration=0.8
        )
        
        monitor.update_execution_state(plan, action, result)
        
        # Check that metrics were updated
        assert monitor.performance_metrics.total_actions_executed == 1
        assert monitor.performance_metrics.successful_actions == 1
        
    def test_monitor_alert_generation(self, monitor):
        """Test alert generation for performance issues"""
        # Set low threshold for testing
        monitor.alert_thresholds["max_execution_time"] = 0.5
        
        # Create slow execution result
        plan = ExecutionPlan(
            plan_id="test_plan",
            actions=[
                MesaAction(
                    agent_id="strategist",
                    action_type="move",
                    parameters={"target_position": (5, 5)},
                    expected_duration=1.0,
                    prerequisites=[]
                )
            ]
        )
        
        action = plan.actions[0]
        slow_result = ExecutionResult(
            action_id="slow_action",
            agent_id="strategist",
            action_type="move",
            status="completed",
            success=True,
            actual_duration=1.0  # Exceeds threshold
        )
        
        initial_alerts = len(monitor.alerts)
        monitor.update_execution_state(plan, action, slow_result)
        
        # Should generate alert for slow execution
        assert len(monitor.alerts) > initial_alerts
        
    def test_monitor_report_generation(self, monitor):
        """Test monitoring report generation"""
        report = monitor.get_monitoring_report()
        
        assert "execution_state" in report
        assert "performance_metrics" in report
        assert "system_health" in report
        assert "recommendations" in report
        assert report["system_health"] in ["healthy", "warning", "degraded"]


class TestPerformanceTracker:
    """Test PerformanceTracker class"""
    
    @pytest.fixture
    def tracker(self):
        """Create tracker instance"""
        return PerformanceTracker()
    
    def test_tracker_initialization(self, tracker):
        """Test tracker initialization"""
        assert tracker is not None
        assert len(tracker.metrics_history) == 0
        assert len(tracker.optimization_suggestions) == 0
        
    def test_performance_tracking(self, tracker):
        """Test performance metrics tracking"""
        metrics = PerformanceMetrics(
            total_actions_executed=10,
            successful_actions=9,
            failed_actions=1,
            average_execution_time=2.5
        )
        
        tracker.track_performance(metrics)
        
        assert len(tracker.metrics_history) == 1
        assert metrics in tracker.metrics_history
        assert len(tracker.execution_patterns["success_rate"]) == 1
        assert tracker.execution_patterns["success_rate"][0] == 0.9
        
    def test_performance_trends(self, tracker):
        """Test performance trend analysis"""
        # Add multiple metrics for trend analysis
        for i in range(5):
            metrics = PerformanceMetrics(
                total_actions_executed=10 + i,
                successful_actions=8 + i,
                failed_actions=2,
                average_execution_time=2.0 + i * 0.1
            )
            tracker.track_performance(metrics)
        
        trends = tracker.get_performance_trends()
        
        assert "success_rate_trend" in trends
        assert "execution_time_trend" in trends
        assert "throughput_trend" in trends
        assert isinstance(trends["optimization_suggestions"], list)
        
    def test_optimization_suggestions(self, tracker):
        """Test optimization suggestion generation"""
        # Add metrics with consistently poor performance
        for i in range(15):
            metrics = PerformanceMetrics(
                total_actions_executed=10 + i,
                successful_actions=6 + i,  # Low success rate
                failed_actions=4,
                average_execution_time=8.0  # High execution time
            )
            tracker.track_performance(metrics)
        
        # Should generate optimization suggestions
        assert len(tracker.optimization_suggestions) > 0


class TestFeedbackProcessor:
    """Test FeedbackProcessor class"""
    
    @pytest.fixture
    def processor(self):
        """Create processor instance"""
        return FeedbackProcessor()
    
    def test_processor_initialization(self, processor):
        """Test processor initialization"""
        assert processor is not None
        assert len(processor.feedback_data) == 0
        assert len(processor.learning_insights) == 0
        
    def test_feedback_processing(self, processor):
        """Test execution feedback processing"""
        # Create mock action handoff
        actions = [
            MesaAction(
                agent_id="strategist",
                action_type="move",
                parameters={"target_position": (5, 5)},
                expected_duration=1.0,
                prerequisites=[]
            )
        ]
        
        handoff = ActionHandoff(
            actions=actions,
            execution_results={},
            conflict_resolutions=[],
            performance_metrics={"success_rate": 0.9}
        )
        
        processor.process_execution_feedback(handoff)
        
        assert len(processor.feedback_data) == 1
        assert len(processor.learning_insights) > 0
        assert "average_success_rate" in processor.learning_insights
        
    def test_learning_recommendations(self, processor):
        """Test learning recommendation generation"""
        # Process multiple feedback sessions with different success rates
        for success_rate in [0.8, 0.7, 0.6]:  # Declining performance
            actions = [
                MesaAction(
                    agent_id="strategist",
                    action_type="move",  
                    parameters={"target_position": (5, 5)},
                    expected_duration=1.0,
                    prerequisites=[]
                )
            ]
            
            handoff = ActionHandoff(
                actions=actions,
                execution_results={},
                conflict_resolutions=[],
                performance_metrics={"success_rate": success_rate}
            )
            
            processor.process_execution_feedback(handoff)
        
        recommendations = processor.get_learning_recommendations()
        
        # Should recommend improvements for low success rate
        assert len(recommendations) > 0
        assert any("success rate" in rec.lower() for rec in recommendations)


class TestPerformanceRequirements:
    """Test performance requirements for execution system"""
    
    @pytest.fixture
    def executor(self):
        return MesaActionExecutor()
    
    @pytest.fixture
    def mock_mesa_model(self):
        """Create mock Mesa model"""
        model = Mock(spec=mesa.Model)
        model.schedule = Mock()
        model.grid = Mock()
        
        # Mock agent
        mock_agent = Mock()
        mock_agent.agent_id = "strategist"
        mock_agent.pos = (5, 5)
        mock_agent.resources = []
        
        model.schedule.agents = [mock_agent]
        model.schedule.steps = 0
        
        return model
    
    def test_execution_performance(self, executor, mock_mesa_model):
        """Test execution performance meets requirements"""
        action = MesaAction(
            agent_id="strategist",
            action_type="move",
            parameters={"target_position": (7, 7)},
            expected_duration=1.0,
            prerequisites=[]
        )
        
        # Measure execution time
        import time
        start_time = time.perf_counter()
        result = executor.execute_action(action, mock_mesa_model)
        end_time = time.perf_counter()
        
        execution_time_ms = (end_time - start_time) * 1000
        
        # Should execute quickly (under 100ms for simple actions)
        assert execution_time_ms < 100
        assert result.success is True
        
    def test_batch_execution_performance(self, executor, mock_mesa_model):
        """Test batch execution performance"""
        actions = []
        for i in range(10):
            action = MesaAction(
                agent_id="strategist",
                action_type="examine",
                parameters={"target": f"object_{i}"},
                expected_duration=1.0,
                prerequisites=[]
            )
            actions.append(action)
        
        import time
        start_time = time.perf_counter()
        results = executor.execute_action_batch(actions, mock_mesa_model)
        end_time = time.perf_counter()
        
        batch_time = end_time - start_time
        
        # Batch execution should be efficient
        assert len(results) == 10
        assert all(r.success for r in results)
        assert batch_time < 1.0  # Less than 1 second for 10 simple actions
        
    def test_monitoring_overhead(self):
        """Test that monitoring doesn't significantly impact performance"""
        monitor = ExecutionMonitor(monitoring_interval=0.01)  # Very frequent monitoring
        
        # Measure monitoring overhead
        import time
        
        start_time = time.perf_counter()
        for _ in range(100):
            plan = ExecutionPlan(
                plan_id=f"plan_{_}",
                actions=[
                    MesaAction(
                        agent_id="strategist",
                        action_type="move",
                        parameters={"target_position": (5, 5)},
                        expected_duration=1.0,
                        prerequisites=[]
                    )
                ]
            )
            
            result = ExecutionResult(
                action_id=f"action_{_}",
                agent_id="strategist",
                action_type="move",
                status="completed",
                success=True,
                actual_duration=0.5
            )
            
            monitor.update_execution_state(plan, plan.actions[0], result)
        
        end_time = time.perf_counter()
        monitoring_time = end_time - start_time
        
        # Monitoring 100 updates should be very fast
        assert monitoring_time < 0.1  # Less than 100ms