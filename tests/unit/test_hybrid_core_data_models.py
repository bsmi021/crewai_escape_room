"""
Unit tests for Mesa-CrewAI Hybrid Architecture Core Data Models

Tests implement TDD methodology for Phase 1.1: Core Data Models and Interfaces
Following the architectural specifications from the multi-agent design process.
"""

import pytest
import asyncio
from datetime import datetime
from typing import Dict, Any, List

from src.escape_room_sim.hybrid.core_architecture import (
    ComponentState,
    PerceptionData,
    DecisionData,
    MesaAction,
    HybridAgent,
    HybridSimulationEngine
)


class TestPerceptionData:
    """Test PerceptionData data model with comprehensive validation"""
    
    def test_perception_data_creation_valid(self):
        """Test PerceptionData creation with valid data"""
        timestamp = datetime.now()
        perception = PerceptionData(
            agent_id="strategist",
            timestamp=timestamp,
            spatial_data={"x": 5, "y": 3, "room": "main"},
            environmental_state={"temperature": 22, "lighting": "dim"},
            nearby_agents=["mediator", "survivor"],
            available_actions=["move", "examine", "communicate"],
            resources={"energy": 0.8, "tools": ["flashlight"]},
            constraints={"movement_range": 2, "action_points": 3}
        )
        
        assert perception.agent_id == "strategist"
        assert perception.timestamp == timestamp
        assert perception.spatial_data["room"] == "main"
        assert len(perception.nearby_agents) == 2
        assert "examine" in perception.available_actions
    
    def test_perception_data_immutable_fields(self):
        """Test that PerceptionData fields are properly typed"""
        perception = PerceptionData(
            agent_id="test",
            timestamp=datetime.now(),
            spatial_data={},
            environmental_state={},
            nearby_agents=[],
            available_actions=[],
            resources={},
            constraints={}
        )
        
        # Verify types
        assert isinstance(perception.agent_id, str)
        assert isinstance(perception.timestamp, datetime)
        assert isinstance(perception.spatial_data, dict)
        assert isinstance(perception.nearby_agents, list)


class TestDecisionData:
    """Test DecisionData model for CrewAI decision representation"""
    
    def test_decision_data_creation_valid(self):
        """Test DecisionData creation with valid parameters"""
        timestamp = datetime.now()
        decision = DecisionData(
            agent_id="mediator",
            timestamp=timestamp,
            chosen_action="communicate",
            action_parameters={"target": "strategist", "message": "coordinate_plan"},
            reasoning="Need to align our strategy before proceeding",
            confidence_level=0.85,
            fallback_actions=["observe", "wait"]
        )
        
        assert decision.agent_id == "mediator"
        assert decision.chosen_action == "communicate"
        assert decision.confidence_level == 0.85
        assert len(decision.fallback_actions) == 2
    
    def test_decision_data_confidence_validation(self):
        """Test confidence level validation"""
        # Valid confidence levels
        decision_low = DecisionData(
            agent_id="test", timestamp=datetime.now(), chosen_action="test",
            action_parameters={}, reasoning="test", confidence_level=0.0,
            fallback_actions=[]
        )
        decision_high = DecisionData(
            agent_id="test", timestamp=datetime.now(), chosen_action="test",
            action_parameters={}, reasoning="test", confidence_level=1.0,
            fallback_actions=[]
        )
        
        assert decision_low.confidence_level == 0.0
        assert decision_high.confidence_level == 1.0


class TestMesaAction:
    """Test MesaAction model for Mesa-compatible actions"""
    
    def test_mesa_action_creation_valid(self):
        """Test MesaAction creation with valid parameters"""
        action = MesaAction(
            agent_id="survivor",
            action_type="move",
            parameters={"target_position": (3, 4), "speed": "normal"},
            expected_duration=2.5,
            prerequisites=["has_energy", "path_clear"]
        )
        
        assert action.agent_id == "survivor"
        assert action.action_type == "move"
        assert action.expected_duration == 2.5
        assert "has_energy" in action.prerequisites
    
    def test_mesa_action_empty_parameters(self):
        """Test MesaAction with empty parameters"""
        action = MesaAction(
            agent_id="test",
            action_type="wait",
            parameters={},
            expected_duration=1.0,
            prerequisites=[]
        )
        
        assert len(action.parameters) == 0
        assert len(action.prerequisites) == 0


class TestComponentState:
    """Test ComponentState enum values"""
    
    def test_component_state_values(self):
        """Test all ComponentState enum values"""
        assert ComponentState.UNINITIALIZED.value == "uninitialized"
        assert ComponentState.INITIALIZING.value == "initializing"
        assert ComponentState.READY.value == "ready"
        assert ComponentState.RUNNING.value == "running"
        assert ComponentState.PAUSED.value == "paused"
        assert ComponentState.ERROR.value == "error"
        assert ComponentState.SHUTDOWN.value == "shutdown"
    
    def test_component_state_lifecycle(self):
        """Test typical component state lifecycle"""
        states = [
            ComponentState.UNINITIALIZED,
            ComponentState.INITIALIZING,
            ComponentState.READY,
            ComponentState.RUNNING,
            ComponentState.PAUSED,
            ComponentState.READY,
            ComponentState.SHUTDOWN
        ]
        
        # Verify all states are valid
        for state in states:
            assert isinstance(state, ComponentState)


class TestHybridAgent:
    """Test HybridAgent composition class"""
    
    @pytest.fixture
    def mock_mesa_agent(self):
        """Create mock Mesa agent"""
        from unittest.mock import Mock, MagicMock
        mock_agent = Mock()
        mock_agent.pos = (2, 3)
        mock_agent.state = {"health": 100, "energy": 0.8}
        mock_agent.agent_id = "test_agent"
        return mock_agent
    
    @pytest.fixture
    def mock_crewai_agent(self):
        """Create mock CrewAI agent"""
        from unittest.mock import Mock
        mock_agent = Mock()
        mock_agent.role = "Test Agent"
        mock_agent.memory = {"last_action": "observe", "plan": "gather_info"}
        return mock_agent
    
    def test_hybrid_agent_creation(self, mock_mesa_agent, mock_crewai_agent):
        """Test HybridAgent creation with both agent types"""
        hybrid_agent = HybridAgent("test_agent", mock_mesa_agent, mock_crewai_agent)
        
        assert hybrid_agent.agent_id == "test_agent"
        assert hybrid_agent.mesa_agent == mock_mesa_agent
        assert hybrid_agent.crewai_agent == mock_crewai_agent
        assert hybrid_agent.state == ComponentState.UNINITIALIZED
        assert hybrid_agent.last_perception is None
        assert hybrid_agent.last_decision is None
    
    def test_hybrid_agent_unified_state(self, mock_mesa_agent, mock_crewai_agent):
        """Test unified state representation"""
        hybrid_agent = HybridAgent("test_agent", mock_mesa_agent, mock_crewai_agent)
        
        unified_state = hybrid_agent.get_unified_state()
        
        assert unified_state["agent_id"] == "test_agent"
        assert unified_state["mesa_position"] == (2, 3)
        assert unified_state["mesa_state"]["health"] == 100
        assert unified_state["crewai_memory"]["plan"] == "gather_info"
        assert unified_state["last_action"] is None
    
    def test_hybrid_agent_performance_metrics(self, mock_mesa_agent, mock_crewai_agent):
        """Test performance metrics tracking"""
        hybrid_agent = HybridAgent("test_agent", mock_mesa_agent, mock_crewai_agent)
        
        # Update performance metrics
        metrics = {"decision_time": 0.5, "action_success": True, "reasoning_depth": 3}
        hybrid_agent.update_performance_metrics(metrics)
        
        assert hybrid_agent.performance_metrics["decision_time"] == 0.5
        assert hybrid_agent.performance_metrics["action_success"] is True
        
        # Verify metrics in unified state
        unified_state = hybrid_agent.get_unified_state()
        assert unified_state["performance"]["reasoning_depth"] == 3


class TestHybridSimulationEngineInitialization:
    """Test HybridSimulationEngine initialization and basic functionality"""
    
    @pytest.fixture
    def mock_components(self):
        """Create mock components for HybridSimulationEngine"""
        from unittest.mock import Mock
        
        # Mock Mesa model
        mock_mesa_model = Mock()
        mock_mesa_model.schedule = Mock()
        mock_mesa_model.schedule.agents = []
        
        # Mock CrewAI agents
        mock_crewai_agents = [
            Mock(role="Strategist"),
            Mock(role="Mediator"),
            Mock(role="Survivor")
        ]
        
        # Mock pipeline components
        mock_perception_pipeline = Mock()
        mock_decision_engine = Mock()
        mock_action_translator = Mock()
        mock_state_synchronizer = Mock()
        
        return {
            "mesa_model": mock_mesa_model,
            "crewai_agents": mock_crewai_agents,
            "perception_pipeline": mock_perception_pipeline,
            "decision_engine": mock_decision_engine,
            "action_translator": mock_action_translator,
            "state_synchronizer": mock_state_synchronizer
        }
    
    def test_hybrid_simulation_engine_creation(self, mock_components):
        """Test HybridSimulationEngine creation"""
        engine = HybridSimulationEngine(**mock_components)
        
        assert engine.mesa_model == mock_components["mesa_model"]
        assert len(engine.crewai_agents) == 3
        assert engine.state == ComponentState.UNINITIALIZED
        assert engine.step_count == 0
        assert engine.error_count == 0
        assert len(engine.hybrid_agents) == 0
    
    def test_hybrid_simulation_engine_initialization_success(self, mock_components):
        """Test successful initialization"""
        engine = HybridSimulationEngine(**mock_components)
        
        # Initialize should succeed with mock components
        engine.initialize()
        
        # Verify state changed to READY
        assert engine.state == ComponentState.READY
    
    def test_hybrid_simulation_engine_state_management(self, mock_components):
        """Test state management during initialization"""
        engine = HybridSimulationEngine(**mock_components)
        
        assert engine.state == ComponentState.UNINITIALIZED
        
        # Try to step without initialization - should fail
        with pytest.raises(RuntimeError) as exc_info:
            asyncio.run(engine.step())
        
        assert "Cannot step in state" in str(exc_info.value)


class TestDataModelValidation:
    """Test data model validation and edge cases"""
    
    def test_perception_data_with_empty_collections(self):
        """Test PerceptionData with empty collections"""
        perception = PerceptionData(
            agent_id="agent1",
            timestamp=datetime.now(),
            spatial_data={},
            environmental_state={},
            nearby_agents=[],
            available_actions=[],
            resources={},
            constraints={}
        )
        
        assert len(perception.nearby_agents) == 0
        assert len(perception.available_actions) == 0
        assert len(perception.resources) == 0
    
    def test_decision_data_with_complex_parameters(self):
        """Test DecisionData with complex action parameters"""
        complex_params = {
            "primary_target": {"type": "agent", "id": "strategist"},
            "secondary_actions": [
                {"action": "move", "position": (1, 2)},
                {"action": "communicate", "message": "status_update"}
            ],
            "conditions": {
                "if_successful": "continue_plan",
                "if_failed": "retreat"
            }
        }
        
        decision = DecisionData(
            agent_id="complex_agent",
            timestamp=datetime.now(),
            chosen_action="complex_multi_action",
            action_parameters=complex_params,
            reasoning="Multi-step plan execution based on current state analysis",
            confidence_level=0.75,
            fallback_actions=["simple_observe", "wait_and_reassess"]
        )
        
        assert decision.action_parameters["primary_target"]["id"] == "strategist"
        assert len(decision.action_parameters["secondary_actions"]) == 2
        assert decision.action_parameters["conditions"]["if_failed"] == "retreat"
    
    def test_mesa_action_duration_edge_cases(self):
        """Test MesaAction with edge case durations"""
        # Instantaneous action
        instant_action = MesaAction(
            agent_id="fast_agent",
            action_type="instant_scan",
            parameters={"scan_type": "quick"},
            expected_duration=0.0,
            prerequisites=[]
        )
        
        # Long duration action
        long_action = MesaAction(
            agent_id="patient_agent",
            action_type="deep_analysis",
            parameters={"analysis_depth": "comprehensive"},
            expected_duration=300.0,  # 5 minutes
            prerequisites=["quiet_environment", "full_energy"]
        )
        
        assert instant_action.expected_duration == 0.0
        assert long_action.expected_duration == 300.0
        assert len(long_action.prerequisites) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])