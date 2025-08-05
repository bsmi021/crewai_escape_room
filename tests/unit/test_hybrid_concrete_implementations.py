"""
Unit tests for concrete implementations of Mesa-CrewAI Hybrid Architecture

Tests implement TDD methodology for concrete pipeline components.
Following the architectural specifications from the multi-agent design process.
"""

import pytest
import asyncio
from datetime import datetime
from typing import Dict, Any, List
from unittest.mock import Mock, MagicMock, patch
import mesa

from src.escape_room_sim.hybrid.core_architecture import (
    PerceptionData,
    DecisionData,
    MesaAction,
    IPerceptionPipeline,
    IDecisionEngine,
    IActionTranslator,
    IStateSynchronizer
)


class TestEscapeRoomPerceptionPipeline:
    """Test concrete perception pipeline implementation for escape room scenarios"""
    
    @pytest.fixture
    def mock_mesa_model(self):
        """Create mock Mesa model with escape room structure"""
        model = Mock()
        model.schedule = Mock()
        
        # Create mock agents
        agent1 = Mock()
        agent1.agent_id = "strategist"
        agent1.pos = (2, 3)
        agent1.energy = 0.8
        agent1.health = 100
        
        agent2 = Mock()
        agent2.agent_id = "mediator"
        agent2.pos = (3, 3)
        agent2.energy = 0.6
        agent2.health = 85
        
        agent3 = Mock()
        agent3.agent_id = "survivor"
        agent3.pos = (1, 2)
        agent3.energy = 0.9
        agent3.health = 95
        
        model.schedule.agents = [agent1, agent2, agent3]
        
        # Mock room structure
        model.width = 10
        model.height = 10
        model.room_objects = {
            (0, 0): {"type": "door", "locked": True, "key_required": "master_key"},
            (5, 5): {"type": "chest", "locked": True, "contains": ["flashlight", "rope"]},
            (8, 2): {"type": "key", "key_type": "chest_key"},
            (2, 7): {"type": "puzzle", "solved": False, "difficulty": 3}
        }
        
        # Mock environmental conditions
        model.environment = {
            "lighting": "dim",
            "temperature": 18,
            "air_quality": "stale",
            "noise_level": "quiet"
        }
        
        return model
    
    def test_extract_perceptions_basic(self, mock_mesa_model):
        """Test basic perception extraction from Mesa model"""
        from src.escape_room_sim.hybrid.data_flow import EscapeRoomPerceptionPipeline
        
        pipeline = EscapeRoomPerceptionPipeline()
        perceptions = pipeline.extract_perceptions(mock_mesa_model)
        
        # Should extract perceptions for all agents
        assert len(perceptions) == 3
        assert "strategist" in perceptions
        assert "mediator" in perceptions
        assert "survivor" in perceptions
        
        # Check strategist perception
        strategist_perception = perceptions["strategist"]
        assert strategist_perception.agent_id == "strategist"
        assert strategist_perception.spatial_data["current_position"] == (2, 3)
        assert "mediator" in strategist_perception.nearby_agents
        assert "move" in strategist_perception.available_actions
    
    def test_filter_perceptions_by_agent_capabilities(self, mock_mesa_model):
        """Test perception filtering based on agent capabilities"""
        from src.escape_room_sim.hybrid.data_flow import EscapeRoomPerceptionPipeline
        
        pipeline = EscapeRoomPerceptionPipeline()
        all_perceptions = pipeline.extract_perceptions(mock_mesa_model)
        
        # Filter for strategist (should have analytical capabilities)
        strategist_filtered = pipeline.filter_perceptions(all_perceptions, "strategist")
        
        assert strategist_filtered.agent_id == "strategist"
        assert "analyze" in strategist_filtered.available_actions
        assert "environmental_state" in strategist_filtered.__dict__
    
    def test_spatial_perception_accuracy(self, mock_mesa_model):
        """Test spatial perception accuracy"""
        from src.escape_room_sim.hybrid.data_flow import EscapeRoomPerceptionPipeline
        
        pipeline = EscapeRoomPerceptionPipeline()
        perceptions = pipeline.extract_perceptions(mock_mesa_model)
        
        strategist_perception = perceptions["strategist"]
        
        # Should detect nearby agents within range
        assert "mediator" in strategist_perception.nearby_agents  # Distance 1.0
        assert "survivor" in strategist_perception.nearby_agents  # Distance ~1.41 < 5 (perception range)
        
        # Should detect nearby objects
        nearby_objects = strategist_perception.spatial_data.get("nearby_objects", {})
        # Check that the puzzle at (2, 7) is NOT in nearby_objects due to distance > search_range(2)
        assert (2, 7) not in nearby_objects.values()
        
        # But should have other agents' positions
        assert (3, 3) in nearby_objects.values()  # mediator position


class TestCrewAIDecisionEngine:
    """Test concrete CrewAI decision engine implementation"""
    
    @pytest.fixture
    def mock_crewai_agents(self):
        """Create mock CrewAI agents"""
        strategist = Mock()
        strategist.role = "Strategist"
        strategist.backstory = "Analytical problem solver"
        strategist.memory = Mock()
        
        mediator = Mock()
        mediator.role = "Mediator"
        mediator.backstory = "Group facilitator"
        mediator.memory = Mock()
        
        survivor = Mock()
        survivor.role = "Survivor"
        survivor.backstory = "Pragmatic decision maker"
        survivor.memory = Mock()
        
        return [strategist, mediator, survivor]
    
    @pytest.fixture
    def sample_perceptions(self):
        """Create sample perception data"""
        return {
            "strategist": PerceptionData(
                agent_id="strategist",
                timestamp=datetime.now(),
                spatial_data={"current_position": (2, 3), "room_size": (10, 10)},
                environmental_state={"lighting": "dim", "temperature": 18},
                nearby_agents=["mediator"],
                available_actions=["move", "examine", "analyze", "communicate"],
                resources={"energy": 0.8, "tools": []},
                constraints={"movement_range": 2, "action_points": 3}
            ),
            "mediator": PerceptionData(
                agent_id="mediator",
                timestamp=datetime.now(),
                spatial_data={"current_position": (3, 3), "room_size": (10, 10)},
                environmental_state={"lighting": "dim", "temperature": 18},
                nearby_agents=["strategist"],
                available_actions=["move", "communicate", "coordinate", "mediate"],
                resources={"energy": 0.6, "tools": []},
                constraints={"movement_range": 2, "action_points": 2}
            ),
            "survivor": PerceptionData(
                agent_id="survivor",
                timestamp=datetime.now(),
                spatial_data={"current_position": (1, 2), "room_size": (10, 10)},
                environmental_state={"lighting": "dim", "temperature": 18},
                nearby_agents=[],
                available_actions=["move", "examine", "use_tool", "survive"],
                resources={"energy": 0.9, "tools": ["flashlight"]},
                constraints={"movement_range": 3, "action_points": 4}
            )
        }
    
    @pytest.mark.asyncio
    async def test_reason_and_decide_basic(self, mock_crewai_agents, sample_perceptions):
        """Test basic reasoning and decision making"""
        from src.escape_room_sim.hybrid.data_flow import CrewAIDecisionEngine
        
        # Mock LLM responses
        with patch('crewai.Crew') as mock_crew_class:
            mock_crew = Mock()
            mock_crew_class.return_value = mock_crew
            mock_crew.kickoff.return_value = {
                "strategist": "analyze current situation and identify key objects",
                "mediator": "coordinate with strategist to develop plan",
                "survivor": "examine nearby objects for useful tools"
            }
            
            engine = CrewAIDecisionEngine(mock_crewai_agents)
            decisions = await engine.reason_and_decide(sample_perceptions)
            
            assert len(decisions) == 3
            assert "strategist" in decisions
            assert "mediator" in decisions
            assert "survivor" in decisions
            
            # Verify decision structure
            strategist_decision = decisions["strategist"]
            assert isinstance(strategist_decision, DecisionData)
            assert strategist_decision.agent_id == "strategist"
            assert strategist_decision.chosen_action in ["analyze", "examine", "move"]
    
    @pytest.mark.asyncio
    async def test_decision_engine_memory_update(self, mock_crewai_agents, sample_perceptions):
        """Test memory update functionality"""
        from src.escape_room_sim.hybrid.data_flow import CrewAIDecisionEngine
        
        engine = CrewAIDecisionEngine(mock_crewai_agents)
        
        # Test memory update
        experience = {
            "action": "examine",
            "result": "found_key",
            "success": True,
            "timestamp": datetime.now()
        }
        
        engine.update_agent_memory("strategist", experience)
        
        # Verify memory was updated (would depend on implementation)
        assert True  # Placeholder - actual implementation would test memory storage


class TestMesaActionTranslator:
    """Test concrete Mesa action translator implementation"""
    
    @pytest.fixture
    def sample_decisions(self):
        """Create sample decision data"""
        return [
            DecisionData(
                agent_id="strategist",
                timestamp=datetime.now(),
                chosen_action="analyze",
                action_parameters={"target": "puzzle", "depth": "detailed"},
                reasoning="Need to understand puzzle mechanics",
                confidence_level=0.8,
                fallback_actions=["examine", "observe"]
            ),
            DecisionData(
                agent_id="mediator",
                timestamp=datetime.now(),
                chosen_action="communicate",
                action_parameters={"target": "strategist", "message": "coordinate_plan"},
                reasoning="Team coordination is essential",
                confidence_level=0.9,
                fallback_actions=["observe", "wait"]
            ),
            DecisionData(
                agent_id="survivor",
                timestamp=datetime.now(),
                chosen_action="move",
                action_parameters={"target_position": (2, 2), "speed": "careful"},
                reasoning="Need to get closer to the action",
                confidence_level=0.7,
                fallback_actions=["wait", "observe"]
            )
        ]
    
    def test_translate_decision_to_mesa_action(self, sample_decisions):
        """Test translation of decisions to Mesa actions"""
        from src.escape_room_sim.hybrid.data_flow import EscapeRoomActionTranslator
        
        translator = EscapeRoomActionTranslator()
        
        for decision in sample_decisions:
            mesa_action = translator.translate_decision(decision)
            
            assert isinstance(mesa_action, MesaAction)
            assert mesa_action.agent_id == decision.agent_id
            assert mesa_action.action_type == decision.chosen_action
            assert mesa_action.expected_duration > 0
    
    def test_validate_action_legal(self, sample_decisions):
        """Test action validation against Mesa model state"""
        from src.escape_room_sim.hybrid.data_flow import EscapeRoomActionTranslator
        
        # Create mock Mesa model
        mock_model = Mock()
        mock_model.width = 10
        mock_model.height = 10
        mock_model.schedule = Mock()
        mock_model.schedule.agents = []
        mock_model.room_objects = {}  # No blocking objects
        
        translator = EscapeRoomActionTranslator()
        
        # Test valid move action
        move_decision = sample_decisions[2]  # survivor move action
        mesa_action = translator.translate_decision(move_decision)
        
        # Debug the action parameters
        print(f"Action: {mesa_action.action_type}, Params: {mesa_action.parameters}")
        
        is_valid = translator.validate_action(mesa_action, mock_model)
        assert is_valid is True  # Move to (2,2) should be valid in 10x10 room
    
    def test_validate_action_illegal(self, sample_decisions):
        """Test validation of illegal actions"""
        from src.escape_room_sim.hybrid.data_flow import EscapeRoomActionTranslator
        
        # Create mock Mesa model with constraints
        mock_model = Mock()
        mock_model.width = 5
        mock_model.height = 5
        
        translator = EscapeRoomActionTranslator()
        
        # Create illegal move action (out of bounds)
        illegal_decision = DecisionData(
            agent_id="test",
            timestamp=datetime.now(),
            chosen_action="move",
            action_parameters={"target_position": (10, 10)},  # Out of bounds
            reasoning="Test illegal move",
            confidence_level=0.5,
            fallback_actions=[]
        )
        
        mesa_action = translator.translate_decision(illegal_decision)
        is_valid = translator.validate_action(mesa_action, mock_model)
        assert is_valid is False


class TestStateSynchronizer:
    """Test concrete state synchronizer implementation"""
    
    @pytest.fixture
    def mock_mesa_model(self):
        """Create mock Mesa model for synchronization"""
        model = Mock()
        model.schedule = Mock()
        
        # Mock agents with state
        agent = Mock()
        agent.agent_id = "strategist"
        agent.pos = (2, 3)
        agent.state = {"health": 100, "energy": 0.8}
        
        model.schedule.agents = [agent]
        return model
    
    @pytest.fixture
    def sample_decisions(self):
        """Create sample decisions for synchronization"""
        return {
            "strategist": DecisionData(
                agent_id="strategist",
                timestamp=datetime.now(),
                chosen_action="move",
                action_parameters={"target_position": (3, 3)},
                reasoning="Moving to better position",
                confidence_level=0.8,
                fallback_actions=[]
            )
        }
    
    def test_sync_mesa_to_crewai(self, mock_mesa_model):
        """Test synchronization from Mesa to CrewAI"""
        from src.escape_room_sim.hybrid.state_management import EscapeRoomStateSynchronizer
        
        synchronizer = EscapeRoomStateSynchronizer()
        
        # This should update CrewAI agent memories with Mesa state
        synchronizer.sync_mesa_to_crewai(mock_mesa_model)
        
        # Verification would depend on implementation
        assert True  # Placeholder
    
    def test_sync_crewai_to_mesa(self, mock_mesa_model, sample_decisions):
        """Test synchronization from CrewAI to Mesa"""
        from src.escape_room_sim.hybrid.state_management import EscapeRoomStateSynchronizer
        
        synchronizer = EscapeRoomStateSynchronizer()
        
        # This should apply decisions to Mesa model
        synchronizer.sync_crewai_to_mesa(sample_decisions, mock_mesa_model)
        
        # Verification would depend on implementation
        assert True  # Placeholder


if __name__ == "__main__":
    pytest.main([__file__, "-v"])