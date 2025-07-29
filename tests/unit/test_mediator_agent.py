"""
Comprehensive unit tests for the Mediator agent.

Tests cover:
- Agent creation and configuration
- Memory and learning systems  
- Agent personality and behavior
- Context-aware agent creation
- Relationship tracking and team dynamics
- Error handling and edge cases
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any

from src.escape_room_sim.agents.mediator import (
    create_mediator_agent,
    create_mediator_with_context,
    MediatorConfig
)


class TestMediatorAgentCreation:
    """Test suite for basic mediator agent creation and configuration."""
    
    @patch('src.escape_room_sim.agents.mediator.Agent')
    def test_create_mediator_agent_default_parameters(self, mock_agent_class):
        """Test creating mediator agent with default parameters."""
        # Arrange
        mock_agent_instance = Mock()
        mock_agent_class.return_value = mock_agent_instance
        
        # Act
        agent = create_mediator_agent()
        
        # Assert
        mock_agent_class.assert_called_once()
        call_args = mock_agent_class.call_args
        
        assert call_args[1]['role'] == "Group Facilitator"
        assert call_args[1]['goal'] == "Build consensus through multiple discussion rounds and maintain team cohesion"
        assert call_args[1]['verbose'] is True
        assert call_args[1]['memory'] is True
        assert call_args[1]['max_iter'] == 3
        assert call_args[1]['allow_delegation'] is False
        assert "former crisis counselor" in call_args[1]['backstory']
        assert "Group Facilitator" in call_args[1]['system_message']
        
    @patch('src.escape_room_sim.agents.mediator.Agent')
    def test_create_mediator_agent_custom_parameters(self, mock_agent_class):
        """Test creating mediator agent with custom parameters."""
        # Arrange
        mock_agent_instance = Mock()
        mock_agent_class.return_value = mock_agent_instance
        
        # Act
        agent = create_mediator_agent(
            memory_enabled=False,
            verbose=False
        )
        
        # Assert
        call_args = mock_agent_class.call_args
        assert call_args[1]['memory'] is False
        assert call_args[1]['verbose'] is False
        
    @pytest.mark.parametrize("memory_enabled,verbose", [
        (True, True),
        (True, False),
        (False, True),
        (False, False)
    ])
    @patch('src.escape_room_sim.agents.mediator.Agent')
    def test_create_mediator_agent_parameter_combinations(self, mock_agent_class, memory_enabled, verbose):
        """Test all combinations of memory_enabled and verbose parameters."""
        # Arrange
        mock_agent_instance = Mock()
        mock_agent_class.return_value = mock_agent_instance
        
        # Act
        agent = create_mediator_agent(
            memory_enabled=memory_enabled,
            verbose=verbose
        )
        
        # Assert
        call_args = mock_agent_class.call_args
        assert call_args[1]['memory'] == memory_enabled
        assert call_args[1]['verbose'] == verbose
        
    @patch('src.escape_room_sim.agents.mediator.Agent')
    def test_create_mediator_agent_returns_agent_instance(self, mock_agent_class):
        """Test that the function returns the agent instance created by CrewAI."""
        # Arrange
        mock_agent_instance = Mock()
        mock_agent_class.return_value = mock_agent_instance
        
        # Act
        result = create_mediator_agent()
        
        # Assert
        assert result == mock_agent_instance


class TestMediatorMemoryAndLearning:
    """Test suite for memory and learning systems."""
    
    @patch('src.escape_room_sim.agents.mediator.Agent')
    def test_create_mediator_agent_with_empty_context(self, mock_agent_class):
        """Test creating mediator with empty iteration context."""
        # Arrange
        mock_agent_instance = Mock()
        mock_agent_class.return_value = mock_agent_instance
        empty_context = {}
        
        # Act
        agent = create_mediator_agent(iteration_context=empty_context)
        
        # Assert
        call_args = mock_agent_class.call_args
        backstory = call_args[1]['backstory']
        
        # Should use base backstory only
        assert "former crisis counselor" in backstory
        assert "IMPORTANT RELATIONSHIP INSIGHTS" not in backstory
        
    @patch('src.escape_room_sim.agents.mediator.Agent')
    def test_create_mediator_agent_with_team_conflicts_context(self, mock_agent_class, mediator_context):
        """Test creating mediator with team conflicts in context."""
        # Arrange
        mock_agent_instance = Mock()
        mock_agent_class.return_value = mock_agent_instance
        
        # Act
        agent = create_mediator_agent(iteration_context=mediator_context)
        
        # Assert
        call_args = mock_agent_class.call_args
        backstory = call_args[1]['backstory']
        
        assert "IMPORTANT RELATIONSHIP INSIGHTS" in backstory
        assert "Strategist vs Survivor disagreement" in backstory
        assert "Communication breakdown under pressure" in backstory
        assert "use your understanding of team relationships" in backstory
        
    @patch('src.escape_room_sim.agents.mediator.Agent')
    def test_create_mediator_agent_with_trust_levels_in_context(self, mock_agent_class, mediator_context):
        """Test creating mediator with trust levels in context."""
        # Arrange
        mock_agent_instance = Mock()
        mock_agent_class.return_value = mock_agent_instance
        
        # Act
        agent = create_mediator_agent(iteration_context=mediator_context)
        
        # Assert
        call_args = mock_agent_class.call_args
        backstory = call_args[1]['backstory']
        
        trust_levels = mediator_context['trust_levels']
        assert "Trust levels you've observed" in backstory
        # Should contain trust level information
        for trust_info in str(trust_levels).split():
            if trust_info.replace('.', '').replace(':', '').replace(',', '').isdigit():
                continue  # Skip just numbers
            if len(trust_info) > 2:  # Skip very short strings
                assert trust_info in backstory or trust_info.replace("'", "") in backstory or trust_info.replace('"', '') in backstory
        
    @patch('src.escape_room_sim.agents.mediator.Agent')
    def test_create_mediator_agent_limits_team_conflicts_display(self, mock_agent_class):
        """Test that only first 3 team conflicts are displayed in backstory."""
        # Arrange
        mock_agent_instance = Mock()
        mock_agent_class.return_value = mock_agent_instance
        
        context = {
            "team_conflicts": [
                "Conflict 1", "Conflict 2", "Conflict 3", 
                "Conflict 4", "Conflict 5", "Conflict 6"
            ]
        }
        
        # Act
        agent = create_mediator_agent(iteration_context=context)
        
        # Assert
        call_args = mock_agent_class.call_args
        backstory = call_args[1]['backstory']
        
        assert "Conflict 1" in backstory
        assert "Conflict 2" in backstory
        assert "Conflict 3" in backstory
        assert "Conflict 4" not in backstory
        assert "Conflict 5" not in backstory
        assert "Conflict 6" not in backstory
        
    @patch('src.escape_room_sim.agents.mediator.Agent')
    def test_create_mediator_agent_context_without_team_conflicts(self, mock_agent_class):
        """Test creating mediator with context that has no team_conflicts key."""
        # Arrange
        mock_agent_instance = Mock()
        mock_agent_class.return_value = mock_agent_instance
        
        context = {
            "trust_levels": {"high": 0.8},
            "successful_collaborations": ["Some success"]
        }
        
        # Act
        agent = create_mediator_agent(iteration_context=context)
        
        # Assert
        call_args = mock_agent_class.call_args
        backstory = call_args[1]['backstory']
        
        # Should use base backstory since no team_conflicts
        assert "former crisis counselor" in backstory
        assert "IMPORTANT RELATIONSHIP INSIGHTS" not in backstory


class TestMediatorPersonalityAndBehavior:
    """Test suite for agent personality and behavior configuration."""
    
    @patch('src.escape_room_sim.agents.mediator.Agent')
    def test_create_mediator_agent_system_message_content(self, mock_agent_class):
        """Test that system message contains expected behavioral instructions."""
        # Arrange
        mock_agent_instance = Mock()
        mock_agent_class.return_value = mock_agent_instance
        
        # Act
        agent = create_mediator_agent()
        
        # Assert
        call_args = mock_agent_class.call_args
        system_message = call_args[1]['system_message']
        
        expected_behaviors = [
            "Foster open communication between all team members",
            "Identify and address conflicts before they escalate",
            "Build consensus on proposed strategies",
            "Ensure everyone's ideas and concerns are heard",
            "Track relationship dynamics and trust levels",
            "Adapt your facilitation style based on team morale"
        ]
        
        for behavior in expected_behaviors:
            assert behavior in system_message
            
    @patch('src.escape_room_sim.agents.mediator.Agent')
    def test_create_mediator_agent_backstory_personality_traits(self, mock_agent_class):
        """Test that backstory reflects mediator personality traits."""
        # Arrange
        mock_agent_instance = Mock()
        mock_agent_class.return_value = mock_agent_instance
        
        # Act
        agent = create_mediator_agent()
        
        # Assert
        call_args = mock_agent_class.call_args
        backstory = call_args[1]['backstory']
        
        personality_indicators = [
            "former crisis counselor",
            "group facilitator",
            "conflict resolution", 
            "team building",
            "communicate openly",
            "work together",
            "everyone's voice is heard"
        ]
        
        for indicator in personality_indicators:
            assert indicator in backstory
            
    def test_mediator_config_personality_traits(self):
        """Test MediatorConfig personality traits configuration."""
        # Act
        traits = MediatorConfig.get_personality_traits()
        
        # Assert
        expected_traits = {
            "empathy_level": "high",
            "conflict_resolution_style": "collaborative",
            "communication_preference": "inclusive",
            "decision_style": "consensus_building",
            "leadership_style": "facilitative"
        }
        
        assert traits == expected_traits
        
    def test_mediator_config_relationship_tracking(self):
        """Test MediatorConfig relationship tracking configuration."""
        # Act
        tracking_config = MediatorConfig.get_relationship_tracking_config()
        
        # Assert
        expected_config = {
            "track_trust_levels": True,
            "monitor_communication_patterns": True,
            "detect_conflict_early": True,
            "measure_team_cohesion": True,
            "adapt_facilitation_style": True
        }
        
        assert tracking_config == expected_config
        
    def test_mediator_config_constants(self):
        """Test MediatorConfig constants are properly defined."""
        # Assert
        assert MediatorConfig.DEFAULT_ROLE == "Group Facilitator"
        assert MediatorConfig.DEFAULT_GOAL == "Build consensus through multiple discussion rounds and maintain team cohesion"
        assert MediatorConfig.EMPATHY_LEVEL == "high"
        assert MediatorConfig.CONFLICT_RESOLUTION_STYLE == "collaborative"
        assert MediatorConfig.COMMUNICATION_PREFERENCE == "inclusive"


class TestMediatorContextAwareCreation:
    """Test suite for context-aware agent creation."""
    
    @patch('src.escape_room_sim.agents.mediator.Agent')
    def test_create_mediator_with_context_comprehensive(self, mock_agent_class, previous_results_comprehensive):
        """Test creating mediator with comprehensive context data."""
        # Arrange
        mock_agent_instance = Mock()
        mock_agent_class.return_value = mock_agent_instance
        
        # Act
        agent = create_mediator_with_context(previous_results_comprehensive)
        
        # Assert
        mock_agent_class.assert_called_once()
        call_args = mock_agent_class.call_args
        
        # Verify context was processed correctly
        backstory = call_args[1]['backstory']
        assert "IMPORTANT RELATIONSHIP INSIGHTS" in backstory
        assert "Disagreement on priority tasks" in backstory
        
        # Verify agent configuration
        assert call_args[1]['memory'] is True
        assert call_args[1]['verbose'] is True
        
    @patch('src.escape_room_sim.agents.mediator.create_mediator_agent')
    def test_create_mediator_with_context_calls_base_function(self, mock_create_mediator):
        """Test that create_mediator_with_context calls base creation function with correct parameters."""
        # Arrange
        mock_agent = Mock()
        mock_create_mediator.return_value = mock_agent
        
        previous_results = {
            "team_conflicts": ["Conflict A", "Conflict B"],
            "trust_levels": {"team": 0.7},
            "successful_collaborations": ["Collaboration 1"],
            "communication_issues": ["Issue 1"]
        }
        
        # Act
        result = create_mediator_with_context(previous_results)
        
        # Assert
        mock_create_mediator.assert_called_once()
        call_args = mock_create_mediator.call_args
        
        assert call_args[1]['memory_enabled'] is True
        assert call_args[1]['verbose'] is True
        
        # Verify context structure
        context = call_args[1]['iteration_context']
        assert context['team_conflicts'] == ["Conflict A", "Conflict B"]
        assert context['trust_levels'] == {"team": 0.7}
        assert context['successful_collaborations'] == ["Collaboration 1"]
        assert context['communication_issues'] == ["Issue 1"]
        
        assert result == mock_agent
        
    @patch('src.escape_room_sim.agents.mediator.create_mediator_agent')
    def test_create_mediator_with_context_empty_results(self, mock_create_mediator):
        """Test creating mediator with empty previous results."""
        # Arrange
        mock_agent = Mock()
        mock_create_mediator.return_value = mock_agent
        
        previous_results = {}
        
        # Act
        result = create_mediator_with_context(previous_results)
        
        # Assert
        call_args = mock_create_mediator.call_args
        context = call_args[1]['iteration_context']
        
        assert context['team_conflicts'] == []
        assert context['trust_levels'] == {}
        assert context['successful_collaborations'] == []
        assert context['communication_issues'] == []
        
    @patch('src.escape_room_sim.agents.mediator.create_mediator_agent')
    def test_create_mediator_with_context_partial_results(self, mock_create_mediator):
        """Test creating mediator with partial previous results."""
        # Arrange
        mock_agent = Mock()
        mock_create_mediator.return_value = mock_agent
        
        previous_results = {
            "team_conflicts": ["Only this conflict occurred"],
            # Missing other keys
        }
        
        # Act
        result = create_mediator_with_context(previous_results)
        
        # Assert
        call_args = mock_create_mediator.call_args
        context = call_args[1]['iteration_context']
        
        assert context['team_conflicts'] == ["Only this conflict occurred"]
        assert context['trust_levels'] == {}              # Default empty dict
        assert context['successful_collaborations'] == [] # Default empty list
        assert context['communication_issues'] == []      # Default empty list


class TestMediatorErrorHandling:
    """Test suite for error handling and edge cases."""
    
    @patch('src.escape_room_sim.agents.mediator.Agent')
    def test_create_mediator_agent_with_none_context(self, mock_agent_class):
        """Test creating mediator with None iteration context."""
        # Arrange
        mock_agent_instance = Mock()
        mock_agent_class.return_value = mock_agent_instance
        
        # Act
        agent = create_mediator_agent(iteration_context=None)
        
        # Assert - Should not raise exception and use base backstory
        call_args = mock_agent_class.call_args
        backstory = call_args[1]['backstory']
        
        assert "former crisis counselor" in backstory
        assert "IMPORTANT RELATIONSHIP INSIGHTS" not in backstory
        
    @patch('src.escape_room_sim.agents.mediator.Agent')
    def test_create_mediator_agent_with_malformed_context(self, mock_agent_class):
        """Test creating mediator with malformed iteration context."""
        # Arrange
        mock_agent_instance = Mock()
        mock_agent_class.return_value = mock_agent_instance
        
        malformed_context = {
            "team_conflicts": "Not a list",  # Should be list
            "trust_levels": "Not a dict",   # Should be dict
            "other_key": {"nested": "value"}
        }
        
        # Act & Assert - Should not raise exception
        agent = create_mediator_agent(iteration_context=malformed_context)
        
        call_args = mock_agent_class.call_args
        backstory = call_args[1]['backstory']
        
        # Should fall back to base backstory when context is malformed
        assert "former crisis counselor" in backstory
        
    @patch('src.escape_room_sim.agents.mediator.Agent', side_effect=Exception("CrewAI Error"))
    def test_create_mediator_agent_crewai_exception(self, mock_agent_class):
        """Test handling of CrewAI agent creation exceptions."""
        # Act & Assert
        with pytest.raises(Exception, match="CrewAI Error"):
            create_mediator_agent()
            
    def test_create_mediator_with_context_none_input(self):
        """Test create_mediator_with_context with None input."""
        # Act & Assert - Should not raise exception
        with patch('src.escape_room_sim.agents.mediator.create_mediator_agent') as mock_create:
            mock_agent = Mock()
            mock_create.return_value = mock_agent
            
            result = create_mediator_with_context(None)
            
            # Should call with empty context structure 
            call_args = mock_create.call_args
            context = call_args[1]['iteration_context']
            
            assert context['team_conflicts'] == []
            assert context['trust_levels'] == {}
            assert context['successful_collaborations'] == []
            assert context['communication_issues'] == []


class TestMediatorAgentProperties:
    """Test suite for verifying agent properties are set correctly."""
    
    @patch('src.escape_room_sim.agents.mediator.Agent')
    def test_mediator_agent_role_property(self, mock_agent_class):
        """Test that mediator agent role is set correctly."""
        # Arrange
        mock_agent_instance = Mock()
        mock_agent_class.return_value = mock_agent_instance
        
        # Act
        agent = create_mediator_agent()
        
        # Assert
        call_args = mock_agent_class.call_args
        assert call_args[1]['role'] == "Group Facilitator"
        
    @patch('src.escape_room_sim.agents.mediator.Agent')
    def test_mediator_agent_goal_property(self, mock_agent_class):
        """Test that mediator agent goal is set correctly."""
        # Arrange
        mock_agent_instance = Mock()
        mock_agent_class.return_value = mock_agent_instance
        
        # Act
        agent = create_mediator_agent()
        
        # Assert
        call_args = mock_agent_class.call_args
        assert call_args[1]['goal'] == "Build consensus through multiple discussion rounds and maintain team cohesion"
        
    @patch('src.escape_room_sim.agents.mediator.Agent')
    def test_mediator_agent_max_iter_property(self, mock_agent_class):
        """Test that mediator agent max_iter is set correctly."""
        # Arrange
        mock_agent_instance = Mock()
        mock_agent_class.return_value = mock_agent_instance
        
        # Act
        agent = create_mediator_agent()
        
        # Assert
        call_args = mock_agent_class.call_args
        assert call_args[1]['max_iter'] == 3
        
    @patch('src.escape_room_sim.agents.mediator.Agent')
    def test_mediator_agent_delegation_property(self, mock_agent_class):
        """Test that mediator agent delegation is disabled."""
        # Arrange
        mock_agent_instance = Mock()
        mock_agent_class.return_value = mock_agent_instance
        
        # Act
        agent = create_mediator_agent()
        
        # Assert
        call_args = mock_agent_class.call_args
        assert call_args[1]['allow_delegation'] is False
        
    @patch('src.escape_room_sim.agents.mediator.Agent')
    def test_mediator_agent_backstory_structure(self, mock_agent_class):
        """Test that mediator agent backstory has proper structure."""
        # Arrange
        mock_agent_instance = Mock()
        mock_agent_class.return_value = mock_agent_instance
        
        # Act
        agent = create_mediator_agent()
        
        # Assert
        call_args = mock_agent_class.call_args
        backstory = call_args[1]['backstory']
        
        # Should be a non-empty string
        assert isinstance(backstory, str)
        assert len(backstory.strip()) > 0
        
        # Should contain key personality elements
        assert "counselor" in backstory.lower() or "facilitator" in backstory.lower()


class TestMediatorRelationshipTracking:
    """Test suite for relationship tracking and team dynamics."""
    
    @patch('src.escape_room_sim.agents.mediator.Agent')
    def test_relationship_context_integration(self, mock_agent_class):
        """Test integration of relationship context into backstory."""
        # Arrange  
        mock_agent_instance = Mock()
        mock_agent_class.return_value = mock_agent_instance
        
        context = {
            "team_conflicts": [
                "Leadership challenge between members",
                "Resource allocation disagreement", 
                "Communication style mismatch"
            ],
            "trust_levels": {
                "overall": 0.6,
                "pairs": {"A_B": 0.8, "A_C": 0.4, "B_C": 0.7}
            }
        }
        
        # Act
        agent = create_mediator_agent(iteration_context=context)
        
        # Assert
        call_args = mock_agent_class.call_args
        backstory = call_args[1]['backstory']
        
        # Check relationship insights are added
        assert "IMPORTANT RELATIONSHIP INSIGHTS" in backstory
        assert str(len(context['team_conflicts'])) in backstory
        
        # Check conflicts are mentioned
        assert "Leadership challenge between members" in backstory
        assert "Resource allocation disagreement" in backstory
        assert "Communication style mismatch" in backstory
        
        # Check trust levels are mentioned
        assert "Trust levels you've observed" in backstory
        
        # Check guidance for relationship management
        assert "use your understanding of team relationships" in backstory
        assert "facilitate better collaboration" in backstory
        
    @patch('src.escape_room_sim.agents.mediator.Agent')
    def test_trust_levels_handling_different_formats(self, mock_agent_class):
        """Test handling of different trust level formats.""" 
        # Arrange
        mock_agent_instance = Mock()
        mock_agent_class.return_value = mock_agent_instance
        
        context = {
            "team_conflicts": ["Some conflict"],
            "trust_levels": "Building initial trust"  # String instead of dict
        }
        
        # Act
        agent = create_mediator_agent(iteration_context=context)
        
        # Assert
        call_args = mock_agent_class.call_args
        backstory = call_args[1]['backstory']
        
        # Should handle string trust levels gracefully
        assert "Trust levels you've observed: Building initial trust" in backstory
        
    @patch('src.escape_room_sim.agents.mediator.Agent')
    def test_relationship_tracking_without_trust_levels(self, mock_agent_class):
        """Test relationship tracking when trust_levels is missing."""
        # Arrange
        mock_agent_instance = Mock()
        mock_agent_class.return_value = mock_agent_instance
        
        context = {
            "team_conflicts": ["Conflict without trust data"]
        }
        
        # Act
        agent = create_mediator_agent(iteration_context=context)
        
        # Assert
        call_args = mock_agent_class.call_args
        backstory = call_args[1]['backstory']
        
        # Should use default trust level message
        assert "Trust levels you've observed: Building initial trust" in backstory
        
    @patch('src.escape_room_sim.agents.mediator.Agent')
    def test_system_message_relationship_focus(self, mock_agent_class):
        """Test that system message emphasizes relationship management."""
        # Arrange
        mock_agent_instance = Mock()
        mock_agent_class.return_value = mock_agent_instance
        
        # Act
        agent = create_mediator_agent()
        
        # Assert
        call_args = mock_agent_class.call_args
        system_message = call_args[1]['system_message']
        
        relationship_focus_indicators = [
            "checking in with team members' emotional state",
            "building trust",
            "maintaining team cohesion",
            "concerns from previous attempts"
        ]
        
        for indicator in relationship_focus_indicators:
            assert indicator in system_message