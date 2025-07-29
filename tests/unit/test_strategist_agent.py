"""
Comprehensive unit tests for the Strategist agent.

Tests cover:
- Agent creation and configuration
- Memory and learning systems  
- Agent personality and behavior
- Context-aware agent creation
- Error handling and edge cases
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any

from src.escape_room_sim.agents.strategist import (
    create_strategist_agent,
    create_strategist_with_context,
    StrategistConfig
)


class TestStrategistAgentCreation:
    """Test suite for basic strategist agent creation and configuration."""
    
    @patch('src.escape_room_sim.agents.strategist.Agent')
    def test_create_strategist_agent_default_parameters(self, mock_agent_class):
        """Test creating strategist agent with default parameters."""
        # Arrange
        mock_agent_instance = Mock()
        mock_agent_class.return_value = mock_agent_instance
        
        # Act
        agent = create_strategist_agent()
        
        # Assert
        mock_agent_class.assert_called_once()
        call_args = mock_agent_class.call_args
        
        assert call_args[1]['role'] == "Strategic Analyst"
        assert call_args[1]['goal'] == "Find the optimal solution through iterative problem-solving and learning from failures"
        assert call_args[1]['verbose'] is True
        assert call_args[1]['memory'] is True
        assert call_args[1]['max_iter'] == 3
        assert call_args[1]['allow_delegation'] is False
        assert "former military tactician" in call_args[1]['backstory']
        assert "Strategic Analyst" in call_args[1]['system_message']
        
    @patch('src.escape_room_sim.agents.strategist.Agent')
    def test_create_strategist_agent_custom_parameters(self, mock_agent_class):
        """Test creating strategist agent with custom parameters."""
        # Arrange
        mock_agent_instance = Mock()
        mock_agent_class.return_value = mock_agent_instance
        
        # Act
        agent = create_strategist_agent(
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
    @patch('src.escape_room_sim.agents.strategist.Agent')
    def test_create_strategist_agent_parameter_combinations(self, mock_agent_class, memory_enabled, verbose):
        """Test all combinations of memory_enabled and verbose parameters."""
        # Arrange
        mock_agent_instance = Mock()
        mock_agent_class.return_value = mock_agent_instance
        
        # Act
        agent = create_strategist_agent(
            memory_enabled=memory_enabled,
            verbose=verbose
        )
        
        # Assert
        call_args = mock_agent_class.call_args
        assert call_args[1]['memory'] == memory_enabled
        assert call_args[1]['verbose'] == verbose
        
    @patch('src.escape_room_sim.agents.strategist.Agent')
    def test_create_strategist_agent_returns_agent_instance(self, mock_agent_class):
        """Test that the function returns the agent instance created by CrewAI."""
        # Arrange
        mock_agent_instance = Mock()
        mock_agent_class.return_value = mock_agent_instance
        
        # Act
        result = create_strategist_agent()
        
        # Assert
        assert result == mock_agent_instance


class TestStrategistMemoryAndLearning:
    """Test suite for memory and learning systems."""
    
    @patch('src.escape_room_sim.agents.strategist.Agent')
    def test_create_strategist_agent_with_empty_context(self, mock_agent_class):
        """Test creating strategist with empty iteration context."""
        # Arrange
        mock_agent_instance = Mock()
        mock_agent_class.return_value = mock_agent_instance
        empty_context = {}
        
        # Act
        agent = create_strategist_agent(iteration_context=empty_context)
        
        # Assert
        call_args = mock_agent_class.call_args
        backstory = call_args[1]['backstory']
        
        # Should use base backstory only
        assert "former military tactician" in backstory
        assert "IMPORTANT LEARNING FROM PREVIOUS ATTEMPTS" not in backstory
        
    @patch('src.escape_room_sim.agents.strategist.Agent')
    def test_create_strategist_agent_with_failed_strategies_context(self, mock_agent_class, strategist_context):
        """Test creating strategist with failed strategies in context."""
        # Arrange
        mock_agent_instance = Mock()
        mock_agent_class.return_value = mock_agent_instance
        
        # Act
        agent = create_strategist_agent(iteration_context=strategist_context)
        
        # Assert
        call_args = mock_agent_class.call_args
        backstory = call_args[1]['backstory']
        
        assert "IMPORTANT LEARNING FROM PREVIOUS ATTEMPTS" in backstory
        assert "Direct assault on main door failed" in backstory
        assert "Overcomplicated puzzle solution approach" in backstory
        assert "avoid repeating these approaches" in backstory
        
    @patch('src.escape_room_sim.agents.strategist.Agent')
    def test_create_strategist_agent_limits_failed_strategies_display(self, mock_agent_class):
        """Test that only first 3 failed strategies are displayed in backstory."""
        # Arrange
        mock_agent_instance = Mock()
        mock_agent_class.return_value = mock_agent_instance
        
        context = {
            "failed_strategies": [
                "Strategy 1", "Strategy 2", "Strategy 3", 
                "Strategy 4", "Strategy 5", "Strategy 6"
            ]
        }
        
        # Act
        agent = create_strategist_agent(iteration_context=context)
        
        # Assert
        call_args = mock_agent_class.call_args
        backstory = call_args[1]['backstory']
        
        assert "Strategy 1" in backstory
        assert "Strategy 2" in backstory
        assert "Strategy 3" in backstory
        assert "Strategy 4" not in backstory
        assert "Strategy 5" not in backstory
        assert "Strategy 6" not in backstory
        
    @patch('src.escape_room_sim.agents.strategist.Agent')
    def test_create_strategist_agent_context_without_failed_strategies(self, mock_agent_class):
        """Test creating strategist with context that has no failed_strategies key."""
        # Arrange
        mock_agent_instance = Mock()
        mock_agent_class.return_value = mock_agent_instance
        
        context = {
            "successful_approaches": ["Some success"],
            "resource_constraints": {"time": 30}
        }
        
        # Act
        agent = create_strategist_agent(iteration_context=context)
        
        # Assert
        call_args = mock_agent_class.call_args
        backstory = call_args[1]['backstory']
        
        # Should use base backstory since no failed_strategies
        assert "former military tactician" in backstory
        assert "IMPORTANT LEARNING FROM PREVIOUS ATTEMPTS" not in backstory


class TestStrategistPersonalityAndBehavior:
    """Test suite for agent personality and behavior configuration."""
    
    @patch('src.escape_room_sim.agents.strategist.Agent')
    def test_create_strategist_agent_system_message_content(self, mock_agent_class):
        """Test that system message contains expected behavioral instructions."""
        # Arrange
        mock_agent_instance = Mock()
        mock_agent_class.return_value = mock_agent_instance
        
        # Act
        agent = create_strategist_agent()
        
        # Assert
        call_args = mock_agent_class.call_args
        system_message = call_args[1]['system_message']
        
        expected_behaviors = [
            "Systematically analyze the escape room situation",
            "Learn from previous failed attempts",
            "Consider all team members' capabilities",
            "Propose logical, evidence-based solutions",
            "Balance optimal outcomes with practical limitations",
            "Track what works and what doesn't work"
        ]
        
        for behavior in expected_behaviors:
            assert behavior in system_message
            
    @patch('src.escape_room_sim.agents.strategist.Agent')
    def test_create_strategist_agent_backstory_personality_traits(self, mock_agent_class):
        """Test that backstory reflects strategist personality traits."""
        # Arrange
        mock_agent_instance = Mock()
        mock_agent_class.return_value = mock_agent_instance
        
        # Act
        agent = create_strategist_agent()
        
        # Assert
        call_args = mock_agent_class.call_args
        backstory = call_args[1]['backstory']
        
        personality_indicators = [
            "former military tactician",
            "strategic planning",
            "crisis management", 
            "systematically",
            "analyzing all available information",
            "solution saves the most lives"  # Adjusted to match actual text
        ]
        
        for indicator in personality_indicators:
            assert indicator in backstory
            
    def test_strategist_config_personality_traits(self):
        """Test StrategistConfig personality traits configuration."""
        # Act
        traits = StrategistConfig.get_personality_traits()
        
        # Assert
        expected_traits = {
            "risk_tolerance": "moderate",
            "collaboration_style": "analytical",
            "learning_rate": "high",
            "decision_style": "evidence_based",
            "communication_style": "direct_analytical"
        }
        
        assert traits == expected_traits
        
    def test_strategist_config_constants(self):
        """Test StrategistConfig constants are properly defined."""
        # Assert
        assert StrategistConfig.DEFAULT_ROLE == "Strategic Analyst"
        assert StrategistConfig.DEFAULT_GOAL == "Find the optimal solution through iterative problem-solving and learning from failures"
        assert StrategistConfig.RISK_TOLERANCE == "moderate"
        assert StrategistConfig.COLLABORATION_STYLE == "analytical"
        assert StrategistConfig.LEARNING_RATE == "high"


class TestStrategistContextAwareCreation:
    """Test suite for context-aware agent creation."""
    
    @patch('src.escape_room_sim.agents.strategist.Agent')
    def test_create_strategist_with_context_comprehensive(self, mock_agent_class, previous_results_comprehensive):
        """Test creating strategist with comprehensive context data."""
        # Arrange
        mock_agent_instance = Mock()
        mock_agent_class.return_value = mock_agent_instance
        
        # Act
        agent = create_strategist_with_context(previous_results_comprehensive)
        
        # Assert
        mock_agent_class.assert_called_once()
        call_args = mock_agent_class.call_args
        
        # Verify context was processed correctly
        backstory = call_args[1]['backstory']
        assert "IMPORTANT LEARNING FROM PREVIOUS ATTEMPTS" in backstory
        assert "Strategy 1 failed due to time constraints" in backstory
        
        # Verify agent configuration
        assert call_args[1]['memory'] is True
        assert call_args[1]['verbose'] is True
        
    @patch('src.escape_room_sim.agents.strategist.create_strategist_agent')
    def test_create_strategist_with_context_calls_base_function(self, mock_create_strategist):
        """Test that create_strategist_with_context calls base creation function with correct parameters."""
        # Arrange
        mock_agent = Mock()
        mock_create_strategist.return_value = mock_agent
        
        previous_results = {
            "failed_strategies": ["Strategy A", "Strategy B"],
            "successful_approaches": ["Approach 1"],
            "resource_constraints": {"time": 30},
            "team_dynamics": {"trust": 0.8}
        }
        
        # Act
        result = create_strategist_with_context(previous_results)
        
        # Assert
        mock_create_strategist.assert_called_once()
        call_args = mock_create_strategist.call_args
        
        assert call_args[1]['memory_enabled'] is True
        assert call_args[1]['verbose'] is True
        
        # Verify context structure
        context = call_args[1]['iteration_context']
        assert context['failed_strategies'] == ["Strategy A", "Strategy B"]
        assert context['successful_approaches'] == ["Approach 1"]
        assert context['resource_constraints'] == {"time": 30}
        assert context['team_dynamics'] == {"trust": 0.8}
        
        assert result == mock_agent
        
    @patch('src.escape_room_sim.agents.strategist.create_strategist_agent')
    def test_create_strategist_with_context_empty_results(self, mock_create_strategist):
        """Test creating strategist with empty previous results."""
        # Arrange
        mock_agent = Mock()
        mock_create_strategist.return_value = mock_agent
        
        previous_results = {}
        
        # Act
        result = create_strategist_with_context(previous_results)
        
        # Assert
        call_args = mock_create_strategist.call_args
        context = call_args[1]['iteration_context']
        
        assert context['failed_strategies'] == []
        assert context['successful_approaches'] == []
        assert context['resource_constraints'] == {}
        assert context['team_dynamics'] == {}
        
    @patch('src.escape_room_sim.agents.strategist.create_strategist_agent')
    def test_create_strategist_with_context_partial_results(self, mock_create_strategist):
        """Test creating strategist with partial previous results."""
        # Arrange
        mock_agent = Mock()
        mock_create_strategist.return_value = mock_agent
        
        previous_results = {
            "failed_strategies": ["Only this strategy failed"],
            # Missing other keys
        }
        
        # Act
        result = create_strategist_with_context(previous_results)
        
        # Assert
        call_args = mock_create_strategist.call_args
        context = call_args[1]['iteration_context']
        
        assert context['failed_strategies'] == ["Only this strategy failed"]
        assert context['successful_approaches'] == []  # Default empty list
        assert context['resource_constraints'] == {}   # Default empty dict
        assert context['team_dynamics'] == {}          # Default empty dict


class TestStrategistErrorHandling:
    """Test suite for error handling and edge cases."""
    
    @patch('src.escape_room_sim.agents.strategist.Agent')
    def test_create_strategist_agent_with_none_context(self, mock_agent_class):
        """Test creating strategist with None iteration context."""
        # Arrange
        mock_agent_instance = Mock()
        mock_agent_class.return_value = mock_agent_instance
        
        # Act
        agent = create_strategist_agent(iteration_context=None)
        
        # Assert - Should not raise exception and use base backstory
        call_args = mock_agent_class.call_args
        backstory = call_args[1]['backstory']
        
        assert "former military tactician" in backstory
        assert "IMPORTANT LEARNING FROM PREVIOUS ATTEMPTS" not in backstory
        
    @patch('src.escape_room_sim.agents.strategist.Agent')
    def test_create_strategist_agent_with_malformed_context(self, mock_agent_class):
        """Test creating strategist with malformed iteration context."""
        # Arrange
        mock_agent_instance = Mock()
        mock_agent_class.return_value = mock_agent_instance
        
        malformed_context = {
            "failed_strategies": "Not a list",  # Should be list
            "other_key": {"nested": "value"}
        }
        
        # Act & Assert - Should not raise exception
        agent = create_strategist_agent(iteration_context=malformed_context)
        
        call_args = mock_agent_class.call_args
        backstory = call_args[1]['backstory']
        
        # Should fall back to base backstory when context is malformed
        assert "former military tactician" in backstory
        
    @patch('src.escape_room_sim.agents.strategist.Agent', side_effect=Exception("CrewAI Error"))
    def test_create_strategist_agent_crewai_exception(self, mock_agent_class):
        """Test handling of CrewAI agent creation exceptions."""
        # Act & Assert
        with pytest.raises(Exception, match="CrewAI Error"):
            create_strategist_agent()
            
    def test_create_strategist_with_context_none_input(self):
        """Test create_strategist_with_context with None input."""
        # Act & Assert - Should not raise exception
        with patch('src.escape_room_sim.agents.strategist.create_strategist_agent') as mock_create:
            mock_agent = Mock()
            mock_create.return_value = mock_agent
            
            result = create_strategist_with_context(None)
            
            # Should call with empty context structure 
            call_args = mock_create.call_args
            context = call_args[1]['iteration_context']
            
            assert context['failed_strategies'] == []
            assert context['successful_approaches'] == []
            assert context['resource_constraints'] == {}
            assert context['team_dynamics'] == {}


class TestStrategistAgentProperties:
    """Test suite for verifying agent properties are set correctly."""
    
    @patch('src.escape_room_sim.agents.strategist.Agent')
    def test_strategist_agent_role_property(self, mock_agent_class):
        """Test that strategist agent role is set correctly."""
        # Arrange
        mock_agent_instance = Mock()
        mock_agent_class.return_value = mock_agent_instance
        
        # Act
        agent = create_strategist_agent()
        
        # Assert
        call_args = mock_agent_class.call_args
        assert call_args[1]['role'] == "Strategic Analyst"
        
    @patch('src.escape_room_sim.agents.strategist.Agent')
    def test_strategist_agent_goal_property(self, mock_agent_class):
        """Test that strategist agent goal is set correctly."""
        # Arrange
        mock_agent_instance = Mock()
        mock_agent_class.return_value = mock_agent_instance
        
        # Act
        agent = create_strategist_agent()
        
        # Assert
        call_args = mock_agent_class.call_args
        assert call_args[1]['goal'] == "Find the optimal solution through iterative problem-solving and learning from failures"
        
    @patch('src.escape_room_sim.agents.strategist.Agent')
    def test_strategist_agent_max_iter_property(self, mock_agent_class):
        """Test that strategist agent max_iter is set correctly."""
        # Arrange
        mock_agent_instance = Mock()
        mock_agent_class.return_value = mock_agent_instance
        
        # Act
        agent = create_strategist_agent()
        
        # Assert
        call_args = mock_agent_class.call_args
        assert call_args[1]['max_iter'] == 3
        
    @patch('src.escape_room_sim.agents.strategist.Agent')
    def test_strategist_agent_delegation_property(self, mock_agent_class):
        """Test that strategist agent delegation is disabled."""
        # Arrange
        mock_agent_instance = Mock()
        mock_agent_class.return_value = mock_agent_instance
        
        # Act
        agent = create_strategist_agent()
        
        # Assert
        call_args = mock_agent_class.call_args
        assert call_args[1]['allow_delegation'] is False
        
    @patch('src.escape_room_sim.agents.strategist.Agent')
    def test_strategist_agent_backstory_structure(self, mock_agent_class):
        """Test that strategist agent backstory has proper structure."""
        # Arrange
        mock_agent_instance = Mock()
        mock_agent_class.return_value = mock_agent_instance
        
        # Act
        agent = create_strategist_agent()
        
        # Assert
        call_args = mock_agent_class.call_args
        backstory = call_args[1]['backstory']
        
        # Should be a non-empty string
        assert isinstance(backstory, str)
        assert len(backstory.strip()) > 0
        
        # Should contain key personality elements
        assert "military" in backstory.lower() or "tactical" in backstory.lower()


class TestStrategistBackstoryAdaptation:
    """Test suite for backstory adaptation based on context."""
    
    @patch('src.escape_room_sim.agents.strategist.Agent')
    def test_backstory_adaptation_with_multiple_failures(self, mock_agent_class):
        """Test backstory adaptation when multiple failures are present."""
        # Arrange  
        mock_agent_instance = Mock()
        mock_agent_class.return_value = mock_agent_instance
        
        context = {
            "failed_strategies": [
                "First failed approach - time management",
                "Second failed approach - resource allocation", 
                "Third failed approach - team coordination",
                "Fourth failed approach - should not appear"
            ]
        }
        
        # Act
        agent = create_strategist_agent(iteration_context=context)
        
        # Assert
        call_args = mock_agent_class.call_args
        backstory = call_args[1]['backstory']
        
        # Check learning context is added
        assert "IMPORTANT LEARNING FROM PREVIOUS ATTEMPTS" in backstory
        assert str(len(context['failed_strategies'])) in backstory
        
        # Check only first 3 strategies are mentioned
        assert "First failed approach" in backstory
        assert "Second failed approach" in backstory
        assert "Third failed approach" in backstory
        assert "Fourth failed approach" not in backstory
        
        # Check adaptation guidance
        assert "avoid repeating these approaches" in backstory
        assert "adapt your strategy based on" in backstory
        
    @patch('src.escape_room_sim.agents.strategist.Agent')
    def test_backstory_adaptation_single_failure(self, mock_agent_class):
        """Test backstory adaptation with single failure."""
        # Arrange
        mock_agent_instance = Mock()
        mock_agent_class.return_value = mock_agent_instance
        
        context = {
            "failed_strategies": ["Only one strategy failed"]
        }
        
        # Act
        agent = create_strategist_agent(iteration_context=context)
        
        # Assert
        call_args = mock_agent_class.call_args
        backstory = call_args[1]['backstory']
        
        assert "IMPORTANT LEARNING FROM PREVIOUS ATTEMPTS" in backstory
        assert "learned from 1 previous failed" in backstory
        assert "Only one strategy failed" in backstory
        
    @patch('src.escape_room_sim.agents.strategist.Agent')
    def test_backstory_adaptation_empty_failures_list(self, mock_agent_class):
        """Test backstory when failed_strategies list is empty."""
        # Arrange
        mock_agent_instance = Mock()
        mock_agent_class.return_value = mock_agent_instance
        
        context = {
            "failed_strategies": []
        }
        
        # Act
        agent = create_strategist_agent(iteration_context=context)
        
        # Assert
        call_args = mock_agent_class.call_args
        backstory = call_args[1]['backstory']
        
        # Should not add learning context for empty failures
        assert "IMPORTANT LEARNING FROM PREVIOUS ATTEMPTS" not in backstory
        assert "former military tactician" in backstory