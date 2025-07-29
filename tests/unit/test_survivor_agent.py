"""
Comprehensive unit tests for the Survivor agent.

Tests cover:
- Agent creation and configuration
- Memory and learning systems  
- Agent personality and behavior
- Context-aware agent creation
- Survival decision criteria and priorities
- Error handling and edge cases
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any

from src.escape_room_sim.agents.survivor import (
    create_survivor_agent,
    create_survivor_with_context,
    SurvivorConfig
)


class TestSurvivorAgentCreation:
    """Test suite for basic survivor agent creation and configuration."""
    
    @patch('src.escape_room_sim.agents.survivor.Agent')
    def test_create_survivor_agent_default_parameters(self, mock_agent_class):
        """Test creating survivor agent with default parameters."""
        # Arrange
        mock_agent_instance = Mock()
        mock_agent_class.return_value = mock_agent_instance
        
        # Act
        agent = create_survivor_agent()
        
        # Assert
        mock_agent_class.assert_called_once()
        call_args = mock_agent_class.call_args
        
        assert call_args[1]['role'] == "Survival Specialist"
        assert call_args[1]['goal'] == "Ensure survival through adaptive decision-making and efficient execution"
        assert call_args[1]['verbose'] is True
        assert call_args[1]['memory'] is True
        assert call_args[1]['max_iter'] == 3
        assert call_args[1]['allow_delegation'] is False
        assert "former special forces operator" in call_args[1]['backstory']
        assert "Survival Specialist" in call_args[1]['system_message']
        
    @patch('src.escape_room_sim.agents.survivor.Agent')
    def test_create_survivor_agent_custom_parameters(self, mock_agent_class):
        """Test creating survivor agent with custom parameters."""
        # Arrange
        mock_agent_instance = Mock()
        mock_agent_class.return_value = mock_agent_instance
        
        # Act
        agent = create_survivor_agent(
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
    @patch('src.escape_room_sim.agents.survivor.Agent')
    def test_create_survivor_agent_parameter_combinations(self, mock_agent_class, memory_enabled, verbose):
        """Test all combinations of memory_enabled and verbose parameters."""
        # Arrange
        mock_agent_instance = Mock()
        mock_agent_class.return_value = mock_agent_instance
        
        # Act
        agent = create_survivor_agent(
            memory_enabled=memory_enabled,
            verbose=verbose
        )
        
        # Assert
        call_args = mock_agent_class.call_args
        assert call_args[1]['memory'] == memory_enabled
        assert call_args[1]['verbose'] == verbose
        
    @patch('src.escape_room_sim.agents.survivor.Agent')
    def test_create_survivor_agent_returns_agent_instance(self, mock_agent_class):
        """Test that the function returns the agent instance created by CrewAI."""
        # Arrange
        mock_agent_instance = Mock()
        mock_agent_class.return_value = mock_agent_instance
        
        # Act
        result = create_survivor_agent()
        
        # Assert
        assert result == mock_agent_instance


class TestSurvivorMemoryAndLearning:
    """Test suite for memory and learning systems."""
    
    @patch('src.escape_room_sim.agents.survivor.Agent')
    def test_create_survivor_agent_with_empty_context(self, mock_agent_class):
        """Test creating survivor with empty iteration context."""
        # Arrange
        mock_agent_instance = Mock()
        mock_agent_class.return_value = mock_agent_instance
        empty_context = {}
        
        # Act
        agent = create_survivor_agent(iteration_context=empty_context)
        
        # Assert
        call_args = mock_agent_class.call_args
        backstory = call_args[1]['backstory']
        
        # Should use base backstory only
        assert "former special forces operator" in backstory
        assert "CRITICAL SURVIVAL LESSONS" not in backstory
        
    @patch('src.escape_room_sim.agents.survivor.Agent')
    def test_create_survivor_agent_with_survival_lessons_context(self, mock_agent_class, survivor_context):
        """Test creating survivor with survival lessons in context."""
        # Arrange
        mock_agent_instance = Mock()
        mock_agent_class.return_value = mock_agent_instance
        
        # Act
        agent = create_survivor_agent(iteration_context=survivor_context)
        
        # Assert
        call_args = mock_agent_class.call_args
        backstory = call_args[1]['backstory']
        
        assert "CRITICAL SURVIVAL LESSONS" in backstory
        assert "Moving too slowly cost valuable time" in backstory
        assert "Overthinking simple problems was dangerous" in backstory
        assert "apply these hard-earned lessons" in backstory
        
    @patch('src.escape_room_sim.agents.survivor.Agent')
    def test_create_survivor_agent_with_resource_insights(self, mock_agent_class, survivor_context):
        """Test creating survivor with resource insights in context."""
        # Arrange
        mock_agent_instance = Mock()
        mock_agent_class.return_value = mock_agent_instance
        
        # Act
        agent = create_survivor_agent(iteration_context=survivor_context)
        
        # Assert
        call_args = mock_agent_class.call_args
        backstory = call_args[1]['backstory']
        
        resource_insights = survivor_context['resource_insights']
        assert "Resource efficiency insights" in backstory
        # Should contain resource insight information
        for insight_key, insight_value in resource_insights.items():
            insight_str = str(insight_value)
            if len(insight_str) > 2:  # Skip very short strings
                assert insight_str in backstory or insight_key in backstory
        
    @patch('src.escape_room_sim.agents.survivor.Agent')
    def test_create_survivor_agent_limits_survival_lessons_display(self, mock_agent_class):
        """Test that only first 3 survival lessons are displayed in backstory."""
        # Arrange
        mock_agent_instance = Mock()
        mock_agent_class.return_value = mock_agent_instance
        
        context = {
            "survival_lessons": [
                "Lesson 1", "Lesson 2", "Lesson 3", 
                "Lesson 4", "Lesson 5", "Lesson 6"
            ]
        }
        
        # Act
        agent = create_survivor_agent(iteration_context=context)
        
        # Assert
        call_args = mock_agent_class.call_args
        backstory = call_args[1]['backstory']
        
        assert "Lesson 1" in backstory
        assert "Lesson 2" in backstory
        assert "Lesson 3" in backstory
        assert "Lesson 4" not in backstory
        assert "Lesson 5" not in backstory
        assert "Lesson 6" not in backstory
        
    @patch('src.escape_room_sim.agents.survivor.Agent')
    def test_create_survivor_agent_context_without_survival_lessons(self, mock_agent_class):
        """Test creating survivor with context that has no survival_lessons key."""
        # Arrange
        mock_agent_instance = Mock()
        mock_agent_class.return_value = mock_agent_instance
        
        context = {
            "resource_insights": {"efficiency": 0.7},
            "execution_failures": ["Some failure"]
        }
        
        # Act
        agent = create_survivor_agent(iteration_context=context)
        
        # Assert
        call_args = mock_agent_class.call_args
        backstory = call_args[1]['backstory']
        
        # Should use base backstory since no survival_lessons
        assert "former special forces operator" in backstory
        assert "CRITICAL SURVIVAL LESSONS" not in backstory


class TestSurvivorPersonalityAndBehavior:
    """Test suite for agent personality and behavior configuration."""
    
    @patch('src.escape_room_sim.agents.survivor.Agent')
    def test_create_survivor_agent_system_message_content(self, mock_agent_class):
        """Test that system message contains expected behavioral instructions."""
        # Arrange
        mock_agent_instance = Mock()
        mock_agent_class.return_value = mock_agent_instance
        
        # Act
        agent = create_survivor_agent()
        
        # Assert
        call_args = mock_agent_class.call_args
        system_message = call_args[1]['system_message']
        
        expected_behaviors = [
            "Execute agreed-upon plans quickly and decisively",
            "Assess risks and benefits of each proposed action",
            "Adapt strategies in real-time based on changing conditions",
            "Learn from execution failures to improve future attempts",
            "Balance team survival with individual survival instincts",
            "Make tough decisions when consensus isn't possible"
        ]
        
        for behavior in expected_behaviors:
            assert behavior in system_message
            
    @patch('src.escape_room_sim.agents.survivor.Agent')
    def test_create_survivor_agent_backstory_personality_traits(self, mock_agent_class):
        """Test that backstory reflects survivor personality traits."""
        # Arrange
        mock_agent_instance = Mock()
        mock_agent_class.return_value = mock_agent_instance
        
        # Act
        agent = create_survivor_agent()
        
        # Assert
        call_args = mock_agent_class.call_args
        backstory = call_args[1]['backstory']
        
        personality_indicators = [
            "former special forces operator",
            "high-pressure survival situations",
            "making quick, decisive actions", 
            "incomplete information",
            "Trust is earned through actions",
            "practical solutions over idealistic plans"
        ]
        
        for indicator in personality_indicators:
            assert indicator in backstory
            
    def test_survivor_config_personality_traits(self):
        """Test SurvivorConfig personality traits configuration."""
        # Act
        traits = SurvivorConfig.get_personality_traits()
        
        # Assert
        expected_traits = {
            "risk_tolerance": "calculated",
            "decision_speed": "fast",
            "team_loyalty": "conditional",
            "pragmatism_level": "high",
            "decision_style": "action_oriented",
            "communication_style": "direct_practical"
        }
        
        assert traits == expected_traits
        
    def test_survivor_config_survival_priorities(self):
        """Test SurvivorConfig survival priorities configuration."""
        # Act
        priorities = SurvivorConfig.get_survival_priorities()
        
        # Assert
        expected_priorities = {
            "self_preservation": 1,
            "team_survival": 2,
            "mission_completion": 3,
            "resource_conservation": 4,
            "relationship_maintenance": 5
        }
        
        assert priorities == expected_priorities
        
        # Verify priority ordering is correct (lower numbers = higher priority)
        assert priorities["self_preservation"] < priorities["team_survival"]
        assert priorities["team_survival"] < priorities["mission_completion"]
        assert priorities["mission_completion"] < priorities["resource_conservation"]
        assert priorities["resource_conservation"] < priorities["relationship_maintenance"]
        
    def test_survivor_config_decision_criteria(self):
        """Test SurvivorConfig decision criteria configuration."""
        # Act
        criteria = SurvivorConfig.get_decision_criteria()
        
        # Assert
        expected_criteria = {
            "survival_probability": 0.4,
            "resource_efficiency": 0.3,
            "execution_feasibility": 0.2,
            "team_consensus": 0.1
        }
        
        assert criteria == expected_criteria
        
        # Verify weights sum close to 1.0
        total_weight = sum(criteria.values())
        assert abs(total_weight - 1.0) < 0.01
        
        # Verify survival is highest weighted criterion
        assert criteria["survival_probability"] > criteria["resource_efficiency"]
        assert criteria["survival_probability"] > criteria["execution_feasibility"] 
        assert criteria["survival_probability"] > criteria["team_consensus"]
        
    def test_survivor_config_constants(self):
        """Test SurvivorConfig constants are properly defined."""
        # Assert
        assert SurvivorConfig.DEFAULT_ROLE == "Survival Specialist"
        assert SurvivorConfig.DEFAULT_GOAL == "Ensure survival through adaptive decision-making and efficient execution"
        assert SurvivorConfig.RISK_TOLERANCE == "calculated"
        assert SurvivorConfig.DECISION_SPEED == "fast"
        assert SurvivorConfig.TEAM_LOYALTY == "conditional"
        assert SurvivorConfig.PRAGMATISM_LEVEL == "high"


class TestSurvivorContextAwareCreation:
    """Test suite for context-aware agent creation."""
    
    @patch('src.escape_room_sim.agents.survivor.Agent')
    def test_create_survivor_with_context_comprehensive(self, mock_agent_class, previous_results_comprehensive):
        """Test creating survivor with comprehensive context data."""
        # Arrange
        mock_agent_instance = Mock()
        mock_agent_class.return_value = mock_agent_instance
        
        # Act
        agent = create_survivor_with_context(previous_results_comprehensive)
        
        # Assert
        mock_agent_class.assert_called_once()
        call_args = mock_agent_class.call_args
        
        # Verify context was processed correctly
        backstory = call_args[1]['backstory']
        assert "CRITICAL SURVIVAL LESSONS" in backstory
        assert "Speed over perfection in crisis" in backstory
        
        # Verify agent configuration
        assert call_args[1]['memory'] is True
        assert call_args[1]['verbose'] is True
        
    @patch('src.escape_room_sim.agents.survivor.create_survivor_agent')
    def test_create_survivor_with_context_calls_base_function(self, mock_create_survivor):
        """Test that create_survivor_with_context calls base creation function with correct parameters."""
        # Arrange
        mock_agent = Mock()
        mock_create_survivor.return_value = mock_agent
        
        previous_results = {
            "survival_lessons": ["Lesson A", "Lesson B"],
            "resource_insights": {"efficiency": 0.6},
            "execution_failures": ["Failure 1"],
            "successful_tactics": ["Tactic 1"]
        }
        
        # Act
        result = create_survivor_with_context(previous_results)
        
        # Assert
        mock_create_survivor.assert_called_once()
        call_args = mock_create_survivor.call_args
        
        assert call_args[1]['memory_enabled'] is True
        assert call_args[1]['verbose'] is True
        
        # Verify context structure
        context = call_args[1]['iteration_context']
        assert context['survival_lessons'] == ["Lesson A", "Lesson B"]
        assert context['resource_insights'] == {"efficiency": 0.6}
        assert context['execution_failures'] == ["Failure 1"]
        assert context['successful_tactics'] == ["Tactic 1"]
        
        assert result == mock_agent
        
    @patch('src.escape_room_sim.agents.survivor.create_survivor_agent')
    def test_create_survivor_with_context_empty_results(self, mock_create_survivor):
        """Test creating survivor with empty previous results."""
        # Arrange
        mock_agent = Mock()
        mock_create_survivor.return_value = mock_agent
        
        previous_results = {}
        
        # Act
        result = create_survivor_with_context(previous_results)
        
        # Assert
        call_args = mock_create_survivor.call_args
        context = call_args[1]['iteration_context']
        
        assert context['survival_lessons'] == []
        assert context['resource_insights'] == {}
        assert context['execution_failures'] == []
        assert context['successful_tactics'] == []
        
    @patch('src.escape_room_sim.agents.survivor.create_survivor_agent')
    def test_create_survivor_with_context_partial_results(self, mock_create_survivor):
        """Test creating survivor with partial previous results."""
        # Arrange
        mock_agent = Mock()
        mock_create_survivor.return_value = mock_agent
        
        previous_results = {
            "survival_lessons": ["Only this lesson learned"],
            # Missing other keys
        }
        
        # Act
        result = create_survivor_with_context(previous_results)
        
        # Assert
        call_args = mock_create_survivor.call_args
        context = call_args[1]['iteration_context']
        
        assert context['survival_lessons'] == ["Only this lesson learned"]
        assert context['resource_insights'] == {}    # Default empty dict
        assert context['execution_failures'] == []   # Default empty list
        assert context['successful_tactics'] == []   # Default empty list


class TestSurvivorErrorHandling:
    """Test suite for error handling and edge cases."""
    
    @patch('src.escape_room_sim.agents.survivor.Agent')
    def test_create_survivor_agent_with_none_context(self, mock_agent_class):
        """Test creating survivor with None iteration context."""
        # Arrange
        mock_agent_instance = Mock()
        mock_agent_class.return_value = mock_agent_instance
        
        # Act
        agent = create_survivor_agent(iteration_context=None)
        
        # Assert - Should not raise exception and use base backstory
        call_args = mock_agent_class.call_args
        backstory = call_args[1]['backstory']
        
        assert "former special forces operator" in backstory
        assert "CRITICAL SURVIVAL LESSONS" not in backstory
        
    @patch('src.escape_room_sim.agents.survivor.Agent')
    def test_create_survivor_agent_with_malformed_context(self, mock_agent_class):
        """Test creating survivor with malformed iteration context."""
        # Arrange
        mock_agent_instance = Mock()
        mock_agent_class.return_value = mock_agent_instance
        
        malformed_context = {
            "survival_lessons": "Not a list",      # Should be list
            "resource_insights": "Not a dict",    # Should be dict
            "other_key": {"nested": "value"}
        }
        
        # Act & Assert - Should not raise exception
        agent = create_survivor_agent(iteration_context=malformed_context)
        
        call_args = mock_agent_class.call_args
        backstory = call_args[1]['backstory']
        
        # Should fall back to base backstory when context is malformed
        assert "former special forces operator" in backstory
        
    @patch('src.escape_room_sim.agents.survivor.Agent', side_effect=Exception("CrewAI Error"))
    def test_create_survivor_agent_crewai_exception(self, mock_agent_class):
        """Test handling of CrewAI agent creation exceptions."""
        # Act & Assert
        with pytest.raises(Exception, match="CrewAI Error"):
            create_survivor_agent()
            
    def test_create_survivor_with_context_none_input(self):
        """Test create_survivor_with_context with None input."""
        # Act & Assert - Should not raise exception
        with patch('src.escape_room_sim.agents.survivor.create_survivor_agent') as mock_create:
            mock_agent = Mock()
            mock_create.return_value = mock_agent
            
            result = create_survivor_with_context(None)
            
            # Should call with empty context structure 
            call_args = mock_create.call_args
            context = call_args[1]['iteration_context']
            
            assert context['survival_lessons'] == []
            assert context['resource_insights'] == {}
            assert context['execution_failures'] == []
            assert context['successful_tactics'] == []


class TestSurvivorAgentProperties:
    """Test suite for verifying agent properties are set correctly."""
    
    @patch('src.escape_room_sim.agents.survivor.Agent')
    def test_survivor_agent_role_property(self, mock_agent_class):
        """Test that survivor agent role is set correctly."""
        # Arrange
        mock_agent_instance = Mock()
        mock_agent_class.return_value = mock_agent_instance
        
        # Act
        agent = create_survivor_agent()
        
        # Assert
        call_args = mock_agent_class.call_args
        assert call_args[1]['role'] == "Survival Specialist"
        
    @patch('src.escape_room_sim.agents.survivor.Agent')
    def test_survivor_agent_goal_property(self, mock_agent_class):
        """Test that survivor agent goal is set correctly."""
        # Arrange
        mock_agent_instance = Mock()
        mock_agent_class.return_value = mock_agent_instance
        
        # Act
        agent = create_survivor_agent()
        
        # Assert
        call_args = mock_agent_class.call_args
        assert call_args[1]['goal'] == "Ensure survival through adaptive decision-making and efficient execution"
        
    @patch('src.escape_room_sim.agents.survivor.Agent')
    def test_survivor_agent_max_iter_property(self, mock_agent_class):
        """Test that survivor agent max_iter is set correctly."""
        # Arrange
        mock_agent_instance = Mock()
        mock_agent_class.return_value = mock_agent_instance
        
        # Act
        agent = create_survivor_agent()
        
        # Assert
        call_args = mock_agent_class.call_args
        assert call_args[1]['max_iter'] == 3
        
    @patch('src.escape_room_sim.agents.survivor.Agent')
    def test_survivor_agent_delegation_property(self, mock_agent_class):
        """Test that survivor agent delegation is disabled."""
        # Arrange
        mock_agent_instance = Mock()
        mock_agent_class.return_value = mock_agent_instance
        
        # Act
        agent = create_survivor_agent()
        
        # Assert
        call_args = mock_agent_class.call_args
        assert call_args[1]['allow_delegation'] is False
        
    @patch('src.escape_room_sim.agents.survivor.Agent')
    def test_survivor_agent_backstory_structure(self, mock_agent_class):
        """Test that survivor agent backstory has proper structure."""
        # Arrange
        mock_agent_instance = Mock()
        mock_agent_class.return_value = mock_agent_instance
        
        # Act
        agent = create_survivor_agent()
        
        # Assert
        call_args = mock_agent_class.call_args
        backstory = call_args[1]['backstory']
        
        # Should be a non-empty string
        assert isinstance(backstory, str)
        assert len(backstory.strip()) > 0
        
        # Should contain key personality elements
        assert "special forces" in backstory.lower() or "survival" in backstory.lower()


class TestSurvivorSurvivalSpecificFeatures:
    """Test suite for survival-specific features and decision making."""
    
    @patch('src.escape_room_sim.agents.survivor.Agent')
    def test_survival_context_integration(self, mock_agent_class):
        """Test integration of survival context into backstory."""
        # Arrange  
        mock_agent_instance = Mock()
        mock_agent_class.return_value = mock_agent_instance
        
        context = {
            "survival_lessons": [
                "Speed is more important than perfection",
                "Resource conservation saves lives", 
                "Trust your instincts in crisis"
            ],
            "resource_insights": {
                "efficiency_score": 0.75,
                "critical_resources": ["time", "energy"],
                "waste_patterns": ["overanalysis", "hesitation"]
            }
        }
        
        # Act
        agent = create_survivor_agent(iteration_context=context)
        
        # Assert
        call_args = mock_agent_class.call_args
        backstory = call_args[1]['backstory']
        
        # Check survival lessons are added
        assert "CRITICAL SURVIVAL LESSONS" in backstory
        assert str(len(context['survival_lessons'])) in backstory
        
        # Check lessons are mentioned
        assert "Speed is more important than perfection" in backstory
        assert "Resource conservation saves lives" in backstory
        assert "Trust your instincts in crisis" in backstory
        
        # Check resource insights are mentioned
        assert "Resource efficiency insights" in backstory
        
        # Check guidance for survival application
        assert "apply these hard-earned lessons" in backstory
        assert "maximize survival chances" in backstory
        
    @patch('src.escape_room_sim.agents.survivor.Agent')
    def test_resource_insights_handling_different_formats(self, mock_agent_class):
        """Test handling of different resource insight formats.""" 
        # Arrange
        mock_agent_instance = Mock()
        mock_agent_class.return_value = mock_agent_instance
        
        context = {
            "survival_lessons": ["Some lesson"],
            "resource_insights": "Still learning optimal resource usage"  # String instead of dict
        }
        
        # Act
        agent = create_survivor_agent(iteration_context=context)
        
        # Assert
        call_args = mock_agent_class.call_args
        backstory = call_args[1]['backstory']
        
        # Should handle string resource insights gracefully
        assert "Resource efficiency insights: Still learning optimal resource usage" in backstory
        
    @patch('src.escape_room_sim.agents.survivor.Agent')
    def test_survival_focus_without_resource_insights(self, mock_agent_class):
        """Test survival focus when resource_insights is missing."""
        # Arrange
        mock_agent_instance = Mock()
        mock_agent_class.return_value = mock_agent_instance
        
        context = {
            "survival_lessons": ["Lesson without resource data"]
        }
        
        # Act
        agent = create_survivor_agent(iteration_context=context)
        
        # Assert
        call_args = mock_agent_class.call_args
        backstory = call_args[1]['backstory']
        
        # Should use default resource insight message
        assert "Resource efficiency insights: Still learning optimal resource usage" in backstory
        
    @patch('src.escape_room_sim.agents.survivor.Agent')
    def test_system_message_action_focus(self, mock_agent_class):
        """Test that system message emphasizes action and execution."""
        # Arrange
        mock_agent_instance = Mock()
        mock_agent_class.return_value = mock_agent_instance
        
        # Act
        agent = create_survivor_agent()
        
        # Assert
        call_args = mock_agent_class.call_args
        system_message = call_args[1]['system_message']
        
        action_focus_indicators = [
            "actionable solutions over extended discussion",
            "what can realistically be accomplished",
            "available resources and time constraints",
            "quickly and decisively"
        ]
        
        for indicator in action_focus_indicators:
            assert indicator in system_message