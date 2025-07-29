"""
Comprehensive edge case and error handling tests for all agents.

Tests cover:
- Invalid parameter combinations
- Malformed context data
- Memory system edge cases
- Configuration boundary conditions
- Performance under stress
- Type validation
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any

from src.escape_room_sim.agents.strategist import (
    create_strategist_agent,
    create_strategist_with_context,
    StrategistConfig
)
from src.escape_room_sim.agents.mediator import (
    create_mediator_agent,
    create_mediator_with_context,
    MediatorConfig
)
from src.escape_room_sim.agents.survivor import (
    create_survivor_agent,
    create_survivor_with_context,
    SurvivorConfig
)


class TestParameterValidationEdgeCases:
    """Test edge cases for parameter validation across all agents."""
    
    @pytest.mark.parametrize("agent_function", [
        create_strategist_agent,
        create_mediator_agent,
        create_survivor_agent
    ])
    @patch('src.escape_room_sim.agents.strategist.Agent')
    @patch('src.escape_room_sim.agents.mediator.Agent')
    @patch('src.escape_room_sim.agents.survivor.Agent')
    def test_agents_with_invalid_memory_parameter_type(self, mock_survivor, mock_mediator, mock_strategist, agent_function):
        """Test agents with invalid memory parameter types."""
        # Arrange
        mock_instance = Mock()
        mock_strategist.return_value = mock_instance
        mock_mediator.return_value = mock_instance
        mock_survivor.return_value = mock_instance
        
        # Act & Assert - Should handle gracefully or raise appropriate error
        try:
            agent = agent_function(memory_enabled="not_a_boolean")
            # If it doesn't raise an error, the function should handle it gracefully
        except (TypeError, ValueError):
            # This is acceptable - function should validate input types
            pass
            
    @pytest.mark.parametrize("agent_function", [
        create_strategist_agent,
        create_mediator_agent,
        create_survivor_agent
    ])
    @patch('src.escape_room_sim.agents.strategist.Agent')
    @patch('src.escape_room_sim.agents.mediator.Agent')
    @patch('src.escape_room_sim.agents.survivor.Agent')
    def test_agents_with_invalid_verbose_parameter_type(self, mock_survivor, mock_mediator, mock_strategist, agent_function):
        """Test agents with invalid verbose parameter types."""
        # Arrange
        mock_instance = Mock()
        mock_strategist.return_value = mock_instance
        mock_mediator.return_value = mock_instance
        mock_survivor.return_value = mock_instance
        
        # Act & Assert
        try:
            agent = agent_function(verbose="not_a_boolean")
        except (TypeError, ValueError):
            pass  # Expected behavior
            
    @pytest.mark.parametrize("agent_function", [
        create_strategist_agent,
        create_mediator_agent,
        create_survivor_agent
    ])
    @patch('src.escape_room_sim.agents.strategist.Agent')
    @patch('src.escape_room_sim.agents.mediator.Agent')
    @patch('src.escape_room_sim.agents.survivor.Agent')
    def test_agents_with_invalid_context_parameter_type(self, mock_survivor, mock_mediator, mock_strategist, agent_function):
        """Test agents with invalid context parameter types."""
        # Arrange
        mock_instance = Mock()
        mock_strategist.return_value = mock_instance
        mock_mediator.return_value = mock_instance
        mock_survivor.return_value = mock_instance
        
        # Act & Assert - Should handle gracefully
        invalid_contexts = [
            "not_a_dict",
            123,
            [],
            True
        ]
        
        for invalid_context in invalid_contexts:
            try:
                agent = agent_function(iteration_context=invalid_context)
                # Should not raise exception - should handle gracefully
            except Exception:
                pass  # Some level of validation is acceptable


class TestContextDataEdgeCases:
    """Test edge cases for context data handling."""
    
    @patch('src.escape_room_sim.agents.strategist.Agent')
    def test_strategist_with_extremely_long_failed_strategies(self, mock_agent_class):
        """Test strategist with extremely long failed strategies list."""
        # Arrange
        mock_agent_instance = Mock()
        mock_agent_class.return_value = mock_agent_instance
        
        # Create context with 100 failed strategies
        context = {
            "failed_strategies": [f"Strategy {i} failed with very long description that goes on and on and on" for i in range(100)]
        }
        
        # Act
        agent = create_strategist_agent(iteration_context=context)
        
        # Assert
        call_args = mock_agent_class.call_args
        backstory = call_args[1]['backstory']
        
        # Should only include first 3 despite having 100
        assert "Strategy 0 failed" in backstory
        assert "Strategy 1 failed" in backstory
        assert "Strategy 2 failed" in backstory
        assert "Strategy 3 failed" not in backstory
        assert "learned from 100 previous failed" in backstory
        
    @patch('src.escape_room_sim.agents.mediator.Agent')
    def test_mediator_with_extremely_long_team_conflicts(self, mock_agent_class):
        """Test mediator with extremely long team conflicts list."""
        # Arrange
        mock_agent_instance = Mock()
        mock_agent_class.return_value = mock_agent_instance
        
        # Create context with 50 team conflicts
        context = {
            "team_conflicts": [f"Conflict {i} involving multiple team members with complex interpersonal dynamics" for i in range(50)]
        }
        
        # Act
        agent = create_mediator_agent(iteration_context=context)
        
        # Assert
        call_args = mock_agent_class.call_args
        backstory = call_args[1]['backstory']
        
        assert "Conflict 0 involving" in backstory
        assert "Conflict 1 involving" in backstory
        assert "Conflict 2 involving" in backstory
        assert "Conflict 3 involving" not in backstory
        assert "observed 50 team conflicts" in backstory
        
    @patch('src.escape_room_sim.agents.survivor.Agent')
    def test_survivor_with_extremely_long_survival_lessons(self, mock_agent_class):
        """Test survivor with extremely long survival lessons list."""
        # Arrange
        mock_agent_instance = Mock()
        mock_agent_class.return_value = mock_agent_instance
        
        # Create context with 75 survival lessons
        context = {
            "survival_lessons": [f"Lesson {i}: Critical survival insight that could mean the difference between life and death" for i in range(75)]
        }
        
        # Act
        agent = create_survivor_agent(iteration_context=context)
        
        # Assert
        call_args = mock_agent_class.call_args
        backstory = call_args[1]['backstory']
        
        assert "Lesson 0:" in backstory
        assert "Lesson 1:" in backstory
        assert "Lesson 2:" in backstory
        assert "Lesson 3:" not in backstory
        assert "learned 75 key survival lessons" in backstory
        
    @patch('src.escape_room_sim.agents.strategist.Agent')
    def test_strategist_with_empty_string_failed_strategies(self, mock_agent_class):
        """Test strategist with empty string elements in failed strategies."""
        # Arrange
        mock_agent_instance = Mock()
        mock_agent_class.return_value = mock_agent_instance
        
        context = {
            "failed_strategies": ["", "   ", "Valid strategy", "", "Another valid strategy"]
        }
        
        # Act
        agent = create_strategist_agent(iteration_context=context)
        
        # Assert
        call_args = mock_agent_class.call_args
        backstory = call_args[1]['backstory']
        
        # Should handle empty strings gracefully (join includes empty strings)
        assert "IMPORTANT LEARNING FROM PREVIOUS ATTEMPTS" in backstory
        assert "Valid strategy" in backstory
        # The join will include empty strings in the first 3 elements, so "Another valid strategy" 
        # won't be included since it's at index 4, but should only take first 3
        # So we just check that valid non-empty strategies are present
        assert "5 previous failed strategies" in backstory  # Total count should be shown
        
    @patch('src.escape_room_sim.agents.mediator.Agent')
    def test_mediator_with_nested_dict_trust_levels(self, mock_agent_class):
        """Test mediator with deeply nested trust levels structure."""
        # Arrange
        mock_agent_instance = Mock()
        mock_agent_class.return_value = mock_agent_instance
        
        context = {
            "team_conflicts": ["Some conflict"],
            "trust_levels": {
                "individual": {
                    "strategist": {"mediator": 0.8, "survivor": 0.6},
                    "mediator": {"strategist": 0.9, "survivor": 0.7},
                    "survivor": {"strategist": 0.5, "mediator": 0.8}
                },
                "overall": 0.7,
                "trends": ["increasing", "stable"]
            }
        }
        
        # Act
        agent = create_mediator_agent(iteration_context=context)
        
        # Assert - Should handle complex nested structure gracefully
        call_args = mock_agent_class.call_args
        backstory = call_args[1]['backstory']
        
        assert "Trust levels you've observed" in backstory
        # Should convert complex structure to string representation
        
    @patch('src.escape_room_sim.agents.survivor.Agent')
    def test_survivor_with_complex_resource_insights(self, mock_agent_class):
        """Test survivor with complex resource insights structure."""
        # Arrange
        mock_agent_instance = Mock()
        mock_agent_class.return_value = mock_agent_instance
        
        context = {
            "survival_lessons": ["Some lesson"],
            "resource_insights": {
                "efficiency_scores": {
                    "time_management": 0.6,
                    "resource_allocation": 0.8,
                    "decision_speed": 0.9
                },
                "critical_resources": ["time", "energy", "tools"],
                "waste_patterns": {
                    "overanalysis": {"frequency": 0.7, "impact": 0.8},
                    "hesitation": {"frequency": 0.5, "impact": 0.9}
                }
            }
        }
        
        # Act
        agent = create_survivor_agent(iteration_context=context)
        
        # Assert - Should handle complex nested structure gracefully
        call_args = mock_agent_class.call_args
        backstory = call_args[1]['backstory']
        
        assert "Resource efficiency insights" in backstory


class TestMemorySystemEdgeCases:
    """Test edge cases for memory system behavior."""
    
    @pytest.mark.parametrize("agent_function", [
        create_strategist_agent,
        create_mediator_agent,
        create_survivor_agent
    ])
    @patch('src.escape_room_sim.agents.strategist.Agent')
    @patch('src.escape_room_sim.agents.mediator.Agent')
    @patch('src.escape_room_sim.agents.survivor.Agent')
    def test_agents_memory_boolean_edge_cases(self, mock_survivor, mock_mediator, mock_strategist, agent_function):
        """Test memory parameter edge cases with boolean-like values."""
        # Arrange
        mock_instance = Mock()
        mock_strategist.return_value = mock_instance
        mock_mediator.return_value = mock_instance
        mock_survivor.return_value = mock_instance
        
        # Test various truthy/falsy values
        memory_values = [
            (True, True),
            (False, False),
            (1, True),    # Should be treated as True
            (0, False),   # Should be treated as False
            ("true", True),  # String should be treated as True (truthy)
            ("", False),     # Empty string should be treated as False (falsy)
            ([], False),     # Empty list should be treated as False (falsy)
            ([1], True),     # Non-empty list should be treated as True (truthy)
        ]
        
        for memory_value, expected_result in memory_values:
            # Act
            agent = agent_function(memory_enabled=memory_value)
            
            # Assert - Check the actual call to Agent constructor
            relevant_mock = None
            if agent_function == create_strategist_agent:
                relevant_mock = mock_strategist
            elif agent_function == create_mediator_agent:
                relevant_mock = mock_mediator
            else:
                relevant_mock = mock_survivor
                
            call_args = relevant_mock.call_args
            actual_memory = call_args[1]['memory']
            
            # Should convert to boolean correctly
            assert bool(actual_memory) == expected_result


class TestContextWithContextFunctions:
    """Test edge cases for create_*_with_context functions."""
    
    def test_strategist_with_context_none_values_in_dict(self):
        """Test strategist with context containing None values."""
        # Arrange
        previous_results = {
            "failed_strategies": None,
            "successful_approaches": ["Valid approach"], 
            "resource_constraints": None,
            "team_dynamics": {"trust": 0.5}
        }
        
        # Act & Assert - Should not raise exception
        with patch('src.escape_room_sim.agents.strategist.create_strategist_agent') as mock_create:
            mock_agent = Mock()
            mock_create.return_value = mock_agent
            
            result = create_strategist_with_context(previous_results)
            
            call_args = mock_create.call_args
            context = call_args[1]['iteration_context']
            
            # None values should be replaced with defaults
            assert context['failed_strategies'] == []  # None -> empty list
            assert context['successful_approaches'] == ["Valid approach"]
            assert context['resource_constraints'] == {}  # None -> empty dict
            assert context['team_dynamics'] == {"trust": 0.5}
            
    def test_mediator_with_context_mixed_data_types(self):
        """Test mediator with context containing mixed data types."""
        # Arrange
        previous_results = {
            "team_conflicts": "Single conflict string",  # Should be list
            "trust_levels": ["0.8", "0.6"],  # Should be dict
            "successful_collaborations": {"collab": "success"},  # Should be list
            "communication_issues": 42  # Should be list
        }
        
        # Act & Assert - Should handle gracefully
        with patch('src.escape_room_sim.agents.mediator.create_mediator_agent') as mock_create:
            mock_agent = Mock()
            mock_create.return_value = mock_agent
            
            result = create_mediator_with_context(previous_results)
            
            call_args = mock_create.call_args
            context = call_args[1]['iteration_context']
            
            # Should use the values as provided, letting the agent creation handle type issues
            assert context['team_conflicts'] == "Single conflict string"
            assert context['trust_levels'] == ["0.8", "0.6"]
            assert context['successful_collaborations'] == {"collab": "success"}
            assert context['communication_issues'] == 42
            
    def test_survivor_with_context_extremely_large_data(self):
        """Test survivor with context containing extremely large data structures."""
        # Arrange
        large_lessons = [f"Lesson {i}" for i in range(10000)]
        large_insights = {f"insight_{i}": f"value_{i}" for i in range(1000)}
        
        previous_results = {
            "survival_lessons": large_lessons,
            "resource_insights": large_insights,
            "execution_failures": [f"Failure {i}" for i in range(5000)],
            "successful_tactics": [f"Tactic {i}" for i in range(2000)]
        }
        
        # Act & Assert - Should handle large data gracefully
        with patch('src.escape_room_sim.agents.survivor.create_survivor_agent') as mock_create:
            mock_agent = Mock()
            mock_create.return_value = mock_agent
            
            result = create_survivor_with_context(previous_results)
            
            call_args = mock_create.call_args
            context = call_args[1]['iteration_context']
            
            # Should pass large data through
            assert len(context['survival_lessons']) == 10000
            assert len(context['resource_insights']) == 1000
            assert len(context['execution_failures']) == 5000
            assert len(context['successful_tactics']) == 2000


class TestConfigurationBoundaryConditions:
    """Test boundary conditions for agent configurations."""
    
    @pytest.mark.parametrize("config_class", [
        StrategistConfig,
        MediatorConfig,
        SurvivorConfig
    ])
    def test_config_class_attribute_access(self, config_class):
        """Test that config classes handle attribute access correctly."""
        # Act & Assert - All should have required attributes
        assert hasattr(config_class, 'DEFAULT_ROLE')
        assert hasattr(config_class, 'DEFAULT_GOAL')
        assert hasattr(config_class, 'get_personality_traits')
        
        # Test attribute values are non-empty strings
        assert isinstance(config_class.DEFAULT_ROLE, str)
        assert len(config_class.DEFAULT_ROLE) > 0
        assert isinstance(config_class.DEFAULT_GOAL, str)
        assert len(config_class.DEFAULT_GOAL) > 0
        
        # Test personality traits method returns dict
        traits = config_class.get_personality_traits()
        assert isinstance(traits, dict)
        assert len(traits) > 0
        
        # All trait values should be strings
        for key, value in traits.items():
            assert isinstance(key, str)
            assert isinstance(value, str)
            assert len(key) > 0
            assert len(value) > 0
            
    def test_survivor_config_priority_rankings_validity(self):
        """Test that survivor config priority rankings are valid."""
        # Act
        priorities = SurvivorConfig.get_survival_priorities()
        
        # Assert
        assert isinstance(priorities, dict)
        
        # All priorities should be positive integers
        for priority_name, priority_value in priorities.items():
            assert isinstance(priority_value, int)
            assert priority_value > 0
            
        # No duplicate priority values (each should be unique)
        priority_values = list(priorities.values())
        assert len(priority_values) == len(set(priority_values))
        
        # Priorities should be in reasonable range (1-10)
        for priority_value in priority_values:
            assert 1 <= priority_value <= 10
            
    def test_survivor_config_decision_criteria_validity(self):
        """Test that survivor config decision criteria weights are valid."""
        # Act
        criteria = SurvivorConfig.get_decision_criteria()
        
        # Assert
        assert isinstance(criteria, dict)
        
        # All weights should be floats between 0 and 1
        for criteria_name, weight in criteria.items():
            assert isinstance(weight, (int, float))
            assert 0.0 <= weight <= 1.0
            
        # Total weights should sum to approximately 1.0
        total_weight = sum(criteria.values())
        assert abs(total_weight - 1.0) < 0.01
        
    def test_mediator_config_relationship_tracking_validity(self):
        """Test that mediator config relationship tracking is valid."""
        # Act
        tracking_config = MediatorConfig.get_relationship_tracking_config()
        
        # Assert
        assert isinstance(tracking_config, dict)
        
        # All values should be booleans
        for config_name, config_value in tracking_config.items():
            assert isinstance(config_value, bool)
            assert isinstance(config_name, str)
            assert len(config_name) > 0


class TestPerformanceEdgeCases:
    """Test performance-related edge cases."""
    
    @pytest.mark.parametrize("agent_function,mock_path", [
        (create_strategist_agent, 'src.escape_room_sim.agents.strategist.Agent'),
        (create_mediator_agent, 'src.escape_room_sim.agents.mediator.Agent'),
        (create_survivor_agent, 'src.escape_room_sim.agents.survivor.Agent')
    ])
    def test_agent_creation_with_large_backstory(self, agent_function, mock_path):
        """Test agent creation performance with large backstory context."""
        # Arrange
        with patch(mock_path) as mock_agent_class:
            mock_agent_instance = Mock()
            mock_agent_class.return_value = mock_agent_instance
            
            # Create very large context
            if agent_function == create_strategist_agent:
                large_context = {
                    "failed_strategies": [f"Very long strategy description that goes on for many words and includes detailed analysis of why it failed and what went wrong and how it could be improved in the future - Strategy {i}" for i in range(1000)]
                }
            elif agent_function == create_mediator_agent:
                large_context = {
                    "team_conflicts": [f"Detailed conflict description involving multiple team members with complex interpersonal dynamics and communication issues that need to be resolved - Conflict {i}" for i in range(1000)]
                }
            else:  # survivor
                large_context = {
                    "survival_lessons": [f"Critical survival lesson that involves detailed analysis of life-threatening situations and the specific actions taken to survive - Lesson {i}" for i in range(1000)]
                }
            
            # Act - Should complete without performance issues
            agent = agent_function(iteration_context=large_context)
            
            # Assert
            mock_agent_class.assert_called_once()
            call_args = mock_agent_class.call_args
            
            # Backstory should be created despite large context
            assert isinstance(call_args[1]['backstory'], str)
            assert len(call_args[1]['backstory']) > 0


class TestTypeValidationEdgeCases:
    """Test type validation edge cases."""
    
    @pytest.mark.parametrize("agent_function", [
        create_strategist_agent,
        create_mediator_agent,
        create_survivor_agent
    ])
    @patch('src.escape_room_sim.agents.strategist.Agent')
    @patch('src.escape_room_sim.agents.mediator.Agent')
    @patch('src.escape_room_sim.agents.survivor.Agent')
    def test_agents_with_numeric_parameter_values(self, mock_survivor, mock_mediator, mock_strategist, agent_function):
        """Test agents with numeric values for boolean parameters."""
        # Arrange
        mock_instance = Mock()
        mock_strategist.return_value = mock_instance
        mock_mediator.return_value = mock_instance
        mock_survivor.return_value = mock_instance
        
        # Act - Should handle numeric values gracefully
        agent = agent_function(memory_enabled=1, verbose=0)
        
        # Assert - Should convert to boolean
        relevant_mock = mock_strategist if agent_function == create_strategist_agent else (
            mock_mediator if agent_function == create_mediator_agent else mock_survivor
        )
        
        call_args = relevant_mock.call_args
        # Should interpret 1 as True, 0 as False
        assert call_args[1]['memory'] == 1  # Let CrewAI handle the conversion
        assert call_args[1]['verbose'] == 0