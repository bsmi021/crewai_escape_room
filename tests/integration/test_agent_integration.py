"""
Integration tests for CrewAI agent interaction with framework.

Tests cover:
- Agent interaction with CrewAI framework
- Mock LLM responses for consistent testing
- Agent collaboration scenarios
- Memory persistence across interactions
- Error handling in real-world scenarios
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any

from src.escape_room_sim.agents.strategist import create_strategist_agent
from src.escape_room_sim.agents.mediator import create_mediator_agent
from src.escape_room_sim.agents.survivor import create_survivor_agent


class TestAgentFrameworkIntegration:
    """Test suite for agent integration with CrewAI framework."""
    
    @patch('src.escape_room_sim.agents.strategist.Agent')
    def test_strategist_agent_framework_initialization(self, mock_agent_class):
        """Test that strategist agent integrates properly with CrewAI framework."""
        # Arrange
        mock_agent_instance = Mock()
        mock_agent_instance.role = "Strategic Analyst"
        mock_agent_instance.goal = "Find the optimal solution through iterative problem-solving and learning from failures"
        mock_agent_instance.memory = True
        mock_agent_class.return_value = mock_agent_instance
        
        # Act
        agent = create_strategist_agent()
        
        # Assert
        mock_agent_class.assert_called_once()
        assert agent == mock_agent_instance
        
        # Verify agent can be used by framework
        assert hasattr(agent, 'role')
        assert hasattr(agent, 'goal')
        assert hasattr(agent, 'memory')
        
    @patch('src.escape_room_sim.agents.mediator.Agent')
    def test_mediator_agent_framework_initialization(self, mock_agent_class):
        """Test that mediator agent integrates properly with CrewAI framework."""
        # Arrange
        mock_agent_instance = Mock()
        mock_agent_instance.role = "Group Facilitator"
        mock_agent_instance.goal = "Build consensus through multiple discussion rounds and maintain team cohesion"
        mock_agent_instance.memory = True
        mock_agent_class.return_value = mock_agent_instance
        
        # Act
        agent = create_mediator_agent()
        
        # Assert
        mock_agent_class.assert_called_once()
        assert agent == mock_agent_instance
        
        # Verify agent can be used by framework
        assert hasattr(agent, 'role')
        assert hasattr(agent, 'goal')
        assert hasattr(agent, 'memory')
        
    @patch('src.escape_room_sim.agents.survivor.Agent')
    def test_survivor_agent_framework_initialization(self, mock_agent_class):
        """Test that survivor agent integrates properly with CrewAI framework."""
        # Arrange
        mock_agent_instance = Mock()
        mock_agent_instance.role = "Survival Specialist"
        mock_agent_instance.goal = "Ensure survival through adaptive decision-making and efficient execution"
        mock_agent_instance.memory = True
        mock_agent_class.return_value = mock_agent_instance
        
        # Act
        agent = create_survivor_agent()
        
        # Assert
        mock_agent_class.assert_called_once()
        assert agent == mock_agent_instance
        
        # Verify agent can be used by framework
        assert hasattr(agent, 'role')
        assert hasattr(agent, 'goal')
        assert hasattr(agent, 'memory')


class TestAgentCollaborationScenarios:
    """Test suite for multi-agent collaboration scenarios."""
    
    @patch('src.escape_room_sim.agents.strategist.Agent')
    @patch('src.escape_room_sim.agents.mediator.Agent')
    @patch('src.escape_room_sim.agents.survivor.Agent')
    def test_three_agent_collaboration_setup(self, mock_survivor_class, mock_mediator_class, mock_strategist_class):
        """Test setting up collaboration between all three agents."""
        # Arrange
        mock_strategist = Mock()
        mock_strategist.role = "Strategic Analyst"
        mock_strategist_class.return_value = mock_strategist
        
        mock_mediator = Mock()
        mock_mediator.role = "Group Facilitator"
        mock_mediator_class.return_value = mock_mediator
        
        mock_survivor = Mock()
        mock_survivor.role = "Survival Specialist"
        mock_survivor_class.return_value = mock_survivor
        
        # Act
        strategist = create_strategist_agent()
        mediator = create_mediator_agent()
        survivor = create_survivor_agent()
        
        # Assert
        assert strategist.role == "Strategic Analyst"
        assert mediator.role == "Group Facilitator"
        assert survivor.role == "Survival Specialist"
        
        # Verify all agents have compatible configuration
        agents = [strategist, mediator, survivor]
        for agent in agents:
            assert hasattr(agent, 'role')
            assert hasattr(agent, 'role')
            assert isinstance(agent.role, str)
            assert len(agent.role) > 0
            
    @patch('src.escape_room_sim.agents.strategist.Agent')
    @patch('src.escape_room_sim.agents.mediator.Agent')
    @patch('src.escape_room_sim.agents.survivor.Agent')
    def test_agents_with_shared_context(self, mock_survivor_class, mock_mediator_class, mock_strategist_class):
        """Test agents created with shared context for collaboration."""
        # Arrange
        mock_strategist = Mock()
        mock_strategist_class.return_value = mock_strategist
        
        mock_mediator = Mock()
        mock_mediator_class.return_value = mock_mediator
        
        mock_survivor = Mock()
        mock_survivor_class.return_value = mock_survivor
        
        shared_context = {
            "failed_strategies": ["Strategy A failed"],
            "team_conflicts": ["Resource disagreement"],
            "survival_lessons": ["Time is critical"],
            "resource_constraints": {"time": 30},
            "team_dynamics": {"trust": 0.7}
        }
        
        strategist_context = {
            "failed_strategies": shared_context["failed_strategies"],
            "resource_constraints": shared_context["resource_constraints"],
            "team_dynamics": shared_context["team_dynamics"]
        }
        
        mediator_context = {
            "team_conflicts": shared_context["team_conflicts"],
            "team_dynamics": shared_context["team_dynamics"]
        }
        
        survivor_context = {
            "survival_lessons": shared_context["survival_lessons"],
            "resource_constraints": shared_context["resource_constraints"]
        }
        
        # Act
        strategist = create_strategist_agent(iteration_context=strategist_context)
        mediator = create_mediator_agent(iteration_context=mediator_context)
        survivor = create_survivor_agent(iteration_context=survivor_context)
        
        # Assert
        # Verify all agents were created with context
        mock_strategist_class.assert_called_once()
        mock_mediator_class.assert_called_once()
        mock_survivor_class.assert_called_once()
        
        # Verify context was passed properly
        strategist_call_args = mock_strategist_class.call_args
        mediator_call_args = mock_mediator_class.call_args
        survivor_call_args = mock_survivor_class.call_args
        
        # Check that backstories contain context-specific information
        assert "Strategy A failed" in strategist_call_args[1]['backstory']
        assert "Resource disagreement" in mediator_call_args[1]['backstory']
        assert "Time is critical" in survivor_call_args[1]['backstory']


class TestMockLLMResponseHandling:
    """Test suite for handling mocked LLM responses."""
    
    @patch('src.escape_room_sim.agents.strategist.Agent')
    def test_strategist_with_mock_llm_response(self, mock_agent_class, mock_llm_response):
        """Test strategist agent behavior with mock LLM response."""
        # Arrange
        mock_agent_instance = Mock()
        mock_agent_instance.execute.return_value = mock_llm_response
        mock_agent_class.return_value = mock_agent_instance
        
        # Act
        agent = create_strategist_agent()
        
        # Simulate agent execution with mock response
        if hasattr(agent, 'execute'):
            response = agent.execute("Test task")
            
            # Assert
            assert response == mock_llm_response
            assert "mock response" in response["content"].lower()
            
    @patch('src.escape_room_sim.agents.mediator.Agent')
    def test_mediator_with_mock_llm_response(self, mock_agent_class, mock_llm_response):
        """Test mediator agent behavior with mock LLM response."""
        # Arrange
        mock_agent_instance = Mock()
        mock_agent_instance.execute.return_value = mock_llm_response
        mock_agent_class.return_value = mock_agent_instance
        
        # Act
        agent = create_mediator_agent()
        
        # Simulate agent execution with mock response
        if hasattr(agent, 'execute'):
            response = agent.execute("Test facilitation task")
            
            # Assert
            assert response == mock_llm_response
            assert response["metadata"]["model"] == "mock-model"
            
    @patch('src.escape_room_sim.agents.survivor.Agent')
    def test_survivor_with_mock_llm_response(self, mock_agent_class, mock_llm_response):
        """Test survivor agent behavior with mock LLM response."""
        # Arrange
        mock_agent_instance = Mock()
        mock_agent_instance.execute.return_value = mock_llm_response
        mock_agent_class.return_value = mock_agent_instance
        
        # Act
        agent = create_survivor_agent()
        
        # Simulate agent execution with mock response
        if hasattr(agent, 'execute'):
            response = agent.execute("Test survival task")
            
            # Assert
            assert response == mock_llm_response
            assert response["metadata"]["tokens_used"] == 100


class TestMemoryPersistenceIntegration:
    """Test suite for memory persistence across agent interactions."""
    
    @patch('src.escape_room_sim.agents.strategist.Agent')
    def test_strategist_memory_enabled_integration(self, mock_agent_class):
        """Test strategist agent with memory enabled integration."""
        # Arrange
        mock_agent_instance = Mock()
        mock_agent_instance.memory = True
        mock_agent_instance.remember = Mock()
        mock_agent_instance.recall = Mock(return_value=["Previous memory 1", "Previous memory 2"])
        mock_agent_class.return_value = mock_agent_instance
        
        # Act
        agent = create_strategist_agent(memory_enabled=True)
        
        # Assert
        assert agent.memory is True
        
        # Simulate memory operations
        if hasattr(agent, 'remember'):
            agent.remember("New strategic insight")
            agent.remember.assert_called_with("New strategic insight")
            
        if hasattr(agent, 'recall'):
            memories = agent.recall()
            assert len(memories) == 2
            assert "Previous memory 1" in memories
            
    @patch('src.escape_room_sim.agents.mediator.Agent')
    def test_mediator_memory_disabled_integration(self, mock_agent_class):
        """Test mediator agent with memory disabled integration."""
        # Arrange
        mock_agent_instance = Mock()
        mock_agent_instance.memory = False
        mock_agent_class.return_value = mock_agent_instance
        
        # Act
        agent = create_mediator_agent(memory_enabled=False)
        
        # Assert
        assert agent.memory is False
        
        # Verify memory operations are not available/configured
        call_args = mock_agent_class.call_args
        assert call_args[1]['memory'] is False
        
    @patch('src.escape_room_sim.agents.survivor.Agent')
    def test_survivor_memory_context_integration(self, mock_agent_class, survivor_context):
        """Test survivor agent memory integration with context."""
        # Arrange
        mock_agent_instance = Mock()
        mock_agent_instance.memory = True
        mock_agent_instance.load_context = Mock()
        mock_agent_class.return_value = mock_agent_instance
        
        # Act
        agent = create_survivor_agent(
            memory_enabled=True,
            iteration_context=survivor_context
        )
        
        # Assert
        assert agent.memory is True
        
        # Verify context was integrated into backstory for memory
        call_args = mock_agent_class.call_args
        backstory = call_args[1]['backstory']
        assert "CRITICAL SURVIVAL LESSONS" in backstory
        
        # Simulate context loading
        if hasattr(agent, 'load_context'):
            agent.load_context(survivor_context)
            agent.load_context.assert_called_with(survivor_context)


class TestErrorHandlingIntegration:
    """Test suite for error handling in integration scenarios."""
    
    @patch('src.escape_room_sim.agents.strategist.Agent')
    def test_strategist_agent_creation_error_handling(self, mock_agent_class):
        """Test error handling during strategist agent creation."""
        # Arrange
        mock_agent_class.side_effect = Exception("Framework initialization failed")
        
        # Act & Assert
        with pytest.raises(Exception, match="Framework initialization failed"):
            create_strategist_agent()
            
    @patch('src.escape_room_sim.agents.mediator.Agent')
    def test_mediator_agent_execution_error_handling(self, mock_agent_class):
        """Test error handling during mediator agent execution."""
        # Arrange
        mock_agent_instance = Mock()
        mock_agent_instance.execute.side_effect = Exception("LLM communication failed")
        mock_agent_class.return_value = mock_agent_instance
        
        # Act
        agent = create_mediator_agent()
        
        # Assert
        if hasattr(agent, 'execute'):
            with pytest.raises(Exception, match="LLM communication failed"):
                agent.execute("Test task")
                
    @patch('src.escape_room_sim.agents.survivor.Agent')
    def test_survivor_agent_memory_error_handling(self, mock_agent_class):
        """Test error handling during survivor agent memory operations."""
        # Arrange
        mock_agent_instance = Mock()
        mock_agent_instance.memory = True
        mock_agent_instance.remember.side_effect = Exception("Memory storage failed")
        mock_agent_class.return_value = mock_agent_instance
        
        # Act
        agent = create_survivor_agent(memory_enabled=True)
        
        # Assert
        if hasattr(agent, 'remember'):
            with pytest.raises(Exception, match="Memory storage failed"):
                agent.remember("Test memory")


class TestAgentConfigurationConsistency:
    """Test suite for agent configuration consistency across framework integration."""
    
    @patch('src.escape_room_sim.agents.strategist.Agent')
    @patch('src.escape_room_sim.agents.mediator.Agent')
    @patch('src.escape_room_sim.agents.survivor.Agent')
    def test_all_agents_consistent_configuration(self, mock_survivor_class, mock_mediator_class, mock_strategist_class):
        """Test that all agents have consistent configuration for framework integration."""
        # Arrange
        mock_strategist = Mock()
        mock_strategist_class.return_value = mock_strategist
        
        mock_mediator = Mock()
        mock_mediator_class.return_value = mock_mediator
        
        mock_survivor = Mock()
        mock_survivor_class.return_value = mock_survivor
        
        # Act
        strategist = create_strategist_agent()
        mediator = create_mediator_agent()
        survivor = create_survivor_agent()
        
        # Assert - Verify consistent configuration across all agents
        agents_and_mocks = [
            (mock_strategist_class, "Strategic Analyst"),
            (mock_mediator_class, "Group Facilitator"),
            (mock_survivor_class, "Survival Specialist")
        ]
        
        for mock_class, expected_role in agents_and_mocks:
            call_args = mock_class.call_args
            
            # All should have consistent basic configuration
            assert call_args[1]['verbose'] is True
            assert call_args[1]['memory'] is True
            assert call_args[1]['max_iter'] == 3
            assert call_args[1]['allow_delegation'] is False
            assert call_args[1]['role'] == expected_role
            
            # All should have non-empty goal and backstory
            assert isinstance(call_args[1]['goal'], str)
            assert len(call_args[1]['goal']) > 0
            assert isinstance(call_args[1]['backstory'], str)
            assert len(call_args[1]['backstory']) > 0
            assert isinstance(call_args[1]['system_message'], str)
            assert len(call_args[1]['system_message']) > 0
            
    @patch('src.escape_room_sim.agents.strategist.Agent')
    @patch('src.escape_room_sim.agents.mediator.Agent')
    @patch('src.escape_room_sim.agents.survivor.Agent')
    def test_agents_with_memory_disabled_consistency(self, mock_survivor_class, mock_mediator_class, mock_strategist_class):
        """Test consistent behavior when memory is disabled across all agents."""
        # Arrange
        mock_strategist = Mock()
        mock_strategist_class.return_value = mock_strategist
        
        mock_mediator = Mock()
        mock_mediator_class.return_value = mock_mediator
        
        mock_survivor = Mock()
        mock_survivor_class.return_value = mock_survivor
        
        # Act
        strategist = create_strategist_agent(memory_enabled=False)
        mediator = create_mediator_agent(memory_enabled=False)
        survivor = create_survivor_agent(memory_enabled=False)
        
        # Assert
        for mock_class in [mock_strategist_class, mock_mediator_class, mock_survivor_class]:
            call_args = mock_class.call_args
            assert call_args[1]['memory'] is False
            
    @patch('src.escape_room_sim.agents.strategist.Agent')
    @patch('src.escape_room_sim.agents.mediator.Agent')
    @patch('src.escape_room_sim.agents.survivor.Agent')
    def test_agents_with_verbose_disabled_consistency(self, mock_survivor_class, mock_mediator_class, mock_strategist_class):
        """Test consistent behavior when verbose is disabled across all agents."""
        # Arrange
        mock_strategist = Mock()
        mock_strategist_class.return_value = mock_strategist
        
        mock_mediator = Mock()
        mock_mediator_class.return_value = mock_mediator
        
        mock_survivor = Mock()
        mock_survivor_class.return_value = mock_survivor
        
        # Act
        strategist = create_strategist_agent(verbose=False)
        mediator = create_mediator_agent(verbose=False)
        survivor = create_survivor_agent(verbose=False)
        
        # Assert
        for mock_class in [mock_strategist_class, mock_mediator_class, mock_survivor_class]:
            call_args = mock_class.call_args
            assert call_args[1]['verbose'] is False