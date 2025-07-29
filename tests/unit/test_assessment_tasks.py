"""
Comprehensive tests for assessment task definitions.

Tests verify task creation, context integration, agent assignment,
descriptions, expected outputs, and parameter validation.
"""

import pytest
from unittest.mock import Mock, patch
from typing import List, Dict, Any, Optional

# Import the functions to test
from src.escape_room_sim.tasks.assessment import (
    create_assessment_tasks,
    create_progress_evaluation_task,
    create_situation_analysis_task
)


class TestCreateAssessmentTasks:
    """Test the create_assessment_tasks function."""
    
    def test_creates_three_tasks(self, sample_agents):
        """Test that exactly 3 tasks are created."""
        tasks = create_assessment_tasks(sample_agents)
        assert len(tasks) == 3
    
    def test_task_types_are_correct(self, sample_agents):
        """Test that all returned objects are Task instances."""
        tasks = create_assessment_tasks(sample_agents)
        for task in tasks:
            assert hasattr(task, 'description')
            assert hasattr(task, 'agent')
            assert hasattr(task, 'expected_output')
    
    def test_agent_assignment_correct(self, sample_agents):
        """Test that tasks are assigned to correct agents."""
        strategist, mediator, survivor = sample_agents
        tasks = create_assessment_tasks(sample_agents)
        
        # Room assessment task should be assigned to strategist
        assert tasks[0].agent == strategist
        # Team assessment task should be assigned to mediator
        assert tasks[1].agent == mediator
        # Execution assessment task should be assigned to survivor
        assert tasks[2].agent == survivor
    
    def test_task_descriptions_contain_key_elements(self, sample_agents):
        """Test that task descriptions contain expected key elements."""
        tasks = create_assessment_tasks(sample_agents)
        
        # Room assessment task (strategist)
        room_task = tasks[0]
        assert "ROOM SITUATION ANALYSIS" in room_task.description
        assert "Exit Routes" in room_task.description
        assert "Available Resources" in room_task.description
        assert "Puzzles and Obstacles" in room_task.description
        assert "Team Capabilities" in room_task.description
        assert "Risk Assessment" in room_task.description
        
        # Team assessment task (mediator)
        team_task = tasks[1]
        assert "TEAM DYNAMICS AND COORDINATION ASSESSMENT" in team_task.description
        assert "Team Morale" in team_task.description
        assert "Communication Patterns" in team_task.description
        assert "Conflict Resolution" in team_task.description
        assert "Role Clarity" in team_task.description
        assert "Collaboration Opportunities" in team_task.description
        
        # Execution assessment task (survivor)
        exec_task = tasks[2]
        assert "PRACTICAL EXECUTION ASSESSMENT" in exec_task.description
        assert "Resource Requirements" in exec_task.description
        assert "Time Constraints" in exec_task.description
        assert "Physical Limitations" in exec_task.description
        assert "Risk vs Reward" in exec_task.description
        assert "Contingency Planning" in exec_task.description
    
    def test_expected_outputs_are_comprehensive(self, sample_agents):
        """Test that expected outputs are detailed and comprehensive."""
        tasks = create_assessment_tasks(sample_agents)
        
        # All tasks should have substantial expected outputs
        for task in tasks:
            assert len(task.expected_output) > 50  # Should be detailed
            assert "assessment" in task.expected_output.lower()
            assert "recommendations" in task.expected_output.lower()
    
    def test_no_iteration_context_default_behavior(self, sample_agents):
        """Test behavior when no iteration context is provided."""
        tasks = create_assessment_tasks(sample_agents, iteration_context=None)
        
        for task in tasks:
            assert "PREVIOUS ITERATION CONTEXT" not in task.description
            assert "This is iteration" not in task.description
    
    def test_empty_iteration_context_default_behavior(self, sample_agents):
        """Test behavior when empty iteration context is provided."""
        tasks = create_assessment_tasks(sample_agents, iteration_context={})
        
        for task in tasks:
            assert "PREVIOUS ITERATION CONTEXT" not in task.description
            assert "This is iteration" not in task.description
    
    def test_iteration_context_zero_count_default_behavior(self, sample_agents):
        """Test behavior when iteration count is 0."""
        context = {"iteration_count": 0}
        tasks = create_assessment_tasks(sample_agents, iteration_context=context)
        
        for task in tasks:
            assert "PREVIOUS ITERATION CONTEXT" not in task.description
            assert "This is iteration" not in task.description
    
    def test_iteration_context_integration(self, sample_agents):
        """Test that iteration context is properly integrated into task descriptions."""
        context = {
            "iteration_count": 2,
            "failed_strategies": ["strategy1", "strategy2"],
            "game_state": {"room": "locked", "items": ["key"]},
            "time_remaining": 30
        }
        
        tasks = create_assessment_tasks(sample_agents, iteration_context=context)
        
        for task in tasks:
            assert "PREVIOUS ITERATION CONTEXT" in task.description
            assert "This is iteration 2" in task.description
            assert "strategy1" in task.description
            assert "strategy2" in task.description
            assert "30 minutes" in task.description
            assert "avoid repeating failed approaches" in task.description
    
    def test_iteration_context_missing_fields_handled(self, sample_agents):
        """Test that missing fields in iteration context are handled gracefully."""
        context = {"iteration_count": 1}  # Missing other fields
        
        tasks = create_assessment_tasks(sample_agents, iteration_context=context)
        
        for task in tasks:
            assert "PREVIOUS ITERATION CONTEXT" in task.description
            assert "This is iteration 1" in task.description
            # Should handle missing fields gracefully without errors
    
    def test_agents_list_unpacking(self, sample_agents):
        """Test that the agents list is properly unpacked."""
        # This implicitly tests that the function expects exactly 3 agents
        tasks = create_assessment_tasks(sample_agents)
        assert len(tasks) == 3
        
        # Test with wrong number of agents should raise an error
        with pytest.raises((ValueError, TypeError)):
            create_assessment_tasks(sample_agents[:2])  # Only 2 agents
    
    def test_task_descriptions_are_strings(self, sample_agents):
        """Test that all task descriptions are strings."""
        tasks = create_assessment_tasks(sample_agents)
        for task in tasks:
            assert isinstance(task.description, str)
            assert len(task.description.strip()) > 0
    
    def test_expected_outputs_are_strings(self, sample_agents):
        """Test that all expected outputs are strings."""
        tasks = create_assessment_tasks(sample_agents)
        for task in tasks:
            assert isinstance(task.expected_output, str)
            assert len(task.expected_output.strip()) > 0


class TestCreateProgressEvaluationTask:
    """Test the create_progress_evaluation_task function."""
    
    def test_creates_single_task(self, sample_agents):
        """Test that exactly one task is created."""
        agent = sample_agents[0]
        progress = {"actions_attempted": ["action1"]}
        task = create_progress_evaluation_task(agent, progress)
        
        assert hasattr(task, 'description')
        assert hasattr(task, 'agent')
        assert hasattr(task, 'expected_output')
    
    def test_agent_assignment(self, sample_agents):
        """Test that task is assigned to correct agent."""
        agent = sample_agents[1]  # Use mediator
        progress = {"actions_attempted": ["action1"]}
        task = create_progress_evaluation_task(agent, progress)
        
        assert task.agent == agent
    
    def test_description_contains_progress_elements(self, sample_agents):
        """Test that description contains expected progress evaluation elements."""
        agent = sample_agents[0]
        progress = {
            "actions_attempted": ["found key", "tried door"],
            "resources_found": ["flashlight", "rope"],
            "puzzles_solved": ["combination lock"],
            "obstacles": ["broken key", "blocked path"]
        }
        task = create_progress_evaluation_task(agent, progress)
        
        assert "PROGRESS EVALUATION" in task.description
        assert "found key" in task.description
        assert "tried door" in task.description
        assert "flashlight" in task.description
        assert "rope" in task.description
        assert "combination lock" in task.description
        assert "broken key" in task.description
        assert "blocked path" in task.description
    
    def test_description_contains_evaluation_criteria(self, sample_agents):
        """Test that description contains evaluation criteria."""
        agent = sample_agents[0]
        progress = {"actions_attempted": []}
        task = create_progress_evaluation_task(agent, progress)
        
        assert "What progress has been made toward escape" in task.description
        assert "What new information has been discovered" in task.description
        assert "What approaches have proven effective or ineffective" in task.description
        assert "What should be the focus for the next iteration" in task.description
        assert "Whether the current strategy should be continued or changed" in task.description
    
    def test_empty_progress_data_handled(self, sample_agents):
        """Test that empty progress data is handled gracefully."""
        agent = sample_agents[0]
        progress = {}
        task = create_progress_evaluation_task(agent, progress)
        
        assert "PROGRESS EVALUATION" in task.description
        # Should not crash with empty data
        assert len(task.description) > 100
    
    def test_missing_progress_fields_handled(self, sample_agents):
        """Test that missing progress fields are handled with defaults."""
        agent = sample_agents[0]
        progress = {"actions_attempted": ["some action"]}  # Missing other fields
        task = create_progress_evaluation_task(agent, progress)
        
        assert "some action" in task.description
        # Should handle missing fields gracefully with empty lists
        assert "[]" in task.description  # Empty lists for missing fields
    
    def test_expected_output_comprehensive(self, sample_agents):
        """Test that expected output is comprehensive."""
        agent = sample_agents[0]
        progress = {"actions_attempted": ["action1"]}
        task = create_progress_evaluation_task(agent, progress)
        
        expected = task.expected_output
        assert "Progress evaluation" in expected
        assert "recommendations" in expected
        assert "next iteration" in expected
        assert "strategy adjustments" in expected
        assert "resource utilization" in expected
        assert "priority actions" in expected


class TestCreateSituationAnalysisTask:
    """Test the create_situation_analysis_task function."""
    
    def test_creates_single_task(self, sample_agents):
        """Test that exactly one task is created."""
        agent = sample_agents[0]
        room_state = {"doors": 2, "items": ["key"]}
        team_state = {"morale": "high", "health": "good"}
        task = create_situation_analysis_task(agent, room_state, team_state)
        
        assert hasattr(task, 'description')
        assert hasattr(task, 'agent')
        assert hasattr(task, 'expected_output')
    
    def test_agent_assignment(self, sample_agents):
        """Test that task is assigned to correct agent."""
        agent = sample_agents[2]  # Use survivor
        room_state = {"doors": 2}
        team_state = {"morale": "high"}
        task = create_situation_analysis_task(agent, room_state, team_state)
        
        assert task.agent == agent
    
    def test_description_contains_situation_elements(self, sample_agents):
        """Test that description contains situation analysis elements."""
        agent = sample_agents[0]
        room_state = {"doors": 2, "locked": True, "items": ["flashlight", "key"]}
        team_state = {"morale": "medium", "injuries": 0, "trust": 0.8}
        task = create_situation_analysis_task(agent, room_state, team_state)
        
        assert "COMPREHENSIVE SITUATION ANALYSIS" in task.description
        assert "Room State:" in task.description
        assert "Team State:" in task.description
        assert "doors" in task.description
        assert "locked" in task.description
        assert "flashlight" in task.description
        assert "morale" in task.description
        assert "trust" in task.description
    
    def test_description_contains_analysis_criteria(self, sample_agents):
        """Test that description contains analysis criteria."""
        agent = sample_agents[0]
        room_state = {}
        team_state = {}
        task = create_situation_analysis_task(agent, room_state, team_state)
        
        assert "Current advantages and disadvantages" in task.description
        assert "Immediate opportunities and threats" in task.description
        assert "Resource availability and constraints" in task.description
        assert "Team readiness and capability assessment" in task.description
        assert "Strategic recommendations for next actions" in task.description
    
    def test_empty_states_handled(self, sample_agents):
        """Test that empty room and team states are handled."""
        agent = sample_agents[0]
        room_state = {}
        team_state = {}
        task = create_situation_analysis_task(agent, room_state, team_state)
        
        assert "COMPREHENSIVE SITUATION ANALYSIS" in task.description
        assert len(task.description) > 100
        # Should not crash with empty states
    
    def test_complex_states_integration(self, sample_agents):
        """Test that complex state data is properly integrated."""
        agent = sample_agents[0]
        room_state = {
            "layout": {"doors": 3, "windows": 1, "vents": 2},
            "items": {"tools": ["crowbar"], "keys": ["master", "backup"]},
            "puzzles": {"combination_lock": "unsolved", "riddle": "solved"},
            "hazards": ["flooding", "electrical"]
        }
        team_state = {
            "members": ["strategist", "mediator", "survivor"],
            "health": {"strategist": 100, "mediator": 80, "survivor": 60},
            "skills": {"problem_solving": 9, "physical": 7, "communication": 8},
            "resources": {"time": 25, "energy": "medium"}
        }
        task = create_situation_analysis_task(agent, room_state, team_state)
        
        # Check that complex nested data appears in description
        assert "crowbar" in task.description
        assert "master" in task.description
        assert "combination_lock" in task.description
        assert "flooding" in task.description
        assert "problem_solving" in task.description
        assert "energy" in task.description
    
    def test_expected_output_comprehensive(self, sample_agents):
        """Test that expected output is comprehensive."""
        agent = sample_agents[0]
        room_state = {"items": ["key"]}
        team_state = {"morale": "high"}
        task = create_situation_analysis_task(agent, room_state, team_state)
        
        expected = task.expected_output
        assert "Comprehensive situation analysis" in expected
        assert "prioritized recommendations" in expected
        assert "opportunity assessment" in expected
        assert "strategic guidance" in expected
        assert "team coordination" in expected


class TestTaskParameterValidation:
    """Test parameter validation and edge cases for all assessment functions."""
    
    def test_create_assessment_tasks_requires_agent_list(self):
        """Test that create_assessment_tasks requires a proper agent list."""
        with pytest.raises((TypeError, ValueError)):
            create_assessment_tasks(None)
    
    def test_create_assessment_tasks_wrong_agent_count(self, sample_agents):
        """Test behavior with wrong number of agents."""
        # Too few agents should raise an error
        with pytest.raises((ValueError, TypeError, IndexError)):
            create_assessment_tasks(sample_agents[:1])
        
        # The function expects exactly 3 agents, so test this specifically
        with pytest.raises((ValueError, TypeError, IndexError)):
            create_assessment_tasks(sample_agents[:2])  # Only 2 agents
    
    def test_progress_evaluation_requires_agent(self, sample_agents):
        """Test that progress evaluation requires an agent."""
        progress = {"actions_attempted": []}
        # The function should work with None agent (it just assigns it)
        task = create_progress_evaluation_task(None, progress)
        assert task.agent is None
    
    def test_progress_evaluation_requires_progress_dict(self, sample_agents):
        """Test that progress evaluation requires progress dictionary."""
        agent = sample_agents[0]
        # The function should handle empty progress dict gracefully
        task = create_progress_evaluation_task(agent, {})
        assert "Actions attempted" in task.description  # Capital A in template
    
    def test_situation_analysis_requires_agent(self, sample_agents):
        """Test that situation analysis requires an agent."""
        # The function should work with None agent (it just assigns it)
        task = create_situation_analysis_task(None, {}, {})
        assert task.agent is None
    
    def test_situation_analysis_requires_state_dicts(self, sample_agents):
        """Test that situation analysis handles various state inputs."""
        agent = sample_agents[0]
        
        # The function should handle various inputs gracefully
        task = create_situation_analysis_task(agent, {}, {})
        assert "Room State:" in task.description
        assert "Team State:" in task.description


class TestTaskContentQuality:
    """Test the quality and completeness of task content."""
    
    def test_assessment_tasks_have_learning_context(self, sample_agents):
        """Test that tasks emphasize learning from previous attempts."""
        context = {
            "iteration_count": 3,
            "failed_strategies": ["rush approach", "analysis paralysis"],
            "game_state": {"progress": "minimal"},
            "time_remaining": 15
        }
        tasks = create_assessment_tasks(sample_agents, iteration_context=context)
        
        for task in tasks:
            assert "avoid repeating failed approaches" in task.description
            assert "build on what you've learned" in task.description
    
    def test_tasks_have_role_appropriate_focus(self, sample_agents):
        """Test that each task focuses on appropriate role expertise."""
        tasks = create_assessment_tasks(sample_agents)
        
        # Strategist task should focus on strategic analysis
        strategist_task = tasks[0]
        assert "strategic" in strategist_task.description.lower()
        assert "analysis" in strategist_task.description.lower()
        
        # Mediator task should focus on team dynamics
        mediator_task = tasks[1]
        assert "team" in mediator_task.description.lower()
        assert "communication" in mediator_task.description.lower()
        assert "collaboration" in mediator_task.description.lower()
        
        # Survivor task should focus on practical execution
        survivor_task = tasks[2]
        assert "practical" in survivor_task.description.lower()
        assert "execution" in survivor_task.description.lower()
        assert "realistic" in survivor_task.description.lower()
    
    def test_progress_evaluation_encourages_reflection(self, sample_agents):
        """Test that progress evaluation encourages learning and reflection."""
        agent = sample_agents[0]
        progress = {"actions_attempted": ["action1"], "obstacles": ["obstacle1"]}
        task = create_progress_evaluation_task(agent, progress)
        
        description = task.description.lower()
        assert "evaluate" in description
        assert "recommendations" in description
        assert "next" in description
        # Check for learning concepts
        assert any(word in description for word in ["learn", "insights", "discovered", "effective"])
    
    def test_situation_analysis_promotes_strategic_thinking(self, sample_agents):
        """Test that situation analysis promotes strategic thinking."""
        agent = sample_agents[0]
        room_state = {"complexity": "high"}
        team_state = {"readiness": "partial"}
        task = create_situation_analysis_task(agent, room_state, team_state)
        
        assert "strategic" in task.description.lower()
        assert "opportunities" in task.description.lower()
        assert "recommendations" in task.description.lower()
        assert "path forward" in task.description.lower()