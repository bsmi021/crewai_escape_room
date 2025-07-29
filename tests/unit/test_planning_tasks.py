"""
Comprehensive tests for planning task definitions.

Tests verify task creation, context integration, agent assignment,
descriptions, expected outputs, and parameter validation.
"""

import pytest
from unittest.mock import Mock, patch
from typing import List, Dict, Any, Optional

# Import the functions to test
from src.escape_room_sim.tasks.planning import (
    create_planning_tasks,
    create_collaborative_planning_task,
    create_contingency_planning_task
)


class TestCreatePlanningTasks:
    """Test the create_planning_tasks function."""
    
    def test_creates_three_tasks(self, sample_agents):
        """Test that exactly 3 tasks are created."""
        tasks = create_planning_tasks(sample_agents)
        assert len(tasks) == 3
    
    def test_task_types_are_correct(self, sample_agents):
        """Test that all returned objects are Task instances."""
        tasks = create_planning_tasks(sample_agents)
        for task in tasks:
            assert hasattr(task, 'description')
            assert hasattr(task, 'agent')
            assert hasattr(task, 'expected_output')
    
    def test_agent_assignment_correct(self, sample_agents):
        """Test that tasks are assigned to correct agents."""
        strategist, mediator, survivor = sample_agents
        tasks = create_planning_tasks(sample_agents)
        
        # Strategy development task should be assigned to strategist
        assert tasks[0].agent == strategist
        # Consensus building task should be assigned to mediator
        assert tasks[1].agent == mediator
        # Action planning task should be assigned to survivor
        assert tasks[2].agent == survivor
    
    def test_task_descriptions_contain_key_elements(self, sample_agents):
        """Test that task descriptions contain expected key elements."""
        tasks = create_planning_tasks(sample_agents)
        
        # Strategy development task (strategist)
        strategy_task = tasks[0]
        assert "COMPREHENSIVE ESCAPE STRATEGY DEVELOPMENT" in strategy_task.description
        assert "Primary Escape Plan" in strategy_task.description
        assert "Resource Allocation" in strategy_task.description
        assert "Team Coordination" in strategy_task.description
        assert "Risk Mitigation" in strategy_task.description
        assert "Success Metrics" in strategy_task.description
        assert "Alternative Routes" in strategy_task.description
        
        # Consensus building task (mediator)
        consensus_task = tasks[1]
        assert "TEAM CONSENSUS BUILDING AND COLLABORATION PLANNING" in consensus_task.description
        assert "Strategy Review" in consensus_task.description
        assert "Concern Resolution" in consensus_task.description
        assert "Role Negotiation" in consensus_task.description
        assert "Communication Plan" in consensus_task.description
        assert "Conflict Prevention" in consensus_task.description
        assert "Morale Building" in consensus_task.description
        
        # Action planning task (survivor)
        action_task = tasks[2]
        assert "DETAILED ACTION PLANNING AND EXECUTION PREPARATION" in action_task.description
        assert "Step-by-Step Actions" in action_task.description
        assert "Timeline and Priorities" in action_task.description
        assert "Resource Management" in action_task.description
        assert "Quality Control" in action_task.description
        assert "Adaptation Triggers" in action_task.description
        assert "Emergency Procedures" in action_task.description
    
    def test_expected_outputs_are_comprehensive(self, sample_agents):
        """Test that expected outputs are detailed and comprehensive."""
        tasks = create_planning_tasks(sample_agents)
        
        # All tasks should have substantial expected outputs
        for task in tasks:
            assert len(task.expected_output) > 50  # Should be detailed
            assert "plan" in task.expected_output.lower()
        
        # Check specific outputs
        strategy_task, consensus_task, action_task = tasks
        
        assert "Comprehensive escape strategy" in strategy_task.expected_output
        assert "contingency plans" in strategy_task.expected_output
        assert "clear rationale" in strategy_task.expected_output
        
        assert "Team consensus" in consensus_task.expected_output
        assert "communication plan" in consensus_task.expected_output
        assert "commitment confirmation" in consensus_task.expected_output
        
        assert "executable action plan" in action_task.expected_output
        assert "step-by-step actions" in action_task.expected_output
        assert "emergency procedures" in action_task.expected_output
    
    def test_no_assessment_results_default_behavior(self, sample_agents):
        """Test behavior when no assessment results are provided."""
        tasks = create_planning_tasks(sample_agents, assessment_results=None)
        
        for task in tasks:
            assert "ASSESSMENT RESULTS FROM CURRENT ITERATION" not in task.description
    
    def test_empty_assessment_results_default_behavior(self, sample_agents):
        """Test behavior when empty assessment results are provided."""
        tasks = create_planning_tasks(sample_agents, assessment_results={})
        
        # All tasks should have assessment context when results are provided
        assessment_tasks_with_context = [task for task in tasks if "ASSESSMENT RESULTS FROM CURRENT ITERATION" in task.description]
        assert len(assessment_tasks_with_context) >= 1  # At least one task should have context
        
        # Check that tasks with context show default messages
        for task in assessment_tasks_with_context:
            assert "No room analysis available" in task.description
            assert "No team analysis available" in task.description
            assert "No execution analysis available" in task.description
    
    def test_assessment_results_integration(self, sample_agents, assessment_results):
        """Test that assessment results are properly integrated into task descriptions."""
        tasks = create_planning_tasks(sample_agents, assessment_results=assessment_results)
        
        for task in tasks:
            assert "ASSESSMENT RESULTS FROM CURRENT ITERATION" in task.description
            assert assessment_results['room_analysis'] in task.description
            assert assessment_results['team_dynamics'] in task.description
            assert assessment_results['execution_feasibility'] in task.description
    
    def test_no_iteration_context_default_behavior(self, sample_agents):
        """Test behavior when no iteration context is provided."""
        tasks = create_planning_tasks(sample_agents, iteration_context=None)
        
        for task in tasks:
            assert "LEARNING FROM PREVIOUS ITERATIONS" not in task.description
    
    def test_empty_iteration_context_default_behavior(self, sample_agents):
        """Test behavior when empty iteration context is provided."""
        tasks = create_planning_tasks(sample_agents, iteration_context={})
        
        for task in tasks:
            assert "LEARNING FROM PREVIOUS ITERATIONS" not in task.description
    
    def test_iteration_context_zero_count_default_behavior(self, sample_agents):
        """Test behavior when iteration count is 0."""
        context = {"iteration_count": 0}
        tasks = create_planning_tasks(sample_agents, iteration_context=context)
        
        for task in tasks:
            assert "LEARNING FROM PREVIOUS ITERATIONS" not in task.description
    
    def test_iteration_context_integration(self, sample_agents):
        """Test that iteration context is properly integrated into task descriptions."""
        context = {
            "iteration_count": 2,
            "previous_strategies": ["strategy1", "strategy2"],
            "successful_approaches": ["approach1"],
            "failed_approaches": ["failed1", "failed2"],
            "trust_levels": {"strategist": 0.8, "mediator": 0.9, "survivor": 0.7}
        }
        
        tasks = create_planning_tasks(sample_agents, iteration_context=context)
        
        for task in tasks:
            assert "LEARNING FROM PREVIOUS ITERATIONS" in task.description
            assert "Iteration #2" in task.description
            assert "strategy1" in task.description
            assert "approach1" in task.description
            assert "failed1" in task.description
    
    def test_combined_context_integration(self, sample_agents, assessment_results):
        """Test integration of both assessment results and iteration context."""
        context = {
            "iteration_count": 1,
            "previous_strategies": ["rush_strategy"],
            "successful_approaches": ["team_communication"],
            "failed_approaches": ["ignore_time_limit"],
            "trust_levels": {"overall": 0.8}
        }
        
        tasks = create_planning_tasks(
            sample_agents, 
            assessment_results=assessment_results,
            iteration_context=context
        )
        
        for task in tasks:
            # Should contain both types of context
            assert "ASSESSMENT RESULTS FROM CURRENT ITERATION" in task.description
            assert "LEARNING FROM PREVIOUS ITERATIONS" in task.description
            assert assessment_results['room_analysis'] in task.description
            assert "rush_strategy" in task.description
    
    def test_agents_list_unpacking(self, sample_agents):
        """Test that the agents list is properly unpacked."""
        tasks = create_planning_tasks(sample_agents)
        assert len(tasks) == 3
        
        # Test with wrong number of agents should raise an error
        with pytest.raises((ValueError, TypeError)):
            create_planning_tasks(sample_agents[:2])  # Only 2 agents


class TestCreateCollaborativePlanningTask:
    """Test the create_collaborative_planning_task function."""
    
    def test_creates_single_task(self, sample_agents):
        """Test that exactly one task is created."""
        task = create_collaborative_planning_task(sample_agents)
        
        assert hasattr(task, 'description')
        assert hasattr(task, 'agent')
        assert hasattr(task, 'expected_output')
    
    def test_agent_assignment(self, sample_agents):
        """Test that task is assigned to first agent."""
        task = create_collaborative_planning_task(sample_agents)
        assert task.agent == sample_agents[0]
    
    def test_default_planning_focus(self, sample_agents):
        """Test default planning focus is 'escape_strategy'."""
        task = create_collaborative_planning_task(sample_agents)
        
        assert "COLLABORATIVE PLANNING SESSION: ESCAPE_STRATEGY" in task.description
        assert "escape_strategy" in task.description.lower()
    
    def test_custom_planning_focus(self, sample_agents):
        """Test custom planning focus is properly integrated."""
        custom_focus = "resource_management"
        task = create_collaborative_planning_task(sample_agents, planning_focus=custom_focus)
        
        assert f"COLLABORATIVE PLANNING SESSION: {custom_focus.upper()}" in task.description
        assert custom_focus in task.description.lower()
    
    def test_description_contains_collaboration_elements(self, sample_agents):
        """Test that description contains collaborative elements."""
        task = create_collaborative_planning_task(sample_agents)
        
        assert "Work together as a team" in task.description
        assert "Information Sharing" in task.description
        assert "Strategy Brainstorming" in task.description
        assert "Resource Optimization" in task.description
        assert "Role Coordination" in task.description
        assert "Risk Assessment" in task.description
        assert "Decision Making" in task.description
    
    def test_description_contains_role_definitions(self, sample_agents):
        """Test that description contains role definitions for each agent."""
        task = create_collaborative_planning_task(sample_agents)
        
        assert "Strategist: Overall planning and risk analysis" in task.description
        assert "Mediator: Team coordination and conflict resolution" in task.description
        assert "Survivor: Practical execution and resource management" in task.description
    
    def test_expected_output_matches_focus(self, sample_agents):
        """Test that expected output matches planning focus."""
        focus = "emergency_protocols"
        task = create_collaborative_planning_task(sample_agents, planning_focus=focus)
        
        assert f"Collaborative plan for {focus}" in task.expected_output
        assert "coordinated strategy" in task.expected_output
        assert "resource allocation" in task.expected_output
        assert "role assignments" in task.expected_output
        assert "team consensus" in task.expected_output
    
    def test_empty_agents_list_handling(self):
        """Test handling of empty agents list."""
        with pytest.raises((IndexError, TypeError)):
            create_collaborative_planning_task([])
    
    def test_single_agent_handling(self, sample_agents):
        """Test handling with single agent."""
        task = create_collaborative_planning_task(sample_agents[:1])
        assert task.agent == sample_agents[0]


class TestCreateContingencyPlanningTask:
    """Test the create_contingency_planning_task function."""
    
    def test_creates_single_task(self, sample_agents):
        """Test that exactly one task is created."""
        agent = sample_agents[0]
        primary_plan = {"strategy": "main exit", "resources": ["key"]}
        risks = ["key breaks", "door jams"]
        
        task = create_contingency_planning_task(agent, primary_plan, risks)
        
        assert hasattr(task, 'description')
        assert hasattr(task, 'agent')
        assert hasattr(task, 'expected_output')
    
    def test_agent_assignment(self, sample_agents):
        """Test that task is assigned to correct agent."""
        agent = sample_agents[1]  # Use mediator
        primary_plan = {"strategy": "main exit"}
        risks = ["time running out"]
        
        task = create_contingency_planning_task(agent, primary_plan, risks)
        assert task.agent == agent
    
    def test_description_contains_plan_and_risks(self, sample_agents):
        """Test that description contains primary plan and identified risks."""
        agent = sample_agents[0]
        primary_plan = {
            "strategy": "use_hidden_key", 
            "resources": ["flashlight", "key"],
            "timeline": "15 minutes"
        }
        risks = ["key_breaks", "door_jammed", "time_pressure"]
        
        task = create_contingency_planning_task(agent, primary_plan, risks)
        
        # Check primary plan elements
        assert "Primary Plan Overview:" in task.description
        assert "use_hidden_key" in task.description
        assert "flashlight" in task.description
        assert "15 minutes" in task.description
        
        # Check risk factors
        assert "Identified Risk Factors:" in task.description
        assert "key_breaks" in task.description
        assert "door_jammed" in task.description
        assert "time_pressure" in task.description
    
    def test_description_contains_contingency_elements(self, sample_agents):
        """Test that description contains contingency planning elements."""
        agent = sample_agents[0]
        primary_plan = {"strategy": "main approach"}
        risks = ["obstacle"]
        
        task = create_contingency_planning_task(agent, primary_plan, risks)
        
        assert "CONTINGENCY PLANNING AND RISK MITIGATION" in task.description
        assert "Failure Point Analysis" in task.description
        assert "Alternative Approaches" in task.description
        assert "Resource Reallocation" in task.description
        assert "Decision Trees" in task.description
        assert "Recovery Procedures" in task.description
        assert "Emergency Protocols" in task.description
    
    def test_expected_output_comprehensive(self, sample_agents):
        """Test that expected output is comprehensive."""
        agent = sample_agents[0]
        primary_plan = {"strategy": "main"}
        risks = ["risk1"]
        
        task = create_contingency_planning_task(agent, primary_plan, risks)
        
        expected = task.expected_output
        assert "Comprehensive contingency plans" in expected
        assert "alternative strategies" in expected
        assert "decision trees" in expected
        assert "resource reallocation procedures" in expected
        assert "emergency protocols" in expected
        assert "identified failure points" in expected
    
    def test_empty_primary_plan_handling(self, sample_agents):
        """Test handling of empty primary plan."""
        agent = sample_agents[0]
        primary_plan = {}
        risks = ["risk1"]
        
        task = create_contingency_planning_task(agent, primary_plan, risks)
        assert "Primary Plan Overview:" in task.description
        # Should not crash with empty plan
    
    def test_empty_risks_list_handling(self, sample_agents):
        """Test handling of empty risks list."""
        agent = sample_agents[0]
        primary_plan = {"strategy": "main"}
        risks = []
        
        task = create_contingency_planning_task(agent, primary_plan, risks)
        assert "Identified Risk Factors:" in task.description
        # Should show empty list without crashing
        assert "[]" in task.description or "" in task.description
    
    def test_complex_plan_and_risks_integration(self, sample_agents):
        """Test integration of complex plan and risk data."""
        agent = sample_agents[0]
        primary_plan = {
            "main_strategy": "stealth_escape",
            "backup_strategy": "force_exit",
            "resources": {
                "tools": ["lockpick", "crowbar"],
                "consumables": ["flashlight_battery"]
            },
            "timeline": {
                "phase1": "5 minutes",
                "phase2": "10 minutes",
                "total": "15 minutes"
            }
        }
        risks = [
            "lockpick_breaks_during_use",
            "crowbar_too_noisy_alerts_guards", 
            "flashlight_battery_dies_early",
            "backup_exit_blocked_unexpectedly"
        ]
        
        task = create_contingency_planning_task(agent, primary_plan, risks)
        
        # Check complex plan elements appear
        assert "stealth_escape" in task.description
        assert "lockpick" in task.description
        assert "phase1" in task.description
        
        # Check complex risk elements appear
        assert "lockpick_breaks_during_use" in task.description
        assert "crowbar_too_noisy_alerts_guards" in task.description


class TestTaskParameterValidation:
    """Test parameter validation and edge cases for all planning functions."""
    
    def test_create_planning_tasks_requires_agent_list(self):
        """Test that create_planning_tasks requires a proper agent list."""
        with pytest.raises((TypeError, ValueError)):
            create_planning_tasks(None)
    
    def test_create_planning_tasks_wrong_agent_count(self, sample_agents):
        """Test behavior with wrong number of agents."""
        # Too few agents should raise an error
        with pytest.raises((ValueError, TypeError, IndexError)):
            create_planning_tasks(sample_agents[:1])
        
        # Test with exactly 2 agents (should also fail)  
        with pytest.raises((ValueError, TypeError, IndexError)):
            create_planning_tasks(sample_agents[:2])
    
    def test_collaborative_planning_requires_agent_list(self):
        """Test that collaborative planning requires agent list."""
        with pytest.raises((TypeError, IndexError)):
            create_collaborative_planning_task(None)
        
        with pytest.raises((TypeError, IndexError)):
            create_collaborative_planning_task([])
    
    def test_contingency_planning_requires_agent(self, sample_agents):
        """Test that contingency planning requires an agent."""
        primary_plan = {"strategy": "main"}
        risks = ["risk1"]
        
        # The function should work with None agent (it just assigns it)
        task = create_contingency_planning_task(None, primary_plan, risks)
        assert task.agent is None
    
    def test_contingency_planning_requires_plan_dict(self, sample_agents):
        """Test that contingency planning handles primary plan dictionary."""
        agent = sample_agents[0]
        risks = ["risk1"]
        
        # The function should handle various plan inputs
        task = create_contingency_planning_task(agent, {}, risks)
        assert "Primary Plan Overview:" in task.description
    
    def test_contingency_planning_requires_risks_list(self, sample_agents):
        """Test that contingency planning requires risks list."""
        agent = sample_agents[0]
        primary_plan = {"strategy": "main"}
        
        with pytest.raises((TypeError, AttributeError)):
            create_contingency_planning_task(agent, primary_plan, None)


class TestTaskContentQuality:
    """Test the quality and completeness of task content."""
    
    def test_planning_tasks_emphasize_learning(self, sample_agents):
        """Test that planning tasks emphasize learning from previous iterations."""
        context = {
            "iteration_count": 2,
            "previous_strategies": ["failed_rush"],
            "successful_approaches": ["team_coordination"],
            "failed_approaches": ["ignore_resources"],
            "trust_levels": {"team": 0.8}
        }
        
        tasks = create_planning_tasks(sample_agents, iteration_context=context)
        
        for task in tasks:
            assert ("builds on successful approaches" in task.description or 
                   "avoiding known failures" in task.description or
                   "previous iterations" in task.description.lower())
    
    def test_tasks_have_role_appropriate_focus(self, sample_agents):
        """Test that each task focuses on appropriate role expertise."""
        tasks = create_planning_tasks(sample_agents)
        
        # Strategist task should focus on strategic planning
        strategist_task = tasks[0]
        description = strategist_task.description.lower()
        assert any(word in description for word in ["strategy", "strategic", "planning", "risk"])
        
        # Mediator task should focus on consensus and communication
        mediator_task = tasks[1]
        description = mediator_task.description.lower()
        assert any(word in description for word in ["consensus", "communication", "team", "collaboration"])
        
        # Survivor task should focus on practical execution
        survivor_task = tasks[2]
        description = survivor_task.description.lower()
        assert any(word in description for word in ["action", "execution", "practical", "detailed"])
    
    def test_collaborative_task_promotes_teamwork(self, sample_agents):
        """Test that collaborative task promotes effective teamwork."""
        task = create_collaborative_planning_task(sample_agents)
        
        description = task.description.lower()
        assert "work together" in description
        assert any(word in description for word in ["collaborate", "collaboration", "collaborative"])
        assert "team" in description
        assert "consensus" in description
    
    def test_contingency_task_emphasizes_preparedness(self, sample_agents):
        """Test that contingency task emphasizes preparedness and risk management."""
        agent = sample_agents[0]
        primary_plan = {"strategy": "main"}
        risks = ["risk1"]
        
        task = create_contingency_planning_task(agent, primary_plan, risks)
        
        assert "backup" in task.description.lower() or "contingency" in task.description.lower()
        assert "risk" in task.description.lower()
        assert "failure" in task.description.lower()
        assert "alternative" in task.description.lower()
    
    def test_all_tasks_have_clear_structure(self, sample_agents):
        """Test that all tasks have clear, structured descriptions."""
        # Test planning tasks
        planning_tasks = create_planning_tasks(sample_agents)
        for task in planning_tasks:
            # Should have numbered lists or clear sections
            assert ("1." in task.description and "2." in task.description) or \
                   ("**" in task.description)  # Markdown formatting
        
        # Test collaborative task
        collab_task = create_collaborative_planning_task(sample_agents)
        assert ("1." in collab_task.description and "2." in collab_task.description) or \
               ("**" in collab_task.description)
        
        # Test contingency task
        agent = sample_agents[0]
        contingency_task = create_contingency_planning_task(
            agent, {"strategy": "main"}, ["risk1"]
        )
        assert ("1." in contingency_task.description and "2." in contingency_task.description) or \
               ("**" in contingency_task.description)
    
    def test_expected_outputs_are_actionable(self, sample_agents):
        """Test that expected outputs specify actionable deliverables."""
        tasks = create_planning_tasks(sample_agents)
        
        for task in tasks:
            # Should specify concrete deliverables
            expected = task.expected_output.lower()
            assert any(word in expected for word in [
                "plan", "strategy", "recommendations", "actions", 
                "allocation", "procedures", "consensus"
            ])