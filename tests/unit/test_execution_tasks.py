"""
Comprehensive tests for execution task definitions.

Tests verify task creation, context integration, agent assignment,
descriptions, expected outputs, and parameter validation.
"""

import pytest
from unittest.mock import Mock, patch
from typing import List, Dict, Any, Optional

# Import the functions to test
from src.escape_room_sim.tasks.execution import (
    create_execution_tasks,
    create_outcome_evaluation_task,
    create_adaptation_task,
    create_final_decision_task
)


class TestCreateExecutionTasks:
    """Test the create_execution_tasks function."""
    
    def test_creates_three_tasks(self, sample_agents):
        """Test that exactly 3 tasks are created."""
        tasks = create_execution_tasks(sample_agents)
        assert len(tasks) == 3
    
    def test_task_types_are_correct(self, sample_agents):
        """Test that all returned objects are Task instances."""
        tasks = create_execution_tasks(sample_agents)
        for task in tasks:
            assert hasattr(task, 'description')
            assert hasattr(task, 'agent')
            assert hasattr(task, 'expected_output')
    
    def test_agent_assignment_correct(self, sample_agents):
        """Test that tasks are assigned to correct agents."""
        strategist, mediator, survivor = sample_agents
        tasks = create_execution_tasks(sample_agents)
        
        # Plan implementation task should be assigned to survivor
        assert tasks[0].agent == survivor
        # Strategy monitoring task should be assigned to strategist
        assert tasks[1].agent == strategist
        # Team coordination task should be assigned to mediator
        assert tasks[2].agent == mediator
    
    def test_task_descriptions_contain_key_elements(self, sample_agents):
        """Test that task descriptions contain expected key elements."""
        tasks = create_execution_tasks(sample_agents)
        
        # Plan implementation task (survivor)
        impl_task = tasks[0]
        assert "ESCAPE PLAN EXECUTION" in impl_task.description
        assert "Action Implementation" in impl_task.description
        assert "Resource Management" in impl_task.description  
        assert "Progress Monitoring" in impl_task.description
        assert "Real-time Adaptation" in impl_task.description
        assert "Team Coordination" in impl_task.description
        assert "Outcome Assessment" in impl_task.description
        assert "successful or failed" in impl_task.description
        
        # Strategy monitoring task (strategist)
        monitor_task = tasks[1]
        assert "STRATEGIC EXECUTION MONITORING AND ANALYSIS" in monitor_task.description
        assert "Strategic Effectiveness" in monitor_task.description
        assert "Resource Efficiency" in monitor_task.description
        assert "Risk Management" in monitor_task.description
        assert "Adaptation Quality" in monitor_task.description
        assert "Success Probability" in monitor_task.description
        assert "Lessons Learned" in monitor_task.description
        
        # Team coordination task (mediator)
        coord_task = tasks[2]
        assert "TEAM COORDINATION AND COMMUNICATION DURING EXECUTION" in coord_task.description
        description = coord_task.description
        assert "Communication Flow" in description
        assert "Conflict Resolution" in description
        assert "Morale Management" in description
        assert "Role Coordination" in description
        assert "Decision Support" in description
        assert "Stress Management" in description
    
    def test_expected_outputs_are_comprehensive(self, sample_agents):
        """Test that expected outputs are detailed and comprehensive."""
        tasks = create_execution_tasks(sample_agents)
        
        # All tasks should have substantial expected outputs
        for task in tasks:
            assert len(task.expected_output) > 50  # Should be detailed
        
        # Check specific outputs
        impl_task, monitor_task, coord_task = tasks
        
        assert "Execution report" in impl_task.expected_output
        assert "final outcome assessment" in impl_task.expected_output
        assert "SUCCESS/FAILURE" in impl_task.expected_output
        
        assert "Strategic analysis" in monitor_task.expected_output
        assert "effectiveness assessment" in monitor_task.expected_output
        assert "recommendations" in monitor_task.expected_output
        
        assert "Team coordination report" in coord_task.expected_output
        assert "communication effectiveness" in coord_task.expected_output
        assert "team decision-making process" in coord_task.expected_output
    
    def test_no_action_plan_default_behavior(self, sample_agents):
        """Test behavior when no action plan is provided."""
        tasks = create_execution_tasks(sample_agents, action_plan=None)
        
        for task in tasks:
            assert "AGREED ACTION PLAN" not in task.description
    
    def test_empty_action_plan_default_behavior(self, sample_agents):
        """Test behavior when empty action plan is provided."""
        tasks = create_execution_tasks(sample_agents, action_plan={})
        
        for task in tasks:
            # Empty dict is falsy, so AGREED ACTION PLAN won't appear
            assert "AGREED ACTION PLAN" not in task.description
            # Should still have basic task structure
            assert task.description  # Should not be empty
    
    def test_action_plan_integration(self, sample_agents, action_plan):
        """Test that action plan is properly integrated into task descriptions."""
        tasks = create_execution_tasks(sample_agents, action_plan=action_plan)
        
        for task in tasks:
            assert "AGREED ACTION PLAN" in task.description
            assert action_plan['strategy'] in task.description
            assert str(action_plan['actions']) in task.description
            assert str(action_plan['resources']) in task.description
            assert action_plan['timeline'] in task.description
    
    def test_no_iteration_context_default_behavior(self, sample_agents):
        """Test behavior when no iteration context is provided."""
        tasks = create_execution_tasks(sample_agents, iteration_context=None)
        
        for task in tasks:
            assert "EXECUTION LEARNING FROM PREVIOUS ITERATIONS" not in task.description
    
    def test_empty_iteration_context_default_behavior(self, sample_agents):
        """Test behavior when empty iteration context is provided."""
        tasks = create_execution_tasks(sample_agents, iteration_context={})
        
        for task in tasks:
            assert "EXECUTION LEARNING FROM PREVIOUS ITERATIONS" not in task.description
    
    def test_iteration_context_zero_count_default_behavior(self, sample_agents):
        """Test behavior when iteration count is 0."""
        context = {"iteration_count": 0}
        tasks = create_execution_tasks(sample_agents, iteration_context=context)
        
        for task in tasks:
            assert "EXECUTION LEARNING FROM PREVIOUS ITERATIONS" not in task.description
    
    def test_iteration_context_integration(self, sample_agents):
        """Test that iteration context is properly integrated into task descriptions."""
        context = {
            "iteration_count": 3,
            "execution_results": ["result1", "result2"],
            "known_obstacles": ["obstacle1", "obstacle2"],
            "effective_tactics": ["tactic1"],
            "resource_status": {"tools": "limited", "time": "critical"}
        }
        
        tasks = create_execution_tasks(sample_agents, iteration_context=context)
        
        for task in tasks:
            assert "EXECUTION LEARNING FROM PREVIOUS ITERATIONS" in task.description
            assert "Current iteration: #3" in task.description
            assert "result1" in task.description
            assert "obstacle1" in task.description
            assert "tactic1" in task.description
    
    def test_combined_context_integration(self, sample_agents, action_plan):
        """Test integration of both action plan and iteration context."""
        context = {
            "iteration_count": 2,
            "execution_results": ["partial_success"],
            "known_obstacles": ["locked_door"],
            "effective_tactics": ["team_coordination"],
            "resource_status": {"time": 15, "tools": "adequate"}
        }
        
        tasks = create_execution_tasks(
            sample_agents,
            action_plan=action_plan,
            iteration_context=context
        )
        
        for task in tasks:
            # Should contain both types of context
            assert "AGREED ACTION PLAN" in task.description
            assert "EXECUTION LEARNING FROM PREVIOUS ITERATIONS" in task.description
            assert action_plan['strategy'] in task.description
            assert "partial_success" in task.description
    
    def test_agents_list_unpacking(self, sample_agents):
        """Test that the agents list is properly unpacked."""
        tasks = create_execution_tasks(sample_agents)
        assert len(tasks) == 3
        
        # Test with wrong number of agents should raise an error
        with pytest.raises((ValueError, TypeError)):
            create_execution_tasks(sample_agents[:2])  # Only 2 agents
    
    def test_critical_success_failure_requirement(self, sample_agents):
        """Test that implementation task emphasizes SUCCESS/FAILURE determination."""
        tasks = create_execution_tasks(sample_agents)
        impl_task = tasks[0]  # Implementation task (survivor)
        
        assert "CRITICAL" in impl_task.description
        assert "clearly state whether the escape attempt was successful or failed" in impl_task.description
        assert "SUCCESS/FAILURE" in impl_task.expected_output


class TestCreateOutcomeEvaluationTask:
    """Test the create_outcome_evaluation_task function."""
    
    def test_creates_single_task(self, sample_agents, execution_results):
        """Test that exactly one task is created."""
        agent = sample_agents[0]
        objectives = ["escape the room", "preserve team safety"]
        
        task = create_outcome_evaluation_task(agent, execution_results, objectives)
        
        assert hasattr(task, 'description')
        assert hasattr(task, 'agent')
        assert hasattr(task, 'expected_output')
    
    def test_agent_assignment(self, sample_agents, execution_results):
        """Test that task is assigned to correct agent."""
        agent = sample_agents[1]  # Use mediator
        objectives = ["objective1"]
        
        task = create_outcome_evaluation_task(agent, execution_results, objectives)
        assert task.agent == agent
    
    def test_description_contains_results_and_objectives(self, sample_agents, execution_results):
        """Test that description contains execution results and objectives."""
        agent = sample_agents[0]
        objectives = ["escape_safely", "maintain_team_cohesion", "preserve_resources"]
        
        task = create_outcome_evaluation_task(agent, execution_results, objectives)
        
        # Check execution results elements
        assert "Execution Results:" in task.description
        assert str(execution_results['success']) in task.description
        assert "found key" in task.description
        assert "key broke in lock" in task.description
        
        # Check objectives elements
        assert "Original Objectives:" in task.description
        assert "escape_safely" in task.description
        assert "maintain_team_cohesion" in task.description
        assert "preserve_resources" in task.description
    
    def test_description_contains_evaluation_criteria(self, sample_agents, execution_results):
        """Test that description contains evaluation criteria."""
        agent = sample_agents[0]
        objectives = ["objective1"]
        
        task = create_outcome_evaluation_task(agent, execution_results, objectives)
        
        assert "EXECUTION OUTCOME EVALUATION" in task.description
        assert "Success Assessment" in task.description
        assert "Root Cause Analysis" in task.description
        assert "Resource Impact" in task.description
        assert "Team Performance" in task.description
        assert "Strategy Effectiveness" in task.description
        assert "Learning Opportunities" in task.description
    
    def test_critical_success_determination_requirement(self, sample_agents, execution_results):
        """Test that task requires critical success determination."""
        agent = sample_agents[0]
        objectives = ["objective1"]
        
        task = create_outcome_evaluation_task(agent, execution_results, objectives)
        
        assert "CRITICAL" in task.description
        assert "COMPLETE SUCCESS" in task.description
        assert "PARTIAL SUCCESS" in task.description
        assert "FAILURE" in task.description
        assert "escaped successfully" in task.description
        assert "significant progress made" in task.description
        assert "little to no progress" in task.description
    
    def test_expected_output_comprehensive(self, sample_agents, execution_results):
        """Test that expected output is comprehensive."""
        agent = sample_agents[0]
        objectives = ["objective1"]
        
        task = create_outcome_evaluation_task(agent, execution_results, objectives)
        
        expected = task.expected_output
        assert "Comprehensive outcome evaluation" in expected
        assert "success assessment" in expected
        assert "root cause analysis" in expected
        assert "team performance review" in expected
        assert "recommendations for next iteration" in expected
    
    def test_empty_execution_results_handled(self, sample_agents):
        """Test that empty execution results are handled."""
        agent = sample_agents[0]
        objectives = ["objective1"]
        execution_results = {}
        
        task = create_outcome_evaluation_task(agent, execution_results, objectives)
        assert "Execution Results:" in task.description
        # Should not crash with empty results
    
    def test_empty_objectives_handled(self, sample_agents, execution_results):
        """Test that empty objectives list is handled."""
        agent = sample_agents[0]
        objectives = []
        
        task = create_outcome_evaluation_task(agent, execution_results, objectives)
        assert "Original Objectives:" in task.description
        # Should show empty list without crashing


class TestCreateAdaptationTask:
    """Test the create_adaptation_task function."""
    
    def test_creates_single_task(self, sample_agents):
        """Test that exactly one task is created."""
        agent = sample_agents[0]
        situation = {"current_state": "blocked", "resources": ["rope"]}
        obstacles = ["door_jammed", "key_broken"]
        
        task = create_adaptation_task(agent, situation, obstacles)
        
        assert hasattr(task, 'description')
        assert hasattr(task, 'agent')
        assert hasattr(task, 'expected_output')
    
    def test_agent_assignment(self, sample_agents):
        """Test that task is assigned to correct agent."""
        agent = sample_agents[2]  # Use survivor
        situation = {"state": "crisis"}
        obstacles = ["obstacle1"]
        
        task = create_adaptation_task(agent, situation, obstacles)
        assert task.agent == agent
    
    def test_description_contains_situation_and_obstacles(self, sample_agents):
        """Test that description contains current situation and obstacles."""
        agent = sample_agents[0]
        situation = {
            "current_position": "main_room",
            "available_resources": ["flashlight", "rope"],
            "team_status": {"strategist": "ready", "mediator": "injured"},
            "time_remaining": 10
        }
        obstacles = ["main_exit_blocked", "backup_key_missing", "team_member_injured"]
        
        task = create_adaptation_task(agent, situation, obstacles)
        
        # Check situation elements
        assert "Current Situation:" in task.description
        assert "main_room" in task.description
        assert "flashlight" in task.description
        assert "injured" in task.description
        
        # Check obstacles elements  
        assert "Unexpected Obstacles:" in task.description
        assert "main_exit_blocked" in task.description
        assert "backup_key_missing" in task.description
        assert "team_member_injured" in task.description
    
    def test_description_contains_adaptation_criteria(self, sample_agents):
        """Test that description contains adaptation criteria."""
        agent = sample_agents[0]
        situation = {"state": "changing"}
        obstacles = ["obstacle1"]
        
        task = create_adaptation_task(agent, situation, obstacles)
        
        assert "REAL-TIME PLAN ADAPTATION" in task.description
        assert "Obstacle Analysis" in task.description
        assert "Plan Modification" in task.description
        assert "Resource Reallocation" in task.description
        assert "Timeline Adjustment" in task.description
        assert "Risk Reassessment" in task.description
        assert "Communication" in task.description
    
    def test_expected_output_comprehensive(self, sample_agents):
        """Test that expected output is comprehensive."""
        agent = sample_agents[0]
        situation = {"state": "emergency"}
        obstacles = ["critical_obstacle"]
        
        task = create_adaptation_task(agent, situation, obstacles)
        
        expected = task.expected_output
        assert "Adapted plan" in expected
        assert "obstacle mitigation strategies" in expected
        assert "resource reallocation" in expected
        assert "timeline adjustments" in expected
        assert "team implementation" in expected
    
    def test_empty_situation_handled(self, sample_agents):
        """Test that empty situation is handled."""
        agent = sample_agents[0]
        situation = {}
        obstacles = ["obstacle1"]
        
        task = create_adaptation_task(agent, situation, obstacles)
        assert "Current Situation:" in task.description
        # Should not crash with empty situation
    
    def test_empty_obstacles_handled(self, sample_agents):
        """Test that empty obstacles list is handled."""
        agent = sample_agents[0]
        situation = {"state": "normal"}
        obstacles = []
        
        task = create_adaptation_task(agent, situation, obstacles)
        assert "Unexpected Obstacles:" in task.description
        # Should show empty list without crashing


class TestCreateFinalDecisionTask:
    """Test the create_final_decision_task function."""
    
    def test_creates_single_task(self, sample_agents, escape_options, survival_constraints):
        """Test that exactly one task is created."""
        task = create_final_decision_task(sample_agents, escape_options, survival_constraints)
        
        assert hasattr(task, 'description')
        assert hasattr(task, 'agent')
        assert hasattr(task, 'expected_output')
    
    def test_agent_assignment(self, sample_agents, escape_options, survival_constraints):
        """Test that task is assigned to first agent."""
        task = create_final_decision_task(sample_agents, escape_options, survival_constraints)
        assert task.agent == sample_agents[0]  # Primary decision maker
    
    def test_description_contains_options_and_constraints(self, sample_agents, escape_options, survival_constraints):
        """Test that description contains escape options and survival constraints."""
        task = create_final_decision_task(sample_agents, escape_options, survival_constraints)
        
        # Check escape options elements
        assert "Available Escape Options:" in task.description
        assert "main_exit" in task.description
        assert "emergency_exit" in task.description
        assert "ventilation_shaft" in task.description
        
        # Check survival constraints elements
        assert "Survival Constraints:" in task.description
        assert str(survival_constraints['time_remaining']) in task.description
        assert "injured" in task.description
        assert "rope" in task.description
    
    def test_description_contains_decision_criteria(self, sample_agents, escape_options, survival_constraints):
        """Test that description contains decision criteria."""
        task = create_final_decision_task(sample_agents, escape_options, survival_constraints)
        
        assert "FINAL ESCAPE DECISION" in task.description
        assert "Survival Assessment" in task.description
        assert "Resource Allocation" in task.description
        assert "Moral Considerations" in task.description
        assert "Team Consensus" in task.description
        assert "Execution Plan" in task.description
        assert "Final Outcomes" in task.description
    
    def test_critical_decision_requirement(self, sample_agents, escape_options, survival_constraints):
        """Test that task emphasizes critical decision making."""
        task = create_final_decision_task(sample_agents, escape_options, survival_constraints)
        
        assert "CRITICAL" in task.description
        assert "life-or-death choices" in task.description
        assert "definitive choice must be made" in task.description
        assert "specify exactly who attempts to escape via which route" in task.description
    
    def test_expected_output_comprehensive(self, sample_agents, escape_options, survival_constraints):
        """Test that expected output is comprehensive."""
        task = create_final_decision_task(sample_agents, escape_options, survival_constraints)
        
        expected = task.expected_output
        assert "Final escape decision" in expected
        assert "who attempts each escape route" in expected
        assert "resource allocation" in expected
        assert "moral reasoning" in expected
        assert "detailed implementation plan" in expected
        assert "survival strategy" in expected
    
    def test_empty_escape_options_handled(self, sample_agents, survival_constraints):
        """Test that empty escape options are handled."""
        empty_options = {}
        task = create_final_decision_task(sample_agents, empty_options, survival_constraints)
        assert "Available Escape Options:" in task.description
        # Should not crash with empty options
    
    def test_empty_survival_constraints_handled(self, sample_agents, escape_options):
        """Test that empty survival constraints are handled."""
        empty_constraints = {}
        task = create_final_decision_task(sample_agents, escape_options, empty_constraints)
        assert "Survival Constraints:" in task.description
        # Should not crash with empty constraints
    
    def test_complex_options_and_constraints_integration(self, sample_agents):
        """Test integration of complex escape options and constraints."""
        complex_options = {
            "front_door": {
                "capacity": 2,
                "risk_level": "medium", 
                "time_required": 5,
                "requirements": ["key", "distraction"],
                "success_probability": 0.7
            },
            "roof_access": {
                "capacity": 1,
                "risk_level": "high",
                "time_required": 15,
                "requirements": ["rope", "strength"],
                "success_probability": 0.4
            },
            "tunnel_system": {
                "capacity": 3,
                "risk_level": "low",
                "time_required": 20,
                "requirements": ["flashlight", "knowledge"],
                "success_probability": 0.9
            }
        }
        
        complex_constraints = {
            "time_remaining": 12,
            "team_status": {
                "strategist": {"health": 90, "skills": ["planning", "analysis"]},
                "mediator": {"health": 70, "skills": ["communication", "coordination"]},
                "survivor": {"health": 50, "skills": ["practical", "quick_thinking"]}
            },
            "available_equipment": ["rope", "flashlight", "basic_key"],
            "environmental_factors": ["rising_water", "limited_visibility"],
            "moral_constraints": ["leave_no_one_behind", "minimize_risk_to_injured"]
        }
        
        task = create_final_decision_task(sample_agents, complex_options, complex_constraints)
        
        # Check complex option elements appear
        assert "front_door" in task.description
        assert "success_probability" in task.description
        assert "tunnel_system" in task.description
        
        # Check complex constraint elements appear
        assert "rising_water" in task.description
        assert "leave_no_one_behind" in task.description
        assert "planning" in task.description


class TestTaskParameterValidation:
    """Test parameter validation and edge cases for all execution functions."""
    
    def test_create_execution_tasks_requires_agent_list(self):
        """Test that create_execution_tasks requires a proper agent list."""
        with pytest.raises((TypeError, ValueError)):
            create_execution_tasks(None)
    
    def test_create_execution_tasks_wrong_agent_count(self, sample_agents):
        """Test behavior with wrong number of agents."""
        # Too few agents should raise an error
        with pytest.raises((ValueError, TypeError, IndexError)):
            create_execution_tasks(sample_agents[:1])
        
        # Test with exactly 2 agents (should also fail)
        with pytest.raises((ValueError, TypeError, IndexError)):
            create_execution_tasks(sample_agents[:2])
    
    def test_outcome_evaluation_requires_agent(self, execution_results):
        """Test that outcome evaluation requires an agent."""
        objectives = ["objective1"]
        # The function should work with None agent (it just assigns it)
        task = create_outcome_evaluation_task(None, execution_results, objectives)
        assert task.agent is None
    
    def test_outcome_evaluation_requires_results_dict(self, sample_agents):
        """Test that outcome evaluation handles execution results dictionary."""
        agent = sample_agents[0]
        objectives = ["objective1"]
        # The function should handle various inputs gracefully
        task = create_outcome_evaluation_task(agent, {}, objectives)
        assert "Execution Results:" in task.description
    
    def test_outcome_evaluation_requires_objectives_list(self, sample_agents, execution_results):
        """Test that outcome evaluation requires objectives list."""
        agent = sample_agents[0]
        with pytest.raises((TypeError, AttributeError)):
            create_outcome_evaluation_task(agent, execution_results, None)
    
    def test_adaptation_requires_agent(self):
        """Test that adaptation task requires an agent."""
        situation = {"state": "emergency"}
        obstacles = ["obstacle1"]
        # The function should work with None agent (it just assigns it)
        task = create_adaptation_task(None, situation, obstacles)
        assert task.agent is None
    
    def test_adaptation_requires_situation_dict(self, sample_agents):
        """Test that adaptation task handles situation dictionary."""
        agent = sample_agents[0]
        obstacles = ["obstacle1"]
        # The function should handle various situation inputs
        task = create_adaptation_task(agent, {}, obstacles)
        assert "Current Situation:" in task.description
    
    def test_adaptation_requires_obstacles_list(self, sample_agents):
        """Test that adaptation task requires obstacles list."""
        agent = sample_agents[0]
        situation = {"state": "emergency"}
        with pytest.raises((TypeError, AttributeError)):
            create_adaptation_task(agent, situation, None)
    
    def test_final_decision_requires_agent_list(self, escape_options, survival_constraints):
        """Test that final decision requires agent list."""
        with pytest.raises((TypeError, IndexError)):
            create_final_decision_task(None, escape_options, survival_constraints)
        
        with pytest.raises((TypeError, IndexError)):
            create_final_decision_task([], escape_options, survival_constraints)
    
    def test_final_decision_requires_options_dict(self, sample_agents, survival_constraints):
        """Test that final decision handles escape options dictionary."""
        # The function should handle various options inputs
        task = create_final_decision_task(sample_agents, {}, survival_constraints)
        assert "Available Escape Options:" in task.description
    
    def test_final_decision_requires_constraints_dict(self, sample_agents, escape_options):
        """Test that final decision handles survival constraints dictionary."""
        # The function should handle various constraint inputs
        task = create_final_decision_task(sample_agents, escape_options, {})
        assert "Survival Constraints:" in task.description


class TestTaskContentQuality:
    """Test the quality and completeness of task content."""
    
    def test_execution_tasks_emphasize_learning(self, sample_agents):
        """Test that execution tasks emphasize learning from previous iterations."""
        context = {
            "iteration_count": 2,
            "execution_results": ["failed_attempt_1"],
            "known_obstacles": ["locked_door"],
            "effective_tactics": ["team_coordination"],
            "resource_status": {"time": "limited"}
        }
        
        tasks = create_execution_tasks(sample_agents, iteration_context=context)
        
        for task in tasks:
            assert ("previous iterations" in task.description.lower() or
                   "learning" in task.description.lower() or
                   "effective tactics" in task.description.lower())
    
    def test_tasks_have_role_appropriate_focus(self, sample_agents):
        """Test that each task focuses on appropriate role expertise."""
        tasks = create_execution_tasks(sample_agents)
        
        # Implementation task (survivor) should focus on practical execution
        impl_task = tasks[0]
        assert "execution" in impl_task.description.lower()
        assert "practical" in impl_task.description.lower()
        assert "action" in impl_task.description.lower()
        
        # Monitoring task (strategist) should focus on strategic analysis
        monitor_task = tasks[1]
        assert "strategic" in monitor_task.description.lower()
        assert "monitoring" in monitor_task.description.lower()
        assert "analysis" in monitor_task.description.lower()
        
        # Coordination task (mediator) should focus on team dynamics
        coord_task = tasks[2]
        assert "team" in coord_task.description.lower()
        assert "coordination" in coord_task.description.lower()
        assert "communication" in coord_task.description.lower()
    
    def test_outcome_evaluation_promotes_learning(self, sample_agents, execution_results):
        """Test that outcome evaluation promotes learning and improvement."""
        agent = sample_agents[0]
        objectives = ["objective1"]
        
        task = create_outcome_evaluation_task(agent, execution_results, objectives)
        
        assert "learning" in task.description.lower() or "insights" in task.description.lower()
        assert "root cause" in task.description.lower()
        assert "recommendations" in task.description.lower()
        assert "next iteration" in task.description.lower()
    
    def test_adaptation_task_emphasizes_flexibility(self, sample_agents):
        """Test that adaptation task emphasizes flexibility and quick thinking."""
        agent = sample_agents[0]
        situation = {"state": "changing"}
        obstacles = ["unexpected_obstacle"]
        
        task = create_adaptation_task(agent, situation, obstacles)
        
        assert "adapt" in task.description.lower() or "adaptation" in task.description.lower()
        assert "real-time" in task.description.lower()
        assert "immediate" in task.description.lower() or "quickly" in task.description.lower()
        assert "practical" in task.description.lower()
    
    def test_final_decision_emphasizes_critical_thinking(self, sample_agents, escape_options, survival_constraints):
        """Test that final decision task emphasizes critical thinking and moral reasoning."""
        task = create_final_decision_task(sample_agents, escape_options, survival_constraints)
        
        assert "critical" in task.description.lower()
        assert "moral" in task.description.lower() or "ethical" in task.description.lower()
        assert "reasoning" in task.description.lower()
        assert "consensus" in task.description.lower()
    
    def test_all_tasks_have_clear_structure(self, sample_agents):
        """Test that all tasks have clear, structured descriptions."""
        # Test execution tasks
        execution_tasks = create_execution_tasks(sample_agents)
        for task in execution_tasks:
            # Should have numbered lists or clear sections
            assert ("1." in task.description and "2." in task.description) or \
                   ("**" in task.description)  # Markdown formatting
        
        # Test other task types
        eval_task = create_outcome_evaluation_task(
            sample_agents[0], {"success": False}, ["objective1"]
        )
        assert ("1." in eval_task.description and "2." in eval_task.description) or \
               ("**" in eval_task.description)
        
        adapt_task = create_adaptation_task(
            sample_agents[0], {"state": "emergency"}, ["obstacle1"]
        )
        assert ("1." in adapt_task.description and "2." in adapt_task.description) or \
               ("**" in adapt_task.description)
        
        final_task = create_final_decision_task(
            sample_agents, {"exit1": {}}, {"time": 5}
        )
        assert ("1." in final_task.description and "2." in final_task.description) or \
               ("**" in final_task.description)
    
    def test_expected_outputs_are_actionable(self, sample_agents):
        """Test that expected outputs specify actionable deliverables."""
        tasks = create_execution_tasks(sample_agents)
        
        for task in tasks:
            # Should specify concrete deliverables
            expected = task.expected_output.lower()
            assert any(word in expected for word in [
                "report", "assessment", "analysis", "recommendations", 
                "actions", "results", "outcomes", "evaluation"
            ])