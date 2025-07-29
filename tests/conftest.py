"""
Shared test fixtures and configuration for CrewAI Escape Room Agent tests.
"""

import pytest
import sys
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, Optional

# Mock the crewai module before any imports
if 'crewai' not in sys.modules:
    sys.modules['crewai'] = Mock()
    
    # Create a mock Agent class
    class MockAgent:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)
    
    # Create a mock Task class
    class MockTask:
        def __init__(self, description=None, agent=None, expected_output=None, **kwargs):
            self.description = description
            self.agent = agent
            self.expected_output = expected_output
            for key, value in kwargs.items():
                setattr(self, key, value)
        
        def __repr__(self):
            return f"MockTask(agent={getattr(self.agent, 'role', 'unknown') if self.agent else None})"
                
    sys.modules['crewai'].Agent = MockAgent
    sys.modules['crewai'].Task = MockTask

# Now we can safely import
try:
    from crewai import Agent, Task
except ImportError:
    # Fallback mock Agent class
    class Agent:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)
    
    # Fallback mock Task class
    class Task:
        def __init__(self, description=None, agent=None, expected_output=None, **kwargs):
            self.description = description
            self.agent = agent
            self.expected_output = expected_output
            for key, value in kwargs.items():
                setattr(self, key, value)


@pytest.fixture
def mock_crewai_agent():
    """Mock CrewAI Agent class for testing."""
    with patch('crewai.Agent') as mock_agent:
        mock_instance = Mock(spec=Agent)
        mock_agent.return_value = mock_instance
        yield mock_agent


@pytest.fixture
def sample_iteration_context():
    """Sample iteration context for testing context-aware agent creation."""
    return {
        "failed_strategies": [
            "Tried brute force approach to locked door",
            "Attempted to use key on wrong lock",
            "Ignored time pressure and ran out of time"
        ],
        "successful_approaches": [
            "Communication between team members worked well",
            "Systematic approach to clue analysis"
        ],
        "resource_constraints": {
            "time_limit": 30,
            "available_tools": ["flashlight", "key", "rope"],
            "team_size": 3
        },
        "team_dynamics": {
            "trust_level": "medium",
            "communication_quality": "good",
            "conflict_count": 1
        }
    }


@pytest.fixture
def strategist_context():
    """Context specific to Strategist agent testing."""
    return {
        "failed_strategies": [
            "Direct assault on main door failed",
            "Overcomplicated puzzle solution approach",
            "Ignored team input and acted alone"
        ],
        "successful_approaches": [
            "Systematic room analysis",
            "Resource inventory worked well"
        ],
        "resource_constraints": {
            "time_remaining": 20,
            "tools_available": ["crowbar", "keys", "map"],
            "team_health": "good"
        },
        "team_dynamics": {
            "cooperation_level": "high",
            "strategy_acceptance": "medium"
        }
    }


@pytest.fixture
def mediator_context():
    """Context specific to Mediator agent testing."""
    return {
        "team_conflicts": [
            "Strategist vs Survivor disagreement on risk level",
            "Communication breakdown under pressure",
            "Resource allocation dispute"
        ],
        "trust_levels": {
            "strategist_survivor": 0.7,
            "strategist_mediator": 0.9,
            "survivor_mediator": 0.8
        },
        "successful_collaborations": [
            "Team worked well on puzzle solving",
            "Effective resource sharing"
        ],
        "communication_issues": [
            "Talking over each other under stress",
            "Not listening to all perspectives"
        ]
    }


@pytest.fixture
def survivor_context():
    """Context specific to Survivor agent testing."""
    return {
        "survival_lessons": [
            "Moving too slowly cost valuable time",
            "Overthinking simple problems was dangerous",
            "Team consensus isn't always possible"
        ],
        "resource_insights": {
            "most_valuable": "time",
            "least_valuable": "complex_tools",
            "efficiency_rating": 0.6
        },
        "execution_failures": [
            "Hesitated too long on door choice",
            "Wasted resources on broken mechanism",
            "Failed to adapt when plan wasn't working"
        ],
        "successful_tactics": [
            "Quick decision-making saved time",
            "Prioritized survival over mission completion"
        ]
    }


@pytest.fixture
def empty_context():
    """Empty context for testing default behavior."""
    return {}


@pytest.fixture
def previous_results_comprehensive():
    """Comprehensive previous results for context creation testing."""
    return {
        "failed_strategies": [
            "Strategy 1 failed due to time constraints",
            "Strategy 2 failed due to resource limitations",
            "Strategy 3 failed due to team coordination issues"
        ],
        "successful_approaches": [
            "Systematic analysis worked well",
            "Team communication was effective"
        ],
        "resource_constraints": {
            "time_limit": 45,
            "available_items": ["key", "rope", "flashlight", "crowbar"],
            "restricted_areas": ["basement", "upper_floor"]
        },
        "team_dynamics": {
            "overall_cohesion": 0.75,
            "leadership_acceptance": 0.6,
            "stress_level": "medium"
        },
        "team_conflicts": [
            "Disagreement on priority tasks",
            "Resource allocation conflict",
            "Leadership challenge"
        ],
        "trust_levels": {
            "average_trust": 0.7,
            "individual_ratings": {"strategist": 0.8, "mediator": 0.9, "survivor": 0.6}
        },
        "successful_collaborations": [
            "Puzzle solving collaboration",
            "Information sharing success"
        ],
        "communication_issues": [
            "Interrupting during discussions",
            "Not clarifying instructions"
        ],
        "survival_lessons": [
            "Speed over perfection in crisis",
            "Backup plans are essential",
            "Trust your instincts"
        ],
        "resource_insights": {
            "optimization_score": 0.65,
            "waste_patterns": ["overanalysis", "hesitation"],
            "efficiency_improvements": ["faster_decisions", "clearer_communication"]
        },
        "execution_failures": [
            "Delayed execution of agreed plan",
            "Poor resource utilization",
            "Failure to adapt to changing conditions"
        ],
        "successful_tactics": [
            "Quick problem identification",
            "Effective team coordination when needed"
        ]
    }


@pytest.fixture
def mock_agent_properties():
    """Mock agent properties for validation testing."""
    return {
        "role": "Test Role",
        "goal": "Test Goal",
        "backstory": "Test Backstory",
        "verbose": True,
        "memory": True,
        "system_message": "Test System Message",
        "max_iter": 3,
        "allow_delegation": False
    }


@pytest.fixture(params=[True, False])
def memory_enabled_param(request):
    """Parametrized fixture for memory enabled/disabled testing."""
    return request.param


@pytest.fixture(params=[True, False])
def verbose_param(request):
    """Parametrized fixture for verbose enabled/disabled testing."""
    return request.param


@pytest.fixture(params=[None, "sample_context", "empty_context"])
def iteration_context_param(request, sample_iteration_context, empty_context):
    """Parametrized fixture for different iteration contexts."""
    if request.param == "sample_context":
        return sample_iteration_context
    elif request.param == "empty_context":
        return empty_context
    else:
        return None


@pytest.fixture
def mock_llm_response():
    """Mock LLM response for integration testing."""
    return {
        "content": "This is a mock response from the LLM",
        "role": "assistant",
        "metadata": {
            "model": "mock-model",
            "tokens_used": 100
        }
    }


@pytest.fixture
def agent_test_scenarios():
    """Common test scenarios for all agents."""
    return {
        "basic_creation": {
            "memory_enabled": True,
            "verbose": True,
            "iteration_context": None
        },
        "memory_disabled": {
            "memory_enabled": False,
            "verbose": True,
            "iteration_context": None
        },
        "context_aware": {
            "memory_enabled": True,
            "verbose": True,
            "iteration_context": "provided"
        },
        "minimal_config": {
            "memory_enabled": False,
            "verbose": False,  
            "iteration_context": None
        }
    }


class MockAgent:
    """Mock Agent class for testing without actual CrewAI dependencies."""
    
    def __init__(self, role=None, goal=None, backstory=None, verbose=True, 
                 memory=True, system_message=None, max_iter=3, allow_delegation=False):
        self.role = role
        self.goal = goal
        self.backstory = backstory
        self.verbose = verbose
        self.memory = memory
        self.system_message = system_message
        self.max_iter = max_iter
        self.allow_delegation = allow_delegation
        
    def __repr__(self):
        return f"MockAgent(role='{self.role}', memory={self.memory})"


@pytest.fixture
def mock_agent_class():
    """Fixture that provides MockAgent class for testing."""
    return MockAgent


@pytest.fixture
def mock_crewai_task():
    """Mock CrewAI Task class for testing."""
    with patch('crewai.Task') as mock_task:
        mock_instance = Mock(spec=Task)
        mock_task.return_value = mock_instance
        yield mock_task


@pytest.fixture
def sample_agents():
    """Sample agent instances for task testing."""
    strategist = MockAgent(role="strategist", goal="analyze and plan", backstory="strategic thinker")
    mediator = MockAgent(role="mediator", goal="facilitate team", backstory="team coordinator")
    survivor = MockAgent(role="survivor", goal="ensure survival", backstory="practical executor")
    return [strategist, mediator, survivor]


@pytest.fixture
def assessment_results():
    """Sample assessment results for planning tasks."""
    return {
        'room_analysis': 'Room has two exits, one locked door, hidden key behind painting',
        'team_dynamics': 'Team cohesion is good, minor stress but manageable',
        'execution_feasibility': 'Plan is feasible with current resources and time constraints'
    }


@pytest.fixture
def action_plan():
    """Sample action plan for execution tasks."""
    return {
        'strategy': 'Use hidden key to unlock door and escape through main exit',
        'actions': ['find key', 'unlock door', 'coordinate exit'],
        'resources': ['flashlight', 'key', 'rope'],
        'timeline': '20 minutes total execution time'
    }


@pytest.fixture
def execution_results():
    """Sample execution results for evaluation tasks."""
    return {
        'success': False,
        'actions_completed': ['found key', 'attempted unlock'],
        'obstacles_encountered': ['key broke in lock', 'backup exit blocked'],
        'resources_used': ['flashlight', 'broken key'],
        'time_elapsed': 25
    }


@pytest.fixture
def escape_options():
    """Sample escape options for final decision tasks."""
    return {
        'main_exit': {'capacity': 2, 'risk': 'medium', 'time_required': 10},
        'emergency_exit': {'capacity': 1, 'risk': 'high', 'time_required': 5},
        'ventilation_shaft': {'capacity': 1, 'risk': 'low', 'time_required': 15}
    }


@pytest.fixture
def survival_constraints():
    """Sample survival constraints for final decisions."""
    return {
        'time_remaining': 8,
        'team_injuries': {'strategist': 'none', 'mediator': 'minor', 'survivor': 'injured'},
        'available_resources': ['rope', 'flashlight'],
        'escape_capacity_total': 2
    }