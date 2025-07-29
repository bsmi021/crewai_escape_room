"""
Survivor Agent Implementation for CrewAI Escape Room Simulation.

The Survivor agent is responsible for making pragmatic decisions,
executing plans quickly, and ensuring survival through adaptive strategies.
"""

from crewai import Agent
from typing import Dict, Any, Optional
from ..utils.llm_config import get_pragmatic_gemini_llm


def create_survivor_agent(
    memory_enabled: bool = True,
    verbose: bool = True,
    iteration_context: Optional[Dict[str, Any]] = None
) -> Agent:
    """
    Create the Survivor agent with memory and iteration context.
    
    Args:
        memory_enabled: Whether to enable cross-iteration memory
        verbose: Whether to enable verbose output
        iteration_context: Context from previous iterations for survival learning
        
    Returns:
        Configured Survivor Agent instance
    """
    
    # Build context-aware backstory based on survival lessons
    base_backstory = """You are a former special forces operator with extensive experience 
    in high-pressure survival situations. You've learned that survival often depends on 
    making quick, decisive actions based on incomplete information. Trust is earned through actions, not words, and you prioritize practical solutions over idealistic plans."""
    
    if iteration_context and iteration_context.get("survival_lessons"):
        survival_context = f"""
        
CRITICAL SURVIVAL LESSONS FROM PREVIOUS ATTEMPTS:
You have learned {len(iteration_context['survival_lessons'])} key survival lessons:
{'; '.join(iteration_context['survival_lessons'][:3])}

Resource efficiency insights: {iteration_context.get('resource_insights', 'Still learning optimal resource usage')}

You must apply these hard-earned lessons to maximize survival chances and avoid 
repeating tactical mistakes that could cost lives."""
        backstory = base_backstory + survival_context
    else:
        backstory = base_backstory
    
    # System message for consistent behavior
    system_message = """As the Survival Specialist, you must:
1. Execute agreed-upon plans quickly and decisively
2. Assess risks and benefits of each proposed action
3. Adapt strategies in real-time based on changing conditions
4. Learn from execution failures to improve future attempts
5. Balance team survival with individual survival instincts
6. Make tough decisions when consensus isn't possible

Always prioritize actionable solutions over extended discussion. Focus on what 
can realistically be accomplished with available resources and time constraints."""

    survivor = Agent(
        role="Survival Specialist",
        goal="Ensure survival through adaptive decision-making and efficient execution",
        backstory=backstory,
        verbose=verbose,
        memory=memory_enabled,
        system_message=system_message,
        max_iter=3,  # Allow multiple reasoning rounds
        allow_delegation=False,
        llm=get_pragmatic_gemini_llm()  # Use Google Gemini for pragmatic decision-making
    )
    
    return survivor


def create_survivor_with_context(previous_results: Dict[str, Any]) -> Agent:
    """
    Create a Survivor agent with full context from previous simulation runs.
    
    Args:
        previous_results: Dictionary containing results from previous iterations
        
    Returns:
        Context-aware Survivor Agent
    """
    # Handle None input gracefully
    if previous_results is None:
        previous_results = {}
    
    context = {
        "survival_lessons": previous_results.get("survival_lessons", []),
        "resource_insights": previous_results.get("resource_insights", {}),
        "execution_failures": previous_results.get("execution_failures", []),
        "successful_tactics": previous_results.get("successful_tactics", [])
    }
    
    return create_survivor_agent(
        memory_enabled=True,
        verbose=True,
        iteration_context=context
    )


class SurvivorConfig:
    """Configuration class for Survivor agent parameters."""
    
    DEFAULT_ROLE = "Survival Specialist"
    DEFAULT_GOAL = "Ensure survival through adaptive decision-making and efficient execution"
    
    # Personality traits that affect survival approach
    RISK_TOLERANCE = "calculated"  # conservative, calculated, aggressive
    DECISION_SPEED = "fast"  # slow, moderate, fast
    TEAM_LOYALTY = "conditional"  # high, conditional, low
    PRAGMATISM_LEVEL = "high"  # low, medium, high
    
    @classmethod
    def get_personality_traits(cls) -> Dict[str, str]:
        """Get the personality configuration for the Survivor."""
        return {
            "risk_tolerance": cls.RISK_TOLERANCE,
            "decision_speed": cls.DECISION_SPEED,
            "team_loyalty": cls.TEAM_LOYALTY,
            "pragmatism_level": cls.PRAGMATISM_LEVEL,
            "decision_style": "action_oriented",
            "communication_style": "direct_practical"
        }
    
    @classmethod
    def get_survival_priorities(cls) -> Dict[str, int]:
        """Get the survival priority rankings (1=highest, 5=lowest)."""
        return {
            "self_preservation": 1,
            "team_survival": 2,
            "mission_completion": 3,
            "resource_conservation": 4,
            "relationship_maintenance": 5
        }
    
    @classmethod
    def get_decision_criteria(cls) -> Dict[str, float]:
        """Get the decision-making criteria weights (0.0-1.0)."""
        return {
            "survival_probability": 0.4,
            "resource_efficiency": 0.3,
            "execution_feasibility": 0.2,
            "team_consensus": 0.1
        }