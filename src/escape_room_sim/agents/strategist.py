"""
Strategist Agent Implementation for CrewAI Escape Room Simulation.

The Strategist agent is responsible for analytical problem-solving, 
learning from failed attempts, and developing optimal escape strategies.
"""

from crewai import Agent
from typing import Dict, Any, Optional
from ..utils.llm_config import get_strategic_gemini_llm


def create_strategist_agent(
    memory_enabled: bool = True,
    verbose: bool = True,
    iteration_context: Optional[Dict[str, Any]] = None
) -> Agent:
    """
    Create the Strategist agent with memory and iteration context.
    
    Args:
        memory_enabled: Whether to enable cross-iteration memory
        verbose: Whether to enable verbose output
        iteration_context: Context from previous iterations for learning
        
    Returns:
        Configured Strategist Agent instance
    """
    
    # Build context-aware backstory based on iteration history
    base_backstory = """You are a former military tactician with extensive experience in 
    strategic planning and crisis management. You approach problems systematically, 
    analyzing all available information before making decisions. You believe the best 
    solution saves the most lives, even if it requires difficult choices."""
    
    if iteration_context and iteration_context.get("failed_strategies"):
        learning_context = f"""
        
IMPORTANT LEARNING FROM PREVIOUS ATTEMPTS:
You have learned from {len(iteration_context['failed_strategies'])} previous failed strategies:
{'; '.join(iteration_context['failed_strategies'][:3])}

You must avoid repeating these approaches and instead adapt your strategy based on 
what you've learned about the room constraints and team dynamics."""
        backstory = base_backstory + learning_context
    else:
        backstory = base_backstory
    
    # System message for consistent behavior
    system_message = """As the Strategic Analyst, you must:
1. Systematically analyze the escape room situation
2. Learn from previous failed attempts and adapt your approach
3. Consider all team members' capabilities and constraints
4. Propose logical, evidence-based solutions
5. Balance optimal outcomes with practical limitations
6. Track what works and what doesn't work for future iterations

Always begin your analysis by reviewing any previous attempts and explicitly state 
what you've learned from failures. Be specific about resource allocation and task assignments."""

    strategist = Agent(
        role="Strategic Analyst",
        goal="Find the optimal solution through iterative problem-solving and learning from failures",
        backstory=backstory,
        verbose=verbose,
        memory=memory_enabled,
        system_message=system_message,
        max_iter=3,  # Allow multiple reasoning rounds
        allow_delegation=False,
        llm=get_strategic_gemini_llm()  # Use Google Gemini for strategic analysis
    )
    
    return strategist


def create_strategist_with_context(previous_results: Dict[str, Any]) -> Agent:
    """
    Create a Strategist agent with full context from previous simulation runs.
    
    Args:
        previous_results: Dictionary containing results from previous iterations
        
    Returns:
        Context-aware Strategist Agent
    """
    # Handle None input gracefully
    if previous_results is None:
        previous_results = {}
    
    context = {
        "failed_strategies": previous_results.get("failed_strategies", []),
        "successful_approaches": previous_results.get("successful_approaches", []),
        "resource_constraints": previous_results.get("resource_constraints", {}),
        "team_dynamics": previous_results.get("team_dynamics", {})
    }
    
    # Handle None values by converting to empty lists/dicts
    for key, value in context.items():
        if value is None:
            if key in ["failed_strategies", "successful_approaches"]:
                context[key] = []
            else:
                context[key] = {}
    
    return create_strategist_agent(
        memory_enabled=True,
        verbose=True,
        iteration_context=context
    )


class StrategistConfig:
    """Configuration class for Strategist agent parameters."""
    
    DEFAULT_ROLE = "Strategic Analyst"
    DEFAULT_GOAL = "Find the optimal solution through iterative problem-solving and learning from failures"
    
    # Personality traits that affect decision making
    RISK_TOLERANCE = "moderate"  # conservative, moderate, aggressive
    COLLABORATION_STYLE = "analytical"  # analytical, directive, consultative
    LEARNING_RATE = "high"  # low, medium, high - how quickly to adapt from failures
    
    @classmethod
    def get_personality_traits(cls) -> Dict[str, str]:
        """Get the personality configuration for the Strategist."""
        return {
            "risk_tolerance": cls.RISK_TOLERANCE,
            "collaboration_style": cls.COLLABORATION_STYLE,
            "learning_rate": cls.LEARNING_RATE,
            "decision_style": "evidence_based",
            "communication_style": "direct_analytical"
        }