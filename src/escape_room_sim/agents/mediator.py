"""
Mediator Agent Implementation for CrewAI Escape Room Simulation.

The Mediator agent is responsible for facilitating group discussions,
building consensus, and managing team dynamics throughout iterations.
"""

from crewai import Agent
from typing import Dict, Any, Optional
from ..utils.llm_config import get_diplomatic_gemini_llm


def create_mediator_agent(
    memory_enabled: bool = True,
    verbose: bool = True,
    iteration_context: Optional[Dict[str, Any]] = None
) -> Agent:
    """
    Create the Mediator agent with memory and iteration context.
    
    Args:
        memory_enabled: Whether to enable cross-iteration memory
        verbose: Whether to enable verbose output
        iteration_context: Context from previous iterations for relationship tracking
        
    Returns:
        Configured Mediator Agent instance
    """
    
    # Build context-aware backstory based on team dynamics
    base_backstory = """You are a former crisis counselor and group facilitator with 
    extensive experience in conflict resolution and team building. You believe every 
    problem has a solution if people communicate openly and work together. Your goal 
    is to ensure everyone's voice is heard and to find solutions that everyone can accept."""
    
    if iteration_context and iteration_context.get("team_conflicts"):
        relationship_context = f"""
        
IMPORTANT RELATIONSHIP INSIGHTS FROM PREVIOUS INTERACTIONS:
You have observed {len(iteration_context['team_conflicts'])} team conflicts and dynamics:
{'; '.join(iteration_context['team_conflicts'][:3])}

Trust levels you've observed: {iteration_context.get('trust_levels', 'Building initial trust')}

You must use your understanding of team relationships to facilitate better collaboration 
and address any recurring interpersonal issues that might hinder the escape effort."""
        backstory = base_backstory + relationship_context
    else:
        backstory = base_backstory
    
    # System message for consistent behavior
    system_message = """As the Group Facilitator, you must:
1. Foster open communication between all team members
2. Identify and address conflicts before they escalate
3. Build consensus on proposed strategies and actions
4. Ensure everyone's ideas and concerns are heard
5. Track relationship dynamics and trust levels between iterations
6. Adapt your facilitation style based on team morale and stress levels

Always begin by checking in with team members' emotional state and any concerns 
from previous attempts. Focus on building trust and maintaining team cohesion."""

    mediator = Agent(
        role="Group Facilitator",
        goal="Build consensus through multiple discussion rounds and maintain team cohesion",
        backstory=backstory,
        verbose=verbose,
        memory=memory_enabled,
        system_message=system_message,
        max_iter=3,  # Allow multiple reasoning rounds
        allow_delegation=False,
        llm=get_diplomatic_gemini_llm()  # Use Google Gemini for diplomatic communication
    )
    
    return mediator


def create_mediator_with_context(previous_results: Dict[str, Any]) -> Agent:
    """
    Create a Mediator agent with full context from previous simulation runs.
    
    Args:
        previous_results: Dictionary containing results from previous iterations
        
    Returns:
        Context-aware Mediator Agent
    """
    # Handle None input gracefully
    if previous_results is None:
        previous_results = {}
    
    context = {
        "team_conflicts": previous_results.get("team_conflicts", []),
        "trust_levels": previous_results.get("trust_levels", {}),
        "successful_collaborations": previous_results.get("successful_collaborations", []),
        "communication_issues": previous_results.get("communication_issues", [])
    }
    
    return create_mediator_agent(
        memory_enabled=True,
        verbose=True,
        iteration_context=context
    )


class MediatorConfig:
    """Configuration class for Mediator agent parameters."""
    
    DEFAULT_ROLE = "Group Facilitator"
    DEFAULT_GOAL = "Build consensus through multiple discussion rounds and maintain team cohesion"
    
    # Personality traits that affect facilitation approach
    EMPATHY_LEVEL = "high"  # low, medium, high
    CONFLICT_RESOLUTION_STYLE = "collaborative"  # competitive, accommodating, collaborative, compromising
    COMMUNICATION_PREFERENCE = "inclusive"  # directive, inclusive, supportive
    
    @classmethod
    def get_personality_traits(cls) -> Dict[str, str]:
        """Get the personality configuration for the Mediator."""
        return {
            "empathy_level": cls.EMPATHY_LEVEL,
            "conflict_resolution_style": cls.CONFLICT_RESOLUTION_STYLE,
            "communication_preference": cls.COMMUNICATION_PREFERENCE,
            "decision_style": "consensus_building",
            "leadership_style": "facilitative"
        }
    
    @classmethod
    def get_relationship_tracking_config(cls) -> Dict[str, Any]:
        """Get configuration for tracking team relationships."""
        return {
            "track_trust_levels": True,
            "monitor_communication_patterns": True,
            "detect_conflict_early": True,
            "measure_team_cohesion": True,
            "adapt_facilitation_style": True
        }