"""
Assessment task definitions for the escape room simulation.

These tasks handle room examination, situation analysis, and progress evaluation.
"""

from crewai import Task
from typing import List, Dict, Any, Optional


def create_assessment_tasks(
    agents: List[Any],
    iteration_context: Optional[Dict[str, Any]] = None
) -> List[Task]:
    """
    Create assessment tasks for room examination and situation analysis.
    
    Args:
        agents: List of agents [strategist, mediator, survivor]
        iteration_context: Context from previous iterations
        
    Returns:
        List of assessment Task instances
    """
    
    strategist, mediator, survivor = agents
    
    # Build context-aware task descriptions
    previous_attempts = ""
    if iteration_context and iteration_context.get("iteration_count", 0) > 0:
        previous_attempts = f"""
        
PREVIOUS ITERATION CONTEXT:
This is iteration {iteration_context['iteration_count']}. You have access to:
- Previous failed strategies: {iteration_context.get('failed_strategies', [])}
- Current game state: {iteration_context.get('game_state', {})}
- Time remaining: {iteration_context.get('time_remaining', 60)} minutes

Use this knowledge to avoid repeating failed approaches and build on what you've learned."""

    # Room Assessment Task (Strategist)
    room_assessment_task = Task(
        description=f"""
ROOM SITUATION ANALYSIS

Examine the escape room and identify all possible exit routes, resources, and constraints.
Your analysis should include:

1. **Exit Routes**: Identify all possible ways to escape and their requirements
2. **Available Resources**: Catalog all items, tools, and materials in the room
3. **Puzzles and Obstacles**: Analyze what needs to be solved or overcome
4. **Team Capabilities**: Assess what each team member can contribute
5. **Risk Assessment**: Evaluate dangers and time constraints

{previous_attempts}

Provide a comprehensive strategic assessment that will guide the team's escape planning.
""",
        agent=strategist,
        expected_output="Detailed strategic analysis including exit routes, resources, puzzles, team capabilities, and risk assessment with specific recommendations."
    )

    # Team Status Assessment Task (Mediator)
    team_assessment_task = Task(
        description=f"""
TEAM DYNAMICS AND COORDINATION ASSESSMENT

Evaluate the current team dynamics and establish effective communication patterns.
Your assessment should cover:

1. **Team Morale**: Assess stress levels and confidence of all team members
2. **Communication Patterns**: Identify how well the team is working together
3. **Conflict Resolution**: Address any disagreements or tensions
4. **Role Clarity**: Ensure everyone understands their responsibilities
5. **Collaboration Opportunities**: Find ways to maximize team synergy

{previous_attempts}

Focus on building trust and ensuring all voices are heard in the escape planning process.
""",
        agent=mediator,
        expected_output="Team dynamics assessment with recommendations for improving collaboration, communication patterns, and conflict resolution strategies."
    )

    # Practical Assessment Task (Survivor)  
    execution_assessment_task = Task(
        description=f"""
PRACTICAL EXECUTION ASSESSMENT

Evaluate the feasibility of proposed escape plans from a practical execution perspective.
Your assessment should address:

1. **Resource Requirements**: What materials and tools are actually needed
2. **Time Constraints**: How long each proposed action will realistically take
3. **Physical Limitations**: What can actually be accomplished given constraints
4. **Risk vs Reward**: Practical analysis of success probability for each approach
5. **Contingency Planning**: Backup options if primary plans fail

{previous_attempts}

Provide honest, realistic assessments that prioritize actionable solutions over idealistic plans.
""",
        agent=survivor,
        expected_output="Practical execution assessment with realistic timelines, resource requirements, success probabilities, and contingency recommendations."
    )

    return [room_assessment_task, team_assessment_task, execution_assessment_task]


def create_progress_evaluation_task(
    agent: Any,
    current_progress: Dict[str, Any]
) -> Task:
    """
    Create a task to evaluate progress made in the current iteration.
    
    Args:
        agent: Agent to assign the evaluation task to
        current_progress: Current state of progress
        
    Returns:
        Progress evaluation Task instance
    """
    
    return Task(
        description=f"""
PROGRESS EVALUATION

Evaluate the progress made in this iteration and determine next steps.

Current Progress Summary:
- Actions attempted: {current_progress.get('actions_attempted', [])}
- Resources discovered: {current_progress.get('resources_found', [])}
- Puzzles solved: {current_progress.get('puzzles_solved', [])}
- Obstacles encountered: {current_progress.get('obstacles', [])}

Your evaluation should determine:
1. What progress has been made toward escape
2. What new information has been discovered
3. What approaches have proven effective or ineffective
4. What should be the focus for the next iteration
5. Whether the current strategy should be continued or changed

Provide clear recommendations for moving forward.
""",
        agent=agent,
        expected_output="Progress evaluation with specific recommendations for next iteration including strategy adjustments, resource utilization, and priority actions."
    )


def create_situation_analysis_task(
    agent: Any,
    room_state: Dict[str, Any],
    team_state: Dict[str, Any]
) -> Task:
    """
    Create a comprehensive situation analysis task.
    
    Args:
        agent: Agent to perform the analysis
        room_state: Current state of the escape room
        team_state: Current state of the team
        
    Returns:
        Situation analysis Task instance
    """
    
    return Task(
        description=f"""
COMPREHENSIVE SITUATION ANALYSIS

Analyze the current situation combining room state and team dynamics.

Room State:
{room_state}

Team State:
{team_state}

Provide analysis covering:
1. Current advantages and disadvantages
2. Immediate opportunities and threats
3. Resource availability and constraints
4. Team readiness and capability assessment
5. Strategic recommendations for next actions

Focus on identifying the most promising path forward given current conditions.
""",
        agent=agent,
        expected_output="Comprehensive situation analysis with prioritized recommendations, opportunity assessment, and strategic guidance for team coordination."
    )