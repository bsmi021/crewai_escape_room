"""
Planning task definitions for the escape room simulation.

These tasks handle strategy discussion, consensus building, and plan development.
"""

from crewai import Task
from typing import List, Dict, Any, Optional


def create_planning_tasks(
    agents: List[Any],
    assessment_results: Optional[Dict[str, Any]] = None,
    iteration_context: Optional[Dict[str, Any]] = None
) -> List[Task]:
    """
    Create planning tasks for strategy development and consensus building.
    
    Args:
        agents: List of agents [strategist, mediator, survivor]
        assessment_results: Results from previous assessment tasks
        iteration_context: Context from previous iterations
        
    Returns:
        List of planning Task instances
    """
    
    strategist, mediator, survivor = agents
    
    # Build assessment context
    assessment_context = ""
    if assessment_results is not None:
        if isinstance(assessment_results, dict) and 'summary' in assessment_results:
            assessment_context = f"""
        
ASSESSMENT RESULTS FROM CURRENT ITERATION:
{assessment_results['summary']}
"""
        else:
            assessment_context = f"""
        
ASSESSMENT RESULTS FROM CURRENT ITERATION:
{str(assessment_results)}
"""

    # Build iteration context
    learning_context = ""
    if iteration_context and iteration_context.get("iteration_count", 0) > 0:
        learning_context = f"""
        
LEARNING FROM PREVIOUS ITERATIONS:
- Iteration #{iteration_context['iteration_count']}
- Previous strategies tried: {iteration_context.get('previous_strategies', [])}
- What worked: {iteration_context.get('successful_approaches', [])}
- What failed: {iteration_context.get('failed_approaches', [])}
- Current team trust levels: {iteration_context.get('trust_levels', {})}
"""

    # Strategy Development Task (Strategist)
    strategy_development_task = Task(
        description=f"""
COMPREHENSIVE ESCAPE STRATEGY DEVELOPMENT

Based on the assessment results, develop a comprehensive escape strategy.

{assessment_context}
{learning_context}

Your strategy should address:

1. **Primary Escape Plan**: Main approach with step-by-step actions
2. **Resource Allocation**: How to efficiently use available materials and tools
3. **Team Coordination**: Role assignments and task distribution
4. **Risk Mitigation**: Plans for handling potential failures or obstacles
5. **Success Metrics**: How to measure progress and determine success
6. **Alternative Routes**: Backup plans if the primary approach fails

Ensure your strategy builds on successful approaches from previous iterations while avoiding known failures.
Provide clear rationale for your recommendations.
""",
        agent=strategist,
        expected_output="Comprehensive escape strategy including primary plan, resource allocation, team roles, risk mitigation, success metrics, and contingency plans with clear rationale."
    )

    # Consensus Building Task (Mediator)
    consensus_building_task = Task(
        description=f"""
TEAM CONSENSUS BUILDING AND COLLABORATION PLANNING

Facilitate discussion to build team consensus on the proposed escape strategy.

{assessment_context}
{learning_context}

Your facilitation should focus on:

1. **Strategy Review**: Present and discuss the proposed escape strategy
2. **Concern Resolution**: Address any team member doubts or objections
3. **Role Negotiation**: Ensure everyone is comfortable with their assigned role
4. **Communication Plan**: Establish how the team will coordinate during execution
5. **Conflict Prevention**: Identify potential points of friction and prevent them
6. **Morale Building**: Maintain team confidence and motivation

Ensure all team members feel heard and are committed to the agreed-upon plan.
If consensus cannot be reached, facilitate compromise and alternative approaches.
""",
        agent=mediator,
        expected_output="Team consensus on escape strategy with role agreements, communication plan, addressed concerns, and commitment confirmation from all members."
    )

    # Action Planning Task (Survivor)
    action_planning_task = Task(
        description=f"""
DETAILED ACTION PLANNING AND EXECUTION PREPARATION

Convert the agreed strategy into a detailed, executable action plan.

{assessment_context}
{learning_context}

Your action plan should specify:

1. **Step-by-Step Actions**: Detailed sequence of specific actions to take
2. **Timeline and Priorities**: When each action should occur and which are critical
3. **Resource Management**: Exactly which resources are needed for each step
4. **Quality Control**: How to verify each step was completed successfully
5. **Adaptation Triggers**: Conditions that would require plan modifications
6. **Emergency Procedures**: What to do if critical steps fail

Focus on practical execution details that maximize chances of success.
Include specific contingencies for likely failure points.
""",
        agent=survivor,
        expected_output="Detailed executable action plan with step-by-step actions, timeline, resource requirements, quality checks, adaptation triggers, and emergency procedures."
    )

    return [strategy_development_task, consensus_building_task, action_planning_task]


def create_collaborative_planning_task(
    agents: List[Any],
    planning_focus: str = "escape_strategy"
) -> Task:
    """
    Create a collaborative planning task that involves all agents.
    
    Args:
        agents: List of agents to collaborate
        planning_focus: Focus area for the planning session
        
    Returns:
        Collaborative planning Task instance
    """
    
    return Task(
        description=f"""
COLLABORATIVE PLANNING SESSION: {planning_focus.upper()}

Work together as a team to develop a coordinated approach to {planning_focus}.

This collaborative session should involve:

1. **Information Sharing**: Each agent shares their unique perspective and insights
2. **Strategy Brainstorming**: Generate multiple approaches and evaluate them together
3. **Resource Optimization**: Determine the best use of available resources
4. **Role Coordination**: Align individual roles with team objectives
5. **Risk Assessment**: Collectively evaluate risks and mitigation strategies
6. **Decision Making**: Reach consensus on the best path forward

Each agent should contribute their expertise:
- Strategist: Overall planning and risk analysis
- Mediator: Team coordination and conflict resolution  
- Survivor: Practical execution and resource management

The goal is to develop a plan that leverages each agent's strengths while maintaining team unity.
""",
        agent=agents[0],  # Primary agent, but expects collaboration with others
        expected_output=f"Collaborative plan for {planning_focus} with coordinated strategy, resource allocation, role assignments, and team consensus on implementation approach."
    )


def create_contingency_planning_task(
    agent: Any,
    primary_plan: Dict[str, Any],
    identified_risks: List[str]
) -> Task:
    """
    Create a task focused on developing contingency plans.
    
    Args:
        agent: Agent to develop contingency plans
        primary_plan: The main escape plan
        identified_risks: List of identified risk factors
        
    Returns:
        Contingency planning Task instance
    """
    
    return Task(
        description=f"""
CONTINGENCY PLANNING AND RISK MITIGATION

Develop comprehensive backup plans for the primary escape strategy.

Primary Plan Overview:
{primary_plan}

Identified Risk Factors:
{', '.join(identified_risks)}

Your contingency planning should address:

1. **Failure Point Analysis**: Identify where the primary plan is most likely to fail
2. **Alternative Approaches**: Develop backup strategies for each failure point
3. **Resource Reallocation**: Plan how to redistribute resources if original plan fails
4. **Decision Trees**: Create clear decision points for when to switch to backup plans
5. **Recovery Procedures**: Steps to take if partial failure occurs
6. **Emergency Protocols**: Last-resort actions if all planned approaches fail

Ensure contingency plans are realistic and can be implemented quickly under stress.
""",
        agent=agent,
        expected_output="Comprehensive contingency plans with alternative strategies, decision trees, resource reallocation procedures, and emergency protocols for identified failure points."
    )