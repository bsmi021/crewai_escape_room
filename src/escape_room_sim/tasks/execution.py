"""
Execution task definitions for the escape room simulation.

These tasks handle plan implementation, outcome evaluation, and adaptation.
"""

from crewai import Task
from typing import List, Dict, Any, Optional


def create_execution_tasks(
    agents: List[Any],
    action_plan: Optional[Dict[str, Any]] = None,
    iteration_context: Optional[Dict[str, Any]] = None
) -> List[Task]:
    """
    Create execution tasks for plan implementation and evaluation.
    
    Args:
        agents: List of agents [strategist, mediator, survivor]
        action_plan: The agreed-upon action plan to execute
        iteration_context: Context from previous iterations
        
    Returns:
        List of execution Task instances
    """
    
    strategist, mediator, survivor = agents
    
    # Build action plan context
    plan_context = ""
    if action_plan:
        if isinstance(action_plan, dict) and 'summary' in action_plan:
            plan_context = f"""
        
AGREED ACTION PLAN:
{action_plan['summary']}
"""
        else:
            plan_context = f"""
        
AGREED ACTION PLAN:
{str(action_plan)}
"""

    # Build iteration context  
    execution_context = ""
    if iteration_context and iteration_context.get("iteration_count", 0) > 0:
        execution_context = f"""
        
EXECUTION LEARNING FROM PREVIOUS ITERATIONS:
- Current iteration: #{iteration_context['iteration_count']}
- Previous execution outcomes: {iteration_context.get('execution_results', [])}
- Known obstacles: {iteration_context.get('known_obstacles', [])}
- Effective tactics: {iteration_context.get('effective_tactics', [])}
- Resource constraints: {iteration_context.get('resource_status', {})}
"""

    # Plan Implementation Task (Survivor)
    plan_implementation_task = Task(
        description=f"""
ESCAPE PLAN EXECUTION

Execute the agreed-upon escape plan with focus on practical implementation and real-time adaptation.

{plan_context}
{execution_context}

Your execution should involve:

1. **Action Implementation**: Carry out each planned action in sequence
2. **Resource Management**: Efficiently use available tools and materials  
3. **Progress Monitoring**: Track completion of each step and identify issues
4. **Real-time Adaptation**: Modify approach when obstacles are encountered
5. **Team Coordination**: Ensure all team members are executing their roles
6. **Outcome Assessment**: Evaluate results of each major action

Be prepared to make quick decisions when the situation doesn't match expectations.
Focus on practical solutions that maximize survival chances.

CRITICAL: At the end, clearly state whether the escape attempt was successful or failed, and why.
""",
        agent=survivor,
        expected_output="Execution report with actions taken, obstacles encountered, adaptations made, resource usage, team coordination, and final outcome assessment (SUCCESS/FAILURE with detailed reasoning)."
    )

    # Strategy Monitoring Task (Strategist)
    strategy_monitoring_task = Task(
        description=f"""
STRATEGIC EXECUTION MONITORING AND ANALYSIS

Monitor the execution from a strategic perspective and provide analytical insights.

{plan_context}
{execution_context}

Your monitoring should assess:

1. **Strategic Effectiveness**: How well the overall strategy is working
2. **Resource Efficiency**: Whether resources are being used optimally
3. **Risk Management**: How well identified risks are being handled
4. **Adaptation Quality**: Whether real-time changes improve or hurt the plan
5. **Success Probability**: Ongoing assessment of likely outcomes
6. **Lessons Learned**: Key insights for future iterations

Provide objective analysis of what's working and what isn't, with recommendations for improvement.
If the current approach is failing, suggest strategic pivots.
""",
        agent=strategist,
        expected_output="Strategic analysis of execution including effectiveness assessment, resource efficiency evaluation, risk management review, and recommendations for current or future iterations."
    )

    # Team Coordination Task (Mediator)
    team_coordination_task = Task(
        description=f"""
TEAM COORDINATION AND COMMUNICATION DURING EXECUTION

Facilitate team communication and coordination during plan execution.

{plan_context}
{execution_context}

Your coordination should focus on:

1. **Communication Flow**: Ensure clear information sharing between team members
2. **Conflict Resolution**: Address any disagreements that arise during execution
3. **Morale Management**: Maintain team confidence and motivation under pressure
4. **Role Coordination**: Help team members adapt their roles as needed
5. **Decision Support**: Facilitate quick group decisions when adaptation is needed
6. **Stress Management**: Help the team stay focused under time pressure

Keep the team working together effectively, especially when things don't go as planned.
""",
        agent=mediator,
        expected_output="Team coordination report including communication effectiveness, conflict resolution actions, morale assessment, role adaptations, and team decision-making process during execution."
    )

    return [plan_implementation_task, strategy_monitoring_task, team_coordination_task]


def create_outcome_evaluation_task(
    agent: Any,
    execution_results: Dict[str, Any],
    original_objectives: List[str]
) -> Task:
    """
    Create a task to evaluate the outcomes of execution attempts.
    
    Args:
        agent: Agent to perform the evaluation
        execution_results: Results from the execution attempt
        original_objectives: List of original objectives
        
    Returns:
        Outcome evaluation Task instance
    """
    
    return Task(
        description=f"""
EXECUTION OUTCOME EVALUATION

Evaluate the results of the escape attempt and determine next steps.

Execution Results:
{execution_results}

Original Objectives:
{', '.join(original_objectives)}

Your evaluation should determine:

1. **Success Assessment**: What objectives were achieved vs. what failed
2. **Root Cause Analysis**: Why certain actions succeeded or failed
3. **Resource Impact**: How resource usage affected outcomes
4. **Team Performance**: How well the team executed their roles
5. **Strategy Effectiveness**: Whether the overall approach was sound
6. **Learning Opportunities**: Key insights for future attempts

CRITICAL: Clearly state whether this iteration resulted in:
- COMPLETE SUCCESS (escaped successfully)
- PARTIAL SUCCESS (significant progress made)
- FAILURE (little to no progress, try different approach)

Provide specific recommendations for the next iteration if escape was not achieved.
""",
        agent=agent,
        expected_output="Comprehensive outcome evaluation with success assessment, root cause analysis, team performance review, and specific recommendations for next iteration (if needed)."
    )


def create_adaptation_task(
    agent: Any,
    current_situation: Dict[str, Any],
    unexpected_obstacles: List[str]
) -> Task:
    """
    Create a task for real-time plan adaptation.
    
    Args:
        agent: Agent to handle the adaptation
        current_situation: Current state of the situation
        unexpected_obstacles: List of obstacles not in original plan
        
    Returns:
        Plan adaptation Task instance
    """
    
    return Task(
        description=f"""
REAL-TIME PLAN ADAPTATION

Adapt the current plan to handle unexpected obstacles and changing conditions.

Current Situation:
{current_situation}

Unexpected Obstacles:
{', '.join(unexpected_obstacles)}

Your adaptation should address:

1. **Obstacle Analysis**: Understand the nature and severity of new obstacles
2. **Plan Modification**: Adjust current actions to work around obstacles
3. **Resource Reallocation**: Redistribute resources to address new challenges
4. **Timeline Adjustment**: Modify timing and sequencing as needed
5. **Risk Reassessment**: Evaluate how obstacles change overall risk profile
6. **Communication**: Inform team of necessary changes and new approach

Focus on practical solutions that can be implemented immediately.
Maintain progress toward escape while addressing new challenges.
""",
        agent=agent,
        expected_output="Adapted plan with obstacle mitigation strategies, resource reallocation, timeline adjustments, and clear instructions for team implementation."
    )


def create_final_decision_task(
    agents: List[Any],
    escape_options: Dict[str, Any],
    survival_constraints: Dict[str, Any]
) -> Task:
    """
    Create the final decision task when escape options are limited.
    
    Args:
        agents: All agents participating in the decision
        escape_options: Available escape routes and their constraints
        survival_constraints: Limitations affecting who can escape
        
    Returns:
        Final decision Task instance
    """
    
    return Task(
        description=f"""
FINAL ESCAPE DECISION

Make the critical decision about who escapes when resources are limited.

Available Escape Options:
{escape_options}

Survival Constraints:
{survival_constraints}

This final decision must address:

1. **Survival Assessment**: Who has the best chance of surviving each escape route
2. **Resource Allocation**: How to distribute limited resources for maximum survival
3. **Moral Considerations**: Ethical implications of different survival choices
4. **Team Consensus**: Building agreement on difficult survival decisions
5. **Execution Plan**: How to implement the chosen escape approach
6. **Final Outcomes**: Clear determination of who attempts which escape route

This decision will determine the final outcome of the simulation.
Each agent should contribute their perspective, but a definitive choice must be made.

CRITICAL: The decision must specify exactly who attempts to escape via which route,
and provide clear reasoning for these life-or-death choices.
""",
        agent=agents[0],  # Primary decision maker, but involves all agents
        expected_output="Final escape decision specifying who attempts each escape route, resource allocation, moral reasoning, and detailed implementation plan for the chosen survival strategy."
    )