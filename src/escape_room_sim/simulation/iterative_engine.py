"""
Iterative Simulation Engine for CrewAI Escape Room.

This engine manages multiple rounds of agent collaboration, enabling learning
from previous attempts and dynamic task generation based on iteration history.
Implements best practices for CrewAI 0.150.0 with proper memory management.
"""

import json
import os
from datetime import datetime
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass, asdict
from crewai import Agent, Task, Crew, Process

from ..agents.strategist import create_strategist_agent, create_strategist_with_context
from ..agents.mediator import create_mediator_agent, create_mediator_with_context
from ..agents.survivor import create_survivor_agent, create_survivor_with_context
from ..memory.persistent_memory import IterativeMemoryManager
from ..room.escape_room_state import EscapeRoomState


def get_strategist_context_for_iteration(
    iteration_num: int, 
    previous_failures: Optional[List[str]], 
    current_resources: Optional[Dict[str, Any]]
) -> str:
    """
    Generate iteration-specific context for the Strategist agent.
    
    Args:
        iteration_num: Current iteration number
        previous_failures: List of previously failed strategies
        current_resources: Dictionary of current available resources
        
    Returns:
        Formatted context string for strategic analysis
    """
    # Handle None parameters gracefully
    if previous_failures is None:
        previous_failures = []
    if current_resources is None:
        current_resources = {}
    
    context_parts = [
        f"STRATEGIST CONTEXT - ITERATION {iteration_num}",
        "=" * 50,
        "",
        "STRATEGIC ANALYSIS FOCUS:",
        f"• This is iteration {iteration_num} of the escape room challenge",
        f"• You are the analytical problem-solver of the team",
        f"• Your role is to provide systematic, logical analysis",
        ""
    ]
    
    # Add previous failures analysis
    if previous_failures:
        context_parts.extend([
            "PREVIOUS STRATEGY FAILURES TO LEARN FROM:",
            ""
        ])
        for i, failure in enumerate(previous_failures[:5], 1):  # Limit to 5 most recent
            context_parts.append(f"{i}. {failure}")
        context_parts.extend([
            "",
            "KEY LEARNING POINTS:",
            "• Analyze why these strategies failed",
            "• Identify patterns in unsuccessful approaches", 
            "• Avoid repeating the same mistakes",
            ""
        ])
    else:
        context_parts.extend([
            "FIRST ITERATION ANALYSIS:",
            "• No previous failures to learn from",
            "• Focus on comprehensive initial assessment",
            "• Establish baseline strategic approach",
            ""
        ])
    
    # Add current resources information
    if current_resources:
        context_parts.extend([
            "CURRENT RESOURCE STATUS:",
            ""
        ])
        for resource, value in current_resources.items():
            if isinstance(value, (list, tuple)):
                context_parts.append(f"• {resource.title()}: {', '.join(map(str, value))}")
            else:
                context_parts.append(f"• {resource.title()}: {value}")
        context_parts.extend([
            "",
            "RESOURCE OPTIMIZATION:",
            "• Consider how to best utilize available resources",
            "• Identify resource constraints and bottlenecks",
            "• Plan resource allocation for maximum effectiveness",
            ""
        ])
    else:
        context_parts.extend([
            "RESOURCE STATUS: Limited information available",
            "• Conduct thorough resource assessment",
            "• Identify what resources are needed",
            "• Plan resource discovery and acquisition",
            ""
        ])
    
    # Add strategic priorities
    context_parts.extend([
        "STRATEGIC PRIORITIES FOR THIS ITERATION:",
        "• Conduct systematic analysis of current situation",
        "• Develop multiple strategic options with risk assessments",
        "• Provide clear recommendations with probability estimates",
        "• Consider both short-term tactics and long-term strategy",
        "• Ensure all analysis is actionable and specific",
        "",
        "EXPECTED DELIVERABLES:",
        "• Comprehensive situation assessment",
        "• Ranked list of strategic options",
        "• Risk analysis for each proposed approach",
        "• Clear recommendation for team action",
        "",
        "Remember: Your analytical approach is crucial for team success.",
        "Be thorough but decisive. Time is limited."
    ])
    
    return "\n".join(context_parts)


def get_mediator_context_for_iteration(
    iteration_num: int,
    relationship_tracker: Optional[Any],
    team_stress_level: float,
    previous_conflicts: Optional[List[str]]
) -> str:
    """
    Generate iteration-specific context for the Mediator agent.
    
    Args:
        iteration_num: Current iteration number
        relationship_tracker: RelationshipTracker instance (can be None)
        team_stress_level: Current team stress level (0.0 to 1.0)
        previous_conflicts: List of previous team conflicts
        
    Returns:
        Formatted context string for mediation and team dynamics
    """
    # Handle None parameters gracefully
    if previous_conflicts is None:
        previous_conflicts = []
    
    context_parts = [
        f"MEDIATOR CONTEXT - ITERATION {iteration_num}",
        "=" * 50,
        "",
        "TEAM FACILITATION FOCUS:",
        f"• This is iteration {iteration_num} of the escape room challenge",
        f"• You are the diplomatic coordinator and team facilitator",
        f"• Your role is to maintain team cohesion and resolve conflicts",
        f"• Current team stress level: {team_stress_level:.2f} (0.0=calm, 1.0=critical)",
        ""
    ]
    
    # Add stress level analysis
    if team_stress_level >= 0.8:
        context_parts.extend([
            "⚠️  CRITICAL STRESS LEVEL ALERT:",
            "• Team is under extreme pressure - immediate intervention needed",
            "• Focus on calming techniques and stress reduction",
            "• Prevent team breakdown and maintain communication",
            "• Consider shorter-term, less risky approaches",
            ""
        ])
    elif team_stress_level >= 0.6:
        context_parts.extend([
            "⚠️  HIGH STRESS LEVEL WARNING:",
            "• Team stress is elevated - monitor carefully",
            "• Encourage breaks and positive reinforcement",
            "• Watch for signs of conflict or frustration",
            "• Maintain optimistic but realistic outlook",
            ""
        ])
    elif team_stress_level >= 0.4:
        context_parts.extend([
            "MODERATE STRESS LEVEL:",
            "• Team is experiencing normal challenge stress",
            "• Continue supportive facilitation",
            "• Encourage open communication",
            "• Build on team strengths and successes",
            ""
        ])
    else:
        context_parts.extend([
            "LOW STRESS LEVEL:",
            "• Team is relatively calm and focused",
            "• Maintain positive team dynamics",
            "• Encourage creative problem-solving",
            "• Foster collaborative decision-making",
            ""
        ])
    
    # Add relationship tracker information
    if relationship_tracker is not None:
        try:
            # Try to get relationship summary
            if hasattr(relationship_tracker, 'get_summary'):
                relationship_summary = relationship_tracker.get_summary()
                context_parts.extend([
                    "CURRENT TEAM RELATIONSHIPS:",
                    f"• {relationship_summary}",
                    ""
                ])
            
            # Try to get team cohesion
            if hasattr(relationship_tracker, 'get_team_cohesion'):
                try:
                    cohesion = relationship_tracker.get_team_cohesion()
                    context_parts.extend([
                        f"TEAM COHESION LEVEL: {cohesion:.2f}",
                        "• Use this information to guide team interactions",
                        "• Focus on strengthening weak relationships",
                        ""
                    ])
                except:
                    # Handle case where get_team_cohesion needs parameters
                    context_parts.extend([
                        "TEAM COHESION: Available for analysis",
                        "• Monitor relationship dynamics closely",
                        ""
                    ])
        except Exception:
            context_parts.extend([
                "RELATIONSHIP TRACKING: Available but limited data",
                "• Focus on observing and improving team dynamics",
                ""
            ])
    else:
        context_parts.extend([
            "RELATIONSHIP TRACKING: Not available",
            "• Rely on direct observation of team interactions",
            "• Pay attention to verbal and non-verbal cues",
            "• Build rapport and trust through active listening",
            ""
        ])
    
    # Add previous conflicts analysis
    if previous_conflicts:
        context_parts.extend([
            "PREVIOUS TEAM CONFLICTS TO ADDRESS:",
            ""
        ])
        for i, conflict in enumerate(previous_conflicts[:5], 1):  # Limit to 5 most recent
            context_parts.append(f"{i}. {conflict}")
        context_parts.extend([
            "",
            "CONFLICT RESOLUTION PRIORITIES:",
            "• Address unresolved tensions from previous conflicts",
            "• Prevent similar conflicts from recurring",
            "• Rebuild trust where it may have been damaged",
            "• Focus on shared goals and common ground",
            ""
        ])
    else:
        context_parts.extend([
            "CONFLICT STATUS: No major previous conflicts recorded",
            "• Maintain positive team atmosphere",
            "• Prevent conflicts before they escalate",
            "• Foster collaborative problem-solving",
            ""
        ])
    
    # Add mediation priorities
    context_parts.extend([
        "MEDIATION PRIORITIES FOR THIS ITERATION:",
        "• Facilitate productive discussion between all team members",
        "• Ensure every voice is heard and valued",
        "• Guide the team toward consensus on strategy",
        "• Monitor emotional state and intervene if needed",
        "• Maintain focus on shared survival goals",
        "• Bridge differences in opinion or approach",
        "",
        "TEAM DYNAMICS MANAGEMENT:",
        "• Encourage active participation from quieter members",
        "• Manage dominant personalities constructively",
        "• Translate between different communication styles",
        "• Build on individual strengths for team benefit",
        "",
        "EXPECTED DELIVERABLES:",
        "• Clear team consensus on chosen approach",
        "• Commitment from all members to the agreed strategy",
        "• Maintained or improved team relationships",
        "• Effective conflict resolution if issues arise",
        "",
        "Remember: Your diplomatic skills are essential for team success.",
        "Unity and collaboration will determine survival."
    ])
    
    return "\n".join(context_parts)


def get_survivor_context_for_iteration(
    iteration_num: int,
    survival_memory: Optional[Any],
    current_threat_level: float,
    resource_status: Optional[Dict[str, Any]]
) -> str:
    """
    Generate iteration-specific context for the Survivor agent.
    
    Args:
        iteration_num: Current iteration number
        survival_memory: SurvivalMemoryBank instance (can be None)
        current_threat_level: Current threat assessment (0.0 to 1.0)
        resource_status: Dictionary of current resource status
        
    Returns:
        Formatted context string for survival decision-making
    """
    # Handle None parameters gracefully
    if resource_status is None:
        resource_status = {}
    
    context_parts = [
        f"SURVIVOR CONTEXT - ITERATION {iteration_num}",
        "=" * 50,
        "",
        "SURVIVAL EXECUTION FOCUS:",
        f"• This is iteration {iteration_num} of the escape room challenge",
        f"• You are the pragmatic decision-maker and action executor",
        f"• Your role is to make tough survival decisions and execute plans",
        f"• Current threat level: {current_threat_level:.2f} (0.0=safe, 1.0=critical)",
        ""
    ]
    
    # Add threat level analysis
    if current_threat_level >= 0.9:
        context_parts.extend([
            "🚨 CRITICAL THREAT LEVEL - IMMEDIATE ACTION REQUIRED:",
            "• Situation is life-threatening - act decisively now",
            "• Override group consensus if necessary for survival",
            "• Focus on immediate escape options, not perfect solutions",
            "• Time for discussion has passed - execution is everything",
            "• Prepare for worst-case scenarios and emergency measures",
            ""
        ])
    elif current_threat_level >= 0.7:
        context_parts.extend([
            "⚠️  HIGH THREAT LEVEL - URGENT ACTION NEEDED:",
            "• Danger is significant and increasing",
            "• Prioritize speed over perfection in execution",
            "• Be ready to make unilateral survival decisions",
            "• Focus on proven strategies over experimental approaches",
            "• Monitor situation closely for rapid deterioration",
            ""
        ])
    elif current_threat_level >= 0.5:
        context_parts.extend([
            "⚠️  MODERATE THREAT LEVEL - CAREFUL EXECUTION:",
            "• Situation requires caution but allows for planning",
            "• Balance team input with decisive action",
            "• Prepare contingency plans for threat escalation",
            "• Focus on sustainable approaches with backup options",
            ""
        ])
    elif current_threat_level >= 0.3:
        context_parts.extend([
            "LOW-MODERATE THREAT LEVEL - STRATEGIC EXECUTION:",
            "• Situation is manageable with proper planning",
            "• Collaborate with team while maintaining readiness",
            "• Use this time to prepare for potential threats",
            "• Focus on thorough execution of chosen strategy",
            ""
        ])
    else:
        context_parts.extend([
            "LOW THREAT LEVEL - METHODICAL APPROACH:",
            "• Situation allows for careful, collaborative execution",
            "• Take time to do things right the first time",
            "• Build team confidence through successful actions",
            "• Prepare for future challenges while addressing current ones",
            ""
        ])
    
    # Add survival memory information
    if survival_memory is not None:
        try:
            # Try to get relevant experiences
            if hasattr(survival_memory, 'get_relevant_experiences'):
                try:
                    experiences = survival_memory.get_relevant_experiences(5)
                    if experiences and experiences.strip():
                        context_parts.extend([
                            "SURVIVAL MEMORY - RELEVANT PAST EXPERIENCES:",
                            f"• {experiences}",
                            "",
                            "LESSONS FROM EXPERIENCE:",
                            "• Apply successful strategies from similar situations",
                            "• Avoid repeating actions that led to close calls",
                            "• Trust your survival instincts based on past learning",
                            ""
                        ])
                    else:
                        context_parts.extend([
                            "SURVIVAL MEMORY: Limited past experiences available",
                            "• Rely on basic survival principles",
                            "• Document this experience for future learning",
                            ""
                        ])
                except:
                    context_parts.extend([
                        "SURVIVAL MEMORY: Available but access limited",
                        "• Use general survival principles",
                        ""
                    ])
            
            # Try to get survival probability
            if hasattr(survival_memory, 'calculate_survival_probability'):
                try:
                    # This might need parameters, so we'll handle gracefully
                    context_parts.extend([
                        "SURVIVAL PROBABILITY ANALYSIS: Available for decision-making",
                        "• Use historical data to assess action success rates",
                        ""
                    ])
                except:
                    pass
        except Exception:
            context_parts.extend([
                "SURVIVAL MEMORY: Available but limited functionality",
                "• Focus on immediate survival priorities",
                ""
            ])
    else:
        context_parts.extend([
            "SURVIVAL MEMORY: Not available",
            "• Rely on basic survival instincts and logic",
            "• Make decisions based on immediate situation assessment",
            "• Document experiences for potential future use",
            ""
        ])
    
    # Add resource status analysis
    if resource_status:
        context_parts.extend([
            "CURRENT RESOURCE STATUS:",
            ""
        ])
        for resource, status in resource_status.items():
            if isinstance(status, (list, tuple)):
                context_parts.append(f"• {resource}: {', '.join(map(str, status))}")
            elif isinstance(status, dict):
                context_parts.append(f"• {resource}: {status}")
            else:
                context_parts.append(f"• {resource}: {status}")
        
        context_parts.extend([
            "",
            "RESOURCE MANAGEMENT PRIORITIES:",
            "• Conserve critical resources for essential actions",
            "• Identify resource bottlenecks that could prevent escape",
            "• Plan resource usage for maximum survival benefit",
            "• Be prepared to sacrifice non-essential resources",
            ""
        ])
    else:
        context_parts.extend([
            "RESOURCE STATUS: Limited information available",
            "• Conduct immediate resource assessment",
            "• Identify what resources are available and needed",
            "• Plan resource acquisition and conservation",
            ""
        ])
    
    # Add survival execution priorities
    context_parts.extend([
        "SURVIVAL EXECUTION PRIORITIES FOR THIS ITERATION:",
        "• Execute the agreed strategy with precision and urgency",
        "• Monitor progress continuously and adapt as needed",
        "• Make critical decisions when team consensus isn't possible",
        "• Prioritize actions that maximize survival probability",
        "• Be prepared to override plans if survival is at stake",
        "• Focus on practical, actionable steps over theoretical solutions",
        "",
        "DECISION-MAKING FRAMEWORK:",
        "• Speed vs. Safety: Balance based on current threat level",
        "• Individual vs. Group: Prioritize group survival when possible",
        "• Risk vs. Reward: Take calculated risks for significant gains",
        "• Known vs. Unknown: Prefer proven approaches under high threat",
        "",
        "EXPECTED DELIVERABLES:",
        "• Clear execution of the chosen strategy",
        "• Real-time assessment of progress and obstacles",
        "• Critical survival decisions when needed",
        "• Determination of success/failure and next steps",
        "",
        "Remember: Your survival instincts and decisive action are crucial.",
        "When others hesitate, you must act. Lives depend on your choices."
    ])
    
    return "\n".join(context_parts)


@dataclass
class IterationResult:
    """Represents the result of a single simulation iteration."""
    iteration_number: int
    timestamp: str
    crew_output: str
    game_state_snapshot: Dict[str, Any]
    agents_consensus: bool
    solution_found: bool
    lessons_learned: List[str]
    next_iteration_needed: bool


@dataclass
class SimulationConfig:
    """Configuration for the iterative simulation."""
    max_iterations: int = 15
    max_time_per_iteration: int = 300  # 5 minutes
    consensus_threshold: float = 0.8
    enable_memory: bool = True
    verbose_output: bool = True
    save_intermediate_results: bool = True
    deadlock_detection_enabled: bool = True
    min_progress_threshold: float = 0.1


class IterativeEscapeSimulation:
    """
    Main iterative simulation engine that manages multi-round agent collaboration.
    
    This engine implements the core iterative workflow:
    1. Initialize agents with memory enabled
    2. Create dynamic tasks based on iteration history
    3. Execute crew.kickoff() for each iteration
    4. Learn from results and update memory
    5. Check stopping conditions
    6. Generate comprehensive final report
    """
    
    def __init__(self, config: SimulationConfig = None, data_dir: str = None):
        """
        Initialize the iterative simulation engine.
        
        Args:
            config: Simulation configuration parameters
            data_dir: Directory to save simulation data (defaults to ./data)
        """
        self.config = config or SimulationConfig()
        self.data_dir = data_dir or "data"
        self.current_iteration = 0
        self.solution_found = False
        self.simulation_active = True
        
        # Create data directory if it doesn't exist
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Initialize core components
        self.game_state = EscapeRoomState()
        self.memory_manager = IterativeMemoryManager(self.data_dir)
        self.relationship_tracker = RelationshipTracker()
        self.survival_memory = SurvivalMemoryBank()
        
        # Track simulation results
        self.iteration_results: List[IterationResult] = []
        self.conversation_history: List[Dict[str, Any]] = []
        
        # Initialize agents with memory
        self.agents = self._initialize_agents()
        
        print(f"🚀 Iterative Escape Room Simulation initialized")
        print(f"📊 Configuration: {self.config.max_iterations} max iterations, memory {'enabled' if self.config.enable_memory else 'disabled'}")
    
    def _initialize_agents(self) -> List[Agent]:
        """Initialize all agents with memory and enhanced capabilities."""
        agents = []
        
        # Strategist - Analytical problem solver
        strategist = create_strategist_agent(
            memory_enabled=self.config.enable_memory,
            verbose=self.config.verbose_output
        )
        agents.append(strategist)
        
        # Mediator - Group facilitator
        mediator = create_mediator_agent(
            memory_enabled=self.config.enable_memory,
            verbose=self.config.verbose_output
        )
        agents.append(mediator)
        
        # Survivor - Pragmatic decision maker
        survivor = create_survivor_agent(
            memory_enabled=self.config.enable_memory,
            verbose=self.config.verbose_output
        )
        agents.append(survivor)
        
        return agents
    
    def create_iteration_tasks(self) -> List[Task]:
        """
        Create tasks for the current iteration based on history and game state.
        
        Returns:
            List of Task objects configured for current iteration
        """
        tasks = []
        
        # Get context for each agent based on their specializations
        strategist_context = get_strategist_context_for_iteration(
            iteration_num=self.current_iteration + 1,
            previous_failures=self.memory_manager.get_failed_strategies(),
            current_resources=self.game_state.get_resource_summary()
        )
        
        mediator_context = get_mediator_context_for_iteration(
            iteration_num=self.current_iteration + 1,
            relationship_tracker=self.relationship_tracker,
            team_stress_level=self.game_state.get_stress_level(),
            previous_conflicts=self.memory_manager.get_interpersonal_conflicts()
        )
        
        survivor_context = get_survivor_context_for_iteration(
            iteration_num=self.current_iteration + 1,
            survival_memory=self.survival_memory,
            current_threat_level=self.game_state.get_threat_level(),
            resource_status=self.game_state.get_resource_status()
        )
        
        # Task 1: Strategic Analysis
        analysis_task = Task(
            description=f"""
            {strategist_context}
            
            STRATEGIC ANALYSIS TASK - ITERATION {self.current_iteration + 1}:
            
            Analyze the current escape room situation using your systematic approach:
            
            1. SITUATION ASSESSMENT:
               - Review current room state: {json.dumps(self.game_state.to_dict(), indent=2)}
               - Identify what has changed since last iteration
               - Assess new information or opportunities discovered
            
            2. LEARN FROM PREVIOUS ATTEMPTS:
               - What strategies failed and why?
               - What partial successes can be built upon?
               - What assumptions proved incorrect?
            
            3. STRATEGIC OPTIONS ANALYSIS:
               - List all possible approaches with probability assessments
               - Identify resource requirements for each option
               - Consider interdependencies and sequence requirements
            
            4. OPTIMIZATION RECOMMENDATIONS:
               - Propose the highest-probability strategy
               - Identify critical success factors
               - Plan for likely failure modes
            
            Your analysis will inform the group discussion. Be thorough but decisive.
            Reference specific lessons from previous iterations.
            """,
            agent=self.agents[0],  # Strategist
            expected_output="Comprehensive strategic analysis with ranked recommendations and risk assessments"
        )
        
        # Task 2: Group Facilitation and Consensus Building
        facilitation_task = Task(
            description=f"""
            {mediator_context}
            
            GROUP FACILITATION TASK - ITERATION {self.current_iteration + 1}:
            
            Facilitate productive collaboration between all team members:
            
            1. TEAM DYNAMICS MANAGEMENT:
               - Address any tensions or conflicts from previous iterations
               - Ensure all perspectives are heard and valued
               - Monitor stress levels and emotional state of team
            
            2. COLLABORATIVE DECISION-MAKING:
               - Guide discussion of the strategic analysis
               - Help team weigh different options and trade-offs
               - Build consensus while respecting individual concerns
            
            3. RELATIONSHIP MAINTENANCE:
               - Strengthen trust and communication
               - Address any emerging alliances or divisions
               - Keep focus on shared survival goals
            
            4. CONSENSUS BUILDING:
               - Work toward agreement on next approach
               - Ensure everyone commits to the chosen strategy
               - Plan for coordinated execution
            
            Your facilitation is crucial for team unity. Balance different perspectives
            while moving toward decisive action.
            """,
            agent=self.agents[1],  # Mediator
            expected_output="Group consensus on strategy with clear roles and commitment from all team members"
        )
        
        # Task 3: Execution and Survival Assessment
        execution_task = Task(
            description=f"""
            {survivor_context}
            
            EXECUTION & SURVIVAL TASK - ITERATION {self.current_iteration + 1}:
            
            Execute the agreed strategy and make critical survival assessments:
            
            1. TACTICAL EXECUTION:
               - Implement the chosen approach with urgency and precision
               - Adapt to unexpected obstacles or opportunities
               - Maintain focus on practical, actionable steps
            
            2. REAL-TIME ASSESSMENT:
               - Monitor progress and effectiveness continuously
               - Identify when to persist vs when to pivot
               - Assess resource consumption and time constraints
            
            3. SURVIVAL DECISION-MAKING:
               - Make tough calls when group consensus isn't possible
               - Prioritize actions that maximize survival probability
               - Prepare contingency plans for emergencies
            
            4. OUTCOME EVALUATION:
               - Determine if current approach solves the escape problem
               - Assess what worked, what didn't, and why
               - If solution found, clearly state "SOLUTION FOUND: [description]"
               - If not solved, identify key obstacles for next iteration
            
            Your execution and assessment will determine if we continue or conclude.
            Make decisions others won't. Time is critical.
            """,
            agent=self.agents[2],  # Survivor
            expected_output="Execution results with clear determination of success/failure and next steps needed"
        )
        
        tasks = [analysis_task, facilitation_task, execution_task]
        return tasks
    
    def run_single_iteration(self) -> IterationResult:
        """
        Execute a single iteration of the simulation.
        
        Returns:
            IterationResult with comprehensive iteration data
        """
        iteration_start_time = datetime.now()
        
        print(f"\n{'='*60}")
        print(f"🔄 ITERATION {self.current_iteration + 1} - Starting Analysis")
        print(f"{'='*60}")
        print(f"⏱️  Time: {iteration_start_time.strftime('%H:%M:%S')}")
        print(f"🎯 Goal: Find escape solution through collaborative problem-solving")
        print(f"⚡ Game State: {self.game_state.get_status_summary()}")
        
        # Create tasks for this iteration
        tasks = self.create_iteration_tasks()
        
        # Create crew for this iteration
        crew = Crew(
            agents=self.agents,
            tasks=tasks,
            process=Process.sequential,
            verbose=2 if self.config.verbose_output else 0,
            memory=self.config.enable_memory,
            max_execution_time=self.config.max_time_per_iteration
        )
        
        try:
            # Execute the crew
            print(f"🚀 Executing crew with {len(self.agents)} agents and {len(tasks)} tasks...")
            crew_result = crew.kickoff()
            
            # Process results
            crew_output = str(crew_result)
            solution_found = self._check_solution_found(crew_output)
            consensus_reached = self._assess_consensus(crew_output)
            lessons_learned = self._extract_lessons_learned(crew_output)
            
            # Update game state based on results
            self._update_game_state_from_results(crew_output)
            
            # Create iteration result
            result = IterationResult(
                iteration_number=self.current_iteration + 1,
                timestamp=iteration_start_time.isoformat(),
                crew_output=crew_output,
                game_state_snapshot=self.game_state.to_dict(),
                agents_consensus=consensus_reached,
                solution_found=solution_found,
                lessons_learned=lessons_learned,
                next_iteration_needed=not solution_found and self.current_iteration < self.config.max_iterations - 1
            )
            
            # Update solution status
            if solution_found:
                self.solution_found = True
                print(f"🎉 SOLUTION FOUND IN ITERATION {self.current_iteration + 1}!")
            
            # Store results
            self.iteration_results.append(result)
            
            # Update memory systems
            self._update_memory_systems(result)
            
            # Save intermediate results if configured
            if self.config.save_intermediate_results:
                self._save_iteration_result(result)
            
            iteration_duration = (datetime.now() - iteration_start_time).total_seconds()
            print(f"✅ Iteration {self.current_iteration + 1} completed in {iteration_duration:.1f} seconds")
            
            return result
            
        except Exception as e:
            print(f"❌ Error in iteration {self.current_iteration + 1}: {str(e)}")
            # Create error result
            error_result = IterationResult(
                iteration_number=self.current_iteration + 1,
                timestamp=iteration_start_time.isoformat(),
                crew_output=f"ERROR: {str(e)}",
                game_state_snapshot=self.game_state.to_dict(),
                agents_consensus=False,
                solution_found=False,
                lessons_learned=[f"Technical error occurred: {str(e)}"],
                next_iteration_needed=True
            )
            self.iteration_results.append(error_result)
            return error_result
    
    def _check_solution_found(self, crew_output: str) -> bool:
        """Check if the crew output indicates a solution was found."""
        solution_indicators = [
            "SOLUTION FOUND",
            "escape solution identified",
            "successful escape plan",
            "viable exit strategy",
            "escape route confirmed"
        ]
        
        crew_output_lower = crew_output.lower()
        return any(indicator.lower() in crew_output_lower for indicator in solution_indicators)
    
    def _assess_consensus(self, crew_output: str) -> bool:
        """Assess whether agents reached consensus based on output content."""
        consensus_indicators = [
            "all agree",
            "consensus reached",
            "team agreement",
            "unanimous decision",
            "agreed upon"
        ]
        
        disagreement_indicators = [
            "disagree",
            "conflict",
            "cannot agree",
            "different opinions",
            "opposing views"
        ]
        
        crew_output_lower = crew_output.lower()
        
        has_consensus = any(indicator in crew_output_lower for indicator in consensus_indicators)
        has_disagreement = any(indicator in crew_output_lower for indicator in disagreement_indicators)
        
        return has_consensus and not has_disagreement
    
    def _extract_lessons_learned(self, crew_output: str) -> List[str]:
        """Extract key lessons learned from the iteration output."""
        lessons = []
        
        # Look for explicit lesson patterns
        lesson_patterns = [
            "learned that",
            "discovered that",
            "realized that",
            "found that",
            "lesson:"
        ]
        
        lines = crew_output.split('\n')
        for line in lines:
            line_lower = line.lower().strip()
            if any(pattern in line_lower for pattern in lesson_patterns):
                lessons.append(line.strip())
        
        # If no explicit lessons, extract key insights
        if not lessons:
            key_insights = [
                line.strip() for line in lines 
                if len(line.strip()) > 20 and 
                any(word in line.lower() for word in ['strategy', 'approach', 'failed', 'worked', 'discovered'])
            ][:3]  # Limit to 3 key insights
            lessons.extend(key_insights)
        
        return lessons
    
    def _update_game_state_from_results(self, crew_output: str):
        """Update game state based on crew execution results."""
        # Time consumption
        self.game_state.consume_time(5)  # Each iteration consumes 5 minutes
        
        # Resource discovery/consumption based on output content
        output_lower = crew_output.lower()
        
        if 'key' in output_lower and 'found' in output_lower:
            self.game_state.add_resource('key')
        
        if 'tool' in output_lower and ('used' in output_lower or 'broke' in output_lower):
            self.game_state.consume_resource('tool')
        
        if 'puzzle' in output_lower and 'solved' in output_lower:
            self.game_state.solve_puzzle()
        
        # Stress level changes based on progress
        if self.solution_found:
            self.game_state.reduce_stress(0.3)
        elif 'progress' in output_lower:
            self.game_state.reduce_stress(0.1)
        else:
            self.game_state.increase_stress(0.1)
    
    def _update_memory_systems(self, result: IterationResult):
        """Update all memory systems with iteration results."""
        # Update memory manager
        self.memory_manager.record_iteration(
            iteration_num=result.iteration_number,
            strategies_tried=self._extract_strategies_from_output(result.crew_output),
            outcomes=result.lessons_learned,
            consensus_reached=result.agents_consensus
        )
        
        # Update relationship tracker based on collaboration patterns
        if result.agents_consensus:
            self.relationship_tracker.record_successful_collaboration(
                agents=["Strategist", "Mediator", "Survivor"],
                strategy="Collaborative problem-solving",
                outcome="Reached consensus on approach"
            )
        
        # Update survival memory with relevant experiences
        if "close call" in result.crew_output.lower() or "dangerous" in result.crew_output.lower():
            self.survival_memory.record_close_call(
                situation=f"Iteration {result.iteration_number} escape attempt",
                threat="Time/resource constraints",
                survival_action="Team collaboration under pressure",
                outcome="Learned valuable lessons" if result.lessons_learned else "Limited progress"
            )
    
    def _extract_strategies_from_output(self, crew_output: str) -> List[str]:
        """Extract strategy descriptions from crew output."""
        strategies = []
        
        strategy_keywords = ['strategy', 'approach', 'plan', 'method', 'technique']
        lines = crew_output.split('\n')
        
        for line in lines:
            if any(keyword in line.lower() for keyword in strategy_keywords):
                if len(line.strip()) > 10:  # Filter out very short lines
                    strategies.append(line.strip())
        
        return strategies[:5]  # Limit to 5 strategies per iteration
    
    def check_stopping_conditions(self) -> Tuple[bool, str]:
        """
        Check if simulation should stop based on various conditions.
        
        Returns:
            Tuple of (should_stop, reason)
        """
        # Solution found
        if self.solution_found:
            return True, "Solution found - escape route identified"
        
        # Maximum iterations reached
        if self.current_iteration >= self.config.max_iterations:
            return True, f"Maximum iterations ({self.config.max_iterations}) reached"
        
        # Time expired in game
        if self.game_state.time_remaining <= 0:
            return True, "Game time expired - simulation failed"
        
        # Deadlock detection
        if self.config.deadlock_detection_enabled and len(self.iteration_results) >= 3:
            recent_progress = self._assess_recent_progress()
            if recent_progress < self.config.min_progress_threshold:
                return True, "Deadlock detected - no significant progress in recent iterations"
        
        # High stress causing breakdown
        if self.game_state.get_stress_level() >= 0.9:
            return True, "Team stress level critical - simulation breakdown"
        
        return False, "Continue simulation"
    
    def _assess_recent_progress(self) -> float:
        """Assess progress made in recent iterations."""
        if len(self.iteration_results) < 3:
            return 1.0  # Assume progress in early iterations
        
        recent_results = self.iteration_results[-3:]
        progress_indicators = 0
        
        for result in recent_results:
            # Count indicators of progress
            output_lower = result.crew_output.lower()
            if any(indicator in output_lower for indicator in 
                   ['progress', 'discovered', 'breakthrough', 'solution', 'success']):
                progress_indicators += 1
            
            if result.lessons_learned:
                progress_indicators += len(result.lessons_learned) * 0.1
        
        return min(1.0, progress_indicators / len(recent_results))
    
    def _save_iteration_result(self, result: IterationResult):
        """Save individual iteration result to file."""
        filename = os.path.join(self.data_dir, f"iteration_{result.iteration_number}.json")
        with open(filename, 'w') as f:
            json.dump(asdict(result), f, indent=2)
    
    def run_full_simulation(self) -> Dict[str, Any]:
        """
        Run the complete iterative simulation until stopping conditions are met.
        
        Returns:
            Comprehensive simulation report
        """
        simulation_start_time = datetime.now()
        
        print(f"🚀 Starting Full Iterative Escape Room Simulation")
        print(f"📋 Configuration: Max {self.config.max_iterations} iterations")
        print(f"🧠 Memory: {'Enabled' if self.config.enable_memory else 'Disabled'}")
        print(f"⏰ Started at: {simulation_start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        try:
            while self.simulation_active:
                # Run current iteration
                iteration_result = self.run_single_iteration()
                
                # Check stopping conditions
                should_stop, stop_reason = self.check_stopping_conditions()
                
                if should_stop:
                    print(f"\n🛑 Simulation ended: {stop_reason}")
                    print(f"📊 Total iterations completed: {self.current_iteration + 1}")
                    break
                
                # Prepare for next iteration
                self.current_iteration += 1
                print(f"\n⏭️  Preparing for iteration {self.current_iteration + 1}...")
                
                # Brief pause for readability
                import time
                time.sleep(1)
        
        except KeyboardInterrupt:
            print(f"\n⚠️  Simulation interrupted by user")
            stop_reason = "User interruption"
        
        except Exception as e:
            print(f"\n❌ Simulation error: {str(e)}")
            stop_reason = f"Technical error: {str(e)}"
        
        finally:
            self.simulation_active = False
        
        # Generate final report
        final_report = self._generate_final_report(simulation_start_time, stop_reason)
        
        # Save comprehensive results
        self._save_final_results(final_report)
        
        return final_report
    
    def _generate_final_report(self, start_time: datetime, stop_reason: str) -> Dict[str, Any]:
        """Generate comprehensive final simulation report."""
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        report = {
            "simulation_metadata": {
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "duration_seconds": duration,
                "total_iterations": len(self.iteration_results),
                "stop_reason": stop_reason,
                "solution_found": self.solution_found
            },
            "configuration": asdict(self.config),
            "final_game_state": self.game_state.to_dict(),
            "agents_summary": {
                "strategist": {"role": self.agents[0].role, "memory_enabled": self.agents[0].memory},
                "mediator": {"role": self.agents[1].role, "memory_enabled": self.agents[1].memory},
                "survivor": {"role": self.agents[2].role, "memory_enabled": self.agents[2].memory}
            },
            "iteration_summary": [
                {
                    "iteration": result.iteration_number,
                    "consensus": result.agents_consensus,
                    "lessons_count": len(result.lessons_learned),
                    "solution_found": result.solution_found
                }
                for result in self.iteration_results
            ],
            "learning_analysis": {
                "total_lessons_learned": sum(len(r.lessons_learned) for r in self.iteration_results),
                "consensus_rate": sum(1 for r in self.iteration_results if r.agents_consensus) / len(self.iteration_results) if self.iteration_results else 0,
                "failed_strategies": self.memory_manager.get_failed_strategies(),
                "successful_strategies": self.memory_manager.get_successful_strategies()
            },
            "outcome": "SUCCESS" if self.solution_found else "INCOMPLETE",
            "detailed_results": [asdict(result) for result in self.iteration_results]
        }
        
        return report
    
    def _save_final_results(self, report: Dict[str, Any]):
        """Save final comprehensive report."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save full report
        report_filename = os.path.join(self.data_dir, f"simulation_report_{timestamp}.json")
        with open(report_filename, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Save memory data
        self.memory_manager.save_all_memories()
        
        print(f"💾 Final report saved to: {report_filename}")
        print(f"📈 Simulation outcome: {report['outcome']}")
        
        if self.solution_found:
            print(f"🎉 SUCCESS: Escape solution found in {report['simulation_metadata']['total_iterations']} iterations")
        else:
            print(f"⏳ INCOMPLETE: Simulation ended after {report['simulation_metadata']['total_iterations']} iterations")


# Example usage and testing
if __name__ == "__main__":
    # Create custom configuration
    config = SimulationConfig(
        max_iterations=8,
        enable_memory=True,
        verbose_output=True,
        deadlock_detection_enabled=True
    )
    
    # Initialize and run simulation
    simulation = IterativeEscapeSimulation(config)
    final_report = simulation.run_full_simulation()
    
    print(f"\n📋 FINAL SIMULATION SUMMARY:")
    print(f"   Outcome: {final_report['outcome']}")
    print(f"   Iterations: {final_report['simulation_metadata']['total_iterations']}")
    print(f"   Duration: {final_report['simulation_metadata']['duration_seconds']:.1f} seconds")
    print(f"   Solution Found: {final_report['simulation_metadata']['solution_found']}")