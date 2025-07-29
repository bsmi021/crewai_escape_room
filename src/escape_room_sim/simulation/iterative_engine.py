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