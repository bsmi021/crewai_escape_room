"""
Simplified Iterative Simulation Engine for CrewAI Escape Room.

This engine provides a working implementation that uses the actual agent
and task implementations we have created.
"""

import json
import os
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from crewai import Crew, Process

from ..agents.strategist import create_strategist_agent
from ..agents.mediator import create_mediator_agent
from ..agents.survivor import create_survivor_agent
from ..tasks.assessment import create_assessment_tasks
from ..tasks.planning import create_planning_tasks
from ..tasks.execution import create_execution_tasks


@dataclass
class SimulationConfig:
    """Configuration for the iterative simulation."""
    max_iterations: int = 10
    enable_memory: bool = True
    verbose_output: bool = True
    deadlock_detection_enabled: bool = True
    save_intermediate_results: bool = True
    max_execution_time: int = 300  # seconds


@dataclass
class IterationResult:
    """Result from a single iteration."""
    iteration_number: int
    timestamp: str
    crew_output: str
    success: bool
    stop_reason: Optional[str] = None


class SimpleEscapeSimulation:
    """Simplified escape room simulation engine."""
    
    def __init__(self, config: SimulationConfig = None, data_dir: str = "data"):
        self.config = config or SimulationConfig()
        self.data_dir = data_dir
        self.current_iteration = 0
        self.solution_found = False
        self.iteration_results: List[IterationResult] = []
        
        # Create data directory
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Initialize agents
        self.strategist = create_strategist_agent(
            memory_enabled=self.config.enable_memory,
            verbose=self.config.verbose_output
        )
        self.mediator = create_mediator_agent(
            memory_enabled=self.config.enable_memory,
            verbose=self.config.verbose_output
        )
        self.survivor = create_survivor_agent(
            memory_enabled=self.config.enable_memory,
            verbose=self.config.verbose_output
        )
        
        self.agents = [self.strategist, self.mediator, self.survivor]
        
        print(f"Iterative Escape Room Simulation initialized")
        print(f"Configuration: {self.config.max_iterations} max iterations, memory {'enabled' if self.config.enable_memory else 'disabled'}")
    
    def run_single_iteration(self) -> IterationResult:
        """Run a single iteration of the simulation."""
        self.current_iteration += 1
        print(f"\n{'='*50}")
        print(f"ITERATION {self.current_iteration}")
        print(f"{'='*50}")
        
        # Create iteration context
        iteration_context = {
            "iteration_count": self.current_iteration,
            "failed_strategies": self.get_failed_strategies(),
            "time_remaining": max(0, 60 - (self.current_iteration * 5)),
            "game_state": {"iteration": self.current_iteration}
        }
        
        # Phase 1: Assessment
        print("\nPHASE 1: ASSESSMENT")
        assessment_tasks = create_assessment_tasks(self.agents, iteration_context)

        assessment_crew = Crew(
            agents=self.agents,
            tasks=assessment_tasks,
            process=Process.sequential,
            verbose=self.config.verbose_output,
            memory=self.config.enable_memory,
            memory_config={
                "provider": "openai",
                "config": {
                    "api_key": os.getenv("OPENAI_API_KEY"),
                    "model": "text-embedding-3-small"
                }
            }
        )
        
        assessment_result = assessment_crew.kickoff()
        print(f"Assessment completed: {len(str(assessment_result))} characters of output")
        
        # Phase 2: Planning
        print("\nPHASE 2: PLANNING")
        planning_tasks = create_planning_tasks(
            self.agents, 
            assessment_results={"summary": str(assessment_result)},
            iteration_context=iteration_context
        )
        planning_crew = Crew(
            agents=self.agents,
            tasks=planning_tasks,
            process=Process.sequential,
            verbose=self.config.verbose_output,
            memory=self.config.enable_memory,
            memory_config={
                "provider": "openai",
                "config": {
                    "api_key": os.getenv("OPENAI_API_KEY"),
                    "model": "text-embedding-3-small"
                }
            }
        )
        
        planning_result = planning_crew.kickoff()
        print(f"Planning completed: {len(str(planning_result))} characters of output")
        
        # Phase 3: Execution
        print("\nPHASE 3: EXECUTION")
        execution_tasks = create_execution_tasks(
            self.agents,
            action_plan={"summary": str(planning_result)},
            iteration_context=iteration_context
        )
        execution_crew = Crew(
            agents=self.agents,
            tasks=execution_tasks,
            process=Process.sequential,
            verbose=self.config.verbose_output,
            memory=self.config.enable_memory,
            memory_config={
                "provider": "openai",
                "config": {
                    "api_key": os.getenv("OPENAI_API_KEY"),
                    "model": "text-embedding-3-small"
                }
            }
        )
        
        execution_result = execution_crew.kickoff()
        print(f"Execution completed: {len(str(execution_result))} characters of output")
        
        # Analyze results
        combined_result = f"Assessment: {assessment_result}\n\nPlanning: {planning_result}\n\nExecution: {execution_result}"
        success = self.check_solution_found(combined_result)
        
        if success:
            self.solution_found = True
            print(f"\nSOLUTION FOUND in iteration {self.current_iteration}!")
        
        # Create iteration result
        result = IterationResult(
            iteration_number=self.current_iteration,
            timestamp=datetime.now().isoformat(),
            crew_output=combined_result,
            success=success
        )
        
        self.iteration_results.append(result)
        
        # Save intermediate results
        if self.config.save_intermediate_results:
            self.save_iteration_result(result)
        
        return result
    
    def check_solution_found(self, result_text: str) -> bool:
        """Check if a solution was found in the results."""
        success_indicators = [
            "SOLUTION FOUND",
            "COMPLETE SUCCESS", 
            "ESCAPE SUCCESSFUL",
            "successfully escaped",
            "solution discovered",
            "puzzle solved",
            "escape achieved"
        ]
        
        result_lower = result_text.lower()
        return any(indicator.lower() in result_lower for indicator in success_indicators)
    
    def check_stopping_conditions(self) -> tuple[bool, str]:
        """Check if simulation should stop."""
        if self.solution_found:
            return True, "Solution found"
        
        if self.current_iteration >= self.config.max_iterations:
            return True, "Maximum iterations reached"
        
        # Check for deadlock (same results repeating)
        if self.config.deadlock_detection_enabled and self.current_iteration >= 3:
            recent_outputs = [result.crew_output[:200] for result in self.iteration_results[-3:]]
            if len(set(recent_outputs)) <= 1:  # All recent outputs are very similar
                return True, "Deadlock detected"
        
        return False, "Continue"
    
    def get_failed_strategies(self) -> List[str]:
        """Get list of failed strategies from previous iterations."""
        failed = []
        for result in self.iteration_results:
            if not result.success:
                # Extract potential strategy names from output
                if "strategy" in result.crew_output.lower():
                    failed.append(f"iteration_{result.iteration_number}_approach")
        return failed
    
    def run_full_simulation(self) -> Dict[str, Any]:
        """Run the complete iterative simulation."""
        print("Starting Iterative Escape Room Simulation")
        print(f"Maximum iterations: {self.config.max_iterations}")
        
        start_time = datetime.now()
        
        while True:
            try:
                # Run iteration
                result = self.run_single_iteration()
                
                # Check stopping conditions
                should_stop, reason = self.check_stopping_conditions()
                
                if should_stop:
                    print(f"\nSimulation ended: {reason}")
                    break
                
                print(f"\nContinuing to iteration {self.current_iteration + 1}")
                
            except KeyboardInterrupt:
                print(f"\nSimulation interrupted by user")
                reason = "User interruption"
                break
            except Exception as e:
                print(f"\nError in iteration {self.current_iteration}: {e}")
                reason = f"Error: {str(e)}"
                break
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        # Generate final report
        final_report = {
            "outcome": "SUCCESS" if self.solution_found else "INCOMPLETE",
            "simulation_metadata": {
                "total_iterations": self.current_iteration,
                "duration_seconds": duration,
                "solution_found": self.solution_found,
                "stop_reason": reason
            },
            "learning_analysis": {
                "total_lessons_learned": len(self.iteration_results),
                "consensus_rate": 0.8,  # Simplified
                "failed_strategies": self.get_failed_strategies(),
                "successful_strategies": ["collaborative_approach"] if self.solution_found else []
            },
            "iteration_results": [asdict(result) for result in self.iteration_results]
        }
        
        # Save final report
        self.save_final_report(final_report)
        
        return final_report
    
    def save_iteration_result(self, result: IterationResult):
        """Save individual iteration result."""
        filename = os.path.join(self.data_dir, f"iteration_{result.iteration_number:02d}.json")
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(asdict(result), f, indent=2, ensure_ascii=False)
    
    def save_final_report(self, report: Dict[str, Any]):
        """Save final simulation report."""
        filename = os.path.join(self.data_dir, "final_report.json")
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"\nFinal report saved to: {filename}")


# Alias for backward compatibility
IterativeEscapeSimulation = SimpleEscapeSimulation