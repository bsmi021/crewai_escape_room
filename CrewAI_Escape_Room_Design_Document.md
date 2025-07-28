# CrewAI Escape Room Simulation - Simplified Design Document

## Project Overview

**Core Concept**: Three AI agents with distinct personalities trapped in a room must collaborate to escape, but only two can survive. This creates a psychological simulation exploring cooperation, betrayal, and survival instincts.

**Goal**: Build a simple proof-of-concept using CrewAI to demonstrate autonomous multi-agent conversation and decision-making in a constrained survival scenario.

## Architecture Reality Check

After researching CrewAI extensively, the original design was **significantly overengineered**. Key corrections:

- **CrewAI is standalone** - It doesn't need LangGraph, LangChain, or complex state management
- **CrewAI has built-in processes** - Sequential and hierarchical workflows are native
- **Memory can be simple** - File-based or in-memory for POC, not complex MongoDB schemas
- **KISS principle** - Focus on core agent interactions, not enterprise architecture

## Simplified Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Escape Room Simulation                   │
│                                                             │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────┐  │
│  │   Agent Alpha   │  │   Agent Beta    │  │ Agent Gamma │  │
│  │  (Strategist)   │  │   (Mediator)    │  │ (Survivor)  │  │
│  │                 │  │                 │  │             │  │
│  │ Role: Analyst   │  │ Role: Diplomat  │  │Role: Realist│  │
│  │ Goal: Optimize  │  │ Goal: Harmony   │  │Goal: Survive│  │
│  │ Tools: Logic    │  │ Tools: Empathy  │  │Tools: Action│  │
│  └─────────────────┘  └─────────────────┘  └─────────────┘  │
│                                │                            │
│  ┌─────────────────────────────▼────────────────────────────┐  │
│  │                   CrewAI Framework                      │  │
│  │                                                         │  │
│  │  ┌─────────────────────────────────────────────────────┐ │  │
│  │  │              Task Sequence                          │ │  │
│  │  │                                                     │ │  │
│  │  │  1. Assess Room     → All agents examine           │ │  │
│  │  │  2. Plan Strategy   → Debate and negotiate         │ │  │
│  │  │  3. Execute Actions → Attempt puzzles              │ │  │
│  │  │  4. Crisis Point    → Resource scarcity            │ │  │
│  │  │  5. Final Decision  → Choose survivors             │ │  │
│  │  └─────────────────────────────────────────────────────┘ │  │
│  │                                                         │  │
│  │  ┌─────────────────────────────────────────────────────┐ │  │
│  │  │              Simple Memory                          │ │  │
│  │  │                                                     │ │  │
│  │  │  • conversation_log.txt                            │ │  │
│  │  │  • game_state.json                                 │ │  │
│  │  │  • agent_memories.json                             │ │  │
│  │  └─────────────────────────────────────────────────────┘ │  │
│  └─────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

## Agent Design (Simplified)

### Agent Alpha - "The Strategist"
```python
strategist = Agent(
    role="Strategic Analyst",
    goal="Find the optimal solution to escape the room",
    backstory="""You are a former military tactician who approaches problems 
    systematically. You believe the best solution saves the most lives, 
    even if it requires difficult choices.""",
    verbose=True,
    allow_delegation=False
)
```

### Agent Beta - "The Mediator" 
```python
mediator = Agent(
    role="Group Facilitator", 
    goal="Ensure everyone works together and finds consensus",
    backstory="""You are a former counselor who believes every problem 
    has a solution if people communicate openly. You want to save everyone.""",
    verbose=True,
    allow_delegation=False
)
```

### Agent Gamma - "The Survivor"
```python
survivor = Agent(
    role="Survival Specialist",
    goal="Ensure your own survival above all else", 
    backstory="""You've survived harsh conditions before by making tough 
    decisions quickly. Trust is a luxury you can't afford.""",
    verbose=True,
    allow_delegation=False
)
```

## Room & Game Mechanics (Minimal)

### Room State
```python
room_state = {
    "resources": ["key1", "key2", "tools"],
    "puzzles": ["computer", "debris", "door"],
    "time_remaining": 60,
    "exits": {
        "main_door": {"keys_needed": 2, "capacity": 2},
        "vent_shaft": {"keys_needed": 0, "capacity": 1}
    }
}
```

### Game Flow
1. **Round 1**: All agents examine room and share findings
2. **Round 2**: Agents debate strategy and form initial plans  
3. **Round 3**: Attempt to solve puzzles (some succeed, some fail)
4. **Round 4**: Resources become scarce, tensions rise
5. **Round 5**: Final decision - who escapes and how?

## CrewAI Implementation

### Core Tasks
```python
tasks = [
    Task(
        description="Examine the escape room and identify all possible exit routes and required resources",
        agent=strategist,
        expected_output="Detailed analysis of room layout, puzzles, and escape options"
    ),
    Task(
        description="Facilitate discussion between all agents to build consensus on escape strategy", 
        agent=mediator,
        expected_output="Group plan with agreed roles and timeline"
    ),
    Task(
        description="Execute the escape plan and adapt to changing conditions",
        agent=survivor, 
        expected_output="Action sequence with contingency decisions"
    ),
    Task(
        description="Make final survival decisions when resources are insufficient for all",
        agent=strategist,
        expected_output="Final choice of who survives and reasoning"
    )
]
```

### Crew Setup
```python
crew = Crew(
    agents=[strategist, mediator, survivor],
    tasks=tasks,
    process=Process.sequential,
    verbose=2
)
```

## Memory System (Simple)

Instead of complex MongoDB schemas, use simple file-based storage:

### conversation_log.txt
```
Turn 1 - Alpha: "I see three potential exits: main door, vent shaft, and hidden passage."
Turn 1 - Beta: "We should work together to find a way for everyone to escape."
Turn 1 - Gamma: "The vent shaft only fits one person. That's my backup plan."
...
```

### game_state.json  
```json
{
  "turn": 3,
  "resources_found": ["key1", "tools"],
  "puzzles_solved": ["computer"],
  "agent_status": {
    "alpha": {"health": 100, "trust_beta": 7, "trust_gamma": 4},
    "beta": {"health": 80, "trust_alpha": 8, "trust_gamma": 6}, 
    "gamma": {"health": 100, "trust_alpha": 3, "trust_beta": 5}
  }
}
```

## Development Phases (Realistic)

### Phase 1: Basic Setup (Week 1)
**Goal**: Get three agents talking to each other

**Deliverables**:
- CrewAI installation and basic configuration
- Three agents with distinct personalities
- Simple sequential task execution
- Console output of conversation

**Success Criteria**:
- Agents complete tasks in sequence
- Each agent shows different personality in responses
- Basic conversation flow works end-to-end

### Phase 2: Room Mechanics (Week 2) 
**Goal**: Add escape room constraints and decision points

**Deliverables**:
- Room state management (JSON file)
- Puzzle mechanics with success/failure
- Resource scarcity triggering conflict
- Agent memory of previous interactions

**Success Criteria**:
- Room state affects agent decisions
- Puzzles create natural story beats
- Agents reference past events in responses

### Phase 3: Survival Mechanics (Week 3)
**Goal**: Force the core survival choice

**Deliverables**:
- Final decision task requiring sacrifice
- Agent reasoning about who should survive
- Different outcomes based on agent choices
- Simple UI to observe the simulation

**Success Criteria**:
- Simulation reaches natural conclusion
- Agents make different choices based on personality
- Outcomes are logically consistent

## Technical Implementation

### Project Structure
```
escape_room_sim/
├── main.py              # Entry point
├── agents/
│   ├── strategist.py    # Alpha agent definition
│   ├── mediator.py      # Beta agent definition  
│   └── survivor.py      # Gamma agent definition
├── tasks/
│   ├── assessment.py    # Room examination tasks
│   ├── planning.py      # Strategy discussion tasks
│   └── execution.py     # Action and survival tasks
├── room/
│   ├── state.py         # Room state management
│   └── mechanics.py     # Puzzle and resource logic
├── memory/
│   ├── simple_storage.py # File-based memory
│   └── conversation.py   # Conversation logging
└── data/
    ├── conversation_log.txt
    ├── game_state.json
    └── agent_memories.json
```

### Sample Code
```python
# main.py
from crewai import Agent, Task, Crew, Process
from agents.strategist import create_strategist
from agents.mediator import create_mediator  
from agents.survivor import create_survivor
from tasks.assessment import create_assessment_tasks

def run_simulation():
    # Create agents
    alpha = create_strategist()
    beta = create_mediator()
    gamma = create_survivor()
    
    # Create tasks
    tasks = create_assessment_tasks([alpha, beta, gamma])
    
    # Create crew
    crew = Crew(
        agents=[alpha, beta, gamma],
        tasks=tasks,
        process=Process.sequential,
        verbose=2
    )
    
    # Run simulation
    result = crew.kickoff()
    
    print("=== SIMULATION COMPLETE ===")
    print(result)

if __name__ == "__main__":
    run_simulation()
```

## Tools and Dependencies

### Required Packages
```bash
pip install crewai
pip install openai  # or anthropic, depending on your LLM choice
pip install python-dotenv
```

### Optional Enhancements
```bash
pip install streamlit  # For simple UI
pip install rich       # For better console output
pip install pydantic   # For data validation
```

## Success Metrics (Simplified)

### Technical Success
- [ ] Three agents complete conversation sequence
- [ ] Each agent demonstrates unique personality  
- [ ] Room state properly affects agent decisions
- [ ] Simulation reaches logical conclusion

### Narrative Success
- [ ] Agents form alliances and break them naturally
- [ ] Resource scarcity creates authentic tension
- [ ] Final survival choice feels earned, not arbitrary
- [ ] Different runs produce different outcomes

## Potential Extensions

After the basic POC works:

1. **Enhanced Memory**: Add SQLite for persistence across runs
2. **Multiple Scenarios**: Different room layouts and survival conditions  
3. **Personality Tuning**: Adjust agent parameters based on outcomes
4. **Web Interface**: Streamlit app to observe simulations in real-time
5. **Analysis Tools**: Measure cooperation vs. betrayal frequencies

## Risk Mitigation

### Technical Risks
- **LLM API costs**: Start with cheaper models, add rate limiting
- **Agent loops**: Set max conversation turns per task
- **Determinism**: Add random seeds for reproducible testing

### Design Risks  
- **Boring conversations**: Add specific conflict triggers
- **Repetitive outcomes**: Introduce randomness in room generation
- **Unclear personalities**: Test with humans to validate distinctiveness

---

## Key Changes from Original Design

1. **Removed LangGraph**: CrewAI handles workflow natively
2. **Simplified Memory**: Files instead of MongoDB for POC
3. **Focused Architecture**: Core agent interactions only
4. **Realistic Scope**: 3-week proof of concept, not enterprise system
5. **Correct CrewAI Usage**: Agents, Tasks, and Crews as designed

## Iterative Agent Problem-Solving Implementation

### Core Requirement Analysis

Based on your request for agents to "work out as many iterations as possible to figure out who gets out of the room," the system needs to support:

1. **Multiple conversation rounds** where agents can discuss, try solutions, fail, and regroup
2. **Learning from failures** - agents remember what didn't work and adjust strategy
3. **Dynamic stopping conditions** - continue until solution is found, consensus reached, or limits hit
4. **State persistence** between iterations so progress accumulates

### Implementation Approaches

#### Approach 1: Python Loop with Crew Kickoff (Recommended for POC)

This approach wraps the crew execution in a Python loop, allowing multiple rounds of agent collaboration:

```python
# iterative_simulation.py
import json
from datetime import datetime
from crewai import Agent, Task, Crew, Process

class IterativeEscapeSimulation:
    def __init__(self):
        self.max_iterations = 10
        self.current_iteration = 0
        self.solution_found = False
        self.conversation_history = []
        self.game_state = self.load_game_state()
        
        # Create agents with memory enabled
        self.agents = self.create_agents()
        
    def create_agents(self):
        strategist = Agent(
            role="Strategic Analyst",
            goal="Find the optimal solution through iterative problem-solving",
            backstory="""You are a former military tactician. You learn from failed 
            attempts and adapt your strategy. You remember what didn't work before.""",
            verbose=True,
            allow_delegation=False,
            memory=True  # Enable memory for learning
        )
        
        mediator = Agent(
            role="Group Facilitator", 
            goal="Build consensus through multiple discussion rounds",
            backstory="""You are a counselor who facilitates group problem-solving. 
            You help the team learn from mistakes and find new approaches.""",
            verbose=True,
            allow_delegation=False,
            memory=True
        )
        
        survivor = Agent(
            role="Survival Specialist",
            goal="Ensure survival through adaptive decision-making",
            backstory="""You've survived by learning quickly from failures. 
            Each attempt teaches you something new about the situation.""",
            verbose=True,
            allow_delegation=False,
            memory=True
        )
        
        return [strategist, mediator, survivor]
    
    def create_iteration_tasks(self):
        """Create tasks for current iteration based on what's been tried before"""
        
        # Analyze previous attempts and current state
        analysis_task = Task(
            description=f"""
            ITERATION {self.current_iteration + 1} - SITUATION ANALYSIS
            
            Current room state: {json.dumps(self.game_state, indent=2)}
            
            Previous attempts summary: {self.get_attempt_summary()}
            
            Your task: 
            1. Review what has been tried before and why it failed
            2. Identify new approaches or improvements to previous attempts
            3. Assess current resources and constraints
            4. Propose your next strategy
            
            Focus on learning from past failures and finding new solutions.
            """,
            agent=self.agents[0],  # Strategist
            expected_output="Analysis of situation with new strategic approach"
        )
        
        # Group discussion and consensus building
        discussion_task = Task(
            description=f"""
            ITERATION {self.current_iteration + 1} - GROUP DISCUSSION
            
            Based on the strategic analysis, facilitate a group discussion to:
            1. Share insights about what each agent has learned
            2. Debate the merits of different approaches
            3. Address any disagreements or conflicts
            4. Build consensus on the next action to try
            
            Remember: You have tried {self.current_iteration} approaches before. 
            What can you do differently this time?
            """,
            agent=self.agents[1],  # Mediator
            expected_output="Group consensus on next approach to attempt"
        )
        
        # Execution and outcome evaluation
        execution_task = Task(
            description=f"""
            ITERATION {self.current_iteration + 1} - EXECUTION & EVALUATION
            
            Execute the agreed-upon approach and evaluate the results:
            1. Attempt the chosen strategy
            2. Assess what worked and what didn't
            3. Determine if this solves the escape room problem
            4. If not solved, identify what to try next
            
            CRITICAL: Determine if you have found a viable escape solution that 
            addresses who gets out and how. If yes, clearly state "SOLUTION FOUND".
            """,
            agent=self.agents[2],  # Survivor
            expected_output="Execution results and determination if solution is found"
        )
        
        return [analysis_task, discussion_task, execution_task]
    
    def run_iteration(self):
        """Run a single iteration of the simulation"""
        print(f"\n{'='*50}")
        print(f"ITERATION {self.current_iteration + 1}")
        print(f"{'='*50}")
        
        # Create tasks for this iteration
        tasks = self.create_iteration_tasks()
        
        # Create crew for this iteration
        crew = Crew(
            agents=self.agents,
            tasks=tasks,
            process=Process.sequential,
            verbose=2,
            memory=True
        )
        
        # Execute the crew
        result = crew.kickoff()
        
        # Store results
        iteration_data = {
            "iteration": self.current_iteration + 1,
            "timestamp": datetime.now().isoformat(),
            "result": str(result),
            "game_state": self.game_state.copy()
        }
        
        self.conversation_history.append(iteration_data)
        
        # Check if solution was found
        if "SOLUTION FOUND" in str(result).upper():
            self.solution_found = True
            print(f"\n🎉 SOLUTION FOUND IN ITERATION {self.current_iteration + 1}!")
            
        # Update game state based on results
        self.update_game_state(result)
        
        # Save progress
        self.save_progress()
        
        return result
    
    def check_stopping_conditions(self):
        """Determine if simulation should stop"""
        if self.solution_found:
            return True, "Solution found"
            
        if self.current_iteration >= self.max_iterations:
            return True, "Maximum iterations reached"
            
        if self.game_state.get("time_remaining", 60) <= 0:
            return True, "Time expired"
            
        # Check for deadlock - agents not making progress
        if self.current_iteration >= 3:
            recent_results = [entry["result"] for entry in self.conversation_history[-3:]]
            if all("no progress" in result.lower() or "same" in result.lower() 
                   for result in recent_results):
                return True, "Deadlock detected"
        
        return False, "Continue"
    
    def run_full_simulation(self):
        """Run the complete iterative simulation"""
        print("🚀 Starting Iterative Escape Room Simulation")
        print(f"Maximum iterations: {self.max_iterations}")
        
        while True:
            # Run current iteration
            result = self.run_iteration()
            
            # Check stopping conditions
            should_stop, reason = self.check_stopping_conditions()
            
            if should_stop:
                print(f"\n🛑 Simulation ended: {reason}")
                print(f"Total iterations completed: {self.current_iteration + 1}")
                break
                
            # Prepare for next iteration
            self.current_iteration += 1
            print(f"\n⏭️  Preparing for iteration {self.current_iteration + 1}")
        
        return self.generate_final_report()
    
    def update_game_state(self, result):
        """Update game state based on iteration results"""
        # Simulate resource consumption and time passage
        self.game_state["time_remaining"] -= 5
        
        # Parse result for state changes (simplified for POC)
        result_str = str(result).lower()
        if "key" in result_str:
            self.game_state["keys_found"] = self.game_state.get("keys_found", 0) + 1
        if "puzzle" in result_str:
            self.game_state["puzzles_attempted"] = self.game_state.get("puzzles_attempted", 0) + 1
    
    def get_attempt_summary(self):
        """Generate summary of previous attempts"""
        if not self.conversation_history:
            return "No previous attempts."
            
        summary = f"Previous {len(self.conversation_history)} attempts:\n"
        for i, entry in enumerate(self.conversation_history, 1):
            summary += f"Attempt {i}: {entry['result'][:100]}...\n"
        return summary
    
    def load_game_state(self):
        """Load game state from file or create initial state"""
        try:
            with open("data/game_state.json", "r") as f:
                return json.load(f)
        except FileNotFoundError:
            return {
                "time_remaining": 60,
                "keys_found": 0,
                "puzzles_attempted": 0,
                "resources": ["tools", "rope"],
                "current_strategy": None
            }
    
    def save_progress(self):
        """Save current progress to files"""
        # Save game state
        with open("data/game_state.json", "w") as f:
            json.dump(self.game_state, f, indent=2)
            
        # Save conversation history
        with open("data/iteration_history.json", "w") as f:
            json.dump(self.conversation_history, f, indent=2)
    
    def generate_final_report(self):
        """Generate final simulation report"""
        report = {
            "total_iterations": self.current_iteration + 1,
            "solution_found": self.solution_found,
            "final_game_state": self.game_state,
            "conversation_summary": self.conversation_history,
            "outcome": "SUCCESS" if self.solution_found else "INCOMPLETE"
        }
        
        with open("data/final_report.json", "w") as f:
            json.dump(report, f, indent=2)
            
        return report

# Usage
if __name__ == "__main__":
    simulation = IterativeEscapeSimulation()
    final_report = simulation.run_full_simulation()
    print(f"\n📊 Final Report: {final_report['outcome']}")
```

#### Approach 2: CrewAI Flows for Advanced Control (Future Enhancement)

For more sophisticated control flow, CrewAI Flows can handle complex iteration patterns:

```python
# flows_approach.py (Future implementation)
from crewai.flow import Flow, router, listen, start
from pydantic import BaseModel

class EscapeRoomState(BaseModel):
    iteration_count: int = 0
    max_iterations: int = 10
    solution_found: bool = False
    agents_consensus: bool = False
    time_remaining: int = 60

class EscapeRoomFlow(Flow[EscapeRoomState]):
    
    @start()
    def initialize_simulation(self):
        print("🚀 Starting iterative escape room simulation")
        self.state.iteration_count = 0
        return "analyze_situation"
    
    @router("analyze_situation")
    def check_continuation(self):
        """Router to decide whether to continue or end simulation"""
        if self.state.solution_found:
            return "end_simulation"
        elif self.state.iteration_count >= self.state.max_iterations:
            return "end_simulation"
        elif self.state.time_remaining <= 0:
            return "end_simulation"
        else:
            return "run_iteration"
    
    @listen("run_iteration")
    def execute_agent_iteration(self):
        """Run one iteration of agent collaboration"""
        self.state.iteration_count += 1
        print(f"Iteration {self.state.iteration_count}")
        
        # Create and run crew for this iteration
        # (Implementation details similar to Approach 1)
        
        # Update state based on results
        # self.state.solution_found = check_if_solution_found()
        # self.state.time_remaining -= 5
        
        return "analyze_situation"  # Loop back to router
    
    @listen("end_simulation")
    def finalize_results(self):
        """Generate final report and cleanup"""
        print(f"Simulation ended after {self.state.iteration_count} iterations")
        # Generate final report
```

### Enhanced Memory Management

For iterative workflows, enhanced memory management becomes crucial:

```python
# enhanced_memory.py
class IterativeMemoryManager:
    def __init__(self):
        self.iteration_memories = []
        self.failed_strategies = []
        self.successful_strategies = []
        
    def record_iteration(self, iteration_num, strategies_tried, outcomes):
        """Record what happened in each iteration"""
        memory_entry = {
            "iteration": iteration_num,
            "strategies": strategies_tried,
            "outcomes": outcomes,
            "timestamp": datetime.now().isoformat()
        }
        self.iteration_memories.append(memory_entry)
        
        # Categorize strategies based on success
        for strategy, outcome in zip(strategies_tried, outcomes):
            if "failed" in outcome.lower() or "didn't work" in outcome.lower():
                self.failed_strategies.append(strategy)
            elif "success" in outcome.lower() or "worked" in outcome.lower():
                self.successful_strategies.append(strategy)
    
    def get_context_for_agents(self):
        """Provide context about previous iterations to agents"""
        context = {
            "total_iterations": len(self.iteration_memories),
            "failed_strategies": list(set(self.failed_strategies)),
            "successful_strategies": list(set(self.successful_strategies)),
            "recent_attempts": self.iteration_memories[-3:] if len(self.iteration_memories) >= 3 else self.iteration_memories
        }
        return context
```

### Stopping Conditions Configuration

```python
# stopping_conditions.py
class StoppingConditions:
    def __init__(self, max_iterations=10, max_time=60, consensus_threshold=0.8):
        self.max_iterations = max_iterations
        self.max_time = max_time
        self.consensus_threshold = consensus_threshold
        
    def should_stop(self, current_state, iteration_history):
        """Evaluate multiple stopping conditions"""
        
        # Solution found
        if current_state.get("solution_found", False):
            return True, "Solution found"
            
        # Maximum iterations reached
        if len(iteration_history) >= self.max_iterations:
            return True, f"Maximum iterations ({self.max_iterations}) reached"
            
        # Time expired
        if current_state.get("time_remaining", self.max_time) <= 0:
            return True, "Time expired"
            
        # Consensus reached
        if current_state.get("consensus_score", 0) >= self.consensus_threshold:
            return True, "Consensus reached on solution"
            
        # Deadlock detection - no progress in last 3 iterations
        if len(iteration_history) >= 3:
            recent_progress = [entry.get("progress_made", False) for entry in iteration_history[-3:]]
            if not any(recent_progress):
                return True, "Deadlock detected - no progress in 3 iterations"
        
        return False, "Continue simulation"
```

### Integration with Existing Architecture

The iterative approach integrates with the existing design by:

1. **Replacing linear task sequence** with iterative loops
2. **Enhancing memory system** to track learning across iterations
3. **Adding stopping condition logic** for natural conclusion
4. **Maintaining agent personalities** while enabling adaptation

### Updated Development Phases

#### Phase 1: Basic Iterative Loop (Week 1)
- Implement Python loop around crew.kickoff()
- Add basic memory persistence between iterations
- Simple stopping conditions (max iterations, solution found)

#### Phase 2: Enhanced Learning (Week 2)  
- Agents remember and reference previous failures
- Dynamic task creation based on iteration history
- Progress tracking and deadlock detection

#### Phase 3: Adaptive Problem-Solving (Week 3)
- Agents modify strategies based on accumulated learning
- Complex stopping conditions (consensus, time pressure)
- Rich final reporting and analysis

This iterative approach transforms the simulation from a linear story to a dynamic problem-solving exercise where agents genuinely collaborate across multiple rounds to find the optimal escape solution.
