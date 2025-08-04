# Code Quality Improvements Specification
## CrewAI Escape Room Simulation - Phase 2 Requirements

### Document Information
- **Document Type**: Technical Specification  
- **Priority**: HIGH - Maintainability and Performance
- **Estimated Time**: 20-30 hours
- **Dependencies**: Critical Fixes (Phase 1) must be completed first
- **Author**: Code Smell Analysis Report
- **Date**: 2025-08-04

---

## Overview

This specification addresses code smells, anti-patterns, and maintainability issues identified in the CrewAI Escape Room Simulation codebase. These improvements will enhance code readability, reduce technical debt, and improve long-term maintainability.

---

## 1. Bloated Code Refactoring

### 1.1 Long Method Refactoring - Iterative Engine

**Priority**: HIGH  
**File**: `src/escape_room_sim/simulation/iterative_engine.py`  
**Issue**: Multiple methods exceed 50+ lines, violating Single Responsibility Principle

#### 1.1.1 Refactor `create_iteration_tasks()` Method

**Current**: Lines 154-262 (108 lines)  
**Target**: Break into focused methods under 20 lines each

```python
def create_iteration_tasks(self) -> List[Task]:
    """Create tasks for current iteration with agent-specific contexts."""
    return [
        self._create_strategic_analysis_task(),
        self._create_group_facilitation_task(), 
        self._create_execution_coordination_task()
    ]

def _create_strategic_analysis_task(self) -> Task:
    """Create strategic analysis task for Strategist agent."""
    strategist_context = self._get_strategist_iteration_context()
    
    return Task(
        description=f"""
STRATEGIC ANALYSIS - ITERATION {self.current_iteration + 1}

{strategist_context}

Your task is to analyze the current escape room situation and develop an evidence-based strategy:

1. SITUATION ASSESSMENT:
   - Evaluate current room state, available resources, and team status
   - Identify key constraints and opportunities
   - Assess time pressure and stress factors

2. STRATEGIC PLANNING:
   - Develop systematic approach based on military tactical principles
   - Consider multiple contingency plans
   - Prioritize actions based on success probability and resource requirements

3. LEARNING INTEGRATION:
   - Apply lessons learned from previous iteration attempts
   - Avoid repeating failed strategies
   - Adapt approach based on team dynamics feedback

4. RECOMMENDATION OUTPUT:
   - Provide clear, actionable strategic recommendations
   - Include resource allocation and timeline estimates
   - Identify critical success factors and potential risks

Focus on evidence-based analysis and systematic problem-solving approach.
        """,
        agent=self.strategist,
        expected_output="Detailed strategic analysis with actionable recommendations and risk assessment",
        context=strategist_context
    )

def _create_group_facilitation_task(self) -> Task:
    """Create group facilitation task for Mediator agent.""" 
    mediator_context = self._get_mediator_iteration_context()
    
    return Task(
        description=f"""
GROUP FACILITATION - ITERATION {self.current_iteration + 1}

{mediator_context}

Your task is to facilitate team collaboration and build consensus:

1. TEAM DYNAMICS ASSESSMENT:
   - Evaluate current stress levels and interpersonal relationships
   - Identify potential conflicts or communication barriers
   - Assess team morale and cohesion

2. CONSENSUS BUILDING:
   - Help team evaluate strategic options presented
   - Facilitate discussion of different approaches
   - Guide group toward collaborative decision-making

3. CONFLICT RESOLUTION:
   - Address any disagreements or competing priorities
   - Ensure all voices are heard and considered
   - Find compromise solutions when needed

4. COORDINATION SUPPORT:
   - Help establish clear roles and responsibilities
   - Ensure effective communication channels
   - Monitor team emotional state throughout process

Focus empathetic communication and collaborative problem-solving.
        """,
        agent=self.mediator,
        expected_output="Team consensus report with collaboration plan and risk mitigation strategies",
        context=mediator_context
    )

def _create_execution_coordination_task(self) -> Task:
    """Create execution coordination task for Survivor agent."""
    survivor_context = self._get_survivor_iteration_context()
    
    return Task(
        description=f"""
EXECUTION COORDINATION - ITERATION {self.current_iteration + 1}

{survivor_context}

Your task is to coordinate practical execution of the agreed strategy:

1. THREAT ASSESSMENT:
   - Evaluate immediate risks and dangers
   - Assess individual vs team survival probability
   - Identify critical decision points

2. RESOURCE MANAGEMENT:
   - Coordinate efficient use of available resources
   - Identify resource constraints and alternatives
   - Plan contingency measures for resource failures

3. EXECUTION PLANNING:
   - Develop step-by-step action plan
   - Assign specific tasks and responsibilities
   - Establish timing and coordination protocols

4. SURVIVAL PRIORITY BALANCE:
   - Balance team collaboration with individual survival instincts
   - Make pragmatic decisions under pressure
   - Ensure execution focuses on highest probability success actions

Focus on practical implementation and survival-oriented decision making.
        """,
        agent=self.survivor,
        expected_output="Detailed execution plan with resource allocation and survival risk assessment",
        context=survivor_context
    )

def _get_strategist_iteration_context(self) -> str:
    """Get context specific to strategist for current iteration."""
    return get_strategist_context_for_iteration(
        iteration_num=self.current_iteration + 1,
        previous_failures=self.memory_manager.get_failed_strategies(),
        current_resources=self.game_state.get_resource_summary()
    )

def _get_mediator_iteration_context(self) -> str:
    """Get context specific to mediator for current iteration."""
    return get_mediator_context_for_iteration(
        iteration_num=self.current_iteration + 1,
        relationship_tracker=self.relationship_tracker,
        team_stress_level=self.game_state.stress_level,
        previous_conflicts=self.memory_manager.get_previous_conflicts()
    )

def _get_survivor_iteration_context(self) -> str:
    """Get context specific to survivor for current iteration."""
    return get_survivor_context_for_iteration(
        iteration_num=self.current_iteration + 1,
        survival_memory=self.survival_memory,
        current_threat_level=self.game_state.calculate_threat_level(),
        resource_status=self.game_state.get_resource_summary()
    )
```

#### 1.1.2 Refactor `run_single_iteration()` Method

**Current**: Lines 264-353 (89 lines)  
**Target**: Extract sub-methods for clarity

```python
def run_single_iteration(self) -> IterationResult:
    """Execute a single iteration of the escape room simulation."""
    iteration_start = time.time()
    
    try:
        # Prepare iteration
        self._prepare_iteration()
        
        # Execute core simulation
        crew_output = self._execute_crew_iteration()
        
        # Process results
        result = self._process_iteration_results(crew_output, iteration_start)
        
        # Update game state
        self._update_game_state(result)
        
        return result
        
    except Exception as e:
        return self._handle_iteration_error(e, iteration_start)

def _prepare_iteration(self):
    """Prepare agents and game state for iteration."""
    self.current_iteration += 1
    console.print(f"\n🎯 Starting Iteration {self.current_iteration}")
    
    # Update game state based on time passage
    self.game_state.consume_time(2)  # Each iteration takes 2 minutes
    
    # Log current state
    if self.config.verbose_output:
        self._log_iteration_state()

def _execute_crew_iteration(self) -> str:
    """Execute the crew iteration and return output."""
    # Create agents with updated context
    agents = self._create_agents_for_iteration()
    
    # Create tasks for this iteration
    tasks = self.create_iteration_tasks()
    
    # Create and execute crew
    crew = self._create_crew(agents, tasks)
    
    console.print("🚀 Executing crew iteration...")
    result = crew.kickoff()
    
    return str(result)

def _process_iteration_results(self, crew_output: str, start_time: float) -> IterationResult:
    """Process crew output into structured iteration result."""
    # Extract key information from output
    lessons_learned = self._extract_lessons_learned(crew_output)
    strategies_attempted = self._extract_strategies(crew_output)
    consensus_reached = self._detect_consensus(crew_output)
    
    # Determine iteration success
    success = self._evaluate_iteration_success(crew_output)
    
    # Create structured result
    result = IterationResult(
        iteration_number=self.current_iteration,
        timestamp=datetime.now().isoformat(),
        crew_output=crew_output,
        success=success,
        execution_time=time.time() - start_time,
        lessons_learned=lessons_learned,
        strategies_attempted=strategies_attempted,
        agents_consensus=consensus_reached,
        game_state_snapshot=self.game_state.export_state()
    )
    
    return result

def _update_game_state(self, result: IterationResult):
    """Update game state based on iteration results."""
    # Update memory systems
    self.memory_manager.record_iteration(result)
    
    # Update relationship tracker
    if result.agents_consensus:
        self.relationship_tracker.record_successful_collaboration(
            agents=["strategist", "mediator", "survivor"],
            strategy=result.strategies_attempted[0] if result.strategies_attempted else "general_cooperation",
            outcome="consensus_reached"
        )
    
    # Update survival memory
    if result.success:
        self.survival_memory.record_successful_strategy(
            situation="escape_planning",
            strategy=result.strategies_attempted[0] if result.strategies_attempted else "collaborative_approach",
            outcome="progress_made",
            agents_involved=["strategist", "mediator", "survivor"]
        )
```

#### 1.1.3 Refactor `run_full_simulation()` Method

**Current**: Lines 546-681 (135 lines)  
**Target**: Extract sub-methods and improve structure

```python
def run_full_simulation(self) -> Dict[str, Any]:
    """Run complete simulation with comprehensive result analysis."""
    simulation_start = time.time()
    
    try:
        # Initialize simulation
        self._initialize_simulation()
        
        # Execute iteration loop
        stop_reason = self._execute_simulation_loop()
        
        # Generate comprehensive results
        final_report = self._generate_final_report(simulation_start, stop_reason)
        
        # Save results and cleanup
        self._finalize_simulation(final_report)
        
        return final_report
        
    except Exception as e:
        return self._handle_simulation_error(e, simulation_start)

def _initialize_simulation(self):
    """Initialize simulation state and logging."""
    self.current_iteration = 0
    self.iteration_results = []
    
    console.print(f"\n🎮 Initializing Full Simulation")
    console.print(f"⚙️ Configuration: {self.config.max_iterations} max iterations, "
                 f"memory {'enabled' if self.config.enable_memory else 'disabled'}")

def _execute_simulation_loop(self) -> str:
    """Execute main simulation loop with safety measures."""
    MAX_TOTAL_TIME = 1800  # 30 minutes
    MAX_STAGNANT = 5
    
    start_time = time.time()
    stagnant_count = 0
    last_progress = None
    
    while True:
        # Safety checks
        if time.time() - start_time > MAX_TOTAL_TIME:
            return "Maximum simulation time exceeded"
            
        current_progress = self._get_progress_hash()
        if current_progress == last_progress:
            stagnant_count += 1
            if stagnant_count >= MAX_STAGNANT:
                return "No progress detected - simulation stagnant"
        else:
            stagnant_count = 0
            last_progress = current_progress
        
        # Execute iteration
        result = self.run_single_iteration()
        self.iteration_results.append(result)
        
        # Check stopping conditions
        should_stop, reason = self.check_stopping_conditions()
        if should_stop:
            return reason
            
        # Save intermediate results if configured
        if self.config.save_intermediate_results:
            self.save_iteration_result(result)

def _generate_final_report(self, start_time: float, stop_reason: str) -> Dict[str, Any]:
    """Generate comprehensive final simulation report."""
    total_time = time.time() - start_time
    
    # Basic metrics
    basic_metrics = self._calculate_basic_metrics(total_time, stop_reason)
    
    # Learning analysis
    learning_analysis = self._analyze_learning_progression()
    
    # Relationship analysis
    relationship_analysis = self._analyze_team_relationships()
    
    # Performance metrics
    performance_metrics = self._calculate_performance_metrics()
    
    return {
        **basic_metrics,
        "learning_analysis": learning_analysis,
        "relationship_analysis": relationship_analysis,
        "performance_metrics": performance_metrics,
        "detailed_iterations": [asdict(result) for result in self.iteration_results],
        "final_game_state": self.game_state.export_state()
    }
```

---

## 2. Code Duplication Elimination

### 2.1 Agent Creation Factory Pattern

**Priority**: HIGH  
**Files**: `src/escape_room_sim/agents/*.py`  
**Issue**: Nearly identical agent creation functions

#### 2.1.1 Create Base Agent Factory

**File**: `src/escape_room_sim/agents/agent_factory.py` (new file)

```python
"""
Agent factory for creating CrewAI agents with consistent configuration.
Eliminates code duplication and centralizes agent creation logic.
"""

from typing import Dict, Any, Optional, List
from crewai import Agent
from enum import Enum

from ..utils.llm_config import create_gemini_llm, create_openai_llm, create_anthropic_llm


class AgentType(Enum):
    """Supported agent types."""
    STRATEGIST = "strategist"
    MEDIATOR = "mediator" 
    SURVIVOR = "survivor"


class AgentPersonality:
    """Agent personality configuration."""
    
    def __init__(
        self,
        role: str,
        goal: str,
        backstory: str,
        llm_temperature: float,
        communication_style: str,
        decision_making_approach: str,
        risk_tolerance: str
    ):
        self.role = role
        self.goal = goal
        self.backstory = backstory
        self.llm_temperature = llm_temperature
        self.communication_style = communication_style
        self.decision_making_approach = decision_making_approach
        self.risk_tolerance = risk_tolerance


class AgentFactory:
    """Factory for creating consistent CrewAI agents."""
    
    # Agent personality configurations
    PERSONALITIES = {
        AgentType.STRATEGIST: AgentPersonality(
            role="Strategic Analyst",
            goal="Develop systematic, evidence-based escape strategies that maximize team survival probability",
            backstory="""You are a former military tactical officer with 15 years of experience in high-pressure strategic planning.
Your analytical mind excels at breaking down complex problems into manageable components and developing
methodical approaches to overcome obstacles. You've successfully led teams through numerous crisis situations
by maintaining calm focus and systematic thinking even under extreme pressure.""",
            llm_temperature=0.5,
            communication_style="analytical, methodical, focused",
            decision_making_approach="systematic, evidence-based, calculated",
            risk_tolerance="moderate with calculated approach"
        ),
        
        AgentType.MEDIATOR: AgentPersonality(
            role="Group Facilitator",
            goal="Build team consensus and maintain group cohesion while facilitating effective collaboration",
            backstory="""You are a former crisis counselor and professional mediator with 12 years of experience
facilitating group dynamics in high-stress situations. Your strength lies in reading interpersonal
dynamics, building trust between conflicting parties, and guiding teams toward collaborative solutions.
You understand that survival depends not just on strategy, but on maintaining team unity and communication.""",
            llm_temperature=0.9,
            communication_style="empathetic, inclusive, diplomatic",
            decision_making_approach="consensus-building, collaborative, relationship-focused",
            risk_tolerance="moderate with emphasis on team cohesion"
        ),
        
        AgentType.SURVIVOR: AgentPersonality(
            role="Survival Specialist",
            goal="Ensure practical execution of escape plans with pragmatic focus on highest survival probability",
            backstory="""You are a former special forces operator with extensive survival training and real-world
experience in life-or-death situations. Your pragmatic approach prioritizes actionable solutions and
resource management. While you value teamwork, your survival instincts and training drive you to make
tough decisions quickly when lives are at stake, even if it means individual rather than group survival.""",
            llm_temperature=0.7,
            communication_style="direct, practical, action-oriented",
            decision_making_approach="pragmatic, survival-focused, decisive",
            risk_tolerance="calculated risk-taking with survival priority"
        )
    }
    
    @classmethod
    def create_agent(
        cls,
        agent_type: AgentType,
        memory_enabled: bool = True,
        verbose: bool = True,
        iteration_context: Optional[Dict[str, Any]] = None,
        llm_provider: str = "gemini"
    ) -> Agent:
        """
        Create a CrewAI agent with specified configuration.
        
        Args:
            agent_type: Type of agent to create
            memory_enabled: Whether to enable memory for the agent
            verbose: Whether to enable verbose output
            iteration_context: Additional context for current iteration
            llm_provider: LLM provider to use ("gemini", "openai", "anthropic")
            
        Returns:
            Configured CrewAI Agent instance
        """
        if agent_type not in cls.PERSONALITIES:
            raise ValueError(f"Unknown agent type: {agent_type}")
        
        personality = cls.PERSONALITIES[agent_type]
        
        # Create LLM with agent-specific configuration
        llm = cls._create_llm(llm_provider, personality.llm_temperature)
        
        # Build enhanced backstory with context
        enhanced_backstory = cls._build_enhanced_backstory(personality, iteration_context)
        
        # Create agent
        agent = Agent(
            role=personality.role,
            goal=personality.goal,
            backstory=enhanced_backstory,
            verbose=verbose,
            memory=memory_enabled,
            llm=llm,
            allow_delegation=False,
            max_iter=3,
            max_execution_time=300
        )
        
        return agent
    
    @classmethod
    def _create_llm(cls, provider: str, temperature: float):
        """Create LLM instance based on provider and temperature."""
        llm_creators = {
            "gemini": create_gemini_llm,
            "openai": create_openai_llm,
            "anthropic": create_anthropic_llm
        }
        
        if provider not in llm_creators:
            raise ValueError(f"Unknown LLM provider: {provider}")
        
        return llm_creators[provider](temperature=temperature)
    
    @classmethod
    def _build_enhanced_backstory(
        cls, 
        personality: AgentPersonality, 
        iteration_context: Optional[Dict[str, Any]]
    ) -> str:
        """Build enhanced backstory with iteration context."""
        enhanced_backstory = personality.backstory
        
        if iteration_context:
            # Add learning context
            if iteration_context.get("failed_strategies"):
                enhanced_backstory += f"""

IMPORTANT LEARNING FROM PREVIOUS ATTEMPTS:
You have learned from {len(iteration_context['failed_strategies'])} previous failed strategies:
{'; '.join(iteration_context['failed_strategies'][:3])}

Use this experience to avoid repeating ineffective approaches and develop better solutions."""

            # Add relationship context
            if iteration_context.get("team_relationships"):
                enhanced_backstory += f"""

CURRENT TEAM DYNAMICS:
{iteration_context['team_relationships']}
Consider these relationship factors in your communication and decision-making approach."""

            # Add resource context
            if iteration_context.get("available_resources"):
                enhanced_backstory += f"""

CURRENT RESOURCE STATUS:
{iteration_context['available_resources']}
Factor these available resources into your strategic thinking and recommendations."""
        
        return enhanced_backstory
    
    @classmethod
    def create_team(
        cls,
        memory_enabled: bool = True,
        verbose: bool = True,
        iteration_context: Optional[Dict[str, Any]] = None,
        llm_provider: str = "gemini"
    ) -> List[Agent]:
        """
        Create complete team of three agents.
        
        Args:
            memory_enabled: Whether to enable memory for agents
            verbose: Whether to enable verbose output
            iteration_context: Additional context for current iteration
            llm_provider: LLM provider to use
            
        Returns:
            List of configured agents [strategist, mediator, survivor]
        """
        return [
            cls.create_agent(AgentType.STRATEGIST, memory_enabled, verbose, iteration_context, llm_provider),
            cls.create_agent(AgentType.MEDIATOR, memory_enabled, verbose, iteration_context, llm_provider),
            cls.create_agent(AgentType.SURVIVOR, memory_enabled, verbose, iteration_context, llm_provider)
        ]
```

#### 2.1.2 Update Individual Agent Files

**Refactor existing agent files to use factory:**

```python
# src/escape_room_sim/agents/strategist.py
from .agent_factory import AgentFactory, AgentType

def create_strategist_agent(
    memory_enabled: bool = True,
    verbose: bool = True,
    iteration_context: Optional[Dict] = None,
    llm_provider: str = "gemini"
) -> Agent:
    """Create strategist agent using factory pattern."""
    return AgentFactory.create_agent(
        AgentType.STRATEGIST,
        memory_enabled=memory_enabled,
        verbose=verbose,
        iteration_context=iteration_context,
        llm_provider=llm_provider
    )

# Similar updates for mediator.py and survivor.py
```

---

## 3. Magic Numbers and Configuration

### 3.1 Create Configuration Constants

**Priority**: MEDIUM  
**File**: `src/escape_room_sim/config/constants.py` (new file)

```python
"""
Configuration constants for the CrewAI Escape Room Simulation.
Centralizes magic numbers and configuration values.
"""

from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class SimulationConstants:
    """Core simulation configuration constants."""
    
    # Time limits
    DEFAULT_TIME_LIMIT_MINUTES: int = 60
    MAX_SIMULATION_TIME_SECONDS: int = 1800  # 30 minutes
    ITERATION_TIME_COST_MINUTES: int = 2
    
    # Iteration limits
    DEFAULT_MAX_ITERATIONS: int = 10
    MAX_STAGNANT_ITERATIONS: int = 5
    MAX_AGENT_ITERATIONS: int = 3
    
    # Memory configuration
    MAX_MEMORY_ENTRIES: int = 1000
    DEFAULT_IMPORTANCE_SCORE: float = 0.5
    MEMORY_CLEANUP_THRESHOLD: int = 500
    
    # Stress and relationship factors
    BASE_STRESS_LEVEL: float = 0.3
    MAX_STRESS_LEVEL: float = 1.0
    STRESS_TIME_FACTOR: float = 0.1
    TRUST_LEVEL_DEFAULT: float = 0.5
    COLLABORATION_TRUST_BONUS: float = 0.1
    CONFLICT_TRUST_PENALTY: float = 0.1


@dataclass  
class LLMConstants:
    """LLM configuration constants."""
    
    # Model configurations
    GEMINI_DEFAULT_MODEL: str = "gemini-2.5-flash-lite"
    OPENAI_DEFAULT_MODEL: str = "gpt-4"
    ANTHROPIC_DEFAULT_MODEL: str = "claude-3-sonnet-20240229"
    
    # Embedding models
    GEMINI_EMBEDDING_MODEL: str = "text-embedding-004"
    OPENAI_EMBEDDING_MODEL: str = "text-embedding-3-small"
    
    # Temperature settings
    STRATEGIST_TEMPERATURE: float = 0.5
    MEDIATOR_TEMPERATURE: float = 0.9
    SURVIVOR_TEMPERATURE: float = 0.7
    
    # Execution limits
    MAX_EXECUTION_TIME_SECONDS: int = 300
    REQUEST_TIMEOUT_SECONDS: int = 60


@dataclass
class RoomConstants:
    """Escape room configuration constants."""
    
    # Resource quantities
    DEFAULT_RESOURCE_QUANTITY: int = 1
    CONSUMABLE_RESOURCE_MAX: int = 3
    
    # Puzzle difficulties
    EASY_PUZZLE_DIFFICULTY: float = 0.3
    MEDIUM_PUZZLE_DIFFICULTY: float = 0.6
    HARD_PUZZLE_DIFFICULTY: float = 0.9
    
    # Exit route capacities
    MAIN_DOOR_CAPACITY: int = 2  # Enforces survival constraint
    VENT_CAPACITY: int = 1
    WINDOW_CAPACITY: int = 2
    HIDDEN_PASSAGE_CAPACITY: int = 2
    
    # Risk levels  
    LOW_RISK: float = 0.2
    MEDIUM_RISK: float = 0.5
    HIGH_RISK: float = 0.8


@dataclass
class TestConstants:
    """Constants for testing and development."""
    
    # Test timeouts
    UNIT_TEST_TIMEOUT: int = 30
    INTEGRATION_TEST_TIMEOUT: int = 120
    
    # Mock data
    MOCK_AGENT_COUNT: int = 3
    MOCK_ITERATION_COUNT: int = 5
    
    # Test file patterns
    TEST_DATA_DIR: str = "tests/fixtures"
    COVERAGE_THRESHOLD: float = 0.90


# Export commonly used constants
DEFAULT_CONFIG = SimulationConstants()
LLM_CONFIG = LLMConstants()
ROOM_CONFIG = RoomConstants()
TEST_CONFIG = TestConstants()
```

### 3.2 Update Code to Use Constants

**Example updates throughout codebase:**

```python
# Before (in escape_room_state.py)
self.time_remaining = 60

# After
from ..config.constants import ROOM_CONFIG
self.time_remaining = ROOM_CONFIG.DEFAULT_TIME_LIMIT_MINUTES

# Before (in simple_engine.py)
"model": "text-embedding-3-small"

# After
from ..config.constants import LLM_CONFIG
"model": LLM_CONFIG.OPENAI_EMBEDDING_MODEL
```

---

## 4. Coupling and Cohesion Improvements

### 4.1 Memory System Decoupling

**Priority**: MEDIUM  
**File**: `src/escape_room_sim/simulation/simple_engine.py`  
**Issue**: Tight coupling between simulation engine and memory internals

#### 4.1.1 Create Memory Interface

**File**: `src/escape_room_sim/memory/memory_interface.py` (new file)

```python
"""
Abstract interface for memory systems.
Decouples simulation engine from specific memory implementations.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
from ..simulation.iteration_result import IterationResult


class MemoryInterface(ABC):
    """Abstract interface for agent memory systems."""
    
    @abstractmethod
    def record_iteration(self, result: IterationResult) -> None:
        """Record iteration result in memory."""
        pass
    
    @abstractmethod
    def get_failed_strategies(self, limit: int = 5) -> List[str]:
        """Get list of previously failed strategies."""
        pass
    
    @abstractmethod
    def get_successful_strategies(self, limit: int = 5) -> List[str]:
        """Get list of previously successful strategies."""
        pass
    
    @abstractmethod
    def get_context_for_agent(self, agent_name: str, iteration: int) -> str:
        """Get memory context for specific agent."""
        pass
    
    @abstractmethod
    def export_data(self) -> Dict[str, Any]:
        """Export memory data for persistence."""
        pass
    
    @abstractmethod
    def get_learning_analysis(self) -> Dict[str, Any]:
        """Get analysis of learning progression."""
        pass


class RelationshipInterface(ABC):
    """Abstract interface for relationship tracking systems."""
    
    @abstractmethod
    def record_successful_collaboration(
        self, 
        agents: List[str], 
        strategy: str, 
        outcome: str
    ) -> None:
        """Record successful collaboration."""
        pass
    
    @abstractmethod
    def record_conflict(
        self, 
        agent_a: str, 
        agent_b: str, 
        reason: str, 
        resolution: str
    ) -> None:
        """Record conflict between agents."""
        pass
    
    @abstractmethod
    def get_team_cohesion(self, agents: List[str]) -> float:
        """Get team cohesion score."""
        pass
    
    @abstractmethod
    def export_data(self) -> Dict[str, Any]:
        """Export relationship data."""
        pass


class SurvivalInterface(ABC):
    """Abstract interface for survival memory systems."""
    
    @abstractmethod
    def record_close_call(
        self, 
        situation: str, 
        threat: str, 
        action: str, 
        outcome: str
    ) -> None:
        """Record survival close call."""
        pass
    
    @abstractmethod
    def get_relevant_experiences(self, count: int = 5) -> str:
        """Get relevant survival experiences."""
        pass
    
    @abstractmethod
    def calculate_survival_probability(
        self, 
        situation: Dict[str, Any], 
        action: str
    ) -> float:
        """Calculate survival probability for action."""
        pass
```

#### 4.1.2 Update Simulation Engine

```python
# In simple_engine.py - use interfaces instead of concrete classes
from ..memory.memory_interface import MemoryInterface, RelationshipInterface, SurvivalInterface

class SimpleEscapeSimulation:
    def __init__(
        self, 
        config: SimulationConfig = None, 
        data_dir: str = "data",
        memory_system: Optional[MemoryInterface] = None,
        relationship_system: Optional[RelationshipInterface] = None,
        survival_system: Optional[SurvivalInterface] = None
    ):
        # Use dependency injection instead of tight coupling
        self.memory_manager = memory_system or self._create_default_memory()
        self.relationship_tracker = relationship_system or self._create_default_relationships()
        self.survival_memory = survival_system or self._create_default_survival()
```

---

## 5. Error Handling Consistency

### 5.1 Standardized Exception Handling

**Priority**: MEDIUM  
**File**: `src/escape_room_sim/exceptions.py` (new file)

```python
"""
Custom exceptions for the CrewAI Escape Room Simulation.
Provides consistent error handling across the application.
"""


class EscapeRoomSimulationError(Exception):
    """Base exception for escape room simulation errors."""
    pass


class ConfigurationError(EscapeRoomSimulationError):
    """Error in simulation configuration."""
    pass


class AgentCreationError(EscapeRoomSimulationError):
    """Error creating or configuring agents."""
    pass


class MemorySystemError(EscapeRoomSimulationError):
    """Error in memory system operations."""
    pass


class GameStateError(EscapeRoomSimulationError):
    """Error in game state management."""
    pass


class LLMProviderError(EscapeRoomSimulationError):
    """Error with LLM provider configuration or communication."""
    pass


class SimulationExecutionError(EscapeRoomSimulationError):
    """Error during simulation execution."""
    pass
```

### 5.2 Consistent Error Handling Pattern

```python
# Example implementation pattern
import logging
from typing import Optional

logger = logging.getLogger(__name__)

def safe_operation_wrapper(operation_name: str):
    """Decorator for consistent error handling."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except EscapeRoomSimulationError:
                # Re-raise our custom exceptions
                raise
            except Exception as e:
                logger.error(f"Unexpected error in {operation_name}: {str(e)}")
                raise SimulationExecutionError(f"Failed to execute {operation_name}: {str(e)}") from e
        return wrapper
    return decorator

# Usage example
@safe_operation_wrapper("agent_creation")
def create_agent_safe(agent_type: AgentType, **kwargs) -> Agent:
    """Create agent with consistent error handling."""
    return AgentFactory.create_agent(agent_type, **kwargs)
```

---

## 6. Implementation Priority and Phases

### Phase 1: Critical Refactoring (Week 1)
1. **Long method refactoring** - Break down 100+ line methods
2. **Agent factory implementation** - Eliminate code duplication
3. **Constants extraction** - Replace magic numbers
4. **Basic error handling** - Add try/catch blocks

### Phase 2: Architecture Improvements (Week 2) 
1. **Interface implementation** - Decouple memory systems
2. **Dependency injection** - Reduce tight coupling  
3. **Exception hierarchy** - Standardize error handling
4. **Configuration management** - Centralize settings

### Phase 3: Polish and Optimization (Week 3)
1. **Performance optimization** - Address O(n²) algorithms
2. **Documentation updates** - Update all docstrings
3. **Code style consistency** - Apply formatting rules
4. **Final testing** - Comprehensive test coverage

---

## 7. Testing Requirements

### 7.1 Refactoring Tests
- All existing tests continue to pass after refactoring
- New factory methods create equivalent agents
- Constants produce same values as magic numbers
- Interfaces work with existing implementations

### 7.2 New Functionality Tests
- Agent factory creates correct agent types
- Memory interfaces work with different implementations
- Error handling produces appropriate exceptions
- Configuration constants are properly loaded

---

## 8. Acceptance Criteria

✅ **Methods under 30 lines each (target 20 lines)**  
✅ **Zero code duplication in agent creation**  
✅ **All magic numbers replaced with named constants**  
✅ **Loose coupling between major components**  
✅ **Consistent error handling throughout codebase**  
✅ **All existing functionality preserved**  
✅ **Test coverage maintained or improved**  
✅ **Performance equivalent or better**

---

## 9. Risk Assessment and Mitigation

**MEDIUM RISK**: Refactoring could introduce bugs  
**MITIGATION**: Comprehensive testing after each change, maintain backwards compatibility

**LOW RISK**: Performance impact from additional abstraction layers  
**MITIGATION**: Profile performance, optimize bottlenecks if needed

**LOW RISK**: Configuration changes break existing setups  
**MITIGATION**: Maintain default values, provide migration guide

---

**Document Version**: 1.0  
**Last Updated**: 2025-08-04  
**Status**: Ready for Implementation