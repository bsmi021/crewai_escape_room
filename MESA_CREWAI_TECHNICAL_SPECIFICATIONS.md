# Mesa-CrewAI Hybrid Architecture: Technical Implementation Specifications

## Executive Summary

This document provides comprehensive technical specifications for implementing the Mesa-CrewAI hybrid architecture. These specifications are designed to enable multiple developers to implement the system independently and produce identical results.

## 1. API Contracts

### 1.1 Core Interfaces

#### IPerceptionPipeline
```python
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import mesa

@dataclass
class PerceptionData:
    """Structured perception data from Mesa environment"""
    agent_id: str
    timestamp: datetime
    spatial_data: Dict[str, Any]  # position, nearby_objects, visibility_map
    environmental_state: Dict[str, Any]  # temperature, lighting, hazards
    nearby_agents: List[str]  # agent_ids within perception radius
    available_actions: List[str]  # valid actions in current state
    resources: Dict[str, Any]  # accessible resources
    constraints: Dict[str, Any]  # movement/action limitations

class IPerceptionPipeline(ABC):
    """Interface for extracting perceptions from Mesa environment"""
    
    @abstractmethod
    def extract_perceptions(self, mesa_model: mesa.Model) -> Dict[str, PerceptionData]:
        """
        Extract structured perceptions for each agent
        
        Args:
            mesa_model: Mesa model instance
            
        Returns:
            Dict[agent_id, PerceptionData]: Perceptions mapped by agent ID
            
        Raises:
            PerceptionExtractionError: When extraction fails
            ValidationError: When extracted data is invalid
        """
        pass
    
    @abstractmethod
    def filter_perceptions(self, perceptions: Dict[str, PerceptionData], 
                         agent_id: str) -> PerceptionData:
        """
        Filter perceptions based on agent capabilities
        
        Args:
            perceptions: All agent perceptions
            agent_id: Target agent ID
            
        Returns:
            PerceptionData: Filtered perceptions for agent
            
        Raises:
            AgentNotFoundError: When agent_id is invalid
        """
        pass
```

#### IDecisionEngine
```python
@dataclass
class DecisionData:
    """Structured decision data from CrewAI reasoning"""
    agent_id: str
    timestamp: datetime
    chosen_action: str  # primary action to execute
    action_parameters: Dict[str, Any]  # action-specific parameters
    reasoning: str  # natural language explanation
    confidence_level: float  # 0.0-1.0 confidence score
    fallback_actions: List[str]  # ordered fallback options
    execution_priority: int = 1  # 1=high, 2=medium, 3=low
    expected_duration: float = 1.0  # seconds
    resource_requirements: List[str] = field(default_factory=list)

class IDecisionEngine(ABC):
    """Interface for CrewAI reasoning and decision making"""
    
    @abstractmethod
    async def reason_and_decide(self, perceptions: Dict[str, PerceptionData]) -> Dict[str, DecisionData]:
        """
        Generate decisions based on perceptions
        
        Args:
            perceptions: Agent perceptions from Mesa
            
        Returns:
            Dict[agent_id, DecisionData]: Decisions mapped by agent ID
            
        Raises:
            ReasoningTimeoutError: When LLM calls timeout
            InvalidPerceptionError: When perception data is malformed
            LLMServiceError: When LLM service is unavailable
        """
        pass
    
    @abstractmethod
    def update_agent_memory(self, agent_id: str, experience: Dict[str, Any]) -> None:
        """
        Update agent memory with experience
        
        Args:
            agent_id: Target agent ID
            experience: Experience data to store
            
        Raises:
            MemoryUpdateError: When memory update fails
            AgentNotFoundError: When agent_id is invalid
        """
        pass
```

#### IActionTranslator
```python
@dataclass
class MesaAction:
    """Mesa-compatible action representation"""
    agent_id: str
    action_type: str  # move, interact, pickup, drop, communicate
    parameters: Dict[str, Any]  # action-specific parameters
    expected_duration: float  # estimated execution time
    prerequisites: List[str]  # required resources/conditions
    validation_rules: List[Callable] = field(default_factory=list)

class IActionTranslator(ABC):
    """Interface for translating CrewAI decisions to Mesa actions"""
    
    @abstractmethod
    def translate_decision(self, decision: DecisionData) -> MesaAction:
        """
        Translate single decision to Mesa action
        
        Args:
            decision: CrewAI decision data
            
        Returns:
            MesaAction: Mesa-compatible action
            
        Raises:
            UnsupportedActionError: When action type is not supported
            ParameterValidationError: When parameters are invalid
        """
        pass
    
    @abstractmethod
    def validate_action(self, action: MesaAction, mesa_model: mesa.Model) -> bool:
        """
        Validate action is legal in current Mesa state
        
        Args:
            action: Action to validate
            mesa_model: Current Mesa model state
            
        Returns:
            bool: True if action is valid
            
        Raises:
            ModelStateError: When model state is invalid
        """
        pass
    
    @abstractmethod
    def get_supported_actions(self) -> List[str]:
        """
        Get list of supported action types
        
        Returns:
            List[str]: Supported action type names
        """
        pass
```

#### IStateSynchronizer
```python
class IStateSynchronizer(ABC):
    """Interface for synchronizing state between frameworks"""
    
    @abstractmethod
    def sync_mesa_to_crewai(self, mesa_model: mesa.Model) -> None:
        """
        Sync Mesa state changes to CrewAI agents
        
        Args:
            mesa_model: Mesa model with updated state
            
        Raises:
            SyncError: When synchronization fails
            StateCorruptionError: When state is corrupted
        """
        pass
    
    @abstractmethod
    def sync_crewai_to_mesa(self, decisions: Dict[str, DecisionData], 
                          mesa_model: mesa.Model) -> None:
        """
        Sync CrewAI decisions to Mesa model
        
        Args:
            decisions: Agent decisions to apply
            mesa_model: Target Mesa model
            
        Raises:
            DecisionApplicationError: When decisions cannot be applied
            ConflictResolutionError: When conflicting decisions occur
        """
        pass
    
    @abstractmethod
    def validate_state_consistency(self, mesa_model: mesa.Model) -> bool:
        """
        Validate state consistency between frameworks
        
        Args:
            mesa_model: Mesa model to validate
            
        Returns:
            bool: True if state is consistent
        """
        pass
```

### 1.2 HybridSimulationEngine API

```python
class ComponentState(Enum):
    """Component lifecycle states"""
    UNINITIALIZED = "uninitialized"
    INITIALIZING = "initializing"
    READY = "ready"
    RUNNING = "running"
    PAUSED = "paused"
    ERROR = "error"
    SHUTDOWN = "shutdown"

class HybridSimulationEngine:
    """Main simulation engine coordinating Mesa and CrewAI"""
    
    def __init__(self, 
                 mesa_model: mesa.Model,
                 crewai_agents: List[Agent],
                 perception_pipeline: IPerceptionPipeline,
                 decision_engine: IDecisionEngine,
                 action_translator: IActionTranslator,
                 state_synchronizer: IStateSynchronizer,
                 config: 'HybridConfig' = None):
        """
        Initialize hybrid simulation engine
        
        Args:
            mesa_model: Mesa simulation model
            crewai_agents: List of CrewAI agents
            perception_pipeline: Perception extraction pipeline
            decision_engine: CrewAI decision engine
            action_translator: Decision to action translator
            state_synchronizer: State synchronization manager
            config: Optional configuration settings
            
        Raises:
            ConfigurationError: When configuration is invalid
            ComponentInitializationError: When components fail to initialize
        """
        pass
    
    def initialize(self) -> None:
        """
        Initialize hybrid simulation
        
        Raises:
            InitializationError: When initialization fails
            DependencyError: When dependencies are not available
        """
        pass
    
    async def step(self) -> Dict[str, Any]:
        """
        Execute one hybrid simulation step
        
        Returns:
            Dict containing step results:
            - step_number: int
            - duration: float (seconds)
            - actions_executed: int
            - mesa_state_summary: Dict
            - crewai_state_summary: Dict
            - errors: List[str]
            - performance_metrics: Dict
            
        Raises:
            SimulationError: When step execution fails
            StateInconsistencyError: When state becomes inconsistent
        """
        pass
    
    def pause(self) -> None:
        """Pause simulation execution"""
        pass
    
    def resume(self) -> None:
        """Resume paused simulation"""
        pass
    
    def shutdown(self) -> None:
        """Shutdown simulation and cleanup resources"""
        pass
    
    def get_simulation_state(self) -> Dict[str, Any]:
        """
        Get complete simulation state
        
        Returns:
            Dict containing:
            - engine_state: ComponentState
            - step_count: int
            - error_count: int
            - hybrid_agents: Dict[str, Dict]
            - mesa_summary: Dict
            - crewai_summary: Dict
            - performance_summary: Dict
        """
        pass
```

## 2. Data Models

### 2.1 Configuration Models

```python
from pydantic import BaseModel, Field, validator
from typing import Optional, Dict, Any, List
from enum import Enum

class LLMProvider(str, Enum):
    """Supported LLM providers"""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GEMINI = "gemini"
    OLLAMA = "ollama"

class HybridConfig(BaseModel):
    """Complete hybrid simulation configuration"""
    
    # Simulation settings
    max_steps: int = Field(default=1000, ge=1, le=10000)
    step_timeout: float = Field(default=30.0, ge=1.0, le=300.0)
    enable_async: bool = Field(default=True)
    
    # Mesa configuration
    mesa_config: 'MesaConfig'
    
    # CrewAI configuration
    crewai_config: 'CrewAIConfig'
    
    # Performance settings
    performance_config: 'PerformanceConfig' = Field(default_factory=lambda: PerformanceConfig())
    
    # Error handling settings
    error_config: 'ErrorConfig' = Field(default_factory=lambda: ErrorConfig())
    
    # Testing settings
    testing_config: Optional['TestingConfig'] = None

class MesaConfig(BaseModel):
    """Mesa-specific configuration"""
    
    # Environment settings
    width: int = Field(default=20, ge=5, le=100)
    height: int = Field(default=20, ge=5, le=100)
    torus: bool = Field(default=False)
    
    # Agent settings
    max_agents: int = Field(default=50, ge=1, le=1000)
    agent_vision_radius: float = Field(default=3.0, ge=0.5, le=10.0)
    
    # Physics settings
    enable_physics: bool = Field(default=True)
    collision_detection: bool = Field(default=True)
    
    # Resource settings
    resource_types: List[str] = Field(default_factory=lambda: ["energy", "materials", "information"])
    resource_regeneration: bool = Field(default=True)
    
    @validator('width', 'height')
    def validate_dimensions(cls, v):
        if v < 5 or v > 100:
            raise ValueError('Dimensions must be between 5 and 100')
        return v

class CrewAIConfig(BaseModel):
    """CrewAI-specific configuration"""
    
    # LLM settings
    llm_provider: LLMProvider = Field(default=LLMProvider.GEMINI)
    model_name: str = Field(default="gemini-2.5-flash-lite")
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    
    # Agent settings
    memory_enabled: bool = Field(default=True)
    memory_limit: int = Field(default=10000, ge=100, le=100000)
    verbose: bool = Field(default=False)
    
    # Reasoning settings
    max_reasoning_time: float = Field(default=10.0, ge=1.0, le=60.0)
    confidence_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    
    # Rate limiting
    max_requests_per_minute: int = Field(default=60, ge=1, le=1000)
    concurrent_requests: int = Field(default=5, ge=1, le=20)

class PerformanceConfig(BaseModel):
    """Performance optimization configuration"""
    
    # Caching settings
    enable_caching: bool = Field(default=True)
    cache_ttl: int = Field(default=300, ge=10, le=3600)  # seconds
    max_cache_size: int = Field(default=1000, ge=10, le=10000)
    
    # Batching settings
    enable_batching: bool = Field(default=True)
    batch_size: int = Field(default=10, ge=1, le=50)
    batch_timeout: float = Field(default=1.0, ge=0.1, le=10.0)
    
    # Resource limits
    max_memory_mb: int = Field(default=1024, ge=128, le=8192)
    max_cpu_percent: float = Field(default=80.0, ge=10.0, le=95.0)

class ErrorConfig(BaseModel):
    """Error handling configuration"""
    
    # Retry settings
    max_retries: int = Field(default=3, ge=0, le=10)
    retry_delay: float = Field(default=1.0, ge=0.1, le=60.0)
    exponential_backoff: bool = Field(default=True)
    
    # Circuit breaker settings
    failure_threshold: int = Field(default=5, ge=1, le=20)
    timeout_duration: int = Field(default=60, ge=10, le=600)
    success_threshold: int = Field(default=3, ge=1, le=10)
    
    # Fallback settings
    enable_fallbacks: bool = Field(default=True)
    fallback_timeout: float = Field(default=5.0, ge=1.0, le=30.0)

class TestingConfig(BaseModel):
    """Testing-specific configuration"""
    
    # Mock settings
    use_mock_llm: bool = Field(default=False)
    mock_latency: float = Field(default=0.1, ge=0.0, le=5.0)
    mock_failure_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    
    # Deterministic settings
    random_seed: Optional[int] = None
    deterministic_mode: bool = Field(default=False)
    
    # Validation settings
    strict_validation: bool = Field(default=True)
    validate_every_step: bool = Field(default=False)
```

### 2.2 Agent Models

```python
class HybridAgentRole(str, Enum):
    """Supported agent roles"""
    STRATEGIST = "strategist"
    MEDIATOR = "mediator"
    SURVIVOR = "survivor"
    EXPLORER = "explorer"
    GUARD = "guard"

class AgentPersonality(BaseModel):
    """Agent personality configuration"""
    
    role: HybridAgentRole
    cooperation_tendency: float = Field(default=0.5, ge=0.0, le=1.0)
    risk_tolerance: float = Field(default=0.5, ge=0.0, le=1.0)
    curiosity_level: float = Field(default=0.5, ge=0.0, le=1.0)
    leadership_inclination: float = Field(default=0.5, ge=0.0, le=1.0)
    analytical_depth: float = Field(default=0.5, ge=0.0, le=1.0)
    
    # Role-specific traits
    strategist_traits: Optional['StrategistTraits'] = None
    mediator_traits: Optional['MediatorTraits'] = None
    survivor_traits: Optional['SurvivorTraits'] = None

class StrategistTraits(BaseModel):
    """Strategist-specific personality traits"""
    analysis_time_preference: float = Field(default=0.8, ge=0.0, le=1.0)
    plan_complexity_preference: float = Field(default=0.7, ge=0.0, le=1.0)
    information_hoarding_tendency: float = Field(default=0.6, ge=0.0, le=1.0)

class MediatorTraits(BaseModel):
    """Mediator-specific personality traits"""
    conflict_resolution_skill: float = Field(default=0.9, ge=0.0, le=1.0)
    empathy_level: float = Field(default=0.8, ge=0.0, le=1.0)
    compromise_willingness: float = Field(default=0.9, ge=0.0, le=1.0)

class SurvivorTraits(BaseModel):
    """Survivor-specific personality traits"""
    pragmatism_level: float = Field(default=0.9, ge=0.0, le=1.0)
    self_preservation_priority: float = Field(default=0.8, ge=0.0, le=1.0)
    adaptability: float = Field(default=0.7, ge=0.0, le=1.0)

class HybridAgentConfig(BaseModel):
    """Configuration for a hybrid agent"""
    
    agent_id: str = Field(regex=r'^[a-zA-Z0-9_]+$')
    personality: AgentPersonality
    
    # Mesa agent settings
    initial_position: Optional[tuple] = None
    initial_energy: float = Field(default=100.0, ge=0.0, le=1000.0)
    max_health: float = Field(default=100.0, ge=1.0, le=1000.0)
    movement_speed: float = Field(default=1.0, ge=0.1, le=10.0)
    
    # CrewAI agent settings
    goal: Optional[str] = None
    backstory: Optional[str] = None
    custom_tools: List[str] = Field(default_factory=list)
    
    # Memory settings
    memory_capacity: int = Field(default=1000, ge=10, le=10000)
    forget_threshold: float = Field(default=0.1, ge=0.0, le=1.0)
```

### 2.3 State Models

```python
class StateType(str, Enum):
    """Types of state changes"""
    MESA_AGENT_MOVE = "mesa_agent_move"
    MESA_RESOURCE_UPDATE = "mesa_resource_update"
    MESA_ENVIRONMENT_CHANGE = "mesa_environment_change"
    CREWAI_MEMORY_UPDATE = "crewai_memory_update"
    CREWAI_DECISION_MADE = "crewai_decision_made"
    AGENT_INTERACTION = "agent_interaction"

class StateChange(BaseModel):
    """Represents a state change in the system"""
    
    change_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = Field(default_factory=datetime.now)
    change_type: StateType
    agent_id: Optional[str] = None
    
    # Change data
    previous_state: Dict[str, Any] = Field(default_factory=dict)
    new_state: Dict[str, Any] = Field(default_factory=dict)
    change_metadata: Dict[str, Any] = Field(default_factory=dict)
    
    # Validation
    is_valid: bool = Field(default=True)
    validation_errors: List[str] = Field(default_factory=list)

class UnifiedAgentState(BaseModel):
    """Unified state representation for a hybrid agent"""
    
    agent_id: str
    timestamp: datetime = Field(default_factory=datetime.now)
    
    # Mesa state
    mesa_state: 'MesaAgentState'
    
    # CrewAI state
    crewai_state: 'CrewAIAgentState'
    
    # Derived state
    derived_metrics: Dict[str, float] = Field(default_factory=dict)
    
    class Config:
        arbitrary_types_allowed = True

class MesaAgentState(BaseModel):
    """Mesa-specific agent state"""
    
    position: tuple
    velocity: Optional[tuple] = None
    health: float = Field(ge=0.0, le=1000.0)
    energy: float = Field(ge=0.0, le=1000.0)
    
    # Inventory
    inventory: Dict[str, int] = Field(default_factory=dict)
    inventory_capacity: int = Field(default=10, ge=1, le=100)
    
    # Status
    is_active: bool = Field(default=True)
    current_action: Optional[str] = None
    action_cooldown: float = Field(default=0.0, ge=0.0)
    
    # Relationships
    visible_agents: List[str] = Field(default_factory=list)
    interaction_history: List[Dict[str, Any]] = Field(default_factory=list)

class CrewAIAgentState(BaseModel):
    """CrewAI-specific agent state"""
    
    # Memory
    short_term_memory: List[Dict[str, Any]] = Field(default_factory=list)
    long_term_memory: List[Dict[str, Any]] = Field(default_factory=list)
    memory_usage: float = Field(default=0.0, ge=0.0, le=1.0)
    
    # Decision making
    last_decision: Optional[DecisionData] = None
    decision_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    reasoning_history: List[str] = Field(default_factory=list)
    
    # Performance
    llm_call_count: int = Field(default=0, ge=0)
    avg_response_time: float = Field(default=0.0, ge=0.0)
    error_count: int = Field(default=0, ge=0)
```

## 3. Interface Definitions

### 3.1 Abstract Base Classes

```python
from abc import ABC, abstractmethod
from typing import Protocol, runtime_checkable

@runtime_checkable
class StateValidator(Protocol):
    """Protocol for state validation"""
    
    def validate(self, state: Dict[str, Any]) -> bool:
        """Validate state consistency"""
        ...
    
    def get_validation_errors(self) -> List[str]:
        """Get list of validation errors"""
        ...

class BaseHybridComponent(ABC):
    """Base class for all hybrid architecture components"""
    
    def __init__(self, config: BaseModel):
        self.config = config
        self.state = ComponentState.UNINITIALIZED
        self.error_count = 0
        self.performance_metrics: Dict[str, float] = {}
    
    @abstractmethod
    async def initialize(self) -> None:
        """Initialize component"""
        pass
    
    @abstractmethod
    async def cleanup(self) -> None:
        """Cleanup component resources"""
        pass
    
    @abstractmethod
    def get_health_status(self) -> Dict[str, Any]:
        """Get component health status"""
        pass
    
    def update_metrics(self, metrics: Dict[str, float]) -> None:
        """Update performance metrics"""
        self.performance_metrics.update(metrics)

class BasePerceptionPipeline(BaseHybridComponent):
    """Base implementation for perception pipelines"""
    
    def __init__(self, config: PerformanceConfig):
        super().__init__(config)
        self.spatial_index: Optional['SpatialIndex'] = None
        self.perception_cache: Dict[str, Any] = {}
    
    @abstractmethod
    def build_spatial_index(self, mesa_model: mesa.Model) -> 'SpatialIndex':
        """Build spatial index for efficient queries"""
        pass
    
    def clear_cache(self) -> None:
        """Clear perception cache"""
        self.perception_cache.clear()

class BaseDecisionEngine(BaseHybridComponent):
    """Base implementation for decision engines"""
    
    def __init__(self, config: CrewAIConfig):
        super().__init__(config)
        self.llm_client: Optional[Any] = None
        self.circuit_breaker: Optional['CircuitBreaker'] = None
    
    @abstractmethod
    async def setup_llm_client(self) -> None:
        """Setup LLM client connection"""
        pass
    
    @abstractmethod
    async def format_prompt(self, agent_id: str, perception: PerceptionData) -> str:
        """Format perception data into LLM prompt"""
        pass
```

### 3.2 Event System Interfaces

```python
class EventType(str, Enum):
    """Types of events in the system"""
    AGENT_MOVED = "agent_moved"
    RESOURCE_COLLECTED = "resource_collected"
    DECISION_MADE = "decision_made"
    ERROR_OCCURRED = "error_occurred"
    SIMULATION_STARTED = "simulation_started"
    SIMULATION_ENDED = "simulation_ended"

class Event(BaseModel):
    """Base event class"""
    
    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = Field(default_factory=datetime.now)
    event_type: EventType
    source_component: str
    data: Dict[str, Any] = Field(default_factory=dict)

class IEventHandler(ABC):
    """Interface for event handlers"""
    
    @abstractmethod
    def can_handle(self, event: Event) -> bool:
        """Check if handler can process this event"""
        pass
    
    @abstractmethod
    async def handle_event(self, event: Event) -> None:
        """Handle the event"""
        pass
    
    @abstractmethod
    def get_priority(self) -> int:
        """Get handler priority (lower = higher priority)"""
        pass

class IEventBus(ABC):
    """Interface for event bus"""
    
    @abstractmethod
    def subscribe(self, event_type: EventType, handler: IEventHandler) -> None:
        """Subscribe handler to event type"""
        pass
    
    @abstractmethod
    def unsubscribe(self, event_type: EventType, handler: IEventHandler) -> None:
        """Unsubscribe handler from event type"""
        pass
    
    @abstractmethod
    async def publish(self, event: Event) -> None:
        """Publish event to subscribers"""
        pass
```

## 4. State Schema

### 4.1 Unified State Structure

```python
class HybridSimulationState(BaseModel):
    """Complete simulation state schema"""
    
    # Metadata
    simulation_id: str
    created_at: datetime
    last_updated: datetime
    version: str = Field(default="1.0.0")
    
    # Engine state
    engine_status: ComponentState
    current_step: int = Field(default=0, ge=0)
    total_elapsed_time: float = Field(default=0.0, ge=0.0)
    
    # Agent states
    agents: Dict[str, UnifiedAgentState] = Field(default_factory=dict)
    
    # Environment state
    environment: 'EnvironmentState'
    
    # Resources
    resources: Dict[str, 'ResourceState'] = Field(default_factory=dict)
    
    # Events
    recent_events: List[Event] = Field(default_factory=list, max_items=1000)
    
    # Performance metrics
    performance: 'PerformanceMetrics'
    
    # Validation
    state_hash: Optional[str] = None
    is_consistent: bool = Field(default=True)
    
    def compute_hash(self) -> str:
        """Compute state hash for integrity checking"""
        import hashlib
        import json
        
        # Create deterministic representation
        state_dict = self.dict(exclude={'state_hash', 'last_updated'})
        state_json = json.dumps(state_dict, sort_keys=True, default=str)
        return hashlib.sha256(state_json.encode()).hexdigest()
    
    def validate_consistency(self) -> List[str]:
        """Validate state consistency and return errors"""
        errors = []
        
        # Check agent position bounds
        for agent_id, agent_state in self.agents.items():
            pos = agent_state.mesa_state.position
            if not self.environment.is_position_valid(pos):
                errors.append(f"Agent {agent_id} position {pos} is out of bounds")
        
        # Check resource conservation
        total_resources = sum(r.quantity for r in self.resources.values())
        agent_resources = sum(
            sum(agent.mesa_state.inventory.values()) 
            for agent in self.agents.values()
        )
        expected_total = self.environment.initial_resource_count
        
        if abs(total_resources + agent_resources - expected_total) > 0.001:
            errors.append(f"Resource conservation violated: {total_resources + agent_resources} != {expected_total}")
        
        return errors

class EnvironmentState(BaseModel):
    """Environment state representation"""
    
    # Spatial properties
    width: int = Field(ge=1, le=1000)
    height: int = Field(ge=1, le=1000)
    topology: str = Field(default="grid")  # grid, continuous, network
    
    # Environmental conditions
    temperature: float = Field(default=20.0, ge=-50.0, le=100.0)
    lighting: float = Field(default=1.0, ge=0.0, le=1.0)
    hazard_zones: List[Dict[str, Any]] = Field(default_factory=list)
    
    # Resource distribution
    resource_spawn_points: Dict[str, List[tuple]] = Field(default_factory=dict)
    initial_resource_count: int = Field(default=0, ge=0)
    
    # Obstacles and barriers
    obstacles: List[Dict[str, Any]] = Field(default_factory=list)
    movement_costs: Dict[tuple, float] = Field(default_factory=dict)
    
    def is_position_valid(self, position: tuple) -> bool:
        """Check if position is within environment bounds"""
        x, y = position
        return 0 <= x < self.width and 0 <= y < self.height
    
    def get_neighbors(self, position: tuple, radius: float = 1.0) -> List[tuple]:
        """Get neighboring positions within radius"""
        x, y = position
        neighbors = []
        
        for dx in range(-int(radius), int(radius) + 1):
            for dy in range(-int(radius), int(radius) + 1):
                new_pos = (x + dx, y + dy)
                if (self.is_position_valid(new_pos) and 
                    (dx*dx + dy*dy) <= radius*radius):
                    neighbors.append(new_pos)
        
        return neighbors

class ResourceState(BaseModel):
    """Individual resource state"""
    
    resource_id: str
    resource_type: str
    position: tuple
    quantity: float = Field(ge=0.0)
    max_quantity: float = Field(ge=0.0)
    
    # Properties
    is_renewable: bool = Field(default=False)
    regeneration_rate: float = Field(default=0.0, ge=0.0)
    last_accessed: Optional[datetime] = None
    
    # Access control
    owned_by: Optional[str] = None
    access_restrictions: List[str] = Field(default_factory=list)
    
    def can_access(self, agent_id: str) -> bool:
        """Check if agent can access this resource"""
        if self.owned_by and self.owned_by != agent_id:
            return False
        return agent_id not in self.access_restrictions

class PerformanceMetrics(BaseModel):
    """Performance metrics for the simulation"""
    
    # Timing metrics
    avg_step_duration: float = Field(default=0.0, ge=0.0)
    min_step_duration: float = Field(default=0.0, ge=0.0)
    max_step_duration: float = Field(default=0.0, ge=0.0)
    
    # LLM metrics
    avg_llm_latency: float = Field(default=0.0, ge=0.0)
    llm_success_rate: float = Field(default=1.0, ge=0.0, le=1.0)
    total_llm_calls: int = Field(default=0, ge=0)
    
    # Memory metrics
    memory_usage_mb: float = Field(default=0.0, ge=0.0)
    peak_memory_mb: float = Field(default=0.0, ge=0.0)
    
    # Error metrics
    total_errors: int = Field(default=0, ge=0)
    error_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    
    # Throughput metrics
    actions_per_second: float = Field(default=0.0, ge=0.0)
    decisions_per_second: float = Field(default=0.0, ge=0.0)
```

## 5. Event System Design

### 5.1 Event Types and Payloads

```python
class AgentMovedEvent(Event):
    """Event fired when agent moves"""
    
    def __init__(self, agent_id: str, old_position: tuple, new_position: tuple):
        super().__init__(
            event_type=EventType.AGENT_MOVED,
            source_component="mesa_agent",
            data={
                "agent_id": agent_id,
                "old_position": old_position,
                "new_position": new_position,
                "distance": ((new_position[0] - old_position[0])**2 + 
                           (new_position[1] - old_position[1])**2)**0.5
            }
        )

class DecisionMadeEvent(Event):
    """Event fired when agent makes decision"""
    
    def __init__(self, decision: DecisionData):
        super().__init__(
            event_type=EventType.DECISION_MADE,
            source_component="crewai_agent",
            data={
                "agent_id": decision.agent_id,
                "action": decision.chosen_action,
                "confidence": decision.confidence_level,
                "reasoning_length": len(decision.reasoning),
                "fallback_count": len(decision.fallback_actions)
            }
        )

class ResourceCollectedEvent(Event):
    """Event fired when resource is collected"""
    
    def __init__(self, agent_id: str, resource_id: str, quantity: float):
        super().__init__(
            event_type=EventType.RESOURCE_COLLECTED,
            source_component="mesa_model",
            data={
                "agent_id": agent_id,
                "resource_id": resource_id,
                "quantity": quantity
            }
        )

class ErrorOccurredEvent(Event):
    """Event fired when error occurs"""
    
    def __init__(self, component: str, error_type: str, error_message: str, severity: str):
        super().__init__(
            event_type=EventType.ERROR_OCCURRED,
            source_component=component,
            data={
                "error_type": error_type,
                "error_message": error_message,
                "severity": severity
            }
        )
```

### 5.2 Event Bus Implementation

```python
import asyncio
from collections import defaultdict
from typing import Set

class AsyncEventBus(IEventBus):
    """Asynchronous event bus implementation"""
    
    def __init__(self):
        self._handlers: Dict[EventType, List[IEventHandler]] = defaultdict(list)
        self._event_queue: asyncio.Queue = asyncio.Queue()
        self._processing_task: Optional[asyncio.Task] = None
        self._is_running = False
        
        # Metrics
        self.events_published = 0
        self.events_processed = 0
        self.processing_errors = 0
    
    async def start(self) -> None:
        """Start event processing"""
        if self._is_running:
            return
        
        self._is_running = True
        self._processing_task = asyncio.create_task(self._process_events())
    
    async def stop(self) -> None:
        """Stop event processing"""
        self._is_running = False
        if self._processing_task:
            self._processing_task.cancel()
            try:
                await self._processing_task
            except asyncio.CancelledError:
                pass
    
    def subscribe(self, event_type: EventType, handler: IEventHandler) -> None:
        """Subscribe handler to event type"""
        handlers = self._handlers[event_type]
        if handler not in handlers:
            handlers.append(handler)
            # Sort by priority
            handlers.sort(key=lambda h: h.get_priority())
    
    def unsubscribe(self, event_type: EventType, handler: IEventHandler) -> None:
        """Unsubscribe handler from event type"""
        handlers = self._handlers[event_type]
        if handler in handlers:
            handlers.remove(handler)
    
    async def publish(self, event: Event) -> None:
        """Publish event to queue"""
        await self._event_queue.put(event)
        self.events_published += 1
    
    async def _process_events(self) -> None:
        """Process events from queue"""
        while self._is_running:
            try:
                # Wait for event with timeout
                event = await asyncio.wait_for(
                    self._event_queue.get(), 
                    timeout=1.0
                )
                
                await self._handle_event(event)
                self.events_processed += 1
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                self.processing_errors += 1
                # Log error but continue processing
                print(f"Event processing error: {e}")
    
    async def _handle_event(self, event: Event) -> None:
        """Handle single event"""
        handlers = self._handlers.get(event.event_type, [])
        
        # Process handlers in priority order
        for handler in handlers:
            try:
                if handler.can_handle(event):
                    await handler.handle_event(event)
            except Exception as e:
                # Log handler error but continue with other handlers
                print(f"Handler error for {handler.__class__.__name__}: {e}")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get event bus metrics"""
        return {
            "events_published": self.events_published,
            "events_processed": self.events_processed,
            "processing_errors": self.processing_errors,
            "queue_size": self._event_queue.qsize(),
            "handler_count": sum(len(handlers) for handlers in self._handlers.values())
        }
```

## 6. Configuration Schema

### 6.1 Runtime Configuration

```python
class RuntimeConfig(BaseModel):
    """Runtime configuration for adaptive behavior"""
    
    # Dynamic scaling
    auto_scale_agents: bool = Field(default=False)
    min_agents: int = Field(default=1, ge=1, le=10)
    max_agents: int = Field(default=100, ge=1, le=1000)
    
    # Performance adaptation
    adaptive_batching: bool = Field(default=True)
    performance_target_ms: float = Field(default=500.0, ge=100.0, le=5000.0)
    
    # Memory management
    auto_memory_cleanup: bool = Field(default=True)
    memory_pressure_threshold: float = Field(default=0.8, ge=0.5, le=0.95)
    
    # Error handling adaptation
    adaptive_retry_delays: bool = Field(default=True)
    escalation_threshold: int = Field(default=10, ge=1, le=100)

class DeploymentConfig(BaseModel):
    """Deployment-specific configuration"""
    
    # Environment
    environment: str = Field(default="development")  # development, staging, production
    debug_mode: bool = Field(default=False)
    profiling_enabled: bool = Field(default=False)
    
    # Monitoring
    metrics_endpoint: Optional[str] = None
    health_check_port: int = Field(default=8080, ge=1024, le=65535)
    
    # Security
    api_key_rotation: bool = Field(default=True)
    rate_limiting_enabled: bool = Field(default=True)
    request_validation: bool = Field(default=True)
    
    # Persistence
    state_persistence: bool = Field(default=True)
    checkpoint_interval: int = Field(default=100, ge=1, le=10000)  # steps
    backup_retention: int = Field(default=10, ge=1, le=100)  # checkpoints

class LoggingConfig(BaseModel):
    """Logging configuration"""
    
    level: str = Field(default="INFO")
    format_style: str = Field(default="json")  # json, structured, simple
    
    # Log destinations
    console_enabled: bool = Field(default=True)
    file_enabled: bool = Field(default=True)
    file_path: str = Field(default="logs/hybrid_simulation.log")
    
    # Rotation
    max_file_size: str = Field(default="100MB")
    backup_count: int = Field(default=5, ge=1, le=20)
    
    # Filtering
    component_filters: Dict[str, str] = Field(default_factory=dict)
    sensitive_data_masking: bool = Field(default=True)
```

### 6.2 Validation Rules

```python
from pydantic import root_validator

class ConfigValidator:
    """Configuration validation utilities"""
    
    @staticmethod
    def validate_llm_config(config: CrewAIConfig) -> List[str]:
        """Validate LLM configuration"""
        errors = []
        
        if config.llm_provider == LLMProvider.OPENAI and not config.api_key:
            errors.append("OpenAI API key is required")
        
        if config.max_requests_per_minute > 1000:
            errors.append("Requests per minute too high, may hit rate limits")
        
        if config.concurrent_requests > config.max_requests_per_minute / 10:
            errors.append("Concurrent requests too high relative to rate limit")
        
        return errors
    
    @staticmethod
    def validate_mesa_config(config: MesaConfig) -> List[str]:
        """Validate Mesa configuration"""
        errors = []
        
        if config.width * config.height > 10000:
            errors.append("Environment too large, may cause performance issues")
        
        if config.max_agents > config.width * config.height:
            errors.append("More agents than available positions")
        
        if config.agent_vision_radius > min(config.width, config.height) / 2:
            errors.append("Agent vision radius too large for environment")
        
        return errors
    
    @staticmethod
    def validate_performance_config(config: PerformanceConfig) -> List[str]:
        """Validate performance configuration"""
        errors = []
        
        if config.batch_size > 50:
            errors.append("Batch size too large, may cause memory issues")
        
        if config.cache_ttl < 10:
            errors.append("Cache TTL too low, may cause excessive cache misses")
        
        if config.max_memory_mb < 128:
            errors.append("Memory limit too low for hybrid simulation")
        
        return errors

class HybridConfigValidator:
    """Main configuration validator"""
    
    @root_validator
    def validate_hybrid_config(cls, values):
        """Validate complete hybrid configuration"""
        errors = []
        
        # Validate individual components
        if 'mesa_config' in values:
            errors.extend(ConfigValidator.validate_mesa_config(values['mesa_config']))
        
        if 'crewai_config' in values:
            errors.extend(ConfigValidator.validate_llm_config(values['crewai_config']))
        
        if 'performance_config' in values:
            errors.extend(ConfigValidator.validate_performance_config(values['performance_config']))
        
        # Cross-component validation
        mesa_config = values.get('mesa_config')
        crewai_config = values.get('crewai_config')
        
        if mesa_config and crewai_config:
            # Ensure LLM can handle the agent load
            max_agents = mesa_config.max_agents
            max_requests = crewai_config.max_requests_per_minute
            
            if max_agents * 2 > max_requests:  # Assume 2 LLM calls per agent per minute
                errors.append("LLM rate limit insufficient for agent count")
        
        if errors:
            raise ValueError(f"Configuration validation failed: {'; '.join(errors)}")
        
        return values
```

## 7. Error Taxonomies

### 7.1 Error Classification System

```python
class ErrorSeverity(IntEnum):
    """Error severity levels with numeric ordering"""
    LOW = 1      # Non-critical, system continues normally
    MEDIUM = 2   # Some degradation, fallback mechanisms activated
    HIGH = 3     # Significant impact, major fallbacks required
    CRITICAL = 4 # System stability threatened, emergency protocols

class ErrorCategory(str, Enum):
    """Comprehensive error categories"""
    
    # LLM-related errors
    LLM_API_ERROR = "llm_api_error"
    LLM_TIMEOUT = "llm_timeout"
    LLM_RATE_LIMIT = "llm_rate_limit"
    LLM_INVALID_RESPONSE = "llm_invalid_response"
    LLM_AUTHENTICATION = "llm_authentication"
    
    # Mesa-related errors
    MESA_AGENT_ERROR = "mesa_agent_error"
    MESA_MODEL_ERROR = "mesa_model_error"
    MESA_SCHEDULER_ERROR = "mesa_scheduler_error"
    MESA_SPACE_ERROR = "mesa_space_error"
    
    # CrewAI-related errors
    CREWAI_AGENT_ERROR = "crewai_agent_error"
    CREWAI_TASK_ERROR = "crewai_task_error"
    CREWAI_CREW_ERROR = "crewai_crew_error"
    CREWAI_MEMORY_ERROR = "crewai_memory_error"
    
    # Integration errors
    STATE_SYNC_ERROR = "state_sync_error"
    PERCEPTION_ERROR = "perception_error"
    DECISION_TRANSLATION_ERROR = "decision_translation_error"
    ACTION_VALIDATION_ERROR = "action_validation_error"
    
    # System errors
    NETWORK_ERROR = "network_error"
    MEMORY_ERROR = "memory_error"
    DISK_ERROR = "disk_error"
    CONFIGURATION_ERROR = "configuration_error"
    
    # Validation errors
    DATA_VALIDATION_ERROR = "data_validation_error"
    SCHEMA_VALIDATION_ERROR = "schema_validation_error"
    CONSTRAINT_VIOLATION = "constraint_violation"
    
    # Performance errors
    TIMEOUT_ERROR = "timeout_error"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    PERFORMANCE_DEGRADATION = "performance_degradation"

class ErrorContext(BaseModel):
    """Comprehensive error context"""
    
    # Basic information
    error_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = Field(default_factory=datetime.now)
    category: ErrorCategory
    severity: ErrorSeverity
    
    # Error details
    component: str
    operation: str
    error_type: str
    error_message: str
    stack_trace: Optional[str] = None
    
    # Context data
    agent_id: Optional[str] = None
    step_number: Optional[int] = None
    simulation_state: Optional[Dict[str, Any]] = None
    
    # Recovery information
    retry_count: int = Field(default=0, ge=0)
    recovery_attempted: bool = Field(default=False)
    recovery_strategy: Optional[str] = None
    
    # Impact assessment
    affected_components: List[str] = Field(default_factory=list)
    expected_recovery_time: Optional[float] = None
    user_impact: str = Field(default="none")  # none, low, medium, high
    
    # Metadata
    correlation_id: Optional[str] = None
    parent_error_id: Optional[str] = None  # For cascading errors
    tags: List[str] = Field(default_factory=list)

class RecoveryStrategy(BaseModel):
    """Recovery strategy specification"""
    
    strategy_name: str
    description: str
    applicable_categories: List[ErrorCategory]
    max_retry_attempts: int = Field(default=3, ge=0, le=10)
    base_delay: float = Field(default=1.0, ge=0.1, le=60.0)
    escalation_threshold: int = Field(default=5, ge=1, le=20)
    
    # Strategy-specific parameters
    parameters: Dict[str, Any] = Field(default_factory=dict)
    
    # Success criteria
    success_indicators: List[str] = Field(default_factory=list)
    failure_indicators: List[str] = Field(default_factory=list)
    
    def is_applicable(self, error_context: ErrorContext) -> bool:
        """Check if strategy is applicable to error"""
        return error_context.category in self.applicable_categories
    
    def should_escalate(self, retry_count: int) -> bool:
        """Check if error should be escalated"""
        return retry_count >= self.escalation_threshold
```

### 7.2 Error Handler Specifications

```python
class IErrorHandler(ABC):
    """Base interface for error handlers"""
    
    @abstractmethod
    def can_handle(self, error_context: ErrorContext) -> bool:
        """Check if handler can process this error"""
        pass
    
    @abstractmethod
    async def handle_error(self, error_context: ErrorContext) -> 'RecoveryResult':
        """Handle the error and attempt recovery"""
        pass
    
    @abstractmethod
    def get_priority(self) -> int:
        """Get handler priority (lower = higher priority)"""
        pass
    
    @abstractmethod
    def get_supported_categories(self) -> List[ErrorCategory]:
        """Get list of supported error categories"""
        pass

class RecoveryResult(BaseModel):
    """Result of error recovery attempt"""
    
    success: bool
    strategy_used: str
    message: str
    execution_time: float = Field(ge=0.0)
    
    # Recovery data
    fallback_data: Optional[Any] = None
    corrective_actions: List[str] = Field(default_factory=list)
    
    # Future prevention
    prevention_recommendations: List[str] = Field(default_factory=list)
    monitoring_suggestions: List[str] = Field(default_factory=list)
    
    # Metadata
    recovery_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    context: Dict[str, Any] = Field(default_factory=dict)

class LLMErrorHandler(IErrorHandler):
    """Specialized handler for LLM-related errors"""
    
    def __init__(self, circuit_breaker: 'CircuitBreaker', 
                 fallback_generator: 'FallbackResponseGenerator'):
        self.circuit_breaker = circuit_breaker
        self.fallback_generator = fallback_generator
        self.retry_config = RetryConfig(max_attempts=3, base_delay=2.0)
    
    def can_handle(self, error_context: ErrorContext) -> bool:
        """Check if this is an LLM-related error"""
        llm_categories = [
            ErrorCategory.LLM_API_ERROR,
            ErrorCategory.LLM_TIMEOUT,
            ErrorCategory.LLM_RATE_LIMIT,
            ErrorCategory.LLM_INVALID_RESPONSE,
            ErrorCategory.LLM_AUTHENTICATION
        ]
        return error_context.category in llm_categories
    
    async def handle_error(self, error_context: ErrorContext) -> RecoveryResult:
        """Handle LLM error with appropriate strategy"""
        
        if error_context.category == ErrorCategory.LLM_RATE_LIMIT:
            return await self._handle_rate_limit(error_context)
        elif error_context.category == ErrorCategory.LLM_TIMEOUT:
            return await self._handle_timeout(error_context)
        elif error_context.category == ErrorCategory.LLM_INVALID_RESPONSE:
            return await self._handle_invalid_response(error_context)
        else:
            return await self._handle_generic_llm_error(error_context)
    
    async def _handle_rate_limit(self, error_context: ErrorContext) -> RecoveryResult:
        """Handle rate limit errors with exponential backoff"""
        delay = self.retry_config.get_delay(error_context.retry_count + 1)
        await asyncio.sleep(delay)
        
        return RecoveryResult(
            success=True,
            strategy_used="exponential_backoff",
            message=f"Applied {delay}s delay for rate limit",
            execution_time=delay
        )
    
    async def _handle_timeout(self, error_context: ErrorContext) -> RecoveryResult:
        """Handle timeout errors with fallback response"""
        fallback_response = await self.fallback_generator.generate_fallback(
            error_context.agent_id,
            error_context.operation
        )
        
        return RecoveryResult(
            success=True,
            strategy_used="fallback_response",
            message="Generated fallback response for timeout",
            fallback_data=fallback_response,
            execution_time=0.1
        )
    
    def get_priority(self) -> int:
        return 1  # High priority for LLM errors
    
    def get_supported_categories(self) -> List[ErrorCategory]:
        return [
            ErrorCategory.LLM_API_ERROR,
            ErrorCategory.LLM_TIMEOUT,
            ErrorCategory.LLM_RATE_LIMIT,
            ErrorCategory.LLM_INVALID_RESPONSE,
            ErrorCategory.LLM_AUTHENTICATION
        ]
```

## 8. Test Specifications

### 8.1 Test Categories and Requirements

```python
class TestCategory(str, Enum):
    """Test categories for comprehensive coverage"""
    UNIT = "unit"                    # Individual component testing
    INTEGRATION = "integration"     # Component interaction testing
    SYSTEM = "system"               # Full system testing
    PERFORMANCE = "performance"     # Performance and load testing
    CHAOS = "chaos"                 # Fault injection testing
    REGRESSION = "regression"       # Regression testing
    ACCEPTANCE = "acceptance"       # User acceptance testing

class TestRequirement(BaseModel):
    """Specification for a test requirement"""
    
    requirement_id: str
    category: TestCategory
    description: str
    priority: int = Field(ge=1, le=5)  # 1 = highest
    
    # Test criteria
    success_criteria: List[str]
    failure_criteria: List[str]
    
    # Test data
    test_inputs: Dict[str, Any] = Field(default_factory=dict)
    expected_outputs: Dict[str, Any] = Field(default_factory=dict)
    
    # Execution parameters
    timeout_seconds: float = Field(default=30.0, ge=1.0, le=300.0)
    max_retries: int = Field(default=3, ge=0, le=10)
    
    # Dependencies
    dependencies: List[str] = Field(default_factory=list)
    prerequisites: List[str] = Field(default_factory=list)

class TestSuite(BaseModel):
    """Test suite specification"""
    
    suite_name: str
    category: TestCategory
    requirements: List[TestRequirement]
    
    # Execution settings
    parallel_execution: bool = Field(default=True)
    max_parallel_tests: int = Field(default=10, ge=1, le=50)
    
    # Coverage requirements
    min_code_coverage: float = Field(default=0.8, ge=0.0, le=1.0)
    min_branch_coverage: float = Field(default=0.7, ge=0.0, le=1.0)
    
    # Environment
    required_resources: List[str] = Field(default_factory=list)
    setup_scripts: List[str] = Field(default_factory=list)
    teardown_scripts: List[str] = Field(default_factory=list)
```

### 8.2 Deterministic Testing Framework

```python
class DeterministicTestConfig(BaseModel):
    """Configuration for deterministic testing"""
    
    # Randomization control
    global_seed: int = Field(default=42)
    per_test_seeds: Dict[str, int] = Field(default_factory=dict)
    
    # Mock configuration
    use_deterministic_mocks: bool = Field(default=True)
    mock_response_latency: float = Field(default=0.1, ge=0.0, le=5.0)
    mock_failure_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    
    # Time control
    freeze_time: bool = Field(default=True)
    mock_time_progression: bool = Field(default=True)
    time_step_ms: int = Field(default=100, ge=1, le=10000)
    
    # State validation
    validate_state_each_step: bool = Field(default=True)
    record_state_history: bool = Field(default=True)
    max_state_history: int = Field(default=1000, ge=10, le=10000)

class MockLLMProvider:
    """Deterministic mock LLM provider for testing"""
    
    def __init__(self, config: DeterministicTestConfig):
        self.config = config
        self.random = random.Random(config.global_seed)
        
        # Response patterns
        self.response_patterns: Dict[str, List[str]] = {}
        self.sequential_responses: List[str] = []
        self.default_responses: List[str] = [
            "I need to analyze this situation carefully.",
            "Let me consider the available options.",
            "I should communicate with my team members.",
            "The best strategy is to gather more information."
        ]
        
        # Call tracking
        self.call_history: List[Dict[str, Any]] = []
        self.call_count = 0
    
    def add_response_pattern(self, pattern: str, responses: List[str]) -> None:
        """Add deterministic response pattern"""
        self.response_patterns[pattern] = responses
    
    def set_sequential_responses(self, responses: List[str]) -> None:
        """Set responses to return in sequence"""
        self.sequential_responses = responses.copy()
    
    async def complete(self, prompt: str, **kwargs) -> str:
        """Generate deterministic response"""
        self.call_count += 1
        
        # Simulate latency
        await asyncio.sleep(self.config.mock_response_latency)
        
        # Find appropriate response
        response = self._find_response(prompt)
        
        # Record call
        self.call_history.append({
            "call_id": self.call_count,
            "timestamp": datetime.now(),
            "prompt": prompt,
            "response": response,
            "kwargs": kwargs
        })
        
        return response
    
    def _find_response(self, prompt: str) -> str:
        """Find deterministic response for prompt"""
        # Use sequential responses first
        if self.sequential_responses:
            return self.sequential_responses.pop(0)
        
        # Check pattern matches
        for pattern, responses in self.response_patterns.items():
            if pattern.lower() in prompt.lower():
                return self.random.choice(responses)
        
        # Use default responses
        return self.random.choice(self.default_responses)
    
    def reset(self) -> None:
        """Reset mock state"""
        self.call_history.clear()
        self.call_count = 0
        self.sequential_responses.clear()

class HybridTestHarness:
    """Comprehensive test harness for hybrid architecture"""
    
    def __init__(self, config: DeterministicTestConfig):
        self.config = config
        self.mock_llm = MockLLMProvider(config)
        self.test_results: List[Dict[str, Any]] = []
        self.state_validator = StateValidator()
    
    async def run_test_suite(self, suite: TestSuite) -> Dict[str, Any]:
        """Run complete test suite"""
        start_time = datetime.now()
        results = {
            "suite_name": suite.suite_name,
            "category": suite.category.value,
            "start_time": start_time,
            "test_results": [],
            "summary": {
                "total_tests": len(suite.requirements),
                "passed": 0,
                "failed": 0,
                "errors": 0,
                "skipped": 0
            }
        }
        
        # Run tests
        for requirement in suite.requirements:
            test_result = await self._run_single_test(requirement)
            results["test_results"].append(test_result)
            
            # Update summary
            status = test_result["status"]
            if status == "PASSED":
                results["summary"]["passed"] += 1
            elif status == "FAILED":
                results["summary"]["failed"] += 1
            elif status == "ERROR":
                results["summary"]["errors"] += 1
            else:
                results["summary"]["skipped"] += 1
        
        # Calculate metrics
        end_time = datetime.now()
        results["end_time"] = end_time
        results["duration"] = (end_time - start_time).total_seconds()
        results["success_rate"] = results["summary"]["passed"] / results["summary"]["total_tests"]
        
        return results
    
    async def _run_single_test(self, requirement: TestRequirement) -> Dict[str, Any]:
        """Run single test requirement"""
        test_start = datetime.now()
        
        try:
            # Setup test environment
            await self._setup_test_environment(requirement)
            
            # Execute test
            test_result = await self._execute_test(requirement)
            
            # Validate results
            validation_result = self._validate_test_result(requirement, test_result)
            
            status = "PASSED" if validation_result["valid"] else "FAILED"
            
            return {
                "requirement_id": requirement.requirement_id,
                "status": status,
                "start_time": test_start,
                "end_time": datetime.now(),
                "duration": (datetime.now() - test_start).total_seconds(),
                "result": test_result,
                "validation": validation_result
            }
            
        except Exception as e:
            return {
                "requirement_id": requirement.requirement_id,
                "status": "ERROR",
                "start_time": test_start,
                "end_time": datetime.now(),
                "duration": (datetime.now() - test_start).total_seconds(),
                "error": str(e),
                "traceback": traceback.format_exc()
            }
    
    async def _setup_test_environment(self, requirement: TestRequirement) -> None:
        """Setup test environment for requirement"""
        # Set random seed
        if requirement.requirement_id in self.config.per_test_seeds:
            seed = self.config.per_test_seeds[requirement.requirement_id]
        else:
            seed = self.config.global_seed
        
        random.seed(seed)
        
        # Reset mock components
        self.mock_llm.reset()
        
        # Configure mocks based on test inputs
        if "llm_responses" in requirement.test_inputs:
            responses = requirement.test_inputs["llm_responses"]
            self.mock_llm.set_sequential_responses(responses)
    
    async def _execute_test(self, requirement: TestRequirement) -> Dict[str, Any]:
        """Execute individual test"""
        # This would be implemented based on the specific test type
        # For now, return a mock result
        return {
            "executed": True,
            "requirement_id": requirement.requirement_id,
            "test_data": requirement.test_inputs
        }
    
    def _validate_test_result(self, requirement: TestRequirement, 
                            result: Dict[str, Any]) -> Dict[str, Any]:
        """Validate test result against success criteria"""
        validation = {
            "valid": True,
            "passed_criteria": [],
            "failed_criteria": [],
            "errors": []
        }
        
        # Check success criteria
        for criterion in requirement.success_criteria:
            try:
                if self._evaluate_criterion(criterion, result):
                    validation["passed_criteria"].append(criterion)
                else:
                    validation["valid"] = False
                    validation["failed_criteria"].append(criterion)
            except Exception as e:
                validation["valid"] = False
                validation["errors"].append(f"Criterion '{criterion}' evaluation error: {e}")
        
        return validation
    
    def _evaluate_criterion(self, criterion: str, result: Dict[str, Any]) -> bool:
        """Evaluate a single success criterion"""
        # Simple criterion evaluation - would be expanded for real implementation
        if "executed" in criterion:
            return result.get("executed", False)
        elif "no_errors" in criterion:
            return "error" not in result
        else:
            return True  # Default to pass for unknown criteria
```

## 9. Performance Benchmarks

### 9.1 Performance Targets

```python
class PerformanceBenchmark(BaseModel):
    """Performance benchmark specification"""
    
    benchmark_name: str
    category: str  # latency, throughput, memory, cpu
    
    # Targets
    target_value: float
    acceptable_value: float
    critical_threshold: float
    
    # Measurement
    unit: str  # ms, ops/sec, MB, %
    measurement_method: str
    sample_size: int = Field(default=100, ge=10, le=10000)
    
    # Conditions
    test_conditions: Dict[str, Any] = Field(default_factory=dict)
    environment_requirements: List[str] = Field(default_factory=list)

class HybridPerformanceTargets:
    """Performance targets for hybrid architecture"""
    
    # Latency benchmarks (milliseconds)
    SIMULATION_STEP_LATENCY = PerformanceBenchmark(
        benchmark_name="simulation_step_latency",
        category="latency",
        target_value=500.0,      # 500ms target
        acceptable_value=1000.0, # 1s acceptable
        critical_threshold=5000.0, # 5s critical
        unit="ms",
        measurement_method="end_to_end_timing",
        test_conditions={
            "agent_count": 10,
            "environment_size": "20x20",
            "llm_provider": "gemini"
        }
    )
    
    LLM_RESPONSE_LATENCY = PerformanceBenchmark(
        benchmark_name="llm_response_latency",
        category="latency",
        target_value=2000.0,     # 2s target
        acceptable_value=5000.0, # 5s acceptable
        critical_threshold=15000.0, # 15s critical
        unit="ms",
        measurement_method="llm_api_timing"
    )
    
    STATE_SYNC_LATENCY = PerformanceBenchmark(
        benchmark_name="state_sync_latency",
        category="latency",
        target_value=10.0,       # 10ms target
        acceptable_value=50.0,   # 50ms acceptable
        critical_threshold=200.0, # 200ms critical
        unit="ms",
        measurement_method="sync_operation_timing"
    )
    
    # Throughput benchmarks (operations per second)
    DECISION_THROUGHPUT = PerformanceBenchmark(
        benchmark_name="decision_throughput",
        category="throughput",
        target_value=10.0,       # 10 decisions/sec
        acceptable_value=5.0,    # 5 decisions/sec acceptable
        critical_threshold=1.0,  # 1 decision/sec critical
        unit="decisions/sec",
        measurement_method="decision_rate_measurement"
    )
    
    ACTION_PROCESSING_THROUGHPUT = PerformanceBenchmark(
        benchmark_name="action_processing_throughput",
        category="throughput",
        target_value=100.0,      # 100 actions/sec
        acceptable_value=50.0,   # 50 actions/sec acceptable
        critical_threshold=10.0, # 10 actions/sec critical
        unit="actions/sec",
        measurement_method="action_rate_measurement"
    )
    
    # Memory benchmarks (megabytes)
    MEMORY_USAGE = PerformanceBenchmark(
        benchmark_name="memory_usage",
        category="memory",
        target_value=256.0,      # 256MB target
        acceptable_value=512.0,  # 512MB acceptable
        critical_threshold=1024.0, # 1GB critical
        unit="MB",
        measurement_method="process_memory_measurement"
    )
    
    # Cache performance
    CACHE_HIT_RATE = PerformanceBenchmark(
        benchmark_name="cache_hit_rate",
        category="cache",
        target_value=0.8,        # 80% target
        acceptable_value=0.6,    # 60% acceptable
        critical_threshold=0.3,  # 30% critical
        unit="ratio",
        measurement_method="cache_statistics"
    )

class PerformanceProfiler:
    """Performance profiling and measurement"""
    
    def __init__(self):
        self.measurements: Dict[str, List[float]] = defaultdict(list)
        self.start_times: Dict[str, float] = {}
        self.memory_snapshots: List[Dict[str, Any]] = []
    
    @contextmanager
    def measure_latency(self, operation_name: str):
        """Context manager for latency measurement"""
        start_time = time.time()
        try:
            yield
        finally:
            end_time = time.time()
            latency_ms = (end_time - start_time) * 1000
            self.measurements[operation_name].append(latency_ms)
    
    def start_timer(self, operation_name: str) -> None:
        """Start timing an operation"""
        self.start_times[operation_name] = time.time()
    
    def end_timer(self, operation_name: str) -> float:
        """End timing and record measurement"""
        if operation_name not in self.start_times:
            raise ValueError(f"Timer {operation_name} was not started")
        
        start_time = self.start_times.pop(operation_name)
        end_time = time.time()
        latency_ms = (end_time - start_time) * 1000
        
        self.measurements[operation_name].append(latency_ms)
        return latency_ms
    
    def record_memory_snapshot(self, label: str) -> None:
        """Record current memory usage"""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        
        self.memory_snapshots.append({
            "label": label,
            "timestamp": time.time(),
            "rss_mb": memory_info.rss / 1024 / 1024,
            "vms_mb": memory_info.vms / 1024 / 1024
        })
    
    def get_statistics(self, operation_name: str) -> Dict[str, float]:
        """Get statistical summary for operation"""
        measurements = self.measurements.get(operation_name, [])
        if not measurements:
            return {}
        
        measurements.sort()
        n = len(measurements)
        
        return {
            "count": n,
            "min": measurements[0],
            "max": measurements[-1],
            "mean": sum(measurements) / n,
            "median": measurements[n // 2],
            "p95": measurements[int(0.95 * n)],
            "p99": measurements[int(0.99 * n)],
            "std_dev": (sum((x - sum(measurements) / n) ** 2 for x in measurements) / n) ** 0.5
        }
    
    def evaluate_benchmark(self, benchmark: PerformanceBenchmark) -> Dict[str, Any]:
        """Evaluate performance against benchmark"""
        stats = self.get_statistics(benchmark.benchmark_name)
        
        if not stats:
            return {"status": "NO_DATA", "message": "No measurements available"}
        
        # Use median for evaluation (more robust than mean)
        measured_value = stats["median"]
        
        if measured_value <= benchmark.target_value:
            status = "EXCELLENT"
        elif measured_value <= benchmark.acceptable_value:
            status = "ACCEPTABLE"
        elif measured_value <= benchmark.critical_threshold:
            status = "POOR"
        else:
            status = "CRITICAL"
        
        return {
            "status": status,
            "measured_value": measured_value,
            "target_value": benchmark.target_value,
            "unit": benchmark.unit,
            "statistics": stats,
            "benchmark": benchmark.dict()
        }
```

### 9.2 Continuous Performance Monitoring

```python
class PerformanceMonitor:
    """Continuous performance monitoring system"""
    
    def __init__(self, benchmarks: List[PerformanceBenchmark]):
        self.benchmarks = {b.benchmark_name: b for b in benchmarks}
        self.profiler = PerformanceProfiler()
        self.alert_thresholds = {}
        self.monitoring_active = False
        
        # Metrics collection
        self.metrics_history: List[Dict[str, Any]] = []
        self.alert_history: List[Dict[str, Any]] = []
    
    async def start_monitoring(self) -> None:
        """Start continuous monitoring"""
        self.monitoring_active = True
        
        # Start monitoring task
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
    
    async def stop_monitoring(self) -> None:
        """Stop continuous monitoring"""
        self.monitoring_active = False
        
        if hasattr(self, 'monitoring_task'):
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
    
    async def _monitoring_loop(self) -> None:
        """Main monitoring loop"""
        while self.monitoring_active:
            try:
                # Collect current metrics
                metrics = await self._collect_metrics()
                
                # Store metrics
                self.metrics_history.append({
                    "timestamp": time.time(),
                    "metrics": metrics
                })
                
                # Check for alerts
                await self._check_performance_alerts(metrics)
                
                # Cleanup old metrics
                self._cleanup_old_metrics()
                
                # Wait before next collection
                await asyncio.sleep(10.0)  # 10 second intervals
                
            except Exception as e:
                print(f"Monitoring error: {e}")
                await asyncio.sleep(1.0)
    
    async def _collect_metrics(self) -> Dict[str, float]:
        """Collect current performance metrics"""
        metrics = {}
        
        # Memory usage
        import psutil
        import os
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        metrics["memory_mb"] = memory_info.rss / 1024 / 1024
        
        # CPU usage
        metrics["cpu_percent"] = process.cpu_percent()
        
        # Operation statistics
        for benchmark_name in self.benchmarks:
            stats = self.profiler.get_statistics(benchmark_name)
            if stats:
                metrics[f"{benchmark_name}_median"] = stats["median"]
                metrics[f"{benchmark_name}_p95"] = stats["p95"]
        
        return metrics
    
    async def _check_performance_alerts(self, metrics: Dict[str, float]) -> None:
        """Check metrics against alert thresholds"""
        alerts = []
        
        for benchmark_name, benchmark in self.benchmarks.items():
            metric_key = f"{benchmark_name}_median"
            
            if metric_key in metrics:
                value = metrics[metric_key]
                
                if value > benchmark.critical_threshold:
                    alerts.append({
                        "level": "CRITICAL",
                        "benchmark": benchmark_name,
                        "value": value,
                        "threshold": benchmark.critical_threshold,
                        "message": f"{benchmark_name} exceeded critical threshold"
                    })
                elif value > benchmark.acceptable_value:
                    alerts.append({
                        "level": "WARNING",
                        "benchmark": benchmark_name,
                        "value": value,
                        "threshold": benchmark.acceptable_value,
                        "message": f"{benchmark_name} exceeded acceptable threshold"
                    })
        
        # Process alerts
        for alert in alerts:
            await self._process_alert(alert)
    
    async def _process_alert(self, alert: Dict[str, Any]) -> None:
        """Process performance alert"""
        alert["timestamp"] = time.time()
        self.alert_history.append(alert)
        
        # Log alert
        print(f"PERFORMANCE ALERT [{alert['level']}]: {alert['message']}")
        
        # Could integrate with external alerting systems here
    
    def _cleanup_old_metrics(self) -> None:
        """Remove old metrics to prevent memory bloat"""
        max_history = 1000
        if len(self.metrics_history) > max_history:
            self.metrics_history = self.metrics_history[-max_history:]
        
        if len(self.alert_history) > max_history:
            self.alert_history = self.alert_history[-max_history:]
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        report = {
            "timestamp": time.time(),
            "monitoring_active": self.monitoring_active,
            "benchmarks": {},
            "recent_alerts": self.alert_history[-10:],  # Last 10 alerts
            "system_metrics": {}
        }
        
        # Evaluate all benchmarks
        for benchmark_name, benchmark in self.benchmarks.items():
            evaluation = self.profiler.evaluate_benchmark(benchmark)
            report["benchmarks"][benchmark_name] = evaluation
        
        # Recent system metrics
        if self.metrics_history:
            latest_metrics = self.metrics_history[-1]["metrics"]
            report["system_metrics"] = latest_metrics
        
        return report
```

## 10. Implementation Patterns

### 10.1 Factory Pattern for Component Creation

```python
class ComponentFactory:
    """Factory for creating hybrid architecture components"""
    
    @staticmethod
    def create_perception_pipeline(config: HybridConfig) -> IPerceptionPipeline:
        """Create perception pipeline based on configuration"""
        
        if config.mesa_config.enable_physics:
            return PhysicsAwarePerceptionPipeline(config.performance_config)
        else:
            return BasicPerceptionPipeline(config.performance_config)
    
    @staticmethod
    def create_decision_engine(config: HybridConfig) -> IDecisionEngine:
        """Create decision engine based on LLM provider"""
        
        provider = config.crewai_config.llm_provider
        
        if provider == LLMProvider.OPENAI:
            return OpenAIDecisionEngine(config.crewai_config)
        elif provider == LLMProvider.ANTHROPIC:
            return AnthropicDecisionEngine(config.crewai_config)
        elif provider == LLMProvider.GEMINI:
            return GeminiDecisionEngine(config.crewai_config)
        elif provider == LLMProvider.OLLAMA:
            return OllamaDecisionEngine(config.crewai_config)
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")
    
    @staticmethod
    def create_action_translator(config: HybridConfig) -> IActionTranslator:
        """Create action translator"""
        
        return StandardActionTranslator(
            mesa_config=config.mesa_config,
            supported_actions=["move", "interact", "pickup", "drop", "communicate"]
        )
    
    @staticmethod
    def create_state_synchronizer(config: HybridConfig) -> IStateSynchronizer:
        """Create state synchronizer"""
        
        return EventDrivenStateSynchronizer(
            event_bus=AsyncEventBus(),
            validation_enabled=config.testing_config.strict_validation if config.testing_config else True
        )

class HybridSimulationBuilder:
    """Builder pattern for constructing hybrid simulations"""
    
    def __init__(self):
        self.config: Optional[HybridConfig] = None
        self.mesa_model: Optional[mesa.Model] = None
        self.crewai_agents: List[Agent] = []
        self.custom_components: Dict[str, Any] = {}
    
    def with_config(self, config: HybridConfig) -> 'HybridSimulationBuilder':
        """Set configuration"""
        self.config = config
        return self
    
    def with_mesa_model(self, model: mesa.Model) -> 'HybridSimulationBuilder':
        """Set Mesa model"""
        self.mesa_model = model
        return self
    
    def with_crewai_agents(self, agents: List[Agent]) -> 'HybridSimulationBuilder':
        """Set CrewAI agents"""
        self.crewai_agents = agents
        return self
    
    def with_custom_component(self, name: str, component: Any) -> 'HybridSimulationBuilder':
        """Add custom component"""
        self.custom_components[name] = component
        return self
    
    def build(self) -> HybridSimulationEngine:
        """Build the hybrid simulation engine"""
        
        if not self.config:
            raise ValueError("Configuration is required")
        
        if not self.mesa_model:
            raise ValueError("Mesa model is required")
        
        if not self.crewai_agents:
            raise ValueError("CrewAI agents are required")
        
        # Create components using factory
        perception_pipeline = ComponentFactory.create_perception_pipeline(self.config)
        decision_engine = ComponentFactory.create_decision_engine(self.config)
        action_translator = ComponentFactory.create_action_translator(self.config)
        state_synchronizer = ComponentFactory.create_state_synchronizer(self.config)
        
        # Override with custom components if provided
        for name, component in self.custom_components.items():
            if name == "perception_pipeline":
                perception_pipeline = component
            elif name == "decision_engine":
                decision_engine = component
            elif name == "action_translator":
                action_translator = component
            elif name == "state_synchronizer":
                state_synchronizer = component
        
        # Create engine
        engine = HybridSimulationEngine(
            mesa_model=self.mesa_model,
            crewai_agents=self.crewai_agents,
            perception_pipeline=perception_pipeline,
            decision_engine=decision_engine,
            action_translator=action_translator,
            state_synchronizer=state_synchronizer,
            config=self.config
        )
        
        return engine
```

### 10.2 Observer Pattern for Event Handling

```python
class Subject:
    """Subject in observer pattern"""
    
    def __init__(self):
        self._observers: List['Observer'] = []
    
    def attach(self, observer: 'Observer') -> None:
        """Attach observer"""
        if observer not in self._observers:
            self._observers.append(observer)
    
    def detach(self, observer: 'Observer') -> None:
        """Detach observer"""
        if observer in self._observers:
            self._observers.remove(observer)
    
    def notify(self, event: Any) -> None:
        """Notify all observers"""
        for observer in self._observers:
            observer.update(self, event)

class Observer(ABC):
    """Observer interface"""
    
    @abstractmethod
    def update(self, subject: Subject, event: Any) -> None:
        """Update observer with event"""
        pass

class SimulationStateObserver(Observer):
    """Observer for simulation state changes"""
    
    def __init__(self, state_manager: 'UnifiedStateManager'):
        self.state_manager = state_manager
        self.change_history: List[Dict[str, Any]] = []
    
    def update(self, subject: Subject, event: Any) -> None:
        """Handle state change event"""
        if isinstance(event, StateChange):
            self.change_history.append({
                "timestamp": event.timestamp,
                "change_type": event.change_type,
                "agent_id": event.agent_id,
                "change_summary": self._summarize_change(event)
            })
            
            # Update state manager
            self.state_manager.apply_change(event)
    
    def _summarize_change(self, change: StateChange) -> str:
        """Create human-readable summary of change"""
        if change.change_type == StateType.MESA_AGENT_MOVE:
            old_pos = change.previous_state.get("position")
            new_pos = change.new_state.get("position")
            return f"Agent moved from {old_pos} to {new_pos}"
        elif change.change_type == StateType.CREWAI_DECISION_MADE:
            action = change.new_state.get("chosen_action")
            return f"Agent decided to {action}"
        else:
            return f"State change: {change.change_type.value}"

class PerformanceObserver(Observer):
    """Observer for performance monitoring"""
    
    def __init__(self, profiler: PerformanceProfiler):
        self.profiler = profiler
        self.metrics_buffer: List[Dict[str, Any]] = []
    
    def update(self, subject: Subject, event: Any) -> None:
        """Handle performance event"""
        if hasattr(event, 'duration'):
            operation_name = getattr(event, 'operation_name', 'unknown')
            duration_ms = event.duration * 1000
            
            self.profiler.measurements[operation_name].append(duration_ms)
            
            # Check for performance anomalies
            self._check_performance_anomaly(operation_name, duration_ms)
    
    def _check_performance_anomaly(self, operation: str, duration: float) -> None:
        """Check for performance anomalies"""
        measurements = self.profiler.measurements[operation]
        
        if len(measurements) >= 10:
            recent_avg = sum(measurements[-10:]) / 10
            if duration > recent_avg * 2:  # 100% increase
                print(f"Performance anomaly detected: {operation} took {duration:.2f}ms (avg: {recent_avg:.2f}ms)")
```

### 10.3 Strategy Pattern for Decision Making

```python
class DecisionStrategy(ABC):
    """Abstract strategy for agent decision making"""
    
    @abstractmethod
    async def make_decision(self, agent_id: str, perception: PerceptionData) -> DecisionData:
        """Make decision based on perception"""
        pass
    
    @abstractmethod
    def is_applicable(self, agent_id: str, situation: Dict[str, Any]) -> bool:
        """Check if strategy is applicable to situation"""
        pass

class StrategistDecisionStrategy(DecisionStrategy):
    """Decision strategy for strategist agents"""
    
    def __init__(self, llm_client: Any):
        self.llm_client = llm_client
    
    async def make_decision(self, agent_id: str, perception: PerceptionData) -> DecisionData:
        """Make analytical decision"""
        
        # Format analytical prompt
        prompt = f"""
        As a strategic analyst, analyze this situation:
        
        Environment: {perception.environmental_state}
        Nearby agents: {perception.nearby_agents}
        Available resources: {perception.resources}
        Possible actions: {perception.available_actions}
        
        Provide a detailed analysis and recommend the best action with reasoning.
        Consider long-term consequences and resource optimization.
        """
        
        response = await self.llm_client.complete(prompt)
        
        # Parse response (simplified)
        action = self._extract_action(response, perception.available_actions)
        
        return DecisionData(
            agent_id=agent_id,
            timestamp=datetime.now(),
            chosen_action=action,
            action_parameters={},
            reasoning=response,
            confidence_level=0.8,
            fallback_actions=perception.available_actions[:2]
        )
    
    def is_applicable(self, agent_id: str, situation: Dict[str, Any]) -> bool:
        """Check if this is a strategist agent"""
        return "strategist" in agent_id.lower()
    
    def _extract_action(self, response: str, available_actions: List[str]) -> str:
        """Extract action from LLM response"""
        response_lower = response.lower()
        
        for action in available_actions:
            if action.lower() in response_lower:
                return action
        
        return available_actions[0] if available_actions else "wait"

class MediatorDecisionStrategy(DecisionStrategy):
    """Decision strategy for mediator agents"""
    
    def __init__(self, llm_client: Any):
        self.llm_client = llm_client
    
    async def make_decision(self, agent_id: str, perception: PerceptionData) -> DecisionData:
        """Make collaborative decision"""
        
        prompt = f"""
        As a mediator focused on team cooperation, analyze this situation:
        
        Team members nearby: {perception.nearby_agents}
        Available resources: {perception.resources}
        Possible actions: {perception.available_actions}
        
        Recommend an action that promotes team cooperation and conflict resolution.
        Prioritize communication and resource sharing.
        """
        
        response = await self.llm_client.complete(prompt)
        action = self._extract_action(response, perception.available_actions)
        
        return DecisionData(
            agent_id=agent_id,
            timestamp=datetime.now(),
            chosen_action=action,
            action_parameters={},
            reasoning=response,
            confidence_level=0.7,
            fallback_actions=["communicate", "share_resource"]
        )
    
    def is_applicable(self, agent_id: str, situation: Dict[str, Any]) -> bool:
        """Check if this is a mediator agent"""
        return "mediator" in agent_id.lower()
    
    def _extract_action(self, response: str, available_actions: List[str]) -> str:
        """Extract collaborative action from response"""
        response_lower = response.lower()
        
        # Prioritize collaborative actions
        collaborative_actions = ["communicate", "share", "help", "coordinate"]
        
        for collab_action in collaborative_actions:
            for action in available_actions:
                if collab_action in action.lower() and action.lower() in response_lower:
                    return action
        
        # Fallback to any mentioned action
        for action in available_actions:
            if action.lower() in response_lower:
                return action
        
        return "communicate" if "communicate" in available_actions else available_actions[0]

class ContextualDecisionEngine(IDecisionEngine):
    """Decision engine using strategy pattern"""
    
    def __init__(self, llm_client: Any):
        self.llm_client = llm_client
        self.strategies: List[DecisionStrategy] = [
            StrategistDecisionStrategy(llm_client),
            MediatorDecisionStrategy(llm_client),
            # Add more strategies as needed
        ]
    
    async def reason_and_decide(self, perceptions: Dict[str, PerceptionData]) -> Dict[str, DecisionData]:
        """Make decisions using appropriate strategies"""
        decisions = {}
        
        for agent_id, perception in perceptions.items():
            # Find applicable strategy
            strategy = self._find_strategy(agent_id, perception)
            
            if strategy:
                decision = await strategy.make_decision(agent_id, perception)
                decisions[agent_id] = decision
            else:
                # Fallback decision
                decisions[agent_id] = self._create_fallback_decision(agent_id, perception)
        
        return decisions
    
    def _find_strategy(self, agent_id: str, perception: PerceptionData) -> Optional[DecisionStrategy]:
        """Find applicable strategy for agent"""
        situation = {
            "agent_count": len(perception.nearby_agents),
            "resource_count": len(perception.resources),
            "action_count": len(perception.available_actions)
        }
        
        for strategy in self.strategies:
            if strategy.is_applicable(agent_id, situation):
                return strategy
        
        return None
    
    def _create_fallback_decision(self, agent_id: str, perception: PerceptionData) -> DecisionData:
        """Create fallback decision when no strategy applies"""
        return DecisionData(
            agent_id=agent_id,
            timestamp=datetime.now(),
            chosen_action=perception.available_actions[0] if perception.available_actions else "wait",
            action_parameters={},
            reasoning="No specific strategy available, using default action",
            confidence_level=0.3,
            fallback_actions=perception.available_actions[1:3]
        )
    
    def update_agent_memory(self, agent_id: str, experience: Dict[str, Any]) -> None:
        """Update agent memory (implementation would vary by strategy)"""
        pass
```

This comprehensive technical specification provides the detailed implementation guidance needed for the Mesa-CrewAI hybrid architecture. The specifications include exact API contracts, data models, interface definitions, state schemas, event systems, configuration schemas, error handling, testing frameworks, performance benchmarks, and implementation patterns.

These specifications enable multiple developers to implement the system independently while ensuring compatibility and consistent behavior across all components.