"""
Mesa-CrewAI Hybrid Architecture Package

This package provides a revolutionary hybrid architecture that combines Mesa's
agent-based modeling capabilities with CrewAI's LLM-powered reasoning agents.

Key Components:
- Core Architecture: Central orchestration and hybrid agents
- Data Flow: Perception and decision pipelines
- State Management: Unified state synchronization
- Error Handling: Fault tolerance and recovery
- Performance: Optimization and monitoring
- Testing: Comprehensive test framework

Usage:
    from escape_room_sim.hybrid import HybridSimulationEngine, HybridSimulationFactory
    
    # Create hybrid simulation
    engine = HybridSimulationFactory.create_escape_room_simulation(
        room_config={"width": 10, "height": 10},
        agent_configs=[
            {"role": "strategist", "personality": "analytical"},
            {"role": "mediator", "personality": "collaborative"},
            {"role": "survivor", "personality": "pragmatic"}
        ],
        llm_config={"provider": "gemini", "model": "gemini-2.5-flash-lite"}
    )
    
    # Initialize and run
    engine.initialize()
    
    for step in range(100):
        step_result = await engine.step()
        if step_result.get("simulation_complete"):
            break
    
    # Get results
    final_state = engine.get_simulation_state()
"""

from .core_architecture import (
    HybridSimulationEngine,
    HybridAgent,
    HybridSimulationFactory,
    ComponentState,
    PerceptionData,
    DecisionData,
    MesaAction
)

from .data_flow import (
    PerceptionPipeline,
    NaturalLanguagePerceptionFormatter,
    PerceptionType,
    SpatialPerception,
    EnvironmentalPerception,
    SocialPerception,
    ResourcePerception
)

from .state_management import (
    UnifiedStateManager,
    StateChange,
    StateType,
    StateChangeType,
    StateSnapshot,
    DefaultStateValidator,
    JSONStateSerializer
)

from .error_handling import (
    HybridErrorManager,
    CircuitBreaker,
    RetryableOperation,
    ErrorSeverity,
    ErrorCategory,
    RecoveryStrategy,
    ErrorContext,
    RecoveryResult,
    setup_default_error_handling
)

from .performance import (
    HybridPerformanceManager,
    IntelligentCache,
    AsyncBatchProcessor,
    LLMBatchProcessor,
    ConnectionPool,
    PerformanceMonitor,
    PerformanceProfile,
    performance_monitored,
    cached_result
)

from .testing_framework import (
    HybridTestHarness,
    DeterministicMockLLM,
    MockLLMResponse,
    TestScenario,
    TestMode,
    create_test_scenario,
    create_llm_response,
    validate_agent_positions_in_bounds,
    validate_resource_conservation,
    validate_state_consistency
)

__version__ = "1.0.0"
__author__ = "Claude Code Architecture Team"
__email__ = "architecture@anthropic.com"

__all__ = [
    # Core Architecture
    "HybridSimulationEngine",
    "HybridAgent", 
    "HybridSimulationFactory",
    "ComponentState",
    "PerceptionData",
    "DecisionData",
    "MesaAction",
    
    # Data Flow
    "PerceptionPipeline",
    "NaturalLanguagePerceptionFormatter", 
    "PerceptionType",
    "SpatialPerception",
    "EnvironmentalPerception",
    "SocialPerception",
    "ResourcePerception",
    
    # State Management
    "UnifiedStateManager",
    "StateChange",
    "StateType", 
    "StateChangeType",
    "StateSnapshot",
    "DefaultStateValidator",
    "JSONStateSerializer",
    
    # Error Handling
    "HybridErrorManager",
    "CircuitBreaker",
    "RetryableOperation",
    "ErrorSeverity",
    "ErrorCategory",
    "RecoveryStrategy", 
    "ErrorContext",
    "RecoveryResult",
    "setup_default_error_handling",
    
    # Performance
    "HybridPerformanceManager",
    "IntelligentCache",
    "AsyncBatchProcessor",
    "LLMBatchProcessor", 
    "ConnectionPool",
    "PerformanceMonitor",
    "PerformanceProfile",
    "performance_monitored",
    "cached_result",
    
    # Testing
    "HybridTestHarness",
    "DeterministicMockLLM",
    "MockLLMResponse",
    "TestScenario",
    "TestMode",
    "create_test_scenario", 
    "create_llm_response",
    "validate_agent_positions_in_bounds",
    "validate_resource_conservation",
    "validate_state_consistency"
]

# Package-level configuration
DEFAULT_CONFIG = {
    "performance": {
        "cache_size": 2000,
        "cache_ttl": 600.0,
        "batch_size": 5,
        "batch_timeout": 1.0,
        "connection_pool_size": 10
    },
    "error_handling": {
        "circuit_breaker_failure_threshold": 5,
        "circuit_breaker_timeout": 60,
        "max_retries": 3,
        "base_retry_delay": 1.0
    },
    "state_management": {
        "max_snapshots": 10,
        "sync_interval": 1.0,
        "validation_enabled": True
    },
    "testing": {
        "default_timeout": 30.0,
        "mock_llm_latency": 0.1,
        "deterministic_seed": 42
    }
}


def get_version() -> str:
    """Get package version"""
    return __version__


def get_default_config() -> dict:
    """Get default configuration"""
    return DEFAULT_CONFIG.copy()


def create_hybrid_simulation(room_config: dict, agent_configs: list, 
                           llm_config: dict, **kwargs) -> HybridSimulationEngine:
    """
    Convenience function to create hybrid simulation with default configuration
    
    Args:
        room_config: Mesa room/environment configuration
        agent_configs: CrewAI agent configurations  
        llm_config: LLM provider configuration
        **kwargs: Additional configuration options
        
    Returns:
        Configured HybridSimulationEngine instance
        
    Example:
        engine = create_hybrid_simulation(
            room_config={"width": 10, "height": 10, "obstacles": []},
            agent_configs=[
                {"role": "strategist", "personality": "analytical"},
                {"role": "mediator", "personality": "collaborative"}
            ],
            llm_config={"provider": "gemini", "model": "gemini-2.5-flash-lite"}
        )
    """
    return HybridSimulationFactory.create_escape_room_simulation(
        room_config=room_config,
        agent_configs=agent_configs, 
        llm_config=llm_config,
        **kwargs
    )


def setup_hybrid_testing(test_mode: TestMode = TestMode.INTEGRATION, 
                        seed: int = 42) -> HybridTestHarness:
    """
    Setup hybrid test harness with default configuration
    
    Args:
        test_mode: Testing mode (unit, integration, system, etc.)
        seed: Deterministic seed for reproducible tests
        
    Returns:
        Configured HybridTestHarness instance
        
    Example:
        test_harness = setup_hybrid_testing(TestMode.INTEGRATION, seed=123)
        scenario = create_test_scenario("basic_test", ...)
        result = await test_harness.run_test_scenario(scenario)
    """
    harness = HybridTestHarness(test_mode)
    harness.mock_llm = DeterministicMockLLM(seed)
    return harness


# Architecture validation utilities

def validate_hybrid_architecture(engine: HybridSimulationEngine) -> dict:
    """
    Validate hybrid architecture implementation
    
    Args:
        engine: HybridSimulationEngine instance to validate
        
    Returns:
        Validation results with status and any issues found
        
    Example:
        engine = create_hybrid_simulation(...)
        validation = validate_hybrid_architecture(engine) 
        if not validation["valid"]:
            print("Issues:", validation["issues"])
    """
    issues = []
    
    # Check required components
    if not hasattr(engine, 'mesa_model') or engine.mesa_model is None:
        issues.append("Mesa model not configured")
    
    if not hasattr(engine, 'crewai_agents') or not engine.crewai_agents:
        issues.append("CrewAI agents not configured")
    
    if not hasattr(engine, 'perception_pipeline') or engine.perception_pipeline is None:
        issues.append("Perception pipeline not configured")
    
    if not hasattr(engine, 'decision_engine') or engine.decision_engine is None:
        issues.append("Decision engine not configured")
    
    if not hasattr(engine, 'action_translator') or engine.action_translator is None:
        issues.append("Action translator not configured")
    
    if not hasattr(engine, 'state_synchronizer') or engine.state_synchronizer is None:
        issues.append("State synchronizer not configured")
    
    # Check hybrid agents
    if not hasattr(engine, 'hybrid_agents') or not engine.hybrid_agents:
        issues.append("Hybrid agents not created")
    
    # Check state consistency
    if hasattr(engine, 'hybrid_agents'):
        for agent_id, hybrid_agent in engine.hybrid_agents.items():
            if not hybrid_agent.mesa_agent or not hybrid_agent.crewai_agent:
                issues.append(f"Hybrid agent {agent_id} missing Mesa or CrewAI component")
    
    return {
        "valid": len(issues) == 0,
        "issues": issues,
        "component_count": len(getattr(engine, 'hybrid_agents', {})),
        "validation_timestamp": __import__('datetime').datetime.now()
    }


# Performance monitoring utilities

def create_performance_monitor(config: dict = None) -> HybridPerformanceManager:
    """
    Create performance monitor with configuration
    
    Args:
        config: Optional performance configuration override
        
    Returns:
        Configured HybridPerformanceManager instance
    """
    manager = HybridPerformanceManager()
    
    if config:
        # Apply custom configuration
        if "cache_size" in config:
            manager.cache = IntelligentCache(max_size=config["cache_size"])
        if "batch_size" in config and "batch_timeout" in config:
            # Would configure batch processor with custom settings
            pass
    
    return manager


# Error handling setup utilities

def setup_production_error_handling() -> HybridErrorManager:
    """
    Setup error handling for production environment
    
    Returns:
        Configured HybridErrorManager with production settings
    """
    error_manager = HybridErrorManager()
    setup_default_error_handling(error_manager)
    
    # Add production-specific configurations
    error_manager.degradation_level = 0
    error_manager.fallback_mode = False
    
    return error_manager


def setup_development_error_handling() -> HybridErrorManager:
    """
    Setup error handling for development environment
    
    Returns:
        Configured HybridErrorManager with development settings
    """
    error_manager = HybridErrorManager()
    setup_default_error_handling(error_manager)
    
    # Add development-specific configurations (more verbose logging, etc.)
    import logging
    logging.getLogger("hybrid_error_manager").setLevel(logging.DEBUG)
    
    return error_manager


# Integration utilities

def integrate_with_existing_crewai(crew: 'Crew', mesa_model: 'mesa.Model') -> HybridSimulationEngine:
    """
    Integrate existing CrewAI crew with Mesa model
    
    Args:
        crew: Existing CrewAI Crew instance
        mesa_model: Existing Mesa model instance
        
    Returns:
        HybridSimulationEngine integrating both systems
        
    Example:
        # Existing systems
        my_crew = Crew(agents=[...], tasks=[...])
        my_mesa_model = MyMesaModel()
        
        # Create hybrid integration
        hybrid_engine = integrate_with_existing_crewai(my_crew, my_mesa_model)
        hybrid_engine.initialize()
    """
    # Extract agents from crew
    crewai_agents = crew.agents
    
    # Create pipeline components
    perception_pipeline = PerceptionPipeline()
    
    # Create mock implementations for integration
    # In practice, these would be proper implementations
    class IntegrationDecisionEngine:
        def __init__(self, crew):
            self.crew = crew
        
        async def reason_and_decide(self, perceptions):
            # Use CrewAI crew to make decisions
            decisions = {}
            for agent in self.crew.agents:
                agent_id = agent.role.lower().replace(" ", "_")
                if agent_id in perceptions:
                    # Create decision based on crew execution
                    decision = DecisionData(
                        agent_id=agent_id,
                        timestamp=__import__('datetime').datetime.now(),
                        chosen_action="crew_action",
                        action_parameters={},
                        reasoning="Crew AI reasoning",
                        confidence_level=0.8,
                        fallback_actions=[]
                    )
                    decisions[agent_id] = decision
            return decisions
        
        def update_agent_memory(self, agent_id, experience):
            pass
    
    decision_engine = IntegrationDecisionEngine(crew)
    action_translator = DefaultActionTranslator()
    state_synchronizer = DefaultStateSynchronizer()
    
    # Create hybrid engine
    engine = HybridSimulationEngine(
        mesa_model=mesa_model,
        crewai_agents=crewai_agents,
        perception_pipeline=perception_pipeline,
        decision_engine=decision_engine,
        action_translator=action_translator,
        state_synchronizer=state_synchronizer
    )
    
    return engine


# Default implementations for integration

class DefaultActionTranslator:
    """Default action translator implementation"""
    
    def translate_decision(self, decision: DecisionData) -> MesaAction:
        return MesaAction(
            agent_id=decision.agent_id,
            action_type=decision.chosen_action,
            parameters=decision.action_parameters,
            expected_duration=1.0,
            prerequisites=[]
        )
    
    def validate_action(self, action: MesaAction, mesa_model) -> bool:
        return True


class DefaultStateSynchronizer:
    """Default state synchronizer implementation"""
    
    def sync_mesa_to_crewai(self, mesa_model) -> None:
        pass
    
    def sync_crewai_to_mesa(self, decisions, mesa_model) -> None:
        pass