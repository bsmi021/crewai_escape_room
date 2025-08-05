# Mesa-CrewAI Hybrid Architecture Implementation Guide

## Overview

This guide provides step-by-step instructions for implementing and using the Mesa-CrewAI hybrid architecture, including code examples, configuration options, and best practices.

## 1. Basic Setup and Initialization

### Setting Up the Hybrid Simulation

```python
from src.escape_room_sim.hybrid.core_architecture import HybridSimulationFactory
from src.escape_room_sim.hybrid.data_flow import PerceptionPipeline, NaturalLanguagePerceptionFormatter
from src.escape_room_sim.hybrid.state_management import UnifiedStateManager, DefaultStateValidator, JSONStateSerializer
from src.escape_room_sim.hybrid.error_handling import HybridErrorManager, setup_default_error_handling
from src.escape_room_sim.hybrid.performance import HybridPerformanceManager

# Configuration for the escape room simulation
room_config = {
    "width": 10,
    "height": 10,
    "initial_resources": ["key", "tool", "rope"],
    "obstacles": [(3, 3), (7, 8)],
    "escape_requirements": ["key", "tool"]
}

# Agent configurations
agent_configs = [
    {
        "role": "Strategist",
        "goal": "Analyze situations and develop optimal strategies",
        "backstory": "Analytical problem-solver with strategic thinking",
        "personality": "methodical"
    },
    {
        "role": "Mediator", 
        "goal": "Facilitate team collaboration and resolve conflicts",
        "backstory": "Diplomatic coordinator focused on team unity",
        "personality": "collaborative"
    },
    {
        "role": "Survivor",
        "goal": "Execute plans and make critical survival decisions", 
        "backstory": "Pragmatic decision-maker with survival instincts",
        "personality": "decisive"
    }
]

# LLM configuration
llm_config = {
    "model": "gemini-2.5-flash-lite",
    "api_key": "your_api_key_here",
    "temperature": 0.7,
    "max_tokens": 1000
}

# Create the hybrid simulation
simulation = HybridSimulationFactory.create_escape_room_simulation(
    room_config=room_config,
    agent_configs=agent_configs,
    llm_config=llm_config
)

# Initialize the simulation
simulation.initialize()
```

### Advanced Configuration with Custom Components

```python
from src.escape_room_sim.hybrid.data_flow import PerceptionPipeline
from src.escape_room_sim.hybrid.state_management import UnifiedStateManager
from src.escape_room_sim.hybrid.performance import IntelligentCache, LLMBatchProcessor

# Custom perception pipeline with enhanced caching
perception_pipeline = PerceptionPipeline(
    enable_caching=True,
    cache_ttl=2.0  # 2-second cache for dynamic environments
)

# State manager with custom validation
validator = DefaultStateValidator()
serializer = JSONStateSerializer()
state_manager = UnifiedStateManager(validator, serializer)

# Performance optimizations
performance_manager = HybridPerformanceManager()
performance_manager.initialize(llm_client=your_llm_client)

# Intelligent cache for LLM responses
llm_cache = IntelligentCache(max_size=500, default_ttl=300.0)

# Batch processor for efficient LLM calls
batch_processor = LLMBatchProcessor(
    llm_client=your_llm_client,
    batch_size=3,
    batch_timeout=1.0
)
```

## 2. Running a Hybrid Simulation

### Basic Simulation Loop

```python
import asyncio
from datetime import datetime

async def run_hybrid_simulation(simulation, max_steps=100):
    """Run the hybrid simulation with proper error handling."""
    
    print("🚀 Starting Mesa-CrewAI Hybrid Simulation")
    
    for step in range(max_steps):
        try:
            print(f"\n--- Step {step + 1} ---")
            
            # Execute one simulation step
            step_result = await simulation.step()
            
            print(f"✅ Step completed in {step_result['duration']:.2f}s")
            print(f"🎯 Actions executed: {step_result['actions_executed']}")
            
            # Check if simulation should continue
            if not step_result.get('continue', True):
                print("🏁 Simulation completed successfully!")
                break
                
            # Optional: Add delay between steps
            await asyncio.sleep(0.1)
            
        except Exception as e:
            print(f"❌ Error in step {step + 1}: {e}")
            # Error recovery is handled internally by the simulation
            continue
    
    # Get final simulation state
    final_state = simulation.get_simulation_state()
    return final_state

# Run the simulation
final_state = asyncio.run(run_hybrid_simulation(simulation))
print(f"📊 Final state: {final_state}")
```

### Monitoring Performance During Simulation

```python
def monitor_simulation_performance(simulation):
    """Monitor and display simulation performance metrics."""
    
    performance_manager = simulation.performance_manager
    
    while simulation.state != "SHUTDOWN":
        # Get current performance report
        report = performance_manager.get_performance_report()
        
        # Display key metrics
        system_summary = report.get('system_summary', {})
        cache_stats = report.get('cache_stats', {})
        
        print(f"\n📈 Performance Report:")
        print(f"   Status: {system_summary.get('status', 'unknown')}")
        print(f"   Avg Latency: {system_summary.get('avg_latency', 0):.2f}s")
        print(f"   Error Rate: {system_summary.get('avg_error_rate', 0):.2%}")
        print(f"   Cache Hit Rate: {cache_stats.get('hit_rate', 0):.2%}")
        
        # Check for bottlenecks
        bottlenecks = system_summary.get('bottlenecks', [])
        if bottlenecks:
            print(f"⚠️  Bottlenecks detected: {', '.join(bottlenecks)}")
        
        time.sleep(10)  # Update every 10 seconds

# Run monitoring in background
import threading
monitor_thread = threading.Thread(target=monitor_simulation_performance, args=(simulation,))
monitor_thread.daemon = True
monitor_thread.start()
```

## 3. Custom Perception Pipeline Implementation

### Creating a Custom Perception Extractor

```python
from src.escape_room_sim.hybrid.data_flow import IPerceptionPipeline, PerceptionData
from typing import Dict
import mesa

class CustomEscapeRoomPerceptionPipeline(IPerceptionPipeline):
    """Custom perception pipeline for escape room scenarios."""
    
    def __init__(self):
        self.perception_cache = {}
        self.cache_ttl = 1.0  # 1 second cache
    
    def extract_perceptions(self, mesa_model: mesa.Model) -> Dict[str, PerceptionData]:
        """Extract escape room specific perceptions."""
        perceptions = {}
        
        for agent in mesa_model.schedule.agents:
            agent_id = self._get_agent_id(agent)
            
            # Extract comprehensive perception data
            perception = PerceptionData(
                agent_id=agent_id,
                timestamp=datetime.now(),
                spatial_data=self._extract_spatial_data(agent, mesa_model),
                environmental_state=self._extract_environment_data(mesa_model),
                nearby_agents=self._find_nearby_agents(agent, mesa_model),
                available_actions=self._determine_actions(agent, mesa_model),
                resources=self._extract_resource_data(agent, mesa_model),
                constraints=self._extract_constraints(agent, mesa_model)
            )
            
            perceptions[agent_id] = perception
        
        return perceptions
    
    def _extract_spatial_data(self, agent, mesa_model) -> Dict:
        """Extract spatial awareness data."""
        pos = getattr(agent, 'pos', (0, 0))
        
        return {
            "current_position": pos,
            "room_bounds": (mesa_model.grid.width, mesa_model.grid.height),
            "visible_area": self._calculate_vision_area(pos, mesa_model),
            "escape_exits": self._find_escape_exits(mesa_model),
            "blocked_paths": self._find_blocked_paths(pos, mesa_model)
        }
    
    def _extract_environment_data(self, mesa_model) -> Dict:
        """Extract environment-specific data."""
        return {
            "time_remaining": getattr(mesa_model, 'time_remaining', None),
            "danger_level": getattr(mesa_model, 'danger_level', 0.0),
            "room_status": getattr(mesa_model, 'room_status', 'locked'),
            "environmental_hazards": getattr(mesa_model, 'hazards', []),
            "puzzle_states": getattr(mesa_model, 'puzzle_states', {})
        }
    
    def _determine_actions(self, agent, mesa_model) -> List[str]:
        """Determine available actions for the agent."""
        actions = ["examine_environment", "rest", "analyze_situation"]
        
        pos = getattr(agent, 'pos', (0, 0))
        
        # Movement actions
        for direction in ['north', 'south', 'east', 'west']:
            if self._can_move_direction(pos, direction, mesa_model):
                actions.append(f"move_{direction}")
        
        # Object interaction actions
        nearby_objects = self._find_nearby_objects(pos, mesa_model)
        for obj in nearby_objects:
            actions.extend([f"examine_{obj}", f"use_{obj}", f"take_{obj}"])
        
        # Collaborative actions
        nearby_agents = self._find_nearby_agents(agent, mesa_model)
        if nearby_agents:
            actions.extend(["communicate", "coordinate", "share_resources"])
        
        # Escape actions
        if self._can_attempt_escape(agent, mesa_model):
            actions.append("attempt_escape")
        
        return actions

# Use the custom pipeline
custom_pipeline = CustomEscapeRoomPerceptionPipeline()
```

### Natural Language Formatting for Different Agent Personalities

```python
from src.escape_room_sim.hybrid.data_flow import NaturalLanguagePerceptionFormatter

class PersonalizedPerceptionFormatter(NaturalLanguagePerceptionFormatter):
    """Formatter that adapts to different agent personalities."""
    
    def format_perception_for_agent(self, perception: PerceptionData, 
                                  agent_personality: str) -> str:
        """Format perception with personality-specific language."""
        
        base_text = super().format_perception_for_agent(perception, agent_personality)
        
        if agent_personality == "strategist":
            return self._format_for_strategist(perception, base_text)
        elif agent_personality == "mediator":
            return self._format_for_mediator(perception, base_text)
        elif agent_personality == "survivor":
            return self._format_for_survivor(perception, base_text)
        else:
            return base_text
    
    def _format_for_strategist(self, perception: PerceptionData, base_text: str) -> str:
        """Format with analytical, strategic language."""
        strategic_analysis = []
        
        # Add probability assessments
        actions = perception.available_actions
        if actions:
            strategic_analysis.append("STRATEGIC ACTION ANALYSIS:")
            for action in actions[:5]:  # Top 5 actions
                probability = self._estimate_action_success_probability(action, perception)
                strategic_analysis.append(f"  • {action}: {probability:.1%} success probability")
        
        # Add resource optimization analysis
        resources = perception.resources
        if resources:
            strategic_analysis.append("\nRESOURCE OPTIMIZATION:")
            strategic_analysis.append(f"  • Current resource efficiency: {self._calculate_resource_efficiency(resources):.1%}")
        
        return f"{base_text}\n\n{chr(10).join(strategic_analysis)}"
    
    def _format_for_mediator(self, perception: PerceptionData, base_text: str) -> str:
        """Format with collaborative, team-focused language."""
        team_analysis = []
        
        # Add team coordination opportunities
        nearby_agents = perception.nearby_agents
        if nearby_agents:
            team_analysis.append("TEAM COORDINATION OPPORTUNITIES:")
            team_analysis.append(f"  • {len(nearby_agents)} team members within communication range")
            team_analysis.append("  • Consider collaborative problem-solving approaches")
            team_analysis.append("  • Opportunity to share resources and coordinate actions")
        
        # Add conflict prevention notes
        if self._detect_potential_conflicts(perception):
            team_analysis.append("\nCONFLICT PREVENTION NOTES:")
            team_analysis.append("  • Monitor for resource competition")
            team_analysis.append("  • Encourage open communication")
        
        return f"{base_text}\n\n{chr(10).join(team_analysis)}"
    
    def _format_for_survivor(self, perception: PerceptionData, base_text: str) -> str:
        """Format with urgent, action-oriented language."""
        survival_analysis = []
        
        # Add immediate threat assessment
        constraints = perception.constraints
        if constraints.get('time_constraints', {}).get('deadline_approaching'):
            survival_analysis.append("🚨 URGENT: TIME RUNNING OUT!")
            survival_analysis.append("  • Immediate action required")
            survival_analysis.append("  • Override group consensus if necessary")
        
        # Add resource conservation notes
        resources = perception.resources
        scarcity_levels = resources.get('scarcity_levels', {})
        critical_resources = [r for r, level in scarcity_levels.items() if level > 0.8]
        if critical_resources:
            survival_analysis.append(f"\n⚠️  CRITICAL RESOURCE SHORTAGE: {', '.join(critical_resources)}")
            survival_analysis.append("  • Conserve these resources for essential actions only")
        
        # Add escape opportunity assessment
        if 'attempt_escape' in perception.available_actions:
            survival_analysis.append("\n🏃 ESCAPE OPPORTUNITY DETECTED!")
            survival_analysis.append("  • Assess if conditions are optimal for escape attempt")
        
        return f"{base_text}\n\n{chr(10).join(survival_analysis)}"

# Use the personalized formatter
formatter = PersonalizedPerceptionFormatter(verbosity_level="detailed")
```

## 4. Error Handling and Recovery

### Setting Up Comprehensive Error Handling

```python
from src.escape_room_sim.hybrid.error_handling import (
    HybridErrorManager, CircuitBreakerConfig, LLMErrorHandler, 
    MesaErrorHandler, setup_default_error_handling
)

# Initialize error manager
error_manager = HybridErrorManager()

# Set up default error handlers
setup_default_error_handling(error_manager)

# Add custom error handlers
class EscapeRoomErrorHandler(IErrorHandler):
    """Custom error handler for escape room specific errors."""
    
    def can_handle(self, error_context: ErrorContext) -> bool:
        return "escape_room" in error_context.component.lower()
    
    async def handle_error(self, error_context: ErrorContext) -> RecoveryResult:
        if "agent_stuck" in error_context.error_message.lower():
            # Reset agent to safe position
            return RecoveryResult(
                success=True,
                strategy_used=RecoveryStrategy.RESET,
                message="Reset agent to safe position",
                fallback_data={"safe_position": (1, 1)}
            )
        elif "resource_conflict" in error_context.error_message.lower():
            # Resolve resource conflict
            return RecoveryResult(
                success=True,
                strategy_used=RecoveryStrategy.FALLBACK,
                message="Used conflict resolution protocol"
            )
        
        return RecoveryResult(success=False, strategy_used=RecoveryStrategy.ESCALATE, message="Cannot handle")
    
    def get_priority(self) -> int:
        return 1  # High priority

# Register custom handler
error_manager.register_error_handler(EscapeRoomErrorHandler())

# Configure circuit breakers for different services
llm_circuit_breaker = error_manager.get_circuit_breaker("llm", CircuitBreakerConfig(
    failure_threshold=3,
    timeout_duration=30,
    success_threshold=2
))

mesa_circuit_breaker = error_manager.get_circuit_breaker("mesa", CircuitBreakerConfig(
    failure_threshold=5,
    timeout_duration=10,
    success_threshold=3
))
```

### Using Error Handling in Practice

```python
from src.escape_room_sim.hybrid.error_handling import error_handling_context

async def safe_simulation_step(simulation, error_manager):
    """Execute simulation step with comprehensive error handling."""
    
    async with error_handling_context(error_manager, "simulation", "step"):
        step_result = await simulation.step()
        return step_result

# Monitor error patterns
def monitor_error_health(error_manager):
    """Monitor system health and error patterns."""
    
    health = error_manager.get_system_health()
    
    print(f"🏥 System Health Report:")
    print(f"   Degradation Level: {health['degradation_level']}/3")
    print(f"   Fallback Mode: {'ON' if health['fallback_mode'] else 'OFF'}")
    print(f"   Recent Errors: {health['recent_error_count']}")
    
    # Alert on high error rates
    if health['degradation_level'] >= 2:
        print("⚠️  WARNING: System experiencing degraded performance")
        
    if health['recent_error_count'] > 10:
        print("🚨 ALERT: High error rate detected")
    
    # Show circuit breaker states
    circuit_breakers = health['circuit_breakers']
    for name, status in circuit_breakers.items():
        state = status['state']
        if state != 'closed':
            print(f"⚡ Circuit breaker '{name}' is {state}")
```

## 5. Performance Optimization

### Enabling Performance Optimizations

```python
from src.escape_room_sim.hybrid.performance import (
    HybridPerformanceManager, IntelligentCache, ConnectionPool, 
    performance_monitored, cached_result
)

# Initialize performance manager
perf_manager = HybridPerformanceManager()
perf_manager.initialize(llm_client=your_llm_client)

# Configure intelligent caching
@cached_result(ttl=120.0, cache_instance=perf_manager.cache)
def expensive_analysis_function(data):
    """Expensive analysis that benefits from caching."""
    # Simulate expensive computation
    time.sleep(2)
    return f"Analysis result for {data}"

# Monitor performance automatically
@performance_monitored("llm_decision_making")
async def make_llm_decision(prompt, agent_id):
    """Make LLM decision with automatic performance monitoring."""
    
    # Use optimized LLM call with batching and caching
    result = await perf_manager.optimize_llm_call(prompt, agent_id=agent_id)
    return result

# Create connection pools for external services
def create_api_connection():
    """Factory function for creating API connections."""
    # Your connection creation logic here
    return your_api_client()

api_pool = perf_manager.create_connection_pool(
    name="llm_api",
    create_connection=create_api_connection,
    max_size=5
)

async def use_pooled_connection():
    """Example of using connection pool."""
    pool = perf_manager.get_connection_pool("llm_api")
    
    connection = await pool.acquire_connection()
    try:
        # Use connection
        result = await connection.make_request("your_request")
        return result
    finally:
        await pool.release_connection(connection)
```

### Performance Tuning Guidelines

```python
def optimize_simulation_performance(simulation):
    """Apply performance optimizations to the simulation."""
    
    # 1. Enable intelligent caching with appropriate TTL
    perception_cache = IntelligentCache(max_size=1000, default_ttl=5.0)
    simulation.perception_pipeline.cache = perception_cache
    
    # 2. Configure LLM batching for efficiency
    if hasattr(simulation, 'decision_engine'):
        simulation.decision_engine.batch_size = 3
        simulation.decision_engine.batch_timeout = 1.0
    
    # 3. Optimize state synchronization frequency
    simulation.state_manager.sync_interval = 2.0  # Sync every 2 seconds
    
    # 4. Enable performance monitoring
    simulation.performance_manager.auto_optimization_enabled = True
    simulation.performance_manager.optimization_interval = 180  # 3 minutes
    
    # 5. Configure memory usage limits
    simulation.memory_manager.max_memory_size = 100 * 1024 * 1024  # 100MB
    
    print("✅ Performance optimizations applied")

# Apply optimizations
optimize_simulation_performance(simulation)
```

## 6. State Management and Persistence

### Managing State Snapshots

```python
from src.escape_room_sim.hybrid.state_management import UnifiedStateManager

def manage_simulation_state(state_manager):
    """Demonstrate state management capabilities."""
    
    # Create snapshots at key points
    initial_snapshot = state_manager.create_snapshot("initial_state")
    print(f"📸 Created initial snapshot: {initial_snapshot}")
    
    # Simulate some state changes
    # ... simulation runs ...
    
    # Create checkpoint snapshot
    checkpoint_snapshot = state_manager.create_snapshot("checkpoint_1")
    print(f"📸 Created checkpoint: {checkpoint_snapshot}")
    
    # If something goes wrong, rollback
    if error_detected:
        success = state_manager.rollback_to_snapshot(initial_snapshot)
        if success:
            print("↩️  Successfully rolled back to initial state")
        else:
            print("❌ Rollback failed")
    
    # Get performance metrics
    metrics = state_manager.get_performance_metrics()
    print(f"📊 State management metrics:")
    print(f"   Sync operations: {metrics['sync_metrics']['sync_count']}")
    print(f"   Average sync time: {metrics['sync_metrics']['avg_sync_time']:.3f}s")
    print(f"   State size: {metrics['state_size']['unified']} bytes")

# Monitor state changes
def state_change_listener(change):
    """Listen for state changes and log them."""
    print(f"🔄 State change: {change.change_type.value} on {change.entity_id}")
    if change.change_type == StateChangeType.MOVE:
        print(f"   Moved from {change.old_value} to {change.new_value}")

# Add listener
state_manager.add_change_listener(StateType.SPATIAL, state_change_listener)
```

### Serializing and Persisting State

```python
import json
from pathlib import Path

def save_simulation_state(simulation, filepath):
    """Save complete simulation state to file."""
    
    state_data = {
        "simulation_state": simulation.get_simulation_state(),
        "performance_metrics": simulation.performance_manager.get_performance_report(),
        "error_history": simulation.error_manager.get_system_health(),
        "timestamp": datetime.now().isoformat()
    }
    
    with open(filepath, 'w') as f:
        json.dump(state_data, f, indent=2, default=str)
    
    print(f"💾 Simulation state saved to {filepath}")

def load_simulation_state(simulation, filepath):
    """Load simulation state from file."""
    
    with open(filepath, 'r') as f:
        state_data = json.load(f)
    
    # Restore state
    simulation.restore_state(state_data["simulation_state"])
    
    print(f"📁 Simulation state loaded from {filepath}")
    return state_data

# Usage
save_simulation_state(simulation, "simulation_checkpoint.json")
# Later...
load_simulation_state(simulation, "simulation_checkpoint.json")
```

## 7. Testing and Validation

### Unit Testing Components

```python
import pytest
from unittest.mock import Mock, MagicMock
from src.escape_room_sim.hybrid.core_architecture import HybridSimulationEngine

@pytest.fixture
def mock_mesa_model():
    """Create mock Mesa model for testing."""
    model = Mock()
    model.schedule.agents = []
    model.grid.width = 10
    model.grid.height = 10
    return model

@pytest.fixture
def mock_crewai_agents():
    """Create mock CrewAI agents for testing."""
    agents = []
    for role in ["Strategist", "Mediator", "Survivor"]:
        agent = Mock()
        agent.role = role
        agent.memory = {}
        agents.append(agent)
    return agents

@pytest.mark.asyncio
async def test_simulation_step(mock_mesa_model, mock_crewai_agents):
    """Test single simulation step execution."""
    
    # Create simulation with mocks
    simulation = HybridSimulationEngine(
        mesa_model=mock_mesa_model,
        crewai_agents=mock_crewai_agents,
        perception_pipeline=Mock(),
        decision_engine=Mock(),
        action_translator=Mock(),
        state_synchronizer=Mock()
    )
    
    # Mock the decision engine to return test decisions
    simulation.decision_engine.reason_and_decide.return_value = {
        "strategist": Mock(chosen_action="analyze", confidence_level=0.8),
        "mediator": Mock(chosen_action="coordinate", confidence_level=0.9),
        "survivor": Mock(chosen_action="execute", confidence_level=0.7)
    }
    
    # Execute step
    result = await simulation.step()
    
    # Validate results
    assert result["step"] == 1
    assert result["actions_executed"] == 3
    assert "duration" in result
    
    # Verify mocks were called
    simulation.decision_engine.reason_and_decide.assert_called_once()
    mock_mesa_model.step.assert_called_once()

def test_perception_extraction():
    """Test perception data extraction."""
    
    pipeline = PerceptionPipeline()
    mock_model = Mock()
    
    # Create mock agent
    mock_agent = Mock()
    mock_agent.pos = (5, 5)
    mock_agent.resources = ["key"]
    mock_model.schedule.agents = [mock_agent]
    
    # Extract perceptions
    perceptions = pipeline.extract_perceptions(mock_model)
    
    # Validate extraction
    assert len(perceptions) == 1
    agent_perception = list(perceptions.values())[0]
    assert agent_perception.spatial_data["current_position"] == (5, 5)

def test_error_handling():
    """Test error handling system."""
    
    error_manager = HybridErrorManager()
    setup_default_error_handling(error_manager)
    
    # Test LLM error handling
    test_error = Exception("LLM API timeout")
    context = {"component": "decision_engine", "operation": "reason_and_decide"}
    
    result = asyncio.run(error_manager.handle_error(test_error, context))
    
    # Should have attempted recovery
    assert result.strategy_used in [RecoveryStrategy.FALLBACK, RecoveryStrategy.RETRY]
```

### Integration Testing

```python
@pytest.mark.integration
@pytest.mark.asyncio
async def test_full_simulation_integration():
    """Test complete simulation integration."""
    
    # Create real simulation with test configuration
    config = {
        "max_iterations": 5,
        "enable_memory": True,
        "verbose_output": False
    }
    
    simulation = create_test_simulation(config)
    
    # Run simulation
    final_state = await run_test_simulation(simulation, max_steps=10)
    
    # Validate final state
    assert final_state["engine_state"] in ["READY", "SHUTDOWN"]
    assert final_state["step_count"] > 0
    assert "performance_summary" in final_state
    
    # Check performance metrics
    perf_summary = final_state["performance_summary"]
    assert perf_summary["avg_step_duration"] > 0
    assert perf_summary["total_steps"] > 0

def test_state_synchronization():
    """Test state synchronization between frameworks."""
    
    state_manager = UnifiedStateManager(
        validator=DefaultStateValidator(),
        serializer=JSONStateSerializer()
    )
    
    # Create test state change
    change = StateChange(
        change_id="test_move",
        timestamp=datetime.now(),
        change_type=StateChangeType.MOVE,
        state_type=StateType.SPATIAL,
        entity_id="agent_1",
        old_value=(0, 0),
        new_value=(1, 1),
        source="mesa"
    )
    
    # Register and apply change
    success = state_manager.register_state_change(change)
    assert success
    
    applied_changes = state_manager.apply_pending_changes()
    assert len(applied_changes) == 1
    assert applied_changes[0].applied
```

## 8. Best Practices and Troubleshooting

### Performance Best Practices

1. **Enable Caching**: Use intelligent caching for expensive operations
2. **Batch LLM Calls**: Process multiple requests together
3. **Monitor Performance**: Continuously track metrics and optimize
4. **Use Connection Pools**: Reuse connections to external services
5. **Optimize State Sync**: Batch state updates and sync periodically

### Common Issues and Solutions

```python
def troubleshoot_common_issues(simulation):
    """Diagnose and fix common simulation issues."""
    
    # 1. High LLM latency
    perf_report = simulation.performance_manager.get_performance_report()
    llm_latency = perf_report.get("system_summary", {}).get("avg_latency", 0)
    
    if llm_latency > 3.0:
        print("⚠️  High LLM latency detected")
        print("   Recommendations:")
        print("   - Enable LLM batching")
        print("   - Use circuit breakers")
        print("   - Implement caching")
        print("   - Consider faster model")
    
    # 2. Memory usage issues
    state_metrics = simulation.state_manager.get_performance_metrics()
    state_size = state_metrics["state_size"]["unified"]
    
    if state_size > 10 * 1024 * 1024:  # 10MB
        print("⚠️  Large state size detected")
        print("   Recommendations:")
        print("   - Enable state compression")
        print("   - Limit memory retention period")
        print("   - Clean up old snapshots")
    
    # 3. Error rate issues
    error_health = simulation.error_manager.get_system_health()
    error_rate = error_health.get("recent_error_count", 0)
    
    if error_rate > 5:
        print("⚠️  High error rate detected")
        print("   Recommendations:")
        print("   - Check circuit breaker states")
        print("   - Review error categories")
        print("   - Implement better fallbacks")
    
    # 4. State synchronization issues
    sync_metrics = state_metrics["sync_metrics"]
    avg_sync_time = sync_metrics.get("avg_sync_time", 0)
    
    if avg_sync_time > 0.1:  # 100ms
        print("⚠️  Slow state synchronization detected")
        print("   Recommendations:")
        print("   - Increase sync interval")
        print("   - Batch state updates")
        print("   - Optimize state structure")

# Run diagnostics
troubleshoot_common_issues(simulation)
```

### Development Tips

1. **Start Simple**: Begin with basic configuration and add complexity gradually
2. **Test Components**: Unit test individual components before integration
3. **Monitor Everything**: Enable comprehensive logging and monitoring
4. **Handle Errors Gracefully**: Implement robust error handling and recovery
5. **Optimize Incrementally**: Profile and optimize based on actual bottlenecks

This implementation guide provides a comprehensive foundation for building and deploying Mesa-CrewAI hybrid simulations with proper architecture, error handling, and performance optimization.