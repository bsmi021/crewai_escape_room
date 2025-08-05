"""
Mesa-CrewAI Hybrid Testing Framework

This module implements a comprehensive testing framework designed to achieve
100% test coverage for the hybrid architecture, including deterministic
testing of non-deterministic LLM agents.

Key Features:
- Mock LLM providers with deterministic responses
- Mesa simulation state validation
- Property-based testing for hybrid interactions
- Performance regression testing
- End-to-end scenario testing
- Fault injection and chaos testing
"""

from typing import Dict, List, Any, Optional, Callable, Union, Type, Generator
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from abc import ABC, abstractmethod
import unittest
import asyncio
import json
import random
import threading
import time
from unittest.mock import Mock, MagicMock, patch
from contextlib import contextmanager, asynccontextmanager
import mesa
from crewai import Agent, Task, Crew

from .core_architecture import HybridSimulationEngine, HybridAgent, PerceptionData, DecisionData
from .state_management import UnifiedStateManager, StateChange, StateType
from .error_handling import HybridErrorManager, ErrorContext, ErrorCategory
from .performance import HybridPerformanceManager


class TestMode(Enum):
    """Testing modes for different scenarios"""
    UNIT = "unit"                    # Individual component testing
    INTEGRATION = "integration"     # Component interaction testing
    SYSTEM = "system"               # Full system testing
    PERFORMANCE = "performance"     # Performance and load testing
    CHAOS = "chaos"                 # Fault injection testing


class MockLLMResponse:
    """Mock LLM response with deterministic behavior"""
    
    def __init__(self, response_text: str, confidence: float = 1.0, 
                 latency: float = 0.1, should_fail: bool = False):
        self.response_text = response_text
        self.confidence = confidence
        self.latency = latency
        self.should_fail = should_fail
        self.call_count = 0
        self.timestamp = datetime.now()


@dataclass
class TestScenario:
    """Test scenario definition"""
    name: str
    description: str
    initial_state: Dict[str, Any]
    expected_outcomes: List[Dict[str, Any]]
    max_steps: int = 50
    timeout_seconds: float = 30.0
    validation_rules: List[Callable] = field(default_factory=list)


class DeterministicMockLLM:
    """
    Deterministic mock LLM provider for reliable testing
    
    Architecture Decision: Deterministic testing of non-deterministic systems
    - Predefined response patterns based on input patterns
    - Configurable failure scenarios
    - Response timing simulation
    - Call tracking and verification
    """
    
    def __init__(self, seed: int = 42):
        self.seed = seed
        self.random = random.Random(seed)
        
        # Response patterns
        self._response_patterns: Dict[str, List[MockLLMResponse]] = {}
        self._default_responses: List[MockLLMResponse] = []
        self._sequential_responses: List[MockLLMResponse] = []
        
        # Call tracking
        self.call_history: List[Dict[str, Any]] = []
        self.total_calls = 0
        
        # Failure simulation
        self.failure_rate = 0.0
        self.failure_patterns: List[str] = []
        
        # Performance simulation
        self.base_latency = 0.1
        self.latency_variance = 0.05
    
    def add_response_pattern(self, pattern: str, responses: List[MockLLMResponse]) -> None:
        """Add response pattern for specific input patterns"""
        self._response_patterns[pattern] = responses
    
    def add_default_responses(self, responses: List[MockLLMResponse]) -> None:
        """Add default responses for unmatched inputs"""
        self._default_responses.extend(responses)
    
    def set_sequential_responses(self, responses: List[MockLLMResponse]) -> None:
        """Set responses to be returned in sequence"""
        self._sequential_responses = responses
    
    def configure_failures(self, failure_rate: float, failure_patterns: List[str] = None) -> None:
        """Configure failure simulation"""
        self.failure_rate = failure_rate
        self.failure_patterns = failure_patterns or []
    
    async def complete(self, prompt: str, **kwargs) -> str:
        """Mock completion with deterministic behavior"""
        self.total_calls += 1
        call_info = {
            "call_id": self.total_calls,
            "timestamp": datetime.now(),
            "prompt": prompt,
            "kwargs": kwargs
        }
        
        # Check for failure conditions
        if self._should_fail(prompt):
            call_info["result"] = "FAILURE"
            call_info["error"] = "Simulated LLM failure"
            self.call_history.append(call_info)
            raise Exception("Simulated LLM failure")
        
        # Find appropriate response
        response = self._find_response(prompt)
        
        # Simulate latency
        await asyncio.sleep(response.latency)
        
        # Track call
        call_info["result"] = "SUCCESS"
        call_info["response"] = response.response_text
        call_info["confidence"] = response.confidence
        self.call_history.append(call_info)
        
        return response.response_text
    
    def _should_fail(self, prompt: str) -> bool:
        """Determine if this call should fail"""
        # Random failure based on failure rate
        if self.random.random() < self.failure_rate:
            return True
        
        # Pattern-based failures
        for pattern in self.failure_patterns:
            if pattern.lower() in prompt.lower():
                return True
        
        return False
    
    def _find_response(self, prompt: str) -> MockLLMResponse:
        """Find appropriate response for prompt"""
        # Use sequential responses if available
        if self._sequential_responses:
            response = self._sequential_responses.pop(0)
            response.call_count += 1
            return response
        
        # Check pattern matches
        for pattern, responses in self._response_patterns.items():
            if pattern.lower() in prompt.lower():
                response = self.random.choice(responses)
                response.call_count += 1
                return response
        
        # Use default responses
        if self._default_responses:
            response = self.random.choice(self._default_responses)
            response.call_count += 1
            return response
        
        # Fallback response
        return MockLLMResponse(
            response_text=f"Mock response for: {prompt[:50]}...",
            latency=self.base_latency + self.random.uniform(-self.latency_variance, self.latency_variance)
        )
    
    def reset(self) -> None:
        """Reset mock LLM state"""
        self.call_history.clear()
        self.total_calls = 0
        self._sequential_responses.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get mock LLM statistics"""
        successful_calls = len([c for c in self.call_history if c["result"] == "SUCCESS"])
        failed_calls = len([c for c in self.call_history if c["result"] == "FAILURE"])
        
        return {
            "total_calls": self.total_calls,
            "successful_calls": successful_calls,
            "failed_calls": failed_calls,
            "success_rate": successful_calls / max(1, self.total_calls),
            "avg_latency": sum(c.get("latency", 0) for c in self.call_history) / max(1, len(self.call_history))
        }


class MockMesaModel:
    """Mock Mesa model for testing hybrid interactions"""
    
    def __init__(self, width: int = 10, height: int = 10):
        self.width = width
        self.height = height
        self.schedule = MockScheduler()
        self.grid = MockGrid(width, height)
        self.step_count = 0
        self.running = True
        
        # Test state
        self.state_changes: List[Dict[str, Any]] = []
        self.step_history: List[Dict[str, Any]] = []
    
    def step(self) -> None:
        """Execute one model step"""
        self.step_count += 1
        
        # Record step
        step_info = {
            "step": self.step_count,
            "timestamp": datetime.now(),
            "agent_count": len(self.schedule.agents)
        }
        self.step_history.append(step_info)
        
        # Step all agents
        self.schedule.step()
    
    def add_agent(self, agent) -> None:
        """Add agent to model"""
        self.schedule.add(agent)
        agent.model = self
    
    def remove_agent(self, agent) -> None:
        """Remove agent from model"""
        self.schedule.remove(agent)


class MockScheduler:
    """Mock Mesa scheduler"""
    
    def __init__(self):
        self.agents: List[Any] = []
        self.steps = 0
    
    def add(self, agent) -> None:
        self.agents.append(agent)
    
    def remove(self, agent) -> None:
        if agent in self.agents:
            self.agents.remove(agent)
    
    def step(self) -> None:
        self.steps += 1
        for agent in self.agents:
            if hasattr(agent, 'step'):
                agent.step()


class MockGrid:
    """Mock Mesa grid"""
    
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self._grid: Dict[tuple, List[Any]] = {}
    
    def place_agent(self, agent, pos: tuple) -> None:
        if pos not in self._grid:
            self._grid[pos] = []
        self._grid[pos].append(agent)
        agent.pos = pos
    
    def move_agent(self, agent, pos: tuple) -> None:
        # Remove from old position
        if hasattr(agent, 'pos') and agent.pos in self._grid:
            self._grid[agent.pos].remove(agent)
        
        # Add to new position
        self.place_agent(agent, pos)
    
    def get_cell_list_contents(self, cell_list: List[tuple]) -> List[Any]:
        contents = []
        for pos in cell_list:
            if pos in self._grid:
                contents.extend(self._grid[pos])
        return contents
    
    def out_of_bounds(self, pos: tuple) -> bool:
        x, y = pos
        return x < 0 or x >= self.width or y < 0 or y >= self.height


class TestStateValidator:
    """Validator for test state consistency"""
    
    def __init__(self):
        self.validation_rules: List[Callable] = []
        self.validation_history: List[Dict[str, Any]] = []
    
    def add_validation_rule(self, rule: Callable) -> None:
        """Add validation rule"""
        self.validation_rules.append(rule)
    
    def validate_state(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Validate state against all rules"""
        results = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "timestamp": datetime.now()
        }
        
        for rule in self.validation_rules:
            try:
                rule_result = rule(state)
                if isinstance(rule_result, dict):
                    if not rule_result.get("valid", True):
                        results["valid"] = False
                        results["errors"].extend(rule_result.get("errors", []))
                    results["warnings"].extend(rule_result.get("warnings", []))
                elif not rule_result:
                    results["valid"] = False
                    results["errors"].append(f"Rule {rule.__name__} failed")
            except Exception as e:
                results["valid"] = False
                results["errors"].append(f"Rule {rule.__name__} exception: {e}")
        
        self.validation_history.append(results)
        return results


class HybridTestHarness:
    """
    Comprehensive test harness for hybrid architecture
    
    Architecture Decision: Comprehensive test automation
    - Supports multiple testing modes (unit, integration, system)
    - Deterministic behavior through mocking
    - Automated validation and assertion checking
    - Performance and regression testing capabilities
    """
    
    def __init__(self, test_mode: TestMode = TestMode.INTEGRATION):
        self.test_mode = test_mode
        
        # Mock components
        self.mock_llm = DeterministicMockLLM()
        self.mock_mesa_model = MockMesaModel()
        self.state_validator = TestStateValidator()
        
        # Test tracking
        self.test_results: List[Dict[str, Any]] = []
        self.current_test: Optional[str] = None
        
        # Performance tracking
        self.performance_baseline: Dict[str, float] = {}
        self.performance_thresholds: Dict[str, float] = {}
    
    def setup_test_scenario(self, scenario: TestScenario) -> None:
        """Setup test scenario"""
        self.current_test = scenario.name
        
        # Reset mock components
        self.mock_llm.reset()
        
        # Configure initial state
        self._configure_initial_state(scenario.initial_state)
        
        # Setup validation rules
        for rule in scenario.validation_rules:
            self.state_validator.add_validation_rule(rule)
    
    async def run_test_scenario(self, scenario: TestScenario) -> Dict[str, Any]:
        """Run complete test scenario"""
        start_time = datetime.now()
        
        try:
            # Setup scenario
            self.setup_test_scenario(scenario)
            
            # Create hybrid simulation with mocks
            simulation_engine = await self._create_test_simulation_engine()
            
            # Run simulation
            simulation_results = await self._run_test_simulation(
                simulation_engine, scenario.max_steps, scenario.timeout_seconds
            )
            
            # Validate results
            validation_results = self._validate_scenario_results(scenario, simulation_results)
            
            # Create test result
            test_result = {
                "scenario_name": scenario.name,
                "status": "PASSED" if validation_results["valid"] else "FAILED",
                "start_time": start_time,
                "end_time": datetime.now(),
                "duration": (datetime.now() - start_time).total_seconds(),
                "simulation_results": simulation_results,
                "validation_results": validation_results,
                "mock_llm_stats": self.mock_llm.get_stats()
            }
            
            self.test_results.append(test_result)
            return test_result
            
        except Exception as e:
            test_result = {
                "scenario_name": scenario.name,
                "status": "ERROR",
                "start_time": start_time,
                "end_time": datetime.now(),
                "duration": (datetime.now() - start_time).total_seconds(),
                "error": str(e),
                "mock_llm_stats": self.mock_llm.get_stats()
            }
            
            self.test_results.append(test_result)
            return test_result
    
    def run_performance_test(self, operation: Callable, iterations: int = 100) -> Dict[str, Any]:
        """Run performance test for specific operation"""
        results = {
            "operation": operation.__name__,
            "iterations": iterations,
            "latencies": [],
            "memory_usage": [],
            "errors": 0
        }
        
        for i in range(iterations):
            start_time = time.time()
            
            try:
                if asyncio.iscoroutinefunction(operation):
                    asyncio.run(operation())
                else:
                    operation()
                
                latency = time.time() - start_time
                results["latencies"].append(latency)
                
            except Exception:
                results["errors"] += 1
        
        # Calculate statistics
        if results["latencies"]:
            results["avg_latency"] = sum(results["latencies"]) / len(results["latencies"])
            results["min_latency"] = min(results["latencies"])
            results["max_latency"] = max(results["latencies"])
            results["p95_latency"] = sorted(results["latencies"])[int(0.95 * len(results["latencies"]))]
        
        results["error_rate"] = results["errors"] / iterations
        
        return results
    
    def assert_state_property(self, state: Dict[str, Any], property_path: str, 
                            expected_value: Any) -> None:
        """Assert that state property has expected value"""
        actual_value = self._get_nested_value(state, property_path.split('.'))
        
        if actual_value != expected_value:
            raise AssertionError(
                f"State property {property_path} expected {expected_value}, got {actual_value}"
            )
    
    def assert_performance_threshold(self, operation_name: str, actual_latency: float) -> None:
        """Assert that operation meets performance threshold"""
        threshold = self.performance_thresholds.get(operation_name)
        if threshold and actual_latency > threshold:
            raise AssertionError(
                f"Operation {operation_name} exceeded threshold: {actual_latency}s > {threshold}s"
            )
    
    def inject_fault(self, component: str, fault_type: str, **kwargs) -> None:
        """Inject fault for chaos testing"""
        if component == "llm" and fault_type == "failure":
            self.mock_llm.configure_failures(kwargs.get("failure_rate", 0.5))
        elif component == "mesa" and fault_type == "state_corruption":
            # Corrupt Mesa model state
            self.mock_mesa_model.running = False
        # Add more fault injection types as needed
    
    def _configure_initial_state(self, initial_state: Dict[str, Any]) -> None:
        """Configure initial state for test"""
        # Configure mock LLM responses
        llm_config = initial_state.get("llm_responses", {})
        for pattern, responses in llm_config.items():
            mock_responses = [
                MockLLMResponse(r["text"], r.get("confidence", 1.0), r.get("latency", 0.1))
                for r in responses
            ]
            self.mock_llm.add_response_pattern(pattern, mock_responses)
        
        # Configure Mesa model
        mesa_config = initial_state.get("mesa_model", {})
        if "agents" in mesa_config:
            for agent_config in mesa_config["agents"]:
                mock_agent = self._create_mock_mesa_agent(agent_config)
                self.mock_mesa_model.add_agent(mock_agent)
    
    def _create_mock_mesa_agent(self, config: Dict[str, Any]) -> Any:
        """Create mock Mesa agent from configuration"""
        class MockMesaAgent:
            def __init__(self, agent_id: str, pos: tuple):
                self.agent_id = agent_id
                self.pos = pos
                self.health = 1.0
                self.energy = 1.0
                self.resources = []
                self.status = "active"
            
            def step(self):
                pass
        
        agent = MockMesaAgent(config["id"], tuple(config["position"]))
        return agent
    
    async def _create_test_simulation_engine(self) -> HybridSimulationEngine:
        """Create hybrid simulation engine with test mocks"""
        # Create mock CrewAI agents
        mock_crewai_agents = self._create_mock_crewai_agents()
        
        # Create pipeline components with mocks
        from .data_flow import PerceptionPipeline
        from .core_architecture import HybridSimulationEngine
        
        perception_pipeline = PerceptionPipeline()
        
        # Create mock decision engine
        decision_engine = MockDecisionEngine(self.mock_llm)
        
        # Create mock action translator and state synchronizer
        action_translator = MockActionTranslator()
        state_synchronizer = MockStateSynchronizer()
        
        # Create simulation engine
        engine = HybridSimulationEngine(
            mesa_model=self.mock_mesa_model,
            crewai_agents=mock_crewai_agents,
            perception_pipeline=perception_pipeline,
            decision_engine=decision_engine,
            action_translator=action_translator,
            state_synchronizer=state_synchronizer
        )
        
        return engine
    
    def _create_mock_crewai_agents(self) -> List[Agent]:
        """Create mock CrewAI agents for testing"""
        agents = []
        
        for role in ["strategist", "mediator", "survivor"]:
            agent = Mock(spec=Agent)
            agent.role = role.title()
            agent.goal = f"Test goal for {role}"
            agent.backstory = f"Test backstory for {role}"
            agent.memory = True
            agents.append(agent)
        
        return agents
    
    async def _run_test_simulation(self, engine: HybridSimulationEngine, 
                                 max_steps: int, timeout: float) -> Dict[str, Any]:
        """Run simulation with test configuration"""
        engine.initialize()
        
        results = {
            "steps_completed": 0,
            "final_state": {},
            "step_history": [],
            "errors": []
        }
        
        try:
            # Run simulation steps
            for step in range(max_steps):
                step_result = await asyncio.wait_for(engine.step(), timeout=timeout)
                results["step_history"].append(step_result)
                results["steps_completed"] += 1
                
                # Check for completion conditions
                if step_result.get("simulation_complete"):
                    break
            
            # Get final state
            results["final_state"] = engine.get_simulation_state()
            
        except asyncio.TimeoutError:
            results["errors"].append("Simulation timed out")
        except Exception as e:
            results["errors"].append(f"Simulation error: {e}")
        
        return results
    
    def _validate_scenario_results(self, scenario: TestScenario, 
                                 results: Dict[str, Any]) -> Dict[str, Any]:
        """Validate scenario results against expected outcomes"""
        validation_results = {"valid": True, "errors": [], "warnings": []}
        
        # Validate expected outcomes
        for expected_outcome in scenario.expected_outcomes:
            outcome_valid = self._validate_single_outcome(expected_outcome, results)
            if not outcome_valid["valid"]:
                validation_results["valid"] = False
                validation_results["errors"].extend(outcome_valid["errors"])
        
        # Run state validation
        if results.get("final_state"):
            state_validation = self.state_validator.validate_state(results["final_state"])
            if not state_validation["valid"]:
                validation_results["valid"] = False
                validation_results["errors"].extend(state_validation["errors"])
        
        return validation_results
    
    def _validate_single_outcome(self, expected: Dict[str, Any], 
                                actual: Dict[str, Any]) -> Dict[str, Any]:
        """Validate single expected outcome"""
        result = {"valid": True, "errors": []}
        
        for key, expected_value in expected.items():
            if key not in actual:
                result["valid"] = False
                result["errors"].append(f"Missing expected key: {key}")
                continue
            
            actual_value = actual[key]
            if actual_value != expected_value:
                result["valid"] = False
                result["errors"].append(
                    f"Key {key}: expected {expected_value}, got {actual_value}"
                )
        
        return result
    
    def _get_nested_value(self, data: Dict[str, Any], path: List[str]) -> Any:
        """Get nested value from dictionary using path"""
        current = data
        for key in path:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return None
        return current
    
    def get_test_summary(self) -> Dict[str, Any]:
        """Get summary of all test results"""
        if not self.test_results:
            return {"status": "no_tests_run"}
        
        total_tests = len(self.test_results)
        passed_tests = len([r for r in self.test_results if r["status"] == "PASSED"])
        failed_tests = len([r for r in self.test_results if r["status"] == "FAILED"])
        error_tests = len([r for r in self.test_results if r["status"] == "ERROR"])
        
        return {
            "total_tests": total_tests,
            "passed": passed_tests,
            "failed": failed_tests,
            "errors": error_tests,
            "success_rate": passed_tests / total_tests,
            "total_duration": sum(r["duration"] for r in self.test_results)
        }


# Mock implementations for testing

class MockDecisionEngine:
    """Mock decision engine for testing"""
    
    def __init__(self, mock_llm: DeterministicMockLLM):
        self.mock_llm = mock_llm
    
    async def reason_and_decide(self, perceptions: Dict[str, PerceptionData]) -> Dict[str, DecisionData]:
        decisions = {}
        
        for agent_id, perception in perceptions.items():
            # Use mock LLM to generate decision
            prompt = f"Agent {agent_id} decision based on: {perception.spatial_data}"
            response = await self.mock_llm.complete(prompt)
            
            # Parse response into decision
            decision = DecisionData(
                agent_id=agent_id,
                timestamp=datetime.now(),
                chosen_action="test_action",
                action_parameters={"test": "parameter"},
                reasoning=response,
                confidence_level=0.8,
                fallback_actions=["fallback_action"]
            )
            
            decisions[agent_id] = decision
        
        return decisions
    
    def update_agent_memory(self, agent_id: str, experience: Dict[str, Any]) -> None:
        pass


class MockActionTranslator:
    """Mock action translator for testing"""
    
    def translate_decision(self, decision: DecisionData):
        from .core_architecture import MesaAction
        return MesaAction(
            agent_id=decision.agent_id,
            action_type=decision.chosen_action,
            parameters=decision.action_parameters,
            expected_duration=1.0,
            prerequisites=[]
        )
    
    def validate_action(self, action, mesa_model) -> bool:
        return True


class MockStateSynchronizer:
    """Mock state synchronizer for testing"""
    
    def sync_mesa_to_crewai(self, mesa_model) -> None:
        pass
    
    def sync_crewai_to_mesa(self, decisions: Dict[str, DecisionData], mesa_model) -> None:
        pass


# Test utilities and fixtures

def create_test_scenario(name: str, **kwargs) -> TestScenario:
    """Create test scenario with defaults"""
    return TestScenario(
        name=name,
        description=kwargs.get("description", f"Test scenario: {name}"),
        initial_state=kwargs.get("initial_state", {}),
        expected_outcomes=kwargs.get("expected_outcomes", []),
        max_steps=kwargs.get("max_steps", 10),
        timeout_seconds=kwargs.get("timeout_seconds", 10.0),
        validation_rules=kwargs.get("validation_rules", [])
    )


def create_llm_response(text: str, **kwargs) -> MockLLMResponse:
    """Create mock LLM response with defaults"""
    return MockLLMResponse(
        response_text=text,
        confidence=kwargs.get("confidence", 1.0),
        latency=kwargs.get("latency", 0.1),
        should_fail=kwargs.get("should_fail", False)
    )


# Standard validation rules

def validate_agent_positions_in_bounds(state: Dict[str, Any]) -> Dict[str, Any]:
    """Validate that all agent positions are within bounds"""
    result = {"valid": True, "errors": [], "warnings": []}
    
    agents = state.get("agents", {})
    environment = state.get("environment", {})
    bounds = environment.get("bounds", {"width": 10, "height": 10})
    
    for agent_id, agent_data in agents.items():
        position = agent_data.get("position")
        if position:
            x, y = position
            if x < 0 or x >= bounds["width"] or y < 0 or y >= bounds["height"]:
                result["valid"] = False
                result["errors"].append(f"Agent {agent_id} position {position} out of bounds")
    
    return result


def validate_resource_conservation(state: Dict[str, Any]) -> Dict[str, Any]:
    """Validate that resources are conserved (not created or destroyed)"""
    result = {"valid": True, "errors": [], "warnings": []}
    
    # This would implement resource conservation checking
    # based on the specific resource model
    
    return result


def validate_state_consistency(state: Dict[str, Any]) -> Dict[str, Any]:
    """Validate overall state consistency"""
    result = {"valid": True, "errors": [], "warnings": []}
    
    # Check for required state components
    required_components = ["agents", "environment", "model"]
    for component in required_components:
        if component not in state:
            result["valid"] = False
            result["errors"].append(f"Missing required state component: {component}")
    
    return result


# Example test usage

class ExampleHybridTest(unittest.TestCase):
    """Example test class showing hybrid testing patterns"""
    
    def setUp(self):
        self.test_harness = HybridTestHarness(TestMode.INTEGRATION)
    
    async def test_basic_simulation_flow(self):
        """Test basic simulation flow with deterministic responses"""
        # Create test scenario
        scenario = create_test_scenario(
            "basic_flow",
            description="Test basic simulation flow",
            initial_state={
                "llm_responses": {
                    "analyze": [{"text": "I will analyze the situation carefully."}],
                    "decision": [{"text": "I choose to move north."}]
                },
                "mesa_model": {
                    "agents": [
                        {"id": "strategist", "position": [5, 5]},
                        {"id": "mediator", "position": [4, 5]},
                        {"id": "survivor", "position": [6, 5]}
                    ]
                }
            },
            expected_outcomes=[
                {"steps_completed": 5}
            ],
            validation_rules=[
                validate_agent_positions_in_bounds,
                validate_state_consistency
            ]
        )
        
        # Run test
        result = await self.test_harness.run_test_scenario(scenario)
        
        # Assertions
        self.assertEqual(result["status"], "PASSED")
        self.assertGreater(result["simulation_results"]["steps_completed"], 0)
    
    def test_performance_regression(self):
        """Test performance regression"""
        def dummy_operation():
            time.sleep(0.01)  # Simulate work
        
        # Run performance test
        perf_result = self.test_harness.run_performance_test(dummy_operation, iterations=10)
        
        # Assert performance thresholds
        self.assertLess(perf_result["avg_latency"], 0.05)  # Should be under 50ms
        self.assertEqual(perf_result["errors"], 0)
    
    def test_fault_injection(self):
        """Test system behavior with faults injected"""
        # Inject LLM failures
        self.test_harness.inject_fault("llm", "failure", failure_rate=0.3)
        
        # System should handle gracefully
        # Additional test logic would go here