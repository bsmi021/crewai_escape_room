"""
Unit tests for Async Decision Engine Implementation

Tests implement TDD methodology for Phase 2 Week 3: Advanced Decision Engine
Following the architectural specifications for Agent B.
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any, List
from unittest.mock import Mock, MagicMock, patch, AsyncMock
import concurrent.futures

from src.escape_room_sim.hybrid.core_architecture import (
    PerceptionData, DecisionData, ComponentState
)


class TestAsyncDecisionEngine:
    """Test advanced async decision engine with circuit breakers"""
    
    @pytest.fixture
    def perception_handoff_data(self):
        """Create mock perception handoff data from Agent A"""
        from dataclasses import dataclass
        
        @dataclass
        class PerceptionHandoff:
            perceptions: Dict[str, PerceptionData]
            performance_metrics: Dict[str, float]
            extraction_timestamp: datetime
            validation_passed: bool
            mesa_model_hash: str
            agent_count: int
        
        perceptions = {
            "strategist": PerceptionData(
                agent_id="strategist",
                timestamp=datetime.now(),
                spatial_data={"current_position": (2, 3), "obstacles": []},
                environmental_state={"lighting": "dim", "temperature": 18},
                nearby_agents=["mediator"],
                available_actions=["move", "examine", "analyze", "communicate"],
                resources={"energy": 0.8, "tools": []},
                constraints={"movement_range": 2, "action_points": 3}
            ),
            "mediator": PerceptionData(
                agent_id="mediator",
                timestamp=datetime.now(),
                spatial_data={"current_position": (3, 3), "obstacles": []},
                environmental_state={"lighting": "dim", "temperature": 18},
                nearby_agents=["strategist", "survivor"],
                available_actions=["move", "communicate", "coordinate", "mediate"],
                resources={"energy": 0.6, "tools": []},
                constraints={"movement_range": 2, "action_points": 2}
            ),
            "survivor": PerceptionData(
                agent_id="survivor",
                timestamp=datetime.now(),
                spatial_data={"current_position": (1, 2), "obstacles": []},
                environmental_state={"lighting": "dim", "temperature": 18},
                nearby_agents=["mediator"],
                available_actions=["move", "examine", "use_tool", "survive"],
                resources={"energy": 0.9, "tools": ["flashlight"]},
                constraints={"movement_range": 3, "action_points": 4}
            )
        }
        
        return PerceptionHandoff(
            perceptions=perceptions,
            performance_metrics={"extraction_time": 0.15, "validation_score": 0.95},
            extraction_timestamp=datetime.now(),
            validation_passed=True,
            mesa_model_hash="abc123def456",
            agent_count=3
        )
    
    @pytest.fixture
    def mock_crewai_agents(self):
        """Create mock CrewAI agents with async capabilities"""
        strategist = Mock()
        strategist.role = "Strategist"
        strategist.memory = Mock()
        strategist.execute_async = AsyncMock(return_value="analyze the puzzle structure")
        
        mediator = Mock()
        mediator.role = "Mediator"
        mediator.memory = Mock()
        mediator.execute_async = AsyncMock(return_value="coordinate with team members")
        
        survivor = Mock()
        survivor.role = "Survivor"
        survivor.memory = Mock()
        survivor.execute_async = AsyncMock(return_value="search for escape tools")
        
        return [strategist, mediator, survivor]
    
    @pytest.mark.asyncio
    async def test_async_decision_engine_initialization(self, mock_crewai_agents):
        """Test AsyncDecisionEngine initialization with circuit breaker"""
        from src.escape_room_sim.hybrid.decision.async_engine import AsyncDecisionEngine
        
        config = {
            "circuit_breaker": {
                "failure_threshold": 5,
                "recovery_timeout": 30.0,
                "timeout": 10.0
            },
            "concurrency": {
                "max_concurrent_decisions": 3,
                "decision_timeout": 5.0
            }
        }
        
        engine = AsyncDecisionEngine(mock_crewai_agents, config)
        
        assert engine.state == ComponentState.UNINITIALIZED
        assert engine.circuit_breaker is not None
        assert engine.max_concurrent == 3
        assert engine.decision_timeout == 5.0
        
        # Initialize
        await engine.initialize()
        assert engine.state == ComponentState.READY
    
    @pytest.mark.asyncio
    async def test_concurrent_decision_generation(self, mock_crewai_agents, perception_handoff_data):
        """Test concurrent decision generation for all agents"""
        from src.escape_room_sim.hybrid.decision.async_engine import AsyncDecisionEngine
        
        config = {
            "circuit_breaker": {
                "failure_threshold": 5,
                "recovery_timeout": 30.0,
                "timeout": 10.0
            },
            "concurrency": {
                "max_concurrent_decisions": 3,
                "decision_timeout": 5.0
            }
        }
        
        engine = AsyncDecisionEngine(mock_crewai_agents, config)
        await engine.initialize()
        
        # Test concurrent decision generation
        start_time = datetime.now()
        decisions = await engine.reason_and_decide(perception_handoff_data.perceptions)
        end_time = datetime.now()
        
        # Should complete within reasonable time (concurrent processing)
        assert (end_time - start_time).total_seconds() < 2.0
        
        # Should have decisions for all agents
        assert len(decisions) == 3
        assert "strategist" in decisions
        assert "mediator" in decisions
        assert "survivor" in decisions
        
        # Verify decision structure
        for agent_id, decision in decisions.items():
            assert isinstance(decision, DecisionData)
            assert decision.agent_id == agent_id
            assert decision.confidence_level >= 0.0
            assert decision.confidence_level <= 1.0
            assert len(decision.fallback_actions) > 0
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_failure_handling(self, mock_crewai_agents, perception_handoff_data):
        """Test circuit breaker opens on LLM failures"""
        from src.escape_room_sim.hybrid.decision.async_engine import AsyncDecisionEngine
        
        # Configure tight circuit breaker for testing
        config = {
            "circuit_breaker": {
                "failure_threshold": 2,  # Low threshold for testing
                "recovery_timeout": 30.0,
                "timeout": 1.0  # Short timeout
            },
            "concurrency": {
                "max_concurrent_decisions": 3,
                "decision_timeout": 0.5  # Very short timeout to trigger failures
            }
        }
        
        # Make LLM calls fail
        for agent in mock_crewai_agents:
            agent.execute_async = AsyncMock(side_effect=asyncio.TimeoutError("LLM timeout"))
        
        engine = AsyncDecisionEngine(mock_crewai_agents, config)
        await engine.initialize()
        
        # First failure
        decisions1 = await engine.reason_and_decide(perception_handoff_data.perceptions)
        assert engine.circuit_breaker.failure_count == 1
        
        # Second failure should open circuit
        decisions2 = await engine.reason_and_decide(perception_handoff_data.perceptions)
        assert engine.circuit_breaker.is_open
        
        # Third call should use fallback immediately
        start_time = datetime.now()
        decisions3 = await engine.reason_and_decide(perception_handoff_data.perceptions)
        end_time = datetime.now()
        
        # Should be very fast (no LLM call)
        assert (end_time - start_time).total_seconds() < 0.1
        
        # Should still return valid fallback decisions
        assert len(decisions3) == 3
        for decision in decisions3.values():
            assert decision.chosen_action in ["observe", "wait", "examine"]
            assert decision.confidence_level < 0.6  # Lower confidence for fallbacks
    
    @pytest.mark.asyncio
    async def test_timeout_handling(self, mock_crewai_agents, perception_handoff_data):
        """Test LLM timeout handling and fallback"""
        from src.escape_room_sim.hybrid.decision.async_engine import AsyncDecisionEngine
        
        config = {
            "circuit_breaker": {
                "failure_threshold": 5,
                "recovery_timeout": 30.0,
                "timeout": 10.0
            },
            "concurrency": {
                "max_concurrent_decisions": 3,
                "decision_timeout": 0.1  # Very short timeout
            }
        }
        
        # Make one agent take too long
        mock_crewai_agents[0].execute_async = AsyncMock(
            side_effect=lambda *args, **kwargs: asyncio.sleep(1.0)  # Takes 1s, timeout is 0.1s
        )
        
        engine = AsyncDecisionEngine(mock_crewai_agents, config)
        await engine.initialize()
        
        decisions = await engine.reason_and_decide(perception_handoff_data.perceptions)
        
        # Should still get decisions for all agents
        assert len(decisions) == 3
        
        # Strategist should have fallback decision due to timeout
        strategist_decision = decisions["strategist"]
        assert strategist_decision.chosen_action in ["observe", "wait", "examine"]
        assert strategist_decision.reasoning.startswith("Fallback decision")
    
    @pytest.mark.asyncio
    async def test_decision_confidence_scoring(self, mock_crewai_agents, perception_handoff_data):
        """Test decision confidence scoring based on LLM response quality"""
        from src.escape_room_sim.hybrid.decision.async_engine import AsyncDecisionEngine
        
        # Set up different quality responses
        mock_crewai_agents[0].execute_async = AsyncMock(
            return_value="I will carefully analyze the puzzle structure and identify key patterns"
        )
        mock_crewai_agents[1].execute_async = AsyncMock(
            return_value="coordinate"  # Short response
        )
        mock_crewai_agents[2].execute_async = AsyncMock(
            return_value="maybe move or something idk"  # Uncertain response
        )
        
        config = {
            "circuit_breaker": {
                "failure_threshold": 5,
                "recovery_timeout": 30.0,
                "timeout": 10.0
            },
            "concurrency": {
                "max_concurrent_decisions": 3,
                "decision_timeout": 5.0
            }
        }
        
        engine = AsyncDecisionEngine(mock_crewai_agents, config)
        await engine.initialize()
        
        decisions = await engine.reason_and_decide(perception_handoff_data.perceptions)
        
        # Strategist should have high confidence (detailed response)
        assert decisions["strategist"].confidence_level > 0.7
        
        # Mediator should have medium confidence (short but relevant)
        assert 0.4 <= decisions["mediator"].confidence_level <= 0.8
        
        # Survivor should have lower confidence (uncertain response)
        assert decisions["survivor"].confidence_level < 0.7
    
    @pytest.mark.asyncio
    async def test_agent_memory_integration(self, mock_crewai_agents, perception_handoff_data):
        """Test agent memory integration for context-aware decisions"""
        from src.escape_room_sim.hybrid.decision.async_engine import AsyncDecisionEngine
        
        config = {
            "circuit_breaker": {
                "failure_threshold": 5,
                "recovery_timeout": 30.0,
                "timeout": 10.0
            },
            "memory": {
                "context_window": 5,
                "memory_decay": 0.9
            }
        }
        
        engine = AsyncDecisionEngine(mock_crewai_agents, config)
        await engine.initialize()
        
        # Add some memory context
        memory_context = {
            "previous_actions": ["examine", "move"],
            "failed_attempts": ["open_door"],
            "successful_strategies": ["team_coordination"],
            "discovered_objects": ["key", "flashlight"]
        }
        
        engine.update_agent_memory("strategist", memory_context)
        
        # Generate decision with memory context
        decisions = await engine.reason_and_decide(perception_handoff_data.perceptions)
        
        strategist_decision = decisions["strategist"]
        
        # Should avoid failed attempts
        assert strategist_decision.chosen_action != "open_door"
        
        # Should prefer successful strategies
        assert "coordinate" in strategist_decision.reasoning.lower() or \
               "team" in strategist_decision.reasoning.lower()
    
    @pytest.mark.asyncio
    async def test_performance_target_compliance(self, mock_crewai_agents, perception_handoff_data):
        """Test that decision generation meets performance targets"""
        from src.escape_room_sim.hybrid.decision.async_engine import AsyncDecisionEngine
        
        config = {
            "circuit_breaker": {
                "failure_threshold": 5,
                "recovery_timeout": 30.0,
                "timeout": 10.0
            },
            "concurrency": {
                "max_concurrent_decisions": 3,
                "decision_timeout": 5.0
            }
        }
        
        engine = AsyncDecisionEngine(mock_crewai_agents, config)
        await engine.initialize()
        
        # Multiple decision rounds to test consistency
        total_time = 0
        success_count = 0
        
        for _ in range(5):
            start_time = datetime.now()
            decisions = await engine.reason_and_decide(perception_handoff_data.perceptions)
            end_time = datetime.now()
            
            round_time = (end_time - start_time).total_seconds()
            total_time += round_time
            
            # Each round should complete in < 1s (performance target)
            assert round_time < 1.0
            
            # Should have valid decisions
            assert len(decisions) == 3
            success_count += 1
        
        # Overall success rate should be > 95%
        success_rate = success_count / 5
        assert success_rate > 0.95
        
        # Average decision time should be reasonable
        avg_time = total_time / 5
        assert avg_time < 0.8


class TestDecisionHandoffProtocol:
    """Test decision handoff protocol to Agent C"""
    
    @pytest.fixture
    def sample_decisions(self):
        """Create sample decisions for handoff testing"""
        return {
            "strategist": DecisionData(
                agent_id="strategist",
                timestamp=datetime.now(),
                chosen_action="analyze",
                action_parameters={"target": "puzzle", "depth": "detailed"},
                reasoning="Complex puzzle requires systematic analysis",
                confidence_level=0.85,
                fallback_actions=["examine", "observe"]
            ),
            "mediator": DecisionData(
                agent_id="mediator",
                timestamp=datetime.now(),
                chosen_action="coordinate",
                action_parameters={"targets": ["strategist", "survivor"], "plan": "analysis_support"},
                reasoning="Team coordination will improve analysis efficiency",
                confidence_level=0.90,
                fallback_actions=["communicate", "wait"]
            ),
            "survivor": DecisionData(
                agent_id="survivor",
                timestamp=datetime.now(),
                chosen_action="search",
                action_parameters={"area": "nearby", "target": "tools"},
                reasoning="Need tools to support team efforts",
                confidence_level=0.75,
                fallback_actions=["move", "examine"]
            )
        }
    
    def test_decision_handoff_creation(self, sample_decisions):
        """Test DecisionHandoff protocol structure"""
        from src.escape_room_sim.hybrid.decision.handoff_protocol import DecisionHandoff
        
        handoff = DecisionHandoff(
            decisions=sample_decisions,
            reasoning_confidence={"strategist": 0.85, "mediator": 0.90, "survivor": 0.75},
            decision_timestamp=datetime.now(),
            llm_response_time=0.45,
            negotiation_outcomes={"team_plan": "coordinated_analysis", "resource_allocation": "shared"},
            fallback_decisions_used=[]
        )
        
        assert len(handoff.decisions) == 3
        assert handoff.llm_response_time == 0.45
        assert "team_plan" in handoff.negotiation_outcomes
        assert len(handoff.fallback_decisions_used) == 0
    
    def test_handoff_with_fallbacks(self, sample_decisions):
        """Test handoff protocol when fallback decisions are used"""
        from src.escape_room_sim.hybrid.decision.handoff_protocol import DecisionHandoff
        
        # Simulate some fallback usage
        sample_decisions["strategist"].chosen_action = "observe"  # Fallback action
        sample_decisions["strategist"].confidence_level = 0.5
        
        handoff = DecisionHandoff(
            decisions=sample_decisions,
            reasoning_confidence={"strategist": 0.5, "mediator": 0.90, "survivor": 0.75},
            decision_timestamp=datetime.now(),
            llm_response_time=10.1,  # Timeout occurred
            negotiation_outcomes={},
            fallback_decisions_used=["strategist"]
        )
        
        assert "strategist" in handoff.fallback_decisions_used
        assert handoff.reasoning_confidence["strategist"] == 0.5
        assert handoff.llm_response_time > 10.0  # Indicates timeout
    
    def test_handoff_serialization(self, sample_decisions):
        """Test handoff can be serialized for inter-agent communication"""
        from src.escape_room_sim.hybrid.decision.handoff_protocol import DecisionHandoff
        import json
        
        handoff = DecisionHandoff(
            decisions=sample_decisions,
            reasoning_confidence={"strategist": 0.85, "mediator": 0.90, "survivor": 0.75},
            decision_timestamp=datetime.now(),
            llm_response_time=0.45,
            negotiation_outcomes={"team_plan": "coordinated_analysis"},
            fallback_decisions_used=[]
        )
        
        # Should be serializable to JSON for Agent C
        serialized = handoff.to_dict()
        assert isinstance(serialized, dict)
        assert "decisions" in serialized
        assert "reasoning_confidence" in serialized
        assert "negotiation_outcomes" in serialized


if __name__ == "__main__":
    pytest.main([__file__, "-v"])