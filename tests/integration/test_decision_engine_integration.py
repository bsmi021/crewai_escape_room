"""
Integration tests for Decision Engine with existing CrewAI agents

Tests integration between the new async decision engine and the existing
escape room simulation agents (strategist, mediator, survivor).
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any, List
from unittest.mock import Mock, MagicMock, patch, AsyncMock
import tempfile
import json

from src.escape_room_sim.hybrid.core_architecture import (
    PerceptionData, DecisionData, ComponentState
)


class TestCrewAIAgentIntegration:
    """Test integration with existing CrewAI agents"""
    
    @pytest.fixture
    def real_crewai_agents(self):
        """Create real CrewAI agents from existing codebase"""
        from src.escape_room_sim.agents.strategist import create_strategist_agent
        from src.escape_room_sim.agents.mediator import create_mediator_agent
        from src.escape_room_sim.agents.survivor import create_survivor_agent
        
        # Create real agents with test configuration
        strategist = create_strategist_agent()
        mediator = create_mediator_agent()
        survivor = create_survivor_agent()
        
        return [strategist, mediator, survivor]
    
    @pytest.fixture
    def escape_room_perceptions(self):
        """Create realistic escape room perceptions"""
        return {
            "strategist": PerceptionData(
                agent_id="strategist",
                timestamp=datetime.now(),
                spatial_data={
                    "current_position": (5, 5),
                    "visible_area": [(4,4), (4,5), (4,6), (5,4), (5,6), (6,4), (6,5), (6,6)],
                    "room_bounds": {"width": 10, "height": 10},
                    "obstacles": [(2, 3), (7, 8)],
                    "nearby_objects": {"puzzle_box": (4, 5), "key": (6, 6)}
                },
                environmental_state={
                    "lighting": "dim",
                    "temperature": 18,
                    "time_remaining": 45.0,
                    "time_pressure": 0.6,
                    "puzzle_complexity": "high",
                    "room_state": {"locked_doors": ["north_exit"], "solved_puzzles": []}
                },
                nearby_agents=["mediator"],
                available_actions=[
                    "move", "examine", "analyze", "communicate", "solve_puzzle",
                    "coordinate", "assess_risk", "plan"
                ],
                resources={
                    "energy": 0.8,
                    "analytical_points": 5,
                    "tools": [],
                    "knowledge": ["room_layout", "puzzle_mechanics"]
                },
                constraints={
                    "movement_range": 2,
                    "action_points": 3,
                    "analysis_time_limit": 10.0,
                    "cooperation_required": True
                }
            ),
            "mediator": PerceptionData(
                agent_id="mediator",
                timestamp=datetime.now(),
                spatial_data={
                    "current_position": (4, 5),
                    "visible_area": [(3,4), (3,5), (3,6), (4,4), (4,6), (5,4), (5,5), (5,6)],
                    "room_bounds": {"width": 10, "height": 10},
                    "obstacles": [(2, 3), (7, 8)],
                    "nearby_objects": {"puzzle_box": (4, 5), "strategist": (5, 5)}
                },
                environmental_state={
                    "team_cohesion": 0.7,
                    "communication_clarity": 0.8,
                    "conflict_level": 0.2,
                    "time_pressure": 0.6,
                    "coordination_opportunities": ["puzzle_solving", "resource_sharing"]
                },
                nearby_agents=["strategist", "survivor"], 
                available_actions=[
                    "move", "communicate", "coordinate", "mediate", "facilitate",
                    "negotiate", "resolve_conflict", "build_consensus"
                ],
                resources={
                    "energy": 0.7,
                    "social_influence": 4,
                    "tools": ["communication_device"],
                    "relationships": {"strategist": 0.8, "survivor": 0.6}
                },
                constraints={
                    "movement_range": 2,
                    "coordination_range": 5,
                    "mediation_energy_cost": 0.1,
                    "max_simultaneous_negotiations": 2
                }
            ),
            "survivor": PerceptionData(
                agent_id="survivor",
                timestamp=datetime.now(),
                spatial_data={
                    "current_position": (3, 7),
                    "visible_area": [(2,6), (2,7), (2,8), (3,6), (3,8), (4,6), (4,7), (4,8)],
                    "room_bounds": {"width": 10, "height": 10},
                    "obstacles": [(2, 3), (7, 8)],
                    "nearby_objects": {"tool_chest": (2, 8), "rope": (4, 7)}
                },
                environmental_state={
                    "threat_level": 0.4,
                    "resource_scarcity": 0.8,
                    "escape_route_analysis": {"north_exit": "blocked", "south_window": "possible"},
                    "survival_priority": 0.9,
                    "time_pressure": 0.6
                },
                nearby_agents=["mediator"],
                available_actions=[
                    "move", "use_tool", "search", "survive", "escape_attempt",
                    "resource_management", "risk_assessment", "emergency_action"
                ],
                resources={
                    "energy": 0.9,
                    "survival_points": 6,
                    "tools": ["flashlight", "rope", "multi_tool"],
                    "discovered_items": ["key_fragment", "rope"]
                },
                constraints={
                    "movement_range": 3,
                    "tool_durability": {"flashlight": 0.7, "rope": 0.9, "multi_tool": 0.8},
                    "energy_consumption_rate": 0.1,
                    "risk_tolerance": 0.6
                }
            )
        }
    
    @pytest.mark.asyncio
    async def test_real_agent_decision_generation(self, real_crewai_agents, escape_room_perceptions):
        """Test decision generation with real CrewAI agents"""
        from src.escape_room_sim.hybrid.decision.async_engine import AsyncDecisionEngine
        
        config = {
            "circuit_breaker": {
                "failure_threshold": 5,
                "recovery_timeout": 30.0,
                "timeout": 10.0
            },
            "concurrency": {
                "max_concurrent_decisions": 3,
                "decision_timeout": 8.0  # Longer for real LLM calls
            }
        }
        
        engine = AsyncDecisionEngine(real_crewai_agents, config)
        await engine.initialize()
        
        # Generate decisions with real agents
        decisions = await engine.reason_and_decide(escape_room_perceptions)
        
        assert len(decisions) == 3
        assert "strategist" in decisions
        assert "mediator" in decisions
        assert "survivor" in decisions
        
        # Verify each agent's decision aligns with their role
        strategist_decision = decisions["strategist"]
        assert strategist_decision.chosen_action in [
            "analyze", "examine", "assess_risk", "plan", "solve_puzzle"
        ]
        assert "analyz" in strategist_decision.reasoning.lower() or \
               "strateg" in strategist_decision.reasoning.lower()
        
        mediator_decision = decisions["mediator"]
        assert mediator_decision.chosen_action in [
            "communicate", "coordinate", "mediate", "negotiate", "facilitate"
        ]
        assert "team" in mediator_decision.reasoning.lower() or \
               "coordinat" in mediator_decision.reasoning.lower()
        
        survivor_decision = decisions["survivor"]
        assert survivor_decision.chosen_action in [
            "use_tool", "search", "survive", "escape_attempt", "move"
        ]
        assert "surviv" in survivor_decision.reasoning.lower() or \
               "tool" in survivor_decision.reasoning.lower()
    
    @pytest.mark.asyncio
    async def test_memory_system_integration(self, real_crewai_agents, escape_room_perceptions):
        """Test integration with existing memory system"""
        from src.escape_room_sim.hybrid.decision.async_engine import AsyncDecisionEngine
        from src.escape_room_sim.memory.persistent_memory import PersistentMemoryManager
        
        # Create memory manager
        with tempfile.TemporaryDirectory() as temp_dir:
            memory_config = {
                "memory_dir": temp_dir,
                "max_memories": 100,
                "similarity_threshold": 0.7
            }
            
            memory_manager = PersistentMemoryManager(memory_config)
            
            engine_config = {
                "circuit_breaker": {
                    "failure_threshold": 5,
                    "recovery_timeout": 30.0,
                    "timeout": 10.0
                },
                "memory_integration": True,
                "memory_manager": memory_manager
            }
            
            engine = AsyncDecisionEngine(real_crewai_agents, engine_config)
            await engine.initialize()
            
            # Add some memory context
            memory_context = {
                "previous_decisions": [
                    {"action": "examine", "result": "found_clue", "success": True},
                    {"action": "solve_puzzle", "result": "failed", "success": False}
                ],
                "learned_patterns": ["puzzle_requires_coordination", "time_pressure_high"],
                "agent_relationships": {"trust_level": 0.8, "cooperation_history": "positive"}
            }
            
            engine.update_agent_memory("strategist", memory_context)
            
            # Generate decisions with memory context
            decisions = await engine.reason_and_decide(escape_room_perceptions)
            
            strategist_decision = decisions["strategist"]
            
            # Should avoid failed actions
            assert strategist_decision.chosen_action != "solve_puzzle" or \
                   "careful" in strategist_decision.reasoning.lower()
            
            # Should consider learned patterns
            assert "coordinat" in strategist_decision.reasoning.lower() or \
                   "team" in strategist_decision.reasoning.lower()
    
    @pytest.mark.asyncio
    async def test_trust_system_integration(self, real_crewai_agents, escape_room_perceptions):
        """Test integration with existing trust tracking system"""
        from src.escape_room_sim.hybrid.decision.async_engine import AsyncDecisionEngine
        from src.escape_room_sim.competitive.trust_tracker import TrustTracker
        
        # Create trust tracker
        trust_tracker = TrustTracker()
        
        # Set initial trust relationships
        trust_tracker.update_trust("strategist", "mediator", 0.8, "successful_coordination")
        trust_tracker.update_trust("strategist", "survivor", 0.6, "resource_sharing")
        trust_tracker.update_trust("mediator", "survivor", 0.7, "communication")
        
        engine_config = {
            "circuit_breaker": {
                "failure_threshold": 5,
                "recovery_timeout": 30.0,
                "timeout": 10.0
            },
            "trust_integration": True,
            "trust_tracker": trust_tracker
        }
        
        engine = AsyncDecisionEngine(real_crewai_agents, engine_config)
        await engine.initialize()
        
        # Generate decisions with trust context
        decisions = await engine.reason_and_decide(escape_room_perceptions)
        
        # Decisions should reflect trust levels
        mediator_decision = decisions["mediator"]
        
        # Should prefer high-trust agents for coordination
        if "coordinate" in mediator_decision.chosen_action:
            target = mediator_decision.action_parameters.get("target")
            if target:
                # Should prefer strategist (higher trust) over survivor
                assert target == "strategist" or target == "team"
    
    @pytest.mark.asyncio
    async def test_competitive_scenario_integration(self, real_crewai_agents, escape_room_perceptions):
        """Test integration with competitive escape room scenarios"""
        from src.escape_room_sim.hybrid.decision.async_engine import AsyncDecisionEngine
        from src.escape_room_sim.competitive.scenario_generator import ScenarioGenerator
        
        # Generate competitive scenario
        scenario_generator = ScenarioGenerator()
        competitive_scenario = scenario_generator.generate_scenario("resource_scarcity")
        
        # Modify perceptions for competitive context
        competitive_perceptions = escape_room_perceptions.copy()
        for agent_id, perception in competitive_perceptions.items():
            # Add competitive elements
            perception.environmental_state["competition_level"] = 0.7
            perception.environmental_state["resource_scarcity"] = 0.9
            perception.constraints["survival_requirement"] = "only_two_can_escape"
            
            # Add competitive actions
            perception.available_actions.extend([
                "compete_for_resource", "form_alliance", "betray", "hoard_resource"
            ])
        
        engine_config = {
            "circuit_breaker": {
                "failure_threshold": 5,
                "recovery_timeout": 30.0,
                "timeout": 10.0
            },
            "competitive_mode": True,
            "scenario": competitive_scenario
        }
        
        engine = AsyncDecisionEngine(real_crewai_agents, engine_config)
        await engine.initialize()
        
        decisions = await engine.reason_and_decide(competitive_perceptions)
        
        # Decisions should reflect competitive nature
        for agent_id, decision in decisions.items():
            # Should consider competitive elements
            assert decision.confidence_level > 0.0
            
            # Different agents should show different competitive behaviors
            if agent_id == "strategist":
                # Strategist might focus on long-term competitive advantage
                assert any(word in decision.reasoning.lower() 
                          for word in ["advantage", "strategy", "plan", "analysis"])
            elif agent_id == "survivor":
                # Survivor might be more direct/aggressive
                assert any(word in decision.reasoning.lower()
                          for word in ["surviv", "resource", "escape", "tool"])
    
    @pytest.mark.asyncio
    async def test_end_to_end_simulation_step(self, real_crewai_agents, escape_room_perceptions):
        """Test end-to-end simulation step with decision engine"""
        from src.escape_room_sim.hybrid.decision.async_engine import AsyncDecisionEngine
        from src.escape_room_sim.simulation.simulation_engine import SimulationEngine
        
        # Create full simulation context
        engine_config = {
            "circuit_breaker": {
                "failure_threshold": 5,
                "recovery_timeout": 30.0,
                "timeout": 10.0
            },
            "integration_mode": True
        }
        
        decision_engine = AsyncDecisionEngine(real_crewai_agents, engine_config)
        await decision_engine.initialize()
        
        # Simulate one complete decision cycle
        start_time = datetime.now()
        
        # 1. Generate decisions
        decisions = await decision_engine.reason_and_decide(escape_room_perceptions)
        
        # 2. Validate decisions are realistic and executable
        for agent_id, decision in decisions.items():
            perception = escape_room_perceptions[agent_id]
            
            # Decision should be from available actions
            assert decision.chosen_action in perception.available_actions
            
            # Should have reasonable confidence
            assert 0.1 <= decision.confidence_level <= 1.0
            
            # Should have fallback actions
            assert len(decision.fallback_actions) > 0
            
            # Should have some reasoning
            assert len(decision.reasoning) > 10
        
        end_time = datetime.now()
        execution_time = (end_time - start_time).total_seconds()
        
        # Should meet performance requirements
        assert execution_time < 2.0  # Complete cycle in under 2 seconds
        
        # 3. Create handoff for Agent C (Action Translation)
        from src.escape_room_sim.hybrid.decision.handoff_protocol import DecisionHandoff
        
        handoff = DecisionHandoff(
            decisions=decisions,
            reasoning_confidence={aid: d.confidence_level for aid, d in decisions.items()},
            decision_timestamp=datetime.now(),
            llm_response_time=execution_time,
            negotiation_outcomes={},
            fallback_decisions_used=[]
        )
        
        assert len(handoff.decisions) == 3
        assert handoff.llm_response_time > 0
        
        # Verify handoff can be serialized for Agent C
        handoff_dict = handoff.to_dict()
        assert isinstance(handoff_dict, dict)
        assert "decisions" in handoff_dict


class TestErrorHandlingIntegration:
    """Test error handling integration with existing systems"""
    
    @pytest.mark.asyncio
    async def test_llm_failure_with_existing_fallbacks(self):
        """Test LLM failures fall back to existing rule-based systems"""
        from src.escape_room_sim.hybrid.decision.async_engine import AsyncDecisionEngine
        from src.escape_room_sim.agents.strategist import create_strategist_agent
        
        # Create agent with failing LLM
        strategist = create_strategist_agent()
        
        # Mock LLM to always fail
        with patch('openai.AsyncOpenAI') as mock_openai:
            mock_client = AsyncMock()
            mock_openai.return_value = mock_client
            mock_client.chat.completions.create = AsyncMock(
                side_effect=Exception("LLM service unavailable")
            )
            
            config = {
                "circuit_breaker": {
                    "failure_threshold": 1,  # Fail fast for testing
                    "recovery_timeout": 30.0,
                    "timeout": 10.0
                }
            }
            
            engine = AsyncDecisionEngine([strategist], config)
            await engine.initialize()
            
            perception = PerceptionData(
                agent_id="strategist",
                timestamp=datetime.now(),
                spatial_data={"current_position": (5, 5)},
                environmental_state={"time_pressure": 0.8},
                nearby_agents=[],
                available_actions=["move", "examine", "analyze"],
                resources={"energy": 0.8},
                constraints={}
            )
            
            # Should still generate decision using fallback
            decisions = await engine.reason_and_decide({"strategist": perception})
            
            assert len(decisions) == 1
            strategist_decision = decisions["strategist"]
            assert strategist_decision.chosen_action in ["move", "examine", "analyze"]
            assert strategist_decision.confidence_level < 0.6  # Lower confidence for fallback
            assert "fallback" in strategist_decision.reasoning.lower()
    
    @pytest.mark.asyncio
    async def test_performance_degradation_handling(self):
        """Test handling of performance degradation scenarios"""
        from src.escape_room_sim.hybrid.decision.async_engine import AsyncDecisionEngine
        from src.escape_room_sim.agents.mediator import create_mediator_agent
        
        mediator = create_mediator_agent()
        
        config = {
            "circuit_breaker": {
                "failure_threshold": 3,
                "recovery_timeout": 30.0,
                "timeout": 0.5  # Very short timeout to trigger failures
            },
            "performance_monitoring": True
        }
        
        engine = AsyncDecisionEngine([mediator], config)
        await engine.initialize()
        
        perception = PerceptionData(
            agent_id="mediator",
            timestamp=datetime.now(),
            spatial_data={"current_position": (3, 3)},
            environmental_state={"team_cohesion": 0.5},
            nearby_agents=["strategist", "survivor"],
            available_actions=["communicate", "coordinate", "mediate"],
            resources={"energy": 0.7},
            constraints={}
        )
        
        # Multiple calls to trigger performance issues
        response_times = []
        for _ in range(5):
            start = datetime.now()
            decisions = await engine.reason_and_decide({"mediator": perception})
            end = datetime.now()
            response_times.append((end - start).total_seconds())
            
            assert len(decisions) == 1
        
        # Performance should adapt (circuit breaker should open)
        # Later calls should be faster due to fallback
        assert response_times[-1] < response_times[0] or response_times[-1] < 0.1


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])  # -s to see output for debugging