"""
Unit tests for Multi-Agent Negotiation Protocols

Tests implement TDD methodology for multi-agent negotiation, coordination,
and conflict resolution in the escape room scenario.
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from unittest.mock import Mock, MagicMock, patch, AsyncMock
from dataclasses import dataclass

from src.escape_room_sim.hybrid.core_architecture import (
    PerceptionData, DecisionData, ComponentState
)


@dataclass
class NegotiationProposal:
    """Represents a negotiation proposal between agents"""
    proposer: str
    target: str
    proposal_type: str
    resource_requirements: Dict[str, float]
    expected_outcome: str
    priority: float
    timestamp: datetime


@dataclass
class NegotiationOutcome:
    """Represents the outcome of a negotiation"""
    participants: List[str]
    agreement_reached: bool
    agreed_actions: Dict[str, str]
    resource_allocation: Dict[str, Dict[str, float]]
    compromise_level: float
    execution_order: List[str]
    timestamp: datetime


class TestNegotiationProtocol:
    """Test multi-agent negotiation protocol implementation"""
    
    @pytest.fixture
    def sample_agents_perceptions(self):
        """Create sample perceptions for negotiation testing"""
        return {
            "strategist": PerceptionData(
                agent_id="strategist",
                timestamp=datetime.now(),
                spatial_data={"current_position": (2, 3), "visible_area": [(1,2), (2,2), (3,3)]},
                environmental_state={"puzzle_complexity": "high", "time_pressure": 0.7},
                nearby_agents=["mediator", "survivor"],
                available_actions=["move", "examine", "analyze", "coordinate"],
                resources={"energy": 0.8, "analytical_points": 5, "tools": []},
                constraints={"movement_range": 2, "analysis_time": 3.0}
            ),
            "mediator": PerceptionData(
                agent_id="mediator",
                timestamp=datetime.now(),
                spatial_data={"current_position": (3, 3), "visible_area": [(2,3), (3,2), (4,3)]},
                environmental_state={"team_cohesion": 0.6, "communication_clarity": 0.8},
                nearby_agents=["strategist", "survivor"],
                available_actions=["move", "communicate", "coordinate", "mediate"],
                resources={"energy": 0.7, "social_influence": 4, "tools": ["radio"]},
                constraints={"movement_range": 2, "coordination_range": 5}
            ),
            "survivor": PerceptionData(
                agent_id="survivor",
                timestamp=datetime.now(),
                spatial_data={"current_position": (1, 2), "visible_area": [(0,2), (1,1), (2,2)]},
                environmental_state={"threat_level": 0.5, "resource_scarcity": 0.8},
                nearby_agents=["strategist", "mediator"],
                available_actions=["move", "use_tool", "search", "survive"],
                resources={"energy": 0.9, "survival_points": 6, "tools": ["flashlight", "rope"]},
                constraints={"movement_range": 3, "tool_durability": 0.7}
            )
        }
    
    @pytest.fixture
    def negotiation_engine(self):
        """Create negotiation engine for testing"""
        from src.escape_room_sim.hybrid.decision.negotiation_engine import MultiAgentNegotiationEngine
        
        config = {
            "negotiation_timeout": 5.0,
            "max_negotiation_rounds": 3,
            "consensus_threshold": 0.7,
            "conflict_resolution": "majority_vote",
            "priority_weights": {
                "strategist": {"analysis": 0.9, "coordination": 0.7, "survival": 0.5},
                "mediator": {"analysis": 0.6, "coordination": 0.9, "survival": 0.7},
                "survivor": {"analysis": 0.4, "coordination": 0.6, "survival": 0.9}
            }
        }
        
        return MultiAgentNegotiationEngine(config)
    
    @pytest.mark.asyncio
    async def test_negotiation_engine_initialization(self, negotiation_engine):
        """Test negotiation engine initialization"""
        assert negotiation_engine.negotiation_timeout == 5.0
        assert negotiation_engine.max_rounds == 3
        assert negotiation_engine.consensus_threshold == 0.7
        assert "strategist" in negotiation_engine.priority_weights
    
    @pytest.mark.asyncio
    async def test_proposal_generation(self, negotiation_engine, sample_agents_perceptions):
        """Test generation of negotiation proposals from agent perceptions"""
        proposals = await negotiation_engine.generate_proposals(sample_agents_perceptions)
        
        assert len(proposals) > 0
        assert all(isinstance(p, NegotiationProposal) for p in proposals)
        
        # Should have proposals from each agent
        proposers = {p.proposer for p in proposals}
        assert "strategist" in proposers
        assert "mediator" in proposers
        assert "survivor" in proposers
        
        # Verify proposal structure
        for proposal in proposals:
            assert proposal.proposer in ["strategist", "mediator", "survivor"]
            assert proposal.priority >= 0.0 and proposal.priority <= 1.0
            assert proposal.proposal_type in ["coordination", "resource_sharing", "task_allocation", "conflict_resolution"]
    
    @pytest.mark.asyncio
    async def test_conflict_detection(self, negotiation_engine, sample_agents_perceptions):
        """Test detection of conflicts between agent intentions"""
        # Create conflicting proposals
        proposals = [
            NegotiationProposal(
                proposer="strategist",
                target="puzzle_area",
                proposal_type="task_allocation",
                resource_requirements={"analytical_points": 5, "time": 10.0},
                expected_outcome="solve_complex_puzzle",
                priority=0.9,
                timestamp=datetime.now()
            ),
            NegotiationProposal(
                proposer="survivor",
                target="puzzle_area",
                proposal_type="resource_sharing",
                resource_requirements={"energy": 0.3, "time": 5.0},
                expected_outcome="quick_resource_grab",
                priority=0.8,
                timestamp=datetime.now()
            )
        ]
        
        conflicts = negotiation_engine.detect_conflicts(proposals)
        
        assert len(conflicts) > 0
        # Should detect resource conflict (both want puzzle_area)
        conflict = conflicts[0]
        assert "puzzle_area" in conflict["resource"]
        assert len(conflict["agents"]) == 2
    
    @pytest.mark.asyncio
    async def test_negotiation_round_execution(self, negotiation_engine, sample_agents_perceptions):
        """Test execution of a single negotiation round"""
        proposals = [
            NegotiationProposal(
                proposer="strategist",
                target="team",
                proposal_type="coordination",
                resource_requirements={"time": 5.0},
                expected_outcome="coordinated_analysis",
                priority=0.8,
                timestamp=datetime.now()
            ),
            NegotiationProposal(
                proposer="mediator",
                target="team",
                proposal_type="coordination",
                resource_requirements={"social_influence": 2},
                expected_outcome="team_alignment",
                priority=0.9,
                timestamp=datetime.now()
            )
        ]
        
        # Mock LLM responses for negotiation
        with patch('src.escape_room_sim.hybrid.llm.client.OptimizedLLMClient') as mock_llm:
            mock_client = Mock()
            mock_client.generate_decisions_batch = AsyncMock(return_value={
                "strategist": "I agree to coordinate analysis efforts with the team",
                "mediator": "I support this coordination and will facilitate communication",
                "survivor": "I'm willing to participate if it helps our escape"
            })
            mock_llm.return_value = mock_client
            
            round_result = await negotiation_engine.execute_negotiation_round(
                proposals, sample_agents_perceptions
            )
            
            assert round_result["round_number"] == 1
            assert "responses" in round_result
            assert len(round_result["responses"]) == 3
            assert round_result["consensus_reached"] is True or False
    
    @pytest.mark.asyncio
    async def test_consensus_building(self, negotiation_engine):
        """Test consensus building from agent responses"""
        agent_responses = {
            "strategist": "I strongly support coordinated analysis - it's our best strategy",
            "mediator": "I agree with coordination and will help facilitate it",
            "survivor": "I'm willing to coordinate if it helps us escape faster"
        }
        
        consensus_score = negotiation_engine.calculate_consensus(agent_responses)
        
        assert 0.0 <= consensus_score <= 1.0
        # All positive responses should yield high consensus
        assert consensus_score > 0.7
    
    @pytest.mark.asyncio
    async def test_conflict_resolution_majority_vote(self, negotiation_engine):
        """Test conflict resolution using majority vote"""
        conflicting_proposals = [
            NegotiationProposal(
                proposer="strategist",
                target="door",
                proposal_type="task_allocation",
                resource_requirements={"time": 10.0},
                expected_outcome="careful_analysis",
                priority=0.8,
                timestamp=datetime.now()
            ),
            NegotiationProposal(
                proposer="survivor", 
                target="door",
                proposal_type="task_allocation",
                resource_requirements={"energy": 0.5},
                expected_outcome="force_door",
                priority=0.9,
                timestamp=datetime.now()
            )
        ]
        
        # Mock agent votes
        votes = {
            "strategist": "careful_analysis",
            "mediator": "careful_analysis",  # Majority for analysis
            "survivor": "force_door"
        }
        
        resolution = negotiation_engine.resolve_conflict_majority_vote(conflicting_proposals, votes)
        
        assert resolution["winner"] == "careful_analysis"
        assert resolution["vote_count"]["careful_analysis"] == 2
        assert resolution["vote_count"]["force_door"] == 1
    
    @pytest.mark.asyncio
    async def test_resource_allocation_negotiation(self, negotiation_engine, sample_agents_perceptions):
        """Test negotiation for resource allocation between agents"""
        # Create resource allocation scenario
        available_resources = {
            "energy": 2.0,
            "tools": ["master_key", "crowbar"],
            "time": 15.0
        }
        
        resource_requests = {
            "strategist": {"energy": 0.5, "time": 10.0},
            "mediator": {"energy": 0.3, "time": 8.0},
            "survivor": {"energy": 1.0, "tools": ["master_key"], "time": 5.0}
        }
        
        allocation = await negotiation_engine.negotiate_resource_allocation(
            available_resources, resource_requests, sample_agents_perceptions
        )
        
        assert "strategist" in allocation
        assert "mediator" in allocation
        assert "survivor" in allocation
        
        # Verify allocations don't exceed available resources
        total_energy = sum(allocation[agent].get("energy", 0) for agent in allocation)
        assert total_energy <= available_resources["energy"]
        
        total_time = sum(allocation[agent].get("time", 0) for agent in allocation)
        assert total_time <= available_resources["time"]
    
    @pytest.mark.asyncio
    async def test_negotiation_timeout_handling(self, negotiation_engine, sample_agents_perceptions):
        """Test handling of negotiation timeouts"""
        # Set very short timeout
        negotiation_engine.negotiation_timeout = 0.1
        
        # Mock slow LLM responses
        with patch('src.escape_room_sim.hybrid.llm.client.OptimizedLLMClient') as mock_llm:
            mock_client = Mock()
            mock_client.generate_decisions_batch = AsyncMock(
                side_effect=lambda *args, **kwargs: asyncio.sleep(1.0)  # Takes longer than timeout
            )
            mock_llm.return_value = mock_client
            
            proposals = [
                NegotiationProposal(
                    proposer="strategist",
                    target="team",
                    proposal_type="coordination",
                    resource_requirements={},
                    expected_outcome="timeout_test",
                    priority=0.5,
                    timestamp=datetime.now()
                )
            ]
            
            outcome = await negotiation_engine.conduct_negotiation(proposals, sample_agents_perceptions)
            
            # Should handle timeout gracefully
            assert outcome.agreement_reached is False
            assert "timeout" in outcome.agreed_actions.get("status", "").lower()


class TestNegotiationOutcomes:
    """Test negotiation outcome generation and validation"""
    
    @pytest.fixture
    def successful_negotiation_data(self):
        """Create data for successful negotiation"""
        return {
            "participants": ["strategist", "mediator", "survivor"],
            "proposals": [
                NegotiationProposal(
                    proposer="strategist",
                    target="team",
                    proposal_type="coordination",
                    resource_requirements={"time": 5.0},
                    expected_outcome="systematic_puzzle_solving",
                    priority=0.8,
                    timestamp=datetime.now()
                )
            ],
            "consensus_score": 0.85,
            "resource_allocation": {
                "strategist": {"energy": 0.3, "time": 8.0},
                "mediator": {"energy": 0.2, "time": 5.0},
                "survivor": {"energy": 0.4, "time": 2.0}
            }
        }
    
    def test_negotiation_outcome_creation(self, successful_negotiation_data):
        """Test creation of negotiation outcomes"""
        from src.escape_room_sim.hybrid.decision.negotiation_engine import create_negotiation_outcome
        
        outcome = create_negotiation_outcome(successful_negotiation_data)
        
        assert isinstance(outcome, NegotiationOutcome)
        assert outcome.agreement_reached is True
        assert len(outcome.participants) == 3
        assert outcome.compromise_level < 1.0  # Some compromise involved
        assert len(outcome.execution_order) > 0
    
    def test_execution_order_optimization(self, successful_negotiation_data):
        """Test optimization of execution order based on dependencies"""
        from src.escape_room_sim.hybrid.decision.negotiation_engine import optimize_execution_order
        
        agreed_actions = {
            "strategist": "analyze_puzzle",
            "mediator": "coordinate_team",  # Should be first
            "survivor": "gather_tools"  # Can be parallel with analysis
        }
        
        dependencies = {
            "analyze_puzzle": [],
            "coordinate_team": [],
            "gather_tools": []
        }
        
        execution_order = optimize_execution_order(agreed_actions, dependencies)
        
        assert len(execution_order) == 3
        assert "mediator" in execution_order  # Coordination should be included
        
        # Verify no circular dependencies
        assert len(set(execution_order)) == len(execution_order)
    
    def test_compromise_level_calculation(self):
        """Test calculation of compromise level in negotiations"""
        from src.escape_room_sim.hybrid.decision.negotiation_engine import calculate_compromise_level
        
        initial_proposals = {
            "strategist": {"priority": 0.9, "resources": {"time": 10.0}},
            "mediator": {"priority": 0.8, "resources": {"time": 8.0}},
            "survivor": {"priority": 0.7, "resources": {"time": 6.0}}
        }
        
        final_agreement = {
            "strategist": {"priority": 0.7, "resources": {"time": 7.0}},
            "mediator": {"priority": 0.8, "resources": {"time": 8.0}},  # No compromise
            "survivor": {"priority": 0.6, "resources": {"time": 5.0}}
        }
        
        compromise_level = calculate_compromise_level(initial_proposals, final_agreement)
        
        assert 0.0 <= compromise_level <= 1.0
        assert compromise_level > 0.0  # Some compromise occurred


class TestCoordinationProtocols:
    """Test coordination protocols for synchronized actions"""
    
    @pytest.fixture
    def coordination_scenario(self):
        """Create coordination scenario"""
        return {
            "coordinated_actions": {
                "strategist": {"action": "analyze_puzzle", "duration": 5.0, "dependencies": []},
                "mediator": {"action": "monitor_progress", "duration": 5.0, "dependencies": ["analyze_puzzle"]},
                "survivor": {"action": "prepare_tools", "duration": 3.0, "dependencies": []}
            },
            "synchronization_points": [
                {"time": 3.0, "agents": ["strategist", "survivor"], "checkpoint": "initial_assessment"},
                {"time": 5.0, "agents": ["strategist", "mediator", "survivor"], "checkpoint": "action_complete"}
            ]
        }
    
    @pytest.mark.asyncio
    async def test_coordination_protocol_execution(self, coordination_scenario):
        """Test execution of coordination protocol"""
        from src.escape_room_sim.hybrid.decision.coordination import CoordinationProtocol
        
        protocol = CoordinationProtocol(coordination_scenario)
        
        execution_plan = await protocol.create_execution_plan()
        
        assert "timeline" in execution_plan
        assert "checkpoints" in execution_plan
        assert "agent_assignments" in execution_plan
        
        # Verify timeline makes sense
        timeline = execution_plan["timeline"]
        assert len(timeline) > 0
        
        # Verify checkpoints are scheduled
        checkpoints = execution_plan["checkpoints"]
        assert len(checkpoints) == 2
    
    @pytest.mark.asyncio
    async def test_synchronization_point_validation(self, coordination_scenario):
        """Test validation of synchronization points"""
        from src.escape_room_sim.hybrid.decision.coordination import validate_synchronization_points
        
        is_valid = validate_synchronization_points(
            coordination_scenario["coordinated_actions"],
            coordination_scenario["synchronization_points"]
        )
        
        assert is_valid is True
        
        # Test invalid synchronization (impossible timing)
        invalid_sync = [
            {"time": 1.0, "agents": ["strategist"], "checkpoint": "too_early"}
        ]
        
        is_invalid = validate_synchronization_points(
            coordination_scenario["coordinated_actions"],
            invalid_sync
        )
        
        assert is_invalid is False
    
    @pytest.mark.asyncio
    async def test_coordination_failure_recovery(self, coordination_scenario):
        """Test recovery from coordination failures"""
        from src.escape_room_sim.hybrid.decision.coordination import CoordinationProtocol
        
        protocol = CoordinationProtocol(coordination_scenario)
        
        # Simulate agent failure during coordination
        failure_scenario = {
            "failed_agent": "strategist",
            "failure_time": 2.5,
            "failure_reason": "resource_exhausted"
        }
        
        recovery_plan = await protocol.handle_coordination_failure(failure_scenario)
        
        assert "replacement_actions" in recovery_plan
        assert "timeline_adjustment" in recovery_plan
        assert "affected_agents" in recovery_plan
        
        # Should reassign strategist's tasks or modify plan
        assert len(recovery_plan["replacement_actions"]) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])