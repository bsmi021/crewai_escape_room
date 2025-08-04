"""
Integration tests for TrustTracker with other competitive components.
Tests the interaction between TrustTracker and other competitive mechanics.
"""
import pytest
from src.escape_room_sim.competitive.trust_tracker import TrustTracker
from src.escape_room_sim.competitive.models import TrustAction, CompetitiveScenario
from src.escape_room_sim.competitive.scenario_generator import ScenarioGenerator


class TestTrustTrackerIntegration:
    """Test TrustTracker integration with competitive scenario components."""
    
    def test_trust_tracker_with_scenario_agents(self):
        """Test TrustTracker initialization with agents from a competitive scenario."""
        # Generate a scenario
        generator = ScenarioGenerator(seed=42)
        scenario = generator.generate_scenario()
        
        # Initialize TrustTracker with typical agent names
        agents = ["strategist", "mediator", "survivor"]
        tracker = TrustTracker(agents)
        
        # Verify all relationships are initialized
        assert len(tracker.trust_matrix) == 6  # 3 agents * 2 directions each
        
        # Test trust updates between scenario agents
        cooperation = TrustAction("cooperation", 0.3)
        betrayal = TrustAction("betrayal", -0.4)
        
        tracker.update_trust("strategist", "mediator", cooperation)
        tracker.update_trust("mediator", "survivor", betrayal)
        
        assert tracker.get_trust_level("strategist", "mediator") == 0.3
        assert tracker.get_trust_level("mediator", "survivor") == -0.4
    
    def test_trust_tracker_reputation_affects_scenario_dynamics(self):
        """Test how trust reputation might affect competitive scenario dynamics."""
        agents = ["agent1", "agent2", "agent3"]
        tracker = TrustTracker(agents)
        
        # Simulate a series of interactions
        # Agent1 cooperates with everyone
        cooperation = TrustAction("cooperation", 0.4)
        tracker.update_trust("agent2", "agent1", cooperation)
        tracker.update_trust("agent3", "agent1", cooperation)
        
        # Agent2 betrays everyone
        betrayal = TrustAction("betrayal", -0.6)
        tracker.update_trust("agent1", "agent2", betrayal)
        tracker.update_trust("agent3", "agent2", betrayal)
        
        # Check reputations
        agent1_reputation = tracker.calculate_reputation("agent1")
        agent2_reputation = tracker.calculate_reputation("agent2")
        agent3_reputation = tracker.calculate_reputation("agent3")
        
        assert agent1_reputation == 0.4  # Trusted by others
        assert agent2_reputation == -0.6  # Distrusted by others
        assert agent3_reputation == 0.0  # Neutral
        
        # Verify this could influence future decisions
        assert agent1_reputation > agent2_reputation
        assert agent1_reputation > agent3_reputation
    
    def test_trust_tracker_history_tracking_comprehensive(self):
        """Test comprehensive history tracking across multiple interactions."""
        tracker = TrustTracker()
        
        # Simulate complex interaction patterns
        interactions = [
            ("agent1", "agent2", TrustAction("cooperation", 0.2)),
            ("agent2", "agent1", TrustAction("cooperation", 0.3)),
            ("agent1", "agent3", TrustAction("betrayal", -0.4)),
            ("agent3", "agent1", TrustAction("betrayal", -0.5)),
            ("agent2", "agent3", TrustAction("cooperation", 0.1)),
            ("agent3", "agent2", TrustAction("betrayal", -0.2)),
        ]
        
        for actor, target, action in interactions:
            tracker.update_trust(actor, target, action)
        
        # Verify history counts
        assert len(tracker.cooperation_history) == 3
        assert len(tracker.betrayal_history) == 3
        
        # Verify agent-specific counts
        assert tracker.get_agent_cooperation_count("agent1") == 1
        assert tracker.get_agent_cooperation_count("agent2") == 2
        assert tracker.get_agent_betrayal_count("agent1") == 1
        assert tracker.get_agent_betrayal_count("agent3") == 2
    
    def test_trust_tracker_with_moral_dilemma_outcomes(self):
        """Test TrustTracker integration with moral dilemma consequences."""
        tracker = TrustTracker(["agent1", "agent2", "agent3"])
        
        # Simulate moral dilemma where agent1 makes selfish choice
        # This would typically come from MoralDilemmaEngine
        selfish_consequences = {
            "agent2": -0.5,  # Trust impact on agent2
            "agent3": -0.3   # Trust impact on agent3
        }
        
        # Apply consequences through TrustTracker
        for target, impact in selfish_consequences.items():
            action = TrustAction("betrayal", impact)
            tracker.update_trust(target, "agent1", action)
        
        # Verify reputation impact
        agent1_reputation = tracker.calculate_reputation("agent1")
        expected_reputation = (-0.5 + -0.3) / 2  # Average of incoming trust
        assert abs(agent1_reputation - expected_reputation) < 1e-10
        
        # Verify this affects future trust calculations
        assert tracker.get_trust_level("agent2", "agent1") == -0.5
        assert tracker.get_trust_level("agent3", "agent1") == -0.3
    
    def test_trust_tracker_boundary_conditions_in_scenarios(self):
        """Test TrustTracker boundary conditions in realistic scenario contexts."""
        tracker = TrustTracker()
        
        # Test extreme trust building scenario
        extreme_cooperation = TrustAction("cooperation", 0.9)
        for _ in range(5):  # Multiple strong cooperation actions
            tracker.update_trust("agent1", "agent2", extreme_cooperation)
        
        # Should be capped at 1.0
        assert tracker.get_trust_level("agent1", "agent2") == 1.0
        
        # Test extreme trust destruction scenario
        extreme_betrayal = TrustAction("betrayal", -0.8)
        for _ in range(5):  # Multiple strong betrayal actions
            tracker.update_trust("agent3", "agent4", extreme_betrayal)
        
        # Should be capped at -1.0
        assert tracker.get_trust_level("agent3", "agent4") == -1.0
    
    def test_trust_tracker_performance_with_many_agents(self):
        """Test TrustTracker performance with scenario-realistic agent counts."""
        # Test with a reasonable number of agents for competitive scenarios
        agents = [f"agent{i}" for i in range(10)]
        tracker = TrustTracker(agents)
        
        # Verify initialization
        expected_relationships = 10 * 9  # n * (n-1) relationships
        assert len(tracker.trust_matrix) == expected_relationships
        
        # Perform many trust updates
        action = TrustAction("cooperation", 0.1)
        for i in range(10):
            for j in range(10):
                if i != j:
                    tracker.update_trust(f"agent{i}", f"agent{j}", action)
        
        # Verify all relationships updated correctly
        for i in range(10):
            for j in range(10):
                if i != j:
                    trust_level = tracker.get_trust_level(f"agent{i}", f"agent{j}")
                    assert trust_level == 0.1
        
        # Test reputation calculation performance
        for i in range(10):
            reputation = tracker.calculate_reputation(f"agent{i}")
            assert reputation == 0.1  # All incoming trust should be 0.1