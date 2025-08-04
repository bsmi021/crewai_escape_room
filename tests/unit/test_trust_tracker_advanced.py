"""
Advanced unit tests for TrustTracker class.
Tests for advanced edge cases and functionality for competitive scenarios.
"""
import pytest
from datetime import datetime, timedelta
from src.escape_room_sim.competitive.trust_tracker import TrustTracker
from src.escape_room_sim.competitive.models import TrustAction


class TestTrustTrackerAdvancedEdgeCases:
    """Test advanced edge cases and functionality for TrustTracker."""
    
    def test_trust_tracker_handles_rapid_trust_changes(self):
        """Test TrustTracker with rapid succession of trust changes."""
        tracker = TrustTracker(["agent1", "agent2"])
        
        # Rapid alternating trust changes
        actions = [
            TrustAction("cooperation", 0.3),
            TrustAction("betrayal", -0.5),
            TrustAction("cooperation", 0.4), 
            TrustAction("betrayal", -0.2),
            TrustAction("cooperation", 0.6)
        ]
        
        # Apply actions rapidly
        for action in actions:
            tracker.update_trust("agent1", "agent2", action)
        
        # Final trust should be properly calculated: 0.0 + 0.3 - 0.5 + 0.4 - 0.2 + 0.6 = 0.6
        assert tracker.get_trust_level("agent1", "agent2") == 0.6
        
        # History should track all actions
        assert tracker.get_agent_cooperation_count("agent1") == 3
        assert tracker.get_agent_betrayal_count("agent1") == 2
    
    def test_trust_tracker_asymmetric_relationships(self):
        """Test that relationships can be completely asymmetric."""
        tracker = TrustTracker(["agent1", "agent2"])
        
        # Agent1 fully trusts agent2
        max_trust = TrustAction("cooperation", 1.0)
        tracker.update_trust("agent1", "agent2", max_trust)
        
        # Agent2 completely distrusts agent1
        max_distrust = TrustAction("betrayal", -1.0)
        tracker.update_trust("agent2", "agent1", max_distrust)
        
        assert tracker.get_trust_level("agent1", "agent2") == 1.0
        assert tracker.get_trust_level("agent2", "agent1") == -1.0
        
        # Reputations should reflect asymmetry
        assert tracker.calculate_reputation("agent1") == -1.0  # agent2 distrusts agent1
        assert tracker.calculate_reputation("agent2") == 1.0   # agent1 trusts agent2
    
    def test_trust_tracker_complex_multi_agent_dynamics(self):
        """Test complex dynamics with multiple agents and varied relationships."""
        agents = ["strategist", "mediator", "survivor"]
        tracker = TrustTracker(agents)
        
        # Simulate complex scenario:
        # Strategist betrays mediator but cooperates with survivor
        # Mediator tries to mediate by cooperating with everyone
        # Survivor is selfish and betrays everyone
        
        # Strategist actions
        tracker.update_trust("strategist", "mediator", TrustAction("betrayal", -0.6))
        tracker.update_trust("strategist", "survivor", TrustAction("cooperation", 0.4))
        
        # Mediator actions (tries to build trust)
        tracker.update_trust("mediator", "strategist", TrustAction("cooperation", 0.5))
        tracker.update_trust("mediator", "survivor", TrustAction("cooperation", 0.3))
        
        # Survivor actions (betrays everyone)
        tracker.update_trust("survivor", "strategist", TrustAction("betrayal", -0.7))
        tracker.update_trust("survivor", "mediator", TrustAction("betrayal", -0.8))
        
        # Check final trust levels
        assert tracker.get_trust_level("strategist", "mediator") == -0.6
        assert tracker.get_trust_level("strategist", "survivor") == 0.4
        assert tracker.get_trust_level("mediator", "strategist") == 0.5
        assert tracker.get_trust_level("mediator", "survivor") == 0.3
        assert tracker.get_trust_level("survivor", "strategist") == -0.7
        assert tracker.get_trust_level("survivor", "mediator") == -0.8
        
        # Check reputations
        strategist_rep = tracker.calculate_reputation("strategist")
        mediator_rep = tracker.calculate_reputation("mediator")
        survivor_rep = tracker.calculate_reputation("survivor")
        
        # Strategist gets mixed reputation (cooperation from mediator, betrayal from survivor)
        assert strategist_rep == (0.5 + -0.7) / 2  # (-0.1)
        
        # Mediator gets mixed reputation (betrayal from strategist, betrayal from survivor)
        assert mediator_rep == (-0.6 + -0.8) / 2  # (-0.7)
        
        # Survivor gets mixed reputation (cooperation from strategist, cooperation from mediator)
        assert survivor_rep == (0.4 + 0.3) / 2  # (0.35)
    
    def test_trust_tracker_handles_neutral_actions(self):
        """Test handling of neutral actions that don't affect trust."""
        tracker = TrustTracker(["agent1", "agent2"])
        
        # Start with some trust
        initial_trust = TrustAction("cooperation", 0.5)
        tracker.update_trust("agent1", "agent2", initial_trust)
        
        # Apply neutral action
        neutral_action = TrustAction("neutral", 0.0)
        tracker.update_trust("agent1", "agent2", neutral_action)
        
        # Trust should remain unchanged
        assert tracker.get_trust_level("agent1", "agent2") == 0.5
        
        # Neutral actions should not be recorded in cooperation or betrayal history
        assert tracker.get_agent_cooperation_count("agent1") == 1
        assert tracker.get_agent_betrayal_count("agent1") == 0
        assert len(tracker.cooperation_history) == 1
        assert len(tracker.betrayal_history) == 0
    
    def test_trust_tracker_relationship_persistence(self):
        """Test that relationships persist correctly over time."""
        tracker = TrustTracker()
        
        # Create relationship
        action = TrustAction("cooperation", 0.3)
        tracker.update_trust("agent1", "agent2", action)
        
        # Verify relationship exists and persists
        assert ("agent1", "agent2") in tracker.trust_matrix
        relationship = tracker.trust_matrix[("agent1", "agent2")]
        
        # Check that all relationship data is properly maintained
        assert relationship.agent1 == "agent1"
        assert relationship.agent2 == "agent2"
        assert relationship.trust_level == 0.3
        assert relationship.cooperation_count == 1
        assert relationship.betrayal_count == 0
        assert isinstance(relationship.last_interaction, datetime)
    
    def test_trust_tracker_handles_string_edge_cases(self):
        """Test edge cases with agent name strings."""
        tracker = TrustTracker()
        
        # Test with unusual but valid agent names
        unusual_names = ["agent_1", "agent-2", "Agent With Spaces", "Agent123"]
        for name in unusual_names:
            action = TrustAction("cooperation", 0.1)
            tracker.update_trust(name, "target", action)
        
        # All should work without errors
        for name in unusual_names:
            assert tracker.get_trust_level(name, "target") == 0.1
    
    def test_trust_tracker_comprehensive_error_handling(self):
        """Test comprehensive error handling for all methods."""
        tracker = TrustTracker()
        
        # Test with whitespace-only strings
        with pytest.raises(ValueError, match="Actor cannot be empty"):
            tracker.update_trust("   ", "agent2", TrustAction("cooperation", 0.1))
        
        with pytest.raises(ValueError, match="Target cannot be empty"):
            tracker.update_trust("agent1", "   ", TrustAction("cooperation", 0.1))
        
        with pytest.raises(ValueError, match="Agent1 cannot be empty"):
            tracker.get_trust_level("   ", "agent2")
        
        with pytest.raises(ValueError, match="Agent2 cannot be empty"):
            tracker.get_trust_level("agent1", "   ")
        
        with pytest.raises(ValueError, match="Agent cannot be empty"):
            tracker.calculate_reputation("   ")
    
    def test_trust_tracker_memory_efficiency(self):
        """Test that TrustTracker manages memory efficiently."""
        tracker = TrustTracker()
        
        # Create many relationships and actions
        num_agents = 50
        agents = [f"agent{i}" for i in range(num_agents)]
        
        # Create relationships between all agents
        for i, agent1 in enumerate(agents):
            for j, agent2 in enumerate(agents):
                if i != j:
                    action = TrustAction("cooperation", 0.1)
                    tracker.update_trust(agent1, agent2, action)
        
        # Verify all relationships exist
        expected_relationships = num_agents * (num_agents - 1)
        assert len(tracker.trust_matrix) == expected_relationships
        
        # Verify history is properly managed
        expected_history_entries = expected_relationships  # One cooperation per relationship
        assert len(tracker.cooperation_history) == expected_history_entries
        assert len(tracker.betrayal_history) == 0
        
        # Test that we can still efficiently query any relationship
        assert tracker.get_trust_level("agent0", "agent49") == 0.1
        assert tracker.get_trust_level("agent25", "agent10") == 0.1
    
    def test_trust_tracker_floating_point_precision(self):
        """Test handling of floating point precision issues."""
        tracker = TrustTracker(["agent1", "agent2"])
        
        # Actions that might cause precision issues
        precise_actions = [
            TrustAction("cooperation", 0.1),
            TrustAction("cooperation", 0.2),
            TrustAction("cooperation", 0.3),
            TrustAction("betrayal", -0.4),
            TrustAction("cooperation", 0.15),
            TrustAction("betrayal", -0.25)
        ]
        
        for action in precise_actions:
            tracker.update_trust("agent1", "agent2", action)
        
        # Final value: 0.0 + 0.1 + 0.2 + 0.3 - 0.4 + 0.15 - 0.25 = 0.1
        trust_level = tracker.get_trust_level("agent1", "agent2")
        
        # Should handle precision correctly
        assert abs(trust_level - 0.1) < 1e-10


class TestTrustTrackerAdvancedFeatures:
    """Test advanced features and methods for TrustTracker."""
    
    def test_get_all_relationships_for_agent(self):
        """Test getting all relationships for a specific agent."""
        tracker = TrustTracker(["agent1", "agent2", "agent3"])
        
        # Create some relationships
        tracker.update_trust("agent1", "agent2", TrustAction("cooperation", 0.5))
        tracker.update_trust("agent1", "agent3", TrustAction("betrayal", -0.3))
        tracker.update_trust("agent2", "agent1", TrustAction("cooperation", 0.2))
        
        # Get outgoing relationships for agent1
        outgoing_relationships = {}
        for (actor, target), relationship in tracker.trust_matrix.items():
            if actor == "agent1":
                outgoing_relationships[target] = relationship.trust_level
        
        assert outgoing_relationships["agent2"] == 0.5
        assert outgoing_relationships["agent3"] == -0.3
        
        # Get incoming relationships for agent1
        incoming_relationships = {}
        for (actor, target), relationship in tracker.trust_matrix.items():
            if target == "agent1":
                incoming_relationships[actor] = relationship.trust_level
        
        assert incoming_relationships["agent2"] == 0.2
        assert "agent3" not in incoming_relationships or incoming_relationships["agent3"] == 0.0
    
    def test_trust_tracker_history_time_tracking(self):
        """Test that history properly tracks timestamps."""
        tracker = TrustTracker()
        
        start_time = datetime.now()
        
        # Perform actions with small delays
        tracker.update_trust("agent1", "agent2", TrustAction("cooperation", 0.3))
        
        # Check that timestamp is reasonable
        cooperation_record = tracker.cooperation_history[0]
        timestamp = cooperation_record["timestamp"]
        
        assert isinstance(timestamp, datetime)
        assert start_time <= timestamp <= datetime.now()
    
    def test_trust_tracker_complex_reputation_scenarios(self):
        """Test reputation calculation in complex scenarios."""
        tracker = TrustTracker()
        
        # Scenario: Agent1 is trusted by some, distrusted by others
        trust_values = [0.8, -0.6, 0.4, -0.2, 0.9, -0.7]
        
        for i, trust_value in enumerate(trust_values):
            action_type = "cooperation" if trust_value > 0 else "betrayal"
            action = TrustAction(action_type, trust_value)
            tracker.update_trust(f"agent{i+2}", "agent1", action)
        
        # Reputation should be the average of all incoming trust
        expected_reputation = sum(trust_values) / len(trust_values)
        actual_reputation = tracker.calculate_reputation("agent1")
        
        assert abs(actual_reputation - expected_reputation) < 1e-10