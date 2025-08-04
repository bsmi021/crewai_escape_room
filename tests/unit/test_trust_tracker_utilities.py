"""
Unit tests for TrustTracker utility methods.
Tests for advanced utility functions for competitive scenarios.
"""
import pytest
from src.escape_room_sim.competitive.trust_tracker import TrustTracker
from src.escape_room_sim.competitive.models import TrustAction


class TestTrustTrackerUtilityMethods:
    """Test utility methods for TrustTracker."""
    
    def test_get_most_trusted_agent(self):
        """Test getting the most trusted agent."""
        tracker = TrustTracker(["agent1", "agent2", "agent3", "agent4"])
        
        # Agent1 trusts different agents with different levels
        tracker.update_trust("agent1", "agent2", TrustAction("cooperation", 0.3))
        tracker.update_trust("agent1", "agent3", TrustAction("cooperation", 0.8))
        tracker.update_trust("agent1", "agent4", TrustAction("betrayal", -0.2))
        
        most_trusted = tracker.get_most_trusted_agent("agent1")
        assert most_trusted == "agent3"
    
    def test_get_most_trusted_agent_no_relationships(self):
        """Test getting most trusted agent when no relationships exist."""
        tracker = TrustTracker()
        
        most_trusted = tracker.get_most_trusted_agent("agent1")
        assert most_trusted is None
    
    def test_get_least_trusted_agent(self):
        """Test getting the least trusted agent."""
        tracker = TrustTracker(["agent1", "agent2", "agent3", "agent4"])
        
        # Agent1 trusts different agents with different levels
        tracker.update_trust("agent1", "agent2", TrustAction("cooperation", 0.3))
        tracker.update_trust("agent1", "agent3", TrustAction("cooperation", 0.8))
        tracker.update_trust("agent1", "agent4", TrustAction("betrayal", -0.6))
        
        least_trusted = tracker.get_least_trusted_agent("agent1")
        assert least_trusted == "agent4"
    
    def test_get_least_trusted_agent_no_relationships(self):
        """Test getting least trusted agent when no relationships exist."""
        tracker = TrustTracker()
        
        least_trusted = tracker.get_least_trusted_agent("agent1")
        assert least_trusted is None
    
    def test_get_agent_trustworthiness_score(self):
        """Test calculating agent trustworthiness score."""
        tracker = TrustTracker()
        
        # Agent with more cooperation than betrayal
        tracker.update_trust("agent1", "agent2", TrustAction("cooperation", 0.3))
        tracker.update_trust("agent1", "agent3", TrustAction("cooperation", 0.4))
        tracker.update_trust("agent1", "agent4", TrustAction("betrayal", -0.5))
        
        trustworthiness = tracker.get_agent_trustworthiness_score("agent1")
        assert trustworthiness == 2/3  # 2 cooperations out of 3 total actions
    
    def test_get_agent_trustworthiness_score_no_actions(self):
        """Test trustworthiness score for agent with no actions."""
        tracker = TrustTracker()
        
        trustworthiness = tracker.get_agent_trustworthiness_score("agent1")
        assert trustworthiness == 0.0
    
    def test_get_agent_trustworthiness_score_all_cooperation(self):
        """Test trustworthiness score for completely cooperative agent."""
        tracker = TrustTracker()
        
        tracker.update_trust("agent1", "agent2", TrustAction("cooperation", 0.3))
        tracker.update_trust("agent1", "agent3", TrustAction("cooperation", 0.4))
        
        trustworthiness = tracker.get_agent_trustworthiness_score("agent1")
        assert trustworthiness == 1.0
    
    def test_get_agent_trustworthiness_score_all_betrayal(self):
        """Test trustworthiness score for completely untrustworthy agent."""
        tracker = TrustTracker()
        
        tracker.update_trust("agent1", "agent2", TrustAction("betrayal", -0.3))
        tracker.update_trust("agent1", "agent3", TrustAction("betrayal", -0.4))
        
        trustworthiness = tracker.get_agent_trustworthiness_score("agent1")
        assert trustworthiness == 0.0
    
    def test_get_mutual_trust_level(self):
        """Test calculating mutual trust level between two agents."""
        tracker = TrustTracker()
        
        # Set up asymmetric trust
        tracker.update_trust("agent1", "agent2", TrustAction("cooperation", 0.6))
        tracker.update_trust("agent2", "agent1", TrustAction("cooperation", 0.4))
        
        mutual_trust = tracker.get_mutual_trust_level("agent1", "agent2")
        assert mutual_trust == 0.5  # (0.6 + 0.4) / 2
    
    def test_get_mutual_trust_level_one_way(self):
        """Test mutual trust when only one direction has trust."""
        tracker = TrustTracker()
        
        tracker.update_trust("agent1", "agent2", TrustAction("cooperation", 0.8))
        # agent2 -> agent1 remains at 0.0 (default)
        
        mutual_trust = tracker.get_mutual_trust_level("agent1", "agent2")
        assert mutual_trust == 0.4  # (0.8 + 0.0) / 2
    
    def test_get_all_agents(self):
        """Test getting all agents from trust matrix."""
        tracker = TrustTracker()
        
        # Create relationships
        tracker.update_trust("alice", "bob", TrustAction("cooperation", 0.3))
        tracker.update_trust("charlie", "alice", TrustAction("betrayal", -0.2))
        tracker.update_trust("bob", "charlie", TrustAction("cooperation", 0.5))
        
        agents = tracker.get_all_agents()
        assert set(agents) == {"alice", "bob", "charlie"}
        assert agents == sorted(agents)  # Should be sorted
    
    def test_get_all_agents_empty_tracker(self):
        """Test getting all agents from empty tracker."""
        tracker = TrustTracker()
        
        agents = tracker.get_all_agents()
        assert agents == []
    
    def test_get_trust_summary(self):
        """Test getting comprehensive trust summary for an agent."""
        tracker = TrustTracker()
        
        # Set up relationships
        tracker.update_trust("agent1", "agent2", TrustAction("cooperation", 0.6))
        tracker.update_trust("agent1", "agent3", TrustAction("betrayal", -0.3))
        tracker.update_trust("agent2", "agent1", TrustAction("cooperation", 0.4))
        tracker.update_trust("agent3", "agent1", TrustAction("betrayal", -0.7))
        
        summary = tracker.get_trust_summary("agent1")
        
        # Check outgoing trust
        assert summary["trusts_agent2"] == 0.6
        assert summary["trusts_agent3"] == -0.3
        
        # Check reputation (incoming trust from others)
        expected_reputation = (0.4 + -0.7) / 2  # -0.15
        assert abs(summary["reputation"] - expected_reputation) < 1e-10
        
        # Check trustworthiness (1 cooperation, 1 betrayal = 0.5)
        assert summary["trustworthiness"] == 0.5
    
    def test_utility_methods_validate_inputs(self):
        """Test that utility methods properly validate inputs."""
        tracker = TrustTracker()
        
        # Test empty agent names
        with pytest.raises(ValueError, match="Agent cannot be empty"):
            tracker.get_most_trusted_agent("")
        
        with pytest.raises(ValueError, match="Agent cannot be empty"):
            tracker.get_least_trusted_agent("   ")
        
        with pytest.raises(ValueError, match="Agent cannot be empty"):
            tracker.get_agent_trustworthiness_score("")
        
        with pytest.raises(ValueError, match="Agent cannot be empty"):
            tracker.get_trust_summary("   ")
        
        # Test mutual trust validation (should use existing validation)
        with pytest.raises(ValueError, match="Agent1 cannot be empty"):
            tracker.get_mutual_trust_level("", "agent2")
    
    def test_complex_utility_scenario(self):
        """Test utility methods in a complex multi-agent scenario."""
        agents = ["strategist", "mediator", "survivor"]
        tracker = TrustTracker(agents)
        
        # Complex scenario: 
        # Strategist cooperates with mediator, betrays survivor
        # Mediator cooperates with everyone
        # Survivor betrays everyone
        
        tracker.update_trust("strategist", "mediator", TrustAction("cooperation", 0.7))
        tracker.update_trust("strategist", "survivor", TrustAction("betrayal", -0.5))
        
        tracker.update_trust("mediator", "strategist", TrustAction("cooperation", 0.6))
        tracker.update_trust("mediator", "survivor", TrustAction("cooperation", 0.4))
        
        tracker.update_trust("survivor", "strategist", TrustAction("betrayal", -0.8))
        tracker.update_trust("survivor", "mediator", TrustAction("betrayal", -0.6))
        
        # Test various utility methods
        assert tracker.get_most_trusted_agent("strategist") == "mediator"
        assert tracker.get_least_trusted_agent("strategist") == "survivor"
        
        # Trustworthiness scores
        assert tracker.get_agent_trustworthiness_score("strategist") == 0.5  # 1 coop, 1 betrayal
        assert tracker.get_agent_trustworthiness_score("mediator") == 1.0   # 2 coop, 0 betrayal
        assert tracker.get_agent_trustworthiness_score("survivor") == 0.0   # 0 coop, 2 betrayal
        
        # Mutual trust
        mutual_strat_med = tracker.get_mutual_trust_level("strategist", "mediator")
        assert mutual_strat_med == (0.7 + 0.6) / 2  # 0.65
        
        # Get all agents
        all_agents = tracker.get_all_agents()
        assert set(all_agents) == {"strategist", "mediator", "survivor"}
        
        # Trust summary for mediator
        mediator_summary = tracker.get_trust_summary("mediator")
        assert mediator_summary["trusts_strategist"] == 0.6
        assert mediator_summary["trusts_survivor"] == 0.4
        assert mediator_summary["trustworthiness"] == 1.0