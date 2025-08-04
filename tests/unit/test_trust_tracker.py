"""
Unit tests for TrustTracker class.
Tests for relationship dynamics and trust management in competitive scenarios.
"""
import pytest
from datetime import datetime, timedelta
from src.escape_room_sim.competitive.trust_tracker import TrustTracker, TrustAction
from src.escape_room_sim.competitive.models import TrustRelationship


class TestTrustTrackerInitialization:
    """Test TrustTracker initialization with empty trust matrix."""
    
    def test_trust_tracker_initializes_with_empty_matrix(self):
        """Test that TrustTracker initializes with empty trust matrix."""
        tracker = TrustTracker()
        assert tracker.trust_matrix == {}
        assert tracker.betrayal_history == []
        assert tracker.cooperation_history == []
    
    def test_trust_tracker_initializes_with_agent_list(self):
        """Test that TrustTracker can initialize with predefined agent list."""
        agents = ["agent1", "agent2", "agent3"]
        tracker = TrustTracker(agents)
        
        # Should create neutral relationships between all agent pairs
        expected_pairs = [
            ("agent1", "agent2"), ("agent1", "agent3"),
            ("agent2", "agent1"), ("agent2", "agent3"),
            ("agent3", "agent1"), ("agent3", "agent2")
        ]
        
        for pair in expected_pairs:
            assert pair in tracker.trust_matrix
            assert tracker.trust_matrix[pair].trust_level == 0.0
    
    def test_trust_tracker_validates_agent_list(self):
        """Test that TrustTracker validates agent list input."""
        with pytest.raises(ValueError, match="Agent list cannot be empty"):
            TrustTracker([])
        
        with pytest.raises(ValueError, match="Duplicate agents not allowed"):
            TrustTracker(["agent1", "agent1", "agent2"])


class TestUpdateTrustMethod:
    """Test update_trust method modifying relationships based on actions."""
    
    def test_update_trust_creates_new_relationship(self):
        """Test that update_trust creates new relationship if it doesn't exist."""
        tracker = TrustTracker()
        action = TrustAction("cooperation", 0.2)
        
        tracker.update_trust("agent1", "agent2", action)
        
        assert ("agent1", "agent2") in tracker.trust_matrix
        relationship = tracker.trust_matrix[("agent1", "agent2")]
        assert relationship.trust_level == 0.2
        assert relationship.cooperation_count == 1
        assert relationship.betrayal_count == 0
    
    def test_update_trust_modifies_existing_relationship(self):
        """Test that update_trust modifies existing relationship."""
        tracker = TrustTracker(["agent1", "agent2"])
        action = TrustAction("cooperation", 0.3)
        
        tracker.update_trust("agent1", "agent2", action)
        
        relationship = tracker.trust_matrix[("agent1", "agent2")]
        assert relationship.trust_level == 0.3
        assert relationship.cooperation_count == 1
    
    def test_update_trust_handles_betrayal_actions(self):
        """Test that update_trust properly handles betrayal actions."""
        tracker = TrustTracker(["agent1", "agent2"])
        action = TrustAction("betrayal", -0.5)
        
        tracker.update_trust("agent1", "agent2", action)
        
        relationship = tracker.trust_matrix[("agent1", "agent2")]
        assert relationship.trust_level == -0.5
        assert relationship.betrayal_count == 1
        assert relationship.cooperation_count == 0
    
    def test_update_trust_enforces_boundaries(self):
        """Test that trust levels are constrained to [-1.0, 1.0]."""
        tracker = TrustTracker(["agent1", "agent2"])
        
        # Test upper boundary
        action = TrustAction("cooperation", 1.5)
        tracker.update_trust("agent1", "agent2", action)
        assert tracker.trust_matrix[("agent1", "agent2")].trust_level == 1.0
        
        # Test lower boundary
        action = TrustAction("betrayal", -2.0)
        tracker.update_trust("agent1", "agent2", action)
        assert tracker.trust_matrix[("agent1", "agent2")].trust_level == -1.0
    
    def test_update_trust_records_history(self):
        """Test that update_trust records actions in history."""
        tracker = TrustTracker()
        cooperation_action = TrustAction("cooperation", 0.2)
        betrayal_action = TrustAction("betrayal", -0.3)
        
        tracker.update_trust("agent1", "agent2", cooperation_action)
        tracker.update_trust("agent2", "agent1", betrayal_action)
        
        assert len(tracker.cooperation_history) == 1
        assert len(tracker.betrayal_history) == 1
        assert tracker.cooperation_history[0]["actor"] == "agent1"
        assert tracker.cooperation_history[0]["target"] == "agent2"
        assert tracker.betrayal_history[0]["actor"] == "agent2"
        assert tracker.betrayal_history[0]["target"] == "agent1"
    
    def test_update_trust_validates_agents(self):
        """Test that update_trust validates agent parameters."""
        tracker = TrustTracker()
        action = TrustAction("cooperation", 0.2)
        
        with pytest.raises(ValueError, match="Actor cannot be empty"):
            tracker.update_trust("", "agent2", action)
        
        with pytest.raises(ValueError, match="Target cannot be empty"):
            tracker.update_trust("agent1", "", action)
        
        with pytest.raises(ValueError, match="Agent cannot have relationship with itself"):
            tracker.update_trust("agent1", "agent1", action)


class TestGetTrustLevelMethod:
    """Test get_trust_level method returning relationship strength."""
    
    def test_get_trust_level_returns_existing_relationship(self):
        """Test that get_trust_level returns trust level for existing relationship."""
        tracker = TrustTracker(["agent1", "agent2"])
        action = TrustAction("cooperation", 0.4)
        tracker.update_trust("agent1", "agent2", action)
        
        trust_level = tracker.get_trust_level("agent1", "agent2")
        assert trust_level == 0.4
    
    def test_get_trust_level_returns_neutral_for_new_relationship(self):
        """Test that get_trust_level returns 0.0 for non-existent relationship."""
        tracker = TrustTracker()
        
        trust_level = tracker.get_trust_level("agent1", "agent2")
        assert trust_level == 0.0
    
    def test_get_trust_level_is_directional(self):
        """Test that trust relationships are directional."""
        tracker = TrustTracker()
        action = TrustAction("cooperation", 0.3)
        tracker.update_trust("agent1", "agent2", action)
        
        # agent1 -> agent2 should have trust level 0.3
        assert tracker.get_trust_level("agent1", "agent2") == 0.3
        # agent2 -> agent1 should still be neutral
        assert tracker.get_trust_level("agent2", "agent1") == 0.0
    
    def test_get_trust_level_validates_agents(self):
        """Test that get_trust_level validates agent parameters."""
        tracker = TrustTracker()
        
        with pytest.raises(ValueError, match="Agent1 cannot be empty"):
            tracker.get_trust_level("", "agent2")
        
        with pytest.raises(ValueError, match="Agent2 cannot be empty"):
            tracker.get_trust_level("agent1", "")
        
        with pytest.raises(ValueError, match="Agent cannot query relationship with itself"):
            tracker.get_trust_level("agent1", "agent1")


class TestCalculateReputationMethod:
    """Test calculate_reputation method computing overall agent standing."""
    
    def test_calculate_reputation_with_no_relationships(self):
        """Test reputation calculation for agent with no relationships."""
        tracker = TrustTracker()
        
        reputation = tracker.calculate_reputation("agent1")
        assert reputation == 0.0
    
    def test_calculate_reputation_with_positive_relationships(self):
        """Test reputation calculation with positive trust relationships."""
        tracker = TrustTracker()
        action = TrustAction("cooperation", 0.4)
        
        tracker.update_trust("agent2", "agent1", action)
        tracker.update_trust("agent3", "agent1", action)
        
        reputation = tracker.calculate_reputation("agent1")
        assert reputation == 0.4  # Average of incoming trust
    
    def test_calculate_reputation_with_mixed_relationships(self):
        """Test reputation calculation with mixed positive and negative relationships."""
        tracker = TrustTracker()
        
        cooperation_action = TrustAction("cooperation", 0.6)
        betrayal_action = TrustAction("betrayal", -0.4)
        
        tracker.update_trust("agent2", "agent1", cooperation_action)
        tracker.update_trust("agent3", "agent1", betrayal_action)
        
        reputation = tracker.calculate_reputation("agent1")
        assert abs(reputation - 0.1) < 1e-10  # (0.6 + (-0.4)) / 2, handle floating point precision
    
    def test_calculate_reputation_ignores_outgoing_trust(self):
        """Test that reputation only considers incoming trust, not outgoing."""
        tracker = TrustTracker()
        action = TrustAction("cooperation", 0.5)
        
        # agent1 trusts agent2, but agent2 doesn't trust agent1
        tracker.update_trust("agent1", "agent2", action)
        
        # agent1's reputation should be neutral (no incoming trust)
        reputation = tracker.calculate_reputation("agent1")
        assert reputation == 0.0
    
    def test_calculate_reputation_validates_agent(self):
        """Test that calculate_reputation validates agent parameter."""
        tracker = TrustTracker()
        
        with pytest.raises(ValueError, match="Agent cannot be empty"):
            tracker.calculate_reputation("")


class TestBetrayalAndCooperationHistory:
    """Test betrayal and cooperation history tracking."""
    
    def test_betrayal_history_tracking(self):
        """Test that betrayal actions are properly tracked in history."""
        tracker = TrustTracker()
        action = TrustAction("betrayal", -0.3)
        
        tracker.update_trust("agent1", "agent2", action)
        
        assert len(tracker.betrayal_history) == 1
        betrayal_record = tracker.betrayal_history[0]
        assert betrayal_record["actor"] == "agent1"
        assert betrayal_record["target"] == "agent2"
        assert betrayal_record["impact"] == -0.3
        assert "timestamp" in betrayal_record
    
    def test_cooperation_history_tracking(self):
        """Test that cooperation actions are properly tracked in history."""
        tracker = TrustTracker()
        action = TrustAction("cooperation", 0.4)
        
        tracker.update_trust("agent1", "agent2", action)
        
        assert len(tracker.cooperation_history) == 1
        cooperation_record = tracker.cooperation_history[0]
        assert cooperation_record["actor"] == "agent1"
        assert cooperation_record["target"] == "agent2"
        assert cooperation_record["impact"] == 0.4
        assert "timestamp" in cooperation_record
    
    def test_mixed_history_tracking(self):
        """Test tracking of mixed cooperation and betrayal actions."""
        tracker = TrustTracker()
        
        cooperation_action = TrustAction("cooperation", 0.3)
        betrayal_action = TrustAction("betrayal", -0.5)
        
        tracker.update_trust("agent1", "agent2", cooperation_action)
        tracker.update_trust("agent2", "agent1", betrayal_action)
        tracker.update_trust("agent1", "agent3", cooperation_action)
        
        assert len(tracker.cooperation_history) == 2
        assert len(tracker.betrayal_history) == 1
    
    def test_get_agent_betrayal_count(self):
        """Test getting betrayal count for specific agent."""
        tracker = TrustTracker()
        betrayal_action = TrustAction("betrayal", -0.4)
        
        tracker.update_trust("agent1", "agent2", betrayal_action)
        tracker.update_trust("agent1", "agent3", betrayal_action)
        
        betrayal_count = tracker.get_agent_betrayal_count("agent1")
        assert betrayal_count == 2
    
    def test_get_agent_cooperation_count(self):
        """Test getting cooperation count for specific agent."""
        tracker = TrustTracker()
        cooperation_action = TrustAction("cooperation", 0.3)
        
        tracker.update_trust("agent1", "agent2", cooperation_action)
        tracker.update_trust("agent1", "agent3", cooperation_action)
        
        cooperation_count = tracker.get_agent_cooperation_count("agent1")
        assert cooperation_count == 2


class TestTrustLevelConstraintsAndBoundaryConditions:
    """Test trust level constraints and boundary conditions."""
    
    def test_trust_level_boundary_enforcement(self):
        """Test that trust levels are strictly enforced within [-1.0, 1.0]."""
        tracker = TrustTracker()
        
        # Test extreme positive values
        extreme_positive = TrustAction("cooperation", 10.0)
        tracker.update_trust("agent1", "agent2", extreme_positive)
        assert tracker.get_trust_level("agent1", "agent2") == 1.0
        
        # Test extreme negative values
        extreme_negative = TrustAction("betrayal", -10.0)
        tracker.update_trust("agent3", "agent4", extreme_negative)
        assert tracker.get_trust_level("agent3", "agent4") == -1.0
    
    def test_cumulative_trust_changes(self):
        """Test cumulative trust changes with boundary enforcement."""
        tracker = TrustTracker()
        
        # Build up trust gradually
        small_cooperation = TrustAction("cooperation", 0.3)
        tracker.update_trust("agent1", "agent2", small_cooperation)
        tracker.update_trust("agent1", "agent2", small_cooperation)
        tracker.update_trust("agent1", "agent2", small_cooperation)
        tracker.update_trust("agent1", "agent2", small_cooperation)
        
        # Should be capped at 1.0
        assert tracker.get_trust_level("agent1", "agent2") == 1.0
    
    def test_trust_precision_handling(self):
        """Test that trust levels handle floating point precision correctly."""
        tracker = TrustTracker()
        
        # Use values that might cause floating point precision issues
        action = TrustAction("cooperation", 0.1)
        for _ in range(10):
            tracker.update_trust("agent1", "agent2", action)
        
        trust_level = tracker.get_trust_level("agent1", "agent2")
        assert trust_level == 1.0  # Should be exactly 1.0, not 0.9999999999
    
    def test_zero_impact_actions(self):
        """Test handling of actions with zero trust impact."""
        tracker = TrustTracker()
        zero_action = TrustAction("neutral", 0.0)
        
        tracker.update_trust("agent1", "agent2", zero_action)
        
        assert tracker.get_trust_level("agent1", "agent2") == 0.0
        # Should still be recorded in history
        assert len(tracker.cooperation_history) == 0
        assert len(tracker.betrayal_history) == 0
    
    def test_relationship_symmetry_independence(self):
        """Test that relationships are independent in both directions."""
        tracker = TrustTracker()
        
        cooperation_action = TrustAction("cooperation", 0.5)
        betrayal_action = TrustAction("betrayal", -0.3)
        
        tracker.update_trust("agent1", "agent2", cooperation_action)
        tracker.update_trust("agent2", "agent1", betrayal_action)
        
        assert tracker.get_trust_level("agent1", "agent2") == 0.5
        assert tracker.get_trust_level("agent2", "agent1") == -0.3
    
    def test_large_number_of_relationships(self):
        """Test handling of large number of relationships."""
        tracker = TrustTracker()
        action = TrustAction("cooperation", 0.1)
        
        # Create relationships between many agents
        agents = [f"agent{i}" for i in range(100)]
        for i, agent1 in enumerate(agents):
            for j, agent2 in enumerate(agents):
                if i != j:
                    tracker.update_trust(agent1, agent2, action)
        
        # Should handle all relationships correctly
        assert len(tracker.trust_matrix) == 100 * 99  # 100 agents, 99 relationships each
        
        # Test reputation calculation with many relationships
        reputation = tracker.calculate_reputation("agent0")
        assert reputation == 0.1  # All incoming trust should be 0.1