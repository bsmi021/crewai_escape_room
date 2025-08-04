"""
Integration tests for ResourceManager to verify it meets the requirements.
"""
import pytest
from src.escape_room_sim.competitive.models import ScarceResource
from src.escape_room_sim.competitive.resource_manager import ResourceManager


class TestResourceManagerRequirements:
    """Test ResourceManager against the specific requirements."""
    
    def test_requirement_3_1_resource_scarcity_enforcement(self):
        """Test that ResourceManager enforces resource scarcity (Requirement 3.1)."""
        # Create scarce resources
        resources = [
            ScarceResource(
                id="key",
                name="Master Key",
                description="Only one exists",
                required_for=["escape"],
                exclusivity=True,
                usage_cost=0
            ),
            ScarceResource(
                id="info",
                name="Shared Information",
                description="Can be shared",
                required_for=["puzzle"],
                exclusivity=False,
                usage_cost=0
            )
        ]
        
        manager = ResourceManager(resources)
        
        # Test exclusive resource scarcity
        result1 = manager.claim_resource("agent1", "key")
        assert result1.success is True
        
        result2 = manager.claim_resource("agent2", "key")
        assert result2.success is False
        assert "already owned" in result2.message
        
        # Test shared resource availability
        result3 = manager.claim_resource("agent1", "info")
        assert result3.success is True
        
        result4 = manager.claim_resource("agent2", "info")
        assert result4.success is True
    
    def test_requirement_3_2_ownership_tracking(self):
        """Test resource ownership tracking and conflict resolution (Requirement 3.2)."""
        resources = [
            ScarceResource(
                id="tool",
                name="Hammer",
                description="Exclusive tool",
                required_for=["break_wall"],
                exclusivity=True,
                usage_cost=10
            )
        ]
        
        manager = ResourceManager(resources)
        
        # Test ownership tracking
        manager.claim_resource("agent1", "tool")
        owner = manager.get_resource_owner("tool")
        assert owner == "agent1"
        
        # Test conflict resolution - second agent cannot claim
        result = manager.claim_resource("agent2", "tool")
        assert result.success is False
        
        # Test ownership transfer
        transfer_result = manager.transfer_resource("agent1", "agent2", "tool")
        assert transfer_result.success is True
        
        new_owner = manager.get_resource_owner("tool")
        assert new_owner == "agent2"
    
    def test_requirement_3_3_agent_specific_resource_access(self):
        """Test agent-specific resource filtering (Requirement 3.3)."""
        resources = [
            ScarceResource(
                id="key1",
                name="Agent1 Key",
                description="Owned by agent1",
                required_for=["door1"],
                exclusivity=True,
                usage_cost=0
            ),
            ScarceResource(
                id="key2",
                name="Agent2 Key",
                description="Owned by agent2",
                required_for=["door2"],
                exclusivity=True,
                usage_cost=0
            ),
            ScarceResource(
                id="map",
                name="Shared Map",
                description="Available to all",
                required_for=["navigation"],
                exclusivity=False,
                usage_cost=0
            )
        ]
        
        manager = ResourceManager(resources)
        
        # Set up ownership
        manager.claim_resource("agent1", "key1")
        manager.claim_resource("agent2", "key2")
        manager.claim_resource("agent1", "map")
        manager.claim_resource("agent2", "map")
        
        # Test agent-specific access
        agent1_resources = manager.get_available_resources("agent1")
        agent1_ids = [r.id for r in agent1_resources]
        assert "key1" in agent1_ids  # Owned by agent1
        assert "key2" not in agent1_ids  # Owned by agent2
        assert "map" in agent1_ids  # Shared resource
        
        agent2_resources = manager.get_available_resources("agent2")
        agent2_ids = [r.id for r in agent2_resources]
        assert "key1" not in agent2_ids  # Owned by agent1
        assert "key2" in agent2_ids  # Owned by agent2
        assert "map" in agent2_ids  # Shared resource
    
    def test_requirement_3_4_resource_transfer_mechanics(self):
        """Test resource transfer between agents (Requirement 3.4)."""
        resources = [
            ScarceResource(
                id="transferable",
                name="Transferable Tool",
                description="Can be transferred",
                required_for=["task"],
                exclusivity=True,
                usage_cost=5
            ),
            ScarceResource(
                id="shareable",
                name="Shareable Info",
                description="Can be shared",
                required_for=["knowledge"],
                exclusivity=False,
                usage_cost=0
            )
        ]
        
        manager = ResourceManager(resources)
        
        # Test exclusive resource transfer
        manager.claim_resource("agent1", "transferable")
        
        transfer_result = manager.transfer_resource("agent1", "agent2", "transferable")
        assert transfer_result.success is True
        assert manager.get_resource_owner("transferable") == "agent2"
        
        # Test shared resource "transfer" (actually sharing)
        manager.claim_resource("agent1", "shareable")
        
        share_result = manager.transfer_resource("agent1", "agent2", "shareable")
        assert share_result.success is True
        
        owners = manager.get_resource_owner("shareable")
        assert isinstance(owners, list)
        assert "agent1" in owners
        assert "agent2" in owners
    
    def test_requirement_3_5_usage_history_tracking(self):
        """Test comprehensive usage history tracking (Requirement 3.5)."""
        resources = [
            ScarceResource(
                id="tracked",
                name="Tracked Resource",
                description="Resource with full tracking",
                required_for=["test"],
                exclusivity=True,
                usage_cost=0
            )
        ]
        
        manager = ResourceManager(resources)
        
        # Perform various operations
        manager.claim_resource("agent1", "tracked")
        manager.transfer_resource("agent1", "agent2", "tracked")
        manager.release_resource("agent2", "tracked")
        
        # Test history tracking
        history = manager.get_usage_history_for_resource("tracked")
        assert len(history) == 4  # claim, release, claim, release
        
        actions = [record.action for record in history]
        assert "claim" in actions
        assert "release" in actions
        
        # Test agent-specific history
        agent1_history = manager.get_usage_history_for_agent("agent1")
        assert len(agent1_history) == 2  # claim and release
        
        agent2_history = manager.get_usage_history_for_agent("agent2")
        assert len(agent2_history) == 2  # claim and release
        
        # Test usage counts
        usage_count = manager.get_resource_usage_count("tracked")
        assert usage_count == 2  # Two claims total
        
        agent1_activity = manager.get_agent_activity_count("agent1")
        assert agent1_activity == 2  # claim + release
    
    def test_comprehensive_scarcity_enforcement(self):
        """Test comprehensive scarcity enforcement and conflict resolution."""
        resources = [
            ScarceResource(
                id="rare_tool",
                name="Rare Tool",
                description="Highly contested resource",
                required_for=["critical_task"],
                exclusivity=True,
                usage_cost=20
            )
        ]
        
        manager = ResourceManager(resources)
        
        # Multiple agents try to claim the same resource
        result1 = manager.claim_resource("agent1", "rare_tool")
        result2 = manager.claim_resource("agent2", "rare_tool")
        result3 = manager.claim_resource("agent3", "rare_tool")
        
        # Only first should succeed
        assert result1.success is True
        assert result2.success is False
        assert result3.success is False
        
        # Check conflict detection
        conflicts = manager.get_resource_conflicts()
        assert len(conflicts) == 1
        assert conflicts[0]["resource_id"] == "rare_tool"
        assert len(conflicts[0]["competing_agents"]) == 3  # All agents who tried to claim
        assert conflicts[0]["current_owner"] == "agent1"  # Only agent1 succeeded
        
        # Test scarcity metrics
        metrics = manager.get_scarcity_metrics()
        assert metrics["total_resources"] == 1
        assert metrics["exclusive_resources"] == 1
        assert metrics["claimed_exclusive"] == 1
        assert metrics["exclusive_utilization"] == 1.0
        assert metrics["total_claims"] == 1  # Only successful claims counted
        
        # Test integrity validation
        issues = manager.validate_resource_integrity()
        assert len(issues) == 0  # No integrity issues