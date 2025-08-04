"""
Unit tests for ResourceManager class using TDD approach.
"""
import pytest
from datetime import datetime
from typing import List, Dict

from src.escape_room_sim.competitive.models import ScarceResource
from src.escape_room_sim.competitive.resource_manager import (
    ResourceManager, ClaimResult, TransferResult, ResourceUsageRecord
)


class TestResourceManagerInitialization:
    """Test ResourceManager initialization with resource list."""
    
    def test_initialization_with_empty_resource_list(self):
        """Test ResourceManager can be initialized with empty resource list."""
        resources = []
        manager = ResourceManager(resources)
        
        assert manager.resources == {}
        assert manager.ownership == {}
        assert manager.usage_history == []
    
    def test_initialization_with_single_resource(self):
        """Test ResourceManager initialization with one resource."""
        resource = ScarceResource(
            id="key1",
            name="Master Key",
            description="Opens main door",
            required_for=["main_exit"],
            exclusivity=True,
            usage_cost=0
        )
        resources = [resource]
        manager = ResourceManager(resources)
        
        assert len(manager.resources) == 1
        assert "key1" in manager.resources
        assert manager.resources["key1"] == resource
        assert manager.ownership == {}
        assert manager.usage_history == []
    
    def test_initialization_with_multiple_resources(self):
        """Test ResourceManager initialization with multiple resources."""
        resources = [
            ScarceResource(
                id="key1",
                name="Master Key",
                description="Opens main door",
                required_for=["main_exit"],
                exclusivity=True,
                usage_cost=0
            ),
            ScarceResource(
                id="tool1",
                name="Lockpick",
                description="Can bypass locks",
                required_for=["side_exit"],
                exclusivity=True,
                usage_cost=30
            )
        ]
        manager = ResourceManager(resources)
        
        assert len(manager.resources) == 2
        assert "key1" in manager.resources
        assert "tool1" in manager.resources
        assert manager.ownership == {}
        assert manager.usage_history == []
    
    def test_initialization_with_duplicate_resource_ids_raises_error(self):
        """Test that duplicate resource IDs raise an error during initialization."""
        resources = [
            ScarceResource(
                id="key1",
                name="Master Key",
                description="Opens main door",
                required_for=["main_exit"],
                exclusivity=True,
                usage_cost=0
            ),
            ScarceResource(
                id="key1",  # Duplicate ID
                name="Backup Key",
                description="Another key",
                required_for=["side_exit"],
                exclusivity=True,
                usage_cost=0
            )
        ]
        
        with pytest.raises(ValueError, match="Duplicate resource ID"):
            ResourceManager(resources)


class TestClaimResourceMethod:
    """Test claim_resource method with exclusivity validation."""
    
    @pytest.fixture
    def manager_with_resources(self):
        """Create ResourceManager with test resources."""
        resources = [
            ScarceResource(
                id="exclusive_key",
                name="Exclusive Key",
                description="Can only be used by one agent",
                required_for=["main_exit"],
                exclusivity=True,
                usage_cost=0
            ),
            ScarceResource(
                id="shared_info",
                name="Shared Information",
                description="Can be shared between agents",
                required_for=["side_exit"],
                exclusivity=False,
                usage_cost=0
            )
        ]
        return ResourceManager(resources)
    
    def test_claim_nonexistent_resource_fails(self, manager_with_resources):
        """Test claiming a resource that doesn't exist fails."""
        result = manager_with_resources.claim_resource("agent1", "nonexistent")
        
        assert result.success is False
        assert "does not exist" in result.message
        assert result.resource_id == "nonexistent"
        assert result.agent_id == "agent1"
    
    def test_claim_exclusive_resource_success(self, manager_with_resources):
        """Test successfully claiming an exclusive resource."""
        result = manager_with_resources.claim_resource("agent1", "exclusive_key")
        
        assert result.success is True
        assert "successfully claimed" in result.message
        assert result.resource_id == "exclusive_key"
        assert result.agent_id == "agent1"
        assert manager_with_resources.ownership["exclusive_key"] == "agent1"
    
    def test_claim_already_owned_exclusive_resource_fails(self, manager_with_resources):
        """Test claiming an exclusive resource already owned by another agent fails."""
        # First agent claims the resource
        manager_with_resources.claim_resource("agent1", "exclusive_key")
        
        # Second agent tries to claim the same resource
        result = manager_with_resources.claim_resource("agent2", "exclusive_key")
        
        assert result.success is False
        assert "already owned by" in result.message
        assert result.resource_id == "exclusive_key"
        assert result.agent_id == "agent2"
        assert manager_with_resources.ownership["exclusive_key"] == "agent1"
    
    def test_claim_shared_resource_by_multiple_agents_succeeds(self, manager_with_resources):
        """Test multiple agents can claim the same shared resource."""
        # First agent claims shared resource
        result1 = manager_with_resources.claim_resource("agent1", "shared_info")
        assert result1.success is True
        
        # Second agent claims same shared resource
        result2 = manager_with_resources.claim_resource("agent2", "shared_info")
        assert result2.success is True
        
        # Both should have access
        assert "shared_info" in manager_with_resources.ownership
        # For shared resources, we track all owners
        assert isinstance(manager_with_resources.ownership["shared_info"], list)
        assert "agent1" in manager_with_resources.ownership["shared_info"]
        assert "agent2" in manager_with_resources.ownership["shared_info"]
    
    def test_claim_resource_with_empty_agent_id_fails(self, manager_with_resources):
        """Test claiming resource with empty agent ID fails."""
        result = manager_with_resources.claim_resource("", "exclusive_key")
        
        assert result.success is False
        assert "Agent ID cannot be empty" in result.message
    
    def test_claim_resource_with_none_agent_id_fails(self, manager_with_resources):
        """Test claiming resource with None agent ID fails."""
        result = manager_with_resources.claim_resource(None, "exclusive_key")
        
        assert result.success is False
        assert "Agent ID cannot be empty" in result.message


class TestResourceOwnershipTracking:
    """Test resource ownership tracking and conflict resolution."""
    
    @pytest.fixture
    def manager_with_mixed_resources(self):
        """Create ResourceManager with both exclusive and shared resources."""
        resources = [
            ScarceResource(
                id="tool1",
                name="Hammer",
                description="Exclusive tool",
                required_for=["break_wall"],
                exclusivity=True,
                usage_cost=10
            ),
            ScarceResource(
                id="info1",
                name="Map",
                description="Shared information",
                required_for=["find_exit"],
                exclusivity=False,
                usage_cost=0
            ),
            ScarceResource(
                id="key1",
                name="Special Key",
                description="Another exclusive resource",
                required_for=["unlock_door"],
                exclusivity=True,
                usage_cost=5
            )
        ]
        return ResourceManager(resources)
    
    def test_get_resource_owner_for_exclusive_resource(self, manager_with_mixed_resources):
        """Test getting owner of an exclusive resource."""
        manager_with_mixed_resources.claim_resource("agent1", "tool1")
        
        owner = manager_with_mixed_resources.get_resource_owner("tool1")
        assert owner == "agent1"
    
    def test_get_resource_owner_for_shared_resource(self, manager_with_mixed_resources):
        """Test getting owners of a shared resource."""
        manager_with_mixed_resources.claim_resource("agent1", "info1")
        manager_with_mixed_resources.claim_resource("agent2", "info1")
        
        owners = manager_with_mixed_resources.get_resource_owner("info1")
        assert isinstance(owners, list)
        assert "agent1" in owners
        assert "agent2" in owners
    
    def test_get_resource_owner_for_unclaimed_resource(self, manager_with_mixed_resources):
        """Test getting owner of unclaimed resource returns None."""
        owner = manager_with_mixed_resources.get_resource_owner("tool1")
        assert owner is None
    
    def test_is_resource_available_for_exclusive_unclaimed(self, manager_with_mixed_resources):
        """Test checking availability of unclaimed exclusive resource."""
        available = manager_with_mixed_resources.is_resource_available("tool1", "agent1")
        assert available is True
    
    def test_is_resource_available_for_exclusive_claimed_by_same_agent(self, manager_with_mixed_resources):
        """Test checking availability of exclusive resource claimed by same agent."""
        manager_with_mixed_resources.claim_resource("agent1", "tool1")
        
        available = manager_with_mixed_resources.is_resource_available("tool1", "agent1")
        assert available is True
    
    def test_is_resource_available_for_exclusive_claimed_by_different_agent(self, manager_with_mixed_resources):
        """Test checking availability of exclusive resource claimed by different agent."""
        manager_with_mixed_resources.claim_resource("agent1", "tool1")
        
        available = manager_with_mixed_resources.is_resource_available("tool1", "agent2")
        assert available is False
    
    def test_is_resource_available_for_shared_resource(self, manager_with_mixed_resources):
        """Test checking availability of shared resource (always available)."""
        manager_with_mixed_resources.claim_resource("agent1", "info1")
        
        # Should be available to other agents too
        available = manager_with_mixed_resources.is_resource_available("info1", "agent2")
        assert available is True
    
    def test_get_agent_resources_returns_owned_resources(self, manager_with_mixed_resources):
        """Test getting all resources owned by a specific agent."""
        manager_with_mixed_resources.claim_resource("agent1", "tool1")
        manager_with_mixed_resources.claim_resource("agent1", "info1")
        manager_with_mixed_resources.claim_resource("agent2", "key1")
        
        agent1_resources = manager_with_mixed_resources.get_agent_resources("agent1")
        assert "tool1" in agent1_resources
        assert "info1" in agent1_resources
        assert "key1" not in agent1_resources
        
        agent2_resources = manager_with_mixed_resources.get_agent_resources("agent2")
        assert "key1" in agent2_resources
        assert "tool1" not in agent2_resources


class TestTransferResourceMethod:
    """Test transfer_resource method between agents."""
    
    @pytest.fixture
    def manager_with_owned_resources(self):
        """Create ResourceManager with pre-owned resources."""
        resources = [
            ScarceResource(
                id="transferable_tool",
                name="Transferable Tool",
                description="Can be transferred between agents",
                required_for=["task1"],
                exclusivity=True,
                usage_cost=0
            ),
            ScarceResource(
                id="non_transferable_info",
                name="Personal Information",
                description="Cannot be transferred",
                required_for=["task2"],
                exclusivity=False,
                usage_cost=0
            )
        ]
        manager = ResourceManager(resources)
        # Pre-claim some resources
        manager.claim_resource("agent1", "transferable_tool")
        manager.claim_resource("agent1", "non_transferable_info")
        return manager
    
    def test_transfer_exclusive_resource_success(self, manager_with_owned_resources):
        """Test successful transfer of exclusive resource between agents."""
        result = manager_with_owned_resources.transfer_resource(
            "agent1", "agent2", "transferable_tool"
        )
        
        assert result.success is True
        assert "successfully transferred" in result.message
        assert result.resource_id == "transferable_tool"
        assert result.from_agent == "agent1"
        assert result.to_agent == "agent2"
        assert manager_with_owned_resources.ownership["transferable_tool"] == "agent2"
    
    def test_transfer_nonexistent_resource_fails(self, manager_with_owned_resources):
        """Test transferring nonexistent resource fails."""
        result = manager_with_owned_resources.transfer_resource(
            "agent1", "agent2", "nonexistent"
        )
        
        assert result.success is False
        assert "does not exist" in result.message
    
    def test_transfer_unowned_resource_fails(self, manager_with_owned_resources):
        """Test transferring resource not owned by from_agent fails."""
        # agent2 doesn't own transferable_tool
        result = manager_with_owned_resources.transfer_resource(
            "agent2", "agent1", "transferable_tool"
        )
        
        assert result.success is False
        assert "does not own" in result.message
    
    def test_transfer_shared_resource_adds_to_owners(self, manager_with_owned_resources):
        """Test transferring shared resource adds to_agent to owners list."""
        result = manager_with_owned_resources.transfer_resource(
            "agent1", "agent2", "non_transferable_info"
        )
        
        assert result.success is True
        owners = manager_with_owned_resources.ownership["non_transferable_info"]
        assert isinstance(owners, list)
        assert "agent1" in owners
        assert "agent2" in owners
    
    def test_transfer_to_same_agent_fails(self, manager_with_owned_resources):
        """Test transferring resource to same agent fails."""
        result = manager_with_owned_resources.transfer_resource(
            "agent1", "agent1", "transferable_tool"
        )
        
        assert result.success is False
        assert "cannot transfer to themselves" in result.message
    
    def test_transfer_with_empty_agent_ids_fails(self, manager_with_owned_resources):
        """Test transfer with empty agent IDs fails."""
        result = manager_with_owned_resources.transfer_resource(
            "", "agent2", "transferable_tool"
        )
        assert result.success is False
        
        result = manager_with_owned_resources.transfer_resource(
            "agent1", "", "transferable_tool"
        )
        assert result.success is False


class TestGetAvailableResourcesMethod:
    """Test get_available_resources method with agent-specific filtering."""
    
    @pytest.fixture
    def complex_manager(self):
        """Create ResourceManager with complex ownership scenario."""
        resources = [
            ScarceResource(
                id="free_tool",
                name="Free Tool",
                description="Unclaimed exclusive tool",
                required_for=["task1"],
                exclusivity=True,
                usage_cost=0
            ),
            ScarceResource(
                id="owned_tool",
                name="Owned Tool",
                description="Tool owned by agent1",
                required_for=["task2"],
                exclusivity=True,
                usage_cost=0
            ),
            ScarceResource(
                id="shared_info",
                name="Shared Info",
                description="Information available to all",
                required_for=["task3"],
                exclusivity=False,
                usage_cost=0
            ),
            ScarceResource(
                id="agent2_tool",
                name="Agent2 Tool",
                description="Tool owned by agent2",
                required_for=["task4"],
                exclusivity=True,
                usage_cost=0
            )
        ]
        manager = ResourceManager(resources)
        manager.claim_resource("agent1", "owned_tool")
        manager.claim_resource("agent1", "shared_info")
        manager.claim_resource("agent2", "agent2_tool")
        manager.claim_resource("agent2", "shared_info")
        return manager
    
    def test_get_available_resources_for_agent1(self, complex_manager):
        """Test getting available resources for agent1."""
        available = complex_manager.get_available_resources("agent1")
        
        resource_ids = [r.id for r in available]
        assert "free_tool" in resource_ids  # Unclaimed
        assert "owned_tool" in resource_ids  # Owned by agent1
        assert "shared_info" in resource_ids  # Shared resource
        assert "agent2_tool" not in resource_ids  # Owned by agent2
    
    def test_get_available_resources_for_agent2(self, complex_manager):
        """Test getting available resources for agent2."""
        available = complex_manager.get_available_resources("agent2")
        
        resource_ids = [r.id for r in available]
        assert "free_tool" in resource_ids  # Unclaimed
        assert "owned_tool" not in resource_ids  # Owned by agent1
        assert "shared_info" in resource_ids  # Shared resource
        assert "agent2_tool" in resource_ids  # Owned by agent2
    
    def test_get_available_resources_for_new_agent(self, complex_manager):
        """Test getting available resources for agent with no resources."""
        available = complex_manager.get_available_resources("agent3")
        
        resource_ids = [r.id for r in available]
        assert "free_tool" in resource_ids  # Unclaimed
        assert "owned_tool" not in resource_ids  # Owned by agent1
        assert "shared_info" in resource_ids  # Shared resource (always available)
        assert "agent2_tool" not in resource_ids  # Owned by agent2
    
    def test_get_available_resources_empty_agent_id_returns_empty(self, complex_manager):
        """Test getting available resources with empty agent ID returns empty list."""
        available = complex_manager.get_available_resources("")
        assert available == []
    
    def test_get_available_resources_filters_by_requirements(self, complex_manager):
        """Test filtering available resources by specific requirements."""
        # This test assumes the method supports filtering by required_for
        available = complex_manager.get_available_resources("agent1", required_for="task1")
        
        resource_ids = [r.id for r in available]
        assert "free_tool" in resource_ids  # Required for task1
        assert "owned_tool" not in resource_ids  # Not required for task1


class TestResourceUsageHistoryTracking:
    """Test resource usage history tracking."""
    
    @pytest.fixture
    def manager_for_history(self):
        """Create ResourceManager for history tracking tests."""
        resources = [
            ScarceResource(
                id="tracked_resource",
                name="Tracked Resource",
                description="Resource with usage tracking",
                required_for=["task1"],
                exclusivity=True,
                usage_cost=10
            )
        ]
        return ResourceManager(resources)
    
    def test_claim_resource_creates_usage_record(self, manager_for_history):
        """Test that claiming a resource creates a usage history record."""
        manager_for_history.claim_resource("agent1", "tracked_resource")
        
        assert len(manager_for_history.usage_history) == 1
        record = manager_for_history.usage_history[0]
        assert record.resource_id == "tracked_resource"
        assert record.agent_id == "agent1"
        assert record.action == "claim"
        assert isinstance(record.timestamp, datetime)
    
    def test_transfer_resource_creates_usage_records(self, manager_for_history):
        """Test that transferring a resource creates usage history records."""
        manager_for_history.claim_resource("agent1", "tracked_resource")
        manager_for_history.transfer_resource("agent1", "agent2", "tracked_resource")
        
        assert len(manager_for_history.usage_history) == 3  # claim, release, claim
        
        # Check the records
        claim_record = manager_for_history.usage_history[0]
        assert claim_record.action == "claim"
        assert claim_record.agent_id == "agent1"
        
        release_record = manager_for_history.usage_history[1]
        assert release_record.action == "release"
        assert release_record.agent_id == "agent1"
        
        new_claim_record = manager_for_history.usage_history[2]
        assert new_claim_record.action == "claim"
        assert new_claim_record.agent_id == "agent2"
    
    def test_get_usage_history_for_resource(self, manager_for_history):
        """Test getting usage history for specific resource."""
        manager_for_history.claim_resource("agent1", "tracked_resource")
        manager_for_history.transfer_resource("agent1", "agent2", "tracked_resource")
        
        history = manager_for_history.get_usage_history_for_resource("tracked_resource")
        assert len(history) == 3
        assert all(record.resource_id == "tracked_resource" for record in history)
    
    def test_get_usage_history_for_agent(self, manager_for_history):
        """Test getting usage history for specific agent."""
        manager_for_history.claim_resource("agent1", "tracked_resource")
        manager_for_history.transfer_resource("agent1", "agent2", "tracked_resource")
        
        agent1_history = manager_for_history.get_usage_history_for_agent("agent1")
        assert len(agent1_history) == 2  # claim and release
        assert all(record.agent_id == "agent1" for record in agent1_history)
        
        agent2_history = manager_for_history.get_usage_history_for_agent("agent2")
        assert len(agent2_history) == 1  # claim only
        assert agent2_history[0].agent_id == "agent2"
    
    def test_get_resource_usage_count(self, manager_for_history):
        """Test getting usage count for a resource."""
        manager_for_history.claim_resource("agent1", "tracked_resource")
        manager_for_history.transfer_resource("agent1", "agent2", "tracked_resource")
        
        usage_count = manager_for_history.get_resource_usage_count("tracked_resource")
        assert usage_count == 2  # Two claims (initial + transfer)
    
    def test_get_agent_activity_count(self, manager_for_history):
        """Test getting activity count for an agent."""
        manager_for_history.claim_resource("agent1", "tracked_resource")
        manager_for_history.transfer_resource("agent1", "agent2", "tracked_resource")
        
        agent1_activity = manager_for_history.get_agent_activity_count("agent1")
        assert agent1_activity == 2  # claim + release
        
        agent2_activity = manager_for_history.get_agent_activity_count("agent2")
        assert agent2_activity == 1  # claim only