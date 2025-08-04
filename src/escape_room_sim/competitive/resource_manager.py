"""
ResourceManager for handling scarcity enforcement in competitive scenarios.
"""
from dataclasses import dataclass
from datetime import datetime
from typing import List, Dict, Optional, Union

from .models import ScarceResource


@dataclass
class ClaimResult:
    """Result of a resource claim attempt."""
    success: bool
    message: str
    resource_id: str
    agent_id: str
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


@dataclass
class TransferResult:
    """Result of a resource transfer attempt."""
    success: bool
    message: str
    resource_id: str
    from_agent: str
    to_agent: str
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


@dataclass
class ResourceUsageRecord:
    """Record of resource usage for history tracking."""
    resource_id: str
    agent_id: str
    action: str  # "claim", "release", "transfer"
    timestamp: datetime
    details: Optional[str] = None


class ResourceManager:
    """Manages scarce resources and enforces competition rules."""
    
    def __init__(self, resources: List[ScarceResource]):
        """Initialize ResourceManager with list of resources."""
        self.resources = {}
        self.ownership = {}
        self.usage_history = []
        
        # Check for duplicate resource IDs
        resource_ids = set()
        for resource in resources:
            if resource.id in resource_ids:
                raise ValueError(f"Duplicate resource ID: {resource.id}")
            resource_ids.add(resource.id)
            self.resources[resource.id] = resource
    
    def claim_resource(self, agent_id: str, resource_id: str) -> ClaimResult:
        """Attempt to claim a scarce resource."""
        # Validate agent ID
        if not agent_id or agent_id is None:
            return ClaimResult(
                success=False,
                message="Agent ID cannot be empty",
                resource_id=resource_id,
                agent_id=str(agent_id) if agent_id is not None else "None"
            )
        
        # Check if resource exists
        if resource_id not in self.resources:
            return ClaimResult(
                success=False,
                message=f"Resource '{resource_id}' does not exist",
                resource_id=resource_id,
                agent_id=agent_id
            )
        
        resource = self.resources[resource_id]
        
        # Handle exclusive resources
        if resource.exclusivity:
            if resource_id in self.ownership and self.ownership[resource_id] != agent_id:
                # Record failed attempt for conflict tracking
                self.usage_history.append(ResourceUsageRecord(
                    resource_id=resource_id,
                    agent_id=agent_id,
                    action="failed_claim",
                    timestamp=datetime.now(),
                    details=f"Resource already owned by {self.ownership[resource_id]}"
                ))
                
                return ClaimResult(
                    success=False,
                    message=f"Resource '{resource_id}' is already owned by {self.ownership[resource_id]}",
                    resource_id=resource_id,
                    agent_id=agent_id
                )
            
            # Claim the exclusive resource
            self.ownership[resource_id] = agent_id
        else:
            # Handle shared resources
            if resource_id not in self.ownership:
                self.ownership[resource_id] = []
            
            if agent_id not in self.ownership[resource_id]:
                self.ownership[resource_id].append(agent_id)
        
        # Record usage history
        self.usage_history.append(ResourceUsageRecord(
            resource_id=resource_id,
            agent_id=agent_id,
            action="claim",
            timestamp=datetime.now()
        ))
        
        return ClaimResult(
            success=True,
            message=f"Resource '{resource_id}' successfully claimed by {agent_id}",
            resource_id=resource_id,
            agent_id=agent_id
        )
    
    def transfer_resource(self, from_agent: str, to_agent: str, resource_id: str) -> TransferResult:
        """Handle resource trading between agents."""
        # Validate agent IDs
        if not from_agent or not to_agent:
            return TransferResult(
                success=False,
                message="Agent IDs cannot be empty",
                resource_id=resource_id,
                from_agent=from_agent or "",
                to_agent=to_agent or ""
            )
        
        if from_agent == to_agent:
            return TransferResult(
                success=False,
                message="Agent cannot transfer to themselves",
                resource_id=resource_id,
                from_agent=from_agent,
                to_agent=to_agent
            )
        
        # Check if resource exists
        if resource_id not in self.resources:
            return TransferResult(
                success=False,
                message=f"Resource '{resource_id}' does not exist",
                resource_id=resource_id,
                from_agent=from_agent,
                to_agent=to_agent
            )
        
        # Check if from_agent owns the resource
        if resource_id not in self.ownership:
            return TransferResult(
                success=False,
                message=f"Agent '{from_agent}' does not own resource '{resource_id}'",
                resource_id=resource_id,
                from_agent=from_agent,
                to_agent=to_agent
            )
        
        resource = self.resources[resource_id]
        owner = self.ownership[resource_id]
        
        # Check ownership for exclusive resources
        if resource.exclusivity:
            if owner != from_agent:
                return TransferResult(
                    success=False,
                    message=f"Agent '{from_agent}' does not own resource '{resource_id}'",
                    resource_id=resource_id,
                    from_agent=from_agent,
                    to_agent=to_agent
                )
            
            # Record release
            self.usage_history.append(ResourceUsageRecord(
                resource_id=resource_id,
                agent_id=from_agent,
                action="release",
                timestamp=datetime.now()
            ))
            
            # Transfer ownership
            self.ownership[resource_id] = to_agent
            
            # Record new claim
            self.usage_history.append(ResourceUsageRecord(
                resource_id=resource_id,
                agent_id=to_agent,
                action="claim",
                timestamp=datetime.now()
            ))
        else:
            # Handle shared resources
            if not isinstance(owner, list) or from_agent not in owner:
                return TransferResult(
                    success=False,
                    message=f"Agent '{from_agent}' does not own resource '{resource_id}'",
                    resource_id=resource_id,
                    from_agent=from_agent,
                    to_agent=to_agent
                )
            
            # Add to_agent to owners list if not already there
            if to_agent not in owner:
                owner.append(to_agent)
                
                # Record the sharing
                self.usage_history.append(ResourceUsageRecord(
                    resource_id=resource_id,
                    agent_id=to_agent,
                    action="claim",
                    timestamp=datetime.now()
                ))
        
        return TransferResult(
            success=True,
            message=f"Resource '{resource_id}' successfully transferred from {from_agent} to {to_agent}",
            resource_id=resource_id,
            from_agent=from_agent,
            to_agent=to_agent
        )
    
    def get_available_resources(self, agent_id: str, required_for: str = None) -> List[ScarceResource]:
        """Return resources accessible to specific agent."""
        if not agent_id:
            return []
        
        available_resources = []
        
        for resource_id, resource in self.resources.items():
            if self.is_resource_available(resource_id, agent_id):
                # Filter by required_for if specified
                if required_for is None or required_for in resource.required_for:
                    available_resources.append(resource)
        
        return available_resources
    
    def get_resource_owner(self, resource_id: str) -> Union[str, List[str], None]:
        """Get the owner(s) of a resource."""
        return self.ownership.get(resource_id)
    
    def is_resource_available(self, resource_id: str, agent_id: str) -> bool:
        """Check if a resource is available to a specific agent."""
        if resource_id not in self.resources:
            return False
        
        resource = self.resources[resource_id]
        
        # Shared resources are always available
        if not resource.exclusivity:
            return True
        
        # Exclusive resources are available if unclaimed or owned by the same agent
        if resource_id not in self.ownership:
            return True
        
        return self.ownership[resource_id] == agent_id
    
    def get_agent_resources(self, agent_id: str) -> List[str]:
        """Get all resources owned by a specific agent."""
        owned_resources = []
        
        for resource_id, owner in self.ownership.items():
            if isinstance(owner, str) and owner == agent_id:
                owned_resources.append(resource_id)
            elif isinstance(owner, list) and agent_id in owner:
                owned_resources.append(resource_id)
        
        return owned_resources
    
    def get_usage_history_for_resource(self, resource_id: str) -> List[ResourceUsageRecord]:
        """Get usage history for a specific resource."""
        return [record for record in self.usage_history if record.resource_id == resource_id]
    
    def get_usage_history_for_agent(self, agent_id: str) -> List[ResourceUsageRecord]:
        """Get usage history for a specific agent."""
        return [record for record in self.usage_history if record.agent_id == agent_id]
    
    def get_resource_usage_count(self, resource_id: str) -> int:
        """Get the number of times a resource has been claimed."""
        return len([record for record in self.usage_history 
                   if record.resource_id == resource_id and record.action == "claim"])
    
    def get_agent_activity_count(self, agent_id: str) -> int:
        """Get the total activity count for an agent."""
        return len([record for record in self.usage_history if record.agent_id == agent_id])
    
    def release_resource(self, agent_id: str, resource_id: str) -> ClaimResult:
        """Release a resource owned by an agent."""
        if resource_id not in self.resources:
            return ClaimResult(
                success=False,
                message=f"Resource '{resource_id}' does not exist",
                resource_id=resource_id,
                agent_id=agent_id
            )
        
        if resource_id not in self.ownership:
            return ClaimResult(
                success=False,
                message=f"Resource '{resource_id}' is not owned by anyone",
                resource_id=resource_id,
                agent_id=agent_id
            )
        
        resource = self.resources[resource_id]
        owner = self.ownership[resource_id]
        
        if resource.exclusivity:
            if owner != agent_id:
                return ClaimResult(
                    success=False,
                    message=f"Resource '{resource_id}' is not owned by {agent_id}",
                    resource_id=resource_id,
                    agent_id=agent_id
                )
            
            # Release exclusive resource
            del self.ownership[resource_id]
        else:
            # Handle shared resource
            if not isinstance(owner, list) or agent_id not in owner:
                return ClaimResult(
                    success=False,
                    message=f"Resource '{resource_id}' is not owned by {agent_id}",
                    resource_id=resource_id,
                    agent_id=agent_id
                )
            
            # Remove agent from owners list
            owner.remove(agent_id)
            if not owner:  # If no owners left, remove the resource from ownership
                del self.ownership[resource_id]
        
        # Record release
        self.usage_history.append(ResourceUsageRecord(
            resource_id=resource_id,
            agent_id=agent_id,
            action="release",
            timestamp=datetime.now()
        ))
        
        return ClaimResult(
            success=True,
            message=f"Resource '{resource_id}' successfully released by {agent_id}",
            resource_id=resource_id,
            agent_id=agent_id
        )
    
    def get_resource_conflicts(self) -> List[Dict[str, any]]:
        """Get list of resources that have conflicts (multiple agents wanting exclusive resources)."""
        conflicts = []
        
        # Identify conflicts from usage history
        for resource_id, resource in self.resources.items():
            if resource.exclusivity:
                # Count unique agents who have tried to claim this resource (successful or failed)
                claimants = set()
                for record in self.usage_history:
                    if (record.resource_id == resource_id and 
                        record.action in ["claim", "failed_claim"]):
                        claimants.add(record.agent_id)
                
                if len(claimants) > 1:
                    conflicts.append({
                        "resource_id": resource_id,
                        "resource_name": resource.name,
                        "competing_agents": list(claimants),
                        "current_owner": self.ownership.get(resource_id)
                    })
        
        return conflicts
    
    def get_scarcity_metrics(self) -> Dict[str, any]:
        """Get metrics about resource scarcity and competition."""
        total_resources = len(self.resources)
        exclusive_resources = sum(1 for r in self.resources.values() if r.exclusivity)
        shared_resources = total_resources - exclusive_resources
        
        claimed_exclusive = sum(1 for resource_id, resource in self.resources.items() 
                               if resource.exclusivity and resource_id in self.ownership)
        
        claimed_shared = sum(1 for resource_id, resource in self.resources.items() 
                            if not resource.exclusivity and resource_id in self.ownership)
        
        return {
            "total_resources": total_resources,
            "exclusive_resources": exclusive_resources,
            "shared_resources": shared_resources,
            "claimed_exclusive": claimed_exclusive,
            "claimed_shared": claimed_shared,
            "exclusive_utilization": claimed_exclusive / exclusive_resources if exclusive_resources > 0 else 0,
            "shared_utilization": claimed_shared / shared_resources if shared_resources > 0 else 0,
            "total_claims": len([r for r in self.usage_history if r.action == "claim"]),
            "total_releases": len([r for r in self.usage_history if r.action == "release"]),
            "conflicts": len(self.get_resource_conflicts())
        }
    
    def validate_resource_integrity(self) -> List[str]:
        """Validate the integrity of resource ownership and history."""
        issues = []
        
        # Check that all owned resources exist
        for resource_id in self.ownership.keys():
            if resource_id not in self.resources:
                issues.append(f"Owned resource '{resource_id}' does not exist in resources")
        
        # Check that exclusive resources have single owners
        for resource_id, owner in self.ownership.items():
            resource = self.resources.get(resource_id)
            if resource and resource.exclusivity:
                if not isinstance(owner, str):
                    issues.append(f"Exclusive resource '{resource_id}' has non-string owner: {owner}")
            elif resource and not resource.exclusivity:
                if not isinstance(owner, list):
                    issues.append(f"Shared resource '{resource_id}' has non-list owner: {owner}")
        
        # Check usage history consistency
        for record in self.usage_history:
            if record.resource_id not in self.resources:
                issues.append(f"Usage record references non-existent resource: {record.resource_id}")
        
        return issues