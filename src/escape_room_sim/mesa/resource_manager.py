"""
Mesa Resource Management System

Agent D: State Management & Integration Specialist
Manages resource scarcity, competition, and allocation in the Mesa environment.
"""

import random
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
import threading


class ResourceType(Enum):
    """Types of resources in the escape room"""
    KEY = "key"
    TOOL = "tool"
    CONSUMABLE = "consumable"
    INFORMATION = "information"
    PUZZLE_PIECE = "puzzle_piece"
    ENERGY = "energy"
    TIME = "time"


class ResourcePriority(Enum):
    """Resource priority levels"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class Resource:
    """Represents a resource in the system"""
    resource_id: str
    resource_type: str
    position: Optional[Tuple[int, int]] = None
    room: Optional[str] = None
    quantity: int = 1
    max_quantity: int = 1
    renewable: bool = False
    renewal_rate: float = 0.0  # Resources per second
    priority: str = ResourcePriority.MEDIUM.value
    owner: Optional[str] = None
    claimed_by: Optional[str] = None
    available: bool = True
    created_timestamp: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ResourceClaim:
    """Represents a claim on a resource"""
    claim_id: str
    agent_id: str
    resource_id: str
    claim_type: str  # "reserve", "claim", "use"
    timestamp: datetime = field(default_factory=datetime.now)
    duration: Optional[float] = None  # seconds
    priority: float = 0.5
    justification: str = ""
    status: str = "pending"  # pending, approved, denied, expired


@dataclass
class ResourceTransaction:
    """Represents a resource transaction between agents"""
    transaction_id: str
    from_agent: str
    to_agent: str
    resource_id: str
    quantity: int
    transaction_type: str  # "transfer", "trade", "gift", "steal"
    price: Optional[Dict[str, Any]] = None  # What was given in return
    timestamp: datetime = field(default_factory=datetime.now)
    status: str = "completed"
    conditions: Dict[str, Any] = field(default_factory=dict)


class MesaResourceManager:
    """
    Manages resources in the Mesa escape room environment
    
    Features:
    - Resource registration and tracking
    - Scarcity mechanics and competition
    - Agent resource claiming and transfer
    - Resource renewal and degradation
    - Competition analytics
    """
    
    def __init__(self):
        self.resources: Dict[str, Resource] = {}
        self.agent_resources: Dict[str, List[str]] = {}
        self.resource_claims: Dict[str, ResourceClaim] = {}
        self.transaction_history: List[ResourceTransaction] = []
        
        # Competition tracking
        self.competition_metrics = {
            "total_claims": 0,
            "successful_claims": 0,
            "failed_claims": 0,
            "resource_conflicts": 0,
            "agent_interactions": 0
        }
        
        # Scarcity settings
        self.scarcity_multiplier = 1.0
        self.enable_degradation = True
        self.enable_renewal = True
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Timer for resource updates
        self.last_update_time = datetime.now()
        self.update_interval = 1.0  # seconds
    
    def register_resource(self, resource_id: str, resource_type: str, 
                         position: Tuple[int, int] = None, room: str = None,
                         quantity: int = 1, renewable: bool = False,
                         priority: str = ResourcePriority.MEDIUM.value) -> bool:
        """Register a new resource in the system"""
        with self._lock:
            if resource_id in self.resources:
                return False  # Resource already exists
            
            resource = Resource(
                resource_id=resource_id,
                resource_type=resource_type,
                position=position,
                room=room,
                quantity=quantity,
                max_quantity=quantity,
                renewable=renewable,
                priority=priority
            )
            
            self.resources[resource_id] = resource
            return True
    
    def claim_resource(self, agent_id: str, resource_id: str, 
                      claim_type: str = "claim", priority: float = 0.5,
                      justification: str = "") -> bool:
        """Agent attempts to claim a resource"""
        with self._lock:
            self.competition_metrics["total_claims"] += 1
            
            # Check if resource exists
            if resource_id not in self.resources:
                self.competition_metrics["failed_claims"] += 1
                return False
            
            resource = self.resources[resource_id]
            
            # Check if resource is available
            if not resource.available or resource.quantity <= 0:
                self.competition_metrics["failed_claims"] += 1
                return False
            
            # Check if already claimed by someone else
            if resource.claimed_by and resource.claimed_by != agent_id:
                self.competition_metrics["failed_claims"] += 1
                self.competition_metrics["resource_conflicts"] += 1
                return False
            
            # Create claim
            claim_id = f"claim_{agent_id}_{resource_id}_{datetime.now().timestamp()}"
            claim = ResourceClaim(
                claim_id=claim_id,
                agent_id=agent_id,
                resource_id=resource_id,
                claim_type=claim_type,
                priority=priority,
                justification=justification,
                status="approved"
            )
            
            self.resource_claims[claim_id] = claim
            
            # Update resource
            resource.claimed_by = agent_id
            resource.owner = agent_id
            resource.quantity -= 1
            if resource.quantity <= 0:
                resource.available = False
            resource.last_updated = datetime.now()
            
            # Update agent resources
            if agent_id not in self.agent_resources:
                self.agent_resources[agent_id] = []
            self.agent_resources[agent_id].append(resource_id)
            
            self.competition_metrics["successful_claims"] += 1
            self.competition_metrics["agent_interactions"] += 1
            
            return True
    
    def transfer_resource(self, resource_id: str, from_agent: str, to_agent: str,
                         transaction_type: str = "transfer", 
                         price: Dict[str, Any] = None) -> bool:
        """Transfer resource between agents"""
        with self._lock:
            # Validate agents have/can receive resource
            if from_agent not in self.agent_resources:
                return False
            
            if resource_id not in self.agent_resources[from_agent]:
                return False
            
            # Perform transfer
            self.agent_resources[from_agent].remove(resource_id)
            
            if to_agent not in self.agent_resources:
                self.agent_resources[to_agent] = []
            self.agent_resources[to_agent].append(resource_id)
            
            # Update resource ownership
            if resource_id in self.resources:
                resource = self.resources[resource_id]
                resource.owner = to_agent
                resource.claimed_by = to_agent
                resource.last_updated = datetime.now()
            
            # Record transaction
            transaction = ResourceTransaction(
                transaction_id=f"tx_{from_agent}_{to_agent}_{resource_id}_{datetime.now().timestamp()}",
                from_agent=from_agent,
                to_agent=to_agent,
                resource_id=resource_id,
                quantity=1,
                transaction_type=transaction_type,
                price=price
            )
            
            self.transaction_history.append(transaction)
            self.competition_metrics["agent_interactions"] += 1
            
            return True
    
    def release_resource(self, agent_id: str, resource_id: str) -> bool:
        """Agent releases a claimed resource"""
        with self._lock:
            if agent_id not in self.agent_resources:
                return False
            
            if resource_id not in self.agent_resources[agent_id]:
                return False
            
            # Remove from agent
            self.agent_resources[agent_id].remove(resource_id)
            
            # Update resource
            if resource_id in self.resources:
                resource = self.resources[resource_id]
                resource.claimed_by = None
                resource.owner = None
                resource.quantity += 1
                resource.available = True
                resource.last_updated = datetime.now()
            
            return True
    
    def get_available_resources(self, room: str = None, 
                              resource_type: str = None) -> List[Resource]:
        """Get list of available resources"""
        with self._lock:
            available = []
            
            for resource in self.resources.values():
                if not resource.available or resource.quantity <= 0:
                    continue
                
                if room and resource.room != room:
                    continue
                
                if resource_type and resource.resource_type != resource_type:
                    continue
                
                available.append(resource)
            
            return available
    
    def get_agent_resources(self, agent_id: str) -> List[str]:
        """Get resources owned by agent"""
        with self._lock:
            return self.agent_resources.get(agent_id, []).copy()
    
    def get_resource_owner(self, resource_id: str) -> Optional[str]:
        """Get current owner of resource"""
        with self._lock:
            if resource_id in self.resources:
                return self.resources[resource_id].owner
            return None
    
    def get_scarcity_info(self) -> Dict[str, Dict[str, Any]]:
        """Get resource scarcity information"""
        with self._lock:
            scarcity_info = {}
            
            for resource_id, resource in self.resources.items():
                scarcity_info[resource_id] = {
                    "available": resource.quantity,
                    "total": resource.max_quantity,
                    "scarcity_level": self._calculate_scarcity_level(resource),
                    "claimed_by": resource.claimed_by,
                    "demand": self._calculate_demand(resource_id),
                    "competition_level": self._calculate_competition_level(resource_id)
                }
            
            return scarcity_info
    
    def update_resources(self, delta_time: float = None):
        """Update resource states (renewal, degradation, etc.)"""
        with self._lock:
            if delta_time is None:
                current_time = datetime.now()
                delta_time = (current_time - self.last_update_time).total_seconds()
                self.last_update_time = current_time
            
            if delta_time < self.update_interval:
                return
            
            for resource in self.resources.values():
                self._update_single_resource(resource, delta_time)
    
    def force_scarcity(self, resource_id: str, new_quantity: int = 0):
        """Force scarcity for testing or game events"""
        with self._lock:
            if resource_id in self.resources:
                resource = self.resources[resource_id]
                resource.quantity = new_quantity
                resource.available = new_quantity > 0
                resource.last_updated = datetime.now()
    
    def get_competition_analytics(self) -> Dict[str, Any]:
        """Get competition analytics"""
        with self._lock:
            total_claims = self.competition_metrics["total_claims"]
            success_rate = (self.competition_metrics["successful_claims"] / total_claims 
                          if total_claims > 0 else 0.0)
            
            # Resource hotspots (most contested resources)
            resource_contest_count = {}
            for claim in self.resource_claims.values():
                resource_id = claim.resource_id
                if resource_id not in resource_contest_count:
                    resource_contest_count[resource_id] = 0
                resource_contest_count[resource_id] += 1
            
            hotspots = sorted(resource_contest_count.items(), 
                            key=lambda x: x[1], reverse=True)[:5]
            
            # Agent competitiveness
            agent_claim_count = {}
            for claim in self.resource_claims.values():
                agent_id = claim.agent_id
                if agent_id not in agent_claim_count:
                    agent_claim_count[agent_id] = 0
                agent_claim_count[agent_id] += 1
            
            return {
                "competition_metrics": self.competition_metrics.copy(),
                "success_rate": success_rate,
                "resource_hotspots": hotspots,
                "agent_competitiveness": agent_claim_count,
                "total_transactions": len(self.transaction_history),
                "average_resource_lifetime": self._calculate_average_resource_lifetime(),
                "scarcity_pressure": self._calculate_overall_scarcity_pressure()
            }
    
    def simulate_resource_discovery(self, room: str, discovery_rate: float = 0.1):
        """Simulate discovery of new resources"""
        with self._lock:
            if random.random() < discovery_rate:
                # Generate new resource
                resource_types = [rt.value for rt in ResourceType]
                resource_type = random.choice(resource_types)
                
                resource_id = f"discovered_{resource_type}_{datetime.now().timestamp()}"
                position = self._get_random_position_in_room(room)
                
                self.register_resource(
                    resource_id=resource_id,
                    resource_type=resource_type,
                    position=position,
                    room=room,
                    quantity=1,
                    renewable=False,
                    priority=random.choice(list(ResourcePriority)).value
                )
                
                return resource_id
        
        return None
    
    def sync_from_unified_state(self, unified_resources: Dict[str, Any]):
        """Synchronize from unified state manager"""
        with self._lock:
            # Update resource states from unified state
            for resource_id, resource_data in unified_resources.items():
                if resource_id in self.resources:
                    resource = self.resources[resource_id]
                    
                    # Update based on unified state data
                    if "quantity" in resource_data:
                        resource.quantity = resource_data["quantity"]
                    if "available" in resource_data:
                        resource.available = resource_data["available"]
                    if "owner" in resource_data:
                        resource.owner = resource_data["owner"]
                    
                    resource.last_updated = datetime.now()
    
    def get_state_summary(self) -> Dict[str, Any]:
        """Get summary of resource manager state"""
        with self._lock:
            return {
                "total_resources": len(self.resources),
                "available_resources": sum(1 for r in self.resources.values() if r.available),
                "claimed_resources": sum(1 for r in self.resources.values() if r.claimed_by),
                "total_agents": len(self.agent_resources),
                "total_claims": len(self.resource_claims),
                "total_transactions": len(self.transaction_history),
                "competition_metrics": self.competition_metrics.copy(),
                "last_update": self.last_update_time.isoformat()
            }
    
    # Private helper methods
    
    def _calculate_scarcity_level(self, resource: Resource) -> str:
        """Calculate scarcity level for a resource"""
        if resource.max_quantity == 0:
            return "infinite"
        
        ratio = resource.quantity / resource.max_quantity
        
        if ratio <= 0.1:
            return "critical"
        elif ratio <= 0.3:
            return "high"
        elif ratio <= 0.6:
            return "medium"
        else:
            return "low"
    
    def _calculate_demand(self, resource_id: str) -> int:
        """Calculate demand for a resource based on claims"""
        demand = 0
        for claim in self.resource_claims.values():
            if claim.resource_id == resource_id:
                demand += 1
        return demand
    
    def _calculate_competition_level(self, resource_id: str) -> str:
        """Calculate competition level for a resource"""
        demand = self._calculate_demand(resource_id)
        
        if resource_id in self.resources:
            supply = self.resources[resource_id].max_quantity
            
            if supply == 0:
                return "none"
            
            competition_ratio = demand / supply
            
            if competition_ratio >= 3.0:
                return "intense"
            elif competition_ratio >= 2.0:
                return "high"
            elif competition_ratio >= 1.5:
                return "moderate"
            else:
                return "low"
        
        return "unknown"
    
    def _update_single_resource(self, resource: Resource, delta_time: float):
        """Update a single resource"""
        # Handle renewable resources
        if self.enable_renewal and resource.renewable and resource.renewal_rate > 0:
            renewal_amount = resource.renewal_rate * delta_time
            if random.random() < renewal_amount:
                resource.quantity = min(resource.max_quantity, resource.quantity + 1)
                if resource.quantity > 0:
                    resource.available = True
        
        # Handle degradation (optional mechanic)
        if self.enable_degradation and resource.resource_type == ResourceType.CONSUMABLE.value:
            degradation_rate = 0.001  # Very slow degradation
            if random.random() < degradation_rate * delta_time:
                resource.quantity = max(0, resource.quantity - 1)
                if resource.quantity <= 0:
                    resource.available = False
    
    def _calculate_average_resource_lifetime(self) -> float:
        """Calculate average resource lifetime"""
        if not self.transaction_history:
            return 0.0
        
        lifetimes = []
        for transaction in self.transaction_history:
            # Simple approximation - more sophisticated tracking would be needed
            lifetimes.append(60.0)  # Assume 60 seconds average
        
        return sum(lifetimes) / len(lifetimes)
    
    def _calculate_overall_scarcity_pressure(self) -> float:
        """Calculate overall scarcity pressure in the system"""
        if not self.resources:
            return 0.0
        
        total_scarcity = 0.0
        for resource in self.resources.values():
            if resource.max_quantity > 0:
                scarcity = 1.0 - (resource.quantity / resource.max_quantity)
                total_scarcity += scarcity
        
        return total_scarcity / len(self.resources)
    
    def _get_random_position_in_room(self, room: str) -> Tuple[int, int]:
        """Get random position in room (placeholder)"""
        # This would integrate with room system to get valid positions
        return (random.randint(0, 9), random.randint(0, 9))