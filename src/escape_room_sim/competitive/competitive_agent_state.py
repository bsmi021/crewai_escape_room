"""
CompetitiveAgentState tracking system for managing agent state in competitive scenarios.

This module implements comprehensive agent state tracking including resource ownership,
secret information known, trust relationships, moral choice history, and ethical burden
calculation with state synchronization capabilities.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from datetime import datetime
import copy
from .models import SecretInformation, MoralChoice


@dataclass
class CompetitiveAgentState:
    """
    Comprehensive state tracking for an agent in competitive scenarios.
    
    Tracks all aspects of an agent's state including owned resources, known secrets,
    trust relationships with other agents, moral choice history, and accumulated
    ethical burden with automatic state synchronization.
    """
    agent_id: str
    resources_owned: List[str] = field(default_factory=list)
    secrets_known: List[SecretInformation] = field(default_factory=list)
    trust_relationships: Dict[str, float] = field(default_factory=dict)
    moral_choice_history: List[Dict[str, Any]] = field(default_factory=list)
    ethical_burden: float = 0.0
    last_updated: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        """Validate agent state initialization."""
        if not self.agent_id or not self.agent_id.strip():
            raise ValueError("Agent ID cannot be empty")
    
    # Resource Ownership Tracking Methods
    
    def add_resource(self, resource_id: str) -> None:
        """Add a resource to the agent's ownership."""
        if resource_id and resource_id not in self.resources_owned:
            self.resources_owned.append(resource_id)
            self._update_timestamp()
    
    def remove_resource(self, resource_id: str) -> bool:
        """Remove a resource from the agent's ownership. Returns True if removed."""
        if resource_id in self.resources_owned:
            self.resources_owned.remove(resource_id)
            self._update_timestamp()
            return True
        return False
    
    def has_resource(self, resource_id: str) -> bool:
        """Check if the agent owns a specific resource."""
        return resource_id in self.resources_owned
    
    def get_resource_count(self) -> int:
        """Get the total number of resources owned by the agent."""
        return len(self.resources_owned)
    
    # Secret Information Tracking Methods
    
    def add_secret(self, secret: SecretInformation) -> None:
        """Add secret information to the agent's knowledge."""
        # Prevent duplicates by checking if secret with same ID already exists
        if not any(existing.id == secret.id for existing in self.secrets_known):
            self.secrets_known.append(secret)
            self._update_timestamp()
    
    def has_secret(self, secret_id: str) -> bool:
        """Check if the agent knows a specific secret."""
        return any(secret.id == secret_id for secret in self.secrets_known)
    
    def get_secret(self, secret_id: str) -> Optional[SecretInformation]:
        """Get a specific secret by ID. Returns None if not known."""
        for secret in self.secrets_known:
            if secret.id == secret_id:
                return secret
        return None
    
    def get_secrets_count(self) -> int:
        """Get the total number of secrets known by the agent."""
        return len(self.secrets_known)
    
    # Trust Relationship Tracking Methods
    
    def update_trust_relationship(self, other_agent: str, trust_level: float) -> None:
        """Update trust level toward another agent."""
        self.trust_relationships[other_agent] = trust_level
        self._update_timestamp()
    
    def get_trust_level(self, other_agent: str) -> float:
        """Get trust level toward another agent. Returns 0.0 if no relationship."""
        return self.trust_relationships.get(other_agent, 0.0)
    
    def has_trust_relationship(self, other_agent: str) -> bool:
        """Check if there's a trust relationship with another agent."""
        return other_agent in self.trust_relationships
    
    def get_all_trust_relationships(self) -> Dict[str, float]:
        """Get a copy of all trust relationships."""
        return self.trust_relationships.copy()
    
    # Moral Choice History and Ethical Burden Methods
    
    def add_moral_choice(self, choice: MoralChoice, context: Optional[Dict[str, Any]] = None) -> None:
        """Add a moral choice to the agent's history and update ethical burden."""
        history_entry = {
            "choice": choice,
            "context": context or {},
            "timestamp": datetime.now()
        }
        self.moral_choice_history.append(history_entry)
        
        # Update ethical burden
        self.ethical_burden += choice.ethical_cost
        self._update_timestamp()
    
    def calculate_ethical_burden(self) -> float:
        """Calculate and return the total ethical burden from all moral choices."""
        total_burden = sum(entry["choice"].ethical_cost for entry in self.moral_choice_history)
        self.ethical_burden = total_burden
        return total_burden
    
    def get_moral_choice_count(self) -> int:
        """Get the total number of moral choices made by the agent."""
        return len(self.moral_choice_history)
    
    def get_recent_moral_choices(self, limit: int) -> List[Dict[str, Any]]:
        """Get the most recent moral choices (most recent first)."""
        # Return most recent choices first
        return list(reversed(self.moral_choice_history[-limit:]))
    
    # State Management and Synchronization Methods
    
    def sync_with_external_state(self, external_state: Dict[str, Any]) -> None:
        """Synchronize agent state with external system state."""
        if "resources_owned" in external_state:
            self.resources_owned = external_state["resources_owned"].copy()
        
        if "secrets_known" in external_state:
            self.secrets_known = external_state["secrets_known"].copy()
        
        if "trust_relationships" in external_state:
            self.trust_relationships = external_state["trust_relationships"].copy()
        
        if "ethical_burden" in external_state:
            self.ethical_burden = external_state["ethical_burden"]
        
        self._update_timestamp()
    
    def get_state_summary(self) -> Dict[str, Any]:
        """Get a comprehensive summary of the agent's current state."""
        return {
            "agent_id": self.agent_id,
            "resources_count": len(self.resources_owned),
            "secrets_count": len(self.secrets_known),
            "trust_relationships_count": len(self.trust_relationships),
            "moral_choices_count": len(self.moral_choice_history),
            "ethical_burden": self.ethical_burden,
            "last_updated": self.last_updated
        }
    
    def validate_state_consistency(self) -> Dict[str, Any]:
        """Validate the consistency of the agent's state."""
        errors = []
        
        # Check ethical burden is non-negative
        if self.ethical_burden < 0:
            errors.append("ethical_burden cannot be negative")
        
        # Check trust levels are within valid range
        for agent, trust in self.trust_relationships.items():
            if not -1.0 <= trust <= 1.0:
                errors.append(f"trust_level for {agent} out of range [-1.0, 1.0]: {trust}")
        
        # Check for duplicate resources
        if len(self.resources_owned) != len(set(self.resources_owned)):
            errors.append("duplicate resources found in resources_owned")
        
        # Check for duplicate secrets
        secret_ids = [secret.id for secret in self.secrets_known]
        if len(secret_ids) != len(set(secret_ids)):
            errors.append("duplicate secrets found in secrets_known")
        
        return {
            "is_valid": len(errors) == 0,
            "errors": errors
        }
    
    def reset_state(self) -> None:
        """Reset the agent state to initial empty state."""
        self.resources_owned.clear()
        self.secrets_known.clear()
        self.trust_relationships.clear()
        self.moral_choice_history.clear()
        self.ethical_burden = 0.0
        self._update_timestamp()
    
    def deep_copy(self) -> 'CompetitiveAgentState':
        """Create a deep copy of the agent state."""
        return copy.deepcopy(self)
    
    def _update_timestamp(self) -> None:
        """Update the last_updated timestamp."""
        self.last_updated = datetime.now()