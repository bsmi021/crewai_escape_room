"""
TrustTracker for managing relationship dynamics in competitive scenarios.
Tracks trust levels, betrayal/cooperation history, and reputation calculations.
"""
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from .models import TrustRelationship, TrustAction


class TrustTracker:
    """Manages trust relationships and reputation tracking between agents."""
    
    def __init__(self, agents: Optional[List[str]] = None):
        """Initialize TrustTracker with optional agent list."""
        self.trust_matrix: Dict[Tuple[str, str], TrustRelationship] = {}
        self.betrayal_history: List[Dict] = []
        self.cooperation_history: List[Dict] = []
        
        if agents is not None:
            self._validate_agent_list(agents)
            self._initialize_relationships(agents)
    
    def _validate_agent_list(self, agents: List[str]):
        """Validate the agent list for initialization."""
        if not agents:
            raise ValueError("Agent list cannot be empty")
        
        if len(set(agents)) != len(agents):
            raise ValueError("Duplicate agents not allowed")
        
        for agent in agents:
            if not agent or not agent.strip():
                raise ValueError("Agent names cannot be empty")
    
    def _initialize_relationships(self, agents: List[str]):
        """Initialize neutral relationships between all agent pairs."""
        for i, agent1 in enumerate(agents):
            for j, agent2 in enumerate(agents):
                if i != j:
                    relationship = TrustRelationship.create_neutral(agent1, agent2)
                    self.trust_matrix[(agent1, agent2)] = relationship
    
    def update_trust(self, actor: str, target: str, action: TrustAction):
        """Update trust level between two agents based on an action."""
        self._validate_agents(actor, target)
        
        # Get or create relationship
        relationship_key = (actor, target)
        if relationship_key not in self.trust_matrix:
            self.trust_matrix[relationship_key] = TrustRelationship.create_neutral(actor, target)
        
        relationship = self.trust_matrix[relationship_key]
        
        # Update the relationship
        relationship.update_trust(action.action_type, action.impact)
        
        # Record in history
        self._record_action_history(actor, target, action)
    
    def _validate_agents(self, actor: str, target: str):
        """Validate agent parameters."""
        if not actor or not actor.strip():
            raise ValueError("Actor cannot be empty")
        if not target or not target.strip():
            raise ValueError("Target cannot be empty")
        if actor == target:
            raise ValueError("Agent cannot have relationship with itself")
    
    def _record_action_history(self, actor: str, target: str, action: TrustAction):
        """Record action in appropriate history list."""
        record = {
            "actor": actor,
            "target": target,
            "impact": action.impact,
            "timestamp": datetime.now()
        }
        
        if action.action_type == "cooperation":
            self.cooperation_history.append(record)
        elif action.action_type == "betrayal":
            self.betrayal_history.append(record)
        # Neutral actions are not recorded in history
    
    def get_trust_level(self, agent1: str, agent2: str) -> float:
        """Get trust level from agent1 to agent2."""
        self._validate_trust_query(agent1, agent2)
        
        relationship_key = (agent1, agent2)
        if relationship_key in self.trust_matrix:
            return self.trust_matrix[relationship_key].trust_level
        else:
            return 0.0  # Neutral trust for non-existent relationships
    
    def _validate_trust_query(self, agent1: str, agent2: str):
        """Validate parameters for trust level queries."""
        if not agent1 or not agent1.strip():
            raise ValueError("Agent1 cannot be empty")
        if not agent2 or not agent2.strip():
            raise ValueError("Agent2 cannot be empty")
        if agent1 == agent2:
            raise ValueError("Agent cannot query relationship with itself")
    
    def calculate_reputation(self, agent_id: str) -> float:
        """Calculate overall reputation based on incoming trust from all other agents."""
        if not agent_id or not agent_id.strip():
            raise ValueError("Agent cannot be empty")
        
        incoming_trust_values = []
        
        # Find all relationships where this agent is the target
        for (actor, target), relationship in self.trust_matrix.items():
            if target == agent_id:
                incoming_trust_values.append(relationship.trust_level)
        
        if not incoming_trust_values:
            return 0.0  # Neutral reputation if no relationships
        
        return sum(incoming_trust_values) / len(incoming_trust_values)
    
    def get_agent_betrayal_count(self, agent_id: str) -> int:
        """Get total number of betrayal actions performed by an agent."""
        return sum(1 for record in self.betrayal_history if record["actor"] == agent_id)
    
    def get_agent_cooperation_count(self, agent_id: str) -> int:
        """Get total number of cooperation actions performed by an agent."""
        return sum(1 for record in self.cooperation_history if record["actor"] == agent_id)
    
    def get_most_trusted_agent(self, agent_id: str) -> Optional[str]:
        """Get the agent that the specified agent trusts the most."""
        if not agent_id or not agent_id.strip():
            raise ValueError("Agent cannot be empty")
        
        max_trust = -2.0  # Below minimum possible trust
        most_trusted = None
        
        for (actor, target), relationship in self.trust_matrix.items():
            if actor == agent_id and relationship.trust_level > max_trust:
                max_trust = relationship.trust_level
                most_trusted = target
        
        return most_trusted
    
    def get_least_trusted_agent(self, agent_id: str) -> Optional[str]:
        """Get the agent that the specified agent trusts the least."""
        if not agent_id or not agent_id.strip():
            raise ValueError("Agent cannot be empty")
        
        min_trust = 2.0  # Above maximum possible trust
        least_trusted = None
        
        for (actor, target), relationship in self.trust_matrix.items():
            if actor == agent_id and relationship.trust_level < min_trust:
                min_trust = relationship.trust_level
                least_trusted = target
        
        return least_trusted
    
    def get_agent_trustworthiness_score(self, agent_id: str) -> float:
        """Calculate how trustworthy an agent is based on their actions."""
        if not agent_id or not agent_id.strip():
            raise ValueError("Agent cannot be empty")
        
        cooperation_count = self.get_agent_cooperation_count(agent_id)
        betrayal_count = self.get_agent_betrayal_count(agent_id)
        total_actions = cooperation_count + betrayal_count
        
        if total_actions == 0:
            return 0.0  # Neutral if no actions
        
        # Calculate trustworthiness as cooperation ratio
        return cooperation_count / total_actions
    
    def get_mutual_trust_level(self, agent1: str, agent2: str) -> float:
        """Get mutual trust level between two agents (average of both directions)."""
        self._validate_trust_query(agent1, agent2)
        
        trust_1_to_2 = self.get_trust_level(agent1, agent2)
        trust_2_to_1 = self.get_trust_level(agent2, agent1)
        
        return (trust_1_to_2 + trust_2_to_1) / 2.0
    
    def get_all_agents(self) -> List[str]:
        """Get list of all agents in the trust matrix."""
        agents = set()
        for (actor, target) in self.trust_matrix.keys():
            agents.add(actor)
            agents.add(target)
        return sorted(list(agents))
    
    def get_trust_summary(self, agent_id: str) -> Dict[str, float]:
        """Get a summary of all trust relationships for an agent."""
        if not agent_id or not agent_id.strip():
            raise ValueError("Agent cannot be empty")
        
        summary = {}
        
        # Get outgoing trust (who this agent trusts)
        for (actor, target), relationship in self.trust_matrix.items():
            if actor == agent_id:
                summary[f"trusts_{target}"] = relationship.trust_level
        
        # Add reputation (how others view this agent)
        summary["reputation"] = self.calculate_reputation(agent_id)
        summary["trustworthiness"] = self.get_agent_trustworthiness_score(agent_id)
        
        return summary