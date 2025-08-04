"""
RelationshipTracker class for monitoring agent interactions and trust levels.

This module provides functionality to track relationships between agents,
record interactions, and calculate team cohesion metrics.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any


@dataclass
class RelationshipEntry:
    """Represents a single interaction between two agents."""
    agent_a: str
    agent_b: str
    interaction_type: str  # "collaboration", "conflict", "support", "disagreement"
    context: str
    outcome: str  # "positive", "negative", "neutral"
    trust_impact: float  # -1.0 to 1.0
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class AgentRelationship:
    """Represents the relationship state between two agents."""
    agent_a: str
    agent_b: str
    trust_level: float = 0.5  # 0.0 to 1.0, starts neutral
    collaboration_count: int = 0
    conflict_count: int = 0
    last_interaction: Optional[datetime] = None
    interaction_history: List[RelationshipEntry] = field(default_factory=list)
    
    def __post_init__(self):
        """Ensure agents are in alphabetical order for consistency."""
        if self.agent_a > self.agent_b:
            self.agent_a, self.agent_b = self.agent_b, self.agent_a


class RelationshipTracker:
    """
    Tracks relationships and interactions between agents in the escape room simulation.
    
    This class maintains trust levels, interaction history, and provides methods
    for recording different types of agent interactions.
    """
    
    def __init__(self):
        """Initialize empty relationship tracking system."""
        self._relationships: Dict[str, AgentRelationship] = {}
    
    def _get_relationship_key(self, agent_a: str, agent_b: str) -> str:
        """
        Generate standardized relationship key with alphabetical ordering.
        
        Args:
            agent_a: First agent name
            agent_b: Second agent name
            
        Returns:
            Standardized key string for relationship lookup
        """
        if agent_a == agent_b:
            raise ValueError("Cannot create relationship between same agent")
        
        # Ensure alphabetical ordering for consistency
        if agent_a > agent_b:
            agent_a, agent_b = agent_b, agent_a
        
        return f"{agent_a}:{agent_b}"
    
    def get_relationship(self, agent_a: str, agent_b: str) -> AgentRelationship:
        """
        Get or create relationship between two agents.
        
        Args:
            agent_a: First agent name
            agent_b: Second agent name
            
        Returns:
            AgentRelationship object for the agent pair
            
        Raises:
            ValueError: If trying to get relationship between same agent
        """
        key = self._get_relationship_key(agent_a, agent_b)
        
        if key not in self._relationships:
            # Create new relationship with alphabetical ordering
            if agent_a > agent_b:
                agent_a, agent_b = agent_b, agent_a
            
            self._relationships[key] = AgentRelationship(
                agent_a=agent_a,
                agent_b=agent_b
            )
        
        return self._relationships[key]
    
    def record_interaction(self, agent_a: str, agent_b: str, interaction_type: str, 
                          context: str, outcome: str, trust_impact: float) -> None:
        """
        Record a general interaction between two agents.
        
        Args:
            agent_a: First agent name
            agent_b: Second agent name
            interaction_type: Type of interaction
            context: Description of the interaction context
            outcome: Result of the interaction
            trust_impact: Impact on trust level (-1.0 to 1.0)
        """
        relationship = self.get_relationship(agent_a, agent_b)
        
        # Create interaction entry
        entry = RelationshipEntry(
            agent_a=agent_a,
            agent_b=agent_b,
            interaction_type=interaction_type,
            context=context,
            outcome=outcome,
            trust_impact=trust_impact
        )
        
        # Update relationship
        relationship.interaction_history.append(entry)
        relationship.last_interaction = entry.timestamp
        
        # Update trust level within bounds (round to avoid floating point precision issues)
        new_trust = relationship.trust_level + trust_impact
        relationship.trust_level = round(max(0.0, min(1.0, new_trust)), 10)
        
        # Update counters
        if outcome == "positive":
            relationship.collaboration_count += 1
        elif outcome == "negative":
            relationship.conflict_count += 1
    
    def record_successful_collaboration(self, agents: List[str], strategy: str, outcome: str) -> None:
        """
        Record successful collaboration between multiple agents.
        
        Args:
            agents: List of agent names involved
            strategy: Description of the collaborative strategy
            outcome: Result of the collaboration
        """
        trust_increase = 0.1
        
        # Record pairwise collaborations
        for i in range(len(agents)):
            for j in range(i + 1, len(agents)):
                self.record_interaction(
                    agent_a=agents[i],
                    agent_b=agents[j],
                    interaction_type="collaboration",
                    context=f"Collaborative strategy: {strategy}",
                    outcome="positive",
                    trust_impact=trust_increase
                )
    
    def record_conflict(self, agent_a: str, agent_b: str, conflict_reason: str, resolution: str) -> None:
        """
        Record conflict between two agents.
        
        Args:
            agent_a: First agent name
            agent_b: Second agent name
            conflict_reason: Reason for the conflict
            resolution: How the conflict was resolved
        """
        # Trust impact depends on resolution quality
        resolution_lower = resolution.lower()
        # Check for unresolved indicators first (more specific)
        if ("unresolved" in resolution_lower or "failed" in resolution_lower or
            "no resolution" in resolution_lower or "tension" in resolution_lower):
            trust_impact = -0.1   # Larger impact if explicitly unresolved
        elif ("resolved" in resolution_lower or "compromise" in resolution_lower or 
              "solution" in resolution_lower or "mediated" in resolution_lower):
            trust_impact = -0.05  # Minor impact if resolved well
        else:
            trust_impact = -0.1   # Default to larger impact if unclear
        
        self.record_interaction(
            agent_a=agent_a,
            agent_b=agent_b,
            interaction_type="conflict",
            context=f"Conflict: {conflict_reason}. Resolution: {resolution}",
            outcome="negative",
            trust_impact=trust_impact
        )
    
    def get_team_cohesion(self, agents: List[str]) -> float:
        """
        Calculate team cohesion based on average trust levels.
        
        Args:
            agents: List of agent names to analyze
            
        Returns:
            Team cohesion score between 0.0 and 1.0
        """
        if len(agents) < 2:
            return 1.0  # Single agent has perfect cohesion
        
        total_trust = 0.0
        relationship_count = 0
        
        # Calculate average trust across all agent pairs
        for i in range(len(agents)):
            for j in range(i + 1, len(agents)):
                try:
                    relationship = self.get_relationship(agents[i], agents[j])
                    total_trust += relationship.trust_level
                    relationship_count += 1
                except ValueError:
                    # Skip if same agent (shouldn't happen with proper loop)
                    continue
        
        if relationship_count == 0:
            return 0.5  # Neutral cohesion if no relationships
        
        return total_trust / relationship_count
    
    def get_summary(self) -> str:
        """
        Get readable summary of current relationship states.
        
        Returns:
            Formatted string describing all relationships
        """
        if not self._relationships:
            return "No relationships tracked yet."
        
        summary_lines = ["Relationship Summary:"]
        
        for key, relationship in self._relationships.items():
            trust_desc = self._get_trust_description(relationship.trust_level)
            summary_lines.append(
                f"  {relationship.agent_a} ↔ {relationship.agent_b}: "
                f"{trust_desc} (Trust: {relationship.trust_level:.2f}, "
                f"Collaborations: {relationship.collaboration_count}, "
                f"Conflicts: {relationship.conflict_count})"
            )
        
        return "\n".join(summary_lines)
    
    def _get_trust_description(self, trust_level: float) -> str:
        """Get descriptive text for trust level."""
        if trust_level >= 0.8:
            return "Strong Trust"
        elif trust_level >= 0.6:
            return "Good Trust"
        elif trust_level >= 0.4:
            return "Neutral"
        elif trust_level >= 0.2:
            return "Low Trust"
        else:
            return "Distrust"
    
    def export_data(self) -> Dict[str, Any]:
        """
        Export relationship data for persistence.
        
        Returns:
            Dictionary containing all relationship data
        """
        export_data = {
            "relationships": {},
            "export_timestamp": datetime.now().isoformat(),
            "total_relationships": len(self._relationships)
        }
        
        for key, relationship in self._relationships.items():
            export_data["relationships"][key] = {
                "agent_a": relationship.agent_a,
                "agent_b": relationship.agent_b,
                "trust_level": relationship.trust_level,
                "collaboration_count": relationship.collaboration_count,
                "conflict_count": relationship.conflict_count,
                "last_interaction": relationship.last_interaction.isoformat() if relationship.last_interaction else None,
                "interaction_count": len(relationship.interaction_history),
                "interaction_history": [
                    {
                        "agent_a": entry.agent_a,
                        "agent_b": entry.agent_b,
                        "interaction_type": entry.interaction_type,
                        "context": entry.context,
                        "outcome": entry.outcome,
                        "trust_impact": entry.trust_impact,
                        "timestamp": entry.timestamp.isoformat()
                    }
                    for entry in relationship.interaction_history
                ]
            }
        
        return export_data