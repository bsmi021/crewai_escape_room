"""
Handoff Protocols for Mesa-CrewAI Hybrid Architecture

Defines the data structures and protocols for passing information
between Agent A (Perception), Agent B (Decision), and Agent C (Action Translation).
"""

from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Dict, Any, List, Optional

from ..core_architecture import PerceptionData, DecisionData


@dataclass
class PerceptionHandoff:
    """
    Data structure for handoff from Agent A (Perception) to Agent B (Decision)
    
    Contains filtered perception data and performance metrics from the
    perception pipeline extraction process.
    """
    perceptions: Dict[str, PerceptionData]
    performance_metrics: Dict[str, float]
    extraction_timestamp: datetime
    validation_passed: bool
    mesa_model_hash: Optional[str]
    agent_count: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        result = asdict(self)
        # Convert PerceptionData objects to dicts
        result["perceptions"] = {
            agent_id: asdict(perception) 
            for agent_id, perception in self.perceptions.items()
        }
        # Convert datetime to ISO string
        result["extraction_timestamp"] = self.extraction_timestamp.isoformat()
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PerceptionHandoff':
        """Create from dictionary (deserialization)"""
        # Convert perception dicts back to PerceptionData objects
        perceptions = {
            agent_id: PerceptionData(**perception_dict)
            for agent_id, perception_dict in data["perceptions"].items()
        }
        
        # Convert ISO string back to datetime
        extraction_timestamp = datetime.fromisoformat(data["extraction_timestamp"])
        
        return cls(
            perceptions=perceptions,
            performance_metrics=data["performance_metrics"],
            extraction_timestamp=extraction_timestamp,
            validation_passed=data["validation_passed"],
            mesa_model_hash=data.get("mesa_model_hash"),
            agent_count=data["agent_count"]
        )


@dataclass
class DecisionHandoff:
    """
    Data structure for handoff from Agent B (Decision) to Agent C (Action Translation)
    
    Contains decision data, confidence scores, negotiation outcomes, and metadata
    needed for action translation.
    """
    decisions: Dict[str, DecisionData]
    reasoning_confidence: Dict[str, float]
    decision_timestamp: datetime
    llm_response_time: float
    negotiation_outcomes: Dict[str, Any]
    fallback_decisions_used: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        result = asdict(self)
        # Convert DecisionData objects to dicts
        result["decisions"] = {
            agent_id: asdict(decision)
            for agent_id, decision in self.decisions.items()
        }
        # Convert datetime to ISO string
        result["decision_timestamp"] = self.decision_timestamp.isoformat()
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DecisionHandoff':
        """Create from dictionary (deserialization)"""
        # Convert decision dicts back to DecisionData objects
        decisions = {
            agent_id: DecisionData(**decision_dict)
            for agent_id, decision_dict in data["decisions"].items()
        }
        
        # Convert ISO string back to datetime
        decision_timestamp = datetime.fromisoformat(data["decision_timestamp"])
        
        return cls(
            decisions=decisions,
            reasoning_confidence=data["reasoning_confidence"],
            decision_timestamp=decision_timestamp,
            llm_response_time=data["llm_response_time"],
            negotiation_outcomes=data["negotiation_outcomes"],
            fallback_decisions_used=data["fallback_decisions_used"]
        )
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary for monitoring"""
        return {
            "total_decisions": len(self.decisions),
            "average_confidence": sum(self.reasoning_confidence.values()) / len(self.reasoning_confidence) if self.reasoning_confidence else 0.0,
            "llm_response_time": self.llm_response_time,
            "fallback_rate": len(self.fallback_decisions_used) / len(self.decisions) if self.decisions else 0.0,
            "negotiation_success": len(self.negotiation_outcomes) > 0,
            "decision_timestamp": self.decision_timestamp.isoformat()
        }
    
    def validate_handoff(self) -> bool:
        """Validate handoff data integrity"""
        try:
            # Check that we have decisions
            if not self.decisions:
                return False
            
            # Check that confidence scores match decisions
            decision_agents = set(self.decisions.keys())
            confidence_agents = set(self.reasoning_confidence.keys())
            if decision_agents != confidence_agents:
                return False
            
            # Check confidence values are in valid range
            for confidence in self.reasoning_confidence.values():
                if not (0.0 <= confidence <= 1.0):
                    return False
            
            # Check that all decisions have required fields
            for decision in self.decisions.values():
                if not decision.agent_id or not decision.chosen_action:
                    return False
                if not (0.0 <= decision.confidence_level <= 1.0):
                    return False
            
            # Check fallback agents exist in decisions
            for fallback_agent in self.fallback_decisions_used:
                if fallback_agent not in self.decisions:
                    return False
            
            return True
            
        except Exception:
            return False


class HandoffValidator:
    """Validates handoff protocols between agents"""
    
    @staticmethod
    def validate_perception_handoff(handoff: PerceptionHandoff) -> Dict[str, Any]:
        """Validate perception handoff from Agent A"""
        validation_result = {
            "valid": True,
            "errors": [],
            "warnings": []
        }
        
        try:
            # Check required fields
            if not handoff.perceptions:
                validation_result["errors"].append("No perceptions provided")
                validation_result["valid"] = False
            
            if handoff.agent_count != len(handoff.perceptions):
                validation_result["errors"].append(
                    f"Agent count mismatch: expected {handoff.agent_count}, got {len(handoff.perceptions)}"
                )
                validation_result["valid"] = False
            
            # Check perception data quality
            for agent_id, perception in handoff.perceptions.items():
                if not perception.agent_id:
                    validation_result["errors"].append(f"Missing agent_id in perception for {agent_id}")
                    validation_result["valid"] = False
                
                if perception.agent_id != agent_id:
                    validation_result["errors"].append(
                        f"Agent ID mismatch: key={agent_id}, perception.agent_id={perception.agent_id}"
                    )
                    validation_result["valid"] = False
                
                if not perception.available_actions:
                    validation_result["warnings"].append(f"No available actions for {agent_id}")
            
            # Check performance metrics
            required_metrics = ["extraction_time", "validation_score"]
            for metric in required_metrics:
                if metric not in handoff.performance_metrics:
                    validation_result["warnings"].append(f"Missing performance metric: {metric}")
            
            # Check timestamp is recent
            time_diff = (datetime.now() - handoff.extraction_timestamp).total_seconds()
            if time_diff > 60:  # More than 1 minute old
                validation_result["warnings"].append(f"Stale perception data: {time_diff:.1f}s old")
            
        except Exception as e:
            validation_result["errors"].append(f"Validation error: {str(e)}")
            validation_result["valid"] = False
        
        return validation_result
    
    @staticmethod
    def validate_decision_handoff(handoff: DecisionHandoff) -> Dict[str, Any]:
        """Validate decision handoff from Agent B"""
        validation_result = {
            "valid": True,
            "errors": [],
            "warnings": []
        }
        
        try:
            # Use built-in validation
            if not handoff.validate_handoff():
                validation_result["errors"].append("Basic handoff validation failed")
                validation_result["valid"] = False
            
            # Check LLM response time
            if handoff.llm_response_time > 10.0:
                validation_result["warnings"].append(
                    f"Slow LLM response: {handoff.llm_response_time:.2f}s"
                )
            
            # Check fallback usage rate
            fallback_rate = len(handoff.fallback_decisions_used) / len(handoff.decisions) if handoff.decisions else 0
            if fallback_rate > 0.5:
                validation_result["warnings"].append(
                    f"High fallback usage: {fallback_rate:.1%} of decisions"
                )
            
            # Check confidence levels
            low_confidence_agents = [
                agent_id for agent_id, confidence in handoff.reasoning_confidence.items()
                if confidence < 0.5
            ]
            if low_confidence_agents:
                validation_result["warnings"].append(
                    f"Low confidence decisions: {', '.join(low_confidence_agents)}"
                )
            
            # Check decision timestamp
            time_diff = (datetime.now() - handoff.decision_timestamp).total_seconds()
            if time_diff > 30:  # More than 30 seconds old
                validation_result["warnings"].append(f"Stale decision data: {time_diff:.1f}s old")
        
        except Exception as e:
            validation_result["errors"].append(f"Validation error: {str(e)}")
            validation_result["valid"] = False
        
        return validation_result