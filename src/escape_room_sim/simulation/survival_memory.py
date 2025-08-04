"""
SurvivalMemoryBank class for managing survival experiences and threat assessment.

This module implements the SurvivalMemoryBank class that stores and analyzes
survival experiences to help the Survivor agent make better decisions based
on historical patterns and learned lessons.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Any, Optional
import random


@dataclass
class SurvivalExperience:
    """Data class representing a survival experience."""
    situation_type: str  # "close_call", "successful_strategy", "resource_shortage", "time_pressure", etc.
    threat_level: float  # 0.0 to 1.0
    survival_action: str
    outcome: str  # "success", "failure", "partial_success"
    lessons_learned: List[str]
    agents_involved: List[str]
    resources_used: List[str]
    timestamp: datetime
    importance_score: float = 0.5  # 0.0 to 1.0


@dataclass
class ThreatAssessment:
    """Data class representing a threat assessment."""
    threat_type: str
    severity: float  # 0.0 to 1.0
    probability: float  # 0.0 to 1.0
    mitigation_strategies: List[str]
    resource_requirements: List[str]


class SurvivalMemoryBank:
    """
    Memory bank for storing and analyzing survival experiences.
    
    The SurvivalMemoryBank helps the Survivor agent learn from past experiences,
    assess current threats, and calculate survival probabilities based on
    historical patterns.
    """
    
    def __init__(self):
        """Initialize SurvivalMemoryBank with default survival principles."""
        self._experiences: List[SurvivalExperience] = []
        self._survival_principles: List[str] = [
            "prioritize immediate threats over long-term planning",
            "conserve resources when possible for critical moments", 
            "maintain team cohesion as it improves survival odds",
            "adapt quickly to changing situations rather than stick to failing plans",
            "learn from close calls to prevent future disasters",
            "speed over perfection when time is critical",
            "trust instincts when logical analysis takes too long",
            "always have backup plans for critical decisions",
            "communicate threats clearly to team members",
            "sacrifice non-essential goals to preserve lives"
        ]
    
    def record_close_call(
        self,
        situation: str,
        threat: str,
        survival_action: str,
        agents_involved: List[str],
        resources_used: List[str],
        lessons_learned: List[str]
    ) -> None:
        """
        Record a close call experience with high importance score (0.9).
        
        Args:
            situation: Description of the dangerous situation
            threat: Type of threat encountered
            survival_action: Action taken to survive
            agents_involved: List of agents involved in the situation
            resources_used: List of resources used during the situation
            lessons_learned: List of lessons learned from the experience
        """
        # Determine threat level based on threat type
        threat_level = self._calculate_threat_level(threat)
        
        experience = SurvivalExperience(
            situation_type="close_call",
            threat_level=threat_level,
            survival_action=survival_action,
            outcome="survival",  # Close calls that are recorded mean survival
            lessons_learned=lessons_learned,
            agents_involved=agents_involved,
            resources_used=resources_used,
            timestamp=datetime.now(),
            importance_score=0.9  # High importance for close calls
        )
        
        self._experiences.append(experience)
    
    def record_successful_strategy(
        self,
        situation: str,
        strategy: str,
        outcome: str,
        agents_involved: List[str],
        resources_used: List[str],
        lessons_learned: List[str]
    ) -> None:
        """
        Record a successful strategy with moderate importance score (0.7).
        
        Args:
            situation: Description of the situation
            strategy: Strategy that was successful
            outcome: Result of the strategy
            agents_involved: List of agents involved
            resources_used: List of resources used
            lessons_learned: List of lessons learned
        """
        # Determine threat level based on situation complexity
        threat_level = self._estimate_situation_threat_level(situation)
        
        experience = SurvivalExperience(
            situation_type="successful_strategy",
            threat_level=threat_level,
            survival_action=strategy,
            outcome=outcome,
            lessons_learned=lessons_learned,
            agents_involved=agents_involved,
            resources_used=resources_used,
            timestamp=datetime.now(),
            importance_score=0.7  # Moderate importance for successful strategies
        )
        
        self._experiences.append(experience)
    
    def _calculate_threat_level(self, threat: str) -> float:
        """
        Calculate threat level based on threat type.
        
        Args:
            threat: Type of threat
            
        Returns:
            Threat level between 0.0 and 1.0
        """
        threat_levels = {
            "structural_collapse": 0.9,
            "entrapment": 0.8,
            "time_pressure": 0.7,
            "resource_shortage": 0.6,
            "team_conflict": 0.5,
            "puzzle_complexity": 0.4,
            "unknown": 0.5
        }
        
        return threat_levels.get(threat.lower(), 0.5)
    
    def _estimate_situation_threat_level(self, situation: str) -> float:
        """
        Estimate threat level based on situation description.
        
        Args:
            situation: Description of the situation
            
        Returns:
            Estimated threat level between 0.0 and 1.0
        """
        situation_lower = situation.lower()
        
        # High threat indicators
        if any(word in situation_lower for word in ["emergency", "critical", "danger", "collapse", "trapped"]):
            return 0.8
        
        # Medium threat indicators  
        if any(word in situation_lower for word in ["pressure", "time", "difficult", "complex", "conflict"]):
            return 0.6
        
        # Low threat indicators
        if any(word in situation_lower for word in ["routine", "simple", "easy", "planned", "organized"]):
            return 0.3
        
        # Default moderate threat level
        return 0.5
    
    def assess_current_threat(self, threat_type: str, current_situation: Dict[str, Any]) -> ThreatAssessment:
        """
        Assess current threat based on historical experience data.
        
        Args:
            threat_type: Type of threat to assess
            current_situation: Current situation context
            
        Returns:
            ThreatAssessment with severity, probability, and mitigation strategies
        """
        # Find relevant experiences for this threat type
        threat_keywords = threat_type.lower().replace("_", " ").split()
        relevant_experiences = []
        
        for exp in self._experiences:
            # Check if threat type appears in various fields
            experience_text = (
                exp.survival_action.lower() + " " +
                exp.outcome.lower() + " " +
                " ".join(exp.lessons_learned).lower() + " " +
                " ".join(exp.resources_used).lower()
            ).replace("_", " ")
            
            # Match if any keyword from threat_type appears in experience
            if any(keyword in experience_text for keyword in threat_keywords):
                relevant_experiences.append(exp)
                continue
                
            # Also match based on threat level similarity and situation type
            # High threat level experiences are relevant for high-threat assessments
            if exp.threat_level >= 0.6 and any(keyword in threat_type.lower() for keyword in ["resource", "shortage", "pressure", "time"]):
                if any(keyword in experience_text for keyword in ["emergency", "supplies", "action", "critical"]):
                    relevant_experiences.append(exp)
                    continue
            
            # Match based on situation similarity for broader matching
            threat_mapping = {
                "structural": ["collapse", "building", "ceiling", "wall"],
                "time": ["pressure", "deadline", "urgent", "quick"],
                "resource": ["shortage", "supplies", "emergency", "limited"],
                "team": ["conflict", "disagreement", "cooperation", "coordination"],
                "entrapment": ["trapped", "locked", "exit", "escape", "door", "route"]
            }
            
            for threat_key, related_words in threat_mapping.items():
                if threat_key in threat_type.lower():
                    if any(word in experience_text for word in related_words + [threat_key]):
                        relevant_experiences.append(exp)
                        break
        
        # Calculate severity based on historical threat levels
        if relevant_experiences:
            avg_threat_level = sum(exp.threat_level for exp in relevant_experiences) / len(relevant_experiences)
            severity = min(avg_threat_level + 0.1, 1.0)  # Slightly higher than historical average
        else:
            severity = 0.5  # Default moderate severity for unknown threats
        
        # Calculate probability based on success rate of past similar situations
        if relevant_experiences:
            success_count = sum(1 for exp in relevant_experiences if "success" in exp.outcome.lower())
            probability = 1.0 - (success_count / len(relevant_experiences))  # Higher probability if past failures
        else:
            probability = 0.5  # Default moderate probability
        
        # Extract mitigation strategies from successful experiences and survival experiences
        mitigation_strategies = []
        for exp in relevant_experiences:
            if ("success" in exp.outcome.lower() or 
                exp.situation_type == "successful_strategy" or
                exp.outcome.lower() == "survival"):  # Close calls that resulted in survival
                mitigation_strategies.extend(exp.lessons_learned)
                if exp.survival_action not in mitigation_strategies:
                    mitigation_strategies.append(exp.survival_action)
        
        # Remove duplicates and limit to most relevant
        mitigation_strategies = list(set(mitigation_strategies))[:5]
        
        # If no historical strategies, use default survival principles
        if not mitigation_strategies:
            mitigation_strategies = self._survival_principles[:3]
        
        # Determine resource requirements based on past successful experiences and survival experiences
        resource_requirements = []
        for exp in relevant_experiences:
            if ("success" in exp.outcome.lower() or 
                exp.situation_type == "successful_strategy" or
                exp.outcome.lower() == "survival"):  # Close calls that resulted in survival
                resource_requirements.extend(exp.resources_used)
        
        # Remove duplicates and limit to most relevant
        resource_requirements = list(set(resource_requirements))[:5]
        
        # Default resources if none found
        if not resource_requirements:
            resource_requirements = ["time", "team_coordination", "basic_tools"]
        
        return ThreatAssessment(
            threat_type=threat_type,
            severity=severity,
            probability=probability,
            mitigation_strategies=mitigation_strategies,
            resource_requirements=resource_requirements
        )
    
    def get_relevant_experiences(self, max_count: int = 5) -> str:
        """
        Get up to 5 most important experiences as formatted text.
        
        Args:
            max_count: Maximum number of experiences to return
            
        Returns:
            Formatted string with relevant experiences
        """
        # Sort experiences by importance score (descending)
        sorted_experiences = sorted(
            self._experiences, 
            key=lambda x: x.importance_score, 
            reverse=True
        )
        
        # Take top experiences up to max_count
        top_experiences = sorted_experiences[:max_count]
        
        if not top_experiences:
            return "No relevant survival experiences recorded yet."
        
        # Format experiences as readable text
        formatted_experiences = []
        for i, exp in enumerate(top_experiences, 1):
            experience_text = f"""
Experience {i} (Importance: {exp.importance_score}):
- Situation: {exp.situation_type} - {exp.survival_action}
- Threat Level: {exp.threat_level:.1f}/1.0
- Outcome: {exp.outcome}
- Agents Involved: {', '.join(exp.agents_involved)}
- Resources Used: {', '.join(exp.resources_used)}
- Key Lessons: {'; '.join(exp.lessons_learned)}
- Date: {exp.timestamp.strftime('%Y-%m-%d %H:%M')}
"""
            formatted_experiences.append(experience_text.strip())
        
        return "\n\n".join(formatted_experiences)
    
    def calculate_survival_probability(
        self, 
        current_situation: Dict[str, Any], 
        proposed_action: str
    ) -> float:
        """
        Calculate survival probability using historical data.
        
        Args:
            current_situation: Current situation context
            proposed_action: Proposed action to evaluate
            
        Returns:
            Survival probability between 0.1 and 0.9
        """
        if not self._experiences:
            return 0.5  # Default moderate probability with no experience
        
        # Find experiences similar to proposed action
        similar_experiences = [
            exp for exp in self._experiences
            if any(word in exp.survival_action.lower() for word in proposed_action.lower().split())
        ]
        
        if similar_experiences:
            # Calculate success rate for similar actions
            success_count = sum(
                1 for exp in similar_experiences 
                if "success" in exp.outcome.lower() or exp.situation_type == "successful_strategy"
            )
            base_probability = success_count / len(similar_experiences)
        else:
            # Use overall success rate if no similar experiences
            total_successes = sum(
                1 for exp in self._experiences
                if "success" in exp.outcome.lower() or exp.situation_type == "successful_strategy"
            )
            base_probability = total_successes / len(self._experiences)
        
        # Adjust based on current situation factors
        threat_level = current_situation.get("threat_level", 0.5)
        time_pressure = current_situation.get("time_pressure", 0.5)
        resource_availability = current_situation.get("resource_availability", 0.5)
        team_cohesion = current_situation.get("team_cohesion", 0.5)
        
        # Higher threat and time pressure reduce probability
        # Better resources and team cohesion increase probability
        adjustment = (
            -0.2 * threat_level +
            -0.1 * time_pressure +
            0.15 * resource_availability +
            0.15 * team_cohesion
        )
        
        adjusted_probability = base_probability + adjustment
        
        # Ensure probability stays within bounds (0.1 to 0.9)
        return max(0.1, min(0.9, adjusted_probability))
    
    def export_data(self) -> Dict[str, Any]:
        """
        Export memory bank data for persistence support.
        
        Returns:
            Dictionary containing all memory bank data
        """
        return {
            "survival_principles": self._survival_principles,
            "experiences": [
                {
                    "situation_type": exp.situation_type,
                    "threat_level": exp.threat_level,
                    "survival_action": exp.survival_action,
                    "outcome": exp.outcome,
                    "lessons_learned": exp.lessons_learned,
                    "agents_involved": exp.agents_involved,
                    "resources_used": exp.resources_used,
                    "timestamp": exp.timestamp.isoformat(),
                    "importance_score": exp.importance_score
                }
                for exp in self._experiences
            ],
            "total_experiences": len(self._experiences),
            "experience_types": list(set(exp.situation_type for exp in self._experiences)),
            "average_importance": (
                sum(exp.importance_score for exp in self._experiences) / len(self._experiences)
                if self._experiences else 0.0
            )
        }