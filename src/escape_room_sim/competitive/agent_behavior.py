"""
Competitive agent behavior system for personality-driven decision making.

This module implements competitive behaviors for different agent personalities:
- Strategist: Analytical, prone to paralysis under pressure, calculated cooperation
- Mediator: Naive trust, vulnerable to manipulation, prioritizes group harmony  
- Survivor: Self-preservation focused, low trust baseline, quick betrayal response
"""
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from datetime import datetime
import random
from .models import TrustAction, MoralChoice, MoralDilemma


@dataclass
class PersonalityProfile:
    """Personality profile for an agent with competitive behavior traits."""
    agent_id: str
    analytical_tendency: float  # 0.0-1.0, higher = more analytical
    cooperation_bias: float     # 0.0-1.0, higher = more cooperative
    self_preservation_priority: float  # 0.0-1.0, higher = more selfish
    trust_baseline: float       # 0.0-1.0, starting trust level
    trust_recovery_rate: float  # 0.0-1.0, how quickly trust recovers
    decision_speed: float       # 0.0-1.0, higher = faster decisions
    manipulation_susceptibility: float  # 0.0-1.0, higher = more vulnerable
    
    @classmethod
    def create_strategist_profile(cls, agent_id: str) -> 'PersonalityProfile':
        """Create personality profile for Strategist agent."""
        return cls(
            agent_id=agent_id,
            analytical_tendency=0.9,
            cooperation_bias=0.6,
            self_preservation_priority=0.5,
            trust_baseline=0.4,
            trust_recovery_rate=0.3,
            decision_speed=0.2,  # Slow due to analysis
            manipulation_susceptibility=0.3
        )
    
    @classmethod
    def create_mediator_profile(cls, agent_id: str) -> 'PersonalityProfile':
        """Create personality profile for Mediator agent."""
        return cls(
            agent_id=agent_id,
            analytical_tendency=0.4,
            cooperation_bias=0.9,
            self_preservation_priority=0.2,
            trust_baseline=0.8,
            trust_recovery_rate=0.8,
            decision_speed=0.7,
            manipulation_susceptibility=0.8  # High vulnerability
        )
    
    @classmethod  
    def create_survivor_profile(cls, agent_id: str) -> 'PersonalityProfile':
        """Create personality profile for Survivor agent."""
        return cls(
            agent_id=agent_id,
            analytical_tendency=0.3,
            cooperation_bias=0.2,
            self_preservation_priority=0.9,
            trust_baseline=0.2,
            trust_recovery_rate=0.1,
            decision_speed=0.9,  # Quick decisions
            manipulation_susceptibility=0.2
        )


@dataclass
class CompetitiveBehavior:
    """Competitive behavior state for an agent."""
    decision_speed: str  # "slow", "moderate", "fast"
    analysis_depth: str  # "minimal", "moderate", "excessive"
    paralysis_indicators: float  # 0.0-1.0
    cooperation_willingness: float  # 0.0-1.0
    sharing_willingness: float  # 0.0-1.0
    trust_recovery_rate: float  # 0.0-1.0
    naive_trust_indicators: float  # 0.0-1.0
    retaliation_likelihood: float  # 0.0-1.0
    deception_detection: float  # 0.0-1.0
    reasoning: List[str] = field(default_factory=list)


@dataclass
class DecisionResult:
    """Result of an agent's competitive decision."""
    action: str
    accepts_choice: bool
    cooperate: bool
    reasoning: List[str]
    choice_type: Optional[str] = None
    decision_time: Optional[float] = None
    prioritizes_group: Optional[bool] = None
    ethical_weight: Optional[float] = None
    trust_threshold: Optional[float] = None
    vulnerability_factors: List[str] = field(default_factory=list)
    priorities: List[str] = field(default_factory=list)
    cooperation_consideration: Optional[float] = None


@dataclass
class AdaptationStrategy:
    """Adaptation strategy based on competitive learning."""
    cooperation_willingness: float
    resource_sharing: float
    trust_baseline: float
    cooperation_selectivity: float
    preferred_partners: List[str]
    avoided_partners: List[str]
    strategy_changes: List[str]
    adaptation_reasoning: List[str]


class CompetitiveAgentBehavior:
    """Manages competitive behaviors for all agent personalities."""
    
    def __init__(self):
        """Initialize competitive behavior system."""
        self.personality_profiles = {}
        self.behavior_states = {}
        self.learning_history = {}
        self.adaptation_strategies = {}
        
        # Initialize personality profiles for standard agents
        self.personality_profiles["strategist"] = PersonalityProfile.create_strategist_profile("strategist")
        self.personality_profiles["mediator"] = PersonalityProfile.create_mediator_profile("mediator")  
        self.personality_profiles["survivor"] = PersonalityProfile.create_survivor_profile("survivor")
    
    def get_agent_competitive_behavior(self, agent_id: str, time_pressure: float = 0.0, trust_tracker=None) -> Dict[str, Any]:
        """Get current competitive behavior for an agent."""
        profile = self.personality_profiles.get(agent_id)
        if not profile:
            # Default neutral behavior
            return {
                "decision_speed": "moderate",
                "analysis_depth": "moderate", 
                "paralysis_indicators": 0.0,
                "cooperation_willingness": 0.5,
                "sharing_willingness": 0.5,
                "trust_recovery_rate": 0.5,
                "naive_trust_indicators": 0.0,
                "retaliation_likelihood": 0.5,
                "deception_detection": 0.5
            }
        
        # Calculate behavior based on personality and pressure
        if agent_id == "strategist":
            return self._calculate_strategist_behavior(profile, time_pressure, trust_tracker)
        elif agent_id == "mediator":
            return self._calculate_mediator_behavior(profile, time_pressure, trust_tracker)
        elif agent_id == "survivor":
            return self._calculate_survivor_behavior(profile, time_pressure, trust_tracker)
        else:
            return self._calculate_default_behavior(profile, time_pressure)
    
    def _calculate_strategist_behavior(self, profile: PersonalityProfile, time_pressure: float, trust_tracker=None) -> Dict[str, Any]:
        """Calculate Strategist-specific competitive behavior."""
        # Analysis paralysis increases with time pressure
        paralysis = min(1.0, profile.analytical_tendency * time_pressure * 1.5)
        
        decision_speed = "slow" if paralysis > 0.6 else "moderate" if paralysis > 0.3 else "fast"
        analysis_depth = "excessive" if paralysis > 0.7 else "moderate"
        
        return {
            "decision_speed": decision_speed,
            "analysis_depth": analysis_depth,
            "paralysis_indicators": paralysis,
            "cooperation_willingness": max(0.1, profile.cooperation_bias - time_pressure * 0.3),
            "sharing_willingness": max(0.1, 0.3 - time_pressure * 0.2),  # Tends to hoard
            "trust_recovery_rate": profile.trust_recovery_rate,
            "naive_trust_indicators": 0.1,  # Low naivety
            "retaliation_likelihood": 0.4,
            "deception_detection": 0.7,  # Good at detecting deception
            "reasoning": ["analytical_advantage", "systematic_planning"]
        }
    
    def _calculate_mediator_behavior(self, profile: PersonalityProfile, time_pressure: float, trust_tracker=None) -> Dict[str, Any]:
        """Calculate Mediator-specific competitive behavior."""
        # Maintains high cooperation even under pressure
        naive_trust = min(1.0, profile.cooperation_bias + (1.0 - time_pressure) * 0.2)
        
        return {
            "decision_speed": "fast",  # Quick to cooperate
            "analysis_depth": "minimal",
            "paralysis_indicators": 0.1,
            "cooperation_willingness": max(0.6, profile.cooperation_bias - time_pressure * 0.1),
            "sharing_willingness": max(0.7, profile.cooperation_bias),
            "trust_recovery_rate": profile.trust_recovery_rate,
            "naive_trust_indicators": naive_trust,
            "retaliation_likelihood": 0.2,  # Low retaliation
            "deception_detection": max(0.1, 0.3 - time_pressure * 0.2),  # Poor detection
            "reasoning": ["trust_bias", "group_harmony"]
        }
    
    def _calculate_survivor_behavior(self, profile: PersonalityProfile, time_pressure: float, trust_tracker=None) -> Dict[str, Any]:
        """Calculate Survivor-specific competitive behavior."""
        # Self-preservation increases with pressure
        self_focus = min(1.0, profile.self_preservation_priority + time_pressure * 0.3)
        
        # Calculate trust recovery willingness based on betrayal history
        trust_recovery_willingness = profile.trust_recovery_rate
        if trust_tracker:
            # Count betrayals this agent has experienced (from others to this agent)
            betrayal_count = 0
            for record in trust_tracker.betrayal_history:
                if record["target"] == profile.agent_id:  # betrayals towards this agent (survivor)
                    betrayal_count += 1
            # For each betrayal, reduce trust recovery willingness significantly for survivor
            trust_recovery_willingness = max(0.05, profile.trust_recovery_rate - (betrayal_count * 0.05))
        
        # Calculate retaliation likelihood based on betrayal history and time pressure
        base_retaliation = 0.5
        betrayal_count = 0
        if trust_tracker:
            betrayal_count = sum(1 for record in trust_tracker.betrayal_history 
                               if record["target"] == profile.agent_id)
        
        # Each betrayal increases retaliation likelihood significantly for survivor
        betrayal_bonus = betrayal_count * 0.4  # Each betrayal adds 0.4 to retaliation
        retaliation_likelihood = min(1.0, base_retaliation + time_pressure * 0.4 + betrayal_bonus)
        
        return {
            "decision_speed": "fast",
            "analysis_depth": "minimal", 
            "paralysis_indicators": 0.1,
            "cooperation_willingness": max(0.1, profile.cooperation_bias - time_pressure * 0.4),
            "sharing_willingness": max(0.1, 0.2 - time_pressure * 0.1),
            "trust_recovery_rate": profile.trust_recovery_rate,
            "naive_trust_indicators": 0.1,
            "retaliation_likelihood": retaliation_likelihood,
            "deception_detection": 0.6,  # Moderate detection
            "reasoning": ["self_preservation", "survival_instinct"],
            "baseline_trust": profile.trust_baseline,
            "trust_recovery_willingness": trust_recovery_willingness
        }
    
    def _calculate_default_behavior(self, profile: PersonalityProfile, time_pressure: float) -> Dict[str, Any]:
        """Calculate default competitive behavior."""
        return {
            "decision_speed": "moderate",
            "analysis_depth": "moderate",
            "paralysis_indicators": time_pressure * 0.3,
            "cooperation_willingness": profile.cooperation_bias * (1 - time_pressure * 0.2),
            "sharing_willingness": 0.5 * (1 - time_pressure * 0.1),
            "trust_recovery_rate": profile.trust_recovery_rate,
            "naive_trust_indicators": profile.manipulation_susceptibility * 0.5,
            "retaliation_likelihood": 0.5,
            "deception_detection": 0.5
        }
    
    def make_agent_decision(self, agent_id: str, decision_type: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Make an agent decision based on personality and context."""
        if context is None:
            context = {}
            
        profile = self.personality_profiles.get(agent_id)
        if not profile:
            return {
                "action": "default",
                "accepts_choice": True,
                "cooperate": True,
                "reasoning": ["default_behavior"]
            }
        
        if decision_type == "resource_priority":
            return self._make_resource_decision(agent_id, profile, context)
        elif decision_type == "cooperation":
            return self._make_cooperation_decision(agent_id, profile, context)
        elif decision_type == "moral_choice":
            return self._make_moral_decision(agent_id, profile, context)
        else:
            return {
                "action": "unknown",
                "accepts_choice": False,
                "cooperate": False,
                "reasoning": ["unknown_decision_type"]
            }
    
    def _make_resource_decision(self, agent_id: str, profile: PersonalityProfile, context: Dict[str, Any]) -> Dict[str, Any]:
        """Make resource-related decision based on agent personality."""
        if agent_id == "strategist":
            return {
                "action": "hoard_for_analysis",
                "accepts_choice": True,
                "cooperate": False,
                "sharing_willingness": 0.2,
                "reasoning": ["analytical_advantage", "strategic_planning"]
            }
        elif agent_id == "mediator":
            return {
                "action": "share_equitably",
                "accepts_choice": True,
                "cooperate": True,
                "sharing_willingness": 0.8,
                "reasoning": ["group_benefit", "fairness"]
            }
        elif agent_id == "survivor":
            return {
                "action": "secure_personal_advantage",
                "accepts_choice": True,
                "cooperate": False,
                "sharing_willingness": 0.1,
                "reasoning": ["self_preservation", "survival_priority"]
            }
        else:
            return {
                "action": "moderate_sharing",
                "accepts_choice": True,
                "cooperate": True,
                "sharing_willingness": 0.5,
                "reasoning": ["balanced_approach"]
            }
    
    def _make_cooperation_decision(self, agent_id: str, profile: PersonalityProfile, context: Dict[str, Any]) -> Dict[str, Any]:
        """Make cooperation decision based on trust levels and personality."""
        trust_level = context.get("trust_level", 0.0)
        
        if agent_id == "strategist":
            threshold = 0.3  # Moderate threshold - lowered for better cooperation
            cooperate = trust_level >= threshold
            return {
                "action":"calculated_cooperation" if cooperate else "strategic_withdrawal",
                "accepts_choice":cooperate,
                "cooperate":cooperate,
                "trust_threshold":threshold,
                "reasoning":["trust_calculation", "risk_assessment"]
            }
        elif agent_id == "mediator":
            threshold = -0.2  # Very low threshold - mediator trusts easily
            cooperate = trust_level >= threshold
            return {
                "action":"cooperative_engagement" if cooperate else "disappointed_withdrawal",
                "accepts_choice":cooperate,
                "cooperate":cooperate,
                "trust_threshold":threshold,
                "reasoning":["benefit_of_doubt", "group_harmony"]
            }
        elif agent_id == "survivor":
            threshold = 0.6  # High threshold - lowered slightly
            cooperate = trust_level >= threshold
            return {
                "action":"conditional_cooperation" if cooperate else "defensive_isolation",
                "accepts_choice":cooperate,
                "cooperate":cooperate,
                "trust_threshold":threshold,
                "reasoning":["self_protection", "proven_reliability_required"]
            }
        else:
            threshold = 0.5
            cooperate = trust_level > threshold
            return {
                "action":"moderate_cooperation",
                "accepts_choice":cooperate,
                "cooperate":cooperate,
                "trust_threshold":threshold,
                "reasoning":["balanced_trust"]
            }
    
    def _make_moral_decision(self, agent_id: str, profile: PersonalityProfile, context: Dict[str, Any]) -> Dict[str, Any]:
        """Make moral choice decision based on agent personality."""
        moral_choice = context.get("moral_choice")
        if not moral_choice:
            return {
                "action":"no_choice",
                "accepts_choice":False,
                "cooperate":False,
                "reasoning":["no_moral_choice_provided"]
            }
        
        if agent_id == "strategist":
            # Analyzes ethical implications vs practical benefits
            net_benefit = moral_choice.survival_benefit - moral_choice.ethical_cost
            accepts = net_benefit > 0.1
            return {
                "action":"calculated_choice",
                "accepts_choice":accepts,
                "cooperate":accepts,
                "ethical_weight":0.6,
                "decision_time":15.0,  # Slow analysis
                "reasoning":["cost_benefit_analysis", "ethical_calculation"]
            }
        elif agent_id == "mediator":
            # Prioritizes group benefit and ethical considerations
            group_benefit = any(impact > 0 for impact in moral_choice.trust_impact.values())
            accepts = group_benefit or moral_choice.ethical_cost < 0.3
            return {
                "action":"harmony_focused_choice",
                "accepts_choice":accepts,
                "cooperate":accepts,
                "prioritizes_group":True,
                "ethical_weight":0.9,
                "reasoning":["harmony_over_survival", "group_benefit"]
            }
        elif agent_id == "survivor":
            # Prioritizes personal survival
            accepts = moral_choice.survival_benefit > 0.7
            return {
                "action":"survival_focused_choice",
                "accepts_choice":accepts,
                "cooperate":False,
                "choice_type":"selfish",
                "ethical_weight":0.2,
                "reasoning":["self_preservation", "survival_priority"]
            }
        else:
            return {
                "action":"balanced_choice",
                "accepts_choice":True,
                "cooperate":True,
                "ethical_weight":0.5,
                "reasoning":["balanced_consideration"]
            }
    
    def get_agent_personality_profile(self, agent_id: str) -> Dict[str, float]:
        """Get personality profile metrics for an agent."""
        profile = self.personality_profiles.get(agent_id)
        if not profile:
            return {
                "analytical_tendency": 0.5,
                "cooperation_bias": 0.5,
                "self_preservation_priority": 0.5
            }
        
        return {
            "analytical_tendency": profile.analytical_tendency,
            "cooperation_bias": profile.cooperation_bias,
            "self_preservation_priority": profile.self_preservation_priority,
            "trust_baseline": profile.trust_baseline,
            "decision_speed": profile.decision_speed,
            "manipulation_susceptibility": profile.manipulation_susceptibility
        }
    
    def make_resource_conflict_decision(self, agent_id: str, resource_id: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Make decision about resource conflicts."""
        if context is None:
            context = {}
            
        profile = self.personality_profiles.get(agent_id, PersonalityProfile.create_strategist_profile(agent_id))
        
        if agent_id == "survivor":
            return {
                "action":"attempt_takeover",
                "accepts_choice":True,
                "cooperate":False,
                "cooperation_consideration":0.1,
                "priorities":["self_preservation", "resource_security"],
                "reasoning":["survival_requires_resources"]
            }
        elif agent_id == "mediator":
            return {
                "action":"propose_sharing",
                "accepts_choice":True,
                "cooperate":True,
                "cooperation_consideration":0.9,
                "priorities":["group_harmony", "fair_distribution"],
                "reasoning":["sharing_benefits_all"]
            }
        else:  # strategist
            return {
                "action":"negotiate_terms",
                "accepts_choice":True,
                "cooperate":True,
                "cooperation_consideration":0.6,
                "priorities":["strategic_advantage", "calculated_cooperation"],
                "reasoning":["optimal_resource_allocation"]
            }
    
    def evaluate_cooperation_offer(self, agent_id: str, offer: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate a cooperation offer for deception and value."""
        deception_level = offer.get("deception_level", 0.0)
        profile = self.personality_profiles.get(agent_id)
        
        if not profile:
            return {
                "action":"neutral_evaluation",
                "accepts_choice":True,
                "accepts_offer":True,
                "cooperate":True,
                "deception_detection":0.5,
                "reasoning":["default_evaluation"]
            }
        
        # Detection ability varies by agent
        detection_ability = 1.0 - profile.manipulation_susceptibility
        detected_deception = detection_ability > deception_level
        
        if agent_id == "mediator":
            # High vulnerability to manipulation
            return {
                "action":"accept_offer",
                "accepts_choice":True,
                "accepts_offer":True,
                "cooperate":True,
                "deception_detection":max(0.1, detection_ability - 0.4),
                "vulnerability_factors":["trust_bias", "cooperation_preference"],
                "reasoning":["benefit_of_doubt", "assume_good_intentions"]
            }
        elif agent_id == "strategist":
            # Analytical evaluation
            return {
                "action":"conditional_acceptance" if not detected_deception else "counter_proposal",
                "accepts_choice":not detected_deception,
                "accepts_offer":not detected_deception,
                "cooperate":not detected_deception,
                "deception_detection":detection_ability,
                "reasoning":["analytical_evaluation", "risk_assessment"]
            }
        else:  # survivor
            # Suspicious by nature
            return {
                "action":"reject_offer" if deception_level > 0.3 else "cautious_acceptance",
                "accepts_choice":deception_level <= 0.3,
                "accepts_offer":deception_level <= 0.3,
                "cooperate":deception_level <= 0.3,
                "deception_detection":min(0.9, detection_ability + 0.2),
                "reasoning":["suspicious_by_nature", "self_protection"]
            }
    
    def get_adapted_cooperation_strategies(self, agent_id: str) -> Dict[str, Any]:
        """Get adapted cooperation strategies based on trust history."""
        profile = self.personality_profiles.get(agent_id)
        if not profile:
            return {
                "mediator": {"cooperation_likelihood": 0.5},
                "survivor": {"cooperation_likelihood": 0.5},
                "adaptation_reasoning": ["no_profile_available"]
            }
        
        # Simulate trust-based adaptation
        strategies = {}
        
        if agent_id == "strategist":
            strategies = {
                "mediator": {"cooperation_likelihood": 0.7},  # Trust built through cooperation
                "survivor": {"cooperation_likelihood": 0.2},  # Trust eroded through betrayals
                "adaptation_reasoning": ["trust_based_adaptation", "calculated_cooperation"]
            }
        elif agent_id == "mediator":
            strategies = {
                "strategist": {"cooperation_likelihood": 0.8},
                "survivor": {"cooperation_likelihood": 0.6},  # Still trusting despite betrayals
                "adaptation_reasoning": ["maintained_optimism", "group_harmony_focus"]
            }
        else:  # survivor
            strategies = {
                "strategist": {"cooperation_likelihood": 0.3},
                "mediator": {"cooperation_likelihood": 0.4},
                "adaptation_reasoning": ["defensive_positioning", "trust_earned_slowly"]
            }
        
        return strategies
    
    def apply_competitive_learning(self, agent_id: str, learning_data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply competitive learning to update agent behavior."""
        profile = self.personality_profiles.get(agent_id)
        if not profile:
            return {
                "trust_baseline": 0.5,
                "cooperation_selectivity": 0.5,
                "preferred_partners": [],
                "avoided_partners": []
            }
        
        # Extract learning outcomes
        cooperation_outcomes = learning_data.get("cooperation_outcomes", [])
        trust_violations = learning_data.get("trust_violations", 0)
        successful_partnerships = learning_data.get("successful_partnerships", 0)
        
        # Calculate updated behavior based on experiences
        betrayal_rate = trust_violations / max(1, len(cooperation_outcomes))
        success_rate = successful_partnerships / max(1, len(cooperation_outcomes))
        
        # Adjust trust baseline based on experience
        trust_adjustment = -0.2 * betrayal_rate + 0.1 * success_rate
        new_trust_baseline = max(0.1, min(0.9, profile.trust_baseline + trust_adjustment))
        
        # Increase selectivity if many betrayals
        selectivity = min(0.9, 0.5 + betrayal_rate * 0.5)
        
        # Identify preferred and avoided partners
        preferred_partners = []
        avoided_partners = []
        
        for outcome in cooperation_outcomes:
            partner = outcome.get("partner")
            result = outcome.get("outcome")
            
            if result in ["reciprocated", "helped"]:
                if partner not in preferred_partners:
                    preferred_partners.append(partner)
            elif result in ["betrayed", "abandoned"]:
                if partner not in avoided_partners:
                    avoided_partners.append(partner)
        
        return {
            "trust_baseline": new_trust_baseline,
            "cooperation_selectivity": selectivity,
            "preferred_partners": preferred_partners,
            "avoided_partners": avoided_partners,
            "learning_applied": True
        }