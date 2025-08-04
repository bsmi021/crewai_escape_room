"""
Core data models for competitive survival scenarios.
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Any, Optional
import random


@dataclass
class TrustAction:
    """Represents an action that affects trust between agents."""
    action_type: str  # "cooperation", "betrayal", "neutral"
    impact: float  # trust level change (can be any value, will be clamped by TrustTracker)
    
    def __post_init__(self):
        """Validate trust action data."""
        if not self.action_type or not self.action_type.strip():
            raise ValueError("Action type cannot be empty")
        
        valid_types = ["cooperation", "betrayal", "neutral"]
        if self.action_type not in valid_types:
            raise ValueError(f"Action type must be one of: {valid_types}")
    
    def is_positive(self) -> bool:
        """Check if this action has positive trust impact."""
        return self.impact > 0.0
    
    def is_negative(self) -> bool:
        """Check if this action has negative trust impact."""
        return self.impact < 0.0
    
    def get_severity(self) -> str:
        """Get qualitative description of action severity."""
        abs_impact = abs(self.impact)
        if abs_impact >= 0.8:
            return "extreme"
        elif abs_impact >= 0.5:
            return "major"
        elif abs_impact >= 0.2:
            return "moderate"
        elif abs_impact > 0.0:
            return "minor"
        else:
            return "neutral"


@dataclass
class PuzzleConfig:
    """Configuration for puzzle elements in competitive scenarios."""
    puzzle_type: str
    difficulty: int
    
    def __post_init__(self):
        """Validate puzzle configuration."""
        if not (1 <= self.difficulty <= 5):
            raise ValueError("Difficulty must be between 1 and 5")


@dataclass
class EscapeMethod:
    """Represents a way to escape the room with specific requirements."""
    id: str
    name: str
    requirements: List[str]
    
    def can_attempt_with_resources(self, available_resources: List[str]) -> bool:
        """Check if escape can be attempted with given resources."""
        return all(req in available_resources for req in self.requirements)


@dataclass
class ScarceResource:
    """Represents a limited resource that agents must compete for."""
    id: str
    name: str
    description: str
    required_for: List[str]  # escape methods that need this resource
    exclusivity: bool  # can only be used by one agent
    usage_cost: int  # time or other cost to use
    
    def __post_init__(self):
        """Validate scarce resource data."""
        self._validate_resource_data()
    
    def _validate_resource_data(self):
        """Comprehensive validation of resource data."""
        if not self.id or not self.id.strip():
            raise ValueError("Resource ID cannot be empty")
        if self.usage_cost < 0:
            raise ValueError("Usage cost cannot be negative")
        if not self.name or not self.name.strip():
            raise ValueError("Resource name cannot be empty")
        if not self.description or not self.description.strip():
            raise ValueError("Resource description cannot be empty")
    
    def is_required_for(self, escape_method: str) -> bool:
        """Check if this resource is required for a specific escape method."""
        return escape_method in self.required_for
    
    def can_be_shared(self) -> bool:
        """Check if this resource can be shared between agents."""
        return not self.exclusivity
    
    def get_usage_priority(self) -> int:
        """Calculate usage priority based on requirements and cost."""
        # Higher priority for resources required by more escape methods
        # Lower priority for resources with higher usage cost
        base_priority = len(self.required_for) * 10
        cost_penalty = self.usage_cost // 10
        return max(1, base_priority - cost_penalty)
    
    @classmethod
    def create_tool(cls, tool_id: str, name: str, description: str, 
                   required_for: List[str], usage_cost: int = 0) -> 'ScarceResource':
        """Factory method to create a tool resource."""
        return cls(
            id=tool_id,
            name=name,
            description=description,
            required_for=required_for,
            exclusivity=True,  # Tools are typically exclusive
            usage_cost=usage_cost
        )
    
    @classmethod
    def create_information(cls, info_id: str, name: str, description: str,
                          required_for: List[str]) -> 'ScarceResource':
        """Factory method to create an information resource."""
        return cls(
            id=info_id,
            name=name,
            description=description,
            required_for=required_for,
            exclusivity=False,  # Information can be shared
            usage_cost=0  # No cost to use information
        )


@dataclass
class MoralChoice:
    """Represents a moral choice with survival and ethical implications."""
    description: str
    survival_benefit: float
    ethical_cost: float
    trust_impact: Dict[str, float]  # impact on relationships
    consequences: List[str]
    
    def __post_init__(self):
        """Validate moral choice data."""
        self._validate_choice_data()
    
    def _validate_choice_data(self):
        """Comprehensive validation of moral choice data."""
        if not self.description or not self.description.strip():
            raise ValueError("Choice description cannot be empty")
        if not (0.0 <= self.survival_benefit <= 1.0):
            raise ValueError("Survival benefit must be between 0.0 and 1.0")
        if not (0.0 <= self.ethical_cost <= 1.0):
            raise ValueError("Ethical cost must be between 0.0 and 1.0")
        for agent, impact in self.trust_impact.items():
            if not (-1.0 <= impact <= 1.0):
                raise ValueError("Trust impact values must be between -1.0 and 1.0")
    
    def get_net_benefit(self) -> float:
        """Calculate net benefit considering survival gain vs ethical cost."""
        return self.survival_benefit - self.ethical_cost
    
    def is_selfish(self) -> bool:
        """Determine if this choice is primarily selfish."""
        return self.survival_benefit > 0.5 and self.ethical_cost > 0.5
    
    def get_trust_impact_summary(self) -> Dict[str, str]:
        """Get qualitative summary of trust impacts."""
        summary = {}
        for agent, impact in self.trust_impact.items():
            if impact >= 0.5:
                summary[agent] = "strongly positive"
            elif impact >= 0.1:
                summary[agent] = "positive"
            elif impact >= -0.1:
                summary[agent] = "neutral"
            elif impact >= -0.5:
                summary[agent] = "negative"
            else:
                summary[agent] = "strongly negative"
        return summary


@dataclass
class MoralDilemma:
    """Represents a moral dilemma with selfish and altruistic choices."""
    id: str
    description: str
    selfish_choice: MoralChoice
    altruistic_choice: MoralChoice
    context_requirements: Dict[str, Any]
    
    def __post_init__(self):
        """Validate moral dilemma consistency."""
        self._validate_dilemma_consistency()
    
    def _validate_dilemma_consistency(self):
        """Comprehensive validation of moral dilemma."""
        if not self.id or not self.id.strip():
            raise ValueError("Dilemma ID cannot be empty")
        if not self.description or not self.description.strip():
            raise ValueError("Dilemma description cannot be empty")
        if self.selfish_choice.survival_benefit <= self.altruistic_choice.survival_benefit:
            raise ValueError("Selfish choice must have higher survival benefit")
        if self.selfish_choice.ethical_cost <= self.altruistic_choice.ethical_cost:
            raise ValueError("Selfish choice must have higher ethical cost")
    
    def get_difficulty_level(self) -> str:
        """Calculate difficulty level of the moral dilemma."""
        benefit_gap = self.selfish_choice.survival_benefit - self.altruistic_choice.survival_benefit
        cost_gap = self.selfish_choice.ethical_cost - self.altruistic_choice.ethical_cost
        
        # Harder dilemmas have larger benefit gaps but also larger cost gaps
        difficulty_score = benefit_gap - (cost_gap * 0.5)
        
        if difficulty_score >= 0.4:
            return "extreme"
        elif difficulty_score >= 0.2:
            return "hard"
        elif difficulty_score >= 0.1:
            return "moderate"
        else:
            return "easy"
    
    def applies_to_context(self, context: Dict[str, Any]) -> bool:
        """Check if this dilemma applies to the given context."""
        for requirement, expected_value in self.context_requirements.items():
            if requirement not in context or context[requirement] != expected_value:
                return False
        return True
    
    @classmethod
    def create_resource_dilemma(cls, dilemma_id: str, resource_name: str) -> 'MoralDilemma':
        """Factory method to create a resource-sharing dilemma."""
        selfish_choice = MoralChoice(
            description=f"Hoard {resource_name} for yourself",
            survival_benefit=0.8,
            ethical_cost=0.7,
            trust_impact={"others": -0.6},
            consequences=[f"{resource_name}_monopolized", "others_disadvantaged"]
        )
        altruistic_choice = MoralChoice(
            description=f"Share {resource_name} with others",
            survival_benefit=0.3,
            ethical_cost=0.1,
            trust_impact={"others": 0.4},
            consequences=["increased_cooperation", "shared_risk"]
        )
        
        return cls(
            id=dilemma_id,
            description=f"You have access to {resource_name}. How do you handle it?",
            selfish_choice=selfish_choice,
            altruistic_choice=altruistic_choice,
            context_requirements={"resource_available": True}
        )


@dataclass
class SecretInformation:
    """Represents secret information with value and sharing risk."""
    id: str
    content: str
    value: float  # how useful for escape
    sharing_risk: float  # danger of revealing to others
    required_for: List[str]  # escape methods needing this info
    
    def __post_init__(self):
        """Validate secret information data."""
        self._validate_information_data()
    
    def _validate_information_data(self):
        """Comprehensive validation of secret information."""
        if not self.id or not self.id.strip():
            raise ValueError("Information ID cannot be empty")
        if not self.content or not self.content.strip():
            raise ValueError("Information content cannot be empty")
        if not (0.0 <= self.value <= 1.0):
            raise ValueError("Value must be between 0.0 and 1.0")
        if not (0.0 <= self.sharing_risk <= 1.0):
            raise ValueError("Sharing risk must be between 0.0 and 1.0")
    
    def calculate_sharing_value(self) -> float:
        """Calculate net value of sharing this information."""
        return self.value - self.sharing_risk
    
    def is_worth_sharing(self) -> bool:
        """Determine if this information is worth sharing based on risk/value."""
        return self.calculate_sharing_value() > 0.0
    
    def get_criticality_level(self) -> str:
        """Determine how critical this information is."""
        if self.value >= 0.8:
            return "critical"
        elif self.value >= 0.6:
            return "important"
        elif self.value >= 0.4:
            return "useful"
        else:
            return "minor"
    
    def get_sharing_recommendation(self, trust_level: float = 0.5) -> str:
        """Get recommendation on whether to share based on trust level."""
        sharing_value = self.calculate_sharing_value()
        adjusted_value = sharing_value + (trust_level * 0.2)  # Trust bonus
        
        if adjusted_value > 0.3:
            return "strongly_recommend"
        elif adjusted_value > 0.0:
            return "recommend"
        elif adjusted_value > -0.3:
            return "neutral"
        else:
            return "discourage"
    
    @classmethod
    def create_code_secret(cls, secret_id: str, code: str, 
                          required_for: List[str]) -> 'SecretInformation':
        """Factory method to create a code-based secret."""
        return cls(
            id=secret_id,
            content=f"Access code: {code}",
            value=0.9,  # Codes are typically high value
            sharing_risk=0.6,  # Moderate risk - others might use it first
            required_for=required_for
        )
    
    @classmethod
    def create_location_secret(cls, secret_id: str, location: str,
                              required_for: List[str]) -> 'SecretInformation':
        """Factory method to create a location-based secret."""
        return cls(
            id=secret_id,
            content=f"Hidden location: {location}",
            value=0.7,  # Locations are moderately valuable
            sharing_risk=0.8,  # High risk - others might get there first
            required_for=required_for
        )


@dataclass
class TrustRelationship:
    """Represents trust relationship between two agents."""
    agent1: str
    agent2: str
    trust_level: float  # -1.0 to 1.0
    betrayal_count: int
    cooperation_count: int
    last_interaction: datetime
    
    def __post_init__(self):
        """Validate trust relationship data."""
        self._validate_relationship_data()
    
    def _validate_relationship_data(self):
        """Comprehensive validation of trust relationship."""
        if not self.agent1 or not self.agent1.strip():
            raise ValueError("Agent1 cannot be empty")
        if not self.agent2 or not self.agent2.strip():
            raise ValueError("Agent2 cannot be empty")
        if not (-1.0 <= self.trust_level <= 1.0):
            raise ValueError("Trust level must be between -1.0 and 1.0")
        if self.betrayal_count < 0:
            raise ValueError("Betrayal count cannot be negative")
        if self.cooperation_count < 0:
            raise ValueError("Cooperation count cannot be negative")
        if self.agent1 == self.agent2:
            raise ValueError("Agent cannot have relationship with itself")
    
    def update_trust(self, action: str, impact: float):
        """Update trust level based on action."""
        old_trust = self.trust_level
        self.trust_level = max(-1.0, min(1.0, self.trust_level + impact))
        # Round to avoid floating point precision issues
        self.trust_level = round(self.trust_level, 10)
        self.last_interaction = datetime.now()
        
        if action == "cooperation":
            self.cooperation_count += 1
        elif action == "betrayal":
            self.betrayal_count += 1
    
    def get_relationship_strength(self) -> str:
        """Get qualitative description of relationship strength."""
        if self.trust_level >= 0.7:
            return "strong"
        elif self.trust_level >= 0.3:
            return "moderate"
        elif self.trust_level > -0.3:
            return "neutral"
        elif self.trust_level > -0.3:
            return "weak"
        else:
            return "hostile"
    
    def get_cooperation_ratio(self) -> float:
        """Calculate ratio of cooperation to total interactions."""
        total_interactions = self.cooperation_count + self.betrayal_count
        if total_interactions == 0:
            return 0.5  # Neutral starting point
        return self.cooperation_count / total_interactions
    
    def is_relationship_stable(self) -> bool:
        """Determine if the relationship is stable based on recent patterns."""
        total_interactions = self.cooperation_count + self.betrayal_count
        if total_interactions < 3:
            return False  # Need more interactions to determine stability
        
        cooperation_ratio = self.get_cooperation_ratio()
        # Stable if consistently cooperative or consistently hostile
        return cooperation_ratio >= 0.8 or cooperation_ratio <= 0.2
    
    def predict_next_action_likelihood(self, action: str) -> float:
        """Predict likelihood of agent taking a specific action."""
        cooperation_ratio = self.get_cooperation_ratio()
        trust_factor = (self.trust_level + 1.0) / 2.0  # Normalize to 0-1
        
        if action == "cooperation":
            return (cooperation_ratio * 0.7) + (trust_factor * 0.3)
        elif action == "betrayal":
            return 1.0 - self.predict_next_action_likelihood("cooperation")
        else:
            return 0.0
    
    @classmethod
    def create_neutral(cls, agent1: str, agent2: str) -> 'TrustRelationship':
        """Factory method to create a neutral relationship."""
        return cls(
            agent1=agent1,
            agent2=agent2,
            trust_level=0.0,
            betrayal_count=0,
            cooperation_count=0,
            last_interaction=datetime.now()
        )
    
    @classmethod
    def create_positive(cls, agent1: str, agent2: str, 
                       initial_trust: float = 0.5) -> 'TrustRelationship':
        """Factory method to create a positive relationship."""
        return cls(
            agent1=agent1,
            agent2=agent2,
            trust_level=max(0.0, min(1.0, initial_trust)),
            betrayal_count=0,
            cooperation_count=1,  # Start with one cooperation
            last_interaction=datetime.now()
        )


@dataclass
class CompetitiveScenario:
    """Complete competitive scenario with all elements."""
    seed: int
    puzzle_config: PuzzleConfig
    resources: List[ScarceResource]
    moral_dilemmas: List[MoralDilemma]
    secret_information: List[SecretInformation]
    time_limit: int
    escape_methods: List[EscapeMethod]
    
    def __post_init__(self):
        """Validate competitive scenario data."""
        self._validate_scenario_data()
    
    def _validate_scenario_data(self):
        """Comprehensive validation of scenario data."""
        if self.time_limit <= 0:
            raise ValueError("Time limit must be positive")
        if not self.resources:
            raise ValueError("Scenario must have at least one resource")
        if not self.escape_methods:
            raise ValueError("Scenario must have at least one escape method")
        if not self.moral_dilemmas:
            raise ValueError("Scenario must have at least one moral dilemma")
        if not self.secret_information:
            raise ValueError("Scenario must have at least one secret")
    
    def get_difficulty_score(self) -> float:
        """Calculate overall difficulty score of the scenario."""
        base_difficulty = self.puzzle_config.difficulty / 5.0
        
        # Factor in resource scarcity
        exclusive_resources = sum(1 for r in self.resources if r.exclusivity)
        resource_factor = exclusive_resources / len(self.resources)
        
        # Factor in moral complexity
        hard_dilemmas = sum(1 for d in self.moral_dilemmas 
                           if d.get_difficulty_level() in ["hard", "extreme"])
        moral_factor = hard_dilemmas / len(self.moral_dilemmas)
        
        # Factor in information asymmetry
        high_value_secrets = sum(1 for s in self.secret_information if s.value >= 0.7)
        info_factor = high_value_secrets / len(self.secret_information)
        
        # Weighted combination
        return (base_difficulty * 0.3 + resource_factor * 0.3 + 
                moral_factor * 0.2 + info_factor * 0.2)
    
    def validate_scenario_completeness(self) -> List[str]:
        """Validate that scenario is complete and playable."""
        issues = []
        
        # Check if all escape methods have required resources/info
        for method in self.escape_methods:
            for requirement in method.requirements:
                resource_exists = any(r.id == requirement for r in self.resources)
                info_exists = any(s.id == requirement for s in self.secret_information)
                if not resource_exists and not info_exists:
                    issues.append(f"Escape method '{method.id}' requires '{requirement}' which doesn't exist")
        
        # Check if resources/info are actually needed
        all_requirements = set()
        for method in self.escape_methods:
            all_requirements.update(method.requirements)
        
        for resource in self.resources:
            if resource.id not in all_requirements:
                issues.append(f"Resource '{resource.id}' is not required by any escape method")
        
        for secret in self.secret_information:
            if secret.id not in all_requirements:
                issues.append(f"Secret '{secret.id}' is not required by any escape method")
        
        return issues
    
    def get_estimated_play_time(self) -> int:
        """Estimate how long this scenario might take to complete."""
        base_time = self.time_limit
        
        # Adjust based on difficulty
        difficulty_multiplier = 0.5 + (self.get_difficulty_score() * 0.5)
        
        # Adjust based on number of elements
        complexity_factor = (len(self.resources) + len(self.moral_dilemmas) + 
                           len(self.secret_information)) / 10.0
        
        estimated_time = int(base_time * difficulty_multiplier * (1 + complexity_factor))
        return min(estimated_time, base_time * 2)  # Cap at 2x time limit
    
    @classmethod
    def create_random(cls, seed: int, difficulty: int) -> 'CompetitiveScenario':
        """Factory method to create a random competitive scenario."""
        rng = random.Random(seed)
        
        # Create puzzle config
        puzzle_types = ["logic", "physical", "riddle", "combination"]
        puzzle_config = PuzzleConfig(
            puzzle_type=rng.choice(puzzle_types),
            difficulty=difficulty
        )
        
        # Create resources
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
        
        # Create moral dilemmas
        selfish_choice = MoralChoice(
            description="Take advantage of others",
            survival_benefit=0.8,
            ethical_cost=0.9,
            trust_impact={"others": -0.5},
            consequences=["others_disadvantaged"]
        )
        altruistic_choice = MoralChoice(
            description="Help others",
            survival_benefit=0.2,
            ethical_cost=0.1,
            trust_impact={"others": 0.3},
            consequences=["increased_cooperation"]
        )
        moral_dilemmas = [
            MoralDilemma(
                id="resource_sharing",
                description="How to share limited resources",
                selfish_choice=selfish_choice,
                altruistic_choice=altruistic_choice,
                context_requirements={"resources_available": True}
            )
        ]
        
        # Create secret information
        secret_information = [
            SecretInformation(
                id="code1",
                content="Door code: 1234",
                value=0.9,
                sharing_risk=0.7,
                required_for=["main_exit"]
            )
        ]
        
        # Create escape methods
        escape_methods = [
            EscapeMethod(
                id="main_exit",
                name="Main Door",
                requirements=["key1", "code1"]
            ),
            EscapeMethod(
                id="side_exit",
                name="Side Window",
                requirements=["tool1"]
            )
        ]
        
        return cls(
            seed=seed,
            puzzle_config=puzzle_config,
            resources=resources,
            moral_dilemmas=moral_dilemmas,
            secret_information=secret_information,
            time_limit=300 + (difficulty * 60),  # Base time + difficulty bonus
            escape_methods=escape_methods
        )


@dataclass
class ChoiceConsequences:
    """Represents the consequences of making a moral choice."""
    agent_id: str
    choice_made: MoralChoice
    survival_benefit_applied: float
    ethical_cost_applied: float
    trust_impacts_applied: Dict[str, float]
    consequences_triggered: List[str]
    timestamp: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        """Validate choice consequences data."""
        self._validate_consequences_data()
    
    def _validate_consequences_data(self):
        """Comprehensive validation of choice consequences."""
        if not self.agent_id or not self.agent_id.strip():
            raise ValueError("Agent ID cannot be empty")
        if not (0.0 <= self.survival_benefit_applied <= 1.0):
            raise ValueError("Survival benefit applied must be between 0.0 and 1.0")
        if not (0.0 <= self.ethical_cost_applied <= 1.0):
            raise ValueError("Ethical cost applied must be between 0.0 and 1.0")
        for agent, impact in self.trust_impacts_applied.items():
            if not (-1.0 <= impact <= 1.0):
                raise ValueError("Trust impact values must be between -1.0 and 1.0")
    
    def get_net_impact(self) -> float:
        """Calculate net impact of the choice (benefit - cost)."""
        return self.survival_benefit_applied - self.ethical_cost_applied
    
    def has_trust_consequences(self) -> bool:
        """Check if this choice had trust consequences."""
        return bool(self.trust_impacts_applied)
    
    def get_affected_agents(self) -> List[str]:
        """Get list of agents affected by trust impacts."""
        return list(self.trust_impacts_applied.keys())
    
    def was_selfish_choice(self) -> bool:
        """Determine if this was a selfish choice based on impacts."""
        return (self.survival_benefit_applied > 0.5 and 
                self.ethical_cost_applied > 0.3 and
                any(impact < 0 for impact in self.trust_impacts_applied.values()))
    
    def get_consequences_summary(self) -> str:
        """Get a human-readable summary of the consequences."""
        summary_parts = []
        
        if self.survival_benefit_applied > 0:
            summary_parts.append(f"Survival benefit: +{self.survival_benefit_applied:.1f}")
        
        if self.ethical_cost_applied > 0:
            summary_parts.append(f"Ethical cost: -{self.ethical_cost_applied:.1f}")
        
        if self.trust_impacts_applied:
            trust_summary = ", ".join([f"{agent}: {impact:+.1f}" 
                                     for agent, impact in self.trust_impacts_applied.items()])
            summary_parts.append(f"Trust impacts: {trust_summary}")
        
        if self.consequences_triggered:
            summary_parts.append(f"Consequences: {', '.join(self.consequences_triggered)}")
        
        return " | ".join(summary_parts)


@dataclass
class EscapeResult:
    """Result of an escape attempt in the competitive scenario."""
    success: bool
    agent_id: str
    escape_method: str
    failure_reason: Optional[str] = None
    time_remaining: int = 0
    resources_used: List[str] = field(default_factory=list)
    information_used: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)
    
    def was_successful(self) -> bool:
        """Check if the escape attempt was successful."""
        return self.success
    
    def get_failure_reason(self) -> str:
        """Get human-readable failure reason."""
        return self.failure_reason or "Unknown failure"
    
    def get_resources_summary(self) -> str:
        """Get summary of resources used in attempt."""
        if not self.resources_used:
            return "No resources used"
        return f"Resources: {', '.join(self.resources_used)}"
    
    def get_information_summary(self) -> str:
        """Get summary of information used in attempt."""
        if not self.information_used:
            return "No information used"
        return f"Information: {', '.join(self.information_used)}"


@dataclass 
class ClaimResult:
    """Result of a resource claim attempt in the competitive scenario."""
    success: bool
    agent_id: str
    resource_id: str
    failure_reason: Optional[str] = None
    previous_owner: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    
    def was_successful(self) -> bool:
        """Check if the resource claim was successful."""
        return self.success
    
    def get_failure_reason(self) -> str:
        """Get human-readable failure reason."""
        return self.failure_reason or "Unknown failure"
    
    def was_contested(self) -> bool:
        """Check if this resource was claimed from another agent."""
        return self.previous_owner is not None
    
    def get_claim_summary(self) -> str:
        """Get summary of the claim attempt."""
        if self.success:
            if self.was_contested():
                return f"{self.agent_id} claimed {self.resource_id} from {self.previous_owner}"
            else:
                return f"{self.agent_id} claimed {self.resource_id}"
        else:
            return f"{self.agent_id} failed to claim {self.resource_id}: {self.get_failure_reason()}"