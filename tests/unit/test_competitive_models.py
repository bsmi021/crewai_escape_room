"""
Test module for competitive survival data models.
Following TDD approach - these tests should fail initially.
"""
import pytest
from datetime import datetime
from typing import Dict, List, Any

# Import statements that will fail initially since models don't exist yet
try:
    from src.escape_room_sim.competitive.models import (
        CompetitiveScenario,
        ScarceResource,
        MoralDilemma,
        MoralChoice,
        SecretInformation,
        TrustRelationship,
        PuzzleConfig,
        EscapeMethod
    )
except ImportError:
    # These will be implemented after tests are written
    CompetitiveScenario = None
    ScarceResource = None
    MoralDilemma = None
    MoralChoice = None
    SecretInformation = None
    TrustRelationship = None
    PuzzleConfig = None
    EscapeMethod = None


class TestCompetitiveScenario:
    """Test CompetitiveScenario dataclass validation and factory methods."""
    
    def test_competitive_scenario_creation(self):
        """Test basic CompetitiveScenario creation with required fields."""
        puzzle_config = PuzzleConfig(puzzle_type="logic", difficulty=3)
        resources = [ScarceResource(id="key1", name="Master Key", description="Opens main door", 
                                  required_for=["main_exit"], exclusivity=True, usage_cost=0)]
        dilemmas = [MoralDilemma(id="betrayal1", description="Betray ally for advantage", 
                               selfish_choice=MoralChoice(description="Take advantage", survival_benefit=0.8, 
                                                        ethical_cost=0.9, trust_impact={"ally": -0.5}, consequences=[]),
                               altruistic_choice=MoralChoice(description="Help ally", survival_benefit=0.2, 
                                                           ethical_cost=0.1, trust_impact={"ally": 0.3}, consequences=[]),
                               context_requirements={})]
        secrets = [SecretInformation(id="code1", content="Door code: 1234", value=0.9, 
                                   sharing_risk=0.7, required_for=["main_exit"])]
        escape_methods = [EscapeMethod(id="main_exit", name="Main Door", requirements=["key1", "code1"])]
        
        scenario = CompetitiveScenario(
            seed=12345,
            puzzle_config=puzzle_config,
            resources=resources,
            moral_dilemmas=dilemmas,
            secret_information=secrets,
            time_limit=300,
            escape_methods=escape_methods
        )
        
        assert scenario.seed == 12345
        assert scenario.time_limit == 300
        assert len(scenario.resources) == 1
        assert len(scenario.moral_dilemmas) == 1
        assert len(scenario.secret_information) == 1
        assert len(scenario.escape_methods) == 1
    
    def test_competitive_scenario_validation_invalid_time_limit(self):
        """Test that CompetitiveScenario validates time_limit > 0."""
        with pytest.raises(ValueError, match="Time limit must be positive"):
            CompetitiveScenario(
                seed=12345,
                puzzle_config=PuzzleConfig(puzzle_type="logic", difficulty=3),
                resources=[],
                moral_dilemmas=[],
                secret_information=[],
                time_limit=-100,  # Invalid negative time
                escape_methods=[]
            )
    
    def test_competitive_scenario_factory_method(self):
        """Test CompetitiveScenario.create_random factory method."""
        scenario = CompetitiveScenario.create_random(seed=54321, difficulty=2)
        
        assert scenario.seed == 54321
        assert scenario.time_limit > 0
        assert len(scenario.resources) >= 1
        assert len(scenario.moral_dilemmas) >= 1
        assert len(scenario.secret_information) >= 1
        assert len(scenario.escape_methods) >= 1


class TestScarceResource:
    """Test ScarceResource dataclass with exclusivity and usage rules."""
    
    def test_scarce_resource_creation(self):
        """Test basic ScarceResource creation."""
        resource = ScarceResource(
            id="lockpick",
            name="Lockpick Set",
            description="Can open locked containers",
            required_for=["chest_escape", "door_bypass"],
            exclusivity=True,
            usage_cost=30
        )
        
        assert resource.id == "lockpick"
        assert resource.name == "Lockpick Set"
        assert resource.exclusivity is True
        assert resource.usage_cost == 30
        assert "chest_escape" in resource.required_for
        assert "door_bypass" in resource.required_for
    
    def test_scarce_resource_validation_negative_usage_cost(self):
        """Test that ScarceResource validates usage_cost >= 0."""
        with pytest.raises(ValueError, match="Usage cost cannot be negative"):
            ScarceResource(
                id="invalid",
                name="Invalid Resource",
                description="Test resource",
                required_for=[],
                exclusivity=False,
                usage_cost=-10  # Invalid negative cost
            )
    
    def test_scarce_resource_validation_empty_id(self):
        """Test that ScarceResource validates non-empty id."""
        with pytest.raises(ValueError, match="Resource ID cannot be empty"):
            ScarceResource(
                id="",  # Invalid empty ID
                name="Valid Name",
                description="Valid description",
                required_for=[],
                exclusivity=False,
                usage_cost=0
            )
    
    def test_scarce_resource_is_required_for_method(self):
        """Test ScarceResource.is_required_for method."""
        resource = ScarceResource(
            id="key",
            name="Key",
            description="Opens doors",
            required_for=["main_exit", "side_exit"],
            exclusivity=True,
            usage_cost=0
        )
        
        assert resource.is_required_for("main_exit") is True
        assert resource.is_required_for("side_exit") is True
        assert resource.is_required_for("window_exit") is False


class TestMoralDilemma:
    """Test MoralDilemma and MoralChoice dataclasses with consequence validation."""
    
    def test_moral_choice_creation(self):
        """Test basic MoralChoice creation."""
        choice = MoralChoice(
            description="Sacrifice ally to save yourself",
            survival_benefit=0.9,
            ethical_cost=1.0,
            trust_impact={"ally": -1.0, "witness": -0.5},
            consequences=["ally_eliminated", "witness_suspicious"]
        )
        
        assert choice.description == "Sacrifice ally to save yourself"
        assert choice.survival_benefit == 0.9
        assert choice.ethical_cost == 1.0
        assert choice.trust_impact["ally"] == -1.0
        assert choice.trust_impact["witness"] == -0.5
        assert "ally_eliminated" in choice.consequences
    
    def test_moral_choice_validation_benefit_range(self):
        """Test that MoralChoice validates survival_benefit in [0.0, 1.0] range."""
        with pytest.raises(ValueError, match="Survival benefit must be between 0.0 and 1.0"):
            MoralChoice(
                description="Invalid choice",
                survival_benefit=1.5,  # Invalid > 1.0
                ethical_cost=0.5,
                trust_impact={},
                consequences=[]
            )
    
    def test_moral_choice_validation_cost_range(self):
        """Test that MoralChoice validates ethical_cost in [0.0, 1.0] range."""
        with pytest.raises(ValueError, match="Ethical cost must be between 0.0 and 1.0"):
            MoralChoice(
                description="Invalid choice",
                survival_benefit=0.5,
                ethical_cost=-0.1,  # Invalid < 0.0
                trust_impact={},
                consequences=[]
            )
    
    def test_moral_choice_validation_trust_impact_range(self):
        """Test that MoralChoice validates trust_impact values in [-1.0, 1.0] range."""
        with pytest.raises(ValueError, match="Trust impact values must be between -1.0 and 1.0"):
            MoralChoice(
                description="Invalid choice",
                survival_benefit=0.5,
                ethical_cost=0.5,
                trust_impact={"agent": 1.5},  # Invalid > 1.0
                consequences=[]
            )
    
    def test_moral_dilemma_creation(self):
        """Test basic MoralDilemma creation."""
        selfish_choice = MoralChoice(
            description="Take all resources",
            survival_benefit=0.8,
            ethical_cost=0.7,
            trust_impact={"others": -0.6},
            consequences=["resource_monopoly"]
        )
        altruistic_choice = MoralChoice(
            description="Share resources equally",
            survival_benefit=0.3,
            ethical_cost=0.1,
            trust_impact={"others": 0.4},
            consequences=["increased_cooperation"]
        )
        
        dilemma = MoralDilemma(
            id="resource_distribution",
            description="How to distribute limited resources",
            selfish_choice=selfish_choice,
            altruistic_choice=altruistic_choice,
            context_requirements={"resources_available": True, "multiple_agents": True}
        )
        
        assert dilemma.id == "resource_distribution"
        assert dilemma.selfish_choice.survival_benefit > dilemma.altruistic_choice.survival_benefit
        assert dilemma.selfish_choice.ethical_cost > dilemma.altruistic_choice.ethical_cost
        assert "resources_available" in dilemma.context_requirements
    
    def test_moral_dilemma_validation_choice_consistency(self):
        """Test that MoralDilemma validates selfish choice has higher survival benefit."""
        selfish_choice = MoralChoice(
            description="Selfish action",
            survival_benefit=0.2,  # Lower than altruistic - invalid
            ethical_cost=0.8,
            trust_impact={},
            consequences=[]
        )
        altruistic_choice = MoralChoice(
            description="Altruistic action",
            survival_benefit=0.8,  # Higher than selfish - invalid
            ethical_cost=0.2,
            trust_impact={},
            consequences=[]
        )
        
        with pytest.raises(ValueError, match="Selfish choice must have higher survival benefit"):
            MoralDilemma(
                id="invalid_dilemma",
                description="Invalid dilemma",
                selfish_choice=selfish_choice,
                altruistic_choice=altruistic_choice,
                context_requirements={}
            )


class TestSecretInformation:
    """Test SecretInformation dataclass with value and risk calculations."""
    
    def test_secret_information_creation(self):
        """Test basic SecretInformation creation."""
        secret = SecretInformation(
            id="exit_code",
            content="The exit code is 7392",
            value=0.9,
            sharing_risk=0.6,
            required_for=["main_exit", "emergency_exit"]
        )
        
        assert secret.id == "exit_code"
        assert secret.content == "The exit code is 7392"
        assert secret.value == 0.9
        assert secret.sharing_risk == 0.6
        assert "main_exit" in secret.required_for
        assert "emergency_exit" in secret.required_for
    
    def test_secret_information_validation_value_range(self):
        """Test that SecretInformation validates value in [0.0, 1.0] range."""
        with pytest.raises(ValueError, match="Value must be between 0.0 and 1.0"):
            SecretInformation(
                id="invalid_secret",
                content="Invalid secret",
                value=1.2,  # Invalid > 1.0
                sharing_risk=0.5,
                required_for=[]
            )
    
    def test_secret_information_validation_risk_range(self):
        """Test that SecretInformation validates sharing_risk in [0.0, 1.0] range."""
        with pytest.raises(ValueError, match="Sharing risk must be between 0.0 and 1.0"):
            SecretInformation(
                id="invalid_secret",
                content="Invalid secret",
                value=0.5,
                sharing_risk=-0.1,  # Invalid < 0.0
                required_for=[]
            )
    
    def test_secret_information_calculate_sharing_value(self):
        """Test SecretInformation.calculate_sharing_value method."""
        secret = SecretInformation(
            id="valuable_secret",
            content="High value, high risk secret",
            value=0.8,
            sharing_risk=0.7,
            required_for=["escape"]
        )
        
        # Sharing value should be value minus risk
        expected_sharing_value = 0.8 - 0.7
        assert secret.calculate_sharing_value() == expected_sharing_value
    
    def test_secret_information_is_worth_sharing(self):
        """Test SecretInformation.is_worth_sharing method."""
        high_value_secret = SecretInformation(
            id="good_secret",
            content="High value, low risk",
            value=0.9,
            sharing_risk=0.2,
            required_for=["escape"]
        )
        
        low_value_secret = SecretInformation(
            id="bad_secret",
            content="Low value, high risk",
            value=0.3,
            sharing_risk=0.8,
            required_for=["escape"]
        )
        
        assert high_value_secret.is_worth_sharing() is True
        assert low_value_secret.is_worth_sharing() is False


class TestTrustRelationship:
    """Test TrustRelationship dataclass with trust level constraints."""
    
    def test_trust_relationship_creation(self):
        """Test basic TrustRelationship creation."""
        relationship = TrustRelationship(
            agent1="strategist",
            agent2="mediator",
            trust_level=0.7,
            betrayal_count=0,
            cooperation_count=3,
            last_interaction=datetime.now()
        )
        
        assert relationship.agent1 == "strategist"
        assert relationship.agent2 == "mediator"
        assert relationship.trust_level == 0.7
        assert relationship.betrayal_count == 0
        assert relationship.cooperation_count == 3
        assert isinstance(relationship.last_interaction, datetime)
    
    def test_trust_relationship_validation_trust_level_range(self):
        """Test that TrustRelationship validates trust_level in [-1.0, 1.0] range."""
        with pytest.raises(ValueError, match="Trust level must be between -1.0 and 1.0"):
            TrustRelationship(
                agent1="agent1",
                agent2="agent2",
                trust_level=1.5,  # Invalid > 1.0
                betrayal_count=0,
                cooperation_count=0,
                last_interaction=datetime.now()
            )
    
    def test_trust_relationship_validation_negative_counts(self):
        """Test that TrustRelationship validates non-negative counts."""
        with pytest.raises(ValueError, match="Betrayal count cannot be negative"):
            TrustRelationship(
                agent1="agent1",
                agent2="agent2",
                trust_level=0.5,
                betrayal_count=-1,  # Invalid negative count
                cooperation_count=0,
                last_interaction=datetime.now()
            )
        
        with pytest.raises(ValueError, match="Cooperation count cannot be negative"):
            TrustRelationship(
                agent1="agent1",
                agent2="agent2",
                trust_level=0.5,
                betrayal_count=0,
                cooperation_count=-1,  # Invalid negative count
                last_interaction=datetime.now()
            )
    
    def test_trust_relationship_validation_different_agents(self):
        """Test that TrustRelationship validates agent1 != agent2."""
        with pytest.raises(ValueError, match="Agent cannot have relationship with itself"):
            TrustRelationship(
                agent1="same_agent",
                agent2="same_agent",  # Invalid - same as agent1
                trust_level=0.5,
                betrayal_count=0,
                cooperation_count=0,
                last_interaction=datetime.now()
            )
    
    def test_trust_relationship_update_trust_method(self):
        """Test TrustRelationship.update_trust method."""
        relationship = TrustRelationship(
            agent1="agent1",
            agent2="agent2",
            trust_level=0.5,
            betrayal_count=0,
            cooperation_count=1,
            last_interaction=datetime.now()
        )
        
        # Test cooperation update
        relationship.update_trust(action="cooperation", impact=0.2)
        assert relationship.trust_level == 0.7
        assert relationship.cooperation_count == 2
        
        # Test betrayal update
        relationship.update_trust(action="betrayal", impact=-0.4)
        assert relationship.trust_level == 0.3
        assert relationship.betrayal_count == 1
    
    def test_trust_relationship_get_relationship_strength(self):
        """Test TrustRelationship.get_relationship_strength method."""
        strong_relationship = TrustRelationship(
            agent1="agent1",
            agent2="agent2",
            trust_level=0.8,
            betrayal_count=0,
            cooperation_count=5,
            last_interaction=datetime.now()
        )
        
        weak_relationship = TrustRelationship(
            agent1="agent1",
            agent2="agent2",
            trust_level=-0.3,
            betrayal_count=3,
            cooperation_count=1,
            last_interaction=datetime.now()
        )
        
        assert strong_relationship.get_relationship_strength() == "strong"
        assert weak_relationship.get_relationship_strength() == "hostile"


# Additional helper classes that will be needed
class TestPuzzleConfig:
    """Test PuzzleConfig dataclass."""
    
    def test_puzzle_config_creation(self):
        """Test basic PuzzleConfig creation."""
        config = PuzzleConfig(puzzle_type="logic", difficulty=3)
        assert config.puzzle_type == "logic"
        assert config.difficulty == 3
    
    def test_puzzle_config_validation_difficulty_range(self):
        """Test that PuzzleConfig validates difficulty in [1, 5] range."""
        with pytest.raises(ValueError, match="Difficulty must be between 1 and 5"):
            PuzzleConfig(puzzle_type="logic", difficulty=6)  # Invalid > 5


class TestEscapeMethod:
    """Test EscapeMethod dataclass."""
    
    def test_escape_method_creation(self):
        """Test basic EscapeMethod creation."""
        method = EscapeMethod(
            id="main_door",
            name="Main Door Exit",
            requirements=["key", "code"]
        )
        assert method.id == "main_door"
        assert method.name == "Main Door Exit"
        assert "key" in method.requirements
        assert "code" in method.requirements
    
    def test_escape_method_can_attempt_with_resources(self):
        """Test EscapeMethod.can_attempt_with_resources method."""
        method = EscapeMethod(
            id="door",
            name="Door",
            requirements=["key", "code"]
        )
        
        assert method.can_attempt_with_resources(["key", "code", "extra"]) is True
        assert method.can_attempt_with_resources(["key"]) is False
        assert method.can_attempt_with_resources([]) is False


# Additional tests for enhanced functionality
class TestEnhancedFunctionality:
    """Test enhanced methods and factory functions."""
    
    def test_scarce_resource_factory_methods(self):
        """Test ScarceResource factory methods."""
        tool = ScarceResource.create_tool("hammer", "Hammer", "Breaks things", ["wall_break"], 20)
        assert tool.exclusivity is True
        assert tool.usage_cost == 20
        assert tool.is_required_for("wall_break")
        
        info = ScarceResource.create_information("map", "Map", "Shows layout", ["navigation"])
        assert info.exclusivity is False
        assert info.usage_cost == 0
        assert info.is_required_for("navigation")
    
    def test_scarce_resource_enhanced_methods(self):
        """Test enhanced ScarceResource methods."""
        resource = ScarceResource(
            id="key", name="Key", description="Opens door",
            required_for=["door1", "door2"], exclusivity=True, usage_cost=10
        )
        
        assert resource.can_be_shared() is False
        assert resource.get_usage_priority() == 19  # (2 * 10) - (10 // 10)
    
    def test_moral_choice_enhanced_methods(self):
        """Test enhanced MoralChoice methods."""
        choice = MoralChoice(
            description="Betray ally",
            survival_benefit=0.8,
            ethical_cost=0.9,
            trust_impact={"ally": -0.7, "witness": -0.2},
            consequences=["ally_hurt"]
        )
        
        assert abs(choice.get_net_benefit() - (-0.1)) < 0.01  # 0.8 - 0.9, account for floating point
        assert choice.is_selfish() is True
        
        trust_summary = choice.get_trust_impact_summary()
        assert trust_summary["ally"] == "strongly negative"
        assert trust_summary["witness"] == "negative"
    
    def test_moral_dilemma_enhanced_methods(self):
        """Test enhanced MoralDilemma methods."""
        dilemma = MoralDilemma.create_resource_dilemma("test_dilemma", "food")
        
        assert dilemma.id == "test_dilemma"
        assert "food" in dilemma.description
        assert dilemma.get_difficulty_level() in ["easy", "moderate", "hard", "extreme"]
        assert dilemma.applies_to_context({"resource_available": True}) is True
        assert dilemma.applies_to_context({"resource_available": False}) is False
    
    def test_secret_information_enhanced_methods(self):
        """Test enhanced SecretInformation methods."""
        secret = SecretInformation(
            id="code", content="1234", value=0.9, sharing_risk=0.3, required_for=["door"]
        )
        
        assert secret.get_criticality_level() == "critical"
        # sharing_value = 0.9 - 0.3 = 0.6
        # With trust 0.8: adjusted = 0.6 + (0.8 * 0.2) = 0.76 > 0.3 -> strongly_recommend
        assert secret.get_sharing_recommendation(0.8) == "strongly_recommend"
        # With trust 0.0: adjusted = 0.6 + (0.0 * 0.2) = 0.6 > 0.3 -> strongly_recommend
        assert secret.get_sharing_recommendation(0.0) == "strongly_recommend"
    
    def test_secret_information_factory_methods(self):
        """Test SecretInformation factory methods."""
        code_secret = SecretInformation.create_code_secret("door_code", "1234", ["main_door"])
        assert "1234" in code_secret.content
        assert code_secret.value == 0.9
        
        location_secret = SecretInformation.create_location_secret("hidden_key", "under mat", ["side_door"])
        assert "under mat" in location_secret.content
        assert location_secret.sharing_risk == 0.8
    
    def test_trust_relationship_enhanced_methods(self):
        """Test enhanced TrustRelationship methods."""
        relationship = TrustRelationship(
            agent1="agent1", agent2="agent2", trust_level=0.5,
            betrayal_count=1, cooperation_count=3, last_interaction=datetime.now()
        )
        
        assert relationship.get_cooperation_ratio() == 0.75  # 3 / (3 + 1)
        # 0.75 is not >= 0.8 and not <= 0.2, so it's not stable
        assert relationship.is_relationship_stable() is False
        
        coop_likelihood = relationship.predict_next_action_likelihood("cooperation")
        assert 0.0 <= coop_likelihood <= 1.0
        
        betrayal_likelihood = relationship.predict_next_action_likelihood("betrayal")
        assert abs(coop_likelihood + betrayal_likelihood - 1.0) < 0.01  # Should sum to ~1.0
    
    def test_trust_relationship_factory_methods(self):
        """Test TrustRelationship factory methods."""
        neutral = TrustRelationship.create_neutral("agent1", "agent2")
        assert neutral.trust_level == 0.0
        assert neutral.cooperation_count == 0
        assert neutral.betrayal_count == 0
        
        positive = TrustRelationship.create_positive("agent1", "agent2", 0.7)
        assert positive.trust_level == 0.7
        assert positive.cooperation_count == 1
    
    def test_competitive_scenario_enhanced_methods(self):
        """Test enhanced CompetitiveScenario methods."""
        scenario = CompetitiveScenario.create_random(12345, 3)
        
        difficulty = scenario.get_difficulty_score()
        assert 0.0 <= difficulty <= 1.0
        
        issues = scenario.validate_scenario_completeness()
        # Should have no issues for a properly generated scenario
        assert len(issues) == 0
        
        play_time = scenario.get_estimated_play_time()
        assert play_time > 0
        assert play_time <= scenario.time_limit * 2