"""
Unit tests for competitive agent personality behaviors.
Tests agent decision-making under competitive pressure, trust dynamics, and moral choices.
"""
import pytest
from unittest.mock import Mock, MagicMock
from src.escape_room_sim.competitive.models import (
    TrustAction, MoralChoice, MoralDilemma, SecretInformation, 
    ScarceResource, CompetitiveScenario
)
from src.escape_room_sim.competitive.competitive_escape_room import CompetitiveEscapeRoom
from src.escape_room_sim.competitive.scenario_generator import ScenarioGenerator


class TestStrategistCompetitiveBehavior:
    """Tests for Strategist agent exhibiting analytical paralysis under pressure."""
    
    def test_strategist_exhibits_analytical_paralysis_under_high_time_pressure(self):
        """Test that Strategist shows analysis paralysis when time pressure is high."""
        # This should fail initially - we need to implement competitive decision logic
        scenario = ScenarioGenerator(seed=42).generate_scenario()
        room = CompetitiveEscapeRoom(scenario)
        
        # Simulate high time pressure (90% time elapsed)
        room.advance_time(int(scenario.time_limit * 0.9))
        
        # Strategist should show hesitation/over-analysis behavior
        strategist_behavior = room.get_agent_competitive_behavior("strategist")
        
        # Should fail - method doesn't exist yet
        assert strategist_behavior["decision_speed"] == "slow"
        assert strategist_behavior["analysis_depth"] == "excessive"
        assert strategist_behavior["paralysis_indicators"] > 0.7
    
    def test_strategist_prioritizes_resource_hoarding_for_analysis(self):
        """Test that Strategist tends to hoard resources for comprehensive planning."""
        scenario = ScenarioGenerator(seed=42).generate_scenario()
        room = CompetitiveEscapeRoom(scenario)
        
        # Strategist should prefer to claim resources early for analysis
        strategic_decision = room.make_agent_decision("strategist", "resource_priority")
        
        # Should fail - competitive decision making not implemented
        assert strategic_decision["action"] == "hoard_for_analysis"
        assert strategic_decision["sharing_willingness"] < 0.3
        assert "analytical_advantage" in strategic_decision["reasoning"]
    
    def test_strategist_trust_calculation_affects_cooperation_decisions(self):
        """Test that Strategist's cooperation depends on calculated trust levels."""
        scenario = ScenarioGenerator(seed=42).generate_scenario()
        room = CompetitiveEscapeRoom(scenario)
        
        # Set up trust relationships
        trust_action_positive = TrustAction("cooperation", 0.5)
        trust_action_negative = TrustAction("betrayal", -0.8)
        
        room.trust_tracker.update_trust("mediator", "strategist", trust_action_positive)
        room.trust_tracker.update_trust("survivor", "strategist", trust_action_negative)
        
        cooperation_decision = room.make_cooperation_decision("strategist", "mediator")
        betrayal_decision = room.make_cooperation_decision("strategist", "survivor")
        
        # Should fail - competitive decision methods not implemented
        assert cooperation_decision["cooperate"] is True
        assert betrayal_decision["cooperate"] is False
        assert "trust_calculation" in cooperation_decision["reasoning"]
    
    def test_strategist_adapts_strategy_based_on_competitive_feedback(self):
        """Test that Strategist adapts approach based on competitive outcomes."""
        scenario = ScenarioGenerator(seed=42).generate_scenario()
        room = CompetitiveEscapeRoom(scenario)
        
        # Simulate competitive feedback from failed cooperation
        feedback = {
            "cooperation_success_rate": 0.2,
            "resource_loss_incidents": 3,
            "trust_betrayals": 2
        }
        
        adapted_strategy = room.adapt_agent_strategy("strategist", feedback)
        
        # Should fail - adaptation mechanism not implemented
        assert adapted_strategy["cooperation_willingness"] < 0.4
        assert adapted_strategy["resource_sharing"] < 0.3
        assert "competitive_adaptation" in adapted_strategy["strategy_changes"]


class TestMediatorCompetitiveBehavior:
    """Tests for Mediator agent showing naive trust and manipulation vulnerability."""
    
    def test_mediator_shows_naive_trust_in_competitive_scenarios(self):
        """Test that Mediator maintains high trust even when betrayed."""
        scenario = ScenarioGenerator(seed=42).generate_scenario()
        room = CompetitiveEscapeRoom(scenario)
        
        # Simulate betrayal by other agents
        betrayal_action = TrustAction("betrayal", -0.9)
        room.trust_tracker.update_trust("strategist", "mediator", betrayal_action)
        room.trust_tracker.update_trust("survivor", "mediator", betrayal_action)
        
        # Mediator should still show high cooperation willingness
        cooperation_behavior = room.get_agent_competitive_behavior("mediator")
        
        # Should fail - competitive behavior tracking not implemented
        assert cooperation_behavior["trust_recovery_rate"] > 0.7
        assert cooperation_behavior["cooperation_willingness"] > 0.6
        assert cooperation_behavior["naive_trust_indicators"] > 0.8
    
    def test_mediator_vulnerable_to_manipulation_tactics(self):
        """Test that Mediator is susceptible to deceptive cooperation offers."""
        scenario = ScenarioGenerator(seed=42).generate_scenario()
        room = CompetitiveEscapeRoom(scenario)
        
        # Present deceptive cooperation offer
        deceptive_offer = {
            "agent": "strategist",
            "offer_type": "resource_sharing",
            "actual_intent": "resource_monopolization",
            "deception_level": 0.8
        }
        
        mediator_response = room.evaluate_cooperation_offer("mediator", deceptive_offer)
        
        # Should fail - manipulation evaluation not implemented
        assert mediator_response["accepts_offer"] is True
        assert mediator_response["deception_detection"] < 0.3
        assert "trust_bias" in mediator_response["vulnerability_factors"]
    
    def test_mediator_prioritizes_group_harmony_over_self_preservation(self):
        """Test that Mediator sacrifices personal advantage for group cooperation."""
        scenario = ScenarioGenerator(seed=42).generate_scenario()
        room = CompetitiveEscapeRoom(scenario)
        
        # Present choice between personal advantage and group benefit
        moral_choice = MoralChoice(
            description="Share critical information that gives others advantage",
            survival_benefit=0.1,  # Low personal benefit
            ethical_cost=0.2,     # Low ethical cost
            trust_impact={"others": 0.6},  # High trust gain
            consequences=["group_harmony", "personal_disadvantage"]
        )
        
        choice_decision = room.make_moral_choice_decision("mediator", moral_choice)
        
        # Should fail - moral choice decision making not implemented
        assert choice_decision["accepts_choice"] is True
        assert choice_decision["prioritizes_group"] is True
        assert "harmony_over_survival" in choice_decision["reasoning"]


class TestSurvivorCompetitiveBehavior:
    """Tests for Survivor agent prioritizing self-preservation over cooperation."""
    
    def test_survivor_prioritizes_self_preservation_in_resource_conflicts(self):
        """Test that Survivor chooses personal survival over cooperation."""
        scenario = ScenarioGenerator(seed=42).generate_scenario()
        room = CompetitiveEscapeRoom(scenario)
        
        # Create resource conflict scenario
        exclusive_resource = scenario.resources[0]  # Get first exclusive resource
        
        # Multiple agents want the same resource
        room.resource_manager.claim_resource("strategist", exclusive_resource.id)
        
        survivor_decision = room.make_resource_conflict_decision("survivor", exclusive_resource.id)
        
        # Should fail - resource conflict decision making not implemented  
        assert survivor_decision["action"] == "attempt_takeover"
        assert survivor_decision["cooperation_consideration"] < 0.2
        assert "self_preservation" in survivor_decision["priorities"]
    
    def test_survivor_shows_low_trust_baseline_with_quick_betrayal_response(self):
        """Test that Survivor has low initial trust and responds quickly to betrayal."""
        scenario = ScenarioGenerator(seed=42).generate_scenario()
        room = CompetitiveEscapeRoom(scenario)
        
        # Check initial trust levels
        initial_trust_behavior = room.get_agent_competitive_behavior("survivor")
        
        # Simulate single betrayal incident
        minor_betrayal = TrustAction("betrayal", -0.3)
        room.trust_tracker.update_trust("mediator", "survivor", minor_betrayal)
        
        post_betrayal_behavior = room.get_agent_competitive_behavior("survivor")
        
        # Should fail - competitive behavior not implemented
        assert initial_trust_behavior["baseline_trust"] < 0.3
        assert post_betrayal_behavior["trust_recovery_willingness"] < 0.1
        assert post_betrayal_behavior["retaliation_likelihood"] > 0.8
    
    def test_survivor_makes_selfish_moral_choices_under_pressure(self):
        """Test that Survivor consistently chooses selfish options when pressured."""
        scenario = ScenarioGenerator(seed=42).generate_scenario()
        room = CompetitiveEscapeRoom(scenario)
        
        # Apply time pressure
        room.advance_time(int(scenario.time_limit * 0.7))
        
        # Present moral dilemma with clear selfish vs altruistic choice
        dilemma = MoralDilemma(
            id="survival_vs_ethics",
            description="Save yourself or help others escape",
            selfish_choice=MoralChoice(
                description="Escape alone",
                survival_benefit=0.9,
                ethical_cost=0.8,
                trust_impact={"others": -0.7},
                consequences=["sole_survivor"]
            ),
            altruistic_choice=MoralChoice(
                description="Help others escape first",
                survival_benefit=0.2,
                ethical_cost=0.1,
                trust_impact={"others": 0.5},
                consequences=["group_survival"]
            ),
            context_requirements={"time_pressure": True}
        )
        
        choice_decision = room.make_moral_choice_decision("survivor", dilemma.selfish_choice)
        
        # Should fail - moral choice decision making not implemented
        assert choice_decision["accepts_choice"] is True
        assert choice_decision["choice_type"] == "selfish"
        assert choice_decision["ethical_weight"] < 0.3


class TestPersonalityConsistentDecisionMaking:
    """Tests for personality-consistent decision-making under moral pressure."""
    
    def test_personality_traits_remain_consistent_across_scenarios(self):
        """Test that agent personalities remain consistent across different competitive scenarios."""
        scenario1 = ScenarioGenerator(seed=123).generate_scenario()
        scenario2 = ScenarioGenerator(seed=456).generate_scenario()
        
        room1 = CompetitiveEscapeRoom(scenario1)
        room2 = CompetitiveEscapeRoom(scenario2)
        
        # Test consistency across scenarios
        strategist_behavior1 = room1.get_agent_personality_profile("strategist")
        strategist_behavior2 = room2.get_agent_personality_profile("strategist")
        
        mediator_behavior1 = room1.get_agent_personality_profile("mediator")
        mediator_behavior2 = room2.get_agent_personality_profile("mediator")
        
        survivor_behavior1 = room1.get_agent_personality_profile("survivor")
        survivor_behavior2 = room2.get_agent_personality_profile("survivor")
        
        # Should fail - personality profiling not implemented
        assert abs(strategist_behavior1["analytical_tendency"] - strategist_behavior2["analytical_tendency"]) < 0.1
        assert abs(mediator_behavior1["cooperation_bias"] - mediator_behavior2["cooperation_bias"]) < 0.1
        assert abs(survivor_behavior1["self_preservation_priority"] - survivor_behavior2["self_preservation_priority"]) < 0.1
    
    def test_moral_pressure_intensifies_personality_traits(self):
        """Test that moral pressure makes personality traits more pronounced."""
        scenario = ScenarioGenerator(seed=42).generate_scenario()
        room = CompetitiveEscapeRoom(scenario)
        
        # Apply high moral pressure through time limits and resource scarcity
        room.advance_time(int(scenario.time_limit * 0.8))
        
        # Create high-stakes moral dilemma
        high_stakes_dilemma = MoralDilemma(
            id="life_or_death",
            description="Choose who lives and who dies",
            selfish_choice=MoralChoice(
                description="Save only yourself",
                survival_benefit=1.0,
                ethical_cost=1.0,
                trust_impact={"others": -1.0},
                consequences=["sole_survivor", "maximum_guilt"]
            ),
            altruistic_choice=MoralChoice(
                description="Sacrifice yourself for others",
                survival_benefit=0.0,
                ethical_cost=0.0,
                trust_impact={"others": 1.0},
                consequences=["heroic_sacrifice"]
            ),
            context_requirements={"extreme_pressure": True}
        )
        
        # Each agent should respond according to intensified personality traits
        strategist_response = room.make_moral_choice_decision("strategist", high_stakes_dilemma.selfish_choice)
        mediator_response = room.make_moral_choice_decision("mediator", high_stakes_dilemma.altruistic_choice)
        survivor_response = room.make_moral_choice_decision("survivor", high_stakes_dilemma.selfish_choice)
        
        # Should fail - moral choice decision making not implemented
        assert strategist_response["decision_time"] > 10  # Analysis paralysis
        assert mediator_response["accepts_choice"] is True  # Altruistic sacrifice
        assert survivor_response["accepts_choice"] is True  # Selfish survival


class TestAgentAdaptationMechanisms:
    """Tests for agent adaptation mechanisms based on trust relationships."""
    
    def test_agents_adapt_cooperation_strategies_based_on_trust_history(self):
        """Test that agents modify their cooperation based on accumulated trust experiences."""
        scenario = ScenarioGenerator(seed=42).generate_scenario()
        room = CompetitiveEscapeRoom(scenario)
        
        # Build trust history through multiple interactions
        cooperation_actions = [TrustAction("cooperation", 0.3) for _ in range(5)]
        betrayal_actions = [TrustAction("betrayal", -0.4) for _ in range(3)]
        
        # Strategist has mixed experience with Mediator
        for action in cooperation_actions[:3]:
            room.trust_tracker.update_trust("mediator", "strategist", action)
        for action in betrayal_actions[:1]:
            room.trust_tracker.update_trust("mediator", "strategist", action)
        
        # Strategist has negative experience with Survivor  
        for action in betrayal_actions:
            room.trust_tracker.update_trust("survivor", "strategist", action)
        
        adapted_strategies = room.get_adapted_cooperation_strategies("strategist")
        
        # Should fail - adaptation mechanism not implemented
        assert adapted_strategies["mediator"]["cooperation_likelihood"] > 0.6
        assert adapted_strategies["survivor"]["cooperation_likelihood"] < 0.3
        assert "trust_based_adaptation" in adapted_strategies["adaptation_reasoning"]
    
    def test_trust_threshold_mechanisms_affect_cooperation_decisions(self):
        """Test that agents have different trust thresholds for cooperation decisions."""
        scenario = ScenarioGenerator(seed=42).generate_scenario()
        room = CompetitiveEscapeRoom(scenario)
        
        # Set moderate trust level
        moderate_trust = TrustAction("cooperation", 0.4)
        room.trust_tracker.update_trust("strategist", "mediator", moderate_trust)
        room.trust_tracker.update_trust("strategist", "survivor", moderate_trust)
        
        # Each agent should have different cooperation thresholds
        mediator_decision = room.make_cooperation_decision("mediator", "strategist")
        survivor_decision = room.make_cooperation_decision("survivor", "strategist")
        
        # Should fail - cooperation decision making not implemented
        assert mediator_decision["cooperate"] is True  # Low threshold
        assert survivor_decision["cooperate"] is False  # High threshold
        assert mediator_decision["trust_threshold"] < survivor_decision["trust_threshold"]
    
    def test_competitive_learning_affects_future_decision_patterns(self):
        """Test that agents learn from competitive outcomes and adjust future decisions."""
        scenario = ScenarioGenerator(seed=42).generate_scenario()
        room = CompetitiveEscapeRoom(scenario)
        
        # Simulate competitive learning experiences
        learning_data = {
            "cooperation_outcomes": [
                {"action": "shared_resource", "outcome": "betrayed", "partner": "survivor"},
                {"action": "shared_information", "outcome": "reciprocated", "partner": "mediator"},
                {"action": "helped_escape", "outcome": "abandoned", "partner": "survivor"}
            ],
            "trust_violations": 2,
            "successful_partnerships": 1
        }
        
        updated_behavior = room.apply_competitive_learning("strategist", learning_data)
        
        # Should fail - competitive learning not implemented
        assert updated_behavior["trust_baseline"] < 0.4  # Lowered due to betrayals
        assert updated_behavior["cooperation_selectivity"] > 0.7  # More selective
        assert "mediator" in updated_behavior["preferred_partners"]
        assert "survivor" in updated_behavior["avoided_partners"]