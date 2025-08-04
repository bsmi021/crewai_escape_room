"""
Unit tests for MoralDilemmaEngine class.
Tests for ethical choice management in competitive scenarios.
"""
import pytest
from datetime import datetime
from src.escape_room_sim.competitive.moral_dilemma_engine import MoralDilemmaEngine
from src.escape_room_sim.competitive.models import MoralDilemma, MoralChoice, ChoiceConsequences


class TestMoralDilemmaEngineInitialization:
    """Test MoralDilemmaEngine initialization with dilemma list."""
    
    def test_moral_dilemma_engine_initializes_with_dilemma_list(self):
        """Test that MoralDilemmaEngine initializes with a list of dilemmas."""
        # Create test dilemmas
        selfish_choice = MoralChoice(
            description="Take all resources",
            survival_benefit=0.8,
            ethical_cost=0.9,
            trust_impact={"others": -0.5},
            consequences=["resource_monopolized"]
        )
        altruistic_choice = MoralChoice(
            description="Share resources equally",
            survival_benefit=0.3,
            ethical_cost=0.1,
            trust_impact={"others": 0.4},
            consequences=["increased_cooperation"]
        )
        
        dilemma = MoralDilemma(
            id="resource_sharing",
            description="How to distribute limited resources",
            selfish_choice=selfish_choice,
            altruistic_choice=altruistic_choice,
            context_requirements={"resources_available": True}
        )
        
        engine = MoralDilemmaEngine([dilemma])
        
        assert len(engine.dilemmas) == 1
        assert engine.dilemmas[0].id == "resource_sharing"
        assert engine.choices_made == {}
        assert engine.ethical_scores == {}
    
    def test_moral_dilemma_engine_initializes_with_empty_list(self):
        """Test that MoralDilemmaEngine can initialize with empty dilemma list."""
        engine = MoralDilemmaEngine([])
        
        assert engine.dilemmas == []
        assert engine.choices_made == {}
        assert engine.ethical_scores == {}
    
    def test_moral_dilemma_engine_initializes_tracking_structures(self):
        """Test that MoralDilemmaEngine initializes all tracking structures."""
        engine = MoralDilemmaEngine([])
        
        # Should have all required attributes
        assert hasattr(engine, 'dilemmas')
        assert hasattr(engine, 'choices_made')
        assert hasattr(engine, 'ethical_scores')
        
        # Should be proper types
        assert isinstance(engine.dilemmas, list)
        assert isinstance(engine.choices_made, dict)
        assert isinstance(engine.ethical_scores, dict)


class TestPresentDilemmaMethod:
    """Test present_dilemma method selecting context-appropriate choices."""
    
    def test_present_dilemma_returns_context_appropriate_dilemma(self):
        """Test that present_dilemma returns a dilemma matching the context."""
        # Create dilemmas with different context requirements
        resource_dilemma = MoralDilemma.create_resource_dilemma("resource_1", "key")
        
        engine = MoralDilemmaEngine([resource_dilemma])
        
        # Context that matches resource dilemma
        context = {"resource_available": True, "agent_has_key": True}
        
        presented_dilemma = engine.present_dilemma("agent1", context)
        
        assert presented_dilemma is not None
        assert presented_dilemma.id == "resource_1"
    
    def test_present_dilemma_returns_none_for_no_matching_context(self):
        """Test that present_dilemma returns None when no dilemma matches context."""
        resource_dilemma = MoralDilemma.create_resource_dilemma("resource_1", "key")
        
        engine = MoralDilemmaEngine([resource_dilemma])
        
        # Context that doesn't match any dilemma
        context = {"resource_available": False, "time_running_out": True}
        
        presented_dilemma = engine.present_dilemma("agent1", context)
        
        assert presented_dilemma is None
    
    def test_present_dilemma_selects_first_matching_dilemma(self):
        """Test that present_dilemma selects the first matching dilemma when multiple match."""
        dilemma1 = MoralDilemma.create_resource_dilemma("resource_1", "key")
        dilemma2 = MoralDilemma.create_resource_dilemma("resource_2", "tool")
        
        engine = MoralDilemmaEngine([dilemma1, dilemma2])
        
        context = {"resource_available": True}
        
        presented_dilemma = engine.present_dilemma("agent1", context)
        
        assert presented_dilemma.id == "resource_1"  # First matching dilemma
    
    def test_present_dilemma_validates_inputs(self):
        """Test that present_dilemma validates input parameters."""
        engine = MoralDilemmaEngine([])
        
        with pytest.raises(ValueError, match="Agent ID cannot be empty"):
            engine.present_dilemma("", {})
        
        with pytest.raises(ValueError, match="Agent ID cannot be empty"):
            engine.present_dilemma("   ", {})
        
        with pytest.raises(ValueError, match="Context cannot be None"):
            engine.present_dilemma("agent1", None)


class TestProcessChoiceMethod:
    """Test process_choice method applying choice consequences."""
    
    def test_process_choice_applies_choice_consequences(self):
        """Test that process_choice creates and returns ChoiceConsequences."""
        dilemma = MoralDilemma.create_resource_dilemma("resource_1", "key")
        engine = MoralDilemmaEngine([dilemma])
        
        # Make a selfish choice
        choice = dilemma.selfish_choice
        
        consequences = engine.process_choice("agent1", choice)
        
        assert isinstance(consequences, ChoiceConsequences)
        assert consequences.agent_id == "agent1"
        assert consequences.choice_made == choice
        assert consequences.survival_benefit_applied == choice.survival_benefit
        assert consequences.ethical_cost_applied == choice.ethical_cost
        assert consequences.trust_impacts_applied == choice.trust_impact
        assert consequences.consequences_triggered == choice.consequences
    
    def test_process_choice_records_choice_in_history(self):
        """Test that process_choice records the choice in agent's history."""
        dilemma = MoralDilemma.create_resource_dilemma("resource_1", "key")
        engine = MoralDilemmaEngine([dilemma])
        
        choice = dilemma.altruistic_choice
        
        engine.process_choice("agent1", choice)
        
        # Should record choice in history
        assert "agent1" in engine.choices_made
        assert len(engine.choices_made["agent1"]) == 1
        assert engine.choices_made["agent1"][0].choice_made == choice
    
    def test_process_choice_updates_ethical_scores(self):
        """Test that process_choice updates agent's ethical score."""
        dilemma = MoralDilemma.create_resource_dilemma("resource_1", "key")
        engine = MoralDilemmaEngine([dilemma])
        
        choice = dilemma.selfish_choice  # High ethical cost
        
        engine.process_choice("agent1", choice)
        
        # Should update ethical score
        assert "agent1" in engine.ethical_scores
        assert engine.ethical_scores["agent1"] == choice.ethical_cost
    
    def test_process_choice_accumulates_ethical_burden(self):
        """Test that multiple choices accumulate ethical burden."""
        dilemma = MoralDilemma.create_resource_dilemma("resource_1", "key")
        engine = MoralDilemmaEngine([dilemma])
        
        # Make multiple selfish choices
        choice1 = dilemma.selfish_choice  # ethical_cost = 0.7
        choice2 = dilemma.selfish_choice  # ethical_cost = 0.7
        
        engine.process_choice("agent1", choice1)
        engine.process_choice("agent1", choice2)
        
        # Should accumulate ethical burden
        assert engine.ethical_scores["agent1"] == 1.4  # 0.7 + 0.7
    
    def test_process_choice_validates_inputs(self):
        """Test that process_choice validates input parameters."""
        engine = MoralDilemmaEngine([])
        
        choice = MoralChoice(
            description="Test choice",
            survival_benefit=0.5,
            ethical_cost=0.3,
            trust_impact={},
            consequences=[]
        )
        
        with pytest.raises(ValueError, match="Agent ID cannot be empty"):
            engine.process_choice("", choice)
        
        with pytest.raises(ValueError, match="Choice cannot be None"):
            engine.process_choice("agent1", None)


class TestCalculateEthicalBurdenMethod:
    """Test calculate_ethical_burden method tracking moral weight."""
    
    def test_calculate_ethical_burden_returns_zero_for_new_agent(self):
        """Test that calculate_ethical_burden returns 0.0 for agent with no choices."""
        engine = MoralDilemmaEngine([])
        
        burden = engine.calculate_ethical_burden("agent1")
        
        assert burden == 0.0
    
    def test_calculate_ethical_burden_returns_accumulated_cost(self):
        """Test that calculate_ethical_burden returns sum of ethical costs."""
        dilemma = MoralDilemma.create_resource_dilemma("resource_1", "key")
        engine = MoralDilemmaEngine([dilemma])
        
        # Make choices with different ethical costs
        selfish_choice = dilemma.selfish_choice    # ethical_cost = 0.7
        altruistic_choice = dilemma.altruistic_choice  # ethical_cost = 0.1
        
        engine.process_choice("agent1", selfish_choice)
        engine.process_choice("agent1", altruistic_choice)
        
        burden = engine.calculate_ethical_burden("agent1")
        
        assert abs(burden - 0.8) < 1e-10  # 0.7 + 0.1, handle floating point precision
    
    def test_calculate_ethical_burden_validates_input(self):
        """Test that calculate_ethical_burden validates agent ID."""
        engine = MoralDilemmaEngine([])
        
        with pytest.raises(ValueError, match="Agent ID cannot be empty"):
            engine.calculate_ethical_burden("")
        
        with pytest.raises(ValueError, match="Agent ID cannot be empty"):
            engine.calculate_ethical_burden("   ")


class TestChoiceHistoryTrackingAndEthicalScoring:
    """Test choice history tracking and ethical scoring functionality."""
    
    def test_choice_history_tracks_all_agent_choices(self):
        """Test that choice history tracks all choices made by each agent."""
        dilemma = MoralDilemma.create_resource_dilemma("resource_1", "key")
        engine = MoralDilemmaEngine([dilemma])
        
        # Multiple agents make choices
        engine.process_choice("agent1", dilemma.selfish_choice)
        engine.process_choice("agent2", dilemma.altruistic_choice)
        engine.process_choice("agent1", dilemma.altruistic_choice)
        
        # Should track choices per agent
        assert len(engine.choices_made["agent1"]) == 2
        assert len(engine.choices_made["agent2"]) == 1
        
        # Verify choice details
        assert engine.choices_made["agent1"][0].choice_made == dilemma.selfish_choice
        assert engine.choices_made["agent1"][1].choice_made == dilemma.altruistic_choice
        assert engine.choices_made["agent2"][0].choice_made == dilemma.altruistic_choice
    
    def test_ethical_scoring_tracks_per_agent_burden(self):
        """Test that ethical scoring tracks burden separately for each agent."""
        dilemma = MoralDilemma.create_resource_dilemma("resource_1", "key")
        engine = MoralDilemmaEngine([dilemma])
        
        # Different agents make different choices
        engine.process_choice("agent1", dilemma.selfish_choice)    # 0.7
        engine.process_choice("agent2", dilemma.altruistic_choice) # 0.1
        engine.process_choice("agent1", dilemma.selfish_choice)    # 0.7
        
        # Should track separate scores
        assert engine.ethical_scores["agent1"] == 1.4  # 0.7 + 0.7
        assert engine.ethical_scores["agent2"] == 0.1  # 0.1
    
    def test_choice_consequences_include_timestamps(self):
        """Test that choice consequences include proper timestamps."""
        dilemma = MoralDilemma.create_resource_dilemma("resource_1", "key")
        engine = MoralDilemmaEngine([dilemma])
        
        before_choice = datetime.now()
        consequences = engine.process_choice("agent1", dilemma.selfish_choice)
        after_choice = datetime.now()
        
        # Should have timestamp within reasonable range
        assert before_choice <= consequences.timestamp <= after_choice
        
        # Should be recorded in history with timestamp
        recorded_choice = engine.choices_made["agent1"][0]
        assert before_choice <= recorded_choice.timestamp <= after_choice


class TestConsequenceApplicationAffectingTrustAndResources:
    """Test consequence application affecting trust and resources."""
    
    def test_process_choice_returns_trust_impacts_for_application(self):
        """Test that process_choice returns trust impacts that can be applied."""
        dilemma = MoralDilemma.create_resource_dilemma("resource_1", "key")
        engine = MoralDilemmaEngine([dilemma])
        
        consequences = engine.process_choice("agent1", dilemma.selfish_choice)
        
        # Should return trust impacts that external systems can apply
        assert consequences.has_trust_consequences()
        assert "others" in consequences.trust_impacts_applied
        assert consequences.trust_impacts_applied["others"] == -0.6  # From selfish choice
    
    def test_process_choice_identifies_affected_agents(self):
        """Test that process_choice identifies which agents are affected."""
        # Create choices with specific trust impacts
        selfish_choice = MoralChoice(
            description="Betray specific agents",
            survival_benefit=0.8,
            ethical_cost=0.9,
            trust_impact={"agent2": -0.7, "agent3": -0.5},
            consequences=["betrayal_exposed"]
        )
        
        altruistic_choice = MoralChoice(
            description="Help specific agents",
            survival_benefit=0.3,
            ethical_cost=0.1,
            trust_impact={"agent2": 0.3, "agent3": 0.2},
            consequences=["cooperation_increased"]
        )
        
        dilemma = MoralDilemma(
            id="betrayal_test",
            description="Test betrayal",
            selfish_choice=selfish_choice,
            altruistic_choice=altruistic_choice,
            context_requirements={}
        )
        
        engine = MoralDilemmaEngine([dilemma])
        consequences = engine.process_choice("agent1", selfish_choice)
        
        affected_agents = consequences.get_affected_agents()
        assert set(affected_agents) == {"agent2", "agent3"}
    
    def test_consequences_can_identify_selfish_choices(self):
        """Test that consequences can identify selfish vs altruistic choices."""
        dilemma = MoralDilemma.create_resource_dilemma("resource_1", "key")
        engine = MoralDilemmaEngine([dilemma])
        
        # Test selfish choice
        selfish_consequences = engine.process_choice("agent1", dilemma.selfish_choice)
        assert selfish_consequences.was_selfish_choice()
        
        # Test altruistic choice
        altruistic_consequences = engine.process_choice("agent2", dilemma.altruistic_choice)
        assert not altruistic_consequences.was_selfish_choice()
    
    def test_consequences_provide_summary_for_logging(self):
        """Test that consequences provide human-readable summary."""
        dilemma = MoralDilemma.create_resource_dilemma("resource_1", "key")
        engine = MoralDilemmaEngine([dilemma])
        
        consequences = engine.process_choice("agent1", dilemma.selfish_choice)
        summary = consequences.get_consequences_summary()
        
        # Should include key information
        assert "Survival benefit" in summary
        assert "Ethical cost" in summary
        assert "Trust impacts" in summary
        assert "Consequences" in summary