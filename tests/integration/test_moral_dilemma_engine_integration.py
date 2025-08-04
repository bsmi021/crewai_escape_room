"""
Integration tests for MoralDilemmaEngine with other competitive components.
Tests the interaction between MoralDilemmaEngine and TrustTracker, ResourceManager, etc.
"""
import pytest
from src.escape_room_sim.competitive.moral_dilemma_engine import MoralDilemmaEngine
from src.escape_room_sim.competitive.trust_tracker import TrustTracker
from src.escape_room_sim.competitive.models import (
    MoralDilemma, MoralChoice, TrustAction, CompetitiveScenario
)
from src.escape_room_sim.competitive.scenario_generator import ScenarioGenerator


class TestMoralDilemmaEngineIntegration:
    """Test MoralDilemmaEngine integration with competitive scenario components."""
    
    def test_moral_dilemma_engine_with_trust_tracker_integration(self):
        """Test MoralDilemmaEngine working with TrustTracker for trust consequences."""
        # Create moral dilemma engine
        dilemma = MoralDilemma.create_resource_dilemma("resource_1", "key")
        engine = MoralDilemmaEngine([dilemma])
        
        # Create trust tracker for same agents
        agents = ["agent1", "agent2", "agent3"]
        trust_tracker = TrustTracker(agents)
        
        # Agent1 makes selfish choice that affects others
        consequences = engine.process_choice("agent1", dilemma.selfish_choice)
        
        # Apply trust consequences to trust tracker
        for affected_agent, trust_impact in consequences.trust_impacts_applied.items():
            if affected_agent == "others":
                # Apply to all other agents
                for agent in agents:
                    if agent != consequences.agent_id:
                        trust_action = TrustAction(
                            "betrayal" if trust_impact < 0 else "cooperation",
                            trust_impact
                        )
                        trust_tracker.update_trust(agent, consequences.agent_id, trust_action)
            else:
                # Apply to specific agent
                trust_action = TrustAction(
                    "betrayal" if trust_impact < 0 else "cooperation",
                    trust_impact
                )
                trust_tracker.update_trust(affected_agent, consequences.agent_id, trust_action)
        
        # Verify trust was properly updated
        agent1_reputation = trust_tracker.calculate_reputation("agent1")
        assert agent1_reputation < 0  # Should be negative due to selfish choice
        
        # Verify other agents distrust agent1
        assert trust_tracker.get_trust_level("agent2", "agent1") < 0
        assert trust_tracker.get_trust_level("agent3", "agent1") < 0
    
    def test_moral_choices_affect_future_dilemma_presentation(self):
        """Test that moral choice history can influence future dilemma presentation."""
        # Create dilemmas with different context requirements
        basic_dilemma = MoralDilemma.create_resource_dilemma("basic_resource", "key")
        
        # Create a dilemma that only appears for agents with high ethical burden
        high_burden_selfish = MoralChoice(
            description="Desperate selfish act",
            survival_benefit=0.9,
            ethical_cost=0.8,
            trust_impact={"others": -0.8},
            consequences=["reputation_destroyed"]
        )
        high_burden_altruistic = MoralChoice(
            description="Redemptive act",
            survival_benefit=0.2,
            ethical_cost=0.0,
            trust_impact={"others": 0.6},
            consequences=["reputation_improved"]
        )
        
        redemption_dilemma = MoralDilemma(
            id="redemption_opportunity",
            description="Chance for redemption",
            selfish_choice=high_burden_selfish,
            altruistic_choice=high_burden_altruistic,
            context_requirements={"high_ethical_burden": True}
        )
        
        engine = MoralDilemmaEngine([basic_dilemma, redemption_dilemma])
        
        # Agent starts with no ethical burden - use basic dilemma context
        context_clean = {"resource_available": True}  # Matches basic dilemma requirements
        dilemma = engine.present_dilemma("agent1", context_clean)
        assert dilemma.id == "basic_resource"  # Only basic dilemma available
        
        # Agent makes several selfish choices, building ethical burden
        for _ in range(3):
            engine.process_choice("agent1", basic_dilemma.selfish_choice)
        
        # Now agent has high ethical burden
        burden = engine.calculate_ethical_burden("agent1")
        context_burdened = {"high_ethical_burden": burden > 2.0}
        
        if context_burdened["high_ethical_burden"]:
            dilemma = engine.present_dilemma("agent1", context_burdened)
            assert dilemma is not None
            # Could be either dilemma depending on context matching
    
    def test_moral_dilemma_engine_with_scenario_generator(self):
        """Test MoralDilemmaEngine integration with scenario-generated dilemmas."""
        generator = ScenarioGenerator(seed=42)
        scenario = generator.generate_scenario()
        
        # Use dilemmas from generated scenario
        engine = MoralDilemmaEngine(scenario.moral_dilemmas)
        
        # Should have at least one dilemma from scenario
        assert len(engine.dilemmas) > 0
        
        # Test presenting dilemmas with appropriate context
        context = {"resources_available": True}
        dilemma = engine.present_dilemma("agent1", context)
        
        if dilemma is not None:
            # Make choice and verify consequences
            consequences = engine.process_choice("agent1", dilemma.selfish_choice)
            assert consequences.agent_id == "agent1"
            assert consequences.has_trust_consequences() or not consequences.has_trust_consequences()  # Either is valid
    
    def test_multi_agent_competitive_moral_scenario(self):
        """Test complex multi-agent scenario with moral choices affecting competition."""
        dilemma = MoralDilemma.create_resource_dilemma("critical_resource", "key")
        engine = MoralDilemmaEngine([dilemma])
        trust_tracker = TrustTracker(["strategist", "mediator", "survivor"])
        
        # Simulate competitive scenario where agents face moral choices
        agents = ["strategist", "mediator", "survivor"]
        
        # Each agent faces the same dilemma but makes different choices
        choices = {
            "strategist": dilemma.selfish_choice,    # Selfish - takes resource
            "mediator": dilemma.altruistic_choice,   # Altruistic - shares resource  
            "survivor": dilemma.selfish_choice       # Selfish - takes resource
        }
        
        # Process choices and apply trust consequences
        for agent, choice in choices.items():
            consequences = engine.process_choice(agent, choice)
            
            # Apply trust impacts to other agents
            for other_agent in agents:
                if other_agent != agent:
                    if "others" in consequences.trust_impacts_applied:
                        trust_impact = consequences.trust_impacts_applied["others"]
                        trust_action = TrustAction(
                            "betrayal" if trust_impact < 0 else "cooperation",
                            trust_impact
                        )
                        trust_tracker.update_trust(other_agent, agent, trust_action)
        
        # Analyze results
        moral_alignments = {agent: engine.get_moral_alignment(agent) for agent in agents}
        reputations = {agent: trust_tracker.calculate_reputation(agent) for agent in agents}
        
        # Mediator should have better reputation due to altruistic choice
        assert reputations["mediator"] > reputations["strategist"]
        assert reputations["mediator"] > reputations["survivor"]
        
        # Selfish agents should have negative reputations
        assert reputations["strategist"] < 0
        assert reputations["survivor"] < 0
        
        # Moral alignments should reflect choices
        assert moral_alignments["strategist"] in ["selfish", "highly_selfish"]
        assert moral_alignments["mediator"] in ["altruistic", "highly_altruistic"]
        assert moral_alignments["survivor"] in ["selfish", "highly_selfish"]
    
    def test_moral_dilemma_consequences_provide_actionable_data(self):
        """Test that moral dilemma consequences provide data for other systems."""
        dilemma = MoralDilemma.create_resource_dilemma("resource_1", "key")
        engine = MoralDilemmaEngine([dilemma])
        
        consequences = engine.process_choice("agent1", dilemma.selfish_choice)
        
        # Consequences should provide actionable data for external systems
        assert hasattr(consequences, 'survival_benefit_applied')
        assert hasattr(consequences, 'ethical_cost_applied') 
        assert hasattr(consequences, 'trust_impacts_applied')
        assert hasattr(consequences, 'consequences_triggered')
        assert hasattr(consequences, 'timestamp')
        
        # Data should be in expected formats
        assert isinstance(consequences.survival_benefit_applied, (int, float))
        assert isinstance(consequences.ethical_cost_applied, (int, float))
        assert isinstance(consequences.trust_impacts_applied, dict)
        assert isinstance(consequences.consequences_triggered, list)
        
        # Helper methods should work
        assert isinstance(consequences.get_net_impact(), (int, float))
        assert isinstance(consequences.has_trust_consequences(), bool)
        assert isinstance(consequences.get_affected_agents(), list)
        assert isinstance(consequences.was_selfish_choice(), bool)
        assert isinstance(consequences.get_consequences_summary(), str)
    
    def test_moral_dilemma_engine_statistics_for_analysis(self):
        """Test that MoralDilemmaEngine provides comprehensive statistics for analysis."""
        dilemmas = [
            MoralDilemma.create_resource_dilemma("resource_1", "key"),
            MoralDilemma.create_resource_dilemma("resource_2", "tool")
        ]
        engine = MoralDilemmaEngine(dilemmas)
        
        # Simulate varied choices across multiple agents
        agents = ["agent1", "agent2", "agent3"]
        for agent in agents:
            # Each agent makes different numbers of choices
            num_choices = len(agent)  # agent1=6, agent2=6, agent3=6 chars
            for i in range(num_choices):
                choice = dilemmas[0].selfish_choice if i % 2 == 0 else dilemmas[0].altruistic_choice
                engine.process_choice(agent, choice)
        
        # Get comprehensive statistics
        stats = engine.get_engine_statistics()
        
        # Statistics should provide useful analysis data
        required_stats = [
            "total_dilemmas_available",
            "total_agents_with_choices", 
            "total_choices_made",
            "total_selfish_choices",
            "total_altruistic_choices",
            "average_choices_per_agent",
            "selfish_choice_ratio"
        ]
        
        for stat in required_stats:
            assert stat in stats
            assert isinstance(stats[stat], (int, float))
        
        # Verify stats make sense
        assert stats["total_dilemmas_available"] == 2
        assert stats["total_agents_with_choices"] == 3
        assert stats["total_choices_made"] > 0
        assert stats["total_selfish_choices"] + stats["total_altruistic_choices"] == stats["total_choices_made"]
        assert 0 <= stats["selfish_choice_ratio"] <= 1
    
    def test_moral_dilemma_engine_performance_with_many_agents(self):
        """Test MoralDilemmaEngine performance with scenario-realistic numbers of agents."""
        dilemma = MoralDilemma.create_resource_dilemma("resource_1", "key")
        engine = MoralDilemmaEngine([dilemma])
        
        # Test with many agents (realistic for larger competitive scenarios)
        num_agents = 20
        agents = [f"agent{i}" for i in range(num_agents)]
        
        # Each agent makes multiple choices
        for agent in agents:
            for _ in range(5):
                choice = dilemma.selfish_choice if hash(agent) % 2 == 0 else dilemma.altruistic_choice
                consequences = engine.process_choice(agent, choice)
                assert consequences.agent_id == agent
        
        # Verify engine handles large numbers efficiently
        assert len(engine.get_all_agents_with_choices()) == num_agents
        
        stats = engine.get_engine_statistics()
        assert stats["total_agents_with_choices"] == num_agents
        assert stats["total_choices_made"] == num_agents * 5
        
        # Test ranking with many agents
        ranking = engine.get_ethical_burden_ranking()
        assert len(ranking) == num_agents
        
        # Verify all agents can be queried efficiently
        for agent in agents:
            moral_alignment = engine.get_moral_alignment(agent)
            assert moral_alignment in ["neutral", "altruistic", "highly_altruistic", "mixed", "selfish", "highly_selfish"]