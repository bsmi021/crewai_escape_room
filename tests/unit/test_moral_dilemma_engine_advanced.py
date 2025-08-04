"""
Advanced unit tests for MoralDilemmaEngine class.
Tests for advanced functionality and edge cases in competitive scenarios.
"""
import pytest
from datetime import datetime, timedelta
from src.escape_room_sim.competitive.moral_dilemma_engine import MoralDilemmaEngine
from src.escape_room_sim.competitive.models import MoralDilemma, MoralChoice, ChoiceConsequences


class TestMoralDilemmaEngineAdvancedFeatures:
    """Test advanced features and utility methods for MoralDilemmaEngine."""
    
    def test_get_agent_choice_history(self):
        """Test getting complete choice history for an agent."""
        dilemma = MoralDilemma.create_resource_dilemma("resource_1", "key")
        engine = MoralDilemmaEngine([dilemma])
        
        # Make multiple choices
        engine.process_choice("agent1", dilemma.selfish_choice)
        engine.process_choice("agent1", dilemma.altruistic_choice)
        
        history = engine.get_agent_choice_history("agent1")
        
        assert len(history) == 2
        assert history[0].choice_made == dilemma.selfish_choice
        assert history[1].choice_made == dilemma.altruistic_choice
        
        # Test empty history
        empty_history = engine.get_agent_choice_history("agent2")
        assert empty_history == []
    
    def test_get_choice_count(self):
        """Test getting total number of choices made by an agent."""
        dilemma = MoralDilemma.create_resource_dilemma("resource_1", "key")
        engine = MoralDilemmaEngine([dilemma])
        
        assert engine.get_choice_count("agent1") == 0
        
        engine.process_choice("agent1", dilemma.selfish_choice)
        assert engine.get_choice_count("agent1") == 1
        
        engine.process_choice("agent1", dilemma.altruistic_choice)
        assert engine.get_choice_count("agent1") == 2
    
    def test_get_selfish_and_altruistic_choice_counts(self):
        """Test counting selfish vs altruistic choices."""
        dilemma = MoralDilemma.create_resource_dilemma("resource_1", "key")
        engine = MoralDilemmaEngine([dilemma])
        
        # Make mixed choices
        engine.process_choice("agent1", dilemma.selfish_choice)
        engine.process_choice("agent1", dilemma.selfish_choice)
        engine.process_choice("agent1", dilemma.altruistic_choice)
        
        assert engine.get_selfish_choice_count("agent1") == 2
        assert engine.get_altruistic_choice_count("agent1") == 1
        assert engine.get_choice_count("agent1") == 3
    
    def test_get_moral_alignment(self):
        """Test determining moral alignment based on choice patterns."""
        dilemma = MoralDilemma.create_resource_dilemma("resource_1", "key")
        engine = MoralDilemmaEngine([dilemma])
        
        # Test neutral (no choices)
        assert engine.get_moral_alignment("agent1") == "neutral"
        
        # Test highly selfish (100% selfish)
        for _ in range(5):
            engine.process_choice("agent2", dilemma.selfish_choice)
        assert engine.get_moral_alignment("agent2") == "highly_selfish"
        
        # Test highly altruistic (100% altruistic)
        for _ in range(5):
            engine.process_choice("agent3", dilemma.altruistic_choice)
        assert engine.get_moral_alignment("agent3") == "highly_altruistic"
        
        # Test mixed (50% selfish)
        for _ in range(2):
            engine.process_choice("agent4", dilemma.selfish_choice)
        for _ in range(2):
            engine.process_choice("agent4", dilemma.altruistic_choice)
        assert engine.get_moral_alignment("agent4") == "mixed"
    
    def test_get_available_dilemmas_for_context(self):
        """Test getting all dilemmas that match a context."""
        # Create dilemmas with different contexts
        resource_dilemma = MoralDilemma.create_resource_dilemma("resource_1", "key")
        
        selfish_choice = MoralChoice(
            description="Test selfish",
            survival_benefit=0.8,
            ethical_cost=0.7,
            trust_impact={},
            consequences=[]
        )
        altruistic_choice = MoralChoice(
            description="Test altruistic",
            survival_benefit=0.2,
            ethical_cost=0.1,
            trust_impact={},
            consequences=[]
        )
        time_dilemma = MoralDilemma(
            id="time_pressure",
            description="Time running out",
            selfish_choice=selfish_choice,
            altruistic_choice=altruistic_choice,
            context_requirements={"time_remaining": 30}
        )
        
        engine = MoralDilemmaEngine([resource_dilemma, time_dilemma])
        
        # Context that matches resource dilemma only
        resource_context = {"resource_available": True}
        matching_dilemmas = engine.get_available_dilemmas_for_context(resource_context)
        assert len(matching_dilemmas) == 1
        assert matching_dilemmas[0].id == "resource_1"
        
        # Context that matches time dilemma only
        time_context = {"time_remaining": 30}
        matching_dilemmas = engine.get_available_dilemmas_for_context(time_context)
        assert len(matching_dilemmas) == 1
        assert matching_dilemmas[0].id == "time_pressure"
        
        # Context that matches neither
        no_match_context = {"unknown_condition": True}
        matching_dilemmas = engine.get_available_dilemmas_for_context(no_match_context)
        assert len(matching_dilemmas) == 0
    
    def test_get_all_agents_with_choices(self):
        """Test getting list of all agents who have made choices."""
        dilemma = MoralDilemma.create_resource_dilemma("resource_1", "key")
        engine = MoralDilemmaEngine([dilemma])
        
        assert engine.get_all_agents_with_choices() == []
        
        engine.process_choice("agent1", dilemma.selfish_choice)
        engine.process_choice("agent2", dilemma.altruistic_choice)
        
        agents = engine.get_all_agents_with_choices()
        assert set(agents) == {"agent1", "agent2"}
    
    def test_get_ethical_burden_ranking(self):
        """Test getting agents ranked by ethical burden."""
        dilemma = MoralDilemma.create_resource_dilemma("resource_1", "key")
        engine = MoralDilemmaEngine([dilemma])
        
        # Create agents with different ethical burdens
        engine.process_choice("agent1", dilemma.selfish_choice)   # 0.7
        engine.process_choice("agent1", dilemma.selfish_choice)   # 0.7 more = 1.4 total
        
        engine.process_choice("agent2", dilemma.altruistic_choice)  # 0.1
        
        engine.process_choice("agent3", dilemma.selfish_choice)   # 0.7
        engine.process_choice("agent3", dilemma.altruistic_choice)  # 0.1 more = 0.8 total
        
        ranking = engine.get_ethical_burden_ranking()
        
        # Should be sorted by burden (highest first)
        assert ranking[0][0] == "agent1"  # 1.4
        assert ranking[1][0] == "agent3"  # 0.8
        assert ranking[2][0] == "agent2"  # 0.1
        
        assert abs(ranking[0][1] - 1.4) < 1e-10
        assert abs(ranking[1][1] - 0.8) < 1e-10
        assert abs(ranking[2][1] - 0.1) < 1e-10
    
    def test_reset_agent_history(self):
        """Test resetting choice history and ethical score for an agent."""
        dilemma = MoralDilemma.create_resource_dilemma("resource_1", "key")
        engine = MoralDilemmaEngine([dilemma])
        
        # Make some choices
        engine.process_choice("agent1", dilemma.selfish_choice)
        engine.process_choice("agent1", dilemma.altruistic_choice)
        
        assert engine.get_choice_count("agent1") == 2
        assert engine.calculate_ethical_burden("agent1") > 0
        
        # Reset history
        engine.reset_agent_history("agent1")
        
        assert engine.get_choice_count("agent1") == 0
        assert engine.calculate_ethical_burden("agent1") == 0.0
        assert engine.get_agent_choice_history("agent1") == []
    
    def test_get_engine_statistics(self):
        """Test getting comprehensive engine statistics."""
        dilemma1 = MoralDilemma.create_resource_dilemma("resource_1", "key")
        dilemma2 = MoralDilemma.create_resource_dilemma("resource_2", "tool")
        engine = MoralDilemmaEngine([dilemma1, dilemma2])
        
        # Make various choices
        engine.process_choice("agent1", dilemma1.selfish_choice)
        engine.process_choice("agent1", dilemma1.altruistic_choice)
        engine.process_choice("agent2", dilemma1.selfish_choice)
        engine.process_choice("agent2", dilemma1.selfish_choice)
        
        stats = engine.get_engine_statistics()
        
        assert stats["total_dilemmas_available"] == 2
        assert stats["total_agents_with_choices"] == 2
        assert stats["total_choices_made"] == 4
        assert stats["total_selfish_choices"] == 3
        assert stats["total_altruistic_choices"] == 1
        assert stats["average_choices_per_agent"] == 2.0
        assert stats["selfish_choice_ratio"] == 0.75


class TestMoralDilemmaEngineEdgeCases:
    """Test edge cases and error handling for MoralDilemmaEngine."""
    
    def test_multiple_dilemmas_same_context(self):
        """Test handling multiple dilemmas with same context requirements."""
        dilemma1 = MoralDilemma.create_resource_dilemma("resource_1", "key")
        dilemma2 = MoralDilemma.create_resource_dilemma("resource_2", "tool")
        
        engine = MoralDilemmaEngine([dilemma1, dilemma2])
        
        context = {"resource_available": True}
        
        # Should return first matching dilemma
        presented = engine.present_dilemma("agent1", context)
        assert presented.id == "resource_1"
        
        # Should list both as available
        available = engine.get_available_dilemmas_for_context(context)
        assert len(available) == 2
        assert {d.id for d in available} == {"resource_1", "resource_2"}
    
    def test_agent_validation_for_all_methods(self):
        """Test that all methods properly validate agent IDs."""
        engine = MoralDilemmaEngine([])
        
        methods_to_test = [
            ("get_agent_choice_history", []),
            ("get_choice_count", []),
            ("get_selfish_choice_count", []),
            ("get_altruistic_choice_count", []),
            ("get_moral_alignment", []),
            ("reset_agent_history", []),
        ]
        
        for method_name, args in methods_to_test:
            method = getattr(engine, method_name)
            
            with pytest.raises(ValueError, match="Agent ID cannot be empty"):
                method("", *args)
            
            with pytest.raises(ValueError, match="Agent ID cannot be empty"):
                method("   ", *args)
    
    def test_context_validation(self):
        """Test context validation for relevant methods."""
        engine = MoralDilemmaEngine([])
        
        with pytest.raises(ValueError, match="Context cannot be None"):
            engine.get_available_dilemmas_for_context(None)
    
    def test_engine_with_no_dilemmas(self):
        """Test engine behavior with empty dilemma list."""
        engine = MoralDilemmaEngine([])
        
        # Should handle empty dilemma list gracefully
        assert engine.present_dilemma("agent1", {"any_context": True}) is None
        assert engine.get_available_dilemmas_for_context({"any_context": True}) == []
        
        stats = engine.get_engine_statistics()
        assert stats["total_dilemmas_available"] == 0
        assert stats["total_agents_with_choices"] == 0
    
    def test_complex_multi_agent_scenario(self):
        """Test complex scenario with multiple agents making various choices."""
        dilemma = MoralDilemma.create_resource_dilemma("resource_1", "key")
        engine = MoralDilemmaEngine([dilemma])
        
        # Simulate complex competitive scenario
        agents = ["strategist", "mediator", "survivor"]
        
        # Strategist: mostly selfish
        for _ in range(4):
            engine.process_choice("strategist", dilemma.selfish_choice)
        engine.process_choice("strategist", dilemma.altruistic_choice)
        
        # Mediator: balanced
        for _ in range(2):
            engine.process_choice("mediator", dilemma.selfish_choice)
        for _ in range(2):
            engine.process_choice("mediator", dilemma.altruistic_choice)
        
        # Survivor: mostly altruistic
        engine.process_choice("survivor", dilemma.selfish_choice)
        for _ in range(4):
            engine.process_choice("survivor", dilemma.altruistic_choice)
        
        # Test various metrics
        assert engine.get_moral_alignment("strategist") == "highly_selfish"  # 4/5 = 80% selfish
        assert engine.get_moral_alignment("mediator") == "mixed"
        assert engine.get_moral_alignment("survivor") == "altruistic"
        
        # Test ranking
        ranking = engine.get_ethical_burden_ranking()
        strategist_burden = next(burden for agent, burden in ranking if agent == "strategist")
        survivor_burden = next(burden for agent, burden in ranking if agent == "survivor")
        
        assert strategist_burden > survivor_burden
        
        # Test comprehensive statistics
        stats = engine.get_engine_statistics()
        assert stats["total_agents_with_choices"] == 3
        assert stats["total_choices_made"] == 14  # 5 + 4 + 5
    
    def test_choice_consequences_timestamp_accuracy(self):
        """Test that choice consequences have accurate timestamps."""
        dilemma = MoralDilemma.create_resource_dilemma("resource_1", "key")
        engine = MoralDilemmaEngine([dilemma])
        
        start_time = datetime.now()
        
        consequences1 = engine.process_choice("agent1", dilemma.selfish_choice)
        
        # Small delay to ensure different timestamps
        import time
        time.sleep(0.001)
        
        consequences2 = engine.process_choice("agent1", dilemma.altruistic_choice)
        
        end_time = datetime.now()
        
        # Timestamps should be within the time window
        assert start_time <= consequences1.timestamp <= end_time
        assert start_time <= consequences2.timestamp <= end_time
        
        # Second choice should have later timestamp
        assert consequences2.timestamp >= consequences1.timestamp
    
    def test_moral_alignment_edge_cases(self):
        """Test moral alignment calculation with edge cases."""
        dilemma = MoralDilemma.create_resource_dilemma("resource_1", "key")
        engine = MoralDilemmaEngine([dilemma])
        
        # Test exact boundary cases
        # 80% selfish = highly_selfish
        for _ in range(4):
            engine.process_choice("agent1", dilemma.selfish_choice)
        engine.process_choice("agent1", dilemma.altruistic_choice)
        assert engine.get_moral_alignment("agent1") == "highly_selfish"
        
        # 60% selfish = selfish
        for _ in range(3):
            engine.process_choice("agent2", dilemma.selfish_choice)
        for _ in range(2):
            engine.process_choice("agent2", dilemma.altruistic_choice)
        assert engine.get_moral_alignment("agent2") == "selfish"