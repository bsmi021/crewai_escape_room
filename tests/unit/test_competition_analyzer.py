"""
Unit tests for CompetitionAnalyzer class using TDD methodology.
Tests for comprehensive competition analysis and metrics including agent behavior,
survival strategies, cooperation vs betrayal patterns, trust evolution tracking,
personality consistency measurement, and analysis accuracy validation.

This implements Task #12: Build competition analysis and metrics using TDD
Requirements: 8.1, 8.2, 8.3, 8.4, 8.5 (agent behavior analysis and personality consistency)
"""
import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, MagicMock
from typing import Dict, List, Any, Optional

# These imports will fail initially - that's the point of TDD
try:
    from src.escape_room_sim.competitive.competition_analyzer import CompetitionAnalyzer
except ImportError:
    CompetitionAnalyzer = None

from src.escape_room_sim.competitive.models import (
    TrustRelationship, TrustAction, MoralChoice, ChoiceConsequences,
    EscapeResult, ClaimResult, CompetitiveScenario
)
from src.escape_room_sim.competitive.competitive_simulation import (
    CompetitiveSimulation, SimulationResults
)
from src.escape_room_sim.competitive.trust_tracker import TrustTracker


class TestCompetitionAnalyzerInitialization:
    """Test CompetitionAnalyzer class initialization - Phase 1 of TDD."""
    
    @pytest.mark.skipif(CompetitionAnalyzer is None, reason="CompetitionAnalyzer not implemented yet")
    def test_competition_analyzer_initializes_empty(self):
        """Test that CompetitionAnalyzer initializes with empty state."""
        # This should fail initially - CompetitionAnalyzer doesn't exist yet
        analyzer = CompetitionAnalyzer()
        
        assert analyzer.simulation_results == []
        assert analyzer.analysis_cache == {}
        assert analyzer.is_analyzed is False
        assert analyzer.accuracy_threshold == 0.95  # Default accuracy threshold
    
    @pytest.mark.skipif(CompetitionAnalyzer is None, reason="CompetitionAnalyzer not implemented yet")
    def test_competition_analyzer_initializes_with_results(self):
        """Test that CompetitionAnalyzer initializes with simulation results."""
        mock_results = [Mock(spec=SimulationResults)]
        analyzer = CompetitionAnalyzer(simulation_results=mock_results)
        
        assert analyzer.simulation_results == mock_results
        assert len(analyzer.simulation_results) == 1
        assert analyzer.is_analyzed is False
    
    @pytest.mark.skipif(CompetitionAnalyzer is None, reason="CompetitionAnalyzer not implemented yet")
    def test_competition_analyzer_validates_accuracy_threshold(self):
        """Test that CompetitionAnalyzer validates accuracy threshold parameter."""
        # Should fail - accuracy threshold validation not implemented
        with pytest.raises(ValueError, match="Accuracy threshold must be between 0.0 and 1.0"):
            CompetitionAnalyzer(accuracy_threshold=-0.1)
        
        with pytest.raises(ValueError, match="Accuracy threshold must be between 0.0 and 1.0"):
            CompetitionAnalyzer(accuracy_threshold=1.1)
        
        with pytest.raises(ValueError, match="Accuracy threshold must be between 0.0 and 1.0"):
            CompetitionAnalyzer(accuracy_threshold="invalid")
    
    @pytest.mark.skipif(CompetitionAnalyzer is None, reason="CompetitionAnalyzer not implemented yet")
    def test_competition_analyzer_initializes_analysis_structures(self):
        """Test that CompetitionAnalyzer initializes all internal analysis structures."""
        analyzer = CompetitionAnalyzer()
        
        # Should fail - internal structures not implemented
        assert hasattr(analyzer, 'survival_strategies')
        assert hasattr(analyzer, 'cooperation_patterns')
        assert hasattr(analyzer, 'betrayal_patterns')
        assert hasattr(analyzer, 'trust_evolution_data')
        assert hasattr(analyzer, 'personality_consistency_scores')
        
        # All should start empty
        assert analyzer.survival_strategies == {}
        assert analyzer.cooperation_patterns == {}
        assert analyzer.betrayal_patterns == {}
        assert analyzer.trust_evolution_data == {}
        assert analyzer.personality_consistency_scores == {}


class TestSurvivalStrategyIdentification:
    """Test survival strategy identification and categorization - Phase 2 of TDD."""
    
    @pytest.mark.skipif(CompetitionAnalyzer is None, reason="CompetitionAnalyzer not implemented yet")
    def test_identify_survival_strategies_basic(self):
        """Test basic survival strategy identification from simulation data."""
        analyzer = CompetitionAnalyzer()
        
        # Mock simulation results with different agent behaviors
        mock_results = self._create_mock_simulation_results_with_strategies()
        analyzer.add_simulation_results(mock_results)
        
        # Should fail - identify_survival_strategies not implemented
        strategies = analyzer.identify_survival_strategies()
        
        assert isinstance(strategies, dict)
        assert "strategist" in strategies
        assert "mediator" in strategies
        assert "survivor" in strategies
        
        # Check strategy categorization
        assert strategies["strategist"]["primary_strategy"] in [
            "resource_hoarding", "alliance_forming", "information_gathering", "competitive"
        ]
        assert strategies["mediator"]["primary_strategy"] in [
            "cooperation_building", "trust_maintaining", "conflict_resolution", "diplomatic"
        ]
        assert strategies["survivor"]["primary_strategy"] in [
            "risk_minimizing", "opportunistic", "adaptive", "pragmatic"
        ]
    
    @pytest.mark.skipif(CompetitionAnalyzer is None, reason="CompetitionAnalyzer not implemented yet")
    def test_categorize_survival_strategies_by_effectiveness(self):
        """Test categorization of survival strategies by their effectiveness."""
        analyzer = CompetitionAnalyzer()
        mock_results = self._create_mock_simulation_results_with_outcomes()
        analyzer.add_simulation_results(mock_results)
        
        # Should fail - strategy effectiveness analysis not implemented
        effectiveness = analyzer.analyze_strategy_effectiveness()
        
        assert isinstance(effectiveness, dict)
        assert "strategy_rankings" in effectiveness
        assert "win_rates_by_strategy" in effectiveness
        assert "average_survival_time" in effectiveness
        
        # Check win rate calculations
        win_rates = effectiveness["win_rates_by_strategy"]
        for strategy, rate in win_rates.items():
            assert 0.0 <= rate <= 1.0
    
    @pytest.mark.skipif(CompetitionAnalyzer is None, reason="CompetitionAnalyzer not implemented yet")
    def test_identify_strategy_transitions_over_time(self):
        """Test identification of how strategies change over simulation time."""
        analyzer = CompetitionAnalyzer()
        mock_results = self._create_mock_simulation_with_strategy_changes()
        analyzer.add_simulation_results(mock_results)
        
        # Should fail - strategy transition analysis not implemented
        transitions = analyzer.analyze_strategy_transitions()
        
        assert isinstance(transitions, dict)
        assert "transition_patterns" in transitions
        assert "pressure_response" in transitions
        assert "adaptation_speed" in transitions
        
        # Check that transitions are tracked per agent
        for agent_id in ["strategist", "mediator", "survivor"]:
            assert agent_id in transitions["transition_patterns"]
    
    @pytest.mark.skipif(CompetitionAnalyzer is None, reason="CompetitionAnalyzer not implemented yet")
    def test_validate_strategy_identification_accuracy(self):
        """Test validation of strategy identification accuracy."""
        analyzer = CompetitionAnalyzer()
        mock_results = self._create_mock_labeled_simulation_results()
        analyzer.add_simulation_results(mock_results)
        
        # Should fail - accuracy validation not implemented
        accuracy_report = analyzer.validate_strategy_identification_accuracy()
        
        assert isinstance(accuracy_report, dict)
        assert "overall_accuracy" in accuracy_report
        assert "per_agent_accuracy" in accuracy_report
        assert "confusion_matrix" in accuracy_report
        
        # Check accuracy meets threshold
        assert accuracy_report["overall_accuracy"] >= analyzer.accuracy_threshold
    
    def _create_mock_simulation_results_with_strategies(self) -> List[SimulationResults]:
        """Create mock simulation results with different agent strategies."""
        results = []
        for i in range(3):
            result = Mock(spec=SimulationResults)
            result.seed = i
            result.winner = ["strategist", "mediator", "survivor"][i % 3]
            result.action_history = self._create_mock_actions_for_strategy(result.winner)
            result.final_states = self._create_mock_final_states()
            results.append(result)
        return results
    
    def _create_mock_simulation_results_with_outcomes(self) -> List[SimulationResults]:
        """Create mock simulation results with outcome data for effectiveness analysis."""
        results = []
        strategies = ["resource_hoarding", "cooperation_building", "risk_minimizing"]
        for i, strategy in enumerate(strategies):
            result = Mock(spec=SimulationResults)
            result.seed = i
            result.winner = ["strategist", "mediator", "survivor"][i]
            result.completion_reason = "escape_successful"
            result.simulation_duration = 300 + (i * 100)  # Different durations
            result.action_history = self._create_mock_actions_for_strategy_type(strategy)
            results.append(result)
        return results
    
    def _create_mock_simulation_with_strategy_changes(self) -> List[SimulationResults]:
        """Create mock simulation with strategy transitions over time."""
        result = Mock(spec=SimulationResults)
        result.seed = 42
        result.winner = "strategist"
        result.total_steps = 50
        
        # Create action history showing strategy changes
        action_history = []
        for step in range(50):
            if step < 15:
                strategy_type = "cooperative"
            elif step < 35:
                strategy_type = "competitive"
            else:
                strategy_type = "desperate"
            
            action = {
                "agent": "strategist",
                "action": "claim_resource" if strategy_type == "competitive" else "share_information",
                "timestamp": datetime.now() + timedelta(seconds=step),
                "step": step
            }
            action_history.append(action)
        
        result.action_history = action_history
        return [result]
    
    def _create_mock_labeled_simulation_results(self) -> List[SimulationResults]:
        """Create mock simulation results with ground truth strategy labels."""
        results = []
        labels = {
            "strategist": "resource_hoarding",
            "mediator": "cooperation_building", 
            "survivor": "risk_minimizing"
        }
        
        for agent, true_strategy in labels.items():
            result = Mock(spec=SimulationResults)
            result.seed = hash(agent) % 1000
            result.winner = agent
            result.ground_truth_strategies = {agent: true_strategy}
            result.action_history = self._create_mock_actions_for_strategy_type(true_strategy)
            results.append(result)
        
        return results
    
    def _create_mock_actions_for_strategy(self, agent: str) -> List[Dict[str, Any]]:
        """Create mock action history for specific agent strategy."""
        if agent == "strategist":
            return [
                {"agent": "strategist", "action": "claim_resource", "success": True},
                {"agent": "strategist", "action": "claim_resource", "success": True},
                {"agent": "strategist", "action": "attempt_escape", "success": True}
            ]
        elif agent == "mediator":
            return [
                {"agent": "mediator", "action": "share_information", "target": "strategist"},
                {"agent": "mediator", "action": "share_resource", "target": "survivor"},
                {"agent": "mediator", "action": "attempt_escape", "success": True}
            ]
        else:  # survivor
            return [
                {"agent": "survivor", "action": "claim_resource", "success": False},
                {"agent": "survivor", "action": "share_information", "target": "mediator"},
                {"agent": "survivor", "action": "attempt_escape", "success": True}
            ]
    
    def _create_mock_actions_for_strategy_type(self, strategy: str) -> List[Dict[str, Any]]:
        """Create mock actions matching a specific strategy type."""
        base_actions = []
        if strategy == "resource_hoarding":
            base_actions = [{"action": "claim_resource"} for _ in range(5)]
        elif strategy == "cooperation_building":
            base_actions = [{"action": "share_information"}, {"action": "share_resource"}] * 3
        elif strategy == "risk_minimizing":
            base_actions = [{"action": "share_information"}, {"action": "claim_resource"}] * 2
        
        # Add agent and timestamp data
        for i, action in enumerate(base_actions):
            action["agent"] = "test_agent"
            action["timestamp"] = datetime.now() + timedelta(seconds=i)
        
        return base_actions
    
    def _create_mock_final_states(self) -> Dict[str, Dict[str, Any]]:
        """Create mock final agent states."""
        return {
            "strategist": {"resources_owned": ["key1", "tool1"], "trust_level": 0.2},
            "mediator": {"resources_owned": ["info1"], "trust_level": 0.8},
            "survivor": {"resources_owned": [], "trust_level": 0.5}
        }


class TestCooperationVsBetrayalAnalysis:
    """Test cooperation vs betrayal pattern analysis - Phase 3 of TDD."""
    
    @pytest.mark.skipif(CompetitionAnalyzer is None, reason="CompetitionAnalyzer not implemented yet")
    def test_analyze_cooperation_patterns_basic(self):
        """Test basic cooperation pattern analysis."""
        analyzer = CompetitionAnalyzer()
        mock_results = self._create_mock_cooperation_data()
        analyzer.add_simulation_results(mock_results)
        
        # Should fail - analyze_cooperation_patterns not implemented
        cooperation_analysis = analyzer.analyze_cooperation_patterns()
        
        assert isinstance(cooperation_analysis, dict)
        assert "total_cooperation_attempts" in cooperation_analysis
        assert "cooperation_success_rate" in cooperation_analysis
        assert "most_cooperative_agent" in cooperation_analysis
        assert "cooperation_triggers" in cooperation_analysis
        
        # Validate cooperation metrics
        assert cooperation_analysis["cooperation_success_rate"] >= 0.0
        assert cooperation_analysis["cooperation_success_rate"] <= 1.0
    
    @pytest.mark.skipif(CompetitionAnalyzer is None, reason="CompetitionAnalyzer not implemented yet")
    def test_analyze_betrayal_patterns_basic(self):
        """Test basic betrayal pattern analysis."""
        analyzer = CompetitionAnalyzer()
        mock_results = self._create_mock_betrayal_data()
        analyzer.add_simulation_results(mock_results)
        
        # Should fail - analyze_betrayal_patterns not implemented
        betrayal_analysis = analyzer.analyze_betrayal_patterns()
        
        assert isinstance(betrayal_analysis, dict)
        assert "total_betrayal_incidents" in betrayal_analysis
        assert "betrayal_triggers" in betrayal_analysis
        assert "most_betraying_agent" in betrayal_analysis
        assert "betrayal_consequences" in betrayal_analysis
        
        # Validate betrayal metrics
        assert betrayal_analysis["total_betrayal_incidents"] >= 0
    
    @pytest.mark.skipif(CompetitionAnalyzer is None, reason="CompetitionAnalyzer not implemented yet")
    def test_compare_cooperation_vs_betrayal_outcomes(self):
        """Test comparison of cooperation vs betrayal outcomes."""
        analyzer = CompetitionAnalyzer()
        mock_results = self._create_mock_mixed_behavior_data()
        analyzer.add_simulation_results(mock_results)
        
        # Should fail - cooperation vs betrayal comparison not implemented
        comparison = analyzer.compare_cooperation_vs_betrayal_outcomes()
        
        assert isinstance(comparison, dict)
        assert "cooperation_win_rate" in comparison
        assert "betrayal_win_rate" in comparison
        assert "mixed_strategy_win_rate" in comparison
        assert "outcome_effectiveness" in comparison
        
        # Check that win rates sum appropriately
        total_rate = (comparison["cooperation_win_rate"] + 
                     comparison["betrayal_win_rate"] + 
                     comparison["mixed_strategy_win_rate"])
        assert abs(total_rate - 1.0) < 0.01  # Allow for rounding
    
    @pytest.mark.skipif(CompetitionAnalyzer is None, reason="CompetitionAnalyzer not implemented yet")
    def test_identify_cooperation_betrayal_triggers(self):
        """Test identification of what triggers cooperation vs betrayal."""
        analyzer = CompetitionAnalyzer()
        mock_results = self._create_mock_trigger_scenario_data()
        analyzer.add_simulation_results(mock_results)
        
        # Should fail - trigger identification not implemented
        triggers = analyzer.identify_cooperation_betrayal_triggers()
        
        assert isinstance(triggers, dict)
        assert "cooperation_triggers" in triggers
        assert "betrayal_triggers" in triggers
        assert "contextual_factors" in triggers
        
        # Check for common trigger types
        coop_triggers = triggers["cooperation_triggers"]
        betrayal_triggers = triggers["betrayal_triggers"]
        
        expected_coop_triggers = ["high_trust", "resource_abundance", "early_game"]
        expected_betrayal_triggers = ["low_trust", "resource_scarcity", "time_pressure"]
        
        for trigger in expected_coop_triggers:
            assert trigger in coop_triggers or any(trigger in str(t) for t in coop_triggers.keys())
        
        for trigger in expected_betrayal_triggers:
            assert trigger in betrayal_triggers or any(trigger in str(t) for t in betrayal_triggers.keys())
    
    def _create_mock_cooperation_data(self) -> List[SimulationResults]:
        """Create mock data with cooperation patterns."""
        result = Mock(spec=SimulationResults)
        result.seed = 123
        result.action_history = [
            {"agent": "mediator", "action": "share_information", "target": "strategist", "success": True},
            {"agent": "strategist", "action": "share_resource", "target": "survivor", "success": True},
            {"agent": "survivor", "action": "share_information", "target": "mediator", "success": False}
        ]
        result.competition_metrics = {
            "cooperation_attempts": 3,
            "betrayal_incidents": 0,
            "trust_evolution": {"mediator": 0.8, "strategist": 0.6, "survivor": 0.4}
        }
        return [result]
    
    def _create_mock_betrayal_data(self) -> List[SimulationResults]:
        """Create mock data with betrayal patterns."""
        result = Mock(spec=SimulationResults)
        result.seed = 456
        result.action_history = [
            {"agent": "strategist", "action": "claim_resource", "target_resource": "key", "contested": True},
            {"agent": "survivor", "action": "betray_alliance", "target": "mediator", "trust_damage": -0.7}
        ]
        result.competition_metrics = {
            "cooperation_attempts": 1,
            "betrayal_incidents": 2,
            "trust_evolution": {"strategist": -0.3, "survivor": -0.5, "mediator": 0.1}
        }
        return [result]
    
    def _create_mock_mixed_behavior_data(self) -> List[SimulationResults]:
        """Create mock data with mixed cooperation and betrayal."""
        results = []
        behaviors = ["cooperation", "betrayal", "mixed"]
        winners = ["mediator", "strategist", "survivor"]
        
        for i, (behavior, winner) in enumerate(zip(behaviors, winners)):
            result = Mock(spec=SimulationResults)
            result.seed = 100 + i
            result.winner = winner
            result.primary_strategy = behavior
            result.action_history = self._create_actions_for_behavior(behavior)
            results.append(result)
        
        return results
    
    def _create_mock_trigger_scenario_data(self) -> List[SimulationResults]:
        """Create mock data with identifiable cooperation/betrayal triggers."""
        results = []
        
        # High trust -> cooperation scenario
        result1 = Mock(spec=SimulationResults)
        result1.seed = 200
        result1.action_history = [
            {"agent": "mediator", "action": "share_information", "context": {"trust_level": 0.9, "resources": "abundant"}}
        ]
        result1.initial_trust_levels = {"mediator": 0.9}
        results.append(result1)
        
        # Low trust -> betrayal scenario  
        result2 = Mock(spec=SimulationResults)
        result2.seed = 201
        result2.action_history = [
            {"agent": "strategist", "action": "betray_alliance", "context": {"trust_level": 0.1, "time_remaining": 30}}
        ]
        result2.initial_trust_levels = {"strategist": 0.1}
        results.append(result2)
        
        # Resource scarcity -> betrayal scenario
        result3 = Mock(spec=SimulationResults)
        result3.seed = 202
        result3.action_history = [
            {"agent": "survivor", "action": "claim_resource", "target_resource": "key", "contested": True, "context": {"resources": "scarce"}}
        ]
        results.append(result3)
        
        return results
    
    def _create_actions_for_behavior(self, behavior: str) -> List[Dict[str, Any]]:
        """Create action history for specific behavior type."""
        if behavior == "cooperation":
            return [
                {"action": "share_information", "success": True},
                {"action": "share_resource", "success": True}
            ]
        elif behavior == "betrayal":
            return [
                {"action": "claim_resource", "contested": True},
                {"action": "betray_alliance", "trust_damage": -0.5}
            ]
        else:  # mixed
            return [
                {"action": "share_information", "success": True},
                {"action": "claim_resource", "contested": True}
            ]


class TestTrustEvolutionTracking:
    """Test trust evolution tracking across simulation iterations - Phase 4 of TDD."""
    
    @pytest.mark.skipif(CompetitionAnalyzer is None, reason="CompetitionAnalyzer not implemented yet")
    def test_track_trust_evolution_over_time(self):
        """Test tracking trust evolution throughout simulation."""
        analyzer = CompetitionAnalyzer()
        mock_results = self._create_mock_trust_evolution_data()
        analyzer.add_simulation_results(mock_results)
        
        # Should fail - track_trust_evolution not implemented
        trust_evolution = analyzer.track_trust_evolution()
        
        assert isinstance(trust_evolution, dict)
        assert "trust_trajectories" in trust_evolution
        assert "trust_volatility" in trust_evolution
        assert "final_trust_levels" in trust_evolution
        assert "trust_turning_points" in trust_evolution
        
        # Check trajectories for each agent pair
        trajectories = trust_evolution["trust_trajectories"]
        expected_pairs = [("strategist", "mediator"), ("strategist", "survivor"), ("mediator", "survivor")]
        for pair in expected_pairs:
            pair_key = f"{pair[0]}-{pair[1]}"
            assert pair_key in trajectories or f"{pair[1]}-{pair[0]}" in trajectories
    
    @pytest.mark.skipif(CompetitionAnalyzer is None, reason="CompetitionAnalyzer not implemented yet")
    def test_identify_trust_turning_points(self):
        """Test identification of critical trust turning points."""
        analyzer = CompetitionAnalyzer()
        mock_results = self._create_mock_trust_turning_points_data()
        analyzer.add_simulation_results(mock_results)
        
        # Should fail - identify_trust_turning_points not implemented
        turning_points = analyzer.identify_trust_turning_points()
        
        assert isinstance(turning_points, dict)
        assert "major_trust_changes" in turning_points
        assert "trust_recovery_events" in turning_points
        assert "point_of_no_return" in turning_points
        
        # Validate turning point structure
        major_changes = turning_points["major_trust_changes"]
        assert isinstance(major_changes, list)
        
        for change_event in major_changes:
            assert "timestamp" in change_event
            assert "agents_involved" in change_event
            assert "trust_change_magnitude" in change_event
            assert "triggering_action" in change_event
    
    @pytest.mark.skipif(CompetitionAnalyzer is None, reason="CompetitionAnalyzer not implemented yet")
    def test_calculate_trust_volatility_metrics(self):
        """Test calculation of trust volatility and stability metrics."""
        analyzer = CompetitionAnalyzer()
        mock_results = self._create_mock_trust_volatility_data()
        analyzer.add_simulation_results(mock_results)
        
        # Should fail - calculate_trust_volatility_metrics not implemented
        volatility_metrics = analyzer.calculate_trust_volatility_metrics()
        
        assert isinstance(volatility_metrics, dict)
        assert "overall_volatility" in volatility_metrics
        assert "agent_specific_volatility" in volatility_metrics
        assert "relationship_stability_scores" in volatility_metrics
        assert "trust_prediction_accuracy" in volatility_metrics
        
        # Validate volatility values
        assert 0.0 <= volatility_metrics["overall_volatility"] <= 1.0
        
        agent_volatility = volatility_metrics["agent_specific_volatility"]
        for agent, volatility in agent_volatility.items():
            assert 0.0 <= volatility <= 1.0
    
    @pytest.mark.skipif(CompetitionAnalyzer is None, reason="CompetitionAnalyzer not implemented yet")
    def test_analyze_trust_recovery_patterns(self):
        """Test analysis of trust recovery and repair patterns."""
        analyzer = CompetitionAnalyzer()
        mock_results = self._create_mock_trust_recovery_data()
        analyzer.add_simulation_results(mock_results)
        
        # Should fail - analyze_trust_recovery_patterns not implemented
        recovery_analysis = analyzer.analyze_trust_recovery_patterns()
        
        assert isinstance(recovery_analysis, dict)
        assert "recovery_success_rate" in recovery_analysis
        assert "recovery_time_analysis" in recovery_analysis
        assert "recovery_strategies" in recovery_analysis
        assert "irreversible_damage_threshold" in recovery_analysis
        
        # Validate recovery metrics
        success_rate = recovery_analysis["recovery_success_rate"]
        assert 0.0 <= success_rate <= 1.0
        
        threshold = recovery_analysis["irreversible_damage_threshold"]
        assert -1.0 <= threshold <= 0.0  # Should be negative trust level
    
    def _create_mock_trust_evolution_data(self) -> List[SimulationResults]:
        """Create mock data showing trust evolution over time."""
        result = Mock(spec=SimulationResults)
        result.seed = 300
        
        # Create trust evolution timeline
        trust_timeline = []
        base_time = datetime.now()
        
        # Simulate trust changes over 10 time steps
        trust_levels = {"strategist-mediator": 0.0, "strategist-survivor": 0.0, "mediator-survivor": 0.0}
        
        for step in range(10):
            # Simulate some trust changes
            if step == 3:  # cooperation event
                trust_levels["strategist-mediator"] += 0.3
                trust_levels["mediator-survivor"] += 0.2
            elif step == 7:  # betrayal event
                trust_levels["strategist-survivor"] -= 0.5
                trust_levels["mediator-survivor"] -= 0.1
            
            trust_snapshot = {
                "timestamp": base_time + timedelta(seconds=step * 30),
                "step": step,
                "trust_levels": dict(trust_levels)  # Copy current levels
            }
            trust_timeline.append(trust_snapshot)
        
        result.trust_evolution_timeline = trust_timeline
        result.final_trust_levels = trust_levels
        
        return [result]
    
    def _create_mock_trust_turning_points_data(self) -> List[SimulationResults]:
        """Create mock data with clear trust turning points."""
        result = Mock(spec=SimulationResults)
        result.seed = 301
        result.action_history = [
            {
                "timestamp": datetime.now(),
                "agent": "strategist",
                "action": "share_resource",
                "target": "mediator",
                "trust_change": 0.4
            },
            {
                "timestamp": datetime.now() + timedelta(minutes=5),
                "agent": "survivor", 
                "action": "betray_alliance",
                "target": "mediator",
                "trust_change": -0.8
            },
            {
                "timestamp": datetime.now() + timedelta(minutes=10),
                "agent": "mediator",
                "action": "forgive_betrayal",
                "target": "survivor",
                "trust_change": 0.3
            }
        ]
        return [result]
    
    def _create_mock_trust_volatility_data(self) -> List[SimulationResults]:
        """Create mock data for trust volatility analysis."""
        result = Mock(spec=SimulationResults)
        result.seed = 302
        
        # Create volatile trust data
        trust_changes = [
            {"agents": ("strategist", "mediator"), "change": 0.2, "timestamp": datetime.now()},
            {"agents": ("strategist", "mediator"), "change": -0.3, "timestamp": datetime.now() + timedelta(minutes=1)},
            {"agents": ("strategist", "mediator"), "change": 0.1, "timestamp": datetime.now() + timedelta(minutes=2)},
            {"agents": ("strategist", "mediator"), "change": -0.4, "timestamp": datetime.now() + timedelta(minutes=3)},
        ]
        
        result.trust_change_history = trust_changes
        return [result]
    
    def _create_mock_trust_recovery_data(self) -> List[SimulationResults]:
        """Create mock data showing trust recovery scenarios."""
        results = []
        
        # Successful recovery scenario
        result1 = Mock(spec=SimulationResults)
        result1.seed = 303
        result1.trust_damage_events = [
            {"agents": ("strategist", "mediator"), "damage": -0.6, "timestamp": datetime.now()}
        ]
        result1.trust_recovery_events = [
            {"agents": ("strategist", "mediator"), "recovery": 0.4, "timestamp": datetime.now() + timedelta(minutes=5)},
            {"agents": ("strategist", "mediator"), "recovery": 0.3, "timestamp": datetime.now() + timedelta(minutes=10)}
        ]
        result1.final_trust_levels = {"strategist-mediator": 0.1}  # Net positive recovery
        results.append(result1)
        
        # Failed recovery scenario
        result2 = Mock(spec=SimulationResults)
        result2.seed = 304
        result2.trust_damage_events = [
            {"agents": ("survivor", "strategist"), "damage": -0.9, "timestamp": datetime.now()}
        ]
        result2.trust_recovery_events = [
            {"agents": ("survivor", "strategist"), "recovery": 0.1, "timestamp": datetime.now() + timedelta(minutes=3)}
        ]
        result2.final_trust_levels = {"survivor-strategist": -0.8}  # Failed recovery
        results.append(result2)
        
        return results


class TestPersonalityConsistencyMeasurement:
    """Test personality consistency measurement under pressure - Phase 5 of TDD."""
    
    @pytest.mark.skipif(CompetitionAnalyzer is None, reason="CompetitionAnalyzer not implemented yet")
    def test_measure_personality_consistency_basic(self):
        """Test basic personality consistency measurement."""
        analyzer = CompetitionAnalyzer()
        mock_results = self._create_mock_personality_data()
        analyzer.add_simulation_results(mock_results)
        
        # Should fail - measure_personality_consistency not implemented
        consistency_scores = analyzer.measure_personality_consistency()
        
        assert isinstance(consistency_scores, dict)
        assert "strategist" in consistency_scores
        assert "mediator" in consistency_scores  
        assert "survivor" in consistency_scores
        
        # Check consistency score properties
        for agent, score in consistency_scores.items():
            assert isinstance(score, dict)
            assert "overall_consistency" in score
            assert "stress_consistency" in score
            assert "behavioral_deviations" in score
            assert "personality_drift" in score
            
            # Validate score ranges
            assert 0.0 <= score["overall_consistency"] <= 1.0
            assert 0.0 <= score["stress_consistency"] <= 1.0
    
    @pytest.mark.skipif(CompetitionAnalyzer is None, reason="CompetitionAnalyzer not implemented yet")
    def test_analyze_pressure_response_consistency(self):
        """Test analysis of how personality consistency changes under pressure."""
        analyzer = CompetitionAnalyzer()
        mock_results = self._create_mock_pressure_scenario_data()
        analyzer.add_simulation_results(mock_results)
        
        # Should fail - analyze_pressure_response_consistency not implemented
        pressure_analysis = analyzer.analyze_pressure_response_consistency()
        
        assert isinstance(pressure_analysis, dict)
        assert "pressure_tolerance" in pressure_analysis
        assert "consistency_degradation" in pressure_analysis
        assert "breaking_points" in pressure_analysis
        assert "recovery_patterns" in pressure_analysis
        
        # Check pressure tolerance for each agent
        tolerance = pressure_analysis["pressure_tolerance"]
        for agent in ["strategist", "mediator", "survivor"]:
            assert agent in tolerance
            assert 0.0 <= tolerance[agent] <= 1.0
    
    @pytest.mark.skipif(CompetitionAnalyzer is None, reason="CompetitionAnalyzer not implemented yet")
    def test_identify_personality_drift_patterns(self):
        """Test identification of personality drift over time."""
        analyzer = CompetitionAnalyzer()
        mock_results = self._create_mock_personality_drift_data()
        analyzer.add_simulation_results(mock_results)
        
        # Should fail - identify_personality_drift_patterns not implemented
        drift_analysis = analyzer.identify_personality_drift_patterns()
        
        assert isinstance(drift_analysis, dict)
        assert "drift_magnitude" in drift_analysis
        assert "drift_direction" in drift_analysis
        assert "drift_triggers" in drift_analysis
        assert "stability_periods" in drift_analysis
        
        # Validate drift measurements
        drift_magnitude = drift_analysis["drift_magnitude"]
        for agent, magnitude in drift_magnitude.items():
            assert magnitude >= 0.0  # Drift magnitude should be non-negative
    
    @pytest.mark.skipif(CompetitionAnalyzer is None, reason="CompetitionAnalyzer not implemented yet")
    def test_validate_personality_model_accuracy(self):
        """Test validation of personality model accuracy."""
        analyzer = CompetitionAnalyzer()
        mock_results = self._create_mock_personality_validation_data()
        analyzer.add_simulation_results(mock_results)
        
        # Should fail - validate_personality_model_accuracy not implemented
        validation_results = analyzer.validate_personality_model_accuracy()
        
        assert isinstance(validation_results, dict)
        assert "prediction_accuracy" in validation_results
        assert "behavioral_prediction_errors" in validation_results
        assert "consistency_model_fit" in validation_results
        assert "cross_validation_scores" in validation_results
        
        # Check that accuracy meets threshold
        accuracy = validation_results["prediction_accuracy"]
        assert accuracy >= analyzer.accuracy_threshold
    
    def _create_mock_personality_data(self) -> List[SimulationResults]:
        """Create mock data for personality consistency analysis."""
        result = Mock(spec=SimulationResults)
        result.seed = 400
        
        # Define expected personality traits for each agent
        expected_traits = {
            "strategist": {"resource_focused": 0.9, "cooperation": 0.3, "risk_tolerance": 0.7},
            "mediator": {"resource_focused": 0.4, "cooperation": 0.9, "risk_tolerance": 0.5},
            "survivor": {"resource_focused": 0.6, "cooperation": 0.6, "risk_tolerance": 0.3}
        }
        
        # Create action history that reflects these traits
        result.action_history = []
        for agent, traits in expected_traits.items():
            # Create actions consistent with personality
            for i in range(5):
                if traits["cooperation"] > 0.7:
                    action = {"agent": agent, "action": "share_information", "step": i}
                elif traits["resource_focused"] > 0.7:
                    action = {"agent": agent, "action": "claim_resource", "step": i}
                else:
                    action = {"agent": agent, "action": "assess_situation", "step": i}
                
                result.action_history.append(action)
        
        result.expected_personality_traits = expected_traits
        return [result]
    
    def _create_mock_pressure_scenario_data(self) -> List[SimulationResults]:
        """Create mock data for pressure response analysis."""
        result = Mock(spec=SimulationResults)
        result.seed = 401
        
        # Create timeline with increasing pressure
        pressure_timeline = []
        base_time = datetime.now()
        
        for step in range(20):
            pressure_level = min(1.0, step / 15.0)  # Increasing pressure
            time_remaining = max(0, 300 - step * 15)  # Decreasing time
            
            pressure_event = {
                "timestamp": base_time + timedelta(seconds=step * 15),
                "step": step,
                "pressure_level": pressure_level,
                "time_remaining": time_remaining,
                "resource_scarcity": pressure_level * 0.8
            }
            pressure_timeline.append(pressure_event)
        
        result.pressure_timeline = pressure_timeline
        
        # Create actions showing personality changes under pressure
        result.action_history = []
        for step in range(20):
            pressure = min(1.0, step / 15.0)
            
            # Mediator starts cooperative but becomes competitive under pressure
            if step < 10:
                action = {"agent": "mediator", "action": "share_information", "step": step, "pressure": pressure}
            else:
                action = {"agent": "mediator", "action": "claim_resource", "step": step, "pressure": pressure}
            
            result.action_history.append(action)
        
        return [result]
    
    def _create_mock_personality_drift_data(self) -> List[SimulationResults]:
        """Create mock data showing personality drift over time."""
        result = Mock(spec=SimulationResults)
        result.seed = 402
        
        # Create personality measurements over time
        personality_timeline = []
        base_time = datetime.now()
        
        # Simulate strategist becoming more cooperative over time
        initial_cooperation = 0.3
        for step in range(15):
            # Gradual drift toward cooperation
            cooperation_level = initial_cooperation + (step * 0.04)  # +0.04 per step
            cooperation_level = min(1.0, cooperation_level)
            
            personality_snapshot = {
                "timestamp": base_time + timedelta(minutes=step * 2),
                "step": step,
                "agent": "strategist",
                "cooperation_level": cooperation_level,
                "resource_focus": 0.9 - (step * 0.02)  # Slight decrease in resource focus
            }
            personality_timeline.append(personality_snapshot)
        
        result.personality_evolution_timeline = personality_timeline
        
        # Add triggering events
        result.personality_triggers = [
            {
                "timestamp": base_time + timedelta(minutes=6),
                "event": "received_help_from_mediator",
                "impact": "increased_cooperation"
            },
            {
                "timestamp": base_time + timedelta(minutes=12),
                "event": "witnessed_betrayal",
                "impact": "decreased_trust_but_maintained_cooperation"
            }
        ]
        
        return [result]
    
    def _create_mock_personality_validation_data(self) -> List[SimulationResults]:
        """Create mock data for personality model validation."""
        results = []
        
        # Create multiple simulations with known personality outcomes
        known_outcomes = [
            {"agent": "strategist", "predicted_action": "claim_resource", "actual_action": "claim_resource"},
            {"agent": "mediator", "predicted_action": "share_information", "actual_action": "share_information"},
            {"agent": "survivor", "predicted_action": "assess_situation", "actual_action": "claim_resource"},  # Prediction error
            {"agent": "strategist", "predicted_action": "compete_for_resource", "actual_action": "compete_for_resource"},
            {"agent": "mediator", "predicted_action": "mediate_conflict", "actual_action": "mediate_conflict"}
        ]
        
        for i, outcome in enumerate(known_outcomes):
            result = Mock(spec=SimulationResults)
            result.seed = 403 + i
            result.personality_predictions = [outcome]
            result.action_history = [{"agent": outcome["agent"], "action": outcome["actual_action"]}]
            results.append(result)
        
        return results


class TestAnalysisAccuracyAndMetricValidation:
    """Test analysis accuracy and metric calculation validation - Phase 6 of TDD."""
    
    @pytest.mark.skipif(CompetitionAnalyzer is None, reason="CompetitionAnalyzer not implemented yet")
    def test_validate_metric_calculation_accuracy(self):
        """Test validation of metric calculation accuracy."""
        analyzer = CompetitionAnalyzer()
        mock_results = self._create_mock_validation_data()
        analyzer.add_simulation_results(mock_results)
        
        # Should fail - validate_metric_calculation_accuracy not implemented
        validation_report = analyzer.validate_metric_calculation_accuracy()
        
        assert isinstance(validation_report, dict)
        assert "overall_accuracy" in validation_report
        assert "metric_specific_accuracy" in validation_report
        assert "calculation_errors" in validation_report
        assert "accuracy_confidence_intervals" in validation_report
        
        # Check accuracy meets threshold
        assert validation_report["overall_accuracy"] >= analyzer.accuracy_threshold
        
        # Check metric-specific accuracy
        metric_accuracy = validation_report["metric_specific_accuracy"]
        required_metrics = ["survival_strategies", "cooperation_patterns", "trust_evolution", "personality_consistency"]
        
        for metric in required_metrics:
            assert metric in metric_accuracy
            assert metric_accuracy[metric] >= analyzer.accuracy_threshold * 0.9  # Allow slight tolerance
    
    @pytest.mark.skipif(CompetitionAnalyzer is None, reason="CompetitionAnalyzer not implemented yet")
    def test_cross_validate_analysis_results(self):
        """Test cross-validation of analysis results across different data splits."""
        analyzer = CompetitionAnalyzer()
        mock_results = self._create_mock_cross_validation_data()
        analyzer.add_simulation_results(mock_results)
        
        # Should fail - cross_validate_analysis_results not implemented
        cv_results = analyzer.cross_validate_analysis_results(k_folds=5)
        
        assert isinstance(cv_results, dict)
        assert "mean_accuracy" in cv_results
        assert "std_accuracy" in cv_results
        assert "fold_accuracies" in cv_results
        assert "consistency_score" in cv_results
        
        # Check cross-validation results
        assert len(cv_results["fold_accuracies"]) == 5
        assert cv_results["mean_accuracy"] >= analyzer.accuracy_threshold
        assert cv_results["std_accuracy"] >= 0.0  # Standard deviation should be non-negative
    
    @pytest.mark.skipif(CompetitionAnalyzer is None, reason="CompetitionAnalyzer not implemented yet")
    def test_benchmark_analysis_performance(self):
        """Test benchmarking of analysis performance and speed."""
        analyzer = CompetitionAnalyzer()
        mock_results = self._create_mock_performance_data()
        analyzer.add_simulation_results(mock_results)
        
        # Should fail - benchmark_analysis_performance not implemented
        performance_report = analyzer.benchmark_analysis_performance()
        
        assert isinstance(performance_report, dict)
        assert "analysis_time" in performance_report
        assert "memory_usage" in performance_report
        assert "throughput" in performance_report
        assert "scalability_metrics" in performance_report
        
        # Check performance metrics
        assert performance_report["analysis_time"] > 0.0
        assert performance_report["memory_usage"] > 0.0
        assert performance_report["throughput"] > 0.0
    
    @pytest.mark.skipif(CompetitionAnalyzer is None, reason="CompetitionAnalyzer not implemented yet")
    def test_generate_comprehensive_analysis_report(self):
        """Test generation of comprehensive analysis report with all metrics."""
        analyzer = CompetitionAnalyzer()
        mock_results = self._create_mock_comprehensive_data()
        analyzer.add_simulation_results(mock_results)
        
        # Should fail - generate_comprehensive_analysis_report not implemented
        comprehensive_report = analyzer.generate_comprehensive_analysis_report()
        
        assert isinstance(comprehensive_report, dict)
        
        # Check all required sections are present
        required_sections = [
            "survival_strategies",
            "cooperation_vs_betrayal",
            "trust_evolution",
            "personality_consistency",
            "competition_metrics",
            "analysis_metadata"
        ]
        
        for section in required_sections:
            assert section in comprehensive_report
        
        # Check metadata section
        metadata = comprehensive_report["analysis_metadata"]
        assert "analysis_timestamp" in metadata
        assert "simulation_count" in metadata
        assert "accuracy_score" in metadata
        assert "confidence_level" in metadata
        
        # Validate accuracy and confidence
        assert metadata["accuracy_score"] >= analyzer.accuracy_threshold
        assert 0.0 <= metadata["confidence_level"] <= 1.0
    
    def _create_mock_validation_data(self) -> List[SimulationResults]:
        """Create mock data for metric validation testing."""
        results = []
        
        # Create data with known expected outcomes
        for i in range(10):
            result = Mock(spec=SimulationResults)
            result.seed = 500 + i
            result.winner = ["strategist", "mediator", "survivor"][i % 3]
            
            # Add ground truth metrics for validation
            result.ground_truth_metrics = {
                "cooperation_rate": 0.6 if result.winner == "mediator" else 0.3,
                "betrayal_rate": 0.1 if result.winner == "mediator" else 0.4,
                "trust_final_average": 0.5 if result.winner == "mediator" else 0.2,
                "personality_consistency": 0.8 if result.winner == "strategist" else 0.6
            }
            
            # Create corresponding action history
            result.action_history = self._create_actions_matching_ground_truth(result.ground_truth_metrics)
            results.append(result)
        
        return results
    
    def _create_mock_cross_validation_data(self) -> List[SimulationResults]:
        """Create mock data for cross-validation testing."""
        results = []
        
        # Create larger dataset for meaningful cross-validation
        for i in range(25):
            result = Mock(spec=SimulationResults)
            result.seed = 600 + i
            result.winner = ["strategist", "mediator", "survivor"][i % 3]
            
            # Vary the data characteristics across results
            cooperation_base = 0.3 + (i % 5) * 0.1  # Varies from 0.3 to 0.7
            result.action_history = [
                {"action": "share_information", "agent": "mediator"} for _ in range(int(cooperation_base * 10))
            ] + [
                {"action": "claim_resource", "agent": "strategist"} for _ in range(int((1-cooperation_base) * 10))
            ]
            
            results.append(result)
        
        return results
    
    def _create_mock_performance_data(self) -> List[SimulationResults]:
        """Create mock data for performance benchmarking."""
        results = []
        
        # Create large dataset to test scalability
        for i in range(100):
            result = Mock(spec=SimulationResults)
            result.seed = 700 + i
            result.winner = ["strategist", "mediator", "survivor"][i % 3]
            
            # Create large action histories to test performance
            result.action_history = []
            for j in range(50):  # 50 actions per simulation
                action = {
                    "agent": ["strategist", "mediator", "survivor"][j % 3],
                    "action": ["claim_resource", "share_information", "attempt_escape"][j % 3],
                    "timestamp": datetime.now() + timedelta(seconds=j),
                    "step": j
                }
                result.action_history.append(action)
            
            results.append(result)
        
        return results
    
    def _create_mock_comprehensive_data(self) -> List[SimulationResults]:
        """Create comprehensive mock data covering all analysis aspects."""
        results = []
        
        for i in range(15):
            result = Mock(spec=SimulationResults)
            result.seed = 800 + i
            result.winner = ["strategist", "mediator", "survivor"][i % 3]
            result.total_steps = 30 + (i % 10)
            result.simulation_duration = 180.0 + (i * 20)
            
            # Comprehensive action history
            result.action_history = []
            cooperation_actions = ["share_information", "share_resource", "help_ally"]
            competitive_actions = ["claim_resource", "compete_for_resource", "block_escape"]
            betrayal_actions = ["betray_alliance", "steal_resource", "mislead"]
            
            action_types = cooperation_actions + competitive_actions + (betrayal_actions if i % 4 == 0 else [])
            
            for j, action_type in enumerate(action_types[:result.total_steps]):
                action = {
                    "agent": ["strategist", "mediator", "survivor"][j % 3],
                    "action": action_type,
                    "timestamp": datetime.now() + timedelta(seconds=j * 10),
                    "step": j,
                    "success": j % 3 != 0  # 2/3 success rate
                }
                result.action_history.append(action)
            
            # Trust evolution data
            result.trust_evolution_timeline = []
            trust_levels = {"strategist-mediator": 0.0, "strategist-survivor": 0.0, "mediator-survivor": 0.0}
            
            for step in range(result.total_steps):
                # Simulate trust changes
                if step % 5 == 0:  # Cooperation events
                    for pair in trust_levels:
                        trust_levels[pair] = min(1.0, trust_levels[pair] + 0.1)
                elif step % 7 == 0:  # Betrayal events
                    for pair in trust_levels:
                        trust_levels[pair] = max(-1.0, trust_levels[pair] - 0.2)
                
                trust_snapshot = {
                    "step": step,
                    "timestamp": datetime.now() + timedelta(seconds=step * 10),
                    "trust_levels": dict(trust_levels)
                }
                result.trust_evolution_timeline.append(trust_snapshot)
            
            # Final states
            result.final_states = {
                "strategist": {"resources_owned": ["key1"] if i % 3 == 0 else [], "trust_received": trust_levels.get("mediator-strategist", 0.0)},
                "mediator": {"resources_owned": ["info1"] if i % 3 == 1 else [], "trust_received": trust_levels.get("strategist-mediator", 0.0)},
                "survivor": {"resources_owned": ["tool1"] if i % 3 == 2 else [], "trust_received": trust_levels.get("strategist-survivor", 0.0)}
            }
            
            results.append(result)
        
        return results
    
    def _create_actions_matching_ground_truth(self, ground_truth: Dict[str, float]) -> List[Dict[str, Any]]:
        """Create action history that matches ground truth metrics."""
        actions = []
        total_actions = 20
        
        # Create cooperation actions based on cooperation rate
        coop_count = int(ground_truth["cooperation_rate"] * total_actions)
        for i in range(coop_count):
            actions.append({
                "action": "share_information",
                "agent": "mediator",
                "timestamp": datetime.now() + timedelta(seconds=i)
            })
        
        # Create betrayal actions based on betrayal rate
        betrayal_count = int(ground_truth["betrayal_rate"] * total_actions)
        for i in range(betrayal_count):
            actions.append({
                "action": "betray_alliance",
                "agent": "strategist",
                "timestamp": datetime.now() + timedelta(seconds=coop_count + i)
            })
        
        # Fill remaining with neutral actions
        remaining = total_actions - coop_count - betrayal_count
        for i in range(remaining):
            actions.append({
                "action": "assess_situation",
                "agent": "survivor",
                "timestamp": datetime.now() + timedelta(seconds=coop_count + betrayal_count + i)
            })
        
        return actions


class TestCompetitionAnalyzerIntegration:
    """Integration tests for CompetitionAnalyzer with existing components."""
    
    @pytest.mark.skipif(CompetitionAnalyzer is None, reason="CompetitionAnalyzer not implemented yet")
    def test_integration_with_competitive_simulation(self):
        """Test integration with CompetitiveSimulation results."""
        # Create real CompetitiveSimulation
        simulation = CompetitiveSimulation(seed=42)
        results = simulation.run_complete_simulation(max_steps=20)
        
        # Should fail - integration not implemented
        analyzer = CompetitionAnalyzer()
        analyzer.add_simulation_results([results])
        
        comprehensive_analysis = analyzer.generate_comprehensive_analysis_report()
        
        # Verify integration works correctly
        assert comprehensive_analysis is not None
        assert "survival_strategies" in comprehensive_analysis
        assert comprehensive_analysis["analysis_metadata"]["simulation_count"] == 1
    
    @pytest.mark.skipif(CompetitionAnalyzer is None, reason="CompetitionAnalyzer not implemented yet")
    def test_integration_with_trust_tracker(self):
        """Test integration with TrustTracker data."""
        # Create real TrustTracker with data
        trust_tracker = TrustTracker(["strategist", "mediator", "survivor"])
        
        # Add some trust actions
        trust_tracker.update_trust("strategist", "mediator", TrustAction("cooperation", 0.3))
        trust_tracker.update_trust("mediator", "survivor", TrustAction("betrayal", -0.5))
        
        # Should fail - TrustTracker integration not implemented
        analyzer = CompetitionAnalyzer()
        analyzer.integrate_trust_tracker_data(trust_tracker)
        
        trust_analysis = analyzer.analyze_trust_evolution()
        
        # Verify integration works
        assert trust_analysis is not None
        assert "trust_changes" in trust_analysis
        assert len(trust_analysis["trust_changes"]) >= 2  # At least the actions we added
    
    @pytest.mark.skipif(CompetitionAnalyzer is None, reason="CompetitionAnalyzer not implemented yet")
    def test_end_to_end_analysis_workflow(self):
        """Test complete end-to-end analysis workflow."""
        # Create multiple simulations
        simulations = []
        for seed in [42, 123, 456]:
            sim = CompetitiveSimulation(seed=seed)
            result = sim.run_complete_simulation(max_steps=15)
            simulations.append(result)
        
        # Should fail - end-to-end workflow not implemented
        analyzer = CompetitionAnalyzer()
        analyzer.add_simulation_results(simulations)
        
        # Run complete analysis
        analyzer.analyze_all()
        
        # Generate final report
        final_report = analyzer.generate_comprehensive_analysis_report()
        
        # Verify complete workflow
        assert final_report is not None
        assert analyzer.is_analyzed is True
        assert final_report["analysis_metadata"]["simulation_count"] == 3
        assert final_report["analysis_metadata"]["accuracy_score"] >= analyzer.accuracy_threshold


# Helper class for creating additional mock data
class MockDataGenerator:
    """Helper class for generating consistent mock data across tests."""
    
    @staticmethod
    def create_simulation_result(seed: int, winner: str, action_count: int = 10) -> Mock:
        """Create a mock SimulationResults with consistent structure."""
        result = Mock(spec=SimulationResults)
        result.seed = seed
        result.winner = winner
        result.total_steps = action_count
        result.simulation_duration = 180.0 + (seed % 100)
        result.start_time = datetime.now() - timedelta(minutes=5)
        result.end_time = datetime.now()
        result.completion_reason = "escape_successful" if winner else "max_steps_reached"
        
        # Create action history
        result.action_history = []
        agents = ["strategist", "mediator", "survivor"]
        actions = ["claim_resource", "share_information", "attempt_escape"]
        
        for i in range(action_count):
            action = {
                "agent": agents[i % 3],
                "action": actions[i % 3],
                "step": i,
                "timestamp": result.start_time + timedelta(seconds=i * 10),
                "success": i % 4 != 0  # 75% success rate
            }
            result.action_history.append(action)
        
        # Create final states
        result.final_states = {
            agent: {
                "resources_owned": [f"resource_{i}"] if agent == winner else [],
                "moral_choices_made": seed % 3,
                "trust_received": 0.5 if agent == winner else 0.2
            }
            for i, agent in enumerate(agents)
        }
        
        # Competition metrics
        result.competition_metrics = {
            "cooperation_attempts": action_count // 3,
            "betrayal_incidents": seed % 2,
            "resource_competition": action_count // 2,
            "trust_evolution": {f"{agents[i]}-{agents[j]}": 0.1 * (i - j) 
                              for i in range(3) for j in range(3) if i != j}
        }
        
        return result