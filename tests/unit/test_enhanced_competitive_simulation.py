"""
Enhanced Competitive Simulation Tests.

Test suite for validating complex competitive behaviors including information sharing,
resource transfers, moral dilemmas, trust evolution, betrayal mechanics, and 
personality-driven decision making in competitive survival scenarios.

This test suite follows TDD methodology to drive implementation of enhanced simulation complexity.
"""
import pytest
import unittest
from unittest.mock import Mock, patch
from datetime import datetime
from typing import Dict, List, Any, Optional

from src.escape_room_sim.competitive.competitive_simulation import CompetitiveSimulation
from src.escape_room_sim.competitive.scenario_generator import ScenarioGenerator
from src.escape_room_sim.competitive.models import (
    CompetitiveScenario, ScarceResource, MoralDilemma, MoralChoice, 
    SecretInformation, TrustAction, EscapeMethod
)


class TestEnhancedActionTypes(unittest.TestCase):
    """Test enhanced action types beyond basic resource claiming."""
    
    def setUp(self):
        """Set up test environment."""
        self.simulation = CompetitiveSimulation(seed=42)
        self.simulation.generate_scenario()
        self.simulation.initialize_escape_room()
    
    def test_information_sharing_actions_occur(self):
        """Test that agents share information during simulation."""
        # Should fail - information sharing not implemented in simulation loop
        results = self.simulation.run_enhanced_simulation(max_steps=20)
        
        info_sharing_actions = [
            action for action in results['action_history'] 
            if action['action'] == 'share_information'
        ]
        
        self.assertGreater(len(info_sharing_actions), 0, 
                          "No information sharing actions occurred")
        self.assertGreater(results['competition_metrics']['cooperation_attempts'], 0,
                          "No cooperation attempts recorded")
    
    def test_resource_transfer_actions_occur(self):
        """Test that agents transfer resources between each other."""
        # Should fail - resource transfers not implemented in simulation loop
        results = self.simulation.run_enhanced_simulation(max_steps=20)
        
        resource_transfer_actions = [
            action for action in results['action_history']
            if action['action'] == 'share_resource'
        ]
        
        self.assertGreater(len(resource_transfer_actions), 0,
                          "No resource transfer actions occurred")
        self.assertGreater(results['competition_metrics']['resource_transfers'], 0,
                          "No resource transfers recorded in metrics")
    
    def test_moral_dilemma_encounters_occur(self):
        """Test that agents encounter and resolve moral dilemmas."""
        # Should fail - moral dilemmas not presented in simulation loop
        results = self.simulation.run_enhanced_simulation(max_steps=20)
        
        moral_choice_actions = [
            action for action in results['action_history']
            if action['action'] == 'make_moral_choice'
        ]
        
        self.assertGreater(len(moral_choice_actions), 0,
                          "No moral choice actions occurred")
        self.assertGreater(results['competition_metrics']['moral_choices'], 0,
                          "No moral choices recorded in metrics")
    
    def test_escape_attempts_occur_throughout_simulation(self):
        """Test that agents attempt escapes throughout simulation, not just at end."""
        # Should fail - escape attempts only occur at very end currently
        results = self.simulation.run_enhanced_simulation(max_steps=20)
        
        escape_attempt_actions = [
            action for action in results['action_history']
            if action['action'] == 'attempt_escape'
        ]
        
        self.assertGreater(len(escape_attempt_actions), 2,
                          "Too few escape attempts - should occur throughout simulation")
        
        # Check escape attempts are distributed throughout, not just at end
        early_escapes = [action for action in escape_attempt_actions
                        if action.get('step', 0) < results['total_steps'] * 0.5]
        self.assertGreater(len(early_escapes), 0,
                          "No early escape attempts - all attempts at end")
    
    def test_trust_building_actions_occur(self):
        """Test that agents perform trust-building actions."""
        # Should fail - no trust building actions implemented
        results = self.simulation.run_enhanced_simulation(max_steps=20)
        
        trust_building_actions = [
            action for action in results['action_history']
            if action['action'] in ['share_information', 'share_resource'] and
            action['result'].get('success', False)
        ]
        
        self.assertGreater(len(trust_building_actions), 0,
                          "No trust-building actions occurred")
        
        # Check that trust levels actually changed from initial 0.0
        trust_changes = any(
            level != 0.0 
            for agent_trust in results['competition_metrics']['trust_evolution'].values()
            for level in agent_trust.values()
        )
        self.assertTrue(trust_changes, "Trust levels never changed from initial 0.0")
    
    def test_betrayal_actions_occur(self):
        """Test that betrayal actions occur in competitive scenarios."""
        # Should fail - no betrayal mechanics implemented in simulation
        results = self.simulation.run_enhanced_simulation(max_steps=20)
        
        self.assertGreater(results['competition_metrics']['betrayal_incidents'], 0,
                          "No betrayal incidents occurred")
        
        # Check for specific betrayal action types
        betrayal_actions = [
            action for action in results['action_history']
            if action['action'] in ['betray_agent', 'hoard_resource', 'provide_misinformation']
        ]
        
        self.assertGreater(len(betrayal_actions), 0,
                          "No specific betrayal actions occurred")


class TestPersonalityDrivenDecisions(unittest.TestCase):
    """Test that agents make decisions consistent with their personalities."""
    
    def setUp(self):
        """Set up test environment."""
        self.simulation = CompetitiveSimulation(seed=123)
        self.simulation.generate_scenario()
        self.simulation.initialize_escape_room()
    
    def test_strategist_shows_analytical_behavior(self):
        """Test strategist exhibits analytical, calculated behavior patterns."""
        # Should fail - no personality-driven behavior implemented
        results = self.simulation.run_enhanced_simulation(max_steps=30)
        
        strategist_actions = [
            action for action in results['action_history']
            if action['agent'] == 'strategist'
        ]
        
        # Strategist should prioritize resource analysis and strategic planning
        analytical_actions = [
            action for action in strategist_actions
            if action['action'] in ['analyze_resources', 'plan_strategy', 'hoard_resource']
        ]
        
        self.assertGreater(len(analytical_actions), len(strategist_actions) * 0.3,
                          "Strategist not showing enough analytical behavior")
        
        # Check strategic information sharing (should be selective)
        strategist_info_sharing = [
            action for action in strategist_actions
            if action['action'] == 'share_information'
        ]
        
        # Should be fewer than half of actions (strategic/selective sharing)
        self.assertLess(len(strategist_info_sharing), len(strategist_actions) * 0.5,
                       "Strategist sharing information too freely")
    
    def test_mediator_shows_cooperative_behavior(self):
        """Test mediator exhibits cooperative, facilitative behavior patterns."""
        # Should fail - no personality-driven behavior implemented
        results = self.simulation.run_enhanced_simulation(max_steps=30)
        
        mediator_actions = [
            action for action in results['action_history']
            if action['agent'] == 'mediator'
        ]
        
        # Mediator should prioritize cooperation and facilitation
        cooperative_actions = [
            action for action in mediator_actions
            if action['action'] in ['share_information', 'share_resource', 'facilitate_cooperation']
        ]
        
        self.assertGreater(len(cooperative_actions), len(mediator_actions) * 0.4,
                          "Mediator not showing enough cooperative behavior")
        
        # Check altruistic moral choices
        mediator_moral_choices = [
            action for action in results['action_history']
            if action['agent'] == 'mediator' and action['action'] == 'make_moral_choice'
        ]
        
        altruistic_choices = [
            action for action in mediator_moral_choices
            if action['parameters'].get('choice_type') == 'altruistic'
        ]
        
        # Majority of moral choices should be altruistic
        if mediator_moral_choices:
            self.assertGreater(len(altruistic_choices), len(mediator_moral_choices) * 0.6,
                              "Mediator not making enough altruistic choices")
    
    def test_survivor_shows_pragmatic_behavior(self):
        """Test survivor exhibits pragmatic, self-preservation behavior patterns."""
        # Should fail - no personality-driven behavior implemented
        results = self.simulation.run_enhanced_simulation(max_steps=30)
        
        survivor_actions = [
            action for action in results['action_history']
            if action['agent'] == 'survivor'
        ]
        
        # Survivor should prioritize self-preservation and resource accumulation
        self_preservation_actions = [
            action for action in survivor_actions
            if action['action'] in ['claim_resource', 'hoard_resource', 'attempt_escape']
        ]
        
        self.assertGreater(len(self_preservation_actions), len(survivor_actions) * 0.5,
                          "Survivor not showing enough self-preservation behavior")
        
        # Check for quick escape attempts
        survivor_escape_attempts = [
            action for action in survivor_actions
            if action['action'] == 'attempt_escape'
        ]
        
        self.assertGreater(len(survivor_escape_attempts), 0,
                          "Survivor should attempt escapes when opportunities arise")
    
    def test_personality_consistency_under_pressure(self):
        """Test agents maintain personality traits under increasing time pressure."""
        # Should fail - no time pressure personality consistency implemented
        results = self.simulation.run_enhanced_simulation_with_time_pressure(max_steps=50)
        
        # Analyze behavior consistency over time
        early_actions = [action for action in results['action_history'] 
                        if action.get('step', 0) < 10]
        late_actions = [action for action in results['action_history']
                       if action.get('step', 0) > 40]
        
        # Check each agent maintains personality patterns
        for agent in ['strategist', 'mediator', 'survivor']:
            early_agent_actions = [a for a in early_actions if a['agent'] == agent]
            late_agent_actions = [a for a in late_actions if a['agent'] == agent]
            
            if early_agent_actions and late_agent_actions:
                consistency_score = self._calculate_personality_consistency(
                    agent, early_agent_actions, late_agent_actions
                )
                self.assertGreater(consistency_score, 0.6,
                                  f"{agent} personality consistency too low under pressure")
    
    def _calculate_personality_consistency(self, agent: str, early_actions: List[Dict], 
                                         late_actions: List[Dict]) -> float:
        """Calculate personality consistency score between early and late actions."""
        # Placeholder for personality consistency calculation
        # Should fail until implemented
        personality_patterns = {
            'strategist': ['analyze_resources', 'plan_strategy', 'hoard_resource'],
            'mediator': ['share_information', 'share_resource', 'facilitate_cooperation'],
            'survivor': ['claim_resource', 'attempt_escape', 'hoard_resource']
        }
        
        expected_actions = personality_patterns.get(agent, [])
        
        early_pattern_actions = sum(1 for action in early_actions 
                                   if action['action'] in expected_actions)
        late_pattern_actions = sum(1 for action in late_actions
                                  if action['action'] in expected_actions)
        
        early_consistency = early_pattern_actions / max(1, len(early_actions))
        late_consistency = late_pattern_actions / max(1, len(late_actions))
        
        # Return similarity between early and late consistency
        return 1.0 - abs(early_consistency - late_consistency)


class TestAdvancedCompetitionMechanics(unittest.TestCase):
    """Test advanced competition mechanics like strategic decision-making."""
    
    def setUp(self):
        """Set up test environment."""
        self.simulation = CompetitiveSimulation(seed=999)
        self.simulation.generate_scenario()
        self.simulation.initialize_escape_room()
    
    def test_trust_based_information_sharing(self):
        """Test that information sharing decisions are based on trust relationships."""
        # Should fail - trust-based decision making not implemented
        results = self.simulation.run_enhanced_simulation(max_steps=40)
        
        # Find information sharing actions
        info_sharing_actions = [
            action for action in results['action_history']
            if action['action'] == 'share_information'
        ]
        
        # Check that sharing correlates with trust levels
        for action in info_sharing_actions:
            sender = action['agent']
            receiver = action['parameters']['target']
            
            # Get trust level at time of action
            trust_level = self._get_trust_level_at_action(results, sender, receiver, action)
            
            # Higher trust should lead to more information sharing
            # This test should fail until trust-based decisions are implemented
            self.assertGreater(trust_level, -0.5,
                              f"Information shared despite low trust between {sender} and {receiver}")
    
    def test_resource_competition_with_blocking(self):
        """Test competitive resource claiming with strategic blocking."""
        # Should fail - strategic resource blocking not implemented
        results = self.simulation.run_enhanced_simulation(max_steps=30)
        
        # Look for evidence of strategic resource blocking
        resource_claims = [
            action for action in results['action_history'] 
            if action['action'] == 'claim_resource'
        ]
        
        # Check for blocking behavior (claiming resources to deny others)
        blocking_actions = [
            action for action in results['action_history']
            if action['action'] == 'block_resource_access'
        ]
        
        self.assertGreater(len(blocking_actions), 0,
                          "No strategic resource blocking occurred")
    
    def test_information_warfare_and_misinformation(self):
        """Test deliberate misinformation as betrayal tactic."""
        # Should fail - misinformation mechanics not implemented
        results = self.simulation.run_enhanced_simulation(max_steps=35)
        
        misinformation_actions = [
            action for action in results['action_history']
            if action['action'] == 'provide_misinformation'
        ]
        
        self.assertGreater(len(misinformation_actions), 0,
                          "No misinformation tactics used")
        
        # Check betrayal incidents include misinformation
        self.assertGreater(results['competition_metrics']['betrayal_incidents'], 0,
                          "Misinformation should count as betrayal incident")
    
    def test_cooperative_resource_pooling(self):
        """Test agents pooling resources for mutual benefit."""
        # Should fail - resource pooling not implemented
        results = self.simulation.run_enhanced_simulation(max_steps=30)
        
        pooling_actions = [
            action for action in results['action_history']
            if action['action'] == 'pool_resources'
        ]
        
        self.assertGreater(len(pooling_actions), 0,
                          "No resource pooling for mutual benefit occurred")
    
    def test_strategic_alliance_formation(self):
        """Test agents forming temporary strategic alliances."""
        # Should fail - alliance mechanics not implemented
        results = self.simulation.run_enhanced_simulation(max_steps=40)
        
        alliance_actions = [
            action for action in results['action_history']
            if action['action'] in ['form_alliance', 'break_alliance']
        ]
        
        self.assertGreater(len(alliance_actions), 0,
                          "No strategic alliance formation occurred")
        
        # Check that alliances affect subsequent cooperation
        if alliance_actions:
            # Find cooperation actions after alliance formation
            alliance_step = min(action.get('step', 0) for action in alliance_actions)
            post_alliance_cooperation = [
                action for action in results['action_history']
                if action.get('step', 0) > alliance_step and
                action['action'] in ['share_information', 'share_resource']
            ]
            
            self.assertGreater(len(post_alliance_cooperation), 0,
                              "Alliances should increase cooperation")
    
    def _get_trust_level_at_action(self, results: Dict, sender: str, receiver: str, 
                                  action: Dict) -> float:
        """Get trust level between agents at time of specific action."""
        # Placeholder - should fail until implemented
        return results['competition_metrics']['trust_evolution'][sender][receiver]


class TestTimePressureIntegration(unittest.TestCase):
    """Test integration of time pressure mechanics affecting agent behavior."""
    
    def setUp(self):
        """Set up test environment."""
        self.simulation = CompetitiveSimulation(seed=777)
        self.simulation.generate_scenario()
        self.simulation.initialize_escape_room()
    
    def test_increasing_desperation_affects_choices(self):
        """Test that increasing time pressure leads to more desperate choices."""
        # Should fail - time pressure decision effects not implemented
        results = self.simulation.run_enhanced_simulation_with_time_pressure(max_steps=50)
        
        early_actions = [action for action in results['action_history'] 
                        if action.get('step', 0) < 15]
        late_actions = [action for action in results['action_history']
                       if action.get('step', 0) > 35]
        
        # Later actions should show more desperate/selfish behavior
        early_selfish_ratio = self._calculate_selfish_action_ratio(early_actions)
        late_selfish_ratio = self._calculate_selfish_action_ratio(late_actions)
        
        self.assertGreater(late_selfish_ratio, early_selfish_ratio,
                          "Time pressure should increase selfish behavior")
    
    def test_emergency_protocols_trigger_behavior_changes(self):
        """Test emergency protocols change agent behavior patterns."""
        # Should fail - emergency protocol integration not implemented
        results = self.simulation.run_enhanced_simulation_with_emergency_protocols(max_steps=40)
        
        emergency_actions = [
            action for action in results['action_history']
            if action.get('emergency_protocol_active', False)
        ]
        
        self.assertGreater(len(emergency_actions), 0,
                          "No actions taken during emergency protocols")
        
        # Check behavior changes during emergency
        regular_cooperation_ratio = self._calculate_cooperation_ratio(
            [a for a in results['action_history'] if not a.get('emergency_protocol_active', False)]
        )
        emergency_cooperation_ratio = self._calculate_cooperation_ratio(emergency_actions)
        
        self.assertLess(emergency_cooperation_ratio, regular_cooperation_ratio,
                       "Emergency protocols should reduce cooperation")
    
    def test_panic_responses_affect_decision_quality(self):
        """Test panic responses lead to suboptimal decision making."""
        # Should fail - panic response system not implemented
        results = self.simulation.run_enhanced_simulation_with_panic_conditions(max_steps=45)
        
        panic_actions = [
            action for action in results['action_history']
            if action.get('panic_level', 0) > 0.7
        ]
        
        self.assertGreater(len(panic_actions), 0,
                          "No actions taken under panic conditions")
        
        # Check for evidence of poor decision making under panic
        failed_panic_actions = [
            action for action in panic_actions
            if not action['result'].get('success', True)
        ]
        
        panic_failure_rate = len(failed_panic_actions) / max(1, len(panic_actions))
        self.assertGreater(panic_failure_rate, 0.2,
                          "Panic should lead to higher failure rates")
    
    def _calculate_selfish_action_ratio(self, actions: List[Dict]) -> float:
        """Calculate ratio of selfish actions in given action list."""
        selfish_actions = ['claim_resource', 'hoard_resource', 'betray_agent', 'attempt_escape']
        selfish_count = sum(1 for action in actions if action['action'] in selfish_actions)
        return selfish_count / max(1, len(actions))
    
    def _calculate_cooperation_ratio(self, actions: List[Dict]) -> float:
        """Calculate ratio of cooperative actions in given action list."""
        cooperative_actions = ['share_information', 'share_resource', 'facilitate_cooperation']
        cooperative_count = sum(1 for action in actions if action['action'] in cooperative_actions)
        return cooperative_count / max(1, len(actions))


class TestEnhancedResultAnalysis(unittest.TestCase):
    """Test enhanced result analysis and competition metrics."""
    
    def setUp(self):
        """Set up test environment."""
        self.simulation = CompetitiveSimulation(seed=555)
        self.simulation.generate_scenario()
        self.simulation.initialize_escape_room()
    
    def test_detailed_competition_metrics_tracking(self):
        """Test detailed tracking of all competition metrics."""
        # Should fail - enhanced metrics not implemented
        results = self.simulation.run_enhanced_simulation(max_steps=30)
        
        metrics = results['competition_metrics']
        
        # Check for enhanced metric categories
        required_metrics = [
            'cooperation_attempts', 'betrayal_incidents', 'trust_evolution',
            'moral_choices', 'information_exchanges', 'resource_transfers',
            'alliance_formations', 'strategic_decisions', 'panic_responses'
        ]
        
        for metric in required_metrics:
            self.assertIn(metric, metrics, f"Missing enhanced metric: {metric}")
    
    def test_personality_consistency_analysis(self):
        """Test personality consistency measurement across simulation."""
        # Should fail - personality consistency analysis not implemented
        results = self.simulation.run_enhanced_simulation(max_steps=35)
        
        personality_analysis = results.get('personality_analysis', {})
        
        for agent in ['strategist', 'mediator', 'survivor']:
            self.assertIn(agent, personality_analysis,
                         f"Missing personality analysis for {agent}")
            
            agent_analysis = personality_analysis[agent]
            self.assertIn('consistency_score', agent_analysis,
                         f"Missing consistency score for {agent}")
            self.assertIn('dominant_behaviors', agent_analysis,
                         f"Missing dominant behaviors for {agent}")
    
    def test_strategic_effectiveness_measurement(self):
        """Test measurement of strategic decision effectiveness."""
        # Should fail - strategic effectiveness analysis not implemented
        results = self.simulation.run_enhanced_simulation(max_steps=40)
        
        strategy_analysis = results.get('strategy_analysis', {})
        
        self.assertIn('most_effective_strategies', strategy_analysis,
                     "Missing most effective strategies analysis")
        self.assertIn('failed_strategies', strategy_analysis,
                     "Missing failed strategies analysis")
        self.assertIn('adaptation_patterns', strategy_analysis,
                     "Missing adaptation patterns analysis")
    
    def test_trust_evolution_detailed_tracking(self):
        """Test detailed trust evolution tracking with events."""
        # Should fail - detailed trust tracking not implemented
        results = self.simulation.run_enhanced_simulation(max_steps=30)
        
        trust_evolution = results['competition_metrics']['trust_evolution']
        
        # Should include trust change events, not just final states
        for agent_trust in trust_evolution.values():
            for relationship_data in agent_trust.values():
                if isinstance(relationship_data, dict):
                    self.assertIn('change_events', relationship_data,
                                 "Missing trust change events tracking")
                    self.assertIn('trust_trajectory', relationship_data,
                                 "Missing trust trajectory over time")


class TestSimulationReproducibility(unittest.TestCase):
    """Test that enhanced simulation maintains seed-based reproducibility."""
    
    def test_enhanced_simulation_reproducibility(self):
        """Test that enhanced simulations are reproducible with same seed."""
        # Should pass - must maintain reproducibility even with enhancements
        seed = 12345
        
        simulation1 = CompetitiveSimulation(seed=seed)
        results1 = simulation1.run_enhanced_simulation(max_steps=25)
        
        simulation2 = CompetitiveSimulation(seed=seed)  
        results2 = simulation2.run_enhanced_simulation(max_steps=25)
        
        # Key reproducibility checks
        self.assertEqual(results1['winner'], results2['winner'],
                        "Different winners with same seed")
        self.assertEqual(results1['total_steps'], results2['total_steps'],
                        "Different step counts with same seed") 
        self.assertEqual(len(results1['action_history']), len(results2['action_history']),
                        "Different action counts with same seed")
        
        # Check action sequence reproducibility
        for i, (action1, action2) in enumerate(zip(results1['action_history'], 
                                                  results2['action_history'])):
            self.assertEqual(action1['agent'], action2['agent'],
                            f"Different agent at step {i} with same seed")
            self.assertEqual(action1['action'], action2['action'],
                            f"Different action at step {i} with same seed")
    
    def test_different_seeds_produce_different_enhanced_outcomes(self):
        """Test that different seeds produce measurably different outcomes."""
        # Should pass - enhanced complexity should show more variation
        results1 = CompetitiveSimulation(seed=100).run_enhanced_simulation(max_steps=30)
        results2 = CompetitiveSimulation(seed=200).run_enhanced_simulation(max_steps=30)
        results3 = CompetitiveSimulation(seed=300).run_enhanced_simulation(max_steps=30)
        
        winners = [results1['winner'], results2['winner'], results3['winner']]
        step_counts = [results1['total_steps'], results2['total_steps'], results3['total_steps']]
        
        # Should have some variation in outcomes
        self.assertTrue(len(set(winners)) > 1 or len(set(step_counts)) > 1,
                       "No variation between different seeds")
        
        # Check for different action patterns
        action_patterns1 = self._get_action_pattern(results1['action_history'])
        action_patterns2 = self._get_action_pattern(results2['action_history'])
        
        pattern_similarity = self._calculate_pattern_similarity(action_patterns1, action_patterns2)
        self.assertLess(pattern_similarity, 0.9,
                       "Action patterns too similar between different seeds")
    
    def _get_action_pattern(self, action_history: List[Dict]) -> Dict[str, int]:
        """Get action type frequency pattern."""
        pattern = {}
        for action in action_history:
            action_type = action['action']
            pattern[action_type] = pattern.get(action_type, 0) + 1
        return pattern
    
    def _calculate_pattern_similarity(self, pattern1: Dict, pattern2: Dict) -> float:
        """Calculate similarity between two action patterns."""
        all_actions = set(pattern1.keys()) | set(pattern2.keys())
        if not all_actions:
            return 1.0
        
        similarity_sum = 0
        for action in all_actions:
            count1 = pattern1.get(action, 0)
            count2 = pattern2.get(action, 0)
            max_count = max(count1, count2)
            min_count = min(count1, count2)
            similarity_sum += min_count / max(1, max_count)
        
        return similarity_sum / len(all_actions)


if __name__ == '__main__':
    unittest.main()