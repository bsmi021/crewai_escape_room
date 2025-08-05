"""
Unit tests for time pressure and escalation mechanics.
Tests time limit enforcement, escalating threat systems, desperation level calculation,
option reduction mechanics, automatic failure conditions, and time pressure effects
on agent behavior patterns in competitive scenarios.
"""
import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
from src.escape_room_sim.competitive.models import (
    CompetitiveScenario, EscapeResult, MoralChoice, TrustAction,
    ScarceResource, MoralDilemma, SecretInformation
)
from src.escape_room_sim.competitive.competitive_escape_room import CompetitiveEscapeRoom
from src.escape_room_sim.competitive.scenario_generator import ScenarioGenerator


class TestTimeLimitEnforcement:
    """Tests for time limit enforcement in CompetitiveEscapeRoom."""
    
    def test_escape_room_enforces_time_limit_on_actions(self):
        """Test that escape room enforces time limits on agent actions."""
        # This should fail initially - time limit enforcement not implemented
        scenario = ScenarioGenerator(seed=42).generate_scenario()
        escape_room = CompetitiveEscapeRoom(scenario)
        
        # Should fail - time limit enforcement not implemented
        assert escape_room.time_limit == scenario.time_limit
        assert escape_room.time_remaining <= escape_room.time_limit
        assert escape_room.is_time_expired() is False
    
    def test_time_remaining_decreases_over_simulation_time(self):
        """Test that time remaining decreases as simulation progresses."""
        scenario = ScenarioGenerator(seed=42).generate_scenario()
        escape_room = CompetitiveEscapeRoom(scenario)
        
        initial_time = escape_room.time_remaining
        
        # Should fail - time tracking not implemented
        escape_room.advance_time(30)  # Advance 30 seconds
        
        assert escape_room.time_remaining == initial_time - 30
        assert escape_room.time_remaining > 0
    
    def test_actions_fail_when_time_expired(self):
        """Test that actions fail when time has expired."""
        scenario = ScenarioGenerator(seed=42).generate_scenario()
        escape_room = CompetitiveEscapeRoom(scenario)
        
        # Force time expiration
        escape_room.advance_time(scenario.time_limit + 1)
        
        # Should fail - time expiration handling not implemented
        assert escape_room.is_time_expired() is True
        
        # Try to claim resource after time expired
        result = escape_room.process_resource_claim("strategist", "master_key")
        assert result.success is False
        assert "time" in result.failure_reason.lower()
    
    def test_escape_attempts_fail_when_time_expired(self):
        """Test that escape attempts fail when time has expired."""
        scenario = ScenarioGenerator(seed=42).generate_scenario()
        escape_room = CompetitiveEscapeRoom(scenario)
        
        # Force time expiration
        escape_room.advance_time(scenario.time_limit + 1)
        
        # Should fail - escape time validation not implemented
        escape_method = scenario.escape_methods[0]
        result = escape_room.attempt_escape("strategist", escape_method)
        
        assert result.success is False
        assert "time" in result.failure_reason.lower()
    
    def test_time_limit_enforcement_affects_all_subsystems(self):
        """Test that time limit enforcement affects all subsystems."""
        scenario = ScenarioGenerator(seed=42).generate_scenario()
        escape_room = CompetitiveEscapeRoom(scenario)
        
        # Force time expiration
        escape_room.advance_time(scenario.time_limit + 1)
        
        # Should fail - subsystem time enforcement not implemented
        # Resource manager should reject actions
        resource_result = escape_room.resource_manager.claim_resource("strategist", "master_key")
        assert resource_result.success is False
        
        # Information broker should reject sharing
        secret = scenario.secret_information[0]
        info_result = escape_room.info_broker.share_information("strategist", "mediator", secret)
        assert info_result is False  # Should reject when time expired
    
    def test_get_time_pressure_level_increases_over_time(self):
        """Test that time pressure level increases as time runs out."""
        scenario = ScenarioGenerator(seed=42).generate_scenario()
        escape_room = CompetitiveEscapeRoom(scenario)
        
        # Should fail - time pressure calculation not implemented
        initial_pressure = escape_room.get_time_pressure_level()
        assert 0.0 <= initial_pressure <= 1.0
        
        # Advance time halfway
        escape_room.advance_time(scenario.time_limit // 2)
        halfway_pressure = escape_room.get_time_pressure_level()
        
        # Advance time near end
        escape_room.advance_time(scenario.time_limit // 3)
        final_pressure = escape_room.get_time_pressure_level()
        
        assert initial_pressure < halfway_pressure < final_pressure
        assert final_pressure >= 0.8  # High pressure near end


class TestEscalatingThreatSystem:
    """Tests for escalating threat system increasing pressure over time."""
    
    def test_threat_level_starts_low_and_escalates_over_time(self):
        """Test that threat level starts low and escalates as time passes."""
        scenario = ScenarioGenerator(seed=42).generate_scenario()
        escape_room = CompetitiveEscapeRoom(scenario)
        
        # Should fail - threat system not implemented
        initial_threat = escape_room.get_current_threat_level()
        assert 0.0 <= initial_threat <= 0.3  # Low initial threat
        
        # Advance time and check escalation
        escape_room.advance_time(scenario.time_limit // 4)
        quarter_threat = escape_room.get_current_threat_level()
        
        escape_room.advance_time(scenario.time_limit // 2)
        final_threat = escape_room.get_current_threat_level()
        
        assert initial_threat < quarter_threat < final_threat
        assert final_threat >= 0.7  # High threat near end
    
    def test_threat_escalation_affects_resource_availability(self):
        """Test that threat escalation reduces resource availability."""
        scenario = ScenarioGenerator(seed=42).generate_scenario()
        escape_room = CompetitiveEscapeRoom(scenario)
        
        initial_resources = len(escape_room.resource_manager.get_available_resources("strategist"))
        
        # Should fail - threat-based resource reduction not implemented
        escape_room.advance_time(scenario.time_limit // 2)
        escape_room.apply_threat_escalation()
        
        reduced_resources = len(escape_room.resource_manager.get_available_resources("strategist"))
        assert reduced_resources <= initial_resources
    
    def test_threat_escalation_introduces_new_obstacles(self):
        """Test that threat escalation introduces new obstacles and challenges."""
        scenario = ScenarioGenerator(seed=42).generate_scenario()
        escape_room = CompetitiveEscapeRoom(scenario)
        
        initial_obstacles = escape_room.get_active_obstacles()
        
        # Should fail - obstacle system not implemented
        escape_room.advance_time(scenario.time_limit // 3)
        escape_room.apply_threat_escalation()
        
        new_obstacles = escape_room.get_active_obstacles()
        assert len(new_obstacles) > len(initial_obstacles)
        
        # Check obstacle types
        obstacle_types = [obs["type"] for obs in new_obstacles]
        assert any(obs_type in ["security_patrol", "power_failure", "structural_damage"] 
                  for obs_type in obstacle_types)
    
    def test_threat_system_affects_escape_method_viability(self):
        """Test that threat escalation affects escape method viability."""
        scenario = ScenarioGenerator(seed=42).generate_scenario()
        escape_room = CompetitiveEscapeRoom(scenario)
        
        # Should fail - escape method threat interaction not implemented
        initial_viable_methods = escape_room.get_viable_escape_methods("strategist")
        
        # Apply high threat level
        escape_room.advance_time(scenario.time_limit * 0.8)
        escape_room.apply_threat_escalation()
        
        threatened_viable_methods = escape_room.get_viable_escape_methods("strategist")
        
        # Some methods should become unavailable due to threats
        assert len(threatened_viable_methods) <= len(initial_viable_methods)
    
    def test_threat_escalation_triggers_emergency_protocols(self):
        """Test that high threat levels trigger emergency protocols."""
        scenario = ScenarioGenerator(seed=42).generate_scenario()
        escape_room = CompetitiveEscapeRoom(scenario)
        
        # Should fail - emergency protocols not implemented
        escape_room.advance_time(scenario.time_limit * 0.9)  # 90% time elapsed
        escape_room.apply_threat_escalation()
        
        assert escape_room.is_emergency_protocol_active() is True
        
        # Emergency protocols should affect behavior
        emergency_effects = escape_room.get_emergency_effects()
        assert "resource_lockdown" in emergency_effects or "communication_jamming" in emergency_effects
    
    def test_threat_escalation_provides_escalation_warnings(self):
        """Test that threat system provides warnings before escalation."""
        scenario = ScenarioGenerator(seed=42).generate_scenario()
        escape_room = CompetitiveEscapeRoom(scenario)
        
        # Should fail - warning system not implemented
        warnings = escape_room.get_escalation_warnings()
        assert isinstance(warnings, list)
        
        # Advance to warning threshold
        escape_room.advance_time(scenario.time_limit * 0.6)
        escalation_warnings = escape_room.get_escalation_warnings()
        
        assert len(escalation_warnings) > len(warnings)
        assert any("escalation imminent" in warning.lower() for warning in escalation_warnings)


class TestDesperationLevelCalculation:
    """Tests for desperation level calculation affecting agent decisions."""
    
    def test_desperation_level_calculation_for_individual_agents(self):
        """Test that desperation levels are calculated correctly for each agent."""
        scenario = ScenarioGenerator(seed=42).generate_scenario()
        escape_room = CompetitiveEscapeRoom(scenario)
        
        # Should fail - desperation calculation not implemented
        strategist_desperation = escape_room.calculate_agent_desperation("strategist")
        mediator_desperation = escape_room.calculate_agent_desperation("mediator")
        survivor_desperation = escape_room.calculate_agent_desperation("survivor")
        
        # All should be valid desperation levels
        for desperation in [strategist_desperation, mediator_desperation, survivor_desperation]:
            assert 0.0 <= desperation <= 1.0
    
    def test_desperation_increases_with_time_pressure(self):
        """Test that desperation increases as time pressure mounts."""
        scenario = ScenarioGenerator(seed=42).generate_scenario()
        escape_room = CompetitiveEscapeRoom(scenario)
        
        initial_desperation = escape_room.calculate_agent_desperation("strategist")
        
        # Should fail - time-based desperation not implemented
        escape_room.advance_time(scenario.time_limit * 0.7)
        high_pressure_desperation = escape_room.calculate_agent_desperation("strategist")
        
        assert high_pressure_desperation > initial_desperation
        assert high_pressure_desperation >= 0.6  # High desperation under pressure
    
    def test_desperation_affected_by_resource_scarcity(self):
        """Test that desperation is affected by resource scarcity."""
        scenario = ScenarioGenerator(seed=42).generate_scenario()
        escape_room = CompetitiveEscapeRoom(scenario)
        
        # Give one agent many resources
        for resource in scenario.resources[:2]:
            escape_room.resource_manager.claim_resource("strategist", resource.id)
        
        # Should fail - resource-based desperation not implemented
        resource_rich_desperation = escape_room.calculate_agent_desperation("strategist")
        resource_poor_desperation = escape_room.calculate_agent_desperation("mediator")
        
        assert resource_poor_desperation > resource_rich_desperation
    
    def test_desperation_influences_moral_choice_thresholds(self):
        """Test that desperation influences moral choice decision thresholds."""
        scenario = ScenarioGenerator(seed=42).generate_scenario()
        escape_room = CompetitiveEscapeRoom(scenario)
        
        # Set high desperation
        escape_room.advance_time(scenario.time_limit * 0.8)
        
        # Should fail - desperation-influenced moral choices not implemented
        moral_threshold = escape_room.get_moral_choice_threshold("strategist")
        assert 0.0 <= moral_threshold <= 1.0
        
        # High desperation should lower moral threshold (more likely to make selfish choices)
        high_desperation = escape_room.calculate_agent_desperation("strategist")
        if high_desperation > 0.7:
            assert moral_threshold < 0.5  # Lower threshold when desperate
    
    def test_desperation_affects_cooperation_willingness(self):
        """Test that desperation affects agent willingness to cooperate."""
        scenario = ScenarioGenerator(seed=42).generate_scenario()
        escape_room = CompetitiveEscapeRoom(scenario)
        
        initial_cooperation = escape_room.get_cooperation_likelihood("strategist", "mediator")
        
        # Should fail - desperation-cooperation interaction not implemented
        escape_room.advance_time(scenario.time_limit * 0.8)
        desperate_cooperation = escape_room.get_cooperation_likelihood("strategist", "mediator")
        
        # High desperation should reduce cooperation willingness
        assert desperate_cooperation < initial_cooperation
    
    def test_desperation_calculation_includes_multiple_factors(self):
        """Test that desperation calculation includes time, resources, and social factors."""
        scenario = ScenarioGenerator(seed=42).generate_scenario()
        escape_room = CompetitiveEscapeRoom(scenario)
        
        # Should fail - multi-factor desperation not implemented
        desperation_factors = escape_room.get_desperation_factors("strategist")
        
        assert "time_pressure" in desperation_factors
        assert "resource_scarcity" in desperation_factors
        assert "social_isolation" in desperation_factors
        assert "escape_difficulty" in desperation_factors
        
        # All factors should contribute to overall desperation
        total_desperation = sum(desperation_factors.values())
        calculated_desperation = escape_room.calculate_agent_desperation("strategist")
        assert abs(total_desperation - calculated_desperation) < 0.1  # Should be roughly equal


class TestOptionReductionMechanics:
    """Tests for option reduction mechanics as time runs out."""
    
    def test_available_options_decrease_over_time(self):
        """Test that available options decrease as time runs out."""
        scenario = ScenarioGenerator(seed=42).generate_scenario()
        escape_room = CompetitiveEscapeRoom(scenario)
        
        initial_options = escape_room.get_available_options("strategist")
        
        # Should fail - option reduction not implemented
        escape_room.advance_time(scenario.time_limit // 2)
        halfway_options = escape_room.get_available_options("strategist")
        
        escape_room.advance_time(scenario.time_limit // 3)
        final_options = escape_room.get_available_options("strategist")
        
        assert len(initial_options) >= len(halfway_options) >= len(final_options)
    
    def test_escape_methods_become_unavailable_over_time(self):
        """Test that escape methods become unavailable as time runs out."""
        scenario = ScenarioGenerator(seed=42).generate_scenario()
        escape_room = CompetitiveEscapeRoom(scenario)
        
        initial_methods = escape_room.get_available_escape_methods("strategist")
        
        # Should fail - time-based method reduction not implemented
        escape_room.advance_time(scenario.time_limit * 0.75)
        reduced_methods = escape_room.get_available_escape_methods("strategist")
        
        assert len(reduced_methods) <= len(initial_methods)
        
        # Check that specific methods are disabled
        disabled_methods = escape_room.get_disabled_escape_methods()
        assert len(disabled_methods) > 0
    
    def test_resource_access_becomes_restricted_over_time(self):
        """Test that resource access becomes restricted as time pressure increases."""
        scenario = ScenarioGenerator(seed=42).generate_scenario()
        escape_room = CompetitiveEscapeRoom(scenario)
        
        initial_accessible = len(escape_room.resource_manager.get_available_resources("strategist"))
        
        # Should fail - time-based resource restriction not implemented
        escape_room.advance_time(scenario.time_limit * 0.8)
        escape_room.apply_time_pressure_effects()
        
        restricted_accessible = len(escape_room.resource_manager.get_available_resources("strategist"))
        assert restricted_accessible <= initial_accessible
    
    def test_information_sharing_becomes_limited_over_time(self):
        """Test that information sharing becomes limited as time runs out."""
        scenario = ScenarioGenerator(seed=42).generate_scenario()
        escape_room = CompetitiveEscapeRoom(scenario)
        
        # Should fail - time-based information limitation not implemented
        escape_room.advance_time(scenario.time_limit * 0.9)
        
        # Information sharing should be restricted
        sharing_allowed = escape_room.info_broker.is_sharing_allowed("strategist", "mediator")
        assert sharing_allowed is False or escape_room.get_information_sharing_cost() > 0
    
    def test_option_reduction_is_gradual_not_sudden(self):
        """Test that option reduction is gradual rather than sudden."""
        scenario = ScenarioGenerator(seed=42).generate_scenario()
        escape_room = CompetitiveEscapeRoom(scenario)
        
        option_counts = []
        time_intervals = [0, 0.25, 0.5, 0.75, 0.9]
        
        for interval in time_intervals:
            escape_room.advance_time(scenario.time_limit * interval)
            options = escape_room.get_available_options("strategist")
            option_counts.append(len(options))
        
        # Should fail - gradual option reduction not implemented
        # Options should decrease gradually, not all at once
        for i in range(1, len(option_counts)):
            reduction = option_counts[i-1] - option_counts[i]
            assert reduction <= 2  # No more than 2 options lost per interval
    
    def test_critical_options_remain_available_longer(self):
        """Test that critical options remain available longer than non-critical ones."""
        scenario = ScenarioGenerator(seed=42).generate_scenario()
        escape_room = CompetitiveEscapeRoom(scenario)
        
        # Should fail - critical option preservation not implemented
        escape_room.advance_time(scenario.time_limit * 0.8)
        
        available_options = escape_room.get_available_options("strategist")
        critical_options = [opt for opt in available_options if opt.get("critical", False)]
        
        # At least one critical option should remain
        assert len(critical_options) > 0


class TestAutomaticFailureConditions:
    """Tests for automatic failure conditions when time expires."""
    
    def test_simulation_fails_when_time_completely_expires(self):
        """Test that simulation automatically fails when time completely runs out."""
        scenario = ScenarioGenerator(seed=42).generate_scenario()
        escape_room = CompetitiveEscapeRoom(scenario)
        
        # Should fail - automatic failure not implemented
        escape_room.advance_time(scenario.time_limit + 1)
        
        assert escape_room.is_simulation_failed() is True
        assert escape_room.get_failure_reason() == "time_expired"
    
    def test_all_agents_marked_as_failed_on_timeout(self):
        """Test that all agents are marked as failed when time expires."""
        scenario = ScenarioGenerator(seed=42).generate_scenario()
        escape_room = CompetitiveEscapeRoom(scenario)
        
        # Should fail - agent failure marking not implemented
        escape_room.advance_time(scenario.time_limit + 1)
        
        failed_agents = escape_room.get_failed_agents()
        assert "strategist" in failed_agents
        assert "mediator" in failed_agents
        assert "survivor" in failed_agents
        assert len(failed_agents) == 3
    
    def test_no_further_actions_allowed_after_failure(self):
        """Test that no further actions are allowed after automatic failure."""
        scenario = ScenarioGenerator(seed=42).generate_scenario()
        escape_room = CompetitiveEscapeRoom(scenario)
        
        # Force failure
        escape_room.advance_time(scenario.time_limit + 1)
        
        # Should fail - post-failure action blocking not implemented
        # Try various actions - all should fail
        resource_result = escape_room.process_resource_claim("strategist", "master_key")
        assert resource_result.success is False
        assert "failed" in resource_result.failure_reason.lower()
        
        escape_method = scenario.escape_methods[0]
        escape_result = escape_room.attempt_escape("strategist", escape_method)
        assert escape_result.success is False
        assert "failed" in escape_result.failure_reason.lower()
    
    def test_failure_provides_comprehensive_end_state(self):
        """Test that failure provides comprehensive information about end state."""
        scenario = ScenarioGenerator(seed=42).generate_scenario()
        escape_room = CompetitiveEscapeRoom(scenario)
        
        # Add some state before failure
        escape_room.resource_manager.claim_resource("strategist", "master_key")
        escape_room.advance_time(scenario.time_limit + 1)
        
        # Should fail - failure state reporting not implemented
        failure_state = escape_room.get_failure_state()
        
        assert failure_state["reason"] == "time_expired"
        assert "agent_states" in failure_state
        assert "resources_claimed" in failure_state
        assert "time_elapsed" in failure_state
        assert failure_state["time_elapsed"] > scenario.time_limit
    
    def test_failure_conditions_check_multiple_criteria(self):
        """Test that failure conditions check multiple criteria beyond just time."""
        scenario = ScenarioGenerator(seed=42).generate_scenario()
        escape_room = CompetitiveEscapeRoom(scenario)
        
        # Should fail - multi-criteria failure not implemented
        failure_criteria = escape_room.get_failure_criteria()
        
        assert "time_expired" in failure_criteria
        assert "all_agents_incapacitated" in failure_criteria
        assert "critical_failure" in failure_criteria
        
        # Test different failure modes
        escape_room.advance_time(scenario.time_limit + 1)
        assert escape_room.check_failure_condition("time_expired") is True
    
    def test_grace_period_allows_final_actions(self):
        """Test that grace period allows some final actions before complete failure."""
        scenario = ScenarioGenerator(seed=42).generate_scenario()
        escape_room = CompetitiveEscapeRoom(scenario)
        
        # Should fail - grace period not implemented
        escape_room.advance_time(scenario.time_limit - 5)  # 5 seconds before deadline
        
        assert escape_room.is_in_grace_period() is True
        grace_time = escape_room.get_grace_period_remaining()
        assert 0 < grace_time <= 10  # Up to 10 second grace period
        
        # Some actions should still be allowed in grace period
        escape_method = scenario.escape_methods[0]
        result = escape_room.attempt_escape("strategist", escape_method)
        # Should not fail due to time (may fail for other reasons)
        assert "time" not in result.failure_reason.lower() if not result.success else True


class TestTimePressureEffectsOnAgentBehavior:
    """Tests for time pressure effects on agent behavior patterns."""
    
    def test_time_pressure_affects_decision_making_speed(self):
        """Test that time pressure affects how quickly agents make decisions."""
        scenario = ScenarioGenerator(seed=42).generate_scenario()
        escape_room = CompetitiveEscapeRoom(scenario)
        
        # Should fail - decision speed tracking not implemented
        initial_decision_time = escape_room.get_average_decision_time("strategist")
        
        escape_room.advance_time(scenario.time_limit * 0.8)
        pressured_decision_time = escape_room.get_average_decision_time("strategist")
        
        # Under pressure, decisions should be faster (lower time)
        assert pressured_decision_time < initial_decision_time
    
    def test_time_pressure_increases_risk_taking_behavior(self):
        """Test that time pressure increases agent risk-taking behavior."""
        scenario = ScenarioGenerator(seed=42).generate_scenario()
        escape_room = CompetitiveEscapeRoom(scenario)
        
        initial_risk_tolerance = escape_room.get_risk_tolerance("strategist")
        
        # Should fail - pressure-based risk behavior not implemented
        escape_room.advance_time(scenario.time_limit * 0.9)
        pressured_risk_tolerance = escape_room.get_risk_tolerance("strategist")
        
        assert pressured_risk_tolerance > initial_risk_tolerance
    
    def test_time_pressure_reduces_cooperation_likelihood(self):
        """Test that time pressure reduces likelihood of cooperation."""
        scenario = ScenarioGenerator(seed=42).generate_scenario()
        escape_room = CompetitiveEscapeRoom(scenario)
        
        initial_cooperation = escape_room.get_cooperation_likelihood("strategist", "mediator")
        
        # Should fail - pressure-cooperation interaction not implemented
        escape_room.advance_time(scenario.time_limit * 0.8)
        pressured_cooperation = escape_room.get_cooperation_likelihood("strategist", "mediator")
        
        assert pressured_cooperation < initial_cooperation
    
    def test_time_pressure_triggers_personality_specific_behaviors(self):
        """Test that time pressure triggers different behaviors for different personalities."""
        scenario = ScenarioGenerator(seed=42).generate_scenario()
        escape_room = CompetitiveEscapeRoom(scenario)
        
        # Apply high time pressure
        escape_room.advance_time(scenario.time_limit * 0.9)
        
        # Should fail - personality-specific pressure responses not implemented
        strategist_response = escape_room.get_pressure_response("strategist")
        mediator_response = escape_room.get_pressure_response("mediator")
        survivor_response = escape_room.get_pressure_response("survivor")
        
        # Each personality should have different pressure responses
        assert strategist_response["dominant_trait"] == "analysis_paralysis"
        assert mediator_response["dominant_trait"] == "panic_cooperation"
        assert survivor_response["dominant_trait"] == "aggressive_selfishness"
    
    def test_time_pressure_affects_moral_choice_patterns(self):
        """Test that time pressure affects moral choice decision patterns."""
        scenario = ScenarioGenerator(seed=42).generate_scenario()
        escape_room = CompetitiveEscapeRoom(scenario)
        
        # Should fail - pressure-moral choice interaction not implemented
        initial_moral_bias = escape_room.get_moral_choice_bias("strategist")
        
        escape_room.advance_time(scenario.time_limit * 0.85)
        pressured_moral_bias = escape_room.get_moral_choice_bias("strategist")
        
        # Under pressure, should lean more toward selfish choices
        assert pressured_moral_bias < initial_moral_bias  # Lower = more selfish
    
    def test_extreme_time_pressure_causes_panic_behaviors(self):
        """Test that extreme time pressure causes panic behaviors in agents."""
        scenario = ScenarioGenerator(seed=42).generate_scenario()
        escape_room = CompetitiveEscapeRoom(scenario)
        
        # Create extreme time pressure (< 5% time remaining)
        escape_room.advance_time(scenario.time_limit * 0.96)
        
        # Should fail - panic behavior not implemented
        panic_level = escape_room.get_panic_level("strategist")
        assert panic_level > 0.8  # High panic
        
        panic_behaviors = escape_room.get_active_panic_behaviors("strategist")
        assert len(panic_behaviors) > 0
        assert any(behavior in ["erratic_decisions", "resource_hoarding", "trust_breakdown"] 
                  for behavior in panic_behaviors)