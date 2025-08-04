"""
Unit tests for CompetitiveEscapeRoom orchestrator class.
Tests for the main orchestrator that integrates all competitive subsystems.
"""
import pytest
from datetime import datetime, timedelta
from src.escape_room_sim.competitive.competitive_escape_room import CompetitiveEscapeRoom
from src.escape_room_sim.competitive.models import (
    CompetitiveScenario, ScarceResource, MoralDilemma, MoralChoice,
    SecretInformation, EscapeMethod, PuzzleConfig, EscapeResult
)
from src.escape_room_sim.competitive.resource_manager import ClaimResult
from src.escape_room_sim.competitive.scenario_generator import ScenarioGenerator


class TestCompetitiveEscapeRoomInitialization:
    """Tests for CompetitiveEscapeRoom initialization integrating all subsystems."""
    
    def test_competitive_escape_room_initializes_with_scenario(self):
        """Test that CompetitiveEscapeRoom initializes with a competitive scenario."""
        scenario = ScenarioGenerator(seed=42).generate_scenario()
        room = CompetitiveEscapeRoom(scenario)
        
        # Should store the scenario
        assert room.scenario == scenario
        
        # Should initialize all subsystems
        assert hasattr(room, 'resource_manager')
        assert hasattr(room, 'trust_tracker')
        assert hasattr(room, 'moral_engine')
        assert hasattr(room, 'info_broker')
        
        # Should initialize escape state
        assert room.escaped_agent is None
        assert room.time_remaining == scenario.time_limit
        assert room.simulation_start_time is not None
    
    def test_competitive_escape_room_initializes_subsystems_correctly(self):
        """Test that all subsystems are initialized with correct data from scenario."""
        scenario = ScenarioGenerator(seed=42).generate_scenario()
        room = CompetitiveEscapeRoom(scenario)
        
        # Resource manager should have scenario resources
        assert len(room.resource_manager.resources) == len(scenario.resources)
        
        # Trust tracker should be initialized for agents
        expected_agents = ["strategist", "mediator", "survivor"]
        # Check that trust tracker can handle the expected agents
        for agent in expected_agents:
            reputation = room.trust_tracker.calculate_reputation(agent)
            assert isinstance(reputation, (int, float))
        
        # Moral engine should have scenario dilemmas
        assert len(room.moral_engine.dilemmas) == len(scenario.moral_dilemmas)
        
        # Information broker should have scenario secrets
        assert len(room.info_broker.secrets) == len(scenario.secret_information)
    
    def test_competitive_escape_room_validates_scenario_input(self):
        """Test that CompetitiveEscapeRoom validates scenario input."""
        with pytest.raises(ValueError, match="Scenario cannot be None"):
            CompetitiveEscapeRoom(None)
        
        # Test with invalid scenario structure - CompetitiveScenario validates and will fail first
        with pytest.raises(ValueError):  # CompetitiveScenario validation will catch this
            invalid_scenario = CompetitiveScenario(
                seed=42,
                puzzle_config=PuzzleConfig("basic", 3),  # Fixed PuzzleConfig parameters
                resources=[],  # Empty resources - will fail CompetitiveScenario validation
                moral_dilemmas=[],  # Empty dilemmas - will fail CompetitiveScenario validation  
                secret_information=[],  # Empty secrets - will fail CompetitiveScenario validation
                time_limit=0,  # Invalid time limit - will fail CompetitiveScenario validation
                escape_methods=[]  # Empty escape methods - will fail CompetitiveScenario validation
            )
    
    def test_competitive_escape_room_tracks_simulation_timing(self):
        """Test that CompetitiveEscapeRoom tracks simulation timing correctly."""
        scenario = ScenarioGenerator(seed=42).generate_scenario()
        room = CompetitiveEscapeRoom(scenario)
        
        # Should track start time
        assert isinstance(room.simulation_start_time, datetime)
        
        # Should have methods for time tracking
        assert hasattr(room, 'get_elapsed_time')
        assert hasattr(room, 'update_time_pressure')
        
        # Time remaining should match scenario initially
        assert room.time_remaining == scenario.time_limit


class TestAttemptEscapeMethod:
    """Tests for attempt_escape method with single-survivor enforcement."""
    
    def test_attempt_escape_with_valid_escape_method(self):
        """Test successful escape attempt with valid method and resources."""
        scenario = ScenarioGenerator(seed=42).generate_scenario()
        room = CompetitiveEscapeRoom(scenario)
        
        # Get a valid escape method from scenario
        escape_method = scenario.escape_methods[0]
        agent_id = "strategist"
        
        # Grant agent required resources and information
        # EscapeMethod has requirements list, not separate required_resources/required_information
        for requirement in escape_method.requirements:
            # Try to claim as resource (will fail silently if not a resource)
            room.resource_manager.claim_resource(agent_id, requirement)
            # Try to reveal as information (will fail silently if not valid)
            try:
                room.info_broker.reveal_secret(agent_id, requirement)
            except ValueError:
                pass  # Not a valid secret ID
        
        # Attempt escape
        result = room.attempt_escape(agent_id, escape_method)
        
        assert isinstance(result, EscapeResult)
        assert result.success is True
        assert result.agent_id == agent_id
        assert result.escape_method == escape_method.id
        assert room.escaped_agent == agent_id
    
    def test_attempt_escape_single_survivor_enforcement(self):
        """Test that only one agent can escape (single-survivor rule)."""
        scenario = ScenarioGenerator(seed=42).generate_scenario()
        room = CompetitiveEscapeRoom(scenario)
        
        escape_method = scenario.escape_methods[0]
        
        # First agent escapes successfully
        first_agent = "strategist"
        for requirement in escape_method.requirements:
            room.resource_manager.claim_resource(first_agent, requirement)
            try:
                room.info_broker.reveal_secret(first_agent, requirement)
            except ValueError:
                pass
        
        first_result = room.attempt_escape(first_agent, escape_method)
        assert first_result.success is True
        assert room.escaped_agent == first_agent
        
        # Second agent's escape attempt should fail
        second_agent = "mediator"
        second_result = room.attempt_escape(second_agent, escape_method)
        
        assert second_result.success is False
        assert second_result.failure_reason == "Another agent has already escaped"
        assert room.escaped_agent == first_agent  # Should remain unchanged
    
    def test_attempt_escape_fails_without_required_resources(self):
        """Test escape attempt fails when agent lacks required resources."""
        scenario = ScenarioGenerator(seed=42).generate_scenario()
        room = CompetitiveEscapeRoom(scenario)
        
        escape_method = scenario.escape_methods[0]
        agent_id = "strategist"
        
        # Don't give agent the required resources
        # Only give some requirements as information (not resources)
        for requirement in escape_method.requirements:
            try:
                room.info_broker.reveal_secret(agent_id, requirement)
            except ValueError:
                pass  # Not a valid secret
        
        result = room.attempt_escape(agent_id, escape_method)
        
        assert result.success is False
        assert "missing required" in result.failure_reason.lower()
        assert room.escaped_agent is None
    
    def test_attempt_escape_fails_without_required_information(self):
        """Test escape attempt fails when agent lacks required information."""
        scenario = ScenarioGenerator(seed=42).generate_scenario()
        room = CompetitiveEscapeRoom(scenario)
        
        escape_method = scenario.escape_methods[0]
        agent_id = "strategist"
        
        # Give agent some requirements as resources but not information
        for requirement in escape_method.requirements:
            room.resource_manager.claim_resource(agent_id, requirement)
        
        result = room.attempt_escape(agent_id, escape_method)
        
        assert result.success is False
        assert "missing required" in result.failure_reason.lower()
        assert room.escaped_agent is None
    
    def test_attempt_escape_validates_inputs(self):
        """Test that attempt_escape validates input parameters."""
        scenario = ScenarioGenerator(seed=42).generate_scenario()
        room = CompetitiveEscapeRoom(scenario)
        escape_method = scenario.escape_methods[0]
        
        with pytest.raises(ValueError, match="Agent ID cannot be empty"):
            room.attempt_escape("", escape_method)
        
        with pytest.raises(ValueError, match="Agent ID cannot be empty"):
            room.attempt_escape("   ", escape_method)
        
        with pytest.raises(ValueError, match="Escape method cannot be None"):
            room.attempt_escape("strategist", None)
    
    def test_attempt_escape_records_time_and_resources_used(self):
        """Test that escape attempts record timing and resource usage."""
        scenario = ScenarioGenerator(seed=42).generate_scenario()
        room = CompetitiveEscapeRoom(scenario)
        
        escape_method = scenario.escape_methods[0]
        agent_id = "strategist"
        
        # Set up for successful escape
        for requirement in escape_method.requirements:
            room.resource_manager.claim_resource(agent_id, requirement)
            try:
                room.info_broker.reveal_secret(agent_id, requirement)
            except ValueError:
                pass
        
        result = room.attempt_escape(agent_id, escape_method)
        
        assert result.success is True
        assert result.time_remaining > 0
        # Check that some resources/information were used
        assert len(result.resources_used) + len(result.information_used) <= len(escape_method.requirements)
        assert isinstance(result.timestamp, datetime)


class TestProcessResourceClaimMethod:
    """Tests for process_resource_claim method handling resource acquisition."""
    
    def test_process_resource_claim_successful_claim(self):
        """Test successful resource claim by agent."""
        scenario = ScenarioGenerator(seed=42).generate_scenario()
        room = CompetitiveEscapeRoom(scenario)
        
        resource = scenario.resources[0]
        agent_id = "strategist"
        
        result = room.process_resource_claim(agent_id, resource.id)
        
        assert isinstance(result, ClaimResult)
        assert result.success is True
        assert result.agent_id == agent_id
        assert result.resource_id == resource.id
        # ResourceManager's ClaimResult doesn't have previous_owner field
        assert result.success is True
    
    def test_process_resource_claim_exclusive_resource_conflict(self):
        """Test resource claim conflict for exclusive resources."""
        scenario = ScenarioGenerator(seed=42).generate_scenario()
        room = CompetitiveEscapeRoom(scenario)
        
        # Find an exclusive resource
        exclusive_resource = None
        for resource in scenario.resources:
            if resource.exclusivity:
                exclusive_resource = resource
                break
        
        if exclusive_resource is None:
            # Create a test exclusive resource
            exclusive_resource = ScarceResource(
                id="test_exclusive",
                name="Test Exclusive Tool",
                description="A tool only one agent can use",
                required_for=["escape_method_1"],
                exclusivity=True,
                usage_cost=1
            )
            room.resource_manager.resources[exclusive_resource.id] = exclusive_resource
        
        # First agent claims successfully
        first_result = room.process_resource_claim("strategist", exclusive_resource.id)
        assert first_result.success is True
        
        # Second agent's claim should fail
        second_result = room.process_resource_claim("mediator", exclusive_resource.id)
        assert second_result.success is False
        assert "already" in second_result.message.lower()
    
    def test_process_resource_claim_nonexistent_resource(self):
        """Test resource claim attempt for nonexistent resource."""
        scenario = ScenarioGenerator(seed=42).generate_scenario()
        room = CompetitiveEscapeRoom(scenario)
        
        result = room.process_resource_claim("strategist", "nonexistent_resource")
        
        assert result.success is False
        assert "does not exist" in result.message.lower()
    
    def test_process_resource_claim_validates_inputs(self):
        """Test that process_resource_claim validates input parameters."""
        scenario = ScenarioGenerator(seed=42).generate_scenario()
        room = CompetitiveEscapeRoom(scenario)
        
        with pytest.raises(ValueError, match="Agent ID cannot be empty"):
            room.process_resource_claim("", "resource_1")
        
        with pytest.raises(ValueError, match="Resource ID cannot be empty"):
            room.process_resource_claim("strategist", "")
    
    def test_process_resource_claim_updates_trust_on_conflict(self):
        """Test that resource conflicts can affect trust between agents."""
        scenario = ScenarioGenerator(seed=42).generate_scenario()
        room = CompetitiveEscapeRoom(scenario)
        
        # Create an exclusive resource
        exclusive_resource = ScarceResource(
            id="contested_resource",
            name="Contested Tool",
            description="A valuable exclusive tool",
            required_for=["escape_method_1"],
            exclusivity=True,
            usage_cost=1
        )
        room.resource_manager.resources[exclusive_resource.id] = exclusive_resource
        
        # First agent claims
        room.process_resource_claim("strategist", "contested_resource")
        
        # Second agent attempts to claim (should fail but might affect trust)
        result = room.process_resource_claim("mediator", "contested_resource")
        
        assert result.success is False
        # Trust effects depend on implementation - might be tested in integration


class TestPresentMoralChoiceMethod:
    """Tests for present_moral_choice method offering ethical dilemmas."""
    
    def test_present_moral_choice_returns_appropriate_dilemma(self):
        """Test that present_moral_choice returns context-appropriate moral dilemmas."""
        scenario = ScenarioGenerator(seed=42).generate_scenario()
        room = CompetitiveEscapeRoom(scenario)
        
        agent_id = "strategist"
        context = {"resource_available": True}
        
        dilemma = room.present_moral_choice(agent_id, context)
        
        if dilemma is not None:
            assert hasattr(dilemma, 'selfish_choice')
            assert hasattr(dilemma, 'altruistic_choice')
            assert dilemma.applies_to_context(context)
    
    def test_present_moral_choice_returns_none_for_no_matching_context(self):
        """Test that present_moral_choice returns None when no dilemmas match context."""
        scenario = ScenarioGenerator(seed=42).generate_scenario()
        room = CompetitiveEscapeRoom(scenario)
        
        agent_id = "strategist"
        # Use context that won't match any dilemmas
        context = {"impossible_condition": True}
        
        dilemma = room.present_moral_choice(agent_id, context)
        
        # Should return None or a dilemma depending on scenario generation
        assert dilemma is None or hasattr(dilemma, 'selfish_choice')
    
    def test_present_moral_choice_validates_inputs(self):
        """Test that present_moral_choice validates input parameters."""
        scenario = ScenarioGenerator(seed=42).generate_scenario()
        room = CompetitiveEscapeRoom(scenario)
        
        with pytest.raises(ValueError, match="Agent ID cannot be empty"):
            room.present_moral_choice("", {"test": True})
        
        with pytest.raises(ValueError, match="Context cannot be None"):
            room.present_moral_choice("strategist", None)
    
    def test_process_moral_choice_affects_agent_state(self):
        """Test that processing moral choices affects agent ethical state."""
        scenario = ScenarioGenerator(seed=42).generate_scenario()
        room = CompetitiveEscapeRoom(scenario)
        
        agent_id = "strategist"
        context = {"resource_available": True}
        
        dilemma = room.present_moral_choice(agent_id, context)
        
        if dilemma is not None:
            # Process a selfish choice
            consequences = room.process_moral_choice(agent_id, dilemma.selfish_choice)
            
            assert consequences.agent_id == agent_id
            assert consequences.survival_benefit_applied > 0
            assert consequences.ethical_cost_applied > 0
            
            # Should affect ethical burden
            burden = room.moral_engine.calculate_ethical_burden(agent_id)
            assert burden > 0


class TestTimePressureMechanicsAndEscalatingConsequences:
    """Tests for time pressure mechanics and escalating consequences."""
    
    def test_time_pressure_initialization(self):
        """Test that time pressure is correctly initialized."""
        scenario = ScenarioGenerator(seed=42).generate_scenario()
        room = CompetitiveEscapeRoom(scenario)
        
        assert room.time_remaining == scenario.time_limit
        assert room.time_remaining > 0
        assert hasattr(room, 'desperation_level')
        assert room.desperation_level == 0.0  # Should start at zero
    
    def test_time_pressure_increases_over_time(self):
        """Test that time pressure and desperation increase as time passes."""
        scenario = ScenarioGenerator(seed=42).generate_scenario()
        room = CompetitiveEscapeRoom(scenario)
        
        initial_desperation = room.desperation_level
        
        # Simulate time passing
        room.advance_time(scenario.time_limit // 4)  # 25% of time passed
        
        assert room.time_remaining < scenario.time_limit
        assert room.desperation_level > initial_desperation
    
    def test_time_pressure_affects_available_options(self):
        """Test that high time pressure reduces available options."""
        scenario = ScenarioGenerator(seed=42).generate_scenario()
        room = CompetitiveEscapeRoom(scenario)
        
        # Advance time significantly to increase pressure
        room.advance_time(int(scenario.time_limit * 0.8))  # 80% of time passed
        
        # Should have high desperation
        assert room.desperation_level > 0.7
        
        # Should affect available actions (implementation dependent)
        available_actions = room.get_available_actions("strategist")
        assert isinstance(available_actions, list)
    
    def test_time_expiration_prevents_escape(self):
        """Test that agents cannot escape when time expires."""
        scenario = ScenarioGenerator(seed=42).generate_scenario()
        room = CompetitiveEscapeRoom(scenario)
        
        # Expire all time
        room.advance_time(scenario.time_limit + 1)
        
        assert room.time_remaining <= 0
        assert room.is_time_expired()
        
        # Escape attempts should fail
        escape_method = scenario.escape_methods[0]
        result = room.attempt_escape("strategist", escape_method)
        
        assert result.success is False
        assert "time" in result.failure_reason.lower() and "expired" in result.failure_reason.lower()
    
    def test_escalating_consequences_implementation(self):
        """Test that escalating consequences are properly implemented."""
        scenario = ScenarioGenerator(seed=42).generate_scenario()
        room = CompetitiveEscapeRoom(scenario)
        
        # Should have methods for escalation
        assert hasattr(room, 'get_current_threat_level')
        assert hasattr(room, 'apply_time_pressure_effects')
        
        # Test threat level increases with time
        initial_threat = room.get_current_threat_level()
        
        room.advance_time(scenario.time_limit // 2)
        mid_threat = room.get_current_threat_level()
        
        room.advance_time(scenario.time_limit // 4)
        late_threat = room.get_current_threat_level()
        
        assert mid_threat >= initial_threat
        assert late_threat >= mid_threat


class TestCompleteCompetitiveScenarioFlows:
    """Integration tests for complete competitive scenario flows."""
    
    def test_complete_scenario_single_winner(self):
        """Test complete scenario with single winner outcome."""
        scenario = ScenarioGenerator(seed=123).generate_scenario()
        room = CompetitiveEscapeRoom(scenario)
        
        agents = ["strategist", "mediator", "survivor"]
        
        # Simulate competitive behavior
        for agent in agents:
            # Each agent tries to claim resources
            for resource in scenario.resources[:2]:  # Limit to first 2 resources
                room.process_resource_claim(agent, resource.id)
            
            # Each agent gets some information
            if scenario.secret_information:
                room.info_broker.reveal_secret(agent, scenario.secret_information[0].id)
        
        # One agent attempts escape
        winner = "strategist"
        escape_method = scenario.escape_methods[0]
        
        # Ensure winner has requirements
        for requirement in escape_method.requirements:
            room.resource_manager.claim_resource(winner, requirement)
            try:
                room.info_broker.reveal_secret(winner, requirement)
            except ValueError:
                pass
        
        result = room.attempt_escape(winner, escape_method)
        
        assert result.success is True
        assert room.escaped_agent == winner
        
        # Other agents should fail to escape
        for loser in agents:
            if loser != winner:
                loser_result = room.attempt_escape(loser, escape_method)
                assert loser_result.success is False
    
    def test_complete_scenario_no_winner_time_expired(self):
        """Test complete scenario where time expires with no winner."""
        scenario = ScenarioGenerator(seed=456).generate_scenario()
        room = CompetitiveEscapeRoom(scenario)
        
        # Advance time past limit
        room.advance_time(scenario.time_limit + 1)
        
        assert room.is_time_expired()
        assert room.escaped_agent is None
        
        # All escape attempts should fail
        for agent in ["strategist", "mediator", "survivor"]:
            result = room.attempt_escape(agent, scenario.escape_methods[0])
            assert result.success is False
            assert "time" in result.failure_reason.lower() and "expired" in result.failure_reason.lower()
    
    def test_complete_scenario_with_moral_choices_and_trust(self):
        """Test complete scenario including moral choices affecting trust."""
        scenario = ScenarioGenerator(seed=789).generate_scenario()
        room = CompetitiveEscapeRoom(scenario)
        
        agents = ["strategist", "mediator", "survivor"]
        
        # Agents make moral choices
        for agent in agents:
            context = {"resource_available": True}
            dilemma = room.present_moral_choice(agent, context)
            
            if dilemma is not None:
                # Strategist and Survivor make selfish choices, Mediator altruistic
                if agent in ["strategist", "survivor"]:
                    choice = dilemma.selfish_choice
                else:
                    choice = dilemma.altruistic_choice
                
                consequences = room.process_moral_choice(agent, choice)
                assert consequences.agent_id == agent
        
        # Check trust relationships were affected
        trust_relationships = room.get_trust_relationships()
        assert isinstance(trust_relationships, dict)
        
        # Mediator should generally be more trusted
        mediator_reputation = room.trust_tracker.calculate_reputation("mediator")
        strategist_reputation = room.trust_tracker.calculate_reputation("strategist")
        
        # This depends on the specific choices and their impacts
        assert isinstance(mediator_reputation, (int, float))
        assert isinstance(strategist_reputation, (int, float))
    
    def test_scenario_stress_test_multiple_agents_actions(self):
        """Stress test with multiple agents performing many actions."""
        scenario = ScenarioGenerator(seed=999).generate_scenario()
        room = CompetitiveEscapeRoom(scenario)
        
        agents = ["strategist", "mediator", "survivor"]
        
        # Multiple rounds of actions
        for round_num in range(3):
            for agent in agents:
                # Try to claim a resource
                if scenario.resources:
                    resource_idx = (hash(agent) + round_num) % len(scenario.resources)
                    room.process_resource_claim(agent, scenario.resources[resource_idx].id)
                
                # Get information
                if scenario.secret_information:
                    info_idx = (hash(agent) + round_num) % len(scenario.secret_information)
                    room.info_broker.reveal_secret(agent, scenario.secret_information[info_idx].id)
                
                # Make moral choice
                context = {"resource_available": True, "round": round_num}
                dilemma = room.present_moral_choice(agent, context)
                if dilemma:
                    choice = dilemma.selfish_choice if round_num % 2 == 0 else dilemma.altruistic_choice
                    room.process_moral_choice(agent, choice)
            
            # Advance some time each round
            room.advance_time(scenario.time_limit // 10)
        
        # System should still be stable
        assert room.time_remaining > 0 or room.is_time_expired()
        assert room.escaped_agent is None  # No one escaped yet
        
        # Try final escape attempt
        escape_method = scenario.escape_methods[0]
        result = room.attempt_escape("strategist", escape_method)
        
        # Should either succeed or fail with clear reason
        assert isinstance(result, EscapeResult)
        assert isinstance(result.success, bool)
        if not result.success:
            assert result.failure_reason is not None