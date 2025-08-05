"""
Unit tests for CompetitiveAgentState tracking system.
Tests agent state management including resource ownership, secrets known, trust relationships,
moral choice history, and ethical burden calculation with state synchronization.
"""
import pytest
from datetime import datetime
from unittest.mock import Mock
from src.escape_room_sim.competitive.models import (
    TrustAction, MoralChoice, MoralDilemma, SecretInformation, 
    ScarceResource, CompetitiveScenario, TrustRelationship
)
from src.escape_room_sim.competitive.competitive_agent_state import CompetitiveAgentState
from src.escape_room_sim.competitive.scenario_generator import ScenarioGenerator


class TestCompetitiveAgentStateInitialization:
    """Tests for CompetitiveAgentState dataclass initialization."""
    
    def test_competitive_agent_state_initializes_with_agent_id(self):
        """Test that CompetitiveAgentState initializes correctly with agent ID."""
        # This should fail initially - CompetitiveAgentState doesn't exist yet
        agent_state = CompetitiveAgentState("strategist")
        
        # Should fail - CompetitiveAgentState not implemented
        assert agent_state.agent_id == "strategist"
        assert agent_state.resources_owned == []
        assert agent_state.secrets_known == []
        assert agent_state.trust_relationships == {}
        assert agent_state.moral_choice_history == []
        assert agent_state.ethical_burden == 0.0
    
    def test_competitive_agent_state_validates_agent_id_input(self):
        """Test that CompetitiveAgentState validates agent ID input."""
        # Should fail - validation not implemented
        with pytest.raises(ValueError, match="Agent ID cannot be empty"):
            CompetitiveAgentState("")
        
        with pytest.raises(ValueError, match="Agent ID cannot be empty"):
            CompetitiveAgentState(None)
        
        with pytest.raises(ValueError, match="Agent ID cannot be empty"):
            CompetitiveAgentState("   ")
    
    def test_competitive_agent_state_initializes_with_default_values(self):
        """Test that CompetitiveAgentState initializes with appropriate default values."""
        agent_state = CompetitiveAgentState("mediator")
        
        # Should fail - default initialization not implemented
        assert isinstance(agent_state.resources_owned, list)
        assert isinstance(agent_state.secrets_known, list)
        assert isinstance(agent_state.trust_relationships, dict)
        assert isinstance(agent_state.moral_choice_history, list)
        assert agent_state.ethical_burden == 0.0
        assert agent_state.last_updated is not None
        assert isinstance(agent_state.last_updated, datetime)


class TestResourceOwnershipTracking:
    """Tests for resource ownership tracking per agent."""
    
    def test_add_resource_ownership_tracks_resources_per_agent(self):
        """Test that agent state tracks resource ownership correctly."""
        agent_state = CompetitiveAgentState("survivor")
        
        # Should fail - resource ownership methods not implemented
        agent_state.add_resource("flashlight")
        agent_state.add_resource("lockpick")
        
        assert "flashlight" in agent_state.resources_owned
        assert "lockpick" in agent_state.resources_owned
        assert len(agent_state.resources_owned) == 2
    
    def test_remove_resource_ownership_updates_tracking(self):
        """Test that resource removal is tracked correctly.""" 
        agent_state = CompetitiveAgentState("strategist")
        agent_state.add_resource("key")
        agent_state.add_resource("hammer")
        
        # Should fail - resource removal not implemented
        removed = agent_state.remove_resource("key")
        
        assert removed is True
        assert "key" not in agent_state.resources_owned
        assert "hammer" in agent_state.resources_owned
        assert len(agent_state.resources_owned) == 1
    
    def test_remove_nonexistent_resource_returns_false(self):
        """Test that removing non-existent resource returns False."""
        agent_state = CompetitiveAgentState("mediator")
        
        # Should fail - resource removal validation not implemented
        removed = agent_state.remove_resource("nonexistent")
        assert removed is False
    
    def test_has_resource_checks_ownership_correctly(self):
        """Test that resource ownership checking works correctly."""
        agent_state = CompetitiveAgentState("survivor")
        agent_state.add_resource("rope")
        
        # Should fail - resource checking not implemented
        assert agent_state.has_resource("rope") is True
        assert agent_state.has_resource("missing") is False
    
    def test_get_resource_count_returns_correct_count(self):
        """Test that resource counting works correctly."""
        agent_state = CompetitiveAgentState("strategist")
        agent_state.add_resource("tool1")
        agent_state.add_resource("tool2")
        agent_state.add_resource("tool3")
        
        # Should fail - resource counting not implemented
        assert agent_state.get_resource_count() == 3
    
    def test_resource_ownership_prevents_duplicates(self):
        """Test that adding the same resource twice doesn't create duplicates."""
        agent_state = CompetitiveAgentState("mediator")
        agent_state.add_resource("duplicate_item")
        agent_state.add_resource("duplicate_item")
        
        # Should fail - duplicate prevention not implemented
        assert agent_state.get_resource_count() == 1
        assert agent_state.resources_owned.count("duplicate_item") == 1


class TestSecretsKnownTracking:
    """Tests for secrets known tracking for information asymmetry."""
    
    def test_add_secret_knowledge_tracks_information_per_agent(self):
        """Test that agent state tracks secret information correctly.""" 
        agent_state = CompetitiveAgentState("strategist")
        
        secret1 = SecretInformation("escape_route", "Hidden tunnel behind bookshelf", 0.8, 0.3, ["tunnel_escape"])
        secret2 = SecretInformation("key_location", "Master key in desk drawer", 0.9, 0.2, ["door_escape"])
        
        # Should fail - secret tracking methods not implemented
        agent_state.add_secret(secret1)
        agent_state.add_secret(secret2)
        
        assert len(agent_state.secrets_known) == 2
        assert secret1 in agent_state.secrets_known
        assert secret2 in agent_state.secrets_known
    
    def test_has_secret_checks_knowledge_correctly(self):
        """Test that secret knowledge checking works correctly."""
        agent_state = CompetitiveAgentState("mediator")
        secret = SecretInformation("combination", "Safe code is 1234", 0.7, 0.4, ["safe_escape"])
        agent_state.add_secret(secret)
        
        # Should fail - secret checking not implemented
        assert agent_state.has_secret("combination") is True
        assert agent_state.has_secret("unknown") is False
    
    def test_get_secret_by_id_returns_correct_secret(self):
        """Test that secret retrieval by ID works correctly."""
        agent_state = CompetitiveAgentState("survivor")
        secret = SecretInformation("weakness", "Guard has bad eyesight", 0.6, 0.5, ["stealth_escape"])
        agent_state.add_secret(secret)
        
        # Should fail - secret retrieval not implemented
        retrieved = agent_state.get_secret("weakness")
        assert retrieved == secret
        assert retrieved.content == "Guard has bad eyesight"
    
    def test_get_secret_by_id_returns_none_for_unknown(self):
        """Test that retrieving unknown secret returns None."""
        agent_state = CompetitiveAgentState("strategist")
        
        # Should fail - secret retrieval validation not implemented
        retrieved = agent_state.get_secret("unknown_secret")
        assert retrieved is None
    
    def test_get_secrets_count_returns_correct_count(self):
        """Test that secret counting works correctly."""
        agent_state = CompetitiveAgentState("mediator")
        
        for i in range(3):
            secret = SecretInformation(f"secret_{i}", f"Description {i}", 0.5, 0.3, [f"method_{i}"])
            agent_state.add_secret(secret)
        
        # Should fail - secret counting not implemented
        assert agent_state.get_secrets_count() == 3
    
    def test_secret_knowledge_prevents_duplicates(self):
        """Test that adding the same secret twice doesn't create duplicates."""
        agent_state = CompetitiveAgentState("survivor")
        secret = SecretInformation("duplicate", "Duplicate information", 0.5, 0.3, ["method"])
        
        agent_state.add_secret(secret)
        agent_state.add_secret(secret)
        
        # Should fail - duplicate prevention not implemented
        assert agent_state.get_secrets_count() == 1


class TestTrustRelationshipStorage:
    """Tests for trust relationship storage for each agent."""
    
    def test_update_trust_relationship_stores_trust_data(self):
        """Test that trust relationships are stored correctly per agent."""
        agent_state = CompetitiveAgentState("strategist")
        
        # Should fail - trust relationship methods not implemented
        agent_state.update_trust_relationship("mediator", 0.6)
        agent_state.update_trust_relationship("survivor", -0.3)
        
        assert agent_state.trust_relationships["mediator"] == 0.6
        assert agent_state.trust_relationships["survivor"] == -0.3
        assert len(agent_state.trust_relationships) == 2
    
    def test_get_trust_level_returns_correct_value(self):
        """Test that trust level retrieval works correctly."""
        agent_state = CompetitiveAgentState("mediator")
        agent_state.update_trust_relationship("strategist", 0.4)
        
        # Should fail - trust level retrieval not implemented
        trust_level = agent_state.get_trust_level("strategist")
        assert trust_level == 0.4
    
    def test_get_trust_level_returns_zero_for_unknown_agent(self):
        """Test that unknown agent returns neutral trust level."""
        agent_state = CompetitiveAgentState("survivor")
        
        # Should fail - default trust handling not implemented
        trust_level = agent_state.get_trust_level("unknown_agent")
        assert trust_level == 0.0
    
    def test_has_trust_relationship_checks_existence(self):
        """Test that trust relationship existence checking works."""
        agent_state = CompetitiveAgentState("strategist")
        agent_state.update_trust_relationship("mediator", 0.2)
        
        # Should fail - relationship checking not implemented
        assert agent_state.has_trust_relationship("mediator") is True
        assert agent_state.has_trust_relationship("survivor") is False
    
    def test_get_all_trust_relationships_returns_copy(self):
        """Test that trust relationships can be retrieved safely."""
        agent_state = CompetitiveAgentState("mediator")
        agent_state.update_trust_relationship("strategist", 0.5)
        agent_state.update_trust_relationship("survivor", -0.2)
        
        # Should fail - trust relationship retrieval not implemented
        relationships = agent_state.get_all_trust_relationships()
        assert len(relationships) == 2
        assert relationships["strategist"] == 0.5
        assert relationships["survivor"] == -0.2
        
        # Modify returned dict shouldn't affect internal state
        relationships["new_agent"] = 1.0
        assert "new_agent" not in agent_state.trust_relationships
    
    def test_trust_relationship_update_overwrites_existing(self):
        """Test that trust relationship updates overwrite existing values."""
        agent_state = CompetitiveAgentState("survivor")
        agent_state.update_trust_relationship("mediator", 0.3)
        agent_state.update_trust_relationship("mediator", -0.7)
        
        # Should fail - trust update behavior not implemented
        assert agent_state.get_trust_level("mediator") == -0.7
        assert len(agent_state.trust_relationships) == 1


class TestMoralChoiceHistoryAndEthicalBurden:
    """Tests for moral choice history and ethical burden calculation."""
    
    def test_add_moral_choice_tracks_history(self):
        """Test that moral choices are tracked in agent history."""
        agent_state = CompetitiveAgentState("strategist")
        
        choice = MoralChoice(
            description="Sacrifice teammate for escape",
            survival_benefit=0.9,
            ethical_cost=0.8,
            trust_impact={"others": -0.6},
            consequences=["guilt", "isolation"]
        )
        
        # Should fail - moral choice tracking not implemented
        agent_state.add_moral_choice(choice)
        
        assert len(agent_state.moral_choice_history) == 1
        assert agent_state.moral_choice_history[0]["choice"] == choice
        assert "timestamp" in agent_state.moral_choice_history[0]
    
    def test_moral_choice_history_includes_context(self):
        """Test that moral choice history includes contextual information."""
        agent_state = CompetitiveAgentState("mediator")
        
        choice = MoralChoice(
            description="Share resources equally",
            survival_benefit=0.4,
            ethical_cost=0.1,
            trust_impact={"others": 0.5},
            consequences=["cooperation"]
        )
        
        context = {"time_pressure": 0.7, "desperation_level": 0.6}
        
        # Should fail - contextual tracking not implemented
        agent_state.add_moral_choice(choice, context)
        
        history_entry = agent_state.moral_choice_history[0]
        assert history_entry["choice"] == choice
        assert history_entry["context"] == context
        assert isinstance(history_entry["timestamp"], datetime)
    
    def test_calculate_ethical_burden_sums_costs(self):
        """Test that ethical burden calculation sums moral costs correctly."""
        agent_state = CompetitiveAgentState("survivor")
        
        choice1 = MoralChoice("Selfish act 1", 0.8, 0.6, {}, [])
        choice2 = MoralChoice("Selfish act 2", 0.7, 0.4, {}, [])
        
        agent_state.add_moral_choice(choice1)
        agent_state.add_moral_choice(choice2)
        
        # Should fail - burden calculation not implemented
        burden = agent_state.calculate_ethical_burden()
        assert burden == 1.0  # 0.6 + 0.4
        assert agent_state.ethical_burden == 1.0
    
    def test_get_moral_choice_count_returns_correct_count(self):
        """Test that moral choice counting works correctly."""
        agent_state = CompetitiveAgentState("strategist")
        
        for i in range(3):
            choice = MoralChoice(f"Choice {i}", 0.5, 0.3, {}, [])
            agent_state.add_moral_choice(choice)
        
        # Should fail - choice counting not implemented
        assert agent_state.get_moral_choice_count() == 3
    
    def test_get_recent_moral_choices_returns_latest_choices(self):
        """Test that recent moral choices can be retrieved with limit."""
        agent_state = CompetitiveAgentState("mediator")
        
        choices = []
        for i in range(5):
            choice = MoralChoice(f"Choice {i}", 0.5, 0.2, {}, [])
            choices.append(choice)
            agent_state.add_moral_choice(choice)
        
        # Should fail - recent choices retrieval not implemented
        recent = agent_state.get_recent_moral_choices(3)
        assert len(recent) == 3
        # Should return most recent choices (last 3)
        assert recent[0]["choice"] == choices[4]  # Most recent first
        assert recent[1]["choice"] == choices[3]
        assert recent[2]["choice"] == choices[2]
    
    def test_ethical_burden_updates_automatically(self):
        """Test that ethical burden updates automatically when choices are added."""
        agent_state = CompetitiveAgentState("survivor")
        
        assert agent_state.ethical_burden == 0.0
        
        choice = MoralChoice("Bad choice", 0.9, 0.7, {}, [])
        agent_state.add_moral_choice(choice)
        
        # Should fail - automatic burden update not implemented
        assert agent_state.ethical_burden == 0.7


class TestAgentStateUpdatesAndSynchronization:
    """Tests for agent state updates and synchronization."""
    
    def test_agent_state_tracks_last_updated_timestamp(self):
        """Test that agent state tracks when it was last updated."""
        agent_state = CompetitiveAgentState("strategist")
        initial_time = agent_state.last_updated
        
        # Small delay to ensure timestamp difference
        import time
        time.sleep(0.01)
        
        # Should fail - timestamp update not implemented
        agent_state.add_resource("new_item")
        
        assert agent_state.last_updated > initial_time
    
    def test_sync_with_external_state_updates_correctly(self):
        """Test that agent state can be synchronized with external systems."""
        agent_state = CompetitiveAgentState("mediator")
        
        external_state = {
            "resources_owned": ["tool1", "tool2"],
            "secrets_known": [
                SecretInformation("secret1", "desc1", 0.5, 0.3, ["method1"])
            ],
            "trust_relationships": {"strategist": 0.4, "survivor": -0.2},
            "ethical_burden": 0.6
        }
        
        # Should fail - synchronization not implemented
        agent_state.sync_with_external_state(external_state)
        
        assert agent_state.resources_owned == ["tool1", "tool2"]
        assert len(agent_state.secrets_known) == 1
        assert agent_state.trust_relationships["strategist"] == 0.4
        assert agent_state.ethical_burden == 0.6
    
    def test_get_state_summary_returns_complete_state(self):
        """Test that complete state summary can be retrieved."""
        agent_state = CompetitiveAgentState("survivor")
        
        # Add some state
        agent_state.add_resource("weapon")
        secret = SecretInformation("escape", "Back door", 0.8, 0.2, ["back_door_escape"])
        agent_state.add_secret(secret)
        agent_state.update_trust_relationship("mediator", -0.3)
        choice = MoralChoice("Betray ally", 0.9, 0.8, {}, [])
        agent_state.add_moral_choice(choice)
        
        # Should fail - state summary not implemented
        summary = agent_state.get_state_summary()
        
        assert summary["agent_id"] == "survivor"
        assert summary["resources_count"] == 1
        assert summary["secrets_count"] == 1
        assert summary["trust_relationships_count"] == 1
        assert summary["moral_choices_count"] == 1
        assert summary["ethical_burden"] == 0.8
        assert "last_updated" in summary
    
    def test_validate_state_consistency_detects_problems(self):
        """Test that state validation can detect consistency problems."""
        agent_state = CompetitiveAgentState("strategist")
        
        # Manually corrupt state to test validation
        agent_state.ethical_burden = -1.0  # Invalid negative burden
        
        # Should fail - state validation not implemented
        validation_result = agent_state.validate_state_consistency()
        
        assert validation_result["is_valid"] is False
        assert any("ethical_burden" in error for error in validation_result["errors"])
    
    def test_reset_agent_state_clears_all_data(self):
        """Test that agent state can be completely reset."""
        agent_state = CompetitiveAgentState("mediator")
        
        # Add data to all categories
        agent_state.add_resource("item")
        secret = SecretInformation("info", "details", 0.5, 0.3, ["info_method"])
        agent_state.add_secret(secret)
        agent_state.update_trust_relationship("strategist", 0.5)
        choice = MoralChoice("choice", 0.5, 0.3, {}, [])
        agent_state.add_moral_choice(choice)
        
        # Should fail - reset functionality not implemented
        agent_state.reset_state()
        
        assert len(agent_state.resources_owned) == 0
        assert len(agent_state.secrets_known) == 0
        assert len(agent_state.trust_relationships) == 0
        assert len(agent_state.moral_choice_history) == 0
        assert agent_state.ethical_burden == 0.0
    
    def test_agent_state_supports_deep_copy(self):
        """Test that agent state can be deep copied safely."""
        agent_state = CompetitiveAgentState("survivor")
        
        agent_state.add_resource("original_item")
        secret = SecretInformation("secret", "info", 0.7, 0.4, ["secret_method"])
        agent_state.add_secret(secret)
        
        # Should fail - deep copy support not implemented
        copied_state = agent_state.deep_copy()
        
        assert copied_state.agent_id == agent_state.agent_id
        assert copied_state.resources_owned == agent_state.resources_owned
        assert len(copied_state.secrets_known) == len(agent_state.secrets_known)
        
        # Modify copy shouldn't affect original
        copied_state.add_resource("new_item")
        assert "new_item" not in agent_state.resources_owned