"""
Unit tests for InformationBroker class.
Tests for knowledge asymmetry management in competitive scenarios.
"""
import pytest
from datetime import datetime
from src.escape_room_sim.competitive.information_broker import InformationBroker
from src.escape_room_sim.competitive.models import SecretInformation


class TestInformationBrokerInitialization:
    """Test InformationBroker initialization with secret information list."""
    
    def test_information_broker_initializes_with_secret_list(self):
        """Test that InformationBroker initializes with a list of secrets."""
        # Create test secrets
        secret1 = SecretInformation.create_code_secret("code1", "1234", ["escape_route_1"])
        secret2 = SecretInformation.create_location_secret("location1", "hidden room", ["escape_route_2"])
        
        broker = InformationBroker([secret1, secret2])
        
        assert len(broker.secrets) == 2
        assert "code1" in broker.secrets
        assert "location1" in broker.secrets
        assert broker.secrets["code1"] == secret1
        assert broker.secrets["location1"] == secret2
        assert broker.agent_knowledge == {}
        assert broker.sharing_history == []
    
    def test_information_broker_initializes_with_empty_list(self):
        """Test that InformationBroker can initialize with empty secret list."""
        broker = InformationBroker([])
        
        assert broker.secrets == {}
        assert broker.agent_knowledge == {}
        assert broker.sharing_history == []
    
    def test_information_broker_initializes_tracking_structures(self):
        """Test that InformationBroker initializes all tracking structures."""
        broker = InformationBroker([])
        
        # Should have all required attributes
        assert hasattr(broker, 'secrets')
        assert hasattr(broker, 'agent_knowledge')
        assert hasattr(broker, 'sharing_history')
        
        # Should be proper types
        assert isinstance(broker.secrets, dict)
        assert isinstance(broker.agent_knowledge, dict)
        assert isinstance(broker.sharing_history, list)
    
    def test_information_broker_validates_secret_list(self):
        """Test that InformationBroker validates secret list input."""
        with pytest.raises(ValueError, match="Secrets list cannot be None"):
            InformationBroker(None)
        
        # Test with duplicate secret IDs
        secret1 = SecretInformation.create_code_secret("duplicate", "1234", ["route1"])
        secret2 = SecretInformation.create_code_secret("duplicate", "5678", ["route2"])
        
        with pytest.raises(ValueError, match="Duplicate secret IDs not allowed"):
            InformationBroker([secret1, secret2])


class TestRevealSecretMethod:
    """Test reveal_secret method granting agent access to information."""
    
    def test_reveal_secret_grants_agent_access(self):
        """Test that reveal_secret grants agent access to specific information."""
        secret = SecretInformation.create_code_secret("code1", "1234", ["escape_route"])
        broker = InformationBroker([secret])
        
        broker.reveal_secret("agent1", "code1")
        
        # Agent should now have access to the secret
        assert "agent1" in broker.agent_knowledge
        assert "code1" in broker.agent_knowledge["agent1"]
    
    def test_reveal_secret_to_multiple_agents(self):
        """Test revealing the same secret to multiple agents."""
        secret = SecretInformation.create_code_secret("code1", "1234", ["escape_route"])
        broker = InformationBroker([secret])
        
        broker.reveal_secret("agent1", "code1")
        broker.reveal_secret("agent2", "code1")
        
        # Both agents should have access
        assert "code1" in broker.agent_knowledge["agent1"]
        assert "code1" in broker.agent_knowledge["agent2"]
    
    def test_reveal_multiple_secrets_to_agent(self):
        """Test revealing multiple secrets to the same agent."""
        secret1 = SecretInformation.create_code_secret("code1", "1234", ["route1"])
        secret2 = SecretInformation.create_location_secret("location1", "room", ["route2"])
        broker = InformationBroker([secret1, secret2])
        
        broker.reveal_secret("agent1", "code1")
        broker.reveal_secret("agent1", "location1")
        
        # Agent should have access to both secrets
        assert len(broker.agent_knowledge["agent1"]) == 2
        assert "code1" in broker.agent_knowledge["agent1"]
        assert "location1" in broker.agent_knowledge["agent1"]
    
    def test_reveal_secret_validates_inputs(self):
        """Test that reveal_secret validates input parameters."""
        secret = SecretInformation.create_code_secret("code1", "1234", ["route"])
        broker = InformationBroker([secret])
        
        with pytest.raises(ValueError, match="Agent ID cannot be empty"):
            broker.reveal_secret("", "code1")
        
        with pytest.raises(ValueError, match="Agent ID cannot be empty"):
            broker.reveal_secret("   ", "code1")
        
        with pytest.raises(ValueError, match="Secret ID cannot be empty"):
            broker.reveal_secret("agent1", "")
        
        with pytest.raises(ValueError, match="Secret not found"):
            broker.reveal_secret("agent1", "nonexistent_secret")
    
    def test_reveal_secret_is_idempotent(self):
        """Test that revealing the same secret to the same agent multiple times is safe."""
        secret = SecretInformation.create_code_secret("code1", "1234", ["route"])
        broker = InformationBroker([secret])
        
        broker.reveal_secret("agent1", "code1")
        broker.reveal_secret("agent1", "code1")  # Reveal again
        
        # Should still only have one copy
        assert len(broker.agent_knowledge["agent1"]) == 1
        assert "code1" in broker.agent_knowledge["agent1"]


class TestShareInformationMethod:
    """Test share_information method handling inter-agent knowledge transfer."""
    
    def test_share_information_transfers_knowledge(self):
        """Test that share_information transfers knowledge between agents."""
        secret = SecretInformation.create_code_secret("code1", "1234", ["route"])
        broker = InformationBroker([secret])
        
        # Give secret to agent1
        broker.reveal_secret("agent1", "code1")
        
        # Agent1 shares with agent2
        result = broker.share_information("agent1", "agent2", "code1")
        
        assert result is True
        assert "code1" in broker.agent_knowledge["agent2"]
    
    def test_share_information_requires_sender_knowledge(self):
        """Test that share_information requires sender to know the secret."""
        secret = SecretInformation.create_code_secret("code1", "1234", ["route"])
        broker = InformationBroker([secret])
        
        # Agent1 doesn't know the secret, tries to share
        result = broker.share_information("agent1", "agent2", "code1")
        
        assert result is False
        assert "agent2" not in broker.agent_knowledge or "code1" not in broker.agent_knowledge.get("agent2", set())
    
    def test_share_information_records_sharing_history(self):
        """Test that share_information records sharing events."""
        secret = SecretInformation.create_code_secret("code1", "1234", ["route"])
        broker = InformationBroker([secret])
        
        broker.reveal_secret("agent1", "code1")
        broker.share_information("agent1", "agent2", "code1")
        
        # Should record sharing event
        assert len(broker.sharing_history) == 1
        sharing_event = broker.sharing_history[0]
        assert sharing_event["from_agent"] == "agent1"
        assert sharing_event["to_agent"] == "agent2"
        assert sharing_event["secret_id"] == "code1"
        assert "timestamp" in sharing_event
        assert isinstance(sharing_event["timestamp"], datetime)
    
    def test_share_information_calculates_sharing_risk(self):
        """Test that share_information considers sharing risk."""
        # Create secret with high sharing risk
        high_risk_secret = SecretInformation(
            id="risky_info",
            content="Dangerous information",
            value=0.9,
            sharing_risk=0.8,
            required_for=["escape"]
        )
        broker = InformationBroker([high_risk_secret])
        
        broker.reveal_secret("agent1", "risky_info")
        result = broker.share_information("agent1", "agent2", "risky_info")
        
        # Should still share but record the risk
        assert result is True
        sharing_event = broker.sharing_history[0]
        assert sharing_event["sharing_risk"] == 0.8
    
    def test_share_information_validates_inputs(self):
        """Test that share_information validates input parameters."""
        secret = SecretInformation.create_code_secret("code1", "1234", ["route"])
        broker = InformationBroker([secret])
        
        with pytest.raises(ValueError, match="Agent ID cannot be empty"):
            broker.share_information("", "agent2", "code1")
        
        with pytest.raises(ValueError, match="Agent ID cannot be empty"):
            broker.share_information("agent1", "", "code1")
        
        with pytest.raises(ValueError, match="Secret ID cannot be empty"):
            broker.share_information("agent1", "agent2", "")
        
        with pytest.raises(ValueError, match="Agent cannot share with itself"):
            broker.share_information("agent1", "agent1", "code1")
        
        with pytest.raises(ValueError, match="Secret not found"):
            broker.share_information("agent1", "agent2", "nonexistent")


class TestGetAgentKnowledgeMethod:
    """Test get_agent_knowledge method returning agent-specific information."""
    
    def test_get_agent_knowledge_returns_known_secrets(self):
        """Test that get_agent_knowledge returns all secrets known to agent."""
        secret1 = SecretInformation.create_code_secret("code1", "1234", ["route1"])
        secret2 = SecretInformation.create_location_secret("location1", "room", ["route2"])
        broker = InformationBroker([secret1, secret2])
        
        broker.reveal_secret("agent1", "code1")
        broker.reveal_secret("agent1", "location1")
        
        knowledge = broker.get_agent_knowledge("agent1")
        
        assert len(knowledge) == 2
        assert secret1 in knowledge
        assert secret2 in knowledge
    
    def test_get_agent_knowledge_returns_empty_for_unknown_agent(self):
        """Test that get_agent_knowledge returns empty list for unknown agent."""
        secret = SecretInformation.create_code_secret("code1", "1234", ["route"])
        broker = InformationBroker([secret])
        
        knowledge = broker.get_agent_knowledge("unknown_agent")
        
        assert knowledge == []
    
    def test_get_agent_knowledge_filters_by_agent(self):
        """Test that get_agent_knowledge only returns secrets for specific agent."""
        secret1 = SecretInformation.create_code_secret("code1", "1234", ["route1"])
        secret2 = SecretInformation.create_location_secret("location1", "room", ["route2"])
        broker = InformationBroker([secret1, secret2])
        
        broker.reveal_secret("agent1", "code1")
        broker.reveal_secret("agent2", "location1")
        
        agent1_knowledge = broker.get_agent_knowledge("agent1")
        agent2_knowledge = broker.get_agent_knowledge("agent2")
        
        assert len(agent1_knowledge) == 1
        assert secret1 in agent1_knowledge
        assert secret2 not in agent1_knowledge
        
        assert len(agent2_knowledge) == 1
        assert secret2 in agent2_knowledge
        assert secret1 not in agent2_knowledge
    
    def test_get_agent_knowledge_validates_input(self):
        """Test that get_agent_knowledge validates agent ID."""
        broker = InformationBroker([])
        
        with pytest.raises(ValueError, match="Agent ID cannot be empty"):
            broker.get_agent_knowledge("")
        
        with pytest.raises(ValueError, match="Agent ID cannot be empty"):
            broker.get_agent_knowledge("   ")


class TestInformationSharingHistoryAndRiskAssessment:
    """Test information sharing history and risk assessment functionality."""
    
    def test_sharing_history_tracks_all_sharing_events(self):
        """Test that sharing history tracks all information sharing events."""
        secret1 = SecretInformation.create_code_secret("code1", "1234", ["route1"])
        secret2 = SecretInformation.create_location_secret("location1", "room", ["route2"])
        broker = InformationBroker([secret1, secret2])
        
        # Set up initial knowledge
        broker.reveal_secret("agent1", "code1")
        broker.reveal_secret("agent2", "location1")
        
        # Multiple sharing events
        broker.share_information("agent1", "agent2", "code1")
        broker.share_information("agent2", "agent1", "location1")
        broker.share_information("agent1", "agent3", "code1")
        
        # Should track all events
        assert len(broker.sharing_history) == 3
        
        # Verify event details
        events = broker.sharing_history
        assert events[0]["from_agent"] == "agent1"
        assert events[0]["to_agent"] == "agent2"
        assert events[0]["secret_id"] == "code1"
        
        assert events[1]["from_agent"] == "agent2"
        assert events[1]["to_agent"] == "agent1"
        assert events[1]["secret_id"] == "location1"
    
    def test_sharing_history_includes_risk_assessment(self):
        """Test that sharing history includes risk assessment for each event."""
        risky_secret = SecretInformation(
            id="risky_code",
            content="High-risk information",
            value=0.9,
            sharing_risk=0.7,
            required_for=["escape"]
        )
        broker = InformationBroker([risky_secret])
        
        broker.reveal_secret("agent1", "risky_code")
        broker.share_information("agent1", "agent2", "risky_code")
        
        sharing_event = broker.sharing_history[0]
        assert sharing_event["sharing_risk"] == 0.7
        assert sharing_event["information_value"] == 0.9
    
    def test_get_sharing_history_for_agent(self):
        """Test getting sharing history filtered by agent."""
        secret = SecretInformation.create_code_secret("code1", "1234", ["route"])
        broker = InformationBroker([secret])
        
        broker.reveal_secret("agent1", "code1")
        broker.share_information("agent1", "agent2", "code1")
        broker.share_information("agent1", "agent3", "code1")
        
        # Get sharing history for agent1 as sender
        agent1_sent = broker.get_sharing_history_for_agent("agent1", role="sender")
        assert len(agent1_sent) == 2
        
        # Get sharing history for agent2 as receiver
        agent2_received = broker.get_sharing_history_for_agent("agent2", role="receiver")
        assert len(agent2_received) == 1
        assert agent2_received[0]["from_agent"] == "agent1"
    
    def test_calculate_information_exposure_risk(self):
        """Test calculating overall information exposure risk for an agent."""
        secrets = [
            SecretInformation.create_code_secret("low_risk", "1234", ["route1"]),  # risk 0.6
            SecretInformation.create_location_secret("high_risk", "room", ["route2"])  # risk 0.8
        ]
        broker = InformationBroker(secrets)
        
        broker.reveal_secret("agent1", "low_risk")
        broker.reveal_secret("agent1", "high_risk")
        
        # Share both secrets
        broker.share_information("agent1", "agent2", "low_risk")
        broker.share_information("agent1", "agent3", "high_risk")
        
        exposure_risk = broker.calculate_information_exposure_risk("agent1")
        
        # Should be average of sharing risks for shared secrets
        expected_risk = (0.6 + 0.8) / 2
        assert abs(exposure_risk - expected_risk) < 1e-10


class TestKnowledgeAsymmetryEnforcementAndAccessValidation:
    """Test knowledge asymmetry enforcement and access validation."""
    
    def test_agent_can_only_access_known_information(self):
        """Test that agents can only access information they know."""
        secret1 = SecretInformation.create_code_secret("code1", "1234", ["route1"])
        secret2 = SecretInformation.create_code_secret("code2", "5678", ["route2"])
        broker = InformationBroker([secret1, secret2])
        
        # Only give code1 to agent1
        broker.reveal_secret("agent1", "code1")
        
        agent1_knowledge = broker.get_agent_knowledge("agent1")
        
        # Agent1 should only know code1, not code2
        assert len(agent1_knowledge) == 1
        assert secret1 in agent1_knowledge
        assert secret2 not in agent1_knowledge
    
    def test_information_asymmetry_maintained_between_agents(self):
        """Test that information asymmetry is maintained between agents."""
        secrets = [
            SecretInformation.create_code_secret("agent1_secret", "1111", ["route1"]),
            SecretInformation.create_code_secret("agent2_secret", "2222", ["route2"]),
            SecretInformation.create_code_secret("shared_secret", "0000", ["route3"])
        ]
        broker = InformationBroker(secrets)
        
        # Distribute secrets asymmetrically
        broker.reveal_secret("agent1", "agent1_secret")
        broker.reveal_secret("agent1", "shared_secret")
        broker.reveal_secret("agent2", "agent2_secret")
        broker.reveal_secret("agent2", "shared_secret")
        
        agent1_knowledge = broker.get_agent_knowledge("agent1")
        agent2_knowledge = broker.get_agent_knowledge("agent2")
        
        # Each agent should have their exclusive secret plus shared secret
        assert len(agent1_knowledge) == 2
        assert len(agent2_knowledge) == 2
        
        # Verify asymmetry
        agent1_secrets = {s.id for s in agent1_knowledge}
        agent2_secrets = {s.id for s in agent2_knowledge}
        
        assert "agent1_secret" in agent1_secrets
        assert "agent1_secret" not in agent2_secrets
        assert "agent2_secret" in agent2_secrets
        assert "agent2_secret" not in agent1_secrets
        assert "shared_secret" in agent1_secrets
        assert "shared_secret" in agent2_secrets
    
    def test_validate_agent_has_required_information(self):
        """Test validating that agent has required information for actions."""
        escape_secret = SecretInformation.create_code_secret("escape_code", "EXIT", ["main_exit"])
        broker = InformationBroker([escape_secret])
        
        # Agent doesn't have the required information
        has_required = broker.validate_agent_has_required_information("agent1", ["escape_code"])
        assert has_required is False
        
        # Give agent the required information
        broker.reveal_secret("agent1", "escape_code")
        has_required = broker.validate_agent_has_required_information("agent1", ["escape_code"])
        assert has_required is True
    
    def test_get_information_asymmetry_report(self):
        """Test generating information asymmetry report for analysis."""
        secrets = [
            SecretInformation.create_code_secret("code1", "1111", ["route1"]),
            SecretInformation.create_code_secret("code2", "2222", ["route2"]),
            SecretInformation.create_code_secret("code3", "3333", ["route3"])
        ]
        broker = InformationBroker(secrets)
        
        # Distribute information
        broker.reveal_secret("agent1", "code1")
        broker.reveal_secret("agent1", "code2")
        broker.reveal_secret("agent2", "code2")
        broker.reveal_secret("agent2", "code3")
        
        report = broker.get_information_asymmetry_report()
        
        # Report should show distribution
        assert report["total_secrets"] == 3
        assert report["agents_with_knowledge"] == 2
        assert "agent1" in report["agent_knowledge_count"]
        assert "agent2" in report["agent_knowledge_count"]
        assert report["agent_knowledge_count"]["agent1"] == 2
        assert report["agent_knowledge_count"]["agent2"] == 2
        assert report["shared_secrets"] == 1  # code2 is shared
        assert report["exclusive_secrets"] == 2  # code1, code3 are exclusive
    
    def test_enforce_information_access_validation(self):
        """Test that information access is properly validated."""
        secret = SecretInformation.create_code_secret("restricted", "SECRET", ["route"])
        broker = InformationBroker([secret])
        
        # Try to access secret without having it
        with pytest.raises(ValueError, match="Agent does not have access to secret"):
            broker.get_secret_content("agent1", "restricted")
        
        # Give access and try again
        broker.reveal_secret("agent1", "restricted")
        content = broker.get_secret_content("agent1", "restricted")
        assert content == "Access code: SECRET"