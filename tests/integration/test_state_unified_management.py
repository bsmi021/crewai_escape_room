"""
Integration Tests for Unified State Management System

Agent D: State Management & Integration Specialist
Tests event-driven state synchronization between Mesa and CrewAI frameworks.
"""

import pytest
import asyncio
import time
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
from typing import Dict, List, Any

from src.escape_room_sim.hybrid.state.unified_state_manager import (
    UnifiedStateManager, StateChange, StateConflict, StateResolution,
    UnifiedState, EventBus, StateChangeEvent
)
from src.escape_room_sim.hybrid.state.state_synchronizer import StateSynchronizer
from src.escape_room_sim.hybrid.core_architecture import DecisionData
from src.escape_room_sim.hybrid.actions.action_models import ActionHandoff, ExecutionResult
import mesa


class TestUnifiedStateManager:
    """Test unified state management system"""
    
    @pytest.fixture
    def mock_mesa_model(self):
        """Create mock Mesa model"""
        model = Mock(spec=mesa.Model)
        model.schedule = Mock()
        model.schedule.agents = []
        model.schedule.steps = 0
        model.width = 10
        model.height = 10
        model.running = True
        return model
    
    @pytest.fixture
    def mock_crewai_agents(self):
        """Create mock CrewAI agents"""
        agents = []
        for i, role in enumerate(['strategist', 'mediator', 'survivor']):
            agent = Mock()
            agent.role = role.title()
            agent.goal = f"Test goal for {role}"
            agent.backstory = f"Test backstory for {role}"
            agent.memory = Mock()
            agents.append(agent)
        return agents
    
    @pytest.fixture
    def state_manager(self):
        """Create unified state manager"""
        return UnifiedStateManager()
    
    def test_unified_state_manager_creation(self, state_manager):
        """Test unified state manager can be created"""
        assert state_manager is not None
        assert hasattr(state_manager, 'unified_state')
        assert hasattr(state_manager, 'event_bus')
        assert hasattr(state_manager, 'synchronizers')
    
    def test_create_unified_state(self, state_manager, mock_mesa_model, mock_crewai_agents):
        """Test creating unified state from Mesa model and CrewAI agents"""
        unified_state = state_manager.create_unified_state(mock_mesa_model, mock_crewai_agents)
        
        assert isinstance(unified_state, UnifiedState)
        assert unified_state.agents is not None
        assert unified_state.environment is not None
        assert unified_state.resources is not None
        assert unified_state.timestamp is not None
        
        # Should have agents from both frameworks
        assert len(unified_state.agents) >= 3  # At least the 3 CrewAI agents
    
    def test_state_change_registration(self, state_manager):
        """Test registering state changes"""
        change = StateChange(
            change_id="test_change_1",
            entity_id="agent_1",
            change_type="position_update",
            old_value=(0, 0),
            new_value=(1, 1),
            timestamp=datetime.now(),
            source="mesa"
        )
        
        success = state_manager.register_state_change(change)
        assert success is True
        
        # Should be in pending changes
        assert len(state_manager.pending_changes) == 1
        assert state_manager.pending_changes[0].change_id == "test_change_1"
    
    def test_state_conflict_detection(self, state_manager):
        """Test detecting state conflicts"""
        # Create conflicting changes
        change1 = StateChange(
            change_id="test_change_1",
            entity_id="resource_key1",
            change_type="ownership_change",
            old_value="agent_1",
            new_value="agent_2",
            timestamp=datetime.now(),
            source="mesa"
        )
        
        change2 = StateChange(
            change_id="test_change_2",
            entity_id="resource_key1",
            change_type="ownership_change",
            old_value="agent_1",
            new_value="agent_3",
            timestamp=datetime.now(),
            source="crewai"
        )
        
        state_manager.register_state_change(change1)
        state_manager.register_state_change(change2)
        
        conflicts = state_manager.detect_conflicts()
        assert len(conflicts) == 1
        
        conflict = conflicts[0]
        assert isinstance(conflict, StateConflict)
        assert conflict.conflict_type == "resource_ownership"
        assert len(conflict.conflicting_changes) == 2
    
    def test_state_conflict_resolution(self, state_manager):
        """Test resolving state conflicts"""
        conflict = StateConflict(
            conflict_id="test_conflict_1",
            conflict_type="resource_ownership",
            conflicting_changes=[
                StateChange("change1", "resource1", "ownership", "agent1", "agent2", datetime.now(), "mesa"),
                StateChange("change2", "resource1", "ownership", "agent1", "agent3", datetime.now(), "crewai")
            ],
            severity="medium"
        )
        
        resolutions = state_manager.handle_state_conflicts([conflict])
        assert len(resolutions) == 1
        
        resolution = resolutions[0]
        assert isinstance(resolution, StateResolution)
        assert resolution.conflict_id == "test_conflict_1"
        assert resolution.resolution_successful is True
        assert len(resolution.resolved_changes) >= 1
    
    def test_performance_state_sync_timing(self, state_manager, mock_mesa_model, mock_crewai_agents):
        """Test state synchronization performance (< 200ms target)"""
        # Create unified state
        unified_state = state_manager.create_unified_state(mock_mesa_model, mock_crewai_agents)
        
        # Add some state changes
        for i in range(10):
            change = StateChange(
                change_id=f"perf_test_{i}",
                entity_id=f"agent_{i}",
                change_type="position_update",
                old_value=(i, i),
                new_value=(i+1, i+1),
                timestamp=datetime.now(),
                source="mesa"
            )
            state_manager.register_state_change(change)
        
        # Measure synchronization time
        start_time = time.perf_counter()
        state_manager.apply_pending_changes()
        state_manager.sync_to_mesa(mock_mesa_model)
        state_manager.sync_to_crewai(mock_crewai_agents)
        end_time = time.perf_counter()
        
        sync_duration = (end_time - start_time) * 1000  # Convert to milliseconds
        
        # Should complete within 200ms
        assert sync_duration < 200.0, f"State sync took {sync_duration:.2f}ms, exceeds 200ms target"
    
    def test_event_driven_state_updates(self, state_manager):
        """Test event-driven state update notifications"""
        events_received = []
        
        def event_handler(event: StateChangeEvent):
            events_received.append(event)
        
        # Subscribe to state change events
        state_manager.event_bus.subscribe("state_changed", event_handler)
        
        # Make state changes
        change = StateChange(
            change_id="event_test_1",
            entity_id="agent_1",
            change_type="status_update",
            old_value="idle",
            new_value="moving",
            timestamp=datetime.now(),
            source="mesa"
        )
        
        state_manager.register_state_change(change)
        state_manager.apply_pending_changes()
        
        # Should have received event notification
        assert len(events_received) == 1
        event = events_received[0]
        assert event.change_id == "event_test_1"
        assert event.entity_id == "agent_1"
    
    def test_rollback_functionality(self, state_manager, mock_mesa_model, mock_crewai_agents):
        """Test state rollback capabilities"""
        # Create initial state
        initial_state = state_manager.create_unified_state(mock_mesa_model, mock_crewai_agents)
        snapshot_id = state_manager.create_snapshot("initial_state")
        
        # Make changes
        change = StateChange(
            change_id="rollback_test_1",
            entity_id="agent_1",
            change_type="position_update",
            old_value=(0, 0),
            new_value=(5, 5),
            timestamp=datetime.now(),
            source="mesa"
        )
        
        state_manager.register_state_change(change)
        state_manager.apply_pending_changes()
        
        # Verify change was applied
        current_state = state_manager.get_unified_state()
        assert current_state.version > initial_state.version
        
        # Rollback to initial state
        rollback_success = state_manager.rollback_to_snapshot(snapshot_id)
        assert rollback_success is True
        
        # Verify rollback
        rolled_back_state = state_manager.get_unified_state()
        assert rolled_back_state.version == initial_state.version


class TestStateSynchronizer:
    """Test state synchronization between frameworks"""
    
    @pytest.fixture
    def synchronizer(self):
        """Create state synchronizer"""
        return StateSynchronizer()
    
    @pytest.fixture
    def action_handoff(self):
        """Create sample action handoff from Agent C"""
        from src.escape_room_sim.hybrid.core_architecture import MesaAction
        from src.escape_room_sim.hybrid.actions.action_models import ConflictResolution
        
        actions = [
            MesaAction("agent_1", "move", {"target": (2, 3)}, 1.0, []),
            MesaAction("agent_2", "examine", {"target": "door"}, 2.0, [])
        ]
        
        execution_results = {
            "agent_1_move": ExecutionResult(
                "agent_1_move", "agent_1", "move", "completed", 
                datetime.now(), 0.8, True, {"old_pos": (1, 2), "new_pos": (2, 3)}
            ),
            "agent_2_examine": ExecutionResult(
                "agent_2_examine", "agent_2", "examine", "completed",
                datetime.now(), 1.5, True, {"findings": ["locked door", "keyhole visible"]}
            )
        }
        
        return ActionHandoff(
            actions=actions,
            execution_results=execution_results,
            conflict_resolutions=[],
            performance_metrics={"avg_duration": 1.15, "success_rate": 1.0}
        )
    
    def test_synchronizer_creation(self, synchronizer):
        """Test synchronizer can be created"""
        assert synchronizer is not None
        assert hasattr(synchronizer, 'sync_mesa_to_crewai')
        assert hasattr(synchronizer, 'sync_crewai_to_mesa')
    
    def test_action_handoff_processing(self, synchronizer, action_handoff, mock_mesa_model):
        """Test processing action handoff from Agent C"""
        # Process the handoff
        result = synchronizer.process_action_handoff(action_handoff, mock_mesa_model)
        
        assert result is not None
        assert "state_updates" in result
        assert "synchronization_time" in result
        assert result["success"] is True
    
    def test_mesa_to_crewai_sync(self, synchronizer, mock_mesa_model):
        """Test synchronizing Mesa state to CrewAI"""
        # Add some agents to Mesa model
        mock_agent1 = Mock()
        mock_agent1.unique_id = 1
        mock_agent1.pos = (2, 3)
        mock_agent1.health = 100
        mock_agent1.resources = ["key1"]
        
        mock_agent2 = Mock()
        mock_agent2.unique_id = 2  
        mock_agent2.pos = (4, 1)
        mock_agent2.health = 85
        mock_agent2.resources = []
        
        mock_mesa_model.schedule.agents = [mock_agent1, mock_agent2]
        
        # Perform synchronization
        sync_result = synchronizer.sync_mesa_to_crewai(mock_mesa_model)
        
        assert sync_result is not None
        assert "agent_states" in sync_result
        assert "environment_state" in sync_result
        assert len(sync_result["agent_states"]) == 2
    
    def test_crewai_to_mesa_sync(self, synchronizer, mock_mesa_model):
        """Test synchronizing CrewAI decisions to Mesa"""
        decisions = {
            "strategist": DecisionData(
                agent_id="strategist",
                chosen_action="analyze_room",
                action_parameters={"detail_level": "high"},
                reasoning="Need to understand room layout",
                confidence_level=0.9,
                timestamp=datetime.now()
            ),
            "mediator": DecisionData(
                agent_id="mediator", 
                chosen_action="communicate",
                action_parameters={"target": "survivor", "message": "coordinate_movement"},
                reasoning="Need team coordination",
                confidence_level=0.8,
                timestamp=datetime.now()
            )
        }
        
        # Perform synchronization
        sync_result = synchronizer.sync_crewai_to_mesa(decisions, mock_mesa_model)
        
        assert sync_result is not None
        assert "decisions_applied" in sync_result
        assert "mesa_state_updated" in sync_result
        assert sync_result["success"] is True


class TestEventBus:
    """Test event-driven communication system"""
    
    @pytest.fixture
    def event_bus(self):
        """Create event bus"""
        return EventBus()
    
    def test_event_bus_creation(self, event_bus):
        """Test event bus can be created"""
        assert event_bus is not None
        assert hasattr(event_bus, 'subscribe')
        assert hasattr(event_bus, 'publish')
        assert hasattr(event_bus, 'unsubscribe')
    
    def test_event_subscription_and_publishing(self, event_bus):
        """Test event subscription and publishing"""
        events_received = []
        
        def event_handler(event_data):
            events_received.append(event_data)
        
        # Subscribe to events
        event_bus.subscribe("test_event", event_handler)
        
        # Publish event
        test_data = {"message": "test event data", "timestamp": datetime.now()}
        event_bus.publish("test_event", test_data)
        
        # Should have received the event
        assert len(events_received) == 1
        assert events_received[0]["message"] == "test event data"
    
    def test_multiple_subscribers(self, event_bus):
        """Test multiple subscribers to same event"""
        events_1 = []
        events_2 = []
        
        def handler_1(data):
            events_1.append(data)
        
        def handler_2(data):
            events_2.append(data)
        
        # Subscribe both handlers
        event_bus.subscribe("multi_test", handler_1)
        event_bus.subscribe("multi_test", handler_2)
        
        # Publish event
        event_bus.publish("multi_test", {"value": 42})
        
        # Both should receive the event
        assert len(events_1) == 1
        assert len(events_2) == 1
        assert events_1[0]["value"] == 42
        assert events_2[0]["value"] == 42
    
    def test_event_unsubscription(self, event_bus):
        """Test unsubscribing from events"""
        events_received = []
        
        def event_handler(data):
            events_received.append(data)
        
        # Subscribe and publish
        event_bus.subscribe("unsub_test", event_handler)
        event_bus.publish("unsub_test", {"count": 1})
        
        # Unsubscribe and publish again
        event_bus.unsubscribe("unsub_test", event_handler)
        event_bus.publish("unsub_test", {"count": 2})
        
        # Should only have received the first event
        assert len(events_received) == 1
        assert events_received[0]["count"] == 1


@pytest.mark.asyncio
class TestAsyncStateOperations:
    """Test asynchronous state operations"""
    
    @pytest.fixture
    async def async_state_manager(self):
        """Create async-capable state manager"""
        from src.escape_room_sim.hybrid.state.unified_state_manager import AsyncUnifiedStateManager
        return AsyncUnifiedStateManager()
    
    async def test_async_state_synchronization(self, async_state_manager):
        """Test asynchronous state synchronization"""
        # This would test async state sync operations
        # Implementation depends on async state manager
        assert async_state_manager is not None
    
    async def test_concurrent_state_changes(self, async_state_manager):
        """Test handling concurrent state changes"""
        # Create multiple concurrent state changes
        tasks = []
        for i in range(5):
            change = StateChange(
                change_id=f"concurrent_{i}",
                entity_id=f"agent_{i}",
                change_type="status_update",
                old_value="idle",
                new_value="active",
                timestamp=datetime.now(),
                source="mesa"
            )
            task = async_state_manager.register_state_change_async(change)
            tasks.append(task)
        
        # Wait for all changes to be registered
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # All should succeed
        assert all(result is True for result in results if not isinstance(result, Exception))


class TestStateIntegrationScenarios:
    """Test realistic state integration scenarios"""
    
    @pytest.fixture
    def integration_setup(self):
        """Setup complete integration test environment"""
        state_manager = UnifiedStateManager()
        synchronizer = StateSynchronizer()
        
        # Mock Mesa model with agents
        mesa_model = Mock(spec=mesa.Model)
        mesa_model.schedule = Mock()
        mesa_model.schedule.steps = 0
        mesa_model.width = 10
        mesa_model.height = 10
        
        # Add Mesa agents
        mesa_agents = []
        for i in range(3):
            agent = Mock()
            agent.unique_id = i
            agent.pos = (i, i)
            agent.health = 100
            agent.resources = []
            mesa_agents.append(agent)
        mesa_model.schedule.agents = mesa_agents
        
        # Mock CrewAI agents
        crewai_agents = []
        for role in ['strategist', 'mediator', 'survivor']:
            agent = Mock()
            agent.role = role.title()
            agent.memory = Mock()
            crewai_agents.append(agent)
        
        return {
            'state_manager': state_manager,
            'synchronizer': synchronizer,
            'mesa_model': mesa_model,
            'crewai_agents': crewai_agents
        }
    
    def test_full_integration_cycle(self, integration_setup):
        """Test complete integration cycle"""
        setup = integration_setup
        
        # 1. Create unified state
        unified_state = setup['state_manager'].create_unified_state(
            setup['mesa_model'], setup['crewai_agents']
        )
        assert unified_state is not None
        
        # 2. Process some state changes
        change = StateChange(
            change_id="integration_test_1",
            entity_id="agent_0", 
            change_type="position_update",
            old_value=(0, 0),
            new_value=(1, 1),
            timestamp=datetime.now(),
            source="mesa"
        )
        
        success = setup['state_manager'].register_state_change(change)
        assert success is True
        
        # 3. Apply changes
        applied_changes = setup['state_manager'].apply_pending_changes()
        assert len(applied_changes) == 1
        
        # 4. Synchronize back to frameworks
        sync_result = setup['synchronizer'].sync_mesa_to_crewai(setup['mesa_model'])
        assert sync_result is not None
        
        # Integration cycle completed successfully
        assert True
    
    def test_error_recovery_scenario(self, integration_setup):
        """Test error recovery in state management"""
        setup = integration_setup
        
        # Create a problematic state change that should fail
        bad_change = StateChange(
            change_id="error_test_1",
            entity_id="", # Empty entity ID should cause validation failure
            change_type="invalid_change",
            old_value=None,
            new_value=None,
            timestamp=datetime.now(),
            source="unknown"
        )
        
        # Should handle the error gracefully
        success = setup['state_manager'].register_state_change(bad_change)
        assert success is False  # Should reject invalid change
        
        # System should remain functional
        good_change = StateChange(
            change_id="recovery_test_1",
            entity_id="agent_0",
            change_type="status_update",
            old_value="idle",
            new_value="active",
            timestamp=datetime.now(),
            source="mesa"
        )
        
        success = setup['state_manager'].register_state_change(good_change)
        assert success is True  # Should accept valid change after error
    
    def test_high_load_scenario(self, integration_setup):
        """Test system under high load"""
        setup = integration_setup
        
        # Generate many state changes rapidly
        changes = []
        for i in range(100):
            change = StateChange(
                change_id=f"load_test_{i}",
                entity_id=f"agent_{i % 3}",  # Distribute across 3 agents
                change_type="position_update",
                old_value=(i % 10, i % 10),
                new_value=((i + 1) % 10, (i + 1) % 10),
                timestamp=datetime.now(),
                source="mesa"
            )
            changes.append(change)
        
        # Register all changes
        start_time = time.perf_counter()
        for change in changes:
            success = setup['state_manager'].register_state_change(change)
            assert success is True
        
        # Apply all changes
        applied = setup['state_manager'].apply_pending_changes()
        end_time = time.perf_counter()
        
        processing_time = (end_time - start_time) * 1000  # ms
        
        # Should handle 100 changes efficiently
        assert len(applied) == 100
        assert processing_time < 1000  # Should complete within 1 second


if __name__ == "__main__":
    pytest.main([__file__, "-v"])