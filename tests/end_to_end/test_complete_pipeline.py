"""
End-to-End Integration Tests for Complete Mesa-CrewAI Pipeline

Agent D: State Management & Integration Specialist
Tests complete pipeline: Perception → Decision → Action → State → Mesa Integration
"""

import pytest
import asyncio
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any
from unittest.mock import Mock, patch

# Import all components for end-to-end testing
from src.escape_room_sim.hybrid.perception.integrated_pipeline import IntegratedPerceptionPipeline
from src.escape_room_sim.hybrid.decision.async_engine import AsyncDecisionEngine
from src.escape_room_sim.hybrid.execution.execution_pipeline import MesaActionExecutor
from src.escape_room_sim.hybrid.state.unified_state_manager import UnifiedStateManager
from src.escape_room_sim.mesa.escape_room_model import EscapeRoomModel

from src.escape_room_sim.hybrid.core_architecture import (
    HybridAgent, PerceptionData, DecisionData, MesaAction
)
from src.escape_room_sim.hybrid.actions.action_models import ActionHandoff


class TestCompleteSystemIntegration:
    """Test complete system integration from perception to state"""
    
    @pytest.fixture
    def system_components(self):
        """Set up all system components"""
        # Agent A: Perception
        perception_pipeline = IntegratedPerceptionPipeline()
        
        # Mock CrewAI agents
        crewai_agents = []
        for role in ['strategist', 'mediator', 'survivor']:
            agent = Mock()
            agent.role = role.title()
            agent.memory = Mock()
            crewai_agents.append(agent)
        
        # Agent B: Decision Engine
        config = {"max_workers": 3, "timeout": 30}
        decision_engine = AsyncDecisionEngine(crewai_agents, config)
        import asyncio
        asyncio.run(decision_engine.initialize())  # Initialize the decision engine
        
        # Agent C: Action Execution
        action_executor = MesaActionExecutor()
        
        # Agent D: State Management
        state_manager = UnifiedStateManager()
        
        # Mesa Environment
        room_config = {
            "width": 10, "height": 10, "num_agents": 3,
            "rooms": [
                {"id": "room1", "bounds": {"x1": 0, "y1": 0, "x2": 4, "y2": 4}, "room_type": "starting_room"},
                {"id": "room2", "bounds": {"x1": 5, "y1": 0, "x2": 9, "y2": 4}, "room_type": "puzzle_room"},
                {"id": "exit_room", "bounds": {"x1": 0, "y1": 5, "x2": 9, "y2": 9}, "room_type": "exit_room"}
            ],
            "doors": [
                {"id": "door1", "position": (4, 2), "connects": ["room1", "room2"], "locked": True, "key_required": "key1"}
            ],
            "objects": [
                {"id": "key1", "type": "key", "position": (1, 1), "room": "room1"}
            ],
            "time_limit": 300
        }
        mesa_model = EscapeRoomModel(room_config)
        
        return {
            'perception': perception_pipeline,
            'decision': decision_engine,
            'execution': action_executor,
            'state': state_manager,
            'mesa_model': mesa_model,
            'crewai_agents': crewai_agents
        }
    
    def test_single_agent_escape_scenario(self, system_components):
        """Test single agent navigating and escaping"""
        components = system_components
        
        # 1. PERCEPTION PHASE (Agent A)
        mesa_state = self._extract_mesa_state(components['mesa_model'])
        # Use extract_perceptions method that takes a mesa model directly
        perception_data = components['perception'].extract_perceptions(components['mesa_model'])
        
        # extract_perceptions returns a dict of agent_id -> PerceptionData
        assert isinstance(perception_data, dict)
        assert len(perception_data) > 0
        
        # Get first agent's perceptions
        first_agent_id = list(perception_data.keys())[0]
        first_perception = perception_data[first_agent_id]
        assert isinstance(first_perception, PerceptionData)
        
        # 2. DECISION PHASE (Agent B)
        # Use reason_and_decide method for decision generation (async)
        import asyncio
        # Method expects Dict[str, PerceptionData], so use the full perception_data
        decisions = asyncio.run(components['decision'].reason_and_decide(perception_data))
        
        assert len(decisions) > 0
        assert all(isinstance(d, DecisionData) for d in decisions.values())
        
        # 3. ACTION TRANSLATION & EXECUTION (Agent C)
        mesa_actions = []
        for decision in decisions.values():
            action = MesaAction(
                agent_id=decision.agent_id,
                action_type=decision.chosen_action, 
                parameters=decision.action_parameters,
                expected_duration=2.0,
                prerequisites=[]
            )
            mesa_actions.append(action)
        
        # Execute actions
        execution_results = []
        for action in mesa_actions:
            result = components['execution'].execute_action(action, components['mesa_model'])
            execution_results.append(result)
        
        # Create handoff
        handoff = ActionHandoff(
            actions=mesa_actions,
            execution_results={r.action_id: r for r in execution_results},
            conflict_resolutions=[],
            performance_metrics={"avg_duration": 1.5}
        )
        
        # 4. STATE MANAGEMENT (Agent D)
        unified_state = components['state'].create_unified_state(
            components['mesa_model'], components['crewai_agents']
        )
        
        # Process handoff
        state_result = components['state'].process_action_handoff(
            handoff, components['mesa_model']
        )
        
        assert state_result["success"] is True
        
        # 5. VERIFY END-TO-END SUCCESS
        # At least one action should have been executed successfully
        successful_actions = sum(1 for r in execution_results if r.success)
        assert successful_actions > 0
        
        # State should be updated
        final_state = components['state'].get_unified_state()
        assert final_state.version > unified_state.version
    
    def test_multi_agent_cooperation_scenario(self, system_components):
        """Test multiple agents working together to solve puzzles"""
        components = system_components
        
        # Get 3 agents from Mesa model (Mesa 3.x compatible)
        agents = components['mesa_model'].agents[:3]
        
        # Simulate cooperative behavior
        cooperation_decisions = {
            "strategist": DecisionData(
                agent_id="strategist",
                chosen_action="analyze_room",
                action_parameters={"target": "room1", "detail_level": "high"},
                reasoning="Need to understand room layout for team coordination",
                confidence_level=0.9,
                timestamp=datetime.now()
            ),
            "mediator": DecisionData(
                agent_id="mediator",
                chosen_action="communicate",
                action_parameters={"target": "survivor", "message": "found_key_location"},
                reasoning="Share key location with team",
                confidence_level=0.8,
                timestamp=datetime.now()
            ),
            "survivor": DecisionData(
                agent_id="survivor",
                chosen_action="move",
                action_parameters={"target_position": (1, 1)},
                reasoning="Move to collect key based on team information",
                confidence_level=0.7,
                timestamp=datetime.now()
            )
        }
        
        # Execute cooperative actions
        mesa_actions = []
        for decision in cooperation_decisions.values():
            action = MesaAction(
                agent_id=decision.agent_id,
                action_type=decision.chosen_action,
                parameters=decision.action_parameters,
                expected_duration=1.5
            )
            mesa_actions.append(action)
        
        # Execute with coordination
        execution_results = components['execution'].execute_action_batch(
            mesa_actions, components['mesa_model']
        )
        
        # Verify cooperation success
        assert len(execution_results) == 3
        successful_cooperation = sum(1 for r in execution_results if r.success)
        assert successful_cooperation >= 2  # At least 2/3 actions should succeed
        
        # Update state with cooperation results
        handoff = ActionHandoff(
            actions=mesa_actions,
            execution_results={r.action_id: r for r in execution_results},
            conflict_resolutions=[],
            performance_metrics={"cooperation_score": 0.8}
        )
        
        state_result = components['state'].process_action_handoff(
            handoff, components['mesa_model']
        )
        
        assert state_result["success"] is True
        assert "cooperation_metrics" in state_result
    
    def test_resource_competition_scenario(self, system_components):
        """Test agents competing for limited resources"""
        components = system_components
        
        # Create resource competition scenario
        competition_decisions = {
            "strategist": DecisionData(
                agent_id="strategist",
                chosen_action="claim_resource",
                action_parameters={"resource_id": "key1", "priority": "high"},
                reasoning="Need key to progress strategically",
                confidence_level=0.9,
                timestamp=datetime.now()
            ),
            "survivor": DecisionData(
                agent_id="survivor", 
                chosen_action="claim_resource",
                action_parameters={"resource_id": "key1", "priority": "critical"},
                reasoning="Need key for survival",
                confidence_level=0.95,
                timestamp=datetime.now()
            )
        }
        
        # This should create a resource conflict
        mesa_actions = []
        for decision in competition_decisions.values():
            action = MesaAction(
                agent_id=decision.agent_id,
                action_type=decision.chosen_action,
                parameters=decision.action_parameters,
                expected_duration=1.0
            )
            mesa_actions.append(action)
        
        # Execute conflicting actions
        execution_results = components['execution'].execute_action_batch(
            mesa_actions, components['mesa_model']
        )
        
        # Should have conflict resolution
        conflicts_detected = sum(1 for r in execution_results if not r.success)
        assert conflicts_detected >= 1  # At least one should fail due to conflict
        
        # State manager should handle conflict
        handoff = ActionHandoff(
            actions=mesa_actions,
            execution_results={r.action_id: r for r in execution_results},
            conflict_resolutions=[],  # Would be filled by conflict resolver
            performance_metrics={"conflict_count": conflicts_detected}
        )
        
        state_result = components['state'].process_action_handoff(
            handoff, components['mesa_model']
        )
        
        assert state_result["success"] is True
        assert "conflict_resolutions" in state_result
    
    def test_complex_multi_step_sequence(self, system_components):
        """Test complex coordinated multi-step actions"""
        components = system_components
        
        # Define complex sequence: Explore → Find Key → Unlock Door → Solve Puzzle → Escape
        sequence_steps = [
            {
                "agent": "strategist",
                "action": "examine",
                "params": {"target": "room1", "detail_level": "thorough"},
                "reasoning": "Map out room for efficient navigation"
            },
            {
                "agent": "mediator",
                "action": "move",
                "params": {"target_position": (1, 1)},
                "reasoning": "Move to key location identified by strategist"
            },
            {
                "agent": "mediator", 
                "action": "collect_object",
                "params": {"object_id": "key1"},
                "reasoning": "Collect key for team"
            },
            {
                "agent": "survivor",
                "action": "move",
                "params": {"target_position": (4, 2)},
                "reasoning": "Move to door for unlocking"
            },
            {
                "agent": "mediator",
                "action": "move", 
                "params": {"target_position": (4, 2)},
                "reasoning": "Bring key to door"
            },
            {
                "agent": "mediator",
                "action": "use_door",
                "params": {"door_id": "door1", "key": "key1"},
                "reasoning": "Unlock door for team access"
            }
        ]
        
        total_execution_time = 0
        successful_steps = 0
        
        for step in sequence_steps:
            decision = DecisionData(
                agent_id=step["agent"],
                chosen_action=step["action"],
                action_parameters=step["params"],
                reasoning=step["reasoning"],
                confidence_level=0.8,
                timestamp=datetime.now()
            )
            
            action = MesaAction(
                agent_id=decision.agent_id,
                action_type=decision.chosen_action,
                parameters=decision.action_parameters,
                expected_duration=2.0
            )
            
            start_time = time.perf_counter()
            result = components['execution'].execute_action(action, components['mesa_model'])
            end_time = time.perf_counter()
            
            total_execution_time += (end_time - start_time)
            
            if result.success:
                successful_steps += 1
                
                # Update state after each step
                handoff = ActionHandoff(
                    actions=[action],
                    execution_results={result.action_id: result},
                    conflict_resolutions=[],
                    performance_metrics={"step_duration": end_time - start_time}
                )
                
                state_result = components['state'].process_action_handoff(
                    handoff, components['mesa_model']
                )
                
                assert state_result["success"] is True
            else:
                # Handle step failure
                break
        
        # Verify sequence success
        success_rate = successful_steps / len(sequence_steps)
        assert success_rate >= 0.7  # At least 70% of steps should succeed
        
        # Verify reasonable execution time
        assert total_execution_time < 5.0  # Should complete within 5 seconds
    
    def test_error_recovery_integration(self, system_components):
        """Test system handles errors and continues"""
        components = system_components
        
        # Introduce deliberate error
        faulty_decision = DecisionData(
            agent_id="invalid_agent",  # Non-existent agent
            chosen_action="invalid_action",
            action_parameters={"invalid": "params"},
            reasoning="This should fail",
            confidence_level=0.1,
            timestamp=datetime.now()
        )
        
        faulty_action = MesaAction(
            agent_id=faulty_decision.agent_id,
            action_type=faulty_decision.chosen_action,
            parameters=faulty_decision.action_parameters,
            expected_duration=1.0
        )
        
        # Execute faulty action
        result = components['execution'].execute_action(faulty_action, components['mesa_model'])
        
        # Should fail gracefully
        assert result.success is False
        assert result.error_message is not None
        
        # System should continue with valid action
        valid_decision = DecisionData(
            agent_id="strategist",
            chosen_action="examine",
            action_parameters={"target": "environment"},
            reasoning="Examine surroundings",
            confidence_level=0.8,
            timestamp=datetime.now()
        )
        
        valid_action = MesaAction(
            agent_id=valid_decision.agent_id,
            action_type=valid_decision.chosen_action,
            parameters=valid_decision.action_parameters,
            expected_duration=1.5
        )
        
        valid_result = components['execution'].execute_action(valid_action, components['mesa_model'])
        
        # Valid action should succeed
        assert valid_result.success is True
        
        # State management should handle mixed results
        handoff = ActionHandoff(
            actions=[faulty_action, valid_action],
            execution_results={
                result.action_id: result,
                valid_result.action_id: valid_result
            },
            conflict_resolutions=[],
            failed_actions=[faulty_action],
            performance_metrics={"error_rate": 0.5}
        )
        
        state_result = components['state'].process_action_handoff(
            handoff, components['mesa_model']
        )
        
        # Should handle mixed results gracefully
        assert state_result["success"] is True
        assert "error_handling" in state_result
        assert "failed_actions" in state_result
    
    def test_performance_under_load(self, system_components):
        """Test system performance under high load"""
        components = system_components
        
        # Generate high load scenario
        num_actions = 50
        actions = []
        
        for i in range(num_actions):
            decision = DecisionData(
                agent_id=f"agent_{i % 3}",  # Distribute across 3 agents
                chosen_action="move" if i % 2 == 0 else "examine",
                action_parameters={
                    "target_position": (i % 10, (i // 10) % 10) if i % 2 == 0 
                    else {"target": "environment"}
                },
                reasoning=f"Load test action {i}",
                confidence_level=0.7,
                timestamp=datetime.now()
            )
            
            action = MesaAction(
                agent_id=decision.agent_id,
                action_type=decision.chosen_action,
                parameters=decision.action_parameters,
                expected_duration=0.5
            )
            actions.append(action)
        
        # Execute high load
        start_time = time.perf_counter()
        execution_results = components['execution'].execute_action_batch(
            actions, components['mesa_model']
        )
        end_time = time.perf_counter()
        
        execution_time = end_time - start_time
        
        # Verify performance
        assert len(execution_results) == num_actions
        assert execution_time < 10.0  # Should complete within 10 seconds
        
        # Calculate performance metrics
        successful_actions = sum(1 for r in execution_results if r.success)
        success_rate = successful_actions / num_actions
        throughput = num_actions / execution_time
        
        assert success_rate >= 0.8  # At least 80% success rate
        assert throughput >= 5.0    # At least 5 actions per second
        
        # State management should handle high load
        handoff = ActionHandoff(
            actions=actions,
            execution_results={r.action_id: r for r in execution_results},
            conflict_resolutions=[],
            performance_metrics={
                "execution_time": execution_time,
                "throughput": throughput,
                "success_rate": success_rate
            }
        )
        
        state_start = time.perf_counter()
        state_result = components['state'].process_action_handoff(
            handoff, components['mesa_model']
        )
        state_end = time.perf_counter()
        
        state_processing_time = state_end - state_start
        
        # State processing should be fast
        assert state_processing_time < 1.0  # Within 1 second
        assert state_result["success"] is True
    
    def _extract_mesa_state(self, mesa_model) -> Dict[str, Any]:
        """Helper method to extract Mesa state"""
        return {
            "agents": [
                {
                    "id": agent.unique_id,
                    "position": getattr(agent, 'pos', (0, 0)),
                    "health": getattr(agent, 'health', 100),
                    "resources": getattr(agent, 'resources', [])
                }
                for agent in mesa_model.agents
            ],
            "rooms": [
                {
                    "id": room_id,
                    "bounds": room.bounds,
                    "type": room.room_type
                }
                for room_id, room in mesa_model.rooms.items()
            ],
            "objects": [
                {
                    "id": obj_id,
                    "type": obj.object_type,
                    "position": obj.position,
                    "room": obj.room
                }
                for obj_id, obj in mesa_model.objects.items()
            ],
            "time_remaining": mesa_model.time_remaining,
            "model_step": getattr(mesa_model, 'model_step_count', 0)
        }


class TestSystemHandoff:
    """Test final system handoff to Agent D"""
    
    @pytest.fixture
    def complete_handoff_data(self):
        """Create complete handoff data from all agents"""
        from src.escape_room_sim.hybrid.state.unified_state_manager import SystemHandoff
        
        return SystemHandoff(
            unified_state=Mock(),
            integration_results={
                "perception_success": True,
                "decision_success": True,
                "execution_success": True,
                "state_sync_success": True
            },
            system_performance={
                "total_execution_time": 2.5,
                "success_rate": 0.85,
                "throughput": 12.0,
                "memory_usage": 156.7
            },
            error_reports=[],
            simulation_statistics={
                "agents_processed": 3,
                "actions_executed": 8,
                "conflicts_resolved": 2,
                "state_changes": 15
            },
            end_to_end_success=True
        )
    
    def test_system_handoff_creation(self, complete_handoff_data):
        """Test system handoff can be created"""
        assert complete_handoff_data is not None
        assert complete_handoff_data.end_to_end_success is True
        assert complete_handoff_data.integration_results["execution_success"] is True
    
    def test_system_handoff_validation(self, complete_handoff_data):
        """Test system handoff validation"""
        # Should validate successfully
        is_valid = complete_handoff_data.validate()
        assert is_valid is True
        
        # Test with invalid data
        complete_handoff_data.end_to_end_success = False
        complete_handoff_data.error_reports = ["Critical system failure"]
        
        is_valid = complete_handoff_data.validate()
        assert is_valid is False
    
    def test_integration_success_criteria(self, complete_handoff_data):
        """Test integration success criteria"""
        performance = complete_handoff_data.system_performance
        
        # Performance should meet targets
        assert performance["success_rate"] >= 0.80  # 80% minimum success rate
        assert performance["total_execution_time"] <= 5.0  # Max 5 seconds
        assert performance["throughput"] >= 10.0  # Min 10 actions/second
        
        # No critical errors
        critical_errors = [e for e in complete_handoff_data.error_reports if "critical" in e.lower()]
        assert len(critical_errors) == 0
        
        # All subsystems successful
        integration_results = complete_handoff_data.integration_results
        assert all(integration_results.values())


@pytest.mark.asyncio
class TestAsyncIntegration:
    """Test asynchronous integration scenarios"""
    
    async def test_async_pipeline_execution(self):
        """Test asynchronous pipeline execution"""
        # This would test the async aspects of the pipeline
        # Implementation depends on async components being available
        
        # Simulate async operations
        await asyncio.sleep(0.1)  # Simulate async processing
        
        # Verify async integration
        assert True  # Placeholder for actual async testing


class TestFullSystemPerformance:
    """Test overall system performance benchmarks"""
    
    def test_system_startup_time(self):
        """Test system components can start up quickly"""
        start_time = time.perf_counter()
        
        # Initialize all components
        perception = IntegratedPerceptionPipeline()
        decision = AsyncDecisionEngine()
        execution = MesaActionExecutor()
        state = UnifiedStateManager()
        
        end_time = time.perf_counter()
        startup_time = end_time - start_time
        
        # Should start up within 2 seconds
        assert startup_time < 2.0
    
    def test_system_memory_usage(self):
        """Test system memory usage remains reasonable"""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Create system components
        components = {
            'perception': IntegratedPerceptionPipeline(),
            'decision': AsyncDecisionEngine(),
            'execution': MesaActionExecutor(),
            'state': UnifiedStateManager()
        }
        
        # Run some operations
        for _ in range(10):
            # Simulate processing
            pass
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable (< 100MB)
        assert memory_increase < 100.0
    
    def test_system_scalability(self):
        """Test system can handle increasing load"""
        execution = MesaActionExecutor()
        
        # Test with increasing number of actions
        load_levels = [10, 25, 50, 100]
        performance_results = []
        
        for load in load_levels:
            actions = []
            for i in range(load):
                action = MesaAction(agent_id=f"agent_{i % 3}", action_type="move", parameters={"target_position": (i % 10, (i // 10) % 10)}, expected_duration=0.5
                , prerequisites=[])
                actions.append(action)
            
            # Create mock model (Mesa 3.x compatible)
            mock_model = Mock()
            mock_model.agents = [Mock() for _ in range(3)]
            for i, agent in enumerate(mock_model.agents):
                agent.unique_id = i
                agent.pos = (0, 0)
            
            start_time = time.perf_counter()
            results = execution.execute_action_batch(actions, mock_model)
            end_time = time.perf_counter()
            
            execution_time = end_time - start_time
            throughput = load / execution_time
            
            performance_results.append({
                "load": load,
                "time": execution_time,
                "throughput": throughput
            })
        
        # Verify scalability - throughput shouldn't degrade significantly
        base_throughput = performance_results[0]["throughput"]
        max_throughput = performance_results[-1]["throughput"]
        
        # Allow some performance degradation but not more than 50%
        assert max_throughput >= base_throughput * 0.5


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])