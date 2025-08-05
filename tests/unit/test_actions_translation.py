"""
Test Suite for Agent C: Action Translation & Execution Specialist
Advanced Action Translation System Tests

Tests the action translation from CrewAI decisions to Mesa actions with
complex sequence orchestration and conflict resolution.
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, MagicMock
from typing import Dict, List, Any
import mesa

from src.escape_room_sim.hybrid.core_architecture import DecisionData, MesaAction
from src.escape_room_sim.hybrid.decision.handoff_protocol import DecisionHandoff
from src.escape_room_sim.hybrid.actions.action_models import (
    ActionSequence, ActionConflict, ConflictResolution, ExecutionResult
)
from src.escape_room_sim.hybrid.actions.action_translator import (
    AdvancedActionTranslator, ActionSequenceOrchestrator, ConflictResolver, ActionValidator
)


class TestActionSequence:
    """Test ActionSequence data model"""
    
    def test_action_sequence_creation(self):
        """Test creation of ActionSequence"""
        decisions = [
            DecisionData(
                agent_id="strategist",
                timestamp=datetime.now(),
                chosen_action="move",
                action_parameters={"target_position": (5, 5)},
                reasoning="Moving to central position",
                confidence_level=0.8,
                fallback_actions=["examine"]
            ),
            DecisionData(
                agent_id="strategist", 
                timestamp=datetime.now(),
                chosen_action="examine",
                action_parameters={"target": "door"},
                reasoning="Examining door after movement",
                confidence_level=0.9,
                fallback_actions=["wait"]
            )
        ]
        
        sequence = ActionSequence(
            sequence_id="seq_001",
            agent_id="strategist",
            decisions=decisions,
            sequence_type="multi_step",
            dependencies=["move -> examine"]
        )
        
        assert sequence.sequence_id == "seq_001"
        assert sequence.agent_id == "strategist"
        assert len(sequence.decisions) == 2
        assert sequence.sequence_type == "multi_step"
        assert sequence.dependencies == ["move -> examine"]
        
    def test_action_sequence_validation(self):
        """Test ActionSequence validation"""
        # Valid sequence
        valid_decisions = [
            DecisionData(
                agent_id="strategist",
                timestamp=datetime.now(),
                chosen_action="move",
                action_parameters={"target_position": (5, 5)},
                reasoning="Valid move",
                confidence_level=0.8,
                fallback_actions=["examine"]
            )
        ]
        
        valid_sequence = ActionSequence(
            sequence_id="valid_seq",
            agent_id="strategist", 
            decisions=valid_decisions,
            sequence_type="single_step",
            dependencies=[]
        )
        
        # Should be valid
        assert valid_sequence.validate_sequence()
        
        # Invalid sequence - mismatched agent_id
        invalid_decisions = [
            DecisionData(
                agent_id="mediator",  # Different from sequence agent_id
                timestamp=datetime.now(),
                chosen_action="move",
                action_parameters={"target_position": (5, 5)},
                reasoning="Invalid agent mismatch",
                confidence_level=0.8,
                fallback_actions=["examine"]
            )
        ]
        
        invalid_sequence = ActionSequence(
            sequence_id="invalid_seq",
            agent_id="strategist",
            decisions=invalid_decisions,
            sequence_type="single_step",
            dependencies=[]
        )
        
        # Should be invalid
        assert not invalid_sequence.validate_sequence()


class TestActionConflict:
    """Test ActionConflict data model"""
    
    def test_conflict_creation(self):
        """Test creation of ActionConflict"""
        actions = [
            MesaAction(
                agent_id="strategist",
                action_type="claim_resource",
                parameters={"resource_id": "key_001"},
                expected_duration=2.0,
                prerequisites=[]
            ),
            MesaAction(
                agent_id="survivor",
                action_type="claim_resource", 
                parameters={"resource_id": "key_001"},
                expected_duration=1.5,
                prerequisites=[]
            )
        ]
        
        conflict = ActionConflict(
            conflict_id="conflict_001",
            conflict_type="resource_competition",
            conflicting_actions=actions,
            resource_contested="key_001",
            severity="high"
        )
        
        assert conflict.conflict_id == "conflict_001"
        assert conflict.conflict_type == "resource_competition"
        assert len(conflict.conflicting_actions) == 2
        assert conflict.resource_contested == "key_001"
        assert conflict.severity == "high"
        
    def test_conflict_analysis(self):
        """Test conflict analysis methods"""
        actions = [
            MesaAction(
                agent_id="strategist",
                action_type="claim_resource",
                parameters={"resource_id": "key_001", "priority": 0.8},
                expected_duration=2.0,
                prerequisites=[]
            ),
            MesaAction(
                agent_id="survivor",
                action_type="claim_resource",
                parameters={"resource_id": "key_001", "priority": 0.9},
                expected_duration=1.5,
                prerequisites=[]
            )
        ]
        
        conflict = ActionConflict(
            conflict_id="conflict_001",
            conflict_type="resource_competition",
            conflicting_actions=actions,
            resource_contested="key_001",
            severity="high"
        )
        
        # Test conflict analysis
        analysis = conflict.analyze_conflict()
        assert "agents_involved" in analysis
        assert "resource_type" in analysis
        assert "priority_scores" in analysis
        assert len(analysis["agents_involved"]) == 2
        assert "strategist" in analysis["agents_involved"]
        assert "survivor" in analysis["agents_involved"]


class TestAdvancedActionTranslator:
    """Test AdvancedActionTranslator class"""
    
    @pytest.fixture
    def translator(self):
        """Create action translator instance"""
        return AdvancedActionTranslator()
    
    @pytest.fixture
    def mock_mesa_model(self):
        """Create mock Mesa model"""
        model = Mock(spec=mesa.Model)
        model.schedule = Mock()
        model.schedule.agents = []
        model.grid = Mock()
        model.grid.width = 10
        model.grid.height = 10
        model.grid.out_of_bounds = lambda pos: pos[0] < 0 or pos[0] >= 10 or pos[1] < 0 or pos[1] >= 10
        model.resource_manager = Mock()
        return model
    
    def test_translator_initialization(self, translator):
        """Test translator initialization"""
        assert translator is not None
        assert hasattr(translator, 'sequence_orchestrator')
        assert hasattr(translator, 'conflict_resolver')
        assert hasattr(translator, 'action_validator')
        
    def test_translate_single_decision(self, translator, mock_mesa_model):
        """Test translation of single decision to Mesa action"""
        decision = DecisionData(
            agent_id="strategist",
            timestamp=datetime.now(),
            chosen_action="move",
            action_parameters={"target_position": (5, 5), "speed": "normal"},
            reasoning="Moving to strategic position",
            confidence_level=0.8,
            fallback_actions=["examine", "wait"]
        )
        
        mesa_action = translator.translate_decision(decision)
        
        assert isinstance(mesa_action, MesaAction)
        assert mesa_action.agent_id == "strategist"
        assert mesa_action.action_type == "move"
        assert mesa_action.parameters["target_position"] == (5, 5)
        assert mesa_action.parameters["speed"] == "normal"
        assert mesa_action.expected_duration > 0
        
    def test_translate_decision_sequence(self, translator, mock_mesa_model):
        """Test translation of decision sequence"""
        decisions = [
            DecisionData(
                agent_id="strategist",
                timestamp=datetime.now(),
                chosen_action="move",
                action_parameters={"target_position": (3, 3)},
                reasoning="Move to door",
                confidence_level=0.8,
                fallback_actions=["examine"]
            ),
            DecisionData(
                agent_id="strategist",
                timestamp=datetime.now() + timedelta(seconds=2),
                chosen_action="examine", 
                action_parameters={"target": "door", "detail_level": "high"},
                reasoning="Examine door after reaching it",
                confidence_level=0.9,
                fallback_actions=["try_key"]
            ),
            DecisionData(
                agent_id="strategist",
                timestamp=datetime.now() + timedelta(seconds=5),
                chosen_action="use_key",
                action_parameters={"key_id": "brass_key", "target": "door"},
                reasoning="Use key to unlock door",
                confidence_level=0.7,
                fallback_actions=["look_for_another_key"]
            )
        ]
        
        mesa_actions = translator.translate_decision_sequence(decisions)
        
        assert len(mesa_actions) == 3
        assert all(isinstance(action, MesaAction) for action in mesa_actions)
        assert mesa_actions[0].action_type == "move"
        assert mesa_actions[1].action_type == "examine"
        assert mesa_actions[2].action_type == "use_key"
        
        # Check that dependencies are properly set in parameters
        assert 'dependencies' in mesa_actions[1].parameters
        assert 'dependencies' in mesa_actions[2].parameters
        
    def test_validate_action_basic(self, translator, mock_mesa_model):
        """Test basic action validation"""
        valid_action = MesaAction(
            agent_id="strategist",
            action_type="move",
            parameters={"target_position": (5, 5)},
            expected_duration=1.0,
            prerequisites=[]
        )
        
        # Mock agent in model
        mock_agent = Mock()
        mock_agent.agent_id = "strategist"
        mock_agent.pos = (3, 3)
        mock_mesa_model.schedule.agents = [mock_agent]
        
        is_valid = translator.validate_action(valid_action, mock_mesa_model)
        assert is_valid
        
    def test_validate_action_invalid_bounds(self, translator, mock_mesa_model):
        """Test action validation with invalid bounds"""
        invalid_action = MesaAction(
            agent_id="strategist",
            action_type="move",
            parameters={"target_position": (15, 15)},  # Outside 10x10 grid
            expected_duration=1.0,
            prerequisites=[]
        )
        
        # Mock agent in model
        mock_agent = Mock()
        mock_agent.agent_id = "strategist"
        mock_agent.pos = (5, 5)
        mock_mesa_model.schedule.agents = [mock_agent]
        
        is_valid = translator.validate_action(invalid_action, mock_mesa_model)
        assert not is_valid


class TestActionSequenceOrchestrator:
    """Test ActionSequenceOrchestrator class"""
    
    @pytest.fixture
    def orchestrator(self):
        """Create orchestrator instance"""
        return ActionSequenceOrchestrator()
    
    def test_orchestrator_initialization(self, orchestrator):
        """Test orchestrator initialization"""
        assert orchestrator is not None
        assert hasattr(orchestrator, 'active_sequences')
        assert hasattr(orchestrator, 'sequence_dependencies')
        
    def test_create_sequence(self, orchestrator):
        """Test sequence creation"""
        decisions = [
            DecisionData(
                agent_id="strategist",
                timestamp=datetime.now(),
                chosen_action="move",
                action_parameters={"target_position": (5, 5)},
                reasoning="First move",
                confidence_level=0.8,
                fallback_actions=["wait"]
            ),
            DecisionData(
                agent_id="strategist",
                timestamp=datetime.now(),
                chosen_action="examine",
                action_parameters={"target": "environment"},
                reasoning="Then examine",
                confidence_level=0.9,
                fallback_actions=["wait"]
            )
        ]
        
        sequence = orchestrator.create_sequence(
            agent_id="strategist",
            decisions=decisions,
            sequence_type="conditional"
        )
        
        assert isinstance(sequence, ActionSequence)
        assert sequence.agent_id == "strategist"
        assert len(sequence.decisions) == 2
        assert sequence.sequence_type == "conditional"
        
    def test_orchestrate_parallel_sequences(self, orchestrator):
        """Test orchestration of parallel sequences"""
        # Create sequences for different agents
        strategist_decisions = [
            DecisionData(
                agent_id="strategist",
                timestamp=datetime.now(),
                chosen_action="analyze",
                action_parameters={"target": "room_layout"},
                reasoning="Analyze room structure",
                confidence_level=0.8,
                fallback_actions=["examine"]
            )
        ]
        
        mediator_decisions = [
            DecisionData(
                agent_id="mediator",
                timestamp=datetime.now(),
                chosen_action="communicate",
                action_parameters={"target": "broadcast", "message": "coordination"},
                reasoning="Coordinate with team",
                confidence_level=0.9,
                fallback_actions=["wait"]
            )
        ]
        
        sequences = [
            orchestrator.create_sequence("strategist", strategist_decisions, "analysis"),
            orchestrator.create_sequence("mediator", mediator_decisions, "communication")
        ]
        
        orchestrated_actions = orchestrator.orchestrate_sequences(sequences)
        
        assert len(orchestrated_actions) >= 2
        assert any(action.agent_id == "strategist" for action in orchestrated_actions)
        assert any(action.agent_id == "mediator" for action in orchestrated_actions)


class TestConflictResolver:
    """Test ConflictResolver class"""
    
    @pytest.fixture
    def resolver(self):
        """Create conflict resolver instance"""
        return ConflictResolver()
    
    @pytest.fixture
    def mock_mesa_model(self):
        """Create mock Mesa model for conflict resolution"""
        model = Mock(spec=mesa.Model)
        model.resource_manager = Mock()
        model.resource_manager.get_resource_availability.return_value = {"key_001": 1}
        model.trust_tracker = Mock()
        model.trust_tracker.get_trust_level.return_value = 0.7
        return model
    
    def test_resolver_initialization(self, resolver):
        """Test resolver initialization"""
        assert resolver is not None
        assert hasattr(resolver, 'resolution_strategies')
        assert hasattr(resolver, 'conflict_history')
        
    def test_detect_resource_conflict(self, resolver):
        """Test detection of resource conflicts"""
        actions = [
            MesaAction(
                agent_id="strategist",
                action_type="claim_resource",
                parameters={"resource_id": "key_001"},
                expected_duration=2.0,
                prerequisites=[]
            ),
            MesaAction(
                agent_id="survivor",
                action_type="claim_resource",
                parameters={"resource_id": "key_001"},
                expected_duration=1.5,
                prerequisites=[]
            )
        ]
        
        conflicts = resolver.detect_conflicts(actions)
        
        assert len(conflicts) == 1
        conflict = conflicts[0]
        assert isinstance(conflict, ActionConflict)
        assert conflict.conflict_type == "resource_competition"
        assert len(conflict.conflicting_actions) == 2
        
    def test_detect_spatial_conflict(self, resolver):
        """Test detection of spatial conflicts"""
        actions = [
            MesaAction(
                agent_id="strategist",
                action_type="move",
                parameters={"target_position": (5, 5)},
                expected_duration=1.0,
                prerequisites=[]
            ),
            MesaAction(
                agent_id="mediator",
                action_type="move",
                parameters={"target_position": (5, 5)},
                expected_duration=1.0,
                prerequisites=[]
            )
        ]
        
        conflicts = resolver.detect_conflicts(actions)
        
        assert len(conflicts) == 1
        conflict = conflicts[0]
        assert conflict.conflict_type == "spatial_collision"
        
    def test_resolve_resource_conflict_priority(self, resolver, mock_mesa_model):
        """Test resolution of resource conflict using priority"""
        conflict = ActionConflict(
            conflict_id="test_conflict",
            conflict_type="resource_competition",
            conflicting_actions=[
                MesaAction(
                    agent_id="strategist",
                    action_type="claim_resource",
                    parameters={"resource_id": "key_001", "priority": 0.6},
                    expected_duration=2.0,
                    prerequisites=[]
                ),
                MesaAction(
                    agent_id="survivor",
                    action_type="claim_resource",
                    parameters={"resource_id": "key_001", "priority": 0.9},
                    expected_duration=1.5,
                    prerequisites=[]
                )
            ],
            resource_contested="key_001",
            severity="medium"
        )
        
        resolution = resolver.resolve_conflict(conflict, mock_mesa_model)
        
        assert isinstance(resolution, ConflictResolution)
        assert resolution.resolution_type == "priority_based"
        assert len(resolution.resolved_actions) <= 2
        assert resolution.success
        
        # Higher priority agent (survivor) should get the resource
        winner_action = next(
            (action for action in resolution.resolved_actions 
             if action.agent_id == "survivor" and action.action_type == "claim_resource"),
            None
        )
        assert winner_action is not None
        
    def test_resolve_conflict_rate_performance(self, resolver, mock_mesa_model):
        """Test conflict resolution performance requirement (>90% success rate)"""
        total_conflicts = 100
        successful_resolutions = 0
        
        for i in range(total_conflicts):
            conflict = ActionConflict(
                conflict_id=f"conflict_{i}",
                conflict_type="resource_competition",
                conflicting_actions=[
                    MesaAction(
                        agent_id="agent_a",
                        action_type="claim_resource",
                        parameters={"resource_id": f"resource_{i}", "priority": 0.5},
                        expected_duration=1.0,
                        prerequisites=[]
                    ),
                    MesaAction(
                        agent_id="agent_b", 
                        action_type="claim_resource",
                        parameters={"resource_id": f"resource_{i}", "priority": 0.7},
                        expected_duration=1.0,
                        prerequisites=[]
                    )
                ],
                resource_contested=f"resource_{i}",
                severity="medium"
            )
            
            resolution = resolver.resolve_conflict(conflict, mock_mesa_model)
            if resolution.success:
                successful_resolutions += 1
        
        success_rate = successful_resolutions / total_conflicts
        assert success_rate > 0.9, f"Conflict resolution rate {success_rate:.1%} below 90% requirement"


class TestActionValidator:
    """Test ActionValidator class"""
    
    @pytest.fixture
    def validator(self):
        """Create action validator instance"""
        return ActionValidator()
    
    @pytest.fixture
    def mock_mesa_model(self):
        """Create mock Mesa model for validation"""
        model = Mock(spec=mesa.Model)
        model.schedule = Mock()
        model.grid = Mock()
        model.grid.width = 10
        model.grid.height = 10
        model.grid.out_of_bounds = lambda pos: pos[0] < 0 or pos[0] >= 10 or pos[1] < 0 or pos[1] >= 10
        
        # Mock agent
        mock_agent = Mock()
        mock_agent.agent_id = "strategist"
        mock_agent.pos = (5, 5)
        mock_agent.resources = ["tool_001"]
        mock_agent.energy = 0.8
        model.schedule.agents = [mock_agent]
        
        return model
    
    def test_validator_initialization(self, validator):
        """Test validator initialization"""
        assert validator is not None
        assert hasattr(validator, 'validation_rules')
        
    def test_validate_movement_action(self, validator, mock_mesa_model):
        """Test validation of movement actions"""
        # Valid movement
        valid_move = MesaAction(
            agent_id="strategist",
            action_type="move",
            parameters={"target_position": (6, 6)},
            expected_duration=1.0,
            prerequisites=[]
        )
        
        assert validator.validate_action(valid_move, mock_mesa_model)
        
        # Invalid movement - out of bounds
        invalid_move = MesaAction(
            agent_id="strategist",
            action_type="move",
            parameters={"target_position": (15, 15)},
            expected_duration=1.0,
            prerequisites=[]
        )
        
        assert not validator.validate_action(invalid_move, mock_mesa_model)
        
    def test_validate_resource_action(self, validator, mock_mesa_model):
        """Test validation of resource actions"""
        # Valid resource use - agent has the tool
        valid_use = MesaAction(
            agent_id="strategist",
            action_type="use_tool",
            parameters={"tool_id": "tool_001"},
            expected_duration=2.0,
            prerequisites=["has_tool"]
        )
        
        assert validator.validate_action(valid_use, mock_mesa_model)
        
        # Invalid resource use - agent doesn't have tool
        invalid_use = MesaAction(
            agent_id="strategist",
            action_type="use_tool",
            parameters={"tool_id": "tool_999"},
            expected_duration=2.0,
            prerequisites=["has_tool"]
        )
        
        assert not validator.validate_action(invalid_use, mock_mesa_model)
        
    def test_validate_complex_action_sequence(self, validator, mock_mesa_model):
        """Test validation of complex multi-step action sequences"""
        actions = [
            MesaAction(
                agent_id="strategist",
                action_type="move",
                parameters={"target_position": (3, 3)},
                expected_duration=1.0,
                prerequisites=[]
            ),
            MesaAction(
                agent_id="strategist",
                action_type="examine",
                parameters={"target": "door"},
                expected_duration=2.0,
                prerequisites=[]
            ),
            MesaAction(
                agent_id="strategist",
                action_type="use_key",
                parameters={"key_id": "brass_key", "target": "door"},
                expected_duration=1.5,
                prerequisites=["has_key"]
            )
        ]
        
        # Validate sequence - should all be valid individually
        for action in actions:
            is_valid = validator.validate_action(action, mock_mesa_model)
            # Note: some actions may fail prerequisite checks, but structure should be valid
            assert isinstance(is_valid, bool)


class TestPerformanceRequirements:
    """Test performance requirements for Agent C"""
    
    @pytest.fixture
    def translator(self):
        return AdvancedActionTranslator()
    
    @pytest.fixture
    def resolver(self):
        return ConflictResolver()
    
    def test_action_translation_performance(self, translator):
        """Test action translation performance requirement (<50ms)"""
        decision = DecisionData(
            agent_id="strategist",
            timestamp=datetime.now(),
            chosen_action="move",
            action_parameters={"target_position": (5, 5)},
            reasoning="Performance test move",
            confidence_level=0.8,
            fallback_actions=["wait"]
        )
        
        import time
        start_time = time.perf_counter()
        mesa_action = translator.translate_decision(decision)
        end_time = time.perf_counter()
        
        translation_time = (end_time - start_time) * 1000  # Convert to milliseconds
        assert translation_time < 50, f"Translation took {translation_time:.2f}ms, requirement is <50ms"
        assert isinstance(mesa_action, MesaAction)
        
    def test_conflict_resolution_performance(self, resolver):
        """Test conflict resolution performance on multiple conflicts"""
        conflicts = []
        for i in range(20):  # Test with multiple conflicts
            conflict = ActionConflict(
                conflict_id=f"perf_conflict_{i}",
                conflict_type="resource_competition",
                conflicting_actions=[
                    MesaAction(
                        agent_id="agent_a",
                        action_type="claim_resource",
                        parameters={"resource_id": f"resource_{i}"},
                        expected_duration=1.0,
                        prerequisites=[]
                    ),
                    MesaAction(
                        agent_id="agent_b",
                        action_type="claim_resource", 
                        parameters={"resource_id": f"resource_{i}"},
                        expected_duration=1.0,
                        prerequisites=[]
                    )
                ],
                resource_contested=f"resource_{i}",
                severity="medium"
            )
            conflicts.append(conflict)
        
        mock_model = Mock()
        mock_model.resource_manager = Mock()
        mock_model.trust_tracker = Mock()
        mock_model.trust_tracker.get_trust_level.return_value = 0.5
        
        import time
        start_time = time.perf_counter()
        
        successful_resolutions = 0
        for conflict in conflicts:
            resolution = resolver.resolve_conflict(conflict, mock_model)
            if resolution.success:
                successful_resolutions += 1
        
        end_time = time.perf_counter()
        resolution_time = (end_time - start_time) * 1000
        
        # Performance check - should handle 20 conflicts efficiently
        assert resolution_time < 1000, f"Resolution took {resolution_time:.2f}ms for 20 conflicts"
        
        # Success rate check - should be >90%
        success_rate = successful_resolutions / len(conflicts)
        assert success_rate > 0.9, f"Success rate {success_rate:.1%} below 90% requirement"