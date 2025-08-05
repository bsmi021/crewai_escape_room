"""
Integration Tests for Mesa Escape Room Environment

Agent D: State Management & Integration Specialist
Tests custom Mesa escape room model with rooms, objects, agents, and mechanics.
"""

import pytest
import mesa
import numpy as np
from typing import Dict, List, Any, Tuple
from unittest.mock import Mock, patch

from src.escape_room_sim.mesa.escape_room_model import (
    EscapeRoomModel, EscapeRoomAgent, Room, Door, Key, Puzzle, Tool
)
from src.escape_room_sim.mesa.room_objects import (
    RoomObject, InteractiveObject, ResourceObject
)
from src.escape_room_sim.mesa.resource_manager import MesaResourceManager
from src.escape_room_sim.mesa.environment_hazards import EnvironmentHazard, HazardManager


class TestEscapeRoomModel:
    """Test custom Mesa escape room model"""
    
    @pytest.fixture
    def basic_room_config(self):
        """Basic room configuration for testing"""
        return {
            "width": 10,
            "height": 10,
            "num_agents": 3,
            "rooms": [
                {
                    "id": "room1",
                    "bounds": {"x1": 0, "y1": 0, "x2": 4, "y2": 4},
                    "room_type": "starting_room"
                },
                {
                    "id": "room2", 
                    "bounds": {"x1": 5, "y1": 0, "x2": 9, "y2": 4},
                    "room_type": "puzzle_room"
                },
                {
                    "id": "exit_room",
                    "bounds": {"x1": 0, "y1": 5, "x2": 9, "y2": 9},
                    "room_type": "exit_room"
                }
            ],
            "doors": [
                {
                    "id": "door1",
                    "position": (4, 2),
                    "connects": ["room1", "room2"],
                    "locked": True,
                    "key_required": "key1"
                },
                {
                    "id": "door2",
                    "position": (2, 4),
                    "connects": ["room1", "exit_room"],
                    "locked": True,
                    "key_required": "key2"
                }
            ],
            "objects": [
                {
                    "id": "key1",
                    "type": "key",
                    "position": (1, 1),
                    "room": "room1"
                },
                {
                    "id": "puzzle1",
                    "type": "puzzle",
                    "position": (7, 2),
                    "room": "room2",
                    "solution": "sequence_123",
                    "reward": "key2"
                }
            ],
            "time_limit": 300,  # 5 minutes
            "escape_conditions": {
                "agents_required": 2,
                "exit_room": "exit_room"
            }
        }
    
    @pytest.fixture
    def escape_room_model(self, basic_room_config):
        """Create escape room model instance"""
        return EscapeRoomModel(basic_room_config)
    
    def test_escape_room_model_creation(self, escape_room_model):
        """Test escape room model can be created"""
        assert escape_room_model is not None
        assert isinstance(escape_room_model, mesa.Model)
        assert hasattr(escape_room_model, 'grid')
        assert hasattr(escape_room_model, 'schedule')
        assert hasattr(escape_room_model, 'running')
    
    def test_room_layout_initialization(self, escape_room_model):
        """Test room layout is properly initialized"""
        # Should have 3 rooms
        assert len(escape_room_model.rooms) == 3
        
        # Check room properties
        room1 = escape_room_model.get_room("room1")
        assert room1 is not None
        assert room1.room_type == "starting_room"
        assert room1.bounds == {"x1": 0, "y1": 0, "x2": 4, "y2": 4}
        
        exit_room = escape_room_model.get_room("exit_room")
        assert exit_room is not None
        assert exit_room.room_type == "exit_room"
    
    def test_door_system_initialization(self, escape_room_model):
        """Test door system is properly set up"""
        # Should have 2 doors
        assert len(escape_room_model.doors) == 2
        
        door1 = escape_room_model.get_door("door1")
        assert door1 is not None
        assert door1.locked is True
        assert door1.key_required == "key1"
        assert door1.position == (4, 2)
        assert "room1" in door1.connects
        assert "room2" in door1.connects
    
    def test_agent_initialization(self, escape_room_model):
        """Test agents are properly initialized"""
        # Should have 3 agents
        assert len(escape_room_model.schedule.agents) == 3
        
        for agent in escape_room_model.schedule.agents:
            assert isinstance(agent, EscapeRoomAgent)
            assert hasattr(agent, 'pos')
            assert hasattr(agent, 'health')
            assert hasattr(agent, 'resources')
            assert hasattr(agent, 'current_room')
            
            # Agents should start in starting room
            assert agent.current_room == "room1"
    
    def test_object_placement(self, escape_room_model):
        """Test room objects are properly placed"""
        # Should have key and puzzle objects
        key1 = escape_room_model.get_object("key1")
        assert key1 is not None
        assert isinstance(key1, Key)
        assert key1.position == (1, 1)
        assert key1.room == "room1"
        
        puzzle1 = escape_room_model.get_object("puzzle1")
        assert puzzle1 is not None
        assert isinstance(puzzle1, Puzzle)
        assert puzzle1.position == (7, 2)
        assert puzzle1.room == "room2"
        assert puzzle1.solution == "sequence_123"
    
    def test_agent_movement_within_room(self, escape_room_model):
        """Test agent can move within a room"""
        agent = escape_room_model.schedule.agents[0]
        initial_pos = agent.pos
        
        # Move agent within room1 bounds
        target_pos = (2, 2)
        success = escape_room_model.move_agent(agent, target_pos)
        
        assert success is True
        assert agent.pos == target_pos
        assert agent.current_room == "room1"  # Should still be in same room
    
    def test_agent_movement_blocked_by_walls(self, escape_room_model):
        """Test agent movement is blocked by room boundaries"""
        agent = escape_room_model.schedule.agents[0]
        
        # Try to move outside room bounds
        invalid_pos = (10, 10)  # Outside grid
        success = escape_room_model.move_agent(agent, invalid_pos)
        
        assert success is False
        
        # Try to move into another room without using door
        invalid_pos = (6, 2)  # Inside room2 but no door access
        success = escape_room_model.move_agent(agent, invalid_pos)
        
        assert success is False
    
    def test_door_interaction_locked(self, escape_room_model):
        """Test interaction with locked door"""
        agent = escape_room_model.schedule.agents[0]
        
        # Move agent to door
        door_pos = (4, 2)
        escape_room_model.move_agent(agent, (3, 2))  # Adjacent to door
        
        # Try to use door without key
        success = escape_room_model.use_door(agent, "door1")
        assert success is False
        assert agent.current_room == "room1"  # Should not have moved
    
    def test_key_collection_and_door_unlocking(self, escape_room_model):
        """Test collecting key and unlocking door"""
        agent = escape_room_model.schedule.agents[0]
        
        # Move agent to key position
        key_pos = (1, 1)
        escape_room_model.move_agent(agent, key_pos)
        
        # Collect key
        success = escape_room_model.collect_object(agent, "key1")
        assert success is True
        assert "key1" in agent.resources
        
        # Move to door and unlock it
        escape_room_model.move_agent(agent, (3, 2))  # Adjacent to door
        success = escape_room_model.use_door(agent, "door1")
        assert success is True
        assert agent.current_room == "room2"
    
    def test_puzzle_solving(self, escape_room_model):
        """Test puzzle solving mechanics"""
        agent = escape_room_model.schedule.agents[0]
        
        # Get agent to room2 (need key first)
        escape_room_model.move_agent(agent, (1, 1))
        escape_room_model.collect_object(agent, "key1")
        escape_room_model.move_agent(agent, (3, 2))
        escape_room_model.use_door(agent, "door1")
        
        # Move to puzzle
        puzzle_pos = (7, 2)
        escape_room_model.move_agent(agent, puzzle_pos)
        
        # Try wrong solution
        success = escape_room_model.solve_puzzle(agent, "puzzle1", "wrong_answer")
        assert success is False
        
        # Try correct solution
        success = escape_room_model.solve_puzzle(agent, "puzzle1", "sequence_123")
        assert success is True
        assert "key2" in agent.resources  # Should receive reward
    
    def test_escape_condition_checking(self, escape_room_model):
        """Test escape condition evaluation"""
        # Initially, no agents should have escaped
        assert escape_room_model.check_escape_conditions() is False
        
        # Get 2 agents to exit room
        agents = escape_room_model.schedule.agents[:2]
        
        for agent in agents:
            # Give them keys and get them to exit room
            agent.resources.extend(["key1", "key2"])
            
            # Move through rooms to exit
            escape_room_model.move_agent(agent, (3, 2))
            escape_room_model.use_door(agent, "door1")
            escape_room_model.move_agent(agent, (2, 4))
            escape_room_model.use_door(agent, "door2")
            
            assert agent.current_room == "exit_room"
        
        # Now escape conditions should be met
        assert escape_room_model.check_escape_conditions() is True
    
    def test_time_limit_mechanics(self, escape_room_model):
        """Test time limit enforcement"""
        initial_time = escape_room_model.time_remaining
        assert initial_time == 300
        
        # Step model multiple times
        for _ in range(10):
            escape_room_model.step()
        
        # Time should have decreased
        assert escape_room_model.time_remaining < initial_time
        
        # Model should stop when time runs out
        escape_room_model.time_remaining = 0
        escape_room_model.step()
        assert escape_room_model.running is False


class TestEscapeRoomAgent:
    """Test escape room agent behavior"""
    
    @pytest.fixture
    def test_agent(self):
        """Create test agent"""
        return EscapeRoomAgent(1, Mock())
    
    def test_agent_creation(self, test_agent):
        """Test agent can be created"""
        assert test_agent is not None
        assert test_agent.unique_id == 1
        assert hasattr(test_agent, 'health')
        assert hasattr(test_agent, 'energy')
        assert hasattr(test_agent, 'resources')
        assert test_agent.health == 100
        assert test_agent.energy == 1.0
    
    def test_agent_health_and_energy(self, test_agent):
        """Test agent health and energy mechanics"""
        initial_health = test_agent.health
        initial_energy = test_agent.energy
        
        # Damage agent
        test_agent.take_damage(20)
        assert test_agent.health == initial_health - 20
        
        # Consume energy
        test_agent.consume_energy(0.3)
        assert test_agent.energy == initial_energy - 0.3
        
        # Agent should be alive
        assert test_agent.is_alive() is True
        
        # Kill agent
        test_agent.health = 0
        assert test_agent.is_alive() is False
    
    def test_agent_resource_management(self, test_agent):
        """Test agent resource management"""
        assert len(test_agent.resources) == 0
        
        # Add resources
        test_agent.add_resource("key1")
        test_agent.add_resource("tool1")
        
        assert len(test_agent.resources) == 2
        assert test_agent.has_resource("key1") is True
        assert test_agent.has_resource("key2") is False
        
        # Remove resource
        removed = test_agent.remove_resource("key1")
        assert removed == "key1"
        assert test_agent.has_resource("key1") is False
        assert len(test_agent.resources) == 1


class TestRoomObjects:
    """Test room object system"""
    
    def test_key_object(self):
        """Test key object functionality"""
        key = Key("key1", (1, 1), "room1")
        
        assert key.object_id == "key1"
        assert key.position == (1, 1)
        assert key.room == "room1"
        assert key.can_be_collected() is True
        
        # Collect key
        result = key.interact(Mock())
        assert result["success"] is True
        assert result["action"] == "collected"
    
    def test_door_object(self):
        """Test door object functionality"""
        door = Door("door1", (4, 2), ["room1", "room2"], True, "key1")
        
        assert door.locked is True
        assert door.key_required == "key1"
        
        # Try to use without key
        agent_no_key = Mock()
        agent_no_key.has_resource.return_value = False
        
        result = door.interact(agent_no_key)
        assert result["success"] is False
        assert result["reason"] == "door_locked"
        
        # Use with key
        agent_with_key = Mock()
        agent_with_key.has_resource.return_value = True
        
        result = door.interact(agent_with_key)
        assert result["success"] is True
        assert result["action"] == "door_opened"
    
    def test_puzzle_object(self):
        """Test puzzle object functionality"""
        puzzle = Puzzle("puzzle1", (7, 2), "room2", "sequence_123", "key2")
        
        assert puzzle.solution == "sequence_123"
        assert puzzle.reward == "key2"
        assert puzzle.solved is False
        
        # Try wrong solution
        result = puzzle.solve("wrong_answer")
        assert result["success"] is False
        assert puzzle.solved is False
        
        # Try correct solution
        result = puzzle.solve("sequence_123")
        assert result["success"] is True
        assert puzzle.solved is True
        assert result["reward"] == "key2"
    
    def test_tool_object(self):
        """Test tool object functionality"""
        tool = Tool("crowbar", (3, 3), "room1", "metal_bar", 10)
        
        assert tool.tool_type == "metal_bar"
        assert tool.durability == 10
        assert tool.can_be_collected() is True
        
        # Use tool
        result = tool.use()
        assert result["success"] is True
        assert tool.durability == 9  # Should decrease
        
        # Use until broken
        for _ in range(9):
            tool.use()
        
        assert tool.durability == 0
        assert tool.is_broken() is True


class TestResourceManager:
    """Test Mesa resource management system"""
    
    @pytest.fixture
    def resource_manager(self):
        """Create resource manager"""
        return MesaResourceManager()
    
    def test_resource_manager_creation(self, resource_manager):
        """Test resource manager can be created"""
        assert resource_manager is not None
        assert hasattr(resource_manager, 'resources')
        assert hasattr(resource_manager, 'agent_resources')
    
    def test_resource_registration(self, resource_manager):
        """Test registering resources"""
        resource_manager.register_resource("key1", "key", (1, 1), "room1")
        resource_manager.register_resource("tool1", "tool", (2, 2), "room1")
        
        assert len(resource_manager.resources) == 2
        assert "key1" in resource_manager.resources
        assert resource_manager.resources["key1"]["type"] == "key"
        assert resource_manager.resources["key1"]["position"] == (1, 1)
    
    def test_resource_claiming(self, resource_manager):
        """Test resource claiming by agents"""
        resource_manager.register_resource("key1", "key", (1, 1), "room1")
        
        # Claim resource
        success = resource_manager.claim_resource("agent1", "key1")
        assert success is True
        assert resource_manager.get_resource_owner("key1") == "agent1"
        assert "key1" in resource_manager.agent_resources.get("agent1", [])
        
        # Try to claim already claimed resource
        success = resource_manager.claim_resource("agent2", "key1")
        assert success is False
    
    def test_resource_transfer(self, resource_manager):
        """Test transferring resources between agents"""
        resource_manager.register_resource("key1", "key", (1, 1), "room1")
        resource_manager.claim_resource("agent1", "key1")
        
        # Transfer resource
        success = resource_manager.transfer_resource("key1", "agent1", "agent2")
        assert success is True
        assert resource_manager.get_resource_owner("key1") == "agent2"
        assert "key1" not in resource_manager.agent_resources.get("agent1", [])
        assert "key1" in resource_manager.agent_resources.get("agent2", [])
    
    def test_resource_scarcity(self, resource_manager):
        """Test resource scarcity mechanics"""
        # Register limited resources
        resource_manager.register_resource("rare_key", "key", (5, 5), "room2", quantity=1)
        
        # First agent claims it
        success = resource_manager.claim_resource("agent1", "rare_key")
        assert success is True
        
        # Second agent can't claim it
        success = resource_manager.claim_resource("agent2", "rare_key") 
        assert success is False
        
        # Check scarcity status
        scarcity_info = resource_manager.get_scarcity_info()
        assert "rare_key" in scarcity_info
        assert scarcity_info["rare_key"]["available"] == 0
        assert scarcity_info["rare_key"]["total"] == 1


class TestEnvironmentHazards:
    """Test environmental hazards and challenges"""
    
    @pytest.fixture
    def hazard_manager(self):
        """Create hazard manager"""
        return HazardManager()
    
    def test_hazard_manager_creation(self, hazard_manager):
        """Test hazard manager can be created"""
        assert hazard_manager is not None
        assert hasattr(hazard_manager, 'hazards')
        assert hasattr(hazard_manager, 'active_hazards')
    
    def test_hazard_creation(self, hazard_manager):
        """Test creating environmental hazards"""
        hazard = EnvironmentHazard(
            "fire1", "fire", (5, 5), "room2", 
            damage_per_turn=10, radius=2
        )
        
        hazard_manager.add_hazard(hazard)
        
        assert len(hazard_manager.hazards) == 1
        assert "fire1" in hazard_manager.hazards
    
    def test_hazard_damage_calculation(self, hazard_manager):
        """Test hazard damage to agents"""
        fire_hazard = EnvironmentHazard(
            "fire1", "fire", (5, 5), "room2",
            damage_per_turn=15, radius=2
        )
        hazard_manager.add_hazard(fire_hazard)
        
        # Agent within hazard radius
        agent_in_range = Mock()
        agent_in_range.pos = (4, 5)  # Distance 1 from hazard
        agent_in_range.current_room = "room2"
        
        damage = hazard_manager.calculate_damage(agent_in_range)
        assert damage > 0
        
        # Agent outside hazard radius
        agent_out_range = Mock()
        agent_out_range.pos = (8, 8)  # Distance > 2 from hazard
        agent_out_range.current_room = "room2"
        
        damage = hazard_manager.calculate_damage(agent_out_range)
        assert damage == 0
    
    def test_dynamic_hazard_activation(self, hazard_manager):
        """Test dynamic hazard activation/deactivation"""
        hazard = EnvironmentHazard("trap1", "spike_trap", (3, 3), "room1")
        hazard.active = False
        hazard_manager.add_hazard(hazard)
        
        # Activate hazard based on trigger
        hazard_manager.trigger_hazard("trap1", trigger_type="pressure_plate")
        
        activated_hazard = hazard_manager.get_hazard("trap1")
        assert activated_hazard.active is True
    
    def test_hazard_mitigation(self, hazard_manager):
        """Test hazard mitigation strategies"""
        poison_hazard = EnvironmentHazard(
            "poison1", "poison_gas", (6, 6), "room2",
            damage_per_turn=5, mitigation_tools=["gas_mask"]
        )
        hazard_manager.add_hazard(poison_hazard)
        
        # Agent without mitigation tool
        agent_no_tool = Mock()
        agent_no_tool.pos = (6, 6)
        agent_no_tool.current_room = "room2"
        agent_no_tool.has_resource.return_value = False
        
        damage = hazard_manager.calculate_damage(agent_no_tool)
        assert damage == 5
        
        # Agent with mitigation tool
        agent_with_tool = Mock()
        agent_with_tool.pos = (6, 6)
        agent_with_tool.current_room = "room2"
        agent_with_tool.has_resource.return_value = True
        
        damage = hazard_manager.calculate_damage(agent_with_tool)
        assert damage == 0  # Mitigated


class TestCollisionDetection:
    """Test collision detection and spatial constraints"""
    
    def test_agent_collision_detection(self):
        """Test agent-to-agent collision detection"""
        from src.escape_room_sim.mesa.collision_detection import CollisionDetector
        
        detector = CollisionDetector()
        
        # Place two agents at same position
        agent1 = Mock()
        agent1.pos = (3, 3)
        agent1.unique_id = 1
        
        agent2 = Mock()
        agent2.pos = (3, 3)
        agent2.unique_id = 2
        
        collision = detector.check_agent_collision([agent1, agent2])
        assert collision is True
        
        # Move agents apart
        agent2.pos = (4, 4)
        collision = detector.check_agent_collision([agent1, agent2])
        assert collision is False
    
    def test_object_collision_detection(self):
        """Test collision with room objects"""
        from src.escape_room_sim.mesa.collision_detection import CollisionDetector
        
        detector = CollisionDetector()
        
        # Create wall object
        wall = RoomObject("wall1", (5, 5), "room1")
        wall.solid = True
        
        # Test collision
        target_pos = (5, 5)
        collision = detector.check_object_collision(target_pos, [wall])
        assert collision is True
        
        # Test no collision
        target_pos = (6, 6)
        collision = detector.check_object_collision(target_pos, [wall])
        assert collision is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])