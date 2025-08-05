"""
Mesa Escape Room Model

Agent D: State Management & Integration Specialist
Custom Mesa model implementing escape room environment with multiple rooms, 
interactive objects, and collaborative survival mechanics.
"""

import mesa
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
import random
import json

from .room_objects import Room, Door, Key, Puzzle, Tool, RoomObject
from .resource_manager import MesaResourceManager
from .environment_hazards import HazardManager, EnvironmentHazard


class EscapeRoomAgent(mesa.Agent):
    """
    Agent in the escape room environment
    
    Represents agents that can move, interact with objects, and collaborate.
    """
    
    def __init__(self, unique_id: int, model: 'EscapeRoomModel'):
        super().__init__(model)
        self.unique_id = unique_id  # Mesa 3.x requires manual setting
        self.health = 100
        self.energy = 1.0
        self.resources: List[str] = []
        self.current_room: Optional[str] = None
        self.status = "active"
        self.last_action: Optional[str] = None
        self.action_history: List[Dict[str, Any]] = []
        
        # Agent capabilities
        self.movement_speed = 1.0
        self.examination_skill = 0.7
        self.communication_range = 3
        
        # State tracking
        self.last_moved_timestamp = datetime.now()
        self.last_interaction_timestamp = datetime.now()
        self.failed_action_count = 0
    
    def step(self):
        """Agent step - called each model step"""
        # Decay energy over time
        self.energy = max(0.0, self.energy - 0.01)
        
        # Update status based on health and energy
        if self.health <= 0:
            self.status = "dead"
        elif self.energy <= 0.1:
            self.status = "exhausted"
        elif self.health < 20:
            self.status = "injured"
        else:
            self.status = "active"
        
        # Determine current room based on position
        self._update_current_room()
    
    def move_to(self, target_pos: Tuple[int, int]) -> bool:
        """Move agent to target position"""
        if not self._can_move():
            return False
        
        # Check if movement is valid
        if not self.model.is_valid_position(target_pos, self):
            return False
        
        # Move agent
        old_pos = self.pos
        self.model.grid.move_agent(self, target_pos)
        self.last_moved_timestamp = datetime.now()
        self.last_action = f"moved_from_{old_pos}_to_{target_pos}"
        
        # Consume energy
        distance = np.sqrt((target_pos[0] - old_pos[0])**2 + (target_pos[1] - old_pos[1])**2)
        energy_cost = distance * 0.05
        self.consume_energy(energy_cost)
        
        self._record_action("move", {"from": old_pos, "to": target_pos, "energy_cost": energy_cost})
        return True
    
    def examine_object(self, object_id: str) -> Dict[str, Any]:
        """Examine a room object"""
        if not self._can_act():
            return {"success": False, "reason": "cannot_act"}
        
        room_object = self.model.get_object(object_id)
        if not room_object:
            return {"success": False, "reason": "object_not_found"}
        
        # Check if object is in range
        distance = np.sqrt((self.pos[0] - room_object.position[0])**2 + 
                          (self.pos[1] - room_object.position[1])**2)
        if distance > 2:
            return {"success": False, "reason": "too_far"}
        
        # Perform examination
        examination_result = room_object.examine(self)
        self.consume_energy(0.1)
        self.last_action = f"examined_{object_id}"
        
        self._record_action("examine", {
            "object_id": object_id,
            "result": examination_result,
            "energy_cost": 0.1
        })
        
        return examination_result
    
    def interact_with_object(self, object_id: str, interaction_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """Interact with a room object"""
        if not self._can_act():
            return {"success": False, "reason": "cannot_act"}
        
        room_object = self.model.get_object(object_id)
        if not room_object:
            return {"success": False, "reason": "object_not_found"}
        
        # Check range
        distance = np.sqrt((self.pos[0] - room_object.position[0])**2 + 
                          (self.pos[1] - room_object.position[1])**2)
        if distance > 1:
            return {"success": False, "reason": "too_far"}
        
        # Perform interaction
        interaction_result = room_object.interact(self, interaction_data or {})
        self.consume_energy(0.15)
        self.last_action = f"interacted_with_{object_id}"
        
        # Handle successful interactions
        if interaction_result.get("success"):
            if interaction_result.get("action") == "collected":
                self.add_resource(object_id)
            elif "reward" in interaction_result:
                self.add_resource(interaction_result["reward"])
        
        self._record_action("interact", {
            "object_id": object_id,
            "interaction_data": interaction_data,
            "result": interaction_result,
            "energy_cost": 0.15
        })
        
        return interaction_result
    
    def communicate_with_agent(self, target_agent_id: int, message: str) -> Dict[str, Any]:
        """Communicate with another agent"""
        if not self._can_act():
            return {"success": False, "reason": "cannot_act"}
        
        target_agent = None
        for agent in self.model.agents:
            if agent.unique_id == target_agent_id:
                target_agent = agent
                break
        
        if not target_agent:
            return {"success": False, "reason": "agent_not_found"}
        
        # Check communication range
        distance = np.sqrt((self.pos[0] - target_agent.pos[0])**2 + 
                          (self.pos[1] - target_agent.pos[1])**2)
        if distance > self.communication_range:
            return {"success": False, "reason": "out_of_range"}
        
        # Send message
        communication_result = {
            "success": True,
            "message": message,
            "sender": self.unique_id,
            "recipient": target_agent_id,
            "timestamp": datetime.now(),
            "distance": distance
        }
        
        # Record in model communication log
        if not hasattr(self.model, 'communication_log'):
            self.model.communication_log = []
        self.model.communication_log.append(communication_result)
        
        self.consume_energy(0.05)
        self.last_action = f"communicated_with_{target_agent_id}"
        
        self._record_action("communicate", communication_result)
        
        return communication_result
    
    def add_resource(self, resource_id: str):
        """Add resource to agent's inventory"""
        if resource_id not in self.resources:
            self.resources.append(resource_id)
    
    def remove_resource(self, resource_id: str) -> Optional[str]:
        """Remove resource from agent's inventory"""
        if resource_id in self.resources:
            self.resources.remove(resource_id)
            return resource_id
        return None
    
    def has_resource(self, resource_id: str) -> bool:
        """Check if agent has specific resource"""
        return resource_id in self.resources
    
    def take_damage(self, damage: int):
        """Take damage and update health"""
        self.health = max(0, self.health - damage)
        if self.health <= 0:
            self.status = "dead"
    
    def consume_energy(self, amount: float):
        """Consume energy and update status"""
        self.energy = max(0.0, self.energy - amount)
        if self.energy <= 0.1:
            self.status = "exhausted"
    
    def is_alive(self) -> bool:
        """Check if agent is alive"""
        return self.health > 0
    
    def can_escape(self) -> bool:
        """Check if agent can escape (in exit room)"""
        return self.current_room == "exit_room" and self.is_alive()
    
    # Private helper methods
    
    def _can_move(self) -> bool:
        """Check if agent can move"""
        return self.is_alive() and self.energy > 0.1 and self.status != "exhausted"
    
    def _can_act(self) -> bool:
        """Check if agent can perform actions"""
        return self.is_alive() and self.energy > 0.05
    
    def _update_current_room(self):
        """Update current room based on position"""
        for room_id, room in self.model.rooms.items():
            if room.contains_position(self.pos):
                self.current_room = room_id
                break
    
    def _record_action(self, action_type: str, action_data: Dict[str, Any]):
        """Record action in history"""
        action_record = {
            "action_type": action_type,
            "timestamp": datetime.now(),
            "position": self.pos,
            "room": self.current_room,
            "data": action_data
        }
        self.action_history.append(action_record)
        
        # Keep only recent history
        if len(self.action_history) > 50:
            self.action_history = self.action_history[-50:]


class EscapeRoomModel(mesa.Model):
    """
    Multi-room escape room Mesa model
    
    Features:
    - Multiple connected rooms
    - Interactive objects (doors, keys, puzzles, tools)
    - Agent collaboration mechanics
    - Time pressure and win conditions
    - Resource management
    - Environmental hazards
    """
    
    def __init__(self, room_config: Dict[str, Any]):
        super().__init__()
        
        # Model configuration
        self.width = room_config.get("width", 10)
        self.height = room_config.get("height", 10)
        self.num_agents = room_config.get("num_agents", 3)
        self.time_limit = room_config.get("time_limit", 300)  # seconds
        
        # Initialize components  
        self.grid = mesa.space.MultiGrid(self.width, self.height, True)
        # Mesa 3.x uses built-in agent management instead of separate scheduler
        self.running = True
        
        # Time tracking
        self.start_time = datetime.now()
        self.time_remaining = self.time_limit
        self.model_step_count = 0
        
        # Game state
        self.rooms: Dict[str, Room] = {}
        self.doors: Dict[str, Door] = {}
        self.objects: Dict[str, RoomObject] = {}
        self.escape_conditions = room_config.get("escape_conditions", {})
        
        # Managers
        self.resource_manager = MesaResourceManager()
        self.hazard_manager = HazardManager()
        
        # Logs
        self.communication_log: List[Dict[str, Any]] = []
        self.event_log: List[Dict[str, Any]] = []
        
        # Initialize room layout
        self._create_rooms(room_config.get("rooms", []))
        self._create_doors(room_config.get("doors", []))
        self._create_objects(room_config.get("objects", []))
        self._create_agents()
        
        # Setup data collection (Mesa 3.x compatible)
        self.datacollector = mesa.DataCollector(
            model_reporters={
                "Time Remaining": "time_remaining",
                "Agents Alive": lambda m: sum(1 for a in m.agents if a.is_alive()),
                "Agents Escaped": lambda m: sum(1 for a in m.agents if a.can_escape()),
                "Objects Collected": lambda m: sum(len(a.resources) for a in m.agents)
            },
            agent_reporters={
                "Health": "health",
                "Energy": "energy",
                "Resources": lambda a: len(a.resources),
                "Room": "current_room",
                "Status": "status"
            }
        )
    
    def step(self):
        """Single model step"""
        # Update time
        self.time_remaining = max(0, self.time_remaining - 1)
        self.model_step_count += 1
        
        # Check if time is up
        if self.time_remaining <= 0:
            self.running = False
            self._log_event("time_expired", {"final_step": self.model_step_count})
            return
        
        # Check escape conditions
        if self.check_escape_conditions():
            self.running = False
            self._log_event("escape_successful", {
                "escaped_agents": [a.unique_id for a in self.agents if a.can_escape()],
                "final_step": self.model_step_count
            })
            return
        
        # Step all agents (Mesa 3.x built-in agent management)
        for agent in self.agents:
            agent.step()
        
        # Update hazards
        self.hazard_manager.step(self.agents)
        
        # Collect data
        self.datacollector.collect(self)
        
        # Log periodic status
        if self.model_step_count % 50 == 0:
            self._log_periodic_status()
    
    def move_agent(self, agent: EscapeRoomAgent, target_pos: Tuple[int, int]) -> bool:
        """Move agent to target position with validation"""
        if not self.is_valid_position(target_pos, agent):
            return False
        
        return agent.move_to(target_pos)
    
    def use_door(self, agent: EscapeRoomAgent, door_id: str) -> bool:
        """Agent attempts to use a door"""
        if door_id not in self.doors:
            return False
        
        door = self.doors[door_id]
        
        # Check if agent is adjacent to door
        distance = np.sqrt((agent.pos[0] - door.position[0])**2 + 
                          (agent.pos[1] - door.position[1])**2)
        if distance > 1:
            return False
        
        # Check if door can be used
        interaction_result = door.interact(agent)
        
        if interaction_result.get("success"):
            # Move agent through door
            from_room = agent.current_room
            
            # Determine target room
            if from_room in door.connects:
                target_room = [r for r in door.connects if r != from_room][0]
                target_room_obj = self.rooms[target_room]
                
                # Find valid position in target room
                target_pos = target_room_obj.get_center_position()
                if self.is_valid_position(target_pos, agent):
                    self.grid.move_agent(agent, target_pos)
                    agent.current_room = target_room
                    agent.last_action = f"entered_{target_room}_via_{door_id}"
                    
                    self._log_event("door_used", {
                        "agent_id": agent.unique_id,
                        "door_id": door_id,
                        "from_room": from_room,
                        "to_room": target_room
                    })
                    
                    return True
        
        return False
    
    def collect_object(self, agent: EscapeRoomAgent, object_id: str) -> bool:
        """Agent attempts to collect an object"""
        if object_id not in self.objects:
            return False
        
        room_object = self.objects[object_id]
        interaction_result = agent.interact_with_object(object_id)
        
        if interaction_result.get("success") and interaction_result.get("action") == "collected":
            # Remove object from model
            del self.objects[object_id]
            
            self._log_event("object_collected", {
                "agent_id": agent.unique_id,
                "object_id": object_id,
                "position": room_object.position
            })
            
            return True
        
        return False
    
    def solve_puzzle(self, agent: EscapeRoomAgent, puzzle_id: str, solution: str) -> bool:
        """Agent attempts to solve a puzzle"""
        if puzzle_id not in self.objects:
            return False
        
        puzzle = self.objects[puzzle_id]
        if not isinstance(puzzle, Puzzle):
            return False
        
        # Check if agent is close enough
        distance = np.sqrt((agent.pos[0] - puzzle.position[0])**2 + 
                          (agent.pos[1] - puzzle.position[1])**2)
        if distance > 1:
            return False
        
        # Attempt to solve
        solve_result = puzzle.solve(solution)
        
        if solve_result.get("success"):
            # Award puzzle reward
            reward = solve_result.get("reward")
            if reward:
                agent.add_resource(reward)
            
            self._log_event("puzzle_solved", {
                "agent_id": agent.unique_id,
                "puzzle_id": puzzle_id,
                "reward": reward,
                "solution": solution
            })
            
            return True
        
        return False
    
    def is_valid_position(self, pos: Tuple[int, int], agent: EscapeRoomAgent) -> bool:
        """Check if position is valid for agent"""
        x, y = pos
        
        # Check bounds
        if not (0 <= x < self.width and 0 <= y < self.height):
            return False
        
        # Check if position is inside a room
        in_room = False
        for room in self.rooms.values():
            if room.contains_position(pos):
                in_room = True
                break
        
        if not in_room:
            return False
        
        # Check for solid objects
        for obj in self.objects.values():
            if hasattr(obj, 'solid') and obj.solid and obj.position == pos:
                return False
        
        return True
    
    def check_escape_conditions(self) -> bool:
        """Check if escape conditions are met"""
        agents_required = self.escape_conditions.get("agents_required", 1)
        exit_room = self.escape_conditions.get("exit_room", "exit_room")
        
        escaped_agents = sum(1 for agent in self.agents 
                           if agent.current_room == exit_room and agent.is_alive())
        
        return escaped_agents >= agents_required
    
    def get_room(self, room_id: str) -> Optional[Room]:
        """Get room by ID"""
        return self.rooms.get(room_id)
    
    def get_door(self, door_id: str) -> Optional[Door]:
        """Get door by ID"""
        return self.doors.get(door_id)
    
    def get_object(self, object_id: str) -> Optional[RoomObject]:
        """Get object by ID"""
        return self.objects.get(object_id)
    
    def get_agents_in_room(self, room_id: str) -> List[EscapeRoomAgent]:
        """Get all agents in a specific room"""
        return [agent for agent in self.agents if agent.current_room == room_id]
    
    def get_model_state(self) -> Dict[str, Any]:
        """Get complete model state"""
        return {
            "time_remaining": self.time_remaining,
            "step_count": self.model_step_count,
            "running": self.running,
            "agents": {
                agent.unique_id: {
                    "position": agent.pos,
                    "health": agent.health,
                    "energy": agent.energy,
                    "resources": agent.resources.copy(),
                    "current_room": agent.current_room,
                    "status": agent.status
                }
                for agent in self.agents
            },
            "rooms": {
                room_id: {
                    "bounds": room.bounds,
                    "room_type": room.room_type,
                    "agents": [a.unique_id for a in self.get_agents_in_room(room_id)]
                }
                for room_id, room in self.rooms.items()
            },
            "objects": {
                obj_id: {
                    "type": type(obj).__name__,
                    "position": obj.position,
                    "room": obj.room,
                    "available": getattr(obj, 'available', True)
                }
                for obj_id, obj in self.objects.items()
            },
            "communications": self.communication_log.copy(),
            "events": self.event_log.copy()
        }
    
    # Private initialization methods
    
    def _create_rooms(self, room_configs: List[Dict[str, Any]]):
        """Create rooms from configuration"""
        for config in room_configs:
            room = Room(
                room_id=config["id"],
                bounds=config["bounds"],
                room_type=config.get("room_type", "standard")
            )
            self.rooms[config["id"]] = room
    
    def _create_doors(self, door_configs: List[Dict[str, Any]]):
        """Create doors from configuration"""
        for config in door_configs:
            door = Door(
                object_id=config["id"],
                position=tuple(config["position"]),
                connects=config["connects"],
                locked=config.get("locked", False),
                key_required=config.get("key_required")
            )
            self.doors[config["id"]] = door
            self.objects[config["id"]] = door
    
    def _create_objects(self, object_configs: List[Dict[str, Any]]):
        """Create objects from configuration"""
        for config in object_configs:
            obj_type = config["type"]
            
            if obj_type == "key":
                obj = Key(
                    object_id=config["id"],
                    position=tuple(config["position"]),
                    room=config["room"]
                )
            elif obj_type == "puzzle":
                obj = Puzzle(
                    object_id=config["id"],
                    position=tuple(config["position"]),
                    room=config["room"],
                    solution=config["solution"],
                    reward=config.get("reward")
                )
            elif obj_type == "tool":
                obj = Tool(
                    object_id=config["id"],
                    position=tuple(config["position"]),
                    room=config["room"],
                    tool_type=config.get("tool_type", "generic"),
                    durability=config.get("durability", 10)
                )
            else:
                # Generic room object
                obj = RoomObject(
                    object_id=config["id"],
                    position=tuple(config["position"]),
                    room=config["room"]
                )
            
            self.objects[config["id"]] = obj
    
    def _create_agents(self):
        """Create agents and place them in starting room"""
        # Find starting room
        starting_room = None
        for room in self.rooms.values():
            if room.room_type == "starting_room":
                starting_room = room
                break
        
        if not starting_room:
            # Use first room as default
            starting_room = list(self.rooms.values())[0]
        
        # Create agents
        for i in range(self.num_agents):
            agent = EscapeRoomAgent(i, self)
            
            # Find valid starting position
            center = starting_room.get_center_position()
            start_pos = self._find_valid_position_near(center, agent)
            
            self.grid.place_agent(agent, start_pos)
            agent.current_room = starting_room.room_id
            # Mesa 3.x: agents are automatically managed by the model
    
    def _find_valid_position_near(self, center: Tuple[int, int], 
                                agent: EscapeRoomAgent) -> Tuple[int, int]:
        """Find valid position near center point"""
        x, y = center
        
        # Try center first
        if self.is_valid_position(center, agent):
            return center
        
        # Search in expanding radius
        for radius in range(1, 5):
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    if abs(dx) == radius or abs(dy) == radius:  # Only check perimeter
                        pos = (x + dx, y + dy)
                        if self.is_valid_position(pos, agent):
                            return pos
        
        # Fallback - return center anyway
        return center
    
    def _log_event(self, event_type: str, event_data: Dict[str, Any]):
        """Log game event"""
        event = {
            "type": event_type,
            "timestamp": datetime.now(),
            "step": self.model_step_count,
            "data": event_data
        }
        self.event_log.append(event)
    
    def _log_periodic_status(self):
        """Log periodic status update"""
        alive_agents = sum(1 for a in self.agents if a.is_alive())
        escaped_agents = sum(1 for a in self.agents if a.can_escape())
        
        self._log_event("periodic_status", {
            "time_remaining": self.time_remaining,
            "agents_alive": alive_agents,
            "agents_escaped": escaped_agents,
            "total_communications": len(self.communication_log),
            "total_objects": len(self.objects)
        })