"""
Room Objects and Interactive Elements

Agent D: State Management & Integration Specialist
Implements rooms, doors, keys, puzzles, tools and other interactive objects.
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime
from abc import ABC, abstractmethod
import random


class RoomObject(ABC):
    """Base class for all room objects"""
    
    def __init__(self, object_id: str, position: Tuple[int, int], room: str):
        self.object_id = object_id
        self.position = position
        self.room = room
        self.object_type = "generic"
        self.solid = False
        self.visible = True
        self.available = True
        self.created_timestamp = datetime.now()
        self.interaction_history: List[Dict[str, Any]] = []
    
    def examine(self, agent) -> Dict[str, Any]:
        """Examine the object - returns information about it"""
        examination_result = {
            "success": True,
            "object_id": self.object_id,
            "object_type": self.object_type,
            "description": self.get_description(),
            "visible_properties": self.get_visible_properties(),
            "timestamp": datetime.now()
        }
        
        self._record_interaction(agent, "examine", examination_result)
        return examination_result
    
    def interact(self, agent, interaction_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """Interact with the object - default implementation"""
        interaction_result = {
            "success": False,
            "reason": "no_interaction_available",
            "object_id": self.object_id,
            "timestamp": datetime.now()
        }
        
        self._record_interaction(agent, "interact", interaction_result)
        return interaction_result
    
    def get_description(self) -> str:
        """Get text description of object"""
        return f"A {self.object_type} object"
    
    def get_visible_properties(self) -> Dict[str, Any]:
        """Get properties visible to agents"""
        return {
            "position": self.position,
            "room": self.room,
            "available": self.available,
            "solid": self.solid
        }
    
    def _record_interaction(self, agent, interaction_type: str, result: Dict[str, Any]):
        """Record interaction in history"""
        record = {
            "agent_id": getattr(agent, 'unique_id', 'unknown'),
            "interaction_type": interaction_type,
            "timestamp": datetime.now(),
            "result": result
        }
        self.interaction_history.append(record)
        
        # Keep limited history
        if len(self.interaction_history) > 20:
            self.interaction_history = self.interaction_history[-20:]


class InteractiveObject(RoomObject):
    """Base class for objects that can be interacted with"""
    
    def __init__(self, object_id: str, position: Tuple[int, int], room: str):
        super().__init__(object_id, position, room)
        self.interaction_count = 0
        self.max_interactions = -1  # -1 = unlimited
        self.requires_tool = None
        self.interaction_cooldown = 0  # seconds
        self.last_interaction_time = None
    
    def can_interact(self, agent) -> Tuple[bool, str]:
        """Check if agent can interact with object"""
        # Check interaction limit
        if self.max_interactions > 0 and self.interaction_count >= self.max_interactions:
            return False, "interaction_limit_reached"
        
        # Check cooldown
        if self.last_interaction_time and self.interaction_cooldown > 0:
            time_since_last = (datetime.now() - self.last_interaction_time).total_seconds()
            if time_since_last < self.interaction_cooldown:
                return False, "interaction_on_cooldown"
        
        # Check required tool
        if self.requires_tool and not agent.has_resource(self.requires_tool):
            return False, f"requires_tool_{self.requires_tool}"
        
        # Check availability
        if not self.available:
            return False, "object_not_available"
        
        return True, "can_interact"
    
    def interact(self, agent, interaction_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """Enhanced interact method with validation"""
        can_interact, reason = self.can_interact(agent)
        
        if not can_interact:
            result = {
                "success": False,
                "reason": reason,
                "object_id": self.object_id,
                "timestamp": datetime.now()
            }
            self._record_interaction(agent, "interact_failed", result)
            return result
        
        # Perform specific interaction
        result = self._perform_interaction(agent, interaction_data or {})
        
        # Update interaction tracking
        if result.get("success"):
            self.interaction_count += 1
            self.last_interaction_time = datetime.now()
        
        self._record_interaction(agent, "interact", result)
        return result
    
    @abstractmethod
    def _perform_interaction(self, agent, interaction_data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform the specific interaction - implemented by subclasses"""
        pass


class ResourceObject(InteractiveObject):
    """Base class for collectible resources"""
    
    def __init__(self, object_id: str, position: Tuple[int, int], room: str):
        super().__init__(object_id, position, room)
        self.max_interactions = 1  # Can only be collected once
        self.collectible = True
    
    def can_be_collected(self) -> bool:
        """Check if object can be collected"""
        return self.collectible and self.available and self.interaction_count == 0
    
    def _perform_interaction(self, agent, interaction_data: Dict[str, Any]) -> Dict[str, Any]:
        """Default collection interaction"""
        if self.can_be_collected():
            self.available = False
            return {
                "success": True,
                "action": "collected",
                "object_id": self.object_id,
                "timestamp": datetime.now()
            }
        else:
            return {
                "success": False,
                "reason": "cannot_collect",
                "object_id": self.object_id,
                "timestamp": datetime.now()
            }


class Room:
    """Represents a room in the escape room"""
    
    def __init__(self, room_id: str, bounds: Dict[str, int], room_type: str = "standard"):
        self.room_id = room_id
        self.bounds = bounds  # {"x1": 0, "y1": 0, "x2": 4, "y2": 4}
        self.room_type = room_type
        self.description = f"A {room_type} room"
        self.objects: List[str] = []
        self.lighting = 1.0  # 0.0 = dark, 1.0 = bright
        self.temperature = 20.0  # Celsius
        self.hazards: List[str] = []
        self.visited_by: List[int] = []  # Agent IDs that have visited
    
    def contains_position(self, pos: Tuple[int, int]) -> bool:
        """Check if position is within room bounds"""
        x, y = pos
        return (self.bounds["x1"] <= x <= self.bounds["x2"] and 
                self.bounds["y1"] <= y <= self.bounds["y2"])
    
    def get_center_position(self) -> Tuple[int, int]:
        """Get center position of room"""
        center_x = (self.bounds["x1"] + self.bounds["x2"]) // 2
        center_y = (self.bounds["y1"] + self.bounds["y2"]) // 2
        return (center_x, center_y)
    
    def get_area(self) -> int:
        """Get room area"""
        width = self.bounds["x2"] - self.bounds["x1"] + 1
        height = self.bounds["y2"] - self.bounds["y1"] + 1
        return width * height
    
    def get_random_position(self) -> Tuple[int, int]:
        """Get random position within room"""
        x = random.randint(self.bounds["x1"], self.bounds["x2"])
        y = random.randint(self.bounds["y1"], self.bounds["y2"])
        return (x, y)
    
    def add_object(self, object_id: str):
        """Add object to room"""
        if object_id not in self.objects:
            self.objects.append(object_id)
    
    def remove_object(self, object_id: str):
        """Remove object from room"""
        if object_id in self.objects:
            self.objects.remove(object_id)
    
    def agent_entered(self, agent_id: int):
        """Record agent entering room"""
        if agent_id not in self.visited_by:
            self.visited_by.append(agent_id)


class Door(InteractiveObject):
    """Door connecting two rooms"""
    
    def __init__(self, object_id: str, position: Tuple[int, int], 
                 connects: List[str], locked: bool = False, 
                 key_required: str = None):
        super().__init__(object_id, position, "boundary")
        self.object_type = "door"
        self.connects = connects  # List of room IDs
        self.locked = locked
        self.key_required = key_required
        self.solid = True  # Blocks movement when locked
        self.door_material = "wood"
        self.door_condition = "good"
        self.times_used = 0
    
    def get_description(self) -> str:
        """Get door description"""
        status = "locked" if self.locked else "unlocked"
        material = self.door_material
        return f"A {material} door that is {status}, connecting {' and '.join(self.connects)}"
    
    def get_visible_properties(self) -> Dict[str, Any]:
        """Get visible door properties"""
        props = super().get_visible_properties()
        props.update({
            "locked": self.locked,
            "connects": self.connects,
            "material": self.door_material,
            "condition": self.door_condition,
            "key_required": self.key_required if self.locked else None
        })
        return props
    
    def _perform_interaction(self, agent, interaction_data: Dict[str, Any]) -> Dict[str, Any]:
        """Attempt to use/open door"""
        if not self.locked:
            # Door is unlocked - can pass through
            self.times_used += 1
            return {
                "success": True,
                "action": "door_opened",
                "from_room": interaction_data.get("from_room"),
                "to_room": interaction_data.get("to_room"),
                "timestamp": datetime.now()
            }
        
        # Door is locked - check for key
        if self.key_required and agent.has_resource(self.key_required):
            # Agent has key - unlock door
            self.locked = False
            self.solid = False
            self.times_used += 1
            
            return {
                "success": True,
                "action": "door_unlocked_and_opened",
                "key_used": self.key_required,
                "from_room": interaction_data.get("from_room"),
                "to_room": interaction_data.get("to_room"),
                "timestamp": datetime.now()
            }
        
        # Cannot open door
        reason = "door_locked"
        if self.key_required:
            reason = f"requires_key_{self.key_required}"
        
        return {
            "success": False,
            "reason": reason,
            "key_required": self.key_required,
            "timestamp": datetime.now()
        }
    
    def is_passable(self) -> bool:
        """Check if door can be passed through"""
        return not self.locked


class Key(ResourceObject):
    """Key object that can unlock doors"""
    
    def __init__(self, object_id: str, position: Tuple[int, int], room: str):
        super().__init__(object_id, position, room)
        self.object_type = "key"
        self.key_material = "metal"
        self.unlocks = []  # List of door IDs this key can unlock
    
    def get_description(self) -> str:
        """Get key description"""
        return f"A {self.key_material} key"
    
    def get_visible_properties(self) -> Dict[str, Any]:
        """Get visible key properties"""
        props = super().get_visible_properties()
        props.update({
            "material": self.key_material,
            "collectible": self.collectible
        })
        return props
    
    def can_unlock(self, door_id: str) -> bool:
        """Check if this key can unlock a specific door"""
        return door_id in self.unlocks or self.object_id in door_id


class Puzzle(InteractiveObject):
    """Puzzle object that requires solving"""
    
    def __init__(self, object_id: str, position: Tuple[int, int], room: str,
                 solution: str, reward: str = None):
        super().__init__(object_id, position, room)
        self.object_type = "puzzle"
        self.solution = solution
        self.reward = reward
        self.puzzle_type = "sequence"  # sequence, logic, pattern, etc.
        self.difficulty = "medium"
        self.solved = False
        self.attempts = 0
        self.max_attempts = 3
        self.hints_given = 0
        self.max_hints = 2
    
    def get_description(self) -> str:
        """Get puzzle description"""
        status = "solved" if self.solved else "unsolved"
        return f"A {self.difficulty} {self.puzzle_type} puzzle that is {status}"
    
    def get_visible_properties(self) -> Dict[str, Any]:
        """Get visible puzzle properties"""
        props = super().get_visible_properties()
        props.update({
            "puzzle_type": self.puzzle_type,
            "difficulty": self.difficulty,
            "solved": self.solved,
            "attempts_remaining": max(0, self.max_attempts - self.attempts),
            "hints_available": max(0, self.max_hints - self.hints_given)
        })
        return props
    
    def _perform_interaction(self, agent, interaction_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle puzzle interaction"""
        interaction_type = interaction_data.get("interaction_type", "solve")
        
        if interaction_type == "hint":
            return self._give_hint(agent)
        elif interaction_type == "solve":
            solution_attempt = interaction_data.get("solution", "")
            return self.solve(solution_attempt)
        else:
            return {
                "success": False,
                "reason": "unknown_interaction_type",
                "available_interactions": ["solve", "hint"],
                "timestamp": datetime.now()
            }
    
    def solve(self, solution_attempt: str) -> Dict[str, Any]:
        """Attempt to solve puzzle"""
        if self.solved:
            return {
                "success": False,
                "reason": "already_solved",
                "timestamp": datetime.now()
            }
        
        if self.attempts >= self.max_attempts:
            return {
                "success": False,
                "reason": "max_attempts_exceeded",
                "max_attempts": self.max_attempts,
                "timestamp": datetime.now()
            }
        
        self.attempts += 1
        
        if solution_attempt.lower().strip() == self.solution.lower().strip():
            # Correct solution
            self.solved = True
            self.available = False  # Puzzle is now completed
            
            result = {
                "success": True,
                "action": "puzzle_solved",
                "attempts_used": self.attempts,
                "timestamp": datetime.now()
            }
            
            if self.reward:
                result["reward"] = self.reward
            
            return result
        else:
            # Incorrect solution
            attempts_remaining = self.max_attempts - self.attempts
            
            return {
                "success": False,
                "reason": "incorrect_solution",
                "attempts_used": self.attempts,
                "attempts_remaining": attempts_remaining,
                "hint_available": self.hints_given < self.max_hints,
                "timestamp": datetime.now()
            }
    
    def _give_hint(self, agent) -> Dict[str, Any]:
        """Give puzzle hint to agent"""
        if self.hints_given >= self.max_hints:
            return {
                "success": False,
                "reason": "no_hints_remaining",
                "timestamp": datetime.now()
            }
        
        self.hints_given += 1
        
        # Generate hint based on solution
        hint = self._generate_hint()
        
        return {
            "success": True,
            "action": "hint_given",
            "hint": hint,
            "hints_used": self.hints_given,
            "hints_remaining": self.max_hints - self.hints_given,
            "timestamp": datetime.now()
        }
    
    def _generate_hint(self) -> str:
        """Generate hint for the puzzle"""
        # Simple hint system - could be more sophisticated
        solution_length = len(self.solution)
        
        if self.hints_given == 1:
            return f"The solution has {solution_length} characters"
        elif self.hints_given == 2:
            if solution_length > 0:
                first_char = self.solution[0]
                return f"The solution starts with '{first_char}'"
            else:
                return "The solution is very short"
        else:
            return "No more hints available"


class Tool(ResourceObject):
    """Tool object that can be used for various tasks"""
    
    def __init__(self, object_id: str, position: Tuple[int, int], room: str,
                 tool_type: str = "generic", durability: int = 10):
        super().__init__(object_id, position, room)
        self.object_type = "tool"
        self.tool_type = tool_type  # hammer, crowbar, screwdriver, etc.
        self.durability = durability
        self.max_durability = durability
        self.effectiveness = 1.0  # Multiplier for tool effectiveness
        self.uses = 0
        self.broken = False
    
    def get_description(self) -> str:
        """Get tool description"""
        condition = self._get_condition_description()
        return f"A {self.tool_type} in {condition} condition"
    
    def get_visible_properties(self) -> Dict[str, Any]:
        """Get visible tool properties"""
        props = super().get_visible_properties()
        props.update({
            "tool_type": self.tool_type,
            "condition": self._get_condition_description(),
            "durability_percent": int((self.durability / self.max_durability) * 100),
            "broken": self.broken,
            "effectiveness": self.effectiveness
        })
        return props
    
    def use(self) -> Dict[str, Any]:
        """Use the tool"""
        if self.broken:
            return {
                "success": False,
                "reason": "tool_broken",
                "timestamp": datetime.now()
            }
        
        # Use the tool
        self.uses += 1
        self.durability = max(0, self.durability - 1)
        
        if self.durability <= 0:
            self.broken = True
            self.effectiveness = 0.0
        elif self.durability < self.max_durability * 0.3:
            self.effectiveness = 0.5  # Reduced effectiveness when damaged
        
        return {
            "success": True,
            "action": "tool_used",
            "tool_type": self.tool_type,
            "durability_remaining": self.durability,
            "effectiveness": self.effectiveness,
            "broken": self.broken,
            "timestamp": datetime.now()
        }
    
    def repair(self, repair_amount: int = None) -> Dict[str, Any]:
        """Repair the tool"""
        if repair_amount is None:
            repair_amount = self.max_durability // 2
        
        old_durability = self.durability
        self.durability = min(self.max_durability, self.durability + repair_amount)
        
        if self.durability > 0:
            self.broken = False
            self.effectiveness = min(1.0, self.durability / self.max_durability)
        
        return {
            "success": True,
            "action": "tool_repaired",
            "durability_restored": self.durability - old_durability,
            "new_durability": self.durability,
            "effectiveness": self.effectiveness,
            "timestamp": datetime.now()
        }
    
    def is_broken(self) -> bool:
        """Check if tool is broken"""
        return self.broken or self.durability <= 0
    
    def _get_condition_description(self) -> str:
        """Get tool condition description"""
        if self.broken:
            return "broken"
        elif self.durability <= 0:
            return "broken"
        elif self.durability < self.max_durability * 0.3:
            return "poor"
        elif self.durability < self.max_durability * 0.7:
            return "fair"
        else:
            return "good"
    
    def _perform_interaction(self, agent, interaction_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle tool interaction"""
        interaction_type = interaction_data.get("interaction_type", "collect")
        
        if interaction_type == "collect":
            return super()._perform_interaction(agent, interaction_data)
        elif interaction_type == "use":
            return self.use()
        elif interaction_type == "repair":
            repair_amount = interaction_data.get("repair_amount")
            return self.repair(repair_amount)
        else:
            return {
                "success": False,
                "reason": "unknown_interaction_type",
                "available_interactions": ["collect", "use", "repair"],
                "timestamp": datetime.now()
            }


class Container(InteractiveObject):
    """Container that can hold other objects"""
    
    def __init__(self, object_id: str, position: Tuple[int, int], room: str,
                 capacity: int = 5):
        super().__init__(object_id, position, room)
        self.object_type = "container"
        self.capacity = capacity
        self.contents: List[str] = []
        self.locked = False
        self.key_required = None
    
    def get_description(self) -> str:
        """Get container description"""
        status = "locked" if self.locked else "unlocked"
        fullness = f"{len(self.contents)}/{self.capacity}"
        return f"A {status} container ({fullness} items)"
    
    def get_visible_properties(self) -> Dict[str, Any]:
        """Get visible container properties"""
        props = super().get_visible_properties()
        props.update({
            "locked": self.locked,
            "capacity": self.capacity,
            "item_count": len(self.contents),
            "contents_visible": not self.locked,
            "contents": self.contents if not self.locked else []
        })
        return props
    
    def _perform_interaction(self, agent, interaction_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle container interaction"""
        if self.locked:
            if self.key_required and agent.has_resource(self.key_required):
                self.locked = False
                return {
                    "success": True,
                    "action": "container_unlocked",
                    "contents": self.contents,
                    "timestamp": datetime.now()
                }
            else:
                return {
                    "success": False,
                    "reason": "container_locked",
                    "key_required": self.key_required,
                    "timestamp": datetime.now()
                }
        
        # Container is unlocked
        interaction_type = interaction_data.get("interaction_type", "examine_contents")
        
        if interaction_type == "examine_contents":
            return {
                "success": True,
                "action": "contents_examined",
                "contents": self.contents,
                "capacity": self.capacity,
                "timestamp": datetime.now()
            }
        elif interaction_type == "take_item":
            item_id = interaction_data.get("item_id")
            return self._take_item(agent, item_id)
        elif interaction_type == "store_item":
            item_id = interaction_data.get("item_id")
            return self._store_item(agent, item_id)
        else:
            return {
                "success": False,
                "reason": "unknown_interaction_type",
                "available_interactions": ["examine_contents", "take_item", "store_item"],
                "timestamp": datetime.now()
            }
    
    def _take_item(self, agent, item_id: str) -> Dict[str, Any]:
        """Take item from container"""
        if item_id not in self.contents:
            return {
                "success": False,
                "reason": "item_not_in_container",
                "timestamp": datetime.now()
            }
        
        self.contents.remove(item_id)
        agent.add_resource(item_id)
        
        return {
            "success": True,
            "action": "item_taken",
            "item_id": item_id,
            "remaining_contents": self.contents,
            "timestamp": datetime.now()
        }
    
    def _store_item(self, agent, item_id: str) -> Dict[str, Any]:
        """Store item in container"""
        if not agent.has_resource(item_id):
            return {
                "success": False,
                "reason": "agent_does_not_have_item",
                "timestamp": datetime.now()
            }
        
        if len(self.contents) >= self.capacity:
            return {
                "success": False,
                "reason": "container_full",
                "capacity": self.capacity,
                "timestamp": datetime.now()
            }
        
        agent.remove_resource(item_id)
        self.contents.append(item_id)
        
        return {
            "success": True,
            "action": "item_stored",
            "item_id": item_id,
            "container_contents": self.contents,
            "timestamp": datetime.now()
        }