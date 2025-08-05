"""
Collision Detection System

Agent D: State Management & Integration Specialist
Implements spatial constraints and collision detection for the Mesa environment.
"""

import math
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
import numpy as np

from .room_objects import RoomObject


class CollisionType(Enum):
    """Types of collisions"""
    AGENT_AGENT = "agent_agent"
    AGENT_OBJECT = "agent_object"
    AGENT_WALL = "agent_wall"
    AGENT_BOUNDARY = "agent_boundary"
    OBJECT_OBJECT = "object_object"


@dataclass
class CollisionInfo:
    """Information about a collision"""
    collision_type: str
    entity1_id: str
    entity2_id: str
    position: Tuple[int, int]
    severity: float = 1.0
    preventable: bool = True
    resolution_suggestions: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SpatialEntity:
    """Represents an entity in spatial system"""
    entity_id: str
    entity_type: str  # "agent", "object", "wall", "boundary"
    position: Tuple[int, int]
    size: float = 1.0
    solid: bool = True
    movable: bool = False
    collision_layer: int = 0  # For collision filtering


class CollisionDetector:
    """
    Collision detection system for the Mesa environment
    
    Handles spatial constraints, collision detection, and resolution suggestions.
    """
    
    def __init__(self, grid_width: int = 10, grid_height: int = 10):
        self.grid_width = grid_width
        self.grid_height = grid_height
        
        # Spatial indexing for performance
        self.spatial_grid: Dict[Tuple[int, int], List[SpatialEntity]] = {}
        self.entities: Dict[str, SpatialEntity] = {}
        
        # Collision history
        self.collision_history: List[CollisionInfo] = []
        self.collision_count = 0
        
        # Configuration
        self.enable_agent_collision = True
        self.enable_object_collision = True
        self.collision_tolerance = 0.1  # Distance tolerance for collisions
        
        self._initialize_spatial_grid()
    
    def _initialize_spatial_grid(self):
        """Initialize spatial grid for fast lookups"""
        for x in range(self.grid_width):
            for y in range(self.grid_height):
                self.spatial_grid[(x, y)] = []
    
    def register_entity(self, entity: SpatialEntity):
        """Register entity in collision system"""
        self.entities[entity.entity_id] = entity
        self._add_to_spatial_grid(entity)
    
    def unregister_entity(self, entity_id: str):
        """Unregister entity from collision system"""
        if entity_id in self.entities:
            entity = self.entities[entity_id]
            self._remove_from_spatial_grid(entity)
            del self.entities[entity_id]
    
    def update_entity_position(self, entity_id: str, new_position: Tuple[int, int]):
        """Update entity position in collision system"""
        if entity_id in self.entities:
            entity = self.entities[entity_id]
            
            # Remove from old position
            self._remove_from_spatial_grid(entity)
            
            # Update position
            entity.position = new_position
            
            # Add to new position
            self._add_to_spatial_grid(entity)
    
    def check_position_valid(self, position: Tuple[int, int], 
                           entity_id: str = None, entity_size: float = 1.0) -> Tuple[bool, str]:
        """Check if position is valid (no collisions)"""
        x, y = position
        
        # Check bounds
        if not (0 <= x < self.grid_width and 0 <= y < self.grid_height):
            return False, "out_of_bounds"
        
        # Check for collisions with existing entities
        collision_info = self._check_collisions_at_position(position, entity_id, entity_size)
        
        if collision_info:
            return False, f"collision_{collision_info.collision_type}"
        
        return True, "valid"
    
    def check_agent_collision(self, agents: List) -> bool:
        """Check for agent-to-agent collisions"""
        if not self.enable_agent_collision:
            return False
        
        agent_positions = {}
        for agent in agents:
            pos = getattr(agent, 'pos', (0, 0))
            agent_id = str(getattr(agent, 'unique_id', id(agent)))
            
            if pos in agent_positions:
                # Collision detected
                other_agent_id = agent_positions[pos]
                collision = CollisionInfo(
                    collision_type=CollisionType.AGENT_AGENT.value,
                    entity1_id=agent_id,
                    entity2_id=other_agent_id,
                    position=pos,
                    resolution_suggestions=["move_to_adjacent_cell", "wait_for_other_agent"]
                )
                
                self._record_collision(collision)
                return True
            
            agent_positions[pos] = agent_id
        
        return False
    
    def check_object_collision(self, position: Tuple[int, int], 
                             objects: List[RoomObject]) -> bool:
        """Check for collision with room objects"""
        if not self.enable_object_collision:
            return False
        
        for obj in objects:
            if hasattr(obj, 'solid') and obj.solid and obj.position == position:
                collision = CollisionInfo(
                    collision_type=CollisionType.AGENT_OBJECT.value,
                    entity1_id="agent",
                    entity2_id=obj.object_id,
                    position=position,
                    resolution_suggestions=["find_alternate_path", "interact_with_object"]
                )
                
                self._record_collision(collision)
                return True
        
        return False
    
    def check_movement_path(self, start_pos: Tuple[int, int], 
                          end_pos: Tuple[int, int],
                          entity_id: str = None) -> Tuple[bool, List[Tuple[int, int]]]:
        """Check if movement path is clear"""
        path = self._calculate_path(start_pos, end_pos)
        clear_path = []
        
        for position in path:
            is_valid, reason = self.check_position_valid(position, entity_id)
            if is_valid:
                clear_path.append(position)
            else:
                # Path blocked
                return False, clear_path
        
        return True, clear_path
    
    def find_nearest_valid_position(self, target_pos: Tuple[int, int],
                                  entity_id: str = None,
                                  max_search_radius: int = 5) -> Optional[Tuple[int, int]]:
        """Find nearest valid position to target"""
        x, y = target_pos
        
        # First check if target position is already valid
        is_valid, _ = self.check_position_valid(target_pos, entity_id)
        if is_valid:
            return target_pos
        
        # Search in expanding radius
        for radius in range(1, max_search_radius + 1):
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    if abs(dx) == radius or abs(dy) == radius:  # Only check perimeter
                        candidate_pos = (x + dx, y + dy)
                        
                        is_valid, _ = self.check_position_valid(candidate_pos, entity_id)
                        if is_valid:
                            return candidate_pos
        
        return None
    
    def get_collision_avoidance_suggestions(self, 
                                          position: Tuple[int, int],
                                          entity_id: str = None) -> List[str]:
        """Get suggestions for avoiding collisions at position"""
        suggestions = []
        
        # Check what's blocking the position
        collision_info = self._check_collisions_at_position(position, entity_id)
        
        if collision_info:
            if collision_info.collision_type == CollisionType.AGENT_AGENT.value:
                suggestions.extend([
                    "wait_for_other_agent_to_move",
                    "communicate_movement_intention",
                    "find_alternate_route"
                ])
            
            elif collision_info.collision_type == CollisionType.AGENT_OBJECT.value:
                suggestions.extend([
                    "interact_with_blocking_object",
                    "find_alternate_path",
                    "use_tool_to_remove_obstacle"
                ])
            
            elif collision_info.collision_type == CollisionType.AGENT_WALL.value:
                suggestions.extend([
                    "find_door_or_opening",
                    "use_tool_to_break_wall",
                    "find_alternate_route"
                ])
        
        # Add general suggestions
        suggestions.extend([
            "move_to_adjacent_cell",
            "wait_one_turn",
            "coordinate_with_other_agents"
        ])
        
        return list(set(suggestions))  # Remove duplicates
    
    def calculate_safe_movement_options(self, current_pos: Tuple[int, int],
                                      entity_id: str = None) -> List[Tuple[int, int]]:
        """Calculate safe movement options from current position"""
        x, y = current_pos
        safe_positions = []
        
        # Check all adjacent positions
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue  # Skip current position
                
                candidate_pos = (x + dx, y + dy)
                is_valid, _ = self.check_position_valid(candidate_pos, entity_id)
                
                if is_valid:
                    safe_positions.append(candidate_pos)
        
        return safe_positions
    
    def get_spatial_density(self, position: Tuple[int, int], radius: int = 2) -> float:
        """Get spatial density around position"""
        x, y = position
        entity_count = 0
        total_cells = 0
        
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                check_pos = (x + dx, y + dy)
                total_cells += 1
                
                if check_pos in self.spatial_grid:
                    entity_count += len(self.spatial_grid[check_pos])
        
        return entity_count / total_cells if total_cells > 0 else 0.0
    
    def detect_bottlenecks(self) -> List[Dict[str, Any]]:
        """Detect spatial bottlenecks in the environment"""
        bottlenecks = []
        
        # Analyze spatial grid for high-density areas
        for position, entities in self.spatial_grid.items():
            if len(entities) > 1:  # Multiple entities at same position
                density = self.get_spatial_density(position, 1)
                
                if density > 0.5:  # High density threshold
                    bottlenecks.append({
                        "position": position,
                        "entity_count": len(entities),
                        "density": density,
                        "affected_entities": [e.entity_id for e in entities]
                    })
        
        return bottlenecks
    
    def get_collision_statistics(self) -> Dict[str, Any]:
        """Get collision system statistics"""
        collision_by_type = {}
        for collision in self.collision_history:
            ctype = collision.collision_type
            if ctype not in collision_by_type:
                collision_by_type[ctype] = 0
            collision_by_type[ctype] += 1
        
        return {
            "total_collisions": len(self.collision_history),
            "collisions_by_type": collision_by_type,
            "collision_rate": len(self.collision_history) / max(1, self.collision_count),
            "registered_entities": len(self.entities),
            "spatial_grid_utilization": self._calculate_grid_utilization(),
            "recent_collisions": self.collision_history[-10:] if self.collision_history else []
        }
    
    # Private helper methods
    
    def _add_to_spatial_grid(self, entity: SpatialEntity):
        """Add entity to spatial grid"""
        pos = entity.position
        if pos in self.spatial_grid:
            self.spatial_grid[pos].append(entity)
        else:
            self.spatial_grid[pos] = [entity]
    
    def _remove_from_spatial_grid(self, entity: SpatialEntity):
        """Remove entity from spatial grid"""
        pos = entity.position
        if pos in self.spatial_grid and entity in self.spatial_grid[pos]:
            self.spatial_grid[pos].remove(entity)
    
    def _check_collisions_at_position(self, position: Tuple[int, int],
                                    entity_id: str = None,
                                    entity_size: float = 1.0) -> Optional[CollisionInfo]:
        """Check for collisions at specific position"""
        if position not in self.spatial_grid:
            return None
        
        entities_at_position = self.spatial_grid[position]
        
        for entity in entities_at_position:
            # Skip self
            if entity_id and entity.entity_id == entity_id:
                continue
            
            # Check if collision would occur
            if entity.solid:
                collision_type = self._determine_collision_type(entity)
                
                return CollisionInfo(
                    collision_type=collision_type,
                    entity1_id=entity_id or "unknown",
                    entity2_id=entity.entity_id,
                    position=position,
                    severity=self._calculate_collision_severity(entity, entity_size)
                )
        
        return None
    
    def _determine_collision_type(self, entity: SpatialEntity) -> str:
        """Determine collision type based on entity"""
        if entity.entity_type == "agent":
            return CollisionType.AGENT_AGENT.value
        elif entity.entity_type == "object":
            return CollisionType.AGENT_OBJECT.value
        elif entity.entity_type == "wall":
            return CollisionType.AGENT_WALL.value
        else:
            return CollisionType.AGENT_BOUNDARY.value
    
    def _calculate_collision_severity(self, entity: SpatialEntity, 
                                    moving_entity_size: float) -> float:
        """Calculate collision severity"""
        size_factor = min(entity.size, moving_entity_size) / max(entity.size, moving_entity_size)
        
        if entity.movable:
            return 0.5 * size_factor  # Less severe if entity can be moved
        else:
            return 1.0 * size_factor  # More severe if entity is immovable
    
    def _calculate_path(self, start_pos: Tuple[int, int], 
                       end_pos: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Calculate simple path between positions"""
        # Simple line interpolation - could be enhanced with A*
        x1, y1 = start_pos
        x2, y2 = end_pos
        
        path = []
        
        # Calculate steps
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        steps = max(dx, dy)
        
        if steps == 0:
            return [start_pos]
        
        # Interpolate path
        for i in range(steps + 1):
            t = i / steps
            x = int(x1 + t * (x2 - x1))
            y = int(y1 + t * (y2 - y1))
            path.append((x, y))
        
        return path
    
    def _record_collision(self, collision: CollisionInfo):
        """Record collision in history"""
        self.collision_history.append(collision)
        self.collision_count += 1
        
        # Keep limited history
        if len(self.collision_history) > 100:
            self.collision_history = self.collision_history[-100:]
    
    def _calculate_grid_utilization(self) -> float:
        """Calculate spatial grid utilization"""
        occupied_cells = sum(1 for entities in self.spatial_grid.values() if entities)
        total_cells = len(self.spatial_grid)
        
        return occupied_cells / total_cells if total_cells > 0 else 0.0


class PathFinder:
    """
    Path finding system for collision-aware navigation
    """
    
    def __init__(self, collision_detector: CollisionDetector):
        self.collision_detector = collision_detector
    
    def find_path(self, start: Tuple[int, int], goal: Tuple[int, int],
                  entity_id: str = None) -> Optional[List[Tuple[int, int]]]:
        """Find collision-free path using A* algorithm"""
        # Simple A* implementation
        from heapq import heappush, heappop
        
        def heuristic(pos1, pos2):
            return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
        
        def get_neighbors(pos):
            x, y = pos
            neighbors = []
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                neighbor = (x + dx, y + dy)
                is_valid, _ = self.collision_detector.check_position_valid(neighbor, entity_id)
                if is_valid:
                    neighbors.append(neighbor)
            return neighbors
        
        # A* algorithm
        open_set = [(0, start)]
        came_from = {}
        g_score = {start: 0}
        f_score = {start: heuristic(start, goal)}
        
        while open_set:
            current = heappop(open_set)[1]
            
            if current == goal:
                # Reconstruct path
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                return path[::-1]
            
            for neighbor in get_neighbors(current):
                tentative_g_score = g_score[current] + 1
                
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                    heappush(open_set, (f_score[neighbor], neighbor))
        
        return None  # No path found
    
    def find_safe_path(self, start: Tuple[int, int], goal: Tuple[int, int],
                      entity_id: str = None, avoid_density: bool = True) -> Optional[List[Tuple[int, int]]]:
        """Find path that avoids high-density areas"""
        path = self.find_path(start, goal, entity_id)
        
        if not path or not avoid_density:
            return path
        
        # Check path for high-density areas and find alternatives
        safe_path = []
        for i, position in enumerate(path):
            density = self.collision_detector.get_spatial_density(position)
            
            if density > 0.5:  # High density threshold
                # Try to find alternate route for this segment
                segment_start = path[i-1] if i > 0 else start
                segment_end = path[i+1] if i < len(path)-1 else goal
                
                alternate = self._find_low_density_route(segment_start, segment_end, entity_id)
                if alternate:
                    safe_path.extend(alternate)
                else:
                    safe_path.append(position)  # Keep original if no alternative
            else:
                safe_path.append(position)
        
        return safe_path
    
    def _find_low_density_route(self, start: Tuple[int, int], end: Tuple[int, int],
                              entity_id: str = None) -> Optional[List[Tuple[int, int]]]:
        """Find route through low-density areas"""
        # Simple implementation - could be enhanced
        x1, y1 = start
        x2, y2 = end
        
        # Try going around high-density area
        mid_options = [
            (x1, y2),  # Go vertical first
            (x2, y1)   # Go horizontal first
        ]
        
        for midpoint in mid_options:
            path1 = self.find_path(start, midpoint, entity_id)
            path2 = self.find_path(midpoint, end, entity_id)
            
            if path1 and path2:
                # Check if this route has lower average density
                combined_path = path1 + path2[1:]  # Avoid duplicate midpoint
                avg_density = sum(self.collision_detector.get_spatial_density(pos) 
                                for pos in combined_path) / len(combined_path)
                
                if avg_density < 0.3:  # Low density threshold
                    return combined_path
        
        return None