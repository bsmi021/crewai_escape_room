"""
High-Performance Spatial Indexing System

This module implements an optimized spatial index for fast agent and object
lookup operations in the Mesa-CrewAI hybrid architecture.

Key Features:
- Grid-based spatial partitioning for O(1) average lookup
- Range queries with configurable radius
- Nearest neighbor search with k-nearest results
- Property-based filtering combined with spatial queries
- Performance monitoring and optimization
- Mesa model integration with automatic synchronization

Architecture:
- Uses spatial hashing with configurable grid cell size
- Maintains object-to-cell mappings for fast updates
- Supports overlapping ranges across multiple cells
- Memory-efficient with lazy evaluation of query results
"""

from typing import Dict, List, Tuple, Set, Optional, Any, Iterator
from dataclasses import dataclass, field
from collections import defaultdict
import time
import math
import mesa
import heapq


@dataclass
class SpatialObject:
    """
    Spatial representation of an object in the simulation
    
    Contains position, type, and arbitrary properties for filtering.
    Designed to be lightweight and fast to create/update.
    """
    object_id: str
    position: Tuple[float, float]
    object_type: str
    properties: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def from_mesa_agent(cls, mesa_agent: mesa.Agent) -> 'SpatialObject':
        """Create SpatialObject from Mesa agent"""
        agent_id = getattr(mesa_agent, 'agent_id', str(mesa_agent.unique_id))
        position = getattr(mesa_agent, 'pos', (0.0, 0.0))
        
        properties = {
            'mesa_id': mesa_agent.unique_id,
            'mesa_type': type(mesa_agent).__name__
        }
        
        # Extract additional properties if available
        if hasattr(mesa_agent, 'role'):
            properties['role'] = mesa_agent.role
        if hasattr(mesa_agent, 'agent_type'):
            properties['agent_type'] = mesa_agent.agent_type
        
        return cls(
            object_id=agent_id,
            position=position,
            object_type="agent",
            properties=properties
        )


@dataclass(frozen=True)
class SpatialRange:
    """
    Specification for spatial range queries
    
    Defines a circular area for spatial searches with center and radius.
    Made frozen to enable hashing for caching.
    """
    center: Tuple[float, float]
    radius: float
    
    def __post_init__(self):
        if self.radius < 0:
            raise ValueError("Radius must be non-negative")
    
    def contains_point(self, point: Tuple[float, float]) -> bool:
        """Check if point is within this range"""
        distance = math.sqrt(
            (point[0] - self.center[0]) ** 2 + (point[1] - self.center[1]) ** 2
        )
        return distance <= self.radius


@dataclass
class SpatialQueryResult:
    """
    Result container for spatial queries
    
    Contains matching objects and performance metrics.
    """
    objects: List[SpatialObject]
    query_time: float
    cells_searched: int = 0
    objects_evaluated: int = 0
    cache_hit: bool = False


class SpatialIndex:
    """
    High-performance spatial index for agent and object lookup
    
    Uses grid-based spatial hashing for O(1) average case lookup performance.
    Optimized for the common case of range queries and nearest neighbor searches.
    
    Performance Characteristics:
    - Insertion: O(1) average case
    - Range query: O(k) where k is result size
    - Nearest neighbor: O(k log k) where k is candidate size
    - Memory: O(n + cells) where n is object count
    """
    
    def __init__(self, grid_size: Tuple[int, int] = (100, 100), cell_size: float = 1.0):
        """
        Initialize spatial index
        
        Args:
            grid_size: (width, height) of the spatial grid
            cell_size: Size of each grid cell in world units
        """
        if grid_size[0] <= 0 or grid_size[1] <= 0:
            raise ValueError("Grid size must be positive")
        if cell_size <= 0:
            raise ValueError("Cell size must be positive")
        
        self.grid_size = grid_size
        self.cell_size = cell_size
        
        # Core data structures
        self._objects: Dict[str, SpatialObject] = {}
        self._grid: Dict[Tuple[int, int], Set[str]] = defaultdict(set)
        self._object_to_cell: Dict[str, Tuple[int, int]] = {}
        
        # Performance tracking
        self._query_cache: Dict[str, Tuple[float, SpatialQueryResult]] = {}
        self._cache_ttl = 1.0  # Cache time-to-live in seconds
    
    @property
    def object_count(self) -> int:
        """Number of objects in the index"""
        return len(self._objects)
    
    @property 
    def is_empty(self) -> bool:
        """Whether the index contains any objects"""
        return len(self._objects) == 0
    
    def add_object(self, obj: SpatialObject) -> bool:
        """
        Add object to spatial index
        
        Args:
            obj: SpatialObject to add
            
        Returns:
            True if added successfully, False if object ID already exists
        """
        if obj.object_id in self._objects:
            return False
        
        # Clamp position to grid bounds
        clamped_position = self._clamp_position(obj.position)
        
        # Create object with clamped position if needed
        if clamped_position != obj.position:
            # Create new object with clamped position (preserve original properties)
            clamped_obj = SpatialObject(
                object_id=obj.object_id,
                position=clamped_position,
                object_type=obj.object_type,
                properties=obj.properties.copy()
            )
            obj = clamped_obj
        
        # Calculate grid cell
        cell = self._position_to_cell(obj.position)
        
        # Add to data structures
        self._objects[obj.object_id] = obj
        self._grid[cell].add(obj.object_id)
        self._object_to_cell[obj.object_id] = cell
        
        # Invalidate cache
        self._invalidate_cache()
        
        return True
    
    def update_object_position(self, object_id: str, new_position: Tuple[float, float]) -> bool:
        """
        Update object position in spatial index
        
        Args:
            object_id: ID of object to update
            new_position: New position as (x, y) tuple
            
        Returns:
            True if updated successfully, False if object not found
        """
        if object_id not in self._objects:
            return False
        
        # Clamp position to grid bounds
        clamped_position = self._clamp_position(new_position)
        
        obj = self._objects[object_id]
        old_cell = self._object_to_cell[object_id]
        new_cell = self._position_to_cell(clamped_position)
        
        # Update object position
        obj.position = clamped_position
        
        # Update grid if cell changed
        if old_cell != new_cell:
            self._grid[old_cell].discard(object_id)
            self._grid[new_cell].add(object_id)
            self._object_to_cell[object_id] = new_cell
            
            # Clean up empty cells
            if not self._grid[old_cell]:
                del self._grid[old_cell]
        
        # Invalidate cache
        self._invalidate_cache()
        
        return True
    
    def remove_object(self, object_id: str) -> bool:
        """
        Remove object from spatial index
        
        Args:
            object_id: ID of object to remove
            
        Returns:
            True if removed successfully, False if object not found
        """
        if object_id not in self._objects:
            return False
        
        # Remove from grid
        cell = self._object_to_cell[object_id]
        self._grid[cell].discard(object_id)
        
        # Clean up empty cell
        if not self._grid[cell]:
            del self._grid[cell]
        
        # Remove from object mappings
        del self._objects[object_id]
        del self._object_to_cell[object_id]
        
        # Invalidate cache
        self._invalidate_cache()
        
        return True
    
    def contains_object(self, object_id: str) -> bool:
        """Check if object exists in index"""
        return object_id in self._objects
    
    def get_object(self, object_id: str) -> Optional[SpatialObject]:
        """Get object by ID"""
        return self._objects.get(object_id)
    
    def find_objects_in_range(self, 
                            spatial_range: SpatialRange,
                            object_type: Optional[str] = None,
                            properties: Optional[Dict[str, Any]] = None) -> SpatialQueryResult:
        """
        Find objects within a spatial range
        
        Args:
            spatial_range: Range specification with center and radius
            object_type: Optional filter by object type
            properties: Optional filter by object properties
            
        Returns:
            SpatialQueryResult with matching objects and metrics
        """
        start_time = time.perf_counter()
        
        # Check cache first
        cache_key = self._create_cache_key("range", spatial_range, object_type, properties)
        cached_result = self._get_cached_result(cache_key)
        if cached_result:
            return cached_result
        
        # Find candidate cells that intersect with range
        candidate_cells = self._get_cells_in_range(spatial_range)
        
        # Collect candidate objects
        candidate_objects = set()
        for cell in candidate_cells:
            if cell in self._grid:
                candidate_objects.update(self._grid[cell])
        
        # Filter objects by actual distance and criteria
        matching_objects = []
        objects_evaluated = 0
        
        for obj_id in candidate_objects:
            obj = self._objects[obj_id]
            objects_evaluated += 1
            
            # Check distance
            if not spatial_range.contains_point(obj.position):
                continue
            
            # Check type filter
            if object_type and obj.object_type != object_type:
                continue
            
            # Check property filters
            if properties and not self._matches_properties(obj, properties):
                continue
            
            matching_objects.append(obj)
        
        query_time = time.perf_counter() - start_time
        
        result = SpatialQueryResult(
            objects=matching_objects,
            query_time=query_time,
            cells_searched=len(candidate_cells),
            objects_evaluated=objects_evaluated
        )
        
        # Cache result
        self._cache_result(cache_key, result)
        
        return result
    
    def find_objects_by_type(self, object_type: str) -> SpatialQueryResult:
        """
        Find all objects of a specific type
        
        Args:
            object_type: Type of objects to find
            
        Returns:
            SpatialQueryResult with matching objects
        """
        start_time = time.perf_counter()
        
        # Check cache first
        cache_key = self._create_cache_key("type", object_type)
        cached_result = self._get_cached_result(cache_key)
        if cached_result:
            return cached_result
        
        # Linear scan for type matching (could be optimized with type index)
        matching_objects = []
        for obj in self._objects.values():
            if obj.object_type == object_type:
                matching_objects.append(obj)
        
        query_time = time.perf_counter() - start_time
        
        result = SpatialQueryResult(
            objects=matching_objects,
            query_time=query_time,
            objects_evaluated=len(self._objects)
        )
        
        # Cache result
        self._cache_result(cache_key, result)
        
        return result
    
    def find_nearest_objects(self, 
                           position: Tuple[float, float], 
                           count: int,
                           object_type: Optional[str] = None) -> SpatialQueryResult:
        """
        Find nearest N objects to a position
        
        Args:
            position: Query position as (x, y) tuple
            count: Maximum number of objects to return
            object_type: Optional filter by object type
            
        Returns:
            SpatialQueryResult with nearest objects sorted by distance
        """
        start_time = time.perf_counter()
        
        # Check cache first
        cache_key = self._create_cache_key("nearest", position, count, object_type)
        cached_result = self._get_cached_result(cache_key)
        if cached_result:
            return cached_result
        
        # Use expanding search radius to find candidates
        candidates = []
        search_radius = self.cell_size
        max_radius = max(self.grid_size) * self.cell_size
        
        while len(candidates) < count * 2 and search_radius <= max_radius:
            range_query = SpatialRange(center=position, radius=search_radius)
            range_result = self.find_objects_in_range(range_query, object_type=object_type)
            
            candidates = range_result.objects
            if len(candidates) >= count:
                break
                
            search_radius *= 2
        
        # Calculate distances and sort
        objects_with_distance = []
        for obj in candidates:
            distance = math.sqrt(
                (obj.position[0] - position[0]) ** 2 + (obj.position[1] - position[1]) ** 2
            )
            objects_with_distance.append((distance, obj))
        
        # Sort by distance and take top N
        objects_with_distance.sort(key=lambda x: x[0])
        nearest_objects = [obj for _, obj in objects_with_distance[:count]]
        
        query_time = time.perf_counter() - start_time
        
        result = SpatialQueryResult(
            objects=nearest_objects,
            query_time=query_time,
            objects_evaluated=len(candidates)
        )
        
        # Cache result
        self._cache_result(cache_key, result)
        
        return result
    
    def find_objects_with_properties(self, properties: Dict[str, Any]) -> SpatialQueryResult:
        """
        Find objects with specific properties
        
        Args:
            properties: Dictionary of property key-value pairs to match
            
        Returns:
            SpatialQueryResult with matching objects
        """
        start_time = time.perf_counter()
        
        # Check cache first
        cache_key = self._create_cache_key("properties", properties)
        cached_result = self._get_cached_result(cache_key)
        if cached_result:
            return cached_result
        
        # Linear scan for property matching
        matching_objects = []
        for obj in self._objects.values():
            if self._matches_properties(obj, properties):
                matching_objects.append(obj)
        
        query_time = time.perf_counter() - start_time
        
        result = SpatialQueryResult(
            objects=matching_objects,
            query_time=query_time,
            objects_evaluated=len(self._objects)
        )
        
        # Cache result
        self._cache_result(cache_key, result)
        
        return result
    
    def build_from_mesa_model(self, mesa_model: mesa.Model) -> None:
        """
        Build spatial index from Mesa model
        
        Args:
            mesa_model: Mesa model to extract objects from
        """
        # Clear existing index
        self.clear()
        
        # Add all agents from Mesa model
        if hasattr(mesa_model, 'schedule') and hasattr(mesa_model.schedule, 'agents'):
            for mesa_agent in mesa_model.schedule.agents:
                spatial_obj = SpatialObject.from_mesa_agent(mesa_agent)
                self.add_object(spatial_obj)
    
    def sync_with_mesa_model(self, mesa_model: mesa.Model) -> None:
        """
        Synchronize index with updated Mesa model
        
        Args:
            mesa_model: Mesa model to sync with
        """
        # Update positions of existing agents
        if hasattr(mesa_model, 'schedule') and hasattr(mesa_model.schedule, 'agents'):
            for mesa_agent in mesa_model.schedule.agents:
                agent_id = getattr(mesa_agent, 'agent_id', str(mesa_agent.unique_id))
                new_position = getattr(mesa_agent, 'pos', (0.0, 0.0))
                
                if self.contains_object(agent_id):
                    self.update_object_position(agent_id, new_position)
                else:
                    # Add new agent
                    spatial_obj = SpatialObject.from_mesa_agent(mesa_agent)
                    self.add_object(spatial_obj)
    
    def clear(self) -> None:
        """Clear all objects from the index"""
        self._objects.clear()
        self._grid.clear()
        self._object_to_cell.clear()
        self._query_cache.clear()
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get index statistics for performance monitoring"""
        return {
            "object_count": self.object_count,
            "cell_count": len(self._grid),
            "cache_size": len(self._query_cache),
            "avg_objects_per_cell": self.object_count / max(1, len(self._grid)),
            "grid_utilization": len(self._grid) / (self.grid_size[0] * self.grid_size[1])
        }
    
    def get_objects_dict(self) -> Dict[str, SpatialObject]:
        """Get internal objects dictionary (for testing only)"""
        return self._objects
    
    # Private methods
    
    def _clamp_position(self, position: Tuple[float, float]) -> Tuple[float, float]:
        """Clamp position to grid boundaries"""
        x, y = position
        max_x = (self.grid_size[0] - 1) * self.cell_size
        max_y = (self.grid_size[1] - 1) * self.cell_size
        
        clamped_x = max(0.0, min(x, max_x))
        clamped_y = max(0.0, min(y, max_y))
        
        return (clamped_x, clamped_y)
    
    def _position_to_cell(self, position: Tuple[float, float]) -> Tuple[int, int]:
        """Convert world position to grid cell coordinates"""
        x, y = position
        cell_x = int(x / self.cell_size)
        cell_y = int(y / self.cell_size)
        
        # Clamp to grid bounds
        cell_x = max(0, min(cell_x, self.grid_size[0] - 1))
        cell_y = max(0, min(cell_y, self.grid_size[1] - 1))
        
        return (cell_x, cell_y)
    
    def _get_cells_in_range(self, spatial_range: SpatialRange) -> List[Tuple[int, int]]:
        """Get all grid cells that intersect with the spatial range"""
        center_x, center_y = spatial_range.center
        radius = spatial_range.radius
        
        # Calculate bounding box of cells
        min_cell_x = max(0, int((center_x - radius) / self.cell_size))
        max_cell_x = min(self.grid_size[0] - 1, int((center_x + radius) / self.cell_size))
        min_cell_y = max(0, int((center_y - radius) / self.cell_size))
        max_cell_y = min(self.grid_size[1] - 1, int((center_y + radius) / self.cell_size))
        
        cells = []
        for cell_x in range(min_cell_x, max_cell_x + 1):
            for cell_y in range(min_cell_y, max_cell_y + 1):
                cells.append((cell_x, cell_y))
        
        return cells
    
    def _matches_properties(self, obj: SpatialObject, properties: Dict[str, Any]) -> bool:
        """Check if object matches all specified properties"""
        for key, value in properties.items():
            if key not in obj.properties or obj.properties[key] != value:
                return False
        return True
    
    def _create_cache_key(self, query_type: str, *args) -> str:
        """Create cache key for query"""
        # Convert args to hashable format
        hashable_args = []
        for arg in args:
            if isinstance(arg, dict):
                # Convert dict to sorted tuple of items
                hashable_args.append(tuple(sorted(arg.items())))
            elif arg is None:
                hashable_args.append(None)
            else:
                hashable_args.append(arg)
        
        return f"{query_type}:{hash(tuple(hashable_args))}"
    
    def _get_cached_result(self, cache_key: str) -> Optional[SpatialQueryResult]:
        """Get cached query result if still valid"""
        if cache_key not in self._query_cache:
            return None
        
        cache_time, result = self._query_cache[cache_key]
        
        # Check if cache is still valid
        if time.perf_counter() - cache_time > self._cache_ttl:
            del self._query_cache[cache_key]
            return None
        
        # Mark as cache hit
        result.cache_hit = True
        return result
    
    def _cache_result(self, cache_key: str, result: SpatialQueryResult) -> None:
        """Cache query result"""
        self._query_cache[cache_key] = (time.perf_counter(), result)
        
        # Limit cache size to prevent memory issues
        if len(self._query_cache) > 100:
            # Remove oldest entry
            oldest_key = min(self._query_cache.keys(), 
                           key=lambda k: self._query_cache[k][0])
            del self._query_cache[oldest_key]
    
    def _invalidate_cache(self) -> None:
        """Invalidate all cached query results"""
        self._query_cache.clear()