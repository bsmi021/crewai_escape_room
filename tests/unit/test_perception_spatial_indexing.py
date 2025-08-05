"""
Test Suite for Spatial Indexing System

This test suite validates the spatial indexing system that optimizes agent and object
lookup operations for the Mesa-CrewAI hybrid perception pipeline.

Test Categories:
- Spatial index creation and updates
- Fast agent/object lookup by position
- Range queries and proximity searches
- Performance benchmarks (<100ms target)
- Memory efficiency validation
"""

import pytest
from typing import Dict, List, Tuple, Set
from unittest.mock import Mock, MagicMock
import time
import mesa
from datetime import datetime

from src.escape_room_sim.hybrid.spatial.spatial_index import (
    SpatialIndex, SpatialObject, SpatialQueryResult, SpatialRange
)


class TestSpatialIndexCreation:
    """Test spatial index creation and initialization"""
    
    def test_spatial_index_initialization(self):
        """Test spatial index creates with default parameters"""
        index = SpatialIndex()
        
        assert index.grid_size == (100, 100)  # Default grid size
        assert index.cell_size == 1.0  # Default cell resolution
        assert index.object_count == 0
        assert index.is_empty == True
    
    def test_spatial_index_custom_parameters(self):
        """Test spatial index with custom grid dimensions"""
        index = SpatialIndex(grid_size=(50, 30), cell_size=2.0)
        
        assert index.grid_size == (50, 30)
        assert index.cell_size == 2.0
        assert index.object_count == 0
    
    def test_spatial_index_invalid_parameters(self):
        """Test spatial index rejects invalid parameters"""
        with pytest.raises(ValueError, match="Grid size must be positive"):
            SpatialIndex(grid_size=(0, 10))
        
        with pytest.raises(ValueError, match="Cell size must be positive"):
            SpatialIndex(cell_size=0.0)


class TestSpatialObjectManagement:
    """Test adding, updating, and removing spatial objects"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.index = SpatialIndex(grid_size=(10, 10))
        self.agent_1 = SpatialObject(
            object_id="agent_1",
            position=(2.5, 3.7),
            object_type="agent",
            properties={"role": "strategist"}
        )
        self.agent_2 = SpatialObject(
            object_id="agent_2", 
            position=(7.1, 8.2),
            object_type="agent",
            properties={"role": "mediator"}
        )
        self.resource = SpatialObject(
            object_id="key_1",
            position=(5.0, 5.0),
            object_type="resource",
            properties={"resource_type": "key"}
        )
    
    def test_add_single_object(self):
        """Test adding a single object to spatial index"""
        result = self.index.add_object(self.agent_1)
        
        assert result == True
        assert self.index.object_count == 1
        assert not self.index.is_empty
        assert self.index.contains_object("agent_1")
    
    def test_add_multiple_objects(self):
        """Test adding multiple objects to spatial index"""
        self.index.add_object(self.agent_1)
        self.index.add_object(self.agent_2)
        self.index.add_object(self.resource)
        
        assert self.index.object_count == 3
        assert self.index.contains_object("agent_1")
        assert self.index.contains_object("agent_2") 
        assert self.index.contains_object("key_1")
    
    def test_add_duplicate_object_id(self):
        """Test adding object with duplicate ID fails"""
        self.index.add_object(self.agent_1)
        
        duplicate_agent = SpatialObject(
            object_id="agent_1",  # Same ID
            position=(9.0, 9.0),
            object_type="agent",
            properties={}
        )
        
        result = self.index.add_object(duplicate_agent)
        assert result == False
        assert self.index.object_count == 1  # No change
    
    def test_update_object_position(self):
        """Test updating object position in spatial index"""
        self.index.add_object(self.agent_1)
        
        # Update position
        updated_result = self.index.update_object_position("agent_1", (8.0, 1.5))
        
        assert updated_result == True
        retrieved_object = self.index.get_object("agent_1")
        assert retrieved_object.position == (8.0, 1.5)
    
    def test_update_nonexistent_object(self):
        """Test updating position of nonexistent object fails"""
        result = self.index.update_object_position("nonexistent", (5.0, 5.0))
        assert result == False
    
    def test_remove_object(self):
        """Test removing object from spatial index"""
        self.index.add_object(self.agent_1)
        self.index.add_object(self.agent_2)
        
        result = self.index.remove_object("agent_1")
        
        assert result == True
        assert self.index.object_count == 1
        assert not self.index.contains_object("agent_1")
        assert self.index.contains_object("agent_2")
    
    def test_remove_nonexistent_object(self):
        """Test removing nonexistent object fails gracefully"""
        result = self.index.remove_object("nonexistent")
        assert result == False


class TestSpatialQueries:
    """Test spatial query operations"""
    
    def setup_method(self):
        """Set up test fixtures with multiple objects"""
        self.index = SpatialIndex(grid_size=(20, 20))
        
        # Add agents in a cluster
        self.agents = [
            SpatialObject("agent_1", (5.0, 5.0), "agent", {"role": "strategist"}),
            SpatialObject("agent_2", (5.5, 5.5), "agent", {"role": "mediator"}),
            SpatialObject("agent_3", (6.0, 6.0), "agent", {"role": "survivor"}),
        ]
        
        # Add resources scattered around
        self.resources = [
            SpatialObject("key_1", (3.0, 7.0), "resource", {"type": "key"}),
            SpatialObject("tool_1", (8.0, 4.0), "resource", {"type": "tool"}),
            SpatialObject("chest_1", (15.0, 15.0), "resource", {"type": "chest"}),
        ]
        
        for obj in self.agents + self.resources:
            self.index.add_object(obj)
    
    def test_find_objects_in_range(self):
        """Test finding objects within a specific range"""
        center = (5.5, 5.5)
        range_query = SpatialRange(center=center, radius=1.0)
        
        result = self.index.find_objects_in_range(range_query)
        
        assert isinstance(result, SpatialQueryResult)
        assert len(result.objects) >= 2  # At least agent_1 and agent_2
        
        # Verify all returned objects are within range
        for obj in result.objects:
            distance = self._calculate_distance(center, obj.position)
            assert distance <= 1.0
    
    def test_find_objects_by_type(self):
        """Test finding objects filtered by type"""
        agents_result = self.index.find_objects_by_type("agent")
        resources_result = self.index.find_objects_by_type("resource")
        
        assert len(agents_result.objects) == 3
        assert len(resources_result.objects) == 3
        
        # Verify type filtering
        for obj in agents_result.objects:
            assert obj.object_type == "agent"
        
        for obj in resources_result.objects:
            assert obj.object_type == "resource"
    
    def test_find_nearest_objects(self):
        """Test finding nearest N objects to a position"""
        query_position = (5.0, 5.0)
        
        result = self.index.find_nearest_objects(query_position, count=3)
        
        assert len(result.objects) == 3
        
        # Verify objects are sorted by distance (nearest first)
        distances = [
            self._calculate_distance(query_position, obj.position)
            for obj in result.objects
        ]
        assert distances == sorted(distances)
    
    def test_find_objects_with_properties(self):
        """Test finding objects with specific properties"""
        # Find strategist agents
        result = self.index.find_objects_with_properties({"role": "strategist"})
        
        assert len(result.objects) == 1
        assert result.objects[0].object_id == "agent_1"
        
        # Find key resources
        result = self.index.find_objects_with_properties({"type": "key"})
        
        assert len(result.objects) == 1
        assert result.objects[0].object_id == "key_1"
    
    def test_complex_spatial_query(self):
        """Test complex query combining multiple criteria"""
        # Find agents within range that have specific role
        center = (5.5, 5.5)
        range_query = SpatialRange(center=center, radius=2.0)
        
        result = self.index.find_objects_in_range(
            range_query,
            object_type="agent",
            properties={"role": "mediator"}
        )
        
        assert len(result.objects) == 1
        assert result.objects[0].object_id == "agent_2"
    
    def test_empty_query_results(self):
        """Test queries that return no results"""
        # Query empty area
        empty_range = SpatialRange(center=(0.0, 0.0), radius=1.0)
        result = self.index.find_objects_in_range(empty_range)
        
        assert len(result.objects) == 0
        assert result.query_time > 0  # Should still track timing
        
        # Query nonexistent type
        result = self.index.find_objects_by_type("nonexistent")
        assert len(result.objects) == 0
    
    def _calculate_distance(self, pos1: Tuple[float, float], pos2: Tuple[float, float]) -> float:
        """Helper method to calculate Euclidean distance"""
        return ((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2) ** 0.5


class TestSpatialIndexPerformance:
    """Test spatial index performance requirements"""
    
    def setup_method(self):
        """Set up performance test environment"""
        self.index = SpatialIndex(grid_size=(100, 100))
        
        # Add many objects for performance testing
        self.object_count = 1000
        import random
        random.seed(42)  # Deterministic for testing
        
        for i in range(self.object_count):
            # Distribute objects more evenly across the grid
            x = random.uniform(0, 99)
            y = random.uniform(0, 99)
            obj = SpatialObject(
                object_id=f"obj_{i}",
                position=(x, y),
                object_type="agent" if i % 3 == 0 else "resource",
                properties={"index": i}
            )
            self.index.add_object(obj)
    
    def test_insertion_performance(self):
        """Test object insertion performance (<100ms for 1000 objects)"""
        new_index = SpatialIndex()
        
        start_time = time.perf_counter()
        
        for i in range(1000):
            obj = SpatialObject(
                object_id=f"perf_obj_{i}",
                position=(i % 50, i // 50),
                object_type="test",
                properties={}
            )
            new_index.add_object(obj)
        
        elapsed_time = (time.perf_counter() - start_time) * 1000  # Convert to ms
        
        assert elapsed_time < 100  # Must be under 100ms
        assert new_index.object_count == 1000
    
    def test_range_query_performance(self):
        """Test range query performance (<100ms target)"""
        center = (50.0, 50.0)
        range_query = SpatialRange(center=center, radius=25.0)
        
        start_time = time.perf_counter()
        
        result = self.index.find_objects_in_range(range_query)
        
        elapsed_time = (time.perf_counter() - start_time) * 1000  # Convert to ms
        
        assert elapsed_time < 100  # Must be under 100ms
        assert len(result.objects) > 0  # Should find objects in large range
        
        # Verify query time is recorded
        assert result.query_time > 0
        assert result.query_time < 0.1  # Query time should be under 100ms
    
    def test_nearest_neighbor_performance(self):
        """Test nearest neighbor query performance"""
        query_position = (25.0, 75.0)
        
        start_time = time.perf_counter()
        
        result = self.index.find_nearest_objects(query_position, count=10)
        
        elapsed_time = (time.perf_counter() - start_time) * 1000  # Convert to ms
        
        assert elapsed_time < 100  # Must be under 100ms
        assert len(result.objects) == 10
    
    def test_update_performance(self):
        """Test object position update performance"""
        object_ids = [f"obj_{i}" for i in range(0, 100, 10)]  # Sample of objects
        
        start_time = time.perf_counter()
        
        for i, obj_id in enumerate(object_ids):
            new_position = (90 - i * 5, 90 - i * 5)
            self.index.update_object_position(obj_id, new_position)
        
        elapsed_time = (time.perf_counter() - start_time) * 1000  # Convert to ms
        
        assert elapsed_time < 100  # Must be under 100ms
    
    def test_memory_efficiency(self):
        """Test spatial index memory usage is reasonable"""
        import sys
        
        # Get approximate memory usage
        index_size = sys.getsizeof(self.index)
        objects_dict = self.index.get_objects_dict()
        objects_size = sum(sys.getsizeof(obj) for obj in objects_dict.values())
        
        total_memory = index_size + objects_size
        memory_per_object = total_memory / self.object_count
        
        # Memory per object should be reasonable (< 1KB per object)
        assert memory_per_object < 1024
        
        # Total memory should be reasonable (< 10MB for 1000 objects)
        assert total_memory < 10 * 1024 * 1024


class TestSpatialIndexIntegration:
    """Integration tests with Mesa model objects"""
    
    def setup_method(self):
        """Set up integration test environment"""
        self.index = SpatialIndex()
        
        # Create mock Mesa agents
        self.mesa_agent_1 = Mock(spec=mesa.Agent)
        self.mesa_agent_1.unique_id = 1
        self.mesa_agent_1.pos = (3.0, 4.0)
        self.mesa_agent_1.agent_id = "strategist"
        
        self.mesa_agent_2 = Mock(spec=mesa.Agent)
        self.mesa_agent_2.unique_id = 2
        self.mesa_agent_2.pos = (7.0, 8.0)
        self.mesa_agent_2.agent_id = "mediator"
    
    def test_index_from_mesa_model(self):
        """Test building spatial index from Mesa model"""
        # Create mock Mesa model
        mesa_model = Mock(spec=mesa.Model)
        mesa_model.schedule = Mock()
        mesa_model.schedule.agents = [self.mesa_agent_1, self.mesa_agent_2]
        
        # Build index from Mesa model
        self.index.build_from_mesa_model(mesa_model)
        
        assert self.index.object_count == 2
        assert self.index.contains_object("strategist")
        assert self.index.contains_object("mediator")
        
        # Verify positions match
        strategist_obj = self.index.get_object("strategist")
        assert strategist_obj.position == (3.0, 4.0)
    
    def test_sync_with_mesa_model(self):
        """Test synchronizing index with updated Mesa model"""
        mesa_model = Mock(spec=mesa.Model)
        mesa_model.schedule = Mock()
        mesa_model.schedule.agents = [self.mesa_agent_1, self.mesa_agent_2]
        
        # Initial build
        self.index.build_from_mesa_model(mesa_model)
        
        # Update agent position in Mesa
        self.mesa_agent_1.pos = (5.0, 6.0)
        
        # Sync index
        self.index.sync_with_mesa_model(mesa_model)
        
        # Verify position updated
        strategist_obj = self.index.get_object("strategist")
        assert strategist_obj.position == (5.0, 6.0)
    
    def test_spatial_object_from_mesa_agent(self):
        """Test creating SpatialObject from Mesa agent"""
        spatial_obj = SpatialObject.from_mesa_agent(self.mesa_agent_1)
        
        assert spatial_obj.object_id == "strategist"
        assert spatial_obj.position == (3.0, 4.0)
        assert spatial_obj.object_type == "agent"
        assert spatial_obj.properties["mesa_id"] == 1


class TestSpatialIndexEdgeCases:
    """Test edge cases and error conditions"""
    
    def setup_method(self):
        """Set up edge case testing"""
        self.index = SpatialIndex()
    
    def test_objects_at_boundary_positions(self):
        """Test objects at grid boundaries"""
        # Objects at edges
        boundary_objects = [
            SpatialObject("edge_1", (0.0, 0.0), "test", {}),
            SpatialObject("edge_2", (99.9, 99.9), "test", {}),
            SpatialObject("edge_3", (0.0, 99.9), "test", {}),
            SpatialObject("edge_4", (99.9, 0.0), "test", {}),
        ]
        
        for obj in boundary_objects:
            result = self.index.add_object(obj)
            assert result == True
        
        assert self.index.object_count == 4
    
    def test_objects_outside_grid(self):
        """Test handling objects outside grid boundaries"""
        outside_obj = SpatialObject("outside", (150.0, 150.0), "test", {})
        
        # Should handle gracefully - either reject or clamp to bounds
        result = self.index.add_object(outside_obj)
        
        if result:
            # If accepted, position should be clamped
            retrieved = self.index.get_object("outside")
            assert retrieved.position[0] <= 100.0
            assert retrieved.position[1] <= 100.0
        else:
            # If rejected, that's also acceptable behavior
            assert not self.index.contains_object("outside")
    
    def test_many_objects_same_position(self):
        """Test multiple objects at same position"""
        same_position = (5.0, 5.0)
        
        objects = [
            SpatialObject(f"obj_{i}", same_position, "test", {"index": i})
            for i in range(10)
        ]
        
        for obj in objects:
            self.index.add_object(obj)
        
        assert self.index.object_count == 10
        
        # Query should return all objects at that position
        range_query = SpatialRange(center=same_position, radius=0.1)
        result = self.index.find_objects_in_range(range_query)
        
        assert len(result.objects) == 10
    
    def test_zero_radius_range_query(self):
        """Test range query with zero radius"""
        obj = SpatialObject("test", (5.0, 5.0), "test", {})
        self.index.add_object(obj)
        
        # Zero radius should find exact position matches
        range_query = SpatialRange(center=(5.0, 5.0), radius=0.0)
        result = self.index.find_objects_in_range(range_query)
        
        assert len(result.objects) == 1
        assert result.objects[0].object_id == "test"
    
    def test_large_radius_range_query(self):
        """Test range query with very large radius"""
        # Add a few objects
        objects = [
            SpatialObject(f"obj_{i}", (i * 10, i * 10), "test", {})
            for i in range(5)
        ]
        
        for obj in objects:
            self.index.add_object(obj)
        
        # Large radius should find all objects
        range_query = SpatialRange(center=(25.0, 25.0), radius=1000.0)
        result = self.index.find_objects_in_range(range_query)
        
        assert len(result.objects) == 5