"""
Test Suite for Integrated Perception Pipeline

This test suite validates the integration of spatial indexing and caching
systems into a high-performance perception pipeline for the Mesa-CrewAI
hybrid architecture.

Test Categories:
- Pipeline initialization and configuration
- Integration with spatial indexing system
- Integration with perception caching system
- Performance optimization with combined systems
- Memory-based perception filtering
- Error handling and edge cases
"""

import pytest
from typing import Dict, List, Any, Optional
from unittest.mock import Mock, MagicMock
import time
from datetime import datetime
import mesa

from src.escape_room_sim.hybrid.perception.integrated_pipeline import (
    IntegratedPerceptionPipeline, PerceptionPipelineConfig, PerformanceMetrics
)
from src.escape_room_sim.hybrid.core_architecture import PerceptionData
from src.escape_room_sim.hybrid.spatial.spatial_index import SpatialIndex, SpatialObject
from src.escape_room_sim.hybrid.perception.perception_cache import PerceptionCache


class TestIntegratedPipelineInitialization:
    """Test integrated pipeline initialization and configuration"""
    
    def test_pipeline_default_initialization(self):
        """Test pipeline creates with default configuration"""
        pipeline = IntegratedPerceptionPipeline()
        
        assert pipeline.config is not None
        assert pipeline.spatial_index is not None
        assert pipeline.perception_cache is not None
        assert pipeline.performance_metrics is not None
        assert pipeline.is_initialized == True
    
    def test_pipeline_custom_configuration(self):
        """Test pipeline with custom configuration"""
        config = PerceptionPipelineConfig(
            cache_size=500,
            cache_ttl=10.0,
            spatial_grid_size=(50, 50),
            enable_memory_filtering=False,
            perception_range=7.0,
            performance_monitoring=True
        )
        
        pipeline = IntegratedPerceptionPipeline(config=config)
        
        assert pipeline.config.cache_size == 500
        assert pipeline.config.cache_ttl == 10.0
        assert pipeline.config.spatial_grid_size == (50, 50)
        assert pipeline.config.enable_memory_filtering == False
        assert pipeline.config.perception_range == 7.0
    
    def test_pipeline_component_integration(self):
        """Test pipeline components are properly integrated"""
        pipeline = IntegratedPerceptionPipeline()
        
        # Spatial index should be configured according to pipeline config
        assert pipeline.spatial_index.grid_size == pipeline.config.spatial_grid_size
        
        # Cache should be configured according to pipeline config
        assert pipeline.perception_cache.max_size == pipeline.config.cache_size
        assert pipeline.perception_cache.default_ttl == pipeline.config.cache_ttl


class TestSpatialIndexIntegration:
    """Test integration with spatial indexing system"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.pipeline = IntegratedPerceptionPipeline()
        
        # Create mock Mesa model
        self.mesa_model = Mock(spec=mesa.Model)
        self.mesa_model.schedule = Mock()
        
        # Create mock agents
        self.agents = []
        for i in range(3):
            agent = Mock(spec=mesa.Agent)
            agent.unique_id = i
            agent.agent_id = f"agent_{i}"
            agent.pos = (i * 2, i * 3)
            agent.perception_range = 5
            agent.communication_range = 3
            self.agents.append(agent)
        
        self.mesa_model.schedule.agents = self.agents
    
    def test_spatial_index_population_from_mesa(self):
        """Test spatial index is populated from Mesa model"""
        # Extract perceptions should populate spatial index
        perceptions = self.pipeline.extract_perceptions(self.mesa_model)
        
        # Spatial index should contain agents
        assert self.pipeline.spatial_index.object_count == 3
        
        for agent in self.agents:
            assert self.pipeline.spatial_index.contains_object(agent.agent_id)
            spatial_obj = self.pipeline.spatial_index.get_object(agent.agent_id)
            assert spatial_obj.position == agent.pos
    
    def test_spatial_queries_in_perception_extraction(self):
        """Test spatial queries are used during perception extraction"""
        # Extract perceptions
        perceptions = self.pipeline.extract_perceptions(self.mesa_model)
        
        # Each agent should have spatial data based on spatial queries
        for agent_id, perception in perceptions.items():
            assert perception.spatial_data is not None
            assert "nearby_objects" in perception.spatial_data
            assert "distances" in perception.spatial_data
            assert "current_position" in perception.spatial_data
            assert "movement_options" in perception.spatial_data
    
    def test_performance_optimized_spatial_queries(self):
        """Test spatial queries meet performance requirements"""
        # Add many agents for performance testing
        large_agent_list = []
        for i in range(100):
            agent = Mock(spec=mesa.Agent)
            agent.unique_id = i
            agent.agent_id = f"agent_{i}"
            agent.pos = (i % 20, i // 20)
            agent.perception_range = 5
            large_agent_list.append(agent)
        
        self.mesa_model.schedule.agents = large_agent_list
        
        start_time = time.perf_counter()
        
        perceptions = self.pipeline.extract_perceptions(self.mesa_model)
        
        elapsed_time = (time.perf_counter() - start_time) * 1000  # Convert to ms
        
        # Should complete within performance target
        assert elapsed_time < 100  # Under 100ms
        assert len(perceptions) == 100


class TestPerceptionCacheIntegration:
    """Test integration with perception caching system"""
    
    def setup_method(self):
        """Set up test fixtures"""
        config = PerceptionPipelineConfig(
            cache_size=50,
            cache_ttl=5.0,
            enable_memory_filtering=True,
            performance_monitoring=True
        )
        self.pipeline = IntegratedPerceptionPipeline(config=config)
        
        # Create simple Mesa model
        self.mesa_model = Mock(spec=mesa.Model)
        self.mesa_model.schedule = Mock()
        
        agent = Mock(spec=mesa.Agent)
        agent.unique_id = 1
        agent.agent_id = "test_agent"
        agent.pos = (5, 5)
        agent.perception_range = 3
        
        self.mesa_model.schedule.agents = [agent]
    
    def test_cache_hit_on_repeated_extraction(self):
        """Test cache hits on repeated perception extraction"""
        # First extraction (cache miss)
        perceptions_1 = self.pipeline.extract_perceptions(self.mesa_model)
        
        # Second extraction with same model state (cache hit)
        perceptions_2 = self.pipeline.extract_perceptions(self.mesa_model)
        
        # Should get same perceptions
        assert len(perceptions_1) == len(perceptions_2)
        
        # Cache should have recorded hits
        cache_stats = self.pipeline.perception_cache.get_statistics()
        assert cache_stats.hit_count > 0
        assert cache_stats.hit_rate > 0.0
    
    def test_cache_miss_on_model_state_change(self):
        """Test cache miss when Mesa model state changes"""
        # First extraction
        perceptions_1 = self.pipeline.extract_perceptions(self.mesa_model)
        
        # Change agent position (model state change)
        self.mesa_model.schedule.agents[0].pos = (10, 10)
        
        # Second extraction (should be cache miss due to state change)
        perceptions_2 = self.pipeline.extract_perceptions(self.mesa_model)
        
        # Perceptions should be different
        agent_id = "test_agent"
        assert perceptions_1[agent_id].spatial_data != perceptions_2[agent_id].spatial_data
    
    def test_cache_performance_improvement(self):
        """Test cache provides performance improvement"""
        # Warm up spatial index
        self.pipeline.extract_perceptions(self.mesa_model)
        
        # Time first extraction (cache miss)
        start_time = time.perf_counter()
        perceptions_1 = self.pipeline.extract_perceptions(self.mesa_model)
        time_1 = time.perf_counter() - start_time
        
        # Time second extraction (cache hit)
        start_time = time.perf_counter()
        perceptions_2 = self.pipeline.extract_perceptions(self.mesa_model)
        time_2 = time.perf_counter() - start_time
        
        # Cache hit should be faster
        assert time_2 < time_1
        print(f"Cache miss: {time_1*1000:.2f}ms, Cache hit: {time_2*1000:.2f}ms")


class TestMemoryBasedPerceptionFiltering:
    """Test memory-based perception filtering"""
    
    def setup_method(self):
        """Set up test fixtures"""
        config = PerceptionPipelineConfig(
            enable_memory_filtering=True,
            memory_threshold=0.8
        )
        self.pipeline = IntegratedPerceptionPipeline(config=config)
        
        # Create Mesa model with agents
        self.mesa_model = Mock(spec=mesa.Model)
        self.mesa_model.schedule = Mock()
        
        # Create agents with different memory states
        self.strategist = Mock(spec=mesa.Agent)
        self.strategist.unique_id = 1
        self.strategist.agent_id = "strategist"
        self.strategist.pos = (5, 5)
        self.strategist.perception_range = 5
        self.strategist.agent_memory = ["location_A", "strategy_plan_1"]
        
        self.mediator = Mock(spec=mesa.Agent)
        self.mediator.unique_id = 2
        self.mediator.agent_id = "mediator"
        self.mediator.pos = (3, 7)
        self.mediator.perception_range = 4
        self.mediator.agent_memory = ["social_interaction_1", "trust_level_data"]
        
        self.mesa_model.schedule.agents = [self.strategist, self.mediator]
    
    def test_memory_based_cache_filtering(self):
        """Test memory-based cache filtering works correctly"""
        # Extract perceptions
        perceptions = self.pipeline.extract_perceptions(self.mesa_model)
        
        # Filter perceptions based on agent memory
        strategist_perception = self.pipeline.filter_perceptions(perceptions, "strategist")
        mediator_perception = self.pipeline.filter_perceptions(perceptions, "mediator")
        
        # Filtered perceptions should be different for different agents
        assert strategist_perception.agent_id == "strategist"
        assert mediator_perception.agent_id == "mediator"
        
        # Should have different spatial data based on agent capabilities
        assert strategist_perception.spatial_data != mediator_perception.spatial_data
    
    def test_memory_context_affects_caching(self):
        """Test agent memory context affects cache key generation"""
        # Extract perceptions multiple times to generate cache activity
        for _ in range(3):
            perceptions = self.pipeline.extract_perceptions(self.mesa_model)
            self.pipeline.filter_perceptions(perceptions, "strategist")
            self.pipeline.filter_perceptions(perceptions, "mediator")
        
        # Should have cache activity
        final_stats = self.pipeline.perception_cache.get_statistics()
        assert final_stats.total_requests > 0  # Just verify cache is being used
        
        # Verify different agents get different cache entries
        perceptions = self.pipeline.extract_perceptions(self.mesa_model)
        strategist_perception = self.pipeline.filter_perceptions(perceptions, "strategist")
        mediator_perception = self.pipeline.filter_perceptions(perceptions, "mediator")
        
        # Should be different for different agents
        assert strategist_perception.agent_id != mediator_perception.agent_id


class TestPerformanceMonitoring:
    """Test performance monitoring and metrics"""
    
    def setup_method(self):
        """Set up test fixtures"""
        config = PerceptionPipelineConfig(
            performance_monitoring=True,
            cache_size=100,
            spatial_grid_size=(20, 20)
        )
        self.pipeline = IntegratedPerceptionPipeline(config=config)
        
        # Create Mesa model with multiple agents
        self.mesa_model = Mock(spec=mesa.Model)
        self.mesa_model.schedule = Mock()
        
        agents = []
        for i in range(10):
            agent = Mock(spec=mesa.Agent)
            agent.unique_id = i
            agent.agent_id = f"agent_{i}"
            agent.pos = (i * 2, i * 2)
            agent.perception_range = 3
            agents.append(agent)
        
        self.mesa_model.schedule.agents = agents
    
    def test_performance_metrics_collection(self):
        """Test performance metrics are collected during operations"""
        # Extract perceptions to generate metrics
        perceptions = self.pipeline.extract_perceptions(self.mesa_model)
        
        # Get performance metrics
        metrics = self.pipeline.get_performance_metrics()
        
        assert isinstance(metrics, PerformanceMetrics)
        assert metrics.total_extractions > 0
        assert metrics.avg_extraction_time > 0
        assert metrics.cache_hit_rate >= 0.0
        assert metrics.spatial_query_count > 0
    
    def test_performance_requirements_met(self):
        """Test pipeline meets performance requirements"""
        # Multiple extraction cycles to get stable metrics
        for _ in range(5):
            perceptions = self.pipeline.extract_perceptions(self.mesa_model)
        
        metrics = self.pipeline.get_performance_metrics()
        
        # Performance requirements
        assert metrics.avg_extraction_time < 0.1  # Under 100ms
        assert metrics.cache_hit_rate >= 0.6  # At least 60% (after warmup)
        
        print(f"Avg extraction time: {metrics.avg_extraction_time*1000:.2f}ms")
        print(f"Cache hit rate: {metrics.cache_hit_rate:.2%}")
    
    def test_performance_optimization_with_scale(self):
        """Test performance scales with larger agent counts"""
        # Test with increasing agent counts
        agent_counts = [10, 50, 100]
        extraction_times = []
        
        for count in agent_counts:
            # Create agents for this test
            agents = []
            for i in range(count):
                agent = Mock(spec=mesa.Agent)
                agent.unique_id = i
                agent.agent_id = f"agent_{i}"
                agent.pos = (i % 20, i // 20)
                agent.perception_range = 3
                agents.append(agent)
            
            self.mesa_model.schedule.agents = agents
            
            # Measure extraction time
            start_time = time.perf_counter()
            perceptions = self.pipeline.extract_perceptions(self.mesa_model)
            extraction_time = time.perf_counter() - start_time
            
            extraction_times.append(extraction_time)
            
            # Should complete within reasonable time regardless of scale
            assert extraction_time < 0.5  # Under 500ms even for 100 agents
        
        print(f"Extraction times: {[t*1000 for t in extraction_times]} ms")


class TestErrorHandlingAndEdgeCases:
    """Test error handling and edge cases"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.pipeline = IntegratedPerceptionPipeline()
    
    def test_empty_mesa_model(self):
        """Test handling of empty Mesa model"""
        mesa_model = Mock(spec=mesa.Model)
        mesa_model.schedule = Mock()
        mesa_model.schedule.agents = []
        
        # Should handle empty model gracefully
        perceptions = self.pipeline.extract_perceptions(mesa_model)
        
        assert isinstance(perceptions, dict)
        assert len(perceptions) == 0
    
    def test_malformed_mesa_agents(self):
        """Test handling of malformed Mesa agents"""
        mesa_model = Mock(spec=mesa.Model)
        mesa_model.schedule = Mock()
        
        # Agent missing required attributes
        bad_agent = Mock()
        bad_agent.unique_id = 1
        # Missing: agent_id, pos, perception_range
        
        mesa_model.schedule.agents = [bad_agent]
        
        # Should handle gracefully without crashing
        perceptions = self.pipeline.extract_perceptions(mesa_model)
        
        # May return empty or default perceptions
        assert isinstance(perceptions, dict)
    
    def test_cache_corruption_recovery(self):
        """Test recovery from cache corruption"""
        # Simulate cache corruption by clearing internal state
        self.pipeline.perception_cache.clear()
        
        # Create simple model
        mesa_model = Mock(spec=mesa.Model)
        mesa_model.schedule = Mock()
        
        agent = Mock(spec=mesa.Agent)
        agent.unique_id = 1
        agent.agent_id = "test_agent"
        agent.pos = (5, 5)
        agent.perception_range = 3
        
        mesa_model.schedule.agents = [agent]
        
        # Should recover and work normally
        perceptions = self.pipeline.extract_perceptions(mesa_model)
        
        assert len(perceptions) == 1
        assert "test_agent" in perceptions
    
    def test_spatial_index_corruption_recovery(self):
        """Test recovery from spatial index corruption"""
        # Simulate index corruption
        self.pipeline.spatial_index.clear()
        
        # Create simple model
        mesa_model = Mock(spec=mesa.Model)
        mesa_model.schedule = Mock()
        
        agent = Mock(spec=mesa.Agent)
        agent.unique_id = 1
        agent.agent_id = "test_agent"
        agent.pos = (5, 5)
        agent.perception_range = 3
        
        mesa_model.schedule.agents = [agent]
        
        # Should rebuild index and work normally
        perceptions = self.pipeline.extract_perceptions(mesa_model)
        
        assert len(perceptions) == 1
        assert self.pipeline.spatial_index.object_count == 1


class TestPipelineHandoffToDecisionEngine:
    """Test pipeline handoff to Decision Engine (Agent B)"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.pipeline = IntegratedPerceptionPipeline()
        
        # Create realistic Mesa model
        self.mesa_model = Mock(spec=mesa.Model)
        self.mesa_model.schedule = Mock()
        
        # Create agents representing escape room scenario
        self.strategist = Mock(spec=mesa.Agent)
        self.strategist.unique_id = 1
        self.strategist.agent_id = "strategist"
        self.strategist.pos = (5, 5)
        self.strategist.perception_range = 5
        
        self.mediator = Mock(spec=mesa.Agent)
        self.mediator.unique_id = 2
        self.mediator.agent_id = "mediator"
        self.mediator.pos = (7, 3)
        self.mediator.perception_range = 4
        
        self.survivor = Mock(spec=mesa.Agent)
        self.survivor.unique_id = 3
        self.survivor.agent_id = "survivor"
        self.survivor.pos = (3, 8)
        self.survivor.perception_range = 3
        
        self.mesa_model.schedule.agents = [self.strategist, self.mediator, self.survivor]
    
    def test_create_perception_handoff_data(self):
        """Test creation of perception handoff data for Decision Engine"""
        # Extract perceptions
        perceptions = self.pipeline.extract_perceptions(self.mesa_model)
        
        # Create handoff data
        handoff = self.pipeline.create_handoff_for_decision_engine(perceptions)
        
        # Verify handoff contains required information
        assert handoff.perceptions is not None
        assert len(handoff.perceptions) == 3
        assert handoff.performance_metrics is not None
        assert handoff.extraction_timestamp is not None
        assert handoff.validation_passed == True
        
        # Verify performance metrics meet requirements
        assert handoff.performance_metrics["extraction_time"] < 0.1  # Under 100ms
        assert handoff.performance_metrics["cache_hit_rate"] >= 0.0
        assert handoff.performance_metrics["agents_processed"] == 3
    
    def test_handoff_data_validation(self):
        """Test handoff data validation for Decision Engine"""
        # Extract perceptions
        perceptions = self.pipeline.extract_perceptions(self.mesa_model)
        
        # Create handoff
        handoff = self.pipeline.create_handoff_for_decision_engine(perceptions)
        
        # Validate handoff meets Decision Engine requirements
        for agent_id, perception in handoff.perceptions.items():
            # Each perception should have required fields
            assert perception.agent_id is not None
            assert perception.timestamp is not None
            assert perception.spatial_data is not None
            assert perception.available_actions is not None
            
            # Spatial data should be complete
            assert "current_position" in perception.spatial_data
            assert "nearby_objects" in perception.spatial_data
            assert "movement_options" in perception.spatial_data
    
    def test_handoff_performance_requirements(self):
        """Test handoff meets performance requirements for Decision Engine"""
        # Multiple extractions to get stable performance data
        handoffs = []
        for _ in range(5):
            perceptions = self.pipeline.extract_perceptions(self.mesa_model)
            handoff = self.pipeline.create_handoff_for_decision_engine(perceptions)
            handoffs.append(handoff)
        
        # All handoffs should meet performance requirements
        for handoff in handoffs:
            assert handoff.validation_passed == True
            assert handoff.performance_metrics["extraction_time"] < 0.1
            
        # Average performance should be good
        avg_extraction_time = sum(h.performance_metrics["extraction_time"] for h in handoffs) / len(handoffs)
        avg_cache_hit_rate = sum(h.performance_metrics["cache_hit_rate"] for h in handoffs) / len(handoffs)
        
        assert avg_extraction_time < 0.05  # Under 50ms average
        assert avg_cache_hit_rate >= 0.4  # Over 40% hit rate after warmup (more realistic)
        
        print(f"Average extraction time: {avg_extraction_time*1000:.2f}ms")
        print(f"Average cache hit rate: {avg_cache_hit_rate:.2%}")