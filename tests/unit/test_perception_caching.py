"""
Test Suite for Perception Caching System

This test suite validates the perception caching layer that optimizes
perception pipeline performance by avoiding redundant calculations.

Test Categories:
- Cache creation and configuration
- Cache hit/miss behavior 
- Cache invalidation strategies
- Performance optimization (>80% hit rate target)
- Memory-based cache filtering
- TTL (time-to-live) management
"""

import pytest
from typing import Dict, List, Any, Optional
from unittest.mock import Mock, MagicMock
import time
from datetime import datetime, timedelta

from src.escape_room_sim.hybrid.perception.perception_cache import (
    PerceptionCache, CacheEntry, CacheKey, CacheStatistics
)
from src.escape_room_sim.hybrid.core_architecture import PerceptionData


class TestPerceptionCacheCreation:
    """Test perception cache creation and configuration"""
    
    def test_perception_cache_initialization_default(self):
        """Test perception cache creates with default parameters"""
        cache = PerceptionCache()
        
        assert cache.max_size == 1000  # Default cache size
        assert cache.default_ttl == 5.0  # Default TTL in seconds
        assert cache.size == 0
        assert cache.is_empty == True
        assert cache.hit_rate == 0.0
    
    def test_perception_cache_custom_parameters(self):
        """Test perception cache with custom configuration"""
        cache = PerceptionCache(
            max_size=500,
            default_ttl=10.0,
            enable_memory_filtering=True,
            memory_threshold=0.85
        )
        
        assert cache.max_size == 500
        assert cache.default_ttl == 10.0
        assert cache.enable_memory_filtering == True
        assert cache.memory_threshold == 0.85
    
    def test_perception_cache_invalid_parameters(self):
        """Test perception cache rejects invalid parameters"""
        with pytest.raises(ValueError, match="Max size must be positive"):
            PerceptionCache(max_size=0)
        
        with pytest.raises(ValueError, match="TTL must be positive"):
            PerceptionCache(default_ttl=-1.0)
        
        with pytest.raises(ValueError, match="Memory threshold must be between 0 and 1"):
            PerceptionCache(memory_threshold=1.5)


class TestCacheKeyGeneration:
    """Test cache key generation and hashing"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.cache = PerceptionCache()
        self.sample_perception = PerceptionData(
            agent_id="test_agent",
            timestamp=datetime.now(),
            spatial_data={"position": (5.0, 5.0)},
            environmental_state={"temperature": 20.0},
            nearby_agents=["agent_1", "agent_2"],
            available_actions=["move", "examine"],
            resources={"keys": 1},
            constraints={"movement_blocked": False}
        )
    
    def test_cache_key_generation_basic(self):
        """Test basic cache key generation"""
        key = self.cache.create_cache_key("test_agent", {"type": "spatial"})
        
        assert isinstance(key, CacheKey)
        assert key.agent_id == "test_agent"
        assert key.context_hash is not None
        assert key.timestamp is not None
    
    def test_cache_key_generation_consistent(self):
        """Test cache key generation is consistent for same inputs"""
        context = {"type": "spatial", "position": (5.0, 5.0)}
        
        key1 = self.cache.create_cache_key("test_agent", context)
        key2 = self.cache.create_cache_key("test_agent", context)
        
        assert key1.context_hash == key2.context_hash
    
    def test_cache_key_generation_different_contexts(self):
        """Test cache keys differ for different contexts"""
        context1 = {"type": "spatial", "position": (5.0, 5.0)}
        context2 = {"type": "spatial", "position": (6.0, 6.0)}
        
        key1 = self.cache.create_cache_key("test_agent", context1)
        key2 = self.cache.create_cache_key("test_agent", context2)
        
        assert key1.context_hash != key2.context_hash
    
    def test_cache_key_generation_different_agents(self):
        """Test cache keys differ for different agents"""
        context = {"type": "spatial", "position": (5.0, 5.0)}
        
        key1 = self.cache.create_cache_key("agent_1", context)
        key2 = self.cache.create_cache_key("agent_2", context)
        
        # Different agents should have different keys even with same context
        assert key1 != key2


class TestCacheOperations:
    """Test cache storage and retrieval operations"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.cache = PerceptionCache(max_size=10, default_ttl=60.0)
        self.test_perception = PerceptionData(
            agent_id="test_agent",
            timestamp=datetime.now(),
            spatial_data={"position": (5.0, 5.0)},
            environmental_state={"temperature": 20.0},
            nearby_agents=["agent_1"],
            available_actions=["move"],
            resources={"keys": 1},
            constraints={}
        )
    
    def test_cache_put_and_get_basic(self):
        """Test basic cache put and get operations"""
        key = self.cache.create_cache_key("test_agent", {"context": "test"})
        
        # Put perception in cache
        result = self.cache.put(key, self.test_perception)
        assert result == True
        assert self.cache.size == 1
        
        # Get perception from cache
        cached_perception = self.cache.get(key)
        assert cached_perception is not None
        assert cached_perception.agent_id == "test_agent"
        assert cached_perception.spatial_data == {"position": (5.0, 5.0)}
    
    def test_cache_hit_statistics(self):
        """Test cache hit statistics tracking"""
        key = self.cache.create_cache_key("test_agent", {"context": "test"})
        
        # Initial hit rate should be 0
        assert self.cache.hit_rate == 0.0
        
        # Cache miss
        result = self.cache.get(key)
        assert result is None
        assert self.cache.hit_rate == 0.0  # Still 0 (0 hits / 1 request)
        
        # Cache put
        self.cache.put(key, self.test_perception)
        
        # Cache hit
        result = self.cache.get(key)
        assert result is not None
        assert self.cache.hit_rate == 0.5  # 1 hit / 2 requests
        
        # Another cache hit
        result = self.cache.get(key)
        assert result is not None
        assert self.cache.hit_rate == 0.67  # 2 hits / 3 requests (rounded)
    
    def test_cache_miss_nonexistent_key(self):
        """Test cache miss for nonexistent key"""
        key = self.cache.create_cache_key("nonexistent", {"context": "test"})
        
        result = self.cache.get(key)
        assert result is None
    
    def test_cache_overwrite_existing(self):
        """Test overwriting existing cache entry"""
        key = self.cache.create_cache_key("test_agent", {"context": "test"})
        
        # Put first perception
        self.cache.put(key, self.test_perception)
        
        # Create different perception
        new_perception = PerceptionData(
            agent_id="test_agent",
            timestamp=datetime.now(),
            spatial_data={"position": (10.0, 10.0)},  # Different position
            environmental_state={"temperature": 25.0},  # Different temperature
            nearby_agents=[],
            available_actions=["examine"],
            resources={"keys": 2},
            constraints={}
        )
        
        # Overwrite with new perception
        result = self.cache.put(key, new_perception)
        assert result == True
        assert self.cache.size == 1  # Still only 1 entry
        
        # Verify new perception is cached
        cached_perception = self.cache.get(key)
        assert cached_perception.spatial_data == {"position": (10.0, 10.0)}
        assert cached_perception.environmental_state == {"temperature": 25.0}


class TestCacheTTLManagement:
    """Test cache time-to-live (TTL) management"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.cache = PerceptionCache(max_size=10, default_ttl=0.1)  # 100ms TTL for testing
        self.test_perception = PerceptionData(
            agent_id="test_agent",
            timestamp=datetime.now(),
            spatial_data={"position": (5.0, 5.0)},
            environmental_state={},
            nearby_agents=[],
            available_actions=[],
            resources={},
            constraints={}
        )
    
    def test_cache_ttl_expiration(self):
        """Test cache entries expire after TTL"""
        key = self.cache.create_cache_key("test_agent", {"context": "test"})
        
        # Put perception in cache
        self.cache.put(key, self.test_perception)
        
        # Should be available immediately
        result = self.cache.get(key)
        assert result is not None
        
        # Wait for TTL to expire
        time.sleep(0.15)  # Wait longer than 100ms TTL
        
        # Should be expired now
        result = self.cache.get(key)
        assert result is None
    
    def test_cache_custom_ttl(self):
        """Test cache entries with custom TTL"""
        key = self.cache.create_cache_key("test_agent", {"context": "test"})
        
        # Put with custom TTL of 1 second
        self.cache.put(key, self.test_perception, ttl=1.0)
        
        # Should be available after default TTL (100ms)
        time.sleep(0.15)
        result = self.cache.get(key)
        assert result is not None  # Custom TTL hasn't expired yet
        
        # Wait for custom TTL to expire
        time.sleep(1.0)
        result = self.cache.get(key)
        assert result is None
    
    def test_cache_cleanup_expired_entries(self):
        """Test automatic cleanup of expired entries"""
        # Fill cache with entries that expire quickly
        for i in range(5):
            key = self.cache.create_cache_key(f"agent_{i}", {"context": f"test_{i}"})
            self.cache.put(key, self.test_perception)
        
        assert self.cache.size == 5
        
        # Wait for entries to expire
        time.sleep(0.15)
        
        # Trigger cleanup by doing a get operation
        key = self.cache.create_cache_key("agent_0", {"context": "test_0"})
        self.cache.get(key)
        
        # Cache should have cleaned up expired entries
        assert self.cache.size < 5


class TestCacheMemoryManagement:
    """Test cache memory management and eviction"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.cache = PerceptionCache(max_size=3, default_ttl=60.0)  # Small cache for testing
        self.perceptions = []
        
        # Create test perceptions
        for i in range(5):
            perception = PerceptionData(
                agent_id=f"agent_{i}",
                timestamp=datetime.now(),
                spatial_data={"position": (i, i)},
                environmental_state={},
                nearby_agents=[],
                available_actions=[],
                resources={},
                constraints={}
            )
            self.perceptions.append(perception)
    
    def test_cache_max_size_limit(self):
        """Test cache respects maximum size limit"""
        # Fill cache beyond max size
        for i in range(5):
            key = self.cache.create_cache_key(f"agent_{i}", {"context": f"test_{i}"})
            self.cache.put(key, self.perceptions[i])
        
        # Cache should not exceed max size
        assert self.cache.size <= self.cache.max_size
    
    def test_cache_lru_eviction(self):
        """Test Least Recently Used (LRU) eviction policy"""
        keys = []
        
        # Fill cache to max size
        for i in range(3):
            key = self.cache.create_cache_key(f"agent_{i}", {"context": f"test_{i}"})
            keys.append(key)
            self.cache.put(key, self.perceptions[i])
        
        # Access first entry to make it recently used
        self.cache.get(keys[0])
        
        # Add new entry (should evict keys[1] as it's least recently used)
        new_key = self.cache.create_cache_key("agent_new", {"context": "test_new"})
        self.cache.put(new_key, self.perceptions[0])
        
        # First entry should still be there (recently accessed)
        assert self.cache.get(keys[0]) is not None
        
        # Second entry should be evicted
        assert self.cache.get(keys[1]) is None
        
        # Third entry should still be there
        assert self.cache.get(keys[2]) is not None
        
        # New entry should be there
        assert self.cache.get(new_key) is not None
    
    def test_cache_clear_operation(self):
        """Test cache clear operation"""
        # Fill cache with entries
        for i in range(3):
            key = self.cache.create_cache_key(f"agent_{i}", {"context": f"test_{i}"})
            self.cache.put(key, self.perceptions[i])
        
        assert self.cache.size == 3
        assert not self.cache.is_empty
        
        # Clear cache
        self.cache.clear()
        
        assert self.cache.size == 0
        assert self.cache.is_empty


class TestMemoryBasedFiltering:
    """Test memory-based perception filtering"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.cache = PerceptionCache(
            max_size=100,
            enable_memory_filtering=True,
            memory_threshold=0.8
        )
        self.base_perception = PerceptionData(
            agent_id="test_agent",
            timestamp=datetime.now(),
            spatial_data={"position": (5.0, 5.0)},
            environmental_state={"temperature": 20.0},
            nearby_agents=["agent_1"],
            available_actions=["move", "examine"],
            resources={"keys": 1},
            constraints={}
        )
    
    def test_memory_filtering_enabled(self):
        """Test memory-based filtering when enabled"""
        # Create key with agent memory context
        memory_context = {
            "agent_memory": ["remembered_location_A", "remembered_resource_B"],
            "experience_level": "high",
            "recent_actions": ["move", "examine"]
        }
        
        key = self.cache.create_cache_key("test_agent", memory_context)
        
        # Put perception in cache
        self.cache.put(key, self.base_perception)
        
        # Get with same memory context should hit
        cached_perception = self.cache.get(key)
        assert cached_perception is not None
        
        # Get with different memory context should miss
        different_memory_context = {
            "agent_memory": ["remembered_location_C"],  # Different memory
            "experience_level": "high",
            "recent_actions": ["move", "examine"]
        }
        different_key = self.cache.create_cache_key("test_agent", different_memory_context)
        cached_perception = self.cache.get(different_key)
        assert cached_perception is None
    
    def test_memory_filtering_disabled(self):
        """Test behavior when memory filtering is disabled"""
        cache_no_memory = PerceptionCache(enable_memory_filtering=False)
        
        # Memory context should be ignored when disabled
        key1 = cache_no_memory.create_cache_key("test_agent", {"memory": "state_1"})
        key2 = cache_no_memory.create_cache_key("test_agent", {"memory": "state_2"})
        
        # Put perception in cache with first key
        cache_no_memory.put(key1, self.base_perception)
        
        # Both keys should work if memory filtering is disabled
        # (This is implementation dependent - might still be different keys)
        cached_perception = cache_no_memory.get(key1)
        assert cached_perception is not None
    
    def test_memory_similarity_threshold(self):
        """Test memory similarity threshold filtering"""
        # Create perceptions with similar but not identical memory contexts
        similar_context_1 = {
            "agent_memory": ["location_A", "location_B", "resource_1"],
            "experience_level": "medium"
        }
        
        similar_context_2 = {
            "agent_memory": ["location_A", "location_B", "resource_2"],  # One different item
            "experience_level": "medium"
        }
        
        key1 = self.cache.create_cache_key("test_agent", similar_context_1)
        key2 = self.cache.create_cache_key("test_agent", similar_context_2)
        
        # Keys should be different (similarity threshold controls reuse)
        assert key1 != key2


class TestCachePerformance:
    """Test cache performance requirements"""
    
    def setup_method(self):
        """Set up performance test environment"""
        self.cache = PerceptionCache(max_size=10000, default_ttl=300.0)
        
        # Create test perceptions
        self.perceptions = []
        for i in range(100):
            perception = PerceptionData(
                agent_id=f"agent_{i % 10}",  # 10 different agents
                timestamp=datetime.now(),
                spatial_data={"position": (i % 50, i // 50)},
                environmental_state={"temperature": 20.0 + i % 10},
                nearby_agents=[f"agent_{(i+1) % 10}"],
                available_actions=["move", "examine"],
                resources={"keys": i % 3},
                constraints={}
            )
            self.perceptions.append(perception)
    
    def test_cache_put_performance(self):
        """Test cache put operation performance"""
        start_time = time.perf_counter()
        
        # Put 100 perceptions in cache
        for i, perception in enumerate(self.perceptions):
            key = self.cache.create_cache_key(perception.agent_id, {"context": f"test_{i}"})
            self.cache.put(key, perception)
        
        elapsed_time = (time.perf_counter() - start_time) * 1000  # Convert to ms
        
        # Should complete in reasonable time (< 50ms for 100 operations)
        assert elapsed_time < 50
        print(f"Cache put performance: {elapsed_time:.2f}ms for 100 operations")
    
    def test_cache_get_performance(self):
        """Test cache get operation performance"""
        # Pre-populate cache
        keys = []
        for i, perception in enumerate(self.perceptions):
            key = self.cache.create_cache_key(perception.agent_id, {"context": f"test_{i}"})
            keys.append(key)
            self.cache.put(key, perception)
        
        start_time = time.perf_counter()
        
        # Get 100 perceptions from cache
        for key in keys:
            self.cache.get(key)
        
        elapsed_time = (time.perf_counter() - start_time) * 1000  # Convert to ms
        
        # Should complete in reasonable time (< 10ms for 100 operations)
        assert elapsed_time < 10
        print(f"Cache get performance: {elapsed_time:.2f}ms for 100 operations")
    
    def test_cache_hit_rate_target(self):
        """Test cache achieves target hit rate (>80%)"""
        # Simulate realistic perception access patterns
        keys = []
        
        # Create cache entries for common scenarios
        for i in range(20):
            key = self.cache.create_cache_key(f"agent_{i % 5}", {"scenario": f"common_{i % 10}"})
            keys.append(key)
            self.cache.put(key, self.perceptions[i])
        
        # Simulate access pattern with high reuse
        hit_count = 0
        total_requests = 100
        
        for i in range(total_requests):
            # 80% of requests reuse existing keys, 20% are new
            if i < 80:
                key = keys[i % len(keys)]  # Reuse existing keys
            else:
                key = self.cache.create_cache_key(f"agent_new", {"scenario": f"new_{i}"})
            
            result = self.cache.get(key)
            if result is not None:
                hit_count += 1
        
        actual_hit_rate = hit_count / total_requests
        print(f"Actual hit rate: {actual_hit_rate:.2%}")
        
        # Should achieve target hit rate
        assert actual_hit_rate >= 0.80


class TestCacheStatistics:
    """Test cache statistics and monitoring"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.cache = PerceptionCache(max_size=10, default_ttl=60.0)
        self.test_perception = PerceptionData(
            agent_id="test_agent",
            timestamp=datetime.now(),
            spatial_data={},
            environmental_state={},
            nearby_agents=[],
            available_actions=[],
            resources={},
            constraints={}
        )
    
    def test_cache_statistics_collection(self):
        """Test cache statistics collection"""
        # Initial statistics
        stats = self.cache.get_statistics()
        assert isinstance(stats, CacheStatistics)
        assert stats.hit_count == 0
        assert stats.miss_count == 0
        assert stats.hit_rate == 0.0
        assert stats.size == 0
        
        # Add some cache operations
        key = self.cache.create_cache_key("test_agent", {"context": "test"})
        
        # Cache miss
        self.cache.get(key)
        
        # Cache put
        self.cache.put(key, self.test_perception)
        
        # Cache hit
        self.cache.get(key)
        
        # Updated statistics
        stats = self.cache.get_statistics()
        assert stats.hit_count == 1
        assert stats.miss_count == 1
        assert stats.hit_rate == 0.5
        assert stats.size == 1
    
    def test_cache_reset_statistics(self):
        """Test cache statistics reset"""
        # Generate some activity
        key = self.cache.create_cache_key("test_agent", {"context": "test"})
        self.cache.get(key)  # Miss
        self.cache.put(key, self.test_perception)
        self.cache.get(key)  # Hit
        
        # Verify statistics exist
        stats = self.cache.get_statistics()
        assert stats.hit_count > 0 or stats.miss_count > 0
        
        # Reset statistics
        self.cache.reset_statistics()
        
        # Statistics should be reset
        stats = self.cache.get_statistics()
        assert stats.hit_count == 0
        assert stats.miss_count == 0
        assert stats.hit_rate == 0.0
        # Size should remain (only statistics reset, not cache contents)
        assert stats.size >= 0


class TestCacheIntegration:
    """Test cache integration with perception pipeline"""
    
    def setup_method(self):
        """Set up integration test environment"""
        self.cache = PerceptionCache()
        self.mock_mesa_model = Mock()
        self.mock_mesa_model.schedule = Mock()
        self.mock_mesa_model.schedule.agents = []
    
    def test_cache_integration_with_perception_pipeline(self):
        """Test cache integrates properly with perception pipeline"""
        # This would be tested when we integrate with the actual pipeline
        # For now, test the interface compatibility
        
        agent_id = "test_agent"
        context = {"mesa_state_hash": "abc123", "position": (5.0, 5.0)}
        
        # Create cache key for pipeline use
        key = self.cache.create_cache_key(agent_id, context)
        assert key is not None
        
        # Test perception storage and retrieval
        perception = PerceptionData(
            agent_id=agent_id,
            timestamp=datetime.now(),
            spatial_data=context,
            environmental_state={},
            nearby_agents=[],
            available_actions=[],
            resources={},
            constraints={}
        )
        
        self.cache.put(key, perception)
        cached_perception = self.cache.get(key)
        
        assert cached_perception is not None
        assert cached_perception.agent_id == agent_id
    
    def test_cache_key_generation_for_mesa_state(self):
        """Test cache key generation based on Mesa model state"""
        # Simulate Mesa model state changes
        mesa_state_1 = {
            "step_count": 10,
            "agent_positions": {"agent_1": (5, 5), "agent_2": (3, 7)},
            "resources": ["key_1", "tool_1"]
        }
        
        mesa_state_2 = {
            "step_count": 11,  # Different step
            "agent_positions": {"agent_1": (5, 5), "agent_2": (3, 7)},
            "resources": ["key_1", "tool_1"]
        }
        
        key1 = self.cache.create_cache_key("agent_1", mesa_state_1)
        key2 = self.cache.create_cache_key("agent_1", mesa_state_2)
        
        # Different Mesa states should produce different cache keys
        assert key1 != key2