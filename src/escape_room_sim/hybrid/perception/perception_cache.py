"""
High-Performance Perception Caching System

This module implements an optimized caching layer for perception data
that reduces redundant calculations in the Mesa-CrewAI hybrid architecture.

Key Features:
- LRU (Least Recently Used) cache eviction policy
- Configurable TTL (Time-To-Live) for cache entries
- Memory-based cache key generation with context awareness
- Performance monitoring with hit rate tracking
- Automatic cleanup of expired entries
- Memory-efficient storage with compression options

Architecture:
- Uses hashable cache keys based on agent context and Mesa state
- Maintains insertion order for LRU eviction
- Supports both global and per-entry TTL settings
- Provides comprehensive statistics for performance optimization
- Integrates seamlessly with perception pipeline
"""

from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from collections import OrderedDict
import time
import hashlib
import json
from datetime import datetime, timedelta

from src.escape_room_sim.hybrid.core_architecture import PerceptionData


@dataclass(frozen=True)
class CacheKey:
    """
    Immutable cache key for perception data
    
    Contains agent ID, context hash, and timestamp for unique identification.
    Made frozen to enable use as dictionary key.
    """
    agent_id: str
    context_hash: str
    timestamp: float
    
    def __hash__(self) -> int:
        """Hash function for use as dictionary key"""
        return hash((self.agent_id, self.context_hash))
    
    def __eq__(self, other) -> bool:
        """Equality comparison for cache key matching"""
        if not isinstance(other, CacheKey):
            return False
        return (self.agent_id == other.agent_id and 
                self.context_hash == other.context_hash)


@dataclass
class CacheEntry:
    """
    Cache entry containing perception data and metadata
    
    Stores the actual perception data along with cache management information.
    """
    perception: PerceptionData
    created_time: float
    ttl: float
    access_count: int = 0
    last_access_time: float = field(default_factory=time.perf_counter)
    
    def is_expired(self) -> bool:
        """Check if cache entry has expired"""
        return time.perf_counter() - self.created_time > self.ttl
    
    def touch(self) -> None:
        """Update access statistics"""
        self.access_count += 1
        self.last_access_time = time.perf_counter()


@dataclass
class CacheStatistics:
    """
    Cache performance statistics
    
    Tracks cache performance metrics for optimization and monitoring.
    """
    hit_count: int = 0
    miss_count: int = 0
    eviction_count: int = 0
    expired_count: int = 0
    size: int = 0
    max_size: int = 0
    
    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate as percentage"""
        total_requests = self.hit_count + self.miss_count
        if total_requests == 0:
            return 0.0
        return self.hit_count / total_requests
    
    @property
    def total_requests(self) -> int:
        """Total number of cache requests"""
        return self.hit_count + self.miss_count


class PerceptionCache:
    """
    High-performance perception cache with LRU eviction and TTL management
    
    Optimized for the common case of repeated perception queries with
    temporal locality and agent-specific context awareness.
    
    Performance Characteristics:
    - Get operation: O(1) average case
    - Put operation: O(1) average case
    - Cleanup operation: O(n) where n is expired entries
    - Memory usage: O(k) where k is cache size
    """
    
    def __init__(self, 
                 max_size: int = 1000,
                 default_ttl: float = 5.0,
                 enable_memory_filtering: bool = True,
                 memory_threshold: float = 0.8,
                 enable_compression: bool = False):
        """
        Initialize perception cache
        
        Args:
            max_size: Maximum number of cache entries
            default_ttl: Default time-to-live in seconds
            enable_memory_filtering: Enable memory-based cache filtering
            memory_threshold: Memory similarity threshold (0.0-1.0)
            enable_compression: Enable cache entry compression (future)
        """
        if max_size <= 0:
            raise ValueError("Max size must be positive")
        if default_ttl <= 0:
            raise ValueError("TTL must be positive")
        if not 0.0 <= memory_threshold <= 1.0:
            raise ValueError("Memory threshold must be between 0 and 1")
        
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.enable_memory_filtering = enable_memory_filtering
        self.memory_threshold = memory_threshold
        self.enable_compression = enable_compression
        
        # Core data structures
        self._cache: OrderedDict[CacheKey, CacheEntry] = OrderedDict()
        
        # Performance tracking
        self._statistics = CacheStatistics(max_size=max_size)
        self._last_cleanup_time = time.perf_counter()
        self._cleanup_interval = 10.0  # Cleanup every 10 seconds
    
    @property
    def size(self) -> int:
        """Current number of cache entries"""
        return len(self._cache)
    
    @property
    def is_empty(self) -> bool:
        """Whether cache is empty"""
        return len(self._cache) == 0
    
    @property
    def hit_rate(self) -> float:
        """Current cache hit rate"""
        return round(self._statistics.hit_rate, 2)
    
    def create_cache_key(self, agent_id: str, context: Dict[str, Any]) -> CacheKey:
        """
        Create cache key from agent ID and context
        
        Args:
            agent_id: Agent identifier
            context: Context dictionary containing relevant state
            
        Returns:
            CacheKey for cache operations
        """
        # Filter context based on memory filtering settings
        if self.enable_memory_filtering:
            filtered_context = self._filter_context_by_memory(context)
        else:
            filtered_context = context
        
        # Create deterministic hash of context
        context_str = json.dumps(filtered_context, sort_keys=True, default=str)
        context_hash = hashlib.md5(context_str.encode()).hexdigest()
        
        return CacheKey(
            agent_id=agent_id,
            context_hash=context_hash,
            timestamp=time.perf_counter()
        )
    
    def get(self, key: CacheKey) -> Optional[PerceptionData]:
        """
        Retrieve perception from cache
        
        Args:
            key: Cache key to retrieve
            
        Returns:
            PerceptionData if found and not expired, None otherwise
        """
        # Periodic cleanup of expired entries
        self._maybe_cleanup_expired()
        
        if key not in self._cache:
            self._statistics.miss_count += 1
            return None
        
        entry = self._cache[key]
        
        # Check if entry has expired
        if entry.is_expired():
            del self._cache[key]
            self._statistics.expired_count += 1
            self._statistics.miss_count += 1
            return None
        
        # Update access statistics and move to end (LRU)
        entry.touch()
        self._cache.move_to_end(key)
        
        self._statistics.hit_count += 1
        return entry.perception
    
    def put(self, key: CacheKey, perception: PerceptionData, ttl: Optional[float] = None) -> bool:
        """
        Store perception in cache
        
        Args:
            key: Cache key for storage
            perception: PerceptionData to store
            ttl: Optional custom TTL, uses default if None
            
        Returns:
            True if stored successfully
        """
        if ttl is None:
            ttl = self.default_ttl
        
        # Create cache entry
        entry = CacheEntry(
            perception=perception,
            created_time=time.perf_counter(),
            ttl=ttl
        )
        
        # Check if key already exists (update case)
        if key in self._cache:
            self._cache[key] = entry
            self._cache.move_to_end(key)
            return True
        
        # Ensure we don't exceed max size
        while len(self._cache) >= self.max_size:
            self._evict_lru_entry()
        
        # Add new entry
        self._cache[key] = entry
        self._statistics.size = len(self._cache)
        
        return True
    
    def clear(self) -> None:
        """Clear all cache entries"""
        self._cache.clear()
        self._statistics = CacheStatistics(max_size=self.max_size)
    
    def get_statistics(self) -> CacheStatistics:
        """Get current cache statistics"""
        # Update size in statistics
        self._statistics.size = len(self._cache)
        return self._statistics
    
    def reset_statistics(self) -> None:
        """Reset cache statistics (but keep cached data)"""
        current_size = len(self._cache)
        self._statistics = CacheStatistics(max_size=self.max_size, size=current_size)
    
    def cleanup_expired(self) -> int:
        """
        Manually cleanup expired entries
        
        Returns:
            Number of expired entries removed
        """
        current_time = time.perf_counter()
        expired_keys = []
        
        for key, entry in self._cache.items():
            if entry.is_expired():
                expired_keys.append(key)
        
        # Remove expired entries
        for key in expired_keys:
            del self._cache[key]
        
        expired_count = len(expired_keys)
        self._statistics.expired_count += expired_count
        self._statistics.size = len(self._cache)
        self._last_cleanup_time = current_time
        
        return expired_count
    
    def get_memory_usage_info(self) -> Dict[str, Any]:
        """Get cache memory usage information"""
        import sys
        
        # Calculate approximate memory usage
        cache_overhead = sys.getsizeof(self._cache)
        
        total_entry_size = 0
        for key, entry in self._cache.items():
            key_size = sys.getsizeof(key)
            entry_size = sys.getsizeof(entry) + sys.getsizeof(entry.perception)
            total_entry_size += key_size + entry_size
        
        total_memory = cache_overhead + total_entry_size
        avg_entry_size = total_entry_size / max(1, len(self._cache))
        
        return {
            "total_memory_bytes": total_memory,
            "cache_overhead_bytes": cache_overhead,
            "avg_entry_size_bytes": avg_entry_size,
            "entry_count": len(self._cache),
            "memory_per_entry": avg_entry_size
        }
    
    # Private methods
    
    def _filter_context_by_memory(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Filter context based on memory-related factors
        
        Args:
            context: Raw context dictionary
            
        Returns:
            Filtered context for cache key generation
        """
        if not self.enable_memory_filtering:
            return context
        
        # Extract memory-relevant context
        filtered = {}
        
        # Spatial context (always relevant)
        if "position" in context:
            filtered["position"] = context["position"]
        if "spatial_data" in context:
            filtered["spatial_data"] = context["spatial_data"]
        
        # Environmental context (relevant for perception)
        environmental_keys = ["temperature", "lighting", "hazards", "time_remaining"]
        for key in environmental_keys:
            if key in context:
                filtered[key] = context[key]
        
        # Agent memory context (if available)
        memory_keys = ["agent_memory", "experience_level", "recent_actions", "trust_levels"]
        for key in memory_keys:
            if key in context:
                filtered[key] = context[key]
        
        # Mesa model state hash and step count (for consistency)
        if "mesa_state_hash" in context:
            filtered["mesa_state_hash"] = context["mesa_state_hash"]
        if "step_count" in context:
            filtered["step_count"] = context["step_count"]
        
        # Agent positions (relevant for spatial awareness)
        if "agent_positions" in context:
            filtered["agent_positions"] = context["agent_positions"]
        
        # Resources (relevant for planning)
        if "resources" in context:
            filtered["resources"] = context["resources"]
        
        return filtered
    
    def _evict_lru_entry(self) -> None:
        """Evict least recently used cache entry"""
        if not self._cache:
            return
        
        # Remove least recently used (first in OrderedDict)
        lru_key, _ = self._cache.popitem(last=False)
        self._statistics.eviction_count += 1
    
    def _maybe_cleanup_expired(self) -> None:
        """Cleanup expired entries if enough time has passed"""
        current_time = time.perf_counter()
        
        if current_time - self._last_cleanup_time > self._cleanup_interval:
            self.cleanup_expired()
    
    def _calculate_memory_similarity(self, context1: Dict[str, Any], 
                                   context2: Dict[str, Any]) -> float:
        """
        Calculate similarity between two memory contexts
        
        Args:
            context1: First context dictionary
            context2: Second context dictionary
            
        Returns:
            Similarity score between 0.0 and 1.0
        """
        # Extract memory-related fields
        memory1 = context1.get("agent_memory", [])
        memory2 = context2.get("agent_memory", [])
        
        if not memory1 and not memory2:
            return 1.0  # Both empty, perfectly similar
        
        if not memory1 or not memory2:
            return 0.0  # One empty, one not
        
        # Calculate Jaccard similarity for memory items
        set1 = set(memory1) if isinstance(memory1, list) else {memory1}
        set2 = set(memory2) if isinstance(memory2, list) else {memory2}
        
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        
        return intersection / union if union > 0 else 0.0