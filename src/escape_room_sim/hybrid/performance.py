"""
Mesa-CrewAI Hybrid Performance Optimization System

This module implements performance optimization strategies for the hybrid
architecture, addressing LLM latency, Mesa simulation bottlenecks, and
overall system throughput.

Key Strategies:
- Async/concurrent processing for LLM calls
- Intelligent caching and memoization
- Batching and pipelining optimizations
- Resource pooling and connection management
- Performance monitoring and adaptive optimization
"""

from typing import Dict, List, Any, Optional, Callable, Awaitable, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from abc import ABC, abstractmethod
import asyncio
import threading
import time
import weakref
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from functools import wraps, lru_cache
import hashlib
import json


class PerformanceMetric(Enum):
    """Types of performance metrics tracked"""
    LATENCY = "latency"                    # Operation duration
    THROUGHPUT = "throughput"              # Operations per second
    MEMORY_USAGE = "memory_usage"          # Memory consumption
    CPU_USAGE = "cpu_usage"               # CPU utilization
    CACHE_HIT_RATE = "cache_hit_rate"     # Cache effectiveness
    ERROR_RATE = "error_rate"             # Error frequency
    QUEUE_SIZE = "queue_size"             # Async queue sizes


class OptimizationStrategy(Enum):
    """Available optimization strategies"""
    CACHING = "caching"
    BATCHING = "batching"
    ASYNC_PROCESSING = "async_processing"
    CONNECTION_POOLING = "connection_pooling"
    PRECOMPUTATION = "precomputation"
    LAZY_LOADING = "lazy_loading"
    COMPRESSION = "compression"
    PARALLEL_PROCESSING = "parallel_processing"


@dataclass
class PerformanceProfile:
    """Performance profile for a specific operation"""
    operation_name: str
    avg_latency: float
    min_latency: float
    max_latency: float
    p95_latency: float
    throughput: float
    memory_usage: float
    error_rate: float
    sample_count: int
    last_updated: datetime


@dataclass
class OptimizationResult:
    """Result of applying an optimization"""
    strategy: OptimizationStrategy
    baseline_metric: float
    optimized_metric: float
    improvement_percent: float
    cost: float  # Resource cost of optimization
    applicable: bool
    message: str


class IPerformanceOptimizer(ABC):
    """Interface for performance optimization strategies"""
    
    @abstractmethod
    def can_optimize(self, profile: PerformanceProfile) -> bool:
        """Check if this optimizer can improve the operation"""
        pass
    
    @abstractmethod
    async def apply_optimization(self, operation: Callable, profile: PerformanceProfile) -> OptimizationResult:
        """Apply optimization to the operation"""
        pass
    
    @abstractmethod
    def get_strategy(self) -> OptimizationStrategy:
        """Get the optimization strategy this implements"""
        pass


class IntelligentCache:
    """
    Intelligent caching system with TTL, size limits, and usage tracking
    
    Architecture Decision: Smart caching with adaptive policies
    - TTL-based expiration for time-sensitive data
    - LRU eviction for memory management
    - Usage pattern tracking for cache optimization
    - Async-safe with proper locking
    """
    
    def __init__(self, max_size: int = 1000, default_ttl: float = 300.0):
        self.max_size = max_size
        self.default_ttl = default_ttl
        
        # Cache storage
        self._cache: Dict[str, Any] = {}
        self._timestamps: Dict[str, datetime] = {}
        self._ttls: Dict[str, float] = {}
        self._access_counts: Dict[str, int] = {}
        self._access_times: Dict[str, List[datetime]] = {}
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Performance metrics
        self.hits = 0
        self.misses = 0
        self.evictions = 0
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        with self._lock:
            if key not in self._cache:
                self.misses += 1
                return None
            
            # Check TTL
            if self._is_expired(key):
                self._remove_key(key)
                self.misses += 1
                return None
            
            # Update access tracking
            self._access_counts[key] = self._access_counts.get(key, 0) + 1
            if key not in self._access_times:
                self._access_times[key] = []
            self._access_times[key].append(datetime.now())
            
            # Keep only recent access times (last hour)
            cutoff = datetime.now() - timedelta(hours=1)
            self._access_times[key] = [t for t in self._access_times[key] if t > cutoff]
            
            self.hits += 1
            return self._cache[key]
    
    def set(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """Set value in cache"""
        with self._lock:
            # Ensure capacity
            if key not in self._cache and len(self._cache) >= self.max_size:
                self._evict_lru()
            
            # Store value
            self._cache[key] = value
            self._timestamps[key] = datetime.now()
            self._ttls[key] = ttl or self.default_ttl
            self._access_counts[key] = self._access_counts.get(key, 0)
            
            if key not in self._access_times:
                self._access_times[key] = []
    
    def invalidate(self, key: str) -> bool:
        """Remove key from cache"""
        with self._lock:
            if key in self._cache:
                self._remove_key(key)
                return True
            return False
    
    def clear(self) -> None:
        """Clear all cache entries"""
        with self._lock:
            self._cache.clear()
            self._timestamps.clear()
            self._ttls.clear()
            self._access_counts.clear()
            self._access_times.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics"""
        with self._lock:
            total_requests = self.hits + self.misses
            hit_rate = self.hits / max(1, total_requests)
            
            return {
                "size": len(self._cache),
                "max_size": self.max_size,
                "hits": self.hits,
                "misses": self.misses,
                "hit_rate": hit_rate,
                "evictions": self.evictions
            }
    
    def _is_expired(self, key: str) -> bool:
        """Check if cache entry is expired"""
        if key not in self._timestamps:
            return True
        
        age = (datetime.now() - self._timestamps[key]).total_seconds()
        ttl = self._ttls.get(key, self.default_ttl)
        return age > ttl
    
    def _evict_lru(self) -> None:
        """Evict least recently used item"""
        if not self._cache:
            return
        
        # Find LRU key based on access patterns
        lru_key = None
        lru_score = float('inf')
        
        for key in self._cache.keys():
            # Score based on recency and frequency
            access_count = self._access_counts.get(key, 0)
            last_access = max(self._access_times.get(key, [datetime.min]))
            age = (datetime.now() - last_access).total_seconds()
            
            # Lower score = more likely to evict
            score = access_count / max(1, age / 3600)  # Weighted by hours
            
            if score < lru_score:
                lru_score = score
                lru_key = key
        
        if lru_key:
            self._remove_key(lru_key)
            self.evictions += 1
    
    def _remove_key(self, key: str) -> None:
        """Remove key and all associated data"""
        self._cache.pop(key, None)
        self._timestamps.pop(key, None)
        self._ttls.pop(key, None)
        self._access_counts.pop(key, None)
        self._access_times.pop(key, None)


class AsyncBatchProcessor:
    """
    Batch processor for optimizing multiple similar operations
    
    Architecture Decision: Batching for efficiency
    - Collects multiple requests before processing
    - Reduces API call overhead for LLM requests
    - Configurable batch size and timeout
    - Maintains request ordering and error handling
    """
    
    def __init__(self, batch_size: int = 5, batch_timeout: float = 1.0):
        self.batch_size = batch_size
        self.batch_timeout = batch_timeout
        
        # Batch storage
        self._pending_requests: List[Dict[str, Any]] = []
        self._request_futures: List[asyncio.Future] = []
        
        # Processing state
        self._processing = False
        self._last_batch_time = datetime.now()
        
        # Performance tracking
        self.batches_processed = 0
        self.total_requests = 0
        self.avg_batch_size = 0.0
    
    async def add_request(self, request_data: Dict[str, Any]) -> Any:
        """Add request to batch and wait for result"""
        future = asyncio.Future()
        
        self._pending_requests.append(request_data)
        self._request_futures.append(future)
        self.total_requests += 1
        
        # Trigger processing if batch is full or timeout reached
        await self._maybe_process_batch()
        
        # Wait for result
        return await future
    
    async def _maybe_process_batch(self) -> None:
        """Process batch if conditions are met"""
        should_process = (
            len(self._pending_requests) >= self.batch_size or
            (self._pending_requests and 
             (datetime.now() - self._last_batch_time).total_seconds() >= self.batch_timeout)
        )
        
        if should_process and not self._processing:
            await self._process_current_batch()
    
    async def _process_current_batch(self) -> None:
        """Process current batch of requests"""
        if not self._pending_requests:
            return
        
        self._processing = True
        
        try:
            # Extract current batch
            batch_requests = self._pending_requests.copy()
            batch_futures = self._request_futures.copy()
            
            # Clear for next batch
            self._pending_requests.clear()
            self._request_futures.clear()
            
            # Process batch
            results = await self._process_batch_requests(batch_requests)
            
            # Resolve futures with results
            for i, (future, result) in enumerate(zip(batch_futures, results)):
                if isinstance(result, Exception):
                    future.set_exception(result)
                else:
                    future.set_result(result)
            
            # Update metrics
            self.batches_processed += 1
            self.avg_batch_size = ((self.avg_batch_size * (self.batches_processed - 1)) + len(batch_requests)) / self.batches_processed
            self._last_batch_time = datetime.now()
            
        finally:
            self._processing = False
    
    async def _process_batch_requests(self, requests: List[Dict[str, Any]]) -> List[Any]:
        """Process batch of requests - to be overridden by subclasses"""
        # Default implementation processes each request individually
        results = []
        for request in requests:
            try:
                # This would be overridden with actual batch processing logic
                result = await self._process_single_request(request)
                results.append(result)
            except Exception as e:
                results.append(e)
        
        return results
    
    async def _process_single_request(self, request: Dict[str, Any]) -> Any:
        """Process single request - to be overridden"""
        raise NotImplementedError("Subclasses must implement _process_single_request")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get batch processor statistics"""
        return {
            "batches_processed": self.batches_processed,
            "total_requests": self.total_requests,
            "avg_batch_size": self.avg_batch_size,
            "pending_requests": len(self._pending_requests),
            "is_processing": self._processing
        }


class LLMBatchProcessor(AsyncBatchProcessor):
    """Batch processor specifically for LLM requests"""
    
    def __init__(self, llm_client, batch_size: int = 3, batch_timeout: float = 0.5):
        super().__init__(batch_size, batch_timeout)
        self.llm_client = llm_client
    
    async def _process_batch_requests(self, requests: List[Dict[str, Any]]) -> List[Any]:
        """Process batch of LLM requests efficiently"""
        try:
            # Combine prompts for batch processing
            batch_prompts = [req.get('prompt', '') for req in requests]
            
            # Process as batch if LLM supports it, otherwise process concurrently
            if hasattr(self.llm_client, 'batch_complete'):
                results = await self.llm_client.batch_complete(batch_prompts)
            else:
                # Process concurrently
                tasks = [self.llm_client.complete(prompt) for prompt in batch_prompts]
                results = await asyncio.gather(*tasks, return_exceptions=True)
            
            return results
            
        except Exception as e:
            # Return exception for all requests in batch
            return [e] * len(requests)


class ConnectionPool:
    """
    Connection pool for managing external service connections
    
    Architecture Decision: Resource pooling for efficiency
    - Reuses connections to reduce overhead
    - Configurable pool size and connection lifecycle
    - Health checking and automatic reconnection
    - Thread-safe connection management
    """
    
    def __init__(self, create_connection: Callable, max_size: int = 10, 
                 max_idle_time: float = 300.0):
        self.create_connection = create_connection
        self.max_size = max_size
        self.max_idle_time = max_idle_time
        
        # Pool storage
        self._available_connections: List[Dict[str, Any]] = []
        self._in_use_connections: set = set()
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Metrics
        self.created_connections = 0
        self.destroyed_connections = 0
        self.pool_hits = 0
        self.pool_misses = 0
    
    async def acquire_connection(self):
        """Acquire connection from pool"""
        with self._lock:
            # Try to get available connection
            while self._available_connections:
                conn_info = self._available_connections.pop(0)
                connection = conn_info['connection']
                created_time = conn_info['created_time']
                
                # Check if connection is still valid
                age = (datetime.now() - created_time).total_seconds()
                if age < self.max_idle_time and self._is_connection_healthy(connection):
                    self._in_use_connections.add(id(connection))
                    self.pool_hits += 1
                    return connection
                else:
                    # Connection is stale, destroy it
                    await self._destroy_connection(connection)
            
            # No available connections, create new one if under limit
            if len(self._in_use_connections) < self.max_size:
                connection = await self.create_connection()
                self._in_use_connections.add(id(connection))
                self.created_connections += 1
                self.pool_misses += 1
                return connection
            
            # Pool exhausted
            raise Exception("Connection pool exhausted")
    
    async def release_connection(self, connection) -> None:
        """Release connection back to pool"""
        with self._lock:
            conn_id = id(connection)
            if conn_id in self._in_use_connections:
                self._in_use_connections.remove(conn_id)
                
                # Check if connection should be returned to pool
                if self._is_connection_healthy(connection):
                    self._available_connections.append({
                        'connection': connection,
                        'created_time': datetime.now()
                    })
                else:
                    await self._destroy_connection(connection)
    
    def _is_connection_healthy(self, connection) -> bool:
        """Check if connection is healthy"""
        # Basic health check - would be more sophisticated in practice
        return hasattr(connection, 'is_closed') and not connection.is_closed()
    
    async def _destroy_connection(self, connection) -> None:
        """Destroy connection and clean up resources"""
        try:
            if hasattr(connection, 'close'):
                await connection.close()
            self.destroyed_connections += 1
        except Exception:
            pass  # Ignore cleanup errors
    
    def get_stats(self) -> Dict[str, Any]:
        """Get connection pool statistics"""
        with self._lock:
            return {
                "available_connections": len(self._available_connections),
                "in_use_connections": len(self._in_use_connections),
                "created_connections": self.created_connections,
                "destroyed_connections": self.destroyed_connections,
                "pool_hits": self.pool_hits,
                "pool_misses": self.pool_misses,
                "hit_rate": self.pool_hits / max(1, self.pool_hits + self.pool_misses)
            }


class PerformanceMonitor:
    """
    Comprehensive performance monitoring system
    
    Architecture Decision: Continuous monitoring with adaptive optimization
    - Real-time performance metric collection
    - Trend analysis and anomaly detection
    - Automatic optimization recommendations
    - Historical performance data retention
    """
    
    def __init__(self, retention_period: int = 7200):  # 2 hours default
        self.retention_period = retention_period
        
        # Metric storage
        self._metrics: Dict[str, List[Dict[str, Any]]] = {}
        self._profiles: Dict[str, PerformanceProfile] = {}
        
        # Monitoring configuration
        self._monitoring_enabled = True
        self._sampling_rate = 1.0  # Sample 100% of operations
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Background cleanup task
        self._cleanup_task: Optional[asyncio.Task] = None
    
    def start_monitoring(self) -> None:
        """Start background monitoring tasks"""
        self._monitoring_enabled = True
        if not self._cleanup_task or self._cleanup_task.done():
            self._cleanup_task = asyncio.create_task(self._cleanup_old_metrics())
    
    def stop_monitoring(self) -> None:
        """Stop monitoring and cleanup"""
        self._monitoring_enabled = False
        if self._cleanup_task and not self._cleanup_task.done():
            self._cleanup_task.cancel()
    
    def record_operation(self, operation_name: str, duration: float, 
                        memory_usage: float = 0.0, error: bool = False,
                        context: Dict[str, Any] = None) -> None:
        """Record performance metrics for an operation"""
        if not self._monitoring_enabled:
            return
        
        # Sample operations based on sampling rate
        if self._sampling_rate < 1.0 and time.time() % 1 > self._sampling_rate:
            return
        
        with self._lock:
            if operation_name not in self._metrics:
                self._metrics[operation_name] = []
            
            # Record metric
            metric = {
                "timestamp": datetime.now(),
                "duration": duration,
                "memory_usage": memory_usage,
                "error": error,
                "context": context or {}
            }
            
            self._metrics[operation_name].append(metric)
            
            # Update performance profile
            self._update_profile(operation_name)
    
    def get_profile(self, operation_name: str) -> Optional[PerformanceProfile]:
        """Get performance profile for operation"""
        with self._lock:
            return self._profiles.get(operation_name)
    
    def get_all_profiles(self) -> Dict[str, PerformanceProfile]:
        """Get all performance profiles"""
        with self._lock:
            return self._profiles.copy()
    
    def get_metrics(self, operation_name: str, 
                   since: Optional[datetime] = None) -> List[Dict[str, Any]]:
        """Get raw metrics for operation"""
        with self._lock:
            if operation_name not in self._metrics:
                return []
            
            metrics = self._metrics[operation_name]
            if since:
                metrics = [m for m in metrics if m["timestamp"] >= since]
            
            return metrics
    
    def _update_profile(self, operation_name: str) -> None:
        """Update performance profile for operation"""
        metrics = self._metrics[operation_name]
        if not metrics:
            return
        
        # Calculate statistics from recent metrics (last hour)
        recent_cutoff = datetime.now() - timedelta(hours=1)
        recent_metrics = [m for m in metrics if m["timestamp"] >= recent_cutoff]
        
        if not recent_metrics:
            return
        
        durations = [m["duration"] for m in recent_metrics]
        memory_usages = [m["memory_usage"] for m in recent_metrics]
        errors = [m["error"] for m in recent_metrics]
        
        # Calculate profile statistics
        avg_latency = sum(durations) / len(durations)
        min_latency = min(durations)
        max_latency = max(durations)
        
        # Calculate 95th percentile
        sorted_durations = sorted(durations)
        p95_index = int(0.95 * len(sorted_durations))
        p95_latency = sorted_durations[p95_index] if sorted_durations else 0.0
        
        # Calculate throughput (operations per second)
        if len(recent_metrics) > 1:
            time_span = (recent_metrics[-1]["timestamp"] - recent_metrics[0]["timestamp"]).total_seconds()
            throughput = len(recent_metrics) / max(1, time_span)
        else:
            throughput = 0.0
        
        avg_memory = sum(memory_usages) / len(memory_usages) if memory_usages else 0.0
        error_rate = sum(errors) / len(errors)
        
        # Create/update profile
        self._profiles[operation_name] = PerformanceProfile(
            operation_name=operation_name,
            avg_latency=avg_latency,
            min_latency=min_latency,
            max_latency=max_latency,
            p95_latency=p95_latency,
            throughput=throughput,
            memory_usage=avg_memory,
            error_rate=error_rate,
            sample_count=len(recent_metrics),
            last_updated=datetime.now()
        )
    
    async def _cleanup_old_metrics(self) -> None:
        """Background task to clean up old metrics"""
        while self._monitoring_enabled:
            try:
                await asyncio.sleep(300)  # Cleanup every 5 minutes
                
                cutoff_time = datetime.now() - timedelta(seconds=self.retention_period)
                
                with self._lock:
                    for operation_name in list(self._metrics.keys()):
                        # Remove old metrics
                        self._metrics[operation_name] = [
                            m for m in self._metrics[operation_name]
                            if m["timestamp"] >= cutoff_time
                        ]
                        
                        # Remove empty metric lists
                        if not self._metrics[operation_name]:
                            del self._metrics[operation_name]
                            # Also remove the profile
                            self._profiles.pop(operation_name, None)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Error in metrics cleanup: {e}")
    
    def get_system_performance_summary(self) -> Dict[str, Any]:
        """Get overall system performance summary"""
        with self._lock:
            profiles = list(self._profiles.values())
            
            if not profiles:
                return {"status": "no_data"}
            
            # Aggregate statistics
            total_throughput = sum(p.throughput for p in profiles)
            avg_latency = sum(p.avg_latency for p in profiles) / len(profiles)
            max_latency = max(p.max_latency for p in profiles)
            avg_error_rate = sum(p.error_rate for p in profiles) / len(profiles)
            total_memory = sum(p.memory_usage for p in profiles)
            
            # Identify bottlenecks
            bottlenecks = [
                p for p in profiles 
                if p.avg_latency > avg_latency * 2 or p.error_rate > 0.1
            ]
            
            return {
                "status": "healthy" if avg_error_rate < 0.05 else "degraded",
                "total_operations": len(profiles),
                "total_throughput": total_throughput,
                "avg_latency": avg_latency,
                "max_latency": max_latency,
                "avg_error_rate": avg_error_rate,
                "total_memory_usage": total_memory,
                "bottlenecks": [b.operation_name for b in bottlenecks],
                "last_updated": datetime.now()
            }


class HybridPerformanceManager:
    """
    Central performance management system for the hybrid architecture
    
    Architecture Decision: Integrated performance optimization
    - Combines monitoring, caching, batching, and connection pooling
    - Adaptive optimization based on runtime performance
    - Pluggable optimization strategies
    - Comprehensive performance reporting
    """
    
    def __init__(self):
        # Core components
        self.monitor = PerformanceMonitor()
        self.cache = IntelligentCache(max_size=2000, default_ttl=600.0)
        self.llm_batch_processor = None  # Set when LLM client is available
        self.connection_pools: Dict[str, ConnectionPool] = {}
        
        # Optimization strategies
        self.optimizers: List[IPerformanceOptimizer] = []
        
        # Configuration
        self.auto_optimization_enabled = True
        self.optimization_interval = 300  # 5 minutes
        
        # Background tasks
        self._optimization_task: Optional[asyncio.Task] = None
    
    def initialize(self, llm_client=None) -> None:
        """Initialize performance manager with dependencies"""
        if llm_client:
            self.llm_batch_processor = LLMBatchProcessor(llm_client)
        
        # Register default optimizers
        self._register_default_optimizers()
        
        # Start monitoring
        self.monitor.start_monitoring()
        
        # Start auto-optimization if enabled
        if self.auto_optimization_enabled:
            self._optimization_task = asyncio.create_task(self._auto_optimization_loop())
    
    def shutdown(self) -> None:
        """Shutdown performance manager"""
        self.monitor.stop_monitoring()
        
        if self._optimization_task and not self._optimization_task.done():
            self._optimization_task.cancel()
    
    async def optimize_llm_call(self, prompt: str, **kwargs) -> Any:
        """Optimize LLM call with caching and batching"""
        # Create cache key
        cache_key = self._create_cache_key("llm_call", prompt, **kwargs)
        
        # Check cache first
        cached_result = self.cache.get(cache_key)
        if cached_result is not None:
            return cached_result
        
        # Use batch processor if available
        if self.llm_batch_processor:
            request_data = {"prompt": prompt, **kwargs}
            result = await self.llm_batch_processor.add_request(request_data)
        else:
            # Fallback to direct call
            # This would be implemented with actual LLM client
            result = f"Response to: {prompt[:50]}..."
        
        # Cache result
        self.cache.set(cache_key, result, ttl=300.0)  # 5-minute TTL for LLM results
        
        return result
    
    def create_connection_pool(self, name: str, create_connection: Callable,
                             max_size: int = 10) -> ConnectionPool:
        """Create and register connection pool"""
        pool = ConnectionPool(create_connection, max_size)
        self.connection_pools[name] = pool
        return pool
    
    def get_connection_pool(self, name: str) -> Optional[ConnectionPool]:
        """Get connection pool by name"""
        return self.connection_pools.get(name)
    
    def record_operation_performance(self, operation_name: str, duration: float,
                                   memory_usage: float = 0.0, error: bool = False,
                                   context: Dict[str, Any] = None) -> None:
        """Record performance metrics for an operation"""
        self.monitor.record_operation(operation_name, duration, memory_usage, error, context)
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report"""
        report = {
            "system_summary": self.monitor.get_system_performance_summary(),
            "cache_stats": self.cache.get_stats(),
            "operation_profiles": {name: profile.__dict__ for name, profile in self.monitor.get_all_profiles().items()}
        }
        
        # Add batch processor stats if available
        if self.llm_batch_processor:
            report["llm_batch_stats"] = self.llm_batch_processor.get_stats()
        
        # Add connection pool stats
        report["connection_pools"] = {
            name: pool.get_stats() for name, pool in self.connection_pools.items()
        }
        
        return report
    
    def _create_cache_key(self, operation: str, *args, **kwargs) -> str:
        """Create deterministic cache key"""
        key_data = {
            "operation": operation,
            "args": args,
            "kwargs": kwargs
        }
        
        # Create hash of key data
        key_json = json.dumps(key_data, sort_keys=True, default=str)
        return hashlib.md5(key_json.encode()).hexdigest()
    
    def _register_default_optimizers(self) -> None:
        """Register default optimization strategies"""
        # This would register concrete optimizer implementations
        pass
    
    async def _auto_optimization_loop(self) -> None:
        """Background loop for automatic optimization"""
        while self.auto_optimization_enabled:
            try:
                await asyncio.sleep(self.optimization_interval)
                await self._run_optimization_cycle()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Error in auto-optimization: {e}")
    
    async def _run_optimization_cycle(self) -> None:
        """Run one cycle of performance optimization"""
        profiles = self.monitor.get_all_profiles()
        
        for profile in profiles.values():
            # Check if operation needs optimization
            if self._needs_optimization(profile):
                await self._optimize_operation(profile)
    
    def _needs_optimization(self, profile: PerformanceProfile) -> bool:
        """Determine if operation needs optimization"""
        # Simple heuristics - could be more sophisticated
        return (
            profile.avg_latency > 5.0 or  # High latency
            profile.error_rate > 0.1 or   # High error rate
            profile.throughput < 1.0       # Low throughput
        )
    
    async def _optimize_operation(self, profile: PerformanceProfile) -> None:
        """Apply optimization to an operation"""
        for optimizer in self.optimizers:
            if optimizer.can_optimize(profile):
                try:
                    # This would apply the optimization
                    # Implementation depends on specific optimizer
                    result = await optimizer.apply_optimization(None, profile)
                    
                    if result.applicable and result.improvement_percent > 10:
                        print(f"Applied {result.strategy.value} optimization to {profile.operation_name}: "
                              f"{result.improvement_percent:.1f}% improvement")
                        break
                        
                except Exception as e:
                    print(f"Optimization failed for {profile.operation_name}: {e}")


# Performance decorators and utilities

def performance_monitored(operation_name: str = None):
    """Decorator to automatically monitor function performance"""
    def decorator(func):
        nonlocal operation_name
        if operation_name is None:
            operation_name = f"{func.__module__}.{func.__name__}"
        
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            error_occurred = False
            
            try:
                result = await func(*args, **kwargs)
                return result
            except Exception as e:
                error_occurred = True
                raise
            finally:
                duration = time.time() - start_time
                # This would record to a global performance manager instance
                # performance_manager.record_operation_performance(operation_name, duration, error=error_occurred)
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            start_time = time.time()
            error_occurred = False
            
            try:
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                error_occurred = True
                raise
            finally:
                duration = time.time() - start_time
                # performance_manager.record_operation_performance(operation_name, duration, error=error_occurred)
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    
    return decorator


def cached_result(ttl: float = 300.0, cache_instance: IntelligentCache = None):
    """Decorator to cache function results"""
    def decorator(func):
        cache = cache_instance or IntelligentCache()
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Create cache key
            key_data = {"func": func.__name__, "args": args, "kwargs": kwargs}
            cache_key = hashlib.md5(json.dumps(key_data, sort_keys=True, default=str).encode()).hexdigest()
            
            # Check cache
            result = cache.get(cache_key)
            if result is not None:
                return result
            
            # Compute and cache result
            result = func(*args, **kwargs)
            cache.set(cache_key, result, ttl)
            return result
        
        return wrapper
    return decorator