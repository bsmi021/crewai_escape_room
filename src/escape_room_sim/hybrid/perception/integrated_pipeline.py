"""
Integrated Perception Pipeline for Mesa-CrewAI Hybrid Architecture

This module implements a high-performance perception pipeline that combines
spatial indexing and caching systems for optimal perception extraction.

Key Features:
- Integration with SpatialIndex for fast spatial queries
- Integration with PerceptionCache for performance optimization
- Memory-based perception filtering with agent context awareness
- Performance monitoring and metrics collection
- Automatic synchronization with Mesa model state changes
- Handoff protocol for Decision Engine integration

Architecture:
- Uses SpatialIndex for O(1) agent/object lookup
- Uses PerceptionCache for redundant calculation avoidance
- Maintains Mesa model state hashing for cache invalidation
- Provides comprehensive performance metrics
- Supports configurable perception ranges and filtering
"""

from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import time
import hashlib
import json
import mesa

from src.escape_room_sim.hybrid.core_architecture import (
    PerceptionData, IPerceptionPipeline
)
from src.escape_room_sim.hybrid.spatial.spatial_index import SpatialIndex, SpatialObject
from src.escape_room_sim.hybrid.perception.perception_cache import PerceptionCache


@dataclass
class PerceptionPipelineConfig:
    """Configuration for integrated perception pipeline"""
    cache_size: int = 1000
    cache_ttl: float = 5.0
    spatial_grid_size: Tuple[int, int] = (100, 100)
    spatial_cell_size: float = 1.0
    enable_memory_filtering: bool = True
    memory_threshold: float = 0.8
    perception_range: float = 5.0
    communication_range: float = 3.0
    performance_monitoring: bool = True
    auto_cleanup_interval: float = 30.0


@dataclass
class PerformanceMetrics:
    """Performance metrics for perception pipeline"""
    total_extractions: int = 0
    total_extraction_time: float = 0.0
    cache_hit_count: int = 0
    cache_miss_count: int = 0
    spatial_query_count: int = 0
    spatial_query_time: float = 0.0
    agents_processed: int = 0
    
    @property
    def avg_extraction_time(self) -> float:
        """Average extraction time per operation"""
        if self.total_extractions == 0:
            return 0.0
        return self.total_extraction_time / self.total_extractions
    
    @property
    def cache_hit_rate(self) -> float:
        """Cache hit rate as percentage"""
        total_requests = self.cache_hit_count + self.cache_miss_count
        if total_requests == 0:
            return 0.0
        return self.cache_hit_count / total_requests
    
    @property
    def avg_spatial_query_time(self) -> float:
        """Average spatial query time"""
        if self.spatial_query_count == 0:
            return 0.0
        return self.spatial_query_time / self.spatial_query_count


@dataclass
class PerceptionHandoff:
    """Handoff data structure for Decision Engine integration"""
    perceptions: Dict[str, PerceptionData]
    performance_metrics: Dict[str, float]
    extraction_timestamp: datetime
    validation_passed: bool
    mesa_model_hash: Optional[str] = None
    agent_count: int = 0


class IntegratedPerceptionPipeline(IPerceptionPipeline):
    """
    High-performance integrated perception pipeline
    
    Combines spatial indexing and caching for optimal perception extraction
    with comprehensive performance monitoring and agent memory integration.
    
    Performance Characteristics:
    - Perception extraction: O(k * log n) where k is result size, n is agent count
    - Cache lookup: O(1) average case
    - Spatial queries: O(m) where m is objects in query range
    - Memory usage: O(n + c) where n is agents, c is cache size
    """
    
    def __init__(self, config: Optional[PerceptionPipelineConfig] = None):
        """
        Initialize integrated perception pipeline
        
        Args:
            config: Pipeline configuration, uses defaults if None
        """
        self.config = config or PerceptionPipelineConfig()
        
        # Initialize spatial indexing system
        self.spatial_index = SpatialIndex(
            grid_size=self.config.spatial_grid_size,
            cell_size=self.config.spatial_cell_size
        )
        
        # Initialize perception caching system
        self.perception_cache = PerceptionCache(
            max_size=self.config.cache_size,
            default_ttl=self.config.cache_ttl,
            enable_memory_filtering=self.config.enable_memory_filtering,
            memory_threshold=self.config.memory_threshold
        )
        
        # Performance monitoring
        self.performance_metrics = PerformanceMetrics()
        self._last_cleanup_time = time.perf_counter()
        self._last_model_hash: Optional[str] = None
        
        # State management
        self.is_initialized = True
    
    def extract_perceptions(self, mesa_model: mesa.Model) -> Dict[str, PerceptionData]:
        """
        Extract perceptions using integrated spatial indexing and caching
        
        Args:
            mesa_model: Mesa model to extract perceptions from
            
        Returns:
            Dictionary mapping agent IDs to perception data
        """
        extraction_start = time.perf_counter()
        
        try:
            # Update spatial index with current Mesa model state
            self._update_spatial_index(mesa_model)
            
            # Generate model state hash for cache key generation
            model_hash = self._generate_model_hash(mesa_model)
            
            # Extract perceptions for each agent
            perceptions = {}
            agents_processed = 0
            
            # Mesa 3.x compatibility: check for agents attribute directly
            if hasattr(mesa_model, 'agents'):
                for mesa_agent in mesa_model.agents:
                    agent_id = self._get_agent_id(mesa_agent)
                    if agent_id:
                        perception = self._extract_agent_perception_cached(
                            mesa_agent, mesa_model, model_hash
                        )
                        if perception:
                            perceptions[agent_id] = perception
                            agents_processed += 1
            
            # Update performance metrics
            extraction_time = time.perf_counter() - extraction_start
            self._update_performance_metrics(extraction_time, agents_processed)
            
            # Periodic cleanup
            self._maybe_perform_cleanup()
            
            return perceptions
            
        except Exception as e:
            # Handle errors gracefully
            extraction_time = time.perf_counter() - extraction_start
            self._update_performance_metrics(extraction_time, 0)
            
            # Return empty perceptions rather than crashing
            return {}
    
    def filter_perceptions(self, perceptions: Dict[str, PerceptionData], 
                         agent_id: str) -> PerceptionData:
        """
        Filter perceptions based on agent capabilities and memory
        
        Args:
            perceptions: All extracted perceptions
            agent_id: ID of agent to filter for
            
        Returns:
            Filtered perception data for the specified agent
        """
        if agent_id not in perceptions:
            return self._create_empty_perception(agent_id)
        
        base_perception = perceptions[agent_id]
        
        # Apply agent-specific filtering
        filtered_perception = self._apply_agent_specific_filtering(
            base_perception, agent_id
        )
        
        return filtered_perception
    
    def get_performance_metrics(self) -> PerformanceMetrics:
        """Get current performance metrics"""
        # Update cache metrics from cache
        cache_stats = self.perception_cache.get_statistics()
        self.performance_metrics.cache_hit_count = cache_stats.hit_count
        self.performance_metrics.cache_miss_count = cache_stats.miss_count
        
        return self.performance_metrics
    
    def create_handoff_for_decision_engine(self, 
                                         perceptions: Dict[str, PerceptionData]) -> PerceptionHandoff:
        """
        Create handoff data for Decision Engine (Agent B)
        
        Args:
            perceptions: Extracted perception data
            
        Returns:
            PerceptionHandoff for Decision Engine
        """
        # Validate perceptions meet Decision Engine requirements
        validation_passed = self._validate_perceptions_for_handoff(perceptions)
        
        # Gather performance metrics
        current_metrics = self.get_performance_metrics()
        performance_data = {
            "extraction_time": current_metrics.avg_extraction_time,
            "cache_hit_rate": current_metrics.cache_hit_rate,
            "spatial_query_time": current_metrics.avg_spatial_query_time,
            "agents_processed": len(perceptions),
            "total_extractions": current_metrics.total_extractions
        }
        
        return PerceptionHandoff(
            perceptions=perceptions,
            performance_metrics=performance_data,
            extraction_timestamp=datetime.now(),
            validation_passed=validation_passed,
            mesa_model_hash=self._last_model_hash,
            agent_count=len(perceptions)
        )
    
    def reset_performance_metrics(self) -> None:
        """Reset performance metrics"""
        self.performance_metrics = PerformanceMetrics()
        self.perception_cache.reset_statistics()
    
    def cleanup_resources(self) -> None:
        """Cleanup resources and perform maintenance"""
        # Cleanup expired cache entries
        expired_count = self.perception_cache.cleanup_expired()
        
        # Get spatial index statistics for optimization
        spatial_stats = self.spatial_index.get_statistics()
        
        # Log cleanup results if performance monitoring enabled
        if self.config.performance_monitoring:
            print(f"Cleaned up {expired_count} expired cache entries")
            print(f"Spatial index: {spatial_stats['object_count']} objects, "
                  f"{spatial_stats['cache_size']} cached queries")
    
    # Private methods
    
    def _update_spatial_index(self, mesa_model: mesa.Model) -> None:
        """Update spatial index with current Mesa model state"""
        spatial_start = time.perf_counter()
        
        try:
            # Rebuild or sync spatial index
            if self.spatial_index.is_empty:
                self.spatial_index.build_from_mesa_model(mesa_model)
            else:
                self.spatial_index.sync_with_mesa_model(mesa_model)
            
            # Update spatial query metrics
            spatial_time = time.perf_counter() - spatial_start
            self.performance_metrics.spatial_query_time += spatial_time
            self.performance_metrics.spatial_query_count += 1
            
        except Exception:
            # Handle spatial index errors gracefully
            self.spatial_index.clear()
            if hasattr(mesa_model, 'schedule'):
                self.spatial_index.build_from_mesa_model(mesa_model)
    
    def _extract_agent_perception_cached(self, 
                                       mesa_agent: mesa.Agent,
                                       mesa_model: mesa.Model,
                                       model_hash: str) -> Optional[PerceptionData]:
        """Extract agent perception with caching"""
        agent_id = self._get_agent_id(mesa_agent)
        
        # Create cache context
        cache_context = self._create_cache_context(mesa_agent, mesa_model, model_hash)
        cache_key = self.perception_cache.create_cache_key(agent_id, cache_context)
        
        # Try cache first
        cached_perception = self.perception_cache.get(cache_key)
        if cached_perception:
            return cached_perception
        
        # Cache miss - extract fresh perception
        perception = self._extract_agent_perception_fresh(mesa_agent, mesa_model)
        
        # Cache the result
        if perception:
            self.perception_cache.put(cache_key, perception)
        
        return perception
    
    def _extract_agent_perception_fresh(self, 
                                      mesa_agent: mesa.Agent,
                                      mesa_model: mesa.Model) -> Optional[PerceptionData]:
        """Extract fresh agent perception without caching"""
        try:
            agent_id = self._get_agent_id(mesa_agent)
            agent_pos = getattr(mesa_agent, 'pos', (0, 0))
            perception_range = getattr(mesa_agent, 'perception_range', self.config.perception_range)
            
            # Extract spatial data using spatial index
            spatial_data = self._extract_spatial_data_optimized(
                agent_pos, perception_range, mesa_agent, mesa_model
            )
            
            # Extract environmental data
            environmental_data = self._extract_environmental_data(mesa_agent, mesa_model)
            
            # Extract social data using spatial queries
            social_data = self._extract_social_data_optimized(
                agent_pos, perception_range, mesa_agent, mesa_model
            )
            
            # Extract resource data
            resource_data = self._extract_resource_data(mesa_agent, mesa_model)
            
            # Determine available actions
            available_actions = self._determine_available_actions(mesa_agent, mesa_model)
            
            # Extract constraints
            constraints = self._extract_constraints(mesa_agent, mesa_model)
            
            return PerceptionData(
                agent_id=agent_id,
                timestamp=datetime.now(),
                spatial_data=spatial_data,
                environmental_state=environmental_data,
                nearby_agents=social_data.get('visible_agents', []),
                available_actions=available_actions,
                resources=resource_data,
                constraints=constraints
            )
            
        except Exception:
            # Return None on error - will be handled by calling function
            return None
    
    def _extract_spatial_data_optimized(self, 
                                      agent_pos: Tuple[float, float],
                                      perception_range: float,
                                      mesa_agent: mesa.Agent,
                                      mesa_model: mesa.Model) -> Dict[str, Any]:
        """Extract spatial data using optimized spatial queries"""
        from src.escape_room_sim.hybrid.spatial.spatial_index import SpatialRange
        
        # Use spatial index for fast range queries
        range_query = SpatialRange(center=agent_pos, radius=perception_range)
        nearby_objects_result = self.spatial_index.find_objects_in_range(range_query)
        
        # Convert spatial objects to perception format
        nearby_objects = {}
        distances = {}
        
        for spatial_obj in nearby_objects_result.objects:
            if spatial_obj.object_id != self._get_agent_id(mesa_agent):  # Exclude self
                nearby_objects[spatial_obj.object_id] = spatial_obj.position
                distance = ((agent_pos[0] - spatial_obj.position[0]) ** 2 + 
                          (agent_pos[1] - spatial_obj.position[1]) ** 2) ** 0.5
                distances[spatial_obj.object_id] = distance
        
        # Calculate movement options
        movement_options = self._calculate_movement_options_optimized(
            agent_pos, mesa_model
        )
        
        # Identify obstacles using spatial queries
        obstacles = self._identify_obstacles_optimized(agent_pos, perception_range)
        
        return {
            "current_position": agent_pos,
            "visible_area": self._calculate_visible_area(agent_pos, perception_range),
            "nearby_objects": nearby_objects,
            "movement_options": movement_options,
            "obstacles": obstacles,
            "distances": distances,
            "perception_range": perception_range,
            "spatial_query_time": nearby_objects_result.query_time
        }
    
    def _extract_social_data_optimized(self,
                                     agent_pos: Tuple[float, float],
                                     perception_range: float,
                                     mesa_agent: mesa.Agent,
                                     mesa_model: mesa.Model) -> Dict[str, Any]:
        """Extract social data using optimized spatial queries"""
        from src.escape_room_sim.hybrid.spatial.spatial_index import SpatialRange
        
        # Find nearby agents using spatial index
        range_query = SpatialRange(center=agent_pos, radius=perception_range)
        nearby_result = self.spatial_index.find_objects_in_range(
            range_query, object_type="agent"
        )
        
        visible_agents = []
        agent_positions = {}
        agent_activities = {}
        
        agent_id = self._get_agent_id(mesa_agent)
        
        for spatial_obj in nearby_result.objects:
            if spatial_obj.object_id != agent_id:  # Exclude self
                visible_agents.append(spatial_obj.object_id)
                agent_positions[spatial_obj.object_id] = spatial_obj.position
                agent_activities[spatial_obj.object_id] = spatial_obj.properties.get(
                    'current_activity', 'unknown'
                )
        
        return {
            "visible_agents": visible_agents,
            "agent_positions": agent_positions,
            "agent_activities": agent_activities,
            "trust_levels": self._get_trust_levels(mesa_agent, mesa_model),
            "communication_history": self._get_recent_communications(mesa_agent, mesa_model),
            "social_query_time": nearby_result.query_time
        }
    
    def _calculate_movement_options_optimized(self,
                                            agent_pos: Tuple[float, float],
                                            mesa_model: mesa.Model) -> List[Tuple[float, float]]:
        """Calculate movement options using spatial index"""
        options = []
        x, y = agent_pos
        
        # Check cardinal directions
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            new_pos = (x + dx, y + dy)
            if (self._is_valid_position(new_pos, mesa_model) and 
                not self._is_blocked_optimized(new_pos)):
                options.append(new_pos)
        
        return options
    
    def _identify_obstacles_optimized(self,
                                    agent_pos: Tuple[float, float],
                                    search_range: float) -> List[Tuple[float, float]]:
        """Identify obstacles using spatial index"""
        from src.escape_room_sim.hybrid.spatial.spatial_index import SpatialRange
        
        # Find objects that might be obstacles
        range_query = SpatialRange(center=agent_pos, radius=search_range)
        objects_result = self.spatial_index.find_objects_in_range(range_query)
        
        obstacles = []
        for spatial_obj in objects_result.objects:
            # Check if object blocks movement
            if spatial_obj.properties.get('blocks_movement', False):
                obstacles.append(spatial_obj.position)
        
        return obstacles
    
    def _is_blocked_optimized(self, position: Tuple[float, float]) -> bool:
        """Check if position is blocked using spatial index"""
        from src.escape_room_sim.hybrid.spatial.spatial_index import SpatialRange
        
        # Query objects at exact position
        range_query = SpatialRange(center=position, radius=0.1)
        objects_result = self.spatial_index.find_objects_in_range(range_query)
        
        # Check if any object blocks movement
        for spatial_obj in objects_result.objects:
            if spatial_obj.properties.get('blocks_movement', False):
                return True
        
        return False
    
    def _generate_model_hash(self, mesa_model: mesa.Model) -> str:
        """Generate hash of Mesa model state for cache invalidation"""
        try:
            # Collect relevant model state
            state_data = {
                "step_count": getattr(mesa_model, 'step_count', 0),
                "agent_count": len(mesa_model.schedule.agents) if hasattr(mesa_model, 'schedule') else 0,
            }
            
            # Add agent positions if available
            if hasattr(mesa_model, 'schedule') and hasattr(mesa_model.schedule, 'agents'):
                agent_positions = {}
                for agent in mesa_model.schedule.agents:
                    agent_id = self._get_agent_id(agent)
                    pos = getattr(agent, 'pos', None)
                    if agent_id and pos:
                        agent_positions[agent_id] = pos
                state_data["agent_positions"] = agent_positions
            
            # Generate hash
            state_str = json.dumps(state_data, sort_keys=True, default=str)
            model_hash = hashlib.md5(state_str.encode()).hexdigest()
            self._last_model_hash = model_hash
            
            return model_hash
            
        except Exception:
            # Fallback to timestamp-based hash
            import time
            return str(int(time.time() * 1000))
    
    def _create_cache_context(self, 
                            mesa_agent: mesa.Agent,
                            mesa_model: mesa.Model,
                            model_hash: str) -> Dict[str, Any]:
        """Create cache context for cache key generation"""
        context = {
            "mesa_state_hash": model_hash,
            "position": getattr(mesa_agent, 'pos', (0, 0)),
            "perception_range": getattr(mesa_agent, 'perception_range', self.config.perception_range),
        }
        
        # Add agent memory if available
        if hasattr(mesa_agent, 'agent_memory'):
            context["agent_memory"] = getattr(mesa_agent, 'agent_memory', [])
        
        # Add experience level if available
        if hasattr(mesa_agent, 'experience_level'):
            context["experience_level"] = getattr(mesa_agent, 'experience_level', 'medium')
        
        return context
    
    def _apply_agent_specific_filtering(self,
                                      perception: PerceptionData,
                                      agent_id: str) -> PerceptionData:
        """Apply agent-specific perception filtering"""
        # Enhanced filtering based on agent type
        if 'strategist' in agent_id.lower():
            # Strategist gets enhanced analytical data
            enhanced_constraints = perception.constraints.copy()
            enhanced_constraints['analysis_capability'] = 'advanced'
            enhanced_constraints['pattern_recognition'] = True
            
            enhanced_actions = perception.available_actions + ['analyze_pattern', 'assess_risk']
            enhanced_actions = list(set(enhanced_actions))  # Remove duplicates
            
            return PerceptionData(
                agent_id=perception.agent_id,
                timestamp=perception.timestamp,
                spatial_data=perception.spatial_data,
                environmental_state=perception.environmental_state,
                nearby_agents=perception.nearby_agents,
                available_actions=enhanced_actions,
                resources=perception.resources,
                constraints=enhanced_constraints
            )
        
        elif 'mediator' in agent_id.lower():
            # Mediator gets enhanced social awareness
            enhanced_spatial = perception.spatial_data.copy()
            enhanced_spatial['team_dynamics_visible'] = True
            enhanced_spatial['communication_channels'] = perception.nearby_agents
            
            enhanced_actions = perception.available_actions + ['coordinate_team', 'mediate_conflict']
            enhanced_actions = list(set(enhanced_actions))
            
            return PerceptionData(
                agent_id=perception.agent_id,
                timestamp=perception.timestamp,
                spatial_data=enhanced_spatial,
                environmental_state=perception.environmental_state,
                nearby_agents=perception.nearby_agents,
                available_actions=enhanced_actions,
                resources=perception.resources,
                constraints=perception.constraints
            )
        
        elif 'survivor' in agent_id.lower():
            # Survivor gets enhanced survival instincts
            enhanced_environmental = perception.environmental_state.copy()
            enhanced_environmental['threat_assessment'] = 'heightened'
            enhanced_environmental['escape_route_analysis'] = True
            enhanced_environmental['resource_scarcity_awareness'] = True
            
            enhanced_actions = perception.available_actions + ['survive', 'escape_attempt', 'hoard_resource']
            enhanced_actions = list(set(enhanced_actions))
            
            return PerceptionData(
                agent_id=perception.agent_id,
                timestamp=perception.timestamp,
                spatial_data=perception.spatial_data,
                environmental_state=enhanced_environmental,
                nearby_agents=perception.nearby_agents,
                available_actions=enhanced_actions,
                resources=perception.resources,
                constraints=perception.constraints
            )
        
        # Default: no filtering
        return perception
    
    def _validate_perceptions_for_handoff(self, 
                                        perceptions: Dict[str, PerceptionData]) -> bool:
        """Validate perceptions meet Decision Engine requirements"""
        try:
            for agent_id, perception in perceptions.items():
                # Check required fields
                if not perception.agent_id:
                    return False
                if not perception.timestamp:
                    return False
                if perception.spatial_data is None:
                    return False
                if perception.available_actions is None:
                    return False
                
                # Check spatial data completeness
                spatial_required = ["current_position", "nearby_objects", "movement_options"]
                for field in spatial_required:
                    if field not in perception.spatial_data:
                        return False
            
            return True
            
        except Exception:
            return False
    
    def _update_performance_metrics(self, extraction_time: float, agents_processed: int) -> None:
        """Update performance metrics"""
        self.performance_metrics.total_extractions += 1
        self.performance_metrics.total_extraction_time += extraction_time
        self.performance_metrics.agents_processed = agents_processed
    
    def _maybe_perform_cleanup(self) -> None:
        """Perform cleanup if enough time has passed"""
        current_time = time.perf_counter()
        
        if current_time - self._last_cleanup_time > self.config.auto_cleanup_interval:
            self.cleanup_resources()
            self._last_cleanup_time = current_time
    
    # Helper methods (using existing implementations from data_flow.py)
    
    def _get_agent_id(self, mesa_agent: mesa.Agent) -> str:
        """Extract agent ID from Mesa agent"""
        if hasattr(mesa_agent, 'agent_id'):
            return mesa_agent.agent_id
        elif hasattr(mesa_agent, 'unique_id'):
            return str(mesa_agent.unique_id)
        else:
            return f"agent_{id(mesa_agent)}"
    
    def _create_empty_perception(self, agent_id: str) -> PerceptionData:
        """Create empty perception data"""
        return PerceptionData(
            agent_id=agent_id,
            timestamp=datetime.now(),
            spatial_data={},
            environmental_state={},
            nearby_agents=[],
            available_actions=[],
            resources={},
            constraints={}
        )
    
    def _extract_environmental_data(self, mesa_agent: mesa.Agent, 
                                  mesa_model: mesa.Model) -> Dict[str, Any]:
        """Extract environmental data (simplified implementation)"""
        return {
            "temperature": getattr(mesa_model, 'temperature', 20.0),
            "lighting": getattr(mesa_model, 'lighting', 1.0),
            "air_quality": getattr(mesa_model, 'air_quality', 1.0),
            "time_remaining": getattr(mesa_model, 'time_remaining', None),
            "hazards": []
        }
    
    def _extract_resource_data(self, mesa_agent: mesa.Agent,
                             mesa_model: mesa.Model) -> Dict[str, Any]:
        """Extract resource data (simplified implementation)"""
        return {
            "owned_resources": getattr(mesa_agent, 'resources', []),
            "available_resources": [],
            "resource_locations": {},
            "scarcity_levels": {}
        }
    
    def _determine_available_actions(self, mesa_agent: mesa.Agent,
                                   mesa_model: mesa.Model) -> List[str]:
        """Determine available actions (simplified implementation)"""
        return ["move", "examine", "communicate", "wait"]
    
    def _extract_constraints(self, mesa_agent: mesa.Agent,
                           mesa_model: mesa.Model) -> Dict[str, Any]:
        """Extract constraints (simplified implementation)"""
        return {
            "movement_blocked": False,
            "resource_limits": getattr(mesa_agent, 'max_resources', 10),
            "health_status": getattr(mesa_agent, 'health', 1.0),
            "energy_level": getattr(mesa_agent, 'energy', 1.0)
        }
    
    def _calculate_visible_area(self, pos: Tuple[float, float], 
                              vision_range: float) -> List[Tuple[float, float]]:
        """Calculate visible area (simplified implementation)"""
        visible = []
        x, y = pos
        for dx in range(-int(vision_range), int(vision_range) + 1):
            for dy in range(-int(vision_range), int(vision_range) + 1):
                if (dx * dx + dy * dy) <= vision_range * vision_range:
                    visible.append((x + dx, y + dy))
        return visible
    
    def _is_valid_position(self, pos: Tuple[float, float], mesa_model: mesa.Model) -> bool:
        """Check if position is valid (simplified implementation)"""
        # Basic bounds checking
        x, y = pos
        return 0 <= x < 100 and 0 <= y < 100
    
    def _get_trust_levels(self, mesa_agent: mesa.Agent, mesa_model: mesa.Model) -> Dict[str, float]:
        """Get trust levels (simplified implementation)"""
        return {}
    
    def _get_recent_communications(self, mesa_agent: mesa.Agent, 
                                 mesa_model: mesa.Model) -> List[Dict[str, Any]]:
        """Get recent communications (simplified implementation)"""
        return []