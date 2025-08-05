"""
Spatial Systems for Mesa-CrewAI Hybrid Architecture

This module provides optimized spatial indexing and query capabilities
for efficient agent and object lookup in the hybrid simulation environment.

Key Components:
- SpatialIndex: High-performance spatial indexing system
- SpatialObject: Spatial representation of Mesa objects
- SpatialQueryResult: Query result container with performance metrics
- SpatialRange: Range query specification

Performance Targets:
- Perception extraction: < 100ms
- Cache hit rate: > 80%
- Spatial queries: < 50ms
"""

from .spatial_index import (
    SpatialIndex,
    SpatialObject, 
    SpatialQueryResult,
    SpatialRange
)

__all__ = [
    'SpatialIndex',
    'SpatialObject',
    'SpatialQueryResult', 
    'SpatialRange'
]