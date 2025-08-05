"""
Unified State Management Package

Agent D: State Management & Integration Specialist
Event-driven state synchronization between Mesa and CrewAI frameworks.
"""

from .unified_state_manager import (
    UnifiedStateManager, AsyncUnifiedStateManager,
    StateChange, StateConflict, StateResolution, UnifiedState
)
from .event_bus import EventBus, StateChangeEvent
from .state_synchronizer import StateSynchronizer

__all__ = [
    'UnifiedStateManager', 'AsyncUnifiedStateManager',
    'StateChange', 'StateConflict', 'StateResolution', 'UnifiedState',
    'EventBus', 'StateChangeEvent',
    'StateSynchronizer'
]