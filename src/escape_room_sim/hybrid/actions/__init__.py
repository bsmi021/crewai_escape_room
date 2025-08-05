"""
Mesa-CrewAI Hybrid Architecture - Action Translation System

Agent C: Action Translation & Execution Specialist

This module provides advanced action translation from CrewAI decisions to Mesa actions
with complex sequence orchestration and conflict resolution.
"""

from .action_translator import (
    AdvancedActionTranslator,
    ActionSequenceOrchestrator, 
    ConflictResolver,
    ActionValidator
)
from .action_models import (
    ActionSequence,
    ActionConflict,
    ConflictResolution,
    ExecutionResult
)

__all__ = [
    'AdvancedActionTranslator',
    'ActionSequenceOrchestrator',
    'ConflictResolver', 
    'ActionValidator',
    'ActionSequence',
    'ActionConflict',
    'ConflictResolution',
    'ExecutionResult'
]