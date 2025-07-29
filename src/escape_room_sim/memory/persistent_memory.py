"""
Persistent Memory Management for CrewAI Iterative Simulation.

This module provides comprehensive memory management capabilities that enable
agents to learn from previous iterations and improve their strategies over time.
Implements both file-based persistence and in-memory tracking for optimal performance.
"""

import json
import os
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict, field
from collections import defaultdict
import uuid


@dataclass
class MemoryEntry:
    """Represents a single memory entry with metadata."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    iteration_number: int = 0
    memory_type: str = ""  # strategy, relationship, lesson, event
    content: str = ""
    tags: List[str] = field(default_factory=list)
    importance: float = 0.5  # 0.0 to 1.0
    agent_source: str = ""  # Which agent generated this memory
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StrategyMemory:
    """Tracks strategy attempts and their outcomes."""
    strategy_id: str
    description: str
    iteration_attempted: int
    agents_involved: List[str]
    resources_required: List[str]
    outcome: str  # success, failure, partial_success
    effectiveness_score: float  # 0.0 to 1.0
    lessons_learned: List[str]
    would_retry: bool = False


@dataclass
class RelationshipMemory:
    """Tracks interpersonal dynamics and trust levels."""
    agent_a: str
    agent_b: str
    interaction_type: str  # collaboration, conflict, negotiation
    iteration_occurred: int
    trust_change: float  # -1.0 to 1.0
    description: str
    resolution: Optional[str] = None


class IterativeMemoryManager:
    """
    Comprehensive memory management system for the iterative escape room simulation.
    
    This class provides:
    - Persistent storage of iteration history
    - Strategy effectiveness tracking
    - Relationship dynamics monitoring
    - Learning pattern analysis
    - Context-aware memory retrieval
    """
    
    def __init__(self, data_dir: str = "data"):
        """
        Initialize the memory manager.
        
        Args:
            data_dir: Directory to store persistent memory files
        """
        self.data_dir = data_dir
        self.memory_entries: List[MemoryEntry] = []
        self.strategy_memories: Dict[str, StrategyMemory] = {}
        self.relationship_memories: List[RelationshipMemory] = []
        
        # Performance tracking
        self.iteration_performance: Dict[int, Dict[str, Any]] = {}
        self.agent_performance: Dict[str, Dict[str, Any]] = defaultdict(dict)
        
        # Ensure data directory exists
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Load existing memories
        self._load_existing_memories()
        
        print(f"🧠 Memory Manager initialized with {len(self.memory_entries)} existing memories")
    
    def record_iteration(self, iteration_num: int, strategies_tried: List[str], 
                        outcomes: List[str], consensus_reached: bool = False):
        """
        Record a complete iteration with all its components.
        
        Args:
            iteration_num: The iteration number
            strategies_tried: List of strategies attempted
            outcomes: List of outcomes/results
            consensus_reached: Whether the agents reached consensus
        """
        # Record iteration performance
        self.iteration_performance[iteration_num] = {
            "strategies_count": len(strategies_tried),
            "outcomes_count": len(outcomes),
            "consensus": consensus_reached,
            "timestamp": datetime.now().isoformat()
        }
        
        # Record strategy memories
        for i, (strategy, outcome) in enumerate(zip(strategies_tried, outcomes)):
            if strategy.strip():  # Only record non-empty strategies
                self._record_strategy_memory(
                    strategy=strategy,
                    outcome=outcome,
                    iteration=iteration_num,
                    position=i
                )
        
        # Record general iteration memory
        iteration_memory = MemoryEntry(
            iteration_number=iteration_num,
            memory_type="iteration_summary",
            content=f"Iteration {iteration_num}: {len(strategies_tried)} strategies tried, consensus: {consensus_reached}",
            tags=["iteration", "summary", "strategies"],
            importance=0.8,
            agent_source="system",
            context={
                "strategies": strategies_tried,
                "outcomes": outcomes,
                "consensus": consensus_reached
            }
        )
        
        self.memory_entries.append(iteration_memory)
        print(f"📝 Recorded iteration {iteration_num} with {len(strategies_tried)} strategies")
    
    def _record_strategy_memory(self, strategy: str, outcome: str, 
                               iteration: int, position: int):
        """Record a specific strategy and its outcome."""
        # Determine effectiveness score based on outcome
        effectiveness = self._calculate_effectiveness_score(outcome)
        
        # Determine if strategy was successful
        success_indicators = ["success", "worked", "effective", "breakthrough", "solved"]
        failure_indicators = ["failed", "didn't work", "unsuccessful", "blocked", "impossible"]
        
        outcome_lower = outcome.lower()
        if any(indicator in outcome_lower for indicator in success_indicators):
            outcome_category = "success"
        elif any(indicator in outcome_lower for indicator in failure_indicators):
            outcome_category = "failure"
        else:
            outcome_category = "partial_success"
        
        strategy_id = f"iter_{iteration}_strat_{position}"
        
        strategy_memory = StrategyMemory(
            strategy_id=strategy_id,
            description=strategy,
            iteration_attempted=iteration,
            agents_involved=["Strategist", "Mediator", "Survivor"],  # All agents involved by default
            resources_required=self._extract_resources_from_strategy(strategy),
            outcome=outcome_category,
            effectiveness_score=effectiveness,
            lessons_learned=self._extract_lessons_from_outcome(outcome),
            would_retry=(effectiveness > 0.6)
        )
        
        self.strategy_memories[strategy_id] = strategy_memory
        
        # Also create a memory entry
        memory_entry = MemoryEntry(
            iteration_number=iteration,
            memory_type="strategy",
            content=f"Strategy: {strategy} | Outcome: {outcome}",
            tags=["strategy", outcome_category, f"iteration_{iteration}"],
            importance=effectiveness,
            agent_source="collective",
            context={
                "strategy_id": strategy_id,
                "effectiveness": effectiveness,
                "outcome_category": outcome_category
            }
        )
        
        self.memory_entries.append(memory_entry)
    
    def _calculate_effectiveness_score(self, outcome: str) -> float:
        """Calculate effectiveness score from outcome description."""
        outcome_lower = outcome.lower()
        
        # Positive indicators
        positive_words = ["success", "worked", "effective", "breakthrough", "solved", 
                         "progress", "discovered", "achieved", "completed"]
        
        # Negative indicators  
        negative_words = ["failed", "unsuccessful", "blocked", "impossible", "error",
                         "stuck", "deadlock", "wasted", "useless"]
        
        positive_count = sum(1 for word in positive_words if word in outcome_lower)
        negative_count = sum(1 for word in negative_words if word in outcome_lower)
        
        # Base score
        if positive_count > negative_count:
            base_score = 0.7 + (positive_count * 0.1)
        elif negative_count > positive_count:
            base_score = 0.3 - (negative_count * 0.1)
        else:
            base_score = 0.5
        
        return max(0.0, min(1.0, base_score))
    
    def _extract_resources_from_strategy(self, strategy: str) -> List[str]:
        """Extract resource requirements from strategy description."""
        common_resources = ["key", "tool", "rope", "time", "energy", "teamwork", 
                           "knowledge", "communication", "trust", "leadership"]
        
        strategy_lower = strategy.lower()
        found_resources = [resource for resource in common_resources 
                          if resource in strategy_lower]
        
        return found_resources
    
    def _extract_lessons_from_outcome(self, outcome: str) -> List[str]:
        """Extract key lessons from outcome description."""
        lessons = []
        
        # Look for explicit lesson patterns
        lesson_patterns = [
            ("learned that", ""),
            ("discovered that", ""),
            ("realized that", ""),
            ("found that", ""),
            ("important to", "It's important to"),
            ("need to", "We need to"),
            ("should", "We should"),
            ("must", "We must")
        ]
        
        outcome_sentences = outcome.split('.')
        for sentence in outcome_sentences:
            sentence = sentence.strip()
            if len(sentence) > 10:  # Only meaningful sentences
                for pattern, prefix in lesson_patterns:
                    if pattern in sentence.lower():
                        lesson = f"{prefix} {sentence}".strip()
                        if lesson and lesson not in lessons:
                            lessons.append(lesson)
        
        return lessons[:3]  # Limit to 3 lessons per outcome
    
    def record_relationship_event(self, agent_a: str, agent_b: str, 
                                 interaction_type: str, trust_change: float,
                                 description: str, iteration: int):
        """Record an interpersonal interaction between agents."""
        relationship_memory = RelationshipMemory(
            agent_a=agent_a,
            agent_b=agent_b,
            interaction_type=interaction_type,
            iteration_occurred=iteration,
            trust_change=trust_change,
            description=description
        )
        
        self.relationship_memories.append(relationship_memory)
        
        # Also create a memory entry
        memory_entry = MemoryEntry(
            iteration_number=iteration,
            memory_type="relationship",
            content=f"{agent_a} & {agent_b}: {description} (trust change: {trust_change:+.1f})",
            tags=["relationship", interaction_type, agent_a.lower(), agent_b.lower()],
            importance=abs(trust_change),  # More significant changes are more important
            agent_source="system",
            context={
                "agent_a": agent_a,
                "agent_b": agent_b,
                "trust_change": trust_change,
                "interaction_type": interaction_type
            }
        )
        
        self.memory_entries.append(memory_entry)
        print(f"👥 Recorded relationship event: {agent_a} & {agent_b} - {interaction_type}")
    
    def get_failed_strategies(self, limit: int = 10) -> List[str]:
        """Get list of previously failed strategies to avoid repeating."""
        failed_strategies = []
        
        for strategy_memory in self.strategy_memories.values():
            if strategy_memory.outcome == "failure" and not strategy_memory.would_retry:
                failed_strategies.append(strategy_memory.description)
        
        # Sort by iteration (most recent first) and limit
        failed_strategies.sort(key=lambda x: self._get_strategy_iteration(x), reverse=True)
        return failed_strategies[:limit]
    
    def get_successful_strategies(self, limit: int = 10) -> List[str]:
        """Get list of previously successful strategies that could be adapted."""
        successful_strategies = []
        
        for strategy_memory in self.strategy_memories.values():
            if strategy_memory.outcome == "success" or strategy_memory.effectiveness_score > 0.7:
                successful_strategies.append(strategy_memory.description)
        
        # Sort by effectiveness score (best first) and limit
        successful_strategies.sort(
            key=lambda x: self._get_strategy_effectiveness(x), 
            reverse=True
        )
        return successful_strategies[:limit]
    
    def _get_strategy_iteration(self, strategy_desc: str) -> int:
        """Get iteration number for a strategy description."""
        for strategy_memory in self.strategy_memories.values():
            if strategy_memory.description == strategy_desc:
                return strategy_memory.iteration_attempted
        return 0
    
    def _get_strategy_effectiveness(self, strategy_desc: str) -> float:
        """Get effectiveness score for a strategy description."""
        for strategy_memory in self.strategy_memories.values():
            if strategy_memory.description == strategy_desc:
                return strategy_memory.effectiveness_score
        return 0.0
    
    def get_interpersonal_conflicts(self, limit: int = 5) -> List[str]:
        """Get list of previous interpersonal conflicts."""
        conflicts = []
        
        for rel_memory in self.relationship_memories:
            if rel_memory.interaction_type == "conflict" or rel_memory.trust_change < -0.2:
                conflict_desc = f"{rel_memory.agent_a} vs {rel_memory.agent_b}: {rel_memory.description}"
                conflicts.append(conflict_desc)
        
        # Sort by iteration (most recent first) and limit
        conflicts.sort(key=lambda x: self._get_conflict_iteration(x), reverse=True)
        return conflicts[:limit]
    
    def _get_conflict_iteration(self, conflict_desc: str) -> int:
        """Get iteration number for a conflict description."""
        for rel_memory in self.relationship_memories:
            if rel_memory.description in conflict_desc:
                return rel_memory.iteration_occurred
        return 0
    
    def get_learning_insights(self) -> Dict[str, Any]:
        """Generate insights about learning patterns and progress."""
        if not self.iteration_performance:
            return {"status": "No iterations recorded yet"}
        
        # Calculate learning trends
        iterations = sorted(self.iteration_performance.keys())
        consensus_trend = [self.iteration_performance[i]["consensus"] for i in iterations]
        strategy_counts = [self.iteration_performance[i]["strategies_count"] for i in iterations]
        
        # Strategy effectiveness over time
        strategy_effectiveness = []
        for iteration in iterations:
            iter_strategies = [s for s in self.strategy_memories.values() 
                             if s.iteration_attempted == iteration]
            if iter_strategies:
                avg_effectiveness = sum(s.effectiveness_score for s in iter_strategies) / len(iter_strategies)
                strategy_effectiveness.append(avg_effectiveness)
            else:
                strategy_effectiveness.append(0.5)
        
        insights = {
            "total_iterations": len(iterations),
            "total_strategies_tried": sum(strategy_counts),
            "total_memories": len(self.memory_entries),
            "learning_trends": {
                "consensus_rate_latest": sum(consensus_trend[-3:]) / min(3, len(consensus_trend)) if consensus_trend else 0,
                "strategy_effectiveness_trend": strategy_effectiveness[-3:] if len(strategy_effectiveness) >= 3 else strategy_effectiveness,
                "strategies_per_iteration_trend": strategy_counts[-3:] if len(strategy_counts) >= 3 else strategy_counts
            },
            "memory_distribution": {
                "strategies": len(self.strategy_memories),
                "relationships": len(self.relationship_memories),
                "total_entries": len(self.memory_entries)
            },
            "top_lessons_learned": self._get_top_lessons(5)
        }
        
        return insights
    
    def _get_top_lessons(self, limit: int) -> List[str]:
        """Get the most important lessons learned."""
        lesson_memories = [m for m in self.memory_entries 
                          if "lesson" in m.tags or "learned" in m.content.lower()]
        
        # Sort by importance and recency
        lesson_memories.sort(key=lambda m: (m.importance, m.iteration_number), reverse=True)
        
        return [m.content for m in lesson_memories[:limit]]
    
    def get_context_for_agent(self, agent_name: str, iteration: int) -> str:
        """
        Generate contextual memory information for a specific agent.
        
        Args:
            agent_name: Name of the agent requesting context
            iteration: Current iteration number
            
        Returns:
            Formatted context string with relevant memories
        """
        # Get relevant memories for this agent
        agent_memories = [m for m in self.memory_entries 
                         if m.agent_source == agent_name.lower() or 
                         agent_name.lower() in m.tags or
                         m.agent_source == "collective"]
        
        # Sort by relevance (importance * recency)
        agent_memories.sort(
            key=lambda m: m.importance * (1 + (iteration - m.iteration_number) * 0.1),
            reverse=True
        )
        
        context = f"MEMORY CONTEXT FOR {agent_name.upper()} - ITERATION {iteration}:\n\n"
        
        # Recent successful strategies
        recent_successes = self.get_successful_strategies(3)
        if recent_successes:
            context += "SUCCESSFUL STRATEGIES TO BUILD UPON:\n"
            for i, strategy in enumerate(recent_successes, 1):
                context += f"{i}. {strategy}\n"
            context += "\n"
        
        # Strategies to avoid
        failures = self.get_failed_strategies(3)
        if failures:
            context += "FAILED STRATEGIES TO AVOID:\n"
            for i, strategy in enumerate(failures, 1):
                context += f"{i}. {strategy}\n"
            context += "\n"
        
        # Key lessons learned
        top_lessons = self._get_top_lessons(3)
        if top_lessons:
            context += "KEY LESSONS LEARNED:\n"
            for i, lesson in enumerate(top_lessons, 1):
                context += f"{i}. {lesson}\n"
            context += "\n"
        
        # Relationship insights (if applicable)
        if agent_name.lower() == "mediator":
            conflicts = self.get_interpersonal_conflicts(2)
            if conflicts:
                context += "INTERPERSONAL DYNAMICS TO MONITOR:\n"
                for i, conflict in enumerate(conflicts, 1):
                    context += f"{i}. {conflict}\n"
        
        return context
    
    def save_all_memories(self):
        """Save all memory data to persistent storage."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save memory entries
        memories_file = os.path.join(self.data_dir, f"memory_entries_{timestamp}.json")
        with open(memories_file, 'w') as f:
            json.dump([asdict(m) for m in self.memory_entries], f, indent=2)
        
        # Save strategy memories
        strategies_file = os.path.join(self.data_dir, f"strategy_memories_{timestamp}.json")
        with open(strategies_file, 'w') as f:
            json.dump({k: asdict(v) for k, v in self.strategy_memories.items()}, f, indent=2)
        
        # Save relationship memories
        relationships_file = os.path.join(self.data_dir, f"relationship_memories_{timestamp}.json")
        with open(relationships_file, 'w') as f:
            json.dump([asdict(r) for r in self.relationship_memories], f, indent=2)
        
        # Save performance data
        performance_file = os.path.join(self.data_dir, f"performance_data_{timestamp}.json")
        performance_data = {
            "iteration_performance": self.iteration_performance,
            "agent_performance": dict(self.agent_performance)
        }
        with open(performance_file, 'w') as f:
            json.dump(performance_data, f, indent=2)
        
        print(f"💾 All memory data saved with timestamp {timestamp}")
    
    def _load_existing_memories(self):
        """Load existing memory data from files."""
        try:
            # Find most recent memory files
            memory_files = [f for f in os.listdir(self.data_dir) if f.startswith("memory_entries_")]
            if memory_files:
                latest_file = max(memory_files)
                with open(os.path.join(self.data_dir, latest_file), 'r') as f:
                    memory_data = json.load(f)
                    self.memory_entries = [MemoryEntry(**m) for m in memory_data]
            
            # Load strategy memories
            strategy_files = [f for f in os.listdir(self.data_dir) if f.startswith("strategy_memories_")]
            if strategy_files:
                latest_file = max(strategy_files)
                with open(os.path.join(self.data_dir, latest_file), 'r') as f:
                    strategy_data = json.load(f)
                    self.strategy_memories = {k: StrategyMemory(**v) for k, v in strategy_data.items()}
            
            # Load relationship memories
            relationship_files = [f for f in os.listdir(self.data_dir) if f.startswith("relationship_memories_")]
            if relationship_files:
                latest_file = max(relationship_files)
                with open(os.path.join(self.data_dir, latest_file), 'r') as f:
                    relationship_data = json.load(f)
                    self.relationship_memories = [RelationshipMemory(**r) for r in relationship_data]
                    
        except Exception as e:
            print(f"⚠️  Warning: Could not load existing memories: {e}")
            # Continue with empty memories


# Example usage and testing
if __name__ == "__main__":
    # Test memory manager
    memory_manager = IterativeMemoryManager("test_data")
    
    # Test recording iteration
    memory_manager.record_iteration(
        iteration_num=1,
        strategies_tried=["Try to break down the door", "Look for hidden keys", "Examine the computer terminal"],
        outcomes=["Door too strong, wasted energy", "Found one key under debris", "Terminal requires password"],
        consensus_reached=True
    )
    
    # Test relationship recording
    memory_manager.record_relationship_event(
        agent_a="Strategist",
        agent_b="Survivor", 
        interaction_type="disagreement",
        trust_change=-0.2,
        description="Disagreed about resource allocation priority",
        iteration=1
    )
    
    # Test memory retrieval
    print("Failed Strategies:", memory_manager.get_failed_strategies())
    print("Successful Strategies:", memory_manager.get_successful_strategies())
    print("Learning Insights:", memory_manager.get_learning_insights())
    
    # Test agent context
    context = memory_manager.get_context_for_agent("Strategist", 2)
    print("\nStrategist Context:")
    print(context)
    
    # Save memories
    memory_manager.save_all_memories()