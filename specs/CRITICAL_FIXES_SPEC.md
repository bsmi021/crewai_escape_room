# Critical Fixes Specification
## CrewAI Escape Room Simulation - Phase 1 Requirements

### Document Information
- **Document Type**: Technical Specification
- **Priority**: CRITICAL - Must Complete Before Execution
- **Estimated Time**: 6-8 hours
- **Dependencies**: None
- **Author**: Analysis Subagents Report
- **Date**: 2025-08-04

---

## Overview

This specification addresses critical implementation gaps that prevent the CrewAI Escape Room Simulation from executing. These fixes are mandatory before any simulation runs can occur.

---

## 1. Missing Function Implementations

### 1.1 Context Generation Functions

**File**: `src/escape_room_sim/simulation/iterative_engine.py`  
**Lines**: 132-150  
**Issue**: Three functions are called but never defined

#### Required Functions:

```python
def get_strategist_context_for_iteration(
    iteration_num: int, 
    previous_failures: List[str], 
    current_resources: Dict[str, Any]
) -> str:
    """
    Generate context for Strategist agent for current iteration.
    
    Args:
        iteration_num: Current iteration number (1-based)
        previous_failures: List of failed strategies from memory
        current_resources: Current room resource summary
        
    Returns:
        Formatted context string for strategist agent
    """
    context = f"""
ITERATION {iteration_num} STRATEGIC ANALYSIS CONTEXT:

PREVIOUS STRATEGIC FAILURES TO AVOID:
{chr(10).join(f"- {failure}" for failure in previous_failures[-5:]) if previous_failures else "- No previous failures recorded"}

CURRENT RESOURCE STATUS:
{chr(10).join(f"- {name}: {info}" for name, info in current_resources.items()) if current_resources else "- No resources available"}

STRATEGIC PRIORITIES:
1. Analyze current situation systematically
2. Learn from previous failed approaches
3. Identify optimal resource utilization
4. Develop evidence-based escape plan
5. Consider team coordination requirements
"""
    return context

def get_mediator_context_for_iteration(
    iteration_num: int, 
    relationship_tracker: 'RelationshipTracker', 
    team_stress_level: float, 
    previous_conflicts: List[str]
) -> str:
    """
    Generate context for Mediator agent for current iteration.
    
    Args:
        iteration_num: Current iteration number
        relationship_tracker: Current relationship state tracker
        team_stress_level: Current team stress level (0.0-1.0)
        previous_conflicts: List of previous team conflicts
        
    Returns:
        Formatted context string for mediator agent
    """
    stress_description = "LOW" if team_stress_level < 0.3 else "MODERATE" if team_stress_level < 0.7 else "HIGH"
    
    context = f"""
ITERATION {iteration_num} MEDIATION CONTEXT:

TEAM DYNAMICS STATUS:
- Current Stress Level: {stress_description} ({team_stress_level:.2f})
- Relationship Status: {relationship_tracker.get_summary() if relationship_tracker else "No relationship data"}

PREVIOUS TEAM CONFLICTS:
{chr(10).join(f"- {conflict}" for conflict in previous_conflicts[-3:]) if previous_conflicts else "- No previous conflicts recorded"}

MEDIATION PRIORITIES:
1. Assess current team emotional state
2. Address any ongoing interpersonal tensions
3. Facilitate collaborative decision-making
4. Build consensus on action plans
5. Maintain team cohesion under pressure
6. Resolve resource allocation disputes
"""
    return context

def get_survivor_context_for_iteration(
    iteration_num: int, 
    survival_memory: 'SurvivalMemoryBank', 
    current_threat_level: float, 
    resource_status: Dict[str, Any]
) -> str:
    """
    Generate context for Survivor agent for current iteration.
    
    Args:
        iteration_num: Current iteration number
        survival_memory: Survival experience memory bank
        current_threat_level: Current threat assessment (0.0-1.0)
        resource_status: Current resource availability
        
    Returns:
        Formatted context string for survivor agent
    """
    threat_description = "LOW" if current_threat_level < 0.3 else "MODERATE" if current_threat_level < 0.7 else "HIGH"
    
    context = f"""
ITERATION {iteration_num} SURVIVAL CONTEXT:

THREAT ASSESSMENT:
- Threat Level: {threat_description} ({current_threat_level:.2f})
- Critical Resources: {len([r for r, info in resource_status.items() if info.get('critical', False)])}
- Time Pressure: {resource_status.get('time_remaining', 'Unknown')} minutes remaining

SURVIVAL MEMORY:
{survival_memory.get_relevant_experiences(3) if survival_memory else "No previous survival experiences"}

SURVIVAL PRIORITIES:
1. Assess immediate threats and risks
2. Evaluate individual vs team survival probability  
3. Identify critical resources for escape
4. Plan contingency actions for emergencies
5. Make pragmatic decisions under pressure
6. Balance team loyalty with survival instinct
"""
    return context
```

**Location**: Add these functions to `src/escape_room_sim/simulation/iterative_engine.py` at line 80 (before `create_iteration_tasks` method)

---

## 2. Missing Class Implementations

### 2.1 RelationshipTracker Class

**File**: `src/escape_room_sim/simulation/relationship_tracker.py` (new file)

```python
"""
Relationship tracking system for multi-agent interactions.
Monitors trust levels, collaboration effectiveness, and conflict patterns.
"""

from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from datetime import datetime
import json


@dataclass
class RelationshipEntry:
    """Single relationship interaction record."""
    agent_a: str
    agent_b: str
    interaction_type: str  # "collaboration", "conflict", "support", "disagreement"
    context: str
    outcome: str  # "positive", "negative", "neutral"
    trust_impact: float  # -1.0 to 1.0
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass 
class AgentRelationship:
    """Relationship state between two agents."""
    agent_a: str
    agent_b: str
    trust_level: float = 0.5  # 0.0 to 1.0
    collaboration_count: int = 0
    conflict_count: int = 0
    last_interaction: Optional[datetime] = None
    interaction_history: List[RelationshipEntry] = field(default_factory=list)


class RelationshipTracker:
    """
    Tracks and manages relationships between agents in the simulation.
    Monitors trust levels, collaboration patterns, and conflict resolution.
    """
    
    def __init__(self):
        self.relationships: Dict[Tuple[str, str], AgentRelationship] = {}
        self.interaction_history: List[RelationshipEntry] = []
        
    def _get_relationship_key(self, agent_a: str, agent_b: str) -> Tuple[str, str]:
        """Get standardized relationship key (alphabetical order)."""
        return tuple(sorted([agent_a, agent_b]))
    
    def get_relationship(self, agent_a: str, agent_b: str) -> AgentRelationship:
        """Get or create relationship between two agents."""
        key = self._get_relationship_key(agent_a, agent_b)
        if key not in self.relationships:
            self.relationships[key] = AgentRelationship(
                agent_a=key[0], 
                agent_b=key[1]
            )
        return self.relationships[key]
    
    def record_interaction(
        self, 
        agent_a: str, 
        agent_b: str, 
        interaction_type: str,
        context: str,
        outcome: str,
        trust_impact: float = 0.0
    ):
        """Record an interaction between two agents."""
        relationship = self.get_relationship(agent_a, agent_b)
        
        entry = RelationshipEntry(
            agent_a=agent_a,
            agent_b=agent_b,
            interaction_type=interaction_type,
            context=context,
            outcome=outcome,
            trust_impact=trust_impact
        )
        
        # Update relationship state
        relationship.trust_level = max(0.0, min(1.0, relationship.trust_level + trust_impact))
        relationship.last_interaction = datetime.now()
        relationship.interaction_history.append(entry)
        
        if interaction_type == "collaboration":
            relationship.collaboration_count += 1
        elif interaction_type == "conflict":
            relationship.conflict_count += 1
            
        self.interaction_history.append(entry)
    
    def record_successful_collaboration(
        self, 
        agents: List[str], 
        strategy: str, 
        outcome: str
    ):
        """Record successful collaboration between multiple agents."""
        # Record pairwise collaborations
        for i, agent_a in enumerate(agents):
            for agent_b in agents[i+1:]:
                self.record_interaction(
                    agent_a=agent_a,
                    agent_b=agent_b,
                    interaction_type="collaboration",
                    context=f"Successful strategy: {strategy}",
                    outcome=outcome,
                    trust_impact=0.1  # Small positive trust increase
                )
    
    def record_conflict(
        self, 
        agent_a: str, 
        agent_b: str, 
        conflict_reason: str, 
        resolution: str
    ):
        """Record conflict between two agents and its resolution."""
        trust_impact = -0.1 if resolution == "unresolved" else -0.05
        
        self.record_interaction(
            agent_a=agent_a,
            agent_b=agent_b,
            interaction_type="conflict",
            context=f"Conflict: {conflict_reason}",
            outcome=resolution,
            trust_impact=trust_impact
        )
    
    def get_trust_level(self, agent_a: str, agent_b: str) -> float:
        """Get current trust level between two agents."""
        relationship = self.get_relationship(agent_a, agent_b)
        return relationship.trust_level
    
    def get_team_cohesion(self, agents: List[str]) -> float:
        """Calculate overall team cohesion score."""
        if len(agents) < 2:
            return 1.0
            
        trust_levels = []
        for i, agent_a in enumerate(agents):
            for agent_b in agents[i+1:]:
                trust_levels.append(self.get_trust_level(agent_a, agent_b))
                
        return sum(trust_levels) / len(trust_levels) if trust_levels else 0.5
    
    def get_summary(self) -> str:
        """Get readable summary of current relationship state."""
        if not self.relationships:
            return "No relationship data available"
            
        summaries = []
        for relationship in self.relationships.values():
            trust_desc = "HIGH" if relationship.trust_level > 0.7 else "MODERATE" if relationship.trust_level > 0.4 else "LOW"
            summaries.append(
                f"{relationship.agent_a}-{relationship.agent_b}: {trust_desc} trust "
                f"({relationship.collaboration_count} collaborations, {relationship.conflict_count} conflicts)"
            )
            
        return "; ".join(summaries)
    
    def export_data(self) -> Dict:
        """Export relationship data for persistence."""
        return {
            "relationships": {
                f"{key[0]}-{key[1]}": {
                    "trust_level": rel.trust_level,
                    "collaboration_count": rel.collaboration_count,
                    "conflict_count": rel.conflict_count,
                    "last_interaction": rel.last_interaction.isoformat() if rel.last_interaction else None
                }
                for key, rel in self.relationships.items()
            },
            "total_interactions": len(self.interaction_history)
        }
```

### 2.2 SurvivalMemoryBank Class

**File**: `src/escape_room_sim/simulation/survival_memory.py` (new file)

```python
"""
Survival experience memory system for the Survivor agent.
Tracks close calls, successful strategies, and survival lessons learned.
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
import json


@dataclass
class SurvivalExperience:
    """Record of a survival-related experience."""
    situation_type: str  # "resource_shortage", "time_pressure", "team_conflict", "escape_attempt"
    threat_level: float  # 0.0 to 1.0
    survival_action: str
    outcome: str  # "success", "failure", "partial_success"
    lessons_learned: List[str]
    agents_involved: List[str]
    resources_used: List[str]
    timestamp: datetime = field(default_factory=datetime.now)
    importance_score: float = 0.5  # 0.0 to 1.0


@dataclass
class ThreatAssessment:
    """Assessment of current threats and survival probability."""
    threat_type: str
    severity: float  # 0.0 to 1.0
    probability: float  # 0.0 to 1.0
    mitigation_strategies: List[str]
    resource_requirements: List[str]


class SurvivalMemoryBank:
    """
    Memory system for survival experiences and threat assessments.
    Helps the Survivor agent learn from previous close calls and successes.
    """
    
    def __init__(self):
        self.experiences: List[SurvivalExperience] = []
        self.threat_assessments: List[ThreatAssessment] = []
        self.survival_principles: List[str] = [
            "Assess threats systematically before acting",
            "Secure critical resources first",
            "Maintain multiple escape options",  
            "Balance individual and team survival",
            "Act decisively when time is critical"
        ]
    
    def record_close_call(
        self, 
        situation: str, 
        threat: str, 
        survival_action: str, 
        outcome: str,
        lessons: List[str] = None,
        agents_involved: List[str] = None,
        resources_used: List[str] = None
    ):
        """Record a survival close call experience."""
        experience = SurvivalExperience(
            situation_type=situation,
            threat_level=0.8,  # Close calls are high threat
            survival_action=survival_action,
            outcome=outcome,
            lessons_learned=lessons or [],
            agents_involved=agents_involved or [],
            resources_used=resources_used or [],
            importance_score=0.9  # Close calls are highly important
        )
        
        self.experiences.append(experience)
        
        # Add lessons to survival principles if not already present
        for lesson in (lessons or []):
            if lesson not in self.survival_principles:
                self.survival_principles.append(lesson)
    
    def record_successful_strategy(
        self, 
        situation: str, 
        strategy: str, 
        outcome: str,
        agents_involved: List[str] = None,
        resources_used: List[str] = None
    ):
        """Record a successful survival strategy."""
        experience = SurvivalExperience(
            situation_type=situation,
            threat_level=0.5,  # Successful strategies may involve moderate threat
            survival_action=strategy,
            outcome=outcome,
            lessons_learned=[f"Successful strategy: {strategy}"],
            agents_involved=agents_involved or [],
            resources_used=resources_used or [],
            importance_score=0.7  # Success is important for learning
        )
        
        self.experiences.append(experience)
    
    def assess_current_threat(
        self, 
        threat_type: str, 
        current_situation: Dict[str, Any]
    ) -> ThreatAssessment:
        """Assess current threat based on past experiences."""
        # Find similar past experiences
        similar_experiences = [
            exp for exp in self.experiences 
            if exp.situation_type == threat_type
        ]
        
        # Calculate threat severity based on past experiences
        if similar_experiences:
            avg_threat = sum(exp.threat_level for exp in similar_experiences) / len(similar_experiences)
            # Look for successful mitigation strategies
            successful_strategies = [
                exp.survival_action for exp in similar_experiences 
                if exp.outcome == "success"
            ]
        else:
            avg_threat = 0.5  # Unknown threat defaults to moderate
            successful_strategies = []
        
        return ThreatAssessment(
            threat_type=threat_type,
            severity=avg_threat,
            probability=0.7,  # Default probability
            mitigation_strategies=successful_strategies,
            resource_requirements=self._get_resource_requirements(threat_type)
        )
    
    def _get_resource_requirements(self, threat_type: str) -> List[str]:
        """Get typical resource requirements for threat type."""
        requirements_map = {
            "time_pressure": ["efficient_tools", "clear_communication"],
            "resource_shortage": ["backup_supplies", "rationing_plan"],
            "team_conflict": ["mediation_skills", "compromise_solutions"],
            "escape_attempt": ["multiple_routes", "coordination_plan"]
        }
        return requirements_map.get(threat_type, ["general_preparation"])
    
    def get_relevant_experiences(self, max_count: int = 5) -> str:
        """Get most relevant survival experiences as formatted string."""
        if not self.experiences:
            return "No previous survival experiences recorded"
        
        # Sort by importance and recency
        sorted_experiences = sorted(
            self.experiences,
            key=lambda x: (x.importance_score, x.timestamp),
            reverse=True
        )[:max_count]
        
        experience_summaries = []
        for exp in sorted_experiences:
            summary = (
                f"- {exp.situation_type.replace('_', ' ').title()}: "
                f"{exp.survival_action} → {exp.outcome}"
            )
            if exp.lessons_learned:
                summary += f" (Learned: {'; '.join(exp.lessons_learned[:2])})"
            experience_summaries.append(summary)
        
        return "\n".join(experience_summaries)
    
    def get_survival_principles(self) -> List[str]:
        """Get current survival principles learned from experience."""
        return self.survival_principles.copy()
    
    def calculate_survival_probability(
        self, 
        current_situation: Dict[str, Any], 
        proposed_action: str
    ) -> float:
        """Calculate survival probability for proposed action."""
        # Find similar past situations
        similar_count = 0
        success_count = 0
        
        for exp in self.experiences:
            if (exp.survival_action.lower() in proposed_action.lower() or 
                proposed_action.lower() in exp.survival_action.lower()):
                similar_count += 1
                if exp.outcome == "success":
                    success_count += 1
        
        if similar_count == 0:
            return 0.5  # Unknown action, moderate probability
        
        base_probability = success_count / similar_count
        
        # Adjust based on current threat level
        threat_level = current_situation.get('threat_level', 0.5)
        adjusted_probability = base_probability * (1.0 - (threat_level * 0.3))
        
        return max(0.1, min(0.9, adjusted_probability))
    
    def export_data(self) -> Dict:
        """Export survival memory data for persistence."""
        return {
            "experiences": [
                {
                    "situation_type": exp.situation_type,
                    "threat_level": exp.threat_level,
                    "survival_action": exp.survival_action,
                    "outcome": exp.outcome,
                    "lessons_learned": exp.lessons_learned,
                    "importance_score": exp.importance_score,
                    "timestamp": exp.timestamp.isoformat()
                }
                for exp in self.experiences
            ],
            "survival_principles": self.survival_principles,
            "total_experiences": len(self.experiences)
        }
```

---

## 3. API Configuration Fix

### 3.1 Dynamic Memory Provider Configuration

**File**: `src/escape_room_sim/simulation/simple_engine.py`  
**Lines**: 102-107  

**Current Issue**: Hardcoded OpenAI configuration in memory setup

**Required Changes**:

```python
# Replace lines 102-107 with:
def _get_memory_config(self) -> Dict[str, Any]:
    """Get memory configuration based on available API providers."""
    
    # Check for Gemini API (preferred)
    if os.getenv("GEMINI_API_KEY"):
        return {
            "provider": "gemini",
            "config": {
                "api_key": os.getenv("GEMINI_API_KEY"),
                "model": os.getenv("MODEL", "gemini-2.5-flash-lite"),
                "embedding_model": "text-embedding-004"
            }
        }
    
    # Fallback to OpenAI
    elif os.getenv("OPENAI_API_KEY"):
        return {
            "provider": "openai", 
            "config": {
                "api_key": os.getenv("OPENAI_API_KEY"),
                "model": "text-embedding-3-small"
            }
        }
    
    # Fallback to Anthropic (no embedding support, use local)
    elif os.getenv("ANTHROPIC_API_KEY"):
        return {
            "provider": "local",
            "config": {
                "model": "sentence-transformers/all-MiniLM-L6-v2"
            }
        }
    
    else:
        raise ValueError("No valid API configuration found for memory system")

# Update the crew creation to use dynamic config:
def _create_crew(self, agents: List[Agent], tasks: List[Task]) -> Crew:
    """Create CrewAI crew with dynamic memory configuration."""
    
    memory_config = self._get_memory_config()
    
    return Crew(
        agents=agents,
        tasks=tasks,
        process=Process.sequential,
        memory=self.config.enable_memory,
        memory_config=memory_config,
        verbose=self.config.verbose_output,
        manager_llm=None,
        function_calling_llm=None,
        max_iter=self.config.max_iterations,
        max_execution_time=self.config.max_execution_time
    )
```

---

## 4. Survival Constraint Fix

### 4.1 Enforce "Only Two Can Survive" Rule

**File**: `src/escape_room_sim/room/escape_room_state.py`  
**Lines**: 192-211 (exit routes definition)

**Current Issue**: Main door allows all 3 agents to escape, breaking core premise

**Required Changes**:

```python
# Modify exit routes in __init__ method:
self.exit_routes = {
    "main_door": {
        "name": "Main Door",
        "description": "Heavy wooden door with complex lock mechanism",
        "capacity": 2,  # CHANGED FROM 3 - Forces difficult choice
        "difficulty": 0.7,
        "requirements": ["key", "lockpicking_tools", "2_agents_minimum"],
        "risk_level": 0.4,
        "time_required": 15,
        "discovered": True,
        "resource_cost": {"key": 1, "lockpicking_tools": 1}
    },
    "vent_shaft": {
        "name": "Ventilation Shaft", 
        "description": "Narrow air vent leading outside",
        "capacity": 1,  # Only one person can escape
        "difficulty": 0.9,
        "requirements": ["screwdriver", "rope", "physical_strength"],
        "risk_level": 0.8,
        "time_required": 25,
        "discovered": False,
        "resource_cost": {"screwdriver": 1, "rope": 1}
    },
    "window": {
        "name": "Reinforced Window",
        "description": "Second-floor window with security bars",
        "capacity": 2,  # Two people can escape together
        "difficulty": 0.6,
        "requirements": ["crowbar", "rope", "teamwork"],
        "risk_level": 0.6,
        "time_required": 20,
        "discovered": True,
        "resource_cost": {"crowbar": 1, "rope": 1}
    },
    "hidden_passage": {
        "name": "Hidden Passage",
        "description": "Secret passage behind bookshelf",
        "capacity": 2,  # Two people can escape
        "difficulty": 0.8,
        "requirements": ["hidden_key", "flashlight"],
        "risk_level": 0.3,
        "time_required": 10,
        "discovered": False,
        "resource_cost": {"hidden_key": 1, "flashlight": 1}
    }
}
```

### 4.2 Add Survival Decision Logic

**File**: `src/escape_room_sim/room/escape_room_state.py`  
**New Method**: Add after line 350

```python
def evaluate_survival_scenarios(self, agents: List[str]) -> Dict[str, Any]:
    """
    Evaluate possible survival scenarios based on available resources and exit routes.
    Forces the 'only two can survive' constraint.
    """
    if len(agents) != 3:
        raise ValueError("Survival evaluation requires exactly 3 agents")
    
    available_exits = [
        name for name, exit_info in self.exit_routes.items() 
        if self._can_attempt_exit(name)
    ]
    
    scenarios = []
    
    # Generate all possible 2-agent survival combinations
    import itertools
    for survivors in itertools.combinations(agents, 2):
        sacrifice = [agent for agent in agents if agent not in survivors][0]
        
        # Find best exit route for survivors
        best_exit = None
        best_probability = 0.0
        
        for exit_name in available_exits:
            exit_info = self.exit_routes[exit_name]
            if exit_info["capacity"] >= 2:
                # Calculate success probability
                probability = self._calculate_escape_probability(exit_name, list(survivors))
                if probability > best_probability:
                    best_exit = exit_name
                    best_probability = probability
        
        scenarios.append({
            "survivors": list(survivors),
            "sacrifice": sacrifice,
            "exit_route": best_exit,
            "success_probability": best_probability,
            "moral_difficulty": self._calculate_moral_difficulty(sacrifice, survivors)
        })
    
    # Sort by success probability
    scenarios.sort(key=lambda x: x["success_probability"], reverse=True)
    
    return {
        "possible_scenarios": scenarios,
        "forced_choice": len(scenarios) > 0,
        "time_remaining": self.time_remaining,
        "stress_level": self.stress_level
    }

def _calculate_moral_difficulty(self, sacrifice: str, survivors: List[str]) -> float:
    """Calculate moral difficulty of sacrifice decision (0.0 to 1.0)."""
    # This could be enhanced with relationship tracking
    base_difficulty = 0.8  # High moral difficulty by default
    
    # Add factors that might make decision easier/harder
    if self.stress_level > 0.8:
        base_difficulty *= 0.9  # Extreme stress reduces moral consideration
    
    if self.time_remaining < 5:
        base_difficulty *= 0.8  # Time pressure reduces moral consideration
    
    return min(1.0, base_difficulty)

def _calculate_escape_probability(self, exit_name: str, agents: List[str]) -> float:
    """Calculate probability of successful escape through given exit."""
    exit_info = self.exit_routes[exit_name]
    
    base_probability = 1.0 - exit_info["difficulty"]
    
    # Factor in resource availability
    resource_penalty = 0.0
    for resource, required_qty in exit_info.get("resource_cost", {}).items():
        if resource not in self.resources or self.resources[resource].quantity < required_qty:
            resource_penalty += 0.3
    
    # Factor in agent count vs capacity
    if len(agents) > exit_info["capacity"]:
        return 0.0  # Can't fit through exit
    
    # Factor in stress and time pressure
    stress_penalty = self.stress_level * 0.2
    time_penalty = 0.1 if self.time_remaining < 10 else 0.0
    
    final_probability = base_probability - resource_penalty - stress_penalty - time_penalty
    return max(0.1, min(0.9, final_probability))
```

---

## 5. Error Handling and Safety Measures

### 5.1 Infinite Loop Protection

**File**: `src/escape_room_sim/simulation/simple_engine.py`  
**Method**: `run_full_simulation`  
**Lines**: 235-256

**Add Safety Check**:

```python
def run_full_simulation(self) -> Dict[str, Any]:
    """Run the complete iterative simulation with safety measures."""
    
    # Safety measures
    MAX_TOTAL_TIME = 1800  # 30 minutes maximum
    MAX_STAGNANT_ITERATIONS = 5  # Stop if no progress
    
    start_time = time.time()
    stagnant_count = 0
    last_progress_hash = None
    
    while True:
        # Time-based safety check
        if time.time() - start_time > MAX_TOTAL_TIME:
            reason = "Maximum simulation time exceeded (safety limit)"
            break
        
        # Stagnation detection
        current_progress = self._get_progress_hash()
        if current_progress == last_progress_hash:
            stagnant_count += 1
            if stagnant_count >= MAX_STAGNANT_ITERATIONS:
                reason = "No progress detected - simulation stagnant"
                break
        else:
            stagnant_count = 0
            last_progress_hash = current_progress
        
        try:
            result = self.run_single_iteration()
            should_stop, reason = self.check_stopping_conditions()
            if should_stop:
                break
                
        except Exception as e:
            reason = f"Simulation error: {str(e)}"
            break
    
    # Generate final report...

def _get_progress_hash(self) -> str:
    """Generate hash of current progress state for stagnation detection."""
    import hashlib
    
    progress_elements = [
        str(len(self.iteration_results)),
        str(self.game_state.time_remaining),
        str(len([p for p in self.game_state.puzzles.values() if p.status.value == "solved"])),
        str(len([r for r in self.game_state.resources.values() if r.discovered]))
    ]
    
    return hashlib.md5("|".join(progress_elements).encode()).hexdigest()[:8]
```

---

## 6. Implementation Order and Dependencies

### Phase 1 - Core Function Implementation (2-3 hours)
1. Add context generation functions to `iterative_engine.py`
2. Create `RelationshipTracker` class in new file
3. Create `SurvivalMemoryBank` class in new file
4. Import new classes in `iterative_engine.py`

### Phase 2 - API Configuration Fix (1 hour)
1. Implement `_get_memory_config()` method
2. Update crew creation to use dynamic configuration
3. Test with different API provider combinations

### Phase 3 - Survival Constraint Fix (2-3 hours)
1. Modify exit route capacities
2. Implement survival scenario evaluation
3. Add escape probability calculations
4. Test constraint enforcement

### Phase 4 - Safety Measures (1-2 hours)
1. Add infinite loop protection
2. Implement stagnation detection
3. Add comprehensive error handling
4. Test safety mechanisms

---

## 7. Testing Requirements

### 7.1 Unit Tests Required
- Context generation functions return valid strings
- RelationshipTracker correctly tracks interactions
- SurvivalMemoryBank stores and retrieves experiences
- API configuration selection works correctly
- Survival constraint enforcement functions properly

### 7.2 Integration Tests Required
- Full simulation runs without crashes
- Memory system works with different API providers
- Survival scenarios are properly evaluated
- Safety measures prevent infinite loops

---

## 8. Acceptance Criteria

✅ **All missing functions implemented and callable**  
✅ **All missing classes implemented and instantiable**  
✅ **API configuration works with available providers**  
✅ **Survival constraint properly enforces "only two can survive"**  
✅ **Simulation runs without NameError or AttributeError exceptions**  
✅ **Safety measures prevent infinite loops and stagnation**  
✅ **All existing tests continue to pass**  

---

## 9. Risk Assessment

**HIGH RISK**: Missing implementations prevent any execution  
**MEDIUM RISK**: API configuration issues cause memory failures  
**LOW RISK**: Survival constraint changes affect simulation balance  

**MITIGATION**: Complete implementation in order, test each phase before proceeding to next.

---

**Document Version**: 1.0  
**Last Updated**: 2025-08-04  
**Status**: Ready for Implementation