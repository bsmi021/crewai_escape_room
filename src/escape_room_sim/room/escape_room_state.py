"""
Escape Room State Management.

This module manages the current state of the escape room, including resources,
puzzles, time constraints, and environmental conditions that affect agent decisions.
"""

import json
import random
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict, field
from enum import Enum


class PuzzleStatus(Enum):
    """Status of escape room puzzles."""
    UNSOLVED = "unsolved"
    IN_PROGRESS = "in_progress"
    SOLVED = "solved"
    BLOCKED = "blocked"


class ExitRoute(Enum):
    """Available exit routes from the room."""
    MAIN_DOOR = "main_door"
    VENT_SHAFT = "vent_shaft"
    HIDDEN_PASSAGE = "hidden_passage"
    WINDOW = "window"


@dataclass
class Puzzle:
    """Represents a puzzle that must be solved to escape."""
    name: str
    description: str
    status: PuzzleStatus = PuzzleStatus.UNSOLVED
    required_resources: List[str] = field(default_factory=list)
    required_cooperation: int = 1  # Number of agents needed
    difficulty: float = 0.5  # 0.0 to 1.0
    time_to_solve: int = 10  # Minutes required
    clues_discovered: List[str] = field(default_factory=list)
    attempts_made: int = 0
    last_attempt_iteration: int = 0


@dataclass
class Resource:
    """Represents a resource available in the room."""
    name: str
    description: str
    quantity: int = 1
    consumable: bool = True
    location: str = "unknown"
    discovered: bool = False
    required_for: List[str] = field(default_factory=list)  # Which puzzles need this


@dataclass
class Exit:
    """Represents an exit route from the room."""
    route: ExitRoute
    capacity: int  # How many people can use this exit
    requirements: List[str] = field(default_factory=list)  # What's needed to use this exit
    risk_level: float = 0.3  # 0.0 to 1.0
    time_to_use: int = 5  # Minutes to execute escape
    discovered: bool = True
    blocked: bool = False
    success_probability: float = 0.8


class EscapeRoomState:
    """
    Manages the complete state of the escape room simulation.
    
    This class tracks:
    - Available resources and their locations
    - Puzzle states and progress
    - Exit routes and their viability
    - Time constraints and pressure
    - Environmental conditions
    - Agent stress and cooperation levels
    """
    
    def __init__(self):
        """Initialize the escape room with default state."""
        self.time_remaining = 60  # Minutes until critical failure
        self.initial_time = 60
        self.iteration_count = 0
        self.stress_level = 0.3  # Team stress level (0.0 to 1.0)
        self.cooperation_level = 0.7  # Team cooperation (0.0 to 1.0)
        
        # Initialize room components
        self.resources = self._initialize_resources()
        self.puzzles = self._initialize_puzzles()
        self.exits = self._initialize_exits()
        
        # Track changes over time
        self.state_history: List[Dict[str, Any]] = []
        self.significant_events: List[str] = []
        
        print("🏠 Escape room initialized with default configuration")
    
    def _initialize_resources(self) -> Dict[str, Resource]:
        """Initialize available resources in the room."""
        resources = {
            "key_red": Resource(
                name="Red Key",
                description="A red metal key found under debris",
                location="under debris pile",
                discovered=False,
                required_for=["main_door"]
            ),
            "key_blue": Resource(
                name="Blue Key", 
                description="A blue electronic key card",
                location="computer terminal",
                discovered=False,
                required_for=["main_door"]
            ),
            "tools": Resource(
                name="Makeshift Tools",
                description="Screwdriver, wire, small hammer improvised from debris",
                quantity=3,
                location="scattered around room",
                discovered=True,
                required_for=["computer_terminal", "vent_access"]
            ),
            "rope": Resource(
                name="Rope",
                description="20 feet of sturdy rope",
                location="emergency supply box",
                discovered=False,
                required_for=["window_escape", "vent_access"]
            ),
            "flashlight": Resource(
                name="Flashlight",
                description="Battery-powered flashlight for dark areas",
                location="emergency supply box", 
                discovered=False,
                required_for=["hidden_passage"]
            ),
            "password_hint": Resource(
                name="Password Hint",
                description="Written clue about computer terminal password",
                location="hidden under loose floorboard",
                discovered=False,
                consumable=False,
                required_for=["computer_terminal"]
            )
        }
        return resources
    
    def _initialize_puzzles(self) -> Dict[str, Puzzle]:
        """Initialize puzzles that must be solved."""
        puzzles = {
            "computer_terminal": Puzzle(
                name="Computer Terminal",
                description="Locked computer that controls electronic systems",
                required_resources=["tools", "password_hint"],
                required_cooperation=2,  # Needs multiple people
                difficulty=0.7,
                time_to_solve=15
            ),
            "debris_clearing": Puzzle(
                name="Debris Clearing",
                description="Heavy debris blocking access to resources",
                required_resources=["tools"],
                required_cooperation=3,  # All agents needed
                difficulty=0.4,
                time_to_solve=10
            ),
            "vent_access": Puzzle(
                name="Vent Access",
                description="Open and prepare ventilation shaft for escape",
                required_resources=["tools", "rope"],
                required_cooperation=1,
                difficulty=0.5,
                time_to_solve=8
            ),
            "main_door_lock": Puzzle(
                name="Main Door Lock",
                description="Complex lock requiring both keys and coordination",
                required_resources=["key_red", "key_blue"],
                required_cooperation=2,
                difficulty=0.6,
                time_to_solve=5
            )
        }
        return puzzles
    
    def _initialize_exits(self) -> Dict[str, Exit]:
        """Initialize available exit routes."""
        exits = {
            "main_door": Exit(
                route=ExitRoute.MAIN_DOOR,
                capacity=3,  # All can escape together
                requirements=["key_red", "key_blue", "main_door_lock"],
                risk_level=0.1,  # Safest option
                time_to_use=3,
                success_probability=0.95
            ),
            "vent_shaft": Exit(
                route=ExitRoute.VENT_SHAFT,
                capacity=1,  # Only one person fits
                requirements=["vent_access"],
                risk_level=0.4,  # Moderate risk
                time_to_use=8,
                success_probability=0.8
            ),
            "window": Exit(
                route=ExitRoute.WINDOW,
                capacity=2,  # Two people maximum
                requirements=["rope", "tools"],
                risk_level=0.6,  # Higher risk
                time_to_use=12,
                success_probability=0.7
            ),
            "hidden_passage": Exit(
                route=ExitRoute.HIDDEN_PASSAGE,
                capacity=2,
                requirements=["flashlight"],
                risk_level=0.5,
                time_to_use=15,
                discovered=False,  # Must be found first
                success_probability=0.75
            )
        }
        return exits
    
    def discover_resource(self, resource_name: str) -> bool:
        """
        Discover a resource in the room.
        
        Args:
            resource_name: Name of the resource to discover
            
        Returns:
            True if resource was successfully discovered
        """
        if resource_name in self.resources:
            resource = self.resources[resource_name]
            if not resource.discovered:
                resource.discovered = True
                self.significant_events.append(f"Discovered {resource.name}: {resource.description}")
                print(f"🔍 Discovered: {resource.name}")
                return True
        return False
    
    def consume_resource(self, resource_name: str, quantity: int = 1) -> bool:
        """
        Consume a resource.
        
        Args:
            resource_name: Name of the resource to consume
            quantity: Amount to consume
            
        Returns:
            True if resource was successfully consumed
        """
        if resource_name in self.resources:
            resource = self.resources[resource_name]
            if resource.discovered and resource.quantity >= quantity and resource.consumable:
                resource.quantity -= quantity
                self.significant_events.append(f"Used {quantity}x {resource.name}")
                print(f"⚡ Consumed: {quantity}x {resource.name}")
                return True
        return False
    
    def add_resource(self, resource_name: str, quantity: int = 1) -> bool:
        """Add or replenish a resource."""
        if resource_name in self.resources:
            self.resources[resource_name].quantity += quantity
            self.significant_events.append(f"Found additional {resource_name}")
            return True
        return False
    
    def attempt_puzzle(self, puzzle_name: str, agents_working: int, 
                      iteration: int) -> Tuple[bool, str]:
        """
        Attempt to solve a puzzle.
        
        Args:
            puzzle_name: Name of the puzzle to attempt
            agents_working: Number of agents working on it
            iteration: Current iteration number
            
        Returns:
            Tuple of (success, result_message)
        """
        if puzzle_name not in self.puzzles:
            return False, f"Unknown puzzle: {puzzle_name}"
        
        puzzle = self.puzzles[puzzle_name]
        puzzle.attempts_made += 1
        puzzle.last_attempt_iteration = iteration
        
        # Check if we have required resources
        missing_resources = []
        for req_resource in puzzle.required_resources:
            if req_resource not in self.resources:
                missing_resources.append(req_resource)
            elif not self.resources[req_resource].discovered:
                missing_resources.append(f"{req_resource} (not discovered)")
            elif self.resources[req_resource].quantity < 1:
                missing_resources.append(f"{req_resource} (insufficient quantity)")
        
        if missing_resources:
            puzzle.status = PuzzleStatus.BLOCKED
            return False, f"Cannot attempt {puzzle.name}: Missing {', '.join(missing_resources)}"
        
        # Check cooperation requirements
        if agents_working < puzzle.required_cooperation:
            return False, f"Puzzle requires {puzzle.required_cooperation} agents, only {agents_working} working"
        
        # Calculate success probability
        base_probability = 1.0 - puzzle.difficulty
        
        # Bonuses
        cooperation_bonus = min(0.2, (agents_working - puzzle.required_cooperation) * 0.1)
        experience_bonus = min(0.1, puzzle.attempts_made * 0.02)  # Small learning bonus
        stress_penalty = self.stress_level * 0.3  # High stress reduces performance
        
        success_probability = base_probability + cooperation_bonus + experience_bonus - stress_penalty
        success_probability = max(0.1, min(0.95, success_probability))  # Clamp between 10% and 95%
        
        # Determine success
        success = random.random() < success_probability
        
        if success:
            puzzle.status = PuzzleStatus.SOLVED
            self.consume_time(puzzle.time_to_solve)
            
            # Consume required resources
            for resource_name in puzzle.required_resources:
                if self.resources[resource_name].consumable:
                    self.consume_resource(resource_name)
            
            result_message = f"Successfully solved {puzzle.name}! Time taken: {puzzle.time_to_solve} minutes"
            self.significant_events.append(f"SOLVED: {puzzle.name}")
            
            # Reduce stress on success
            self.reduce_stress(0.1)
            
        else:
            puzzle.status = PuzzleStatus.IN_PROGRESS
            time_wasted = min(puzzle.time_to_solve // 2, 5)  # Waste some time on failure
            self.consume_time(time_wasted)
            
            result_message = f"Failed to solve {puzzle.name}. Time wasted: {time_wasted} minutes"
            
            # Add clue on failure (learning from attempts)
            if len(puzzle.clues_discovered) < 3:
                new_clue = f"Attempt {puzzle.attempts_made}: {self._generate_puzzle_clue(puzzle_name)}"
                puzzle.clues_discovered.append(new_clue)
                result_message += f" | New insight: {new_clue}"
            
            # Increase stress on failure
            self.increase_stress(0.05)
        
        print(f"🧩 Puzzle attempt - {puzzle.name}: {'SUCCESS' if success else 'FAILED'}")
        return success, result_message
    
    def _generate_puzzle_clue(self, puzzle_name: str) -> str:
        """Generate a helpful clue for a puzzle based on attempts."""
        clues = {
            "computer_terminal": [
                "The password might be related to the room's purpose",
                "There are scratches near certain keys on the keyboard",
                "The screen briefly shows a hint when powered on"
            ],
            "debris_clearing": [
                "The debris is heavier on one side than the other",
                "Some pieces are loose and can be moved individually",
                "There's a specific sequence that makes removal easier"
            ],
            "vent_access": [
                "The vent cover has different types of screws",
                "Some screws are already partially loose",
                "The vent opening might need to be widened"
            ],
            "main_door_lock": [
                "The keys must be turned simultaneously",
                "There's a specific order for using the keys",
                "The lock mechanism has a time-sensitive component"
            ]
        }
        
        puzzle_clues = clues.get(puzzle_name, ["The solution requires careful observation"])
        return random.choice(puzzle_clues)
    
    def solve_puzzle(self, puzzle_name: str = None):
        """Mark a puzzle as solved (used by external systems)."""
        if puzzle_name and puzzle_name in self.puzzles:
            self.puzzles[puzzle_name].status = PuzzleStatus.SOLVED
        
        # General puzzle solving feedback
        self.reduce_stress(0.1)
    
    def consume_time(self, minutes: int):
        """Consume time from the remaining time."""
        self.time_remaining = max(0, self.time_remaining - minutes)
        if self.time_remaining <= 10:
            self.increase_stress(0.2)  # High stress when time is running out
        elif self.time_remaining <= 20:
            self.increase_stress(0.1)
    
    def increase_stress(self, amount: float):
        """Increase team stress level."""
        self.stress_level = min(1.0, self.stress_level + amount)
        if self.stress_level > 0.8:
            self.cooperation_level = max(0.1, self.cooperation_level - 0.1)
    
    def reduce_stress(self, amount: float):
        """Reduce team stress level."""
        self.stress_level = max(0.0, self.stress_level - amount)
        if self.stress_level < 0.5:
            self.cooperation_level = min(1.0, self.cooperation_level + 0.05)
    
    def get_available_resources(self) -> List[str]:
        """Get list of discovered and available resources."""
        return [name for name, resource in self.resources.items() 
                if resource.discovered and resource.quantity > 0]
    
    def get_solvable_puzzles(self) -> List[str]:
        """Get puzzles that could potentially be solved with current resources."""
        solvable = []
        for name, puzzle in self.puzzles.items():
            if puzzle.status != PuzzleStatus.SOLVED:
                # Check if all required resources are available
                can_solve = all(
                    req_resource in self.resources and 
                    self.resources[req_resource].discovered and
                    self.resources[req_resource].quantity > 0
                    for req_resource in puzzle.required_resources
                )
                if can_solve:
                    solvable.append(name)
        return solvable
    
    def get_viable_exits(self) -> List[str]:
        """Get exit routes that are currently viable."""
        viable = []
        for name, exit_route in self.exits.items():
            if exit_route.discovered and not exit_route.blocked:
                # Check if requirements are met
                requirements_met = all(
                    req in self.puzzles and self.puzzles[req].status == PuzzleStatus.SOLVED
                    or req in self.resources and self.resources[req].discovered
                    for req in exit_route.requirements
                )
                if requirements_met:
                    viable.append(name)
        return viable
    
    def get_resource_summary(self) -> Dict[str, Any]:
        """Get summary of current resource status."""
        return {
            "discovered": [name for name, r in self.resources.items() if r.discovered],
            "available": self.get_available_resources(),
            "total_resources": len(self.resources),
            "discovery_rate": len([r for r in self.resources.values() if r.discovered]) / len(self.resources)
        }
    
    def get_resource_status(self) -> Dict[str, Any]:
        """Get detailed resource status for agents."""
        return {
            "assets": self.get_available_resources(),
            "critical_needs": self._identify_critical_needs(),
            "time_remaining": self.time_remaining
        }
    
    def _identify_critical_needs(self) -> List[str]:
        """Identify resources critically needed for progress."""
        critical = []
        
        # Resources needed for viable exit routes
        for exit_name, exit_route in self.exits.items():
            if exit_route.discovered and not exit_route.blocked:
                for req in exit_route.requirements:
                    if req not in self.get_available_resources():
                        critical.append(req)
        
        return list(set(critical))  # Remove duplicates
    
    def get_stress_level(self) -> float:
        """Get current team stress level."""
        return self.stress_level
    
    def get_threat_level(self) -> float:
        """Calculate current threat level based on multiple factors."""
        time_threat = 1.0 - (self.time_remaining / self.initial_time)
        stress_threat = self.stress_level
        cooperation_threat = 1.0 - self.cooperation_level
        
        # Weighted average
        threat_level = (time_threat * 0.4 + stress_threat * 0.3 + cooperation_threat * 0.3)
        return min(1.0, threat_level)
    
    def get_status_summary(self) -> str:
        """Get concise status summary."""
        return f"Time: {self.time_remaining}min | Stress: {self.stress_level:.1f} | Resources: {len(self.get_available_resources())}"
    
    def record_state_snapshot(self, iteration: int):
        """Record current state for historical analysis."""
        snapshot = {
            "iteration": iteration,
            "timestamp": datetime.now().isoformat(),
            "time_remaining": self.time_remaining,
            "stress_level": self.stress_level,
            "cooperation_level": self.cooperation_level,
            "resources_discovered": len([r for r in self.resources.values() if r.discovered]),
            "puzzles_solved": len([p for p in self.puzzles.values() if p.status == PuzzleStatus.SOLVED]),
            "viable_exits": len(self.get_viable_exits()),
            "significant_events": self.significant_events.copy()
        }
        self.state_history.append(snapshot)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert state to dictionary for serialization."""
        return {
            "time_remaining": self.time_remaining,
            "stress_level": self.stress_level,
            "cooperation_level": self.cooperation_level,
            "iteration_count": self.iteration_count,
            "resources": {name: asdict(resource) for name, resource in self.resources.items()},
            "puzzles": {name: asdict(puzzle) for name, puzzle in self.puzzles.items()},
            "exits": {name: asdict(exit_route) for name, exit_route in self.exits.items()},
            "significant_events": self.significant_events,
            "available_resources": self.get_available_resources(),
            "solvable_puzzles": self.get_solvable_puzzles(),
            "viable_exits": self.get_viable_exits()
        }
    
    def can_escape_successfully(self) -> Tuple[bool, str, List[str]]:
        """
        Determine if escape is currently possible and how.
        
        Returns:
            Tuple of (can_escape, best_route, required_actions)
        """
        viable_exits = self.get_viable_exits()
        
        if not viable_exits:
            # Check what's needed for each exit
            needed_actions = []
            for exit_name, exit_route in self.exits.items():
                if exit_route.discovered:
                    missing = []
                    for req in exit_route.requirements:
                        if req in self.puzzles and self.puzzles[req].status != PuzzleStatus.SOLVED:
                            missing.append(f"Solve {req}")
                        elif req in self.resources and not self.resources[req].discovered:
                            missing.append(f"Find {req}")
                    if missing:
                        needed_actions.append(f"{exit_name}: {', '.join(missing)}")
            
            return False, "No viable exit routes", needed_actions
        
        # Find best exit route (lowest risk, highest capacity)
        best_exit = None
        best_score = -1
        
        for exit_name in viable_exits:
            exit_route = self.exits[exit_name]
            # Score based on capacity (can save more people) and safety (lower risk)
            score = exit_route.capacity * 2 + (1.0 - exit_route.risk_level) + exit_route.success_probability
            if score > best_score:
                best_score = score
                best_exit = exit_name
        
        exit_route = self.exits[best_exit]
        return True, best_exit, [f"Use {best_exit} (capacity: {exit_route.capacity}, risk: {exit_route.risk_level:.1f})"]


# Example usage and testing
if __name__ == "__main__":
    # Test escape room state
    room = EscapeRoomState()
    
    print("🏠 Initial Room State:")
    print(f"   Time remaining: {room.time_remaining} minutes")
    print(f"   Available resources: {room.get_available_resources()}")
    print(f"   Solvable puzzles: {room.get_solvable_puzzles()}")
    print(f"   Viable exits: {room.get_viable_exits()}")
    
    # Test resource discovery
    room.discover_resource("tools")
    room.discover_resource("rope")
    print(f"\n🔍 After discovery: {room.get_available_resources()}")
    
    # Test puzzle attempt
    success, message = room.attempt_puzzle("debris_clearing", 2, 1)
    print(f"\n🧩 Puzzle attempt: {message}")
    
    # Test escape analysis
    can_escape, route, actions = room.can_escape_successfully()
    print(f"\n🚪 Escape analysis: {'Possible' if can_escape else 'Not possible'}")
    if can_escape:
        print(f"   Best route: {route}")
    else:
        print(f"   Required actions: {actions}")
    
    # Record state snapshot
    room.record_state_snapshot(1)
    print(f"\n📊 State recorded: {len(room.state_history)} snapshots")