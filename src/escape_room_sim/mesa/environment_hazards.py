"""
Environmental Hazards System

Agent D: State Management & Integration Specialist
Implements dynamic environmental hazards and challenges in the Mesa environment.
"""

import math
import random
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
import threading


class HazardType(Enum):
    """Types of environmental hazards"""
    FIRE = "fire"
    POISON_GAS = "poison_gas"
    SPIKE_TRAP = "spike_trap"
    ELECTRIC_SHOCK = "electric_shock"
    FLOODING = "flooding"
    TEMPERATURE_EXTREME = "temperature_extreme"
    DARKNESS = "darkness"
    TOXIC_SPILL = "toxic_spill"
    FALLING_DEBRIS = "falling_debris"
    ENERGY_DRAIN = "energy_drain"


class HazardSeverity(Enum):
    """Severity levels for hazards"""
    MINOR = "minor"
    MODERATE = "moderate"
    SEVERE = "severe"
    CRITICAL = "critical"


class TriggerType(Enum):
    """Types of hazard triggers"""
    TIME_BASED = "time_based"
    PRESSURE_PLATE = "pressure_plate"
    PROXIMITY = "proximity"
    INTERACTION = "interaction"
    RANDOM = "random"
    AGENT_COUNT = "agent_count"
    RESOURCE_DEPLETION = "resource_depletion"


@dataclass
class HazardEffect:
    """Effect of a hazard on agents or environment"""
    effect_type: str
    magnitude: float
    duration: float = 0.0  # 0 = instant effect
    area_of_effect: float = 1.0
    damage_per_turn: int = 0
    energy_drain_per_turn: float = 0.0
    movement_impairment: float = 0.0  # 0-1, reduces movement speed
    visibility_reduction: float = 0.0  # 0-1, reduces visibility
    special_effects: Dict[str, Any] = field(default_factory=dict)


@dataclass
class HazardTrigger:
    """Trigger conditions for hazards"""
    trigger_type: str
    conditions: Dict[str, Any] = field(default_factory=dict)
    probability: float = 1.0
    cooldown: float = 0.0  # seconds between triggers
    max_triggers: int = -1  # -1 = unlimited
    triggered_count: int = 0
    last_triggered: Optional[datetime] = None


class EnvironmentHazard:
    """
    Represents an environmental hazard in the escape room
    
    Hazards can be triggered by various conditions and affect agents
    within their area of effect.
    """
    
    def __init__(self, hazard_id: str, hazard_type: str, 
                 position: Tuple[int, int], room: str,
                 damage_per_turn: int = 0, radius: float = 1.0,
                 severity: str = HazardSeverity.MODERATE.value):
        self.hazard_id = hazard_id
        self.hazard_type = hazard_type
        self.position = position
        self.room = room
        self.severity = severity
        self.active = False
        self.permanent = False
        
        # Effect properties
        self.effect = HazardEffect(
            effect_type=hazard_type,
            magnitude=damage_per_turn,
            area_of_effect=radius,
            damage_per_turn=damage_per_turn
        )
        
        # Timing
        self.created_timestamp = datetime.now()
        self.activated_timestamp: Optional[datetime] = None
        self.duration = -1.0  # -1 = permanent until deactivated
        
        # Trigger system
        self.triggers: List[HazardTrigger] = []
        
        # Mitigation
        self.mitigation_tools: List[str] = []
        self.can_be_disabled = True
        self.disabled_by_tools: List[str] = []
        
        # Affected agents tracking
        self.affected_agents: Set[int] = set()
        self.total_damage_dealt = 0
        
        # Escalation
        self.escalation_rate = 0.0  # Damage increase per second
        self.max_escalation = 100
        
        self._configure_hazard_specifics()
    
    def _configure_hazard_specifics(self):
        """Configure hazard-specific properties"""
        if self.hazard_type == HazardType.FIRE.value:
            self.effect.damage_per_turn = 15
            self.effect.area_of_effect = 2.0
            self.effect.special_effects["spread_chance"] = 0.1
            self.mitigation_tools = ["fire_extinguisher", "water"]
            self.escalation_rate = 1.0
            
        elif self.hazard_type == HazardType.POISON_GAS.value:
            self.effect.damage_per_turn = 5
            self.effect.area_of_effect = 3.0
            self.effect.duration = 10.0
            self.mitigation_tools = ["gas_mask", "ventilation_tool"]
            
        elif self.hazard_type == HazardType.SPIKE_TRAP.value:
            self.effect.damage_per_turn = 25
            self.effect.area_of_effect = 1.0
            self.can_be_disabled = True
            self.disabled_by_tools = ["crowbar", "tools"]
            
        elif self.hazard_type == HazardType.ELECTRIC_SHOCK.value:
            self.effect.damage_per_turn = 20
            self.effect.energy_drain_per_turn = 0.3
            self.effect.area_of_effect = 1.5
            self.mitigation_tools = ["rubber_gloves", "insulation"]
            
        elif self.hazard_type == HazardType.FLOODING.value:
            self.effect.damage_per_turn = 8
            self.effect.movement_impairment = 0.5
            self.effect.area_of_effect = 4.0
            self.escalation_rate = 0.5
            
        elif self.hazard_type == HazardType.TEMPERATURE_EXTREME.value:
            self.effect.damage_per_turn = 10
            self.effect.energy_drain_per_turn = 0.2
            self.effect.area_of_effect = 2.5
            
        elif self.hazard_type == HazardType.DARKNESS.value:
            self.effect.damage_per_turn = 0
            self.effect.visibility_reduction = 0.8
            self.effect.movement_impairment = 0.3
            self.effect.area_of_effect = 5.0
            self.mitigation_tools = ["flashlight", "torch", "light_source"]
            
        elif self.hazard_type == HazardType.ENERGY_DRAIN.value:
            self.effect.damage_per_turn = 0
            self.effect.energy_drain_per_turn = 0.5
            self.effect.area_of_effect = 2.0
    
    def add_trigger(self, trigger_type: str, conditions: Dict[str, Any] = None,
                   probability: float = 1.0, cooldown: float = 0.0):
        """Add trigger condition for hazard activation"""
        trigger = HazardTrigger(
            trigger_type=trigger_type,
            conditions=conditions or {},
            probability=probability,
            cooldown=cooldown
        )
        self.triggers.append(trigger)
    
    def check_triggers(self, agents: List, current_time: datetime, 
                      environment_data: Dict[str, Any] = None) -> bool:
        """Check if any triggers should activate the hazard"""
        if self.active:
            return False
        
        for trigger in self.triggers:
            if self._evaluate_trigger(trigger, agents, current_time, environment_data):
                if random.random() <= trigger.probability:
                    return self._activate_hazard(trigger, current_time)
        
        return False
    
    def _evaluate_trigger(self, trigger: HazardTrigger, agents: List,
                         current_time: datetime, environment_data: Dict[str, Any]) -> bool:
        """Evaluate if a trigger condition is met"""
        # Check cooldown
        if trigger.last_triggered and trigger.cooldown > 0:
            time_since_trigger = (current_time - trigger.last_triggered).total_seconds()
            if time_since_trigger < trigger.cooldown:
                return False
        
        # Check max triggers
        if trigger.max_triggers > 0 and trigger.triggered_count >= trigger.max_triggers:
            return False
        
        # Evaluate specific trigger types
        if trigger.trigger_type == TriggerType.PROXIMITY.value:
            return self._check_proximity_trigger(trigger, agents)
        
        elif trigger.trigger_type == TriggerType.PRESSURE_PLATE.value:
            return self._check_pressure_plate_trigger(trigger, agents)
        
        elif trigger.trigger_type == TriggerType.TIME_BASED.value:
            return self._check_time_trigger(trigger, current_time)
        
        elif trigger.trigger_type == TriggerType.AGENT_COUNT.value:
            return self._check_agent_count_trigger(trigger, agents)
        
        elif trigger.trigger_type == TriggerType.INTERACTION.value:
            return self._check_interaction_trigger(trigger, environment_data)
        
        elif trigger.trigger_type == TriggerType.RANDOM.value:
            return self._check_random_trigger(trigger)
        
        return False
    
    def _check_proximity_trigger(self, trigger: HazardTrigger, agents: List) -> bool:
        """Check proximity-based trigger"""
        proximity_distance = trigger.conditions.get("distance", 2.0)
        
        for agent in agents:
            distance = math.sqrt((agent.pos[0] - self.position[0])**2 + 
                               (agent.pos[1] - self.position[1])**2)
            if distance <= proximity_distance:
                return True
        
        return False
    
    def _check_pressure_plate_trigger(self, trigger: HazardTrigger, agents: List) -> bool:
        """Check pressure plate trigger"""
        for agent in agents:
            if agent.pos == self.position:
                return True
        return False
    
    def _check_time_trigger(self, trigger: HazardTrigger, current_time: datetime) -> bool:
        """Check time-based trigger"""
        activation_time = trigger.conditions.get("activation_time")
        if activation_time:
            time_since_creation = (current_time - self.created_timestamp).total_seconds()
            return time_since_creation >= activation_time
        return False
    
    def _check_agent_count_trigger(self, trigger: HazardTrigger, agents: List) -> bool:
        """Check agent count trigger"""
        min_agents = trigger.conditions.get("min_agents", 1)
        max_agents = trigger.conditions.get("max_agents", float('inf'))
        
        agents_in_room = [a for a in agents if getattr(a, 'current_room', None) == self.room]
        agent_count = len(agents_in_room)
        
        return min_agents <= agent_count <= max_agents
    
    def _check_interaction_trigger(self, trigger: HazardTrigger, 
                                 environment_data: Dict[str, Any]) -> bool:
        """Check interaction-based trigger"""
        required_interaction = trigger.conditions.get("interaction_type")
        if not required_interaction or not environment_data:
            return False
        
        return environment_data.get("interaction_type") == required_interaction
    
    def _check_random_trigger(self, trigger: HazardTrigger) -> bool:
        """Check random trigger"""
        base_chance = trigger.conditions.get("base_chance", 0.01)
        return random.random() < base_chance
    
    def _activate_hazard(self, trigger: HazardTrigger, current_time: datetime) -> bool:
        """Activate the hazard"""
        self.active = True
        self.activated_timestamp = current_time
        
        # Update trigger
        trigger.triggered_count += 1
        trigger.last_triggered = current_time
        
        return True
    
    def apply_effects(self, agents: List) -> Dict[int, Dict[str, Any]]:
        """Apply hazard effects to agents in range"""
        if not self.active:
            return {}
        
        effects_applied = {}
        
        for agent in agents:
            if not agent.is_alive():
                continue
            
            # Check if agent is in range
            distance = math.sqrt((agent.pos[0] - self.position[0])**2 + 
                               (agent.pos[1] - self.position[1])**2)
            
            if distance <= self.effect.area_of_effect:
                # Check if agent has mitigation tools
                mitigation_factor = self._calculate_mitigation_factor(agent)
                
                # Apply damage
                damage = int(self.effect.damage_per_turn * mitigation_factor)
                if damage > 0:
                    agent.take_damage(damage)
                    self.total_damage_dealt += damage
                
                # Apply energy drain
                energy_drain = self.effect.energy_drain_per_turn * mitigation_factor
                if energy_drain > 0:
                    agent.consume_energy(energy_drain)
                
                # Track affected agent
                self.affected_agents.add(agent.unique_id)
                
                # Record effects
                effects_applied[agent.unique_id] = {
                    "damage": damage,
                    "energy_drain": energy_drain,
                    "movement_impairment": self.effect.movement_impairment,
                    "visibility_reduction": self.effect.visibility_reduction,
                    "mitigation_factor": mitigation_factor,
                    "distance": distance
                }
        
        return effects_applied
    
    def _calculate_mitigation_factor(self, agent) -> float:
        """Calculate mitigation factor based on agent's tools"""
        if not self.mitigation_tools:
            return 1.0
        
        mitigation = 1.0
        
        for tool in self.mitigation_tools:
            if agent.has_resource(tool):
                # Each mitigation tool reduces effect by 50%
                mitigation *= 0.5
        
        # Minimum 10% effect even with all mitigation
        return max(0.1, mitigation)
    
    def update(self, delta_time: float):
        """Update hazard state"""
        if not self.active:
            return
        
        current_time = datetime.now()
        
        # Check duration
        if self.duration > 0 and self.activated_timestamp:
            time_active = (current_time - self.activated_timestamp).total_seconds()
            if time_active >= self.duration:
                self.deactivate()
                return
        
        # Handle escalation
        if self.escalation_rate > 0:
            escalation_amount = self.escalation_rate * delta_time
            self.effect.damage_per_turn = min(
                self.max_escalation,
                self.effect.damage_per_turn + escalation_amount
            )
        
        # Handle spreading (for fire)
        if (self.hazard_type == HazardType.FIRE.value and 
            self.effect.special_effects.get("spread_chance", 0) > 0):
            self._attempt_fire_spread()
    
    def _attempt_fire_spread(self):
        """Attempt to spread fire to adjacent areas"""
        spread_chance = self.effect.special_effects.get("spread_chance", 0.1)
        if random.random() < spread_chance:
            # Would create new fire hazard at adjacent position
            # Implementation would depend on hazard manager
            pass
    
    def deactivate(self):
        """Deactivate the hazard"""
        self.active = False
    
    def disable(self, agent, tool_used: str = None) -> bool:
        """Attempt to disable the hazard"""
        if not self.can_be_disabled:
            return False
        
        if self.disabled_by_tools and tool_used:
            if tool_used in self.disabled_by_tools:
                self.deactivate()
                return True
        
        # Check if agent has appropriate tools
        for tool in self.disabled_by_tools:
            if agent.has_resource(tool):
                self.deactivate()
                return True
        
        return False
    
    def get_status(self) -> Dict[str, Any]:
        """Get hazard status information"""
        return {
            "hazard_id": self.hazard_id,
            "hazard_type": self.hazard_type,
            "position": self.position,
            "room": self.room,
            "active": self.active,
            "severity": self.severity,
            "damage_per_turn": self.effect.damage_per_turn,
            "area_of_effect": self.effect.area_of_effect,
            "affected_agents": list(self.affected_agents),
            "total_damage_dealt": self.total_damage_dealt,
            "can_be_disabled": self.can_be_disabled,
            "mitigation_tools": self.mitigation_tools,
            "triggers": len(self.triggers),
            "created": self.created_timestamp.isoformat(),
            "activated": self.activated_timestamp.isoformat() if self.activated_timestamp else None
        }


class HazardManager:
    """
    Manages all environmental hazards in the escape room
    
    Handles hazard creation, activation, updates, and interactions.
    """
    
    def __init__(self):
        self.hazards: Dict[str, EnvironmentHazard] = {}
        self.active_hazards: Set[str] = set()
        self.hazard_history: List[Dict[str, Any]] = []
        
        # Global hazard settings
        self.hazard_intensity_multiplier = 1.0
        self.enable_hazard_spreading = True
        self.max_active_hazards = 10
        
        # Statistics
        self.total_damage_dealt = 0
        self.total_agents_affected = set()
        self.hazards_triggered = 0
        self.hazards_disabled = 0
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Update timing
        self.last_update_time = datetime.now()
    
    def add_hazard(self, hazard: EnvironmentHazard):
        """Add hazard to the manager"""
        with self._lock:
            self.hazards[hazard.hazard_id] = hazard
    
    def remove_hazard(self, hazard_id: str) -> bool:
        """Remove hazard from the manager"""
        with self._lock:
            if hazard_id in self.hazards:
                if hazard_id in self.active_hazards:
                    self.active_hazards.remove(hazard_id)
                del self.hazards[hazard_id]
                return True
            return False
    
    def trigger_hazard(self, hazard_id: str, trigger_type: str = "manual",
                      environment_data: Dict[str, Any] = None) -> bool:
        """Manually trigger a hazard"""
        with self._lock:
            if hazard_id not in self.hazards:
                return False
            
            hazard = self.hazards[hazard_id]
            
            if hazard.active:
                return False
            
            # Create manual trigger
            current_time = datetime.now()
            manual_trigger = HazardTrigger(trigger_type="manual")
            
            success = hazard._activate_hazard(manual_trigger, current_time)
            if success:
                self.active_hazards.add(hazard_id)
                self.hazards_triggered += 1
                
                self._record_hazard_event("triggered", hazard_id, {
                    "trigger_type": trigger_type,
                    "manual": True
                })
            
            return success
    
    def step(self, agents: List):
        """Update all hazards (called each model step)"""
        current_time = datetime.now()
        delta_time = (current_time - self.last_update_time).total_seconds()
        self.last_update_time = current_time
        
        with self._lock:
            # Check triggers for inactive hazards
            for hazard_id, hazard in self.hazards.items():
                if not hazard.active:
                    triggered = hazard.check_triggers(agents, current_time)
                    if triggered:
                        self.active_hazards.add(hazard_id)
                        self.hazards_triggered += 1
                        
                        self._record_hazard_event("auto_triggered", hazard_id, {
                            "trigger_count": len(hazard.triggers)
                        })
            
            # Update active hazards
            hazards_to_deactivate = []
            
            for hazard_id in self.active_hazards:
                hazard = self.hazards[hazard_id]
                
                # Update hazard
                hazard.update(delta_time)
                
                # Apply effects to agents
                effects = hazard.apply_effects(agents)
                
                # Track statistics
                for agent_id, effect_data in effects.items():
                    self.total_damage_dealt += effect_data.get("damage", 0)
                    self.total_agents_affected.add(agent_id)
                
                # Check if hazard should be deactivated
                if not hazard.active:
                    hazards_to_deactivate.append(hazard_id)
            
            # Remove deactivated hazards
            for hazard_id in hazards_to_deactivate:
                self.active_hazards.remove(hazard_id)
                self._record_hazard_event("deactivated", hazard_id, {
                    "natural_expiration": True
                })
    
    def disable_hazard(self, hazard_id: str, agent, tool_used: str = None) -> bool:
        """Attempt to disable a hazard"""
        with self._lock:
            if hazard_id not in self.hazards:
                return False
            
            hazard = self.hazards[hazard_id]
            success = hazard.disable(agent, tool_used)
            
            if success:
                if hazard_id in self.active_hazards:
                    self.active_hazards.remove(hazard_id)
                
                self.hazards_disabled += 1
                
                self._record_hazard_event("disabled", hazard_id, {
                    "agent_id": agent.unique_id,
                    "tool_used": tool_used
                })
            
            return success
    
    def get_hazard(self, hazard_id: str) -> Optional[EnvironmentHazard]:
        """Get hazard by ID"""
        return self.hazards.get(hazard_id)
    
    def get_active_hazards(self) -> List[EnvironmentHazard]:
        """Get all active hazards"""
        with self._lock:
            return [self.hazards[hid] for hid in self.active_hazards]
    
    def get_hazards_in_room(self, room: str) -> List[EnvironmentHazard]:
        """Get all hazards in a specific room"""
        return [h for h in self.hazards.values() if h.room == room]
    
    def calculate_damage(self, agent) -> int:
        """Calculate total damage to agent from all active hazards"""
        total_damage = 0
        
        with self._lock:
            for hazard_id in self.active_hazards:
                hazard = self.hazards[hazard_id]
                
                # Check if agent is in range
                distance = math.sqrt((agent.pos[0] - hazard.position[0])**2 + 
                                   (agent.pos[1] - hazard.position[1])**2)
                
                if distance <= hazard.effect.area_of_effect:
                    mitigation_factor = hazard._calculate_mitigation_factor(agent)
                    damage = int(hazard.effect.damage_per_turn * mitigation_factor)
                    total_damage += damage
        
        return total_damage
    
    def create_environmental_challenge(self, room: str, challenge_type: str = "random"):
        """Create environmental challenge in room"""
        with self._lock:
            if len(self.active_hazards) >= self.max_active_hazards:
                return None
            
            # Determine hazard type
            if challenge_type == "random":
                hazard_type = random.choice(list(HazardType)).value
            else:
                hazard_type = challenge_type
            
            # Generate hazard
            hazard_id = f"challenge_{hazard_type}_{room}_{datetime.now().timestamp()}"
            position = self._get_random_position_in_room(room)
            
            hazard = EnvironmentHazard(
                hazard_id=hazard_id,
                hazard_type=hazard_type,
                position=position,
                room=room,
                severity=random.choice(list(HazardSeverity)).value
            )
            
            # Add random trigger
            trigger_types = [TriggerType.PROXIMITY, TriggerType.TIME_BASED, TriggerType.RANDOM]
            trigger_type = random.choice(trigger_types).value
            
            if trigger_type == TriggerType.PROXIMITY.value:
                hazard.add_trigger(trigger_type, {"distance": 2.0}, probability=0.8)
            elif trigger_type == TriggerType.TIME_BASED.value:
                hazard.add_trigger(trigger_type, {"activation_time": random.uniform(30, 120)})
            else:
                hazard.add_trigger(trigger_type, {"base_chance": 0.02})
            
            self.add_hazard(hazard)
            
            self._record_hazard_event("created", hazard_id, {
                "room": room,
                "hazard_type": hazard_type,
                "challenge_type": challenge_type
            })
            
            return hazard_id
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get hazard system statistics"""
        with self._lock:
            return {
                "total_hazards": len(self.hazards),
                "active_hazards": len(self.active_hazards),
                "hazards_triggered": self.hazards_triggered,
                "hazards_disabled": self.hazards_disabled,
                "total_damage_dealt": self.total_damage_dealt,
                "total_agents_affected": len(self.total_agents_affected),
                "hazard_types": {
                    htype.value: sum(1 for h in self.hazards.values() if h.hazard_type == htype.value)
                    for htype in HazardType
                },
                "severity_distribution": {
                    severity.value: sum(1 for h in self.hazards.values() if h.severity == severity.value)
                    for severity in HazardSeverity
                },
                "recent_events": self.hazard_history[-10:] if self.hazard_history else []
            }
    
    def _record_hazard_event(self, event_type: str, hazard_id: str, 
                           event_data: Dict[str, Any]):
        """Record hazard event in history"""
        event = {
            "event_type": event_type,
            "hazard_id": hazard_id,
            "timestamp": datetime.now().isoformat(),
            "data": event_data
        }
        
        self.hazard_history.append(event)
        
        # Keep limited history
        if len(self.hazard_history) > 100:
            self.hazard_history = self.hazard_history[-100:]
    
    def _get_random_position_in_room(self, room: str) -> Tuple[int, int]:
        """Get random position in room (placeholder)"""
        # This would integrate with room system
        return (random.randint(0, 9), random.randint(0, 9))