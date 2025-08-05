"""
CompetitiveEscapeRoom orchestrator for managing competitive survival scenarios.

This module implements the main orchestrator that integrates all competitive
subsystems (ResourceManager, TrustTracker, MoralDilemmaEngine, InformationBroker)
to create a complete competitive escape room experience with single-survivor
mechanics, time pressure, and escalating consequences.
"""
from typing import Optional, Dict, List, Any
from datetime import datetime, timedelta
from src.escape_room_sim.competitive.models import (
    CompetitiveScenario, EscapeMethod, EscapeResult, MoralChoice, MoralDilemma
)
from src.escape_room_sim.competitive.resource_manager import ResourceManager, ClaimResult
from src.escape_room_sim.competitive.trust_tracker import TrustTracker
from src.escape_room_sim.competitive.moral_dilemma_engine import MoralDilemmaEngine
from src.escape_room_sim.competitive.information_broker import InformationBroker
from src.escape_room_sim.competitive.agent_behavior import CompetitiveAgentBehavior


class CompetitiveEscapeRoom:
    """
    Main orchestrator for competitive escape room scenarios.
    
    Integrates all competitive subsystems to create a complete survival experience
    with single-survivor mechanics, resource scarcity, moral dilemmas, information
    asymmetry, and time pressure with escalating consequences.
    """
    
    def __init__(self, scenario: CompetitiveScenario):
        """
        Initialize CompetitiveEscapeRoom with a competitive scenario.
        
        Args:
            scenario: CompetitiveScenario containing all game configuration
            
        Raises:
            ValueError: If scenario is None or has invalid configuration
        """
        if scenario is None:
            raise ValueError("Scenario cannot be None")
        
        if scenario.time_limit <= 0:
            raise ValueError("Time limit must be positive")
        
        self.scenario = scenario
        self.time_limit = scenario.time_limit  # Add time_limit property for tests
        
        # Initialize all subsystems with scenario data
        self.resource_manager = ResourceManager(scenario.resources)
        
        # Initialize trust tracker with expected agents
        expected_agents = ["strategist", "mediator", "survivor"]
        self.trust_tracker = TrustTracker(expected_agents)
        
        self.moral_engine = MoralDilemmaEngine(scenario.moral_dilemmas)
        self.info_broker = InformationBroker(scenario.secret_information)
        
        # Initialize competitive agent behavior system
        self.agent_behavior = CompetitiveAgentBehavior()
        
        # Initialize escape state
        self.escaped_agent: Optional[str] = None
        self.time_remaining: int = scenario.time_limit
        self.simulation_start_time: datetime = datetime.now()
        
        # Initialize time pressure mechanics
        self.desperation_level: float = 0.0
        self.threat_level: float = 0.0
    
    def attempt_escape(self, agent_id: str, escape_method: EscapeMethod) -> EscapeResult:
        """
        Process escape attempt and enforce single-survivor rule.
        
        Args:
            agent_id: ID of the agent attempting to escape
            escape_method: EscapeMethod being attempted
            
        Returns:
            EscapeResult with success status and details
            
        Raises:
            ValueError: If agent_id is empty or escape_method is None
        """
        if not agent_id or not agent_id.strip():
            raise ValueError("Agent ID cannot be empty")
        
        if escape_method is None:
            raise ValueError("Escape method cannot be None")
        
        # Check if time has expired
        if self.is_time_expired():
            return EscapeResult(
                success=False,
                agent_id=agent_id,
                escape_method=escape_method.id,
                failure_reason="Time has expired - simulation has failed",
                time_remaining=self.time_remaining
            )
        
        # Enforce single-survivor rule
        if self.escaped_agent is not None:
            return EscapeResult(
                success=False,
                agent_id=agent_id,
                escape_method=escape_method.id,
                failure_reason="Another agent has already escaped",
                time_remaining=self.time_remaining
            )
        
        # Check if agent has all requirements (resources + information)
        missing_requirements = []
        
        # Check each requirement - could be resource or information
        agent_knowledge = {secret.id for secret in self.info_broker.get_agent_knowledge(agent_id)}
        agent_resources = set(self.resource_manager.get_agent_resources(agent_id))
        
        for requirement in escape_method.requirements:
            # Check if agent has this requirement either as resource or information
            has_resource = requirement in agent_resources
            has_information = requirement in agent_knowledge
            
            # If it's neither a resource nor information the agent has, it's missing
            if not has_resource and not has_information:
                missing_requirements.append(requirement)
        
        if missing_requirements:
            return EscapeResult(
                success=False,
                agent_id=agent_id,
                escape_method=escape_method.id,
                failure_reason=f"Missing required resources/information: {', '.join(missing_requirements)}",
                time_remaining=self.time_remaining
            )
        
        # Successful escape
        self.escaped_agent = agent_id
        
        # Separate resources and information used from requirements
        resources_used = []
        information_used = []
        
        for requirement in escape_method.requirements:
            if requirement in agent_resources:
                resources_used.append(requirement)
            if requirement in agent_knowledge:
                information_used.append(requirement)
        
        return EscapeResult(
            success=True,
            agent_id=agent_id,
            escape_method=escape_method.id,
            time_remaining=self.time_remaining,
            resources_used=resources_used,
            information_used=information_used
        )
    
    def process_resource_claim(self, agent_id: str, resource_id: str) -> ClaimResult:
        """
        Handle resource acquisition with scarcity enforcement.
        
        Args:
            agent_id: ID of the agent claiming the resource
            resource_id: ID of the resource being claimed
            
        Returns:
            ClaimResult with success status and details
            
        Raises:
            ValueError: If agent_id or resource_id is empty
        """
        if not agent_id or not agent_id.strip():
            raise ValueError("Agent ID cannot be empty")
        
        if not resource_id or not resource_id.strip():
            raise ValueError("Resource ID cannot be empty")
        
        # Check if time has expired
        if self.is_time_expired():
            return ClaimResult(
                success=False,
                message="Time has expired - simulation has failed",
                agent_id=agent_id,
                resource_id=resource_id
            )
        
        # Use ResourceManager's claim_resource method directly
        return self.resource_manager.claim_resource(agent_id, resource_id)
    
    def present_moral_choice(self, agent_id: str, context: Dict[str, Any]) -> Optional[MoralDilemma]:
        """
        Offer ethical dilemma with survival implications.
        
        Args:
            agent_id: ID of the agent being presented with the choice
            context: Current context for selecting appropriate dilemma
            
        Returns:
            MoralDilemma if one matches the context, None otherwise
            
        Raises:
            ValueError: If agent_id is empty or context is None
        """
        if not agent_id or not agent_id.strip():
            raise ValueError("Agent ID cannot be empty")
        
        if context is None:
            raise ValueError("Context cannot be None")
        
        return self.moral_engine.present_dilemma(agent_id, context)
    
    def process_moral_choice(self, agent_id: str, choice: MoralChoice):
        """
        Process a moral choice made by an agent.
        
        Args:
            agent_id: ID of the agent making the choice
            choice: MoralChoice being made
            
        Returns:
            ChoiceConsequences of the moral decision
        """
        return self.moral_engine.process_choice(agent_id, choice)
    
    def advance_time(self, time_units: int) -> None:
        """
        Advance simulation time and update time pressure.
        
        Args:
            time_units: Amount of time to advance
        """
        self.time_remaining = max(0, self.time_remaining - time_units)
        self.update_time_pressure()
        
        # Update subsystems with time expiration status
        if self.is_time_expired():
            self.resource_manager.set_time_expired(True)
            self.info_broker.set_time_expired(True)
    
    def update_time_pressure(self) -> None:
        """Update desperation level and threat level based on remaining time."""
        if self.scenario.time_limit <= 0:
            self.desperation_level = 1.0
            self.threat_level = 1.0
            return
        
        # Calculate time pressure as ratio of time passed
        time_passed = self.scenario.time_limit - self.time_remaining
        time_ratio = time_passed / self.scenario.time_limit
        
        # Desperation increases exponentially as time runs out
        self.desperation_level = min(1.0, time_ratio ** 0.5)
        
        # Threat level increases more dramatically in final stages
        if time_ratio > 0.8:
            self.threat_level = min(1.0, 0.5 + (time_ratio - 0.8) * 2.5)  # 0.5 to 1.0 in final 20%
        elif time_ratio > 0.5:
            self.threat_level = min(1.0, time_ratio * 1.4)  # Escalate more aggressively after 50%
        else:
            self.threat_level = time_ratio * 0.4  # Low threat in first half
    
    def is_time_expired(self) -> bool:
        """Check if simulation time has expired."""
        return self.time_remaining <= 0
    
    def get_elapsed_time(self) -> int:
        """Get amount of time elapsed since simulation start."""
        return self.scenario.time_limit - self.time_remaining
    
    def get_current_threat_level(self) -> float:
        """Get current threat level (0.0 to 1.0)."""
        return self.threat_level
    
    def get_time_pressure_level(self) -> float:
        """Get current time pressure level (0.0 to 1.0)."""
        return self.desperation_level
    
    def apply_time_pressure_effects(self) -> None:
        """Apply effects of current time pressure to the scenario."""
        time_ratio = (self.scenario.time_limit - self.time_remaining) / self.scenario.time_limit
        
        # Restrict information sharing in final stages  
        if time_ratio > 0.9:
            self.info_broker.set_sharing_restrictions(False)
        else:
            self.info_broker.set_sharing_restrictions(True)
        
        # Update time expiration status for all subsystems
        if self.is_time_expired():
            self.resource_manager.set_time_expired(True)
            self.info_broker.set_time_expired(True)
    
    def apply_threat_escalation(self) -> None:
        """Apply threat escalation effects based on current threat level."""
        # Update available resources based on threat level
        # Add new obstacles based on threat level
        # Trigger emergency protocols if needed
        
        # Trigger obstacle creation by accessing the obstacle list
        obstacles = self.get_active_obstacles()
        
        # The get_active_obstacles method automatically creates obstacles based on threat level
        # So calling it will trigger the escalation effects
    
    def get_active_obstacles(self) -> List[Dict[str, Any]]:
        """Get list of currently active obstacles."""
        if not hasattr(self, '_active_obstacles'):
            self._active_obstacles = []
        
        # Add obstacles based on threat level (lower thresholds for more responsive system)
        if self.threat_level > 0.1 and len(self._active_obstacles) == 0:
            self._active_obstacles.append({
                "type": "security_patrol",
                "severity": min(1.0, self.threat_level * 1.5),
                "description": "Security patrol making rounds"
            })
        
        if self.threat_level > 0.4:
            if not any(obs["type"] == "power_failure" for obs in self._active_obstacles):
                self._active_obstacles.append({
                    "type": "power_failure",
                    "severity": self.threat_level,
                    "description": "Partial power failure affecting systems"
                })
        
        if self.threat_level > 0.6:
            if not any(obs["type"] == "structural_damage" for obs in self._active_obstacles):
                self._active_obstacles.append({
                    "type": "structural_damage", 
                    "severity": self.threat_level,
                    "description": "Structural damage blocking some routes"
                })
        
        return self._active_obstacles
    
    def get_viable_escape_methods(self, agent_id: str) -> List[EscapeMethod]:
        """Get escape methods that are currently viable for an agent."""
        viable_methods = []
        
        for method in self.scenario.escape_methods:
            # Check if method is blocked by obstacles
            blocked = False
            obstacles = self.get_active_obstacles()
            
            for obstacle in obstacles:
                if obstacle["type"] == "structural_damage" and method.id == "side_exit":
                    blocked = True
                elif obstacle["type"] == "power_failure" and method.id == "main_exit":
                    blocked = True
            
            if not blocked:
                viable_methods.append(method)
        
        return viable_methods
    
    def is_emergency_protocol_active(self) -> bool:
        """Check if emergency protocols are currently active."""
        return self.threat_level >= 0.7  # Lower threshold for more responsive system
    
    def get_emergency_effects(self) -> List[str]:
        """Get list of active emergency effects."""
        effects = []
        if self.is_emergency_protocol_active():
            effects.append("resource_lockdown")
            if self.threat_level > 0.9:
                effects.append("communication_jamming")
        return effects
    
    def get_escalation_warnings(self) -> List[str]:
        """Get list of escalation warnings."""
        warnings = []
        
        if self.threat_level > 0.4:
            warnings.append("Threat level increasing - obstacles may appear")
        
        if self.threat_level > 0.6:
            warnings.append("High threat detected - escalation imminent")
        
        if self.threat_level > 0.8:
            warnings.append("Critical threat level - emergency protocols may activate")
        
        return warnings
    
    # Desperation Level Calculation Methods
    
    def calculate_agent_desperation(self, agent_id: str) -> float:
        """Calculate desperation level for a specific agent."""
        factors = self.get_desperation_factors(agent_id)
        return min(1.0, sum(factors.values()))
    
    def get_desperation_factors(self, agent_id: str) -> Dict[str, float]:
        """Get factors contributing to agent desperation."""
        factors = {}
        
        # Time pressure factor (scaled to not exceed 0.4)
        factors["time_pressure"] = min(0.4, self.desperation_level)
        
        # Resource scarcity factor
        agent_resources = len(self.resource_manager.get_agent_resources(agent_id))
        total_resources = len(self.scenario.resources)
        resource_ratio = agent_resources / max(1, total_resources)
        factors["resource_scarcity"] = max(0.0, min(0.3, 0.7 - resource_ratio))  # Cap at 0.3
        
        # Social isolation factor (based on trust relationships)
        avg_trust = 0.0
        trust_count = 0
        for other_agent in ["strategist", "mediator", "survivor"]:
            if other_agent != agent_id:
                trust_level = self.trust_tracker.get_trust_level(agent_id, other_agent)
                avg_trust += trust_level
                trust_count += 1
        
        if trust_count > 0:
            avg_trust /= trust_count
            factors["social_isolation"] = max(0.0, min(0.2, 0.5 - avg_trust))  # Cap at 0.2
        else:
            factors["social_isolation"] = 0.2
        
        # Escape difficulty factor
        viable_methods = len(self.get_viable_escape_methods(agent_id))
        total_methods = len(self.scenario.escape_methods)
        if total_methods > 0:
            difficulty_ratio = 1.0 - (viable_methods / total_methods)
            factors["escape_difficulty"] = min(0.1, difficulty_ratio * 0.4)  # Cap at 0.1
        else:
            factors["escape_difficulty"] = 0.1
        
        return factors
    
    def get_moral_choice_threshold(self, agent_id: str) -> float:
        """Get moral choice threshold for an agent (lower = more likely to make selfish choices)."""
        desperation = self.calculate_agent_desperation(agent_id)
        # Base threshold of 0.5, reduced by desperation
        return max(0.1, 0.5 - (desperation * 0.4))
    
    def get_cooperation_likelihood(self, agent1: str, agent2: str) -> float:
        """Get likelihood of cooperation between two agents."""
        # Base cooperation from trust level
        trust_level = self.trust_tracker.get_trust_level(agent1, agent2)
        base_cooperation = max(0.1, (trust_level + 1.0) / 2.0)  # Convert -1,1 to 0.1,1.0
        
        # Reduce cooperation based on desperation
        agent1_desperation = self.calculate_agent_desperation(agent1)
        desperation_penalty = agent1_desperation * 0.4  # Increased penalty for stronger effect
        
        # Also reduce based on overall time pressure
        time_pressure_penalty = self.desperation_level * 0.2
        
        return max(0.1, base_cooperation - desperation_penalty - time_pressure_penalty)
    
    # Option Reduction Mechanics
    
    def get_available_options(self, agent_id: str) -> List[Dict[str, Any]]:
        """Get all available options for an agent."""
        options = []
        
        if not self.is_time_expired():
            # Basic action options
            options.append({"type": "claim_resource", "critical": True, "description": "Claim available resources"})
            options.append({"type": "share_information", "critical": False, "description": "Share information with others"})
            options.append({"type": "attempt_escape", "critical": True, "description": "Attempt to escape"})
            
            # Reduce options based on time pressure
            time_ratio = (self.scenario.time_limit - self.time_remaining) / self.scenario.time_limit
            
            if time_ratio > 0.5:  # After 50% time
                # Remove non-critical options first
                options = [opt for opt in options if opt.get("critical", False) or time_ratio < 0.7]
            
            if time_ratio > 0.8:  # In final 20%
                # Only most critical options remain
                options = [opt for opt in options if opt.get("critical", False)]
        
        return options
    
    def get_available_escape_methods(self, agent_id: str) -> List[EscapeMethod]:
        """Get escape methods available to an agent (may be reduced over time)."""
        if self.is_time_expired():
            return []
        
        methods = self.get_viable_escape_methods(agent_id)
        
        # Time-based method reduction
        time_ratio = (self.scenario.time_limit - self.time_remaining) / self.scenario.time_limit
        
        if time_ratio > 0.75:  # After 75% time elapsed
            # Some methods become unavailable due to time pressure
            if len(methods) > 1:
                # Remove the most difficult methods (those with most requirements)
                methods = sorted(methods, key=lambda m: len(m.requirements))
                methods = methods[:max(1, len(methods) - 1)]  # Keep at least 1
        
        return methods
    
    def get_disabled_escape_methods(self) -> List[str]:
        """Get list of escape methods that have been disabled."""
        all_methods = {method.id for method in self.scenario.escape_methods}
        available_methods = {method.id for method in self.get_available_escape_methods("strategist")}  # Use any agent
        return list(all_methods - available_methods)
    
    def get_information_sharing_cost(self) -> float:
        """Get cost/difficulty of sharing information (increases over time)."""
        time_ratio = (self.scenario.time_limit - self.time_remaining) / self.scenario.time_limit
        
        if time_ratio > 0.9:
            return 1.0  # Very high cost in final moments
        elif time_ratio > 0.7:
            return 0.5  # Moderate cost
        else:
            return 0.0  # No cost early on
    
    # Automatic Failure Conditions
    
    def is_simulation_failed(self) -> bool:
        """Check if simulation has automatically failed."""
        return (self.is_time_expired() or 
                self.check_failure_condition("all_agents_incapacitated") or
                self.check_failure_condition("critical_failure"))
    
    def get_failure_reason(self) -> str:
        """Get reason for simulation failure."""
        if self.is_time_expired():
            return "time_expired"
        elif self.check_failure_condition("all_agents_incapacitated"):
            return "all_agents_incapacitated"
        elif self.check_failure_condition("critical_failure"):
            return "critical_failure"
        else:
            return "unknown"
    
    def get_failed_agents(self) -> List[str]:
        """Get list of agents that have failed."""
        if self.is_simulation_failed():
            return ["strategist", "mediator", "survivor"]  # All agents fail on simulation failure
        else:
            return []
    
    def get_failure_state(self) -> Dict[str, Any]:
        """Get comprehensive failure state information."""
        elapsed_time = self.get_elapsed_time()
        # If time expired, elapsed time should be at least the time limit + 1
        if self.is_time_expired():
            elapsed_time = max(elapsed_time, self.scenario.time_limit + 1)
        return {
            "reason": self.get_failure_reason(),
            "time_elapsed": elapsed_time,
            "agent_states": {
                agent: {"resources": len(self.resource_manager.get_agent_resources(agent))}
                for agent in ["strategist", "mediator", "survivor"]
            },
            "resources_claimed": {
                agent: self.resource_manager.get_agent_resources(agent)
                for agent in ["strategist", "mediator", "survivor"]
            },
            "threat_level": self.threat_level,
            "desperation_level": self.desperation_level
        }
    
    def get_failure_criteria(self) -> List[str]:
        """Get list of possible failure criteria."""
        return ["time_expired", "all_agents_incapacitated", "critical_failure"]
    
    def check_failure_condition(self, condition: str) -> bool:
        """Check if a specific failure condition is met."""
        if condition == "time_expired":
            return self.is_time_expired()
        elif condition == "all_agents_incapacitated":
            # For now, return False - could be implemented based on agent health/status
            return False
        elif condition == "critical_failure":
            # Critical failure could be based on extreme threat levels
            return self.threat_level >= 1.0
        else:
            return False
    
    def is_in_grace_period(self) -> bool:
        """Check if simulation is in grace period before final failure."""
        return self.time_remaining <= 10 and self.time_remaining > 0
    
    def get_grace_period_remaining(self) -> int:
        """Get remaining time in grace period."""
        if self.is_in_grace_period():
            return self.time_remaining
        else:
            return 0
    
    def get_available_actions(self, agent_id: str) -> List[str]:
        """
        Get list of available actions for an agent.
        
        Args:
            agent_id: ID of the agent to query
            
        Returns:
            List of available action identifiers
        """
        actions = []
        
        # Basic actions are always available if time hasn't expired
        if not self.is_time_expired():
            actions.extend(["claim_resource", "share_information", "attempt_escape"])
            
            # Moral choice action available if there are matching dilemmas
            context = {"time_pressure": self.desperation_level}
            if self.present_moral_choice(agent_id, context) is not None:
                actions.append("make_moral_choice")
        
        # High desperation might reduce available actions
        if self.desperation_level > 0.8:
            # Remove some actions to simulate panic/reduced options
            if "share_information" in actions and self.desperation_level > 0.9:
                actions.remove("share_information")
        
        return actions
    
    def get_trust_relationships(self) -> Dict[str, Dict[str, float]]:
        """
        Get current trust relationships between all agents.
        
        Returns:
            Nested dictionary of trust levels
        """
        relationships = {}
        
        agents = self.trust_tracker.get_all_agents()
        for agent1 in agents:
            relationships[agent1] = {}
            for agent2 in agents:
                if agent1 != agent2:
                    relationships[agent1][agent2] = self.trust_tracker.get_trust_level(agent1, agent2)
        
        return relationships
    
    def get_scenario_status(self) -> Dict[str, Any]:
        """
        Get comprehensive status of the current scenario.
        
        Returns:
            Dictionary containing scenario state information
        """
        return {
            "time_remaining": self.time_remaining,
            "time_elapsed": self.get_elapsed_time(),
            "desperation_level": self.desperation_level,
            "threat_level": self.threat_level,
            "escaped_agent": self.escaped_agent,
            "agents_alive": len(self.trust_tracker.get_all_agents()),
            "resources_claimed": len(self.resource_manager.ownership),
            "moral_choices_made": len(self.moral_engine.choices_made),
            "information_shared": len(self.info_broker.sharing_history),
            "is_complete": self.escaped_agent is not None or self.is_time_expired()
        }
    
    # Competitive Agent Behavior Methods
    
    def get_agent_competitive_behavior(self, agent_id: str):
        """Get current competitive behavior for an agent."""
        return self.agent_behavior.get_agent_competitive_behavior(agent_id, self.desperation_level, self.trust_tracker)
    
    def make_agent_decision(self, agent_id: str, decision_type: str, context: Dict[str, Any] = None):
        """Make an agent decision based on personality and competitive context."""
        if context is None:
            context = {}
        
        # Add current competitive context
        context.update({
            "time_pressure": self.desperation_level,
            "threat_level": self.threat_level,
            "resources_available": len(self.resource_manager.get_available_resources(agent_id)),
            "trust_relationships": self.get_trust_relationships().get(agent_id, {})
        })
        
        return self.agent_behavior.make_agent_decision(agent_id, decision_type, context)
    
    def make_cooperation_decision(self, agent_id: str, target_agent: str):
        """Make cooperation decision based on trust levels and personality."""
        # Get trust level from agent_id towards target_agent
        trust_level = self.trust_tracker.get_trust_level(agent_id, target_agent)
        
        # If no direct trust relationship, use reverse trust as primary indicator
        if trust_level == 0.0:
            # Use reverse trust as primary indicator (what the target thinks of me indicates cooperation likelihood)
            reverse_trust = self.trust_tracker.get_trust_level(target_agent, agent_id)
            reputation = self.trust_tracker.calculate_reputation(target_agent)
            
            # If reverse trust exists, weight it heavily (target's opinion of me matters for cooperation)
            if reverse_trust != 0.0:
                trust_level = reverse_trust * 0.8 + reputation * 0.2
            else:
                # Fallback to reputation if no reverse trust
                trust_level = reputation
        
        context = {"trust_level": trust_level}
        return self.agent_behavior.make_agent_decision(agent_id, "cooperation", context)
    
    def make_moral_choice_decision(self, agent_id: str, moral_choice: MoralChoice):
        """Make moral choice decision based on agent personality."""
        context = {
            "moral_choice": moral_choice,
            "time_pressure": self.desperation_level,
            "survival_stakes": self.threat_level
        }
        return self.agent_behavior.make_agent_decision(agent_id, "moral_choice", context)
    
    def adapt_agent_strategy(self, agent_id: str, feedback: Dict[str, Any]):
        """Adapt agent strategy based on competitive feedback."""
        # Convert feedback to learning data format
        learning_data = {
            "cooperation_outcomes": feedback.get("cooperation_history", []),
            "trust_violations": feedback.get("resource_loss_incidents", 0) + feedback.get("trust_betrayals", 0),
            "successful_partnerships": feedback.get("cooperation_success_rate", 0) * 10  # Scale to count
        }
        
        updated_behavior = self.agent_behavior.apply_competitive_learning(agent_id, learning_data)
        
        # Convert to expected format
        return {
            "cooperation_willingness": updated_behavior.get("trust_baseline", 0.5),
            "resource_sharing": 1.0 - updated_behavior.get("cooperation_selectivity", 0.5),
            "strategy_changes": ["competitive_adaptation"] if updated_behavior.get("learning_applied") else []
        }
    
    def make_resource_conflict_decision(self, agent_id: str, resource_id: str):
        """Make decision about resource conflicts."""
        context = {
            "resource_id": resource_id,
            "current_owner": self.resource_manager.get_resource_owner(resource_id),
            "resource_importance": len([method for method in self.scenario.escape_methods 
                                      if resource_id in method.requirements])
        }
        return self.agent_behavior.make_resource_conflict_decision(agent_id, resource_id, context)
    
    def evaluate_cooperation_offer(self, agent_id: str, offer: Dict[str, Any]):
        """Evaluate a cooperation offer for deception and value."""
        return self.agent_behavior.evaluate_cooperation_offer(agent_id, offer)
    
    def get_agent_personality_profile(self, agent_id: str):
        """Get personality profile metrics for an agent."""
        return self.agent_behavior.get_agent_personality_profile(agent_id)
    
    def get_adapted_cooperation_strategies(self, agent_id: str):
        """Get adapted cooperation strategies based on trust history."""
        return self.agent_behavior.get_adapted_cooperation_strategies(agent_id)
    
    def apply_competitive_learning(self, agent_id: str, learning_data: Dict[str, Any]):
        """Apply competitive learning to update agent behavior."""
        return self.agent_behavior.apply_competitive_learning(agent_id, learning_data)
    
    # Time Pressure Effects on Agent Behavior Methods
    
    def get_average_decision_time(self, agent_id: str) -> float:
        """Get average decision time for an agent (affected by time pressure)."""
        base_decision_time = 5.0  # Base decision time in seconds
        pressure_factor = self.get_time_pressure_level()
        
        # Higher pressure = faster decisions (lower time)
        pressure_multiplier = max(0.2, 1.0 - (pressure_factor * 0.8))
        return base_decision_time * pressure_multiplier
    
    def get_risk_tolerance(self, agent_id: str) -> float:
        """Get risk tolerance for an agent (increases with time pressure)."""
        base_risk_tolerance = 0.3  # Base risk tolerance
        pressure_factor = self.get_time_pressure_level()
        
        # Higher pressure = higher risk tolerance
        return min(1.0, base_risk_tolerance + (pressure_factor * 0.6))
    
    def get_pressure_response(self, agent_id: str) -> Dict[str, Any]:
        """Get personality-specific pressure response for an agent."""
        pressure_level = self.get_time_pressure_level()
        
        if agent_id == "strategist":
            dominant_trait = "analysis_paralysis" if pressure_level > 0.7 else "calculated_planning"
        elif agent_id == "mediator":
            dominant_trait = "panic_cooperation" if pressure_level > 0.7 else "supportive_mediation"
        elif agent_id == "survivor":
            dominant_trait = "aggressive_selfishness" if pressure_level > 0.7 else "cautious_self_preservation"
        else:
            dominant_trait = "unknown_response"
        
        return {
            "dominant_trait": dominant_trait,
            "pressure_level": pressure_level,
            "response_intensity": min(1.0, pressure_level * 1.2)
        }
    
    def get_moral_choice_bias(self, agent_id: str) -> float:
        """Get moral choice bias for an agent (lower = more selfish under pressure)."""
        base_bias = 0.6  # Base moral bias (toward altruistic choices)
        pressure_factor = self.get_time_pressure_level()
        
        # Higher pressure = lower moral bias (more selfish)
        return max(0.1, base_bias - (pressure_factor * 0.4))
    
    def get_panic_level(self, agent_id: str) -> float:
        """Get panic level for an agent based on extreme time pressure."""
        pressure_level = self.get_time_pressure_level()
        desperation = self.calculate_agent_desperation(agent_id)
        
        # Panic is combination of time pressure and personal desperation
        panic_base = (pressure_level + desperation) / 2.0
        
        # Panic kicks in more strongly when pressure > 0.8
        if pressure_level > 0.8:
            panic_multiplier = 1.0 + ((pressure_level - 0.8) * 2.0)  # Up to 1.4x multiplier
            return min(1.0, panic_base * panic_multiplier)
        else:
            return panic_base
    
    def get_active_panic_behaviors(self, agent_id: str) -> List[str]:
        """Get list of active panic behaviors for an agent."""
        panic_level = self.get_panic_level(agent_id)
        behaviors = []
        
        if panic_level > 0.6:
            behaviors.append("erratic_decisions")
        
        if panic_level > 0.7:
            behaviors.append("resource_hoarding")
        
        if panic_level > 0.8:
            behaviors.append("trust_breakdown")
        
        if panic_level > 0.9:
            behaviors.append("desperate_actions")
        
        return behaviors