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
                failure_reason="Time has expired - no escape possible",
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
            self.threat_level = min(1.0, (time_ratio - 0.8) * 5.0)
        else:
            self.threat_level = time_ratio * 0.4
    
    def is_time_expired(self) -> bool:
        """Check if simulation time has expired."""
        return self.time_remaining <= 0
    
    def get_elapsed_time(self) -> int:
        """Get amount of time elapsed since simulation start."""
        return self.scenario.time_limit - self.time_remaining
    
    def get_current_threat_level(self) -> float:
        """Get current threat level (0.0 to 1.0)."""
        return self.threat_level
    
    def apply_time_pressure_effects(self) -> None:
        """Apply effects of current time pressure to the scenario."""
        # This method can be expanded to modify available actions,
        # increase resource costs, etc. based on desperation level
        pass
    
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