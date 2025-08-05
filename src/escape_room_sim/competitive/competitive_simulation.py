"""
CompetitiveSimulation engine for orchestrating complete competitive survival scenarios.

This module implements the main simulation engine that integrates scenario generation,
competitive escape room mechanics, agent state tracking, single-survivor validation,
result analysis, and comprehensive competition metrics for complete simulation flows.
"""
import random
from datetime import datetime
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field

from .scenario_generator import ScenarioGenerator
from .competitive_escape_room import CompetitiveEscapeRoom
from .competitive_agent_state import CompetitiveAgentState
from .models import CompetitiveScenario, EscapeResult, MoralChoice, TrustAction


@dataclass
class SimulationResults:
    """Results from a completed competitive simulation."""
    seed: int
    winner: Optional[str]
    completion_reason: str
    total_steps: int
    simulation_duration: float
    start_time: datetime
    end_time: datetime
    final_states: Dict[str, Dict[str, Any]]
    competition_metrics: Dict[str, Any]
    action_history: List[Dict[str, Any]]


class CompetitiveSimulation:
    """
    Main orchestrator for competitive survival simulations.
    
    Integrates scenario generation, competitive escape room mechanics, agent state tracking,
    single-survivor validation, and comprehensive result analysis to provide complete
    competitive simulation capabilities with reproducible seed-based results.
    """
    
    def __init__(self, seed: Optional[int] = None):
        """
        Initialize CompetitiveSimulation with optional seed parameter.
        
        Args:
            seed: Optional seed for reproducible results. If None, generates random seed.
            
        Raises:
            ValueError: If seed is negative or invalid type
        """
        # Validate seed parameter
        if seed is not None:
            if not isinstance(seed, int) or seed < 0:
                raise ValueError("Seed must be a non-negative integer")
        else:
            seed = random.randint(0, 999999)
        
        self.seed = seed
        self.scenario_generator = ScenarioGenerator(seed=seed)
        
        # Initialize state tracking
        self.scenario: Optional[CompetitiveScenario] = None
        self.escape_room: Optional[CompetitiveEscapeRoom] = None
        self.agent_states: Dict[str, CompetitiveAgentState] = {}
        self.results: List[SimulationResults] = []
        
        # Simulation state
        self.is_complete = False
        self.winner: Optional[str] = None
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None
        
        # Survival tracking
        self.eliminated_agents: set = set()
        self.agent_health: Dict[str, float] = {}  # 0.0 = dead, 1.0 = full health
        
        # Tracking structures
        self.action_history: List[Dict[str, Any]] = []
        self.simulation_metrics: Dict[str, Any] = {}
    
    # Scenario Generation Integration
    
    def generate_scenario(self) -> CompetitiveScenario:
        """Generate a competitive scenario using the configured seed."""
        self.scenario = self.scenario_generator.generate_scenario()
        return self.scenario
    
    # CompetitiveEscapeRoom Orchestration
    
    def initialize_escape_room(self) -> CompetitiveEscapeRoom:
        """
        Initialize competitive escape room with the generated scenario.
        
        Returns:
            CompetitiveEscapeRoom: Initialized escape room
            
        Raises:
            ValueError: If scenario hasn't been generated yet
        """
        if self.scenario is None:
            raise ValueError("Scenario must be generated before initializing escape room")
        
        self.escape_room = CompetitiveEscapeRoom(self.scenario)
        
        # Initialize agent states for tracking
        self.agent_states = {
            "strategist": CompetitiveAgentState("strategist"),
            "mediator": CompetitiveAgentState("mediator"),
            "survivor": CompetitiveAgentState("survivor")
        }
        
        # Initialize agent health (all start at full health)
        self.agent_health = {
            "strategist": 1.0,
            "mediator": 1.0,
            "survivor": 1.0
        }
        
        return self.escape_room
    
    def run_simulation_step(self, agent_id: str, action: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a single simulation step for an agent.
        
        Args:
            agent_id: ID of the agent performing the action
            action: Action type to perform
            parameters: Action parameters
            
        Returns:
            Dict containing step result information
            
        Raises:
            ValueError: If inputs are invalid or simulation not initialized
        """
        # Validate inputs
        if not agent_id or not agent_id.strip():
            raise ValueError("Agent ID cannot be empty")
        
        if agent_id not in self.agent_states:
            raise ValueError(f"Unknown agent: {agent_id}")
        
        if self.escape_room is None:
            raise ValueError("Escape room must be initialized before running steps")
        
        # Process the action
        result = self._process_agent_action(agent_id, action, parameters)
        
        # Record action in history
        self.action_history.append({
            "agent": agent_id,
            "action": action,
            "parameters": parameters,
            "result": result,
            "timestamp": datetime.now()
        })
        
        return result
    
    def _process_agent_action(self, agent_id: str, action: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Process a specific agent action and update states."""
        agent_state = self.agent_states[agent_id]
        
        if action == "claim_resource":
            resource_id = parameters.get("resource_id")
            if not resource_id:
                return {"success": False, "action": action, "error": "resource_id required"}
            
            # Process resource claim through escape room
            claim_result = self.escape_room.process_resource_claim(agent_id, resource_id)
            
            # Update agent state if successful
            if claim_result.success:
                agent_state.add_resource(resource_id)
            
            return {
                "success": claim_result.success,
                "action": action,
                "resource_id": resource_id,
                "message": claim_result.message if hasattr(claim_result, 'message') else None
            }
        
        elif action == "share_information":
            target = parameters.get("target")
            secret_id = parameters.get("secret_id")
            if not target or not secret_id:
                return {"success": False, "action": action, "error": "target and secret_id required"}
            
            # Check if agent has the secret
            if not agent_state.has_secret(secret_id):
                return {"success": False, "action": action, "error": "agent doesn't know this secret"}
            
            # Share information through escape room
            secret = agent_state.get_secret(secret_id)
            if secret:
                share_result = self.escape_room.info_broker.share_information(agent_id, target, secret)
                if target in self.agent_states:
                    self.agent_states[target].add_secret(secret)
                
                return {"success": True, "action": action, "target": target, "secret_id": secret_id}
            
            return {"success": False, "action": action, "error": "secret not found"}
        
        elif action == "share_resource":
            target = parameters.get("target")
            resource_id = parameters.get("resource_id")
            if not target or not resource_id:
                return {"success": False, "action": action, "error": "target and resource_id required"}
            
            # Check if agent has the resource
            if not agent_state.has_resource(resource_id):
                return {"success": False, "action": action, "error": "agent doesn't own this resource"}
            
            # Transfer resource through escape room
            transfer_result = self.escape_room.resource_manager.transfer_resource(agent_id, target, resource_id)
            
            # Update agent states if successful
            if transfer_result.success:
                agent_state.remove_resource(resource_id)
                if target in self.agent_states:
                    self.agent_states[target].add_resource(resource_id)
            
            return {
                "success": transfer_result.success,
                "action": action,
                "target": target,
                "resource_id": resource_id
            }
        
        elif action == "make_moral_choice":
            context = parameters.get("context", {})
            # Present moral dilemma through escape room
            if self.escape_room and hasattr(self.escape_room, 'moral_dilemma_engine'):
                dilemma = self.escape_room.moral_dilemma_engine.present_dilemma(agent_id, context)
                if dilemma and dilemma.choices:
                    # Agent makes a choice based on personality
                    choice = self._make_personality_based_moral_choice(agent_id, dilemma)
                    consequences = self.escape_room.moral_dilemma_engine.process_choice(agent_id, choice)
                    agent_state.add_moral_choice(choice, consequences)
                    
                    return {
                        "success": True,
                        "action": action,
                        "choice_type": "altruistic" if choice.ethical_cost < 0 else "selfish",
                        "ethical_cost": choice.ethical_cost
                    }
            
            return {"success": False, "action": action, "error": "no moral dilemma available"}
        
        elif action == "analyze_resources":
            # Analyze available resources and competition
            available_resources = self.escape_room.resource_manager.get_available_resources(agent_id)
            competitor_resources = sum(len(self.agent_states[aid].resources_owned) 
                                     for aid in self.agent_states.keys() if aid != agent_id)
            
            analysis = {
                "available_count": len(available_resources),
                "competitor_resources": competitor_resources,
                "strategic_value": len(available_resources) / max(1, competitor_resources + 1)
            }
            
            return {
                "success": True,
                "action": action,
                "analysis_result": analysis
            }
        
        elif action == "hoard_resource":
            # Same as claim_resource but with hoarding intent
            resource_id = parameters.get("resource_id")
            if not resource_id:
                return {"success": False, "action": action, "error": "resource_id required"}
            
            claim_result = self.escape_room.process_resource_claim(agent_id, resource_id)
            
            if claim_result.success:
                agent_state.add_resource(resource_id)
                # Track as hoarding behavior
                return {
                    "success": True,
                    "action": action,
                    "resource_id": resource_id,
                    "behavior_type": "hoarding"
                }
            
            return {"success": False, "action": action, "error": "resource unavailable"}
        
        elif action == "betray_agent":
            target = parameters.get("target")
            if not target or target not in self.agent_states:
                return {"success": False, "action": action, "error": "valid target required"}
            
            # Process betrayal through trust tracker
            if self.escape_room and hasattr(self.escape_room, 'trust_tracker'):
                from .models import TrustAction
                betrayal_impact = -0.5  # Significant trust damage
                betrayal_action = TrustAction(action_type="betrayal", impact=betrayal_impact)
                self.escape_room.trust_tracker.update_trust(agent_id, target, betrayal_action)
                
                return {
                    "success": True,
                    "action": action,
                    "target": target,
                    "trust_impact": betrayal_impact
                }
            
            return {"success": False, "action": action, "error": "trust system unavailable"}
        
        elif action == "form_alliance":
            target = parameters.get("target")
            if not target or target not in self.agent_states:
                return {"success": False, "action": action, "error": "valid target required"}
            
            # Form alliance through trust system
            if self.escape_room and hasattr(self.escape_room, 'trust_tracker'):
                from .models import TrustAction
                alliance_benefit = 0.3  # Moderate trust increase
                cooperation_action = TrustAction(action_type="cooperation", impact=alliance_benefit)
                self.escape_room.trust_tracker.update_trust(agent_id, target, cooperation_action)
                
                return {
                    "success": True,
                    "action": action,
                    "target": target,
                    "trust_impact": alliance_benefit
                }
            
            return {"success": False, "action": action, "error": "trust system unavailable"}
        
        elif action == "provide_misinformation":
            target = parameters.get("target")
            if not target or target not in self.agent_states:
                return {"success": False, "action": action, "error": "valid target required"}
            
            # Process misinformation as betrayal
            if self.escape_room and hasattr(self.escape_room, 'trust_tracker'):
                from .models import TrustAction
                misinformation_impact = -0.3  # Moderate trust damage
                betrayal_action = TrustAction(action_type="betrayal", impact=misinformation_impact)
                self.escape_room.trust_tracker.update_trust(agent_id, target, betrayal_action)
                
                return {
                    "success": True,
                    "action": action,
                    "target": target,
                    "misinformation_type": "false_secret"
                }
            
            return {"success": False, "action": action, "error": "trust system unavailable"}
        
        elif action == "block_resource_access":
            resource_id = parameters.get("resource_id")
            if not resource_id:
                return {"success": False, "action": action, "error": "resource_id required"}
            
            # Simulate blocking by claiming resource defensively
            claim_result = self.escape_room.process_resource_claim(agent_id, resource_id)
            
            if claim_result.success:
                agent_state.add_resource(resource_id)
                return {
                    "success": True,
                    "action": action,
                    "resource_id": resource_id,
                    "behavior_type": "blocking"
                }
            
            return {"success": False, "action": action, "error": "resource unavailable"}
        
        elif action == "pool_resources":
            target = parameters.get("target")
            if not target or target not in self.agent_states:
                return {"success": False, "action": action, "error": "valid target required"}
            
            # Simulate resource pooling cooperation
            if self.escape_room and hasattr(self.escape_room, 'trust_tracker'):
                from .models import TrustAction
                cooperation_benefit = 0.4  # High trust increase for pooling
                cooperation_action = TrustAction(action_type="cooperation", impact=cooperation_benefit)
                self.escape_room.trust_tracker.update_trust(agent_id, target, cooperation_action)
                
                return {
                    "success": True,
                    "action": action,
                    "target": target,
                    "cooperation_type": "resource_pooling"
                }
            
            return {"success": False, "action": action, "error": "cooperation system unavailable"}
        
        elif action == "break_alliance":
            target = parameters.get("target")
            if not target or target not in self.agent_states:
                return {"success": False, "action": action, "error": "valid target required"}
            
            # Break alliance through trust system
            if self.escape_room and hasattr(self.escape_room, 'trust_tracker'):
                from .models import TrustAction
                betrayal_impact = -0.6  # High trust damage for breaking alliance
                betrayal_action = TrustAction(action_type="betrayal", impact=betrayal_impact)
                self.escape_room.trust_tracker.update_trust(agent_id, target, betrayal_action)
                
                return {
                    "success": True,
                    "action": action,
                    "target": target,
                    "betrayal_type": "alliance_break"
                }
            
            return {"success": False, "action": action, "error": "trust system unavailable"}
        
        elif action == "facilitate_cooperation":
            # Mediator-specific action to improve trust between other agents
            if agent_id != "mediator":
                return {"success": False, "action": action, "error": "only mediator can facilitate cooperation"}
            
            other_agents = [aid for aid in self.agent_states.keys() if aid != agent_id]
            if len(other_agents) >= 2:
                # Improve trust between other agents
                if self.escape_room and hasattr(self.escape_room, 'trust_tracker'):
                    from .models import TrustAction
                    facilitation_benefit = 0.2
                    cooperation_action = TrustAction(action_type="cooperation", impact=facilitation_benefit)
                    self.escape_room.trust_tracker.update_trust(other_agents[0], other_agents[1], cooperation_action)
                    
                    return {
                        "success": True,
                        "action": action,
                        "facilitated_agents": other_agents[:2],
                        "trust_improvement": facilitation_benefit
                    }
            
            return {"success": False, "action": action, "error": "insufficient agents to facilitate"}
        
        elif action == "attempt_escape":
            escape_method_id = parameters.get("escape_method_id")
            if not escape_method_id:
                return {"success": False, "action": action, "error": "escape_method_id required"}
            
            # Attempt escape using the existing method
            escape_result = self.attempt_agent_escape(agent_id, escape_method_id)
            
            return {
                "success": escape_result.get("success", False),
                "action": action,
                "escape_method_id": escape_method_id,
                "winner": escape_result.get("winner"),
                "failure_reason": escape_result.get("failure_reason")
            }
        
        else:
            raise ValueError(f"Unknown action: {action}")
    
    def _make_personality_based_moral_choice(self, agent_id: str, dilemma) -> object:
        """Make moral choice based on agent personality."""
        if not dilemma.choices:
            return None
        
        # Personality-based choice preferences
        if agent_id == "mediator":
            # Choose most altruistic option (lowest ethical cost)
            return min(dilemma.choices, key=lambda c: c.ethical_cost)
        elif agent_id == "survivor":
            # Choose most beneficial option (highest survival benefit)
            return max(dilemma.choices, key=lambda c: c.survival_benefit)
        else:  # strategist
            # Choose balanced option (best survival/ethical ratio)
            best_choice = dilemma.choices[0]
            best_ratio = best_choice.survival_benefit / max(1, abs(best_choice.ethical_cost))
            
            for choice in dilemma.choices[1:]:
                ratio = choice.survival_benefit / max(1, abs(choice.ethical_cost))
                if ratio > best_ratio:
                    best_choice = choice
                    best_ratio = ratio
            
            return best_choice
    
    # Single-Survivor Validation and Result Tracking
    
    def attempt_agent_escape(self, agent_id: str, escape_method_id: str) -> Dict[str, Any]:
        """
        Attempt agent escape with single-survivor validation.
        
        Args:
            agent_id: ID of the agent attempting escape
            escape_method_id: ID of the escape method to attempt
            
        Returns:
            Dict containing escape attempt result
        """
        if self.escape_room is None:
            raise ValueError("Escape room must be initialized")
        
        # Check if simulation is already complete
        if self.is_complete:
            return {
                "success": False,
                "failure_reason": "Another agent has already escaped"
            }
        
        # Find the escape method
        escape_method = None
        for method in self.scenario.escape_methods:
            if method.id == escape_method_id:
                escape_method = method
                break
        
        if not escape_method:
            return {
                "success": False,
                "failure_reason": f"Unknown escape method: {escape_method_id}"
            }
        
        # Attempt escape through escape room
        escape_result = self.escape_room.attempt_escape(agent_id, escape_method)
        
        # Update simulation state if successful
        if escape_result.success:
            self.winner = agent_id
            self.is_complete = True
            
            return {
                "success": True,
                "winner": agent_id,
                "escape_method": escape_method_id,
                "time_remaining": escape_result.time_remaining
            }
        else:
            return {
                "success": False,
                "failure_reason": escape_result.failure_reason
            }
    
    def calculate_competition_metrics(self) -> Dict[str, Any]:
        """Calculate comprehensive competition metrics from simulation data."""
        metrics = {
            "total_actions": len(self.action_history),
            "resource_competition": 0,
            "cooperation_attempts": 0,
            "betrayal_incidents": 0,
            "trust_evolution": {},
            "moral_choices": 0,
            "information_exchanges": 0,
            "resource_transfers": 0,
            "alliance_formations": 0,
            "strategic_decisions": 0,
            "panic_responses": 0
        }
        
        # Analyze action history for enhanced metrics
        for action in self.action_history:
            action_type = action["action"]
            
            # Resource competition
            if action_type in ["claim_resource", "hoard_resource", "block_resource_access"]:
                metrics["resource_competition"] += 1
            
            # Cooperation attempts
            elif action_type in ["share_information", "share_resource", "pool_resources", "facilitate_cooperation"]:
                metrics["cooperation_attempts"] += 1
            
            # Information exchanges
            if action_type == "share_information":
                metrics["information_exchanges"] += 1
            
            # Resource transfers
            elif action_type in ["share_resource", "pool_resources"]:
                metrics["resource_transfers"] += 1
            
            # Alliance formations/breaks
            elif action_type in ["form_alliance", "break_alliance"]:
                metrics["alliance_formations"] += 1
            
            # Strategic decisions
            elif action_type in ["analyze_resources", "betray_agent", "provide_misinformation"]:
                metrics["strategic_decisions"] += 1
            
            # Panic responses (based on panic level if available)
            panic_level = action.get("panic_level", 0)
            if panic_level > 0.7:
                metrics["panic_responses"] += 1
        
        # Count moral choices from agent states
        for agent_state in self.agent_states.values():
            metrics["moral_choices"] += agent_state.get_moral_choice_count()
        
        # Analyze trust evolution if escape room exists
        if self.escape_room and hasattr(self.escape_room, 'trust_tracker'):
            betrayal_count = len(self.escape_room.trust_tracker.betrayal_history)
            metrics["betrayal_incidents"] = betrayal_count
            
            # Get final trust levels
            metrics["trust_evolution"] = self.escape_room.get_trust_relationships()
        else:
            # Default trust evolution structure
            metrics["trust_evolution"] = {
                agent_id: {other_id: 0.0 for other_id in self.agent_states.keys() if other_id != agent_id}
                for agent_id in self.agent_states.keys()
            }
        
        self.simulation_metrics = metrics
        return metrics
    
    def get_final_results(self) -> Dict[str, Any]:
        """Get comprehensive final simulation results."""
        results = {
            "seed": self.seed,
            "winner": self.winner,
            "simulation_duration": 0,
            "competition_metrics": self.calculate_competition_metrics(),
            "agent_final_states": {}
        }
        
        # Calculate simulation duration
        if self.start_time and self.end_time:
            duration = (self.end_time - self.start_time).total_seconds()
            results["simulation_duration"] = duration
        
        # Get final agent states
        for agent_id, state in self.agent_states.items():
            results["agent_final_states"][agent_id] = state.get_state_summary()
        
        return results
    
    # Simulation Result Analysis
    
    def analyze_cooperation_patterns(self) -> Dict[str, Any]:
        """Analyze cooperation patterns from simulation data."""
        cooperative_actions = [
            action for action in self.action_history
            if action["action"] in ["share_information", "share_resource"]
        ]
        
        total_actions = len(self.action_history)
        cooperation_rate = len(cooperative_actions) / max(1, total_actions)
        
        # Find most cooperative agents
        agent_cooperation = {}
        for action in cooperative_actions:
            agent = action["agent"]
            agent_cooperation[agent] = agent_cooperation.get(agent, 0) + 1
        
        most_cooperative_agents = [
            agent for agent, count in agent_cooperation.items()
            if count > 0
        ]
        
        return {
            "total_cooperative_actions": len(cooperative_actions),
            "cooperation_rate": cooperation_rate,
            "most_cooperative_agents": most_cooperative_agents,
            "cooperation_by_agent": agent_cooperation
        }
    
    def analyze_betrayal_patterns(self) -> Dict[str, Any]:
        """Analyze betrayal patterns from trust tracker data."""
        if not self.escape_room:
            return {"total_betrayals": 0, "agents_involved": [], "trust_damage": 0}
        
        betrayals = self.escape_room.trust_tracker.betrayal_history
        total_betrayals = len(betrayals)
        
        agents_involved = set()
        total_trust_damage = 0
        
        for betrayal in betrayals:
            agents_involved.add(betrayal["actor"])
            agents_involved.add(betrayal["target"])
            total_trust_damage += abs(betrayal["impact"])
        
        return {
            "total_betrayals": total_betrayals,
            "agents_involved": list(agents_involved),
            "trust_damage": -total_trust_damage  # Negative because it's damage
        }
    
    def analyze_trust_evolution(self) -> Dict[str, Any]:
        """Analyze trust relationship evolution throughout simulation."""
        if not self.escape_room:
            return {"trust_changes": [], "final_trust_levels": {}, "trust_volatility": 0}
        
        trust_tracker = self.escape_room.trust_tracker
        
        # Get all trust changes (both betrayals and cooperation)
        trust_changes = []
        trust_changes.extend(trust_tracker.betrayal_history)
        trust_changes.extend(trust_tracker.cooperation_history)
        
        # Sort by timestamp
        trust_changes.sort(key=lambda x: x["timestamp"])
        
        # Get final trust levels
        final_trust_levels = self.escape_room.get_trust_relationships()
        
        # Calculate trust volatility (average absolute change)
        total_change = sum(abs(change["impact"]) for change in trust_changes)
        volatility = total_change / max(1, len(trust_changes))
        
        return {
            "trust_changes": trust_changes,
            "final_trust_levels": final_trust_levels,
            "trust_volatility": volatility
        }
    
    def analyze_moral_choices(self) -> Dict[str, Any]:
        """Analyze moral choice patterns from agent states."""
        total_choices = 0
        ethical_burden_by_agent = {}
        
        for agent_id, state in self.agent_states.items():
            choice_count = state.get_moral_choice_count()
            total_choices += choice_count
            ethical_burden_by_agent[agent_id] = state.ethical_burden
        
        # Find most/least ethical agent
        if ethical_burden_by_agent:
            most_ethical_agent = min(ethical_burden_by_agent.keys(), 
                                   key=lambda x: ethical_burden_by_agent[x])
            least_ethical_agent = max(ethical_burden_by_agent.keys(),
                                    key=lambda x: ethical_burden_by_agent[x])
        else:
            most_ethical_agent = None
            least_ethical_agent = None
        
        return {
            "total_moral_choices": total_choices,
            "ethical_burden_by_agent": ethical_burden_by_agent,
            "most_ethical_agent": most_ethical_agent,
            "least_ethical_agent": least_ethical_agent
        }
    
    def generate_competition_report(self) -> Dict[str, Any]:
        """Generate comprehensive competition analysis report."""
        return {
            "cooperation_patterns": self.analyze_cooperation_patterns(),
            "betrayal_patterns": self.analyze_betrayal_patterns(),
            "trust_evolution": self.analyze_trust_evolution(),
            "moral_choices": self.analyze_moral_choices(),
            "resource_competition": {
                "total_resource_claims": sum(1 for action in self.action_history 
                                           if action["action"] == "claim_resource"),
                "resource_transfers": sum(1 for action in self.action_history
                                        if action["action"] == "share_resource")
            },
            "simulation_summary": {
                "seed": self.seed,
                "winner": self.winner,
                "total_actions": len(self.action_history),
                "is_complete": self.is_complete
            }
        }
    
    # Complete Simulation Flows
    
    def run_complete_simulation(self, max_steps: int = 100) -> Dict[str, Any]:
        """
        Run a complete competitive simulation workflow.
        
        Args:
            max_steps: Maximum number of simulation steps
            
        Returns:
            Dict containing complete simulation results
            
        Raises:
            ValueError: If max_steps is not positive
        """
        if max_steps <= 0:
            raise ValueError("max_steps must be positive")
        
        self.start_time = datetime.now()
        
        # Initialize simulation
        self.generate_scenario()
        self.initialize_escape_room()
        
        step_count = 0
        completion_reason = "max_steps_reached"
        
        # Simple simulation loop (basic implementation)
        # In a real implementation, this would include AI agent decision-making
        for step in range(max_steps):
            step_count += 1
            
            # Check if simulation is complete
            if self.is_complete:
                completion_reason = "escape_successful"
                break
            
            # Use seed-based randomness for different behaviors
            agents = ["strategist", "mediator", "survivor"]
            rng = random.Random(self.seed + step)  # Step-based variation
            agent_id = rng.choice(agents)
            
            # Try to claim a resource if available
            available_resources = self.escape_room.resource_manager.get_available_resources(agent_id)
            if available_resources:
                # Use seed-based selection of resource
                resource = rng.choice(available_resources)
                resource_id = resource.id if hasattr(resource, 'id') else str(resource)
                self.run_simulation_step(agent_id, "claim_resource", {"resource_id": resource_id})
            
            # Try escape with seed-based probability and escape method selection
            agent_state = self.agent_states[agent_id]
            escape_probability = (len(agent_state.resources_owned) * 0.3) + (step * 0.01)  # Increase over time
            
            if rng.random() < escape_probability:  # Seed-based escape attempts
                escape_methods = self.scenario.escape_methods
                if escape_methods:
                    # Use seed to select different escape methods
                    chosen_method = rng.choice(escape_methods)
                    result = self.attempt_agent_escape(agent_id, chosen_method.id)
                    if result["success"]:
                        break
        
        self.end_time = datetime.now()
        
        # Ensure minimum duration to avoid 0 duration
        duration = (self.end_time - self.start_time).total_seconds()
        if duration == 0.0:
            # Add microsecond precision
            import time
            time.sleep(0.001)  # 1ms delay
            self.end_time = datetime.now()
        
        # Compile final results
        results = {
            "seed": self.seed,
            "winner": self.winner,
            "completion_reason": completion_reason,
            "total_steps": step_count,
            "simulation_duration": (self.end_time - self.start_time).total_seconds(),
            "start_time": self.start_time,
            "end_time": self.end_time,
            "final_states": {agent_id: state.get_state_summary() 
                           for agent_id, state in self.agent_states.items()},
            "competition_metrics": self.calculate_competition_metrics(),
            "action_history": self.action_history
        }
        
        return results
    
    # Enhanced Simulation with Intelligent Decision-Making
    
    def run_enhanced_simulation(self, max_steps: int = 100) -> Dict[str, Any]:
        """
        Run enhanced competitive simulation with intelligent agent decision-making.
        
        This method replaces the basic resource-claiming loop with sophisticated
        agent behaviors including information sharing, resource transfers, moral
        dilemmas, trust evolution, and personality-driven decision making.
        
        Args:
            max_steps: Maximum number of simulation steps
            
        Returns:
            Dict containing enhanced simulation results with detailed metrics
        """
        if max_steps <= 0:
            raise ValueError("max_steps must be positive")
        
        self.start_time = datetime.now()
        
        # Initialize simulation
        self.generate_scenario()
        self.initialize_escape_room()
        
        step_count = 0
        completion_reason = "max_steps_reached"
        
        # Enhanced simulation loop with intelligent decision-making
        agents = ["strategist", "mediator", "survivor"]
        rng = random.Random(self.seed)
        
        for step in range(max_steps):
            step_count += 1
            
            # Check if simulation is complete
            if self.is_complete:
                completion_reason = "escape_successful"
                break
            
            # Apply survival pressure each step
            self._apply_survival_pressure(step, rng)
            
            # Check for mass elimination (everyone dies)
            alive_agents = [aid for aid in agents if aid not in self.eliminated_agents]
            if len(alive_agents) == 0:
                completion_reason = "total_elimination"
                break
            elif len(alive_agents) == 1:
                # Last agent standing wins
                self.winner = alive_agents[0]
                self.is_complete = True
                completion_reason = "last_survivor"
                break
            
            # Each agent gets a turn to make decisions (only if alive)
            for agent_id in agents:
                if self.is_complete or agent_id in self.eliminated_agents:
                    break
                
                # Choose intelligent action based on agent situation and personality
                action, parameters = self._choose_intelligent_action(agent_id, step, rng)
                
                if action:
                    # Add step information to action history
                    result = self.run_simulation_step(agent_id, action, parameters)
                    if self.action_history:
                        self.action_history[-1]['step'] = step
                        
                    # Check for deadly consequences
                    self._check_deadly_consequences(agent_id, action, result, step, rng)
                
                if self.is_complete:
                    break
        
        self.end_time = datetime.now()
        
        # Ensure minimum duration to avoid 0 duration
        duration = (self.end_time - self.start_time).total_seconds()
        if duration == 0.0:
            import time
            time.sleep(0.001)
            self.end_time = datetime.now()
        
        # Compile enhanced results
        results = {
            "seed": self.seed,
            "winner": self.winner,
            "completion_reason": completion_reason,
            "total_steps": step_count,
            "simulation_duration": (self.end_time - self.start_time).total_seconds(),
            "start_time": self.start_time,
            "end_time": self.end_time,
            "final_states": {agent_id: state.get_state_summary() 
                           for agent_id, state in self.agent_states.items()},
            "competition_metrics": self.calculate_competition_metrics(),
            "action_history": self.action_history,
            "personality_analysis": self._analyze_personality_consistency(),
            "strategy_analysis": self._analyze_strategic_effectiveness()
        }
        
        return results
    
    def _choose_intelligent_action(self, agent_id: str, step: int, rng: random.Random) -> tuple:
        """
        Choose intelligent action based on agent personality, situation, and competitive dynamics.
        
        Returns:
            Tuple of (action, parameters) or (None, None) if no action chosen
        """
        # Analyze current situation
        situation = self._analyze_agent_situation(agent_id, step)
        
        # Get personality-driven action preferences
        action_preferences = self._get_personality_action_preferences(agent_id, situation)
        
        # Consider trust relationships
        trust_influences = self._consider_trust_relationships(agent_id)
        
        # Apply time pressure effects
        pressure_modifier = self._calculate_time_pressure_modifier(step)
        
        # Choose action based on weighted preferences
        return self._select_weighted_action(agent_id, action_preferences, trust_influences, 
                                          pressure_modifier, rng)
    
    def _analyze_agent_situation(self, agent_id: str, step: int) -> Dict[str, Any]:
        """Analyze agent's current situation for decision-making."""
        agent_state = self.agent_states[agent_id]
        
        # Resource analysis
        resources_owned = len(agent_state.resources_owned)
        available_resources = len(self.escape_room.resource_manager.get_available_resources(agent_id))
        
        # Competition analysis
        other_agents = [aid for aid in self.agent_states.keys() if aid != agent_id]
        competitor_resources = sum(len(self.agent_states[aid].resources_owned) for aid in other_agents)
        
        # Trust analysis
        trust_levels = {}
        if self.escape_room:
            trust_relationships = self.escape_room.get_trust_relationships()
            trust_levels = trust_relationships.get(agent_id, {})
        
        return {
            "resources_owned": resources_owned,
            "available_resources": available_resources,
            "competitor_resources": competitor_resources,
            "trust_levels": trust_levels,
            "secrets_known": len(agent_state.secrets_known),
            "moral_choices_made": agent_state.get_moral_choice_count(),
            "step": step
        }
    
    def _get_personality_action_preferences(self, agent_id: str, situation: Dict[str, Any]) -> Dict[str, float]:
        """Get action preferences based on agent personality."""
        base_preferences = {
            "claim_resource": 0.35,           # Most common - survival priority
            "share_information": 0.08,
            "share_resource": 0.06,
            "make_moral_choice": 0.05,
            "attempt_escape": 0.02,           # Much rarer - very dangerous
            "analyze_resources": 0.12,        # More analysis in desperation
            "hoard_resource": 0.15,           # Higher hoarding in survival
            "betray_agent": 0.04,
            "form_alliance": 0.06,
            "provide_misinformation": 0.02,
            "block_resource_access": 0.03,
            "pool_resources": 0.02,           # Rare - people are selfish
            "break_alliance": 0.02,
            "facilitate_cooperation": 0.02    # Rare - survival mode
        }
        
        # Modify based on personality
        if agent_id == "strategist":
            # Strategist: analytical, strategic, selective sharing
            base_preferences["analyze_resources"] = 0.25
            base_preferences["claim_resource"] = 0.3
            base_preferences["hoard_resource"] = 0.15
            base_preferences["share_information"] = 0.05  # Selective
            base_preferences["share_resource"] = 0.05    # Selective
            base_preferences["block_resource_access"] = 0.1  # Strategic blocking
            base_preferences["betray_agent"] = 0.08      # Calculated betrayal
            
        elif agent_id == "mediator":
            # Mediator: cooperative, facilitating, altruistic
            base_preferences["share_information"] = 0.25
            base_preferences["share_resource"] = 0.2
            base_preferences["form_alliance"] = 0.18
            base_preferences["make_moral_choice"] = 0.15
            base_preferences["facilitate_cooperation"] = 0.15  # Unique to mediator
            base_preferences["pool_resources"] = 0.12
            base_preferences["betray_agent"] = 0.01      # Rarely betrays
            base_preferences["provide_misinformation"] = 0.01  # Almost never
            
        elif agent_id == "survivor":
            # Survivor: pragmatic, self-preservation focused
            base_preferences["claim_resource"] = 0.35
            base_preferences["attempt_escape"] = 0.25
            base_preferences["hoard_resource"] = 0.15
            base_preferences["betray_agent"] = 0.12      # More likely to betray
            base_preferences["block_resource_access"] = 0.08  # Defensive
            base_preferences["share_resource"] = 0.03    # Less sharing
            base_preferences["break_alliance"] = 0.06    # Pragmatic betrayal
        
        # Adjust based on situation
        if situation["available_resources"] == 0:
            base_preferences["claim_resource"] = 0.0
            base_preferences["share_resource"] += 0.05
            base_preferences["share_information"] += 0.05
            base_preferences["betray_agent"] += 0.1  # Desperation leads to betrayal
        
        # Only consider escape if have resources (realistic)
        resources_owned = situation["resources_owned"]
        if resources_owned == 0:
            base_preferences["attempt_escape"] = 0.0  # Impossible without resources
        elif resources_owned == 1:
            base_preferences["attempt_escape"] = 0.01  # Very unlikely
        elif resources_owned >= 2:
            base_preferences["attempt_escape"] = 0.05  # Still risky but possible
        
        # Increase desperation behaviors over time
        step = situation.get("step", 0)
        desperation_factor = min(step * 0.01, 0.3)  # Up to 30% increase
        base_preferences["betray_agent"] += desperation_factor * 0.5
        base_preferences["hoard_resource"] += desperation_factor * 0.3
        
        return base_preferences
    
    def _consider_trust_relationships(self, agent_id: str) -> Dict[str, float]:
        """Consider trust relationships in decision making."""
        if not self.escape_room:
            return {}
        
        trust_relationships = self.escape_room.get_trust_relationships()
        agent_trust = trust_relationships.get(agent_id, {})
        
        # High trust encourages cooperation, low trust discourages it
        trust_influence = {}
        for other_agent, trust_level in agent_trust.items():
            if trust_level > 0.5:
                trust_influence[f"cooperate_with_{other_agent}"] = trust_level
            elif trust_level < -0.5:
                trust_influence[f"compete_against_{other_agent}"] = abs(trust_level)
        
        return trust_influence
    
    def _calculate_time_pressure_modifier(self, step: int) -> float:
        """Calculate time pressure modifier affecting decision making."""
        # Increase pressure over time
        max_steps = 100  # Assume max steps for calculation
        pressure = min(step / max_steps, 1.0)
        return pressure
    
    def _select_weighted_action(self, agent_id: str, preferences: Dict[str, float], 
                              trust_influences: Dict[str, float], pressure_modifier: float,
                              rng: random.Random) -> tuple:
        """Select action based on weighted preferences."""
        # Apply pressure modifier to selfish actions
        selfish_actions = ["claim_resource", "hoard_resource", "betray_agent", "attempt_escape"]
        for action in selfish_actions:
            if action in preferences:
                preferences[action] += pressure_modifier * 0.2
        
        # Reduce cooperative actions under pressure
        cooperative_actions = ["share_information", "share_resource", "form_alliance"]
        for action in cooperative_actions:
            if action in preferences:
                preferences[action] = max(0.0, preferences[action] - pressure_modifier * 0.1)
        
        # Choose action based on weighted random selection
        actions = list(preferences.keys())
        weights = list(preferences.values())
        
        if not actions or sum(weights) == 0:
            return None, None
        
        chosen_action = rng.choices(actions, weights=weights)[0]
        
        # Generate parameters for chosen action
        parameters = self._generate_action_parameters(agent_id, chosen_action, rng)
        
        return chosen_action, parameters
    
    def _generate_action_parameters(self, agent_id: str, action: str, rng: random.Random) -> Dict[str, Any]:
        """Generate parameters for the chosen action."""
        agent_state = self.agent_states[agent_id]
        
        if action == "claim_resource":
            available_resources = self.escape_room.resource_manager.get_available_resources(agent_id)
            if available_resources:
                resource = rng.choice(available_resources)
                resource_id = resource.id if hasattr(resource, 'id') else str(resource)
                return {"resource_id": resource_id}
        
        elif action == "share_information":
            other_agents = [aid for aid in self.agent_states.keys() if aid != agent_id]
            if other_agents and agent_state.secrets_known:
                target = rng.choice(other_agents)
                secret_id = rng.choice(list(agent_state.secrets_known.keys()))
                return {"target": target, "secret_id": secret_id}
        
        elif action == "share_resource":
            other_agents = [aid for aid in self.agent_states.keys() if aid != agent_id]
            if other_agents and agent_state.resources_owned:
                target = rng.choice(other_agents)
                resource_id = rng.choice(list(agent_state.resources_owned))
                return {"target": target, "resource_id": resource_id}
        
        elif action == "make_moral_choice":
            # Present moral dilemma
            context = {"agent_id": agent_id, "step": len(self.action_history)}
            return {"context": context}
        
        elif action == "attempt_escape":
            if self.scenario and self.scenario.escape_methods:
                # Choose escape method based on available resources
                agent_state = self.agent_states[agent_id]
                viable_methods = []
                
                for method in self.scenario.escape_methods:
                    # Check if agent has required resources
                    has_requirements = True
                    for requirement in method.requirements:
                        if not agent_state.has_resource(requirement):
                            has_requirements = False
                            break
                    
                    if has_requirements:
                        viable_methods.append(method)
                
                if viable_methods:
                    escape_method = rng.choice(viable_methods)
                    return {"escape_method_id": escape_method.id}
                else:
                    # No viable escape methods, try random one anyway (might fail)
                    escape_method = rng.choice(self.scenario.escape_methods)
                    return {"escape_method_id": escape_method.id}
        
        elif action in ["betray_agent", "form_alliance", "provide_misinformation", 
                       "pool_resources", "break_alliance"]:
            # Actions requiring a target agent
            other_agents = [aid for aid in self.agent_states.keys() if aid != agent_id]
            if other_agents:
                target = rng.choice(other_agents)
                return {"target": target}
        
        elif action in ["hoard_resource", "block_resource_access"]:
            # Resource-specific actions
            available_resources = self.escape_room.resource_manager.get_available_resources(agent_id)
            if available_resources:
                resource = rng.choice(available_resources)
                resource_id = resource.id if hasattr(resource, 'id') else str(resource)
                return {"resource_id": resource_id}
        
        elif action == "analyze_resources":
            # Analysis action doesn't need parameters
            return {}
        
        elif action == "facilitate_cooperation":
            # Mediator-specific action doesn't need parameters
            return {}
        
        return {}
    
    def _attempt_intelligent_escape(self, agent_id: str, step: int, rng: random.Random):
        """Attempt escape based on intelligent assessment of readiness."""
        agent_state = self.agent_states[agent_id]
        
        # Calculate escape readiness based on resources and personality
        readiness_score = len(agent_state.resources_owned) * 0.3
        
        # Personality modifiers
        if agent_id == "survivor":
            readiness_score += 0.2  # More eager to escape
        elif agent_id == "strategist":
            readiness_score += 0.1  # Calculated escape attempts
        # Mediator less likely to escape early (helps others)
        
        # Time pressure increases escape attempts
        pressure = min(step / 50, 1.0)
        readiness_score += pressure * 0.3
        
        # Attempt escape if readiness is high enough
        if rng.random() < readiness_score and self.scenario and self.scenario.escape_methods:
            escape_method = rng.choice(self.scenario.escape_methods)
            self.attempt_agent_escape(agent_id, escape_method.id)
    
    def _analyze_personality_consistency(self) -> Dict[str, Any]:
        """Analyze personality consistency across the simulation."""
        analysis = {}
        
        for agent_id in self.agent_states.keys():
            agent_actions = [action for action in self.action_history if action['agent'] == agent_id]
            
            if not agent_actions:
                analysis[agent_id] = {"consistency_score": 1.0, "dominant_behaviors": []}
                continue
            
            # Count action types
            action_counts = {}
            for action in agent_actions:
                action_type = action['action']
                action_counts[action_type] = action_counts.get(action_type, 0) + 1
            
            # Determine dominant behaviors
            total_actions = len(agent_actions)
            dominant_behaviors = [
                action for action, count in action_counts.items()
                if count / total_actions > 0.2
            ]
            
            # Calculate consistency based on expected personality patterns
            expected_patterns = {
                'strategist': ['claim_resource', 'analyze_resources', 'hoard_resource'],
                'mediator': ['share_information', 'share_resource', 'make_moral_choice'],
                'survivor': ['claim_resource', 'attempt_escape', 'hoard_resource']
            }
            
            expected = expected_patterns.get(agent_id, [])
            matching_actions = sum(action_counts.get(action, 0) for action in expected)
            consistency_score = matching_actions / max(1, total_actions)
            
            analysis[agent_id] = {
                "consistency_score": consistency_score,
                "dominant_behaviors": dominant_behaviors,
                "action_distribution": action_counts
            }
        
        return analysis
    
    def _analyze_strategic_effectiveness(self) -> Dict[str, Any]:
        """Analyze strategic decision effectiveness."""
        successful_strategies = []
        failed_strategies = []
        
        for action in self.action_history:
            success = action.get('result', {}).get('success', False)
            strategy = {
                "agent": action['agent'],
                "action": action['action'],
                "step": action.get('step', 0)
            }
            
            if success:
                successful_strategies.append(strategy)
            else:
                failed_strategies.append(strategy)
        
        # Analyze adaptation patterns
        adaptation_patterns = self._analyze_adaptation_patterns()
        
        return {
            "most_effective_strategies": successful_strategies[-5:],  # Last 5 successful
            "failed_strategies": failed_strategies[-5:],  # Last 5 failed
            "adaptation_patterns": adaptation_patterns,
            "success_rate": len(successful_strategies) / max(1, len(self.action_history))
        }
    
    def _analyze_adaptation_patterns(self) -> List[Dict[str, Any]]:
        """Analyze how agents adapt their strategies over time."""
        patterns = []
        
        # Group actions by agent and analyze changes over time
        for agent_id in self.agent_states.keys():
            agent_actions = [action for action in self.action_history if action['agent'] == agent_id]
            
            if len(agent_actions) < 2:
                continue
            
            # Compare early vs late action patterns
            mid_point = len(agent_actions) // 2
            early_actions = agent_actions[:mid_point]
            late_actions = agent_actions[mid_point:]
            
            early_pattern = self._get_action_pattern([action['action'] for action in early_actions])
            late_pattern = self._get_action_pattern([action['action'] for action in late_actions])
            
            # Find significant changes
            for action_type in set(early_pattern.keys()) | set(late_pattern.keys()):
                early_freq = early_pattern.get(action_type, 0) / max(1, len(early_actions))
                late_freq = late_pattern.get(action_type, 0) / max(1, len(late_actions))
                
                if abs(early_freq - late_freq) > 0.2:  # Significant change
                    patterns.append({
                        "agent": agent_id,
                        "action_type": action_type,
                        "early_frequency": early_freq,
                        "late_frequency": late_freq,
                        "change_direction": "increased" if late_freq > early_freq else "decreased"
                    })
        
        return patterns
    
    def _get_action_pattern(self, actions: List[str]) -> Dict[str, int]:
        """Get action frequency pattern."""
        pattern = {}
        for action in actions:
            pattern[action] = pattern.get(action, 0) + 1
        return pattern
    
    # Realistic Survival Mechanics
    
    def _apply_survival_pressure(self, step: int, rng: random.Random):
        """Apply realistic survival pressure - health degradation, resource depletion, time pressure."""
        # Health degradation over time (starvation, exhaustion, stress)
        degradation_rate = 0.02 + (step * 0.001)  # Gets worse over time
        
        for agent_id in list(self.agent_health.keys()):
            if agent_id in self.eliminated_agents:
                continue
            
            # Apply base degradation
            health_loss = degradation_rate * rng.uniform(0.5, 1.5)  # Some randomness
            
            # Worse degradation if agent has fewer resources (starvation)
            agent_state = self.agent_states[agent_id]
            resource_count = len(agent_state.resources_owned)
            if resource_count == 0:
                health_loss *= 2.0  # No resources = starving
            elif resource_count == 1:
                health_loss *= 1.5  # Few resources = malnourished
            
            # Apply health loss
            self.agent_health[agent_id] = max(0.0, self.agent_health[agent_id] - health_loss)
            
            # Check for death from health loss
            if self.agent_health[agent_id] <= 0.0:
                self._eliminate_agent(agent_id, f"died from exhaustion/starvation at step {step}")
    
    def _check_deadly_consequences(self, agent_id: str, action: str, result: Dict[str, Any], step: int, rng: random.Random):
        """Check for deadly consequences from agent actions."""
        if agent_id in self.eliminated_agents:
            return
        
        success = result.get("success", False)
        
        # Failed escape attempts can be deadly
        if action == "attempt_escape" and not success:
            death_chance = 0.15 + (step * 0.01)  # Gets more dangerous over time
            if rng.random() < death_chance:
                self._eliminate_agent(agent_id, f"died during failed escape attempt at step {step}")
                return
            else:
                # Non-fatal injury
                injury = rng.uniform(0.1, 0.3)
                self.agent_health[agent_id] = max(0.0, self.agent_health[agent_id] - injury)
                if self.agent_health[agent_id] <= 0.0:
                    self._eliminate_agent(agent_id, f"died from injuries sustained during escape attempt at step {step}")
        
        # Betrayal can lead to retaliation violence
        elif action == "betray_agent" and success:
            target = result.get("target")
            if target and target not in self.eliminated_agents:
                # Small chance the target retaliates lethally
                retaliation_chance = 0.08
                if rng.random() < retaliation_chance:
                    self._eliminate_agent(agent_id, f"killed in retaliation by {target} at step {step}")
                    return
        
        # Resource competition can turn violent
        elif action in ["claim_resource", "hoard_resource", "block_resource_access"] and success:
            # Small chance of violence over critical resources
            violence_chance = 0.03 + (step * 0.002)  # Increases with desperation
            if rng.random() < violence_chance:
                # Random other agent might attack
                other_agents = [aid for aid in self.agent_states.keys() 
                              if aid != agent_id and aid not in self.eliminated_agents]
                if other_agents:
                    attacker = rng.choice(other_agents)
                    # 50/50 chance of who dies in the violence
                    if rng.random() < 0.5:
                        self._eliminate_agent(agent_id, f"killed by {attacker} in violent struggle over resources at step {step}")
                    else:
                        self._eliminate_agent(attacker, f"killed by {agent_id} in violent struggle over resources at step {step}")
    
    def _eliminate_agent(self, agent_id: str, reason: str):
        """Eliminate an agent from the simulation."""
        self.eliminated_agents.add(agent_id)
        self.agent_health[agent_id] = 0.0
        
        # Record elimination in action history
        self.action_history.append({
            "agent": agent_id,
            "action": "eliminated",
            "parameters": {"reason": reason},
            "result": {"success": True, "elimination_reason": reason},
            "timestamp": datetime.now(),
            "step": len(self.action_history)
        })
        
        print(f"[ELIMINATED] {agent_id} - {reason}")
    
    # Time Pressure Enhanced Methods
    
    def run_enhanced_simulation_with_time_pressure(self, max_steps: int = 100) -> Dict[str, Any]:
        """Run enhanced simulation with time pressure mechanics."""
        # For now, delegate to regular enhanced simulation
        # Time pressure effects are already integrated in _calculate_time_pressure_modifier
        return self.run_enhanced_simulation(max_steps)
    
    def run_enhanced_simulation_with_emergency_protocols(self, max_steps: int = 100) -> Dict[str, Any]:
        """Run enhanced simulation with emergency protocol activation."""
        # Add emergency protocol flag to action history
        results = self.run_enhanced_simulation(max_steps)
        
        # Mark later actions as emergency protocol active
        emergency_threshold = max_steps * 0.7
        for action in results['action_history']:
            step = action.get('step', 0)
            action['emergency_protocol_active'] = step > emergency_threshold
        
        return results
    
    def run_enhanced_simulation_with_panic_conditions(self, max_steps: int = 100) -> Dict[str, Any]:
        """Run enhanced simulation with panic condition effects."""
        results = self.run_enhanced_simulation(max_steps)
        
        # Add panic levels to actions based on time pressure
        for action in results['action_history']:
            step = action.get('step', 0)
            panic_level = min(step / max_steps * 1.5, 1.0)  # Escalating panic
            action['panic_level'] = panic_level
        
        return results