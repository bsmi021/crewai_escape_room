"""
Fallback Decision Generator for LLM Failures

Provides rule-based decision generation when LLM services are unavailable.
"""

import random
from datetime import datetime
from typing import Dict, Any, List

from ..core_architecture import PerceptionData, DecisionData


class FallbackDecisionGenerator:
    """
    Rule-based decision generator for fallback scenarios
    
    When LLM services fail, this generator provides reasonable decisions
    based on agent roles and current perceptions.
    """
    
    def __init__(self):
        # Define agent-specific action preferences
        self.agent_preferences = {
            "strategist": {
                "primary": ["analyze", "examine", "assess_risk", "plan"],
                "secondary": ["communicate", "coordinate", "move"],
                "fallback": ["observe", "wait"]
            },
            "mediator": {
                "primary": ["communicate", "coordinate", "mediate", "facilitate"],
                "secondary": ["examine", "negotiate", "move"],
                "fallback": ["observe", "wait"]
            },
            "survivor": {
                "primary": ["survive", "use_tool", "search", "move"],
                "secondary": ["examine", "escape_attempt", "resource_management"],
                "fallback": ["observe", "wait"]
            }
        }
        
        # Action selection rules based on environmental conditions
        self.condition_rules = {
            "high_time_pressure": {
                "strategist": ["analyze", "plan"],
                "mediator": ["coordinate", "communicate"],
                "survivor": ["escape_attempt", "use_tool"]
            },
            "low_energy": {
                "strategist": ["observe", "examine"],
                "mediator": ["communicate", "wait"],
                "survivor": ["wait", "resource_management"]
            },
            "nearby_agents": {
                "strategist": ["communicate", "coordinate"],
                "mediator": ["mediate", "facilitate"],
                "survivor": ["communicate", "coordinate"]
            },
            "resources_available": {
                "strategist": ["examine", "analyze"],
                "mediator": ["coordinate", "negotiate"],
                "survivor": ["search", "use_tool"]
            }
        }
    
    def generate_fallback_decision(self, perception: PerceptionData) -> DecisionData:
        """
        Generate a fallback decision based on perception data
        
        Args:
            perception: Current perception data for the agent
            
        Returns:
            DecisionData with fallback decision
        """
        agent_id = perception.agent_id
        agent_type = self._get_agent_type(agent_id)
        
        # Analyze current conditions
        conditions = self._analyze_conditions(perception)
        
        # Select best action based on rules
        chosen_action = self._select_action(agent_type, perception.available_actions, conditions)
        
        # Generate action parameters
        action_parameters = self._generate_action_parameters(chosen_action, perception)
        
        # Generate reasoning
        reasoning = self._generate_reasoning(agent_type, chosen_action, conditions)
        
        # Generate fallback actions
        fallback_actions = self._generate_fallback_actions(agent_type, chosen_action, perception.available_actions)
        
        # Lower confidence for fallback decisions
        confidence_level = self._calculate_fallback_confidence(chosen_action, conditions)
        
        return DecisionData(
            agent_id=agent_id,
            timestamp=datetime.now(),
            chosen_action=chosen_action,
            action_parameters=action_parameters,
            reasoning=reasoning,
            confidence_level=confidence_level,
            fallback_actions=fallback_actions
        )
    
    def _get_agent_type(self, agent_id: str) -> str:
        """Determine agent type from agent ID"""
        agent_id_lower = agent_id.lower()
        if "strategist" in agent_id_lower:
            return "strategist"
        elif "mediator" in agent_id_lower:
            return "mediator"
        elif "survivor" in agent_id_lower:
            return "survivor"
        else:
            return "strategist"  # Default fallback
    
    def _analyze_conditions(self, perception: PerceptionData) -> List[str]:
        """Analyze current conditions to inform decision making"""
        conditions = []
        
        # Check time pressure
        env_state = perception.environmental_state
        time_pressure = env_state.get("time_pressure", 0.0)
        if time_pressure > 0.7:
            conditions.append("high_time_pressure")
        
        # Check energy level
        resources = perception.resources
        energy = resources.get("energy", 1.0)
        if energy < 0.3:
            conditions.append("low_energy")
        
        # Check for nearby agents
        if perception.nearby_agents:
            conditions.append("nearby_agents")
        
        # Check for available resources
        available_resources = resources.get("available_resources", [])
        tools = resources.get("tools", [])
        if available_resources or tools:
            conditions.append("resources_available")
        
        # Check for threats
        threat_level = env_state.get("threat_level", 0.0)
        if threat_level > 0.6:
            conditions.append("high_threat")
        
        # Check for puzzles or obstacles
        room_state = env_state.get("room_state", {})
        if room_state.get("puzzles_unsolved") or room_state.get("locked_doors"):
            conditions.append("obstacles_present")
        
        return conditions
    
    def _select_action(self, agent_type: str, available_actions: List[str], conditions: List[str]) -> str:
        """Select the best action based on agent type and conditions"""
        preferences = self.agent_preferences.get(agent_type, self.agent_preferences["strategist"])
        
        # Start with condition-based rules
        for condition in conditions:
            if condition in self.condition_rules:
                condition_actions = self.condition_rules[condition].get(agent_type, [])
                for action in condition_actions:
                    if action in available_actions:
                        return action
        
        # Fall back to agent preferences
        for preference_level in ["primary", "secondary", "fallback"]:
            preferred_actions = preferences[preference_level]
            for action in preferred_actions:
                if action in available_actions:
                    return action
        
        # Ultimate fallback - any available action
        if available_actions:
            return available_actions[0]
        
        return "observe"  # Default action
    
    def _generate_action_parameters(self, action: str, perception: PerceptionData) -> Dict[str, Any]:
        """Generate parameters for the chosen action"""
        params = {}
        
        if action == "move":
            # Simple movement strategy
            current_pos = perception.spatial_data.get("current_position", (0, 0))
            movement_options = perception.spatial_data.get("movement_options", [])
            if movement_options:
                # Choose a random valid move
                target_pos = random.choice(movement_options)
                params["target_position"] = target_pos
                params["speed"] = "normal"
            else:
                # Stay in place
                params["target_position"] = current_pos
                params["speed"] = "normal"
        
        elif action == "communicate":
            nearby_agents = perception.nearby_agents
            if nearby_agents:
                params["target"] = nearby_agents[0]  # Communicate with first nearby agent
                params["message"] = "status_update"
            else:
                params["target"] = "broadcast"
                params["message"] = "requesting_assistance"
        
        elif action in ["examine", "analyze"]:
            # Examine nearby objects or environment
            nearby_objects = perception.spatial_data.get("nearby_objects", {})
            if nearby_objects:
                # Examine first available object
                object_name = list(nearby_objects.keys())[0]
                params["target"] = object_name
            else:
                params["target"] = "environment"
            params["depth"] = "detailed" if action == "analyze" else "surface"
        
        elif action == "use_tool":
            tools = perception.resources.get("tools", [])
            if tools:
                params["tool"] = tools[0]  # Use first available tool
                params["purpose"] = "investigation"
            else:
                params["tool"] = "hands"
                params["purpose"] = "manual_search"
        
        elif action == "coordinate":
            nearby_agents = perception.nearby_agents
            if nearby_agents:
                params["targets"] = nearby_agents
                params["plan"] = "collaborative_action"
            else:
                params["targets"] = ["all_agents"]
                params["plan"] = "information_sharing"
        
        return params
    
    def _generate_reasoning(self, agent_type: str, action: str, conditions: List[str]) -> str:
        """Generate reasoning for the fallback decision"""
        base_reasoning = f"Fallback decision due to LLM unavailability. As a {agent_type}, "
        
        if "high_time_pressure" in conditions:
            base_reasoning += f"with time running out, I chose to {action} for maximum efficiency. "
        elif "low_energy" in conditions:
            base_reasoning += f"with low energy, I chose to {action} to conserve resources. "
        elif "nearby_agents" in conditions:
            base_reasoning += f"with team members nearby, I chose to {action} for coordination. "
        elif "resources_available" in conditions:
            base_reasoning += f"with resources available, I chose to {action} to utilize them. "
        else:
            base_reasoning += f"based on current situation analysis, I chose to {action}. "
        
        base_reasoning += "This decision follows established protocols for my role."
        
        return base_reasoning
    
    def _generate_fallback_actions(self, agent_type: str, chosen_action: str, available_actions: List[str]) -> List[str]:
        """Generate fallback actions if the chosen action fails"""
        preferences = self.agent_preferences.get(agent_type, self.agent_preferences["strategist"])
        fallbacks = []
        
        # Add actions from fallback category
        for action in preferences["fallback"]:
            if action != chosen_action and action in available_actions:
                fallbacks.append(action)
        
        # Add secondary actions
        for action in preferences["secondary"]:
            if action != chosen_action and action in available_actions and action not in fallbacks:
                fallbacks.append(action)
        
        # Ensure we have at least one fallback
        if not fallbacks:
            safe_actions = ["observe", "wait", "examine"]
            for action in safe_actions:
                if action != chosen_action and action in available_actions:
                    fallbacks.append(action)
                    break
        
        return fallbacks[:3]  # Limit to 3 fallback actions
    
    def _calculate_fallback_confidence(self, action: str, conditions: List[str]) -> float:
        """Calculate confidence level for fallback decisions"""
        base_confidence = 0.4  # Lower confidence for fallbacks
        
        # Increase confidence for well-matched conditions
        confidence_boosts = {
            "high_time_pressure": 0.1,
            "nearby_agents": 0.1,
            "resources_available": 0.1
        }
        
        for condition in conditions:
            if condition in confidence_boosts:
                base_confidence += confidence_boosts[condition]
        
        # Safe actions get higher confidence
        safe_actions = ["observe", "wait", "examine", "communicate"]
        if action in safe_actions:
            base_confidence += 0.1
        
        return min(base_confidence, 0.7)  # Cap at 0.7 for fallbacks