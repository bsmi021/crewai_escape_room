"""
Async Decision Engine with Circuit Breaker and Multi-Agent Coordination

Implements advanced decision-making capabilities with async processing,
circuit breaker pattern, and multi-agent negotiation protocols.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Union
import concurrent.futures
import logging

from ..core_architecture import (
    IDecisionEngine, PerceptionData, DecisionData, ComponentState
)
from ..llm.client import OptimizedLLMClient
from ..llm.fallback import FallbackDecisionGenerator
from ..llm.exceptions import CircuitOpenError, LLMError
from .handoff_protocol import DecisionHandoff


logger = logging.getLogger(__name__)


class AsyncDecisionEngine(IDecisionEngine):
    """
    Advanced async decision engine with circuit breaker and multi-agent coordination
    
    Features:
    - Concurrent decision generation for all agents
    - Circuit breaker pattern for LLM failures
    - Fallback decision systems
    - Performance monitoring and optimization
    - Agent memory integration
    """
    
    def __init__(self, crewai_agents: List[Any], config: Dict[str, Any]):
        """
        Initialize the async decision engine
        
        Args:
            crewai_agents: List of CrewAI agent instances
            config: Configuration dictionary
        """
        self.crewai_agents = crewai_agents
        self.config = config
        self.state = ComponentState.UNINITIALIZED
        
        # Concurrency settings
        self.max_concurrent = config.get("concurrency", {}).get("max_concurrent_decisions", 3)
        self.decision_timeout = config.get("concurrency", {}).get("decision_timeout", 5.0)
        
        # LLM client with circuit breaker
        llm_config = config.get("llm", {})
        llm_config.update(config.get("circuit_breaker", {}))
        self.llm_client = OptimizedLLMClient(llm_config)
        
        # Fallback generator
        self.fallback_generator = FallbackDecisionGenerator()
        
        # Agent memory integration
        self.memory_integration = config.get("memory_integration", False)
        self.memory_manager = config.get("memory_manager")
        self.agent_memories: Dict[str, List[Dict[str, Any]]] = {}
        
        # Trust system integration
        self.trust_integration = config.get("trust_integration", False)
        self.trust_tracker = config.get("trust_tracker")
        
        # Performance tracking
        self.performance_metrics = {
            "total_decisions": 0,
            "successful_decisions": 0,
            "fallback_decisions": 0,
            "average_decision_time": 0.0,
            "circuit_breaker_opens": 0,
            "last_reset": datetime.now()
        }
        
        # Initialize agent mappings
        self.agent_mapping = {}
        self._initialize_agent_mapping()
    
    def _initialize_agent_mapping(self):
        """Initialize mapping between agent IDs and CrewAI agents"""
        for agent in self.crewai_agents:
            # Extract agent type from role
            role = getattr(agent, 'role', '').lower()
            agent_id = role.replace(' ', '_')
            self.agent_mapping[agent_id] = agent
            
            # Initialize memory for this agent
            self.agent_memories[agent_id] = []
    
    async def initialize(self) -> None:
        """Initialize the decision engine"""
        self.state = ComponentState.INITIALIZING
        
        try:
            # Validate configuration
            self._validate_config()
            
            # Initialize components
            await self._initialize_components()
            
            self.state = ComponentState.READY
            logger.info("AsyncDecisionEngine initialized successfully")
            
        except Exception as e:
            self.state = ComponentState.ERROR
            logger.error(f"Failed to initialize AsyncDecisionEngine: {e}")
            raise
    
    def _validate_config(self):
        """Validate configuration parameters"""
        if not self.crewai_agents:
            raise ValueError("No CrewAI agents provided")
        
        if self.max_concurrent <= 0:
            raise ValueError("max_concurrent_decisions must be positive")
        
        if self.decision_timeout <= 0:
            raise ValueError("decision_timeout must be positive")
    
    async def _initialize_components(self):
        """Initialize internal components"""
        # Test LLM client connectivity (optional)
        try:
            test_response = await self.llm_client.generate_decision(
                "Test connectivity", "system"
            )
            logger.info("LLM client connectivity verified")
        except Exception as e:
            logger.warning(f"LLM connectivity test failed: {e}")
    
    async def reason_and_decide(self, perceptions: Dict[str, PerceptionData]) -> Dict[str, DecisionData]:
        """
        Generate decisions for all agents concurrently
        
        Args:
            perceptions: Dictionary mapping agent_id to PerceptionData
            
        Returns:
            Dictionary mapping agent_id to DecisionData
        """
        if self.state != ComponentState.READY:
            raise RuntimeError(f"Cannot generate decisions in state: {self.state}")
        
        start_time = datetime.now()
        decisions = {}
        fallback_agents = []
        
        try:
            # Create decision tasks for each agent
            tasks = []
            agent_ids = []
            
            for agent_id, perception in perceptions.items():
                task = self._generate_agent_decision(agent_id, perception)
                tasks.append(task)
                agent_ids.append(agent_id)
            
            # Execute tasks concurrently with semaphore for concurrency control
            semaphore = asyncio.Semaphore(self.max_concurrent)
            
            async def bounded_task(task, agent_id):
                async with semaphore:
                    return await task
            
            bounded_tasks = [
                bounded_task(task, agent_id) 
                for task, agent_id in zip(tasks, agent_ids)
            ]
            
            # Wait for all decisions with timeout
            results = await asyncio.wait_for(
                asyncio.gather(*bounded_tasks, return_exceptions=True),
                timeout=self.decision_timeout
            )
            
            # Process results
            for i, result in enumerate(results):
                agent_id = agent_ids[i]
                
                if isinstance(result, Exception):
                    # Generate fallback decision
                    logger.warning(f"Decision generation failed for {agent_id}: {result}")
                    fallback_decision = self._generate_fallback_decision(
                        agent_id, perceptions[agent_id]
                    )
                    decisions[agent_id] = fallback_decision
                    fallback_agents.append(agent_id)
                else:
                    decisions[agent_id] = result
            
            # Update performance metrics
            end_time = datetime.now()
            decision_time = (end_time - start_time).total_seconds()
            self._update_performance_metrics(len(decisions), len(fallback_agents), decision_time)
            
            return decisions
            
        except asyncio.TimeoutError:
            logger.error(f"Decision generation timed out after {self.decision_timeout}s")
            
            # Generate fallback decisions for all agents
            for agent_id, perception in perceptions.items():
                if agent_id not in decisions:
                    fallback_decision = self._generate_fallback_decision(agent_id, perception)
                    decisions[agent_id] = fallback_decision
                    fallback_agents.append(agent_id)
            
            return decisions
        
        except Exception as e:
            logger.error(f"Unexpected error in decision generation: {e}")
            raise
    
    async def _generate_agent_decision(self, agent_id: str, perception: PerceptionData) -> DecisionData:
        """
        Generate decision for a single agent
        
        Args:
            agent_id: ID of the agent
            perception: Perception data for the agent
            
        Returns:
            DecisionData for the agent
        """
        try:
            # Get CrewAI agent
            crewai_agent = self.agent_mapping.get(agent_id)
            if not crewai_agent:
                logger.warning(f"No CrewAI agent found for {agent_id}")
                return self._generate_fallback_decision(agent_id, perception)
            
            # Create reasoning prompt
            prompt = self._create_reasoning_prompt(agent_id, perception)
            
            # Add memory context if enabled
            if self.memory_integration:
                prompt = self._add_memory_context(agent_id, prompt)
            
            # Add trust context if enabled
            if self.trust_integration:
                prompt = self._add_trust_context(agent_id, prompt, perception)
            
            # Generate decision using LLM
            try:
                llm_response = await self.llm_client.generate_decision(prompt, agent_id)
                
                # Parse LLM response into structured decision
                decision = self._parse_llm_response(agent_id, llm_response, perception)
                
                # Calculate confidence based on response quality
                confidence = self._calculate_decision_confidence(llm_response, perception)
                decision.confidence_level = confidence
                
                return decision
                
            except (CircuitOpenError, LLMError) as e:
                logger.warning(f"LLM error for {agent_id}: {e}")
                return self._generate_fallback_decision(agent_id, perception)
            
        except Exception as e:
            logger.error(f"Error generating decision for {agent_id}: {e}")
            return self._generate_fallback_decision(agent_id, perception)
    
    def _create_reasoning_prompt(self, agent_id: str, perception: PerceptionData) -> str:
        """Create reasoning prompt for the agent"""
        agent_type = agent_id.replace('_', ' ').title()
        
        prompt_parts = [
            f"You are a {agent_type} in an escape room scenario.",
            f"Current situation analysis:",
            f"- Your position: {perception.spatial_data.get('current_position', 'unknown')}",
            f"- Nearby agents: {', '.join(perception.nearby_agents) if perception.nearby_agents else 'none'}",
            f"- Available actions: {', '.join(perception.available_actions)}",
            "",
            f"Environmental conditions:",
            f"- Time pressure: {perception.environmental_state.get('time_pressure', 0.0):.1f}",
            f"- Resource availability: {len(perception.resources.get('tools', []))} tools available",
            "",
            f"Your role as {agent_type}:",
        ]
        
        # Add role-specific context
        if "strategist" in agent_id.lower():
            prompt_parts.append("- Focus on analysis, planning, and risk assessment")
            prompt_parts.append("- Consider long-term strategies and puzzle-solving approaches")
        elif "mediator" in agent_id.lower():
            prompt_parts.append("- Focus on team coordination and communication")
            prompt_parts.append("- Facilitate cooperation and resolve conflicts")
        elif "survivor" in agent_id.lower():
            prompt_parts.append("- Focus on immediate survival and practical solutions")
            prompt_parts.append("- Prioritize resource gathering and escape opportunities")
        
        prompt_parts.extend([
            "",
            "Based on this situation, choose ONE action from your available actions and explain your reasoning.",
            "Format your response as: ACTION: [chosen_action] REASONING: [your reasoning]"
        ])
        
        return "\n".join(prompt_parts)
    
    def _add_memory_context(self, agent_id: str, prompt: str) -> str:
        """Add memory context to the prompt"""
        if agent_id not in self.agent_memories:
            return prompt
        
        memories = self.agent_memories[agent_id]
        if not memories:
            return prompt
        
        # Get recent memories
        recent_memories = memories[-5:]  # Last 5 memories
        
        memory_context = "\nRecent experiences:\n"
        for memory in recent_memories:
            if "action" in memory and "result" in memory:
                memory_context += f"- {memory['action']}: {memory['result']}\n"
        
        return prompt + memory_context
    
    def _add_trust_context(self, agent_id: str, prompt: str, perception: PerceptionData) -> str:
        """Add trust relationship context to the prompt"""
        if not self.trust_tracker or not perception.nearby_agents:
            return prompt
        
        trust_context = "\nTeam relationships:\n"
        for other_agent in perception.nearby_agents:
            trust_level = self.trust_tracker.get_trust_level(agent_id, other_agent)
            if trust_level is not None:
                trust_desc = "high" if trust_level > 0.7 else "medium" if trust_level > 0.3 else "low"
                trust_context += f"- Trust with {other_agent}: {trust_desc} ({trust_level:.2f})\n"
        
        return prompt + trust_context
    
    def _parse_llm_response(self, agent_id: str, response: str, perception: PerceptionData) -> DecisionData:
        """Parse LLM response into structured DecisionData"""
        response_lower = response.lower()
        
        # Extract action and reasoning
        chosen_action = "observe"  # Default
        reasoning = response
        
        # Simple parsing - look for ACTION: and REASONING: markers
        if "action:" in response_lower:
            parts = response.split("REASONING:")
            if len(parts) >= 2:
                action_part = parts[0].replace("ACTION:", "").strip()
                reasoning = parts[1].strip()
                
                # Find the action in available actions
                for action in perception.available_actions:
                    if action.lower() in action_part.lower():
                        chosen_action = action
                        break
        else:
            # Fallback: look for action words in response
            for action in perception.available_actions:
                if action.lower() in response_lower:
                    chosen_action = action
                    break
        
        # Generate action parameters
        action_parameters = self._generate_action_parameters(chosen_action, perception)
        
        # Generate fallback actions
        fallback_actions = self._generate_fallback_actions(chosen_action, perception.available_actions)
        
        return DecisionData(
            agent_id=agent_id,
            timestamp=datetime.now(),
            chosen_action=chosen_action,
            action_parameters=action_parameters,
            reasoning=reasoning,
            confidence_level=0.8,  # Will be updated by caller
            fallback_actions=fallback_actions
        )
    
    def _generate_action_parameters(self, action: str, perception: PerceptionData) -> Dict[str, Any]:
        """Generate parameters for the chosen action"""
        params = {}
        
        if action == "move":
            current_pos = perception.spatial_data.get("current_position", (0, 0))
            movement_options = perception.spatial_data.get("movement_options", [])
            if movement_options:
                # Choose first available option
                params["target_position"] = movement_options[0]
            else:
                params["target_position"] = current_pos
            params["speed"] = "normal"
        
        elif action == "communicate":
            if perception.nearby_agents:
                params["target"] = perception.nearby_agents[0]
                params["message"] = "status_update"
            else:
                params["target"] = "broadcast"
                params["message"] = "requesting_assistance"
        
        elif action in ["examine", "analyze"]:
            nearby_objects = perception.spatial_data.get("nearby_objects", {})
            if nearby_objects:
                params["target"] = list(nearby_objects.keys())[0]
            else:
                params["target"] = "environment"
            params["depth"] = "detailed" if action == "analyze" else "surface"
        
        elif action == "coordinate":
            if perception.nearby_agents:
                params["targets"] = perception.nearby_agents
                params["plan"] = "collaborative_action"
        
        return params
    
    def _generate_fallback_actions(self, chosen_action: str, available_actions: List[str]) -> List[str]:
        """Generate fallback actions"""
        fallbacks = []
        safe_actions = ["observe", "wait", "examine"]
        
        for action in safe_actions:
            if action != chosen_action and action in available_actions:
                fallbacks.append(action)
        
        # Add some available actions as additional fallbacks
        for action in available_actions:
            if action != chosen_action and action not in fallbacks:
                fallbacks.append(action)
                if len(fallbacks) >= 3:
                    break
        
        return fallbacks
    
    def _calculate_decision_confidence(self, response: str, perception: PerceptionData) -> float:
        """Calculate confidence level based on response quality"""
        base_confidence = 0.7
        
        # Adjust based on response length and detail
        if len(response) > 100:
            base_confidence += 0.1
        if len(response) < 20:
            base_confidence -= 0.2
        
        # Adjust based on certainty indicators
        uncertain_words = ["maybe", "perhaps", "might", "could", "unsure", "idk"]
        certain_words = ["will", "must", "should", "definitely", "clearly"]
        
        response_lower = response.lower()
        for word in uncertain_words:
            if word in response_lower:
                base_confidence -= 0.1
        
        for word in certain_words:
            if word in response_lower:
                base_confidence += 0.05
        
        # Ensure confidence is in valid range
        return max(0.1, min(1.0, base_confidence))
    
    def _generate_fallback_decision(self, agent_id: str, perception: PerceptionData) -> DecisionData:
        """Generate fallback decision using rule-based system"""
        return self.fallback_generator.generate_fallback_decision(perception)
    
    def update_agent_memory(self, agent_id: str, experience: Dict[str, Any]) -> None:
        """Update agent memory with new experience"""
        if agent_id not in self.agent_memories:
            self.agent_memories[agent_id] = []
        
        experience_entry = {
            "timestamp": datetime.now(),
            "experience": experience,
            "type": experience.get("type", "general")
        }
        
        self.agent_memories[agent_id].append(experience_entry)
        
        # Keep only recent memories (last 50)
        if len(self.agent_memories[agent_id]) > 50:
            self.agent_memories[agent_id] = self.agent_memories[agent_id][-50:]
        
        # Update external memory manager if available
        if self.memory_manager:
            try:
                self.memory_manager.store_memory(agent_id, experience)
            except Exception as e:
                logger.warning(f"Failed to store memory externally: {e}")
    
    def _update_performance_metrics(self, total_decisions: int, fallback_count: int, decision_time: float):
        """Update performance metrics"""
        self.performance_metrics["total_decisions"] += total_decisions
        self.performance_metrics["successful_decisions"] += (total_decisions - fallback_count)
        self.performance_metrics["fallback_decisions"] += fallback_count
        
        # Update average decision time
        current_avg = self.performance_metrics["average_decision_time"]
        total_calls = self.performance_metrics["total_decisions"]
        
        if total_calls > 0:
            self.performance_metrics["average_decision_time"] = (
                (current_avg * (total_calls - total_decisions) + decision_time) / total_calls
            )
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics"""
        metrics = self.performance_metrics.copy()
        
        # Add LLM client metrics
        llm_metrics = self.llm_client.get_performance_metrics()
        metrics["llm"] = llm_metrics
        
        # Calculate derived metrics
        if metrics["total_decisions"] > 0:
            metrics["fallback_rate"] = metrics["fallback_decisions"] / metrics["total_decisions"]
        else:
            metrics["fallback_rate"] = 0.0
        
        return metrics
    
    def create_handoff(self, decisions: Dict[str, DecisionData], 
                      negotiation_outcomes: Dict[str, Any] = None,
                      llm_response_time: float = 0.0) -> DecisionHandoff:
        """Create handoff for Agent C (Action Translation)"""
        reasoning_confidence = {
            agent_id: decision.confidence_level 
            for agent_id, decision in decisions.items()
        }
        
        fallback_agents = [
            agent_id for agent_id, decision in decisions.items()
            if decision.reasoning.startswith("Fallback decision")
        ]
        
        return DecisionHandoff(
            decisions=decisions,
            reasoning_confidence=reasoning_confidence,
            decision_timestamp=datetime.now(),
            llm_response_time=llm_response_time,
            negotiation_outcomes=negotiation_outcomes or {},
            fallback_decisions_used=fallback_agents
        )
    
    @property
    def circuit_breaker(self):
        """Access to circuit breaker for monitoring"""
        return self.llm_client.circuit_breaker