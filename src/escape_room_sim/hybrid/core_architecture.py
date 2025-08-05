"""
Mesa-CrewAI Hybrid Architecture Core Components

This module defines the foundational architecture for integrating Mesa's
agent-based modeling capabilities with CrewAI's LLM-powered reasoning.

Key Design Principles:
- Clean separation of concerns between physics and reasoning
- Testable components with dependency injection
- Performance-optimized with async capabilities
- 100% test coverage with deterministic testing
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import asyncio
import mesa
from crewai import Agent, Task, Crew


class ComponentState(Enum):
    """Component lifecycle states"""
    UNINITIALIZED = "uninitialized"
    INITIALIZING = "initializing"
    READY = "ready"
    RUNNING = "running"
    PAUSED = "paused"
    ERROR = "error"
    SHUTDOWN = "shutdown"


@dataclass
class PerceptionData:
    """Structured perception data from Mesa environment"""
    agent_id: str
    timestamp: datetime
    spatial_data: Dict[str, Any]
    environmental_state: Dict[str, Any]
    nearby_agents: List[str]
    available_actions: List[str]
    resources: Dict[str, Any]
    constraints: Dict[str, Any]


@dataclass
class DecisionData:
    """Structured decision data from CrewAI reasoning"""
    agent_id: str
    timestamp: datetime
    chosen_action: str
    action_parameters: Dict[str, Any]
    reasoning: str
    confidence_level: float
    fallback_actions: List[str]


@dataclass
class MesaAction:
    """Mesa-compatible action representation"""
    agent_id: str
    action_type: str
    parameters: Dict[str, Any]
    expected_duration: float
    prerequisites: List[str]


class IPerceptionPipeline(ABC):
    """Interface for extracting perceptions from Mesa environment"""
    
    @abstractmethod
    def extract_perceptions(self, mesa_model: mesa.Model) -> Dict[str, PerceptionData]:
        """Extract structured perceptions for each agent"""
        pass
    
    @abstractmethod
    def filter_perceptions(self, perceptions: Dict[str, PerceptionData], 
                         agent_id: str) -> PerceptionData:
        """Filter perceptions based on agent capabilities"""
        pass


class IDecisionEngine(ABC):
    """Interface for CrewAI reasoning and decision making"""
    
    @abstractmethod
    async def reason_and_decide(self, perceptions: Dict[str, PerceptionData]) -> Dict[str, DecisionData]:
        """Generate decisions based on perceptions"""
        pass
    
    @abstractmethod
    def update_agent_memory(self, agent_id: str, experience: Dict[str, Any]) -> None:
        """Update agent memory with experience"""
        pass


class IActionTranslator(ABC):
    """Interface for translating CrewAI decisions to Mesa actions"""
    
    @abstractmethod
    def translate_decision(self, decision: DecisionData) -> MesaAction:
        """Translate single decision to Mesa action"""
        pass
    
    @abstractmethod
    def validate_action(self, action: MesaAction, mesa_model: mesa.Model) -> bool:
        """Validate action is legal in current Mesa state"""
        pass


class IStateSynchronizer(ABC):
    """Interface for synchronizing state between frameworks"""
    
    @abstractmethod
    def sync_mesa_to_crewai(self, mesa_model: mesa.Model) -> None:
        """Sync Mesa state changes to CrewAI agents"""
        pass
    
    @abstractmethod
    def sync_crewai_to_mesa(self, decisions: Dict[str, DecisionData], 
                          mesa_model: mesa.Model) -> None:
        """Sync CrewAI decisions to Mesa model"""
        pass


class HybridAgent:
    """
    Hybrid agent that bridges Mesa and CrewAI
    
    Architecture Decision: Composition over inheritance
    - Contains both Mesa agent and CrewAI agent instances
    - Handles conversion between frameworks
    - Maintains unified state representation
    """
    
    def __init__(self, agent_id: str, mesa_agent: mesa.Agent, crewai_agent: Agent):
        self.agent_id = agent_id
        self.mesa_agent = mesa_agent
        self.crewai_agent = crewai_agent
        self.state = ComponentState.UNINITIALIZED
        self.last_perception: Optional[PerceptionData] = None
        self.last_decision: Optional[DecisionData] = None
        self.performance_metrics: Dict[str, float] = {}
    
    def get_unified_state(self) -> Dict[str, Any]:
        """Get unified state representation"""
        return {
            "agent_id": self.agent_id,
            "mesa_position": getattr(self.mesa_agent, 'pos', None),
            "mesa_state": getattr(self.mesa_agent, 'state', {}),
            "crewai_memory": getattr(self.crewai_agent, 'memory', {}),
            "last_action": self.last_decision.chosen_action if self.last_decision else None,
            "performance": self.performance_metrics
        }
    
    def update_performance_metrics(self, metrics: Dict[str, float]) -> None:
        """Update agent performance tracking"""
        self.performance_metrics.update(metrics)


class HybridSimulationEngine:
    """
    Main simulation engine coordinating Mesa and CrewAI
    
    Architecture Decision: Single responsibility principle
    - Mesa handles environment, physics, spatial relationships
    - CrewAI handles reasoning, decisions, natural language
    - Engine orchestrates communication and synchronization
    """
    
    def __init__(self, 
                 mesa_model: mesa.Model,
                 crewai_agents: List[Agent],
                 perception_pipeline: IPerceptionPipeline,
                 decision_engine: IDecisionEngine,
                 action_translator: IActionTranslator,
                 state_synchronizer: IStateSynchronizer):
        
        # Core components
        self.mesa_model = mesa_model
        self.crewai_agents = crewai_agents
        
        # Pipeline components
        self.perception_pipeline = perception_pipeline
        self.decision_engine = decision_engine
        self.action_translator = action_translator
        self.state_synchronizer = state_synchronizer
        
        # Hybrid agents
        self.hybrid_agents: Dict[str, HybridAgent] = {}
        
        # State management
        self.state = ComponentState.UNINITIALIZED
        self.step_count = 0
        self.error_count = 0
        self.performance_history: List[Dict[str, Any]] = []
    
    def initialize(self) -> None:
        """Initialize hybrid simulation"""
        self.state = ComponentState.INITIALIZING
        
        try:
            # Create hybrid agents
            self._create_hybrid_agents()
            
            # Initialize all components
            self._initialize_components()
            
            self.state = ComponentState.READY
            
        except Exception as e:
            self.state = ComponentState.ERROR
            raise RuntimeError(f"Failed to initialize hybrid simulation: {e}")
    
    async def step(self) -> Dict[str, Any]:
        """Execute one hybrid simulation step"""
        if self.state != ComponentState.READY:
            raise RuntimeError(f"Cannot step in state: {self.state}")
        
        self.state = ComponentState.RUNNING
        step_start = datetime.now()
        
        try:
            # 1. Mesa advances environment
            self.mesa_model.step()
            
            # 2. Extract perceptions
            perceptions = self.perception_pipeline.extract_perceptions(self.mesa_model)
            
            # 3. CrewAI reasoning (async for performance)
            decisions = await self.decision_engine.reason_and_decide(perceptions)
            
            # 4. Translate and validate actions
            validated_actions = self._translate_and_validate_actions(decisions)
            
            # 5. Apply actions to Mesa
            self._apply_actions_to_mesa(validated_actions)
            
            # 6. Synchronize state
            self.state_synchronizer.sync_crewai_to_mesa(decisions, self.mesa_model)
            
            # 7. Update performance metrics
            step_duration = (datetime.now() - step_start).total_seconds()
            self._update_performance_metrics(step_duration, validated_actions)
            
            self.step_count += 1
            self.state = ComponentState.READY
            
            return self._create_step_result(step_duration, validated_actions)
            
        except Exception as e:
            self.error_count += 1
            self.state = ComponentState.ERROR
            raise RuntimeError(f"Step {self.step_count} failed: {e}")
    
    def _create_hybrid_agents(self) -> None:
        """Create hybrid agents from Mesa and CrewAI agents"""
        # Architecture Decision: Match agents by role/name
        for crewai_agent in self.crewai_agents:
            agent_id = crewai_agent.role.lower().replace(" ", "_")
            
            # Find corresponding Mesa agent
            mesa_agent = self._find_mesa_agent_by_id(agent_id)
            if mesa_agent:
                hybrid_agent = HybridAgent(agent_id, mesa_agent, crewai_agent)
                self.hybrid_agents[agent_id] = hybrid_agent
    
    def _find_mesa_agent_by_id(self, agent_id: str) -> Optional[mesa.Agent]:
        """Find Mesa agent by ID"""
        # Implementation depends on Mesa model structure
        for agent in self.mesa_model.schedule.agents:
            if hasattr(agent, 'agent_id') and agent.agent_id == agent_id:
                return agent
        return None
    
    def _initialize_components(self) -> None:
        """Initialize all pipeline components"""
        # Components should be dependency-injected and pre-configured
        pass
    
    def _translate_and_validate_actions(self, decisions: Dict[str, DecisionData]) -> Dict[str, MesaAction]:
        """Translate decisions to Mesa actions and validate"""
        validated_actions = {}
        
        for agent_id, decision in decisions.items():
            # Translate decision
            mesa_action = self.action_translator.translate_decision(decision)
            
            # Validate action
            if self.action_translator.validate_action(mesa_action, self.mesa_model):
                validated_actions[agent_id] = mesa_action
            else:
                # Log validation failure and use fallback
                self._handle_invalid_action(agent_id, decision, mesa_action)
        
        return validated_actions
    
    def _apply_actions_to_mesa(self, actions: Dict[str, MesaAction]) -> None:
        """Apply validated actions to Mesa model"""
        for agent_id, action in actions.items():
            hybrid_agent = self.hybrid_agents.get(agent_id)
            if hybrid_agent and hybrid_agent.mesa_agent:
                # Apply action to Mesa agent
                self._execute_mesa_action(hybrid_agent.mesa_agent, action)
    
    def _execute_mesa_action(self, mesa_agent: mesa.Agent, action: MesaAction) -> None:
        """Execute action on Mesa agent"""
        # Implementation depends on Mesa agent interface
        if hasattr(mesa_agent, 'execute_action'):
            mesa_agent.execute_action(action.action_type, action.parameters)
        else:
            # Fallback to direct attribute/method calls
            self._execute_action_fallback(mesa_agent, action)
    
    def _execute_action_fallback(self, mesa_agent: mesa.Agent, action: MesaAction) -> None:
        """Fallback action execution method"""
        # Handle common Mesa actions
        if action.action_type == "move":
            if hasattr(mesa_agent, 'move_to'):
                target_pos = action.parameters.get('target_position')
                if target_pos:
                    mesa_agent.move_to(target_pos)
        elif action.action_type == "interact":
            if hasattr(mesa_agent, 'interact_with'):
                target = action.parameters.get('target')
                if target:
                    mesa_agent.interact_with(target)
    
    def _handle_invalid_action(self, agent_id: str, decision: DecisionData, 
                             invalid_action: MesaAction) -> None:
        """Handle invalid action with fallback strategy"""
        # Try fallback actions from decision
        for fallback_action_name in decision.fallback_actions:
            fallback_decision = DecisionData(
                agent_id=agent_id,
                timestamp=datetime.now(),
                chosen_action=fallback_action_name,
                action_parameters={},
                reasoning="Fallback due to invalid primary action",
                confidence_level=0.5,
                fallback_actions=[]
            )
            
            fallback_mesa_action = self.action_translator.translate_decision(fallback_decision)
            if self.action_translator.validate_action(fallback_mesa_action, self.mesa_model):
                self._execute_mesa_action(
                    self.hybrid_agents[agent_id].mesa_agent, 
                    fallback_mesa_action
                )
                return
        
        # No valid fallback - agent does nothing this step
        pass
    
    def _update_performance_metrics(self, step_duration: float, 
                                  actions: Dict[str, MesaAction]) -> None:
        """Update performance tracking"""
        metrics = {
            "step_duration": step_duration,
            "actions_executed": len(actions),
            "llm_reasoning_time": 0.0,  # Would be measured in decision_engine
            "mesa_simulation_time": 0.0,  # Would be measured in mesa_model.step()
            "validation_failures": 0,  # Would be counted during validation
        }
        
        self.performance_history.append({
            "step": self.step_count,
            "timestamp": datetime.now(),
            "metrics": metrics
        })
    
    def _create_step_result(self, duration: float, actions: Dict[str, MesaAction]) -> Dict[str, Any]:
        """Create step result summary"""
        return {
            "step": self.step_count,
            "duration": duration,
            "actions_executed": len(actions),
            "mesa_state": self._extract_mesa_state_summary(),
            "crewai_state": self._extract_crewai_state_summary(),
            "performance": self.performance_history[-1]["metrics"] if self.performance_history else {}
        }
    
    def _extract_mesa_state_summary(self) -> Dict[str, Any]:
        """Extract Mesa state summary"""
        return {
            "agent_count": len(self.mesa_model.schedule.agents),
            "step_count": self.mesa_model.schedule.steps,
            "model_state": "running" if hasattr(self.mesa_model, 'running') else "unknown"
        }
    
    def _extract_crewai_state_summary(self) -> Dict[str, Any]:
        """Extract CrewAI state summary"""
        return {
            "agent_count": len(self.crewai_agents),
            "memory_states": {agent.role: len(agent.memory) if hasattr(agent, 'memory') else 0 
                            for agent in self.crewai_agents}
        }
    
    def get_simulation_state(self) -> Dict[str, Any]:
        """Get complete simulation state"""
        return {
            "engine_state": self.state.value,
            "step_count": self.step_count,
            "error_count": self.error_count,
            "hybrid_agents": {aid: agent.get_unified_state() 
                            for aid, agent in self.hybrid_agents.items()},
            "mesa_summary": self._extract_mesa_state_summary(),
            "crewai_summary": self._extract_crewai_state_summary(),
            "performance_summary": self._get_performance_summary()
        }
    
    def _get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary statistics"""
        if not self.performance_history:
            return {}
        
        recent_metrics = [entry["metrics"] for entry in self.performance_history[-10:]]
        
        return {
            "avg_step_duration": sum(m["step_duration"] for m in recent_metrics) / len(recent_metrics),
            "avg_actions_per_step": sum(m["actions_executed"] for m in recent_metrics) / len(recent_metrics),
            "total_steps": len(self.performance_history)
        }


class HybridSimulationFactory:
    """
    Factory for creating hybrid simulations with proper dependency injection
    
    Architecture Decision: Factory pattern for complex object creation
    - Encapsulates complex initialization logic
    - Allows for different simulation configurations
    - Supports testing with mock dependencies
    """
    
    @staticmethod
    def create_escape_room_simulation(
        room_config: Dict[str, Any],
        agent_configs: List[Dict[str, Any]],
        llm_config: Dict[str, Any]
    ) -> HybridSimulationEngine:
        """Create escape room hybrid simulation"""
        
        # Create Mesa model
        mesa_model = HybridSimulationFactory._create_mesa_escape_room(room_config)
        
        # Create CrewAI agents
        crewai_agents = HybridSimulationFactory._create_crewai_agents(agent_configs, llm_config)
        
        # Create pipeline components
        perception_pipeline = HybridSimulationFactory._create_perception_pipeline()
        decision_engine = HybridSimulationFactory._create_decision_engine(crewai_agents)
        action_translator = HybridSimulationFactory._create_action_translator()
        state_synchronizer = HybridSimulationFactory._create_state_synchronizer()
        
        # Create hybrid engine
        engine = HybridSimulationEngine(
            mesa_model=mesa_model,
            crewai_agents=crewai_agents,
            perception_pipeline=perception_pipeline,
            decision_engine=decision_engine,
            action_translator=action_translator,
            state_synchronizer=state_synchronizer
        )
        
        return engine
    
    @staticmethod
    def _create_mesa_escape_room(config: Dict[str, Any]):
        """Create Mesa escape room model"""
        # Would implement Mesa-based escape room
        pass
    
    @staticmethod
    def _create_crewai_agents(configs: List[Dict[str, Any]], llm_config: Dict[str, Any]) -> List[Agent]:
        """Create CrewAI agents"""
        # Would create agents similar to existing implementation
        pass
    
    @staticmethod
    def _create_perception_pipeline() -> IPerceptionPipeline:
        """Create perception pipeline"""
        # Would return concrete implementation
        pass
    
    @staticmethod
    def _create_decision_engine(agents: List[Agent]) -> IDecisionEngine:
        """Create decision engine"""
        # Would return concrete implementation
        pass
    
    @staticmethod
    def _create_action_translator() -> IActionTranslator:
        """Create action translator"""
        # Would return concrete implementation
        pass
    
    @staticmethod
    def _create_state_synchronizer() -> IStateSynchronizer:
        """Create state synchronizer"""
        # Would return concrete implementation
        pass