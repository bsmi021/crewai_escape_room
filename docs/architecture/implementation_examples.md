# Mesa-CrewAI Implementation Examples

## 1. Concrete Component Implementations

### EscapeRoomPerceptionPipeline Implementation

```python
"""
Concrete implementation of perception pipeline for escape room scenario
"""
from typing import Dict, List, Any, Tuple
import mesa
import numpy as np
from ..hybrid.core_architecture import IPerceptionPipeline, PerceptionData
from datetime import datetime

class EscapeRoomPerceptionPipeline(IPerceptionPipeline):
    """
    Extracts structured perceptions from Mesa escape room environment
    
    Architecture Decision: Domain-specific perception extraction
    - Focuses on escape room specific elements (doors, keys, puzzles)
    - Filters spatial data relevant to escape scenarios
    - Provides rich contextual information for CrewAI reasoning
    """
    
    def __init__(self, perception_radius: float = 5.0, detail_level: str = "full"):
        self.perception_radius = perception_radius
        self.detail_level = detail_level
        self.room_analyzer = RoomStateAnalyzer()
        self.spatial_processor = SpatialRelationshipProcessor()
    
    def extract_perceptions(self, mesa_model: mesa.Model) -> Dict[str, PerceptionData]:
        """Extract structured perceptions for each agent in the model"""
        perceptions = {}
        
        # Get current room state
        room_state = self._analyze_room_state(mesa_model)
        
        for agent in mesa_model.schedule.agents:
            if hasattr(agent, 'agent_id'):
                agent_perception = self._extract_agent_perception(
                    agent, mesa_model, room_state
                )
                perceptions[agent.agent_id] = agent_perception
        
        return perceptions
    
    def filter_perceptions(self, perceptions: Dict[str, PerceptionData], 
                         agent_id: str) -> PerceptionData:
        """Filter perceptions based on agent-specific capabilities"""
        base_perception = perceptions.get(agent_id)
        if not base_perception:
            return self._create_empty_perception(agent_id)
        
        # Agent-specific filtering based on role
        if "strategist" in agent_id.lower():
            return self._filter_for_strategist(base_perception)
        elif "mediator" in agent_id.lower():
            return self._filter_for_mediator(base_perception)
        elif "survivor" in agent_id.lower():
            return self._filter_for_survivor(base_perception)
        
        return base_perception
    
    def _extract_agent_perception(self, agent: mesa.Agent, model: mesa.Model, 
                                room_state: Dict) -> PerceptionData:
        """Extract perception data for a specific agent"""
        agent_pos = getattr(agent, 'pos', (0, 0))
        
        # Spatial data extraction
        spatial_data = {
            "position": agent_pos,
            "facing_direction": getattr(agent, 'direction', 0),
            "movement_range": self._get_movement_range(agent, model),
            "visible_area": self._get_visible_area(agent_pos, model),
            "blocked_paths": self._get_blocked_paths(agent_pos, model)
        }
        
        # Environmental state
        environmental_state = {
            "room_layout": room_state.get("layout", {}),
            "lighting_level": room_state.get("lighting", 1.0),
            "temperature": room_state.get("temperature", 20),
            "air_quality": room_state.get("air_quality", 1.0),
            "structural_integrity": room_state.get("integrity", 1.0),
            "time_remaining": room_state.get("time_limit", 3600)
        }
        
        # Nearby agents
        nearby_agents = self._find_nearby_agents(agent, model)
        
        # Available actions
        available_actions = self._determine_available_actions(agent, model, room_state)
        
        # Resources
        resources = self._identify_accessible_resources(agent_pos, model)
        
        # Constraints
        constraints = self._identify_constraints(agent, model, room_state)
        
        return PerceptionData(
            agent_id=agent.agent_id,
            timestamp=datetime.now(),
            spatial_data=spatial_data,
            environmental_state=environmental_state,
            nearby_agents=nearby_agents,
            available_actions=available_actions,
            resources=resources,
            constraints=constraints
        )
    
    def _analyze_room_state(self, model: mesa.Model) -> Dict[str, Any]:
        """Analyze current room state"""
        return {
            "layout": self._extract_room_layout(model),
            "objects": self._catalog_room_objects(model),
            "exits": self._identify_exits(model),
            "puzzles": self._identify_puzzles(model),
            "hazards": self._identify_hazards(model),
            "lighting": 0.8,  # Example values
            "temperature": 22,
            "air_quality": 0.9,
            "integrity": 1.0,
            "time_limit": getattr(model, 'time_remaining', 3600)
        }
    
    def _find_nearby_agents(self, agent: mesa.Agent, model: mesa.Model) -> List[str]:
        """Find agents within perception radius"""
        nearby = []
        agent_pos = getattr(agent, 'pos', (0, 0))
        
        for other_agent in model.schedule.agents:
            if other_agent == agent:
                continue
            
            other_pos = getattr(other_agent, 'pos', (0, 0))
            distance = np.sqrt(
                (agent_pos[0] - other_pos[0])**2 + 
                (agent_pos[1] - other_pos[1])**2
            )
            
            if distance <= self.perception_radius:
                nearby.append(getattr(other_agent, 'agent_id', 'unknown'))
        
        return nearby
    
    def _determine_available_actions(self, agent: mesa.Agent, model: mesa.Model,
                                   room_state: Dict) -> List[str]:
        """Determine what actions agent can take"""
        actions = ["wait", "observe"]  # Base actions always available
        
        agent_pos = getattr(agent, 'pos', (0, 0))
        
        # Movement actions
        if self._can_move_north(agent_pos, model):
            actions.append("move_north")
        if self._can_move_south(agent_pos, model):
            actions.append("move_south")
        if self._can_move_east(agent_pos, model):
            actions.append("move_east")
        if self._can_move_west(agent_pos, model):
            actions.append("move_west")
        
        # Interaction actions
        nearby_objects = self._get_nearby_objects(agent_pos, model)
        for obj in nearby_objects:
            if obj.get("type") == "door" and obj.get("locked"):
                actions.append("unlock_door")
            elif obj.get("type") == "key":
                actions.append("pick_up_key")
            elif obj.get("type") == "puzzle":
                actions.append("solve_puzzle")
            elif obj.get("type") == "tool":
                actions.append("use_tool")
        
        # Communication actions
        if self._has_nearby_agents(agent, model):
            actions.extend(["communicate", "coordinate", "share_information"])
        
        return actions
    
    def _identify_accessible_resources(self, pos: Tuple[int, int], 
                                     model: mesa.Model) -> Dict[str, Any]:
        """Identify resources accessible to agent"""
        resources = {
            "keys": [],
            "tools": [],
            "information": [],
            "energy": 100,  # Agent's energy level
            "time": getattr(model, 'time_remaining', 3600)
        }
        
        # Check for nearby resources
        nearby_objects = self._get_nearby_objects(pos, model)
        for obj in nearby_objects:
            obj_type = obj.get("type", "unknown")
            if obj_type == "key":
                resources["keys"].append(obj["name"])
            elif obj_type == "tool":
                resources["tools"].append(obj["name"])
            elif obj_type == "information":
                resources["information"].append(obj["content"])
        
        return resources
    
    def _identify_constraints(self, agent: mesa.Agent, model: mesa.Model,
                            room_state: Dict) -> Dict[str, Any]:
        """Identify constraints affecting agent"""
        constraints = {
            "movement": [],
            "actions": [],
            "time": {"remaining": room_state.get("time_limit", 3600)},
            "physical": [],
            "knowledge": []
        }
        
        agent_pos = getattr(agent, 'pos', (0, 0))
        
        # Movement constraints
        if not self._can_move_freely(agent_pos, model):
            constraints["movement"].append("blocked_paths")
        
        # Physical constraints
        if room_state.get("lighting", 1.0) < 0.3:
            constraints["physical"].append("low_visibility")
        
        if room_state.get("air_quality", 1.0) < 0.5:
            constraints["physical"].append("poor_air_quality")
        
        # Action constraints
        agent_energy = getattr(agent, 'energy', 100)
        if agent_energy < 20:
            constraints["actions"].append("low_energy")
        
        return constraints
    
    # Helper methods for spatial analysis
    def _get_movement_range(self, agent: mesa.Agent, model: mesa.Model) -> List[Tuple[int, int]]:
        """Get possible movement positions"""
        # Implementation would calculate reachable positions
        return [(0, 0)]  # Simplified
    
    def _get_visible_area(self, pos: Tuple[int, int], model: mesa.Model) -> List[Tuple[int, int]]:
        """Get visible positions from current location"""
        # Implementation would use line-of-sight algorithms
        return [pos]  # Simplified
    
    def _get_nearby_objects(self, pos: Tuple[int, int], model: mesa.Model) -> List[Dict]:
        """Get objects near position"""
        # Implementation would query Mesa's space for nearby objects
        return []  # Simplified
```

### CrewAIDecisionEngine Implementation

```python
"""
Concrete implementation of decision engine using CrewAI
"""
from typing import Dict, List, Any
import asyncio
from crewai import Agent, Task, Crew, Process
from ..hybrid.core_architecture import IDecisionEngine, DecisionData, PerceptionData
from datetime import datetime

class CrewAIDecisionEngine(IDecisionEngine):
    """
    Uses CrewAI framework for agent reasoning and decision making
    
    Architecture Decision: Contextual task generation
    - Creates tasks dynamically based on current perceptions
    - Maintains agent memory across decisions
    - Processes crew output into structured decisions
    """
    
    def __init__(self, agents: List[Agent], max_reasoning_time: int = 120):
        self.agents = agents
        self.max_reasoning_time = max_reasoning_time
        self.task_factory = DynamicTaskFactory()
        self.output_parser = CrewOutputParser()
        self.memory_manager = AgentMemoryManager()
    
    async def reason_and_decide(self, perceptions: Dict[str, PerceptionData]) -> Dict[str, DecisionData]:
        """Generate decisions based on perceptions using CrewAI reasoning"""
        
        # Create contextual tasks based on perceptions
        tasks = self.task_factory.create_perception_based_tasks(
            perceptions, self.agents
        )
        
        # Create crew for this reasoning session
        crew = Crew(
            agents=self.agents,
            tasks=tasks,
            process=Process.sequential,
            verbose=False,
            memory=True
        )
        
        # Execute reasoning with timeout
        try:
            reasoning_output = await asyncio.wait_for(
                self._execute_crew_reasoning(crew),
                timeout=self.max_reasoning_time
            )
        except asyncio.TimeoutError:
            # Fallback to quick decisions if reasoning takes too long
            return self._generate_fallback_decisions(perceptions)
        
        # Parse crew output into structured decisions
        decisions = self.output_parser.parse_to_decisions(
            reasoning_output, perceptions
        )
        
        # Update agent memories with experience
        self._update_agent_memories(perceptions, decisions)
        
        return decisions
    
    async def _execute_crew_reasoning(self, crew: Crew) -> str:
        """Execute CrewAI reasoning asynchronously"""
        # CrewAI's kickoff is synchronous, so we run it in thread pool
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, crew.kickoff)
        return str(result)
    
    def update_agent_memory(self, agent_id: str, experience: Dict[str, Any]) -> None:
        """Update specific agent memory with experience"""
        self.memory_manager.add_experience(agent_id, experience)
        
        # Update corresponding CrewAI agent memory
        for agent in self.agents:
            if agent.role.lower().replace(" ", "_") == agent_id:
                if hasattr(agent, 'memory'):
                    agent.memory.add(experience)
                break
    
    def _generate_fallback_decisions(self, perceptions: Dict[str, PerceptionData]) -> Dict[str, DecisionData]:
        """Generate simple fallback decisions when reasoning fails"""
        decisions = {}
        
        for agent_id, perception in perceptions.items():
            # Simple heuristic-based decision
            if "move" in perception.available_actions:
                chosen_action = "move_towards_exit"
            elif "communicate" in perception.available_actions:
                chosen_action = "coordinate_with_team"
            else:
                chosen_action = "observe_environment"
            
            decisions[agent_id] = DecisionData(
                agent_id=agent_id,
                timestamp=datetime.now(),
                chosen_action=chosen_action,
                action_parameters={},
                reasoning="Fallback decision due to reasoning timeout",
                confidence_level=0.3,
                fallback_actions=["wait", "observe"]
            )
        
        return decisions
    
    def _update_agent_memories(self, perceptions: Dict[str, PerceptionData],
                             decisions: Dict[str, DecisionData]) -> None:
        """Update all agent memories with current experience"""
        for agent_id in perceptions.keys():
            if agent_id in decisions:
                experience = {
                    "perception": perceptions[agent_id],
                    "decision": decisions[agent_id],
                    "timestamp": datetime.now().isoformat(),
                    "context": "hybrid_simulation_step"
                }
                self.update_agent_memory(agent_id, experience)


class DynamicTaskFactory:
    """Creates CrewAI tasks based on current perceptions and game state"""
    
    def create_perception_based_tasks(self, perceptions: Dict[str, PerceptionData],
                                    agents: List[Agent]) -> List[Task]:
        """Create tasks tailored to current perceptions"""
        tasks = []
        
        # Analyze collective situation
        situation_summary = self._analyze_situation(perceptions)
        
        for i, agent in enumerate(agents):
            agent_id = agent.role.lower().replace(" ", "_")
            agent_perception = perceptions.get(agent_id)
            
            if not agent_perception:
                continue
            
            # Create role-specific task
            if "strategist" in agent_id:
                task = self._create_strategist_task(agent_perception, situation_summary)
            elif "mediator" in agent_id:
                task = self._create_mediator_task(agent_perception, situation_summary)
            elif "survivor" in agent_id:
                task = self._create_survivor_task(agent_perception, situation_summary)
            else:
                task = self._create_generic_task(agent_perception, situation_summary)
            
            task.agent = agent
            tasks.append(task)
        
        return tasks
    
    def _create_strategist_task(self, perception: PerceptionData, 
                              situation: Dict) -> Task:
        """Create strategic analysis task"""
        return Task(
            description=f"""
            STRATEGIC ANALYSIS - Spatial Reasoning Task
            
            Current Spatial Context:
            - Position: {perception.spatial_data.get('position')}
            - Visible Area: {len(perception.spatial_data.get('visible_area', []))} locations
            - Movement Range: {len(perception.spatial_data.get('movement_range', []))} positions
            - Blocked Paths: {perception.spatial_data.get('blocked_paths', [])}
            
            Environmental State:
            - Room Layout: {perception.environmental_state.get('room_layout', 'unknown')}
            - Time Remaining: {perception.environmental_state.get('time_remaining', 'unknown')}
            - Structural Integrity: {perception.environmental_state.get('structural_integrity', 1.0)}
            
            Available Resources:
            - Keys: {perception.resources.get('keys', [])}
            - Tools: {perception.resources.get('tools', [])}
            - Information: {len(perception.resources.get('information', []))} pieces
            
            Team Situation:
            - Nearby Agents: {perception.nearby_agents}
            - Team Status: {situation.get('team_cohesion', 'unknown')}
            
            STRATEGIC ANALYSIS REQUIRED:
            1. Assess the most efficient escape route based on spatial layout
            2. Identify critical bottlenecks or obstacles
            3. Prioritize resource acquisition and usage
            4. Evaluate risk-reward for different approaches
            5. Recommend optimal coordination strategy for team
            
            Provide strategic recommendation with:
            - Primary approach with success probability
            - Resource requirements and allocation
            - Risk assessment and contingency plans
            """,
            expected_output="Strategic recommendation with spatial analysis and resource optimization plan"
        )
    
    def _create_mediator_task(self, perception: PerceptionData,
                            situation: Dict) -> Task:
        """Create team coordination task"""
        return Task(
            description=f"""
            TEAM COORDINATION - Social Dynamics Task
            
            Current Social Context:
            - Nearby Team Members: {perception.nearby_agents}
            - Available Communication: {'communicate' in perception.available_actions}
            - Team Stress Level: {situation.get('stress_level', 0.5)}
            
            Environmental Pressure:
            - Time Pressure: {perception.environmental_state.get('time_remaining', 'unknown')} remaining
            - Physical Constraints: {perception.constraints.get('physical', [])}
            - Action Limitations: {perception.constraints.get('actions', [])}
            
            COORDINATION OBJECTIVES:
            1. Facilitate team communication and consensus
            2. Resolve any conflicts or disagreements
            3. Ensure all team members are aligned on strategy
            4. Monitor team morale and stress levels
            5. Coordinate role assignments and responsibilities
            
            Available Actions for Coordination:
            {[action for action in perception.available_actions if 'communicate' in action or 'coordinate' in action]}
            
            Provide coordination plan with:
            - Communication strategy
            - Role assignments
            - Conflict resolution approach
            - Team motivation tactics
            """,
            expected_output="Team coordination plan with communication strategy and role assignments"
        )
    
    def _create_survivor_task(self, perception: PerceptionData,
                           situation: Dict) -> Task:
        """Create execution and survival task"""
        return Task(
            description=f"""
            SURVIVAL EXECUTION - Action Implementation Task
            
            Current Action Context:
            - Available Actions: {perception.available_actions}
            - Resource Status: Energy {perception.resources.get('energy', 100)}%
            - Time Constraint: {perception.environmental_state.get('time_remaining', 'unknown')}
            - Movement Options: {len(perception.spatial_data.get('movement_range', []))} positions
            
            Immediate Threats/Constraints:
            - Physical: {perception.constraints.get('physical', [])}
            - Movement: {perception.constraints.get('movement', [])}
            - Time: {perception.constraints.get('time', {})}
            
            Resource Inventory:
            - Tools: {perception.resources.get('tools', [])}
            - Keys: {perception.resources.get('keys', [])}
            - Information: {len(perception.resources.get('information', []))} pieces
            
            EXECUTION PRIORITIES:
            1. Implement the agreed strategy with precision
            2. Make critical survival decisions under pressure
            3. Adapt to unexpected obstacles or opportunities
            4. Monitor progress and effectiveness
            5. Determine success/failure and next steps
            
            Based on current situation, choose PRIMARY ACTION:
            {perception.available_actions}
            
            Provide execution decision with:
            - Chosen action and parameters
            - Risk assessment
            - Contingency plans
            - Success/failure criteria
            """,
            expected_output="Execution decision with chosen action, parameters, and success criteria"
        )


class CrewOutputParser:
    """Parses CrewAI output into structured DecisionData"""
    
    def parse_to_decisions(self, crew_output: str, 
                         perceptions: Dict[str, PerceptionData]) -> Dict[str, DecisionData]:
        """Parse crew output into structured decisions"""
        decisions = {}
        
        # Simple parsing - in practice would be more sophisticated
        output_lines = crew_output.split('\n')
        
        for agent_id in perceptions.keys():
            decision = self._extract_agent_decision(
                agent_id, output_lines, perceptions[agent_id]
            )
            decisions[agent_id] = decision
        
        return decisions
    
    def _extract_agent_decision(self, agent_id: str, output_lines: List[str],
                              perception: PerceptionData) -> DecisionData:
        """Extract decision for specific agent from output"""
        
        # Look for action keywords in output
        chosen_action = "observe"  # Default
        confidence = 0.7
        reasoning = "Standard reasoning process"
        parameters = {}
        
        # Simple keyword extraction
        for line in output_lines:
            if any(action in line.lower() for action in perception.available_actions):
                for action in perception.available_actions:
                    if action in line.lower():
                        chosen_action = action
                        break
        
        # Extract confidence if mentioned
        for line in output_lines:
            if "confidence" in line.lower():
                # Simple confidence extraction
                try:
                    confidence = float(line.split()[-1])
                except:
                    confidence = 0.7
        
        return DecisionData(
            agent_id=agent_id,
            timestamp=datetime.now(),
            chosen_action=chosen_action,
            action_parameters=parameters,
            reasoning=reasoning,
            confidence_level=confidence,
            fallback_actions=["wait", "observe"]
        )
```

### EscapeRoomActionTranslator Implementation

```python
"""
Concrete implementation of action translator for escape room
"""
from typing import Dict, Any, Optional
import mesa
from ..hybrid.core_architecture import IActionTranslator, DecisionData, MesaAction

class EscapeRoomActionTranslator(IActionTranslator):
    """
    Translates CrewAI decisions into Mesa-compatible actions
    
    Architecture Decision: Domain-specific action mapping
    - Maps high-level CrewAI decisions to specific Mesa actions
    - Validates actions against current Mesa state
    - Provides fallback actions for invalid operations
    """
    
    def __init__(self):
        self.action_mappings = self._create_action_mappings()
        self.validator = ActionValidator()
    
    def translate_decision(self, decision: DecisionData) -> MesaAction:
        """Translate CrewAI decision to Mesa action"""
        action_type = decision.chosen_action
        parameters = decision.action_parameters.copy()
        
        # Get Mesa action mapping
        mesa_action_info = self.action_mappings.get(action_type)
        if not mesa_action_info:
            # Unknown action - use fallback
            return self._create_fallback_action(decision)
        
        # Transform parameters for Mesa
        mesa_parameters = self._transform_parameters(
            parameters, mesa_action_info.get("parameter_mapping", {})
        )
        
        return MesaAction(
            agent_id=decision.agent_id,
            action_type=mesa_action_info["mesa_action"],
            parameters=mesa_parameters,
            expected_duration=mesa_action_info.get("duration", 1.0),
            prerequisites=mesa_action_info.get("prerequisites", [])
        )
    
    def validate_action(self, action: MesaAction, mesa_model: mesa.Model) -> bool:
        """Validate action is legal in current Mesa state"""
        return self.validator.validate(action, mesa_model)
    
    def _create_action_mappings(self) -> Dict[str, Dict[str, Any]]:
        """Create mapping from CrewAI actions to Mesa actions"""
        return {
            # Movement actions
            "move_north": {
                "mesa_action": "move",
                "parameter_mapping": {"direction": "north"},
                "duration": 1.0,
                "prerequisites": ["can_move"]
            },
            "move_south": {
                "mesa_action": "move", 
                "parameter_mapping": {"direction": "south"},
                "duration": 1.0,
                "prerequisites": ["can_move"]
            },
            "move_east": {
                "mesa_action": "move",
                "parameter_mapping": {"direction": "east"}, 
                "duration": 1.0,
                "prerequisites": ["can_move"]
            },
            "move_west": {
                "mesa_action": "move",
                "parameter_mapping": {"direction": "west"},
                "duration": 1.0, 
                "prerequisites": ["can_move"]
            },
            "move_towards_exit": {
                "mesa_action": "move_towards_target",
                "parameter_mapping": {"target_type": "exit"},
                "duration": 2.0,
                "prerequisites": ["can_move", "exit_visible"]
            },
            
            # Interaction actions
            "pick_up_key": {
                "mesa_action": "pick_up_object",
                "parameter_mapping": {"object_type": "key"},
                "duration": 1.0,
                "prerequisites": ["object_nearby", "has_space"]
            },
            "unlock_door": {
                "mesa_action": "use_object",
                "parameter_mapping": {"object_type": "key", "target_type": "door"},
                "duration": 2.0,
                "prerequisites": ["has_key", "door_nearby"]
            },
            "solve_puzzle": {
                "mesa_action": "interact_with_object",
                "parameter_mapping": {"object_type": "puzzle"},
                "duration": 5.0,
                "prerequisites": ["puzzle_nearby", "has_energy"]
            },
            "use_tool": {
                "mesa_action": "use_object",
                "parameter_mapping": {"object_type": "tool"},
                "duration": 3.0,
                "prerequisites": ["has_tool", "tool_applicable"]
            },
            
            # Communication actions
            "communicate": {
                "mesa_action": "send_message",
                "parameter_mapping": {"message_type": "general"},
                "duration": 1.0,
                "prerequisites": ["agents_nearby"]
            },
            "coordinate_with_team": {
                "mesa_action": "coordinate",
                "parameter_mapping": {"coordination_type": "strategy"},
                "duration": 2.0,
                "prerequisites": ["agents_nearby"]
            },
            "share_information": {
                "mesa_action": "share_info",
                "parameter_mapping": {"info_type": "discovery"},
                "duration": 1.0,
                "prerequisites": ["agents_nearby", "has_information"]
            },
            
            # Observation actions
            "observe": {
                "mesa_action": "observe_environment",
                "parameter_mapping": {},
                "duration": 1.0,
                "prerequisites": []
            },
            "observe_environment": {
                "mesa_action": "detailed_observation",
                "parameter_mapping": {"detail_level": "high"},
                "duration": 2.0,
                "prerequisites": []
            },
            
            # Wait/passive actions
            "wait": {
                "mesa_action": "wait",
                "parameter_mapping": {},
                "duration": 1.0,
                "prerequisites": []
            }
        }
    
    def _transform_parameters(self, crewai_params: Dict[str, Any],
                            mapping: Dict[str, str]) -> Dict[str, Any]:
        """Transform CrewAI parameters to Mesa parameters"""
        mesa_params = {}
        
        # Apply direct mappings
        for crewai_key, mesa_key in mapping.items():
            if crewai_key in crewai_params:
                mesa_params[mesa_key] = crewai_params[crewai_key]
            else:
                # Use mapping value as default if it's not a parameter reference
                mesa_params[mesa_key] = mesa_key
        
        # Pass through unmapped parameters
        for key, value in crewai_params.items():
            if key not in mapping:
                mesa_params[key] = value
        
        return mesa_params
    
    def _create_fallback_action(self, decision: DecisionData) -> MesaAction:
        """Create fallback action for unknown decisions"""
        return MesaAction(
            agent_id=decision.agent_id,
            action_type="wait",
            parameters={},
            expected_duration=1.0,
            prerequisites=[]
        )


class ActionValidator:
    """Validates Mesa actions against current model state"""
    
    def validate(self, action: MesaAction, mesa_model: mesa.Model) -> bool:
        """Validate action against model state"""
        
        # Find agent in model
        agent = self._find_agent(action.agent_id, mesa_model)
        if not agent:
            return False
        
        # Check prerequisites
        if not self._check_prerequisites(action, agent, mesa_model):
            return False
        
        # Validate specific action types
        if action.action_type == "move":
            return self._validate_movement(action, agent, mesa_model)
        elif action.action_type == "pick_up_object":
            return self._validate_pickup(action, agent, mesa_model)
        elif action.action_type == "use_object":
            return self._validate_object_use(action, agent, mesa_model)
        else:
            # Default validation for unknown actions
            return True
    
    def _find_agent(self, agent_id: str, mesa_model: mesa.Model) -> Optional[mesa.Agent]:
        """Find agent by ID in Mesa model"""
        for agent in mesa_model.schedule.agents:
            if hasattr(agent, 'agent_id') and agent.agent_id == agent_id:
                return agent
        return None
    
    def _check_prerequisites(self, action: MesaAction, agent: mesa.Agent,
                           mesa_model: mesa.Model) -> bool:
        """Check if action prerequisites are met"""
        for prereq in action.prerequisites:
            if not self._check_prerequisite(prereq, action, agent, mesa_model):
                return False
        return True
    
    def _check_prerequisite(self, prereq: str, action: MesaAction,
                          agent: mesa.Agent, mesa_model: mesa.Model) -> bool:
        """Check individual prerequisite"""
        if prereq == "can_move":
            return not getattr(agent, 'is_blocked', False)
        elif prereq == "has_energy":
            return getattr(agent, 'energy', 100) > 10
        elif prereq == "agents_nearby":
            return len(self._get_nearby_agents(agent, mesa_model)) > 0
        elif prereq == "object_nearby":
            return len(self._get_nearby_objects(agent, mesa_model)) > 0
        else:
            # Unknown prerequisite - assume met
            return True
    
    def _validate_movement(self, action: MesaAction, agent: mesa.Agent,
                         mesa_model: mesa.Model) -> bool:
        """Validate movement action"""
        current_pos = getattr(agent, 'pos', (0, 0))
        direction = action.parameters.get('direction', 'north')
        
        # Calculate target position
        target_pos = self._calculate_target_position(current_pos, direction)
        
        # Check if target position is valid
        return self._is_valid_position(target_pos, mesa_model)
    
    def _calculate_target_position(self, current_pos: tuple, direction: str) -> tuple:
        """Calculate target position based on direction"""
        x, y = current_pos
        if direction == "north":
            return (x, y + 1)
        elif direction == "south": 
            return (x, y - 1)
        elif direction == "east":
            return (x + 1, y)
        elif direction == "west":
            return (x - 1, y)
        else:
            return current_pos
    
    def _is_valid_position(self, pos: tuple, mesa_model: mesa.Model) -> bool:
        """Check if position is valid in model"""
        # Check bounds
        if hasattr(mesa_model, 'grid'):
            return mesa_model.grid.out_of_bounds(pos) == False
        return True  # Assume valid if no grid
```

## 2. Integration Test Examples

```python
"""
Integration tests demonstrating Mesa-CrewAI hybrid functionality
"""
import pytest
import asyncio
from unittest.mock import Mock, patch
from ..hybrid.core_architecture import HybridSimulationEngine, HybridAgent
from .concrete_implementations import (
    EscapeRoomPerceptionPipeline,
    CrewAIDecisionEngine, 
    EscapeRoomActionTranslator
)

class TestMesaCrewAIIntegration:
    """Integration tests for Mesa-CrewAI hybrid system"""
    
    @pytest.fixture
    def mock_mesa_model(self):
        """Create mock Mesa model for testing"""
        model = Mock()
        model.schedule = Mock()
        
        # Create mock agents
        agent1 = Mock()
        agent1.agent_id = "strategist"
        agent1.pos = (1, 1)
        agent1.energy = 100
        
        agent2 = Mock()
        agent2.agent_id = "mediator"
        agent2.pos = (2, 2)
        agent2.energy = 100
        
        agent3 = Mock()
        agent3.agent_id = "survivor"
        agent3.pos = (3, 3)
        agent3.energy = 100
        
        model.schedule.agents = [agent1, agent2, agent3]
        model.time_remaining = 3600
        
        return model
    
    @pytest.fixture
    def mock_crewai_agents(self):
        """Create mock CrewAI agents"""
        agents = []
        for role in ["Strategist", "Mediator", "Survivor"]:
            agent = Mock()
            agent.role = role
            agent.memory = {}
            agents.append(agent)
        return agents
    
    @pytest.fixture
    def hybrid_engine(self, mock_mesa_model, mock_crewai_agents):
        """Create hybrid simulation engine"""
        perception_pipeline = EscapeRoomPerceptionPipeline()
        decision_engine = CrewAIDecisionEngine(mock_crewai_agents)
        action_translator = EscapeRoomActionTranslator()
        state_synchronizer = Mock()
        
        engine = HybridSimulationEngine(
            mesa_model=mock_mesa_model,
            crewai_agents=mock_crewai_agents,
            perception_pipeline=perception_pipeline,
            decision_engine=decision_engine,
            action_translator=action_translator,
            state_synchronizer=state_synchronizer
        )
        
        return engine
    
    def test_perception_extraction(self, mock_mesa_model):
        """Test perception extraction from Mesa model"""
        pipeline = EscapeRoomPerceptionPipeline()
        
        perceptions = pipeline.extract_perceptions(mock_mesa_model)
        
        # Verify perceptions extracted for all agents
        assert "strategist" in perceptions
        assert "mediator" in perceptions
        assert "survivor" in perceptions
        
        # Verify perception structure
        strategist_perception = perceptions["strategist"]
        assert strategist_perception.agent_id == "strategist"
        assert "position" in strategist_perception.spatial_data
        assert "room_layout" in strategist_perception.environmental_state
        assert isinstance(strategist_perception.available_actions, list)
    
    @pytest.mark.asyncio
    async def test_decision_generation(self, mock_crewai_agents):
        """Test decision generation using CrewAI"""
        decision_engine = CrewAIDecisionEngine(mock_crewai_agents)
        
        # Create mock perceptions
        perceptions = {
            "strategist": Mock(
                agent_id="strategist",
                available_actions=["move_north", "observe", "communicate"],
                spatial_data={"position": (1, 1)},
                environmental_state={"time_remaining": 3600}
            )
        }
        
        with patch.object(decision_engine, '_execute_crew_reasoning', 
                         return_value="Move north to explore"):
            decisions = await decision_engine.reason_and_decide(perceptions)
        
        # Verify decisions generated
        assert "strategist" in decisions
        assert decisions["strategist"].agent_id == "strategist"
        assert decisions["strategist"].chosen_action in ["move_north", "observe", "communicate"]
    
    def test_action_translation(self):
        """Test translation of decisions to Mesa actions"""
        translator = EscapeRoomActionTranslator()
        
        from ..hybrid.core_architecture import DecisionData
        from datetime import datetime
        
        decision = DecisionData(
            agent_id="strategist",
            timestamp=datetime.now(),
            chosen_action="move_north",
            action_parameters={"speed": "normal"},
            reasoning="Moving north to explore",
            confidence_level=0.8,
            fallback_actions=["wait", "observe"]
        )
        
        mesa_action = translator.translate_decision(decision)
        
        # Verify translation
        assert mesa_action.agent_id == "strategist"
        assert mesa_action.action_type == "move"
        assert "direction" in mesa_action.parameters
        assert mesa_action.parameters["direction"] == "north"
    
    def test_action_validation(self, mock_mesa_model):
        """Test validation of Mesa actions"""
        translator = EscapeRoomActionTranslator()
        
        from ..hybrid.core_architecture import MesaAction
        
        valid_action = MesaAction(
            agent_id="strategist",
            action_type="move",
            parameters={"direction": "north"},
            expected_duration=1.0,
            prerequisites=["can_move"]
        )
        
        # Mock agent has movement capability
        mock_mesa_model.schedule.agents[0].is_blocked = False
        
        is_valid = translator.validate_action(valid_action, mock_mesa_model)
        assert is_valid == True
    
    @pytest.mark.asyncio
    async def test_full_simulation_step(self, hybrid_engine):
        """Test complete hybrid simulation step"""
        hybrid_engine.initialize()
        
        with patch.object(hybrid_engine.decision_engine, '_execute_crew_reasoning',
                         return_value="Coordinate team movement north"):
            result = await hybrid_engine.step()
        
        # Verify step completed successfully
        assert "step" in result
        assert "duration" in result
        assert "actions_executed" in result
        assert hybrid_engine.step_count == 1
    
    def test_error_handling_and_recovery(self, hybrid_engine):
        """Test error handling and recovery mechanisms"""
        hybrid_engine.initialize()
        
        # Simulate decision engine failure
        with patch.object(hybrid_engine.decision_engine, 'reason_and_decide',
                         side_effect=Exception("Decision engine failed")):
            
            with pytest.raises(RuntimeError, match="Step .* failed"):
                asyncio.run(hybrid_engine.step())
        
        # Verify error state
        assert hybrid_engine.error_count == 1
        assert hybrid_engine.state.value == "error"
    
    def test_performance_monitoring(self, hybrid_engine):
        """Test performance monitoring and metrics"""
        hybrid_engine.initialize()
        
        # Run multiple steps and monitor performance
        async def run_steps():
            for i in range(3):
                with patch.object(hybrid_engine.decision_engine, '_execute_crew_reasoning',
                                return_value=f"Step {i} action"):
                    await hybrid_engine.step()
        
        asyncio.run(run_steps())
        
        # Verify performance tracking
        assert len(hybrid_engine.performance_history) == 3
        assert all("step_duration" in entry["metrics"] 
                  for entry in hybrid_engine.performance_history)
        
        # Test performance summary
        summary = hybrid_engine._get_performance_summary()
        assert "avg_step_duration" in summary
        assert "total_steps" in summary
        assert summary["total_steps"] == 3
```

This implementation provides concrete, testable components that development teams can immediately use to build Mesa-CrewAI hybrid simulations. The architecture maintains clean separation of concerns while enabling rich integration between spatial simulation and cognitive reasoning capabilities.