# Mesa-CrewAI Hybrid Development Quickstart

## Quick Implementation Guide

This guide provides everything needed to start building Mesa-CrewAI hybrid simulations immediately.

## 1. Project Setup (5 minutes)

### Install Dependencies
```bash
# Core frameworks
pip install mesa>=2.0.0
pip install crewai>=0.15.0

# Supporting libraries
pip install numpy>=1.26.0
pip install pandas>=2.0.0
pip install asyncio
pip install pytest>=8.0.0

# Optional: Visualization
pip install matplotlib
pip install networkx>=3.0
```

### Directory Structure
```
your_project/
├── src/
│   ├── hybrid/
│   │   ├── __init__.py
│   │   ├── core_architecture.py      # From docs/architecture/
│   │   ├── perception_pipeline.py
│   │   ├── decision_engine.py
│   │   ├── action_translator.py
│   │   └── state_synchronizer.py
│   ├── mesa_models/
│   │   ├── __init__.py
│   │   └── escape_room_model.py
│   └── crewai_agents/
│       ├── __init__.py
│       └── agent_definitions.py
├── tests/
│   ├── unit/
│   └── integration/
└── examples/
    └── basic_escape_room.py
```

## 2. Minimal Working Example (15 minutes)

### Step 1: Create Basic Mesa Model
```python
# src/mesa_models/escape_room_model.py
import mesa
from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa.space import MultiGrid

class EscapeRoomAgent(Agent):
    """Mesa agent with hybrid capabilities"""
    
    def __init__(self, unique_id, model, agent_id):
        super().__init__(unique_id, model)
        self.agent_id = agent_id
        self.energy = 100
        self.inventory = []
        self.is_blocked = False
    
    def step(self):
        # Mesa step - will be coordinated by hybrid engine
        pass
    
    def execute_action(self, action_type, parameters):
        """Execute action from hybrid engine"""
        if action_type == "move":
            self._move(parameters.get("direction"))
        elif action_type == "pick_up_object":
            self._pick_up(parameters.get("object_type"))
    
    def _move(self, direction):
        """Handle movement"""
        if self.is_blocked:
            return
        
        current_pos = self.pos
        if direction == "north":
            new_pos = (current_pos[0], current_pos[1] + 1)
        elif direction == "south":
            new_pos = (current_pos[0], current_pos[1] - 1)
        elif direction == "east":
            new_pos = (current_pos[0] + 1, current_pos[1])
        elif direction == "west":
            new_pos = (current_pos[0] - 1, current_pos[1])
        else:
            return
        
        if not self.model.grid.out_of_bounds(new_pos):
            self.model.grid.move_agent(self, new_pos)

class EscapeRoomModel(Model):
    """Mesa model for escape room simulation"""
    
    def __init__(self, width=10, height=10, num_agents=3):
        super().__init__()
        self.grid = MultiGrid(width, height, True)
        self.schedule = RandomActivation(self)
        self.time_remaining = 3600
        
        # Create agents
        agent_ids = ["strategist", "mediator", "survivor"]
        for i in range(num_agents):
            agent = EscapeRoomAgent(i, self, agent_ids[i])
            self.schedule.add(agent)
            
            # Place agent randomly
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            self.grid.place_agent(agent, (x, y))
    
    def step(self):
        """Mesa model step"""
        self.schedule.step()
        self.time_remaining -= 1
```

### Step 2: Create Basic CrewAI Agents
```python
# src/crewai_agents/agent_definitions.py
from crewai import Agent

def create_hybrid_strategist():
    """Create strategist agent for hybrid simulation"""
    return Agent(
        role="Strategist",
        goal="Analyze the escape room systematically and develop optimal escape strategies",
        backstory="""You are a brilliant analytical mind who excels at breaking down complex 
        problems into manageable components. You see patterns others miss and can quickly 
        assess risks and opportunities.""",
        verbose=True,
        allow_delegation=False,
        memory=True
    )

def create_hybrid_mediator():
    """Create mediator agent for hybrid simulation"""
    return Agent(
        role="Mediator", 
        goal="Facilitate team coordination and maintain group cohesion under pressure",
        backstory="""You are a natural diplomat and team facilitator who excels at reading 
        social dynamics and bringing out the best in others. You can resolve conflicts and 
        build consensus even in high-stress situations.""",
        verbose=True,
        allow_delegation=False,
        memory=True
    )

def create_hybrid_survivor():
    """Create survivor agent for hybrid simulation"""
    return Agent(
        role="Survivor",
        goal="Execute strategies decisively and make critical survival decisions",
        backstory="""You are a pragmatic decision-maker who thrives under pressure. When others 
        hesitate, you act. You have strong survival instincts and aren't afraid to make tough 
        choices when lives are on the line.""",
        verbose=True,
        allow_delegation=False,
        memory=True
    )
```

### Step 3: Basic Integration Example
```python
# examples/basic_escape_room.py
import asyncio
from src.hybrid.core_architecture import HybridSimulationEngine, HybridSimulationFactory
from src.mesa_models.escape_room_model import EscapeRoomModel
from src.crewai_agents.agent_definitions import (
    create_hybrid_strategist, create_hybrid_mediator, create_hybrid_survivor
)

async def run_basic_hybrid_simulation():
    """Run basic Mesa-CrewAI hybrid simulation"""
    
    # Create Mesa model
    mesa_model = EscapeRoomModel(width=10, height=10, num_agents=3)
    
    # Create CrewAI agents
    crewai_agents = [
        create_hybrid_strategist(),
        create_hybrid_mediator(), 
        create_hybrid_survivor()
    ]
    
    # Create hybrid simulation (simplified version)
    # Note: In production, use HybridSimulationFactory.create_escape_room_simulation()
    
    print("🚀 Starting Mesa-CrewAI Hybrid Simulation")
    print(f"📊 Mesa Model: {len(mesa_model.schedule.agents)} agents in {mesa_model.grid.width}x{mesa_model.grid.height} grid")
    print(f"🤖 CrewAI Agents: {len(crewai_agents)} reasoning agents")
    
    # Run simplified hybrid steps
    for step in range(5):
        print(f"\n{'='*50}")
        print(f"HYBRID STEP {step + 1}")
        print(f"{'='*50}")
        
        # 1. Mesa physics step
        print("🔄 Mesa: Updating environment and agent positions")
        mesa_model.step()
        
        # 2. Extract perceptions (simplified)
        print("👁️  Extracting spatial perceptions from Mesa")
        perceptions = extract_simple_perceptions(mesa_model)
        
        # 3. CrewAI reasoning (simplified)
        print("🧠 CrewAI: Agents analyzing situation and making decisions")
        decisions = await simple_crewai_reasoning(crewai_agents, perceptions)
        
        # 4. Apply decisions to Mesa (simplified)
        print("⚡ Applying decisions to Mesa model")
        apply_simple_decisions(decisions, mesa_model)
        
        print(f"✅ Step {step + 1} completed")
        print(f"⏰ Time remaining: {mesa_model.time_remaining}")
        
        # Simple termination condition
        if mesa_model.time_remaining <= 0:
            print("⏰ Time expired!")
            break
    
    print("\n🎯 Hybrid simulation completed!")

def extract_simple_perceptions(mesa_model):
    """Simplified perception extraction"""
    perceptions = {}
    
    for agent in mesa_model.schedule.agents:
        perceptions[agent.agent_id] = {
            "position": agent.pos,
            "energy": agent.energy,
            "time_remaining": mesa_model.time_remaining,
            "nearby_agents": len([a for a in mesa_model.schedule.agents if a != agent])
        }
    
    return perceptions

async def simple_crewai_reasoning(agents, perceptions):
    """Simplified CrewAI reasoning"""
    decisions = {}
    
    # Simple decision logic
    for agent in agents:
        agent_id = agent.role.lower()
        
        if agent_id == "strategist":
            decisions[agent_id] = "analyze_environment"
        elif agent_id == "mediator":
            decisions[agent_id] = "coordinate_team"
        elif agent_id == "survivor":
            decisions[agent_id] = "move_towards_exit"
    
    return decisions

def apply_simple_decisions(decisions, mesa_model):
    """Apply decisions to Mesa model"""
    for mesa_agent in mesa_model.schedule.agents:
        decision = decisions.get(mesa_agent.agent_id)
        
        if decision == "move_towards_exit":
            # Simple movement logic
            mesa_agent.execute_action("move", {"direction": "north"})
        elif decision == "analyze_environment":
            # Strategist observes
            mesa_agent.energy -= 5  # Thinking costs energy
        elif decision == "coordinate_team":
            # Mediator facilitates
            pass  # No direct Mesa action needed

if __name__ == "__main__":
    asyncio.run(run_basic_hybrid_simulation())
```

## 3. Full Implementation Checklist

### Phase 1: Core Integration ✅
- [ ] Copy `core_architecture.py` from docs to your project
- [ ] Implement `EscapeRoomPerceptionPipeline` 
- [ ] Implement `CrewAIDecisionEngine`
- [ ] Implement `EscapeRoomActionTranslator`
- [ ] Create basic Mesa model with hybrid capabilities
- [ ] Test basic perception → reasoning → action flow

### Phase 2: Enhanced Features
- [ ] Implement `UnifiedStateSynchronizer` for state management
- [ ] Add error handling and recovery mechanisms
- [ ] Implement performance monitoring
- [ ] Add async processing for better performance
- [ ] Create comprehensive test suite

### Phase 3: Production Features
- [ ] Add configuration management
- [ ] Implement logging and debugging tools
- [ ] Create visualization components
- [ ] Add simulation result analysis
- [ ] Build deployment and scaling capabilities

## 4. Common Patterns and Best Practices

### Pattern 1: Perception Filtering by Agent Role
```python
def filter_perceptions_by_role(perception: PerceptionData, agent_role: str) -> PerceptionData:
    """Filter perceptions based on agent specialization"""
    
    if "strategist" in agent_role.lower():
        # Strategic agents focus on spatial patterns and resource optimization
        return PerceptionData(
            agent_id=perception.agent_id,
            timestamp=perception.timestamp,
            spatial_data=perception.spatial_data,  # Full spatial awareness
            environmental_state=perception.environmental_state,
            nearby_agents=perception.nearby_agents,
            available_actions=[a for a in perception.available_actions if "analyze" in a or "plan" in a],
            resources=perception.resources,  # Full resource awareness
            constraints=perception.constraints
        )
    
    elif "mediator" in agent_role.lower():
        # Mediators focus on social dynamics and communication
        return PerceptionData(
            agent_id=perception.agent_id,
            timestamp=perception.timestamp,
            spatial_data={"position": perception.spatial_data.get("position")},  # Minimal spatial
            environmental_state={"time_remaining": perception.environmental_state.get("time_remaining")},
            nearby_agents=perception.nearby_agents,  # Full social awareness
            available_actions=[a for a in perception.available_actions if "communicate" in a or "coordinate" in a],
            resources={"energy": perception.resources.get("energy")},
            constraints=perception.constraints
        )
    
    elif "survivor" in agent_role.lower():
        # Survivors focus on immediate threats and action opportunities  
        return PerceptionData(
            agent_id=perception.agent_id,
            timestamp=perception.timestamp,
            spatial_data=perception.spatial_data,
            environmental_state=perception.environmental_state,
            nearby_agents=perception.nearby_agents,
            available_actions=[a for a in perception.available_actions if "move" in a or "use" in a or "execute" in a],
            resources=perception.resources,
            constraints=perception.constraints  # Full constraint awareness
        )
    
    return perception  # Default: no filtering
```

### Pattern 2: Action Validation with Fallbacks
```python
def validate_with_fallbacks(action: MesaAction, mesa_model: mesa.Model, 
                          fallback_actions: List[str]) -> MesaAction:
    """Validate action and use fallbacks if invalid"""
    
    validator = ActionValidator()
    
    # Try primary action
    if validator.validate(action, mesa_model):
        return action
    
    # Try fallback actions
    for fallback_name in fallback_actions:
        fallback_action = create_fallback_action(action.agent_id, fallback_name)
        if validator.validate(fallback_action, mesa_model):
            return fallback_action
    
    # Ultimate fallback: wait
    return MesaAction(
        agent_id=action.agent_id,
        action_type="wait",
        parameters={},
        expected_duration=1.0,
        prerequisites=[]
    )
```

### Pattern 3: Performance Monitoring
```python
def monitor_step_performance(func):
    """Decorator to monitor hybrid step performance"""
    
    def wrapper(self, *args, **kwargs):
        start_time = time.time()
        
        try:
            result = func(self, *args, **kwargs)
            
            # Record success metrics
            duration = time.time() - start_time
            self.performance_metrics.append({
                "step": self.step_count,
                "duration": duration,
                "status": "success",
                "actions_executed": result.get("actions_executed", 0)
            })
            
            return result
            
        except Exception as e:
            # Record failure metrics
            duration = time.time() - start_time
            self.performance_metrics.append({
                "step": self.step_count,
                "duration": duration,
                "status": "error",
                "error": str(e)
            })
            raise
    
    return wrapper
```

## 5. Debugging and Testing Tips

### Enable Detailed Logging
```python
import logging

# Configure hybrid simulation logging
logging.basicConfig(level=logging.DEBUG)
hybrid_logger = logging.getLogger("hybrid_simulation")

# In your hybrid engine:
def step(self):
    hybrid_logger.info(f"Starting step {self.step_count}")
    
    # Log perception extraction
    perceptions = self.perception_pipeline.extract_perceptions(self.mesa_model)
    hybrid_logger.debug(f"Extracted {len(perceptions)} perceptions")
    
    # Log decision making
    decisions = await self.decision_engine.reason_and_decide(perceptions)
    hybrid_logger.debug(f"Generated {len(decisions)} decisions")
    
    # Log action execution
    hybrid_logger.info(f"Executing {len(decisions)} actions")
```

### Unit Test Template
```python
def test_perception_pipeline():
    """Template for testing perception pipeline"""
    
    # Create mock Mesa model
    mesa_model = create_mock_mesa_model()
    
    # Create perception pipeline
    pipeline = EscapeRoomPerceptionPipeline()
    
    # Extract perceptions
    perceptions = pipeline.extract_perceptions(mesa_model)
    
    # Verify structure
    assert len(perceptions) > 0
    for agent_id, perception in perceptions.items():
        assert isinstance(perception, PerceptionData)
        assert perception.agent_id == agent_id
        assert "position" in perception.spatial_data
        assert len(perception.available_actions) > 0
```

## 6. Next Steps

1. **Start with the basic example** - Get the minimal version running first
2. **Add one component at a time** - Don't try to implement everything at once
3. **Test extensively** - Each component should have unit tests
4. **Monitor performance** - Track step duration and memory usage
5. **Scale gradually** - Start with 3 agents, then expand

The architecture is designed to be **iterative and extensible** - you can start simple and add sophistication over time while maintaining clean interfaces between Mesa and CrewAI components.

## 7. Support and Resources

- **Architecture Diagrams**: `docs/architecture/mesa_crewai_hybrid_architecture.md`
- **Implementation Examples**: `docs/architecture/implementation_examples.md`  
- **Mesa Documentation**: https://mesa.readthedocs.io/
- **CrewAI Documentation**: https://github.com/joaomdmoura/crewAI

Start building and iterate based on your specific simulation requirements!