# Mesa-Based Architecture Refactor Requirements

## Project Context

The current CrewAI escape room simulation suffers from over-engineered competitive mechanics that constrain natural LLM reasoning. This document defines requirements for a complete architectural refactor using Mesa for environment/spatial modeling and CrewAI for agent reasoning.

## Executive Summary

**Goal**: Create a hybrid Mesa-CrewAI architecture where:
- Mesa handles the physical environment, spatial dynamics, and simulation framework
- CrewAI handles agent reasoning, decision-making, and natural language interactions
- Agents make emergent decisions based on LLM reasoning rather than hardcoded behaviors
- Clean slate approach with 100% test coverage and TDD methodology

## Problem Statement

### Current System Issues
1. **Manual Action Simulation**: Agents follow hardcoded action patterns with explicit action types ("claim_resource", "share_information") rather than natural decision-making
2. **Rigid Behavioral Coding**: Personality profiles with fixed numerical parameters (e.g., `analytical_tendency=0.9`) instead of emergent LLM-driven personalities
3. **Competition Loop Traps**: Complex resource/escape mechanics cause agents to get stuck in predictable competition patterns
4. **Artificial Escape Constraints**: Agents rarely attempt escapes due to rigid probability calculations and requirement checking
5. **Over-engineered Subsystems**: Trust tracking, moral dilemmas, and resource management create artificial constraints on natural reasoning

---

## 1. Functional Requirements

### 1.1 Environment Modeling (Mesa)

**FR-ENV-001**: The system SHALL use Mesa to model the physical escape room environment
- Grid-based spatial representation (configurable size)
- Object placement and interaction zones
- Physical constraints (walls, doors, obstacles)
- Line-of-sight calculations for agent awareness

**FR-ENV-002**: The system SHALL support dynamic environment states
- Doors can be locked/unlocked
- Objects can be moved, combined, or transformed
- Environmental hazards can emerge over time
- Light levels affect visibility

**FR-ENV-003**: The system SHALL track spatial relationships
- Agent positions and movement
- Object locations and accessibility
- Distance calculations for interaction range
- Path finding for movement planning

### 1.2 Agent Architecture (Hybrid Mesa-CrewAI)

**FR-AGENT-001**: Agents SHALL be Mesa agents with CrewAI reasoning capabilities
- Mesa agent base class for physical presence and movement
- CrewAI agent wrapper for decision-making and reasoning
- Clean separation between physical actions and cognitive processes

**FR-AGENT-002**: Agents SHALL perceive their environment naturally
- Limited field of view based on position and orientation
- Can only interact with objects within range
- Must discover information through exploration
- No omniscient knowledge of room state

**FR-AGENT-003**: Agent decisions SHALL emerge from LLM reasoning
- No hardcoded personality parameters
- No predetermined action probabilities
- Decisions based on situation assessment and goals
- Natural language reasoning for all choices

### 1.3 Simulation Engine

**FR-SIM-001**: The simulation SHALL use Mesa's scheduler for turn management
- Configurable activation order (random, ordered, staged)
- Fair turn distribution among agents
- Time step tracking for simulation progress

**FR-SIM-002**: The simulation SHALL support multiple run modes
- Step-by-step execution with inspection
- Continuous run until completion
- Checkpoint and resume functionality
- Replay from saved states

**FR-SIM-003**: The simulation SHALL enforce physical constraints
- Agents cannot occupy same space
- Objects have realistic interaction requirements
- Movement follows pathfinding rules
- Actions have spatial prerequisites

### 1.4 Agent Communication

**FR-COMM-001**: Agents SHALL communicate through natural language
- Direct speech when in proximity
- No artificial information sharing mechanics
- Communication range limits based on distance
- Potential for miscommunication or deception

**FR-COMM-002**: Information SHALL spread naturally
- Agents must explicitly share discoveries
- No automatic knowledge synchronization
- Memory of communications for decision context
- Trust emerges from interaction history

### 1.5 Escape Mechanics

**FR-ESC-001**: Escape methods SHALL be discovered through exploration
- Multiple potential escape routes
- Requirements emerge from environment inspection
- No predefined escape "recipes"
- Creative solutions possible

**FR-ESC-002**: The single-survivor constraint SHALL create natural tension
- Agents aware of the constraint through initial briefing
- No mechanical enforcement of competition
- Cooperation/betrayal emerges from agent reasoning
- Moral dilemmas arise naturally from the situation

---

## 2. Non-Functional Requirements

### 2.1 Performance

**NFR-PERF-001**: Simulation step execution < 2 seconds
- Mesa step calculation < 100ms
- CrewAI reasoning call < 1.5s per agent
- Parallel agent reasoning where possible
- Efficient spatial calculations

**NFR-PERF-002**: Memory usage SHALL scale linearly
- O(n) memory for n agents
- Configurable history retention
- Automatic pruning of old states
- Lazy loading of agent memories

### 2.2 Testability

**NFR-TEST-001**: 100% test coverage required
- Unit tests for all Mesa components
- Integration tests for Mesa-CrewAI interaction
- Mock LLM responses for deterministic testing
- Property-based testing for edge cases

**NFR-TEST-002**: TDD methodology enforced
- Tests written before implementation
- Red-green-refactor cycle
- Behavior-driven test descriptions
- Automated test execution in CI/CD

### 2.3 Maintainability

**NFR-MAINT-001**: Clean architecture separation
- Mesa components in dedicated modules
- CrewAI logic isolated from environment
- Clear interfaces between systems
- Dependency injection for flexibility

**NFR-MAINT-002**: Comprehensive documentation
- API documentation for all public methods
- Architecture decision records (ADRs)
- Example notebooks for common scenarios
- Performance profiling guides

### 2.4 Extensibility

**NFR-EXT-001**: Pluggable environment components
- Easy addition of new object types
- Configurable room layouts
- Custom interaction mechanics
- Environmental effect plugins

**NFR-EXT-002**: Flexible agent configuration
- Support for different LLM backends
- Adjustable agent capabilities
- Custom goal definitions
- Personality through prompting only

---

## 3. Agent Behavior Requirements

### 3.1 Natural Decision Making

**ABR-DEC-001**: Agents SHALL reason about actions in natural language
- "I see a key on the table. I should pick it up."
- "The door is locked. I need to find a way to open it."
- "Alex seems nervous. Maybe they know something."
- No numerical calculations or probabilities

**ABR-DEC-002**: Personality SHALL emerge from consistent reasoning
- Strategist: Analytical observations and planning
- Mediator: Focus on group dynamics and fairness
- Survivor: Pragmatic self-preservation instincts
- Expressed through language, not parameters

### 3.2 Emergent Behaviors

**ABR-EMER-001**: Cooperation SHALL arise from mutual benefit
- Agents recognize when they need help
- Trust builds through successful interactions
- Betrayal possible but has social consequences
- No mechanical trust scores

**ABR-EMER-002**: Competition SHALL emerge from scarcity
- Limited resources create natural conflict
- Time pressure increases urgency
- Single-survivor rule creates ultimate tension
- Agents must balance cooperation and self-interest

### 3.3 Learning and Adaptation

**ABR-LEARN-001**: Agents SHALL remember past interactions
- Who helped or hindered them
- What strategies worked or failed
- Where they found useful items
- Natural language memory storage

**ABR-LEARN-002**: Behavior SHALL adapt based on experience
- Failed attempts inform new strategies
- Successful patterns may be repeated
- Social dynamics evolve over time
- No explicit learning algorithms

---

## 4. Environment Requirements

### 4.1 Spatial Modeling

**ENV-SPACE-001**: Grid-based room representation
- Minimum 10x10 grid for meaningful exploration
- Each cell can contain agents, objects, or obstacles
- Clear movement rules (4 or 8 directional)
- Visualization support for debugging

**ENV-SPACE-002**: Rich object interactions
- Objects have properties (weight, size, function)
- Can be examined, picked up, used, combined
- State changes affect functionality
- Spatial requirements for interaction

### 4.2 Physics and Constraints

**ENV-PHYS-001**: Realistic movement constraints
- Agents move at walking speed
- Cannot pass through walls or locked doors
- Must navigate around obstacles
- Line of sight affects awareness

**ENV-PHYS-002**: Object manipulation rules
- Weight limits for carrying
- Size constraints for inventory
- Two-handed operations require dropping items
- Combination requires proximity

### 4.3 Environmental Dynamics

**ENV-DYN-001**: Time-based changes
- Decreasing oxygen or increasing danger
- Timed locks or mechanisms
- Degrading conditions affect visibility
- Urgency increases naturally

**ENV-DYN-002**: Interactive cause and effect
- Switches affect multiple objects
- Puzzles have logical solutions
- Actions have consequences
- Environment responds to agent behavior

---

## 5. Integration Requirements

### 5.1 Mesa-CrewAI Bridge

**INT-BRIDGE-001**: Clean action interface
```python
# Mesa provides available actions
available_actions = mesa_agent.get_available_actions()

# CrewAI decides what to do
chosen_action = crewai_agent.decide(available_actions, context)

# Mesa executes the action
result = mesa_agent.execute(chosen_action)
```

**INT-BRIDGE-002**: Perception to reasoning pipeline
- Mesa provides sensory information
- Formatted as natural language descriptions
- CrewAI processes and reasons about it
- Decisions flow back to Mesa for execution

### 5.2 State Synchronization

**INT-STATE-001**: Consistent world state
- Single source of truth in Mesa model
- Agents have partial, perception-based views
- No state duplication between systems
- Clear update propagation

**INT-STATE-002**: Memory integration
- Short-term memory in CrewAI context
- Long-term memory persisted appropriately
- Spatial memory linked to Mesa positions
- Social memory from CrewAI interactions

### 5.3 Communication Protocol

**INT-COMM-001**: Message passing interface
- Structured format for agent communications
- Range-based delivery in Mesa
- Content processing in CrewAI
- Response generation pipeline

**INT-COMM-002**: Action result feedback
- Mesa provides action outcomes
- Success/failure with explanations
- Environmental changes described
- CrewAI incorporates into reasoning

---

## 6. Testing Requirements

### 6.1 Unit Testing

**TEST-UNIT-001**: Mesa component isolation
- Test movement without full simulation
- Test object interactions independently
- Test perception calculations
- Mock agent decisions

**TEST-UNIT-002**: CrewAI reasoning validation
- Test decision making with mock perceptions
- Verify personality consistency
- Test memory formation and retrieval
- Mock LLM responses for determinism

### 6.2 Integration Testing

**TEST-INT-001**: Full cycle validation
- Perception → Reasoning → Action → Result
- Multi-agent interaction scenarios
- Communication flow testing
- State consistency verification

**TEST-INT-002**: Edge case coverage
- Simultaneous actions on same object
- Communication at range boundaries  
- Resource contention scenarios
- Escape attempt conflicts

### 6.3 System Testing

**TEST-SYS-001**: End-to-end scenarios
- Complete escape room runs
- Various difficulty configurations
- Different agent combinations
- Performance benchmarks

**TEST-SYS-002**: Emergent behavior validation
- Cooperation emergence testing
- Betrayal scenario testing
- Natural language coherence
- Goal achievement rates

---

## 7. Scope Definition

### 7.1 In Scope

1. **Core Architecture**
   - Mesa-based environment modeling
   - CrewAI agent reasoning system
   - Integration bridge between frameworks
   - Basic escape room mechanics

2. **Agent Capabilities**
   - Movement and navigation
   - Object interaction
   - Natural language communication
   - Memory and learning

3. **Simulation Features**
   - Turn-based execution
   - Spatial constraints
   - Time pressure mechanics
   - Single-survivor rule

4. **Testing Infrastructure**
   - Unit test framework
   - Integration test suite
   - Performance benchmarks
   - Mock LLM system

### 7.2 Out of Scope

1. **Advanced Features**
   - Real-time visualization UI
   - Multiplayer human participation
   - Voice-based communication
   - AR/VR integration

2. **Complex Mechanics**  
   - Physics simulation beyond basic constraints
   - Detailed health/stamina systems
   - Complex crafting mechanics
   - Procedural room generation

3. **External Integrations**
   - Database persistence (file-based only)
   - Web API endpoints
   - Cloud deployment
   - Analytics dashboards

4. **Backwards Compatibility**
   - Migration from old system
   - Data format conversion
   - Legacy API support
   - Previous simulation compatibility

---

## 8. Success Criteria

### 8.1 Technical Success

- ✓ Mesa and CrewAI successfully integrated
- ✓ Agents make natural language decisions
- ✓ No hardcoded behavioral parameters
- ✓ 100% test coverage achieved
- ✓ Performance meets requirements

### 8.2 Behavioral Success  

- ✓ Emergent cooperation observed
- ✓ Natural competition arises
- ✓ Agents attempt escapes organically
- ✓ Personality expressed through language
- ✓ No artificial behavior loops

### 8.3 Quality Success

- ✓ Clean architecture separation
- ✓ Comprehensive documentation
- ✓ All tests passing
- ✓ Code review approved
- ✓ Performance benchmarks met

---

## 9. Risk Mitigation

### 9.1 Technical Risks

**Risk**: LLM latency affects simulation performance
- **Mitigation**: Implement async reasoning with timeouts
- **Fallback**: Basic rule-based actions if LLM fails

**Risk**: Mesa-CrewAI integration complexity
- **Mitigation**: Clear interface definitions and mocks
- **Fallback**: Modular design allows framework swapping

### 9.2 Behavioral Risks

**Risk**: Agents may not naturally compete
- **Mitigation**: Clear initial prompts about survival constraint
- **Fallback**: Environmental pressure to encourage action

**Risk**: Emergent behaviors unpredictable
- **Mitigation**: Extensive testing of various scenarios
- **Fallback**: Adjustable prompts to guide behavior

---

## 10. Implementation Priority

### Phase 1: Foundation (Week 1)
1. Mesa environment setup
2. Basic agent movement
3. Object interaction system
4. CrewAI integration bridge

### Phase 2: Core Mechanics (Week 2)
1. Perception system
2. Natural language decisions
3. Communication framework
4. Basic escape mechanics

### Phase 3: Emergence (Week 3)
1. Memory systems
2. Time pressure mechanics
3. Complete scenarios
4. Behavioral validation

### Phase 4: Quality (Week 4)
1. Full test coverage
2. Performance optimization
3. Documentation completion
4. Final integration testing