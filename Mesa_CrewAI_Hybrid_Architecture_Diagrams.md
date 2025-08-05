# Mesa-CrewAI Hybrid Architecture: Data Flow and Sequence Diagrams

## Overview

This document provides detailed data flow and sequence diagrams for the Mesa-CrewAI hybrid architecture, showing exactly how data flows through the system and how the frameworks interact step-by-step.

## 1. Complete Data Flow Diagram

### Perception → Reasoning → Action Pipeline

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           MESA-CREWAI HYBRID DATA FLOW                         │
└─────────────────────────────────────────────────────────────────────────────────┘

┌──────────────┐    ┌─────────────────┐    ┌──────────────────┐    ┌─────────────┐
│   Mesa       │    │   Perception    │    │    CrewAI        │    │   Mesa      │
│ Environment  │    │   Pipeline      │    │   Reasoning      │    │  Actions    │
│              │    │                 │    │                  │    │             │
│ ┌──────────┐ │    │ ┌─────────────┐ │    │ ┌──────────────┐ │    │ ┌─────────┐ │
│ │Agents    │ │────▶│ │Extract      │ │────▶│ │LLM Prompt    │ │────▶│ │Execute  │ │
│ │Positions │ │    │ │Spatial Data │ │    │ │Generation    │ │    │ │Mesa     │ │
│ └──────────┘ │    │ └─────────────┘ │    │ └──────────────┘ │    │ │Actions  │ │
│              │    │                 │    │                  │    │ └─────────┘ │
│ ┌──────────┐ │    │ ┌─────────────┐ │    │ ┌──────────────┐ │    │             │
│ │Resources │ │────▶│ │Extract      │ │────▶│ │Agent Memory  │ │    │ ┌─────────┐ │
│ │Available │ │    │ │Resource     │ │    │ │Integration   │ │    │ │Update   │ │
│ └──────────┘ │    │ │Data         │ │    │ └──────────────┘ │    │ │State    │ │
│              │    │ └─────────────┘ │    │                  │    │ └─────────┘ │
│ ┌──────────┐ │    │                 │    │ ┌──────────────┐ │    │             │
│ │Social    │ │────▶│ ┌─────────────┐ │────▶│ │Decision      │ │    └─────────────┘
│ │Relations │ │    │ │Extract      │ │    │ │Generation    │ │           │
│ └──────────┘ │    │ │Social Data  │ │    │ └──────────────┘ │           │
│              │    │ └─────────────┘ │    │                  │           │
│ ┌──────────┐ │    │                 │    │ ┌──────────────┐ │           │
│ │Hazards/  │ │────▶│ ┌─────────────┐ │────▶│ │Action        │ │           │
│ │Threats   │ │    │ │Extract      │ │    │ │Translation   │ │           │
│ └──────────┘ │    │ │Environmental│ │    │ └──────────────┘ │           │
│              │    │ │Data         │ │    │                  │           │
└──────────────┘    │ └─────────────┘ │    └──────────────────┘           │
                    │                 │                                   │
                    │ ┌─────────────┐ │    ┌──────────────────┐           │
                    │ │Natural      │ │────▶│    Unified       │◀──────────┘
                    │ │Language     │ │    │  State Manager   │
                    │ │Formatting   │ │    │                  │
                    │ └─────────────┘ │    │ ┌──────────────┐ │
                    └─────────────────┘    │ │State Sync   │ │
                                          │ │& Validation │ │
                                          │ └──────────────┘ │
                                          └──────────────────┘
```

### Key Data Transformations:

1. **Mesa State → PerceptionData**
   ```
   Mesa Agent Position (x, y) → SpatialPerception {
     current_position: (x, y),
     visible_area: [(x±range, y±range)],
     movement_options: [valid_positions],
     obstacles: [blocked_positions]
   }
   ```

2. **PerceptionData → Natural Language**
   ```
   SpatialPerception → "You are at position (5, 3). You can move north, east, 
   or south. There are obstacles blocking the west path."
   ```

3. **CrewAI Decision → MesaAction**
   ```
   DecisionData {
     chosen_action: "move_north",
     parameters: {target: (5, 4)}
   } → MesaAction {
     action_type: "move",
     parameters: {target_position: (5, 4)}
   }
   ```

## 2. Simulation Step Execution Sequence

### Main Simulation Loop

```plantuml
@startuml
participant "HybridSimulationEngine" as Engine
participant "Mesa Model" as Mesa
participant "PerceptionPipeline" as Perception
participant "DecisionEngine" as Decision
participant "ActionTranslator" as Translator
participant "StateSynchronizer" as Sync
participant "ErrorManager" as Error
participant "PerformanceMonitor" as Perf

note over Engine: step() called
Engine -> Perf: record_operation_start("simulation_step")

== Mesa Environment Update ==
Engine -> Mesa: model.step()
note over Mesa: Advance environment\nUpdate agent positions\nProcess physics/rules

== Perception Extraction ==
Engine -> Perception: extract_perceptions(mesa_model)
Perception -> Mesa: get_agent_positions()
Perception -> Mesa: get_resource_states()
Perception -> Mesa: get_environmental_data()
Perception -> Engine: Dict[agent_id, PerceptionData]
note over Perception: Cache perceptions\nfor performance

== LLM Reasoning (Async) ==
Engine -> Decision: reason_and_decide(perceptions)
activate Decision

loop for each agent
    Decision -> Decision: format_perception_to_prompt(perception)
    Decision -> Decision: call_llm_with_circuit_breaker(prompt)
    note over Decision: Batch LLM calls\nfor performance
    Decision -> Decision: parse_llm_response_to_decision()
end

Decision -> Engine: Dict[agent_id, DecisionData]
deactivate Decision

== Action Translation & Validation ==
Engine -> Translator: translate_decisions(decisions)

loop for each decision
    Translator -> Translator: translate_decision(decision)
    Translator -> Mesa: validate_action(mesa_action, model)
    alt action valid
        Translator -> Translator: add_to_validated_actions()
    else action invalid
        Translator -> Error: handle_invalid_action(decision)
        Error -> Translator: fallback_action
    end
end

Translator -> Engine: Dict[agent_id, MesaAction]

== Action Execution ==
Engine -> Mesa: apply_actions(validated_actions)
loop for each action
    Mesa -> Mesa: execute_action_on_agent(action)
    note over Mesa: Update agent state\nTrigger environment changes
end

== State Synchronization ==
Engine -> Sync: sync_crewai_to_mesa(decisions, model)
Engine -> Sync: sync_mesa_to_crewai(model, agents)

== Performance Tracking ==
Engine -> Perf: record_operation_end("simulation_step", duration)

== Error Handling (if needed) ==
alt error occurred
    Engine -> Error: handle_error(exception, context)
    Error -> Error: classify_error(exception)
    Error -> Error: find_appropriate_handler()
    Error -> Engine: RecoveryResult
end

Engine -> Engine: create_step_result()
@enduml
```

### Timing Breakdown:
- Mesa step: ~10-50ms
- Perception extraction: ~5-20ms  
- LLM reasoning: ~500-3000ms (batched)
- Action translation: ~1-5ms
- State synchronization: ~5-15ms
- **Total: ~520-3090ms per step**

## 3. Error Handling and Recovery Sequence

### Circuit Breaker Pattern for LLM Failures

```plantuml
@startuml
participant "DecisionEngine" as Engine
participant "CircuitBreaker" as CB
participant "LLMClient" as LLM
participant "ErrorManager" as Error
participant "FallbackSystem" as Fallback

Engine -> CB: call(llm_function, prompt)

alt Circuit Breaker CLOSED (normal operation)
    CB -> LLM: complete(prompt)
    alt LLM success
        LLM -> CB: response
        CB -> CB: record_success()
        CB -> Engine: response
    else LLM failure
        LLM -> CB: exception
        CB -> CB: record_failure()
        CB -> CB: check_failure_threshold()
        alt threshold exceeded
            CB -> CB: open_circuit()
        end
        CB -> Engine: exception
    end

else Circuit Breaker OPEN (failing fast)
    CB -> Engine: CircuitBreakerOpenError
    Engine -> Error: handle_error(error, context)
    Error -> Fallback: generate_fallback_response(context)
    Fallback -> Error: fallback_response
    Error -> Engine: RecoveryResult(success=true, fallback_data)

else Circuit Breaker HALF_OPEN (testing recovery)
    CB -> LLM: complete(prompt) [test call]
    alt test success
        LLM -> CB: response
        CB -> CB: record_success()
        CB -> CB: close_circuit()
        CB -> Engine: response
    else test failure
        LLM -> CB: exception
        CB -> CB: record_failure()
        CB -> CB: open_circuit()
        CB -> Engine: exception
    end
end
@enduml
```

### Error Recovery Strategies:

1. **LLM Failure Recovery**:
   ```
   LLM API Error → Circuit Breaker → Fallback Response Generator
   ├── Personality-based responses
   ├── Rule-based decision making  
   └── Previous successful responses cache
   ```

2. **Mesa Failure Recovery**:
   ```
   Mesa Position Error → State Validator → Position Reset
   Mesa Bounds Error → Boundary Checker → Safe Position Assignment
   ```

3. **State Sync Failure Recovery**:
   ```
   Sync Conflict → State Validator → Rollback to Last Valid State
   Data Corruption → State Serializer → Restore from Snapshot
   ```

## 4. State Management and Synchronization

### Unified State Structure

```plantuml
@startuml
!define COMPONENT_COLOR #lightblue
!define DATA_COLOR #lightgreen
!define SYNC_COLOR #orange

package "Unified State Manager" COMPONENT_COLOR {
    [State Validator] SYNC_COLOR
    [State Synchronizer] SYNC_COLOR
    [Snapshot Manager] SYNC_COLOR
    
    database "Unified State" DATA_COLOR {
        [agents: {...}]
        [environment: {...}]
        [resources: {...}]
        [social: {...}]
        [memories: {...}]
    }
}

package "Mesa Framework" COMPONENT_COLOR {
    [Mesa Model]
    [Mesa Agents]
    [Mesa Grid]
    
    database "Mesa State" DATA_COLOR {
        [agent.pos]
        [agent.resources]
        [model.schedule]
        [grid.contents]
    }
}

package "CrewAI Framework" COMPONENT_COLOR {
    [CrewAI Agents]
    [Agent Memory]
    [Task System]
    
    database "CrewAI State" DATA_COLOR {
        [agent.memory]
        [agent.backstory]
        [conversation_history]
    }
}

[State Synchronizer] <--> [Mesa State] : Mesa → Unified
[State Synchronizer] <--> [CrewAI State] : CrewAI → Unified
[State Synchronizer] <--> [Unified State] : Bidirectional Sync

[State Validator] --> [Unified State] : Validate Changes
[Snapshot Manager] --> [Unified State] : Create Snapshots
@enduml
```

### State Synchronization Flow:

```plantuml
@startuml
participant "Mesa Model" as Mesa
participant "UnifiedStateManager" as USM
participant "CrewAI Agents" as CrewAI

== Mesa State Change ==
Mesa -> USM: register_state_change(change)
USM -> USM: validate_change(change)
USM -> USM: apply_to_unified_state(change)
USM -> USM: create_snapshot_if_needed()

== Synchronization to CrewAI ==
USM -> CrewAI: sync_cognitive_state()
USM -> CrewAI: update_agent_memory(unified_data)
USM -> CrewAI: sync_social_relationships()

== CrewAI State Change ==
CrewAI -> USM: register_memory_update(agent_id, memory)
USM -> USM: validate_memory_change()
USM -> USM: apply_to_unified_state()

== Conflict Resolution ==
alt conflicting changes detected
    USM -> USM: resolve_conflict(changes)
    note over USM: Strategy: Most Recent Wins\nOr: Priority-based Resolution
    USM -> Mesa: sync_resolved_state()
    USM -> CrewAI: sync_resolved_state()
end

== Performance Optimization ==
note over USM: Batch updates every N seconds\nCache unchanged data\nLazy synchronization for non-critical data
@enduml
```

## 5. Performance Monitoring and Caching

### Intelligent Caching Flow

```plantuml
@startuml
participant "Client" as Client
participant "IntelligentCache" as Cache
participant "PerformanceMonitor" as Monitor
participant "DataSource" as Source

Client -> Cache: get(key)

alt Cache Hit
    Cache -> Cache: check_ttl(key)
    alt TTL valid
        Cache -> Cache: update_access_statistics()
        Cache -> Monitor: record_cache_hit()
        Cache -> Client: cached_value
    else TTL expired
        Cache -> Cache: remove_expired_entry()
        Cache -> Source: fetch_fresh_data()
        Source -> Cache: fresh_data
        Cache -> Monitor: record_cache_miss()
        Cache -> Client: fresh_data
    end

else Cache Miss
    Cache -> Monitor: record_cache_miss()
    Cache -> Source: fetch_data()
    Source -> Cache: data
    
    alt cache_full
        Cache -> Cache: evict_lru_entry()
        note over Cache: Based on access frequency\nand recency weighted score
    end
    
    Cache -> Cache: store(key, data, ttl)
    Cache -> Cache: update_statistics()
    Cache -> Client: data
end

== Background Optimization ==
loop every 5 minutes
    Cache -> Cache: cleanup_expired_entries()
    Cache -> Cache: optimize_cache_size()
    Cache -> Monitor: update_cache_metrics()
end
@enduml
```

### Performance Monitoring Data Flow:

```
┌─────────────────┐    ┌──────────────────┐    ┌────────────────┐
│   Operation     │────▶│  Performance    │────▶│  Optimization │
│   Execution     │    │   Monitoring    │    │   Engine      │
│                 │    │                 │    │               │
│ ┌─────────────┐ │    │ ┌─────────────┐ │    │ ┌───────────┐ │
│ │LLM Call     │ │────▶│ │Record       │ │────▶│ │Batch LLM │ │
│ │Duration     │ │    │ │Latency      │ │    │ │Calls      │ │
│ └─────────────┘ │    │ └─────────────┘ │    │ └───────────┘ │
│                 │    │                 │    │               │
│ ┌─────────────┐ │    │ ┌─────────────┐ │    │ ┌───────────┐ │
│ │Memory Usage │ │────▶│ │Track        │ │────▶│ │Enable     │ │
│ │Metrics      │ │    │ │Resource     │ │    │ │Caching    │ │
│ └─────────────┘ │    │ │Usage        │ │    │ └───────────┘ │
│                 │    │ └─────────────┘ │    │               │
│ ┌─────────────┐ │    │                 │    │ ┌───────────┐ │
│ │Error Rate   │ │────▶│ ┌─────────────┐ │────▶│ │Circuit    │ │
│ │Tracking     │ │    │ │Calculate    │ │    │ │Breaker    │ │
│ └─────────────┘ │    │ │Error Rate   │ │    │ │Activation │ │
└─────────────────┘    │ └─────────────┘ │    │ └───────────┘ │
                       │                 │    │               │
                       │ ┌─────────────┐ │    │ ┌───────────┐ │
                       │ │Performance  │ │────▶│ │Connection │ │
                       │ │Profile      │ │    │ │Pool       │ │
                       │ │Generation   │ │    │ │Management │ │
                       │ └─────────────┘ │    │ └───────────┘ │
                       └──────────────────┘    └────────────────┘
```

## 6. Async Processing and Batching

### LLM Batch Processing Sequence

```plantuml
@startuml
participant "Agent1" as A1
participant "Agent2" as A2  
participant "Agent3" as A3
participant "LLMBatchProcessor" as Batch
participant "LLMClient" as LLM

== Request Collection Phase ==
A1 -> Batch: add_request(prompt1)
note over Batch: Start batch timer
A2 -> Batch: add_request(prompt2)
A3 -> Batch: add_request(prompt3)

alt Batch Size Reached OR Timeout
    note over Batch: Trigger batch processing
    
    == Batch Processing Phase ==
    Batch -> Batch: collect_current_batch()
    Batch -> LLM: batch_complete([prompt1, prompt2, prompt3])
    
    alt LLM Batch Support Available
        LLM -> LLM: process_batch_efficiently()
        LLM -> Batch: [response1, response2, response3]
    else No Batch Support
        par
            LLM -> LLM: complete(prompt1)
        and
            LLM -> LLM: complete(prompt2)  
        and
            LLM -> LLM: complete(prompt3)
        end
        LLM -> Batch: [response1, response2, response3]
    end
    
    == Response Distribution ==
    Batch -> A1: response1
    Batch -> A2: response2
    Batch -> A3: response3
    
    note over Batch: Update batch statistics\nStart new batch timer
end
@enduml
```

### Connection Pool Management

```plantuml
@startuml
participant "Client" as Client
participant "ConnectionPool" as Pool
participant "Connection" as Conn
participant "ExternalService" as Service

== Connection Acquisition ==
Client -> Pool: acquire_connection()

alt Available Connection Exists
    Pool -> Pool: get_available_connection()
    Pool -> Pool: validate_connection_health()
    alt Connection Healthy
        Pool -> Client: connection
    else Connection Stale
        Pool -> Conn: close()
        Pool -> Pool: create_new_connection()
        Pool -> Client: new_connection
    end

else No Available Connections
    alt Pool Under Limit
        Pool -> Service: create_connection()
        Service -> Pool: new_connection
        Pool -> Client: new_connection
    else Pool Exhausted
        Pool -> Client: PoolExhaustedError
    end
end

== Connection Usage ==
Client -> Conn: perform_operation()
Conn -> Service: api_call()
Service -> Conn: response
Conn -> Client: response

== Connection Release ==
Client -> Pool: release_connection(connection)
Pool -> Pool: validate_connection_health()
alt Connection Healthy
    Pool -> Pool: return_to_available_pool()
else Connection Unhealthy
    Pool -> Conn: close()
    Pool -> Pool: update_statistics()
end
@enduml
```

## 7. Integration Points and Data Handoffs

### Key Integration Points:

1. **Mesa → Perception Pipeline**
   ```python
   # Data transformation point
   mesa_agent.pos → PerceptionData.spatial_data['current_position']
   mesa_model.resources → PerceptionData.resources['available_resources']
   ```

2. **Perception → CrewAI Prompt**
   ```python
   # Natural language formatting
   PerceptionData → "You are at position (5,3). Resources nearby: key, tool."
   ```

3. **CrewAI Response → Mesa Action**
   ```python
   # Decision parsing and translation
   "I want to move north" → MesaAction(type="move", params={'direction': 'north'})
   ```

4. **State Synchronization Points**
   ```python
   # Bidirectional sync
   unified_state.agents[id].position ↔ mesa_agent.pos
   unified_state.memories[id] ↔ crewai_agent.memory
   ```

### Performance Considerations:

- **LLM Latency**: 500-3000ms per call (mitigated by batching)
- **Perception Extraction**: ~20ms (cached for performance)
- **State Synchronization**: ~15ms (batched updates)
- **Memory Operations**: ~10ms (intelligent caching)

### Error Recovery Pathways:

1. **LLM Failures** → Circuit Breaker → Fallback Responses
2. **Mesa Errors** → State Validation → Position/State Reset  
3. **Sync Conflicts** → Conflict Resolution → State Rollback
4. **Performance Degradation** → Auto-optimization → Caching/Batching

This architecture provides a robust, performant hybrid system that gracefully handles the integration challenges between Mesa's spatial simulation and CrewAI's LLM-powered reasoning capabilities.