# Competitive Survival Mechanics

This module implements the competitive survival mechanics for the escape room simulation, transforming the cooperative scenario into a competitive one where only one agent can escape.

## Components

### ResourceManager

The `ResourceManager` class handles scarcity enforcement and resource competition between agents. It implements all requirements from the competitive survival mechanics specification:

#### Key Features

- **Resource Scarcity Enforcement (Req 3.1)**: Manages exclusive and shared resources with proper scarcity rules
- **Ownership Tracking (Req 3.2)**: Tracks resource ownership and resolves conflicts between agents
- **Agent-Specific Access (Req 3.3)**: Provides agent-specific resource filtering and availability checking
- **Resource Transfer (Req 3.4)**: Handles resource trading and sharing between agents
- **Usage History (Req 3.5)**: Comprehensive tracking of all resource operations for analysis

#### Core Methods

- `claim_resource(agent_id, resource_id)`: Attempt to claim a resource
- `transfer_resource(from_agent, to_agent, resource_id)`: Transfer resource between agents
- `get_available_resources(agent_id)`: Get resources available to a specific agent
- `release_resource(agent_id, resource_id)`: Release a claimed resource
- `get_resource_conflicts()`: Identify resource conflicts and competition
- `get_scarcity_metrics()`: Get metrics about resource utilization and competition

#### Data Models

- `ScarceResource`: Represents a limited resource with exclusivity rules
- `ClaimResult`: Result of resource claim attempts
- `TransferResult`: Result of resource transfer attempts
- `ResourceUsageRecord`: Historical record of resource operations

#### Testing

The ResourceManager is fully tested using TDD methodology with:
- 35 unit tests covering all functionality
- 6 integration tests verifying requirements compliance
- Comprehensive test coverage for edge cases and error conditions

## Usage Example

```python
from src.escape_room_sim.competitive.models import ScarceResource
from src.escape_room_sim.competitive.resource_manager import ResourceManager

# Create resources
resources = [
    ScarceResource(
        id="key",
        name="Master Key",
        description="Opens the main door",
        required_for=["main_exit"],
        exclusivity=True,
        usage_cost=0
    ),
    ScarceResource(
        id="map",
        name="Room Map",
        description="Shows room layout",
        required_for=["navigation"],
        exclusivity=False,
        usage_cost=0
    )
]

# Initialize manager
manager = ResourceManager(resources)

# Agent operations
result = manager.claim_resource("agent1", "key")
if result.success:
    print(f"Agent1 claimed the key: {result.message}")

# Check availability for other agents
available = manager.get_available_resources("agent2")
print(f"Agent2 can access: {[r.name for r in available]}")

# Transfer resources
transfer_result = manager.transfer_resource("agent1", "agent2", "key")
if transfer_result.success:
    print(f"Key transferred: {transfer_result.message}")
```

This implementation provides the foundation for competitive resource management in the escape room simulation, ensuring that agents must compete for limited resources while maintaining proper tracking and conflict resolution.