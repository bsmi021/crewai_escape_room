# Project Structure & Organization

## Directory Layout

```
crewai-escape-room/
├── src/escape_room_sim/           # Main application code
│   ├── agents/                    # AI agent implementations
│   │   ├── strategist.py         # Strategic analysis agent
│   │   ├── mediator.py           # Diplomatic coordination agent
│   │   └── survivor.py           # Pragmatic execution agent
│   ├── tasks/                     # CrewAI task definitions
│   │   ├── assessment.py         # Room analysis tasks
│   │   ├── planning.py           # Strategy development tasks
│   │   └── execution.py          # Action coordination tasks
│   ├── simulation/                # Core simulation engines
│   │   ├── simple_engine.py      # Main simulation controller
│   │   └── iterative_engine.py   # Advanced iteration logic
│   ├── memory/                    # Persistent learning system
│   │   └── persistent_memory.py  # Cross-iteration memory
│   ├── room/                      # Virtual environment
│   │   └── escape_room_state.py  # Room state management
│   └── utils/                     # Configuration and utilities
│       └── llm_config.py         # AI model configurations
├── tests/                         # Comprehensive test suite
│   ├── unit/                     # Unit tests for components
│   ├── integration/              # Integration tests
│   ├── fixtures/                 # Test data and fixtures
│   └── conftest.py              # Shared test configuration
├── data/                         # Simulation results and logs
├── docs/                         # Documentation
├── scripts/                      # Utility scripts
├── main.py                       # Primary application entry point
├── runme.py                      # Alternative entry point
└── requirements.txt              # Python dependencies
```

## Code Organization Patterns

### Agent Architecture
- Each agent (Strategist, Mediator, Survivor) has its own module
- Agents are created via factory functions: `create_[agent]_agent()`
- Memory-enabled agents adapt backstory based on iteration context
- All agents support verbose output and configurable parameters

### Task Structure
- Tasks are organized by simulation phase: assessment → planning → execution
- Each task module provides factory functions for creating task lists
- Tasks are designed to work with specific agent combinations
- Task descriptions include expected outputs and success criteria

### Simulation Engine
- `SimpleEscapeSimulation` class manages the full simulation lifecycle
- `SimulationConfig` dataclass for configuration management
- `IterationResult` dataclass for structured result storage
- JSON-based persistence for results and intermediate data

### Memory System
- Cross-iteration learning through persistent memory
- Context-aware agent creation based on previous failures
- Learning limits (first 3 failed strategies) to prevent context overflow
- Structured context data with specific keys for each agent type

## File Naming Conventions

### Python Modules
- Snake_case for all Python files
- Descriptive names indicating functionality
- Agent files named after their role: `strategist.py`, `mediator.py`, `survivor.py`
- Task files named after simulation phase: `assessment.py`, `planning.py`, `execution.py`

### Data Files
- JSON format for structured data
- Timestamped iteration files: `iteration_01.json`, `iteration_02.json`
- Final results in `final_report.json`
- Coverage reports in `tests/coverage_html_final/`

### Configuration Files
- `.env` for environment variables (not committed)
- `.env.example` as template
- `pyproject.toml` for project metadata
-