# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a CrewAI Escape Room Simulation project where three AI agents with distinct personalities (Strategist, Mediator, Survivor) must collaborate to escape a room, but only two can survive. The project focuses on autonomous multi-agent conversation and iterative problem-solving using the CrewAI framework.

## Project Status

This repository is currently in **Phase 1 setup**. The project structure, dependencies, and development environment are complete. Core agent implementations and simulation engine are the next development priorities.

## Architecture Overview

The planned architecture uses a simplified approach centered around CrewAI's native capabilities:

### Core Components
- **Agents**: Three distinct personality agents (Strategist, Mediator, Survivor) with different goals and backstories
- **Iterative Simulation Engine**: Python loop around CrewAI's crew.kickoff() to enable multiple rounds of collaboration
- **Simple Memory System**: File-based storage for conversation logs, game state, and agent memories
- **Room Mechanics**: JSON-based room state with puzzles, resources, and constraints
- **Stopping Conditions**: Configurable conditions for natural simulation conclusion

### Planned Project Structure
```
escape_room_sim/
├── src/escape_room_sim/
│   ├── agents/          # Agent definitions (strategist, mediator, survivor)
│   ├── tasks/           # Task definitions (assessment, planning, execution)
│   ├── room/            # Room state and mechanics
│   ├── memory/          # Simple file-based memory system
│   └── simulation/      # Iterative simulation engine
├── tests/               # Comprehensive test suite (100% coverage target)
├── data/                # Game state, conversation logs, agent memories
└── scripts/             # Development and deployment scripts
```

## Key Design Principles

1. **KISS Principle**: Avoid overengineering - CrewAI is standalone and doesn't need LangGraph or complex state management
2. **Iterative Problem-Solving**: Agents work through multiple conversation rounds to find solutions
3. **Memory Persistence**: Agents learn from previous attempts and failed strategies
4. **Natural Stopping**: Dynamic conditions determine when simulation concludes

## Development Commands

This is a Windows 11 PowerShell environment. Use these commands for development:

```powershell
# Environment setup (if not done)
python -m venv venv
& .\venv\Scripts\Activate.ps1
pip install -r requirements.txt

# Copy environment template and add API key
copy .env.example .env
# Edit .env file to add OPENAI_API_KEY=your_key_here

# Development workflow
& .\venv\Scripts\Activate.ps1; python main.py                    # Run main simulation
& .\venv\Scripts\Activate.ps1; python -m pytest --cov=src --cov-report=html  # Run tests with coverage
& .\venv\Scripts\Activate.ps1; python -m black src\              # Format code
& .\venv\Scripts\Activate.ps1; python -m flake8 src\            # Lint code
& .\venv\Scripts\Activate.ps1; python -m mypy src\              # Type check

# Run specific tests
& .\venv\Scripts\Activate.ps1; python -m pytest tests\unit\test_agents.py  # Single test file
& .\venv\Scripts\Activate.ps1; python -m pytest -k "test_strategist"       # Test pattern
```

## Dependencies

Currently installed packages (see requirements.txt for exact versions):

**Core Framework:**
- `crewai>=0.55.0` - Multi-agent framework (v0.150.0 installed)
- `crewai-tools>=0.55.0` - Additional CrewAI tools
- `openai>=1.97.0` - OpenAI API client (latest)

**Configuration & Data:**
- `python-dotenv>=1.0.0` - Environment variable management
- `pydantic>=2.11.0` - Data validation and settings
- `pydantic-settings>=2.0.0` - Settings management
- `rich>=13.0.0` - Terminal formatting

**Development & Testing:**
- `pytest>=8.0.0` - Testing framework
- `pytest-cov>=4.0.0` - Coverage reporting  
- `black>=24.0.0` - Code formatting
- `flake8>=7.0.0` - Linting
- `mypy>=1.8.0` - Type checking

## Development Phases

1. **Phase 1 (Week 1)**: Basic CrewAI setup with three agents having sequential conversations
2. **Phase 2 (Week 2)**: Add room mechanics, resource constraints, and memory persistence
3. **Phase 3 (Week 3)**: Implement survival mechanics, iterative problem-solving, and UI

## Key Files for Understanding

- `CrewAI_Escape_Room_Design_Document.md` - Complete technical design and architecture
- `Comprehensive_Development_ToDo_List.md` - 371+ specific development tasks with professional standards
- `README.md` - Project overview and file locations

## Implementation Notes

- Agents should maintain distinct personalities across iterations
- Memory system tracks failed/successful strategies to prevent repeated failures
- Stopping conditions include solution found, consensus reached, time expired, or deadlock detected
- Target 100% test coverage with comprehensive unit and integration tests

## Current Implementation Status

**Completed:**
- ✅ Project structure with proper Python packages
- ✅ Virtual environment with all dependencies installed  
- ✅ Development tooling (pytest, black, flake8, mypy)
- ✅ Environment configuration (.env.example)
- ✅ Main entry point (main.py) with API key validation

**Next Development Priorities:**
1. Implement three core agents (Strategist, Mediator, Survivor) in `src/escape_room_sim/agents/`
2. Create task definitions in `src/escape_room_sim/tasks/`
3. Build iterative simulation engine in `src/escape_room_sim/simulation/`
4. Implement file-based memory system in `src/escape_room_sim/memory/`

## Environment Setup Requirements

**Windows 11 PowerShell Environment:**
- Use PowerShell commands only (avoid Linux/bash commands)
- File paths must use Windows format with double backslashes when needed
- Virtual environment activation: `& .\venv\Scripts\Activate.ps1`

**API Configuration:**
- Copy `.env.example` to `.env` 
- Add `OPENAI_API_KEY=your_key_here` to `.env` file
- API key validation is built into main.py

**Core Development Pattern:**
- CrewAI uses standalone framework (no LangGraph/LangChain complexity)
- Agents have memory enabled for learning across iterations
- Simulation uses Python loop around `crew.kickoff()` for multiple rounds
- File-based storage for simplicity (JSON/text files in `data/` directory)