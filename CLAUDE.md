# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a CrewAI Escape Room Simulation project where three AI agents with distinct personalities (Strategist, Mediator, Survivor) must collaborate to escape a room, but only two can survive. The project focuses on autonomous multi-agent conversation and iterative problem-solving using the CrewAI framework.

## Project Status

This repository is a **fully functional CrewAI multi-agent simulation**. All core components are implemented including three personality agents, iterative simulation engine, memory system, room mechanics, and comprehensive test suite. The project is ready for simulation runs and further feature development.

## Architecture Overview

The implemented architecture uses a simplified approach centered around CrewAI's native capabilities:

### Core Components
- **Agents**: Three distinct personality agents (Strategist, Mediator, Survivor) with different goals and backstories
- **Iterative Simulation Engine**: Python loop around CrewAI's crew.kickoff() to enable multiple rounds of collaboration
- **Simple Memory System**: File-based storage for conversation logs, game state, and agent memories
- **Room Mechanics**: JSON-based room state with puzzles, resources, and constraints
- **Stopping Conditions**: Configurable conditions for natural simulation conclusion

### Current Project Structure
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

# Copy environment template and configure API
copy .env.example .env
# Edit .env file to add GEMINI_API_KEY=your_key_here (preferred)
# Or add OPENAI_API_KEY=your_key_here (fallback)

# Run the simulation
python main.py                                          # Interactive simulation with config options

# Development workflow  
python -m pytest tests/run_tests.py                    # Run comprehensive test suite
python -m pytest --cov=src --cov-report=html          # Run tests with coverage
python -m black src/                                   # Format code
python -m flake8 src/                                  # Lint code
python -m mypy src/                                    # Type check

# Run specific test categories
python tests/run_tests.py tests/unit/test_strategist_agent.py    # Single test file
python -m pytest tests/unit/ -v                                  # All unit tests
python -m pytest tests/integration/ -v                           # All integration tests
python -m pytest -k "strategist" -v                             # Test pattern matching
```

## Dependencies

Currently installed packages (see requirements.txt for exact versions):

**Core Framework:**
- `crewai>=0.15.0` - Multi-agent framework
- `langchain-community>=0.0.21` - LangChain integrations
- `langchain-ollama>=0.1.3` - Local LLM support
- `numpy>=1.26.0` - Numerical computing

**LLM API Support:**
- `openai>=1.97.0` - OpenAI API client
- `anthropic>=0.25.0` - Anthropic Claude API
- Google Gemini API support (via environment configuration)

**Configuration & Data:**
- `python-dotenv>=1.0.0` - Environment variable management
- `pydantic>=2.11.0` - Data validation and settings
- `pydantic-settings>=2.0.0` - Settings management

**Development & Testing:**
- `pytest>=8.0.0` - Testing framework with async support
- `pytest-cov>=4.0.0` - Coverage reporting
- `pytest-asyncio>=0.24.0` - Async test support
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

**Fully Implemented:**
- ✅ Complete project structure with proper Python packages
- ✅ Three personality agents (Strategist, Mediator, Survivor) with distinct roles and backstories
- ✅ Task system (Assessment, Planning, Execution) with memory integration
- ✅ Iterative simulation engine with configurable stopping conditions
- ✅ File-based memory system with persistent agent learning
- ✅ Room state mechanics with JSON-based game state
- ✅ Interactive main.py with configuration options and rich console output
- ✅ Comprehensive test suite (unit + integration tests) with >90% coverage
- ✅ Development tooling (pytest, black, flake8, mypy)
- ✅ Multi-LLM API support (Gemini preferred, OpenAI/Anthropic fallback)

**Current Capabilities:**
- Run full escape room simulations with configurable iterations
- Agents learn and adapt strategies across multiple rounds
- Memory persistence between simulation runs
- Detailed logging and result analysis
- Deadlock detection and natural stopping conditions

**Future Enhancement Opportunities:**
- Advanced room puzzles and mechanics
- Web-based user interface
- Real-time collaboration features
- Additional AI model integrations

## Environment Setup Requirements

**Windows 11 PowerShell Environment:**
- Use PowerShell commands only (avoid Linux/bash commands)
- File paths must use Windows format with double backslashes when needed
- Virtual environment activation: `& .\venv\Scripts\Activate.ps1`

**API Configuration (Multiple Provider Support):**
- Copy `.env.example` to `.env`
- **Primary:** Add `GEMINI_API_KEY=your_key_here` and `MODEL=gemini-2.5-flash-lite` for Google Gemini (preferred)
- **Fallback:** Add `OPENAI_API_KEY=your_key_here` for OpenAI GPT models
- **Alternative:** Add `ANTHROPIC_API_KEY=your_key_here` for Claude models
- API validation and automatic fallback is built into main.py

**Core Development Patterns:**
- CrewAI standalone framework with LangChain integrations where beneficial
- Memory-enabled agents that learn from previous iterations and failures
- Iterative simulation engine using `SimpleEscapeSimulation` class
- JSON-based persistence for game state, agent memories, and results
- Rich console interface with interactive configuration
- Comprehensive test coverage with `tests/run_tests.py` script
- Modular architecture allowing easy extension of agents, tasks, and mechanics