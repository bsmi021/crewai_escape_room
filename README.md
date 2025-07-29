# CrewAI Escape Room Simulation

<div align="center">
  <svg width="200" height="200" viewBox="0 0 200 200" xmlns="http://www.w3.org/2000/svg" role="img" aria-labelledby="logo-title" aria-describedby="logo-desc">
    <title id="logo-title">CrewAI Escape Room Logo</title>
    <desc id="logo-desc">Three AI agents collaborating to escape from a locked room with puzzle elements</desc>
    
    <!-- Background circle -->
    <circle cx="100" cy="100" r="95" fill="#1a1a2e" stroke="#16213e" stroke-width="3"/>
    
    <!-- Room walls -->
    <rect x="30" y="30" width="140" height="140" fill="none" stroke="#4a90e2" stroke-width="2"/>
    
    <!-- Door with lock -->
    <rect x="75" y="25" width="50" height="10" fill="#8b4513" rx="2"/>
    <rect x="80" y="30" width="40" height="5" fill="#654321"/>
    <circle cx="95" cy="32" r="3" fill="#ffd700" stroke="#ff8c00" stroke-width="1"/>
    
    <!-- Three AI agents as geometric figures -->
    <!-- Agent 1: Strategist (Triangle - analytical) -->
    <polygon points="50,140 40,160 60,160" fill="#00d4aa" opacity="0.8"/>
    <circle cx="50" cy="155" r="2" fill="#ffffff"/>
    <text x="50" y="170" text-anchor="middle" font-family="Arial, sans-serif" font-size="8" fill="#00d4aa">S</text>
    
    <!-- Agent 2: Mediator (Circle - diplomatic) -->
    <circle cx="100" cy="150" r="12" fill="#ff6b6b" opacity="0.8"/>
    <circle cx="100" cy="150" r="2" fill="#ffffff"/>
    <text x="100" y="170" text-anchor="middle" font-family="Arial, sans-serif" font-size="8" fill="#ff6b6b">M</text>
    
    <!-- Agent 3: Survivor (Square - pragmatic) -->
    <rect x="138" y="138" width="24" height="24" fill="#4ecdc4" opacity="0.8"/>
    <circle cx="150" cy="150" r="2" fill="#ffffff"/>
    <text x="150" y="170" text-anchor="middle" font-family="Arial, sans-serif" font-size="8" fill="#4ecdc4">R</text>
    
    <!-- Puzzle elements -->
    <!-- Gears -->
    <g transform="translate(130, 60)">
      <circle cx="0" cy="0" r="8" fill="none" stroke="#ffd700" stroke-width="2"/>
      <polygon points="-6,-2 -2,-6 2,-6 6,-2 6,2 2,6 -2,6 -6,2" fill="#ffd700" opacity="0.6"/>
    </g>
    
    <!-- Key -->
    <g transform="translate(60, 80)">
      <circle cx="0" cy="0" r="4" fill="none" stroke="#ff8c00" stroke-width="2"/>
      <rect x="4" y="-1" width="12" height="2" fill="#ff8c00"/>
      <rect x="14" y="-3" width="2" height="2" fill="#ff8c00"/>
      <rect x="14" y="1" width="2" height="2" fill="#ff8c00"/>
    </g>
    
    <!-- Code symbols -->
    <text x="45" y="65" font-family="monospace" font-size="12" fill="#4a90e2">01</text>
    <text x="155" y="100" font-family="monospace" font-size="12" fill="#4a90e2">X3</text>
    
    <!-- Connection lines showing collaboration -->
    <line x1="50" y1="140" x2="100" y2="138" stroke="#ffffff" stroke-width="1" opacity="0.3" stroke-dasharray="2,2"/>
    <line x1="100" y1="138" x2="150" y2="138" stroke="#ffffff" stroke-width="1" opacity="0.3" stroke-dasharray="2,2"/>
    <line x1="150" y1="138" x2="50" y2="140" stroke="#ffffff" stroke-width="1" opacity="0.3" stroke-dasharray="2,2"/>
    
    <!-- Memory/learning indicator -->
    <g transform="translate(100, 80)">
      <circle cx="0" cy="0" r="15" fill="none" stroke="#9b59b6" stroke-width="2" opacity="0.4"/>
      <path d="M -8,-8 Q 0,-15 8,-8 Q 0,0 -8,-8" fill="#9b59b6" opacity="0.3"/>
      <text x="0" y="2" text-anchor="middle" font-family="Arial, sans-serif" font-size="8" fill="#9b59b6">AI</text>
    </g>
  </svg>
</div>

<p align="center">
  <strong>🤖 Multi-Agent AI Collaboration Simulation</strong><br>
  <em>Three AI personalities working together to solve puzzles and make survival decisions</em>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.12+-blue.svg" alt="Python Version"/>
  <img src="https://img.shields.io/badge/CrewAI-0.15.0+-green.svg" alt="CrewAI Version"/>
  <img src="https://img.shields.io/badge/AI%20Agents-3-orange.svg" alt="AI Agents"/>
  <img src="https://img.shields.io/badge/Memory%20Enabled-Yes-purple.svg" alt="Memory Enabled"/>
  <img src="https://img.shields.io/badge/Status-Active%20Development-yellow.svg" alt="Development Status"/>
</p>

---

## 🎯 Project Overview

**CrewAI Escape Room Simulation** is an innovative multi-agent AI system where three distinct AI personalities collaborate to escape from a virtual room. Each agent brings unique strengths—strategic thinking, diplomatic mediation, and pragmatic survival instincts—creating dynamic group problem-solving scenarios.

### ✨ What Makes This Special

- **🧠 Memory-Enhanced Learning**: Agents remember failed strategies and adapt their approach across iterations
- **🎭 Distinct Personalities**: Each agent has unique decision-making patterns and collaborative styles  
- **🔄 Iterative Problem-Solving**: Multiple simulation rounds allow for complex puzzle-solving evolution
- **📊 Comprehensive Analytics**: Detailed tracking of agent interactions, strategies, and learning patterns
- **⚡ Real-Time Adaptation**: Dynamic adjustment of strategies based on room constraints and team dynamics

## 🤖 Meet the AI Team

| Agent | Role | Personality | Specialty |
|-------|------|-------------|-----------|
| **🎯 Strategist** | Analytical Leader | Military tactician with systematic approach | Strategic planning, risk assessment, optimal solutions |
| **🤝 Mediator** | Team Facilitator | Diplomatic coordinator focused on consensus | Communication, conflict resolution, team harmony |
| **🛡️ Survivor** | Pragmatic Executor | Practical decision-maker prioritizing outcomes | Resource management, practical solutions, execution |

## 🚀 Quick Start

### Prerequisites
- **Python 3.12+** installed on your system
- **Google Gemini API key** (preferred) or OpenAI/Anthropic API key
- **Windows 11** with PowerShell support

### Installation

1. **Clone the repository**
   ```powershell
   git clone https://github.com/your-username/crewai-escape-room.git
   cd crewai-escape-room
   ```

2. **Set up the virtual environment**
   ```powershell
   python -m venv venv
   & .\venv\Scripts\Activate.ps1
   ```

3. **Install dependencies**
   ```powershell
   pip install -r requirements.txt
   ```

4. **Configure your API key**
   ```powershell
   copy .env.example .env
   # Edit .env file and add your API key:
   # GOOGLE_API_KEY=your_google_api_key_here
   # MODEL=gemini-2.5-flash-lite
   ```

5. **Run the simulation**
   ```powershell
   python main.py
   ```

## 🎮 How It Works

### Simulation Flow

The simulation follows a structured approach where agents collaborate through multiple phases:

1. **🔍 Room Assessment**: Systematic analysis of environment and constraints
2. **📋 Strategic Planning**: Development of escape strategies with contingencies  
3. **🤝 Collaborative Execution**: Coordinated implementation with real-time adaptation
4. **📈 Learning Integration**: Memory-based improvement across iterations

### Core Features

- **🔍 Intelligent Room Analysis**: Comprehensive search and documentation of environment
- **🧩 Dynamic Puzzle Solving**: Adaptive approach to codes, locks, and logical challenges  
- **💭 Strategic Planning**: Multi-phase execution plans with contingency options
- **🗣️ Team Communication**: Structured dialogue system with conflict resolution
- **📈 Progress Tracking**: Real-time monitoring of success rates and learning metrics

## 📋 Configuration Options

The simulation supports extensive customization through interactive prompts:

- **Max Iterations**: Control simulation length (default: 10)
- **Memory System**: Enable/disable cross-iteration learning
- **Verbose Output**: Detailed logs of agent reasoning and decisions
- **Deadlock Detection**: Automatic identification of stuck scenarios
- **Result Persistence**: Save intermediate results for analysis

## 🛠️ Development Commands

```powershell
# Activate environment
& .\venv\Scripts\Activate.ps1

# Run simulation
python main.py

# Run tests with coverage
python -m pytest --cov=src --cov-report=html

# Code formatting and linting
python -m black src\
python -m flake8 src\
python -m mypy src\

# Run specific test categories
python -m pytest tests\unit\test_agents.py          # Agent behavior tests
python -m pytest tests\integration\                  # Integration tests
python -m pytest -k "test_strategist"               # Pattern-based testing
```

## 📊 Example Output

```
CrewAI Escape Room Simulation
Iterative Multi-Agent Collaboration System

Three AI agents work together to escape:
   • Strategist: Analytical problem solver
   • Mediator: Group facilitator and diplomat  
   • Survivor: Pragmatic decision maker

Memory-enabled agents learn from each iteration

FINAL SIMULATION RESULTS
┌─ Simulation Complete ──────────────────────────┐
│                                                │
│ Outcome: SUCCESS                               │
│ Total Iterations: 2                           │
│ Duration: 355.4 seconds                       │
│ Solution Found: Yes                           │
│ Stop Reason: Solution found                   │
│                                                │
│ Learning Summary:                              │
│ • Total Lessons Learned: 2                    │
│ • Consensus Rate: 80.0%                       │
│ • Failed Strategies: 1                        │
│ • Successful Strategies: 1                    │
└────────────────────────────────────────────────┘

SUCCESS! The agents found an escape solution!
```

## 🏗️ Project Structure

```
crewai_escape_room/
├── src/escape_room_sim/
│   ├── agents/              # AI agent implementations
│   │   ├── strategist.py    # Strategic analysis agent
│   │   ├── mediator.py      # Diplomatic coordination agent
│   │   └── survivor.py      # Pragmatic execution agent
│   ├── tasks/               # CrewAI task definitions
│   │   ├── assessment.py    # Room analysis tasks
│   │   ├── planning.py      # Strategy development tasks
│   │   └── execution.py     # Action coordination tasks
│   ├── simulation/          # Core simulation engines
│   │   ├── simple_engine.py # Main simulation controller
│   │   └── iterative_engine.py # Advanced iteration logic
│   ├── memory/              # Persistent learning system
│   │   └── persistent_memory.py # Cross-iteration memory
│   ├── room/                # Virtual environment
│   │   └── escape_room_state.py # Room state management
│   └── utils/               # Configuration and utilities
│       └── llm_config.py    # AI model configurations
├── tests/                   # Comprehensive test suite
├── data/                    # Simulation results and logs
└── main.py                  # Application entry point
```

## 🔧 API Configuration

### Google Gemini (Recommended)
```bash
GOOGLE_API_KEY=your_google_api_key_here
MODEL=gemini-2.5-flash-lite
```

### Alternative Providers
```bash
# OpenAI
OPENAI_API_KEY=your_openai_api_key_here

# Anthropic Claude
ANTHROPIC_API_KEY=your_anthropic_api_key_here
```

## 📈 Advanced Features

### Memory System
- **Cross-Iteration Learning**: Agents remember failed strategies
- **Adaptive Behavior**: Personality-based learning patterns
- **Strategy Evolution**: Progressive improvement over iterations

### Analytics Dashboard
- **Success Rate Tracking**: Monitor escape success over time
- **Agent Performance**: Individual and team effectiveness metrics
- **Strategy Analysis**: Identification of successful vs. failed approaches
- **Learning Curves**: Visual representation of agent improvement

### Extensibility
- **Custom Room Designs**: Configurable puzzle types and layouts
- **Agent Personality Tuning**: Adjustable behavioral parameters
- **Scenario Templates**: Pre-built challenge configurations

## 🧪 Testing

Comprehensive test suite ensures reliable simulation behavior:

```powershell
# Run all tests
python -m pytest

# Generate coverage report
python -m pytest --cov=src --cov-report=html
open htmlcov/index.html

# Test specific components
python -m pytest tests/unit/test_agents.py -v
python -m pytest tests/integration/ -v
```

**Coverage Target**: 100% for core simulation logic

## 🤝 Contributing

We welcome contributions! Here's how to get started:

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/amazing-feature`
3. **Make your changes** with proper tests
4. **Run the test suite**: `pytest --cov=src`
5. **Format your code**: `black src/`
6. **Submit a pull request**

### Development Standards
- **Type Hints**: All functions must include type annotations
- **Docstrings**: Comprehensive documentation for all public APIs
- **Test Coverage**: New features require corresponding tests
- **Code Style**: Follow Black formatting and flake8 linting

## 📚 Documentation

- **[Design Document](CrewAI_Escape_Room_Design_Document.md)**: Complete technical architecture
- **[Development Guide](Comprehensive_Development_ToDo_List.md)**: Detailed task breakdown
- **[API Reference](docs/)**: Comprehensive code documentation
- **[Test Suite](tests/README.md)**: Testing strategy and guidelines

## 🔮 Roadmap

### Phase 1: Foundation ✅
- ✅ Project structure and dependencies
- ✅ Basic agent implementations
- ✅ Core simulation engine
- ✅ Memory system integration

### Phase 2: Enhancement 🚧
- 🔄 Advanced puzzle mechanics
- 🔄 Enhanced agent personality systems
- 🔄 Real-time analytics dashboard
- 🔄 Scenario configuration UI

### Phase 3: Scale 📋
- 📋 Multi-room environments
- 📋 Agent learning optimization
- 📋 Performance benchmarking
- 📋 Cloud deployment options

## 🐛 Troubleshooting

### Common Issues

**API Key Not Found**
```
Error: No valid API configuration found!
```
**Solution**: Ensure your `.env` file contains a valid API key for your chosen provider.

**Import Errors**
```
ModuleNotFoundError: No module named 'crewai'
```
**Solution**: Activate your virtual environment and reinstall dependencies:
```powershell
& .\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

**Windows Path Issues**
```
FileNotFoundError: [Errno 2] No such file or directory
```
**Solution**: Use PowerShell and ensure you're in the project root directory.

## 📄 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

- **[CrewAI](https://github.com/joaomdmoura/crewAI)**: Powerful multi-agent framework
- **[OpenAI](https://openai.com)**: GPT API support  
- **[Google](https://ai.google.dev)**: Gemini API integration
- **[Anthropic](https://anthropic.com)**: Claude API compatibility

---

<div align="center">
  <strong>🚀 Ready to watch AI agents collaborate and learn?</strong><br>
  <em>Start your simulation journey today!</em><br><br>
  
  **[Get Started](#-quick-start) • [Documentation](docs/) • [Report Issues](https://github.com/your-username/crewai-escape-room/issues)**
</div>
