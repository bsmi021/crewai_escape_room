# Technology Stack & Development

## Core Technologies

### Framework & Libraries
- **Python 3.12+** - Primary language
- **CrewAI 0.15.0+** - Multi-agent framework for AI collaboration
- **LangChain Community** - LLM integration and utilities
- **Pydantic 2.11+** - Data validation and settings management
- **Rich Console** - Terminal UI and formatting

### AI/LLM Integration
- **Google Gemini** (Primary) - `gemini-2.5-flash-lite` model
- **OpenAI GPT** (Fallback) - GPT models support
- **Anthropic Claude** (Fallback) - Claude models support

### Development Tools
- **pytest** - Testing framework with coverage reporting
- **black** - Code formatting
- **flake8** - Linting
- **mypy** - Type checking
- **python-dotenv** - Environment configuration

## Build System

### Package Management
- **pip** with `requirements.txt` for dependencies
- **pyproject.toml** for project metadata
- Virtual environment recommended: `python -m venv venv`

### Environment Setup
```powershell
# Create and activate virtual environment
python -m venv venv
& .\venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt

# Configure API keys
copy .env.example .env
# Edit .env with your API keys
```

## Common Commands

### Running the Application
```powershell
# Main simulation
python main.py

# Alternative entry point
python runme.py
```

### Development & Testing
```powershell
# Run all tests with coverage
python -m pytest --cov=src --cov-report=html

# Run specific test categories
python -m pytest tests/unit/test_agents.py
python -m pytest tests/integration/
python -m pytest -k "test_strategist"

# Code quality checks
python -m black src/
python -m flake8 src/
python -m mypy src/
```

### Data Management
```powershell
# Results are saved in data/ directory
# - final_report.json (complete results)
# - iteration_XX.json (per-iteration data)
```

## Configuration

### Environment Variables (.env)
- `GOOGLE_API_KEY` - Primary AI provider
- `MODEL` - Model name (default: gemini-2.5-flash-lite)
- `OPENAI_API_KEY` - Fallback provider
- `ANTHROPIC_API_KEY` - Fallback provider
- `MAX_ITERATIONS` - Simulation limit
- `ENABLE_MEMORY` - Cross-iteration learning
- `VERBOSE_OUTPUT` - Detailed logging

### Platform Requirements
- **Windows 11** with PowerShell support
- **Python 3.12+** installed
- **Internet connection** for AI API calls