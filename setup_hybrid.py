"""
Setup configuration for Mesa-CrewAI Hybrid Architecture Package

This setup.py file configures the hybrid architecture as an installable package
with proper dependency management and optional feature sets.
"""

from setuptools import setup, find_packages
import os
import sys

# Ensure Python version compatibility
if sys.version_info < (3, 8):
    sys.exit("Python 3.8 or higher is required for Mesa-CrewAI Hybrid Architecture")

# Read README for long description
def read_readme():
    readme_path = os.path.join(os.path.dirname(__file__), "README.md")
    if os.path.exists(readme_path):
        with open(readme_path, "r", encoding="utf-8") as f:
            return f.read()
    return "Mesa-CrewAI Hybrid Architecture for intelligent agent simulations"

# Read version from package
def read_version():
    version_path = os.path.join(os.path.dirname(__file__), "src", "escape_room_sim", "hybrid", "__init__.py")
    if os.path.exists(version_path):
        with open(version_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("__version__"):
                    return line.split("=")[1].strip().strip('"').strip("'")
    return "1.0.0"

# Core dependencies (always required)
install_requires = [
    # Framework dependencies
    "mesa>=2.1.5",
    "crewai>=0.15.0", 
    "langchain-community>=0.0.21",
    "numpy>=1.26.0",
    
    # LLM API support
    "openai>=1.97.0",
    "anthropic>=0.25.0",
    "google-generativeai>=0.3.0",
    
    # Async and concurrency
    "aiohttp>=3.9.0",
    "asyncio-throttle>=1.0.2",
    "tenacity>=8.2.0",
    
    # Data processing
    "pydantic>=2.11.0",
    "pydantic-settings>=2.0.0",
    
    # Configuration
    "python-dotenv>=1.0.0",
    "pyyaml>=6.0.1",
    
    # Utilities
    "rich>=13.7.0",
    "click>=8.1.0",
    "structlog>=23.2.0",
    "psutil>=5.9.6"
]

# Optional dependencies organized by feature
extras_require = {
    # Development dependencies
    "dev": [
        "pytest>=8.0.0",
        "pytest-asyncio>=0.24.0", 
        "pytest-cov>=4.0.0",
        "pytest-mock>=3.12.0",
        "pytest-benchmark>=4.0.0",
        "black>=24.0.0",
        "flake8>=7.0.0",
        "mypy>=1.8.0",
        "isort>=5.13.0",
        "pre-commit>=3.6.0",
        "hypothesis>=6.92.0",
        "faker>=20.1.0",
        "memory-profiler>=0.61.0"
    ],
    
    # Documentation dependencies
    "docs": [
        "sphinx>=7.2.0",
        "sphinx-rtd-theme>=2.0.0",
        "myst-parser>=2.0.0",
        "sphinx-autodoc-typehints>=1.25.0"
    ],
    
    # Performance optimization dependencies
    "performance": [
        "redis>=5.0.0",
        "cachetools>=5.3.0",
        "msgpack>=1.0.7",
        "orjson>=3.9.10",
        "lru-dict>=1.3.0",
        "prometheus-client>=0.19.0"
    ],
    
    # Visualization dependencies
    "viz": [
        "mesa-viz-tornado>=0.1.2",
        "plotly>=5.17.0",
        "bokeh>=3.3.0",
        "matplotlib>=3.8.0",
        "networkx>=3.2.0"
    ],
    
    # Machine learning dependencies
    "ml": [
        "scikit-learn>=1.3.0",
        "scipy>=1.11.0",
        "pandas>=2.1.0",
        # Optional heavy dependencies
        # "transformers>=4.35.0",
        # "torch>=2.1.0",
        # "gymnasium>=0.29.0"
    ],
    
    # Database integration dependencies
    "database": [
        "sqlalchemy>=2.0.0",
        "psycopg2-binary>=2.9.0",
        "pymongo>=4.6.0"
    ],
    
    # Cloud integration dependencies  
    "cloud": [
        "boto3>=1.34.0",
        "google-cloud-storage>=2.10.0",
        "azure-storage-blob>=12.19.0"
    ],
    
    # Distributed computing dependencies
    "distributed": [
        "celery>=5.3.0", 
        "redis>=5.0.0",
        "dask>=2023.12.0"
    ],
    
    # Security dependencies
    "security": [
        "cryptography>=41.0.0",
        "bcrypt>=4.1.0",
        "pyjwt>=2.8.0"
    ],
    
    # Networking dependencies
    "networking": [
        "websockets>=12.0",
        "pika>=1.3.0"  # RabbitMQ
    ],
    
    # Spatial/Geographic dependencies
    "spatial": [
        "shapely>=2.0.0",
        "geopandas>=0.14.0", 
        "rtree>=1.1.0"
    ]
}

# Convenience combinations
extras_require["all"] = list(set(sum(extras_require.values(), [])))
extras_require["testing"] = extras_require["dev"] + extras_require["performance"]
extras_require["production"] = extras_require["performance"] + extras_require["database"] + extras_require["security"]

# Package metadata
setup(
    name="mesa-crewai-hybrid",
    version=read_version(),
    author="Claude Code Architecture Team",
    author_email="architecture@anthropic.com",
    description="Revolutionary hybrid architecture combining Mesa agent-based modeling with CrewAI LLM agents",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/anthropic/mesa-crewai-hybrid", 
    project_urls={
        "Documentation": "https://mesa-crewai-hybrid.readthedocs.io/",
        "Source": "https://github.com/anthropic/mesa-crewai-hybrid",
        "Tracker": "https://github.com/anthropic/mesa-crewai-hybrid/issues",
    },
    
    # Package discovery
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    
    # Include non-Python files
    include_package_data=True,
    package_data={
        "escape_room_sim.hybrid": [
            "*.yml",
            "*.yaml", 
            "*.json",
            "*.toml",
            "templates/*.txt",
            "config/*.conf"
        ]
    },
    
    # Dependencies
    python_requires=">=3.8",
    install_requires=install_requires,
    extras_require=extras_require,
    
    # Classification
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9", 
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Operating System :: OS Independent",
        "Framework :: AsyncIO",
        "Topic :: System :: Distributed Computing"
    ],
    
    # Keywords for PyPI search
    keywords=[
        "mesa", "crewai", "agent-based-modeling", "llm", "ai", "simulation",
        "multi-agent", "hybrid-architecture", "spatial-modeling", "reasoning"
    ],
    
    # Entry points for command-line tools
    entry_points={
        "console_scripts": [
            "hybrid-sim=escape_room_sim.hybrid.cli:main",
            "hybrid-test=escape_room_sim.hybrid.testing_framework:cli_main",
            "hybrid-bench=escape_room_sim.hybrid.performance:benchmark_cli"
        ]
    },
    
    # Zip safety
    zip_safe=False,
    
    # Test suite configuration
    test_suite="tests",
    tests_require=extras_require["dev"],
    
    # Documentation
    cmdclass={},
)

# Post-installation message
print("""
Mesa-CrewAI Hybrid Architecture Installation Complete!

Quick Start:
-----------
1. Set up your environment:
   export GEMINI_API_KEY=your_key_here
   
2. Run a simple simulation:
   python -c "
   from escape_room_sim.hybrid import create_hybrid_simulation
   engine = create_hybrid_simulation(
       room_config={'width': 10, 'height': 10},
       agent_configs=[{'role': 'strategist'}],
       llm_config={'provider': 'gemini'}
   )
   print('Hybrid simulation created successfully!')
   "

3. Run tests:
   pytest tests/hybrid/

Optional Features:
-----------------
Install optional dependencies for additional features:

• Performance optimization: pip install mesa-crewai-hybrid[performance]
• Visualization: pip install mesa-crewai-hybrid[viz] 
• Machine learning: pip install mesa-crewai-hybrid[ml]
• All features: pip install mesa-crewai-hybrid[all]

Documentation:
-------------
• Docs: https://mesa-crewai-hybrid.readthedocs.io/
• Examples: examples/
• API Reference: docs/api/

Need Help?
----------
• GitHub Issues: https://github.com/anthropic/mesa-crewai-hybrid/issues
• Discord: https://discord.gg/mesa-crewai
• Email: architecture@anthropic.com
""")