"""
Main entry point for the CrewAI Escape Room Simulation.

This script initializes and runs the iterative escape room simulation
where three AI agents collaborate to solve puzzles and make survival decisions.
"""

import os
from dotenv import load_dotenv

def main():
    """Main entry point for the simulation."""
    # Load environment variables
    load_dotenv()
    
    # Verify required environment variables
    if not os.getenv("OPENAI_API_KEY") and not os.getenv("ANTHROPIC_API_KEY"):
        print("❌ Error: No API key found. Please set OPENAI_API_KEY or ANTHROPIC_API_KEY in your .env file")
        print("📋 Copy .env.example to .env and add your API key")
        return
    
    print("🚀 Starting CrewAI Escape Room Simulation...")
    print("📋 Project structure created successfully!")
    print("📋 Next steps:")
    print("   1. Install dependencies: pip install -r requirements.txt")
    print("   2. Copy .env.example to .env and add your API key")
    print("   3. Run the simulation once agents are implemented")

if __name__ == "__main__":
    main()