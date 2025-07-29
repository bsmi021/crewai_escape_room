"""Simple test for Google Gemini integration."""

import os
import sys
from dotenv import load_dotenv

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Load environment variables
load_dotenv()

def main():
    print("Google Gemini Integration Test")
    print("=" * 40)
    
    # Test 1: Configuration
    print("\n1. Testing configuration...")
    try:
        from escape_room_sim.utils.llm_config import validate_gemini_configuration
        is_valid, message = validate_gemini_configuration()
        if is_valid:
            print(f"PASS: {message}")
        else:
            print(f"FAIL: {message}")
            return
    except Exception as e:
        print(f"FAIL: {e}")
        return
    
    # Test 2: LLM Creation
    print("\n2. Testing LLM creation...")
    try:
        from escape_room_sim.utils.llm_config import get_strategic_gemini_llm
        llm = get_strategic_gemini_llm()
        print("PASS: LLM created successfully")
    except Exception as e:
        print(f"FAIL: {e}")
        return
    
    # Test 3: Agent Creation
    print("\n3. Testing agent creation...")
    try:
        from escape_room_sim.agents.strategist import create_strategist_agent
        agent = create_strategist_agent()
        print("PASS: Agent created successfully")
    except Exception as e:
        print(f"FAIL: {e}")
        return
    
    print("\nAll tests passed! Gemini integration is working.")

if __name__ == "__main__":
    main()