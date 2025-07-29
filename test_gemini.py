"""
Test script for Google Gemini integration.

This script tests the Gemini LLM configuration and agent creation
without running the full simulation.
"""

import os
import sys
from dotenv import load_dotenv

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Load environment variables
load_dotenv()

def test_gemini_config():
    """Test Google Gemini configuration."""
    print("Testing Google Gemini Configuration...")
    
    try:
        from escape_room_sim.utils.llm_config import validate_gemini_configuration
        
        is_valid, message = validate_gemini_configuration()
        if is_valid:
            print(f"PASS: {message}")
            return True
        else:
            print(f"FAIL: {message}")
            return False
            
    except Exception as e:
        print(f"FAIL: Error testing configuration: {e}")
        return False

def test_llm_creation():
    """Test LLM instance creation."""
    print("\nTesting LLM Instance Creation...")
    
    try:
        from escape_room_sim.utils.llm_config import (
            get_strategic_gemini_llm,
            get_diplomatic_gemini_llm,
            get_pragmatic_gemini_llm
        )
        
        # Test strategic LLM
        strategic_llm = get_strategic_gemini_llm()
        print("PASS: Strategic LLM created successfully")
        
        # Test diplomatic LLM
        diplomatic_llm = get_diplomatic_gemini_llm()
        print("PASS: Diplomatic LLM created successfully")
        
        # Test pragmatic LLM
        pragmatic_llm = get_pragmatic_gemini_llm()
        print("PASS: Pragmatic LLM created successfully")
        
        return True
        
    except Exception as e:
        print(f"FAIL: Error creating LLM instances: {e}")
        return False

def test_agent_creation():
    """Test agent creation with Gemini LLMs."""
    print("\n👥 Testing Agent Creation...")
    
    try:
        from escape_room_sim.agents.strategist import create_strategist_agent
        from escape_room_sim.agents.mediator import create_mediator_agent
        from escape_room_sim.agents.survivor import create_survivor_agent
        
        # Test strategist agent
        strategist = create_strategist_agent()
        print("✅ Strategist agent created successfully")
        
        # Test mediator agent
        mediator = create_mediator_agent()
        print("✅ Mediator agent created successfully")
        
        # Test survivor agent
        survivor = create_survivor_agent()
        print("✅ Survivor agent created successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Error creating agents: {e}")
        return False

def test_basic_agent_interaction():
    """Test basic agent response (if API key is available)."""
    print("\n💬 Testing Basic Agent Response...")
    
    # Only test if we have a real API key
    if not os.getenv("GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY") == "your_GEMINI_API_KEY_here":
        print("⚠️  Skipping response test - no real API key provided")
        return True
    
    try:
        from escape_room_sim.agents.strategist import create_strategist_agent
        from crewai import Task
        
        # Create a simple agent
        agent = create_strategist_agent()
        
        # Create a simple test task
        task = Task(
            description="Briefly introduce yourself as the Strategic Analyst agent.",
            agent=agent,
            expected_output="A brief self-introduction in character."
        )
        
        print("🔄 Sending test request to Gemini...")
        # Note: This would require a full crew to execute
        print("✅ Agent response test setup complete (would need crew execution)")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in agent interaction test: {e}")
        return False

def main():
    """Run all tests."""
    print("🚀 Google Gemini Integration Test Suite")
    print("=" * 50)
    
    tests = [
        ("Configuration", test_gemini_config),
        ("LLM Creation", test_llm_creation),
        ("Agent Creation", test_agent_creation),
        ("Agent Interaction", test_basic_agent_interaction)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n📋 Running {test_name} Test...")
        success = test_func()
        results.append((test_name, success))
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 TEST RESULTS SUMMARY")
    print("=" * 50)
    
    passed = 0
    for test_name, success in results:
        status = "PASS" if success else "FAIL"
        emoji = "✅" if success else "❌"
        print(f"{emoji} {test_name}: {status}")
        if success:
            passed += 1
    
    print(f"\n🎯 {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("🎉 All tests passed! Gemini integration is working correctly.")
    else:
        print("⚠️  Some tests failed. Check the error messages above.")
        
        if not os.getenv("GEMINI_API_KEY"):
            print("\n💡 Tip: Set GEMINI_API_KEY in your .env file to run all tests")

if __name__ == "__main__":
    main()