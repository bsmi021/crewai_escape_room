"""
Test setup and verification script for CrewAI Escape Room Agent tests.
Validates that the test infrastructure works correctly.
"""

import sys
import os
from pathlib import Path
from unittest.mock import Mock, patch

# Add the src directory to Python path
project_root = Path(__file__).parent.parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

# Mock the crewai module before any imports
sys.modules['crewai'] = Mock()

# Mock Agent class with required interface
class MockAgent:
    def __init__(self, role=None, goal=None, backstory=None, verbose=True, 
                 memory=True, system_message=None, max_iter=3, allow_delegation=False):
        self.role = role
        self.goal = goal
        self.backstory = backstory
        self.verbose = verbose
        self.memory = memory
        self.system_message = system_message
        self.max_iter = max_iter
        self.allow_delegation = allow_delegation

# Set up the mock Agent in the crewai module
sys.modules['crewai'].Agent = MockAgent

def test_imports():
    """Test that all agent modules can be imported successfully."""
    print("Testing agent module imports...")
    
    try:
        from escape_room_sim.agents.strategist import create_strategist_agent, StrategistConfig
        print("PASS: Strategist agent imported successfully")
    except Exception as e:
        print(f"FAIL: Strategist agent import failed: {e}")
        return False

    try:
        from escape_room_sim.agents.mediator import create_mediator_agent, MediatorConfig
        print("PASS: Mediator agent imported successfully")
    except Exception as e:
        print(f"FAIL: Mediator agent import failed: {e}")
        return False

    try:
        from escape_room_sim.agents.survivor import create_survivor_agent, SurvivorConfig
        print("PASS: Survivor agent imported successfully")
    except Exception as e:
        print(f"FAIL: Survivor agent import failed: {e}")
        return False

    return True

def test_agent_creation():
    """Test that agents can be created with mock dependencies."""
    print("\nTesting agent creation...")
    
    try:
        from escape_room_sim.agents.strategist import create_strategist_agent
        from escape_room_sim.agents.mediator import create_mediator_agent
        from escape_room_sim.agents.survivor import create_survivor_agent
        
        # Test default creation
        strategist = create_strategist_agent()
        print("PASS: Strategist agent created successfully")
        
        mediator = create_mediator_agent()
        print("PASS: Mediator agent created successfully")
        
        survivor = create_survivor_agent()
        print("PASS: Survivor agent created successfully")
        
        # Test with parameters
        strategist_custom = create_strategist_agent(memory_enabled=False, verbose=False)
        print("PASS: Strategist agent with custom parameters created successfully")
        
        # Test with context
        context = {"failed_strategies": ["Test strategy"]}
        strategist_context = create_strategist_agent(iteration_context=context)
        print("PASS: Strategist agent with context created successfully")
        
        return True
        
    except Exception as e:
        print(f"FAIL: Agent creation failed: {e}")
        return False

def test_config_classes():
    """Test that configuration classes work correctly."""
    print("\nTesting configuration classes...")
    
    try:
        from escape_room_sim.agents.strategist import StrategistConfig
        from escape_room_sim.agents.mediator import MediatorConfig
        from escape_room_sim.agents.survivor import SurvivorConfig
        
        # Test StrategistConfig
        strategist_traits = StrategistConfig.get_personality_traits()
        assert isinstance(strategist_traits, dict)
        assert len(strategist_traits) > 0
        print("PASS: StrategistConfig working correctly")
        
        # Test MediatorConfig
        mediator_traits = MediatorConfig.get_personality_traits()
        mediator_tracking = MediatorConfig.get_relationship_tracking_config()
        assert isinstance(mediator_traits, dict)
        assert isinstance(mediator_tracking, dict)
        print("PASS: MediatorConfig working correctly")
        
        # Test SurvivorConfig
        survivor_traits = SurvivorConfig.get_personality_traits()
        survivor_priorities = SurvivorConfig.get_survival_priorities()
        survivor_criteria = SurvivorConfig.get_decision_criteria()
        assert isinstance(survivor_traits, dict)
        assert isinstance(survivor_priorities, dict)
        assert isinstance(survivor_criteria, dict)
        print("PASS: SurvivorConfig working correctly")
        
        return True
        
    except Exception as e:
        print(f"FAIL: Configuration class testing failed: {e}")
        return False

def run_sample_tests():
    """Run a sample of the actual tests to verify test infrastructure."""
    print("\nRunning sample tests...")
    
    try:
        # Import a simple test
        import pytest
        
        # Try to run a simple test function
        from unittest.mock import patch
        from escape_room_sim.agents.strategist import create_strategist_agent
        
        # Mock the Agent class
        with patch('escape_room_sim.agents.strategist.Agent') as mock_agent:
            mock_instance = Mock()
            mock_agent.return_value = mock_instance
            
            # Create agent
            agent = create_strategist_agent()
            
            # Verify mock was called
            mock_agent.assert_called_once()
            assert agent == mock_instance
            
        print("PASS: Sample test passed successfully")
        return True
        
    except Exception as e:
        print(f"FAIL: Sample test failed: {e}")
        return False

def main():
    """Main test setup verification."""
    print("CrewAI Escape Room Agent Test Setup Verification")
    print("=" * 50)
    
    all_passed = True
    
    all_passed &= test_imports()
    all_passed &= test_agent_creation()
    all_passed &= test_config_classes()
    all_passed &= run_sample_tests()
    
    print("\n" + "=" * 50)
    if all_passed:
        print("PASS: All test setup verifications passed!")
        print("The test infrastructure is ready for comprehensive testing.")
        return 0
    else:
        print("FAIL: Some test setup verifications failed.")
        print("Please check the errors above before running the full test suite.")
        return 1

if __name__ == "__main__":
    sys.exit(main())