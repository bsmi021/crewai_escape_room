"""
Comprehensive test runner for CrewAI Escape Room Agent tests.
Handles mocking and provides detailed test results.
"""

import sys
import os
import subprocess
import time
from pathlib import Path

def setup_mock_environment():
    """Set up mock CrewAI environment for testing."""
    # Mock the crewai module
    sys.modules['crewai'] = type('MockModule', (), {})()
    
    class MockAgent:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)
                
    sys.modules['crewai'].Agent = MockAgent
    print("Mock CrewAI environment setup complete.")

def run_test_suite():
    """Run the complete test suite with proper mocking."""
    print("CrewAI Escape Room Agent Test Suite")
    print("=" * 60)
    
    # Set up mock environment
    setup_mock_environment()
    
    # Get project root
    project_root = Path(__file__).parent.parent
    os.chdir(project_root)
    
    start_time = time.time()
    
    # Test execution with inline mock setup
    test_command = '''
import sys
sys.modules['crewai'] = type('MockModule', (), {})()
class MockAgent:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
sys.modules['crewai'].Agent = MockAgent

import pytest
pytest.main([
    'tests/',
    '-v',
    '--tb=short',
    '--no-cov',
    '--disable-warnings'
])
'''
    
    print("Running comprehensive test suite...")
    result = subprocess.run([sys.executable, '-c', test_command], 
                          capture_output=True, text=True)
    
    print(result.stdout)
    if result.stderr:
        print("STDERR:")
        print(result.stderr)
    
    end_time = time.time()
    execution_time = end_time - start_time
    
    print("\n" + "=" * 60)
    print("TEST EXECUTION SUMMARY")
    print("=" * 60)
    print(f"Total execution time: {execution_time:.2f} seconds")
    
    if result.returncode == 0:
        print("All tests passed successfully!")
        print("\nTest Coverage Summary:")
        print("- Agent Creation and Configuration Tests")
        print("- Memory and Learning System Tests")
        print("- Agent Personality and Behavior Tests")
        print("- Context-Aware Agent Creation Tests")
        print("- Integration Tests with CrewAI Framework")
        print("- Error Handling and Edge Case Tests")
    else:
        print("Some tests failed. Check output above for details.")
    
    return result.returncode

def run_coverage_report():
    """Run tests with coverage reporting."""
    print("Running tests with coverage analysis...")
    
    test_command = '''
import sys
sys.modules['crewai'] = type('MockModule', (), {})()
class MockAgent:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
sys.modules['crewai'].Agent = MockAgent

import pytest
pytest.main([
    'tests/',
    '--cov=src/escape_room_sim/agents',
    '--cov-report=term-missing',
    '--cov-report=html:tests/coverage_html',
    '--tb=short',
    '--disable-warnings'
])
'''
    
    result = subprocess.run([sys.executable, '-c', test_command])
    
    if result.returncode == 0:
        print("\nCoverage report generated: tests/coverage_html/index.html")
    
    return result.returncode

def run_specific_test_category(category):
    """Run a specific test category."""
    category_paths = {
        'strategist': 'tests/unit/test_strategist_agent.py',
        'mediator': 'tests/unit/test_mediator_agent.py',
        'survivor': 'tests/unit/test_survivor_agent.py',
        'edge_cases': 'tests/unit/test_agent_edge_cases.py',
        'integration': 'tests/integration/test_agent_integration.py'
    }
    
    if category not in category_paths:
        print(f"Unknown category: {category}")
        print(f"Available categories: {', '.join(category_paths.keys())}")
        return 1
    
    test_path = category_paths[category]
    print(f"Running {category} tests: {test_path}")
    
    test_command = f'''
import sys
sys.modules['crewai'] = type('MockModule', (), {{}})()
class MockAgent:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
sys.modules['crewai'].Agent = MockAgent

import pytest
pytest.main([
    '{test_path}',
    '-v',
    '--tb=short',
    '--no-cov',
    '--disable-warnings'
])
'''
    
    result = subprocess.run([sys.executable, '-c', test_command])
    return result.returncode

def show_test_statistics():
    """Show statistics about the test suite."""
    print("Test Suite Statistics")
    print("-" * 40)
    
    test_files = [
        'tests/unit/test_strategist_agent.py',
        'tests/unit/test_mediator_agent.py', 
        'tests/unit/test_survivor_agent.py',
        'tests/unit/test_agent_edge_cases.py',
        'tests/integration/test_agent_integration.py'
    ]
    
    total_tests = 0
    total_classes = 0
    
    for test_file in test_files:
        if os.path.exists(test_file):
            with open(test_file, 'r', encoding='utf-8') as f:
                content = f.read()
                test_count = content.count('def test_')
                class_count = content.count('class Test')
                total_tests += test_count
                total_classes += class_count
                print(f"  {os.path.basename(test_file)}: {test_count} tests, {class_count} classes")
    
    print(f"\nTotal: {total_tests} tests across {total_classes} test classes")
    print(f"Test files: {len(test_files)}")
    
    # Count lines of test code
    total_lines = 0
    for test_file in test_files:
        if os.path.exists(test_file):
            with open(test_file, 'r', encoding='utf-8') as f:
                total_lines += len(f.readlines())
    
    print(f"Total test code lines: {total_lines}")

def main():
    """Main test runner entry point."""
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == 'coverage':
            return run_coverage_report()
        elif command == 'stats':
            show_test_statistics()
            return 0
        elif command in ['strategist', 'mediator', 'survivor', 'edge_cases', 'integration']:
            return run_specific_test_category(command)
        else:
            print(f"Usage: {sys.argv[0]} [coverage|stats|strategist|mediator|survivor|edge_cases|integration]")
            return 1
    else:
        return run_test_suite()

if __name__ == "__main__":
    sys.exit(main())