"""
Test runner script for CrewAI Escape Room Agent tests.

Provides comprehensive test execution with coverage reporting,
performance metrics, and detailed test result analysis.
"""

import sys
import os
import subprocess
import time
from pathlib import Path


def run_tests():
    """Run all tests with coverage reporting."""
    print("🚀 Starting CrewAI Escape Room Agent Test Suite")
    print("=" * 60)
    
    # Get the project root directory
    project_root = Path(__file__).parent.parent
    os.chdir(project_root)
    
    # Start timing
    start_time = time.time()
    
    # Test categories to run
    test_categories = [
        ("Unit Tests - Strategist Agent", "tests/unit/test_strategist_agent.py"),
        ("Unit Tests - Mediator Agent", "tests/unit/test_mediator_agent.py"),
        ("Unit Tests - Survivor Agent", "tests/unit/test_survivor_agent.py"),
        ("Unit Tests - Edge Cases", "tests/unit/test_agent_edge_cases.py"),
        ("Integration Tests", "tests/integration/test_agent_integration.py"),
    ]
    
    results = {}
    
    for category_name, test_path in test_categories:
        print(f"\n📋 Running {category_name}")
        print("-" * 40)
        
        try:
            # Run pytest with coverage for this category
            cmd = [
                sys.executable, "-m", "pytest",
                test_path,
                "-v",
                "--tb=short",
                "--cov=src/escape_room_sim/agents",
                "--cov-report=term-missing",
                "--cov-report=html:tests/coverage_html",
                "--cov-append",
                "--durations=10"
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            results[category_name] = {
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr
            }
            
            if result.returncode == 0:
                print(f"✅ {category_name} - PASSED")
            else:
                print(f"❌ {category_name} - FAILED")
                print(f"Error: {result.stderr}")
                
        except Exception as e:
            print(f"💥 Error running {category_name}: {e}")
            results[category_name] = {"error": str(e)}
    
    # Run all tests together for final coverage report
    print(f"\n🎯 Running Complete Test Suite")
    print("-" * 40)
    
    try:
        cmd = [
            sys.executable, "-m", "pytest",
            "tests/",
            "-v",
            "--cov=src/escape_room_sim/agents",
            "--cov-report=term",
            "--cov-report=html:tests/coverage_html_final",
            "--cov-report=xml:tests/coverage.xml",
            "--durations=20",
            "--tb=short"
        ]
        
        final_result = subprocess.run(cmd, capture_output=True, text=True)
        
        print(final_result.stdout)
        if final_result.stderr:
            print("Warnings/Errors:")
            print(final_result.stderr)
            
    except Exception as e:
        print(f"💥 Error running complete test suite: {e}")
    
    # Calculate execution time
    end_time = time.time()
    execution_time = end_time - start_time
    
    # Print summary
    print("\n" + "=" * 60)
    print("📊 TEST EXECUTION SUMMARY")
    print("=" * 60)
    
    passed_categories = 0
    failed_categories = 0
    
    for category_name, result in results.items():
        if result.get("returncode") == 0:
            print(f"✅ {category_name}")
            passed_categories += 1
        else:
            print(f"❌ {category_name}")
            failed_categories += 1
    
    print(f"\n📈 Results: {passed_categories} passed, {failed_categories} failed")
    print(f"⏱️  Total execution time: {execution_time:.2f} seconds")
    print(f"📁 Coverage report: tests/coverage_html_final/index.html")
    
    if failed_categories > 0:
        print("\n⚠️  Some tests failed. Check the output above for details.")
        return 1
    else:
        print("\n🎉 All test categories passed successfully!")
        return 0


def run_specific_test(test_file):
    """Run a specific test file."""
    print(f"🎯 Running specific test: {test_file}")
    
    cmd = [
        sys.executable, "-m", "pytest",
        test_file,
        "-v",
        "--tb=long",
        "--cov=src/escape_room_sim/agents",
        "--cov-report=term"
    ]
    
    try:
        result = subprocess.run(cmd)
        return result.returncode
    except Exception as e:
        print(f"💥 Error running test: {e}")
        return 1


def check_test_coverage():
    """Check current test coverage without running tests."""
    print("📊 Checking Test Coverage")
    print("-" * 40)
    
    cmd = [
        sys.executable, "-m", "pytest",
        "tests/",
        "--cov=src/escape_room_sim/agents",
        "--cov-report=term",
        "--collect-only",
        "-q"
    ]
    
    try:
        result = subprocess.run(cmd)
        return result.returncode
    except Exception as e:
        print(f"💥 Error checking coverage: {e}")
        return 1


if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "coverage":
            sys.exit(check_test_coverage())
        else:
            # Run specific test file
            test_file = sys.argv[1]
            sys.exit(run_specific_test(test_file))
    else:
        # Run all tests
        sys.exit(run_tests())