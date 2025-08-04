# Implementation Plan

- [x] 1. Create failing tests for missing functions using existing test infrastructure





  - Add test cases to existing test files for missing context generation functions
  - Create new test files for RelationshipTracker and SurvivalMemoryBank classes
  - Utilize existing fixtures from conftest.py for consistent test patterns
  - Follow existing mock patterns for CrewAI dependencies
  - _Requirements: 1.1, 1.2, 1.3, 1.4_

- [x] 2. Implement context generation functions with TDD approach





  - [x] 2.1 Create failing tests for get_strategist_context_for_iteration function


    - Write test that calls function with various parameter combinations
    - Test edge cases like empty lists and None values
    - Verify output format contains required sections (iteration number, failures, resources)
    - _Requirements: 1.1_

  - [x] 2.2 Implement get_strategist_context_for_iteration function to pass tests


    - Write function that takes iteration_num, previous_failures, current_resources parameters
    - Return formatted string with strategic analysis context
    - Handle edge cases identified in tests
    - _Requirements: 1.1_

  - [x] 2.3 Create failing tests for get_mediator_context_for_iteration function


    - Write test that calls function with relationship tracker and team dynamics
    - Test with mock RelationshipTracker object
    - Verify output includes team dynamics and stress level information
    - _Requirements: 1.2_

  - [x] 2.4 Implement get_mediator_context_for_iteration function to pass tests


    - Write function that takes iteration_num, relationship_tracker, team_stress_level, previous_conflicts
    - Return formatted string with mediation context
    - Handle None relationship_tracker gracefully
    - _Requirements: 1.2_

  - [x] 2.5 Create failing tests for get_survivor_context_for_iteration function


    - Write test that calls function with survival memory and threat assessment
    - Test with mock SurvivalMemoryBank object
    - Verify output includes threat level and survival priorities
    - _Requirements: 1.3_

  - [x] 2.6 Implement get_survivor_context_for_iteration function to pass tests


    - Write function that takes iteration_num, survival_memory, current_threat_level, resource_status
    - Return formatted string with survival context
    - Handle None survival_memory gracefully
    - _Requirements: 1.3_

- [x] 3. Implement RelationshipTracker class with comprehensive TDD




  - [x] 3.1 Create failing tests for RelationshipTracker in new test file


    - Create tests/unit/test_relationship_tracker.py following existing test patterns
    - Use pytest fixtures and mock patterns from conftest.py
    - Test RelationshipTracker instantiation creates empty tracking system
    - Test get_relationship method creates new relationships when needed
    - Test relationship key standardization (alphabetical ordering)
    - _Requirements: 2.1_

  - [x] 3.2 Implement RelationshipTracker class constructor and basic methods


    - Create src/escape_room_sim/simulation/relationship_tracker.py
    - Implement RelationshipTracker class with __init__ method
    - Implement get_relationship method with alphabetical key ordering
    - Create internal data structures for relationship storage
    - _Requirements: 2.1_

  - [x] 3.3 Create failing tests for interaction recording functionality


    - Test record_interaction method updates trust levels correctly
    - Test collaboration recording increases trust by 0.1
    - Test conflict recording decreases trust by 0.05-0.1
    - Test interaction history is maintained
    - _Requirements: 2.2, 2.3, 2.4_

  - [x] 3.4 Implement interaction recording methods to pass tests


    - Write record_interaction method that updates relationship state
    - Implement record_successful_collaboration method
    - Implement record_conflict method with trust impact calculation
    - Ensure trust levels stay within 0.0-1.0 bounds
    - _Requirements: 2.2, 2.3, 2.4_

  - [x] 3.5 Create failing tests for team cohesion and summary functionality


    - Test get_team_cohesion returns value between 0.0 and 1.0
    - Test get_summary returns readable string with relationship states
    - Test export_data returns properly formatted dictionary
    - _Requirements: 2.5, 2.6_

  - [x] 3.6 Implement team analysis and export methods to pass tests


    - Write get_team_cohesion method that calculates average trust levels
    - Implement get_summary method that formats relationship descriptions
    - Create export_data method for persistence support
    - _Requirements: 2.5, 2.6_

- [ ] 4. Implement SurvivalMemoryBank class with TDD methodology
  - [ ] 4.1 Create failing tests for SurvivalMemoryBank in new test file
    - Create tests/unit/test_survival_memory.py following existing test patterns
    - Use pytest fixtures and AAA pattern from existing tests
    - Test SurvivalMemoryBank instantiation with default survival principles
    - Test record_close_call method stores high importance experiences (0.9)
    - Test record_successful_strategy stores moderate importance experiences (0.7)
    - _Requirements: 3.1, 3.2, 3.3_

  - [ ] 4.2 Implement SurvivalMemoryBank constructor and experience recording
    - Create src/escape_room_sim/simulation/survival_memory.py
    - Implement SurvivalMemoryBank class with default survival principles
    - Implement record_close_call method with importance score 0.9
    - Implement record_successful_strategy method with importance score 0.7
    - Create internal data structures for experience storage
    - _Requirements: 3.1, 3.2, 3.3_

  - [ ] 4.3 Create failing tests for threat assessment functionality
    - Test assess_current_threat returns ThreatAssessment with severity and probability
    - Test threat assessment uses historical experience data
    - Test mitigation strategies are based on past successes
    - _Requirements: 3.4_

  - [ ] 4.4 Implement threat assessment methods to pass tests
    - Write assess_current_threat method that analyzes historical patterns
    - Create ThreatAssessment dataclass with required fields
    - Implement logic to extract mitigation strategies from past experiences
    - _Requirements: 3.4_

  - [ ] 4.5 Create failing tests for experience retrieval and probability calculation
    - Test get_relevant_experiences returns up to 5 most important experiences
    - Test calculate_survival_probability returns value between 0.1 and 0.9
    - Test survival probability calculation uses historical success rates
    - _Requirements: 3.5, 3.6_

  - [ ] 4.6 Implement experience analysis and probability methods to pass tests
    - Write get_relevant_experiences method with importance-based sorting
    - Implement calculate_survival_probability using historical data
    - Create export_data method for persistence support
    - _Requirements: 3.5, 3.6_

- [ ] 5. Implement dynamic API configuration with test coverage
  - [ ] 5.1 Create failing tests for API configuration detection
    - Add tests to tests/unit/test_simple_engine.py (create if needed)
    - Use existing mock patterns and environment variable mocking
    - Test _get_memory_config with Gemini API key returns Gemini configuration
    - Test fallback to OpenAI when only OpenAI key available
    - Test fallback to local embeddings when only Anthropic key available
    - Test ValueError raised when no API keys available
    - _Requirements: 4.1, 4.2, 4.3, 4.4_

  - [ ] 5.2 Implement dynamic API configuration method to pass tests
    - Write _get_memory_config method that checks environment variables in priority order
    - Implement Gemini configuration with text-embedding-004
    - Implement OpenAI fallback with text-embedding-3-small
    - Implement Anthropic fallback with local sentence-transformers
    - Add clear error handling for missing API keys
    - _Requirements: 4.1, 4.2, 4.3, 4.4_

  - [ ] 5.3 Create failing tests for crew creation with dynamic configuration
    - Test _create_crew method uses dynamic memory configuration
    - Test crew creation succeeds with different API provider configurations
    - Test crew creation handles configuration errors gracefully
    - _Requirements: 4.5_

  - [ ] 5.4 Update crew creation method to use dynamic configuration
    - Modify _create_crew method to call _get_memory_config
    - Apply dynamic configuration to CrewAI Crew instantiation
    - Add error handling for configuration failures
    - _Requirements: 4.5_

- [ ] 6. Implement survival constraint enforcement with TDD
  - [ ] 6.1 Create failing tests for exit route capacity modifications
    - Add tests to tests/unit/test_escape_room_state.py (create if needed)
    - Follow existing test patterns with AAA structure
    - Test main door capacity is set to 2 agents maximum
    - Test vent shaft capacity is set to 1 agent maximum
    - Test window and hidden passage capacities allow 2 agents
    - _Requirements: 5.1, 5.2_

  - [ ] 6.2 Modify exit route definitions to enforce survival constraints
    - Update main_door capacity from 3 to 2 in escape_room_state.py
    - Verify vent_shaft capacity is 1 agent maximum
    - Ensure other exit routes have appropriate capacity limits
    - _Requirements: 5.1, 5.2_

  - [ ] 6.3 Create failing tests for survival scenario evaluation
    - Test evaluate_survival_scenarios generates all 2-agent combinations
    - Test scenarios include moral difficulty scores for sacrifice decisions
    - Test scenarios are sorted by success probability
    - Test escape probability returns 0.0 when agent count exceeds capacity
    - _Requirements: 5.3, 5.4, 5.5_

  - [ ] 6.4 Implement survival scenario evaluation methods to pass tests
    - Write evaluate_survival_scenarios method that generates 2-agent combinations
    - Implement _calculate_escape_probability with capacity checking
    - Implement _calculate_moral_difficulty with stress and time factors
    - Add scenario sorting by success probability
    - _Requirements: 5.3, 5.4, 5.5_

- [ ] 7. Implement comprehensive safety measures with test coverage
  - [ ] 7.1 Create failing tests for infinite loop protection
    - Test simulation enforces maximum total time of 30 minutes
    - Test stagnation detection stops simulation after 5 iterations without progress
    - Test progress hash calculation includes key state elements
    - _Requirements: 6.1, 6.2, 6.4_

  - [ ] 7.2 Implement safety measures in simulation engine to pass tests
    - Add maximum execution time check in run_full_simulation
    - Implement stagnation detection with progress hash comparison
    - Create _get_progress_hash method with iteration count, time, puzzles, resources
    - Add safety counters and termination logic
    - _Requirements: 6.1, 6.2, 6.4_

  - [ ] 7.3 Create failing tests for error handling and graceful termination
    - Test simulation catches exceptions and stops gracefully
    - Test stopping conditions return both decision and reason
    - Test final report includes termination reason
    - _Requirements: 6.3, 6.5, 6.6_

  - [ ] 7.4 Implement comprehensive error handling to pass tests
    - Add try-catch blocks around iteration execution
    - Implement check_stopping_conditions with reason strings
    - Update final report generation to include stop reasons
    - Add graceful termination for all error scenarios
    - _Requirements: 6.3, 6.5, 6.6_

- [ ] 8. Create integration tests and validate complete system functionality
  - [ ] 8.1 Create integration tests for full simulation execution
    - Add tests to tests/integration/test_critical_fixes_integration.py
    - Follow existing integration test patterns from test_agent_integration.py
    - Write test that runs complete simulation without NameError exceptions
    - Test simulation with different API provider configurations
    - Test memory systems work together correctly
    - Test survival constraints are properly enforced during simulation
    - _Requirements: 7.1, 7.2, 7.4, 7.5_

  - [ ] 8.2 Run integration tests using existing test infrastructure
    - Execute tests using python tests/run_all_tests.py
    - Use existing coverage reporting with python tests/run_all_tests.py coverage
    - Debug and fix any integration issues discovered
    - Verify all missing functions are callable without errors
    - Confirm survival constraint enforcement works in practice
    - _Requirements: 7.1, 7.2, 7.4, 7.5_

  - [ ] 8.3 Create performance and safety validation tests
    - Add performance tests to existing test structure
    - Test simulation completes within safety time limits
    - Test stagnation detection prevents infinite loops
    - Test memory usage remains reasonable during execution
    - Test error handling prevents crashes
    - _Requirements: 7.6, 7.7_

  - [ ] 8.4 Validate all acceptance criteria using existing test runners
    - Run python tests/run_all_tests.py to verify all tests pass
    - Use python tests/run_all_tests.py stats to check test coverage
    - Verify all missing functions implemented and callable
    - Confirm all missing classes implemented and instantiable
    - Test API configuration works with available providers
    - Validate survival constraint properly enforces "only two can survive"
    - Confirm simulation runs without NameError or AttributeError exceptions
    - Verify safety measures prevent infinite loops and stagnation
    - Ensure all existing tests continue to pass
    - _Requirements: 7.6, 7.7_