"""
Integration tests for seed reproducibility across full simulation workflows.

This test suite validates that the complete competitive simulation system
produces reproducible results when using the same seed values, and that
all components properly integrate with seed-based randomization.
"""
import sys
import os
import unittest
from unittest.mock import patch, MagicMock
import tempfile
import shutil
import json
from datetime import datetime, timedelta

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from escape_room_sim.competitive.competitive_simulation import CompetitiveSimulation
import main


class TestSeedReproducibilityIntegration(unittest.TestCase):
    """Integration tests for seed reproducibility across complete simulation workflows."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_seed = 99999
    
    def tearDown(self):
        """Clean up test fixtures."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    # FAILING TESTS: End-to-End Seed Reproducibility
    
    def test_complete_simulation_reproducibility_with_identical_seeds(self):
        """Test that complete simulation workflow produces identical results with same seed."""
        seed = 12345
        
        # Run complete simulation twice with same seed
        sim1 = CompetitiveSimulation(seed=seed)
        sim2 = CompetitiveSimulation(seed=seed)
        
        results1 = sim1.run_complete_simulation(max_steps=50)
        results2 = sim2.run_complete_simulation(max_steps=50)
        
        # Core results should be identical
        self.assertEqual(results1['seed'], results2['seed'])
        self.assertEqual(results1['winner'], results2['winner'])
        self.assertEqual(results1['completion_reason'], results2['completion_reason'])
        self.assertEqual(results1['total_steps'], results2['total_steps'])
        
        # Action sequences should be identical
        self.assertEqual(len(results1['action_history']), len(results2['action_history']))
        
        for action1, action2 in zip(results1['action_history'], results2['action_history']):
            self.assertEqual(action1['agent'], action2['agent'])
            self.assertEqual(action1['action'], action2['action'])
            self.assertEqual(action1['parameters'], action2['parameters'])
    
    def test_different_seeds_produce_measurably_different_outcomes(self):
        """Test that different seeds produce statistically different simulation outcomes."""
        seed1 = 11111
        seed2 = 99999
        
        sim1 = CompetitiveSimulation(seed=seed1)
        sim2 = CompetitiveSimulation(seed=seed2)
        
        results1 = sim1.run_complete_simulation(max_steps=100)
        results2 = sim2.run_complete_simulation(max_steps=100)
        
        # Seeds should be different
        self.assertNotEqual(results1['seed'], results2['seed'])
        
        # At least one major outcome should be different
        outcomes_different = (
            results1['winner'] != results2['winner'] or
            results1['completion_reason'] != results2['completion_reason'] or
            abs(results1['total_steps'] - results2['total_steps']) > 5
        )
        self.assertTrue(outcomes_different, "Different seeds should produce different outcomes")
    
    def test_main_workflow_integration_with_seed_reproducibility(self):
        """Test that main.py workflow maintains seed reproducibility."""
        # Should fail initially - main.py doesn't have competitive simulation integration
        
        with patch.dict(os.environ, {'GEMINI_API_KEY': 'test_key'}):
            with patch('sys.argv', ['main.py', '--seed', '55555', '--competitive']):
                with patch('main.console.input', side_effect=['10', 'y', 'n']):  # Config inputs
                    
                    # Mock the competitive simulation to return consistent results
                    with patch('main.CompetitiveSimulation') as mock_sim_class:
                        mock_sim1 = MagicMock()
                        mock_sim1.run_complete_simulation.return_value = {
                            'seed': 55555,
                            'winner': 'strategist',
                            'total_steps': 42,
                            'completion_reason': 'escape_successful',
                            'simulation_duration': 125.5
                        }
                        
                        mock_sim2 = MagicMock()
                        mock_sim2.run_complete_simulation.return_value = {
                            'seed': 55555,
                            'winner': 'strategist',
                            'total_steps': 42,
                            'completion_reason': 'escape_successful',
                            'simulation_duration': 125.5
                        }
                        
                        mock_sim_class.side_effect = [mock_sim1, mock_sim2]
                        
                        # Run main workflow twice
                        result1 = main.main()
                        result2 = main.main()
                        
                        # Should produce identical results
                        self.assertEqual(result1, result2)

    # FAILING TESTS: Component Integration with Seeds
    
    def test_scenario_generator_seed_integration(self):
        """Test that scenario generator integrates properly with simulation seed."""
        seed = 77777
        
        sim1 = CompetitiveSimulation(seed=seed)
        sim2 = CompetitiveSimulation(seed=seed)
        
        scenario1 = sim1.generate_scenario()
        scenario2 = sim2.generate_scenario()
        
        # Scenarios should be identical with same seed
        self.assertEqual(scenario1.room_layout, scenario2.room_layout)
        self.assertEqual(scenario1.resource_distribution, scenario2.resource_distribution)
        self.assertEqual(len(scenario1.escape_methods), len(scenario2.escape_methods))
        
        # Escape methods should be in same order with same properties
        for method1, method2 in zip(scenario1.escape_methods, scenario2.escape_methods):
            self.assertEqual(method1.id, method2.id)
            self.assertEqual(method1.required_resources, method2.required_resources)
    
    def test_competitive_escape_room_seed_consistency(self):
        """Test that competitive escape room maintains seed consistency."""
        seed = 33333
        
        # Create two simulations with same seed
        sim1 = CompetitiveSimulation(seed=seed)
        sim2 = CompetitiveSimulation(seed=seed)
        
        # Generate identical scenarios
        scenario1 = sim1.generate_scenario()
        scenario2 = sim2.generate_scenario()
        
        # Initialize escape rooms
        room1 = sim1.initialize_escape_room()
        room2 = sim2.initialize_escape_room()
        
        # Rooms should have identical initial states
        self.assertEqual(room1.get_trust_relationships(), room2.get_trust_relationships())
        
        # Resource managers should have same available resources
        resources1 = room1.resource_manager.get_available_resources("strategist")
        resources2 = room2.resource_manager.get_available_resources("strategist")
        self.assertEqual(len(resources1), len(resources2))
    
    def test_agent_state_tracking_with_seeds(self):
        """Test that agent state tracking maintains consistency with seeds."""
        seed = 44444
        
        sim1 = CompetitiveSimulation(seed=seed)
        sim2 = CompetitiveSimulation(seed=seed)
        
        # Initialize both simulations
        sim1.generate_scenario()
        sim1.initialize_escape_room()
        sim2.generate_scenario()
        sim2.initialize_escape_room()
        
        # Perform identical actions on both simulations
        actions = [
            ("strategist", "claim_resource", {"resource_id": "key"}),
            ("mediator", "claim_resource", {"resource_id": "tool"}),
            ("survivor", "claim_resource", {"resource_id": "map"})
        ]
        
        for agent_id, action, params in actions:
            # Skip if resource doesn't exist (depends on scenario)
            try:
                result1 = sim1.run_simulation_step(agent_id, action, params)
                result2 = sim2.run_simulation_step(agent_id, action, params)
                self.assertEqual(result1['success'], result2['success'])
            except (ValueError, KeyError):
                # Resource might not exist in generated scenario
                pass
        
        # Agent states should be identical
        for agent_id in ["strategist", "mediator", "survivor"]:
            state1 = sim1.agent_states[agent_id].get_state_summary()
            state2 = sim2.agent_states[agent_id].get_state_summary()
            self.assertEqual(state1, state2)

    # FAILING TESTS: Performance and Stress Testing with Seeds
    
    def test_seed_reproducibility_under_different_step_limits(self):
        """Test that seed reproducibility holds under different simulation constraints."""
        seed = 66666
        
        # Run same seed with different step limits
        sim1 = CompetitiveSimulation(seed=seed)
        sim2 = CompetitiveSimulation(seed=seed)
        
        results1 = sim1.run_complete_simulation(max_steps=20)
        results2 = sim2.run_complete_simulation(max_steps=20)
        
        # Results should be identical up to the step limit
        self.assertEqual(results1['seed'], results2['seed'])
        self.assertEqual(results1['total_steps'], results2['total_steps'])
        
        # If both completed naturally (not due to step limit), all outcomes should match
        if (results1['completion_reason'] != 'max_steps_reached' and 
            results2['completion_reason'] != 'max_steps_reached'):
            self.assertEqual(results1['winner'], results2['winner'])
            self.assertEqual(results1['completion_reason'], results2['completion_reason'])
    
    def test_batch_simulation_reproducibility(self):
        """Test reproducibility across batch simulation runs."""
        # Should fail - batch simulation functionality doesn't exist yet
        
        seeds = [111, 222, 333, 444, 555]
        
        # Run batch simulations twice
        batch_results1 = main.run_batch_simulations(seeds, max_steps=30)
        batch_results2 = main.run_batch_simulations(seeds, max_steps=30)
        
        self.assertEqual(len(batch_results1), len(batch_results2))
        
        # Each corresponding simulation should have identical results
        for result1, result2 in zip(batch_results1, batch_results2):
            self.assertEqual(result1['seed'], result2['seed'])
            self.assertEqual(result1['winner'], result2['winner'])
            self.assertEqual(result1['total_steps'], result2['total_steps'])
    
    def test_seed_reproducibility_with_memory_persistence(self):
        """Test that seed reproducibility works with memory persistence enabled."""
        seed = 88888
        
        # Create temporary directories for memory storage
        memory_dir1 = os.path.join(self.temp_dir, 'memory1')
        memory_dir2 = os.path.join(self.temp_dir, 'memory2')
        os.makedirs(memory_dir1, exist_ok=True)
        os.makedirs(memory_dir2, exist_ok=True)
        
        # Run simulations with memory persistence
        with patch('main.create_seed_based_result_directory') as mock_create_dir:
            mock_create_dir.side_effect = [memory_dir1, memory_dir2]
            
            config1 = main.CompetitiveSimulationConfig(
                seed=seed,
                enable_memory=True,
                save_results=True
            )
            config2 = main.CompetitiveSimulationConfig(
                seed=seed,
                enable_memory=True,
                save_results=True
            )
            
            result1 = main.run_competitive_simulation_with_config(config1)
            result2 = main.run_competitive_simulation_with_config(config2)
            
            # Core simulation results should be identical despite memory persistence
            self.assertEqual(result1['seed'], result2['seed'])
            self.assertEqual(result1['winner'], result2['winner'])

    # FAILING TESTS: Error Handling and Edge Cases
    
    def test_seed_reproducibility_with_simulation_interruption(self):
        """Test that seed reproducibility handles simulation interruptions gracefully."""
        seed = 99999
        
        sim1 = CompetitiveSimulation(seed=seed)
        sim2 = CompetitiveSimulation(seed=seed)
        
        # Simulate interruption by limiting steps severely
        results1 = sim1.run_complete_simulation(max_steps=5)
        results2 = sim2.run_complete_simulation(max_steps=5)
        
        # Even with early termination, results should be identical
        self.assertEqual(results1['seed'], results2['seed'])
        self.assertEqual(results1['total_steps'], results2['total_steps'])
        self.assertEqual(results1['completion_reason'], results2['completion_reason'])
    
    def test_seed_boundary_value_reproducibility(self):
        """Test reproducibility with boundary seed values."""
        boundary_seeds = [0, 1, 999999, 2**31-1]  # Various boundary values
        
        for seed in boundary_seeds:
            with self.subTest(seed=seed):
                try:
                    sim1 = CompetitiveSimulation(seed=seed)
                    sim2 = CompetitiveSimulation(seed=seed)
                    
                    results1 = sim1.run_complete_simulation(max_steps=10)
                    results2 = sim2.run_complete_simulation(max_steps=10)
                    
                    # Should be reproducible even with boundary seeds
                    self.assertEqual(results1['seed'], results2['seed'])
                    self.assertEqual(results1['winner'], results2['winner'])
                    
                except ValueError as e:
                    # Some boundary values might be invalid - that's acceptable
                    self.assertIn('seed', str(e).lower())
    
    def test_cross_platform_seed_reproducibility(self):
        """Test that seed reproducibility works across different environments."""
        # This tests consistent behavior of random number generation
        seed = 12345
        
        # Simulate different environment conditions
        with patch('random.seed') as mock_seed:
            sim1 = CompetitiveSimulation(seed=seed)
            sim2 = CompetitiveSimulation(seed=seed)
            
            # Both should use the same seed
            expected_calls = [unittest.mock.call(seed), unittest.mock.call(seed)]
            mock_seed.assert_has_calls(expected_calls, any_order=True)
    
    def test_simulation_result_serialization_preserves_seed_info(self):
        """Test that result serialization preserves all seed-related information."""
        seed = 54321
        
        sim = CompetitiveSimulation(seed=seed)
        results = sim.run_complete_simulation(max_steps=25)
        
        # Serialize and deserialize results
        with patch('builtins.open', unittest.mock.mock_open()) as mock_file:
            main.save_simulation_results_with_metadata(results, self.temp_dir)
            
            # Extract written content
            written_content = ''.join(call.args[0] for call in mock_file().write.call_args_list)
            parsed_results = json.loads(written_content)
            
            # All seed information should be preserved
            self.assertEqual(parsed_results['seed'], seed)
            self.assertIn('metadata', parsed_results)
            self.assertIn('reproducibility_info', parsed_results['metadata'])
            
            # Should include timestamp for correlation
            self.assertIn('timestamp', parsed_results['metadata'])


if __name__ == '__main__':
    unittest.main()