"""
Tests for main.py competitive simulation integration with seed parameter support.

This test suite implements TDD methodology for Task #13 - integrating seed parameter
into main simulation interface, replacing collaborative simulation with competitive
survival mechanics, and ensuring full reproducibility.
"""
import sys
import os
import unittest
from unittest.mock import patch, MagicMock, call
import tempfile
import shutil
from datetime import datetime
import argparse

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

# Import main module functions
import main
from escape_room_sim.competitive.competitive_simulation import CompetitiveSimulation


class TestMainCompetitiveIntegration(unittest.TestCase):
    """Test competitive simulation integration in main.py."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.original_argv = sys.argv.copy()
    
    def tearDown(self):
        """Clean up test fixtures."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
        sys.argv = self.original_argv

    # FAILING TESTS: Command Line Seed Parameter Support
    
    def test_main_accepts_seed_parameter_from_command_line(self):
        """Test that main.py accepts --seed parameter from command line."""
        # This should fail initially - main.py doesn't have seed parameter support
        sys.argv = ['main.py', '--seed', '12345']
        
        with patch('main.run_competitive_simulation') as mock_run:
            mock_run.return_value = {'seed': 12345, 'winner': 'strategist'}
            
            # This should parse seed from command line and pass to simulation
            result = main.parse_command_line_args()
            self.assertEqual(result.seed, 12345)
    
    def test_main_accepts_optional_seed_parameter(self):
        """Test that seed parameter is optional in command line."""
        sys.argv = ['main.py']  # No seed provided
        
        result = main.parse_command_line_args()
        self.assertIsNone(result.seed)  # Should be None when not provided
    
    def test_main_validates_seed_parameter_type(self):
        """Test that main.py validates seed parameter is integer."""
        sys.argv = ['main.py', '--seed', 'invalid_seed']
        
        with self.assertRaises(SystemExit):  # argparse exits on invalid input
            main.parse_command_line_args()
    
    def test_main_accepts_negative_seed_parameter(self):
        """Test that main.py accepts negative seed values."""
        sys.argv = ['main.py', '--seed', '-999']
        
        result = main.parse_command_line_args()
        self.assertEqual(result.seed, -999)

    # FAILING TESTS: Simulation Configuration with Seed Handling
    
    def test_get_competitive_simulation_config_includes_seed(self):
        """Test that competitive simulation config includes seed parameter."""
        with patch('main.console.input', side_effect=['10', 'y', 'y']):
            config = main.get_competitive_simulation_config(seed=42)
            
            self.assertEqual(config.seed, 42)
            self.assertEqual(config.max_iterations, 10)
            self.assertTrue(config.enable_memory)
            self.assertTrue(config.verbose_output)
    
    def test_get_competitive_simulation_config_generates_seed_when_none(self):
        """Test that config generates random seed when none provided."""
        with patch('main.console.input', side_effect=['5', 'n', 'n']):
            config = main.get_competitive_simulation_config(seed=None)
            
            self.assertIsInstance(config.seed, int)
            self.assertGreaterEqual(config.seed, 0)
            self.assertLessEqual(config.seed, 999999)
    
    def test_competitive_simulation_config_dataclass_structure(self):
        """Test that CompetitiveSimulationConfig has required fields."""
        # Should fail - CompetitiveSimulationConfig doesn't exist yet
        from main import CompetitiveSimulationConfig
        
        config = CompetitiveSimulationConfig(
            seed=123,
            max_iterations=10,
            enable_memory=True,
            verbose_output=True,
            save_results=True
        )
        
        self.assertEqual(config.seed, 123)
        self.assertEqual(config.max_iterations, 10)
        self.assertTrue(config.enable_memory)
        self.assertTrue(config.verbose_output)
        self.assertTrue(config.save_results)

    # FAILING TESTS: Seed Logging and Result Correlation
    
    def test_simulation_logs_seed_at_start(self):
        """Test that simulation logs seed parameter at startup."""
        with patch('main.console') as mock_console:
            with patch('main.run_competitive_simulation') as mock_run:
                mock_run.return_value = {'seed': 42, 'winner': 'mediator'}
                
                main.run_simulation_with_seed(42)
                
                # Should log seed value at start
                mock_console.print.assert_any_call(
                    "[bold]Simulation Seed: 42[/bold]"
                )
    
    def test_simulation_saves_seed_in_results_file(self):
        """Test that simulation saves seed in results file for correlation."""
        with patch('main.CompetitiveSimulation') as mock_sim_class:
            mock_sim = MagicMock()
            mock_sim.run_complete_simulation.return_value = {
                'seed': 99999,
                'winner': 'survivor',
                'completion_reason': 'escape_successful'
            }
            mock_sim_class.return_value = mock_sim
            
            with patch('builtins.open', unittest.mock.mock_open()) as mock_file:
                main.run_and_save_simulation_results(99999, self.temp_dir)
                
                # Should save seed in results file
                mock_file.assert_called()
                written_content = ''.join(call.args[0] for call in mock_file().write.call_args_list)
                self.assertIn('"seed": 99999', written_content)
    
    def test_simulation_creates_seed_correlated_filename(self):
        """Test that results file includes seed in filename for correlation."""
        with patch('main.CompetitiveSimulation') as mock_sim_class:
            mock_sim = MagicMock()
            mock_sim.run_complete_simulation.return_value = {'seed': 77777}
            mock_sim_class.return_value = mock_sim
            
            with patch('builtins.open', unittest.mock.mock_open()) as mock_file:
                main.run_and_save_simulation_results(77777, self.temp_dir)
                
                # Filename should include seed
                filename_used = mock_file.call_args[0][0]
                self.assertIn('seed_77777', filename_used)

    # FAILING TESTS: Automatic Seed Generation
    
    def test_automatic_seed_generation_when_none_provided(self):
        """Test automatic seed generation when none provided."""
        # Should fail - generate_random_seed function doesn't exist
        seed = main.generate_random_seed()
        
        self.assertIsInstance(seed, int)
        self.assertGreaterEqual(seed, 0)
        self.assertLessEqual(seed, 999999)
    
    def test_automatic_seed_generation_is_different_each_time(self):
        """Test that automatic seed generation produces different seeds."""
        seed1 = main.generate_random_seed()
        seed2 = main.generate_random_seed()
        
        # Should be different (with very high probability)
        self.assertNotEqual(seed1, seed2)
    
    def test_automatic_seed_logged_when_generated(self):
        """Test that automatically generated seed is logged."""
        with patch('main.console') as mock_console:
            with patch('main.random.randint', return_value=555555):
                seed = main.generate_random_seed()
                
                # Should log the generated seed
                mock_console.print.assert_called_with(
                    "[yellow]No seed provided, generated random seed: 555555[/yellow]"
                )

    # FAILING TESTS: Seed-Based Result Comparison Utilities
    
    def test_compare_simulation_results_by_seed(self):
        """Test utility to compare simulation results by seed."""
        # Should fail - compare_simulation_results function doesn't exist
        results1 = {'seed': 123, 'winner': 'strategist', 'total_steps': 50}
        results2 = {'seed': 123, 'winner': 'strategist', 'total_steps': 50}
        results3 = {'seed': 456, 'winner': 'mediator', 'total_steps': 75}
        
        comparison = main.compare_simulation_results([results1, results2, results3])
        
        # Should group by seed and identify reproducibility
        self.assertEqual(len(comparison['seed_groups']), 2)
        self.assertIn(123, comparison['seed_groups'])
        self.assertIn(456, comparison['seed_groups'])
        self.assertTrue(comparison['seed_groups'][123]['reproducible'])
    
    def test_analyze_seed_reproducibility(self):
        """Test analysis of seed reproducibility across multiple runs."""
        results = [
            {'seed': 100, 'winner': 'strategist', 'total_steps': 25},
            {'seed': 100, 'winner': 'strategist', 'total_steps': 25},  # Same seed, same result
            {'seed': 200, 'winner': 'mediator', 'total_steps': 30},
            {'seed': 200, 'winner': 'survivor', 'total_steps': 45},   # Same seed, different result
        ]
        
        analysis = main.analyze_seed_reproducibility(results)
        
        self.assertEqual(analysis['reproducible_seeds'], [100])
        self.assertEqual(analysis['non_reproducible_seeds'], [200])
        self.assertEqual(analysis['reproducibility_rate'], 0.5)
    
    def test_find_similar_simulation_outcomes(self):
        """Test utility to find simulations with similar outcomes."""
        results = [
            {'seed': 1, 'winner': 'strategist', 'total_steps': 20, 'completion_reason': 'escape_successful'},
            {'seed': 2, 'winner': 'strategist', 'total_steps': 22, 'completion_reason': 'escape_successful'},
            {'seed': 3, 'winner': 'mediator', 'total_steps': 45, 'completion_reason': 'max_steps_reached'},
        ]
        
        similar = main.find_similar_simulation_outcomes(results, similarity_threshold=0.8)
        
        # Should find seeds 1 and 2 as similar (same winner, similar steps)
        self.assertEqual(len(similar), 1)
        self.assertIn(1, similar[0]['seeds'])
        self.assertIn(2, similar[0]['seeds'])

    # FAILING TESTS: Integration Tests for Full Seed Reproducibility
    
    def test_full_simulation_reproducibility_with_same_seed(self):
        """Test that full simulation produces identical results with same seed."""
        seed = 12345
        
        # Run simulation twice with same seed
        with patch.dict(os.environ, {'GEMINI_API_KEY': 'test_key'}):
            with patch('main.CompetitiveSimulation') as mock_sim_class:
                # Configure mock to return identical results
                mock_sim1 = MagicMock()
                mock_sim1.run_complete_simulation.return_value = {
                    'seed': seed,
                    'winner': 'strategist',
                    'total_steps': 42,
                    'completion_reason': 'escape_successful'
                }
                
                mock_sim2 = MagicMock()
                mock_sim2.run_complete_simulation.return_value = {
                    'seed': seed,
                    'winner': 'strategist',
                    'total_steps': 42,
                    'completion_reason': 'escape_successful'
                }
                
                mock_sim_class.side_effect = [mock_sim1, mock_sim2]
                
                result1 = main.run_competitive_simulation_with_seed(seed)
                result2 = main.run_competitive_simulation_with_seed(seed)
                
                # Results should be identical
                self.assertEqual(result1['winner'], result2['winner'])
                self.assertEqual(result1['total_steps'], result2['total_steps'])
                self.assertEqual(result1['completion_reason'], result2['completion_reason'])
    
    def test_different_seeds_produce_different_results(self):
        """Test that different seeds produce different simulation results."""
        with patch.dict(os.environ, {'GEMINI_API_KEY': 'test_key'}):
            with patch('main.CompetitiveSimulation') as mock_sim_class:
                # Configure mocks to return different results for different seeds
                mock_sim1 = MagicMock()
                mock_sim1.run_complete_simulation.return_value = {
                    'seed': 111,
                    'winner': 'strategist',
                    'total_steps': 30
                }
                
                mock_sim2 = MagicMock()
                mock_sim2.run_complete_simulation.return_value = {
                    'seed': 222,
                    'winner': 'survivor',
                    'total_steps': 65
                }
                
                mock_sim_class.side_effect = [mock_sim1, mock_sim2]
                
                result1 = main.run_competitive_simulation_with_seed(111)
                result2 = main.run_competitive_simulation_with_seed(222)
                
                # Results should be different
                self.assertNotEqual(result1['winner'], result2['winner'])
                self.assertNotEqual(result1['total_steps'], result2['total_steps'])
    
    def test_main_function_integration_with_competitive_simulation(self):
        """Test that main() function integrates competitive simulation correctly."""
        sys.argv = ['main.py', '--seed', '99999', '--competitive']
        
        with patch.dict(os.environ, {'GEMINI_API_KEY': 'test_key'}):
            with patch('main.console.input', side_effect=['5', 'y', 'y']):
                with patch('main.CompetitiveSimulation') as mock_sim_class:
                    mock_sim = MagicMock()
                    mock_sim.run_complete_simulation.return_value = {
                        'seed': 99999,
                        'winner': 'mediator',
                        'completion_reason': 'escape_successful'
                    }
                    mock_sim_class.return_value = mock_sim
                    
                    with patch('main.console') as mock_console:
                        # Should run without errors and use competitive simulation
                        main.main()
                        
                        # Should display competitive simulation results
                        mock_console.print.assert_any_call(
                            "[bold green]COMPETITIVE SIMULATION COMPLETE![/bold green]"
                        )

    # FAILING TESTS: Main.py Replacement of Collaborative Simulation
    
    def test_main_defaults_to_competitive_simulation(self):
        """Test that main.py defaults to competitive simulation instead of collaborative."""
        sys.argv = ['main.py']
        
        with patch.dict(os.environ, {'GEMINI_API_KEY': 'test_key'}):
            with patch('main.console.input', side_effect=['3', 'n', 'n']):
                with patch('main.CompetitiveSimulation') as mock_competitive_sim:
                    with patch('main.EscapeRoomSimulation') as mock_collaborative_sim:
                        mock_competitive_sim.return_value.run_complete_simulation.return_value = {
                            'seed': 123, 'winner': 'strategist'
                        }
                        
                        main.main()
                        
                        # Should use competitive simulation, not collaborative
                        mock_competitive_sim.assert_called_once()
                        mock_collaborative_sim.assert_not_called()
    
    def test_main_prints_competitive_welcome_message(self):
        """Test that main.py prints competitive simulation welcome message."""
        with patch('main.console') as mock_console:
            main.print_competitive_welcome()
            
            # Should print competitive-specific welcome message
            mock_console.print.assert_called_once()
            welcome_panel = mock_console.print.call_args[0][0]
            self.assertIn("Competitive Survival Simulation", str(welcome_panel))
            self.assertIn("Only one agent can survive", str(welcome_panel))


if __name__ == '__main__':
    unittest.main()