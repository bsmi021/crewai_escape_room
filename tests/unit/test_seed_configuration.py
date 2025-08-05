"""
Tests for seed-based configuration and reproducibility utilities.

This test suite covers advanced seed handling functionality including
configuration management, result correlation, and reproducibility analysis.
"""
import sys
import os
import unittest
from unittest.mock import patch, MagicMock
import tempfile
import shutil
import json
from datetime import datetime

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

# Import main module functions - these should fail initially
import main


class TestSeedConfiguration(unittest.TestCase):
    """Test seed-based configuration functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test fixtures."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    # FAILING TESTS: Seed Configuration Management
    
    def test_competitive_simulation_config_with_seed(self):
        """Test CompetitiveSimulationConfig dataclass with seed parameter."""
        # Should fail - CompetitiveSimulationConfig doesn't exist yet
        config = main.CompetitiveSimulationConfig(
            seed=42,
            max_iterations=15,
            enable_memory=True,
            verbose_output=False,
            save_results=True,
            time_limit=300
        )
        
        self.assertEqual(config.seed, 42)
        self.assertEqual(config.max_iterations, 15)
        self.assertTrue(config.enable_memory)
        self.assertFalse(config.verbose_output)
        self.assertTrue(config.save_results)
        self.assertEqual(config.time_limit, 300)
    
    def test_config_validation_for_seed_values(self):
        """Test that configuration validates seed values properly."""
        # Valid positive seed
        config1 = main.CompetitiveSimulationConfig(seed=12345)
        self.assertEqual(config1.seed, 12345)
        
        # Valid zero seed
        config2 = main.CompetitiveSimulationConfig(seed=0)
        self.assertEqual(config2.seed, 0)
        
        # Should validate negative seeds based on requirements
        with self.assertRaises(ValueError):
            main.CompetitiveSimulationConfig(seed=-1)
    
    def test_config_defaults_when_seed_none(self):
        """Test that configuration handles None seed by generating one."""
        config = main.CompetitiveSimulationConfig(seed=None)
        
        # Should generate a valid seed
        self.assertIsInstance(config.seed, int)
        self.assertGreaterEqual(config.seed, 0)
        self.assertLessEqual(config.seed, 999999)
    
    def test_config_serialization_includes_seed(self):
        """Test that configuration can be serialized with seed included."""
        config = main.CompetitiveSimulationConfig(seed=77777, max_iterations=10)
        
        serialized = main.serialize_config(config)
        deserialized = main.deserialize_config(serialized)
        
        self.assertEqual(deserialized.seed, 77777)
        self.assertEqual(deserialized.max_iterations, 10)

    # FAILING TESTS: Result File Management with Seeds
    
    def test_save_simulation_results_with_seed_metadata(self):
        """Test saving simulation results with comprehensive seed metadata."""
        results = {
            'seed': 12345,
            'winner': 'strategist',
            'total_steps': 50,
            'completion_reason': 'escape_successful',
            'simulation_duration': 125.5,
            'timestamp': datetime.now().isoformat()
        }
        
        with patch('builtins.open', unittest.mock.mock_open()) as mock_file:
            main.save_simulation_results_with_metadata(results, self.temp_dir)
            
            # Should save with seed in filename
            expected_filename = os.path.join(self.temp_dir, 'competitive_simulation_seed_12345_results.json')
            mock_file.assert_called_with(expected_filename, 'w')
            
            # Should include metadata section
            written_content = ''.join(call.args[0] for call in mock_file().write.call_args_list)
            parsed_content = json.loads(written_content)
            
            self.assertEqual(parsed_content['seed'], 12345)
            self.assertIn('metadata', parsed_content)
            self.assertIn('reproducibility_info', parsed_content['metadata'])
    
    def test_load_simulation_results_by_seed(self):
        """Test loading simulation results filtered by seed value."""
        # Create test result files
        results1 = {'seed': 111, 'winner': 'strategist'}
        results2 = {'seed': 222, 'winner': 'mediator'}
        results3 = {'seed': 111, 'winner': 'strategist'}  # Same seed
        
        with patch('os.listdir') as mock_listdir:
            with patch('builtins.open', unittest.mock.mock_open()) as mock_file:
                mock_listdir.return_value = [
                    'competitive_simulation_seed_111_results.json',
                    'competitive_simulation_seed_222_results.json',
                    'competitive_simulation_seed_111_results_2.json'
                ]
                
                # Mock file contents
                mock_file.return_value.read.side_effect = [
                    json.dumps(results1),
                    json.dumps(results3)  # Only seed 111 files
                ]
                
                loaded_results = main.load_simulation_results_by_seed(self.temp_dir, seed=111)
                
                self.assertEqual(len(loaded_results), 2)
                self.assertTrue(all(result['seed'] == 111 for result in loaded_results))
    
    def test_create_seed_based_result_directory(self):
        """Test creating organized directory structure based on seeds."""
        seed = 54321
        
        result_dir = main.create_seed_based_result_directory(self.temp_dir, seed)
        
        expected_dir = os.path.join(self.temp_dir, f'seed_{seed}')
        self.assertEqual(result_dir, expected_dir)
        
        # Directory should be created
        with patch('os.makedirs') as mock_makedirs:
            main.create_seed_based_result_directory(self.temp_dir, seed)
            mock_makedirs.assert_called_with(expected_dir, exist_ok=True)

    # FAILING TESTS: Reproducibility Analysis
    
    def test_calculate_reproducibility_metrics(self):
        """Test calculation of reproducibility metrics across multiple runs."""
        results = [
            {'seed': 100, 'winner': 'strategist', 'total_steps': 25, 'completion_reason': 'escape_successful'},
            {'seed': 100, 'winner': 'strategist', 'total_steps': 25, 'completion_reason': 'escape_successful'},
            {'seed': 200, 'winner': 'mediator', 'total_steps': 40, 'completion_reason': 'max_steps_reached'},
            {'seed': 200, 'winner': 'survivor', 'total_steps': 45, 'completion_reason': 'escape_successful'},
            {'seed': 300, 'winner': 'strategist', 'total_steps': 30, 'completion_reason': 'escape_successful'}
        ]
        
        metrics = main.calculate_reproducibility_metrics(results)
        
        self.assertEqual(metrics['total_unique_seeds'], 3)
        self.assertEqual(metrics['total_runs'], 5)
        self.assertEqual(metrics['seeds_with_multiple_runs'], 2)  # Seeds 100 and 200
        self.assertEqual(metrics['fully_reproducible_seeds'], 1)  # Only seed 100
        self.assertEqual(metrics['reproducibility_rate'], 1/2)  # 1 out of 2 multi-run seeds
    
    def test_analyze_seed_outcome_variance(self):
        """Test analysis of outcome variance for seeds with multiple runs."""
        results = [
            {'seed': 123, 'winner': 'strategist', 'total_steps': 20},
            {'seed': 123, 'winner': 'strategist', 'total_steps': 22},  # Slight variance
            {'seed': 456, 'winner': 'strategist', 'total_steps': 30},
            {'seed': 456, 'winner': 'mediator', 'total_steps': 45},   # High variance
        ]
        
        variance_analysis = main.analyze_seed_outcome_variance(results)
        
        self.assertIn(123, variance_analysis)
        self.assertIn(456, variance_analysis)
        
        # Seed 123 should have low variance (same winner, similar steps)
        self.assertLess(variance_analysis[123]['winner_consistency'], 1.0)
        self.assertLess(variance_analysis[123]['step_variance'], variance_analysis[456]['step_variance'])
        
        # Seed 456 should have high variance (different winners)
        self.assertEqual(variance_analysis[456]['winner_consistency'], 0.0)  # Different winners
    
    def test_generate_reproducibility_report(self):
        """Test generation of comprehensive reproducibility report."""
        results = [
            {'seed': 1, 'winner': 'strategist', 'total_steps': 25, 'simulation_duration': 120.0},
            {'seed': 1, 'winner': 'strategist', 'total_steps': 25, 'simulation_duration': 125.0},
            {'seed': 2, 'winner': 'mediator', 'total_steps': 50, 'simulation_duration': 200.0}
        ]
        
        report = main.generate_reproducibility_report(results)
        
        # Should include summary statistics
        self.assertIn('summary', report)
        self.assertIn('seed_analysis', report)
        self.assertIn('recommendations', report)
        
        # Summary should have key metrics
        summary = report['summary']
        self.assertEqual(summary['total_runs'], 3)
        self.assertEqual(summary['unique_seeds'], 2)
        self.assertIn('reproducibility_rate', summary)
        
        # Should include recommendations for improving reproducibility
        self.assertIsInstance(report['recommendations'], list)
        self.assertGreater(len(report['recommendations']), 0)

    # FAILING TESTS: Seed-Based Performance Analysis
    
    def test_analyze_performance_by_seed_range(self):
        """Test performance analysis grouped by seed ranges."""
        results = [
            {'seed': 100, 'total_steps': 20, 'simulation_duration': 100.0},
            {'seed': 150, 'total_steps': 25, 'simulation_duration': 120.0},
            {'seed': 900, 'total_steps': 45, 'simulation_duration': 200.0},
            {'seed': 950, 'total_steps': 50, 'simulation_duration': 220.0}
        ]
        
        analysis = main.analyze_performance_by_seed_range(results, range_size=500)
        
        # Should group into ranges: 0-499 and 500-999
        self.assertIn('0-499', analysis)
        self.assertIn('500-999', analysis)
        
        # Each range should have performance statistics
        range_0_499 = analysis['0-499']
        self.assertEqual(range_0_499['run_count'], 2)
        self.assertEqual(range_0_499['avg_steps'], 22.5)  # (20+25)/2
        
        range_500_999 = analysis['500-999']
        self.assertEqual(range_500_999['run_count'], 2)
        self.assertEqual(range_500_999['avg_steps'], 47.5)  # (45+50)/2
    
    def test_find_optimal_seed_ranges(self):
        """Test identification of seed ranges with optimal performance."""
        results = [
            {'seed': 100, 'winner': 'strategist', 'total_steps': 15, 'completion_reason': 'escape_successful'},
            {'seed': 150, 'winner': 'strategist', 'total_steps': 18, 'completion_reason': 'escape_successful'},
            {'seed': 800, 'winner': None, 'total_steps': 100, 'completion_reason': 'max_steps_reached'},  # Poor performance
            {'seed': 850, 'winner': None, 'total_steps': 95, 'completion_reason': 'max_steps_reached'}   # Poor performance
        ]
        
        optimal_ranges = main.find_optimal_seed_ranges(results, range_size=500)
        
        # Should identify range 0-499 as optimal (better success rate and fewer steps)
        self.assertEqual(len(optimal_ranges), 1)
        self.assertEqual(optimal_ranges[0]['range'], '0-499')
        self.assertEqual(optimal_ranges[0]['success_rate'], 1.0)  # 100% success
        self.assertLess(optimal_ranges[0]['avg_steps'], 50)  # Fewer average steps

    # FAILING TESTS: Advanced Seed Utilities
    
    def test_generate_seed_batch_for_testing(self):
        """Test generation of seed batches for reproducibility testing."""
        seed_batch = main.generate_seed_batch_for_testing(
            base_seed=1000,
            batch_size=5,
            distribution='uniform'
        )
        
        self.assertEqual(len(seed_batch), 5)
        self.assertTrue(all(isinstance(seed, int) for seed in seed_batch))
        self.assertTrue(all(seed >= 0 for seed in seed_batch))
        
        # Seeds should be distributed around base_seed for uniform distribution
        self.assertIn(1000, seed_batch)  # Should include base seed
    
    def test_validate_seed_reproducibility_across_runs(self):
        """Test validation that same seed produces same results across multiple runs."""
        seed = 12345
        
        # Mock two identical simulation results
        result1 = {'seed': seed, 'winner': 'strategist', 'total_steps': 30}
        result2 = {'seed': seed, 'winner': 'strategist', 'total_steps': 30}
        
        is_reproducible = main.validate_seed_reproducibility([result1, result2])
        self.assertTrue(is_reproducible)
        
        # Mock different results with same seed (should fail reproducibility)
        result3 = {'seed': seed, 'winner': 'mediator', 'total_steps': 45}
        
        is_not_reproducible = main.validate_seed_reproducibility([result1, result3])
        self.assertFalse(is_not_reproducible)
    
    def test_export_seed_performance_data(self):
        """Test export of seed performance data for external analysis."""
        results = [
            {'seed': 1, 'winner': 'strategist', 'total_steps': 20, 'simulation_duration': 100.0},
            {'seed': 2, 'winner': 'mediator', 'total_steps': 35, 'simulation_duration': 150.0}
        ]
        
        with patch('builtins.open', unittest.mock.mock_open()) as mock_file:
            main.export_seed_performance_data(results, self.temp_dir, format='csv')
            
            # Should create CSV file
            mock_file.assert_called_once()
            filename = mock_file.call_args[0][0]
            self.assertTrue(filename.endswith('.csv'))
            
            # Should write CSV headers and data
            written_calls = mock_file().write.call_args_list
            headers_written = any('seed,winner,total_steps' in str(call) for call in written_calls)
            self.assertTrue(headers_written)


if __name__ == '__main__':
    unittest.main()