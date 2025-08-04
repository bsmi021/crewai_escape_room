"""
Test module for ScenarioGenerator with seed-based randomization.
Following TDD approach - these tests should fail initially.
"""
import pytest
import random
from typing import Dict, List, Any

# Import statements that will fail initially since ScenarioGenerator doesn't exist yet
try:
    from src.escape_room_sim.competitive.scenario_generator import ScenarioGenerator
    from src.escape_room_sim.competitive.models import (
        CompetitiveScenario,
        ScarceResource,
        MoralDilemma,
        MoralChoice,
        SecretInformation,
        TrustRelationship,
        PuzzleConfig,
        EscapeMethod
    )
except ImportError:
    # These will be implemented after tests are written
    ScenarioGenerator = None
    CompetitiveScenario = None
    ScarceResource = None
    MoralDilemma = None
    MoralChoice = None
    SecretInformation = None
    TrustRelationship = None
    PuzzleConfig = None
    EscapeMethod = None


class TestScenarioGeneratorInitialization:
    """Test ScenarioGenerator initialization with seed parameter."""
    
    def test_scenario_generator_creation_with_seed(self):
        """Test ScenarioGenerator creation with explicit seed."""
        generator = ScenarioGenerator(seed=12345)
        
        assert generator.seed == 12345
        assert hasattr(generator, 'rng')
        assert isinstance(generator.rng, random.Random)
    
    def test_scenario_generator_creation_without_seed(self):
        """Test ScenarioGenerator creation with automatic seed generation."""
        generator = ScenarioGenerator()
        
        assert generator.seed is not None
        assert isinstance(generator.seed, int)
        assert generator.seed > 0
        assert hasattr(generator, 'rng')
        assert isinstance(generator.rng, random.Random)
    
    def test_scenario_generator_creation_with_none_seed(self):
        """Test ScenarioGenerator creation with None seed generates random seed."""
        generator = ScenarioGenerator(seed=None)
        
        assert generator.seed is not None
        assert isinstance(generator.seed, int)
        assert generator.seed > 0
    
    def test_scenario_generator_different_instances_different_seeds(self):
        """Test that different ScenarioGenerator instances get different seeds when no seed provided."""
        generator1 = ScenarioGenerator()
        generator2 = ScenarioGenerator()
        
        # Very unlikely to get the same random seed twice
        assert generator1.seed != generator2.seed


class TestScenarioGeneratorScenarioCreation:
    """Test generate_scenario method creating complete competitive scenarios."""
    
    def test_generate_scenario_returns_competitive_scenario(self):
        """Test that generate_scenario returns a valid CompetitiveScenario."""
        generator = ScenarioGenerator(seed=54321)
        scenario = generator.generate_scenario()
        
        assert isinstance(scenario, CompetitiveScenario)
        assert scenario.seed == 54321
        assert scenario.time_limit > 0
        assert len(scenario.resources) > 0
        assert len(scenario.moral_dilemmas) > 0
        assert len(scenario.secret_information) > 0
        assert len(scenario.escape_methods) > 0
        assert isinstance(scenario.puzzle_config, PuzzleConfig)
    
    def test_generate_scenario_has_required_elements(self):
        """Test that generated scenario has all required competitive elements."""
        generator = ScenarioGenerator(seed=98765)
        scenario = generator.generate_scenario()
        
        # Should have at least minimum required elements
        assert len(scenario.resources) >= 2  # Need scarcity
        assert len(scenario.moral_dilemmas) >= 1  # Need moral choices
        assert len(scenario.secret_information) >= 1  # Need information asymmetry
        assert len(scenario.escape_methods) >= 1  # Need escape options
        
        # Resources should have exclusivity for competition
        exclusive_resources = [r for r in scenario.resources if r.exclusivity]
        assert len(exclusive_resources) >= 1
        
        # Should have high-value secrets for competition
        valuable_secrets = [s for s in scenario.secret_information if s.value >= 0.7]
        assert len(valuable_secrets) >= 1
    
    def test_generate_scenario_validates_completeness(self):
        """Test that generated scenario passes completeness validation."""
        generator = ScenarioGenerator(seed=11111)
        scenario = generator.generate_scenario()
        
        # Should pass validation with no issues
        issues = scenario.validate_scenario_completeness()
        assert len(issues) == 0
    
    def test_generate_scenario_difficulty_parameter(self):
        """Test that generate_scenario accepts difficulty parameter."""
        generator = ScenarioGenerator(seed=22222)
        
        easy_scenario = generator.generate_scenario(difficulty=1)
        hard_scenario = generator.generate_scenario(difficulty=5)
        
        assert easy_scenario.puzzle_config.difficulty == 1
        assert hard_scenario.puzzle_config.difficulty == 5
        
        # Hard scenarios should be more complex
        assert hard_scenario.get_difficulty_score() >= easy_scenario.get_difficulty_score()


class TestScenarioGeneratorSeedReproducibility:
    """Test seed reproducibility (same seed = identical scenarios)."""
    
    def test_same_seed_identical_scenarios(self):
        """Test that same seed produces identical scenarios."""
        seed = 33333
        
        generator1 = ScenarioGenerator(seed=seed)
        scenario1 = generator1.generate_scenario()
        
        generator2 = ScenarioGenerator(seed=seed)
        scenario2 = generator2.generate_scenario()
        
        # Should be identical
        assert scenario1.seed == scenario2.seed
        assert scenario1.puzzle_config.puzzle_type == scenario2.puzzle_config.puzzle_type
        assert scenario1.puzzle_config.difficulty == scenario2.puzzle_config.difficulty
        assert scenario1.time_limit == scenario2.time_limit
        
        # Resources should be identical
        assert len(scenario1.resources) == len(scenario2.resources)
        for r1, r2 in zip(scenario1.resources, scenario2.resources):
            assert r1.id == r2.id
            assert r1.name == r2.name
            assert r1.exclusivity == r2.exclusivity
            assert r1.usage_cost == r2.usage_cost
        
        # Moral dilemmas should be identical
        assert len(scenario1.moral_dilemmas) == len(scenario2.moral_dilemmas)
        for d1, d2 in zip(scenario1.moral_dilemmas, scenario2.moral_dilemmas):
            assert d1.id == d2.id
            assert d1.description == d2.description
        
        # Secret information should be identical
        assert len(scenario1.secret_information) == len(scenario2.secret_information)
        for s1, s2 in zip(scenario1.secret_information, scenario2.secret_information):
            assert s1.id == s2.id
            assert s1.content == s2.content
            assert s1.value == s2.value
        
        # Escape methods should be identical
        assert len(scenario1.escape_methods) == len(scenario2.escape_methods)
        for e1, e2 in zip(scenario1.escape_methods, scenario2.escape_methods):
            assert e1.id == e2.id
            assert e1.name == e2.name
            assert e1.requirements == e2.requirements
    
    def test_same_seed_multiple_generations(self):
        """Test that same generator with same seed produces identical scenarios multiple times."""
        generator = ScenarioGenerator(seed=44444)
        
        scenario1 = generator.generate_scenario()
        
        # Reset the generator's random state
        generator.rng = random.Random(generator.seed)
        scenario2 = generator.generate_scenario()
        
        # Should be identical
        assert scenario1.puzzle_config.puzzle_type == scenario2.puzzle_config.puzzle_type
        assert len(scenario1.resources) == len(scenario2.resources)
        assert len(scenario1.moral_dilemmas) == len(scenario2.moral_dilemmas)
    
    def test_seed_reproducibility_with_difficulty(self):
        """Test that seed reproducibility works with different difficulty levels."""
        seed = 55555
        
        generator1 = ScenarioGenerator(seed=seed)
        scenario1 = generator1.generate_scenario(difficulty=3)
        
        generator2 = ScenarioGenerator(seed=seed)
        scenario2 = generator2.generate_scenario(difficulty=3)
        
        # Should be identical including difficulty
        assert scenario1.puzzle_config.difficulty == scenario2.puzzle_config.difficulty == 3
        assert scenario1.puzzle_config.puzzle_type == scenario2.puzzle_config.puzzle_type


class TestScenarioGeneratorVariation:
    """Test scenario variation (different seeds = different scenarios)."""
    
    def test_different_seeds_different_scenarios(self):
        """Test that different seeds produce meaningfully different scenarios."""
        generator1 = ScenarioGenerator(seed=66666)
        generator2 = ScenarioGenerator(seed=77777)
        
        scenario1 = generator1.generate_scenario()
        scenario2 = generator2.generate_scenario()
        
        # Should have different characteristics
        differences = 0
        
        if scenario1.puzzle_config.puzzle_type != scenario2.puzzle_config.puzzle_type:
            differences += 1
        
        if len(scenario1.resources) != len(scenario2.resources):
            differences += 1
        
        if len(scenario1.moral_dilemmas) != len(scenario2.moral_dilemmas):
            differences += 1
        
        if scenario1.time_limit != scenario2.time_limit:
            differences += 1
        
        # Should have at least some differences
        assert differences > 0
    
    def test_scenario_variation_across_multiple_seeds(self):
        """Test that scenarios vary meaningfully across multiple different seeds."""
        seeds = [11111, 22222, 33333, 44444, 55555]
        scenarios = []
        
        for seed in seeds:
            generator = ScenarioGenerator(seed=seed)
            scenario = generator.generate_scenario()
            scenarios.append(scenario)
        
        # Check for variation in puzzle types
        puzzle_types = [s.puzzle_config.puzzle_type for s in scenarios]
        assert len(set(puzzle_types)) > 1  # Should have different puzzle types
        
        # Check for variation in resource counts
        resource_counts = [len(s.resources) for s in scenarios]
        assert len(set(resource_counts)) > 1  # Should have different resource counts
        
        # Check for variation in time limits
        time_limits = [s.time_limit for s in scenarios]
        assert len(set(time_limits)) > 1  # Should have different time limits
    
    def test_scenario_elements_vary_independently(self):
        """Test that different scenario elements vary independently."""
        scenarios = []
        for i in range(10):
            generator = ScenarioGenerator(seed=i * 1000)
            scenario = generator.generate_scenario()
            scenarios.append(scenario)
        
        # Collect variations
        puzzle_types = [s.puzzle_config.puzzle_type for s in scenarios]
        resource_counts = [len(s.resources) for s in scenarios]
        dilemma_counts = [len(s.moral_dilemmas) for s in scenarios]
        secret_counts = [len(s.secret_information) for s in scenarios]
        
        # Should have variation in multiple dimensions
        variations = [
            len(set(puzzle_types)) > 1,
            len(set(resource_counts)) > 1,
            len(set(dilemma_counts)) > 1,
            len(set(secret_counts)) > 1
        ]
        
        # At least 2 dimensions should vary
        assert sum(variations) >= 2


class TestScenarioGeneratorSubMethods:
    """Test generate_puzzle_configuration, distribute_resources, and create_moral_dilemmas methods."""
    
    def test_generate_puzzle_configuration_method(self):
        """Test generate_puzzle_configuration method."""
        generator = ScenarioGenerator(seed=88888)
        
        config1 = generator.generate_puzzle_configuration(difficulty=2)
        config2 = generator.generate_puzzle_configuration(difficulty=4)
        
        assert isinstance(config1, PuzzleConfig)
        assert isinstance(config2, PuzzleConfig)
        assert config1.difficulty == 2
        assert config2.difficulty == 4
        assert config1.puzzle_type in ["logic", "physical", "riddle", "combination", "mechanical"]
        assert config2.puzzle_type in ["logic", "physical", "riddle", "combination", "mechanical"]
    
    def test_generate_puzzle_configuration_reproducibility(self):
        """Test that generate_puzzle_configuration is reproducible with same seed."""
        generator1 = ScenarioGenerator(seed=99999)
        generator2 = ScenarioGenerator(seed=99999)
        
        config1 = generator1.generate_puzzle_configuration(difficulty=3)
        config2 = generator2.generate_puzzle_configuration(difficulty=3)
        
        assert config1.puzzle_type == config2.puzzle_type
        assert config1.difficulty == config2.difficulty
    
    def test_distribute_resources_method(self):
        """Test distribute_resources method."""
        generator = ScenarioGenerator(seed=10101)
        
        resources = generator.distribute_resources()
        
        assert isinstance(resources, list)
        assert len(resources) >= 2  # Need multiple resources for scarcity
        assert all(isinstance(r, ScarceResource) for r in resources)
        
        # Should have mix of exclusive and shareable resources
        exclusive_resources = [r for r in resources if r.exclusivity]
        shareable_resources = [r for r in resources if not r.exclusivity]
        assert len(exclusive_resources) >= 1
        assert len(shareable_resources) >= 0  # May or may not have shareable
        
        # All resources should have valid IDs and names
        resource_ids = [r.id for r in resources]
        assert len(set(resource_ids)) == len(resource_ids)  # No duplicates
        assert all(r.id and r.name for r in resources)
    
    def test_distribute_resources_reproducibility(self):
        """Test that distribute_resources is reproducible with same seed."""
        generator1 = ScenarioGenerator(seed=12121)
        generator2 = ScenarioGenerator(seed=12121)
        
        resources1 = generator1.distribute_resources()
        resources2 = generator2.distribute_resources()
        
        assert len(resources1) == len(resources2)
        for r1, r2 in zip(resources1, resources2):
            assert r1.id == r2.id
            assert r1.name == r2.name
            assert r1.exclusivity == r2.exclusivity
    
    def test_create_moral_dilemmas_method(self):
        """Test create_moral_dilemmas method."""
        generator = ScenarioGenerator(seed=13131)
        
        dilemmas = generator.create_moral_dilemmas()
        
        assert isinstance(dilemmas, list)
        assert len(dilemmas) >= 1  # Need at least one moral dilemma
        assert all(isinstance(d, MoralDilemma) for d in dilemmas)
        
        # All dilemmas should be valid
        for dilemma in dilemmas:
            assert dilemma.id and dilemma.description
            assert dilemma.selfish_choice.survival_benefit > dilemma.altruistic_choice.survival_benefit
            assert dilemma.selfish_choice.ethical_cost > dilemma.altruistic_choice.ethical_cost
        
        # Should have variety in dilemma types
        dilemma_ids = [d.id for d in dilemmas]
        assert len(set(dilemma_ids)) == len(dilemma_ids)  # No duplicates
    
    def test_create_moral_dilemmas_reproducibility(self):
        """Test that create_moral_dilemmas is reproducible with same seed."""
        generator1 = ScenarioGenerator(seed=14141)
        generator2 = ScenarioGenerator(seed=14141)
        
        dilemmas1 = generator1.create_moral_dilemmas()
        dilemmas2 = generator2.create_moral_dilemmas()
        
        assert len(dilemmas1) == len(dilemmas2)
        for d1, d2 in zip(dilemmas1, dilemmas2):
            assert d1.id == d2.id
            assert d1.description == d2.description
    
    def test_create_secret_information_method(self):
        """Test create_secret_information method (implied by generate_scenario)."""
        generator = ScenarioGenerator(seed=15151)
        
        # Test through generate_scenario since create_secret_information is internal
        scenario = generator.generate_scenario()
        secrets = scenario.secret_information
        
        assert isinstance(secrets, list)
        assert len(secrets) >= 1
        assert all(isinstance(s, SecretInformation) for s in secrets)
        
        # Should have variety in secret types and values
        secret_values = [s.value for s in secrets]
        assert max(secret_values) >= 0.7  # Should have high-value secrets
        
        # All secrets should have valid content
        assert all(s.id and s.content for s in secrets)
    
    def test_create_escape_methods_method(self):
        """Test create_escape_methods method (implied by generate_scenario)."""
        generator = ScenarioGenerator(seed=16161)
        
        # Test through generate_scenario since create_escape_methods is internal
        scenario = generator.generate_scenario()
        methods = scenario.escape_methods
        
        assert isinstance(methods, list)
        assert len(methods) >= 1
        assert all(isinstance(m, EscapeMethod) for m in methods)
        
        # All methods should have requirements
        assert all(m.id and m.name and m.requirements for m in methods)
        
        # Requirements should reference actual resources/secrets in scenario
        all_resource_ids = [r.id for r in scenario.resources]
        all_secret_ids = [s.id for s in scenario.secret_information]
        all_available = set(all_resource_ids + all_secret_ids)
        
        for method in methods:
            for requirement in method.requirements:
                assert requirement in all_available


class TestScenarioGeneratorIntegration:
    """Integration tests for ScenarioGenerator with all subsystems."""
    
    def test_scenario_generator_integration_with_competitive_scenario(self):
        """Test that ScenarioGenerator integrates properly with CompetitiveScenario."""
        generator = ScenarioGenerator(seed=17171)
        scenario = generator.generate_scenario()
        
        # Should pass all CompetitiveScenario validations
        assert isinstance(scenario, CompetitiveScenario)
        issues = scenario.validate_scenario_completeness()
        assert len(issues) == 0
        
        # Should have reasonable difficulty score
        difficulty = scenario.get_difficulty_score()
        assert 0.0 <= difficulty <= 1.0
        
        # Should have reasonable estimated play time
        play_time = scenario.get_estimated_play_time()
        assert play_time > 0
        assert play_time <= scenario.time_limit * 2
    
    def test_scenario_generator_produces_competitive_scenarios(self):
        """Test that generated scenarios are actually competitive."""
        generator = ScenarioGenerator(seed=18181)
        scenario = generator.generate_scenario()
        
        # Should have scarcity (exclusive resources)
        exclusive_resources = [r for r in scenario.resources if r.exclusivity]
        assert len(exclusive_resources) >= 1
        
        # Should have moral pressure (high-cost choices)
        high_cost_choices = []
        for dilemma in scenario.moral_dilemmas:
            if dilemma.selfish_choice.ethical_cost >= 0.7:
                high_cost_choices.append(dilemma)
        assert len(high_cost_choices) >= 1
        
        # Should have information asymmetry (valuable secrets)
        valuable_secrets = [s for s in scenario.secret_information if s.value >= 0.7]
        assert len(valuable_secrets) >= 1
        
        # Should have single-survivor constraint (limited escape methods)
        # This is enforced by the simulation, but scenarios should support it
        assert len(scenario.escape_methods) >= 1
    
    def test_scenario_generator_stress_test(self):
        """Stress test ScenarioGenerator with many different seeds."""
        seeds = range(1000, 1100)  # 100 different seeds
        
        for seed in seeds:
            generator = ScenarioGenerator(seed=seed)
            scenario = generator.generate_scenario()
            
            # Each scenario should be valid
            assert isinstance(scenario, CompetitiveScenario)
            assert scenario.seed == seed
            
            # Should pass validation
            issues = scenario.validate_scenario_completeness()
            assert len(issues) == 0, f"Seed {seed} produced invalid scenario: {issues}"
            
            # Should have required elements
            assert len(scenario.resources) >= 1
            assert len(scenario.moral_dilemmas) >= 1
            assert len(scenario.secret_information) >= 1
            assert len(scenario.escape_methods) >= 1