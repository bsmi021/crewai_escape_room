"""
Main entry point for the CrewAI Competitive Survival Simulation.

This script initializes and runs the competitive escape room simulation
where three AI agents compete for survival - only one can escape alive.
Implements reproducible seed-based simulations with comprehensive result tracking.
"""

import os
import sys
import argparse
import random
import json
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from escape_room_sim.competitive.competitive_simulation import CompetitiveSimulation
from escape_room_sim.utils.llm_config import validate_gemini_configuration

console = Console()


@dataclass
class CompetitiveSimulationConfig:
    """Configuration for competitive simulation with seed support."""
    seed: Optional[int] = None
    max_iterations: int = 10
    enable_memory: bool = True
    verbose_output: bool = True
    save_results: bool = True
    time_limit: int = 300

    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.seed is not None:
            if not isinstance(self.seed, int) or self.seed < 0:
                raise ValueError("Seed must be a non-negative integer")
        
        if self.seed is None:
            self.seed = self._generate_seed()
    
    def _generate_seed(self) -> int:
        """Generate a random seed internally."""
        return random.randint(0, 999999)

def parse_command_line_args():
    """Parse command line arguments for simulation configuration."""
    parser = argparse.ArgumentParser(description='CrewAI Competitive Survival Simulation')
    parser.add_argument('--seed', type=int, help='Seed for reproducible simulation results')
    parser.add_argument('--competitive', action='store_true', default=True, 
                       help='Run competitive simulation (default)')
    parser.add_argument('--max-iterations', type=int, default=10,
                       help='Maximum simulation iterations')
    
    return parser.parse_args()

def generate_random_seed() -> int:
    """Generate a random seed for simulation reproducibility."""
    seed = random.randint(0, 999999)
    console.print(f"[yellow]No seed provided, generated random seed: {seed}[/yellow]")
    return seed

def print_competitive_welcome():
    """Print welcome message for competitive simulation."""
    welcome_text = Text()
    welcome_text.append("CrewAI Competitive Survival Simulation\n", style="bold red")
    welcome_text.append("Single-Survivor Competition System\n\n", style="red")
    welcome_text.append("Three AI agents compete for survival:\n", style="white")
    welcome_text.append("   • Strategist: Analytical problem solver\n", style="cyan")
    welcome_text.append("   • Mediator: Group facilitator and diplomat\n", style="green") 
    welcome_text.append("   • Survivor: Pragmatic decision maker\n", style="yellow")
    welcome_text.append("\nOnly one agent can survive", style="bold red")
    welcome_text.append(" - Trust, betrayal, and moral choices matter", style="magenta")
    
    console.print(Panel(welcome_text, title="Starting Competitive Simulation", border_style="bright_red"))

def print_welcome():
    """Print welcome message with simulation info."""
    welcome_text = Text()
    welcome_text.append("CrewAI Escape Room Simulation\n", style="bold blue")
    welcome_text.append("Iterative Multi-Agent Collaboration System\n\n", style="blue")
    welcome_text.append("Three AI agents work together to escape:\n", style="white")
    welcome_text.append("   • Strategist: Analytical problem solver\n", style="cyan")
    welcome_text.append("   • Mediator: Group facilitator and diplomat\n", style="green") 
    welcome_text.append("   • Survivor: Pragmatic decision maker\n", style="yellow")
    welcome_text.append("\nMemory-enabled agents learn from each iteration", style="magenta")
    
    console.print(Panel(welcome_text, title="Starting Simulation", border_style="bright_blue"))

def get_competitive_simulation_config(seed: Optional[int] = None) -> CompetitiveSimulationConfig:
    """Get competitive simulation configuration from user or use defaults."""
    console.print("\n[bold cyan]Competitive Simulation Configuration[/bold cyan]")
    
    try:
        max_iterations = console.input("Max iterations [bright_blue](default: 10)[/bright_blue]: ").strip()
        max_iterations = int(max_iterations) if max_iterations else 10
        
        enable_memory = console.input("Enable agent memory [bright_blue](Y/n)[/bright_blue]: ").strip().lower()
        enable_memory = enable_memory != 'n'
        
        verbose = console.input("Verbose output [bright_blue](Y/n)[/bright_blue]: ").strip().lower()
        verbose = verbose != 'n'
        
        config = CompetitiveSimulationConfig(
            seed=seed,
            max_iterations=max_iterations,
            enable_memory=enable_memory,
            verbose_output=verbose,
            save_results=True,
            time_limit=300
        )
        
        console.print(f"\n[green]Configuration set: {max_iterations} iterations, memory {'enabled' if enable_memory else 'disabled'}, seed: {config.seed}[/green]")
        return config
        
    except (KeyboardInterrupt, EOFError):
        console.print("\n[yellow]Using default configuration...[/yellow]")
        return CompetitiveSimulationConfig(seed=seed)
    except ValueError:
        console.print("\n[yellow]Invalid input, using defaults...[/yellow]") 
        return CompetitiveSimulationConfig(seed=seed)

def get_simulation_config() -> CompetitiveSimulationConfig:
    """Get simulation configuration from user or use defaults."""
    console.print("\n[bold cyan]Simulation Configuration[/bold cyan]")
    
    try:
        max_iterations = console.input("Max iterations [bright_blue](default: 10)[/bright_blue]: ").strip()
        max_iterations = int(max_iterations) if max_iterations else 10
        
        enable_memory = console.input("Enable agent memory [bright_blue](Y/n)[/bright_blue]: ").strip().lower()
        enable_memory = enable_memory != 'n'
        
        verbose = console.input("Verbose output [bright_blue](Y/n)[/bright_blue]: ").strip().lower()
        verbose = verbose != 'n'
        
        config = SimulationConfig(
            max_iterations=max_iterations,
            enable_memory=enable_memory,
            verbose_output=verbose,
            deadlock_detection_enabled=True,
            save_intermediate_results=True
        )
        
        console.print(f"\n[green]Configuration set: {max_iterations} iterations, memory {'enabled' if enable_memory else 'disabled'}[/green]")
        return config
        
    except (KeyboardInterrupt, EOFError):
        console.print("\n[yellow]Using default configuration...[/yellow]")
        return SimulationConfig()
    except ValueError:
        console.print("\n[yellow]Invalid input, using defaults...[/yellow]") 
        return CompetitiveSimulationConfig()

def run_simulation_with_seed(seed: int) -> Dict[str, Any]:
    """Run competitive simulation with specified seed."""
    console.print(f"[bold]Simulation Seed: {seed}[/bold]")
    
    # Create and run competitive simulation
    simulation = CompetitiveSimulation(seed=seed)
    results = simulation.run_enhanced_simulation(max_steps=100)
    
    return results

def run_competitive_simulation_with_seed(seed: int) -> Dict[str, Any]:
    """Run competitive simulation with seed and return results."""
    simulation = CompetitiveSimulation(seed=seed)
    return simulation.run_enhanced_simulation(max_steps=100)

def run_competitive_simulation_with_config(config: CompetitiveSimulationConfig) -> Dict[str, Any]:
    """Run competitive simulation with full configuration."""
    simulation = CompetitiveSimulation(seed=config.seed)
    return simulation.run_enhanced_simulation(max_steps=config.max_iterations)

def run_and_save_simulation_results(seed: int, data_dir: str) -> Dict[str, Any]:
    """Run simulation and save results with seed correlation."""
    simulation = CompetitiveSimulation(seed=seed)
    results = simulation.run_enhanced_simulation(max_steps=100)
    
    # Save results with seed in filename
    filename = os.path.join(data_dir, f'competitive_simulation_seed_{seed}_results.json')
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    return results

def save_simulation_results_with_metadata(results: Dict[str, Any], data_dir: str):
    """Save simulation results with comprehensive metadata."""
    seed = results['seed']
    filename = os.path.join(data_dir, f'competitive_simulation_seed_{seed}_results.json')
    
    # Add metadata section
    enhanced_results = results.copy()
    enhanced_results['metadata'] = {
        'timestamp': datetime.now().isoformat(),
        'reproducibility_info': {
            'seed': seed,
            'version': '1.0.0'
        }
    }
    
    with open(filename, 'w') as f:
        json.dump(enhanced_results, f, indent=2, default=str)

def create_seed_based_result_directory(base_dir: str, seed: int) -> str:
    """Create organized directory structure based on seed."""
    result_dir = os.path.join(base_dir, f'seed_{seed}')
    os.makedirs(result_dir, exist_ok=True)
    return result_dir

def load_simulation_results_by_seed(data_dir: str, seed: int) -> List[Dict[str, Any]]:
    """Load simulation results filtered by seed value."""
    results = []
    for filename in os.listdir(data_dir):
        if f'seed_{seed}' in filename and filename.endswith('.json'):
            filepath = os.path.join(data_dir, filename)
            with open(filepath, 'r') as f:
                result = json.load(f)
                if result.get('seed') == seed:
                    results.append(result)
    return results

# Result comparison and analysis utilities
def compare_simulation_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compare simulation results by seed for reproducibility analysis."""
    seed_groups = {}
    
    for result in results:
        seed = result['seed']
        if seed not in seed_groups:
            seed_groups[seed] = {'results': [], 'reproducible': True}
        
        seed_groups[seed]['results'].append(result)
    
    # Check reproducibility for each seed
    for seed, group in seed_groups.items():
        if len(group['results']) > 1:
            first_result = group['results'][0]
            for other_result in group['results'][1:]:
                if (first_result.get('winner') != other_result.get('winner') or
                    first_result.get('total_steps') != other_result.get('total_steps')):
                    group['reproducible'] = False
                    break
    
    return {'seed_groups': seed_groups}

def analyze_seed_reproducibility(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze seed reproducibility across multiple runs."""
    seed_counts = {}
    reproducible_seeds = []
    non_reproducible_seeds = []
    
    # Group results by seed
    for result in results:
        seed = result['seed']
        if seed not in seed_counts:
            seed_counts[seed] = []
        seed_counts[seed].append(result)
    
    # Check reproducibility for seeds with multiple runs
    for seed, seed_results in seed_counts.items():
        if len(seed_results) > 1:
            # Check if all results are identical
            first = seed_results[0]
            is_reproducible = all(
                r.get('winner') == first.get('winner') and
                r.get('total_steps') == first.get('total_steps')
                for r in seed_results[1:]
            )
            
            if is_reproducible:
                reproducible_seeds.append(seed)
            else:
                non_reproducible_seeds.append(seed)
    
    total_multi_run_seeds = len(reproducible_seeds) + len(non_reproducible_seeds)
    reproducibility_rate = len(reproducible_seeds) / max(1, total_multi_run_seeds)
    
    return {
        'reproducible_seeds': reproducible_seeds,
        'non_reproducible_seeds': non_reproducible_seeds,
        'reproducibility_rate': reproducibility_rate
    }

def find_similar_simulation_outcomes(results: List[Dict[str, Any]], similarity_threshold: float = 0.8) -> List[Dict[str, Any]]:
    """Find simulations with similar outcomes."""
    similar_groups = []
    
    # Simple similarity based on winner and step count
    for i, result1 in enumerate(results):
        similar_seeds = [result1['seed']]
        
        for j, result2 in enumerate(results[i+1:], i+1):
            if (result1.get('winner') == result2.get('winner') and
                abs(result1.get('total_steps', 0) - result2.get('total_steps', 0)) <= 5):
                similar_seeds.append(result2['seed'])
        
        if len(similar_seeds) > 1:
            similar_groups.append({'seeds': similar_seeds})
    
    return similar_groups

def calculate_reproducibility_metrics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calculate comprehensive reproducibility metrics."""
    seeds = [r['seed'] for r in results]
    unique_seeds = set(seeds)
    seed_counts = {seed: seeds.count(seed) for seed in unique_seeds}
    
    seeds_with_multiple_runs = [seed for seed, count in seed_counts.items() if count > 1]
    fully_reproducible_seeds = []
    
    # Check reproducibility for seeds with multiple runs
    for seed in seeds_with_multiple_runs:
        seed_results = [r for r in results if r['seed'] == seed]
        first = seed_results[0]
        is_reproducible = all(
            r.get('winner') == first.get('winner') and
            r.get('completion_reason') == first.get('completion_reason')
            for r in seed_results[1:]
        )
        if is_reproducible:
            fully_reproducible_seeds.append(seed)
    
    return {
        'total_unique_seeds': len(unique_seeds),
        'total_runs': len(results),
        'seeds_with_multiple_runs': len(seeds_with_multiple_runs),
        'fully_reproducible_seeds': len(fully_reproducible_seeds),
        'reproducibility_rate': len(fully_reproducible_seeds) / max(1, len(seeds_with_multiple_runs))
    }

def analyze_seed_outcome_variance(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze outcome variance for seeds with multiple runs."""
    seed_groups = {}
    
    for result in results:
        seed = result['seed']
        if seed not in seed_groups:
            seed_groups[seed] = []
        seed_groups[seed].append(result)
    
    variance_analysis = {}
    
    for seed, seed_results in seed_groups.items():
        if len(seed_results) > 1:
            winners = [r.get('winner') for r in seed_results]
            steps = [r.get('total_steps', 0) for r in seed_results]
            
            # Winner consistency (1.0 = all same, 0.0 = all different)
            unique_winners = len(set(winners))
            winner_consistency = 1.0 - (unique_winners - 1) / max(1, len(winners) - 1)
            
            # Step variance
            avg_steps = sum(steps) / len(steps)
            step_variance = sum((s - avg_steps) ** 2 for s in steps) / len(steps)
            
            variance_analysis[seed] = {
                'winner_consistency': winner_consistency,
                'step_variance': step_variance
            }
    
    return variance_analysis

def generate_reproducibility_report(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Generate comprehensive reproducibility report."""
    metrics = calculate_reproducibility_metrics(results)
    variance = analyze_seed_outcome_variance(results)
    
    report = {
        'summary': {
            'total_runs': metrics['total_runs'],
            'unique_seeds': metrics['total_unique_seeds'],
            'reproducibility_rate': metrics['reproducibility_rate']
        },
        'seed_analysis': variance,
        'recommendations': [
            "Use consistent seed values for reproducible testing",
            "Run multiple simulations with same seed to verify reproducibility",
            "Document seed values for important simulation results"
        ]
    }
    
    return report

def analyze_performance_by_seed_range(results: List[Dict[str, Any]], range_size: int = 1000) -> Dict[str, Any]:
    """Analyze performance grouped by seed ranges."""
    ranges = {}
    
    for result in results:
        seed = result['seed']
        range_start = (seed // range_size) * range_size
        range_end = range_start + range_size - 1
        range_key = f"{range_start}-{range_end}"
        
        if range_key not in ranges:
            ranges[range_key] = {
                'run_count': 0,
                'total_steps': 0,
                'total_duration': 0
            }
        
        ranges[range_key]['run_count'] += 1
        ranges[range_key]['total_steps'] += result.get('total_steps', 0)
        ranges[range_key]['total_duration'] += result.get('simulation_duration', 0)
    
    # Calculate averages
    for range_data in ranges.values():
        range_data['avg_steps'] = range_data['total_steps'] / range_data['run_count']
        range_data['avg_duration'] = range_data['total_duration'] / range_data['run_count']
    
    return ranges

def find_optimal_seed_ranges(results: List[Dict[str, Any]], range_size: int = 1000) -> List[Dict[str, Any]]:
    """Identify seed ranges with optimal performance."""
    ranges = analyze_performance_by_seed_range(results, range_size)
    optimal_ranges = []
    
    for range_key, range_data in ranges.items():
        # Calculate success rate
        range_results = [r for r in results 
                        if (r['seed'] // range_size) * range_size == int(range_key.split('-')[0])]
        successful_runs = sum(1 for r in range_results if r.get('winner') is not None)
        success_rate = successful_runs / len(range_results) if range_results else 0
        
        if success_rate > 0.5 and range_data['avg_steps'] < 75:  # Good performance criteria
            optimal_ranges.append({
                'range': range_key,
                'success_rate': success_rate,
                'avg_steps': range_data['avg_steps']
            })
    
    return sorted(optimal_ranges, key=lambda x: x['success_rate'], reverse=True)

def generate_seed_batch_for_testing(base_seed: int, batch_size: int = 10, distribution: str = 'uniform') -> List[int]:
    """Generate seed batch for reproducibility testing."""
    seeds = [base_seed]  # Include base seed
    
    if distribution == 'uniform':
        # Generate seeds around base_seed
        for i in range(1, batch_size):
            offset = i * 100
            seeds.append(base_seed + offset)
    
    return seeds[:batch_size]

def validate_seed_reproducibility(results: List[Dict[str, Any]]) -> bool:
    """Validate that same seed produces same results."""
    if len(results) < 2:
        return True
    
    first = results[0]
    for other in results[1:]:
        if (first.get('seed') == other.get('seed') and
            (first.get('winner') != other.get('winner') or
             first.get('total_steps') != other.get('total_steps'))):
            return False
    
    return True

def export_seed_performance_data(results: List[Dict[str, Any]], data_dir: str, format: str = 'csv'):
    """Export seed performance data for external analysis."""
    filename = os.path.join(data_dir, f'seed_performance_data.{format}')
    
    with open(filename, 'w') as f:
        if format == 'csv':
            # Write CSV headers
            f.write('seed,winner,total_steps,completion_reason,simulation_duration\n')
            
            # Write data rows
            for result in results:
                f.write(f"{result.get('seed', '')},{result.get('winner', '')},{result.get('total_steps', '')},"
                       f"{result.get('completion_reason', '')},{result.get('simulation_duration', '')}\n")

def run_batch_simulations(seeds: List[int], max_steps: int = 50) -> List[Dict[str, Any]]:
    """Run batch simulations with specified seeds."""
    results = []
    
    for seed in seeds:
        simulation = CompetitiveSimulation(seed=seed)
        result = simulation.run_enhanced_simulation(max_steps=max_steps)
        results.append(result)
    
    return results

def serialize_config(config: CompetitiveSimulationConfig) -> str:
    """Serialize configuration to JSON string."""
    return json.dumps(asdict(config))

def deserialize_config(config_str: str) -> CompetitiveSimulationConfig:
    """Deserialize configuration from JSON string."""
    data = json.loads(config_str)
    return CompetitiveSimulationConfig(**data)

def main():
    """Main entry point for the competitive simulation."""
    # Load environment variables
    load_dotenv()
    
    # Parse command line arguments
    args = parse_command_line_args()
    
    # Print welcome message for competitive simulation
    print_competitive_welcome()
    
    # Verify Google Gemini configuration (primary)
    is_valid, validation_message = validate_gemini_configuration()
    if is_valid:
        console.print(f"\n[green]{validation_message}[/green]")
        api_provider = "Google Gemini"
    else:
        console.print(f"\n[red]Gemini configuration error: {validation_message}[/red]")
        
        # Fallback to other providers
        if os.getenv("OPENAI_API_KEY"):
            console.print("[yellow]Falling back to OpenAI API[/yellow]")
            api_provider = "OpenAI"
        elif os.getenv("ANTHROPIC_API_KEY"):
            console.print("[yellow]Falling back to Anthropic API[/yellow]")
            api_provider = "Anthropic"
        else:
            console.print("\n[red]Error: No valid API configuration found![/red]")
            console.print("[yellow]Please configure Google Gemini (preferred) in your .env file:[/yellow]")
            console.print("[dim]1. Copy .env.example to .env[/dim]")
            console.print("[dim]2. Add GEMINI_API_KEY=your_key_here[/dim]")
            console.print("[dim]3. Add MODEL=gemini-2.5-flash-lite[/dim]")
            console.print("[dim]4. Run the simulation again[/dim]")
            return
    
    try:
        # Get competitive simulation configuration
        config = get_competitive_simulation_config(seed=args.seed)
        
        console.print(f"\n[bold]Initializing Competitive Simulation Engine...[/bold]")
        console.print(f"[bold]Simulation Seed: {config.seed}[/bold]")
        
        # Create data directory
        data_dir = "data"
        os.makedirs(data_dir, exist_ok=True)
        
        # Initialize and run competitive simulation
        simulation = CompetitiveSimulation(seed=config.seed)
        
        console.print(f"\n[bold green]COMPETITIVE SIMULATION COMPLETE![/bold green]")
        console.print(f"[dim]Press Ctrl+C at any time to stop the simulation...[/dim]\n")
        
        # Run the competitive simulation
        final_results = simulation.run_enhanced_simulation(max_steps=config.max_iterations)
        
        # Save results with seed correlation
        save_simulation_results_with_metadata(final_results, data_dir)
        
        # Display final results
        console.print(f"\n[bold]COMPETITIVE SIMULATION RESULTS[/bold]")
        console.print(Panel(
            f"""
[bold]Seed:[/bold] {final_results['seed']}
[bold]Winner:[/bold] {final_results['winner'] if final_results['winner'] else 'None'}
[bold]Completion Reason:[/bold] {final_results['completion_reason']}
[bold]Total Steps:[/bold] {final_results['total_steps']}
[bold]Duration:[/bold] {final_results['simulation_duration']:.1f} seconds

[bold]Competition Metrics:[/bold]
• Total Actions: {final_results['competition_metrics']['total_actions']}
• Resource Competition: {final_results['competition_metrics']['resource_competition']}
• Cooperation Attempts: {final_results['competition_metrics']['cooperation_attempts']}
• Betrayal Incidents: {final_results['competition_metrics']['betrayal_incidents']}
• Moral Choices: {final_results['competition_metrics']['moral_choices']}
            """,
            title="Competitive Simulation Complete",
            border_style="green" if final_results['winner'] else "yellow"
        ))
        
        # Show data files created
        console.print(f"\n[dim]Results saved in '{data_dir}' directory with seed correlation[/dim]")
        
        if final_results['winner']:
            console.print(f"\n[bold green]WINNER: {final_results['winner']} agent survived![/bold green]")
        else:
            console.print(f"\n[yellow]No winner - simulation ended without successful escape[/yellow]")
            
        return final_results
            
    except KeyboardInterrupt:
        console.print(f"\n[yellow]Simulation interrupted by user[/yellow]")
        console.print(f"[dim]Partial results may be saved in the data directory[/dim]")
        
    except Exception as e:
        console.print(f"\n[red]Simulation error: {str(e)}[/red]")
        console.print(f"[dim]Check your API key and configuration[/dim]")
        raise

if __name__ == "__main__":
    main()