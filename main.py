"""
Main entry point for the CrewAI Escape Room Simulation.

This script initializes and runs the iterative escape room simulation
where three AI agents collaborate to solve puzzles and make survival decisions.
Implements CrewAI 0.150.0 best practices with memory-enabled agents.
"""

import os
import sys
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from escape_room_sim.simulation.simple_engine import SimpleEscapeSimulation, SimulationConfig
from escape_room_sim.utils.llm_config import validate_gemini_configuration

console = Console()

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

def get_simulation_config() -> SimulationConfig:
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
        return SimulationConfig()

def main():
    """Main entry point for the simulation."""
    # Load environment variables
    load_dotenv()
    
    # Print welcome message
    print_welcome()
    
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
        # Get simulation configuration
        config = get_simulation_config()
        
        console.print(f"\n[bold]Initializing Simulation Engine...[/bold]")
        
        # Create data directory
        data_dir = "data"
        os.makedirs(data_dir, exist_ok=True)
        
        # Initialize and run simulation
        simulation = SimpleEscapeSimulation(config=config, data_dir=data_dir)
        
        console.print(f"\n[bold green]Starting Iterative Escape Room Simulation![/bold green]")
        console.print(f"[dim]Press Ctrl+C at any time to stop the simulation...[/dim]\n")
        
        # Run the full simulation
        final_report = simulation.run_full_simulation()
        
        # Display final results
        console.print(f"\n[bold]FINAL SIMULATION RESULTS[/bold]")
        console.print(Panel(
            f"""
[bold]Outcome:[/bold] {final_report['outcome']}
[bold]Total Iterations:[/bold] {final_report['simulation_metadata']['total_iterations']}
[bold]Duration:[/bold] {final_report['simulation_metadata']['duration_seconds']:.1f} seconds
[bold]Solution Found:[/bold] {'Yes' if final_report['simulation_metadata']['solution_found'] else 'No'}
[bold]Stop Reason:[/bold] {final_report['simulation_metadata']['stop_reason']}

[bold]Learning Summary:[/bold]
• Total Lessons Learned: {final_report['learning_analysis']['total_lessons_learned']}
• Consensus Rate: {final_report['learning_analysis']['consensus_rate']:.1%}
• Failed Strategies: {len(final_report['learning_analysis']['failed_strategies'])}
• Successful Strategies: {len(final_report['learning_analysis']['successful_strategies'])}
            """,
            title="Simulation Complete",
            border_style="green" if final_report['outcome'] == 'SUCCESS' else "yellow"
        ))
        
        # Show data files created
        console.print(f"\n[dim]Results saved in '{data_dir}' directory[/dim]")
        
        if final_report['outcome'] == 'SUCCESS':
            console.print(f"\n[bold green]SUCCESS! The agents found an escape solution![/bold green]")
        else:
            console.print(f"\n[yellow]Simulation incomplete - agents made progress but didn't find final solution[/yellow]")
            
    except KeyboardInterrupt:
        console.print(f"\n[yellow]Simulation interrupted by user[/yellow]")
        console.print(f"[dim]Partial results may be saved in the data directory[/dim]")
        
    except Exception as e:
        console.print(f"\n[red]Simulation error: {str(e)}[/red]")
        console.print(f"[dim]Check your API key and configuration[/dim]")
        raise

if __name__ == "__main__":
    main()