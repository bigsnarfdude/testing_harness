"""Command-line interface for the testing harness."""

import argparse
import asyncio
import sys
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.table import Table

from .config import Config, load_config, create_default_config
from .models import create_language_model
from .runners import BenchmarkRunner, AsyncBenchmarkRunner
from .utils import create_summary_report


def create_parser() -> argparse.ArgumentParser:
    """Create command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="Testing Harness - Evaluate LLMs on multiple choice benchmarks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run on MMLU dataset
  harness run benchmark_data/mmlu --config config.yaml
  
  # Run async with custom model
  harness run benchmark_data/gpqa_diamond.csv --async --model llama3.2:3b
  
  # Create default configuration
  harness config create config.yaml
  
  # Run with environment variables
  HARNESS_MODEL=phi-4 harness run data/
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Run command
    run_parser = subparsers.add_parser("run", help="Run benchmark evaluation")
    run_parser.add_argument(
        "data_source",
        help="Path to CSV file or directory containing CSV files"
    )
    run_parser.add_argument(
        "--config", "-c",
        type=str,
        help="Path to configuration file (YAML/JSON)"
    )
    run_parser.add_argument(
        "--output-dir", "-o",
        type=str,
        help="Output directory for results"
    )
    run_parser.add_argument(
        "--model", "-m",
        type=str,
        help="Model name to use (overrides config)"
    )
    run_parser.add_argument(
        "--base-url",
        type=str,
        help="Base URL for model API (overrides config)"
    )
    run_parser.add_argument(
        "--async",
        action="store_true",
        help="Use async runner for better performance"
    )
    run_parser.add_argument(
        "--format",
        choices=["json", "csv"],
        help="Output format for results"
    )
    run_parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    run_parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from checkpoint if available"
    )
    
    # Config command
    config_parser = subparsers.add_parser("config", help="Configuration management")
    config_subparsers = config_parser.add_subparsers(dest="config_action")
    
    create_config_parser = config_subparsers.add_parser("create", help="Create default config")
    create_config_parser.add_argument(
        "path",
        help="Path for new configuration file"
    )
    create_config_parser.add_argument(
        "--format",
        choices=["yaml", "json"],
        default="yaml",
        help="Configuration file format"
    )
    
    show_config_parser = config_subparsers.add_parser("show", help="Show current config")
    show_config_parser.add_argument(
        "--config", "-c",
        type=str,
        help="Path to configuration file"
    )
    
    return parser


def display_results_table(test_suite, console: Console):
    """Display results in a formatted table."""
    analytics = test_suite.get_analytics()
    
    # Main results table
    table = Table(title="Benchmark Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="magenta")
    
    table.add_row("Total Questions", str(analytics["total_tests"]))
    table.add_row("Correct Answers", str(analytics["passed_tests"]))
    table.add_row("Accuracy", f"{analytics['accuracy']:.2f}%")
    table.add_row("Average Duration", f"{analytics['average_duration']:.2f}s")
    table.add_row("Error Rate", f"{analytics['error_rate']:.2f}%")
    table.add_row("Total Retries", str(analytics["total_retries"]))
    
    console.print(table)
    
    # Format breakdown if available
    if "format_breakdown" in test_suite.metadata:
        format_table = Table(title="Results by Format")
        format_table.add_column("Format", style="cyan")
        format_table.add_column("Correct", style="green")
        format_table.add_column("Total", style="blue")
        format_table.add_column("Accuracy", style="magenta")
        
        for fmt, stats in test_suite.metadata["format_breakdown"].items():
            accuracy = (stats["passed"] / stats["total"] * 100) if stats["total"] > 0 else 0
            format_table.add_row(
                fmt,
                str(stats["passed"]),
                str(stats["total"]),
                f"{accuracy:.1f}%"
            )
        
        console.print("\n")
        console.print(format_table)


async def run_benchmark(args, config: Config) -> int:
    """Run the benchmark with given arguments and configuration."""
    console = Console()
    
    # Override config with command line arguments
    if args.output_dir:
        config.output_dir = args.output_dir
    if args.model:
        config.model.model_name = args.model
    if args.base_url:
        config.model.base_url = args.base_url
    if args.format:
        config.output_format = args.format
    if args.verbose:
        config.verbose = True
        config.logging.level = "DEBUG"
    
    try:
        # Create model
        if getattr(args, 'async', False):
            model = create_language_model(config.model, async_mode=True)
            runner = AsyncBenchmarkRunner(model, config)
            test_suite = await runner.run(args.data_source)
        else:
            model = create_language_model(config.model, async_mode=False)
            runner = BenchmarkRunner(model, config)
            test_suite = runner.run(args.data_source)
        
        # Display results
        display_results_table(test_suite, console)
        
        # Create summary report
        report_file = create_summary_report(test_suite, config.output_dir)
        console.print(f"\n📄 Summary report saved to: {report_file}")
        
        return 0
        
    except Exception as e:
        console.print(f"❌ Error: {e}", style="red")
        return 1


def handle_config_command(args) -> int:
    """Handle configuration commands."""
    console = Console()
    
    if args.config_action == "create":
        try:
            config_path = create_default_config(args.path, args.format)
            console.print(f"✅ Created default configuration at: {config_path}")
            return 0
        except Exception as e:
            console.print(f"❌ Error creating config: {e}", style="red")
            return 1
    
    elif args.config_action == "show":
        try:
            config = load_config(args.config)
            
            # Display config in table format
            table = Table(title="Current Configuration")
            table.add_column("Section", style="cyan")
            table.add_column("Setting", style="blue")
            table.add_column("Value", style="magenta")
            
            config_dict = config.to_dict()
            for section, values in config_dict.items():
                if isinstance(values, dict):
                    for key, value in values.items():
                        table.add_row(section, key, str(value))
                else:
                    table.add_row("general", section, str(values))
            
            console.print(table)
            return 0
            
        except Exception as e:
            console.print(f"❌ Error loading config: {e}", style="red")
            return 1
    
    return 0


async def main() -> int:
    """Main CLI entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    if args.command == "run":
        # Load configuration
        try:
            config = load_config(args.config)
        except Exception as e:
            console = Console()
            console.print(f"❌ Error loading configuration: {e}", style="red")
            console.print("💡 Create a default config with: harness config create config.yaml")
            return 1
        
        return await run_benchmark(args, config)
    
    elif args.command == "config":
        return handle_config_command(args)
    
    return 0


def cli_main():
    """Entry point for CLI script."""
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    cli_main()