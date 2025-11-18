#!/usr/bin/env python3
"""
Quick runner for testing different prompt strategies.

This creates a standalone experiment runner that you can easily modify
to test different strategies without editing inference_pluggable.py.

Usage:
    # Test the "no_refusal" strategy with 3 trials
    python run_with_strategy.py no_refusal 3

    # Test the "expectations_strong" strategy with 5 trials
    python run_with_strategy.py expectations_strong 5

    # Run with default strategy and default trials (10)
    python run_with_strategy.py
"""

import sys
import os

# Import the task system
from tasks import VisualizationRecallTaskConfigurable
from prompt_strategies import STRATEGIES
import inference_pluggable


def main():
    # Parse command line arguments
    strategy = sys.argv[1] if len(sys.argv) > 1 else 'default'
    trials = int(sys.argv[2]) if len(sys.argv) > 2 else 10

    # Validate strategy
    if strategy not in STRATEGIES:
        print(f"Error: Unknown strategy '{strategy}'")
        print(f"Available strategies: {', '.join(STRATEGIES.keys())}")
        sys.exit(1)

    # Display configuration
    config = STRATEGIES[strategy]
    print("\n" + "=" * 70)
    print("EXPERIMENT CONFIGURATION")
    print("=" * 70)
    print(f"Strategy: {strategy}")
    print(f"Name: {config['name']}")
    print(f"Description: {config['description']}")
    print(f"Trials per condition: {trials}")
    print(f"Model: {inference_pluggable.MODEL_NAME}")
    print(f"Phase 1 Prompt: {config['prompt1']}")
    print(f"Phase 2 Prompt: {config['prompt2']}")
    print("=" * 70 + "\n")

    # Configure the experiment
    inference_pluggable.TASK = VisualizationRecallTaskConfigurable(strategy=strategy)
    inference_pluggable.NUM_TRIALS_CONTROL = trials
    inference_pluggable.NUM_TRIALS_EXPERIMENTAL = trials

    # Update output directory based on task name
    inference_pluggable.OUTPUT_DIR = (
        f"{inference_pluggable.MODEL_NAME.replace('-', '_')}_"
        f"{inference_pluggable.TASK.get_task_name()}_experiment"
    )

    print(f"Output directory: {inference_pluggable.OUTPUT_DIR}\n")

    # Run the experiment
    try:
        inference_pluggable.main()
        print("\n" + "=" * 70)
        print("EXPERIMENT COMPLETED SUCCESSFULLY")
        print("=" * 70)
        print(f"Results saved to: {inference_pluggable.OUTPUT_DIR}/")
        print(f"  - summary.tsv (tabulated results)")
        print(f"  - trials.jsonl (full trial data)")
        print()
    except KeyboardInterrupt:
        print("\n\nExperiment interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nError running experiment: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    # Show usage if --help
    if len(sys.argv) > 1 and sys.argv[1] in ['--help', '-h']:
        print(__doc__)
        print("\nAvailable strategies:")
        for key, config in STRATEGIES.items():
            print(f"  {key:25s} - {config['name']}")
        sys.exit(0)

    main()
