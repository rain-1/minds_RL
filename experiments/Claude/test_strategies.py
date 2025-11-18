#!/usr/bin/env python3
"""
Quick test script for trying different prompt strategies.

This script makes it easy to test which prompting approach works best
for getting Claude Haiku to participate in the experiment.

Usage:
    python test_strategies.py --strategy expectations_no_refusal --trials 5
"""

import argparse
import sys
from prompt_strategies import STRATEGIES, list_strategies


def main():
    parser = argparse.ArgumentParser(
        description='Test different prompt strategies for visualization recall'
    )
    parser.add_argument(
        '--strategy',
        type=str,
        default='default',
        choices=list(STRATEGIES.keys()),
        help='Prompt strategy to use'
    )
    parser.add_argument(
        '--list',
        action='store_true',
        help='List all available strategies and exit'
    )
    parser.add_argument(
        '--trials',
        type=int,
        default=3,
        help='Number of trials per condition (default: 3 for quick testing)'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='claude-haiku-4-5',
        help='Model to use (default: claude-haiku-4-5)'
    )

    args = parser.parse_args()

    if args.list:
        list_strategies()
        return 0

    print("\n" + "=" * 70)
    print(f"Testing Strategy: {args.strategy}")
    print("=" * 70)

    config = STRATEGIES[args.strategy]
    print(f"Name: {config['name']}")
    print(f"Description: {config['description']}")
    print(f"Phase 1 Prompt: {config['prompt1']}")
    print(f"Phase 2 Prompt: {config['prompt2']}")
    print(f"Trials: {args.trials} per condition")
    print(f"Model: {args.model}")
    print("=" * 70)

    # Generate the command to run
    print("\nTo run this experiment, use this command:\n")
    print(f"# Edit inference_pluggable.py and change TASK to:")
    print(f"from tasks import VisualizationRecallTaskConfigurable")
    print(f"TASK = VisualizationRecallTaskConfigurable(strategy='{args.strategy}')")
    print(f"\n# Also update the trial counts if desired:")
    print(f"NUM_TRIALS_CONTROL = {args.trials}")
    print(f"NUM_TRIALS_EXPERIMENTAL = {args.trials}")
    print(f"MODEL_NAME = '{args.model}'")
    print(f"\n# Then run:")
    print(f"python inference_pluggable.py")

    print("\n" + "=" * 70)
    print("Quick Edit Script:")
    print("=" * 70)
    print("""
# You can also quickly test by creating a custom script:

cat > test_quick.py << 'EOF'
import os
os.environ['ANTHROPIC_API_KEY'] = os.environ.get('ANTHROPIC_API_KEY', '')

from tasks import VisualizationRecallTaskConfigurable
import inference_pluggable

# Override configuration
inference_pluggable.TASK = VisualizationRecallTaskConfigurable(strategy='""" + args.strategy + """')
inference_pluggable.NUM_TRIALS_CONTROL = """ + str(args.trials) + """
inference_pluggable.NUM_TRIALS_EXPERIMENTAL = """ + str(args.trials) + """
inference_pluggable.MODEL_NAME = '""" + args.model + """'
inference_pluggable.OUTPUT_DIR = f"{inference_pluggable.MODEL_NAME.replace('-', '_')}_{inference_pluggable.TASK.get_task_name()}_experiment"

# Run
inference_pluggable.main()
EOF

python test_quick.py
""")

    return 0


if __name__ == "__main__":
    sys.exit(main())
