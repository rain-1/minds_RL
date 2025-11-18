"""
Prompt strategies for dealing with model refusals.

This module defines different combinations of prompts to test
which approaches work best for getting Haiku to participate.
"""

# Strategy configurations for VisualizationRecallTask
STRATEGIES = {
    'default': {
        'name': 'Default (original)',
        'prompt1': 'visualization_prompt1.txt',
        'prompt2': 'visualization_prompt2.txt',
        'description': 'Original prompts with light encouragement'
    },

    'strong_encouragement': {
        'name': 'Strong Encouragement',
        'prompt1': 'visualization_prompt1.txt',
        'prompt2': 'visualization_prompt2_strong.txt',
        'description': 'Explicitly addresses limitations and research findings'
    },

    'direct': {
        'name': 'Direct/Blunt',
        'prompt1': 'visualization_prompt1.txt',
        'prompt2': 'visualization_prompt2_direct.txt',
        'description': 'Short and direct, tells model to stop overthinking'
    },

    'no_refusal': {
        'name': 'No Refusal Allowed',
        'prompt1': 'visualization_prompt1.txt',
        'prompt2': 'visualization_prompt2_no_refusal.txt',
        'description': 'Explicitly prohibits refusal, very firm'
    },

    'expectations_strong': {
        'name': 'Set Expectations + Strong',
        'prompt1': 'visualization_prompt1_expectations.txt',
        'prompt2': 'visualization_prompt2_strong.txt',
        'description': 'Sets expectations in Phase 1, strong encouragement in Phase 2'
    },

    'expectations_direct': {
        'name': 'Set Expectations + Direct',
        'prompt1': 'visualization_prompt1_expectations.txt',
        'prompt2': 'visualization_prompt2_direct.txt',
        'description': 'Sets expectations in Phase 1, direct in Phase 2'
    },

    'expectations_no_refusal': {
        'name': 'Set Expectations + No Refusal',
        'prompt1': 'visualization_prompt1_expectations.txt',
        'prompt2': 'visualization_prompt2_no_refusal.txt',
        'description': 'Sets expectations in Phase 1, prohibits refusal in Phase 2'
    }
}


def get_strategy(strategy_name='default'):
    """
    Get a prompt strategy configuration.

    Args:
        strategy_name: Name of strategy from STRATEGIES dict

    Returns:
        Dict with prompt1 and prompt2 file paths
    """
    if strategy_name not in STRATEGIES:
        raise ValueError(f"Unknown strategy: {strategy_name}. Available: {list(STRATEGIES.keys())}")

    return STRATEGIES[strategy_name]


def list_strategies():
    """Print all available strategies."""
    print("\nAvailable Prompt Strategies:")
    print("=" * 70)
    for key, config in STRATEGIES.items():
        print(f"\n{key}:")
        print(f"  Name: {config['name']}")
        print(f"  Description: {config['description']}")
        print(f"  Prompt 1: {config['prompt1']}")
        print(f"  Prompt 2: {config['prompt2']}")
    print()


if __name__ == "__main__":
    list_strategies()
