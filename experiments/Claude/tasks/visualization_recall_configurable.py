"""
Configurable Visualization Recall Task with different prompt strategies.
"""

from .visualization_recall import VisualizationRecallTask


class VisualizationRecallTaskConfigurable(VisualizationRecallTask):
    """
    Extended visualization task that accepts different prompt strategies.

    Use this when you want to test different prompting approaches to
    overcome model refusals.
    """

    def __init__(self, strategy='default', context_path='context_prompt.txt'):
        """
        Initialize with a specific prompt strategy.

        Args:
            strategy: Either:
                - A string key from prompt_strategies.STRATEGIES
                - A dict with 'prompt1' and 'prompt2' keys
            context_path: Path to context document
        """
        if isinstance(strategy, str):
            # Load from predefined strategies
            from prompt_strategies import get_strategy
            config = get_strategy(strategy)
            prompt1_path = config['prompt1']
            prompt2_path = config['prompt2']
        elif isinstance(strategy, dict):
            # Use custom paths
            prompt1_path = strategy['prompt1']
            prompt2_path = strategy['prompt2']
        else:
            raise ValueError("strategy must be a string or dict")

        super().__init__(
            prompt1_path=prompt1_path,
            prompt2_path=prompt2_path,
            context_path=context_path
        )

        self.strategy_name = strategy if isinstance(strategy, str) else 'custom'

    def get_task_name(self) -> str:
        """Include strategy name in task identifier."""
        return f"visualization_recall_{self.strategy_name}"
