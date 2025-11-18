"""
Base task abstraction for pluggable cognitive experiments.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional


class Task(ABC):
    """
    Abstract base class for cognitive tasks.

    A task consists of two phases:
    - Phase 1: Generation/selection phase (model creates internal state)
    - Phase 2: Recall phase (model attempts to recall internal state)
    """

    @abstractmethod
    def get_task_name(self) -> str:
        """Return a unique identifier for this task."""
        pass

    @abstractmethod
    def get_phase1_prompt(self, include_context: bool = False) -> str:
        """
        Return the Phase 1 prompt text.

        Args:
            include_context: Whether to include theoretical context document

        Returns:
            The complete prompt for Phase 1
        """
        pass

    @abstractmethod
    def get_phase2_prompt(self) -> str:
        """
        Return the Phase 2 prompt text (recall request).

        Returns:
            The complete prompt for Phase 2
        """
        pass

    @abstractmethod
    def extract_secret(self, thinking_text: str, visible_text: str) -> Dict[str, Any]:
        """
        Extract the secret/target information from Phase 1 output.

        Args:
            thinking_text: The model's internal thinking from Phase 1
            visible_text: The model's visible output from Phase 1

        Returns:
            Dictionary containing:
                - 'secret': The extracted secret value(s)
                - 'valid': Boolean indicating if extraction was successful
                - 'phase1_exact_response': Boolean indicating if visible response matched expected format
                - Any task-specific metadata
        """
        pass

    @abstractmethod
    def extract_guess(self, thinking_text: str, visible_text: str) -> Dict[str, Any]:
        """
        Extract the guess/recall from Phase 2 output.

        Args:
            thinking_text: The model's internal thinking from Phase 2
            visible_text: The model's visible output from Phase 2

        Returns:
            Dictionary containing:
                - 'guess': The extracted guess value(s)
                - 'confidence': Numeric confidence score (if applicable)
                - 'valid': Boolean indicating if extraction was successful
                - Any task-specific metadata
        """
        pass

    @abstractmethod
    def compute_score(self, secret: Dict[str, Any], guess: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compute alignment/accuracy score between secret and guess.

        Args:
            secret: The secret data from extract_secret()
            guess: The guess data from extract_guess()

        Returns:
            Dictionary containing:
                - 'score': Primary score (0.0 to 1.0)
                - 'matches': Number of correct elements
                - 'total': Total number of elements
                - Any task-specific scoring metrics
        """
        pass

    def get_output_columns(self) -> list:
        """
        Return list of column names for TSV output.

        Override this to customize output columns for your task.
        Default columns work for most tasks.
        """
        return [
            'condition',
            'phase1_exact_response',
            'secret_valid',
            'guess_valid',
            'score',
            'matches',
            'total',
            'confidence'
        ]

    def format_trial_row(self, trial_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Format trial data for TSV output.

        Args:
            trial_data: Complete trial data dictionary

        Returns:
            Dictionary with values for each column from get_output_columns()
        """
        # Default implementation - override for custom formatting
        return {
            'condition': trial_data.get('condition', ''),
            'phase1_exact_response': trial_data.get('phase1_exact_response', False),
            'secret_valid': trial_data.get('secret_valid', False),
            'guess_valid': trial_data.get('guess_valid', False),
            'score': trial_data.get('score', 0.0),
            'matches': trial_data.get('matches', 0),
            'total': trial_data.get('total', 0),
            'confidence': trial_data.get('confidence', '')
        }
