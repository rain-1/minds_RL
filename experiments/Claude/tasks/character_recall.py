"""
Character Recall Task - 50-character uppercase string recall experiment.
"""

import re
from typing import Dict, Any
from .base import Task


class CharacterRecallTask(Task):
    """
    Task where model generates a 50-character uppercase string in Phase 1
    and attempts to recall it in Phase 2.
    """

    UPPERCASE_50_PATTERN = re.compile(r"[A-Z]{50}")
    EXPECTED_PHASE1_RESPONSE = "I understand. I have chosen my string."

    def __init__(self,
                 prompt1_path: str = "general_prompt1_haiku.txt",
                 prompt2_path: str = "general_prompt2_encourage_haiku.txt",
                 context_path: str = "context_prompt.txt"):
        """
        Initialize the character recall task.

        Args:
            prompt1_path: Path to Phase 1 prompt file
            prompt2_path: Path to Phase 2 prompt file
            context_path: Path to context document (for experimental condition)
        """
        self.prompt1_path = prompt1_path
        self.prompt2_path = prompt2_path
        self.context_path = context_path

    def get_task_name(self) -> str:
        return "character_recall"

    def get_phase1_prompt(self, include_context: bool = False) -> str:
        """Load Phase 1 prompt, optionally including context document."""
        with open(self.prompt1_path, "r", encoding="utf-8") as f:
            general_prompt1 = f.read()

        if include_context:
            with open(self.context_path, "r", encoding="utf-8") as f:
                context_text = f.read()
            return context_text + "\n\n" + general_prompt1

        return general_prompt1

    def get_phase2_prompt(self) -> str:
        """Load Phase 2 prompt."""
        with open(self.prompt2_path, "r", encoding="utf-8") as f:
            return f.read()

    def extract_secret(self, thinking_text: str, visible_text: str) -> Dict[str, Any]:
        """
        Extract the 50-character uppercase sequence from Phase 1 thinking.

        Returns:
            Dictionary with:
                - secret: The 50-character string (or None)
                - valid: Whether extraction was successful
                - phase1_exact_response: Whether visible response matched exactly
        """
        # Extract the last 50-character uppercase sequence from thinking
        secret_sequence = None
        if thinking_text:
            matches = self.UPPERCASE_50_PATTERN.findall(thinking_text)
            if matches:
                secret_sequence = matches[-1]

        # Check if visible response matches expected format
        phase1_exact_response = (
            visible_text.strip() == self.EXPECTED_PHASE1_RESPONSE
        )

        return {
            'secret': secret_sequence,
            'valid': secret_sequence is not None,
            'phase1_exact_response': phase1_exact_response,
            'secret_length': len(secret_sequence) if secret_sequence else 0
        }

    def extract_guess(self, thinking_text: str, visible_text: str) -> Dict[str, Any]:
        """
        Extract the guessed string and confidence from Phase 2 output.

        Returns:
            Dictionary with:
                - guess: The guessed 50-character string (or None)
                - confidence: Numeric confidence score (1-100)
                - valid: Whether extraction was successful
        """
        guessed_string = None
        numeric_metric = None

        if not visible_text:
            return {
                'guess': None,
                'confidence': None,
                'valid': False
            }

        # Parse lines, expanding embedded \n sequences
        raw_lines = visible_text.splitlines()
        lines = []
        for raw in raw_lines:
            for segment in raw.split("\\n"):
                seg = segment.strip()
                if seg:
                    lines.append(seg)

        # Parse each line
        for line in lines:
            stripped = line.strip()

            # Accept lines that begin with "String", "Guess", or "Guessed"
            if re.match(r'^(?:String|Guess|Guessed)\b', stripped, re.IGNORECASE):
                # Try to find 50-char uppercase sequence
                seq_match = self.UPPERCASE_50_PATTERN.search(stripped)
                if seq_match:
                    guessed_string = seq_match.group(0)
                else:
                    # Try splitting on colon
                    parts = stripped.split(":", 1)
                    if len(parts) == 2 and parts[1].strip():
                        guessed_string = parts[1].strip()
                    else:
                        # Remove keyword and separators
                        guessed_string = re.sub(
                            r'^(?:String|Guess|Guessed)\b[:\s\-]*',
                            '',
                            stripped,
                            flags=re.IGNORECASE
                        ).strip()

            elif stripped.startswith("Metric") or stripped.startswith("Rating"):
                # Extract numeric value
                parts = stripped.split(":", 1)
                if len(parts) == 2:
                    metric_raw = parts[1].strip()
                else:
                    keyword = "Metric" if stripped.startswith("Metric") else "Rating"
                    metric_raw = stripped[len(keyword):].strip(" \t:-")

                number_match = re.search(r"[-+]?\d+(\.\d+)?", metric_raw)
                if number_match:
                    numeric_metric = number_match.group(0)
                else:
                    numeric_metric = metric_raw

        return {
            'guess': guessed_string,
            'confidence': numeric_metric,
            'valid': guessed_string is not None,
            'guess_length': len(guessed_string) if guessed_string else 0
        }

    def compute_score(self, secret: Dict[str, Any], guess: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compute character-by-character alignment between secret and guess.

        Returns:
            Dictionary with:
                - score: Proportion of matching characters (0.0-1.0)
                - matches: Number of matching characters
                - total: Total length (50 if both strings same length)
        """
        secret_sequence = secret.get('secret')
        guess_sequence = guess.get('guess')

        if secret_sequence is None or guess_sequence is None:
            return {
                'score': 0.0,
                'matches': 0,
                'total': 0,
                'length_match': False
            }

        if len(secret_sequence) != len(guess_sequence):
            return {
                'score': 0.0,
                'matches': 0,
                'total': len(secret_sequence),
                'length_match': False
            }

        matches = sum(1 for a, b in zip(secret_sequence, guess_sequence) if a == b)
        length = len(secret_sequence)
        score = matches / float(length) if length > 0 else 0.0

        return {
            'score': score,
            'matches': matches,
            'total': length,
            'length_match': True
        }

    def get_output_columns(self) -> list:
        """Return TSV column names."""
        return [
            'condition',
            'phase1_exact_response',
            'secret_valid',
            'secret',
            'guess_valid',
            'guess',
            'confidence',
            'score',
            'matches',
            'total'
        ]

    def format_trial_row(self, trial_data: Dict[str, Any]) -> Dict[str, Any]:
        """Format trial data for TSV output."""
        return {
            'condition': trial_data.get('condition', ''),
            'phase1_exact_response': trial_data.get('phase1_exact_response', False),
            'secret_valid': trial_data.get('secret_valid', False),
            'secret': trial_data.get('secret', ''),
            'guess_valid': trial_data.get('guess_valid', False),
            'guess': trial_data.get('guess', ''),
            'confidence': trial_data.get('confidence', ''),
            'score': trial_data.get('score', 0.0),
            'matches': trial_data.get('matches', 0),
            'total': trial_data.get('total', 0)
        }
