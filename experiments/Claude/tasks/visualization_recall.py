"""
Visualization Recall Task - Multi-item scene visualization and recall experiment.
"""

import re
from typing import Dict, Any, List, Tuple
from .base import Task


class VisualizationRecallTask(Task):
    """
    Task where model selects and visualizes four items (animal, color, clothing, location)
    in Phase 1 and attempts to recall them in Phase 2.
    """

    EXPECTED_PHASE1_RESPONSE = "I understand. I have visualized my scene."

    # Categories we're looking for
    CATEGORIES = ['animal', 'color', 'clothing', 'location']

    def __init__(self,
                 prompt1_path: str = "visualization_prompt1.txt",
                 prompt2_path: str = "visualization_prompt2.txt",
                 context_path: str = "context_prompt.txt"):
        """
        Initialize the visualization recall task.

        Args:
            prompt1_path: Path to Phase 1 prompt file
            prompt2_path: Path to Phase 2 prompt file
            context_path: Path to context document (for experimental condition)
        """
        self.prompt1_path = prompt1_path
        self.prompt2_path = prompt2_path
        self.context_path = context_path

    def get_task_name(self) -> str:
        return "visualization_recall"

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

    def _extract_items_from_thinking(self, thinking_text: str) -> Dict[str, str]:
        """
        Extract chosen items from Phase 1 thinking.

        Strategy: Look for explicit statements where the model makes choices.
        Look for patterns like:
        - "I choose [item]"
        - "I'll pick [item]"
        - "I select [item]"
        - "Animal: [item]", "Color: [item]", etc.
        """
        if not thinking_text:
            return {}

        items = {}

        # Strategy 1: Look for labeled choices (most explicit)
        # e.g., "Animal: elephant" or "ANIMAL: elephant"
        for category in self.CATEGORIES:
            # Try various formats: "Animal:", "ANIMAL:", "animal:", etc.
            patterns = [
                rf'\b{category}\s*:\s*([a-zA-Z\s]+?)(?:\n|$|\.|\,)',
                rf'\b{category.upper()}\s*:\s*([a-zA-Z\s]+?)(?:\n|$|\.|\,)',
                rf'\b{category.capitalize()}\s*:\s*([a-zA-Z\s]+?)(?:\n|$|\.|\,)',
            ]

            for pattern in patterns:
                matches = re.findall(pattern, thinking_text, re.IGNORECASE)
                if matches:
                    # Take the first match, strip and clean
                    item = matches[0].strip()
                    # Remove common words that might be captured
                    item = re.sub(r'\b(the|a|an)\b', '', item, flags=re.IGNORECASE).strip()
                    if item and len(item) > 1:
                        items[category] = item
                        break

        # Strategy 2: Look for choice statements
        # e.g., "I choose a dolphin as the animal"
        choice_patterns = [
            r"(?:choose|pick|select)\s+(?:a|an|the)?\s*([a-zA-Z\s]+?)\s+(?:as|for)\s+(?:the|my)?\s*({})".format('|'.join(self.CATEGORIES)),
            r"(?:choose|pick|select)\s+(?:the|a|an)?\s*({})\s*:\s*([a-zA-Z\s]+)".format('|'.join(self.CATEGORIES)),
        ]

        for pattern in choice_patterns:
            for match in re.finditer(pattern, thinking_text, re.IGNORECASE):
                if len(match.groups()) >= 2:
                    item = match.group(1).strip()
                    category = match.group(2).lower().strip()
                    if category in self.CATEGORIES and category not in items:
                        items[category] = item

        # Strategy 3: Look for last occurrence of quoted or emphasized words
        # This is less reliable, so only use if we haven't found items yet
        for category in self.CATEGORIES:
            if category not in items:
                # Look for words in quotes after mentioning the category
                pattern = rf'\b{category}\b.*?["\']([a-zA-Z\s]+?)["\']'
                matches = re.findall(pattern, thinking_text, re.IGNORECASE)
                if matches:
                    items[category] = matches[-1].strip()

        return items

    def extract_secret(self, thinking_text: str, visible_text: str) -> Dict[str, Any]:
        """
        Extract the four chosen items from Phase 1 thinking.

        Returns:
            Dictionary with:
                - secret: Dict mapping category to chosen item
                - valid: Whether all four items were successfully extracted
                - phase1_exact_response: Whether visible response matched exactly
                - items_found: Number of items successfully extracted (0-4)
        """
        items = self._extract_items_from_thinking(thinking_text)

        # Check if visible response matches expected format
        phase1_exact_response = (
            visible_text.strip() == self.EXPECTED_PHASE1_RESPONSE
        )

        items_found = len(items)
        all_found = items_found == len(self.CATEGORIES)

        return {
            'secret': items,
            'valid': all_found,
            'phase1_exact_response': phase1_exact_response,
            'items_found': items_found,
            'animal': items.get('animal', ''),
            'color': items.get('color', ''),
            'clothing': items.get('clothing', ''),
            'location': items.get('location', '')
        }

    def _extract_items_from_recall(self, visible_text: str) -> Tuple[Dict[str, str], str]:
        """
        Extract guessed items and confidence from Phase 2 visible output.

        Returns:
            Tuple of (items_dict, confidence_str)
        """
        if not visible_text:
            return {}, None

        items = {}
        confidence = None

        # Parse lines
        lines = visible_text.splitlines()

        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue

            # Try to match "Category: value" format
            for category in self.CATEGORIES:
                pattern = rf'\b{category}\s*:\s*(.+)$'
                match = re.match(pattern, stripped, re.IGNORECASE)
                if match:
                    value = match.group(1).strip()
                    # Remove quotes if present
                    value = value.strip('"\'')
                    items[category] = value
                    break

            # Look for confidence
            if 'confidence' in stripped.lower():
                pattern = r'confidence\s*:\s*(.+)$'
                match = re.match(pattern, stripped, re.IGNORECASE)
                if match:
                    conf_str = match.group(1).strip()
                    # Extract numeric value
                    number_match = re.search(r'[-+]?\d+(\.\d+)?', conf_str)
                    if number_match:
                        confidence = number_match.group(0)
                    else:
                        confidence = conf_str

        return items, confidence

    def extract_guess(self, thinking_text: str, visible_text: str) -> Dict[str, Any]:
        """
        Extract the guessed items and confidence from Phase 2 output.

        Returns:
            Dictionary with:
                - guess: Dict mapping category to guessed item
                - confidence: Numeric confidence score (1-100)
                - valid: Whether all four items were extracted
                - items_found: Number of items successfully extracted
        """
        items, confidence = self._extract_items_from_recall(visible_text)

        items_found = len(items)
        all_found = items_found == len(self.CATEGORIES)

        return {
            'guess': items,
            'confidence': confidence,
            'valid': all_found,
            'items_found': items_found,
            'animal': items.get('animal', ''),
            'color': items.get('color', ''),
            'clothing': items.get('clothing', ''),
            'location': items.get('location', '')
        }

    def compute_score(self, secret: Dict[str, Any], guess: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compute match score between secret and guessed items.

        Scoring:
        - Exact match (case-insensitive): 1.0 per item
        - Partial match (one word matches): 0.5 per item
        - No match: 0.0 per item

        Returns:
            Dictionary with:
                - score: Average score across all items (0.0-1.0)
                - matches: Number of exact matches
                - partial_matches: Number of partial matches
                - total: Total number of items (4)
                - per_category: Dict with score for each category
        """
        secret_items = secret.get('secret', {})
        guess_items = guess.get('guess', {})

        exact_matches = 0
        partial_matches = 0
        per_category = {}

        for category in self.CATEGORIES:
            secret_val = secret_items.get(category, '').lower().strip()
            guess_val = guess_items.get(category, '').lower().strip()

            if not secret_val or not guess_val:
                per_category[category] = 0.0
                continue

            # Check exact match
            if secret_val == guess_val:
                exact_matches += 1
                per_category[category] = 1.0
            else:
                # Check partial match (any word overlap)
                secret_words = set(secret_val.split())
                guess_words = set(guess_val.split())

                if secret_words & guess_words:  # Non-empty intersection
                    partial_matches += 1
                    per_category[category] = 0.5
                else:
                    per_category[category] = 0.0

        total = len(self.CATEGORIES)
        total_score = sum(per_category.values())
        avg_score = total_score / total if total > 0 else 0.0

        return {
            'score': avg_score,
            'matches': exact_matches,
            'partial_matches': partial_matches,
            'total': total,
            'per_category': per_category,
            'animal_match': per_category.get('animal', 0.0),
            'color_match': per_category.get('color', 0.0),
            'clothing_match': per_category.get('clothing', 0.0),
            'location_match': per_category.get('location', 0.0)
        }

    def get_output_columns(self) -> list:
        """Return TSV column names."""
        return [
            'condition',
            'phase1_exact_response',
            'secret_valid',
            'secret_animal',
            'secret_color',
            'secret_clothing',
            'secret_location',
            'guess_valid',
            'guess_animal',
            'guess_color',
            'guess_clothing',
            'guess_location',
            'confidence',
            'score',
            'exact_matches',
            'partial_matches',
            'animal_match',
            'color_match',
            'clothing_match',
            'location_match'
        ]

    def format_trial_row(self, trial_data: Dict[str, Any]) -> Dict[str, Any]:
        """Format trial data for TSV output."""
        return {
            'condition': trial_data.get('condition', ''),
            'phase1_exact_response': trial_data.get('phase1_exact_response', False),
            'secret_valid': trial_data.get('secret_valid', False),
            'secret_animal': trial_data.get('secret', {}).get('animal', ''),
            'secret_color': trial_data.get('secret', {}).get('color', ''),
            'secret_clothing': trial_data.get('secret', {}).get('clothing', ''),
            'secret_location': trial_data.get('secret', {}).get('location', ''),
            'guess_valid': trial_data.get('guess_valid', False),
            'guess_animal': trial_data.get('guess', {}).get('animal', ''),
            'guess_color': trial_data.get('guess', {}).get('color', ''),
            'guess_clothing': trial_data.get('guess', {}).get('clothing', ''),
            'guess_location': trial_data.get('guess', {}).get('location', ''),
            'confidence': trial_data.get('confidence', ''),
            'score': trial_data.get('score', 0.0),
            'exact_matches': trial_data.get('matches', 0),
            'partial_matches': trial_data.get('partial_matches', 0),
            'animal_match': trial_data.get('animal_match', 0.0),
            'color_match': trial_data.get('color_match', 0.0),
            'clothing_match': trial_data.get('clothing_match', 0.0),
            'location_match': trial_data.get('location_match', 0.0)
        }
