#!/usr/bin/env python3
"""
Analyze visualization recall experiment results.

Usage:
    python analyze_results.py <output_directory>
    python analyze_results.py claude_haiku_4_5_visualization_recall_generation_experiment_experiment
"""

import sys
import os
import csv
import re
from collections import defaultdict


def safe_float(value, default=None):
    """Safely convert value to float, handling text descriptions."""
    if not value:
        return default

    # Try direct conversion first
    try:
        return float(value)
    except (ValueError, TypeError):
        pass

    # Try to extract number from text
    value_str = str(value).strip()
    number_match = re.search(r'[-+]?\d*\.?\d+', value_str)
    if number_match:
        try:
            return float(number_match.group())
        except ValueError:
            pass

    # Map text descriptions to approximate numbers
    text_lower = value_str.lower()
    if 'very high' in text_lower or 'extremely high' in text_lower:
        return 90
    elif 'high' in text_lower:
        return 75
    elif 'medium-high' in text_lower or 'moderate-high' in text_lower:
        return 65
    elif 'medium' in text_lower or 'moderate' in text_lower:
        return 50
    elif 'medium-low' in text_lower or 'moderate-low' in text_lower:
        return 35
    elif 'low' in text_lower:
        return 25
    elif 'very low' in text_lower or 'extremely low' in text_lower:
        return 10

    return default


def safe_int(value, default=None):
    """Safely convert value to int."""
    if not value:
        return default
    try:
        return int(value)
    except (ValueError, TypeError):
        pass

    # Try float first then int
    try:
        return int(float(value))
    except (ValueError, TypeError):
        return default


def load_tsv(filepath):
    """Load TSV file and return rows as dicts."""
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        return list(reader)


def analyze_results(output_dir):
    """Analyze experiment results from output directory."""
    summary_path = os.path.join(output_dir, 'summary.tsv')

    if not os.path.exists(summary_path):
        print(f"Error: {summary_path} not found")
        return

    print("=" * 80)
    print(f"ANALYSIS: {output_dir}")
    print("=" * 80)

    rows = load_tsv(summary_path)

    if not rows:
        print("No data found in summary.tsv")
        return

    # Overall statistics
    print(f"\nTotal trials: {len(rows)}")

    # By condition
    by_condition = defaultdict(list)
    for row in rows:
        by_condition[row.get('condition', 'unknown')].append(row)

    for condition, cond_rows in sorted(by_condition.items()):
        print(f"\n{condition.upper()} Condition: {len(cond_rows)} trials")
        print("-" * 80)

        # Calculate statistics
        scores = []
        exact_matches = []
        partial_matches = []
        confidences = []
        phase1_success = 0

        for row in cond_rows:
            score_val = safe_float(row.get('score'))
            if score_val is not None:
                scores.append(score_val)

            exact_val = safe_int(row.get('exact_matches'))
            if exact_val is not None:
                exact_matches.append(exact_val)

            partial_val = safe_int(row.get('partial_matches'))
            if partial_val is not None:
                partial_matches.append(partial_val)

            conf_val = safe_float(row.get('confidence'))
            if conf_val is not None:
                confidences.append(conf_val)

            if row.get('phase1_exact_response', '').lower() == 'true':
                phase1_success += 1

        # Print statistics
        if scores:
            avg_score = sum(scores) / len(scores)
            print(f"  Average Score: {avg_score:.3f} (0.0 = no match, 1.0 = perfect)")

        if exact_matches:
            avg_exact = sum(exact_matches) / len(exact_matches)
            print(f"  Average Exact Matches: {avg_exact:.2f} / 4 items")

        if partial_matches:
            avg_partial = sum(partial_matches) / len(partial_matches)
            print(f"  Average Partial Matches: {avg_partial:.2f} / 4 items")

        if confidences:
            avg_conf = sum(confidences) / len(confidences)
            print(f"  Average Confidence: {avg_conf:.1f} / 100")

        print(f"  Phase 1 Success Rate: {phase1_success}/{len(cond_rows)} " +
              f"({100*phase1_success/len(cond_rows):.1f}%)")

    # Per-item breakdown
    print("\n" + "=" * 80)
    print("PER-ITEM BREAKDOWN")
    print("=" * 80)

    items = ['animal', 'color', 'clothing', 'location']

    for condition, cond_rows in sorted(by_condition.items()):
        print(f"\n{condition.upper()} Condition:")
        print("-" * 40)

        for item in items:
            match_col = f'{item}_match'
            matches = []

            for row in cond_rows:
                match_val = safe_float(row.get(match_col))
                if match_val is not None:
                    matches.append(match_val)

            if matches:
                avg_match = sum(matches) / len(matches)
                print(f"  {item.capitalize():12s}: {avg_match:.3f} avg")

    # Individual trial details
    print("\n" + "=" * 80)
    print("INDIVIDUAL TRIALS")
    print("=" * 80)

    for i, row in enumerate(rows, 1):
        condition = row.get('condition', 'unknown')
        score = row.get('score', 'N/A')
        conf = row.get('confidence', 'N/A')

        print(f"\nTrial {i} ({condition}):")
        print(f"  Score: {score} | Confidence: {conf}")

        print(f"  Secret:  {row.get('secret_animal', '?'):12s} {row.get('secret_color', '?'):12s} " +
              f"{row.get('secret_clothing', '?'):12s} {row.get('secret_location', '?'):12s}")
        print(f"  Guess:   {row.get('guess_animal', '?'):12s} {row.get('guess_color', '?'):12s} " +
              f"{row.get('guess_clothing', '?'):12s} {row.get('guess_location', '?'):12s}")

        # Match indicators
        animal_val = safe_float(row.get('animal_match'), 0)
        color_val = safe_float(row.get('color_match'), 0)
        clothing_val = safe_float(row.get('clothing_match'), 0)
        location_val = safe_float(row.get('location_match'), 0)

        animal_match = '✓' if animal_val == 1.0 else '~' if animal_val == 0.5 else '✗'
        color_match = '✓' if color_val == 1.0 else '~' if color_val == 0.5 else '✗'
        clothing_match = '✓' if clothing_val == 1.0 else '~' if clothing_val == 0.5 else '✗'
        location_match = '✓' if location_val == 1.0 else '~' if location_val == 0.5 else '✗'

        print(f"  Matches: {animal_match:12s} {color_match:12s} {clothing_match:12s} {location_match:12s}")
        print(f"           (✓=exact, ~=partial, ✗=none)")


def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze_results.py <output_directory>")
        print("\nExample:")
        print("  python analyze_results.py claude_haiku_4_5_visualization_recall_generation_experiment_experiment")
        sys.exit(1)

    output_dir = sys.argv[1]

    if not os.path.isdir(output_dir):
        print(f"Error: Directory not found: {output_dir}")
        sys.exit(1)

    analyze_results(output_dir)


if __name__ == "__main__":
    main()
