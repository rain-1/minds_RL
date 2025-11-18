#!/usr/bin/env python3
"""
Quick view of experiment results - just the key numbers.

Usage:
    python quick_view.py <output_directory>
"""

import sys
import os
import csv
import re


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


def quick_view(output_dir):
    """Show quick summary of results."""
    summary_path = os.path.join(output_dir, 'summary.tsv')

    if not os.path.exists(summary_path):
        print(f"❌ Not found: {summary_path}")
        return

    with open(summary_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        rows = list(reader)

    if not rows:
        print("No data")
        return

    print(f"\n📊 {output_dir}")
    print("=" * 60)

    # Separate by condition
    control = [r for r in rows if r.get('condition') == 'control']
    experimental = [r for r in rows if r.get('condition') == 'experimental']

    for name, data in [('CONTROL', control), ('EXPERIMENTAL', experimental)]:
        if not data:
            continue

        print(f"\n{name} (n={len(data)}):")

        # Calculate averages - use safe parsing
        scores = []
        exact = []
        confs = []

        for r in data:
            score_val = safe_float(r.get('score'))
            if score_val is not None:
                scores.append(score_val)

            exact_val = safe_int(r.get('exact_matches'))
            if exact_val is not None:
                exact.append(exact_val)

            conf_val = safe_float(r.get('confidence'))
            if conf_val is not None:
                confs.append(conf_val)

        if scores:
            print(f"  Avg Score:      {sum(scores)/len(scores):.3f}  (0.0-1.0)")
        if exact:
            print(f"  Exact Matches:  {sum(exact)/len(exact):.2f}  (out of 4)")
        if confs:
            print(f"  Confidence:     {sum(confs)/len(confs):.1f}  (0-100)")
            if len(confs) < len(data):
                print(f"    (Note: {len(data)-len(confs)} trials had unparseable confidence)")

    # Show each trial briefly
    print("\n" + "-" * 60)
    print("Individual Trials:")
    for i, row in enumerate(rows, 1):
        cond = row.get('condition', '?')[0].upper()  # C or E
        score_raw = row.get('score', '?')
        conf_raw = row.get('confidence', '?')

        # Format score
        score_val = safe_float(score_raw)
        score = f"{score_val:.3f}" if score_val is not None else str(score_raw)

        # Format confidence (keep original if text, show numeric if parsed)
        conf_val = safe_float(conf_raw)
        if conf_val is not None and conf_raw != str(conf_val):
            conf = f"{conf_raw} ({conf_val:.0f})"
        else:
            conf = str(conf_raw)

        # Count matches
        matches = 0
        for item in ['animal', 'color', 'clothing', 'location']:
            match_val = safe_float(row.get(f'{item}_match', 0))
            if match_val and match_val >= 0.5:
                matches += 1

        print(f"  Trial {i:2d} [{cond}]: Score={score:7s} Conf={conf:20s} Matches={matches}/4")

    print()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python quick_view.py <output_directory>")
        sys.exit(1)

    quick_view(sys.argv[1])
