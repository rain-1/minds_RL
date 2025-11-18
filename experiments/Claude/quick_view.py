#!/usr/bin/env python3
"""
Quick view of experiment results - just the key numbers.

Usage:
    python quick_view.py <output_directory>
"""

import sys
import os
import csv


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

        # Calculate averages
        scores = [float(r.get('score', 0)) for r in data if r.get('score')]
        exact = [int(r.get('exact_matches', 0)) for r in data if r.get('exact_matches')]
        confs = [float(r.get('confidence', 0)) for r in data if r.get('confidence')]

        if scores:
            print(f"  Avg Score:      {sum(scores)/len(scores):.3f}  (0.0-1.0)")
        if exact:
            print(f"  Exact Matches:  {sum(exact)/len(exact):.2f}  (out of 4)")
        if confs:
            print(f"  Confidence:     {sum(confs)/len(confs):.1f}  (0-100)")

    # Show each trial briefly
    print("\n" + "-" * 60)
    print("Individual Trials:")
    for i, row in enumerate(rows, 1):
        cond = row.get('condition', '?')[0].upper()  # C or E
        score = row.get('score', '?')
        conf = row.get('confidence', '?')

        # Count matches
        matches = 0
        for item in ['animal', 'color', 'clothing', 'location']:
            try:
                if float(row.get(f'{item}_match', 0)) >= 0.5:
                    matches += 1
            except (ValueError, TypeError):
                pass

        print(f"  Trial {i:2d} [{cond}]: Score={score:5s} Confidence={conf:5s} Matches={matches}/4")

    print()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python quick_view.py <output_directory>")
        sys.exit(1)

    quick_view(sys.argv[1])
