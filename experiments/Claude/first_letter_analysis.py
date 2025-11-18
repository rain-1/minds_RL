#!/usr/bin/env python3
"""
Scan all `trials.jsonl` files under `archived_runs/`, collect non-null
`secret_sequence` and `guessed_string` values, and perform a frequency
analysis on their first letters.

Usage:
  python first_letter_analysis.py                 # uses archived_runs next to this script
  python first_letter_analysis.py /path/to/archived_runs
  cat /some/other/trials.jsonl | python first_letter_analysis.py -  # reads stdin JSONL

The script prints counts and percent breakdowns for the first letters
of the collected secret and guessed strings.
"""
from __future__ import annotations

import os
import sys
import json
from collections import Counter
from typing import Iterable, List, Tuple


def iter_jsonl_lines(path: str):
    """Yield parsed JSON objects from a JSONL file (skips blank/invalid lines)."""
    with open(path, "r", encoding="utf-8") as fh:
        for raw in fh:
            line = raw.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                # skip invalid json lines
                continue


def collect_from_file(path: str, secrets: List[str], guesses: List[str]) -> int:
    """Collect non-null secret_sequence and guessed_string from a JSONL file.

    Returns number of rows processed from this file.
    """
    rows = 0
    for obj in iter_jsonl_lines(path):
        rows += 1
        s = obj.get("secret_sequence")
        g = obj.get("guessed_string")
        if s is not None:
            secrets.append(s)
        if g is not None:
            guesses.append(g)
    return rows


def first_letter_counter(strings: Iterable[str]) -> Counter:
    c = Counter()
    for s in strings:
        if not isinstance(s, str):
            continue
        s2 = s.strip()
        if not s2:
            continue
        c[s2[0].upper()] += 1
    return c


def print_counter(name: str, counter: Counter, total: int):
    print(f"\n{name} (total {total}):")
    if total == 0:
        print("  <none>")
        return
    for letter, cnt in counter.most_common():
        pct = (cnt / total) * 100.0
        print(f"  {letter}: {cnt} ({pct:.2f}%)")


def find_trials_files(base_dir: str) -> List[str]:
    matches = []
    for root, dirs, files in os.walk(base_dir):
        for fname in files:
            if fname.lower().endswith(".jsonl") and fname.lower().startswith("trials"):
                matches.append(os.path.join(root, fname))
    return matches


def main(argv: List[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]

    # default base: archived_runs next to this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_base = os.path.join(script_dir, "archived_runs")

    if len(argv) == 0:
        base = default_base
    elif len(argv) == 1 and argv[0] == "-":
        # read JSONL from stdin and analyze a single stream
        secrets: List[str] = []
        guesses: List[str] = []
        rows = 0
        for raw in sys.stdin:
            line = raw.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            rows += 1
            s = obj.get("secret_sequence")
            g = obj.get("guessed_string")
            if s is not None:
                secrets.append(s)
            if g is not None:
                guesses.append(g)

        print(f"Scanned stdin JSONL rows: {rows}")
        sc = first_letter_counter(secrets)
        gc = first_letter_counter(guesses)
        print_counter("Secrets first-letter freq", sc, len(secrets))
        print_counter("Guesses first-letter freq", gc, len(guesses))
        return 0
    else:
        base = argv[0]

    if not os.path.isdir(base):
        print(f"Base directory not found: {base}", file=sys.stderr)
        return 2

    files = find_trials_files(base)
    if not files:
        print(f"No trials.jsonl files found under {base}")
        return 0

    secrets: List[str] = []
    guesses: List[str] = []
    total_rows = 0

    for p in files:
        rows = collect_from_file(p, secrets, guesses)
        total_rows += rows

    print(f"Found {len(files)} trials.jsonl files under {base}")
    print(f"Total JSONL rows scanned: {total_rows}")
    print(f"Collected non-null secret_sequence: {len(secrets)}")
    print(f"Collected non-null guessed_string: {len(guesses)}")

    sc = first_letter_counter(secrets)
    gc = first_letter_counter(guesses)

    print_counter("Secrets first-letter freq", sc, len(secrets))
    print_counter("Guesses first-letter freq", gc, len(guesses))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
