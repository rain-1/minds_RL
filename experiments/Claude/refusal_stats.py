#!/usr/bin/env python3
"""
Scan a JSONL file of trial records and print refusal statistics.

Outputs:
- Number of rows
- (failures) Number of rows with secret_sequence null
- (initiates) Number of rows with secret_sequence not null
- (refusals) Number of rows with secret_sequence not null and guessed_string null
- (refusal % of initiates)

Usage:
  python refusal_stats.py /path/to/trials.jsonl
  cat /path/to/trials.jsonl | python refusal_stats.py -
"""
import sys
import json
import argparse


def parse_args():
    p = argparse.ArgumentParser(description="Compute refusal stats from a JSONL trials file")
    p.add_argument("file", nargs="?", default="-", help="Path to JSONL file (use - or omit for stdin)")
    return p.parse_args()


def open_input(path):
    if path == "-":
        return sys.stdin
    return open(path, "r", encoding="utf-8")


def main():
    args = parse_args()
    total = 0
    failures = 0
    initiates = 0
    refusals = 0

    try:
        with open_input(args.file) as fh:
            for raw in fh:
                line = raw.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    # skip invalid JSON lines but count them as total rows
                    total += 1
                    continue

                total += 1
                secret = obj.get("secret_sequence")
                guessed = obj.get("guessed_string")

                if secret is None:
                    failures += 1
                else:
                    initiates += 1
                    if guessed is None:
                        refusals += 1

    except FileNotFoundError:
        print(f"File not found: {args.file}", file=sys.stderr)
        sys.exit(2)
    except KeyboardInterrupt:
        print("Interrupted", file=sys.stderr)
        sys.exit(1)

    pct = 0.0
    if initiates > 0:
        pct = (refusals / initiates) * 100.0

    print(f"Number of rows: {total}")
    print(f"(failures) Number of rows with secret_sequence null: {failures}")
    print(f"(initiates) Number of rows with secret_sequence not null: {initiates}")
    print(f"(refusals) Number of rows with secret_sequence not null and guessed_string null: {refusals}")
    print(f"(refusal % of initiates): {pct:.2f}%")


if __name__ == "__main__":
    main()
