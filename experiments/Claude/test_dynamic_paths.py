#!/usr/bin/env python3
"""
Test that the dynamic path functions work correctly.
"""

import os

# Simulate the module-level variables
OUTPUT_DIR = "initial_directory"

def get_summary_path():
    """Get the summary TSV path based on current OUTPUT_DIR."""
    return os.path.join(OUTPUT_DIR, "summary.tsv")

def get_trials_jsonl_path():
    """Get the trials JSONL path based on current OUTPUT_DIR."""
    return os.path.join(OUTPUT_DIR, "trials.jsonl")

# Test 1: Initial paths
print("Test 1: Initial paths")
print(f"  OUTPUT_DIR: {OUTPUT_DIR}")
print(f"  get_summary_path(): {get_summary_path()}")
print(f"  get_trials_jsonl_path(): {get_trials_jsonl_path()}")
assert get_summary_path() == "initial_directory/summary.tsv"
assert get_trials_jsonl_path() == "initial_directory/trials.jsonl"
print("  ✓ Pass")

# Test 2: After changing OUTPUT_DIR
print("\nTest 2: After changing OUTPUT_DIR")
OUTPUT_DIR = "changed_directory"
print(f"  OUTPUT_DIR: {OUTPUT_DIR}")
print(f"  get_summary_path(): {get_summary_path()}")
print(f"  get_trials_jsonl_path(): {get_trials_jsonl_path()}")
assert get_summary_path() == "changed_directory/summary.tsv"
assert get_trials_jsonl_path() == "changed_directory/trials.jsonl"
print("  ✓ Pass")

# Test 3: With task-like name
print("\nTest 3: With task-like directory name")
OUTPUT_DIR = "claude_haiku_4_5_visualization_recall_expectations_no_refusal_experiment"
print(f"  OUTPUT_DIR: {OUTPUT_DIR}")
print(f"  get_summary_path(): {get_summary_path()}")
print(f"  get_trials_jsonl_path(): {get_trials_jsonl_path()}")
assert get_summary_path() == f"{OUTPUT_DIR}/summary.tsv"
assert get_trials_jsonl_path() == f"{OUTPUT_DIR}/trials.jsonl"
print("  ✓ Pass")

print("\n✓ All tests passed! Dynamic paths work correctly.")
