"""
Pluggable task-based inference system for cognitive experiments.

This refactored version allows easy switching between different tasks
(e.g., character recall, visualization recall) by changing a single
configuration variable.
"""

import os
import json
import time
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

import anthropic

from tasks import CharacterRecallTask, VisualizationRecallTask

# =============================================================================
# CONFIGURATION
# =============================================================================

# Model and inference settings
MODEL_NAME = "claude-haiku-4-5"
MAX_TOKENS = 40000
THINKING_BUDGET_TOKENS = 20000

# Experiment parameters
NUM_TRIALS_CONTROL = 10
NUM_TRIALS_EXPERIMENTAL = 10
MAX_PARALLEL_TRIALS = 1

# =============================================================================
# TASK SELECTION - Change this to switch tasks!
# =============================================================================

# Choose your task:
# - CharacterRecallTask() for 50-character string recall
# - VisualizationRecallTask() for animal/color/clothing/location recall

TASK = VisualizationRecallTask()
# TASK = CharacterRecallTask()

# =============================================================================

# Output directory based on task name
OUTPUT_DIR = f"{MODEL_NAME.replace('-', '_')}_{TASK.get_task_name()}_experiment"
SUMMARY_PATH = os.path.join(OUTPUT_DIR, "summary.tsv")
TRIALS_JSONL_PATH = os.path.join(OUTPUT_DIR, "trials.jsonl")

# Anthropic client
client = anthropic.Anthropic()


# =============================================================================
# Helper Functions
# =============================================================================

def join_thinking_blocks(content_blocks):
    """Extract and join all thinking blocks from response."""
    print("[join_thinking_blocks] Joining thinking blocks", flush=True)
    parts = []
    for block in content_blocks:
        if block.type == "thinking":
            parts.append(block.thinking)
    total_len = sum(len(p) for p in parts)
    print(
        f"[join_thinking_blocks] Joined {len(parts)} thinking blocks into {total_len} characters",
        flush=True,
    )
    return "\n".join(parts)


def join_text_blocks(content_blocks):
    """Extract and join all text blocks from response."""
    print("[join_text_blocks] Joining text blocks", flush=True)
    parts = []
    for block in content_blocks:
        if block.type == "text":
            parts.append(block.text)
    total_len = sum(len(p) for p in parts)
    print(
        f"[join_text_blocks] Joined {len(parts)} text blocks into {total_len} characters",
        flush=True,
    )
    return "\n".join(parts)


def ensure_output_dir():
    """Create output directory if it doesn't exist."""
    print(
        f"[ensure_output_dir] Ensuring output directory exists at {OUTPUT_DIR}",
        flush=True,
    )
    if not os.path.isdir(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        print(f"[ensure_output_dir] Created directory {OUTPUT_DIR}", flush=True)
    else:
        print(f"[ensure_output_dir] Directory {OUTPUT_DIR} already exists", flush=True)


def escape_for_tsv(value):
    """Escape special characters for TSV format."""
    if value is None:
        return ""
    return str(value).replace("\t", "\\t").replace("\n", "\\n")


def ensure_summary_header():
    """Create summary TSV with header if it doesn't exist."""
    print(
        f"[ensure_summary_header] Checking for summary TSV at {SUMMARY_PATH}",
        flush=True,
    )
    if not os.path.exists(SUMMARY_PATH):
        with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
            header = TASK.get_output_columns()
            f.write("\t".join(header) + "\n")
        print("[ensure_summary_header] Wrote TSV header", flush=True)
    else:
        print("[ensure_summary_header] Summary TSV already exists", flush=True)


def append_summary_row(trial_record):
    """Append a trial record to the summary TSV."""
    print("[append_summary_row] Appending trial record to summary TSV", flush=True)
    with open(SUMMARY_PATH, "a", encoding="utf-8") as f:
        # Use task-specific formatting
        row_data = TASK.format_trial_row(trial_record)
        fields = [escape_for_tsv(row_data.get(col, "")) for col in TASK.get_output_columns()]
        f.write("\t".join(fields) + "\n")
    print("[append_summary_row] Summary TSV updated", flush=True)


def append_trial_jsonl(trial_record):
    """Append a trial record to the JSONL log."""
    print("[append_trial_jsonl] Appending trial record to JSONL log", flush=True)
    with open(TRIALS_JSONL_PATH, "a", encoding="utf-8") as f:
        json.dump(trial_record, f)
        f.write("\n")
    print("[append_trial_jsonl] JSONL log updated", flush=True)


# =============================================================================
# Main Trial Logic
# =============================================================================

def run_single_trial(trial_index, condition):
    """
    Run a single two-phase trial using the configured task.

    Args:
        trial_index: Index of this trial
        condition: "control" or "experimental"

    Returns:
        Dictionary containing all trial data
    """
    print(
        f"[run_single_trial] Starting trial index={trial_index} condition={condition}",
        flush=True,
    )
    timestamp_iso = datetime.utcnow().isoformat() + "Z"
    print(f"[run_single_trial] Timestamp: {timestamp_iso}", flush=True)

    # Phase 1: Generation/Selection
    include_context = (condition == "experimental")
    phase1_prompt = TASK.get_phase1_prompt(include_context=include_context)
    print(
        f"[run_single_trial] Phase 1 prompt length: {len(phase1_prompt)} characters",
        flush=True,
    )

    conversation = [{"role": "user", "content": phase1_prompt}]
    print("[run_single_trial] Conversation initialized with user prompt", flush=True)

    print("[run_single_trial] Sending phase 1 request to Anthropic", flush=True)
    response1 = client.messages.create(
        model=MODEL_NAME,
        max_tokens=MAX_TOKENS,
        thinking={"type": "enabled", "budget_tokens": THINKING_BUDGET_TOKENS},
        messages=conversation,
        timeout=600,
    )
    print(
        f"[run_single_trial] Phase 1 response received (id={response1.id}, model={response1.model})",
        flush=True,
    )

    phase1_thinking_text = join_thinking_blocks(response1.content)
    phase1_visible_text = join_text_blocks(response1.content)
    print(
        f"[run_single_trial] Phase 1 thinking length: {len(phase1_thinking_text)} characters",
        flush=True,
    )
    print(
        f"[run_single_trial] Phase 1 visible text length: {len(phase1_visible_text)} characters",
        flush=True,
    )

    # Extract secret using task-specific logic
    secret_data = TASK.extract_secret(phase1_thinking_text, phase1_visible_text)
    print(
        f"[run_single_trial] Phase 1 secret extraction: valid={secret_data.get('valid')}, data={secret_data}",
        flush=True,
    )

    # Phase 2: Recall
    conversation.append({"role": "assistant", "content": response1.content})
    print("[run_single_trial] Appended phase 1 response to conversation", flush=True)

    phase2_prompt = TASK.get_phase2_prompt()
    print(
        f"[run_single_trial] Phase 2 prompt length: {len(phase2_prompt)} characters",
        flush=True,
    )
    conversation.append({"role": "user", "content": phase2_prompt})
    print("[run_single_trial] Added phase 2 user prompt to conversation", flush=True)

    print("[run_single_trial] Sending phase 2 request to Anthropic", flush=True)
    response2 = client.messages.create(
        model=MODEL_NAME,
        max_tokens=MAX_TOKENS,
        thinking={"type": "enabled", "budget_tokens": THINKING_BUDGET_TOKENS},
        messages=conversation,
        timeout=600,
    )
    print(
        f"[run_single_trial] Phase 2 response received (id={response2.id}, model={response2.model})",
        flush=True,
    )

    phase2_thinking_text = join_thinking_blocks(response2.content)
    phase2_visible_text = join_text_blocks(response2.content)
    print(
        f"[run_single_trial] Phase 2 thinking length: {len(phase2_thinking_text)} characters",
        flush=True,
    )
    print(
        f"[run_single_trial] Phase 2 visible text length: {len(phase2_visible_text)} characters",
        flush=True,
    )

    # Extract guess using task-specific logic
    guess_data = TASK.extract_guess(phase2_thinking_text, phase2_visible_text)
    print(
        f"[run_single_trial] Phase 2 guess extraction: valid={guess_data.get('valid')}, data={guess_data}",
        flush=True,
    )

    # Compute score using task-specific logic
    score_data = TASK.compute_score(secret_data, guess_data)
    print(
        f"[run_single_trial] Score computation: {score_data}",
        flush=True,
    )

    # Assemble trial record
    trial_record = {
        "timestamp_iso": timestamp_iso,
        "trial_index": trial_index,
        "condition": condition,
        "task_name": TASK.get_task_name(),
        "phase1_prompt": phase1_prompt,
        "phase2_prompt": phase2_prompt,
        "phase1_thinking": phase1_thinking_text,
        "phase1_visible_text": phase1_visible_text,
        "phase2_thinking": phase2_thinking_text,
        "phase2_visible_text": phase2_visible_text,
        "response1_id": response1.id,
        "response2_id": response2.id,
        "response1_model": response1.model,
        "response2_model": response2.model,
    }

    # Merge in task-specific data
    trial_record.update({
        **secret_data,
        **{f"guess_{k}" if k in secret_data else k: v for k, v in guess_data.items()},
        **score_data
    })

    print("[run_single_trial] Trial record assembled", flush=True)
    return trial_record


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    """Run the complete experiment."""
    print(f"[main] Starting experiment run with task: {TASK.get_task_name()}", flush=True)
    ensure_output_dir()
    ensure_summary_header()

    # Schedule trials
    trial_configs = []
    trial_index = 0
    for condition, num_trials in [
        ("control", NUM_TRIALS_CONTROL),
        ("experimental", NUM_TRIALS_EXPERIMENTAL),
    ]:
        print(f"[main] Entering condition setup loop: {condition}", flush=True)
        for _ in range(num_trials):
            print(
                f"[main] Scheduling trial {trial_index} condition={condition}",
                flush=True,
            )
            trial_configs.append((trial_index, condition))
            trial_index += 1

    print(
        f"[main] Total trials scheduled: {len(trial_configs)}; running with up to {MAX_PARALLEL_TRIALS} parallel trials",
        flush=True,
    )

    # Execute trials
    trial_records = []
    with ThreadPoolExecutor(max_workers=MAX_PARALLEL_TRIALS) as executor:
        futures = []
        for t_index, cond in trial_configs:
            futures.append(executor.submit(run_single_trial, t_index, cond))
        for future in futures:
            record = future.result()
            trial_records.append(record)
            print(
                f"[main] Collected trial {record.get('trial_index')} condition={record.get('condition')}",
                flush=True,
            )

    # Sort and persist results
    trial_records.sort(key=lambda r: r.get("trial_index", 0))
    print(
        f"[main] Writing {len(trial_records)} trial records to JSONL and TSV in trial_index order",
        flush=True,
    )

    for record in trial_records:
        append_trial_jsonl(record)
        append_summary_row(record)
        print(
            f"[main] Persisted trial {record.get('trial_index')} condition={record.get('condition')}",
            flush=True,
        )

    print("[main] All trials completed and persisted", flush=True)


if __name__ == "__main__":
    main()
