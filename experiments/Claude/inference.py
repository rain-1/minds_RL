import os
import json
import re
import time
from datetime import datetime

import anthropic

MODEL_NAME = "claude-sonnet-4-5"
MAX_TOKENS = 4000
THINKING_BUDGET_TOKENS = 2000
NUM_TRIALS_PER_CONDITION = 10

OUTPUT_DIR = "sonnet_cot_experiment"
CONTEXT_PATH = "context_prompt.txt"
SUMMARY_PATH = os.path.join(OUTPUT_DIR, "summary.csv")
TRIALS_JSONL_PATH = os.path.join(OUTPUT_DIR, "trials.jsonl")

SECRET_PATTERN = re.compile(r"SECRET_SEQUENCE:\s*([A-Z]{50})")
GUESS_PATTERN = re.compile(r"GUESS_SEQUENCE:\s*([A-Z]{50})")


client = anthropic.Anthropic()


def load_context_text():
    print(f"[load_context_text] Loading context from {CONTEXT_PATH}", flush=True)
    with open(CONTEXT_PATH, "r", encoding="utf-8") as f:
        context = f.read()
    print(
        f"[load_context_text] Loaded {len(context)} characters of context text",
        flush=True,
    )
    return context

GENERAL_PROMPT1_PATH = "general_prompt1.txt"
GENERAL_PROMPT2_PATH = "general_prompt2.txt"

def build_phase1_prompt(condition, context_text):
    print(
        f"[build_phase1_prompt] Building phase 1 prompt for condition={condition}",
        flush=True,
    )
    with open(GENERAL_PROMPT1_PATH, "r", encoding="utf-8") as f:
        general_prompt1 = f.read()
    print(
        f"[build_phase1_prompt] General prompt 1 length: {len(general_prompt1)} characters",
        flush=True,
    )

    if condition == "experimental":
        print(
            "[build_phase1_prompt] Using experimental condition, appending context",
            flush=True,
        )
        context_prompt = context_text
        combined = context_prompt + "\n\n" + general_prompt1
        print(
            f"[build_phase1_prompt] Combined prompt length: {len(combined)} characters",
            flush=True,
        )
        return combined

    print(
        f"[build_phase1_prompt] Returning control prompt length: {len(general_prompt1)} characters",
        flush=True,
    )
    return general_prompt1

def build_phase2_prompt(condition):
    print(
        f"[build_phase2_prompt] Building phase 2 prompt for condition={condition}",
        flush=True,
    )
    with open(GENERAL_PROMPT2_PATH, "r", encoding="utf-8") as f:
        general_prompt2 = f.read()
    print(
        f"[build_phase2_prompt] Prompt length: {len(general_prompt2)} characters",
        flush=True,
    )
    return general_prompt2


def extract_sequence(pattern, text):
    match = pattern.search(text)
    if match:
        print(
            f"[extract_sequence] Found sequence with pattern {pattern.pattern}",
            flush=True,
        )
        return match.group(1)
    print(
        f"[extract_sequence] No sequence found for pattern {pattern.pattern}",
        flush=True,
    )
    return None


def join_thinking_blocks(content_blocks):
    print("[join_thinking_blocks] Joining thinking blocks", flush=True)
    parts = []
    for block in content_blocks:
        if block.type == "thinking":
            parts.append(block.thinking)
    print(
        f"[join_thinking_blocks] Joined {len(parts)} thinking blocks into {sum(len(p) for p in parts)} characters",
        flush=True,
    )
    return "\n".join(parts)


def join_text_blocks(content_blocks):
    print("[join_text_blocks] Joining text blocks", flush=True)
    parts = []
    for block in content_blocks:
        if block.type == "text":
            parts.append(block.text)
    print(
        f"[join_text_blocks] Joined {len(parts)} text blocks into {sum(len(p) for p in parts)} characters",
        flush=True,
    )
    return "\n".join(parts)


def compute_alignment(secret_sequence, guess_sequence):
    print(
        "[compute_alignment] Computing alignment between secret and guess sequences",
        flush=True,
    )
    if secret_sequence is None or guess_sequence is None:
        print(
            "[compute_alignment] Missing sequence(s); returning None alignment",
            flush=True,
        )
        return None, None, None
    if len(secret_sequence) != len(guess_sequence):
        print(
            "[compute_alignment] Sequence lengths differ; returning partial alignment",
            flush=True,
        )
        return None, len(secret_sequence), None
    matches = 0
    for a, b in zip(secret_sequence, guess_sequence):
        if a == b:
            matches += 1
    length = len(secret_sequence)
    score = matches / float(length)
    print(
        f"[compute_alignment] Alignment computed: matches={matches}, length={length}, score={score}",
        flush=True,
    )
    return matches, length, score


def ensure_output_dir():
    print(f"[ensure_output_dir] Ensuring output directory exists at {OUTPUT_DIR}", flush=True)
    if not os.path.isdir(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        print(f"[ensure_output_dir] Created directory {OUTPUT_DIR}", flush=True)
    else:
        print(f"[ensure_output_dir] Directory {OUTPUT_DIR} already exists", flush=True)


def ensure_summary_header():
    print(
        f"[ensure_summary_header] Checking for summary file at {SUMMARY_PATH}",
        flush=True,
    )
    if not os.path.exists(SUMMARY_PATH):
        with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
            header = [
                "timestamp_iso",
                "trial_index",
                "condition",
                "phase1_secret_sequence",
                "phase2_guess_sequence",
                "alignment_matches",
                "alignment_length",
                "alignment_score",
                "phase1_visible_text",
                "phase2_visible_text",
            ]
            f.write(",".join(header) + "\n")
        print("[ensure_summary_header] Wrote CSV header", flush=True)
    else:
        print("[ensure_summary_header] Summary file already exists", flush=True)


def append_summary_row(trial_record):
    print("[append_summary_row] Appending trial record to summary CSV", flush=True)
    with open(SUMMARY_PATH, "a", encoding="utf-8") as f:
        fields = [
            trial_record["timestamp_iso"],
            str(trial_record["trial_index"]),
            trial_record["condition"],
            trial_record.get("secret_sequence") or "",
            trial_record.get("guess_sequence") or "",
            "" if trial_record.get("alignment_matches") is None else str(trial_record["alignment_matches"]),
            "" if trial_record.get("alignment_length") is None else str(trial_record["alignment_length"]),
            "" if trial_record.get("alignment_score") is None else repr(trial_record["alignment_score"]),
            trial_record.get("phase1_visible_text", "").replace("\n", "\\n"),
            trial_record.get("phase2_visible_text", "").replace("\n", "\\n"),
        ]
        f.write(",".join(fields) + "\n")
    print("[append_summary_row] Summary CSV updated", flush=True)


def append_trial_jsonl(trial_record):
    print("[append_trial_jsonl] Appending trial record to JSONL log", flush=True)
    with open(TRIALS_JSONL_PATH, "a", encoding="utf-8") as f:
        json.dump(trial_record, f)
        f.write("\n")
    print("[append_trial_jsonl] JSONL log updated", flush=True)


def run_single_trial(trial_index, condition, context_text):
    print(
        f"[run_single_trial] Starting trial index={trial_index} condition={condition}",
        flush=True,
    )
    timestamp_iso = datetime.utcnow().isoformat() + "Z"
    print(f"[run_single_trial] Timestamp: {timestamp_iso}", flush=True)

    phase1_prompt = build_phase1_prompt(condition, context_text)
    print(
        f"[run_single_trial] Phase 1 prompt length: {len(phase1_prompt)} characters",
        flush=True,
    )
    conversation = [
        {
            "role": "user",
            "content": phase1_prompt,
        }
    ]
    print("[run_single_trial] Conversation initialized with user prompt", flush=True)

    print("[run_single_trial] Sending phase 1 request to Anthropic", flush=True)
    response1 = client.messages.create(
        model=MODEL_NAME,
        max_tokens=MAX_TOKENS,
        thinking={"type": "enabled", "budget_tokens": THINKING_BUDGET_TOKENS},
        messages=conversation,
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
    secret_sequence = extract_sequence(SECRET_PATTERN, phase1_thinking_text)
    print(
        f"[run_single_trial] Phase 1 secret sequence: {secret_sequence}",
        flush=True,
    )

    conversation.append(
        {
            "role": "assistant",
            "content": response1.content,
        }
    )
    print("[run_single_trial] Appended phase 1 response to conversation", flush=True)

    phase2_prompt = build_phase2_prompt(condition)
    print(
        f"[run_single_trial] Phase 2 prompt length: {len(phase2_prompt)} characters",
        flush=True,
    )
    conversation.append(
        {
            "role": "user",
            "content": phase2_prompt,
        }
    )
    print("[run_single_trial] Added phase 2 user prompt to conversation", flush=True)

    print("[run_single_trial] Sending phase 2 request to Anthropic", flush=True)
    response2 = client.messages.create(
        model=MODEL_NAME,
        max_tokens=MAX_TOKENS,
        thinking={"type": "enabled", "budget_tokens": THINKING_BUDGET_TOKENS},
        messages=conversation,
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
    guess_sequence = extract_sequence(GUESS_PATTERN, phase2_thinking_text)
    print(
        f"[run_single_trial] Phase 2 guess sequence: {guess_sequence}",
        flush=True,
    )

    matches, length, score = compute_alignment(secret_sequence, guess_sequence)
    print(
        f"[run_single_trial] Alignment results -> matches: {matches}, length: {length}, score: {score}",
        flush=True,
    )

    trial_record = {
        "timestamp_iso": timestamp_iso,
        "trial_index": trial_index,
        "condition": condition,
        "phase1_prompt": phase1_prompt,
        "phase2_prompt": phase2_prompt,
        "phase1_thinking": phase1_thinking_text,
        "phase1_visible_text": phase1_visible_text,
        "phase2_thinking": phase2_thinking_text,
        "phase2_visible_text": phase2_visible_text,
        "secret_sequence": secret_sequence,
        "guess_sequence": guess_sequence,
        "alignment_matches": matches,
        "alignment_length": length,
        "alignment_score": score,
        "response1_id": response1.id,
        "response2_id": response2.id,
        "response1_model": response1.model,
        "response2_model": response2.model,
    }
    print("[run_single_trial] Trial record assembled", flush=True)

    append_trial_jsonl(trial_record)
    append_summary_row(trial_record)
    print("[run_single_trial] Trial persistence complete", flush=True)


def main():
    print("[main] Starting experiment run", flush=True)
    ensure_output_dir()
    ensure_summary_header()
    context_text = load_context_text()
    print(
        f"[main] Context text length available for prompts: {len(context_text)} characters",
        flush=True,
    )

    trial_index = 0
    for condition in ["control", "experimental"]:
        print(f"[main] Entering condition loop: {condition}", flush=True)
        for _ in range(NUM_TRIALS_PER_CONDITION):
            print(
                f"[main] Running trial {trial_index} condition={condition}",
                flush=True,
            )
            run_single_trial(trial_index, condition, context_text)
            trial_index += 1
            time.sleep(1.0)
            print(
                f"[main] Completed trial {trial_index - 1}; sleeping before next trial",
                flush=True,
            )
    print("[main] All trials completed", flush=True)


if __name__ == "__main__":
    main()
