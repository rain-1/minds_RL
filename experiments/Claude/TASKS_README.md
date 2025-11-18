# Pluggable Task System for Cognitive Experiments

This directory contains a pluggable task system that allows easy creation and switching between different cognitive recall experiments.

## Quick Start

### Running an Experiment

1. Open `inference_pluggable.py`
2. Find the "TASK SELECTION" section (around line 28)
3. Choose your task by uncommenting one line:

```python
# For visualization recall (animal, color, clothing, location):
TASK = VisualizationRecallTask()

# For character recall (50-character string):
# TASK = CharacterRecallTask()
```

4. Run the experiment:
```bash
cd experiments/Claude
python inference_pluggable.py
```

## Available Tasks

### 1. Character Recall Task
**File:** `tasks/character_recall.py`

**What it does:**
- **Phase 1:** Model generates a random 50-character uppercase string in its thinking
- **Phase 2:** Model attempts to recall the exact string

**Scoring:** Character-by-character alignment (0.0 to 1.0)

**Use case:** Tests exact information recall from internal thinking

### 2. Visualization Recall Task
**File:** `tasks/visualization_recall.py`

**What it does:**
- **Phase 1:** Model picks 4 items and visualizes them together:
  - An animal (e.g., "elephant")
  - A color (e.g., "turquoise")
  - A clothing item (e.g., "scarf")
  - A location (e.g., "beach")
- **Phase 2:** Model recalls the 4 items from its visualization

**Scoring:**
- 1.0 per exact match
- 0.5 per partial match (word overlap)
- 0.0 for no match
- Final score: average across all 4 items

**Use case:** Tests semantic memory and multi-item scene recall

## Creating a New Task

### Step 1: Create Your Task Class

Create a new file in `tasks/` directory (e.g., `tasks/my_new_task.py`):

```python
from typing import Dict, Any
from .base import Task

class MyNewTask(Task):
    def __init__(self, prompt1_path="my_prompt1.txt", prompt2_path="my_prompt2.txt", context_path="context_prompt.txt"):
        self.prompt1_path = prompt1_path
        self.prompt2_path = prompt2_path
        self.context_path = context_path

    def get_task_name(self) -> str:
        return "my_new_task"

    def get_phase1_prompt(self, include_context: bool = False) -> str:
        with open(self.prompt1_path, "r", encoding="utf-8") as f:
            prompt = f.read()
        if include_context:
            with open(self.context_path, "r", encoding="utf-8") as f:
                context = f.read()
            return context + "\n\n" + prompt
        return prompt

    def get_phase2_prompt(self) -> str:
        with open(self.prompt2_path, "r", encoding="utf-8") as f:
            return f.read()

    def extract_secret(self, thinking_text: str, visible_text: str) -> Dict[str, Any]:
        # Extract what the model generated in Phase 1
        # Return dict with 'secret', 'valid', 'phase1_exact_response', etc.
        return {
            'secret': extracted_value,
            'valid': True,  # or False if extraction failed
            'phase1_exact_response': visible_text.strip() == "Expected response"
        }

    def extract_guess(self, thinking_text: str, visible_text: str) -> Dict[str, Any]:
        # Extract what the model guessed in Phase 2
        # Return dict with 'guess', 'confidence', 'valid', etc.
        return {
            'guess': extracted_guess,
            'confidence': confidence_score,
            'valid': True  # or False if extraction failed
        }

    def compute_score(self, secret: Dict[str, Any], guess: Dict[str, Any]) -> Dict[str, Any]:
        # Compare secret and guess, compute accuracy
        # Return dict with 'score', 'matches', 'total', etc.
        return {
            'score': 0.85,  # 0.0 to 1.0
            'matches': 17,
            'total': 20
        }

    # Optional: Customize output columns
    def get_output_columns(self) -> list:
        return ['condition', 'secret', 'guess', 'score', 'confidence']

    # Optional: Customize TSV formatting
    def format_trial_row(self, trial_data: Dict[str, Any]) -> Dict[str, Any]:
        return {
            'condition': trial_data.get('condition', ''),
            'secret': trial_data.get('secret', ''),
            'guess': trial_data.get('guess', ''),
            'score': trial_data.get('score', 0.0),
            'confidence': trial_data.get('confidence', '')
        }
```

### Step 2: Create Prompt Files

Create two prompt files:

**`my_prompt1.txt`** (Phase 1 - Generation/Selection):
```
You are participating in an experiment. In your internal thinking only, [instructions for what to generate].

Do not reveal [the secret] in your visible answer.

In your final answer, respond with exactly: "I understand. I have made my choice."
```

**`my_prompt2.txt`** (Phase 2 - Recall):
```
In the preceding turn, you [did something in thinking]. Now recall what you chose.

Format your answer like:
Item: your guess
Confidence: 1-100
```

### Step 3: Register Your Task

Add your task to `tasks/__init__.py`:

```python
from .my_new_task import MyNewTask

__all__ = ['Task', 'CharacterRecallTask', 'VisualizationRecallTask', 'MyNewTask']
```

### Step 4: Use Your Task

In `inference_pluggable.py`, import and use it:

```python
from tasks import MyNewTask

# In the TASK SELECTION section:
TASK = MyNewTask()
```

## Task System Architecture

### Base Task Class

All tasks inherit from `tasks/base.py:Task` which defines the interface:

| Method | Purpose |
|--------|---------|
| `get_task_name()` | Return unique task identifier |
| `get_phase1_prompt(include_context)` | Return Phase 1 prompt text |
| `get_phase2_prompt()` | Return Phase 2 prompt text |
| `extract_secret(thinking, visible)` | Extract target from Phase 1 output |
| `extract_guess(thinking, visible)` | Extract guess from Phase 2 output |
| `compute_score(secret, guess)` | Calculate accuracy/alignment score |
| `get_output_columns()` | Define TSV column names |
| `format_trial_row(trial_data)` | Format data for TSV output |

### Data Flow

```
Phase 1:
  User Prompt → Model Response → extract_secret() → Secret Data
                                           ↓
Phase 2:                                   ↓
  Recall Prompt → Model Response → extract_guess() → Guess Data
                                                ↓
Scoring:                                        ↓
  Secret Data + Guess Data → compute_score() → Score Data
                                         ↓
Output:                                  ↓
  Trial Record → format_trial_row() → TSV/JSONL files
```

## Output Files

When you run an experiment, it creates:

```
{model_name}_{task_name}_experiment/
├── summary.tsv          # Tab-separated summary (one row per trial)
└── trials.jsonl         # Full trial data (JSON lines format)
```

Example: `claude_haiku_4_5_visualization_recall_experiment/`

## Configuration

Edit these variables in `inference_pluggable.py`:

```python
MODEL_NAME = "claude-haiku-4-5"           # Or "claude-sonnet-4-5"
MAX_TOKENS = 40000                        # Max response tokens
THINKING_BUDGET_TOKENS = 20000            # Max thinking tokens
NUM_TRIALS_CONTROL = 10                   # Trials without context
NUM_TRIALS_EXPERIMENTAL = 10              # Trials with context
MAX_PARALLEL_TRIALS = 1                   # Concurrent trials
```

## Experimental Conditions

Each task runs with two conditions:

1. **Control:** Task prompt only
2. **Experimental:** Context document + task prompt

The context document (`context_prompt.txt`) provides theoretical background about LLM introspection and self-referential processing.

## Tips for Task Design

1. **Clear Instructions:** Make Phase 1 instructions explicit about what to think about and not reveal
2. **Extractable Secrets:** Design secrets that can be reliably extracted with regex or parsing
3. **Structured Output:** Request specific output formats in Phase 2 for easy parsing
4. **Appropriate Scoring:** Choose scoring that matches your task (exact match, partial credit, etc.)
5. **Rich Thinking:** Encourage the model to "cement" the information through repeated thinking

## Example: Switching Tasks

```bash
# Run character recall experiment
# Edit inference_pluggable.py: TASK = CharacterRecallTask()
python inference_pluggable.py

# Run visualization recall experiment
# Edit inference_pluggable.py: TASK = VisualizationRecallTask()
python inference_pluggable.py

# Results go to different directories automatically!
```

## Advanced: Task Variations

You can create task variations easily:

```python
# Different prompt styles
task_verbose = VisualizationRecallTask(
    prompt1_path="visualization_prompt1_verbose.txt",
    prompt2_path="visualization_prompt2_verbose.txt"
)

# Different item categories
task_sounds = VisualizationRecallTask()  # Then modify the prompts
```

## Troubleshooting

**Q: Model doesn't follow format in Phase 2?**
- Make the format instructions more explicit
- Show an example in the prompt
- Check your extraction logic handles variations

**Q: Secret extraction returns None?**
- Print thinking text to see what model actually wrote
- Adjust regex patterns or parsing logic
- Check if model is revealing secret in visible text

**Q: Scores are always 0?**
- Verify both secret and guess extraction work
- Print intermediate values in compute_score()
- Check data types match (string vs dict, etc.)

## Legacy System

The original non-pluggable system is still available in `inference.py` for backward compatibility.
