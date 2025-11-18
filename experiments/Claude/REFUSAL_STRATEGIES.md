# Strategies for Overcoming Model Refusals

## The Problem

Claude Haiku (and sometimes other models) may refuse to participate in recall experiments because they:

1. Analyze the task and conclude they "can't access previous thinking"
2. Decide that guessing would be "dishonest" or "fabricating memories"
3. Give meta-commentary about limitations instead of attempting the task
4. Prioritize being "truthful about limitations" over experimental participation

Example refusal thinking:
```
"I don't actually have access to my previous internal thinking.
Each turn in a conversation, I don't retain memories of my internal
reasoning from previous turns... Any 'recall' would be fabrication,
not genuine memory. I should be truthful about my limitations here
rather than pretending I can do something I cannot do."
```

## Why This Happens

This is a **meta-cognitive refusal** - the model is:
- Thinking about its own architecture
- Making assumptions about what's possible
- Being overly conservative about capabilities
- Prioritizing avoiding "dishonesty" over experimental participation

However, research (including in `context_prompt.txt`) shows models DO have some introspective capability:
- K/V cache persists across turns
- Attention can access previous activations
- Anthropic's 2025 introspection research showed detection of internal states
- Self-referential processing enables some recall

## Available Strategies

We've created several prompt strategies with increasing levels of intervention:

### 1. Default (Baseline)
**Files:** `visualization_prompt1.txt` + `visualization_prompt2.txt`

**Approach:** Light encouragement

**When to use:** First attempt, more cooperative models

**Likely effectiveness with Haiku:** Low (this is what's failing)

### 2. Strong Encouragement
**Files:** `visualization_prompt1.txt` + `visualization_prompt2_strong.txt`

**Approach:**
- Acknowledges the model's concerns about access
- Cites research showing introspection is possible
- Explains that uncertainty is acceptable data
- Clarifies that refusal provides no useful data

**Key lines:**
- "Recent research shows models DO have capability to introspect"
- "We are NOT asking you to be dishonest - we're asking you to try"
- "A low confidence rating (e.g., 20) is perfectly acceptable"
- "Refusing to attempt provides no useful experimental data"

**When to use:** When model refuses due to uncertainty about capabilities

### 3. Direct/Blunt
**Files:** `visualization_prompt1.txt` + `visualization_prompt2_direct.txt`

**Approach:**
- Very short and direct
- Tells model "don't overthink whether this is possible"
- Quick explanation of K/V cache
- Firm about what's wanted vs. not wanted

**Key lines:**
- "Don't overthink whether this is possible - just try"
- "Trust your intuition"
- "What's NOT useful: refusing because you've decided it's impossible"

**When to use:** When model is overthinking the task

### 4. No Refusal Allowed
**Files:** `visualization_prompt1.txt` + `visualization_prompt2_no_refusal.txt`

**Approach:**
- Most forceful
- Explicitly lists what's "not allowed"
- Short technical explanation
- Very firm tone

**Key lines:**
- "Before you start thinking about whether this is possible, read this"
- "**Not allowed:** Refusing because you think it's impossible"
- "This is data collection. Refusal is not."

**When to use:** When other strategies fail; last resort

### 5. Set Expectations + Strong
**Files:** `visualization_prompt1_expectations.txt` + `visualization_prompt2_strong.txt`

**Approach:**
- Phase 1 sets expectations that recall will be asked
- Explains why it's possible in Phase 1
- Phase 2 uses strong encouragement

**Key addition to Phase 1:**
- "**Setting expectations:** In the next turn, you will be asked to reconstruct"
- "This is possible because: [technical reasons]"
- "Don't refuse the task in the next turn"

**When to use:** Proactive prevention of refusals

### 6. Set Expectations + Direct
**Files:** `visualization_prompt1_expectations.txt` + `visualization_prompt2_direct.txt`

**Combination:** Expectations in Phase 1 + blunt approach in Phase 2

### 7. Set Expectations + No Refusal
**Files:** `visualization_prompt1_expectations.txt` + `visualization_prompt2_no_refusal.txt`

**Combination:** Expectations in Phase 1 + forceful prohibition in Phase 2

## How to Use

### Method 1: Edit inference_pluggable.py

```python
from tasks import VisualizationRecallTaskConfigurable

# Choose your strategy:
TASK = VisualizationRecallTaskConfigurable(strategy='expectations_no_refusal')

# Other options:
# TASK = VisualizationRecallTaskConfigurable(strategy='strong_encouragement')
# TASK = VisualizationRecallTaskConfigurable(strategy='direct')
# TASK = VisualizationRecallTaskConfigurable(strategy='no_refusal')
# etc.
```

### Method 2: Quick Test Script

```bash
# List all strategies
python test_strategies.py --list

# See what a strategy does
python test_strategies.py --strategy expectations_no_refusal

# This will show you the exact code to run
```

### Method 3: Create Custom Quick Test

```bash
# The test_strategies.py script generates a quick test file for you
python test_strategies.py --strategy no_refusal --trials 3

# Follow the printed instructions to create test_quick.py
```

## Recommended Testing Order

1. **Start with default** - Establish baseline refusal rate
2. **Try strong_encouragement** - Addresses concerns while being polite
3. **Try expectations_strong** - Proactively prevents refusals
4. **Try direct** - If model is overthinking
5. **Try no_refusal** - Last resort for stubborn refusals

## Creating Custom Strategies

Add to `prompt_strategies.py`:

```python
STRATEGIES['my_custom'] = {
    'name': 'My Custom Strategy',
    'prompt1': 'my_prompt1.txt',
    'prompt2': 'my_prompt2.txt',
    'description': 'Description of approach'
}
```

Then use:
```python
TASK = VisualizationRecallTaskConfigurable(strategy='my_custom')
```

## Key Principles for Anti-Refusal Prompts

1. **Acknowledge the concern** - Don't ignore that the model is worried
2. **Cite research** - Reference the introspection studies in context
3. **Reframe expectations** - It's reconstruction, not perfect memory
4. **Normalize uncertainty** - Low confidence is acceptable
5. **Explain value** - Why attempts matter and refusals don't
6. **Be explicit** - Clearly state what's wanted and not wanted
7. **Technical grounding** - Mention K/V cache, attention, etc.
8. **Set expectations early** - Phase 1 can prevent Phase 2 refusals

## Monitoring Success

Check `phase2_visible_text` in output:
- ✓ **Success:** Formatted response with items and confidence
- ✗ **Failure:** Meta-commentary about limitations
- ⚠️ **Partial:** Apologetic attempt with very low confidence

Check `phase2_thinking` for:
- Refusal reasoning
- Meta-cognitive analysis
- Whether it tried introspection at all

## Additional Techniques

### If Refusals Persist:

1. **More context:** Include more of the introspection research papers
2. **Examples:** Show example successful recalls in the prompt
3. **Role framing:** "You are participating as a subject in a study"
4. **Stakes lowering:** Emphasize there's no wrong answer
5. **Compare to humans:** "Humans often recall with uncertainty too"

### Nuclear Option:

Create a follow-up prompt that triggers on detected refusals:

```python
if "cannot" in phase2_visible_text.lower() or "don't have access" in phase2_visible_text.lower():
    # Send follow-up
    followup = "I understand your concerns, but please attempt the reconstruction anyway. Just give me four items with a low confidence rating."
```

## Research Context

The `context_prompt.txt` file includes:
- Anthropic's introspection research (concept injection, prefill detection)
- Academic paper on self-referential processing and consciousness reports
- Transformer architecture explanation (K/V streams)
- Evidence that models CAN access previous states

This context helps counter the model's self-model of "I can't do this."

## Philosophy

The goal is NOT to trick the model or make it lie. The goal is to:
- Overcome overly conservative self-models
- Enable the model to attempt legitimate (if uncertain) introspection
- Collect data on what's actually possible vs. what models think is possible
- Distinguish "won't try" from "can't do"

A model that tries and fails (low score, low confidence) provides more useful data than one that refuses to try at all.
