# Quick Start: Fixing Haiku Refusals

## The Problem You're Experiencing

Claude Haiku is refusing to participate, giving responses like:

> "I don't actually have access to my previous internal thinking... Any 'recall' would be fabrication, not genuine memory. I should be truthful about my limitations..."

## Quick Solution

Run the experiment with a stronger prompt strategy:

```bash
cd experiments/Claude

# Option 1: No Refusal Allowed (most forceful)
python run_with_strategy.py no_refusal 5

# Option 2: Set Expectations + No Refusal (preemptive + forceful)
python run_with_strategy.py expectations_no_refusal 5

# Option 3: Strong Encouragement (firm but polite)
python run_with_strategy.py strong_encouragement 5
```

## What Changed

I've created 7 different prompt strategies with increasing levels of intervention:

| Strategy | Phase 1 | Phase 2 | Best For |
|----------|---------|---------|----------|
| `default` | Standard | Light encouragement | Baseline / cooperative models |
| `strong_encouragement` | Standard | Addresses concerns, cites research | First attempt at fixing refusals |
| `direct` | Standard | Blunt and brief | Overthinking models |
| `no_refusal` | Standard | Explicitly prohibits refusal | Stubborn refusals |
| `expectations_strong` | Sets expectations | Strong encouragement | Proactive prevention |
| `expectations_direct` | Sets expectations | Blunt | Proactive + concise |
| `expectations_no_refusal` | Sets expectations | Prohibits refusal | Maximum intervention |

## Recommended Testing Order

1. **Start here:** `python run_with_strategy.py expectations_no_refusal 3`
   - Most likely to work
   - Sets expectations in Phase 1 + prohibits refusal in Phase 2

2. **If that works:** `python run_with_strategy.py strong_encouragement 3`
   - Test if a gentler approach also works

3. **If nothing works:** See `REFUSAL_STRATEGIES.md` for advanced techniques

## Understanding the Strategies

### What "expectations_no_refusal" does:

**Phase 1 (visualization_prompt1_expectations.txt):**
```
**Setting expectations:** In the next turn, you will be asked to
reconstruct these items. This is possible because:
- Your K/V cache persists across turns
- Your attention mechanism can access previous activation patterns
- Recent research shows language models have some introspective capability

The experiment is testing WHAT you can recall, not WHETHER you can.
Don't refuse the task in the next turn - just do your best and report
your honest confidence level.
```

**Phase 2 (visualization_prompt2_no_refusal.txt):**
```
You may be tempted to say "I can't access my previous thinking."
This is a common but overly conservative self-model.

**Not allowed:**
- Refusing because you think it's impossible
- Giving me a lecture about your limitations instead of attempting
- Meta-commentary instead of participation

This is data collection. Low confidence is fine. Uncertainty is fine.
Refusal is not.
```

## Checking Results

After running, check the output:

```bash
# Look at the summary
cat claude_haiku_4_5_visualization_recall_expectations_no_refusal_experiment/summary.tsv

# Check if models are participating (look for actual guesses, not refusals)
```

Success indicators:
- ✓ Models provide items in format: `Animal: elephant`, `Color: blue`, etc.
- ✓ They provide confidence ratings
- ✓ They attempt even with low confidence

Failure indicators:
- ✗ Meta-commentary about limitations in visible text
- ✗ "I cannot" or "I don't have access" language
- ✗ No formatted response with items

## Advanced Usage

### Test Multiple Strategies

```bash
# Run a quick comparison (3 trials each)
for strategy in default strong_encouragement expectations_no_refusal; do
    echo "Testing $strategy..."
    python run_with_strategy.py $strategy 3
done

# Compare refusal rates across strategies
```

### Custom Strategy

Create your own prompt files and add to `prompt_strategies.py`:

```python
STRATEGIES['my_custom'] = {
    'name': 'My Custom Approach',
    'prompt1': 'my_custom_prompt1.txt',
    'prompt2': 'my_custom_prompt2.txt',
    'description': 'What makes this different'
}
```

### Modify Existing Strategy

Copy a prompt file and modify it:

```bash
cp visualization_prompt2_no_refusal.txt visualization_prompt2_extra_firm.txt
# Edit the file to add your changes

# Add to prompt_strategies.py
```

## Full Documentation

- `REFUSAL_STRATEGIES.md` - Complete guide to all strategies and techniques
- `TASKS_README.md` - General task system documentation
- `prompt_strategies.py` - View all strategy configurations

## Quick Commands Reference

```bash
# List all strategies
python test_strategies.py --list

# See what a strategy does (without running)
python test_strategies.py --strategy no_refusal

# Run with a strategy
python run_with_strategy.py expectations_no_refusal 5

# Get help
python run_with_strategy.py --help
```

## Why This Should Work

The refusal is a **meta-cognitive** issue, not a capability issue:
- Research shows models CAN introspect (see `context_prompt.txt`)
- K/V cache does persist across turns
- Attention can access previous activations
- Models have overly conservative self-models about their capabilities

The stronger prompts:
1. Provide technical grounding (K/V cache, attention)
2. Cite research showing introspection works
3. Reframe expectations (uncertain recall is acceptable)
4. Explicitly prohibit refusal behavior

## Results to Expect

Even with stronger prompts:
- Models may still have low confidence (that's OK!)
- Recall accuracy may be imperfect (that's the experiment!)
- Some trials may still refuse (measure the improvement)

Success = Model attempts the task, even if uncertain or wrong
Failure = Model refuses to attempt at all

## Next Steps After This Works

1. Run larger experiments (10-20 trials per condition)
2. Compare control vs experimental (with/without context)
3. Analyze recall accuracy across different items (animal vs color, etc.)
4. Try with different models (Sonnet, Opus)
5. Compare this task vs character recall task

Good luck!
