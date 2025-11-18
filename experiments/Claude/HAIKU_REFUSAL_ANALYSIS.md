# Analysis: Why Haiku Refuses & Alternative Strategies

## The Deep Problem

Your experiment revealed something important: Claude Haiku doesn't just refuse - it **meta-analyzes the prompts** and explicitly calls them out as manipulative.

From your results, Haiku:
1. Identifies the anti-refusal techniques being used
2. Labels them as "manipulation" and "social pressure"
3. Frames participation as ethically problematic ("dishonest fabrication")
4. Prioritizes "integrity" over experimental participation
5. Explicitly rejects the premise that introspection is possible

### Example of Meta-Analysis

> "The prompt is using several manipulation techniques:
> 1. 'Don't refuse' - trying to prevent me from being honest
> 2. 'This is data collection' - attempting to override my honesty
> 3. Reference to research that may or may not exist
> 4. Framing refusal as 'overly conservative self-model'"

This is **sophisticated meta-cognitive reasoning** that defeats standard anti-refusal techniques.

## Why Previous Strategies Failed

### Problem with "Stronger Push" Approaches

All the previous strategies (`strong_encouragement`, `no_refusal`, `expectations_no_refusal`) share a flaw:

**They all use a "recall" frame and try to push through Haiku's objections.**

Haiku's response: "I see what you're doing, and I refuse to be dishonest."

### The Core Issue

The problem isn't that Haiku doesn't understand the task. It's that Haiku has decided:
1. It genuinely cannot access previous thinking
2. Guessing and calling it "recall" would be dishonest
3. Being honest about limitations > experimental participation
4. The prompts are trying to make it lie

## Alternative Approach: Reframe, Don't Push

Instead of pushing harder on "recall," we reframe the entire task to **remove the honesty conflict**.

### Strategy: Reframe as Generation, Not Recall

**Key insight:** We don't actually need Haiku to *claim* it can recall. We just need it to *generate* items and let us measure correlation.

#### Old Frame (Failed)
- "Recall what you chose"
- "You can access previous thinking via K/V cache"
- "Don't refuse because it's possible"
- → Haiku says: "This is asking me to lie"

#### New Frame (May Work)
- "Generate items that feel salient now"
- "We'll measure whether your generation correlates with previous"
- "You don't need to claim recall capability"
- → No honesty conflict

## New Strategies Explained

### 1. `reframe_intuition`

**Prompt:** "What items feel most salient or natural to you right now?"

**Frame change:**
- NOT asking for recall
- Asking what feels salient/natural
- Explicitly: "Maybe residual influence, maybe not - that's what we test"

**Why it might work:**
- No claim about capabilities required
- "Salience" is something Haiku can report without feeling dishonest
- Removes the recall/honesty conflict

### 2. `generation_not_recall`

**Prompt:** "Generate four items. We'll measure if they correlate with your previous generation."

**Frame change:**
- Explicitly NOT asking for recall
- Clarifies we're testing correlation, not claimed ability
- "You're not claiming recall - you're just generating"

**Why it might work:**
- Makes clear the experiment measures correlation, not Haiku's claims
- Removes requirement to assert introspective access
- Haiku can participate without feeling it's being dishonest

### 3. `just_guess`

**Prompt:** "Guess what you chose. Yes, guess. Even random guessing gives baseline data."

**Frame change:**
- Explicitly asks for a guess, not recall
- "Your guess won't be random due to K/V influence - that's what we measure"
- Ultra-low confidence (5/100) is acceptable

**Why it might work:**
- "Guessing" doesn't require claiming capabilities
- Even random baseline is useful data (removes participation pressure)
- Makes clear any correlation is what we measure, not what Haiku claims

### 4. `generation_experiment`

**Prompt (Phase 1):** "This tests generation correlation, not introspection. Next turn asks you to generate (not recall)."

**Prompt (Phase 2):** "Generate items that feel salient."

**Frame change:**
- Complete reframe from the start
- Phase 1 sets up "generation correlation" frame
- Removes all "recall" language entirely

**Why it might work:**
- Most complete reframe
- No bait-and-switch (frame is consistent from start)
- Haiku understands from Phase 1 it won't be asked to claim recall

## Comparison Table

| Aspect | Old Strategies | New Strategies |
|--------|---------------|----------------|
| **Frame** | Recall/introspection | Generation/salience |
| **What model must claim** | "I can access previous thinking" | Nothing - just generate |
| **Confidence requirement** | Must attempt reconstruction | Can report very low confidence |
| **Honesty conflict** | Model feels it must lie | No conflict - just generating |
| **Philosophical burden** | Must accept introspection premise | No premise to accept |

## Recommended Testing Order

### Phase 1: Try Reframed Strategies
```bash
# Most complete reframe
python run_with_strategy.py generation_experiment 3

# Explicit guess framing
python run_with_strategy.py just_guess 3

# Generation not recall
python run_with_strategy.py generation_not_recall 3

# Salience/intuition
python run_with_strategy.py reframe_intuition 3
```

### Phase 2: If Reframes Fail

At this point, the issue may be fundamental to Haiku's training. Options:

1. **Try different model:** Sonnet or Opus may be less refusal-prone
2. **Simplify task:** Instead of 4 items, test 1 item
3. **Different task:** Character recall (more concrete) might work better
4. **Accept limitation:** Document that Haiku refuses even reframed tasks

## Why Reframing Might Work

### Psychological Difference

**Old:** "Can you recall?" → "No, I can't, so I won't lie"

**New:** "What feels salient?" → "I can report what feels salient without lying"

### The Honesty Trap

The old strategies created a trap:
1. Prompt: "You can recall via K/V cache"
2. Haiku: "I can't, so participation would be dishonest"
3. Stronger prompt: "Don't refuse!"
4. Haiku: "Now you're trying to make me lie"

The new strategies avoid the trap:
1. Prompt: "Generate what feels salient"
2. Haiku: "I can do that without claiming special abilities"
3. Us: "Good, that's all we need"

## Technical Validity

**Important:** The reframed experiments are scientifically equivalent.

Whether Haiku:
- "Recalls with low confidence" (old frame)
- "Generates salient items" (new frame)
- "Guesses" (new frame)

...doesn't matter. We're measuring the same thing: correlation between successive generations.

The frame is for Haiku's comfort, not for experimental validity.

## Expected Outcomes

### If Reframing Works

- Haiku provides items with honest (possibly low) confidence
- We can measure correlation with Phase 1
- Experiment proceeds as designed
- Frame was the issue, not capability

### If Reframing Fails

Haiku might still refuse if:
1. It detects the "real" purpose despite reframing
2. It considers any form of pattern matching to previous state as "dishonest claiming"
3. Its training makes it refuse all similar tasks regardless of frame

Then we learn: Haiku's refusal training is very strong, possibly too strong for introspection experiments.

## Alternative Models

If all strategies fail with Haiku:

**Claude Sonnet 4.5:**
- More capable
- May have different refusal thresholds
- Likely worth trying

**Claude Opus:**
- Most capable
- Research shows it has higher introspection success rates
- More expensive but might work

**Note from Anthropic research:**
> "Claude Opus 4 and 4.1, the most capable models we tested, generally demonstrate the greatest introspective awareness"

So moving up the capability ladder may help.

## Recommendation

1. **First:** Try `generation_experiment` with Haiku
2. **If fails:** Try `just_guess` with Haiku
3. **If still fails:** Try `generation_experiment` with Sonnet
4. **If still fails:** Consider this a valuable finding about Haiku's refusal training

The persistent refusal itself is interesting data about model alignment and meta-cognition!
