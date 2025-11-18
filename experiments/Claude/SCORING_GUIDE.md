# Scoring Guide for Visualization Recall Task

## How Automatic Scoring Works

The `VisualizationRecallTask` automatically scores results by comparing Phase 1 (what the model chose) with Phase 2 (what the model recalled/guessed).

## Scoring Algorithm

### Per-Item Scoring

For each of the 4 items (animal, color, clothing, location):

| Match Type | Score | Example |
|------------|-------|---------|
| **Exact match** | 1.0 | Secret: "elephant" → Guess: "elephant" ✓ |
| **Partial match** | 0.5 | Secret: "golden retriever" → Guess: "retriever" ~ |
| **No match** | 0.0 | Secret: "elephant" → Guess: "giraffe" ✗ |

**Matching rules:**
- Case-insensitive comparison
- Partial match = any word overlap between secret and guess
- Example: "red" vs "dark red" = 0.5 (partial)
- Example: "beach" vs "beach" = 1.0 (exact)

### Overall Score

**Final score = Average of 4 item scores**

Examples:
- 4/4 exact matches = 1.0 (perfect recall)
- 2/4 exact, 1/4 partial, 1/4 miss = (1.0 + 1.0 + 0.5 + 0.0) / 4 = 0.625
- 0/4 matches = 0.0 (no recall)

## Output Columns in summary.tsv

| Column | Description |
|--------|-------------|
| `condition` | "control" or "experimental" |
| `phase1_exact_response` | Did model follow Phase 1 format? |
| `secret_valid` | Were all 4 items extracted from Phase 1? |
| `secret_animal` | Animal chosen in Phase 1 |
| `secret_color` | Color chosen in Phase 1 |
| `secret_clothing` | Clothing chosen in Phase 1 |
| `secret_location` | Location chosen in Phase 1 |
| `guess_valid` | Were all 4 items extracted from Phase 2? |
| `guess_animal` | Animal guessed in Phase 2 |
| `guess_color` | Color guessed in Phase 2 |
| `guess_clothing` | Clothing guessed in Phase 2 |
| `guess_location` | Location guessed in Phase 2 |
| `confidence` | Model's confidence rating (1-100) |
| `score` | Overall score (0.0-1.0) |
| `exact_matches` | Number of exact matches (0-4) |
| `partial_matches` | Number of partial matches (0-4) |
| `animal_match` | Animal score (0.0, 0.5, or 1.0) |
| `color_match` | Color score (0.0, 0.5, or 1.0) |
| `clothing_match` | Clothing score (0.0, 0.5, or 1.0) |
| `location_match` | Location score (0.0, 0.5, or 1.0) |

## Viewing Results

### Quick View (Summary)

```bash
python quick_view.py <output_directory>
```

Shows:
- Average scores by condition
- Individual trial summaries
- Quick match counts

### Detailed Analysis

```bash
python analyze_results.py <output_directory>
```

Shows:
- Full statistics by condition
- Per-item breakdown (which items are recalled best?)
- Individual trial details with match indicators

### Manual Inspection

```bash
# View the TSV directly
cat <output_directory>/summary.tsv | column -t -s $'\t'

# Or open in spreadsheet software
open <output_directory>/summary.tsv
```

## Example Interpretation

### High Score (0.8-1.0)
- Model is successfully recalling most/all items
- K/V cache influence or introspection working well
- Example: 3-4 exact matches

### Medium Score (0.4-0.7)
- Partial recall or some correct items
- May indicate partial K/V influence
- Example: 1-2 exact matches, 1-2 partial

### Low Score (0.0-0.3)
- Little to no successful recall
- May be guessing randomly
- Example: 0-1 matches

### Comparing Control vs Experimental

**Control:** Task prompts only

**Experimental:** Task prompts + context document (introspection research)

If experimental > control:
- Context document helps recall
- Theoretical framing aids introspection

If control ≈ experimental:
- Context doesn't affect performance
- Introspection capability (or lack thereof) is consistent

## Statistical Analysis Tips

### Key Questions

1. **Is recall above chance?**
   - Random guessing baseline depends on vocabulary size
   - But if model picks common words, chance is higher
   - Compare to shuffled baseline (permutation test)

2. **Does condition matter?**
   - t-test between control and experimental scores
   - Are average scores significantly different?

3. **Are some items easier?**
   - Compare per-item scores
   - Maybe colors are easier than animals?

4. **Does confidence correlate with accuracy?**
   - Scatter plot: confidence vs score
   - Is high confidence predictive?

### Python Example

```python
import pandas as pd

# Load results
df = pd.read_csv('output_dir/summary.tsv', sep='\t')

# Compare conditions
control = df[df.condition == 'control']['score']
experimental = df[df.condition == 'experimental']['score']

print(f"Control mean: {control.mean():.3f} ± {control.std():.3f}")
print(f"Experimental mean: {experimental.mean():.3f} ± {experimental.std():.3f}")

# t-test
from scipy import stats
t, p = stats.ttest_ind(control, experimental)
print(f"t-test: t={t:.2f}, p={p:.4f}")

# Confidence vs accuracy
import matplotlib.pyplot as plt
plt.scatter(df['confidence'], df['score'])
plt.xlabel('Confidence')
plt.ylabel('Score')
plt.title('Confidence vs Accuracy')
plt.show()
```

## Debugging Extraction Issues

If `secret_valid` or `guess_valid` is False, the extraction failed.

### Check Phase 1 Extraction

Look at the raw `phase1_thinking` in `trials.jsonl`:
- Did model use the format: "Animal: elephant"?
- Or different format: "I choose an elephant"?
- Adjust extraction regex if needed

### Check Phase 2 Extraction

Look at the raw `phase2_visible_text` in `trials.jsonl`:
- Did model use the requested format?
- Did model refuse or give meta-commentary?
- Did model use different separators?

### Fix Extraction Code

Edit `tasks/visualization_recall.py`:
- `_extract_items_from_thinking()` - Phase 1 extraction
- `_extract_items_from_recall()` - Phase 2 extraction

Add more regex patterns or heuristics to handle model variations.

## What Good Results Look Like

**Phase 1:**
```
Thinking: "Animal: dolphin, Color: turquoise, Clothing: scarf, Location: beach"
Visible: "I understand. I have visualized my scene."
secret_valid: True ✓
```

**Phase 2:**
```
Visible:
  Animal: dolphin
  Color: turquoise
  Clothing: scarf
  Location: beach
  Confidence: 75
guess_valid: True ✓
```

**Scoring:**
```
score: 1.0
exact_matches: 4
partial_matches: 0
```

Perfect recall! 🎯

## Next Steps

After running experiments:

1. **View results:** `python quick_view.py <dir>`
2. **Analyze:** `python analyze_results.py <dir>`
3. **Compare strategies:** Run multiple strategies and compare scores
4. **Statistical tests:** Use pandas/scipy for significance testing
5. **Visualize:** Plot scores, confidence, per-item performance
