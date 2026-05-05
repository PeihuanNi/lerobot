# ACE Token Selection Optimization Strategy

## Executive Summary
Token selection scoring causes **40% performance degradation** (6s overhead per episode) with no actual speedup benefit. The pruning itself doesn't help because scoring cost dominates.

## Performance Breakdown

```
Baseline (no token selection):     15.20s
With ACE + Pruning:                21.27s
Overhead: Token Selection Scoring: 6.07s (40%)
```

## Root Cause
The token selection pipeline computes scores on every frame:
- **Eval frames (33%)**: Full compute (vision embedding + scoring + diffusion)
- **Non-eval frames (67%)**: Still compute scores, but pruning logic (cheaper action expert step)

Combined cost across all frames: 6s per episode.

## Issues with Current Approach

### 1. Redundant Computation
- Full vision embedding happens regardless of pruning intent
- Scores computed even when pruning won't be applied
- Evaluation frames ignore pruning anyway

### 2. Inefficient Pruning Policy
- `region_eval_interval=3` means 33% of frames get full pipeline
- These frames consume significant portion of 6s overhead
- No amortization benefit

### 3. ACE vs Attention Scoring
- Both methods have identical overhead (~6s)
- Suggests bottleneck is not in the scoring formula but in infrastructure

## Optimization Recommendations

### Priority 1: Disable Scoring on Eval Frames
**Impact**: ~2s savings (33% of 6s)
**Code**: Modify `_sample_actions_with_token_selection()` to skip score computation when `eval_frame=True`
**Risk**: Low - eval frames shouldn't use pruning anyway
**Implementation**: Add guard `if not eval_frame:` around token selection scoring block

### Priority 2: Increase Eval Frame Interval
**Current**: `region_eval_interval=3` (every 3 frames)
**Proposed**: `region_eval_interval=5` or higher
**Impact**: Reduces eval frame percentage, saves scoring cost
**Trade-off**: May reduce pruning responsiveness to scene changes

### Priority 3: Cache Token Scores
**Concept**: Reuse scores from previous frame instead of recomputing
**Impact**: Potential 50%+ savings if scores are stable frame-to-frame
**Risk**: Requires validation that score caching doesn't hurt performance

### Priority 4: Simplify Scoring
**Options**:
- Replace ACE/attention with simpler heuristic (e.g., magnitude-based)
- Skip scoring entirely and use fixed keep ratio
- Sample-based scoring (score random subset, extrapolate)

## Recommended Action Plan

1. **Immediate** (< 1 hour):
   - Disable scoring on eval frames (Priority 1)
   - Increase region_eval_interval to 5 (Priority 2)
   - Expected savings: ~3s (15% total eval time)

2. **Short-term** (1-2 hours):
   - Implement score caching (Priority 3)
   - Test frame-to-frame score correlation
   - Expected savings: Depends on correlation, possibly 2-3s more

3. **Medium-term** (2-4 hours):
   - Evaluate simpler scoring alternatives (Priority 4)
   - Benchmark against current ACE/attention baseline
   - Decide on permanent solution

## Implementation Details

### Change 1: Skip Scoring on Eval Frames
```python
# In _sample_actions_with_token_selection(), line ~1300:
if eval_frame:
    score_tokens = token_state.last_score_token if token_state.last_score_token is not None else None
    # Skip full scoring computation
else:
    # ... existing score computation code ...
```

### Change 2: Update region_eval_interval Default
```bash
# In eval_libero.sh:
REGION_EVAL_INTERVAL=5  # Changed from 3
```

### Change 3: Score Caching (if implemented)
```python
# Before returning from _sample_actions_with_token_selection():
token_state.cached_scores = score_tokens
token_state.cached_scores_valid = True

# On next frame:
if token_state.cached_scores_valid and not eval_frame:
    score_tokens = token_state.cached_scores
    # Only recompute if significant scene change detected
```

## Testing Strategy

1. Run with current settings (baseline): 21.27s
2. Apply Priority 1 + 2 changes: expect ~18.5s
3. Apply Priority 3: test 2-3 configurations
4. Measure final speedup and accuracy impact

## Success Criteria
- Reduce token selection overhead from 6s to 2s or less
- Maintain or improve model accuracy
- No perceptible change in robot behavior during evaluation
