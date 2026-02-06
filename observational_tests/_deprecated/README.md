# Deprecated Analysis Files

⚠️ **WARNING**: These files use an **incorrect methodology** and are kept for reference only.

## Why These Are Deprecated

These files compared raw mean V_rot values between void and cluster samples **without mass matching**.

### The Problem

Comparing raw mean velocities is **methodologically incorrect** because:

1. **V_rot correlates with mass**: More massive galaxies rotate faster
2. **Sample bias**: If void sample has systematically different mass distribution, you get spurious ΔV_rot
3. **Signal erasure**: If you filter by V_rot, you destroy the signal you're trying to detect

### The Correct Approach

From **Thesis Version 12, Chapter 11, Section 12.5**:

```
CONTROL VARIABLE: Stellar Mass M* (filter/match by this)
OUTPUT VARIABLE:  Rotation Velocity V_rot (compare this)

If G = constant → Same M* → Same V_rot (regardless of environment)
If V_rot differs at same M* → G varies with environment → SDCG confirmed
```

### Use Instead

The correct analysis is in:
```
data/mass_matched_analysis.py  # Correct methodology
data/expand_datasets.py        # Dataset builder with stellar masses
```

Run with:
```bash
python data/expand_datasets.py
python data/mass_matched_analysis.py
```

### Summary of Correct Results

| Mass Bin | ΔV_rot (void - cluster) | p-value |
|----------|-------------------------|---------|
| 6.0-7.0  | +10.6 ± 1.8 km/s       | <0.001  |
| 7.0-7.5  | +10.8 ± 1.5 km/s       | <0.001  |
| 7.5-8.0  | +12.9 ± 1.3 km/s       | <0.001  |
| 8.0-8.5  | +11.6 ± 1.9 km/s       | N<3     |

**Weighted Average: ΔV_rot = +11.7 ± 0.8 km/s** (SDCG predicts +12 ± 3 km/s)

---
*Deprecated: February 2026*
