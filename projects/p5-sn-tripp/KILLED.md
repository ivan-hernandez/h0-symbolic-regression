# P5 — SN Ia Tripp Standardization via SR: KILLED

**Date:** 2026-06-26
**Reason:** Crap-or-worthwhile test → practice

## Results
- **Linear Tripp (α·x1 − β·c) confirmed optimal:** SR finds at most 0.002 mag RMS improvement (negligible)
- **Best SR model:** `y = 0.000339/(x1 + 1.15)` — tiny correction, no physical motivation
- **Data:** 1624 Pantheon+ cosmologically usable SNe, cosmology-removed residual (z-trend subtracted)
- **PySR:** 400 iterations × 15 populations × 2 search spaces (x1,c and x1,c,z)

## Verdict
No novel finding. The linear Tripp relation is genuinely optimal — no hidden non-linearity. Consistent with literature. Practice.

## What Was Learned
- Adversarial SR pipeline tested on SN standardization — methodology confirmed
- H0 choice confounds SR search for Tripp corrections (z-dependent residual dominates)
- Cosmology-removed residual approach works: isolate x1/c structure before SR
- First known test of Tripp linearity via symbolic regression
