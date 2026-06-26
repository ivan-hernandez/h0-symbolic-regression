# P4 — Ringdown Null Test: KILLED

**Date:** 2026-06-26
**Reason:** Crap-or-worthwhile test → practice (not novel)

## Results
- **dω/ω:** Combined 0.021 ± 0.020 (1.1σ) — consistent with GR
- **dτ/τ:** Combined 0.142 ± 0.065 (2.2σ) — known mild tension (already in LVK paper)
- **SR null test:** No parameter-dependent pattern found across 11 models (null, constant, linear, quadratic, etc.)
- **Kerr verification:** pSEOBNR GR predictions differ from exact Kerr QNM by 5–38% (by design — NR fit formula), but dω/ω is uncorrelated with this error (r = −0.09)

## Verdict
No novel finding. The result doesn't change what anyone would assume or do. LVK already published dτ/τ bias; our SR search found no hidden pattern. Code kept for reference.

## What Was Learned
- SR methodology works on GW ringdown posterior summaries
- `qnm` package matches pSEOBNR GR to exact Kerr (useful for future waveform systematics checks)
- Ringdown data release structure (pSEOBNR + IMR + pyRing) understood for future reference
