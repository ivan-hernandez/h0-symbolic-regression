# P7: Radial Acceleration Relation — KILLED (Practice)

**Date:** 2026-06-26  
**Result:** SR confirms RAR functional form is degenerate. Power law, McGaugh, and 1/x forms fit identically.

## Why it was killed
- Three functional forms (power law, McGaugh, SR 1/x) all fit with RMS=0.195–0.199 dex
- Individual-point scatter (0.2 dex) is larger than the functional differences between forms
- Binned RMS = 0.11 — consistent with McGaugh+2016
- SR at C=5 (1/x form: `log(g_obs) = -72.89/log(g_bar) - 17.08`) captures RAR curvature but improves over power law by only 2%
- **Crap-or-worthwhile:** "consistent with literature" — Li+2018 already showed ΛCDM reproduces RAR
- Fourth consecutive practice kill (P4 ringdown, P5 Tripp, P6 BTFR, P7 RAR)

## Results
- **Power law:** log(g_obs) = 0.69·log(g_bar) - 2.19, RMS = 0.199
- **McGaugh+2016 (g†=1.2e-10):** RMS = 0.199 — identical
- **SR best (C=5):** log(g_obs) = -72.89/log(g_bar) - 17.08, RMS = 0.195
- All higher-complexity models improve RMS only to ~0.194 (0.005 dex total)
- Binned McGaugh RMS = 0.11

## Data
- SPARC catalog: 3389 radial points from 175 galaxies
- g_bar from Vbar²/R (Υ_disk=0.5, Υ_bul=0.7)
- g_obs from Vobs²/R
- Range: log g_bar ∈ [-12.68, -8.13]

## Scripts
- `rar_data.py`: Data loader (SPARC mass models → gbar/gobs)
- `rar_sr_remote.py`: SR runner (300 iterations, 15 populations)
- `parse_sparc.py`: SPARC mass model parser (from rotation_curves/)
