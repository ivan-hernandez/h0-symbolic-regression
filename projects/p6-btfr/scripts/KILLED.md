# P6: Baryonic Tully-Fisher Relation — KILLED (Practice)

**Date:** 2026-06-26  
**Result:** SR confirms BTFR is a pure power law. No hidden structure found.

## Why it was killed
- The functional form of BTFR was never in dispute — both ΛCDM and MOND predict power laws
- The slope (3.42 ± 0.12) is consistent with literature (3.0–4.0 range)
- SR at C=5 recovers the standard power law exactly; C=6-11 improve RMS by only 0.004 dex
- **Crap-or-worthwhile:** "consistent with literature" → practice
- Third consecutive practice kill (P4 ringdown, P5 Tripp, P6 BTFR)

## Results
- **Best model (C=5):** log Mbary = 3.41·log Vflat + 2.92 (RMS = 0.248 dex)
- **Bootstrap (2000 resamples):** a = 3.42 ± 0.12 (68% CL)
- MOND (a=4) excluded at >99.9% CL
- ΛCDM (a=3-3.5) preferred (73% of bootstrap within range)
- No significant non-linear structure at any complexity

## Data
- SPARC catalog: 135 galaxies with Vflat > 0
- Baseline Υ = 0.5 M☉/L☉, Q ≤ 2 quality cut
- Slope varies with Υ (3.19–3.69), inclination (3.41–3.58), and quality (3.19–3.56)

## Scripts
- `btfr_fit.py`: Data loader + baseline power-law fit
- `btfr_sr_remote.py`: SR runner (2 runs, 15 populations, 300 iterations)
- `btfr_validate.py`: Bootstrap + sensitivity tests
- `btfr_data.pkl`: Cached data for analysis
