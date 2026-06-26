# P5 — SNe Ia Tripp Standardization via Symbolic Regression

**Goal:** Discover whether the standard linear Tripp correction 
(μ = m_B - M_B + α·x1 - β·c) is truly optimal, or whether 
symbolic regression can find a non-linear form that reduces 
scatter and/or changes H0.

**Status:** Active (Phase 1 — data pipeline)

## Crap-or-Worthwhile Test
- **Novel if:** SR finds non-linear f(x1, c) that reduces Hubble 
  diagram scatter or shifts H0 by >0.5 km/s
- **Practice if:** Linear Tripp is confirmed optimal (RMS unchanged)
- **Verdict pending** — decision after SR runs complete

## Data
- **Pantheon+** (1590 cosmologically usable SNe): m_b, x1, c, zHD
- **DES-SN5YR** (1820 SNe): independent cross-validation sample

## Method
1. Reference cosmology: flat ΛCDM (H0=70, Ωm=0.3) → distance moduli
2. Define y = m_b - μ(z) — contains M_B offset + Tripp correction
3. Linear baseline: y = M_B - α·x1 + β·c
4. SR: y = f(x1, c) — free functional form
5. Compare RMS, AIC, and resulting H0 between linear and SR forms
6. Cross-validate on DES-SN5YR
7. Adversarial debate

## Pipeline
- `scripts/download_data.py`: Download Pantheon+ and DES-SN5YR
- `scripts/tripp_sr.py`: Main SR discovery script (remote)
- `scripts/validate.py`: Bootstrap + cross-validation
- `scripts/plot_results.py`: Comparison plots
