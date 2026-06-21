# Phase 2 Extensions — RAR Symbolic Regression

## Overview
Three extensions to validate and deepen the SPARC RAR results.

---

## Extension 1: M/L Sensitivity Test

**Motivation:** The SPARC default mass-to-light ratios (M/L = 0.5 disk, 0.7 bulge) are the single largest systematic uncertainty in the RAR. Different stellar population models give M/L ranging from 0.3–1.0 for disks and 0.5–1.2 for bulges.

**Method:**
- Recompute gbar for (Upsilon_disk, Upsilon_bul) grid:
  - Upsilon_disk ∈ {0.3, 0.5, 0.7, 1.0}
  - Upsilon_bul ∈ {0.3, 0.5, 0.7, 1.0}
- For each combination: refit MOND Simple, SR CPX3, SR CPX5
- Track: a₀ shift, χ² change, equation parameter variation

**Success:**
- If a₀ and SR parameters are stable (Δa₀ < 50%) → RAR is robust
- If they vary wildly → need external M/L constraints (e.g., SED fitting)

---

## Extension 2: Blind MOND Recovery Test

**Motivation:** Does PySR actually recover a known MOND interpolating function when presented with mock data generated from one? This validates the toolchain.

**Method:**
- Generate mock data: gbar_grid from 10^-13 to 10^-8, compute gobs = MOND_Simple(gbar, a₀=1.2e-10)
- Add realistic log-normal scatter (0.1 dex, matching SPARC scatter)
- Run PySR with identical settings as real run
- Check: does the best equation approximate the MOND Simple form?

**Variants:**
- Test with McGaugh RAR formula instead of MOND Simple
- Test with varying scatter levels
- Test with only SPARC-like gbar sampling (not uniform grid)

**Success criteria:**
- PySR recovers form that asymptotes to g_obs ∝ g_bar at high g and g_obs ∝ √(g_bar) at low g
- Fitted a₀ is within 20% of input a₀

---

## Extension 3: Adversarial Validation

**Motivation:** Following the Hubble project playbook, test the RAR conclusions against systematic objections using a debate framework.

**Testable objections:**
1. **"M/L uncertainty drives result"** — addressed by Extension 1
2. **"SR is fishing; any smooth function fits"** — addressed by Extension 2 blind test
3. **"Galaxy sample is biased"** — subset analysis: split by Hubble type, surface brightness, gas fraction
4. **"SPARC uses old M/L values"** — compare with newer stellar population models (e.g., Schombert+2024)
5. **"Data scatter is non-Gaussian; χ² comparison invalid"** — bootstrap uncertainty estimates
6. **"RAR is not a universal relation"** — per-galaxy fits, check for galaxy-to-galaxy parameter variation
7. **"MOND predicts EFE; scatter in RAR is EFE, not model failure"** — test if residuals correlate with large-scale environment

**Procedure:**
- Implement 2-round adversarial debate (following Hubble template)
- Round 1: system raise objections, defender responds
- Round 2: deeper counter-attacks
- Log in debate_log_RAR.md
