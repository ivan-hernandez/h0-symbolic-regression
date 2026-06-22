# Symbolic Regression in Astrophysics — Research Proposal

**Status: Original proposal (static — written before execution)**

---

## Overview

We propose a three-phase research program applying symbolic regression (PySR)
to two foundational open problems in astrophysics:

1. **The Hubble tension** — a >5σ discrepancy between early- and late-Universe measurements of the expansion rate \(H_0\).
2. **The radial acceleration relation** — a tight empirical correlation in galaxy rotation curves that both dark matter and modified gravity must explain.
3. **Synthesis** — can the discovered empirical forms jointly constrain galaxy formation physics and discriminate between theories?

The unifying theme: **let the data discover the functional form, with minimal theoretical priors.** Symbolic regression searches the space of analytic expressions for the simplest equation that minimizes χ². We then stress-test every discovered form with adversarial validation, bootstrap resampling, M/L sensitivity sweeps, holdout testing, and blind recovery experiments.

---

## Phase 1 — The Hubble Constant

### Objective
Use SR on cosmic chronometers (CC), BAO (SDSS + DESI), and supernova (Pantheon+) data to discover the expansion history \(H(z)\) with minimal priors, extract \(H_0\), and test whether the Hubble tension resides in the expansion history shape or the Cepheid calibration of the supernova absolute magnitude \(M\).

### Proposed Methodology
1. Run PySR on CC + BAO data to discover candidate \(H(z)\) forms with a weak \(z=0\) prior \(H_0 = 67.4 \pm 20\) km/s/Mpc to suppress pathological solutions.
2. Evaluate SN likelihood post-hoc on each candidate, treating the absolute magnitude \(M\) as a free parameter (analytically marginalized).
3. Construct profile likelihood for \(H_0\) and compare with Planck (early) and SH0ES (late) measurements.
4. Stress-test with systematic objection suite: CC-only, BAO-only, DESI vs SDSS, full covariance vs diagonal, Fix-M test, Union3 cross-check, \(\Lambda\)CDM comparison.
5. Adversarial validation: multi-round debate challenging every aspect of the analysis.

### Expected Outcomes
- A data-driven \(H(z)\) functional form competitive with \(\Lambda\)CDM.
- A model-independent \(H_0\) measurement that can adjudicate the Hubble tension.
- Identification of whether the tension lies in \(H(z)\) shape or the Cepheid \(M\) calibration.

---

## Phase 2 — The Radial Acceleration Relation

### Objective
Use SR on SPARC rotation curves (175 galaxies, ~3,400 points) to discover the functional form of the RAR \(g_{\rm obs} = F(g_{\rm bar})\) and compare with MOND interpolating functions.

### Proposed Methodology
1. Run PySR in log-log space on SPARC data with default M/L (\( \Upsilon_{\rm disk} = 0.5, \Upsilon_{\rm bul} = 0.7\)).
2. Multi-seed validation (≥3 independent SR runs) to verify equation convergence.
3. Compare discovered forms with MOND IFs (McGaugh RAR IF, Simple, Standard) on AIC/χ².
4. Validation suite: bootstrap (galaxy-wise resampling), holdout (10-fold), M/L grid sweep (16 combinations), blind MOND recovery test, per-galaxy fitting.
5. Extend to weak-lensing regime using Mistele+2024 data (joint SPARC+lensing SR, 6.5 dex dynamic range).
6. Test MOND-specific predictions: \(\sqrt{g_{\rm bar}}\) asymptote, External Field Effect (via SIMBAD 3D coordinates).
7. Search for FIRE-2-predicted non-monotonic "hook" features in individual RAR tracks.
8. Compare with literature: Desmond+2023 ESR, McGaugh+2016, simulation suites (EAGLE, IllustrisTNG, FIRE-2).
9. Adversarial validation.

### Expected Outcomes
- The best empirical description of the RAR over the kinematic dynamic range.
- A quantitative test of whether the MOND \(\sqrt{g_{\rm bar}}\) asymptote is required by data.
- Robustness assessment of the RAR shape vs. M/L assumptions.

---

## Phase 3 — Synthesis: Dark Matter vs. Modified Gravity

### Objective
Use the SR-discovered forms from Phases 1 and 2 as empirical discriminants between dark matter models and MOND, and as constraints on galaxy formation physics.

### Proposed Methodology
1. **Simulation classification:** Fit the CPX5 RAR form to mock rotation curves from EAGLE, IllustrisTNG, FIRE-2, and \(\Lambda\)CDM baryonification models. Test whether different DM models produce systematically different CPX5 parameter values — turning the RAR form into a DM model classifier.
2. **Hook quantification:** Apply the FIRE-2 hook search algorithm with a proper permutation null test (mock Gaussian rotation curves with matched covariance) to determine whether 68% hook prevalence exceeds chance.
3. **Joint cosmological constraints:** Combine the Phase 1 \(H(z)\) form with the Phase 2 RAR form in a unified Bayesian analysis. Can the two jointly constrain \(\Omega_m\) and \(\sigma_8\) without assuming a DM halo profile?
4. **Baryonification tolerance:** If CPX5 is universal, any viable \(\Lambda\)CDM baryonification model must recover it within some tolerance. Map the feedback parameter space that produces CPX5-compatible RARs.
5. **MOND consistency test:** Does the Phase 1 \(H(z)\) form, when extrapolated, satisfy MOND's cosmological predictions? Does the Phase 2 RAR form, when combined with lensing, converge to MOND in the deep limit?
6. **Detection significance:** With Roman Space Telescope forecasts (2027 launch, H₀ to 1.3%), project when the Phase 1 and Phase 2 forms can be distinguished from \(\Lambda\)CDM and MOND at >5σ.

### Expected Outcomes
- A quantitative statement about whether current simulation suites naturally produce the observed RAR.
- Statistical significance (or lack thereof) of FIRE-2 hook features.
- Joint constraints on cosmology + galaxy formation from data-driven functional forms.
- A roadmap for future surveys (Euclid, Rubin, Roman) to definitively test the discovered forms.

---

## Timeline

| Phase | Status | Key Deliverable |
|-------|--------|-----------------|
| Phase 1 — Hubble | Proposed | Paper + \(H_0\) measurement |
| Phase 2 — RAR | Proposed | Paper + RAR functional form |
| Phase 3 — Synthesis | Proposed | Paper + DM/MOND discrimination |

---

## Reproducibility

All analysis code, processed data, and adversarial debate logs will be publicly archived on GitHub and Zenodo. Heavy computation runs on a dedicated 12-core/15 GB RAM machine.

---

## Methodology Principles (All Phases)

1. **Data-driven:** Let symbolic regression discover forms without theoretical priors.
2. **Adversarially validated:** Every claim tested by an independent challenger.
3. **M/L-robust:** All results checked against plausible stellar mass-to-light ratio ranges.
4. **Multi-seed:** SR runs replicated with independent random seeds.
5. **Falsifiable:** Every discovered form makes testable predictions for future data.
6. **Reproducible:** Full code and data publicly archived.
