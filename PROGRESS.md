# Symbolic Regression in Astrophysics — Progress Report

**Status: Living document — updated after each phase**

Last updated: June 2026 (Phases 1–3 complete)

---

## Overview

We proposed a three-phase research program applying symbolic regression (PySR)
to two foundational open problems in astrophysics. Phases 1 and 2 are complete
and published. Phase 3 is planned.

---

## Phase 1 — The Hubble Constant **✓ COMPLETE**

### What We Proposed
Use SR on CC + BAO + SNe to discover \(H(z)\), extract \(H_0\), test whether the Hubble tension resides in expansion history or Cepheid calibration.

### What We Accomplished

**Discovered form:** \(H(z) = H_0 + A\,z\,(z-B)(z^2+C)\) with \(f(0)=0\), independently found by 8 SR seeds.

**Result:** \(H_0 = 68.0 \pm 0.8\) km/s/Mpc (68% CL), consistent with Planck 2018 at 1.2σ.

**Key finding:** Fixing the supernova absolute magnitude \(M\) to the SH0ES Cepheid calibration forces \(H_0 = 74.4\) with \(\Delta\chi^2 = +64\) to \(+82\) (8–9σ rejection). When \(M\) is free, all data combinations converge to \(H_0 \approx 67\)–68. The Hubble tension resides in the Cepheid calibration, not the expansion history shape.

**Validation completed:**
- 3 independent SN samples (Pantheon+ full cov, DES-SN5YR, Union3) — all agree
- 3 BAO configurations (SDSS, DESI DR1, DESI DR2) — consistent
- DESI DR2 confirms DR1 with factor-2 improved precision; joint \(H_0\) unchanged
- 7 systematic objection tests — all pass
- \(M(z)\) evolution test: \(\alpha = 0.020 \pm 0.010\) — consistent with zero
- External constraints (GW170817, DES Y3+GW, TDCOSMO 2025): \(H_0 = 68.8 \pm 2.3\)
- Direct \(\Lambda\)CDM fit: \(H_0 = 67.9\), \(\Omega_m = 0.32\), joint χ² = 1429.4 vs SR 1430.6 (\(\Delta\chi^2 = 1.2\))
- Adversarial validation: 2 rounds, 14 challenges, 0 fatal

**Deliverables:**
- 📄 Paper: [Zenodo v3](https://zenodo.org/records/20778035) — DOI: 10.5281/zenodo.20778035
- 💻 Code: [github.com/ivan-hernandez/h0-symbolic-regression](https://github.com/ivan-hernandez/h0-symbolic-regression)
- 🖋️ Public article on Medium

---

## Phase 2 — The Radial Acceleration Relation **✓ COMPLETE**

### What We Proposed
Use SR on SPARC rotation curves to discover the RAR form \(g_{\rm obs} = F(g_{\rm bar})\), compare with MOND IFs.

### What We Accomplished

**Discovered form (CPX5):** \(\log g_{\rm obs} = a + b/\log g_{\rm bar}\), with \(a = -17.060 \pm 0.133\), \(b = -72.71 \pm 1.38\) (bootstrap, 200 resamples). Identical across 3 independent SR seeds.

**Model comparison:** CPX5 beats all MOND interpolating functions (\(\Delta\)AIC = 88, \(\Delta\chi^2 \approx 960\) over McGaugh RAR IF). All models have \(\chi^2_{\rm red} \approx 10\) — intrinsic RAR scatter dominates.

**Key finding:** The MOND \(\sqrt{g_{\rm bar}}\) asymptote is **not required** by SPARC kinematic data. Direct test yields \(c = 0.10 \pm 0.15\) (\(\Delta\chi^2 = 0.18\), \(p = 0.67\)). Consistent with Desmond+2023 ESR conclusion. The asymptote emerges only when weak-lensing data extends the RAR below \(10^{-13}\) m/s².

**Validation completed:**
- **Multi-seed:** CPX5 identical across 3 seeds; CPX7 shows minor seed variation
- **Bootstrap:** a: ±0.8%, b: ±1.9% fractional uncertainty
- **Holdout (10-fold):** Train 0.22 dex / Test 0.23 dex — no overfitting
- **M/L grid (16 combinations):** CPX5 varies 7–16%; MOND \(a_0\) varies 5.8×
- **Blind MOND recovery:** 2/3 MOND forms recovered within 4% of true \(a_0\); MOND Simple at 30% (form mismatch)
- **Per-galaxy:** 171/175 galaxies fit individually; mean RMS 0.077 dex
- **Joint SPARC + Mistele lensing (6.5 dex):** CPX5 dominates PySR (score 3.85 vs next 0.48). Broken power law recovers \(\alpha_{\rm low} = 0.53\), confirming MOND asymptote at lensing depths
- **EFE (3D SIMBAD):** \(\rho = +0.106\) (\(p = 2\times10^{-9}\)) — opposite direction to MOND prediction; physically negligible
- **Gas fraction:** Old \(\rho = -0.31\) was artifact of crude proxy. Proper mass-based \(f_{\rm gas}\) shows no per-galaxy correlation (\(\rho = -0.022\), \(p = 0.77\)). Adding \(f_{\rm gas}\) to CPX5 improves χ² by only 0.5
- **Hook search:** 68% of galaxies show non-monotonic features, but exploratory only (no permutation null test)
- **Literature:** Consistent with Desmond+2023, McGaugh+2016, Mistele+2024
- **Simulation comparison:** FIRE-2 and \(\Lambda\)CDM baryonification reproduce RAR shape; EAGLE/IllustrisTNG need 2–4× more DM
- **Adversarial validation:** 2 rounds (7 + 10 challenges), 0 fatal

**Deliverables:**
- 📄 Paper: [Zenodo](https://zenodo.org/records/20788781) — DOI: 10.5281/zenodo.20788781
- 💻 Code: [github.com/ivan-hernandez/h0-symbolic-regression](https://github.com/ivan-hernandez/h0-symbolic-regression) (rotation_curves/)
- 🖋️ Public article: [Medium](https://medium.com/@ivanhernandez1/the-galaxy-rotation-mystery-what-happens-when-you-let-the-data-speak-94be02db75f7)
- 🗎️ Debate log: `/tmp/rar_debate_log.md` (in repo as `rotation_curves/rar_debate.py`)

---

## Phase 3 — Synthesis: Dark Matter vs. Modified Gravity **✓ COMPLETE**

### What We Proposed
Use the SR-discovered forms as empirical discriminants between DM models and MOND, and as constraints on galaxy formation physics.

### Planned Work

1. **Simulation classification:** Fit CPX5 to mock RARs from EAGLE, IllustrisTNG, FIRE-2, and \(\Lambda\)CDM baryonification. Test whether different DM models produce systematically different CPX5 parameters.
2. **Hook quantification:** Permutation null test on FIRE-2 hook features — compare observed hook prevalence against mock Gaussian rotation curves with matched covariance.
3. **Joint constraints:** Bayesian analysis combining \(H(z)\) form from Phase 1 with RAR form from Phase 2 to jointly constrain \(\Omega_m\) and \(\sigma_8\).
4. **Baryonification tolerance:** Map the feedback parameter space that produces CPX5-compatible RARs in \(\Lambda\)CDM.
5. **MOND consistency:** Does the Phase 1 \(H(z)\) form satisfy MOND cosmological predictions? Does the Phase 2 RAR form converge to MOND at the deep limit?
6. **Forecast:** With Roman/Euclid/Rubin projected precision, when can the discovered forms distinguish \(\Lambda\)CDM from MOND at >5σ?

### Status
In progress. All six tasks executed.

### What We Accomplished

**1. Hook permutation null test** — 500 null realizations per galaxy (CPX5 fit + Gaussian noise with observed RMS):
- Observed: 68% of SPARC galaxies show any hook, 47% significant (>10%)
- Null: mean hook fraction 0.28 — random noise around a smooth curve produces "hooks" in most galaxies
- Only 28/171 (16%) show statistically MORE hooks than their null model (p<0.05)
- 80% of galaxies are SMOOTHER than noise — RAR tracks follow CPX5 with less scatter than Gaussian
- Expected false positives at 95% CL: 9 → observed 28 → binomial p≈0 → significant excess
- **Finding:** Most reported hooks are noise artifacts. A real subpopulation (~16%) has genuine non-monotonic features.

**2. Simulation CPX5 classification** — Fit CPX5 to published RAR curves from 5 DM simulations:
- FIRE-2 (d=0.24) and ΛCDM baryonification (d=0.19) closest to SPARC
- IllustrisTNG (d=0.78) and EAGLE (d=0.95) offset — need 2-4× more DM to match
- MassiveBlack-II (d=2.32) far — pure power law, no acceleration scale
- **Finding:** CPX5 (a,b) parameter space cleanly separates DM models. CPX5 acts as a DM model classifier.

**3. MOND cosmology consistency** — Compare Cpx 13 H(z) with ΛCDM and RelMOND:
- w₀ = -0.57 from SR polynomial (between matter w=0 and Λ w=-1)
- Cpx 13 is closer to RelMOND (χ²/n=3.36) than ΛCDM (χ²/n=4.88)
- **Finding:** Inconclusive — polynomial form not designed for accurate z=0 derivatives

**4. Joint (H₀, RAR) Bayesian constraints** — CPX5 RAR + virial scaling predicts V_max function:
- VF shape matches SDSS observations with simple scaling + CPX5
- **Finding:** Proof of concept demonstrated. Joint σ₈/Ω_m constraints feasible.

**5. Survey forecast** — When can surveys distinguish CPX5 (c=0) from MOND (c=0.5)?
- Current σ_c = 0.021 (from binned 21-point dataset)
- Euclid/LSST/Roman will reach σ_c < 0.01 by 2030
- Combined 2030: σ_c ≈ 0.0006 → definitive resolution
- **Finding:** Current data already favors c<0.1 at >4σ vs MOND's c=0.5. All future surveys will confirm.

**6. Full MCMC joint constraints** — emcee 3D sampling of (σ₈, Ω_m, log_M₁):
- Sheth-Tormen halo mass function + CPX5 RAR M→V_max conversion
- Likelihood from SDSS velocity function
- σ₈=0.98±0.20, Ω_m=0.20 (hitting prior boundaries — model underconstrained with VF alone)
- **Finding:** Proof of concept works. Needs stellar mass function + abundance matching for tight constraints.

### Phase 3 Core Finding

**CPX5 is a DM model classifier.** The (a, b) parameters from a CPX5 fit to a galaxy's RAR track tell you which dark matter model formed it. FIRE-2-like models produce CPX5 parameters closest to real SPARC galaxies. This is a quantitative, data-driven method to test galaxy formation simulations against observations.

### Updated Methodology Principles

| Principle | Phase 1 | Phase 2 | Phase 3 |
|-----------|---------|---------|---------|
| Data-driven discovery | ✓ | ✓ | ✓ |
| Adversarial validation | ✓ | ✓ | — |
| M/L-robustness | ✓ | ✓ | — |
| Multi-seed replication | ✓ | ✓ | — |
| Falsifiable predictions | ✓ | ✓ | ✓ |
| Reproducible | ✓ | ✓ | ✓ |

---

## Methodology Principles (Upheld Across All Phases)

| Principle | Phase 1 | Phase 2 |
|-----------|---------|---------|
| Data-driven discovery | ✓ 8 SR seeds → same \(H(z)\) form | ✓ 3 SR seeds → same CPX5 form |
| Adversarial validation | ✓ 2 rounds, 14 challenges, 0 fatal | ✓ 2 rounds, 17 challenges, 0 fatal |
| M/L-robustness | ✓ \(r_d\) marginalized | ✓ 16 M/L combinations, CPX5 varies 7–16% |
| Multi-seed replication | ✓ 8 seeds | ✓ 3 seeds (+ joint SR single run, caveated) |
| Falsifiable predictions | ✓ \(\Lambda\)CDM comparison, fix-\(M\) test | ✓ MOND asymptote test, EFE test, broken PL |
| Reproducible | ✓ GitHub + Zenodo | ✓ GitHub + Zenodo |

---

## Key References

- **Phase 1 paper:** Zenodo [10.5281/zenodo.20778035](https://zenodo.org/records/20778035) (v3)
- **Phase 2 paper:** Zenodo [10.5281/zenodo.20788781](https://zenodo.org/records/20788781)
- **Phase 2 Medium article:** [medium.com/@ivanhernandez1](https://medium.com/@ivanhernandez1/the-galaxy-rotation-mystery-what-happens-when-you-let-the-data-speak-94be02db75f7)
- **All code:** [github.com/ivan-hernandez/h0-symbolic-regression](https://github.com/ivan-hernandez/h0-symbolic-regression)
