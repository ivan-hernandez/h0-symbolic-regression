# Project: Symbolic Regression Discovery of Modified Gravity from Data

## Objective
Use symbolic regression (Φ-SO/PySR) with minimal physical priors to derive modified gravitational field equations directly from observations, eliminating the need for dark matter and dark energy as free parameters.

---

## Phase 1: The Hubble Tension (3-4 months)

**Problem:** Early-universe (Planck CMB, H0 ≈ 67.4) and late-universe (SH0ES SNeIa, H0 ≈ 73.0) measurements of the expansion rate disagree at 5σ. ΛCDM cannot reconcile them without new physics.

**Data inputs:**
- Pantheon+ SNeIa compilation (z < 2.3)
- Cosmic chronometer H(z) measurements
- BAO data (DESI, SDSS, 6dF)
- Planck CMB likelihood (as high-z anchor)
- SH0ES Cepheid distance ladder

**Method:**
- Start with the FLRW metric (assume homogeneity/isotropy)
- Parameterize the Friedmann equations as: H² = (8πG/3)ρ + f(ρ, a, H, ...) where f is a learned symbolic correction
- Train symbolic regressor on joint H(z) + BAO + SNe dataset
- Constrain: must reduce to GR at high redshift / high density (solar system tests preserved)
- Output: minimal symbolic correction term that resolves the tension

**Success criteria:**
- Finds a correction that fits both early and late data within 1σ
- Correction is simpler (lower MDL) than adding free DE parameters
- Correction involves a physically meaningful scale (H0, a0, etc.)

**Risk:** The data may be insufficient to uniquely determine the form (same problem Desmond et al. found with SPARC).

---

## Phase 2: Galactic Rotation Curves + RAR (3-4 months)

**Problem:** The radial acceleration relation (RAR) shows a tight correlation between baryonic and total acceleration that MOND predicts but ΛCDM cannot explain.

**Data inputs:**
- SPARC galaxy rotation curves (175 galaxies)
- LSB + HSB galaxies spanning wide dynamic range
- Recent Mistele et al. (2025) million-light-year lensing data

**Method:**
- Use Class Symbolic Regression (2024, Φ-SO extension) to find one functional form that fits *all* galaxies simultaneously, each with galaxy-specific parameters (M/L, distance)
- Operators: +, -, ×, ÷, √, power, exp
- Dimensional analysis constraints baked in
- Blind test: generate mock MOND data and verify the SR recovers MOND-like form

**Success criteria:**
- Discovers a 1-parameter interpolating function that fits SPARC as well as or better than MOND's standard μ-function
- The discovered function generalizes to holdout galaxies
- The characteristic scale a0 emerges naturally from the fit, not imposed

**Risk:** Likely produces something MOND-like, which is useful validation but not a breakthrough.

---

## Phase 3: Joint Cosmology + Galaxies (6+ months)

**Problem:** A viable modified gravity theory must simultaneously explain cosmic expansion, large-scale structure, and galactic dynamics.

**Data inputs:**
- All of Phase 1 + Phase 2 datasets
- Matter power spectrum (SDSS, DESI)
- Strong gravitational lensing (time delays)
- Bullet Cluster / merging cluster dynamics
- GW170817 neutron star merger (speed of gravity constraint)

**Method:**
- Propose the candidate forms from Phase 1 and Phase 2
- Test each against all remaining datasets
- Iterate: if a candidate fails, use the discrepancy to guide SR toward a better form
- Final candidate must: (i) resolve Hubble tension, (ii) fit rotation curves, (iii) pass solar system tests, (iv) satisfy GW speed constraint, (v) produce correct large-scale structure

**Success criteria:**
- A single, closed-form modification to the Einstein-Hilbert action (or equivalent) that fits all data without DM or DE
- Predicts a new observable that can distinguish it from ΛCDM

---

## Near-term Next Steps

1. **Environment setup** — Get Φ-SO or ESR running on a machine with a good GPU
2. **Phase 1 pilot** — Start with the Hubble tension dataset alone (simplest, cleanest data)
3. **Reproduce Desmond et al. 2023** — Verify ESR on SPARC data to validate the toolchain
4. **Extend** — Add new datasets incrementally
